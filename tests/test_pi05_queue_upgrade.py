"""Exercise the empty-queue handoff with mocked process and signal APIs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/deploy"


@pytest.fixture
def module(monkeypatch):
  monkeypatch.syspath_prepend(str(SCRIPTS))
  spec = importlib.util.spec_from_file_location(
    "queue_upgrade_under_test", SCRIPTS / "upgrade_pi05_new_tasks_queue.py"
  )
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


@pytest.fixture
def empty_queue(module, tmp_path, monkeypatch):
  root = tmp_path / "old_job"
  root.mkdir()
  bundle = tmp_path / "old_snapshot"
  monkeypatch.setattr(module, "OLD_ROOT", root)
  monkeypatch.setattr(module, "OLD_BUNDLE", bundle)
  groups = [
    {
      "group_index": index,
      "task": task,
      "step": step,
      "checkpoint": str(checkpoint),
      "status": "queued",
      "output_dir": str(root / task / f"step{step}_N20_video20"),
    }
    for index, (task, step, checkpoint) in enumerate(module.queue.worker.NEW_GROUPS)
  ]
  status = {
    "pid": module.OLD_PID,
    "status": "running",
    "stage": "waiting_for_capacity",
    "active_new_groups": 0,
    "completed_trials": 0,
    "settings": {"total_checkpoint_worker_cap": 4},
    "groups": groups,
  }
  command = [
    str(module.queue.CONTROLLER_PYTHON),
    str(bundle / "kaitactilesim/scripts/deploy/run_pi05_new_tasks_queue.py"),
    "--job-root",
    str(root),
    "--simulation-root",
    str(bundle / "kaitactilesim"),
    "--openpi-root",
    str(bundle / "openpi"),
  ]
  process = {"state": "S", "ppid": 1, "starttime": 12345, "exit_status": 0}
  monkeypatch.setattr(module.queue, "read_json", lambda _: status)
  monkeypatch.setattr(module.queue, "process_info", lambda _: process)
  monkeypatch.setattr(module, "command_line", lambda _: command)
  monkeypatch.setattr(module, "children", lambda _: [])
  return SimpleNamespace(
    root=root, groups=groups, status=status, command=command, process=process
  )


def test_empty_queue_requires_exact_owned_command(module, empty_queue):
  result = module.validate_empty_queue()
  assert result["pid"] == module.OLD_PID
  assert result["starttime"] == 12345
  assert result["command"] == empty_queue.command
  empty_queue.command[-1] = "/unrelated/model/project"
  with pytest.raises(RuntimeError, match="process identity changed"):
    module.validate_empty_queue()


@pytest.mark.parametrize(
  "change",
  ["worker", "running_group", "checkpoint", "output", "children", "zombie", "parent"],
)
def test_empty_queue_rejects_live_work_or_wrong_identity(
  module, empty_queue, monkeypatch, change
):
  if change == "worker":
    empty_queue.groups[0]["worker_pid"] = 456
  elif change == "running_group":
    empty_queue.groups[0]["status"] = "running"
  elif change == "checkpoint":
    empty_queue.groups[0]["checkpoint"] = "/other/checkpoint/24999"
  elif change == "output":
    Path(empty_queue.groups[0]["output_dir"]).mkdir(parents=True)
  elif change == "children":
    monkeypatch.setattr(module, "children", lambda _: [456])
  elif change == "zombie":
    empty_queue.process["state"] = "Z"
  else:
    empty_queue.process["ppid"] = 456
  with pytest.raises(RuntimeError, match="no longer empty"):
    module.validate_empty_queue()


@pytest.fixture
def signal_probe(module, monkeypatch):
  events = []
  monkeypatch.setattr(
    module,
    "os",
    SimpleNamespace(
      pidfd_open=lambda pid: events.append(("open", pid)) or 97,
      close=lambda descriptor: events.append(("close", descriptor)),
    ),
  )
  monkeypatch.setattr(
    module,
    "signal",
    SimpleNamespace(
      SIGSTOP=module.signal.SIGSTOP,
      SIGTERM=module.signal.SIGTERM,
      SIGCONT=module.signal.SIGCONT,
      pidfd_send_signal=lambda descriptor, signum: events.append(
        ("signal", descriptor, signum)
      ),
    ),
  )
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  return events


def test_stopped_second_check_failure_restores_controller(
  module, monkeypatch, signal_probe
):
  checks = iter([{"starttime": 12345}, RuntimeError("new worker appeared")])

  def validate():
    result = next(checks)
    if isinstance(result, Exception):
      raise result
    return result

  monkeypatch.setattr(module, "validate_empty_queue", validate)
  monkeypatch.setattr(
    module.queue, "process_info", lambda _: {"state": "T", "starttime": 12345}
  )
  with pytest.raises(RuntimeError, match="new worker appeared"):
    module.stop_empty_queue()
  assert signal_probe == [
    ("open", module.OLD_PID),
    ("signal", 97, module.signal.SIGSTOP),
    ("signal", 97, module.signal.SIGCONT),
    ("close", 97),
  ]


def test_pid_reuse_during_stop_is_rejected_and_original_fd_is_closed(
  module, monkeypatch, signal_probe
):
  monkeypatch.setattr(module, "validate_empty_queue", lambda: {"starttime": 12345})
  monkeypatch.setattr(
    module.queue, "process_info", lambda _: {"state": "T", "starttime": 99999}
  )
  with pytest.raises(RuntimeError, match="PID was reused"):
    module.stop_empty_queue()
  assert ("signal", 97, module.signal.SIGTERM) not in signal_probe
  assert signal_probe[-2:] == [
    ("signal", 97, module.signal.SIGCONT),
    ("close", 97),
  ]


def test_gone_controller_during_resume_does_not_hide_failure_or_leak_pidfd(
  module, monkeypatch, signal_probe
):
  monkeypatch.setattr(module, "validate_empty_queue", lambda: {"starttime": 12345})
  monkeypatch.setattr(module.queue, "process_info", lambda _: None)
  send = module.signal.pidfd_send_signal

  def disappeared(descriptor, signum):
    send(descriptor, signum)
    if signum == module.signal.SIGCONT:
      raise ProcessLookupError("owned controller already exited")

  monkeypatch.setattr(module.signal, "pidfd_send_signal", disappeared)
  with pytest.raises(RuntimeError, match="owned queue disappeared"):
    module.stop_empty_queue()
  assert signal_probe[-1] == ("close", 97)


def test_normal_handoff_signals_only_owned_pidfd_and_confirms_interruption(
  module, monkeypatch, signal_probe
):
  monkeypatch.setattr(module, "validate_empty_queue", lambda: {"starttime": 12345})
  processes = iter([{"state": "T", "starttime": 12345}, None])
  monkeypatch.setattr(module.queue, "process_info", lambda _: next(processes))
  monkeypatch.setattr(
    module.queue,
    "read_json",
    lambda _: {"status": "interrupted", "active_new_groups": 0},
  )
  result = module.stop_empty_queue()
  assert signal_probe == [
    ("open", module.OLD_PID),
    ("signal", 97, module.signal.SIGSTOP),
    ("signal", 97, module.signal.SIGTERM),
    ("signal", 97, module.signal.SIGCONT),
    ("close", 97),
  ]
  assert result["previous_controller_stopped"] is True
  assert result["previous_queue_was_empty"] is True
  assert result["previous_results_deleted"] is False
  assert result["existing_evaluation_processes_untouched"] is True


def test_failed_initial_validation_never_sends_signals(
  module, monkeypatch, signal_probe
):
  def reject():
    raise RuntimeError("live worker")

  monkeypatch.setattr(module, "validate_empty_queue", reject)
  with pytest.raises(RuntimeError, match="live worker"):
    module.stop_empty_queue()
  assert signal_probe == [("open", module.OLD_PID), ("close", 97)]


def test_success_requires_old_controller_to_publish_interrupted(
  module, monkeypatch, signal_probe
):
  monkeypatch.setattr(module, "validate_empty_queue", lambda: {"starttime": 12345})
  processes = iter([{"state": "T", "starttime": 12345}, None])
  monkeypatch.setattr(module.queue, "process_info", lambda _: next(processes))
  monkeypatch.setattr(module.queue, "read_json", lambda _: {"status": "running"})
  with pytest.raises(RuntimeError, match="interrupted status"):
    module.stop_empty_queue()
  assert signal_probe[-1] == ("close", 97)
