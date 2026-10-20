"""Exercise the handoff without sending any real process signal or spawning work."""

from __future__ import annotations

import copy
import importlib.util
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1] / "scripts/deploy/run_pi05_parallel_handoff.py"
)


@pytest.fixture
def module():
  spec = importlib.util.spec_from_file_location("parallel_handoff_under_test", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def dump(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value))


def batch_for(group):
  return {
    "task": group["task"],
    "deployment_id": f"deployment-{group['step']}",
    "checkpoint_sha256": f"checkpoint-{group['step']}",
    "planned": 20,
    "attempted": 20,
    "valid_trials": 20,
    "successes": 0,
    "results": [
      {
        "seed": seed,
        "returncode": 0,
        "valid_trial": True,
        "deployment_integrity": {"status": "verified"},
      }
      for seed in range(20)
    ],
  }


@pytest.fixture
def harness(module, monkeypatch, tmp_path):
  old_root = tmp_path / "serial"
  job_root = tmp_path / "parallel"
  old_root.mkdir()
  job_root.mkdir()
  groups = [
    {
      "task": task,
      "step": step,
      "checkpoint": str(tmp_path / "checkpoints" / task / str(step)),
      "output_dir": str(old_root / task / f"step{step}_N20_video20"),
      "status": "complete" if index == 0 else "evaluating" if index == 1 else "pending",
    }
    for index, (task, step) in enumerate(
      (task, step)
      for task in ("whiteboard-wipe-horizontal", "usb-unplug", "bulb-unscrew")
      for step in (20000, 24999)
    )
  ]
  old = {
    "pid": module.SERIAL_PID,
    "status": "running",
    "task": "whiteboard-wipe-horizontal",
    "checkpoint_step": 24999,
    "stage": "evaluate",
    "settings": {"execute_steps": "horizon", "num_trials": 20},
    "groups": groups,
  }
  for group in groups[:2]:
    batch = batch_for(group)
    output = Path(group["output_dir"])
    dump(output / "run/evaluation/summary.json", batch)
    dump(
      output / "deployment/deployment_manifest.json",
      {
        "deployment_id": batch["deployment_id"],
        "checkpoint_sha256": batch["checkpoint_sha256"],
      },
    )
  current_manifest = Path(groups[1]["output_dir"]) / "run/run_manifest.json"
  dump(current_manifest, {"status": "evaluating"})
  dump(old_root / "status.json", old)
  state = SimpleNamespace(
    parent_state="S",
    child_state="Z",
    exit_status=0,
    term_sent=False,
    signals=[],
    launches=[],
    closed=[],
    old=old,
    groups=groups,
    job_root=job_root,
    module=module,
    real_validate_serial=module.validate_serial,
  )
  serial_identity = {
    "pid": module.SERIAL_PID,
    "starttime": 100,
    "ppid": 1,
    "state": "S",
    "exit_status": 0,
  }
  evaluator_identity = {
    "pid": 2742515,
    "starttime": 200,
    "ppid": module.SERIAL_PID,
    "state": "S",
    "exit_status": 0,
  }

  def process_info(pid):
    if pid == module.SERIAL_PID:
      return {**serial_identity, "state": state.parent_state}
    assert pid == evaluator_identity["pid"], "unexpected or unrelated PID inspection"
    return {
      **evaluator_identity,
      "state": state.child_state,
      "exit_status": state.exit_status,
    }

  def send_signal(fd, signum, *args):
    assert fd == 9011, "the evaluator must never receive STOP, TERM, or CONT"
    state.signals.append(signum)
    if signum == signal.SIGSTOP:
      state.parent_state = "T"
    elif signum == signal.SIGTERM:
      assert state.parent_state == "T", "TERM must be queued before CONT"
      state.term_sent = True
    elif signum == signal.SIGCONT:
      if state.term_sent:
        state.parent_state = "Z"
        interrupted = copy.deepcopy(old)
        interrupted["status"] = "interrupted"
        interrupted["groups"][1]["status"] = "interrupted"
        dump(old_root / "status.json", interrupted)
        dump(
          current_manifest,
          {
            "status": "error",
            "managed_server_stopped": True,
            "evaluation_process_stopped": True,
          },
        )
      else:
        state.parent_state = "S"
    else:
      pytest.fail(f"unexpected signal {signum}")

  class FakeWorker:
    def __init__(self, command, **kwargs):
      index = int(command[command.index("--group-index") + 1])
      self.pid = 30000 + index
      self.returncode = 0
      state.launches.append((command, kwargs))
      group = groups[2 + index]
      output = job_root / group["task"] / f"step{group['step']}_N20_video20"
      dump(
        output / "group_status.json", {"status": "complete", "output_dir": str(output)}
      )
      dump(output / "run/evaluation/summary.json", batch_for(group))

    def poll(self):
      return self.returncode

    def terminate(self):
      pytest.fail("completed fake worker must not be terminated")

    def wait(self, **kwargs):
      return self.returncode

  real_close = module.os.close

  def close(fd):
    if fd in (9011, 9012):
      state.closed.append(fd)
    else:
      real_close(fd)

  monkeypatch.setattr(module, "OLD_ROOT", old_root)
  monkeypatch.setattr(
    module,
    "validate_serial",
    lambda: (copy.deepcopy(old), serial_identity, evaluator_identity),
  )
  monkeypatch.setattr(module, "process_info", process_info)
  monkeypatch.setattr(module, "assert_ports_free", lambda: None)
  monkeypatch.setattr(
    module.os,
    "pidfd_open",
    lambda pid: 9011 if pid == module.SERIAL_PID else 9012,
    raising=False,
  )
  monkeypatch.setattr(module.os, "close", close)
  monkeypatch.setattr(module.signal, "pidfd_send_signal", send_signal, raising=False)
  monkeypatch.setattr(module.signal, "signal", lambda *args: None)
  monkeypatch.setattr(
    module.os, "kill", lambda *args: pytest.fail("real PID signaling is forbidden")
  )
  monkeypatch.setattr(
    module.os,
    "killpg",
    lambda *args: pytest.fail("process group signaling is forbidden"),
  )
  monkeypatch.setattr(module.subprocess, "Popen", FakeWorker)
  monkeypatch.setattr(module.time, "sleep", lambda *args: None)
  return state


def test_success_retains_40_trials_launches_four_frozen_independent_workers(harness):
  state = harness
  assert state.module.run(state.job_root) == 0
  assert state.signals == [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT]
  assert state.closed == [9011, 9012]
  assert len(state.launches) == 4
  for index, (command, kwargs) in enumerate(state.launches):
    assert command[0] == str(state.module.SIM_PYTHON)
    assert command[command.index("--simulation-root") + 1] == str(state.module.SIM_ROOT)
    assert command[command.index("--openpi-root") + 1] == str(state.module.OPENPI_ROOT)
    assert command[command.index("--endpoint") + 1] == f"ws://127.0.0.1:{18784 + index}"
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["cwd"] == state.module.SIM_ROOT
    assert str(state.module.SIM_ROOT / "src") in kwargs["env"]["PYTHONPATH"]
    assert str(state.module.OPENPI_ROOT / "src") in kwargs["env"]["PYTHONPATH"]
  result = state.module.read_json(state.job_root / "status.json")
  assert result["status"] == "complete"
  assert result["completed_trials"] == result["valid_trials"] == 120
  assert result["completed_checkpoint_count"] == 6
  for actual, original in zip(result["groups"][:2], state.groups[:2], strict=True):
    assert actual["preserved_results"] is True
    assert actual["completed_trials"] == 20
    assert actual["output_dir"] == original["output_dir"]
  audit = state.module.read_json(state.job_root / "handoff_record.json")
  assert audit["evaluation_exit_code"] == 0
  assert audit["whiteboard_valid_trials"] == 20
  assert audit["original_serial_status"] == "interrupted"
  assert audit["original_run_status"] == "error"
  assert audit["no_remaining_group_started_by_serial"] is True


@pytest.mark.parametrize("failure", ["invalid_batch", "nonzero_exit"])
def test_failure_resumes_old_orchestrator_without_launching_workers(harness, failure):
  state = harness
  if failure == "nonzero_exit":
    state.exit_status = 2 << 8
  else:
    path = Path(state.groups[1]["output_dir"]) / "run/evaluation/summary.json"
    batch = state.module.read_json(path)
    batch["valid_trials"] = 19
    dump(path, batch)
  with pytest.raises(RuntimeError):
    state.module.run(state.job_root)
  assert state.signals == [signal.SIGSTOP, signal.SIGCONT]
  assert state.parent_state == "S"
  assert state.term_sent is False
  assert state.launches == []
  assert state.closed == [9011, 9012]
  result = state.module.read_json(state.job_root / "status.json")
  assert result["status"] == "failed"
  assert result["serial_resumed_after_handoff_failure"] is True
  assert result["serial_orchestrator_paused"] is False
  assert not (state.job_root / "handoff_record.json").exists()


@pytest.mark.parametrize(
  "mutation",
  ["duplicate_seed", "child_failure", "integrity_failure", "identity_mismatch"],
)
def test_batch_validation_rejects_incomplete_or_untrusted_results(harness, mutation):
  state = harness
  group = state.groups[1]
  path = Path(group["output_dir"]) / "run/evaluation/summary.json"
  batch = state.module.read_json(path)
  if mutation == "duplicate_seed":
    batch["results"][-1]["seed"] = 0
  elif mutation == "child_failure":
    batch["results"][-1]["returncode"] = 2
  elif mutation == "integrity_failure":
    batch["results"][-1]["deployment_integrity"]["status"] = "mismatch"
  else:
    batch["deployment_id"] = "other-deployment"
  dump(path, batch)
  with pytest.raises(RuntimeError):
    state.module.validated_batch(path, group)
  assert not state.signals
  assert not state.launches


def test_serial_validation_discovers_child_when_running_manifest_has_no_pid(
  harness, monkeypatch
):
  state = harness
  module = state.module
  state.child_state = "S"
  run_root = Path(state.groups[1]["output_dir"]) / "run"
  dump(
    run_root / "run_manifest.json",
    {
      "status": "evaluating",
      "task": "whiteboard-wipe-horizontal",
      "num_trials": 20,
      "video_count": 20,
      "execute_steps": "horizon",
      "guard_mode": "observe",
      "max_sim_seconds": 90,
    },
  )
  original_read_text = Path.read_text

  def read_text(path, *args, **kwargs):
    if str(path) == f"/proc/{module.SERIAL_PID}/task/{module.SERIAL_PID}/children":
      return "2741538 2742515"
    return original_read_text(path, *args, **kwargs)

  commands = {
    module.SERIAL_PID: [
      str(module.SIM_ROOT / "scripts/deploy/run_pi05_checkpoint_serial.py"),
      str(module.OLD_ROOT),
    ],
    2741538: ["unrelated-model-service"],
    2742515: [
      str(module.SIM_ROOT / "scripts/evaluate/evaluate.py"),
      "--output-dir",
      str(run_root / "evaluation"),
    ],
  }
  monkeypatch.setattr(Path, "read_text", read_text)
  monkeypatch.setattr(module, "command_line", lambda pid: commands[pid])
  old, serial, evaluator = state.real_validate_serial()
  assert old["checkpoint_step"] == 24999
  assert serial["pid"] == module.SERIAL_PID
  assert evaluator["pid"] == 2742515
  assert evaluator["ppid"] == module.SERIAL_PID
  assert not state.signals


def test_start_uses_pidfd_capable_controller_python_not_simulator(
  harness, monkeypatch, tmp_path
):
  state = harness
  module = state.module
  monkeypatch.setattr(module, "EVALUATION_PARENT", tmp_path)
  tokenizer = module.OLD_ROOT / ".cache/openpi/big_vision/paligemma_tokenizer.model"
  tokenizer.parent.mkdir(parents=True)
  tokenizer.write_bytes(b"test-only-tokenizer")
  for group in state.groups[2:]:
    committed = Path(group["checkpoint"]) / "_CHECKPOINT_METADATA"
    committed.parent.mkdir(parents=True)
    committed.write_text("{}")
  launched = []

  def spawn(command, **kwargs):
    launched.append((command, kwargs))
    return SimpleNamespace(pid=42001)

  monkeypatch.setattr(module.subprocess, "Popen", spawn)
  job_root = tmp_path / "pi05_h50_k50_parallel_test"
  assert module.start(job_root) == 0
  assert len(launched) == 1
  command, kwargs = launched[0]
  assert command[0] == str(module.CONTROLLER_PYTHON)
  assert command[0] != str(module.SIM_PYTHON)
  assert kwargs["start_new_session"] is True
  assert kwargs["stdin"] is subprocess.DEVNULL
  assert (job_root / "parallel.pid").read_text().strip() == "42001"
  assert not state.signals


@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_worker_uses_public_cleanup_record_before_marking_complete(
  tmp_path, monkeypatch, cleanup_ok
):
  worker_path = SCRIPT.with_name("run_pi05_checkpoint_worker.py")
  spec = importlib.util.spec_from_file_location(
    "checkpoint_worker_under_test", worker_path
  )
  worker = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(worker)
  job_root = tmp_path / "job"
  job_root.mkdir()
  calls = []

  def invoke(name, result):
    def operation(*args, **kwargs):
      calls.append((name, args, kwargs))
      return result

    return operation

  deployment = SimpleNamespace(
    build_prepare_plan=invoke("build", "plan"),
    execute_prepare=invoke("prepare", 0),
    verify_deployment=invoke("verify", {}),
    write_pipeline_deployment=invoke("audit", None),
    run_deployment=invoke(
      "run",
      {
        "evaluation_returncode": 0,
        "managed_server_stopped": cleanup_ok,
        "evaluation_process_stopped": cleanup_ok,
      },
    ),
  )
  monkeypatch.setattr(worker, "validate_scope", lambda *args: None)
  monkeypatch.setattr(worker, "validate_assets", lambda *args: None)
  monkeypatch.setattr(
    worker,
    "load_api",
    lambda *args: SimpleNamespace(deployment=deployment, profile="frozen-profile"),
  )
  monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
  endpoint = "ws://127.0.0.1:18784"
  if cleanup_ok:
    assert (
      worker.run_worker(
        job_root,
        tmp_path / "frozen/kaitactilesim",
        tmp_path / "frozen/openpi",
        0,
        endpoint,
      )
      == 0
    )
  else:
    with pytest.raises(RuntimeError, match="cleanup failed"):
      worker.run_worker(
        job_root,
        tmp_path / "frozen/kaitactilesim",
        tmp_path / "frozen/openpi",
        0,
        endpoint,
      )
  assert [item[0] for item in calls] == ["build", "prepare", "verify", "audit", "run"]
  settings = calls[-1][2]
  assert settings["profile"] == "frozen-profile"
  assert settings["endpoint"] == endpoint
  assert settings["num_trials"] == settings["video_count"] == 20
  assert settings["max_sim_seconds"] == 90
  assert settings["execute_steps"] == "horizon"
  assert settings["guard_mode"] == "observe"
  group_status = json.loads(
    (job_root / "usb-unplug/step20000_N20_video20/group_status.json").read_text()
  )
  assert group_status["status"] == ("complete" if cleanup_ok else "failed")
