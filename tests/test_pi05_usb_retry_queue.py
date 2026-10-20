"""USB retry must never relaunch or signal healthy primary RAM/Poker workers."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/deploy"


@pytest.fixture
def module(monkeypatch):
  monkeypatch.syspath_prepend(str(SCRIPTS))
  spec = importlib.util.spec_from_file_location(
    "usb_retry_under_test", SCRIPTS / "retry_pi05_usb_groups.py"
  )
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def dump(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value))


def row(seed, valid=True):
  return {
    "seed": seed,
    "valid_trial": valid,
    "returncode": 0 if valid else 1,
    "success": False,
  }


@pytest.fixture
def primary(module, tmp_path, monkeypatch):
  root = tmp_path / "primary"
  previous = tmp_path / "previous"
  monkeypatch.setattr(module, "PRIMARY_ROOT", root)
  monkeypatch.setattr(module, "PRIMARY_CONTROLLER_PID", 7000)
  monkeypatch.setattr(module.resume, "SOURCE_ROOT", previous)
  monkeypatch.setattr(module.queue.worker, "EVALUATION_PARENT", tmp_path)
  monkeypatch.setattr(module.queue, "port_free", lambda _: True)
  monkeypatch.setattr(
    module.queue, "process_info", lambda pid: {"state": "S"} if pid == 7000 else None
  )
  groups = []
  for index, (task, step, checkpoint) in enumerate(module.queue.worker.NEW_GROUPS):
    source = previous if index == 0 else root
    output = source / task / f"step{step}_N20_video20"
    count = [20, 20, 14, 13, 0, 0][index]
    groups.append(
      {
        "group_index": index,
        "task": task,
        "step": step,
        "checkpoint": str(checkpoint),
        "output_dir": str(output),
        "status": "complete" if index < 2 else "running" if index < 4 else "failed",
        "worker_pid": 7100 + index,
        "completed_trials": count,
        "valid_trials": count,
        "preserved_completed_trials": [20, 19, 13, 12, 0, 0][index],
        "fresh_completed_trials": count - [20, 19, 13, 12, 0, 0][index],
        "owned_cleanup_confirmed": index != 2 and index != 3,
      }
    )
    if index < 4:
      continue
    dump(
      output / "group_status.json",
      {
        "pid": 7100 + index,
        "status": "failed",
        "managed_server_stopped": True,
        "evaluation_process_stopped": True,
        "errors": ["cold compile timeout"],
      },
    )
    dump(
      output / "run/run_manifest.json",
      {
        "server_pid": 7200 + index,
        "evaluation_pid": 7300 + index,
        "managed_server_stopped": True,
        "evaluation_process_stopped": True,
        "num_trials": 20,
        "seed_start": 0,
        "execute_steps": "horizon",
        "max_sim_seconds": 90,
        "guard_mode": "observe",
      },
    )
    dump(
      output / "deployment/deployment_manifest.json",
      {
        "task": task,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": f"checkpoint{index}",
      },
    )
    dump(
      output / "run/evaluation/summary.json",
      {"trials": [row(0, False)], "valid_trials": 0},
    )
  status = {
    "pid": 7000,
    "status": "running",
    "groups": groups,
    "settings": {
      "total_checkpoint_worker_cap": 6,
      "seed_workers_per_checkpoint": 1,
      "prediction_horizon": 50,
      "execute_steps": "horizon",
      "max_sim_seconds": 90,
      "guard_mode": "observe",
    },
    "errors": [
      "usb-insert step20000: worker exit 1",
      "usb-insert step24999: worker exit 1",
    ],
  }
  dump(root / "status.json", status)
  return SimpleNamespace(root=root, status=status, groups=groups)


def test_plan_retries_only_usb_40_valid_zero_trials_without_repeating_primary(
  module, primary
):
  plan = module.build_retry_plan()
  assert plan["retry_trials_planned"] == 40
  assert plan["total_trials"] == 120
  assert plan["prior_results_modified"] is False
  assert plan["primary_workers_signalled"] is False
  assert [g["group_index"] for g in plan["usb_groups"]] == [4, 5]
  assert all(g["seed_start"] == 0 and g["num_trials"] == 20 for g in plan["usb_groups"])
  assert [g["prior_failure_errors"] for g in plan["usb_groups"]] == [
    ["cold compile timeout"]
  ] * 2


@pytest.mark.parametrize("where", ["group", "detail", "record"])
def test_cleanup_false_rejects_retry(module, primary, where):
  group = primary.groups[4]
  if where == "group":
    group["owned_cleanup_confirmed"] = False
    dump(primary.root / "status.json", primary.status)
  else:
    p = Path(group["output_dir"]) / (
      "group_status.json" if where == "detail" else "run/run_manifest.json"
    )
    value = json.loads(p.read_text())
    value["managed_server_stopped"] = False
    dump(p, value)
  with pytest.raises(RuntimeError, match="cleanup"):
    module.build_retry_plan()


@pytest.mark.parametrize("pid", [7104, 7204, 7304, 7105, 7205, 7305])
def test_any_live_owned_prior_usb_pid_rejects_retry(module, primary, monkeypatch, pid):
  monkeypatch.setattr(
    module.queue,
    "process_info",
    lambda actual: {"state": "S"} if actual in (7000, pid) else None,
  )
  with pytest.raises(RuntimeError, match="still alive"):
    module.build_retry_plan()


def test_dead_primary_with_running_poker_rejects_retry(module, primary, monkeypatch):
  monkeypatch.setattr(module.queue, "process_info", lambda _: None)
  with pytest.raises(RuntimeError, match="primary controller exited"):
    module.build_retry_plan()


@pytest.mark.parametrize("port", [18804, 18805])
def test_occupied_usb_port_rejects_retry(module, primary, monkeypatch, port):
  monkeypatch.setattr(module.queue, "port_free", lambda actual: actual != port)
  with pytest.raises(RuntimeError, match="port is occupied"):
    module.build_retry_plan()


def test_prior_valid_usb_seed_refuses_repeat(module, primary):
  p = Path(primary.groups[4]["output_dir"]) / "run/evaluation/summary.json"
  dump(p, {"trials": [row(0)], "valid_trials": 1})
  with pytest.raises(ValueError, match="refusing to repeat"):
    module.build_retry_plan()


@pytest.mark.parametrize("change", ["pid", "checkpoint", "settings", "output"])
def test_changed_primary_binding_refuses_retry(module, primary, change):
  if change == "pid":
    primary.status["pid"] = 7999
  elif change == "checkpoint":
    primary.groups[2]["checkpoint"] = "/unapproved/checkpoint"
  elif change == "settings":
    primary.status["settings"]["prediction_horizon"] = 30
  else:
    primary.groups[4]["output_dir"] = "/unapproved/output"
  dump(primary.root / "status.json", primary.status)
  with pytest.raises(ValueError):
    module.build_retry_plan()


def make_status(module, primary, root):
  plan = module.build_retry_plan()
  groups = copy.deepcopy(primary.groups[:4]) + copy.deepcopy(plan["usb_groups"])
  for g in groups[4:]:
    g.update(
      status="running",
      output_dir=str(root / "usb-insert" / f"step{g['step']}_N20_video20"),
      completed_trials=0,
      valid_trials=0,
    )
  return {"groups": groups, "usb_errors": [], "errors": []}


@pytest.mark.parametrize("native_key", ["results", "trials"])
def test_refresh_primary_preserved_plus_fresh_and_new_usb_without_old_failures(
  module, primary, tmp_path, native_key
):
  root = tmp_path / "retry"
  root.mkdir()
  status = make_status(module, primary, root)
  for index, count in ((4, 2), (5, 3)):
    dump(
      Path(status["groups"][index]["output_dir"]) / "run/evaluation/summary.json",
      {native_key: [row(seed) for seed in range(count)], "valid_trials": count},
    )
  module.refresh(status, root)
  assert status["completed_trials"] == status["valid_trials"] == 72
  assert status["active_primary_groups"] == 2
  assert status["active_usb_retry_groups"] == 2
  assert status["retry_completed_trials"] == 5
  assert status["groups"][2]["preserved_completed_trials"] == 13
  assert status["groups"][2]["fresh_completed_trials"] == 1
  assert status["errors"] == []
  module.refresh(status, root)
  assert status["completed_trials"] == 72


def test_refresh_rejects_duplicate_or_nonzero_usb_seed_prefix(
  module, primary, tmp_path
):
  root = tmp_path / "retry"
  root.mkdir()
  status = make_status(module, primary, root)
  dump(
    Path(status["groups"][4]["output_dir"]) / "run/evaluation/summary.json",
    {"trials": [row(1)], "valid_trials": 1},
  )
  with pytest.raises(ValueError, match="zero-based"):
    module.refresh(status, root)


def test_refresh_preserves_real_primary_failure(module, primary, tmp_path):
  root = tmp_path / "retry"
  root.mkdir()
  status = make_status(module, primary, root)
  primary.groups[2]["status"] = "failed"
  primary.status["errors"].append("poker-draw step20000: worker exit 1")
  dump(primary.root / "status.json", primary.status)
  module.refresh(status, root)
  assert any("poker-draw" in err for err in status["errors"])
  assert all("usb-insert" not in err for err in status["errors"])


def mock_run(
  module, primary, tmp_path, monkeypatch, *, missing=False, fail_usb=False, cleanup=True
):
  root = tmp_path / "retry"
  root.mkdir()
  dump(root / "usb_retry_plan.json", module.build_retry_plan())
  state = SimpleNamespace(tick=0, launches=[], terminated=[])
  monkeypatch.setattr(module.queue, "validate_configuration", lambda *_: None)
  monkeypatch.setattr(module.signal, "signal", lambda *_: None)

  class FakeProcess:
    def __init__(self, index):
      self.index, self.pid = index, 9000 + index

    def poll(self):
      return (1 if fail_usb and self.index == 4 else 0) if state.tick else None

    def terminate(self):
      state.terminated.append(self.index)

    def wait(self, timeout):
      assert timeout == 90
      return -15

  def launch(group, *_):
    assert group["group_index"] in (4, 5)
    process = FakeProcess(group["group_index"])
    group.update(status="running", worker_pid=process.pid)
    state.launches.append(group["group_index"])
    count = 19 if missing and process.index == 4 else 20
    dump(
      Path(group["output_dir"]) / "run/evaluation/summary.json",
      {"trials": [row(s) for s in range(count)], "valid_trials": count},
    )
    return process

  def finalize(group, code):
    group.update(
      status="complete" if code == 0 and cleanup else "failed",
      owned_cleanup_confirmed=cleanup,
      worker_returncode=code,
    )
    return cleanup

  def sleep(seconds):
    assert seconds == 15
    state.tick += 1
    assert state.tick < 4
    if state.tick == 2:
      for group in primary.groups[:4]:
        group.update(status="complete", completed_trials=20, valid_trials=20)
      primary.status["status"] = "failed"  # Failed old USB groups remain history.
      dump(primary.root / "status.json", primary.status)

  monkeypatch.setattr(module.resume, "launch_group", launch)
  monkeypatch.setattr(module.queue, "finalize_worker", finalize)
  monkeypatch.setattr(module.time, "sleep", sleep)
  return root, state


def test_scheduler_only_launches_two_usb_and_waits_for_primary_completion(
  module, primary, tmp_path, monkeypatch
):
  root, state = mock_run(module, primary, tmp_path, monkeypatch)
  assert module.run(root, tmp_path / "sim", tmp_path / "openpi") == 0
  assert state.launches == [4, 5]
  assert state.terminated == []
  assert state.tick == 2
  status = json.loads((root / "status.json").read_text())
  assert status["status"] == "complete"
  assert status["completed_trials"] == 120
  assert status["completed_checkpoint_count"] == 6
  assert status["errors"] == []


@pytest.mark.parametrize("failure", ["missing", "worker"])
def test_scheduler_incomplete_or_failed_usb_does_not_kill_healthy_groups(
  module, primary, tmp_path, monkeypatch, failure
):
  root, state = mock_run(
    module,
    primary,
    tmp_path,
    monkeypatch,
    missing=failure == "missing",
    fail_usb=failure == "worker",
  )
  assert module.run(root, tmp_path / "sim", tmp_path / "openpi") == 1
  assert state.launches == [4, 5]
  assert state.terminated == []
  status = json.loads((root / "status.json").read_text())
  assert status["groups"][4]["status"] == "failed"
  assert status["groups"][5]["status"] == "complete"
  assert all(g["status"] == "complete" for g in status["groups"][:4])


def test_failed_owned_cleanup_aborts_only_owned_usb_children(
  module, primary, tmp_path, monkeypatch
):
  root, state = mock_run(module, primary, tmp_path, monkeypatch, cleanup=False)
  with pytest.raises(RuntimeError, match="USB cleanup unconfirmed"):
    module.run(root, tmp_path / "sim", tmp_path / "openpi")
  assert state.launches == [4, 5]
  assert all(index in (4, 5) for index in state.terminated)


def test_start_refuses_existing_output_before_any_launch(
  module, primary, tmp_path, monkeypatch
):
  root = tmp_path / "existing"
  root.mkdir()
  monkeypatch.setattr(module.queue, "validate_configuration", lambda *_: None)
  monkeypatch.setattr(
    module.subprocess, "Popen", lambda *_a, **_k: pytest.fail("must not launch")
  )
  with pytest.raises(FileExistsError):
    module.start(root, tmp_path / "sim", tmp_path / "openpi")
