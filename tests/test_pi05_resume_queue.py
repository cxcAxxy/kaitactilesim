"""Validate preserved seed prefixes and fresh, isolated resume scheduling."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/deploy"
COUNTS = (20, 19, 13, 12, 0, 0)


@pytest.fixture
def module(monkeypatch):
  monkeypatch.syspath_prepend(str(SCRIPTS))
  spec = importlib.util.spec_from_file_location(
    "resume_queue_under_test", SCRIPTS / "resume_pi05_new_tasks_queue.py"
  )
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def dump(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value))


def trial_row(seed):
  return {
    "seed": seed,
    "valid_trial": True,
    "returncode": 0,
    "success": False,
    "summary": f"seed_{seed:03d}/summary.json",
  }


@pytest.fixture
def prior(module, tmp_path, monkeypatch):
  root = tmp_path / "prior"
  monkeypatch.setattr(module, "SOURCE_ROOT", root)
  monkeypatch.setattr(module.queue, "process_info", lambda _: None)
  groups = []
  for index, ((task, step, checkpoint), count) in enumerate(
    zip(module.queue.worker.NEW_GROUPS, COUNTS, strict=True)
  ):
    output = root / task / f"step{step}_N20_video20"
    group = {
      "task": task,
      "step": step,
      "checkpoint": str(checkpoint),
      "output_dir": str(output),
      "status": "failed" if count else "queued",
      "worker_pid": 7100 + index if count else None,
    }
    groups.append(group)
    if not count:
      continue
    record = {
      "managed_server_stopped": True,
      "evaluation_process_stopped": True,
      "server_pid": 7200 + index,
      "evaluation_pid": 7300 + index,
      "num_trials": 20,
      "seed_start": 0,
      "execute_steps": "horizon",
      "max_sim_seconds": 90,
      "guard_mode": "observe",
    }
    dump(output / "run/run_manifest.json", record)
    manifest = {
      "task": task,
      "checkpoint_path": str(checkpoint),
      "checkpoint_sha256": f"checkpoint_{index}",
      "deployment_id": f"deployment_{index}",
    }
    dump(output / "deployment/deployment_manifest.json", manifest)
    rows = [trial_row(seed) for seed in range(count)]
    key = "results" if index < 2 else "trials"
    dump(output / "run/evaluation/summary.json", {key: rows, "valid_trials": count})
    for row in rows:
      dump(
        output / "run/evaluation" / row["summary"],
        {
          "task": task,
          "seed": row["seed"],
          "prediction_horizon": 50,
          "execute_steps": 50,
          "max_sim_seconds": 90,
          "guard_mode": "observe",
          "action_interpolation": "linear",
          "physics_valid": True,
          "checkpoint_sha256": manifest["checkpoint_sha256"],
          "deployment_id": manifest["deployment_id"],
          "status": "task_not_completed",
          "deployment_integrity": {"status": "verified"},
        },
      )
  status = {
    "status": "failed",
    "active_new_groups": 0,
    "pid": 7000,
    "groups": groups,
  }
  dump(root / "status.json", status)
  return SimpleNamespace(root=root, groups=groups, status=status)


def test_build_plan_preserves_64_and_only_schedules_56_missing_seeds(module, prior):
  plan = module.build_resume_plan(prior.root)
  assert plan["preserved_trials"] == 64
  assert plan["remaining_trials"] == 56
  assert plan["prior_results_modified"] is False
  assert [g["pending_seeds"] for g in plan["groups"]] == [
    [],
    [19],
    list(range(13, 20)),
    list(range(12, 20)),
    list(range(20)),
    list(range(20)),
  ]
  for group, count in zip(plan["groups"], COUNTS, strict=True):
    assert group["preserved_seeds"] == list(range(count))
    assert not set(group["preserved_seeds"]) & set(group["pending_seeds"])
    assert sorted(group["preserved_seeds"] + group["pending_seeds"]) == list(range(20))


@pytest.mark.parametrize("pid", [7000, 7100, 7200, 7300])
def test_build_plan_rejects_any_live_prior_controller_worker_or_service(
  module, prior, monkeypatch, pid
):
  monkeypatch.setattr(
    module.queue,
    "process_info",
    lambda actual: {"state": "S"} if actual == pid else None,
  )
  with pytest.raises(RuntimeError, match="still alive"):
    module.build_resume_plan(prior.root)


@pytest.mark.parametrize(
  "change",
  ["checkpoint", "manifest", "settings", "cleanup", "queued_output", "row_count"],
)
def test_build_plan_rejects_changed_identity_settings_or_cleanup(module, prior, change):
  output = Path(prior.groups[0]["output_dir"])
  if change == "checkpoint":
    prior.groups[0]["checkpoint"] = "/unapproved/checkpoint"
    dump(prior.root / "status.json", prior.status)
  elif change == "queued_output":
    Path(prior.groups[4]["output_dir"]).mkdir(parents=True)
  else:
    name = (
      "deployment/deployment_manifest.json"
      if change == "manifest"
      else "run/evaluation/summary.json"
      if change == "row_count"
      else "run/run_manifest.json"
    )
    value = json.loads((output / name).read_text())
    if change == "manifest":
      value["checkpoint_path"] = "/unapproved/checkpoint"
    elif change == "settings":
      value["max_sim_seconds"] = 88
    elif change == "cleanup":
      value["managed_server_stopped"] = False
    else:
      value["valid_trials"] = 19
    dump(output / name, value)
  with pytest.raises((ValueError, RuntimeError)):
    module.build_resume_plan(prior.root)


@pytest.mark.parametrize(
  "change",
  [
    "seed",
    "checkpoint_sha256",
    "deployment_id",
    "guard",
    "identity_errors",
    "integrity_errors",
  ],
)
def test_validate_rows_rejects_wrong_seed_or_trial_identity(module, prior, change):
  output = Path(prior.groups[0]["output_dir"])
  manifest = json.loads((output / "deployment/deployment_manifest.json").read_text())
  rows = [trial_row(0)]
  if change == "seed":
    rows[0]["seed"] = 1
  elif change == "identity_errors":
    rows[0]["identity_errors"] = {"checkpoint": "mismatch"}
  else:
    path = output / "run/evaluation/seed_000/summary.json"
    value = json.loads(path.read_text())
    if change == "integrity_errors":
      value["deployment_integrity"]["errors"] = ["source changed"]
    else:
      value["guard_mode" if change == "guard" else change] = "changed"
    dump(path, value)
  with pytest.raises(ValueError):
    module.validate_rows(rows, "install-ram", manifest, output)


@pytest.mark.parametrize("rows_key", ["results", "trials"])
def test_refresh_combines_preserved_and_fresh_without_double_counting_complete_ram(
  module, prior, tmp_path, rows_key
):
  plan = module.build_resume_plan(prior.root)
  root = tmp_path / "fresh"
  root.mkdir()
  status = {"groups": copy.deepcopy(plan["groups"])}
  for group in status["groups"]:
    group.update(status="queued", output_dir=str(root / str(group["group_index"])))
  # Point completed RAM at the existing 20-row native summary; it must be skipped.
  status["groups"][0]["output_dir"] = plan["groups"][0]["preserved_output_dir"]
  for index, seeds in [(1, [19]), (2, [13, 14])]:
    output = Path(status["groups"][index]["output_dir"])
    dump(
      output / "run/evaluation/summary.json", {rows_key: [trial_row(s) for s in seeds]}
    )
  module.refresh(status, root)
  assert status["completed_trials"] == status["valid_trials"] == 67
  assert status["resumed_completed_trials"] == 3
  assert [g["completed_trials"] for g in status["groups"]] == [20, 20, 15, 12, 0, 0]
  module.refresh(status, root)
  assert status["completed_trials"] == 67


def test_refresh_rejects_seed_overlap_with_preserved_trials(module, prior, tmp_path):
  plan = module.build_resume_plan(prior.root)
  root = tmp_path / "fresh"
  root.mkdir()
  group = copy.deepcopy(plan["groups"][2])
  group.update(status="running", output_dir=str(root / "poker"))
  dump(
    Path(group["output_dir"]) / "run/evaluation/summary.json",
    {"trials": [trial_row(12)]},
  )
  with pytest.raises(ValueError, match="overlap preserved"):
    module.refresh({"groups": [group]}, root)


def mock_scheduler(module, prior, tmp_path, monkeypatch, cleanup=True, missing=False):
  root = tmp_path / "fresh"
  root.mkdir()
  dump(root / "resume_plan.json", module.build_resume_plan(prior.root))
  state = SimpleNamespace(tick=0, launches=[], terminated=[])
  monkeypatch.setattr(module.queue.worker, "EVALUATION_PARENT", tmp_path)
  monkeypatch.setattr(module.queue, "validate_configuration", lambda *_args: None)
  monkeypatch.setattr(
    module.queue, "old_reservations", lambda _: (0, {"reserved_slots": 0})
  )
  monkeypatch.setattr(module.signal, "signal", lambda *_args: None)

  class FakeProcess:
    def __init__(self, index):
      self.pid = 9000 + index
      self.index = index

    def poll(self):
      return 0 if state.tick else None

    def terminate(self):
      state.terminated.append(self.index)

    def wait(self, timeout):
      assert timeout == 90
      return -15

  def launch(group, *_args):
    process = FakeProcess(group["group_index"])
    group.update(status="running", worker_pid=process.pid)
    state.launches.append((state.tick, process.index))
    assert len(state.launches) <= module.queue.CAPACITY
    seeds = group["pending_seeds"]
    if missing and process.index == 1:
      seeds = []
    dump(
      Path(group["output_dir"]) / "run/evaluation/summary.json",
      {"results": [trial_row(seed) for seed in seeds]},
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
    assert state.tick < 5

  monkeypatch.setattr(module, "launch_group", launch)
  monkeypatch.setattr(module.queue, "finalize_worker", finalize)
  monkeypatch.setattr(module.time, "sleep", sleep)
  return root, state


def test_scheduler_launches_only_five_remaining_groups_immediately_and_finishes_120(
  module, prior, tmp_path, monkeypatch
):
  root, state = mock_scheduler(module, prior, tmp_path, monkeypatch)
  assert module.run(root, tmp_path / "sim", tmp_path / "openpi") == 0
  assert state.launches == [(0, index) for index in range(1, 6)]
  status = json.loads((root / "status.json").read_text())
  assert status["status"] == "complete"
  assert status["completed_trials"] == 120
  assert status["resumed_completed_trials"] == 56
  assert status["completed_checkpoint_count"] == 6
  assert all(g["completed_trials"] == 20 for g in status["groups"])


def test_scheduler_still_fails_closed_when_worker_cleanup_is_unconfirmed(
  module, prior, tmp_path, monkeypatch
):
  root, state = mock_scheduler(module, prior, tmp_path, monkeypatch, cleanup=False)
  with pytest.raises(RuntimeError, match="owned cleanup unconfirmed"):
    module.run(root, tmp_path / "sim", tmp_path / "openpi")
  status = json.loads((root / "status.json").read_text())
  assert status["status"] == "failed"
  assert len(state.launches) == 5
  assert any("owned cleanup unconfirmed" in error for error in status["errors"])


def test_zero_exit_without_all_results_cannot_mark_resume_complete(
  module, prior, tmp_path, monkeypatch
):
  root, _ = mock_scheduler(module, prior, tmp_path, monkeypatch, missing=True)
  try:
    code = module.run(root, tmp_path / "sim", tmp_path / "openpi")
  except (RuntimeError, ValueError):
    code = 1
  status = json.loads((root / "status.json").read_text())
  assert code != 0
  assert status["status"] == "failed"
  assert status["completed_trials"] == 119


def test_run_rejects_tampered_plan_before_any_launch(
  module, prior, tmp_path, monkeypatch
):
  root, state = mock_scheduler(module, prior, tmp_path, monkeypatch)
  plan = json.loads((root / "resume_plan.json").read_text())
  plan["groups"][1]["seed_start"] = 18
  plan["groups"][1]["pending_seeds"] = [18, 19]
  plan["groups"][1]["num_trials"] = 2
  dump(root / "resume_plan.json", plan)
  with pytest.raises((ValueError, RuntimeError)):
    module.run(root, tmp_path / "sim", tmp_path / "openpi")
  assert state.launches == []
