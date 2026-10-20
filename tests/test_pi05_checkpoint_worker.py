"""Verify the fixed checkpoint workers without starting GPU or NAS workloads."""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1] / "scripts/deploy/run_pi05_checkpoint_worker.py"
)


@pytest.fixture
def module():
  spec = importlib.util.spec_from_file_location("pi05_worker_under_test", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


@pytest.fixture
def harness(module, monkeypatch, tmp_path):
  job_root = tmp_path / "pi05_h50_k50_parallel_test"
  job_root.mkdir()
  profile = object()
  state = SimpleNamespace(
    module=module,
    job_root=job_root,
    simulation_root=tmp_path / "snapshot/kaitactilesim",
    openpi_root=tmp_path / "snapshot/openpi",
    profile=profile,
    group_index=0,
    new_tasks=False,
    num_trials=20,
    seed_start=0,
    calls=[],
    mode="success",
  )

  def validate_assets(*args):
    if state.mode == "preflight":
      raise RuntimeError("mock preflight failure")

  def build(task, family, **kwargs):
    state.calls.append(("build", task, family, kwargs))
    groups = module.NEW_GROUPS if state.new_tasks else module.GROUPS
    assert task == groups[state.group_index][0]
    assert family == "pi05"
    assert kwargs["checkpoint"] == groups[state.group_index][2]
    assert kwargs["profile"] is profile
    return "mock-prepare-plan"

  def prepare(plan):
    state.calls.append(("prepare", plan))
    return 2 if state.mode == "prepare" else 0

  def verify(manifest, **kwargs):
    state.calls.append(("verify", manifest, kwargs))
    if state.mode == "verify":
      raise RuntimeError("mock verification failure")
    return {"status": "verified"}

  def provenance(output, **kwargs):
    state.calls.append(("provenance", output, kwargs))

  def run(manifest, **kwargs):
    state.calls.append(("run", manifest, kwargs))
    if state.mode == "interrupted":
      raise KeyboardInterrupt("mock owned API interruption")
    if state.mode == "signal_handler":
      signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    record = {
      "evaluation_returncode": 2 if state.mode == "badreturn" else 0,
      "managed_server_stopped": state.mode != "server_cleanup",
      "evaluation_process_stopped": state.mode != "evaluation_cleanup",
    }
    if state.mode == "missing_returncode":
      del record["evaluation_returncode"]
    return record

  api = SimpleNamespace(
    profile=profile,
    deployment=SimpleNamespace(
      build_prepare_plan=build,
      execute_prepare=prepare,
      verify_deployment=verify,
      write_pipeline_deployment=provenance,
      run_deployment=run,
    ),
  )
  monkeypatch.setattr(module, "validate_scope", lambda *args: None)
  monkeypatch.setattr(module, "validate_assets", validate_assets)
  monkeypatch.setattr(module, "load_api", lambda *args: api)
  return state


def invoke(state):
  return state.module.run_worker(
    state.job_root,
    state.simulation_root,
    state.openpi_root,
    state.group_index,
    f"ws://127.0.0.1:{(18800 if state.new_tasks else 18784) + state.group_index}",
    new_tasks=state.new_tasks,
    num_trials=state.num_trials,
    seed_start=state.seed_start,
  )


def status_path(state):
  groups = state.module.NEW_GROUPS if state.new_tasks else state.module.GROUPS
  task, step, _ = groups[state.group_index]
  return state.job_root / task / f"step{step}_N20_video20/group_status.json"


def current_handlers():
  return {
    signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
  }


@pytest.mark.parametrize("group_index", range(4))
def test_fixed_groups_api_order_and_unchanged_settings(harness, group_index):
  harness.group_index = group_index
  before = current_handlers()
  assert invoke(harness) == 0
  assert current_handlers() == before
  assert [call[0] for call in harness.calls] == [
    "build",
    "prepare",
    "verify",
    "provenance",
    "run",
  ]
  task, step, checkpoint = harness.module.GROUPS[group_index]
  assert (task, step) == (
    ("usb-unplug", 20000),
    ("usb-unplug", 24999),
    ("bulb-unscrew", 20000),
    ("bulb-unscrew", 24999),
  )[group_index]
  kwargs = harness.calls[-1][2]
  assert kwargs == {
    "output_dir": status_path(harness).parent / "run",
    "profile": harness.profile,
    "service_mode": "managed",
    "endpoint": f"ws://127.0.0.1:{18784 + group_index}",
    "num_trials": 20,
    "seed_start": 0,
    "video_count": 20,
    "execute_steps": "horizon",
    "max_sim_seconds": 90,
    "guard_mode": "observe",
    "server_timeout": 600,
  }
  status = json.loads(status_path(harness).read_text())
  assert status["status"] == status["stage"] == "complete"
  assert status["group_index"] == group_index
  assert status["checkpoint"] == str(checkpoint)
  assert status["port"] == 18784 + group_index
  assert status["model_python"].endswith(".venv-pi05/bin/python")
  assert status["managed_server_stopped"] is True
  assert status["evaluation_process_stopped"] is True
  assert status["settings"]["prediction_horizon"] == 50
  assert status["settings"]["num_trials"] == 20
  assert status["settings"]["seed_start"] == 0
  assert status["settings"]["video_count"] == 20
  assert status["settings"]["training_started"] is False
  assert status["errors"] == []
  assert not list(status_path(harness).parent.glob(".group_status.*"))


@pytest.mark.parametrize(
  "group_index,seed_start,num_trials",
  [(1, 19, 1), (2, 13, 7), (3, 12, 8), (4, 0, 20), (5, 0, 20)],
)
def test_new_task_remaining_ranges_reach_api_and_status(
  harness, group_index, seed_start, num_trials
):
  harness.new_tasks = True
  harness.group_index = group_index
  harness.seed_start = seed_start
  harness.num_trials = num_trials
  assert invoke(harness) == 0
  kwargs = harness.calls[-1][2]
  assert kwargs["num_trials"] == num_trials
  assert kwargs["seed_start"] == seed_start
  assert kwargs["video_count"] == num_trials
  assert kwargs["endpoint"] == f"ws://127.0.0.1:{18800 + group_index}"
  assert kwargs["execute_steps"] == "horizon"
  status = json.loads(status_path(harness).read_text())
  assert status["status"] == "complete"
  assert status["settings"]["num_trials"] == num_trials
  assert status["settings"]["seed_start"] == seed_start
  assert status["settings"]["video_count"] == num_trials
  assert status["settings"]["prediction_horizon"] == 50
  assert status_path(harness).parent.name.endswith("_N20_video20")


@pytest.mark.parametrize(
  "num_trials,seed_start",
  [
    (0, 0),
    (21, 0),
    (1, -1),
    (1, 20),
    (2, 19),
    (20, 1),
    (1.5, 0),
    (1, 0.5),
    (True, 0),
    (1, False),
  ],
)
def test_invalid_trial_range_has_no_writes_or_api_calls(
  harness, num_trials, seed_start
):
  harness.new_tasks = True
  harness.num_trials = num_trials
  harness.seed_start = seed_start
  with pytest.raises(ValueError, match="trial range requires"):
    invoke(harness)
  assert list(harness.job_root.iterdir()) == []
  assert harness.calls == []


@pytest.mark.parametrize("num_trials,seed_start", [(1, 19), (7, 13), (8, 12)])
def test_old_groups_reject_nondefault_trial_ranges_before_writes(
  harness, num_trials, seed_start
):
  harness.num_trials = num_trials
  harness.seed_start = seed_start
  with pytest.raises(ValueError, match="require --new-tasks"):
    invoke(harness)
  assert list(harness.job_root.iterdir()) == []
  assert harness.calls == []


@pytest.mark.parametrize(
  "options,new_tasks,num_trials,seed_start",
  [
    ([], False, 20, 0),
    (["--new-tasks", "--num-trials", "1", "--seed-start", "19"], True, 1, 19),
  ],
)
def test_cli_propagates_optional_trial_range(
  module, monkeypatch, tmp_path, options, new_tasks, num_trials, seed_start
):
  job_root = tmp_path / "job"
  simulation_root = tmp_path / "kaitactilesim"
  openpi_root = tmp_path / "openpi"
  for directory in (job_root, simulation_root, openpi_root):
    directory.mkdir()
  calls = []

  def run(*args):
    calls.append(args)
    return 0

  monkeypatch.setattr(module, "run_worker", run)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT),
      "--job-root",
      str(job_root),
      "--simulation-root",
      str(simulation_root),
      "--openpi-root",
      str(openpi_root),
      "--group-index",
      "0",
      "--endpoint",
      "ws://127.0.0.1:18800" if new_tasks else "ws://127.0.0.1:18784",
      *options,
    ],
  )
  assert module.main() == 0
  assert len(calls) == 1
  assert calls[0][-3:] == (new_tasks, num_trials, seed_start)


def test_resumed_worker_cannot_overwrite_existing_output(harness):
  harness.new_tasks = True
  harness.group_index = 1
  harness.num_trials = 1
  harness.seed_start = 19
  assert invoke(harness) == 0
  previous = status_path(harness).read_bytes()
  previous_calls = list(harness.calls)
  with pytest.raises(FileExistsError):
    invoke(harness)
  assert status_path(harness).read_bytes() == previous
  assert harness.calls == previous_calls


def test_duplicate_invocation_preserves_existing_status_and_outputs(harness):
  assert invoke(harness) == 0
  previous = status_path(harness).read_bytes()
  marker = status_path(harness).parent / "user_result.txt"
  marker.write_text("preserve existing results")
  previous_calls = list(harness.calls)
  with pytest.raises(FileExistsError):
    invoke(harness)
  assert status_path(harness).read_bytes() == previous
  assert marker.read_text() == "preserve existing results"
  assert harness.calls == previous_calls


@pytest.mark.parametrize(
  "mode,expected_calls",
  [
    ("preflight", []),
    ("prepare", ["build", "prepare"]),
    ("verify", ["build", "prepare", "verify"]),
    ("badreturn", ["build", "prepare", "verify", "provenance", "run"]),
    ("server_cleanup", ["build", "prepare", "verify", "provenance", "run"]),
    ("evaluation_cleanup", ["build", "prepare", "verify", "provenance", "run"]),
    ("missing_returncode", ["build", "prepare", "verify", "provenance", "run"]),
  ],
)
def test_failed_prepare_evaluation_or_cleanup_never_marks_complete(
  harness, mode, expected_calls
):
  harness.mode = mode
  before = current_handlers()
  with pytest.raises(RuntimeError):
    invoke(harness)
  assert current_handlers() == before
  status = json.loads(status_path(harness).read_text())
  assert status["status"] == "failed"
  assert status["stage"] != "complete"
  assert status["errors"]
  assert [call[0] for call in harness.calls] == expected_calls


@pytest.mark.parametrize("mode", ["interrupted", "signal_handler"])
def test_interruption_records_failure_and_restores_handlers(harness, mode):
  harness.mode = mode
  before = current_handlers()
  with pytest.raises(KeyboardInterrupt):
    invoke(harness)
  assert current_handlers() == before
  status = json.loads(status_path(harness).read_text())
  assert status["status"] == "interrupted"
  assert status["stage"] == "evaluate"
  assert status["errors"]


@pytest.mark.parametrize("group_index", range(4))
def test_group_scope_requires_its_unique_loopback_port(
  module, monkeypatch, tmp_path, group_index
):
  evaluation_parent = tmp_path / "evaluations"
  snapshot_parent = tmp_path / "snapshots"
  job_root = evaluation_parent / "pi05_h50_k50_parallel_test"
  job_root.mkdir(parents=True)
  simulation_root = snapshot_parent / "fixed/kaitactilesim"
  openpi_root = snapshot_parent / "fixed/openpi"
  simulation_root.mkdir(parents=True)
  openpi_root.mkdir()
  monkeypatch.setattr(module, "EVALUATION_PARENT", evaluation_parent)
  monkeypatch.setattr(module, "SNAPSHOT_PARENT", snapshot_parent)
  arguments = (job_root, simulation_root, openpi_root, group_index)
  module.validate_scope(*arguments, f"ws://127.0.0.1:{18784 + group_index}")
  with pytest.raises(ValueError, match="requires endpoint"):
    module.validate_scope(*arguments, f"ws://127.0.0.1:{18785 + group_index}")
  with pytest.raises(ValueError, match="requires endpoint"):
    module.validate_scope(*arguments, f"ws://0.0.0.0:{18784 + group_index}")
