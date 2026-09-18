"""Precontact collection wiring using fake simulators and tiny HDF5 files only."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.recording import (
  EpisodeRecorder,
  TerminalStability,
  ValidationReport,
)

_PATH = Path(__file__).resolve().parents[1] / "scripts/workcell/record_dataset.py"
_SPEC = importlib.util.spec_from_file_location("record_precontact_script", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
record = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = record
_SPEC.loader.exec_module(record)


def _args(tmp_path, *extra):
  return record._parse_args(
    [
      "--scene",
      "poker-draw",
      "--preset",
      record.PRECONTACT_PRESET,
      "--output-dir",
      str(tmp_path),
      "--no-cameras",
      *extra,
    ]
  )


def _job(tmp_path, *extra):
  return record._build_jobs(_args(tmp_path, *extra))[0]


def _initial_noise(job):
  return {
    "schema_version": "poker-precontact-arm-noise-v1",
    "settings": record.precontact_noise_settings(job.precontact_noise_std_rad),
    "seed": job.episode_seed,
    "configured": True,
    "contact_latched": False,
    "contact_detected_time_s": None,
    "contact_tactile_time_s": None,
    "stop_reason": None,
    "random_sample_count": 0,
    "physics_step_count": 0,
    "postcontact_random_sample_count": 0,
    "command_handoff_count": 0,
  }


def test_precontact_defaults_and_source_hashes_are_isolated(tmp_path):
  job = _job(tmp_path)
  assert job.object_xy_jitter == 0.004
  assert job.object_yaw_jitter == math.radians(0.5)
  assert job.precontact_noise_std_rad == math.radians(0.03)
  assert job.config.physics_hz == 500
  assert job.config.tactile_provider == "solver_contact_proxy_v1"
  settings = record._preset_settings_for_job(job)
  assert settings["action_noise"] == record.precontact_noise_settings(
    job.precontact_noise_std_rad
  )
  assert settings["randomization_scope"] == (
    "initial card pose and precontact right-arm control targets"
  )
  assert settings["robot_initial_state_noise"] is None
  assert settings["observation_noise"] is None
  hashes = record._middle_source_hashes(job.preset)
  randomized = record._middle_source_hashes(record.RANDOMIZED_PRESET)
  assert "shared/cameras.py" in hashes
  assert "poker_draw/acceptance.py" in hashes
  assert hashes.keys() == randomized.keys() | {"poker_draw/precontact_noise.py"}
  assert all(hashes[name] == digest for name, digest in randomized.items())
  for preset, expected_xy in (
    (record.PRODUCTION_PRESET, 0),
    (record.MIDDLE_FORCE_PRESET, 0),
    (record.RANDOMIZED_PRESET, 0.002),
  ):
    old = _job(tmp_path / preset, "--preset", preset)
    assert old.object_xy_jitter == expected_xy
    assert old.object_yaw_jitter == 0
    assert old.precontact_noise_std_rad is None
    if preset != record.PRODUCTION_PRESET:
      assert record._preset_settings_for_job(old)["action_noise"] is None


@pytest.mark.parametrize("sigma", [0, 0.03, 0.05])
def test_sigma_and_episode_seed_are_explicit_and_pairable(tmp_path, sigma):
  jobs = record._build_jobs(
    _args(
      tmp_path,
      "--episodes",
      "2",
      "--start-index",
      "3",
      "--seed",
      "10",
      "--precontact-noise-std-deg",
      str(sigma),
    )
  )
  assert [job.episode_seed for job in jobs] == [13, 14]
  assert [job.episode_index for job in jobs] == [3, 4]
  for job in jobs:
    assert job.precontact_noise_std_rad == math.radians(sigma)
    record._validate_middle_job(job)
    assert job.object_xy_jitter == 0.004
    assert job.object_yaw_jitter == math.radians(0.5)


@pytest.mark.parametrize(
  "options",
  [
    ["--precontact-noise-std-deg", "-0.001"],
    ["--precontact-noise-std-deg", "0.050001"],
    ["--precontact-noise-std-deg", "nan"],
    ["--precontact-noise-std-deg", "inf"],
    ["--seed", "-1", "--start-index", "10"],
    ["--workers", "2"],
    ["--scene", "pick-place"],
    ["--object", "cylinder"],
    ["--side", "left"],
    ["--press-force", "0.4"],
    ["--object-xy-jitter", "0.005001"],
    ["--object-yaw-jitter", str(math.radians(1.001))],
    ["--tactile-source", "genesis_probe_bimanual_clean_v1"],
  ],
)
def test_invalid_cli_fails_before_outputs_or_model(tmp_path, options):
  output = tmp_path / "not-created"
  with pytest.raises(SystemExit):
    _args(output, *options)
  assert not output.exists()


@pytest.mark.parametrize(
  "preset",
  [
    record.PRODUCTION_PRESET,
    record.MIDDLE_FORCE_PRESET,
    record.RANDOMIZED_PRESET,
  ],
)
def test_other_presets_reject_even_zero_precontact_option(tmp_path, preset):
  with pytest.raises(SystemExit):
    _args(tmp_path, "--preset", preset, "--precontact-noise-std-deg", "0")


@pytest.mark.parametrize(
  "changes",
  [
    {"precontact_noise_std_rad": None},
    {"precontact_noise_std_rad": -0.01},
    {"precontact_noise_std_rad": math.radians(0.051)},
    {"precontact_noise_std_rad": float("nan")},
    {"episode_seed": -1},
    {"episode_seed": True},
    {"episode_seed": 1.5},
    {"scene": "pick-place"},
    {"side": "left"},
    {"object_xy_jitter": 0.005001},
  ],
)
def test_programmatic_guards_run_before_factory(tmp_path, monkeypatch, changes):
  import kaihand_tactile_env.tasks.poker_draw.precontact_noise as noise

  monkeypatch.setattr(
    noise,
    "precontact_force_simulation",
    lambda *_: pytest.fail("invalid jobs must fail before creating a model"),
  )
  with pytest.raises(ValueError):
    with record._simulation_for_job(replace(_job(tmp_path), **changes)):
      pytest.fail("invalid job entered simulation")


def test_only_new_preset_uses_precontact_factory(tmp_path, monkeypatch):
  import kaihand_tactile_env.tasks.poker_draw.precontact_noise as noise

  job = _job(tmp_path)
  events = []
  configured = (object(), object())

  @contextmanager
  def factory(path):
    assert path == job.config.model_path
    events.append("enter")
    yield configured
    events.append("exit")

  monkeypatch.setattr(noise, "precontact_force_simulation", factory)
  with record._simulation_for_job(job) as actual:
    assert actual is configured
  assert events == ["enter", "exit"]


@dataclass(frozen=True)
class _Result:
  success: bool
  object_name: str
  phases: tuple[str, ...]
  initial_card_pose: np.ndarray
  edge_card_pose: np.ndarray
  preinspection_card_pose: np.ndarray
  final_card_pose: np.ndarray
  inspection_target_card_position: np.ndarray


@pytest.mark.parametrize(
  "failure", [None, "execute", "interrupt", "slide", "handoff", "audit"]
)
def test_reset_plan_configure_capture_and_partial_trace_order(
  tmp_path, monkeypatch, failure
):
  job = _job(tmp_path)
  events, saved = [], {}
  pose = np.array([0.5, 0, 1, 1, 0, 0, 0.0])
  noise_metadata = _initial_noise(job)
  trace = {"time_s": np.array([0.002, 0.004])}
  simulation = SimpleNamespace(
    reset=lambda **_: pytest.fail("must use audited randomized reset"),
    object_twist=lambda _: np.zeros(6),
    precontact_noise_metadata=lambda: dict(noise_metadata),
    precontact_noise_trace=lambda: trace,
  )

  def reset(sim, **kwargs):
    assert sim is simulation
    assert kwargs == {
      "seed": job.episode_seed,
      "xy_jitter_m": 0.004,
      "yaw_jitter_rad": math.radians(0.5),
    }
    events.append("reset")
    return {"seed": job.episode_seed, "action_noise": None}

  def plan(side):
    assert side == "right"
    events.append("plan")
    return "plan"

  def configure(**kwargs):
    assert kwargs == {"seed": job.episode_seed, "std_rad": job.precontact_noise_std_rad}
    events.append("configure")

  simulation.configure_precontact_noise = configure
  result = _Result(True, "card", ("inspect",), pose, pose, pose, pose, pose[:3])

  def execute(plan):
    assert plan == "plan"
    events.append("execute")
    noise_metadata.update(physics_step_count=2, contact_latched=True)
    if failure == "execute":
      raise RuntimeError("execution failed")
    if failure == "interrupt":
      raise KeyboardInterrupt("interrupted")
    return result

  executor = SimpleNamespace(
    execute=execute,
    refresh_terminal_result=lambda result: result,
    edge_outcome={
      "full_slide_qualified": failure != "slide",
      "target_reached": True,
      "held_at_edge": True,
    },
    handoff_outcome={"completed": failure != "handoff"},
    control_metadata=lambda: {},
  )

  class Recorder:
    def __init__(self, *args, **kwargs):
      events.append("recorder")
      saved["metadata"] = kwargs["metadata"]
      assert kwargs["capture_taskspace"] is True

    def __enter__(self):
      return self

    def __exit__(self, exception_type, *_):
      saved["exception_type"] = exception_type
      events.append("close")

    def record_initial(self):
      events.append("initial")

    def record_terminal(self):
      events.append("terminal")

    def observe(self, *_):
      pass

    def set_outcome(self, outcome):
      self._outcome = dict(outcome)
      saved["outcome"] = self._outcome

    def write_precontact_noise_trace(self, values, *, metadata):
      assert values is trace
      saved["trace_metadata"] = metadata
      events.append("trace")

  monkeypatch.setattr(record, "reset_randomized_card", reset)
  monkeypatch.setattr(record, "PokerDrawPlanner", lambda _: SimpleNamespace(plan=plan))
  monkeypatch.setattr(record, "EpisodeRecorder", Recorder)
  monkeypatch.setattr(record, "_poker_executor_for_job", lambda *args: executor)
  monkeypatch.setattr(
    record,
    "wait_until_object_stable",
    lambda *a, **k: TerminalStability(0.1, 0.1, 0, 0, 50),
  )
  monkeypatch.setattr(
    record, "validate_episode", lambda _: ValidationReport(True, (), (), 10, {}, 1)
  )

  def audit(_):
    events.append("audit")
    if failure == "audit":
      raise RuntimeError("control archive invalid")

  monkeypatch.setattr(record, "_validate_precontact_archive", audit)
  monkeypatch.setattr(
    record, "_demote_completed_episode", lambda _: saved.update(demoted=True)
  )
  if failure:
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else RuntimeError):
      record._record_simulation_episode(job, simulation, {})
    assert (saved["exception_type"] is not None) == (failure != "audit")
  else:
    assert record._record_simulation_episode(job, simulation, {}).success
    assert saved["exception_type"] is None
  assert events[:6] == ["reset", "plan", "configure", "recorder", "initial", "execute"]
  assert [event for event in events if event != "audit"][-2:] == ["trace", "close"]
  assert ("audit" in events) == (failure in (None, "audit"))
  assert saved.get("demoted", False) == (failure == "audit")
  assert saved["metadata"]["precontact_noise"]["physics_step_count"] == 0
  assert saved["metadata"]["initial_card_randomization"]["action_noise"] is None
  assert saved["outcome"]["precontact_noise"]["physics_step_count"] == 2
  assert saved["trace_metadata"] == saved["outcome"]["precontact_noise"]
  if failure in ("execute", "interrupt", "slide", "handoff"):
    assert saved["outcome"]["success"] is False
    assert saved["outcome"]["edge_outcome"] == executor.edge_outcome
    assert saved["outcome"]["handoff_outcome"] == executor.handoff_outcome
    assert saved["outcome"]["task_failure"]["episode_seed"] == job.episode_seed
    assert saved["outcome"]["task_failure"]["interrupted"] == (failure == "interrupt")


def test_poker_failure_archive_preserves_original_exception_if_disk_write_fails(
  tmp_path,
):
  job = _job(tmp_path)

  def fail(_):
    raise OSError("diagnostic disk full")

  recorder = SimpleNamespace(set_outcome=fail)
  original = KeyboardInterrupt("stop capture")
  with pytest.raises(KeyboardInterrupt) as caught:
    with record._record_poker_failure(job, SimpleNamespace(), recorder, {}):
      raise original
  assert caught.value is original
  assert "diagnostic disk full" in original.__notes__[0]


def test_poker_failure_archive_keeps_cached_and_state_clocks_separate(tmp_path):
  class Recorder:
    # Bind the real replacement API so a merging fake cannot hide lost fields.
    set_outcome = record.EpisodeRecorder.set_outcome

    def __init__(self):
      self._outcome = {"precontact_noise": {"postcontact_random_sample_count": 0}}

  recorder = Recorder()
  simulation = SimpleNamespace(
    data=SimpleNamespace(time=31.02), observation_time=31.018
  )
  edge = {
    "target_reached": True,
    "full_slide_qualified": False,
    "slide_force_quality": {"maximum_contact_gaps_s": [0.026, 0.022, 0.022, 0.02]},
  }
  executor = SimpleNamespace(edge_outcome=edge, handoff_outcome=None)
  with pytest.raises(RuntimeError, match="pickup is prohibited"):
    with record._record_poker_failure(
      _job(tmp_path),
      simulation,
      recorder,
      {"executor": executor},
    ):
      raise RuntimeError("middle-pressure draw did not qualify; pickup is prohibited")
  saved = recorder._outcome
  assert saved["edge_outcome"] == edge
  assert "handoff_outcome" not in saved
  assert saved["task_failure"]["simulation_time_s"] == 31.02
  assert saved["task_failure"]["observation_time_s"] == 31.018
  assert saved["precontact_noise"]["postcontact_random_sample_count"] == 0


def test_failure_and_noise_are_both_preserved_in_partial_outcome_json(tmp_path):
  path = tmp_path / "diagnostic.h5.partial"

  class Recorder:
    set_outcome = record.EpisodeRecorder.set_outcome

    def __init__(self):
      self._outcome = {}

    def write_precontact_noise_trace(self, values, *, metadata):
      assert values == {"time_s": [0.0]}
      assert metadata == {"postcontact_random_sample_count": 0}

  recorder = Recorder()
  simulation = SimpleNamespace(
    precontact_noise_trace=lambda: {"time_s": [0.0]},
    precontact_noise_metadata=lambda: {"postcontact_random_sample_count": 0},
  )
  job = _job(tmp_path)
  executor = SimpleNamespace(edge_outcome={"full_slide_qualified": False})
  with pytest.raises(RuntimeError, match="draw failed"):
    with (
      record._record_poker_failure(job, simulation, recorder, {"executor": executor}),
      record._record_precontact_trace(job, simulation, recorder),
    ):
      raise RuntimeError("draw failed")
  with h5py.File(path, "w") as file:
    file.attrs["outcome_json"] = json.dumps(recorder._outcome)
  with h5py.File(path) as file:
    saved = json.loads(file.attrs["outcome_json"])
  assert saved["precontact_noise"] == {"postcontact_random_sample_count": 0}
  assert saved["edge_outcome"] == {"full_slide_qualified": False}
  assert saved["task_failure"]["error_message"] == "draw failed"
  assert saved["success"] is False


def test_archive_failure_does_not_mask_execution_or_interrupt(tmp_path):
  job = _job(tmp_path)

  def archive_failure(*args, **kwargs):
    raise OSError("disk full")

  recorder = SimpleNamespace(
    write_precontact_noise_trace=archive_failure, set_outcome=lambda _: None
  )
  simulation = SimpleNamespace(
    precontact_noise_trace=lambda: {}, precontact_noise_metadata=lambda: {}
  )
  original = KeyboardInterrupt("task interrupted")
  with pytest.raises(KeyboardInterrupt) as caught:
    with record._record_precontact_trace(job, simulation, recorder):
      raise original
  assert caught.value is original
  assert "disk full" in original.__notes__[0]
  with pytest.raises(OSError, match="disk full"):
    with record._record_precontact_trace(job, simulation, recorder):
      pass


@pytest.mark.parametrize("valid", [True, False])
def test_completed_archive_requires_successful_control_audit(
  tmp_path, monkeypatch, valid
):
  import kaihand_tactile_env.shared.tict_source_audit as audit

  output = tmp_path / "episode.h5"
  with h5py.File(output, "w"):
    pass
  calls = []

  def check(file):
    calls.append(file.filename)
    return {"valid": valid, "errors": [] if valid else ["actual ctrl mismatch"]}

  monkeypatch.setattr(audit, "audit_precontact_noise", check)
  if valid:
    record._validate_precontact_archive(output)
  else:
    with pytest.raises(RuntimeError, match="actual ctrl mismatch"):
      record._validate_precontact_archive(output)
  assert calls == [str(output)]


@pytest.mark.parametrize("count", [0, 501])
def test_trace_writer_preserves_all_physics_rows_in_datasets(tmp_path, count):
  path = tmp_path / "episode.h5.partial"
  trace = {
    "time_s": np.arange(1, count + 1) / 500,
    "contact": np.zeros((count, 5), dtype=bool),
    "actual_ctrl_rad": np.zeros((count, 7)),
  }
  metadata = {"physics_step_count": count}
  recorder = EpisodeRecorder.__new__(EpisodeRecorder)
  recorder._closed = False
  with h5py.File(path, "w") as recorder._file:
    recorder.write_precontact_noise_trace(trace, metadata=metadata)
  with h5py.File(path, "r") as file:
    group = file["control/precontact_noise"]
    assert set(group) == set(trace)
    assert json.loads(group.attrs["metadata_json"]) == metadata
    assert set(group.attrs) == {"metadata_json"}
    for name, values in trace.items():
      np.testing.assert_array_equal(group[name], values)


@pytest.mark.parametrize(
  "trace",
  [
    {},
    {"time_s": 0},
    {"bad/path": np.zeros(1)},
    {"time_s": np.zeros(2), "contact": np.zeros((1, 5))},
    {"time_s": np.array([np.nan])},
    {"time_s": np.array(["bad"])},
  ],
)
def test_trace_writer_rejects_malformed_arrays_before_mutation(tmp_path, trace):
  recorder = EpisodeRecorder.__new__(EpisodeRecorder)
  recorder._closed = False
  with h5py.File(tmp_path / "trace.h5", "w") as recorder._file:
    with pytest.raises(ValueError):
      recorder.write_precontact_noise_trace(trace, metadata={})
    assert "control" not in recorder._file


def _resume_fixture(tmp_path, monkeypatch):
  import kaihand_tactile_env.shared.tict_source_audit as audit
  from test_record_randomization import _write_resume_metadata

  job = _job(tmp_path)
  metadata = _write_resume_metadata(job)
  metadata["precontact_noise"] = _initial_noise(job)
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
  calls = []
  monkeypatch.setattr(
    audit,
    "audit_precontact_noise",
    lambda file, **kwargs: calls.append(file.filename) or {"valid": True},
  )
  return job, metadata, calls


def test_resume_uses_full_noise_audit_and_rejects_changed_sigma(tmp_path, monkeypatch):
  job, _, calls = _resume_fixture(tmp_path, monkeypatch)
  assert record._episode_matches_job(job.output, job)
  assert calls == [str(job.output)]
  assert not record._episode_matches_job(
    job.output, replace(job, precontact_noise_std_rad=0)
  )
  import kaihand_tactile_env.shared.tict_source_audit as audit

  monkeypatch.setattr(audit, "audit_precontact_noise", lambda *a, **k: {"valid": False})
  assert not record._episode_matches_job(job.output, job)


def test_resume_delegates_initial_contact_latch_to_raw_tactile_audit(
  tmp_path, monkeypatch
):
  job, metadata, calls = _resume_fixture(tmp_path, monkeypatch)
  metadata["precontact_noise"].update(
    contact_latched=True,
    contact_detected_time_s=0.0,
    contact_tactile_time_s=0.0,
    stop_reason="first_right_fingertip_contact",
  )
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
  assert record._episode_matches_job(job.output, job)
  assert calls == [str(job.output)]


@pytest.mark.parametrize(
  "mutation",
  [
    "seed",
    "settings",
    "physics_step_count",
    "random_sample_count",
    "configured",
    "missing",
    "raw_pose",
    "reset_seed",
    "first_time",
  ],
)
def test_resume_rejects_unproven_noise_start_or_realized_reset(
  tmp_path, monkeypatch, mutation
):
  job, metadata, calls = _resume_fixture(tmp_path, monkeypatch)
  noise = metadata["precontact_noise"]
  if mutation == "seed":
    noise["seed"] += 1
  elif mutation == "settings":
    noise["settings"] = record.precontact_noise_settings(0)
  elif mutation in ("physics_step_count", "random_sample_count"):
    noise[mutation] = 1
  elif mutation == "configured":
    noise["configured"] = False
  elif mutation == "missing":
    del metadata["precontact_noise"]
  elif mutation == "reset_seed":
    metadata["initial_card_randomization"]["seed"] += 1
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    if mutation == "raw_pose":
      file["objects/card/pose_wxyz"][0, 0] += 0.001
    elif mutation == "first_time":
      file["state/timestamp"][0] = 0.002
  assert not record._episode_matches_job(job.output, job)
  assert not calls
