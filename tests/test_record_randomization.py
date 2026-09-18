"""Bounded randomized collection wiring; no robot or renderer is instantiated."""

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
from kaihand_tactile_env.shared.recording import TerminalStability, ValidationReport

_PATH = Path(__file__).resolve().parents[1] / "scripts/workcell/record_dataset.py"
_SPEC = importlib.util.spec_from_file_location("record_randomization_script", _PATH)
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
      "middle-force-randomized-v1",
      "--output-dir",
      str(tmp_path),
      "--no-cameras",
      *extra,
    ]
  )


def _job(tmp_path, *extra):
  return record._build_jobs(_args(tmp_path, *extra))[0]


def test_randomized_preset_defaults_preserve_the_middle_physics(tmp_path):
  job = _job(tmp_path)
  assert job.preset == record.RANDOMIZED_PRESET
  assert job.object_xy_jitter == 0.002
  assert job.object_yaw_jitter == 0.0
  assert job.side == "right" and job.object_name == "card"
  assert job.config.tactile_provider == "solver_contact_proxy_v1"
  assert record._press_force_for_job(job) == 0.5
  settings = record._preset_settings_for_job(job)
  assert settings["preset"] == record.RANDOMIZED_PRESET
  assert settings["object_xy_jitter_m"] == 0.002
  assert settings["object_yaw_jitter_rad"] == 0
  assert settings["pressure_window"]["table_friction"] == 1
  assert settings["pressure_window"]["drive_limit_n"] == 4
  assert settings["pressure_window"]["slide_speed_m_s"] == 0.005
  assert settings["robot_initial_state_noise"] is None
  assert settings["observation_noise"] is None
  assert settings["action_noise"] is None
  assert settings["randomization_bounds_are_success_guarantee"] is False


@pytest.mark.parametrize(
  "xy,yaw",
  [
    (0, 0),
    (0.002, 0),
    (0.002, math.radians(1)),
    (0.005, math.radians(1)),
    (0.003, math.radians(0.5)),
  ],
)
def test_randomized_bounds_and_metadata_reflect_requested_values(tmp_path, xy, yaw):
  job = _job(tmp_path, "--object-xy-jitter", str(xy), "--object-yaw-jitter", str(yaw))
  record._validate_middle_job(job)
  settings = record._preset_settings_for_job(job)
  assert settings["object_xy_jitter_m"] == xy
  assert settings["object_yaw_jitter_rad"] == yaw


@pytest.mark.parametrize(
  "options",
  [
    ["--scene", "pick-place"],
    ["--side", "left"],
    ["--workers", "2"],
    ["--object", "cylinder"],
    ["--seed", "-1"],
    ["--seed", "-1", "--start-index", "10"],
    ["--press-force", "0.35"],
    ["--press-force", "nan"],
    ["--object-xy-jitter", "0.005001"],
    ["--object-xy-jitter", "-0.001"],
    ["--object-xy-jitter", "nan"],
    ["--object-yaw-jitter", str(math.radians(1.001))],
    ["--object-yaw-jitter", "-0.001"],
    ["--object-yaw-jitter", "inf"],
    ["--tactile-source", "genesis_probe_bimanual_clean_v1"],
  ],
)
def test_randomized_cli_rejects_unsafe_or_different_physics(tmp_path, options):
  with pytest.raises(SystemExit):
    _args(tmp_path, *options)


@pytest.mark.parametrize(
  "changes",
  [
    {"scene": "pick-place"},
    {"side": "left"},
    {"object_name": "cylinder"},
    {"episode_seed": -1},
    {"press_force_per_finger_n": 0.35},
    {"object_xy_jitter": 0.005001},
    {"object_yaw_jitter": math.radians(1.001)},
    {"object_xy_jitter": float("nan")},
    {"object_yaw_jitter": -0.001},
  ],
)
def test_programmatic_jobs_cannot_bypass_guards(tmp_path, changes):
  with pytest.raises(ValueError):
    record._validate_middle_job(replace(_job(tmp_path), **changes))


def test_negative_seed_fails_before_jobs_or_models_and_names_actual_preset(
  tmp_path,
  capsys,
):
  with pytest.raises(SystemExit):
    _args(tmp_path / "not-created", "--seed", "-1")
  assert not (tmp_path / "not-created").exists()
  assert (
    "middle-force-randomized-v1 requires a non-negative --seed"
    in capsys.readouterr().err
  )


def test_programmatic_negative_seed_fails_before_entering_factory(
  tmp_path, monkeypatch
):
  import kaihand_tactile_env.tasks.poker_draw.mid_full as mid

  def forbidden_factory(*_):
    pytest.fail("negative seeds must fail before model initialization")

  monkeypatch.setattr(mid, "middle_force_simulation", forbidden_factory)
  job = replace(_job(tmp_path), episode_seed=-1)
  with pytest.raises(ValueError, match="middle-force-randomized-v1.*seed"):
    with record._simulation_for_job(job):
      pytest.fail("invalid job entered model context")


def test_different_episode_indices_get_reproducible_distinct_seeds(tmp_path):
  jobs = record._build_jobs(
    _args(
      tmp_path,
      "--episodes",
      "3",
      "--start-index",
      "10",
      "--seed",
      "200",
    )
  )
  assert [job.episode_index for job in jobs] == [10, 11, 12]
  assert [job.episode_seed for job in jobs] == [210, 211, 212]
  assert len({job.output for job in jobs}) == 3


def test_new_module_fingerprint_is_randomized_only(tmp_path):
  job = _job(tmp_path)
  randomized = record._middle_source_hashes(job.preset)
  original = record._middle_source_hashes(record.MIDDLE_FORCE_PRESET)
  assert set(randomized) == set(original) | {"poker_draw/randomization.py"}
  assert all(randomized[name] == digest for name, digest in original.items())
  assert len(randomized["poker_draw/randomization.py"]) == 64
  control = record._press_control_for_job(job)
  assert control["preset"] == job.preset
  assert control["controller_source_sha256"] == randomized


def test_same_factory_and_executor_are_used_for_both_middle_presets(
  tmp_path, monkeypatch
):
  import kaihand_tactile_env.tasks.poker_draw.mid_full as mid

  job = _job(tmp_path)
  simulation, contact = object(), object()
  events = []

  @contextmanager
  def factory(path):
    assert path == job.config.model_path
    events.append("enter")
    yield simulation, contact
    events.append("exit")

  monkeypatch.setattr(mid, "middle_force_simulation", factory)
  with record._simulation_for_job(job) as configured:
    assert configured == (simulation, contact)
    assert events == ["enter"]
  assert events == ["enter", "exit"]
  calls = []
  monkeypatch.setattr(
    record,
    "MidForcePokerExecutor",
    lambda *args, **kwargs: calls.append((args, kwargs)),
  )
  observer = object()
  record._poker_executor_for_job(job, simulation, observer)
  assert calls == [((simulation,), {"observer": observer})]


def test_randomized_recording_termination_guard_restores_sigterm(tmp_path, monkeypatch):
  previous = object()
  handlers = []

  def install(signum, handler):
    assert signum == record.signal.SIGTERM
    handlers.append(handler)
    return previous

  monkeypatch.setattr(record.signal, "signal", install)
  with pytest.raises(KeyboardInterrupt, match="partial is not a successful"):
    with record._recording_termination_guard(_job(tmp_path)):
      handlers[0](record.signal.SIGTERM, None)
  assert handlers[-1] is previous


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


@pytest.mark.parametrize("failed_gate", [None, "task", "slide", "handoff"])
def test_randomized_reset_metadata_and_full_task_gates(
  tmp_path, monkeypatch, failed_gate
):
  job = _job(tmp_path, "--object-yaw-jitter", str(math.radians(1)))
  pose = np.array([0.5, 0, 1, 1, 0, 0, 0.0])
  reset_calls = []
  sample_metadata = {"seed": job.episode_seed, "sampled_pose": pose.tolist()}
  simulation = SimpleNamespace(
    reset=lambda **_: pytest.fail("randomized path must use audited reset"),
    object_twist=lambda _: np.zeros(6),
  )

  def randomized_reset(sim, **kwargs):
    assert sim is simulation
    reset_calls.append(kwargs)
    return sample_metadata

  monkeypatch.setattr(record, "reset_randomized_card", randomized_reset)
  result = _Result(
    failed_gate != "task", "card", ("inspect",), pose, pose, pose, pose, pose[:3]
  )
  executor = SimpleNamespace(
    execute=lambda plan: result,
    refresh_terminal_result=lambda result: result,
    edge_outcome={
      "full_slide_qualified": failed_gate != "slide",
      "target_reached": True,
      "held_at_edge": True,
    },
    handoff_outcome={"completed": failed_gate != "handoff"},
    control_metadata=lambda: {"preset": "middle_force_full_task_v1"},
  )
  saved = {}

  class Recorder:
    def __init__(self, *args, **kwargs):
      saved["metadata"] = kwargs["metadata"]
      assert kwargs["capture_taskspace"] is True

    def __enter__(self):
      return self

    def __exit__(self, *_):
      pass

    def record_initial(self):
      pass

    def record_terminal(self):
      pass

    def observe(self, *_):
      pass

    def set_outcome(self, outcome):
      saved["outcome"] = outcome

  monkeypatch.setattr(record, "EpisodeRecorder", Recorder)
  monkeypatch.setattr(
    record, "PokerDrawPlanner", lambda sim: SimpleNamespace(plan=lambda side: "plan")
  )
  monkeypatch.setattr(record, "_poker_executor_for_job", lambda *args: executor)
  monkeypatch.setattr(
    record,
    "wait_until_object_stable",
    lambda *a, **k: TerminalStability(0.1, 0.1, 0, 0, 50),
  )
  monkeypatch.setattr(
    record, "validate_episode", lambda output: ValidationReport(True, (), (), 10, {}, 1)
  )
  contact = {"contact_friction_impedance_ratio_used": 100}
  if failed_gate:
    with pytest.raises(RuntimeError):
      record._record_simulation_episode(job, simulation, contact)
    assert "outcome" not in saved
  else:
    assert record._record_simulation_episode(job, simulation, contact).success
    assert saved["outcome"]["preset"] == job.preset
    assert saved["outcome"]["edge_outcome"]["full_slide_qualified"]
    assert saved["outcome"]["handoff_outcome"]["completed"]
  assert saved["metadata"]["initial_card_randomization"] is sample_metadata
  assert reset_calls == [
    {"seed": job.episode_seed, "xy_jitter_m": 0.002, "yaw_jitter_rad": math.radians(1)}
  ]


def _write_resume_metadata(job):
  from kaihand_tactile_env.tasks.poker_draw.randomization import (
    RANDOMIZATION_SCHEMA,
    offset_card_pose,
    sample_card_offsets,
  )

  xy, yaw = sample_card_offsets(
    job.episode_seed,
    job.object_xy_jitter,
    job.object_yaw_jitter,
  )
  nominal = [0.59, -0.1, 0.8, 1, 0, 0, 0]
  sampled = offset_card_pose(nominal, xy, yaw)
  randomization = {
    "schema_version": RANDOMIZATION_SCHEMA,
    "mode": "seeded_uniform",
    "distribution": "independent_uniform",
    "seed": job.episode_seed,
    "xy_jitter_m": job.object_xy_jitter,
    "yaw_jitter_rad": job.object_yaw_jitter,
    "nominal_pose_wxyz": nominal,
    "sampled_pose_wxyz": sampled.tolist(),
    "sampled_offset_xy_m": xy.tolist(),
    "sampled_yaw_offset_rad": yaw,
    "perturbation_scope": "reset_only_card_xy_yaw",
    "height_unchanged": True,
    "tilt_unchanged": True,
    "observation_noise": None,
    "action_noise": None,
    "table_support": {
      "valid": True,
      "minimum_corner_margin_xy_m": 0.005,
      "bottom_gap_m": 0.0001,
    },
  }
  metadata = {
    "recording_contract": record.POKER_RECORDING_CONTRACT_VERSION,
    "episode_index": job.episode_index,
    "seed": job.episode_seed,
    "scene": job.scene,
    "object": job.object_name,
    "side": job.side,
    "object_xy_jitter": job.object_xy_jitter,
    "object_yaw_jitter": job.object_yaw_jitter,
    "model_layout": "task-isolated-v1",
    "active_objects": ["card"],
    "press_force_per_finger_n": record._press_force_for_job(job),
    "press_control": record._press_control_for_job(job),
    "preset": job.preset,
    "preset_settings": record._preset_settings_for_job(job),
    "initial_card_randomization": randomization,
  }
  digest, fingerprint = record._model_identity_for_job(job)
  with h5py.File(job.output, "w") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    for name in ("physics_hz", "control_hz", "camera_hz"):
      file.attrs[name] = getattr(job.config, name)
    file.attrs["tactile_source"] = job.config.tactile_provider
    file.attrs["model_sha256"] = digest
    file.attrs["model_fingerprint"] = fingerprint
    file.create_group("tactile_contact_force")
    file.create_dataset("state/timestamp", data=[0.0])
    file.create_dataset("objects/card/pose_wxyz", data=sampled[None, :])
  return metadata


def test_resume_requires_matching_deterministic_sample_and_raw_pose(tmp_path):
  job = _job(tmp_path, "--object-yaw-jitter", str(math.radians(1)))
  _write_resume_metadata(job)
  assert record._episode_matches_job(job.output, job)
  # Quaternion sign represents the same rotation.
  with h5py.File(job.output, "r+") as file:
    file["objects/card/pose_wxyz"][0, 3:] *= -1
  assert record._episode_matches_job(job.output, job)


@pytest.mark.parametrize(
  "mutation",
  [
    "preset",
    "jitter",
    "seed",
    "sampled_offset",
    "sampled_pose",
    "fixed_mode",
    "missing_randomization",
    "missing_raw_pose",
    "raw_position",
    "raw_rotation",
    "raw_nan",
    "not_reset_time",
    "source_hash",
    "noise",
    "support",
  ],
)
def test_resume_rejects_modified_or_unproven_realized_randomization(tmp_path, mutation):
  job = _job(tmp_path, "--object-yaw-jitter", str(math.radians(1)))
  metadata = _write_resume_metadata(job)
  sampled = metadata["initial_card_randomization"]
  if mutation == "preset":
    metadata["preset"] = record.MIDDLE_FORCE_PRESET
  elif mutation == "jitter":
    metadata["preset_settings"]["object_xy_jitter_m"] = 0.003
  elif mutation == "seed":
    sampled["seed"] += 1
  elif mutation == "sampled_offset":
    sampled["sampled_offset_xy_m"][0] += 0.0001
  elif mutation == "sampled_pose":
    sampled["sampled_pose_wxyz"][0] += 0.0001
  elif mutation == "fixed_mode":
    sampled["mode"] = "fixed_validation_offset"
  elif mutation == "missing_randomization":
    del metadata["initial_card_randomization"]
  elif mutation == "source_hash":
    metadata["preset_settings"]["controller_source_sha256"][
      "poker_draw/randomization.py"
    ] = "changed"
  elif mutation == "noise":
    sampled["observation_noise"] = "gaussian"
  elif mutation == "support":
    sampled["table_support"]["valid"] = False
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    if mutation == "missing_raw_pose":
      del file["objects/card/pose_wxyz"]
    elif mutation == "raw_position":
      file["objects/card/pose_wxyz"][0, 0] += 0.0001
    elif mutation == "raw_rotation":
      file["objects/card/pose_wxyz"][0, 4] += 0.0001
    elif mutation == "raw_nan":
      file["objects/card/pose_wxyz"][0, 0] = np.nan
    elif mutation == "not_reset_time":
      file["state/timestamp"][0] = 0.1
  assert not record._episode_matches_job(job.output, job)


def test_resume_does_not_reuse_a_different_seed_or_range(tmp_path):
  job = _job(tmp_path)
  _write_resume_metadata(job)
  for changes in (
    {"episode_seed": job.episode_seed + 1},
    {"object_xy_jitter": 0.003},
    {"object_yaw_jitter": math.radians(1)},
    {"preset": record.MIDDLE_FORCE_PRESET},
  ):
    assert not record._episode_matches_job(job.output, replace(job, **changes))
