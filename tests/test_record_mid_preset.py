"""Middle-force collection wiring checks without loading the robot or rendering."""

from __future__ import annotations

import importlib.util
import json
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
_SPEC = importlib.util.spec_from_file_location("record_mid_preset_script", _PATH)
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
      "middle-force-v1",
      "--output-dir",
      str(tmp_path),
      "--no-cameras",
      *extra,
    ]
  )


def _job(tmp_path, *extra):
  return record._build_jobs(_args(tmp_path, *extra))[0]


def test_middle_preset_resolves_to_accepted_zero_perturbation_settings(tmp_path):
  job = _job(tmp_path)
  assert job.preset == "middle-force-v1"
  assert job.press_force_per_finger_n == 0.50
  assert job.object_xy_jitter == job.object_yaw_jitter == 0
  assert job.side == "right"
  assert job.config.tactile_provider == "solver_contact_proxy_v1"
  settings = record._preset_settings_for_job(job)
  assert settings["pressure_window"]["table_friction"] == 1.0
  assert settings["pressure_window"]["drive_limit_n"] == 4.0
  assert settings["pressure_window"]["contact_time_constant_s"] == 0.010
  assert settings["pressure_window"]["contact_friction_impedance_ratio"] == 100.0
  assert settings["observation_noise"] is settings["action_noise"] is None
  assert all(
    len(digest) == 64 for digest in settings["controller_source_sha256"].values()
  )


@pytest.mark.parametrize("scene", ["pick-place", "poker-draw"])
def test_production_defaults_are_unchanged(scene):
  args = record._parse_args(["--scene", scene])
  assert args.preset == "production"
  assert args.tactile_source == "genesis_probe_bimanual_clean_v1"
  if scene == "poker-draw":
    assert args.press_force_per_finger_n == 0.35


def test_camera_subset_rgb_only_is_explicit_and_defaults_use_training_contract(tmp_path):
  default = record._parse_args(["--output-dir", str(tmp_path / "default")])
  default_job = record._build_jobs(default)[0]
  assert [camera.name for camera in default_job.config.cameras] == [
    "head",
    "left_wrist",
    "right_wrist",
  ]
  assert all(
    camera.rgb and camera.depth and camera.segmentation
    for camera in default_job.config.cameras
  )
  args = record._parse_args(
    [
      "--scene",
      "poker-draw",
      "--preset",
      "middle-force-v1",
      "--output-dir",
      str(tmp_path / "middle"),
      "--cameras",
      "head",
      "--rgb-only",
      "--camera-hz",
      "10",
    ]
  )
  job = record._build_jobs(args)[0]
  assert len(job.config.cameras) == 1
  camera = job.config.cameras[0]
  assert (camera.name, camera.width, camera.height) == ("head", 320, 240)
  assert camera.rgb and not camera.depth and not camera.segmentation
  assert job.config.camera_hz == 10


@pytest.mark.parametrize("wrist", ["right_wrist", "left_wrist"])
def test_wrist_rgb_is_an_explicit_second_recorded_camera(tmp_path, wrist):
  args = record._parse_args([
    "--scene", "poker-draw", "--preset", "middle-force-precontact-v1",
    "--output-dir", str(tmp_path), "--cameras", "head", wrist,
    "--rgb-only", "--camera-hz", "10", "--workers", "1",
  ])
  job = record._build_jobs(args)[0]
  assert [camera.name for camera in job.config.cameras] == ["head", wrist]
  assert all(camera.rgb and not camera.depth and not camera.segmentation
             for camera in job.config.cameras)


def test_duplicate_camera_names_are_rejected():
  with pytest.raises(SystemExit):
    record._parse_args(["--cameras", "head", "head"])


@pytest.mark.parametrize(
  "options",
  [
    ["--scene", "pick-place"],
    ["--side", "left"],
    ["--workers", "2"],
    ["--press-force", "0.35"],
    ["--press-force", "nan"],
    ["--object-xy-jitter", "0.001"],
    ["--object-yaw-jitter", "0.001"],
    ["--object-xy-jitter", "nan"],
    ["--object-yaw-jitter", "inf"],
    ["--tactile-source", "genesis_probe_bimanual_clean_v1"],
  ],
)
def test_middle_rejects_unvalidated_overrides(tmp_path, options):
  with pytest.raises(SystemExit):
    _args(tmp_path, *options)


@pytest.mark.parametrize(
  "changes",
  [
    {"scene": "pick-place"},
    {"side": "left"},
    {"object_name": "cylinder"},
    {"press_force_per_finger_n": 0.35},
    {"object_xy_jitter": 0.001},
    {"object_yaw_jitter": float("nan")},
  ],
)
def test_programmatic_middle_jobs_cannot_bypass_guards(tmp_path, changes):
  with pytest.raises(ValueError):
    record._validate_middle_job(replace(_job(tmp_path), **changes))


def test_mid_factory_context_covers_entire_recording(tmp_path, monkeypatch):
  import kaihand_tactile_env.tasks.poker_draw.mid_full as mid

  job = _job(tmp_path)
  events = []
  simulation = object()
  contact = {"table_card_pair_friction": [1.0, 1.0, 0.005, 0.0005, 0.0005]}

  @contextmanager
  def factory(path):
    assert path == job.config.model_path
    events.append("enter")
    try:
      yield simulation, contact
    finally:
      events.append("exit")

  def run(actual_job, actual_sim, actual_contact):
    assert actual_job == job and actual_sim is simulation
    assert actual_contact == contact
    assert events == ["enter"]
    events.append("record")
    return "done"

  monkeypatch.setattr(mid, "middle_force_simulation", factory, raising=False)
  monkeypatch.setattr(record, "_record_simulation_episode", run)
  assert record._record_claimed_episode(job) == "done"
  assert events == ["enter", "record", "exit"]


def test_executor_selection_does_not_feed_preset_a_custom_force(tmp_path, monkeypatch):
  calls = []
  monkeypatch.setattr(
    record,
    "MidForcePokerExecutor",
    lambda *args, **kwargs: calls.append((args, kwargs)),
  )
  simulation, observer = object(), object()
  record._poker_executor_for_job(_job(tmp_path), simulation, observer)
  assert calls == [((simulation,), {
    "observer": observer, "acceptance_policy": "strict-force-v1",
  })]


def _write_resume_metadata(job):
  metadata = {
    "recording_contract": record.POKER_RECORDING_CONTRACT_VERSION,
    "episode_index": job.episode_index,
    "seed": job.episode_seed,
    "scene": job.scene,
    "object": job.object_name,
    "side": job.side,
    "object_xy_jitter": 0,
    "object_yaw_jitter": 0,
    "model_layout": "task-isolated-v1",
    "active_objects": ["card"],
    "press_force_per_finger_n": record._press_force_for_job(job),
    "press_control": record._press_control_for_job(job),
    "preset": job.preset,
    "preset_settings": record._preset_settings_for_job(job),
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
  return metadata


def test_resume_accepts_same_wrapper_identity_without_loading_model(tmp_path):
  job = _job(tmp_path)
  _write_resume_metadata(job)
  assert record._episode_matches_job(job.output, job)
  assert record._model_identity_for_job(job) == record._model_identity_for_job(job)


@pytest.mark.parametrize("mutation", ["preset", "settings", "hash"])
def test_resume_rejects_changed_preset_settings_and_source(tmp_path, mutation):
  job = _job(tmp_path)
  metadata = _write_resume_metadata(job)
  if mutation == "preset":
    metadata["preset"] = "production"
  elif mutation == "settings":
    metadata["preset_settings"]["pressure_window"]["drive_limit_n"] = 5
  else:
    metadata["preset_settings"]["controller_source_sha256"][
      "poker_draw/mid_full.py"
    ] = "changed"
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
  assert not record._episode_matches_job(job.output, job)


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


@pytest.mark.parametrize("qualified", [True, False])
def test_full_record_path_persists_handoff_and_rejects_unqualified_slide(
  tmp_path, monkeypatch, qualified
):
  job = _job(tmp_path)
  pose = np.array([0.5, 0, 1.0, 1, 0, 0, 0.0])
  simulation = SimpleNamespace(
    reset=lambda **_: None,
    object_twist=lambda _: np.zeros(6),
  )
  result = _Result(True, "card", ("inspect",), pose, pose, pose, pose, pose[:3])
  executor = SimpleNamespace(
    execute=lambda plan: result,
    refresh_terminal_result=lambda result: result,
    edge_outcome={
      "full_slide_qualified": qualified,
      "target_reached": True,
      "held_at_edge": True,
    },
    handoff_outcome={"completed": True, "stage": "joint_servo_ready"},
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
    lambda *args, **kwargs: TerminalStability(0.1, 0.1, 0, 0, 50),
  )
  monkeypatch.setattr(
    record,
    "validate_episode",
    lambda output: ValidationReport(True, (), (), 10, {}, 1.0),
  )
  contact = {"contact_friction_impedance_ratio_used": 100}
  if not qualified:
    with pytest.raises(RuntimeError, match="qualified slide"):
      record._record_simulation_episode(job, simulation, contact)
    assert saved["outcome"]["success"] is False
    assert saved["outcome"]["edge_outcome"]["full_slide_qualified"] is False
    assert saved["outcome"]["task_failure"]["error_type"] == "RuntimeError"
    return
  summary = record._record_simulation_episode(job, simulation, contact)
  assert summary.success
  assert saved["metadata"]["preset_settings"]["pressure_window"]["drive_limit_n"] == 4
  assert saved["metadata"]["contact_model"] == contact
  assert saved["outcome"]["preset"] == "middle-force-v1"
  assert saved["outcome"]["handoff_outcome"]["completed"]
  assert saved["outcome"]["edge_outcome"]["full_slide_qualified"]
  assert saved["outcome"]["phases"][-1] == "terminal_settle"


@pytest.mark.parametrize("termination", ["normal", "sigterm", "other_error"])
def test_middle_termination_guard_restores_previous_handler(
  tmp_path, monkeypatch, termination
):
  previous = object()
  calls = []

  def install(signum, handler):
    assert signum == record.signal.SIGTERM
    calls.append(handler)
    return previous

  monkeypatch.setattr(record.signal, "signal", install)
  job = _job(tmp_path)

  def guarded():
    with record._recording_termination_guard(job):
      assert len(calls) == 1 and callable(calls[0])
      if termination == "sigterm":
        calls[0](record.signal.SIGTERM, None)
      elif termination == "other_error":
        raise RuntimeError("synthetic recording exception")

  if termination == "sigterm":
    with pytest.raises(KeyboardInterrupt, match="partial is not a successful"):
      guarded()
  elif termination == "other_error":
    with pytest.raises(RuntimeError, match="synthetic"):
      guarded()
  else:
    guarded()
  assert calls[-1] is previous
  assert len(calls) == 2


def test_production_recording_does_not_install_sigterm_handler(tmp_path, monkeypatch):
  def forbidden(*_):
    raise AssertionError("production signal behavior must remain unchanged")

  monkeypatch.setattr(record.signal, "signal", forbidden)
  with record._recording_termination_guard(
    replace(_job(tmp_path), preset="production")
  ):
    pass
