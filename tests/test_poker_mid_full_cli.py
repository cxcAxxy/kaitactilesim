"""Zero-noise full-task CLI contract tests; no robot model or renderer allocated."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest


def _script():
  return run_path(
    str(Path(__file__).parents[1] / "scripts/workcell/validate_poker_mid_full.py")
  )


def test_defaults_are_one_fixed_middle_condition_without_recording(tmp_path):
  script = _script()
  target = tmp_path / "new"
  args = script["_parse_args"](["--output-dir", str(target)])
  assert not args.record
  assert args.record_fps == 5.0
  assert args.output_dir == target
  assert not target.exists()
  settings = script["MID_FORCE_SETTINGS"]
  assert script["MID_FORCE_PER_FINGER_N"] == 0.5
  assert settings.table_friction == 1.0
  assert settings.drive_limit_n == 4.0
  assert settings.slide_speed_m_s == 0.005
  assert settings.goal == "table-edge"


@pytest.mark.parametrize(
  "options",
  [
    ["--record-fps", "0"],
    ["--record-fps", "nan"],
    ["--record-fps", "inf"],
    ["--record-fps", "10.1"],
    ["--seed", "1"],
    ["--press-force", "0.2"],
    ["--table-friction", "0.8"],
    ["--object-xy-jitter", "0.001"],
    ["--repeat", "2"],
    ["--viewer"],
  ],
)
def test_invalid_or_non_acceptance_options_are_rejected_before_allocation(
  tmp_path, options
):
  target = tmp_path / "new"
  with pytest.raises(SystemExit) as caught:
    _script()["_parse_args"](["--output-dir", str(target), *options])
  assert caught.value.code == 2
  assert not target.exists()


@pytest.mark.parametrize("task_success", [False, True])
@pytest.mark.parametrize("edge_reached", [False, True])
@pytest.mark.parametrize("edge_held", [False, True])
@pytest.mark.parametrize("edge_qualified", [False, True])
@pytest.mark.parametrize("handoff_completed", [False, True])
@pytest.mark.parametrize("error", [None, "failed controller"])
def test_acceptance_requires_true_full_task_and_all_edge_checks(
  task_success, edge_reached, edge_held, edge_qualified, handoff_completed, error
):
  accepted = _script()["_accepted"](
    {"success": task_success},
    {
      "target_reached": edge_reached,
      "held_at_edge": edge_held,
      "full_slide_qualified": edge_qualified,
    },
    {"completed": handoff_completed},
    error,
  )
  assert accepted is bool(
    task_success
    and edge_reached
    and edge_held
    and edge_qualified
    and handoff_completed
    and error is None
  )


@pytest.mark.parametrize(
  "result,edge", [(None, None), ({}, {}), ({"success": True}, None)]
)
def test_partial_or_missing_outcome_never_passes(result, edge):
  assert not _script()["_accepted"](result, edge, {"completed": True}, None)


def test_missing_handoff_does_not_pass_even_if_task_and_edge_claim_success():
  edge = {
    "target_reached": True,
    "held_at_edge": True,
    "full_slide_qualified": True,
  }
  assert not _script()["_accepted"]({"success": True}, edge, None, None)


def test_existing_directory_is_rejected_before_model_work(tmp_path, monkeypatch):
  script = _script()

  def forbidden(*_args, **_kwargs):
    pytest.fail("existing directory must fail before model work")

  monkeypatch.setitem(script["main"].__globals__, "default_model_path", forbidden)
  with pytest.raises(FileExistsError):
    script["main"](["--output-dir", str(tmp_path)])
  assert not list(tmp_path.iterdir())


@dataclass
class _Result:
  success: bool
  terminal_pinch: bool = True
  retained_at_end: bool = True


def _harness(
  monkeypatch,
  *,
  task_success=True,
  edge_qualified=True,
  control_error=None,
  recording_failure=None,
):
  script = _script()
  global_values = script["main"].__globals__
  events = []
  simulations = []
  videos = []
  resets = []

  class Simulation:
    def __init__(self, path, **kwargs):
      assert path == "fake-scene"
      assert kwargs == {"scene": "poker-draw", "add_genesis_probes": False}
      self.model = SimpleNamespace(
        pair=lambda name: SimpleNamespace(id=0),
        geom=lambda name: SimpleNamespace(id=0),
        pair_solimp=np.asarray([[0.98, 0.995, 0.0005, 0.5, 2.0]]),
        geom_solimp=np.asarray([[0.98, 0.995, 0.0005, 0.5, 2.0]]),
      )
      self.data = SimpleNamespace(time=0.0)
      simulations.append(self)

    def reset(self, **kwargs):
      resets.append(kwargs)

    def drive_state(self):
      return {"drive_actual_fx_n": 0.0}

  class Telemetry:
    def __init__(self, simulation, **kwargs):
      assert kwargs == {"target_force_n": 0.5}

    def summary(self):
      return {"measured": True}

    def metadata(self):
      return {"force_unit": "N"}

  class Observer:
    def __init__(self, simulation, telemetry, stream, video):
      self.video = video
      self.stream = stream
      self.last_time_s = 0.0
      self.last_phase = None
      self.phases = []
      stream.write("time,phase\n")

    def __call__(self, simulation, phase):
      self.last_time_s = simulation.data.time
      self.last_phase = phase
      self.phases.append(phase)
      self.stream.write(f"{simulation.data.time},{phase}\n")
      if self.video is not None:
        try:
          self.video.observe(simulation, phase)
        except Exception as error:
          raise script["PressureWindowVideoError"](str(error)) from error

    def summary(self):
      return {"phases": list(self.phases)}

  class Planner:
    def __init__(self, simulation):
      self.simulation = simulation

    def plan(self):
      return "plan"

  class Executor:
    def __init__(self, simulation, observer):
      self.simulation = simulation
      self.observer = observer
      self.edge_outcome = {
        "target_reached": True,
        "held_at_edge": True,
        "full_slide_qualified": edge_qualified,
      }
      self.handoff_outcome = {"completed": True, "measured_supported_handoff": True}
      self.lift_compensation = []

    def control_metadata(self):
      return {
        "object_motion_or_slip_used_for_control": True,
        "geometry_usage": "endpoint and supported edge retreat, then measured card/tool lever-arm lift compensation",
        "slip_feedback_used_for_pressure_adjustment": False,
        "integral_gain_rad_n_s": 0.1,
        "lift_waypoint_compensation": self.lift_compensation,
      }

    def execute(self, plan):
      assert plan == "plan"
      # Replacing the list models the real executor's episode-state reset.
      self.lift_compensation = [{"phase": "lift_card", "delta_z_m": 0.003}]
      for time, phase in enumerate(("slide_card", "thumb_face_press", "inspect"), 1):
        self.simulation.data.time = float(time)
        self.observer(self.simulation, phase)
        if control_error is not None:
          # A last physical step without callback must still reach the CSV.
          self.simulation.data.time += 0.002
          raise RuntimeError(control_error)
      self.lift_compensation.append({"phase": "raise_card", "delta_z_m": 0.004})
      return _Result(task_success)

  class Video:
    def __init__(self, output):
      self.output = output
      self.outcome = None
      self.success = None
      self.closed = False
      videos.append(self)

    def observe(self, _simulation, phase):
      events.append(("video_observe", phase))
      if recording_failure == "observe" and phase == "slide_card":
        raise RuntimeError("frame encoding failed")

    def set_outcome(self, row):
      self.outcome = row.copy()

    def finish(self, *, success, error):
      events.append(("video_finish", success, error))
      measurement = self.output.with_name("trial_001_measurement.json")
      assert measurement.exists(), "numeric evidence must precede encoding"
      document = json.loads(measurement.read_text())
      assert document["result"]["success"] is success
      trace = self.output.with_name("trial_001_timeseries.csv").read_text()
      assert "slide_card" in trace, "CSV must be flushed before encoding"
      self.success = success
      if recording_failure == "finish":
        raise RuntimeError("ffmpeg failed")
      return self.output, self.output.with_suffix(".json")

    def close(self):
      self.closed = True

  def make_video(_sim, output, _args, metadata):
    assert metadata["video_cameras"]["main"] == "fixed_oblique_free_camera"
    if recording_failure == "initialize":
      raise RuntimeError("renderer unavailable")
    return Video(output)

  def wrapper(scene, friction):
    assert scene == "fake-scene"
    assert friction == 1.0
    return nullcontext(scene)

  for name, value in {
    "default_model_path": lambda scene: "fake-scene",
    "model_fingerprint": lambda path: "fingerprint",
    "_source_hashes": lambda: {"mid_full.py": "hash"},
    "model_with_table_card_friction": wrapper,
    "MidForcePokerSimulation": Simulation,
    "_configure_contact_model": lambda sim, settings: {"timestep_s": 0.002},
    "PokerFrictionTelemetry": Telemetry,
    "PokerDrawPlanner": Planner,
    "MidForcePokerExecutor": Executor,
    "_Observer": Observer,
    "_make_video": make_video,
  }.items():
    monkeypatch.setitem(global_values, name, value)
  return SimpleNamespace(
    script=script,
    events=events,
    simulations=simulations,
    videos=videos,
    resets=resets,
  )


@pytest.mark.parametrize("record", [False, True])
def test_one_run_uses_fixed_zero_noise_and_saves_real_full_task_result(
  tmp_path, monkeypatch, record
):
  harness = _harness(monkeypatch)
  target = tmp_path / "output"
  args = ["--output-dir", str(target), *(["--record"] if record else [])]
  assert harness.script["main"](args) == 0
  assert len(harness.simulations) == 1
  assert harness.resets == [
    {"seed": 0, "object_xy_jitter": 0.0, "object_yaw_jitter": 0.0}
  ]
  document = json.loads((target / "summary.json").read_text())
  assert document["scope"] == "middle_force_full_task_zero_perturbation_acceptance"
  assert not document["state_noise"]
  assert not document["action_noise"]
  assert document["thread_environment"] == dict.fromkeys(
    harness.script["_THREAD_VARIABLES"], "1"
  )
  assert document["trial_count"] == 1
  assert document["headless"]
  assert document["record_video"] is record
  assert document["settings"]["table_friction"] == 1.0
  assert (
    document["contact_model"]["contact_model_version"] == "poker-compliant-contact-v2"
  )
  assert document["contact_model"]["table_card_pair_solimp_used"][:2] == [0.90, 0.95]
  assert document["contact_model"]["card_geom_solimp_used"][:2] == [0.95, 0.98]
  assert document["video_cameras"]["main_lookat_m"] == [0.49, -0.135, 0.955]
  assert document["video_cameras"]["overhead"] == "fixed:overhead"
  assert len(document["trials"]) == 1
  row = document["trials"][0]
  assert row["success"]
  assert row["classification"] == "full_task_accepted"
  assert row["task_result"]["terminal_pinch"]
  assert row["handoff_outcome"] == {
    "completed": True,
    "measured_supported_handoff": True,
  }
  assert row["last_observed_phase"] == "inspect"
  control = row["experimental_control"]
  assert control["object_motion_or_slip_used_for_control"]
  assert control["slip_feedback_used_for_pressure_adjustment"] is False
  assert "supported edge retreat" in control["geometry_usage"]
  assert "lever-arm lift compensation" in control["geometry_usage"]
  assert control["integral_gain_rad_n_s"] == 0.1
  assert control["lift_waypoint_compensation"] == [
    {"phase": "lift_card", "delta_z_m": 0.003},
    {"phase": "raise_card", "delta_z_m": 0.004},
  ]
  assert row["observed"]["phases"] == ["slide_card", "thumb_face_press", "inspect"]
  assert len(harness.videos) == int(record)
  if record:
    assert harness.videos[0].success
    assert harness.videos[0].closed
    assert harness.videos[0].outcome["experimental_control"] == control
    assert row["video_artifacts"]["mp4"].endswith("trial_001_video.mp4")
  else:
    assert row["video_artifacts"] is None
    assert not harness.events


@pytest.mark.parametrize(
  "options", [{"task_success": False}, {"edge_qualified": False}]
)
def test_protocol_return_or_edge_only_never_gives_false_success_video(
  tmp_path, monkeypatch, options
):
  harness = _harness(monkeypatch, **options)
  target = tmp_path / "output"
  assert harness.script["main"](["--output-dir", str(target), "--record"]) == 1
  row = json.loads((target / "trial_001.json").read_text())["result"]
  assert row["protocol_completed"]
  assert not row["success"]
  assert row["classification"] == "full_task_not_accepted"
  assert not harness.videos[0].success


def test_control_error_preserves_edge_handoff_and_last_unobserved_step(
  tmp_path, monkeypatch
):
  harness = _harness(monkeypatch, control_error="pinch failed")
  target = tmp_path / "output"
  assert harness.script["main"](["--output-dir", str(target), "--record"]) == 1
  row = json.loads((target / "trial_001.json").read_text())["result"]
  assert row["error"] == "pinch failed"
  assert row["recording_error"] is None
  assert row["classification"] == "control_error"
  assert row["task_result"] is None
  assert row["edge_outcome"]["full_slide_qualified"]
  assert row["handoff_outcome"] is not None
  assert row["experimental_control"]["lift_waypoint_compensation"] == [
    {"phase": "lift_card", "delta_z_m": 0.003}
  ]
  assert (
    harness.videos[0].outcome["experimental_control"] == row["experimental_control"]
  )
  assert row["observed"]["phases"] == ["slide_card", "terminal_unobserved"]
  assert not harness.videos[0].success


@pytest.mark.parametrize("stage", ["initialize", "observe", "finish"])
def test_recording_failure_is_separate_from_control_and_preserves_numeric_results(
  tmp_path, monkeypatch, stage
):
  harness = _harness(monkeypatch, recording_failure=stage)
  target = tmp_path / "output"
  assert harness.script["main"](["--output-dir", str(target), "--record"]) == 1
  row = json.loads((target / "summary.json").read_text())["trials"][0]
  assert row["error"] is None
  assert row["recording_error"]
  assert (target / "trial_001_measurement.json").exists()
  assert (target / "trial_001_timeseries.csv").exists()
  if stage == "observe":
    assert not row["success"]
    assert not row["protocol_completed"]
    assert row["classification"] == "recording_interrupted"
  else:
    assert row["success"]
    assert row["classification"] == "full_task_accepted"
  assert all(video.closed for video in harness.videos)


def test_video_factory_uses_solver_provider_without_genesis_or_preview(monkeypatch):
  import kaihand_tactile_env.shared.tactile as tactile
  import kaihand_tactile_env.shared.task_video as task_video

  script = _script()
  captured = {}
  simulation = SimpleNamespace(model=object())
  provider = object()
  monkeypatch.setattr(tactile, "SolverContactTactileProvider", lambda model: provider)

  def recorder(sim, output, **kwargs):
    captured.update(sim=sim, output=output, **kwargs)
    return SimpleNamespace(
      follow_viewer_camera=lambda camera: captured.update(camera=camera),
      close=lambda: captured.update(closed=True),
    )

  monkeypatch.setattr(task_video, "TaskVideoRecorder", recorder)
  result = script["_make_video"](
    simulation, Path("full.mp4"), SimpleNamespace(record_fps=5.0), {"key": "value"}
  )
  assert result is not None
  assert captured["tactile_provider"] is provider
  assert captured["preview"] is False
  assert captured["fps"] == 5.0
  assert captured["sim"] is simulation
  assert captured["metadata"] == {"key": "value"}
  camera = captured["camera"]
  assert isinstance(camera, script["mujoco"].MjvCamera)
  assert camera.type == script["mujoco"].mjtCamera.mjCAMERA_FREE
  np.testing.assert_array_equal(camera.lookat, [0.49, -0.135, 0.955])
  assert camera.distance == 0.70
  assert camera.azimuth == 135.0
  assert camera.elevation == -25.0
  assert "closed" not in captured


def test_video_camera_setup_failure_closes_recorder(monkeypatch):
  import kaihand_tactile_env.shared.tactile as tactile
  import kaihand_tactile_env.shared.task_video as task_video

  script = _script()
  calls = []

  def failed_camera(_camera):
    raise RuntimeError("cannot select camera")

  recorder = SimpleNamespace(
    follow_viewer_camera=failed_camera, close=lambda: calls.append("closed")
  )
  monkeypatch.setattr(tactile, "SolverContactTactileProvider", lambda model: object())
  monkeypatch.setattr(task_video, "TaskVideoRecorder", lambda *args, **kw: recorder)
  with pytest.raises(RuntimeError, match="cannot select camera"):
    script["_make_video"](
      SimpleNamespace(model=object()),
      Path("full.mp4"),
      SimpleNamespace(record_fps=5.0),
      {},
    )
  assert calls == ["closed"]
