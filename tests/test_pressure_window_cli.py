"""Pressure-window CLI checks with fake simulations; no robot model or renderer."""

import csv
import io
import json
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest


def _script():
  return run_path(
    str(
      Path(__file__).parents[1] / "scripts/workcell/experiment_poker_pressure_window.py"
    )
  )


def test_candidate_defaults_are_explicit_and_parsing_has_no_side_effects(tmp_path):
  target = tmp_path / "not_created"
  args = _script()["_parse_args"](["--output-dir", str(target)])
  assert args.press_force == [0.15, 0.35, 1.20]
  assert args.record_fps == 5.0
  assert not args.record
  assert args.settings.table_friction == 1.30
  assert args.settings.drive_limit_n == 4.0
  assert args.settings.slide_distance_m == 0.030
  assert args.settings.slide_speed_m_s == 0.005
  assert args.settings.contact_time_constant_s == 0.010
  assert args.settings.contact_friction_impedance_ratio == 100.0
  assert args.settings.finger_servo_velocity_gain is None
  assert args.settings.physics_timestep_s is None
  assert args.settings.goal == "short"
  assert args.settings.max_slide_time_s == 30.0
  assert args.settings.max_slide_travel_m == 0.16
  assert not target.exists()


@pytest.mark.parametrize(
  "options",
  [
    ["--press-force", "nan"],
    ["--press-force", "0"],
    ["--press-force", "-1"],
    ["--press-force", *(["0.2"] * 11)],
    ["--table-friction", "inf"],
    ["--table-friction", "-1"],
    ["--drive-limit", "0"],
    ["--drive-limit", "nan"],
    ["--slide-distance", "0.061"],
    ["--slide-speed", "0.0001"],
    ["--hold-seconds", "0"],
    ["--contact-time-constant", "-0.01"],
    ["--friction-impedance-ratio", "0"],
    ["--friction-impedance-ratio", "-1"],
    ["--friction-impedance-ratio", "nan"],
    ["--friction-impedance-ratio", "inf"],
    ["--finger-servo-gain", "0"],
    ["--finger-servo-gain", "-0.3"],
    ["--finger-servo-gain", "nan"],
    ["--finger-servo-gain", "inf"],
    ["--timestep", "0"],
    ["--timestep", "-0.001"],
    ["--timestep", "nan"],
    ["--timestep", "inf"],
    ["--timestep", "0.00049"],
    ["--timestep", "0.00201"],
    ["--record-fps", "0"],
    ["--record-fps", "nan"],
    ["--record-fps", "10.1"],
    ["--goal", "pickup"],
    ["--max-slide-time", "0"],
    ["--max-slide-time", "nan"],
    ["--max-slide-time", "inf"],
    ["--max-slide-time", "45.01"],
    ["--max-slide-travel", "0"],
    ["--max-slide-travel", "nan"],
    ["--max-slide-travel", "inf"],
    ["--max-slide-travel", "0.2001"],
  ],
)
def test_invalid_options_fail_before_output_creation(tmp_path, options):
  target = tmp_path / "not_created"
  with pytest.raises(SystemExit) as caught:
    _script()["_parse_args"](["--output-dir", str(target), *options])
  assert caught.value.code == 2
  assert not target.exists()


def test_table_edge_goal_has_explicit_finite_bounds(tmp_path):
  target = tmp_path / "not_created"
  args = _script()["_parse_args"](
    [
      "--output-dir",
      str(target),
      "--goal",
      "table-edge",
      "--max-slide-time",
      "24",
      "--max-slide-travel",
      "0.15",
    ]
  )
  assert args.settings.goal == "table-edge"
  assert args.settings.max_slide_time_s == 24.0
  assert args.settings.max_slide_travel_m == 0.15
  assert args.settings.target_overhang_fraction == 0.49
  assert args.settings.edge_dwell_s == 0.10
  assert not target.exists()


@pytest.mark.parametrize(
  ("outcome", "expected_classification", "expected_success"),
  [
    (None, "unclassified", False),
    (
      {"target_reached": False, "full_slide_qualified": False},
      "edge_not_reached",
      False,
    ),
    (
      {"target_reached": False, "full_slide_qualified": True},
      "edge_not_reached",
      False,
    ),
    (
      {"target_reached": True, "full_slide_qualified": False},
      "edge_reached_contact_unqualified",
      False,
    ),
    (
      {"target_reached": True, "full_slide_qualified": True},
      "edge_reached_contact_qualified",
      True,
    ),
  ],
)
def test_edge_classification_and_video_success_require_measured_qualified_completion(
  outcome, expected_classification, expected_success
):
  script = _script()
  assert (
    script["_classification"]("table-edge", outcome, None) == expected_classification
  )
  assert script["_video_success"]("table-edge", outcome, None) is expected_success
  assert (
    script["_classification"]("table-edge", outcome, "safety check") == "control_error"
  )
  assert script["_video_success"]("table-edge", outcome, "safety check") is False
  assert script["_classification"]("short", outcome, None) == "unclassified"
  assert script["_video_success"]("short", outcome, None) is True


def test_existing_output_directory_is_rejected_before_model_allocation(
  tmp_path, monkeypatch
):
  script = _script()
  global_values = script["main"].__globals__

  def forbidden_model(*_args, **_kwargs):
    pytest.fail("must reject an existing directory before allocating a model")

  monkeypatch.setitem(global_values, "model_with_table_card_friction", forbidden_model)
  with pytest.raises(FileExistsError):
    script["main"](["--output-dir", str(tmp_path)])
  assert not list(tmp_path.iterdir())


def test_strict_json_sanitizes_nonfinite_numbers_and_never_overwrites(tmp_path):
  script = _script()
  target = tmp_path / "result.json"
  script["_write_json"](
    target, {"a": float("nan"), "b": [float("inf"), np.float64(0.2)]}
  )
  original = target.read_text()
  assert json.loads(original) == {"a": None, "b": [None, 0.2]}
  with pytest.raises(FileExistsError):
    script["_write_json"](target, {"changed": True})
  assert target.read_text() == original


def test_streaming_statistics_use_duration_weights_and_clip_tail_boundary():
  script = _script()
  statistics = script["_SampleStatistics"]()
  fields = script["_observed_fields"]()
  first = dict.fromkeys(fields, 1.0)
  first.update(time_s=0.2, sample_dt_s=0.2)
  second = dict.fromkeys(fields, 3.0)
  second.update(time_s=0.6, sample_dt_s=0.4)
  statistics.add(first)
  statistics.add(second)
  result = statistics.summary()
  assert result["duration_s"] == pytest.approx(0.6)
  assert result["means"]["drive_actual_fx_n"] == pytest.approx(7 / 3)
  assert result["maximum_absolute"]["drive_actual_fx_n"] == 3
  tail = script["_tail_summary"](deque((first, second)))
  assert tail["duration_s"] == pytest.approx(0.5)
  assert tail["means"]["drive_actual_fx_n"] == pytest.approx(2.6)
  assert script["_tail_summary"](deque())["means"]["card_vx_m_s"] is None


def test_tail_storage_remains_bounded():
  script = _script()
  rows = deque()
  for index in range(2000):
    sample = dict.fromkeys(script["_observed_fields"](), 0.2)
    sample.update(time_s=(index + 1) * 0.002, sample_dt_s=0.002)
    script["_Observer"]._append_tail(rows, sample)
  assert len(rows) <= 251
  result = script["_tail_summary"](rows)
  assert result["duration_s"] == pytest.approx(0.5)
  assert all(value == pytest.approx(0.2) for value in result["means"].values())


def _simulation():
  ids = {
    f"hand_r_{finger}_link4_tactile_pad_col": index
    for index, finger in enumerate(("index", "middle", "ring", "pinky"))
  }
  ids["card_core_geom"] = 4
  actuator_names = [
    f"hand_{side}_{finger}_joint{joint}"
    for side in ("r", "l")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in (1, 2, 3, 4)
  ] + [f"arm_{side}_joint{joint}" for side in ("r", "l") for joint in range(7)]
  actuator_ids = {name: index for index, name in enumerate(actuator_names)}
  gain = np.tile(np.arange(10, dtype=float), (len(actuator_names), 1))
  bias = gain.copy()
  gain[:, 0] = 0.9 + 0.1 * np.arange(len(actuator_names))
  bias[:, 2] = -gain[:, 0]
  model = SimpleNamespace(
    opt=SimpleNamespace(impratio=1.0, timestep=0.002),
    geom=lambda name: SimpleNamespace(id=ids[name]),
    pair=lambda _name: SimpleNamespace(id=0),
    geom_bodyid=np.arange(5),
    site_bodyid=np.array([0]),
    body_rootid=np.arange(5),
    pair_solref=np.array([[0.002, 1.0]]),
    geom_solref=np.tile([0.002, 1.0], (5, 1)),
    pair_friction=np.array([[1.25, 1.25, 0.005, 0.0005, 0.0005]]),
    geom_friction=np.tile([1.4, 0.02, 0.002], (5, 1)),
    geom_priority=np.array([0, 0, 0, 0, 1]),
    actuator=lambda name: SimpleNamespace(id=actuator_ids[name]),
    actuator_gainprm=gain,
    actuator_biasprm=bias,
    actuator_forcerange=np.tile([-50.0, 50.0], (len(actuator_names), 1)),
    actuator_ctrlrange=np.tile([-100.0, 100.0], (len(actuator_names), 1)),
  )
  data = SimpleNamespace(
    time=0.0,
    geom_xpos=np.tile([0.6, 0, 0], (5, 1)),
    site_xpos=np.array([[0.62, 0, 0]]),
    cvel=np.tile([0, 0, 0, -0.01, 0, 0], (5, 1)),
    subtree_com=np.zeros((5, 3)),
  )
  sim = SimpleNamespace(
    model=model, data=data, _site_id={"right": 0}, drive_limit_n=None, timestep=0.002
  )
  sim.drive_state = lambda: {
    "drive_requested_fx_n": -5.0,
    "drive_limited_fx_n": -1.6,
    "drive_actual_fx_n": -1.6,
    "drive_saturated": 1.0,
    "drive_cap_error_n": 0.0,
    "drive_jacobian_condition": 5.0,
  }
  return sim


class _Telemetry:
  def __init__(self, simulation, *, target_force_n=0.2):
    self.simulation = simulation
    self.target = target_force_n
    self.phases = []

  def sample(self, phase):
    self.phases.append(phase)
    row = {
      "time_s": self.simulation.data.time,
      "solver_time_s": self.simulation.data.time - 0.002,
      "sample_dt_s": 0.002,
      "phase": phase,
      "card_x_m": float(self.simulation.data.geom_xpos[4, 0]),
      "card_vx_m_s": -0.01,
    }
    for finger in ("index", "middle", "ring", "pinky"):
      row.update(
        {
          f"{finger}_fn_n": self.target,
          f"{finger}_ft_n": 0.1,
          f"{finger}_contact": 1,
          f"{finger}_pressure_qualified": 1,
        }
      )
    return row

  def summary(self):
    return {"phase_sequence": self.phases}

  def metadata(self):
    return {"source": "fake", "target_force_per_finger_n": self.target}


def _advance(simulation):
  simulation.data.time += 0.002
  simulation.data.geom_xpos[:, 0] -= 0.00002
  simulation.data.site_xpos[:, 0] -= 0.00002


@pytest.mark.parametrize("hold_phase", ["edge_hold", "final_hold"])
def test_observer_streams_all_phases_and_preserves_terminal_drive(hold_phase):
  script = _script()
  simulation = _simulation()
  stream = io.StringIO(newline="")
  observer = script["_Observer"](simulation, _Telemetry(simulation), stream)
  phases = ["four_finger_press", "slide_card", hold_phase, "terminal_unobserved"]
  for phase in phases:
    if phase == "slide_card":
      simulation.drive_limit_n = 1.6
    _advance(simulation)
    observer(simulation, phase)
  stream.seek(0)
  rows = list(csv.DictReader(stream))
  assert [row["phase"] for row in rows] == phases
  assert float(rows[0]["drive_actual_fx_n"]) == 0.0  # no stale reset state
  assert float(rows[-1]["drive_actual_fx_n"]) == -1.6
  assert rows[-1]["drive_budget_active"] == "1"
  assert float(rows[1]["mean_pad_vx_m_s"]) == -0.01
  # Added wrench/pose diagnostic fields remain rectangular even when an older
  # drive-state implementation omits them; supplied actual Fx stays intact.
  for key in script["_DRIVE_KEYS"]:
    assert key in rows[-1]
  assert float(rows[-1]["drive_requested_fy_n"]) == 0.0
  assert float(rows[-1]["drive_actual_tz_nm"]) == 0.0
  assert float(rows[-1]["drive_pose_error_y_m"]) == 0.0
  result = observer.summary()
  assert result["slide_start"]["time_s"] == pytest.approx(0.002)
  assert result["active_slide_displacement"]["card_toward_robot_m"] == pytest.approx(
    0.00002
  )
  assert result["slide_and_hold_displacement"]["card_toward_robot_m"] == pytest.approx(
    0.00004
  )
  assert result["active_slide"]["sample_count"] == 1
  assert result["slide_and_hold"]["sample_count"] == 2
  assert result["slide_and_hold"]["means"]["four_fingers_loaded"] == 1.0
  assert observer.last_time_s == pytest.approx(0.008)


def test_video_failure_is_not_a_control_error_and_csv_is_preserved():
  script = _script()
  simulation = _simulation()
  stream = io.StringIO(newline="")

  class BrokenVideo:
    def observe(self, _sim, _phase):
      raise RuntimeError("encoder failed")

  observer = script["_Observer"](
    simulation, _Telemetry(simulation), stream, BrokenVideo()
  )
  _advance(simulation)
  with pytest.raises(script["PressureWindowVideoError"], match="encoder failed"):
    observer(simulation, "slide_card")
  assert not issubclass(script["PressureWindowVideoError"], RuntimeError)
  stream.seek(0)
  assert len(list(csv.DictReader(stream))) == 1


def test_soft_contact_override_changes_only_card_and_explicit_pair():
  script = _script()
  simulation = _simulation()
  settings = script["PressureWindowSettings"](
    contact_time_constant_s=0.01, contact_friction_impedance_ratio=1.0
  )
  metadata = script["_configure_contact_model"](simulation, settings)
  assert simulation.model.pair_solref[0].tolist() == [0.01, 1.0]
  assert simulation.model.geom_solref[4].tolist() == [0.01, 1.0]
  assert np.all(simulation.model.geom_solref[:4, 0] == 0.002)
  assert metadata["card_geom_solref_original"] == [0.002, 1.0]
  assert metadata["finger_card_nominal_sliding_friction"] == 1.4
  assert "not measured" in metadata["finger_card_friction_source"]


def test_impedance_ratio_override_is_validated_and_changes_only_experiment_model(
  tmp_path,
):
  script = _script()
  args = script["_parse_args"](
    [
      "--output-dir",
      str(tmp_path / "not_created"),
      "--friction-impedance-ratio",
      "10",
    ]
  )
  assert args.settings.contact_friction_impedance_ratio == 10.0
  baseline = _simulation()
  experiment = _simulation()
  old_geom_solref = experiment.model.geom_solref.copy()
  old_pair_solref = experiment.model.pair_solref.copy()
  old_pair_friction = experiment.model.pair_friction.copy()
  old_geom_friction = experiment.model.geom_friction.copy()
  # This test isolates one override; the calibrated CLI preset also changes
  # contact time constants, so explicitly retain the original contact setting.
  settings = script["PressureWindowSettings"](
    contact_time_constant_s=None,
    contact_friction_impedance_ratio=args.settings.contact_friction_impedance_ratio,
  )
  metadata = script["_configure_contact_model"](experiment, settings)
  assert experiment.model.opt.impratio == 10.0
  assert baseline.model.opt.impratio == 1.0
  assert metadata["contact_friction_impedance_ratio_original"] == 1.0
  assert metadata["contact_friction_impedance_ratio_used"] == 10.0
  assert np.array_equal(experiment.model.geom_solref, old_geom_solref)
  assert np.array_equal(experiment.model.pair_solref, old_pair_solref)
  assert np.array_equal(experiment.model.pair_friction, old_pair_friction)
  assert np.array_equal(experiment.model.geom_friction, old_geom_friction)
  assert not (tmp_path / "not_created").exists()


def test_default_finger_servo_gain_leaves_all_actuators_unchanged():
  script = _script()
  simulation = _simulation()
  attributes = (
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrlrange",
  )
  before = {name: getattr(simulation.model, name).copy() for name in attributes}
  metadata = script["_configure_contact_model"](
    simulation, script["PressureWindowSettings"]()
  )
  assert simulation.timestep == 0.002
  assert simulation.model.opt.timestep == 0.002
  assert metadata["physics_timestep_original_s"] == 0.002
  assert metadata["physics_timestep_used_s"] == 0.002
  assert metadata["experimental_right_finger_servo_gains"] == {}
  for name, original in before.items():
    assert np.array_equal(getattr(simulation.model, name), original)


def test_finger_servo_gain_override_targets_exactly_eight_right_non_thumb_actuators(
  tmp_path,
):
  script = _script()
  args = script["_parse_args"](
    [
      "--output-dir",
      str(tmp_path / "not_created"),
      "--finger-servo-gain",
      "0.3",
    ]
  )
  assert args.settings.finger_servo_velocity_gain == 0.3
  baseline = _simulation()
  experiment = _simulation()
  gain_before = experiment.model.actuator_gainprm.copy()
  bias_before = experiment.model.actuator_biasprm.copy()
  force_before = experiment.model.actuator_forcerange.copy()
  control_before = experiment.model.actuator_ctrlrange.copy()
  settings = script["PressureWindowSettings"](
    contact_time_constant_s=None,
    contact_friction_impedance_ratio=1.0,
    finger_servo_velocity_gain=args.settings.finger_servo_velocity_gain,
  )
  metadata = script["_configure_contact_model"](experiment, settings)
  expected_names = {
    f"hand_r_{finger}_joint{joint}"
    for finger in ("index", "middle", "ring", "pinky")
    for joint in (2, 3)
  }
  expected_ids = [experiment.model.actuator(name).id for name in expected_names]
  assert set(metadata["experimental_right_finger_servo_gains"]) == expected_names
  assert len(expected_ids) == 8
  expected_gain = gain_before.copy()
  expected_gain[expected_ids, 0] = 0.3
  expected_bias = bias_before.copy()
  expected_bias[expected_ids, 2] = -0.3
  # Full-array checks cover other columns, thumb, both arms, the other hand,
  # and joint1/joint4 on active fingers, not only the eight selected entries.
  assert np.array_equal(experiment.model.actuator_gainprm, expected_gain)
  assert np.array_equal(experiment.model.actuator_biasprm, expected_bias)
  assert np.array_equal(experiment.model.actuator_forcerange, force_before)
  assert np.array_equal(experiment.model.actuator_ctrlrange, control_before)
  assert np.array_equal(baseline.model.actuator_gainprm, gain_before)
  assert np.array_equal(baseline.model.actuator_biasprm, bias_before)
  for name in expected_names:
    assert metadata["experimental_right_finger_servo_gains"][name] == {
      "original": gain_before[experiment.model.actuator(name).id, 0],
      "used": 0.3,
    }
  assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("timestep", [0.0005, 0.001, 0.002])
def test_timestep_override_updates_only_local_simulation_and_records_metadata(
  tmp_path, timestep
):
  script = _script()
  args = script["_parse_args"](
    [
      "--output-dir",
      str(tmp_path / "not_created"),
      "--timestep",
      str(timestep),
    ]
  )
  assert args.settings.physics_timestep_s == timestep
  baseline = _simulation()
  experiment = _simulation()
  attributes = (
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrlrange",
    "geom_solref",
    "pair_solref",
    "geom_friction",
    "pair_friction",
  )
  before = {name: getattr(experiment.model, name).copy() for name in attributes}
  settings = script["PressureWindowSettings"](
    contact_time_constant_s=None,
    contact_friction_impedance_ratio=1.0,
    physics_timestep_s=args.settings.physics_timestep_s,
  )
  metadata = script["_configure_contact_model"](experiment, settings)
  assert experiment.timestep == timestep
  assert experiment.model.opt.timestep == timestep
  assert baseline.timestep == 0.002
  assert baseline.model.opt.timestep == 0.002
  assert metadata["physics_timestep_original_s"] == 0.002
  assert metadata["physics_timestep_used_s"] == timestep
  assert experiment.model.opt.impratio == 1.0
  for name, original in before.items():
    assert np.array_equal(getattr(experiment.model, name), original)
  assert not (tmp_path / "not_created").exists()


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("failure_stage", [None, "planner", "draw"])
@pytest.mark.parametrize("goal", ["short", "table-edge"])
def test_full_fake_cli_reuses_one_model_and_retains_each_trial(
  tmp_path, monkeypatch, record, failure_stage, goal
):
  from kaihand_tactile_env.shared import tactile, task_video
  from kaihand_tactile_env.tasks.poker_draw import pressure_video

  script = _script()
  global_values = script["main"].__globals__
  simulations, resets, executions, videos, metadata_calls = [], [], [], [], []
  tactile_providers = []
  camera_calls = []
  control_error = failure_stage is not None

  def create_simulation(_model_path, **kwargs):
    assert kwargs == {"scene": "poker-draw", "add_genesis_probes": False}
    simulation = _simulation()

    def reset(**options):
      resets.append(options)
      simulation.data.time = 0.0
      simulation.data.geom_xpos[:, 0] = 0.6
      simulation.data.site_xpos[:, 0] = 0.62
      simulation.drive_limit_n = None

    simulation.reset = reset
    simulations.append(simulation)
    return simulation

  class FakeExecutor:
    def __init__(self, simulation, *, press_force_per_finger_n, observer):
      self.simulation = simulation
      self.force = press_force_per_finger_n
      self.observer = observer
      self.edge_outcome = None

    def control_metadata(self):
      metadata_calls.append(self.force)
      return {"controller": "fake_experimental", "target_force_n": self.force}

    def draw(self, _plan, settings):
      assert metadata_calls[-1] == self.force  # snapshot before any stepping
      executions.append(self.force)
      _advance(self.simulation)
      self.observer(self.simulation, "four_finger_press")
      self.simulation.drive_limit_n = settings.drive_limit_n
      _advance(self.simulation)
      self.observer(self.simulation, "slide_card")
      _advance(self.simulation)
      if settings.goal == "table-edge":
        self.edge_outcome = {
          "target_reached": failure_stage != "draw",
          "terminal_reason": "edge_reached"
          if failure_stage != "draw"
          else "interrupted",
          "required_card_travel_m": 0.114,
          "commanded_travel_m": 0.12,
          "actual_card_travel_m": 0.114 if failure_stage != "draw" else 0.02,
          "final_overhang_fraction": 0.49 if failure_stage != "draw" else 0.0,
          "maximum_overhang_fraction": 0.49 if failure_stage != "draw" else 0.0,
          "slide_force_quality": {"all_fingers_loaded": failure_stage != "draw"},
          "full_slide_qualified": failure_stage != "draw",
        }
      if failure_stage == "draw":
        raise RuntimeError("force budget check failed")
      self.observer(self.simulation, "edge_hold")
      return self.edge_outcome

  def fake_plan():
    if failure_stage == "planner":
      raise RuntimeError("planner failed before constructing executor")
    return "fake_plan"

  class FakeVideo:
    def __init__(
      self, simulation, output_path, *, fps, preview, metadata, tactile_provider
    ):
      assert simulation is simulations[0]
      assert tactile_provider is tactile_providers[-1]
      assert tactile_provider.model is simulation.model
      assert fps == 5.0
      assert preview is False
      assert metadata["scope"] == (
        "pressure_window_to_table_edge_not_pickup"
        if goal == "table-edge"
        else "short_pressure_window_not_full_draw"
      )
      assert "production_reference_press_control" in metadata
      assert "press_control" not in metadata
      self.path = output_path
      self.phases = []
      self.finished = []
      self.closed = False
      videos.append(self)

    def __enter__(self):
      return self

    def __exit__(self, *_args):
      self.closed = True

    def observe(self, _simulation, phase):
      self.phases.append(phase)

    def set_outcome(self, row):
      self.outcome = row.copy()

    def finish(self, *, success, error=None):
      assert error is None  # physical/control error belongs in the task outcome
      self.finished.append(success)
      with self.path.open("x") as stream:
        stream.write("fake video; no rendering")
      sidecar = self.path.with_suffix(".json")
      with sidecar.open("x") as stream:
        json.dump({"protocol_completed": success}, stream)
      return self.path, sidecar

  def create_tactile_provider(model):
    assert record, "numeric-only trials must not allocate a video tactile provider"
    assert model is simulations[0].model
    provider = SimpleNamespace(model=model)
    tactile_providers.append(provider)
    return provider

  def configure_camera(video, simulation):
    assert goal == "table-edge"
    assert simulation is simulations[0]
    assert not video.phases  # Configure before recording the initial frame.
    camera_calls.append((video, simulation))
    return {"configuration": "fake_pressure_window_closeup"}

  monkeypatch.setitem(global_values, "ForceLimitedPokerSimulation", create_simulation)
  monkeypatch.setitem(
    global_values,
    "model_with_table_card_friction",
    lambda *_args: nullcontext(Path("fake_model.xml")),
  )
  monkeypatch.setitem(
    global_values, "default_model_path", lambda _scene: Path("fake_scene.xml")
  )
  monkeypatch.setitem(
    global_values, "model_fingerprint", lambda _scene: "fake_fingerprint"
  )
  monkeypatch.setitem(
    global_values, "_source_hashes", lambda: {"pressure_window.py": "fake_hash"}
  )
  monkeypatch.setitem(
    global_values,
    "PokerDrawPlanner",
    lambda _simulation: SimpleNamespace(plan=fake_plan),
  )
  monkeypatch.setitem(global_values, "PressureWindowExecutor", FakeExecutor)
  monkeypatch.setitem(global_values, "PokerFrictionTelemetry", _Telemetry)
  monkeypatch.setattr(tactile, "SolverContactTactileProvider", create_tactile_provider)
  monkeypatch.setattr(task_video, "TaskVideoRecorder", FakeVideo)
  monkeypatch.setattr(
    pressure_video, "configure_pressure_window_video", configure_camera
  )
  directory = tmp_path / "new_trial"
  args = ["--output-dir", str(directory), "--press-force", ".06", ".20", "--goal", goal]
  if record:
    args.append("--record")
  script["main"](args)
  summary = json.loads((directory / "summary.json").read_text())
  assert len(simulations) == 1
  expected_executions = [] if failure_stage == "planner" else [0.06, 0.20]
  assert executions == expected_executions
  assert metadata_calls == expected_executions
  assert resets == [{"seed": 0, "object_xy_jitter": 0.0, "object_yaw_jitter": 0.0}] * 2
  assert summary["validation_status"] == "candidate_not_validated"
  assert "production_reference_press_control" in summary
  assert "press_control" not in summary
  assert not summary["state_noise"]
  assert len(summary["trials"]) == 2
  # Cap stays latched after a draw; an early planner failure never enables it.
  assert simulations[0].drive_limit_n == (None if failure_stage == "planner" else 4.0)
  for index, row in enumerate(summary["trials"], 1):
    assert row["classification"] == (
      "control_error"
      if control_error
      else (
        "edge_reached_contact_qualified" if goal == "table-edge" else "unclassified"
      )
    )
    assert row["protocol_completed"] is not control_error
    if goal == "short" or failure_stage == "planner":
      assert row["goal_outcome"] is None
    else:
      assert row["goal_outcome"]["required_card_travel_m"] == 0.114
      assert row["goal_outcome"]["full_slide_qualified"] is not control_error
      if failure_stage == "draw":
        assert row["goal_outcome"]["actual_card_travel_m"] == 0.02
        assert row["goal_outcome"]["terminal_reason"] == "interrupted"
    expected_control = (
      None
      if failure_stage == "planner"
      else {
        "controller": "fake_experimental",
        "target_force_n": row["press_force_per_finger_n"],
      }
    )
    if expected_control is not None and goal == "table-edge":
      expected_control.update(
        object_motion_or_slip_used_for_control=True,
        geometry_usage="endpoint speed/stop/dwell only; no slip-based pressure adjustment",
        slip_feedback_used_for_pressure_adjustment=False,
      )
    assert row["experimental_control"] == expected_control
    assert row["terminal_drive_state"]["drive_actual_fx_n"] == -1.6
    assert Path(row["timeseries_csv"]).is_file()
    assert (directory / f"trial_{index:03d}_measurement.json").is_file()
    assert Path(row["trial_json"]).is_file()
    assert (
      json.loads(Path(row["trial_json"]).read_text())["result"]["experimental_control"]
      == expected_control
    )
    assert (
      json.loads((directory / f"trial_{index:03d}_measurement.json").read_text())[
        "result"
      ]["experimental_control"]
      == expected_control
    )
    with Path(row["timeseries_csv"]).open(newline="") as stream:
      rows = list(csv.DictReader(stream))
    if failure_stage == "planner":
      assert rows == []
    else:
      assert len(rows) == 3
      assert rows[-1]["phase"] == (
        "terminal_unobserved" if control_error else "edge_hold"
      )
      assert float(rows[-1]["drive_actual_fx_n"]) == -1.6
    if record:
      assert Path(row["video_artifacts"]["mp4"]).is_file()
      assert videos[index - 1].closed
      assert videos[index - 1].finished == [not control_error]
      assert videos[index - 1].phases[0] == "initial"
      assert videos[index - 1].outcome["experimental_control"] == expected_control
      assert row["video_camera"] == (
        {"configuration": "fake_pressure_window_closeup"}
        if goal == "table-edge"
        else None
      )
    else:
      assert row["video_artifacts"] is None
      assert row["video_camera"] is None
  assert len(videos) == (2 if record else 0)
  assert len(tactile_providers) == (2 if record else 0)
  assert len(camera_calls) == (2 if record and goal == "table-edge" else 0)


def test_record_cli_initializes_real_tactile_without_genesis_probe_geometry(
  tmp_path, monkeypatch
):
  """Exercise the real recorder/provider path with eleven tiny static bodies.

  Only RGB rendering/encoding and the robot planner are faked. No full robot
  model, physics stepping, OpenGL context, preview or encoder is allocated.
  """
  import mujoco
  from kaihand_tactile_env.shared import task_video
  from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
  from kaihand_tactile_env.shared.tactile import (
    GenesisProbeTactileProvider,
    load_fingertip_layout,
  )

  bodies = "".join(
    f'<body name="{name}" pos="{index * 0.04} 0 .1">'
    f'<geom name="{name}_tactile_pad_col" type="box" size=".01 .01 .002"/>'
    "</body>"
    for index, name in enumerate(FINGERTIP_LINK_NAMES)
  )
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><compiler fusestatic="false"/><worldbody>'
    + bodies
    + '<site name="right_ee"/><body name="card" pos="0 0 .2">'
    '<geom name="card_core_geom" type="box" size=".03 .04 .001"/>'
    "</body></worldbody></mujoco>"
  )
  data = mujoco.MjData(model)
  layout = load_fingertip_layout()
  # The unpatched recorder default reproduces the exact reported exception.
  with pytest.raises(RuntimeError, match="workcell_genesis_probe_0000"):
    GenesisProbeTactileProvider(model, layout)
  simulation = SimpleNamespace(
    model=model,
    data=data,
    genesis_probe_layout=layout,
    scene="poker-draw",
    timestep=model.opt.timestep,
    drive_limit_n=None,
    _site_id={"right": model.site("right_ee").id},
    drive_state=lambda: {},
  )

  def reset(**_options):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

  simulation.reset = reset
  allocations = []
  frames = []

  def create_simulation(_model_path, **options):
    assert options == {"scene": "poker-draw", "add_genesis_probes": False}
    allocations.append(simulation)
    return simulation

  class TinyRenderer:
    def __init__(self, actual_model, *, height, width):
      assert actual_model is model
      self.shape = (height, width, 3)
      self.scene = SimpleNamespace(
        flags=np.ones(int(mujoco.mjtRndFlag.mjNRNDFLAG), dtype=np.uint8)
      )

    def disable_depth_rendering(self):
      pass

    def disable_segmentation_rendering(self):
      pass

    def update_scene(self, actual_data, *, camera):
      assert actual_data is data
      assert camera in ("front", "overhead")

    def render(self):
      return np.zeros(self.shape, dtype=np.uint8)

    def close(self):
      pass

  class TinyWriter:
    def __init__(self, path, **_options):
      self.path = path

    def write(self, frame):
      frames.append(frame.shape)

    def finish(self):
      self.path.touch(exist_ok=True)

    def close(self):
      self.finish()

  def stop_before_robot_motion():
    raise RuntimeError("tiny fixture has no robot planner")

  script = _script()
  replacements = {
    "ForceLimitedPokerSimulation": create_simulation,
    "model_with_table_card_friction": lambda *_args: nullcontext(Path("tiny.xml")),
    "default_model_path": lambda _scene: Path("tiny.xml"),
    "model_fingerprint": lambda _scene: "tiny-fixture",
    "_source_hashes": lambda: {},
    "_configure_contact_model": lambda *_args: {"tiny_fixture": True},
    "PokerFrictionTelemetry": _Telemetry,
    "PokerDrawPlanner": lambda _sim: SimpleNamespace(plan=stop_before_robot_motion),
  }
  for name, value in replacements.items():
    monkeypatch.setitem(script["main"].__globals__, name, value)
  monkeypatch.setattr(task_video.mujoco, "Renderer", TinyRenderer)
  monkeypatch.setattr(task_video, "_FfmpegPipeWriter", TinyWriter)
  monkeypatch.setattr(task_video, "_find_ffmpeg_executable", lambda: "/fake/ffmpeg")
  output = tmp_path / "tiny_recording"
  script["main"](["--output-dir", str(output), "--press-force", ".35", "--record"])
  assert allocations == [simulation]
  assert frames == [(480, 640, 3)] * 2  # initial and marked terminal frames
  sidecar = json.loads((output / "trial_001_video.json").read_text())
  assert sidecar["encoding_complete"]
  assert sidecar["recording_complete"]
  assert sidecar["tactile"]["source"] == "solver_contact_distributed_taxel_v1"
  assert sidecar["tactile"]["hand"] == "right"
  assert sidecar["tactile"]["display_rows"] == [
    "normal_taxel_force_n",
    "norm(tangent_taxel_force_n, axis=-1)",
  ]
  assert sidecar["tactile"]["normal_force_unit"] == "N"
  assert sidecar["task_outcome"]["error"] == "tiny fixture has no robot planner"
  assert not sidecar["task_success"]
