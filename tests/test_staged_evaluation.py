"""Read-only staged measurements, ordering, and continuous physics-time proof."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from kaihand_tactile_env.shared.staged_evaluation import (
  EVALUATION_VERSION,
  STAGE_NAMES,
  StageThresholds,
  StagedEvaluationMonitor,
)


def physical(**overrides):
  return {
    "time_s": 0.0,
    "opposed_grasp_now": False,
    "lift_from_initial_m": 0.0,
    "support_free_now": False,
    "near_mouth_distance_m": 0.30,
    "seated_now": False,
    "thread_engaged_now": False,
    "fully_clear_now": False,
    "object_position_xyz_m": [0.50, -0.20, 0.70],
    "linear_speed_m_s": 0.0,
    "angular_speed_rad_s": 0.0,
    "table_contact_count": 1,
    "table_normal_load_n": 1.0,
    "robot_contact_count": 0,
    "contacted_by_hand_now": False,
    "planar_displacement_m": 0.0,
    "table_clearance_m": 0.0,
    "poker_terminal_position_error_m": 0.2,
    "poker_face_to_head_cosine": 0.0,
    "poker_face_to_robot_cosine": 0.0,
    **overrides,
  }


def observe_for(monitor, measurements, duration_s, dt=0.01):
  start = 0.0 if monitor._last_time is None else monitor._last_time + dt
  # Inclusive endpoints make a duration an observed span, not a sample count.
  for tick in range(round(duration_s / dt) + 1):
    monitor.observe({**measurements, "time_s": start + tick * dt}, timestep=dt)
  return monitor.report()


def lifted():
  return physical(
    opposed_grasp_now=True, lift_from_initial_m=0.04,
    support_free_now=True, robot_contact_count=2,
    table_contact_count=0, table_normal_load_n=0.0,
  )


def prepare_poker_terminal(monitor):
  observe_for(monitor, physical(
    contacted_by_hand_now=True, planar_displacement_m=0.03,
  ), 0.06)
  observe_for(monitor, lifted(), 0.16)
  assert monitor.report()["highest_stage"] == 2
  return {**lifted(), "table_clearance_m": 0.08,
          "poker_terminal_position_error_m": 0.01,
          "poker_face_to_head_cosine": 0.95,
          "poker_face_to_robot_cosine": 0.95}


def test_thresholds_are_fixed_and_json_serializable():
  thresholds = StageThresholds()
  assert thresholds.near_mouth_m == 0.05
  assert thresholds.grasp_hold_s == 0.15
  assert thresholds.poker_terminal_hold_s == 3.0
  assert thresholds.usb_extracted_hold_s == 2.0
  assert json.loads(json.dumps(asdict(thresholds))) == asdict(thresholds)
  with pytest.raises(ValueError):
    StageThresholds(lift_m=float("nan"))
  with pytest.raises(ValueError):
    StageThresholds(grasp_hold_s=0)


@pytest.mark.parametrize("task", ["bulb-screw", "usb-insert"])
def test_insertion_requires_ordered_pickup_near_and_mechanical_seating(task):
  monitor = StagedEvaluationMonitor(task)
  result = observe_for(monitor, physical(seated_now=True, thread_engaged_now=True), 0.4)
  assert result["highest_stage"] == 0  # Initial seating/phase is not pickup.
  observe_for(monitor, lifted(), 0.16)
  assert monitor.report()["highest_stage"] == 1
  observe_for(monitor, {**lifted(), "near_mouth_distance_m": 0.04}, 0.16)
  assert monitor.report()["highest_stage"] == 2
  result = observe_for(monitor, physical(seated_now=True, thread_engaged_now=True), 0.21)
  assert result["complete_success"] is True
  assert result["stages"] == {"stage1": True, "stage2": True, "stage3": True}
  assert list(result["first_times"].values()) == sorted(result["first_times"].values())


def test_bulb_merely_engaged_thread_is_not_seated_success():
  monitor = StagedEvaluationMonitor("bulb-screw")
  observe_for(monitor, lifted(), 0.16)
  observe_for(monitor, {**lifted(), "near_mouth_distance_m": 0.02}, 0.16)
  result = observe_for(monitor, physical(thread_engaged_now=True, seated_now=False), 1.0)
  assert result["highest_stage"] == 2
  assert not result["complete_success"]


def test_null_seating_does_not_become_a_positive():
  monitor = StagedEvaluationMonitor("usb-insert")
  observe_for(monitor, lifted(), 0.16)
  observe_for(monitor, {**lifted(), "near_mouth_distance_m": 0.02}, 0.16)
  result = observe_for(monitor, physical(seated_now=None, success=True, phase="done"), 1.0)
  assert result["measurements"]["seated_now"] is None
  assert not result["complete_success"]


def test_ram_pushed_near_slot_is_not_carried_stage1():
  monitor = StagedEvaluationMonitor("install-ram")
  result = observe_for(monitor, physical(near_mouth_distance_m=0.01, seated_now=True), 1.0)
  assert result["highest_stage"] == 0
  observe_for(monitor, lifted(), 0.16)
  observe_for(monitor, {**lifted(), "near_mouth_distance_m": 0.03}, 0.16)
  assert monitor.report()["highest_stage"] == 1
  result = observe_for(monitor, physical(seated_now=True), 0.21)
  assert result["complete_success"]


def test_bulb_extraction_any_table_position_allowed_but_drop_is_not_placement():
  monitor = StagedEvaluationMonitor("bulb-unscrew")
  clear = {**lifted(), "fully_clear_now": True}
  observe_for(monitor, clear, 0.16)
  assert monitor.report()["highest_stage"] == 1
  # A released object lying anywhere on the table cannot prove held transport.
  result = observe_for(monitor, physical(fully_clear_now=True), 0.3)
  assert not result["complete_success"]
  observe_for(monitor, clear, 0.16)
  carried = {**clear, "object_position_xyz_m": [0.90, 0.30, 0.90]}
  observe_for(monitor, carried, 0.16)
  assert monitor.report()["transport_verified"]
  observe_for(monitor, {
    **carried, "support_free_now": False,
    "table_contact_count": 2, "table_normal_load_n": 0.4,
  }, 0.06)
  assert monitor.report()["controlled_table_placement_verified"]
  result = observe_for(monitor, {
    **physical(fully_clear_now=True),
    "object_position_xyz_m": [0.90, 0.30, 0.69],
  }, 0.21)
  assert result["complete_success"]


def test_grasp_break_cannot_fake_transport_by_regrasp_far_away():
  monitor = StagedEvaluationMonitor("bulb-unscrew")
  clear = {**lifted(), "fully_clear_now": True}
  observe_for(monitor, clear, 0.16)
  observe_for(monitor, physical(fully_clear_now=True), 0.03)
  observe_for(monitor, {**clear, "object_position_xyz_m": [0.90, 0.30, 0.90]}, 0.16)
  assert not monitor.report()["transport_verified"]


def usb_held_clear(**overrides):
  return physical(**{
    "fully_clear_now": True, "hand_contact_count": 1, "table_contact_count": 0,
    # The new task has no opposition, speed, pose or transport requirement.
    "opposed_grasp_now": None, "contacted_by_hand_now": None, "support_free_now": None,
    "object_position_xyz_m": None, "linear_speed_m_s": None, "angular_speed_rad_s": None,
    **overrides,
  })


def test_usb_extraction_requires_two_seconds_of_continuous_clear_hand_hold():
  monitor = StagedEvaluationMonitor("usb-unplug")
  result = observe_for(monitor, usb_held_clear(), 1.99)
  assert result["stage_definitions"] == {
    "stage1": "fully_extracted", "stage2": "extracted_and_held_2s",
  }
  assert result["completion_protocol"] == "usb-unplug-held-clear-2s-v2"
  assert result["stages"] == {"stage1": True, "stage2": False}
  assert result["complete_success"] is False
  result = monitor.observe({**usb_held_clear(), "time_s": 2.0}, timestep=0.01)
  assert result["stages"] == {"stage1": True, "stage2": True}
  assert result["complete_success"] is True
  assert result["named_first_times"]["extracted_and_held_2s"] == pytest.approx(2.0)
  assert result["transport_verified"] is False
  assert result["controlled_table_placement_verified"] is False
  # Achievement remains cumulative after the observed two-second task ends.
  result = monitor.observe({**physical(), "time_s": 2.01}, timestep=0.01)
  assert result["complete_success"] is True


@pytest.mark.parametrize("overrides", [
  {"hand_contact_count": 0},
  {"hand_contact_count": None},
  {"fully_clear_now": False},
  {"fully_clear_now": None},
  {"table_contact_count": 1},
  {"table_contact_count": None},
])
def test_usb_extraction_hold_resets_on_reinsertion_release_table_or_missing(overrides):
  monitor = StagedEvaluationMonitor("usb-unplug")
  observe_for(monitor, usb_held_clear(), 1.99)
  # A native completion flag may not substitute for physical hold evidence.
  interrupted = {
    **usb_held_clear(), **overrides,
    "success": True, "native_task_state": {"success": True, "extracted_and_held_2s": True},
    "time_s": 2.0,
  }
  result = monitor.observe(interrupted, timestep=0.01)
  assert result["complete_success"] is False
  assert result["continuous_durations_s"]["usb_extracted_and_held"] == 0
  result = observe_for(monitor, usb_held_clear(), 1.99)
  assert result["complete_success"] is False
  result = monitor.observe({**usb_held_clear(), "time_s": monitor._last_time + 0.01}, timestep=0.01)
  assert result["complete_success"] is True


@pytest.mark.parametrize("missing", ["hand_contact_count", "table_contact_count", "fully_clear_now"])
def test_usb_extraction_missing_fields_do_not_prove_a_hold(missing):
  monitor = StagedEvaluationMonitor("usb-unplug")
  measurements = usb_held_clear()
  del measurements[missing]
  result = observe_for(monitor, measurements, 3.0)
  assert result["complete_success"] is False
  assert result["continuous_durations_s"]["usb_extracted_and_held"] == 0


def test_usb_extraction_sampling_gap_restarts_hold_but_preserves_reached_stage():
  monitor = StagedEvaluationMonitor("usb-unplug")
  observe_for(monitor, usb_held_clear(), 1.99)
  result = monitor.observe({**usb_held_clear(), "time_s": 10.0}, timestep=0.01)
  assert result["observation_gap_count"] == 1
  assert result["stages"] == {"stage1": True, "stage2": False}
  assert result["continuous_durations_s"]["usb_extracted_and_held"] == 0
  result = observe_for(monitor, usb_held_clear(), 1.98)
  assert result["complete_success"] is False
  result = monitor.observe({**usb_held_clear(), "time_s": 12.0}, timestep=0.01)
  assert result["complete_success"] is True


def test_usb_native_flag_without_hand_contact_cannot_complete_new_protocol():
  monitor = StagedEvaluationMonitor("usb-unplug")
  result = observe_for(monitor, physical(
    fully_clear_now=True, hand_contact_count=0, table_contact_count=0,
    native_task_state={"success": True, "fully_extracted": True, "extracted_and_held_2s": True},
  ), 3.0)
  assert result["stages"] == {"stage1": True, "stage2": False}
  assert result["complete_success"] is False


@pytest.mark.parametrize("opposed", [False, None])
def test_usb_extraction_no_opposed_grasp_or_force_threshold_required(opposed):
  monitor = StagedEvaluationMonitor("usb-unplug")
  result = observe_for(monitor, {
    **usb_held_clear(), "opposed_grasp_now": opposed,
    "finger_load_n": {"thumb": 0, "index": 0}, "hand_normal_load_n": 0,
  }, 2.0)
  assert result["complete_success"] is True


def test_poker_drag_does_not_require_reaching_table_edge():
  monitor = StagedEvaluationMonitor("poker-draw")
  result = observe_for(monitor, physical(
    contacted_by_hand_now=True, planar_displacement_m=0.03,
  ), 0.06)
  assert result["highest_stage"] == 1


def test_poker_aerial_card_motion_is_not_a_table_drag():
  monitor = StagedEvaluationMonitor("poker-draw")
  result = observe_for(monitor, {**lifted(), "contacted_by_hand_now": True,
    "planar_displacement_m": 0.08, "table_clearance_m": 0.1}, 0.4)
  assert result["highest_stage"] == 0


def poker_table_measurement_simulation(*, include_poker_table=True, include_card_geom=True):
  """A real two-height contact scene, with the low worktop declared first."""
  playing_surface = 0.70 if include_poker_table else 0.20
  raised_table = (
    '<geom name="poker_table_top" type="box" pos="0 0 0.65" size="0.2 0.2 0.05"/>'
    if include_poker_table else ""
  )
  card_name = "card_core_geom" if include_card_geom else "unnamed_card_core"
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.001"/>
      <default><geom solref="0.002 1"/></default>
      <worldbody>
        <geom name="tabletop" type="box" pos="0 0 0.15" size="0.2 0.2 0.05"/>
        {raised_table}
        <body name="card" pos="0 0 {playing_surface + 0.0009}">
          <freejoint/>
          <geom name="{card_name}" type="box" size="0.025 0.035 0.001" mass="0.004"/>
        </body>
        <body name="hand_r_index" pos="0.03 0 {playing_surface + 0.0068}">
          <geom name="hand_r_index_tactile_pad_col" type="sphere" size="0.005"/>
        </body>
      </worldbody>
    </mujoco>
  """)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  card_body = model.body("card").id
  return SimpleNamespace(
    model=model, data=data, timestep=float(model.opt.timestep),
    object_pose=lambda _name: np.r_[data.xpos[card_body], data.xquat[card_body]],
    object_twist=lambda _name: np.zeros(6),
  )


def test_poker_clearance_uses_raised_playing_surface_not_first_aggregate_table_geom():
  simulation = poker_table_measurement_simulation()
  monitor = StagedEvaluationMonitor("poker-draw", simulation)
  result = monitor.update()
  measured = result["measurements"]
  assert monitor._adapter.table_geoms == {
    simulation.model.geom("tabletop").id,
    simulation.model.geom("poker_table_top").id,
  }
  assert measured["table_contact_count"] > 0
  assert measured["table_clearance_m"] == pytest.approx(-0.0001)
  assert measured["table_clearance_reference_geom"] == "poker_table_top"
  assert not measured["missing_measurements"]
  # Lift geometry independently of contact forces: the reference remains the
  # raised surface, not the worktop half a metre below it.
  simulation.data.qpos[2] += 0.03
  mujoco.mj_forward(simulation.model, simulation.data)
  measured = monitor.update()["measurements"]
  assert measured["table_clearance_m"] == pytest.approx(0.0299)


def test_poker_real_raised_table_drag_reaches_stage1():
  simulation = poker_table_measurement_simulation()
  monitor = StagedEvaluationMonitor("poker-draw", simulation)
  # Capture the reset origin before moving the card on its playing surface.
  simulation.data.qpos[0] += 0.03
  for tick in range(61):
    simulation.data.time = tick * simulation.timestep
    mujoco.mj_forward(simulation.model, simulation.data)
    result = monitor.update()
    assert result["measurements"]["contacted_by_hand_now"] is True
    assert result["measurements"]["table_contact_count"] > 0
    assert result["measurements"]["table_clearance_m"] == pytest.approx(-0.0001)
  assert result["stages"] == {"stage1": True, "stage2": False, "stage3": False}


def test_poker_missing_named_playing_surface_does_not_guess_worktop_or_fake_drag():
  simulation = poker_table_measurement_simulation(include_poker_table=False)
  monitor = StagedEvaluationMonitor("poker-draw", simulation)
  simulation.data.qpos[0] += 0.03
  for tick in range(61):
    simulation.data.time = tick * simulation.timestep
    mujoco.mj_forward(simulation.model, simulation.data)
    result = monitor.update()
  measured = result["measurements"]
  assert measured["table_contact_count"] > 0  # Aggregate contact counts survive.
  assert measured["table_clearance_m"] is None
  assert measured["table_clearance_reference_geom"] is None
  assert "poker_table_top_geom" in measured["missing_measurements"]
  assert result["highest_stage"] == 0
  assert result["complete_success"] is False


def test_poker_missing_card_geometry_reports_unavailable_clearance():
  simulation = poker_table_measurement_simulation(include_card_geom=False)
  result = StagedEvaluationMonitor("poker-draw", simulation).update()
  measured = result["measurements"]
  assert measured["table_clearance_m"] is None
  assert measured["table_clearance_reference_geom"] is None
  assert "card_core_geom" in measured["missing_measurements"]
  assert result["highest_stage"] == 0


def test_poker_terminal_requires_strictly_more_than_3_sim_seconds():
  monitor = StagedEvaluationMonitor("poker-draw")
  terminal = prepare_poker_terminal(monitor)
  result = observe_for(monitor, terminal, 3.0)
  assert result["continuous_durations_s"]["terminal_hold"] == pytest.approx(3.0)
  assert result["highest_stage"] == 2
  result = observe_for(monitor, terminal, 0.01)
  assert result["highest_stage"] == 3


def test_poker_hold_resets_on_grasp_loss_and_does_not_accumulate_fragments():
  monitor = StagedEvaluationMonitor("poker-draw")
  terminal = prepare_poker_terminal(monitor)
  observe_for(monitor, terminal, 2.0)
  observe_for(monitor, {**terminal, "opposed_grasp_now": False}, 0.03)
  result = observe_for(monitor, terminal, 2.0)
  assert result["highest_stage"] == 2
  result = observe_for(monitor, terminal, 1.03)
  assert result["highest_stage"] == 3


@pytest.mark.parametrize("field,value", [
  ("poker_terminal_position_error_m", 0.04),
  ("poker_face_to_head_cosine", 0.5),
  ("poker_face_to_robot_cosine", 0.5),
  ("table_clearance_m", 0.0),
  ("table_contact_count", 1),
  ("linear_speed_m_s", 0.04),
  ("angular_speed_rad_s", 0.4),
])
def test_poker_each_terminal_measurement_matters(field, value):
  monitor = StagedEvaluationMonitor("poker-draw")
  terminal = prepare_poker_terminal(monitor)
  result = observe_for(monitor, {**terminal, field: value}, 3.2)
  assert not result["complete_success"]


def test_sparse_observations_cannot_prove_continuous_hold():
  monitor = StagedEvaluationMonitor("poker-draw")
  terminal = prepare_poker_terminal(monitor)
  start = monitor._last_time
  for index in range(1, 10):
    monitor.observe({**terminal, "time_s": start + index}, timestep=0.01)
  result = monitor.report()
  assert result["observation_gap_count"] == 9
  assert result["continuous_durations_s"]["terminal_hold"] == 0
  assert not result["complete_success"]


def test_duplicate_timestamp_does_not_advance_and_rewind_fails_closed():
  monitor = StagedEvaluationMonitor("bulb-screw")
  for _ in range(100):
    monitor.observe(lifted(), timestep=0.01)
  assert monitor.report()["highest_stage"] == 0
  monitor.observe({**lifted(), "time_s": 0.01}, timestep=0.01)
  with pytest.raises(ValueError, match="rewound"):
    monitor.observe(lifted(), timestep=0.01)


@pytest.mark.parametrize("task", [
  "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
])
def test_whiteboard_pickup_does_not_claim_complete_wipe_success(task):
  monitor = StagedEvaluationMonitor(task)
  result = observe_for(monitor, lifted(), 0.16)
  assert result["stages"] == {"stage1": True}
  assert result["complete_success"] is None
  monitor.attach_continuous_metrics({"cleanliness": 0.5, "geometric_contact_ratio": 0.7})
  result = monitor.report()
  assert result["continuous_metrics"]["cleanliness"] == 0.5
  assert json.loads(json.dumps(result, allow_nan=False))["version"] == EVALUATION_VERSION
  result["stages"]["stage1"] = False
  assert monitor.report()["stages"]["stage1"]  # returned snapshots are detached


@pytest.mark.parametrize("task", sorted(STAGE_NAMES))
def test_real_simulation_read_only_initial_measurements(task):
  if task == "poker-draw":
    from kaihand_tactile_env.shared.simulation import ArmHandSimulation
    simulation = ArmHandSimulation(scene="poker-draw")
    state = None
    adapter = None
  elif task == "usb-insert":
    from kaihand_tactile_env.shared.simulation import ArmHandSimulation
    from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion
    from kaihand_tactile_env.tasks.usb_insert.task import UsbInsertionMonitor
    simulation = ArmHandSimulation(scene="usb-insert")
    initialize_for_insertion(simulation, seed=0)
    native_monitor = UsbInsertionMonitor(simulation)
    state = native_monitor.measure()
    adapter = None
  else:
    from kaihand_tactile_env.shared.policy_tasks import TaskPolicyAdapter
    adapter = TaskPolicyAdapter(task, seed=0, guard_mode="observe")
    simulation, state = adapter.simulation, adapter.state
  monitor = StagedEvaluationMonitor(task, simulation)
  if task.startswith("whiteboard-"):
    # Same integration point as the production recorder; physics is unchanged.
    assert simulation.evaluation_contact_observer is None
    simulation.evaluation_contact_observer = monitor.capture_integrated_contacts
  before = {name: getattr(simulation.data, name).copy() for name in (
    "qpos", "qvel", "ctrl", "eq_active", "qfrc_applied", "xfrc_applied",
  )}
  before_time = float(simulation.data.time)
  before_eq = simulation.model.eq_data.copy()
  result = monitor.update(task_state=state)
  for name, value in before.items():
    np.testing.assert_array_equal(getattr(simulation.data, name), value)
  np.testing.assert_array_equal(simulation.model.eq_data, before_eq)
  assert simulation.data.time == before_time
  assert result["highest_stage"] == 0
  assert result["measurements"]["measurement_source"] == "mujoco_solver_and_geometry_read_only"
  assert len(result["measurements"]["object_pose_wxyz"]) == 7
  assert result["measurements"]["robot_contact_count"] >= 0
  assert result["measurements"]["table_contact_count"] >= 0
  assert not result["simulation_state_mutated"]
  assert not result["expert_trajectory_used"]
  json.dumps(result, allow_nan=False)
  for _ in range(2):
    if adapter is not None:
      adapter.step()
      state = adapter.state
    else:
      simulation.step()
      if task == "usb-insert":
        state = native_monitor.update()
    before_pos, before_vel, before_control = (
      simulation.data.qpos.copy(), simulation.data.qvel.copy(), simulation.data.ctrl.copy(),
    )
    before_time = float(simulation.data.time)
    result = monitor.update(task_state=state)
    np.testing.assert_array_equal(simulation.data.qpos, before_pos)
    np.testing.assert_array_equal(simulation.data.qvel, before_vel)
    np.testing.assert_array_equal(simulation.data.ctrl, before_control)
    assert simulation.data.time == before_time
    assert result["observations"] >= 2
    assert result["highest_stage"] == 0
    assert result["measurements"]["simulation_time_s"] == before_time
    if task.startswith("whiteboard-"):
      assert result["measurements"]["contact_force_source"] == "integrated_solver_contacts_before_post_step_mj_forward"
      assert result["measurements"]["contact_interval_end_s"] == before_time
      assert result["measurements"]["opposed_grasp_now"] is not None


def test_runtime_update_rejects_simulation_switch():
  monitor = StagedEvaluationMonitor("bulb-screw")
  with pytest.raises(ValueError, match="between simulations"):
    monitor.update(object())


def test_runtime_update_without_simulation_fails_closed():
  monitor = StagedEvaluationMonitor("poker-draw")
  with pytest.raises(ValueError, match="requires a simulation"):
    monitor.update()


def test_nonfinite_measurements_are_persisted_as_null_not_nan():
  monitor = StagedEvaluationMonitor("usb-insert")
  result = observe_for(monitor, physical(near_mouth_distance_m=float("nan")), 0.2)
  assert result["measurements"]["near_mouth_distance_m"] is None
  assert not result["complete_success"]
  json.dumps(result, allow_nan=False)

