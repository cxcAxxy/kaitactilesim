"""Middle-pressure handover regressions with array-only simulation doubles."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.poker_draw.mid_full import (
  CONTACT_MODEL_VERSION,
  MID_CARD_CONTACT_IMPEDANCE,
  MID_FORCE_PER_FINGER_N,
  MID_FORCE_SETTINGS,
  MID_TABLE_CONTACT_IMPEDANCE,
  MidForcePokerExecutor,
  MidForcePokerSimulation,
  configure_middle_contact_impedance,
)
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (
  PressureWindowExecutor,
  PressureWindowSettings,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawExecutor


def _simulation() -> MidForcePokerSimulation:
  """Do not invoke a robot constructor, MuJoCo compiler, or renderer."""
  sim = object.__new__(MidForcePokerSimulation)
  sim.timestep = 0.002
  sim._arm_actuators = {"right": np.arange(2, 9), "left": np.asarray([0, 1])}
  sim._arm_goal = {"right": np.full(7, 1.7), "left": np.asarray([0.1, 0.2])}
  sim._arm_command = {"right": np.full(7, 1.5), "left": np.asarray([0.2, 0.3])}
  sim.data = SimpleNamespace(
    time=31.0,
    ctrl=np.asarray([0.7, 0.8, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]),
    qpos=np.linspace(-0.3, -0.1, 12),
    qvel=np.linspace(0.01, 0.12, 12),
    qfrc_applied=np.linspace(0.0, 0.11, 12),
  )
  sim.model = SimpleNamespace(
    actuator_ctrlrange=np.tile([-2.0, 2.0], (9, 1)),
    geom_friction=np.asarray([[1.0, 0.005, 0.0001], [1.4, 0.005, 0.0001]]),
    geom_solref=np.asarray([[0.01, 1.0], [0.01, 1.0]]),
    opt=SimpleNamespace(impratio=100.0, timestep=sim.timestep),
  )
  sim.drive_limit_n = 4.0
  sim._cartesian_drive = {
    "position": np.asarray([0.52, -0.16, 0.9]),
    "rotation": np.eye(3),
    "preload": np.asarray([0.0, 0.0, 2.0, 0.1, 0.0, 0.0]),
  }
  sim._drive_state = {"drive_actual_fx_n": -0.25, "drive_saturated": 0.0}
  return sim


def _physical_snapshot(sim: MidForcePokerSimulation) -> dict[str, object]:
  return {
    "time": sim.data.time,
    "ctrl": sim.data.ctrl.copy(),
    "qpos": sim.data.qpos.copy(),
    "qvel": sim.data.qvel.copy(),
    "qfrc_applied": sim.data.qfrc_applied.copy(),
    "ctrlrange": sim.model.actuator_ctrlrange.copy(),
    "friction": sim.model.geom_friction.copy(),
    "solref": sim.model.geom_solref.copy(),
    "impratio": sim.model.opt.impratio,
    "timestep": sim.model.opt.timestep,
  }


def _assert_same_snapshot(before: dict[str, object], after: dict[str, object]) -> None:
  assert before.keys() == after.keys()
  for key in before:
    np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_mid_preset_is_explicit_without_changing_legacy_defaults() -> None:
  assert MID_FORCE_PER_FINGER_N == 0.50
  assert MID_FORCE_SETTINGS.goal == "table-edge"
  assert MID_FORCE_SETTINGS.table_friction == 1.0
  assert MID_FORCE_SETTINGS.drive_limit_n == 4.0
  assert MID_FORCE_SETTINGS.slide_speed_m_s == 0.005
  assert MID_FORCE_SETTINGS.contact_time_constant_s == 0.010
  assert MID_FORCE_SETTINGS.contact_friction_impedance_ratio == 100.0
  assert PressureWindowSettings().goal == "short"
  assert PressureWindowSettings().table_friction == 1.30


def test_contact_compliance_is_pair_local_recorded_and_does_not_modify_state():
  sim = _simulation()
  sim.model.pair_solimp = np.tile([0.98, 0.995, 0.0005, 0.5, 2.0], (3, 1))
  sim.model.geom_solimp = sim.model.pair_solimp.copy()
  sim.model.pair = lambda name: SimpleNamespace(id=1)
  sim.model.geom = lambda name: SimpleNamespace(id=2)
  original_pairs = sim.model.pair_solimp.copy()
  original_geoms = sim.model.geom_solimp.copy()
  before = _physical_snapshot(sim)

  metadata = configure_middle_contact_impedance(sim)

  _assert_same_snapshot(before, _physical_snapshot(sim))
  np.testing.assert_array_equal(sim.model.pair_solimp[1], MID_TABLE_CONTACT_IMPEDANCE)
  np.testing.assert_array_equal(sim.model.geom_solimp[2], MID_CARD_CONTACT_IMPEDANCE)
  np.testing.assert_array_equal(sim.model.pair_solimp[[0, 2]], original_pairs[[0, 2]])
  np.testing.assert_array_equal(sim.model.geom_solimp[:2], original_geoms[:2])
  assert metadata["contact_model_version"] == CONTACT_MODEL_VERSION
  assert metadata["table_card_pair_solimp_original"] == original_pairs[1].tolist()
  assert metadata["card_geom_solimp_original"] == original_geoms[2].tolist()
  assert metadata["table_card_pair_solimp_used"] == list(MID_TABLE_CONTACT_IMPEDANCE)
  assert metadata["card_geom_solimp_used"] == list(MID_CARD_CONTACT_IMPEDANCE)
  assert metadata["tactile_temporal_smoothing"] is False
  sim.model.pair_solimp[1, 0] = 0.5
  assert metadata["table_card_pair_solimp_used"][0] == 0.90


def test_handover_preserves_executed_control_not_stale_goal_or_measured_q() -> None:
  sim = _simulation()
  before = _physical_snapshot(sim)
  old_goal = sim._arm_goal["right"].copy()
  control = sim.data.ctrl[sim._arm_actuators["right"]].copy()
  left_goal = sim._arm_goal["left"].copy()
  left_command = sim._arm_command["left"].copy()

  result = sim.end_cartesian_drive()

  np.testing.assert_array_equal(sim._arm_goal["right"], control)
  np.testing.assert_array_equal(sim._arm_command["right"], control)
  assert not np.array_equal(control, old_goal)
  assert not np.array_equal(control, sim.data.qpos[:7])
  np.testing.assert_array_equal(sim._arm_goal["left"], left_goal)
  np.testing.assert_array_equal(sim._arm_command["left"], left_command)
  _assert_same_snapshot(before, _physical_snapshot(sim))
  assert sim.drive_limit_n is None
  assert "_cartesian_drive" not in sim.__dict__
  assert "_drive_state" not in sim.__dict__
  assert sim.drive_state()["drive_actual_fx_n"] == 0.0
  assert result["preserved_actuator_control_rad"] == control.tolist()
  assert result["discarded_arm_goal_rad"] == old_goal.tolist()
  assert result["previous_drive_limit_n"] == 4.0
  assert result["previous_drive_state"]["drive_actual_fx_n"] == -0.25
  assert result["control_jump_rad"] == 0.0
  assert result["object_state_modified"] is False
  assert result["physics_parameters_modified"] is False
  np.testing.assert_array_equal(result["qpos_before"], before["qpos"])
  np.testing.assert_array_equal(result["qvel_before"], before["qvel"])
  assert result["maximum_qpos_change"] == 0.0
  assert result["maximum_qvel_change"] == 0.0


def test_handover_goal_command_and_metadata_are_independent_copies() -> None:
  sim = _simulation()
  original_control = sim.data.ctrl.copy()

  result = sim.end_cartesian_drive()
  sim._arm_goal["right"][0] = 1.0

  assert sim._arm_command["right"][0] == original_control[2]
  np.testing.assert_array_equal(sim.data.ctrl, original_control)
  assert result["preserved_actuator_control_rad"][0] == original_control[2]
  sim._arm_command["right"][1] = 1.0
  assert result["preserved_actuator_control_rad"][1] == original_control[3]
  recorded_qpos = result["qpos_before"][0]
  recorded_qvel = result["qvel_before"][0]
  sim.data.qpos[0] = 2.0
  sim.data.qvel[0] = 3.0
  assert result["qpos_before"][0] == recorded_qpos
  assert result["qvel_before"][0] == recorded_qvel


def test_handover_resumes_original_step_with_zero_initial_control_slew(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  sim = _simulation()
  actual_control = sim.data.ctrl[sim._arm_actuators["right"]].copy()
  calls = []

  def original_step(instance: object, steps: int = 1) -> None:
    assert instance is sim
    np.testing.assert_array_equal(sim._arm_goal["right"], actual_control)
    np.testing.assert_array_equal(sim._arm_command["right"], actual_control)
    np.testing.assert_array_equal(
      sim._arm_goal["right"] - sim._arm_command["right"], np.zeros(7)
    )
    calls.append(steps)

  monkeypatch.setattr(ArmHandSimulation, "step", original_step)
  sim.end_cartesian_drive()
  sim.step(3)

  assert calls == [3]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -2.001, 2.001])
def test_invalid_actual_control_rejects_handover_without_releasing_cap(
  invalid: float,
) -> None:
  sim = _simulation()
  sim.data.ctrl[4] = invalid
  before = _physical_snapshot(sim)
  old_goal = sim._arm_goal["right"].copy()
  old_command = sim._arm_command["right"].copy()
  cartesian = sim._cartesian_drive

  with pytest.raises(RuntimeError, match="invalid bounded servo command"):
    sim.end_cartesian_drive()

  assert sim.drive_limit_n == 4.0
  assert sim._cartesian_drive is cartesian
  assert sim._drive_state["drive_actual_fx_n"] == -0.25
  np.testing.assert_array_equal(sim._arm_goal["right"], old_goal)
  np.testing.assert_array_equal(sim._arm_command["right"], old_command)
  _assert_same_snapshot(before, _physical_snapshot(sim))


@pytest.mark.parametrize("limit,cartesian", [(None, True), (4.0, False), (None, False)])
def test_handover_requires_active_cartesian_drive(limit: float | None, cartesian: bool):
  sim = _simulation()
  sim.drive_limit_n = limit
  if not cartesian:
    del sim._cartesian_drive
  before = _physical_snapshot(sim)

  with pytest.raises(RuntimeError, match="requires an active Cartesian drive"):
    sim.end_cartesian_drive()

  _assert_same_snapshot(before, _physical_snapshot(sim))
  assert sim.drive_limit_n == limit


def test_handover_cannot_be_released_twice() -> None:
  sim = _simulation()
  sim.end_cartesian_drive()
  with pytest.raises(RuntimeError, match="requires an active Cartesian drive"):
    sim.end_cartesian_drive()


@pytest.mark.parametrize("bound", [-2.0, 2.0])
def test_exact_original_control_limit_is_preserved_without_clipping(
  bound: float,
) -> None:
  sim = _simulation()
  sim.data.ctrl[2:9] = bound

  result = sim.end_cartesian_drive()

  np.testing.assert_array_equal(sim._arm_goal["right"], np.full(7, bound))
  assert result["control_jump_rad"] == 0.0


def test_executor_constructor_rejects_other_simulation_types() -> None:
  with pytest.raises(TypeError, match="requires MidForcePokerSimulation"):
    MidForcePokerExecutor(SimpleNamespace())


@pytest.mark.parametrize("target", [0.02, 0.50, 1.20, None])
def test_executor_fixed_preset_rejects_pressure_override(target: float | None) -> None:
  with pytest.raises(ValueError, match="fixes pressure"):
    MidForcePokerExecutor(_simulation(), press_force_per_finger_n=target)


def test_executor_constructor_passes_only_fixed_pressure_to_parent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  received = {}
  sim = _simulation()
  observer = object()

  def parent_init(self: object, simulation: object, **kwargs: object) -> None:
    received.update(simulation=simulation, **kwargs)

  monkeypatch.setattr(PressureWindowExecutor, "__init__", parent_init)
  executor = MidForcePokerExecutor(sim, observer=observer)

  assert received == {
    "simulation": sim,
    "observer": observer,
    "press_force_per_finger_n": 0.50,
  }
  assert executor.handoff_outcome is None
  assert executor._lift_reference is None
  assert executor._lift_compensation == []


def test_full_metadata_declares_timing_handoff_and_unchanged_physics(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor = object.__new__(MidForcePokerExecutor)
  executor._lift_compensation = []
  parent = {
    "normal_controller": "PressureWindowForceController",
    "target_force_per_finger_n": 0.50,
    "object_motion_or_slip_used_for_control": False,
  }
  monkeypatch.setattr(PressureWindowExecutor, "control_metadata", lambda _self: parent)

  result = executor.control_metadata()

  assert result is not parent
  assert result["normal_controller"] == parent["normal_controller"]
  assert result["target_force_per_finger_n"] == 0.50
  assert parent["object_motion_or_slip_used_for_control"] is False
  assert result["preset"] == "middle_force_full_task_v1"
  assert result["object_motion_or_slip_used_for_control"] is True
  assert result["slip_feedback_used_for_pressure_adjustment"] is False
  assert result["lift_card_duration_scale"] == 2.0
  assert result["raise_card_to_view_duration_scale"] == 2.0
  assert result["turn_card_inward_duration_scale"] == 2.0
  assert result["geometry_usage"] == (
    "draw endpoint, supported edge retreat, and measured card/tool lever-arm "
    "compensation of pickup waypoints; never object-state writes"
  )
  assert result["lift_waypoint_compensation"] == []
  assert result["force_limit_scope"] == (
    "slide_card and edge_hold; disabled by explicit handover"
  )
  assert result["pickup_controller"] == "bounded tactile normal-force pinch v2"
  assert result["pinch_force_control"]["raw_tactile_filtered"] is False
  assert result["contact_model_scope"] == "unchanged throughout the whole episode"


@dataclass
class _Summary:
  sample_count: int = 250


def test_pinch_feedback_responds_to_force_even_when_geometric_contacts_exist():
  executor = object.__new__(MidForcePokerExecutor)
  executor._pinch_flexion_targets = np.ones(4)
  executor._pinch_thumb_joint5_target = 1.0
  executor._pinch_force_reference = None
  executor._pinch_force_filtered = None
  executor._pinch_force_steps = 0
  calls = []
  executor.sim = SimpleNamespace(
    timestep=0.002,
    set_hand_joint_targets=lambda names, values: calls.append(
      (names, np.asarray(values).copy())
    ),
  )
  fingers = ("index", "middle", "ring", "pinky")
  forces = dict(zip((*fingers, "thumb"), [0.0, 0.25, 1.0, 0.25, 2.0], strict=True))
  executor._current_card_face_contact_details = lambda: (
    set(fingers),
    {"thumb"},
    forces.copy(),
  )
  for _ in range(4):
    executor._maintain_dynamic_pinch(set(fingers), {"thumb"}, 1.5, 2.0)
  assert not calls
  executor._maintain_dynamic_pinch(set(fingers), {"thumb"}, 1.5, 2.0)
  assert executor._pinch_flexion_targets[0] > 1.0  # touching but unloaded
  assert executor._pinch_flexion_targets[1] == 1.0
  assert executor._pinch_flexion_targets[2] < 1.0  # excess squeeze releases
  assert executor._pinch_thumb_joint5_target < 1.0
  assert (
    np.max(abs(executor._pinch_flexion_targets - 1)) <= np.deg2rad(1.5) * 0.010 + 1e-12
  )
  for _ in range(3000):
    executor._maintain_dynamic_pinch(set(fingers), {"thumb"}, 1.5, 2.0)
  assert np.max(abs(executor._pinch_flexion_targets - 1)) <= np.deg2rad(1) + 1e-12
  assert abs(executor._pinch_thumb_joint5_target - 1) <= np.deg2rad(1) + 1e-12
  assert forces == dict(zip((*fingers, "thumb"), [0.0, 0.25, 1.0, 0.25, 2.0], strict=True))


def _executor_harness(
  monkeypatch: pytest.MonkeyPatch,
  *,
  outcome: object = None,
  qualified: bool = True,
  card_follows: bool = True,
) -> tuple[MidForcePokerExecutor, SimpleNamespace, SimpleNamespace]:
  sim = _simulation()
  executor = object.__new__(MidForcePokerExecutor)
  executor.sim = sim
  executor.handoff_outcome = {"completed": True, "stage": "previous_episode"}
  executor._lift_reference = (np.ones(3), np.ones(3))
  executor._lift_compensation = [{"previous_episode": True}]
  executor._press_controller_active = True
  executor._record_slide_force_metrics = False
  executor._slide_force_monitor = SimpleNamespace(summary=lambda: _Summary())
  state = SimpleNamespace(
    ee=np.asarray([0.52, -0.16, 0.9]),
    card=np.asarray([0.465 + 0.01 * 0.063, -0.16, 0.841, 1.0, 0.0, 0.0, 0.0]),
    draw_calls=[],
    samples=[],
    targets=[],
    quality_calls=[],
    after_step=None,
    time_scale=1.0,
  )
  plan = SimpleNamespace(
    side="right",
    table_edge_x=0.465,
    waypoints=(
      SimpleNamespace(phase="clear_card"),
      SimpleNamespace(phase="hover_card"),
      SimpleNamespace(phase="precontact_card"),
    ),
  )
  draw_result = {"full_slide_qualified": True} if outcome is None else outcome

  def draw(received_plan: object, settings: object) -> object:
    state.draw_calls.append((received_plan, settings))
    return draw_result

  def quality(summary: object) -> bool:
    state.quality_calls.append(summary)
    return qualified

  monkeypatch.setattr(executor, "draw", draw)
  monkeypatch.setattr(executor, "_slide_press_control_qualified", quality)
  monkeypatch.setattr(executor, "_projected_card_length_x", lambda: 0.063)
  monkeypatch.setattr(sim, "object_pose", lambda _name: state.card.copy())
  monkeypatch.setattr(
    sim, "current_pose_matrix", lambda _side: (state.ee.copy(), np.eye(3))
  )

  def step(phase: str, _edge: float) -> None:
    assert phase == "edge_hold"
    assert sim.drive_limit_n == 4.0
    assert executor._press_controller_active
    assert executor._record_slide_force_metrics
    target = sim._cartesian_drive["position"].copy()
    dx = target[0] - state.ee[0]
    if card_follows:
      state.card[0] += dx
    state.ee[:] = target
    sim.data.time += sim.timestep * state.time_scale
    if state.after_step is not None:
      state.after_step()
    state.samples.append((phase, sim.data.time, state.card.copy()))
    state.targets.append(target)

  monkeypatch.setattr(executor, "_step", step)
  return executor, plan, state


@pytest.mark.parametrize(
  "outcome",
  [{}, {"full_slide_qualified": False}, {"target_reached": True}, False],
)
def test_unqualified_edge_prohibits_pickup_and_keeps_cap_latched(
  monkeypatch: pytest.MonkeyPatch, outcome: object
) -> None:
  executor, plan, state = _executor_harness(monkeypatch, outcome=outcome)
  phases = []

  with pytest.raises(RuntimeError, match="pickup is prohibited"):
    executor._prepare_and_slide(plan, phases)

  assert executor.sim.drive_limit_n == 4.0
  assert "_cartesian_drive" in executor.sim.__dict__
  assert executor.handoff_outcome is None
  assert executor._lift_reference is None
  assert executor._lift_compensation == []
  assert executor._press_controller_active
  assert phases == []
  assert state.samples == []
  assert state.draw_calls == [(plan, MID_FORCE_SETTINGS)]


def test_draw_error_does_not_release_cap_or_reuse_previous_handoff(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _executor_harness(monkeypatch)

  def fail(*_args: object) -> None:
    raise RuntimeError("draw control failed")

  monkeypatch.setattr(executor, "draw", fail)
  with pytest.raises(RuntimeError, match="draw control failed"):
    executor._prepare_and_slide(plan, [])

  assert executor.sim.drive_limit_n == 4.0
  assert executor.handoff_outcome is None
  assert not state.samples


def test_supported_retreat_physically_moves_card_under_cap_before_handover(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _executor_harness(monkeypatch)
  original_card = state.card.copy()
  actual_control = executor.sim.data.ctrl[executor.sim._arm_actuators["right"]].copy()
  phases = []

  seed, position, edge_pose = executor._prepare_and_slide(plan, phases)

  assert state.draw_calls == [(plan, MID_FORCE_SETTINGS)]
  assert len(state.quality_calls) == 1
  assert state.card[0] > original_card[0]
  assert 0.44 <= executor._overhang_fraction(plan.table_edge_x) <= 0.48
  assert len(state.samples) > 50
  assert all(sample[0] == "edge_hold" for sample in state.samples)
  assert all(target[0] >= state.targets[0][0] for target in state.targets)
  np.testing.assert_array_equal(seed, actual_control)
  np.testing.assert_array_equal(position, state.ee)
  np.testing.assert_array_equal(edge_pose, state.card)
  assert executor.sim.drive_limit_n is None
  assert not executor._press_controller_active
  assert not executor._record_slide_force_metrics
  assert executor.handoff_outcome["completed"] is True
  assert executor.handoff_outcome["stage"] == "joint_servo_ready"
  assert executor.handoff_outcome["retreat_commanded_travel_m"] > 0.0
  assert executor.handoff_outcome["transition"]["control_jump_rad"] == 0.0
  assert phases == [
    "clear_card",
    "hover_card",
    "precontact_card",
    "four_finger_press",
    "slide_card",
    "edge_hold",
  ]


@pytest.mark.parametrize("time_scale", [1.0, 3.0])
def test_failed_physical_retreat_hits_travel_or_time_bound_without_releasing_cap(
  monkeypatch: pytest.MonkeyPatch, time_scale: float
) -> None:
  executor, plan, state = _executor_harness(monkeypatch, card_follows=False)
  state.time_scale = time_scale
  initial_card = state.card.copy()
  started = executor.sim.data.time

  with pytest.raises(RuntimeError, match="could not regain supported edge margin"):
    executor._prepare_and_slide(plan, [])

  assert executor.sim.drive_limit_n == 4.0
  assert executor._press_controller_active
  assert not executor._record_slide_force_metrics
  assert executor.handoff_outcome["completed"] is False
  assert executor.handoff_outcome["stage"] == "supported_edge_retreat"
  np.testing.assert_array_equal(state.card, initial_card)
  assert executor.sim.data.time - started <= 2.0 + executor.sim.timestep * time_scale
  assert state.targets[-1][0] - 0.52 <= 0.008 + 1e-12
  if time_scale == 1.0:
    assert state.targets[-1][0] - 0.52 == pytest.approx(0.008)


def test_post_retreat_force_quality_failure_cannot_release_drive(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _executor_harness(monkeypatch, qualified=False)

  with pytest.raises(RuntimeError, match="lost continuous force quality"):
    executor._prepare_and_slide(plan, [])

  assert state.samples
  assert executor.sim.drive_limit_n == 4.0
  assert executor._press_controller_active
  assert executor.handoff_outcome["completed"] is False
  assert not executor._record_slide_force_metrics


@pytest.mark.parametrize("overhang", [0.439, 0.481])
def test_rebound_outside_supported_margin_fails_handoff(
  monkeypatch: pytest.MonkeyPatch, overhang: float
) -> None:
  executor, plan, state = _executor_harness(monkeypatch)

  def after_step() -> None:
    # Change only the double's measured card geometry, after the inward path
    # has already reached its target and the settled hold is in progress.
    if len(state.samples) > 120:
      state.card[0] = plan.table_edge_x + (0.5 - overhang) * 0.063

  state.after_step = after_step
  with pytest.raises(RuntimeError, match="left the supported edge handover geometry"):
    executor._prepare_and_slide(plan, [])

  assert executor.sim.drive_limit_n == 4.0
  assert executor.handoff_outcome["completed"] is False
  assert not executor._record_slide_force_metrics


def test_full_execute_inherits_original_pickup_and_resets_only_before_single_draw(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  assert MidForcePokerExecutor.execute is PokerDrawExecutor.execute
  executor, plan, state = _executor_harness(monkeypatch)
  calls = []
  monkeypatch.setattr(
    executor, "_validate_plan", lambda _plan: calls.append("validate")
  )
  monkeypatch.setattr(
    executor, "_reset_episode_state", lambda _plan: calls.append("reset")
  )

  class PickupReached(Exception):
    pass

  def first_pickup(
    received_plan: object, target: np.ndarray, seed: np.ndarray, **kwargs: object
  ) -> None:
    assert received_plan is plan
    assert executor.handoff_outcome["completed"] is True
    assert executor.sim.drive_limit_n is None
    assert len(state.draw_calls) == 1
    np.testing.assert_allclose(target, state.ee + [0.0, 0.0, 0.015])
    np.testing.assert_array_equal(seed, executor.sim.arm_goal["right"])
    assert kwargs["phase"] == "thumb_face_press"
    calls.append("pickup")
    raise PickupReached

  monkeypatch.setattr(executor, "_move_pose_linear", first_pickup)
  with pytest.raises(PickupReached):
    executor.execute(plan)

  assert calls == ["validate", "reset", "pickup"]
  assert state.draw_calls == [(plan, MID_FORCE_SETTINGS)]


@pytest.mark.parametrize(
  "phase,factor", [("raise_card_to_view", 2.0), ("hold_card", 1.0)]
)
def test_guarded_view_raise_only_changes_duration_not_pose_or_guard(
  monkeypatch: pytest.MonkeyPatch, phase: str, factor: float
) -> None:
  executor = object.__new__(MidForcePokerExecutor)
  plan = object()
  position, rotation, seed = np.ones(3), np.eye(3), np.arange(7, dtype=float)
  received = {}
  returned_seed = np.full(7, 0.25)

  def original_guard(
    self: object,
    received_plan: object,
    target: np.ndarray,
    initial: np.ndarray,
    **kwargs: object,
  ) -> np.ndarray:
    assert self is executor
    assert received_plan is plan
    assert target is position
    assert initial is seed
    received.update(kwargs)
    return returned_seed

  monkeypatch.setattr(
    PokerDrawExecutor, "_move_pose_with_guarded_pinch", original_guard
  )
  result = executor._move_pose_with_guarded_pinch(
    plan, position, seed, duration=0.65, phase=phase, end_effector_rotation=rotation
  )

  assert result is returned_seed
  assert received["duration"] == pytest.approx(0.65 * factor)
  assert received["phase"] == phase
  assert received["end_effector_rotation"] is rotation
  assert PokerDrawExecutor._move_pose_with_guarded_pinch is original_guard


@pytest.mark.parametrize(
  "phase,factor", [("turn_card_inward", 2.0), ("lift_card", 1.0), ("inspect_card", 1.0)]
)
def test_inward_turn_only_changes_duration_not_joint_path(
  monkeypatch: pytest.MonkeyPatch, phase: str, factor: float
) -> None:
  executor = object.__new__(MidForcePokerExecutor)
  plan = object()
  target, seed = np.arange(7, dtype=float), np.full(7, 0.3)
  received = {}
  returned_seed = np.full(7, 0.25)

  def original_gesture(
    self: object,
    received_plan: object,
    final: np.ndarray,
    initial: np.ndarray,
    **kwargs: object,
  ) -> np.ndarray:
    assert self is executor
    assert received_plan is plan
    assert final is target
    assert initial is seed
    received.update(kwargs)
    return returned_seed

  monkeypatch.setattr(PokerDrawExecutor, "_move_arm_joints_linear", original_gesture)
  result = executor._move_arm_joints_linear(
    plan, target, seed, duration=0.85, phase=phase
  )

  assert result is returned_seed
  assert received == {"duration": pytest.approx(0.85 * factor), "phase": phase}
  assert PokerDrawExecutor._move_arm_joints_linear is original_gesture


def _rotate_y(angle_deg: float) -> np.ndarray:
  angle = np.deg2rad(angle_deg)
  c, s = np.cos(angle), np.sin(angle)
  return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _lift_harness(
  monkeypatch: pytest.MonkeyPatch,
  *,
  current_rotation: np.ndarray | None = None,
) -> tuple[MidForcePokerExecutor, SimpleNamespace, SimpleNamespace]:
  sim = _simulation()
  executor = object.__new__(MidForcePokerExecutor)
  executor.sim = sim
  executor._lift_reference = None
  executor._lift_compensation = []
  initial_rotation = np.eye(3) if current_rotation is None else current_rotation
  position = np.asarray([0.30, -0.16, 0.87])
  local_offset = np.asarray([0.147, 0.0, -0.02])
  state = SimpleNamespace(
    ee=position.copy(),
    rotation=initial_rotation.copy(),
    card=position + initial_rotation @ local_offset,
    calls=[],
    returned_seed=np.full(7, 0.25),
  )
  plan = SimpleNamespace(side="right")
  monkeypatch.setattr(
    sim, "current_pose_matrix", lambda _side: (state.ee.copy(), state.rotation.copy())
  )
  monkeypatch.setattr(
    sim, "object_pose", lambda _name: np.concatenate((state.card, [1.0, 0.0, 0.0, 0.0]))
  )

  def parent_guard(
    self: object,
    received_plan: object,
    target: np.ndarray,
    seed: np.ndarray,
    **kwargs: object,
  ) -> np.ndarray:
    assert self is executor
    assert received_plan is plan
    state.calls.append((target.copy(), seed, kwargs))
    return state.returned_seed

  monkeypatch.setattr(PokerDrawExecutor, "_move_pose_with_guarded_pinch", parent_guard)
  return executor, plan, state


@pytest.mark.parametrize("angle_deg", [0.0, 3.0, 4.0, 8.0, -8.0])
def test_lift_compensation_moves_card_up_instead_of_pivoting_it_into_table(
  monkeypatch: pytest.MonkeyPatch, angle_deg: float
) -> None:
  executor, plan, state = _lift_harness(monkeypatch)
  original_pose = state.ee.copy()
  original_card = state.card.copy()
  original_input = original_pose + [0.0, 0.0, 0.004]
  original_input_snapshot = original_input.copy()
  rotation = _rotate_y(angle_deg)
  physical_before = _physical_snapshot(executor.sim)
  seed = np.zeros(7)

  result = executor._move_pose_with_guarded_pinch(
    plan,
    original_input,
    seed,
    duration=0.26,
    phase="lift_card",
    end_effector_rotation=rotation,
  )

  command, returned_initial, kwargs = state.calls[0]
  local_offset = original_card - original_pose
  np.testing.assert_allclose(
    command + rotation @ local_offset, original_card + [0.0, 0.0, 0.004], atol=1e-14
  )
  if angle_deg > 0.0:
    assert command[2] > original_input[2]
  elif angle_deg == 0.0:
    np.testing.assert_allclose(command, original_input, atol=1e-14)
  assert result is state.returned_seed
  assert returned_initial is seed
  assert kwargs["duration"] == 0.52
  assert kwargs["phase"] == "lift_card"
  assert kwargs["end_effector_rotation"] is rotation
  np.testing.assert_array_equal(original_input, original_input_snapshot)
  np.testing.assert_array_equal(state.ee, original_pose)
  np.testing.assert_array_equal(state.card, original_card)
  _assert_same_snapshot(physical_before, _physical_snapshot(executor.sim))
  assert len(executor._lift_compensation) == 1
  record = executor._lift_compensation[0]
  np.testing.assert_allclose(
    record["desired_card_center_m"], original_card + [0, 0, 0.004]
  )
  np.testing.assert_allclose(record["compensated_tool_target_m"], command)
  np.testing.assert_array_equal(record["original_tool_target_m"], original_input)


def test_lift_offset_is_transformed_by_current_orientation_not_world_vector(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  current_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  executor, plan, state = _lift_harness(monkeypatch, current_rotation=current_rotation)
  target_rotation = _rotate_y(8.0) @ current_rotation
  desired_card = state.card + [0.0, 0.0, 0.025]
  local_offset = np.asarray([0.147, 0.0, -0.02])

  executor._move_pose_with_guarded_pinch(
    plan,
    state.ee + [0.0, 0.0, 0.025],
    np.zeros(7),
    duration=0.5,
    phase="lift_card",
    end_effector_rotation=target_rotation,
  )

  command = state.calls[0][0]
  np.testing.assert_allclose(command + target_rotation @ local_offset, desired_card)
  np.testing.assert_allclose(
    executor._lift_compensation[0]["measured_local_card_offset_m"], local_offset
  )


def test_second_lift_uses_original_reference_and_updated_measured_lever_arm(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _lift_harness(monkeypatch)
  origin_tool, origin_card = state.ee.copy(), state.card.copy()
  first_rotation, second_rotation = _rotate_y(4.0), _rotate_y(8.0)
  seed = np.zeros(7)
  executor._move_pose_with_guarded_pinch(
    plan,
    origin_tool + [0.0, 0.0, 0.004],
    seed,
    duration=0.26,
    phase="lift_card",
    end_effector_rotation=first_rotation,
  )
  # Emulate imperfect actual motion and a small physical slip before stage 2.
  state.ee[:] = state.calls[-1][0] + [0.001, 0.0, -0.0005]
  state.rotation[:] = first_rotation
  state.card[:] = origin_card + [0.002, 0.0, 0.003]
  measured_offset = state.rotation.T @ (state.card - state.ee)

  executor._move_pose_with_guarded_pinch(
    plan,
    origin_tool + [0.0, 0.0, 0.025],
    seed,
    duration=0.5,
    phase="lift_card",
    end_effector_rotation=second_rotation,
  )

  np.testing.assert_array_equal(executor._lift_reference[0], origin_tool)
  np.testing.assert_array_equal(executor._lift_reference[1], origin_card)
  target = state.calls[1][0]
  np.testing.assert_allclose(
    target + second_rotation @ measured_offset, origin_card + [0.0, 0.0, 0.025]
  )
  np.testing.assert_allclose(
    executor._lift_compensation[1]["measured_local_card_offset_m"], measured_offset
  )
  assert len(executor._lift_compensation) == 2


def test_excessive_lift_compensation_fails_before_any_motion_or_state_change(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _lift_harness(monkeypatch)
  physical_before = _physical_snapshot(executor.sim)
  card_before = state.card.copy()

  with pytest.raises(RuntimeError, match="exceeded 40 mm bound"):
    executor._move_pose_with_guarded_pinch(
      plan,
      state.ee + [0.0, 0.0, 0.004],
      np.zeros(7),
      duration=0.26,
      phase="lift_card",
      end_effector_rotation=_rotate_y(30.0),
    )

  assert state.calls == []
  assert executor._lift_compensation == []
  _assert_same_snapshot(physical_before, _physical_snapshot(executor.sim))
  np.testing.assert_array_equal(state.card, card_before)
