"""Analytical pressure-window and actuator-wrench limits; no robot is loaded."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.poker_draw.press_control import FourFingerForceController
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (
  ForceLimitedPokerSimulation,
  PressureWindowExecutor,
  PressureWindowForceController,
  PressureWindowSettings,
  ideal_pressure_window,
  limit_horizontal_wrench,
)


def _full_rank_jacobian() -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(20260908)
  left, _, right = np.linalg.svd(rng.normal(size=(6, 7)), full_matrices=True)
  jacobian = left @ np.diag([2.4, 2.0, 1.4, 1.0, 0.8, 0.5]) @ right[:6]
  return jacobian, right[6].copy()


def test_ideal_window_uses_total_four_finger_normal_force() -> None:
  mass, gravity = 0.00567872, 9.81
  lower, upper = ideal_pressure_window(1.25, 1.4, mass, gravity, 1.6)

  assert lower == pytest.approx(1.25 * mass * gravity / (1.4 - 1.25))
  assert upper == pytest.approx(1.6 / 1.25 - mass * gravity)
  assert lower / 4 == pytest.approx(0.11605884)
  assert upper / 4 == pytest.approx(0.3060729392)
  assert 4 * 0.06 < lower < 4 * 0.20 < upper < 4 * 0.40


def test_candidate_window_is_a_force_balance_not_an_outcome_prediction() -> None:
  table_mu, finger_mu, drive_limit = 1.25, 1.4, 1.6
  weight = 0.00567872 * 9.81
  light, middle, heavy = (4 * value for value in (0.06, 0.20, 0.40))

  assert finger_mu * light < table_mu * (light + weight)
  assert table_mu * (middle + weight) < min(finger_mu * middle, drive_limit)
  assert table_mu * (heavy + weight) > drive_limit


def test_zero_table_friction_has_no_positive_quasistatic_lower_or_upper_bound() -> None:
  assert ideal_pressure_window(0.0, 1.4, 0.005, 9.81, 1.6) == (0.0, np.inf)


@pytest.mark.parametrize(
  "table_mu,finger_mu,mass,gravity,drive_limit",
  [
    (1.4, 1.4, 0.005, 9.81, 1.6),
    (1.5, 1.4, 0.005, 9.81, 1.6),
    (1.25, 1.4, 0.005, 9.81, 0.01),
    (1.0, 2.0, 1.0, 1.0, 2.0),
  ],
)
def test_impossible_or_zero_width_window_is_explicitly_empty(
  table_mu: float,
  finger_mu: float,
  mass: float,
  gravity: float,
  drive_limit: float,
) -> None:
  assert ideal_pressure_window(table_mu, finger_mu, mass, gravity, drive_limit) == (
    np.inf,
    -np.inf,
  )


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -0.1])
def test_pressure_window_rejects_invalid_physical_parameters(
  index: int, invalid: float
) -> None:
  parameters = [1.25, 1.4, 0.00567872, 9.81, 1.6]
  parameters[index] = invalid
  with pytest.raises(ValueError):
    ideal_pressure_window(*parameters)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_non_table_parameters_must_be_strictly_positive(index: int) -> None:
  parameters = [1.25, 1.4, 0.00567872, 9.81, 1.6]
  parameters[index] = 0.0
  with pytest.raises(ValueError):
    ideal_pressure_window(*parameters)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_zero_table_friction_does_not_bypass_other_parameter_validation(
  index: int,
) -> None:
  parameters = [0.0, 1.4, 0.00567872, 9.81, 1.6]
  parameters[index] = np.nan
  with pytest.raises(ValueError):
    ideal_pressure_window(*parameters)


@pytest.mark.parametrize("requested_x", [-6.0, 6.0])
def test_wrench_limit_preserves_other_five_axes_and_seven_dof_nullspace(
  requested_x: float,
) -> None:
  jacobian, null_vector = _full_rank_jacobian()
  requested = np.asarray([requested_x, 0.8, 2.4, -0.3, 0.1, 0.6])
  null_torque = 0.73 * null_vector
  torque = jacobian.T @ requested + null_torque
  original_torque, original_jacobian = torque.copy(), jacobian.copy()

  limited_torque, measured_request, limited_wrench = limit_horizontal_wrench(
    jacobian, torque, 1.6
  )

  expected = requested.copy()
  expected[0] = np.copysign(1.6, requested_x)
  np.testing.assert_allclose(measured_request, requested, atol=1e-12)
  np.testing.assert_allclose(limited_wrench, expected, atol=1e-12)
  np.testing.assert_allclose(
    limited_torque, jacobian.T @ expected + null_torque, atol=1e-12
  )
  np.testing.assert_allclose(
    limited_torque - jacobian.T @ limited_wrench, null_torque, atol=1e-12
  )
  realized = np.linalg.lstsq(jacobian.T, limited_torque, rcond=None)[0]
  np.testing.assert_allclose(realized, expected, atol=1e-12)
  np.testing.assert_array_equal(torque, original_torque)
  np.testing.assert_array_equal(jacobian, original_jacobian)


@pytest.mark.parametrize("requested_x", [-1.6, -0.25, 0.0, 0.25, 1.6])
def test_torque_within_horizontal_limit_is_unchanged(requested_x: float) -> None:
  jacobian, null_vector = _full_rank_jacobian()
  wrench = np.asarray([requested_x, -0.7, 2.0, 0.3, 0.0, -0.4])
  torque = jacobian.T @ wrench + 0.3 * null_vector

  limited_torque, requested, limited = limit_horizontal_wrench(jacobian, torque, 1.6)

  np.testing.assert_allclose(limited_torque, torque, atol=1e-12)
  np.testing.assert_allclose(requested, wrench, atol=1e-12)
  np.testing.assert_allclose(limited, wrench, atol=1e-12)


def test_pure_nullspace_torque_does_not_spend_horizontal_force_budget() -> None:
  jacobian, null_vector = _full_rank_jacobian()
  torque = 20.0 * null_vector

  limited_torque, requested, limited = limit_horizontal_wrench(jacobian, torque, 0.1)

  np.testing.assert_allclose(limited_torque, torque, atol=1e-12)
  np.testing.assert_allclose(requested, np.zeros(6), atol=1e-12)
  np.testing.assert_allclose(limited, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("limit", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_wrench_limit_requires_finite_positive_limit(limit: float) -> None:
  jacobian, _ = _full_rank_jacobian()
  with pytest.raises(ValueError):
    limit_horizontal_wrench(jacobian, np.zeros(7), limit)


@pytest.mark.parametrize("shape", [(6, 6), (7, 6), (7, 7), (6,), (6, 7, 1)])
def test_wrench_limit_requires_six_by_seven_jacobian(shape: tuple[int, ...]) -> None:
  with pytest.raises(ValueError):
    limit_horizontal_wrench(np.zeros(shape), np.zeros(7), 1.6)


@pytest.mark.parametrize("shape", [(6,), (8,), (7, 1), (1, 7)])
def test_wrench_limit_requires_seven_torques(shape: tuple[int, ...]) -> None:
  jacobian, _ = _full_rank_jacobian()
  with pytest.raises(ValueError):
    limit_horizontal_wrench(jacobian, np.zeros(shape), 1.6)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["jacobian", "torque"])
def test_wrench_limit_rejects_nonfinite_inputs(invalid: float, target: str) -> None:
  jacobian, _ = _full_rank_jacobian()
  torque = np.zeros(7)
  if target == "jacobian":
    jacobian[2, 3] = invalid
  else:
    torque[3] = invalid
  with pytest.raises(ValueError):
    limit_horizontal_wrench(jacobian, torque, 1.6)


def test_wrench_limit_rejects_rank_deficient_jacobian() -> None:
  jacobian, _ = _full_rank_jacobian()
  jacobian[3] = jacobian[2]
  with pytest.raises(ValueError):
    limit_horizontal_wrench(jacobian, np.ones(7), 1.6)


def test_wrench_limit_rejects_ill_conditioned_full_rank_jacobian() -> None:
  jacobian = np.column_stack((np.diag([1.0] * 5 + [1e-7]), np.zeros(6)))
  assert np.linalg.matrix_rank(jacobian) == 6
  with pytest.raises(ValueError):
    limit_horizontal_wrench(jacobian, np.ones(7), 1.6)


def test_wrench_limit_accepts_well_conditioned_nondiagonal_jacobian() -> None:
  jacobian, _ = _full_rank_jacobian()
  assert np.linalg.cond(jacobian) < 5.0
  torque = jacobian.T @ np.asarray([-2.0, 0.2, 1.4, 0.1, -0.1, 0.2])

  limited_torque, _, limited = limit_horizontal_wrench(jacobian, torque, 1.6)

  assert limited[0] == pytest.approx(-1.6)
  assert np.all(np.isfinite(limited_torque))


def test_inactive_simulation_delegates_to_exact_production_step(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation = object.__new__(ForceLimitedPokerSimulation)
  calls: list[tuple[object, int]] = []
  monkeypatch.setattr(
    ArmHandSimulation,
    "step",
    lambda instance, steps=1: calls.append((instance, steps)),
  )

  simulation.step(3)

  assert calls == [(simulation, 3)]
  assert "model" not in simulation.__dict__


def _fake_force_limited_simulation(
  monkeypatch: pytest.MonkeyPatch,
) -> tuple[ForceLimitedPokerSimulation, np.ndarray, list[object]]:
  """Exercise the step wiring without compiling or stepping any MuJoCo model."""
  simulation = object.__new__(ForceLimitedPokerSimulation)
  count = 16
  arm_ids = {"left": np.arange(7), "right": np.arange(7, 14)}
  gain = np.zeros((count, 10))
  gain[:, 0] = 100.0
  bias = np.zeros((count, 10))
  bias[:, 1] = -100.0
  bias[:, 2] = -10.0
  gear = np.zeros((count, 6))
  gear[:, 0] = 1.0
  ctrlrange = np.tile((-3.0, 3.0), (count, 1))
  ctrlrange[14:] = (-0.02, 0.02)
  simulation.model = SimpleNamespace(
    nv=count,
    actuator_gainprm=gain,
    actuator_biasprm=bias,
    actuator_gear=gear,
    actuator_ctrlrange=ctrlrange,
    actuator_forcerange=np.tile((-108.0, 108.0), (count, 1)),
  )
  simulation.data = SimpleNamespace(
    qpos=np.zeros(count),
    qvel=np.linspace(-0.01, 0.01, count),
    qfrc_bias=np.linspace(0.0, 1.5, count),
    qfrc_applied=np.ones(count),
    qfrc_actuator=np.zeros(count),
    ctrl=np.zeros(count),
    site_xpos=np.asarray([[0.5, 0.16, 0.9], [0.5, -0.16, 0.9]]),
    site_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
  )
  simulation.ik_data = SimpleNamespace(
    qpos=np.zeros(count),
    site_xpos=simulation.data.site_xpos.copy(),
    site_xmat=simulation.data.site_xmat.copy(),
  )
  simulation._arm_dofs = arm_ids
  simulation._arm_qpos = arm_ids
  simulation._arm_actuators = arm_ids
  simulation._hand_actuators = {
    "left": {"left_finger": 14},
    "right": {"right_finger": 15},
  }
  simulation._hand_targets = {
    "left": {"left_finger": 1.0},
    "right": {"right_finger": -1.0},
  }
  simulation._qpos_address = {"left_finger": 14, "right_finger": 15}
  simulation._site_id = {"left": 0, "right": 1}
  simulation.arm_speed_limit = 3.0
  simulation.timestep = 0.002
  simulation.hand_position_gain = 24.0
  simulation.drive_limit_n = 1.6
  jacobian, null_vector = _full_rank_jacobian()
  wrench = np.asarray([-6.0, 0.8, 2.4, -0.3, 0.1, 0.6])
  torque = jacobian.T @ wrench + 0.73 * null_vector
  initial_right_control = (torque + 10.0 * simulation.data.qvel[7:14]) / 100.0
  simulation._arm_command = {
    "left": np.full(7, 0.2),
    "right": initial_right_control.copy(),
  }
  simulation._arm_goal = {
    "left": np.full(7, 0.8),
    "right": initial_right_control.copy(),
  }
  calls: list[object] = []

  def fake_kinematics(_model: object, data: object) -> None:
    assert data is simulation.ik_data
    calls.append("scratch_kinematics")

  def fake_com_pos(_model: object, data: object) -> None:
    assert data is simulation.ik_data
    calls.append("scratch_com_pos")

  def fake_jacobian(
    _model: object,
    data: object,
    position: np.ndarray,
    rotation: np.ndarray,
    site_id: int,
  ) -> None:
    assert data is simulation.ik_data
    assert site_id == 1
    position[:, 7:14] = jacobian[:3]
    rotation[:, 7:14] = jacobian[3:]

  def fake_step(_model: object, data: object) -> None:
    assert data is simulation.data
    calls.append("physics_step")
    simulation.data.qfrc_actuator[:14] = (
      100.0 * (simulation.data.ctrl[:14] - simulation.data.qpos[:14])
      - 10.0 * simulation.data.qvel[:14]
    )

  monkeypatch.setattr(mujoco, "mj_kinematics", fake_kinematics)
  monkeypatch.setattr(mujoco, "mj_comPos", fake_com_pos)
  monkeypatch.setattr(mujoco, "mj_jacSite", fake_jacobian)
  monkeypatch.setattr(mujoco, "mj_step", fake_step)
  return simulation, jacobian, calls


def test_active_step_limits_real_actuator_command_and_preserves_other_controls(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, jacobian, calls = _fake_force_limited_simulation(monkeypatch)
  initial_qpos = simulation.data.qpos.copy()
  initial_qvel = simulation.data.qvel.copy()

  simulation.step()

  assert calls == ["scratch_kinematics", "scratch_com_pos", "physics_step"]
  state = simulation.drive_state()
  assert state["drive_requested_fx_n"] == pytest.approx(-6.0)
  assert state["drive_limited_fx_n"] == pytest.approx(-1.6)
  assert state["drive_actual_fx_n"] == pytest.approx(-1.6)
  assert state["drive_saturated"] == 1.0
  assert state["drive_cap_error_n"] < 1e-12
  np.testing.assert_allclose(simulation.data.ctrl[:7], np.full(7, 0.206))
  np.testing.assert_allclose(simulation.data.ctrl[14:], (0.02, -0.02))
  np.testing.assert_allclose(
    simulation.data.qfrc_applied[:14], simulation.data.qfrc_bias[:14]
  )
  np.testing.assert_array_equal(simulation.data.qfrc_applied[14:], (0.0, 0.0))
  np.testing.assert_array_equal(simulation.data.qpos, initial_qpos)
  np.testing.assert_array_equal(simulation.data.qvel, initial_qvel)
  realized = np.linalg.lstsq(
    jacobian.T, simulation.data.qfrc_actuator[7:14], rcond=None
  )[0]
  np.testing.assert_allclose(realized, [-1.6, 0.8, 2.4, -0.3, 0.1, 0.6], atol=1e-12)
  state["drive_actual_fx_n"] = 123.0
  assert simulation.drive_state()["drive_actual_fx_n"] == pytest.approx(-1.6)


@pytest.mark.parametrize("range_name", ["actuator_ctrlrange", "actuator_forcerange"])
def test_active_step_rejects_original_actuator_limits_before_physics(
  monkeypatch: pytest.MonkeyPatch, range_name: str
) -> None:
  simulation, _, calls = _fake_force_limited_simulation(monkeypatch)
  getattr(simulation.model, range_name)[7:14] = (-1e-6, 1e-6)

  with pytest.raises(RuntimeError, match="original.*limits"):
    simulation.step()

  assert "physics_step" not in calls


@pytest.mark.parametrize("invalid_parameter", ["kp", "gear"])
def test_active_step_rejects_incompatible_servos(
  monkeypatch: pytest.MonkeyPatch, invalid_parameter: str
) -> None:
  simulation, _, calls = _fake_force_limited_simulation(monkeypatch)
  if invalid_parameter == "kp":
    simulation.model.actuator_gainprm[7, 0] = 0.0
  else:
    simulation.model.actuator_gear[7, 0] = 2.0

  with pytest.raises(RuntimeError, match="unit-gear position servos"):
    simulation.step()

  assert "physics_step" not in calls


def test_actual_actuator_force_cap_violation_is_not_silently_accepted(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, jacobian, _ = _fake_force_limited_simulation(monkeypatch)

  def unexpected_actuation(_model: object, _data: object) -> None:
    simulation.data.qfrc_actuator[7:14] = jacobian.T @ np.asarray(
      [-2.0, 0.8, 2.4, -0.3, 0.1, 0.6]
    )

  monkeypatch.setattr(mujoco, "mj_step", unexpected_actuation)

  with pytest.raises(RuntimeError, match="actual arm servo Fx exceeded"):
    simulation.step()

  assert simulation.drive_state()["drive_cap_error_n"] == pytest.approx(0.4)


def _recorded_wrench(state: dict[str, float], prefix: str) -> np.ndarray:
  return np.asarray(
    [state[f"drive_{prefix}_{component}_n"] for component in ("fx", "fy", "fz")]
    + [state[f"drive_{prefix}_{component}_nm"] for component in ("tx", "ty", "tz")]
  )


def test_cartesian_x_error_cannot_leak_into_pose_holding_wrench(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, jacobian, _ = _fake_force_limited_simulation(monkeypatch)
  start = simulation.begin_cartesian_drive(1.6)
  simulation.step()
  before = _recorded_wrench(simulation.drive_state(), "requested")
  preload = simulation._cartesian_drive["preload"].copy()
  simulation.set_cartesian_drive_position(start + [-0.03, 0.0, 0.0])
  # Mimic independently changing IK joint goals: their non-X wrench must not
  # sneak back into the Cartesian pose controller through the joint servo.
  simulation._arm_goal["right"] += np.asarray([0.02, -0.02, 0.01, 0, -0.02, 0, 0.01])

  simulation.step()

  state = simulation.drive_state()
  after = _recorded_wrench(state, "requested")
  assert after[0] - before[0] == pytest.approx(-800.0 * 0.03)
  np.testing.assert_allclose(after[1:], before[1:], atol=1e-12)
  np.testing.assert_allclose(
    _recorded_wrench(state, "actual")[1:], before[1:], atol=1e-12
  )
  np.testing.assert_array_equal(simulation._cartesian_drive["preload"], preload)
  assert state["drive_actual_fx_n"] == pytest.approx(-1.6)
  assert state["drive_pose_error_y_m"] == 0.0
  assert state["drive_pose_error_z_m"] == 0.0
  assert state["drive_rotation_error_rad"] == 0.0
  assert state["drive_ee_y_m"] == pytest.approx(-0.16)
  assert state["drive_ee_z_m"] == pytest.approx(0.9)

  raw_torque = (
    100.0 * (simulation._arm_command["right"] - simulation.data.qpos[7:14])
    - 10.0 * simulation.data.qvel[7:14]
  )
  raw_wrench = np.linalg.lstsq(jacobian.T, raw_torque, rcond=None)[0]
  expected_null = raw_torque - jacobian.T @ raw_wrench
  actual_null = simulation.data.qfrc_actuator[7:14] - jacobian.T @ _recorded_wrench(
    state, "actual"
  )
  np.testing.assert_allclose(actual_null, expected_null, atol=1e-12)


def test_cartesian_pose_error_has_independent_axis_stiffness(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, _, _ = _fake_force_limited_simulation(monkeypatch)
  start = simulation.begin_cartesian_drive(1.6)
  simulation._cartesian_drive["preload"] = np.zeros(6)
  simulation.data.qvel[:] = 0.0
  simulation.set_cartesian_drive_position(start + [0.0, 0.001, -0.001])
  angle = 0.01
  simulation._cartesian_drive["rotation"] = np.asarray(
    [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
  )

  simulation.step()

  state = simulation.drive_state()
  np.testing.assert_allclose(
    _recorded_wrench(state, "requested"), [0.0, 3.0, -8.0, 0.0, 0.0, 0.8], atol=1e-11
  )
  np.testing.assert_allclose(
    _recorded_wrench(state, "actual"), [0.0, 3.0, -8.0, 0.0, 0.0, 0.8], atol=1e-11
  )
  assert state["drive_pose_error_y_m"] == pytest.approx(0.001)
  assert state["drive_pose_error_z_m"] == pytest.approx(-0.001)
  assert state["drive_rotation_error_rad"] == pytest.approx(angle)


def test_cartesian_damping_opposes_each_measured_task_velocity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, jacobian, _ = _fake_force_limited_simulation(monkeypatch)
  simulation.begin_cartesian_drive(1.6)
  simulation._cartesian_drive["preload"] = np.zeros(6)
  task_velocity = np.asarray([0.01, -0.02, 0.003, 0.01, 0.0, -0.01])
  simulation.data.qvel[7:14] = np.linalg.lstsq(jacobian, task_velocity, rcond=None)[0]

  simulation.step()

  np.testing.assert_allclose(
    _recorded_wrench(simulation.drive_state(), "requested"),
    [-0.8, 2.4, -0.6, -0.08, 0.0, 0.08],
    atol=1e-12,
  )


def test_cartesian_target_and_initial_pose_are_not_aliased(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation, _, _ = _fake_force_limited_simulation(monkeypatch)
  start = simulation.begin_cartesian_drive(1.6)
  expected = start.copy()
  start[2] += 100.0
  np.testing.assert_array_equal(simulation._cartesian_drive["position"], expected)
  np.testing.assert_array_equal(simulation.data.site_xpos[1], expected)
  target = expected + [-0.03, 0.0, 0.0]
  simulation.set_cartesian_drive_position(target)
  target[2] += 100.0
  np.testing.assert_array_equal(
    simulation._cartesian_drive["position"], expected + [-0.03, 0.0, 0.0]
  )


def test_reset_releases_budget_and_clears_state_before_production_reset(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation = object.__new__(ForceLimitedPokerSimulation)
  simulation.drive_limit_n = 1.6
  simulation._cartesian_drive = {"position": np.ones(3)}
  simulation._drive_state = {"drive_actual_fx_n": -1.6}
  calls: list[dict[str, object]] = []

  def fake_reset(instance: ForceLimitedPokerSimulation, **kwargs: object) -> None:
    assert instance is simulation
    assert instance.drive_limit_n is None
    assert "_cartesian_drive" not in instance.__dict__
    assert "_drive_state" not in instance.__dict__
    calls.append(kwargs)

  monkeypatch.setattr(ArmHandSimulation, "reset", fake_reset)

  simulation.reset(seed=7, object_xy_jitter=0.0)

  assert calls == [{"seed": 7, "object_xy_jitter": 0.0}]
  assert simulation.drive_state()["drive_actual_fx_n"] == 0.0


@pytest.mark.parametrize("fail_during_draw", [False, True])
def test_draw_keeps_budget_latched_even_on_error_until_explicit_reset(
  monkeypatch: pytest.MonkeyPatch, fail_during_draw: bool
) -> None:
  simulation, _, _ = _fake_force_limited_simulation(monkeypatch)
  executor = object.__new__(PressureWindowExecutor)
  executor.sim = simulation
  start = simulation.data.site_xpos[1].copy()
  plan = SimpleNamespace(side="right", table_edge_x=0.465)
  monkeypatch.setattr(
    executor,
    "prepare",
    lambda _plan: (simulation._arm_goal["right"].copy(), start, np.eye(3)),
  )
  monkeypatch.setattr(
    simulation,
    "solve_ik",
    lambda *_args, **_kwargs: SimpleNamespace(
      success=True, joint_positions=simulation._arm_goal["right"].copy()
    ),
  )
  monkeypatch.setattr(
    simulation,
    "set_arm_joint_goal",
    lambda side, goal: simulation._arm_goal.__setitem__(side, goal.copy()),
  )
  phases: list[str] = []

  def fake_step(phase: str, _edge: float) -> None:
    assert simulation.drive_limit_n == settings.drive_limit_n
    phases.append(phase)
    if fail_during_draw:
      raise RuntimeError("deliberate calibration interruption")

  monkeypatch.setattr(executor, "_step", fake_step)
  monkeypatch.setattr(executor, "_advance_fixed", lambda *_args: None)
  settings = PressureWindowSettings(slide_distance_m=0.0002, slide_speed_m_s=0.01)

  if fail_during_draw:
    with pytest.raises(RuntimeError, match="deliberate calibration interruption"):
      executor.draw(plan, settings)
  else:
    executor.draw(plan, settings)

  assert phases
  assert simulation.drive_limit_n == settings.drive_limit_n
  assert "_cartesian_drive" in simulation.__dict__
  assert executor._record_slide_force_metrics is False


def _force_controller_parameters(**overrides: float) -> dict[str, float | int]:
  parameters = {
    "finger_count": 4,
    "target_force_n": 0.06,
    "timestep": 0.002,
    "update_period_s": 0.002,
    "filter_time_constant_s": 0.010,
    "integral_gain_rad_per_n_s": 0.02,
    "maximum_offset_rad": np.deg2rad(5.0),
    "maximum_offset_rate_rad_s": np.deg2rad(1.5),
    "contact_force_n": 0.015,
    "contact_recovery_rate_rad_s": 0.01,
    "force_deadband_n": 0.003,
  }
  parameters.update(overrides)
  return parameters


def test_momentary_zero_contact_does_not_increase_already_excessive_filtered_force() -> (
  None
):
  parameters = _force_controller_parameters()
  experimental = PressureWindowForceController(**parameters)
  production = FourFingerForceController(**parameters)
  experimental.reset(np.full(4, 0.30))
  production.reset(np.full(4, 0.30))

  experimental_offsets, updated = experimental.observe(np.zeros(4))
  production_offsets, _ = production.observe(np.zeros(4))

  assert updated
  assert np.all(experimental.filtered_forces_n > experimental.target_force_n)
  assert np.all(experimental_offsets < 0.0)
  # The accepted task still owns its original recontact rule. Only this
  # experiment removes that unconditional positive-seek contribution.
  assert np.all(production_offsets > 0.0)
  assert np.all(
    np.abs(experimental_offsets)
    <= experimental.maximum_offset_rate * experimental.timestep
  )


def test_underpressure_produces_bounded_downward_recovery_without_contact() -> None:
  controller = PressureWindowForceController(**_force_controller_parameters())
  controller.reset(np.zeros(4))

  offsets, updated = controller.observe(np.zeros(4))

  assert updated
  expected = (
    controller.integral_gain
    * (controller.target_force_n - controller.force_deadband_n)
    * controller.timestep
  )
  np.testing.assert_allclose(offsets, np.full(4, expected), atol=1e-15)
  assert np.all(offsets > 0.0)
  assert np.all(offsets <= controller.maximum_offset_rate * controller.timestep)


def test_persistent_unloading_eventually_recovers_after_stale_high_filter_decays() -> (
  None
):
  controller = PressureWindowForceController(**_force_controller_parameters())
  controller.reset(np.full(4, 0.30))
  deltas: list[float] = []
  previous = np.zeros(4)

  for _ in range(30):
    offsets, _ = controller.observe(np.zeros(4))
    deltas.append(float(offsets[0] - previous[0]))
    previous = offsets

  assert deltas[0] < 0.0
  assert deltas[-1] > 0.0
  assert np.all(controller.filtered_forces_n < controller.target_force_n)
  assert max(abs(value) for value in deltas) <= (
    controller.maximum_offset_rate * controller.timestep
  )


@pytest.mark.parametrize("measured_force,sign", [(0.0, 1), (10.0, -1)])
def test_signed_force_updates_respect_per_step_rate_limit_in_both_directions(
  measured_force: float, sign: int
) -> None:
  controller = PressureWindowForceController(
    **_force_controller_parameters(integral_gain_rad_per_n_s=100.0)
  )
  controller.reset(np.full(4, measured_force))

  offsets, updated = controller.observe(np.full(4, measured_force))

  assert updated
  np.testing.assert_allclose(
    offsets,
    np.full(4, sign * controller.maximum_offset_rate * controller.timestep),
    atol=1e-15,
  )


def test_force_controller_handles_each_finger_and_target_deadband_independently() -> (
  None
):
  controller = PressureWindowForceController(**_force_controller_parameters())
  forces = np.asarray([0.30, 0.0, 0.06, 0.061])
  controller.reset(forces)

  offsets, _ = controller.observe(forces)

  assert offsets[0] < 0.0
  assert offsets[1] > 0.0
  np.testing.assert_array_equal(offsets[2:], [0.0, 0.0])


def test_force_controller_update_period_uses_full_elapsed_time() -> None:
  controller = PressureWindowForceController(
    **_force_controller_parameters(update_period_s=0.010)
  )
  controller.reset(np.zeros(4))

  for _ in range(4):
    offsets, updated = controller.observe(np.zeros(4))
    assert updated is False
    np.testing.assert_array_equal(offsets, np.zeros(4))
  offsets, updated = controller.observe(np.zeros(4))

  assert updated is True
  expected = (
    controller.integral_gain
    * (controller.target_force_n - controller.force_deadband_n)
    * (controller.update_steps * controller.timestep)
  )
  np.testing.assert_allclose(offsets, np.full(4, expected))


@pytest.mark.parametrize("measured_force,sign", [(0.0, 1), (10.0, -1)])
def test_signed_force_controller_bounds_total_linkage_offset(
  measured_force: float, sign: int
) -> None:
  maximum_offset = 0.0001
  controller = PressureWindowForceController(
    **_force_controller_parameters(maximum_offset_rad=maximum_offset)
  )
  controller.reset(np.full(4, measured_force))

  for _ in range(100):
    offsets, _ = controller.observe(np.full(4, measured_force))
    assert np.all(np.abs(offsets) <= maximum_offset)

  np.testing.assert_allclose(offsets, np.full(4, sign * maximum_offset))


def test_force_controller_returns_offset_copy_and_reset_clears_integral() -> None:
  controller = PressureWindowForceController(**_force_controller_parameters())
  offsets, _ = controller.observe(np.zeros(4))
  expected = offsets.copy()
  offsets[:] = 100.0
  np.testing.assert_array_equal(controller.offsets_rad, expected)

  controller.reset(np.full(4, 0.06))

  np.testing.assert_array_equal(controller.offsets_rad, np.zeros(4))
  np.testing.assert_array_equal(controller.filtered_forces_n, np.full(4, 0.06))


@pytest.mark.parametrize(
  "forces",
  [np.zeros(3), np.zeros((4, 1)), np.asarray([0.06, np.nan, 0.06, 0.06]), -np.ones(4)],
)
def test_signed_force_controller_rejects_invalid_force_samples(
  forces: np.ndarray,
) -> None:
  controller = PressureWindowForceController(**_force_controller_parameters())
  with pytest.raises(ValueError):
    controller.observe(forces)


def _fake_press_establishment(
  monkeypatch: pytest.MonkeyPatch,
  sample: Callable[[int], tuple[float, float]],
) -> tuple[PressureWindowExecutor, list[float]]:
  executor = object.__new__(PressureWindowExecutor)
  executor.sim = SimpleNamespace(timestep=0.010)
  executor.press_force_per_finger_n = 0.06
  executor._press_controller = SimpleNamespace(filtered_forces_n=np.zeros(4))
  samples: list[float] = []
  forces = np.zeros(4)

  def fake_step(_phase: str, _edge: float) -> None:
    normal, filtered = sample(len(samples))
    forces[:] = normal
    executor._press_controller.filtered_forces_n[:] = filtered
    samples.append(normal)

  monkeypatch.setattr(executor, "_step", fake_step)
  monkeypatch.setattr(
    executor, "_current_card_finger_normal_forces", lambda: forces.copy()
  )
  return executor, samples


def test_light_pressure_establishment_rejects_old_absolute_tolerance_shortcut(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, samples = _fake_press_establishment(
    monkeypatch, lambda step: (0.025, 0.025) if step < 20 else (0.06, 0.06)
  )

  executor._stabilize_four_finger_press(SimpleNamespace(table_edge_x=0.465))

  assert len(samples) == 30
  np.testing.assert_allclose(
    executor._established_press_normal_forces_n, np.full(4, 0.06)
  )


@pytest.mark.parametrize("transient", ["filtered_error", "loss_of_load"])
def test_press_establishment_requires_consecutive_stable_samples(
  monkeypatch: pytest.MonkeyPatch, transient: str
) -> None:
  def sample(step: int) -> tuple[float, float]:
    if step < 9:
      return 0.065, 0.06
    if step == 9:
      return (0.06, 0.02) if transient == "filtered_error" else (0.0, 0.06)
    return 0.06, 0.06

  executor, samples = _fake_press_establishment(monkeypatch, sample)

  executor._stabilize_four_finger_press(SimpleNamespace(table_edge_x=0.465))

  assert len(samples) == 20
  # Samples before the interruption must not contaminate the reported preload.
  np.testing.assert_allclose(
    executor._established_press_normal_forces_n, np.full(4, 0.06)
  )


def test_invalid_low_pressure_establishment_times_out_without_starting_draw(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, samples = _fake_press_establishment(
    monkeypatch, lambda _step: (0.025, 0.025)
  )

  with pytest.raises(RuntimeError, match="relative-tolerance normal pressure"):
    executor._stabilize_four_finger_press(SimpleNamespace(table_edge_x=0.465))

  assert len(samples) == 1200
  assert "_established_press_normal_forces_n" not in executor.__dict__


def test_candidate_defaults_are_explicit_and_keep_optional_hardware_changes_disabled() -> (
  None
):
  assert asdict(PressureWindowSettings()) == {
    "table_friction": 1.30,
    "drive_limit_n": 4.0,
    "slide_distance_m": 0.030,
    "slide_speed_m_s": 0.005,
    "hold_seconds": 0.5,
    "contact_time_constant_s": 0.010,
    "contact_friction_impedance_ratio": 100.0,
    "finger_servo_velocity_gain": None,
    "physics_timestep_s": None,
    "goal": "short",
    "max_slide_time_s": 30.0,
    "max_slide_travel_m": 0.16,
    "target_overhang_fraction": 0.49,
    "edge_dwell_s": 0.10,
  }


@pytest.mark.parametrize("timestep", [None, 0.0005, 0.001, 0.002])
def test_optional_timestep_accepts_only_supported_bounded_overrides(
  timestep: float | None,
) -> None:
  assert (
    PressureWindowSettings(physics_timestep_s=timestep).physics_timestep_s == timestep
  )


@pytest.mark.parametrize("timestep", [0.0, -0.001, np.nan, np.inf, 0.00049, 0.00201])
def test_optional_timestep_rejects_invalid_or_out_of_bound_values(
  timestep: float,
) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(physics_timestep_s=timestep)


@pytest.mark.parametrize(
  "parameter", ["contact_time_constant_s", "finger_servo_velocity_gain"]
)
def test_optional_contact_or_finger_overrides_can_be_left_disabled(
  parameter: str,
) -> None:
  assert getattr(PressureWindowSettings(**{parameter: None}), parameter) is None


@pytest.mark.parametrize(
  "parameter", ["contact_time_constant_s", "finger_servo_velocity_gain"]
)
@pytest.mark.parametrize("invalid", [0.0, -0.01, np.nan, np.inf])
def test_optional_contact_or_finger_overrides_validate_when_supplied(
  parameter: str, invalid: float
) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(**{parameter: invalid})


@pytest.mark.parametrize("target_force", [0.015, 0.15, 0.35, 1.2])
@pytest.mark.parametrize("timestep", [0.0005, 0.002])
def test_control_metadata_exactly_reports_live_force_controller_values(
  target_force: float, timestep: float
) -> None:
  executor = object.__new__(PressureWindowExecutor)
  executor.press_force_per_finger_n = target_force
  controller = PressureWindowForceController(
    **_force_controller_parameters(
      target_force_n=target_force,
      timestep=timestep,
      update_period_s=0.010,
      integral_gain_rad_per_n_s=0.073,
      force_deadband_n=min(0.005, target_force * 0.05),
      contact_force_n=target_force * 0.25,
    )
  )
  executor._press_controller = controller

  metadata = executor.control_metadata()

  assert metadata == {
    "normal_controller": "PressureWindowForceController",
    "normal_feedback": "per-finger measured Fn; signed filtered error integral",
    "target_force_per_finger_n": controller.target_force_n,
    "timestep_s": controller.timestep,
    "update_steps": controller.update_steps,
    "filter_alpha": controller.filter_alpha,
    "integral_gain_rad_per_n_s": controller.integral_gain,
    "maximum_offset_rad": controller.maximum_offset,
    "maximum_offset_rate_rad_s": controller.maximum_offset_rate,
    "force_deadband_n": controller.force_deadband_n,
    "unconditional_contact_seek": False,
    "pause_on_low_normal_force": False,
    "establish_tolerance_n": max(0.005, 0.10 * target_force),
    "establish_stable_duration_s": 0.10,
    "establish_timeout_s": 12.0,
    "cartesian_axis_order": ["x", "y", "z", "rx", "ry", "rz"],
    "cartesian_stiffness_n_per_m_or_nm_per_rad": (
      800.0,
      3000.0,
      8000.0,
      80.0,
      80.0,
      80.0,
    ),
    "cartesian_damping_ns_per_m_or_nms_per_rad": (80.0, 120.0, 200.0, 8.0, 8.0, 8.0),
    "object_motion_or_slip_used_for_control": False,
  }
  assert metadata["integral_gain_rad_per_n_s"] == pytest.approx(0.073)
  json.dumps(metadata, allow_nan=False)
  metadata["cartesian_axis_order"][0] = "mutated"
  assert executor.control_metadata()["cartesian_axis_order"][0] == "x"
