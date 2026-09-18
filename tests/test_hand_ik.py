from __future__ import annotations

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.workcell.simulation import (
  HAND_FINGERS,
  ArmHandSimulation,
)


@pytest.fixture(scope="module")
def simulation() -> ArmHandSimulation:
  # Genesis probes do not participate in kinematics and make this focused
  # test needlessly expensive to construct.
  return ArmHandSimulation(add_genesis_probes=False)


def _fingertips_at_hand_posture(
  simulation: ArmHandSimulation,
  side: str,
  joint_positions: np.ndarray,
  *,
  arm_joint_positions: np.ndarray | None = None,
) -> np.ndarray:
  qpos, qvel = simulation.full_state()
  if arm_joint_positions is not None:
    simulation.data.qpos[simulation._arm_qpos[side]] = arm_joint_positions
  simulation.data.qpos[simulation._hand_qpos[side]] = joint_positions
  thumb_joint5_index = simulation._thumb_joint5_index[side]
  simulation.data.qpos[simulation._thumb_joint6_qpos[side]] = joint_positions[
    thumb_joint5_index
  ]
  mujoco.mj_forward(simulation.model, simulation.data)
  positions = simulation.fingertip_positions(side)
  simulation.restore_full_state(qpos, qvel)
  return positions


@pytest.mark.parametrize("side", ("left", "right"))
def test_hand_ik_reaches_known_coupled_thumb_posture(
  simulation: ArmHandSimulation, side: str
) -> None:
  simulation.reset()
  # A sizeable jump exercises the deterministic thumb restart.  In FK the
  # passive joint6 is set equal to joint5, as it will be by the MJCF equality
  # during simulation.
  known_posture = np.array(
    [
      0.12,
      0.70,
      0.28,
      0.62,
      0.08,
      0.45,
      0.65,
      0.28,
      -0.06,
      0.55,
      0.72,
      0.24,
      0.04,
      0.50,
      0.62,
      0.31,
      -0.10,
      0.58,
      0.68,
      0.22,
    ]
  )
  targets = _fingertips_at_hand_posture(simulation, side, known_posture)

  result = simulation.solve_hand_ik(
    side,
    targets,
    max_iterations=100,
    position_tolerance=5.0e-4,
  )

  assert result.success
  assert result.joint_positions.shape == (20,)
  assert result.fingertip_errors.shape == (5,)
  assert result.position_error <= 5.0e-4
  assert tuple(name.split("_")[2] for name in result.joint_names[::4]) == HAND_FINGERS
  joint_ids = simulation._hand_joint_ids[side]
  assert np.all(result.joint_positions >= simulation.model.jnt_range[joint_ids, 0])
  assert np.all(result.joint_positions <= simulation.model.jnt_range[joint_ids, 1])
  achieved = _fingertips_at_hand_posture(simulation, side, result.joint_positions)
  np.testing.assert_allclose(
    np.linalg.norm(achieved - targets, axis=1),
    result.fingertip_errors,
    atol=1.0e-9,
  )


def test_set_fingertip_targets_uses_explicit_future_arm_posture(
  simulation: ArmHandSimulation,
) -> None:
  simulation.reset()
  side = "right"
  arm_posture = simulation.arm_goal[side]
  arm_posture[6] += 0.12
  hand_posture = np.array(
    [
      0.04,
      0.35,
      0.10,
      0.18,
      0.02,
      0.20,
      0.25,
      0.10,
      -0.01,
      0.22,
      0.28,
      0.12,
      0.01,
      0.18,
      0.24,
      0.08,
      -0.02,
      0.25,
      0.30,
      0.14,
    ]
  )
  targets = _fingertips_at_hand_posture(
    simulation,
    side,
    hand_posture,
    arm_joint_positions=arm_posture,
  )

  result = simulation.set_fingertip_targets(
    side,
    targets,
    arm_joint_positions=arm_posture,
    apply_best_effort=False,
    position_tolerance=7.5e-4,
  )

  assert result.success
  commanded = np.array(
    [simulation._hand_targets[side][name] for name in result.joint_names]
  )
  np.testing.assert_allclose(commanded, result.joint_positions)


def test_failed_strict_hand_ik_does_not_replace_previous_targets(
  simulation: ArmHandSimulation,
) -> None:
  simulation.reset()
  side = "left"
  original = simulation.command_state()[1][:20].copy()
  unreachable = simulation.fingertip_positions(side) + np.array([0.0, 0.0, 1.0])

  result = simulation.set_fingertip_targets(
    side,
    unreachable,
    apply_best_effort=False,
    max_iterations=10,
  )

  assert not result.success
  assert np.all(np.isfinite(result.joint_positions))
  joint_ids = simulation._hand_joint_ids[side]
  assert np.all(result.joint_positions >= simulation.model.jnt_range[joint_ids, 0])
  assert np.all(result.joint_positions <= simulation.model.jnt_range[joint_ids, 1])
  np.testing.assert_allclose(simulation.command_state()[1][:20], original)


@pytest.mark.parametrize(
  "targets",
  (
    np.zeros((4, 3)),
    np.zeros((5, 2)),
    np.full((5, 3), np.nan),
  ),
)
def test_hand_ik_rejects_invalid_targets(
  simulation: ArmHandSimulation, targets: np.ndarray
) -> None:
  with pytest.raises(ValueError, match="target_positions"):
    simulation.solve_hand_ik("left", targets)
