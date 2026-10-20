from __future__ import annotations

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation


@pytest.fixture(scope="module")
def simulation() -> ArmHandSimulation:
  return ArmHandSimulation(add_genesis_probes=False)


def _fingertip_poses_at_hand_posture(
  simulation: ArmHandSimulation,
  side: str,
  joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  np.copyto(simulation.ik_data.qpos, simulation.data.qpos)
  np.copyto(simulation.ik_data.qvel, simulation.data.qvel)
  simulation.ik_data.qpos[simulation._arm_qpos[side]] = simulation._arm_goal[side]
  simulation.ik_data.qpos[simulation._hand_qpos[side]] = joint_positions
  thumb_joint5_index = simulation._thumb_joint5_index[side]
  simulation.ik_data.qpos[simulation._thumb_joint6_qpos[side]] = joint_positions[
    thumb_joint5_index
  ]
  mujoco.mj_forward(simulation.model, simulation.ik_data)
  site_ids = simulation._fingertip_site_ids[side]
  positions = simulation.ik_data.site_xpos[site_ids].copy()
  rotations = simulation.ik_data.site_xmat[site_ids].reshape(5, 3, 3).copy()
  return positions, rotations


def _rotation_errors(target: np.ndarray, actual: np.ndarray) -> np.ndarray:
  relative = target @ np.swapaxes(actual, -1, -2)
  cosine = np.clip(
    (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
    -1.0,
    1.0,
  )
  return np.arccos(cosine)


def test_pose_aware_ik_selects_distal_posture_and_coupled_thumb_branch(
  simulation: ArmHandSimulation,
) -> None:
  simulation.reset()
  side = "right"
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
  target_positions, target_rotations = _fingertip_poses_at_hand_posture(
    simulation, side, known_posture
  )
  seed = np.zeros(20)

  position_only = simulation.solve_hand_ik(
    side,
    target_positions,
    seed=seed,
    max_iterations=100,
  )
  pose_aware = simulation.solve_hand_ik(
    side,
    target_positions,
    target_rotations=target_rotations,
    seed=seed,
    max_iterations=100,
    orientation_weights=(0.02, 0.01, 0.01, 0.008, 0.006),
  )

  assert position_only.success
  assert pose_aware.success
  _, position_only_rotations = _fingertip_poses_at_hand_posture(
    simulation, side, position_only.joint_positions
  )
  achieved_positions, achieved_rotations = _fingertip_poses_at_hand_posture(
    simulation, side, pose_aware.joint_positions
  )
  position_only_orientation_errors = _rotation_errors(
    target_rotations, position_only_rotations
  )
  pose_orientation_errors = _rotation_errors(target_rotations, achieved_rotations)

  assert np.max(np.linalg.norm(target_positions - achieved_positions, axis=1)) < 1.0e-4
  assert np.max(pose_orientation_errors) < np.deg2rad(0.1)
  assert pose_orientation_errors[0] < 0.05 * position_only_orientation_errors[0]
  thumb_joint5 = simulation._thumb_joint5_index[side]
  assert abs(pose_aware.joint_positions[thumb_joint5] - known_posture[thumb_joint5]) < (
    0.05
    * abs(
      position_only.joint_positions[thumb_joint5]
      - known_posture[thumb_joint5]
    )
  )


@pytest.mark.parametrize(
  "rotations",
  (
    np.zeros((5, 3)),
    np.full((5, 3, 3), np.nan),
    np.tile(2.0 * np.eye(3), (5, 1, 1)),
    np.tile(np.diag([-1.0, 1.0, 1.0]), (5, 1, 1)),
  ),
)
def test_pose_aware_ik_rejects_invalid_rotations(
  simulation: ArmHandSimulation, rotations: np.ndarray
) -> None:
  simulation.reset()
  with pytest.raises(ValueError, match="target_rotations"):
    simulation.solve_hand_ik(
      "right",
      simulation.fingertip_positions("right"),
      target_rotations=rotations,
    )


@pytest.mark.parametrize(
  ("keyword", "weights"),
  (
    ("position_weights", 0.0),
    ("position_weights", (1.0, 1.0)),
    ("orientation_weights", -0.01),
    ("orientation_weights", (0.01, 0.01, np.nan, 0.01, 0.01)),
  ),
)
def test_pose_aware_ik_rejects_invalid_weights(
  simulation: ArmHandSimulation,
  keyword: str,
  weights: float | tuple[float, ...],
) -> None:
  simulation.reset()
  targets = simulation.fingertip_positions("right")
  with pytest.raises(ValueError, match=keyword):
    simulation.solve_hand_ik("right", targets, **{keyword: weights})


def test_coupled_thumb_reduced_rotational_jacobian_matches_finite_difference(
  simulation: ArmHandSimulation,
) -> None:
  simulation.reset()
  side = "right"
  posture = np.array(
    [
      0.10,
      0.60,
      0.20,
      0.45,
      *([0.0, 0.35, 0.45, 0.20] * 4),
    ]
  )
  _fingertip_poses_at_hand_posture(simulation, side, posture)
  thumb_site = int(simulation._fingertip_site_ids[side][0])
  thumb_joint5_index = simulation._thumb_joint5_index[side]
  thumb_joint5_dof = simulation._hand_dofs[side][thumb_joint5_index]
  thumb_joint6_dof = simulation._thumb_joint6_dof[side]
  jacobian_rotation = np.zeros((3, simulation.model.nv))
  mujoco.mj_jacSite(
    simulation.model,
    simulation.ik_data,
    None,
    jacobian_rotation,
    thumb_site,
  )
  reduced_jacobian_column = (
    jacobian_rotation[:, thumb_joint5_dof]
    + jacobian_rotation[:, thumb_joint6_dof]
  )

  epsilon = 1.0e-6
  joint5_qpos = simulation._hand_qpos[side][thumb_joint5_index]
  joint6_qpos = simulation._thumb_joint6_qpos[side]
  simulation.ik_data.qpos[joint5_qpos] += epsilon
  simulation.ik_data.qpos[joint6_qpos] += epsilon
  mujoco.mj_forward(simulation.model, simulation.ik_data)
  rotation_plus = simulation.ik_data.site_xmat[thumb_site].reshape(3, 3).copy()
  simulation.ik_data.qpos[joint5_qpos] -= 2.0 * epsilon
  simulation.ik_data.qpos[joint6_qpos] -= 2.0 * epsilon
  mujoco.mj_forward(simulation.model, simulation.ik_data)
  rotation_minus = simulation.ik_data.site_xmat[thumb_site].reshape(3, 3).copy()

  relative = rotation_plus @ rotation_minus.T
  finite_difference = np.array(
    [
      relative[2, 1] - relative[1, 2],
      relative[0, 2] - relative[2, 0],
      relative[1, 0] - relative[0, 1],
    ]
  ) / (4.0 * epsilon)
  np.testing.assert_allclose(
    reduced_jacobian_column,
    finite_difference,
    atol=1.0e-8,
    rtol=1.0e-6,
  )
