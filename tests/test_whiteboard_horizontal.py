"""Independent horizontal scene, unchanged ink physics, actuator-only workflow."""

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.tasks.whiteboard_wipe import config as tilted_config
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal import config as C
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.execution import run_episode
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
  HorizontalWhiteboardWipeSimulation,
)


def test_horizontal_scene_is_independent_fixed_and_preserves_robot_and_cameras():
  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  tilted = WhiteboardWipeSimulation(add_genesis_probes=False)
  assert sim.scene == C.SCENE_NAME
  assert tilted.scene == tilted_config.SCENE_NAME
  assert sim.task_config is C
  assert tilted.task_config is tilted_config
  assert sim.object_names == ("eraser",)
  board = sim.model.body("whiteboard")
  assert sim.model.body_jntnum[board.id] == 0
  np.testing.assert_allclose(sim.data.xpos[board.id], C.BOARD_CENTER)
  np.testing.assert_allclose(sim.data.xmat[board.id].reshape(3, 3), np.eye(3))
  np.testing.assert_allclose(C.BOARD_NORMAL, [0, 0, 1])
  assert C.BOARD_SURFACE[2] == pytest.approx(0.700)
  np.testing.assert_allclose(
    tilted.data.xmat[tilted.model.body("whiteboard").id].reshape(3, 3),
    tilted_config.BOARD_ROTATION,
  )
  assert tilted_config.BOARD_NORMAL[0] < 0
  for name in SHARED_CAMERA_NAMES:
    for field in ("pos", "quat", "fovy"):
      np.testing.assert_array_equal(
        getattr(sim.model.camera(name), field),
        getattr(tilted.model.camera(name), field),
      )
  np.testing.assert_array_equal(sim.model.jnt_range, tilted.model.jnt_range)
  np.testing.assert_array_equal(
    sim.model.geom_pos[sim.ink_ids], tilted.model.geom_pos[tilted.ink_ids]
  )
  pair = sim.model.pair("eraser_board_contact").id
  old_pair = tilted.model.pair("eraser_board_contact").id
  for field in ("pair_friction", "pair_solref", "pair_solimp", "pair_margin"):
    np.testing.assert_array_equal(
      getattr(sim.model, field)[pair], getattr(tilted.model, field)[old_pair]
    )
  assert sim.cleaning.__class__ is tilted.cleaning.__class__
  assert sim.model.neq == tilted.model.neq
  assert sim.model.body("hand_l_base_link").id >= 0
  assert sim.model.body("hand_r_base_link").id >= 0
  geom_names = [sim.model.geom(i).name for i in range(sim.model.ngeom)]
  assert not any(name.startswith("board_support_") for name in geom_names)
  assert not any(name.startswith("board_foot_") for name in geom_names)
  assert model_fingerprint(default_model_path(C.SCENE_NAME)) != model_fingerprint(
    default_model_path(tilted_config.SCENE_NAME)
  )
  sim.step(400)
  assert sim.table_force > 0.5
  assert sim.forces.read(sim.data).normal_force_n.sum() == 0
  np.testing.assert_array_equal(sim.remaining, 1)


def test_horizontal_reset_randomizes_only_ink_and_does_not_mutate_tilted_config():
  sim = HorizontalWhiteboardWipeSimulation(ink_seed=12, add_genesis_probes=False)
  initial = sim.model.geom_pos[sim.ink_ids].copy()
  tilted_rotation = tilted_config.BOARD_ROTATION.copy()
  for seed in range(4):
    sim.reset(ink_seed=seed)
    offset = np.asarray(sim.ink_randomization["center_offset_board_m"])
    assert np.all(abs(offset) <= C.INK_CENTER_RANGE_M)
    np.testing.assert_allclose(
      sim.ink_surface_center, C.BOARD_SURFACE + np.r_[offset, 0.0]
    )
    np.testing.assert_array_equal(sim.remaining, 1)
    assert sim.data.time == 0
  sim.reset(ink_seed=12)
  np.testing.assert_array_equal(sim.model.geom_pos[sim.ink_ids], initial)
  np.testing.assert_array_equal(tilted_config.BOARD_ROTATION, tilted_rotation)


def test_horizontal_post_step_observer_is_read_only_and_preserves_pre_hook():
  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  reference = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  pre_times, post_times, board_forces = [], [], []
  sim.physics_observer = lambda current: pre_times.append(float(current.data.time))

  def observe_completed_step(current):
    post_times.append(float(current.data.time))
    board_forces.append(current.board_force)

  sim.post_physics_observer = observe_completed_step
  sim.step(40)
  reference.step(40)
  np.testing.assert_allclose(pre_times, np.arange(40) * sim.timestep)
  np.testing.assert_allclose(post_times, np.arange(1, 41) * sim.timestep)
  np.testing.assert_array_equal(sim.data.qpos, reference.data.qpos)
  np.testing.assert_array_equal(sim.data.qvel, reference.data.qvel)
  np.testing.assert_array_equal(sim.remaining, reference.remaining)
  assert sim.mean_board_force == pytest.approx(np.mean(board_forces))


@pytest.mark.parametrize("friction", [0.0, 0.65])
def test_horizontal_cleaning_requires_actual_loaded_friction_work(friction):
  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  center = C.BOARD_SURFACE - C.WIPE_ROTATION @ C.PAD_BOTTOM
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, C.WIPE_ROTATION.ravel())
  sim.set_object_pose("eraser", center - 0.0005 * C.BOARD_NORMAL, quat)
  pair = sim.model.pair("eraser_board_contact").id
  sim.model.pair_friction[pair, :2] = friction
  sim.data.xfrc_applied[sim.eraser_body, :3] = -3 * C.BOARD_NORMAL
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert sim.board_force > 1.5
  np.testing.assert_array_equal(sim.remaining, 1)  # Static pressure never cleans.
  sim.data.qvel[sim._object_dofs["eraser"] + 1] = 0.05
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert sim.patch_speed.max() == pytest.approx(0.05)
  if friction == 0:
    np.testing.assert_array_equal(sim.cleaning.work_j, 0)
    np.testing.assert_array_equal(sim.remaining, 1)
  else:
    assert sim.board_tangent_force > 0.1
    assert sim.cleaning.work_j.sum() > 0
    assert sim.remaining.min() < 1


@pytest.mark.parametrize(
  "ink_offset", [(0.0, 0.0), (-0.04, 0.06)], ids=["nominal", "north_west_corner"]
)
def test_horizontal_full_actuator_episode_cleans_replaces_and_releases(ink_offset):
  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  ink_positions = sim._initial_ink_positions.copy()
  ink_positions[:, :2] += ink_offset
  sim.set_ink_layout(ink_positions)
  physical_steps = completed_steps = 0

  def verify_no_eraser_assistance(current):
    np.testing.assert_array_equal(current.data.xfrc_applied[current.eraser_body], 0)
    np.testing.assert_array_equal(
      current.data.qfrc_applied[
        current._object_dofs["eraser"] : current._object_dofs["eraser"] + 6
      ],
      0,
    )

  def observe_physics(current):
    nonlocal physical_steps
    physical_steps += 1
    verify_no_eraser_assistance(current)

  def observe_completed_physics(current):
    nonlocal completed_steps
    completed_steps += 1
    verify_no_eraser_assistance(current)

  sim.physics_observer = observe_physics
  sim.post_physics_observer = observe_completed_physics
  result = run_episode(sim)
  assert result["success"], result
  assert result["home_return_verified"]
  assert result["right_arm_home_error_rad"] < 0.01
  assert result["right_hand_home_error_rad"] < C.OPEN_HAND_MAX_ERROR_RAD
  for key in ("restore_approach_motion", "return_home_motion"):
    assert result[key]["maximum_joint_speed_deg_s"] < 70.0
    assert result[key]["maximum_joint_acceleration_deg_s2"] < 100.0
  assert result["pickup_verified"] and result["released_on_table"]
  assert result["pickup_sequence"]["valid"]
  assert result["tactile_validation"]["reasonable"]
  assert result["maximum_lift_m"] >= 0.12
  assert result["peak_direct_hand_board_force_n"] < 1e-9
  assert physical_steps > 1000
  assert completed_steps == physical_steps
  assert completed_steps * sim.timestep == pytest.approx(result["elapsed_s"])
  assert result["solver_warnings"] == [0] * len(sim.data.warning)
  np.testing.assert_array_equal(sim.remaining, 0)
  assert sim.table_force > 0.2
  assert np.linalg.norm(sim.object_pose("eraser")[:2] - sim.pickup_center[:2]) < 0.02
  assert sim.forces.read(sim.data).normal_force_n.sum() < 0.05
  phases = result["phases"]
  for phase in (
    "lift",
    "load_board",
    "wipe",
    "leave_board",
    "return",
    "release",
    "open_for_home",
    "settle_home_hand",
    "restore_approach_branch",
    "home",
    "verify_home",
  ):
    assert phase in phases
