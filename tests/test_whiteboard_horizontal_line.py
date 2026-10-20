"""Independent eight-patch top bar, preserved Z defaults, physical completion."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.tasks.whiteboard_wipe import config as tilted_config
from kaihand_tactile_env.tasks.whiteboard_wipe.execution import WhiteboardWipeExecutor
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal import (
  config as horizontal_config,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
  HorizontalWhiteboardWipeSimulation,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line import config as C
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.execution import (
  HorizontalLineWhiteboardWipeExecutor,
  run_episode,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.task import (
  HorizontalLineWhiteboardWipeSimulation,
)


def test_line_scene_contains_only_original_top_eight_patches_and_same_workcell():
  sim = HorizontalLineWhiteboardWipeSimulation(add_genesis_probes=False)
  original = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  tilted = WhiteboardWipeSimulation(add_genesis_probes=False)
  assert sim.scene == C.SCENE_NAME
  assert sim.task_config is C
  assert C.INK_COUNT == len(sim.ink_ids) == 8
  assert horizontal_config.INK_COUNT == len(original.ink_ids) == 25
  assert tilted_config.INK_COUNT == len(tilted.ink_ids) == 25
  assert C.INK_PATTERN == "z_top_bar_v1"
  np.testing.assert_array_equal(
    sim._initial_ink_positions, original._initial_ink_positions[:8]
  )
  assert sim.model.body_jntnum[sim.model.body("whiteboard").id] == 0
  np.testing.assert_array_equal(
    sim.model.body("whiteboard").pos, original.model.body("whiteboard").pos
  )
  np.testing.assert_allclose(
    sim.data.xmat[sim.model.body("whiteboard").id].reshape(3, 3), np.eye(3)
  )
  for camera in SHARED_CAMERA_NAMES:
    for field in ("pos", "quat", "fovy"):
      np.testing.assert_array_equal(
        getattr(sim.model.camera(camera), field),
        getattr(original.model.camera(camera), field),
      )
  np.testing.assert_array_equal(sim.model.jnt_range, original.model.jnt_range)
  assert sim.model.neq == original.model.neq
  assert sim.cleaning.__class__ is original.cleaning.__class__
  pair = sim.model.pair("eraser_board_contact").id
  old_pair = original.model.pair("eraser_board_contact").id
  for field in ("pair_friction", "pair_solref", "pair_solimp", "pair_margin"):
    np.testing.assert_array_equal(
      getattr(sim.model, field)[pair], getattr(original.model, field)[old_pair]
    )
  ink_names = {
    sim.model.geom(i).name
    for i in range(sim.model.ngeom)
    if sim.model.geom(i).name.startswith("ink_")
  }
  assert ink_names == {f"ink_{i}" for i in range(8)}
  # Everything besides the explicit independent model name and omitted ink
  # geoms is the same XML: robot, eraser, board, cameras and contact settings.
  line_xml = ET.parse(default_model_path(C.SCENE_NAME)).getroot()
  z_xml = ET.parse(default_model_path(horizontal_config.SCENE_NAME)).getroot()
  line_xml.attrib.pop("model")
  z_xml.attrib.pop("model")
  for board in z_xml.iter("body"):
    for geom in list(board.findall("geom")):
      name = geom.get("name", "")
      if name.startswith("ink_") and int(name[4:]) >= 8:
        board.remove(geom)
  # Ignore indentation whitespace, which is not part of MJCF semantics.
  for root in (line_xml, z_xml):
    for element in root.iter():
      element.text = element.tail = None
  assert ET.tostring(line_xml) == ET.tostring(z_xml)
  assert model_fingerprint(default_model_path(C.SCENE_NAME)) != model_fingerprint(
    default_model_path(horizontal_config.SCENE_NAME)
  )
  sim.step(400)
  assert sim.table_force > 0.5
  assert sim.forces.read(sim.data).normal_force_n.sum() == 0
  np.testing.assert_array_equal(sim.remaining, np.ones(8))


def test_line_reset_is_reproducible_and_rejects_old_twenty_five_patch_layout():
  sim = HorizontalLineWhiteboardWipeSimulation(ink_seed=12, add_genesis_probes=False)
  positions = sim.model.geom_pos[sim.ink_ids].copy()
  for seed in range(4):
    sim.reset(ink_seed=seed)
    assert sim.remaining.shape == (8,)
    offset = np.asarray(sim.ink_randomization["center_offset_board_m"])
    assert np.all(abs(offset) <= C.INK_CENTER_RANGE_M)
    np.testing.assert_allclose(
      sim.model.geom_pos[sim.ink_ids] - sim._initial_ink_positions,
      np.tile(np.r_[offset, 0.0], (8, 1)),
    )
    np.testing.assert_array_equal(sim.remaining, np.ones(8))
    assert sim.ink_randomization["pattern"] == C.INK_PATTERN
    assert sim.data.time == 0
  sim.reset(ink_seed=12)
  np.testing.assert_array_equal(sim.model.geom_pos[sim.ink_ids], positions)
  with pytest.raises(ValueError, match=r"finite \[8, 3\] array"):
    sim.set_ink_layout(np.zeros((25, 3)))


@pytest.mark.parametrize("ink_y", [-0.06, 0.0, 0.06])
def test_line_path_has_one_loaded_bar_with_frame_safe_lead_in_and_out(ink_y):
  sim = HorizontalLineWhiteboardWipeSimulation(add_genesis_probes=False)
  expert = HorizontalLineWhiteboardWipeExecutor(sim)
  limit = C.BOARD_STROKE_CENTER_LIMIT_M
  start = np.array([0.07, min(0.100, limit - ink_y)])
  path = expert.build_wipe_path(start, ink_y=ink_y, stroke_center_limit=limit)
  assert len(path) == 2
  assert [segment[-1] for segment in path] == ["load_board", "wipe"]
  assert [segment[1] for segment in path] == [2.0, 5.5]
  np.testing.assert_array_equal(path[0][0], start)
  end = path[1][0]
  assert end[0] == start[0] == C.INK_LINE_VERTICES[0, 0]
  assert start[1] > C.INK_LINE_VERTICES[0, 1]
  assert end[1] < C.INK_LINE_VERTICES[1, 1]
  assert start[1] + ink_y <= limit + 1e-12
  assert end[1] + ink_y >= -limit - 1e-12
  assert path[0][4:6] == (0.0, C.BOARD_TARGET_FORCE_N)
  assert path[1][4:6] == (C.BOARD_TARGET_FORCE_N,) * 2


@pytest.mark.parametrize("task_config", [tilted_config, horizontal_config])
def test_extracting_wipe_builder_preserves_original_three_stroke_z(task_config):
  sim = WhiteboardWipeSimulation(task_config=task_config, add_genesis_probes=False)
  expert = WhiteboardWipeExecutor(sim)
  limit = getattr(task_config, "BOARD_STROKE_CENTER_LIMIT_M", 0.140)
  start = np.array([0.070, min(0.100, limit - 0.06)])
  path = expert.build_wipe_path(start, ink_y=0.06, stroke_center_limit=limit)
  assert len(path) == 4
  assert [segment[-1] for segment in path] == ["load_board", "wipe", "wipe", "wipe"]
  assert [segment[1] for segment in path] == [2.0, 5.5, 6.5, 5.5]
  np.testing.assert_array_equal(path[0][0], start)
  np.testing.assert_array_equal(path[1][0], task_config.INK_Z_VERTICES[1])
  np.testing.assert_array_equal(path[2][0], task_config.INK_Z_VERTICES[2])
  np.testing.assert_array_equal(path[3][0], [-0.070, max(-0.100, -limit - 0.06)])
  assert expert.position_above_ink_phase == "position_above_z"
  assert sim.remaining.shape == (25,)


@pytest.mark.parametrize(
  "ink_offset", [(0.0, 0.0), (-0.04, 0.06)], ids=["nominal", "north_west_corner"]
)
def test_line_full_actuator_episode_cleans_eight_patches_and_returns_home(ink_offset):
  sim = HorizontalLineWhiteboardWipeSimulation(add_genesis_probes=False)
  positions = sim._initial_ink_positions.copy()
  positions[:, :2] += ink_offset
  sim.set_ink_layout(positions)
  pre_steps = post_steps = 0
  maximum_loaded_line_x_error = 0.0
  prohibited_dofs = np.ones(sim.model.nv, dtype=bool)
  for dofs in sim._arm_dofs.values():
    prohibited_dofs[dofs] = False

  def assert_no_external_assistance(current):
    assert not np.any(current.data.xfrc_applied)
    assert not np.any(current.data.qfrc_applied[prohibited_dofs])

  def observe_pre(current):
    nonlocal pre_steps
    pre_steps += 1
    assert_no_external_assistance(current)

  def observe_post(current):
    nonlocal post_steps, maximum_loaded_line_x_error
    post_steps += 1
    assert_no_external_assistance(current)
    if current.phase == "wipe" and current.board_force > 0.05:
      tool_board = C.BOARD_ROTATION.T @ (
        current.object_pose("eraser")[:3] - current.ink_surface_center
      )
      maximum_loaded_line_x_error = max(
        maximum_loaded_line_x_error, abs(tool_board[0] - 0.07)
      )

  sim.physics_observer = observe_pre
  sim.post_physics_observer = observe_post
  result = run_episode(sim)
  assert result["success"], result
  assert result["pickup_verified"] and result["released_on_table"]
  assert result["home_return_verified"]
  assert result["pickup_sequence"]["valid"]
  assert result["tactile_validation"]["reasonable"]
  assert result["ink_count"] == len(result["ink_remaining"]) == 8
  assert result["wipe_stroke_count"] == 1
  assert result["wipe_pattern"] == C.INK_PATTERN
  assert result["mechanics_version"] == C.MECHANICS_VERSION
  np.testing.assert_array_equal(sim.remaining, np.zeros(8))
  assert result["maximum_lift_m"] >= 0.12
  assert result["peak_direct_hand_board_force_n"] < 1e-9
  assert result["right_arm_home_error_rad"] < 0.01
  assert result["right_hand_home_error_rad"] < C.OPEN_HAND_MAX_ERROR_RAD
  assert maximum_loaded_line_x_error < 0.005
  for key in ("restore_approach_motion", "return_home_motion"):
    assert result[key]["maximum_joint_speed_deg_s"] < 70.0
    assert result[key]["maximum_joint_acceleration_deg_s2"] < 100.0
  assert pre_steps == post_steps > 1000
  assert post_steps * sim.timestep == pytest.approx(result["elapsed_s"])
  assert result["solver_warnings"] == [0] * len(sim.data.warning)
  assert sim.table_force > 0.2
  assert sim.forces.read(sim.data).normal_force_n.sum() < 0.05
  assert np.linalg.norm(sim.object_pose("eraser")[:2] - sim.pickup_center[:2]) < 0.02
  phases = result["phases"]
  for phase in (
    "position_above_line",
    "load_board",
    "wipe",
    "leave_board",
    "return",
    "release",
    "restore_approach_branch",
    "home",
    "verify_home",
  ):
    assert phase in phases
  assert "position_above_z" not in phases
