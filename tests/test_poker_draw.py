from __future__ import annotations

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.tasks.poker_draw.acceptance import (
  TASK_COMPLETION_POLICY,
  pressure_quality_label,
)
from kaihand_tactile_env.tasks.poker_draw.config import (
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
)
from kaihand_tactile_env.workcell.poker import PokerDrawExecutor, PokerDrawPlanner
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation


def _anatomical_palm_normal_z(
  simulation: ArmHandSimulation,
  side: str,
) -> float:
  body = simulation.model.body(f"hand_{side[0]}_base_link")
  rotation = simulation.data.xmat[body.id].reshape(3, 3)
  local_normal_sign = -1.0 if side == "left" else 1.0
  return float(local_normal_sign * rotation[2, 1])


def test_poker_scene_is_isolated_from_pick_place() -> None:
  pick_place = ArmHandSimulation(add_genesis_probes=False, scene="pick-place")
  poker = ArmHandSimulation(add_genesis_probes=False, scene="poker-draw")

  assert pick_place.object_pose("cylinder")[2] == pytest.approx(0.77)
  assert poker.object_pose("card")[2] == pytest.approx(0.84175)
  assert pick_place.object_names == ("cylinder",)
  assert poker.object_names == ("card",)
  for model, absent_bodies, absent_geoms in (
    (pick_place.model, ("card", "poker_table"), ("card_core_geom", "poker_table_top")),
    (poker.model, ("cylinder", "box"), ("cylinder_geom", "box_bottom")),
  ):
    for name in absent_bodies:
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    for name in absent_geoms:
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
  assert pick_place.model.geom("cylinder_geom").contype[0] != 0
  assert poker.model.geom("card_core_geom").contype[0] != 0
  for side in ("left", "right"):
    # All tasks now share the inward-facing idle palms; task orientation is
    # established during the physical approach, not during reset.
    assert _anatomical_palm_normal_z(pick_place, side) == pytest.approx(
      _anatomical_palm_normal_z(poker, side)
    )
  for geom_name in ("poker_table_base", "poker_table_top"):
    geom = poker.model.geom(geom_name)
    assert int(geom.contype[0]) & 1024
    assert int(geom.conaffinity[0]) & 1024
    assert int(geom.contype[0]) & 8192
    assert int(geom.conaffinity[0]) & 8192

  card = poker.model.geom("card_core_geom")
  expected_pad_names = {
    f"hand_{side}_{finger}_{link}_tactile_pad_col"
    for side in ("l", "r")
    for finger, link in (
      ("thumb", "link6"),
      ("index", "link4"),
      ("middle", "link4"),
      ("ring", "link4"),
      ("pinky", "link4"),
    )
  }
  colliding_hand_geoms: set[str] = set()
  for geom_id in range(poker.model.ngeom):
    body_name = poker.model.body(int(poker.model.geom_bodyid[geom_id])).name or ""
    if not body_name.startswith("hand_"):
      continue
    geom = poker.model.geom(geom_id)
    can_collide = bool(
      int(card.contype[0]) & int(geom.conaffinity[0])
      or int(geom.contype[0]) & int(card.conaffinity[0])
    )
    if can_collide:
      colliding_hand_geoms.add(geom.name)
  assert colliding_hand_geoms == expected_pad_names
  for forbidden_equality in (
    "right_card_grasp_stabilizer",
    "card_edge_support",
  ):
    assert (
      mujoco.mj_name2id(
        poker.model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        forbidden_equality,
      )
      == -1
    )
  card_body_id = poker.model.body("card").id
  for equality_id in range(poker.model.neq):
    equality_type = int(poker.model.eq_type[equality_id])
    if equality_type not in {
      int(mujoco.mjtEq.mjEQ_CONNECT),
      int(mujoco.mjtEq.mjEQ_WELD),
    }:
      continue
    assert card_body_id not in {
      int(poker.model.eq_obj1id[equality_id]),
      int(poker.model.eq_obj2id[equality_id]),
    }
  assert int(poker.model.joint("card_freejoint").type[0]) == int(
    mujoco.mjtJoint.mjJNT_FREE
  )


def test_poker_draw_reaches_edge_contacts_face_and_lifts() -> None:
  """Validate the tip-side tactile draw and unchanged physical pinch/view."""
  simulation = ArmHandSimulation(add_genesis_probes=False, scene="poker-draw")
  simulation.reset(
    seed=0,
    object_xy_jitter=0.0,
    object_yaw_jitter=0.0,
    randomized_objects=("card",),
  )

  plan = PokerDrawPlanner(simulation).plan()
  observed: dict[str, np.ndarray] = {}
  turn_joint_positions: list[np.ndarray] = []
  press_samples: dict[str, list[np.ndarray]] = {"slide_card": [], "edge_hold": []}
  draw_finger_qpos: list[np.ndarray] = []
  draw_thumb_qpos: list[float] = []
  tactile = SolverDistributedTactileProvider(simulation.model)
  tactile_sample = None
  draw_joint_ids = np.array(
    [
      [
        simulation.model.joint(f"hand_r_{finger}_joint{joint}").qposadr[0]
        for joint in (2, 3)
      ]
      for finger in ("index", "middle", "ring", "pinky")
    ]
  )
  thumb_tip_joint = simulation.model.joint("hand_r_thumb_joint5").qposadr[0]

  def observe(sim: ArmHandSimulation, phase: str) -> None:
    nonlocal tactile_sample
    if phase in {
      "ready_card",
      "hover_card",
      "precontact_card",
      "four_finger_press",
      "slide_card",
    }:
      assert _anatomical_palm_normal_z(sim, "right") < -0.95
    if phase in press_samples:
      press_samples[phase].append(executor._current_card_finger_normal_forces())
    if phase == "slide_card":
      if tactile_sample is None:
        tactile_sample = tactile.read(sim.data)
      observed["last_slide"] = sim.object_pose("card")
      draw_finger_qpos.append(sim.data.qpos[draw_joint_ids].copy())
      draw_thumb_qpos.append(float(sim.data.qpos[thumb_tip_joint]))
    elif phase == "edge_hold" and "first_edge_hold" not in observed:
      observed["first_edge_hold"] = sim.object_pose("card")
    elif phase == "raise_card_to_view" and "first_raise_card" not in observed:
      observed["first_raise_card"] = sim.object_pose("card")
    elif phase == "turn_card_inward":
      if "first_turn_card" not in observed:
        observed["first_turn_card"] = sim.object_pose("card")
      turn_joint_positions.append(sim.data.qpos[sim._arm_qpos["right"]].copy())
    elif phase == "inspect_card":
      observed["last_inspect_card"] = sim.object_pose("card")

  executor = PokerDrawExecutor(
    simulation, observer=observe, acceptance_policy=TASK_COMPLETION_POLICY
  )
  result = executor.execute(plan)

  assert result.success
  assert result.task_completed
  assert result.acceptance_policy == TASK_COMPLETION_POLICY
  assert result.draw_posture_version == "tip-pad-v7"
  assert (
    result.slide_fingertip_plane_angle_limit_degrees
    == _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
  )
  assert result.pressure_quality == pressure_quality_label(
    True, result.slide_press_control_qualified
  )
  assert result.draw_fingers_contacted == ("index", "middle", "ring", "pinky")
  assert result.simultaneous_four_finger_contact
  assert result.press_force_target_per_finger_n == pytest.approx(0.35)
  assert min(result.slide_finger_contact_fractions) >= 0.98
  assert max(result.slide_finger_maximum_contact_gaps_s) <= 0.02
  # The small edge correction is still part of pressing/sliding.  Test its
  # physical forces separately: brief tip-contact fluctuations are allowed,
  # but a whole phase with lost/weak tactile contact must not be hidden.
  for phase, samples in press_samples.items():
    assert len(samples) > 1, phase
    assert np.mean(np.all(np.asarray(samples) >= 0.25 * 0.35, axis=1)) >= 0.95, phase
  np.testing.assert_allclose(result.slide_finger_normal_force_means_n, 0.35, atol=0.035)
  assert min(result.slide_finger_target_band_fractions) >= 0.90
  assert result.slide_four_finger_target_band_fraction >= 0.85
  assert (
    30.0
    < result.maximum_slide_fingertip_plane_angle_degrees
    <= _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
  )
  assert tactile_sample is not None
  right_fingers = [
    tactile_sample.link_names.index(f"hand_r_{f}_link4")
    for f in ("index", "middle", "ring", "pinky")
  ]
  assert np.all(tactile_sample.normal_force_n[right_fingers] > 0.1)
  assert np.all(
    tactile_sample.normal_taxel_force_n[right_fingers].max(axis=(1, 2)) > 0.02
  )
  # Verify the measured joints throughout dragging, not only configured goals:
  # visible proximal flexion plus a curled thumb, while the pad-angle/contact
  # assertions above require real tactile-pad contact even near the tip.
  finger_degrees = np.rad2deg(np.asarray(draw_finger_qpos))
  assert np.all(finger_degrees[:, :, 0] > 40.0)
  assert np.all(finger_degrees[:, :, 1] > 4.0)
  assert min(np.rad2deg(draw_thumb_qpos)) > 10.0
  assert result.half_overhang_reached
  assert result.maximum_overhang_fraction >= 0.49
  assert result.thumb_face_contact
  assert result.sustained_pinch
  assert result.lift_opposition_fraction >= 0.95
  assert result.lift_four_finger_fraction > 0.05
  assert result.maximum_lift_opposition_gap <= 0.03
  assert result.hold_opposition_fraction >= 0.95
  assert result.hold_four_finger_fraction > 0.30
  assert result.maximum_preinspection_card_tilt_degrees < 35.0
  assert result.inspection_rotation_degrees > 80.0
  assert result.wrist_inward_turn_degrees > 80.0
  assert result.inspection_face_alignment >= 0.80
  assert result.inspection_face_robot_alignment >= 0.80
  assert result.inspection_position_error < 0.03
  assert result.inspection_opposition_fraction >= 0.90
  assert result.inspection_four_finger_fraction >= 0.55
  assert result.maximum_inspection_opposition_gap <= 0.03
  assert result.inspection_hold_opposition_fraction >= 0.95
  assert result.inspection_hold_four_finger_fraction >= 0.95
  assert result.minimum_inspection_card_height >= result.edge_card_pose[2] + 0.005
  assert result.terminal_pinch
  assert result.terminal_grip_fingers == ("index", "middle", "ring", "pinky")
  assert result.terminal_thumb_normal_force >= 0.05
  assert result.terminal_finger_normal_force >= 0.05
  assert result.minimum_terminal_finger_pad_alignment >= 0.94
  assert result.terminal_thumb_pad_alignment >= 0.84
  assert result.maximum_terminal_fingertip_angle_to_card_plane_degrees <= 21.0
  assert result.retained_at_end
  assert result.final_card_pose[2] > result.edge_card_pose[2] + 0.08
  # The curved draw uses a tilted but still downward-facing palm.
  # Full-episode maximum includes the inward-facing shared home and rotation.
  assert np.isfinite(result.maximum_palm_normal_z)
  assert executor._maximum_task_palm_normal_z < -0.95
  assert result.toward_robot_displacement > 0.08
  assert result.edge_card_pose[0] < result.initial_card_pose[0]
  assert result.lateral_card_displacement < 0.015
  assert result.minimum_supported_card_clearance >= -0.0006
  assert result.minimum_card_back_clearance > 0.0
  assert result.maximum_card_tilt_degrees > 80.0
  head_camera_id = simulation.model.camera("head").id
  card_in_head_camera = simulation.data.cam_xmat[head_camera_id].reshape(3, 3).T @ (
    result.final_card_pose[:3] - simulation.data.cam_xpos[head_camera_id]
  )
  assert card_in_head_camera[2] < -0.30
  assert abs(card_in_head_camera[0]) < 0.20
  # Check the actual shared 70-degree frustum, not an optical-axis offset
  # hard-coded for the historical camera. All card corners need a 10% margin;
  # use a square frustum, conservative for the landscape training images.
  card_geom = simulation.model.geom("card_core_geom")
  corners_local = np.array(
    [
      [x, y, z]
      for x in (-card_geom.size[0], card_geom.size[0])
      for y in (-card_geom.size[1], card_geom.size[1])
      for z in (-card_geom.size[2], card_geom.size[2])
    ]
  )
  corners_world = (
    corners_local @ simulation.data.geom_xmat[card_geom.id].reshape(3, 3).T
    + simulation.data.geom_xpos[card_geom.id]
  )
  corners_camera = (
    corners_world - simulation.data.cam_xpos[head_camera_id]
  ) @ simulation.data.cam_xmat[head_camera_id].reshape(3, 3)
  assert np.all(corners_camera[:, 2] < -0.30)
  half_extent = -corners_camera[:, 2] * np.tan(
    np.deg2rad(simulation.model.cam_fovy[head_camera_id] / 2.0)
  )
  assert np.all(np.abs(corners_camera[:, :2]) < 0.9 * half_extent[:, None])
  # Force establishment and smooth 15 mm/s sliding replace the old faster
  # uncontrolled preload trajectory.  Keep a bounded deterministic runtime.
  assert simulation.data.time < 30.0
  assert observed.keys() >= {
    "last_slide",
    "first_edge_hold",
    "first_raise_card",
    "first_turn_card",
    "last_inspect_card",
  }
  assert (
    np.linalg.norm(observed["first_edge_hold"][:3] - observed["last_slide"][:3]) < 0.002
  )
  assert result.phases == (
    "clear_card",
    "ready_card",
    "hover_card",
    "precontact_card",
    "four_finger_press",
    "slide_card",
    "edge_hold",
    "thumb_face_press",
    "lift_card",
    "hold_card",
    "raise_card_to_view",
    "turn_card_inward",
    "inspect_card",
  )
  np.testing.assert_allclose(
    result.preinspection_card_pose,
    observed["first_raise_card"],
    atol=0.004,
  )
  np.testing.assert_allclose(
    result.final_card_pose,
    observed["last_inspect_card"],
    atol=0.004,
  )
  assert len(turn_joint_positions) > 10
  turn_joints = np.asarray(turn_joint_positions)
  coordinated_spans_degrees = np.rad2deg(np.ptp(turn_joints[:, :6], axis=0))
  assert np.count_nonzero(coordinated_spans_degrees > 5.0) >= 5
  assert np.linalg.norm(coordinated_spans_degrees) > 30.0
  assert np.rad2deg(turn_joints[0, 6] - turn_joints[-1, 6]) > 50.0
  terminal_contact_geoms = {
    simulation.model.geom(
      int(
        contact.geom2
        if contact.geom1 == simulation.model.geom("card_core_geom").id
        else contact.geom1
      )
    ).name
    for contact in simulation.data.contact
    if simulation.model.geom("card_core_geom").id
    in (int(contact.geom1), int(contact.geom2))
  }
  assert "hand_r_thumb_link6_tactile_pad_col" in terminal_contact_geoms
  assert {
    f"hand_r_{finger}_link4_tactile_pad_col"
    for finger in ("index", "middle", "ring", "pinky")
  }.issubset(terminal_contact_geoms)
  np.testing.assert_allclose(
    result.final_card_pose,
    simulation.object_pose("card"),
  )


def test_raised_palm_keeps_fingertip_support_height() -> None:
  """Raise palm/wrist visibly from v6, keeping distal pad contact possible."""
  from kaihand_tactile_env.tasks.poker_draw import config
  from kaihand_tactile_env.tasks.poker_draw.task import _tilt_in_world_xy

  sim = ArmHandSimulation(scene="poker-draw", add_genesis_probes=False)
  home_ee_rotation = sim.current_pose_matrix("right")[1]
  home_hand_rotation = sim.data.xmat[sim.model.body("hand_r_base_link").id].reshape(
    3, 3
  )
  ee_to_hand = home_ee_rotation.T @ home_hand_rotation
  rotation = PokerDrawPlanner(sim).plan().end_effector_rotation
  yaw = np.deg2rad(12.0)
  rz = np.array(
    [
      [np.cos(yaw), -np.sin(yaw), 0.0],
      [np.sin(yaw), np.cos(yaw), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  palm_down = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
  v6_rotation = rz @ _tilt_in_world_xy(palm_down, 0.0, -8.0) @ ee_to_hand.T
  v6_fingers = np.array(
    [
      [0.0, 31.582096, 3.857904, -10.0],
      [0.0, 33.375664, 2.624336, -10.0],
      [0.0, 32.115073, 4.444927, -10.0],
      [0.0, 30.115495, 6.264505, -10.0],
    ]
  )
  joint_ids = np.array(
    [
      [
        sim.model.joint(f"hand_r_{finger}_joint{joint}").qposadr[0]
        for joint in (1, 2, 3, 4)
      ]
      for finger in config._FINGERS
    ]
  )
  palm_heights, wrist_heights, support_heights = [], [], []
  for orientation, offset, fingers in (
    (v6_rotation, np.array([-0.149738, -0.030284, 0.044306]), v6_fingers),
    (rotation, config._FLAT_DRAW_PRECONTACT_OFFSET, config._FLAT_DRAW_FINGER_DEGREES),
  ):
    ik = sim.solve_ik(
      "right",
      sim.object_pose("card")[:3] + offset,
      orientation,
      seed=np.deg2rad([-55, -65, 70, -60, 120, 0, 0]),
      max_iterations=700,
      position_tolerance=0.00005,
      orientation_tolerance=0.002,
    )
    assert ik.success
    # FK-only test; production motion still uses the physical arm servo.
    sim.data.qpos[sim._arm_qpos["right"]] = ik.joint_positions
    sim.data.qpos[joint_ids] = np.deg2rad(fingers)
    mujoco.mj_forward(sim.model, sim.data)
    palm_heights.append(float(sim.data.xpos[sim.model.body("hand_r_base_link").id, 2]))
    wrist_heights.append(float(sim.current_pose_matrix("right")[0][2]))
    heights = []
    for finger in config._FINGERS:
      geom = sim.model.geom(f"hand_r_{finger}_link4_tactile_pad_col")
      assert geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH
      mesh = int(geom.dataid[0])
      start = sim.model.mesh_vertadr[mesh]
      vertices = sim.model.mesh_vert[start : start + sim.model.mesh_vertnum[mesh]]
      world = (
        vertices @ sim.data.geom_xmat[geom.id].reshape(3, 3).T
        + sim.data.geom_xpos[geom.id]
      )
      heights.append(float(world[:, 2].min()))
    support_heights.append(heights)
  assert 0.020 < palm_heights[1] - palm_heights[0] < 0.032
  assert 0.020 < wrist_heights[1] - wrist_heights[0] < 0.032
  np.testing.assert_allclose(support_heights[1], support_heights[0], atol=0.00015)
