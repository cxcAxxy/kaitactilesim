from __future__ import annotations

from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config as usb_config
from kaihand_tactile_env.tasks.usb_insert.execution import (
  UsbInsertionExecutor,
  _interpolate_rotation,
  _rotation_step,
  _rotation_vector_world,
)
from kaihand_tactile_env.tasks.usb_insert.precontact_noise import (
  CONTACT_THRESHOLD_N,
  RECOVERY_DURATION_S,
)
from kaihand_tactile_env.tasks.usb_insert.setup import (
  initialize_face_down,
  initialize_for_insertion,
)
from kaihand_tactile_env.tasks.usb_insert.task import UsbInsertionMonitor


def axis_rotation(axis, angle):
  """Independent Rodrigues oracle, including mixed-sign half-turn axes."""
  axis = np.asarray(axis, dtype=float)
  axis /= np.linalg.norm(axis)
  x, y, z = axis
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  return (
    np.cos(angle) * np.eye(3)
    + (1 - np.cos(angle)) * np.outer(axis, axis)
    + np.sin(angle) * skew
  )


@pytest.mark.parametrize("axis", [(1, -2, 3), (-1, 0, 1)])
def test_exact_half_turn_retains_relative_axis_signs(axis):
  axis = np.asarray(axis, dtype=float)
  axis /= np.linalg.norm(axis)
  start = axis_rotation((2, 1, -1), 0.73)
  end = (2 * np.outer(axis, axis) - np.eye(3)) @ start

  vector = _rotation_vector_world(end, start)

  assert np.linalg.norm(vector) == pytest.approx(np.pi, abs=1e-12)
  # A half-turn permits either global axis sign, but not independent signs.
  assert abs(np.dot(vector / np.pi, axis)) == pytest.approx(1, abs=1e-12)
  np.testing.assert_allclose(_rotation_step(vector) @ start, end, atol=1e-12)


@pytest.mark.parametrize("degrees,sign", [(179.9, 1), (180.1, -1)])
def test_near_half_turn_uses_the_correct_shortest_world_rotation(degrees, sign):
  axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14)
  start = axis_rotation((-1, 4, 2), 0.61)
  end = axis_rotation(axis, np.deg2rad(degrees)) @ start

  vector = _rotation_vector_world(end, start)

  np.testing.assert_allclose(vector, sign * np.deg2rad(179.9) * axis, atol=1e-12)
  np.testing.assert_allclose(_rotation_step(vector) @ start, end, atol=1e-12)


@pytest.mark.parametrize("degrees", [179.9, 180.0, 180.1])
def test_rotation_interpolation_is_proper_and_continuous_along_each_arc(degrees):
  start = np.diag([1.0, -1.0, -1.0])
  end = axis_rotation((-1, 0, 1), np.deg2rad(degrees)) @ start
  rotations = [_interpolate_rotation(start, end, u) for u in np.linspace(0, 1, 21)]

  np.testing.assert_allclose(rotations[0], start, atol=1e-12)
  np.testing.assert_allclose(rotations[-1], end, atol=1e-12)
  for rotation in rotations:
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-12)
  halfway = rotations[10] @ start.T
  np.testing.assert_allclose(halfway @ halfway, end @ start.T, atol=1e-12)

  # The shortest arc switches branches across pi. Continuity is required
  # within each selected arc, not between the two different half-turn paths.
  segment_angle = np.deg2rad(min(degrees, 360 - degrees)) / 20
  for previous, following in zip(rotations[:-1], rotations[1:], strict=True):
    relative = following @ previous.T
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    assert angle == pytest.approx(segment_angle, abs=1e-12)


@pytest.fixture(scope="module")
def simulation():
  return ArmHandSimulation(scene="usb-insert", add_genesis_probes=False)


@pytest.fixture
def prepared_simulation(simulation):
  simulation.reset()
  # This is fixture setup, before execution starts: the baseline's required
  # initial orientation. All object setters are forbidden below.
  initialize_for_insertion(simulation)
  iterations = simulation.model.opt.noslip_iterations
  yield simulation
  simulation.model.opt.noslip_iterations = iterations


def forbid_object_relocation(simulation, monkeypatch):
  for method in (
    "reset",
    "set_object_pose",
    "restore_full_state",
    "set_object_stabilizer",
  ):
    monkeypatch.setattr(
      simulation,
      method,
      Mock(side_effect=AssertionError(f"execution must not call {method}")),
    )


def assert_holding_measured_right_joints(simulation):
  measured = simulation.data.qpos[simulation._arm_qpos["right"]]
  np.testing.assert_array_equal(simulation.arm_goal["right"], measured)
  np.testing.assert_array_equal(simulation._arm_command["right"], measured)
  np.testing.assert_array_equal(
    simulation.data.ctrl[simulation._arm_actuators["right"]], measured
  )
  for name in simulation._hand_joint_names["right"]:
    lower, upper = simulation.model.jnt_range[simulation._joint_id[name]]
    expected = np.clip(
      simulation.data.qpos[simulation._qpos_address[name]], lower, upper
    )
    assert simulation._hand_targets["right"][name] == pytest.approx(expected, abs=1e-12)


def test_immediate_cancellation_does_not_step_or_relocate_and_holds_current_pose(
  prepared_simulation, monkeypatch
):
  simulation = prepared_simulation
  measured_arm = simulation.data.qpos[simulation._arm_qpos["right"]].copy()
  simulation.set_arm_joint_goal("right", measured_arm + 0.04)
  simulation.set_hand_closure("right", 1.0)
  assert simulation.arm_goal_error("right") > 0.01
  assert simulation.hand_goal_error("right") > 0.01
  qpos, qvel = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  plug_pose = simulation.object_pose("usb_plug").copy()
  external_wrench = simulation.data.xfrc_applied.copy()
  left_goal = simulation.arm_goal["left"]
  left_command = simulation._arm_command["left"].copy()
  left_hand = simulation._hand_targets["left"].copy()
  observer = Mock()
  forbidden_step = Mock(side_effect=AssertionError("cancelled execution must not step"))
  monkeypatch.setattr(simulation, "step", forbidden_step)
  monkeypatch.setattr(mujoco, "mj_step", forbidden_step)
  forbid_object_relocation(simulation, monkeypatch)

  result = UsbInsertionExecutor(
    simulation, observer, should_stop=lambda: True
  ).execute()

  assert not result.success and not result.grasp_verified and not result.released
  assert result.failure_reason == "cancelled"
  assert result.phases == ()
  assert result.elapsed_simulation_s == 0
  assert result.maximum_lift_m == 0
  assert result.minimum_lift_grip_force_n == (0.0, 0.0)
  forbidden_step.assert_not_called()
  observer.assert_not_called()
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  np.testing.assert_array_equal(simulation.data.xfrc_applied, external_wrench)
  np.testing.assert_array_equal(result.final_plug_pose_wxyz, plug_pose)
  assert_holding_measured_right_joints(simulation)
  np.testing.assert_array_equal(simulation.arm_goal["left"], left_goal)
  np.testing.assert_array_equal(simulation._arm_command["left"], left_command)
  assert simulation._hand_targets["left"] == left_hand


def test_cancellation_after_real_steps_returns_last_state_without_an_extra_step(
  prepared_simulation, monkeypatch
):
  simulation = prepared_simulation
  simulation.set_arm_joint_goal("right", simulation.arm_goal["right"] + 0.04)
  simulation.set_hand_closure("right", 0.5)
  snapshots = []

  def observe(simulation, phase):
    snapshots.append(
      (
        phase,
        simulation.data.qpos.copy(),
        simulation.data.qvel.copy(),
        simulation.object_pose("usb_plug").copy(),
      )
    )

  step = Mock(wraps=simulation.step)
  monkeypatch.setattr(simulation, "step", step)
  forbid_object_relocation(simulation, monkeypatch)
  result = UsbInsertionExecutor(
    simulation, observe, should_stop=lambda: len(snapshots) == 4
  ).execute()

  assert not result.success and not result.grasp_verified and not result.released
  assert result.failure_reason == "cancelled"
  assert result.phases == ("settle",)
  assert result.elapsed_simulation_s == pytest.approx(4 * simulation.timestep)
  assert step.call_count == len(snapshots) == 4
  phase, qpos, qvel, plug_pose = snapshots[-1]
  assert phase == "settle"
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  np.testing.assert_array_equal(result.final_plug_pose_wxyz, plug_pose)
  assert_holding_measured_right_joints(simulation)
  assert not np.any(simulation.data.xfrc_applied)


def test_face_down_initialization_only_changes_orientation_at_time_zero(simulation):
  simulation.reset()
  joint = simulation.model.joint("usb_plug_freejoint")
  address = int(joint.qposadr[0])
  original_qpos = simulation.data.qpos.copy()
  original_qvel = simulation.data.qvel.copy()
  original_goals = simulation.arm_goal
  np.testing.assert_array_equal(original_qpos[address + 3 : address + 7], [1, 0, 0, 0])

  initialize_face_down(simulation)

  expected_qpos = original_qpos.copy()
  expected_qpos[address + 3 : address + 7] = [0, 1, 0, 0]
  np.testing.assert_array_equal(simulation.data.qpos, expected_qpos)
  np.testing.assert_array_equal(simulation.data.qvel, original_qvel)
  for side in ("left", "right"):
    np.testing.assert_array_equal(simulation.arm_goal[side], original_goals[side])
  assert simulation.data.time == 0
  simulation.reset()
  np.testing.assert_array_equal(simulation.data.qpos, original_qpos)


def test_automatic_initialization_uses_equivalent_short_pickup_coordinate(simulation):
  simulation.reset()
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  wrist_address = int(simulation.model.joint("right_arm_joint5").qposadr[0])
  original_qpos = simulation.data.qpos.copy()
  original_qvel = simulation.data.qvel.copy()
  original_goals = simulation.arm_goal
  original_wrist_pose = simulation.current_pose_matrix("right")
  expected_quaternion = np.asarray(usb_config.AUTO_PLUG_QUATERNION_WXYZ)
  assert np.linalg.norm(expected_quaternion) == pytest.approx(1.0, abs=1e-12)

  initialization = initialize_for_insertion(simulation)

  expected_qpos = original_qpos.copy()
  expected_qpos[address + 3 : address + 7] = expected_quaternion
  expected_qpos[wrist_address] += 2 * np.pi
  np.testing.assert_array_equal(simulation.data.qpos, expected_qpos)
  np.testing.assert_array_equal(simulation.data.qvel, original_qvel)
  np.testing.assert_array_equal(simulation.arm_goal["left"], original_goals["left"])
  expected_right_goal = original_goals["right"].copy()
  expected_right_goal[4] += 2 * np.pi
  np.testing.assert_array_equal(simulation.arm_goal["right"], expected_right_goal)
  current_wrist_pose = simulation.current_pose_matrix("right")
  np.testing.assert_allclose(current_wrist_pose[0], original_wrist_pose[0], atol=1e-14)
  np.testing.assert_allclose(current_wrist_pose[1], original_wrist_pose[1], atol=1e-14)
  assert initialization["pickup_joint_wrap"]["physical_pose_changed"] is False
  assert initialization["pickup_joint_wrap"]["turns"] == 1
  assert simulation.data.time == 0.0
  simulation.reset()
  np.testing.assert_array_equal(simulation.data.qpos, original_qpos)


@pytest.mark.parametrize(
  "initialize",
  (initialize_face_down, initialize_for_insertion),
  ids=("legacy_face_down", "automatic"),
)
def test_usb_initialization_rejects_an_episode_that_has_started(simulation, initialize):
  simulation.reset()
  simulation.step()
  qpos, qvel = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  with pytest.raises(ValueError, match="only allowed immediately after reset"):
    initialize(simulation)
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  assert simulation.data.time == simulation.timestep


def test_usb_initialization_and_executor_reject_another_real_scene():
  simulation = ArmHandSimulation(scene="poker-draw", add_genesis_probes=False)
  qpos, qvel = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  for operation in (
    initialize_face_down,
    initialize_for_insertion,
    UsbInsertionExecutor,
  ):
    with pytest.raises(ValueError, match="requires scene='usb-insert'"):
      operation(simulation)
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  assert simulation.data.time == 0


def test_default_scene_orientation_fails_precondition_without_robot_motion(
  simulation, monkeypatch
):
  simulation.reset()
  np.testing.assert_array_equal(simulation.object_pose("usb_plug")[3:], [1, 0, 0, 0])
  qpos, qvel = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  goals = simulation.arm_goal
  iterations = simulation.model.opt.noslip_iterations
  step = Mock(side_effect=AssertionError("invalid initial orientation must not step"))
  observer = Mock()
  monkeypatch.setattr(simulation, "step", step)
  monkeypatch.setattr(mujoco, "mj_step", step)
  forbid_object_relocation(simulation, monkeypatch)

  result = UsbInsertionExecutor(simulation, observer).execute()

  assert not result.success and not result.grasp_verified and not result.released
  assert "requires" in result.failure_reason
  assert result.phases == ()
  assert result.elapsed_simulation_s == 0
  step.assert_not_called()
  observer.assert_not_called()
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  for side in ("left", "right"):
    np.testing.assert_array_equal(simulation.arm_goal[side], goals[side])
  assert simulation.model.opt.noslip_iterations == iterations


@pytest.mark.parametrize("yaw_deg", [-5.0, 5.0])
def test_planar_randomization_passes_precondition_without_relocating_plug(
  simulation, monkeypatch, yaw_deg
):
  simulation.reset()
  initialize_for_insertion(simulation, yaw_offset_rad=np.deg2rad(yaw_deg))
  initial_pose = simulation.object_pose("usb_plug").copy()
  forbid_object_relocation(simulation, monkeypatch)
  result = UsbInsertionExecutor(
    simulation, should_stop=lambda: simulation.data.time > 0
  ).execute()
  assert result.failure_reason == "cancelled"
  assert result.phases == ("settle",)
  assert result.elapsed_simulation_s == pytest.approx(simulation.timestep)
  # One gravity step cannot turn a five-degree yaw offset back to nominal.
  np.testing.assert_allclose(
    result.final_plug_pose_wxyz[3:], initial_pose[3:], atol=1e-8
  )


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0)])
def test_tilted_initialization_still_fails_before_motion(simulation, monkeypatch, axis):
  simulation.reset()
  initialize_for_insertion(simulation)
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  tilt_quat = np.r_[
    np.cos(np.deg2rad(3) / 2), np.array(axis) * np.sin(np.deg2rad(3) / 2)
  ]
  quaternion = np.empty(4)
  mujoco.mju_mulQuat(
    quaternion, tilt_quat, simulation.data.qpos[address + 3 : address + 7]
  )
  simulation.data.qpos[address + 3 : address + 7] = quaternion
  mujoco.mj_forward(simulation.model, simulation.data)
  qpos, qvel = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  forbid_object_relocation(simulation, monkeypatch)
  step = Mock(side_effect=AssertionError("tilted placement must not step"))
  monkeypatch.setattr(simulation, "step", step)
  result = UsbInsertionExecutor(simulation).execute()
  assert not result.success
  assert "requires a flat, mark-down" in result.failure_reason
  assert result.phases == ()
  step.assert_not_called()
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)


@pytest.mark.parametrize(
  "initialization,noise_seed,motion_profile",
  [
    ({}, None, "fast"),
    ({"offset_xy_m": [0.01, 0.01], "yaw_offset_rad": np.deg2rad(-5.0)}, None, "fast"),
    ({}, 1, "fast"),
    ({}, 2, "fast"),
    ({"offset_xy_m": [0.01, 0.01], "yaw_offset_rad": np.deg2rad(-5.0)}, 3, "fast"),
    ({}, None, "baseline"),
  ],
  ids=[
    "nominal",
    "randomized_elbow_boundary",
    "noise_1",
    "noise_2",
    "randomized_noise_3",
    "baseline",
  ],
)
def test_real_robot_grasps_lifts_inserts_and_releases_without_object_assistance(
  monkeypatch, record_property, initialization, noise_seed, motion_profile
):
  """Run the complete controller with real physics, contacts and success monitor."""
  simulation = ArmHandSimulation(scene="usb-insert", add_genesis_probes=True)
  assert simulation.genesis_probe_layout is not None
  initialize_for_insertion(simulation, **initialization)
  model, data = simulation.model, simulation.data
  plug_body = model.body("usb_plug").id
  plug_joint = model.joint("usb_plug_freejoint")
  plug_qpos = int(plug_joint.qposadr[0])
  plug_dof = int(plug_joint.dofadr[0])
  original_qpos = data.qpos.copy()
  original_mass = model.body_mass.copy()
  original_friction = model.geom_friction.copy()
  equality_fields = ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data")
  equalities = {name: getattr(model, name).copy() for name in equality_fields}
  equality_active = data.eq_active.copy()
  body_constraints = np.isin(
    model.eq_type, [mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT]
  )
  assert not np.any(
    body_constraints & ((model.eq_obj1id == plug_body) | (model.eq_obj2id == plug_body))
  )

  # Guard the actual MuJoCo boundary: every qpos/qvel seen before the next
  # physics step must be exactly the previous physics output. This catches
  # direct array teleports as well as forbidding the public replay/reset APIs.
  last_qpos, last_qvel = data.qpos.copy(), data.qvel.copy()
  actual_step = mujoco.mj_step
  step_count = 0

  def assert_no_object_assistance():
    assert not np.any(data.xfrc_applied[plug_body])
    # Arm gravity compensation is part of the shared robot controller.
    assert not np.any(data.qfrc_applied[plug_dof : plug_dof + 6])
    np.testing.assert_array_equal(data.eq_active, equality_active)
    for name, values in equalities.items():
      np.testing.assert_array_equal(getattr(model, name), values)

  def audited_step(observed_model, observed_data, *args, **kwargs):
    nonlocal step_count
    assert observed_model is model and observed_data is data
    np.testing.assert_array_equal(data.qpos, last_qpos)
    np.testing.assert_array_equal(data.qvel, last_qvel)
    assert_no_object_assistance()
    before = data.time
    actual_step(model, data, *args, **kwargs)
    assert data.time - before == pytest.approx(simulation.timestep, abs=1e-12)
    np.copyto(last_qpos, data.qpos)
    np.copyto(last_qvel, data.qvel)
    step_count += 1
    assert_no_object_assistance()

  monkeypatch.setattr(mujoco, "mj_step", audited_step)
  forbid_object_relocation(simulation, monkeypatch)
  for name in ("mj_resetData", "mj_resetDataKeyframe", "mj_setState"):
    monkeypatch.setattr(
      mujoco, name, Mock(side_effect=AssertionError(f"execution must not call {name}"))
    )

  plug_geoms = model.geom_bodyid == plug_body
  hand_geoms = np.array(
    [
      model.body(int(body)).name.startswith(("hand_r_", "hand_l_"))
      for body in model.geom_bodyid
    ]
  )
  socket_geoms = np.array(
    [model.geom(i).name.startswith("usb_socket_") for i in range(model.ngeom)]
  )
  geom_body_names = [model.body(int(body)).name for body in model.geom_bodyid]
  robot_geoms = np.array(
    [
      name.startswith(("hand_r_", "hand_l_", "right_arm_", "left_arm_"))
      or name == "torso"
      for name in geom_body_names
    ]
  )
  table_fixture_geoms = np.array(
    [
      name in ("table", "usb_fixture", "usb_socket") or name.startswith("usb_socket_")
      for name in geom_body_names
    ]
  )
  shoulder_body = model.body("right_arm_link1").id
  elbow_body = model.body("right_arm_link4").id
  wrist_site = model.site("right_ee_site").id
  palm_body = model.body("hand_r_base_link").id
  arm_limits = model.jnt_range[simulation._arm_joint_ids["right"]]
  # These MCP/CMC root origins identify the side of the palm independently
  # of distal curling, which can reverse fingertip ordering in a pinch.
  finger_root_bodies = [
    model.body(name).id
    for name in ("hand_r_index_link2", "hand_r_pinky_link2", "hand_r_thumb_link1")
  ]
  trace = []
  force = np.empty(6)
  tactile_geoms = np.array(
    [model.geom(i).name.endswith("_tactile_pad_col") for i in range(model.ngeom)]
  )
  first_tactile = None
  cutoff_draws = None
  grip_force_trace = []
  grip_tangent_trace = []
  alignment_monitor = UsbInsertionMonitor(simulation)
  alignment_dwell_s = 0.0
  longest_alignment_dwell_s = 0.0
  grip_pad_ids = [
    model.geom(name).id
    for name in (
      "hand_r_thumb_link6_tactile_pad_col",
      "hand_r_index_link4_tactile_pad_col",
    )
  ]
  first_close_checked = False

  def observe(simulation, phase):
    nonlocal first_tactile, cutoff_draws
    nonlocal alignment_dwell_s, longest_alignment_dwell_s
    nonlocal first_close_checked
    if phase in {"hover", "approach"} or (phase == "close" and not first_close_checked):
      tips = data.site_xpos[simulation._fingertip_site_ids["right"][:2]]
      assert np.linalg.norm(tips[1] - tips[0]) > 0.050
    if phase == "close" and not first_close_checked:
      from kaihand_tactile_env.tasks.usb_insert.grasp import calibrated_grasp

      pickup = calibrated_grasp(
        simulation, pinch_tilt_rad=0.05 if motion_profile == "baseline" else 0.06
      )
      wrist, rotation = simulation.current_pose_matrix("right")
      assert np.linalg.norm(wrist - pickup.wrist_position) < 0.003
      assert (
        np.linalg.norm(_rotation_vector_world(pickup.wrist_rotation, rotation)) < 0.03
      )
      first_close_checked = True
    if phase == "align":
      measured = alignment_monitor.measure()
      stationary_above_socket = (
        abs(measured.insertion_depth_m + 0.10) < 0.00015
        and np.linalg.norm(measured.lateral_error_m) < 0.00008
        and measured.orientation_error_rad < 0.004
        and measured.linear_speed_m_s < 0.001
        and measured.angular_speed_rad_s < 0.01
      )
      alignment_dwell_s = (
        alignment_dwell_s + simulation.timestep if stationary_above_socket else 0.0
      )
      longest_alignment_dwell_s = max(longest_alignment_dwell_s, alignment_dwell_s)
      if measured.insertion_depth_m > -0.05:
        assert longest_alignment_dwell_s >= 0.6 - 1e-12
    assert step_count == len(trace) + 1
    np.testing.assert_array_equal(data.qpos, last_qpos)
    np.testing.assert_array_equal(data.qvel, last_qvel)
    goal = simulation.arm_goal["right"]
    # The requested 10 cm hover uses more arm workspace than the low hover;
    # retain a 10-degree margin there and the original 15 near insertion.
    required_margin = (
      10.0 if phase == "align" and executor._state.insertion_depth_m < -0.03 else 15.0
    )
    assert np.min(np.minimum(goal - arm_limits[:, 0], arm_limits[:, 1] - goal)) >= (
      np.deg2rad(required_margin)
    ), phase
    pairs = np.column_stack((data.contact.geom1, data.contact.geom2)).copy()
    first, second = pairs[:, 0], pairs[:, 1]
    # Read solver forces independently of the controller/taxel distributor.
    # Holding the plug must not produce the former contact-count force steps.
    grip_forces = np.zeros(2)
    grip_tangents = np.zeros(2)
    for finger, pad in enumerate(grip_pad_ids):
      contacts = ((first == pad) & plug_geoms[second]) | (
        (second == pad) & plug_geoms[first]
      )
      for index in np.flatnonzero(contacts):
        mujoco.mj_contactForce(model, data, int(index), force)
        grip_forces[finger] += abs(float(force[0]))
        grip_tangents[finger] += float(np.linalg.norm(force[1:3]))
    grip_force_trace.append(grip_forces)
    grip_tangent_trace.append(grip_tangents)
    if phase in {"hover", "approach"}:
      assert max(grip_forces) < 1e-3, "pinch made contact before reaching pickup"
    if noise_seed is not None:
      # Independently read all pad/USB solver forces every physics step up to
      # first signal. Do not reuse the executor's bilateral grip measurement.
      if first_tactile is None:
        tactile_contacts = (plug_geoms[first] & tactile_geoms[second]) | (
          plug_geoms[second] & tactile_geoms[first]
        )
        pad_loads = {}
        for index in np.flatnonzero(tactile_contacts):
          mujoco.mj_contactForce(model, data, int(index), force)
          a, b = pairs[index]
          pad = int(b if plug_geoms[a] else a)
          pad_loads[pad] = pad_loads.get(pad, 0.0) + abs(float(force[0]))
        if pad_loads and max(pad_loads.values()) > CONTACT_THRESHOLD_N:
          pad = max(pad_loads, key=pad_loads.get)
          first_tactile = {
            "time_s": float(data.time),
            "phase": phase,
            "pad_name": model.geom(pad).name,
            "normal_force_n": pad_loads[pad],
          }
          cutoff_draws = executor._noise.draw_count
      assert executor._noise.first_contact == first_tactile
      if first_tactile is not None:
        assert executor._noise.draw_count == cutoff_draws
        assert not np.any(executor._noise.random_offset_m)
        if data.time >= first_tactile["time_s"] + RECOVERY_DURATION_S + 0.021:
          assert not np.any(executor._noise.recovery_offset_m)
    hand_contacts = (plug_geoms[first] & hand_geoms[second]) | (
      plug_geoms[second] & hand_geoms[first]
    )
    socket_contacts = (plug_geoms[first] & socket_geoms[second]) | (
      plug_geoms[second] & socket_geoms[first]
    )
    robot_obstacle_contacts = (robot_geoms[first] & table_fixture_geoms[second]) | (
      robot_geoms[second] & table_fixture_geoms[first]
    )
    assert not np.any(robot_obstacle_contacts), (
      phase,
      [
        (model.geom(int(a)).name, model.geom(int(b)).name)
        for a, b in pairs[robot_obstacle_contacts]
      ],
    )
    socket_load = 0.0
    for index in np.flatnonzero(socket_contacts):
      mujoco.mj_contactForce(model, data, int(index), force)
      socket_load += abs(float(force[0]))
    trace.append(
      (
        phase,
        data.qpos.copy(),
        data.qvel.copy(),
        pairs,
        int(np.count_nonzero(hand_contacts)),
        socket_load,
        (data.xpos[elbow_body] - data.xpos[shoulder_body]).copy(),
        data.site_xmat[wrist_site].reshape(3, 3).copy(),
        data.xpos[finger_root_bodies].copy(),
        # Fingers flex toward hand-local +Y: this is the palm-side normal.
        data.xmat[palm_body].reshape(3, 3)[:, 1].copy(),
        executor._state.insertion_depth_m,
      )
    )

  executor = UsbInsertionExecutor(
    simulation,
    observe,
    precontact_noise_std_m=0.0005 if noise_seed is not None else 0.0,
    noise_seed=noise_seed,
    motion_profile=motion_profile,
  )
  result = executor.execute()

  if noise_seed is not None:
    assert first_tactile is not None
    report = result.precontact_noise
    assert report["first_contact"] == first_tactile
    assert report["gaussian_knot_draws"] == cutoff_draws
    assert report["maximum_random_offset_norm_m"] > 0.0001
    assert report["commands"][-1]["time_s"] == pytest.approx(
      first_tactile["time_s"] + RECOVERY_DURATION_S, abs=0.021
    )
    for command in report["commands"]:
      random = np.array(command["random_offset_m"])
      recovery = np.array(command["recovery_offset_m"])
      np.testing.assert_allclose(
        command["applied_wrist_position_m"],
        np.array(command["nominal_wrist_position_m"]) + random + recovery,
        atol=1e-14,
      )
      assert random[2] == recovery[2] == 0
      assert np.max(np.abs(random)) <= 0.0015
      if command["time_s"] >= first_tactile["time_s"]:
        assert not np.any(random)
        assert command["gaussian_knot_draws"] == cutoff_draws
    record_property("noise_seed", noise_seed)
    record_property("first_tactile_time_s", first_tactile["time_s"])
    record_property("maximum_random_offset_m", report["maximum_random_offset_norm_m"])
  else:
    assert not result.precontact_noise["enabled"]
    assert result.precontact_noise["commands"] == []

  assert result.success, result.failure_reason
  assert first_close_checked
  assert longest_alignment_dwell_s >= 0.6 - 1e-12
  record_property("stationary_alignment_dwell_s", longest_alignment_dwell_s)
  assert result.failure_reason is None
  assert result.grasp_verified and result.released
  assert result.active_bottom_out_confirmed
  assert result.bottom_out_hold_s >= usb_config.BOTTOM_OUT_HOLD_S - 1e-12
  assert result.peak_backstop_axial_resistance_n >= usb_config.BOTTOM_OUT_MIN_FORCE_N
  # Keep the faster motion schedule while still passing all physical and
  # posture checks below. The preceding nominal rollout took 25.736 seconds.
  assert result.motion_profile == motion_profile
  # Physical bottom contact, controlled preload and unloading add a short
  # confirmation to the existing fast transport; no free-travel early exit.
  # Allow the longer descent from the new 10 cm stationary hover.
  # Wide-to-fine closing now occurs stationary at pickup, adding this ramp.
  assert result.elapsed_simulation_s <= (
    (25.0 if motion_profile == "fast" else 28.5) + 0.6 * executor.motion.grasp_ramp_s
  )
  assert 0.03 < result.maximum_lift_m < 0.25
  assert 0.0119 <= result.insertion.insertion_depth_m <= 0.0121
  assert result.insertion.bottom_out_confirmed
  assert result.peak_socket_normal_load_n < usb_config.MAX_SOCKET_NORMAL_LOAD_N
  assert len(trace) == step_count > 1000
  assert result.elapsed_simulation_s == pytest.approx(step_count * simulation.timestep)
  assert {
    "lift",
    "insert",
    "bottom_out",
    "unload",
    "release",
    "retreat",
    "verify",
  }.issubset(result.phases)
  positions = np.array([sample[1] for sample in trace])
  velocities = np.array([sample[2] for sample in trace])
  assert np.isfinite(positions).all() and np.isfinite(velocities).all()
  maximum_height_gain = (
    np.max(positions[:, plug_qpos + 2]) - original_qpos[plug_qpos + 2]
  )
  assert 0.03 < maximum_height_gain < 0.25
  phases = np.array([sample[0] for sample in trace])
  holding = np.isin(phases, ("rotate_and_transfer", "align", "insert", "bottom_out"))
  adjacent_holding = holding[1:] & holding[:-1] & (phases[1:] == phases[:-1])
  normal_steps = np.abs(np.diff(grip_force_trace, axis=0))[adjacent_holding]
  maximum_normal_step = float(np.max(normal_steps))
  assert maximum_normal_step < 0.15  # N per 2 ms, excluding grasp/release onset.
  record_property("maximum_holding_normal_force_step_n", maximum_normal_step)
  tangents = np.asarray(grip_tangent_trace)
  insertion = np.isin(phases, ("insert", "bottom_out"))
  consecutive = insertion[1:] & insertion[:-1]
  tangent_step = float(np.max(np.abs(np.diff(tangents, axis=0))[consecutive]))
  # The old bottom impact changed Ft by about 0.5 N in one 2 ms step.
  assert tangent_step < 0.2
  release_tangent = float(np.max(tangents[phases == "release"]))
  assert release_tangent < 0.15  # no former 0.68 N regrasp spike
  unloading = np.isin(phases, ("unload", "release"))
  consecutive = unloading[1:] & unloading[:-1]
  unloading_normal_step = float(
    np.max(np.abs(np.diff(grip_force_trace, axis=0))[consecutive])
  )
  assert unloading_normal_step < 0.35
  record_property("maximum_insertion_tangent_step_n", tangent_step)
  record_property("maximum_release_tangent_n", release_tangent)
  record_property("maximum_unloading_normal_step_n", unloading_normal_step)

  lift_mask = phases == "lift"
  # Check the initial table clearance separately from the later transfer.
  lift_clearance = (
    np.max(positions[lift_mask, plug_qpos + 2]) - original_qpos[plug_qpos + 2]
  )
  assert 0.03 < lift_clearance < 0.07

  # Evaluate actual joint angles and FK at every 2 ms physics step. The old
  # rollout lifted its elbow to shoulder height and wound J7 to 90 degrees.
  # Grasp/lift can still require a bent wrist; placement must stay relaxed.
  arm_positions = positions[:, simulation._arm_qpos["right"]]
  # A reproducible real-state checkpoint, for comparing equal initial poses
  # across the independently seeded noise cases in the saved JUnit report.
  record_property(
    "actual_arm_at_2s_rad", arm_positions[round(2 / simulation.timestep) - 1].tolist()
  )
  arm_degrees = np.rad2deg(arm_positions)
  pickup_frames = np.flatnonzero(
    np.isin(phases, ("settle", "preshape", "hover", "approach"))
  )
  pickup_j5 = arm_degrees[pickup_frames, 4]
  pickup_j5_travel = float(np.abs(np.diff(pickup_j5)).sum())
  assert pickup_j5[0] > 180.0
  assert pickup_j5[-1] < pickup_j5[0]
  assert pickup_j5_travel < 140.0
  record_property("pickup_j5_travel_deg", pickup_j5_travel)
  close_frames = np.flatnonzero(phases == "close")
  insert_frames = np.flatnonzero(phases == "insert")
  assert len(close_frames) and len(insert_frames)
  grasp_frame = int(close_frames[-1])
  inserted_frame = int(insert_frames[-1])
  index_mcp, pinky_mcp, thumb_cmc = trace[grasp_frame][8]
  index_to_pinky = pinky_mcp - index_mcp
  pinky_index_forward_m = float(index_to_pinky[0])
  pinky_index_forward_cosine = float(index_to_pinky[0] / np.linalg.norm(index_to_pinky))
  pinky_thumb_forward_m = float(pinky_mcp[0] - thumb_cmc[0])
  assert pinky_index_forward_m >= 0.03
  assert pinky_index_forward_cosine >= 0.5
  assert pinky_thumb_forward_m > 0.0

  # Reversing the USB must also fix the insertion hand, not merely turn
  # hand and object together and reproduce the same upside-down grip.
  insertion_palm_normals = np.array([trace[i][9] for i in insert_frames])
  insertion_roots = np.array([trace[i][8] for i in insert_frames])
  assert np.all(insertion_palm_normals[:, 2] < -0.2)
  assert np.all(insertion_roots[:, 0, 2] - insertion_roots[:, 1, 2] > 0.03)
  assert np.all(insertion_roots[:, 2, 2] - insertion_roots[:, 1, 2] > 0.015)

  # Track the whole physical turn, including any excursion and return that
  # a final wrist angle alone would miss. The upright grip uses a different
  # rotation arc from the previous upside-down hand; neither wrist may wind
  # through a large detour or approach its mechanical limit.
  carrying_degrees = arm_degrees[grasp_frame : inserted_frame + 1]
  wrist_travel_deg = np.abs(np.diff(carrying_degrees[:, 5:7], axis=0)).sum(axis=0)
  ee_rotations = np.array(
    [sample[7] for sample in trace[grasp_frame : inserted_frame + 1]]
  )
  ee_relative = ee_rotations[1:] @ ee_rotations[:-1].transpose(0, 2, 1)
  ee_step_angles = np.arccos(
    np.clip((np.trace(ee_relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
  )
  ee_cumulative_rotation_deg = float(np.rad2deg(ee_step_angles.sum()))
  ee_net_rotation_deg = float(
    np.rad2deg(
      np.arccos(
        np.clip((np.trace(ee_rotations[-1] @ ee_rotations[0].T) - 1.0) / 2.0, -1.0, 1.0)
      )
    )
  )
  # The longer rise and descent add wrist travel, while the per-frame limits
  # below still constrain the high hover and the final insertion posture.
  assert wrist_travel_deg[0] < 100.0
  # A different pickup yaw changes the starting wrist angle and therefore
  # the required net J7 travel. Physical friction alignment can add a few
  # degrees of correction; retain the joint/posture bounds and separately
  # limit travel beyond the necessary endpoint difference below.
  yaw_allowance_deg = abs(float(np.rad2deg(initialization.get("yaw_offset_rad", 0.0))))
  assert wrist_travel_deg[1] < 75.0 + yaw_allowance_deg
  wrist_j7_net_deg = abs(carrying_degrees[-1, 6] - carrying_degrees[0, 6])
  assert wrist_travel_deg[1] - wrist_j7_net_deg < 20.0
  assert ee_cumulative_rotation_deg < 170.0
  elbow_relative = np.array([sample[6] for sample in trace])
  carry_mask = np.isin(phases, ("lift", "rotate_and_transfer"))
  turn_mask = phases == "rotate_and_transfer"
  # The high hover has a different arm posture. Apply insertion-elbow bounds
  # near the mouth, while joint margins/collision checks still cover all steps.
  placement_mask = np.isin(
    phases, ("insert", "bottom_out", "unload", "release", "retreat", "verify")
  ) | ((phases == "align") & (np.array([sample[10] for sample in trace]) > -0.01))
  assert np.any(turn_mask) and np.any(placement_mask)
  transfer_wrist_travel_deg = np.abs(np.diff(arm_degrees[turn_mask, 5:7], axis=0)).sum(
    axis=0
  )
  assert transfer_wrist_travel_deg[0] < 70.0
  assert transfer_wrist_travel_deg[1] < 28.0 + yaw_allowance_deg
  carry_max_j7 = float(np.max(np.abs(arm_degrees[carry_mask, 6])))
  turn_min_elbow_below = float(np.min(-elbow_relative[turn_mask, 2]))
  placement_max_j6 = float(np.max(np.abs(arm_degrees[placement_mask, 5])))
  placement_max_j7 = float(np.max(np.abs(arm_degrees[placement_mask, 6])))
  placement_min_elbow_below = float(np.min(-elbow_relative[placement_mask, 2]))
  # Open the elbow slightly compared with the preceding upright rollout
  # (which reached Y=-0.133 m), while retaining the same palm-down hand pose.
  placement_min_elbow_outside = float(np.min(-elbow_relative[placement_mask, 1]))
  placement_max_elbow_world_y = float(
    data.xpos[shoulder_body, 1] + np.max(elbow_relative[placement_mask, 1])
  )
  limits = arm_limits
  all_joint_margin = np.minimum(
    arm_positions - limits[:, 0], limits[:, 1] - arm_positions
  )
  high_hover = (phases == "align") & (
    np.array([sample[10] for sample in trace]) < -0.03
  )
  assert np.all(
    np.rad2deg(all_joint_margin.min(axis=1)) >= np.where(high_hover, 10.0, 15.0)
  )
  joint_margin = np.minimum(
    arm_positions[placement_mask] - limits[:, 0],
    limits[:, 1] - arm_positions[placement_mask],
  )
  placement_min_joint_margin = float(np.rad2deg(np.min(joint_margin)))
  assert carry_max_j7 < 70.0
  assert turn_min_elbow_below >= 0.12
  assert placement_max_j6 < 45.0
  # Reaching the real 12 mm stop adds less than a degree to this posture;
  # keep a tight 66-degree bound and the original 15-degree joint margin.
  # True bottom contact plus the open-elbow correction needs about 1 degree
  # more wrist travel; retain the independent 15-degree joint-limit guard.
  assert placement_max_j7 < 67.0
  assert placement_min_elbow_below >= 0.12
  assert placement_max_elbow_world_y <= -0.15
  assert placement_min_joint_margin >= 15.0
  assert (
    np.max(
      np.abs(
        positions[:, simulation._arm_qpos["right"]]
        - original_qpos[simulation._arm_qpos["right"]]
      )
    )
    > 0.1
  )
  assert any(sample[0] == "lift" and sample[4] > 0 for sample in trace)
  assert trace[-1][4] == 0
  assert max(sample[5] for sample in trace) < 3.0
  assert max(sample[5] for sample in trace) == pytest.approx(
    result.peak_socket_normal_load_n, abs=1e-12
  )
  independent_final = UsbInsertionMonitor(simulation).measure()
  # A fresh monitor cannot invent the prior physical bottom-out history.
  # After unloading, spring friction may support the plug with zero bottom
  # load; independently verify current geometry and stability in that case.
  assert not independent_final.bottom_out_confirmed
  assert independent_final.seated == independent_final.backstop_contact
  assert independent_final.shell_fits_aperture
  assert independent_final.linear_speed_m_s <= usb_config.SEATED_LINEAR_SPEED_M_S
  assert independent_final.angular_speed_rad_s <= 0.05
  assert 0.0119 <= independent_final.insertion_depth_m <= 0.0121
  np.testing.assert_array_equal(data.qpos, last_qpos)
  np.testing.assert_array_equal(data.qvel, last_qvel)
  np.testing.assert_array_equal(model.body_mass, original_mass)
  np.testing.assert_array_equal(model.geom_friction, original_friction)
  assert_no_object_assistance()
  record_property("elapsed_simulation_s", result.elapsed_simulation_s)
  record_property("motion_profile", result.motion_profile)
  record_property("maximum_lift_m", result.maximum_lift_m)
  record_property("final_insertion_depth_m", independent_final.insertion_depth_m)
  record_property("peak_socket_normal_load_n", result.peak_socket_normal_load_n)
  record_property("initial_lift_clearance_m", lift_clearance)
  record_property("carry_max_abs_j7_deg", carry_max_j7)
  record_property("turn_min_elbow_below_shoulder_m", turn_min_elbow_below)
  record_property("placement_max_abs_j6_deg", placement_max_j6)
  record_property("placement_max_abs_j7_deg", placement_max_j7)
  record_property("placement_min_elbow_below_shoulder_m", placement_min_elbow_below)
  record_property("placement_min_elbow_outside_shoulder_m", placement_min_elbow_outside)
  record_property("placement_max_elbow_world_y_m", placement_max_elbow_world_y)
  record_property("placement_min_joint_limit_margin_deg", placement_min_joint_margin)
  record_property("robot_table_fixture_contact_count", 0)
  record_property("grasp_pinky_minus_index_mcp_forward_m", pinky_index_forward_m)
  record_property(
    "grasp_pinky_minus_index_mcp_forward_cosine", pinky_index_forward_cosine
  )
  record_property("grasp_pinky_minus_thumb_cmc_forward_m", pinky_thumb_forward_m)
  record_property(
    "insertion_max_palm_normal_z", float(insertion_palm_normals[:, 2].max())
  )
  record_property(
    "insertion_min_index_above_pinky_m",
    float((insertion_roots[:, 0, 2] - insertion_roots[:, 1, 2]).min()),
  )
  record_property("grasp_to_insert_wrist_j6_cumulative_deg", float(wrist_travel_deg[0]))
  record_property("grasp_to_insert_wrist_j7_cumulative_deg", float(wrist_travel_deg[1]))
  record_property("initial_yaw_offset_deg", yaw_allowance_deg)
  record_property(
    "grasp_to_insert_wrist_j7_excess_deg", float(wrist_travel_deg[1] - wrist_j7_net_deg)
  )
  record_property(
    "grasp_to_insert_ee_cumulative_rotation_deg", ee_cumulative_rotation_deg
  )
  record_property("grasp_to_insert_ee_net_rotation_deg", ee_net_rotation_deg)
  record_property(
    "transfer_wrist_j6_cumulative_deg", float(transfer_wrist_travel_deg[0])
  )
  record_property(
    "transfer_wrist_j7_cumulative_deg", float(transfer_wrist_travel_deg[1])
  )
  if noise_seed is not None:
    executor.should_stop = lambda: True
    retry = executor.execute()
    assert retry.failure_reason == "cancelled"
    assert retry.elapsed_simulation_s == 0.0
    assert retry.precontact_noise["first_contact"] == first_tactile
    assert retry.precontact_noise["gaussian_knot_draws"] == cutoff_draws
    assert len(trace) == step_count
