from __future__ import annotations

import gc

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES, default_model_path
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import (
  BulbScrewMonitor,
  BulbScrewSimulation,
)


@pytest.fixture(scope="module")
def sim():
  simulation = BulbScrewSimulation()
  yield simulation
  del simulation
  gc.collect()


def test_scene_reuses_shared_sources_and_usb_cameras():
  import xml.etree.ElementTree as ET

  bulb = ET.parse(default_model_path("bulb-screw")).getroot()
  usb = ET.parse(default_model_path("usb-insert")).getroot()
  assert [e.attrib for e in bulb.findall("include")] == [
    e.attrib for e in usb.findall("include")
  ]
  # Both tasks inherit the head calibration from their common robot include.
  for task in (bulb, usb):
    assert task.find("default/camera") is None
    assert task.find(".//camera[@name='head']") is None


def test_scene_isolated_and_passive(sim):
  sim.reset()
  assert sim.object_names == ("bulb",)
  assert sim.genesis_probe_layout.count == 350
  assert sim.model.nu == 54  # 14 arm + 40 hand; no bulb motor.
  for name in ("usb_plug", "usb_socket", "card", "cylinder", "poker_table", "box"):
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
  sim.step(1000)
  assert not sim.thread_engaged
  assert np.isfinite(sim.data.qpos).all()
  assert sim.object_pose("bulb")[2] == pytest.approx(
    config.TABLETOP_HEIGHT_M, abs=0.0001
  )
  assert not BulbScrewMonitor(sim).measure().success


def test_pickup_cradle_is_open_passive_and_clears_neck(sim):
  cradle = sim.model.body("bulb_pickup_cradle").id
  assert sim.model.body_jntnum[cradle] == 0
  assert sim.model.body_geomnum[cradle] == config.PICKUP_CRADLE_PRONG_COUNT
  prongs = np.array(
    [
      sim.model.geom(f"bulb_pickup_cradle_prong_{i:02d}").id
      for i in range(config.PICKUP_CRADLE_PRONG_COUNT)
    ]
  )
  assert set(sim.model.geom_bodyid[prongs]) == {cradle}
  assert np.all(sim.model.geom_type[prongs] == mujoco.mjtGeom.mjGEOM_CAPSULE)
  np.testing.assert_allclose(
    sim.model.geom_size[prongs, 0], config.PICKUP_CRADLE_PRONG_RADIUS_M
  )
  inner_radius = (
    np.linalg.norm(sim.model.geom_pos[prongs, :2], axis=1)
    - sim.model.geom_size[prongs, 0]
  )
  np.testing.assert_allclose(inner_radius, config.PICKUP_CRADLE_INNER_RADIUS_M)
  height = 2 * (sim.model.geom_size[prongs, 0] + sim.model.geom_size[prongs, 1])
  np.testing.assert_allclose(height, config.PICKUP_CRADLE_HEIGHT_M)
  assert np.all(sim.model.geom_friction[prongs, 0] == pytest.approx(0.12))

  neck = sim.model.geom("bulb_neck").id
  assert min(inner_radius) > sim.model.geom_size[neck, 0]
  # No hidden attachment: the free bulb remains a different body and lifts out.
  assert sim.model.geom_bodyid[neck] == sim.model.body("bulb").id

  globe = sim.model.geom("bulb_globe").id
  mark = sim.model.geom("bulb_orientation_mark").id
  np.testing.assert_allclose(
    sim.model.geom_size[globe, :2], config.BULB_GLOBE_RADIUS_XY_M
  )
  assert sim.model.geom_size[globe, 2] == pytest.approx(0.034)
  assert sim.model.geom_pos[mark, 0] == pytest.approx(
    config.BULB_GLOBE_RADIUS_XY_M - 0.0002
  )


@pytest.mark.parametrize(
  "offset,quat",
  [
    ([0.005, 0, 0], [1, 0, 0, 0]),
    ([0, 0, -0.005], [1, 0, 0, 0]),
    ([0, 0, 0], [np.cos(0.15), np.sin(0.15), 0, 0]),
  ],
)
def test_misalignment_cannot_capture(sim, offset, quat):
  sim.reset()
  sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M + offset, quat)
  assert not sim.try_engage_thread()


def test_capture_preserves_free_pose_and_reset_clears_thread(sim):
  sim.reset()
  assert not sim.bulb_lit
  with pytest.raises(ValueError, match="not seated"):
    sim.confirm_tightening()
  quat = [np.cos(0.0005), 0, 0, np.sin(0.0005)]
  sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M, quat)
  dof = int(sim.model.joint("bulb_freejoint").dofadr[0])
  sim.data.qvel[dof : dof + 6] = (0, 0, -0.002, 0, 0, 0.001)
  before = sim.data.qpos.copy()
  velocity = sim.data.qvel.copy()
  assert sim.try_engage_thread()
  free_qpos = int(sim.model.joint("bulb_freejoint").qposadr[0])
  np.testing.assert_array_equal(
    sim.data.qpos[free_qpos : free_qpos + 7],
    before[free_qpos : free_qpos + 7],
  )
  np.testing.assert_array_equal(
    sim.data.qvel[dof : dof + 6],
    velocity[dof : dof + 6],
  )
  sim.step(100)
  np.testing.assert_allclose(sim.object_pose("bulb")[3:], quat, atol=0.003)
  sim.reset(seed=5, object_xy_jitter=0.001)
  first = sim.object_pose("bulb")
  sim.reset(seed=5, object_xy_jitter=0.001)
  np.testing.assert_array_equal(sim.object_pose("bulb"), first)
  assert not sim.thread_engaged
  np.testing.assert_array_equal(
    sim.model.eq_data[sim._weld, 3:10], [0, 0, 0, 1, 0, 0, 0]
  )


@pytest.mark.parametrize(
  ("twist", "margin"),
  (
    (
      (0, 0, -1.01 * config.CAPTURE_SPEED_M_S, 0, 0, 0),
      "capture_speed_margin_m_s",
    ),
    (
      (0, 0, 0, 0, 0, 1.01 * config.CAPTURE_ANGULAR_SPEED_RAD_S),
      "capture_angular_speed_margin_rad_s",
    ),
  ),
)
def test_capture_rejects_high_linear_or_angular_speed(sim, twist, margin):
  sim.reset()
  sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
  dof = int(sim.model.joint("bulb_freejoint").dofadr[0])
  sim.data.qvel[dof : dof + 6] = twist
  mujoco.mj_forward(sim.model, sim.data)

  state = BulbScrewMonitor(sim).measure()
  assert getattr(state, margin) < 0
  assert not state.capture_ready
  assert not sim.try_engage_thread()


def test_capture_initializes_thread_phase_and_smoothly_removes_gate_residual(sim):
  sim.reset()
  yaw = np.deg2rad(340.0)
  tilt = np.deg2rad(1.9)
  yaw_quaternion = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
  tilt_quaternion = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0, 0])
  quaternion = np.empty(4)
  mujoco.mju_mulQuat(quaternion, yaw_quaternion, tilt_quaternion)
  depth_offset = 0.0002
  position = config.THREAD_ENTRY_POSITION_M.copy()
  position[0] += 0.00039
  position[2] -= depth_offset
  sim.set_object_pose("bulb", position, quaternion)
  before = sim.object_pose("bulb").copy()

  assert sim.thread_contact_load() > config.THREAD_CAPTURE_LOAD_N
  assert sim.try_engage_thread()
  np.testing.assert_array_equal(sim.object_pose("bulb"), before)
  assert sim.data.qpos[sim._slide] == pytest.approx(depth_offset, abs=1e-12)
  assert sim.data.qpos[sim._hinge] == pytest.approx(
    2 * np.pi * depth_offset / config.THREAD_PITCH_M,
    abs=1e-12,
  )

  peak_linear = 0.0
  peak_angular = 0.0
  for _ in range(round(config.CAPTURE_ALIGNMENT_S / sim.timestep) + 10):
    sim.step()
    twist = sim.object_twist("bulb")
    peak_linear = max(peak_linear, float(np.linalg.norm(twist[:3])))
    peak_angular = max(peak_angular, float(np.linalg.norm(twist[3:])))
  _, lateral, final_tilt = sim._alignment()
  assert peak_linear < 0.02
  assert peak_angular < 0.35
  assert lateral < 0.00002
  assert final_tilt < np.deg2rad(0.1)
  assert np.isfinite(sim.data.qpos).all()


def test_real_torque_advances_pitch_and_reverse_releases(sim):
  sim.initialize_threaded()
  monitor = BulbScrewMonitor(sim)
  sim.data.xfrc_applied[sim.model.body("bulb").id, 5] = -config.MECHANICS_DEMO_TORQUE_NM
  sim.step(200)
  state = monitor.measure()
  assert state.engaged and 0.2 < state.clockwise_turns < config.TARGET_TURNS
  assert state.axial_travel_m > 0.0008
  assert abs(state.thread_error_m) < 0.0001
  assert not state.seated
  sim.data.xfrc_applied[sim.model.body("bulb").id, 5] = config.MECHANICS_DEMO_TORQUE_NM
  for _ in range(2000):
    previous_pose = sim.object_pose("bulb").copy()
    sim.step()
    if not sim.thread_engaged:
      released_pose = sim.object_pose("bulb")
      break
  assert not sim.thread_engaged
  assert np.linalg.norm(released_pose[:3] - previous_pose[:3]) < 0.001
  hinge = sim.model.joint("bulb_thread_angle")
  assert sim.data.qpos[sim._hinge] > hinge.range[0] + 0.1
  assert not sim.try_engage_thread()  # Must lift clear before recapturing.
  sim.reset()


def test_small_torque_compresses_rim_without_certifying_tightness(sim):
  sim.initialize_threaded()
  monitor = BulbScrewMonitor(sim)
  sim.data.xfrc_applied[sim.model.body("bulb").id, 5] = -0.03
  for _ in range(3000):
    sim.step()
    state = monitor.update()
  assert state.clockwise_turns > 0.9 * config.TARGET_TURNS
  assert state.cushion_contact_load_n > 1.0
  assert state.backstop_load_n < 0.01
  assert not state.seated and not state.success
  sim.reset()


def test_bottom_contact_and_continuous_dwell(sim):
  sim.initialize_threaded()
  monitor = BulbScrewMonitor(sim, auto_confirm_tightening=False)
  touched = False
  for _ in range(4500):
    sim.data.xfrc_applied[sim.model.body("bulb").id, 5] = (
      0 if touched else -config.MECHANICS_DEMO_TORQUE_NM
    )
    sim.step()
    state = monitor.update()
    touched |= state.backstop_load_n > 0.2
  assert touched
  assert state.success
  assert not sim.bulb_lit  # Seating alone does not confirm loaded tightening.
  qpos, qvel = sim.full_state()
  sim.confirm_tightening()
  assert sim.bulb_lit
  assert (
    sim.model.geom_matid[sim.model.geom("bulb_globe").id]
    == sim.model.material("bulb_glowing").id
  )
  np.testing.assert_array_equal(sim.data.qpos, qpos)
  np.testing.assert_array_equal(sim.data.qvel, qvel)
  assert state.clockwise_turns == pytest.approx(config.TARGET_TURNS, abs=0.08)
  assert state.backstop_load_n > 0.01
  assert state.exposed_thread_m == 0
  assert abs(state.shoulder_gap_m) < config.SEATED_SHOULDER_GAP_M
  assert state.shoulder_contact_load_n > 0.01
  # A raised bottom stop alone must never certify a long, exposed base as done.
  neck = sim.model.geom("bulb_neck").id
  position, size = sim.model.geom_pos[neck].copy(), sim.model.geom_size[neck].copy()
  crest = sim.model.geom(
    f"bulb_thread_crest_{config.EXTERNAL_THREAD_SEGMENTS - 1:03d}"
  ).id
  crest_position = sim.model.geom_pos[crest].copy()
  try:
    sim.model.geom_pos[neck, 2] = 0.035
    sim.model.geom_size[neck, 1] = 0.011
    incomplete = monitor.measure()
    assert incomplete.backstop_load_n > 0.01
    assert incomplete.shoulder_gap_m > 0.013
    assert not incomplete.seated
    sim.model.geom_pos[neck], sim.model.geom_size[neck] = position, size
    sim.model.geom_pos[crest, 2] += 0.014
    incomplete = monitor.measure()
    assert incomplete.exposed_thread_m > 0.013
    assert not incomplete.seated
  finally:
    sim.model.geom_pos[neck], sim.model.geom_size[neck] = position, size
    sim.model.geom_pos[crest] = crest_position
  assert not monitor.measure().success  # A measurement does not accrue time.
  sim.step(10)
  assert not monitor.update().success  # A sampling gap cannot prove dwell.
  assert sim.bulb_lit
  # Seated rim friction needs more torque than the loose-thread demo.
  sim.data.xfrc_applied[sim.model.body("bulb").id, 5] = 0.3
  for _ in range(1000):
    sim.step()
    if not sim.bulb_lit:
      break
  assert not sim.bulb_lit  # Backing the bulb out opens the visual circuit.
  sim._set_bulb_lit(True)
  sim.reset()
  assert not sim.bulb_lit
  assert (
    sim.model.geom_matid[sim.model.geom("bulb_globe").id]
    == sim.model.material("bulb_frosted").id
  )
  assert not monitor.update().success


def test_actual_finger_contact_drives_shared_tactile(sim):
  sim.reset()
  model, data = sim.model, sim.data
  layout = sim.genesis_probe_layout
  probes = GenesisProbeTactileProvider(model, layout)
  forces = SolverDistributedTactileProvider(model, layout)
  expected = {
    model.geom(n).id
    for n in (
      "bulb_globe",
      "bulb_neck",
      "bulb_screw_base",
      "bulb_tip_contact",
      *(f"bulb_thread_crest_{i:03d}" for i in range(config.EXTERNAL_THREAD_SEGMENTS)),
    )
  }
  assert set(probes._target_geom_ids) == set(forces._target_geom_ids) == expected
  name = "hand_r_index_link4"
  body = model.body(name).id
  index = FINGERTIP_LINK_NAMES.index(name)
  ids = np.flatnonzero(np.asarray(layout.body_names) == name)
  rotation = data.xmat[body].reshape(3, 3)
  normal = rotation @ layout.local_normal[ids].mean(axis=0)
  normal /= np.linalg.norm(normal)
  center = data.xpos[body] + rotation @ layout.local_pos[ids].mean(axis=0)
  # Align the ellipsoid's 29 mm X radius with the tactile pad normal.
  reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
  tangent = np.cross(normal, reference)
  tangent /= np.linalg.norm(tangent)
  object_rotation = np.column_stack((normal, tangent, np.cross(normal, tangent)))
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, object_rotation.ravel())
  position = center + normal * 0.0285 - object_rotation @ np.array([0, 0, 0.072])
  sim.set_object_pose("bulb", position, quat)
  depth, force = probes.read(data), forces.read(data)
  assert depth.contact[index]
  assert force.normal_force_n[index] > 0
  np.testing.assert_allclose(
    force.normal_taxel_force_n.sum(axis=(1, 2)), force.normal_force_n
  )
  assert all(a.target_geom_id in expected for a in force.assignments)
  sim.reset()


def test_mouth_cannot_capture_and_threads_have_real_contact(sim):
  sim.reset()
  sim.set_object_pose("bulb", config.SOCKET_MOUTH_POSITION_M)
  assert not sim.try_engage_thread()
  assert sim.thread_contact_load() == 0
  sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
  assert sim.thread_contact_load() > config.THREAD_CAPTURE_LOAD_N
  assert sim.try_engage_thread()
  state = BulbScrewMonitor(sim).measure()
  assert state.insertion_depth_m == pytest.approx(config.THREAD_ENTRY_DEPTH_M)
  assert state.thread_contact_load_n > 0
  # Removing the contact pair must prevent capture, even at the nominal pose.
  sim.reset()
  saved = sim.model.geom_contype.copy(), sim.model.geom_conaffinity.copy()
  try:
    ids = list(sim._external_thread_geoms | sim._internal_thread_geoms)
    sim.model.geom_contype[ids] = 0
    sim.model.geom_conaffinity[ids] = 0
    sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
    assert not sim.try_engage_thread()
  finally:
    sim.model.geom_contype[:], sim.model.geom_conaffinity[:] = saved
    sim.reset()


def test_funnel_contacts_and_guides_misaligned_bulb(sim):
  import xml.etree.ElementTree as ET

  scene = ET.parse(default_model_path("bulb-screw")).getroot()
  mesh = scene.find("asset/mesh[@name='bulb_funnel_sector']")
  assert mesh is not None
  vertices = np.asarray([float(v) for v in mesh.attrib["vertex"].split()]).reshape(
    -1, 3
  )
  bottom_radius = np.linalg.norm(vertices[0, :2])
  throat_radius = np.linalg.norm(vertices[4, :2])
  top_radius = np.linalg.norm(vertices[8, :2])
  outer_radius = np.linalg.norm(vertices[10, :2])
  assert throat_radius == pytest.approx(bottom_radius, abs=1e-8)
  assert throat_radius == pytest.approx(0.0168, abs=1e-6)
  assert top_radius == pytest.approx(0.030, abs=1e-6)
  assert outer_radius == pytest.approx(0.044, abs=1e-6)
  assert sim.model.geom("bulb_fixture_base").size[0] == pytest.approx(
    outer_radius, abs=1e-6
  )
  slope_angle = np.arctan2(vertices[8, 2] - vertices[4, 2], top_radius - throat_radius)
  assert slope_angle > np.deg2rad(60)

  # The 11 mm lateral error is outside the original socket wall. The sloped
  # collision faces must actually deflect a falling free bulb toward its axis.
  sim.reset()
  sim.set_object_pose("bulb", [0.631, -0.180, 0.738])
  funnel = {sim.model.geom(f"bulb_socket_funnel_{i:02d}").id for i in range(24)}
  assert any(
    sim.data.contact[i].geom1 in funnel or sim.data.contact[i].geom2 in funnel
    for i in range(sim.data.ncon)
  )
  sim.step(25)
  assert sim.object_pose("bulb")[0] < 0.630

  # The original entrance still admits a centered bulb and thread capture.
  sim.reset()
  sim.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
  assert not any(
    sim.data.contact[i].geom1 in funnel or sim.data.contact[i].geom2 in funnel
    for i in range(sim.data.ncon)
  )
  assert sim.try_engage_thread()
  sim.reset()


def test_axial_loading_does_not_create_nonhelical_steps(sim):
  sim.initialize_threaded()
  monitor = BulbScrewMonitor(sim)
  bulb = sim.model.body("bulb").id
  try:
    for load in [0, 3, -3, 0]:
      sim.data.xfrc_applied[bulb, 2] = load
      sim.step(250)
      state = monitor.measure()
      assert state.engaged
      assert abs(state.thread_error_m) < 0.00001
    assert state.thread_contact_load_n > 0.01
  finally:
    sim.reset()
