"""Dimensions, mechanical keying and continuous seating acceptance."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.tasks.install_ram import config
from kaihand_tactile_env.tasks.install_ram.task import (
  RamInstallationMonitor,
  RamInstallSimulation,
)


@pytest.fixture(scope="module")
def sim():
  return RamInstallSimulation(add_genesis_probes=False)


def place_bottom(sim, depth, *, xy=(0.0, 0.0), quaternion=(1, 0, 0, 0)):
  sim.reset()
  mouth = sim.data.site_xpos[sim.model.site("ram_socket_mouth").id].copy()
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, np.array(quaternion, dtype=float))
  local = sim.model.site("ram_bottom").pos
  centre = mouth + [*xy, -depth] - rotation.reshape(3, 3) @ local
  sim.set_object_pose("ram", centre, quaternion)
  # A geometry fixture starting inside the socket must also initialize the
  # passive banks' deflection; otherwise it starts with overlapping springs.
  if depth > 0.0032:
    for side in ("front", "back"):
      joint = sim.model.joint(f"ram_socket_spring_{side}_slide")
      sim.data.qpos[joint.qposadr[0]] = 0.0003
    mujoco.mj_forward(sim.model, sim.data)


def test_shared_configuration_and_real_scale(sim):
  root = ET.parse(default_model_path("install-ram")).getroot()
  assert [node.attrib["file"] for node in root.findall("include")] == [
    "../../shared/mjcf/robot.xml",
    "../../shared/mjcf/control.xml",
  ]
  assert root.find(".//camera[@name='head']") is None
  assert root.find(".//camera[@name='right_wrist']") is None
  assert sim.object_names == ("ram",)
  assert config.RAM_LENGTH_M == pytest.approx(0.13335)
  assert config.RAM_HEIGHT_M == pytest.approx(0.03125)
  assert 0.001 <= config.RAM_THICKNESS_M <= 0.0015
  assert sim.model.opt.timestep == pytest.approx(0.002)
  body = sim.model.body("ram").id
  weld = sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD
  assert not np.any(sim.model.eq_obj1id[weld] == body)
  assert not np.any(sim.model.eq_obj2id[weld] == body)


@pytest.mark.parametrize(
  "depth,xy,quaternion",
  [
    (-0.001, (0, 0), (1, 0, 0, 0)),
    (0.003, (0, 0), (1, 0, 0, 0)),
    (0.006, (0, 0.001), (1, 0, 0, 0)),
    (0.006, (0, 0), (0, 0, 0, 1)),
    (0.007, (0, 0), (1, 0, 0, 0)),
  ],
)
def test_incomplete_offset_reversed_and_overdeep_are_rejected(
  sim, depth, xy, quaternion
):
  place_bottom(sim, depth, xy=xy, quaternion=quaternion)
  state = RamInstallationMonitor(sim).measure()
  assert not state.seated
  assert not state.success


def test_aligned_slot_is_open_and_reversed_key_physically_collides(sim):
  place_bottom(sim, config.TARGET_INSERTION_DEPTH_M - 0.0002)
  monitor = RamInstallationMonitor(sim)
  aligned = monitor.measure()
  assert aligned.aperture_fits
  assert aligned.maximum_socket_penetration_m < 0.00005
  place_bottom(sim, config.TARGET_INSERTION_DEPTH_M - 0.0002, quaternion=(0, 0, 0, 1))
  reversed_state = monitor.measure()
  assert reversed_state.maximum_socket_penetration_m > 0.0005
  assert not reversed_state.seated


def test_free_module_cannot_claim_loaded_bottom_press(sim):
  place_bottom(sim, config.TARGET_INSERTION_DEPTH_M - 0.0002)
  monitor = RamInstallationMonitor(sim)
  for _ in range(500):
    sim.step()
    state = monitor.update()
  assert not state.success
  assert not state.bottom_out_confirmed
  assert state.backstop_load_n < config.BOTTOM_OUT_MIN_FORCE_N
  assert not monitor.measure().success  # A snapshot never certifies a dwell.
  sim.step(5)  # Missing physical observations invalidate the previous dwell.
  assert not monitor.update().success
  sim.reset()
  assert not monitor.update().success
  assert np.isfinite(sim.data.qpos).all()


def test_passive_slot_has_sliding_friction_then_loaded_backstop(sim):
  """A test-only fixture wrench characterizes the passive socket, not a policy."""
  place_bottom(sim, -0.001)
  monitor = RamInstallationMonitor(sim)
  body = sim.model.body("ram").id
  sliding, bottom = [], []
  try:
    for step in range(3500):
      t = step * sim.timestep
      depth_goal = -0.001 + min(t / 5, 1) * 0.009
      desired = (
        config.SOCKET_MOUTH_POSITION_M - config.RAM_BOTTOM_LOCAL_M - [0, 0, depth_goal]
      )
      pose, twist = sim.object_pose("ram"), sim.object_twist("ram")
      force = (
        1000 * (desired - pose[:3]) - 5 * twist[:3] + [0, 0, config.RAM_MASS_KG * 9.81]
      )
      force[2] = np.clip(force[2], -3.0, 3.0)
      sim.data.xfrc_applied[body, :3] = force
      sim.data.xfrc_applied[body, 3:] = -0.02 * pose[4:] - 0.0002 * twist[3:]
      sim.step()
      state = monitor.update()
      assert state.maximum_socket_penetration_m < config.MAX_SOCKET_PENETRATION_M
      if 0.0035 < state.insertion_depth_m < 0.0055:
        sliding.append(state.spring_axial_resistance_n)
        assert state.backstop_load_n == 0
      if t > 6:
        bottom.append(state.backstop_load_n)
    assert len(sliding) > 100 and len(bottom) > 100
    assert 0.4 < np.mean(sliding) < 0.9
    assert np.mean(bottom) > 2 * np.mean(sliding)
    assert state.bottom_out_confirmed and state.success
    sim.step(5)
    assert not monitor.update().bottom_out_confirmed
    sim.reset()
    assert not monitor.update().bottom_out_confirmed
  finally:
    sim.data.xfrc_applied[:] = 0


def test_initial_loading_pose_is_stable_without_false_seating(sim):
  sim.reset()
  monitor = RamInstallationMonitor(sim)
  for _ in range(500):
    sim.step()
  state = monitor.update()
  assert np.isfinite(sim.data.qpos).all()
  assert np.isfinite(sim.data.qvel).all()
  assert not state.success
  assert not state.seated


def test_full_installation_uses_contacts_and_remains_seated_after_release():
  from kaihand_tactile_env.tasks.install_ram.execution import RamInstallExecutor

  simulation = RamInstallSimulation(add_genesis_probes=False)
  executor = RamInstallExecutor(simulation)
  object_pose = simulation.object_pose("ram").copy()
  home_qpos = simulation.data.qpos.copy()
  executor.prepare()
  np.testing.assert_array_equal(simulation.data.qpos, home_qpos)
  np.testing.assert_array_equal(simulation.object_pose("ram"), object_pose)
  initial = simulation.data.qpos.copy()
  executor.prepare()
  np.testing.assert_array_equal(simulation.data.qpos, initial)
  body = simulation.model.body("ram").id
  dof = simulation._object_dofs["ram"]
  collision = simulation.model.geom_contype.copy()
  affinity = simulation.model.geom_conaffinity.copy()
  released = []
  axial_targets = []

  def observe(current, phase):
    assert np.isfinite(current.data.qpos).all()
    assert np.isfinite(current.data.qvel).all()
    assert not np.any(current.data.xfrc_applied[body])
    assert not np.any(current.data.qfrc_applied[dof : dof + 6])
    if phase == "approach":
      assert np.linalg.norm(current.object_pose("ram")[:3] - object_pose[:3]) < 0.0001
    if phase == "verify":
      released.append(executor.state.seated)
    if phase in {"insert", "bottom_press"}:
      axial_targets.append(
        np.r_[
          executor._command_position[:2],
          executor._command_rotation.ravel(),
          [current._hand_targets["right"][n] for n in executor._names],
        ]
      )
      assert executor.state.aperture_fits

  result = executor.run(observer=observe)
  assert result.success, result.reason
  assert "approach" in result.phases
  assert result.initialization == "open_hands_at_home_then_physical_approach"
  assert result.maximum_lift_m >= 0.02
  assert result.final_state.success
  assert result.final_state.bottom_out_confirmed
  assert result.bottom_press_duration_s >= config.BOTTOM_OUT_HOLD_S
  assert result.minimum_palm_down_cosine > 0.5
  assert result.final_state.stable_duration_s >= config.SEATED_DWELL_S
  assert max(result.final_fingertip_load_n) < 0.01
  assert released and all(released)
  assert executor._aligned_duration >= 0.2
  assert axial_targets
  np.testing.assert_allclose(
    np.asarray(axial_targets),
    np.broadcast_to(axial_targets[0], np.asarray(axial_targets).shape),
    rtol=0,
    atol=1e-12,
  )
  np.testing.assert_array_equal(simulation.model.geom_contype, collision)
  np.testing.assert_array_equal(simulation.model.geom_conaffinity, affinity)
  with pytest.raises(RuntimeError, match="fresh executor"):
    executor.run()
