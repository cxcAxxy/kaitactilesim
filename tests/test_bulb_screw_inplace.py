"""Actual reset mechanics plus ordered, independently measured success gates."""

from __future__ import annotations

import gc
import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.tasks.bulb_screw import config as screw
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewMonitor
from kaihand_tactile_env.tasks.bulb_screw_inplace import config
from kaihand_tactile_env.tasks.bulb_screw_inplace.task import (
  BulbScrewInplaceMonitor,
  BulbScrewInplaceSimulation,
  BulbScrewInplaceState,
)


@pytest.fixture(scope="module")
def simulation():
  sim = BulbScrewInplaceSimulation(position_seed=7, add_genesis_probes=False)
  yield sim
  del sim
  gc.collect()


def test_scene_preserves_v15_mechanics_and_removes_only_unused_cradle():
  actual = ET.parse(default_model_path(config.SCENE_NAME)).getroot()
  expected = ET.parse(default_model_path("bulb-screw")).getroot()
  expected.set("model", "kaihand_bulb_screw_inplace")
  world, asset = expected.find("worldbody"), expected.find("asset")
  world.remove(world.find("body[@name='bulb_pickup_cradle']"))
  asset.remove(asset.find("material[@name='bulb_cradle']"))

  def structure(node):
    return node.tag, node.attrib, (node.text or "").strip(), [structure(n) for n in node]

  assert structure(actual) == structure(expected)
  assert not any("cradle" in name for name in config.SCENE_GEOM_NAMES)
  inplace = mujoco.MjModel.from_xml_path(str(default_model_path(config.SCENE_NAME)))
  original = mujoco.MjModel.from_xml_path(str(default_model_path("bulb-screw")))
  assert inplace.nu == original.nu
  assert inplace.neq == original.neq
  for name in ("qpos0", "jnt_type", "jnt_range", "dof_damping", "eq_type", "eq_data"):
    np.testing.assert_array_equal(getattr(inplace, name), getattr(original, name))
  bulb = inplace.body("bulb").id
  assert inplace.body_gravcomp[bulb] == original.body_gravcomp[original.body("bulb").id]
  assert inplace.body_gravcomp[bulb] == 0


@pytest.mark.parametrize(
  "name",
  [
    "THREAD_PITCH_M", "THREAD_TRAVEL_M", "TARGET_TURNS", "CAPTURE_DEPTH_M",
    "CAPTURE_LATERAL_M", "THREAD_RELEASE_DEPTH_OFFSET_M", "TIGHTENING_HOLD_S",
    "TIGHTENING_MIN_TORQUE_NM", "TIGHTENING_MAX_ROTATION_RAD", "FINGER_STROKE_TURNS",
  ],
)
def test_shared_numerical_mechanics_are_unchanged(name):
  assert getattr(config, name) == getattr(screw, name)


def test_reset_is_unseated_unlit_threaded_and_physically_settled(simulation):
  sim = simulation
  sim.reset(position_seed=7)
  state = BulbScrewInplaceMonitor(sim).measure()
  assert isinstance(state, BulbScrewInplaceState)
  assert state.initially_threaded_verified and state.engaged
  assert not state.seated and not state.bulb_lit and not state.success
  assert not state.grasp_verified and not state.effective_rotation_verified
  assert not state.tightening_verified and not state.released_stably
  assert not state.home_return_verified
  assert state.robot_contact_count == 0
  assert sim.data.time == 0
  assert sim.initialization["prezero_physically_settled"] is True
  assert sim.initialization["model_parameters_changed"] is False
  assert sim.initialization["position_randomization_applied"] is False
  assert sim.initialization["prezero_linear_speed_m_s"] <= config.SETTLED_LINEAR_SPEED_M_S
  assert sim.initialization["prezero_angular_speed_rad_s"] <= config.SETTLED_ANGULAR_SPEED_RAD_S
  assert state.clockwise_turns == sim.initialization["initial_clockwise_turns"]
  assert state.insertion_depth_m == sim.initialization["initial_insertion_depth_m"]
  np.testing.assert_allclose(sim.data.qpos[sim._arm_qpos["right"]], config.ARM_HOME["right"], atol=1e-5)
  assert sim.initial_position_randomization["enabled"] is False
  assert sim.model.body_gravcomp[sim.model.body("bulb").id] == 0
  # The shared arm controller retains its existing robot gravity compensation;
  # neither the bulb nor its passive carriage receives an external task force.
  object_dofs = slice(sim._object_dofs["bulb"], sim._object_dofs["bulb"] + 6)
  assert not sim.data.qfrc_applied[object_dofs].any()
  assert not sim.data.qfrc_applied[[sim._slide_dof, sim._hinge_dof]].any()
  assert not sim.data.xfrc_applied.any()


def test_idle_does_not_complete_or_autoslide_and_never_writes_bulb_pose(simulation, monkeypatch):
  sim = simulation
  sim.reset()
  monitor = BulbScrewInplaceMonitor(sim)
  angle = monitor.measure().clockwise_turns

  def forbidden(*args, **kwargs):
    raise AssertionError("an episode step must not overwrite the object pose")

  monkeypatch.setattr(sim, "set_object_pose", forbidden)
  for _ in range(round(0.5 / sim.timestep)):
    sim.step()
    state = monitor.update()
  assert state.engaged and not state.seated and not state.bulb_lit
  assert not state.grasp_verified and not state.success
  assert abs(state.clockwise_turns - angle) * 2 * np.pi < config.INITIAL_MAX_ROTATION_RAD
  object_dofs = slice(sim._object_dofs["bulb"], sim._object_dofs["bulb"] + 6)
  assert not sim.data.qfrc_applied[object_dofs].any()
  assert not sim.data.qfrc_applied[[sim._slide_dof, sim._hinge_dof]].any()
  assert not sim.data.xfrc_applied.any()


@pytest.mark.parametrize("refresh", ["update", "verify_tightening"])
def test_monitor_idempotence_and_reset_invalidation_at_same_timestamp(simulation, refresh):
  sim = simulation
  sim.reset()
  monitor = BulbScrewInplaceMonitor(sim)
  first = monitor.update()
  assert monitor.update() is first
  monitor._grasp_verified = monitor._effective_rotation_verified = True
  monitor._tightening_certified = True
  serial = sim.reset_serial
  sim.reset()
  assert sim.reset_serial == serial + 1
  fresh = getattr(monitor, refresh)()
  assert fresh.timestamp == 0
  assert not fresh.grasp_verified and not fresh.tightening_verified and not fresh.success


def _synthetic_stream(sim, monkeypatch):
  sim.reset()
  monitor = BulbScrewInplaceMonitor(sim, auto_confirm_tightening=False)
  baseline = BulbScrewMonitor(sim, auto_confirm_tightening=False).measure()
  signal = {
    "engaged": True,
    "seated": False,
    "bulb_lit": False,
    "fingertip_normal_force_n": (5.0,) * 5,
    "hand_clockwise_torque_nm": 0.12,
    "right_hand_contact_count": 5,
    "right_hand_nonpad_contact_count": 0,
    "robot_contact_count": 5,
    "released": False,
  }
  home = {"now": False}

  def measure():
    raw = replace(baseline, timestamp=float(sim.data.time), **signal)
    return replace(monitor._extra(raw), hand_home_now=home["now"])

  monkeypatch.setattr(monitor, "measure", measure)
  monkeypatch.setattr(sim, "confirm_tightening", lambda: sim._set_bulb_lit(True))

  def advance(duration):
    state = monitor.update()
    for _ in range(round(duration / sim.timestep)):
      sim.data.time += sim.timestep
      state = monitor.update()
    return state

  return monitor, signal, home, advance


def test_complete_physical_chain_does_not_require_home_and_light_is_not_proof(simulation, monkeypatch):
  monitor, signal, home, advance = _synthetic_stream(simulation, monkeypatch)
  state = advance(0.12)
  assert state.grasp_verified and not state.effective_rotation_verified
  signal.update(
    clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    seated=True,
    bulb_lit=True,
  )
  state = advance(0.01)
  assert state.effective_rotation_verified
  assert not state.tightening_verified and not state.success
  state = advance(0.32)
  assert state.tightening_verified
  assert not state.success
  confirmed = monitor.verify_tightening()
  assert confirmed.tightening_verified and confirmed.bulb_lit
  assert monitor.verify_tightening() is confirmed
  signal.update(
    fingertip_normal_force_n=(0.0,) * 5,
    right_hand_contact_count=0,
    robot_contact_count=0,
    released=True,
  )
  state = advance(0.21)
  assert state.released_stably and state.physical_success and state.success
  assert not state.home_return_verified
  home["now"] = True
  state = advance(0.11)
  assert state.home_return_verified and state.success
  signal["robot_contact_count"] = 1
  signal["released"] = False
  state = advance(simulation.timestep)
  assert not state.released_stably and not state.physical_success and not state.success


@pytest.mark.parametrize("missing", ["no_grasp", "no_rotation", "no_torque", "one_pad", "hidden_support"])
def test_invalid_physical_chain_cannot_succeed(simulation, monkeypatch, missing):
  monitor, signal, _, advance = _synthetic_stream(simulation, monkeypatch)
  if missing == "no_grasp":
    signal["fingertip_normal_force_n"] = (0.0,) * 5
  advance(0.12)
  signal.update(seated=True, bulb_lit=True)
  if missing != "no_rotation":
    signal.update(
      clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
      insertion_depth_m=screw.SEATED_DEPTH_M,
    )
  if missing == "no_torque":
    signal["hand_clockwise_torque_nm"] = 0.0
  if missing == "one_pad":
    signal["fingertip_normal_force_n"] = (5.0, 5.0, 0.0, 5.0, 5.0)
  if missing == "hidden_support":
    signal["right_hand_nonpad_contact_count"] = 1
  state = advance(0.4)
  assert not state.tightening_verified
  signal.update(released=True, robot_contact_count=0, right_hand_contact_count=0)
  state = advance(0.25)
  assert not state.success


def test_observation_gap_cannot_complete_a_loaded_stall_window(simulation, monkeypatch):
  monitor, signal, _, advance = _synthetic_stream(simulation, monkeypatch)
  advance(0.12)
  signal.update(
    clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    seated=True,
  )
  advance(0.15)
  simulation.data.time += 1.0
  state = monitor.update()
  assert not state.tightening_verified
  state = advance(0.16)
  assert not state.tightening_verified


def test_two_sparse_loaded_samples_do_not_prove_continuous_stall(simulation, monkeypatch):
  monitor, signal, _, advance = _synthetic_stream(simulation, monkeypatch)
  advance(0.12)
  signal.update(
    clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    seated=True,
  )
  advance(2 * simulation.timestep)
  assert len(monitor._tightening_history) == 1
  simulation.data.time += screw.TIGHTENING_HOLD_S
  state = monitor.update()
  assert len(monitor._tightening_history) == 1
  assert not state.tightening_verified
  assert state.tightening_stall_duration_s == 0


@pytest.mark.parametrize("loss", ["disengaged", "backed_out"])
def test_real_thread_or_seat_loss_requires_new_loaded_stall(simulation, monkeypatch, loss):
  monitor, signal, _, advance = _synthetic_stream(simulation, monkeypatch)
  advance(0.12)
  signal.update(
    clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    seated=True,
  )
  assert advance(0.34).tightening_verified
  signal["seated"] = False
  if loss == "disengaged":
    signal["engaged"] = False
  else:
    signal["insertion_depth_m"] -= 3 * screw.SEATED_DEPTH_TOLERANCE_M
  assert not advance(simulation.timestep).tightening_verified
  # Re-seating, re-lighting and releasing alone must not resurrect the old
  # torque/stall certificate. A new complete loaded window is mandatory.
  signal.update(
    engaged=True,
    seated=True,
    bulb_lit=True,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    hand_clockwise_torque_nm=0.0,
    fingertip_normal_force_n=(0.0,) * 5,
    right_hand_contact_count=0,
    robot_contact_count=0,
    released=True,
  )
  state = advance(0.4)
  assert not state.tightening_verified and not state.success
  signal.update(
    hand_clockwise_torque_nm=0.12,
    fingertip_normal_force_n=(5.0,) * 5,
    right_hand_contact_count=5,
    robot_contact_count=5,
    released=False,
  )
  assert advance(0.32).tightening_verified


def test_brief_speed_or_contact_fluctuation_preserves_measured_stall(simulation, monkeypatch):
  monitor, signal, _, advance = _synthetic_stream(simulation, monkeypatch)
  advance(0.12)
  signal.update(
    clockwise_turns=simulation.initialization["initial_clockwise_turns"] + screw.TARGET_TURNS,
    insertion_depth_m=screw.SEATED_DEPTH_M,
    seated=True,
  )
  assert advance(0.34).tightening_verified
  signal.update(seated=False, angular_speed_rad_s=0.1)
  assert advance(0.02).tightening_verified
  signal.update(seated=True, angular_speed_rad_s=0.0)
  signal.update(
    fingertip_normal_force_n=(0.0,) * 5,
    right_hand_contact_count=0,
    robot_contact_count=0,
    released=True,
  )
  state = advance(0.22)
  assert state.tightening_verified and state.success
