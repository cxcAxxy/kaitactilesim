from __future__ import annotations

import gc
import math
import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewMonitor
from kaihand_tactile_env.tasks.bulb_unscrew import config
from kaihand_tactile_env.tasks.bulb_unscrew.execution import BulbUnscrewExecutor
from kaihand_tactile_env.tasks.bulb_unscrew.task import (
  BulbUnscrewMonitor,
  BulbUnscrewSimulation,
)

MILESTONE_FIELDS = (
  "initially_installed_verified",
  "grasp_verified",
  "breakaway_verified",
  "thread_disengaged",
  "disengagement_valid",
  "fully_extracted",
  "placed_on_table",
  "released",
  "hand_retreated",
)


@pytest.fixture(scope="module")
def simulation():
  sim = BulbUnscrewSimulation(position_seed=7, add_genesis_probes=False)
  yield sim
  del sim
  gc.collect()


def test_executor_run_exactly_dispatches_to_execute(simulation, monkeypatch):
  executor = BulbUnscrewExecutor(simulation)
  expected = object()
  calls = []

  def execute():
    calls.append(executor)
    return expected

  monkeypatch.setattr(executor, "execute", execute)
  assert callable(executor.run)
  assert executor.run() is expected
  assert calls == [executor]


def test_scene_removes_only_return_cradle_and_preserves_socket_bulb_mechanics():
  scene_path = default_model_path(config.SCENE_NAME)
  root = ET.parse(scene_path).getroot()
  assert root.tag == "mujoco"
  assert root.attrib == {"model": "kaihand_bulb_unscrew"}
  screw_root = ET.parse(default_model_path("bulb-screw")).getroot()
  world = screw_root.find("worldbody")
  asset = screw_root.find("asset")
  assert world is not None and asset is not None
  world.remove(world.find("body[@name='bulb_pickup_cradle']"))
  asset.remove(asset.find("material[@name='bulb_cradle']"))
  screw_root.set("model", "kaihand_bulb_unscrew")

  def tree_structure(element):
    return (
      element.tag,
      element.attrib,
      (element.text or "").strip(),
      [tree_structure(child) for child in element],
    )

  assert tree_structure(root) == tree_structure(screw_root)
  assert "bulb_pickup_cradle" not in scene_path.read_text()
  assert not any("cradle" in name for name in config.SCENE_GEOM_NAMES)

  unscrew = mujoco.MjModel.from_xml_path(str(scene_path))
  screw = mujoco.MjModel.from_xml_path(str(default_model_path("bulb-screw")))
  for name in (
    "nq",
    "nv",
    "nu",
    "njnt",
    "neq",
    "nsensor",
  ):
    assert getattr(unscrew, name) == getattr(screw, name)
  assert unscrew.nbody == screw.nbody - 1
  assert unscrew.ngeom == screw.ngeom - 4
  for name in (
    "qpos0",
    "jnt_type",
    "jnt_axis",
    "jnt_range",
    "dof_damping",
    "eq_type",
    "eq_data",
    "actuator_trnid",
    "actuator_ctrlrange",
  ):
    np.testing.assert_array_equal(getattr(unscrew, name), getattr(screw, name))
  for index in range(unscrew.nbody):
    name = mujoco.mj_id2name(unscrew, mujoco.mjtObj.mjOBJ_BODY, index)
    other = screw.body(name).id
    for field in ("body_pos", "body_quat", "body_mass", "body_inertia"):
      np.testing.assert_array_equal(
        getattr(unscrew, field)[index], getattr(screw, field)[other]
      )
  # Preserve anonymous robot visual geoms too, not just named task geoms.
  retained_geoms = screw.geom_bodyid != screw.body("bulb_pickup_cradle").id
  for field in (
    "geom_type",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_contype",
    "geom_conaffinity",
  ):
    np.testing.assert_array_equal(
      getattr(unscrew, field), getattr(screw, field)[retained_geoms]
    )


def test_installed_reset_is_physical_loaded_and_uses_only_native_capture(simulation):
  sim = simulation
  sim.reset(position_seed=7)
  model, data = sim.model, sim.data
  initialization = sim.initialization
  assert initialization is not None
  assert initialization["schema"] == "bulb_unscrew_installed_reset_v1"
  assert initialization["time_s"] == 0.0
  assert initialization["thread_capture_active"] is True
  assert initialization["settled_under_physics_s"] == config.INSTALLED_SETTLE_S
  assert initialization["velocity_zeroed"] is True
  assert initialization["physically_validated"] is True
  assert initialization["model_parameters_changed"] is False
  assert sim.destination_randomization["support_body"] is None
  assert sim.destination_randomization["support_body_position_m"] is None
  assert config.MECHANICS_VERSION.endswith("_reverse_tabletop_v2")
  assert data.time == 0.0
  np.testing.assert_array_equal(data.qvel, np.zeros(model.nv))
  np.testing.assert_array_equal(data.qfrc_applied, np.zeros(model.nv))
  np.testing.assert_array_equal(data.xfrc_applied, np.zeros((model.nbody, 6)))

  capture = model.eq("bulb_thread_capture").id
  welds = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
  np.testing.assert_array_equal(welds, [capture])
  assert model.eq_obj1id[capture] == model.body("bulb_thread_carriage").id
  assert model.eq_obj2id[capture] == model.body("bulb").id
  np.testing.assert_array_equal(model.eq_data[capture, 3:10], (0, 0, 0, 1, 0, 0, 0))
  assert data.eq_active[capture]
  assert initialization["thread_depth_qpos_m"] == pytest.approx(
    data.qpos[sim._slide], abs=0.0
  )
  assert initialization["thread_angle_qpos_rad"] == pytest.approx(
    data.qpos[sim._hinge], abs=0.0
  )

  carriage = model.body("bulb_thread_carriage").id
  bulb_pose = sim.object_pose("bulb")
  assert np.linalg.norm(bulb_pose[:3] - data.xpos[carriage]) < 0.0001
  assert abs(float(np.dot(bulb_pose[3:], data.xquat[carriage]))) > 1 - 1e-8
  assert np.linalg.norm(bulb_pose[:3] - sim.table_target_position) > 0.1
  np.testing.assert_array_equal(
    sim.destination_pose_wxyz, initialization["destination_pose_wxyz"]
  )

  physical = BulbScrewMonitor(sim, auto_confirm_tightening=False).measure()
  assert physical.engaged and physical.seated and physical.bulb_lit
  assert physical.backstop_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
  assert physical.shoulder_contact_load_n >= config.INSTALLED_MIN_CONTACT_LOAD_N
  assert abs(physical.thread_error_m) <= config.SEATED_DEPTH_TOLERANCE_M
  assert physical.robot_contact_count == 0
  assert physical.linear_speed_m_s <= config.INSTALLED_LINEAR_SPEED_M_S
  assert physical.angular_speed_rad_s <= config.INSTALLED_ANGULAR_SPEED_RAD_S
  assert sim.bulb_lit
  assert (
    model.geom_matid[model.geom("bulb_globe").id] == model.material("bulb_glowing").id
  )

  state = BulbUnscrewMonitor(sim).measure()
  assert state.initially_installed_now
  assert state.engaged and state.seated and state.bulb_lit
  assert not any(getattr(state, name) for name in MILESTONE_FIELDS)
  assert state.failure_reason is None
  assert not state.success


def test_repeated_installed_reset_is_deterministic_and_restarts_clock(simulation):
  sim = simulation
  sim.reset(position_seed=17)

  def snapshot():
    return {
      "qpos": sim.data.qpos.copy(),
      "qvel": sim.data.qvel.copy(),
      "qacc": sim.data.qacc.copy(),
      "eq_data": sim.model.eq_data.copy(),
      "eq_active": sim.data.eq_active.copy(),
      "bulb_pose": sim.object_pose("bulb").copy(),
      "destination_pose": sim.destination_pose_wxyz.copy(),
      "initialization": dict(sim.initialization),
    }

  expected = snapshot()
  for steps in (7, 3):
    sim.step(steps)
    assert sim.data.time == pytest.approx(steps * sim.timestep, abs=1e-12)
    sim.reset()
    assert sim.data.time == 0.0
    actual = snapshot()
    for name in (
      "qpos",
      "qvel",
      "qacc",
      "eq_data",
      "eq_active",
      "bulb_pose",
      "destination_pose",
    ):
      np.testing.assert_array_equal(actual[name], expected[name])
    assert actual["initialization"] == expected["initialization"]


def test_placement_requires_actual_table_support_at_the_unassisted_target(
  simulation, monkeypatch
):
  sim = simulation
  sim.reset(position_seed=17)
  assert (
    mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "bulb_pickup_cradle") == -1
  )
  nominal = config.BULB_INITIAL_POSITION_M
  target = sim.table_target_position
  assert np.all(np.abs(target[:2] - nominal[:2]) <= 0.002)
  assert target[2] == nominal[2]
  monitor = BulbUnscrewMonitor(sim)
  physical = replace(
    monitor._instant.measure(),
    engaged=False,
    bulb_lit=False,
    bulb_pose_wxyz=tuple(np.r_[target, [1, 0, 0, 0]]),
    tilt_rad=0.0,
    linear_speed_m_s=0.0,
    angular_speed_rad_s=0.0,
  )
  monkeypatch.setattr(monitor._instant, "measure", lambda: physical)
  support = [(0.0, 0, 0.0, 0)]
  monkeypatch.setattr(sim, "_support_contact_metrics", lambda: support[0])
  assert not monitor.measure().at_table_target_now
  support[0] = (0.0, 0, config.PLACEMENT_MIN_SUPPORT_LOAD_N, 1)
  assert monitor.measure().at_table_target_now
  support[0] = (0.0, 0, config.PLACEMENT_MIN_SUPPORT_LOAD_N, 0)
  assert not monitor.measure().at_table_target_now
  support[0] = (0.0, 0, config.MAX_PLACEMENT_SUPPORT_LOAD_N + 0.01, 1)
  assert not monitor.measure().at_table_target_now
  support[0] = (0.2, 1, config.PLACEMENT_MIN_SUPPORT_LOAD_N, 1)
  assert not monitor.measure().at_table_target_now


def test_post_reset_steps_cannot_use_pose_writes_wrenches_or_new_welds(
  simulation, monkeypatch
):
  sim = simulation
  sim.reset(position_seed=19)
  model, data = sim.model, sim.data
  equality = {
    name: getattr(model, name).copy()
    for name in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data")
  }
  equality_active = data.eq_active.copy()
  bulb_dof = int(model.joint("bulb_freejoint").dofadr[0])
  last_qpos, last_qvel = data.qpos.copy(), data.qvel.copy()
  actual_step = mujoco.mj_step
  step_count = 0

  def forbidden(*_args, **_kwargs):
    raise AssertionError("normal task stepping attempted an object-state shortcut")

  monkeypatch.setattr(sim, "set_object_pose", forbidden)
  monkeypatch.setattr(sim, "restore_full_state", forbidden)

  def assert_unassisted():
    assert not np.any(data.xfrc_applied)
    assert not np.any(data.qfrc_applied[bulb_dof : bulb_dof + 6])
    np.testing.assert_array_equal(data.eq_active, equality_active)
    for name, expected in equality.items():
      np.testing.assert_array_equal(getattr(model, name), expected)

  def audited_step(observed_model, observed_data, *args, **kwargs):
    nonlocal step_count
    assert observed_model is model and observed_data is data
    np.testing.assert_array_equal(data.qpos, last_qpos)
    np.testing.assert_array_equal(data.qvel, last_qvel)
    assert_unassisted()
    actual_step(model, data, *args, **kwargs)
    np.copyto(last_qpos, data.qpos)
    np.copyto(last_qvel, data.qvel)
    step_count += 1
    assert_unassisted()

  monkeypatch.setattr(mujoco, "mj_step", audited_step)
  sim.step(5)
  assert step_count == 5


def test_physics_observer_runs_once_after_each_requested_substep_and_not_reset():
  observations = []

  def observe(sim):
    observations.append((sim, float(sim.data.time), sim.data.qpos.copy()))

  sim = BulbUnscrewSimulation(
    position_seed=23,
    physics_observer=observe,
    add_genesis_probes=False,
  )
  assert observations == []
  sim.step(7)
  assert len(observations) == 7
  assert all(observed is sim for observed, _, _ in observations)
  np.testing.assert_allclose(
    [timestamp for _, timestamp, _ in observations],
    sim.timestep * np.arange(1, 8),
    atol=1e-12,
    rtol=0,
  )
  count = len(observations)
  sim.reset()
  assert len(observations) == count
  assert sim.data.time == 0.0
  sim.step(3)
  assert len(observations) == count + 3
  np.testing.assert_allclose(
    [timestamp for _, timestamp, _ in observations[-3:]],
    sim.timestep * np.arange(1, 4),
    atol=1e-12,
    rtol=0,
  )


def test_monitor_requires_contiguous_samples_for_installed_history(simulation):
  sim = simulation
  sim.reset(position_seed=29)
  monitor = BulbUnscrewMonitor(sim)
  for _ in range(5):
    assert not monitor.measure().initially_installed_verified
  assert not monitor.update().initially_installed_verified

  sim.step(2)
  state = monitor.update()
  assert state.initially_installed_now
  assert not state.initially_installed_verified

  continuous_steps = math.ceil(config.INITIAL_STABLE_DWELL_S / sim.timestep) + 1
  for _ in range(continuous_steps):
    sim.step()
    state = monitor.update()
  assert state.initially_installed_verified
  assert not state.grasp_verified
  assert not state.success

  sim.step(2)
  state = monitor.update()
  assert state.initially_installed_now
  assert not state.initially_installed_verified
  assert not any(getattr(state, name) for name in MILESTONE_FIELDS)


def test_monitor_rejects_terminal_pose_after_early_disengagement(
  simulation, monkeypatch
):
  monitor = BulbUnscrewMonitor(simulation)
  current = [
    replace(
      monitor.measure(),
      timestamp=0.0,
      engaged=True,
      seated=True,
      bulb_lit=True,
      clockwise_turns=config.TARGET_TURNS,
      turns_removed=0.0,
      initially_installed_now=False,
      five_finger_grasp_now=False,
      fully_clear_now=False,
      at_table_target_now=False,
      released_now=False,
      hand_home_now=False,
    )
  ]
  monkeypatch.setattr(monitor, "measure", lambda: current[0])
  monitor.update()

  current[0] = replace(
    current[0],
    timestamp=simulation.timestep,
    engaged=False,
    seated=False,
    bulb_lit=False,
    fully_clear_now=True,
    at_table_target_now=True,
    released_now=True,
    hand_home_now=True,
  )
  state = monitor.update()
  assert state.failure_reason == (
    "thread disengaged before verified counterclockwise removal"
  )
  assert not any(getattr(state, name) for name in MILESTONE_FIELDS)
  assert not state.success


def test_monitor_accepts_only_the_complete_ordered_history(simulation, monkeypatch):
  sim = simulation
  monitor = BulbUnscrewMonitor(sim)
  current = [
    replace(
      monitor.measure(),
      timestamp=0.0,
      engaged=True,
      seated=True,
      bulb_lit=True,
      clockwise_turns=config.TARGET_TURNS,
      turns_removed=0.0,
      hand_counterclockwise_torque_nm=0.0,
      initially_installed_now=True,
      five_finger_grasp_now=False,
      fully_clear_now=False,
      at_table_target_now=False,
      released_now=False,
      hand_home_now=False,
    )
  ]
  monkeypatch.setattr(monitor, "measure", lambda: current[0])
  timestamp = 0.0

  def tick(**changes):
    nonlocal timestamp
    current[0] = replace(current[0], timestamp=timestamp, **changes)
    state = monitor.update()
    timestamp += sim.timestep
    return state

  def dwell(duration, **changes):
    state = None
    for _ in range(math.ceil(duration / sim.timestep) + 2):
      state = tick(**changes)
    return state

  state = dwell(config.INITIAL_STABLE_DWELL_S)
  assert state.initially_installed_verified
  assert not state.grasp_verified

  state = dwell(config.GRASP_STABLE_DWELL_S, five_finger_grasp_now=True)
  assert state.grasp_verified
  assert not state.breakaway_verified

  breakaway_removed = config.BREAKAWAY_MIN_TURNS + 1e-4
  state = tick(
    seated=False,
    bulb_lit=False,
    initially_installed_now=False,
    clockwise_turns=config.TARGET_TURNS - breakaway_removed,
    turns_removed=breakaway_removed,
    hand_counterclockwise_torque_nm=(config.BREAKAWAY_MIN_COUNTERCLOCKWISE_TORQUE_NM),
  )
  assert state.breakaway_verified
  assert not state.thread_disengaged

  release_removed = config.EXPECTED_RELEASE_TURNS - config.RELEASE_TURN_TOLERANCE + 1e-4
  tick(
    clockwise_turns=config.TARGET_TURNS - release_removed,
    turns_removed=release_removed,
  )
  state = tick(engaged=False)
  assert state.thread_disengaged and state.disengagement_valid
  assert state.failure_reason is None
  assert not state.fully_extracted

  state = dwell(config.FULL_CLEAR_DWELL_S, fully_clear_now=True)
  assert state.fully_extracted
  assert not state.placed_on_table

  state = dwell(
    config.PLACEMENT_DWELL_S,
    fully_clear_now=False,
    at_table_target_now=True,
  )
  assert state.placed_on_table
  assert not state.released

  state = dwell(config.RELEASE_DWELL_S, released_now=True)
  assert state.released
  assert not state.hand_retreated

  state = dwell(config.RETREAT_DWELL_S, hand_home_now=True)
  assert all(getattr(state, name) for name in MILESTONE_FIELDS)
  assert state.failure_reason is None
  assert state.success
