from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config
from kaihand_tactile_env.tasks.usb_insert.task import UsbInsertionMonitor


@pytest.fixture(scope="module")
def simulation() -> ArmHandSimulation:
  return ArmHandSimulation(scene="usb-insert", add_genesis_probes=False)


def place_tip(simulation, depth, *, lateral=0.0, quaternion=(1, 0, 0, 0)):
  mouth = simulation.model.site("usb_socket_mouth").id
  socket_rotation = simulation.data.site_xmat[mouth].reshape(3, 3)
  world_quaternion = np.empty(4)
  mujoco.mju_mulQuat(
    world_quaternion,
    simulation.data.xquat[simulation.model.body("usb_socket").id],
    np.asarray(quaternion, dtype=float),
  )
  rotation = np.zeros(9)
  mujoco.mju_quat2Mat(rotation, world_quaternion)
  position = (
    simulation.data.site_xpos[mouth]
    + socket_rotation @ [depth, lateral, 0.0]
    - rotation.reshape(3, 3) @ config.PLUG_TIP_LOCAL_M
  )
  simulation.set_object_pose("usb_plug", position, world_quaternion)


def synthetic_monitor(
  monkeypatch,
  *,
  depth=0.012,
  contacts=(("backstop", 0.18, 0.0, False, -0.00001),),
  axial_speed=0.0,
):
  """Inject known solver wrenches to isolate measurement and seating logic.

  The physical insertion test below still exercises the actual spring and
  bottom geometries. These synthetic contacts never advance a simulation.
  """
  names = (
    "usb_plug_handle",
    "usb_socket_backstop",
    "usb_socket_spring_test_contact",
    "usb_socket_wall_left",
    "usb_socket_tongue",
  )
  geom_ids = {
    name: index + 1
    for index, name in enumerate(("backstop", "spring", "wall", "tongue"))
  }
  model = SimpleNamespace(
    nbody=4,
    ngeom=len(names),
    body_parentid=np.array([0, 0, 0, 2]),
    geom_bodyid=np.array([1, 2, 3, 2, 2]),
    body=lambda name: SimpleNamespace(id={"usb_plug": 1, "usb_socket": 2}[name]),
    site=lambda name: SimpleNamespace(
      id={"usb_socket_mouth": 0, "usb_plug_tip": 1}[name]
    ),
    geom=lambda index: SimpleNamespace(name=names[index]),
  )
  rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
  frames = {
    "backstop": np.diag([-1.0, -1.0, 1.0]),
    "spring": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "wall": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "tongue": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
  }
  solver_contacts, wrenches = [], []
  for kind, normal, friction, reverse, distance in contacts:
    frame = frames[kind].copy()
    if reverse:
      frame[:2] *= -1
    solver_contacts.append(
      SimpleNamespace(
        geom1=0 if reverse else geom_ids[kind],
        geom2=geom_ids[kind] if reverse else 0,
        frame=(frame @ rotation.T).ravel(),
        dist=distance,
      )
    )
    wrenches.append(np.array([normal, friction, 0.0, 0.0, 0.0, 0.0]))
  data = SimpleNamespace(
    time=0.0,
    ncon=len(solver_contacts),
    contact=solver_contacts,
    site_xmat=np.tile(rotation.ravel(), (2, 1)),
    site_xpos=np.stack((np.zeros(3), rotation @ [depth, 0.0, 0.0])),
    xmat=np.tile(rotation.ravel(), (4, 1)),
  )
  twist = np.r_[rotation @ [axial_speed, 0.0, 0.0], np.zeros(3)]
  simulation = SimpleNamespace(
    scene="usb-insert",
    model=model,
    data=data,
    timestep=0.002,
    object_twist=lambda name: twist.copy(),
    _test_twist=twist,
  )
  monkeypatch.setattr(mujoco, "mj_forward", lambda model, data: None)
  monkeypatch.setattr(
    mujoco,
    "mj_contactForce",
    lambda model, data, index, output: np.copyto(output, wrenches[index]),
  )
  return UsbInsertionMonitor(simulation)


def test_socket_opens_upwards_and_insertion_axis_points_down(simulation):
  simulation.reset()
  mouth = simulation.model.site("usb_socket_mouth").id
  rotation = simulation.data.site_xmat[mouth].reshape(3, 3)
  np.testing.assert_allclose(rotation[:, 0], [0, 0, -1], atol=1e-12)
  np.testing.assert_allclose(
    simulation.data.site_xpos[mouth], config.SOCKET_MOUTH_POSITION_M
  )
  np.testing.assert_allclose(
    simulation.model.body("usb_socket").quat, config.SOCKET_QUATERNION_WXYZ
  )


def test_free_plug_settles_on_shared_table_without_false_success(simulation):
  simulation.reset()
  monitor = UsbInsertionMonitor(simulation)
  simulation.step(1000)
  state = monitor.update()
  assert np.isfinite(simulation.data.qpos).all()
  assert np.isfinite(simulation.data.qvel).all()
  assert simulation.object_pose("usb_plug")[2] == pytest.approx(0.686, abs=0.0001)
  assert state.linear_speed_m_s < 1e-5
  assert state.socket_contact_count == 0
  assert not state.success
  assert simulation.model.body("usb_plug").mass[0] == pytest.approx(0.018)
  body_id = simulation.model.body("usb_plug").id
  weld = simulation.model.eq_type == mujoco.mjtEq.mjEQ_WELD
  assert not np.any(simulation.model.eq_obj1id[weld] == body_id)
  assert not np.any(simulation.model.eq_obj2id[weld] == body_id)


@pytest.mark.parametrize(
  "depth,lateral,quaternion",
  [
    (0.0112, 0.0, (1, 0, 0, 0)),
    (-0.001, 0.0, (1, 0, 0, 0)),
    (0.005, 0.0, (1, 0, 0, 0)),
    (0.0112, 0.0006, (1, 0, 0, 0)),
    (0.0112, 0.0, (0, 1, 0, 0)),
    (0.013, 0.0, (1, 0, 0, 0)),
  ],
)
def test_seating_requires_depth_alignment_keying_and_clearance(
  simulation, depth, lateral, quaternion
):
  simulation.reset()
  place_tip(simulation, depth, lateral=lateral, quaternion=quaternion)
  state = UsbInsertionMonitor(simulation).measure()
  assert not state.seated
  # No single snapshot alone qualifies as sustained insertion success.
  assert not state.success
  if lateral != 0 or quaternion != (1, 0, 0, 0) or depth > 0.0121:
    assert state.maximum_socket_penetration_m > 0.0001
  if depth < config.BACKSTOP_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M:
    assert not state.backstop_contact


@pytest.mark.parametrize(
  "lateral,quaternion,depth_range",
  [
    (0.0, (1, 0, 0, 0), (0.011, 0.0121)),
    (0.0006, (1, 0, 0, 0), (-0.0002, 0.0002)),
    (0.0, (0, 1, 0, 0), (0.0028, 0.0038)),
  ],
)
def test_physics_passes_aligned_shell_and_blocks_offset_or_reversed_shell(
  simulation, lateral, quaternion, depth_range
):
  """A bounded test-only wrench probes the bore; this is not a robot policy."""
  simulation.reset()
  place_tip(simulation, -0.001, lateral=lateral, quaternion=quaternion)
  body = simulation.model.body("usb_plug").id
  initial = simulation.data.xpos[body].copy()
  monitor = UsbInsertionMonitor(simulation)
  inverse = simulation.data.xquat[body].copy() * [1, -1, -1, -1]
  mouth = simulation.model.site("usb_socket_mouth").id
  frame = simulation.data.site_xmat[mouth].reshape(3, 3).copy()
  try:
    for _ in range(600):
      velocity = simulation.object_twist("usb_plug")
      local_velocity = frame.T @ velocity[:3]
      local_offset = frame.T @ (simulation.data.xpos[body] - initial)
      local_force = [
        np.clip(30 * (0.03 - local_velocity[0]), -1.0, 1.0),
        -300 * local_offset[1] - 2 * local_velocity[1],
        -300 * local_offset[2] - 2 * local_velocity[2],
      ]
      simulation.data.xfrc_applied[body, :3] = frame @ local_force + [
        0,
        0,
        0.018 * 9.81,
      ]
      error = np.zeros(4)
      mujoco.mju_mulQuat(error, simulation.data.xquat[body], inverse)
      if error[0] < 0:
        error *= -1
      simulation.data.xfrc_applied[body, 3:] = (
        -0.015 * error[1:] - 0.0008 * velocity[3:]
      )
      simulation.step()
      state = monitor.update()
    assert depth_range[0] <= state.insertion_depth_m <= depth_range[1]
    assert state.axial_resistance_n == pytest.approx(0.9, abs=0.03)
    assert state.socket_force_on_plug_n[0] < 0
    assert state.maximum_socket_penetration_m < 0.0001
    assert state.success is (lateral == 0 and quaternion == (1, 0, 0, 0))
    if state.success:
      assert state.backstop_contact
      assert state.backstop_axial_resistance_n > 0.0
      assert state.spring_contact_count > 0
      assert state.spring_normal_load_n > 0.0
      assert state.insertion_depth_m >= (
        config.BACKSTOP_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
      )
  finally:
    simulation.data.xfrc_applied.fill(0)


def test_success_timer_rejects_gaps_rewinds_motion_and_lost_seating(monkeypatch):
  monitor = synthetic_monitor(monkeypatch)
  sample = monitor.measure()
  assert sample.seated
  for index in range(51):
    monkeypatch.setattr(
      monitor, "measure", lambda i=index: replace(sample, timestamp=i * 0.002)
    )
    state = monitor.update()
  assert state.success
  # Losing seating cancels success immediately, not just at episode reset.
  monkeypatch.setattr(
    monitor, "measure", lambda: replace(sample, timestamp=0.102, backstop_contact=False)
  )
  assert not monitor.update().success
  for timestamp in (0.104, 0.5, 0.0):
    monkeypatch.setattr(
      monitor, "measure", lambda t=timestamp: replace(sample, timestamp=t)
    )
    assert not monitor.update().success
  monitor.reset()
  assert not monitor.update().success
  monitor.simulation._test_twist[:3] = [0.0, 0.0, -0.003]
  assert not UsbInsertionMonitor(monitor.simulation).measure().seated


@pytest.mark.parametrize("reverse", [False, True])
def test_socket_loads_are_classified_and_sign_corrected(monkeypatch, reverse):
  monitor = synthetic_monitor(
    monkeypatch,
    contacts=(
      ("backstop", 0.8, 0.0, reverse, -0.00001),
      ("spring", 0.6, 0.2, reverse, -0.00002),
      ("wall", 0.1, 0.03, reverse, -0.00003),
      ("tongue", 0.05, 0.01, reverse, -0.00004),
    ),
  )
  state = monitor.measure()
  assert state.socket_contact_count == 4
  assert state.backstop_contact_count == 1
  assert state.spring_contact_count == 1
  assert state.backstop_normal_load_n == pytest.approx(0.8)
  assert state.spring_normal_load_n == pytest.approx(0.6)
  assert state.wall_normal_load_n == pytest.approx(0.1)
  assert state.tongue_normal_load_n == pytest.approx(0.05)
  assert state.socket_normal_load_n == pytest.approx(1.55)
  np.testing.assert_allclose(state.backstop_force_on_plug_n, [-0.8, 0.0, 0.0])
  np.testing.assert_allclose(state.socket_force_on_plug_n, [-1.04, 0.7, 0.05])
  assert state.axial_resistance_n == pytest.approx(1.04)
  assert state.backstop_axial_resistance_n == pytest.approx(0.8)
  assert state.spring_axial_resistance_n == pytest.approx(0.2)
  assert state.maximum_socket_penetration_m == pytest.approx(0.00004)
  assert state.backstop_contact
  assert state.seated


@pytest.mark.parametrize(
  "depth,axial_speed,backstop_load,penetration,seated",
  [
    (0.0112, 0.0, 0.18, 0.00001, False),
    (0.01189, 0.0, 0.18, 0.00001, False),
    (0.01211, 0.0, 0.18, 0.00001, False),
    (0.012, 0.003, 0.18, 0.00001, False),
    (0.012, -0.003, 0.18, 0.00001, False),
    (0.012, 0.0006, 0.18, 0.00001, False),
    (0.012, 0.0, 0.0, 0.00001, False),
    (0.012, 0.0, 0.00001, 0.00001, False),
    (0.012, 0.0, 0.18, 0.00011, False),
    (0.01195, 0.0, 0.18, 0.00001, True),
    (0.01205, 0.0, 0.18, 0.00005, True),
    (0.012, 0.0004, 0.18, 0.00001, True),
  ],
)
def test_seating_requires_loaded_bottom_and_near_zero_speed(
  monkeypatch, depth, axial_speed, backstop_load, penetration, seated
):
  monitor = synthetic_monitor(
    monkeypatch,
    depth=depth,
    axial_speed=axial_speed,
    contacts=(("backstop", backstop_load, 0.0, False, -penetration),),
  )
  state = monitor.measure()
  assert state.seated is seated
  assert state.axial_speed_m_s == pytest.approx(axial_speed)
  assert not state.success


def test_spring_friction_cannot_substitute_for_bottom_contact(monkeypatch):
  state = synthetic_monitor(
    monkeypatch,
    contacts=(("spring", 1.0, 0.4, False, -0.00001),),
  ).measure()
  assert state.axial_resistance_n == pytest.approx(0.4)
  assert state.spring_axial_resistance_n == pytest.approx(0.4)
  assert state.backstop_contact_count == 0
  assert not state.backstop_contact
  assert not state.seated


def _confirmed_bottom_monitor(monkeypatch):
  monitor = synthetic_monitor(
    monkeypatch, contacts=(("backstop", 0.65, 0.0, False, -0.00001),)
  )
  for index in range(76):
    monitor.simulation.data.time = index * 0.002
    state = monitor.update()
  assert state.bottom_out_confirmed
  return monitor


def test_bottom_history_needs_real_elapsed_steps_and_measure_is_read_only(monkeypatch):
  monitor = synthetic_monitor(
    monkeypatch, contacts=(("backstop", 0.65, 0.0, False, -0.00001),)
  )
  for index in range(75):
    monitor.simulation.data.time = index * 0.002
    assert not monitor.update().bottom_out_confirmed
  for _ in range(100):
    assert not monitor.update().bottom_out_confirmed
    assert not monitor.measure().bottom_out_confirmed
  monitor.simulation.data.time = 0.15
  assert not monitor.measure().bottom_out_confirmed
  assert monitor.update().bottom_out_confirmed


def test_bottom_confirmation_is_retained_after_unloading(monkeypatch):
  monitor = _confirmed_bottom_monitor(monkeypatch)
  data = monitor.simulation.data
  data.ncon = 0
  data.time += 0.002
  state = monitor.update()
  assert not state.backstop_contact
  assert state.backstop_axial_resistance_n == 0
  assert state.bottom_out_confirmed
  assert state.seated
  assert state.success
  # A newly started monitor at the exact same unloaded pose has no history.
  fresh = UsbInsertionMonitor(monitor.simulation).update()
  assert not fresh.bottom_out_confirmed
  assert not fresh.seated
  assert not fresh.success


@pytest.mark.parametrize(
  "loss", ["reset", "rewind", "gap", "depth", "lateral", "penetration"]
)
def test_bottom_confirmation_clears_when_its_history_is_no_longer_valid(
  monkeypatch, loss
):
  monitor = _confirmed_bottom_monitor(monkeypatch)
  data = monitor.simulation.data
  data.time += 0.002
  data.ncon = 0
  if loss == "reset":
    monitor.reset()
  elif loss == "rewind":
    data.time = 0.05
  elif loss == "gap":
    data.time += 0.01
  elif loss == "depth":
    data.site_xpos[1, 2] = -0.0118
  elif loss == "lateral":
    data.site_xpos[1, 1] = 0.001
  else:
    data.ncon = 1
    data.contact[0].dist = -0.00011
  assert not monitor.update().bottom_out_confirmed
  # Returning to the original pose without a new actual bottom load must
  # not restore the cleared history.
  data.ncon = 0
  data.site_xpos[1] = [0.0, 0.0, -0.012]
  data.time += 0.002
  state = monitor.update()
  assert not state.bottom_out_confirmed
  assert not state.seated


def test_retained_bottom_history_does_not_excuse_motion(monkeypatch):
  monitor = _confirmed_bottom_monitor(monkeypatch)
  monitor.simulation.data.ncon = 0
  monitor.simulation._test_twist[:3] = [0.0, 0.0, -0.003]
  monitor.simulation.data.time += 0.002
  state = monitor.update()
  assert state.bottom_out_confirmed
  assert not state.seated
  assert not state.success


def test_monitor_synchronizes_site_state_after_physics_step(simulation):
  simulation.reset()
  place_tip(simulation, 0.0109)
  joint = simulation.model.joint("usb_plug_freejoint")
  mouth = simulation.model.site("usb_socket_mouth").id
  frame = simulation.data.site_xmat[mouth].reshape(3, 3).copy()
  simulation.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 3] = frame[:, 0] * 0.1
  simulation.step()
  state = UsbInsertionMonitor(simulation).measure()
  pose = simulation.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7]
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, pose[3:])
  tip = pose[:3] + rotation.reshape(3, 3) @ config.PLUG_TIP_LOCAL_M
  assert state.insertion_depth_m == pytest.approx(
    (frame.T @ (tip - config.SOCKET_MOUTH_POSITION_M))[0], abs=1e-12
  )
  assert state.insertion_depth_m > 0.011
  assert state.timestamp == simulation.data.time


@pytest.mark.parametrize(
  "script_name,arguments",
  [
    ("view_usb_insert.py", []),
    ("view_workcell.py", ["--scene", "usb-insert"]),
  ],
)
def test_usb_view_commands_reach_window_creation(script_name, arguments, monkeypatch):
  script = run_path(str(Path(__file__).parents[1] / "scripts/workcell" / script_name))
  monkeypatch.setattr(sys, "argv", [script_name, *arguments])

  class WindowReached(Exception):
    pass

  def launch(model, data, **kwargs):
    assert model.body("usb_plug").id >= 0
    assert model.nq == len(data.qpos)
    raise WindowReached

  monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
  with pytest.raises(WindowReached):
    script["main"]()
