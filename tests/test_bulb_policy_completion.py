from dataclasses import asdict, replace

import numpy as np
import pytest
from kaihand_tactile_env.shared.policy_tasks import TaskPolicyAdapter
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import (
  BulbScrewMonitor,
  BulbScrewSimulation,
)
from kaihand_tactile_env.tasks.bulb_screw_inplace.task import BulbScrewInplaceState
from kaihand_tactile_env.tasks.bulb_unscrew.task import (
  BulbUnscrewMonitor,
  BulbUnscrewSimulation,
)


@pytest.fixture(scope="module")
def simulation():
  return BulbScrewSimulation()


def _advance_monitor(simulation, monitor, duration_s):
  count = round(duration_s / simulation.timestep) + 2
  state = None
  for _ in range(count):
    simulation.data.time += simulation.timestep
    state = monitor.update()
  return state


def test_strict_completion_is_default_and_seated_only_is_explicit(
  simulation, monkeypatch
):
  simulation.reset()
  measured = BulbScrewMonitor(simulation).measure()
  seated = replace(
    measured,
    seated=True,
    bulb_lit=False,
    tightening_verified=False,
    released=True,
    fingertip_normal_force_n=(0.0,) * 5,
  )

  seated_only = BulbScrewMonitor(simulation, auto_confirm_tightening=False)
  monkeypatch.setattr(
    seated_only,
    "measure",
    lambda: replace(seated, timestamp=float(simulation.data.time)),
  )
  assert _advance_monitor(simulation, seated_only, config.SEATED_DWELL_S).success

  simulation.data.time = 0.0
  policy = BulbScrewMonitor(simulation)
  monkeypatch.setattr(
    policy,
    "measure",
    lambda: replace(seated, timestamp=float(simulation.data.time)),
  )
  state = _advance_monitor(simulation, policy, 2 * config.SEATED_DWELL_S)
  assert state.seated and state.seated_duration_s >= config.SEATED_DWELL_S
  assert not state.tightening_verified
  assert not state.success


def test_policy_tightening_uses_loaded_stall_thresholds(simulation, monkeypatch):
  simulation.reset()
  monitor = BulbScrewMonitor(simulation, auto_confirm_tightening=True)
  base = replace(
    monitor.measure(),
    seated=True,
    bulb_lit=False,
    tightening_verified=False,
    released=False,
    hand_clockwise_torque_nm=config.TIGHTENING_MIN_TORQUE_NM + 0.01,
  )
  confirmations = []
  monkeypatch.setattr(
    simulation,
    "confirm_tightening",
    lambda: confirmations.append(float(simulation.data.time)),
  )
  steps = round(config.TIGHTENING_HOLD_S / simulation.timestep) + 1

  state = base
  for tick in range(steps):
    simulation.data.time = tick * simulation.timestep
    loads = (4.1,) * 5 if tick != steps // 2 else (4.1, 3.9, 4.1, 4.1, 4.1)
    state = monitor._update_tightening(
      replace(
        base, timestamp=float(simulation.data.time), fingertip_normal_force_n=loads
      )
    )
  assert not state.tightening_verified
  assert not confirmations

  monitor.reset()
  for tick in range(steps + 1):
    simulation.data.time = tick * simulation.timestep
    state = monitor._update_tightening(
      replace(
        base,
        timestamp=float(simulation.data.time),
        fingertip_normal_force_n=(4.1,) * 5,
      )
    )
    if state.tightening_verified:
      break
  assert state.tightening_verified and state.bulb_lit
  assert state.tightening_stall_duration_s >= config.TIGHTENING_HOLD_S - 1e-9
  assert confirmations == pytest.approx([simulation.data.time])


def test_policy_completion_requires_release_dwell_and_resets_on_time_gap(
  simulation, monkeypatch
):
  simulation.reset()
  monitor = BulbScrewMonitor(simulation, auto_confirm_tightening=True)
  current = {
    "state": replace(
      monitor.measure(),
      seated=True,
      bulb_lit=True,
      tightening_verified=True,
      released=False,
      fingertip_normal_force_n=(5.0,) * 5,
    )
  }
  monkeypatch.setattr(
    monitor,
    "measure",
    lambda: replace(current["state"], timestamp=float(simulation.data.time)),
  )

  state = _advance_monitor(simulation, monitor, 2 * config.SEATED_DWELL_S)
  assert state.tightening_verified and not state.released and not state.success

  current["state"] = replace(
    current["state"], released=True, fingertip_normal_force_n=(0.0,) * 5
  )
  state = _advance_monitor(simulation, monitor, config.SEATED_DWELL_S)
  assert state.release_stable_duration_s >= config.SEATED_DWELL_S
  assert state.success

  simulation.data.time += 2 * simulation.timestep
  state = monitor.update()
  assert state.release_stable_duration_s == 0.0
  assert not state.success


def test_release_rejects_non_fingertip_right_hand_contact(simulation, monkeypatch):
  simulation.reset()
  monitor = BulbScrewMonitor(simulation, auto_confirm_tightening=True)
  zero_fingertips = np.zeros(5)
  monkeypatch.setattr(
    simulation,
    "hand_object_contact_metrics",
    lambda: (zero_fingertips, 0.0, 0.02, 1, 1, 0.02, 1),
  )

  state = monitor.measure()
  assert state.fingertip_normal_force_n == (0.0,) * 5
  assert state.right_hand_contact_load_n == pytest.approx(0.02)
  assert state.right_hand_contact_count == 1
  assert state.right_hand_nonpad_contact_count == 1
  assert state.robot_contact_count == 1
  assert not state.released

  monkeypatch.setattr(
    simulation,
    "hand_object_contact_metrics",
    lambda: (zero_fingertips, 0.0, 0.0, 0, 0, 0.02, 1),
  )
  assert not monitor.measure().released

  monkeypatch.setattr(
    simulation,
    "hand_object_contact_metrics",
    lambda: (zero_fingertips, 0.0, 0.0, 0, 0, 0.0, 0),
  )
  assert monitor.measure().released


def test_tightening_rejects_nonpad_or_other_robot_support(simulation, monkeypatch):
  simulation.reset()
  monitor = BulbScrewMonitor(simulation)
  base = replace(
    monitor.measure(),
    seated=True,
    hand_clockwise_torque_nm=config.TIGHTENING_MIN_TORQUE_NM + 0.01,
    fingertip_normal_force_n=(4.1,) * 5,
    right_hand_contact_count=5,
    right_hand_nonpad_contact_count=1,
    robot_contact_count=6,
  )
  confirmations = []
  monkeypatch.setattr(simulation, "confirm_tightening", lambda: confirmations.append(1))
  for tick in range(round(config.TIGHTENING_HOLD_S / simulation.timestep) + 2):
    state = monitor._update_tightening(
      replace(base, timestamp=tick * simulation.timestep)
    )
  assert not state.tightening_verified
  assert not confirmations


def test_policy_guard_rejects_excessive_bulb_fingertip_load(simulation):
  simulation.reset()
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = "bulb-screw"
  adapter.simulation = simulation
  adapter.state = replace(
    BulbScrewMonitor(simulation).measure(),
    fingertip_normal_force_n=(0.0, 0.0, 0.0, 0.0, config.MAX_FINGERTIP_LOAD_N + 0.1),
  )

  with pytest.raises(RuntimeError, match="fingertip normal force exceeded"):
    adapter._guard()


def test_bulb_policy_snapshot_and_stages_expose_contact_and_capture_diagnostics(
  simulation,
):
  simulation.reset()
  state = BulbScrewMonitor(simulation).measure()
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = "bulb-screw"
  adapter._success = False
  adapter.state = state
  adapter.initial_object_pose = simulation.object_pose("bulb").copy()

  snapshot = adapter.snapshot()
  assert snapshot["clockwise_angle_deg"] == pytest.approx(state.clockwise_turns * 360)
  assert (
    set(
      (
        "bulb_pose_wxyz",
        "fingertip_normal_force_n",
        "right_hand_contact_load_n",
        "right_hand_contact_count",
        "right_hand_nonpad_contact_count",
        "robot_contact_load_n",
        "robot_contact_count",
        "five_finger_contact_duration_s",
        "five_finger_contact_fraction",
        "capture_depth_margin_m",
        "capture_lateral_margin_m",
        "capture_tilt_margin_rad",
        "capture_speed_margin_m_s",
        "capture_angular_speed_margin_rad_s",
        "capture_contact_margin_n",
      )
    )
    <= snapshot.keys()
  )
  assert adapter.stage() == "pick_bulb"

  adapter.state = replace(state, seated=True)
  assert adapter.stage() == "tighten"
  adapter.state = replace(
    state,
    seated=True,
    bulb_lit=True,
    tightening_verified=True,
    released=False,
  )
  assert adapter.stage() == "release_bulb"
  adapter.state = replace(adapter.state, released=True)
  assert adapter.stage() == "verify_released"


def test_inplace_snapshot_records_both_thread_and_initial_relative_angles(simulation):
  simulation.reset()
  values = asdict(BulbScrewMonitor(simulation).measure())
  values["clockwise_turns"] = 100 / 360
  state = BulbScrewInplaceState(**values, rotation_from_initial_turns=99.5 / 360)
  before = asdict(state)
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = "bulb-screw-inplace"
  adapter._success = False
  adapter.state = state

  snapshot = adapter.snapshot()
  assert snapshot["clockwise_angle_deg"] == pytest.approx(100)
  assert snapshot["rotation_from_initial_deg"] == pytest.approx(99.5)
  assert snapshot["rotation_from_initial_turns"] == state.rotation_from_initial_turns
  assert asdict(state) == before
  assert snapshot["success"] is False


def test_unscrew_snapshot_records_removal_angle_without_changing_native_turns():
  simulation = BulbUnscrewSimulation()
  state = replace(BulbUnscrewMonitor(simulation).measure(), turns_removed=131.5 / 360)
  before = asdict(state)
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = "bulb-unscrew"
  adapter._success = False
  adapter.state = state

  snapshot = adapter.snapshot()
  assert snapshot["angle_removed_deg"] == pytest.approx(131.5)
  assert snapshot["turns_removed"] == state.turns_removed
  assert asdict(state) == before
  assert snapshot["success"] is False
