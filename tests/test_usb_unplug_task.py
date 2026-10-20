from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.usb_unplug import config
from kaihand_tactile_env.tasks.usb_unplug.setup import (
  Simulation,
  initialize_installed_state,
)
from kaihand_tactile_env.tasks.usb_unplug.task import UsbUnplugMonitor


def test_installed_reset_sets_plug_and_springs_before_first_step():
  simulation = Simulation(add_genesis_probes=False)
  model, data = simulation.model, simulation.data
  plug_joint = model.joint("usb_plug_freejoint")
  plug_dof = int(plug_joint.dofadr[0])
  springs = tuple(
    model.joint(f"usb_socket_spring_{side}_slide") for side in ("top", "bottom")
  )
  assert data.time == 0.0
  np.testing.assert_allclose(
    [data.qpos[int(joint.qposadr[0])] for joint in springs],
    config.INITIAL_SPRING_DISPLACEMENT_M,
    rtol=0,
    atol=1e-15,
  )
  np.testing.assert_array_equal(data.qvel[plug_dof : plug_dof + 6], 0.0)
  assert simulation.initialization["socket_mechanics_changed"] is False
  assert simulation.initialization["object_assistance_applied"] is False

  monitor = UsbUnplugMonitor(simulation)
  initial = monitor.measure()
  assert initial.initially_seated_now
  assert initial.maximum_socket_penetration_m <= config.MAX_SOCKET_PENETRATION_M
  for _ in range(round(0.4 / simulation.timestep)):
    simulation.step()
    state = monitor.update()
  assert state.initially_seated_verified
  assert state.initially_seated_now
  assert state.maximum_socket_penetration_m <= config.MAX_SOCKET_PENETRATION_M
  assert state.backstop_normal_load_n > 0
  assert state.spring_normal_load_n > 0


def test_installed_initializer_rejects_started_episode_without_mutation():
  simulation = Simulation(add_genesis_probes=False)
  simulation.step()
  before = simulation.full_state()
  with pytest.raises(ValueError, match="only allowed immediately after reset"):
    initialize_installed_state(simulation)
  after = simulation.full_state()
  np.testing.assert_array_equal(after[0], before[0])
  np.testing.assert_array_equal(after[1], before[1])


def test_physics_observer_runs_after_each_individual_substep():
  simulation = Simulation(add_genesis_probes=False)
  samples = []

  def observe(current):
    samples.append((float(current.data.time), current.data.qpos.copy()))

  simulation.physics_observer = observe
  simulation.step(3)
  assert [sample[0] for sample in samples] == pytest.approx([0.002, 0.004, 0.006])
  assert all(np.isfinite(sample[1]).all() for sample in samples)


def test_monitor_preserves_legacy_ordered_milestones_without_using_them_for_success(monkeypatch):
  simulation = Simulation(add_genesis_probes=False)
  monitor = UsbUnplugMonitor(simulation)
  base = monitor.measure()
  sample = base
  timestamp = 0.0

  def advance(count, **changes):
    nonlocal sample, timestamp
    for _ in range(count):
      timestamp += simulation.timestep
      sample = replace(sample, timestamp=timestamp, **changes)
      monkeypatch.setattr(monitor, "measure", lambda value=sample: value)
      result = monitor.update()
    return result

  state = advance(51, initially_seated_now=True)
  assert state.initially_seated_verified and not state.grasp_verified
  state = advance(76, initially_seated_now=False, bilateral_grasp=True)
  assert state.grasp_verified and not state.fully_extracted
  state = advance(51, bilateral_grasp=True, fully_clear_now=True)
  assert state.fully_extracted and not state.placed_on_table
  state = advance(151, bilateral_grasp=True, fully_clear_now=False, at_home_now=True)
  assert state.placed_on_table and not state.released
  state = advance(
    51, bilateral_grasp=False, right_hand_contact_count=0, at_home_now=True
  )
  assert state.released and not state.hand_retreated
  state = advance(
    51,
    right_hand_contact_count=0,
    at_home_now=True,
    wrist_object_distance_m=config.RETREAT_CLEARANCE_M + 0.01,
  )
  assert state.hand_retreated and not state.success
  assert state.completion_protocol == config.COMPLETION_PROTOCOL


def test_signed_socket_force_reports_both_directions():
  simulation = Simulation(add_genesis_probes=False)
  state = UsbUnplugMonitor(simulation).measure()
  assert state.insertion_resistance_n == pytest.approx(
    max(0.0, -state.signed_axial_force_n)
  )
  assert state.withdrawal_resistance_n == pytest.approx(
    max(0.0, state.signed_axial_force_n)
  )
  assert min(state.insertion_resistance_n, state.withdrawal_resistance_n) == 0.0


@pytest.fixture(scope="module")
def sample_state():
  return UsbUnplugMonitor(Simulation(add_genesis_probes=False)).measure()


@pytest.fixture
def synthetic_hold(sample_state, monkeypatch):
  monitor = UsbUnplugMonitor.__new__(UsbUnplugMonitor)
  monitor.simulation = SimpleNamespace(timestep=0.002)
  monitor.reset()
  sample = replace(
    sample_state,
    initially_seated_now=False,
    bilateral_grasp=False,
    grip_force_n=(0.0, 0.0),
    insertion_depth_m=config.FULL_CLEAR_DEPTH_M,
    socket_contact_count=0,
    right_hand_contact_count=1,
    table_contact_count=0,
    fully_clear_now=True,
    at_home_now=False,
    # Motion is permitted; these are not hold completion thresholds.
    linear_speed_m_s=1.0,
    angular_speed_rad_s=10.0,
  )

  def observe(timestamp, **changes):
    nonlocal sample
    sample = replace(sample, timestamp=timestamp, **changes)
    monkeypatch.setattr(monitor, "measure", lambda: sample)
    return monitor.update()

  return monitor, observe


def test_extracted_hold_requires_two_seconds_and_no_initial_grasp_or_home_flags(synthetic_hold):
  monitor, observe = synthetic_hold
  assert config.EXTRACTION_HOLD_DWELL_S == 2.0
  assert config.COMPLETION_PROTOCOL == "usb-unplug-held-clear-2s-v2"
  state = observe(0.0)
  assert state.extraction_hold_now and state.extraction_hold_duration_s == 0.0
  for index in range(1, 1000):
    state = observe(index * 0.002)
  assert not state.success
  assert state.extraction_hold_duration_s == pytest.approx(1.998)
  state = observe(2.0)
  assert state.success and state.extraction_hold_verified
  assert state.extraction_hold_duration_s == pytest.approx(2.0)
  assert not state.initially_seated_verified
  assert not state.grasp_verified
  assert not state.fully_extracted
  assert not state.placed_on_table
  assert not state.released
  assert not state.hand_retreated
  assert monitor._last_extraction_hold_valid


def test_first_valid_endpoint_does_not_claim_preceding_invalid_interval(synthetic_hold):
  _, observe = synthetic_hold
  observe(0.0, right_hand_contact_count=0)
  state = observe(0.002, right_hand_contact_count=1)
  assert state.extraction_hold_now and state.extraction_hold_duration_s == 0.0
  state = observe(0.004)
  assert state.extraction_hold_duration_s == pytest.approx(0.002)


@pytest.mark.parametrize("interruption", [
  {"right_hand_contact_count": 0},
  {"table_contact_count": 1},
  {"socket_contact_count": 1},
  {"insertion_depth_m": -0.001999},
])
def test_contact_loss_table_support_socket_recontact_or_reinsertion_cancels_hold(
  synthetic_hold, interruption,
):
  _, observe = synthetic_hold
  for index in range(1001):
    state = observe(index * 0.002)
  assert state.success
  state = observe(2.002, **interruption)
  assert not state.extraction_hold_now
  assert not state.success and not state.extraction_hold_verified
  assert state.extraction_hold_duration_s == 0.0
  state = observe(
    2.004,
    right_hand_contact_count=1,
    table_contact_count=0,
    socket_contact_count=0,
    insertion_depth_m=config.FULL_CLEAR_DEPTH_M,
  )
  assert state.extraction_hold_now
  assert not state.success and state.extraction_hold_duration_s == 0.0
  for index in range(1, 1000):
    state = observe(2.004 + index * 0.002)
  assert not state.success
  state = observe(4.004)
  assert state.success and state.extraction_hold_duration_s == pytest.approx(2.0)


@pytest.mark.parametrize("new_timestamp", [2.2, 0.0, float("nan")])
def test_sampling_discontinuity_cannot_bridge_a_verified_hold(synthetic_hold, new_timestamp):
  _, observe = synthetic_hold
  for index in range(1001):
    state = observe(index * 0.002)
  assert state.success
  state = observe(new_timestamp)
  assert not state.success and not state.extraction_hold_verified
  assert state.extraction_hold_duration_s == 0.0


def test_duplicate_sample_does_not_accumulate_time_and_reset_clears_hold(synthetic_hold):
  monitor, observe = synthetic_hold
  observe(0.0)
  state = observe(0.002)
  for _ in range(2000):
    state = observe(0.002)
  assert state.extraction_hold_duration_s == pytest.approx(0.002)
  assert not state.success
  monitor.reset()
  state = observe(0.004)
  assert state.extraction_hold_duration_s == 0.0
  assert not state.success and not state.extraction_hold_verified


def test_policy_snapshot_exposes_hold_protocol_and_current_stage(synthetic_hold):
  from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS, TaskPolicyAdapter

  _, observe = synthetic_hold
  adapter = TaskPolicyAdapter.__new__(TaskPolicyAdapter)
  adapter.task = "usb-unplug"
  adapter._success = False
  adapter.state = observe(0.0)
  assert adapter.stage() == "hold_extracted_plug"
  snapshot = adapter.snapshot()
  assert snapshot["completion_protocol"] == config.COMPLETION_PROTOCOL
  assert snapshot["extraction_hold_now"]
  assert snapshot["extraction_hold_duration_s"] == 0.0
  assert not snapshot["extraction_hold_verified"]
  assert "two seconds" in TASK_INSTRUCTIONS["usb-unplug"]
  assert "Do not put it down or release it" in TASK_INSTRUCTIONS["usb-unplug"]
