import mujoco
import numpy as np
from kaihand_tactile_env.tasks.usb_unplug import config
from kaihand_tactile_env.tasks.usb_unplug.execution import UsbUnplugExecutor
from kaihand_tactile_env.tasks.usb_unplug.setup import Simulation


def test_real_expert_unplugs_and_holds_two_seconds_without_object_assistance(monkeypatch):
  simulation = Simulation(add_genesis_probes=False)
  model, data = simulation.model, simulation.data
  plug = model.body("usb_plug").id
  joint = model.joint("usb_plug_freejoint")
  plug_dof = int(joint.dofadr[0])
  equality = {
    name: getattr(model, name).copy()
    for name in ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data")
  }
  equality_active = data.eq_active.copy()
  body_mass = model.body_mass.copy()
  geom_friction = model.geom_friction.copy()
  last_qpos, last_qvel = data.qpos.copy(), data.qvel.copy()
  actual_step = mujoco.mj_step
  steps = 0

  def audited_step(observed_model, observed_data, *args, **kwargs):
    nonlocal steps
    assert observed_model is model and observed_data is data
    np.testing.assert_array_equal(data.qpos, last_qpos)
    np.testing.assert_array_equal(data.qvel, last_qvel)
    assert not np.any(data.xfrc_applied[plug])
    assert not np.any(data.qfrc_applied[plug_dof : plug_dof + 6])
    np.testing.assert_array_equal(data.eq_active, equality_active)
    actual_step(observed_model, observed_data, *args, **kwargs)
    np.copyto(last_qpos, data.qpos)
    np.copyto(last_qvel, data.qvel)
    steps += 1

  monkeypatch.setattr(mujoco, "mj_step", audited_step)
  result = UsbUnplugExecutor(simulation).run()
  assert result.success, result.failure_reason
  assert result.initially_seated_verified
  assert result.grasp_verified
  assert result.fully_extracted
  assert not result.placed_on_table
  assert not result.released
  assert not result.hand_retreated
  assert result.completion_protocol == config.COMPLETION_PROTOCOL
  assert result.extraction_hold_now and result.extraction_hold_verified
  assert result.extraction_hold_duration_s >= config.EXTRACTION_HOLD_DWELL_S - 1e-12
  assert result.final_state.completion_protocol == config.COMPLETION_PROTOCOL
  assert result.mechanics_version == config.MECHANICS_VERSION
  assert result.peak_withdrawal_resistance_n <= config.MAX_WITHDRAWAL_RESISTANCE_N
  assert result.maximum_socket_penetration_m <= config.MAX_SOCKET_PENETRATION_M
  assert result.final_state.insertion_depth_m <= config.FULL_CLEAR_DEPTH_M
  assert result.final_state.socket_contact_count == 0
  assert result.final_state.right_hand_contact_count > 0
  assert result.final_state.table_contact_count == 0
  assert result.phases[-1] == "hold_extracted"
  assert not {"place", "release", "retreat", "return_home"}.intersection(result.phases)
  assert steps > 1000
  np.testing.assert_array_equal(model.body_mass, body_mass)
  np.testing.assert_array_equal(model.geom_friction, geom_friction)
  for name, expected in equality.items():
    np.testing.assert_array_equal(getattr(model, name), expected)
  np.testing.assert_array_equal(data.eq_active, equality_active)


def test_immediate_cancellation_returns_canonical_unsuccessful_result():
  simulation = Simulation(add_genesis_probes=False)
  before = simulation.full_state()
  result = UsbUnplugExecutor(simulation, should_stop=lambda: True).run()
  assert not result.success
  assert result.failure_reason == "cancelled"
  assert not result.grasp_verified
  assert not result.fully_extracted
  assert not result.placed_on_table
  assert not result.released
  assert not result.hand_retreated
  assert result.completion_protocol == config.COMPLETION_PROTOCOL
  assert not result.extraction_hold_now and not result.extraction_hold_verified
  assert result.extraction_hold_duration_s == 0.0
  after = simulation.full_state()
  np.testing.assert_array_equal(after[0], before[0])
  np.testing.assert_array_equal(after[1], before[1])
