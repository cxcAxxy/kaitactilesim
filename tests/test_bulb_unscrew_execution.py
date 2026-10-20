"""Full physical expert audit for installed bulb removal and bare tabletop return."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.tasks.bulb_unscrew import config
from kaihand_tactile_env.tasks.bulb_unscrew.execution import BulbUnscrewExecutor
from kaihand_tactile_env.tasks.bulb_unscrew.task import BulbUnscrewSimulation


@pytest.mark.parametrize("position_seed", [None, 0])
def test_five_finger_expert_completes_without_object_or_equality_shortcuts(
  monkeypatch, position_seed
):
  sim = BulbUnscrewSimulation(position_seed=position_seed, add_genesis_probes=False)
  executor = BulbUnscrewExecutor(sim, speed="fast")
  model, data = sim.model, sim.data
  non_arm_dofs = np.ones(model.nv, dtype=bool)
  for dofs in sim._arm_dofs.values():
    non_arm_dofs[dofs] = False
  capture = model.eq("bulb_thread_capture").id
  equality_data = model.eq_data.copy()
  last_active = data.eq_active.copy()
  last_qpos = data.qpos.copy()
  last_qvel = data.qvel.copy()
  actual_step = mujoco.mj_step
  release_steps: list[int] = []
  physics_observations: list[tuple[float, str]] = []
  step_count = 0

  def forbidden(*_args, **_kwargs):
    raise AssertionError("expert attempted a reset/object-state shortcut")

  monkeypatch.setattr(sim, "set_object_pose", forbidden)
  monkeypatch.setattr(sim, "restore_full_state", forbidden)
  monkeypatch.setattr(sim, "reset", forbidden)
  monkeypatch.setattr(sim, "initialize_installed", forbidden)

  def audited_step(observed_model, observed_data, *args, **kwargs):
    nonlocal step_count
    assert observed_model is model and observed_data is data
    # Only mj_step may integrate physical state between audit boundaries.
    np.testing.assert_array_equal(data.qpos, last_qpos)
    np.testing.assert_array_equal(data.qvel, last_qvel)
    assert not np.any(data.xfrc_applied)
    assert not np.any(data.qfrc_applied[non_arm_dofs])
    np.testing.assert_array_equal(model.eq_data, equality_data)

    changed = np.flatnonzero(data.eq_active != last_active)
    if changed.size:
      np.testing.assert_array_equal(changed, [capture])
      assert last_active[capture] and not data.eq_active[capture]
      assert data.qpos[sim._slide] < config.THREAD_RELEASE_DEPTH_OFFSET_M
      release_steps.append(step_count)
      np.copyto(last_active, data.eq_active)

    actual_step(model, data, *args, **kwargs)
    np.copyto(last_qpos, data.qpos)
    np.copyto(last_qvel, data.qvel)
    step_count += 1
    assert not np.any(data.xfrc_applied)
    assert not np.any(data.qfrc_applied[non_arm_dofs])
    np.testing.assert_array_equal(model.eq_data, equality_data)
    np.testing.assert_array_equal(data.eq_active, last_active)

  monkeypatch.setattr(mujoco, "mj_step", audited_step)
  sim.physics_observer = lambda observed: physics_observations.append(
    (float(observed.data.time), observed.task_phase)
  )
  result = executor.execute()

  assert result.success, result.reason
  assert result.reason == "completed"
  assert all(
    (
      result.initially_installed_verified,
      result.grasp_verified,
      result.thread_disengaged,
      result.fully_extracted,
      result.placed_on_table,
      result.released,
      result.hand_retreated,
    )
  )
  assert len(release_steps) == 1
  assert step_count == len(physics_observations)
  assert np.all(np.diff([sample[0] for sample in physics_observations]) > 0)
  assert not data.eq_active[capture]
  assert not sim.bulb_lit
  assert not np.any(data.xfrc_applied)
  assert not np.any(data.qfrc_applied[non_arm_dofs])

  assert result.strokes == 5
  assert result.state.maximum_turns_removed >= config.EXPECTED_RELEASE_TURNS
  assert result.state.maximum_counterclockwise_torque_nm > 0.0
  assert min(result.turn_contact_fraction) > 0.8
  assert result.simultaneous_turn_contact_fraction > 0.8
  assert result.maximum_wrist_rotation_deg < 0.5
  assert result.maximum_wrist_displacement_m < 0.002

  assert (
    config.PLACEMENT_MIN_SUPPORT_LOAD_N
    <= result.placement_peak_support_load_n
    <= config.MAX_PLACEMENT_SUPPORT_LOAD_N
  )
  assert result.placement_peak_right_hand_nonpad_contacts <= 1
  assert result.placement_release_drop_height_m <= 0.0001
  assert result.state.table_contact_load_n >= config.PLACEMENT_MIN_SUPPORT_LOAD_N
  assert result.state.robot_contact_count == 0
  assert result.state.table_xy_error_m <= config.PLACEMENT_XY_TOLERANCE_M
  assert result.state.table_z_error_m <= config.PLACEMENT_Z_TOLERANCE_M
  assert result.state.arm_home_error_rad <= config.ARM_HOME_TOLERANCE_RAD
  assert result.state.hand_home_error_rad <= config.HAND_HOME_TOLERANCE_RAD
  assert result.state.home_joint_speed_rad_s <= config.HOME_SPEED_RAD_S

  ordered = (
    "validate_installed",
    "verify_grasp",
    "breakaway",
    "unscrew_thread",
    "extract_clear",
    "transfer_to_table",
    "settle_on_table",
    "release_on_table",
    "verify_release",
    "return_home",
    "verify_complete",
  )
  indices = [result.phases.index(phase) for phase in ordered]
  assert indices == sorted(indices)
  observed_transitions: list[str] = []
  for _, phase in physics_observations:
    if not observed_transitions or observed_transitions[-1] != phase:
      observed_transitions.append(phase)
  assert tuple(observed_transitions) == result.phases
