"""Array-only checks: no MuJoCo model, renderer, or dataset capture is started."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared import simulation as shared
from kaihand_tactile_env.shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from kaihand_tactile_env.tasks.poker_draw import precontact_noise as noise
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (
  ForceLimitedPokerSimulation,
)


class _Provider:
  source = "solver_contact_proxy_v1"

  def __init__(self, model, *, link_names):
    assert tuple(link_names) == RIGHT_FINGERTIP_LINK_NAMES

  def read(self, data):
    return SimpleNamespace(
      contact=data.fingertip_contact.copy(), normal_force=data.fingertip_force.copy()
    )


def _simulation(monkeypatch, *, cls=noise.PrecontactPokerSimulation, signals=None):
  sim = object.__new__(cls)
  sim.scene = "poker-draw"
  sim.timestep = 0.002
  sim.arm_speed_limit = 0.8
  sim.hand_position_gain = 1.0
  sim.drive_limit_n = None
  sim._arm_actuators = {"left": np.arange(7), "right": np.arange(7, 14)}
  sim._arm_joint_ids = {side: ids.copy() for side, ids in sim._arm_actuators.items()}
  sim._arm_dofs = {side: ids.copy() for side, ids in sim._arm_actuators.items()}
  sim._arm_command = {"left": np.full(7, -0.2), "right": np.full(7, 0.2)}
  sim._arm_goal = {"left": np.full(7, -0.6), "right": np.full(7, 0.6)}
  sim._hand_actuators = {"left": {"hand_l_test": 14}, "right": {"hand_r_test": 15}}
  sim._hand_targets = {"left": {"hand_l_test": 0.2}, "right": {"hand_r_test": 0.3}}
  sim._qpos_address = {"hand_l_test": 14, "hand_r_test": 15}
  names = [f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)]
  sim.model = SimpleNamespace(
    actuator_ctrlrange=np.tile([-1.0, 1.0], (16, 1)),
    jnt_range=np.tile([-1.0, 1.0], (14, 1)),
    actuator=lambda index: SimpleNamespace(name=names[index]),
    fake_sim=sim,
  )
  sim.data = SimpleNamespace(
    time=0.0,
    ctrl=np.zeros(16),
    qpos=np.linspace(-0.1, 0.1, 16),
    qvel=np.linspace(0.0, 0.2, 16),
    qfrc_bias=np.linspace(0.0, 0.3, 16),
    qfrc_applied=np.zeros(16),
    fingertip_contact=np.zeros(5, dtype=bool),
    fingertip_force=np.zeros(5),
    physics_steps=0,
    applied=[],
  )
  sim._fake_signals = signals or {}

  def physics(model, data):
    data.applied.append(data.ctrl.copy())
    data.physics_steps += 1
    data.time += model.fake_sim.timestep
    data.fingertip_contact[:] = False
    data.fingertip_force[:] = 0.0
    finger = model.fake_sim._fake_signals.get(data.physics_steps)
    if finger is not None:
      data.fingertip_contact[finger] = True
      # Any contact signal, including zero force, must stop random input.

  monkeypatch.setattr(shared.mujoco, "mj_step", physics)
  monkeypatch.setattr(noise, "SolverContactTactileProvider", _Provider)
  return sim


def test_settings_bounds_are_pure_and_preset_is_separate():
  assert noise.PRECONTACT_PRESET == "middle-force-precontact-v1"
  assert noise.DEFAULT_XY_JITTER_M == 0.004
  assert noise.DEFAULT_YAW_JITTER_RAD == pytest.approx(np.deg2rad(0.5))
  assert noise.precontact_noise_settings()["std_rad"] == pytest.approx(np.deg2rad(0.03))
  assert noise.precontact_noise_settings(0)["maximum_offset_rad"] == 0
  for value in (-0.1, np.nan, np.inf, np.deg2rad(0.0501), True, None, [0.0], "0"):
    with pytest.raises(ValueError):
      noise.precontact_noise_settings(value)


def test_saturated_motion_changes_actual_controls_only_on_right_arm(monkeypatch):
  noisy = _simulation(monkeypatch)
  baseline = _simulation(monkeypatch, cls=shared.ArmHandSimulation)
  qpos = noisy.data.qpos.copy()
  qvel = noisy.data.qvel.copy()
  goals = {side: value.copy() for side, value in noisy._arm_goal.items()}
  noisy.configure_precontact_noise(seed=17)
  noisy.step(80)
  baseline.step(80)
  trace = noisy.precontact_noise_trace()
  assert np.any(np.abs(trace["actual_ctrl_rad"] - trace["nominal_ctrl_rad"]) > 1e-8)
  np.testing.assert_allclose(
    np.diff(trace["nominal_ctrl_rad"], axis=0), noisy.arm_speed_limit * noisy.timestep
  )
  controls = np.asarray(noisy.data.applied)
  clean_controls = np.asarray(baseline.data.applied)
  np.testing.assert_array_equal(controls[:, :7], clean_controls[:, :7])
  np.testing.assert_array_equal(controls[:, 14:], clean_controls[:, 14:])
  np.testing.assert_array_equal(noisy.data.qpos, qpos)
  np.testing.assert_array_equal(noisy.data.qvel, qvel)
  for side in goals:
    np.testing.assert_array_equal(noisy._arm_goal[side], goals[side])
  # Initial ctrl is zero after reset; first executed command starts at home.
  assert np.all(trace["actual_ctrl_rad"][0] > 0.19)


def test_noise_and_actual_controls_respect_amplitude_rate_and_limits(monkeypatch):
  sim = _simulation(monkeypatch)
  sim.configure_precontact_noise(seed=271)
  sim.step(160)
  trace = sim.precontact_noise_trace()
  settings = noise.precontact_noise_settings()
  offsets = trace["noise_offset_rad"]
  assert np.max(np.abs(offsets)) <= settings["maximum_offset_rad"] + 1e-12
  assert np.max(np.abs(np.diff(np.vstack((np.zeros(7), offsets)), axis=0))) <= (
    settings["maximum_offset_rate_rad_s"] * sim.timestep + 1e-12
  )
  actual = trace["actual_ctrl_rad"]
  initial = sim.precontact_noise_metadata()["initial_arm_command_rad"]
  assert np.max(np.abs(np.diff(np.vstack((initial, actual)), axis=0))) <= (
    sim.arm_speed_limit * sim.timestep + 1e-12
  )
  assert np.all(actual >= -1.0) and np.all(actual <= 1.0)
  np.testing.assert_allclose(actual, trace["nominal_ctrl_rad"] + offsets, atol=1e-15)


def test_joint_and_actuator_limits_both_bound_the_perturbation(monkeypatch):
  sim = _simulation(monkeypatch)
  sim._arm_command["right"][:] = 0.0
  sim._arm_goal["right"][:] = 0.0
  sim.model.jnt_range[7:14, 0] = -0.00004
  sim.model.actuator_ctrlrange[7:14, 1] = 0.00004
  sim.configure_precontact_noise(seed=173)
  sim.step(120)
  actual = sim.precontact_noise_trace()["actual_ctrl_rad"]
  assert np.all(actual >= -0.00004 - 1e-15)
  assert np.all(actual <= 0.00004 + 1e-15)


def test_one_transient_thumb_signal_latches_before_next_control_forever(monkeypatch):
  sim = _simulation(monkeypatch, signals={12: 0})
  sim.configure_precontact_noise(seed=13)
  original_goal = sim._arm_goal["right"].copy()
  sim.step(40)
  trace = sim.precontact_noise_trace()
  metadata = sim.precontact_noise_metadata()
  assert trace["contact"][11, 0]
  assert not np.any(trace["contact"][12:])
  assert not np.any(trace["latched_before_step"][:12])
  assert np.all(trace["latched_before_step"][12:])
  np.testing.assert_array_equal(trace["noise_offset_rad"][12:], np.zeros((28, 7)))
  assert metadata["random_sample_count"] == 12
  assert metadata["postcontact_random_sample_count"] == 0
  assert metadata["contact_latched"] and metadata["postcontact_noise_disabled"]
  assert metadata["contact_detected_time_s"] == pytest.approx(12 * sim.timestep)
  assert metadata["contact_tactile_time_s"] == pytest.approx(11 * sim.timestep)
  assert metadata["command_handoff_count"] == 1
  np.testing.assert_array_equal(sim._arm_goal["right"], original_goal)
  assert np.max(np.abs(np.diff(trace["actual_ctrl_rad"], axis=0))) <= (
    sim.arm_speed_limit * sim.timestep + 1e-12
  )
  assert not np.array_equal(
    trace["nominal_ctrl_rad"][12], trace["nominal_ctrl_rad"][11]
  )
  np.testing.assert_allclose(trace["time_s"], np.arange(40) * sim.timestep)
  np.testing.assert_allclose(trace["tactile_time_s"], trace["time_s"], atol=1e-14)


def test_initial_contact_stops_all_sampling_without_rebasing_to_reset_ctrl(monkeypatch):
  sim = _simulation(monkeypatch)
  sim.data.fingertip_contact[4] = True
  sim.configure_precontact_noise(seed=10)
  sim.step(5)
  assert sim.precontact_noise_metadata()["random_sample_count"] == 0
  assert sim.precontact_noise_metadata()["contact_detected_time_s"] == 0.0
  assert sim.precontact_noise_metadata()["command_handoff_count"] == 0
  trace = sim.precontact_noise_trace()
  assert np.all(trace["latched_before_step"])
  assert np.all(trace["actual_ctrl_rad"][0] > 0.19)


def test_seed_reproduction_batching_and_zero_noise_match_original(monkeypatch):
  batched = _simulation(monkeypatch, signals={16: 2})
  individual = _simulation(monkeypatch, signals={16: 2})
  other = _simulation(monkeypatch, signals={16: 2})
  for sim, seed in ((batched, 20), (individual, 20), (other, 21)):
    sim.configure_precontact_noise(seed=seed)
  batched.step(25)
  other.step(25)
  for _ in range(25):
    individual.step()
  for field, expected in batched.precontact_noise_trace().items():
    np.testing.assert_array_equal(individual.precontact_noise_trace()[field], expected)
  assert not np.array_equal(
    batched.precontact_noise_trace()["noise_offset_rad"],
    other.precontact_noise_trace()["noise_offset_rad"],
  )
  paired = _simulation(monkeypatch, signals={3: 1})
  baseline = _simulation(monkeypatch, cls=shared.ArmHandSimulation, signals={3: 1})
  paired.configure_precontact_noise(seed=44, std_rad=0.0)
  paired.step(9)
  baseline.step(9)
  np.testing.assert_array_equal(paired.data.applied, baseline.data.applied)
  assert paired.precontact_noise_metadata()["random_sample_count"] == 0
  assert paired.precontact_noise_metadata()["contact_latched"]


def test_configuration_cannot_rearm_during_episode_and_reset_clears_state(monkeypatch):
  sim = _simulation(monkeypatch, signals={1: 1})
  for seed in (-1, True, 1.0, "1"):
    with pytest.raises(ValueError):
      sim.configure_precontact_noise(seed=seed)
  sim.configure_precontact_noise(seed=1)
  sim.step()
  with pytest.raises(RuntimeError, match="reset time"):
    sim.configure_precontact_noise(seed=1)

  def reset(_sim, *args, **kwargs):
    _sim.data.time = 0.0
    _sim.drive_limit_n = None

  monkeypatch.setattr(noise.MidForcePokerSimulation, "reset", reset)
  sim.reset()
  assert sim.precontact_noise_metadata() == {
    "schema_version": noise.PRECONTACT_SCHEMA,
    "configured": False,
  }
  assert sim.precontact_noise_trace()["contact"].shape == (0, 5)


def test_cartesian_budget_is_rejected_until_contact_even_with_zero_noise(monkeypatch):
  sim = _simulation(monkeypatch, signals={1: 1})
  sim.configure_precontact_noise(seed=11, std_rad=0)
  with pytest.raises(RuntimeError, match="contact latch"):
    sim.begin_cartesian_drive(4.0)
  sim.step()
  calls = []

  def begin(_sim, limit):
    calls.append(limit)
    _sim.drive_limit_n = limit
    return np.zeros(3)

  monkeypatch.setattr(noise.MidForcePokerSimulation, "begin_cartesian_drive", begin)
  sim.begin_cartesian_drive(4.0)
  assert calls == [4.0]

  def force_step(_sim, count):
    assert count == 1
    _sim.data.ctrl[_sim._arm_actuators["right"]] = 0.9
    _sim.data.time += _sim.timestep

  monkeypatch.setattr(ForceLimitedPokerSimulation, "step", force_step)
  sim.step(3)
  trace = sim.precontact_noise_trace()
  np.testing.assert_array_equal(trace["control_mode"], [0, 1, 1, 1])
  np.testing.assert_array_equal(trace["actual_ctrl_rad"][1:], np.full((3, 7), 0.9))
  np.testing.assert_array_equal(
    trace["nominal_ctrl_rad"][1:], trace["actual_ctrl_rad"][1:]
  )
  assert not np.any(trace["noise_offset_rad"])


def test_physics_exception_does_not_change_task_goals_or_retry_random_input(
  monkeypatch,
):
  sim = _simulation(monkeypatch)
  sim.configure_precontact_noise(seed=23)
  goal = sim._arm_goal["right"].copy()
  physical = (sim.data.qpos.copy(), sim.data.qvel.copy())

  def fail(model, data):
    raise RuntimeError("fake physics failure")

  monkeypatch.setattr(shared.mujoco, "mj_step", fail)
  with pytest.raises(RuntimeError, match="fake physics failure"):
    sim.step()
  np.testing.assert_array_equal(sim._arm_goal["right"], goal)
  np.testing.assert_array_equal(sim.data.qpos, physical[0])
  np.testing.assert_array_equal(sim.data.qvel, physical[1])
  assert sim.precontact_noise_metadata()["failed"]
  assert sim.precontact_noise_metadata()["physics_step_count"] == 0
  with pytest.raises(RuntimeError, match="reset is required"):
    sim.step()
  assert sim.precontact_noise_metadata()["random_sample_count"] == 1


def test_invalid_tactile_data_requires_reset_before_more_motion(monkeypatch):
  sim = _simulation(monkeypatch)
  sim.configure_precontact_noise(seed=12)
  sim.data.fingertip_force[0] = np.nan
  with pytest.raises(RuntimeError, match="invalid right fingertip"):
    sim.step()
  sim.data.fingertip_force[:] = 0
  with pytest.raises(RuntimeError, match="reset is required"):
    sim.step()
  assert sim.data.physics_steps == 0
  assert sim.precontact_noise_metadata()["random_sample_count"] == 0


def test_metadata_and_trace_return_copies(monkeypatch):
  sim = _simulation(monkeypatch)
  sim.configure_precontact_noise(seed=31)
  sim.step(2)
  metadata = sim.precontact_noise_metadata()
  metadata["settings"]["contact_link_names"].clear()
  metadata["initial_arm_command_rad"][0] = 9.0
  trace = sim.precontact_noise_trace()
  trace["actual_ctrl_rad"][:] = 9.0
  assert sim.precontact_noise_metadata()["settings"]["contact_link_names"]
  assert sim.precontact_noise_metadata()["initial_arm_command_rad"][0] == 0.2
  assert np.max(sim.precontact_noise_trace()["actual_ctrl_rad"]) < 1.0


def test_precontact_factory_only_selects_subclass(monkeypatch):
  events = []
  sentinel = object()

  @contextmanager
  def factory(path, *, simulation_type):
    events.append((path, simulation_type))
    yield sentinel, {"accepted_contact_model": True}

  monkeypatch.setattr(noise, "middle_force_simulation", factory)
  with noise.precontact_force_simulation("model.xml") as (sim, metadata):
    assert sim is sentinel
    assert metadata == {"accepted_contact_model": True}
  assert events == [("model.xml", noise.PrecontactPokerSimulation)]
