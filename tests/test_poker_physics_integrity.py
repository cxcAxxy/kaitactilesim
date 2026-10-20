"""The light card's free-flight damping must not amplify into an auto-reset."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.physics_integrity import (
  PhysicsIntegrityError,
  check_physics_clock,
  check_physics_state,
)
from kaihand_tactile_env.shared.recording import EpisodeRecorder
from kaihand_tactile_env.tasks.poker_draw.mid_full import (
  MidForcePokerSimulation,
  configure_poker_physics_integrity,
)
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (
  ForceLimitedPokerSimulation,
)


def _free_card():
  # Principal-axis permutation places the real card's minimum inertia on Y.
  # Same mass, minimum inertia, free-joint damping and clock as poker scene.
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody><body name="card">
        <joint name="card_freejoint" type="free" damping=".002"/>
        <inertial pos="0 0 0" mass=".00567872"
          diaginertia="5.45952752e-6 1.844595307904e-6 3.617036641237333e-6"/>
        <geom type="box" size=".028 .044 .0004" contype="0" conaffinity="0"/>
      </body></worldbody>
      <actuator><general joint="card_freejoint" gear="1 0 0 0 0 0"/></actuator>
    </mujoco>""")
  data = mujoco.MjData(model)
  data.qvel[4] = 1.0
  return SimpleNamespace(model=model, data=data, timestep=float(model.opt.timestep))


def test_reproduce_original_free_card_angular_damping_instability():
  if mujoco.__version__ != "3.8.1":
    pytest.skip("version-specific implicitfast midpoint regression")
  sim = _free_card()
  ratio = 1.0 - sim.timestep * sim.model.dof_damping[4] / sim.model.body_inertia[1, 1]
  for _ in range(10):
    mujoco.mj_step(sim.model, sim.data)
  assert abs(ratio) > 1
  assert sim.data.qvel[4] == pytest.approx(ratio**10)
  assert abs(sim.data.qvel[4]) > 4


def test_poker_override_changes_only_integrator_and_keeps_free_flight_stable():
  sim = _free_card()
  before = {
    name: getattr(sim.model, name).copy()
    for name in (
      "body_mass",
      "body_inertia",
      "dof_damping",
      "geom_friction",
      "actuator_gainprm",
    )
  }
  initial_qpos = sim.data.qpos.copy()
  metadata = configure_poker_physics_integrity(sim)
  assert metadata["physics_integrator_original"] == "mjINT_IMPLICITFAST"
  assert metadata["physics_integrator_used"] == "mjINT_IMPLICIT"
  assert sim.model.opt.timestep == sim.timestep == 0.002
  for name, values in before.items():
    np.testing.assert_array_equal(getattr(sim.model, name), values)
  np.testing.assert_array_equal(sim.data.qpos, initial_qpos)
  assert sim.data.qvel[4] == 1
  previous_speed = 1.0
  for _ in range(25000):  # full 50-second flight, no solver constraints
    before_time = float(sim.data.time)
    mujoco.mj_step(sim.model, sim.data)
    check_physics_state(sim.data, stage="test")
    check_physics_clock(before_time, float(sim.data.time), timestep=sim.timestep)
    assert abs(sim.data.qvel[4]) <= previous_speed + 1e-14
    previous_speed = abs(sim.data.qvel[4])
  assert sim.data.time == pytest.approx(50.0)
  assert previous_speed < 1e-12


def test_real_autoreset_is_detected_even_when_state_is_finite(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)  # keep MuJoCo warning log away from the worktree
  sim = _free_card()
  sim.data.time = 34.0
  sim.data.qpos[0] = 0.4
  sim.data.qacc[4] = 1e12
  mujoco.mj_checkAcc(sim.model, sim.data)
  assert sim.data.time == 0
  assert np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
  with pytest.raises(PhysicsIntegrityError) as raised:
    check_physics_state(sim.data, stage="after_step")
  assert raised.value.details["warning"] == "mjWARN_BADQACC"
  with pytest.raises(PhysicsIntegrityError, match="backwards"):
    check_physics_clock(34.0, float(sim.data.time), timestep=0.002)


@pytest.mark.parametrize("field", ["qpos", "qvel", "qacc", "ctrl"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e11])
def test_invalid_arrays_fail_closed(field, value):
  sim = _free_card()
  getattr(sim.data, field)[0] = value
  with pytest.raises(PhysicsIntegrityError) as raised:
    check_physics_state(sim.data, stage="before_step")
  assert raised.value.details["field"] == field


@pytest.mark.parametrize("name", ["BADQPOS", "BADQVEL", "BADQACC", "BADCTRL"])
def test_numerical_warning_fails_even_with_forward_clock(name):
  sim = _free_card()
  warning = getattr(mujoco.mjtWarning, f"mjWARN_{name}")
  sim.data.warning[warning].number = 1
  with pytest.raises(PhysicsIntegrityError, match="numerical warning"):
    check_physics_state(sim.data, stage="after_step")


@pytest.mark.parametrize("after", [0.0, 34.0, 34.004, np.nan, np.inf])
def test_clock_rejects_reset_stall_wrong_step_and_nonfinite(after):
  with pytest.raises(PhysicsIntegrityError):
    check_physics_clock(34.0, after, timestep=0.002)


def test_multistep_checks_each_step_and_stops_before_catchup(monkeypatch):
  tiny = _free_card()
  sim = object.__new__(MidForcePokerSimulation)
  sim.model, sim.data, sim.timestep = tiny.model, tiny.data, tiny.timestep
  sim.data.time = 34.0
  calls = []

  def injected_step(instance, steps):
    assert steps == 1
    calls.append(steps)
    instance.data.time = instance.data.time + 0.002 if len(calls) == 1 else 0.0

  monkeypatch.setattr(ForceLimitedPokerSimulation, "step", injected_step)
  with pytest.raises(PhysicsIntegrityError, match="backwards"):
    sim.step(100)
  assert calls == [1, 1]


def test_recorder_catches_rollback_between_sample_deadlines():
  sim = _free_card()
  recorder = object.__new__(EpisodeRecorder)
  recorder.sim = sim
  recorder.config = SimpleNamespace(cameras=())
  recorder.taskspace_capture = None
  recorder._last_observed_time = 0.0
  recorder._next_state_time = recorder._next_camera_time = 1.0
  sim.data.time = 0.008
  recorder.observe(sim, "test")
  sim.data.time = 0.004  # neither observation reaches a recording deadline
  with pytest.raises(PhysicsIntegrityError, match="backwards"):
    recorder.observe(sim, "test")
  assert recorder._last_observed_time == 0.008
