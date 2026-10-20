from __future__ import annotations

import ast
import gc
import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def simulation():
  pytest.importorskip("mujoco")
  from kaihand_tactile_env.tasks.bulb_screw_inplace.task import (
    BulbScrewInplaceSimulation,
  )

  simulation = BulbScrewInplaceSimulation(position_seed=0)
  yield simulation
  del simulation
  gc.collect()


def _executor(simulation, **kwargs):
  from kaihand_tactile_env.tasks.bulb_screw_inplace.execution import (
    BulbScrewInplaceExecutor,
  )

  return BulbScrewInplaceExecutor(simulation, **kwargs)


def _cli():
  pytest.importorskip("mujoco")
  path = ROOT / "scripts/workcell/view_bulb_screw_inplace.py"
  spec = importlib.util.spec_from_file_location("bulb_screw_inplace_cli", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_inplace_expert_does_not_write_object_state_or_apply_wrenches():
  path = ROOT / "src/kaihand_tactile_env/tasks/bulb_screw_inplace/execution.py"
  tree = ast.parse(path.read_text())
  forbidden_calls = {
    "set_object_pose",
    "initialize_threaded",
    "confirm_tightening",
    "reset",
  }
  forbidden_writes = {
    "qpos", "qvel", "eq_active", "xfrc_applied", "qfrc_applied", "eq_data"
  }
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      assert node.func.attr not in forbidden_calls
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
      assert node.attr not in forbidden_writes
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
      assert not (
        isinstance(node.value, ast.Attribute)
        and node.value.attr in forbidden_writes
      )


def test_default_expert_is_five_finger_and_constructor_does_not_move_scene(simulation):
  simulation.reset()
  qpos = simulation.data.qpos.copy()
  qvel = simulation.data.qvel.copy()
  pose = simulation.object_pose("bulb").copy()
  executor = _executor(simulation)
  assert executor.finger_count == 5
  assert executor.speed == "fast"
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  np.testing.assert_array_equal(simulation.object_pose("bulb"), pose)


@pytest.mark.parametrize(
  "arguments", [{"max_strokes": 0}, {"max_strokes": True}, {"max_strokes": 1.5},
                {"speed": "turbo"}, {"grasp_mode": "pinch"}]
)
def test_invalid_expert_configuration(simulation, arguments):
  with pytest.raises(ValueError):
    _executor(simulation, **arguments)


def test_legacy_tabletop_simulation_is_not_accepted():
  pytest.importorskip("mujoco")
  from kaihand_tactile_env.tasks.bulb_screw_inplace.execution import (
    BulbScrewInplaceExecutor,
  )

  with pytest.raises(ValueError, match="requires BulbScrewInplaceSimulation"):
    BulbScrewInplaceExecutor(object())


def test_cancellation_is_not_task_success_and_cannot_repeat(simulation):
  simulation.reset()
  executor = _executor(simulation, should_stop=lambda: True)
  result = executor.run()
  assert result.task == "bulb-screw-inplace"
  assert not result.success
  assert result.reason == "cancelled"
  assert result.elapsed_s == 0
  assert not result.physical_success
  assert not result.home_return_verified
  assert result.strokes == 0
  assert asdict(result)["state"]["success"] is False
  with pytest.raises(RuntimeError, match="fresh executor"):
    executor.execute()


def test_rejects_disengaged_initial_scene_without_advancing(simulation, monkeypatch):
  from dataclasses import replace

  simulation.reset()
  executor = _executor(simulation)
  executor._state = replace(executor._state, initially_threaded_verified=False)
  monkeypatch.setattr(
    executor, "_move_ee",
    lambda *args, **kwargs: pytest.fail("invalid initial scene must not move"),
  )
  result = executor.execute()
  assert not result.success and result.elapsed_s == 0
  assert "thread entrance" in result.reason


def test_cleanup_failure_retains_physical_completion(simulation, monkeypatch):
  from dataclasses import replace

  from kaihand_tactile_env.tasks.bulb_screw.execution import _TaskFailure

  simulation.reset()
  executor = _executor(simulation)

  def interrupted_cleanup():
    executor._state = replace(
      executor._state,
      success=True,
      physical_success=True,
      released_stably=True,
      home_return_verified=False,
    )
    raise _TaskFailure("cancelled")

  monkeypatch.setattr(executor, "_motion", interrupted_cleanup)
  result = executor.execute()
  assert not result.success
  assert result.reason == "cleanup_failed: cancelled"
  assert result.physical_success and result.state.success
  assert not result.home_return_verified


def test_headless_cli_defaults_and_seed():
  args = _cli().parse_args(["--headless", "--run-task", "--seed", "4"])
  assert args.headless and args.run_task and args.seed == 4
  assert args.max_strokes == 20 and args.speed == "fast"


@pytest.mark.parametrize(
  "arguments",
  [["--seed", "-1"], ["--max-strokes", "0"], ["--duration", "nan"],
   ["--duration", "0"], ["--mechanics-demo"], ["--threaded"], ["--grasp", "pinch"]],
)
def test_headless_cli_rejects_invalid_or_shortcut_modes(arguments):
  with pytest.raises(SystemExit) as error:
    _cli().parse_args(arguments)
  assert error.value.code == 2


def test_full_inplace_expert_uses_only_actuators_and_measured_progress(
  simulation, monkeypatch
):
  """A CPU-only physical episode; no renderer, GPU, policy, or source writes."""
  from kaihand_tactile_env.tasks.bulb_screw import config

  simulation.reset()
  position = simulation.object_pose("bulb")[:3].copy()

  def forbidden(*args, **kwargs):
    pytest.fail("expert attempted an object-state or reset shortcut")

  monkeypatch.setattr(simulation, "set_object_pose", forbidden)
  monkeypatch.setattr(simulation, "initialize_threaded", forbidden)
  previous_position = position.copy()
  observed_phases = []
  physical_before_home = False
  executor = None

  def observe(sim, phase):
    nonlocal previous_position, physical_before_home
    observed_phases.append(phase)
    assert sim.task_phase == phase
    assert not np.any(sim.data.xfrc_applied)
    # Shared robot arm gravity compensation is unchanged and permitted;
    # neither the bulb freejoint nor the passive thread receives assistance.
    bulb_dof = int(sim.model.joint("bulb_freejoint").dofadr[0])
    assert not np.any(sim.data.qfrc_applied[bulb_dof:bulb_dof + 6])
    assert sim.data.qfrc_applied[sim._slide_dof] == 0
    assert sim.data.qfrc_applied[sim._hinge_dof] == 0
    actual = sim.object_pose("bulb")[:3]
    assert np.linalg.norm(actual - previous_position) < 0.002
    previous_position = actual.copy()
    assert phase not in {"lift", "transfer", "align_thread", "engage_thread"}
    if phase == "clear_for_home":
      physical_before_home = physical_before_home or executor._state.physical_success

  executor = _executor(simulation, observer=observe)
  result = executor.execute()
  assert result.success, result.reason
  assert result.task == "bulb-screw-inplace"
  assert result.initially_threaded_verified
  assert result.grasp_verified and result.effective_rotation_verified
  assert result.tightening_verified and result.released_stably
  assert result.physical_success and result.home_return_verified
  assert physical_before_home
  assert result.state.clockwise_turns == pytest.approx(config.TARGET_TURNS, abs=0.02)
  assert 1 <= result.strokes <= 6  # Diagnostic, not a fixed-count success gate.
  assert result.maximum_lift_m < 0.002
  assert result.tightening_stall_duration_s >= config.TIGHTENING_HOLD_S - 1e-9
  assert result.maximum_wrist_rotation_deg < 2
  assert result.maximum_wrist_displacement_m < 0.002
  assert {"approach", "grasp", "turn", "tighten", "release", "verify_seated",
          "return_home", "verify_complete"} <= set(observed_phases)
  assert max(result.final_fingertip_load_n) < config.RELEASE_FINGERTIP_LOAD_N
  assert simulation.arm_goal_error("right") < 0.02
