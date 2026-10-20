"""Regression checks for EgoTouch fingertip-pose execution plumbing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def modules():
  root = Path(__file__).parents[1]
  sys.path.insert(0, str(root / "src"))
  sys.path.insert(0, str(root / "scripts/workcell"))
  try:
    adapter = importlib.import_module("kaihand_tactile_env.shared.egosteer_adapter")
    shared_runner = importlib.import_module("run_shared_task_egotouch_policy")
    executor = importlib.import_module("run_egosteer_policy")
    yield adapter, shared_runner, executor
  finally:
    sys.path.remove(str(root / "scripts/workcell"))
    sys.path.remove(str(root / "src"))


def _z_rotation(angle: float) -> np.ndarray:
  cosine, sine = np.cos(angle), np.sin(angle)
  return np.array(
    ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
  )


def test_decode_world_targets_preserves_complete_fingertip_poses(
  monkeypatch, modules
) -> None:
  _adapter, runner, _executor = modules
  horizon = 2
  wrists = np.broadcast_to(np.eye(4), (horizon, 2, 4, 4)).copy()
  tips = np.broadcast_to(np.eye(4), (horizon, 2, 5, 4, 4)).copy()
  for step in range(horizon):
    for side in range(2):
      for finger in range(5):
        tips[step, side, finger, :3, :3] = _z_rotation(
          0.05 * (1 + step + side + finger)
        )
        tips[step, side, finger, :3, 3] = (
          0.01 * step,
          0.02 * side,
          0.03 * finger,
        )

  monkeypatch.setattr(runner, "_named_site_pose", lambda *_args: np.eye(4))
  simulation = SimpleNamespace(
    current_pose_matrix=lambda _side: (np.zeros(3), np.eye(3))
  )
  decoded = runner.decode_world_targets(
    simulation,
    {"wrists_world": wrists.tolist(), "tips_world": tips.tolist()},
  )

  for side_index, side in enumerate(("left", "right")):
    target = decoded.for_side(side)
    np.testing.assert_allclose(
      target.fingertip_rotations_world, tips[:, side_index, :, :3, :3]
    )
    np.testing.assert_allclose(
      target.fingertip_poses_world, tips[:, side_index]
    )


def test_worker_nonfinite_rejection_is_replanned_only_in_observe_mode(
  modules,
) -> None:
  _adapter, runner, _executor = modules
  response = {
    "policy_action_rejected": "nonfinite_action",
    "error": "EgoTouch policy actions contain nonfinite values",
    "seconds": 0.1,
  }
  simulation = SimpleNamespace(data=SimpleNamespace(time=1.25))
  guards = importlib.import_module(
    "kaihand_tactile_env.shared.evaluation_guards"
  )
  guards.configure_evaluation_guards(simulation, "observe")

  assert runner.decode_policy_response(simulation, response, horizon=50) is None
  assert simulation._evaluation_action_rejected
  event = guards.evaluation_guard_report(simulation)["guard_events"][0]
  assert event["kind"] == "nonfinite_action"
  assert event["count"] == 1

  guards.configure_evaluation_guards(simulation, "strict")
  with pytest.raises(
    RuntimeError, match="EgoTouch policy actions contain nonfinite values"
  ):
    runner.decode_policy_response(simulation, response, horizon=50)


@pytest.mark.parametrize(
  "response",
  [
    {
      "policy_action_rejected": "different_rejection",
      "error": "EgoTouch policy actions contain nonfinite values",
    },
    {
      "policy_action_rejected": "nonfinite_action",
      "error": "different error",
    },
  ],
)
def test_worker_rejection_marker_must_match_exact_contract(
  modules, response
) -> None:
  _adapter, runner, _executor = modules
  simulation = SimpleNamespace(data=SimpleNamespace(time=0.0))
  guards = importlib.import_module(
    "kaihand_tactile_env.shared.evaluation_guards"
  )
  guards.configure_evaluation_guards(simulation, "observe")

  with pytest.raises(RuntimeError, match="invalid EgoTouch policy-action"):
    runner.decode_policy_response(simulation, response, horizon=50)
  assert guards.evaluation_guard_report(simulation)["guard_events"] == []


class _Model:
  def __init__(self, joint_names: tuple[str, ...]) -> None:
    self._ids = {name: index for index, name in enumerate(joint_names)}
    self.jnt_range = np.tile((-1.0, 1.0), (len(joint_names), 1))
    self.jnt_limited = np.ones(len(joint_names), dtype=bool)

  def joint(self, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=self._ids[name])


class _Simulation:
  def __init__(self, *, hand_success: bool = True) -> None:
    self.joint_names = tuple(f"joint_{index}" for index in range(20))
    self.model = _Model(self.joint_names)
    self.ik_data = SimpleNamespace(
      site_xpos=np.zeros((5, 3), dtype=float),
      site_xmat=np.tile(np.eye(3).reshape(1, 9), (5, 1)),
    )
    self._fingertip_site_ids = {"right": np.arange(5)}
    self._hand_targets = {
      "right": {name: 0.0 for name in self.joint_names}
    }
    self._arm_goal = {"right": np.zeros(7, dtype=float)}
    self._pose_target_position = {"right": np.zeros(3, dtype=float)}
    self._pose_target_rotation = {"right": np.eye(3)}
    self._target_mocap = {"right": 0}
    self.data = SimpleNamespace(
      time=0.0,
      mocap_pos=np.zeros((1, 3), dtype=float),
      mocap_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
    )
    self.hand_success = hand_success
    self.solve_kwargs = None
    self.commanded = None

  def current_pose_matrix(self, side: str):
    assert side == "right"
    return np.zeros(3), np.eye(3)

  def set_pose_target(self, *_args):
    return SimpleNamespace(
      success=True,
      joint_positions=np.zeros(7),
      position_error=0.0,
      orientation_error=0.0,
      iterations=1,
    )

  def solve_hand_ik(self, _side, _positions, **kwargs):
    self.solve_kwargs = kwargs
    error = 0.0 if self.hand_success else 0.002
    return SimpleNamespace(
      success=self.hand_success,
      joint_names=self.joint_names,
      joint_positions=np.linspace(-0.5, 0.5, 20),
      fingertip_errors=np.full(5, error),
      position_error=error,
      iterations=7,
    )

  def set_hand_joint_targets(self, names, positions):
    self.commanded = (tuple(names), np.asarray(positions).copy())
    return len(names)


def _arguments(executor, **overrides):
  values = {
    "control_side": "right",
    "max_wrist_jump": 0.15,
    "max_arm_position_error": 0.03,
    "max_arm_orientation_error": 0.30,
    "hand_ik_tolerance": 0.0015,
    "max_hand_error": 0.02,
    "strict_ik": False,
    "hold_failed_hand_ik": True,
    "hand_ik_orientation_weights": executor.DEFAULT_HAND_IK_ORIENTATION_WEIGHTS,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _targets(adapter, rotations):
  target = adapter.SideActionTargets(
    site_positions_world=np.zeros((1, 3)),
    site_rotations_world=np.eye(3)[None],
    site_quaternions_wxyz=np.array(((1.0, 0.0, 0.0, 0.0),)),
    fingertips_world=np.zeros((1, 5, 3)),
    fingertip_rotations_world=rotations,
  )
  return SimpleNamespace(for_side=lambda _side: target)


def test_pose_targets_enable_orientation_ik_and_emit_fk_diagnostics(modules) -> None:
  adapter, _runner, executor = modules
  rotations = np.tile(np.eye(3), (1, 5, 1, 1))
  rotations[0, 0] = _z_rotation(0.25)
  simulation = _Simulation()
  records = []

  returned = executor._command_action_step(
    simulation,
    _targets(adapter, rotations),
    0,
    _arguments(executor),
    executor.RolloutStats(),
    diagnostics=records,
  )

  assert returned is records and len(records) == 1
  np.testing.assert_allclose(
    simulation.solve_kwargs["target_rotations"], rotations[0]
  )
  np.testing.assert_allclose(
    simulation.solve_kwargs["orientation_weights"],
    executor.DEFAULT_HAND_IK_ORIENTATION_WEIGHTS,
  )
  assert records[0]["ik_mode"] == "position_and_orientation"
  assert records[0]["orientation_error_rad"][0] == pytest.approx(0.25)
  expected_pad_error = np.arccos(
    np.clip(
      executor.FINGERTIP_PAD_NORMAL_AXES_LOCAL["right"][0]
      @ rotations[0, 0]
      @ executor.FINGERTIP_PAD_NORMAL_AXES_LOCAL["right"][0],
      -1.0,
      1.0,
    )
  )
  assert records[0]["pad_normal_axis_error_rad"][0] == pytest.approx(
    expected_pad_error
  )
  assert records[0]["maximum_absolute_joint_delta_rad"] == pytest.approx(0.5)
  assert simulation.commanded is not None


def test_position_only_target_keeps_legacy_solver_call(modules) -> None:
  adapter, _runner, executor = modules
  simulation = _Simulation()
  records = []
  records = executor._command_action_step(
    simulation,
    _targets(adapter, None),
    0,
    _arguments(executor),
    executor.RolloutStats(),
    diagnostics=records,
  )

  assert "target_rotations" not in simulation.solve_kwargs
  assert "orientation_weights" not in simulation.solve_kwargs
  assert records[0]["ik_mode"] == "position_only"
  assert records[0]["orientation_error_rad"] is None
  assert records[0]["pad_normal_axis_error_rad"] is None


def test_failed_pose_ik_holds_previous_hand_command(modules) -> None:
  adapter, _runner, executor = modules
  rotations = np.tile(np.eye(3), (1, 5, 1, 1))
  simulation = _Simulation(hand_success=False)
  records = []
  records = executor._command_action_step(
    simulation,
    _targets(adapter, rotations),
    0,
    _arguments(executor),
    executor.RolloutStats(),
    diagnostics=records,
  )

  assert simulation.commanded is None
  assert records[0]["hold_last"] is True
  assert records[0]["command_applied"] is False
