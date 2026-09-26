from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
  path = ROOT / "scripts/workcell/run_usb_openwam_policy.py"
  spec = importlib.util.spec_from_file_location("run_usb_openwam_policy_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
  unit = np.asarray(axis, dtype=np.float64)
  unit /= np.linalg.norm(unit)
  skew = np.array(
    (
      (0.0, -unit[2], unit[1]),
      (unit[2], 0.0, -unit[0]),
      (-unit[1], unit[0], 0.0),
    )
  )
  return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _pose(position: tuple[float, float, float], rotation: np.ndarray) -> np.ndarray:
  result = np.eye(4, dtype=np.float64)
  result[:3, :3] = rotation
  result[:3, 3] = position
  return result


class _Model:
  def __init__(self) -> None:
    self.requested_sites: list[str] = []

  def site(self, name: str) -> SimpleNamespace:
    self.requested_sites.append(name)
    if name != "hand_r_base_link_site":
      raise KeyError(name)
    return SimpleNamespace(id=0)


class _Simulation:
  def __init__(self, native_wrist: np.ndarray, control: np.ndarray, joint_names) -> None:
    self.model = _Model()
    self.data = SimpleNamespace(
      site_xpos=native_wrist[None, :3, 3].copy(),
      site_xmat=native_wrist[None, :3, :3].reshape(1, 9).copy(),
      qpos=np.zeros(64, dtype=np.float64),
    )
    self._control = control
    # Reverse the storage addresses so the test detects accidental qpos slicing.
    self._qpos_address = {
      name: 40 - index for index, name in enumerate(joint_names)
    }
    self.pose_target = None
    self.hand_target = None

  def current_pose_matrix(self, side: str):
    assert side == "right"
    return self._control[:3, 3].copy(), self._control[:3, :3].copy()

  def set_pose_target(self, side: str, position: np.ndarray, quaternion: np.ndarray):
    self.pose_target = (side, np.asarray(position), np.asarray(quaternion))
    return SimpleNamespace(success=True, position_error=0.0)

  def set_hand_joint_targets(self, names, values: np.ndarray) -> int:
    self.hand_target = (tuple(names), np.asarray(values))
    return len(names)


def test_world_from_wrist_reads_native_hand_base_site() -> None:
  runner = _load_runner()
  native = _pose((0.31, -0.22, 0.87), _rotation((1.0, -2.0, 0.5), 0.63))
  unrelated_control = _pose(
    (-0.45, 0.17, 1.12), _rotation((0.0, 1.0, 1.0), -0.41)
  )
  simulation = _Simulation(native, unrelated_control, runner.RIGHT_HAND_JOINT_NAMES)

  actual = runner._world_from_wrist(simulation)

  np.testing.assert_allclose(actual, native, atol=1.0e-12)
  assert simulation.model.requested_sites == ["hand_r_base_link_site"]


def test_state_29_is_native_wrist_then_ordered_right_hand_joints() -> None:
  runner = _load_runner()
  native = _pose((0.28, 0.16, 0.91), _rotation((-1.0, 0.5, 2.0), -0.72))
  simulation = _Simulation(native, np.eye(4), runner.RIGHT_HAND_JOINT_NAMES)
  hand = np.linspace(-0.4, 0.7, len(runner.RIGHT_HAND_JOINT_NAMES))
  for name, value in zip(runner.RIGHT_HAND_JOINT_NAMES, hand, strict=True):
    simulation.data.qpos[simulation._qpos_address[name]] = value

  state = runner._state_29(simulation)

  expected_rot6d = np.concatenate((native[:3, 0], native[:3, 1]))
  expected = np.concatenate((native[:3, 3], expected_rot6d, hand)).astype(np.float32)
  assert state.shape == (29,)
  assert state.dtype == np.float32
  np.testing.assert_allclose(state, expected, atol=1.0e-7)


def test_apply_action_preserves_native_wrist_to_control_transform() -> None:
  runner = _load_runner()
  initial_control = _pose(
    (0.17, -0.38, 0.79), _rotation((0.5, 1.0, -0.5), 0.47)
  )
  control_from_native_wrist = _pose(
    (0.025, -0.014, 0.031), _rotation((1.0, -0.5, 0.25), -0.36)
  )
  initial_native_wrist = initial_control @ control_from_native_wrist
  simulation = _Simulation(
    initial_native_wrist, initial_control, runner.RIGHT_HAND_JOINT_NAMES
  )

  target_native_wrist = _pose(
    (0.42, 0.11, 0.96), _rotation((-0.25, 1.0, 0.75), 0.81)
  )
  target_hand = np.linspace(-0.2, 0.9, len(runner.RIGHT_HAND_JOINT_NAMES))
  action = np.concatenate(
    (
      target_native_wrist[:3, 3],
      target_native_wrist[:3, 0],
      target_native_wrist[:3, 1],
      target_hand,
    )
  )

  runner._apply_action(simulation, action)

  expected_control = target_native_wrist @ np.linalg.inv(control_from_native_wrist)
  assert simulation.pose_target is not None
  side, position, quaternion = simulation.pose_target
  assert side == "right"
  np.testing.assert_allclose(position, expected_control[:3, 3], atol=1.0e-12)
  expected_quaternion = runner._quaternion_wxyz_from_rotation(
    expected_control[:3, :3]
  )
  np.testing.assert_allclose(quaternion, expected_quaternion, atol=1.0e-12)

  assert simulation.hand_target is not None
  names, values = simulation.hand_target
  assert names == runner.RIGHT_HAND_JOINT_NAMES
  np.testing.assert_allclose(values, target_hand, atol=1.0e-12)


def test_recording_requires_a_reference_dataset() -> None:
  runner = _load_runner()

  with pytest.raises(SystemExit):
    runner.parse_args(["--output-dir", "/tmp/openwam-eval"])

  args = runner.parse_args(
    [
      "--output-dir", "/tmp/openwam-eval",
      "--reference-dataset", "/dataset/0920_200",
    ]
  )
  assert args.record is True
  assert args.reference_dataset == Path("/dataset/0920_200")

  args = runner.parse_args(["--output-dir", "/tmp/openwam-eval", "--no-record"])
  assert args.record is False
  assert args.reference_dataset is None


def test_openwam_rejects_short_action_chunk() -> None:
  runner = _load_runner()
  client = runner.OpenWAMClient("ws://test", timeout=1.0)

  class Socket:
    async def send(self, _payload):
      return None

    async def recv(self):
      return json.dumps({"type": "action", "action": [[0.0] * 29]})

  client.ws = Socket()
  rgb = np.zeros((2, 2, 3), dtype=np.uint8)
  with pytest.raises(RuntimeError, match="at least 16 OpenWAM actions"):
    asyncio.run(client.infer(rgb, rgb, np.zeros(29), required_steps=16))


def test_openwam_rejects_large_ik_orientation_error() -> None:
  runner = _load_runner()
  simulation = _Simulation(np.eye(4), np.eye(4), runner.RIGHT_HAND_JOINT_NAMES)
  simulation.set_pose_target = lambda *_: SimpleNamespace(
    success=False, position_error=0.0, orientation_error=0.5
  )
  action = np.zeros(29)
  action[3:9] = (1, 0, 0, 0, 1, 0)
  with pytest.raises(RuntimeError, match="orientation_error"):
    runner._apply_action(simulation, action)
