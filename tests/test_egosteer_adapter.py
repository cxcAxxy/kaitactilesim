from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from kaihand_tactile_env.workcell.egosteer_adapter import (
  CV_FROM_MUJOCO_CAMERA,
  EGOSTEER_HOMOGENEOUS_EPS,
  FINGERS,
  SITE_FROM_EGOSTEER_WRIST,
  camera_from_world_opencv,
  decode_action_targets,
  model_state_history,
  raw_unified_from_simulation,
  relative_to_absolute,
)


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


def _pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
  result = np.eye(4, dtype=np.float64)
  result[:3, :3] = rotation
  result[:3, 3] = position
  return result


def _rot6d(rotation: np.ndarray) -> np.ndarray:
  return np.concatenate((rotation[:, 0], rotation[:, 1]))


def _put_wrist(state: np.ndarray, side: str, pose: np.ndarray) -> None:
  if side == "left":
    state[0:3] = pose[:3, 3]
    state[6:12] = _rot6d(pose[:3, :3])
  else:
    state[3:6] = pose[:3, 3]
    state[12:18] = _rot6d(pose[:3, :3])


def _wrist(state: np.ndarray, side: str) -> np.ndarray:
  if side == "left":
    position, six = state[0:3], state[6:12]
  else:
    position, six = state[3:6], state[12:18]
  first = six[:3] / np.linalg.norm(six[:3])
  second = six[3:] - first * np.dot(first, six[3:])
  second /= np.linalg.norm(second)
  return _pose(position, np.column_stack((first, second, np.cross(first, second))))


def _fingertip_slice(side: str) -> slice:
  return slice(18, 33) if side == "left" else slice(33, 48)


@dataclass(frozen=True)
class _NamedSite:
  id: int


class _FakeModel:
  def __init__(self, names: tuple[str, ...]) -> None:
    self._sites = {name: _NamedSite(index) for index, name in enumerate(names)}

  def site(self, name: str) -> _NamedSite:
    if name not in self._sites:
      raise KeyError(name)
    return self._sites[name]


class _FakeData:
  def __init__(self, poses: tuple[np.ndarray, ...]) -> None:
    self.site_xpos = np.stack([pose[:3, 3] for pose in poses])
    self.site_xmat = np.stack([pose[:3, :3].reshape(-1) for pose in poses])


class _FakeSimulation:
  def __init__(
    self,
    named_poses: dict[str, np.ndarray],
    control_poses: dict[str, np.ndarray],
  ) -> None:
    names = tuple(named_poses)
    self.model = _FakeModel(names)
    self.data = _FakeData(tuple(named_poses[name] for name in names))
    self._control_poses = control_poses

  def current_pose_matrix(self, side: str) -> tuple[np.ndarray, np.ndarray]:
    pose = self._control_poses[side]
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def _fake_simulation() -> tuple[_FakeSimulation, dict[str, np.ndarray]]:
  bases = {
    "left": _pose(np.array((0.31, 0.22, 0.91)), _rotation((1.0, -2.0, 0.5), 0.43)),
    "right": _pose(np.array((-0.28, 0.17, 0.84)), _rotation((-0.5, 1.0, 2.0), -0.61)),
  }
  # The controlled arm site need not share the hand-base site's orientation.
  control_from_base = {
    "left": _pose(np.array((0.01, -0.02, 0.03)), _rotation((0.0, 1.0, 0.0), 0.2)),
    "right": _pose(np.array((-0.03, 0.01, 0.02)), _rotation((1.0, 0.0, 0.0), -0.3)),
  }
  controls = {
    side: bases[side] @ np.linalg.inv(control_from_base[side])
    for side in ("left", "right")
  }
  named_poses: dict[str, np.ndarray] = {}
  for side in ("left", "right"):
    named_poses[f"hand_{side[0]}_base_link_site"] = bases[side]
    for finger_index, finger in enumerate(FINGERS):
      link = "link6" if finger == "thumb" else "link4"
      offset = np.array(
        (0.015 + 0.011 * finger_index, (-1.0 if side == "left" else 1.0) * 0.008, -0.07)
      )
      fingertip = bases[side].copy()
      fingertip[:3, 3] = bases[side][:3, :3] @ offset + bases[side][:3, 3]
      named_poses[f"hand_{side[0]}_{finger}_{link}_site"] = fingertip
  return _FakeSimulation(named_poses, controls), bases


def test_camera_conversion_matches_converter_opencv_rdf_convention() -> None:
  world_from_camera = _pose(
    np.array((0.25, -0.5, 1.0)), _rotation((1.0, 2.0, -1.0), 0.7)
  )

  expected = CV_FROM_MUJOCO_CAMERA @ np.linalg.inv(world_from_camera)
  np.testing.assert_allclose(
    camera_from_world_opencv(world_from_camera), expected, atol=1.0e-12
  )


def test_raw_unified_matches_converter_layout_and_canonical_wrist_axes() -> None:
  simulation, bases = _fake_simulation()

  state = raw_unified_from_simulation(simulation)

  assert state.shape == (48,)
  assert state.dtype == np.float32
  for side in ("left", "right"):
    expected_wrist = bases[side].copy()
    expected_wrist[:3, :3] = bases[side][:3, :3] @ SITE_FROM_EGOSTEER_WRIST[side]
    np.testing.assert_allclose(_wrist(state, side), expected_wrist, atol=1.0e-7)
    expected_fingertips = np.stack(
      [
        simulation.data.site_xpos[
          simulation.model.site(
            f"hand_{side[0]}_{finger}_{'link6' if finger == 'thumb' else 'link4'}_site"
          ).id
        ]
        for finger in FINGERS
      ]
    )
    np.testing.assert_allclose(
      state[_fingertip_slice(side)].reshape(5, 3),
      expected_fingertips,
      atol=1.0e-7,
    )


def test_model_history_uses_current_camera_for_all_wrists_and_local_fingertips() -> (
  None
):
  wrist_world = {
    "left": (
      _pose(np.array((0.1, 0.2, 0.8)), _rotation((1.0, 0.5, -0.2), 0.3)),
      _pose(np.array((0.2, 0.1, 0.9)), _rotation((0.1, 1.0, 0.4), -0.4)),
    ),
    "right": (
      _pose(np.array((-0.3, 0.4, 0.7)), _rotation((0.2, -0.3, 1.0), 0.5)),
      _pose(np.array((-0.2, 0.3, 0.6)), _rotation((1.0, 0.2, 0.3), 0.2)),
    ),
  }
  local = {
    "left": np.arange(30, dtype=np.float64).reshape(2, 5, 3) * 0.002 - 0.02,
    "right": np.arange(30, 60, dtype=np.float64).reshape(2, 5, 3) * -0.001 + 0.04,
  }
  raw = np.empty((2, 48), dtype=np.float32)
  for frame in range(2):
    for side in ("left", "right"):
      _put_wrist(raw[frame], side, wrist_world[side][frame])
      world_points = (
        local[side][frame] @ wrist_world[side][frame][:3, :3].T
        + wrist_world[side][frame][:3, 3]
      )
      raw[frame, _fingertip_slice(side)] = world_points.reshape(-1)

  current_camera_from_world = np.linalg.inv(
    _pose(np.array((0.4, -0.1, 1.4)), _rotation((-0.2, 1.0, 0.7), 0.8))
  )
  processed = model_state_history(raw, current_camera_from_world)

  assert processed.shape == (2, 48)
  assert processed.dtype == np.float32
  for frame in range(2):
    for side in ("left", "right"):
      np.testing.assert_allclose(
        _wrist(processed[frame], side),
        current_camera_from_world @ wrist_world[side][frame],
        atol=2.0e-7,
      )
      np.testing.assert_allclose(
        processed[frame, _fingertip_slice(side)].reshape(5, 3),
        local[side][frame] / (1.0 + EGOSTEER_HOMOGENEOUS_EPS),
        atol=1.0e-7,
      )


def test_relative_actions_are_left_multiplied_by_current_wrist_pose() -> None:
  state = np.zeros(48, dtype=np.float32)
  relative = np.zeros((2, 48), dtype=np.float32)
  current_poses = {
    "left": _pose(np.array((0.2, -0.1, 0.9)), _rotation((1.0, 2.0, 0.0), 0.4)),
    "right": _pose(np.array((-0.2, 0.3, 1.0)), _rotation((0.0, 1.0, 1.0), -0.5)),
  }
  relative_poses = {
    "left": (
      _pose(np.array((0.01, 0.02, -0.03)), _rotation((1.0, 0.0, 0.0), 0.1)),
      _pose(np.array((-0.02, 0.01, 0.04)), _rotation((0.0, 1.0, 0.0), -0.2)),
    ),
    "right": (
      _pose(np.array((0.03, -0.01, 0.02)), _rotation((0.0, 0.0, 1.0), 0.3)),
      _pose(np.array((0.01, 0.04, -0.02)), _rotation((1.0, 1.0, 0.0), 0.15)),
    ),
  }
  state[18:] = np.linspace(-0.2, 0.2, 30)
  relative[:, 18:] = np.arange(60).reshape(2, 30) * 0.001
  for side in ("left", "right"):
    _put_wrist(state, side, current_poses[side])
    for step in range(2):
      _put_wrist(relative[step], side, relative_poses[side][step])

  absolute = relative_to_absolute(state, relative)

  assert absolute.dtype == np.float32
  for side in ("left", "right"):
    for step in range(2):
      np.testing.assert_allclose(
        _wrist(absolute[step], side),
        current_poses[side] @ relative_poses[side][step],
        atol=2.0e-7,
      )
  np.testing.assert_allclose(absolute[:, 18:], relative[:, 18:] + state[18:])


def test_decode_maps_canonical_wrist_to_control_site_and_fingertips_to_world() -> None:
  simulation, bases = _fake_simulation()
  world_from_camera = _pose(
    np.array((0.5, -0.4, 1.2)), _rotation((0.2, 0.3, 1.0), -0.7)
  )
  camera_from_world = np.linalg.inv(world_from_camera)
  desired_world_wrist = {
    "left": _pose(np.array((0.22, 0.31, 0.92)), _rotation((1.0, -0.3, 0.4), 0.55)),
    "right": _pose(np.array((-0.18, 0.27, 0.88)), _rotation((0.4, 1.0, -0.2), -0.45)),
  }
  local_fingertips = {
    "left": np.arange(15, dtype=np.float64).reshape(5, 3) * 0.003 - 0.02,
    "right": np.arange(15, 30, dtype=np.float64).reshape(5, 3) * -0.002 + 0.03,
  }
  actions = np.empty((1, 48), dtype=np.float32)
  for side in ("left", "right"):
    _put_wrist(actions[0], side, camera_from_world @ desired_world_wrist[side])
    actions[0, _fingertip_slice(side)] = local_fingertips[side].reshape(-1)

  targets = decode_action_targets(simulation, actions, camera_from_world)

  for side in ("left", "right"):
    current_control_position, current_control_rotation = simulation.current_pose_matrix(
      side
    )
    current_world_control = _pose(current_control_position, current_control_rotation)
    current_world_wrist = bases[side].copy()
    current_world_wrist[:3, :3] = bases[side][:3, :3] @ SITE_FROM_EGOSTEER_WRIST[side]
    control_from_wrist = np.linalg.inv(current_world_control) @ current_world_wrist
    expected_world_control = desired_world_wrist[side] @ np.linalg.inv(
      control_from_wrist
    )
    side_targets = targets.for_side(side)
    np.testing.assert_allclose(
      side_targets.site_positions_world[0], expected_world_control[:3, 3], atol=2.0e-7
    )
    np.testing.assert_allclose(
      side_targets.site_rotations_world[0], expected_world_control[:3, :3], atol=2.0e-7
    )
    quaternion = side_targets.site_quaternions_wxyz[0]
    assert quaternion[0] >= 0.0
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    expected_fingertips = (
      local_fingertips[side] @ desired_world_wrist[side][:3, :3].T
      + desired_world_wrist[side][:3, 3]
    ) / (1.0 + EGOSTEER_HOMOGENEOUS_EPS)
    np.testing.assert_allclose(
      side_targets.fingertips_world[0], expected_fingertips, atol=2.0e-7
    )


@pytest.mark.parametrize(
  ("function", "args"),
  (
    (model_state_history, (np.zeros(48), np.eye(4))),
    (relative_to_absolute, (np.zeros(48), np.zeros(48))),
  ),
)
def test_sequence_inputs_reject_missing_time_dimension(function, args) -> None:
  with pytest.raises(ValueError, match="shape"):
    function(*args)
