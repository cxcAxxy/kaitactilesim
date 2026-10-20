"""Checkpoint-bound KaiHand observation and action codec for FTP-1.

The poker checkpoint was trained with a 120-dimensional canonical state/action
layout.  KaiHand contributes a right wrist pose, seven right arm joints and
twenty right hand joints; absent robot groups stay zero and must never be sent
to the physical controller.  This module deliberately contains no model or
MuJoCo dependency so conversion, serving and contract tests can share the same
layout code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

FTP1_ACTION_HORIZON: Final = 32
FTP1_MODEL_ACTION_DIM: Final = 120
FTP1_PHYSICAL_ACTION_DIM: Final = 27
FTP1_SINGLE_ARM_REP_DIM: Final = 48
FTP1_ARM_JOINT_DIM: Final = 7
FTP1_HAND_SLOT_DIM: Final = 32
FTP1_RESERVED_ACTION_DIM: Final = 15
FTP1_ACTION_START_INDEX: Final = 1
FTP1_CONTROL_HZ: Final = 30
FTP1_IMAGE_SHAPE_HWC: Final = (240, 320, 3)
FTP1_TACTILE_RAW_SHAPE: Final = (5, 3, 7, 5)
FTP1_TACTILE_SHAPE: Final = (1, 5, 7, 5, 3)
FTP1_TACTILE_KEY: Final = "right_tactile_fingers"
FTP1_TACTILE_SENSOR: Final = "KaiHandSimTaxelV1"
FTP1_TACTILE_FUNCTION_AREAS: Final = (0, 1, 2, 3, 4)
FTP1_ACTION_JOINT_REPS: Final = (
  "mix",
  "arm_relative_hand_absolute",
)

RIGHT_ARM_JOINT_NAMES: Final = tuple(
  f"right_arm_joint{joint}" for joint in range(1, 8)
)
RIGHT_HAND_JOINT_NAMES: Final = (
  "hand_r_thumb_joint1",
  "hand_r_thumb_joint2",
  "hand_r_thumb_joint3",
  "hand_r_thumb_joint5",
  "hand_r_index_joint1",
  "hand_r_index_joint2",
  "hand_r_index_joint3",
  "hand_r_index_joint4",
  "hand_r_middle_joint1",
  "hand_r_middle_joint2",
  "hand_r_middle_joint3",
  "hand_r_middle_joint4",
  "hand_r_ring_joint1",
  "hand_r_ring_joint2",
  "hand_r_ring_joint3",
  "hand_r_ring_joint4",
  "hand_r_pinky_joint1",
  "hand_r_pinky_joint2",
  "hand_r_pinky_joint3",
  "hand_r_pinky_joint4",
)
RIGHT_JOINT_NAMES: Final = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# This is the exact FAAS mapping frozen into the converted poker training set.
RIGHT_HAND_FAAS_INDICES: Final = (
  1,
  26,
  2,
  3,
  6,
  7,
  8,
  9,
  11,
  12,
  13,
  14,
  16,
  17,
  18,
  19,
  21,
  22,
  23,
  24,
)

RIGHT_TACTILE_LINK_NAMES: Final = (
  "hand_r_thumb_link6",
  "hand_r_index_link4",
  "hand_r_middle_link4",
  "hand_r_ring_link4",
  "hand_r_pinky_link4",
)

_RIGHT_WRIST_SLICE: Final = slice(0, 9)
_RIGHT_ARM_SLICE: Final = slice(9, 16)
_RIGHT_HAND_OFFSET: Final = 16
_ROTATION_EPS: Final = 1.0e-8


@dataclass(frozen=True)
class FTP1DecodedActions:
  """Denormalized FTP-1 predictions decoded for the KaiHand workcell."""

  joint_actions: np.ndarray
  wrist_world_transforms: np.ndarray


def _finite_array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be numeric") from error
  if array.shape != shape:
    raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} must contain only finite values")
  return array


def validate_world_transform(name: str, value: object) -> np.ndarray:
  """Validate and return a rigid ``world_T_frame`` transform as float64."""

  transform = _finite_array(name, value, (4, 4)).copy()
  if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7):
    raise ValueError(f"{name} must have homogeneous final row [0,0,0,1]")
  rotation = transform[:3, :3]
  if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6):
    raise ValueError(f"{name} rotation must be orthonormal")
  if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
    raise ValueError(f"{name} rotation must be proper")
  return transform


def _matrix_to_pose9d(transform: np.ndarray) -> np.ndarray:
  """Match ``openpi.pose_utils.mat_to_pose9d`` (translation + first two rows)."""

  translation = transform[..., :3, 3]
  rotation_6d = transform[..., :2, :3].reshape(transform.shape[:-2] + (6,))
  return np.concatenate((translation, rotation_6d), axis=-1)


def _pose9d_to_matrix(pose: np.ndarray) -> np.ndarray:
  """Invert FTP-1's row-major 6D rotation representation, rejecting degeneracy."""

  first = pose[..., 3:6]
  second = pose[..., 6:9]
  first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
  if np.any(first_norm < _ROTATION_EPS):
    raise ValueError("FTP-1 wrist rotation has a degenerate first 6D axis")
  row0 = first / first_norm
  second_orthogonal = second - np.sum(row0 * second, axis=-1, keepdims=True) * row0
  second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
  if np.any(second_norm < _ROTATION_EPS):
    raise ValueError("FTP-1 wrist rotation has collinear 6D axes")
  row1 = second_orthogonal / second_norm
  row2 = np.cross(row0, row1)
  rotation = np.stack((row0, row1, row2), axis=-2)

  transforms = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
  transforms[..., :3, :3] = rotation
  transforms[..., :3, 3] = pose[..., :3]
  transforms[..., 3, 3] = 1.0
  return transforms


def ftp1_action_mask(*, horizon: int | None = None) -> np.ndarray:
  """Return the exact KaiHand-valid FTP-1 action mask.

  The checkpoint also predicts left-arm, head and reserved blocks.  Those
  dimensions, plus unused right-hand FAAS slots, remain masked even though the
  current FTP-1 sampler may emit nonzero values for them.
  """

  mask = np.zeros(FTP1_MODEL_ACTION_DIM, dtype=np.float32)
  mask[_RIGHT_WRIST_SLICE] = 1.0
  mask[_RIGHT_ARM_SLICE] = 1.0
  mask[_RIGHT_HAND_OFFSET + np.asarray(RIGHT_HAND_FAAS_INDICES)] = 1.0
  if horizon is None:
    return mask
  if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
    raise ValueError("horizon must be an integer")
  if int(horizon) <= 0:
    raise ValueError("horizon must be positive")
  return np.broadcast_to(mask, (int(horizon), FTP1_MODEL_ACTION_DIM)).copy()


def encode_ftp1_state(
  state: object,
  world_T_head: object,
  world_T_wrist: object,
) -> np.ndarray:
  """Encode one synchronized KaiHand state into raw FTP-1 scale.

  ``state`` is the physical 27-joint vector in ``RIGHT_JOINT_NAMES`` order.
  The checkpoint uses one history frame, absolute proprioceptive joints and a
  wrist pose relative to that same current wrist pose.  Consequently the wrist
  state is the 9D identity pose, while its camera-frame base is retained by the
  caller's transforms for action decoding.
  """

  joints = _finite_array("state", state, (FTP1_PHYSICAL_ACTION_DIM,))
  head = validate_world_transform("world_T_head", world_T_head)
  wrist = validate_world_transform("world_T_wrist", world_T_wrist)

  head_T_wrist = np.linalg.inv(head) @ wrist
  relative_wrist = np.linalg.inv(head_T_wrist) @ head_T_wrist
  encoded = np.zeros((1, FTP1_MODEL_ACTION_DIM), dtype=np.float32)
  encoded[0, _RIGHT_WRIST_SLICE] = _matrix_to_pose9d(relative_wrist).astype(
    np.float32
  )
  encoded[0, _RIGHT_ARM_SLICE] = joints[:FTP1_ARM_JOINT_DIM].astype(np.float32)
  hand_slots = _RIGHT_HAND_OFFSET + np.asarray(RIGHT_HAND_FAAS_INDICES)
  encoded[0, hand_slots] = joints[FTP1_ARM_JOINT_DIM:].astype(np.float32)
  return encoded


def pack_ftp1_tactile(tactile_map_raw: object) -> np.ndarray:
  """Pack runtime ``[finger,channel,row,col]`` taxels for FTP-1.

  Runtime channel order is normal, signed grid-column tangent and signed
  grid-row tangent.  FTP-1 consumes ``[T,finger,row,col,channel]`` with T=1.
  Values are newtons allocated per taxel and are not clipped or normalized
  here.  The tiny negative tolerance matches the existing dataset converter.
  """

  tactile = _finite_array("tactile_map_raw", tactile_map_raw, FTP1_TACTILE_RAW_SHAPE)
  normal = tactile[:, 0]
  if np.any(normal < -1.0e-7):
    raise ValueError("tactile normal force cannot be negative")
  packed = np.transpose(tactile, (0, 2, 3, 1)).astype(np.float32, copy=True)
  packed[..., 0] = np.maximum(packed[..., 0], 0.0)
  return packed[np.newaxis, ...]


def decode_ftp1_actions(
  model_actions: object,
  state: object,
  world_T_head: object,
  world_T_wrist: object,
  *,
  action_joint_rep: str = "mix",
) -> FTP1DecodedActions:
  """Decode denormalized H32x120 FTP-1 output into absolute joint targets.

  Both supported training recipes use relative right-arm joints.  Under
  ``mix``, the mapped hand slots are relative too; mix's one absolute gripper
  slot is FAAS 28, which is absent from this KaiHand mapping.  Under
  ``arm_relative_hand_absolute``, all twenty mapped hand slots are absolute
  joint targets.  Wrist pose9d is relative to the current wrist in the current
  head-camera frame for both recipes.
  """

  if action_joint_rep not in FTP1_ACTION_JOINT_REPS:
    raise ValueError(
      f"unsupported FTP-1 action_joint_rep: {action_joint_rep!r}"
    )

  actions = _finite_array(
    "model_actions",
    model_actions,
    (FTP1_ACTION_HORIZON, FTP1_MODEL_ACTION_DIM),
  )
  joints = _finite_array("state", state, (FTP1_PHYSICAL_ACTION_DIM,))
  head = validate_world_transform("world_T_head", world_T_head)
  wrist = validate_world_transform("world_T_wrist", world_T_wrist)

  joint_actions = np.empty(
    (FTP1_ACTION_HORIZON, FTP1_PHYSICAL_ACTION_DIM), dtype=np.float32
  )
  joint_actions[:, :FTP1_ARM_JOINT_DIM] = (
    joints[None, :FTP1_ARM_JOINT_DIM] + actions[:, _RIGHT_ARM_SLICE]
  ).astype(np.float32)
  hand_slots = _RIGHT_HAND_OFFSET + np.asarray(RIGHT_HAND_FAAS_INDICES)
  hand_actions = actions[:, hand_slots]
  if action_joint_rep == "mix":
    hand_actions = joints[None, FTP1_ARM_JOINT_DIM:] + hand_actions
  joint_actions[:, FTP1_ARM_JOINT_DIM:] = hand_actions.astype(np.float32)
  if not np.all(np.isfinite(joint_actions)):
    raise ValueError("decoded joint actions are nonfinite")

  relative_wrist = _pose9d_to_matrix(actions[:, _RIGHT_WRIST_SLICE])
  head_T_wrist = np.linalg.inv(head) @ wrist
  head_T_targets = np.matmul(head_T_wrist[None, ...], relative_wrist)
  wrist_world_transforms = np.matmul(head[None, ...], head_T_targets)
  if not np.all(np.isfinite(wrist_world_transforms)):
    raise ValueError("decoded wrist targets are nonfinite")

  return FTP1DecodedActions(
    joint_actions=joint_actions,
    wrist_world_transforms=wrist_world_transforms,
  )


__all__ = [
  "FTP1_ACTION_HORIZON",
  "FTP1_ACTION_JOINT_REPS",
  "FTP1_ACTION_START_INDEX",
  "FTP1_CONTROL_HZ",
  "FTP1DecodedActions",
  "FTP1_IMAGE_SHAPE_HWC",
  "FTP1_MODEL_ACTION_DIM",
  "FTP1_PHYSICAL_ACTION_DIM",
  "FTP1_TACTILE_FUNCTION_AREAS",
  "FTP1_TACTILE_KEY",
  "FTP1_TACTILE_RAW_SHAPE",
  "FTP1_TACTILE_SENSOR",
  "FTP1_TACTILE_SHAPE",
  "RIGHT_ARM_JOINT_NAMES",
  "RIGHT_HAND_FAAS_INDICES",
  "RIGHT_HAND_JOINT_NAMES",
  "RIGHT_JOINT_NAMES",
  "RIGHT_TACTILE_LINK_NAMES",
  "decode_ftp1_actions",
  "encode_ftp1_state",
  "ftp1_action_mask",
  "pack_ftp1_tactile",
  "validate_world_transform",
]
