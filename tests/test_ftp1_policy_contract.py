"""FTP-1 KaiHand codec stays identical to the frozen training contract."""

from __future__ import annotations

import numpy as np
import pytest
from kaihand_tactile_env.shared.ftp1_policy_contract import (
  FTP1_ACTION_HORIZON,
  FTP1_MODEL_ACTION_DIM,
  RIGHT_HAND_FAAS_INDICES,
  decode_ftp1_actions,
  encode_ftp1_state,
  ftp1_action_mask,
  pack_ftp1_tactile,
  validate_world_transform,
)


def transform(*, xyz=(0.0, 0.0, 0.0), yaw_rad=0.0):
  cosine = np.cos(yaw_rad)
  sine = np.sin(yaw_rad)
  value = np.eye(4, dtype=np.float64)
  value[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
  value[:3, 3] = xyz
  return value


def identity_pose_actions():
  actions = np.zeros((FTP1_ACTION_HORIZON, FTP1_MODEL_ACTION_DIM), dtype=np.float32)
  actions[:, 3] = 1.0
  actions[:, 7] = 1.0
  return actions


def test_encode_state_uses_absolute_joints_and_faas_slots():
  joints = np.arange(27, dtype=np.float32) / 10.0
  encoded = encode_ftp1_state(
    joints,
    transform(xyz=(0.2, -0.1, 1.4), yaw_rad=0.2),
    transform(xyz=(0.5, 0.3, 1.0), yaw_rad=-0.4),
  )

  assert encoded.shape == (1, 120)
  assert encoded.dtype == np.float32
  np.testing.assert_allclose(
    encoded[0, :9],
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    atol=1.0e-6,
  )
  np.testing.assert_array_equal(encoded[0, 9:16], joints[:7])
  np.testing.assert_array_equal(
    encoded[0, 16 + np.asarray(RIGHT_HAND_FAAS_INDICES)], joints[7:]
  )
  expected_nonzero = set(range(9, 16)) | {
    16 + index for index in RIGHT_HAND_FAAS_INDICES
  }
  for index in range(9, 120):
    if index not in expected_nonzero:
      assert encoded[0, index] == 0.0


def test_action_mask_only_enables_trained_kaihand_dimensions():
  mask = ftp1_action_mask()
  assert mask.shape == (120,)
  assert mask.dtype == np.float32
  assert int(mask.sum()) == 9 + 7 + 20
  np.testing.assert_array_equal(mask[:16], 1.0)
  np.testing.assert_array_equal(
    mask[16 + np.asarray(RIGHT_HAND_FAAS_INDICES)], 1.0
  )
  assert mask[16 + 28] == 0.0
  batched = ftp1_action_mask(horizon=32)
  assert batched.shape == (32, 120)
  np.testing.assert_array_equal(batched[17], mask)


def test_pack_tactile_transposes_channels_last_without_normalization():
  raw = np.arange(5 * 3 * 7 * 5, dtype=np.float32).reshape(5, 3, 7, 5)
  packed = pack_ftp1_tactile(raw)
  assert packed.shape == (1, 5, 7, 5, 3)
  assert packed.dtype == np.float32
  np.testing.assert_array_equal(packed[0, 3, :, :, 0], raw[3, 0])
  np.testing.assert_array_equal(packed[0, 3, :, :, 1], raw[3, 1])
  np.testing.assert_array_equal(packed[0, 3, :, :, 2], raw[3, 2])

  raw[0, 0, 0, 0] = -5.0e-8
  assert pack_ftp1_tactile(raw)[0, 0, 0, 0, 0] == 0.0
  raw[0, 0, 0, 0] = -1.0e-3
  with pytest.raises(ValueError, match="normal force"):
    pack_ftp1_tactile(raw)


def test_decode_mix_actions_returns_absolute_physical_joints_and_world_wrist():
  joints = np.linspace(-0.5, 0.5, 27, dtype=np.float32)
  head = transform(xyz=(0.2, 0.1, 1.2), yaw_rad=0.3)
  wrist = transform(xyz=(0.6, -0.2, 0.8), yaw_rad=-0.1)
  actions = identity_pose_actions()
  actions[:, 9:16] = 0.25
  actions[:, 16 + np.asarray(RIGHT_HAND_FAAS_INDICES)] = -0.125
  # Untrained blocks may be nonzero after diffusion sampling; they must not
  # influence either physical joint targets or wrist decoding.
  actions[:, 48:] = 999.0

  decoded = decode_ftp1_actions(actions, joints, head, wrist)

  assert decoded.joint_actions.shape == (32, 27)
  assert decoded.joint_actions.dtype == np.float32
  np.testing.assert_allclose(
    decoded.joint_actions[:, :7],
    np.broadcast_to(joints[:7] + 0.25, (32, 7)),
  )
  np.testing.assert_allclose(
    decoded.joint_actions[:, 7:],
    np.broadcast_to(joints[7:] - 0.125, (32, 20)),
  )
  np.testing.assert_allclose(
    decoded.wrist_world_transforms,
    np.broadcast_to(wrist, (32, 4, 4)),
    atol=1.0e-6,
  )


def test_decode_arm_relative_hand_absolute_preserves_absolute_hand_targets():
  joints = np.linspace(-0.5, 0.5, 27, dtype=np.float32)
  actions = identity_pose_actions()
  actions[:, 9:16] = 0.25
  hand_slots = 16 + np.asarray(RIGHT_HAND_FAAS_INDICES)
  actions[:, hand_slots] = np.linspace(-0.2, 0.2, 20, dtype=np.float32)

  decoded = decode_ftp1_actions(
    actions,
    joints,
    np.eye(4),
    np.eye(4),
    action_joint_rep="arm_relative_hand_absolute",
  )

  np.testing.assert_allclose(
    decoded.joint_actions[:, :7],
    np.broadcast_to(joints[:7] + 0.25, (32, 7)),
  )
  np.testing.assert_allclose(
    decoded.joint_actions[:, 7:],
    np.broadcast_to(actions[0, hand_slots], (32, 20)),
  )


def test_decode_relative_wrist_uses_current_wrist_as_pose_base():
  joints = np.zeros(27, dtype=np.float32)
  head = transform(xyz=(10.0, -4.0, 2.0), yaw_rad=0.7)
  wrist = transform(xyz=(10.3, -3.8, 1.5), yaw_rad=-0.2)
  actions = identity_pose_actions()
  actions[:, :3] = (0.1, -0.2, 0.05)

  decoded = decode_ftp1_actions(actions, joints, head, wrist)
  expected = wrist.copy()
  expected[:3, 3] += wrist[:3, :3] @ np.asarray((0.1, -0.2, 0.05))
  np.testing.assert_allclose(decoded.wrist_world_transforms[5], expected, atol=1.0e-6)


def test_contract_rejects_bad_shapes_nonfinite_and_degenerate_rotations():
  with pytest.raises(ValueError, match="shape"):
    encode_ftp1_state(np.zeros(26), np.eye(4), np.eye(4))
  malformed = np.eye(4)
  malformed[3, 0] = 1.0
  with pytest.raises(ValueError, match="homogeneous"):
    validate_world_transform("bad", malformed)
  with pytest.raises(ValueError, match="finite"):
    pack_ftp1_tactile(np.full((5, 3, 7, 5), np.nan))
  actions = np.zeros((32, 120), dtype=np.float32)
  with pytest.raises(ValueError, match="degenerate"):
    decode_ftp1_actions(actions, np.zeros(27), np.eye(4), np.eye(4))
  with pytest.raises(ValueError, match="unsupported FTP-1 action_joint_rep"):
    decode_ftp1_actions(
      identity_pose_actions(),
      np.zeros(27),
      np.eye(4),
      np.eye(4),
      action_joint_rep="absolute",
    )
