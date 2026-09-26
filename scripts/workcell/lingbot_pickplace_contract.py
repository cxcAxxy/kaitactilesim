"""Frozen, LingBot-only KaiHand PickPlace inference contract.

This module intentionally has no model or simulator imports so the deployment
preflight can run before CUDA-dependent LingBot code is imported.
"""

from __future__ import annotations

ACTION_DIM = 27
MODEL_ACTION_DIM = 55
CONTROL_HZ = 30
HORIZON = 50
TASK = "pick-place"
MODEL_FAMILY = "LingBot-VLA-2.0"
DEPLOYMENT_SCHEMA = "pickplace_lingbot_vla2_deployment_v1"
SCHEMA = DEPLOYMENT_SCHEMA
INSTRUCTION = (
  "Pick up the cylinder with the right hand, move it into the target box, "
  "release it, then withdraw the hand."
)

RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint{index}" for index in range(1, 8))
RIGHT_HAND_JOINT_NAMES = (
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
RIGHT_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
JOINT_NAMES = RIGHT_JOINT_NAMES

OBSERVATION_CONTRACT = {
  "cameras": ["head", "right_wrist"],
  "image_shape_hwc": [240, 320, 3],
  "image_dtype": "uint8",
  "state_dim": ACTION_DIM,
  "tactile_sent_to_model": False,
}
ACTION_REPRESENTATION = {
  "training_right_arm": "relative_joint_position_delta",
  "server_right_arm": "absolute_joint_position_target",
  "right_hand": "absolute_joint_position_target",
  "execution": "direct_30hz_joint_position_targets",
}
