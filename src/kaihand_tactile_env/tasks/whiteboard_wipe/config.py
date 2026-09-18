"""Task-local geometry and contact-driven ink removal, in SI units."""

import numpy as np

SCENE_NAME = "whiteboard-wipe"
OBJECT_NAMES = ("eraser",)
OBJECT_STABILIZERS = {}
ARM_HOME = {
  "left": np.deg2rad([55, -65, -70, -60, -120, 0, 0]),
  "right": np.deg2rad([-55, -65, 70, -60, 120, 0, 0]),
}
DEFAULT_PLAYBACK_SPEED = 1.0
CONTROL_PERIOD_S = 0.01
HAND_VELOCITY_GAIN = 1.0
DEFAULT_XY_JITTER = DEFAULT_YAW_JITTER = 0.0
SCENE_GEOM_NAMES = ("board_surface", "eraser_handle", "eraser_pad")
TABLE_HEIGHT = 0.68
BOARD_CENTER = np.array([0.62, -0.12, 0.825])
BOARD_ROTATION = np.array([[2**-0.5, 0, -(2**-0.5)], [0, 1, 0], [2**-0.5, 0, 2**-0.5]])
BOARD_NORMAL = BOARD_ROTATION[:, 2]
BOARD_SURFACE = BOARD_CENTER + 0.008 * BOARD_NORMAL
PAD_BOTTOM = np.array([0.0, 0.0, -0.015])
INK_COUNT = 25
# Ink center translation in the board plane; keep the frame outside the stroke.
INK_CENTER_RANGE_M = np.array([0.04, 0.06])
MIN_NORMAL_FORCE = 1.5
MAX_NORMAL_FORCE = 8.0
MIN_SLIDING_SPEED = 0.005
REQUIRED_SLIDING_DISTANCE = 0.03

INITIAL_ERASER_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
WIPE_ROTATION = BOARD_ROTATION @ INITIAL_ERASER_ROTATION
