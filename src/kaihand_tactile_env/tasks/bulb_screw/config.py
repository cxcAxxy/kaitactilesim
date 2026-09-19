"""Metre/radian benchmark dimensions, not an electrical or E27 standard model."""

import numpy as np

from ...shared.posture import ARM_HOME as ARM_HOME

SCENE_NAME = "bulb-screw"
OBJECT_NAMES = ("bulb",)
OBJECT_STABILIZERS: dict[tuple[str, str], str] = {}
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_XY_JITTER = 0.0
DEFAULT_YAW_JITTER = 0.0
TABLETOP_HEIGHT_M = 0.680
BULB_INITIAL_POSITION_M = np.array([0.500, -0.180, 0.6805])
SOCKET_MOUTH_POSITION_M = np.array([0.620, -0.180, 0.736])
THREAD_ENTRY_POSITION_M = np.array([0.620, -0.180, 0.730])
THREAD_ENTRY_DEPTH_M = 0.006
THREAD_CAPTURE_LOAD_N = 0.02
MECHANICS_DEMO_TORQUE_NM = 0.06
MAX_THREAD_ERROR_M = 0.0001
THREAD_PITCH_M = 0.004
THREAD_TRAVEL_M = THREAD_PITCH_M * 100 / 360
TARGET_TURNS = THREAD_TRAVEL_M / THREAD_PITCH_M
SEATED_DEPTH_M = THREAD_ENTRY_DEPTH_M + THREAD_TRAVEL_M
CAPTURE_LATERAL_M = 0.0004
CAPTURE_DEPTH_M = 0.0003
CAPTURE_TILT_RAD = np.deg2rad(2.0)
CAPTURE_SPEED_M_S = 0.1
SEATED_DEPTH_TOLERANCE_M = 0.0003
SEATED_DWELL_S = 0.2
EXTERNAL_THREAD_SEGMENTS = 32
INTERNAL_THREAD_SEGMENTS = 48
SEATED_SHOULDER_GAP_M = 0.0001
MECHANICS_VERSION = "short_thread_finger_tightening_v11"
ROTATION_DRIVER = "finger_gait_fixed_wrist_v3"
FINGER_STROKE_TURNS = 30 / 360
TIGHTENING_DURATION_S = 1.2
TIGHTENING_COMMAND_RAD = np.deg2rad(18.0)
TIGHTENING_EFFORT_HOLD_S = 0.4
TIGHTENING_LOAD_RECOVERY_S = 2.0
TIGHTENING_LIGHT_HOLD_S = 0.4
TIGHTENING_HOLD_S = 0.3
TIGHTENING_MIN_TORQUE_NM = 0.1
TIGHTENING_MAX_ROTATION_RAD = np.deg2rad(0.3)
SCENE_GEOM_NAMES = (
  "bulb_globe",
  "bulb_neck",
  "bulb_screw_base",
  "bulb_tip_contact",
  "bulb_orientation_mark",
  "bulb_fixture_base",
  "bulb_fixture_pedestal",
  "bulb_socket_backstop",
  *(f"bulb_socket_wall_{i:02d}" for i in range(16)),
  *(f"bulb_socket_cushion_{i:02d}" for i in range(16)),
  *(f"bulb_socket_thread_{i:03d}" for i in range(INTERNAL_THREAD_SEGMENTS)),
  *(f"bulb_thread_crest_{i:03d}" for i in range(EXTERNAL_THREAD_SEGMENTS)),
)
