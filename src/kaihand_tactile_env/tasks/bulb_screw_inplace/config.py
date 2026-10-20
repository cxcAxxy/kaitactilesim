"""Independent reset/task contract using the unchanged v15 bulb mechanics."""

from __future__ import annotations

import numpy as np

from ..bulb_screw import config as screw

# Inherit numerical mechanics, not the tabletop pickup scene or its task name.
# Explicitly copying the uppercase contract keeps existing recording tools
# compatible without changing the source task's globals or numerical arrays.
for _name in dir(screw):
  if _name.isupper():
    _value = getattr(screw, _name)
    globals()[_name] = _value.copy() if isinstance(_value, np.ndarray) else _value

SCENE_NAME = "bulb-screw-inplace"
SCENE_GEOM_NAMES = tuple(
  name for name in screw.SCENE_GEOM_NAMES if not name.startswith("bulb_pickup_cradle_")
)
BULB_INITIAL_POSITION_M = screw.THREAD_ENTRY_POSITION_M.copy()
MECHANICS_VERSION = "bulb_screw_inplace_inherits_phase_aware_strict_completion_v15_v1"
ROTATION_DRIVER = screw.ROTATION_DRIVER

# This is an explicitly reset-only preset. Neither gravity compensation nor
# any other support is added to the free bulb during the episode.
THREADED_SETTLE_S = 0.30
INITIAL_MAX_ROTATION_RAD = np.deg2rad(8.0)
INITIAL_MAX_DEPTH_CHANGE_M = 0.00015
SETTLED_LINEAR_SPEED_M_S = 0.001
SETTLED_ANGULAR_SPEED_RAD_S = 0.05
GRASP_MIN_FORCE_N = 0.025
GRASP_STABLE_DWELL_S = 0.10
EFFECTIVE_ROTATION_MIN_TURNS = 30 / 360
EFFECTIVE_DEPTH_MIN_M = screw.THREAD_PITCH_M * EFFECTIVE_ROTATION_MIN_TURNS * 0.65
RELEASE_DWELL_S = screw.SEATED_DWELL_S
HOME_DWELL_S = 0.10
ARM_HOME_TOLERANCE_RAD = np.deg2rad(1.0)
HAND_HOME_TOLERANCE_RAD = np.deg2rad(4.0)
HOME_SPEED_RAD_S = np.deg2rad(2.0)

__all__ = [name for name in globals() if name.isupper()]
