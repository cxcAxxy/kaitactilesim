"""Sponge pi0.5 timing follows its 100 Hz control-tick camera scheduler."""

import sys
from pathlib import Path

import numpy as np
import pytest

WORKCELL = Path(__file__).parents[1] / "scripts/workcell"
sys.path.insert(0, str(WORKCELL))

from convert_shared_to_lerobot import _sponge_control_tick_prefix  # noqa: E402


def test_sponge_camera_deadlines_are_quantized_to_control_ticks():
  frame = np.arange(682)
  ticks = (frame * 100 + 29) // 30
  timestamps = ticks / 100.0
  prefix, maximum_error = _sponge_control_tick_prefix(timestamps, 4000)
  assert prefix == len(timestamps)
  assert maximum_error == 0
  np.testing.assert_allclose(timestamps[:5], [0, 0.04, 0.07, 0.10, 0.14])


def test_sponge_camera_rejects_an_internal_off_grid_frame():
  timestamps = np.array([0.0, 0.04, 0.07, 0.11, 0.14])
  prefix, _ = _sponge_control_tick_prefix(timestamps, 4000)
  assert prefix == 3


def test_sponge_camera_rejects_missing_first_deadline():
  with pytest.raises(ValueError, match="no two-frame"):
    _sponge_control_tick_prefix(np.array([0.0, 0.03, 0.07]), 4000)
