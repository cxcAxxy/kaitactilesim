"""Unequal RGB/state rates are causal, not forced to share timestamps."""

import numpy as np
from kaihand_tactile_env.shared.tict_source_audit import _audit_camera_state_clock


def test_30hz_camera_between_100hz_states_is_valid():
  state = np.arange(0, 101, 10, dtype=np.int64) * 1_000_000
  force = np.maximum(state - 2_000_000, 0)
  capture = np.array([0, 34, 68, 100]) * 1_000_000
  render = np.maximum(capture - 2_000_000, 0)
  errors = []
  _audit_camera_state_clock(state, force, capture, render, np.array([0, 3, 6, 10]), errors)
  assert not errors


def test_camera_link_cannot_silently_skip_a_newer_state():
  errors = []
  _audit_camera_state_clock(
    np.array([0, 10, 20]), np.array([0, 8, 18]),
    np.array([0, 24]), np.array([0, 22]), np.array([0, 1]), errors,
  )
  assert errors == ["camera state_index must identify the latest nonfuture state"]


def test_camera_cannot_refer_to_a_future_solver_cache():
  errors = []
  _audit_camera_state_clock(
    np.array([0, 10, 20]), np.array([0, 8, 18]),
    np.array([0, 20]), np.array([0, 17]), np.array([0, 2]), errors,
  )
  assert errors == ["indexed state solver epoch is later than the rendered image"]
