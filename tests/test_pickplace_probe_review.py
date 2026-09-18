"""Pick-place Genesis proxy review is explicit about source and dimensions."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.pickplace_probe_review import (
  _check_source,
  _probe_indices,
  compose_probe_frame,
)


def test_probe_layout_and_causal_camera_state_alignment(tmp_path):
  path = tmp_path / "pickplace.h5"
  with h5py.File(path, "w") as file:
    file.create_dataset("state/timestamp", data=np.array([0.0, 0.01, 0.02]))
    camera = file.create_group("cameras/head")
    camera.create_dataset("timestamp", data=[0.0, 0.024])
    camera.create_dataset("state_index", data=[0, 2])
    camera.create_dataset("rgb", data=np.zeros((2, 240, 320, 3), dtype=np.uint8))
    force = file.create_group("tactile_genesis")
    force.create_dataset(
      "link_names", data=list(FINGERTIP_LINK_NAMES), dtype=h5py.string_dtype()
    )
    force.create_dataset(
      "probe_link_names",
      data=[name for name in FINGERTIP_LINK_NAMES for _ in range(35)],
      dtype=h5py.string_dtype(),
    )
    force.create_dataset("probe_depth", data=np.zeros((3, 350)))
    force.create_dataset("probe_contact_instantaneous", data=np.zeros((3, 350)))
    force.create_dataset("force_local", data=np.zeros((3, 10, 3)))
  with h5py.File(path) as file:
    _, state_indices, _ = _check_source(file)
    np.testing.assert_array_equal(state_indices, [0, 2])
    np.testing.assert_array_equal(_probe_indices(file)[5], np.arange(175, 210))


def test_probe_compositor_keeps_proxy_labels_and_rejects_wrong_shape():
  rgb = np.zeros((240, 320, 3), dtype=np.uint8)
  depth = np.zeros((10, 7, 5))
  contact = np.zeros((10, 7, 5))
  history = np.zeros((2, 10, 3))
  history[1, 5] = [12, -3, 17]
  image = compose_probe_frame(
    rgb,
    depth,
    contact,
    history,
    width=1920,
    height=1080,
    depth_max_mm=4,
    proxy_abs_max=100,
    timestamp_s=1.0,
    phase="grasp",
  )
  assert image.size == (1920, 1080)
  with pytest.raises(ValueError, match="probe maps"):
    compose_probe_frame(
      rgb,
      depth[:5],
      contact,
      history,
      width=1920,
      height=1080,
      depth_max_mm=4,
      proxy_abs_max=100,
      timestamp_s=1.0,
      phase="grasp",
    )
