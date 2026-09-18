from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

_CONVERTER = run_path(
  str(
    Path(__file__).parents[1]
    / "scripts"
    / "workcell"
    / "convert_to_egosteer.py"
  )
)
CV_FROM_MUJOCO_CAMERA = _CONVERTER["CV_FROM_MUJOCO_CAMERA"]
SITE_FROM_EGOSTEER_WRIST = _CONVERTER["SITE_FROM_EGOSTEER_WRIST"]
_camera_from_world_cv = _CONVERTER["_camera_from_world_cv"]
_rot6d = _CONVERTER["_rot6d"]
_strict_30hz_prefix = _CONVERTER["_strict_30hz_prefix"]


def test_rot6d_uses_first_two_rotation_columns() -> None:
  rotation = np.array(
    (
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0),
      (7.0, 8.0, 9.0),
    )
  )

  np.testing.assert_array_equal(
    _rot6d(rotation),
    np.array((1.0, 4.0, 7.0, 2.0, 5.0, 8.0)),
  )


def test_camera_extrinsic_converts_mujoco_to_opencv_rdf() -> None:
  world_from_camera = np.eye(4)
  world_from_camera[:3, 3] = (0.25, -0.5, 1.0)

  expected = CV_FROM_MUJOCO_CAMERA @ np.linalg.inv(world_from_camera)
  np.testing.assert_allclose(
    _camera_from_world_cv(world_from_camera), expected, atol=1.0e-12
  )


def test_strict_30hz_prefix_accepts_physics_clock_quantization() -> None:
  frame_indices = np.arange(8)
  timestamps = np.round((frame_indices / 30.0) * 500.0) / 500.0
  timestamps = np.concatenate((timestamps, (timestamps[-1] + 0.01,)))

  prefix, maximum_error = _strict_30hz_prefix(timestamps, physics_hz=500)

  assert prefix == 8
  assert maximum_error == pytest.approx(2.0 / 3000.0)


@pytest.mark.parametrize("side", ("left", "right"))
def test_canonical_wrist_mapping_is_a_proper_rotation(side: str) -> None:
  rotation = SITE_FROM_EGOSTEER_WRIST[side]

  np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
  assert np.linalg.det(rotation) == pytest.approx(1.0)
