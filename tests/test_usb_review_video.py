"""USB offline replay video layout and tactile value checks."""

from __future__ import annotations

import numpy as np
from kaihand_tactile_env.shared.usb_review_video import _force_means


def test_force_means_preserve_signed_tangent_and_average_taxels():
  normal = np.zeros((10, 7, 5))
  tangent = np.zeros((10, 7, 5, 2))
  normal[3] = 0.03
  tangent[3, ..., 0] = -0.02
  tangent[3, ..., 1] = 0.01

  means = _force_means(normal, tangent)

  assert means.shape == (10, 3)
  np.testing.assert_allclose(means[3], [-0.02, 0.01, 0.03])
  np.testing.assert_array_equal(means[[0, 1, 2, 4, 5, 6, 7, 8, 9]], 0)
