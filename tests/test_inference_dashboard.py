from __future__ import annotations

import numpy as np
from kaihand_tactile_env.workcell.inference_dashboard import (
  fingertip_grid_indices,
)
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation
from kaihand_tactile_env.workcell.tactile import GenesisProbeTactileProvider


def test_fingertip_dashboard_covers_bilateral_layout_exactly_once() -> None:
  simulation = ArmHandSimulation(scene="pick-place")
  tactile = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )

  grids = fingertip_grid_indices(tactile)

  assert len(grids) == 10
  assert all(indices.shape == (35,) for indices in grids.values())
  combined = np.concatenate(tuple(grids.values()))
  np.testing.assert_array_equal(np.sort(combined), np.arange(350))
  assert tuple(grids) == (
    "hand_l_thumb_link6",
    "hand_l_index_link4",
    "hand_l_middle_link4",
    "hand_l_ring_link4",
    "hand_l_pinky_link4",
    "hand_r_thumb_link6",
    "hand_r_index_link4",
    "hand_r_middle_link4",
    "hand_r_ring_link4",
    "hand_r_pinky_link4",
  )
