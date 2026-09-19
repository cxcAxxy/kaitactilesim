"""Seeded stain geometry isolation and friction-work requirements."""

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.tasks.vase_wipe import config
from kaihand_tactile_env.tasks.vase_wipe.cleaning import CleaningProgress
from kaihand_tactile_env.tasks.vase_wipe.sponge import points
from kaihand_tactile_env.tasks.vase_wipe.task import (
  VaseWipeSimulation,
  _external_flex_contact_mask,
)


def test_seeded_layout_reset_and_contact_isolation():
  sim = VaseWipeSimulation()
  ids = sim._stain_ids
  nominal = sim.model.geom_pos.copy()
  sizes = sim.model.geom_size.copy()
  friction = sim.model.geom_friction.copy()
  contact = sim.model.geom_contype.copy()
  other = np.ones(sim.model.ngeom, dtype=bool)
  other[ids] = False
  sim.stain_seed = 42
  sim.reset()
  randomized = sim.model.geom_pos.copy()
  first = sim.stain_randomization
  assert not np.array_equal(randomized[ids], nominal[ids])
  sim.reset()
  assert sim.stain_randomization == first
  np.testing.assert_array_equal(sim.model.geom_pos, randomized)
  np.testing.assert_array_equal(sim.model.geom_pos[other], nominal[other])
  np.testing.assert_array_equal(sim.model.geom_size[other], sizes[other])
  np.testing.assert_array_equal(sim.model.geom_friction, friction)
  np.testing.assert_array_equal(sim.model.geom_contype, contact)
  for seed in range(100):
    area, metadata = sim._stain_layout.apply(sim.model, seed)
    pos = sim.model.geom_pos[ids]
    assert np.all((area >= 0.81) & (area <= 1.21))
    assert pos[:, 2].min() >= min(config.STAIN_HEIGHTS) - 0.0018
    assert pos[:, 2].max() <= max(config.STAIN_HEIGHTS) + 0.0018
    np.testing.assert_allclose(
      np.linalg.norm(pos[:, :2], axis=1),
      [config.inner_radius(z) - 0.0002 for z in pos[:, 2]],
    )
    assert np.abs(np.arctan2(pos[:, 1], pos[:, 0])).max() < 0.22
    assert metadata["seed"] == seed
  sim.stain_seed = None
  sim.reset()
  np.testing.assert_array_equal(sim.model.geom_pos, nominal)
  np.testing.assert_array_equal(sim.model.geom_size, sizes)
  mujoco.mj_forward(sim.model, sim.data)
  np.testing.assert_array_equal(sim.cleaning.area_scale, np.ones(21))


def test_size_changes_work_but_touch_alone_never_cleans():
  cleaning = CleaningProgress(2, area_scale=[0.81, 1.21])
  for _ in range(100):
    cleaning.update(1, 1, np.ones(2), np.zeros(2), np.zeros(2), 0.01)
  np.testing.assert_array_equal(cleaning.remaining, [1, 1])
  # Identical real sliding dose: smaller stain reaches completion sooner.
  for _ in range(14):
    cleaning.update(1, 1, np.ones(2), np.full(2, 0.00032), np.full(2, 0.012), 0.01)
  assert cleaning.remaining[0] < 1e-12
  assert 0 < cleaning.remaining[1] < 1
  assert (
    cleaning.report()["required_work_j"][0] < cleaning.report()["required_work_j"][1]
  )
  with pytest.raises(ValueError):
    CleaningProgress(2, area_scale=[1, float("nan")])


def test_requested_appearance_geometry_and_short_pickup_path():
  np.testing.assert_allclose(np.ptp(points(), axis=0), [0.045, 0.062, 0.110])
  sim = VaseWipeSimulation()
  stain_rgba = sim.model.geom_rgba[sim._stain_ids]
  np.testing.assert_allclose(
    stain_rgba,
    np.tile(config.STAIN_COLOR, (config.PATCH_ROWS * config.PATCH_COLUMNS, 1)),
  )
  pickup_delta = sim.approach_arm - config.ARM_HOME["right"]
  assert pickup_delta[4] < 0
  assert np.max(np.abs(pickup_delta)) < np.deg2rad(100)


def test_contact_audit_excludes_only_internal_tetrahedron_constraints():
  geom = np.array(
    [
      [-1, -1],  # Internal vertex-to-opposite-face constraint.
      [-1, -1],  # True flex surface self-contact.
      [283, -1],  # Flex against vase wall.
      [12, -1],  # Flex against hand or table.
    ]
  )
  elem = np.array([[44, -1], [44, 91], [-1, 33], [-1, 18]])
  np.testing.assert_array_equal(
    _external_flex_contact_mask(geom, elem), [False, True, True, True]
  )
