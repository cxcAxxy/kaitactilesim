"""Seeded sponge pickup poses stay bounded and reproducible."""

import numpy as np
import pytest
from kaihand_tactile_env.shared.policy_tasks import TaskPolicyAdapter
from kaihand_tactile_env.tasks.sponge_grasp import config
from kaihand_tactile_env.tasks.sponge_grasp.task import SpongeGraspSimulation


def test_seeded_sponge_xy_is_reproducible_and_moves_the_whole_flex():
  sim = SpongeGraspSimulation(
    position_seed=2264256753,
    randomize_xy=True,
    add_genesis_probes=False,
  )
  first_xy = sim.sponge_initial_xy_m.copy()
  first_vertices = sim.data.flexvert_xpos[
    sim._flex_start : sim._flex_start + sim._flex_count
  ].copy()
  assert np.all(sim.sponge_xy_offset_m >= config.COLLECTION_XY_OFFSET_LOW_M)
  assert np.all(sim.sponge_xy_offset_m <= config.COLLECTION_XY_OFFSET_HIGH_M)
  assert np.any(sim.sponge_xy_offset_m != 0)

  sim.reset()
  np.testing.assert_array_equal(sim.sponge_initial_xy_m, first_xy)
  np.testing.assert_allclose(
    sim.data.flexvert_xpos[sim._flex_start : sim._flex_start + sim._flex_count],
    first_vertices,
    atol=1e-12,
  )

  sim.position_seed = 3356062397
  sim.reset()
  assert not np.array_equal(sim.sponge_initial_xy_m, first_xy)
  np.testing.assert_allclose(
    sim.data.flexvert_xpos[
      sim._flex_start : sim._flex_start + sim._flex_count, :2
    ] - first_vertices[:, :2],
    np.broadcast_to(sim.sponge_initial_xy_m - first_xy, first_vertices[:, :2].shape),
    atol=1e-12,
  )


def test_unseeded_sponge_stays_at_calibrated_pose():
  sim = SpongeGraspSimulation(add_genesis_probes=False)
  np.testing.assert_array_equal(sim.sponge_initial_xy_m, config.SPONGE_TABLE_XY)
  np.testing.assert_array_equal(sim.sponge_xy_offset_m, np.zeros(2))


@pytest.mark.parametrize(
  "kwargs",
  (
    {"position_seed": -1},
    {"randomize_xy": True},
  ),
)
def test_sponge_rejects_invalid_randomization(kwargs):
  with pytest.raises(ValueError):
    SpongeGraspSimulation(**kwargs)


def test_policy_adapter_uses_seeded_collection_reset_and_strict_initial_state():
  adapter = TaskPolicyAdapter("sponge-grasp", 17)
  expected = np.random.default_rng(17).uniform(
    config.COLLECTION_XY_OFFSET_LOW_M,
    config.COLLECTION_XY_OFFSET_HIGH_M,
  )

  assert adapter.simulation.randomize_xy is True
  assert np.allclose(adapter.simulation.sponge_xy_offset_m, expected)
  assert adapter.instruction == (
    "Pick up the upright sponge with the right hand, carry it to the plate on "
    "the robot's right, place it inside the plate, and release it."
  )
  assert adapter.success is False
  assert adapter.stage() == "pick_sponge"
  assert adapter.state.grasp_established is False
  assert adapter.state.lift_established is False
  assert adapter.state.release_allowed is False
  assert adapter.state.released is False
  assert adapter.simulation.grasp_patch.enabled is False
