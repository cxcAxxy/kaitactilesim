"""Small-box and pure-array tests; no full robot or rendering."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.randomization import (
  MAX_YAW_JITTER_RAD,
  offset_card_pose,
  reset_randomized_card,
  sample_card_offsets,
  validate_randomization_bounds,
  validate_recorded_randomization,
)


def _tiny_sim():
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="poker_table_top" type="box" pos=".58 -.18 .8405" size=".115 .07 .0005"/>
    <body name="card" pos=".58 -.16 .84175"><freejoint/>
    <geom name="card_core_geom" type="box" size=".0315 .044 .00055" mass=".004"/>
    </body></worldbody></mujoco>""")
  data = mujoco.MjData(model)
  sim = SimpleNamespace(
    model=model,
    data=data,
    scene="poker-draw",
    _initial_object_pose={"card": model.qpos0.copy()},
    _observation_time=99.0,
    calls=0,
  )

  def set_pose(name, position, quaternion):
    assert name == "card"
    data.qpos[:] = [*position, *quaternion]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

  def reset(*, seed, object_xy_jitter, object_yaw_jitter):
    sim.calls += 1
    mujoco.mj_resetData(model, data)
    rng = np.random.default_rng(seed)
    pose = model.qpos0.copy()
    if object_xy_jitter:
      pose[:2] += rng.uniform(-object_xy_jitter, object_xy_jitter, size=2)
    if object_yaw_jitter:
      yaw = rng.uniform(-object_yaw_jitter, object_yaw_jitter)
      q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
      mujoco.mju_mulQuat(pose[3:], q, model.qpos0[3:])
    set_pose("card", pose[:3], pose[3:])

  sim.reset = reset
  sim.set_object_pose = set_pose
  sim.object_pose = lambda name: np.concatenate(
    (data.body(name).xpos, data.body(name).xquat)
  )
  return sim


@pytest.mark.parametrize(
  "xy,yaw", [(0, 0), (0.002, 0), (0, 0.01), (0.005, MAX_YAW_JITTER_RAD)]
)
def test_rng_matches_shared_order_and_reset_is_reproducible(xy, yaw):
  a, b = _tiny_sim(), _tiny_sim()
  meta = reset_randomized_card(a, 3, xy, yaw)
  assert meta == reset_randomized_card(b, 3, xy, yaw)
  assert a._observation_time == 0
  assert a.calls == 1
  assert validate_recorded_randomization(meta, 3, xy, yaw)
  assert meta["sampled_pose_wxyz"][2] == meta["nominal_pose_wxyz"][2]
  assert meta["table_support"]["minimum_corner_margin_xy_m"] > 0.00025


@pytest.mark.parametrize(
  "xy,yaw",
  [
    (-0.001, 0),
    (0, -0.001),
    (0.00501, 0),
    (0, MAX_YAW_JITTER_RAD + 1e-8),
    (float("nan"), 0),
    (0, float("inf")),
  ],
)
def test_out_of_scope_bounds_rejected(xy, yaw):
  with pytest.raises(ValueError):
    validate_randomization_bounds(xy, yaw)


@pytest.mark.parametrize("seed", [-1, True, 1.1])
def test_invalid_seed_rejected(seed):
  with pytest.raises(ValueError):
    sample_card_offsets(seed, 0.002, 0)


def test_zero_xy_does_not_consume_two_random_values():
  xy, yaw = sample_card_offsets(17, 0, 0.01)
  np.testing.assert_array_equal(xy, [0, 0])
  assert yaw == np.random.default_rng(17).uniform(-0.01, 0.01)


@pytest.mark.parametrize(
  "key",
  [
    "seed",
    "sampled_offset_xy_m",
    "sampled_yaw_offset_rad",
    "sampled_pose_wxyz",
    "observation_noise",
    "action_noise",
  ],
)
def test_missing_metadata_rejected(key):
  meta = reset_randomized_card(_tiny_sim(), 0, 0.002, 0.01)
  del meta[key]
  assert not validate_recorded_randomization(meta, 0, 0.002, 0.01)


def test_random_samples_are_not_duplicates():
  poses = [
    reset_randomized_card(_tiny_sim(), seed, 0.002, 0.01)["sampled_pose_wxyz"]
    for seed in (0, 1, 2)
  ]
  assert len({tuple(p) for p in poses}) == 3


def test_table_corner_guard_rejects_bad_initial_support_without_resampling():
  sim = _tiny_sim()
  sim._initial_object_pose["card"][1] += 0.008
  sim.model.qpos0[1] += 0.008
  with pytest.raises(ValueError, match="fully supported"):
    reset_randomized_card(sim, 0, 0, 0)
  assert sim.calls == 1


def test_fixed_boundary_is_explicit_and_cannot_masquerade_as_random_dataset():
  sim = _tiny_sim()
  meta = reset_randomized_card(
    sim, 0, 0.005, MAX_YAW_JITTER_RAD, fixed_offset=(0.005, 0.005, MAX_YAW_JITTER_RAD)
  )
  assert meta["mode"] == "fixed_validation_offset"
  np.testing.assert_allclose(meta["sampled_offset_xy_m"], [0.005, 0.005])
  assert not validate_recorded_randomization(meta, 0, 0.005, MAX_YAW_JITTER_RAD)
  with pytest.raises(ValueError, match="outside"):
    reset_randomized_card(sim, 0, 0.002, 0, fixed_offset=(0.003, 0, 0))


def test_yaw_is_world_left_multiplication_and_preserves_height():
  nominal = np.array([0.58, -0.16, 0.84175, 0.92387953, 0.38268343, 0, 0])
  nominal[3:] /= np.linalg.norm(nominal[3:])
  result = offset_card_pose(nominal, [0.001, -0.002], 0.01)
  expected = np.empty(4)
  mujoco.mju_mulQuat(
    expected, np.array([np.cos(0.005), 0, 0, np.sin(0.005)]), nominal[3:]
  )
  np.testing.assert_allclose(result[3:], expected)
  assert result[2] == nominal[2]
