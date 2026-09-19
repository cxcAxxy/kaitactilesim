"""Initialization geometry only; no automatic robot motion is executed here."""

import json

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion


@pytest.fixture(scope="module")
def usb_model():
  return ArmHandSimulation(scene="usb-insert", add_genesis_probes=False)


@pytest.fixture
def simulation(usb_model):
  usb_model.reset()
  return usb_model


def test_zero_jitter_preserves_legacy_initialization_exactly(simulation):
  before, velocity = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  goals = simulation.arm_goal
  record = initialize_for_insertion(simulation)
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  wrist_address = int(simulation.model.joint("right_arm_joint5").qposadr[0])
  before[address + 3 : address + 7] = config.AUTO_PLUG_QUATERNION_WXYZ
  before[wrist_address] += 2 * np.pi
  np.testing.assert_array_equal(simulation.data.qpos, before)
  np.testing.assert_array_equal(simulation.data.qvel, velocity)
  np.testing.assert_array_equal(simulation.arm_goal["left"], goals["left"])
  expected_right_goal = goals["right"].copy()
  expected_right_goal[4] += 2 * np.pi
  np.testing.assert_array_equal(simulation.arm_goal["right"], expected_right_goal)
  assert simulation.data.time == 0.0
  assert record["offset_xy_m"] == [0.0, 0.0]
  assert record["yaw_offset_rad"] == 0.0
  assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize("yaw", [-np.pi / 2, -0.02, 0.02, np.pi / 2])
def test_explicit_offsets_apply_world_yaw_and_preserve_every_other_state(
  simulation, yaw
):
  before, velocity = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  record = initialize_for_insertion(
    simulation, offset_xy_m=[0.012, -0.008], yaw_offset_rad=yaw
  )
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  expected_position = before[address : address + 3] + [0.012, -0.008, 0.0]
  np.testing.assert_array_equal(
    simulation.data.qpos[address : address + 3], expected_position
  )
  base_rotation, actual_rotation = np.empty(9), np.empty(9)
  mujoco.mju_quat2Mat(base_rotation, config.AUTO_PLUG_QUATERNION_WXYZ)
  mujoco.mju_quat2Mat(actual_rotation, simulation.data.qpos[address + 3 : address + 7])
  c, s = np.cos(yaw), np.sin(yaw)
  world_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  np.testing.assert_allclose(
    actual_rotation.reshape(3, 3), world_yaw @ base_rotation.reshape(3, 3), atol=1e-14
  )
  keep = np.ones(len(before), dtype=bool)
  keep[address : address + 7] = False
  keep[int(simulation.model.joint("right_arm_joint5").qposadr[0])] = False
  np.testing.assert_array_equal(simulation.data.qpos[keep], before[keep])
  np.testing.assert_array_equal(simulation.data.qvel, velocity)
  assert np.linalg.norm(record["initial_pose_wxyz"][3:]) == pytest.approx(1.0)
  assert record["offset_xy_m"] == [0.012, -0.008]
  assert record["yaw_offset_rad"] == yaw
  assert simulation.data.time == 0.0


def test_random_offsets_are_bounded_repeatable_and_seed_dependent(simulation):
  records = []
  for seed in range(16):
    simulation.reset()
    first = initialize_for_insertion(
      simulation, seed=seed, xy_jitter_m=0.01, yaw_jitter_rad=0.1
    )
    simulation.reset()
    second = initialize_for_insertion(
      simulation, seed=seed, xy_jitter_m=0.01, yaw_jitter_rad=0.1
    )
    assert first == second
    assert np.all(np.abs(first["offset_xy_m"]) <= 0.01)
    assert abs(first["yaw_offset_rad"]) <= 0.1
    records.append(first)
  assert len({tuple(record["initial_pose_wxyz"]) for record in records}) == len(records)
  values = np.array(
    [record["offset_xy_m"] + [record["yaw_offset_rad"]] for record in records]
  )
  assert np.all(values.min(axis=0) < 0) and np.all(values.max(axis=0) > 0)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"xy_jitter_m": -0.01},
    {"yaw_jitter_rad": -0.1},
    {"xy_jitter_m": np.nan},
    {"yaw_jitter_rad": np.inf},
    {"offset_xy_m": [0.0]},
    {"offset_xy_m": [0.0, 0.0, 0.0]},
    {"offset_xy_m": [np.nan, 0.0]},
    {"yaw_offset_rad": np.inf},
    {"offset_xy_m": [0.0, 0.0], "xy_jitter_m": 0.01},
    {"yaw_offset_rad": 0.0, "yaw_jitter_rad": 0.1},
    {"seed": -1},
    {"seed": 1.5},
  ],
)
def test_invalid_parameters_are_rejected_without_mutation(simulation, kwargs):
  before, velocity = simulation.data.qpos.copy(), simulation.data.qvel.copy()
  with pytest.raises(ValueError):
    initialize_for_insertion(simulation, **kwargs)
  np.testing.assert_array_equal(simulation.data.qpos, before)
  np.testing.assert_array_equal(simulation.data.qvel, velocity)
  assert simulation.data.time == 0.0


def test_randomization_is_forbidden_after_physics_begins(simulation):
  simulation.step()
  before = simulation.data.qpos.copy()
  with pytest.raises(ValueError, match="only allowed immediately after reset"):
    initialize_for_insertion(simulation, xy_jitter_m=0.01)
  np.testing.assert_array_equal(simulation.data.qpos, before)


def test_randomization_rejects_other_scenes_before_accessing_model():
  from types import SimpleNamespace

  with pytest.raises(ValueError, match="requires scene='usb-insert'"):
    initialize_for_insertion(SimpleNamespace(scene="poker-draw"), xy_jitter_m=0.01)
