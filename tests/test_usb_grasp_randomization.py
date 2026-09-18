"""The same physical pinch must follow a translated and rotated tabletop USB."""

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert.grasp import calibrated_grasp
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion


@pytest.fixture(scope="module")
def simulation():
  return ArmHandSimulation(scene="usb-insert", add_genesis_probes=False)


def object_frame_grasp(simulation, grasp):
  body = simulation.model.body("usb_plug").id
  position = simulation.data.xpos[body]
  rotation = simulation.data.xmat[body].reshape(3, 3)
  return (
    rotation.T @ (grasp.wrist_position - position),
    rotation.T @ grasp.wrist_rotation,
  )


@pytest.mark.parametrize(
  "offset,yaw_deg",
  [((0.01, -0.01), 5.0), ((-0.01, 0.01), -5.0), ((0.005, 0.002), 0.0)],
)
def test_randomized_pickup_preserves_object_relative_pinch(simulation, offset, yaw_deg):
  simulation.reset()
  initialize_for_insertion(simulation)
  nominal = calibrated_grasp(simulation)
  expected_position, expected_rotation = object_frame_grasp(simulation, nominal)

  simulation.reset()
  initialize_for_insertion(
    simulation, offset_xy_m=offset, yaw_offset_rad=np.deg2rad(yaw_deg)
  )
  qpos, qvel, ctrl = (
    simulation.data.qpos.copy(),
    simulation.data.qvel.copy(),
    simulation.data.ctrl.copy(),
  )
  goals = {side: goal.copy() for side, goal in simulation.arm_goal.items()}
  grasp = calibrated_grasp(simulation)
  position, rotation = object_frame_grasp(simulation, grasp)

  np.testing.assert_allclose(position, expected_position, atol=1e-12)
  np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
  np.testing.assert_array_equal(grasp.open_hand, nominal.open_hand)
  np.testing.assert_array_equal(grasp.closed_hand, nominal.closed_hand)
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  np.testing.assert_array_equal(simulation.data.ctrl, ctrl)
  for side, goal in goals.items():
    np.testing.assert_array_equal(simulation.arm_goal[side], goal)


def test_zero_yaw_retains_calibrated_wrist_exactly(simulation):
  simulation.reset()
  initialize_for_insertion(simulation)
  grasp = calibrated_grasp(simulation)
  # Snapshot of the finite-patch calibration (3.44-degree pinch-axis roll).
  expected = np.array([0.4387999973382907, -0.16420145244810566, 0.8142748814702918])
  expected += simulation.object_pose("usb_plug")[:3] - [0.5, -0.18, 0.6865]
  np.testing.assert_array_equal(grasp.wrist_position, expected)
  np.testing.assert_array_equal(
    grasp.wrist_rotation,
    [
      [0.7597394294994173, 0.43881519142401765, 0.47983041487529876],
      [-0.005618116619780522, -0.733484615307639, 0.6796828347638698],
      [0.6502033805122404, -0.5190775923525364, -0.5547918682604741],
    ],
  )


def test_quaternion_sign_does_not_change_grasp(simulation):
  simulation.reset()
  initialize_for_insertion(simulation, yaw_offset_rad=np.deg2rad(5))
  expected = calibrated_grasp(simulation)
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  simulation.data.qpos[address + 3 : address + 7] *= -1
  mujoco.mj_forward(simulation.model, simulation.data)
  actual = calibrated_grasp(simulation)
  np.testing.assert_array_equal(actual.wrist_position, expected.wrist_position)
  np.testing.assert_array_equal(actual.wrist_rotation, expected.wrist_rotation)
