"""All task resets preserve the same folded, symmetric robot home pose."""

import numpy as np
import pytest
from kaihand_tactile_env.shared.config import ARM_HOME, SCENE_NAMES, task_config
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation
from kaihand_tactile_env.tasks.install_ram.task import RamInstallSimulation
from kaihand_tactile_env.tasks.vase_wipe.task import VaseWipeSimulation
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_task_reset_uses_shared_home_without_task_pose_overrides(scene):
  factories = {
    "whiteboard-wipe": WhiteboardWipeSimulation,
    "vase-wipe": VaseWipeSimulation,
    "bulb-screw": BulbScrewSimulation,
    "install-ram": RamInstallSimulation,
  }
  sim = (
    factories[scene](add_genesis_probes=False)
    if scene in factories
    else ArmHandSimulation(scene=scene, add_genesis_probes=False)
  )
  assert task_config(scene).ARM_HOME is ARM_HOME
  # Check construction AND reset after actual motion has changed the state.
  for _ in range(2):
    for side in ("left", "right"):
      np.testing.assert_array_equal(sim.data.qpos[sim._arm_qpos[side]], ARM_HOME[side])
      np.testing.assert_array_equal(sim.arm_goal[side], ARM_HOME[side])
      np.testing.assert_array_equal(sim._arm_command[side], ARM_HOME[side])
    np.testing.assert_array_equal(
      sim.data.qpos[sim._hand_qpos["left"]],
      sim.data.qpos[sim._hand_qpos["right"]],
    )
    mirror = np.array([1, -1, 1])
    for body in ("arm_link2", "arm_link4", "arm_link5", "arm_flange"):
      np.testing.assert_allclose(
        sim.data.body(f"right_{body}").xpos,
        mirror * sim.data.body(f"left_{body}").xpos,
        atol=1e-12,
      )
    for side in ("left", "right"):
      shoulder = sim.data.body(f"{side}_arm_link2").xpos
      elbow = sim.data.body(f"{side}_arm_link4").xpos
      wrist = sim.data.body(f"{side}_arm_link5").xpos
      upper, forearm = shoulder - elbow, wrist - elbow
      angle = np.degrees(
        np.arccos(upper @ forearm / np.linalg.norm(upper) / np.linalg.norm(forearm))
      )
      assert 50 < angle < 85
      assert abs(np.degrees(np.arctan2(forearm[2], np.linalg.norm(forearm[:2])))) < 5
    sim.set_arm_joint_goal("right", ARM_HOME["right"] + 0.01)
    sim.step(5)
    sim.reset()
