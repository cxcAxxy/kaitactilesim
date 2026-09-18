"""Seeded bulb/RAM pickup layouts, contact isolation and archive metadata."""

import json

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import WorkcellConfig
from kaihand_tactile_env.tasks.bulb_screw.execution import BulbScrewExecutor
from kaihand_tactile_env.tasks.bulb_screw.grasp import calibrated_grasp as bulb_grasp
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation
from kaihand_tactile_env.tasks.install_ram.execution import RamInstallExecutor
from kaihand_tactile_env.tasks.install_ram.grasp import calibrated_grasp as ram_grasp
from kaihand_tactile_env.tasks.install_ram.task import RamInstallSimulation


@pytest.mark.parametrize(
  "cls,name,grasp_fn",
  [
    (BulbScrewSimulation, "bulb", lambda sim: bulb_grasp(sim, five_finger=True)),
    (RamInstallSimulation, "ram", ram_grasp),
  ],
)
def test_reset_bounds_and_grasp_follow_object(cls, name, grasp_fn):
  sim = cls()
  pose = sim.object_pose(name).copy()
  body_pos = sim.model.body_pos.copy()
  friction = sim.model.geom_friction.copy()
  collision = sim.model.geom_contype.copy()
  hand_home = sim.data.qpos[sim._arm_qpos["right"]].copy()
  nominal_grasp = grasp_fn(sim)
  for seed in range(50):
    sim.position_seed = seed
    sim.reset()
    offset = sim.object_pose(name)[:3] - pose[:3]
    assert np.max(np.abs(offset[:2])) <= 0.002
    np.testing.assert_array_equal(sim.object_pose(name)[2:], pose[2:])
    np.testing.assert_array_equal(sim.data.qpos[sim._arm_qpos["right"]], hand_home)
    expected_bodies = body_pos.copy()
    if name == "ram":
      stand = sim.model.body("ram_presentation_stand").id
      expected_bodies[stand] += offset
      # Preserve the complete RAM/support contact geometry after translation.
      relative = sim.data.xpos[sim.model.body("ram").id] - sim.data.xpos[stand]
      np.testing.assert_allclose(relative, pose[:3] - body_pos[stand], atol=1e-12)
    np.testing.assert_allclose(sim.model.body_pos, expected_bodies, atol=1e-15)
    if seed in (0, 3, 5, 8):
      target = grasp_fn(sim)
      np.testing.assert_allclose(
        target.wrist_position - nominal_grasp.wrist_position, offset, atol=1e-12
      )
  metadata = sim.initial_position_randomization
  sim.reset()
  assert sim.initial_position_randomization == metadata
  np.testing.assert_array_equal(sim.model.geom_friction, friction)
  np.testing.assert_array_equal(sim.model.geom_contype, collision)
  sim.position_seed = None
  sim.reset()
  np.testing.assert_array_equal(sim.model.body_pos, body_pos)
  np.testing.assert_array_equal(sim.object_pose(name), pose)
  sim.step()
  with pytest.raises(RuntimeError, match="only allowed at reset"):
    sim._pickup_randomization.apply(sim, 1)


@pytest.mark.parametrize("name", ["bulb", "ram"])
def test_seeded_position_is_archived_in_hdf5(tmp_path, monkeypatch, name):
  if name == "bulb":
    from kaihand_tactile_env.tasks.bulb_screw.example import BulbExampleRecorder

    sim = BulbScrewSimulation(position_seed=5)
    executor = BulbScrewExecutor(sim, grasp_mode="five-finger", speed="fast")
    recorder_type = BulbExampleRecorder
  else:
    from kaihand_tactile_env.tasks.install_ram.example import RamExampleRecorder

    sim = RamInstallSimulation(position_seed=5)
    executor = RamInstallExecutor(sim)
    executor.prepare()
    recorder_type = RamExampleRecorder
  initial = sim.object_pose(name).copy()
  path = tmp_path / "short.h5"
  capture = WorkcellConfig(model_path=sim.model_path, cameras=())
  with recorder_type(path, executor, capture) as recorder:
    recorder.record_initial()
    for _ in range(5):
      sim.step()
      executor._state = executor.monitor.update()
      recorder.observe(sim, "randomized_initial")
    recorder.record_terminal()
    recorder.set_outcome({"object_name": name, "success": False})
  with h5py.File(path) as file:
    layout = json.loads(file.attrs["metadata_json"])["initial_position_randomization"]
    assert layout == sim.initial_position_randomization
    np.testing.assert_array_equal(layout["initial_pose_xyz_wxyz"], initial)
    assert layout["seed"] == 5
    assert file.attrs["hdf5_buffer_rows"] == (128 if name == "bulb" else 64)
    assert "taskspace_capture_source_sha256" in file.attrs
    assert "commands/actuator_control" in file
    assert "tactile_contact_force/timestamp" in file

    if name == "ram":
      from kaihand_tactile_env.tasks.install_ram import example

      class CheckedRenderer:
        def __init__(self, model, cameras):
          self.stand = model.body("ram_presentation_stand").id

        def __enter__(self):
          return self

        def __exit__(self, *args):
          pass

        def capture(self, data, camera):
          np.testing.assert_array_equal(
            data.xpos[self.stand], layout["support_body_position_m"]
          )
          return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}

      monkeypatch.setattr(example, "WorkcellRenderer", CheckedRenderer)
      example._closeups(file, tmp_path)
      assert (tmp_path / "ram_closeup_initial.png").exists()
