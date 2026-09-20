from __future__ import annotations

from pathlib import Path

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import WorkcellConfig, default_model_path
from kaihand_tactile_env.shared.recording import (
  WRIST_WRENCH_SCHEMA_VERSION,
  EpisodeRecorder,
  WristWrenchSensor,
  validate_episode,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider

SCENES = (
  "pick-place",
  "poker-draw",
  "usb-insert",
  "install-ram",
  "vase-wipe",
  "whiteboard-wipe",
  "bulb-screw",
)


@pytest.mark.parametrize("scene", SCENES)
def test_every_task_model_has_shared_bimanual_wrist_ft_sensors(scene: str) -> None:
  model = mujoco.MjModel.from_xml_path(str(default_model_path(scene)))
  for side in ("left", "right"):
    site = model.site(f"{side}_wrist_ft_site")
    force = model.sensor(f"{side}_wrist_force")
    torque = model.sensor(f"{side}_wrist_torque")
    assert model.body(int(site.bodyid[0])).name == f"{side}_hand_mount"
    assert int(force.type[0]) == int(mujoco.mjtSensor.mjSENS_FORCE)
    assert int(torque.type[0]) == int(mujoco.mjtSensor.mjSENS_TORQUE)
    assert int(force.dim[0]) == int(torque.dim[0]) == 3


def test_shared_recorder_saves_aligned_bimanual_six_axis_wrist_data(
  tmp_path: Path,
) -> None:
  simulation = ArmHandSimulation(
    default_model_path("pick-place"),
    scene="pick-place",
    add_genesis_probes=False,
  )
  config = WorkcellConfig(
    model_path=simulation.model_path,
    cameras=(),
    tactile_provider=SolverContactTactileProvider.source,
  )
  output = tmp_path / "wrist_wrench.h5"
  with EpisodeRecorder(output, simulation, config) as recorder:
    recorder.record_initial()
    for _ in range(5):
      simulation.step()
      recorder.observe(simulation, "test")

  report = validate_episode(output)
  assert report.valid, report.errors
  with h5py.File(output) as file:
    group = file["wrist_wrench"]
    assert group.attrs["schema_version"] == WRIST_WRENCH_SCHEMA_VERSION
    assert group["side_names"].asstr()[:].tolist() == ["left", "right"]
    np.testing.assert_array_equal(group["timestamp"], file["state/timestamp"])
    count = report.state_samples
    assert group["force_local_n"].shape == (count, 2, 3)
    assert group["torque_local_nm"].shape == (count, 2, 3)
    assert group["force_world_n"].shape == (count, 2, 3)
    assert group["torque_world_nm"].shape == (count, 2, 3)
    rotation = group["world_from_sensor_rotation"][:]
    np.testing.assert_allclose(
      group["force_world_n"][:],
      np.einsum("tsij,tsj->tsi", rotation, group["force_local_n"][:]),
      atol=1e-12,
    )
    np.testing.assert_allclose(
      group["torque_world_nm"][:],
      np.einsum("tsij,tsj->tsi", rotation, group["torque_local_nm"][:]),
      atol=1e-12,
    )


def test_right_wrist_sensor_responds_to_force_and_torque_on_hand_subtree() -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  sensor = WristWrenchSensor(simulation.model)
  baseline = sensor.read(simulation.data)
  body = simulation.model.body("hand_r_base_link").id
  simulation.data.xfrc_applied[body] = [1.25, -2.0, 3.5, 0.2, -0.1, 0.3]
  mujoco.mj_forward(simulation.model, simulation.data)
  loaded = sensor.read(simulation.data)
  assert np.linalg.norm(loaded.force_local_n[1] - baseline.force_local_n[1]) > 0.1
  assert (
    np.linalg.norm(loaded.torque_local_nm[1] - baseline.torque_local_nm[1]) > 0.01
  )


def test_validator_keeps_legacy_v1_episode_compatible(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  config = WorkcellConfig(
    cameras=(), tactile_provider=SolverContactTactileProvider.source
  )
  output = tmp_path / "legacy.h5"
  with EpisodeRecorder(output, simulation, config) as recorder:
    recorder.record_initial()
  with h5py.File(output, "r+") as file:
    del file["wrist_wrench"]
  assert validate_episode(output).valid


def test_validator_rejects_corrupted_wrist_wrench_transform(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  config = WorkcellConfig(
    cameras=(), tactile_provider=SolverContactTactileProvider.source
  )
  output = tmp_path / "corrupt.h5"
  with EpisodeRecorder(output, simulation, config) as recorder:
    recorder.record_initial()
  with h5py.File(output, "r+") as file:
    file["wrist_wrench/force_world_n"][0, 0, 0] += 1.0
  report = validate_episode(output)
  assert not report.valid
  assert any("world/local transform" in error for error in report.errors)
