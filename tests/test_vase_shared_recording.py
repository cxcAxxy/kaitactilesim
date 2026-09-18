import h5py
from kaihand_tactile_env.shared.config import WorkcellConfig
from kaihand_tactile_env.shared.recording import validate_episode
from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider
from kaihand_tactile_env.tasks.vase_wipe.review import VaseEpisodeRecorder
from kaihand_tactile_env.tasks.vase_wipe.task import VaseWipeSimulation


def test_vase_uses_shared_raw_schema_without_changing_task_step(tmp_path):
  sim = VaseWipeSimulation()
  capture = WorkcellConfig(
    model_path=sim.model_path,
    physics_hz=4000,
    control_hz=500,
    camera_hz=30,
    cameras=(),
    tactile_provider=SolverContactTactileProvider.source,
  )
  output = tmp_path / "episode.h5"
  with VaseEpisodeRecorder(output, sim, capture) as recorder:
    recorder.record_initial("tabletop_ready")
    sim.step(8)
    recorder.observe(sim, "tabletop_ready")
    recorder.record_terminal("terminal_settle")
    recorder.set_outcome({"success": False, "task": "vase-wipe"})

  report = validate_episode(output)
  assert report.valid, report.errors
  with h5py.File(output, "r") as file:
    assert file.attrs["schema_version"] == "kaihand_tactile_episode_v1"
    assert file.attrs["hdf5_buffer_rows"] == 128
    assert "taskspace_capture_source_sha256" in file.attrs
    assert "commands/actuator_control" in file
    assert file.attrs["contact_force_source"] == sim.forces.source
    assert "vase_wipe/flex_vertices_m" in file
    assert "tactile_contact_force/normal_taxel_force_n" in file
    assert file["state/timestamp"].shape[0] == 2
