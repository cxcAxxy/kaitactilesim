"""Short real-physics checks for the RAM recording contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import WorkcellConfig, model_fingerprint
from kaihand_tactile_env.shared.recording import validate_episode
from kaihand_tactile_env.tasks.install_ram.diagnostics import ForceTraceRecorder
from kaihand_tactile_env.tasks.install_ram.example import (
  RamExampleRecorder,
  _forces,
  _provenance,
  _publish_example,
  _sha256,
)
from kaihand_tactile_env.tasks.install_ram.force_analysis import _noise_metrics
from kaihand_tactile_env.tasks.install_ram.task import (
  RamInstallationMonitor,
  RamInstallSimulation,
)


def test_ram_recording_keeps_exact_terminal_clock_and_real_task_state(tmp_path):
  sim = RamInstallSimulation()
  monitor = RamInstallationMonitor(sim)
  executor = SimpleNamespace(sim=sim, state=monitor.measure())
  capture = WorkcellConfig(model_path=sim.model_path, cameras=())
  path = tmp_path / "idle.h5"
  with RamExampleRecorder(path, executor, capture) as recorder:
    recorder.record_initial()
    for _ in range(17):
      sim.step()
      executor.state = monitor.update()
      recorder.observe(sim, "idle_schema_test")
    recorder.record_terminal()
    recorder.set_outcome({"object_name": "ram", "success": False})
  report = validate_episode(path)
  assert report.valid, report.errors
  assert report.state_samples == 5
  with h5py.File(path, "r") as file:
    timestamps, *_ = _forces(file)
    np.testing.assert_allclose(timestamps, (0, 0.01, 0.02, 0.03, 0.034), atol=1e-12)
    np.testing.assert_array_equal(timestamps, file["state/timestamp"][:])
    np.testing.assert_array_equal(timestamps, file["install_ram/timestamp"][:])
    assert file["commands/phase"].asstr()[-1] == "terminal_settle"
    assert not file["install_ram/seated"][-1]
    assert file["install_ram/ram_qfrc_applied"].shape == (5, 6)
    assert not np.any(file["install_ram/ram_qfrc_applied"][:])
    assert not np.any(file["install_ram/xfrc_applied"][:])
    assert file["install_ram/eq_active"].shape == (5, sim.model.neq)
    assert set(file["objects"]) == {"ram"}


def test_ram_provenance_preserves_shared_recursive_model_fingerprint(tmp_path):
  from kaihand_tactile_env.tasks.install_ram import config

  model = Path(config.__file__).with_name("scene.xml")
  manifest = _provenance(tmp_path, model)
  snapshot = tmp_path / "sources/src/kaihand_tactile_env/tasks/install_ram/scene.xml"
  assert model_fingerprint(snapshot) == model_fingerprint(model)
  assert manifest["model_fingerprint"] == model_fingerprint(model)
  assert "src/kaihand_tactile_env/shared/mjcf/robot.xml" in manifest["source_files"]
  assert "src/kaihand_tactile_env/shared/mjcf/control.xml" in manifest["source_files"]
  assert "src/kaihand_tactile_env/shared/contact_tactile.py" in manifest["source_files"]
  assert "src/kaihand_tactile_env/shared/pickup_randomization.py" in manifest["source_files"]


def test_ram_buffer_preserves_every_recorded_value_and_terminal_tail(tmp_path):
  sim = RamInstallSimulation()
  monitor = RamInstallationMonitor(sim)
  executor = SimpleNamespace(sim=sim, state=monitor.measure())
  capture = WorkcellConfig(model_path=sim.model_path, cameras=())
  paths = [tmp_path / f"rows_{rows}.h5" for rows in (0, 8)]
  recorders = [
    RamExampleRecorder(path, executor, capture, buffer_rows=rows)
    for path, rows in zip(paths, (0, 8), strict=True)
  ]
  trace = ForceTraceRecorder(sim)
  trace.record("reset", executor.state)
  for recorder in recorders:
    recorder.record_initial()
  for _ in range(57):
    sim.step()
    executor.state = monitor.update()
    trace.record("mutable_live_physics", executor.state)
    for recorder in recorders:
      recorder.observe(sim, "mutable_live_physics")
  for recorder in recorders:
    recorder.record_terminal()
    recorder.set_outcome({"object_name": "ram", "success": False})
    recorder.close()
  with h5py.File(paths[0]) as reference, h5py.File(paths[1]) as buffered:
    names = []
    reference.visititems(
      lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
    )
    for name in names:
      assert reference[name].dtype == buffered[name].dtype
      np.testing.assert_array_equal(
        reference[name][...], buffered[name][...], err_msg=name
      )
    statistics = json.loads(buffered.attrs["hdf5_buffer_statistics_json"])
    assert statistics["pending_rows"] == 0
    assert buffered["state/timestamp"][-1] == pytest.approx(0.114)
    arrays = trace.arrays()
    indices = np.searchsorted(arrays["time"], buffered["state/timestamp"][:])
    np.testing.assert_array_equal(
      arrays["time"][indices], buffered["state/timestamp"][:]
    )
    np.testing.assert_array_equal(
      arrays["fn"][indices], buffered["tactile_contact_force/normal_force_n"][:]
    )
    np.testing.assert_array_equal(
      arrays["ft"][indices], buffered["tactile_contact_force/tangent_force_n"][:]
    )
    assert not trace.record("terminal_settle", executor.state)


def test_failed_replacement_restores_previous_completed_example(tmp_path, monkeypatch):
  output, partial = tmp_path / "example", tmp_path / "example.partial"
  output.mkdir()
  partial.mkdir()
  (output / "delivery_manifest.json").write_text('{"old": true}')
  (output / "raw.h5").write_bytes(b"preserve previous observations")
  (partial / "summary.json").write_text('{"completed": true}')
  previous = _sha256(output / "delivery_manifest.json")
  original = Path.rename

  def fail_new_publish(path, target):
    if path == partial:
      raise OSError("simulated destination rename failure")
    return original(path, target)

  monkeypatch.setattr(Path, "rename", fail_new_publish)
  with pytest.raises(OSError, match="simulated"):
    _publish_example(partial, output, previous)
  assert (output / "raw.h5").read_bytes() == b"preserve previous observations"
  assert partial.exists()
  assert not output.with_name("example.replaced").exists()


def test_raw_high_frequency_metric_distinguishes_chatter_from_smooth_loading():
  time = np.arange(200) / 100
  signals = np.column_stack(
    (np.sin(2 * np.pi * 40 * time), np.sin(2 * np.pi * 5 * time))
  )
  untouched = signals.copy()
  report = _noise_metrics(time, signals, np.ones(len(time), dtype=bool))
  assert report["spectrum"]["band_power_fraction"][0] > 0.99
  assert report["spectrum"]["band_power_fraction"][1] < 0.001
  assert report["spectrum"]["band_2_to_10hz_rms_n"][0] < 0.001
  assert report["spectrum"]["band_2_to_10hz_rms_n"][1] > 0.6
  assert (
    report["adjacent_difference_rms_n"][0] > 3 * report["adjacent_difference_rms_n"][1]
  )
  np.testing.assert_array_equal(signals, untouched)
