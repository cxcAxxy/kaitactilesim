import hashlib
import json

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import whiteboard_contact_reference as reference_module
from kaihand_tactile_env.shared.whiteboard_contact_reference import (
  load_historical_whiteboard_contact_trial,
  load_whiteboard_contact_reference,
)


def _expert(path, *, success=True, task="whiteboard-wipe"):
  fingers = ("thumb", "index", "middle", "ring", "pinky")
  names = [f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}" for side in ("r", "l") for finger in fingers]
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps({"scene": task, "ink_randomization": {"seed": 123}})
    file.attrs["outcome_json"] = json.dumps({"success": success, "reason": "completed", "elapsed_s": 2.03, "ink_remaining": [0.0, 0.0], "peak_board_force_n": 9.0, "peak_direct_hand_board_force_n": 0.0})
    file["state/timestamp"] = [2.0, 2.01, 2.03]
    file["commands/phase"] = np.asarray(["grasp", "wipe", "return"], dtype=h5py.string_dtype())
    file["tactile_contact_force/timestamp"] = [2.0, 2.01, 2.03]
    file["tactile_contact_force/link_names"] = np.asarray(names, dtype=h5py.string_dtype())
    file["tactile_contact_force/normal_force_n"] = np.tile(np.arange(10, dtype=float), (3, 1))
    file["tactile_contact_force/tangent_load_n"] = np.ones((3, 10)) * 2
    file["whiteboard_wipe/board_force_n"] = [0.0, 4.0, 1.0]
    file["whiteboard_wipe/board_tangent_force_n"] = [0.0, 2.0, 0.5]
    file["whiteboard_wipe/direct_hand_board_force_n"] = [0.0, 0.0, 0.0]
    file["whiteboard_wipe/ink_remaining"] = [[1.0, 1.0], [0.8, 0.2], [0.0, 0.0]]
    file["whiteboard_wipe/patch_work_j"] = [[0.0, 0.0], [0.1, 0.2], [0.3, 0.4]]
  return path


def test_reference_uses_named_right_fingers_native_clocks_and_reported_peaks(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  np.testing.assert_allclose(trace.time_s, [0.0, 0.01, 0.03])
  np.testing.assert_allclose(trace.duration_s, [0.01, 0.02, 0.0])
  np.testing.assert_allclose(trace.channels["grip_normal_n"], [10.0] * 3)
  np.testing.assert_allclose(trace.channels["grip_tangent_load_n"], [10.0] * 3)
  assert trace.metadata["reported_peaks_n"]["board_normal_n"] == 9.0
  assert trace.channels["board_normal_n"].max() == 4.0
  assert trace.metadata["reference_episode_count"] == 1
  assert trace.metadata["ink_randomization"]["seed"] == 123
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [0.0, 0.5, 1.0])
  np.testing.assert_allclose(trace.signals["cleaning_work_j"], [0.0, 0.3, 0.7])
  assert "local ink patches" in trace.metadata["cleaning_work_semantics"]
  assert trace.metadata["final_cleaned_fraction"] == 1.0
  assert trace.metadata["stop_reason"] == "completed"


def test_reference_pairs_cleaning_state_rows_and_rejects_wrong_clocks(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  with h5py.File(path, "a") as file:
    file["tactile_contact_force/timestamp"][:] = [1.999, 2.009, 2.029]
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [0.0, 0.5, 1.0])
  assert trace.metadata["maximum_force_state_clock_skew_s"] == pytest.approx(0.001)
  with h5py.File(path, "a") as file:
    file["whiteboard_wipe/timestamp"] = [2.0, 2.03, 2.06]
  with pytest.raises(ValueError, match="Whiteboard signal clock"):
    load_whiteboard_contact_reference(reference_hdf5=path)
  with h5py.File(path, "a") as file:
    del file["whiteboard_wipe/timestamp"]
    file["tactile_contact_force/timestamp"][:] = [1.0, 1.01, 1.03]
  with pytest.raises(ValueError, match="force/state clocks"):
    load_whiteboard_contact_reference(reference_hdf5=path)


def test_reference_preserves_invalid_cleaning_rows_and_terminal_only_results(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  with h5py.File(path, "a") as file:
    file["whiteboard_wipe/ink_remaining"][1, 0] = np.nan
    file["whiteboard_wipe/patch_work_j"][1, 0] = np.nan
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [0.0, np.nan, 1.0], equal_nan=True)
  np.testing.assert_allclose(trace.signals["cleaning_work_j"], [0.0, np.nan, 0.7], equal_nan=True)
  assert trace.metadata["missing_cleaned_fraction_sample_count"] == 1
  with h5py.File(path, "a") as file:
    del file["whiteboard_wipe/ink_remaining"]
    del file["whiteboard_wipe/patch_work_j"]
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  assert np.isnan(trace.signals["cleaned_fraction"]).all()
  assert "cleaning_work_j" not in trace.signals
  assert trace.metadata["cleaned_fraction_signal_status"] == "missing_time_series"
  assert trace.metadata["final_cleaned_fraction"] == 1.0


def test_reference_direct_cleaned_fraction_invalid_values_remain_nan(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  with h5py.File(path, "a") as file:
    file["whiteboard_wipe/cleaned_fraction"] = [0.0, 1.1, np.nan]
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [0.0, np.nan, np.nan], equal_nan=True)
  assert trace.metadata["signal_sources"]["cleaned_fraction"] == "whiteboard_wipe/cleaned_fraction"


def test_replay_resolution_preserves_declared_not_verified_hash(tmp_path):
  expert = _expert(tmp_path / "expert.h5")
  folder = tmp_path / "_metadata" / "whiteboard-wipe"
  folder.mkdir(parents=True)
  replay = folder / "replay.json"
  replay.write_text(json.dumps({"task": "whiteboard-wipe", "completed": True, "source_hdf5": str(expert), "source_sha256": "declared"}))
  trace = load_whiteboard_contact_reference(demo_root=tmp_path)
  assert trace.metadata["source_replay_json"] == str(replay)
  assert trace.metadata["declared_source_sha256"] == "declared"
  assert trace.metadata["source_sha256_verified"] is False


def test_relocated_replay_resolves_exact_source_and_declared_hash(tmp_path, monkeypatch):
  expert = _expert(tmp_path / "local_expert.h5")
  folder = tmp_path / "_metadata" / "whiteboard-wipe"
  folder.mkdir(parents=True)
  replay = folder / "replay.json"
  original = "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/source.h5"
  digest = "a" * 64
  payload = json.dumps({"task": "whiteboard-wipe", "completed": True,
                        "source_hdf5": original, "source_sha256": digest}).encode()
  replay.write_bytes(payload)
  calls = []

  def resolve(path, *, demo_root, expected_sha256):
    calls.append((str(path), demo_root, expected_sha256))
    return expert

  monkeypatch.setattr(reference_module, "resolve_demo_path", resolve)
  trace = load_whiteboard_contact_reference(demo_root=tmp_path)
  assert calls == [(original, tmp_path, digest)]
  assert trace.metadata["source_hdf5"] == str(expert)
  assert trace.metadata["original_source_hdf5"] == original
  assert trace.metadata["declared_source_sha256"] == digest
  assert trace.metadata["source_sha256_verified"] is False
  assert replay.read_bytes() == payload


@pytest.mark.parametrize("matches", [True, False])
def test_reference_uses_local_migration_manifest_without_opening_nas(
  tmp_path, monkeypatch, matches,
):
  root = tmp_path / "task_demo"
  expert = root / "raw_sources" / "expert.h5"
  expert.parent.mkdir(parents=True)
  _expert(expert)
  digest = hashlib.sha256(expert.read_bytes()).hexdigest()
  original = "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/source.h5"
  (root / "migration_manifest.json").write_text(json.dumps({
    "schema": "kaihand-task-demo-migration-v1",
    "original_root": "/nas/chenxianchi/datasets/sim/task_demo",
    "files": [], "raw_sources": [{"path": "raw_sources/expert.h5",
      "original_path": original, "size": expert.stat().st_size, "sha256": digest}],
  }))
  folder = root / "_metadata" / "whiteboard-wipe"
  folder.mkdir(parents=True)
  replay = folder / "replay.json"
  payload = json.dumps({
    "task": "whiteboard-wipe", "completed": True,
    "source_hdf5": original, "source_sha256": digest if matches else "0" * 64,
  }).encode()
  replay.write_bytes(payload)
  opened = []
  original_open = h5py.File

  def local_only(path, *args, **kwargs):
    assert str(path) == str(expert)
    opened.append(path)
    return original_open(path, *args, **kwargs)

  monkeypatch.setattr(reference_module.h5py, "File", local_only)
  if matches:
    trace = load_whiteboard_contact_reference(demo_root=root)
    assert trace.metadata["source_hdf5"] == str(expert)
    assert trace.metadata["original_source_hdf5"] == original
    assert trace.metadata["source_sha256_verified"] is False
    assert len(opened) == 1
  else:
    with pytest.raises(ValueError, match="source SHA-256"):
      load_whiteboard_contact_reference(demo_root=root)
    assert opened == []
  assert replay.read_bytes() == payload


def test_explicit_reference_path_does_not_use_replay_relocation(tmp_path, monkeypatch):
  expert = _expert(tmp_path / "explicit.h5")

  def unexpected(*_args, **_kwargs):
    pytest.fail("an explicit reference must retain the caller's chosen path")

  monkeypatch.setattr(reference_module, "resolve_demo_path", unexpected)
  trace = load_whiteboard_contact_reference(reference_hdf5=expert, demo_root=tmp_path)
  assert trace.metadata["source_hdf5"] == str(expert)
  assert trace.metadata["original_source_hdf5"] is None


@pytest.mark.parametrize("missing", ["mapping", "manifest"])
def test_migrated_reference_never_falls_back_when_migration_evidence_is_missing(
  tmp_path, monkeypatch, missing,
):
  root = tmp_path / "task_demo"
  folder = root / "_metadata" / "whiteboard-wipe"
  folder.mkdir(parents=True)
  (folder / "replay.json").write_text(json.dumps({
    "task": "whiteboard-wipe", "completed": True,
    "source_hdf5": "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/source.h5",
    "source_sha256": "a" * 64,
  }))
  if missing == "mapping":
    (root / "migration_manifest.json").write_text(json.dumps({
      "schema": "kaihand-task-demo-migration-v1",
      "original_root": "/nas/chenxianchi/datasets/sim/task_demo",
      "files": [], "raw_sources": [],
    }))
  else:
    monkeypatch.setattr(reference_module, "DEFAULT_DEMO_ROOT", root)

  def unexpected(*_args, **_kwargs):
    pytest.fail("missing migration evidence must not open a NAS HDF5")

  monkeypatch.setattr(reference_module.h5py, "File", unexpected)
  message = "refusing external fallback" if missing == "mapping" else "manifest is missing"
  with pytest.raises(ValueError, match=message):
    load_whiteboard_contact_reference(demo_root=root)


@pytest.mark.parametrize("success,task", [(False, "whiteboard-wipe"), (True, "whiteboard-wipe-horizontal")])
def test_reference_rejects_failed_or_wrong_variant(tmp_path, success, task):
  path = _expert(tmp_path / "expert.h5", success=success, task=task)
  with pytest.raises(ValueError):
    load_whiteboard_contact_reference(reference_hdf5=path)


def test_reference_does_not_substitute_net_tangent_force(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  with h5py.File(path, "a") as file:
    del file["tactile_contact_force/tangent_load_n"]
    file["tactile_contact_force/tangent_force_n"] = np.ones((3, 10, 2))
  trace = load_whiteboard_contact_reference(reference_hdf5=path)
  assert "grip_tangent_load_n" not in trace.channels


@pytest.mark.parametrize("legacy", [False, True])
def test_historical_exact_join_and_missing_channels_are_not_fabricated(tmp_path, legacy):
  trace_dir = tmp_path / "review" if legacy else tmp_path
  trace_dir.mkdir(exist_ok=True)
  summary = {"evaluation": {"task": "whiteboard-wipe", "success": False, "contact_guard_diagnostics": {"peak_board_normal_force_n": 9.0, "peak_direct_hand_board_force_n": 7.0}}}
  (tmp_path / "summary.json").write_text(json.dumps(summary))
  np.savez(trace_dir / "evaluation_rollout_trace.npz", simulation_time_s=[0.0, 0.03, 0.06, 0.09], fingertip_normal_force_n=np.ones((4, 5)), fingertip_tangent_force_n=np.ones((4, 5)))
  rows = [{"sim_time": time, "task": {"board_normal_force_n": force}} for time, force in [(0.03, 2.0), (0.06, 3.0)]]
  (tmp_path / "control_steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
  trace = load_historical_whiteboard_contact_trial(tmp_path)
  np.testing.assert_allclose(trace.channels["board_normal_n"], [np.nan, 2, 3, np.nan], equal_nan=True)
  np.testing.assert_allclose(trace.duration_s, [0.03, 0.03, 0.03, 0.0])
  assert "hand_board_normal_n" not in trace.channels
  assert "grip_tangent_load_n" not in trace.channels
  assert trace.metadata["reported_peaks_n"]["hand_board_normal_n"] == 7.0
  assert trace.metadata["missing_board_sample_count"] == 2
  assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_reference_rejects_invalid_force_values(tmp_path):
  path = _expert(tmp_path / "expert.h5")
  with h5py.File(path, "a") as file:
    file["tactile_contact_force/normal_force_n"][1, 2] = np.nan
  with pytest.raises(ValueError, match="finite nonnegative"):
    load_whiteboard_contact_reference(reference_hdf5=path)


def _historical(tmp_path, rows, *, final=0.0):
  summary = {"status": "error", "error": "RuntimeError: hand struck the whiteboard", "sim_seconds": 0.09, "evaluation": {"task": "whiteboard-wipe", "success": False, "state": {"cleaned_fraction": final}}}
  (tmp_path / "summary.json").write_text(json.dumps(summary))
  np.savez(tmp_path / "evaluation_rollout_trace.npz", simulation_time_s=[0.0, 0.03, 0.06, 0.09], fingertip_normal_force_n=np.ones((4, 5)))
  (tmp_path / "control_steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
  return load_historical_whiteboard_contact_trial(tmp_path)


def test_historical_zero_progress_is_recorded_and_unmatched_rows_stay_missing(tmp_path):
  rows = [{"sim_time": time, "task": {"cleaned_fraction": value}} for time, value in [(0.03, 0.0), (0.04, 0.9), (0.06, 0.0), (np.nan, 0.8)]]
  trace = _historical(tmp_path, rows)
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [np.nan, 0.0, 0.0, np.nan], equal_nan=True)
  assert trace.metadata["missing_cleaned_fraction_sample_count"] == 2
  assert trace.metadata["cleaned_fraction_signal_status"] == "recorded"
  assert trace.metadata["final_cleaned_fraction"] == 0.0
  assert trace.metadata["trial_status"] == "error"
  assert trace.metadata["task_success"] is False
  assert trace.metadata["trial_sim_seconds"] == 0.09
  assert "hand struck" in trace.metadata["stop_reason"]


def test_historical_does_not_use_terminal_progress_to_fill_curve(tmp_path):
  trace = _historical(tmp_path, [{"sim_time": 0.03, "task": {"board_normal_force_n": 1.0}}], final=0.8)
  assert np.isnan(trace.signals["cleaned_fraction"]).all()
  assert trace.metadata["final_cleaned_fraction"] == 0.8
  assert trace.metadata["cleaned_fraction_signal_status"] == "missing_time_series"


def test_historical_invalid_progress_is_missing_and_frame_fallback_is_exact(tmp_path):
  rows = [{"sim_time": time, "task": {"cleaned_fraction": value}} for time, value in [(0.03, np.nan), (0.06, -0.1)]]
  _historical(tmp_path, rows)
  frames = [{"simulation_time_s": 0.03, "task": {"cleaned_fraction": 0.0}}, {"simulation_time_s": 0.061, "task": {"cleaned_fraction": 0.7}}, {"simulation_time_s": 0.09, "task_snapshot": {"state": {"cleaned_fraction": 0.1}}}]
  (tmp_path / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
  trace = load_historical_whiteboard_contact_trial(tmp_path)
  np.testing.assert_allclose(trace.signals["cleaned_fraction"], [np.nan, 0.0, np.nan, 0.1], equal_nan=True)
