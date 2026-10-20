from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import whiteboard_tactile_reference as readers


def _sources(tmp_path):
  positions, bases = readers._canonical_coordinates()
  fingers = ("thumb", "index", "middle", "ring", "pinky")
  names = [
    f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}"
    for side in ("l", "r")
    for finger in fingers
  ]
  source = tmp_path / "expert.h5"
  times = np.array([0.0, 0.01, 0.02])
  with h5py.File(source, "w") as file:
    file.attrs["metadata_json"] = json.dumps({"scene": "whiteboard-wipe"})
    file.attrs["outcome_json"] = json.dumps(
      {"success": True, "elapsed_s": 0.02, "cleaned_fraction": 1.0}
    )
    file["state/timestamp"] = times
    group = file.create_group("tactile_contact_force")
    group.attrs["metadata_json"] = json.dumps({"source": readers.SOURCE})
    group["link_names"] = np.array(names, dtype="S")
    group["timestamp"] = times
    group["normal_force_n"] = np.ones((3, 10))
    group["normal_taxel_force_n"] = np.ones((3, 10, 7, 5)) / 35
    group["tangent_taxel_force_n"] = np.zeros((3, 10, 7, 5, 2))
    group["taxel_positions_local_m"] = np.concatenate((positions, positions))
    group["tangent_basis_local"] = np.concatenate((bases, bases))
    file["whiteboard_wipe/board_force_n"] = [0.0, 1.0, 1.0]
    file["whiteboard_wipe/ink_remaining"] = [[1.0], [0.5], [0.0]]
  reference = readers.load_whiteboard_tactile_reference(reference_hdf5=source)
  trial = tmp_path / "trial"
  trial.mkdir()
  manifest = trial / "deployment.json"
  manifest.write_text(
    json.dumps(
      {
        "evaluation_source_files": [
          {
            "path": relative,
            "sha256": hashlib.sha256(
              (readers._SOURCE_ROOT / relative).read_bytes()
            ).hexdigest(),
          }
          for relative in readers._COORDINATE_SOURCE_FILES
        ]
      }
    )
  )
  summary = {
    "task": "whiteboard-wipe",
    "deployment_manifest": str(manifest),
    "video": {"tactile_source": readers.SOURCE},
    "evaluation": {"success": False},
    "sim_seconds": 0.02,
    "status": "error",
  }
  (trial / "summary.json").write_text(json.dumps(summary))
  np.savez(
    trial / "evaluation_rollout_trace.npz",
    simulation_time_s=times,
    fingertip_link_names=np.array(names[5:]),
    fingertip_normal_force_n=np.ones((3, 5)),
    normal_taxel_force_n=np.ones((3, 5, 7, 5)) / 35,
    tangent_taxel_force_n=np.zeros((3, 5, 7, 5, 2)),
  )
  return source, reference, trial, manifest


def test_loads_native_expert_grids_and_verifies_historical_coordinates(tmp_path):
  _, reference, trial, _ = _sources(tmp_path)
  trace = readers.load_historical_whiteboard_tactile_trial(trial, reference)
  assert reference.normal_taxel_force_n.shape == (3, 5, 7, 5)
  np.testing.assert_allclose(trace.normal_taxel_force_n, reference.normal_taxel_force_n)
  np.testing.assert_array_equal(
    trace.taxel_positions_local_m, reference.taxel_positions_local_m
  )
  assert trace.metadata["coordinate_verification"]["status"] == "verified"
  assert trace.metadata["original_artifacts_modified"] is False


def test_missing_or_mismatched_frozen_layout_is_not_silently_assumed(tmp_path):
  _, reference, trial, manifest = _sources(tmp_path)
  content = json.loads(manifest.read_text())
  content["evaluation_source_files"][0]["sha256"] = "0" * 64
  manifest.write_text(json.dumps(content))
  with pytest.raises(ValueError, match="历史版本不一致"):
    readers.load_historical_whiteboard_tactile_trial(trial, reference)
  content["evaluation_source_files"] = []
  manifest.write_text(json.dumps(content))
  with pytest.raises(ValueError, match="缺少指尖坐标来源"):
    readers.load_historical_whiteboard_tactile_trial(trial, reference)


def test_expert_and_model_coordinate_mismatch_is_rejected(tmp_path):
  _, reference, trial, _ = _sources(tmp_path)
  reference.taxel_positions_local_m[0, 0, 0] += 0.001
  with pytest.raises(ValueError, match="专家指尖坐标"):
    readers.load_historical_whiteboard_tactile_trial(trial, reference)


def test_incompatible_provider_is_rejected(tmp_path):
  _, reference, trial, _ = _sources(tmp_path)
  path = trial / "summary.json"
  summary = json.loads(path.read_text())
  summary["video"]["tactile_source"] = "different_provider"
  path.write_text(json.dumps(summary))
  with pytest.raises(ValueError, match="触觉来源不一致"):
    readers.load_historical_whiteboard_tactile_trial(trial, reference)


def test_expert_spatial_fields_are_required(tmp_path):
  source, _, _, _ = _sources(tmp_path)
  with h5py.File(source, "a") as file:
    del file["tactile_contact_force/taxel_positions_local_m"]
  with pytest.raises(ValueError, match="空间触觉字段"):
    readers.load_whiteboard_tactile_reference(reference_hdf5=source)


def test_expert_grid_conservation_is_checked(tmp_path):
  source, _, _, _ = _sources(tmp_path)
  with h5py.File(source, "a") as file:
    file["tactile_contact_force/normal_taxel_force_n"][0, 5, 0, 0] += 0.1
  with pytest.raises(ValueError, match="专家逐格法向力"):
    readers.load_whiteboard_tactile_reference(reference_hdf5=source)


def test_snapshot_selection_stays_within_shared_native_time(tmp_path):
  _, reference, trial, _ = _sources(tmp_path)
  trace = readers.load_historical_whiteboard_tactile_trial(trial, reference)
  snapshots = readers.choose_snapshot_times(trace, reference)
  assert len(snapshots) == 3
  assert all(
    reference.time_s[0] <= t <= min(trace.time_s[-1], reference.time_s[-1])
    for t in snapshots
  )
