"""Native progress adapters preserve failures, missing data and source identity."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_progress_experiment.adapters import load_policy_episode, load_reference_episode


RIGHT_NAMES = [f"hand_r_{finger}_link{link}" for finger, link in
               (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4))]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def provenance(path, role):
    raw = path.read_bytes()
    return {"path": str(path), "role": role, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class ProgressAdapterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def policy(self, *, horizon=50, progress=True, task="whiteboard-wipe", force_times=None):
        source = self.root / "archive" / "seed_000"
        write_json(source / "summary.json", {"task": task, "prediction_horizon": horizon})
        report = {"trial": {"trial_id": "policy-fixture", "task": task, "source": str(source), "prediction_horizon": 50},
                  "prediction_horizon_evidence": {"status": "verified", "prediction_horizon": 50},
                  "outcome": {"status": "fail"}, "provenance": [provenance(source / "summary.json", "recorded_trial_outcome")]}
        times = [0, .01, .02, .03]
        series = [{"key": "board_normal_force_n", "time_s": times, "values": [0, 2, 3, 0], "gap_after": [False] * 4}]
        if progress:
            series.append({"key": "cleaned_fraction", "time_s": times, "values": [0, 0, .2, .2], "gap_after": [False] * 4})
        visual = {"schema_version": "evaluation-visualization-v1", "task_summary": {"task": task},
                  "clock_aligned": True, "time_origin_s": 0, "task_series": series,
                  "tactile_series": {"time_s": force_times or times, "normal_force_n": [[1, 2, 3, 4, 5]] * 4},
                  "channels": [{"name": name} for name in RIGHT_NAMES], "media": {"status": "missing"}}
        report_path = self.root / "site" / "policy" / "report.json"
        write_json(report_path, report)
        write_json(report_path.parent / "visualization.json", visual)
        return report_path

    def reference(self, *, ink=True, force_times=None, outcome="pass"):
        source = self.root / "raw" / "episode.h5"
        source.parent.mkdir(parents=True, exist_ok=True)
        times = np.array([0, .01, .02, .03])
        with h5py.File(source, "w") as file:
            file.attrs["schema_version"] = "kaihand_tactile_episode_v1"
            file.attrs["metadata_json"] = json.dumps({"scene": "whiteboard-wipe", "recording_contract": "whiteboard_wipe_shared_raw_v1"})
            file.create_dataset("state/timestamp", data=times)
            if ink:
                file.create_dataset("whiteboard_wipe/ink_remaining", data=[[1, 1], [1, .5], [.5, .5], [0, 0]])
            file.create_dataset("whiteboard_wipe/board_force_n", data=[0, 1, 2, 0])
            file.create_dataset("tactile_contact_force/timestamp", data=times if force_times is None else force_times)
            file.create_dataset("tactile_contact_force/normal_force_n", data=np.ones((4, 5)))
            file.create_dataset("tactile_contact_force/link_names", data=np.array(RIGHT_NAMES, dtype=h5py.string_dtype()))
            file["tactile_contact_force"].attrs["force_unit"] = "N"
            file["tactile_contact_force"].attrs["clock_semantics"] = "solver evaluation time before integration; state/timestamp is post-integration"
        hdf_row = provenance(source, "expert_hdf5")
        replay_path = self.root / "metadata" / "replay.json"
        write_json(replay_path, {"schema_version": "kaihand-offline-multimodal-replay-v1", "completed": True,
                               "task": "whiteboard-wipe", "source_hdf5": str(source), "source_sha256": hdf_row["sha256"]})
        report = {"trial": {"trial_id": "reference-fixture", "task": "whiteboard-wipe", "source": str(source)},
                  "outcome": {"status": outcome}, "reference_import": {"status": "imported", "source_schema": "kaihand_tactile_episode_v1"},
                  "provenance": [hdf_row, provenance(replay_path, "explicit_task_demo_replay_selection")]}
        report_path = self.root / "reference_site" / "whiteboard" / "report.json"
        write_json(report_path, report)
        write_json(report_path.parent.parent / "reference_manifest.json", {
            "schema_version": "conditional-reference-manifest-v1", "role": "expert_reference", "declared_by": "test curator",
            "reference_id": "fixture", "selection_rule": "success-selected illustrative examples", "evidence": "audited selection fixture",
            "members": [{"trial_id": "reference-fixture", "source": str(source), "observation_sha256": hdf_row["sha256"]}]})
        return report_path

    def refresh_reference_hashes(self, report_path):
        report = json.loads(report_path.read_text())
        hdf = next(row for row in report["provenance"] if row["role"] == "expert_hdf5")
        hdf.update(provenance(Path(hdf["path"]), "expert_hdf5"))
        replay_row = next(row for row in report["provenance"] if row["role"] == "explicit_task_demo_replay_selection")
        replay_path = Path(replay_row["path"])
        replay = json.loads(replay_path.read_text())
        replay["source_sha256"] = hdf["sha256"]
        write_json(replay_path, replay)
        replay_row.update(provenance(replay_path, replay_row["role"]))
        write_json(report_path, report)
        manifest_path = report_path.parent.parent / "reference_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["members"][0]["observation_sha256"] = hdf["sha256"]
        write_json(manifest_path, manifest)

    def test_policy_failure_and_native_exact_match_preserved(self):
        path = self.policy()
        result = load_policy_episode(path)
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["progress"], [0, 0, .2, .2])
        self.assertEqual(result["signals"]["hand_normal_sum"]["values"], [15] * 4)
        self.assertEqual(result["signals"]["board_normal"]["contact_role"], "eraser_board")
        self.assertEqual(result["metadata"]["quantitative_comparison"], "unverified")
        json.dumps(result, allow_nan=False)

    def test_policy_hand_nearby_timestamps_are_not_filled(self):
        result = load_policy_episode(self.policy(force_times=[0, .011, .021, .031]))
        self.assertEqual(result["signals"]["hand_normal_sum"]["values"], [15, None, None, None])

    def test_signal_gaps_remain_when_progress_clock_is_continuous(self):
        path = self.policy()
        visual_path = path.parent / "visualization.json"
        visual = json.loads(visual_path.read_text())
        visual["task_series"][0]["gap_after"] = [False, True, False, False]
        visual["tactile_series"]["gap_after"] = [False, False, True, False]
        write_json(visual_path, visual)
        result = load_policy_episode(path)
        self.assertEqual(result["progress"], [0, 0, .2, .2])
        self.assertEqual(result["signals"]["board_normal"]["values"], [0, 2, 3, 0])
        self.assertEqual(result["signals"]["hand_normal_sum"]["values"], [15] * 4)
        self.assertEqual(result["gap_after"], [False, True, True, False])
        self.assertIn("Conservative union", result["metadata"]["gap_semantics"])

    def test_missing_policy_progress_retains_episode(self):
        result = load_policy_episode(self.policy(progress=False))
        self.assertEqual(result["progress"], [None] * 4)
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["signals"]["board_normal"]["values"], [0, 2, 3, 0])

    def test_source_h30_cannot_be_hidden_by_h50_report(self):
        with self.assertRaisesRegex(ValueError, "H50"):
            load_policy_episode(self.policy(horizon=30))

    def test_horizontal_whiteboard_not_mixed_with_tilted(self):
        with self.assertRaisesRegex(ValueError, "tilted"):
            load_policy_episode(self.policy(task="whiteboard-wipe-horizontal"))

    def test_changed_source_summary_rejected(self):
        path = self.policy()
        source = Path(json.loads(path.read_text())["trial"]["source"])
        write_json(source / "summary.json", {"task": "whiteboard-wipe", "prediction_horizon": 50, "changed": True})
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_policy_episode(path)

    def test_nonmonotone_or_mismatched_policy_curve_rejected(self):
        path = self.policy()
        visual_path = path.parent / "visualization.json"
        visual = json.loads(visual_path.read_text())
        visual["task_series"][0]["time_s"] = [0, .01, .01, .03]
        write_json(visual_path, visual)
        with self.assertRaisesRegex(ValueError, "clock"):
            load_policy_episode(path)

    def test_json_duplicates_and_oversized_metadata_rejected(self):
        path = self.policy()
        with patch("evaluation_progress_experiment.adapters.MAX_JSON_BYTES", 8), self.assertRaisesRegex(ValueError, "oversized"):
            load_policy_episode(path)
        path.write_text('{"trial":{},"trial":{}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_policy_episode(path)

    def test_reference_derived_progress_keeps_phase_and_bias_limits(self):
        result = load_reference_episode(self.reference())
        self.assertEqual(result["role"], "expert_reference")
        self.assertEqual(result["progress"], [0, .25, .5, 1])
        self.assertEqual(result["signals"]["hand_normal_sum"]["values"], [5] * 4)
        self.assertEqual(result["metadata"]["physical_phase_alignment"], "unverified_or_offset")
        self.assertEqual(result["metadata"]["progress_definition_comparability"], "unverified")
        self.assertIn("success-selected", result["metadata"]["selection_bias"])
        self.assertIn("before integration", result["metadata"]["force_clock_semantics"])
        json.dumps(result, allow_nan=False)

    def test_missing_reference_progress_and_failed_outcome_retained(self):
        result = load_reference_episode(self.reference(ink=False, outcome="fail"))
        self.assertEqual(result["progress"], [None] * 4)
        self.assertEqual(result["outcome"], "fail")

    def test_reference_nearby_tactile_clock_is_not_matched(self):
        result = load_reference_episode(self.reference(force_times=[0, .011, .021, .031]))
        self.assertEqual(result["signals"]["hand_normal_sum"]["values"], [5, None, None, None])

    def test_expert_membership_is_not_inferred_from_success(self):
        path = self.reference()
        manifest_path = path.parent.parent / "reference_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["role"] = "ordinary_successful_trial"
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "expert-reference"):
            load_reference_episode(path)

    def test_reference_membership_and_replay_identity_required(self):
        path = self.reference()
        manifest_path = path.parent.parent / "reference_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["members"][0]["observation_sha256"] = "0" * 64
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "declared member"):
            load_reference_episode(path)

    def test_hdf_change_invalidates_archived_provenance(self):
        path = self.reference()
        report = json.loads(path.read_text())
        with h5py.File(report["trial"]["source"], "r+") as file:
            file["whiteboard_wipe/ink_remaining"][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_reference_episode(path)

    def test_hdf_external_link_rejected_even_with_valid_file_hash(self):
        path = self.reference()
        report = json.loads(path.read_text())
        with h5py.File(report["trial"]["source"], "r+") as file:
            del file["whiteboard_wipe/ink_remaining"]
            file["whiteboard_wipe/ink_remaining"] = h5py.ExternalLink("unrelated.h5", "/ink")
        self.refresh_reference_hashes(path)
        with self.assertRaisesRegex(ValueError, "Linked HDF5"):
            load_reference_episode(path)

    def test_hdf_wrong_ink_dimensions_rejected(self):
        path = self.reference()
        report = json.loads(path.read_text())
        with h5py.File(report["trial"]["source"], "r+") as file:
            del file["whiteboard_wipe/ink_remaining"]
            file.create_dataset("whiteboard_wipe/ink_remaining", data=[0, 0, 0, 0])
        self.refresh_reference_hashes(path)
        with self.assertRaisesRegex(ValueError, "Ink-patch dimensions"):
            load_reference_episode(path)

    def test_invalid_ink_value_is_null_not_clipped(self):
        path = self.reference()
        report = json.loads(path.read_text())
        with h5py.File(report["trial"]["source"], "r+") as file:
            file["whiteboard_wipe/ink_remaining"][1] = [1, 2]
        self.refresh_reference_hashes(path)
        self.assertEqual(load_reference_episode(path)["progress"], [0, None, .5, 1])


if __name__ == "__main__":
    unittest.main()
