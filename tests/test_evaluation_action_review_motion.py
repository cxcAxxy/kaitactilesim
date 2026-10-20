"""Native wrist-only evidence remains checksum-bound and exactly clocked."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_action_review import motion
from evaluation_progress_experiment.adapters import load_policy_episode


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(path, role):
    raw = path.read_bytes()
    return {"path": str(path), "role": role, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class WristMotionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source/seed_000"
        self.source.mkdir(parents=True)
        self.trace = self.source / "evaluation_rollout_trace.npz"
        self.report_path = self.root / "site/case/report.json"
        self.times = [0., .01, .02, .03]
        self.wrist = np.arange(36, dtype=float).reshape(4, 9) / 100
        self.write_trace()
        self.summary = {"task": "whiteboard-wipe", "prediction_horizon": 50, "video": {"comparison_plots": {
            "schema": "kaihand-evaluation-comparison-v1",
            "state": {"layout": "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof",
                      "dimension": 29, "wrist_site": "hand_r_base_link_site"},
            "rollout": {"samples": 4, "source_start_time_s": 0},
            "artifacts": {"rollout_trace": str(self.trace)}}}}
        self.report = {"trial": {"trial_id": "motion-fixture", "task": "whiteboard-wipe",
                                 "source": str(self.source), "prediction_horizon": 50},
                       "prediction_horizon_evidence": {"status": "verified", "prediction_horizon": 50},
                       "outcome": {"status": "fail"}}
        visual = {"schema_version": "evaluation-visualization-v1", "task_summary": {"task": "whiteboard-wipe"},
                  "clock_aligned": True, "time_origin_s": 0,
                  "task_series": [{"key": "cleaned_fraction", "time_s": self.times, "values": [0, 0, .2, .2]}],
                  "media": {"status": "missing"}}
        write_json(self.report_path.parent / "visualization.json", visual)
        self.refresh()

    def write_trace(self, *, times=None, simulation=None, wrist=None):
        times = np.asarray(self.times if times is None else times)
        np.savez_compressed(self.trace, time_s=times,
                            simulation_time_s=times if simulation is None else simulation,
                            wrist_state=self.wrist if wrist is None else wrist)

    def refresh(self, *, trace=True):
        write_json(self.source / "summary.json", self.summary)
        self.report["provenance"] = [row(self.source / "summary.json", "recorded_trial_outcome")]
        if trace:
            self.report["provenance"].append(row(self.trace, "native_control_cadence_tactile"))
        write_json(self.report_path, self.report)

    def episode(self):
        return load_policy_episode(self.report_path)

    def test_available_native_positions_exactly_preserve_episode_clock(self):
        episode = self.episode()
        before = deepcopy(episode)
        result = motion.load_wrist_motion(episode)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["time_s"], episode["time_s"])
        self.assertEqual(result["position_m"], self.wrist[:, :3].tolist())
        self.assertEqual((result["unit"], result["coordinate_frame"], result["site"]),
                         ("m", "world", "hand_r_base_link_site"))
        self.assertEqual(result["matched_count"], 4)
        self.assertEqual(result["source_sample_count"], 4)
        self.assertEqual(len(result["provenance"]), 3)
        self.assertEqual(episode, before)
        self.assertIn("不是板擦", " ".join(result["limitations"]))
        json.dumps(result, allow_nan=False)

    def test_nearby_and_outside_times_are_not_interpolated(self):
        episode = self.episode()
        episode["time_s"] = [-.01, 0, .01000000000001, .02, .04]
        result = motion.load_wrist_motion(episode)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["position_m"], [None, self.wrist[0, :3].tolist(), None,
                                               self.wrist[2, :3].tolist(), None])

    def test_missing_bound_trace_retains_nulls(self):
        episode = self.episode()
        self.trace.unlink()
        result = motion.load_wrist_motion(episode)
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["position_m"], [None] * 4)

    def test_unbound_existing_trace_is_not_discovered(self):
        self.refresh(trace=False)
        result = motion.load_wrist_motion(self.episode())
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["position_m"], [None] * 4)

    def test_changed_trace_hash_stops_export(self):
        episode = self.episode()
        self.write_trace(wrist=np.ones((4, 9)))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            motion.load_wrist_motion(episode)

    def test_changed_report_hash_stops_export(self):
        episode = self.episode()
        self.report["extra"] = "changed"
        write_json(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            motion.load_wrist_motion(episode)

    def test_changed_summary_hash_stops_export(self):
        episode = self.episode()
        self.summary["extra"] = "changed"
        write_json(self.source / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            motion.load_wrist_motion(episode)

    def test_hash_rechecked_after_decoding(self):
        episode = self.episode()
        original = motion._decode
        def changed(path):
            result = original(path)
            self.write_trace(wrist=np.ones((4, 9)))
            return result
        with patch.object(motion, "_decode", changed), self.assertRaisesRegex(ValueError, "changed during"):
            motion.load_wrist_motion(episode)

    def test_episode_source_bindings_cannot_disagree_with_report(self):
        episode = self.episode()
        next(item for item in episode["provenance"] if item["path"] == str(self.trace))["sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            motion.load_wrist_motion(episode)

    def test_wrong_episode_identity_rejected(self):
        episode = self.episode()
        episode["id"] = "another-case"
        with self.assertRaisesRegex(ValueError, "identity"):
            motion.load_wrist_motion(episode)

    def test_only_h50_policy_supported(self):
        episode = self.episode()
        for key, value in (("role", "expert_reference"), ("task", "whiteboard-wipe-horizontal")):
            wrong = {**episode, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "H50 policy"):
                motion.load_wrist_motion(wrong)
        episode["metadata"]["prediction_horizon"] = 30
        with self.assertRaisesRegex(ValueError, "H50 policy"):
            motion.load_wrist_motion(episode)

    def test_bad_episode_clocks_rejected(self):
        episode = self.episode()
        for times in ([0, 0], [1, 0], [0, float("nan")], [0, float("inf")], [0, True], ["0", "1"]):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "clock"):
                motion.load_wrist_motion({**episode, "time_s": times})

    def test_bad_native_clocks_retain_unknown_positions(self):
        for times in ([0, 0, .02, .03], [0, .02, .01, .03], [0, float("nan"), .02, .03]):
            with self.subTest(times=times):
                self.write_trace(times=times)
                self.refresh()
                result = motion.load_wrist_motion(self.episode())
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["position_m"], [None] * 4)

    def test_mismatched_elapsed_simulation_clock_rejected(self):
        self.write_trace(simulation=np.array([0, .011, .02, .03]))
        self.refresh()
        self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "invalid")

    def test_nonzero_origin_verified_without_rezeroing_episode(self):
        simulation = np.array([10, 10.01, 10.02, 10.03])
        elapsed = simulation - simulation[0]
        self.write_trace(times=elapsed, simulation=simulation)
        self.summary["video"]["comparison_plots"]["rollout"]["source_start_time_s"] = 10
        self.refresh()
        episode = self.episode()
        episode["metadata"]["source_time_origin_s"] = 10
        episode["time_s"] = elapsed.tolist()
        self.assertEqual(motion.load_wrist_motion(episode)["status"], "available")
        episode["metadata"]["source_time_origin_s"] = 0
        self.assertEqual(motion.load_wrist_motion(episode)["status"], "invalid")

    def test_nonfinite_xyz_is_null_not_zero(self):
        wrist = self.wrist.copy()
        wrist[1, 0] = np.nan
        self.write_trace(wrist=wrist)
        self.refresh()
        result = motion.load_wrist_motion(self.episode())
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["position_m"][1])
        json.dumps(result, allow_nan=False)

    def test_bad_wrist_shape_and_object_dtype_rejected(self):
        for wrist in (np.ones((4, 3)), np.ones((3, 9)), np.full((4, 9), "object", dtype=object)):
            with self.subTest(shape=wrist.shape, dtype=wrist.dtype):
                self.write_trace(wrist=wrist)
                self.refresh()
                self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "invalid")

    def test_bound_semantics_required_and_conflicting_units_rejected(self):
        original = deepcopy(self.summary)
        for key, value in (("layout", "commanded_wrist"), ("position_unit", "mm"),
                           ("coordinate_frame", "local"), ("wrist_site", "eraser")):
            with self.subTest(key=key):
                self.summary = deepcopy(original)
                self.summary["video"]["comparison_plots"]["state"][key] = value
                self.refresh()
                self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "invalid")
        self.summary = deepcopy(original)
        del self.summary["video"]["comparison_plots"]
        self.refresh()
        self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "invalid")

    def test_header_shape_bomb_is_rejected_without_allocation(self):
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {"descr": "<f8", "fortran_order": False,
                                                     "shape": (10**12,)})
        with zipfile.ZipFile(self.trace, "w") as archive:
            archive.writestr("time_s.npy", payload.getvalue())
        self.refresh()
        self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "invalid")

    def test_no_taxel_arrays_are_loaded(self):
        # A bad unused member must not execute or be loaded by a broad np.load.
        with zipfile.ZipFile(self.trace, "a") as archive:
            archive.writestr("normal_taxel_force_n.npy", b"not-an-npy")
        self.refresh()
        self.assertEqual(motion.load_wrist_motion(self.episode())["status"], "available")


if __name__ == "__main__":
    unittest.main()
