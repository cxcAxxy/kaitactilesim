"""Generic case assembly retains identities, native clocks, and missing data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_workbench import data
from evaluation_workbench.diagnostic_plots import load_diagnostic_plots
from test_evaluation_workbench_export import diagnostic_fixture


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class GenericCaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "seed_000"
        self.root.mkdir()

    def fixture(self, *, task="usb-insert", subdirectory=False, normal=True, tangent=True, taxels=True):
        directory = self.root / "review" if subdirectory else self.root
        directory.mkdir(exist_ok=True)
        names = list(data._LABELS)
        times = np.array([10., 10.1, 10.2, 10.8])
        normal_values = np.arange(20, dtype=float).reshape(4, 5) / 10
        normal_taxels = np.broadcast_to(normal_values[:, :, None, None] / 35, (4, 5, 7, 5)).copy()
        tangent_values = np.ones((4, 5)) * .25
        archive = {"simulation_time_s": times, "time_s": times - times[0], "fingertip_link_names": np.array(names)}
        if normal:
            archive["fingertip_normal_force_n"] = normal_values
        if tangent:
            archive["fingertip_tangent_force_n"] = tangent_values
        if taxels:
            archive["normal_taxel_force_n"] = normal_taxels
            archive["tangent_taxel_force_n"] = np.zeros((4, 5, 7, 5, 2))
        np.savez(directory / "evaluation_rollout_trace.npz", **archive)
        schema = {"unit": "N", "taxel_grid": [7, 5], "link_names": names,
                  "Fn": "sum of all 7x5 normal taxel forces", "Ft": next(iter(data._FT_FORMULAS))}
        comparison = {"schema": "kaihand-evaluation-comparison-v1", "status": "reference_unavailable",
                      "artifacts": {"rollout_trace": str(directory / "evaluation_rollout_trace.npz")}, "tactile": schema}
        review = {"schema": "kaihand-policy-evaluation-review-v3", "task": task, "seed": 0, "prediction_horizon": 50,
                  "completed": True, "fps": 10, "frame_count": 4, "first_pose_time_s": 10., "last_pose_time_s": 10.8,
                  "video_clock": "simulation time; final off-grid frame allowed", "comparison_plots": comparison,
                  "checkpoint_sha256": "a" * 64}
        summary = {"task": task, "seed": 0, "prediction_horizon": 50, "status": "task_not_completed",
                   "checkpoint_path": "/checkpoints/20000", "checkpoint_sha256": "a" * 64,
                   "formal_metrics_valid": False, "diagnostic_only": True, "evaluation": {"success": False},
                   "video": deepcopy(review)}
        write_json(self.root / "summary.json", summary)
        write_json(directory / "review.json", review)
        frames = [{"frame": i, "simulation_time_s": float(t), "camera_pose_time_s": float(t), "tactile_time_s": float(t),
                   "fingertip_order": names, "normal_taxel_force_n": normal_taxels[i].tolist(),
                   "tangent_taxel_force_n": np.zeros((5, 7, 5, 2)).tolist(),
                   "task_metrics": {"insertion_depth_mm": float(i)}} for i, t in enumerate(times)]
        (directory / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
        (directory / "review.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"x" * 64)
        return directory, summary, frames, archive

    def test_root_and_review_layouts_preserve_native_five_finger_data_and_failed_trial(self):
        directory, _, _, _ = self.fixture()
        bundle = data.build_case(self.root)
        loads, visual, report = (bundle[key] for key in ("tactile_loads", "visualization", "report"))
        self.assertEqual(report["outcome"]["status"], "fail")
        self.assertTrue(report["outcome"]["diagnostic_only"])
        self.assertFalse(report["outcome"]["formal_metrics_valid"])
        self.assertEqual(len(loads["time_s"]), 4)
        self.assertEqual(loads["normal_force_n"][1], [.5, .6, .7, .8, .9])
        self.assertEqual(loads["tangent_total_n"], [1.25] * 4)
        self.assertEqual(loads["gap_after"], [False, False, True, False])
        self.assertEqual(visual["spatial_frames"]["native_frame_count"], 4)
        self.assertEqual(visual["media"]["sync_status"], "mapped")
        self.assertEqual(visual["media"]["video_time_s"], [0., .1, .2, .3])
        self.assertNotAlmostEqual(visual["media"]["time_s"][-1], .3)
        self.assertEqual(bundle["config"]["diagnostic_plots"]["status"], "missing")
        self.assertEqual(bundle["copies"][0]["source"], directory / "review.mp4")
        # Move only the test recorder bundle; the next assembly must use review/.
        review = self.root / "review"
        review.mkdir()
        for name in ("evaluation_rollout_trace.npz", "review.json", "frames.jsonl", "review.mp4"):
            (self.root / name).rename(review / name)
        for path in (self.root / "summary.json", review / "review.json"):
            value = json.loads(path.read_text())
            video = value["video"] if path.name == "summary.json" else value
            video["comparison_plots"]["artifacts"]["rollout_trace"] = str(review / "evaluation_rollout_trace.npz")
            write_json(path, value)
        self.assertEqual(data.build_case(self.root)["visualization"]["media"]["sync_status"], "mapped")

    def test_missing_ft_and_nonfinite_normal_never_become_zero(self):
        directory, _, _, archive = self.fixture(tangent=False)
        archive["fingertip_normal_force_n"][1, 2] = np.nan
        np.savez(directory / "evaluation_rollout_trace.npz", **archive)
        loads = data.build_case(self.root)["tactile_loads"]
        self.assertIsNone(loads["normal_force_n"][1][2])
        self.assertIsNone(loads["normal_total_n"][1])
        self.assertEqual(loads["tangent_total_n"], [None] * 4)
        self.assertEqual(loads["tangent_status"], "missing")
        json.dumps(loads, allow_nan=False)

    def test_no_tactile_still_retains_trial_and_independent_video(self):
        directory, _, _, _ = self.fixture()
        (directory / "evaluation_rollout_trace.npz").unlink()
        (directory / "frames.jsonl").unlink()
        bundle = data.build_case(self.root)
        self.assertEqual(bundle["report"]["outcome"]["status"], "fail")
        self.assertEqual(bundle["tactile_loads"]["status"], "missing")
        self.assertEqual(bundle["tactile_loads"]["time_s"], [])
        self.assertEqual(bundle["visualization"]["media"]["status"], "available")
        self.assertEqual(bundle["visualization"]["media"]["sync_status"], "unverified")

    def test_camera_offset_is_not_silently_ignored_or_duration_stretched(self):
        directory, _, frames, _ = self.fixture()
        frames[1]["camera_pose_time_s"] -= .002
        (directory / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
        media = data.build_case(self.root)["visualization"]["media"]
        self.assertEqual(media["sync_status"], "unverified")
        self.assertEqual(media["time_s"], [])
        self.assertIn("Camera pose", media["reason"])

    def test_audited_distinct_pose_clock_maps_rgb_without_altering_fn_clock(self):
        directory, _, frames, _ = self.fixture(task="poker-draw")
        for frame in frames[1:]:
            frame["camera_pose_time_s"] -= .002
            frame["tactile_time_s"] -= .002
        (directory / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
        for path in (self.root / "summary.json", directory / "review.json"):
            value = json.loads(path.read_text())
            record = value["video"] if path.name == "summary.json" else value
            record["last_pose_time_s"] = frames[-1]["camera_pose_time_s"]
            write_json(path, value)
        with patch.object(data, "_pose_clock_evidence", return_value=[{"audited_fixture": True}]) as evidence:
            bundle = data.build_case(self.root)
        self.assertTrue(evidence.called)
        media = bundle["visualization"]["media"]
        self.assertEqual(media["sync_status"], "mapped")
        self.assertEqual(media["mapping_method"], "recorded_camera_pose_time_s")
        self.assertAlmostEqual(media["time_s"][1], .098)
        self.assertAlmostEqual(bundle["tactile_loads"]["time_s"][1], .1)
        self.assertAlmostEqual(media["camera_pose_minus_frame_simulation_s"]["minimum"], -.002)
        self.assertEqual(media["physical_phase_alignment"], "not_certified")
        self.assertEqual(media["video_time_s"], [0., .1, .2, .3])

    def pose_binding_fixture(self, contents):
        project = self.root / "frozen_source"
        rows, audit = [], {}
        for relative, payload in contents.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            audit[relative] = hashlib.sha256(payload).hexdigest()
            rows.append({"path": relative, "size": len(payload), "sha256": audit[relative]})
        manifest = self.root / "deployment.json"
        identity = {"task": "poker-draw", "deployment_id": "audited deployment", "checkpoint_sha256": "a" * 64}
        deployment = {**identity, "evaluation_project_path": str(project),
                      "evaluation_source_sha256": data.content_hash(rows), "evaluation_source_files": rows}
        write_json(manifest, deployment)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        summary = {**identity, "deployment_manifest": str(manifest), "deployment_integrity": {
            "status": "verified", "manifest": {"path": str(manifest), "expected_sha256": digest,
                                                  "actual_sha256": digest},
            "evaluation_source": {"status": "verified", "root": str(project),
                                  "expected_sha256": data.content_hash(rows), "actual_sha256": data.content_hash(rows)}}}
        return data.TrialData(root=self.root, task="poker-draw", summary=summary), audit, manifest

    def test_pose_clock_accepts_complete_legacy_and_staged_source_combinations(self):
        legacy = {"recorder.py": b"same cached recorder", "runner.py": b"legacy runner"}
        staged = {"recorder.py": b"same cached recorder", "runner.py": b"staged runner",
                  "observer.py": b"read-only corrected table geometry", "staged_recording.py": b"snapshot only"}
        audits = tuple({key: hashlib.sha256(value).hexdigest() for key, value in contents.items()}
                       for contents in (legacy, staged))
        with patch.object(data, "_POSE_CLOCK_SOURCE_SETS", audits):
            for contents in (legacy, staged):
                with self.subTest(staged="observer.py" in contents):
                    trial, _, _ = self.pose_binding_fixture(contents)
                    provenance = data._pose_clock_evidence(trial, [{"timing": data._CACHED_TIMING}])
                    self.assertEqual(len(provenance), len(contents) + 1)
                    self.assertTrue(all(row["sha256"] for row in provenance))
            unknown = {**staged, "observer.py": b"unreviewed observer"}
            trial, _, _ = self.pose_binding_fixture(unknown)
            with self.assertRaisesRegex(ValueError, "audited pose-clock combination"):
                data._pose_clock_evidence(trial, [{"timing": data._CACHED_TIMING}])

    def test_pose_clock_rejects_inventory_and_source_tampering(self):
        contents = {"recorder.py": b"cached recorder", "runner.py": b"bound runner"}
        trial, audit, manifest = self.pose_binding_fixture(contents)
        with patch.object(data, "_POSE_CLOCK_SOURCE_SETS", (audit,)):
            (self.root / "frozen_source/runner.py").write_bytes(b"changed runner")
            with self.assertRaisesRegex(ValueError, "source SHA256/size mismatch"):
                data._pose_clock_evidence(trial, [{"timing": data._CACHED_TIMING}])
            trial, _, manifest = self.pose_binding_fixture(contents)
            deployment = json.loads(manifest.read_text())
            deployment["evaluation_source_files"].append({"path": "undeclared.py", "sha256": "b" * 64, "size": 1})
            write_json(manifest, deployment)
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            trial.summary["deployment_integrity"]["manifest"].update(expected_sha256=digest, actual_sha256=digest)
            with self.assertRaisesRegex(ValueError, "inventory SHA256 mismatch"):
                data._pose_clock_evidence(trial, [{"timing": data._CACHED_TIMING}])

    def test_spatial_fallback_has_its_own_native_clock_not_fn_sample_count(self):
        directory, summary, frames, _ = self.fixture(taxels=False)
        frames = [frames[0], frames[-1]]
        frames[-1]["frame"] = 1
        (directory / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
        for path in (self.root / "summary.json", directory / "review.json"):
            value = json.loads(path.read_text())
            video = value["video"] if path.name == "summary.json" else value
            video["frame_count"] = 2
            write_json(path, value)
        bundle = data.build_case(self.root)
        self.assertEqual(len(bundle["tactile_loads"]["time_s"]), 4)
        self.assertEqual(bundle["visualization"]["spatial_frames"]["native_frame_count"], 2)
        self.assertIn("Separate native", bundle["visualization"]["spatial_frames"]["sampling"])

    def test_horizon_wrong_source_and_ambiguous_layout_fail_closed(self):
        directory, summary, _, _ = self.fixture()
        for value in (30, None):
            summary["prediction_horizon"] = value
            write_json(self.root / "summary.json", summary)
            with self.subTest(horizon=value), self.assertRaisesRegex(ValueError, "H50"):
                data.build_case(self.root)
        summary["prediction_horizon"] = 50
        summary["video"]["comparison_plots"]["artifacts"]["rollout_trace"] = str(self.root / "summary.json")
        write_json(self.root / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "another source"):
            data.build_case(self.root)
        (self.root / "review").mkdir()
        (self.root / "review/evaluation_rollout_trace.npz").write_bytes(b"unrelated duplicate")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            data.build_case(self.root)

    def test_changed_source_after_decode_is_fatal(self):
        self.fixture()
        original = data.load_diagnostic_plots
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.root / "summary.json"
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(data, "load_diagnostic_plots", side_effect=mutate), self.assertRaisesRegex(ValueError, "SHA256"):
            data.build_case(self.root)

    def test_optional_frame_read_mutation_cannot_be_downgraded_to_missing(self):
        self.fixture()
        original = data._read
        def mutate(path, **kwargs):
            if Path(path).name == "frames.jsonl":
                raise ValueError("Evidence changed while reading")
            return original(path, **kwargs)
        with patch.object(data, "_read", side_effect=mutate), self.assertRaisesRegex(ValueError, "changed while reading"):
            data.build_case(self.root)

    def test_generic_plots_accept_explicit_task_and_missing_images_are_missing(self):
        diagnostic_fixture(self.root, "poker-draw")
        config, copies = load_diagnostic_plots(self.root, expected_task="poker-draw")
        self.assertEqual(config["task"], "poker-draw")
        self.assertEqual(len(copies), 3)
        (self.root / "evaluation_right_wrist_state.png").unlink()
        config, copies = load_diagnostic_plots(self.root, expected_task="poker-draw")
        self.assertEqual(config["status"], "missing")
        self.assertEqual(copies, [])


if __name__ == "__main__":
    unittest.main()
