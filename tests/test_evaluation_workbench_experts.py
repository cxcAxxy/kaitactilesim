"""Expert attachments are explicit, immutable and independent of simulation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_workbench import experts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExpertBindingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.demo = self.root / "demo"
        self.demo.mkdir()
        self.video = self.demo / "declared.mp4"
        self.video.write_bytes(b"audited immutable video fixture")
        self.replay = self.demo / "recording/replay.json"
        self.frames = self.demo / "recording/frames.jsonl"
        self.source = "/explicit/source/episode.h5"
        self.metadata = {
            "schema_version": "kaihand-offline-multimodal-replay-v1",
            "completed": True, "task": "whiteboard-wipe", "source_hdf5": self.source,
            "source_sha256": "c" * 64, "source_episode_index": 0,
            "source_modified": False, "physics_rerun": False,
            "output_size": [1920, 1080], "fps": 10, "output_frame_count": 2,
            "video_duration_s": .2,
        }
        write_json(self.replay, self.metadata)
        self.frame_rows = [{"output_index": i, "playback_timestamp_s": i / 10,
                            "camera_pose_timestamp_s": i / 10} for i in range(2)]
        self.frames.write_text("\n".join(json.dumps(row) for row in self.frame_rows) + "\n")
        self.audit = {
            "task": "whiteboard-wipe", "video": "declared.mp4",
            "replay": "recording/replay.json", "frames": "recording/frames.jsonl",
            "video_sha256": sha(self.video), "replay_sha256": sha(self.replay),
            "frames_sha256": sha(self.frames), "source_hdf5": self.source,
            "source_sha256": "c" * 64, "fps": 10, "frame_count": 2, "publication": "",
        }
        self.registry_patch = patch.object(experts, "EXPERT_REGISTRY", {"whiteboard-wipe": self.audit})
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        self.probe = {"streams": [{"width": 1920, "height": 1080, "r_frame_rate": "10/1",
                                    "nb_frames": "2", "duration": "0.2"}],
                      "format": {"duration": "0.2"}}
        self.probe_patch = patch.object(experts, "_probe_video", return_value=self.probe)
        self.probe_mock = self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)
        self.trial, self.summary = self.make_trial("seed_000", 0)

    def make_trial(self, name, seed, *, source=None, task="whiteboard-wipe", nested=False):
        trial = self.root / name
        directory = trial / "review" if nested else trial
        reference = {"dataset_root": "/explicit/reference/dataset", "source_hdf5": source or self.source,
                     "output_episode_index": 0}
        comparison = {"schema": "kaihand-evaluation-comparison-v1", "status": "ok", "reference": reference}
        review = {"task": task, "seed": seed, "checkpoint_sha256": "a" * 64,
                  "completed": True, "comparison_plots": comparison}
        summary = {"task": task, "seed": seed, "checkpoint_sha256": "a" * 64,
                   "prediction_horizon": 50, "execute_steps": 50, "video": deepcopy(review)}
        write_json(trial / "summary.json", summary)
        write_json(directory / "review.json", review)
        write_json(directory / "evaluation_comparison.json", comparison)
        return trial, summary

    def prepare(self, trials=None):
        return experts.prepare_expert_bindings(trials or [self.trial], self.demo)

    def change_replay(self, field, value):
        self.metadata[field] = value
        write_json(self.replay, self.metadata)
        self.audit["replay_sha256"] = sha(self.replay)

    def test_exact_source_recording_and_copy_contract(self):
        result = experts.expert_for_case(self.summary, self.prepare())
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["reference_relation"], "same_recording")
        self.assertEqual(result["clock"], "independent_video_seconds")
        self.assertEqual(result["duration_s"], .2)
        self.assertEqual(result["source_sha256"], "c" * 64)
        self.assertEqual(result["copy_jobs"], [{"source": self.video, "url": "media/expert.mp4",
                                              "sha256": sha(self.video), "size": self.video.stat().st_size}])
        self.assertEqual(result["media_views"]["views"], [])
        self.assertFalse(Path(self.source).exists())
        self.assertIn("not reread or rehashed", result["source_verification"])

    def test_other_recording_is_explicit_not_the_png_blue_line(self):
        trial, summary = self.make_trial("other", 1, source="/other/recording.h5", nested=True)
        result = experts.expert_for_case(summary, self.prepare([trial]))
        self.assertEqual(result["reference_relation"], "same_task_other_recording")
        self.assertEqual(result["title"], "同任务专家示范")
        self.assertIn("与当前参考不同", result["source_label"])
        self.assertIn("不是 PNG 蓝线", result["comparison_note"])

    def test_missing_plot_reference_still_allows_explicit_same_task_demo(self):
        summary = deepcopy(self.summary)
        comparison = summary["video"]["comparison_plots"]
        comparison["reference"].pop("source_hdf5")
        comparison["status"] = "reference_unavailable"
        write_json(self.trial / "summary.json", summary)
        write_json(self.trial / "review.json", summary["video"])
        write_json(self.trial / "evaluation_comparison.json", comparison)
        result = experts.expert_for_case(summary, self.prepare())
        self.assertEqual(result["reference_relation"], "same_task_other_recording")
        self.assertIn("本页未绘制专家参考线", result["comparison_note"])
        self.assertNotIn("PNG 蓝线", result["comparison_note"])

    def test_staged_identity_does_not_relabel_another_expert_recording(self):
        trial, summary = self.make_trial("staged", 4, source="/new/staged/reference.h5")
        summary["staged_evaluation"] = {"version": "staged_v1", "task": "whiteboard-wipe", "seed": 4}
        write_json(trial / "summary.json", summary)
        result = experts.expert_for_case(summary, self.prepare([trial]))
        self.assertEqual(result["reference_relation"], "same_task_other_recording")
        self.assertEqual(result["clock"], "independent_video_seconds")
        self.assertIn("不进行阶段对齐", result["comparison_note"])
        self.assertEqual(result["source_hdf5"], self.source)

    def test_unknown_task_is_not_paired_by_filename(self):
        trial, summary = self.make_trial("unknown", 1, task="not-registered")
        result = experts.expert_for_case(summary, self.prepare([trial]))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["copy_jobs"], [])
        self.probe_mock.assert_not_called()

    def test_h30_or_wrong_execution_horizon_cannot_attach(self):
        for key in ("prediction_horizon", "execute_steps"):
            summary = deepcopy(self.summary)
            summary[key] = 30
            write_json(self.trial / "summary.json", summary)
            result = experts.expert_for_case(summary, self.prepare())
            self.assertEqual(result["status"], "unavailable")
        self.probe_mock.assert_not_called()

    def test_media_verified_once_but_trial_reference_witnesses_are_distinct(self):
        second, summary = self.make_trial("seed_001", 1)
        bindings = self.prepare([self.trial, self.trial, second])
        self.assertEqual(len(bindings), 2)
        self.assertEqual(self.probe_mock.call_count, 1)
        first = experts.expert_for_case(self.summary, bindings)
        last = experts.expert_for_case(summary, bindings)
        self.assertIn("seed_000", first["reference_binding_evidence"][0]["path"])
        self.assertIn("seed_001", last["reference_binding_evidence"][0]["path"])
        first["copy_jobs"].clear()
        self.assertEqual(len(experts.expert_for_case(self.summary, bindings)["copy_jobs"]), 1)

    def test_changed_reference_does_not_get_task_only_fallback(self):
        bindings = self.prepare()
        summary = deepcopy(self.summary)
        summary["video"]["comparison_plots"]["reference"]["source_hdf5"] = "/changed.h5"
        self.assertEqual(experts.expert_for_case(summary, bindings)["status"], "unavailable")

    def test_metadata_cross_file_mismatch_rejected(self):
        review = deepcopy(self.summary["video"])
        review["comparison_plots"]["reference"]["source_hdf5"] = "/different.h5"
        write_json(self.trial / "review.json", review)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare()

    def test_video_tampering_rejected(self):
        self.video.write_bytes(b"changed video")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.prepare()

    def test_same_basename_alternate_manifest_is_not_guessed(self):
        self.replay.rename(self.replay.with_name("same-content.json"))
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_replay_schema_task_source_and_physics_are_checked_after_hash(self):
        baseline = deepcopy(self.metadata)
        for key, value in [("schema_version", "unknown"), ("task", "poker-draw"),
                           ("source_hdf5", "/wrong.h5"), ("source_sha256", "d" * 64),
                           ("source_modified", True), ("physics_rerun", True),
                           ("output_size", [1280, 720]), ("completed", False)]:
            with self.subTest(key=key):
                self.metadata = deepcopy(baseline)
                self.change_replay(key, value)
                with self.assertRaisesRegex(ValueError, "differs from explicit audit"):
                    self.prepare()

    def test_encoded_dimensions_fps_frames_and_duration_are_verified(self):
        for key, value in [("width", 1280), ("r_frame_rate", "30/1"),
                           ("nb_frames", "3"), ("duration", "4.0")]:
            with self.subTest(key=key):
                probe = deepcopy(self.probe)
                probe["streams"][0][key] = value
                self.probe_mock.return_value = probe
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    self.prepare()

    def test_rehashed_bad_frame_clock_is_rejected(self):
        self.frame_rows[1]["output_index"] = 0
        self.frames.write_text("\n".join(json.dumps(row) for row in self.frame_rows))
        self.audit["frames_sha256"] = sha(self.frames)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self.prepare()

    def test_publication_must_identify_current_video_and_replay(self):
        self.audit["publication"] = "publication.json"
        publication = {"schema": "kaihand_task_demo_publication_v1", "task": "whiteboard-wipe",
                       "video_sha256": sha(self.video), "replay_sha256": sha(self.replay),
                       "canonical_video": str(self.video), "source_review_dir": str(self.replay.parent)}
        path = self.demo / "publication.json"
        write_json(path, publication)
        self.assertEqual(experts.expert_for_case(self.summary, self.prepare())["status"], "available")
        publication["replay_sha256"] = "0" * 64
        write_json(path, publication)
        with self.assertRaisesRegex(ValueError, "publication does not bind"):
            self.prepare()

    def test_symlink_and_registry_path_escape_are_rejected(self):
        target = self.video.with_name("target.mp4")
        self.video.rename(target)
        self.video.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            self.prepare()
        self.audit["video"] = "../outside.mp4"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.prepare()

    def test_model_views_never_guess_whiteboard_crop(self):
        views = experts.model_views_for_case({"video": {"output_size": [1280, 720]}}, "a" * 64)
        self.assertEqual(views["views"], [])
        self.assertEqual(views["source_width"], 1280)
        with self.assertRaises(ValueError):
            experts.model_views_for_case(self.summary, "unbound")

    def test_audited_expert_crop_requires_exact_video_and_first_frame(self):
        first = self.replay.with_name("first_frame.png")
        first.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
                          + (1920).to_bytes(4, "big") + (1080).to_bytes(4, "big"))
        self.change_replay("camera_names", ["head", "right_wrist"])
        crop = {sha(self.video): {"first_frame": "recording/first_frame.png",
                                  "first_frame_sha256": sha(first)}}
        with patch.object(experts, "_EXPERT_CROP_AUDIT", crop):
            result = experts.expert_for_case(self.summary, self.prepare())
            self.assertEqual(result["media_views"]["status"], "verified")
            self.assertEqual(result["media_views"]["views"][0],
                             {"id": "head", "label": "主视角", "x": 10, "y": 108,
                              "width": 632, "height": 456})
            self.assertEqual(result["provenance"][-1]["sha256"], sha(first))
            first.write_bytes(b"changed first frame")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.prepare()

    def test_wrong_expert_camera_layout_does_not_reuse_crop(self):
        first = self.replay.with_name("first_frame.png")
        first.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
                          + (1920).to_bytes(4, "big") + (1080).to_bytes(4, "big"))
        self.change_replay("camera_names", ["global", "head"])
        crop = {sha(self.video): {"first_frame": "recording/first_frame.png",
                                  "first_frame_sha256": sha(first)}}
        with patch.object(experts, "_EXPERT_CROP_AUDIT", crop):
            with self.assertRaisesRegex(ValueError, "geometry differs"):
                self.prepare()

    def test_genesis_probe_semantics_are_not_presented_as_force(self):
        self.change_replay("tactile_source_kind", "genesis_probe")
        self.change_replay("curve_unit", "Genesis local proxy; not N")
        self.change_replay("tactile_heatmaps", [{"label": "Depth", "unit": "mm"},
                                               {"label": "Contact", "unit": "0/1"}])
        result = experts.expert_for_case(self.summary, self.prepare())
        self.assertEqual(result["curve_unit"], "Genesis local proxy; not N")
        self.assertIn("不是 Fn/Ft（N）", result["comparison_note"])
        self.assertEqual(result["tactile_heatmaps"][0]["unit"], "mm")

    def model_crop_fixture(self):
        renderer = self.root / "audited_renderer.py"
        renderer.write_bytes(b"fixed audited 3-camera renderer fixture")
        rows = [{"path": experts._MODEL_RENDERER_PATH, "size": renderer.stat().st_size,
                 "sha256": sha(renderer)}]
        inventory = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = self.root / "model_manifest.json"
        write_json(manifest, {"task": "whiteboard-wipe", "checkpoint_sha256": "a" * 64,
                              "deployment_id": "explicit deployment", "prediction_horizon": 50,
                              "evaluation_source_files": rows, "evaluation_source_sha256": inventory})
        summary = deepcopy(self.summary)
        summary.update(deployment_manifest=str(manifest), deployment_id="explicit deployment",
                       deployment_integrity={"status": "verified", "manifest_sha256": sha(manifest),
                                             "evaluation_source": {"status": "verified",
                                                                   "expected_sha256": inventory,
                                                                   "actual_sha256": inventory}})
        summary["video"].update(schema="kaihand-policy-evaluation-review-v3", layout=experts._MODEL_LAYOUT,
                                output_size=[1920, 1080], review_render_size=[640, 480],
                                model_input_cameras_displayed=["head", "right_wrist"],
                                review_only_cameras_displayed=["global"], right_wrist_camera_role="model_input",
                                fps=10, frame_count=2)
        return summary, renderer, manifest

    def test_model_crop_is_bound_to_renderer_snapshot_manifest_and_video_sha(self):
        summary, renderer, manifest = self.model_crop_fixture()
        with patch.object(experts, "_MODEL_RENDERER_AUDIT", {sha(renderer): str(renderer)}):
            result = experts.model_views_for_case(summary, "f" * 64)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["sha256"], "f" * 64)
            self.assertEqual(result["views"][0]["width"], 434)
            self.assertEqual([view["id"] for view in result["views"]], ["head", "wrist", "global"])
            self.assertEqual(result["views"][1], {"id": "wrist", "label": "右腕视角", "x": 214,
                                                  "y": 440, "width": 434, "height": 294})
            self.assertEqual(result["renderer_sha256"], sha(renderer))
            self.assertEqual(result["provenance"][0]["sha256"], sha(manifest))
            renderer.write_bytes(b"changed renderer")
            self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["views"], [])

    def test_model_crop_falls_back_on_layout_or_manifest_tamper(self):
        summary, renderer, manifest = self.model_crop_fixture()
        with patch.object(experts, "_MODEL_RENDERER_AUDIT", {sha(renderer): str(renderer)}):
            wrong = deepcopy(summary)
            wrong["video"]["model_input_cameras_displayed"] = ["head"]
            self.assertEqual(experts.model_views_for_case(wrong, "f" * 64)["views"], [])
            wrong = deepcopy(summary)
            wrong["video"]["right_wrist_camera_role"] = "review_only"
            self.assertEqual(experts.model_views_for_case(wrong, "f" * 64)["views"], [])
            manifest.write_text("{}")
            self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["views"], [])

    def test_staged_crop_uses_its_own_bound_frozen_renderer_not_observer_name(self):
        summary, renderer, manifest = self.model_crop_fixture()
        project = self.root / "staged_snapshot"
        frozen_renderer = project / experts._MODEL_RENDERER_PATH
        frozen_renderer.parent.mkdir(parents=True)
        frozen_renderer.write_bytes(renderer.read_bytes())
        deployment = json.loads(manifest.read_text())
        deployment["evaluation_project_path"] = str(project)
        deployment["evaluation_source_files"].append({"path": "shared/staged_evaluation.py",
                                                      "size": 1, "sha256": "7" * 64})
        inventory = hashlib.sha256(json.dumps(deployment["evaluation_source_files"], sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
        deployment["evaluation_source_sha256"] = inventory
        write_json(manifest, deployment)
        summary["deployment_integrity"].update(manifest_sha256=sha(manifest))
        summary["deployment_integrity"]["evaluation_source"].update(root=str(project), expected_sha256=inventory,
                                                                    actual_sha256=inventory)
        summary["staged_evaluation"] = {"version": "staged_v1"}
        with patch.object(experts, "_MODEL_RENDERER_AUDIT", {sha(renderer): str(renderer)}):
            result = experts.model_views_for_case(summary, "f" * 64)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["provenance"][1]["path"], str(frozen_renderer))
            self.assertEqual([view["id"] for view in result["views"]], ["head", "wrist", "global"])
            frozen_renderer.write_bytes(b"different layout despite same staged task name")
            self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["status"], "full_frame_only")

    def test_model_crop_missing_legacy_manifest_and_unknown_renderer_keep_mosaic(self):
        summary, renderer, manifest = self.model_crop_fixture()
        with patch.object(experts, "_MODEL_RENDERER_AUDIT", {}):
            self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["views"], [])
        manifest.unlink()
        self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["views"], [])

    def legacy_model_crop_fixture(self):
        summary, renderer, _ = self.model_crop_fixture()
        summary.pop("deployment_integrity")
        summary["deployment_manifest"] = str(self.root / "missing_old_manifest.json")
        summary["task"] = summary["video"]["task"] = "pick-place"
        summary["video"]["deployment_id"] = summary["deployment_id"]
        source_summary, review = self.root / "legacy_summary.json", self.root / "legacy_review.json"
        write_json(source_summary, summary)
        write_json(review, summary["video"])
        first = self.root / "legacy_first.png"
        first.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR"
                          + (1920).to_bytes(4, "big") + (1080).to_bytes(4, "big"))
        protocol, sheet = self.root / "legacy_protocol.json", self.root / "sheet.png"
        sheet.write_bytes(b"persistent visual audit fixture")
        write_json(protocol, {"task": "pick-place", "checkpoint_sha256": summary["checkpoint_sha256"],
                              "deployment_id": summary["deployment_id"], "prediction_horizon": 50,
                              "execute_steps": 50, "seeds": [0],
                              "source_hashes": {experts._MODEL_RENDERER_PATH: sha(renderer)},
                              "review": {"fps": 10, "resolution": [1920, 1080], "second_camera": "global"}})
        def witness(path):
            return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}
        record = {**{k: summary[k] for k in ("task", "seed", "checkpoint_sha256", "deployment_id")},
                  "contract": {k: summary["video"].get(k) for k in experts._MODEL_CONTRACT_KEYS},
                  "pane_label_signatures_verified": True, "decoded_vs_saved_first_rgb_mae": 1.2,
                  "summary": witness(source_summary), "review": witness(review), "first_frame": witness(first),
                  "video": witness(self.video), "protocol": witness(protocol)}
        audit = {"schema": "evaluation-workbench-model-crop-audit-v1", "views": deepcopy(experts._MODEL_VIEWS),
                 "representative_visual_review": "passed", "renderer": witness(renderer),
                 "contact_sheet": witness(sheet), "recordings": {sha(self.video): record}}
        path = self.root / "legacy_crop_audit.json"
        write_json(path, audit)
        for name, value in [("_MODEL_LEGACY_AUDIT_PATH", path), ("_MODEL_LEGACY_AUDIT_SHA256", sha(path)),
                            ("_MODEL_RENDERER_AUDIT", {sha(renderer): str(renderer)})]:
            mock = patch.object(experts, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        return summary, audit, path

    def test_registered_legacy_media_gets_wrist_without_missing_manifest(self):
        summary, audit, path = self.legacy_model_crop_fixture()
        result = experts.model_views_for_case(summary, sha(self.video))
        self.assertEqual(result["status"], "verified")
        self.assertEqual([v["id"] for v in result["views"]], ["head", "wrist", "global"])
        self.assertEqual(result["views"][1]["height"], 294)
        self.assertEqual(len(result["provenance"]), 8)
        self.assertTrue(all(row["role"].startswith("model_crop_") for row in result["provenance"]))
        self.assertEqual(result["provenance"][0]["sha256"], sha(path))
        self.assertEqual(experts.model_views_for_case(summary, "f" * 64)["views"], [])

    def test_legacy_media_cannot_be_relabelled_for_another_seed_or_checkpoint(self):
        summary, _, _ = self.legacy_model_crop_fixture()
        for key, value in [("seed", 1), ("checkpoint_sha256", "9" * 64), ("deployment_id", "different")]:
            altered = deepcopy(summary)
            altered[key] = value
            self.assertEqual(experts.model_views_for_case(altered, sha(self.video))["views"], [])

    def test_legacy_crop_rechecks_all_source_files_and_rejects_tamper(self):
        summary, audit, _ = self.legacy_model_crop_fixture()
        record = audit["recordings"][sha(self.video)]
        targets = [record[key] for key in ("summary", "review", "first_frame", "video", "protocol")]
        targets += [audit["renderer"], audit["contact_sheet"]]
        original_video_sha = sha(self.video)
        for witness in targets:
            path = Path(witness["path"])
            before = path.read_bytes()
            with self.subTest(path=path.name):
                try:
                    path.write_bytes(b"tampered")
                    self.assertEqual(experts.model_views_for_case(summary, original_video_sha)["views"], [])
                finally:
                    path.write_bytes(before)

    def test_legacy_rehashed_protocol_cannot_claim_unknown_renderer(self):
        summary, audit, path = self.legacy_model_crop_fixture()
        record = audit["recordings"][sha(self.video)]
        protocol_path = Path(record["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text())
        protocol["source_hashes"][experts._MODEL_RENDERER_PATH] = "9" * 64
        write_json(protocol_path, protocol)
        record["protocol"].update(sha256=sha(protocol_path), size=protocol_path.stat().st_size)
        write_json(path, audit)
        with patch.object(experts, "_MODEL_LEGACY_AUDIT_SHA256", sha(path)):
            self.assertEqual(experts.model_views_for_case(summary, sha(self.video))["views"], [])

    def test_legacy_audit_digest_and_geometry_are_pinned(self):
        summary, audit, path = self.legacy_model_crop_fixture()
        path.write_text("{}")
        self.assertEqual(experts.model_views_for_case(summary, sha(self.video))["views"], [])
        audit["views"][1]["y"] = 0
        write_json(path, audit)
        with patch.object(experts, "_MODEL_LEGACY_AUDIT_SHA256", sha(path)):
            self.assertEqual(experts.model_views_for_case(summary, sha(self.video))["views"], [])


if __name__ == "__main__":
    unittest.main()
