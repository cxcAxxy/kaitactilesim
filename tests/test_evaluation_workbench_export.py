"""Workbench exports preserve payload bytes and fail before publishing entries."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_workbench import export
from evaluation_workbench.diagnostic_plots import load_diagnostic_plots
from export_evaluation_workbench import main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def digest(path):
    raw = path.read_bytes()
    return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def diagnostic_fixture(trial, task="whiteboard-wipe"):
    trial.mkdir(parents=True, exist_ok=True)
    def chunk(kind, raw):
        return struct.pack(">I", len(raw)) + kind + raw + struct.pack(">I", zlib.crc32(kind + raw) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))
    artifacts = {}
    for key in ("right_fingertip_tactile", "right_hand_actuated_dof", "right_wrist_state"):
        path = trial / f"evaluation_{key}.png"
        path.write_bytes(png)
        artifacts[key] = str(path)
    shared = {"styles": {"reference": "solid", "evaluation_rollout": "dashed",
                          "time_axis": "elapsed seconds; each sequence starts at zero"},
              "reference": {"dataset_root": "/explicit/reference", "source_hdf5": "/explicit/reference/episode.h5",
                            "output_episode_index": 0, "samples": 2},
              "rollout": {"samples": 2, "source_start_time_s": 0, "duration_s": 1},
              "state": {"layout": "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof", "dimension": 29,
                        "wrist_site": "hand_r_base_link_site", "right_hand_actuated_dof": 20,
                        "rot6d": "first two columns", "joint_names": [f"joint_{i}" for i in range(20)]},
              "tactile": {"finger_order": ["thumb", "index", "middle", "ring", "little"],
                          "link_names": [f"finger_{i}" for i in range(5)], "taxel_grid": [7, 5],
                          "Fn": "sum positive normals", "unit": "N"}}
    comparison = {**deepcopy(shared), "schema": "kaihand-evaluation-comparison-v1", "status": "ok",
                  "artifacts": {**artifacts, "metadata": str(trial / "evaluation_comparison.json")}}
    plots = {**deepcopy(shared), "schema": "kaihand-openwam-evaluation-plots-v1",
             "artifacts": {**artifacts, "metadata": str(trial / "evaluation_evaluation_plots.json")}}
    review = {"task": task, "seed": 0, "completed": True, "checkpoint_sha256": "a" * 64,
              "comparison_plots": comparison}
    summary = {"task": task, "seed": 0, "prediction_horizon": 50, "status": "task_not_completed",
               "checkpoint_path": "/checkpoint/" + ("20000" if task == "bulb-unscrew" else "24999"),
               "checkpoint_sha256": "a" * 64,
               "video": {**review, "reference_dataset": "/explicit/reference", "reference_episode_index": 0}}
    for name, value in (("summary.json", summary), ("review.json", review), ("evaluation_comparison.json", comparison),
                        ("evaluation_evaluation_plots.json", plots)):
        write_json(trial / name, value)
    return summary


class WorkbenchExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / "tactile_r2"
        self.case = self.site / "009_case"
        self.stable = self.root / "stable_h50"
        self.output = self.root / "workbench"
        self.original = self.root / "old/009_case/report.json"
        self.h50 = self.stable / "004_case/report.json"
        self.source = self.root / "raw/seed_000"
        self.oldcode = self.root / "old_code"
        self.web = self.root / "new_web"
        for folder in (self.case / "media", self.site / "assets", self.oldcode, self.web):
            folder.mkdir(parents=True)
        self.video = b"fixture video with audited crop layout"
        self.video_sha = hashlib.sha256(self.video).hexdigest()
        (self.case / "media/rollout.mp4").write_bytes(self.video)
        (self.oldcode / "old.py").write_text("# old source\n")
        for name in ("workbench.js", "workbench.css"):
            (self.web / name).write_text("/* new workbench fixture */")
        for name in ("dashboard.js", "dashboard.css", "bootstrap.js", "tactile_loads.js", "tactile_loads.css"):
            (self.site / "assets" / name).write_text("original " + name)
        (self.site / "index.html").write_text("original site entry")
        (self.case / "audit.html").write_text("original audit")
        write_json(self.source / "summary.json", {"task": "whiteboard-wipe", "prediction_horizon": 50})
        self.report = {"schema_version": "task-tactile-evaluation-v1", "suite_schema_version": "task-tactile-suite-v2",
                       "trial": {"task": "whiteboard-wipe", "trial_id": "same-trial", "source": str(self.source)}}
        self.visual = {"schema_version": "evaluation-visualization-v1", "time_origin_s": 0,
                       "tactile_series": {"time_s": [0, .01], "normal_force_n": [[1] * 5, [2] * 5]},
                       "media": {"status": "available", "sync_status": "mapped", "url": "media/rollout.mp4",
                                 "sha256": self.video_sha}}
        write_json(self.original, self.report)
        write_json(self.original.parent / "visualization.json", self.visual)
        stable_report = deepcopy(self.report)
        stable_report["trial"]["prediction_horizon"] = 50
        write_json(self.h50, stable_report)
        write_json(self.h50.parent / "visualization.json", self.visual)
        write_json(self.stable / "batch.json", {"horizon_selection": {"required_prediction_horizon": 50},
                    "trials": [{"prediction_horizon": 50, "report_path": "004_case/report.html"}]})
        (self.stable / "index.html").write_text("stable H50 entry")
        (self.case / "report.json").write_bytes(self.original.read_bytes())
        (self.case / "visualization.json").write_bytes((self.original.parent / "visualization.json").read_bytes())
        self.loads = {"schema_version": "evaluation-five-finger-load-extension-v1", "prediction_horizon": 50,
                      "trial_id": "same-trial", "task": "whiteboard-wipe", "source_report": str(self.original),
                      "h50_report": str(self.h50), "time_s": [0, .01], "normal_total_n": [5, 10],
                      "tangent_total_n": [3, 4], "implementation": [digest(self.oldcode / "old.py")],
                      "provenance": [digest(path) for path in (self.original, self.original.parent / "visualization.json",
                                     self.h50, self.h50.parent / "visualization.json", self.source / "summary.json")]}
        write_json(self.case / "tactile_loads.json", self.loads)
        self.write_page()
        self.inherited = {"schema_version": "evaluation-experiment-rollback-v1", "stable_site": str(self.stable),
                          "protected_files": [digest(self.oldcode / "old.py"), digest(self.stable / "batch.json")]}
        write_json(self.site / "rollback.json", self.inherited)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(export, "_WEB_DIR", self.web))
        self.stack.enter_context(patch.object(export, "_MEDIA_SHA", self.video_sha))
        self.stack.enter_context(patch.object(export, "_old_code_roots", return_value=[self.oldcode]))
        self.stack.enter_context(patch.object(export, "_old_code_files", return_value=[]))
        self.stack.enter_context(patch.object(export, "_implementation", side_effect=lambda: [digest(path) for path in sorted(self.web.iterdir())]))

    def write_page(self, *, report=None, visual=None, loads=None):
        payload = json.dumps({"report": self.report if report is None else report,
                              "visualization": self.visual if visual is None else visual}, ensure_ascii=False)
        load_payload = json.dumps(self.loads if loads is None else loads, ensure_ascii=False)
        (self.case / "report.html").write_text(
            '<!doctype html><html><head><link rel="stylesheet" href="../assets/dashboard.css">'
            '<script src="../assets/dashboard.js" defer></script><script src="../assets/bootstrap.js" defer></script>'
            '<link rel="stylesheet" href="../assets/tactile_loads.css"></head><body><div id="kai-evaluation"></div>'
            f'<script id="kai-evaluation-data" type="application/json">{payload}</script>'
            f'<script type="application/json" id="kai-tactile-load-data">{load_payload}</script>'
            '<script src="../assets/tactile_loads.js" defer></script></body></html>', encoding="utf-8")

    def export(self, source=None, output=None):
        return export.export_workbench(source or self.case / "report.html", output or self.output)

    def snapshot(self):
        return {str(path): path.read_bytes() for root in (self.site, self.stable, self.oldcode)
                for path in root.rglob("*") if path.is_file()}

    def test_payloads_and_all_nonentry_artifacts_are_byte_identical(self):
        before = self.snapshot()
        config = self.export()
        self.assertEqual(self.snapshot(), before)
        page = (self.output / "009_case/report.html").read_text()
        original = (self.case / "report.html").read_text()
        for identifier in ("kai-evaluation-data", "kai-tactile-load-data"):
            self.assertEqual(export._embedded(page, identifier)[0], export._embedded(original, identifier)[0])
        for path in self.case.rglob("*"):
            if path.is_file() and path.name != "report.html":
                self.assertEqual(path.read_bytes(), (self.output / "009_case" / path.relative_to(self.case)).read_bytes())
        for path in (self.site / "assets").iterdir():
            self.assertEqual(path.read_bytes(), (self.output / "assets" / path.name).read_bytes())
        self.assertNotIn('src="../assets/tactile_loads.js"', page)
        self.assertNotIn('href="../assets/tactile_loads.css"', page)
        self.assertIn('src="../assets/dashboard.js"', page)
        self.assertIn('src="../assets/bootstrap.js"', page)
        self.assertIn('src="../assets/workbench.js"', page)
        self.assertGreater(page.index('src="../assets/workbench.js"'), page.index('id="kai-tactile-load-data"'))
        self.assertEqual(config, export._embedded(page, "kai-workbench-config")[1])
        self.assertEqual(config["baseline_url"], "../../tactile_r2/009_case/report.html")
        self.assertEqual(config["stable_url"], "../../stable_h50/index.html")
        self.assertEqual(config["media_views"]["views"][1]["height"], 294)
        self.assertTrue(export.verify_rollback(self.output / "rollback.json")["verified"])
        self.assertTrue((self.output / "index.html").is_file())
        self.assertFalse(json.loads((self.output / "verification.json").read_text())["fn_ft_recomputed"])

    def test_json_input_supported_without_recomputing_loads(self):
        with patch("evaluation_tactile_load_extension.data.load_extension", side_effect=AssertionError("must not recompute")):
            result = self.export(self.case / "report.json")
        self.assertEqual(result["trial_id"], "same-trial")

    def test_existing_nested_and_symlink_destinations_rejected(self):
        with self.assertRaises(FileExistsError): self.export(output=self.site)
        with self.assertRaises(ValueError): self.export(output=self.site / "nested")
        alias = self.root / "alias"
        alias.symlink_to(self.site, target_is_directory=True)
        with self.assertRaises(FileExistsError): self.export(output=alias)

    def test_symlinks_in_source_are_rejected(self):
        (self.site / "linked.txt").symlink_to(self.oldcode / "old.py")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_each_embedded_payload_must_match_json(self):
        for key in ("report", "visual", "loads"):
            with self.subTest(key=key):
                self.write_page(**{key: {"different": True}})
                with self.assertRaisesRegex(ValueError, "payload differs"):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_inherited_protected_hash_failure_stops_before_output(self):
        (self.oldcode / "old.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_bad_source_provenance_and_horizon_rejected(self):
        self.loads["provenance"][0]["sha256"] = "a" * 64
        write_json(self.case / "tactile_loads.json", self.loads)
        self.write_page()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.export()
        self.loads["prediction_horizon"] = 30
        write_json(self.case / "tactile_loads.json", self.loads)
        self.write_page()
        with self.assertRaisesRegex(ValueError, "H50"):
            self.export()

    def test_unverified_video_cannot_use_hardcoded_crops(self):
        (self.case / "media/rollout.mp4").write_bytes(b"different video")
        with self.assertRaisesRegex(ValueError, "Video SHA256"):
            self.export()

    def test_wrong_identity_in_load_payload_rejected(self):
        self.loads["trial_id"] = "another-trial"
        write_json(self.case / "tactile_loads.json", self.loads)
        self.write_page()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.export()

    def test_legacy_addon_must_exist_once(self):
        path = self.case / "report.html"
        path.write_text(path.read_text().replace('<script src="../assets/tactile_loads.js" defer></script>', ""))
        with self.assertRaisesRegex(ValueError, "legacy tactile addon"):
            self.export()

    def test_final_verification_failure_publishes_no_entries(self):
        original = export._verify
        def fail_after_copy(rows):
            if self.output.exists():
                raise ValueError("changed at final verification")
            return original(rows)
        with patch.object(export, "_verify", side_effect=fail_after_copy), self.assertRaisesRegex(ValueError, "final verification"):
            self.export()
        self.assertFalse((self.output / "009_case/report.html").exists())
        self.assertFalse((self.output / "index.html").exists())

    def test_copied_data_tampering_publishes_no_entries(self):
        original = export._copy_checked
        def tamper(source, destination, row):
            original(source, destination, row)
            if destination.name == "tactile_loads.json":
                destination.write_text("{}")
        with patch.object(export, "_copy_checked", side_effect=tamper), self.assertRaisesRegex(ValueError, "Copied report"):
            self.export()
        self.assertFalse((self.output / "009_case/report.html").exists())
        self.assertFalse((self.output / "index.html").exists())

    def test_rollback_rechecks_added_source_files_as_well_as_hashes(self):
        self.export()
        (self.site / "unexpected.txt").write_text("new mutation")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            export.verify_rollback(self.output / "rollback.json")

    def test_previously_bound_source_alias_keeps_payload_and_pins_target(self):
        original_source = self.source
        canonical = self.root / "reorganized/seed_000"
        canonical.parent.mkdir()
        original_source.rename(canonical)
        original_source.symlink_to(canonical, target_is_directory=True)
        self.export()
        manifest = json.loads((self.output / "rollback.json").read_text())
        self.assertEqual(len(manifest["bound_source_aliases"]), 1)
        self.assertEqual(manifest["bound_source_aliases"][0]["resolved_path"], str(canonical / "summary.json"))
        self.assertTrue(export.verify_rollback(self.output / "rollback.json")["verified"])
        alternate = self.root / "other/seed_000"
        alternate.mkdir(parents=True)
        (alternate / "summary.json").write_bytes((canonical / "summary.json").read_bytes())
        original_source.unlink()
        original_source.symlink_to(alternate, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "alias target changed"):
            export.verify_rollback(self.output / "rollback.json")

    def test_bound_leaf_symlink_is_still_rejected(self):
        original = self.source / "summary.json"
        target = self.source / "summary-target.json"
        original.rename(target)
        original.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Leaf symlink"):
            self.export()

    def test_verify_cli_and_argument_exclusion(self):
        self.export()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--verify-rollback", str(self.output / "rollback.json")]), 0)
        self.assertTrue(json.loads(output.getvalue())["verified"])
        with self.assertRaises(SystemExit):
            main(["--verify-rollback", str(self.output / "rollback.json"), "--output", str(self.output)])

    def test_payload_escape_cannot_add_executable_script(self):
        html = (self.case / "report.html").read_text()
        rendered = export._render(html, {"note": "</script><script>bad()</script>"})
        self.assertNotIn("</script><script>bad()", rendered)
        self.assertEqual(export._embedded(rendered, "kai-workbench-config")[1]["note"], "</script><script>bad()</script>")

    def output_snapshot(self):
        return {str(path.relative_to(self.output)): path.read_bytes() for path in self.output.rglob("*") if path.is_file()}

    def change_web(self):
        (self.web / "workbench.js").write_text("/* updated independent workbench JS */")
        (self.web / "workbench.css").write_text("/* updated independent workbench CSS */")

    def test_refresh_updates_only_explicit_three_file_allowlist(self):
        self.export()
        protected_before, before = self.snapshot(), self.output_snapshot()
        self.change_web()
        result = export.refresh_web(self.output)
        after = self.output_snapshot()
        self.assertEqual(set(before), set(after))
        changed = {path for path in before if before[path] != after[path]}
        self.assertEqual(changed, {"assets/workbench.js", "assets/workbench.css", "verification.json"})
        self.assertEqual(self.snapshot(), protected_before)
        self.assertEqual((self.output / "assets/workbench.js").read_bytes(), (self.web / "workbench.js").read_bytes())
        self.assertEqual(result["web_refresh_count"], 1)
        self.assertTrue(result["payloads_unchanged"])
        self.assertTrue(export.verify_rollback(self.output / "rollback.json")["verified"])
        audit = json.loads((self.output / "verification.json").read_text())
        self.assertEqual(audit["implementation"], [digest(path) for path in sorted(self.web.iterdir())])
        self.assertEqual(export.refresh_web(self.output)["web_refresh_count"], 2)

    def test_refresh_rejects_original_sites_and_unknown_root(self):
        self.export()
        before = self.snapshot()
        for path in (self.site, self.stable, self.root / "missing"):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                export.refresh_web(path)
        self.assertEqual(self.snapshot(), before)

    def test_refresh_rejects_wrong_config_and_changed_payload(self):
        self.export()
        self.change_web()
        config_path = self.output / "workbench.json"
        config = json.loads(config_path.read_text())
        config["trial_id"] = "another-trial"
        write_json(config_path, config)
        before = self.output_snapshot()
        with self.assertRaisesRegex(ValueError, "config identity"):
            export.refresh_web(self.output)
        self.assertEqual(self.output_snapshot(), before)
        config["trial_id"] = "same-trial"
        write_json(config_path, config)
        (self.output / "009_case/tactile_loads.json").write_text("{}")
        before = self.output_snapshot()
        with self.assertRaisesRegex(ValueError, "payload differs"):
            export.refresh_web(self.output)
        self.assertEqual(self.output_snapshot(), before)

    def test_refresh_protected_source_failure_changes_no_output(self):
        self.export()
        self.change_web()
        before = self.output_snapshot()
        (self.oldcode / "old.py").write_text("changed original")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export.refresh_web(self.output)
        self.assertEqual(self.output_snapshot(), before)

    def test_refresh_symlinked_asset_is_not_followed(self):
        self.export()
        destination = self.output / "assets/workbench.js"
        destination.unlink()
        destination.symlink_to(self.oldcode / "old.py")
        original = (self.oldcode / "old.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "Symlink"):
            export.refresh_web(self.output)
        self.assertEqual((self.oldcode / "old.py").read_bytes(), original)

    def test_refresh_commit_failure_restores_all_previous_bytes(self):
        self.export()
        self.change_web()
        before = self.output_snapshot()
        original, count = os.replace, 0
        def fail_second(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated second replacement failure")
            return original(source, destination)
        with patch.object(export.os, "replace", side_effect=fail_second), self.assertRaisesRegex(OSError, "second replacement"):
            export.refresh_web(self.output)
        self.assertEqual(self.output_snapshot(), before)
        self.assertFalse(list(self.output.glob(".workbench-refresh-*")))

    def test_refresh_source_web_change_while_staging_fails_before_commit(self):
        self.export()
        self.change_web()
        before = self.output_snapshot()
        original = export._copy_checked
        def mutate(source, destination, row):
            original(source, destination, row)
            if source == self.web / "workbench.js":
                source.write_text("changed while staging")
        with patch.object(export, "_copy_checked", side_effect=mutate), self.assertRaisesRegex(ValueError, "SHA256"):
            export.refresh_web(self.output)
        self.assertEqual(self.output_snapshot(), before)

    def test_refresh_cli_uses_same_existing_url(self):
        self.export()
        self.change_web()
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["--refresh-web", str(self.output)]), 0)
        result = json.loads(captured.getvalue())
        self.assertEqual(result["output"], str(self.output))
        self.assertTrue(result["refreshed"])
        with self.assertRaises(SystemExit):
            main(["--refresh-web", str(self.output), "--verify-rollback", str(self.output / "rollback.json")])

    def expert_fixture(self):
        self.export()
        directory = self.root / "audited_expert"
        directory.mkdir()
        manifest = directory / "explicit-replay.json"
        video = directory / "explicitly-bound-video.mp4"
        video.write_bytes(b"audited expert video fixture")
        report = directory / "reference/report.json"
        membership = directory / "reference_manifest.json"
        hdf = str(directory / "large-hdf-not-read.h5")
        sha = "f" * 64
        self.expert_replay = {"schema_version": "kaihand-offline-multimodal-replay-v1", "task": "whiteboard-wipe",
                             "completed": True, "source_modified": False, "physics_rerun": False,
                             "source_hdf5": hdf, "source_sha256": sha, "output_size": [1920, 1080],
                             "output_frame_count": 669, "fps": 10.0, "video_duration_s": 66.9}
        write_json(manifest, self.expert_replay)
        self.expert_reference = {"trial": {"task": "whiteboard-wipe", "trial_id": "explicit-expert", "source": hdf},
            "reference_import": {"status": "imported", "source_schema": "kaihand_tactile_episode_v1", "source_sha256_verified": True},
            "provenance": [{**digest(manifest), "role": "explicit_task_demo_replay_selection"},
                           {"path": hdf, "sha256": sha, "role": "expert_hdf5"}]}
        write_json(report, self.expert_reference)
        self.expert_membership = {"schema_version": "conditional-reference-manifest-v1", "role": "expert_reference",
            "declared_by": "fixture curator", "selection_rule": "one success-selected scripted task-demo example",
            "evidence": "explicit replay/HDF declaration", "members": [{"trial_id": "explicit-expert", "source": hdf, "observation_sha256": sha}]}
        write_json(membership, self.expert_membership)
        self.expert_audit = {"manifest": digest(manifest), "video": digest(video), "report": digest(report),
                             "membership": digest(membership), "source_hdf5": hdf, "source_sha256": sha,
                             "trial_id": "explicit-expert"}
        self.stack.enter_context(patch.object(export, "_EXPERT_AUDIT", self.expert_audit))
        self.expert_manifest = manifest
        self.expert_output = self.root / "expert_workbench"
        return manifest

    def export_expert(self):
        return export.export_workbench(self.case / "report.html", self.expert_output,
                                      expert_manifest=self.expert_manifest, previous_workbench=self.output)

    def test_expert_is_new_video_only_and_previous_workbench_is_protected(self):
        self.expert_fixture()
        previous = self.output_snapshot()
        baseline = self.snapshot()
        config = self.export_expert()
        self.assertEqual(self.output_snapshot(), previous)
        self.assertEqual(self.snapshot(), baseline)
        expert = config["expert_video"]
        self.assertEqual(expert["clock"], "independent_video_seconds")
        self.assertEqual(expert["duration_s"], 66.9)
        self.assertEqual(expert["url"], "media/expert.mp4")
        self.assertIn("初始墨迹与模型不同", expert["comparison_note"])
        self.assertEqual(expert["media_views"]["views"], [{"id": "head", "label": "头部视角", "x": 10, "y": 108, "width": 632, "height": 456}])
        self.assertEqual(config["previous_workbench_url"], "../../workbench/009_case/report.html")
        self.assertEqual((self.expert_output / "009_case/media/expert.mp4").read_bytes(), Path(self.expert_audit["video"]["path"]).read_bytes())
        new_html = (self.expert_output / "009_case/report.html").read_text()
        old_html = (self.case / "report.html").read_text()
        for identifier in ("kai-evaluation-data", "kai-tactile-load-data"):
            self.assertEqual(export._embedded(new_html, identifier)[0], export._embedded(old_html, identifier)[0])
        rollback = json.loads((self.expert_output / "rollback.json").read_text())
        roots = {entry["path"] for entry in rollback["protected_roots"]}
        self.assertIn(str(self.output), roots)
        paths = {row["path"] for row in rollback["protected_files"]}
        self.assertTrue(all(str(self.output / path) in paths for path in previous))
        self.assertTrue(export.verify_rollback(self.expert_output / "rollback.json")["verified"])
        self.assertIn("观察台 · 独立专家回放", (self.expert_output / "index.html").read_text())
        # No HDF file was even created; only its archived identity was checked.
        self.assertFalse(Path(self.expert_audit["source_hdf5"]).exists())

    def test_expert_requires_explicit_previous_site_and_path_binding(self):
        self.expert_fixture()
        with self.assertRaisesRegex(ValueError, "supplied together"):
            export.export_workbench(self.case / "report.json", self.expert_output, expert_manifest=self.expert_manifest)
        alternate = self.expert_manifest.with_name("same-name-does-not-pair.json")
        alternate.write_bytes(self.expert_manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "manifest path"):
            export.export_workbench(self.case / "report.json", self.expert_output,
                                   expert_manifest=alternate, previous_workbench=self.output)
        self.assertFalse(self.expert_output.exists())

    def test_expert_wrong_video_or_manifest_hash_fails_before_creation(self):
        self.expert_fixture()
        video = Path(self.expert_audit["video"]["path"])
        original = video.read_bytes()
        video.write_bytes(b"wrong expert")
        with self.assertRaisesRegex(ValueError, "video SHA256"):
            self.export_expert()
        video.write_bytes(original)
        self.expert_manifest.write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            self.export_expert()
        self.assertFalse(self.expert_output.exists())

    def test_expert_wrong_task_and_missing_role_rejected_even_if_hashes_match(self):
        self.expert_fixture()
        self.expert_replay["task"] = "whiteboard-wipe-horizontal"
        write_json(self.expert_manifest, self.expert_replay)
        self.expert_audit["manifest"] = digest(self.expert_manifest)
        with self.assertRaisesRegex(ValueError, "replay task"):
            self.export_expert()
        self.expert_replay["task"] = "whiteboard-wipe"
        write_json(self.expert_manifest, self.expert_replay)
        self.expert_audit["manifest"] = digest(self.expert_manifest)
        path = Path(self.expert_audit["membership"]["path"])
        self.expert_membership["role"] = "unknown"
        write_json(path, self.expert_membership)
        self.expert_audit["membership"] = digest(path)
        with self.assertRaisesRegex(ValueError, "expert-reference role"):
            self.export_expert()

    def test_expert_previous_site_mutation_is_detected_by_new_rollback(self):
        self.expert_fixture()
        self.export_expert()
        (self.output / "index.html").write_text("changed previous workbench")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            export.verify_rollback(self.expert_output / "rollback.json")

    def test_expert_copied_video_tampering_prevents_publication(self):
        self.expert_fixture()
        original = export._copy_checked
        def mutate(source, destination, row):
            original(source, destination, row)
            if destination.name == "expert.mp4":
                destination.write_bytes(b"bad copy")
        with patch.object(export, "_copy_checked", side_effect=mutate), self.assertRaisesRegex(ValueError, "Copied expert"):
            self.export_expert()
        self.assertFalse((self.expert_output / "index.html").exists())
        self.assertFalse((self.expert_output / "009_case/report.html").exists())

    def test_expert_refresh_changes_neither_expert_video_nor_previous_site(self):
        self.expert_fixture()
        self.export_expert()
        previous = self.output_snapshot()
        video = self.expert_output / "009_case/media/expert.mp4"
        before = video.read_bytes()
        self.change_web()
        self.assertTrue(export.refresh_web(self.expert_output)["refreshed"])
        self.assertEqual(video.read_bytes(), before)
        self.assertEqual(self.output_snapshot(), previous)
        video.write_bytes(b"changed expert output")
        with self.assertRaisesRegex(ValueError, "Expert video/config"):
            export.refresh_web(self.expert_output)

    def test_expert_cli_is_optional_and_cannot_be_combined_with_refresh(self):
        self.expert_fixture()
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["--source-report", str(self.case / "report.json"), "--output", str(self.expert_output),
                                   "--expert-manifest", str(self.expert_manifest), "--previous-workbench", str(self.output)]), 0)
        self.assertTrue((self.expert_output / "009_case/media/expert.mp4").is_file())
        with self.assertRaises(SystemExit):
            main(["--refresh-web", str(self.output), "--expert-manifest", str(self.expert_manifest)])

    def prepare_diagnostics(self):
        diagnostic_fixture(self.source)
        self.loads["provenance"] = [digest(Path(row["path"])) for row in self.loads["provenance"]]
        write_json(self.case / "tactile_loads.json", self.loads)
        self.write_page()
        bulb = self.root / "explicit_bulb"
        diagnostic_fixture(bulb, "bulb-unscrew")
        return bulb

    def test_diagnostic_copy_and_standalone_keep_previous_site_and_model_payloads(self):
        bulb = self.prepare_diagnostics()
        self.expert_fixture()
        before, previous = self.snapshot(), self.output_snapshot()
        config = export.export_workbench(self.case / "report.html", self.expert_output,
            expert_manifest=self.expert_manifest, previous_workbench=self.output,
            diagnostic_plots=True, diagnostic_example=bulb)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(previous, self.output_snapshot())
        self.assertEqual(config["expert_video"]["clock"], "independent_video_seconds")
        self.assertEqual(config["diagnostic_plots"]["prediction_horizon"], 50)
        self.assertEqual(config["diagnostic_plots"]["checkpoint"], "24999")
        for plot in config["diagnostic_plots"]["plots"]:
            copied = self.expert_output / "009_case" / plot["url"]
            self.assertEqual(copied.read_bytes(), (self.source / copied.name).read_bytes())
            self.assertEqual(digest(copied)["sha256"], plot["sha256"])
        standalone = self.expert_output / "diagnostics/bulb-unscrew_step20000_seed_000.html"
        text = standalone.read_text()
        self.assertIn('id="kai-diagnostic-plots" class="kai-eval wb-root"', text)
        self.assertNotIn("dashboard.js", text)
        _, example = export._embedded(text, "kai-diagnostic-config")
        self.assertEqual(example["workbench_url"], "../009_case/report.html")
        for plot in example["plots"]:
            self.assertEqual(digest(standalone.parent / plot["url"])["sha256"], plot["sha256"])
        new_page = (self.expert_output / "009_case/report.html").read_text()
        for identifier in ("kai-evaluation-data", "kai-tactile-load-data"):
            self.assertEqual(export._embedded(new_page, identifier)[0],
                             export._embedded((self.case / "report.html").read_text(), identifier)[0])
        self.assertTrue(export.verify_rollback(self.expert_output / "rollback.json")["verified"])
        self.assertIn(standalone.name, (self.expert_output / "index.html").read_text())
        self.change_web()
        self.assertTrue(export.refresh_web(self.expert_output)["refreshed"])
        (standalone.parent / example["plots"][0]["url"]).write_bytes(b"changed output")
        with self.assertRaisesRegex(ValueError, "Copied diagnostic PNG SHA256"):
            export.refresh_web(self.expert_output)

    def test_diagnostic_h50_task_and_exact_model_summary_identity_are_required(self):
        trial = self.root / "diagnostic_trial"
        summary = diagnostic_fixture(trial)
        with self.assertRaisesRegex(ValueError, "selected model trial"):
            load_diagnostic_plots(trial, expected_task="whiteboard-wipe", expected_summary_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "task/seed/checkpoint"):
            load_diagnostic_plots(trial, expected_task="bulb-unscrew")
        for value in (30, None):
            summary["prediction_horizon"] = value
            write_json(trial / "summary.json", summary)
            with self.subTest(horizon=value), self.assertRaisesRegex(ValueError, "H50"):
                load_diagnostic_plots(trial, expected_task="whiteboard-wipe")

    def test_diagnostic_metadata_reference_and_artifact_binding_must_agree(self):
        trial = self.root / "diagnostic_trial"
        diagnostic_fixture(trial)
        path = trial / "evaluation_evaluation_plots.json"
        original = json.loads(path.read_text())
        for key in ("reference", "artifacts"):
            changed = deepcopy(original)
            if key == "reference":
                changed[key]["source_hdf5"] = "/different/reference.h5"
            else:
                alternate = trial / "not-bound.png"
                alternate.write_bytes((trial / "evaluation_right_wrist_state.png").read_bytes())
                changed[key]["right_wrist_state"] = str(alternate)
            write_json(path, changed)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "reference|artifact"):
                load_diagnostic_plots(trial, expected_task="whiteboard-wipe")
        write_json(path, original)
        image = trial / "evaluation_right_wrist_state.png"
        image.write_bytes(image.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "PNG"):
            load_diagnostic_plots(trial, expected_task="whiteboard-wipe")

    def test_diagnostic_copy_tampering_prevents_any_entry_publication(self):
        bulb = self.prepare_diagnostics()
        original = export._copy_checked
        def mutate(source, destination, row):
            original(source, destination, row)
            if destination.suffix == ".png":
                destination.write_bytes(b"changed copied plot")
        with patch.object(export, "_copy_checked", side_effect=mutate), self.assertRaisesRegex(ValueError, "Copied diagnostic PNG SHA256"):
            export.export_workbench(self.case / "report.html", self.output, diagnostic_plots=True, diagnostic_example=bulb)
        for relative in ("index.html", "009_case/report.html", "diagnostics/bulb-unscrew_step20000_seed_000.html"):
            self.assertFalse((self.output / relative).exists())


if __name__ == "__main__":
    unittest.main()
