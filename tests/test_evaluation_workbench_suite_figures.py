"""Small transaction tests for model-only completion; no renderer or simulator."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.protocol import content_hash
from evaluation_workbench import suite, suite_figures


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def fixture(root):
    previous, output = root / "previous", root / "current"
    previous.mkdir(parents=True)
    (previous / "index.html").write_text("older site untouched", encoding="utf-8")
    output.mkdir()
    sources, entries = [], []
    for index, task in enumerate(("usb-insert", "whiteboard-wipe")):
        source = root / "sources" / task / "seed_000"
        write_json(source / "summary.json", {"task": task, "seed": 0, "H": 50})
        provenance = [suite._digest(source / "summary.json")]
        trial_id, slug = task + ":fixture", f"{index:03d}_{task}_seed_000"
        case = output / slug
        visual = {"time_s": [0, 1], "values": [0, None]}
        report = {"schema_version": "task-tactile-evaluation-v1", "trial": {"trial_id": trial_id,
                  "task": task, "source": str(source), "prediction_horizon": 50, "seed": 0},
                  "provenance": provenance, "outcome": {"status": "fail"},
                  "visualization": {"sha256": content_hash(visual)}}
        loads = {"trial_id": trial_id, "normal_total_n": [0, None]}
        config = {"trial_id": trial_id, "task": task, "prediction_horizon": 50,
                  "diagnostic_plots": {"status": "missing", "plots": []}}
        for name, value in (("report.json", report), ("visualization.json", visual),
                            ("tactile_loads.json", loads), ("workbench.json", config)):
            write_json(case / name, value)
        (case / "report.html").write_text("original report page", encoding="utf-8")
        (case / "media").mkdir()
        (case / "media" / "video.mp4").write_bytes(b"untouched video")
        sources.append({"trial_id": trial_id, "source": str(source), "provenance": provenance})
        entries.append({"trial_id": trial_id, "task": task, "source": str(source), "prediction_horizon": 50,
                        "seed": 0, "report_path": slug + "/report.html", "plots": "missing", "outcome": "fail"})
    batch = {"schema_version": suite.SCHEMA, "prediction_horizon": 50, "trial_count": 2, "task_count": 2,
             "trials": entries, "coverage": {"normal": 0, "tangent": 0, "plots": 0, "expert": 0},
             "previous_url": "../previous/index.html"}
    write_json(output / "batch.json", batch)
    write_json(output / "source_manifest.json", {"trials": sources})
    write_json(output / "verification.json", {"status": "passed"})
    write_json(output / "rollback.json", {"schema_version": "evaluation-experiment-rollback-v1",
               "experiment": suite.SCHEMA, "previous_workbench": str(previous),
               "protected_files": [suite._digest(previous / "index.html")],
               "protected_roots": suite._inventory([previous])})
    (output / "index.html").write_text("original portal", encoding="utf-8")
    (output / "assets").mkdir()
    for name in suite_figures._ASSETS:
        (output / "assets" / name).write_text("previous " + name, encoding="utf-8")
    write_json(output / "artifact_manifest.json", {"files": [suite._digest(path) for path in sorted(output.rglob("*")) if path.is_file()]})
    return output, previous, entries


def fake_generate(jobs, workers):
    descriptors = []
    for source, directory, report in jobs:
        plots = []
        for identifier, name, _, _ in suite_figures._PLOTS:
            path = directory / "plots" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic model-only PNG " + identifier.encode())
            plots.append({"id": identifier, "url": "plots/" + name, "sha256": suite._digest(path)["sha256"]})
        descriptors.append({"status": "available", "comparison_kind": "model_only", "task": "usb-insert",
                            "seed": 0, "prediction_horizon": 50, "plots": plots, "provenance": report["provenance"]})
    return descriptors


def model_view_fixture(root):
    output, previous, entries = fixture(root)
    source_manifest = json.loads((output / "source_manifest.json").read_text())
    for entry, source_entry in zip(entries, source_manifest["trials"]):
        directory = output / Path(entry["report_path"]).parent
        source = Path(entry["source"])
        raw = b"synthetic recorded video " + entry["task"].encode()
        (source / "review.mp4").write_bytes(raw)
        (directory / "media" / "rollout.mp4").write_bytes(raw)
        video_row = {**suite._digest(source / "review.mp4"), "role": "copied_review_video"}
        report = json.loads((directory / "report.json").read_text())
        visual = json.loads((directory / "visualization.json").read_text())
        visual["media"] = {"status": "available", "url": "media/rollout.mp4", "sha256": video_row["sha256"],
                           "sync_status": "unverified"}
        report["provenance"].append(video_row)
        source_entry["provenance"].append(video_row)
        report["visualization"]["sha256"] = content_hash(visual)
        write_json(directory / "report.json", report)
        write_json(directory / "visualization.json", visual)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "artifact_manifest.json", {"files": [suite._digest(path) for path in sorted(output.rglob("*"))
               if path.is_file() and path.name != "artifact_manifest.json"]})
    witness = root / "audited_renderer.txt"
    witness.write_bytes(b"explicit model crop renderer witness")
    witness_row = {**suite._digest(witness), "role": "model_crop_audited_renderer_bytes"}
    def provider(summary, sha):
        if summary["task"] == "usb-insert":
            return {"status": "verified", "sha256": sha, "source_width": 1920, "source_height": 1080,
                    "views": [{"id": "wrist", "label": "右腕视角", "x": 660, "y": 78, "width": 434, "height": 324}],
                    "provenance": [witness_row], "reason": "explicitly audited crop"}
        return {"status": "full_frame_only", "sha256": sha, "views": [], "reason": "本录制没有可核验的腕部布局"}
    return output, previous, entries, provider, witness_row


class SuiteFigureTests(unittest.TestCase):
    def complete(self, output, generator=fake_generate, replace=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(suite_figures, "_generate_all", side_effect=generator))
            stack.enter_context(patch.object(suite_figures, "_page", side_effect=lambda report, visual, loads, config: json.dumps(config)))
            if replace:
                stack.enter_context(patch.object(suite_figures.os, "replace", side_effect=replace))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return suite_figures.complete_missing_model_plots(output, workers=2)

    def test_success_only_usb_gallery_changes_and_complete_suite_still_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, previous, entries = fixture(root)
            before, old, source = snapshot(output), snapshot(previous), snapshot(root / "sources")
            result = self.complete(output)
            self.assertEqual(result["generated_count"], 1)
            self.assertEqual(result["png_count"], 3)
            self.assertFalse(result["metrics_recomputed"])
            self.assertTrue(result["payloads_unchanged"])
            changed = set(result["updated"])
            after = snapshot(output)
            for path, raw in before.items():
                if path not in changed:
                    self.assertEqual(after[path], raw, path)
            self.assertEqual(snapshot(previous), old)
            self.assertEqual(snapshot(root / "sources"), source)
            batch = json.loads((output / "batch.json").read_text())
            self.assertEqual(batch["trials"][0]["plot_kind"], "model_only")
            self.assertEqual(batch["trials"][0]["plots"], "available")
            self.assertEqual(batch["trials"][1]["plots"], "missing")
            self.assertEqual(batch["coverage"]["plots"], 1)
            self.assertEqual([row["outcome"] for row in batch["trials"]], ["fail", "fail"])
            self.assertTrue(suite.verify_suite(output)["suite_verified"])
            self.assertEqual(self.complete(output)["generated_count"], 0)
            self.assertEqual(after, snapshot(output))

    def test_worker_or_mid_commit_failure_restores_exact_before_bytes(self):
        original_replace = suite_figures.os.replace
        for failure in ("worker", "commit"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output, previous, entries = fixture(Path(temporary))
                before, old = snapshot(output), snapshot(previous)
                def partial_worker(jobs, workers):
                    fake_generate(jobs, workers)
                    raise RuntimeError("synthetic worker failure")
                calls = []
                def interrupted_replace(source, target):
                    calls.append(str(target))
                    if len(calls) == 4:
                        raise RuntimeError("synthetic commit failure")
                    return original_replace(source, target)
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    self.complete(output, generator=partial_worker if failure == "worker" else fake_generate,
                                  replace=interrupted_replace if failure == "commit" else None)
                self.assertEqual(snapshot(output), before)
                self.assertEqual(snapshot(previous), old)
                self.assertFalse((output / ".complete-model-plots.lock").exists())
                self.assertFalse(list(output.rglob("model_right_*.png")))
                self.assertTrue(suite.verify_suite(output)["suite_verified"])

    def test_tampered_case_old_site_and_outside_case_target_are_rejected(self):
        for failure in ("tampered", "old_site", "outside"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output, previous, entries = fixture(Path(temporary))
                if failure == "tampered":
                    case = output / Path(entries[0]["report_path"]).parent
                    (case / "tactile_loads.json").write_text('{"modified":true}', encoding="utf-8")
                elif failure == "old_site":
                    rollback = json.loads((output / "rollback.json").read_text())
                    rollback["experiment"] = "evaluation-workbench-v1"
                    write_json(output / "rollback.json", rollback)
                else:
                    batch = json.loads((output / "batch.json").read_text())
                    batch["trials"][0]["report_path"] = "../previous/report.html"
                    write_json(output / "batch.json", batch)
                    manifest = json.loads((output / "artifact_manifest.json").read_text())
                    manifest["files"] = [suite._digest(output / "batch.json") if row["path"] == str(output / "batch.json") else row for row in manifest["files"]]
                    write_json(output / "artifact_manifest.json", manifest)
                before, old = snapshot(output), snapshot(previous)
                with self.assertRaises(ValueError):
                    self.complete(output)
                self.assertEqual(snapshot(output), before)
                self.assertEqual(snapshot(previous), old)

    def test_asset_refresh_changes_only_six_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, previous, entries = fixture(root)
            before, old, source = snapshot(output), snapshot(previous), snapshot(root / "sources")
            result = suite_figures.refresh_suite_assets(output)
            allowlist = {"assets/" + name for name in suite_figures._ASSETS} | {"verification.json", "artifact_manifest.json"}
            self.assertEqual(set(result["updated"]), allowlist)
            self.assertTrue(result["html_unchanged"])
            self.assertTrue(result["payloads_unchanged"])
            self.assertFalse(result["metrics_recomputed"])
            after = snapshot(output)
            self.assertEqual(set(before), set(after))
            for path, raw in before.items():
                if path not in allowlist:
                    self.assertEqual(after[path], raw, path)
            for name in suite_figures._ASSETS:
                self.assertEqual(after["assets/" + name], (suite_figures.WEB / name).read_bytes())
            verification = json.loads(after["verification.json"])
            self.assertTrue(verification.pop("implementation"))
            self.assertEqual(verification, json.loads(before["verification.json"]))
            self.assertEqual(snapshot(previous), old)
            self.assertEqual(snapshot(root / "sources"), source)
            self.assertTrue(suite.verify_suite(output)["suite_verified"])

    def test_asset_refresh_rejects_bad_bindings_and_rolls_back_commit_failure(self):
        original_replace = suite_figures.os.replace
        for failure in ("commit", "asset_sha", "source_manifest_sha", "old_site"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output, previous, entries = fixture(Path(temporary))
                if failure == "asset_sha":
                    (output / "assets" / "workbench.js").write_bytes(b"unexpected modification")
                elif failure == "source_manifest_sha":
                    write_json(output / "source_manifest.json", {"trials": []})
                elif failure == "old_site":
                    rollback = json.loads((output / "rollback.json").read_text())
                    rollback["experiment"] = "evaluation-workbench-v1"
                    write_json(output / "rollback.json", rollback)
                before, old = snapshot(output), snapshot(previous)
                calls = []
                def interrupted_replace(source, target):
                    calls.append(str(target))
                    if len(calls) == 5:
                        raise RuntimeError("synthetic refresh commit failure")
                    return original_replace(source, target)
                if failure == "commit":
                    with patch.object(suite_figures.os, "replace", side_effect=interrupted_replace), self.assertRaisesRegex(RuntimeError, "synthetic refresh"):
                        suite_figures.refresh_suite_assets(output)
                    self.assertTrue(suite.verify_suite(output)["suite_verified"])
                else:
                    with self.assertRaises(ValueError):
                        suite_figures.refresh_suite_assets(output)
                self.assertEqual(snapshot(output), before)
                self.assertEqual(snapshot(previous), old)
                self.assertFalse((output / ".complete-model-plots.lock").exists())

    def test_model_view_refresh_keeps_payloads_and_binds_additional_crop_witnesses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, previous, entries, provider, witness = model_view_fixture(root)
            before, old, source = snapshot(output), snapshot(previous), snapshot(root / "sources")
            with patch("evaluation_workbench.experts.model_views_for_case", side_effect=provider), \
                    patch.object(suite_figures, "_page", side_effect=lambda r, v, l, c: json.dumps(c)):
                result = suite_figures.refresh_suite_model_views(output)
            self.assertEqual(result["verified_wrist_count"], 1)
            self.assertEqual(result["full_frame_only_count"], 1)
            self.assertEqual(result["updated_case_count"], 2)
            self.assertFalse(result["metrics_recomputed"])
            after = snapshot(output)
            for path, raw in before.items():
                if path not in result["updated"]:
                    self.assertEqual(after[path], raw, path)
            self.assertEqual(set(after), set(before))
            self.assertEqual(snapshot(previous), old)
            self.assertEqual(snapshot(root / "sources"), source)
            first = output / Path(entries[0]["report_path"]).parent
            second = output / Path(entries[1]["report_path"]).parent
            config = json.loads((first / "workbench.json").read_text())
            fallback = json.loads((second / "workbench.json").read_text())["media_views"]
            self.assertEqual(config["media_views"]["views"][0]["id"], "wrist")
            self.assertEqual(fallback["status"], "full_frame_only")
            self.assertTrue(fallback["reason"])
            self.assertEqual(fallback["views"], [])
            report = json.loads((first / "report.json").read_text())
            sources = json.loads((output / "source_manifest.json").read_text())["trials"]
            self.assertIn(witness, sources[0]["provenance"])
            self.assertNotIn(witness, report["provenance"])
            self.assertTrue(suite_figures._report_provenance_is_preserved(report["provenance"], sources[0]["provenance"]))
            self.assertTrue(suite.verify_suite(output)["suite_verified"])
            # Adding crop witnesses must not prevent the independent model-only
            # plot completion from accepting the same original report evidence.
            self.assertEqual(self.complete(output)["generated_count"], 1)
            self.assertTrue(suite.verify_suite(output)["suite_verified"])

    def test_model_view_refresh_rejects_video_and_summary_or_extra_evidence_tampering(self):
        for failure in ("video", "summary", "extra_role", "conflicting_sha"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                output, previous, entries, provider, witness = model_view_fixture(Path(temporary))
                first = output / Path(entries[0]["report_path"]).parent
                if failure == "video":
                    (first / "media" / "rollout.mp4").write_bytes(b"substituted video")
                elif failure == "summary":
                    (Path(entries[0]["source"]) / "summary.json").write_bytes(b'{"task":"other"}')
                else:
                    sources = json.loads((output / "source_manifest.json").read_text())
                    if failure == "extra_role":
                        sources["trials"][0]["provenance"].append({**witness, "role": "not_a_crop_witness"})
                    else:
                        sources["trials"][0]["provenance"][0]["sha256"] = "0" * 64
                    write_json(output / "source_manifest.json", sources)
                    manifest = json.loads((output / "artifact_manifest.json").read_text())
                    manifest["files"] = [suite._digest(output / "source_manifest.json") if row["path"] == str(output / "source_manifest.json") else row for row in manifest["files"]]
                    write_json(output / "artifact_manifest.json", manifest)
                before, old = snapshot(output), snapshot(previous)
                with patch("evaluation_workbench.experts.model_views_for_case", side_effect=provider), self.assertRaises(ValueError):
                    suite_figures.refresh_suite_model_views(output)
                self.assertEqual(snapshot(output), before)
                self.assertEqual(snapshot(previous), old)

    def test_model_view_refresh_mid_commit_error_restores_all_bytes(self):
        original_replace = suite_figures.os.replace
        with tempfile.TemporaryDirectory() as temporary:
            output, previous, entries, provider, witness = model_view_fixture(Path(temporary))
            before, old = snapshot(output), snapshot(previous)
            calls = []
            def interrupted_replace(source, target):
                calls.append(str(target))
                if len(calls) == 4:
                    raise RuntimeError("synthetic crop commit failure")
                return original_replace(source, target)
            with patch("evaluation_workbench.experts.model_views_for_case", side_effect=provider), \
                    patch.object(suite_figures, "_page", side_effect=lambda r, v, l, c: json.dumps(c)), \
                    patch.object(suite_figures.os, "replace", side_effect=interrupted_replace), \
                    self.assertRaisesRegex(RuntimeError, "synthetic crop commit"):
                suite_figures.refresh_suite_model_views(output)
            self.assertEqual(snapshot(output), before)
            self.assertEqual(snapshot(previous), old)
            self.assertFalse((output / ".complete-model-plots.lock").exists())
            self.assertTrue(suite.verify_suite(output)["suite_verified"])


if __name__ == "__main__":
    unittest.main()
