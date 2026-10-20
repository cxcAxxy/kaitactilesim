"""End-to-end suite exports use temporary archives and never mutate sources."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ANALYSIS = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(ANALYSIS))

FIXTURE_SPEC = importlib.util.spec_from_file_location("v1_suite_fixture", ANALYSIS / "tests/test_evaluation_v1_core.py")
FIXTURES = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURES)
def archived_trial(*args, **kwargs):
    """These synthetic V2 fixtures explicitly declare the current H50 cohort."""
    path = FIXTURES.archived_trial(*args, **kwargs)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["prediction_horizon"] = 50
    summary_path.write_text(json.dumps(summary))
    return path

import evaluate_suite as CLI
from evaluation_v1.core import evaluate_trial
from evaluation_v2.evidence import META_NAME, make_metadata, write_evidence
from evaluation_v2.suite import export_suite


def snapshot(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def sidecar(trial, task="whiteboard-wipe"):
    meta = make_metadata(task, clock_id="synthetic-native-clock", source_kind="synthetic_test", nominal_interval_s=.05)
    meta["complete"] = True
    return write_evidence(trial, meta, [
        {"time_s": time, "valid": True, "task_values": {}}
        for time in [0, .05, .1]])


class EvaluationSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.trial = archived_trial(self.root / "seed_000")
        self.output = self.root / "export"

    def test_v1_metrics_outcomes_and_input_files_preserved(self):
        baseline = evaluate_trial(self.trial)
        before = snapshot(self.trial)
        batch = export_suite([self.trial], self.output, include_video=False)
        self.assertEqual(snapshot(self.trial), before)
        self.assertEqual(batch["trial_count"], 1)
        self.assertEqual(batch["by_task"]["whiteboard-wipe"]["outcome_counts"], {"pass": 0, "fail": 1, "unknown": 0})
        generated = json.loads(next(self.output.glob("*/report.json")).read_text())
        for key in ("outcome", "tactile", "safety", "efficiency", "task_specific", "protocol_sha256"):
            self.assertEqual(generated[key], baseline[key], key)
        self.assertEqual(generated["v1_implementation"], baseline["implementation"])
        self.assertIn("task_review", generated)
        self.assertIn("contact_quality", generated)
        self.assertEqual(generated["trial"]["prediction_horizon"], 50)
        self.assertEqual(generated["trial"]["cohort"]["prediction_horizon"], 50)
        self.assertEqual(batch["horizon_selection"]["required_prediction_horizon"], 50)
        for filename in ("index.html", "batch.json", "batch_audit.html", "validation.json", "task_catalog.json",
                         "reference_comparison.json", "integration.html", "component_integration.html"):
            self.assertTrue((self.output / filename).is_file(), filename)

    def test_h30_and_unknown_excluded_before_metrics_not_only_hidden(self):
        legacy = archived_trial(self.root / "old_H30_seed_000")
        unknown = archived_trial(self.root / "named_h50_but_unknown_seed_000")
        for path, horizon in ((legacy, 30), (unknown, None)):
            summary_path = path / "summary.json"
            summary = json.loads(summary_path.read_text())
            if horizon is None:
                del summary["prediction_horizon"]
            else:
                summary["prediction_horizon"] = horizon
            summary.update(execute_steps=50, control_hz=30)
            summary["evaluation"]["success"] = True
            summary_path.write_text(json.dumps(summary))
            # A corrupt old observation must never be loaded for metric computation.
            (path / "review/evaluation_rollout_trace.npz").write_bytes(b"invalid excluded observation")
        before = snapshot(self.root)
        batch = export_suite([self.trial, legacy, unknown], self.output, include_video=False)
        self.assertEqual(batch["trial_count"], 1)
        self.assertEqual(batch["validation"]["trial_count"], 1)
        self.assertEqual(batch["by_task"]["whiteboard-wipe"]["outcome_counts"], {"pass": 0, "fail": 1, "unknown": 0})
        self.assertEqual(batch["horizon_selection"]["input_count"], 3)
        self.assertEqual(len(batch["horizon_selection"]["excluded"]), 2)
        self.assertEqual(len(list(self.output.glob("*/report.json"))), 1)
        self.assertTrue((self.output / "horizon_selection.json").is_file())
        self.assertIn("horizon_selection", (self.output / "index.html").read_text())
        for relative, digest in before.items():
            self.assertEqual(hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest)

    def test_no_matching_h50_does_not_publish_empty_or_misleading_site(self):
        summary_path = self.trial / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["prediction_horizon"] = 30
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "No verified H50"):
            export_suite([self.trial], self.output, include_video=False)
        self.assertFalse(self.output.exists())

    def test_unconfigured_reference_is_explicit_in_batch_and_trial(self):
        batch = export_suite([self.trial], self.output, include_video=False)
        self.assertEqual(batch["reference_comparison"]["status"], "not_configured")
        self.assertEqual(batch["reference_comparison"]["groups"], {})
        generated = json.loads(next(self.output.glob("*/report.json")).read_text())
        self.assertEqual(generated["reference_comparison"]["status"], "not_configured")
        self.assertTrue(generated["reference_comparison"]["reasons"])

    def test_valid_sidecar_is_hashed_without_changing_original_outcome(self):
        sidecar(self.trial)
        before = snapshot(self.trial)
        export_suite([self.trial], self.output, include_video=False)
        self.assertEqual(snapshot(self.trial), before)
        generated = json.loads(next(self.output.glob("*/report.json")).read_text())
        self.assertEqual(generated["evidence_v2"]["source"], "sidecar")
        self.assertEqual(generated["outcome"]["status"], "fail")
        self.assertEqual(generated["task_review"]["review"]["status"], "unknown")
        roles = {row["role"] for row in generated["provenance"]}
        self.assertIn("evaluation_evidence_records", roles)

    def test_partial_or_malformed_evidence_rejects_before_creating_output(self):
        path = self.trial / META_NAME
        path.write_text("{}", encoding="utf-8")
        before = snapshot(self.trial)
        with self.assertRaises(ValueError):
            export_suite([self.trial], self.output, include_video=False)
        self.assertFalse(self.output.exists())
        self.assertEqual(snapshot(self.trial), before)

    def test_foreign_task_evidence_rejected_even_if_recorded_outcome_exists(self):
        sidecar(self.trial, task="install-ram")
        before = snapshot(self.trial)
        with self.assertRaisesRegex(ValueError, "[Tt]ask|任务"):
            export_suite([self.trial], self.output, include_video=False)
        self.assertFalse(self.output.exists())
        self.assertEqual(snapshot(self.trial), before)

    def test_no_overwrite_and_no_export_inside_original_trial(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("belongs to an earlier export")
        before = snapshot(self.output)
        with self.assertRaises(FileExistsError):
            export_suite([self.trial], self.output, include_video=False)
        self.assertEqual(snapshot(self.output), before)
        with self.assertRaisesRegex(ValueError, "outside source"):
            export_suite([self.trial], self.trial / "new-site", include_video=False)
        self.assertFalse((self.trial / "new-site").exists())

    def test_duplicate_sources_rejected_not_counted_as_new_episodes(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            export_suite([self.trial, self.trial], self.output, include_video=False)
        self.assertFalse(self.output.exists())

    def test_reference_report_requires_explicit_manifest(self):
        with self.assertRaisesRegex(ValueError, "manifest"):
            export_suite([self.trial], self.output, include_video=False, reference_reports=[{"trial": {}}])
        self.assertFalse(self.output.exists())

    def test_changed_configuration_provenance_rejected_before_export(self):
        config = self.root / "task-configs.json"
        config.write_text("{}")
        provenance = [{"path": str(config), "sha256": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "changed"):
            export_suite([self.trial], self.output, include_video=False, configuration_provenance=provenance)
        self.assertFalse(self.output.exists())

    def test_failure_unknown_population_survives_export(self):
        unknown = archived_trial(self.root / "seed_unknown")
        path = unknown / "summary.json"
        summary = json.loads(path.read_text())
        summary["evaluation"].pop("success")
        path.write_text(json.dumps(summary))
        batch = export_suite([self.trial, unknown], self.output, include_video=False)
        self.assertEqual(batch["by_task"]["whiteboard-wipe"]["outcome_counts"], {"pass": 0, "fail": 1, "unknown": 1})
        self.assertEqual(len(batch["trials"]), 2)

    def test_cli_exports_summary_and_refuses_overwrite(self):
        arguments = ["--trial-dir", str(self.trial), "--output-dir", str(self.output), "--no-video"]
        before = snapshot(self.trial)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(CLI.main(arguments), 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["trial_count"], 1)
        self.assertEqual(result["reference_status"], "not_configured")
        self.assertEqual(result["prediction_horizon"], 50)
        self.assertEqual(result["excluded_trial_count"], 0)
        self.assertEqual(snapshot(self.trial), before)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            CLI.main(arguments)
        self.assertEqual(error.exception.code, 2)

    def test_cli_bad_configuration_fails_without_partial_export(self):
        config = self.root / "bad.json"
        config.write_text('{"not-complete":')
        args = ["--trial-dir", str(self.trial), "--output-dir", str(self.output), "--task-configs", str(config), "--no-video"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            CLI.main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
