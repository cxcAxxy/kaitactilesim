"""Protocol, episode denominators, immutable output and end-to-end contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from evaluation_v1.core import aggregate_reports, evaluate_trial
from evaluation_v1.protocol import content_hash, make_protocol

SPEC = importlib.util.spec_from_file_location("evaluate_trials", SCRIPTS / "evaluate_trials.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def archived_trial(root: Path, task="whiteboard-wipe", success=False):
    root.mkdir()
    (root / "review").mkdir()
    summary = {
        "task": task, "status": "success" if success else "task_not_completed",
        "sim_seconds": 1.0, "wall_seconds": 4.0, "seed": 0,
        "control_hz": 20, "execute_steps": 5,
        "observation_contract": {"tactile_sent_to_model": False},
        "evaluation": {"success": success, "state": {
            "cleaned_fraction": .5, "released_on_table": False,
            "maximum_lift_m": .2,
        }},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    time = np.arange(21) / 20
    normal = np.ones((21, 3, 2, 2)) * .1
    np.savez(root / "review/evaluation_rollout_trace.npz", time_s=time,
             normal_taxel_force_n=normal,
             fingertip_normal_force_n=normal.sum(axis=(-1, -2)),
             fingertip_link_names=np.array(["left", "center", "right"]))
    return root


def batch_trial(source, status, value, task="task-a", cohort="one"):
    protocol = make_protocol()
    return {"trial": {"trial_id": source, "source": source, "task": task,
                      "cohort": {"task": task, "checkpoint": cohort}},
            "outcome": {"status": status}, "efficiency": {"sim_seconds": value},
            "protocol": protocol, "protocol_sha256": content_hash(protocol),
            "implementation": {"sha256": "same"}, "tactile": {"scopes": {}}}


class EvaluationProtocolTests(unittest.TestCase):
    def test_frozen_protocol_rejects_semantic_and_bad_numeric_overrides(self):
        for override in ({"reference_required": True}, {"typo": 1},
                         {"contact_on_n": float("nan")}, {"contact_off_n": .06},
                         {"analysis_bin_s": 0}, {"change_lag_s": .15},
                         {"contact_on_n": True}, [], "invalid"):
            with self.assertRaises(ValueError):
                make_protocol(override)

    def test_protocol_hash_is_order_independent_and_changes_with_threshold(self):
        default = make_protocol()
        self.assertEqual(content_hash(default), content_hash(dict(reversed(list(default.items())))))
        self.assertNotEqual(content_hash(default), content_hash(make_protocol({"contact_on_n": .08})))

    def test_unknown_and_failure_stay_in_denominator(self):
        result = aggregate_reports([batch_trial("a", "pass", 1),
                                    batch_trial("b", "fail", 3),
                                    batch_trial("c", "unknown", 100)])
        task = result["by_task"]["task-a"]
        self.assertEqual(task["outcome_counts"], {"pass": 1, "fail": 1, "unknown": 1})
        self.assertEqual(task["confirmed_success_fraction"], 1 / 3)
        self.assertEqual(task["possible_success_fraction"], 2 / 3)
        self.assertEqual(task["resolved_success_rate"], .5)
        cohort = next(iter(result["by_cohort"].values()))
        self.assertEqual(cohort["metrics"]["efficiency.sim_seconds"]["median"], 3)

    def test_unknown_only_is_not_zero_resolved_rate(self):
        result = aggregate_reports([batch_trial("a", "unknown", 1)])
        self.assertIsNone(result["by_task"]["task-a"]["resolved_success_rate"])

    def test_cohorts_separate_and_macro_weights_tasks_equally(self):
        reports = [batch_trial("a", "pass", 1), batch_trial("b", "pass", 3, cohort="two"),
                   batch_trial("c", "fail", 1, task="task-b")]
        result = aggregate_reports(reports)
        self.assertEqual(len(result["by_cohort"]), 3)
        self.assertEqual(result["macro_confirmed_success_fraction"], .5)
        self.assertIsNone(result["by_task"]["task-a"]["wilson95_interval"])

    def test_duplicate_or_mixed_protocol_never_inflates_batch(self):
        report = batch_trial("a", "pass", 1)
        with self.assertRaises(ValueError):
            aggregate_reports([report, report])
        other = batch_trial("b", "fail", 1)
        other["protocol_sha256"] = "different"
        with self.assertRaises(ValueError):
            aggregate_reports([report, other])


class EvaluationPipelineTests(unittest.TestCase):
    def test_relative_trial_api_cannot_write_inside_original_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = archived_trial(Path(directory) / "seed_000")
            relative = Path(os.path.relpath(trial, Path.cwd()))
            with self.assertRaisesRegex(ValueError, "outside original trial"):
                CLI.export_reports([relative], trial / "new_report", make_protocol())
            self.assertFalse((trial / "new_report").exists())

    def test_visualization_rejects_new_optional_primary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = archived_trial(Path(directory) / "seed_000")
            report = evaluate_trial(trial)
            # A new control log would switch the task evidence used for plots.
            (trial / "control_steps.jsonl").write_text(json.dumps({
                "sim_time": .1, "task": {"board_normal_force_n": 1.}
            }))
            with self.assertRaisesRegex(ValueError, "evidence selection changed"):
                CLI._visual_package(report, include_video=False)

    def test_elapsed_clock_never_receives_unverified_external_task_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = archived_trial(Path(directory) / "seed_000")
            (trial / "control_steps.jsonl").write_text("\n".join(json.dumps({
                "sim_time": t, "task": {"board_normal_force_n": 1.}
            }) for t in (10., 10.1, 10.2)))
            report = evaluate_trial(trial)
            self.assertEqual(report["tactile"]["scopes"]["board_contact"]["status"], "unavailable")
            self.assertEqual(report["tactile"]["scopes"]["overall"]["status"], "available")

    def test_fractional_task_boundary_does_not_leak_post_phase_force(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = archived_trial(Path(directory) / "seed_000")
            np.savez(trial / "review/evaluation_rollout_trace.npz",
                     simulation_time_s=[0., .1, .2], fingertip_normal_force_n=[[0.], [10.], [20.]])
            (trial / "control_steps.jsonl").write_text("\n".join(json.dumps({
                "sim_time": t, "task": {"board_normal_force_n": value}
            }) for t, value in ((.05, 1.), (.15, 0.), (.25, 0.))))
            report = evaluate_trial(trial)
            scope = report["tactile"]["scopes"]["board_contact"]
            self.assertEqual(scope["status"], "unavailable")
            self.assertIsNone(scope["channels"])
            self.assertIn("between native samples", scope["reason"])

    def test_nested_formal_and_guard_flags_separate_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = []
            for i, mode in enumerate(("enforce", "observe")):
                trial = archived_trial(Path(directory) / f"seed_{i:03d}")
                summary_path = trial / "summary.json"
                summary = json.loads(summary_path.read_text())
                summary["evaluation"].update(guard_mode=mode, formal_metrics_valid=i == 0, diagnostic_only=i == 1)
                summary_path.write_text(json.dumps(summary))
                reports.append(evaluate_trial(trial))
            self.assertEqual([r["trial"]["cohort"]["guard_mode"] for r in reports], ["enforce", "observe"])
            self.assertEqual([r["trial"]["cohort"]["formal_metrics_valid"] for r in reports], [True, False])
            self.assertEqual(len(aggregate_reports(reports)["by_cohort"]), 2)

    def test_offline_report_does_not_modify_input_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = archived_trial(root / "seed_000")
            files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in trial.rglob("*") if path.is_file()}
            output = root / "output"
            batch = CLI.export_reports([trial], output, make_protocol())
            self.assertEqual(batch["by_task"]["whiteboard-wipe"]["outcome_counts"]["fail"], 1)
            self.assertTrue((output / "index.html").exists())
            report = json.loads(next(output.glob("*/report.json")).read_text())
            self.assertEqual(report["outcome"]["status"], "fail")
            self.assertEqual(len(report["tactile"]["scopes"]["overall"]["channels"]), 3)
            self.assertTrue(report["provenance"])
            for filename, before in files.items():
                self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), before)
            with self.assertRaises(FileExistsError):
                CLI.export_reports([trial], output, make_protocol())
            with self.assertRaises(ValueError):
                CLI.export_reports([trial], trial / "new_report", make_protocol())

    def test_outcome_survives_absent_tactile_and_unknown_tasks_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({
                "task": "custom-task", "evaluation": {"success": True}, "sim_seconds": 1,
            }))
            report = evaluate_trial(root)
            self.assertEqual(report["outcome"]["status"], "pass")
            self.assertEqual(report["tactile"]["status"], "missing")

    def test_duplicate_paths_and_batch_without_trials_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = archived_trial(root / "seed_000")
            with self.assertRaises(ValueError):
                CLI.discover_trials([trial, trial / "."], [])
            self.assertEqual(CLI.discover_trials([], [root]), [trial.resolve()])
            with self.assertRaises(ValueError):
                CLI.discover_trials([], [trial])


if __name__ == "__main__":
    unittest.main()
