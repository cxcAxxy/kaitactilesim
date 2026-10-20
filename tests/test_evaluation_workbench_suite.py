"""Offline suite integration tests; no simulator, GPU, or archived input writes."""
from copy import deepcopy
from concurrent.futures import Future
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.adapters import TrialData
from evaluation_v1.protocol import content_hash
from evaluation_workbench import suite


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class DiscoveryTests(unittest.TestCase):
    def test_canonical_aliases_count_once_and_h30_running_incomplete_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "whiteboard-wipe" / "pi05" / "checkpoint"
            for index in range(5):
                write_json(group / f"seed_{index:03d}" / "summary.json", {
                    "task": "whiteboard-wipe", "status": "running" if index == 2 else "finished",
                    "video": {"completed": index != 3},
                })
            alias = root / "whiteboard-wipe" / "alias" / "checkpoint" / "seed_000"
            alias.parent.mkdir(parents=True)
            alias.symlink_to(group / "seed_000", target_is_directory=True)
            hidden = root / "_archive" / "pi05" / "checkpoint" / "seed_000"
            write_json(hidden / "summary.json", {"task": "whiteboard-wipe", "video": {"completed": True}})

            def horizon(path):
                return {"status": "unverified" if path.name == "seed_004" else "verified",
                        "prediction_horizon": 30 if path.name == "seed_001" else 50}

            with patch.object(suite, "inspect_prediction_horizon", side_effect=horizon):
                trials, selection = suite.discover_trials(root)
            self.assertEqual(trials, [group / "seed_000"])
            self.assertEqual(selection["included_count"], 1)
            self.assertEqual(len(selection["aliases"]), 1)
            self.assertEqual(len(selection["excluded"]), 4)
            reasons = [row["reason"] for row in selection["excluded"]]
            self.assertEqual(reasons.count("prediction_horizon_not_verified_H50"), 2)
            self.assertIn("in_progress", reasons)
            self.assertIn("recording_not_complete", reasons)

    def test_mode_flags_are_strict_and_never_inferred_from_outcome(self):
        for outcome in ("pass", "fail", "unknown"):
            for formal, diagnostic, expected in ((True, False, "formal"), (False, True, "diagnostic"),
                                                (True, True, "unverified"), (False, False, "unverified"),
                                                (None, None, "unverified"), (1, 0, "unverified"),
                                                ("true", "false", "unverified")):
                with self.subTest(outcome=outcome, formal=formal, diagnostic=diagnostic):
                    self.assertEqual(suite._mode({"status": outcome, "formal_metrics_valid": formal,
                                                  "diagnostic_only": diagnostic}), expected)


class AssemblyTests(unittest.TestCase):
    def test_augmentation_updates_hash_without_filling_native_missing_values(self):
        rows = [{"sim_time": 10, "task": {"insertion_depth_m": 0}},
                {"sim_time": 10.1, "task": {"insertion_depth_m": None}, "gap_after": True},
                {"sim_time": 10.2, "task": {"insertion_depth_m": .006}}]
        frames = [{"simulation_time_s": 10, "task_metrics": {"orientation_error_rad": None}},
                  {"simulation_time_s": 10.2, "task_metrics": {"orientation_error_rad": 0}}]
        trial = TrialData(root=Path("/synthetic/trial"), task="install-ram",
                          summary={"task": "install-ram", "evaluation": {"success": False}},
                          control_rows=rows, task_tactile_clock_aligned=True)
        visual = {"time_origin_s": 10, "clock_aligned": True, "task_series": [],
                  "tactile_series": {"time_s": [0, .1], "normal_force_n": [[0], [None]],
                                     "gap_after": [True, False]}}
        original_hash, tactile_before, rows_before = content_hash(visual), deepcopy(visual["tactile_series"]), deepcopy(rows)
        case = {"trial": trial, "frame_rows": frames, "report": {"trial": {"task": trial.task},
                "outcome": {"status": "fail"}, "protocol": {"max_gap_factor": 1.5},
                "visualization": {"sha256": original_hash}}, "visualization": visual,
                "tactile_loads": {"normal_total_n": [0, None]}, "config": {}}
        with patch("evaluation_workbench.data.build_case", return_value=case):
            assembled = suite._assemble_trial((trial.root, {}))
        by_key = {item["key"]: item for item in assembled["visualization"]["task_series"]}
        self.assertEqual(by_key["insertion_depth_m"]["values"], [0, None, .006])
        self.assertEqual(by_key["insertion_depth_m"]["gap_after"], [False, True, False])
        self.assertEqual(by_key["orientation_error_rad"]["values"], [None, 0])
        self.assertEqual(len(by_key["orientation_error_rad"]["time_s"]), 2)
        self.assertEqual(assembled["report"]["visualization"]["sha256"], content_hash(visual))
        self.assertNotEqual(assembled["report"]["visualization"]["sha256"], original_hash)
        self.assertEqual(visual["tactile_series"], tactile_before)
        self.assertEqual(rows, rows_before)
        self.assertEqual(assembled["tactile_loads"]["normal_total_n"], [0, None])
        self.assertEqual(assembled["report"]["outcome"]["status"], "fail")
        self.assertEqual(assembled["config"]["review_status"], "not_verified")
        self.assertTrue(assembled["config"]["generic"])
        self.assertNotIn("trial", assembled)
        self.assertNotIn("frame_rows", assembled)

    def test_worker_failure_cancels_pending_and_exits_executor(self):
        class Future:
            def __init__(self, index):
                self.index, self.cancelled = index, False
            def result(self):
                if self.index == 0:
                    raise RuntimeError("synthetic worker failure")
                return {"index": self.index}
            def cancel(self):
                self.cancelled = True

        class Executor:
            def __init__(self):
                self.futures, self.exited = [], False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.exited = True
            def submit(self, *args):
                future = Future(len(self.futures)); self.futures.append(future)
                return future

        executor = Executor()
        with patch.object(suite, "ProcessPoolExecutor", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                list(suite._prepared_cases([Path("a"), Path("b"), Path("c")], {}, 2))
        self.assertTrue(executor.exited)
        self.assertEqual(len(executor.futures), 2)
        self.assertTrue(executor.futures[1].cancelled)

    def test_export_consumer_error_also_cancels_pending_workers(self):
        futures = [Future() for _ in range(3)]
        futures[0].set_result({"index": 0})
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.submit.side_effect = futures
        with patch.object(suite, "ProcessPoolExecutor", return_value=executor):
            with self.assertRaisesRegex(ValueError, "synthetic render failure"):
                for _, case in enumerate(suite._prepared_cases([Path("a"), Path("b"), Path("c")], {}, 2)):
                    raise ValueError("synthetic render failure")
        executor.__exit__.assert_called_once()
        self.assertTrue(futures[1].cancelled())
        self.assertTrue(futures[2].cancelled())


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous = self.root / "previous"
        self.previous.mkdir()
        (self.previous / "index.html").write_text("preserved previous workbench", encoding="utf-8")
        self.protected = self.root / "original.txt"
        self.protected.write_text("read-only original", encoding="utf-8")
        write_json(self.previous / "rollback.json", {"schema_version": "evaluation-experiment-rollback-v1",
                   "protected_files": [suite._digest(self.protected)], "protected_roots": []})
        self.output = self.root / "new-suite"
        self.paths, self.cases = [], []
        for index, (outcome, formal, diagnostic) in enumerate((("pass", True, False), ("fail", False, True),
                                                               ("unknown", None, None))):
            path = self.root / "source" / f"seed_{index:03d}"
            summary = {"task": "pick-place", "seed": index, "status": "completed",
                       "video": {"completed": True}, "model_family": "fixture", "checkpoint_path": "/step50"}
            write_json(path / "summary.json", summary)
            report = {"trial": {"task": "pick-place", "seed": index, "trial_id": f"case-{index}",
                                "cohort": {"model_family": "fixture"}},
                      "outcome": {"status": outcome, "formal_metrics_valid": formal, "diagnostic_only": diagnostic},
                      "provenance": [suite._digest(path / "summary.json")]}
            self.paths.append(path)
            self.cases.append({"report": report, "summary": summary, "visualization": {"task_series": [], "media": {}},
                               "tactile_loads": {"normal_status": "missing", "tangent_status": "missing",
                                                 "normal_total_n": [], "tangent_total_n": []},
                               "config": {"task_presentation": {"title": "抓取放置"}}, "copies": []})

    def export(self, *, cases=None, workers=1):
        with ExitStack() as stack:
            stack.enter_context(patch("evaluation_workbench.experts.prepare_expert_bindings", return_value={}))
            stack.enter_context(patch("evaluation_workbench.experts.expert_for_case", return_value=None))
            stack.enter_context(patch("evaluation_workbench.experts.model_views_for_case", return_value={"views": [], "provenance": []}))
            stack.enter_context(patch.object(suite, "inspect_prediction_horizon", return_value={"status": "verified", "prediction_horizon": 50}))
            stack.enter_context(patch.object(suite, "_prepared_cases", side_effect=lambda *_: cases if cases is not None else iter(deepcopy(self.cases))))
            stack.enter_context(patch.object(suite, "_page", return_value="<!doctype html><title>fixture</title>"))
            stack.enter_context(patch.object(suite, "render_report", return_value="<!doctype html><title>audit fixture</title>"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return suite.export_suite(self.paths, self.output, previous_workbench=self.previous,
                                      task_demo_root=self.root, workers=workers)

    def test_all_outcomes_and_modes_retained_and_previous_site_unchanged(self):
        before = {path: path.read_bytes() for path in self.previous.rglob("*") if path.is_file()}
        batch = self.export()
        self.assertEqual(batch["trial_count"], 3)
        self.assertEqual(batch["mode_counts"], {"formal": 1, "diagnostic": 1, "unverified": 1})
        self.assertEqual([row["outcome"] for row in batch["trials"]], ["pass", "fail", "unknown"])
        self.assertEqual(batch["coverage"]["normal"], 0)
        for row in batch["trials"]:
            directory = self.output / Path(row["report_path"]).parent
            loads = json.loads((directory / "tactile_loads.json").read_text())
            self.assertEqual(loads["normal_total_n"], [])
            self.assertEqual(row["tactile"], "missing")
            self.assertEqual(row["task_curve_count"], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(self.protected.read_text(), "read-only original")
        self.assertTrue(suite.verify_suite(self.output)["verified"])

    def test_source_changed_after_early_case_rejected_before_publishing(self):
        def changing_cases():
            yield deepcopy(self.cases[0])
            (self.paths[0] / "summary.json").write_text('{"changed":true}', encoding="utf-8")
            yield deepcopy(self.cases[1])
            yield deepcopy(self.cases[2])
        with self.assertRaisesRegex(ValueError, "SHA256|changed"):
            self.export(cases=changing_cases())
        self.assertFalse((self.output / "index.html").exists())
        self.assertFalse((self.output / "verification.json").exists())

    def test_invalid_worker_count_does_not_create_output(self):
        for workers in (0, 9, True):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers"):
                    self.export(workers=workers)
                self.assertFalse(self.output.exists())

    def test_explicit_trials_cannot_bypass_running_or_horizon_filter(self):
        for status, completed, horizon in (("running", True, 50), ("completed", False, 50),
                                            ("completed", True, 30)):
            with self.subTest(status=status, completed=completed, horizon=horizon):
                summary = deepcopy(self.cases[0]["summary"])
                summary["status"] = status
                summary["video"]["completed"] = completed
                write_json(self.paths[0] / "summary.json", summary)
                with patch.object(suite, "inspect_prediction_horizon", return_value={"status": "verified", "prediction_horizon": horizon}):
                    with self.assertRaises(ValueError):
                        suite.export_suite(self.paths, self.output, previous_workbench=self.previous,
                                           task_demo_root=self.root)
                self.assertFalse(self.output.exists())

    def test_existing_previous_site_and_source_nested_target_are_refused(self):
        with self.assertRaisesRegex(ValueError, "new non-symlink"):
            suite.export_suite(self.paths, self.previous, previous_workbench=self.previous, task_demo_root=self.root)
        nested_previous = self.paths[0] / "previous"
        nested_previous.mkdir()
        with self.assertRaisesRegex(ValueError, "inside original trials"):
            suite.export_suite(self.paths, self.paths[0] / "new-suite", previous_workbench=nested_previous,
                               task_demo_root=self.root)
        self.assertFalse((self.paths[0] / "new-suite").exists())


if __name__ == "__main__":
    unittest.main()
