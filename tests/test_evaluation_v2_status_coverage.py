"""Independent read-only selection and coverage regressions for the V2 suite."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v2.status_input import completed_status_trials
from evaluation_v2.validation import coverage_report


def report(task="whiteboard-wipe", seed=0, outcome="fail", trial_source=None, **evidence):
    return {"trial": {"task": task, "seed": seed, "source": trial_source or f"/fixtures/{task}/seed_{seed}",
                       "trial_id": f"{task}:{seed}:{trial_source or ''}"},
            "outcome": {"status": outcome}, "task_review": {"review": {"status": "unknown"}},
            "evidence_v2": {"source": "sidecar", "status": "available", "recorded_complete": True,
                            "sampled_complete": True, **evidence}}


class StatusSelectionTests(unittest.TestCase):
    def test_managed_run_layout_and_ambiguous_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "group/run/evaluation/seed_000"
            nested.mkdir(parents=True)
            (nested / "summary.json").write_text("{}")
            path = root / "status.json"
            path.write_text(json.dumps({"groups": [{"status": "complete", "evaluation_returncode": 0,
                                                   "output_dir": "group"}]}))
            trials, selection = completed_status_trials(path)
            self.assertEqual(trials, [nested])
            self.assertEqual(selection["included_groups"][0]["evaluation_dir"], str(nested.parent))
            alternative = root / "group/seed_001"
            alternative.mkdir()
            (alternative / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                completed_status_trials(path)

    def make_group(self, root, name, *, status="complete", returncode=0, count=2):
        output = root / name
        output.mkdir()
        for index in range(count):
            seed = output / f"seed_{index:03d}"
            seed.mkdir()
            (seed / "summary.json").write_text(json.dumps({"task": name, "evaluation": {"success": False}}))
        return {"task": name, "step": 20000, "status": status,
                "evaluation_returncode": returncode, "output_dir": name}

    def test_selects_only_successfully_completed_groups_and_retains_selection_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = self.make_group(root, "good")
            bad = self.make_group(root, "failed", returncode=1)
            # A running group points outside and must not even be inspected.
            running = {"task": "live", "status": "running", "evaluation_returncode": None,
                       "output_dir": "/a/nonexistent/live/simulation"}
            path = root / "status.json"
            raw = json.dumps({"groups": [good, bad, running]}).encode()
            path.write_bytes(raw)
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            trials, selection = completed_status_trials(path)
            self.assertEqual(trials, [root / "good/seed_000", root / "good/seed_001"])
            self.assertEqual(selection["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(selection["included_groups"]), 1)
            self.assertEqual(len(selection["skipped_groups"]), 2)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*.json")})

    def test_boolean_and_float_return_codes_are_not_successful_process_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = [self.make_group(root, "bool", returncode=False),
                      self.make_group(root, "float", returncode=0.0),
                      self.make_group(root, "true", returncode=True)]
            path = root / "status.json"; path.write_text(json.dumps({"groups": groups}))
            trials, selection = completed_status_trials(path)
            self.assertEqual(trials, [])
            self.assertEqual(len(selection["skipped_groups"]), 3)

    def test_group_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, elsewhere = Path(directory), Path(outside)
            group = self.make_group(elsewhere, "foreign")
            group["output_dir"] = str(elsewhere / "foreign")
            path = root / "status.json"; path.write_text(json.dumps({"groups": [group]}))
            with self.assertRaisesRegex(ValueError, "inside"): completed_status_trials(path)

    def test_seed_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, elsewhere = Path(directory), Path(outside)
            group = self.make_group(root, "group", count=0)
            (elsewhere / "summary.json").write_text("{}")
            (root / "group/seed_000").symlink_to(elsewhere, target_is_directory=True)
            path = root / "status.json"; path.write_text(json.dumps({"groups": [group]}))
            with self.assertRaisesRegex(ValueError, "escapes"): completed_status_trials(path)

    def test_empty_completed_group_duplicates_and_malformed_status_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "status.json"
            empty = self.make_group(root, "empty", count=0)
            full = self.make_group(root, "full", count=1)
            for value in ({"groups": [empty]}, {"groups": [full, full]}, {"groups": [42]}, [], {"groups": {}}):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError): completed_status_trials(path)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{"task": "whiteboard-wipe", "contract_id": "whiteboard:v2"},
                        {"task": "sponge-grasp", "contract_id": "sponge:v2"},
                        {"task": "install-ram", "contract_id": "ram:v2"}]

    def test_failures_unknowns_unseen_tasks_and_boolean_seed_are_not_discarded(self):
        reports = [report(seed=0, outcome="pass"), report(seed=1, outcome="fail"),
                   report(seed=True, outcome="unknown", trial_source="/fixtures/boolean-seed")]
        result = coverage_report(reports, self.catalog)
        self.assertEqual(result["trial_count"], 3)
        self.assertEqual(result["task_count_with_trials"], 1)
        rows = {row["task"]: row for row in result["tasks"]}
        self.assertEqual(rows["whiteboard-wipe"]["recorded_outcome_counts"], {"pass": 1, "fail": 1, "unknown": 1})
        self.assertEqual(rows["whiteboard-wipe"]["recorded_seeds"], [0, 1])
        self.assertEqual(rows["sponge-grasp"]["trial_count"], 0)
        self.assertTrue(rows["sponge-grasp"]["coverage_gaps"])
        self.assertEqual(result["status"], "descriptive_only")
        self.assertIn("Causal benefit", " ".join(result["not_established"]))

    def test_unrecognized_task_keeps_samples_without_invented_contract(self):
        result = coverage_report([report(task="custom")], self.catalog)
        rows = {row["task"]: row for row in result["tasks"]}
        self.assertEqual(len(rows), 4)
        self.assertFalse(rows["custom"]["contract_available"])
        self.assertEqual(rows["custom"]["trial_count"], 1)

    def test_sidecar_presence_alone_is_not_complete_evidence(self):
        cases = ({"status": "missing"}, {"status": "empty"}, {"recorded_complete": False},
                 {"sampled_complete": False}, {"source": "legacy", "status": "legacy_partial"})
        for fields in cases:
            with self.subTest(fields=fields):
                result = coverage_report([report(**fields)], self.catalog)
                row = next(row for row in result["tasks"] if row["task"] == "whiteboard-wipe")
                self.assertTrue(any("sidecar" in gap.lower() or "evidence" in gap.lower() for gap in row["coverage_gaps"]))

    def test_complete_sidecar_and_distinct_seed_outcomes_remove_only_observed_coverage_gaps(self):
        result = coverage_report([report(seed=0, outcome="pass"), report(seed=1, outcome="fail")], self.catalog)
        row = next(row for row in result["tasks"] if row["task"] == "whiteboard-wipe")
        self.assertEqual(row["coverage_gaps"], [])
        self.assertTrue(result["not_established"])
        self.assertEqual(result["status"], "descriptive_only")

    def test_reexported_same_source_is_rejected_as_duplicate_not_new_independent_trial(self):
        first = report()
        second = deepcopy(first)
        second["trial"]["trial_id"] += ":reexport"
        second["report_path"] = "another-export/report.html"
        with self.assertRaises(ValueError): coverage_report([first, second], self.catalog)

    def test_missing_outcome_is_unknown_and_reporting_does_not_mutate_inputs(self):
        value = report(); value.pop("outcome")
        before, before_catalog = deepcopy(value), deepcopy(self.catalog)
        result = coverage_report([value], self.catalog)
        row = next(row for row in result["tasks"] if row["task"] == "whiteboard-wipe")
        self.assertEqual(row["recorded_outcome_counts"], {"unknown": 1})
        self.assertEqual(value, before); self.assertEqual(self.catalog, before_catalog)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
