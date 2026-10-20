"""Pure progress experiment invariants; no simulator or report mutation."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_progress_experiment.metrics import analyze_episode, summarize_population


def episode(progress=None, times=None, forces=None, gaps=None, **metadata):
    progress = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if progress is None else progress
    count = len(progress)
    return {
        "id": "test", "role": "policy", "task": "whiteboard-wipe", "outcome": "unknown", **metadata,
        "time_s": list(range(count)) if times is None else times, "progress": progress,
        "gap_after": [False] * count if gaps is None else gaps,
        "signals": {"object_surface": {"label": "Object-surface normal load", "unit": "N",
                                        "contact_role": "eraser_board", "values": [2.0] * count if forces is None else forces}},
    }


class ProgressMetricsTests(unittest.TestCase):
    def test_time_stretch_preserves_bins_but_not_duration(self):
        fast = analyze_episode(episode())
        slow = analyze_episode(episode(times=[10 + 3 * i for i in range(6)]))
        for left, right in zip(fast["bins"], slow["bins"]):
            self.assertEqual(left["sample_count"], right["sample_count"])
            self.assertEqual(left["progress_delta"], right["progress_delta"])
            self.assertAlmostEqual(right["observed_duration_s"], 3 * left["observed_duration_s"])
            self.assertEqual(left["signals"]["object_surface"]["mean_n"], right["signals"]["object_surface"]["mean_n"])
        self.assertEqual(fast["duration_s"], 5)
        self.assertEqual(slow["duration_s"], 15)

    def test_failed_endpoint_stays_point_four(self):
        result = analyze_episode(episode(progress=[0, 0.1, 0.4], outcome="fail"))
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["final_progress"], 0.4)
        self.assertEqual(result["max_progress"], 0.4)
        self.assertEqual(result["bins"][-1]["status"], "not_observed")
        self.assertEqual(result["milestones"][-1]["status"], "not_observed")
        self.assertIsNone(result["milestones"][-1]["first_observed_time_s"])

    def test_no_interpolation_of_jumped_bins(self):
        result = analyze_episode(episode(progress=[0, 0.8, 1.0]))
        self.assertEqual([item["status"] for item in result["bins"]],
                         ["observed", "not_observed", "not_observed", "not_observed", "observed"])
        self.assertEqual(result["bins"][0]["progress_delta"], 0.8)
        self.assertEqual(result["milestones"][2]["first_observed_time_s"], 1)

    def test_gaps_and_unknown_progress_are_not_invented(self):
        result = analyze_episode(episode(progress=[0, 0.2, None, 0.8, 1.0], gaps=[True, False, False, False, False]))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["gap_duration_s"], 1)
        self.assertEqual(result["observed_duration_s"], 3)
        self.assertEqual(result["unknown_progress_duration_s"], 1)
        self.assertIsNone(result["bins"][0]["progress_delta"])
        self.assertIsNone(result["bins"][1]["progress_delta"])
        self.assertEqual(result["bins"][1]["known_transition_duration_s"], 0)
        self.assertEqual(result["bins"][1]["observed_duration_s"], 1)

    def test_terminal_sample_peak_has_no_mean_or_integral_weight(self):
        result = analyze_episode(episode(progress=[0.8, 0.9, 1.0], forces=[2, 4, 900]))
        signal = result["bins"][-1]["signals"]["object_surface"]
        self.assertEqual(signal["mean_n"], 3)
        self.assertEqual(signal["p95_n"], 4)
        self.assertEqual(signal["sampled_peak_n"], 900)
        self.assertEqual(signal["load_integral_ns"], 6)
        self.assertEqual(signal["known_duration_s"], 2)

    def test_single_terminal_sample_reports_only_peak(self):
        result = analyze_episode(episode(progress=[0.5], times=[12.0], forces=[10]))
        item = result["bins"][2]
        signal = item["signals"]["object_surface"]
        self.assertEqual(item["status"], "observed")
        self.assertEqual(item["sample_count"], 1)
        self.assertIsNone(item["progress_delta"])
        self.assertEqual(signal["sampled_peak_n"], 10)
        for key in ("mean_n", "p95_n", "load_integral_ns", "coverage_fraction"):
            self.assertIsNone(signal[key], key)

    def test_regression_revisits_and_unchanged_are_descriptive(self):
        result = analyze_episode(episode(progress=[0.1, 0.3, 0.1, 0.1, 0.4]))
        self.assertEqual(result["regression_count"], 1)
        self.assertEqual(result["bins"][0]["visit_count"], 2)
        self.assertEqual(result["bins"][0]["sample_count"], 3)
        self.assertEqual(result["bins"][0]["unchanged_progress_duration_s"], 1)
        self.assertAlmostEqual(result["bins"][1]["progress_delta"], -0.2)
        self.assertEqual(result["outcome"], "unknown")

    def test_gap_splits_visits_and_does_not_count_regression(self):
        result = analyze_episode(episode(progress=[0.15, 0.1], gaps=[True, False]))
        self.assertEqual(result["bins"][0]["visit_count"], 2)
        self.assertEqual(result["regression_count"], 0)

    def test_all_missing_remains_unknown(self):
        result = analyze_episode(episode(progress=[None, None, None], forces=[None, None, None]))
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["max_progress"])
        self.assertEqual(result["unknown_progress_duration_s"], 2)
        self.assertTrue(all(item["status"] == "not_observed" for item in result["bins"]))
        self.assertIsNone(result["bins"][0]["signals"]["object_surface"]["load_integral_ns"])

    def test_missing_final_not_silently_replaced(self):
        result = analyze_episode(episode(progress=[0, 0.4, None]))
        self.assertIsNone(result["final_progress"])
        self.assertEqual(result["last_observed_progress"], 0.4)

    def test_zero_force_is_valid_not_missing(self):
        result = analyze_episode(episode(progress=[0, 0.1, 0.2], forces=[0, None, 0]))
        signal = result["bins"][0]["signals"]["object_surface"]
        self.assertEqual(signal["mean_n"], 0)
        self.assertEqual(signal["load_integral_ns"], 0)
        self.assertEqual(signal["coverage_fraction"], 0.5)

    def test_time_weighted_mean_and_quantile(self):
        result = analyze_episode(episode(progress=[0, 0, 0], times=[0, 99, 100], forces=[2, 200, 999]))
        signal = result["bins"][0]["signals"]["object_surface"]
        self.assertAlmostEqual(signal["mean_n"], 3.98)
        self.assertEqual(signal["p95_n"], 2)
        self.assertEqual(signal["sampled_peak_n"], 999)
        self.assertAlmostEqual(signal["load_integral_ns"], 398)

    def test_invalid_numeric_measurements_become_null_and_serializable(self):
        result = analyze_episode(episode(progress=[-0.1, float("nan"), 2, 0.4], forces=[-1, float("inf"), float("nan"), 0]))
        self.assertEqual([point["progress"] for point in result["points"]], [None, None, None, 0.4])
        self.assertEqual([point["signals"]["object_surface"] for point in result["points"]], [None, None, None, 0])
        json.dumps(result, allow_nan=False)

    def test_large_force_integral_overflow_is_unknown_not_infinity(self):
        result = analyze_episode(episode(progress=[0, 0], times=[0, 100], forces=[1e308, 1e308]))
        self.assertIsNone(result["bins"][0]["signals"]["object_surface"]["load_integral_ns"])
        json.dumps(result, allow_nan=False)

    def test_time_values_and_types_rejected(self):
        for times in ([0, 0], [1, 0], [0, None], [0, float("inf")], [0, True], [0, "1"], [-1e308, 1e308]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                analyze_episode(episode(progress=[0, 1], times=times))

    def test_dimension_and_type_errors_rejected(self):
        bad = []
        base = episode()
        for key, value in (("progress", [0]), ("gap_after", [False]), ("time_s", "0,1"),
                           ("role", "expert"), ("outcome", "good"), ("task", ""), ("signals", [])):
            item = copy.deepcopy(base)
            item[key] = value
            bad.append(item)
        for field, value in (("unit", "raw"), ("values", [0]), ("values", [True] * 6),
                             ("values", ["1"] * 6), ("contact_role", None)):
            item = copy.deepcopy(base)
            item["signals"]["object_surface"][field] = value
            bad.append(item)
        item = copy.deepcopy(base)
        item["gap_after"][0] = 0
        bad.append(item)
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                analyze_episode(item)

    def test_invalid_edges_rejected(self):
        for edges in ([0.1, 1], [0, 0.5], [0, 0.5, 0.5, 1], [0, float("nan"), 1], [False, 1]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                analyze_episode(episode(), edges)

    def test_empty_episode_and_population(self):
        result = analyze_episode(episode(progress=[], forces=[]))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["duration_s"], 0)
        self.assertEqual(summarize_population([])["groups"], {})

    def test_population_episode_weighted_and_keeps_failed_unknown(self):
        analyses = [
            analyze_episode(episode(id="short", progress=[0, 0.1], forces=[2, 2], outcome="pass")),
            analyze_episode(episode(id="long", progress=[0] * 101, forces=[10] * 101, outcome="fail")),
            analyze_episode(episode(id="missing", progress=[0, 0.1], forces=[None, None], outcome="unknown")),
            analyze_episode(episode(id="reference", role="expert_reference", progress=[0, 0.1], forces=[100, 100], outcome="pass")),
        ]
        result = summarize_population(analyses)
        group = result["groups"]["policy"]["whiteboard-wipe"]
        self.assertEqual(group["episode_count"], 3)
        self.assertEqual(group["outcome_counts"], {"pass": 1, "fail": 1, "unknown": 1})
        metric = group["bins"][0]["signals"]["object_surface"]["mean_n"]
        self.assertEqual(metric["mean"], 6)
        self.assertEqual(metric["known_episode_count"], 2)
        self.assertEqual(metric["unknown_episode_count"], 1)
        self.assertEqual(result["groups"]["expert_reference"]["whiteboard-wipe"]["episode_count"], 1)
        self.assertEqual(len(group["episodes"]), 3)

    def test_population_does_not_count_unvisited_bins_as_zero(self):
        group = summarize_population([analyze_episode(episode(progress=[0, 0.1], outcome="fail"))])["groups"]["policy"]["whiteboard-wipe"]
        item = group["bins"][-1]
        self.assertEqual(item["not_observed_episode_count"], 1)
        self.assertIsNone(item["summary"]["observed_duration_s"]["mean"])
        self.assertIsNone(item["signals"]["object_surface"]["mean_n"]["mean"])

    def test_population_rejects_duplicate_identity_different_edges_and_contact_role(self):
        first = analyze_episode(episode())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_population([first, first])
        with self.assertRaisesRegex(ValueError, "identical progress edges"):
            summarize_population([first, analyze_episode(episode(id="different"), [0, 0.5, 1])])
        other = episode(id="different", role="expert_reference")
        other["signals"]["object_surface"]["contact_role"] = "finger_eraser"
        with self.assertRaisesRegex(ValueError, "contact roles"):
            summarize_population([first, analyze_episode(other)])

    def test_input_is_not_mutated(self):
        source = episode()
        before = copy.deepcopy(source)
        analyze_episode(source)
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
