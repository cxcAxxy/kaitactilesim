"""Action-review discovery is complete, factual, and independent of outcomes."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_action_review.events import build_review, DEFAULT_CONFIG


def episode(progress, times=None, force=None, gaps=None, **metadata):
    count = len(progress)
    return {"id": "policy:test", "role": "policy", "task": "whiteboard-wipe", "outcome": "fail", **metadata,
            "time_s": list(range(count)) if times is None else times,
            "progress": progress, "gap_after": [False] * count if gaps is None else gaps,
            "signals": {"board_normal": {"label": "Board load", "unit": "N", "contact_role": "eraser_board",
                                         "values": [1.0] * count if force is None else force},
                        "hand_normal_sum": {"label": "Hand load", "unit": "N", "contact_role": "unverified",
                                            "values": [None] * count}}}


class ActionReviewEventsTests(unittest.TestCase):
    def test_native_windows_and_preview_do_not_drop_events(self):
        result = build_review(episode([0, 0, 0, .2, .3, .3, .3, .4, .5, .5, .5]))
        self.assertEqual([(event["kind"], event["start_s"], event["end_s"]) for event in result["events"]],
                         [("plateau", 0, 2), ("growth", 2, 4), ("plateau", 4, 6), ("growth", 6, 8), ("plateau", 8, 10)])
        self.assertEqual(len(result["preview_event_ids"]), 3)
        self.assertEqual(len(result["events"]), 5)
        self.assertEqual(result["coverage"]["marked_duration_s"], 10)
        self.assertEqual(result["outcome"], "fail")

    def test_gaps_split_both_plateau_and_growth_without_fill(self):
        result = build_review(episode([0, 0, 0, 0, .1, .2, .3], gaps=[False, False, True, False, True, False, False]))
        self.assertEqual([(e["kind"], e["start_s"], e["end_s"]) for e in result["events"]],
                         [("plateau", 0, 2), ("growth", 3, 4), ("growth", 5, 6)])
        self.assertEqual(result["coverage"]["gap_duration_s"], 2)
        self.assertEqual(result["coverage"]["marked_duration_s"], 4)

    def test_null_progress_never_bridged(self):
        result = build_review(episode([0, 0, None, 0, 0, 0, .1]))
        self.assertEqual([(e["kind"], e["start_s"], e["end_s"]) for e in result["events"]],
                         [("plateau", 3, 5), ("growth", 5, 6)])
        self.assertEqual(result["coverage"]["unknown_progress_duration_s"], 2)
        self.assertEqual(result["coverage"]["unmarked_duration_s"], 3)

    def test_quantized_plateau_is_observation_not_physical_failure(self):
        result = build_review(episode([.2, .2, .2, .21]))
        event = result["events"][0]
        self.assertEqual(event["kind"], "plateau")
        self.assertEqual(event["progress_delta"], 0)
        self.assertIn("记录未变化", event["label"])
        self.assertTrue(any("量化" in text for text in result["limitations"]))
        self.assertNotIn("score", result)
        self.assertNotIn("diagnosis", event)

    def test_initial_zero_and_terminal_one_are_not_deleted_or_failed(self):
        result = build_review(episode([0, 0, 0, 1, 1, 1], outcome="pass"))
        plateaus = [e for e in result["events"] if e["kind"] == "plateau"]
        self.assertEqual([e["progress_region"] for e in plateaus], ["initial_zero", "terminal_one"])
        self.assertEqual(result["outcome"], "pass")
        self.assertIn("接近", plateaus[0]["label"])
        self.assertIn("收尾", plateaus[1]["label"])

    def test_no_success_selection(self):
        reviews = [build_review(episode([0, 0, 0], outcome=outcome)) for outcome in ("pass", "fail", "unknown")]
        self.assertEqual([r["outcome"] for r in reviews], ["pass", "fail", "unknown"])
        self.assertEqual(reviews[0]["events"], reviews[1]["events"])
        self.assertEqual(reviews[1]["events"], reviews[2]["events"])

    def test_coverage_conservation_no_context_double_counting(self):
        result = build_review(episode([0, 0, 0, .1, None, .2, .2, .2, .1], gaps=[False, False, False, False, True, False, False, False, False]))
        coverage = result["coverage"]
        self.assertAlmostEqual(coverage["marked_duration_s"] + coverage["unmarked_duration_s"] + coverage["gap_duration_s"], coverage["total_duration_s"])
        self.assertAlmostEqual(sum(e["duration_s"] for e in result["events"]), coverage["marked_duration_s"])
        self.assertLessEqual(coverage["unknown_progress_duration_s"], coverage["unmarked_duration_s"])

    def test_recorded_timestamps_and_context_not_invented(self):
        times = [10.03, 11.31, 12.51, 13.71, 14.82]
        result = build_review(episode([0, .2, .2, .2, .4], times=times))
        for event in result["events"]:
            for key in ("start_s", "end_s", "context_start_s", "context_end_s"):
                self.assertIn(event[key], times)
        plateau = next(e for e in result["events"] if e["kind"] == "plateau")
        self.assertEqual(plateau["start_s"], 11.31)
        self.assertEqual(plateau["end_s"], 13.71)

    def test_zero_force_is_known_not_missing_or_proof_of_separation(self):
        event = build_review(episode([.2, .2, .2], force=[0, 0, 0]))["events"][0]
        stats = event["contact_stats"]
        self.assertEqual(stats["observation"], "all_zero")
        self.assertEqual(stats["time_weighted_mean_n"], 0)
        self.assertEqual(stats["known_duration_s"], 2)
        self.assertEqual(stats["zero_sample_count"], 3)
        self.assertNotIn("失去接触", event["label"])

    def test_missing_force_and_terminal_force_weight(self):
        event = build_review(episode([.2, .2, .2], force=[None, 2, 99]))["events"][0]
        stats = event["contact_stats"]
        self.assertEqual(stats["observation"], "partial")
        self.assertEqual(stats["time_weighted_mean_n"], 2)
        self.assertEqual(stats["max_n"], 99)
        self.assertEqual(stats["coverage_fraction"], .5)
        self.assertIsNone(event["hand_stats"]["min_n"])

    def test_preview_rule_uses_entire_series_not_case_time(self):
        data = episode([0, 0, 0, .1, .1, .1, .1, .2], times=[100 + i for i in range(8)], force=[0, 0, 0, 1, 1, 1, 1, 0])
        result = build_review(data)
        preferred = next(e for e in result["events"] if e["id"] == result["preferred_event_id"])
        self.assertEqual((preferred["start_s"], preferred["end_s"]), (103, 106))
        self.assertEqual(result["rules"]["min_plateau_s"], DEFAULT_CONFIG["min_plateau_s"])
        self.assertFalse(result["rules"]["data_adaptive_thresholds"])
        self.assertEqual(result["events"], build_review(copy.deepcopy(data))["events"])

    def test_plateau_tolerance_does_not_allow_unbounded_cumulative_drift(self):
        result = build_review(episode([0, .9e-12, 1.8e-12, 2.7e-12, 3.6e-12]))
        self.assertEqual(result["events"], [])
        self.assertEqual(result["coverage"]["unmarked_duration_s"], 4)

    def test_growth_not_joined_over_regression_or_flat_interval(self):
        result = build_review(episode([0, .1, .2, .2, .1, .2, .3]))
        self.assertEqual([(e["start_s"], e["end_s"]) for e in result["events"]], [(0, 2), (4, 6)])
        self.assertEqual(result["coverage"]["unmarked_duration_s"], 2)

    def test_motion_requires_exact_clock_and_preserves_partial_path(self):
        data = episode([.2] * 4)
        motion = {"time_s": [0, 1, 2, 3], "position_m": [[0, 0, 0], [.1, 0, 0], None, [.3, 0, 0]],
                  "label": "Wrist", "object": "wrist", "frame": "world"}
        result = build_review(data, motion=motion)
        stats = result["events"][0]["motion_stats"]
        self.assertEqual(stats["status"], "partial")
        self.assertAlmostEqual(stats["wrist_path_m"], .1)
        self.assertIsNone(stats["wrist_displacement_m"])
        self.assertEqual(stats["known_duration_s"], 1)
        self.assertIsNone(result["series"][2]["position_m"])
        motion["time_s"][1] = 1.000001
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_review(data, motion=motion)

    def test_motion_gap_not_bridged_and_full_path_has_net_displacement(self):
        data = episode([.2, .2, .2])
        motion = {"time_s": [0, 1, 2], "position_m": [[0, 0, 0], [.1, 0, 0], [0, 0, 0]]}
        full = build_review(data, motion=motion)["events"][0]["motion_stats"]
        self.assertEqual(full["status"], "available")
        self.assertAlmostEqual(full["wrist_path_m"], .2)
        self.assertEqual(full["wrist_displacement_m"], 0)
        motion["gap_after"] = [True, False, False]
        partial = build_review(data, motion=motion)["events"][0]["motion_stats"]
        self.assertEqual(partial["status"], "partial")
        self.assertAlmostEqual(partial["wrist_path_m"], .1)
        self.assertIsNone(partial["wrist_displacement_m"])
        self.assertEqual(partial["coverage_fraction"], .5)

    def test_all_missing_empty_and_single_point_produce_no_fabricated_events(self):
        for progress in ([None, None, None], [], [0]):
            with self.subTest(progress=progress):
                result = build_review(episode(progress))
                self.assertEqual(result["events"], [])
                self.assertIsNone(result["preferred_event_id"])
                self.assertEqual(result["preview_event_ids"], [])
                json.dumps(result, allow_nan=False)

    def test_bad_config_and_motion_dimensions_rejected(self):
        for config in ({"min_plateau_s": 0}, {"context_s": -1}, {"equality_tolerance": float("nan")},
                       {"equality_tolerance": 1}, {"min_plateau_s": True}, {"special_case_time": 62.634}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_review(episode([0, 0, 0]), config=config)
        with self.assertRaises(ValueError):
            build_review(episode([0, 0, 0]), motion={"time_s": [0, 1, 2], "position_m": [[0, 1]] * 3})

    def test_no_mutation(self):
        data = episode([0, 0, 0, .1])
        before = copy.deepcopy(data)
        build_review(data)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
