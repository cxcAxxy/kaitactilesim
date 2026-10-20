"""Synthetic display contracts: no archived logs, HDF, model or physics reads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_unified import report


def fixture_result(*, whiteboard=False):
    definitions = {"stage1": "picked_up"}
    flags, first = {"stage1": True}, {"stage1": 123.49}
    if not whiteboard:
        definitions["stage2"] = "inserted_and_seated"
        flags["stage2"], first["stage2"] = False, None
    times = [123.5, 123.531, 123.566, 125.0, 125.034]
    rows = []
    for index, time in enumerate(times):
        measurements = {"lift_from_initial_m": index * .01,
                        "hand_normal_load_n": None if index == 2 else index * .25,
                        "opposed_grasp_now": [True, False, None, True, False][index]}
        if index == 0:
            measurements["table_normal_load_n"] = 0.0
        rows.append({"simulation_time_s": time, "time_s": time,
                     "control_tick": index, "measurements": measurements})
    return {
        "native_result": {"task_success": False, "status": "task_not_completed",
                          "stop_reason": "time_limit", "physics_valid": True,
                          "diagnostic_only": True, "formal_metrics_valid": False},
        "staged_result": {"version": "staged_v1", "stage_definitions": definitions,
                          "stage_flags": flags, "first_times": first, "highest_stage": 1,
                          "complete_success": None if whiteboard else False,
                          "whiteboard_complete_success_defined": False if whiteboard else None,
                          "stage_achievement_policy": "ordered_cumulative_ever_reached",
                          "thresholds": {"lift_m": .02, "grasp_hold_s": .15},
                          "continuous_metrics": {"cleaned_fraction": None} if whiteboard else {},
                          "final_facts": {"seated_now": None, "missing_measurements": ["seated_now"]}},
        "state_rows": rows,
        "rule_identity": {"sha256": "d" * 64, "observer_sha256": "e" * 64,
                          "current_rules_reexecuted": False},
        "data_evidence": {"sample_count": len(rows), "interval_gap_count": 1,
                          "clock": "mujoco_data_time_seconds", "video_tactile_alignment": "not_performed"},
        "warnings": ["Archived cumulative milestones are not current physical state."],
        "provenance": [
            {"path": "/synthetic/seed_000/summary.json", "sha256": "a" * 64,
             "size": 100, "role": "staged_summary"},
            {"path": "/synthetic/seed_000/staged_states.jsonl", "sha256": "b" * 64,
             "size": 200, "role": "staged_state_log"},
        ],
    }


def fixture_case():
    visualization = {"media": {"sync_status": "unverified", "time_s": [],
                               "reason": "Independent video; no physical phase certification"},
                     "tactile_series": {"time_s": [0.0, .25, 2.0], "normal_force_n": [[1.], [None], [2.]]}}
    provenance = [
        {"path": "/synthetic/seed_000/summary.json", "sha256": "a" * 64,
         "size": 100, "role": "recorded_trial_outcome", "native_note": "preserve this evidence row"},
        {"path": "/synthetic/seed_000/evaluation_rollout_trace.npz", "sha256": "c" * 64,
         "size": 300, "role": "native_control_cadence_tactile"},
    ]
    return {
        "config": {"task": "whiteboard-wipe", "native_setting": "unchanged"},
        "report": {"trial": {"trial_id": "native-trial-id", "task": "whiteboard-wipe"},
                   "outcome": {"status": "fail", "success": False, "diagnostic_only": True,
                               "formal_metrics_valid": False},
                   "task": {"stage1": "native semantics", "success": False, "coverage": None},
                   "tactile": {"status": "available", "mean_fn_n": 1.0},
                   "warnings": ["Native evidence warning"], "provenance": deepcopy(provenance),
                   "visualization": {"path": "visualization.json", "sha256": report.content_hash(visualization)}},
        "visualization": visualization,
        "tactile_loads": {"time_s": [0.0, .25, 2.0], "normal_total_n": [1., None, 2.],
                          "provenance": deepcopy(provenance)},
        "copies": [{"source": Path("/synthetic/seed_000/review.mp4"), "url": "media/rollout.mp4"}],
    }


class StagedProjectionTests(unittest.TestCase):
    def test_native_result_and_staged_achievement_never_replace_each_other(self):
        result = fixture_result()
        result["staged_result"]["complete_success"] = True
        result["staged_result"]["stage_flags"]["stage2"] = True
        result["staged_result"]["first_times"]["stage2"] = 124.0
        display = report.projection(result, 123.0)
        self.assertFalse(display["native_result"]["task_success"])
        self.assertEqual(display["native_result"]["status"], "task_not_completed")
        self.assertTrue(display["staged_result"]["complete_success"])
        self.assertTrue(display["native_result"]["diagnostic_only"])
        self.assertFalse(display["native_result"]["formal_metrics_valid"])

    def test_whiteboard_pickup_is_not_completion_or_cleaning_success(self):
        display = report.projection(fixture_result(whiteboard=True), 123.0)
        self.assertIsNone(display["staged_result"]["complete_success"])
        self.assertIs(display["staged_result"]["whiteboard_complete_success_defined"], False)
        self.assertEqual(len(display["stages"]), 1)
        card = display["stages"][0]
        self.assertEqual((card["key"], card["name"]), ("stage1", "picked_up"))
        self.assertEqual(card["label"], report.STAGE_LABELS["picked_up"])
        self.assertTrue(card["reached"])
        self.assertNotIn("success", card)
        self.assertIsNone(display["staged_result"]["continuous_metrics"]["cleaned_fraction"])

    def test_first_achievement_and_all_samples_subtract_same_native_origin(self):
        source = fixture_result()
        first = deepcopy(source["staged_result"]["first_times"])
        times = [row["simulation_time_s"] for row in source["state_rows"]]
        display = report.projection(source, 123.0)
        self.assertAlmostEqual(display["stages"][0]["first_time_s"], .49)
        self.assertIsNone(display["stages"][1]["first_time_s"])
        self.assertEqual(display["staged_result"]["first_times"], first)
        self.assertEqual(display["timeline"]["time_s"], [time - 123.0 for time in times])
        # Do not independently zero-align staged time at its first observation.
        self.assertEqual(display["timeline"]["time_s"][0], .5)
        self.assertEqual(display["timeline"]["origin_simulation_time_s"], 123.0)

    def test_projection_does_not_consume_source_rows_and_can_be_repeated(self):
        source = fixture_result()
        before = deepcopy(source)
        first = report.projection(source, 123.0)
        self.assertEqual(source, before)
        second = report.projection(source, 123.0)
        self.assertEqual(first, second)

    def test_raw_timestamps_values_nulls_and_real_zero_are_not_resampled(self):
        source = fixture_result()
        original_rows = deepcopy(source["state_rows"])
        timeline = report.projection(source, 123.0)["timeline"]
        self.assertEqual(len(timeline["time_s"]), len(original_rows))
        keys = [item["key"] for item in timeline["signals"]]
        for index, row in enumerate(original_rows):
            self.assertEqual(timeline["values"][index], [row["measurements"].get(key) for key in keys])
        self.assertIsNone(timeline["values"][2][keys.index("hand_normal_load_n")])
        self.assertEqual(timeline["values"][0][keys.index("table_normal_load_n")], 0.0)
        self.assertIsNone(timeline["values"][1][keys.index("table_normal_load_n")])
        self.assertEqual(timeline["gap_after"], [False, False, True, False, False])
        self.assertAlmostEqual(timeline["nominal_interval_s"], (.034 + .035) / 2)
        self.assertIn("no interpolation", timeline["sampling"])

    def test_instantaneous_conditions_are_not_filled_from_latched_stage(self):
        display = report.projection(fixture_result(), 123.0)
        self.assertTrue(display["stages"][0]["reached"])
        timeline = display["timeline"]
        keys = [item["key"] for item in timeline["conditions"]]
        self.assertEqual([row[keys.index("opposed_grasp_now")] for row in timeline["condition_values"]],
                         [True, False, None, True, False])
        self.assertTrue(all(row[keys.index("seated_now")] is None for row in timeline["condition_values"]))

    def test_unknown_definition_and_absent_measurements_remain_unknown(self):
        result = fixture_result()
        result["staged_result"].update(stage_definitions={"stage9": "archived_unknown_rule"},
                                       stage_flags={}, first_times={"stage9": None}, complete_success=None)
        result["state_rows"] = [{"simulation_time_s": 124.0, "measurements": {}}]
        display = report.projection(result, 123.0)
        self.assertEqual(display["stages"][0]["label"], "archived_unknown_rule")
        self.assertIsNone(display["stages"][0]["reached"])
        self.assertIsNone(display["stages"][0]["first_time_s"])
        self.assertIsNone(display["staged_result"]["complete_success"])
        self.assertTrue(all(value is None for value in display["timeline"]["values"][0]))
        self.assertTrue(all(value is None for value in display["timeline"]["condition_values"][0]))
        self.assertIsNone(display["timeline"]["nominal_interval_s"])
        json.dumps(display, allow_nan=False)

    def test_empty_rows_do_not_create_a_fake_zero_measurement(self):
        result = fixture_result()
        result["state_rows"] = []
        timeline = report.projection(result, 123.0)["timeline"]
        for key in ("time_s", "values", "condition_values", "gap_after"):
            self.assertEqual(timeline[key], [])
        self.assertIsNone(timeline["nominal_interval_s"])

    def test_nonfinite_native_origin_and_invalid_recorded_clocks_are_rejected(self):
        for origin in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                report.projection(fixture_result(), origin)
        for timestamp in (None, True, "123", float("nan"), float("inf")):
            result = fixture_result()
            result["state_rows"][0]["simulation_time_s"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                report.projection(result, 123.0)


class AugmentStagedCaseTests(unittest.TestCase):
    def setUp(self):
        self.trial = SimpleNamespace(root=Path("/synthetic/seed_000"), time_s=[123.0, 123.25, 125.0],
                                     task_tactile_clock_aligned=True)

    def augment(self, case, result):
        with patch.object(report, "load_staged_result", return_value=result) as loader:
            returned = report.augment_staged_case(case, self.trial)
        loader.assert_called_once_with(self.trial.root, include_state_log=True)
        self.assertIs(returned, case)
        return returned

    def test_augment_preserves_native_report_media_tactile_and_copy_fields(self):
        case, result = fixture_case(), fixture_result(whiteboard=True)
        before = deepcopy(case)
        after = self.augment(case, result)
        for key, value in before["report"].items():
            if key != "provenance":
                self.assertEqual(after["report"][key], value, key)
        self.assertEqual(after["visualization"], before["visualization"])
        self.assertEqual(after["copies"], before["copies"])
        for key, value in before["tactile_loads"].items():
            if key != "provenance":
                self.assertEqual(after["tactile_loads"][key], value)
        self.assertEqual(after["config"]["native_setting"], "unchanged")
        display = after["config"]["staged_evaluation"]
        self.assertEqual(display["timeline"]["model_clock_mapping"], "mapped")
        self.assertEqual(display["timeline"]["origin_simulation_time_s"], 123.0)
        self.assertEqual(display["timeline"]["time_s"][0], .5)
        self.assertIsNone(after["report"]["recorded_staged_evaluation"]["staged_result"]["complete_success"])
        self.assertNotIn("timeline", after["report"]["recorded_staged_evaluation"])
        self.assertEqual(after["report"]["visualization"]["sha256"], report.content_hash(after["visualization"]))

    def test_native_provenance_fields_are_retained_and_staged_witnesses_are_separate(self):
        case, result = fixture_case(), fixture_result()
        original = deepcopy(case["report"]["provenance"])
        staged_provenance = deepcopy(result["provenance"])
        after = self.augment(case, result)
        by_path = {row["path"]: row for row in after["report"]["provenance"]}
        for row in original:
            for key, value in row.items():
                self.assertEqual(by_path[row["path"]].get(key), value, (row["path"], key))
        self.assertEqual(after["unified_staged"]["provenance"], staged_provenance)
        self.assertEqual(after["tactile_loads"]["provenance"], after["report"]["provenance"])
        self.assertIn("/synthetic/seed_000/staged_states.jsonl", by_path)
        self.assertEqual(after["unified_staged"]["rule_identity"], result["rule_identity"])

    def test_conflicting_source_hash_or_size_is_not_silently_rebound(self):
        for field, changed in (("sha256", "f" * 64), ("size", 101)):
            result = fixture_result()
            result["provenance"][0][field] = changed
            with patch.object(report, "load_staged_result", return_value=result):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Conflicting"):
                    report.augment_staged_case(fixture_case(), self.trial)

    def assert_independent_case_preserved(self, case):
        result = fixture_result(whiteboard=True)
        source = deepcopy(result)
        before = deepcopy(case)
        after = self.augment(case, result)
        display = after["config"]["staged_evaluation"]
        timeline = display["timeline"]
        self.assertEqual(timeline["model_clock_mapping"], "unmapped")
        self.assertIsNone(timeline["origin_simulation_time_s"])
        self.assertEqual(timeline["clock"], "independent archived simulation_time_s")
        self.assertEqual(timeline["time_s"], [row["simulation_time_s"] for row in source["state_rows"]])
        self.assertEqual(display["stages"][0]["first_time_s"], 123.49)
        self.assertEqual(display["staged_result"]["first_times"], source["staged_result"]["first_times"])
        self.assertEqual(display["native_result"], source["native_result"])
        self.assertIsNone(display["staged_result"]["complete_success"])
        self.assertTrue(any("独立时间" in warning for warning in display["warnings"]))
        self.assertEqual(result, source)
        for key, value in before["report"].items():
            if key != "provenance":
                self.assertEqual(after["report"][key], value, key)
        self.assertEqual(after["visualization"], before["visualization"])
        self.assertEqual(after["copies"], before["copies"])
        for key, value in before["tactile_loads"].items():
            if key != "provenance":
                self.assertEqual(after["tactile_loads"][key], value, key)
        by_path = {row["path"]: row for row in after["report"]["provenance"]}
        for row in before["report"]["provenance"]:
            self.assertEqual(by_path[row["path"]], row)
        self.assertEqual(after["unified_staged"]["provenance"], source["provenance"])
        self.assertEqual(after["tactile_loads"]["provenance"], after["report"]["provenance"])
        self.assertIn("recorded_staged_evaluation", after["report"])

    def test_unverified_tactile_clock_never_implicitly_subtracts_its_origin(self):
        self.trial.time_s = [100.0, 100.25, 102.0]
        self.trial.task_tactile_clock_aligned = False
        self.assert_independent_case_preserved(fixture_case())

    def test_absent_native_clock_keeps_case_and_absolute_staged_clock(self):
        for times in (None, []):
            with self.subTest(times=times):
                self.trial.time_s = times
                # An alignment flag alone cannot establish a missing origin.
                self.trial.task_tactile_clock_aligned = True
                case = fixture_case()
                case["report"]["tactile"] = {"status": "missing", "mean_fn_n": None}
                case["tactile_loads"].update(status="missing", time_s=[], normal_total_n=[])
                self.assert_independent_case_preserved(case)

    def test_absent_alignment_declaration_does_not_infer_shared_clock(self):
        del self.trial.task_tactile_clock_aligned
        self.assert_independent_case_preserved(fixture_case())


if __name__ == "__main__":
    unittest.main()
