"""Task display mappings preserve archived semantics without simulator imports."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.adapters import TrialData
from evaluation_workbench.task_fields import (
    augment_task_visualization, build_task_summary, task_presentation,
)


def fixture(task="whiteboard-wipe", *, rows=None, summary=None, origin=0):
    trial = TrialData(root=Path("/synthetic/read-only/trial"), task=task,
                      summary=summary or {"task": task, "evaluation": {"success": False}},
                      control_rows=rows or [], task_tactile_clock_aligned=True)
    report = {"trial": {"task": task}, "outcome": {"status": "fail"},
              "protocol": {"max_gap_factor": 1.5}}
    visual = {"time_origin_s": origin, "clock_aligned": True,
              "task_series": [{"key": "untrusted_arbitrary_numeric_field"}],
              "tactile_series": {"time_s": [0, 1], "normal_force_n": [[0], [1]]}}
    return trial, report, visual


def control(time, **values):
    return {"sim_time": time, "task": values, "_source": f"control_steps.jsonl:{time}"}


def frame(time, **values):
    return {"simulation_time_s": time, "task_metrics": values, "_source": f"frames.jsonl:{time}"}


class TaskPresentationTests(unittest.TestCase):
    def test_all_requested_tasks_have_distinct_explicit_presentations(self):
        tasks = ("whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
                 "install-ram", "usb-insert", "usb-unplug", "bulb-screw", "bulb-unscrew", "poker-draw", "pick-place")
        titles = set()
        for task in tasks:
            with self.subTest(task=task):
                presentation = task_presentation(task)
                self.assertTrue(presentation["supported"])
                self.assertNotIn(presentation["title"], titles)
                titles.add(presentation["title"])
                self.assertIsInstance(presentation["final_metrics"], list)
                self.assertIn("不是独立复核", presentation["task_note"])
                json.dumps(presentation, allow_nan=False)
                if not task.startswith("whiteboard-wipe"):
                    self.assertNotIn("board_normal_force_n", presentation["primary_task_keys"])
                    self.assertNotIn("cleaned_fraction", presentation["primary_task_keys"])

    def test_horizontal_requires_own_pickup_and_home_labels(self):
        keys = lambda task: {row["key"] for row in task_presentation(task)["final_metrics"]}
        self.assertTrue({"pickup_verified", "home_return_verified"} <= keys("whiteboard-wipe-horizontal"))
        self.assertNotIn("home_return_verified", keys("whiteboard-wipe"))

    def test_usb_unplug_title_and_protocol_note_do_not_relabel_historical_results(self):
        p = task_presentation("usb-unplug")
        self.assertEqual(p["title"], "USB 拔出")
        self.assertIn("连续手持至少 2 秒", p["task_note"])
        self.assertIn("历史严格放回规则以各条归档协议为准", p["task_note"])
        finals = {row["key"] for row in p["final_metrics"]}
        self.assertTrue({"extraction_hold_duration_s", "extraction_hold_now", "extraction_hold_verified",
                         "placed_on_table", "released", "hand_retreated"} <= finals)

    def test_bulb_primary_and_final_fields_use_canonical_degree_keys(self):
        expected = {
            "bulb-screw": ("clockwise_angle_deg", "灯泡净旋入角度（顺时针）"),
            "bulb-unscrew": ("angle_removed_deg", "灯泡净旋出角度（逆时针）"),
            "bulb-screw-inplace": ("rotation_from_initial_deg", "相对实际初态的净旋入角度"),
        }
        for task, (key, label) in expected.items():
            with self.subTest(task=task):
                presentation = task_presentation(task)
                self.assertIn(key, presentation["primary_task_keys"])
                self.assertFalse(any(name.endswith("_turns") for name in presentation["primary_task_keys"]))
                final = next(row for row in presentation["final_metrics"] if row["key"] == key)
                self.assertEqual((final["label"], final["unit"]), ("终态" + label, "deg"))
                self.assertEqual(final["conversion"]["factor"], 360.0)

    def test_catalog_objects_are_not_mutable_global_state(self):
        p = task_presentation("usb-insert")
        p["primary_task_keys"].append("board_normal_force_n")
        p["final_metrics"][0]["source_paths"].append("made_up")
        self.assertNotIn("board_normal_force_n", task_presentation("usb-insert")["primary_task_keys"])
        self.assertNotIn("made_up", task_presentation("usb-insert")["final_metrics"][0]["source_paths"])

    def test_unknown_task_does_not_inherit_whiteboard(self):
        p = task_presentation("novel-task")
        self.assertFalse(p["supported"])
        self.assertEqual(p["primary_task_keys"], [])
        t, r, v = fixture("novel-task", rows=[control(0, cleaned_fraction=.5)])
        result = augment_task_visualization(t, r, v)
        self.assertEqual(result["task_series_status"], "unsupported")
        self.assertEqual(v["task_series"], [])


class TaskSeriesTests(unittest.TestCase):
    def test_native_clock_origin_gaps_and_missing_values_preserved(self):
        rows = [control(10, cleaned_fraction=0), control(10.1, cleaned_fraction=None),
                control(10.2, cleaned_fraction=.1), control(11, cleaned_fraction=.2)]
        rows[0]["gap_after"] = True
        t, r, v = fixture(rows=rows, origin=10)
        before_rows, before_report, before_tactile = deepcopy(rows), deepcopy(r), deepcopy(v["tactile_series"])
        result = augment_task_visualization(t, r, v)
        series = v["task_series"][0]
        self.assertEqual(series["values"], [0, None, .1, .2])
        self.assertEqual(series["gap_after"], [True, False, True, False])
        self.assertAlmostEqual(series["time_s"][1], .1)
        self.assertEqual(series["time_s"][-1], 1)
        self.assertEqual(series["status"], "partial")
        self.assertEqual(rows, before_rows)
        self.assertEqual(r, before_report)
        self.assertEqual(v["tactile_series"], before_tactile)
        self.assertEqual(result["review_status"], "not_verified")

    def test_frames_supply_absent_fields_without_resampling(self):
        t, r, v = fixture("install-ram", rows=[control(0, insertion_depth_m=0), control(.033, insertion_depth_m=.003)])
        frames = [frame(0, axial_resistance_n=1), frame(.1, axial_resistance_n=2), frame(.2, axial_resistance_n=3)]
        before = deepcopy(frames)
        augment_task_visualization(t, r, v, frame_rows=frames)
        series = {row["key"]: row for row in v["task_series"]}
        self.assertEqual(series["insertion_depth_m"]["time_s"], [0, .033])
        self.assertEqual(series["axial_resistance_n"]["time_s"], [0, .1, .2])
        self.assertEqual(series["axial_resistance_n"]["source"], "frames.task_metrics.axial_resistance_n")
        self.assertEqual(frames, before)

    def test_frames_never_backfill_partial_or_entirely_null_control_field(self):
        for value in (None, 1):
            with self.subTest(value=value):
                t, r, v = fixture("install-ram", rows=[control(0, axial_resistance_n=None), control(.1, axial_resistance_n=value)])
                result = augment_task_visualization(t, r, v, frame_rows=[frame(0, axial_resistance_n=5), frame(.1, axial_resistance_n=6)])
                self.assertEqual(v["task_series"][0]["values"], [None, value])
                self.assertEqual(v["task_series"][0]["source"], "control_rows.task.axial_resistance_n")
                self.assertEqual(result["task_series_status"], "missing" if value is None else "partial")

    def test_invalid_control_clock_not_silently_replaced_by_frames(self):
        for times in ([0, 0], [1, 0], [0, None], [0, math.inf], [0, True]):
            with self.subTest(times=times):
                t, r, v = fixture("install-ram", rows=[control(times[0], insertion_depth_m=0), control(times[1], insertion_depth_m=.006)])
                result = augment_task_visualization(t, r, v, frame_rows=[frame(0, insertion_depth_m=.006), frame(1, insertion_depth_m=.006)])
                self.assertEqual(v["task_series"], [])
                self.assertTrue(result["warnings"])

    def test_clock_unverified_has_summary_but_no_shared_cursor_curves(self):
        for change in ("trial", "visual", "origin"):
            with self.subTest(change=change):
                t, r, v = fixture(rows=[control(0, cleaned_fraction=0)])
                if change == "trial": t.task_tactile_clock_aligned = False
                elif change == "visual": v["clock_aligned"] = False
                else: v["time_origin_s"] = None
                result = augment_task_visualization(t, r, v)
                self.assertEqual(result["task_series_status"], "unverified_clock")
                self.assertEqual(v["task_series"], [])
                self.assertTrue(result["task_summary"])

    def test_boolean_string_nan_inf_and_invalid_fraction_are_unknown(self):
        values = [0, .2, None, True, "0.5", float("nan"), float("inf"), -.1, 1.1]
        t, r, v = fixture(rows=[control(i, cleaned_fraction=value) for i, value in enumerate(values)])
        augment_task_visualization(t, r, v)
        self.assertEqual(v["task_series"][0]["values"], [0, .2, None, None, None, None, None, None, None])
        json.dumps(v, allow_nan=False)

    def test_zero_normal_force_is_valid_negative_is_unknown(self):
        t, r, v = fixture(rows=[control(0, board_normal_force_n=0), control(1, board_normal_force_n=-.1)])
        augment_task_visualization(t, r, v)
        self.assertEqual(v["task_series"][0]["values"], [0, None])

    def test_signed_quantities_do_not_get_clipped_or_normalized(self):
        t, r, v = fixture("bulb-unscrew", rows=[control(0, insertion_depth_m=-.003, thread_error_m=-.002, turns_removed=-.1)])
        augment_task_visualization(t, r, v)
        series = {row["key"]: row for row in v["task_series"]}
        self.assertEqual(series["insertion_depth_m"]["values"], [-.003])
        self.assertEqual(series["thread_error_m"]["values"], [-.002])
        self.assertEqual(series["angle_removed_deg"]["values"], [-36.0])
        self.assertEqual(series["angle_removed_deg"]["unit"], "deg")
        self.assertEqual(series["angle_removed_deg"]["source"], "control_rows.task.turns_removed")
        self.assertEqual(series["angle_removed_deg"]["conversion"]["factor"], 360.0)
        self.assertNotIn("progress", series)

    def test_legacy_bulb_turn_fields_derive_degree_curves_without_filling_missing(self):
        cases = (
            ("bulb-screw", "clockwise_turns", "clockwise_angle_deg", .25, 90.0),
            ("bulb-unscrew", "turns_removed", "angle_removed_deg", .125, 45.0),
            ("bulb-screw-inplace", "rotation_from_initial_turns", "rotation_from_initial_deg", .1, 36.0),
        )
        for task, source_key, display_key, raw, expected in cases:
            with self.subTest(task=task):
                rows = [control(0), control(.1, **{source_key: raw})]
                before = deepcopy(rows)
                t, r, v = fixture(task, rows=rows)
                augment_task_visualization(t, r, v)
                series = {row["key"]: row for row in v["task_series"]}[display_key]
                self.assertEqual(series["values"], [None, expected])
                self.assertEqual(series["unit"], "deg")
                self.assertEqual(series["status"], "partial")
                self.assertEqual(rows, before)

    def test_usb_frame_units_and_gated_zero_remain_exactly_recorded(self):
        rows = [frame(0, insertion_depth_mm=0, orientation_error_deg=90, aligned_with_socket=False),
                frame(.1, insertion_depth_mm=12, orientation_error_deg=1, aligned_with_socket=True)]
        t, r, v = fixture("usb-insert")
        result = augment_task_visualization(t, r, v, frame_rows=rows)
        series = {row["key"]: row for row in v["task_series"]}
        self.assertEqual(series["insertion_depth_mm"]["unit"], "mm")
        self.assertEqual(series["insertion_depth_mm"]["values"], [0, 12])
        self.assertEqual(series["orientation_error_deg"]["unit"], "deg")
        self.assertIn("未对准记 0", series["insertion_depth_mm"]["label"])
        self.assertIn("并非原始几何深度为零", result["presentation"]["task_note"])
        self.assertNotIn("insertion_depth_m", series)

    def test_explicit_two_pad_grip_vector_not_five_finger_sum(self):
        t, r, v = fixture("usb-unplug", rows=[control(0, grip_force_n=[0, 2]),
            control(1, grip_force_n=[9]), control(2, grip_force_n=[1, None])])
        augment_task_visualization(t, r, v)
        series = {row["key"]: row for row in v["task_series"]}
        self.assertEqual(series["grip_thumb_force_n"]["values"], [0, None, 1])
        self.assertEqual(series["grip_index_force_n"]["values"], [2, None, None])
        self.assertIn("拇指", series["grip_thumb_force_n"]["label"])
        self.assertNotIn("board_normal_force_n", series)

    def test_usb_unplug_current_hold_time_keeps_resets_missing_values_and_native_clock(self):
        t, r, v = fixture("usb-unplug", rows=[control(0, extraction_hold_duration_s=0),
            control(.05, extraction_hold_duration_s=1.95), control(.1, extraction_hold_duration_s=0),
            control(.15, extraction_hold_duration_s=None), control(.2, extraction_hold_duration_s=-1)])
        result = augment_task_visualization(t, r, v)
        series = next(row for row in v["task_series"] if row["key"] == "extraction_hold_duration_s")
        self.assertEqual(series["values"], [0, 1.95, 0, None, None])
        self.assertEqual(series["time_s"], [0, .05, .1, .15, .2])
        self.assertEqual(series["unit"], "s")
        self.assertIn("extraction_hold_duration_s", result["presentation"]["primary_task_keys"])
        self.assertEqual(r["outcome"]["status"], "fail")

    def test_terminal_only_forces_do_not_generate_bulb_curves(self):
        t, r, v = fixture("bulb-unscrew", rows=[control(0, turns_removed=.2)],
                          summary={"task": "bulb-unscrew", "evaluation": {"state": {"socket_contact_load_n": 8}}})
        augment_task_visualization(t, r, v)
        self.assertEqual([row["key"] for row in v["task_series"]], ["angle_removed_deg"])

    def test_poker_distance_flag_does_not_turn_into_task_edge_or_success(self):
        t, r, v = fixture("poker-draw", summary={"task": "poker-draw", "evaluation": {"success": False, "edge_reached": False}},
                          rows=[control(0, card_displacement_now_mm=82, card_displacement_peak_mm=82, edge_distance_reached=True)])
        result = augment_task_visualization(t, r, v)
        self.assertEqual(r["outcome"]["status"], "fail")
        self.assertFalse(next(row for row in result["task_summary"] if row["key"] == "edge_reached")["value"])
        self.assertNotIn("edge_distance_reached", [row["key"] for row in v["task_series"]])
        self.assertIn("不等于到达桌边", result["presentation"]["task_note"])

    def test_pick_place_empty_response_is_missing_not_zero(self):
        t, r, v = fixture("pick-place", rows=[control(0), control(.1)])
        result = augment_task_visualization(t, r, v)
        self.assertEqual(v["task_series"], [])
        self.assertEqual(result["task_series_status"], "not_recorded")
        self.assertIn("没有可同步", result["limitations"][-1])

    def test_only_whitelisted_fields_are_drawn(self):
        t, r, v = fixture("poker-draw", rows=[control(0, cleaned_fraction=.7, board_normal_force_n=3,
            arbitrary_score=100, success=True, card_displacement_now_mm=0)])
        augment_task_visualization(t, r, v)
        self.assertEqual([row["key"] for row in v["task_series"]], ["card_displacement_now_mm"])

    def test_bad_gap_and_bad_container_are_not_interpreted_as_no_gap(self):
        t, r, v = fixture(rows=[control(0, cleaned_fraction=0)])
        t.control_rows[0]["gap_after"] = "false"
        result = augment_task_visualization(t, r, v)
        self.assertEqual(v["task_series"], [])
        self.assertIn("invalid gap", result["warnings"][0])
        t.control_rows = {"sim_time": 0}
        with self.assertRaises(ValueError): augment_task_visualization(t, r, v)

    def test_invalid_gap_factor_is_rejected(self):
        for value in (0, .5, float("nan"), True, "1.5"):
            t, r, v = fixture()
            r["protocol"]["max_gap_factor"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): augment_task_visualization(t, r, v)


class FinalSummaryTests(unittest.TestCase):
    def test_bulb_final_turn_fields_are_reported_as_degrees(self):
        cases = (
            ("bulb-screw", "clockwise_turns", "clockwise_angle_deg", .25, 90.0),
            ("bulb-unscrew", "turns_removed", "angle_removed_deg", .125, 45.0),
            ("bulb-screw-inplace", "rotation_from_initial_turns", "rotation_from_initial_deg", .1, 36.0),
        )
        for task, source_key, display_key, raw, expected in cases:
            with self.subTest(task=task):
                summary = {"task": task, "evaluation": {"state": {source_key: raw}}}
                before = deepcopy(summary)
                t, r, _ = fixture(task, summary=summary)
                row = next(item for item in build_task_summary(t, r) if item["key"] == display_key)
                self.assertEqual((row["value"], row["unit"], row["status"]),
                                 (expected, "deg", "recorded"))
                self.assertEqual(row["conversion"]["factor"], 360.0)
                self.assertEqual(summary, before)

    def test_usb_unplug_new_current_hold_finals_are_separate_from_legacy_flags(self):
        t, r, _ = fixture("usb-unplug", summary={"task": "usb-unplug", "evaluation": {"state": {
            "extraction_hold_duration_s": 2.0, "extraction_hold_now": True,
            "extraction_hold_verified": True, "placed_on_table": False, "released": False}}})
        rows = {row["key"]: row for row in build_task_summary(t, r)}
        self.assertEqual(rows["extraction_hold_duration_s"]["value"], 2.0)
        self.assertTrue(rows["extraction_hold_verified"]["value"])
        self.assertFalse(rows["placed_on_table"]["value"])
        self.assertFalse(rows["released"]["value"])
        self.assertIsNone(rows["hand_retreated"]["value"])
        # Legacy completion flags cannot fill the new continuous-window fields.
        t.summary["evaluation"]["state"] = {"fully_extracted": True, "placed_on_table": True,
                                              "released": True, "hand_retreated": True}
        old = {row["key"]: row for row in build_task_summary(t, r)}
        self.assertTrue(old["fully_extracted"]["value"])
        for key in ("extraction_hold_duration_s", "extraction_hold_now", "extraction_hold_verified"):
            self.assertEqual(old[key]["status"], "missing")
            self.assertIsNone(old[key]["value"])

    def test_usb_nested_insertion_is_read_not_missing_state(self):
        t, r, _ = fixture("usb-insert", summary={"task": "usb-insert", "evaluation": {
            "insertion": {"insertion_depth_m": .031, "seated": False, "bottom_out_confirmed": False},
            "maximum_insertion_depth_m": 0}})
        summary = {row["key"]: row for row in build_task_summary(t, r)}
        self.assertEqual(summary["insertion_depth_m"]["value"], .031)
        self.assertEqual(summary["maximum_insertion_depth_m"]["value"], 0)
        self.assertEqual(summary["seated"]["value"], False)
        self.assertEqual(summary["seated"]["status"], "recorded")
        self.assertEqual(summary["seated"]["source"], "summary.json:evaluation.insertion.seated")

    def test_null_final_not_replaced_by_last_recorded_progress(self):
        t, r, _ = fixture(rows=[control(0, cleaned_fraction=1)],
                          summary={"task": "whiteboard-wipe", "evaluation": {"state": {"cleaned_fraction": None}}})
        row = build_task_summary(t, r)[0]
        self.assertIsNone(row["value"])
        self.assertEqual(row["status"], "missing")

    def test_conflicting_explicit_final_sources_are_unknown(self):
        t, r, _ = fixture("usb-insert", summary={"task": "usb-insert", "evaluation": {
            "insertion": {"seated": False}, "state": {"seated": True}}})
        row = next(row for row in build_task_summary(t, r) if row["key"] == "seated")
        self.assertIsNone(row["value"])
        self.assertEqual(row["status"], "conflicting")

    def test_fail_unknown_and_diagnostic_labels_are_not_recertified(self):
        for outcome in ("fail", "unknown", "pass"):
            t, r, v = fixture("pick-place", summary={"task": "pick-place", "diagnostic_only": True,
                "formal_metrics_valid": False, "evaluation": {"success": True, "placed_in_box": True}})
            r["outcome"].update(status=outcome, diagnostic_only=True, formal_metrics_valid=False)
            before = deepcopy(r)
            result = augment_task_visualization(t, r, v)
            self.assertEqual(r, before)
            self.assertEqual(result["review_status"], "not_verified")

    def test_wrong_task_rejected_instead_of_reusing_whiteboard_labels(self):
        t, r, v = fixture("usb-unplug")
        r["trial"]["task"] = "whiteboard-wipe"
        with self.assertRaises(ValueError): augment_task_visualization(t, r, v)

    def test_invalid_final_values_not_serialized_as_nan_or_truthy_flags(self):
        t, r, _ = fixture("whiteboard-wipe", summary={"task": "whiteboard-wipe", "evaluation": {"state": {
            "cleaned_fraction": float("nan"), "maximum_lift_m": -1, "released_on_table": "true"}}})
        rows = build_task_summary(t, r)
        self.assertTrue(all(row["value"] is None for row in rows))
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
