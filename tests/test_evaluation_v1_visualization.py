"""Display payloads must preserve native evidence without inventing alignment."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.adapters import TrialData
from evaluation_v1.visualization import build_visualization


def trial(times=None, force=None, grid=None, *, task="whiteboard-wipe", aligned=True, rows=None):
    return TrialData(root=Path("/not/read/by/visualization"), summary={}, task=task,
                     time_s=np.asarray(times, dtype=float) if times is not None else None,
                     normal_force_n=np.asarray(force, dtype=float) if force is not None else None,
                     normal_taxel_force_n=np.asarray(grid, dtype=float) if grid is not None else None,
                     channel_names=[f"pad_{i}" for i in range(np.shape(force)[1])] if force is not None else [],
                     task_tactile_clock_aligned=aligned, control_rows=rows or [])


def report():
    return {"protocol": {"max_gap_factor": 1.5}, "warnings": [],
            "outcome": {"status": "fail"}, "safety": {"status": "unknown"},
            "efficiency": {"simulation_duration_s": 90}, "task_specific": {"cleaned_fraction": .5},
            "events": [], "scope_intervals": []}


class VisualizationTests(unittest.TestCase):
    def test_native_curve_and_spatial_peak_frames_are_preserved(self):
        times = np.arange(1001) / 100
        grid = np.ones((1001, 5, 2, 3))
        peaks = [13, 177, 501, 843, 1000]
        for channel, sample in enumerate(peaks):
            grid[sample, channel, 0, 0] = 30 + channel
        data = trial(times, grid.sum(axis=(2, 3)), grid)
        result = build_visualization(data, report(), max_spatial_frames=17)
        self.assertEqual(result["schema_version"], "evaluation-visualization-v1")
        self.assertEqual(len(result["tactile_series"]["time_s"]), 1001)
        np.testing.assert_equal(result["tactile_series"]["normal_force_n"], data.normal_force_n)
        selected = result["spatial_frames"]["source_sample_index"]
        self.assertEqual(len(selected), 17)
        self.assertTrue(set(peaks) <= set(selected))
        np.testing.assert_equal(result["spatial_frames"]["normal_taxel_force_n"], grid[selected])
        np.testing.assert_equal(result["spatial_frames"]["time_s"], times[selected])
        self.assertEqual(result["spatial_frames"]["scale_max_n"], 34)

    def test_nonfinite_negative_and_missing_values_are_json_null(self):
        data = trial([0, .1, .2, .3], [[1, 2], [np.nan, 3], [-1, np.inf], [2, 4]],
                     np.array([1., np.nan, -2., np.inf, 0., 1., 2., 3.]).reshape(4, 2, 1, 1))
        result = build_visualization(data, report())
        values = result["tactile_series"]["normal_force_n"]
        self.assertIsNone(values[1][0]); self.assertIsNone(values[2][0]); self.assertIsNone(values[2][1])
        self.assertEqual(result["tactile_series"]["status"], "partial")
        self.assertIsNone(result["spatial_frames"]["normal_taxel_force_n"][0][1][0][0])
        json.dumps(result, allow_nan=False)

    def test_long_gap_breaks_after_the_correct_native_sample(self):
        data = trial([10, 10.1, 10.2, 11, 11.1], np.ones((5, 1)))
        result = build_visualization(data, report())
        self.assertEqual(result["time_origin_s"], 10)
        self.assertEqual(result["tactile_series"]["gap_after"], [False, False, True, False, False])
        np.testing.assert_allclose(result["tactile_series"]["time_s"], [0, .1, .2, 1, 1.1])
        self.assertAlmostEqual(result["tactile_series"]["nominal_interval_s"], .1)

    def test_budget_is_hard_and_impossible_peak_budget_is_explicit(self):
        grid = np.ones((500, 3, 1, 1))
        for channel, index in enumerate([11, 41, 63]):
            grid[index, channel, 0, 0] = 99
        data = trial(np.arange(500) / 100, grid.sum(axis=(2, 3)), grid)
        capped = build_visualization(data, report(), 999)
        self.assertEqual(capped["spatial_frames"]["selected_frame_count"], 180)
        small = build_visualization(data, report(), 2)
        self.assertEqual(small["spatial_frames"]["status"], "insufficient_frame_budget")
        self.assertEqual(small["spatial_frames"]["source_sample_index"], [])
        self.assertEqual(len(small["tactile_series"]["time_s"]), 500)
        self.assertEqual(small["task_summary"]["outcome"]["status"], "fail")

    def test_color_scale_uses_unsampled_native_taxel_peak(self):
        # The strongest individual taxel is not at the total-force peak.
        # With only one frame allowed, total-force peak retention wins, while
        # the colour scale still has to include the original taxel maximum.
        grid = np.array([[[[9., 0.]]], [[[5., 5.]]], [[[0., 0.]]]])
        data = trial([0, .1, .2], grid.sum(axis=(2, 3)), grid)
        result = build_visualization(data, report(), 1)
        self.assertEqual(result["spatial_frames"]["source_sample_index"], [1])
        self.assertEqual(result["spatial_frames"]["scale_max_n"], 9)
        self.assertEqual(np.max(result["spatial_frames"]["normal_taxel_force_n"]), 5)

    def test_heatmap_without_geometry_does_not_invent_coordinates(self):
        grid = np.ones((3, 2, 2, 3))
        data = trial([0, .1, .2], grid.sum(axis=(2, 3)), grid)
        result = build_visualization(data, report())
        self.assertEqual(result["channels"][0]["grid_shape"], [2, 3])
        self.assertEqual(result["spatial_frames"]["geometry_status"], "missing")
        self.assertIsNone(result["spatial_frames"]["taxel_positions_local_m"])
        self.assertTrue(any("no millimetre coordinates" in warning for warning in result["warnings"]))
        data.taxel_positions_local_m = np.arange(36).reshape(2, 2, 3, 3) / 1000
        known = build_visualization(data, report())
        self.assertEqual(known["spatial_frames"]["geometry_status"], "available")
        np.testing.assert_equal(known["spatial_frames"]["taxel_positions_local_m"], data.taxel_positions_local_m)

    def test_unknown_clock_has_no_task_or_event_overlay(self):
        data = trial([10, 11], [[1], [2]], aligned=False,
                     rows=[{"sim_time": 100, "task": {"cleaned_fraction": 0}},
                           {"sim_time": 101, "task": {"cleaned_fraction": 1}}])
        rep = report(); rep["events"] = [{"time_s": 100, "name": "not synchronized"}]
        rep["scope_intervals"] = [{"start_s": 100, "end_s": 101, "name": "not synchronized"}]
        result = build_visualization(data, rep)
        self.assertFalse(result["clock_aligned"])
        for name in ("task_series", "events", "scope_intervals"):
            self.assertEqual(result[name], [])
        self.assertEqual(result["task_summary"]["task_specific"], rep["task_specific"])

    def test_same_clock_uses_one_origin_and_keeps_ground_truth_flag(self):
        rows = [{"sim_time": 10.1, "task": {"cleaned_fraction": .1, "board_normal_force_n": 1}},
                {"sim_time": 10.2, "task": {"cleaned_fraction": .3}},
                {"sim_time": 10.3, "task": {"cleaned_fraction": .4, "board_normal_force_n": 0}}]
        data = trial([10, 10.1, 10.2, 10.3], np.ones((4, 1)), rows=rows)
        rep = report(); rep["events"] = [{"time_s": 10.2, "type": "declared_stage", "ground_truth": False}]
        rep["scope_intervals"] = [{"name": "board_contact", "start_s": 10.1, "end_s": 10.3}]
        before = deepcopy(rep)
        result = build_visualization(data, rep)
        values = {row["key"]: row for row in result["task_series"]}
        np.testing.assert_allclose(values["cleaned_fraction"]["time_s"], [.1, .2, .3])
        self.assertEqual(values["board_normal_force_n"]["values"], [1., None, 0.])
        self.assertAlmostEqual(result["events"][0]["time_s"], .2)
        self.assertFalse(result["events"][0]["ground_truth"])
        self.assertAlmostEqual(result["scope_intervals"][0]["start_s"], .1)
        self.assertEqual(rep, before)

    def test_no_tactile_keeps_summary_without_invented_series(self):
        data = trial(rows=[{"sim_time": 1, "task": {"cleaned_fraction": .8}}])
        result = build_visualization(data, report())
        self.assertEqual(result["task_summary"]["outcome"]["status"], "fail")
        self.assertIsNone(result["time_origin_s"])
        self.assertEqual(result["tactile_series"]["time_s"], [])
        self.assertEqual(result["spatial_frames"]["time_s"], [])
        self.assertEqual(result["task_series"], [])

    def test_task_whitelist_preserves_signed_insertion_and_rejects_other_values(self):
        rows = [{"sim_time": 0, "task": {"insertion_depth_m": -.04, "orientation_error_rad": 0, "arbitrary_score": 123}},
                {"sim_time": .1, "task": {"insertion_depth_m": .005, "orientation_error_rad": float("inf")}}]
        data = trial([0, .1], [[1], [1]], task="install-ram", rows=rows)
        result = build_visualization(data, report())
        curves = {row["key"]: row for row in result["task_series"]}
        self.assertEqual(curves["insertion_depth_m"]["values"], [-.04, .005])
        self.assertEqual(curves["orientation_error_rad"]["values"], [0., None])
        self.assertNotIn("arbitrary_score", curves)

    def test_task_gaps_do_not_connect_and_fraction_not_filled_from_summary(self):
        rows = [{"sim_time": t, "task": {"cleaned_fraction": value}} for t, value in
                [(0, 0), (.1, .2), (.2, None), (1, .4), (1.1, 1.5)]]
        data = trial([0, .1, .2, 1, 1.1], np.ones((5, 1)), rows=rows)
        result = build_visualization(data, report())
        curve = result["task_series"][0]
        self.assertEqual(curve["values"], [0., .2, None, .4, None])
        self.assertEqual(curve["gap_after"], [False, False, True, False, False])

    def test_invalid_task_clock_is_not_silently_sorted(self):
        for timestamps in ([0, 0], [1, 0], [0, None]):
            with self.subTest(timestamps=timestamps):
                rows = [{"sim_time": t, "task": {"cleaned_fraction": .5}} for t in timestamps]
                result = build_visualization(trial([0, .1], [[1], [1]], rows=rows), report())
                self.assertEqual(result["task_series"], [])
                self.assertEqual(len(result["tactile_series"]["time_s"]), 2)

    def test_input_arrays_are_not_modified(self):
        force = np.array([[np.nan, 2], [-1, 0], [3, 4.]])
        grid = force.reshape(3, 2, 1, 1).copy()
        data = trial([4, 4.1, 4.2], force, grid)
        before_force, before_grid = data.normal_force_n.copy(), data.normal_taxel_force_n.copy()
        build_visualization(data, report())
        np.testing.assert_equal(data.normal_force_n, before_force)
        np.testing.assert_equal(data.normal_taxel_force_n, before_grid)
        self.assertEqual(data.warnings, [])

    def test_single_frame_and_explicit_disable(self):
        data = trial([7], [[3]], [[[[3]]]])
        result = build_visualization(data, report())
        self.assertEqual(result["tactile_series"]["gap_after"], [False])
        self.assertEqual(result["spatial_frames"]["source_sample_index"], [0])
        self.assertEqual(result["spatial_frames"]["time_s"], [0.])
        disabled = build_visualization(data, report(), 0)
        self.assertEqual(disabled["spatial_frames"]["status"], "disabled")
        self.assertEqual(disabled["spatial_frames"]["source_sample_index"], [])

    def test_bad_optional_grid_does_not_remove_force_or_outcome(self):
        data = trial([0, .1], [[1], [2]], np.ones((7, 2, 3)))
        result = build_visualization(data, report())
        self.assertEqual(result["spatial_frames"]["status"], "missing")
        self.assertEqual(result["channels"][0]["grid_shape"], None)
        self.assertEqual(result["tactile_series"]["normal_force_n"], [[1.], [2.]])
        self.assertEqual(result["task_summary"]["outcome"]["status"], "fail")

    def test_budget_validation(self):
        for budget in (-1, True, 1.5, "3"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                build_visualization(trial(), report(), budget)


class ContactGlyphTests(unittest.TestCase):
    def test_native_samples_peaks_and_centers_do_not_depend_on_heatmap_budget(self):
        grid = np.ones((31, 2, 2, 3))
        grid[17, 0, 1, 2] = 101
        data = trial(10 + np.arange(31) / 100, grid.sum(axis=(2, 3)), grid)
        result = build_visualization(data, report(), max_spatial_frames=0)
        glyph = result["contact_glyph"]
        self.assertEqual(len(glyph["time_s"]), 31)
        self.assertEqual(glyph["time_s"], result["tactile_series"]["time_s"])
        self.assertEqual(glyph["normal_force_n"], result["tactile_series"]["normal_force_n"])
        self.assertEqual(glyph["force_scale_max_n"], 106)
        self.assertEqual(glyph["normal_force_n"][17][0], 106)
        self.assertEqual(glyph["channels"][0]["coordinate_status"], "index")
        self.assertEqual(glyph["channels"][0]["unit"], "index")
        self.assertEqual(glyph["channels"][0]["taxel_xy"], [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        np.testing.assert_allclose(glyph["center_xy"][17][0], [206 / 106, 103 / 106])
        self.assertEqual(result["spatial_frames"]["selected_frame_count"], 0)

    def test_physical_xy_requires_recorded_nondegenerate_geometry_per_channel(self):
        grid = np.ones((2, 3, 2, 2))
        data = trial([0, .1], grid.sum(axis=(2, 3)), grid)
        data.taxel_positions_local_m = np.array([
            [[[0, 0, 0], [.002, 0, 0]], [[0, .004, 0], [.002, .004, 0]]],
            [[[0, 0, 0], [0, .002, 0]], [[0, 0, .004], [0, .002, .004]]],
            [[[0, 0, 0], [.002, 0, 0]], [[0, np.nan, 0], [.002, .004, 0]]],
        ])
        glyph = build_visualization(data, report())["contact_glyph"]
        self.assertEqual([row["coordinate_status"] for row in glyph["channels"]], ["physical", "index", "index"])
        self.assertEqual(glyph["channels"][0]["unit"], "mm")
        self.assertEqual(glyph["channels"][0]["taxel_xy"], [[0, 0], [2, 0], [0, 4], [2, 4]])
        self.assertEqual(glyph["center_xy"][0], [[1, 2], [.5, .5], [.5, .5]])

    def test_zero_force_and_partial_taxels_have_no_centroid(self):
        grid = np.array([[[[1, 3]]], [[[1, 3]]], [[[1, np.nan]]], [[[0, 0]]], [[[1, -1]]]])
        data = trial(np.arange(5) / 10, [[4], [0], [4], [2], [1]], grid)
        glyph = build_visualization(data, report())["contact_glyph"]
        self.assertEqual(glyph["center_xy"], [[[.75, 0]], [None], [None], [None], [None]])
        self.assertEqual(glyph["center_status"], "partial")
        self.assertEqual(glyph["normal_force_n"][3], [2])  # Recorded Fn, not taxel sum.

    def test_signed_tangent_resultant_cancellation_precedes_differencing(self):
        grid = np.ones((3, 1, 1, 2))
        data = trial([0, .1, .2], [[2], [3], [1]], grid)
        data.tangent_taxel_force_n = np.array([
            [[[[1, 0], [-1, 0]]]],
            [[[[3, 0], [-3, 0]]]],
            [[[[2, -2], [1, 6]]]],
        ], dtype=float)
        glyph = build_visualization(data, report())["contact_glyph"]
        self.assertEqual(glyph["normal_change_n"], [[None], [1], [-2]])
        self.assertEqual(glyph["tangent_change_n"], [[None], [[0, 0]], [[3, 4]]])
        self.assertEqual(glyph["tangent_change_scale_max_n"], 5)
        self.assertEqual(glyph["normal_change_scale_max_n"], 2)
        self.assertEqual(glyph["tangent_status"], "available")
        self.assertEqual(glyph["tangent_coordinate_status"], "recorded_components")
        self.assertEqual(glyph["tangent_axis_labels"], ["t₁", "t₂"])

    def test_lag_matches_native_seconds_not_a_fixed_frame_count_or_nearest_sample(self):
        for times, last_delta in ((np.arange(7) / 30, 3), (np.arange(7) * .034, None)):
            with self.subTest(times=times):
                data = trial(times, np.arange(7).reshape(-1, 1))
                glyph = build_visualization(data, report())["contact_glyph"]
                self.assertEqual(glyph["window_s"], .1)
                self.assertEqual(glyph["normal_change_n"][:3], [[None]] * 3)
                self.assertEqual(glyph["normal_change_n"][-1], [last_delta])
        # Arbitrary time translation does not change native lag differences.
        a = build_visualization(trial(np.arange(11) / 100, np.arange(11).reshape(-1, 1)), report())
        b = build_visualization(trial(12345 + np.arange(11) / 100, np.arange(11).reshape(-1, 1)), report())
        self.assertEqual(a["contact_glyph"]["normal_change_n"], b["contact_glyph"]["normal_change_n"])

    def test_changes_reject_windows_crossing_a_gap_even_with_exact_endpoints(self):
        times = [0, .025, .075, .1, .125, .15, .175, .2]
        data = trial(times, np.arange(8).reshape(-1, 1), np.ones((8, 1, 1, 1)))
        data.tangent_taxel_force_n = np.ones((8, 1, 1, 1, 2))
        glyph = build_visualization(data, report())["contact_glyph"]
        self.assertEqual(glyph["gap_after"], [False, True, False, False, False, False, False, False])
        self.assertEqual(glyph["normal_change_n"][3], [None])
        self.assertEqual(glyph["normal_change_n"][4], [None])
        self.assertEqual(glyph["tangent_change_n"][3], [None])
        self.assertEqual(glyph["normal_change_n"][-1], [4])

    def test_missing_interior_sample_invalidates_only_its_corresponding_channel(self):
        data = trial(np.arange(5) / 40, np.ones((5, 2)), np.ones((5, 2, 1, 1)))
        data.normal_force_n[2, 0] = np.nan
        data.tangent_taxel_force_n = np.ones((5, 2, 1, 1, 2))
        data.tangent_taxel_force_n[2, 1, 0, 0, 0] = np.nan
        glyph = build_visualization(data, report())["contact_glyph"]
        self.assertEqual(glyph["normal_change_n"][-1], [None, 0])
        self.assertEqual(glyph["tangent_change_n"][-1], [[0, 0], None])
        self.assertIsNone(glyph["center_xy"][2][0])
        self.assertEqual(glyph["tangent_status"], "partial")
        json.dumps(glyph, allow_nan=False)

    def test_missing_or_bad_optional_evidence_does_not_infer_tangent_or_geometry(self):
        for tangent in (None, "not numeric", np.ones((2, 1, 1, 1)), np.ones((2, 1, 2, 2, 2))):
            with self.subTest(tangent=type(tangent).__name__):
                data = trial([0, .1], [[1], [2]], np.ones((2, 1, 1, 1)))
                data.tangent_taxel_force_n = tangent
                data.taxel_positions_local_m = "not numeric"
                glyph = build_visualization(data, report())["contact_glyph"]
                self.assertEqual(glyph["tangent_status"], "missing")
                self.assertEqual(glyph["tangent_change_n"], [[None], [None]])
                self.assertEqual(glyph["channels"][0]["coordinate_status"], "index")
                self.assertEqual(glyph["normal_change_n"], [[None], [1]])
        missing = build_visualization(trial(), report())["contact_glyph"]
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["time_s"], [])

    def test_glyph_generation_mutates_neither_inputs_nor_report_and_is_finite_json(self):
        data = trial([0, .05, .1], [[1, np.inf], [0, 2], [np.nan, 4]], np.ones((3, 2, 2, 2)))
        data.tangent_taxel_force_n = np.ones((3, 2, 2, 2, 2))
        data.tangent_taxel_force_n[1, 0, 0, 0, 0] = np.inf
        data.taxel_positions_local_m = np.zeros((2, 2, 2, 3))
        before = deepcopy(data)
        rep = report(); before_report = deepcopy(rep)
        result = build_visualization(data, rep)
        for key in ("time_s", "normal_force_n", "normal_taxel_force_n", "tangent_taxel_force_n", "taxel_positions_local_m"):
            np.testing.assert_equal(getattr(data, key), getattr(before, key))
        self.assertEqual(rep, before_report)
        self.assertEqual(data.warnings, [])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
