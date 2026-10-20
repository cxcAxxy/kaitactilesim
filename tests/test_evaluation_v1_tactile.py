"""Synthetic evidence tests for reference-free tactile statistics."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "scripts/analysis/evaluation_v1/tactile.py"
SPEC = importlib.util.spec_from_file_location("evaluation_v1_tactile_under_test", MODULE)
TACTILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TACTILE)
analyze = TACTILE.analyze_tactile


def channel(report, name="channel_0", scope="overall"):
    return report["scopes"][scope]["channels"][name]


class TactileAnalysisTests(unittest.TestCase):
    def test_time_translation_preserves_metrics(self):
        times = np.arange(101) / 100
        force = np.where(times[:, None] < 0.3, 0.0, 2.0)
        base = analyze(times, force)
        translated = analyze(times + 1234.0, force)
        for value in (base, translated):
            value["timebase"].pop("source_start_time_s")
        self.assertAlmostEqual(channel(base)["intensity"]["all_observed"]["mean_n"], channel(translated)["intensity"]["all_observed"]["mean_n"], places=10)
        self.assertAlmostEqual(channel(base)["change"]["mean_absolute_change_n"], channel(translated)["change"]["mean_absolute_change_n"], places=10)
        self.assertEqual(channel(base)["change"]["pair_count"], channel(translated)["change"]["pair_count"])
        self.assertEqual(channel(base)["continuity"]["contact_episode_count"], channel(translated)["continuity"]["contact_episode_count"])

    def test_wait_preserves_active_scope_but_costs_elapsed_time(self):
        times = np.arange(101) / 100
        base = analyze(times, np.full((101, 1), 2.0), scope_masks={"active": np.ones(101, dtype=bool)})
        delayed_times = np.arange(201) / 100
        active = delayed_times >= 1
        delayed = analyze(delayed_times, np.where(active[:, None], 2.0, 0.0), scope_masks={"active": active})
        self.assertAlmostEqual(delayed["timebase"]["duration_s"] - base["timebase"]["duration_s"], 1)
        for report in (base, delayed):
            self.assertAlmostEqual(channel(report, scope="active")["intensity"]["all_observed"]["mean_n"], 2)
            self.assertAlmostEqual(channel(report, scope="active")["change"]["mean_absolute_change_n"], 0, places=12)
        self.assertAlmostEqual(channel(delayed)["intensity"]["all_observed"]["mean_n"], 1)

    def test_short_and_terminal_spikes_survive_native_peak(self):
        times = np.arange(101) / 100
        force = np.zeros((101, 1)); force[35, 0] = 20; force[-1, 0] = 50
        item = channel(analyze(times, force))
        self.assertEqual(item["intensity"]["sampled_peak_n"], 50)
        self.assertAlmostEqual(item["intensity"]["all_observed"]["mean_n"], 0.2)
        self.assertEqual(item["continuity"]["candidate_episode_count"], 1)
        self.assertEqual(item["continuity"]["short_contact_episode_count"], 1)
        self.assertEqual(item["continuity"]["contact_episode_count"], 0)
        self.assertLess(item["change"]["max_absolute_change_n"], 20)

    def test_gap_is_unknown_and_no_change_jumps_over_it(self):
        times = np.array([0, .1, .2, .5, .6, .7])
        force = np.array([[1], [1], [1], [10], [10], [10]], dtype=float)
        report = analyze(times, force)
        item = channel(report)
        self.assertEqual(report["timebase"]["gap_interval_count"], 1)
        self.assertAlmostEqual(report["timebase"]["gap_duration_s"], .3)
        self.assertAlmostEqual(item["intensity"]["observed_duration_s"], .4)
        self.assertAlmostEqual(item["intensity"]["coverage_fraction"], 4/7)
        self.assertAlmostEqual(item["change"]["max_absolute_change_n"], 0, places=12)
        self.assertEqual(item["change"]["pair_count"], 2)
        self.assertAlmostEqual(item["continuity"]["unknown_duration_s"], .3)
        self.assertEqual(item["continuity"]["contact_episode_count"], 2)
        self.assertEqual(item["continuity"]["observed_internal_break_count"], 0)

    def test_missing_and_negative_values_are_not_zero_force(self):
        times = np.arange(6) / 10
        force = np.array([[2], [np.nan], [-1], [2], [2], [np.inf]])
        report = analyze(times, force)
        item = channel(report)
        self.assertEqual(item["status"], "partial")
        self.assertAlmostEqual(item["intensity"]["all_observed"]["mean_n"], 2)
        self.assertAlmostEqual(item["intensity"]["coverage_fraction"], .6)
        self.assertAlmostEqual(item["continuity"]["unknown_duration_s"], .2)
        json.dumps(report, allow_nan=False)

    def test_unreached_and_terminal_only_scope(self):
        report = analyze([0, .1, .2], np.ones((3, 1)), scope_masks={
            "unreached": [False, False, False], "terminal": [False, False, True]})
        self.assertEqual(report["scopes"]["unreached"]["status"], "not_reached")
        self.assertIsNone(report["scopes"]["unreached"]["channels"])
        self.assertIsNone(report["scopes"]["unreached"]["duration_s"])
        terminal = channel(report, scope="terminal")
        self.assertIsNone(terminal["intensity"]["all_observed"]["mean_n"])
        self.assertEqual(terminal["intensity"]["sampled_peak_n"], 1)
        self.assertEqual(terminal["status"], "insufficient_duration")

    def test_constant_signal_agrees_at_100_and_30_hz(self):
        reports = [analyze(np.arange(rate + 1) / rate, np.full((rate + 1, 3), 2.0)) for rate in (100, 30)]
        for report in reports:
            item = channel(report)
            self.assertAlmostEqual(item["intensity"]["all_observed"]["mean_n"], 2)
            self.assertAlmostEqual(item["intensity"]["all_observed"]["p95_n"], 2)
            self.assertAlmostEqual(item["intensity"]["coverage_fraction"], 1)
            self.assertAlmostEqual(item["change"]["max_absolute_change_n"], 0, places=12)
            self.assertEqual(item["change"]["complete_bin_count"], 10)
            self.assertEqual(item["change"]["pair_count"], 9)

    def test_spatial_redistribution_separate_from_force_amplitude(self):
        times = np.arange(41) / 100
        grid = np.zeros((41, 1, 2)); grid[:20, 0, 0] = 2; grid[20:, 0, 1] = 2
        force = grid.sum(axis=-1)
        geometry = np.array([[[-.001, 0, 0], [.001, 0, 0]]])
        item = channel(analyze(times, force, normal_taxel_force_n=grid, taxel_positions_local_m=geometry))
        self.assertAlmostEqual(item["change"]["max_absolute_change_n"], 0, places=12)
        self.assertAlmostEqual(item["spatial"]["distribution_change"]["p95_total_variation"], 1)
        self.assertAlmostEqual(item["spatial"]["center_of_pressure"]["p95_displacement_mm"], 2)
        amplitude_grid = grid.copy(); amplitude_grid[20:, 0, :] *= 3
        scaled = channel(analyze(times, amplitude_grid.sum(axis=-1), normal_taxel_force_n=amplitude_grid, taxel_positions_local_m=geometry))
        self.assertAlmostEqual(scaled["spatial"]["distribution_change"]["mean_total_variation"], item["spatial"]["distribution_change"]["mean_total_variation"])
        self.assertAlmostEqual(scaled["change"]["max_absolute_change_n"], 4)

    def test_missing_geometry_never_manufactures_mm(self):
        times = np.arange(31) / 100
        grid = np.ones((31, 1, 2, 2))
        item = channel(analyze(times, grid.sum(axis=(2, 3)), normal_taxel_force_n=grid))
        self.assertEqual(item["spatial"]["center_of_pressure"]["status"], "missing")
        self.assertIsNone(item["spatial"]["center_of_pressure"]["mean_local_mm"])
        self.assertEqual(item["spatial"]["distribution_change"]["status"], "available")

    def test_arbitrary_channels_and_normal_load_shares(self):
        values = np.arange(1, 8, dtype=float)
        report = analyze(np.arange(31) / 100, np.tile(values, (31, 1)), channel_names=[f"pad_{i}" for i in range(7)])
        shares = report["scopes"]["overall"]["load_shares"]["channels"]
        self.assertEqual(len(shares), 7)
        self.assertAlmostEqual(sum(item["mean_fraction"] for item in shares.values()), 1)
        self.assertAlmostEqual(shares["pad_6"]["mean_fraction"], 7/28)

    def test_hysteresis_boundaries_and_uninitialized_band(self):
        # Exactly on does not enter; exactly off exits; a fresh in-band
        # segment is unknown rather than an invented absence.
        force = np.array([[.05], [.06], [.04], [.03], [.05], [.06], [.06]])
        item = channel(analyze(np.arange(7) / 10, force))
        events = item["continuity"]
        self.assertEqual(events["candidate_episode_count"], 2)
        self.assertAlmostEqual(events["observed_contact_duration_s"], .3)
        self.assertAlmostEqual(events["unknown_duration_s"], .1)
        self.assertAlmostEqual(events["observed_internal_break_duration_s"], .2)
        self.assertAlmostEqual(item["intensity"]["contact"]["duration_s"], .2)

    def test_tangent_vectors_cancel_before_norm(self):
        times = np.arange(31) / 100
        tangent = np.zeros((31, 1, 2, 2)); tangent[:, 0, 0, 0] = 3; tangent[:, 0, 1, 0] = -3
        item = channel(analyze(times, np.ones((31, 1)), tangent_taxel_force_n=tangent))
        self.assertEqual(item["tangent"]["net_magnitude"]["mean_n"], 0)
        self.assertEqual(item["tangent"]["sampled_peak_n"], 0)
        self.assertIn("non-cancelling", item["tangent"]["semantics"])

    def test_spatial_force_mismatch_is_excluded(self):
        times = np.arange(31) / 100
        grid = np.ones((31, 1, 2))
        item = channel(analyze(times, np.ones((31, 1)), normal_taxel_force_n=grid))
        self.assertEqual(item["spatial"]["status"], "no_valid_contact")
        self.assertEqual(item["spatial"]["inconsistent_force_sum_sample_count"], 31)

    def test_disjoint_scope_does_not_create_cross_stage_change(self):
        times = np.arange(101) / 100
        mask = (times < .3) | (times >= .7)
        force = np.where(times[:, None] < .5, 1.0, 5.0)
        item = channel(analyze(times, force, scope_masks={"phase": mask}), scope="phase")
        self.assertAlmostEqual(item["change"]["max_absolute_change_n"], 0, places=12)

    def test_no_force_data_yields_null_metrics_and_json(self):
        report = analyze([0, .1, .2], np.full((3, 1), np.nan))
        item = channel(report)
        self.assertEqual(item["status"], "missing")
        self.assertIsNone(item["intensity"]["all_observed"]["mean_n"])
        self.assertIsNone(item["intensity"]["sampled_peak_n"])
        self.assertIsNone(item["continuity"]["contact_episode_count"])
        self.assertEqual(report["scopes"]["overall"]["load_shares"]["status"], "missing")
        json.dumps(report, allow_nan=False)

    def test_one_sample_only_has_a_peak(self):
        report = analyze([100], [[7]])
        self.assertEqual(channel(report)["intensity"]["sampled_peak_n"], 7)
        self.assertIsNone(channel(report)["intensity"]["all_observed"]["mean_n"])
        self.assertEqual(channel(report)["change"]["pair_count"], 0)
        self.assertEqual(report["scopes"]["overall"]["load_shares"]["status"], "insufficient_duration")

    def test_nonadjacent_lag_cannot_skip_a_missing_bin(self):
        times = np.arange(101) / 100
        force = np.ones((101, 1)); force[20:40] = np.nan; force[40:] = 10
        item = channel(analyze(times, force, config={"change_lag_s": .3}))
        self.assertAlmostEqual(item["change"]["max_absolute_change_n"], 0, places=12)
        self.assertEqual(item["change"]["pair_count"], 3)

    def test_observed_zero_contact_differs_from_missing_shares(self):
        report = analyze([0, .1, .2], np.zeros((3, 2)))
        shares = report["scopes"]["overall"]["load_shares"]
        self.assertEqual(shares["status"], "no_valid_contact")
        self.assertEqual(shares["coverage_fraction"], 1)
        self.assertIsNone(shares["channels"]["channel_0"]["mean_fraction"])

    def test_validation(self):
        invalid = [
            lambda: analyze([0, 0], [[1], [1]]),
            lambda: analyze([0, 1], [1, 1]),
            lambda: analyze([0, 1], [[1], [1]], scope_masks={"x": [1, 0]}),
            lambda: analyze([0, 1], [[1], [1]], config={"change_lag_s": .15}),
            lambda: analyze([0, 1], [[1], [1]], config={"contact_off_n": .1}),
            lambda: analyze([0, 1], [[1], [1]], channel_names=["a", "a"]),
            lambda: analyze([0, 1], [[1], [1]], config={"unknown": 1}),
        ]
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
