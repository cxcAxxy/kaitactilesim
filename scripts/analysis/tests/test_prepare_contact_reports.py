"""Joint-clock reports preserve a longer model without extending the expert."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_contact_reports", SCRIPTS / "prepare_contact_reports.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparationClockTests(unittest.TestCase):
    def test_clock_contains_longer_model_and_both_true_endpoints(self):
        reference = np.array([0., .1, .205])
        rollout = np.array([0., .1, .2, .3, .40000000000002])
        display = MODULE.display_clock(reference, rollout, [0., .1, .2])
        np.testing.assert_allclose(display, [0., .1, .2, .205, .3, .40000000000002])
        self.assertEqual(display[-1], rollout[-1])
        self.assertTrue(np.all(np.diff(display) > 1e-9))

    def test_clock_rejects_bad_inputs(self):
        for times in ([0.], [0., 0.], [0., float("nan")], [.1, .2]):
            with self.assertRaises(ValueError):
                MODULE.display_clock(times, [0., .1], [0.])
        with self.assertRaises(ValueError):
            MODULE.display_clock([0., .1], [0., .2], [.3])

    def test_existing_report_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "contact_intensity.json").write_text("original")
            with self.assertRaises(FileExistsError):
                MODULE.reject_existing(target)
            self.assertEqual((target / "contact_intensity.json").read_text(), "original")


@unittest.skipUnless(MODULE.DEFAULT_ANALYSIS_SOURCE.is_dir(), "legacy offline analysis source unavailable")
class PreparationSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = MODULE.analysis_modules(MODULE.DEFAULT_ANALYSIS_SOURCE)

    def trace(self, times, forces=None):
        times = np.asarray(times, dtype=float)
        normal = np.zeros((len(times), 5, 7, 5))
        normal[:, :, 3, 2] = np.asarray(forces if forces is not None else np.ones(len(times)))[:, None]
        tangent = np.zeros(normal.shape + (2,))
        tangent[..., 0] = normal
        grid = np.array([[x, y, 0.] for y in range(7) for x in range(5)]) * .001
        positions = np.repeat(grid[None], 5, axis=0)
        basis = np.repeat(np.array([[[1., 0., 0.], [0., 1., 0.]]]), 5, axis=0)
        return self.modules["whiteboard_tactile_comparison"].TactileTrace(
            times, normal, tangent, positions, basis, {})

    def test_late_model_data_and_delta_scale_are_retained(self):
        reference = self.trace([0., .1, .2])
        rollout = self.trace([0., .1, .2, .3, .4], [1., 1., 1., 1., 21.])
        summary, archive = MODULE.build_full_glyph_samples(
            rollout, reference, self.modules["whiteboard_tactile_glyphs"], snapshot_times_s=[0., .1, .2])
        self.assertEqual(summary["reference"]["end_time_s"], .2)
        self.assertEqual(summary["rollout"]["end_time_s"], .4)
        self.assertEqual(summary["display_end_time_s"], .4)
        self.assertEqual(summary["scales"]["normal_change_abs_max_n"], 20.)
        self.assertEqual(summary["scales"]["tangent_change_max_n"], 20.)
        self.assertTrue(np.isnan(archive["reference_normal_force_n"][-1]).all())
        np.testing.assert_equal(archive["rollout_normal_force_n"][-1], np.full(5, 21.))
        self.assertEqual(len(archive["reference_native_normal_force_n"]), 3)
        self.assertEqual(len(archive["rollout_native_normal_force_n"]), 5)

    def test_missing_gaps_and_changes_are_not_bridged(self):
        reference = self.trace([0., .1, .2, 1., 1.1])
        rollout = self.trace(np.arange(13) / 10)
        _, archive = MODULE.build_full_glyph_samples(
            rollout, reference, self.modules["whiteboard_tactile_glyphs"], snapshot_times_s=[0., .1, .2])
        gap = np.flatnonzero(np.isclose(archive["display_time_s"], .4))[0]
        returned = np.flatnonzero(np.isclose(archive["display_time_s"], 1.))[0]
        self.assertTrue(np.isnan(archive["reference_sample_time_s"][gap]))
        self.assertFalse(archive["reference_normal_valid"][gap].any())
        self.assertTrue(archive["reference_normal_valid"][returned].all())
        self.assertFalse(archive["reference_normal_change_valid"][returned].any())


if __name__ == "__main__":
    unittest.main()
