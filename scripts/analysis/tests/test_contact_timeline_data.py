"""Past-only timeline sampling, phase coverage and truthful comparison masks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("contact_timeline_data", SCRIPTS / "export_contact_timeline_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContactTimelineDataTests(unittest.TestCase):
    def test_left_hold_missing_boundaries_and_clock_gap(self):
        time = np.array([0, .1, .2, 1, 1.1])
        display = np.array([-.01, 0, .15, .4, 1, 1.11])
        result = MODULE.hold(time, np.arange(10).reshape(5, 2), display)
        self.assertTrue(np.isnan(result[[0, 3, 5]]).all())
        np.testing.assert_equal(result[[1, 2, 4]], [[0, 1], [2, 3], [6, 7]])

    def test_nonfinite_channel_is_not_filled(self):
        result = MODULE.hold(np.array([0, 1, 2]), np.array([[1, np.nan], [2, 3], [4, 5]]), np.array([0, .5]))
        self.assertTrue(np.isnan(result[:, 1]).all())
        np.testing.assert_equal(result[:, 0], [1, 1])

    def test_invalid_time_rejected(self):
        for time in ([0, 0], [1, 0], [0, np.nan], [0]):
            with self.assertRaises(ValueError):
                MODULE.held_indices(np.asarray(time), np.array([0]))

    def test_phase_intervals_no_terminal_extension_or_gap_bridging(self):
        result = MODULE.phase_intervals(np.array([0, .1, .2, 1, 1.1]),
                                        np.array([.1, .1, .8, .1, .1]),
                                        np.array(["pre_contact"] * 5))
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["endS"], .35)
        self.assertEqual(result[1]["endS"], 1.1)

    def test_comparison_requires_contact_on_both_sides(self):
        glyph, spatial = {}, {}
        for source in ("reference", "rollout"):
            glyph[f"{source}_sample_time_s"] = np.array([0, 1, np.nan])
            glyph[f"{source}_normal_valid"] = np.ones((3, 5), bool)
            glyph[f"{source}_contact"] = np.ones((3, 5), bool)
            glyph[f"{source}_center_local_mm"] = np.zeros((3, 5, 2))
            spatial[f"{source}_time_s"] = np.array([0, 1])
            spatial[f"{source}_normal_valid"] = np.ones((2, 5), bool)
            spatial[f"{source}_normalized_normal_distribution"] = np.zeros((2, 5, 7, 5))
            spatial[f"{source}_normalized_normal_distribution"][:, :, 0, 0] = 1
        glyph["rollout_contact"][1, 4] = False
        glyph["rollout_center_local_mm"][:, :, 0] = 3
        result = MODULE.comparison_at_display(glyph, spatial)
        self.assertEqual(result["centerDistanceMm"][0, 0], 3)
        self.assertEqual(result["distributionDifference"][0, 0], 0)
        self.assertTrue(np.isnan(result["centerDistanceMm"][1, 4]))
        self.assertTrue(np.isnan(result["distributionDifference"][2]).all())

    def test_hdf5_reference_uses_named_measured_joints_and_rotation_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.h5"
            names = [f"joint{i}" for i in range(20)]
            with h5py.File(path, "w") as file:
                head = file.create_group("cameras/head")
                head.attrs["wrist_sites_json"] = json.dumps(["hand_r_base_link_site"])
                head.attrs["side_names_json"] = json.dumps(["right"])
                head["timestamp"] = [0, .11]
                head["state_index"] = [0, 1]
                matrices = np.tile(np.eye(4), (2, 1, 1, 1))
                matrices[1, 0, :3, 3] = [1, 2, 3]
                matrices[1, 0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
                head["world_from_wrist"] = matrices
                file["state/timestamp"] = [0, .1, .2]
                file["state/joint_names"] = np.array(names, dtype="S")
                file["state/robot_joint_position"] = np.arange(60).reshape(3, 20)
            result = MODULE.read_reference_state(path, 2, names[::-1])
            np.testing.assert_equal(result["wrist"][1], [1, 2, 3, 0, 1, 0, -1, 0, 0])
            np.testing.assert_equal(result["joints"][1], np.arange(39, 19, -1))
            np.testing.assert_equal(result["jointStateTime"], [0, .1])

    def test_asset_is_strict_escaped_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "details.js"
            MODULE.write_asset({"value": None, "note": "</script>"}, path)
            text = path.read_text()
            self.assertNotIn("</script>", text)
            self.assertEqual(json.loads(text.removeprefix("window.CONTACT_DETAILS = ").removesuffix(";\n")),
                             {"value": None, "note": "</script>"})
            with self.assertRaises(FileExistsError):
                MODULE.write_asset({}, path)


if __name__ == "__main__":
    unittest.main()
