"""Model-only PNG fallback never reconstructs a missing reference trajectory."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_workbench import model_plots
from evaluation_workbench.data import build_case
import test_evaluation_workbench_data as fixture_data

write_json = fixture_data.write_json


def fake_render(stage, arrays, gaps, title):
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))
    for row in model_plots._PLOTS:
        (stage / row[1]).write_bytes(png)


class ModelPlotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "seed_000"
        self.root.mkdir()
        self.output = self.parent / "new_case"
        directory, _, _, archive = fixture_data.GenericCaseTests.fixture(self)
        joints = [f"hand_r_joint_{index}" for index in range(20)]
        state = np.arange(4 * 29, dtype=float).reshape(4, 29) / 100
        archive.update(state_29=state, wrist_state=state[:, :9], hand_joint_position=state[:, 9:], joint_names=np.array(joints))
        self.archive = archive
        np.savez(directory / "evaluation_rollout_trace.npz", **archive)
        for path in (self.root / "summary.json", directory / "review.json"):
            value = json.loads(path.read_text())
            review = value["video"] if path.name == "summary.json" else value
            comparison = review["comparison_plots"]
            comparison["state"] = {"layout": "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof", "dimension": 29,
                                   "wrist_site": "hand_r_base_link_site", "rot6d": "first rotation column followed by second rotation column",
                                   "joint_names": joints}
            comparison["rollout"] = {"samples": 4, "source_start_time_s": float(archive["simulation_time_s"][0]),
                                     "duration_s": float(archive["time_s"][-1])}
            write_json(path, value)
        self.report = build_case(self.root)["report"]

    def rebind_trace(self):
        path = self.root / "evaluation_rollout_trace.npz"
        np.savez(path, **self.archive)
        row = next(row for row in self.report["provenance"] if row["path"] == str(path))
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)

    def generate(self):
        return model_plots.generate_model_plots(self.root, self.output, self.report)

    def test_source_bound_only_model_files_and_descriptor(self):
        before = {str(path): path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        with patch.object(model_plots, "_render", side_effect=fake_render):
            config = self.generate()
        self.assertEqual(config["comparison_kind"], "model_only")
        self.assertFalse(config["generated"]["reference_loaded"])
        self.assertFalse(config["generated"]["resampled"])
        self.assertFalse(config["generated"]["metrics_recomputed"])
        self.assertEqual(config["generated"]["sample_count"], 4)
        self.assertEqual(config["generated"]["gap_count"], 1)
        self.assertEqual(config["generated"]["units"]["joint"], "rad")
        self.assertEqual(len(config["plots"]), 3)
        self.assertEqual(set(config["plots"][i]["id"] for i in range(3)), {"tactile", "hand-dof", "wrist"})
        for plot in config["plots"]:
            path = self.output / plot["url"]
            self.assertTrue(path.name.startswith("model_right_"))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), plot["sha256"])
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.iterdir() if path.is_file()})
        json.dumps(config, allow_nan=False)
        with self.assertRaises(FileExistsError):
            self.generate()

    def test_bad_sha_or_horizon_never_creates_output(self):
        path = self.root / "evaluation_rollout_trace.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"modified")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.generate()
        path.write_bytes(original)
        self.report["trial"]["prediction_horizon"] = 30
        with self.assertRaisesRegex(ValueError, "H50"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_state_shape_and_cross_member_consistency_are_required(self):
        self.archive["wrist_state"] = np.zeros((4, 9))
        self.rebind_trace()
        with self.assertRaisesRegex(ValueError, "arrays or clocks"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_wrong_clock_and_object_dtype_fail_before_plotting(self):
        self.archive["time_s"] = np.array([0., .1, .1, .8])
        self.rebind_trace()
        with self.assertRaisesRegex(ValueError, "clock"):
            self.generate()
        self.archive["time_s"] = self.archive["simulation_time_s"] - self.archive["simulation_time_s"][0]
        self.archive["state_29"] = np.array([["unknown"] * 29] * 4, dtype=object)
        self.rebind_trace()
        with self.assertRaisesRegex(ValueError, "shape, dtype"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_input_changes_during_render_prevent_publication(self):
        def mutate(stage, arrays, gaps, title):
            fake_render(stage, arrays, gaps, title)
            path = self.root / "summary.json"
            path.write_bytes(path.read_bytes() + b" ")
        with patch.object(model_plots, "_render", side_effect=mutate), self.assertRaisesRegex(ValueError, "SHA256"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_refuses_writing_under_original_trial(self):
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            model_plots.generate_model_plots(self.root, self.root / "derived", self.report)
        self.assertFalse((self.root / "derived").exists())
        self.output.mkdir()
        (self.output / "plots").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.generate()
        self.assertFalse(list(self.root.glob("model_right_*.png")))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is in the project pixi interpreter")
    def test_real_agg_renderer_outputs_three_expected_dimension_pngs(self):
        from matplotlib.figure import Figure
        original, shapes = Figure.subplots, []
        def record_shape(figure, rows=1, columns=1, **kwargs):
            shapes.append((rows, columns))
            return original(figure, rows, columns, **kwargs)
        with patch.object(Figure, "subplots", record_shape):
            config = self.generate()
        self.assertEqual(shapes, [(5, 2), (5, 4), (3, 3)])
        self.assertEqual([(row["width"], row["height"]) for row in config["plots"]],
                         [(1560, 1800), (1920, 1920), (1560, 1200)])
        self.assertTrue(all((self.output / row["url"]).stat().st_size > 10_000 for row in config["plots"]))


if __name__ == "__main__":
    unittest.main()
