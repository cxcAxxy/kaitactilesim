"""Five-finger scalar loads retain source identity, cancellation and unknowns."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_tactile_load_extension import data


NAMES = ["hand_r_thumb_link6", "hand_r_index_link4", "hand_r_middle_link4", "hand_r_ring_link4", "hand_r_pinky_link4"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(path, role):
    raw = path.read_bytes()
    return {"path": str(path), "role": role, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class LoadSeriesTests(unittest.TestCase):
    def build(self, **overrides):
        args = {"time_s": [0, .01], "normal_rows": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                "tangent_rows": [[.1, .2, .3, .4, .5], [.2, .3, .4, .5, .6]],
                "channel_names": NAMES.copy(), "gap_after": [False, False], "nominal_interval_s": .01}
        args.update(overrides)
        return data.build_load_series(**args)

    def test_full_native_values_and_scalar_totals(self):
        normal = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
        result = self.build(normal_rows=normal)
        self.assertEqual(result["normal_force_n"], normal)
        self.assertEqual(result["normal_total_n"], [15, 20])
        self.assertEqual(result["tangent_total_n"], [1.5, 2.0])
        self.assertEqual(result["channels"][0], {"name": NAMES[0], "label": "右拇指"})
        self.assertIn("不是全手全局净切向力", result["semantics"]["tangent_total_n"])
        json.dumps(result, allow_nan=False)

    def test_missing_finger_never_becomes_partial_hand_total(self):
        result = self.build(normal_rows=[[1, None, 3, 4, 5], [2, 3, 4, 5, 6]],
                            tangent_rows=[[.1, .2, .3, .4, .5], [None, .3, .4, .5, .6]])
        self.assertEqual(result["normal_total_n"], [None, 20])
        self.assertEqual(result["tangent_total_n"], [1.5, None])
        self.assertEqual(result["normal_status"], "partial")

    def test_negative_nonfinite_bool_and_missing_rows_are_null(self):
        result = self.build(normal_rows=[[-1e-15, np.nan, np.inf, True, 1], None], tangent_rows=None)
        self.assertEqual(result["normal_force_n"][0], [None, None, None, None, 1])
        self.assertEqual(result["normal_total_n"], [None, None])
        self.assertEqual(result["tangent_status"], "missing")
        self.assertEqual(result["tangent_total_n"], [None, None])
        json.dumps(result, allow_nan=False)

    def test_sum_overflow_remains_json_safe(self):
        result = self.build(normal_rows=[[1e308] * 5] * 2)
        self.assertEqual(result["normal_total_n"], [None, None])
        json.dumps(result, allow_nan=False)

    def test_identity_is_explicit_and_order_is_preserved(self):
        result = self.build(channel_names=list(reversed(NAMES)))
        self.assertEqual(result["channels"][0]["label"], "右小指")
        for names in (NAMES[:4], [NAMES[0]] * 5, ["channel_0"] + NAMES[1:]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.build(channel_names=names)

    def test_invalid_clocks_and_intervals_rejected_without_sorting(self):
        for times in ([0, 0], [.01, 0], [0, np.nan], [0, True], ["0", ".01"], np.array(0)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.build(time_s=times)
        for interval in (0, -1, True, np.nan):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.build(nominal_interval_s=interval)

    def test_vectors_and_wrong_shapes_not_accepted_as_Ft(self):
        for tangent in ([[[1, 2]] * 5] * 2, [[1] * 4] * 2, [[1] * 5]):
            with self.subTest(tangent=tangent), self.assertRaises(ValueError):
                self.build(tangent_rows=tangent)

    def test_declared_gaps_preserved_and_long_gaps_detected(self):
        self.assertEqual(self.build(gap_after=[True, False])["gap_after"], [True, False])
        result = data.build_load_series([0, .01, .02, 1], [[0] * 5] * 4, None, NAMES,
                                        [False] * 4, .01)
        self.assertEqual(result["gap_after"], [False, False, True, False])


class LoadExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source/seed_000"
        self.source.mkdir(parents=True)
        self.trace = self.source / "evaluation_rollout_trace.npz"
        self.old_path = self.root / "old/009/report.json"
        self.h50_path = self.root / "h50/004/report.json"
        self.times = [0., .01, .02, .03]
        self.normal = np.arange(20, dtype=float).reshape(4, 5) / 10
        self.tangent = self.normal / 2
        self.write_trace()
        self.summary = {"task": "whiteboard-wipe", "prediction_horizon": 50, "video": {"comparison_plots": {
            "schema": "kaihand-evaluation-comparison-v1",
            "rollout": {"samples": 4, "source_start_time_s": 0},
            "tactile": {"unit": "N", "taxel_grid": [7, 5], "link_names": NAMES,
                        "Fn": "sum of all 7x5 normal taxel forces",
                        "Ft": "sqrt(sum(Ft_x over 7x5)^2 + sum(Ft_y over 7x5)^2)"},
            "artifacts": {"rollout_trace": str(self.trace)}}}}
        self.visual = {"schema_version": "evaluation-visualization-v1", "task_summary": {"task": "whiteboard-wipe"},
                       "clock_aligned": True, "time_origin_s": 0, "channels": [{"name": name} for name in NAMES],
                       "task_series": [{"key": "cleaned_fraction", "time_s": self.times, "values": [0, 0, .2, .2]}],
                       "tactile_series": {"time_s": self.times, "normal_force_n": self.normal.tolist(),
                                          "gap_after": [False] * 4, "nominal_interval_s": .01},
                       "contact_glyph": {"tangent_change_n": [[[99, 99]] * 5] * 4},
                       "media": {"status": "missing"}}
        self.refresh()

    def write_trace(self, *, tangent=True, overrides=None):
        values = {"time_s": np.array(self.times), "simulation_time_s": np.array(self.times),
                  "fingertip_link_names": np.array(NAMES), "fingertip_normal_force_n": self.normal}
        if tangent:
            values["fingertip_tangent_force_n"] = self.tangent
        values.update(overrides or {})
        np.savez_compressed(self.trace, **values)

    def refresh(self, *, trace=True):
        write_json(self.source / "summary.json", self.summary)
        self.report = {"schema_version": "task-tactile-evaluation-v1", "suite_schema_version": "task-tactile-suite-v2",
                       "trial": {"trial_id": "fixture", "task": "whiteboard-wipe", "source": str(self.source)},
                       "outcome": {"status": "fail"},
                       "provenance": [row(self.source / "summary.json", "recorded_trial_outcome")]}
        if trace:
            self.report["provenance"].append(row(self.trace, "native_control_cadence_tactile"))
        write_json(self.old_path, self.report)
        stable = deepcopy(self.report)
        stable["trial"]["prediction_horizon"] = 50
        stable["prediction_horizon_evidence"] = {"status": "verified", "prediction_horizon": 50}
        write_json(self.h50_path, stable)
        for path in (self.old_path, self.h50_path):
            write_json(path.parent / "visualization.json", self.visual)

    def load(self):
        return data.load_extension(self.old_path, self.h50_path)

    def test_full_native_fields_and_provenance(self):
        result = self.load()
        self.assertEqual(result["prediction_horizon"], 50)
        self.assertEqual(result["trial_id"], "fixture")
        self.assertEqual(result["time_s"], self.times)
        self.assertEqual(result["normal_force_n"], self.normal.tolist())
        self.assertEqual(result["tangent_force_n"], self.tangent.tolist())
        self.assertEqual(result["tangent_status"], "available")
        self.assertEqual(result["warnings"], [])
        self.assertTrue(any(item["path"] == str(self.trace) for item in result["provenance"]))
        json.dumps(result, allow_nan=False)

    def test_missing_Ft_never_uses_delta_Ft(self):
        self.write_trace(tangent=False)
        self.refresh()
        result = self.load()
        self.assertEqual(result["tangent_force_n"], [[None] * 5] * 4)
        self.assertEqual(result["tangent_total_n"], [None] * 4)
        self.assertIn("tangent_change_n", " ".join(result["warnings"]))

    def test_missing_trace_binding_never_discovers_unbound_npz(self):
        self.refresh(trace=False)
        result = self.load()
        self.assertIsNone(result["native_trace"])
        self.assertEqual(result["normal_force_n"], self.normal.tolist())
        self.assertEqual(result["tangent_status"], "missing")

    def test_missing_Ft_formula_is_unknown_even_if_array_exists(self):
        del self.summary["video"]["comparison_plots"]["tactile"]["Ft"]
        self.refresh()
        self.assertEqual(self.load()["tangent_status"], "missing")

    def test_invalid_Ft_cells_are_unknown_and_not_clamped(self):
        self.tangent[1, 2] = -.0000000001
        self.tangent[2, 3] = np.nan
        self.write_trace()
        self.refresh()
        result = self.load()
        self.assertIsNone(result["tangent_force_n"][1][2])
        self.assertEqual(result["tangent_total_n"][1:3], [None, None])

    def test_old_and_h50_visual_differences_rejected(self):
        changed = deepcopy(self.visual)
        changed["tactile_series"]["normal_force_n"][0][0] = 3
        write_json(self.old_path.parent / "visualization.json", changed)
        with self.assertRaisesRegex(ValueError, "visualizations differ"):
            self.load()

    def test_same_modified_visuals_still_must_match_native_normal(self):
        self.visual["tactile_series"]["normal_force_n"][0][0] = 3
        self.refresh()
        with self.assertRaisesRegex(ValueError, "values disagree"):
            self.load()

    def test_wrong_old_trial_and_source_identity_rejected(self):
        for key, value in (("trial_id", "different"), ("source", str(self.root / "other"))):
            with self.subTest(key=key):
                self.refresh()
                self.report["trial"][key] = value
                write_json(self.old_path, self.report)
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.load()

    def test_stale_old_source_binding_and_changed_native_bytes_failclosed(self):
        self.report["provenance"][1]["sha256"] = "a" * 64
        write_json(self.old_path, self.report)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()
        self.refresh()
        self.write_trace(overrides={"fingertip_tangent_force_n": self.tangent + 1})
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_wrong_units_and_Ft_formula_rejected(self):
        original = deepcopy(self.summary)
        for key, value in (("unit", "Pa"), ("Ft", "sum of per-taxel magnitudes")):
            with self.subTest(key=key):
                self.summary = deepcopy(original)
                self.summary["video"]["comparison_plots"]["tactile"][key] = value
                self.refresh()
                with self.assertRaisesRegex(ValueError, "schema|definition"):
                    self.load()

    def test_nonmonotone_or_wrong_native_clocks_and_names_rejected(self):
        for overrides in ({"simulation_time_s": [0, .01, .01, .03]},
                          {"simulation_time_s": [0, .011, .02, .03]},
                          {"fingertip_link_names": np.array(list(reversed(NAMES)))}):
            with self.subTest(overrides=overrides):
                self.write_trace(overrides=overrides)
                self.refresh()
                with self.assertRaises(ValueError):
                    self.load()

    def test_source_changed_during_decode_fails_final_hash_check(self):
        original = data._read_trace
        def changed(path):
            result = original(path)
            self.write_trace(overrides={"fingertip_tangent_force_n": self.tangent + 1})
            return result
        with patch.object(data, "_read_trace", changed), self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_header_shape_bomb_and_object_arrays_rejected(self):
        self.write_trace(overrides={"fingertip_tangent_force_n": np.full((4, 5), "x", dtype=object)})
        self.refresh()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.load()
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 5)})
        self.write_trace(tangent=False)
        with zipfile.ZipFile(self.trace, "a") as archive:
            archive.writestr("fingertip_tangent_force_n.npy", payload.getvalue())
        self.refresh()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load()

    def test_taxel_arrays_are_not_loaded(self):
        with zipfile.ZipFile(self.trace, "a") as archive:
            archive.writestr("tangent_taxel_force_n.npy", b"not a valid array")
        self.refresh()
        self.assertEqual(self.load()["tangent_status"], "available")


if __name__ == "__main__":
    unittest.main()
