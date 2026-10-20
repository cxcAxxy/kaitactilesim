"""Recorded rigid-geometry diagnostics never masquerade as pressure truth."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_unified import staged
from evaluation_workbench import whiteboard_contact_area as area


PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class WhiteboardAreaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / "seed_000"
        self.trial.mkdir()
        self.archive = self.root / "frozen-source"
        geometry = self.archive / area._GEOMETRY_PATH
        geometry.parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / area._GEOMETRY_PATH, geometry)
        self.identity = {"task": "whiteboard-wipe", "seed": 0, "prediction_horizon": 50,
                         "checkpoint_sha256": "a" * 64, "deployment_id": "b" * 64}
        self.inventory = [{"path": key, "size": 10, "sha256": character * 64}
                          for key, character in zip(staged._RULE_FILES.values(), ("c", "d"))]
        self.inventory.append({"path": area._GEOMETRY_PATH, "size": geometry.stat().st_size,
                               "sha256": hashlib.sha256(geometry.read_bytes()).hexdigest()})
        self.manifest_path = self.trial / "deployment_manifest.json"
        self.rows = [self.row(10.0, 0, False, 0, 0, 0),
                     self.row(10.1, 1, True, 0, 0, .1),
                     self.row(10.2, 2, True, .5, .25, .2),
                     self.row(11.0, 3, False, 0, .4, .3)]
        self.save()

    def metrics(self, contact, ratio, mean, duration):
        return {"schema_version": area._METRIC_SCHEMA,
                "measurement_type": "loaded_rigid_mesh_geometric_area_estimate",
                "is_true_deformable_contact_area": False,
                "area_denominator": area._DENOMINATOR, "area_integration": area._INTEGRATION,
                "full_pad_bottom_area_m2": .005,
                "effective_pad_contact_area_m2": ratio * .005,
                "effective_pad_contact_area_ratio": ratio,
                "contact_time_weighted_mean_effective_area_ratio": mean,
                "pad_board_contact_duration_s": duration,
                "pad_board_loaded_contact": contact}

    def row(self, time, tick, contact, ratio, mean, duration):
        return {**self.identity, "version": "staged_v1", "stage_count": 1,
                "stage_definitions": {"stage1": "picked_up"}, "stages": {"stage1": False},
                "stage_flags": {"stage1": False}, "named_stages": {"picked_up": False},
                "highest_stage": 0, "first_times": {"stage1": None}, "named_first_times": {"picked_up": None},
                "success": None, "complete_success": None, "whiteboard_complete_success_defined": False,
                "thresholds": {"lift_m": .02}, "simulation_time_s": time, "time_s": time,
                "control_tick": tick, "time_basis": "mujoco_data_time_seconds",
                "stage_achievement_policy": "ordered_cumulative_ever_reached",
                "measurements": {"simulation_time_s": time, "time_s": time,
                                 "contact_force_source": "recorded_solver", "missing_measurements": [],
                                 "native_task_state": {"cleaned_fraction": .2, "maximum_ink_remaining": .9}},
                "continuous_metrics": self.metrics(contact, ratio, mean, duration),
                "observation_gap_count": 0}

    def save(self):
        inventory_sha = staged._digest(self.inventory)
        manifest = {**self.identity, "evaluation_source_files": self.inventory,
                    "evaluation_source_sha256": inventory_sha}
        write_json(self.manifest_path, manifest)
        result = {k: deepcopy(v) for k, v in self.rows[-1].items() if k != "control_tick"}
        result.update(full_duration_evaluation=True, state_log="staged_states.jsonl",
                      state_log_samples=len(self.rows), legacy_task_success=False,
                      state_log_sampling="recorded control endpoints")
        self.summary = {**self.identity, "status": "task_not_completed", "stop_reason": "time_limit",
                        "full_duration_evaluation": True, "sim_seconds": self.rows[-1]["simulation_time_s"],
                        "max_sim_seconds": self.rows[-1]["simulation_time_s"],
                        "task_success": False, "physics_valid": True, "diagnostic_only": True,
                        "formal_metrics_valid": False, "video": {"completed": True},
                        "deployment_manifest": str(self.manifest_path), "staged_evaluation": result,
                        "deployment_integrity": {"status": "verified", "errors": [],
                            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                            "evaluation_source": {"status": "verified", "root": str(self.archive),
                                "expected_sha256": inventory_sha, "actual_sha256": inventory_sha}}}
        write_json(self.trial / "summary.json", self.summary)
        write_json(self.trial / "staged_evaluation.json", result)
        (self.trial / "staged_states.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows), encoding="utf-8")
        loaded = staged.load_staged_result(self.trial, include_state_log=False)
        self.report = {"trial": {**self.identity, "source": str(self.trial),
                                 "cohort": {"checkpoint": self.identity["checkpoint_sha256"]}},
                       "recorded_staged_evaluation": {
                           "staged_result": loaded["staged_result"], "rule_identity": loaded["rule_identity"],
                           "data_evidence": loaded["data_evidence"]},
                       "provenance": loaded["provenance"], "outcome": {"task_success": False}}

    def build(self, **kwargs):
        kwargs.setdefault("time_origin_s", 0.)
        return area.build_whiteboard_contact_area(self.identity["task"], self.report, **kwargs)

    def assert_unavailable(self, value):
        self.assertEqual(value["status"], "unavailable", value)
        self.assertEqual(value["series"], [])
        self.assertIsNone(value["summary"][0]["value"])

    def test_loaded_zero_is_real_zero_but_unloaded_zero_is_unknown(self):
        result = self.build(time_origin_s=10.)
        self.assertEqual(result["status"], "verified", result)
        instant, mean, residual = result["series"]
        self.assertEqual(instant["values"], [None, 0., .5, None])
        self.assertEqual(mean["values"], [None, 0., .25, .4])
        self.assertEqual(residual["key"], "maximum_ink_remaining")
        self.assertEqual(residual["values"], [.9] * 4)
        self.assertEqual(instant["time_s"], [row["time_s"] - 10. for row in self.rows])
        self.assertEqual(instant["gap_after"], [False, False, True, False])
        self.assertEqual(result["summary"][0]["value"], .4)
        self.assertEqual(result["summary"][1]["value"], .3)
        self.assertFalse(result["binding"]["rules_reexecuted"])
        self.assertFalse(result["binding"]["is_true_deformable_contact_area"])
        self.assertTrue(all(row["presentation_extension"] == area.SCHEMA_VERSION for row in result["series"]))

    def test_mean_is_archived_physics_integral_not_mean_of_plot_samples(self):
        result = self.build()
        # Deliberately different from arithmetic/30 Hz mean of instantaneous samples.
        self.assertEqual(result["summary"][0]["value"], .4)
        self.assertNotEqual(result["summary"][0]["value"], (0. + .5) / 2)
        self.assertIn("不从 30 Hz", result["series"][1]["note"])

    def test_no_contact_average_is_not_applicable_not_zero(self):
        for row in self.rows:
            row["continuous_metrics"] = self.metrics(False, 0, 0, 0)
        self.save()
        result = self.build()
        self.assertEqual(result["status"], "verified")  # Native ink residual still exists.
        self.assertTrue(all(v is None for v in result["series"][0]["values"]))
        self.assertTrue(all(v is None for v in result["series"][1]["values"]))
        self.assertIsNone(result["summary"][0]["value"])
        self.assertEqual(result["summary"][0]["status"], "no_contact")
        self.assertEqual(result["summary"][0]["recorded_value"], 0)
        self.assertEqual(result["summary"][1]["value"], 0)

    def test_no_contact_provenance_remains_verified_without_native_ink_samples(self):
        for row in self.rows:
            row["continuous_metrics"] = self.metrics(False, 0, 0, 0)
            row["measurements"]["native_task_state"].pop("maximum_ink_remaining")
        self.save()
        result = self.build()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["series"]), 2)
        self.assertTrue(all(value is None for series in result["series"] for value in series["values"]))
        self.assertEqual(result["summary"][0]["status"], "no_contact")

    def test_missing_sample_does_not_copy_final_value_or_create_zero(self):
        self.rows[1]["continuous_metrics"] = {}
        self.rows[1]["measurements"]["native_task_state"].pop("maximum_ink_remaining")
        self.save()
        result = self.build()
        self.assertEqual(result["status"], "verified")
        self.assertIsNone(result["series"][0]["values"][1])
        self.assertIsNone(result["series"][1]["values"][1])
        self.assertIsNone(result["series"][2]["values"][1])

    def test_cannot_infer_maximum_residual_from_mean_cleanliness(self):
        for row in self.rows:
            row["measurements"]["native_task_state"].pop("maximum_ink_remaining")
        self.save()
        self.assertEqual(len(self.build()["series"]), 2)

    def test_unknown_contact_boolean_does_not_count_as_contact(self):
        self.rows[2]["continuous_metrics"]["pad_board_loaded_contact"] = 1
        self.save()
        self.assertIsNone(self.build()["series"][0]["values"][2])

    def test_ratio_numerator_mismatch_remains_unknown(self):
        self.rows[2]["continuous_metrics"]["effective_pad_contact_area_m2"] = .001
        self.save()
        self.assertIsNone(self.build()["series"][0]["values"][2])

    def test_invalid_or_infinite_ratios_are_not_clipped_into_success(self):
        for value in (-.1, 1.1, True, "0.5", None):
            with self.subTest(value=value):
                self.rows[2]["continuous_metrics"]["effective_pad_contact_area_ratio"] = value
                self.save()
                self.assertIsNone(self.build()["series"][0]["values"][2])
        self.assertIsNone(area._fraction(float("inf")))
        self.assertIsNone(area._fraction(float("nan")))

    def test_changed_denominator_or_unreviewed_method_fails_closed(self):
        original = deepcopy(self.rows)
        for key, value in (("full_pad_bottom_area_m2", .006), ("area_denominator", "ink_union"),
                           ("is_true_deformable_contact_area", True), ("schema_version", "future-v2")):
            with self.subTest(key=key):
                self.rows = deepcopy(original)
                self.rows[1]["continuous_metrics"][key] = value
                self.save()
                self.assert_unavailable(self.build())

    def test_missing_metrics_are_explicitly_unavailable(self):
        for row in self.rows:
            row["continuous_metrics"] = {}
        self.save()
        self.assert_unavailable(self.build())

    def test_source_report_and_native_outcomes_never_mutated(self):
        original = deepcopy(self.report)
        self.build()
        self.assertEqual(self.report, original)
        self.assertIsNone(self.report["recorded_staged_evaluation"]["staged_result"]["complete_success"])
        self.assertFalse(self.report["outcome"]["task_success"])

    def test_merged_crop_manifest_role_is_verified_not_assumed(self):
        witness = next(row for row in self.report["provenance"] if row["role"] == "staged_deployment_manifest")
        witness["role"] = "model_crop_deployment_manifest"
        self.assertEqual(self.build()["status"], "verified")
        witness["sha256"] = "f" * 64
        self.assert_unavailable(self.build())

    def test_report_sha_binds_all_four_imported_evidence_artifacts(self):
        for role in ("staged_summary", "staged_result", "staged_deployment_manifest", "staged_state_log"):
            with self.subTest(role=role):
                report = deepcopy(self.report)
                next(row for row in self.report["provenance"] if row["role"] == role)["sha256"] = "f" * 64
                self.assert_unavailable(self.build())
                self.report = report

    def test_source_file_missing_or_changed_has_no_current_default_fallback(self):
        geometry = self.archive / area._GEOMETRY_PATH
        original = geometry.read_bytes()
        geometry.write_bytes(original + b"\n")
        self.assert_unavailable(self.build())
        geometry.unlink()
        self.assert_unavailable(self.build())

    def test_unreviewed_inventory_and_contradictory_report_fail_closed(self):
        self.inventory[-1]["sha256"] = "e" * 64
        self.save()
        self.assert_unavailable(self.build())
        self.inventory[-1]["sha256"] = area._GEOMETRY_SHA
        self.save()
        self.report["recorded_staged_evaluation"]["staged_result"]["continuous_metrics"]["contact_time_weighted_mean_effective_area_ratio"] = 1.
        self.assert_unavailable(self.build())

    def test_tampered_log_is_not_accepted_as_new_truth(self):
        path = self.trial / "staged_states.jsonl"
        path.write_text(path.read_text().replace('"maximum_ink_remaining": 0.9', '"maximum_ink_remaining": 0.8', 1))
        self.assert_unavailable(self.build())

    def test_all_three_whiteboard_tasks_supported_without_cross_task_fallback(self):
        for task in sorted(area._TASKS):
            with self.subTest(task=task):
                self.identity["task"] = task
                for row in self.rows:
                    row["task"] = task
                self.save()
                self.assertEqual(self.build()["status"], "verified")
        self.assertEqual(area.build_whiteboard_contact_area("usb-insert", {})["status"], "not_applicable")

    def test_invalid_origin_rejected_instead_of_shifting_or_sorting(self):
        for value in (True, None, "0", float("inf")):
            with self.subTest(value=value):
                self.assert_unavailable(self.build(time_origin_s=value))
        self.assert_unavailable(area.build_whiteboard_contact_area(self.identity["task"], self.report))

    def test_clock_evidence_mismatch_is_not_silently_aligned(self):
        self.report["recorded_staged_evaluation"]["data_evidence"]["clock"] = "wall_time_seconds"
        self.assert_unavailable(self.build())


if __name__ == "__main__":
    unittest.main()
