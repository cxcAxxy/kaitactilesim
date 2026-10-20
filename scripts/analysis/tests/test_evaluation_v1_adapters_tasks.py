"""Task evidence must survive missing tactile and never be inferred from stage names."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v1.adapters import InputChangedError, TrialData, load_trial
from evaluation_v1.tasks import build_task_report


class AdapterTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_summary(self, task="whiteboard-wipe", success=False, **kwargs):
        summary = {"task": task, "status": "task_not_completed", "sim_seconds": 90,
                   "max_sim_seconds": 90, "wall_seconds": 150,
                   "evaluation": {"task": task, "success": success, "state": {}}}
        summary.update(kwargs)
        (self.root / "summary.json").write_text(json.dumps(summary))
        return summary

    def write_npz(self, **kwargs):
        (self.root / "review").mkdir(exist_ok=True)
        values = {"simulation_time_s": np.array([0, .033, .067]),
                  "normal_taxel_force_n": np.ones((3, 2, 2, 2)),
                  "tangent_taxel_force_n": np.ones((3, 2, 2, 2, 2)),
                  "fingertip_link_names": np.array(["thumb", "index"])}
        values.update(kwargs)
        np.savez(self.root / "review/evaluation_rollout_trace.npz", **values)

    def write_frames(self):
        (self.root / "review").mkdir(exist_ok=True)
        rows = []
        for i in range(3):
            rows.append({"simulation_time_s": i / 10, "phase": "inserting",
                         "fingertip_order": ["hand_l_thumb_link6", "hand_r_thumb_link6"],
                         "normal_taxel_force_n": [[[2., 2.]], [[3., 3.]]],
                         "tangent_taxel_force_n": [[[[1., -1.], [1., -1.]]], [[[2., -2.], [2., -2.]]]],
                         "task_metrics": {"seated": False, "stage": "inserting"}})
        (self.root / "review/frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    def test_outcome_survives_missing_tactile_and_timeout_remains_failure(self):
        self.write_summary()
        trial = load_trial(self.root)
        self.assertIsNone(trial.normal_force_n)
        report = build_task_report(trial, {})
        self.assertEqual(report["outcome"]["status"], "fail")
        self.assertEqual(report["outcome"]["termination_reason"], "time_limit")
        self.assertEqual(report["efficiency"]["wall_duration_s"], 150)
        self.assertEqual(report["efficiency"]["simulation_duration_s"], 90)
        self.assertEqual(report["safety"]["status"], "unknown")

    def test_npz_preserves_signed_taxels_unknown_geometry_and_source_hash(self):
        self.write_summary()
        self.write_npz(tangent_taxel_force_n=-np.ones((3, 2, 2, 2, 2)))
        trial = load_trial(self.root)
        np.testing.assert_equal(trial.normal_force_n, np.full((3, 2), 4.))
        self.assertTrue(np.all(trial.tangent_taxel_force_n == -1))
        self.assertIsNone(trial.taxel_positions_local_m)
        self.assertEqual(trial.channel_names, ["thumb", "index"])
        self.assertTrue(trial.task_tactile_clock_aligned)
        for source in trial.provenance:
            data = Path(source["path"]).read_bytes()
            self.assertEqual(source["sha256"], sha256(data).hexdigest())
            self.assertEqual(source["size"], len(data))

    def test_review_only_fallback_is_explicit_and_uses_recorded_channel_identity(self):
        self.write_summary(task="install-ram")
        self.write_frames()
        trial = load_trial(self.root)
        np.testing.assert_allclose(trial.time_s, [0, .1, .2])
        np.testing.assert_allclose(trial.normal_force_n, [[4., 6.], [4., 6.], [4., 6.]])
        self.assertEqual(trial.channel_names, ["hand_l_thumb_link6", "hand_r_thumb_link6"])
        self.assertTrue(trial.task_tactile_clock_aligned)
        self.assertIn("10_hz", trial.control_rows[0]["_sampling"])
        self.assertTrue(any("10 Hz" in warning for warning in trial.warnings))
        report = build_task_report(trial, {})
        self.assertEqual(report["scope_intervals"], [])
        self.assertFalse(report["events"][0]["ground_truth"])

    def test_npz_not_replaced_by_lower_frequency_frame_tactile(self):
        self.write_summary(task="sponge-grasp")
        self.write_npz()
        self.write_frames()
        trial = load_trial(self.root)
        np.testing.assert_allclose(trial.time_s, [0, .033, .067])
        self.assertEqual(trial.normal_force_n.shape, (3, 2))
        self.assertEqual(len(trial.control_rows), 3)

    def test_control_records_take_precedence_over_frame_task_metrics(self):
        self.write_summary()
        self.write_npz()
        (self.root / "control_steps.jsonl").write_text(json.dumps({"sim_time": .033, "task": {"cleaned_fraction": .25}}))
        self.write_frames()
        trial = load_trial(self.root)
        self.assertEqual(trial.control_rows[0]["task"]["cleaned_fraction"], .25)
        self.assertFalse(any(row["path"].endswith("frames.jsonl") for row in trial.provenance))

    def test_nonmonotonic_native_clock_is_not_silently_sorted(self):
        self.write_summary()
        self.write_npz(simulation_time_s=np.array([0, .1, .1]))
        trial = load_trial(self.root)
        self.assertIsNone(trial.time_s)
        self.assertTrue(any("strictly increasing" in warning for warning in trial.warnings))
        self.assertEqual(build_task_report(trial, {})["outcome"]["status"], "fail")

    def test_two_dimensional_taxel_positions_do_not_fabricate_surface_geometry(self):
        self.write_summary()
        self.write_npz(taxel_positions_local_m=np.zeros((2, 2, 2, 2)))
        trial = load_trial(self.root)
        self.assertIsNone(trial.taxel_positions_local_m)
        self.assertIsNotNone(trial.normal_force_n)
        self.assertTrue(any("2D taxel" in warning for warning in trial.warnings))

    def test_three_dimensional_taxel_positions_are_preserved(self):
        self.write_summary()
        geometry = np.arange(24).reshape(2, 2, 2, 3) * .001
        self.write_npz(taxel_positions_local_m=geometry)
        trial = load_trial(self.root)
        np.testing.assert_equal(trial.taxel_positions_local_m, geometry)

    def test_elapsed_only_npz_does_not_claim_task_clock_alignment(self):
        self.write_summary()
        (self.root / "review").mkdir()
        np.savez(self.root / "review/evaluation_rollout_trace.npz",
                 time_s=np.array([0., .1, .2]), fingertip_normal_force_n=np.ones((3, 2)))
        (self.root / "control_steps.jsonl").write_text("\n".join(json.dumps({
            "sim_time": time, "task": {"board_normal_force_n": 1.}
        }) for time in (10., 10.1, 10.2)))
        trial = load_trial(self.root)
        self.assertFalse(trial.task_tactile_clock_aligned)
        self.assertIsNotNone(trial.normal_force_n)
        np.testing.assert_equal(trial.time_s, [0., .1, .2])
        self.assertTrue(any("alignment is unverified" in warning for warning in trial.warnings))

    def test_negative_or_missing_normal_force_is_unknown_not_zero(self):
        self.write_summary()
        values = np.ones((3, 2, 2, 2))
        values[1, 0, 0, 0] = -1
        self.write_npz(normal_taxel_force_n=values)
        trial = load_trial(self.root)
        self.assertTrue(np.isnan(trial.normal_force_n[1, 0]))
        self.assertEqual(trial.normal_force_n[1, 1], 4.)

    def test_nonfinite_summary_values_export_as_unknown_with_original_provenance(self):
        self.write_summary(evaluation={"success": False, "state": {"cleaned_fraction": float("nan")}})
        trial = load_trial(self.root)
        report = build_task_report(trial, {})
        self.assertIsNone(report["task_specific"]["partial_state"]["cleaned_fraction"])
        json.dumps(report, allow_nan=False)
        self.assertTrue(any("nonfinite JSON" in warning for warning in trial.warnings))

    def test_corrupt_optional_npz_does_not_destroy_outcome(self):
        self.write_summary()
        (self.root / "review").mkdir()
        (self.root / "review/evaluation_rollout_trace.npz").write_bytes(b"PK\x03\x04broken")
        trial = load_trial(self.root)
        self.assertIsNone(trial.time_s)
        self.assertEqual(build_task_report(trial, {})["outcome"]["status"], "fail")

    def test_mutating_input_is_rejected(self):
        self.write_summary()
        with patch("evaluation_v1.adapters._signature", side_effect=[(1, 2, 10, 1), (1, 2, 10, 1), (1, 2, 11, 2)]):
            with self.assertRaises(InputChangedError):
                load_trial(self.root)

    def test_generic_does_not_guess_success_from_arbitrary_fields(self):
        self.write_summary(task="unregistered-task", success=None,
            evaluation={"task": "unregistered-task", "state": {"seated": True, "success": True}})
        report = build_task_report(load_trial(self.root), {})
        self.assertEqual(report["outcome"]["status"], "unknown")
        self.assertEqual(report["task_specific"]["plugin"], "generic_recorded_outcome")
        self.assertEqual(report["task_specific"]["partial_state"]["seated"], True)

    def test_generic_preserves_explicit_outcome_without_recertification(self):
        self.write_summary(task="unregistered-task", success=True)
        report = build_task_report(load_trial(self.root), {})
        self.assertEqual(report["outcome"]["status"], "pass")
        self.assertIn("not_independent", report["outcome"]["scope"])

    def test_conflicting_success_and_invalid_physics_are_unknown(self):
        for evaluation in ({"success": True, "state": {"success": False}},
                           {"success": True, "physics_valid": False, "state": {}}):
            self.write_summary(evaluation=evaluation)
            report = build_task_report(load_trial(self.root), {})
            self.assertEqual(report["outcome"]["status"], "unknown")
            self.assertTrue(report["outcome"]["recorded_success"])

    def test_observe_safety_is_not_enforcement(self):
        self.write_summary(evaluation={"success": False, "state": {}, "contact_guard_diagnostics": {
            "mode": "observe_only", "board_normal_force_exceeded": True,
            "peak_board_normal_force_n": 9., "board_normal_force_guard_threshold_n": 8.}})
        report = build_task_report(load_trial(self.root), {})
        self.assertEqual(report["safety"]["enforcement"], "observe")
        self.assertEqual(report["safety"]["status"], "fail")

    def test_nonterminal_success_label_is_not_final_outcome(self):
        for success in (False, True):
            self.write_summary(success=success, status="running")
            report = build_task_report(load_trial(self.root), {})
            self.assertEqual(report["outcome"]["status"], "unknown")

    def test_guard_occurrence_without_count_and_explicit_untriggered_guard(self):
        for event, expected in (({"type": "collision_guard"}, "fail"),
                                ({"kind": "force_guard", "triggered": False}, "unknown")):
            self.write_summary(evaluation={"success": False, "guard_events": [event]})
            report = build_task_report(load_trial(self.root), {})
            self.assertEqual(report["safety"]["status"], expected)

    def test_board_contact_scopes_do_not_bridge_unknown_gap_or_terminal_sample(self):
        summary = self.write_summary()
        trial = TrialData(self.root, summary, "whiteboard-wipe")
        trial.control_rows = [{"sim_time": t, "task": {"board_normal_force_n": value, "stage": "wiping"}, "_source": str(i)}
                              for i, (t, value) in enumerate([(0, 1.), (.1, None), (.2, 1.), (1., 1.), (1.1, 1.), (1.2, 1.)])]
        report = build_task_report(trial, {})
        self.assertEqual([(s["start_s"], s["end_s"]) for s in report["scope_intervals"]], [(0., .1), (1., 1.2)])
        self.assertTrue(all(s["attempt_id"] is None for s in report["scope_intervals"]))

    def test_initial_whiteboard_release_does_not_create_release_or_holding_event(self):
        summary = self.write_summary()
        trial = TrialData(self.root, summary, "whiteboard-wipe", control_rows=[
            {"sim_time": t, "task": {"maximum_lift_m": 0., "released_on_table": True, "stage": "pick_eraser"}}
            for t in (0., .1, .2)])
        report = build_task_report(trial, {})
        self.assertEqual(report["scope_intervals"], [])
        self.assertEqual({event["type"] for event in report["events"]}, {"declared_stage"})

    def test_all_false_scope_is_distinct_from_missing_predicate(self):
        summary = self.write_summary()
        for value, expected, fully in ((0., "not_observed", True), (None, "missing", False)):
            trial = TrialData(self.root, summary, "whiteboard-wipe", control_rows=[
                {"sim_time": t, "task": {"board_normal_force_n": value}}
                for t in (0., .1, .2)])
            report = build_task_report(trial, {})
            self.assertEqual(report["scope_intervals"], [])
            availability = report["scope_availability"][0]
            self.assertEqual(availability["status"], expected)
            self.assertEqual(availability["fully_observed"], fully)
            self.assertEqual(availability["coverage_fraction"], 1. if fully else 0.)

    def test_partial_missing_scope_reports_coverage_without_inventing_intervals(self):
        summary = self.write_summary()
        trial = TrialData(self.root, summary, "whiteboard-wipe", control_rows=[
            {"sim_time": t, "task": {"board_normal_force_n": value}}
            for t, value in ((0., 0.), (.1, None), (.2, 0.))])
        availability = build_task_report(trial, {})["scope_availability"][0]
        self.assertEqual(availability["status"], "not_observed")
        self.assertEqual(availability["known_sample_count"], 2)
        self.assertAlmostEqual(availability["coverage_fraction"], .5)
        self.assertFalse(availability["fully_observed"])

    def test_scope_with_invalid_clock_or_no_rows_is_missing(self):
        summary = self.write_summary()
        for rows in ([], [{"sim_time": 0., "task": {"board_normal_force_n": 1.}},
                           {"sim_time": 0., "task": {"board_normal_force_n": 1.}}]):
            trial = TrialData(self.root, summary, "whiteboard-wipe", control_rows=rows)
            report = build_task_report(trial, {})
            self.assertEqual(report["scope_availability"][0]["status"], "missing")
            self.assertEqual(report["scope_intervals"], [])

    def test_unreached_sponge_transport_is_not_perfect_transport(self):
        self.write_summary(task="sponge-grasp", evaluation={"success": False, "state": {
            "grasp_established": False, "lift_established": False, "transport_contact_valid": True,
            "release_allowed": False, "released": False}})
        report = build_task_report(load_trial(self.root), {})
        predicates = {p["name"]: p for p in report["task_specific"]["predicates"]}
        transport = predicates["recorded_transport_contact_valid"]
        self.assertEqual(transport["status"], "unknown")
        self.assertEqual(transport["applicability"], "not_reached")
        self.assertTrue(transport["value"])

    def test_usb_partial_depth_does_not_override_recorded_failure(self):
        self.write_summary(task="usb-insert", evaluation={"success": False,
            "state": {"insertion_depth_m": .03, "seated": False}})
        report = build_task_report(load_trial(self.root), {})
        self.assertEqual(report["outcome"]["status"], "fail")
        self.assertEqual(report["task_specific"]["plugin"], "insertion")
        self.assertEqual(report["task_specific"]["partial_state"]["insertion_depth_m"], .03)


if __name__ == "__main__":
    unittest.main()
