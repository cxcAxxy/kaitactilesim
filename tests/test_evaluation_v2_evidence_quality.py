"""Independent contact evidence is never fabricated from force or task stage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v2.evidence import (
    META_NAME, RECORDS_NAME, EvidenceRecorder, adapt_legacy_evidence,
    describe_evidence, load_evidence, make_metadata, validate_evidence, write_evidence,
)
from evaluation_v2.quality import assess_contact_quality


def metadata():
    meta = make_metadata("synthetic-contact-task", clock_id="test_clock", source_kind="synthetic_test", nominal_interval_s=.1)
    meta.update(complete=True, bodies={"hand": "test hand", "object": "test object"},
                coordinate_frames={"world": "test metric world frame"},
                contact_pairs={"hold": {"body_a": "hand", "body_b": "object", "target": True,
                    "role": "retention", "measurement_source": "synthetic fixture independent contact sensor",
                    "allowed_relative_motion": "none", "tangent_basis": "fixed object-local contact tangent basis",
                    "thresholds": {"contact_on_n": .05, "max_normal_force_n": 2.5, "slip_speed_m_s": .1},
                    "threshold_source": "explicit synthetic test contract, not a real safety limit"}})
    return meta


def label(value, independent=True):
    return {"value": value, "source": "independent synthetic fixture", "independent": independent}


def records(times=(0, .1, .2, .3), forces=(2, 0, 3, 9)):
    return [{"time_s": time, "valid": True, "contacts": {"hold": {
        "valid": True, "normal_force_n": force, "expected_contact": label(True)}}}
        for time, force in zip(times, forces)]


class EvidenceValidationTests(unittest.TestCase):
    def test_absent_legacy_sidecar_is_explicit_missing_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = load_evidence(directory)
            self.assertEqual(bundle.status, "missing")
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(assess_contact_quality(bundle)["status"], "missing")

    def test_roundtrip_provenance_and_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = write_evidence(directory, metadata(), records())
            self.assertEqual(bundle.status, "available")
            self.assertEqual(len(bundle.provenance), 2)
            for row in bundle.provenance:
                self.assertEqual(row["sha256"], hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest())
            before = {path.name: path.read_bytes() for path in Path(directory).iterdir()}
            with self.assertRaises(FileExistsError): write_evidence(directory, metadata(), records())
            self.assertEqual(before, {path.name: path.read_bytes() for path in Path(directory).iterdir()})
            self.assertEqual(describe_evidence(bundle)["sample_count"], 4)
            self.assertNotIn("records", bundle.summary)
            json.dumps(bundle.summary, allow_nan=False)

    def test_partial_files_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / META_NAME).write_text(json.dumps(metadata()))
            with self.assertRaisesRegex(ValueError, "incomplete"): load_evidence(root)
            other = Path(outside) / RECORDS_NAME
            other.write_text("")
            (root / RECORDS_NAME).symlink_to(other)
            with self.assertRaisesRegex(ValueError, "symlink"): load_evidence(root)

    def test_schema_nonfinite_unknown_pairs_negative_force_and_bad_clock_rejected(self):
        mutations = [lambda m, r: m.update(schema_version="unknown"),
                     lambda m, r: m.update(invented_ground_truth=True),
                     lambda m, r: r[0]["contacts"].update(unknown={"valid": True}),
                     lambda m, r: r[0]["contacts"]["hold"].update(normal_force_n=-1),
                     lambda m, r: r[0]["contacts"]["hold"].update(normal_force_n=float("nan")),
                     lambda m, r: r[1].update(time_s=0),
                     lambda m, r: m["clock"].update(aligned_to_trial=True),
                     lambda m, r: m["contact_pairs"]["hold"].pop("threshold_source")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                meta, rows = metadata(), records(); mutation(meta, rows)
                with self.assertRaises(ValueError): validate_evidence(meta, rows)

    def test_poses_need_declared_frame_and_normalized_quaternion(self):
        rows = records()
        rows[0]["objects"] = {"object": {"position_m": [0, 0, .1], "quaternion_wxyz": [1, 0, 0, 0],
                                            "frame": "world", "valid": True, "source": "test measured pose"}}
        rows[0]["goals"] = {"placement": {"position_m": [0, 0, .2], "frame": "world", "valid": True, "source": "configured goal"}}
        validate_evidence(metadata(), rows)
        rows[0]["objects"]["object"]["quaternion_wxyz"] = [2, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "normalized"): validate_evidence(metadata(), rows)
        rows[0]["objects"]["object"]["quaternion_wxyz"] = [1, 0, 0, 0]
        rows[0]["objects"]["object"]["frame"] = "guessed"
        with self.assertRaisesRegex(ValueError, "frame"): validate_evidence(metadata(), rows)

    def test_recorder_defensive_copy_and_task_evidence_contract(self):
        meta = metadata()
        meta["task_fields"] = {"lateral_error_m": {"source": "observed geometry", "kind": "measurement", "unit": "m"}}
        meta["task_config"] = {"contract_id": "test-v1", "source_sha256": "a" * 64}
        recorder = EvidenceRecorder(meta)
        row = records()[0]; row["task_values"] = {"lateral_error_m": [.1, -.2]}
        recorder.record(row); row["task_values"]["lateral_error_m"][0] = 9
        recorder.record(records()[1])
        with self.assertRaises(ValueError): recorder.record(records()[1])
        with tempfile.TemporaryDirectory() as directory:
            bundle = recorder.write(directory)
        view = bundle.task_evidence()
        self.assertEqual(view["task_samples"][0]["values"]["lateral_error_m"], [.1, -.2])
        self.assertTrue(view["complete"])
        self.assertEqual(view["contract"]["contract_id"], "test-v1")
        self.assertIn("not establish continuous physics", view["sampling"]["resolution"])
        self.assertEqual(view["field_sources"]["lateral_error_m"]["kind"], "measurement")

    def test_streaming_recorder_flushes_and_never_claims_live_capture_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvidenceRecorder(metadata(), trial_dir=directory)
            for row in records(): recorder.record(row)
            provisional = load_evidence(directory)
            self.assertEqual(len(provisional.records), 4)
            self.assertFalse(provisional.metadata["complete"])
            self.assertFalse(provisional.task_evidence()["complete"])
            final = recorder.close(complete=True)
            self.assertTrue(final.metadata["complete"])
            with self.assertRaises(ValueError): recorder.record(records()[-1])
            with self.assertRaises(FileExistsError): EvidenceRecorder(metadata(), trial_dir=directory)
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceRecorder(metadata(), trial_dir=directory) as recorder:
                recorder.record(records()[0])
            self.assertFalse(load_evidence(directory).metadata["complete"])


class ContactQualityTests(unittest.TestCase):
    def test_target_duration_expected_interruption_and_native_final_peak(self):
        result = assess_contact_quality(validate_evidence(metadata(), records()))
        pair = result["contact_pairs"]["hold"]
        self.assertAlmostEqual(pair["correct_target_contact"]["observed_duration_s"], .2)
        interruption = pair["expected_contact_interruptions"]
        self.assertAlmostEqual(interruption["observed_interruption_duration_s"], .1)
        self.assertEqual(interruption["criterion_status"], "fail")
        self.assertEqual(len(interruption["episodes"]), 1)
        load = pair["normal_load"]
        self.assertEqual(load["sampled_peak"], 9)
        self.assertAlmostEqual(load["duration_weighted_mean"], 5/3)
        self.assertAlmostEqual(load["overload"]["observed_duration_s"], .1)
        self.assertAlmostEqual(load["overload"]["excess_impulse_ns"], .05)
        json.dumps(result, allow_nan=False)

    def test_gaps_and_missing_contact_cannot_be_passed_or_bridged(self):
        rows = records((0, .1, .2, 1, 1.1), (0, 0, 0, 0, 0))
        rows[1]["contacts"]["hold"]["valid"] = False
        bundle = validate_evidence(metadata(), rows)
        self.assertFalse(bundle.task_evidence()["complete"])
        pair = assess_contact_quality(bundle)["contact_pairs"]["hold"]
        self.assertAlmostEqual(pair["contact"]["known_duration_s"], .2)
        self.assertAlmostEqual(pair["contact"]["unknown_duration_s"], .9)
        interruption = pair["expected_contact_interruptions"]
        self.assertEqual(interruption["status"], "partial")
        self.assertAlmostEqual(interruption["unknown_expected_duration_s"], .1)
        self.assertEqual(len(interruption["episodes"]), 2)

    def test_absent_expectation_and_undeclared_safety_remain_unknown(self):
        meta = metadata(); meta["contact_pairs"]["hold"]["thresholds"] = {"contact_on_n": .05}
        rows = records()
        for row in rows: row["contacts"]["hold"].pop("expected_contact")
        pair = assess_contact_quality(validate_evidence(meta, rows))["contact_pairs"]["hold"]
        self.assertEqual(pair["expected_contact_interruptions"]["status"], "missing")
        self.assertIsNone(pair["expected_contact_interruptions"]["observed_interruption_duration_s"])
        self.assertEqual(pair["normal_load"]["overload"]["status"], "unknown")
        self.assertEqual(pair["independently_labeled_slip"]["status"], "missing")
        self.assertEqual(pair["relative_motion_supported_slip"]["status"], "missing")

    def test_controller_expectation_ignored_and_partial_expectation_not_pass(self):
        rows = records(forces=(2, 2, 2, 2))
        for row in rows: row["contacts"]["hold"]["expected_contact"] = label(True, independent=False)
        pair = assess_contact_quality(validate_evidence(metadata(), rows))["contact_pairs"]["hold"]
        self.assertEqual(pair["expected_contact_interruptions"]["criterion_status"], "unknown")
        rows[0]["contacts"]["hold"]["expected_contact"] = label(True)
        pair = assess_contact_quality(validate_evidence(metadata(), rows))["contact_pairs"]["hold"]
        self.assertEqual(pair["expected_contact_interruptions"]["status"], "partial")
        self.assertEqual(pair["expected_contact_interruptions"]["criterion_status"], "unknown")

    def test_allowed_sliding_is_not_relative_motion_slip(self):
        rows = records(forces=(2, 2, 2, 2))
        for row in rows:
            row["contacts"]["hold"].update(tangential_relative_velocity_m_s=[.3, .4], tangent_force_n=[3, 4])
        meta = metadata()
        pair = assess_contact_quality(validate_evidence(meta, rows))["contact_pairs"]["hold"]
        self.assertAlmostEqual(pair["relative_motion_supported_slip"]["observed_duration_s"], .3)
        self.assertEqual(pair["tangent_load"]["sampled_peak"], 5)
        self.assertEqual(pair["independently_labeled_slip"]["status"], "missing")
        meta["contact_pairs"]["hold"]["allowed_relative_motion"] = "sliding"
        pair = assess_contact_quality(validate_evidence(meta, rows))["contact_pairs"]["hold"]
        self.assertEqual(pair["relative_motion_supported_slip"]["status"], "not_applicable")
        self.assertEqual(pair["tangential_relative_speed"]["sampled_peak_m_s"], .5)

    def test_saturation_excludes_load_but_preserves_explicit_contact(self):
        rows = records()
        rows[0]["contacts"]["hold"].update(saturated=True, contact_present=True)
        pair = assess_contact_quality(validate_evidence(metadata(), rows))["contact_pairs"]["hold"]
        self.assertAlmostEqual(pair["normal_load"]["known_duration_s"], .2)
        self.assertAlmostEqual(pair["contact"]["known_duration_s"], .3)
        self.assertAlmostEqual(pair["saturation"]["observed_duration_s"], .1)

    def test_effects_observed_progress_not_causal_quality_and_gaps_not_filled(self):
        meta = metadata(); meta["effects"] = {"progress": {"source": "independent fixture", "unit": "fraction", "independent": True,
            "target_min": .9, "contact_pairs": ["hold"]}}
        rows = records((0, .1, 1, 1.1), (2, 2, 2, 2))
        for row, value in zip(rows, (.1, .2, .7, .8)): row["effects"] = {"progress": value}
        effect = assess_contact_quality(validate_evidence(meta, rows))["effects"]["progress"]
        self.assertAlmostEqual(effect["contiguous_observed_change"], .2)
        self.assertEqual(effect["target_status"], "fail")
        self.assertIn("not causal", effect["semantics"])

    def test_no_input_mutation_and_target_role_separation(self):
        meta, rows = metadata(), records()
        meta["contact_pairs"]["hold"]["target"] = False
        before_meta, before_rows = deepcopy(meta), deepcopy(rows)
        result = assess_contact_quality(validate_evidence(meta, rows))
        self.assertEqual(meta, before_meta); self.assertEqual(rows, before_rows)
        pair = result["contact_pairs"]["hold"]
        self.assertEqual(pair["correct_target_contact"]["status"], "not_applicable")
        self.assertAlmostEqual(pair["non_target_contact"]["observed_duration_s"], .2)

    def test_legacy_whiteboard_adapter_requires_explicit_contact_threshold(self):
        source = SimpleNamespace(task="whiteboard-wipe", task_tactile_clock_aligned=True,
            control_rows=[{"sim_time": i/10, "task": {"board_normal_force_n": value, "cleaned_fraction": i/10, "stage": "wipe"}}
                          for i, value in enumerate((0, 2, 3))], provenance=[])
        unspecified = assess_contact_quality(adapt_legacy_evidence(source))
        self.assertEqual(unspecified["contact_pairs"]["eraser_board"]["contact"]["status"], "missing")
        result = assess_contact_quality(adapt_legacy_evidence(source, contact_threshold_n=.05))
        pair = result["contact_pairs"]["eraser_board"]
        self.assertAlmostEqual(pair["correct_target_contact"]["observed_duration_s"], .1)
        self.assertEqual(pair["normal_load"]["overload"]["status"], "unknown")
        self.assertEqual(pair["expected_contact_interruptions"]["status"], "missing")
        self.assertEqual(pair["relative_motion_supported_slip"]["status"], "not_applicable")
        self.assertEqual(result["source_kind"], "legacy_simulation_record")
        source.task = "install-ram"
        self.assertEqual(adapt_legacy_evidence(source).status, "missing")


if __name__ == "__main__":
    unittest.main()
