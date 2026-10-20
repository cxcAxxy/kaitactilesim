"""Prediction horizon filtering must never rely on execution or naming hints."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v2.horizon import filter_prediction_horizon, inspect_prediction_horizon


class PredictionHorizonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def trial(self, name="seed_000", **summary):
        root = self.root / "experiment" / "evaluation" / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return root

    def bind(self, trial, manifest, **summary_changes):
        directory = trial.parent / "deployment_snapshot"
        directory.mkdir(exist_ok=True)
        raw = json.dumps(manifest).encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = directory / f"deployment_manifest.{digest}.json"
        path.write_bytes(raw)
        summary = json.loads((trial / "summary.json").read_text())
        summary.update(deployment_manifest=str(path), deployment_integrity={"status": "verified", "manifest": {
            "path": str(path), "expected_sha256": digest, "actual_sha256": digest}})
        summary.update(summary_changes)
        (trial / "summary.json").write_text(json.dumps(summary))
        return path

    def test_summary_explicit_integer_is_sufficient_without_manifest(self):
        root = self.trial(prediction_horizon=50, execute_steps=8, control_hz=30)
        result = inspect_prediction_horizon(root)
        self.assertEqual((result["status"], result["prediction_horizon"]), ("verified", 50))
        self.assertEqual(result["provenance"][0]["sha256"], hashlib.sha256((root / "summary.json").read_bytes()).hexdigest())

    def test_execution_frequency_and_h50_names_cannot_establish_horizon(self):
        root = self.trial("seed_h50", execute_steps=50, control_hz=30,
                          action_horizon=50, model_config_name="pi05_h50")
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["prediction_horizon"])
        self.assertEqual(result["evidence"], [])

    def test_manifest_can_supply_missing_summary_declaration(self):
        root = self.trial(execute_steps=50)
        self.bind(root, {"prediction_horizon": 50})
        result = inspect_prediction_horizon(root)
        self.assertEqual((result["status"], result["prediction_horizon"]), ("verified", 50))
        self.assertEqual(len(result["provenance"]), 2)

    def test_declared_prediction_conflict_excludes_even_summary_h50(self):
        root = self.trial(prediction_horizon=50)
        self.bind(root, {"prediction_horizon": 30})
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "conflict")
        self.assertIsNone(result["prediction_horizon"])
        self.assertEqual({row["value"] for row in result["evidence"]}, {30, 50})

    def test_invalid_explicit_declarations_cannot_fall_back_to_manifest(self):
        for index, value in enumerate([None, True, "50", 50.0, 0, -50, [], {}]):
            with self.subTest(value=value):
                root = self.trial(f"seed_{index}", prediction_horizon=value)
                self.bind(root, {"prediction_horizon": 50})
                self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_bad_manifest_horizon_invalidates_summary(self):
        root = self.trial(prediction_horizon=50)
        self.bind(root, {"prediction_horizon": "50"})
        self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_replaced_manifest_does_not_bypass_recorded_hash(self):
        root = self.trial(prediction_horizon=50)
        path = self.bind(root, {"prediction_horizon": 50})
        path.write_text('{"prediction_horizon": 30}')
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("SHA256 mismatch" in text for text in result["warnings"]))

    def test_binding_identity_and_integrity_failures_are_invalid(self):
        root = self.trial(prediction_horizon=50, deployment_id="trial-deployment")
        self.bind(root, {"prediction_horizon": 50, "deployment_id": "different-deployment"})
        self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")
        for changes in [{"deployment_integrity": None}, {"deployment_integrity": {"manifest": []}},
                        {"deployment_integrity": {"status": "failed"}},
                        {"deployment_manifest_sha256": "not-a-sha256"}]:
            with self.subTest(changes=changes):
                self.bind(root, {"prediction_horizon": 50}, **changes)
                self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_manifest_must_be_bound_no_directory_discovery(self):
        root = self.trial()
        (root / "deployment_manifest.json").write_text('{"prediction_horizon": 50}')
        self.assertEqual(inspect_prediction_horizon(root)["status"], "unknown")

    def test_external_manifest_is_not_read_but_direct_summary_remains_evidence(self):
        root = self.trial(prediction_horizon=50, deployment_manifest="/etc/deployment_manifest.json")
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["provenance"]), 1)
        self.assertTrue(any("bounded experiment" in text for text in result["warnings"]))

    def test_symlink_manifest_is_not_read(self):
        root = self.trial(prediction_horizon=50, deployment_manifest="deployment_manifest.json")
        target = self.root / "unrelated.json"
        target.write_text('{"prediction_horizon": 50}')
        (root / "deployment_manifest.json").symlink_to(target)
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["provenance"]), 1)

    def test_summary_symlink_is_rejected(self):
        root = self.trial()
        summary = root / "summary.json"
        summary.unlink()
        target = self.root / "unrelated.json"
        target.write_text('{"prediction_horizon": 50}')
        summary.symlink_to(target)
        self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_relative_manifest_path_and_legacy_hash_provenance(self):
        root = self.trial(prediction_horizon=30, deployment_manifest="../deployment/deployment_manifest.json")
        directory = root.parent / "deployment"
        directory.mkdir()
        (directory / "deployment_manifest.json").write_text('{"prediction_horizon": 30}')
        result = inspect_prediction_horizon(root)
        self.assertEqual((result["status"], result["prediction_horizon"]), ("verified", 30))
        self.assertEqual(len(result["provenance"]), 2)

    def test_missing_manifest_preserves_explicit_summary_metadata_only(self):
        root = self.trial(prediction_horizon=50, deployment_manifest="deployment_manifest.json")
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["prediction_horizon"], 50)
        self.assertTrue(any("unavailable and not verified" in text for text in result["warnings"]))

    def test_missing_manifest_cannot_establish_horizon_without_summary_declaration(self):
        root = self.trial(deployment_manifest="deployment_manifest.json", deployment_manifest_sha256="0" * 64)
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["prediction_horizon"])

    def test_missing_manifest_hash_does_not_claim_cryptographic_verification(self):
        root = self.trial(prediction_horizon=50, deployment_manifest="deployment_manifest.json",
                          deployment_manifest_sha256="0" * 64)
        result = inspect_prediction_horizon(root)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["provenance"]), 1)
        self.assertTrue(any("unavailable and not verified" in text for text in result["warnings"]))

    def test_unavailable_manifest_cannot_hide_recorded_integrity_failure(self):
        for path in ("deployment_manifest.json", "/etc/deployment_manifest.json"):
            for integrity in ({"status": "failed"}, {"status": "verified", "errors": ["sha mismatch"]}):
                with self.subTest(path=path, integrity=integrity):
                    root = self.trial(prediction_horizon=50, deployment_manifest=path,
                                      deployment_integrity=integrity)
                    result = inspect_prediction_horizon(root)
                    self.assertEqual(result["status"], "invalid")
                    self.assertIsNone(result["prediction_horizon"])
                    included, audit = filter_prediction_horizon([root])
                    self.assertEqual(included, [])
                    self.assertEqual(audit["excluded"][0]["reason"], "prediction_horizon_invalid")

    def test_unavailable_manifest_cannot_hide_invalid_integrity_shape(self):
        for integrity in (None, [], {"manifest": []}, {"errors": None}, {"errors": ""},
                          {"errors": {}}, {"status": None}):
            with self.subTest(integrity=integrity):
                root = self.trial(prediction_horizon=50, deployment_manifest="/etc/deployment_manifest.json",
                                  deployment_integrity=integrity)
                self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_unavailable_manifest_cannot_hide_contradictory_hashes(self):
        for path in ("deployment_manifest.json", "/etc/deployment_manifest.json"):
            for hashes in ({"expected_sha256": "a" * 64, "actual_sha256": "b" * 64},
                           {"expected_sha256": "malformed"}):
                with self.subTest(path=path, hashes=hashes):
                    root = self.trial(prediction_horizon=50, deployment_manifest=path,
                                      deployment_integrity={"status": "verified", "manifest": hashes})
                    self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")
        root = self.trial(prediction_horizon=50, deployment_manifest="/etc/deployment_manifest.json",
                          deployment_manifest_sha256="a" * 64,
                          deployment_integrity={"manifest": {"expected_sha256": "b" * 64}})
        self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_unavailable_manifest_cannot_hide_contradictory_or_malformed_paths(self):
        for path in ("deployment_manifest.json", None, [], ""):
            with self.subTest(path=path):
                root = self.trial(prediction_horizon=50, deployment_manifest="/etc/deployment_manifest.json",
                                  deployment_integrity={"manifest": {"path": path}})
                self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_recorded_integrity_failure_without_manifest_is_invalid(self):
        root = self.trial(prediction_horizon=50, deployment_integrity={"status": "failed"})
        self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_consistent_integrity_with_unavailable_manifest_retains_summary_h50(self):
        for path in ("deployment_manifest.json", "/etc/deployment_manifest.json"):
            with self.subTest(path=path):
                root = self.trial(prediction_horizon=50, deployment_manifest=path,
                                  deployment_integrity={"status": "verified", "errors": [], "manifest": {
                                      "path": path, "expected_sha256": "a" * 64, "actual_sha256": "A" * 64}})
                result = inspect_prediction_horizon(root)
                self.assertEqual((result["status"], result["prediction_horizon"]), ("verified", 50))
                self.assertEqual(len(result["provenance"]), 1)
                self.assertTrue(any("unavailable and not verified" in text for text in result["warnings"]))

    def test_bad_json_duplicate_keys_and_nonobjects_rejected(self):
        root = self.trial()
        for value in ['{"prediction_horizon":', '[]', '{"prediction_horizon": 30, "prediction_horizon": 50}',
                      '{"prediction_horizon": NaN}', '{"prediction_horizon": 1e309}']:
            with self.subTest(value=value):
                (root / "summary.json").write_text(value)
                result = inspect_prediction_horizon(root)
                self.assertEqual(result["status"], "invalid")
                json.dumps(result, allow_nan=False)

    def test_metadata_reads_are_bounded(self):
        root = self.trial(prediction_horizon=50)
        with patch("evaluation_v2.horizon.MAX_METADATA_BYTES", 8):
            self.assertEqual(inspect_prediction_horizon(root)["status"], "invalid")

    def test_filter_audits_every_input_and_reads_no_npz(self):
        roots = [self.trial("seed_h50", prediction_horizon=50), self.trial("seed_h30", prediction_horizon=30),
                 self.trial("seed_unknown", execute_steps=50)]
        for root in roots:
            (root / "evaluation_rollout_trace.npz").write_bytes(b"deliberately invalid archive")
        included, audit = filter_prediction_horizon(roots)
        self.assertEqual(included, roots[:1])
        self.assertEqual((audit["required_prediction_horizon"], audit["input_count"], audit["included_count"]), (50, 3, 1))
        self.assertEqual(len(audit["excluded"]), 2)
        self.assertEqual(len(audit["provenance"]), 3)
        self.assertTrue(all(set(row) == {"source", "inspection", "reason"} for row in audit["included"] + audit["excluded"]))
        json.dumps(audit, allow_nan=False)

    def test_missing_summary_is_unknown_and_invalid_requirement_rejected(self):
        self.assertEqual(inspect_prediction_horizon(self.root / "absent")["status"], "unknown")
        for value in [True, None, "50", 0, 50.0]:
            with self.assertRaises(ValueError):
                filter_prediction_horizon([], required=value)


if __name__ == "__main__":
    unittest.main()
