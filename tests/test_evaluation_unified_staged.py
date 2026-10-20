"""Offline staged import: no simulator, model or large dataset is loaded."""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_unified import staged


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


class StagedImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / "usb-insert/pi05/cohort_staged_v1/seed_000"
        self.trial.mkdir(parents=True)
        self.identity = {"task": "usb-insert", "seed": 0, "prediction_horizon": 50,
                         "checkpoint_sha256": "a" * 64, "deployment_id": "b" * 64}
        self.definitions = {"stage1": "recorded_pickup", "stage2": "recorded_seating"}
        self.rows = [self.snapshot(0.0, 0), self.snapshot(1.0, 30)]
        self.result = {k: v for k, v in self.rows[-1].items() if k != "control_tick"}
        self.result.update(full_duration_evaluation=True, state_log="staged_states.jsonl",
                           state_log_samples=2, legacy_task_success=False,
                           state_log_sampling="recorded control endpoints")
        inventory = [{"path": path, "size": 10, "sha256": character * 64}
                     for path, character in zip(
                         staged._RULE_FILES.values(), ("c", "d"), strict=True
                     )]
        self.manifest = {**self.identity, "evaluation_source_files": inventory,
                         "evaluation_source_sha256": staged._digest(inventory)}
        self.manifest_path = self.trial / "deployment_manifest.json"
        write_json(self.manifest_path, self.manifest)
        self.summary = {**self.identity, "status": "task_not_completed", "stop_reason": "time_limit",
                        "full_duration_evaluation": True, "sim_seconds": 1.0, "max_sim_seconds": 1.0,
                        "task_success": False, "physics_valid": True, "diagnostic_only": True,
                        "formal_metrics_valid": False, "video": {"completed": True},
                        "deployment_manifest": str(self.manifest_path),
                        "deployment_integrity": {"status": "verified", "errors": [],
                            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                            "evaluation_source": {"status": "verified",
                                "expected_sha256": self.manifest["evaluation_source_sha256"],
                                "actual_sha256": self.manifest["evaluation_source_sha256"]}}}
        self.save()

    def snapshot(self, timestamp, tick):
        return {**self.identity, "version": "staged_v1", "stage_count": 2,
                "stage_definitions": self.definitions.copy(),
                "stages": {"stage1": False, "stage2": False},
                "stage_flags": {"stage1": False, "stage2": False},
                "named_stages": {v: False for v in self.definitions.values()},
                "highest_stage": 0, "first_times": {"stage1": None, "stage2": None},
                "success": False, "complete_success": False,
                "thresholds": {"hold_s": 0.2}, "simulation_time_s": timestamp, "time_s": timestamp,
                "control_tick": tick, "time_basis": "mujoco_data_time_seconds",
                "stage_achievement_policy": "ordered_cumulative_ever_reached",
                "measurements": {"simulation_time_s": timestamp, "time_s": timestamp,
                                 "opposition_cosine": None, "contact_force_source": "recorded_solver",
                                 "missing_measurements": ["opposition_cosine"]},
                "continuous_metrics": {}, "observation_gap_count": 0}

    def save(self):
        self.summary["staged_evaluation"] = copy.deepcopy(self.result)
        write_json(self.trial / "summary.json", self.summary)
        write_json(self.trial / "staged_evaluation.json", self.result)
        self.save_log()

    def save_log(self):
        (self.trial / "staged_states.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.rows), encoding="utf-8")

    def rebind_manifest(self):
        self.manifest["evaluation_source_sha256"] = staged._digest(self.manifest["evaluation_source_files"])
        write_json(self.manifest_path, self.manifest)
        integrity = self.summary["deployment_integrity"]
        integrity["manifest_sha256"] = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        integrity["evaluation_source"].update(expected_sha256=self.manifest["evaluation_source_sha256"],
                                               actual_sha256=self.manifest["evaluation_source_sha256"])
        self.save()

    def use_tactile_profile(self, *, horizon=30, family="pi05+cpm"):
        staged._INVENTORY_FILE_CACHE.clear()
        checkpoint = self.root / ("best.pt" if family == "pi05+cpm" else "checkpoint/model.safetensors")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_path = checkpoint if family == "pi05+cpm" else checkpoint.parent
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        runtime_suffixes = [
            *staged._TACTILE_RUNTIME_FILES.values(),
            ("scripts/workcell/cpm_policy_backend.py" if family == "pi05+cpm"
             else "scripts/workcell/ftp1_policy_backend.py"),
        ]
        files = {str(checkpoint): checkpoint_sha}
        for suffix in runtime_suffixes:
            path = self.root / "frozen" / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"" if suffix.endswith("staged_recording.py") else suffix.encode())
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        offset = 0 if family == "pi05+cpm" else 1
        model_dim = 32 if family == "pi05+cpm" else 120
        joint_names = [f"joint_{index}" for index in range(27)]
        manifest = {
            "schema": "poker_tactile_deployment_v1",
            "task": "poker-draw",
            "model_family": family,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": "best" if family == "pi05+cpm" else "10000",
            "prediction_horizon": horizon,
            "action_start_index": offset,
            "execute_steps": 5,
            "model_action_dim": model_dim,
            "action_dim": 27,
            "control_hz": 30,
            "joint_names": joint_names,
            "controller_version": "poker-tactile-direct-joints-v1",
            "required_action_interpolation": "zoh",
            "files": files,
        }
        manifest["deployment_id"] = staged._digest(manifest)
        self.manifest = manifest
        write_json(self.manifest_path, manifest)
        manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        common = {
            "task": "poker-draw",
            "prediction_horizon": horizon,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_path": str(checkpoint_path),
            "deployment_id": manifest["deployment_id"],
            "execute_steps": 5,
        }
        for payload in [*self.rows, self.result]:
            payload.update(common)
        self.summary.update(
            **common,
            model_family=family,
            action_start_index=offset,
            checkpoint_step=manifest["checkpoint_step"],
            model_action_dim=model_dim,
            action_dim=27,
            control_hz=30,
            controller=manifest["controller_version"],
            required_action_interpolation="zoh",
            deployment_manifest_sha256=manifest_sha,
            tactile_sent_to_model=True,
            inference_backend="isolated-local-worker",
            simulation_paused_during_inference=True,
        )
        self.summary["server_metadata"] = {
            key: manifest[key] for key in (
                "task", "model_family", "deployment_id", "checkpoint_path",
                "checkpoint_sha256", "prediction_horizon", "action_start_index",
                "execute_steps", "model_action_dim", "action_dim", "control_hz", "joint_names",
            )
        }
        self.summary["server_metadata"].update(
            deployment_manifest_sha256=manifest_sha,
            tactile_sent_to_model=True,
        )
        self.save()

    def test_failed_native_trial_is_completed_and_retained(self):
        found = staged.discover_staged_trials(self.root)
        self.assertEqual(found["counts"]["included"], 1)
        loaded = staged.load_staged_result(self.trial)
        self.assertFalse(loaded["native_result"]["task_success"])
        self.assertTrue(loaded["completion"]["completed"])
        self.assertEqual(loaded["data_evidence"]["sample_count"], 2)
        self.assertIsNone(loaded["staged_result"]["final_facts"]["opposition_cosine"])
        self.assertNotIn("state_rows", loaded)
        self.assertEqual(len(loaded["provenance"]), 4)
        json.dumps(loaded, allow_nan=False)

    def test_three_layers_do_not_replace_native_label(self):
        for payload in (self.rows[-1], self.result):
            payload.update(stages={"stage1": True, "stage2": True},
                           stage_flags={"stage1": True, "stage2": True},
                           named_stages={v: True for v in self.definitions.values()},
                           first_times={"stage1": 1.0, "stage2": 1.0}, highest_stage=2,
                           success=True, complete_success=True)
        self.save()
        result = staged.load_staged_result(self.trial, include_state_log=True)
        self.assertFalse(result["native_result"]["task_success"])
        self.assertTrue(result["staged_result"]["complete_success"])
        self.assertEqual([row["time_s"] for row in result["state_rows"]], [0.0, 1.0])

    def test_whiteboard_undefined_completion_remains_null(self):
        for row in [*self.rows, self.result]:
            row.update(success=None, complete_success=None, whiteboard_complete_success_defined=False)
        self.save()
        self.assertIsNone(staged.load_staged_result(self.trial)["staged_result"]["complete_success"])

    def test_non_h50_and_absent_declaration_excluded(self):
        for declaration in (30, None, True, "50"):
            with self.subTest(declaration=declaration):
                self.summary["prediction_horizon"] = declaration
                self.save()
                found = staged.discover_staged_trials(self.root)
                self.assertEqual(found["counts"]["included"], 0)
        self.assertIn("prediction_horizon", found["excluded"][0]["reason"])

    def test_explicit_h30_and_h32_import_without_changing_h50_default(self):
        for horizon in (30, 32):
            with self.subTest(horizon=horizon):
                self.summary["prediction_horizon"] = horizon
                self.result["prediction_horizon"] = horizon
                self.manifest["prediction_horizon"] = horizon
                for row in self.rows:
                    row["prediction_horizon"] = horizon
                self.rebind_manifest()
                loaded = staged.load_staged_result(
                    self.trial, required_prediction_horizon=horizon
                )
                self.assertEqual(loaded["identity"]["prediction_horizon"], horizon)
                with self.assertRaisesRegex(ValueError, "prediction_horizon=50"):
                    staged.load_staged_result(self.trial)

    def test_invalid_required_horizon_rejected(self):
        for value in (True, 0, -1, 30.0):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                staged.load_staged_result(
                    self.trial, required_prediction_horizon=value
                )

    def test_explicit_tactile_inventory_profile_binds_worker_and_full_inventory(self):
        self.use_tactile_profile()
        loaded = staged.load_staged_result(
            self.trial,
            required_prediction_horizon=30,
            integrity_profile="tactile_manifest_inventory_v1",
        )
        rules = loaded["rule_identity"]
        self.assertEqual(rules["binding_basis"], "tactile_manifest_inventory_v1")
        self.assertEqual(rules["binding"], "tactile_manifest_inventory_v1")
        self.assertEqual(
            rules["observer_sha256"],
            self.manifest["files"][next(
                path for path in self.manifest["files"]
                if path.endswith(staged._RULE_FILES["observer_sha256"])
            )],
        )
        inventory_rows = [row for row in loaded["provenance"]
                          if row["role"] == "tactile_inventory_file"]
        self.assertEqual(len(inventory_rows), len(self.manifest["files"]))
        with self.assertRaisesRegex(ValueError, "Deployment (integrity|manifest SHA)"):
            staged.load_staged_result(self.trial, required_prediction_horizon=30)

    def test_tactile_profile_rejects_handshake_inventory_and_execute_step_tamper(self):
        for mode in ("handshake", "inventory", "execute_steps", "manifest_task", "offset_bool"):
            with self.subTest(mode=mode):
                self.use_tactile_profile()
                if mode == "handshake":
                    self.summary["server_metadata"]["deployment_manifest_sha256"] = "f" * 64
                    self.save()
                    pattern = "SHA/worker handshake"
                elif mode == "inventory":
                    source = next(path for path in self.manifest["files"]
                                  if path.endswith("staged_evaluation.py"))
                    Path(source).write_bytes(b"changed")
                    pattern = "inventory file"
                else:
                    if mode == "execute_steps":
                        self.rows[0]["execute_steps"] = 4
                        self.save_log()
                        pattern = "execute_steps"
                    else:
                        if mode == "manifest_task":
                            self.manifest["task"] = "usb-insert"
                            pattern = "identity mismatch: task"
                        else:
                            self.manifest["action_start_index"] = False
                            pattern = "identity mismatch: action_start_index"
                        self.manifest["deployment_id"] = staged._digest({
                            key: value for key, value in self.manifest.items()
                            if key != "deployment_id"
                        })
                        write_json(self.manifest_path, self.manifest)
                        manifest_sha = hashlib.sha256(
                            self.manifest_path.read_bytes()
                        ).hexdigest()
                        self.summary["deployment_id"] = self.manifest["deployment_id"]
                        self.summary["deployment_manifest_sha256"] = manifest_sha
                        self.summary["server_metadata"]["deployment_id"] = self.manifest["deployment_id"]
                        self.summary["server_metadata"]["deployment_manifest_sha256"] = manifest_sha
                        for payload in [*self.rows, self.result]:
                            payload["deployment_id"] = self.manifest["deployment_id"]
                        self.save()
                with self.assertRaisesRegex(ValueError, pattern):
                    staged.load_staged_result(
                        self.trial,
                        required_prediction_horizon=30,
                        integrity_profile="tactile_manifest_inventory_v1",
                    )

    def test_incomplete_run_excluded_even_with_success_label(self):
        self.summary.update(task_success=True, status="success", sim_seconds=0.5)
        self.result["legacy_task_success"] = True
        self.save()
        found = staged.discover_staged_trials(self.root)
        self.assertEqual(found["excluded"][0]["status"], "not_completed")

    def test_nonstaged_payload_is_not_inferred_from_folder(self):
        self.result["version"] = "other"
        self.save()
        self.assertEqual(staged.discover_staged_trials(self.root)["excluded"][0]["status"], "not_staged")

    def test_discovery_does_not_read_large_log(self):
        (self.trial / "staged_states.jsonl").unlink()
        self.assertEqual(staged.discover_staged_trials(self.root)["counts"]["included"], 1)
        with self.assertRaises(OSError):
            staged.load_staged_result(self.trial)

    def test_manifest_sha_tamper_rejected(self):
        self.manifest_path.write_text(self.manifest_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            staged.load_staged_result(self.trial)

    def test_recorded_manifest_failure_is_not_ignored(self):
        self.summary["deployment_integrity"]["manifest"] = {"status": "failed", "errors": ["bad binding"]}
        self.save()
        with self.assertRaisesRegex(ValueError, "recorded error"):
            staged.load_staged_result(self.trial)

    def test_integer_flags_are_not_booleans(self):
        self.result["stage_flags"]["stage1"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "stage_flags"):
            staged.load_staged_result(self.trial)

    def test_sidecar_mismatch_rejected(self):
        modified = copy.deepcopy(self.result)
        modified["thresholds"]["hold_s"] = 9
        write_json(self.trial / "staged_evaluation.json", modified)
        with self.assertRaisesRegex(ValueError, "differs"):
            staged.load_staged_result(self.trial)

    def test_same_version_different_archived_observer_has_new_identity(self):
        before = staged.load_staged_result(self.trial)["rule_identity"]
        self.manifest["evaluation_source_files"][0]["sha256"] = "e" * 64
        self.rebind_manifest()
        after = staged.load_staged_result(self.trial)["rule_identity"]
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["definitions_thresholds_sha256"], after["definitions_thresholds_sha256"])

    def set_protocol(self, protocol):
        for payload in [*self.rows, self.result]:
            payload["completion_protocol"] = protocol
        self.save()

    def test_old_archive_protocol_absence_and_rule_hash_remain_unchanged(self):
        loaded = staged.load_staged_result(self.trial, include_state_log=True)
        self.assertNotIn("completion_protocol", loaded["staged_result"])
        self.assertNotIn("completion_protocol", loaded["rule_identity"])
        self.assertTrue(all("completion_protocol" not in row for row in loaded["state_rows"]))
        original_contract = {key: self.result.get(key) for key in (
            "version", "task", "stage_definitions", "thresholds", "stage_achievement_policy", "time_basis")}
        self.assertEqual(loaded["rule_identity"]["definitions_thresholds_sha256"], staged._digest(original_contract))

    def test_recorded_protocol_is_preserved_and_bound_without_current_rule_lookup(self):
        original = staged.load_staged_result(self.trial)["rule_identity"]
        previous = original
        for protocol in ("usb-unplug-held-clear-2s-v2", "unknown-future-protocol", None):
            with self.subTest(protocol=protocol):
                self.set_protocol(protocol)
                loaded = staged.load_staged_result(self.trial, include_state_log=True)
                identity = loaded["rule_identity"]
                self.assertIn("completion_protocol", loaded["staged_result"])
                self.assertEqual(loaded["staged_result"]["completion_protocol"], protocol)
                self.assertEqual(identity["completion_protocol"], protocol)
                self.assertEqual([row["completion_protocol"] for row in loaded["state_rows"]], [protocol, protocol])
                self.assertNotEqual(identity["sha256"], previous["sha256"])
                self.assertNotEqual(identity["definitions_thresholds_sha256"], previous["definitions_thresholds_sha256"])
                self.assertEqual(identity["evaluation_source_sha256"], original["evaluation_source_sha256"])
                self.assertFalse(identity["current_rules_reexecuted"])
                previous = identity

    def test_log_protocol_drift_disappearance_or_addition_is_rejected(self):
        protocol = "usb-unplug-held-clear-2s-v2"
        for mode in ("changed", "missing", "null", "added", "added-null"):
            with self.subTest(mode=mode):
                self.set_protocol(protocol)
                if mode == "changed":
                    self.rows[0]["completion_protocol"] = "different-protocol"
                elif mode == "missing":
                    self.rows[0].pop("completion_protocol")
                elif mode == "null":
                    self.rows[0]["completion_protocol"] = None
                else:
                    for payload in [*self.rows, self.result]:
                        payload.pop("completion_protocol")
                    self.save()
                    self.rows[0]["completion_protocol"] = protocol if mode == "added" else None
                self.save_log()
                with self.assertRaisesRegex(ValueError, "contract changed: completion_protocol"):
                    staged.load_staged_result(self.trial)

    def test_malformed_recorded_completion_protocol_fails_closed(self):
        for protocol in (False, 2, "", []):
            with self.subTest(protocol=protocol):
                self.set_protocol(protocol)
                with self.assertRaisesRegex(ValueError, "completion_protocol"):
                    staged.load_staged_result(self.trial)

    def test_log_identity_and_clock_checked(self):
        for key, value in (("seed", 9), ("simulation_time_s", 0.0), ("time_s", 0.8)):
            with self.subTest(key=key):
                original = self.rows[-1][key]
                self.rows[-1][key] = value
                self.save_log()
                with self.assertRaises(ValueError):
                    staged.load_staged_result(self.trial)
                self.rows[-1][key] = original

    def test_log_final_measurement_mismatch_and_truncation_rejected(self):
        self.rows[-1]["measurements"]["opposition_cosine"] = 0.7
        self.save_log()
        with self.assertRaisesRegex(ValueError, "final record"):
            staged.load_staged_result(self.trial)
        self.rows[-1]["measurements"]["opposition_cosine"] = None
        self.save_log()
        path = self.trial / "staged_states.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        with self.assertRaisesRegex(ValueError, "incomplete final"):
            staged.load_staged_result(self.trial)

    def test_count_and_cumulative_flags_checked(self):
        self.result["state_log_samples"] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, "sample count"):
            staged.load_staged_result(self.trial)

    def test_leaf_symlink_rejected(self):
        original = self.trial / "staged_evaluation.json"
        target = self.root / "other.json"
        original.rename(target)
        original.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            staged.load_staged_result(self.trial)

    def test_duplicate_json_key_rejected(self):
        path = self.trial / "summary.json"
        path.write_text(path.read_text().replace('{', '{"task":"other",', 1))
        self.assertIn("Duplicate JSON key", staged.discover_staged_trials(self.root)["excluded"][0]["reason"])

    def test_changed_source_during_import_fails_closed(self):
        original = staged._state_log
        def changed(*args):
            result = original(*args)
            path = self.trial / "summary.json"
            path.write_text(path.read_text() + " ")
            return result
        with patch.object(staged, "_state_log", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                staged.load_staged_result(self.trial)

    def job_copy(self):
        target = self.root / "_jobs/run/groups/usb-insert/checkpoint/run/evaluation/seed_000"
        target.parent.mkdir(parents=True)
        shutil.copytree(self.trial, target)
        return target

    def test_jobs_are_opt_in_and_byte_identical_alias_deduplicated(self):
        alias = self.job_copy()
        self.assertEqual(staged.discover_staged_trials(self.root)["counts"]["candidates"], 1)
        result = staged.discover_staged_trials(self.root, include_jobs=True)
        self.assertEqual(result["counts"], {"candidates": 2, "included": 1, "excluded": 0, "duplicates": 1})
        self.assertEqual(result["included"][0]["source"], str(self.trial))
        self.assertEqual(result["included"][0]["aliases"], [str(alias)])

    def test_conflicting_alias_is_not_chosen_by_success(self):
        alias = self.job_copy()
        changed = copy.deepcopy(self.summary)
        changed["wall_seconds"] = 99
        write_json(alias / "summary.json", changed)
        result = staged.discover_staged_trials(self.root, include_jobs=True)
        self.assertEqual(result["counts"]["included"], 0)
        self.assertEqual([x["status"] for x in result["excluded"]], ["conflict", "conflict"])

    def test_explicit_extra_root_includes_completed_failed_group(self):
        alias = self.job_copy()
        group = alias.parents[2]
        write_json(group / "group_status.json", {"status": "failed", "error": "post-run batch finalizer"})
        # Only the explicit jobs root is selected here; the other root has no canonical cohorts.
        empty_root = self.root / "empty"
        empty_root.mkdir()
        result = staged.discover_staged_trials(empty_root, extra_roots=[alias.parent])
        self.assertEqual(result["counts"]["included"], 1)
        selection = result["included"][0]["selection"]
        self.assertEqual(selection["kind"], "explicit_extra_root")
        self.assertEqual(selection["group_status"]["status"], "failed")
        self.assertTrue(result["included"][0]["warnings"])
        self.assertFalse(staged.load_staged_result(alias)["native_result"]["task_success"])


if __name__ == "__main__":
    unittest.main()
