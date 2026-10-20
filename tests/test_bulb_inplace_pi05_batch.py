"""Small native-artifact fixtures; no models, services, or simulation launched."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate/evaluate_bulb_inplace_pi05_batch.py"
SPEC = importlib.util.spec_from_file_location("inplace_batch_tests", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class BulbInplaceBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "jobs/inplace"
        self.patch = patch.object(batch, "CHECKPOINT_ROOT", self.root / "checkpoints")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.plan = batch.build_plan(self.output)

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def fixture(self, index=0):
        group = self.plan["groups"][index]
        attempt = self.output / group["group_dir"] / "attempt_001"
        evaluation = attempt / "run/evaluation"
        deployment = dict(task=batch.TASK, checkpoint_path=group["checkpoint"], checkpoint_step=group["step"],
                          model_config_name=batch.CONFIG, prediction_horizon=50,
                          checkpoint_sha256="a" * 64, deployment_id="dedicated-id")
        self.write(attempt / "deployment/deployment_manifest.json", deployment)
        self.write(attempt / "run/run_manifest.json", dict(evaluation_returncode=0, managed_server_stopped=True,
                                                          evaluation_process_stopped=True))
        self.write(evaluation / "protocol.json", dict(max_sim_seconds=90, guard_mode="observe",
                                                      action_interpolation="linear", controller=batch.CONTROLLER))
        stages = {key: dict(definition=name, count=0, rate=0.0) for key, name in batch.DEFINITIONS.items()}
        aggregate = dict(version="staged_v1", task=batch.TASK, valid_staged_trials=5, stages=stages,
                         complete_success_count=0, complete_success_rate=0.0, thresholds=dict(seating_hold_s=.2))
        rows = []
        for seed in range(5):
            trial = evaluation / f"seed_{seed:03d}"
            flags = {key: False for key in batch.DEFINITIONS}
            report = dict(version="staged_v1", task=batch.TASK, stage_definitions=batch.DEFINITIONS,
                          stages=flags, stage_flags=flags, complete_success=False, full_duration_evaluation=True,
                          native_task_success_preserved=True, expert_trajectory_used=False, simulation_state_mutated=False,
                          thresholds=aggregate["thresholds"])
            detail = dict(task=batch.TASK, seed=seed, checkpoint_path=group["checkpoint"], prediction_horizon=50,
                          execute_steps=50, checkpoint_sha256="a" * 64, deployment_id="dedicated-id",
                          sim_seconds=90, full_duration_evaluation=True, staged_evaluation=report,
                          action_interpolation="linear", controller=batch.CONTROLLER,
                          task_success=False, status="task_not_completed", evaluation=dict(
                              state=dict(rotation_from_initial_turns=.1, clockwise_turns=.1004),
                              threaded_initialization=dict(initial_clockwise_turns=.0004,
                                  position_randomization_applied=False, physically_validated=True,
                                  initially_threaded_verified=True, initially_seated=False, initially_lit=False)))
            self.write(trial / "summary.json", detail)
            self.write(trial / "staged_evaluation.json", report)
            self.write(trial / "review.json", dict(model_input_cameras_displayed=["head", "right_wrist"]))
            (trial / "review.mp4").write_bytes(b"native-video-fixture")
            (trial / "frames.jsonl").write_text("{}\n")
            with (trial / "staged_states.jsonl").open("w") as stream:
                for tick in range(2701):
                    stream.write(json.dumps(dict(version="staged_v1", task=batch.TASK, seed=seed,
                        prediction_horizon=50, execute_steps=50, checkpoint_path=group["checkpoint"],
                        checkpoint_sha256="a" * 64, deployment_id="dedicated-id", time_s=tick / 30,
                        control_tick=tick, stage_definitions=batch.DEFINITIONS, thresholds=aggregate["thresholds"],
                        stages=flags, stage_flags=flags)) + "\n")
            rows.append(dict(seed=seed, valid_trial=True))
        summary = dict(task=batch.TASK, prediction_horizon=50, execute_steps=50,
                       diagnostic_only=True, formal_metrics_valid=False, planned=5, attempted=5,
                       valid_trials=5, results=rows, staged_evaluation=aggregate)
        self.write(evaluation / "summary.json", summary)
        return group, attempt

    def result(self, group, attempt):
        return batch.summarize_group(self.plan, group, attempt)

    def test_exact_requested_contract_no_outputs(self):
        self.assertEqual(tuple(row["step"] for row in self.plan["groups"]), batch.STEPS)
        self.assertEqual(self.plan["total_trials"], 25)
        self.assertEqual(self.plan["schema"], "kaihand-pi05-bulb-inplace-stage-batch-v1")
        self.assertFalse(self.plan["initial_state_randomization"])
        self.assertEqual(self.plan["trial_scope"], "fixed_initial_state_repeats")
        self.assertFalse(self.output.exists())
        self.assertEqual(len(batch.DEFINITIONS), 4)
        self.assertNotIn("picked_up", batch.DEFINITIONS.values())

    def test_reject_unauthorized_scope(self):
        for kwargs in (dict(steps=(10000, 15000)), dict(num_trials=20), dict(gpu="0,1"), dict(port=80)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                batch.build_plan(self.output, **kwargs)

    def test_native_zero_success_is_complete_and_angles_are_degrees(self):
        group, attempt = self.fixture()
        result = self.result(group, attempt)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["native_task_success_rate"], 0)
        self.assertEqual(result["complete_success_rate"], 0)
        self.assertEqual(result["final_net_angle_deg"]["mean"], 36)

    def test_wrong_interpolation_and_task_are_integrity_errors(self):
        group, attempt = self.fixture()
        path = attempt / "run/evaluation/seed_000/summary.json"
        original = batch.read_json(path)
        for key, value in (("action_interpolation", "zoh"), ("task", "bulb-screw")):
            detail = {**original, key: value}
            self.write(path, detail)
            result = self.result(group, attempt)
            self.assertFalse(result["complete"])
            self.assertEqual(result["failure_kind"], "integrity")

    def test_physical_initial_pose_and_net_angle_are_bound(self):
        group, attempt = self.fixture()
        path = attempt / "run/evaluation/seed_000/summary.json"
        original = batch.read_json(path)
        for key, value in (("position_randomization_applied", True), ("initially_seated", True),
                           ("initial_clockwise_turns", .03)):
            detail = copy.deepcopy(original)
            detail["evaluation"]["threaded_initialization"][key] = value
            self.write(path, detail)
            self.assertFalse(self.result(group, attempt)["complete"])

    def test_success_does_not_justify_short_staged_run(self):
        group, attempt = self.fixture()
        path = attempt / "run/evaluation/seed_000/summary.json"
        detail = batch.read_json(path)
        detail.update(task_success=True, status="success", sim_seconds=30)
        self.write(path, detail)
        result = self.result(group, attempt)
        self.assertFalse(result["complete"])
        self.assertIn("full 90", result["errors"][0])

    def test_missing_artifacts_infrastructure_but_invalid_native_is_not_retryable(self):
        group, attempt = self.fixture()
        self.assertEqual(self.result(group, attempt / "missing")["failure_kind"], "infrastructure")
        path = attempt / "run/evaluation/summary.json"
        summary = batch.read_json(path)
        summary["valid_trials"] = 4
        summary["results"][0]["valid_trial"] = False
        self.write(path, summary)
        result = self.result(group, attempt)
        self.assertEqual(result["failure_kind"], "integrity")
        self.assertIsNone(result["native_task_success_rate"])

    def test_gapped_or_truncated_log_not_admitted(self):
        group, attempt = self.fixture()
        path = attempt / "run/evaluation/seed_000/staged_states.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
        self.assertFalse(self.result(group, attempt)["complete"])

    def test_aggregate_counts_must_equal_trial_flags(self):
        group, attempt = self.fixture()
        path = attempt / "run/evaluation/summary.json"
        summary = batch.read_json(path)
        summary["staged_evaluation"]["stages"]["stage1"].update(count=1, rate=.2)
        self.write(path, summary)
        self.assertIn("Aggregate counts", self.result(group, attempt)["errors"][0])

    def test_preflight_requires_dedicated_normalizer_task_and_controller_actions(self):
        record = dict(task=batch.TASK, config_name=batch.CONFIG,
                      normalizer=dict(asset_id="normalizer_h50", use_quantile_norm=True),
                      action_contract=dict(id="recorded-controller-target-v1"))
        report = dict(checkpoints=[dict(valid=True, training_provenance=dict(record=copy.deepcopy(record))) for _ in range(5)])
        with patch.object(batch._helpers, "preflight", return_value=copy.deepcopy(report)):
            self.assertTrue(batch.preflight(self.plan)["all_selected_valid"])
        report["checkpoints"][0]["training_provenance"]["record"]["config_name"] = "pi05_kaihand_h50"
        with patch.object(batch._helpers, "preflight", return_value=report):
            self.assertFalse(batch.preflight(self.plan)["all_selected_valid"])

    def test_unified_deploy_commands_exact_25_scope(self):
        prepare, run = batch._helpers.deploy_commands(self.plan, self.plan["groups"][0], self.output / "attempt", self.output / "runtime.json")
        self.assertIn("prepare", prepare)
        for flag, value in (("--num-trials", "5"), ("--video-count", "5"), ("--execute-steps", "50"),
                            ("--max-sim-seconds", "90"), ("--guard-mode", "observe")):
            self.assertEqual(run[run.index(flag) + 1], value)
        self.assertEqual(run[-3:], ["--", "--staged-evaluation-version", "staged_v1"])

    def test_existing_output_and_bad_resume_identity_never_overwritten(self):
        self.output.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "existing output"):
            batch.execute_batch(self.plan, dict(all_selected_valid=True))
        changed = copy.deepcopy(self.plan)
        changed["groups"][0]["step"] = 10000
        self.write(self.output / "batch.json", changed)
        with self.assertRaisesRegex(ValueError, "authorized plan"):
            batch.load_owned_plan(self.output)

    def test_resume_skips_all_zero_complete_attempts_without_launching(self):
        self.write(self.output / "batch.json", self.plan)
        self.write(self.output / "status.json", dict(groups=[dict(state="complete", attempt=1) for _ in range(5)]))
        complete = dict(complete=True, errors=[], failure_kind=None)
        with patch.object(batch, "summarize_group", return_value=complete), \
             patch.object(batch, "summarize_batch", return_value=dict(complete=True)), \
             patch.object(batch._helpers, "run_logged") as run:
            result = batch.execute_batch(self.plan, dict(all_selected_valid=True), resume=True)
        self.assertTrue(result["complete"])
        run.assert_not_called()

    def test_resume_refuses_integrity_failed_attempt_before_launch(self):
        self.write(self.output / "batch.json", self.plan)
        self.write(self.output / "status.json", dict(groups=[dict(state="failed", attempt=1) for _ in range(5)]))
        with patch.object(batch, "summarize_group", return_value=dict(complete=False, failure_kind="integrity")), \
             patch.object(batch._helpers, "run_logged") as run:
            with self.assertRaisesRegex(ValueError, "integrity errors"):
                batch.execute_batch(self.plan, dict(all_selected_valid=True), resume=True)
        run.assert_not_called()

    def test_default_plan_never_launches_or_creates_output(self):
        report = dict(all_selected_valid=True, checkpoints=[])
        with patch.object(batch, "preflight", return_value=report), \
             patch.object(batch, "execute_batch") as execute, patch("builtins.print"):
            self.assertEqual(batch.main(["--output-dir", str(self.output)]), 0)
        execute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cpu_helper_cannot_enable_gpu(self):
        env = batch._helpers.environment(self.plan, cpu=True)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["JAX_PLATFORMS"], "cpu")


if __name__ == "__main__":
    unittest.main()
