"""Temporary fixtures only; never load models, launch services, or run simulation."""
import copy
import errno
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate/evaluate_pi05_stage_batch.py"
SPEC = importlib.util.spec_from_file_location("pi05_stage_batch_test", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class StageBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "sim"
        self.output = self.root / "batches/new"
        self.data_patch = patch.object(batch, "DATA_ROOT", self.data)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.plan = batch.build_plan(self.output, allow_transfer=True)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def checkpoint_metadata(self):
        for group in self.plan["groups"]:
            for name in ("_CHECKPOINT_METADATA", "params/_METADATA", "assets/training_provenance.json",
                         "assets/conversion_manifest.snapshot.json"):
                self.write(Path(group["checkpoint"]) / name, {})

    def dryrun_response(self, command, **kwargs):
        self.assertEqual(kwargs["env"]["JAX_PLATFORMS"], "cpu")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("--dry-run", command)
        checkpoint = command[command.index("--checkpoint") + 1]
        group = next(row for row in self.plan["groups"] if row["checkpoint"] == checkpoint)
        transfer = None
        if group["task"] == "usb-unplug":
            if batch.TRANSFER_FLAG not in command:
                return SimpleNamespace(returncode=2, stdout="", stderr="Frozen prompt/protocol differs; explicit opt-in required")
            transfer = dict(training_completion_protocol="usb-unplug-return-release-v1",
                            evaluation_completion_protocol=batch.USB_EVALUATION_PROTOCOL,
                            diagnostic_only=True, formal_metrics_valid=False)
        binding = dict(status="bound", record=dict(config_name="pi05_kaihand_h50",
                       training_action_transform=dict(action_horizon=50),
                       action_contract=dict(evaluation_execution=dict(physical_substep_interpolation_required="linear"))))
        if transfer:
            binding["completion_protocol_transfer"] = transfer
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(dict(valid=True, task=group["task"],
                               prediction_horizon=50, training_provenance=binding)))

    def test_exact_18_checkpoints_90_seeds_no_h30_and_usb_alias(self):
        self.assertEqual(len(self.plan["groups"]), 18)
        self.assertEqual({row["step"] for row in self.plan["groups"]}, {10000, 15000})
        self.assertEqual(self.plan["total_trials"], 90)
        self.assertEqual(self.plan["settings"]["num_trials"], 5)
        for group in self.plan["groups"]:
            if group["task"] == "usb-insert":
                self.assertIn("/usb_insert/pi05/0921_200/", group["checkpoint"])
            if group["task"] == "bulb-screw":
                self.assertIn("pi05_kaihand_bulb_controller_target_h50_v1", group["checkpoint"])
        self.assertFalse(self.output.exists())

    def test_forbidden_scope_variations(self):
        for kwargs in (dict(steps=(20000, 24999)), dict(num_trials=20), dict(gpu="0,1"), dict(port=80)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                batch.build_plan(self.output, **kwargs)

    def test_cpu_preflight_18_contracts_writes_nothing(self):
        self.checkpoint_metadata()
        before = sorted(path.relative_to(self.root) for path in self.root.rglob("*"))
        with patch.object(batch.subprocess, "run", side_effect=self.dryrun_response) as run:
            report = batch.preflight(self.plan)
        self.assertTrue(report["all_selected_valid"])
        self.assertEqual(run.call_count, 18)
        self.assertFalse(report["model_weights_loaded"])
        self.assertFalse(report["outputs_written"])
        self.assertEqual(before, sorted(path.relative_to(self.root) for path in self.root.rglob("*")))
        usb = next(row for row in report["checkpoints"] if row["task"] == "usb-unplug")
        self.assertEqual(usb["training_completion_protocol"], "usb-unplug-return-release-v1")
        self.assertEqual(usb["evaluation_completion_protocol"], batch.USB_EVALUATION_PROTOCOL)

    def test_default_without_usb_optin_rejected_and_no_directory_created(self):
        self.checkpoint_metadata()
        self.plan["allow_usb_unplug_protocol_transfer_diagnostic"] = False
        with patch.object(batch.subprocess, "run", side_effect=self.dryrun_response):
            report = batch.preflight(self.plan)
        self.assertFalse(report["all_selected_valid"])
        self.assertEqual(sum(not row["valid"] for row in report["checkpoints"]), 2)
        with self.assertRaisesRegex(ValueError, "All 18"):
            batch.execute_batch(self.plan, report)
        self.assertFalse(self.output.exists())

    def test_missing_checkpoint_does_not_call_prepare(self):
        with patch.object(batch.subprocess, "run") as run:
            report = batch.preflight(self.plan)
        run.assert_not_called()
        self.assertFalse(report["all_selected_valid"])
        self.assertEqual(report["count"], 18)
        self.assertIn("metadata missing", report["checkpoints"][0]["error"])

    def test_deploy_arguments_managed_full_duration_and_usb_optin(self):
        group = next(row for row in self.plan["groups"] if row["task"] == "usb-unplug")
        prepare, run = batch.deploy_commands(self.plan, group, self.output / "attempt", self.output / "runtime.json")
        self.assertEqual(prepare[-2:], ["--", batch.TRANSFER_FLAG])
        for option, value in (("--num-trials", "5"), ("--video-count", "5"), ("--execute-steps", "50"),
                              ("--max-sim-seconds", "90"), ("--service-mode", "managed"), ("--guard-mode", "observe")):
            self.assertEqual(run[run.index(option) + 1], value)
        self.assertEqual(run[-3:], ["--", "--staged-evaluation-version", "staged_v1"])

    def test_port_probe_matches_server_bind_and_allows_address_reuse(self):
        with patch.object(batch.socket, "socket") as factory:
            probe = factory.return_value.__enter__.return_value
            self.assertTrue(batch.port_free(18980))
        probe.setsockopt.assert_called_once_with(batch.socket.SOL_SOCKET, batch.socket.SO_REUSEADDR, 1)
        probe.bind.assert_called_once_with(("0.0.0.0", 18980))
        probe.listen.assert_called_once_with(1)

    def test_only_address_in_use_is_reported_as_an_occupied_port(self):
        with patch.object(batch.socket, "socket") as factory:
            probe = factory.return_value.__enter__.return_value
            probe.bind.side_effect = OSError(errno.EADDRINUSE, "Address already in use")
            self.assertFalse(batch.port_free(18980))
        for code in (errno.EACCES, errno.EPERM, errno.EADDRNOTAVAIL):
            with self.subTest(errno=code), patch.object(batch.socket, "socket") as factory:
                error = OSError(code, "Port check failed")
                factory.return_value.__enter__.return_value.bind.side_effect = error
                with self.assertRaises(OSError) as caught:
                    batch.port_free(18980)
                self.assertIs(caught.exception, error)

    def group_fixture(self, task="bulb-screw"):
        group = next(row for row in self.plan["groups"] if row["task"] == task)
        attempt = self.output / group["group_dir"] / "attempt_001"
        evaluation = attempt / "run/evaluation"
        deployment = dict(task=task, checkpoint_path=group["checkpoint"], prediction_horizon=50,
                          checkpoint_sha256="a" * 64, deployment_id="bound-id")
        if task == "usb-unplug":
            deployment["training_provenance"] = dict(completion_protocol_transfer=dict(
                training_completion_protocol="usb-unplug-return-release-v1",
                evaluation_completion_protocol=batch.USB_EVALUATION_PROTOCOL))
        self.write(attempt / "deployment/deployment_manifest.json", deployment)
        self.write(attempt / "run/run_manifest.json", dict(evaluation_returncode=0, managed_server_stopped=True,
                                                          evaluation_process_stopped=True))
        self.write(evaluation / "protocol.json", dict(max_sim_seconds=90, guard_mode="observe"))
        definitions = {f"stage{i + 1}": name for i, name in enumerate(batch.STAGES[task])}
        stages = {key: dict(definition=value, count=3, rate=0.6) for key, value in definitions.items()}
        aggregate = dict(version="staged_v1", task=task, valid_staged_trials=5, stages=stages,
                         complete_success_count=None if task.startswith("whiteboard-") else 3,
                         complete_success_rate=None if task.startswith("whiteboard-") else 0.6,
                         completion_protocol=batch.USB_EVALUATION_PROTOCOL if task == "usb-unplug" else None,
                         thresholds=dict(lift_m=0.02))
        rows = []
        for seed in range(5):
            rows.append(dict(seed=seed, valid_trial=True, error=None, record_error=None))
            report = dict(version="staged_v1", task=task, stage_definitions=definitions,
                          stages={key: seed < 3 for key in definitions}, full_duration_evaluation=True,
                          stage_flags={key: seed < 3 for key in definitions}, thresholds=dict(lift_m=0.02),
                          native_task_success_preserved=True, expert_trajectory_used=False, simulation_state_mutated=False)
            detail = dict(task=task, seed=seed, checkpoint_path=group["checkpoint"], prediction_horizon=50,
                          execute_steps=50, checkpoint_sha256="a" * 64, deployment_id="bound-id",
                          sim_seconds=90, staged_evaluation=report, task_success=seed == 0,
                          status="success" if seed == 0 else "task_not_completed", full_duration_evaluation=True)
            trial = evaluation / f"seed_{seed:03d}"
            self.write(trial / "summary.json", detail)
            self.write(trial / "staged_evaluation.json", report)
            self.write(trial / "review.json", dict(model_input_cameras_displayed=["head", "right_wrist"]))
            (trial / "review.mp4").write_bytes(b"nonempty native-validated video fixture")
            (trial / "frames.jsonl").write_text("{}\n")
            with (trial / "staged_states.jsonl").open("w") as stream:
                for i in range(2701):
                    stream.write(json.dumps(dict(version="staged_v1", task=task, seed=seed, prediction_horizon=50,
                                                execute_steps=50, checkpoint_sha256="a" * 64, time_s=i / 30,
                                                checkpoint_path=group["checkpoint"], deployment_id="bound-id", control_tick=i,
                                                stage_definitions=definitions, thresholds=dict(lift_m=0.02),
                                                stages=report["stages"], stage_flags=report["stages"])) + "\n")
        summary = dict(task=task, prediction_horizon=50, execute_steps=50, diagnostic_only=True, formal_metrics_valid=False,
                       planned=5, attempted=5, valid_trials=5, staged_evaluation=aggregate)
        summary["trials" if task in {"usb-insert", "poker-draw"} else "results"] = rows
        self.write(evaluation / "summary.json", summary)
        return group, attempt, summary

    def test_group_stage_denominator_and_native_outcome_are_separate(self):
        group, attempt, _ = self.group_fixture()
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(result["valid_staged_trials"], 5)
        self.assertEqual(result["stages"]["stage1"]["rate"], 0.6)
        self.assertEqual(result["native_task_success_rate"], 0.2)

    def test_shared_usb_poker_all_native_summary_array_formats(self):
        for task in ("bulb-screw", "usb-insert", "poker-draw"):
            with self.subTest(task=task):
                group, attempt, _ = self.group_fixture(task)
                result = batch.summarize_group(self.plan, group, attempt)
                self.assertTrue(result["complete"], result["errors"])
                self.assertEqual(result["valid_staged_trials"], 5)

    def test_89_point_94_seconds_is_not_a_full_rollout(self):
        group, attempt, _ = self.group_fixture()
        path = attempt / "run/evaluation/seed_000/summary.json"
        detail = batch.read_json(path)
        detail["sim_seconds"] = 89.94
        self.write(path, detail)
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertFalse(result["complete"])
        self.assertIn("full 90 seconds", result["errors"][0])

    def test_whiteboard_only_pickup_complete_success_remains_null(self):
        group, attempt, _ = self.group_fixture("whiteboard-wipe-horizontal-line")
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertTrue(result["complete"], result["errors"])
        self.assertEqual(len(result["stages"]), 1)
        self.assertIsNone(result["complete_success_rate"])

    def test_invalid_trial_is_explicit_and_never_a_zero(self):
        group, attempt, summary = self.group_fixture()
        summary["valid_trials"] = 4
        summary["results"][0]["valid_trial"] = False
        summary["staged_evaluation"]["valid_staged_trials"] = 4
        self.write(attempt / "run/evaluation/summary.json", summary)
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertFalse(result["complete"])
        self.assertEqual(result["valid_trials"], 4)
        self.assertEqual(result["valid_staged_trials"], 4)
        self.assertIsNone(result["stages"])
        self.assertIsNone(result["native_task_success_rate"])
        self.assertIn("Not all five trials", result["errors"][0])

    def test_short_or_gapped_logs_and_missing_wrist_video_rejected(self):
        group, attempt, _ = self.group_fixture()
        trial = attempt / "run/evaluation/seed_000"
        self.write(trial / "review.json", dict(model_input_cameras_displayed=["head"]))
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertFalse(result["complete"])
        self.assertIn("head/wrist", result["errors"][0])
        self.write(trial / "review.json", dict(model_input_cameras_displayed=["head", "right_wrist"]))
        (trial / "staged_states.jsonl").write_text("\n")
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["stages"])

    def test_counts_cannot_silently_disagree_with_trial_flags(self):
        group, attempt, summary = self.group_fixture()
        summary["staged_evaluation"]["stages"]["stage1"].update(count=4, rate=0.8)
        self.write(attempt / "run/evaluation/summary.json", summary)
        result = batch.summarize_group(self.plan, group, attempt)
        self.assertFalse(result["complete"])
        self.assertIn("disagree", result["errors"][0])

    def test_missing_groups_visible_csv_empty_rates_not_zero(self):
        self.write(self.output / "batch.json", self.plan)
        self.write(self.output / "status.json", dict(groups=[dict(attempt=0, state="pending") for _ in range(18)]))
        result = batch.summarize_batch(self.output)
        self.assertFalse(result["complete"])
        self.assertEqual(result["complete_groups"], 0)
        self.assertEqual(len(result["groups"]), 18)
        self.assertTrue(all(row["stages"] is None and row["errors"] for row in result["groups"]))
        text = batch.csv_summary(result)
        self.assertEqual(len(text.splitlines()), 19)
        self.assertIn("valid_staged_trials", text.splitlines()[0])
        self.assertNotIn(",0.0,", text)

    def test_existing_outputs_never_overwritten(self):
        self.output.mkdir(parents=True)
        (self.output / "sentinel").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "existing output"):
            batch.execute_batch(self.plan, dict(all_selected_valid=True))
        self.assertEqual((self.output / "sentinel").read_text(), "preserve")

    def test_resume_refuses_mismatched_plan_before_execution(self):
        altered = copy.deepcopy(self.plan)
        altered["settings"]["execute_steps"] = 8
        self.write(self.output / "batch.json", altered)
        with patch.object(batch, "run_logged") as run, self.assertRaisesRegex(ValueError, "Resume plan differs"):
            batch.execute_batch(self.plan, dict(all_selected_valid=True), resume=True)
        run.assert_not_called()

    def test_port_failure_resume_skips_completed_group_and_preserves_logs(self):
        self.write(self.output / "batch.json", self.plan)
        rows = [dict(state="pending", attempt=0) for _ in self.plan["groups"]]
        rows[0] = dict(state="complete", attempt=2)
        self.write(self.output / "status.json", dict(state="failed", groups=rows))
        first_log = self.output / self.plan["groups"][0]["group_dir"] / "attempt_002/evaluation.log"
        first_log.parent.mkdir(parents=True)
        first_log.write_text("completed group: preserve")
        report = dict(all_selected_valid=True)

        with patch.object(batch, "port_free", return_value=False), \
             patch.object(batch, "summarize_group", return_value=dict(complete=True)), \
             patch.object(batch, "run_logged") as run:
            with self.assertRaisesRegex(ValueError, "Service port occupied"):
                batch.execute_batch(self.plan, report, resume=True)
            run.assert_not_called()
        status = batch.read_json(self.output / "status.json")
        self.assertEqual(status["groups"][0]["state"], "complete")
        self.assertEqual(status["groups"][1]["attempt"], 0)
        self.assertEqual(status["groups"][1]["state"], "failed")
        self.assertEqual(first_log.read_text(), "completed group: preserve")

        def command(command, path, env):
            path.write_text("next group fixture")
            return 0

        with patch.object(batch, "port_free", return_value=True), \
             patch.object(batch, "summarize_group", return_value=dict(complete=True, errors=[])), \
             patch.object(batch, "summarize_batch", return_value=dict(complete=True)), \
             patch.object(batch, "run_logged", side_effect=command) as run:
            batch.execute_batch(self.plan, report, resume=True)
        self.assertEqual(run.call_count, 34, "Only the remaining 17 groups may prepare/run")
        self.assertEqual(first_log.read_text(), "completed group: preserve")
        status = batch.read_json(self.output / "status.json")
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["groups"][0]["attempt"], 2)
        self.assertEqual(status["groups"][1]["attempt"], 1)

    def test_run_serial_fresh_and_resume_preserves_old_attempts(self):
        tokenizer = self.root / "paligemma_tokenizer.model"
        tokenizer.write_bytes(b"fixture tokenizer")
        vendor = self.root / "vendor.json"
        vendor.write_text("{}")
        report = dict(all_selected_valid=True)
        call_tasks = []

        def command(command, path, env):
            path.write_text("fixture log")
            operation = command[2]
            if operation == "prepare":
                task = command[command.index("--task") + 1]
                call_tasks.append(task)
                destination = Path(command[command.index("--output-dir") + 1])
                destination.mkdir(parents=True)
            # A first failed attempt stops the batch; no later services start.
            return 1 if len(call_tasks) == 1 and operation == "run" else 0

        with patch.object(batch, "TOKENIZER", tokenizer), patch.object(batch, "EGL_VENDOR", vendor), \
             patch.object(batch, "port_free", return_value=True), patch.object(batch, "run_logged", side_effect=command), \
             patch.object(batch, "summarize_group", return_value=dict(complete=True, errors=[])), \
             patch.object(batch, "summarize_batch", return_value=dict(complete=True)):
            with self.assertRaisesRegex(ValueError, "Evaluation failed"):
                batch.execute_batch(self.plan, report)
            self.assertEqual(len(call_tasks), 1)
            status = batch.read_json(self.output / "status.json")
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["groups"][0]["attempt"], 1)
            failed_log = self.output / self.plan["groups"][0]["group_dir"] / "attempt_001/evaluation.log"
            self.assertEqual(failed_log.read_text(), "fixture log")
            batch.execute_batch(self.plan, report, resume=True)
            self.assertEqual(len(call_tasks), 19)
            self.assertEqual(failed_log.read_text(), "fixture log")
            status = batch.read_json(self.output / "status.json")
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["groups"][0]["attempt"], 2)
            self.assertEqual(status["groups"][1]["attempt"], 1)
            batch.execute_batch(self.plan, report, resume=True)
            self.assertEqual(len(call_tasks), 19, "Complete groups must not start another service")

    def test_default_cli_is_plan_not_execution(self):
        with patch.object(batch, "preflight", return_value=dict(all_selected_valid=True)), \
             patch.object(batch, "execute_batch") as run, patch("builtins.print"):
            self.assertEqual(batch.main(["--output-dir", str(self.output)]), 0)
        run.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
