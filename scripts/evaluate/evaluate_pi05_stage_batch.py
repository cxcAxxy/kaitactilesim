#!/usr/bin/env python3
"""Explicit, serial 9-task H50/K50 staged diagnostic evaluation.

Default/--plan performs CPU-only contract preflight, without creating outputs,
hashing model weights, or loading a model. --run is the only execution opt-in.
No canonical archives or previous evaluation directories are overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import errno
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT.parent / "openpi"
MODEL_PYTHON = OPENPI / ".venv-pi05/bin/python"
SIM_PYTHON = ROOT / ".pixi/envs/default/bin/python"
DATA_ROOT = Path("/nas/chenxianchi/datasets/sim")
SCHEMA = "kaihand-pi05-stage-batch-v1"
TRANSFER_FLAG = "--allow-usb-unplug-protocol-transfer-diagnostic"
USB_EVALUATION_PROTOCOL = "usb-unplug-held-clear-2s-v2"
TOKENIZER = Path("/nas/chenxianchi/datasets/sim/_evaluations/pi05_h50_k50_serial_20261013_111821_frozen_r2/.cache/openpi/big_vision/paligemma_tokenizer.model")
EGL_VENDOR = Path("/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json")
# Exact dataset/run identities; never auto-select the newest directory or H30.
RUNS = (
    ("bulb-screw", "1012_200", "pi05_kaihand_bulb_controller_target_h50_v1", "bulb-screw_1012_200_bs128_25k"),
    ("bulb-unscrew", "1010_200", "pi05_kaihand_h50", "bulb_unscrew_tabletop_1010_h50_bs128_25k_fsdp8_20261011_222530"),
    ("install-ram", "0920_200", "pi05_kaihand_h50", "install_ram_0920_h50_bs128_25k"),
    ("poker-draw", "0920_200", "pi05_kaihand_h50", "poker_draw_0920_h50_bs128_25k"),
    ("usb-insert", "0921_200", "pi05_kaihand_h50", "usb_insert_0921_h50_hand_tail_v1_bs128_25k"),
    ("usb-unplug", "1010_200", "pi05_kaihand_h50", "usb_unplug_1010_h50_bs128_25k_fsdp8_20261011_222315"),
    ("whiteboard-wipe", "0929_200", "pi05_kaihand_h50", "whiteboard-wipe_bs128_25k"),
    ("whiteboard-wipe-horizontal", "1010_200", "pi05_kaihand_h50", "whiteboard_wipe_horizontal_1010_h50_bs128_25k_fsdp8_20261011_222102"),
    ("whiteboard-wipe-horizontal-line", "1012_200", "pi05_kaihand_h50", "whiteboard-wipe-horizontal-line_1012_200_bs128_25k"),
)
STAGES = {
    "bulb-screw": ("picked_up", "held_near_socket", "threaded_and_seated"),
    "bulb-unscrew": ("fully_extracted", "released_stably_on_table"),
    "install-ram": ("picked_up_and_held_near_slot", "inserted_and_seated"),
    "poker-draw": ("dragged_on_table", "stable_grasp", "terminal_hold_over_3s"),
    "usb-insert": ("picked_up", "held_near_socket", "inserted_and_seated"),
    "usb-unplug": ("fully_extracted", "extracted_and_held_2s"),
    **{task: ("picked_up",) for task, *_ in RUNS if task.startswith("whiteboard-")},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json(path, value):
    """Atomic updates only inside the explicitly owned fresh batch directory."""
    descriptor, name = tempfile.mkstemp(prefix=".batch-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    Path(name).replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def build_plan(output, *, steps=(10000, 15000), num_trials=5, gpu="0", port=18980,
               allow_transfer=False, egl_device=None):
    require(tuple(steps) == (10000, 15000), "This batch requires exactly --steps 10000 15000")
    require(type(num_trials) is int and num_trials == 5, "This batch requires exactly --num-trials 5")
    require(isinstance(gpu, str) and re.fullmatch(r"0|[1-9][0-9]*", gpu), "--gpu must select one physical GPU index")
    require(type(port) is int and 1024 <= port <= 65535, "--port must be 1024..65535")
    if egl_device is None:
        egl_device = int(gpu)
    require(type(egl_device) is int and egl_device >= 0, "--egl-device must be nonnegative")
    output = Path(output).absolute()
    require(output.name not in {"", ".", ".."} and output != ROOT and len(output.parts) >= 4,
            "Select a dedicated new batch directory")
    require(not output.is_symlink(), "Batch directory cannot be a symlink")
    groups = []
    for task, dataset, config, run in RUNS:
        source_task = "usb_insert" if task == "usb-insert" else task
        for step in steps:
            checkpoint = DATA_ROOT / source_task / "pi05" / dataset / "checkpoints" / config / run / str(step)
            groups.append(dict(task=task, dataset=dataset, step=step, checkpoint=str(checkpoint),
                               group_dir=f"groups/{task}/step{step}"))
    return dict(schema=SCHEMA, output_dir=str(output), simulation_root=str(ROOT), openpi_root=str(OPENPI),
                model_python=str(MODEL_PYTHON), simulation_python=str(SIM_PYTHON),
                diagnostic_only=True, formal_metrics_valid=False,
                settings=dict(prediction_horizon=50, execute_steps=50, num_trials=5, video_count=5,
                              seed_start=0, max_sim_seconds=90, guard_mode="observe",
                              staged_evaluation_version="staged_v1", parallel_groups=1,
                              gpu=gpu, egl_device=egl_device, port=port),
                allow_usb_unplug_protocol_transfer_diagnostic=bool(allow_transfer),
                total_groups=18, total_trials=90, groups=groups)


def environment(plan, *, cpu=False):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(OPENPI / "src"),
                                         str(OPENPI / "packages/openpi-client/src"))),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if cpu:
        # Config/provenance validation only. No CUDA or model restore allowed.
        env.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu", CUDA_VISIBLE_DEVICES="")
    else:
        output = Path(plan["output_dir"])
        env.pop("JAX_PLATFORMS", None)
        env.pop("JAX_PLATFORM_NAME", None)
        env.update(CUDA_VISIBLE_DEVICES=plan["settings"]["gpu"], MUJOCO_GL="egl",
                   MUJOCO_EGL_DEVICE_ID=str(plan["settings"]["egl_device"]),
                   PYOPENGL_PLATFORM="egl", KAIHAND_RENDER_BACKEND="hardware", LIBGL_ALWAYS_SOFTWARE="0",
                   OPENPI_DATA_HOME=str(output / ".cache/openpi"), HF_HOME=str(output / ".cache/huggingface"),
                   MPLCONFIGDIR=str(output / ".cache/matplotlib"),
                   __EGL_VENDOR_LIBRARY_FILENAMES=str(EGL_VENDOR))
    return env


def prepare_command(plan, group, output, *, dry_run=False):
    backend = {"poker-draw": "prepare_poker_pi05_deployment.py",
               "usb-insert": "prepare_usb_pi05_deployment.py"}.get(group["task"], "prepare_shared_task_pi05_deployment.py")
    command = [str(MODEL_PYTHON), str(ROOT / "scripts/workcell" / backend),
               "--checkpoint", group["checkpoint"], "--output-dir", str(output), "--openpi-root", str(OPENPI)]
    if backend == "prepare_shared_task_pi05_deployment.py":
        command += ["--task", group["task"]]
    if group["task"] == "usb-unplug" and plan["allow_usb_unplug_protocol_transfer_diagnostic"]:
        command.append(TRANSFER_FLAG)
    if dry_run:
        command.append("--dry-run")
    return command


def preflight(plan):
    """Read only: backend dry-runs authenticate small metadata, not model shards."""
    rows = []
    for group in plan["groups"]:
        row = {**group, "valid": False, "model_weights_loaded": False, "outputs_written": False}
        try:
            checkpoint = Path(group["checkpoint"])
            for relative in ("_CHECKPOINT_METADATA", "params/_METADATA", "assets/training_provenance.json",
                             "assets/conversion_manifest.snapshot.json"):
                require((checkpoint / relative).is_file(), f"Committed checkpoint metadata missing: {checkpoint / relative}")
            result = subprocess.run(prepare_command(plan, group, Path(plan["output_dir"]) / "preflight-never-created", dry_run=True),
                                    cwd=ROOT, env=environment(plan, cpu=True), capture_output=True, text=True, check=False)
            require(result.returncode == 0, f"CPU prepare preflight rejected checkpoint: {(result.stderr or result.stdout)[-6000:]}")
            report = json.loads(result.stdout)
            require(report.get("valid") is True and report.get("task") == group["task"]
                    and report.get("prediction_horizon") == 50, "Dry-run task/H50 identity mismatch")
            binding = report.get("training_provenance")
            require(isinstance(binding, dict) and binding.get("status") == "bound", "Checkpoint provenance is not bound")
            record = binding["record"]
            transform = record["training_action_transform"]
            execution = record["action_contract"]["evaluation_execution"]
            require(transform.get("action_horizon") == 50 and execution.get("physical_substep_interpolation_required") == "linear",
                    "Checkpoint must bind H50 and physical linear interpolation")
            transfer = binding.get("completion_protocol_transfer")
            if group["task"] == "usb-unplug":
                require(plan["allow_usb_unplug_protocol_transfer_diagnostic"] and isinstance(transfer, dict)
                        and transfer.get("evaluation_completion_protocol") == USB_EVALUATION_PROTOCOL
                        and transfer.get("diagnostic_only") is True and transfer.get("formal_metrics_valid") is False,
                        "Legacy USB checkpoint requires explicit diagnostic protocol transfer")
            row.update(valid=True, model_config_name=record["config_name"], binding_status="bound",
                       prediction_horizon=50, execute_steps=50, required_action_interpolation="linear",
                       training_completion_protocol=transfer.get("training_completion_protocol") if transfer else record.get("completion_protocol"),
                       evaluation_completion_protocol=transfer.get("evaluation_completion_protocol") if transfer else record.get("completion_protocol"),
                       completion_protocol_transfer=transfer,
                       training_provenance=binding)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
        rows.append(row)
    return dict(all_selected_valid=all(row["valid"] for row in rows), count=len(rows), checkpoints=rows,
                model_weights_loaded=False, outputs_written=False)


def port_free(port):
    """Mirror the IPv4 server bind policy without claiming an occupied port."""
    with socket.socket() as probe:
        # asyncio's Unix servers reuse addresses left in TIME_WAIT. Match
        # their wildcard bind, but never enable SO_REUSEPORT or touch owners.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
            probe.listen(1)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def run_logged(command, path, env):
    """Only forward interrupts to our own child process group, never other jobs."""
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()

    with path.open("xb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        prior = signal.signal(signal.SIGTERM, interrupted)
        try:
            return child.wait()
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                # deploy.py owns cleanup of its sidecar/evaluator. Do not kill
                # the manager before it has recorded its cleanup result.
                child.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, prior)


def deploy_commands(plan, group, attempt, runtime):
    common = [str(SIM_PYTHON), str(ROOT / "scripts/deploy/deploy.py")]
    manifest = attempt / "deployment/deployment_manifest.json"
    prepare = common + ["prepare", "--task", group["task"], "--model-family", "pi05",
                        "--checkpoint", group["checkpoint"], "--output-dir", str(manifest.parent),
                        "--runtime-config", str(runtime), "--runtime", "pi05"]
    if group["task"] == "usb-unplug" and plan["allow_usb_unplug_protocol_transfer_diagnostic"]:
        prepare += ["--", TRANSFER_FLAG]
    run = common + ["run", "--deployment-manifest", str(manifest), "--output-dir", str(attempt / "run"),
                    "--runtime-config", str(runtime), "--runtime", "pi05", "--service-mode", "managed",
                    "--endpoint", f"ws://127.0.0.1:{plan['settings']['port']}", "--num-trials", "5",
                    "--seed-start", "0", "--video-count", "5", "--execute-steps", "50",
                    "--max-sim-seconds", "90", "--guard-mode", "observe", "--server-timeout", "900",
                    "--", "--staged-evaluation-version", "staged_v1"]
    return prepare, run


def batch_identity(plan):
    return {key: value for key, value in plan.items() if key not in {"preflight", "created_utc"}}


def execute_batch(plan, report, *, resume=False):
    require(report["all_selected_valid"], "All 18 checkpoints must pass CPU preflight before creating outputs")
    output = Path(plan["output_dir"])
    if resume:
        require(output.is_dir() and not output.is_symlink(), "--resume requires this tool's existing batch directory")
        prior = read_json(output / "batch.json")
        require(batch_identity(prior) == batch_identity(plan), "Resume plan differs from the recorded batch identity/settings")
        status = read_json(output / "status.json")
    else:
        require(not output.exists(), "Refusing existing output directory; choose a fresh path or explicit --resume")
        require(TOKENIZER.is_file(), f"Cached tokenizer missing: {TOKENIZER}")
        require(EGL_VENDOR.is_file(), f"NVIDIA EGL vendor configuration missing: {EGL_VENDOR}")
        require(MODEL_PYTHON.is_file() and SIM_PYTHON.is_file(), "Model/simulation Python missing")
        require(port_free(plan["settings"]["port"]), "Selected loopback service port is occupied")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "batch.json", {**plan, "created_utc": timestamp(), "preflight": report})
        status = dict(schema=SCHEMA, output_dir=str(output), state="starting", groups=[dict(state="pending", attempt=0) for _ in plan["groups"]])
        write_json(output / "status.json", status)
        cache = output / ".cache/openpi/big_vision"
        cache.mkdir(parents=True)
        shutil.copy2(TOKENIZER, cache / TOKENIZER.name)
        env = environment(plan)
        write_json(output / "runtime.json", dict(default_runtime="pi05", runtimes=dict(pi05=dict(
            python=str(MODEL_PYTHON), project_root=str(OPENPI), env={key: env[key] for key in
                ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENPI_DATA_HOME")}))))
    require(len(status["groups"]) == len(plan["groups"]), "Batch status group inventory mismatch")
    with (output / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another controller owns this batch; refusing concurrent resume") from error
        env = environment(plan)
        status.update(state="running", controller_pid=os.getpid(), updated_utc=timestamp())
        write_json(output / "status.json", status)
        for index, group in enumerate(plan["groups"]):
            row = status["groups"][index]
            try:
                if row["state"] == "complete":
                    prior_attempt = output / group["group_dir"] / f"attempt_{row['attempt']:03d}"
                    require(summarize_group(plan, group, prior_attempt)["complete"],
                            "A previously complete group is no longer valid; do not silently reuse or overwrite it")
                    continue
                require(port_free(plan["settings"]["port"]), "Service port occupied; not touching that process")
                row.update(state="running", attempt=row["attempt"] + 1, started_utc=timestamp())
                attempt = output / group["group_dir"] / f"attempt_{row['attempt']:03d}"
                attempt.mkdir(parents=True, exist_ok=False)
                write_json(output / "status.json", status)
                prepare, run = deploy_commands(plan, group, attempt, output / "runtime.json")
                require(run_logged(prepare, attempt / "prepare.log", env) == 0, f"Prepare/verify failed; see {attempt / 'prepare.log'}")
                require(run_logged(run, attempt / "evaluation.log", env) == 0, f"Evaluation failed; see {attempt / 'evaluation.log'}")
                validation = summarize_group(plan, group, attempt)
                write_json(attempt / "validation.json", validation)
                require(validation["complete"], f"Invalid/incomplete group artifacts: {validation['errors']}")
                row.update(state="complete", finished_utc=timestamp())
                write_json(output / "status.json", status)
            except (Exception, KeyboardInterrupt) as error:
                row.update(state="failed", error=str(error), finished_utc=timestamp())
                status.update(state="failed", updated_utc=timestamp())
                write_json(output / "status.json", status)
                raise
        summary = summarize_batch(output)
        write_json(output / "stage_summary.json", summary)
        status.update(state="complete" if summary["complete"] else "failed", updated_utc=timestamp())
        write_json(output / "status.json", status)
        return summary


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def summarize_group(plan, group, attempt):
    result = dict(task=group["task"], dataset=group["dataset"], step=group["step"], checkpoint=group["checkpoint"],
                  complete=False, planned_trials=5, valid_trials=None, valid_staged_trials=None,
                  native_task_success_count=None, native_task_success_rate=None,
                  stages=None, complete_success_count=None, complete_success_rate=None, errors=[])
    errors = result["errors"]
    evaluation = attempt / "run/evaluation"
    try:
        summary = read_json(evaluation / "summary.json")
        protocol = read_json(evaluation / "protocol.json")
        run = read_json(attempt / "run/run_manifest.json")
        deployment = read_json(attempt / "deployment/deployment_manifest.json")
        require(summary.get("task") == group["task"] and summary.get("prediction_horizon") == summary.get("execute_steps") == 50,
                "Summary task/H50/K50 mismatch")
        require(summary.get("diagnostic_only") is True and summary.get("formal_metrics_valid") is False,
                "Staged results must remain diagnostic, never formal")
        require(protocol.get("max_sim_seconds") == 90 and protocol.get("guard_mode") == "observe", "Expected 90 simulated seconds/observe protocol")
        require(deployment.get("checkpoint_path") == group["checkpoint"] and deployment.get("task") == group["task"]
                and deployment.get("prediction_horizon") == 50, "Deployment checkpoint/task/H50 mismatch")
        require(run.get("evaluation_returncode") == 0 and run.get("managed_server_stopped") is True
                and run.get("evaluation_process_stopped") is True, "Managed evaluator/service did not finish cleanly")
        rows = summary.get("results", summary.get("trials"))
        require(isinstance(rows, list), "Summary results missing")
        result.update(valid_trials=summary.get("valid_trials"), attempted_trials=summary.get("attempted"))
        stage = summary.get("staged_evaluation")
        if isinstance(stage, dict):
            result["valid_staged_trials"] = stage.get("valid_staged_trials")
        require(summary.get("planned") == summary.get("attempted") == 5 and len(rows) == 5
                and {row.get("seed") for row in rows} == set(range(5)), "Expected exactly five seeds 0..4")
        require(all(row.get("valid_trial") is True for row in rows) and summary.get("valid_trials") == 5,
                "Not all five trials are valid; missing/invalid trials are not failures or zero rates")
        require(isinstance(stage, dict) and stage.get("version") == "staged_v1" and stage.get("task") == group["task"],
                "staged_v1 aggregate missing or wrong task")
        denominator = stage.get("valid_staged_trials")
        result["valid_staged_trials"] = denominator
        require(type(denominator) is int and denominator == 5, "Expected five valid staged trials; do not use planned count as denominator")
        expected = {f"stage{i + 1}": name for i, name in enumerate(STAGES[group["task"]])}
        stages = stage.get("stages")
        require(isinstance(stages, dict) and set(stages) == set(expected), "Wrong staged definitions/inventory")
        for key, definition in expected.items():
            value = stages[key]
            require(value.get("definition") == definition and type(value.get("count")) is int
                    and 0 <= value["count"] <= denominator and finite(value.get("rate"))
                    and math.isclose(value["rate"], value["count"] / denominator), "Stage count/rate/definition mismatch")
        board = group["task"].startswith("whiteboard-")
        if board:
            require(stage.get("complete_success_count") is None and stage.get("complete_success_rate") is None,
                    "Whiteboard pickup is not cleaning completion success")
        else:
            last = stages[f"stage{len(expected)}"]
            require(stage.get("complete_success_count") == last["count"] and stage.get("complete_success_rate") == last["rate"],
                    "Staged completion count/rate mismatch")
        if group["task"] == "usb-unplug":
            transfer = deployment.get("training_provenance", {}).get("completion_protocol_transfer")
            require(isinstance(transfer, dict) and transfer.get("evaluation_completion_protocol") == USB_EVALUATION_PROTOCOL,
                    "USB protocol-transfer evidence missing")
            require(stage.get("completion_protocol") == USB_EVALUATION_PROTOCOL, "USB staged protocol is not held-clear-2s")
            result["completion_protocol_transfer"] = transfer
        reports = []
        native_count = 0
        for row in rows:
            seed = row["seed"]
            trial = evaluation / f"seed_{seed:03d}"
            detail = read_json(trial / "summary.json")
            report = read_json(trial / "staged_evaluation.json")
            require(detail.get("staged_evaluation") == report, f"seed {seed}: staged summary/sidecar differs")
            require(detail.get("task") == group["task"] and detail.get("seed") == seed
                    and detail.get("prediction_horizon") == detail.get("execute_steps") == 50
                    and detail.get("checkpoint_path") == group["checkpoint"]
                    and detail.get("checkpoint_sha256") == deployment.get("checkpoint_sha256")
                    and detail.get("deployment_id") == deployment.get("deployment_id"), f"seed {seed}: identity mismatch")
            require(finite(detail.get("sim_seconds")) and detail["sim_seconds"] >= 90 - 1e-8
                    and detail.get("full_duration_evaluation") is True
                    and report.get("full_duration_evaluation") is True, f"seed {seed}: full 90 seconds not recorded")
            require(not detail.get("error") and not detail.get("record_error") and not row.get("error")
                    and not row.get("record_error"), f"seed {seed}: infrastructure/recording error")
            require(type(detail.get("task_success")) is bool, f"seed {seed}: native task success absent")
            require(detail.get("status") == ("success" if detail["task_success"] else "task_not_completed"),
                    f"seed {seed}: status disagrees with native task success")
            native_count += detail["task_success"]
            require(report.get("stage_definitions") == expected and report.get("version") == "staged_v1"
                    and report.get("task") == group["task"]
                    and isinstance(report.get("stages"), dict) and set(report["stages"]) == set(expected)
                    and all(type(value) is bool for value in report["stages"].values()), f"seed {seed}: stage identity/flags mismatch")
            require(report.get("stage_flags") == report["stages"]
                    and report.get("native_task_success_preserved") is True
                    and report.get("expert_trajectory_used") is False
                    and report.get("simulation_state_mutated") is False,
                    f"seed {seed}: observer must remain passive and native outcome independent")
            require(report.get("thresholds") == stage.get("thresholds"), f"seed {seed}: stage thresholds changed")
            reports.append(report)
            # Native batch recording validation already checks camera coverage.
            # Independently require nonempty recorded artifacts here as well.
            review = trial / "review" if (trial / "review/review.json").is_file() else trial
            for name in ("review.mp4", "review.json", "frames.jsonl"):
                require((review / name).is_file() and (review / name).stat().st_size > 0, f"seed {seed}: missing video/review artifact {name}")
            metadata = read_json(review / "review.json")
            require(metadata.get("model_input_cameras_displayed") == ["head", "right_wrist"], f"seed {seed}: head/wrist recording missing")
            validate_state_log(trial / "staged_states.jsonl", group, seed, deployment, report)
        for key in expected:
            require(sum(report["stages"][key] for report in reports) == stages[key]["count"], "Aggregate stage counts disagree with trial flags")
        result.update(complete=True, stages=stages, native_task_success_count=native_count,
                      native_task_success_rate=native_count / 5,
                      complete_success_count=stage.get("complete_success_count"), complete_success_rate=stage.get("complete_success_rate"),
                      continuous_metrics=stage.get("continuous_metrics", {}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return result


def validate_state_log(path, group, seed, deployment, report):
    count, first, previous, previous_flags = 0, None, None, None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = row.get("time_s")
            require(row.get("version") == "staged_v1" and row.get("task") == group["task"]
                    and row.get("seed") == seed and row.get("prediction_horizon") == row.get("execute_steps") == 50
                    and row.get("checkpoint_path") == group["checkpoint"]
                    and row.get("deployment_id") == deployment.get("deployment_id")
                    and row.get("checkpoint_sha256") == deployment.get("checkpoint_sha256"), "Staged log task/seed/checkpoint/H50/K50 identity mismatch")
            require(finite(stamp) and stamp >= 0 and (previous is None or 0 < stamp - previous <= 0.1), "Staged state log time gap or invalid timestamp")
            require(type(row.get("control_tick")) is int and row["control_tick"] == count,
                    "Staged log control_tick is missing or not consecutive from zero")
            require(row.get("stage_definitions") == report["stage_definitions"]
                    and row.get("thresholds") == report.get("thresholds")
                    and isinstance(row.get("stages"), dict) and set(row["stages"]) == set(report["stages"])
                    and all(type(value) is bool for value in row["stages"].values())
                    and row.get("stage_flags") == row["stages"], "Staged state log definitions/flags differ")
            require(previous_flags is None or all(not value or row["stages"][key] for key, value in previous_flags.items()),
                    "Cumulative staged achievement flags regressed")
            if first is None:
                first = stamp
            previous, previous_flags, count = stamp, row["stages"], count + 1
    require(count >= 2698 and first is not None and first <= 0.1 and previous >= 89.93, "Staged state log does not cover the full 90 seconds at 30 Hz")
    require(previous_flags == report["stages"], "Final state-log stage flags disagree with the final staged report")


def summarize_batch(output):
    output = Path(output).absolute()
    require(not output.is_symlink(), "Batch directory cannot be a symlink")
    plan = read_json(output / "batch.json")
    require(plan.get("schema") == SCHEMA and plan.get("output_dir") == str(output), "Not this tool's owned batch directory")
    settings = plan["settings"]
    expected = build_plan(output, gpu=settings["gpu"], port=settings["port"], egl_device=settings["egl_device"],
                          allow_transfer=plan["allow_usb_unplug_protocol_transfer_diagnostic"])
    require(batch_identity(plan) == expected, "Recorded batch inventory/settings/checkpoints were modified")
    status = read_json(output / "status.json")
    require(len(status["groups"]) == 18, "Status group inventory mismatch")
    groups = []
    for group, state in zip(plan["groups"], status["groups"], strict=True):
        number = state.get("attempt")
        require(type(number) is int and 0 <= number <= 999, "Invalid attempt number")
        attempt = output / group["group_dir"] / f"attempt_{number:03d}"
        require(attempt.resolve().is_relative_to(output.resolve()), "Attempt path escapes owned batch directory")
        result = summarize_group(plan, group, attempt)
        result.update(controller_state=state.get("state"), attempt=number)
        groups.append(result)
    return dict(schema=SCHEMA, diagnostic_only=True, formal_metrics_valid=False,
                metric_scope="per-task, per-checkpoint cumulative staged diagnostics; no cross-checkpoint pooling",
                complete=all(group["complete"] for group in groups), total_groups=18, planned_trials=90,
                complete_groups=sum(group["complete"] for group in groups), groups=groups,
                whiteboard_note="picked_up measures eraser pickup only; complete_success_rate is null, not cleaning success")


def csv_summary(summary):
    output = io.StringIO()
    columns = ("task", "dataset", "step", "complete", "valid_trials", "valid_staged_trials",
               "native_task_success_count", "native_task_success_rate", "stage", "definition", "count", "rate", "errors")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for group in summary["groups"]:
        base = {key: group.get(key) for key in columns if key not in {"stage", "definition", "count", "rate", "errors"}}
        base["errors"] = " | ".join(group["errors"])
        for key, value in (group.get("stages") or {None: {}}).items():
            writer.writerow({**base, "stage": key, "definition": value.get("definition"), "count": value.get("count"), "rate": value.get("rate")})
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="default; CPU dry-run only, creates no files")
    modes.add_argument("--run", action="store_true", help="explicitly execute all 90 GPU trials serially")
    modes.add_argument("--summarize", action="store_true", help="read this batch only; stdout, no files changed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="one physical GPU index; ignored by summarize")
    parser.add_argument("--egl-device", type=int, help="physical EGL device index; default same as --gpu")
    parser.add_argument("--port", type=int, default=18980, help="owned loopback sidecar; groups reuse serially")
    parser.add_argument("--steps", nargs=2, type=int, default=[10000, 15000])
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument(TRANSFER_FLAG, action="store_true")
    parser.add_argument("--resume", action="store_true", help="with --run only: skip completed groups, preserve failed attempts")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="summarize stdout format")
    args = parser.parse_args(argv)
    try:
        require(not args.resume or args.run, "--resume requires --run")
        if args.summarize:
            summary = summarize_batch(args.output_dir)
            print(csv_summary(summary) if args.format == "csv" else json.dumps(summary, ensure_ascii=False, indent=2), end="\n")
            return 0 if summary["complete"] else 2
        plan = build_plan(args.output_dir, steps=args.steps, num_trials=args.num_trials, gpu=args.gpu, port=args.port,
                          egl_device=args.egl_device, allow_transfer=args.allow_usb_unplug_protocol_transfer_diagnostic)
        require(args.resume or not args.output_dir.exists(), "Select a fresh --output-dir; existing batches require --run --resume")
        report = preflight(plan)
        if not args.run:
            print(json.dumps({**plan, "preflight": report}, ensure_ascii=False, indent=2))
            return 0 if report["all_selected_valid"] else 2
        result = execute_batch(plan, report, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["complete"] else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Batch error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; owned artifacts preserved. Inspect status.json before explicit --run --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
