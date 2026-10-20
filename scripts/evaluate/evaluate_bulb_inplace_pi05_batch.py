#!/usr/bin/env python3
"""Exact 25-trial in-place bulb H50/K50 diagnostic batch, with explicit resume.

The default --plan only authenticates checkpoint metadata on CPU. --run creates
a fresh owned job; resume never retries a valid task failure or chooses the best
attempt. Prior infrastructure-failed attempts remain available for audit.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

_HELPER_PATH = Path(__file__).with_name("evaluate_pi05_stage_batch.py")
_SPEC = importlib.util.spec_from_file_location("inplace_stage_batch_helpers", _HELPER_PATH)
_helpers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_helpers)

ROOT = _helpers.ROOT
CHECKPOINT_ROOT = Path("/nas/chenxianchi/datasets/sim/bulb-screw-inplace/pi05/1018_200/checkpoints/pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1/bulb-screw-inplace_bs128_25k")
SCHEMA = "kaihand-pi05-bulb-inplace-stage-batch-v1"
TASK = "bulb-screw-inplace"
CONFIG = "pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1"
STEPS = (15000, 18000, 21000, 24000, 24999)
STAGES = ("stable_inplace_grasp", "threaded_and_seated", "tightening_verified", "released_stably")
DEFINITIONS = {f"stage{i + 1}": name for i, name in enumerate(STAGES)}
CONTROLLER = "pi05-absolute-joint-physics-interpolation-v2"
require = _helpers.require
read_json = _helpers.read_json
write_json = _helpers.write_json
timestamp = _helpers.timestamp
finite = _helpers.finite


def build_plan(output, *, steps=STEPS, num_trials=5, gpu="0", port=18986, egl_device=None):
    require(tuple(steps) == STEPS, "Exactly checkpoints 15000 18000 21000 24000 24999 are authorized")
    require(type(num_trials) is int and num_trials == 5, "Exactly five trials per checkpoint are authorized")
    # Reuse the original resource/path validations, not its 18-group contract.
    base = _helpers.build_plan(output, gpu=gpu, port=port, egl_device=egl_device)
    groups = [dict(task=TASK, dataset="1018_200", step=step,
                   checkpoint=str(CHECKPOINT_ROOT / str(step)),
                   group_dir=f"groups/{TASK}/step{step}") for step in STEPS]
    return {**base, "schema": SCHEMA, "groups": groups, "total_groups": 5, "total_trials": 25,
            "initial_state_randomization": False, "trial_scope": "fixed_initial_state_repeats",
            "initial_state_note": "Seeds 0..4 identify fixed-thread-entry repetitions, not randomized initial poses.",
            "settings": {**base["settings"], "position_randomization_applied": False}}


def preflight(plan):
    report = _helpers.preflight(plan)
    for row in report["checkpoints"]:
        if not row["valid"]:
            continue
        record = row["training_provenance"]["record"]
        try:
            require(record.get("task") == TASK and record.get("config_name") == CONFIG,
                    "Checkpoint must bind the dedicated in-place task/config")
            require(record.get("action_contract", {}).get("id") == "recorded-controller-target-v1",
                    "Checkpoint must bind recorded-controller-target-v1")
            normalizer = record.get("normalizer", {})
            require(normalizer.get("asset_id") == "normalizer_h50" and normalizer.get("use_quantile_norm") is True,
                    "Dedicated H50 checkpoint normalizer contract differs")
        except ValueError as error:
            row.update(valid=False, error=str(error))
    report["all_selected_valid"] = len(report["checkpoints"]) == 5 and all(row["valid"] for row in report["checkpoints"])
    return report


def batch_identity(plan):
    return _helpers.batch_identity(plan)


def load_owned_plan(output):
    output = Path(output).absolute()
    require(output.is_dir() and not output.is_symlink(), "Expected an existing owned batch directory")
    plan = read_json(output / "batch.json")
    require(plan.get("schema") == SCHEMA and plan.get("output_dir") == str(output), "Not an in-place batch owned by this tool")
    settings = plan["settings"]
    expected = build_plan(output, gpu=settings["gpu"], port=settings["port"], egl_device=settings["egl_device"])
    require(batch_identity(plan) == expected, "Recorded task/checkpoints/settings differ from the exact authorized plan")
    return plan


def summarize_group(plan, group, attempt):
    """Admit existing native results; do not rescore physics or stage evidence."""
    result = dict(task=TASK, dataset=group["dataset"], step=group["step"], checkpoint=group["checkpoint"],
                  complete=False, planned_trials=5, valid_trials=None, valid_staged_trials=None,
                  native_task_success_count=None, native_task_success_rate=None,
                  complete_success_count=None, complete_success_rate=None, stages=None,
                  final_net_angle_deg=None, initial_state_randomization=False,
                  trial_scope="fixed_initial_state_repeats", errors=[], failure_kind=None)
    evaluation = attempt / "run/evaluation"
    try:
        deployment = read_json(attempt / "deployment/deployment_manifest.json")
        run = read_json(attempt / "run/run_manifest.json")
        summary = read_json(evaluation / "summary.json")
        protocol = read_json(evaluation / "protocol.json")
        require(deployment.get("task") == TASK and deployment.get("checkpoint_path") == group["checkpoint"]
                and deployment.get("prediction_horizon") == 50 and deployment.get("checkpoint_step") == group["step"]
                and deployment.get("model_config_name") == CONFIG, "Deployment task/checkpoint/config/H50 identity mismatch")
        require(summary.get("task") == TASK and summary.get("prediction_horizon") == summary.get("execute_steps") == 50,
                "Native batch task/H50/K50 identity mismatch")
        require(summary.get("diagnostic_only") is True and summary.get("formal_metrics_valid") is False,
                "Observe rollouts must remain diagnostic, not formal claims")
        require(protocol.get("max_sim_seconds") == 90 and protocol.get("guard_mode") == "observe"
                and protocol.get("action_interpolation") == "linear" and protocol.get("controller") == CONTROLLER,
                "Evaluation protocol differs from 90-second observe contract")
        rows = summary.get("results")
        require(isinstance(rows, list) and summary.get("planned") == summary.get("attempted") == len(rows) == 5
                and {row.get("seed") for row in rows} == set(range(5)), "Exactly five seeds 0..4 must be recorded")
        result["valid_trials"] = summary.get("valid_trials")
        if not (run.get("evaluation_returncode") == 0 and run.get("managed_server_stopped") is True
                and run.get("evaluation_process_stopped") is True):
            raise OSError("Managed evaluation/service did not finish cleanly")
        if not (summary.get("valid_trials") == 5 and all(row.get("valid_trial") is True for row in rows)):
            require(False, "Native trials are invalid; inspect cause manually, never automatically replace policy/physics failures")
        stage = summary.get("staged_evaluation")
        require(isinstance(stage, dict) and stage.get("version") == "staged_v1" and stage.get("task") == TASK,
                "Missing or wrong staged_v1 aggregate identity")
        result["valid_staged_trials"] = stage.get("valid_staged_trials")
        require(type(stage.get("valid_staged_trials")) is int and stage["valid_staged_trials"] == 5,
                "Stage denominator is not five valid trials")
        stages = stage.get("stages")
        require(isinstance(stages, dict) and set(stages) == set(DEFINITIONS), "Wrong independent four-stage inventory")
        for key, definition in DEFINITIONS.items():
            item = stages[key]
            require(item.get("definition") == definition and type(item.get("count")) is int
                    and 0 <= item["count"] <= 5 and finite(item.get("rate"))
                    and math.isclose(item["rate"], item["count"] / 5), "Stage definition/count/rate mismatch")
        reports, native, angles = [], 0, []
        for row in rows:
            seed = row["seed"]
            trial = evaluation / f"seed_{seed:03d}"
            detail = read_json(trial / "summary.json")
            report = read_json(trial / "staged_evaluation.json")
            require(detail.get("staged_evaluation") == report, f"seed {seed}: sidecar differs from native summary")
            require(detail.get("task") == TASK and detail.get("seed") == seed
                    and detail.get("prediction_horizon") == detail.get("execute_steps") == 50
                    and detail.get("checkpoint_path") == group["checkpoint"]
                    and detail.get("deployment_id") == deployment.get("deployment_id")
                    and detail.get("checkpoint_sha256") == deployment.get("checkpoint_sha256")
                    and detail.get("action_interpolation") == "linear"
                    and detail.get("controller") == CONTROLLER, f"seed {seed}: identity/controller/interpolation mismatch")
            require(finite(detail.get("sim_seconds")) and detail["sim_seconds"] >= 90 - 1e-8
                    and detail.get("full_duration_evaluation") is True and report.get("full_duration_evaluation") is True,
                    f"seed {seed}: staged contract requires full 90 simulated seconds, including after success")
            if detail.get("error") or detail.get("record_error") or row.get("error") or row.get("record_error"):
                raise OSError(f"seed {seed}: native infrastructure/recording error")
            require(type(detail.get("task_success")) is bool
                    and detail.get("status") == ("success" if detail["task_success"] else "task_not_completed"),
                    f"seed {seed}: native status/outcome mismatch")
            require(report.get("version") == "staged_v1" and report.get("task") == TASK
                    and report.get("stage_definitions") == DEFINITIONS and report.get("thresholds") == stage.get("thresholds")
                    and isinstance(report.get("stages"), dict) and set(report["stages"]) == set(DEFINITIONS)
                    and all(type(value) is bool for value in report["stages"].values())
                    and report.get("stage_flags") == report["stages"], f"seed {seed}: stage flags/identity/thresholds differ")
            require(report.get("native_task_success_preserved") is True and report.get("expert_trajectory_used") is False
                    and report.get("simulation_state_mutated") is False, f"seed {seed}: observer is not passive")
            require(report.get("complete_success") is report["stages"]["stage4"], f"seed {seed}: staged completion differs from final stage")
            review = trial / "review" if (trial / "review/review.json").is_file() else trial
            for name in ("review.mp4", "review.json", "frames.jsonl"):
                if not (review / name).is_file() or (review / name).stat().st_size <= 0:
                    raise OSError(f"seed {seed}: missing/nonempty review artifact {name}")
            require(read_json(review / "review.json").get("model_input_cameras_displayed") == ["head", "right_wrist"],
                    f"seed {seed}: model head/wrist views not recorded")
            _helpers.validate_state_log(trial / "staged_states.jsonl", group, seed, deployment, report)
            native_evaluation = detail.get("evaluation", {})
            state = native_evaluation.get("state", {})
            initial = native_evaluation.get("threaded_initialization", {})
            initial_turns = initial.get("initial_clockwise_turns")
            require(initial.get("position_randomization_applied") is False
                    and initial.get("physically_validated") is True and initial.get("initially_threaded_verified") is True
                    and initial.get("initially_seated") is False and initial.get("initially_lit") is False
                    and finite(initial_turns), f"seed {seed}: physical fixed-thread-entry initialization not bound")
            turns = state.get("rotation_from_initial_turns")
            require(finite(turns), f"seed {seed}: actual-initial-relative final net rotation not recorded")
            require(finite(state.get("clockwise_turns"))
                    and math.isclose(turns, state["clockwise_turns"] - initial_turns, rel_tol=1e-9, abs_tol=1e-10),
                    f"seed {seed}: net rotation disagrees with this trial's actual initial angle")
            angles.append(dict(seed=seed, value=turns * 360, unit="deg", source="evaluation.state.rotation_from_initial_turns"))
            native += detail["task_success"]
            reports.append(report)
        for key in DEFINITIONS:
            require(sum(report["stages"][key] for report in reports) == stages[key]["count"], "Aggregate counts disagree with native reports")
        require(stage.get("complete_success_count") == stages["stage4"]["count"]
                and stage.get("complete_success_rate") == stages["stage4"]["rate"], "Stage completion aggregate differs")
        result.update(complete=True, stages=stages, native_task_success_count=native, native_task_success_rate=native / 5,
                      complete_success_count=stage["complete_success_count"], complete_success_rate=stage["complete_success_rate"],
                      final_net_angle_deg=dict(mean=sum(item["value"] for item in angles) / 5,
                                              minimum=min(item["value"] for item in angles),
                                              maximum=max(item["value"] for item in angles), trials=angles))
    except OSError as error:
        result.update(failure_kind="infrastructure", errors=[str(error)])
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result.update(failure_kind="integrity", errors=[str(error)])
    return result


def summarize_batch(output):
    plan = load_owned_plan(output)
    status = read_json(Path(output) / "status.json")
    require(len(status["groups"]) == 5, "Status group inventory mismatch")
    groups = []
    for group, state in zip(plan["groups"], status["groups"], strict=True):
        attempt = state.get("attempt")
        require(type(attempt) is int and 0 <= attempt <= 999, "Invalid attempt number")
        path = Path(output) / group["group_dir"] / f"attempt_{attempt:03d}"
        require(path.resolve().is_relative_to(Path(output).resolve()), "Attempt escapes owned batch")
        result = summarize_group(plan, group, path)
        groups.append({**result, "controller_state": state["state"], "attempt": attempt})
    return dict(schema=SCHEMA, complete=all(group["complete"] for group in groups), total_groups=5,
                complete_groups=sum(group["complete"] for group in groups), planned_trials=25, groups=groups,
                diagnostic_only=True, formal_metrics_valid=False, initial_state_randomization=False,
                trial_scope="fixed_initial_state_repeats", metric_scope="each checkpoint separately; no outcome-selected attempts")


def execute_batch(plan, report, *, resume=False):
    require(report["all_selected_valid"], "All five checkpoints must pass CPU provenance preflight")
    output = Path(plan["output_dir"])
    if resume:
        require(batch_identity(load_owned_plan(output)) == batch_identity(plan), "Resume changes the authorized batch")
    else:
        require(not output.exists(), "Refusing existing output directory; use explicit --run --resume")
        require(_helpers.TOKENIZER.is_file() and _helpers.EGL_VENDOR.is_file(), "Tokenizer/EGL cached assets unavailable")
        require(_helpers.MODEL_PYTHON.is_file() and _helpers.SIM_PYTHON.is_file(), "Runtime Python unavailable")
        require(_helpers.port_free(plan["settings"]["port"]), "Selected service port is occupied; not touching its owner")
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another controller owns this job") from error
        if resume:
            status = read_json(output / "status.json")
        else:
            write_json(output / "batch.json", {**plan, "created_utc": timestamp(), "preflight": report})
            status = dict(schema=SCHEMA, output_dir=str(output), state="starting",
                          groups=[dict(state="pending", attempt=0) for _ in plan["groups"]])
            write_json(output / "status.json", status)
            cache = output / ".cache/openpi/big_vision"
            cache.mkdir(parents=True)
            shutil.copy2(_helpers.TOKENIZER, cache / _helpers.TOKENIZER.name)
            env = _helpers.environment(plan)
            write_json(output / "runtime.json", dict(default_runtime="pi05", runtimes=dict(pi05=dict(
                python=str(_helpers.MODEL_PYTHON), project_root=str(_helpers.OPENPI), env={key: env[key] for key in
                    ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENPI_DATA_HOME")}))))
        require(len(status["groups"]) == 5, "Status inventory differs")
        status.update(state="running", controller_pid=os.getpid(), updated_utc=timestamp())
        write_json(output / "status.json", status)
        for group, row in zip(plan["groups"], status["groups"], strict=True):
            try:
                if row["attempt"]:
                    prior = output / group["group_dir"] / f"attempt_{row['attempt']:03d}"
                    validation = summarize_group(plan, group, prior)
                    if validation["complete"]:
                        row.update(state="complete", finished_utc=timestamp())
                        write_json(output / "status.json", status)
                        continue  # All-zero native outcomes are completed trials too.
                    require(row["state"] != "complete" and validation["failure_kind"] == "infrastructure",
                            "Existing attempt has integrity errors; no automatic re-evaluation or outcome selection")
                    row.setdefault("prior_attempts", []).append(dict(attempt=row["attempt"], validation=validation,
                                                                      retry_reason="infrastructure_only"))
                require(_helpers.port_free(plan["settings"]["port"]), "Service port occupied; not touching its owner")
                row.update(state="running", attempt=row["attempt"] + 1, started_utc=timestamp())
                attempt = output / group["group_dir"] / f"attempt_{row['attempt']:03d}"
                attempt.mkdir(parents=True, exist_ok=False)
                write_json(output / "status.json", status)
                prepare, run = _helpers.deploy_commands(plan, group, attempt, output / "runtime.json")
                require(_helpers.run_logged(prepare, attempt / "prepare.log", _helpers.environment(plan)) == 0,
                        f"Prepare/verify failed: {attempt / 'prepare.log'}")
                returncode = _helpers.run_logged(run, attempt / "evaluation.log", _helpers.environment(plan))
                validation = summarize_group(plan, group, attempt)
                write_json(attempt / "validation.json", validation)
                require(returncode == 0 and validation["complete"], f"Native group incomplete: {validation['errors']}")
                row.update(state="complete", finished_utc=timestamp())
                write_json(output / "status.json", status)
            except (Exception, KeyboardInterrupt) as error:
                row.update(state="failed", error=f"{type(error).__name__}: {error}", finished_utc=timestamp())
                status.update(state="failed", updated_utc=timestamp())
                write_json(output / "status.json", status)
                raise
        summary = summarize_batch(output)
        write_json(output / "stage_summary.json", summary)
        status.update(state="complete" if summary["complete"] else "failed", updated_utc=timestamp())
        write_json(output / "status.json", status)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="CPU metadata preflight; no outputs, model, or simulation")
    modes.add_argument("--run", action="store_true", help="execute exactly 25 trials serially")
    modes.add_argument("--summarize", action="store_true", help="read existing native artifacts; stdout only")
    parser.add_argument("--resume", action="store_true", help="with --run; only infrastructure-failed attempts can retry")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs=5, type=int, default=STEPS)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--egl-device", type=int)
    parser.add_argument("--port", type=int, default=18986)
    args = parser.parse_args(argv)
    try:
        require(not args.resume or args.run, "--resume requires --run")
        if args.summarize:
            summary = summarize_batch(args.output_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["complete"] else 2
        plan = build_plan(args.output_dir, steps=args.steps, num_trials=args.num_trials, gpu=args.gpu,
                          port=args.port, egl_device=args.egl_device)
        require(args.resume or not args.output_dir.exists(), "Select a fresh directory or explicit --run --resume")
        report = preflight(plan)
        result = execute_batch(plan, report, resume=args.resume) if args.run else {**plan, "preflight": report}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if (result.get("complete") if args.run else report["all_selected_valid"]) else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Batch error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; artifacts preserved. Inspect status.json before --run --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
