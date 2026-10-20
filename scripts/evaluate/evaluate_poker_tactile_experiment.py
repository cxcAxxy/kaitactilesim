#!/usr/bin/env python3
"""Supervise the requested five-model poker experiment without mixing evidence.

All seeds, horizons, checkpoints, and attempt choices are fixed before inference.
The output is a fresh job directory; an invalid or missing trial remains visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FTP_RUN = Path("/nas/chenxianchi/datasets/sim/poker-draw/ftp1_checkpoints/ftp1/"
               "poker_draw_0920_200_h32_bs128_25k_ddp8_20261012_114006")
CPM = Path("/nas/chenxianchi/datasets/touchguidence/"
           "recovery_state_cpm_ratio_1_1_b1024_g8_20261013_165142/best.pt")
BASE = Path("/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k")
DATA = Path("/nas/chenxianchi/datasets/touchguidence/augmentation/"
            "poker_fingertip_recovery_rules_v1_random10/ratio_1_1")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def make_plan(job_root, *, num_trials=5, max_sim_seconds=90, concurrency=3):
    groups = []
    for label, family, checkpoint, horizon, start in [
        ("cpm_best", "pi05+cpm", CPM, 30, 0),
        *((f"ftp1_step{step}", "ftp1", FTP_RUN / str(step), 32, 1)
          for step in (10000, 15000, 20000, 24999)),
    ]:
        groups.append(dict(label=label, task="poker-draw", model_family=family,
            checkpoint=str(checkpoint), checkpoint_step="best" if family == "pi05+cpm" else checkpoint.name,
            prediction_horizon=horizon, action_start_index=start, execute_steps=5,
            deployment_manifest=str(job_root / "deployments" / label / "deployment_manifest.json"),
            evaluation_path=str(job_root / "groups" / label / "attempt_001" / "evaluation")))
    return dict(schema="kaihand-poker-tactile-experiment-plan-v1", created_utc=utc(),
        job_root=str(job_root), task="poker-draw", seeds=list(range(num_trials)),
        planned_trials=num_trials * len(groups), groups=groups, concurrency=concurrency,
        protocol=dict(max_sim_seconds=max_sim_seconds, guard_mode="observe", action_interpolation="zoh",
                      execute_steps=5, control_hz=30, record_fps=10,
                      staged_evaluation_version="staged_v1", diagnostic_only=True, formal_metrics_valid=False),
        attempt_selection="Fixed attempt_001 for every group, never selected by success score.",
        source_script=dict(path=str(Path(__file__).resolve()),
                           sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def prepare_command(group):
    cpm = group["model_family"] == "pi05+cpm"
    command = [sys.executable, "-B", str(ROOT / "scripts/workcell/prepare_poker_tactile_deployment.py"),
        "--task", "poker-draw", "--model-family", group["model_family"],
        "--checkpoint", group["checkpoint"],
        "--output-dir", str(Path(group["deployment_manifest"]).parent),
        "--model-project", str(Path("/cpfs_infra/user/chenxianchi/yidu/openpi-main") if cpm
                               else Path("/cpfs_infra/user/chenxianchi/code/ftp1-policy")),
        "--model-python", str(Path("/cpfs_infra/user/chenxianchi/miniconda3/envs/touchguidance/bin/python") if cpm
                              else Path("/cpfs_infra/user/chenxianchi/code/ftp1-policy/.venv/bin/python")),
        "--execute-steps", "5"]
    if cpm:
        command += ["--base-checkpoint", str(BASE), "--data-root", str(DATA),
                    "--tokenizer-model", "/root/.cache/openpi/big_vision/paligemma_tokenizer.model",
                    "--guidance-scale", "1", "--guidance-steps", "3"]
    return command


def evaluation_command(group, plan, trial_wall_limit):
    return [sys.executable, "-B", str(ROOT / "scripts/workcell/evaluate_poker_tactile_policy_batch.py"),
        "--deployment-manifest", group["deployment_manifest"], "--output-dir", group["evaluation_path"],
        "--seeds", *(str(seed) for seed in plan["seeds"]), "--video-count", str(len(plan["seeds"])),
        "--execute-steps", "5", "--max-sim-seconds", str(plan["protocol"]["max_sim_seconds"]),
        "--guard-mode", "observe", "--record-fps", "10", "--trial-wall-limit", str(trial_wall_limit),
        "--staged-evaluation-version", "staged_v1", "--action-interpolation", "zoh"]


def batch_progress(group):
    path = Path(group["evaluation_path"]) / "batch_summary.json"
    try:
        batch = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    trials = batch.get("trials", [])
    return dict(observed_trials=len(trials), valid_trials=sum(t.get("valid") is True for t in trials),
                invalid_trials=sum(t.get("valid") is not True for t in trials),
                batch_complete=batch.get("complete") is True,
                batch_summary=str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--max-sim-seconds", type=float, default=90)
    parser.add_argument("--concurrency", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--trial-wall-limit", type=float, default=3600)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if (args.num_trials <= 0 or not math.isfinite(args.max_sim_seconds)
            or not math.isfinite(args.trial_wall_limit)
            or args.max_sim_seconds <= 0 or args.trial_wall_limit <= 0):
        parser.error("trial count and time limits must be positive")
    job = args.job_root.absolute()
    if job.exists() or job.resolve() != job:
        parser.error("job root must be a fresh canonical path")
    job.mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "tmp").mkdir()
    (job / "cache/mpl").mkdir(parents=True)
    env = dict(os.environ, TMPDIR=str(job / "tmp"), MPLCONFIGDIR=str(job / "cache/mpl"),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=args.gpu,
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MUJOCO_GL="egl")
    plan = make_plan(job, num_trials=args.num_trials, max_sim_seconds=args.max_sim_seconds,
                     concurrency=args.concurrency)
    plan["runtime_environment"] = {key: env[key] for key in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TMPDIR", "MPLCONFIGDIR")}
    write_json(job / "plan.json", plan)
    status = dict(schema="kaihand-poker-tactile-experiment-status-v1", state="preparing",
                  updated_utc=utc(), groups={g["label"]:dict(state="pending", accepted_attempt=None,
                      evaluation_path=g["evaluation_path"]) for g in plan["groups"]})
    running = {}

    def save():
        status["updated_utc"] = utc()
        write_json(job / "status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        for group in plan["groups"]:
            row = status["groups"][group["label"]]
            row.update(state="preparing", prepare_command=prepare_command(group))
            save()
            with (job / "logs" / (group["label"] + ".prepare.log")).open("x") as log:
                result = subprocess.run(row["prepare_command"], cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                row.update(state="prepare_failed", returncode=result.returncode)
                raise RuntimeError(f"Deployment preparation failed for {group['label']}")
            row.update(state="prepared", deployment_manifest=group["deployment_manifest"])
            save()
            print(json.dumps(dict(event="prepared", group=group["label"])), flush=True)
        pending = list(plan["groups"])
        status["state"] = "running"
        while pending or running:
            while pending and len(running) < args.concurrency:
                group = pending.pop(0)
                row = status["groups"][group["label"]]
                command = evaluation_command(group, plan, args.trial_wall_limit)
                logfile = (job / "logs" / (group["label"] + ".evaluation.log")).open("x")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                running[group["label"]] = (group, process, logfile)
                row.update(state="running", pid=process.pid, command=command, started_utc=utc())
                print(json.dumps(dict(event="started", group=group["label"], pid=process.pid)), flush=True)
            for label, (group, process, logfile) in list(running.items()):
                row = status["groups"][label]
                row.update(batch_progress(group))
                returncode = process.poll()
                if returncode is None:
                    continue
                logfile.close()
                row.update(state="complete" if row.get("batch_complete") else "failed",
                           returncode=returncode, ended_utc=utc(),
                           accepted_attempt="attempt_001" if row.get("batch_complete") else None)
                del running[label]
                print(json.dumps(dict(event="finished", group=label, **row)), flush=True)
            save()
            if pending or running:
                time.sleep(10)
        status["state"] = "complete" if all(row["state"] == "complete" for row in status["groups"].values()) else "incomplete"
        save()
        return 0 if status["state"] == "complete" else 1
    except BaseException as error:
        for _, process, _ in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process, logfile in running.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            logfile.close()
        status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
