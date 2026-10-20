#!/usr/bin/env python3
"""Serial native-horizon CPM/FTP-1 trials, with honest validity accounting."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kaihand_tactile_env.shared.tactile_deployment import load_manifest, sha256_file
from kaihand_tactile_env.shared.tactile_task_contracts import SHARED_TACTILE_TASKS


def main(argv=None, *, shared_tasks=False):
  parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
  parser.add_argument("--video-count", type=int, default=3)
  parser.add_argument("--execute-steps", type=int)
  parser.add_argument("--max-sim-seconds", type=float, default=90)
  parser.add_argument("--trial-wall-limit", type=float, default=1800)
  parser.add_argument("--guard-mode", choices=("observe", "strict"), default="observe")
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--save-raw", action="store_true")
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  args, extra = parser.parse_known_args(argv)
  if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
    parser.error("seeds must be distinct and nonnegative")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("invalid video count")
  if any(not math.isfinite(t) or t <= 0 for t in (args.max_sim_seconds, args.trial_wall_limit)):
    parser.error("time limits must be positive and finite")
  manifest_path, manifest = load_manifest(args.deployment_manifest, verify_files=True)
  task = manifest["task"]
  if shared_tasks:
    if task not in SHARED_TACTILE_TASKS or manifest["model_family"] != "ftp1":
      parser.error("shared tactile batch requires a task-specific FTP-1 manifest")
    if args.save_raw:
      parser.error("shared tactile Raw capture is not implemented; omit --save-raw")
  elif task != "poker-draw":
    parser.error("poker tactile batch requires a poker-draw manifest")
  owned = {"--task", "--deployment-manifest", "--expected-deployment-manifest-sha256", "--output-dir", "--seed", "--execute-steps", "--record", "--no-record"}
  if any(token.split("=", 1)[0] in owned for token in extra):
    parser.error("runner-specific arguments cannot override frozen task/trial identity")
  steps = manifest["execute_steps"] if args.execute_steps is None else args.execute_steps
  if steps != manifest["execute_steps"]:
    parser.error("K is frozen in manifest; prepare another deployment for a different K")
  args.output_dir.mkdir(parents=True, exist_ok=False)
  batch = dict(schema="task_tactile_batch_v2" if shared_tasks else "poker_tactile_batch_v1", task=task, model_family=manifest["model_family"],
    prediction_horizon=manifest["prediction_horizon"], action_start_index=manifest["action_start_index"],
    execute_steps=steps, deployment_id=manifest["deployment_id"],
    deployment_manifest_sha256=sha256_file(manifest_path), complete=False, trials=[])
  started = time.monotonic()
  interrupted = False
  for index, seed in enumerate(args.seeds):
    output = args.output_dir / f"seed_{seed:04d}"
    runner = "run_shared_task_tactile_policy.py" if shared_tasks else "run_poker_tactile_policy.py"
    command = [sys.executable, str(ROOT / "scripts/workcell" / runner),
      "--deployment-manifest", str(manifest_path), "--output-dir", str(output),
      "--seed", str(seed), "--execute-steps", str(steps),
      "--max-sim-seconds", str(args.max_sim_seconds), "--guard-mode", args.guard_mode,
      "--record-fps", str(args.record_fps), "--record" if index < args.video_count else "--no-record"]
    if shared_tasks:
      command += ["--task", task, "--action-interpolation", "zoh",
                  "--expected-deployment-manifest-sha256", batch["deployment_manifest_sha256"]]
    if args.save_raw:
      command.append("--save-raw")
    if args.reference_dataset:
      command += ["--reference-dataset", str(args.reference_dataset), "--reference-episode-index", str(args.reference_episode_index)]
    command += extra
    with (args.output_dir / f"seed_{seed:04d}.log").open("x") as log:
      process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
      timed_out = False
      try:
        returncode = process.wait(timeout=args.trial_wall_limit)
      except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        timed_out = isinstance(error, subprocess.TimeoutExpired)
        interrupted = not timed_out
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=10)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
          process.wait(timeout=5)
        returncode = process.returncode
    summary = output / "summary.json"
    trial = json.loads(summary.read_text()) if summary.exists() else {}
    identity_matches = all(trial.get(key) == manifest[key] for key in (
      "task", "model_family", "prediction_horizon", "action_start_index", "execute_steps", "deployment_id"))
    identity_matches = identity_matches and trial.get("deployment_manifest_sha256") == batch["deployment_manifest_sha256"]
    valid = (returncode == 0 and trial.get("valid_trial") is True
      and trial.get("trial_validity") == "valid" and trial.get("physics_valid") is True
      and type(trial.get("task_success")) is bool and type(trial.get("formal_metrics_valid")) is bool
      and trial.get("formal_metrics_valid") is False and trial.get("diagnostic_only") is True
      and identity_matches and not timed_out and not interrupted and not any(k.endswith("record_error") for k in trial))
    batch["trials"].append(dict(seed=seed, summary=str(summary.relative_to(args.output_dir)),
      returncode=returncode, timed_out=timed_out, valid=valid,
      task_success=trial.get("task_success") if valid else None,
      status=trial.get("status", "missing_summary"), formal_metrics_valid=trial.get("formal_metrics_valid"),
      identity_matches=identity_matches,
      invalid_reason=None if valid else "supervised_trial_failed_validation"))
    batch.update(wall_seconds=time.monotonic() - started)
    (args.output_dir / "batch_summary.json").write_text(json.dumps(batch, indent=2) + "\n")
    if interrupted:
      break
  valid = [row for row in batch["trials"] if row["valid"]]
  batch.update(complete=len(batch["trials"]) == len(args.seeds), valid_trials=len(valid),
    invalid_trials=len(batch["trials"]) - len(valid),
    success_rate=None if not valid else sum(row["task_success"] for row in valid) / len(valid),
    diagnostic_only=True, formal_metrics_valid=False)
  (args.output_dir / "batch_summary.json").write_text(json.dumps(batch, indent=2) + "\n")
  print(json.dumps(batch, indent=2))
  return 0 if batch["complete"] and not batch["invalid_trials"] else 1


def install_termination_handler():
  def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")
  signal.signal(signal.SIGTERM, interrupted)


if __name__ == "__main__":
  install_termination_handler()
  raise SystemExit(main())
