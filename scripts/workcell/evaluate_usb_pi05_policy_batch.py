#!/usr/bin/env python3
"""Serial USB pi0.5 evaluation with deterministic review-video selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.evaluation_guards import (
  DEFAULT_MAX_SIM_SECONDS,
  GUARD_MODES,
)
from kaihand_tactile_env.shared.evaluation_video import evaluation_artifact_directory

try:
  from prepare_usb_pi05_deployment import validate_deployment_identity
  from run_usb_pi05_policy import action_execution, evaluation_validity
except ModuleNotFoundError as error:
  if error.name != "prepare_usb_pi05_deployment":
    raise
  from scripts.workcell.prepare_usb_pi05_deployment import validate_deployment_identity
  from scripts.workcell.run_usb_pi05_policy import action_execution, evaluation_validity

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/workcell/run_usb_pi05_policy.py"
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
OPENPI_CLIENT = (
  Path("/cpfs_infra/user/chenxianchi/code/openpi") / "packages/openpi-client/src"
)
REVIEW_FILES = (
  "review.mp4",
  "review.json",
  "frames.jsonl",
  "first_frame.png",
  "last_frame.png",
)


def load_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if payload.get("schema") not in {"usb_pi05_deployment_v1", "usb_pi05_deployment_v2"}:
    raise RuntimeError(
      "expected usb_pi05_deployment_v1 or usb_pi05_deployment_v2 manifest"
    )
  if payload["schema"] == "usb_pi05_deployment_v2":
    validate_deployment_identity(payload)
  return resolved, payload


def fingerprints() -> dict:
  paths = [
    RUNNER,
    Path(__file__).resolve(),
    ROOT / "src/kaihand_tactile_env/tasks/usb_insert/config.py",
    ROOT / "src/kaihand_tactile_env/tasks/usb_insert/setup.py",
    ROOT / "src/kaihand_tactile_env/tasks/usb_insert/task.py",
    ROOT / "src/kaihand_tactile_env/tasks/usb_insert/review_metrics.py",
    ROOT / "src/kaihand_tactile_env/shared/simulation.py",
    ROOT / "src/kaihand_tactile_env/shared/rendering.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_video.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_guards.py",
  ]
  result = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths
  }
  result["scene_include_fingerprint"] = model_fingerprint(
    default_model_path("usb-insert")
  )
  return result


def classify(report: dict) -> tuple[str, bool]:
  if "record_error" in report:
    return "recording_error", False
  if report.get("physics_valid") is False:
    return "fatal_physics", False
  if report.get("status") == "success" and (
    report.get("task_success") is True
    or report.get("evaluation", {}).get("success") is True
  ):
    return "success", True
  if report.get("status") == "task_not_completed":
    return "time_limit", True
  error = report.get("error", "")
  for needle, reason in (
    ("socket total normal load", "socket_load_guard"),
    ("socket axial resistance", "socket_axial_guard"),
    ("socket side-wall load", "socket_wall_guard"),
    ("socket penetration", "penetration_guard"),
    ("plug fell", "plug_dropped"),
    ("nonfinite simulation", "nonfinite_state"),
    ("policy actions contain nonfinite", "nonfinite_action"),
  ):
    if needle in error:
      return reason, True
  return "infrastructure_or_contract_error", False


def guard_event_counts(results: list[dict]) -> dict[str, int]:
  counts = Counter()
  for result in results:
    for event in result.get("guard_events", []):
      counts[event.get("kind", "unknown")] += int(event.get("count", 1))
  return dict(counts)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--server", default="ws://127.0.0.1:18783")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--max-sim-seconds", type=float, default=DEFAULT_MAX_SIM_SECONDS)
  parser.add_argument(
    "--guard-mode",
    choices=GUARD_MODES,
    default="observe",
    help=(
      "observe records task/action violations and continues; strict terminates "
      "on the first violation"
    ),
  )
  parser.add_argument("--video-count", type=int, default=3)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument("--staged-evaluation-version", choices=("staged_v1",), default=None)
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help="Deprecated observe-mode alias: omit penetration-limit events",
  )
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  args = parser.parse_args()
  if args.reference_dataset is not None:
    args.reference_dataset = args.reference_dataset.expanduser().resolve()
  if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
    parser.error("distinct nonnegative seeds required")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be between zero and number of seeds")
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if not all(
    math.isfinite(value) and value > 0
    for value in (args.max_sim_seconds, args.trial_wall_limit)
  ):
    parser.error("time limits must be positive")
  if not all(
    math.isfinite(value) and value >= 0
    for value in (args.xy_jitter_mm, args.yaw_jitter_deg)
  ):
    parser.error("randomization ranges must be nonnegative")
  if args.disable_penetration_guard and args.guard_mode == "strict":
    parser.error("--disable-penetration-guard is incompatible with --guard-mode strict")
  if not EGL_VENDOR.is_file():
    raise FileNotFoundError(f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}")

  deployment_path, deployment = load_manifest(args.deployment_manifest)
  manifest_sha256 = hashlib.sha256(deployment_path.read_bytes()).hexdigest()
  interpolation = action_execution(deployment)
  validity = evaluation_validity(deployment, args.guard_mode)
  horizon = int(deployment["prediction_horizon"])
  if not 1 <= args.execute_steps <= horizon:
    parser.error(f"execute-steps must be in [1, {horizon}]")
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  frozen = fingerprints()
  protocol = {
    "task": "usb-insert",
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "reference_dataset": (
      None if args.reference_dataset is None else str(args.reference_dataset)
    ),
    "reference_episode_index": args.reference_episode_index,
    "model_family": deployment["model_family"],
    "server": args.server,
    "seeds": args.seeds,
    "workers": 1,
    "prediction_horizon": horizon,
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "execute_steps": args.execute_steps,
    "control_hz": 30,
    "replan_period_s": args.execute_steps / 30,
    "guard_mode": args.guard_mode,
    **validity,
    "action_interpolation": interpolation,
    "deployment_manifest_sha256": manifest_sha256,
    "penetration_guard": {
      "enabled": not args.disable_penetration_guard,
      "threshold_m": 0.0003,
      "other_guards_enabled": True,
    },
    "max_sim_seconds": args.max_sim_seconds,
    "trial_wall_limit": args.trial_wall_limit,
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "record_video": {
      "selection": "first N seeds in explicit --seeds order",
      "count": args.video_count,
      "fps": args.record_fps,
      "output_size": [1920, 1080],
      "bilateral_fingertip_force": True,
      "second_camera": "global",
      "model_views": ["head", "right_wrist"],
    },
    "initial_randomization": {
      "xy_uniform_mm": [-args.xy_jitter_mm, args.xy_jitter_mm],
      "yaw_uniform_deg": [-args.yaw_jitter_deg, args.yaw_jitter_deg],
      "online_control_noise": False,
    },
    "controller": (
      "pi05-absolute-joint-physics-interpolation-v2"
      if interpolation == "linear"
      else "direct-30hz-pi05-absolute-joint-v1"
    ),
    "success_rule": (
      "canonical UsbInsertionMonitor stable mechanical fit: aligned near-bottom, "
      "measured bottom-out evidence, stationary, seated continuously for 0.1s"
    ),
    "task_guard_policy": (
      "record_and_continue" if args.guard_mode == "observe" else "stop"
    ),
    "guard_failures_counted_as_failures": args.guard_mode == "strict",
    "source_hashes": frozen,
  }
  (output / "protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
  )

  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MUJOCO_GL": "egl",
    "KAIHAND_RENDER_BACKEND": "hardware",
    "__EGL_VENDOR_LIBRARY_FILENAMES": str(EGL_VENDOR),
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
  }
  client_source = (
    Path(deployment.get("model_project_path", OPENPI_CLIENT.parents[2]))
    / "packages/openpi-client/src"
  )
  pythonpath = [str(ROOT / "src"), str(client_source)]
  if environment.get("PYTHONPATH"):
    pythonpath.append(environment["PYTHONPATH"])
  environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
  environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
  for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
  ):
    environment.pop(key, None)

  results = []
  started = time.monotonic()
  for trial_index, seed in enumerate(args.seeds):
    if fingerprints() != frozen:
      raise RuntimeError("code/config changed during batch; refusing mixed protocol")
    if hashlib.sha256(deployment_path.read_bytes()).hexdigest() != manifest_sha256:
      raise RuntimeError("deployment manifest changed during batch")
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable,
      str(RUNNER),
      "--server",
      args.server,
      "--deployment-manifest",
      str(deployment_path),
      "--expected-deployment-manifest-sha256",
      manifest_sha256,
      "--seed",
      str(seed),
      "--xy-jitter-mm",
      str(args.xy_jitter_mm),
      "--yaw-jitter-deg",
      str(args.yaw_jitter_deg),
      "--execute-steps",
      str(args.execute_steps),
      "--max-sim-seconds",
      str(args.max_sim_seconds),
      "--guard-mode",
      args.guard_mode,
      "--no-save-first-request",
      "--output-dir",
      str(trial),
      "--record-fps",
      str(args.record_fps),
      "--review-width",
      "1920",
      "--review-height",
      "1080",
      "--review-second-camera",
      "global",
      "--record" if trial_index < args.video_count else "--no-record",
    ]
    if args.reference_dataset is not None:
      command.extend(
        (
          "--reference-dataset",
          str(args.reference_dataset),
          "--reference-episode-index",
          str(args.reference_episode_index),
        )
      )
    if args.disable_penetration_guard:
      command.append("--disable-penetration-guard")
    if args.staged_evaluation_version:
      command.extend(("--staged-evaluation-version", args.staged_evaluation_version))
    print(f"START {trial_index + 1}/{len(args.seeds)} seed={seed}", flush=True)
    trial_started = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x") as log:
      try:
        completed = subprocess.run(
          command,
          cwd=ROOT,
          env=environment,
          stdout=log,
          stderr=subprocess.STDOUT,
          timeout=args.trial_wall_limit,
        )
        returncode = completed.returncode
      except subprocess.TimeoutExpired:
        returncode = -999
    summary_path = trial / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    expected = {
      "task": "usb-insert",
      "deployment_id": deployment["deployment_id"],
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "model_family": "pi0.5",
      "prediction_horizon": horizon,
      "model_action_dim": deployment["model_action_dim"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "control_hz": 30,
      "guard_mode": args.guard_mode,
      "diagnostic_only": validity["diagnostic_only"],
      "action_interpolation": interpolation,
      "penetration_guard_enabled": not args.disable_penetration_guard,
    }
    identity_errors = {
      key: {"expected": value, "actual": summary.get(key)}
      for key, value in expected.items()
      if summary.get(key) != value
    }
    if identity_errors:
      raise RuntimeError(f"seed {seed}: trial identity mismatch: {identity_errors}")
    if deployment.get("schema") == "usb_pi05_deployment_v2" and any(
      summary.get(key, {}).get("status") != "verified"
      for key in ("deployment_integrity", "deployment_integrity_after")
    ):
      raise RuntimeError(f"seed {seed}: trial deployment integrity mismatch")
    reason, valid = classify(summary)
    if trial_index < args.video_count:
      review = evaluation_artifact_directory(trial)
      missing = [name for name in REVIEW_FILES if not (review / name).is_file()]
      if missing:
        if not valid:
          raise RuntimeError(
            f"seed {seed}: trial failed before review completion: "
            f"{summary.get('error') or f'returncode={returncode}'}; missing={missing}"
          )
        raise RuntimeError(f"seed {seed}: incomplete review artifact: {missing}")
      review_metadata = json.loads((review / "review.json").read_text())
      if review_metadata.get("completed") is not True:
        raise RuntimeError(f"seed {seed}: review did not complete")
      if review_metadata.get("model_input_cameras_displayed") != [
        "head",
        "right_wrist",
      ]:
        raise RuntimeError(f"seed {seed}: review omitted a model camera")
      if review_metadata.get("diagnostic_only") is not validity["diagnostic_only"]:
        raise RuntimeError(f"seed {seed}: review diagnostic-mode mismatch")

    task_success = bool(summary.get("task_success", reason == "success"))
    row = {
      "seed": seed,
      "success": task_success,
      "task_success": task_success,
      "valid_trial": valid,
      "termination": reason,
      "returncode": returncode,
      "wall_seconds": time.monotonic() - trial_started,
      "sim_seconds": summary.get("sim_seconds"),
      "error": summary.get("error"),
      "record_error": summary.get("record_error"),
      "requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "evaluation": summary.get("evaluation", {}),
      "guard_mode": summary.get("guard_mode"),
      "guard_events": summary.get("guard_events", []),
      "stop_reason": summary.get("stop_reason"),
      "physics_valid": summary.get("physics_valid"),
      "diagnostic_only": summary.get("diagnostic_only"),
      "formal_metrics_valid": summary.get("formal_metrics_valid"),
      "formal_result_valid": summary.get("formal_metrics_valid"),
      "deployment_integrity": summary.get("deployment_integrity"),
      "deployment_integrity_after": summary.get("deployment_integrity_after"),
      "action_interpolation": interpolation,
      "penetration_diagnostics": summary.get("penetration_diagnostics", {}),
      "summary": str(summary_path.relative_to(output)),
    }
    results.append(row)
    valid_count = sum(result["valid_trial"] for result in results)
    task_successes = sum(result["task_success"] for result in results)
    successes = sum(
      result["task_success"] and result["valid_trial"] for result in results
    )
    physics_valid = all(result["physics_valid"] is True for result in results)
    formal_metrics_valid = all(
      result["formal_metrics_valid"] is True for result in results
    )
    observed_success_rate = successes / valid_count if valid_count else None
    aggregate = {
      "task": "usb-insert",
      "deployment_id": deployment["deployment_id"],
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "model_family": "pi0.5",
      "prediction_horizon": horizon,
      "model_action_dim": deployment["model_action_dim"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "control_hz": 30,
      "replan_period_s": args.execute_steps / 30,
      "guard_mode": args.guard_mode,
      "diagnostic_only": validity["diagnostic_only"],
      "diagnostic_reasons": validity["diagnostic_reasons"],
      "action_interpolation": interpolation,
      "physics_valid": physics_valid,
      "formal_metrics_valid": formal_metrics_valid,
      "penetration_guard_enabled": not args.disable_penetration_guard,
      "penetration_guard_threshold_m": 0.0003,
      "max_sim_seconds": args.max_sim_seconds,
      "planned": len(args.seeds),
      "attempted": len(results),
      "valid_trials": valid_count,
      "successes": successes,
      "task_successes": task_successes,
      "success_rate": observed_success_rate,
      "formal_success_rate": (observed_success_rate if formal_metrics_valid else None),
      "diagnostic_success_rate": observed_success_rate,
      "infrastructure_errors": len(results) - valid_count,
      "recording_errors": sum(result["record_error"] is not None for result in results),
      "complete": len(results) == len(args.seeds) and valid_count == len(args.seeds),
      "termination_counts": dict(Counter(row["termination"] for row in results)),
      "stop_reason_counts": dict(Counter(row["stop_reason"] for row in results)),
      "guard_event_counts": guard_event_counts(results),
      "failure_stage_counts": dict(
        Counter(
          row["evaluation"].get("failure_stage")
          for row in results
          if row["valid_trial"] and not row["success"]
        )
      ),
      "wall_seconds": time.monotonic() - started,
      "trials": results,
    }
    from kaihand_tactile_env.shared.staged_recording import add_staged_batch_results

    add_staged_batch_results(aggregate, output, results)
    (output / "summary.json").write_text(
      json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"DONE seed={seed} success={row['success']} reason={reason} "
      f"sim={row['sim_seconds']} wall={row['wall_seconds']:.1f}s "
      f"cumulative={successes}/{valid_count}",
      flush=True,
    )
    if not valid:
      raise RuntimeError(f"seed {seed}: infrastructure/contract error; batch paused")
  print(f"COMPLETE results={output / 'summary.json'}", flush=True)


if __name__ == "__main__":
  main()
