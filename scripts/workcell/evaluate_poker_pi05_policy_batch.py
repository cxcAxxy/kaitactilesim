#!/usr/bin/env python3
"""Serial poker-draw pi0.5 evaluation with deterministic review selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.tasks.poker_draw.mid_full import CONTACT_MODEL_VERSION

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/workcell/run_poker_pi05_policy.py"
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
RAW_FILES = (
  "raw/episode.h5",
  "raw/episode.json",
  "raw/episode.result.json",
  "curves/right_hand_force_curves.png",
)


def load_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if payload.get("schema") != "poker_pi05_deployment_v1":
    raise RuntimeError("expected poker_pi05_deployment_v1 manifest")
  return resolved, payload


def fingerprints() -> dict:
  paths = [
    RUNNER,
    Path(__file__).resolve(),
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/mid_full.py",
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/friction.py",
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/randomization.py",
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/policy_control.py",
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/rollout_eval.py",
    ROOT / "src/kaihand_tactile_env/tasks/poker_draw/review_metrics.py",
    ROOT / "src/kaihand_tactile_env/shared/simulation.py",
    ROOT / "src/kaihand_tactile_env/shared/rendering.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_video.py",
    ROOT / "src/kaihand_tactile_env/shared/policy_video.py",
    ROOT / "scripts/workcell/poker_pi05_rollout_artifacts.py",
  ]
  result = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths
  }
  result["scene_include_fingerprint"] = model_fingerprint(
    default_model_path("poker-draw")
  )
  return result


def classify(report: dict) -> tuple[str, bool]:
  if (
    report.get("status") == "success"
    and report.get("evaluation", {}).get("success") is True
  ):
    return "success", True
  if report.get("status") == "task_not_completed":
    return "time_limit", True
  error = report.get("error", "")
  for needle, reason in (
    ("card penetrated", "penetration_guard"),
    ("card fell", "card_dropped"),
    ("nonfinite simulation", "nonfinite_state"),
    ("policy actions contain nonfinite", "nonfinite_action"),
  ):
    if needle in error:
      return reason, True
  return "infrastructure_or_contract_error", False


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--server", default="ws://127.0.0.1:18784")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--success-hold-seconds", type=float, default=0.10)
  parser.add_argument("--full-duration-evaluation", action="store_true")
  parser.add_argument("--video-count", type=int, default=3)
  parser.add_argument("--save-raw", action="store_true")
  parser.add_argument(
    "--review-second-camera", choices=("global", "overhead"), default="global"
  )
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: let trials continue beyond the 0.6 mm supported "
      "card/table penetration limit"
    ),
  )
  parser.add_argument("--xy-jitter-mm", type=float, default=4.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
  args = parser.parse_args()
  if args.full_duration_evaluation:
    args.disable_penetration_guard = True
  if not 0 < args.success_hold_seconds <= args.max_sim_seconds:
    parser.error("success-hold-seconds must be positive and at most max-sim-seconds")
  if args.reference_dataset is not None:
    args.reference_dataset = args.reference_dataset.expanduser().resolve()
  if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
    parser.error("distinct nonnegative seeds required")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be between zero and number of seeds")
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if args.max_sim_seconds <= 0 or args.trial_wall_limit <= 0:
    parser.error("time limits must be positive")
  if not 0 <= args.xy_jitter_mm <= 5.0 or not 0 <= args.yaw_jitter_deg <= 1.0:
    parser.error("poker randomization is limited to 5 mm XY and 1 degree yaw")
  if not EGL_VENDOR.is_file():
    raise FileNotFoundError(f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}")

  deployment_path, deployment = load_manifest(args.deployment_manifest)
  horizon = int(deployment["prediction_horizon"])
  if not 1 <= args.execute_steps <= horizon:
    parser.error(f"execute-steps must be in [1, {horizon}]")
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  frozen = fingerprints()
  protocol = {
    "task": "poker-draw",
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
    "success_hold_seconds": args.success_hold_seconds,
    "full_duration_evaluation": args.full_duration_evaluation,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard": {
      "enabled": not args.disable_penetration_guard,
      "threshold_m": 0.0006,
      "nonfinite_state_guard_enabled": True,
      "fallen_card_guard_enabled": not args.full_duration_evaluation,
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
      "second_camera": args.review_second_camera,
      "model_views": ["head", "right_wrist"],
      "audit_view": args.review_second_camera,
      "bilateral_fingertip_force": True,
    },
    "save_raw": args.save_raw,
    "initial_randomization": {
      "xy_uniform_mm": [-args.xy_jitter_mm, args.xy_jitter_mm],
      "yaw_uniform_deg": [-args.yaw_jitter_deg, args.yaw_jitter_deg],
      "online_observation_noise": False,
      "online_action_noise": False,
    },
    "controller": "direct-30hz-pi05-absolute-joint-v1",
    "contact_model": f"{CONTACT_MODEL_VERSION} (same fixed contact preset as source data)",
    "success_rule": (
      "contact; supported flat card >=40% overhang and >=50mm robotward travel; "
      "opposed lift >=20mm; face-to-head and face-to-robot cosines >=0.8; "
      f"inspection position error <=30mm; card clearance >=20mm; stable for {args.success_hold_seconds:g}s"
    ),
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
  pythonpath = [str(ROOT / "src"), str(OPENPI_CLIENT)]
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
    trial = output / f"seed_{seed:03d}"
    record = trial_index < args.video_count
    command = [
      sys.executable,
      str(RUNNER),
      "--server",
      args.server,
      "--deployment-manifest",
      str(deployment_path),
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
      "--success-hold-seconds",
      str(args.success_hold_seconds),
      "--output-dir",
      str(trial),
      "--record-fps",
      str(args.record_fps),
      "--review-width",
      "1920",
      "--review-height",
      "1080",
      "--review-second-camera",
      args.review_second_camera,
      "--show-model-wrist-in-review",
      "--record" if record else "--no-record",
      "--save-raw" if args.save_raw else "--no-save-raw",
      "--save-first-request" if record else "--no-save-first-request",
    ]
    if args.reference_dataset is not None:
      command.extend((
        "--reference-dataset", str(args.reference_dataset),
        "--reference-episode-index", str(args.reference_episode_index),
      ))
    if args.disable_penetration_guard:
      command.append("--disable-penetration-guard")
    if args.full_duration_evaluation:
      command.append("--full-duration-evaluation")
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
      "task": "poker-draw",
      "deployment_id": deployment["deployment_id"],
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "model_family": "pi0.5",
      "prediction_horizon": horizon,
      "model_action_dim": deployment["model_action_dim"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "success_hold_seconds": args.success_hold_seconds,
      "full_duration_evaluation": args.full_duration_evaluation,
      "control_hz": 30,
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_metrics_valid": not args.disable_penetration_guard,
      "penetration_guard_enabled": not args.disable_penetration_guard,
    }
    identity_errors = {
      key: {"expected": value, "actual": summary.get(key)}
      for key, value in expected.items()
      if summary.get(key) != value
    }
    if identity_errors:
      raise RuntimeError(f"seed {seed}: trial identity mismatch: {identity_errors}")
    reason, valid = classify(summary)
    if record and not valid:
      detail = summary.get("error")
      if not detail:
        detail = (
          "trial exceeded its wall-time limit"
          if returncode == -999
          else f"runner exited with returncode={returncode}"
        )
      raise RuntimeError(
        f"seed {seed}: runner failed before recorded-artifact validation: "
        f"{detail}; summary={summary_path}; "
        f"log={output / f'seed_{seed:03d}.log'}"
      )
    if args.save_raw and valid:
      missing_raw = [name for name in RAW_FILES if not (trial / name).is_file()]
      if missing_raw or summary.get("raw", {}).get("validated") is not True:
        raise RuntimeError(
          f"seed {seed}: incomplete Raw/force artifacts: {missing_raw}; "
          f"error={summary.get('raw_record_error')}"
        )
    if record:
      review = trial / "review"
      missing = [name for name in REVIEW_FILES if not (review / name).is_file()]
      if missing:
        if not valid:
          raise RuntimeError(
            f"seed {seed}: trial failed before review completion: "
            f"{summary.get('error') or f'returncode={returncode}'}; missing={missing}"
          )
        raise RuntimeError(f"seed {seed}: incomplete review artifact: {missing}")
      if not (trial / "first_request.npz").is_file():
        if summary.get("error"):
          raise RuntimeError(
            f"seed {seed}: runner failed before first-request audit: "
            f"{summary['error']}; summary={summary_path}; "
            f"log={output / f'seed_{seed:03d}.log'}"
          )
        raise RuntimeError(f"seed {seed}: missing two-camera first request audit")
      review_metadata = json.loads((review / "review.json").read_text())
      if review_metadata.get("completed") is not True:
        raise RuntimeError(f"seed {seed}: review did not complete")
      if review_metadata.get("model_input_cameras_displayed") != [
        "head",
        "right_wrist",
      ]:
        raise RuntimeError(f"seed {seed}: review omitted a model camera")
      if review_metadata.get("diagnostic_only") is not bool(
        args.disable_penetration_guard
      ):
        raise RuntimeError(f"seed {seed}: review diagnostic-mode mismatch")

    row = {
      "seed": seed,
      "success": reason == "success",
      "valid_trial": valid,
      "termination": reason,
      "returncode": returncode,
      "wall_seconds": time.monotonic() - trial_started,
      "sim_seconds": summary.get("sim_seconds"),
      "error": summary.get("error"),
      "requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "evaluation": summary.get("evaluation", {}),
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_result_valid": not args.disable_penetration_guard,
      "penetration_diagnostics": summary.get("penetration_diagnostics", {}),
      "summary": str(summary_path.relative_to(output)),
      "raw": summary.get("raw"),
    }
    results.append(row)
    valid_count = sum(result["valid_trial"] for result in results)
    successes = sum(result["success"] for result in results)
    aggregate = {
      "task": "poker-draw",
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
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_metrics_valid": not args.disable_penetration_guard,
      "penetration_guard_enabled": not args.disable_penetration_guard,
      "penetration_guard_threshold_m": 0.0006,
      "max_sim_seconds": args.max_sim_seconds,
      "save_raw": args.save_raw,
      "planned": len(args.seeds),
      "attempted": len(results),
      "valid_trials": valid_count,
      "successes": successes,
      "success_rate": successes / valid_count if valid_count else None,
      "infrastructure_errors": len(results) - valid_count,
      "complete": len(results) == len(args.seeds) and valid_count == len(args.seeds),
      "termination_counts": dict(Counter(row["termination"] for row in results)),
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
