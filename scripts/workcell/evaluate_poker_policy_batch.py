"""Serial headless model evaluation with deterministic review-video selection."""

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

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/workcell/run_poker_egosteer_policy.py"
REVIEW_FILES = (
  "review.mp4",
  "review.json",
  "frames.jsonl",
  "first_frame.png",
  "last_frame.png",
)


def load_deployment_manifest(path):
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise RuntimeError("deployment manifest must contain a JSON object")
  required = (
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_horizon",
    "action_dim",
    "observation_contract",
    "action_representation",
  )
  missing = [key for key in required if key not in payload]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  return resolved, payload


def fingerprints():
  files = [
    RUNNER,
    Path(__file__).resolve(),
    ROOT / "scripts/workcell/run_egosteer_policy.py",
  ]
  source = ROOT / "src/kaihand_tactile_env"
  files += [
    source / "tasks/poker_draw" / name
    for name in (
      "policy_control.py",
      "rollout_eval.py",
      "review_metrics.py",
      "mid_full.py",
      "pressure_window.py",
      "press_control.py",
      "task.py",
      "config.py",
      "randomization.py",
    )
  ]
  files += [
    source / "shared" / name
    for name in (
      "simulation.py",
      "egosteer_adapter.py",
      "egosteer_client.py",
      "rendering.py",
      "evaluation_video.py",
    )
  ]
  result = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
  }
  result["scene_include_fingerprint"] = model_fingerprint(
    default_model_path("poker-draw")
  )
  return result


def classify(report):
  if (
    report.get("status") == "success"
    and report.get("evaluation", {}).get("success") is True
  ):
    return "success", True
  if report.get("status") == "task_not_completed":
    return "time_limit", True
  error = report.get("error", "")
  for needle, reason in (
    ("wrist target jump", "wrist_target_guard"),
    ("arm IK failed", "arm_ik_guard"),
    ("hand IK", "hand_ik_guard"),
    ("hand target is too far", "hand_ik_guard"),
    ("multiple draw fingers", "contact_loss"),
    ("card penetrated", "penetration_guard"),
    ("card fell", "card_dropped"),
    ("press failed", "press_timeout"),
    ("force-budget", "servo_guard"),
    ("actual arm servo", "servo_guard"),
    ("nonfinite simulation", "nonfinite_state"),
    ("arm Jacobian", "servo_guard"),
  ):
    if needle in error:
      return reason, True
  return "infrastructure_or_unclassified_error", False


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--server", default="ws://127.0.0.1:18766")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
  parser.add_argument("--trial-wall-limit", type=float, default=600.0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument(
    "--video-count",
    type=int,
    default=1,
    help="Record the first N requested seeds; 0 disables review videos",
  )
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument(
    "--execute-steps",
    type=int,
    default=5,
    help="Model actions executed at 30 Hz before the next inference request",
  )
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help="Diagnostic-only: do not stop on supported card/table penetration >0.6 mm",
  )
  args = parser.parse_args()
  if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
    parser.error("distinct nonnegative seeds required")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be between zero and the number of seeds")
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if args.max_sim_seconds <= 0 or args.trial_wall_limit <= 0:
    parser.error("time limits must be positive")
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  if args.execute_steps > deployment["prediction_horizon"]:
    parser.error(
      f"execute-steps={args.execute_steps} exceeds deployment horizon "
      f"{deployment['prediction_horizon']}"
    )
  output = args.output_dir.resolve()
  output.mkdir(parents=True, exist_ok=False)
  frozen = fingerprints()
  protocol = {
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "model_family": deployment["model_family"],
    "server": args.server,
    "seeds": args.seeds,
    "workers": 1,
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "replan_period_s": args.execute_steps / 30,
    "control_hz": 30,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard": {
      "enabled": not args.disable_penetration_guard,
      "threshold_m": 0.0006,
      "other_guards_enabled": True,
    },
    "max_sim_seconds": args.max_sim_seconds,
    "trial_wall_limit": args.trial_wall_limit,
    "viewer": "none",
    "record_video": {
      "selection": "first N seeds in the explicit --seeds order",
      "count": args.video_count,
      "fps": args.record_fps,
      "output_size": [1920, 1080],
      "bilateral_fingertip_force": True,
    },
    "save_rgb": False,
    "head_input": "320x240 RGB, JPEG95, 30Hz; history6/stride30",
    "initial_randomization": "XY +/-4mm, yaw +/-0.5 degrees; no online noise",
    "server_sampling_seed_fixed": False,
    "controller": "poker-contact-feedback-v2",
    "success_rule": "contact; supported flat card >=40% overhang and >=50mm robotward travel; opposed lift >=20mm above tabletop; head/robot face cosines>=.8, head-front reference error<=30mm, linear speed<.02m/s angular<.2rad/s for .1s",
    "guard_failures_counted_as_failures": True,
    "infrastructure_failures_excluded": True,
    "retry": False,
    "source_hashes": frozen,
  }
  (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
  results = []
  started = time.monotonic()
  env = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "LP_NUM_THREADS": "1",
    "MUJOCO_GL": "egl",
    "KAIHAND_RENDER_BACKEND": "hardware",
    "__EGL_VENDOR_LIBRARY_FILENAMES": str(
      ROOT / ".venv/etc/kaihand/10_nvidia.json"
    ),
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
  }
  env.pop("LIBGL_ALWAYS_SOFTWARE", None)
  for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
  ):
    env.pop(key, None)
  for trial_index, seed in enumerate(args.seeds):
    if fingerprints() != frozen:
      raise RuntimeError("code/config changed during batch; refusing mixed protocol")
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable,
      str(RUNNER),
      "--server",
      args.server,
      "--deployment-manifest",
      str(deployment_path),
      "--viewer",
      "none",
      "--seed",
      str(seed),
      "--execute-steps",
      str(args.execute_steps),
      "--max-sim-seconds",
      str(args.max_sim_seconds),
      "--evaluate",
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
    if args.disable_penetration_guard:
      command.append("--disable-penetration-guard")
    print(f"START {len(results) + 1}/{len(args.seeds)} seed={seed}", flush=True)
    wall = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x") as log:
      try:
        completed = subprocess.run(
          command,
          cwd=ROOT,
          env=env,
          stdout=log,
          stderr=subprocess.STDOUT,
          timeout=args.trial_wall_limit,
        )
        code = completed.returncode
      except subprocess.TimeoutExpired:
        code = -999
    summary_path = trial / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    expected_identity = {
      "deployment_id": deployment["deployment_id"],
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "model_family": deployment["model_family"],
      "prediction_horizon": deployment["prediction_horizon"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "control_hz": 30,
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_metrics_valid": not args.disable_penetration_guard,
      "penetration_guard_enabled": not args.disable_penetration_guard,
    }
    identity_errors = {
      key: {"expected": value, "actual": summary.get(key)}
      for key, value in expected_identity.items()
      if summary.get(key) != value
    }
    if identity_errors:
      raise RuntimeError(
        f"seed {seed}: trial identity mismatch: {identity_errors}"
      )
    if trial_index < args.video_count:
      review = trial / "review"
      missing_review = [name for name in REVIEW_FILES if not (review / name).is_file()]
      if missing_review:
        raise RuntimeError(
          f"seed {seed}: incomplete review artifact: {missing_review}"
        )
      review_metadata = json.loads((review / "review.json").read_text())
      if review_metadata.get("completed") is not True:
        raise RuntimeError(f"seed {seed}: review did not complete")
      if review_metadata.get("diagnostic_only") is not bool(
        args.disable_penetration_guard
      ):
        raise RuntimeError(f"seed {seed}: review diagnostic-mode mismatch")
    reason, valid = classify(summary)
    evaluation = summary.get("evaluation", {})
    row = {
      "seed": seed,
      "success": reason == "success",
      "valid_trial": valid,
      "termination": reason,
      "returncode": code,
      "wall_seconds": time.monotonic() - wall,
      "sim_seconds": summary.get("sim_seconds"),
      "error": summary.get("error"),
      "requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "evaluation": evaluation,
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_result_valid": not args.disable_penetration_guard,
      "penetration_diagnostics": summary.get("penetration_diagnostics", {}),
      "summary": str(summary_path.relative_to(output)),
    }
    results.append(row)
    valid_count = sum(r["valid_trial"] for r in results)
    successes = sum(r["success"] for r in results)
    aggregate = {
      "deployment_id": deployment["deployment_id"],
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "model_family": deployment["model_family"],
      "prediction_horizon": deployment["prediction_horizon"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "control_hz": 30,
      "replan_period_s": args.execute_steps / 30,
      "diagnostic_only": bool(args.disable_penetration_guard),
      "formal_metrics_valid": not args.disable_penetration_guard,
      "penetration_guard_enabled": not args.disable_penetration_guard,
      "penetration_guard_threshold_m": 0.0006,
      "max_sim_seconds": args.max_sim_seconds,
      "planned": len(args.seeds),
      "attempted": len(results),
      "valid_trials": valid_count,
      "successes": successes,
      "success_rate": successes / valid_count if valid_count else None,
      "infrastructure_errors": len(results) - valid_count,
      "complete": len(results) == len(args.seeds) and valid_count == len(args.seeds),
      "termination_counts": dict(Counter(r["termination"] for r in results)),
      "failure_stage_counts": dict(
        Counter(
          r["evaluation"].get("failure_stage")
          for r in results
          if r["valid_trial"] and not r["success"]
        )
      ),
      "wall_seconds": time.monotonic() - started,
      "trials": results,
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(
      f"DONE seed={seed} success={row['success']} reason={reason} sim={row['sim_seconds']} wall={row['wall_seconds']:.1f}s cumulative={successes}/{valid_count}",
      flush=True,
    )
    if not valid:
      raise RuntimeError(
        f"seed {seed}: infrastructure/unclassified failure; batch paused, not counted as model failure"
      )
  print(f"COMPLETE results={output / 'summary.json'}", flush=True)


if __name__ == "__main__":
  main()
