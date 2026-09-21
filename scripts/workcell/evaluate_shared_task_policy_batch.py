#!/usr/bin/env python3
"""Serial evaluation for the shared Bulb, RAM, Vase, and Whiteboard adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
OPENPI_CLIENT = ROOT.parent / "openpi/packages/openpi-client/src"
TASKS = ("bulb-screw", "install-ram", "vase-wipe", "whiteboard-wipe")
RUNNERS = {
  "egosteer": ROOT / "scripts/workcell/run_shared_task_egosteer_policy.py",
  "pi05": ROOT / "scripts/workcell/run_shared_task_pi05_policy.py",
  "egotouch": ROOT / "scripts/workcell/run_shared_task_egotouch_policy.py",
}


def _family(value: object) -> str:
  normalized = str(value).lower().replace(".", "")
  aliases = {"pi05": "pi05", "π05": "pi05"}
  return aliases.get(normalized, normalized)


def load_deployment(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve(strict=True)
  deployment = json.loads(resolved.read_text(encoding="utf-8"))
  required = (
    "task",
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_horizon",
    "action_dim",
    "observation_contract",
    "action_representation",
  )
  if not isinstance(deployment, dict):
    raise RuntimeError("deployment manifest must contain a JSON object")
  missing = [key for key in required if key not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if deployment["task"] not in TASKS:
    raise RuntimeError(f"unsupported shared evaluation task: {deployment['task']!r}")
  family = _family(deployment["model_family"])
  if family not in RUNNERS:
    raise RuntimeError(f"unsupported shared evaluation model family: {family!r}")
  return resolved, deployment


def classify(report: dict) -> tuple[str, bool]:
  if (
    report.get("status") == "success"
    and report.get("evaluation", {}).get("success") is True
  ):
    return "success", True
  if report.get("status") == "task_not_completed":
    return "time_limit", True
  error = str(report.get("error", ""))
  for needle, reason in (
    ("wrist target jump", "wrist_target_guard"),
    ("arm IK failed", "arm_ik_guard"),
    ("hand IK", "hand_ik_guard"),
    ("hand target is too far", "hand_ik_guard"),
    ("fell below", "object_dropped"),
    ("force exceeded", "force_guard"),
    ("resistance exceeded", "force_guard"),
    ("penetration exceeded", "penetration_guard"),
    ("Sponge penetration", "penetration_guard"),
    ("Hand/environment penetration", "penetration_guard"),
    ("Hand struck", "collision_guard"),
    ("Sponge element inverted", "sponge_integrity_guard"),
    ("nonfinite simulation", "nonfinite_state"),
  ):
    if needle in error:
      return reason, True
  return "infrastructure_or_contract_error", False


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seeds", nargs="+", type=int, required=True)
  parser.add_argument("--video-count", type=int, required=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--server")
  parser.add_argument("--model-python")
  parser.add_argument("--model-project", type=Path)
  parser.add_argument("--snapshot", type=Path)
  args, passthrough = parser.parse_known_args(argv)
  if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
    parser.error("--seeds must contain distinct nonnegative integers")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be between zero and the number of seeds")
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if args.max_sim_seconds <= 0 or args.trial_wall_limit <= 0:
    parser.error("time limits must be positive")
  reserved = {
    "--task",
    "--seed",
    "--output-dir",
    "--deployment-manifest",
    "--record",
    "--no-record",
  }
  conflict = sorted(
    option
    for option in reserved
    if any(value == option or value.startswith(option + "=") for value in passthrough)
  )
  if conflict:
    parser.error(f"batch-owned options cannot be passed through: {conflict}")
  return parser, args, passthrough


def _environment() -> dict[str, str]:
  if not EGL_VENDOR.is_file():
    raise FileNotFoundError(f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}")
  if not (OPENPI_CLIENT / "openpi_client/__init__.py").is_file():
    raise FileNotFoundError(f"OpenPI client source is missing: {OPENPI_CLIENT}")
  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "LP_NUM_THREADS": "1",
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
  return environment


def main(argv=None):
  parser, args, passthrough = parse_args(argv)
  deployment_path, deployment = load_deployment(args.deployment_manifest)
  family = _family(deployment["model_family"])
  if family in {"egosteer", "pi05"} and not args.server:
    parser.error(f"--server is required for {family}")
  if family == "egotouch" and (not args.model_python or args.model_project is None):
    parser.error("--model-python and --model-project are required for egotouch")
  if args.execute_steps > int(deployment["prediction_horizon"]):
    parser.error(
      f"execute-steps={args.execute_steps} exceeds deployment horizon "
      f"{deployment['prediction_horizon']}"
    )

  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  protocol = {
    "task": deployment["task"],
    "model_family": deployment["model_family"],
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "seeds": args.seeds,
    "video_count": args.video_count,
    "video_selection": "first N seeds in explicit --seeds order",
    "record_fps": args.record_fps,
    "execute_steps": args.execute_steps,
    "control_hz": 30,
    "replan_period_s": args.execute_steps / 30,
    "max_sim_seconds": args.max_sim_seconds,
    "trial_wall_limit": args.trial_wall_limit,
    "workers": 1,
    "retry": False,
  }
  (output / "protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
  )

  results = []
  started = time.monotonic()
  for trial_index, seed in enumerate(args.seeds):
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable,
      str(RUNNERS[family]),
      "--task",
      deployment["task"],
      "--deployment-manifest",
      str(deployment_path),
      "--output-dir",
      str(trial),
      "--seed",
      str(seed),
      "--execute-steps",
      str(args.execute_steps),
      "--max-sim-seconds",
      str(args.max_sim_seconds),
      "--record-fps",
      str(args.record_fps),
      "--save-first-request" if trial_index == 0 else "--no-save-first-request",
      "--record" if trial_index < args.video_count else "--no-record",
    ]
    if family in {"egosteer", "pi05"}:
      command.extend(("--server", args.server))
    else:
      command.extend(
        (
          "--model-python",
          args.model_python,
          "--model-project",
          str(args.model_project),
        )
      )
      if args.snapshot is not None:
        command.extend(("--snapshot", str(args.snapshot)))
    command.extend(passthrough)

    print(f"START {trial_index + 1}/{len(args.seeds)} seed={seed}", flush=True)
    trial_started = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x", encoding="utf-8") as log:
      try:
        completed = subprocess.run(
          command,
          cwd=ROOT,
          env=_environment(),
          stdout=log,
          stderr=subprocess.STDOUT,
          timeout=args.trial_wall_limit,
        )
        returncode = completed.returncode
      except subprocess.TimeoutExpired:
        returncode = -999
    summary_path = trial / "summary.json"
    report = (
      json.loads(summary_path.read_text(encoding="utf-8"))
      if summary_path.is_file()
      else {"status": "error", "error": "trial produced no summary.json"}
    )
    identity = {
      "task": deployment["task"],
      "model_family": deployment["model_family"],
      "deployment_id": deployment["deployment_id"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "prediction_horizon": deployment["prediction_horizon"],
      "action_dim": deployment["action_dim"],
      "execute_steps": args.execute_steps,
      "control_hz": 30,
    }
    identity_errors = {
      key: {"expected": value, "actual": report.get(key)}
      for key, value in identity.items()
      if report.get(key) != value
    }
    reason, valid = classify(report)
    if identity_errors:
      reason, valid = "trial_identity_mismatch", False
    if valid and trial_index < args.video_count:
      required_review = ("review.mp4", "review.json", "frames.jsonl")
      missing = [
        name for name in required_review if not (trial / "review" / name).is_file()
      ]
      if missing:
        reason, valid = "incomplete_review_artifact", False
      else:
        review_metadata = json.loads(
          (trial / "review" / "review.json").read_text(encoding="utf-8")
        )
        expected_model_views = ["head"]
        if "right_wrist" in deployment["observation_contract"].get("cameras", []):
          expected_model_views.append("right_wrist")
        if review_metadata.get("model_input_cameras_displayed") != expected_model_views:
          reason, valid = "review_omitted_model_camera", False
    row = {
      "seed": seed,
      "success": reason == "success",
      "valid_trial": valid,
      "termination": reason,
      "returncode": returncode,
      "wall_seconds": time.monotonic() - trial_started,
      "sim_seconds": report.get("sim_seconds"),
      "error": report.get("error"),
      "identity_errors": identity_errors,
      "requests": report.get("stats", {}).get("requests", 0),
      "action_steps": report.get("stats", {}).get("action_steps", 0),
      "evaluation": report.get("evaluation", {}),
      "summary": (
        str(summary_path.relative_to(output)) if summary_path.exists() else None
      ),
    }
    results.append(row)
    valid_count = sum(item["valid_trial"] for item in results)
    successes = sum(item["success"] for item in results)
    aggregate = {
      **identity,
      "planned": len(args.seeds),
      "attempted": len(results),
      "valid_trials": valid_count,
      "successes": successes,
      "success_rate": successes / valid_count if valid_count else None,
      "termination_counts": dict(Counter(item["termination"] for item in results)),
      "wall_seconds": time.monotonic() - started,
      "results": results,
    }
    (output / "summary.json").write_text(
      json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"DONE seed={seed} termination={reason} valid={valid} success={row['success']}",
      flush=True,
    )
  return 0 if all(item["valid_trial"] for item in results) else 2


if __name__ == "__main__":
  raise SystemExit(main())
