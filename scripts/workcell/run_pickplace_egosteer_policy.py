#!/usr/bin/env python3
"""Run one manifest-bound PickPlace EgoSteer trial using the canonical controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import run_egosteer_policy as base
from kaihand_tactile_env.shared.egosteer_client import EgoSteerPolicyClient


async def verify_server(server: str, deployment: dict) -> None:
  async with EgoSteerPolicyClient(server) as client:
    metadata = client.metadata or {}
  expected = {
    "task": "pick-place",
    "deployment_id": deployment["deployment_id"],
    "model_family": "EgoSteer",
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "action_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "cameras": deployment["observation_contract"].get("cameras", ["head"]),
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
  }
  mismatch = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
  if mismatch:
    raise RuntimeError(f"PickPlace EgoSteer deployment mismatch: {mismatch}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--server", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--execute-steps", required=True, type=int)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--diagnostic-relax-ik", action="store_true",
    help="Simulation-only diagnostic: accept best-effort IK targets without the wrist/arm/hand reachability abort thresholds",
  )
  args = parser.parse_args()
  path = args.deployment_manifest.resolve(strict=True)
  deployment = json.loads(path.read_text(encoding="utf-8"))
  if deployment.get("schema") != "pickplace_egosteer_deployment_v1":
    raise RuntimeError("expected PickPlace EgoSteer deployment")
  if not 1 <= args.execute_steps <= int(deployment["prediction_horizon"]):
    parser.error("execute-steps must be in [1, model prediction_horizon]")
  asyncio.run(verify_server(args.server, deployment))
  output = args.output_dir.resolve()
  generic_argv = [
    "run_egosteer_policy.py", "--server", args.server, "--seed", str(args.seed),
    "--viewer", "none", "--output-dir", str(output),
    "--record" if args.record else "--no-record",
    "--record-fps", str(args.record_fps), "--review-width", "1920",
    "--review-height", "1080", "--review-second-camera", "global",
    "--max-sim-seconds", str(args.max_sim_seconds),
    "--execute-steps", str(args.execute_steps), "--no-real-time",
  ]
  if args.reference_dataset is not None:
    generic_argv.extend([
      "--reference-dataset", str(args.reference_dataset),
      "--reference-episode-index", str(args.reference_episode_index),
    ])
  if args.diagnostic_relax_ik:
    generic_argv.extend([
      "--max-wrist-jump", "inf", "--max-arm-position-error", "inf",
      "--max-arm-orientation-error", "inf", "--max-hand-error", "inf",
    ])
  old_argv = sys.argv
  try:
    sys.argv = generic_argv
    rollout_args = base._parse_args()
  finally:
    sys.argv = old_argv
  started = time.monotonic()
  stats = None
  error_text = None
  try:
    stats = asyncio.run(base._run(rollout_args))
  except Exception as error:
    error_text = f"{type(error).__name__}: {error}"
    raise
  finally:
    review_path = output / "review/review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    if review:
      status = review.get("task_status", "error")
      evaluation = review.get("evaluation") or {}
      sim_seconds = review.get("last_pose_time_s")
    elif stats is not None and error_text is None:
      status = "success" if stats.stable_success else "task_not_completed"
      evaluation = {"success": bool(stats.stable_success)}
      sim_seconds = stats.action_steps / 30.0
    else:
      status = "error"
      evaluation = {}
      sim_seconds = None
    result = {
      "task": "pick-place", "controller": "egosteer-pickplace-canonical-v1",
      "checkpoint_path": deployment["checkpoint_path"],
      "checkpoint_sha256": deployment["checkpoint_sha256"],
      "deployment_id": deployment["deployment_id"], "deployment_manifest": str(path),
      "model_family": "EgoSteer", "prediction_horizon": deployment["prediction_horizon"],
      "action_dim": deployment["action_dim"],
      "observation_contract": deployment["observation_contract"],
      "action_representation": deployment["action_representation"],
      "execute_steps": args.execute_steps, "control_hz": 30,
      "replan_period_s": args.execute_steps / 30, "seed": args.seed,
      "max_sim_seconds": args.max_sim_seconds,
      "initial_randomization": {"object_xy_jitter_m": 0.01, "object_yaw_jitter_rad": 0.05},
      "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
      "diagnostic_only": args.diagnostic_relax_ik,
      "formal_metrics_valid": not args.diagnostic_relax_ik,
      "kinematic_reachability_guards_enabled": not args.diagnostic_relax_ik,
      "status": status, "evaluation": evaluation,
      "sim_seconds": sim_seconds,
      "wall_seconds": time.monotonic() - started,
      "stats": asdict(stats) if stats is not None else {},
      "error": error_text or review.get("task_error"),
    }
    if review:
      review.update(
        checkpoint_path=deployment["checkpoint_path"], checkpoint_sha256=deployment["checkpoint_sha256"],
        model_family="EgoSteer", prediction_horizon=deployment["prediction_horizon"],
        action_dim=deployment["action_dim"], execute_steps=args.execute_steps,
        control_hz=30, replan_period_s=args.execute_steps / 30, seed=args.seed,
        success=bool(evaluation.get("success")), inference_requests=result["stats"].get("requests"),
        simulation_time_s=result["sim_seconds"], wall_time_s=result["wall_seconds"],
        diagnostic_only=args.diagnostic_relax_ik,
        formal_metrics_valid=not args.diagnostic_relax_ik,
        kinematic_reachability_guards_enabled=not args.diagnostic_relax_ik,
      )
      review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
  return None


if __name__ == "__main__":
  main()
