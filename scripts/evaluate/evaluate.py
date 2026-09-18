#!/usr/bin/env python3
"""Unified dispatch for task/model-specific simulation evaluation adapters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kaihand_tactile_env.shared.policy_cameras import policy_camera_names

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
  ("pick-place", "egosteer"): "evaluate_pickplace_policy_batch.py",
  ("pick-place", "pi05"): "evaluate_pickplace_policy_batch.py",
  ("poker-draw", "egosteer"): "evaluate_poker_policy_batch.py",
  ("poker-draw", "pi05"): "evaluate_poker_pi05_policy_batch.py",
  ("usb-insert", "egosteer"): "evaluate_usb_policy_batch.py",
  ("usb-insert", "pi05"): "evaluate_usb_pi05_policy_batch.py",
  ("bulb-screw", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("bulb-screw", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("bulb-screw", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "egotouch"): "evaluate_shared_task_policy_batch.py",
}


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="Pass runner-specific flags after --, for example: -- --server ws://127.0.0.1:18783",
  )
  parser.add_argument(
    "--task",
    required=True,
    choices=(
      "pick-place", "poker-draw", "usb-insert", "bulb-screw", "vase-wipe",
      "install-ram", "whiteboard-wipe",
    ),
  )
  parser.add_argument(
    "--model-family", required=True, choices=("egosteer", "pi05", "egotouch")
  )
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument(
    "--num-trials",
    type=int,
    default=20,
    help="Number of evaluation trials; seeds are seed-start..seed-start+N-1 (default: 20)",
  )
  parser.add_argument(
    "--seed-start",
    type=int,
    default=0,
    help="First deterministic evaluation seed (default: 0)",
  )
  parser.add_argument(
    "--video-count",
    type=int,
    default=None,
    help="Record common tactile review videos for the first N trials; 0 disables them (default: min(3, num-trials))",
  )
  parser.add_argument(
    "--execute-steps",
    type=int,
    default=5,
    help="Execute this many actions from each predicted chunk before querying again (default: 5)",
  )
  parser.add_argument(
    "--max-sim-seconds",
    type=float,
    help="Maximum simulated seconds per trial (default: task runner default)",
  )
  parser.add_argument(
    "--record-fps",
    type=int,
    choices=(5, 10),
    help="Review-video frame rate (default: task runner default, currently 10)",
  )
  parser.add_argument(
    "--cameras",
    nargs="+",
    choices=("head", "left_wrist", "right_wrist"),
    help="Assert the frozen model input cameras; does not rewrite a checkpoint",
  )
  parser.add_argument("--dry-run", action="store_true")
  args, rest = parser.parse_known_args(argv)
  if rest and rest[0] == "--":
    rest = rest[1:]
  if args.num_trials <= 0:
    parser.error("--num-trials must be positive")
  if args.seed_start < 0:
    parser.error("--seed-start must be nonnegative")
  if args.video_count is None:
    args.video_count = min(3, args.num_trials)
  if not 0 <= args.video_count <= args.num_trials:
    parser.error("--video-count must be between 0 and --num-trials")
  if args.execute_steps <= 0:
    parser.error("--execute-steps must be positive")
  if args.max_sim_seconds is not None and args.max_sim_seconds <= 0:
    parser.error("--max-sim-seconds must be positive")
  duplicated = [
    option
    for option in (
      "--seeds",
      "--video-count",
      "--execute-steps",
      "--max-sim-seconds",
      "--record-fps",
    )
    if any(value == option or value.startswith(option + "=") for value in rest)
  ]
  if duplicated:
    parser.error(
      f"{', '.join(duplicated)} belong to the unified interface; "
      "place them before -- and use --num-trials/--seed-start instead of --seeds"
    )
  return args, rest


def _validate_manifest(path, task, family, cameras=None):
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("deployment manifest must contain a JSON object")
  declared_task = payload.get("task")
  if declared_task is not None and declared_task != task:
    raise ValueError(
      f"deployment task {declared_task!r} differs from requested task {task!r}"
    )
  declared_family = str(payload.get("model_family", "")).lower().replace(".", "")
  aliases = {"egosteer": "egosteer", "pi05": "pi05", "π05": "pi05"}
  normalized_family = aliases.get(declared_family, declared_family)
  if declared_family and normalized_family != family:
    raise ValueError(
      f"deployment model_family {payload.get('model_family')!r} differs from {family!r}"
    )
  declared_cameras = payload.get("observation_contract", {}).get("cameras")
  camera_names = policy_camera_names(payload.get("observation_contract", {}))
  if normalized_family == "egotouch" and camera_names != ("head",):
    raise ValueError(
      "the current EgoTouch worker is single-RGB and requires cameras=['head']"
    )
  if cameras is not None and tuple(declared_cameras or ()) != tuple(cameras):
    raise ValueError(
      f"deployment cameras {declared_cameras!r} differ from requested {cameras!r}"
    )
  return resolved, payload


def main(argv=None):
  args, runner_args = parse_args(argv)
  manifest, payload = _validate_manifest(
    args.deployment_manifest, args.task, args.model_family, args.cameras
  )
  horizon = payload.get("prediction_horizon")
  if horizon is not None and args.execute_steps > int(horizon):
    raise ValueError(
      f"execute-steps={args.execute_steps} exceeds deployment prediction_horizon={horizon}"
    )
  try:
    runner = RUNNERS[(args.task, args.model_family)]
  except KeyError as error:
    raise ValueError(
      f"no evaluation adapter for {args.task} + {args.model_family}; "
      "register a runner without changing this CLI contract"
    ) from error
  command = [
    sys.executable,
    str(ROOT / "scripts/workcell" / runner),
    "--deployment-manifest", str(manifest),
    "--output-dir", str(args.output_dir),
    "--seeds",
    *(str(seed) for seed in range(args.seed_start, args.seed_start + args.num_trials)),
    "--video-count", str(args.video_count),
  ]
  command.extend(("--execute-steps", str(args.execute_steps)))
  if args.max_sim_seconds is not None:
    command.extend(("--max-sim-seconds", str(args.max_sim_seconds)))
  if args.record_fps is not None:
    command.extend(("--record-fps", str(args.record_fps)))
  command.extend(runner_args)
  print(json.dumps({
    "task": args.task,
    "model_family": args.model_family,
    "deployment_id": payload.get("deployment_id"),
    "cameras": payload.get("observation_contract", {}).get("cameras"),
    "num_trials": args.num_trials,
    "seeds": list(range(args.seed_start, args.seed_start + args.num_trials)),
    "video_count": args.video_count,
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "record_fps": args.record_fps,
    "command": command,
  }, indent=2, ensure_ascii=False))
  if args.dry_run:
    return 0
  if args.output_dir.exists():
    raise FileExistsError(f"output must be a new directory: {args.output_dir}")
  return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
  raise SystemExit(main())
