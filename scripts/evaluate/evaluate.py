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
  ("pick-place", "lingbot-vla2"): "evaluate_pickplace_lingbot_vla2_batch.py",
  ("poker-draw", "egosteer"): "evaluate_poker_policy_batch.py",
  ("poker-draw", "pi05"): "evaluate_poker_pi05_policy_batch.py",
  ("poker-draw", "pi05+trex"): "evaluate_poker_pi05+trex_policy_batch.py",
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

_SOURCE_TASK_DIRECTORY = {"usb-insert": "usb_insert"}


def _reference_dataset_from_checkpoint(payload: dict, task: str) -> Path | None:
  """Infer only the matching task/version LeRobot source, never a newer dataset.

  Fast pi0.5 training keeps the dataset and ``checkpoints`` in the same version
  directory.  Older conversions keep checkpoints under a model-family tree and
  the reference under the sibling ``lerobot_v3`` tree.  Preserve the legacy
  layout when it exists; otherwise use a colocated dataset whose LeRobot
  metadata is present.  Keep the legacy path as the final fallback for
  manifests that are inspected away from their source machine.
  """
  checkpoint = payload.get("checkpoint_path")
  if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
    return None
  parts = Path(checkpoint).parts
  task_directory = _SOURCE_TASK_DIRECTORY.get(task, task)
  for index, component in enumerate(parts):
    if (
      component == "sim"
      and len(parts) > index + 4
      and parts[index + 1] == task_directory
      and parts[index + 2] in {"pi05", "egosteer", "egotouch"}
      and parts[index + 4] == "checkpoints"
    ):
      colocated = Path(*parts[:index + 4])
      legacy = Path(*parts[:index + 2]) / "lerobot_v3" / parts[index + 3]
      if (legacy / "meta/info.json").is_file():
        return legacy
      if (colocated / "meta/info.json").is_file():
        return colocated
      return legacy
  return None


def _execute_steps_request(value: str) -> int | str:
  normalized = value.strip().lower()
  if normalized == "horizon":
    return normalized
  try:
    steps = int(normalized)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      "execute steps must be a positive integer or 'horizon'"
    ) from error
  if steps <= 0:
    raise argparse.ArgumentTypeError("execute steps must be positive")
  return steps


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="Pass runner-specific flags after --, for example: -- --server ws://127.0.0.1:18783",
  )
  parser.add_argument(
    "--list-support", action="store_true",
    help="list registered task/model evaluation runners without a deployment",
  )
  parser.add_argument(
    "--task",
    choices=(
      "pick-place", "poker-draw", "usb-insert", "bulb-screw", "vase-wipe",
      "install-ram", "whiteboard-wipe",
    ),
  )
  parser.add_argument(
    "--model-family",
    choices=("egosteer", "pi05", "pi05+trex", "egotouch", "lingbot-vla2")
  )
  parser.add_argument("--deployment-manifest", type=Path)
  parser.add_argument("--output-dir", type=Path)
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
    help="Record common tactile review videos for the first N trials; 0 disables them (default: num-trials)",
  )
  parser.add_argument(
    "--execute-steps",
    type=_execute_steps_request,
    nargs="+",
    default=("horizon",),
    metavar="N|horizon",
    help=(
      "One or more action-chunk execution lengths. The default executes the "
      "deployment prediction horizon"
    ),
  )
  parser.add_argument(
    "--reference-dataset", type=Path,
    help="LeRobot dataset containing a reference episode for comparison plots",
  )
  parser.add_argument("--reference-episode-index", type=int, default=0)
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
  if args.list_support:
    if rest:
      parser.error("--list-support does not accept runner-specific arguments")
    return args, rest
  for name in ("task", "model_family", "deployment_manifest", "output_dir"):
    if getattr(args, name) is None:
      parser.error(f"--{name.replace('_', '-')} is required")
  if args.num_trials <= 0:
    parser.error("--num-trials must be positive")
  if args.seed_start < 0:
    parser.error("--seed-start must be nonnegative")
  if args.video_count is None:
    args.video_count = args.num_trials
  if not 0 <= args.video_count <= args.num_trials:
    parser.error("--video-count must be between 0 and --num-trials")
  if len(args.execute_steps) != len(set(args.execute_steps)):
    parser.error("--execute-steps entries must be distinct")
  if args.max_sim_seconds is not None and args.max_sim_seconds <= 0:
    parser.error("--max-sim-seconds must be positive")
  if args.reference_episode_index < 0:
    parser.error("--reference-episode-index must be nonnegative")
  duplicated = [
    option
    for option in (
      "--seeds",
      "--video-count",
      "--execute-steps",
      "--max-sim-seconds",
      "--record-fps",
      "--reference-dataset",
      "--reference-episode-index",
      "--deployment-manifest",
      "--output-dir",
      "--task",
      "--model-family",
      "--cameras",
      "--num-trials",
      "--seed-start",
      "--dry-run",
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
  if not isinstance(declared_task, str) or not declared_task:
    raise ValueError("deployment manifest must declare task")
  if declared_task != task:
    raise ValueError(
      f"deployment task {declared_task!r} differs from requested task {task!r}"
    )
  declared_family = str(payload.get("model_family", "")).lower().replace(".", "")
  aliases = {
    "egosteer": "egosteer", "pi05": "pi05", "π05": "pi05",
    "lingbot-vla-20": "lingbot-vla2",
  }
  if not declared_family:
    raise ValueError("deployment manifest must declare model_family")
  normalized_family = aliases.get(declared_family, declared_family)
  # The tactile checkpoint extends this frozen pi0.5 deployment; the tactile
  # batch runner and single-trial runner validate its separate identity.
  accepted_families = {"pi05", "pi05+trex"} if family == "pi05+trex" else {family}
  if normalized_family not in accepted_families:
    raise ValueError(
      f"deployment model_family {payload.get('model_family')!r} differs from {family!r}"
    )
  observation = payload.get("observation_contract")
  if not isinstance(observation, dict) or "cameras" not in observation:
    raise ValueError("deployment manifest must declare observation_contract.cameras")
  declared_cameras = observation["cameras"]
  camera_names = policy_camera_names(observation)
  if normalized_family == "egotouch" and camera_names != ("head",):
    raise ValueError(
      "the current EgoTouch worker is single-RGB and requires cameras=['head']"
    )
  if cameras is not None and tuple(declared_cameras or ()) != tuple(cameras):
    raise ValueError(
      f"deployment cameras {declared_cameras!r} differ from requested {cameras!r}"
    )
  return resolved, payload


def _resolve_execution_modes(requests, payload):
  declared_horizon = payload.get("prediction_horizon")
  horizon = int(declared_horizon) if declared_horizon is not None else None
  modes = []
  seen = set()
  for request in requests:
    if request == "horizon":
      if horizon is None or horizon <= 0:
        raise ValueError(
          "--execute-steps horizon requires a positive prediction_horizon in the deployment manifest"
        )
      value = horizon
      name = f"execute_steps_horizon_{horizon}"
    else:
      value = int(request)
      name = f"execute_steps_{value}"
    if horizon is not None and value > horizon:
      raise ValueError(
        f"execute-steps={value} exceeds deployment prediction_horizon={horizon}"
      )
    if value in seen:
      continue
    seen.add(value)
    modes.append({"request": request, "value": value, "name": name})
  return modes


def main(argv=None):
  args, runner_args = parse_args(argv)
  if args.list_support:
    print(json.dumps({
      "runners": [
        {"task": task, "model_family": family, "backend": backend}
        for (task, family), backend in sorted(RUNNERS.items())
      ],
    }, indent=2, ensure_ascii=False))
    return 0
  manifest, payload = _validate_manifest(
    args.deployment_manifest, args.task, args.model_family, args.cameras
  )
  reference_dataset = args.reference_dataset
  if reference_dataset is None:
    declared_reference = payload.get("reference_dataset")
    if isinstance(declared_reference, str) and declared_reference:
      reference_dataset = Path(declared_reference)
      if not reference_dataset.is_absolute():
        reference_dataset = manifest.parent / reference_dataset
  if reference_dataset is None:
    reference_dataset = _reference_dataset_from_checkpoint(payload, args.task)
  if reference_dataset is not None:
    reference_dataset = reference_dataset.expanduser().resolve()
  modes = _resolve_execution_modes(args.execute_steps, payload)
  try:
    runner = RUNNERS[(args.task, args.model_family)]
  except KeyError as error:
    raise ValueError(
      f"no evaluation adapter for {args.task} + {args.model_family}; "
      "register a runner without changing this CLI contract"
    ) from error
  multiple_modes = len(modes) > 1
  evaluations = []
  for mode in modes:
    output = args.output_dir / mode["name"] if multiple_modes else args.output_dir
    command = [
      sys.executable,
      str(ROOT / "scripts/workcell" / runner),
      "--deployment-manifest", str(manifest),
      "--output-dir", str(output),
      "--seeds",
      *(str(seed) for seed in range(args.seed_start, args.seed_start + args.num_trials)),
      "--video-count", str(args.video_count),
      "--execute-steps", str(mode["value"]),
    ]
    if args.max_sim_seconds is not None:
      command.extend(("--max-sim-seconds", str(args.max_sim_seconds)))
    if args.record_fps is not None:
      command.extend(("--record-fps", str(args.record_fps)))
    if reference_dataset is not None:
      command.extend((
        "--reference-dataset", str(reference_dataset),
        "--reference-episode-index", str(args.reference_episode_index),
      ))
    command.extend(runner_args)
    evaluations.append({**mode, "output_dir": str(output), "command": command})
  description = {
    "task": args.task,
    "model_family": args.model_family,
    "deployment_id": payload.get("deployment_id"),
    "cameras": payload.get("observation_contract", {}).get("cameras"),
    "num_trials": args.num_trials,
    "seeds": list(range(args.seed_start, args.seed_start + args.num_trials)),
    "video_count": args.video_count,
    "execute_steps": [mode["value"] for mode in modes],
    "max_sim_seconds": args.max_sim_seconds,
    "record_fps": args.record_fps,
    "reference_dataset": (
      None if reference_dataset is None else str(reference_dataset)
    ),
    "reference_episode_index": args.reference_episode_index,
    "evaluations": evaluations,
  }
  if not multiple_modes:
    description["execute_steps"] = modes[0]["value"]
    description["command"] = evaluations[0]["command"]
  print(json.dumps(description, indent=2, ensure_ascii=False))
  if args.dry_run:
    return 0
  if args.output_dir.exists():
    raise FileExistsError(f"output must be a new directory: {args.output_dir}")
  if not multiple_modes:
    return subprocess.run(evaluations[0]["command"], check=False).returncode

  args.output_dir.mkdir(parents=True, exist_ok=False)
  matrix_path = args.output_dir / "evaluation_matrix.json"
  matrix = {**description, "complete": False, "results": []}
  matrix_path.write_text(
    json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
  )
  for evaluation in evaluations:
    returncode = subprocess.run(evaluation["command"], check=False).returncode
    matrix["results"].append({
      "request": evaluation["request"],
      "execute_steps": evaluation["value"],
      "output_dir": evaluation["output_dir"],
      "returncode": returncode,
    })
    matrix["complete"] = (
      len(matrix["results"]) == len(evaluations)
      and all(result["returncode"] == 0 for result in matrix["results"])
    )
    matrix_path.write_text(
      json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if returncode:
      return returncode
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
