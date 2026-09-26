#!/usr/bin/env python3
"""Evaluate one frozen LingBot-VLA-2.0 PickPlace deployment on explicit seeds.

This is deliberately separate from the EgoSteer/pi0.5 PickPlace batch runner.
It does not alter their inference, action, or evaluation protocols.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.evaluation_resume import require_valid_resume_trials
from lingbot_pickplace_contract import (
  ACTION_DIM,
  ACTION_REPRESENTATION,
  CONTROL_HZ,
  DEPLOYMENT_SCHEMA,
  HORIZON,
  INSTRUCTION,
  MODEL_ACTION_DIM,
  MODEL_FAMILY,
  OBSERVATION_CONTRACT,
  RIGHT_JOINT_NAMES,
  TASK,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/workcell/run_pickplace_lingbot_vla2_policy.py"
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
VIDEO_FILES = (
  "review.mp4", "review.json", "frames.jsonl", "first_frame.png",
  "last_frame.png",
)
COMPARISON_FILES = (
  "evaluation_rollout_trace.npz", "evaluation_comparison.json",
  "evaluation_right_wrist_state.png",
  "evaluation_right_hand_actuated_dof.png",
  "evaluation_right_fingertip_tactile.png",
)


def _definition_fingerprint(path: Path, names: set[str]) -> str:
  """Hash only the legacy helpers this dedicated runner imports."""
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  definitions: dict[str, ast.AST] = {}
  for node in tree.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      definitions[node.name] = node
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      for target in targets:
        if isinstance(target, ast.Name):
          definitions[target.id] = node
  missing = names - definitions.keys()
  if missing:
    raise RuntimeError(f"fingerprint definitions missing from {path}: {sorted(missing)}")
  selected = [
    ast.dump(node, include_attributes=False)
    for node in tree.body if any(definitions[name] is node for name in names)
  ]
  return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def fingerprints() -> dict[str, str]:
  """Freeze every local contract or execution source used by this batch."""
  files = (
    RUNNER,
    Path(__file__).resolve(),
    ROOT / "scripts/workcell/serve_pickplace_lingbot_vla2_policy.py",
    ROOT / "scripts/workcell/prepare_pickplace_lingbot_vla2_deployment.py",
    ROOT / "scripts/workcell/lingbot_pickplace_contract.py",
    ROOT / "scripts/workcell/lingbot_pickplace_reference.py",
    ROOT / "scripts/workcell/run_usb_pi05_policy.py",
    ROOT / "src/kaihand_tactile_env/shared/simulation.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_video.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_comparison.py",
    ROOT / "src/kaihand_tactile_env/shared/openwam_evaluation_plots.py",
    ROOT / "src/kaihand_tactile_env/shared/contact_tactile.py",
    ROOT / "src/kaihand_tactile_env/shared/policy_cameras.py",
    ROOT / "src/kaihand_tactile_env/shared/rendering.py",
    ROOT / "src/kaihand_tactile_env/shared/render_backend.py",
    ROOT / "src/kaihand_tactile_env/shared/config.py",
    ROOT / "src/kaihand_tactile_env/shared/recording.py",
    ROOT / "src/kaihand_tactile_env/tasks/pick_place/task.py",
  )
  hashes = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in files
  }
  legacy_runner = ROOT / "scripts/workcell/run_egosteer_policy.py"
  hashes["scripts/workcell/run_egosteer_policy.py#lingbot_helpers_ast"] = (
    _definition_fingerprint(legacy_runner, {
      "DEFAULT_FREE_CLOSE_FORCE_LIMIT", "DEFAULT_FREE_CLOSE_CUTOFF_M",
      "DEFAULT_FREE_CLOSE_MIN_CLOSURE", "FINGERTIP_LINKS",
      "StablePlacementDetector",
      "AutomaticGraspStabilizer", "AdaptiveFreeCloseForce",
    })
  )
  hashes["scene"] = model_fingerprint(default_model_path("pick-place"))
  return hashes


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
  resolved = path.expanduser().resolve(strict=True)
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("deployment manifest must be a JSON object")
  expected = {
    "schema": DEPLOYMENT_SCHEMA,
    "task": TASK,
    "model_family": MODEL_FAMILY,
    "prediction_horizon": HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": list(RIGHT_JOINT_NAMES),
    "observation_contract": OBSERVATION_CONTRACT,
    "action_representation": ACTION_REPRESENTATION,
    "instruction": INSTRUCTION,
    "reference_episode_index": 0,
  }
  mismatch = {
    key: (value, payload.get(key))
    for key, value in expected.items() if payload.get(key) != value
  }
  for key in (
    "deployment_id", "checkpoint_path", "checkpoint_sha256", "reference_dataset",
  ):
    if not isinstance(payload.get(key), str) or not payload[key]:
      mismatch[key] = ("nonempty string", payload.get(key))
  if mismatch:
    raise ValueError(f"LingBot-VLA-2.0 PickPlace deployment mismatch: {mismatch}")
  return resolved, payload


def _reference_dataset(
  manifest_path: Path, manifest: dict[str, Any], explicit: Path | None
) -> Path:
  declared = manifest.get("reference_dataset")
  assert isinstance(declared, str) and declared
  reference = Path(declared).expanduser()
  if not reference.is_absolute():
    reference = manifest_path.parent / reference
  resolved = reference.resolve()
  if explicit is not None and explicit.expanduser().resolve() != resolved:
    raise ValueError("--reference-dataset must match the frozen training dataset")
  return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", required=True)
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--seeds", nargs="+", required=True, type=int)
  parser.add_argument(
    "--execute-steps", type=int,
    help="Action steps per prediction; defaults to the frozen 50-step horizon",
  )
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--video-count", type=int)
  parser.add_argument("--record-fps", choices=(5, 10), type=int, default=10)
  parser.add_argument(
    "--reference-dataset", type=Path,
    help="Assert the manifest's frozen training reference dataset",
  )
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--allow-source-change", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args(argv)
  if args.allow_source_change and not args.resume:
    parser.error("--allow-source-change requires --resume")
  if args.dry_run and args.resume:
    parser.error("--dry-run and --resume cannot be combined")
  if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
    parser.error("--seeds must be distinct nonnegative integers")
  if args.video_count is None:
    args.video_count = len(args.seeds)
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("--video-count must be between zero and the number of seeds")
  if any(
    not math.isfinite(value) or value <= 0
    for value in (args.max_sim_seconds, args.trial_wall_limit)
  ):
    parser.error("time limits must be finite and positive")
  if args.reference_episode_index != 0:
    parser.error("LingBot PickPlace compares against training episode 00")
  return args


def _trial_command(
  args: argparse.Namespace, manifest_path: Path, seed: int, index: int,
  execute_steps: int, reference_dataset: Path | None, output: Path,
) -> list[str]:
  command = [
    sys.executable, str(RUNNER), "--server", args.server,
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output / f"seed_{seed:03d}"),
    "--seed", str(seed), "--execute-steps", str(execute_steps),
    "--max-sim-seconds", str(args.max_sim_seconds),
    "--record-fps", str(args.record_fps),
    "--record" if index < args.video_count else "--no-record",
  ]
  if reference_dataset is not None:
    command.extend((
      "--reference-dataset", str(reference_dataset),
      "--reference-episode-index", str(args.reference_episode_index),
    ))
  return command


def _environment() -> dict[str, str]:
  use_osmesa = os.environ.get("MUJOCO_GL", "").lower() == "osmesa"
  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
  }
  if use_osmesa:
    environment.update({
      "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
      "KAIHAND_RENDER_BACKEND": "software",
      "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
    })
    environment.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
  else:
    if not EGL_VENDOR.is_file():
      raise FileNotFoundError(f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}")
    environment.update({
      "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
      "KAIHAND_RENDER_BACKEND": "hardware",
      "__EGL_VENDOR_LIBRARY_FILENAMES": str(EGL_VENDOR),
    })
    environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
    environment.pop("GALLIUM_DRIVER", None)
  paths = [str(ROOT / "src"), "/cpfs_infra/user/chenxianchi/code/openpi/packages/openpi-client/src"]
  if environment.get("PYTHONPATH"):
    paths.append(environment["PYTHONPATH"])
  environment["PYTHONPATH"] = os.pathsep.join(paths)
  for key in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
    "all_proxy",
  ):
    environment.pop(key, None)
  return environment


def _validate_comparison(review: Path, reference_dataset: Path) -> None:
  missing = [
    name for name in COMPARISON_FILES
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if missing:
    raise RuntimeError(f"comparison files missing or empty: {missing}")
  comparison = json.loads((review / "evaluation_comparison.json").read_text(encoding="utf-8"))
  if comparison.get("status") != "ok":
    raise RuntimeError(
      "LingBot comparison against its training episode 00 is incomplete: "
      f"{comparison.get('status')}: {comparison.get('reason')}"
    )
  reference = comparison.get("reference", {})
  if (
    reference.get("dataset_root") != str(reference_dataset)
    or reference.get("output_episode_index") != 0
  ):
    raise RuntimeError("LingBot comparison reference differs from training episode 00")
  with np.load(review / "evaluation_rollout_trace.npz") as trace:
    times = np.asarray(trace["simulation_time_s"], dtype=np.float64)
    states = np.asarray(trace["state_29"])
    normal = np.asarray(trace["fingertip_normal_force_n"])
    tangent = np.asarray(trace["fingertip_tangent_force_n"])
  if (
    times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
    or states.shape != (len(times), 29)
    or normal.shape != (len(times), 5)
    or tangent.shape != (len(times), 5)
  ):
    raise RuntimeError("LingBot comparison trace is malformed or has too few samples")
  intervals = np.diff(times)
  if np.any(intervals <= 0) or abs(float(np.median(intervals)) - 1.0 / CONTROL_HZ) > 0.005:
    raise RuntimeError("LingBot comparison trace is not sampled at control-step cadence")


def _validate_video(review: Path) -> None:
  missing = [
    name for name in VIDEO_FILES
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if missing:
    raise RuntimeError(f"video files missing or empty: {missing}")
  metadata = json.loads((review / "review.json").read_text(encoding="utf-8"))
  if metadata.get("completed") is not True or metadata.get("second_camera") != "global":
    raise RuntimeError("incomplete LingBot review video")
  if metadata.get("model_input_cameras_displayed") != ["head", "right_wrist"]:
    raise RuntimeError("LingBot review omitted a model-input camera")
  if metadata.get("time_series_displayed") is not False:
    raise RuntimeError("LingBot review must not embed tactile time curves")
  if metadata.get("comparison_requested") is not True:
    raise RuntimeError("LingBot review did not request the episode-00 comparison")
  review_comparison = metadata.get("comparison_plots")
  if not isinstance(review_comparison, dict) or review_comparison.get("status") != "ok":
    raise RuntimeError("LingBot review metadata does not confirm the three plots")


def _aggregate(
  protocol: dict[str, Any], rows: list[dict[str, Any]], elapsed: float,
) -> dict[str, Any]:
  valid_count = sum(bool(row["valid_trial"]) for row in rows)
  successes = sum(bool(row["success"]) for row in rows)
  return {
    **{key: protocol[key] for key in (
      "task", "deployment_id", "checkpoint_path", "checkpoint_sha256",
      "model_family", "prediction_horizon", "model_action_dim", "action_dim",
      "execute_steps", "control_hz", "replan_period_s", "max_sim_seconds",
      "diagnostic_only", "formal_metrics_valid",
    )},
    "planned": len(protocol["seeds"]), "attempted": len(rows),
    "valid_trials": valid_count, "successes": successes,
    "success_rate": successes / valid_count if valid_count else None,
    "complete": len(rows) == len(protocol["seeds"]) and valid_count == len(rows),
    "status_counts": dict(Counter(row["status"] for row in rows)),
    "wall_seconds": elapsed,
    "source_hash_consistent": not bool(protocol.get("source_transitions")),
    "source_transitions": protocol.get("source_transitions", []),
    "trials": rows,
  }


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  manifest_path, manifest = load_manifest(args.deployment_manifest)
  horizon = int(manifest["prediction_horizon"])
  execute_steps = horizon if args.execute_steps is None else args.execute_steps
  if not 1 <= execute_steps <= horizon:
    raise ValueError(f"execute-steps must be in [1,{horizon}]")
  reference_dataset = _reference_dataset(
    manifest_path, manifest, args.reference_dataset
  )
  output = args.output_dir.expanduser().resolve()
  protocol: dict[str, Any] = {
    "task": "pick-place", "deployment_manifest": str(manifest_path),
    "deployment_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "model_family": MODEL_FAMILY, "prediction_horizon": horizon,
    "model_action_dim": manifest["model_action_dim"],
    "action_dim": manifest["action_dim"],
    "joint_names": manifest["joint_names"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
    "instruction": manifest["instruction"],
    "reference_dataset": str(reference_dataset) if reference_dataset is not None else None,
    "reference_episode_index": args.reference_episode_index,
    "seeds": args.seeds, "execute_steps": execute_steps,
    "control_hz": CONTROL_HZ, "replan_period_s": execute_steps / CONTROL_HZ,
    "render_backend": (
      "software_osmesa"
      if os.environ.get("MUJOCO_GL", "").lower() == "osmesa"
      else "hardware_egl"
    ),
    "max_sim_seconds": args.max_sim_seconds, "trial_wall_limit": args.trial_wall_limit,
    "randomization": {"xy_jitter_m": 0.01, "yaw_jitter_rad": 0.05},
    "success_rule": "cylinder in box, released and stable for dataset terminal duration",
    "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
    "diagnostic_only": False, "formal_metrics_valid": True,
    "review": {
      "count": args.video_count, "fps": args.record_fps,
      "comparison_count": len(args.seeds),
      "resolution": [1920, 1080], "second_camera": "global",
      "model_input_cameras": ["head", "right_wrist"],
      "tactile_heatmaps": True, "time_series_displayed": False,
      "control_step_comparison": True,
    },
  }
  if args.dry_run:
    print(json.dumps({
      "protocol": protocol,
      "commands": [
        _trial_command(
          args, manifest_path, seed, index, execute_steps, reference_dataset, output
        ) for index, seed in enumerate(args.seeds)
      ],
    }, ensure_ascii=False, indent=2))
    return 0

  frozen = fingerprints()
  protocol["source_hashes"] = frozen
  rows: list[dict[str, Any]] = []
  previous_wall_seconds = 0.0
  if args.resume:
    protocol_path = output / "protocol.json"
    summary_path = output / "summary.json"
    if not protocol_path.is_file() or not summary_path.is_file():
      raise RuntimeError("resume requires existing protocol.json and summary.json")
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    ignored = {"source_hashes", "initial_source_hashes", "source_transitions"}
    mismatch = {
      key: (value, existing.get(key))
      for key, value in protocol.items()
      if key not in ignored and existing.get(key) != value
    }
    if mismatch:
      raise RuntimeError(f"resume protocol mismatch: {mismatch}")
    old_hashes = existing.get("source_hashes", {})
    if old_hashes != frozen:
      changed = {
        key: {"before": old_hashes.get(key), "after": frozen.get(key)}
        for key in sorted(set(old_hashes) | set(frozen))
        if old_hashes.get(key) != frozen.get(key)
      }
      if not args.allow_source_change:
        raise RuntimeError(
          f"resume source hashes changed; inspect and pass --allow-source-change: {changed}"
        )
      existing.setdefault("initial_source_hashes", old_hashes)
      completed = json.loads(summary_path.read_text(encoding="utf-8")).get("trials", [])
      existing.setdefault("source_transitions", []).append({
        "acknowledged": True,
        "completed_seeds_before_transition": [row.get("seed") for row in completed],
        "changed": changed,
      })
      existing["source_hashes"] = frozen
      protocol_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = existing
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(previous.get("trials", []))
    previous_wall_seconds = float(previous.get("wall_seconds", 0.0))
    completed_seeds = [row.get("seed") for row in rows]
    if completed_seeds != args.seeds[:len(completed_seeds)]:
      raise RuntimeError("resume trials must be the requested seed prefix")
    require_valid_resume_trials(rows)
    for index, seed in enumerate(completed_seeds):
      if not (output / f"seed_{seed:03d}" / "summary.json").is_file():
        raise RuntimeError(f"resume trial seed={seed} summary missing")
      assert reference_dataset is not None
      review = output / f"seed_{seed:03d}" / "review"
      _validate_comparison(review, reference_dataset)
      if index < args.video_count:
        _validate_video(review)
  else:
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
      json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

  environment = _environment()
  started = time.monotonic()
  for index, seed in enumerate(args.seeds):
    if index < len(rows):
      continue
    if fingerprints() != frozen:
      raise RuntimeError("evaluation source changed during batch")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != protocol["deployment_manifest_sha256"]:
      raise RuntimeError("deployment manifest changed during batch")
    command = _trial_command(
      args, manifest_path, seed, index, execute_steps, reference_dataset, output
    )
    trial = output / f"seed_{seed:03d}"
    print(f"START {index + 1}/{len(args.seeds)} family={MODEL_FAMILY} seed={seed}", flush=True)
    trial_started = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x", encoding="utf-8") as log:
      try:
        process = subprocess.run(
          command, cwd=ROOT, env=environment, stdout=log,
          stderr=subprocess.STDOUT, timeout=args.trial_wall_limit, check=False,
        )
        returncode = process.returncode
      except subprocess.TimeoutExpired:
        returncode = -999
    summary_path = trial / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    if returncode != 0 or summary.get("status") == "error":
      cause = summary.get("error") or (
        "trial exceeded its wall-time limit" if returncode == -999
        else f"trial process exited with code {returncode}"
      )
      raise RuntimeError(
        f"seed={seed}: trial failed: {cause}; see {output / f'seed_{seed:03d}.log'}"
      )
    expected = {
      "task": "pick-place", "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "deployment_id": manifest["deployment_id"],
      "model_family": MODEL_FAMILY, "prediction_horizon": horizon,
      "model_action_dim": manifest["model_action_dim"],
      "action_dim": manifest["action_dim"], "execute_steps": execute_steps,
      "control_hz": CONTROL_HZ, "seed": seed,
      "observation_contract": manifest["observation_contract"],
      "action_representation": manifest["action_representation"],
      "reference_dataset": str(reference_dataset),
      "reference_episode_index": args.reference_episode_index,
      "diagnostic_only": False, "formal_metrics_valid": True,
    }
    mismatches = {
      key: (value, summary.get(key))
      for key, value in expected.items() if summary.get(key) != value
    }
    if mismatches:
      raise RuntimeError(f"seed={seed}: summary identity mismatch: {mismatches}")
    assert reference_dataset is not None
    _validate_comparison(trial / "review", reference_dataset)
    if index < args.video_count:
      _validate_video(trial / "review")
    status = summary.get("status")
    success = status == "success" and summary.get("evaluation", {}).get("success") is True
    valid = status in ("success", "task_not_completed")
    rows.append({
      "seed": seed, "status": status, "success": success,
      "valid_trial": valid, "returncode": returncode,
      "sim_seconds": summary.get("sim_seconds"),
      "wall_seconds": time.monotonic() - trial_started,
      "inference_requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "error": summary.get("error"),
      "summary": str(summary_path.relative_to(output)),
    })
    aggregate = _aggregate(
      protocol, rows, previous_wall_seconds + time.monotonic() - started
    )
    (output / "summary.json").write_text(
      json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"DONE seed={seed} success={success} status={status} "
      f"cumulative={aggregate['successes']}/{aggregate['valid_trials']}",
      flush=True,
    )
    if not valid or returncode != 0:
      raise RuntimeError(f"seed={seed}: trial error; see {output / f'seed_{seed:03d}.log'}")
  print(f"COMPLETE {output / 'summary.json'}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
