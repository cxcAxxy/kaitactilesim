#!/usr/bin/env python3
"""Evaluate one frozen PickPlace checkpoint on explicit seeds with review videos."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint

ROOT = Path(__file__).resolve().parents[2]
REVIEW_FILES = ("review.mp4", "review.json", "frames.jsonl", "first_frame.png", "last_frame.png")
RUNNERS = {
  "EgoSteer": ROOT / "scripts/workcell/run_pickplace_egosteer_policy.py",
  "pi0.5": ROOT / "scripts/workcell/run_pickplace_pi05_policy.py",
}


def definition_fingerprint(path: Path, names: set[str]) -> str:
  """Hash selected top-level definitions without unrelated runner code."""
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  selected = []
  for node in tree.body:
    node_names: set[str] = set()
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
      node_names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      node_names.update(
        target.id for target in targets if isinstance(target, ast.Name)
      )
    if node_names & names:
      selected.append(ast.dump(node, include_attributes=False))
  missing = names - {
    name
    for node in tree.body
    for name in (
      [node.name]
      if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
      else [
        target.id
        for target in (
          node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
      ]
      if isinstance(node, (ast.Assign, ast.AnnAssign))
      else []
    )
  }
  if missing:
    raise RuntimeError(f"fingerprint definitions missing from {path}: {sorted(missing)}")
  return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def fingerprints(runner: Path) -> dict[str, str]:
  files = [
    runner, Path(__file__).resolve(),
    ROOT / "scripts/workcell/run_usb_pi05_policy.py",
    ROOT / "src/kaihand_tactile_env/shared/simulation.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_video.py",
    ROOT / "src/kaihand_tactile_env/shared/recording.py",
    ROOT / "src/kaihand_tactile_env/tasks/pick_place/task.py",
  ]
  ego_runner = ROOT / "scripts/workcell/run_egosteer_policy.py"
  if runner == RUNNERS["EgoSteer"]:
    files.append(ego_runner)
  hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
  if runner == RUNNERS["pi0.5"]:
    hashes["scripts/workcell/run_egosteer_policy.py#pi05_helpers_ast"] = (
      definition_fingerprint(ego_runner, {
        "FINGERTIP_LINKS", "DEFAULT_FREE_CLOSE_FORCE_LIMIT",
        "DEFAULT_FREE_CLOSE_CUTOFF_M", "DEFAULT_FREE_CLOSE_MIN_CLOSURE",
        "StablePlacementDetector", "AutomaticGraspStabilizer",
        "AdaptiveFreeCloseForce",
      })
    )
  hashes["scene"] = model_fingerprint(default_model_path("pick-place"))
  return hashes


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", required=True)
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--seeds", nargs="+", required=True, type=int)
  parser.add_argument("--execute-steps", required=True, type=int)
  parser.add_argument("--max-sim-seconds", type=float, default=50)
  parser.add_argument("--trial-wall-limit", type=float, default=1800)
  parser.add_argument("--video-count", type=int, default=20)
  parser.add_argument("--record-fps", choices=(5, 10), type=int, default=10)
  parser.add_argument(
    "--diagnostic-relax-ik", action="store_true",
    help="EgoSteer-only simulation diagnostic: execute best-effort IK instead of aborting on reachability thresholds",
  )
  parser.add_argument(
    "--resume", action="store_true",
    help="Continue an interrupted batch in its existing output directory",
  )
  parser.add_argument(
    "--allow-source-change", action="store_true",
    help="With --resume, record and acknowledge changed evaluation source hashes",
  )
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args()
  if args.allow_source_change and not args.resume:
    parser.error("--allow-source-change requires --resume")
  if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
    parser.error("seeds must be distinct nonnegative integers")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be in [0, number of seeds]")
  if args.max_sim_seconds <= 0 or args.trial_wall_limit <= 0:
    parser.error("time limits must be positive")
  manifest_path = args.deployment_manifest.resolve(strict=True)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("task") != "pick-place" or manifest.get("model_family") not in RUNNERS:
    raise RuntimeError("expected frozen PickPlace EgoSteer or pi0.5 manifest")
  family = manifest["model_family"]
  if args.diagnostic_relax_ik and family != "EgoSteer":
    parser.error("--diagnostic-relax-ik is only applicable to EgoSteer")
  horizon = int(manifest["prediction_horizon"])
  if not 1 <= args.execute_steps <= horizon:
    parser.error(f"execute-steps must be in [1,{horizon}]")
  runner = RUNNERS[family]
  frozen = fingerprints(runner)
  output = args.output_dir.resolve()
  desired_protocol = {
    "task": "pick-place", "deployment_manifest": str(manifest_path),
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "model_family": family, "prediction_horizon": horizon,
    "action_dim": manifest["action_dim"], "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
    "seeds": args.seeds, "execute_steps": args.execute_steps,
    "control_hz": 30, "replan_period_s": args.execute_steps / 30,
    "max_sim_seconds": args.max_sim_seconds, "trial_wall_limit": args.trial_wall_limit,
    "randomization": {"xy_jitter_m": 0.01, "yaw_jitter_rad": 0.05},
    "success_rule": "cylinder in box, released and stable for dataset terminal duration",
    "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
    "diagnostic_only": args.diagnostic_relax_ik,
    "formal_metrics_valid": not args.diagnostic_relax_ik,
    "kinematic_reachability_guards_enabled": (
      not args.diagnostic_relax_ik if family == "EgoSteer" else None
    ),
    "review": {"count": args.video_count, "fps": args.record_fps, "resolution": [1920, 1080], "second_camera": "global", "tactile_heatmaps_and_curves": True},
    "source_hashes": frozen,
  }
  rows = []
  previous_wall_seconds = 0.0
  if args.resume:
    if not output.is_dir():
      raise RuntimeError(f"resume output directory does not exist: {output}")
    protocol_path = output / "protocol.json"
    summary_path = output / "summary.json"
    if not protocol_path.is_file() or not summary_path.is_file():
      raise RuntimeError("resume requires existing protocol.json and summary.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ignored = {"source_hashes", "initial_source_hashes", "source_transitions"}
    mismatch = {
      key: (value, protocol.get(key))
      for key, value in desired_protocol.items()
      if key not in ignored and protocol.get(key) != value
    }
    if mismatch:
      raise RuntimeError(f"resume protocol mismatch: {mismatch}")
    old_hashes = protocol.get("source_hashes", {})
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
      protocol.setdefault("initial_source_hashes", old_hashes)
      protocol.setdefault("source_transitions", []).append({
        "acknowledged": True,
        "completed_seeds_before_transition": [
          row.get("seed")
          for row in json.loads(summary_path.read_text(encoding="utf-8")).get("trials", [])
        ],
        "changed": changed,
      })
      protocol["source_hashes"] = frozen
      protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
      )
    aggregate = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(aggregate.get("trials", []))
    previous_wall_seconds = float(aggregate.get("wall_seconds", 0.0))
    completed_seeds = [row.get("seed") for row in rows]
    expected_prefix = args.seeds[:len(completed_seeds)]
    if completed_seeds != expected_prefix:
      raise RuntimeError(
        f"resume trials must be the requested seed prefix: {completed_seeds} != {expected_prefix}"
      )
    for index, seed in enumerate(completed_seeds):
      trial = output / f"seed_{seed:03d}"
      if not (trial / "summary.json").is_file():
        raise RuntimeError(f"resume trial summary missing: {trial}")
      if index < args.video_count:
        missing = [
          name for name in REVIEW_FILES
          if not (trial / "review" / name).is_file()
          or (trial / "review" / name).stat().st_size == 0
        ]
        if missing:
          raise RuntimeError(f"resume trial seed={seed} review missing: {missing}")
    print(
      f"RESUME {output}: completed={len(rows)}/{len(args.seeds)} "
      f"next_seed={args.seeds[len(rows)] if len(rows) < len(args.seeds) else 'none'}",
      flush=True,
    )
  else:
    output.mkdir(parents=True, exist_ok=False)
    protocol = desired_protocol
    (output / "protocol.json").write_text(
      json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
  environment = {
    **os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "MUJOCO_GL": "egl", "KAIHAND_RENDER_BACKEND": "hardware",
    "__EGL_VENDOR_LIBRARY_FILENAMES": str(ROOT / ".venv/etc/kaihand/10_nvidia.json"),
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
  }
  paths = [str(ROOT / "src"), "/cpfs_infra/user/chenxianchi/code/openpi/packages/openpi-client/src"]
  if environment.get("PYTHONPATH"):
    paths.append(environment["PYTHONPATH"])
  environment["PYTHONPATH"] = os.pathsep.join(paths)
  for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "LIBGL_ALWAYS_SOFTWARE"):
    environment.pop(key, None)

  started = time.monotonic()
  for index, seed in enumerate(args.seeds):
    if index < len(rows):
      continue
    if fingerprints(runner) != frozen:
      raise RuntimeError("evaluation source changed during batch")
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable, str(runner), "--server", args.server,
      "--deployment-manifest", str(manifest_path), "--output-dir", str(trial),
      "--seed", str(seed), "--execute-steps", str(args.execute_steps),
      "--max-sim-seconds", str(args.max_sim_seconds), "--record-fps", str(args.record_fps),
      "--record" if index < args.video_count else "--no-record",
    ]
    if args.diagnostic_relax_ik:
      command.append("--diagnostic-relax-ik")
    print(f"START {index + 1}/{len(args.seeds)} family={family} seed={seed}", flush=True)
    trial_started = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x") as log:
      try:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.trial_wall_limit)
        returncode = result.returncode
      except subprocess.TimeoutExpired:
        returncode = -999
    summary_path = trial / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    expected = {
      "task": "pick-place", "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": family, "prediction_horizon": horizon,
      "action_dim": manifest["action_dim"], "execute_steps": args.execute_steps,
      "control_hz": 30, "seed": seed,
      "diagnostic_only": args.diagnostic_relax_ik,
      "formal_metrics_valid": not args.diagnostic_relax_ik,
    }
    mismatches = {key: (value, summary.get(key)) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
      raise RuntimeError(f"seed={seed}: summary identity mismatch: {mismatches}")
    if index < args.video_count:
      review = trial / "review"
      missing = [name for name in REVIEW_FILES if not (review / name).is_file() or (review / name).stat().st_size == 0]
      if missing:
        raise RuntimeError(f"seed={seed}: review missing or empty: {missing}")
      review_meta = json.loads((review / "review.json").read_text(encoding="utf-8"))
      if review_meta.get("completed") is not True or review_meta.get("second_camera") != "global":
        raise RuntimeError(f"seed={seed}: incomplete review video")
      expected_model_views = ["head"]
      if "right_wrist" in manifest["observation_contract"].get("cameras", []):
        expected_model_views.append("right_wrist")
      if review_meta.get("model_input_cameras_displayed") != expected_model_views:
        raise RuntimeError(f"seed={seed}: review omitted a model camera")
    status = summary.get("status")
    success = status == "success" and summary.get("evaluation", {}).get("success") is True
    valid = status in ("success", "task_not_completed")
    row = {
      "seed": seed, "status": status, "success": success, "valid_trial": valid,
      "returncode": returncode, "sim_seconds": summary.get("sim_seconds"),
      "wall_seconds": time.monotonic() - trial_started,
      "inference_requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "error": summary.get("error"), "summary": str(summary_path.relative_to(output)),
    }
    rows.append(row)
    valid_count = sum(item["valid_trial"] for item in rows)
    successes = sum(item["success"] for item in rows)
    aggregate = {
      **{key: protocol[key] for key in ("task", "deployment_id", "checkpoint_path", "checkpoint_sha256", "model_family", "prediction_horizon", "action_dim", "execute_steps", "control_hz", "replan_period_s", "max_sim_seconds", "diagnostic_only", "formal_metrics_valid")},
      "planned": len(args.seeds), "attempted": len(rows), "valid_trials": valid_count,
      "successes": successes, "success_rate": successes / valid_count if valid_count else None,
      "complete": len(rows) == len(args.seeds) and valid_count == len(args.seeds),
      "status_counts": dict(Counter(item["status"] for item in rows)),
      "wall_seconds": previous_wall_seconds + time.monotonic() - started,
      "source_hash_consistent": not bool(protocol.get("source_transitions")),
      "source_transitions": protocol.get("source_transitions", []),
      "trials": rows,
    }
    (output / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE seed={seed} success={success} status={status} cumulative={successes}/{valid_count}", flush=True)
    if not valid or returncode != 0:
      raise RuntimeError(f"seed={seed}: trial error; batch paused, see {output / f'seed_{seed:03d}.log'}")
  print(f"COMPLETE {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
  main()
