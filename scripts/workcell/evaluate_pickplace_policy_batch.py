#!/usr/bin/env python3
"""Evaluate one frozen PickPlace checkpoint on explicit seeds with review videos."""

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

from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.evaluation_resume import require_valid_resume_trials
from kaihand_tactile_env.shared.evaluation_video import evaluation_artifact_directory

ROOT = Path(__file__).resolve().parents[2]
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
REVIEW_FILES = ("review.mp4", "review.json", "frames.jsonl", "first_frame.png", "last_frame.png")
COMPARISON_FILES = (
  "evaluation_rollout_trace.npz",
  "evaluation_comparison.json",
  "evaluation_right_wrist_state.png",
  "evaluation_right_hand_actuated_dof.png",
  "evaluation_right_fingertip_tactile.png",
)
RUNNERS = {
  "EgoSteer": ROOT / "scripts/workcell/run_pickplace_egosteer_policy.py",
  "pi0.5": ROOT / "scripts/workcell/run_pickplace_pi05_policy.py",
  "EgoTouch": ROOT / "scripts/workcell/run_pickplace_egotouch_policy.py",
}

EGOTOUCH_CAMERAS = ["head", "right_wrist"]
EGOTOUCH_VIEW_SLOTS = ["head", "left_wrist", "right_wrist"]
EGOTOUCH_VIEW_MASK = [True, False, True]
EGOTOUCH_POLICY_REJECTION_STATUS = "policy_action_rejected"
EGOTOUCH_REACHABILITY_GUARDS = {
  "wrist_target_jump",
  "arm_ik",
  "hand_ik",
  "hand_reachability",
}
RENDER_BACKENDS = ("hardware-egl", "software-osmesa")
EGOTOUCH_EXECUTE_STEPS = 5
EGOTOUCH_HAND_IK_ORIENTATION_WEIGHTS = (0.020, 0.0, 0.0, 0.0, 0.0)


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
    ROOT / "src/kaihand_tactile_env/shared/evaluation_guards.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_video.py",
    ROOT / "src/kaihand_tactile_env/shared/evaluation_comparison.py",
    ROOT / "src/kaihand_tactile_env/shared/openwam_evaluation_plots.py",
    ROOT / "src/kaihand_tactile_env/shared/contact_tactile.py",
    ROOT / "src/kaihand_tactile_env/shared/recording.py",
    ROOT / "src/kaihand_tactile_env/tasks/pick_place/task.py",
  ]
  ego_runner = ROOT / "scripts/workcell/run_egosteer_policy.py"
  if runner in (RUNNERS["EgoSteer"], RUNNERS["EgoTouch"]):
    files.append(ego_runner)
  if runner == RUNNERS["EgoSteer"]:
    files.append(ROOT / "src/kaihand_tactile_env/shared/egosteer_client.py")
  if runner == RUNNERS["EgoTouch"]:
    files.extend((
      ROOT / "scripts/workcell/egotouch_model_worker.py",
      ROOT / "scripts/workcell/prepare_pickplace_egotouch_deployment.py",
      ROOT / "scripts/workcell/run_shared_task_egotouch_policy.py",
    ))
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


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_hash(value) -> str:
  return hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()


def _valid_egotouch_policy_failure(value) -> bool:
  if not isinstance(value, dict):
    return False
  request = value.get("request")
  action_index = value.get("action_index")
  return (
    value.get("kind") == "kinematic_reachability_guard"
    and value.get("guard") in EGOTOUCH_REACHABILITY_GUARDS
    and isinstance(value.get("message"), str)
    and bool(value["message"])
    and isinstance(request, int)
    and not isinstance(request, bool)
    and request >= 1
    and isinstance(action_index, int)
    and not isinstance(action_index, bool)
    and action_index >= 0
    and value.get("action_executed") is False
  )


def _require_valid_resume_rows(rows: list[dict], family: str) -> None:
  if family != "EgoTouch":
    require_valid_resume_trials(rows)
    return
  for index, row in enumerate(rows):
    status = row.get("status") if isinstance(row, dict) else None
    valid_status = status in {"success", "task_not_completed"}
    if status == EGOTOUCH_POLICY_REJECTION_STATUS:
      valid_status = _valid_egotouch_policy_failure(row.get("policy_failure"))
    if (
      not valid_status
      or row.get("valid_trial") is not True
      or row.get("returncode") != 0
      or row.get("success") is not (status == "success")
    ):
      raise RuntimeError(
        f"resume trial row {index} is invalid or incomplete; "
        "start a new output directory after resolving the trial error"
      )


def _validate_review_artifacts(
  trial: Path,
  seed: int,
  manifest: dict,
  *,
  require_comparison: bool,
) -> dict:
  review = evaluation_artifact_directory(trial)
  missing = [
    name for name in REVIEW_FILES
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if missing:
    raise RuntimeError(f"seed={seed}: review missing or empty: {missing}")
  review_meta = json.loads((review / "review.json").read_text(encoding="utf-8"))
  if review_meta.get("completed") is not True or review_meta.get(
    "second_camera"
  ) != "global":
    raise RuntimeError(f"seed={seed}: incomplete review video")
  expected_model_views = ["head"]
  if "right_wrist" in manifest["observation_contract"].get("cameras", []):
    expected_model_views.append("right_wrist")
  if review_meta.get("model_input_cameras_displayed") != expected_model_views:
    raise RuntimeError(f"seed={seed}: review omitted a model camera")
  if not require_comparison:
    return review_meta

  missing_comparison = [
    name for name in COMPARISON_FILES
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if missing_comparison:
    raise RuntimeError(
      f"seed={seed}: reference comparison missing or empty: {missing_comparison}"
    )
  comparison = review_meta.get("comparison_plots")
  if not isinstance(comparison, dict) or comparison.get("status") != "ok":
    raise RuntimeError(
      f"seed={seed}: reference comparison did not complete: {comparison!r}"
    )
  comparison_meta = json.loads(
    (review / "evaluation_comparison.json").read_text(encoding="utf-8")
  )
  if not isinstance(comparison_meta, dict) or comparison_meta.get("status") != "ok":
    raise RuntimeError(
      f"seed={seed}: comparison metadata did not report status=ok"
    )
  return review_meta


def _egotouch_runtime(args, manifest: dict) -> tuple[Path, Path, Path, Path]:
  observation = manifest.get("observation_contract")
  expected_observation = {
    "cameras": EGOTOUCH_CAMERAS,
    "image_shape_hwc": [240, 320, 3],
    "view_slots": EGOTOUCH_VIEW_SLOTS,
    "view_mask": EGOTOUCH_VIEW_MASK,
    "tactile_sent_to_model": False,
  }
  if observation != expected_observation:
    raise RuntimeError(
      f"EgoTouch observation_contract mismatch: {observation!r}"
    )
  expected = {
    "schema": "pickplace_egotouch_deployment_v1",
    "task": "pick-place",
    "model_family": "EgoTouch",
    "prediction_horizon": 50,
    "action_dim": 108,
    "control_hz": 30,
    "checkpoint_schema": "egotouch-external-sim-candidate-v1",
    "model_weights": "ema",
  }
  mismatch = {
    key: (value, manifest.get(key))
    for key, value in expected.items()
    if manifest.get(key) != value
  }
  if mismatch:
    raise RuntimeError(f"EgoTouch deployment contract mismatch: {mismatch}")
  epoch = manifest.get("checkpoint_epoch")
  if type(epoch) is not int or epoch <= 0:
    raise RuntimeError("EgoTouch checkpoint_epoch must be a positive integer")
  paths = {}
  for key, requested in (
    ("checkpoint_path", None),
    ("binding_path", None),
    ("model_python", args.model_python),
    ("model_project_path", args.model_project),
  ):
    declared = manifest.get(key)
    if not isinstance(declared, str) or not declared:
      raise RuntimeError(f"EgoTouch deployment is missing {key}")
    path = Path(declared).expanduser().resolve(strict=True)
    if requested is not None and requested.expanduser().resolve(strict=True) != path:
      raise RuntimeError(f"--{key.replace('_path', '').replace('_', '-')} differs from frozen deployment")
    paths[key] = path
  if not paths["checkpoint_path"].is_file() or not paths["binding_path"].is_file():
    raise RuntimeError("EgoTouch checkpoint and binding must be regular files")
  if paths["checkpoint_path"].name != f"epoch_{epoch:04d}.pt":
    raise RuntimeError("EgoTouch checkpoint filename differs from checkpoint_epoch")
  if not paths["model_python"].is_file() or not paths["model_project_path"].is_dir():
    raise RuntimeError("EgoTouch model Python/project paths have the wrong type")
  for path_key, digest_key in (
    ("checkpoint_path", "checkpoint_sha256"),
    ("binding_path", "binding_file_sha256"),
    ("model_python", "model_python_sha256"),
  ):
    actual = _sha256_file(paths[path_key])
    if actual != manifest.get(digest_key):
      raise RuntimeError(
        f"EgoTouch {path_key} SHA-256 mismatch: {actual} != {manifest.get(digest_key)}"
      )
  declared_sources = manifest.get("model_source_files")
  if not isinstance(declared_sources, list) or not declared_sources:
    raise RuntimeError("EgoTouch deployment is missing model_source_files")
  actual_sources = []
  for declared in declared_sources:
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
      raise RuntimeError("invalid EgoTouch model source entry")
    source = (paths["model_project_path"] / declared["path"]).resolve(strict=True)
    if not source.is_relative_to(paths["model_project_path"]) or not source.is_file():
      raise RuntimeError(f"EgoTouch model source escapes project: {source}")
    actual_sources.append({
      "path": declared["path"],
      "bytes": source.stat().st_size,
      "sha256": _sha256_file(source),
    })
  if actual_sources != declared_sources:
    raise RuntimeError("EgoTouch model source files changed after deployment freeze")
  if _canonical_hash(actual_sources) != manifest.get("model_source_sha256"):
    raise RuntimeError("EgoTouch model source aggregate SHA-256 mismatch")
  return (
    paths["checkpoint_path"], paths["binding_path"],
    paths["model_python"], paths["model_project_path"],
  )


def _render_environment(render_backend: str) -> dict[str, str]:
  if render_backend not in RENDER_BACKENDS:
    raise ValueError(f"unsupported render backend: {render_backend}")
  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
  }
  if render_backend == "software-osmesa":
    environment.update({
      "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
      "KAIHAND_RENDER_BACKEND": "software",
      "LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
    })
    environment.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
  else:
    if not EGL_VENDOR.is_file():
      raise FileNotFoundError(
        f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}"
      )
    environment.update({
      "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
      "KAIHAND_RENDER_BACKEND": "hardware",
      "__EGL_VENDOR_LIBRARY_FILENAMES": str(EGL_VENDOR),
    })
    environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
    environment.pop("GALLIUM_DRIVER", None)
  paths = [
    str(ROOT / "src"),
    "/cpfs_infra/user/chenxianchi/code/openpi/packages/openpi-client/src",
  ]
  if environment.get("PYTHONPATH"):
    paths.append(environment["PYTHONPATH"])
  environment["PYTHONPATH"] = os.pathsep.join(paths)
  for key in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
    "all_proxy",
  ):
    environment.pop(key, None)
  return environment


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server")
  parser.add_argument("--model-python", type=Path)
  parser.add_argument("--model-project", type=Path)
  parser.add_argument("--model-device", choices=("cpu", "cuda:0"), default="cuda:0")
  parser.add_argument(
    "--render-backend", choices=RENDER_BACKENDS, default="hardware-egl",
    help=(
      "MuJoCo rendering contract: hardware-egl (default, NVIDIA EGL) or "
      "software-osmesa (CPU llvmpipe)"
    ),
  )
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--seeds", nargs="+", required=True, type=int)
  parser.add_argument("--execute-steps", type=int)
  parser.add_argument(
    "--hand-ik-orientation-weights",
    type=float,
    nargs=5,
    default=EGOTOUCH_HAND_IK_ORIENTATION_WEIGHTS,
    metavar="WEIGHT",
    help="EgoTouch thumb/index/middle/ring/pinky soft orientation weights in m/rad",
  )
  parser.add_argument(
    "--hold-failed-hand-ik",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="EgoTouch only: hold the previous hand command on a sub-threshold IK miss",
  )
  parser.add_argument("--max-sim-seconds", type=float, default=90)
  parser.add_argument(
    "--guard-mode", choices=("observe", "strict"), default="observe"
  )
  parser.add_argument("--trial-wall-limit", type=float, default=1800)
  parser.add_argument("--video-count", type=int, default=20)
  parser.add_argument("--record-fps", choices=(5, 10), type=int, default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
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
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args(argv)
  if args.reference_dataset is not None:
    args.reference_dataset = args.reference_dataset.expanduser().resolve()
  if args.allow_source_change and not args.resume:
    parser.error("--allow-source-change requires --resume")
  if args.dry_run and args.resume:
    parser.error("--dry-run cannot be combined with --resume")
  if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
    parser.error("seeds must be distinct nonnegative integers")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be in [0, number of seeds]")
  if any(
    not math.isfinite(value) or value <= 0
    for value in (args.max_sim_seconds, args.trial_wall_limit)
  ):
    parser.error("time limits must be finite and positive")
  if any(
    not isinstance(weight, float)
    or not (weight >= 0.0 and weight < float("inf"))
    for weight in args.hand_ik_orientation_weights
  ):
    parser.error("hand-ik-orientation-weights must be finite and non-negative")
  manifest_path = args.deployment_manifest.resolve(strict=True)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("task") != "pick-place" or manifest.get("model_family") not in RUNNERS:
    raise RuntimeError("expected frozen PickPlace EgoSteer, pi0.5, or EgoTouch manifest")
  family = manifest["model_family"]
  if args.execute_steps is None:
    if family == "EgoTouch":
      args.execute_steps = EGOTOUCH_EXECUTE_STEPS
    else:
      parser.error("--execute-steps is required for EgoSteer and pi0.5")
  egotouch_paths = None
  if family == "EgoTouch":
    if args.server is not None:
      parser.error("EgoTouch is an in-process model and does not accept --server")
    egotouch_paths = _egotouch_runtime(args, manifest)
  else:
    if not args.server:
      parser.error("--server is required for EgoSteer and pi0.5")
    if args.model_python is not None or args.model_project is not None:
      parser.error("--model-python/--model-project are only applicable to EgoTouch")
    if args.model_device != "cuda:0":
      parser.error("--model-device is only applicable to EgoTouch")
  if args.diagnostic_relax_ik and family != "EgoSteer":
    parser.error("--diagnostic-relax-ik is only applicable to EgoSteer")
  if family == "EgoTouch" and args.execute_steps != EGOTOUCH_EXECUTE_STEPS:
    parser.error(
      "PickPlace EgoTouch requires --execute-steps=5; H50 is the prediction "
      "horizon, not the open-loop execution length"
    )
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
    "reference_dataset": (
      None if args.reference_dataset is None else str(args.reference_dataset)
    ),
    "reference_episode_index": args.reference_episode_index,
    "model_family": family, "prediction_horizon": horizon,
    "action_dim": manifest["action_dim"], "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
    "seeds": args.seeds, "execute_steps": args.execute_steps,
    "control_hz": 30, "replan_period_s": args.execute_steps / 30,
    "hand_ik_orientation_weights_m_per_rad": (
      list(args.hand_ik_orientation_weights) if family == "EgoTouch" else None
    ),
    "hold_failed_hand_ik": (
      args.hold_failed_hand_ik if family == "EgoTouch" else None
    ),
    "render_backend": args.render_backend,
    "guard_mode": args.guard_mode,
    "max_sim_seconds": args.max_sim_seconds, "trial_wall_limit": args.trial_wall_limit,
    "randomization": {"xy_jitter_m": 0.01, "yaw_jitter_rad": 0.05},
    "success_rule": "cylinder in box, released and stable for dataset terminal duration",
    "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
    "diagnostic_only": args.diagnostic_relax_ik or args.guard_mode == "observe",
    "formal_metrics_valid": not args.diagnostic_relax_ik and args.guard_mode == "strict",
    "kinematic_reachability_guards_enabled": (
      True if family == "EgoTouch"
      else not args.diagnostic_relax_ik if family == "EgoSteer"
      else None
    ),
    "model_runtime": (
      {
        "model_python": str(egotouch_paths[2]),
        "model_project": str(egotouch_paths[3]),
        "model_device": args.model_device,
        "binding_path": str(egotouch_paths[1]),
        "binding_sha256": manifest["binding_sha256"],
        "binding_file_sha256": manifest["binding_file_sha256"],
      }
      if egotouch_paths is not None else None
    ),
    "review": {
      "count": args.video_count,
      "fps": args.record_fps,
      "resolution": [1920, 1080],
      "second_camera": "global",
      "tactile_heatmaps": True,
      "time_series_displayed": False,
    },
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
    _require_valid_resume_rows(rows, family)
    for index, seed in enumerate(completed_seeds):
      trial = output / f"seed_{seed:03d}"
      if not (trial / "summary.json").is_file():
        raise RuntimeError(f"resume trial summary missing: {trial}")
      if index < args.video_count:
        _validate_review_artifacts(
          trial,
          seed,
          manifest,
          require_comparison=args.reference_dataset is not None,
        )
    print(
      f"RESUME {output}: completed={len(rows)}/{len(args.seeds)} "
      f"next_seed={args.seeds[len(rows)] if len(rows) < len(args.seeds) else 'none'}",
      flush=True,
    )
  else:
    if args.dry_run:
      print(json.dumps({"protocol": desired_protocol}, ensure_ascii=False, indent=2))
      return 0
    output.mkdir(parents=True, exist_ok=False)
    protocol = desired_protocol
    (output / "protocol.json").write_text(
      json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
  environment = _render_environment(args.render_backend)

  started = time.monotonic()
  for index, seed in enumerate(args.seeds):
    if index < len(rows):
      continue
    if fingerprints(runner) != frozen:
      raise RuntimeError("evaluation source changed during batch")
    if egotouch_paths is not None:
      _egotouch_runtime(args, manifest)
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable, str(runner),
      "--deployment-manifest", str(manifest_path), "--output-dir", str(trial),
      "--seed", str(seed), "--execute-steps", str(args.execute_steps),
      "--max-sim-seconds", str(args.max_sim_seconds), "--record-fps", str(args.record_fps),
      "--guard-mode", args.guard_mode,
      "--record" if index < args.video_count else "--no-record",
    ]
    if egotouch_paths is None:
      command[2:2] = ["--server", args.server]
    else:
      command.extend((
        "--checkpoint", str(egotouch_paths[0]),
        "--binding", str(egotouch_paths[1]),
        "--model-python", str(egotouch_paths[2]),
        "--model-project", str(egotouch_paths[3]),
        "--model-device", args.model_device,
      ))
      command.extend((
        "--hand-ik-orientation-weights",
        *(str(weight) for weight in args.hand_ik_orientation_weights),
        "--hold-failed-hand-ik" if args.hold_failed_hand_ik
        else "--no-hold-failed-hand-ik",
      ))
    if args.reference_dataset is not None:
      command.extend((
        "--reference-dataset", str(args.reference_dataset),
        "--reference-episode-index", str(args.reference_episode_index),
      ))
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
      "guard_mode": args.guard_mode,
      "diagnostic_only": args.diagnostic_relax_ik or args.guard_mode == "observe",
    }
    if family == "EgoTouch":
      expected.update({
        "hand_ik_orientation_weights_m_per_rad": list(
          args.hand_ik_orientation_weights
        ),
        "hold_failed_hand_ik": args.hold_failed_hand_ik,
      })
    mismatches = {key: (value, summary.get(key)) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
      raise RuntimeError(f"seed={seed}: summary identity mismatch: {mismatches}")
    has_record_error = any(
      value for key, value in summary.items() if key.endswith("record_error")
    )
    if index < args.video_count and not has_record_error:
      _validate_review_artifacts(
        trial,
        seed,
        manifest,
        require_comparison=args.reference_dataset is not None,
      )
    status = summary.get("status")
    success = (
      returncode == 0 and status == "success"
      and summary.get("evaluation", {}).get("success") is True
    )
    policy_failure = summary.get("policy_failure")
    policy_rejection = (
      family == "EgoTouch"
      and status == EGOTOUCH_POLICY_REJECTION_STATUS
      and summary.get("evaluation", {}).get("success") is False
      and summary.get("error") is None
      and summary.get("worker_shutdown_error") is None
      and summary.get("record_error") is None
      and _valid_egotouch_policy_failure(policy_failure)
    )
    valid = returncode == 0 and not has_record_error and (
      status in ("success", "task_not_completed") or policy_rejection
    )
    row = {
      "seed": seed, "status": status, "success": success, "valid_trial": valid,
      "task_success": bool(summary.get("task_success", False)),
      "returncode": returncode, "sim_seconds": summary.get("sim_seconds"),
      "wall_seconds": time.monotonic() - trial_started,
      "inference_requests": summary.get("stats", {}).get("requests", 0),
      "action_steps": summary.get("stats", {}).get("action_steps", 0),
      "error": summary.get("error"), "policy_failure": policy_failure,
      "record_error": summary.get("record_error"),
      "guard_mode": summary.get("guard_mode"),
      "guard_events": summary.get("guard_events", []),
      "stop_reason": summary.get("stop_reason"),
      "physics_valid": summary.get("physics_valid"),
      "formal_metrics_valid": summary.get("formal_metrics_valid"),
      "summary": str(summary_path.relative_to(output)),
    }
    rows.append(row)
    valid_count = sum(item["valid_trial"] for item in rows)
    successes = sum(item["success"] for item in rows)
    task_successes = sum(
      bool(item.get("task_success", item["success"])) for item in rows
    )
    aggregate = {
      **{key: protocol[key] for key in ("task", "deployment_id", "checkpoint_path", "checkpoint_sha256", "model_family", "prediction_horizon", "action_dim", "execute_steps", "control_hz", "replan_period_s", "hand_ik_orientation_weights_m_per_rad", "hold_failed_hand_ik", "max_sim_seconds", "guard_mode", "diagnostic_only", "formal_metrics_valid")},
      "planned": len(args.seeds), "attempted": len(rows), "valid_trials": valid_count,
      "successes": successes, "success_rate": successes / valid_count if valid_count else None,
      "task_successes": task_successes,
      "complete": len(rows) == len(args.seeds) and valid_count == len(args.seeds),
      "status_counts": dict(Counter(item["status"] for item in rows)),
      "stop_reason_counts": dict(Counter(item["stop_reason"] for item in rows)),
      "guard_event_counts": dict(Counter(
        event.get("kind")
        for item in rows for event in item.get("guard_events", [])
        if event.get("kind")
      )),
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
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
