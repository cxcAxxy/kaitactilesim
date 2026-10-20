#!/usr/bin/env python3
"""Run one manifest-bound PickPlace EgoTouch checkpoint in the local workcell."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.egosteer_adapter import (
  CV_FROM_MUJOCO_CAMERA,
  _base_site_name,
  _fingertip_site_name,
  _named_site_pose,
)
from kaihand_tactile_env.shared.evaluation_guards import (
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
)
from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
from kaihand_tactile_env.shared.policy_cameras import image_shape_hwc
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.pick_place.task import cylinder_is_in_box
from run_egosteer_policy import (
  DEFAULT_FREE_CLOSE_CUTOFF_M,
  DEFAULT_FREE_CLOSE_FORCE_LIMIT,
  DEFAULT_FREE_CLOSE_MIN_CLOSURE,
  AdaptiveFreeCloseForce,
  AutomaticGraspStabilizer,
  PolicyActionRejected,
  RolloutStats,
  StablePlacementDetector,
  _command_action_step,
)
from run_shared_task_egotouch_policy import (
  decode_policy_response,
  read_response,
  write_ik_diagnostics,
)

CONTROL_HZ = 30
MODEL_CAMERAS = ("head", "right_wrist")
VIEW_SLOTS = ("head", "left_wrist", "right_wrist")
VIEW_MASK = (True, False, True)
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# PickPlace A/B testing found that thumb pose resolves the harmful opposition
# branch, while constraining full SO(3) on the other four fingers prevents the
# small q3/q4 redistribution needed to wrap their tactile pads around the
# cylinder.  Keep their predicted rotations for diagnostics, but not the IK
# objective by default.
PICKPLACE_HAND_IK_ORIENTATION_WEIGHTS = (0.020, 0.0, 0.0, 0.0, 0.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--model-python", required=True)
  parser.add_argument("--model-project", type=Path, required=True)
  parser.add_argument("--model-device", default="cuda:0")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--binding", type=Path, required=True)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument(
    "--guard-mode", choices=("observe", "strict"), default="observe"
  )
  parser.add_argument("--response-timeout", type=float, default=600.0)
  parser.add_argument(
    "--record", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  parser.add_argument("--review-render-width", type=int, default=640)
  parser.add_argument("--review-render-height", type=int, default=480)
  parser.add_argument(
    "--review-second-camera",
    choices=("global", "right_wrist", "overhead"),
    default="global",
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--max-wrist-jump", type=float, default=0.15)
  parser.add_argument("--max-arm-position-error", type=float, default=0.03)
  parser.add_argument("--max-arm-orientation-error", type=float, default=0.30)
  parser.add_argument("--hand-ik-tolerance", type=float, default=0.0015)
  parser.add_argument("--max-hand-error", type=float, default=0.02)
  parser.add_argument(
    "--hand-ik-orientation-weights",
    type=float,
    nargs=5,
    default=PICKPLACE_HAND_IK_ORIENTATION_WEIGHTS,
    metavar="WEIGHT",
    help="thumb/index/middle/ring/pinky soft orientation weights in m/rad",
  )
  parser.add_argument(
    "--hold-failed-hand-ik",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="retain the previous hand command when fingertip position IK misses tolerance",
  )
  parser.add_argument("--strict-ik", action="store_true")
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0:
    parser.error("seed and max-requests must be nonnegative")
  if args.execute_steps != 5:
    parser.error("PickPlace EgoTouch must execute exactly 5 waypoints per replan")
  if (
    not np.isfinite(args.max_sim_seconds)
    or not np.isfinite(args.response_timeout)
    or args.max_sim_seconds <= 0.0
    or args.response_timeout <= 0.0
  ):
    parser.error("time limits must be finite and positive")
  if not args.model_device:
    parser.error("model-device cannot be empty")
  if args.reference_episode_index < 0:
    parser.error("reference-episode-index must be nonnegative")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  if min(
    args.max_wrist_jump,
    args.max_arm_position_error,
    args.max_arm_orientation_error,
    args.hand_ik_tolerance,
    args.max_hand_error,
  ) <= 0.0:
    parser.error("IK thresholds must be positive")
  if args.hand_ik_tolerance > args.max_hand_error:
    parser.error("hand-ik-tolerance cannot exceed max-hand-error")
  if any(
    not np.isfinite(weight) or weight < 0.0
    for weight in args.hand_ik_orientation_weights
  ):
    parser.error("hand-ik-orientation-weights must be finite and non-negative")
  args.control_side = "right"
  return args


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _binding_digest(binding: Mapping[str, Any]) -> str:
  return _canonical_hash(binding)


def _canonical_hash(value: Any) -> str:
  canonical = json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(canonical).hexdigest()


def _resolved_declared_path(value: Any, field: str) -> Path:
  if not isinstance(value, str) or not value:
    raise RuntimeError(f"deployment manifest {field} must be a path string")
  try:
    return Path(value).expanduser().resolve(strict=True)
  except OSError as error:
    raise RuntimeError(f"deployment manifest {field} is missing: {value}") from error


def _resolved_explicit_path(value: str | Path, option: str) -> Path:
  try:
    return Path(value).expanduser().resolve(strict=True)
  except OSError as error:
    raise RuntimeError(f"explicit {option} path is missing: {value}") from error


def _validate_model_runtime(
  deployment: Mapping[str, Any],
  model_python_path: str | Path,
  model_project_path: Path,
) -> tuple[Path, Path]:
  model_python = _resolved_explicit_path(model_python_path, "--model-python")
  model_project = _resolved_explicit_path(model_project_path, "--model-project")
  declared_python = _resolved_declared_path(
    deployment.get("model_python"), "model_python"
  )
  declared_project = _resolved_declared_path(
    deployment.get("model_project_path"), "model_project_path"
  )
  if model_python != declared_python:
    raise RuntimeError("explicit --model-python differs from deployment model_python")
  if model_project != declared_project:
    raise RuntimeError(
      "explicit --model-project differs from deployment model_project_path"
    )
  if not model_python.is_file() or not model_project.is_dir():
    raise RuntimeError("model Python/project paths have the wrong type")
  actual_python_sha256 = _sha256(model_python)
  if actual_python_sha256 != deployment.get("model_python_sha256"):
    raise RuntimeError(
      "model Python SHA-256 mismatch: "
      f"{actual_python_sha256} != {deployment.get('model_python_sha256')}"
    )

  declared_sources = deployment.get("model_source_files")
  if not isinstance(declared_sources, list) or not declared_sources:
    raise RuntimeError("deployment manifest is missing model_source_files")
  actual_sources = []
  for declared in declared_sources:
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
      raise RuntimeError("invalid EgoTouch model source entry")
    try:
      source = (model_project / declared["path"]).resolve(strict=True)
    except OSError as error:
      raise RuntimeError(
        f"declared EgoTouch model source is missing: {declared['path']}"
      ) from error
    if not source.is_relative_to(model_project) or not source.is_file():
      raise RuntimeError(f"EgoTouch model source escapes project: {source}")
    actual_sources.append({
      "path": declared["path"],
      "bytes": source.stat().st_size,
      "sha256": _sha256(source),
    })
  if actual_sources != declared_sources:
    raise RuntimeError("EgoTouch model source files changed after deployment freeze")
  if _canonical_hash(actual_sources) != deployment.get("model_source_sha256"):
    raise RuntimeError("EgoTouch model source aggregate SHA-256 mismatch")
  return model_python, model_project


def load_deployment(
  manifest_path: Path,
  checkpoint_path: Path,
  binding_path: Path,
  model_python_path: str | Path,
  model_project_path: Path,
) -> tuple[Path, dict[str, Any], Path, Path, Path, Path]:
  """Load and cryptographically bind a deployment to the explicit CLI files."""
  manifest_resolved = manifest_path.expanduser().resolve(strict=True)
  deployment = json.loads(manifest_resolved.read_text(encoding="utf-8"))
  if not isinstance(deployment, dict):
    raise RuntimeError("deployment manifest must contain a JSON object")
  required = (
    "schema",
    "task",
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "binding_path",
    "binding_sha256",
    "binding_file_sha256",
    "model_weights",
    "prediction_horizon",
    "action_dim",
    "observation_contract",
    "action_representation",
    "model_python",
    "model_python_sha256",
    "model_project_path",
    "model_source_sha256",
    "model_source_files",
  )
  missing = [name for name in required if name not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if deployment["schema"] != "pickplace_egotouch_deployment_v1":
    raise RuntimeError("expected pickplace_egotouch_deployment_v1 manifest")
  if deployment["task"] != "pick-place" or deployment["model_family"] != "EgoTouch":
    raise RuntimeError("runner requires a PickPlace EgoTouch deployment")
  if deployment["model_weights"] != "ema":
    raise RuntimeError("PickPlace external-simulation checkpoints must use EMA weights")

  checkpoint = checkpoint_path.expanduser().resolve(strict=True)
  binding = binding_path.expanduser().resolve(strict=True)
  if not checkpoint.is_file() or not binding.is_file():
    raise RuntimeError("checkpoint and binding must both be regular files")
  if _resolved_declared_path(deployment["checkpoint_path"], "checkpoint_path") != checkpoint:
    raise RuntimeError("explicit checkpoint differs from deployment checkpoint_path")
  if _resolved_declared_path(deployment["binding_path"], "binding_path") != binding:
    raise RuntimeError("explicit binding differs from deployment binding_path")
  if _sha256(checkpoint) != deployment["checkpoint_sha256"]:
    raise RuntimeError("checkpoint differs from deployment checkpoint_sha256")
  binding_payload = json.loads(binding.read_text(encoding="utf-8"))
  if not isinstance(binding_payload, dict):
    raise RuntimeError("EgoTouch binding must contain a JSON object")
  if _binding_digest(binding_payload) != deployment["binding_sha256"]:
    raise RuntimeError("binding differs from deployment binding_sha256")
  if _sha256(binding) != deployment["binding_file_sha256"]:
    raise RuntimeError("binding differs from deployment binding_file_sha256")
  model_python, model_project = _validate_model_runtime(
    deployment, model_python_path, model_project_path
  )

  observation = deployment["observation_contract"]
  if not isinstance(observation, dict):
    raise RuntimeError("observation_contract must be an object")
  if observation.get("cameras") != list(MODEL_CAMERAS):
    raise RuntimeError("PickPlace EgoTouch requires cameras=['head','right_wrist']")
  if observation.get("view_slots") != list(VIEW_SLOTS):
    raise RuntimeError("PickPlace EgoTouch requires head/left/right view slots")
  view_mask = observation.get("view_mask")
  if (
    not isinstance(view_mask, list)
    or any(type(value) is not bool for value in view_mask)
    or tuple(view_mask) != VIEW_MASK
  ):
    raise RuntimeError("PickPlace EgoTouch requires view_mask=[true,false,true]")
  if image_shape_hwc(observation) != (240, 320, 3):
    raise RuntimeError("PickPlace EgoTouch requires image_shape_hwc=[240,320,3]")
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("this PickPlace EgoTouch checkpoint is no-tactile")
  if isinstance(deployment["prediction_horizon"], bool) or not isinstance(
    deployment["prediction_horizon"], int
  ) or deployment["prediction_horizon"] <= 0:
    raise RuntimeError("prediction_horizon must be a positive integer")
  if isinstance(deployment["action_dim"], bool) or not isinstance(
    deployment["action_dim"], int
  ) or deployment["action_dim"] <= 0:
    raise RuntimeError("action_dim must be a positive integer")
  return (
    manifest_resolved,
    deployment,
    checkpoint,
    binding,
    model_python,
    model_project,
  )


def _requested_render_environment() -> dict[str, str | None]:
  """Capture the explicit GL selection independently of the detected backend."""
  return {
    name: os.environ.get(name)
    for name in (
      "MUJOCO_GL",
      "PYOPENGL_PLATFORM",
      "KAIHAND_RENDER_BACKEND",
      "MUJOCO_EGL_DEVICE_ID",
      "__EGL_VENDOR_LIBRARY_FILENAMES",
    )
  }


def _model_runtime_report(
  deployment: Mapping[str, Any],
  model_python: Path,
  model_project: Path,
  model_device: str,
) -> dict[str, str]:
  return {
    "model_python": str(model_python),
    "model_python_sha256": deployment["model_python_sha256"],
    "model_project_path": str(model_project),
    "model_source_sha256": deployment["model_source_sha256"],
    "model_device": model_device,
  }


def _worker_command(
  args: argparse.Namespace,
  checkpoint: Path,
  binding: Path,
  model_python: Path,
  model_project: Path,
) -> list[str]:
  return [
    str(model_python),
    str(Path(__file__).with_name("egotouch_model_worker.py")),
    "--project",
    str(model_project),
    "--checkpoint",
    str(checkpoint),
    "--binding",
    str(binding),
    "--device",
    args.model_device,
  ]


def _shutdown_worker(process: subprocess.Popen, timeout: float = 15.0) -> None:
  """Close the model worker and reject every observed nonzero exit status."""
  try:
    returncode = process.poll()
    if returncode is None:
      if process.stdin is None:
        raise RuntimeError("EgoTouch worker stdin is unavailable during shutdown")
      process.stdin.write('{"close":true}\n')
      process.stdin.flush()
      returncode = process.wait(timeout=timeout)
    if returncode != 0:
      raise RuntimeError(
        f"EgoTouch worker exited with return code {returncode}"
      )
  except (OSError, subprocess.TimeoutExpired):
    if process.poll() is None:
      process.kill()
      process.wait()
    raise


def _validate_worker_metadata(
  metadata: Mapping[str, Any], deployment: Mapping[str, Any], execute_steps: int
) -> int:
  expected = {
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "binding_sha256": deployment["binding_sha256"],
    "model_weights": deployment["model_weights"],
    "action_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "cameras": deployment["observation_contract"]["cameras"],
    "view_slots": deployment["observation_contract"]["view_slots"],
    "view_mask": deployment["observation_contract"]["view_mask"],
  }
  mismatches = {
    name: {"expected": value, "actual": metadata.get(name)}
    for name, value in expected.items()
    if metadata.get(name) != value
  }
  if mismatches:
    raise RuntimeError(f"EgoTouch worker deployment mismatch: {mismatches}")
  horizon = metadata["action_horizon"]
  if execute_steps != 5:
    raise RuntimeError("PickPlace EgoTouch execute_steps must be exactly 5")
  if horizon < execute_steps:
    raise RuntimeError(f"execute_steps={execute_steps} exceeds horizon={horizon}")
  return horizon


def _physical_hand_poses(simulation: ArmHandSimulation) -> tuple[np.ndarray, np.ndarray]:
  wrists = np.stack(
    [_named_site_pose(simulation, _base_site_name(side)) for side in SIDES]
  )
  tips = np.stack(
    [
      np.stack(
        [
          _named_site_pose(simulation, _fingertip_site_name(side, finger))
          for finger in FINGERS
        ]
      )
      for side in SIDES
    ]
  )
  return wrists, np.linalg.inv(wrists)[:, None] @ tips


def build_request(
  simulation: ArmHandSimulation,
  renderer: WorkcellRenderer,
  cameras: Mapping[str, CameraConfig],
  *,
  noise_seed: int,
  save_input: Path | None = None,
) -> dict[str, Any]:
  """Capture both RGBs without advancing physics and attach physical hand poses."""
  if tuple(cameras) != MODEL_CAMERAS:
    raise RuntimeError(f"model cameras must be {list(MODEL_CAMERAS)}")
  calibration = renderer.calibration(simulation.data, cameras["head"])
  camera_to_world = calibration.world_from_camera @ CV_FROM_MUJOCO_CAMERA
  rgbs = {}
  for name, camera in cameras.items():
    rgb = np.asarray(renderer.capture(simulation.data, camera)["rgb"])
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
      raise RuntimeError(f"{name} RGB must be uint8 (240,320,3), got {rgb.dtype} {rgb.shape}")
    rgbs[name] = base64.b64encode(rgb.tobytes()).decode("ascii")
  wrists, tips_relative = _physical_hand_poses(simulation)
  request = {
    "rgbs": rgbs,
    "c2w": camera_to_world.tolist(),
    "wrists": wrists.tolist(),
    "tips_relative": tips_relative.tolist(),
    "noise_seed": int(noise_seed),
  }
  if save_input is not None:
    request["save_input"] = str(save_input.resolve())
  return request


def _review_identity(
  deployment: Mapping[str, Any], args: argparse.Namespace, report: Mapping[str, Any]
) -> dict[str, Any]:
  return {
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": report["deployment_manifest"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "binding_path": deployment["binding_path"],
    "binding_sha256": deployment["binding_sha256"],
    "model_family": "EgoTouch",
    "model_weights": deployment["model_weights"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "fingertip_execution": "position plus weighted distal-site orientation IK",
    "hand_ik_orientation_weights_m_per_rad": list(
      args.hand_ik_orientation_weights
    ),
    "hold_failed_hand_ik": args.hold_failed_hand_ik,
    "ik_diagnostics": "ik_diagnostics.jsonl",
    "seed": args.seed,
    "guard_mode": args.guard_mode,
    "guard_events": [],
    "stop_reason": None,
    "physics_valid": False,
    "diagnostic_only": args.guard_mode == "observe",
    "formal_metrics_valid": args.guard_mode == "strict",
    "task_success": False,
    "kinematic_reachability_guards_enabled": True,
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  if args.execute_steps != 5:
    raise RuntimeError(
      "PickPlace EgoTouch must execute exactly 5 waypoints per replan"
    )
  (
    deployment_path,
    deployment,
    checkpoint,
    binding,
    model_python,
    model_project,
  ) = load_deployment(
    args.deployment_manifest,
    args.checkpoint,
    args.binding,
    args.model_python,
    args.model_project,
  )
  if args.execute_steps > deployment["prediction_horizon"]:
    raise RuntimeError(
      f"--execute-steps={args.execute_steps} exceeds deployment horizon "
      f"{deployment['prediction_horizon']}"
    )
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  started = time.monotonic()
  report: dict[str, Any] = {
    "task": "pick-place",
    "controller": "direct-30hz-egotouch-world-taskspace-pickplace-v2",
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "binding_path": deployment["binding_path"],
    "binding_sha256": deployment["binding_sha256"],
    "model_family": "EgoTouch",
    "model_weights": deployment["model_weights"],
    "model_runtime": _model_runtime_report(
      deployment, model_python, model_project, args.model_device
    ),
    "requested_render_environment": _requested_render_environment(),
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "fingertip_execution": "position plus weighted distal-site orientation IK",
    "hand_ik_orientation_weights_m_per_rad": list(
      args.hand_ik_orientation_weights
    ),
    "hold_failed_hand_ik": args.hold_failed_hand_ik,
    "ik_diagnostics": "ik_diagnostics.jsonl",
    "seed": args.seed,
    "max_sim_seconds": args.max_sim_seconds,
    "initial_randomization": {
      "object_xy_jitter_m": 0.01,
      "object_yaw_jitter_rad": 0.05,
    },
    "grasp_aids": {
      "auto_grasp_stabilizer": True,
      "adaptive_free_close_force": True,
    },
    "tactile_sent_to_model": False,
    "guard_mode": args.guard_mode,
    "guard_events": [],
    "stop_reason": None,
    "physics_valid": False,
    "diagnostic_only": args.guard_mode == "observe",
    "formal_metrics_valid": args.guard_mode == "strict",
    "kinematic_reachability_guards_enabled": True,
    "expert_used": False,
  }
  stats = RolloutStats()
  simulation = process = worker_log = recorder = stabilizer = free_close = None
  control_tick = 0
  episode_start = 0.0
  failure: BaseException | None = None
  policy_rejection: dict[str, Any] | None = None
  try:
    worker_log = (output / "model_worker.log").open("x", encoding="utf-8")
    process = subprocess.Popen(
      _worker_command(
        args, checkpoint, binding, model_python, model_project
      ),
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=worker_log,
      text=True,
      bufsize=1,
    )
    ready = read_response(process, args.response_timeout).get("ready")
    if not isinstance(ready, dict):
      raise RuntimeError("EgoTouch worker did not return ready metadata")
    horizon = _validate_worker_metadata(ready, deployment, args.execute_steps)
    report["worker_metadata"] = ready
    report["prediction_horizon"] = horizon

    simulation = ArmHandSimulation(scene="pick-place")
    simulation.reset(
      seed=args.seed,
      object_xy_jitter=0.01,
      object_yaw_jitter=0.05,
      randomized_objects=("cylinder",),
    )
    configure_evaluation_guards(simulation, args.guard_mode)
    ensure_evaluation_physics(simulation)
    stats.maximum_object_height = float(simulation.object_pose("cylinder")[2])
    stabilizer = AutomaticGraspStabilizer(
      simulation, enabled=True, contact_steps=3, release_closure=0.35
    )
    free_close = AdaptiveFreeCloseForce(
      simulation,
      enabled=True,
      force_limit=DEFAULT_FREE_CLOSE_FORCE_LIMIT,
      distance_cutoff=DEFAULT_FREE_CLOSE_CUTOFF_M,
      minimum_closure=DEFAULT_FREE_CLOSE_MIN_CLOSURE,
    )
    placement = StablePlacementDetector(timestep=simulation.timestep)
    episode_start = float(simulation.data.time)
    cameras = {
      name: CameraConfig(
        name, width=320, height=240, rgb=True, depth=False, segmentation=False
      )
      for name in MODEL_CAMERAS
    }
    with (
      WorkcellRenderer(
        simulation.model,
        tuple(cameras.values()),
      ) as renderer,
      (output / "requests.jsonl").open("x", encoding="utf-8") as request_log,
      (output / "ik_diagnostics.jsonl").open(
        "x", encoding="utf-8"
      ) as ik_log,
    ):
      report["render_backend"] = renderer.backend_info
      if args.record:
        recorder = EvaluationVideo(
          simulation,
          output,
          fps=args.record_fps,
          width=args.review_width,
          height=args.review_height,
          render_width=args.review_render_width,
          render_height=args.review_render_height,
          second_camera=args.review_second_camera,
          include_model_wrist=args.review_second_camera != "right_wrist",
          comparison=True,
          reference_dataset=args.reference_dataset,
          reference_episode_index=args.reference_episode_index,
          heading="PICKPLACE EGOTOUCH MODEL EVALUATION",
          metadata=_review_identity(deployment, args, report),
        )
        recorder.capture(0, "initial")

      stop_requested = False
      while float(simulation.data.time) - episode_start < args.max_sim_seconds:
        if stats.stable_success:
          break
        if args.max_requests and stats.requests >= args.max_requests:
          break
        request = build_request(
          simulation,
          renderer,
          cameras,
          noise_seed=args.seed * 10000 + stats.requests,
          save_input=(
            output / "first_request.npz"
            if stats.requests == 0 and args.save_first_request
            else None
          ),
        )
        request_started = time.monotonic()
        assert process.stdin is not None
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        response = read_response(process, args.response_timeout)
        stats.requests += 1
        decoded = decode_policy_response(simulation, response, horizon)
        request_log.write(
          json.dumps(
            {
              "request": stats.requests,
              "sim_time": float(simulation.data.time) - episode_start,
              "round_trip_s": time.monotonic() - request_started,
              "inference_s": response.get("seconds"),
              "roundtrip_max_error": response.get("roundtrip_max_error"),
              "right_wrist_targets_world": (
                None if decoded is None
                else decoded.right.site_positions_world.tolist()
              ),
              "right_fingertip_target_poses_world": (
                None if decoded is None
                else decoded.right.fingertip_poses_world.tolist()
              ),
            },
            ensure_ascii=False,
          )
          + "\n"
        )
        request_log.flush()
        print(
          f"[EgoTouch] seed={args.seed} request={stats.requests} "
          f"sim={simulation.data.time - episode_start:.3f}s "
          f"inference={response.get('seconds', float('nan')):.3f}s",
          flush=True,
        )

        action_count = 1 if decoded is None else args.execute_steps
        for action_index in range(action_count):
          if float(simulation.data.time) - episode_start >= args.max_sim_seconds:
            break
          step_diagnostics: list[dict] = []
          try:
            if decoded is not None:
              _command_action_step(
                simulation,
                decoded,
                action_index,
                args,
                stats,
                diagnostics=step_diagnostics,
              )
          except PolicyActionRejected as error:
            policy_rejection = {
              "kind": "kinematic_reachability_guard",
              "guard": error.guard,
              "message": str(error),
              "request": stats.requests,
              "action_index": action_index,
              "sim_time": float(simulation.data.time) - episode_start,
              "action_executed": False,
            }
            report["policy_failure"] = policy_rejection
            stop_requested = True
            break
          finally:
            write_ik_diagnostics(
              ik_log,
              step_diagnostics,
              request=stats.requests,
              sim_time=float(simulation.data.time) - episode_start,
            )
          action_rejected = decoded is None or bool(
            getattr(simulation, "_evaluation_action_rejected", False)
          )
          if not action_rejected:
            stabilizer.after_command()
          control_tick += 1
          target_time = episode_start + control_tick / CONTROL_HZ
          while (
            simulation.data.time < target_time - 1.0e-12
            and float(simulation.data.time) - episode_start
            < args.max_sim_seconds - 1.0e-12
          ):
            free_close.before_physics_step(
              closure=stabilizer.closure(), grasp_active=stabilizer.active
            )
            before_time = float(simulation.data.time)
            simulation.step()
            ensure_evaluation_physics(simulation, before_time=before_time)
            stabilizer.after_physics_step()
            twist = simulation.object_twist("cylinder")
            if placement.update(
              linear_speed=float(np.linalg.norm(twist[:3])),
              angular_speed=float(np.linalg.norm(twist[3:])),
              placed_in_box=cylinder_is_in_box(simulation),
              grasp_active=stabilizer.active,
            ):
              stats.stable_success = True
              stop_requested = True
              break
          stats.action_steps += 1
          stats.maximum_object_height = max(
            stats.maximum_object_height,
            float(simulation.object_pose("cylinder")[2]),
          )
          if recorder is not None:
            recorder.capture(
              control_tick, "placed" if stats.stable_success else "policy"
            )
          if stop_requested:
            break
          if action_rejected:
            break
        if stop_requested:
          break

      if policy_rejection is not None:
        report["status"] = "policy_action_rejected"
      else:
        report["status"] = (
          "success" if stats.stable_success else "task_not_completed"
        )
      # Capture the terminal pose while the model-input renderer's GL context
      # is still alive.  OSMesa does not reliably preserve a second renderer
      # after another context is destroyed; the finally-block call below then
      # becomes a same-timestamp no-op while still covering exceptional exits.
      if recorder is not None:
        recorder.capture(
          control_tick,
          "placed" if stats.stable_success else "terminal",
          force=True,
        )
  except BaseException as error:
    failure = error
    report.update(status="error", error=f"{type(error).__name__}: {error}")
  finally:
    if process is not None:
      try:
        _shutdown_worker(process)
      except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        report["worker_shutdown_error"] = f"{type(error).__name__}: {error}"
        if failure is None:
          failure = error
          report.update(status="error", error=report["worker_shutdown_error"])
    if worker_log is not None:
      worker_log.close()
    if simulation is not None:
      guard_report = evaluation_guard_report(
        simulation, status=report.get("status"), error=report.get("error")
      )
      report.update(guard_report)
      sim_time = float(simulation.data.time)
      report["sim_seconds"] = sim_time if np.isfinite(sim_time) else None
      report["evaluation"] = {
        "success": bool(stats.stable_success),
        "placed_in_box": (
          bool(cylinder_is_in_box(simulation))
          if guard_report["physics_valid"] else None
        ),
      }
      if report.get("status") == "policy_action_rejected":
        report["stop_reason"] = "guard"
      if (
        report.get("status") == "task_not_completed"
        and args.max_requests
        and stats.requests >= args.max_requests
        and report["sim_seconds"] is not None
        and report["sim_seconds"] - episode_start < args.max_sim_seconds
      ):
        report["stop_reason"] = "request_limit"
    else:
      report["sim_seconds"] = None
      report["evaluation"] = {"success": False}
    report.update(stats=asdict(stats), wall_seconds=time.monotonic() - started)
    report["task_success"] = bool(stats.stable_success)
    if recorder is not None:
      try:
        if not getattr(simulation, "_evaluation_physics_invalid", False):
          recorder.capture(
            control_tick,
            "placed" if stats.stable_success else "terminal",
            force=True,
          )
        review = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report["evaluation"],
          error=report.get("error"),
        )
        review.update(
          success=bool(stats.stable_success),
          task_success=report["task_success"],
          inference_requests=stats.requests,
          simulation_time_s=report["sim_seconds"],
          wall_time_s=report["wall_seconds"],
          guard_mode=report.get("guard_mode"),
          guard_events=report.get("guard_events", []),
          stop_reason=report.get("stop_reason"),
          physics_valid=report.get("physics_valid"),
          diagnostic_only=report.get("diagnostic_only"),
          formal_metrics_valid=report.get("formal_metrics_valid"),
        )
        if policy_rejection is not None:
          review["policy_failure"] = policy_rejection
        (output / "review.json").write_text(
          json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["video"] = review
      except BaseException as error:
        report["record_error"] = f"{type(error).__name__}: {error}"
        report["stop_reason"] = "fatal_error"
        report["formal_metrics_valid"] = False
        if failure is None:
          failure = error
          report.update(status="error", error=report["record_error"])
    if free_close is not None:
      free_close.close()
    if stabilizer is not None:
      stabilizer.close()
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
  if failure is not None:
    raise failure
  return report


if __name__ == "__main__":
  run(parse_args())
