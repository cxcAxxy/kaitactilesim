#!/usr/bin/env python3
"""Run one local EgoTouch trial for Bulb, RAM, or Vase."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import selectors
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.egosteer_adapter import (
  CV_FROM_MUJOCO_CAMERA,
  ActionTargets,
  SideActionTargets,
  _base_site_name,
  _fingertip_site_name,
  _named_site_pose,
  _quaternion_wxyz_from_rotation,
)
from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  policy_camera_names,
)
from kaihand_tactile_env.shared.policy_tasks import (
  TASK_INSTRUCTIONS,
  create_task_policy_adapter,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from run_egosteer_policy import RolloutStats, _command_action_step

CONTROL_HZ = 30
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TASKS = tuple(TASK_INSTRUCTIONS)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=TASKS, required=True)
  parser.add_argument("--model-python", required=True)
  parser.add_argument("--model-project", type=Path, required=True)
  parser.add_argument("--snapshot", type=Path)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--response-timeout", type=float, default=180.0)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
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
  parser.add_argument("--strict-ik", action="store_true")
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  if args.max_sim_seconds <= 0 or args.response_timeout <= 0:
    parser.error("time limits must be positive")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  args.control_side = "right"
  return args


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_deployment(path: Path, task: str, snapshot: Path | None):
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
  if deployment["task"] != task or str(deployment["model_family"]).lower() != "egotouch":
    raise RuntimeError(f"runner requires a {task} EgoTouch deployment")
  observation = deployment["observation_contract"]
  if policy_camera_names(observation) != ("head",):
    raise RuntimeError(
      "the current EgoTouch worker is a single-RGB checkpoint and requires cameras=['head']"
    )
  if image_shape_hwc(observation) != (240, 320, 3):
    raise RuntimeError("the current EgoTouch worker requires image_shape_hwc=[240,320,3]")
  if observation.get("tactile_sent_to_model", False):
    raise RuntimeError("the current EgoTouch worker was trained without tactile input")

  declared = Path(deployment["checkpoint_path"]).expanduser()
  declared_snapshot = declared.parent if declared.name == "best.pt" else declared
  selected_snapshot = (
    snapshot.expanduser().resolve(strict=True)
    if snapshot is not None
    else declared_snapshot.resolve(strict=True)
  )
  checkpoint = selected_snapshot / "best.pt"
  if not checkpoint.is_file():
    raise RuntimeError(f"EgoTouch checkpoint is missing: {checkpoint}")
  if _sha256(checkpoint) != deployment["checkpoint_sha256"]:
    raise RuntimeError("local EgoTouch best.pt differs from deployment checkpoint_sha256")
  return resolved, deployment, selected_snapshot


def read_response(process: subprocess.Popen, timeout: float) -> dict:
  with selectors.DefaultSelector() as selector:
    selector.register(process.stdout, selectors.EVENT_READ)
    if not selector.select(timeout=timeout):
      raise TimeoutError(f"EgoTouch worker response exceeded {timeout:g} seconds")
  line = process.stdout.readline()
  if not line:
    raise RuntimeError("EgoTouch worker exited; see model_worker.log")
  response = json.loads(line)
  if not isinstance(response, dict):
    raise RuntimeError("EgoTouch worker response must be a JSON object")
  return response


def decode_world_targets(simulation, response: dict) -> ActionTargets:
  wrists = np.asarray(response["wrists_world"], dtype=np.float64)
  tips = np.asarray(response["tips_world"], dtype=np.float64)
  if wrists.ndim != 4 or wrists.shape[1:] != (2, 4, 4):
    raise RuntimeError(f"invalid EgoTouch wrist output shape: {wrists.shape}")
  if tips.shape != (wrists.shape[0], 2, 5, 4, 4):
    raise RuntimeError(f"invalid EgoTouch fingertip output shape: {tips.shape}")
  if not np.isfinite(wrists).all() or not np.isfinite(tips).all():
    raise RuntimeError("EgoTouch output contains nonfinite values")
  result = {}
  for side_index, side in enumerate(SIDES):
    xyz, rotation = simulation.current_pose_matrix(side)
    control_world = np.eye(4)
    control_world[:3, :3], control_world[:3, 3] = rotation, xyz
    base_world = _named_site_pose(simulation, _base_site_name(side))
    base_from_control = np.linalg.inv(base_world) @ control_world
    control_targets = wrists[:, side_index] @ base_from_control
    rotations = control_targets[:, :3, :3]
    result[side] = SideActionTargets(
      control_targets[:, :3, 3],
      rotations,
      _quaternion_wxyz_from_rotation(rotations),
      tips[:, side_index, :, :3, 3],
    )
  return ActionTargets(**result)


def _worker_metadata(metadata: dict, deployment: dict, execute_steps: int) -> int:
  horizon = metadata.get("action_horizon")
  expected = {
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "action_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
  }
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"EgoTouch worker deployment mismatch: {mismatches}")
  if not isinstance(horizon, int) or not 1 <= execute_steps <= horizon:
    raise RuntimeError(f"execute_steps must be in [1, {horizon}]")
  return horizon


def run(args) -> dict:
  deployment_path, deployment, snapshot = load_deployment(
    args.deployment_manifest, args.task, args.snapshot
  )
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  instruction = deployment.get("instruction", TASK_INSTRUCTIONS[args.task])
  report = {
    "task": args.task,
    "controller": "direct-30hz-egotouch-world-taskspace-v1",
    "model_family": deployment["model_family"],
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "instruction": instruction,
    "seed": args.seed,
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "max_sim_seconds": args.max_sim_seconds,
    "expert_used": False,
  }
  stats = RolloutStats()
  started = time.monotonic()
  adapter = recorder = process = None
  worker_log = None
  control_tick = 0
  run_error = None
  try:
    worker_log = (output / "model_worker.log").open("x", encoding="utf-8")
    process = subprocess.Popen(
      [
        args.model_python,
        str(Path(__file__).with_name("egotouch_model_worker.py")),
        "--project",
        str(args.model_project.expanduser().resolve(strict=True)),
        "--snapshot",
        str(snapshot),
      ],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=worker_log,
      text=True,
      bufsize=1,
    )
    metadata = read_response(process, args.response_timeout).get("ready")
    if not isinstance(metadata, dict):
      raise RuntimeError("EgoTouch worker did not return ready metadata")
    horizon = _worker_metadata(metadata, deployment, args.execute_steps)
    report["worker_metadata"] = metadata
    report["prediction_horizon"] = horizon

    adapter = create_task_policy_adapter(args.task, args.seed)
    simulation = adapter.simulation
    camera = CameraConfig(
      "head", width=320, height=240, rgb=True, depth=False, segmentation=False
    )
    with WorkcellRenderer(
      simulation.model, (camera,), shadows=False
    ) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
      report["render_backend"] = renderer.backend_info
      if args.record:
        recorder = EvaluationVideo(
          simulation,
          output / "review",
          fps=args.record_fps,
          width=args.review_width,
          height=args.review_height,
          render_width=args.review_render_width,
          render_height=args.review_render_height,
          second_camera=args.review_second_camera,
          metrics=adapter,
          heading=f"{args.task.upper()} EGOTOUCH MODEL EVALUATION",
          metadata={
            "deployment_id": deployment["deployment_id"],
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "observation_contract": deployment["observation_contract"],
            "action_representation": deployment["action_representation"],
            "seed": args.seed,
          },
        )
        recorder.capture(0, adapter.stage())

      while simulation.data.time < args.max_sim_seconds and not adapter.success:
        if args.max_requests and stats.requests >= args.max_requests:
          break
        calibration = renderer.calibration(simulation.data, camera)
        camera_to_world = (
          calibration.world_from_camera @ CV_FROM_MUJOCO_CAMERA
        )
        rgb = renderer.capture(simulation.data, camera)["rgb"]
        wrists = np.stack(
          [_named_site_pose(simulation, _base_site_name(side)) for side in SIDES]
        )
        tips = np.stack(
          [
            np.stack(
              [
                _named_site_pose(
                  simulation, _fingertip_site_name(side, finger)
                )
                for finger in FINGERS
              ]
            )
            for side in SIDES
          ]
        )
        tips_relative = np.linalg.inv(wrists)[:, None] @ tips
        request = {
          "rgb": base64.b64encode(rgb.tobytes()).decode(),
          "c2w": camera_to_world.tolist(),
          "wrists": wrists.tolist(),
          "tips_relative": tips_relative.tolist(),
          "noise_seed": args.seed * 10000 + stats.requests,
        }
        if stats.requests == 0 and args.save_first_request:
          request["save_input"] = str((output / "first_request.npz").resolve())
        request_started = time.monotonic()
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        response = read_response(process, args.response_timeout)
        stats.requests += 1
        decoded = decode_world_targets(simulation, response)
        if len(decoded.right.site_positions_world) != horizon:
          raise RuntimeError("EgoTouch decoded horizon differs from deployment")
        log.write(
          json.dumps(
            {
              "request": stats.requests,
              "sim_time": float(simulation.data.time),
              "stage": adapter.stage(),
              "round_trip_s": time.monotonic() - request_started,
              "inference_s": response.get("seconds"),
              "roundtrip_max_error": response.get("roundtrip_max_error"),
              "right_wrist_targets_world": (
                decoded.right.site_positions_world.tolist()
              ),
            },
            ensure_ascii=False,
          )
          + "\n"
        )
        log.flush()

        for action_index in range(args.execute_steps):
          if simulation.data.time >= args.max_sim_seconds or adapter.success:
            break
          _command_action_step(
            simulation, decoded, action_index, args, stats
          )
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while simulation.data.time < target_time - 1.0e-12:
            adapter.step()
            if adapter.success:
              stats.stable_success = True
              break
          stats.action_steps += 1
          stats.maximum_object_height = max(
            stats.maximum_object_height,
            float(simulation.object_pose(adapter.object_name)[2]),
          )
          if recorder is not None:
            recorder.capture(control_tick, adapter.stage())

      report["status"] = "success" if adapter.success else "task_not_completed"
      report["evaluation"] = adapter.report()
      stats.stable_success = adapter.success
  except Exception as error:
    run_error = f"{type(error).__name__}: {error}"
    report.update(status="error", error=run_error)
    raise
  finally:
    if process is not None and process.poll() is None:
      try:
        process.stdin.write('{"close":true}\n')
        process.stdin.flush()
        process.wait(timeout=15)
      except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()
    if worker_log is not None:
      worker_log.close()
    if adapter is not None:
      report["sim_seconds"] = float(adapter.simulation.data.time)
      report["evaluation"] = adapter.report()
      stats.stable_success = adapter.success
    report.update(stats=asdict(stats), wall_seconds=time.monotonic() - started)
    if recorder is not None:
      try:
        recorder.capture(control_tick, adapter.stage(), force=True)
        report["video"] = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=run_error,
        )
      except Exception as error:
        report["record_error"] = f"{type(error).__name__}: {error}"
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
  return report


if __name__ == "__main__":
  run(parse_args())
