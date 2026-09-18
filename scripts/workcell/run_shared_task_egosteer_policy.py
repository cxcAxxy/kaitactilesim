#!/usr/bin/env python3
"""Run one manifest-bound EgoSteer trial for Bulb, RAM, or Vase."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.egosteer_adapter import (
  camera_from_world_opencv,
  decode_action_targets,
  model_state_history,
  relative_to_absolute,
)
from kaihand_tactile_env.shared.egosteer_client import (
  EgoSteerPolicyClient,
  ObservationHistory,
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
from run_egosteer_policy import (
  RolloutStats,
  _append_model_observations,
  _command_action_step,
  _infer_while_servicing_viewer,
  _model_action_horizon,
  _select_model_observation,
  _validated_prediction,
)

CONTROL_HZ = 30
TASKS = tuple(TASK_INSTRUCTIONS)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=TASKS, required=True)
  parser.add_argument("--server", required=True)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--response-timeout", type=float, default=600.0)
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


def load_deployment(path: Path, task: str) -> tuple[Path, dict]:
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
  if deployment["task"] != task or str(deployment["model_family"]).lower() != "egosteer":
    raise RuntimeError(f"runner requires a {task} EgoSteer deployment")
  if int(deployment["action_dim"]) != 48:
    raise RuntimeError("EgoSteer shared-task runner requires action_dim=48")
  observation = deployment["observation_contract"]
  if observation.get("tactile_sent_to_model", False):
    raise RuntimeError("this runner does not silently omit declared tactile model input")
  policy_camera_names(observation)
  image_shape_hwc(observation)
  for key in ("image_history", "image_stride"):
    if int(observation.get(key, 0)) <= 0:
      raise RuntimeError(f"observation_contract.{key} must be positive")
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  horizon = _model_action_horizon(metadata)
  expected = {
    "task": deployment["task"],
    "deployment_id": deployment["deployment_id"],
    "model_family": deployment["model_family"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "action_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
  }
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"server deployment identity mismatch: {mismatches}")
  if execute_steps > horizon:
    raise RuntimeError(
      f"--execute-steps={execute_steps} exceeds server horizon {horizon}"
    )
  return horizon


async def run(args) -> dict:
  deployment_path, deployment = load_deployment(
    args.deployment_manifest, args.task
  )
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  instruction = deployment.get("instruction", TASK_INSTRUCTIONS[args.task])
  report = {
    "task": args.task,
    "controller": "direct-30hz-egosteer-taskspace-v1",
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
  started = time.monotonic()
  stats = RolloutStats()
  adapter = recorder = None
  control_tick = 0
  run_error = None
  try:
    async with EgoSteerPolicyClient(
      args.server, response_timeout=args.response_timeout
    ) as client:
      metadata = client.metadata or {}
      horizon = validate_server(metadata, deployment, args.execute_steps)
      report["server_metadata"] = metadata
      report["prediction_horizon"] = horizon

      adapter = create_task_policy_adapter(args.task, args.seed)
      simulation = adapter.simulation
      observation = deployment["observation_contract"]
      camera_names = policy_camera_names(observation)
      image_height, image_width, _ = image_shape_hwc(observation)
      cameras = {
        name: CameraConfig(
          name,
          width=image_width,
          height=image_height,
          rgb=True,
          depth=False,
          segmentation=False,
        )
        for name in camera_names
      }
      histories = {
        name: ObservationHistory(
          horizon=int(observation["image_history"]),
          stride=int(observation["image_stride"]),
        )
        for name in camera_names
      }
      with WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
        report["render_backend"] = renderer.backend_info
        calibrations = _append_model_observations(
          simulation, renderer, cameras, histories
        )
        report["camera_intrinsics"] = {
          name: calibrations[name].intrinsic.tolist() for name in camera_names
        }
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
            include_model_wrist=(
              "right_wrist" in camera_names
              and args.review_second_camera != "right_wrist"
            ),
            metrics=adapter,
            heading=f"{args.task.upper()} EGOSTEER MODEL EVALUATION",
            metadata={
              "server": args.server,
              "deployment_id": deployment["deployment_id"],
              "checkpoint_path": deployment["checkpoint_path"],
              "checkpoint_sha256": deployment["checkpoint_sha256"],
              "observation_contract": observation,
              "action_representation": deployment["action_representation"],
              "seed": args.seed,
            },
          )
          recorder.capture(0, adapter.stage())

        while simulation.data.time < args.max_sim_seconds and not adapter.success:
          if args.max_requests and stats.requests >= args.max_requests:
            break
          selected = _select_model_observation(
            camera_names, histories, calibrations
          )
          camera_from_world = camera_from_world_opencv(
            calibrations["head"].world_from_camera
          )
          states = model_state_history(selected.raw_states, camera_from_world)
          request_started = time.monotonic()
          response = await _infer_while_servicing_viewer(
            client,
            None,
            None,
            images=selected.images,
            states=states,
            camera_intrinsics=selected.camera_intrinsics,
            instruction=instruction,
            image_format="jpeg",
            jpeg_quality=95,
          )
          stats.requests += 1
          predicted = _validated_prediction(
            response.pred_actions,
            horizon=horizon,
            action_dim=int(deployment["action_dim"]),
          )
          decoded = decode_action_targets(
            simulation,
            relative_to_absolute(states[-1], predicted),
            camera_from_world,
          )
          row = {
            "request": stats.requests,
            "sim_time": float(simulation.data.time),
            "stage": adapter.stage(),
            "round_trip_s": time.monotonic() - request_started,
            "server_timing": response.server_timing,
            "right_wrist_targets_world": (
              decoded.right.site_positions_world.tolist()
            ),
          }
          log.write(json.dumps(row, ensure_ascii=False) + "\n")
          log.flush()
          if stats.requests == 1 and args.save_first_request:
            camera_fields = {
              f"{name}_images": selected.selected_images[name]
              for name in camera_names
            }
            np.savez_compressed(
              output / "first_request.npz",
              states=states,
              predicted_actions=predicted,
              **camera_fields,
            )

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
            calibrations = _append_model_observations(
              simulation, renderer, cameras, histories
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
  asyncio.run(run(parse_args()))
