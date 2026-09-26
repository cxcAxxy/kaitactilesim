#!/usr/bin/env python3
"""Run one manifest-bound pi0.5 trial for a shared household task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.policy_tasks import (
  TASK_INSTRUCTIONS,
  create_task_policy_adapter,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from run_usb_pi05_policy import (
  apply_action,
  joint_limits,
  right_joint_state,
  validated_actions,
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
  parser.add_argument(
    "--pre-roll-steps",
    type=int,
    default=0,
    help="execute dataset absolute actions before the first policy request",
  )
  parser.add_argument(
    "--pre-roll-actions",
    type=Path,
    help="parquet file containing absolute actions for the pre-roll",
  )
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
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
  args = parser.parse_args(argv)
  if (
    args.seed < 0
    or args.max_requests < 0
    or args.execute_steps <= 0
    or args.pre_roll_steps < 0
  ):
    parser.error(
      "seed/requests/pre-roll-steps must be nonnegative; execute-steps must be positive"
    )
  if args.pre_roll_steps and args.pre_roll_actions is None:
    parser.error("--pre-roll-actions is required when --pre-roll-steps is nonzero")
  if not np.isfinite(args.max_sim_seconds) or args.max_sim_seconds <= 0:
    parser.error("max-sim-seconds must be finite and positive")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  return args


def load_pre_roll_actions(path: Path | None, steps: int) -> np.ndarray:
  if steps == 0:
    return np.empty((0, 27), dtype=np.float32)
  if path is None:
    raise RuntimeError("pre-roll action path is required")
  import pyarrow.parquet as parquet

  resolved = path.expanduser().resolve(strict=True)
  table = parquet.read_table(resolved, columns=["action"])
  if table.num_rows < steps:
    raise RuntimeError(
      f"pre-roll action file has {table.num_rows} rows, need {steps}: {resolved}"
    )
  actions = np.asarray(
    [table["action"][index].as_py() for index in range(steps)], dtype=np.float32
  )
  if actions.shape != (steps, 27) or not np.isfinite(actions).all():
    raise RuntimeError(
      f"pre-roll actions must be finite with shape ({steps}, 27), got {actions.shape}"
    )
  return actions


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
    "joint_names",
    "observation_contract",
    "action_representation",
  )
  missing = [key for key in required if key not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  family = str(deployment["model_family"]).lower().replace(".", "")
  if deployment["task"] != task or family not in {"pi05", "π05"}:
    raise RuntimeError(f"runner requires a {task} pi0.5 deployment")
  if int(deployment["action_dim"]) != 27:
    raise RuntimeError("pi0.5 shared-task runner requires 27 absolute joint actions")
  if deployment["observation_contract"].get("tactile_sent_to_model", False):
    raise RuntimeError("this runner does not silently omit declared tactile model input")
  policy_camera_names(deployment["observation_contract"])
  image_shape_hwc(deployment["observation_contract"])
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": deployment["task"],
    "deployment_id": deployment["deployment_id"],
    "model_family": deployment["model_family"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "control_hz": deployment.get("control_hz", CONTROL_HZ),
    "joint_names": deployment["joint_names"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
  }
  if "model_action_dim" in deployment:
    expected["model_action_dim"] = deployment["model_action_dim"]
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"server deployment identity mismatch: {mismatches}")
  horizon = int(metadata["prediction_horizon"])
  if not 1 <= execute_steps <= horizon:
    raise RuntimeError(f"execute_steps must be in [1, {horizon}]")
  if int(metadata["control_hz"]) != CONTROL_HZ:
    raise RuntimeError(f"runner requires {CONTROL_HZ} Hz policy control")
  return horizon


def run(args) -> dict:
  from openpi_client import websocket_client_policy

  deployment_path, deployment = load_deployment(args.deployment_manifest, args.task)
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  instruction = deployment.get("instruction", TASK_INSTRUCTIONS[args.task])
  report = {
    "task": args.task,
    "controller": "direct-30hz-pi05-absolute-joint-v1",
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
    "pre_roll_steps": args.pre_roll_steps,
    "pre_roll_actions": (
      str(args.pre_roll_actions.expanduser().resolve())
      if args.pre_roll_actions is not None
      else None
    ),
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "max_sim_seconds": args.max_sim_seconds,
    "expert_used": False,
  }
  started = time.monotonic()
  client = adapter = recorder = None
  requests = action_steps = clipped_values = control_tick = 0
  pre_roll_action_steps = 0
  run_error = None
  try:
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    adapter = create_task_policy_adapter(args.task, args.seed)
    pre_roll_actions = load_pre_roll_actions(
      args.pre_roll_actions, args.pre_roll_steps
    )
    simulation = adapter.simulation
    joint_names = list(deployment["joint_names"])
    lower, upper = joint_limits(simulation, joint_names)
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
    with WorkcellRenderer(
      simulation.model, tuple(cameras.values()), shadows=False
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
          include_model_wrist=(
            "right_wrist" in camera_names
            and args.review_second_camera != "right_wrist"
          ),
          include_review_wrist=(
            "right_wrist" not in camera_names
            and args.review_second_camera != "right_wrist"
          ),
          comparison=True,
          reference_dataset=args.reference_dataset,
          reference_episode_index=args.reference_episode_index,
          metrics=adapter,
          heading=f"{args.task.upper()} pi0.5 MODEL EVALUATION",
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

      for action in pre_roll_actions:
        if simulation.data.time >= args.max_sim_seconds or adapter.success:
          break
        clipped_values += apply_action(
          simulation, joint_names, action, lower, upper
        )
        control_tick += 1
        target_time = control_tick / CONTROL_HZ
        while simulation.data.time < target_time - 1.0e-12:
          adapter.step()
          if adapter.success:
            break
        action_steps += 1
        pre_roll_action_steps += 1
        if recorder is not None:
          recorder.capture(control_tick, adapter.stage())

      while simulation.data.time < args.max_sim_seconds and not adapter.success:
        if args.max_requests and requests >= args.max_requests:
          break
        images = {
          name: renderer.capture(simulation.data, camera)["rgb"]
          for name, camera in cameras.items()
        }
        state = right_joint_state(simulation, joint_names)
        request_started = time.monotonic()
        response = client.infer(
          {
            **pi05_image_payload(images, observation),
            "state": state,
            "prompt": instruction,
          }
        )
        requests += 1
        actions = validated_actions(
          response,
          horizon=horizon,
          action_dim=int(deployment["action_dim"]),
        )
        row = {
          "request": requests,
          "sim_time": float(simulation.data.time),
          "stage": adapter.stage(),
          "round_trip_s": time.monotonic() - request_started,
          "server_timing": response.get("server_timing", {}),
          "state": state.tolist(),
          "predicted_action_min": np.min(actions, axis=0).tolist(),
          "predicted_action_max": np.max(actions, axis=0).tolist(),
        }
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        if requests == 1 and args.save_first_request:
          np.savez_compressed(
            output / "first_request.npz",
            **{f"{name}_image": image for name, image in images.items()},
            state=state,
            predicted_actions=actions,
          )
        for action in actions[: args.execute_steps]:
          if simulation.data.time >= args.max_sim_seconds or adapter.success:
            break
          clipped_values += apply_action(
            simulation, joint_names, action, lower, upper
          )
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while simulation.data.time < target_time - 1.0e-12:
            adapter.step()
            if adapter.success:
              break
          action_steps += 1
          if recorder is not None:
            recorder.capture(control_tick, adapter.stage())
      report["status"] = "success" if adapter.success else "task_not_completed"
      report["evaluation"] = adapter.report()
  except Exception as error:
    run_error = f"{type(error).__name__}: {error}"
    report.update(status="error", error=run_error)
    raise
  finally:
    if client is not None:
      try:
        client._ws.close()
      except Exception:
        pass
    if adapter is not None:
      report["sim_seconds"] = float(adapter.simulation.data.time)
      report["evaluation"] = adapter.report()
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "pre_roll_action_steps": pre_roll_action_steps,
      "clipped_action_values": clipped_values,
    }
    report["wall_seconds"] = time.monotonic() - started
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
