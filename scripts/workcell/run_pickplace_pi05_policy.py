#!/usr/bin/env python3
"""Run one manifest-bound PickPlace pi0.5 policy trial."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.evaluation_guards import (
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
  recover_nonfinite_action,
)
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.pick_place.task import cylinder_is_in_box
from run_egosteer_policy import (
  DEFAULT_FREE_CLOSE_CUTOFF_M,
  DEFAULT_FREE_CLOSE_FORCE_LIMIT,
  DEFAULT_FREE_CLOSE_MIN_CLOSURE,
  AdaptiveFreeCloseForce,
  AutomaticGraspStabilizer,
  StablePlacementDetector,
)
from run_usb_pi05_policy import (
  apply_action,
  joint_limits,
  right_joint_state,
  validated_actions,
)

CONTROL_HZ = 30
REVIEW_FILES = ("review.mp4", "review.json", "frames.jsonl", "first_frame.png", "last_frame.png")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", required=True)
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--execute-steps", required=True, type=int)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument(
    "--guard-mode", choices=("observe", "strict"), default="observe"
  )
  parser.add_argument("--object-xy-jitter", type=float, default=0.01)
  parser.add_argument("--object-yaw-jitter", type=float, default=0.05)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  args = parser.parse_args()
  if (
    args.seed < 0
    or args.execute_steps < 1
    or not np.isfinite(args.max_sim_seconds)
    or args.max_sim_seconds <= 0
  ):
    parser.error(
      "seed must be nonnegative; execute-steps must be positive; "
      "max-sim-seconds must be finite and positive"
    )
  if args.object_xy_jitter < 0 or args.object_yaw_jitter < 0:
    parser.error("randomization must be nonnegative")
  if args.review_width < 2 or args.review_height < 2 or args.review_width % 2 or args.review_height % 2:
    parser.error("review size must be positive and even")
  return args


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": "pick-place",
    "deployment_id": deployment["deployment_id"],
    "model_family": "pi0.5",
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_horizon": deployment["prediction_horizon"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "control_hz": CONTROL_HZ,
    "joint_names": deployment["joint_names"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
  }
  mismatches = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
  if mismatches:
    raise RuntimeError(f"pi0.5 server contract mismatch: {mismatches}")
  horizon = int(metadata["prediction_horizon"])
  if not 1 <= execute_steps <= horizon:
    raise RuntimeError(f"execute_steps={execute_steps} must be in [1,{horizon}]")
  policy_camera_names(metadata["observation_contract"])
  image_shape_hwc(metadata["observation_contract"])
  return horizon


def run(args: argparse.Namespace) -> dict:
  from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
  from openpi_client import websocket_client_policy

  deployment_path = args.deployment_manifest.resolve(strict=True)
  deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
  if deployment.get("schema") != "pickplace_pi05_deployment_v1":
    raise RuntimeError("expected PickPlace pi0.5 deployment")
  output = args.output_dir.resolve()
  output.mkdir(parents=True, exist_ok=False)
  started = time.monotonic()
  report = {
    "task": "pick-place",
    "controller": "direct-30hz-pi05-absolute-joint-with-demonstration-grasp-aids-v1",
    "seed": args.seed,
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "model_family": "pi0.5",
    "prediction_horizon": deployment["prediction_horizon"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "max_sim_seconds": args.max_sim_seconds,
    "initial_randomization": {"object_xy_jitter_m": args.object_xy_jitter, "object_yaw_jitter_rad": args.object_yaw_jitter},
    "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
    "guard_mode": args.guard_mode,
    "guard_events": [],
    "stop_reason": None,
    "physics_valid": False,
    "diagnostic_only": args.guard_mode == "observe",
    "formal_metrics_valid": args.guard_mode == "strict",
    "task_success": False,
  }
  client = simulation = recorder = stabilizer = free_close_force = detector = None
  requests = action_steps = clipped_values = control_tick = 0
  run_error = None
  record_failure = None
  try:
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    simulation = ArmHandSimulation(scene="pick-place")
    simulation.reset(
      seed=args.seed,
      object_xy_jitter=args.object_xy_jitter,
      object_yaw_jitter=args.object_yaw_jitter,
      randomized_objects=("cylinder",),
    )
    configure_evaluation_guards(simulation, args.guard_mode)
    ensure_evaluation_physics(simulation)
    report["initial_cylinder_pose"] = simulation.object_pose("cylinder").tolist()
    names = list(deployment["joint_names"])
    lower, upper = joint_limits(simulation, names)
    stabilizer = AutomaticGraspStabilizer(
      simulation, enabled=True, contact_steps=3, release_closure=0.35
    )
    free_close_force = AdaptiveFreeCloseForce(
      simulation, enabled=True, force_limit=DEFAULT_FREE_CLOSE_FORCE_LIMIT,
      distance_cutoff=DEFAULT_FREE_CLOSE_CUTOFF_M,
      minimum_closure=DEFAULT_FREE_CLOSE_MIN_CLOSURE,
    )
    detector = StablePlacementDetector(timestep=simulation.timestep)
    observation = deployment["observation_contract"]
    camera_names = policy_camera_names(observation)
    image_height, image_width, _ = image_shape_hwc(observation)
    cameras = {
      name: CameraConfig(
        name, width=image_width, height=image_height, rgb=True,
        depth=False, segmentation=False,
      )
      for name in camera_names
    }
    with WorkcellRenderer(simulation.model, tuple(cameras.values())) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
      report["render_backend"] = renderer.backend_info
      if args.record:
        recorder = EvaluationVideo(
          simulation, output, fps=args.record_fps,
          width=args.review_width, height=args.review_height, second_camera="global",
          include_model_wrist="right_wrist" in camera_names,
          include_review_wrist="right_wrist" not in camera_names,
          comparison=True,
          reference_dataset=args.reference_dataset,
          reference_episode_index=args.reference_episode_index,
          heading="PICKPLACE pi0.5 MODEL EVALUATION",
          metadata={
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "model_family": "pi0.5", "prediction_horizon": horizon,
            "action_dim": deployment["action_dim"], "execute_steps": args.execute_steps,
            "control_hz": CONTROL_HZ, "replan_period_s": args.execute_steps / CONTROL_HZ,
            "seed": args.seed, "max_sim_seconds": args.max_sim_seconds,
            "observation_contract": deployment["observation_contract"],
          },
        )
        recorder.capture(0, "initial")
      while simulation.data.time < args.max_sim_seconds and not detector.succeeded:
        images = {
          name: renderer.capture(simulation.data, camera)["rgb"]
          for name, camera in cameras.items()
        }
        state = right_joint_state(simulation, names)
        request_start = time.monotonic()
        response = client.infer({
          **pi05_image_payload(images, observation),
          "state": state,
          "prompt": deployment["instruction"],
        })
        requests += 1
        try:
          actions = validated_actions(
            response, horizon=horizon, action_dim=deployment["action_dim"]
          )
        except Exception as error:
          if not recover_nonfinite_action(simulation, error):
            raise
          # A rejected chunk holds the last valid joint targets for one normal
          # control tick, then immediately replans from a fresh observation.
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while (
            simulation.data.time < target_time - 1e-12
            and simulation.data.time < args.max_sim_seconds - 1e-12
          ):
            free_close_force.before_physics_step(
              closure=stabilizer.closure(), grasp_active=stabilizer.active
            )
            before_time = float(simulation.data.time)
            simulation.step()
            ensure_evaluation_physics(simulation, before_time=before_time)
            stabilizer.after_physics_step()
          action_steps += 1
          if recorder is not None:
            recorder.capture(control_tick, "policy_guard_hold")
          continue
        log.write(json.dumps({
          "request": requests, "sim_time": float(simulation.data.time),
          "round_trip_s": time.monotonic() - request_start,
          "predicted_action_min": np.min(actions, axis=0).tolist(),
          "predicted_action_max": np.max(actions, axis=0).tolist(),
        }) + "\n")
        log.flush()
        if requests == 1:
          np.savez_compressed(
            output / "first_request.npz",
            **{f"{name}_image": image for name, image in images.items()},
            state=state,
            predicted_actions=actions,
          )
        for action in actions[:args.execute_steps]:
          if simulation.data.time >= args.max_sim_seconds or detector.succeeded:
            break
          clipped_values += apply_action(simulation, names, action, lower, upper)
          stabilizer.after_command()
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while (
            simulation.data.time < target_time - 1e-12
            and simulation.data.time < args.max_sim_seconds - 1e-12
          ):
            free_close_force.before_physics_step(closure=stabilizer.closure(), grasp_active=stabilizer.active)
            before_time = float(simulation.data.time)
            simulation.step()
            ensure_evaluation_physics(simulation, before_time=before_time)
            stabilizer.after_physics_step()
            twist = simulation.object_twist("cylinder")
            detector.update(
              linear_speed=float(np.linalg.norm(twist[:3])),
              angular_speed=float(np.linalg.norm(twist[3:])),
              placed_in_box=cylinder_is_in_box(simulation),
              grasp_active=stabilizer.active,
            )
            if detector.succeeded:
              break
          action_steps += 1
          if recorder is not None:
            recorder.capture(control_tick, "placed" if detector.succeeded else "policy")
      report["status"] = "success" if detector.succeeded else "task_not_completed"
      report["evaluation"] = {"success": bool(detector.succeeded), "placed_in_box": bool(cylinder_is_in_box(simulation))}
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
    if free_close_force is not None:
      free_close_force.close()
    if stabilizer is not None:
      stabilizer.close()
    if simulation is not None:
      guard_report = evaluation_guard_report(
        simulation, status=report.get("status"), error=run_error
      )
      report.update(guard_report)
      sim_time = float(simulation.data.time)
      report["sim_seconds"] = sim_time if np.isfinite(sim_time) else None
      if guard_report["physics_valid"]:
        report["final_cylinder_pose"] = simulation.object_pose(
          "cylinder"
        ).tolist()
      else:
        report["final_cylinder_pose"] = None
        report["evaluation"] = {
          "success": bool(detector is not None and detector.succeeded),
          "placed_in_box": None,
        }
    report["wall_seconds"] = time.monotonic() - started
    report["task_success"] = bool(
      report.get("evaluation", {}).get("success", False)
    )
    report["stats"] = {"requests": requests, "action_steps": action_steps, "clipped_action_values": clipped_values}
    if recorder is not None:
      try:
        if not getattr(simulation, "_evaluation_physics_invalid", False):
          recorder.capture(control_tick, "terminal", force=True)
        review = recorder.finish(status=report.get("status", "error"), evaluation=report.get("evaluation"), error=run_error)
        review.update(
          checkpoint_path=deployment["checkpoint_path"], checkpoint_sha256=deployment["checkpoint_sha256"],
          model_family="pi0.5", prediction_horizon=deployment["prediction_horizon"],
          action_dim=deployment["action_dim"], execute_steps=args.execute_steps,
          control_hz=CONTROL_HZ, replan_period_s=args.execute_steps / CONTROL_HZ,
          seed=args.seed, success=report.get("evaluation", {}).get("success", False),
          task_success=report["task_success"],
          inference_requests=requests, simulation_time_s=report.get("sim_seconds"), wall_time_s=report["wall_seconds"],
          guard_mode=report.get("guard_mode"),
          guard_events=report.get("guard_events", []),
          stop_reason=report.get("stop_reason"),
          physics_valid=report.get("physics_valid"),
          diagnostic_only=report.get("diagnostic_only"),
          formal_metrics_valid=report.get("formal_metrics_valid"),
        )
        (output / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        report["video"] = review
      except Exception as error:
        report["record_error"] = f"{type(error).__name__}: {error}"
        report.update(stop_reason="fatal_error", formal_metrics_valid=False)
        if run_error is None:
          report.update(status="error", error=report["record_error"])
        record_failure = error
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if record_failure is not None and run_error is None:
      raise RuntimeError(report["record_error"]) from record_failure
  return report


if __name__ == "__main__":
  run(parse_args())
