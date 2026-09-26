#!/usr/bin/env python3
"""Run one PickPlace LingBot-VLA-2.0 checkpoint trial in the KaiHand scene.

The LingBot server reverses its training normalization and arm-delta transform.
Its two returned action arrays are therefore *absolute* arm and hand targets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.pick_place.task import cylinder_is_in_box
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
from run_egosteer_policy import (
  DEFAULT_FREE_CLOSE_CUTOFF_M,
  DEFAULT_FREE_CLOSE_FORCE_LIMIT,
  DEFAULT_FREE_CLOSE_MIN_CLOSURE,
  AdaptiveFreeCloseForce,
  AutomaticGraspStabilizer,
  StablePlacementDetector,
)
from run_usb_pi05_policy import apply_action, joint_limits, right_joint_state

SCHEMA = DEPLOYMENT_SCHEMA
PREDICTION_HORIZON = HORIZON
CAMERAS = tuple(OBSERVATION_CONTRACT["cameras"])
DEFAULT_REFERENCE_DATASET = Path("/nas/chenxianchi/datasets/sim/pick-place/lerobot_v3/0920_200")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", required=True, help="LingBot websocket host:port (without ws://)")
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--execute-steps", required=True, type=int)
  parser.add_argument("--max-sim-seconds", type=float, default=90.0)
  parser.add_argument("--object-xy-jitter", type=float, default=0.01)
  parser.add_argument("--object-yaw-jitter", type=float, default=0.05)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  args = parser.parse_args(argv)
  if args.seed < 0 or not 1 <= args.execute_steps <= PREDICTION_HORIZON:
    parser.error("seed must be nonnegative and execute-steps must be in [1, 50]")
  if not np.isfinite(args.max_sim_seconds) or args.max_sim_seconds <= 0:
    parser.error("max-sim-seconds must be positive and finite")
  if any(not np.isfinite(value) or value < 0 for value in (args.object_xy_jitter, args.object_yaw_jitter)):
    parser.error("object jitter limits must be finite and nonnegative")
  if args.reference_episode_index < 0:
    parser.error("reference-episode-index must be nonnegative")
  if min(args.review_width, args.review_height) < 2 or args.review_width % 2 or args.review_height % 2:
    parser.error("review size must be positive and even")
  return args


def validate_deployment(deployment: dict) -> None:
  expected = {
    "schema": SCHEMA,
    "task": TASK,
    "instruction": INSTRUCTION,
    "model_family": MODEL_FAMILY,
    "prediction_horizon": PREDICTION_HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": list(RIGHT_JOINT_NAMES),
    "action_representation": ACTION_REPRESENTATION,
  }
  mismatches = {key: (value, deployment.get(key)) for key, value in expected.items() if deployment.get(key) != value}
  if mismatches:
    raise RuntimeError(f"LingBot PickPlace deployment contract mismatch: {mismatches}")
  observation = deployment.get("observation_contract")
  if not isinstance(observation, dict):
    raise RuntimeError("deployment observation_contract must be an object")
  if policy_camera_names(observation) != CAMERAS:
    raise RuntimeError("LingBot PickPlace requires head and right_wrist policy cameras")
  if image_shape_hwc(observation) != (240, 320, 3):
    raise RuntimeError("LingBot PickPlace requires 240x320 RGB policy images")
  if observation.get("state_dim") != ACTION_DIM or observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("LingBot PickPlace requires 27-D joint state and no tactile model input")
  if observation.get("image_dtype") != "uint8":
    raise RuntimeError("LingBot PickPlace policy images must be uint8")
  if observation != OBSERVATION_CONTRACT:
    raise RuntimeError("LingBot PickPlace observation contract differs from frozen training contract")
  for key in ("deployment_id", "checkpoint_path", "checkpoint_sha256"):
    if not isinstance(deployment.get(key), str) or not deployment[key]:
      raise RuntimeError(f"deployment {key} must be a nonempty string")
  declared_reference = deployment.get("reference_dataset")
  if not isinstance(declared_reference, str) or not Path(declared_reference).is_absolute():
    raise RuntimeError("deployment reference_dataset must be an absolute training dataset path")
  if deployment.get("reference_episode_index") != 0:
    raise RuntimeError("LingBot PickPlace reference_episode_index must be 0")


def bound_reference(deployment: dict, requested_dataset: Path | None, requested_episode: int) -> Path:
  """A CLI reference option may confirm the manifest, never replace it."""
  dataset = Path(deployment["reference_dataset"]).expanduser().resolve(strict=True)
  if not dataset.is_dir():
    raise RuntimeError(f"manifest reference_dataset is not a directory: {dataset}")
  if requested_dataset is not None and requested_dataset.expanduser().resolve() != dataset:
    raise RuntimeError("--reference-dataset differs from the frozen training reference_dataset")
  if requested_episode != deployment["reference_episode_index"]:
    raise RuntimeError("--reference-episode-index differs from the frozen episode 00")
  return dataset


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  if not isinstance(metadata, dict):
    raise RuntimeError("LingBot server metadata must be an object")
  expected = {
    "evaluation_task": TASK,
    "deployment_id": deployment["deployment_id"],
    "model_family": MODEL_FAMILY,
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_horizon": deployment["prediction_horizon"],
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": deployment["joint_names"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "instruction": deployment["instruction"],
    "reference_dataset": deployment["reference_dataset"],
  }
  mismatches = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
  if mismatches:
    raise RuntimeError(f"LingBot server contract mismatch: {mismatches}")
  if not 1 <= execute_steps <= PREDICTION_HORIZON:
    raise RuntimeError(f"execute_steps={execute_steps} must be in [1, {PREDICTION_HORIZON}]")
  return PREDICTION_HORIZON


def make_observation(images: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> dict:
  """Construct the flat LeRobot keys expected by LingBot FeatureTransform."""
  if tuple(images) != CAMERAS:
    raise RuntimeError(f"captured cameras {list(images)} differ from {list(CAMERAS)}")
  prepared = {}
  for camera in CAMERAS:
    image = np.asarray(images[camera])
    if image.shape != (240, 320, 3) or image.dtype != np.uint8:
      raise RuntimeError(f"{camera} image must be HWC uint8 with shape (240, 320, 3)")
    prepared[f"observation.images.{camera}"] = np.ascontiguousarray(image)
  state = np.asarray(state, dtype=np.float32)
  if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
    raise RuntimeError("right joint state must be finite and 27-D")
  if not isinstance(instruction, str) or not instruction:
    raise RuntimeError("LingBot task instruction must be a nonempty string")
  prepared["observation.state.right_arm_joint_position"] = state[:7].copy()
  prepared["observation.state.right_hand_joint_position"] = state[7:].copy()
  prepared["task"] = instruction
  return prepared


def validated_actions(response: dict, *, horizon: int) -> np.ndarray:
  """Read already-denormalized absolute joint targets, without adding state."""
  if not isinstance(response, dict):
    raise RuntimeError("LingBot server response must be an object")
  fields = (
    ("auxiliary.action.right_arm_joint_target", 7),
    ("action.right_hand_joint_position", 20),
  )
  values = []
  for key, dimension in fields:
    if key not in response:
      raise RuntimeError(f"LingBot server response is missing {key}")
    value = np.asarray(response[key], dtype=np.float64)
    if value.shape != (horizon, dimension) or not np.isfinite(value).all():
      raise RuntimeError(f"{key} must have finite shape {(horizon, dimension)}, got {value.shape}")
    values.append(value)
  return np.concatenate(values, axis=1)


def finish_comparison(comparison_trace, reference, review_dir: Path) -> dict:
  """Write a full-rate physical trace and episode-00 comparison charts."""
  from kaihand_tactile_env.shared.openwam_evaluation_plots import OpenWAMEvaluationPlots
  from lingbot_pickplace_reference import REFERENCE_TACTILE_SOURCE

  trace_summary = comparison_trace.finish(review_dir)
  if not trace_summary["rollout"]["samples"]:
    raise RuntimeError("evaluation has no control-step physical samples")
  plots = OpenWAMEvaluationPlots(reference, artifact_prefix="evaluation")
  with np.load(review_dir / "evaluation_rollout_trace.npz") as trace_data:
    for index, timestamp in enumerate(trace_data["simulation_time_s"]):
      plots.capture(
        float(timestamp), trace_data["state_29"][index],
        normal_taxel_force_n=trace_data["normal_taxel_force_n"][index],
        tangent_taxel_force_n=trace_data["tangent_taxel_force_n"][index],
      )
  plot_summary = plots.finish(review_dir)
  plot_summary["reference"]["alignment"] = (
    "head-camera state_index selects the source state; wrist FK uses saved qpos; "
    "fingertip force uses saved MuJoCo pad-contact solver wrenches"
  )
  plot_summary["reference"]["tactile_source"] = REFERENCE_TACTILE_SOURCE
  Path(plot_summary["artifacts"]["metadata"]).write_text(
    json.dumps(plot_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  trace_summary["status"] = "ok"
  trace_summary.pop("reason", None)
  trace_summary["reference"].update(plot_summary["reference"])
  trace_summary["tactile"]["reference_source"] = REFERENCE_TACTILE_SOURCE
  trace_summary["tactile"]["comparison_note"] = (
    "Episode-00 reference is reconstructed from recorded MuJoCo pad-contact "
    "solver wrenches; both reference and rollout use the same conservative "
    "7x5 force-sum and signed tangent-vector semantics."
  )
  trace_summary["artifacts"].update(plot_summary["artifacts"])
  (review_dir / "evaluation_comparison.json").write_text(
    json.dumps(trace_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  return trace_summary


def run(args: argparse.Namespace) -> dict:
  from kaihand_tactile_env.shared.contact_tactile import (
    SolverDistributedTactileProvider,
  )
  from kaihand_tactile_env.shared.evaluation_comparison import EvaluationComparisonTrace
  from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
  from lingbot_pickplace_reference import load_pickplace_reference
  from openpi_client import websocket_client_policy

  deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
  deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
  validate_deployment(deployment)
  reference_dataset = bound_reference(deployment, args.reference_dataset, args.reference_episode_index)
  reference_episode_index = deployment["reference_episode_index"]
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  started = time.monotonic()
  report = {
    "task": "pick-place",
    "controller": "direct-30hz-lingbot-vla2-absolute-joint-with-demonstration-grasp-aids-v1",
    "seed": args.seed,
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "model_family": MODEL_FAMILY,
    "prediction_horizon": PREDICTION_HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "observation_contract": deployment["observation_contract"],
    "action_representation": ACTION_REPRESENTATION,
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "max_sim_seconds": args.max_sim_seconds,
    "reference_dataset": str(reference_dataset),
    "reference_episode_index": reference_episode_index,
    "initial_randomization": {"object_xy_jitter_m": args.object_xy_jitter, "object_yaw_jitter_rad": args.object_yaw_jitter},
    "grasp_aids": {"auto_grasp_stabilizer": True, "adaptive_free_close_force": True},
    "diagnostic_only": False,
    "formal_metrics_valid": True,
  }
  client = simulation = recorder = stabilizer = free_close_force = comparison_trace = reference = None
  requests = action_steps = clipped_values = control_tick = 0
  run_error = None
  artifact_error = None
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
    report["initial_cylinder_pose"] = simulation.object_pose("cylinder").tolist()
    names = list(RIGHT_JOINT_NAMES)
    lower, upper = joint_limits(simulation, names)
    stabilizer = AutomaticGraspStabilizer(
      simulation, enabled=True, contact_steps=3, release_closure=0.35,
    )
    free_close_force = AdaptiveFreeCloseForce(
      simulation, enabled=True, force_limit=DEFAULT_FREE_CLOSE_FORCE_LIMIT,
      distance_cutoff=DEFAULT_FREE_CLOSE_CUTOFF_M,
      minimum_closure=DEFAULT_FREE_CLOSE_MIN_CLOSURE,
    )
    detector = StablePlacementDetector(timestep=simulation.timestep)
    image_height, image_width, _ = image_shape_hwc(deployment["observation_contract"])
    cameras = {
      name: CameraConfig(name, width=image_width, height=image_height, rgb=True, depth=False, segmentation=False)
      for name in CAMERAS
    }
    with WorkcellRenderer(simulation.model, tuple(cameras.values())) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
      report["render_backend"] = renderer.backend_info
      reference = load_pickplace_reference(reference_dataset, output_episode_index=reference_episode_index)
      if args.record:
        recorder = EvaluationVideo(
          simulation, output / "review", fps=args.record_fps,
          width=args.review_width, height=args.review_height, second_camera="global",
          include_model_wrist=True, comparison=False,
          reference_dataset=reference_dataset,
          reference_episode_index=reference_episode_index,
          heading="PICKPLACE LingBot-VLA-2.0 MODEL EVALUATION",
          metadata={
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "model_family": MODEL_FAMILY,
            "prediction_horizon": horizon,
            "model_action_dim": MODEL_ACTION_DIM,
            "action_dim": ACTION_DIM,
            "execute_steps": args.execute_steps,
            "control_hz": CONTROL_HZ,
            "replan_period_s": args.execute_steps / CONTROL_HZ,
            "seed": args.seed,
            "max_sim_seconds": args.max_sim_seconds,
            "observation_contract": deployment["observation_contract"],
          },
        )
        recorder.capture(0, "initial")
      tactile_provider = recorder.provider if recorder is not None else SolverDistributedTactileProvider(simulation.model)
      comparison_trace = EvaluationComparisonTrace(tactile_provider=tactile_provider)
      comparison_trace.capture(simulation)
      while simulation.data.time < args.max_sim_seconds and not detector.succeeded:
        images = {name: renderer.capture(simulation.data, camera)["rgb"] for name, camera in cameras.items()}
        state = right_joint_state(simulation, names)
        request_start = time.monotonic()
        response = client.infer(make_observation(images, state, deployment["instruction"]))
        actions = validated_actions(response, horizon=horizon)
        requests += 1
        log.write(json.dumps({
          "request": requests,
          "sim_time": float(simulation.data.time),
          "round_trip_s": time.monotonic() - request_start,
          "state": state.tolist(),
          "predicted_actions": actions.tolist(),
          "predicted_action_min": np.min(actions, axis=0).tolist(),
          "predicted_action_max": np.max(actions, axis=0).tolist(),
          "server_timing": response.get("server_timing"),
        }) + "\n")
        log.flush()
        if requests == 1:
          np.savez_compressed(
            output / "first_request.npz",
            **{f"{name}_image": image for name, image in images.items()},
            state=state, predicted_actions=actions,
          )
        for action in actions[:args.execute_steps]:
          if simulation.data.time >= args.max_sim_seconds or detector.succeeded:
            break
          clipped_values += apply_action(simulation, names, action, lower, upper)
          stabilizer.after_command()
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while simulation.data.time < target_time - 1.0e-12:
            free_close_force.before_physics_step(closure=stabilizer.closure(), grasp_active=stabilizer.active)
            simulation.step()
            stabilizer.after_physics_step()
            twist = simulation.object_twist("cylinder")
            detector.update(
              linear_speed=float(np.linalg.norm(twist[:3])),
              angular_speed=float(np.linalg.norm(twist[3:])),
              placed_in_box=cylinder_is_in_box(simulation),
              grasp_active=stabilizer.active,
            )
            if not np.isfinite(simulation.data.qpos).all() or not np.isfinite(simulation.data.qvel).all():
              raise RuntimeError("nonfinite simulation state")
            if detector.succeeded:
              break
          action_steps += 1
          if recorder is not None:
            recorder.capture(control_tick, "placed" if detector.succeeded else "policy")
          comparison_trace.capture(simulation)
      report["status"] = "success" if detector.succeeded else "task_not_completed"
      report["evaluation"] = {
        "success": bool(detector.succeeded),
        "placed_in_box": bool(cylinder_is_in_box(simulation)),
      }
  except Exception as error:
    run_error = f"{type(error).__name__}: {error}"
    report.update(status="error", error=run_error, formal_metrics_valid=False)
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
      report["sim_seconds"] = float(simulation.data.time)
      report["final_cylinder_pose"] = simulation.object_pose("cylinder").tolist()
    report["wall_seconds"] = time.monotonic() - started
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    review = None
    if recorder is not None:
      try:
        recorder.capture(control_tick, "terminal", force=True)
        review = recorder.finish(status=report.get("status", "error"), evaluation=report.get("evaluation"), error=run_error)
      except Exception as error:
        artifact_error = error
        report["record_error"] = f"{type(error).__name__}: {error}"
    if comparison_trace is not None:
      try:
        comparison_trace.capture(simulation)
        assert reference is not None
        trace_summary = finish_comparison(comparison_trace, reference, output / "review")
        report["comparison_plots"] = trace_summary
        if review is not None:
          review["comparison_requested"] = True
          review["comparison_plots"] = trace_summary
      except Exception as error:
        artifact_error = artifact_error or error
        report["comparison_error"] = f"{type(error).__name__}: {error}"
    if review is not None:
      review.update(
        checkpoint_path=deployment["checkpoint_path"], checkpoint_sha256=deployment["checkpoint_sha256"],
        model_family=MODEL_FAMILY, prediction_horizon=PREDICTION_HORIZON,
        model_action_dim=MODEL_ACTION_DIM, action_dim=ACTION_DIM,
        execute_steps=args.execute_steps, control_hz=CONTROL_HZ,
        replan_period_s=args.execute_steps / CONTROL_HZ,
        seed=args.seed, success=report.get("evaluation", {}).get("success", False),
        inference_requests=requests, simulation_time_s=report.get("sim_seconds"),
        wall_time_s=report["wall_seconds"], diagnostic_only=False,
        formal_metrics_valid=artifact_error is None and run_error is None,
      )
      (output / "review/review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
      report["video"] = review
    if artifact_error is not None:
      report["formal_metrics_valid"] = False
      if run_error is None:
        report.update(status="error", error=f"evaluation artifact failed: {type(artifact_error).__name__}: {artifact_error}")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
  if artifact_error is not None and run_error is None:
    raise RuntimeError(report["error"]) from artifact_error
  return report


if __name__ == "__main__":
  run(parse_args())
