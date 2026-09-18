#!/usr/bin/env python3
"""Run a manifest-bound pi0.5 joint policy in the USB-insert simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config as usb_config
from kaihand_tactile_env.tasks.usb_insert.review_metrics import UsbPolicyOutcome
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion

CONTROL_HZ = 30
CONTROLLER = "direct-30hz-pi05-absolute-joint-v1"
PENETRATION_GUARD_THRESHOLD_M = 0.0003


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:18783")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: continue after USB socket penetration exceeds 0.3 mm; "
      "socket load, drop and nonfinite-state guards remain enabled"
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--record", action=argparse.BooleanOptionalAction, default=True
  )
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
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in ("xy_jitter_mm", "yaw_jitter_deg", "max_sim_seconds"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0 or (
      key == "max_sim_seconds" and value == 0
    ):
      parser.error(f"{key} must be finite and nonnegative")
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


def load_deployment_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  deployment = json.loads(resolved.read_text(encoding="utf-8"))
  required = (
    "schema",
    "task",
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_step",
    "prediction_horizon",
    "model_action_dim",
    "action_dim",
    "control_hz",
    "joint_names",
    "observation_contract",
    "action_representation",
    "instruction",
  )
  missing = [key for key in required if key not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if deployment["schema"] != "usb_pi05_deployment_v1":
    raise RuntimeError("unsupported deployment manifest schema")
  if deployment["task"] != "usb-insert" or deployment["model_family"] != "pi0.5":
    raise RuntimeError("runner requires a USB pi0.5 deployment")
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": "usb-insert",
    "deployment_id": deployment["deployment_id"],
    "model_family": "pi0.5",
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_horizon": deployment["prediction_horizon"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "control_hz": deployment["control_hz"],
    "joint_names": deployment["joint_names"],
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
  horizon = int(metadata["prediction_horizon"])
  if not 1 <= execute_steps <= horizon:
    raise RuntimeError(
      f"execute_steps must be in [1, {horizon}], got {execute_steps}"
    )
  if int(metadata["control_hz"]) != CONTROL_HZ:
    raise RuntimeError(f"runner requires {CONTROL_HZ} Hz policy control")
  observation = metadata["observation_contract"]
  policy_camera_names(observation)
  image_shape_hwc(observation)
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("this pi0.5 checkpoint must not receive tactile input")
  return horizon


def right_joint_state(
  simulation: ArmHandSimulation, joint_names: list[str]
) -> np.ndarray:
  if len(joint_names) != 27 or len(set(joint_names)) != 27:
    raise RuntimeError("deployment must declare 27 distinct joint names")
  try:
    state = np.asarray(
      [simulation.data.qpos[simulation._qpos_address[name]] for name in joint_names],
      dtype=np.float32,
    )
  except KeyError as error:
    raise RuntimeError(f"simulator is missing model joint {error.args[0]!r}") from error
  if state.shape != (27,) or not np.isfinite(state).all():
    raise RuntimeError("right-side policy state is nonfinite or malformed")
  return state


def validated_actions(
  response: dict,
  *,
  horizon: int,
  action_dim: int,
) -> np.ndarray:
  if not isinstance(response, dict) or "actions" not in response:
    raise RuntimeError("pi0.5 server response has no actions")
  actions = np.asarray(response["actions"], dtype=np.float64)
  if actions.shape != (horizon, action_dim):
    raise RuntimeError(
      f"expected policy actions shape {(horizon, action_dim)}, got {actions.shape}"
    )
  if not np.isfinite(actions).all():
    raise RuntimeError("policy actions contain nonfinite values")
  return actions


def joint_limits(
  simulation: ArmHandSimulation, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
  ids = np.asarray([simulation._joint_id[name] for name in joint_names], dtype=int)
  ranges = np.asarray(simulation.model.jnt_range[ids], dtype=np.float64)
  if ranges.shape != (27, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
    raise RuntimeError("simulator has invalid right-side joint limits")
  return ranges[:, 0], ranges[:, 1]


def guard_usb_state(
  outcome: UsbPolicyOutcome,
  simulation: ArmHandSimulation,
  *,
  penetration_guard_enabled: bool = True,
) -> None:
  state = outcome.state
  if state.socket_normal_load_n > usb_config.MAX_SOCKET_NORMAL_LOAD_N:
    raise RuntimeError("USB socket total normal load exceeded task limit")
  if state.axial_resistance_n > usb_config.MAX_SOCKET_AXIAL_FORCE_N:
    raise RuntimeError("USB socket axial resistance exceeded task limit")
  if state.wall_normal_load_n > usb_config.MAX_SOCKET_WALL_LOAD_N:
    raise RuntimeError("USB socket side-wall load exceeded task limit")
  if (
    penetration_guard_enabled
    and state.maximum_socket_penetration_m > PENETRATION_GUARD_THRESHOLD_M
  ):
    raise RuntimeError("USB socket penetration exceeded 0.3 mm")
  if float(simulation.object_pose("usb_plug")[2]) < 0.5:
    raise RuntimeError("USB plug fell below work surface")
  if not np.isfinite(simulation.data.qpos).all() or not np.isfinite(
    simulation.data.qvel
  ).all():
    raise RuntimeError("nonfinite simulation state")


def apply_action(
  simulation: ArmHandSimulation,
  joint_names: list[str],
  action: np.ndarray,
  lower: np.ndarray,
  upper: np.ndarray,
) -> int:
  # The simulator's position servos provide their normal velocity/force safety.
  # We additionally bound extrapolated policy values to physical joint limits.
  clipped = np.clip(action, lower, upper)
  clipped_count = int(np.count_nonzero(np.abs(clipped - action) > 1.0e-9))
  simulation.set_arm_joint_goal("right", clipped[:7])
  accepted = simulation.set_hand_joint_targets(joint_names[7:], clipped[7:])
  if accepted != 20:
    raise RuntimeError(f"right hand accepted only {accepted}/20 joint targets")
  return clipped_count


def run(args) -> dict:
  from openpi_client import websocket_client_policy

  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "task": "usb-insert",
    "controller": CONTROLLER,
    "server": args.server,
    "seed": args.seed,
    "instruction": deployment["instruction"],
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "model_family": deployment["model_family"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "expert_used": False,
    "tactile_sent_to_model": False,
    "render_shadows": False,
  }
  started = time.monotonic()
  client = simulation = recorder = outcome = None
  requests = action_steps = clipped_values = control_tick = 0
  error_text = None
  first_penetration_limit_exceeded_s = None
  try:
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    print(
      f"[policy] connected step={deployment['checkpoint_step']} H={horizon} "
      f"execute_steps={args.execute_steps}",
      flush=True,
    )

    simulation = ArmHandSimulation(scene="usb-insert")
    simulation.reset(seed=args.seed)
    report["initial_randomization"] = initialize_for_insertion(
      simulation,
      seed=args.seed,
      xy_jitter_m=args.xy_jitter_mm / 1000.0,
      yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
    )
    report["initial_plug_pose"] = simulation.object_pose("usb_plug").tolist()
    outcome = UsbPolicyOutcome(simulation)
    if outcome.state.maximum_socket_penetration_m > PENETRATION_GUARD_THRESHOLD_M:
      first_penetration_limit_exceeded_s = float(simulation.data.time)
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
        from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

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
          metrics=outcome,
          heading="USB INSERT pi0.5 MODEL EVALUATION",
          metadata={
            "server": args.server,
            "seed": args.seed,
            "model_family": deployment["model_family"],
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "prediction_horizon": horizon,
            "model_action_dim": deployment["model_action_dim"],
            "action_dim": deployment["action_dim"],
            "execute_steps": args.execute_steps,
            "control_hz": CONTROL_HZ,
            "replan_period_s": args.execute_steps / CONTROL_HZ,
            "max_sim_seconds": args.max_sim_seconds,
            "diagnostic_only": bool(args.disable_penetration_guard),
            "formal_metrics_valid": not args.disable_penetration_guard,
            "penetration_guard_enabled": not args.disable_penetration_guard,
            "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
            "observation_contract": deployment["observation_contract"],
            "action_representation": deployment["action_representation"],
            "initial_randomization": report["initial_randomization"],
          },
        )
        recorder.capture(0, outcome.stage())

      while simulation.data.time < args.max_sim_seconds and not outcome.success:
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
            "prompt": deployment["instruction"],
          }
        )
        requests += 1
        actions = validated_actions(
          response, horizon=horizon, action_dim=deployment["action_dim"]
        )
        row = {
          "request": requests,
          "sim_time": float(simulation.data.time),
          "stage": outcome.stage(),
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
        print(
          f"[policy] request={requests} sim_t={simulation.data.time:.3f}s "
          f"round_trip={row['round_trip_s']:.2f}s stage={outcome.stage()}",
          flush=True,
        )

        for index in range(args.execute_steps):
          if simulation.data.time >= args.max_sim_seconds or outcome.success:
            break
          clipped_values += apply_action(
            simulation, joint_names, actions[index], lower, upper
          )
          control_tick += 1
          target_time = control_tick / CONTROL_HZ
          while simulation.data.time < target_time - 1.0e-12:
            simulation.step()
            outcome.update(simulation)
            if (
              first_penetration_limit_exceeded_s is None
              and outcome.state.maximum_socket_penetration_m
              > PENETRATION_GUARD_THRESHOLD_M
            ):
              first_penetration_limit_exceeded_s = float(simulation.data.time)
            guard_usb_state(
              outcome,
              simulation,
              penetration_guard_enabled=not args.disable_penetration_guard,
            )
            if outcome.success:
              break
          action_steps += 1
          if recorder is not None:
            recorder.capture(control_tick, outcome.stage())

      report["status"] = "success" if outcome.success else "task_not_completed"
  except Exception as error:
    error_text = f"{type(error).__name__}: {error}"
    report.update(status="error", error=error_text)
    raise
  finally:
    if client is not None:
      try:
        client._ws.close()
      except Exception:
        pass
    if simulation is not None:
      report["sim_seconds"] = float(simulation.data.time)
      report["final_plug_pose"] = simulation.object_pose("usb_plug").tolist()
    if outcome is not None:
      report["evaluation"] = outcome.report()
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_socket_penetration_m": (
        float(outcome.maximum_socket_penetration_m)
        if outcome is not None else 0.0
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    report["wall_seconds"] = time.monotonic() - started
    if recorder is not None:
      try:
        recorder.capture(control_tick, outcome.stage(), force=True)
        video = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=error_text,
        )
        video.update(
          seed=args.seed,
          success=bool(report.get("evaluation", {}).get("success", False)),
          inference_requests=requests,
          simulation_time_s=report.get("sim_seconds"),
          wall_time_s=report["wall_seconds"],
          model_family=deployment["model_family"],
          checkpoint_path=deployment["checkpoint_path"],
          checkpoint_sha256=deployment["checkpoint_sha256"],
          prediction_horizon=report.get("prediction_horizon"),
          model_action_dim=deployment["model_action_dim"],
          action_dim=deployment["action_dim"],
          execute_steps=args.execute_steps,
          control_hz=CONTROL_HZ,
          replan_period_s=args.execute_steps / CONTROL_HZ,
          diagnostic_only=bool(args.disable_penetration_guard),
          formal_metrics_valid=not args.disable_penetration_guard,
          penetration_guard_enabled=not args.disable_penetration_guard,
          penetration_guard_threshold_m=PENETRATION_GUARD_THRESHOLD_M,
          maximum_socket_penetration_m=(
            report["penetration_diagnostics"]["maximum_socket_penetration_m"]
          ),
          first_penetration_limit_exceeded_sim_time_s=(
            first_penetration_limit_exceeded_s
          ),
        )
        (output / "review" / "review.json").write_text(
          json.dumps(video, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["video"] = video
      except Exception as record_error:
        report["record_error"] = f"{type(record_error).__name__}: {record_error}"
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"[report] {output / 'summary.json'} status={report.get('status')} "
      f"requests={requests} action_steps={action_steps}",
      flush=True,
    )
  return report


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
