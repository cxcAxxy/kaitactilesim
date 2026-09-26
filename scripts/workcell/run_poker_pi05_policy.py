#!/usr/bin/env python3
"""Run a manifest-bound pi0.5 joint policy in the poker-draw simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig, default_model_path
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import PokerPolicyController
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_usb_pi05_policy import (
  apply_action,
  joint_limits,
  right_joint_state,
  validated_actions,
)

CONTROL_HZ = 30
CONTROLLER = "direct-30hz-pi05-absolute-joint-v1"
PENETRATION_GUARD_THRESHOLD_M = 0.0006


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:18784")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=4.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
  parser.add_argument("--execute-steps", type=int, default=8)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--success-hold-seconds", type=float, default=0.10)
  parser.add_argument(
    "--full-duration-evaluation", action="store_true",
    help="Diagnostic rollout: continue after card/table penetration or a dropped card; only nonfinite states abort",
  )
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: continue after supported card/table penetration exceeds "
      "0.6 mm; nonfinite-state and fallen-card guards remain enabled"
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--save-raw", action=argparse.BooleanOptionalAction, default=False,
    help="Save a synchronized Raw HDF5 episode and five-finger force plot",
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
    "--show-model-wrist-in-review",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.full_duration_evaluation:
    args.disable_penetration_guard = True
  if not np.isfinite(args.success_hold_seconds) or args.success_hold_seconds <= 0:
    parser.error("success-hold-seconds must be positive and finite")
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in ("xy_jitter_mm", "yaw_jitter_deg", "max_sim_seconds"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0 or (
      key == "max_sim_seconds" and value == 0
    ):
      parser.error(f"{key} must be finite and nonnegative")
  if args.xy_jitter_mm > 5.0 or args.yaw_jitter_deg > 1.0:
    parser.error("poker randomization is limited to 5 mm XY and 1 degree yaw")
  if args.show_model_wrist_in_review and args.review_second_camera == "right_wrist":
    parser.error("review second camera duplicates the model wrist view")
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
  if deployment["schema"] != "poker_pi05_deployment_v1":
    raise RuntimeError("unsupported deployment manifest schema")
  if deployment["task"] != "poker-draw" or deployment["model_family"] != "pi0.5":
    raise RuntimeError("runner requires a poker-draw pi0.5 deployment")
  return resolved, deployment


def validate_server(metadata: dict, deployment: dict, execute_steps: int) -> int:
  expected = {
    "evaluation_task": "poker-draw",
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


def outcome_stage(monitor: PokerOutcomeMonitor) -> str:
  if monitor.success:
    return "success"
  if monitor.lifted:
    return "lifted"
  if monitor.edge_reached:
    return "edge_reached"
  if monitor.contacted:
    return "contacted"
  return "awaiting_contact"


def guard_poker_state(
  observer: PokerPolicyController, *, penetration_guard_enabled: bool = True,
  fallen_card_guard_enabled: bool = True,
) -> float:
  simulation = observer.sim
  if not np.isfinite(simulation.data.qpos).all() or not np.isfinite(
    simulation.data.qvel
  ).all():
    raise RuntimeError("nonfinite simulation state")
  if fallen_card_guard_enabled and float(simulation.object_pose("card")[2]) < 0.5:
    raise RuntimeError("card fell below work surface")
  supported_penetration_m = (
    max(0.0, -float(observer._card_table_clearance()))
    if observer._supported()
    else 0.0
  )
  if (
    penetration_guard_enabled
    and supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M
  ):
    raise RuntimeError("card penetrated supported tabletop beyond 0.6 mm")
  return supported_penetration_m


def run(args) -> dict:
  from openpi_client import websocket_client_policy

  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "task": "poker-draw",
    "controller": CONTROLLER,
    "server": args.server,
    "seed": args.seed,
    "instruction": deployment["instruction"],
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "checkpoint_step": deployment["checkpoint_step"],
    "model_family": deployment["model_family"],
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "success_hold_seconds": args.success_hold_seconds,
    "full_duration_evaluation": args.full_duration_evaluation,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "expert_used": False,
    "low_level_tactile_feedback": False,
    "tactile_sent_to_model": False,
    "render_shadows": False,
  }
  started = time.monotonic()
  client = simulation = recorder = observer = monitor = review_metrics = raw_capture = None
  requests = action_steps = clipped_values = control_tick = 0
  maximum_supported_table_penetration_m = 0.0
  first_penetration_limit_exceeded_s = None
  error_text = None
  try:
    client = websocket_client_policy.WebsocketClientPolicy(args.server)
    metadata = client.get_server_metadata()
    horizon = validate_server(metadata, deployment, args.execute_steps)
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    print(
      f"[policy] connected step={deployment['checkpoint_step']} H={horizon} "
      f"execute_steps={args.execute_steps} "
      f"diagnostic_only={args.disable_penetration_guard}",
      flush=True,
    )

    with middle_force_simulation(default_model_path("poker-draw")) as (
      simulation,
      contact,
    ):
      report["contact_model"] = contact
      report["initial_card_randomization"] = reset_randomized_card(
        simulation,
        seed=args.seed,
        xy_jitter_m=args.xy_jitter_mm / 1000.0,
        yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
      )
      report["initial_card_pose"] = simulation.object_pose("card").tolist()
      monitor = PokerOutcomeMonitor(
        required_hold_seconds=args.success_hold_seconds,
        require_clearance_during_hold=args.success_hold_seconds > 0.10,
      )
      # This object is used only for read-only contact/geometry extraction. Its
      # contact-conditioned command hooks are intentionally never called.
      observer = PokerPolicyController(
        simulation,
        penetration_guard_enabled=not args.disable_penetration_guard,
      )
      review_metrics = PokerReviewMetrics(
        report["initial_card_pose"][0], outcome_monitor=monitor
      )
      review_metrics.update(simulation)
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
      if args.save_raw:
        from poker_pi05_rollout_artifacts import PokerPi05RawCapture

        raw_capture = PokerPi05RawCapture(
          simulation, output, cameras,
          metadata={
            "model_family": "pi0.5",
            "deployment_id": deployment["deployment_id"],
            "checkpoint_path": deployment["checkpoint_path"],
            "checkpoint_sha256": deployment["checkpoint_sha256"],
            "seed": args.seed,
            "execute_steps": args.execute_steps,
            "policy_control_hz": CONTROL_HZ,
            "initial_card_randomization": report["initial_card_randomization"],
          },
        )
      with WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
        report["render_backend"] = renderer.backend_info
        if args.record:
          from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo

          recorder = PokerPolicyVideo(
            simulation,
            output / "review",
            fps=args.record_fps,
            width=args.review_width,
            height=args.review_height,
            render_width=args.review_render_width,
            render_height=args.review_render_height,
            second_camera=args.review_second_camera,
            include_model_wrist=(
              args.show_model_wrist_in_review and "right_wrist" in camera_names
            ),
            include_review_wrist=(
              "right_wrist" not in camera_names
              and args.review_second_camera != "right_wrist"
            ),
            comparison=True,
            reference_dataset=args.reference_dataset,
            reference_episode_index=args.reference_episode_index,
            metrics=review_metrics,
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
              "initial_randomization": report["initial_card_randomization"],
              "contact_model": contact,
            },
          )
          recorder.capture(0, outcome_stage(monitor))

        while simulation.data.time < args.max_sim_seconds and not monitor.success:
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
            "stage": outcome_stage(monitor),
            "round_trip_s": time.monotonic() - request_started,
            "server_timing": response.get("server_timing", {}),
            "state": state.tolist(),
            "predicted_action_min": np.min(actions, axis=0).tolist(),
            "predicted_action_max": np.max(actions, axis=0).tolist(),
            "predicted_actions": actions.tolist() if args.save_raw else None,
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
            f"round_trip={row['round_trip_s']:.2f}s stage={outcome_stage(monitor)}",
            flush=True,
          )

          for index in range(args.execute_steps):
            if simulation.data.time >= args.max_sim_seconds or monitor.success:
              break
            clipped_values += apply_action(
              simulation, joint_names, actions[index], lower, upper
            )
            control_tick += 1
            target_time = control_tick / CONTROL_HZ
            while (
              simulation.data.time < target_time - 1.0e-12
              and simulation.data.time < args.max_sim_seconds - 1.0e-12
            ):
              simulation.step()
              review_metrics.update(simulation)
              observe_simulation(
                monitor, observer, report["initial_card_pose"][0]
              )
              if raw_capture is not None:
                raw_capture.observe(simulation, outcome_stage(monitor))
              supported_penetration_m = guard_poker_state(
                observer,
                penetration_guard_enabled=not args.disable_penetration_guard,
                fallen_card_guard_enabled=not args.full_duration_evaluation,
              )
              maximum_supported_table_penetration_m = max(
                maximum_supported_table_penetration_m, supported_penetration_m
              )
              if (
                supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M
                and first_penetration_limit_exceeded_s is None
              ):
                first_penetration_limit_exceeded_s = float(simulation.data.time)
              if monitor.success:
                break
            action_steps += 1
            if recorder is not None:
              recorder.capture(control_tick, outcome_stage(monitor))

      report["status"] = "success" if monitor.success else "task_not_completed"
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
      report["final_card_pose"] = simulation.object_pose("card").tolist()
      report["last_drive_state"] = simulation.drive_state()
    if observer is not None:
      report["terminal_metrics"] = observer.terminal_metrics()
    if monitor is not None:
      report["evaluation"] = monitor.report()
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_supported_table_penetration_m": (
        maximum_supported_table_penetration_m
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report["wall_seconds"] = time.monotonic() - started
    if raw_capture is not None:
      try:
        report["raw"] = raw_capture.finish(report, outcome_stage(monitor))
      except Exception as raw_error:
        report["raw_record_error"] = f"{type(raw_error).__name__}: {raw_error}"
      finally:
        raw_capture.close_incomplete()
    if recorder is not None:
      try:
        recorder.capture(control_tick, outcome_stage(monitor), force=True)
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
          maximum_supported_table_penetration_m=(
            maximum_supported_table_penetration_m
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
