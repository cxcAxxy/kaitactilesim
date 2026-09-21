"""Model-only poker rollout with contact-conditioned low-level feedback.

Free-space joint control, supported Cartesian 4 N control and 0.50 N/finger
feedback are selected from live sensing, not expert timestamps or trajectories.
This is a model adapter, not a claim of full expert-controller equivalence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig, default_model_path
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
from kaihand_tactile_env.shared.policy_cameras import (
  include_model_right_wrist_panel,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import (
  CONTROLLER_VERSION,
  PENETRATION_GUARD_THRESHOLD_M,
  PokerPolicyController,
)
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)

# Reuse only generic IK/transport/view helpers, never the pick-place runner.
from run_egosteer_policy import (
  RolloutStats,
  ViewerClosed,
  _append_model_observations,
  _command_action_step,
  _infer_while_servicing_viewer,
  _model_action_horizon,
  _model_camera_names,
  _select_model_observation,
  _validated_prediction,
  _viewer_context,
)

INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the fingers "
  "and thumb, lift it, and turn its face toward the robot to look at it."
)
CONTROLLER = CONTROLLER_VERSION


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:8766")
  parser.add_argument(
    "--deployment-manifest",
    type=Path,
    help="JSON identity contract for the exact checkpoint served at --server",
  )
  parser.add_argument("--viewer", choices=("native", "none"), default="native")
  parser.add_argument("--seed", type=int, default=90903)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: continue after supported card/table penetration exceeds "
      "0.6 mm; all other termination guards remain enabled"
    ),
  )
  parser.add_argument("--response-timeout", type=float, default=180.0)
  parser.add_argument("--viewer-hz", type=float, default=30.0)
  parser.add_argument(
    "--real-time", action=argparse.BooleanOptionalAction, default=None
  )
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--record",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save the 1080p bilateral tactile evaluation video",
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
    "--audit-actions",
    action="store_true",
    help="Log read-only wrist/fingertip tracking and contact distances",
  )
  parser.add_argument("--max-wrist-jump", type=float, default=0.15)
  parser.add_argument("--max-arm-position-error", type=float, default=0.03)
  parser.add_argument("--max-arm-orientation-error", type=float, default=0.30)
  parser.add_argument("--hand-ik-tolerance", type=float, default=0.0015)
  parser.add_argument("--max-hand-error", type=float, default=0.02)
  parser.add_argument("--strict-ik", action="store_true")
  parser.add_argument(
    "--evaluate",
    action="store_true",
    help="Read-only staged task outcome evaluation, stop on stable success",
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in (
    "max_sim_seconds",
    "response_timeout",
    "viewer_hz",
    "max_wrist_jump",
    "max_arm_position_error",
    "max_arm_orientation_error",
    "hand_ik_tolerance",
    "max_hand_error",
  ):
    value = getattr(args, key)
    if not np.isfinite(value) or value <= 0:
      parser.error(f"{key} must be finite and positive")
  if args.hand_ik_tolerance > args.max_hand_error:
    parser.error("hand IK tolerance cannot exceed maximum error")
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
  if args.real_time is None:
    args.real_time = args.viewer == "native"
  return args


def _load_deployment_manifest(path):
  if path is None:
    return None
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise RuntimeError("deployment manifest must contain a JSON object")
  required = (
    "deployment_id",
    "model_family",
    "checkpoint_path",
    "checkpoint_sha256",
    "model_config_path",
    "model_config_sha256",
    "normalizer_path",
    "normalizer_sha256",
    "prediction_horizon",
    "action_dim",
    "observation_contract",
    "action_representation",
  )
  missing = [key for key in required if key not in payload]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  return payload


def validate_server(metadata, *, execute_steps=None, deployment=None):
  if metadata.get("action_dim") != 48:
    raise RuntimeError(f"Expected poker action_dim=48: {metadata}")
  horizon = _model_action_horizon(metadata)
  if execute_steps is not None and execute_steps > horizon:
    raise RuntimeError(
      f"--execute-steps={execute_steps} exceeds server horizon {horizon}"
    )
  if deployment is not None:
    expected = {
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
  return horizon


async def run(args):
  output = args.output_dir or Path("artifacts") / time.strftime(
    "poker_policy_%Y%m%d_%H%M%S"
  )
  # Never overwrite another rollout's evidence.
  output.mkdir(parents=True, exist_ok=False)
  stats = RolloutStats()
  deployment = _load_deployment_manifest(args.deployment_manifest)
  report = {
    "controller": CONTROLLER,
    "server": args.server,
    "seed": args.seed,
    "training_controller_equivalent": False,
    "expert_used": False,
    "success_evaluated": args.evaluate,
    "tactile_sent_to_model": False,
    "control_hz": 30,
    "history_horizon": 6,
    "history_stride": 30,
    "force_limit_scope": "supported slide/hold only; contact-conditioned Cartesian servo",
    "low_level_tactile_feedback": True,
    "precontact_noise": False,
    "instruction": INSTRUCTION,
    "execute_steps": args.execute_steps,
    "max_sim_seconds": args.max_sim_seconds,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "render_shadows": False,
  }
  if deployment is not None:
    report.update(
      deployment_id=deployment["deployment_id"],
      checkpoint_path=deployment["checkpoint_path"],
      checkpoint_sha256=deployment["checkpoint_sha256"],
      model_family=deployment["model_family"],
      action_dim=deployment["action_dim"],
      observation_contract=deployment["observation_contract"],
      action_representation=deployment["action_representation"],
      deployment_manifest=str(args.deployment_manifest.expanduser().resolve()),
    )
  started = time.monotonic()
  sim = None
  controller = None
  recorder = None
  review_metrics = None
  audit_log = None
  control_tick = 0
  monitor = PokerOutcomeMonitor() if args.evaluate else None
  try:
    # Handshake before allocating a robot or renderer.
    async with EgoSteerPolicyClient(
      args.server, response_timeout=args.response_timeout
    ) as client:
      prediction_horizon = validate_server(
        client.metadata or {},
        execute_steps=args.execute_steps,
        deployment=deployment,
      )
      report["server_metadata"] = client.metadata
      report["prediction_horizon"] = prediction_horizon
      report["replan_period_s"] = args.execute_steps / 30
      camera_names = _model_camera_names(client.metadata or {})
      observation = (client.metadata or {}).get("observation_contract", {})
      print(
        f"[policy] connected: {client.metadata}; controller={CONTROLLER}", flush=True
      )
      with middle_force_simulation(default_model_path("poker-draw")) as (sim, contact):
        report["contact_model"] = contact
        report["initial_card_randomization"] = reset_randomized_card(
          sim, seed=args.seed, xy_jitter_m=0.004, yaw_jitter_rad=float(np.deg2rad(0.5))
        )
        report["initial_card_pose"] = sim.object_pose("card").tolist()
        review_metrics = PokerReviewMetrics(
          report["initial_card_pose"][0], outcome_monitor=monitor
        )
        review_metrics.update(sim)
        controller = PokerPolicyController(
          sim, penetration_guard_enabled=not args.disable_penetration_guard
        )
        head = CameraConfig(
          "head", width=320, height=240, rgb=True, depth=False, segmentation=False
        )
        left_wrist = CameraConfig(
          "left_wrist", width=320, height=240, rgb=True, depth=False,
          segmentation=False,
        )
        wrist = CameraConfig(
          "right_wrist", width=320, height=240, rgb=True, depth=False,
          segmentation=False,
        )
        available_cameras = {
          "head": head,
          "left_wrist": left_wrist,
          "right_wrist": wrist,
        }
        model_cameras = {name: available_cameras[name] for name in camera_names}
        histories = {
          name: ObservationHistory(
            horizon=int(observation.get("image_history", 6)),
            stride=int(observation.get("image_stride", 30)),
          )
          for name in camera_names
        }
        with (
          WorkcellRenderer(sim.model, tuple(model_cameras.values()), shadows=False) as renderer,
          _viewer_context(sim, args.viewer) as viewer,
        ):
          report["render_backend"] = renderer.backend_info
          calibrations = _append_model_observations(
            sim, renderer, model_cameras, histories
          )
          report["camera_intrinsics"] = {
            name: calibrations[name].intrinsic.tolist() for name in camera_names
          }
          report["head_intrinsic"] = calibrations["head"].intrinsic.tolist()
          report["head_world_from_camera"] = (
            calibrations["head"].world_from_camera.tolist()
          )
          if args.audit_actions:
            audit_log = (output / "action_audit.jsonl").open("x")
          if args.record:
            from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo

            recorder = PokerPolicyVideo(
              sim,
              output / "review",
              fps=args.record_fps,
              width=args.review_width,
              height=args.review_height,
              render_width=args.review_render_width,
              render_height=args.review_render_height,
              second_camera=args.review_second_camera,
              include_model_wrist=include_model_right_wrist_panel(
                camera_names, args.review_second_camera
              ),
              metrics=review_metrics,
              metadata={
                "server": args.server,
                "server_metadata": client.metadata,
                "seed": args.seed,
                "model_family": report.get("model_family"),
                "checkpoint_path": report.get("checkpoint_path"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "prediction_horizon": prediction_horizon,
                "action_dim": 48,
                "execute_steps": args.execute_steps,
                "control_hz": 30,
                "replan_period_s": args.execute_steps / 30,
                "max_sim_seconds": args.max_sim_seconds,
                "diagnostic_only": bool(args.disable_penetration_guard),
                "formal_metrics_valid": not args.disable_penetration_guard,
                "penetration_guard_enabled": not args.disable_penetration_guard,
                "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
                "observation_contract": report.get("observation_contract"),
                "action_representation": report.get("action_representation"),
              },
            )
            recorder.capture(0, controller.mode)
          if viewer is not None:
            viewer.sync()
          control_tick = 0
          last_sync = 0.0
          with (output / "requests.jsonl").open("x") as log:
            while sim.data.time < args.max_sim_seconds and not (
              monitor and monitor.success
            ):
              if args.max_requests and stats.requests >= args.max_requests:
                break
              selected = _select_model_observation(
                camera_names, histories, calibrations
              )
              camera_from_world = camera_from_world_opencv(
                calibrations["head"].world_from_camera
              )
              states = model_state_history(selected.raw_states, camera_from_world)
              request_start = time.monotonic()
              response = await _infer_while_servicing_viewer(
                client,
                viewer,
                None,
                images=selected.images,
                states=states,
                camera_intrinsics=selected.camera_intrinsics,
                instruction=INSTRUCTION,
                image_format="jpeg",
                jpeg_quality=95,
              )
              stats.requests += 1
              predicted_actions = _validated_prediction(
                response.pred_actions,
                horizon=prediction_horizon,
                action_dim=48,
              )
              decoded = decode_action_targets(
                sim,
                relative_to_absolute(states[-1], predicted_actions),
                camera_from_world,
              )
              row = {
                "request": stats.requests,
                "sim_time": float(sim.data.time),
                "control_mode": controller.mode,
                "round_trip_s": time.monotonic() - request_start,
                "server_timing": response.server_timing,
                "right_wrist_current_world": sim.current_pose_matrix("right")[
                  0
                ].tolist(),
                "right_wrist_targets_world": decoded.right.site_positions_world.tolist(),
              }
              log.write(json.dumps(row) + "\n")
              log.flush()
              if stats.requests == 1 and args.save_first_request:
                camera_fields = {
                  f"{name}_images": selected.selected_images[name]
                  for name in camera_names
                }
                intrinsic_fields = {
                  f"{name}_camera_intrinsics": calibrations[name].intrinsic
                  for name in camera_names
                }
                if camera_names == ("head",):
                  camera_fields["images"] = selected.selected_images["head"]
                  intrinsic_fields["camera_intrinsics"] = calibrations[
                    "head"
                  ].intrinsic
                np.savez_compressed(
                  output / "first_request.npz",
                  states=states,
                  predicted_actions=response.pred_actions,
                  **camera_fields,
                  **intrinsic_fields,
                )
              print(
                f"[policy] request={stats.requests} sim_t={sim.data.time:.3f}s "
                f"round_trip={row['round_trip_s']:.2f}s",
                flush=True,
              )
              for index in range(args.execute_steps):
                if sim.data.time >= args.max_sim_seconds:
                  break
                if viewer is not None and not viewer.is_running():
                  raise ViewerClosed("native viewer closed")
                tick_start = time.monotonic()
                controller.before_command()
                _command_action_step(sim, decoded, index, args, stats)
                controller.after_command(decoded.right, index)
                control_tick += 1
                while sim.data.time < control_tick / 30.0 - 1e-12:
                  sim.step()
                  controller.after_step()
                  review_metrics.update(sim)
                  if monitor is not None:
                    observe_simulation(
                      monitor, controller, report["initial_card_pose"][0]
                    )
                    if monitor.success:
                      break
                  if not np.all(np.isfinite(sim.data.qpos)) or not np.all(
                    np.isfinite(sim.data.qvel)
                  ):
                    raise RuntimeError("nonfinite simulation state")
                  if sim.object_pose("card")[2] < 0.5:
                    raise RuntimeError("card fell below work surface")
                stats.action_steps += 1
                stats.maximum_object_height = max(
                  stats.maximum_object_height, float(sim.object_pose("card")[2])
                )
                calibrations = _append_model_observations(
                  sim, renderer, model_cameras, histories
                )
                if audit_log is not None:
                  from kaihand_tactile_env.tasks.poker_draw.execution_audit import (
                    execution_sample,
                  )

                  audit_log.write(
                    json.dumps(execution_sample(sim, controller, decoded.right, index))
                    + "\n"
                  )
                  audit_log.flush()
                if recorder is not None:
                  recorder.capture(control_tick, controller.mode)
                if (
                  viewer is not None
                  and time.monotonic() - last_sync >= 1.0 / args.viewer_hz
                ):
                  viewer.sync()
                  last_sync = time.monotonic()
                if args.real_time:
                  await asyncio.sleep(
                    max(0.0, 1.0 / 30.0 - (time.monotonic() - tick_start))
                  )
                if monitor is not None and monitor.success:
                  stats.stable_success = True
                  break
            report["sim_seconds"] = float(sim.data.time)
            report["final_card_pose"] = sim.object_pose("card").tolist()
            report["last_drive_state"] = sim.drive_state()
      report["status"] = (
        ("success" if monitor.success else "task_not_completed")
        if monitor
        else "preview_completed_not_task_acceptance"
      )
  except ViewerClosed as error:
    report.update(status="viewer_closed", error=str(error))
  except Exception as error:
    report.update(status="error", error=f"{type(error).__name__}: {error}")
    raise
  finally:
    if audit_log is not None:
      audit_log.close()
    if sim is not None:
      report["sim_seconds"] = float(sim.data.time)
      report["final_card_pose"] = sim.object_pose("card").tolist()
      report["last_drive_state"] = sim.drive_state()
    if controller is not None:
      report["controller_report"] = controller.report()
      report["terminal_metrics"] = controller.terminal_metrics()
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_supported_table_penetration_m": (
        controller.maximum_supported_table_penetration_m
        if controller is not None else 0.0
      ),
      "first_limit_exceeded_sim_time_s": (
        controller.first_penetration_limit_exceeded_s
        if controller is not None else None
      ),
    }
    if monitor is not None:
      report["evaluation"] = monitor.report()
    report.update(stats=asdict(stats), wall_seconds=time.monotonic() - started)
    if recorder is not None:
      try:
        recorder.capture(control_tick, controller.mode, force=True)
      except Exception as error:
        report["record_error"] = str(error)
      try:
        report["video"] = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=report.get("error") or report.get("record_error"),
        )
        report["video"].update(
          seed=args.seed,
          success=bool(report.get("evaluation", {}).get("success", False)),
          inference_requests=stats.requests,
          simulation_time_s=report.get("sim_seconds"),
          wall_time_s=report["wall_seconds"],
          model_family=report.get("model_family"),
          checkpoint_path=report.get("checkpoint_path"),
          checkpoint_sha256=report.get("checkpoint_sha256"),
          prediction_horizon=report.get("prediction_horizon"),
          action_dim=48,
          execute_steps=args.execute_steps,
          control_hz=30,
          replan_period_s=args.execute_steps / 30,
          diagnostic_only=bool(args.disable_penetration_guard),
          formal_metrics_valid=not args.disable_penetration_guard,
          penetration_guard_enabled=not args.disable_penetration_guard,
          penetration_guard_threshold_m=PENETRATION_GUARD_THRESHOLD_M,
          maximum_supported_table_penetration_m=(
            report["penetration_diagnostics"]["maximum_supported_table_penetration_m"]
          ),
          first_penetration_limit_exceeded_sim_time_s=(
            report["penetration_diagnostics"]["first_limit_exceeded_sim_time_s"]
          ),
        )
        (output / "review" / "review.json").write_text(
          json.dumps(report["video"], ensure_ascii=False, indent=2),
          encoding="utf-8",
        )
      except Exception as error:
        report["record_error"] = str(error)
    with (output / "summary.json").open("x") as file:
      json.dump(report, file, indent=2)
    print(
      f"[report] {output / 'summary.json'}; action_steps={stats.action_steps}",
      flush=True,
    )


def main():
  asyncio.run(run(parse_args()))


if __name__ == "__main__":
  main()
