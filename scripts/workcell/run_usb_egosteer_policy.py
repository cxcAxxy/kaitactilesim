#!/usr/bin/env python3
"""Run an EgoSteer policy in the USB-insert MuJoCo task."""

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
from kaihand_tactile_env.shared.policy_cameras import (
  include_model_right_wrist_panel,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config as usb_config
from kaihand_tactile_env.tasks.usb_insert.review_metrics import UsbPolicyOutcome
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion
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
  "Grasp the USB plug with the right hand, lift and align it with the "
  "upward-facing socket, insert it until seated, then release it and "
  "withdraw the hand."
)
CONTROL_HZ = 30
CONTROLLER = "direct-30hz-egosteer-taskspace-v1"
PENETRATION_GUARD_THRESHOLD_M = 0.0003


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:18782")
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--viewer", choices=("native", "none"), default="native")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  parser.add_argument("--execute-steps", type=int, default=5)
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
  parser.add_argument("--response-timeout", type=float, default=180.0)
  parser.add_argument("--viewer-hz", type=float, default=30.0)
  parser.add_argument(
    "--real-time", action=argparse.BooleanOptionalAction, default=None
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
  parser.add_argument("--max-wrist-jump", type=float, default=0.15)
  parser.add_argument("--max-arm-position-error", type=float, default=0.03)
  parser.add_argument("--max-arm-orientation-error", type=float, default=0.30)
  parser.add_argument("--hand-ik-tolerance", type=float, default=0.0015)
  parser.add_argument("--max-hand-error", type=float, default=0.02)
  parser.add_argument("--strict-ik", action="store_true")
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  for key in ("xy_jitter_mm", "yaw_jitter_deg"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0:
      parser.error(f"{key} must be finite and nonnegative")
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


def load_deployment_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  required = (
    "task",
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
  if not isinstance(payload, dict):
    raise RuntimeError("deployment manifest must contain a JSON object")
  missing = [key for key in required if key not in payload]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if payload["task"] != "usb-insert":
    raise RuntimeError(f"USB runner received task={payload['task']!r} manifest")
  return resolved, payload


def validate_server(metadata, *, execute_steps: int, deployment: dict) -> int:
  horizon = _model_action_horizon(metadata)
  expected = {
    "task": "usb-insert",
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
  observation = metadata["observation_contract"]
  _model_camera_names(metadata)
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("this USB checkpoint contract must not receive tactile input")
  if metadata["action_dim"] != 48:
    raise RuntimeError("USB EgoSteer runner requires action_dim=48")
  return horizon


def _guard_usb_state(
  outcome: UsbPolicyOutcome, simulation, *, penetration_guard_enabled: bool = True
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


async def run(args):
  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "task": "usb-insert",
    "controller": CONTROLLER,
    "server": args.server,
    "seed": args.seed,
    "instruction": INSTRUCTION,
    "deployment_manifest": str(deployment_path),
    "deployment_id": deployment["deployment_id"],
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "model_family": deployment["model_family"],
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
    "training_controller_equivalent": False,
    "expert_used": False,
    "tactile_sent_to_model": False,
    "low_level_tactile_feedback": False,
    "render_shadows": False,
  }
  stats = RolloutStats()
  started = time.monotonic()
  simulation = recorder = outcome = None
  control_tick = 0
  run_error = None
  first_penetration_limit_exceeded_s = None
  try:
    async with EgoSteerPolicyClient(
      args.server, response_timeout=args.response_timeout
    ) as client:
      metadata = client.metadata or {}
      prediction_horizon = validate_server(
        metadata, execute_steps=args.execute_steps, deployment=deployment
      )
      report["server_metadata"] = metadata
      report["prediction_horizon"] = prediction_horizon
      observation = metadata["observation_contract"]
      camera_names = _model_camera_names(metadata)
      print(f"[policy] connected: {metadata}; controller={CONTROLLER}", flush=True)

      simulation = ArmHandSimulation(scene="usb-insert")
      simulation.reset(seed=args.seed)
      report["initial_randomization"] = initialize_for_insertion(
        simulation,
        seed=args.seed,
        xy_jitter_m=args.xy_jitter_mm / 1000.0,
        yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
      )
      report["initial_plug_pose"] = simulation.object_pose("usb_plug").tolist()
      stats.maximum_object_height = float(report["initial_plug_pose"][2])
      outcome = UsbPolicyOutcome(simulation)
      if outcome.state.maximum_socket_penetration_m > PENETRATION_GUARD_THRESHOLD_M:
        first_penetration_limit_exceeded_s = float(simulation.data.time)
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
          horizon=int(observation["image_history"]),
          stride=int(observation["image_stride"]),
        )
        for name in camera_names
      }
      with (
        WorkcellRenderer(
          simulation.model, tuple(model_cameras.values()), shadows=False
        ) as renderer,
        _viewer_context(simulation, args.viewer) as viewer,
      ):
        report["render_backend"] = renderer.backend_info
        calibrations = _append_model_observations(
          simulation, renderer, model_cameras, histories
        )
        report["camera_intrinsics"] = {
          name: calibrations[name].intrinsic.tolist() for name in camera_names
        }
        report["head_intrinsic"] = calibrations["head"].intrinsic.tolist()
        report["head_world_from_camera"] = (
          calibrations["head"].world_from_camera.tolist()
        )
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
            include_model_wrist=include_model_right_wrist_panel(
              camera_names, args.review_second_camera
            ),
            metrics=outcome,
            heading="USB INSERT MODEL EVALUATION",
            metadata={
              "server": args.server,
              "server_metadata": metadata,
              "seed": args.seed,
              "model_family": deployment["model_family"],
              "checkpoint_path": deployment["checkpoint_path"],
              "checkpoint_sha256": deployment["checkpoint_sha256"],
              "prediction_horizon": prediction_horizon,
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
        if viewer is not None:
          viewer.sync()

        with (output / "requests.jsonl").open("x") as log:
          while simulation.data.time < args.max_sim_seconds and not outcome.success:
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
            predicted = _validated_prediction(
              response.pred_actions,
              horizon=prediction_horizon,
              action_dim=deployment["action_dim"],
            )
            decoded = decode_action_targets(
              simulation,
              relative_to_absolute(states[-1], predicted),
              camera_from_world,
            )
            row = {
              "request": stats.requests,
              "sim_time": float(simulation.data.time),
              "stage": outcome.stage(),
              "round_trip_s": time.monotonic() - request_start,
              "server_timing": response.server_timing,
              "right_wrist_current_world": simulation.current_pose_matrix("right")[
                0
              ].tolist(),
              "right_wrist_targets_world": (
                decoded.right.site_positions_world.tolist()
              ),
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
              f"[policy] request={stats.requests} sim_t={simulation.data.time:.3f}s "
              f"round_trip={row['round_trip_s']:.2f}s stage={outcome.stage()}",
              flush=True,
            )

            for action_index in range(args.execute_steps):
              if simulation.data.time >= args.max_sim_seconds or outcome.success:
                break
              if viewer is not None and not viewer.is_running():
                raise ViewerClosed("native viewer closed")
              tick_started = time.monotonic()
              _command_action_step(
                simulation, decoded, action_index, args, stats
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
                _guard_usb_state(
                  outcome,
                  simulation,
                  penetration_guard_enabled=not args.disable_penetration_guard,
                )
                if outcome.success:
                  stats.stable_success = True
                  break
              stats.action_steps += 1
              stats.maximum_object_height = max(
                stats.maximum_object_height,
                float(simulation.object_pose("usb_plug")[2]),
              )
              calibrations = _append_model_observations(
                simulation, renderer, model_cameras, histories
              )
              if recorder is not None:
                recorder.capture(control_tick, outcome.stage())
              if viewer is not None:
                viewer.sync()
              if args.real_time:
                await asyncio.sleep(
                  max(0.0, 1.0 / CONTROL_HZ - (time.monotonic() - tick_started))
                )

        report["status"] = "success" if outcome.success else "task_not_completed"
  except ViewerClosed as error:
    report.update(status="viewer_closed", error=str(error))
  except Exception as error:
    run_error = f"{type(error).__name__}: {error}"
    report.update(status="error", error=run_error)
    raise
  finally:
    if simulation is not None:
      report["sim_seconds"] = float(simulation.data.time)
      report["final_plug_pose"] = simulation.object_pose("usb_plug").tolist()
    if outcome is not None:
      report["evaluation"] = outcome.report()
      stats.stable_success = outcome.success
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_socket_penetration_m": (
        float(outcome.maximum_socket_penetration_m)
        if outcome is not None else 0.0
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report.update(stats=asdict(stats), wall_seconds=time.monotonic() - started)
    if recorder is not None:
      try:
        recorder.capture(control_tick, outcome.stage(), force=True)
      except Exception as error:
        report["record_error"] = f"{type(error).__name__}: {error}"
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
          model_family=deployment["model_family"],
          checkpoint_path=deployment["checkpoint_path"],
          checkpoint_sha256=deployment["checkpoint_sha256"],
          prediction_horizon=report.get("prediction_horizon"),
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
          json.dumps(report["video"], ensure_ascii=False, indent=2),
          encoding="utf-8",
        )
      except Exception as error:
        report["record_error"] = f"{type(error).__name__}: {error}"
    with (output / "summary.json").open("x") as file:
      json.dump(report, file, indent=2)
    print(
      f"[report] {output / 'summary.json'}; "
      f"success={stats.stable_success} action_steps={stats.action_steps}",
      flush=True,
    )


def main():
  asyncio.run(run(parse_args()))


if __name__ == "__main__":
  main()
