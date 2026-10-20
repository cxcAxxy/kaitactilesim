#!/usr/bin/env python3
"""Run one manifest-bound pi0.5 trial for a shared household task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.action_diagnostics import ActionClipDiagnostics
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.evaluation_guards import (
  DEFAULT_MAX_SIM_SECONDS,
  GUARD_MODES,
  ensure_evaluation_physics,
  evaluation_guard_report,
  recover_nonfinite_action,
)
from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  SHARED_DEPLOYMENT_SCHEMA_V1,
  SHARED_DEPLOYMENT_SCHEMA_V2,
  deployment_diagnostic_reasons,
  inspect_shared_pi05_runtime_integrity,
  validate_shared_deployment_semantics,
)
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)
from kaihand_tactile_env.shared.policy_tasks import (
  POLICY_TASK_INSTRUCTIONS,
  create_task_policy_adapter,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.whiteboard_contact_recorder import (
  finish_whiteboard_contact_recording,
  start_whiteboard_contact_recording,
)
from run_usb_pi05_policy import (
  apply_action,
  joint_limits,
  right_joint_state,
  validated_actions,
)

CONTROL_HZ = 30
TASKS = tuple(POLICY_TASK_INSTRUCTIONS)
ROOT = Path(__file__).resolve().parents[2]
ACTION_INTERPOLATIONS = ("linear", "zoh")
CONTROLLER_VERSION = "pi05-absolute-joint-physics-interpolation-v2"


def current_controller_target(simulation, joint_names: list[str]) -> np.ndarray:
  """Read the active right arm/hand goals in policy joint order."""

  if len(joint_names) != 27 or len(set(joint_names)) != 27:
    raise ValueError("controller target requires 27 unique policy joints")
  expected_arm = [f"right_arm_joint{index}" for index in range(1, 8)]
  if joint_names[:7] != expected_arm:
    raise ValueError("the first seven policy joints must be the ordered right arm")
  try:
    _, all_hand_targets, all_hand_names = simulation.command_state()
    if len(all_hand_names) != len(set(all_hand_names)):
      raise ValueError("simulator controller hand-target names are not unique")
    hand_target_by_name = dict(zip(all_hand_names, all_hand_targets, strict=True))
    target = np.concatenate(
      (
        np.asarray(simulation.arm_goal["right"], dtype=np.float64),
        np.asarray(
          [hand_target_by_name[name] for name in joint_names[7:]],
          dtype=np.float64,
        ),
      )
    )
  except KeyError as error:
    raise ValueError(
      f"simulator has no controller target for {error.args[0]!r}"
    ) from error
  if target.shape != (27,) or not np.isfinite(target).all():
    raise ValueError("simulator controller target must be finite with width 27")
  return target


def linear_substep_action(
  start_action: np.ndarray,
  end_action: np.ndarray,
  *,
  interval_start_s: float,
  interval_end_s: float,
  current_time_s: float,
  timestep_s: float,
) -> np.ndarray:
  """Interpolate adjacent control endpoints for the next physics step."""

  start = np.asarray(start_action, dtype=np.float64)
  end = np.asarray(end_action, dtype=np.float64)
  duration = interval_end_s - interval_start_s
  if (
    start.shape != (27,)
    or end.shape != (27,)
    or not np.isfinite(start).all()
    or not np.isfinite(end).all()
    or not np.isfinite(
      (interval_start_s, interval_end_s, current_time_s, timestep_s)
    ).all()
    or duration <= 0
    or timestep_s <= 0
  ):
    raise ValueError("finite width-27 actions and positive finite timing are required")
  next_time = min(current_time_s + timestep_s, interval_end_s)
  alpha = float(np.clip((next_time - interval_start_s) / duration, 0.0, 1.0))
  return start + alpha * (end - start)


def parse_args(argv=None, *, task_choices=TASKS):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=tuple(task_choices), required=True)
  parser.add_argument("--server", required=True)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--expected-deployment-manifest-sha256")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--staged-evaluation-version", choices=("staged_v1",), default=None)
  parser.add_argument(
    "--action-interpolation",
    choices=ACTION_INTERPOLATIONS,
    default="linear",
    help=(
      "linear (default) interpolates adjacent 30 Hz targets at every physics "
      "step; zoh applies each endpoint immediately for ablation"
    ),
  )
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
  parser.add_argument(
    "--guard-mode", choices=GUARD_MODES, default="observe",
    help="observe records recoverable task guards; strict stops on the first one",
  )
  parser.add_argument(
    "--max-sim-seconds", type=float, default=DEFAULT_MAX_SIM_SECONDS
  )
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument(
    "--contact-reference-hdf5", type=Path,
    help="Whiteboard contact metrics: explicit expert Raw HDF5 reference",
  )
  parser.add_argument(
    "--contact-demo-root", type=Path,
    default=Path("/nas/chenxianchi/datasets/sim/task_demo"),
    help="Whiteboard contact metrics: directory containing expert demo metadata",
  )
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
  parser.add_argument(
    "--diagnostic-relax-whiteboard-contact-guards",
    action="store_true",
    help=(
      "diagnostic-only: observe but do not terminate on the whiteboard 8 N "
      "pad-force and 0.5 N direct-hand contact guards; valid loaded pad "
      "sliding can still accumulate modelled cleaning above 8 N"
    ),
  )
  args = parser.parse_args(argv)
  if args.contact_reference_hdf5 is not None and args.task not in {
    "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
  }:
    parser.error("--contact-reference-hdf5 requires a whiteboard task")
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
  if (
    min(
      args.review_width,
      args.review_height,
      args.review_render_width,
      args.review_render_height,
    )
    <= 0
  ):
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  if args.diagnostic_relax_whiteboard_contact_guards and args.task != "whiteboard-wipe":
    parser.error(
      "--diagnostic-relax-whiteboard-contact-guards is only valid for "
      "--task whiteboard-wipe"
    )
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
    "schema",
    "task",
    "deployment_id",
    "model_family",
    "model_config_name",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_horizon",
    "action_dim",
    "joint_names",
    "observation_contract",
    "action_representation",
    "checkpoint_files",
    "model_project_path",
    "openpi_source_files",
    "openpi_source_sha256",
    "evaluation_source_files",
    "evaluation_source_sha256",
  )
  missing = [key for key in required if key not in deployment]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  family = str(deployment["model_family"]).lower().replace(".", "")
  if (
    deployment["schema"] not in {
      SHARED_DEPLOYMENT_SCHEMA_V1,
      SHARED_DEPLOYMENT_SCHEMA_V2,
    }
    or deployment["task"] != task
    or family not in {"pi05", "π05"}
  ):
    raise RuntimeError(f"runner requires a {task} pi0.5 deployment")
  if int(deployment["action_dim"]) != 27:
    raise RuntimeError("pi0.5 shared-task runner requires 27 absolute joint actions")
  if deployment["observation_contract"].get("tactile_sent_to_model", False):
    raise RuntimeError(
      "this runner does not silently omit declared tactile model input"
    )
  policy_camera_names(deployment["observation_contract"])
  image_shape_hwc(deployment["observation_contract"])
  validate_shared_deployment_semantics(deployment)
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
    "model_config_name": deployment["model_config_name"],
  }
  diagnostic_reasons = deployment_diagnostic_reasons(deployment)
  expected.update(
    training_provenance=deployment.get("training_provenance"),
    required_action_interpolation=deployment.get(
      "required_action_interpolation", "linear"
    ),
    diagnostic_only=bool(diagnostic_reasons),
    formal_metrics_valid=not diagnostic_reasons,
    diagnostic_reasons=diagnostic_reasons,
  )
  transfer = deployment.get("training_provenance", {}).get("completion_protocol_transfer")
  if transfer is not None:
    expected["completion_protocol_transfer"] = transfer
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


def _evaluation_validity_metadata(
  args,
  deployment: dict | None = None,
  diagnostic_context: str | None = None,
) -> dict[str, object]:
  reasons = (
    deployment_diagnostic_reasons(
      deployment,
      action_interpolation=getattr(args, "action_interpolation", "linear"),
    )
    if deployment is not None
    else []
  )
  if diagnostic_context is not None:
    if not isinstance(diagnostic_context, str) or not diagnostic_context.strip():
      raise ValueError("diagnostic_context must be a nonempty string")
    diagnostic_context = diagnostic_context.strip()
    reasons.append(diagnostic_context)
  elif args.diagnostic_relax_whiteboard_contact_guards:
    diagnostic_context = "relaxed_whiteboard_contact_guards"
    reasons.append(diagnostic_context)
  if getattr(args, "guard_mode", "observe") == "observe":
    reasons.append("guard_mode_observe")
  reasons = list(dict.fromkeys(reasons))
  diagnostic_only = bool(reasons)
  return {
    "diagnostic_only": diagnostic_only,
    "formal_metrics_valid": not diagnostic_only,
    "diagnostic_context": diagnostic_context,
    "diagnostic_reasons": reasons,
    "guard_mode": getattr(args, "guard_mode", "observe"),
    "whiteboard_contact_guards_enabled": (
      not args.diagnostic_relax_whiteboard_contact_guards
      if args.task == "whiteboard-wipe"
      else None
    ),
  }


def run(
  args,
  *,
  diagnostic_context: str | None = None,
  policy_binding=None,
) -> dict:
  deployment_path, deployment = (
    load_deployment(args.deployment_manifest, args.task)
    if policy_binding is None
    else policy_binding.load_manifest(args.deployment_manifest)
  )
  if deployment.get("task") != args.task:
    raise RuntimeError(
      f"deployment task {deployment.get('task')!r} does not match --task {args.task!r}"
    )
  if policy_binding is not None and (
    args.pre_roll_steps != 0 or args.pre_roll_actions is not None
  ):
    raise ValueError("tactile policy trials do not permit expert pre-roll actions")
  binding_identity = (
    None if policy_binding is None else policy_binding.report_identity()
  )
  if (
    binding_identity is not None
    and args.expected_deployment_manifest_sha256 is not None
    and binding_identity.get("deployment_manifest_sha256")
    != args.expected_deployment_manifest_sha256
  ):
    raise RuntimeError("tactile deployment manifest hash differs from parent batch")
  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  instruction = deployment.get("instruction", POLICY_TASK_INSTRUCTIONS[args.task])
  if policy_binding is None:
    validity = _evaluation_validity_metadata(args, deployment, diagnostic_context)
  else:
    diagnostic_reasons = policy_binding.diagnostic_reasons(
      args.action_interpolation
    )
    if diagnostic_context is not None:
      if not isinstance(diagnostic_context, str) or not diagnostic_context.strip():
        raise ValueError("diagnostic_context must be a nonempty string")
      diagnostic_reasons.append(diagnostic_context.strip())
    if args.guard_mode == "observe":
      diagnostic_reasons.append("guard_mode_observe")
    diagnostic_reasons = list(dict.fromkeys(diagnostic_reasons))
    validity = {
      "diagnostic_only": bool(diagnostic_reasons),
      "formal_metrics_valid": not diagnostic_reasons,
      "diagnostic_context": diagnostic_context,
      "diagnostic_reasons": diagnostic_reasons,
      "guard_mode": args.guard_mode,
      "whiteboard_contact_guards_enabled": (
        not args.diagnostic_relax_whiteboard_contact_guards
        if args.task == "whiteboard-wipe"
        else None
      ),
    }
  report = {
    "task": args.task,
    "controller": CONTROLLER_VERSION,
    "action_interpolation": args.action_interpolation,
    "controller_contract": {
      "endpoint_rate_hz": CONTROL_HZ,
      "initial_interpolation_start": "current simulator arm/hand controller targets",
      "subsequent_interpolation_start": (
        "previous clipped action endpoint, persistent across chunks and replans"
      ),
      "endpoint_clipping": "once per 30 Hz action before physics substeps",
      "physics_substeps": (
        "linear interpolation at every simulator physics step"
        if args.action_interpolation == "linear"
        else "zero-order hold of each clipped endpoint"
      ),
    },
    "model_family": deployment["model_family"],
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "instruction": instruction,
    "completion_protocol_transfer": (
      deployment.get("training_provenance", {}).get("completion_protocol_transfer")
    ),
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
    "guard_events": [],
    "stop_reason": None,
    "physics_valid": None,
    **validity,
  }
  if args.task in {
    "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
  }:
    report.update(
      contact_reference_hdf5=(
        str(args.contact_reference_hdf5.expanduser().absolute())
        if args.contact_reference_hdf5 is not None else None
      ),
      contact_demo_root=str(args.contact_demo_root.expanduser().absolute()),
    )
  if binding_identity is not None:
    report.update(binding_identity)
  started = time.monotonic()
  client = adapter = recorder = clip_diagnostics = None
  staged_log = None
  staged_version = getattr(args, "staged_evaluation_version", None)
  report["full_duration_evaluation"] = bool(staged_version)
  contact_recorder = None
  requests = action_steps = clipped_values = control_tick = 0
  pre_roll_action_steps = 0
  request_limit_reached = False
  run_error = None
  last_valid_evaluation = None
  try:
    if policy_binding is None:
      deployment_integrity = inspect_shared_pi05_runtime_integrity(
        deployment_path,
        deployment,
        evaluation_root=ROOT,
        expected_manifest_sha256=args.expected_deployment_manifest_sha256,
      )
      report["deployment_integrity"] = deployment_integrity
      if deployment_integrity["status"] != "verified":
        detail = json.dumps(deployment_integrity, ensure_ascii=False, sort_keys=True)
        raise RuntimeError(
          f"child deployment integrity verification failed: {detail}"
        )

      from openpi_client import websocket_client_policy

      client = websocket_client_policy.WebsocketClientPolicy(args.server)
    else:
      client = policy_binding.start(output)
    metadata = client.get_server_metadata()
    horizon = (
      validate_server(metadata, deployment, args.execute_steps)
      if policy_binding is None
      else policy_binding.validate_server(metadata, args.execute_steps)
    )
    report["server_metadata"] = metadata
    report["prediction_horizon"] = horizon
    adapter = create_task_policy_adapter(
      args.task,
      args.seed,
      guard_mode=args.guard_mode,
      diagnostic_relax_whiteboard_contact_guards=(
        args.diagnostic_relax_whiteboard_contact_guards
      ),
    )
    contact_recorder = start_whiteboard_contact_recording(adapter)
    if staged_version:
      from kaihand_tactile_env.shared.staged_recording import StagedStateLog

      adapter.enable_staged_evaluation(staged_version)
      staged_log = StagedStateLog(output, adapter.staged_evaluation, identity=report)
      staged_log.record(0)
    pre_roll_actions = load_pre_roll_actions(args.pre_roll_actions, args.pre_roll_steps)
    simulation = adapter.simulation
    ensure_evaluation_physics(simulation)
    if policy_binding is not None:
      policy_binding.attach(simulation)
    last_valid_evaluation = adapter.report()
    joint_names = list(deployment["joint_names"])
    lower, upper = joint_limits(simulation, joint_names)
    clip_diagnostics = ActionClipDiagnostics(joint_names, lower, upper)
    initial_controller_target = current_controller_target(simulation, joint_names)
    last_controller_target = np.clip(initial_controller_target, lower, upper)
    control_epoch_s = float(simulation.data.time)
    report["controller_contract"].update(
      physics_timestep_s=simulation.timestep,
      physics_hz=int(round(1.0 / simulation.timestep)),
      control_epoch_s=control_epoch_s,
      initial_controller_target=last_controller_target.tolist(),
      initial_controller_target_clipped_values=int(
        np.count_nonzero(
          np.abs(last_controller_target - initial_controller_target) > 1.0e-9
        )
      ),
    )
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
    with (
      WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer,
      (output / "requests.jsonl").open("x", encoding="utf-8") as log,
      (output / "control_steps.jsonl").open("x", encoding="utf-8") as control_log,
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
          include_model_wrist=(
            "right_wrist" in camera_names and args.review_second_camera != "right_wrist"
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
            "controller": CONTROLLER_VERSION,
            "action_interpolation": args.action_interpolation,
            "controller_contract": report["controller_contract"],
            "seed": args.seed,
            **validity,
          },
        )
        recorder.capture(0, adapter.stage())

      def execute_action(
        action: np.ndarray,
        *,
        source: str,
        request_index: int | None = None,
        chunk_index: int | None = None,
      ) -> None:
        nonlocal action_steps, clipped_values, control_tick
        nonlocal last_controller_target, last_valid_evaluation
        stage_before = adapter.stage()
        clipping = clip_diagnostics.record(action, stage_before)
        endpoint = np.asarray(clipping["applied_action"], dtype=np.float64)
        clipped_values += clipping["clipped_values"]
        control_tick += 1
        target_time = min(
          args.max_sim_seconds,
          control_epoch_s + control_tick / CONTROL_HZ,
        )
        interval_start_time = float(simulation.data.time)
        interval_start_target = last_controller_target.copy()
        physics_steps = 0
        if args.action_interpolation == "zoh":
          if apply_action(simulation, joint_names, endpoint, lower, upper):
            raise RuntimeError("pre-clipped zero-order-hold endpoint was clipped")
          last_controller_target = endpoint.copy()
        while simulation.data.time < target_time - 1.0e-12:
          if args.action_interpolation == "linear":
            substep_action = linear_substep_action(
              interval_start_target,
              endpoint,
              interval_start_s=interval_start_time,
              interval_end_s=target_time,
              current_time_s=float(simulation.data.time),
              timestep_s=simulation.timestep,
            )
            if apply_action(simulation, joint_names, substep_action, lower, upper):
              raise RuntimeError("interpolation between clipped endpoints was clipped")
            last_controller_target = substep_action
          adapter.step()
          physics_steps += 1
          if adapter.success and not staged_version:
            break
        last_valid_evaluation = adapter.report()
        action_steps += 1
        stage_after = adapter.stage()
        control_log.write(
          json.dumps(
            {
              "control_tick": control_tick,
              "source": source,
              "request": request_index,
              "chunk_index": chunk_index,
              "sim_time": float(simulation.data.time),
              "stage_before_action": stage_before,
              "stage_after_action": stage_after,
              "action": clipping,
              "controller": {
                "version": CONTROLLER_VERSION,
                "interpolation": args.action_interpolation,
                "physics_steps": physics_steps,
                "interval_start_time_s": interval_start_time,
                "interval_target_time_s": target_time,
                "interval_start_target": interval_start_target.tolist(),
                "last_applied_target": last_controller_target.tolist(),
              },
              "task": adapter.snapshot(),
            },
            ensure_ascii=False,
          )
          + "\n"
        )
        control_log.flush()
        if staged_log is not None:
          staged_log.record(control_tick)
        if recorder is not None:
          recorder.capture(control_tick, stage_after)

      for action in pre_roll_actions:
        if simulation.data.time >= args.max_sim_seconds or (adapter.success and not staged_version):
          break
        execute_action(
          action,
          source="pre_roll",
          chunk_index=pre_roll_action_steps,
        )
        pre_roll_action_steps += 1

      while simulation.data.time < args.max_sim_seconds and (staged_version or not adapter.success):
        if args.max_requests and requests >= args.max_requests:
          request_limit_reached = True
          break
        images = {
          name: renderer.capture(simulation.data, camera)["rgb"]
          for name, camera in cameras.items()
        }
        state = right_joint_state(simulation, joint_names)
        request_started = time.monotonic()
        observation_sim_time = float(simulation.data.time)
        request = (
          {
            **pi05_image_payload(images, observation),
            "state": state,
            "prompt": instruction,
          }
          if policy_binding is None
          else policy_binding.request(
            simulation,
            images,
            state,
            args.seed,
            requests,
            args.execute_steps,
          )
        )
        response = client.infer(request)
        if (
          policy_binding is not None
          and float(simulation.data.time) != observation_sim_time
        ):
          raise RuntimeError(
            "simulation advanced during synchronized observation/inference"
          )
        requests += 1
        try:
          actions = validated_actions(
            response,
            horizon=horizon,
            action_dim=int(deployment["action_dim"]),
          )
        except Exception as error:
          if not recover_nonfinite_action(simulation, error):
            raise
          actions = None
        row = {
          "request": requests,
          "sim_time": float(simulation.data.time),
          "stage": adapter.stage(),
          "round_trip_s": time.monotonic() - request_started,
          "server_timing": response.get("server_timing", {}),
          **(
            {}
            if policy_binding is None
            else policy_binding.request_log(request, response)
          ),
          "state": state.tolist(),
          "task": adapter.snapshot(),
          "predicted_action_min": (
            None if actions is None else np.min(actions, axis=0).tolist()
          ),
          "predicted_action_max": (
            None if actions is None else np.max(actions, axis=0).tolist()
          ),
          "policy_action_rejected": actions is None,
        }
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        if requests == 1 and args.save_first_request and actions is not None:
          np.savez_compressed(
            output / "first_request.npz",
            **{f"{name}_image": image for name, image in images.items()},
            state=state,
            predicted_actions=actions,
            **(
              {}
              if policy_binding is None
              else policy_binding.request_arrays(request)
            ),
          )
        if actions is None:
          stage_before = adapter.stage()
          control_tick += 1
          target_time = min(
            args.max_sim_seconds,
            control_epoch_s + control_tick / CONTROL_HZ,
          )
          interval_start_time = float(simulation.data.time)
          physics_steps = 0
          while simulation.data.time < target_time - 1.0e-12:
            adapter.step()
            physics_steps += 1
            if adapter.success and not staged_version:
              break
          last_valid_evaluation = adapter.report()
          action_steps += 1
          control_log.write(
            json.dumps(
              {
                "control_tick": control_tick,
                "source": "policy_rejected",
                "request": requests,
                "chunk_index": None,
                "sim_time": float(simulation.data.time),
                "stage_before_action": stage_before,
                "stage_after_action": adapter.stage(),
                "action": None,
                "controller": {
                  "version": CONTROLLER_VERSION,
                  "interpolation": args.action_interpolation,
                  "physics_steps": physics_steps,
                  "interval_start_time_s": interval_start_time,
                  "interval_target_time_s": target_time,
                  "interval_start_target": last_controller_target.tolist(),
                  "last_applied_target": last_controller_target.tolist(),
                  "recovery": "hold_last_valid_command_and_replan",
                },
                "task": adapter.snapshot(),
              },
              ensure_ascii=False,
            )
            + "\n"
          )
          control_log.flush()
          if staged_log is not None:
            staged_log.record(control_tick)
          if recorder is not None:
            recorder.capture(control_tick, adapter.stage())
          continue
        action_start_index = (
          0 if policy_binding is None else policy_binding.action_start_index
        )
        selected_actions = actions[
          action_start_index : action_start_index + args.execute_steps
        ]
        if len(selected_actions) != args.execute_steps:
          raise RuntimeError(
            "decoded action horizon does not contain the frozen execution window"
          )
        for chunk_index, action in enumerate(selected_actions):
          if simulation.data.time >= args.max_sim_seconds or (adapter.success and not staged_version):
            break
          execute_action(
            action,
            source="policy",
            request_index=requests,
            chunk_index=chunk_index,
          )
      report["status"] = "success" if adapter.success else "task_not_completed"
      last_valid_evaluation = adapter.report()
      report["evaluation"] = last_valid_evaluation
  except Exception as error:
    run_error = f"{type(error).__name__}: {error}"
    report.update(status="error", error=run_error)
    if adapter is None:
      report["stop_reason"] = "fatal_error"
    raise
  finally:
    if client is not None:
      try:
        if policy_binding is None:
          client._ws.close()
        else:
          client.close()
      except Exception:
        pass
    if adapter is not None:
      guard_report = evaluation_guard_report(
        adapter.simulation,
        status=report.get("status"),
        error=run_error,
      )
      sim_seconds = float(adapter.simulation.data.time)
      report["sim_seconds"] = sim_seconds if np.isfinite(sim_seconds) else None
      if guard_report["physics_valid"]:
        last_valid_evaluation = adapter.report()
      if last_valid_evaluation is not None:
        last_valid_evaluation = {
          **last_valid_evaluation,
          **guard_report,
        }
      report["evaluation"] = last_valid_evaluation
      task_success = bool(
        last_valid_evaluation and last_valid_evaluation.get("success", False)
      )
      report["task_success"] = task_success
      guard_report["formal_metrics_valid"] = bool(
        report.get("formal_metrics_valid", True)
        and guard_report["formal_metrics_valid"]
      )
      guard_report["diagnostic_only"] = bool(
        report.get("diagnostic_only", False)
        or guard_report["diagnostic_only"]
      )
      report.update(guard_report)
      if request_limit_reached and not task_success:
        report["stop_reason"] = "request_limit"
    action_clipping = (
      clip_diagnostics.summary() if clip_diagnostics is not None else None
    )
    report["stats"] = {
      "requests": requests,
      "action_steps": action_steps,
      "pre_roll_action_steps": pre_roll_action_steps,
      "clipped_action_values": clipped_values,
      "action_clipping": action_clipping,
    }
    if action_clipping is not None:
      report["action_clipping_artifact"] = "action_clipping.json"
      (output / "action_clipping.json").write_text(
        json.dumps(action_clipping, ensure_ascii=False, indent=2),
        encoding="utf-8",
      )
    report["wall_seconds"] = time.monotonic() - started
    if recorder is not None:
      recording_errors = []
      if report.get("physics_valid", True):
        try:
          recorder.capture(control_tick, adapter.stage(), force=True)
        except Exception as error:
          recording_errors.append(f"final capture: {type(error).__name__}: {error}")
      try:
        report["video"] = recorder.finish(
          status="error" if recording_errors else report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=run_error or (recording_errors[0] if recording_errors else None),
        )
        if policy_binding is not None:
          report["video"].update(policy_binding.report_identity())
      except Exception as error:
        recording_errors.append(f"finish: {type(error).__name__}: {error}")
      if recording_errors:
        report["record_error"] = "; ".join(recording_errors)
        report.update(
          status="error",
          stop_reason="fatal_error",
          formal_metrics_valid=False,
        )
    if contact_recorder is not None:
      if staged_version and adapter is not None:
        adapter.detach_staged_contact_observer()
      report["contact_intensity"] = finish_whiteboard_contact_recording(
        contact_recorder, output, report,
        reference_hdf5=args.contact_reference_hdf5,
        demo_root=args.contact_demo_root,
      )
    if staged_log is not None:
      staged_log.finish(report)
    if policy_binding is not None:
      policy_binding.finalize_report(report)
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
  return report


if __name__ == "__main__":
  run(parse_args())
