#!/usr/bin/env python3
"""Serial evaluation for the shared Bulb, RAM, Vase, Whiteboard, and Sponge adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from math import isfinite
from pathlib import Path

from kaihand_tactile_env.shared.action_diagnostics import (
  merge_action_clip_summaries,
)
from kaihand_tactile_env.shared.evaluation_guards import (
  DEFAULT_MAX_SIM_SECONDS,
  GUARD_MODES,
)
from kaihand_tactile_env.shared.evaluation_source_integrity import (
  verify_evaluation_sources,
)
from kaihand_tactile_env.shared.evaluation_video import evaluation_artifact_directory
from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  SHARED_DEPLOYMENT_SCHEMA_V1,
  SHARED_DEPLOYMENT_SCHEMA_V2,
  deployment_diagnostic_reasons,
  stable_file_bytes,
  validate_shared_deployment_semantics,
)
from kaihand_tactile_env.shared.task_demo import DEFAULT_DEMO_ROOT

ROOT = Path(__file__).resolve().parents[2]
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
OPENPI_CLIENT = ROOT.parent / "openpi/packages/openpi-client/src"
PI05_CONTROLLER_VERSION = "pi05-absolute-joint-physics-interpolation-v2"
TASKS = (
  "bulb-screw",
  "install-ram",
  "vase-wipe",
  "whiteboard-wipe",
  "sponge-grasp",
  "usb-unplug",
  "whiteboard-wipe-horizontal",
  "whiteboard-wipe-horizontal-line",
  "bulb-unscrew",
  "bulb-screw-inplace",
)
RUNNERS = {
  "egosteer": ROOT / "scripts/workcell/run_shared_task_egosteer_policy.py",
  "pi05": ROOT / "scripts/workcell/run_shared_task_pi05_policy.py",
  "egotouch": ROOT / "scripts/workcell/run_shared_task_egotouch_policy.py",
}
_REVIEW_ARTIFACTS = ("review.mp4", "review.json", "frames.jsonl")
_COMPARISON_ARTIFACTS = (
  "evaluation_rollout_trace.npz",
  "evaluation_comparison.json",
  "evaluation_right_wrist_state.png",
  "evaluation_right_hand_actuated_dof.png",
  "evaluation_right_fingertip_tactile.png",
)


def _family(value: object) -> str:
  normalized = str(value).lower().replace(".", "")
  aliases = {"pi05": "pi05", "π05": "pi05"}
  return aliases.get(normalized, normalized)


def _enforce_source_integrity(family: str, deployment: dict) -> bool:
  """Require the snapshot for pi0.5 and honor it when another family declares it."""
  declared = any(
    key in deployment for key in ("evaluation_source_files", "evaluation_source_sha256")
  )
  return family == "pi05" or declared


def load_deployment(path: Path) -> tuple[Path, dict]:
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
  if deployment["task"] not in TASKS:
    raise RuntimeError(f"unsupported shared evaluation task: {deployment['task']!r}")
  family = _family(deployment["model_family"])
  if family not in RUNNERS:
    raise RuntimeError(f"unsupported shared evaluation model family: {family!r}")
  if family == "pi05":
    pi05_required = (
      "schema",
      "model_project_path",
      "checkpoint_files",
      "openpi_source_files",
      "openpi_source_sha256",
      "evaluation_source_files",
      "evaluation_source_sha256",
    )
    pi05_missing = [key for key in pi05_required if key not in deployment]
    if pi05_missing:
      raise RuntimeError(f"pi0.5 deployment manifest missing fields: {pi05_missing}")
    if deployment["schema"] not in {
      SHARED_DEPLOYMENT_SCHEMA_V1,
      SHARED_DEPLOYMENT_SCHEMA_V2,
    }:
      raise RuntimeError("pi0.5 batch requires a shared-task deployment manifest")
    validate_shared_deployment_semantics(deployment)
  return resolved, deployment


def freeze_deployment_manifest(
  source: Path,
  deployment: dict,
  output: Path,
) -> tuple[Path, str, int]:
  """Create this batch's content-addressed, read-only manifest snapshot."""
  payload, digest = stable_file_bytes(source)
  try:
    observed = json.loads(payload)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise RuntimeError(f"deployment manifest is not stable JSON: {source}") from error
  if observed != deployment:
    raise RuntimeError("deployment manifest changed after batch preflight")
  snapshot = output / "deployment_snapshot"
  snapshot.mkdir(mode=0o755)
  destination = snapshot / f"deployment_manifest.{digest}.json"
  with destination.open("xb") as frozen:
    frozen.write(payload)
    frozen.flush()
    os.fsync(frozen.fileno())
  destination.chmod(0o444)
  snapshot.chmod(0o555)
  frozen_payload, frozen_digest = stable_file_bytes(destination)
  if frozen_payload != payload or frozen_digest != digest:
    raise RuntimeError("content-addressed deployment manifest copy verification failed")
  mode = destination.stat().st_mode & 0o777
  if mode & 0o222:
    raise RuntimeError("frozen deployment manifest remains writable")
  if snapshot.stat().st_mode & 0o222:
    raise RuntimeError("frozen deployment manifest directory remains writable")
  return destination.resolve(strict=True), digest, len(payload)


def frozen_manifest_report(path: Path, expected_sha256: str) -> dict:
  report = {
    "status": "verified",
    "path": str(path),
    "expected_sha256": expected_sha256,
    "actual_sha256": None,
    "size": None,
    "file_mode": None,
    "directory_mode": None,
    "error": None,
  }
  try:
    payload, digest = stable_file_bytes(path)
    file_mode = path.stat().st_mode & 0o777
    directory_mode = path.parent.stat().st_mode & 0o777
    report.update(
      actual_sha256=digest,
      size=len(payload),
      file_mode=oct(file_mode),
      directory_mode=oct(directory_mode),
    )
    if digest != expected_sha256:
      raise RuntimeError("frozen deployment manifest SHA-256 changed")
    if file_mode & 0o222 or directory_mode & 0o222:
      raise RuntimeError("frozen deployment manifest or its directory is writable")
  except (OSError, RuntimeError) as error:
    report.update(status="mismatch", error=f"{type(error).__name__}: {error}")
  return report


def classify(report: dict) -> tuple[str, bool]:
  if report.get("record_error"):
    return "record_error", False
  guard_events = report.get("guard_events")
  if not isinstance(guard_events, list):
    guard_events = []
  event_kinds = [
    event.get("kind")
    for event in guard_events
    if isinstance(event, dict) and isinstance(event.get("kind"), str)
  ]
  if report.get("physics_valid") is False:
    return (event_kinds[-1] if event_kinds else "physics_invalid"), False
  if (
    report.get("status") == "success"
    and (
      report.get("task_success") is True
      or report.get("evaluation", {}).get("success") is True
    )
  ):
    return "success", True
  if report.get("stop_reason") == "request_limit":
    return "request_limit", True
  if (
    report.get("stop_reason") == "time_limit"
    or report.get("status") == "task_not_completed"
  ):
    return "time_limit", True
  if report.get("stop_reason") == "guard":
    return (event_kinds[-1] if event_kinds else "guard"), True
  if report.get("stop_reason") == "fatal_error":
    return "fatal_error", False
  # Guard messages originate in several task adapters and are not guaranteed
  # to use the same capitalization.  They are task terminations, so classify
  # them as valid failed trials instead of infrastructure errors.
  error = str(report.get("error", "")).casefold()
  for needle, reason in (
    ("wrist target jump", "wrist_target_guard"),
    ("arm IK failed", "arm_ik_guard"),
    ("hand IK", "hand_ik_guard"),
    ("hand target is too far", "hand_ik_guard"),
    ("fell below", "object_dropped"),
    ("force exceeded", "force_guard"),
    ("resistance exceeded", "force_guard"),
    ("penetration exceeded", "penetration_guard"),
    ("Sponge penetration", "penetration_guard"),
    ("Hand/table penetration", "penetration_guard"),
    ("Hand/environment penetration", "penetration_guard"),
    ("Hand struck", "collision_guard"),
    ("Sponge element inverted", "sponge_integrity_guard"),
    ("nonfinite simulation", "nonfinite_state"),
  ):
    if needle.casefold() in error:
      return reason, True
  return "infrastructure_or_contract_error", False


def validate_recorded_review(
  trial: Path,
  deployment: dict,
  reference_dataset: Path | None,
  reference_episode_index: int,
  diagnostic_only: bool = False,
) -> str | None:
  """Return an invalid-trial reason when recorded artifacts break the contract."""
  review = evaluation_artifact_directory(trial)
  missing = [
    name
    for name in _REVIEW_ARTIFACTS
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if missing:
    return "incomplete_review_artifact"
  try:
    metadata = json.loads((review / "review.json").read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    return "invalid_review_metadata"
  if not isinstance(metadata, dict):
    return "invalid_review_metadata"
  expected_model_views = ["head"]
  if "right_wrist" in deployment["observation_contract"].get("cameras", []):
    expected_model_views.append("right_wrist")
  if metadata.get("model_input_cameras_displayed") != expected_model_views:
    return "review_omitted_model_camera"
  if (
    metadata.get("diagnostic_only", False) is not diagnostic_only
    or metadata.get("formal_metrics_valid", True) is diagnostic_only
  ):
    return "review_diagnostic_mode_mismatch"
  if reference_dataset is None:
    return None

  comparison_missing = [
    name
    for name in _COMPARISON_ARTIFACTS
    if not (review / name).is_file() or (review / name).stat().st_size == 0
  ]
  if comparison_missing:
    return "incomplete_comparison_artifact"
  try:
    recorded_reference = Path(metadata["reference_dataset"]).expanduser().resolve()
  except (KeyError, TypeError, ValueError, OSError):
    return "reference_comparison_mismatch"
  if (
    recorded_reference != reference_dataset
    or metadata.get("reference_episode_index") != reference_episode_index
  ):
    return "reference_comparison_mismatch"
  comparison = metadata.get("comparison_plots")
  if not isinstance(comparison, dict) or comparison.get("status") != "ok":
    return "reference_comparison_unavailable"
  return None


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seeds", nargs="+", type=int, required=True)
  parser.add_argument("--video-count", type=int, required=True)
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument(
    "--contact-reference-hdf5", type=Path,
    help="Whiteboard contact metrics: explicit expert Raw HDF5 reference",
  )
  parser.add_argument(
    "--contact-demo-root", type=Path,
    default=DEFAULT_DEMO_ROOT,
    help="Whiteboard contact metrics: directory containing expert demo metadata",
  )
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument(
    "--action-interpolation",
    choices=("linear", "zoh"),
    default="linear",
    help=(
      "pi0.5 only: interpolate adjacent 30 Hz action endpoints at physics rate "
      "(default), or use zero-order hold for ablation"
    ),
  )
  parser.add_argument(
    "--guard-mode", choices=GUARD_MODES, default="observe",
    help="observe records recoverable task guards; strict stops on the first one",
  )
  parser.add_argument(
    "--max-sim-seconds", type=float, default=DEFAULT_MAX_SIM_SECONDS
  )
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--server")
  parser.add_argument("--model-python")
  parser.add_argument("--model-project", type=Path)
  parser.add_argument("--snapshot", type=Path)
  parser.add_argument(
    "--diagnostic-relax-whiteboard-contact-guards",
    action="store_true",
    help=(
      "pi0.5 Whiteboard diagnostic only: continue after the 8 N pad-force "
      "and 0.5 N direct-hand contact guards; valid loaded pad sliding can "
      "still accumulate modelled cleaning above 8 N"
    ),
  )
  args, passthrough = parser.parse_known_args(argv)
  if args.contact_reference_hdf5 is not None:
    args.contact_reference_hdf5 = args.contact_reference_hdf5.expanduser().absolute()
  args.contact_demo_root = args.contact_demo_root.expanduser().absolute()
  if args.reference_dataset is not None:
    args.reference_dataset = args.reference_dataset.expanduser().resolve()
  if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
    parser.error("--seeds must contain distinct nonnegative integers")
  if not 0 <= args.video_count <= len(args.seeds):
    parser.error("video-count must be between zero and the number of seeds")
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if args.reference_episode_index < 0:
    parser.error("reference-episode-index must be nonnegative")
  if (
    not isfinite(args.max_sim_seconds)
    or not isfinite(args.trial_wall_limit)
    or args.max_sim_seconds <= 0
    or args.trial_wall_limit <= 0
  ):
    parser.error("time limits must be positive")
  reserved = {
    "--task",
    "--seed",
    "--output-dir",
    "--deployment-manifest",
    "--expected-deployment-manifest-sha256",
    "--action-interpolation",
    "--guard-mode",
    "--record",
    "--no-record",
  }
  conflict = sorted(
    option
    for option in reserved
    if any(value == option or value.startswith(option + "=") for value in passthrough)
  )
  if conflict:
    parser.error(f"batch-owned options cannot be passed through: {conflict}")
  return parser, args, passthrough


def _environment(openpi_root: Path | None = None) -> dict[str, str]:
  egl_vendor = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", str(EGL_VENDOR))
  egl_vendor_paths = [Path(value) for value in egl_vendor.split(os.pathsep) if value]
  missing_egl_vendors = [path for path in egl_vendor_paths if not path.is_file()]
  if not egl_vendor_paths or missing_egl_vendors:
    missing = missing_egl_vendors or [Path(egl_vendor)]
    raise FileNotFoundError(f"NVIDIA EGL vendor configuration is missing: {missing}")
  openpi_client = (
    OPENPI_CLIENT
    if openpi_root is None
    else openpi_root.expanduser().resolve(strict=True) / "packages/openpi-client/src"
  )
  if not (openpi_client / "openpi_client/__init__.py").is_file():
    raise FileNotFoundError(f"OpenPI client source is missing: {openpi_client}")
  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "LP_NUM_THREADS": "1",
    "MUJOCO_GL": "egl",
    "KAIHAND_RENDER_BACKEND": "hardware",
    "__EGL_VENDOR_LIBRARY_FILENAMES": egl_vendor,
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
  }
  pythonpath = [str(ROOT / "src"), str(openpi_client)]
  if environment.get("PYTHONPATH"):
    pythonpath.append(environment["PYTHONPATH"])
  environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
  environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
  for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
  ):
    environment.pop(key, None)
  return environment


def _batch_summary(identity, planned, results, started, source_integrity_initial):
  valid_count = sum(item["valid_trial"] for item in results)
  task_successes = sum(item.get("task_success", item["success"]) for item in results)
  successes = sum(item["success"] and item["valid_trial"] for item in results)
  valid_action_clipping = [
    item["action_clipping"]
    for item in results
    if item["valid_trial"] and isinstance(item["action_clipping"], dict)
  ]
  source_integrity_final = source_integrity_initial
  if results:
    source_integrity_final = (
      results[-1].get("evaluation_source_integrity_after")
      or results[-1].get("evaluation_source_integrity_before")
      or source_integrity_initial
    )
  formal_metrics_valid = bool(
    identity.get("formal_metrics_valid", True)
    and len(results) == planned
    and all(item.get("formal_metrics_valid", False) for item in results)
  )
  return {
    **identity,
    "formal_metrics_valid": formal_metrics_valid,
    "planned": planned,
    "attempted": sum(item.get("trial_started", True) for item in results),
    "valid_trials": valid_count,
    "successes": successes,
    "task_successes": task_successes,
    "success_rate": successes / valid_count if valid_count else None,
    "formal_success_rate": (
      successes / valid_count if valid_count and formal_metrics_valid else None
    ),
    "termination_counts": dict(Counter(item["termination"] for item in results)),
    "action_clipping": merge_action_clip_summaries(valid_action_clipping),
    "evaluation_source_integrity": {
      "initial": source_integrity_initial,
      "final": source_integrity_final,
    },
    "wall_seconds": time.monotonic() - started,
    "results": results,
  }


def _write_batch_summary(
  output, identity, planned, results, started, source_integrity_initial
):
  aggregate = _batch_summary(
    identity, planned, results, started, source_integrity_initial
  )
  from kaihand_tactile_env.shared.staged_recording import add_staged_batch_results

  add_staged_batch_results(aggregate, output, results)
  (output / "summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
  )


def main(argv=None):
  parser, args, passthrough = parse_args(argv)
  deployment_path, deployment = load_deployment(args.deployment_manifest)
  family = _family(deployment["model_family"])
  whiteboard_contact = deployment["task"] in {
    "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
  }
  if args.contact_reference_hdf5 is not None and not whiteboard_contact:
    parser.error("--contact-reference-hdf5 requires a whiteboard deployment")
  contact_identity = (
    {
      "contact_reference_hdf5": (
        str(args.contact_reference_hdf5)
        if args.contact_reference_hdf5 is not None else None
      ),
      "contact_demo_root": str(args.contact_demo_root),
    }
    if whiteboard_contact else {}
  )
  if family in {"egosteer", "pi05"} and not args.server:
    parser.error(f"--server is required for {family}")
  if family == "egotouch" and (not args.model_python or args.model_project is None):
    parser.error("--model-python and --model-project are required for egotouch")
  if args.diagnostic_relax_whiteboard_contact_guards and (
    deployment["task"] != "whiteboard-wipe" or family != "pi05"
  ):
    parser.error(
      "--diagnostic-relax-whiteboard-contact-guards is only valid for a "
      "whiteboard-wipe pi0.5 deployment"
    )
  if args.execute_steps > int(deployment["prediction_horizon"]):
    parser.error(
      f"execute-steps={args.execute_steps} exceeds deployment horizon "
      f"{deployment['prediction_horizon']}"
    )
  diagnostic_reasons = (
    deployment_diagnostic_reasons(
      deployment,
      action_interpolation=args.action_interpolation,
    )
    if family == "pi05"
    else []
  )
  if args.diagnostic_relax_whiteboard_contact_guards:
    diagnostic_reasons.append("relaxed_whiteboard_contact_guards")
  if args.guard_mode == "observe":
    diagnostic_reasons.append("guard_mode_observe")
  diagnostic_reasons = list(dict.fromkeys(diagnostic_reasons))
  diagnostic_only = bool(diagnostic_reasons)
  formal_metrics_valid = not diagnostic_only
  source_integrity_enforced = _enforce_source_integrity(family, deployment)
  source_integrity_initial = verify_evaluation_sources(deployment, ROOT)
  if source_integrity_enforced and source_integrity_initial["status"] != "verified":
    detail = json.dumps(source_integrity_initial, ensure_ascii=False, sort_keys=True)
    raise RuntimeError(f"formal evaluation source integrity failed: {detail}")

  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  original_deployment_path = deployment_path
  deployment_path, manifest_sha256, manifest_size = freeze_deployment_manifest(
    original_deployment_path,
    deployment,
    output,
  )
  deployment_path, frozen_deployment = load_deployment(deployment_path)
  if frozen_deployment != deployment:
    raise RuntimeError("frozen deployment manifest differs from batch preflight")
  deployment = frozen_deployment
  protocol = {
    **contact_identity,
    "task": deployment["task"],
    "model_family": deployment["model_family"],
    "deployment_id": deployment["deployment_id"],
    "deployment_manifest": str(deployment_path),
    "original_deployment_manifest": str(original_deployment_path),
    "deployment_manifest_sha256": manifest_sha256,
    "deployment_manifest_size": manifest_size,
    "deployment_manifest_read_only": True,
    "checkpoint_path": deployment["checkpoint_path"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "reference_dataset": (
      None if args.reference_dataset is None else str(args.reference_dataset)
    ),
    "reference_episode_index": args.reference_episode_index,
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "observation_contract": deployment["observation_contract"],
    "action_representation": deployment["action_representation"],
    "seeds": args.seeds,
    "video_count": args.video_count,
    "video_selection": "first N seeds in explicit --seeds order",
    "record_fps": args.record_fps,
    "execute_steps": args.execute_steps,
    "controller": PI05_CONTROLLER_VERSION if family == "pi05" else None,
    "action_interpolation": (args.action_interpolation if family == "pi05" else None),
    "control_hz": 30,
    "replan_period_s": args.execute_steps / 30,
    "max_sim_seconds": args.max_sim_seconds,
    "guard_mode": args.guard_mode,
    "trial_wall_limit": args.trial_wall_limit,
    "workers": 1,
    "retry": False,
    "evaluation_source_integrity": source_integrity_initial,
    "evaluation_source_integrity_enforced": source_integrity_enforced,
    "diagnostic_only": diagnostic_only,
    "formal_metrics_valid": formal_metrics_valid,
    "diagnostic_reasons": diagnostic_reasons,
    "whiteboard_contact_guards_enabled": (
      not args.diagnostic_relax_whiteboard_contact_guards
      if deployment["task"] == "whiteboard-wipe"
      else None
    ),
  }
  (output / "protocol.json").write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
  )

  identity = {
    **contact_identity,
    "task": deployment["task"],
    "model_family": deployment["model_family"],
    "deployment_id": deployment["deployment_id"],
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "prediction_horizon": deployment["prediction_horizon"],
    "action_dim": deployment["action_dim"],
    "execute_steps": args.execute_steps,
    "controller": PI05_CONTROLLER_VERSION if family == "pi05" else None,
    "action_interpolation": (args.action_interpolation if family == "pi05" else None),
    "control_hz": 30,
    "guard_mode": args.guard_mode,
    "diagnostic_only": diagnostic_only,
    "formal_metrics_valid": formal_metrics_valid,
    "diagnostic_reasons": diagnostic_reasons,
  }

  results = []
  started = time.monotonic()
  openpi_root = Path(deployment["model_project_path"]) if family == "pi05" else None
  for trial_index, seed in enumerate(args.seeds):
    manifest_before = frozen_manifest_report(deployment_path, manifest_sha256)
    if manifest_before["status"] != "verified":
      row = {
        "seed": seed,
        "trial_started": False,
        "success": False,
        "valid_trial": False,
        "termination": "deployment_manifest_drift_before_trial",
        "action_clipping": None,
        "deployment_manifest_integrity_before": manifest_before,
        "deployment_manifest_integrity_after": None,
        "evaluation_source_integrity_before": None,
        "evaluation_source_integrity_after": None,
      }
      results.append(row)
      _write_batch_summary(
        output,
        identity,
        len(args.seeds),
        results,
        started,
        source_integrity_initial,
      )
      print(f"STOP seed={seed} deployment manifest drift before trial", flush=True)
      break
    source_before = (
      verify_evaluation_sources(deployment, ROOT)
      if source_integrity_enforced
      else source_integrity_initial
    )
    if source_integrity_enforced and source_before["status"] != "verified":
      row = {
        "seed": seed,
        "trial_started": False,
        "success": False,
        "valid_trial": False,
        "termination": "evaluation_source_drift_before_trial",
        "action_clipping": None,
        "deployment_manifest_integrity_before": manifest_before,
        "deployment_manifest_integrity_after": None,
        "evaluation_source_integrity_before": source_before,
        "evaluation_source_integrity_after": None,
      }
      results.append(row)
      _write_batch_summary(
        output,
        identity,
        len(args.seeds),
        results,
        started,
        source_integrity_initial,
      )
      print(f"STOP seed={seed} evaluation source drift before trial", flush=True)
      break
    trial = output / f"seed_{seed:03d}"
    command = [
      sys.executable,
      str(RUNNERS[family]),
      "--task",
      deployment["task"],
      "--deployment-manifest",
      str(deployment_path),
      "--output-dir",
      str(trial),
      "--seed",
      str(seed),
      "--execute-steps",
      str(args.execute_steps),
      "--max-sim-seconds",
      str(args.max_sim_seconds),
      "--guard-mode",
      args.guard_mode,
      "--record-fps",
      str(args.record_fps),
      "--save-first-request" if trial_index == 0 else "--no-save-first-request",
      "--record" if trial_index < args.video_count else "--no-record",
    ]
    if family == "pi05":
      command.extend(
        (
          "--expected-deployment-manifest-sha256",
          manifest_sha256,
          "--action-interpolation",
          args.action_interpolation,
        )
      )
    if whiteboard_contact:
      command.extend(("--contact-demo-root", str(args.contact_demo_root)))
      if args.contact_reference_hdf5 is not None:
        command.extend(
          ("--contact-reference-hdf5", str(args.contact_reference_hdf5))
        )
    if args.reference_dataset is not None:
      command.extend(
        (
          "--reference-dataset",
          str(args.reference_dataset),
          "--reference-episode-index",
          str(args.reference_episode_index),
        )
      )
    if family in {"egosteer", "pi05"}:
      command.extend(("--server", args.server))
    else:
      command.extend(
        (
          "--model-python",
          args.model_python,
          "--model-project",
          str(args.model_project),
        )
      )
      if args.snapshot is not None:
        command.extend(("--snapshot", str(args.snapshot)))
    if args.diagnostic_relax_whiteboard_contact_guards:
      command.append("--diagnostic-relax-whiteboard-contact-guards")
    command.extend(passthrough)

    print(f"START {trial_index + 1}/{len(args.seeds)} seed={seed}", flush=True)
    trial_started = time.monotonic()
    with (output / f"seed_{seed:03d}.log").open("x", encoding="utf-8") as log:
      try:
        completed = subprocess.run(
          command,
          cwd=ROOT,
          env=_environment(openpi_root),
          stdout=log,
          stderr=subprocess.STDOUT,
          timeout=args.trial_wall_limit,
        )
        returncode = completed.returncode
      except subprocess.TimeoutExpired:
        returncode = -999
    summary_path = trial / "summary.json"
    report = (
      json.loads(summary_path.read_text(encoding="utf-8"))
      if summary_path.is_file()
      else {"status": "error", "error": "trial produced no summary.json"}
    )
    source_after = (
      verify_evaluation_sources(deployment, ROOT)
      if source_integrity_enforced
      else source_integrity_initial
    )
    manifest_after = frozen_manifest_report(deployment_path, manifest_sha256)
    identity_errors = {
      key: {"expected": value, "actual": report.get(key)}
      for key, value in identity.items()
      if key != "formal_metrics_valid"
      if report.get(key) != value
    }
    reason, valid = classify(report)
    if identity_errors:
      reason, valid = "trial_identity_mismatch", False
    if report.get("record_error"):
      reason, valid = "record_error", False
    child_integrity = report.get("deployment_integrity")
    if family == "pi05" and (
      not isinstance(child_integrity, dict)
      or child_integrity.get("status") != "verified"
      or child_integrity.get("manifest", {}).get("actual_sha256") != manifest_sha256
      or child_integrity.get("deployment_id", {}).get("recomputed")
      != deployment["deployment_id"]
    ):
      reason, valid = "trial_deployment_integrity_mismatch", False
    if valid and trial_index < args.video_count:
      artifact_error = validate_recorded_review(
        trial,
        deployment,
        args.reference_dataset,
        args.reference_episode_index,
        bool(report.get("diagnostic_only", diagnostic_only)),
      )
      if artifact_error is not None:
        reason, valid = artifact_error, False
    if source_integrity_enforced and source_after["status"] != "verified":
      reason, valid = "evaluation_source_drift_during_trial", False
    if manifest_after["status"] != "verified":
      reason, valid = "deployment_manifest_drift_during_trial", False
    row = {
      "seed": seed,
      "trial_started": True,
      "success": bool(report.get("task_success", reason == "success")),
      "task_success": bool(report.get("task_success", reason == "success")),
      "valid_trial": valid,
      "termination": reason,
      "stop_reason": report.get("stop_reason"),
      "guard_mode": report.get("guard_mode", args.guard_mode),
      "guard_events": report.get("guard_events", []),
      "physics_valid": report.get("physics_valid"),
      "returncode": returncode,
      "wall_seconds": time.monotonic() - trial_started,
      "sim_seconds": report.get("sim_seconds"),
      "error": report.get("error"),
      "record_error": report.get("record_error"),
      "identity_errors": identity_errors,
      "deployment_integrity": child_integrity,
      "requests": report.get("stats", {}).get("requests", 0),
      "action_steps": report.get("stats", {}).get("action_steps", 0),
      "action_clipping": report.get("stats", {}).get("action_clipping"),
      "evaluation": report.get("evaluation", {}),
      "diagnostic_only": report.get("diagnostic_only", diagnostic_only),
      "formal_metrics_valid": report.get(
        "formal_metrics_valid", formal_metrics_valid
      ),
      "diagnostic_reasons": report.get(
        "diagnostic_reasons", diagnostic_reasons
      ),
      "summary": (
        str(summary_path.relative_to(output)) if summary_path.exists() else None
      ),
      "deployment_manifest_integrity_before": manifest_before,
      "deployment_manifest_integrity_after": manifest_after,
      "evaluation_source_integrity_before": source_before,
      "evaluation_source_integrity_after": source_after,
    }
    if whiteboard_contact:
      row["contact_intensity"] = report.get("contact_intensity")
    results.append(row)
    _write_batch_summary(
      output,
      identity,
      len(args.seeds),
      results,
      started,
      source_integrity_initial,
    )
    print(
      f"DONE seed={seed} termination={reason} valid={valid} success={row['success']}",
      flush=True,
    )
    if (
      source_integrity_enforced and source_after["status"] != "verified"
    ) or manifest_after["status"] != "verified":
      break
  return 0 if all(item["valid_trial"] for item in results) else 2


if __name__ == "__main__":
  raise SystemExit(main())
