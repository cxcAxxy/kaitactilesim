#!/usr/bin/env python3
"""Freeze a shared-task KaiHand pi0.5 checkpoint for model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  BULB_SCREW_CONTROLLER_TASKS,
  CONVERSION_SNAPSHOT_FILENAME,
  FORMAL_ACTION_INTERPOLATION,
  LEGACY_ACTION_CONTRACT,
  LEGACY_KAIHAND_CONFIG,
  SHARED_DEPLOYMENT_SCHEMA_V2,
  SHARED_DEPLOYMENT_V2_ID_FIELDS,
  SHARED_KAIHAND_ACTION_HORIZONS,
  TRAINING_PROVENANCE_FILENAME,
  USB_UNPLUG_LEGACY_INSTRUCTION,
  action_representation_from_provenance,
  bulb_controller_target_configs,
  load_checkpoint_training_provenance,
  openpi_runtime_source_paths,
  unbound_training_provenance,
  validate_shared_deployment_semantics,
)
from kaihand_tactile_env.shared.policy_tasks import (
  TASK_INSTRUCTIONS as SHARED_INSTRUCTIONS,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = SHARED_DEPLOYMENT_SCHEMA_V2
NORMALIZER_ASSET_ID = "normalizer"
TASK_INSTRUCTIONS = {
  **{name: SHARED_INSTRUCTIONS[name] for name in (
    "usb-unplug", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
    "bulb-unscrew", "bulb-screw-inplace",
  )},
  "bulb-screw": (
    "Pick up the light bulb with the right hand, align it with the socket, "
    "screw it clockwise until mechanically seated, then release it."
  ),
  "install-ram": (
    "Pick up the RAM module with the right hand, align its keyed edge with "
    "the socket, press it straight down until seated, then release it."
  ),
  "vase-wipe": (
    "Pick up the sponge with the right hand and wipe the marked dirt from "
    "the inside wall of the vase until it is clean."
  ),
  "whiteboard-wipe": (
    "Pick up the eraser with the right hand, wipe all ink from the tilted "
    "whiteboard with loaded sliding contact, then return and release the eraser."
  ),
  "sponge-grasp": (
    "Pick up the upright sponge with the right hand, carry it to the plate on "
    "the robot's right, place it inside the plate, and release it."
  ),
}
TASK_PACKAGE_DIRS = {
  "usb-unplug": ("usb_unplug", "usb_insert"),
  "whiteboard-wipe-horizontal": ("whiteboard_wipe_horizontal", "whiteboard_wipe"),
  "whiteboard-wipe-horizontal-line": (
    "whiteboard_wipe_horizontal_line", "whiteboard_wipe_horizontal", "whiteboard_wipe",
  ),
  "bulb-unscrew": ("bulb_unscrew", "bulb_screw"),
  "bulb-screw-inplace": ("bulb_screw_inplace", "bulb_screw"),
  "bulb-screw": ("bulb_screw",),
  "install-ram": ("install_ram",),
  "vase-wipe": ("vase_wipe",),
  "whiteboard-wipe": ("whiteboard_wipe",),
  # SpongeGraspSimulation reuses the deformable sponge implementation.
  "sponge-grasp": ("sponge_grasp", "vase_wipe"),
}
EXPECTED_POLICY_METADATA = {
  "action_dim": 27,
  "model_action_dim": 32,
  "control_hz": 30,
  "suggested_replan_steps": 8,
  "image_keys": [
    "observation.images.head",
    "observation.images.right_wrist",
  ],
}


def evaluation_source_paths(task: str) -> list[Path]:
  """Return the complete local runtime/scene closure for formal evaluation.

  ``shared.config`` imports every task configuration at module import time and
  the selected MJCF recursively loads common robot meshes and textures.  A
  short hand-maintained allowlist therefore is not a dependency closure.  Keep
  the small set of entry-point scripts explicit, but freeze every non-bytecode
  package source/asset so a running batch cannot silently mix controller,
  rendering, tactile, task, or MJCF semantics.
  """
  if task not in TASK_PACKAGE_DIRS:
    raise ValueError(f"unsupported shared policy task: {task!r}")
  entry_points = [
    ROOT / "scripts/workcell/evaluate_shared_task_policy_batch.py",
    ROOT / "scripts/workcell/diagnose_bulb_replay_and_stages.py",
    ROOT / "scripts/workcell/prepare_shared_task_pi05_deployment.py",
    ROOT / "scripts/workcell/replay_episode.py",
    ROOT / "scripts/workcell/run_shared_task_pi05_policy.py",
    ROOT / "scripts/workcell/run_usb_pi05_policy.py",
    ROOT / "scripts/workcell/serve_shared_task_pi05_policy.py",
    ROOT / "scripts/collect/nvidia_egl_vendor.json",
  ]
  package_root = ROOT / "src/kaihand_tactile_env"
  package_sources = [
    path
    for path in package_root.rglob("*")
    if path.is_file()
    and "__pycache__" not in path.parts
    and path.suffix.casefold() != ".pyc"
  ]
  paths = sorted(
    [*entry_points, *package_sources],
    key=lambda path: path.relative_to(ROOT).as_posix(),
  )
  if len(paths) != len(set(paths)):
    raise RuntimeError(f"duplicate evaluation source path for task {task!r}")
  missing = [path for path in paths if not path.is_file()]
  if missing:
    raise FileNotFoundError(f"missing evaluation source files: {missing}")
  return paths


def openpi_source_paths(openpi_root: Path) -> list[Path]:
  """Return the server and websocket-client source closure used at runtime."""
  return openpi_runtime_source_paths(openpi_root)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def canonical_hash(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def normalizer_asset_id(checkpoint: Path, asset_id: str = NORMALIZER_ASSET_ID) -> str:
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise ValueError("checkpoint normalizer asset id must be one directory name")
  norm_stats = checkpoint / "assets" / asset_id / "norm_stats.json"
  if not norm_stats.is_file():
    raise ValueError(f"missing checkpoint normalization stats: {norm_stats}")
  return asset_id


def validate_train_config(config) -> dict:
  metadata = dict(config.policy_metadata or {})
  if config.name not in SHARED_KAIHAND_ACTION_HORIZONS:
    raise RuntimeError(f"unsupported shared KaiHand training config: {config.name!r}")
  expected_metadata = {
    **EXPECTED_POLICY_METADATA,
    "action_horizon": SHARED_KAIHAND_ACTION_HORIZONS[config.name],
  }
  mismatches = {
    key: {"expected": expected, "actual": metadata.get(key)}
    for key, expected in expected_metadata.items()
    if metadata.get(key) != expected
  }
  if mismatches:
    raise RuntimeError(f"OpenPI shared-task config contract mismatch: {mismatches}")
  if int(config.model.action_horizon) != metadata["action_horizon"]:
    raise RuntimeError("model horizon and policy metadata disagree")
  if int(config.model.action_dim) != metadata["model_action_dim"]:
    raise RuntimeError("model action dimension and policy metadata disagree")
  if config.data.__class__.__name__ != "LeRobotKaiHandDataConfig":
    raise RuntimeError(
      f"expected LeRobotKaiHandDataConfig, got {config.data.__class__.__name__}"
    )
  if config.data.use_delta_arm_actions is not True:
    raise RuntimeError(
      "shared-task pi0.5 inference requires delta arm action transforms"
    )
  joint_names = metadata.get("joint_names")
  if (
    not isinstance(joint_names, list)
    or len(joint_names) != 27
    or not all(isinstance(name, str) and name for name in joint_names)
    or len(set(joint_names)) != 27
  ):
    raise RuntimeError("OpenPI shared-task config must declare 27 unique joints")
  action_semantics = metadata.get("action_semantics")
  if (
    not isinstance(action_semantics, dict)
    or action_semantics.get("policy_output_is_absolute") is not True
  ):
    raise RuntimeError(
      "OpenPI shared-task policy output must be absolute joint targets"
    )
  return metadata


def checkpoint_identity(checkpoint: Path, workers: int) -> tuple[str, list[dict]]:
  committed = checkpoint / "_CHECKPOINT_METADATA"
  if not committed.is_file():
    raise ValueError(f"checkpoint is not committed: {committed} is missing")
  paths = [committed]
  for directory in (checkpoint / "params", checkpoint / "assets"):
    if not directory.is_dir():
      raise ValueError(f"missing inference checkpoint directory: {directory}")
    paths.extend(path for path in directory.rglob("*") if path.is_file())
  paths = sorted(set(paths), key=lambda path: path.relative_to(checkpoint).as_posix())
  snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
  with ThreadPoolExecutor(max_workers=workers) as pool:
    hashes = list(pool.map(sha256_file, paths))
  changed = [
    path
    for path in paths
    if (path.stat().st_size, path.stat().st_mtime_ns) != snapshots[path]
  ]
  if changed:
    raise RuntimeError(f"checkpoint changed while hashing: {changed[0]}")
  rows = [
    {
      "path": path.relative_to(checkpoint).as_posix(),
      "size": snapshots[path][0],
      "sha256": digest,
    }
    for path, digest in zip(paths, hashes, strict=True)
  ]
  return canonical_hash(rows), rows


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=tuple(TASK_INSTRUCTIONS), required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--openpi-root",
    type=Path,
    default=Path("/cpfs_infra/user/chenxianchi/code/openpi"),
  )
  parser.add_argument("--hash-workers", type=int, default=4)
  parser.add_argument(
    "--allow-legacy-unbound-diagnostic",
    action="store_true",
    help=(
      "Allow a checkpoint created before training-provenance assets existed. "
      "The resulting deployment is permanently diagnostic-only and retains "
      "the legacy measured-arm action contract."
    ),
  )
  parser.add_argument(
    "--allow-usb-unplug-protocol-transfer-diagnostic",
    action="store_true",
    help=(
      "Explicitly evaluate a bound legacy USB return/release policy against the "
      "current held-clear-2s completion rule. Preserve its original prompt and "
      "training assets; require unchanged scene/physics; mark diagnostic-only."
    ),
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate the checkpoint/config contract without hashing or writing output",
  )
  args = parser.parse_args(argv)
  if args.hash_workers <= 0:
    parser.error("hash-workers must be positive")
  if args.allow_usb_unplug_protocol_transfer_diagnostic:
    if args.task != "usb-unplug":
      parser.error("USB protocol transfer is only valid for --task usb-unplug")
    if args.allow_legacy_unbound_diagnostic:
      parser.error("USB protocol transfer requires bound assets and cannot use --allow-legacy-unbound-diagnostic")
  return args


def main(argv=None) -> int:
  args = parse_args(argv)

  # Run with the OpenPI environment so the contract comes from the exact
  # inference config used to restore the checkpoint.
  from openpi.training import config as openpi_config

  checkpoint = args.checkpoint.expanduser().resolve(strict=True)
  openpi_root = args.openpi_root.expanduser().resolve(strict=True)
  output = args.output_dir.expanduser().resolve()
  if not checkpoint.is_dir():
    raise NotADirectoryError(checkpoint)
  try:
    step = int(checkpoint.name)
  except ValueError as error:
    raise ValueError("checkpoint directory name must be an integer step") from error
  if output.exists() and not args.dry_run:
    raise FileExistsError(f"refusing to overwrite deployment directory: {output}")

  provenance_asset = checkpoint / "assets" / TRAINING_PROVENANCE_FILENAME
  conversion_snapshot = checkpoint / "assets" / CONVERSION_SNAPSHOT_FILENAME
  provenance_assets_present = (
    provenance_asset.is_file(),
    conversion_snapshot.is_file(),
  )
  if all(provenance_assets_present):
    if args.allow_usb_unplug_protocol_transfer_diagnostic:
      training_provenance = load_checkpoint_training_provenance(
        checkpoint, allow_usb_unplug_protocol_transfer_diagnostic=True,
      )
    else:
      training_provenance = load_checkpoint_training_provenance(checkpoint)
    record = training_provenance["record"]
    if record.get("task") != args.task:
      raise RuntimeError(
        "requested deployment task differs from checkpoint training provenance"
      )
    config_name = record["config_name"]
    action_representation = action_representation_from_provenance(record)
    diagnostic_only = "completion_protocol_transfer" in training_provenance
    formal_metrics_valid = not diagnostic_only
  elif any(provenance_assets_present):
    raise RuntimeError(
      "checkpoint training provenance is incomplete; both provenance assets "
      "must be present or both absent"
    )
  elif args.allow_legacy_unbound_diagnostic and args.task != "usb-unplug":
    training_provenance = unbound_training_provenance(
      "legacy_checkpoint_missing_bound_training_provenance_assets"
    )
    config_name = LEGACY_KAIHAND_CONFIG
    action_representation = None
    diagnostic_only = True
    formal_metrics_valid = False
  else:
    raise RuntimeError(
      "checkpoint has no bound training provenance; pass "
      "--allow-legacy-unbound-diagnostic only for explicitly diagnostic rollouts"
    )

  instruction = (
    USB_UNPLUG_LEGACY_INSTRUCTION
    if "completion_protocol_transfer" in training_provenance
    else TASK_INSTRUCTIONS[args.task]
  )

  config = openpi_config.get_config(config_name)
  metadata = validate_train_config(config)
  if action_representation is None:
    action_representation = metadata["action_semantics"]
    if action_representation.get("contract_id") != LEGACY_ACTION_CONTRACT:
      raise RuntimeError(
        "unbound diagnostic checkpoint must retain the legacy action contract"
      )
  elif metadata["action_semantics"] != action_representation:
    raise RuntimeError(
      "current OpenPI config action semantics differ from checkpoint provenance"
    )
  if (
    args.task in BULB_SCREW_CONTROLLER_TASKS
    and not diagnostic_only
    and config_name not in bulb_controller_target_configs(args.task)
  ):
    raise RuntimeError(
      f"formal {args.task} deployment requires the dedicated controller-target config"
    )
  asset_id = NORMALIZER_ASSET_ID
  if training_provenance.get("status") == "bound":
    bound_normalizer = training_provenance["record"].get("normalizer")
    if not isinstance(bound_normalizer, dict):
      raise RuntimeError("checkpoint training provenance must bind its normalizer")
    asset_id = bound_normalizer.get("asset_id")
  asset_id = normalizer_asset_id(checkpoint, asset_id)
  committed = checkpoint / "_CHECKPOINT_METADATA"
  if not committed.is_file() or not (checkpoint / "params").is_dir():
    raise ValueError(f"checkpoint is not a committed inference payload: {checkpoint}")
  if args.dry_run:
    print(
      json.dumps(
        {
          "valid": True,
          "task": args.task,
          "checkpoint": str(checkpoint),
          "checkpoint_step": step,
          "model_config_name": config_name,
          "prediction_horizon": metadata["action_horizon"],
          "normalizer_asset_id": asset_id,
          "cameras": ["head", "right_wrist"],
          "training_provenance": training_provenance,
          "required_action_interpolation": FORMAL_ACTION_INTERPOLATION,
          "diagnostic_only": diagnostic_only,
          "formal_metrics_valid": formal_metrics_valid,
          "instruction": instruction,
        },
        indent=2,
        ensure_ascii=False,
      )
    )
    return 0

  print(f"Hashing committed inference payload under {checkpoint} ...", flush=True)
  checkpoint_sha256, checkpoint_files = checkpoint_identity(
    checkpoint, args.hash_workers
  )
  openpi_sources = openpi_source_paths(openpi_root)
  evaluation_sources = evaluation_source_paths(args.task)

  def source_rows(paths: list[Path], base: Path) -> list[dict]:
    rows = []
    for path in paths:
      before = path.stat()
      digest = sha256_file(path)
      after = path.stat()
      identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
      )
      identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
      )
      if identity_before != identity_after:
        raise RuntimeError(f"source changed while hashing: {path}")
      rows.append(
        {
          "path": path.relative_to(base).as_posix(),
          "size": after.st_size,
          "sha256": digest,
        }
      )
    return rows

  openpi_rows = source_rows(openpi_sources, openpi_root)
  evaluation_rows = source_rows(evaluation_sources, ROOT)
  identity = {
    "task": args.task,
    "model_family": "pi0.5",
    "model_config_name": config_name,
    "normalizer_asset_id": asset_id,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_step": step,
    "prediction_horizon": metadata["action_horizon"],
    "model_action_dim": metadata["model_action_dim"],
    "action_dim": metadata["action_dim"],
    "control_hz": metadata["control_hz"],
    "suggested_execute_steps": metadata["suggested_replan_steps"],
    "joint_names": metadata["joint_names"],
    "instruction": instruction,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "image_dtype": "uint8",
      "state": "27 measured right-side joint positions in joint_names order",
      "language_instruction": True,
      "tactile_sent_to_model": False,
    },
    "action_representation": action_representation,
    "model_project_path": str(openpi_root),
    "openpi_source_sha256": canonical_hash(openpi_rows),
    "evaluation_source_sha256": canonical_hash(evaluation_rows),
    "training_provenance": training_provenance,
    "required_action_interpolation": FORMAL_ACTION_INTERPOLATION,
    "diagnostic_only": diagnostic_only,
    "formal_metrics_valid": formal_metrics_valid,
  }
  if tuple(identity) != SHARED_DEPLOYMENT_V2_ID_FIELDS:
    raise RuntimeError("shared deployment identity fields differ from verifier")
  manifest = {
    "schema": SCHEMA,
    "deployment_id": canonical_hash(identity),
    **identity,
    "checkpoint_hash_algorithm": (
      "sha256(canonical JSON of path,size,file_sha256) over committed "
      "_CHECKPOINT_METADATA, params/** and assets/**; train_state excluded"
    ),
    "checkpoint_files": checkpoint_files,
    "checkpoint_total_inference_bytes": sum(row["size"] for row in checkpoint_files),
    "checkpoint_frozen_by_content_hash": True,
    "openpi_source_files": openpi_rows,
    "evaluation_source_files": evaluation_rows,
  }
  validate_shared_deployment_semantics(manifest)
  output.mkdir(parents=True)
  destination = output / "deployment_manifest.json"
  destination.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"deployment_manifest={destination}")
  print(f"deployment_id={manifest['deployment_id']}")
  print(f"checkpoint_sha256={checkpoint_sha256}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
