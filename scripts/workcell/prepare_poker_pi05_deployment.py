#!/usr/bin/env python3
"""Create an immutable inference identity for a KaiHand card pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  CONVERSION_SNAPSHOT_FILENAME,
  FORMAL_ACTION_INTERPOLATION,
  LEGACY_ACTION_CONTRACT,
  POKER_DEPLOYMENT_SCHEMA_V1,
  POKER_DEPLOYMENT_SCHEMA_V2,
  POKER_DEPLOYMENT_V1_ID_FIELDS,
  POKER_DEPLOYMENT_V2_ID_FIELDS,
  TRAINING_PROVENANCE_FILENAME,
  action_representation_from_provenance,
  load_checkpoint_training_provenance,
  openpi_runtime_source_paths,
  poker_deployment_identity,
  stable_file_bytes,
  validate_poker_deployment_semantics,
)

try:
  from prepare_usb_pi05_deployment import (
    _canonical_hash,
    checkpoint_identity,
    normalizer_asset_id,
    sha256_file,
  )
except ModuleNotFoundError as error:
  if error.name != "prepare_usb_pi05_deployment":
    raise
  from scripts.workcell.prepare_usb_pi05_deployment import (
    _canonical_hash,
    checkpoint_identity,
    normalizer_asset_id,
    sha256_file,
  )

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "pi05_kaihand"
CONFIG_HORIZONS = {CONFIG_NAME: 30, "pi05_kaihand_h50": 50}
SCHEMA_V1 = POKER_DEPLOYMENT_SCHEMA_V1
SCHEMA_V2 = POKER_DEPLOYMENT_SCHEMA_V2
_IDENTITY_FIELDS = POKER_DEPLOYMENT_V1_ID_FIELDS
_V2_IDENTITY_FIELDS = POKER_DEPLOYMENT_V2_ID_FIELDS
EXPECTED_TASK = (
  "Slide the face-down card toward the table edge, pinch it between the fingers "
  "and thumb, lift it, and turn its face toward the robot to look at it."
)


def checkpoint_contract(checkpoint: Path) -> tuple[str, str, dict | None]:
  """Choose the checkpoint-bound config/normalizer, retaining historical H30."""
  assets = checkpoint / "assets"
  presence = (
    (assets / TRAINING_PROVENANCE_FILENAME).is_file(),
    (assets / CONVERSION_SNAPSHOT_FILENAME).is_file(),
  )
  if not any(presence):
    return CONFIG_NAME, normalizer_asset_id(checkpoint), None
  if not all(presence):
    raise RuntimeError("incomplete Poker checkpoint training provenance")
  binding = load_checkpoint_training_provenance(checkpoint)
  record = binding["record"]
  if record.get("task") != "poker-draw":
    raise RuntimeError("Poker checkpoint training provenance task must be poker-draw")
  config_name = record.get("config_name")
  if config_name not in CONFIG_HORIZONS:
    raise RuntimeError(f"unsupported Poker checkpoint config: {config_name!r}")
  if record["action_contract"].get("id") != LEGACY_ACTION_CONTRACT:
    raise RuntimeError("Poker checkpoint must retain legacy measured-arm semantics")
  snapshot_bytes, _ = stable_file_bytes(assets / CONVERSION_SNAPSHOT_FILENAME)
  snapshot = json.loads(snapshot_bytes)
  prompt_key = (
    "instruction"
    if record["conversion_manifest"]["filename"] == "kaihand_fast_pi05_conversion.json"
    else "task_instruction"
  )
  if snapshot.get(prompt_key) != EXPECTED_TASK:
    raise RuntimeError("Poker checkpoint conversion instruction differs from task")
  asset_id = normalizer_asset_id(checkpoint, record["normalizer"].get("asset_id"))
  return config_name, asset_id, binding


def deployment_identity(manifest: dict) -> dict:
  return poker_deployment_identity(manifest)


def validate_deployment_identity(manifest: dict) -> None:
  schema = manifest.get("schema")
  if schema not in {SCHEMA_V1, SCHEMA_V2}:
    raise RuntimeError("expected poker_pi05_deployment_v1 or poker_pi05_deployment_v2")
  if manifest.get("task") != "poker-draw" or manifest.get("model_family") != "pi0.5":
    raise RuntimeError("manifest is not a poker-draw pi0.5 deployment")
  config_name = manifest.get("model_config_name")
  if schema == SCHEMA_V1:
    if config_name != CONFIG_NAME or manifest.get("prediction_horizon") != 30:
      raise RuntimeError("historical Poker manifest must use pi05_kaihand H30")
    return
  if config_name not in CONFIG_HORIZONS:
    raise RuntimeError(f"unsupported Poker deployment config: {config_name!r}")
  if manifest.get("prediction_horizon") != CONFIG_HORIZONS[config_name]:
    raise RuntimeError("Poker deployment config and horizon disagree")
  if manifest.get("deployment_id") != _canonical_hash(deployment_identity(manifest)):
    raise RuntimeError("deployment_id does not authenticate Poker deployment identity")
  if manifest.get("instruction") != EXPECTED_TASK:
    raise RuntimeError("Poker deployment instruction differs from the task prompt")
  validate_poker_deployment_semantics(manifest, verify_checkpoint_assets=False)


def validate_manifest_training_binding(manifest: dict) -> dict | None:
  validate_deployment_identity(manifest)
  observed = checkpoint_contract(Path(manifest["checkpoint_path"]))
  if observed[:2] != (
    manifest["model_config_name"],
    manifest["normalizer_asset_id"],
  ):
    raise RuntimeError("Poker manifest config/asset differs from checkpoint assets")
  if manifest["schema"] == SCHEMA_V2 and manifest["training_provenance"] != observed[2]:
    raise RuntimeError(
      "Poker manifest training provenance differs from checkpoint assets"
    )
  return observed[2]


def evaluation_source_paths() -> list[Path]:
  entry_points = (
    "scripts/workcell/prepare_poker_pi05_deployment.py",
    "scripts/workcell/prepare_usb_pi05_deployment.py",
    "scripts/workcell/serve_poker_pi05_policy.py",
    "scripts/workcell/serve_shared_task_pi05_policy.py",
    "scripts/workcell/evaluate_poker_pi05_policy_batch.py",
    "scripts/workcell/run_poker_pi05_policy.py",
    "scripts/workcell/run_usb_pi05_policy.py",
    "scripts/workcell/run_shared_task_pi05_policy.py",
    "scripts/workcell/poker_pi05_rollout_artifacts.py",
    "scripts/collect/nvidia_egl_vendor.json",
  )
  package = ROOT / "src/kaihand_tactile_env"
  paths = sorted(
    [
      *(ROOT / relative for relative in entry_points),
      *(
        path
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.casefold() != ".pyc"
      ),
    ],
    key=lambda path: path.relative_to(ROOT).as_posix(),
  )
  if len(paths) != len(set(paths)) or not paths:
    raise RuntimeError("Poker evaluation source closure is empty or duplicated")
  for path in paths:
    if not path.is_file():
      raise FileNotFoundError(path)
  return paths


def validate_train_config(train_config, config_name: str, binding: dict | None) -> dict:
  if train_config.name != config_name:
    raise RuntimeError("OpenPI runtime config name differs from checkpoint provenance")
  metadata = dict(train_config.policy_metadata or {})
  required_metadata = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": CONFIG_HORIZONS[config_name],
    "control_hz": 30,
    "suggested_replan_steps": 8,
    "image_keys": ["observation.images.head", "observation.images.right_wrist"],
  }
  mismatches = {
    key: {"expected": expected, "actual": metadata.get(key)}
    for key, expected in required_metadata.items()
    if metadata.get(key) != expected
  }
  if mismatches:
    raise RuntimeError(f"OpenPI card config contract mismatch: {mismatches}")
  if int(train_config.model.action_horizon) != metadata["action_horizon"]:
    raise RuntimeError("model horizon and policy metadata disagree")
  if int(train_config.model.action_dim) != metadata["model_action_dim"]:
    raise RuntimeError("model action dimension and policy metadata disagree")
  if train_config.data.__class__.__name__ != "LeRobotKaiHandDataConfig":
    raise RuntimeError(
      f"expected LeRobotKaiHandDataConfig, got {train_config.data.__class__.__name__}"
    )
  if train_config.data.use_delta_arm_actions is not True:
    raise RuntimeError("Poker pi0.5 inference requires delta arm action transforms")
  joints = metadata.get("joint_names")
  if (
    not isinstance(joints, list)
    or len(joints) != 27
    or len(set(joints)) != 27
    or not all(isinstance(name, str) and name for name in joints)
  ):
    raise RuntimeError("OpenPI Poker config must declare 27 unique joints")
  if binding is not None:
    record = binding["record"]
    if joints != record["action_contract"]["joint_order"]:
      raise RuntimeError("OpenPI joint order differs from checkpoint provenance")
    if metadata.get("action_semantics") != action_representation_from_provenance(
      record
    ):
      raise RuntimeError("OpenPI action semantics differ from checkpoint provenance")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.use_quantile_norm is not record["normalizer"]["use_quantile_norm"]:
      raise RuntimeError("OpenPI normalization mode differs from checkpoint provenance")
  return metadata


def _source_rows(paths: list[Path], base: Path) -> list[dict]:
  rows = []
  for path in paths:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
      after.st_dev,
      after.st_ino,
      after.st_size,
      after.st_mtime_ns,
    ):
      raise RuntimeError(f"source changed while hashing: {path}")
    rows.append(
      {
        "path": path.relative_to(base).as_posix(),
        "size": after.st_size,
        "sha256": digest,
      }
    )
  return rows


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--openpi-root",
    type=Path,
    default=Path("/cpfs_infra/user/chenxianchi/code/openpi"),
  )
  parser.add_argument("--hash-workers", type=int, default=4)
  parser.add_argument(
    "--dry-run", action="store_true", help="Validate without hashing or writing output"
  )
  args = parser.parse_args(argv)
  if args.hash_workers <= 0:
    parser.error("hash-workers must be positive")

  from openpi.training import config as openpi_config

  checkpoint = args.checkpoint.expanduser().resolve()
  openpi_root = args.openpi_root.expanduser().resolve()
  output = args.output_dir.expanduser().resolve()
  if not checkpoint.is_dir():
    raise FileNotFoundError(checkpoint)
  try:
    step = int(checkpoint.name)
  except ValueError as error:
    raise ValueError("checkpoint directory name must be an integer step") from error
  if output.exists() and not args.dry_run:
    raise FileExistsError(f"refusing to overwrite deployment directory: {output}")

  config_name, asset_id, training_provenance = checkpoint_contract(checkpoint)
  train_config = openpi_config.get_config(config_name)
  metadata = validate_train_config(train_config, config_name, training_provenance)
  if (
    not (checkpoint / "_CHECKPOINT_METADATA").is_file()
    or not (checkpoint / "params").is_dir()
  ):
    raise RuntimeError("Poker checkpoint is not a committed inference payload")
  if args.dry_run:
    print(
      json.dumps(
        {
          "valid": True,
          "task": "poker-draw",
          "checkpoint": str(checkpoint),
          "checkpoint_step": step,
          "model_config_name": config_name,
          "prediction_horizon": metadata["action_horizon"],
          "normalizer_asset_id": asset_id,
          "training_provenance": training_provenance,
          "required_action_interpolation": FORMAL_ACTION_INTERPOLATION,
          "diagnostic_only": True,
          "formal_metrics_valid": False,
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
  if checkpoint_contract(checkpoint) != (config_name, asset_id, training_provenance):
    raise RuntimeError("Poker checkpoint training assets changed while hashing")
  source_paths = (
    openpi_runtime_source_paths(openpi_root)
    if training_provenance is not None
    else [
      openpi_root / "src/openpi/training/config.py",
      openpi_root / "src/openpi/policies/kaihand_policy.py",
      openpi_root / "src/openpi/policies/policy_config.py",
      openpi_root / "src/openpi/serving/websocket_policy_server.py",
    ]
  )
  source_rows = _source_rows(source_paths, openpi_root)
  evaluation_rows = (
    _source_rows(evaluation_source_paths(), ROOT)
    if training_provenance is not None
    else []
  )
  identity = {
    "task": "poker-draw",
    "model_family": "pi0.5",
    "model_config_name": config_name,
    "normalizer_asset_id": asset_id,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_step": step,
    "prediction_horizon": metadata["action_horizon"],
    "model_action_dim": metadata["model_action_dim"],
    "action_dim": metadata["action_dim"],
    "joint_names": metadata["joint_names"],
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "image_dtype": "uint8",
      "state": "27 measured right-side joint positions in joint_names order",
      "language_instruction": True,
      "tactile_sent_to_model": False,
    },
    "action_representation": metadata["action_semantics"],
  }
  if training_provenance is not None:
    identity.update(
      {
        "training_provenance": training_provenance,
        "control_hz": metadata["control_hz"],
        "suggested_execute_steps": metadata["suggested_replan_steps"],
        "instruction": EXPECTED_TASK,
        "model_project_path": str(openpi_root),
        "openpi_source_sha256": _canonical_hash(source_rows),
        "evaluation_source_sha256": _canonical_hash(evaluation_rows),
        "required_action_interpolation": FORMAL_ACTION_INTERPOLATION,
        # Dataset mechanics were not recorded. Training binding alone cannot
        # authenticate a formal comparison with this simulator revision.
        "diagnostic_only": True,
        "formal_metrics_valid": False,
      }
    )
  deployment_id = _canonical_hash(identity)
  manifest = {
    "schema": SCHEMA_V2 if training_provenance is not None else SCHEMA_V1,
    "deployment_id": deployment_id,
    **identity,
    "checkpoint_hash_algorithm": (
      "sha256(canonical JSON of path,size,file_sha256) over committed "
      "_CHECKPOINT_METADATA, params/** and assets/**; train_state excluded"
    ),
    "checkpoint_files": checkpoint_files,
    "checkpoint_total_inference_bytes": sum(row["size"] for row in checkpoint_files),
    "checkpoint_frozen_by_content_hash": True,
    "control_hz": metadata["control_hz"],
    "suggested_execute_steps": metadata["suggested_replan_steps"],
    "instruction": EXPECTED_TASK,
    "model_project_path": str(openpi_root),
    "openpi_source_sha256": _canonical_hash(source_rows),
    "openpi_source_files": source_rows,
  }
  if training_provenance is not None:
    manifest["evaluation_source_files"] = evaluation_rows
    manifest["evaluation_project_path"] = str(ROOT)
    validate_manifest_training_binding(manifest)
  output.mkdir(parents=True)
  path = output / "deployment_manifest.json"
  path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"deployment_manifest={path}")
  print(f"deployment_id={deployment_id}")
  print(f"checkpoint_sha256={checkpoint_sha256}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
