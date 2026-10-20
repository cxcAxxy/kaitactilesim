#!/usr/bin/env python3
"""Create an immutable inference identity for a KaiHand USB pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONFIG_NAME = "pi05_kaihand"
NORMALIZER_ASSET_ID = "normalizer"
CONFIG_HORIZONS = {CONFIG_NAME: 30, "pi05_kaihand_h50": 50}
SCHEMA_V1 = "usb_pi05_deployment_v1"
SCHEMA_V2 = "usb_pi05_deployment_v2"
TRAINING_PROVENANCE_FILENAME = "training_provenance.json"
CONVERSION_SNAPSHOT_FILENAME = "conversion_manifest.snapshot.json"
USB_MIGRATION_FILENAME = "kaihand_usb_migration_manifest.json"
USB_MIGRATION_SCHEMA = "kaihand_usb_lerobot_v2_1_interpolation_migration_v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = (
  "task",
  "model_family",
  "model_config_name",
  "normalizer_asset_id",
  "checkpoint_path",
  "checkpoint_sha256",
  "checkpoint_step",
  "prediction_horizon",
  "model_action_dim",
  "action_dim",
  "joint_names",
  "observation_contract",
  "action_representation",
)
_V2_IDENTITY_FIELDS = (
  *_IDENTITY_FIELDS,
  "training_provenance",
  "control_hz",
  "suggested_execute_steps",
  "instruction",
  "model_project_path",
  "openpi_source_sha256",
  "evaluation_source_sha256",
  "required_action_interpolation",
)
EXPECTED_TASK = (
  "Grasp the USB plug with the right hand, align it with the socket, "
  "insert it until seated, then release it and withdraw the hand."
)
ROOT = Path(__file__).resolve().parents[2]


def evaluation_source_paths() -> list[Path]:
  """Freeze the simulator, scene assets and USB evaluation entry points."""
  entries = [
    ROOT / "scripts/workcell" / name
    for name in (
      "prepare_usb_pi05_deployment.py",
      "serve_usb_pi05_policy.py",
      "run_usb_pi05_policy.py",
      "evaluate_usb_pi05_policy_batch.py",
    )
  ]
  entries.append(ROOT / "scripts/collect/nvidia_egl_vendor.json")
  entries.extend(
    (ROOT / "scripts/evaluate/evaluate.py", ROOT / "scripts/deploy/deploy.py")
  )
  entries.extend(
    path
    for path in (ROOT / "src/kaihand_tactile_env").rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
  )
  return sorted(entries, key=lambda path: path.relative_to(ROOT).as_posix())


def openpi_source_paths(root: Path) -> list[Path]:
  """Include model and websocket client closures, not just a config allowlist."""
  entries = [
    path
    for directory in (
      root / "src/openpi",
      root / "packages/openpi-client/src/openpi_client",
    )
    for path in directory.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
  ]
  return sorted(entries, key=lambda path: path.relative_to(root).as_posix())


def source_rows(paths: list[Path], root: Path) -> list[dict]:
  return [
    {
      "path": path.relative_to(root).as_posix(),
      "size": path.stat().st_size,
      "sha256": sha256_file(path),
    }
    for path in paths
  ]


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


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_hash(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def stable_file_bytes(path: Path) -> tuple[bytes, str]:
  # Keep legacy PickPlace/Poker imports of the hashing helpers dependency-free.
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    stable_file_bytes as read,
  )

  return read(path)


def _normalizer_semantic_sha256(path: Path) -> str:
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    _normalizer_semantic_sha256 as semantic_hash,
  )

  return semantic_hash(path)


def action_representation_from_provenance(record: dict) -> dict:
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    action_representation_from_provenance as representation,
  )

  return representation(record)


def _object(payload: bytes, label: str) -> dict:
  value = json.loads(payload)
  if not isinstance(value, dict):
    raise RuntimeError(f"{label} must contain one JSON object")
  return value


def _artifact_identity(value: object, label: str) -> dict:
  if (
    not isinstance(value, dict)
    or type(value.get("size_bytes")) is not int
    or value["size_bytes"] < 0
    or not isinstance(value.get("sha256"), str)
    or _SHA256.fullmatch(value["sha256"]) is None
  ):
    raise RuntimeError(f"invalid {label} identity")
  return {"size_bytes": value["size_bytes"], "sha256": value["sha256"]}


def _usb_snapshot_artifacts(snapshot: dict, dataset: dict) -> dict:
  """Recompute the recorded dataset identity without reading source data."""
  root = dataset.get("root")
  lineage = snapshot.get("lineage")
  if (
    not isinstance(root, str)
    or not Path(root).is_absolute()
    or snapshot.get("output_dir") != root
    or not isinstance(lineage, dict)
    or not isinstance(lineage.get("source_dataset"), str)
    or not Path(lineage["source_dataset"]).is_absolute()
    or lineage["source_dataset"] == root
  ):
    raise RuntimeError("USB conversion snapshot dataset lineage is invalid")
  source_root = Path(lineage["source_dataset"])
  names = {
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/info.json",
    "meta/tasks.jsonl",
    "meta/kaihand_usb_source_snapshot.json",
    "meta/kaihand_usb_conversion_manifest.json",
    "meta/interpolation_manifest.json",
  }
  metadata = snapshot.get("metadata_artifacts")
  if not isinstance(metadata, dict) or set(metadata) != names:
    raise RuntimeError("USB conversion snapshot metadata inventory differs")
  rows = [
    {"path": str(Path(root) / name), **_artifact_identity(metadata[name], name)}
    for name in sorted(names)
  ]
  rows.append(
    {
      "path": str(source_root / "meta/kaihand_usb_conversion_manifest.json"),
      **_artifact_identity(
        lineage.get("source_conversion_manifest"), "USB source manifest"
      ),
    }
  )
  episodes = snapshot.get("episodes")
  total = snapshot.get("total_episodes")
  if (
    not isinstance(episodes, list)
    or type(total) is not int
    or total <= 0
    or len(episodes) != total
  ):
    raise RuntimeError("USB conversion snapshot episode inventory differs")
  for index, episode in enumerate(episodes):
    relative = f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
    if (
      not isinstance(episode, dict)
      or type(episode.get("output_episode_index")) is not int
      or episode.get("output_episode_index") != index
      or episode.get("parquet_path") != relative
      or not isinstance(episode.get("source_parquet"), dict)
      or episode["source_parquet"].get("path") != str(source_root / relative)
    ):
      raise RuntimeError(f"USB conversion snapshot episode {index} identity differs")
    rows.extend(
      (
        {
          "path": str(source_root / relative),
          **_artifact_identity(
            episode["source_parquet"], f"USB source episode {index}"
          ),
        },
        {
          "path": str(Path(root) / relative),
          **_artifact_identity(
            {
              "size_bytes": episode.get("parquet_size_bytes"),
              "sha256": episode.get("parquet_sha256"),
            },
            f"USB output episode {index}",
          ),
        },
      )
    )
  rows.sort(key=lambda row: row["path"])
  return {
    "algorithm": "sha256",
    "file_count": len(rows),
    "total_size_bytes": sum(row["size_bytes"] for row in rows),
    "canonical_sha256": _canonical_hash(rows),
  }


def _validate_v2_normalization(record: dict, **assets: bytes) -> None:
  # Only the model environment imports OpenPI; script discovery and historical
  # H30 compatibility do not require its JAX/model dependencies.
  from openpi.training.kaihand_normalization_provenance import validate_checkpoint_v2

  validate_checkpoint_v2(record, **assets)


def checkpoint_contract(
  checkpoint: Path, *, verify_numeric_normalization: bool = True
) -> tuple[str, str, dict | None]:
  """Resolve authenticated assets, without reopening the source dataset.

  Prepare and the model service always perform full numeric report validation.
  Simulator children may verify every frozen byte/semantic binding instead,
  provided their returned proof exactly equals that model-validated deployment.
  This avoids importing model-only dependencies into the simulator process.
  """
  assets = checkpoint / "assets"
  proof_path = assets / TRAINING_PROVENANCE_FILENAME
  snapshot_path = assets / CONVERSION_SNAPSHOT_FILENAME
  if not proof_path.exists():
    if snapshot_path.exists():
      raise RuntimeError("incomplete USB checkpoint training provenance")
    return CONFIG_NAME, normalizer_asset_id(checkpoint), None
  proof_bytes, proof_sha256 = stable_file_bytes(proof_path)
  snapshot_bytes, snapshot_sha256 = stable_file_bytes(snapshot_path)
  record = _object(proof_bytes, "training provenance")
  snapshot = _object(snapshot_bytes, "conversion snapshot")
  if (
    record.get("schema") != "kaihand_training_provenance_v1"
    or record.get("binding_status") != "bound"
  ):
    raise RuntimeError("USB checkpoint training provenance must be bound")
  if record.get("task") != "usb_insert" or snapshot.get("task") != "usb_insert":
    raise RuntimeError("USB checkpoint training provenance task must be usb_insert")
  config_name = record.get("config_name")
  if config_name not in CONFIG_HORIZONS:
    raise RuntimeError(f"unsupported USB checkpoint config: {config_name!r}")
  conversion = record.get("conversion_manifest")
  expected_conversion = {
    "filename": USB_MIGRATION_FILENAME,
    "schema": USB_MIGRATION_SCHEMA,
    "sha256": snapshot_sha256,
    "size_bytes": len(snapshot_bytes),
  }
  if (
    conversion != expected_conversion
    or snapshot.get("schema_version") != USB_MIGRATION_SCHEMA
  ):
    raise RuntimeError(
      "USB conversion snapshot identity differs from training provenance"
    )
  transform = record.get("training_action_transform")
  expected_transform = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": CONFIG_HORIZONS[config_name],
    "arm_delta_mask": [True] * 7 + [False] * 20,
  }
  if (
    not isinstance(transform, dict)
    or any(transform.get(key) != value for key, value in expected_transform.items())
    or any(
      type(transform.get(key)) is not int
      for key in ("action_dim", "model_action_dim", "action_horizon")
    )
    or not isinstance(transform.get("arm_delta_mask"), list)
    or any(type(value) is not bool for value in transform["arm_delta_mask"])
    or type(transform.get("skip_episode_prefix_frames")) is not int
    or transform["skip_episode_prefix_frames"] < 0
  ):
    raise RuntimeError("USB checkpoint training action transform mismatch")
  contract = record.get("action_contract")
  if (
    not isinstance(contract, dict)
    or snapshot.get("action_contract") != contract
    or record.get("action_contract_sha256") != _canonical_hash(contract)
    or contract.get("id") != "legacy-measured-arm-hybrid-v1"
    or contract.get("legacy") is not True
    or contract.get("dimensions") != 27
    or not isinstance(contract.get("joint_order"), list)
    or len(contract["joint_order"]) != 27
    or len(set(contract["joint_order"])) != 27
    or not all(isinstance(name, str) and name for name in contract["joint_order"])
  ):
    raise RuntimeError("USB checkpoint action contract is not exactly bound")
  execution = contract.get("evaluation_execution")
  if not isinstance(execution, dict) or any(
    execution.get(key) != expected
    for key, expected in {
      "endpoint_rate_hz": 30,
      "physical_substep_interpolation_required": "linear",
      "zero_order_hold_equivalent": False,
    }.items()
  ):
    raise RuntimeError("USB checkpoint action execution contract mismatch")
  dataset = record.get("dataset")
  lerobot = snapshot.get("lerobot")
  if (
    not isinstance(dataset, dict)
    or not isinstance(dataset.get("repo_id"), str)
    or not dataset["repo_id"]
    or not isinstance(lerobot, dict)
    or not isinstance(lerobot.get("repo_id"), str)
    or not lerobot["repo_id"]
    or dataset.get("manifest_repo_id") != lerobot["repo_id"]
    or dataset.get("artifacts") != _usb_snapshot_artifacts(snapshot, dataset)
  ):
    raise RuntimeError("USB dataset identity differs from conversion snapshot")
  normalizer = record.get("normalizer")
  if not isinstance(normalizer, dict) or normalizer.get("keys") != ["actions", "state"]:
    raise RuntimeError("USB checkpoint normalizer record is invalid")
  if type(normalizer.get("use_quantile_norm")) is not bool:
    raise RuntimeError("USB checkpoint normalizer must bind normalization mode")
  asset_id = normalizer_asset_id(checkpoint, normalizer.get("asset_id"))
  directory = assets / asset_id
  stats_path = directory / "norm_stats.json"
  stats_bytes, stats_sha256 = stable_file_bytes(stats_path)
  if normalizer.get("semantic_sha256") != _normalizer_semantic_sha256(stats_path):
    raise RuntimeError("USB checkpoint normalizer semantic SHA-256 mismatch")
  stats_proof = normalizer.get("stats_provenance")
  if (
    not isinstance(stats_proof, dict)
    or stats_proof.get("filename") != "norm_stats.provenance.json"
  ):
    raise RuntimeError("USB normalization-statistics provenance is missing")
  _artifact_identity(stats_proof, "normalization-statistics provenance")
  stats_record = stats_proof.get("record")
  base_stats_record = {
    "schema": "kaihand_norm_stats_provenance_v1",
    "dataset_root": dataset["root"],
    "conversion_manifest": {
      key: conversion[key] for key in ("filename", "sha256", "size_bytes")
    },
    "parameters": {
      "prefix_frames": transform["skip_episode_prefix_frames"],
      "action_horizon": transform["action_horizon"],
      "arm_dim": 7,
    },
    "norm_stats": {
      "filename": "norm_stats.json",
      "sha256": stats_sha256,
      "size_bytes": len(stats_bytes),
    },
  }
  if not isinstance(stats_record, dict):
    raise RuntimeError("USB normalization-statistics provenance record is missing")
  sidecar = directory / "norm_stats.provenance.json"
  if stats_record.get("schema") == "kaihand_norm_stats_provenance_v2":
    sidecar_bytes, _ = stable_file_bytes(sidecar)
    report_bytes, _ = stable_file_bytes(directory / "norm_stats.scale_report.json")
    expected_v2 = {
      **base_stats_record,
      "schema": "kaihand_norm_stats_provenance_v2",
      "normalization_policy": stats_record.get("normalization_policy"),
      "scale_report": {
        "filename": "norm_stats.scale_report.json",
        "size_bytes": len(report_bytes),
        "sha256": hashlib.sha256(report_bytes).hexdigest(),
      },
    }
    if (
      stats_record != expected_v2
      or _object(sidecar_bytes, "normalization provenance") != stats_record
      or _artifact_identity(stats_proof, "normalization provenance")
      != {
        "size_bytes": len(sidecar_bytes),
        "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
      }
      or normalizer.get("use_quantile_norm") is not True
      or not isinstance(stats_record.get("normalization_policy"), dict)
      or stats_record["normalization_policy"].get("method")
      != "usb_insert_hand_tail_minmax_v1"
    ):
      raise RuntimeError(
        "USB v2 normalization bytes/report/policy differ from checkpoint binding"
      )
    if verify_numeric_normalization:
      _validate_v2_normalization(
        record,
        conversion_manifest_bytes=snapshot_bytes,
        norm_stats_bytes=stats_bytes,
        sidecar_bytes=sidecar_bytes,
        scale_report_bytes=report_bytes,
      )
  elif stats_record != base_stats_record:
    raise RuntimeError(
      "USB normalization statistics are not bound to checkpoint assets"
    )
  elif sidecar.exists():
    sidecar_bytes, sidecar_sha256 = stable_file_bytes(sidecar)
    if _object(
      sidecar_bytes, "normalization provenance"
    ) != stats_record or _artifact_identity(
      stats_proof, "normalization provenance"
    ) != {
      "size_bytes": len(sidecar_bytes),
      "sha256": sidecar_sha256,
    }:
      raise RuntimeError("USB normalization provenance sidecar identity differs")
  binding = {
    "status": "bound",
    "provenance_asset": {
      "path": f"assets/{TRAINING_PROVENANCE_FILENAME}",
      "size_bytes": len(proof_bytes),
      "sha256": proof_sha256,
    },
    "conversion_manifest_snapshot": {
      "path": f"assets/{CONVERSION_SNAPSHOT_FILENAME}",
      "size_bytes": len(snapshot_bytes),
      "sha256": snapshot_sha256,
    },
    "record": record,
  }
  return config_name, asset_id, binding


def deployment_identity(manifest: dict) -> dict:
  fields = (
    _V2_IDENTITY_FIELDS if manifest.get("schema") == SCHEMA_V2 else _IDENTITY_FIELDS
  )
  missing = [key for key in fields if key not in manifest]
  if missing:
    raise RuntimeError(f"USB deployment identity fields are missing: {missing}")
  return {key: manifest[key] for key in fields}


def validate_deployment_identity(manifest: dict) -> None:
  """Authenticate v2 identity without opening model/dataset/checkpoint files."""
  schema = manifest.get("schema")
  if schema not in {SCHEMA_V1, SCHEMA_V2}:
    raise RuntimeError(
      "expected usb_pi05_deployment_v1 or usb_pi05_deployment_v2 manifest"
    )
  if manifest.get("task") != "usb-insert" or manifest.get("model_family") != "pi0.5":
    raise RuntimeError("manifest is not a USB pi0.5 deployment")
  if schema == SCHEMA_V1:
    if manifest.get("model_config_name") != CONFIG_NAME:
      raise RuntimeError(f"historical manifest must use OpenPI config {CONFIG_NAME!r}")
    return
  config_name = manifest.get("model_config_name")
  if config_name not in CONFIG_HORIZONS:
    raise RuntimeError(f"unsupported USB deployment config: {config_name!r}")
  if manifest.get("prediction_horizon") != CONFIG_HORIZONS[config_name]:
    raise RuntimeError("USB deployment config and horizon disagree")
  if manifest.get("action_dim") != 27 or manifest.get("model_action_dim") != 32:
    raise RuntimeError(
      "USB deployment must bind physical/model action dimensions 27/32"
    )
  asset_id = manifest.get("normalizer_asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("USB deployment normalizer asset id must be one directory name")
  if manifest.get("deployment_id") != _canonical_hash(deployment_identity(manifest)):
    raise RuntimeError("deployment_id does not authenticate USB deployment identity")
  if manifest.get("required_action_interpolation") != "linear":
    raise RuntimeError("USB v2 requires physical-substep linear action interpolation")
  binding = manifest.get("training_provenance")
  if not isinstance(binding, dict) or binding.get("status") != "bound":
    raise RuntimeError(
      "USB v2 deployment requires bound checkpoint training provenance"
    )
  record = binding.get("record")
  if (
    not isinstance(record, dict)
    or record.get("task") != "usb_insert"
    or record.get("config_name") != config_name
    or not isinstance(record.get("normalizer"), dict)
    or record["normalizer"].get("asset_id") != manifest.get("normalizer_asset_id")
  ):
    raise RuntimeError(
      "USB manifest config/task/asset differs from bound training provenance"
    )


def checkpoint_identity(checkpoint: Path, workers: int) -> tuple[str, list[dict]]:
  paths = [checkpoint / "_CHECKPOINT_METADATA"]
  for directory in (checkpoint / "params", checkpoint / "assets"):
    if not directory.is_dir():
      raise ValueError(f"missing inference checkpoint directory: {directory}")
    paths.extend(path for path in directory.rglob("*") if path.is_file())
  if not paths[0].is_file():
    raise ValueError(f"checkpoint is not committed: {paths[0]} is missing")
  paths = sorted(set(paths), key=lambda path: path.relative_to(checkpoint).as_posix())
  with ThreadPoolExecutor(max_workers=workers) as pool:
    hashes = list(pool.map(sha256_file, paths))
  rows = [
    {
      "path": path.relative_to(checkpoint).as_posix(),
      "size": path.stat().st_size,
      "sha256": digest,
    }
    for path, digest in zip(paths, hashes, strict=True)
  ]
  return _canonical_hash(rows), rows


def main(argv=None) -> int | None:
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
    "--dry-run",
    action="store_true",
    help="Validate committed checkpoint/config/provenance without hashing weights or creating output",
  )
  args = parser.parse_args(argv)
  if args.hash_workers <= 0:
    parser.error("hash-workers must be positive")

  # This script is intentionally run with the model environment so metadata is
  # derived from the exact OpenPI config rather than duplicated by hand.
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
  if output.exists():
    raise FileExistsError(f"refusing to overwrite deployment directory: {output}")

  config_name, asset_id, training_provenance = checkpoint_contract(checkpoint)
  train_config = openpi_config.get_config(config_name)
  if train_config.name != config_name:
    raise RuntimeError("OpenPI runtime config name differs from checkpoint provenance")
  metadata = dict(train_config.policy_metadata or {})
  required_metadata = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": CONFIG_HORIZONS[config_name],
    "control_hz": 30,
    "suggested_replan_steps": 8,
    "image_keys": [
      "observation.images.head",
      "observation.images.right_wrist",
    ],
  }
  mismatches = {
    key: {"expected": expected, "actual": metadata.get(key)}
    for key, expected in required_metadata.items()
    if metadata.get(key) != expected
  }
  if mismatches:
    raise RuntimeError(f"OpenPI USB config contract mismatch: {mismatches}")
  if int(train_config.model.action_horizon) != metadata["action_horizon"]:
    raise RuntimeError("model horizon and policy metadata disagree")
  if int(train_config.model.action_dim) != metadata["model_action_dim"]:
    raise RuntimeError("model action dimension and policy metadata disagree")
  if train_config.data.__class__.__name__ != "LeRobotKaiHandDataConfig":
    raise RuntimeError(
      f"expected LeRobotKaiHandDataConfig, got {train_config.data.__class__.__name__}"
    )
  if train_config.data.use_delta_arm_actions is not True:
    raise RuntimeError("USB pi0.5 inference requires delta arm action transforms")
  if training_provenance is not None:
    record = training_provenance["record"]
    if metadata.get("joint_names") != record["action_contract"]["joint_order"]:
      raise RuntimeError("OpenPI joint order differs from checkpoint provenance")
    if metadata.get("action_semantics") != action_representation_from_provenance(
      record
    ):
      raise RuntimeError("OpenPI action semantics differ from checkpoint provenance")
    if hasattr(train_config.model, "model_type"):
      model_type = train_config.model.model_type
      expected_quantile = getattr(model_type, "value", model_type) != "pi0"
      if record["normalizer"]["use_quantile_norm"] is not expected_quantile:
        raise RuntimeError(
          "checkpoint quantile normalization differs from OpenPI model config"
        )

  if args.dry_run:
    for required in (
      checkpoint / "_CHECKPOINT_METADATA",
      checkpoint / "params/_METADATA",
    ):
      if not required.is_file():
        raise RuntimeError(f"checkpoint is not committed: missing {required}")
    print(
      json.dumps(
        {
          "valid": True,
          "task": "usb-insert",
          "config_name": config_name,
          "prediction_horizon": metadata["action_horizon"],
          "normalizer_asset_id": asset_id,
          "training_provenance": training_provenance,
        },
        ensure_ascii=False,
        indent=2,
      )
    )
    return 0

  print(f"Hashing committed inference payload under {checkpoint} ...", flush=True)
  checkpoint_sha256, checkpoint_files = checkpoint_identity(
    checkpoint, args.hash_workers
  )
  if checkpoint_contract(checkpoint) != (config_name, asset_id, training_provenance):
    raise RuntimeError("USB checkpoint training assets changed while hashing")
  source_paths = [
    openpi_root / "src/openpi/training/config.py",
    openpi_root / "src/openpi/policies/kaihand_policy.py",
    openpi_root / "src/openpi/policies/policy_config.py",
    openpi_root / "src/openpi/serving/websocket_policy_server.py",
  ]
  if training_provenance is not None:
    source_paths = openpi_source_paths(openpi_root)
  model_source_rows = [
    {
      "path": path.relative_to(openpi_root).as_posix(),
      "size": path.stat().st_size,
      "sha256": sha256_file(path),
    }
    for path in source_paths
  ]
  evaluation_rows = (
    source_rows(evaluation_source_paths(), ROOT)
    if training_provenance is not None
    else None
  )
  identity = {
    "task": "usb-insert",
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
        "openpi_source_sha256": _canonical_hash(model_source_rows),
        "evaluation_source_sha256": _canonical_hash(evaluation_rows),
        "required_action_interpolation": "linear",
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
    "openpi_source_sha256": _canonical_hash(model_source_rows),
    "openpi_source_files": model_source_rows,
  }
  if training_provenance is not None:
    manifest["evaluation_source_files"] = evaluation_rows
  validate_deployment_identity(manifest)
  output.mkdir(parents=True)
  path = output / "deployment_manifest.json"
  path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"deployment_manifest={path}")
  print(f"deployment_id={deployment_id}")
  print(f"checkpoint_sha256={checkpoint_sha256}")


if __name__ == "__main__":
  main()
