#!/usr/bin/env python3
"""Freeze a shared-task KaiHand pi0.5 checkpoint for model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "pi05_kaihand"
SCHEMA = "shared_task_pi05_deployment_v1"
NORMALIZER_ASSET_ID = "normalizer"
TASK_INSTRUCTIONS = {
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
}
EXPECTED_POLICY_METADATA = {
  "action_dim": 27,
  "model_action_dim": 32,
  "action_horizon": 30,
  "control_hz": 30,
  "suggested_replan_steps": 8,
  "image_keys": [
    "observation.images.head",
    "observation.images.right_wrist",
  ],
}


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


def normalizer_asset_id(checkpoint: Path) -> str:
  norm_stats = checkpoint / "assets" / NORMALIZER_ASSET_ID / "norm_stats.json"
  if not norm_stats.is_file():
    raise ValueError(f"missing checkpoint normalization stats: {norm_stats}")
  return NORMALIZER_ASSET_ID


def validate_train_config(config) -> dict:
  metadata = dict(config.policy_metadata or {})
  mismatches = {
    key: {"expected": expected, "actual": metadata.get(key)}
    for key, expected in EXPECTED_POLICY_METADATA.items()
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
    "--dry-run",
    action="store_true",
    help="Validate the checkpoint/config contract without hashing or writing output",
  )
  args = parser.parse_args(argv)
  if args.hash_workers <= 0:
    parser.error("hash-workers must be positive")
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

  config = openpi_config.get_config(CONFIG_NAME)
  metadata = validate_train_config(config)
  asset_id = normalizer_asset_id(checkpoint)
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
          "model_config_name": CONFIG_NAME,
          "prediction_horizon": metadata["action_horizon"],
          "normalizer_asset_id": asset_id,
          "cameras": ["head", "right_wrist"],
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
  openpi_sources = [
    openpi_root / "src/openpi/training/config.py",
    openpi_root / "src/openpi/policies/kaihand_policy.py",
    openpi_root / "src/openpi/policies/policy_config.py",
    openpi_root / "src/openpi/serving/websocket_policy_server.py",
  ]
  evaluation_sources = [
    ROOT / "src/kaihand_tactile_env/shared/policy_tasks.py",
    ROOT / "scripts/workcell/run_shared_task_pi05_policy.py",
    ROOT / "scripts/workcell/serve_shared_task_pi05_policy.py",
  ]

  def source_rows(paths: list[Path], base: Path) -> list[dict]:
    return [
      {
        "path": path.relative_to(base).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
      }
      for path in paths
    ]

  openpi_rows = source_rows(openpi_sources, openpi_root)
  evaluation_rows = source_rows(evaluation_sources, ROOT)
  identity = {
    "task": args.task,
    "model_family": "pi0.5",
    "model_config_name": CONFIG_NAME,
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
    "instruction": TASK_INSTRUCTIONS[args.task],
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "image_dtype": "uint8",
      "state": "27 measured right-side joint positions in joint_names order",
      "language_instruction": True,
      "tactile_sent_to_model": False,
    },
    "action_representation": metadata["action_semantics"],
    "model_project_path": str(openpi_root),
    "openpi_source_sha256": canonical_hash(openpi_rows),
    "evaluation_source_sha256": canonical_hash(evaluation_rows),
  }
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
