#!/usr/bin/env python3
"""Create an immutable inference identity for a KaiHand card pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_usb_pi05_deployment import (
  _canonical_hash,
  checkpoint_identity,
  sha256_file,
)

CONFIG_NAME = "pi05_kaihand_card_0914_200"
EXPECTED_TASK = (
  "Slide the face-down card toward the table edge, pinch it between the fingers "
  "and thumb, lift it, and turn its face toward the robot to look at it."
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--openpi-root",
    type=Path,
    default=Path("/cpfs_infra/user/chenxianchi/code/openpi"),
  )
  parser.add_argument("--hash-workers", type=int, default=4)
  args = parser.parse_args()
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
  if output.exists():
    raise FileExistsError(f"refusing to overwrite deployment directory: {output}")

  train_config = openpi_config.get_config(CONFIG_NAME)
  metadata = dict(train_config.policy_metadata or {})
  required_metadata = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": 30,
    "control_hz": 30,
    "suggested_replan_steps": 8,
    "image_keys": [
      "observation.images.head",
      "observation.images.right_wrist",
    ],
    "task": EXPECTED_TASK,
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

  print(f"Hashing committed inference payload under {checkpoint} ...", flush=True)
  checkpoint_sha256, checkpoint_files = checkpoint_identity(
    checkpoint, args.hash_workers
  )
  source_paths = [
    openpi_root / "src/openpi/training/config.py",
    openpi_root / "src/openpi/policies/kaihand_policy.py",
    openpi_root / "src/openpi/policies/kaihand_card_policy.py",
    openpi_root / "src/openpi/policies/policy_config.py",
    openpi_root / "src/openpi/serving/websocket_policy_server.py",
  ]
  source_rows = [
    {
      "path": path.relative_to(openpi_root).as_posix(),
      "size": path.stat().st_size,
      "sha256": sha256_file(path),
    }
    for path in source_paths
  ]
  identity = {
    "task": "poker-draw",
    "model_family": "pi0.5",
    "model_config_name": CONFIG_NAME,
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
  deployment_id = _canonical_hash(identity)
  manifest = {
    "schema": "poker_pi05_deployment_v1",
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
    "instruction": metadata["task"],
    "model_project_path": str(openpi_root),
    "openpi_source_sha256": _canonical_hash(source_rows),
    "openpi_source_files": source_rows,
  }
  output.mkdir(parents=True)
  path = output / "deployment_manifest.json"
  path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"deployment_manifest={path}")
  print(f"deployment_id={deployment_id}")
  print(f"checkpoint_sha256={checkpoint_sha256}")


if __name__ == "__main__":
  main()
