#!/usr/bin/env python3
"""Freeze a PickPlace pi0.5 checkpoint and its head-only policy contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_usb_pi05_deployment import _canonical_hash, checkpoint_identity, sha256_file

CONFIG_NAME = "pi05_kaihand_pickplace"
TASK = "Put the red cylinder into the blue box."


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--openpi-root", type=Path, default=Path("/cpfs_infra/user/chenxianchi/code/openpi"))
  parser.add_argument("--hash-workers", type=int, default=4)
  args = parser.parse_args()
  if args.hash_workers < 1:
    parser.error("hash-workers must be positive")

  from openpi.training import config as openpi_config

  checkpoint = args.checkpoint.resolve(strict=True)
  root = args.openpi_root.resolve(strict=True)
  output = args.output_dir.resolve()
  if not checkpoint.is_dir() or output.exists():
    raise RuntimeError("checkpoint must be a directory and output must be new")
  step = int(checkpoint.name)
  config = openpi_config.get_config(CONFIG_NAME)
  metadata = dict(config.policy_metadata or {})
  if (int(config.model.action_horizon) != int(metadata.get("action_horizon", -1))
      or int(config.model.action_dim) != int(metadata.get("model_action_dim", -1))
      or metadata.get("action_dim") != 27
      or metadata.get("model_action_dim") != 32
      or metadata.get("control_hz") != 30
      or metadata.get("task") != TASK):
    raise RuntimeError(f"PickPlace pi0.5 model/metadata contract mismatch: {metadata}")
  source = root / "src/openpi/training/config.py"
  if '"image": "observation.images.head"' not in source.read_text(encoding="utf-8"):
    raise RuntimeError("expected head camera mapping is absent from model config")
  checkpoint_hash, files = checkpoint_identity(checkpoint, args.hash_workers)
  identity = {
    "task": "pick-place",
    "model_family": "pi0.5",
    "model_config_name": CONFIG_NAME,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_step": step,
    "prediction_horizon": int(metadata["action_horizon"]),
    "model_action_dim": int(metadata["model_action_dim"]),
    "action_dim": int(metadata["action_dim"]),
    "control_hz": int(metadata["control_hz"]),
    "joint_names": list(metadata["joint_names"]),
    "instruction": TASK,
    "observation_contract": {
      "cameras": ["head"],
      "image_shape_hwc": [240, 320, 3],
      "state_dim": 27,
      "tactile_sent_to_model": False,
    },
    "action_representation": metadata["action_semantics"],
    "model_project_path": str(root),
    "model_source_sha256": sha256_file(source),
  }
  manifest = {
    "schema": "pickplace_pi05_deployment_v1",
    "deployment_id": _canonical_hash(identity),
    **identity,
    "checkpoint_files": files,
    "checkpoint_hash_algorithm": "sha256(canonical path,size,file_sha256 over committed inference payload)",
  }
  output.mkdir(parents=True)
  destination = output / "deployment_manifest.json"
  destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"deployment_manifest={destination} H={manifest['prediction_horizon']} sha256={checkpoint_hash}", flush=True)


if __name__ == "__main__":
  main()
