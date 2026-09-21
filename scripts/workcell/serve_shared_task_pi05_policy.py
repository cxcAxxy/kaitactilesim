#!/usr/bin/env python3
"""Serve a manifest-bound shared-task KaiHand pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

CONFIG_NAME = "pi05_kaihand"
SCHEMA = "shared_task_pi05_deployment_v1"
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
TASKS = tuple(TASK_INSTRUCTIONS)
EXPECTED_IMAGE_KEYS = [
  "observation.images.head",
  "observation.images.right_wrist",
]


def normalizer_asset_id(manifest: dict) -> str:
  asset_id = manifest.get("normalizer_asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("manifest normalizer_asset_id must be one directory name")
  return asset_id


def bind_checkpoint_assets(config, manifest: dict):
  asset_id = normalizer_asset_id(manifest)
  assets = dataclasses.replace(config.data.assets, asset_id=asset_id)
  data = dataclasses.replace(config.data, assets=assets)
  return dataclasses.replace(config, data=data)


def _checkpoint_file(checkpoint: Path, relative: object) -> Path:
  if not isinstance(relative, str) or not relative:
    raise RuntimeError("checkpoint file path must be a nonempty relative string")
  path = Path(relative)
  if path.is_absolute() or ".." in path.parts:
    raise RuntimeError(f"unsafe checkpoint file path: {relative!r}")
  return checkpoint / path


def load_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve(strict=True)
  manifest = json.loads(resolved.read_text(encoding="utf-8"))
  required = (
    "schema",
    "task",
    "deployment_id",
    "model_family",
    "model_config_name",
    "normalizer_asset_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_step",
    "prediction_horizon",
    "model_action_dim",
    "action_dim",
    "control_hz",
    "suggested_execute_steps",
    "joint_names",
    "instruction",
    "observation_contract",
    "action_representation",
    "checkpoint_files",
  )
  missing = [key for key in required if key not in manifest]
  if missing:
    raise RuntimeError(f"deployment manifest missing fields: {missing}")
  if manifest["schema"] != SCHEMA:
    raise RuntimeError(f"expected {SCHEMA} manifest")
  if manifest["task"] not in TASKS or manifest["model_family"] != "pi0.5":
    raise RuntimeError("manifest is not a supported shared-task pi0.5 deployment")
  if manifest["instruction"] != TASK_INSTRUCTIONS[manifest["task"]]:
    raise RuntimeError("manifest instruction differs from the frozen task prompt")
  if manifest["model_config_name"] != CONFIG_NAME:
    raise RuntimeError(f"manifest must use OpenPI config {CONFIG_NAME!r}")
  observation = manifest["observation_contract"]
  if (
    observation.get("cameras") != ["head", "right_wrist"]
    or observation.get("image_shape_hwc") != [240, 320, 3]
    or observation.get("tactile_sent_to_model") is not False
  ):
    raise RuntimeError("unsupported shared-task pi0.5 observation contract")
  checkpoint = Path(manifest["checkpoint_path"])
  if not checkpoint.is_dir():
    raise FileNotFoundError(checkpoint)
  rows = manifest["checkpoint_files"]
  if not isinstance(rows, list) or not rows:
    raise RuntimeError("checkpoint_files must be a nonempty list")
  seen = set()
  for row in rows:
    if not isinstance(row, dict):
      raise RuntimeError("checkpoint_files entries must be objects")
    relative = row.get("path")
    source = _checkpoint_file(checkpoint, relative)
    if relative in seen:
      raise RuntimeError(f"duplicate checkpoint file path: {relative!r}")
    seen.add(relative)
    if not source.is_file() or source.stat().st_size != row.get("size"):
      raise RuntimeError(f"checkpoint file snapshot mismatch: {source}")
  asset_id = normalizer_asset_id(manifest)
  normalizer = checkpoint / "assets" / asset_id / "norm_stats.json"
  if not normalizer.is_file():
    raise FileNotFoundError(
      f"manifest-bound normalization stats are missing: {normalizer}"
    )
  return resolved, manifest


def validate_runtime_config(config, manifest: dict) -> dict:
  metadata = dict(config.policy_metadata or {})
  expected = {
    "action_dim": manifest["action_dim"],
    "model_action_dim": manifest["model_action_dim"],
    "action_horizon": manifest["prediction_horizon"],
    "control_hz": manifest["control_hz"],
    "suggested_replan_steps": manifest["suggested_execute_steps"],
    "joint_names": manifest["joint_names"],
    "image_keys": EXPECTED_IMAGE_KEYS,
    "action_semantics": manifest["action_representation"],
  }
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"current OpenPI config differs from deployment: {mismatches}")
  if int(config.model.action_horizon) != int(manifest["prediction_horizon"]):
    raise RuntimeError("model horizon differs from deployment")
  if int(config.model.action_dim) != int(manifest["model_action_dim"]):
    raise RuntimeError("model action dimension differs from deployment")
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


def deployment_metadata(metadata: dict, manifest_path: Path, manifest: dict) -> dict:
  return {
    **metadata,
    "evaluation_task": manifest["task"],
    "model_family": manifest["model_family"],
    "model_config_name": manifest["model_config_name"],
    "deployment_manifest": str(manifest_path),
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "prediction_horizon": manifest["prediction_horizon"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
  }


def main(argv=None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--port", type=int, default=18783)
  args = parser.parse_args(argv)
  if not 1 <= args.port <= 65535:
    parser.error("port must be in 1..65535")

  from openpi.policies import policy_config
  from openpi.serving import websocket_policy_server
  from openpi.training import config as openpi_config

  manifest_path, manifest = load_manifest(args.deployment_manifest)
  config = bind_checkpoint_assets(
    openpi_config.get_config(manifest["model_config_name"]), manifest
  )
  metadata = validate_runtime_config(config, manifest)
  policy = policy_config.create_trained_policy(
    config,
    manifest["checkpoint_path"],
    default_prompt=manifest["instruction"],
  )
  server_metadata = deployment_metadata(metadata, manifest_path, manifest)
  logging.info(
    "Serving task=%s deployment=%s step=%s H=%s on port %s",
    manifest["task"],
    manifest["deployment_id"],
    manifest["checkpoint_step"],
    manifest["prediction_horizon"],
    args.port,
  )
  websocket_policy_server.WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=args.port,
    metadata=server_metadata,
  ).serve_forever()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, force=True)
  main()
