#!/usr/bin/env python3
"""Serve one manifest-bound KaiHand card pi0.5 checkpoint over WebSocket."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from kaihand_tactile_env.shared.pi05_deployment_integrity import verify_pi05_deployment

CONFIG_NAME = "pi05_kaihand"


def bind_checkpoint_assets(config, manifest: dict):
  asset_id = manifest.get("normalizer_asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("manifest normalizer_asset_id must be one directory name")
  assets = dataclasses.replace(config.data.assets, asset_id=asset_id)
  data = dataclasses.replace(config.data, assets=assets)
  return dataclasses.replace(config, data=data)


def _load_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  if payload.get("schema") != "poker_pi05_deployment_v1":
    raise RuntimeError("expected poker_pi05_deployment_v1 manifest")
  if payload.get("task") != "poker-draw" or payload.get("model_family") != "pi0.5":
    raise RuntimeError("manifest is not a poker-draw pi0.5 deployment")
  if payload.get("model_config_name") != CONFIG_NAME:
    raise RuntimeError(f"manifest must use OpenPI config {CONFIG_NAME!r}")
  checkpoint = Path(payload["checkpoint_path"])
  if not checkpoint.is_dir():
    raise FileNotFoundError(checkpoint)
  for row in payload.get("checkpoint_files", []):
    source = checkpoint / row["path"]
    if not source.is_file() or source.stat().st_size != row["size"]:
      raise RuntimeError(f"checkpoint file snapshot mismatch: {source}")
  asset_id = payload.get("normalizer_asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("manifest normalizer_asset_id must be one directory name")
  normalizer = checkpoint / "assets" / asset_id / "norm_stats.json"
  if not normalizer.is_file():
    raise FileNotFoundError(
      f"manifest-bound normalization stats are missing: {normalizer}"
    )
  return resolved, payload


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--port", type=int, default=18784)
  args = parser.parse_args()
  if not 1 <= args.port <= 65535:
    parser.error("port must be in 1..65535")

  from openpi.policies import policy_config
  from openpi.serving import websocket_policy_server
  from openpi.training import config as openpi_config

  manifest_path, manifest = _load_manifest(args.deployment_manifest)
  verify_pi05_deployment(manifest)
  config = bind_checkpoint_assets(
    openpi_config.get_config(manifest["model_config_name"]), manifest
  )
  expected = {
    "action_dim": manifest["action_dim"],
    "model_action_dim": manifest["model_action_dim"],
    "action_horizon": manifest["prediction_horizon"],
    "control_hz": manifest["control_hz"],
    "suggested_replan_steps": manifest["suggested_execute_steps"],
    "joint_names": manifest["joint_names"],
    "image_keys": [
      "observation.images.head",
      "observation.images.right_wrist",
    ],
  }
  metadata = dict(config.policy_metadata or {})
  mismatches = {
    key: {"expected": value, "actual": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"current OpenPI config differs from deployment: {mismatches}")
  policy = policy_config.create_trained_policy(
    config,
    manifest["checkpoint_path"],
    default_prompt=manifest["instruction"],
  )
  server_metadata = {
    **metadata,
    "evaluation_task": "poker-draw",
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
  logging.info(
    "Serving deployment %s step=%s H=%s physical_action_dim=%s on port %s",
    manifest["deployment_id"],
    manifest["checkpoint_step"],
    manifest["prediction_horizon"],
    manifest["action_dim"],
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
