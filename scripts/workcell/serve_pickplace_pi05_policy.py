#!/usr/bin/env python3
"""Serve a frozen PickPlace pi0.5 deployment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", required=True, type=Path)
  parser.add_argument("--port", required=True, type=int)
  args = parser.parse_args()
  if not 1 <= args.port <= 65535:
    parser.error("port must be in 1..65535")

  from openpi.policies import policy_config
  from openpi.serving import websocket_policy_server
  from openpi.training import config as openpi_config

  manifest_path = args.deployment_manifest.resolve(strict=True)
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  if manifest.get("schema") != "pickplace_pi05_deployment_v1":
    raise RuntimeError("not a PickPlace pi0.5 deployment")
  checkpoint = Path(manifest["checkpoint_path"])
  for row in manifest["checkpoint_files"]:
    path = checkpoint / row["path"]
    if not path.is_file() or path.stat().st_size != row["size"]:
      raise RuntimeError(f"checkpoint payload changed: {path}")
  config = openpi_config.get_config(manifest["model_config_name"])
  metadata = dict(config.policy_metadata or {})
  expected = {
    "action_horizon": manifest["prediction_horizon"],
    "model_action_dim": manifest["model_action_dim"],
    "action_dim": manifest["action_dim"],
    "control_hz": manifest["control_hz"],
    "joint_names": manifest["joint_names"],
    "task": manifest["instruction"],
  }
  if any(metadata.get(key) != value for key, value in expected.items()):
    raise RuntimeError("current OpenPI config differs from frozen deployment")
  policy = policy_config.create_trained_policy(config, checkpoint, default_prompt=manifest["instruction"])
  server_metadata = {
    **metadata,
    "evaluation_task": "pick-place",
    "model_family": "pi0.5",
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "prediction_horizon": manifest["prediction_horizon"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
  }
  logging.info("Serving PickPlace step=%s H=%s port=%s", manifest["checkpoint_step"], manifest["prediction_horizon"], args.port)
  websocket_policy_server.WebsocketPolicyServer(
    policy=policy, host="127.0.0.1", port=args.port, metadata=server_metadata
  ).serve_forever()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, force=True)
  main()
