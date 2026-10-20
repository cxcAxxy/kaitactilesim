#!/usr/bin/env python3
"""Serve one manifest-bound KaiHand USB pi0.5 checkpoint over WebSocket."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from kaihand_tactile_env.shared.evaluation_source_integrity import (
  verify_evaluation_sources,
)
from kaihand_tactile_env.shared.pi05_deployment_integrity import verify_pi05_deployment
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  pi05_image_payload,
  policy_camera_names,
)

try:
  from prepare_usb_pi05_deployment import (
    ROOT,
    SCHEMA_V2,
    checkpoint_contract,
    evaluation_source_paths,
    validate_deployment_identity,
  )
except ModuleNotFoundError as error:
  if error.name != "prepare_usb_pi05_deployment":
    raise
  from scripts.workcell.prepare_usb_pi05_deployment import (
    ROOT,
    SCHEMA_V2,
    checkpoint_contract,
    evaluation_source_paths,
    validate_deployment_identity,
  )


def bind_checkpoint_assets(config, manifest: dict):
  asset_id = manifest.get("normalizer_asset_id")
  if (
    not isinstance(asset_id, str)
    or not asset_id
    or Path(asset_id).name != asset_id
    or asset_id in {".", ".."}
  ):
    raise RuntimeError("manifest normalizer_asset_id must be one directory name")
  replacements = {"asset_id": asset_id}
  if hasattr(config.data.assets, "assets_dir"):
    replacements["assets_dir"] = str(Path(manifest["checkpoint_path"]) / "assets")
  assets = dataclasses.replace(config.data.assets, **replacements)
  data = dataclasses.replace(config.data, assets=assets)
  return dataclasses.replace(config, data=data)


def _load_manifest(path: Path) -> tuple[Path, dict]:
  resolved = path.expanduser().resolve()
  payload = json.loads(resolved.read_text(encoding="utf-8"))
  validate_deployment_identity(payload)
  checkpoint = Path(payload["checkpoint_path"])
  if not checkpoint.is_dir():
    raise FileNotFoundError(checkpoint)
  for row in payload.get("checkpoint_files", []):
    source = checkpoint / row["path"]
    if not source.is_file() or source.stat().st_size != row["size"]:
      raise RuntimeError(f"checkpoint file snapshot mismatch: {source}")
  normalizer = (
    checkpoint / "assets" / payload.get("normalizer_asset_id", "") / "norm_stats.json"
  )
  if not normalizer.is_file():
    raise FileNotFoundError(
      f"manifest-bound normalization stats are missing: {normalizer}"
    )
  config_name, asset_id, training_provenance = checkpoint_contract(checkpoint)
  if (
    payload["model_config_name"] != config_name
    or payload["normalizer_asset_id"] != asset_id
  ):
    raise RuntimeError(
      "manifest config/asset differs from checkpoint training provenance"
    )
  if (
    payload["schema"] == SCHEMA_V2
    and payload.get("training_provenance") != training_provenance
  ):
    raise RuntimeError("manifest training provenance differs from checkpoint assets")
  if payload["schema"] == SCHEMA_V2:
    proof = verify_evaluation_sources(payload, ROOT)
    declared = {row["path"] for row in payload["evaluation_source_files"]}
    actual = {entry.relative_to(ROOT).as_posix() for entry in evaluation_source_paths()}
    if proof["status"] != "verified" or declared != actual:
      raise RuntimeError("USB evaluation source content or inventory changed")
  return resolved, payload


def warmup_policy(policy: Any, manifest: dict, *, monotonic=time.monotonic) -> dict:
  """Compile the exact USB request before listening, without consuming its RNG."""
  if not hasattr(policy, "_rng"):
    raise RuntimeError("USB pi0.5 policy must expose its JAX RNG")
  observation = manifest["observation_contract"]
  image_shape = image_shape_hwc(observation)
  action_shape = (
    int(manifest["prediction_horizon"]),
    int(manifest["action_dim"]),
  )
  images = {
    name: np.zeros(image_shape, dtype=np.uint8)
    for name in policy_camera_names(observation)
  }
  request = {
    **pi05_image_payload(images, observation),
    "state": np.zeros(action_shape[1], dtype=np.float32),
    "prompt": manifest["instruction"],
  }
  original_rng = policy._rng
  started = monotonic()
  try:
    response = policy.infer(request)
  finally:
    # The first scored request must receive the same key as without compilation.
    policy._rng = original_rng
  elapsed = monotonic() - started
  if not isinstance(response, dict) or "actions" not in response:
    raise RuntimeError("USB pi0.5 warmup response must contain actions")
  actions = np.asarray(response["actions"])
  if actions.shape != action_shape:
    raise RuntimeError(
      f"USB pi0.5 warmup actions must have shape {action_shape}, got {actions.shape}"
    )
  if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
    raise RuntimeError("USB pi0.5 warmup actions must be finite numeric values")
  return {
    "completed": True,
    "duration_s": elapsed,
    "request_image_shape_hwc": list(image_shape),
    "state_dim": action_shape[1],
    "action_shape": list(action_shape),
    "rng_restored": True,
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
  if (
    int(config.model.action_horizon) != manifest["prediction_horizon"]
    or int(config.model.action_dim) != manifest["model_action_dim"]
    or config.data.use_delta_arm_actions is not True
    or metadata.get("action_semantics") != manifest["action_representation"]
  ):
    raise RuntimeError("current OpenPI action transforms differ from deployment")
  _, _, training_provenance = checkpoint_contract(Path(manifest["checkpoint_path"]))
  expected_mode = (
    training_provenance["record"]["normalizer"]["use_quantile_norm"]
    if training_provenance is not None
    else None
  )
  policy = policy_config.create_trained_policy(
    config,
    manifest["checkpoint_path"],
    default_prompt=manifest["instruction"],
    expected_use_quantile_norm=expected_mode,
  )
  warmup = None
  if manifest["schema"] == SCHEMA_V2:
    logging.info(
      "WARMUP_START task=usb-insert deployment=%s H=%s",
      manifest["deployment_id"],
      manifest["prediction_horizon"],
    )
    warmup = warmup_policy(policy, manifest)
    logging.info(
      "WARMUP_OK duration_s=%.3f action_shape=%s rng_restored=%s",
      warmup["duration_s"],
      warmup["action_shape"],
      warmup["rng_restored"],
    )
  server_metadata = {
    **metadata,
    "evaluation_task": "usb-insert",
    "model_family": manifest["model_family"],
    "model_config_name": manifest["model_config_name"],
    "normalizer_asset_id": manifest["normalizer_asset_id"],
    "training_provenance": training_provenance,
    "deployment_manifest": str(manifest_path),
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "prediction_horizon": manifest["prediction_horizon"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
  }
  if manifest["schema"] == SCHEMA_V2:
    server_metadata["server_warmup"] = warmup
    server_metadata["required_action_interpolation"] = manifest[
      "required_action_interpolation"
    ]
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
