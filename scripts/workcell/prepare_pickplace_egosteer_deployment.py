#!/usr/bin/env python3
"""Freeze a PickPlace EgoSteer checkpoint and its camera-aware serving contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_poker_egosteer_deployment import (
  checkpoint_identity,
  parse_model_contract,
  sha256_file,
  source_identity,
)

INSTRUCTION = "Put the red cylinder into the blue box."


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("checkpoint", "model-config", "normalizer", "model-project", "model-python", "pretrained-vlm", "output-dir"):
    parser.add_argument(f"--{name}", required=True, type=Path)
  parser.add_argument("--port", required=True, type=int)
  parser.add_argument(
    "--camera-views", choices=("head", "head,right_wrist"),
    help="Override unresolved legacy camera_views interpolation.",
  )
  args = parser.parse_args()
  checkpoint = args.checkpoint.resolve(strict=True)
  config = args.model_config.resolve(strict=True)
  normalizer = args.normalizer.resolve(strict=True)
  model_project = args.model_project.resolve(strict=True)
  model_python = args.model_python.resolve(strict=True)
  vlm = args.pretrained_vlm.resolve(strict=True)
  output = args.output_dir.resolve()
  if output.exists() or not 1 <= args.port <= 65535:
    raise RuntimeError("output must be new and port must be in 1..65535")
  contract = parse_model_contract(config, model_python, args.camera_views)
  checkpoint_hash, files = checkpoint_identity(checkpoint)
  identity = {
    "task": "pick-place", "model_family": "EgoSteer",
    "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
    "model_config_path": str(config), "model_config_sha256": sha256_file(config),
    "normalizer_path": str(normalizer), "normalizer_sha256": sha256_file(normalizer),
    "prediction_horizon": contract["prediction_horizon"], "action_dim": contract["action_dim"],
    "observation_contract": contract["observation_contract"],
    "action_representation": contract["action_representation"],
    "model_source": source_identity(model_project),
  }
  deployment_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  manifest = {
    "schema": "pickplace_egosteer_deployment_v1", "deployment_id": deployment_id,
    **identity, "checkpoint_files": files, "model_python": str(model_python),
    "pretrained_vlm_path": str(vlm), "server": f"ws://127.0.0.1:{args.port}",
  }
  server_metadata = {
    "task": "pick-place", "deployment_id": deployment_id,
    "model_family": "EgoSteer", "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_hash, "model_config_path": str(config),
    "model_config_sha256": identity["model_config_sha256"],
    "normalizer_path": str(normalizer), "normalizer_sha256": identity["normalizer_sha256"],
    "action_horizon": contract["prediction_horizon"], "action_dim": contract["action_dim"],
    "cameras": contract["observation_contract"]["cameras"],
    "observation_contract": contract["observation_contract"],
    "action_representation": contract["action_representation"],
  }
  inference = {
    "serving": {
      "host": "127.0.0.1", "port": args.port, "device": "cuda", "autocast": True,
      "warmup_enabled": True, "warmup_iters": 1, "warmup_instruction": INSTRUCTION,
      "warmup_image_shape": [*contract["target_image_size"], 3],
      "warmup_depth_shape": [*contract["target_image_size"], 3],
      "warmup_intrinsic": [500.0, 500.0, 160.0, 120.0],
      "warmup_camera_setup_mode": "single", "warmup_image_mode": "rgb",
      "log_obs_details": False, "record_enabled": False, "record_root_dir": None,
      "profile_enabled": False, "profile_dir": str(output / "profile"),
      "profile_steps": 5, "profile_skip_first": 5, "flops_count_enabled": False,
    },
    "policy": {
      "_target_": "src.policy.egosteer_inference_wrapper.EgoSteerInference",
      "model_config_path": str(config), "checkpoint_path": str(checkpoint),
      "pretrained_vlm_path": str(vlm), "teacher_path": None,
      "use_mixed_precision": True, "tokenizer_padding": "max_length",
      "max_length": contract["max_vlm_tokens"], "default_instruction": None,
      "flow_sampling_steps": 10, "normalizer_path": str(normalizer),
      "use_relative_action": True, "attention_recording": False,
      "camera_views": contract["observation_contract"]["cameras"],
      "compile": {"enabled": False},
    },
    "env_wrapper": {
      "enabled": True, "camera_setup_mode": "single", "image_mode": "rgb",
      "head_camera_name": "head", "chest_camera_name": "chest",
      "image_key": "image", "depth_key": "depth_image",
      "intrinsic_key": "camera_intrinsics", "instruction_key": "instruction",
      "states_key": "states", "prev_action_chunk_key": "action_rtc",
    },
    "deployment_metadata": server_metadata,
  }
  output.mkdir(parents=True)
  (output / "deployment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  (output / "inference.yaml").write_text(json.dumps(inference, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"deployment_manifest={output / 'deployment_manifest.json'} H={contract['prediction_horizon']} sha256={checkpoint_hash}", flush=True)


if __name__ == "__main__":
  main()
