#!/usr/bin/env python3
"""Freeze an auditable EgoSteer deployment contract for USB insertion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_poker_egosteer_deployment import (
  checkpoint_identity,
  parse_model_contract,
  sha256_file,
  source_identity,
)

MODEL_FAMILY = "EgoSteer"
TASK = "usb-insert"
INSTRUCTION = (
  "Grasp the USB plug with the right hand, lift and align it with the "
  "upward-facing socket, insert it until seated, then release it and "
  "withdraw the hand."
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--model-config", type=Path, required=True)
  parser.add_argument("--normalizer", type=Path, required=True)
  parser.add_argument("--model-project", type=Path, required=True)
  parser.add_argument("--model-python", type=Path, required=True)
  parser.add_argument("--pretrained-vlm", type=Path, required=True)
  parser.add_argument("--port", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--camera-views", choices=("head", "head,right_wrist"),
    help="Override unresolved legacy camera_views interpolation.",
  )
  args = parser.parse_args()

  checkpoint = args.checkpoint.expanduser().resolve()
  model_config = args.model_config.expanduser().resolve()
  normalizer = args.normalizer.expanduser().resolve()
  model_project = args.model_project.expanduser().resolve()
  model_python = args.model_python.expanduser().resolve()
  pretrained_vlm = args.pretrained_vlm.expanduser().resolve()
  for path in (
    checkpoint,
    model_config,
    normalizer,
    model_project,
    model_python,
    pretrained_vlm,
  ):
    if not path.exists():
      raise FileNotFoundError(path)
  if not 1 <= args.port <= 65535:
    parser.error("port must be in [1, 65535]")

  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  contract = parse_model_contract(
    model_config, model_python, args.camera_views
  )
  checkpoint_sha256, checkpoint_files = checkpoint_identity(checkpoint)
  model_config_sha256 = sha256_file(model_config)
  normalizer_sha256 = sha256_file(normalizer)
  model_source = source_identity(model_project)
  identity_core = {
    "task": TASK,
    "model_family": MODEL_FAMILY,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "model_config_path": str(model_config),
    "model_config_sha256": model_config_sha256,
    "normalizer_path": str(normalizer),
    "normalizer_sha256": normalizer_sha256,
    "prediction_horizon": contract["prediction_horizon"],
    "action_dim": contract["action_dim"],
    "observation_contract": contract["observation_contract"],
    "action_representation": contract["action_representation"],
    "model_source": model_source,
  }
  deployment_id = hashlib.sha256(
    json.dumps(identity_core, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  manifest = {
    "schema": "usb-egosteer-deployment-v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "deployment_id": deployment_id,
    **identity_core,
    "checkpoint_hash_algorithm": (
      "sha256(canonical JSON of filename,size,file_sha256)"
    ),
    "checkpoint_files": checkpoint_files,
    "model_python": str(model_python),
    "pretrained_vlm_path": str(pretrained_vlm),
    "server": f"ws://127.0.0.1:{args.port}",
  }
  deployment_metadata = {
    "task": TASK,
    "deployment_id": deployment_id,
    "model_family": MODEL_FAMILY,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "model_config_path": str(model_config),
    "model_config_sha256": model_config_sha256,
    "normalizer_path": str(normalizer),
    "normalizer_sha256": normalizer_sha256,
    "action_horizon": contract["prediction_horizon"],
    "action_dim": contract["action_dim"],
    "cameras": contract["observation_contract"]["cameras"],
    "observation_contract": contract["observation_contract"],
    "action_representation": contract["action_representation"],
  }
  inference_config = {
    "serving": {
      "host": "127.0.0.1",
      "port": args.port,
      "device": "cuda",
      "autocast": True,
      "warmup_enabled": True,
      "warmup_iters": 1,
      "warmup_instruction": INSTRUCTION,
      "warmup_image_shape": [*contract["target_image_size"], 3],
      "warmup_depth_shape": [*contract["target_image_size"], 3],
      "warmup_intrinsic": [500.0, 500.0, 160.0, 120.0],
      "warmup_camera_setup_mode": "single",
      "warmup_image_mode": "rgb",
      "log_obs_details": False,
      "record_enabled": False,
      "record_root_dir": None,
      "profile_enabled": False,
      "profile_dir": str(output / "profile"),
      "profile_steps": 5,
      "profile_skip_first": 5,
      "flops_count_enabled": False,
    },
    "policy": {
      "_target_": "src.policy.egosteer_inference_wrapper.EgoSteerInference",
      "model_config_path": str(model_config),
      "checkpoint_path": str(checkpoint),
      "pretrained_vlm_path": str(pretrained_vlm),
      "teacher_path": None,
      "use_mixed_precision": True,
      "tokenizer_padding": "max_length",
      "max_length": contract["max_vlm_tokens"],
      "default_instruction": None,
      "flow_sampling_steps": 10,
      "normalizer_path": str(normalizer),
      "use_relative_action": True,
      "attention_recording": False,
      "camera_views": contract["observation_contract"]["cameras"],
      "compile": {"enabled": False},
    },
    "env_wrapper": {
      "enabled": True,
      "camera_setup_mode": "single",
      "image_mode": "rgb",
      "head_camera_name": "head",
      "chest_camera_name": "chest",
      "image_key": "image",
      "depth_key": "depth_image",
      "intrinsic_key": "camera_intrinsics",
      "instruction_key": "instruction",
      "states_key": "states",
      "prev_action_chunk_key": "action_rtc",
    },
    "deployment_metadata": deployment_metadata,
  }
  manifest_path = output / "deployment_manifest.json"
  config_path = output / "inference.yaml"
  manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  config_path.write_text(
    json.dumps(inference_config, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"[ready] manifest={manifest_path}", flush=True)
  print(f"[ready] inference_config={config_path}", flush=True)
  print(f"[ready] deployment_id={deployment_id}", flush=True)


if __name__ == "__main__":
  main()
