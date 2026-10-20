#!/usr/bin/env python3
"""Freeze an audited PickPlace EgoTouch external-simulation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK = "pick-place"
MODEL_FAMILY = "EgoTouch"
CHECKPOINT_SCHEMA = "egotouch-external-sim-candidate-v1"
DEPLOYMENT_SCHEMA = "pickplace_egotouch_deployment_v1"
HORIZON = 50
ACTION_DIM = 108
CONTROL_HZ = 30
VIEW_SLOTS = ["head", "left_wrist", "right_wrist"]
VIEW_MASK = [True, False, True]
CAMERAS = ["head", "right_wrist"]
ACTION_REPRESENTATION = "wrist_fingertip_pose9"
OBSERVATION_CONTRACT = {
  "cameras": CAMERAS,
  "image_shape_hwc": [240, 320, 3],
  "view_slots": VIEW_SLOTS,
  "view_mask": VIEW_MASK,
  "tactile_sent_to_model": False,
}
MODEL_SOURCE_PATHS = (
  "inference/embodiment_policy.py",
  "inference/receding_horizon.py",
  "tools/train_usb_tict_ddp.py",
  "training/FlowMatchingDataloader.py",
  "training/FlowMatchingModel.py",
  "utils/tict_schema.py",
  "utils/vision_schema.py",
)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def canonical_hash(value: Any) -> str:
  payload = json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def inspect_checkpoint(model_python: Path, checkpoint: Path) -> dict[str, Any]:
  code = r"""
import json
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
  raise TypeError("checkpoint payload must be a mapping")
cfg = payload.get("cfg")
model = payload.get("model")
if not isinstance(cfg, dict) or not isinstance(model, dict):
  raise TypeError("checkpoint cfg/model must be mappings")
required_tensors = ("action_proj.weight", "head_v.weight")
if any(name not in model or not hasattr(model[name], "shape") for name in required_tensors):
  raise KeyError("checkpoint is missing action projection tensors")
result = {
  "schema_version": payload.get("schema_version"),
  "epoch": payload.get("epoch"),
  "global_step": payload.get("global_step"),
  "variant": payload.get("variant"),
  "binding_sha256": payload.get("binding_sha256"),
  "model_weights": payload.get("model_weights"),
  "model_tensor_count": len(model),
  "action_proj_shape": list(model["action_proj.weight"].shape),
  "head_v_shape": list(model["head_v.weight"].shape),
  "cfg": {key: cfg.get(key) for key in (
    "pred_horizon", "use_tactile_input", "action_mode", "frame_mode",
    "centric_mode", "single_hand", "image_size", "num_inference_steps",
  )},
}
print(json.dumps(result, separators=(",", ":")))
"""
  result = subprocess.run(
    [str(model_python), "-c", code, str(checkpoint)],
    check=False,
    capture_output=True,
    text=True,
    timeout=180,
  )
  if result.returncode:
    raise RuntimeError(
      "EgoTouch Python could not inspect checkpoint: "
      f"{result.stderr.strip() or result.stdout.strip()}"
    )
  try:
    payload = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError(
      f"checkpoint inspector returned invalid JSON: {result.stdout!r}"
    ) from error
  if not isinstance(payload, dict):
    raise RuntimeError("checkpoint inspector returned a non-object")
  return payload


def validate_checkpoint(metadata: dict[str, Any], checkpoint: Path) -> None:
  match = re.fullmatch(r"epoch_(\d{4})\.pt", checkpoint.name)
  if match is None:
    raise ValueError("checkpoint filename must be epoch_NNNN.pt")
  expected = {
    "schema_version": CHECKPOINT_SCHEMA,
    "epoch": int(match.group(1)),
    "variant": "no_tactile",
    "model_weights": "ema",
    "action_proj_shape": [384, ACTION_DIM],
    "head_v_shape": [ACTION_DIM, 384],
  }
  mismatch = {
    key: (value, metadata.get(key))
    for key, value in expected.items()
    if metadata.get(key) != value
  }
  if mismatch:
    raise ValueError(f"checkpoint identity mismatch: {mismatch}")
  if type(metadata["epoch"]) is not int or metadata["epoch"] <= 0:
    raise ValueError("checkpoint epoch must be a positive integer")
  if not isinstance(metadata.get("global_step"), int) or metadata["global_step"] <= 0:
    raise ValueError("checkpoint global_step must be a positive integer")
  if not isinstance(metadata.get("model_tensor_count"), int) or metadata["model_tensor_count"] < 1:
    raise ValueError("checkpoint has no model tensors")
  binding_digest = metadata.get("binding_sha256")
  if not isinstance(binding_digest, str) or re.fullmatch(r"[0-9a-f]{64}", binding_digest) is None:
    raise ValueError("checkpoint binding_sha256 is invalid")
  cfg = metadata.get("cfg")
  expected_cfg = {
    "pred_horizon": HORIZON,
    "use_tactile_input": False,
    "action_mode": "absolute",
    "frame_mode": "camera_frame",
    "centric_mode": "ego_centric",
    "single_hand": False,
    "image_size": [240, 320],
  }
  if not isinstance(cfg, dict):
    raise ValueError("checkpoint cfg is missing")
  cfg_mismatch = {
    key: (value, cfg.get(key))
    for key, value in expected_cfg.items()
    if cfg.get(key) != value
  }
  if cfg_mismatch:
    raise ValueError(f"checkpoint cfg mismatch: {cfg_mismatch}")
  if not isinstance(cfg.get("num_inference_steps"), int) or cfg["num_inference_steps"] <= 0:
    raise ValueError("checkpoint num_inference_steps must be positive")


def validate_binding(binding: Any, checkpoint_metadata: dict[str, Any]) -> None:
  if not isinstance(binding, dict):
    raise ValueError("binding must contain a JSON object")
  expected = {
    "schema_version": "egotouch-usb-tict-binding-v1",
    "pred_horizon": HORIZON,
    "img_name": "rgb.png",
    "view_names": VIEW_SLOTS,
    "view_image_names": {
      "head": "rgb.png",
      "left_wrist": "rgb_left_wrist.png",
      "right_wrist": "rgb_right_wrist.png",
    },
  }
  mismatch = {
    key: (value, binding.get(key))
    for key, value in expected.items()
    if binding.get(key) != value
  }
  if mismatch:
    raise ValueError(f"binding contract mismatch: {mismatch}")
  digest = canonical_hash(binding)
  if digest != checkpoint_metadata["binding_sha256"]:
    raise ValueError(
      "checkpoint/binding SHA-256 mismatch: "
      f"{checkpoint_metadata['binding_sha256']} != {digest}"
    )
  stats = binding.get("stats")
  if not isinstance(stats, dict):
    raise ValueError("binding is missing normalization statistics")
  for name in ("pos", "finger_pos", "tactile"):
    group = stats.get(name)
    if not isinstance(group, dict):
      raise ValueError(f"binding stats.{name} is missing")
    mean, std = group.get("mean"), group.get("std")
    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
      raise ValueError(f"binding stats.{name} must contain three moments")
    values = [*mean, *std]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
      raise ValueError(f"binding stats.{name} contains non-finite values")
    if any(value <= 0 for value in std):
      raise ValueError(f"binding stats.{name}.std must be positive")
  selectors = binding.get("selector_records")
  if not isinstance(selectors, dict) or not selectors:
    raise ValueError("binding selector_records is empty")
  frame_count = 0
  for session, records in selectors.items():
    if not isinstance(records, dict) or not records:
      raise ValueError(f"binding selector_records[{session!r}] is empty")
    for frame, record in records.items():
      images = record.get("images") if isinstance(record, dict) else None
      if not isinstance(images, dict) or set(images) != set(VIEW_SLOTS):
        raise ValueError(f"{session}/{frame} has invalid visual slots")
      actual_mask = [images[name].get("valid") for name in VIEW_SLOTS]
      if actual_mask != VIEW_MASK:
        raise ValueError(
          f"{session}/{frame} visual mask {actual_mask!r} != {VIEW_MASK!r}"
        )
      frame_count += 1
  if frame_count < HORIZON:
    raise ValueError("binding has fewer frames than one prediction horizon")


def source_identity(project: Path) -> tuple[str, list[dict[str, Any]]]:
  files = []
  for relative in MODEL_SOURCE_PATHS:
    path = project / relative
    if not path.is_file():
      raise FileNotFoundError(f"required EgoTouch source is missing: {path}")
    files.append({
      "path": relative,
      "bytes": path.stat().st_size,
      "sha256": sha256_file(path),
    })
  return canonical_hash(files), files


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--binding", required=True, type=Path)
  parser.add_argument("--model-project", required=True, type=Path)
  parser.add_argument("--model-python", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args(argv)

  checkpoint = args.checkpoint.expanduser().resolve(strict=True)
  binding_path = args.binding.expanduser().resolve(strict=True)
  model_project = args.model_project.expanduser().resolve(strict=True)
  model_python = args.model_python.expanduser().resolve(strict=True)
  output = args.output_dir.expanduser().resolve()
  if not checkpoint.is_file() or not binding_path.is_file():
    raise ValueError("checkpoint and binding must be regular files")
  if not model_project.is_dir() or not model_python.is_file():
    raise ValueError("model project/Python paths have the wrong type")
  if output.exists():
    raise FileExistsError(f"output must be a new directory: {output}")

  checkpoint_metadata = inspect_checkpoint(model_python, checkpoint)
  validate_checkpoint(checkpoint_metadata, checkpoint)
  binding = json.loads(binding_path.read_text(encoding="utf-8"))
  validate_binding(binding, checkpoint_metadata)
  source_sha256, source_files = source_identity(model_project)
  checkpoint_sha256 = sha256_file(checkpoint)
  binding_file_sha256 = sha256_file(binding_path)
  model_python_sha256 = sha256_file(model_python)
  identity = {
    "task": TASK,
    "model_family": MODEL_FAMILY,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_schema": checkpoint_metadata["schema_version"],
    "checkpoint_epoch": checkpoint_metadata["epoch"],
    "checkpoint_global_step": checkpoint_metadata["global_step"],
    "model_weights": checkpoint_metadata["model_weights"],
    "binding_path": str(binding_path),
    "binding_sha256": checkpoint_metadata["binding_sha256"],
    "binding_file_sha256": binding_file_sha256,
    "prediction_horizon": HORIZON,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "observation_contract": OBSERVATION_CONTRACT,
    "action_representation": ACTION_REPRESENTATION,
    "flow_sampling_steps": checkpoint_metadata["cfg"]["num_inference_steps"],
    "model_project_path": str(model_project),
    "model_source_sha256": source_sha256,
    "model_python": str(model_python),
    "model_python_sha256": model_python_sha256,
  }
  manifest = {
    "schema": DEPLOYMENT_SCHEMA,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "deployment_id": canonical_hash(identity),
    **identity,
    "model_source_files": source_files,
    "checkpoint_hash_algorithm": "sha256(file bytes)",
    "binding_hash_algorithm": {
      "binding_sha256": "sha256(canonical JSON)",
      "binding_file_sha256": "sha256(file bytes)",
    },
  }
  output.mkdir(parents=True, exist_ok=False)
  destination = output / "deployment_manifest.json"
  destination.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(
    f"deployment_manifest={destination} epoch={manifest['checkpoint_epoch']} "
    f"H={HORIZON} D={ACTION_DIM} sha256={checkpoint_sha256}",
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
