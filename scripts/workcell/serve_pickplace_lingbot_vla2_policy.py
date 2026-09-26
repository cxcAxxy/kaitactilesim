#!/usr/bin/env python3
"""Serve the frozen KaiHand PickPlace LingBot-VLA-2.0 checkpoint.

Run this script with the LingBot Python environment.  ``--validate-only`` checks
the deployment without importing CUDA-dependent model code or loading weights.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from lingbot_pickplace_contract import (  # noqa: E402
  ACTION_DIM,
  ACTION_REPRESENTATION,
  CONTROL_HZ,
  HORIZON,
  INSTRUCTION,
  JOINT_NAMES,
  MODEL_ACTION_DIM,
  MODEL_FAMILY,
  OBSERVATION_CONTRACT,
  SCHEMA,
  TASK,
)


def _canonical_sha256(value: Any) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _require_hash(value: Any, name: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or any(
    char not in "0123456789abcdef" for char in value
  ):
    raise ValueError(f"{name} must be a lowercase SHA-256 digest")
  return value


def _read_json(path: Path) -> dict:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return payload


def _frozen_file(manifest: dict, path_key: str, hash_key: str) -> Path:
  source = Path(manifest[path_key]).expanduser().resolve(strict=True)
  if not source.is_file():
    raise ValueError(f"{path_key} must identify a file: {source}")
  if _sha256_file(source) != _require_hash(manifest[hash_key], hash_key):
    raise RuntimeError(f"frozen {path_key} changed: {source}")
  return source


def load_manifest(path: Path) -> tuple[Path, dict]:
  """Check the model/robot contract before any heavyweight LingBot import."""
  resolved = path.expanduser().resolve(strict=True)
  manifest = _read_json(resolved)
  expected = {
    "schema": SCHEMA,
    "task": TASK,
    "model_family": MODEL_FAMILY,
    "prediction_horizon": HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": list(JOINT_NAMES),
    "instruction": INSTRUCTION,
    "observation_contract": OBSERVATION_CONTRACT,
    "action_representation": ACTION_REPRESENTATION,
    "reference_episode_index": 0,
  }
  mismatches = {
    key: {"expected": value, "actual": manifest.get(key)}
    for key, value in expected.items()
    if manifest.get(key) != value
  }
  if mismatches:
    raise RuntimeError(f"LingBot PickPlace manifest contract mismatch: {mismatches}")
  _require_hash(manifest.get("deployment_id"), "deployment_id")
  _require_hash(manifest.get("checkpoint_sha256"), "checkpoint_sha256")
  for name in (
    "checkpoint_path",
    "training_config_path",
    "training_config_sha256",
    "robot_config_path",
    "robot_config_sha256",
    "norm_stats_path",
    "norm_stats_sha256",
    "lingbot_repo_path",
    "reference_dataset",
  ):
    if not isinstance(manifest.get(name), str) or not manifest[name]:
      raise ValueError(f"manifest is missing {name}")
  for name in (
    "checkpoint_path", "training_config_path", "robot_config_path",
    "norm_stats_path", "lingbot_repo_path", "reference_dataset",
  ):
    if not Path(manifest[name]).is_absolute():
      raise ValueError(f"manifest {name} must be an absolute path")
  identity = {key: value for key, value in manifest.items() if key != "deployment_id"}
  if _canonical_sha256(identity) != manifest["deployment_id"]:
    raise RuntimeError("deployment_id does not match frozen manifest contents")
  return resolved, manifest


def _check_checkpoint_files(manifest: dict, checkpoint: Path, *, verify_hashes: bool) -> None:
  rows = manifest.get("checkpoint_files")
  if not isinstance(rows, list) or not rows:
    raise ValueError("manifest checkpoint_files must be a nonempty list")
  names: set[str] = set()
  normalized: list[dict] = []
  for row in rows:
    if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
      raise ValueError("checkpoint_files rows must contain path, size and sha256")
    name = row["path"]
    relative = Path(name) if isinstance(name, str) else Path("/")
    if (
      not isinstance(name, str)
      or not name
      or relative.is_absolute()
      or ".." in relative.parts
      or relative.as_posix() != name
      or name in names
    ):
      raise ValueError(f"unsafe or duplicate checkpoint file path: {name!r}")
    if type(row["size"]) is not int or row["size"] < 0:
      raise ValueError(f"invalid checkpoint file size: {name}")
    _require_hash(row["sha256"], f"checkpoint file {name} sha256")
    source = checkpoint / relative
    if not source.resolve().is_relative_to(checkpoint):
      raise ValueError(f"checkpoint file escapes its root: {name}")
    if not source.is_file() or source.stat().st_size != row["size"]:
      raise RuntimeError(f"checkpoint payload changed: {source}")
    # Hash configs/tokenizer/indices on every start; the large weight shards
    # are opt-in because reading 23 GB delays every GPU deployment.
    if (verify_hashes or row["size"] <= 16 * 1024 * 1024) and _sha256_file(
      source
    ) != row["sha256"]:
      raise RuntimeError(f"checkpoint SHA-256 mismatch: {source}")
    names.add(name)
    normalized.append(row)
  if rows != sorted(rows, key=lambda item: item["path"]):
    raise ValueError("checkpoint_files must be sorted by path")
  if _canonical_sha256(normalized) != manifest["checkpoint_sha256"]:
    raise RuntimeError("checkpoint file identity differs from checkpoint_sha256")
  actual = {
    path.relative_to(checkpoint).as_posix()
    for path in checkpoint.rglob("*")
    if path.is_file()
  }
  if actual != names:
    raise RuntimeError(
      f"checkpoint file list differs from frozen manifest: "
      f"missing={sorted(names - actual)}, extra={sorted(actual - names)}"
    )
  if not {
    "config.json", "model.safetensors.index.json", "chat_template.jinja"
  }.issubset(names):
    raise RuntimeError(
      "checkpoint lacks config.json, safetensors shard index or chat template"
    )
  shard_index = _read_json(checkpoint / "model.safetensors.index.json")
  shards = set(shard_index.get("weight_map", {}).values())
  if not shards or not all(isinstance(name, str) for name in shards):
    raise RuntimeError("checkpoint safetensors shard index is invalid")
  if not shards.issubset(names):
    raise RuntimeError("checkpoint safetensors shard index references missing files")


def _validate_robot_mapping(robot: dict) -> None:
  expected_states = [
    {"observation.state.arm.position": {
      "origin_keys": "observation.state.right_arm_joint_position"
    }},
    {"observation.state.hand.position": {
      "origin_keys": [{"observation.state.right_hand_joint_position": {
        "start": 0, "end": 12
      }}]
    }},
    {"observation.state.hand_extra_ring.position": {
      "origin_keys": [{"observation.state.right_hand_joint_position": {
        "start": 12, "end": 16
      }}]
    }},
    {"observation.state.hand_extra_pinky.position": {
      "origin_keys": [{"observation.state.right_hand_joint_position": {
        "start": 16, "end": 20
      }}]
    }},
  ]
  expected_actions = [
    {"action.arm.position": {
      "origin_keys": "auxiliary.action.right_arm_joint_target",
      "subtract_state": True,
      "relative_type": "delta",
    }},
    {"action.hand.position": {
      "origin_keys": [{"action.right_hand_joint_position": {
        "start": 0, "end": 12
      }}],
      "subtract_state": False,
    }},
    {"action.hand_extra_ring.position": {
      "origin_keys": [{"action.right_hand_joint_position": {
        "start": 12, "end": 16
      }}],
      "subtract_state": False,
    }},
    {"action.hand_extra_pinky.position": {
      "origin_keys": [{"action.right_hand_joint_position": {
        "start": 16, "end": 20
      }}],
      "subtract_state": False,
    }},
  ]
  expected_images = [
    {"observation.images.camera_top": {
      "origin_keys": "observation.images.head"
    }},
    {"observation.images.camera_wrist_right": {
      "origin_keys": "observation.images.right_wrist"
    }},
  ]
  for key, expected in (
    ("states", expected_states),
    ("actions", expected_actions),
    ("images", expected_images),
  ):
    if robot.get(key) != expected:
      raise RuntimeError(f"current LingBot robot mapping differs at {key}")


def checkpoint_chat_template(checkpoint: Path) -> str:
  """Read the prompt template saved with this exact, content-hashed checkpoint."""
  path = checkpoint / "chat_template.jinja"
  template = path.read_text(encoding="utf-8")
  if not template.strip() or "<|im_start|>" not in template or "<|im_end|>" not in template:
    raise RuntimeError(f"LingBot checkpoint chat template is invalid: {path}")
  return template


def install_checkpoint_chat_template(policy: Any, checkpoint: Path) -> None:
  """Restore the checkpoint prompt format to LingBot's base-path processor.

  LingBot loads its processor from the training base-model directory, whose
  tokenizer has no chat template in this deployment.  The step-3905 hf_ckpt
  *does* save the format used by this model as chat_template.jinja.
  """
  template = checkpoint_chat_template(checkpoint)
  try:
    tokenizers = (
      policy.processor.tokenizer,
      policy.language_tokenizer,
      policy.vla.feature_transform.tokenizer,
    )
  except AttributeError as error:
    raise RuntimeError("LingBot policy tokenizer structure changed") from error
  for tokenizer in tokenizers:
    current = getattr(tokenizer, "chat_template", None)
    if current and current != template:
      raise RuntimeError("LingBot base tokenizer template differs from checkpoint")
    tokenizer.chat_template = template
  rendered = tokenizers[-1].apply_chat_template(
    [{"role": "user", "content": INSTRUCTION}],
    tokenize=False,
    add_generation_prompt=False,
  )
  if not isinstance(rendered, str) or INSTRUCTION not in rendered:
    raise RuntimeError("LingBot checkpoint chat template failed a prompt preflight")


def _validate_training_config(training: dict, manifest: dict, checkpoint_config: dict) -> None:
  model = training.get("model", {})
  data = training.get("data", {})
  train = training.get("train", {})
  required = {
    "model.config_key": (model.get("config_key"), "LingbotVLAV2Config"),
    "data.data_name": (data.get("data_name"), "kaihand_usb"),
    "data.cameras": (data.get("cameras"), ["camera_top", "camera_wrist_right"]),
    "data.img_size": (data.get("img_size"), 256),
    "train.chunk_size": (train.get("chunk_size"), HORIZON),
    "train.action_dim": (train.get("action_dim"), MODEL_ACTION_DIM),
    "train.max_action_dim": (train.get("max_action_dim"), MODEL_ACTION_DIM),
    "train.max_state_dim": (train.get("max_state_dim"), MODEL_ACTION_DIM),
  }
  mismatches = {
    name: {"actual": actual, "expected": expected}
    for name, (actual, expected) in required.items()
    if actual != expected
  }
  if mismatches:
    raise RuntimeError(f"LingBot training contract changed: {mismatches}")
  if Path(data.get("train_path", "")).resolve() != Path(
    manifest["reference_dataset"]
  ).resolve():
    raise RuntimeError("LingBot training dataset differs from reference_dataset")
  if Path(data.get("norm_stats_file", "")).resolve() != Path(
    manifest["norm_stats_path"]
  ).resolve():
    raise RuntimeError("LingBot training normalizer differs from manifest")
  tokenizer = Path(model.get("tokenizer_path", "")).resolve()
  if not (tokenizer / "tokenizer_config.json").is_file():
    raise RuntimeError(f"LingBot tokenizer is missing: {tokenizer}")
  if Path(checkpoint_config.get("tokenizer_path", "")).resolve() != tokenizer:
    raise RuntimeError("checkpoint tokenizer differs from training configuration")
  joints = data.get("joints")
  if not isinstance(joints, list):
    raise RuntimeError("LingBot joint slot layout is missing")
  decoded = [ast.literal_eval(value) if isinstance(value, str) else value for value in joints]
  slots = {key: value for entry in decoded for key, value in entry.items()}
  expected_slots = {
    "arm.position": 14, "end.position": 14, "effector.position": 2,
    "hand_extra_pinky.position": 4, "head.position": 2,
    "base.position": 3, "hand.position": 12,
    "hand_extra_ring.position": 4,
  }
  if slots != expected_slots or sum(slots.values()) != MODEL_ACTION_DIM:
    raise RuntimeError("LingBot 55-slot action layout changed")
  norm_types = data.get("norm_type")
  if not isinstance(norm_types, list):
    raise RuntimeError("LingBot normalization types are missing")
  decoded_norm = [
    ast.literal_eval(value) if isinstance(value, str) else value
    for value in norm_types
  ]
  expected_norm = {
    "arm.position": "meanstd",
    "hand.position": "meanstd",
    "hand_extra_ring.position": "meanstd",
    "hand_extra_pinky.position": "meanstd",
  }
  if {key: value for entry in decoded_norm for key, value in entry.items()} != expected_norm:
    raise RuntimeError("LingBot normalization contract changed")


def validate_runtime_assets(
  manifest: dict, lingbot_repo: Path, *, verify_hashes: bool = False
) -> Path:
  """Validate immutable files and their action/image mappings, without CUDA."""
  repo = lingbot_repo.expanduser().resolve(strict=True)
  if repo != Path(manifest["lingbot_repo_path"]).expanduser().resolve(strict=True):
    raise RuntimeError("--lingbot-repo differs from frozen deployment")
  if not (repo / "deploy/lingbot_vla_v2_policy.py").is_file():
    raise FileNotFoundError(repo / "deploy/lingbot_vla_v2_policy.py")
  if not (repo / "deploy/websocket_policy_server.py").is_file():
    raise FileNotFoundError(repo / "deploy/websocket_policy_server.py")
  checkpoint = Path(manifest["checkpoint_path"]).expanduser().resolve(strict=True)
  if not checkpoint.is_dir() or checkpoint.name != "hf_ckpt":
    raise ValueError("checkpoint_path must be a LingBot hf_ckpt directory")
  _check_checkpoint_files(manifest, checkpoint, verify_hashes=verify_hashes)
  checkpoint_config = _read_json(checkpoint / "config.json")
  expected_config = {
    "action_dim": MODEL_ACTION_DIM,
    "max_action_dim": MODEL_ACTION_DIM,
    "max_state_dim": MODEL_ACTION_DIM,
    "chunk_size": HORIZON,
    "n_action_steps": HORIZON,
    "post_training": True,
  }
  for key, expected in expected_config.items():
    if checkpoint_config.get(key) != expected:
      raise RuntimeError(
        f"checkpoint config {key} must be {expected}, got {checkpoint_config.get(key)}"
      )
  if "LingbotVlaV2Policy" not in checkpoint_config.get("architectures", []):
    raise RuntimeError("checkpoint is not a LingBot VLA v2 policy")
  checkpoint_chat_template(checkpoint)

  training_path = _frozen_file(
    manifest, "training_config_path", "training_config_sha256"
  )
  expected_training = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
  if training_path != expected_training.resolve(strict=True):
    raise RuntimeError("training config is not adjacent to this checkpoint")
  robot_path = _frozen_file(manifest, "robot_config_path", "robot_config_sha256")
  if robot_path != (repo / "configs/robot_configs/kaihand_usb.yaml").resolve(strict=True):
    raise RuntimeError("robot mapping is not the KaiHand USB joint mapping")
  norm_path = _frozen_file(manifest, "norm_stats_path", "norm_stats_sha256")

  try:
    import yaml
  except ImportError as error:
    raise RuntimeError(
      "PyYAML is required for LingBot deployment validation; use its Python environment"
    ) from error
  training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
  robot = yaml.safe_load(robot_path.read_text(encoding="utf-8"))
  if not isinstance(training, dict) or not isinstance(robot, dict):
    raise RuntimeError("LingBot YAML configuration must contain mappings")
  _validate_training_config(training, manifest, checkpoint_config)
  _validate_robot_mapping(robot)

  norm = _read_json(norm_path).get("norm_stats")
  if not isinstance(norm, dict):
    raise RuntimeError("LingBot normalizer lacks norm_stats")
  for suffix, dim in (
    ("arm.position", 7),
    ("hand.position", 12),
    ("hand_extra_ring.position", 4),
    ("hand_extra_pinky.position", 4),
  ):
    for prefix in ("action", "observation.state"):
      key = f"{prefix}.{suffix}"
      if key not in norm or len(norm[key].get("mean", [])) != dim:
        raise RuntimeError(f"LingBot normalization statistics differ for {key}")
  reference = Path(manifest["reference_dataset"]).expanduser().resolve(strict=True)
  if not (reference / "meta/info.json").is_file():
    raise RuntimeError(f"reference_dataset is not a LeRobot dataset: {reference}")
  return checkpoint


def deployment_metadata(manifest_path: Path, manifest: dict) -> dict:
  """Advertise every field the evaluator must compare against its manifest."""
  return {
    "schema": manifest["schema"],
    "evaluation_task": manifest["task"],
    "model_family": manifest["model_family"],
    "deployment_manifest": str(manifest_path),
    "deployment_id": manifest["deployment_id"],
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "training_config_path": manifest["training_config_path"],
    "training_config_sha256": manifest["training_config_sha256"],
    "robot_config_path": manifest["robot_config_path"],
    "robot_config_sha256": manifest["robot_config_sha256"],
    "norm_stats_path": manifest["norm_stats_path"],
    "norm_stats_sha256": manifest["norm_stats_sha256"],
    "prediction_horizon": manifest["prediction_horizon"],
    "action_horizon": manifest["prediction_horizon"],
    "model_action_dim": manifest["model_action_dim"],
    "action_dim": manifest["action_dim"],
    "control_hz": manifest["control_hz"],
    "joint_names": manifest["joint_names"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
    "instruction": manifest["instruction"],
    "reference_dataset": manifest["reference_dataset"],
  }


def pin_qwen_base_model(manifest: dict, *, set_env: bool = True) -> Path:
  """Prevent LingBot's QWEN3VL_PATH override from changing the frozen model."""
  config = _read_json(Path(manifest["checkpoint_path"]) / "config.json")
  tokenizer_value = config.get("tokenizer_path")
  if not isinstance(tokenizer_value, str) or not tokenizer_value:
    raise RuntimeError("checkpoint config has no tokenizer_path")
  tokenizer = Path(tokenizer_value).expanduser().resolve(strict=True)
  override = os.environ.get("QWEN3VL_PATH")
  if override and Path(override).expanduser().resolve() != tokenizer:
    raise RuntimeError(
      f"QWEN3VL_PATH points to {override}, but frozen checkpoint requires {tokenizer}"
    )
  # LingBot's load_vla reads this variable for both AutoConfig and processor.
  # An absolute, validated value also remains correct after the cwd changes.
  if set_env:
    os.environ["QWEN3VL_PATH"] = str(tokenizer)
  return tokenizer


def serve(
  manifest_path: Path,
  manifest: dict,
  lingbot_repo: Path,
  *,
  host: str,
  port: int,
  use_compile: bool = False,
) -> None:
  """Load the actual policy only after static validation and a CUDA preflight."""
  pin_qwen_base_model(manifest)
  try:
    import torch
  except ImportError as error:
    raise RuntimeError("PyTorch is required; use the LingBot Python environment") from error
  if not torch.cuda.is_available():
    raise RuntimeError(
      "LingBot-VLA-2.0 requires a CUDA GPU; this environment has none. "
      "Use --validate-only for a CPU-only deployment check."
    )

  repo = lingbot_repo.resolve(strict=True)
  os.chdir(repo)  # LingBot reset('kaihand_usb') resolves robot YAML relative to cwd.
  if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
  from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
  from deploy.websocket_policy_server import WebsocketPolicyServer

  policy = LingbotVLAv2Server(
    path_to_pi_model=manifest["checkpoint_path"],
    robot_norm_path=manifest["norm_stats_path"],
    use_length=HORIZON,
    chunk_ret=True,
    use_bf16=True,
    use_fp32=False,
    use_compile=use_compile,
  )
  policy.reset("kaihand_usb")
  install_checkpoint_chat_template(policy, Path(manifest["checkpoint_path"]))
  logging.info(
    "Serving %s deployment %s (H=%s, port=%s)",
    MODEL_FAMILY,
    manifest["deployment_id"],
    HORIZON,
    port,
  )
  WebsocketPolicyServer(
    policy=policy,
    host=host,
    port=port,
    metadata=deployment_metadata(manifest_path, manifest),
  ).serve_forever()


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--lingbot-repo", type=Path)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8006)
  parser.add_argument(
    "--verify-hashes", action="store_true",
    help="Re-read and hash all checkpoint files (about 23 GB for step 3905)",
  )
  parser.add_argument(
    "--validate-only", action="store_true",
    help="Check the frozen deployment without importing/loading the model",
  )
  parser.add_argument(
    "--use-compile", action="store_true",
    help="Enable torch.compile for LingBot inference (off by default)",
  )
  args = parser.parse_args(argv)
  if not 1 <= args.port <= 65535:
    parser.error("port must be in 1..65535")
  manifest_path, manifest = load_manifest(args.deployment_manifest)
  repo = args.lingbot_repo or Path(manifest["lingbot_repo_path"])
  checkpoint = validate_runtime_assets(
    manifest, repo, verify_hashes=args.verify_hashes
  )
  pin_qwen_base_model(manifest, set_env=False)
  logging.info("Validated LingBot checkpoint and mapping: %s", checkpoint)
  if args.validate_only:
    print(
      f"validated_deployment={manifest['deployment_id']} "
      f"checkpoint={checkpoint} horizon={HORIZON}",
      flush=True,
    )
    return
  serve(
    manifest_path, manifest, repo,
    host=args.host, port=args.port, use_compile=args.use_compile,
  )


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, force=True)
  main()
