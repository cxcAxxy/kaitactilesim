#!/usr/bin/env python3
"""Freeze the LingBot-VLA-2.0 KaiHand PickPlace checkpoint for evaluation.

This entry point is independent of the pi0.5 and other task deployments.  It
hashes the complete Hugging Face checkpoint and the configuration files that
define LingBot's image, joint, normalization, and action mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lingbot_pickplace_contract import (
  ACTION_DIM,
  ACTION_REPRESENTATION,
  CONTROL_HZ,
  HORIZON,
  INSTRUCTION,
  MODEL_ACTION_DIM,
  MODEL_FAMILY,
  OBSERVATION_CONTRACT,
  RIGHT_JOINT_NAMES,
  SCHEMA,
  TASK,
)
from serve_pickplace_lingbot_vla2_policy import validate_runtime_assets

DEFAULT_LINGBOT_REPO = Path("/cpfs_infra/user/chenxianchi/code/lingbot-vla-v2")


def canonical_sha256(value: Any) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
  before = path.lstat()
  if not path.is_file() or path.is_symlink():
    raise ValueError(f"deployment asset must be a regular file: {path}")
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  after = path.lstat()
  if (before.st_ino, before.st_size, before.st_mtime_ns) != (
    after.st_ino, after.st_size, after.st_mtime_ns
  ):
    raise RuntimeError(f"deployment asset changed while hashing: {path}")
  return digest.hexdigest()


def checkpoint_identity(
  checkpoint: Path, hash_workers: int = 2
) -> tuple[str, list[dict[str, Any]]]:
  """Hash every regular payload file; reject links and unsupported entries."""
  if hash_workers < 1:
    raise ValueError("hash_workers must be positive")
  root = checkpoint.expanduser().resolve(strict=True)
  if not root.is_dir() or root.name != "hf_ckpt":
    raise ValueError("checkpoint must be a LingBot hf_ckpt directory")
  paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
  if not paths or any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in paths):
    raise ValueError("checkpoint contains a symlink or unsupported file entry")
  files = [path for path in paths if path.is_file()]
  if not files:
    raise ValueError(f"checkpoint has no files: {root}")

  def row(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    return {
      "path": path.relative_to(root).as_posix(),
      "size": path.stat().st_size,
      "sha256": digest,
    }

  with ThreadPoolExecutor(max_workers=hash_workers) as executor:
    rows = list(executor.map(row, files))
  if "config.json" not in {row["path"] for row in rows} or (
    "model.safetensors.index.json" not in {row["path"] for row in rows}
  ):
    raise RuntimeError("checkpoint lacks LingBot config or safetensors shard index")
  return canonical_sha256(rows), rows


def _json_object(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return payload


def _training_assets(
  checkpoint: Path, lingbot_repo: Path
) -> tuple[int, Path, Path, Path, Path]:
  match = re.fullmatch(r"global_step_(\d+)", checkpoint.parent.name)
  if match is None or checkpoint.parent.parent.name != "checkpoints":
    raise ValueError("checkpoint must be checkpoints/global_step_N/hf_ckpt")
  step = int(match.group(1))
  training_path = (
    checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
  ).resolve(strict=True)
  try:
    import yaml
  except ImportError as error:
    raise RuntimeError(
      "PyYAML is required to prepare LingBot; use its Python environment"
    ) from error
  training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
  if not isinstance(training, dict):
    raise ValueError("LingBot training configuration must be a mapping")
  data = training.get("data", {})
  train = training.get("train", {})
  if not isinstance(data, dict) or not isinstance(train, dict):
    raise ValueError("LingBot training data/train settings must be mappings")
  if Path(train.get("output_dir", "")).resolve() != training_path.parent:
    raise RuntimeError("checkpoint is not under the configured training output")
  if not isinstance(train.get("max_steps"), int) or step > train["max_steps"]:
    raise RuntimeError("checkpoint step exceeds this training run's max_steps")
  reference = Path(data.get("train_path", "")).expanduser().resolve(strict=True)
  norm_stats = Path(data.get("norm_stats_file", "")).expanduser().resolve(strict=True)
  robot_config = (
    lingbot_repo / "configs/robot_configs/kaihand_usb.yaml"
  ).resolve(strict=True)
  schema = _json_object(reference / "meta/kaihand_schema.json")
  if schema.get("task", {}).get("instruction") != INSTRUCTION:
    raise RuntimeError("training dataset task instruction is not PickPlace")
  info = _json_object(reference / "meta/info.json")
  if info.get("fps") != CONTROL_HZ or info.get("total_episodes", 0) < 1:
    raise RuntimeError("PickPlace training dataset must contain 30 Hz episodes")
  return step, training_path, robot_config, norm_stats, reference


def build_manifest(
  checkpoint: Path, lingbot_repo: Path, *, hash_workers: int = 2
) -> dict[str, Any]:
  """Build and statically validate a complete frozen deployment manifest."""
  checkpoint = checkpoint.expanduser().resolve(strict=True)
  repo = lingbot_repo.expanduser().resolve(strict=True)
  step, training, robot, norm, reference = _training_assets(checkpoint, repo)
  print(f"Hashing LingBot inference payload under {checkpoint} ...", flush=True)
  checkpoint_sha256, checkpoint_files = checkpoint_identity(checkpoint, hash_workers)
  manifest: dict[str, Any] = {
    "schema": SCHEMA,
    "task": TASK,
    "model_family": MODEL_FAMILY,
    "checkpoint_path": str(checkpoint),
    "checkpoint_step": step,
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_hash_algorithm": "sha256(canonical JSON of all path,size,file_sha256 rows)",
    "checkpoint_files": checkpoint_files,
    "checkpoint_total_bytes": sum(row["size"] for row in checkpoint_files),
    "training_config_path": str(training),
    "training_config_sha256": sha256_file(training),
    "robot_config_path": str(robot),
    "robot_config_sha256": sha256_file(robot),
    "norm_stats_path": str(norm),
    "norm_stats_sha256": sha256_file(norm),
    "lingbot_repo_path": str(repo),
    "reference_dataset": str(reference),
    "reference_episode_index": 0,
    "prediction_horizon": HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": list(RIGHT_JOINT_NAMES),
    "instruction": INSTRUCTION,
    "observation_contract": OBSERVATION_CONTRACT,
    "action_representation": ACTION_REPRESENTATION,
  }
  manifest["deployment_id"] = canonical_sha256(manifest)
  validate_runtime_assets(manifest, repo)
  return manifest


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--lingbot-repo", default=DEFAULT_LINGBOT_REPO, type=Path)
  parser.add_argument("--hash-workers", default=2, type=int)
  args = parser.parse_args(argv)
  if args.hash_workers < 1:
    parser.error("--hash-workers must be positive")
  output = args.output_dir.expanduser().resolve()
  if output.exists():
    raise FileExistsError(f"refusing to overwrite deployment directory: {output}")
  manifest = build_manifest(
    args.checkpoint, args.lingbot_repo, hash_workers=args.hash_workers
  )
  output.mkdir(parents=True, exist_ok=False)
  target = output / "deployment_manifest.json"
  target.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(f"deployment_manifest={target}")
  print(f"deployment_id={manifest['deployment_id']}")
  print(f"checkpoint_sha256={manifest['checkpoint_sha256']}")


if __name__ == "__main__":
  main()
