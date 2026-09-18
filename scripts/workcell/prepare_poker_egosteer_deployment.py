"""Freeze one Poker EgoSteer checkpoint into an auditable serving contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MODEL_FAMILY = "EgoSteer"
TASK = "poker-draw"
SUPPORTED_CAMERAS = (("head",), ("head", "right_wrist"))


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while block := source.read(16 * 1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def checkpoint_identity(checkpoint: Path) -> tuple[str, list[dict]]:
  files = sorted(path for path in checkpoint.iterdir() if path.is_file())
  metadata = checkpoint / ".metadata"
  shards = sorted(checkpoint.glob("__*_*.distcp"))
  if metadata not in files or not shards:
    raise RuntimeError(
      f"incomplete DCP checkpoint (need .metadata and distcp shards): {checkpoint}"
    )
  unexpected = [path.name for path in files if path != metadata and path not in shards]
  if unexpected:
    raise RuntimeError(f"unexpected files in checkpoint: {unexpected}")
  rows = []
  for path in files:
    before = path.stat()
    print(f"[hash] {path} ({before.st_size} bytes)", flush=True)
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
      raise RuntimeError(f"checkpoint changed while hashing: {path}")
    rows.append(
      {
        "name": path.name,
        "size": before.st_size,
        "sha256": digest,
      }
    )
  canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(canonical).hexdigest(), rows


def source_identity(project: Path) -> dict:
  files = sorted((project / "src").rglob("*.py"))
  if not files:
    raise RuntimeError(f"model project has no Python sources: {project}")
  rows = []
  for path in files:
    rows.append(
      {
        "path": str(path.relative_to(project)),
        "sha256": sha256_file(path),
      }
    )
  canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
  head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=project,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  status = subprocess.run(
    ["git", "status", "--short"],
    cwd=project,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.splitlines()
  return {
    "project": str(project),
    "git_head": head,
    "git_status": status,
    "python_source_sha256": hashlib.sha256(canonical).hexdigest(),
    "python_file_count": len(rows),
  }


def parse_model_contract(
  config_path: Path,
  model_python: Path,
  camera_views_override: str | None = None,
) -> dict:
  query = """
import json, sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1])
camera_views = OmegaConf.select(c, "dataset.vla_dataset.camera_views", default="head")
if isinstance(camera_views, str):
  camera_views = [item.strip() for item in camera_views.split(",") if item.strip()]
else:
  camera_views = list(camera_views)
print(json.dumps({
  "prediction_horizon": int(c.data.shape_meta.action.horizon),
  "action_dim": int(c.dims.action_dim),
  "state_dim": int(c.dims.state_dim),
  "target_image_size": list(c.data.target_image_size),
  "max_vlm_tokens": int(c.data.max_vlm_tokens),
  "image_history": int(c.data.shape_meta.obs.rgb.horizon),
  "image_stride": int(c.data.shape_meta.obs.rgb.stride),
  "state_history": int(c.data.shape_meta.obs.state.horizon),
  "state_stride": int(c.data.shape_meta.obs.state.stride),
  "history_pad_mode": str(c.data.shape_meta.history_pad_mode),
  "use_relative_action": bool(c.dataset.vla_dataset.use_relative_action),
  "load_tactile": bool(c.dataset.vla_dataset.load_tactile),
  "camera_views": camera_views,
}))
"""
  completed = subprocess.run(
    [str(model_python), "-c", query, str(config_path)],
    check=True,
    capture_output=True,
    text=True,
  )
  config = json.loads(completed.stdout)
  action_dim = config["action_dim"]
  state_dim = config["state_dim"]
  prediction_horizon = config["prediction_horizon"]
  if action_dim != 48 or state_dim != 48:
    raise RuntimeError(
      f"Poker EgoSteer requires action_dim=state_dim=48, got {action_dim}/{state_dim}"
    )
  if prediction_horizon <= 0:
    raise RuntimeError("prediction horizon must be positive")
  if config["use_relative_action"] is not True:
    raise RuntimeError("this adapter requires relative-action training")
  if config["load_tactile"] is not False:
    raise RuntimeError("this deployment contract expects no model tactile input")
  declared_cameras = (
    camera_views_override.split(",")
    if camera_views_override is not None
    else config["camera_views"]
  )
  cameras = tuple(str(name).strip() for name in declared_cameras if str(name).strip())
  if cameras not in SUPPORTED_CAMERAS:
    raise RuntimeError(
      "KaiHand EgoSteer deployment supports cameras ['head'] or "
      f"['head', 'right_wrist'], got {list(cameras)}"
    )
  return {
    "prediction_horizon": prediction_horizon,
    "action_dim": action_dim,
    "state_dim": state_dim,
    "target_image_size": config["target_image_size"],
    "max_vlm_tokens": config["max_vlm_tokens"],
    "observation_contract": {
      "cameras": list(cameras),
      "rgb": True,
      "depth": False,
      "tactile_sent_to_model": False,
      "image_history": config["image_history"],
      "image_stride": config["image_stride"],
      "state_history": config["state_history"],
      "state_stride": config["state_stride"],
      "state_dim": state_dim,
      "history_pad_mode": config["history_pad_mode"],
      "camera_intrinsics": True,
      "language_instruction": True,
    },
    "action_representation": {
      "type": "relative",
      "shape": [prediction_horizon, action_dim],
      "wrist": "relative SE(3) as xyz+rot6d for left and right wrists",
      "fingertips": "additive xyz delta in each canonical wrist frame",
      "adapter": "kaihand_tactile_env.shared.egosteer_adapter",
    },
  }


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
    "--camera-views",
    choices=("head", "head,right_wrist"),
    help=(
      "Override unresolved legacy camera_views interpolation. Omit to read the "
      "checkpoint training config; old configs fall back to head."
    ),
  )
  args = parser.parse_args()

  checkpoint = args.checkpoint.expanduser().resolve()
  model_config = args.model_config.expanduser().resolve()
  normalizer = args.normalizer.expanduser().resolve()
  model_project = args.model_project.expanduser().resolve()
  model_python = args.model_python.expanduser().resolve()
  pretrained_vlm = args.pretrained_vlm.expanduser().resolve()
  for path in (checkpoint, model_config, normalizer, model_project, model_python, pretrained_vlm):
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
    "schema": "poker-egosteer-deployment-v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "deployment_id": deployment_id,
    **identity_core,
    "checkpoint_hash_algorithm": "sha256(canonical JSON of filename,size,file_sha256)",
    "checkpoint_files": checkpoint_files,
    "model_python": str(model_python),
    "pretrained_vlm_path": str(pretrained_vlm),
    "server": f"ws://127.0.0.1:{args.port}",
  }
  deployment_metadata = {
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
      "warmup_instruction": (
        "Slide the face-down card toward the table edge, pinch it between the "
        "fingers and thumb, lift it, and turn its face toward the robot to look at it."
      ),
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
