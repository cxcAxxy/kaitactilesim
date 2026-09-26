#!/usr/bin/env python3
"""Convert successful KaiHand HDF5 episodes to EgoSteer WebDataset shards.

The source directory is opened read-only and is never modified.  The output is
built in a sibling staging directory and renamed into place only after every
episode and manifest has been written successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import model_fingerprint
from PIL import Image
from unified_lerobot_collection import discover_unified_artifacts

SOURCE_SCHEMA_VERSION = "kaihand_tactile_episode_v1"
SOURCE_MANIFEST_VERSION = "kaihand_hdf5_source_snapshot_v1"
EGOSTEER_MANIFEST_VERSION = "egosteer_webdataset_head_v1"
EGOSTEER_FPS = 30
BASE_LOWDIM_DIM = 96
CAMERA_BLOCK_DIM = 20
LOWDIM_DIM = 116
CANONICAL_CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
EPISODE_NAME_PATTERN = re.compile(r"^episode_(\d+)_.*\.h5$")
CV_FROM_MUJOCO_CAMERA = np.diag((1.0, -1.0, -1.0, 1.0))

# Columns are EgoSteer/MANO canonical wrist axes expressed in the KaiHand
# base-link site frame.  The two matrices account for the mirrored hands.
# wrist translation remains the base-link site origin.
SITE_FROM_EGOSTEER_WRIST = {
  "left": np.array(
    (
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 0.0),
      (-1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
  ),
  "right": np.array(
    (
      (0.0, 0.0, 1.0),
      (0.0, -1.0, 0.0),
      (1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
  ),
}


def canonical_camera_views(cameras: Sequence[str]) -> tuple[str, ...]:
  """Validate and canonicalize the camera subset used by EgoSteer."""

  declared = tuple(str(camera) for camera in cameras)
  if not declared or "head" not in declared:
    raise ValueError("EgoSteer camera views must include head")
  if len(set(declared)) != len(declared):
    raise ValueError(f"duplicate EgoSteer camera views: {declared!r}")
  unsupported = set(declared) - set(CANONICAL_CAMERA_ORDER)
  if unsupported:
    raise ValueError(f"unsupported EgoSteer camera views: {sorted(unsupported)!r}")
  canonical = tuple(camera for camera in CANONICAL_CAMERA_ORDER if camera in declared)
  if canonical != declared:
    raise ValueError(
      f"camera views must follow canonical order {CANONICAL_CAMERA_ORDER!r}"
    )
  return canonical


def camera_image_member(camera: str) -> str:
  if camera not in CANONICAL_CAMERA_ORDER:
    raise ValueError(f"unsupported EgoSteer camera: {camera!r}")
  return "image.jpg" if camera == "head" else f"{camera}_image.jpg"


def camera_lowdim_dim(cameras: Sequence[str]) -> int:
  return BASE_LOWDIM_DIM + CAMERA_BLOCK_DIM * len(canonical_camera_views(cameras))


def camera_member_order(cameras: Sequence[str]) -> tuple[str, ...]:
  views = canonical_camera_views(cameras)
  return tuple(camera_image_member(camera) for camera in views) + (
    "lowdim.npy",
    "meta.json",
  )


def _tree_content_signature(root: Path) -> tuple[tuple[str, int, str], ...]:
  """Record every output path, size, and SHA-256 before publication."""

  return tuple(
    sorted(
      (path.relative_to(root).as_posix(), path.stat().st_size, _sha256_file(path))
      for path in root.rglob("*")
      if path.is_file() and not path.is_symlink()
    )
  )


def publish_validated_staging(staging_dir: Path, output_dir: Path) -> None:
  """Atomically publish a validated tree, including across filesystems.

  Same-filesystem publication is a single rename.  For CPFS-to-NAS publication,
  copy into a hidden NAS directory, compare complete path/size/content hashes,
  then rename that hidden directory into place.
  """

  if os.path.lexists(output_dir):
    raise FileExistsError(f"output appeared during conversion: {output_dir}")
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  if staging_dir.stat().st_dev == output_dir.parent.stat().st_dev:
    os.rename(staging_dir, output_dir)
    return

  upload_dir = output_dir.with_name(
    f".{output_dir.name}.upload-{os.getpid()}-{uuid.uuid4().hex}"
  )
  try:
    source_signature = _tree_content_signature(staging_dir)
    total_files = len(source_signature)
    copied_files = 0

    def copy_with_progress(source: str, destination: str) -> str:
      nonlocal copied_files
      result = shutil.copy2(source, destination)
      copied_files += 1
      if copied_files == total_files or copied_files % 20 == 0:
        print(
          f"[publish {copied_files}/{total_files}] {output_dir}",
          flush=True,
        )
      return result

    print(
      f"publishing validated dataset: files={total_files} "
      f"from={staging_dir} to={output_dir}",
      flush=True,
    )
    shutil.copytree(staging_dir, upload_dir, copy_function=copy_with_progress)
    if _tree_content_signature(upload_dir) != source_signature:
      raise OSError("cross-filesystem publication content verification failed")
    if os.path.lexists(output_dir):
      raise FileExistsError(f"output appeared during publication: {output_dir}")
    os.rename(upload_dir, output_dir)
  except BaseException:
    if upload_dir.exists():
      shutil.rmtree(upload_dir)
    raise
  shutil.rmtree(staging_dir)


@dataclass(frozen=True)
class SourceEpisode:
  """Immutable identity and FK requirements for one source episode."""

  path: Path
  sidecar_path: Path
  episode_index: int
  hdf5_sha256: str
  sidecar_sha256: str
  size_bytes: int
  mtime_ns: int
  model_path: Path
  model_sha256: str


@dataclass(frozen=True)
class EpisodeConversion:
  """Auditable timing and sample counts from one converted episode."""

  episode_index: int
  split: str
  source_frames: int
  strict_prefix_frames: int
  exported_samples: int
  off_grid_frames_discarded: int
  maximum_grid_error_seconds: float
  image_width: int
  image_height: int


@dataclass(frozen=True)
class ShardSummary:
  """Digest and contents of one completed WebDataset shard."""

  path: str
  sha256: str
  size_bytes: int
  episodes: int
  episode_indices: tuple[int, ...]
  samples: int


@dataclass
class Kinematics:
  """Reusable MuJoCo FK state for one recorded model digest."""

  model: mujoco.MjModel
  data: mujoco.MjData
  wrist_site_ids: dict[str, int]
  fingertip_site_ids: dict[str, tuple[int, ...]]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Convert successful KaiHand HDF5 episodes to strict head-only "
      "EgoSteer WebDataset shards. Existing paths are never overwritten."
    )
  )
  parser.add_argument("--input-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--instruction",
    default="Put the red cylinder into the blue box.",
  )
  parser.add_argument("--dataset-name", default="kaihand_pick_place")
  parser.add_argument("--val-fraction", type=float, default=0.1)
  parser.add_argument("--split-seed", type=int, default=0)
  parser.add_argument("--episodes-per-shard", type=int, default=20)
  parser.add_argument("--jpeg-quality", type=int, default=95)
  parser.add_argument(
    "--model-path",
    type=Path,
    help=(
      "Optional relocated MJCF. Its SHA-256 must match every episode's "
      "recorded model_sha256."
    ),
  )
  args = parser.parse_args(argv)
  if not args.input_dir.is_dir():
    parser.error(f"--input-dir is not a directory: {args.input_dir}")
  if args.output_dir.exists():
    parser.error(f"--output-dir already exists: {args.output_dir}")
  if args.input_dir.expanduser().resolve() == args.output_dir.expanduser().resolve():
    parser.error("--output-dir must differ from --input-dir")
  if not args.instruction.strip():
    parser.error("--instruction cannot be empty")
  if not args.dataset_name.strip():
    parser.error("--dataset-name cannot be empty")
  if not 0.0 <= args.val_fraction < 1.0:
    parser.error("--val-fraction must be in [0, 1)")
  if args.episodes_per_shard <= 0:
    parser.error("--episodes-per-shard must be positive")
  if not 1 <= args.jpeg_quality <= 100:
    parser.error("--jpeg-quality must be in [1, 100]")
  if args.model_path is not None and not args.model_path.is_file():
    parser.error(f"--model-path is not a file: {args.model_path}")
  return args


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _json_attr(file: h5py.File, name: str) -> dict[str, Any]:
  raw = file.attrs.get(name)
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8")
  if not isinstance(raw, str):
    raise ValueError(f"{file.filename}: missing string attribute {name}")
  try:
    value = json.loads(raw)
  except json.JSONDecodeError as error:
    raise ValueError(f"{file.filename}: invalid JSON attribute {name}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{file.filename}: attribute {name} must contain an object")
  return value


def _parse_filename_episode_index(path: Path) -> int:
  match = EPISODE_NAME_PATTERN.fullmatch(path.name)
  if match is None:
    raise ValueError(f"unsupported episode filename: {path.name}")
  return int(match.group(1))


def _discover_sources(
  input_dir: Path,
  model_override: Path | None,
) -> tuple[SourceEpisode, ...]:
  discovered = discover_unified_artifacts(
    input_dir, task="pick-place", expected_episodes=0, limit=None
  )
  if discovered is None:
    paths = sorted(input_dir.glob("episode_*.h5"))
    outer_indices = {}
  else:
    artifacts, _ = discovered
    paths = [artifact.hdf5_path for artifact in artifacts]
    outer_indices = {
      artifact.hdf5_path: artifact.episode_index for artifact in artifacts
    }
  if not paths:
    raise FileNotFoundError(f"no episode_*.h5 files found in {input_dir}")

  model_hash_cache: dict[Path, str] = {}
  model_source_cache: dict[Path, str] = {}
  sources: list[SourceEpisode] = []
  seen_indices: set[int] = set()
  for number, path in enumerate(paths, start=1):
    path = path.resolve()
    print(f"fingerprint [{number}/{len(paths)}] {path.name}", flush=True)
    before = path.stat()
    hdf5_sha256 = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
      raise RuntimeError(f"source changed while hashing: {path}")

    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
      raise FileNotFoundError(f"missing source sidecar: {sidecar_path}")
    sidecar_sha256 = _sha256_file(sidecar_path)
    try:
      sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
      raise ValueError(f"invalid source sidecar JSON: {sidecar_path}") from error
    if not isinstance(sidecar, dict):
      raise ValueError(f"source sidecar must contain an object: {sidecar_path}")
    if sidecar.get("episode") != path.name:
      raise ValueError(f"source sidecar episode mismatch: {sidecar_path}")
    if sidecar.get("sha256") != hdf5_sha256:
      raise ValueError(f"source sidecar SHA-256 mismatch: {sidecar_path}")

    filename_index = _parse_filename_episode_index(path)
    with h5py.File(path, "r") as file:
      if str(file.attrs.get("schema_version", "")) != SOURCE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported source schema_version")
      metadata = _json_attr(file, "metadata_json")
      outcome = _json_attr(file, "outcome_json")
      if outcome.get("success") is not True or outcome.get("placed_in_box") is not True:
        raise ValueError(f"{path}: episode is not a successful pick-and-place")
      episode_index = metadata.get("episode_index")
      if not isinstance(episode_index, int) or isinstance(episode_index, bool):
        raise ValueError(f"{path}: metadata episode_index must be an integer")
      if episode_index != filename_index:
        raise ValueError(f"{path}: filename and metadata episode indices differ")
      episode_index = outer_indices.get(path, episode_index)
      recorded_model_hash = str(file.attrs.get("model_sha256", ""))
      if len(recorded_model_hash) != 64:
        raise ValueError(f"{path}: missing or invalid model_sha256")
      recorded_model_path = Path(str(file.attrs.get("model_path", "")))
      recorded_source_fingerprint = str(file.attrs.get("model_fingerprint", ""))

    if episode_index in seen_indices:
      raise ValueError(f"duplicate episode_index {episode_index}")
    seen_indices.add(episode_index)
    selected_model_path = (
      model_override.expanduser().resolve()
      if model_override is not None
      else recorded_model_path.expanduser().resolve()
    )
    if not selected_model_path.is_file():
      raise FileNotFoundError(
        f"{path}: recorded model is unavailable; pass --model-path: "
        f"{selected_model_path}"
      )
    if selected_model_path not in model_hash_cache:
      model_hash_cache[selected_model_path] = _sha256_file(selected_model_path)
    if model_hash_cache[selected_model_path] != recorded_model_hash:
      raise ValueError(
        f"{path}: selected model SHA-256 does not match the recorded model"
      )
    if recorded_source_fingerprint:
      if selected_model_path not in model_source_cache:
        model_source_cache[selected_model_path] = model_fingerprint(selected_model_path)
      if model_source_cache[selected_model_path] != recorded_source_fingerprint:
        raise ValueError(
          f"{path}: selected model or shared MJCF includes differ from the recording"
        )
    sources.append(
      SourceEpisode(
        path=path,
        sidecar_path=sidecar_path,
        episode_index=episode_index,
        hdf5_sha256=hdf5_sha256,
        sidecar_sha256=sidecar_sha256,
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        model_path=selected_model_path,
        model_sha256=recorded_model_hash,
      )
    )
  return tuple(sorted(sources, key=lambda item: item.episode_index))


def _split_sources(
  sources: tuple[SourceEpisode, ...],
  val_fraction: float,
  split_seed: int,
) -> dict[str, tuple[SourceEpisode, ...]]:
  count = len(sources)
  if count == 1 or val_fraction == 0.0:
    validation_count = 0
  else:
    validation_count = round(count * val_fraction)
    validation_count = min(count - 1, max(1, validation_count))
  scored = sorted(
    sources,
    key=lambda item: (
      hashlib.sha256(f"{split_seed}:{item.episode_index}".encode()).digest(),
      item.episode_index,
    ),
  )
  validation_indices = {source.episode_index for source in scored[:validation_count]}
  return {
    "train": tuple(
      source for source in sources if source.episode_index not in validation_indices
    ),
    "val": tuple(
      source for source in sources if source.episode_index in validation_indices
    ),
  }


def _require_site(model: mujoco.MjModel, name: str) -> int:
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
  if site_id < 0:
    raise ValueError(f"MuJoCo model is missing required site {name!r}")
  return site_id


def _make_kinematics(source: SourceEpisode) -> Kinematics:
  model = mujoco.MjModel.from_xml_path(str(source.model_path))
  data = mujoco.MjData(model)
  wrist_site_ids = {
    side: _require_site(model, f"hand_{side[0]}_base_link_site")
    for side in ("left", "right")
  }
  fingertip_site_ids = {
    side: tuple(
      _require_site(
        model,
        f"hand_{side[0]}_{finger}_{'link6' if finger == 'thumb' else 'link4'}_site",
      )
      for finger in FINGERS
    )
    for side in ("left", "right")
  }
  return Kinematics(model, data, wrist_site_ids, fingertip_site_ids)


def _rot6d(rotation: np.ndarray) -> np.ndarray:
  return np.concatenate((rotation[:, 0], rotation[:, 1]))


def _unified_state(
  qpos: np.ndarray, kinematics: Kinematics
) -> tuple[np.ndarray, np.ndarray]:
  if qpos.shape != (kinematics.model.nq,):
    raise ValueError(f"qpos has shape {qpos.shape}, expected ({kinematics.model.nq},)")
  if not np.all(np.isfinite(qpos)):
    raise ValueError("qpos contains non-finite values")
  kinematics.data.qpos[:] = qpos
  kinematics.data.qvel[:] = 0.0
  mujoco.mj_forward(kinematics.model, kinematics.data)

  wrist_positions: list[np.ndarray] = []
  wrist_rotations: list[np.ndarray] = []
  fingertip_positions: list[np.ndarray] = []
  for side in ("left", "right"):
    wrist_site_id = kinematics.wrist_site_ids[side]
    wrist_positions.append(kinematics.data.site_xpos[wrist_site_id].copy())
    world_from_site = kinematics.data.site_xmat[wrist_site_id].reshape(3, 3)
    world_from_wrist = world_from_site @ SITE_FROM_EGOSTEER_WRIST[side]
    wrist_rotations.append(_rot6d(world_from_wrist))
    fingertip_positions.extend(
      kinematics.data.site_xpos[site_id].copy()
      for site_id in kinematics.fingertip_site_ids[side]
    )
  wrist = np.concatenate((*wrist_positions, *wrist_rotations)).astype(np.float32)
  hand = np.concatenate(fingertip_positions).astype(np.float32)
  if wrist.shape != (18,) or hand.shape != (30,):
    raise RuntimeError("internal unified-state shape error")
  return wrist, hand


def _interpolate_robot_qpos(
  model: mujoco.MjModel,
  state_timestamps: np.ndarray,
  state_qpos: np.ndarray,
  target_timestamps: np.ndarray,
) -> np.ndarray:
  """Interpolate robot hinges at image time and nearest-sample other joints."""
  if state_timestamps.ndim != 1 or state_timestamps.size < 2:
    raise ValueError("state timestamps must contain at least two samples")
  if not np.all(np.isfinite(state_timestamps)) or not np.all(
    np.diff(state_timestamps) > 0.0
  ):
    raise ValueError("state timestamps must be finite and strictly increasing")
  if state_qpos.shape != (state_timestamps.size, model.nq):
    raise ValueError(
      f"state qpos has shape {state_qpos.shape}, expected "
      f"({state_timestamps.size}, {model.nq})"
    )
  if not np.all(np.isfinite(state_qpos)):
    raise ValueError("state qpos contains non-finite values")
  if target_timestamps.ndim != 1 or not np.all(np.isfinite(target_timestamps)):
    raise ValueError("target timestamps must be a finite vector")
  tolerance = 1.0e-12
  if (
    target_timestamps[0] < state_timestamps[0] - tolerance
    or target_timestamps[-1] > state_timestamps[-1] + tolerance
  ):
    raise ValueError("camera timestamps fall outside the recorded state interval")

  right = np.searchsorted(state_timestamps, target_timestamps, side="left")
  right = np.clip(right, 0, state_timestamps.size - 1)
  exact = np.isclose(
    state_timestamps[right], target_timestamps, rtol=0.0, atol=tolerance
  )
  left = np.maximum(right - 1, 0)
  left[exact] = right[exact]
  left_distance = np.abs(target_timestamps - state_timestamps[left])
  right_distance = np.abs(state_timestamps[right] - target_timestamps)
  nearest = np.where(left_distance <= right_distance, left, right)
  output = state_qpos[nearest].copy()

  hinge_qpos_addresses = np.array(
    [
      int(model.jnt_qposadr[joint_id])
      for joint_id in range(model.njnt)
      if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ],
    dtype=np.int64,
  )
  if hinge_qpos_addresses.size == 0:
    raise ValueError("MuJoCo model has no robot hinge joints")
  denominator = state_timestamps[right] - state_timestamps[left]
  alpha = np.zeros_like(target_timestamps, dtype=np.float64)
  bracketed = denominator > 0.0
  alpha[bracketed] = (
    target_timestamps[bracketed] - state_timestamps[left[bracketed]]
  ) / denominator[bracketed]
  left_hinges = state_qpos[left][:, hinge_qpos_addresses]
  right_hinges = state_qpos[right][:, hinge_qpos_addresses]
  output[:, hinge_qpos_addresses] = left_hinges + alpha[:, None] * (
    right_hinges - left_hinges
  )
  return output


def _strict_30hz_prefix(
  timestamps: np.ndarray,
  physics_hz: int,
) -> tuple[int, float]:
  if timestamps.ndim != 1 or timestamps.size < 2:
    raise ValueError("head camera needs at least two timestamps")
  if not np.all(np.isfinite(timestamps)):
    raise ValueError("head camera timestamps contain non-finite values")
  if not np.all(np.diff(timestamps) > 0.0):
    raise ValueError("head camera timestamps are not strictly increasing")
  if physics_hz <= 0:
    raise ValueError("physics_hz must be positive")
  expected = timestamps[0] + np.arange(timestamps.size) / EGOSTEER_FPS
  error = np.abs(timestamps - expected)
  # Camera deadlines are observed on the 500 Hz physics clock, so nominal
  # 30 Hz samples may be quantized by half a physics step in either direction.
  tolerance = 0.51 / physics_hz + 1.0e-12
  mismatches = np.flatnonzero(error > tolerance)
  prefix = int(mismatches[0]) if mismatches.size else int(timestamps.size)
  if prefix < 2:
    raise ValueError(
      "head camera does not contain a two-frame strict 30 Hz prefix "
      f"(tolerance={tolerance:.9f}s)"
    )
  return prefix, float(np.max(error[:prefix]))


def _camera_from_world_cv(world_from_camera: np.ndarray) -> np.ndarray:
  if world_from_camera.shape != (4, 4):
    raise ValueError("world_from_camera must have shape (4, 4)")
  if not np.all(np.isfinite(world_from_camera)):
    raise ValueError("world_from_camera contains non-finite values")
  if not np.allclose(world_from_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
    raise ValueError("world_from_camera has an invalid homogeneous final row")
  rotation = world_from_camera[:3, :3]
  if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
    raise ValueError("world_from_camera rotation is not orthonormal")
  if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
    raise ValueError("world_from_camera rotation is not proper")
  return CV_FROM_MUJOCO_CAMERA @ np.linalg.inv(world_from_camera)


def _intrinsic_vector(intrinsic: np.ndarray) -> np.ndarray:
  if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
    raise ValueError("head intrinsic must be a finite 3x3 matrix")
  vector = np.array(
    (intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]),
    dtype=np.float32,
  )
  if vector[0] <= 0.0 or vector[1] <= 0.0:
    raise ValueError("head focal lengths must be positive")
  return vector


def _jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
  if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
    raise ValueError(f"RGB frame must be uint8 HxWx3, got {rgb.shape} {rgb.dtype}")
  output = io.BytesIO()
  Image.fromarray(rgb, mode="RGB").save(
    output,
    format="JPEG",
    quality=quality,
    subsampling=0,
    optimize=False,
  )
  return output.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
  output = io.BytesIO()
  np.save(output, array, allow_pickle=False)
  return output.getvalue()


def _json_bytes(value: dict[str, Any]) -> bytes:
  return (
    json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")


def _add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
  info = tarfile.TarInfo(name)
  info.size = len(payload)
  info.mode = 0o644
  info.mtime = 0
  info.uid = 0
  info.gid = 0
  info.uname = ""
  info.gname = ""
  archive.addfile(info, io.BytesIO(payload))


def _validate_camera_datasets(file: h5py.File, source: SourceEpisode) -> h5py.Group:
  if int(file.attrs.get("camera_hz", -1)) != EGOSTEER_FPS:
    raise ValueError(f"{source.path}: source camera_hz must be exactly 30")
  required = (
    "state/timestamp",
    "state/qpos",
    "cameras/head/timestamp",
    "cameras/head/state_index",
    "cameras/head/world_from_camera",
    "cameras/head/intrinsic",
    "cameras/head/rgb",
  )
  missing = [name for name in required if name not in file]
  if missing:
    raise ValueError(f"{source.path}: missing datasets: {', '.join(missing)}")
  camera = file["cameras/head"]
  frame_count = camera["timestamp"].shape[0]
  for name in ("state_index", "world_from_camera", "rgb"):
    if camera[name].shape[0] != frame_count:
      raise ValueError(f"{source.path}: head/{name} frame count mismatch")
  if file["state/qpos"].shape[0] != file["state/timestamp"].shape[0]:
    raise ValueError(f"{source.path}: state qpos/timestamp count mismatch")
  return camera


def _convert_episode(
  source: SourceEpisode,
  split: str,
  archive: tarfile.TarFile,
  kinematics_cache: dict[str, Kinematics],
  *,
  instruction: str,
  dataset_name: str,
  jpeg_quality: int,
) -> EpisodeConversion:
  with h5py.File(source.path, "r") as file:
    camera = _validate_camera_datasets(file, source)
    timestamps = np.asarray(camera["timestamp"], dtype=np.float64)
    physics_hz = int(file.attrs.get("physics_hz", -1))
    prefix_frames, maximum_grid_error = _strict_30hz_prefix(timestamps, physics_hz)
    source_frames = int(timestamps.size)
    state_indices = np.asarray(camera["state_index"][:prefix_frames], dtype=np.int64)
    state_count = file["state/qpos"].shape[0]
    if np.any(state_indices < 0) or np.any(state_indices >= state_count):
      raise ValueError(f"{source.path}: head state_index is out of range")
    if not np.all(np.diff(state_indices) > 0):
      raise ValueError(f"{source.path}: 30 Hz head state indices must increase")
    indexed_state_timestamps = np.asarray(file["state/timestamp"])[state_indices]
    if np.any(indexed_state_timestamps > timestamps[:prefix_frames] + 1.0e-12):
      raise ValueError(f"{source.path}: head frame refers to a future state")

    if source.model_sha256 not in kinematics_cache:
      kinematics_cache[source.model_sha256] = _make_kinematics(source)
    kinematics = kinematics_cache[source.model_sha256]
    all_state_timestamps = np.asarray(file["state/timestamp"], dtype=np.float64)
    all_state_qpos = np.asarray(file["state/qpos"], dtype=np.float64)
    qpos = _interpolate_robot_qpos(
      kinematics.model,
      all_state_timestamps,
      all_state_qpos,
      timestamps[:prefix_frames],
    )
    if qpos.shape[1:] != (kinematics.model.nq,):
      raise ValueError(
        f"{source.path}: recorded qpos dimension does not match recorded model"
      )
    unified = [_unified_state(row, kinematics) for row in qpos]
    wrists = np.stack([item[0] for item in unified])
    hands = np.stack([item[1] for item in unified])

    intrinsic = _intrinsic_vector(np.asarray(camera["intrinsic"], dtype=np.float64))
    rgb_dataset = camera["rgb"]
    if rgb_dataset.ndim != 4 or rgb_dataset.shape[-1] != 3:
      raise ValueError(f"{source.path}: head RGB must have shape NxHxWx3")
    height, width = int(rgb_dataset.shape[1]), int(rgb_dataset.shape[2])
    if rgb_dataset.dtype != np.uint8:
      raise ValueError(f"{source.path}: head RGB must use uint8")
    world_from_camera = np.asarray(
      camera["world_from_camera"][: prefix_frames - 1], dtype=np.float64
    )

    meta = {
      "cameras": ["head"],
      "dataset_name": dataset_name,
      "episode_index": source.episode_index,
      "high_quality": 1,
      "instruction": instruction,
      "instruction_num": 1,
    }
    meta_payload = _json_bytes(meta)
    exported_samples = prefix_frames - 1
    for frame_index in range(exported_samples):
      camera_from_world = _camera_from_world_cv(world_from_camera[frame_index])
      lowdim = np.concatenate(
        (
          wrists[frame_index],
          hands[frame_index],
          wrists[frame_index + 1],
          hands[frame_index + 1],
          camera_from_world.reshape(-1).astype(np.float32),
          intrinsic,
        )
      ).astype(np.float32, copy=False)
      if lowdim.shape != (LOWDIM_DIM,) or not np.all(np.isfinite(lowdim)):
        raise ValueError(f"{source.path}: invalid lowdim at frame {frame_index}")
      key = f"episode_{source.episode_index:06d}_frame_{frame_index:06d}"
      rgb = np.asarray(rgb_dataset[frame_index])
      _add_tar_member(archive, f"{key}.image.jpg", _jpeg_bytes(rgb, jpeg_quality))
      _add_tar_member(archive, f"{key}.lowdim.npy", _npy_bytes(lowdim))
      _add_tar_member(archive, f"{key}.meta.json", meta_payload)

  current_stat = source.path.stat()
  if (current_stat.st_size, current_stat.st_mtime_ns) != (
    source.size_bytes,
    source.mtime_ns,
  ):
    raise RuntimeError(f"source changed during conversion: {source.path}")
  return EpisodeConversion(
    episode_index=source.episode_index,
    split=split,
    source_frames=source_frames,
    strict_prefix_frames=prefix_frames,
    exported_samples=exported_samples,
    off_grid_frames_discarded=source_frames - prefix_frames,
    maximum_grid_error_seconds=maximum_grid_error,
    image_width=width,
    image_height=height,
  )


def _write_shards(
  staging_dir: Path,
  splits: dict[str, tuple[SourceEpisode, ...]],
  *,
  instruction: str,
  dataset_name: str,
  episodes_per_shard: int,
  jpeg_quality: int,
) -> tuple[dict[str, list[ShardSummary]], tuple[EpisodeConversion, ...]]:
  shard_summaries: dict[str, list[ShardSummary]] = {"train": [], "val": []}
  conversions: list[EpisodeConversion] = []
  kinematics_cache: dict[str, Kinematics] = {}
  total_episodes = sum(len(items) for items in splits.values())
  completed_episodes = 0
  expected_image_size: tuple[int, int] | None = None

  for split in ("train", "val"):
    split_dir = staging_dir / split
    split_dir.mkdir()
    split_sources = splits[split]
    chunks = (
      split_sources[start : start + episodes_per_shard]
      for start in range(0, len(split_sources), episodes_per_shard)
    )
    for shard_index, chunk in enumerate(chunks):
      shard_path = split_dir / f"shard-{shard_index:06d}.tar"
      shard_conversions: list[EpisodeConversion] = []
      with tarfile.open(shard_path, mode="x", format=tarfile.USTAR_FORMAT) as archive:
        for source in chunk:
          completed_episodes += 1
          print(
            f"convert [{completed_episodes}/{total_episodes}] "
            f"{source.path.name} -> {split}/{shard_path.name}",
            flush=True,
          )
          conversion = _convert_episode(
            source,
            split,
            archive,
            kinematics_cache,
            instruction=instruction,
            dataset_name=dataset_name,
            jpeg_quality=jpeg_quality,
          )
          image_size = (conversion.image_width, conversion.image_height)
          if expected_image_size is None:
            expected_image_size = image_size
          elif image_size != expected_image_size:
            raise ValueError(
              f"episode {conversion.episode_index}: image size {image_size} "
              f"differs from {expected_image_size}"
            )
          conversions.append(conversion)
          shard_conversions.append(conversion)
      shard_summaries[split].append(
        ShardSummary(
          path=shard_path.relative_to(staging_dir).as_posix(),
          sha256=_sha256_file(shard_path),
          size_bytes=shard_path.stat().st_size,
          episodes=len(shard_conversions),
          episode_indices=tuple(item.episode_index for item in shard_conversions),
          samples=sum(item.exported_samples for item in shard_conversions),
        )
      )
  return shard_summaries, tuple(conversions)


def _canonical_digest(value: Any) -> str:
  payload = json.dumps(
    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
  path.write_text(
    json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def _source_manifest(
  input_dir: Path,
  sources: tuple[SourceEpisode, ...],
  created_utc: str,
) -> dict[str, Any]:
  episodes = [
    {
      "episode_index": source.episode_index,
      "hdf5": source.path.relative_to(input_dir).as_posix(),
      "hdf5_sha256": source.hdf5_sha256,
      "model_path": str(source.model_path),
      "model_sha256": source.model_sha256,
      "sidecar": source.sidecar_path.relative_to(input_dir).as_posix(),
      "sidecar_sha256": source.sidecar_sha256,
      "size_bytes": source.size_bytes,
    }
    for source in sources
  ]
  identity = [
    {
      "episode_index": item["episode_index"],
      "hdf5": item["hdf5"],
      "hdf5_sha256": item["hdf5_sha256"],
      "sidecar": item["sidecar"],
      "sidecar_sha256": item["sidecar_sha256"],
    }
    for item in episodes
  ]
  return {
    "aggregate_sha256": _canonical_digest(identity),
    "created_utc": created_utc,
    "episodes": episodes,
    "episodes_count": len(episodes),
    "schema_version": SOURCE_MANIFEST_VERSION,
    "source_root": str(input_dir),
  }


def _shard_json(summary: ShardSummary) -> dict[str, Any]:
  return {
    "episode_indices": list(summary.episode_indices),
    "episodes": summary.episodes,
    "path": summary.path,
    "samples": summary.samples,
    "sha256": summary.sha256,
    "size_bytes": summary.size_bytes,
  }


def _dataset_manifest(
  *,
  created_utc: str,
  dataset_name: str,
  instruction: str,
  split_seed: int,
  val_fraction: float,
  jpeg_quality: int,
  source_manifest_sha256: str,
  sources: tuple[SourceEpisode, ...],
  shards: dict[str, list[ShardSummary]],
  conversions: tuple[EpisodeConversion, ...],
  cameras: Sequence[str] = ("head",),
) -> dict[str, Any]:
  camera_views = canonical_camera_views(cameras)
  if not conversions:
    raise ValueError("conversion produced no samples")
  image_sizes = {(item.image_width, item.image_height) for item in conversions}
  if len(image_sizes) != 1:
    raise ValueError(f"converted image sizes differ: {sorted(image_sizes)}")
  image_width, image_height = image_sizes.pop()
  conversions_by_index = {item.episode_index: item for item in conversions}
  source_json = []
  for source in sources:
    conversion = conversions_by_index[source.episode_index]
    source_json.append(
      {
        "episode_index": source.episode_index,
        "exported_samples": conversion.exported_samples,
        "hdf5": source.path.name,
        "hdf5_sha256": source.hdf5_sha256,
        "maximum_grid_error_seconds": conversion.maximum_grid_error_seconds,
        "model_sha256": source.model_sha256,
        "off_grid_frames_discarded": conversion.off_grid_frames_discarded,
        "source_frames": conversion.source_frames,
        "split": conversion.split,
        "strict_prefix_frames": conversion.strict_prefix_frames,
      }
    )

  split_json: dict[str, Any] = {}
  for split in ("train", "val"):
    split_conversions = [item for item in conversions if item.split == split]
    split_json[split] = {
      "episode_indices": [item.episode_index for item in split_conversions],
      "episodes": len(split_conversions),
      "samples": sum(item.exported_samples for item in split_conversions),
      "shards": [_shard_json(item) for item in shards[split]],
    }

  return {
    "action_alignment": "next_30hz_frame",
    "cameras": list(camera_views),
    "coordinate_conventions": {
      "camera_extrinsic": "row-major 4x4 OpenCV world_to_camera",
      "fingertips": "world xyz; thumb,index,middle,ring,pinky; left then right",
      "right_handed_world": True,
      "rot6d": "R[:,0] followed by R[:,1], wrist_to_world",
      "site_from_egosteer_wrist": {
        side: SITE_FROM_EGOSTEER_WRIST[side].tolist() for side in ("left", "right")
      },
      "wrist_translation": "hand_{l,r}_base_link_site world position",
    },
    "created_utc": created_utc,
    "dataset_name": dataset_name,
    "depth_included": False,
    "fps": EGOSTEER_FPS,
    "image_encoding": {
      "format": "JPEG",
      "quality": jpeg_quality,
      "subsampling": 0,
    },
    "image_size": [image_width, image_height],
    "instruction": instruction,
    "lowdim_dim": camera_lowdim_dim(camera_views),
    "lowdim_dtype": "float32",
    "member_order": list(camera_member_order(camera_views)),
    "sample_key": "episode_{episode_index:06d}_frame_{frame_index:06d}",
    "schema_version": EGOSTEER_MANIFEST_VERSION,
    "source_manifest": {
      "path": "source_snapshot_manifest.json",
      "sha256": source_manifest_sha256,
    },
    "sources": source_json,
    "split_config": {
      "method": "sha256(split_seed:episode_index), exact validation count",
      "requested_val_fraction": val_fraction,
      "split_seed": split_seed,
    },
    "splits": split_json,
    "tactile_included": False,
    "time_alignment": {
      "non_hinge_qpos": "nearest state sample at each camera timestamp",
      "robot_hinge_qpos": "linear interpolation at each camera timestamp",
      "source_state_index_used_for_fk": False,
      "terminal_action": "last strict-grid frame is action-only; N-1 samples",
    },
  }


def _run(args: argparse.Namespace) -> None:
  input_dir = args.input_dir.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  if output_dir.exists():
    raise FileExistsError(f"output already exists: {output_dir}")

  lock_path = output_dir.with_name(f".{output_dir.name}.conversion.lock")
  try:
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
  except FileExistsError as error:
    raise RuntimeError(f"another conversion owns {lock_path}") from error
  staging_dir = output_dir.with_name(
    f".{output_dir.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
  )
  try:
    os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
    staging_dir.mkdir()
    created_utc = datetime.now(timezone.utc).isoformat()
    sources = _discover_sources(input_dir, args.model_path)
    splits = _split_sources(sources, args.val_fraction, args.split_seed)

    source_manifest = _source_manifest(input_dir, sources, created_utc)
    source_manifest_path = staging_dir / "source_snapshot_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    source_manifest_sha256 = _sha256_file(source_manifest_path)

    shards, conversions = _write_shards(
      staging_dir,
      splits,
      instruction=args.instruction.strip(),
      dataset_name=args.dataset_name.strip(),
      episodes_per_shard=args.episodes_per_shard,
      jpeg_quality=args.jpeg_quality,
    )
    manifest = _dataset_manifest(
      created_utc=created_utc,
      dataset_name=args.dataset_name.strip(),
      instruction=args.instruction.strip(),
      split_seed=args.split_seed,
      val_fraction=args.val_fraction,
      jpeg_quality=args.jpeg_quality,
      source_manifest_sha256=source_manifest_sha256,
      sources=sources,
      shards=shards,
      conversions=conversions,
    )
    _write_json(staging_dir / "dataset_manifest.json", manifest)
    if output_dir.exists():
      raise FileExistsError(f"output appeared during conversion: {output_dir}")
    os.rename(staging_dir, output_dir)
  except BaseException:
    if staging_dir.exists():
      shutil.rmtree(staging_dir)
    raise
  finally:
    os.close(lock_descriptor)
    lock_path.unlink(missing_ok=True)

  train = manifest["splits"]["train"]
  validation = manifest["splits"]["val"]
  print(
    f"completed atomically at {output_dir}: "
    f"train={train['episodes']} episodes/{train['samples']} samples, "
    f"val={validation['episodes']} episodes/{validation['samples']} samples",
    flush=True,
  )


def main() -> None:
  _run(_parse_args())


if __name__ == "__main__":
  main()
