#!/usr/bin/env python3
"""Convert a flat USB raw batch to standard fixed-camera EgoSteer shards.

The input directory contains ``summary.json`` plus ``usb_NNNNNN.h5`` and its
capture/result JSON sidecars.  Every episode is written to its own tar shard so
independent worker processes never share an output file.  Publication is
atomic: workers write to a sibling staging directory which is renamed only
after the complete EgoSteer validator passes.

By default the HDF5 SHA-256 recorded by the capture sidecar is trusted.  Pass
``--verify-source-hash`` when an expensive byte-for-byte rehash of every
selected source is required.
"""

# Thread limits must be set before importing NumPy (including through helpers).
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Keep each process single-threaded. Parallelism is at episode granularity.
for _thread_variable in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
):
  os.environ.setdefault(_thread_variable, "1")

import h5py
import numpy as np
from convert_to_egosteer import (
  EGOSTEER_FPS,
  SITE_FROM_EGOSTEER_WRIST,
  EpisodeConversion,
  ShardSummary,
  SourceEpisode,
  _add_tar_member,
  _camera_from_world_cv,
  _dataset_manifest,
  _intrinsic_vector,
  _jpeg_bytes,
  _json_bytes,
  _npy_bytes,
  _sha256_file,
  _source_manifest,
  _strict_30hz_prefix,
  _write_json,
  camera_image_member,
  camera_lowdim_dim,
  canonical_camera_views,
  publish_validated_staging,
)
from kaihand_tactile_env.shared.egosteer_archive import archived_motion, text
from usb_delivery_common import validate_usb_outcome
from validate_egosteer_dataset import DatasetValidator

DEFAULT_INPUT = Path(
  "/nas/chenxianchi/datasets/sim/usb_insert/raw/0914_200"
)
DEFAULT_OUTPUT = Path(
  "/nas/chenxianchi/datasets/sim/usb_insert/egosteer/0914_200"
)
DEFAULT_DATASET_NAME = "usb_insert_0914_200"
DEFAULT_INSTRUCTION = (
  "Grasp the USB plug with the right hand, lift and align it with the "
  "upward-facing socket, insert it until seated, then release it and "
  "withdraw the hand."
)
RAW_SCHEMA = "kaihand_tactile_episode_v1"
BATCH_SCHEMA = "usb_insert_raw_batch_v1"
CAPTURE_NAME = re.compile(r"^usb_(?P<episode>[0-9]{6})\.h5$")
CAPTURE_SIDECAR_NAME = re.compile(r"^usb_(?P<episode>[0-9]{6})\.json$")
RESULT_SIDECAR_NAME = re.compile(
  r"^usb_(?P<episode>[0-9]{6})\.result\.json$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RawEpisode:
  """Relocated source identity validated from the three flat-batch files."""

  episode_index: int
  hdf5_path: Path
  capture_path: Path
  result_path: Path
  recorded_hdf5_sha256: str
  capture_sidecar_sha256: str
  result_sidecar_sha256: str
  size_bytes: int
  mtime_ns: int
  object_seed: int
  noise_seed: int
  motion_profile: str


@dataclass(frozen=True)
class EpisodeFacts:
  """Validated source facts needed for provenance and output manifests."""

  episode_index: int
  source_frames: int
  strict_prefix_frames: int
  exported_samples: int
  off_grid_frames_discarded: int
  maximum_grid_error_seconds: float
  image_width: int
  image_height: int
  model_path: str
  model_sha256: str
  source_hash_verified: bool


@dataclass(frozen=True)
class EpisodeExport:
  """One independently written episode shard."""

  facts: EpisodeFacts
  split: str
  shard_path: str
  shard_sha256: str
  shard_size_bytes: int


@dataclass(frozen=True)
class _OpenEpisode:
  """Validated arrays retained only inside one worker process."""

  facts: EpisodeFacts
  acquisition: np.ndarray
  pose_times: np.ndarray
  wrists: np.ndarray
  hands: np.ndarray
  intrinsics: dict[str, np.ndarray]
  world_from_cameras: dict[str, np.ndarray]


def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError("must be a positive integer")
  return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--workers", type=_positive_int, default=4)
  parser.add_argument("--expected-episodes", type=_positive_int, default=200)
  parser.add_argument(
    "--limit",
    type=_positive_int,
    help=(
      "convert only the first N episodes after validating the complete batch "
      "summary and all source/sidecar pairs"
    ),
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help=(
      "preflight every episode in the complete batch without creating an "
      "output directory; with --limit, full summary/pair checks still run but "
      "the expensive HDF5 preflight is limited"
    ),
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help="rehash complete HDF5 files instead of trusting capture sidecar SHA-256",
  )
  parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
  parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
  parser.add_argument("--jpeg-quality", type=_positive_int, default=95)
  parser.add_argument("--rgb-block-size", type=_positive_int, default=16)
  parser.add_argument(
    "--cameras",
    nargs="+",
    default=["head"],
    choices=("head", "left_wrist", "right_wrist"),
    help="Canonical EgoSteer camera set; use: --cameras head right_wrist",
  )
  parser.add_argument(
    "--staging-root",
    type=Path,
    help="Optional fast CPFS/local staging root before atomic publication.",
  )
  args = parser.parse_args(argv)
  if args.jpeg_quality > 100:
    parser.error("--jpeg-quality must be in [1, 100]")
  if not args.instruction.strip():
    parser.error("--instruction cannot be empty")
  if not args.dataset_name.strip():
    parser.error("--dataset-name cannot be empty")
  if args.limit is not None and not args.validate_only:
    parser.error("--limit is only allowed with --validate-only")
  try:
    args.cameras = canonical_camera_views(args.cameras)
  except ValueError as error:
    parser.error(str(error))
  return args


def _read_json(path: Path, label: str) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not readable JSON: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{label} must contain a JSON object: {path}")
  return value


def _is_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
  if not _is_int(value) or value < minimum:
    raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
  return value


def _validate_path_isolation(input_dir: Path, output_dir: Path) -> None:
  """Keep generated EgoSteer payloads separate from raw and π0.5 data."""

  source = input_dir.expanduser().resolve()
  destination = output_dir.expanduser().resolve()
  if source == destination:
    raise ValueError("--output-dir must differ from --input-dir")
  if destination.is_relative_to(source) or source.is_relative_to(destination):
    raise ValueError("input and output directories must not contain one another")
  if any(part.casefold() == "pi05" for part in destination.parts):
    raise ValueError("EgoSteer output must not be written under a pi05 directory")


def _validate_outcome_shape(outcome: object, label: str) -> dict[str, Any]:
  if not isinstance(outcome, dict):
    raise ValueError(f"{label}.outcome must be an object")
  required_true = (
    "success",
    "released",
    "grasp_verified",
    "active_bottom_out_confirmed",
    "source_files_unchanged",
  )
  for field in required_true:
    if outcome.get(field) is not True:
      raise ValueError(f"{label}.outcome USB success gate failed: {field}")
  if outcome.get("object_name") != "usb_plug":
    raise ValueError(f"{label}.outcome object_name must be 'usb_plug'")
  insertion = outcome.get("insertion")
  if not isinstance(insertion, dict):
    raise ValueError(f"{label}.outcome.insertion must be an object")
  for field in ("success", "seated"):
    if insertion.get(field) is not True:
      raise ValueError(f"{label}.outcome insertion gate failed: {field}")
  return outcome


def _validate_flat_pair(
  input_dir: Path,
  row: dict[str, Any],
) -> RawEpisode:
  index = _require_int(row.get("episode_index"), "summary episode_index")
  expected_name = f"usb_{index:06d}.h5"
  raw_path = row.get("raw_path")
  if not isinstance(raw_path, str) or Path(raw_path).name != expected_name:
    raise ValueError(f"summary raw_path does not identify {expected_name}")
  if row.get("status") != "success" or row.get("success") is not True:
    raise ValueError(f"summary episode {index} is not a successful recording")
  if row.get("failure_reason") is not None:
    raise ValueError(f"summary episode {index} has a failure_reason")

  source = input_dir / expected_name
  capture_path = source.with_suffix(".json")
  result_path = input_dir / f"usb_{index:06d}.result.json"
  for label, path in (
    ("HDF5", source),
    ("capture sidecar", capture_path),
    ("result sidecar", result_path),
  ):
    if not path.is_file() or path.is_symlink():
      raise ValueError(f"episode {index} {label} must be a regular file: {path}")

  capture = _read_json(capture_path, "capture sidecar")
  result = _read_json(result_path, "result sidecar")
  if capture.get("schema_version") != RAW_SCHEMA:
    raise ValueError(f"episode {index} capture schema is not {RAW_SCHEMA}")
  if capture.get("episode") != expected_name:
    raise ValueError(f"episode {index} capture sidecar filename mismatch")
  recorded_digest = capture.get("sha256")
  if not isinstance(recorded_digest, str) or SHA256.fullmatch(recorded_digest) is None:
    raise ValueError(f"episode {index} capture SHA-256 is invalid")
  capture_outcome = _validate_outcome_shape(
    capture.get("outcome"), f"episode {index} capture"
  )
  state_samples = _require_int(
    capture.get("state_samples"), f"episode {index} capture state_samples", minimum=1
  )
  camera_samples = capture.get("camera_samples")
  if not isinstance(camera_samples, dict):
    raise ValueError(f"episode {index} capture camera_samples must be an object")
  head_samples = _require_int(
    camera_samples.get("head"),
    f"episode {index} capture head frames",
    minimum=2,
  )

  if not _is_int(result.get("episode_index")) or result["episode_index"] != index:
    raise ValueError(f"episode {index} result index mismatch")
  result_raw_path = result.get("raw_path")
  if not isinstance(result_raw_path, str) or Path(result_raw_path).name != expected_name:
    raise ValueError(f"episode {index} result raw_path mismatch")
  if (
    result.get("status") != "success"
    or result.get("success") is not True
    or result.get("failure_reason") is not None
    or result.get("recording_errors") != []
  ):
    raise ValueError(f"episode {index} result does not describe a clean success")
  validation = result.get("validation")
  if (
    not isinstance(validation, dict)
    or validation.get("valid") is not True
    or validation.get("errors") != []
    or validation.get("state_samples") != state_samples
    or validation.get("camera_samples") != camera_samples
  ):
    raise ValueError(f"episode {index} capture-time validation is incomplete")
  result_outcome = _validate_outcome_shape(
    result.get("outcome"), f"episode {index} result"
  )
  if result_outcome != capture_outcome:
    raise ValueError(f"episode {index} result/capture outcomes differ")

  for field in ("object_seed", "noise_seed"):
    expected = _require_int(row.get(field), f"summary episode {index} {field}")
    if not _is_int(result.get(field)) or result[field] != expected:
      raise ValueError(f"episode {index} result/summary {field} mismatch")
  motion_profile = row.get("motion_profile")
  if not isinstance(motion_profile, str) or not motion_profile:
    raise ValueError(f"summary episode {index} motion_profile is invalid")
  if result.get("motion_profile") != motion_profile:
    raise ValueError(f"episode {index} result/summary motion_profile mismatch")
  if validation["camera_samples"]["head"] != head_samples:
    raise ValueError(f"episode {index} head-frame count mismatch")

  stat = source.stat()
  return RawEpisode(
    episode_index=index,
    hdf5_path=source.resolve(),
    capture_path=capture_path.resolve(),
    result_path=result_path.resolve(),
    recorded_hdf5_sha256=recorded_digest,
    capture_sidecar_sha256=_sha256_file(capture_path),
    result_sidecar_sha256=_sha256_file(result_path),
    size_bytes=stat.st_size,
    mtime_ns=stat.st_mtime_ns,
    object_seed=row["object_seed"],
    noise_seed=row["noise_seed"],
    motion_profile=motion_profile,
  )


def _discover_batch(input_dir: Path, expected_episodes: int) -> tuple[
  dict[str, Any], tuple[RawEpisode, ...]
]:
  """Validate the complete summary and every HDF5/JSON file pairing."""

  if not input_dir.is_dir():
    raise FileNotFoundError(f"--input-dir is not a directory: {input_dir}")
  if input_dir.is_symlink():
    raise ValueError(f"--input-dir must not be a symbolic link: {input_dir}")
  summary_path = input_dir / "summary.json"
  summary = _read_json(summary_path, "batch summary")
  if summary.get("schema_version") != BATCH_SCHEMA:
    raise ValueError(f"summary schema must equal {BATCH_SCHEMA!r}")
  if summary.get("complete") is not True:
    raise ValueError("raw batch summary is not complete")
  rows = summary.get("episodes")
  if not isinstance(rows, list):
    raise ValueError("summary.episodes must be a list")
  declared = {
    field: _require_int(summary.get(field), f"summary.{field}")
    for field in (
      "planned_attempts",
      "recorded_attempts",
      "successful_episodes",
    )
  }
  declared.update({
    "episodes": len(rows),
  })
  if any(value != expected_episodes for value in declared.values()):
    raise ValueError(
      f"expected a complete {expected_episodes}-episode batch, got {declared}"
    )

  parsed_rows: list[dict[str, Any]] = []
  indexes: set[int] = set()
  for position, row in enumerate(rows):
    if not isinstance(row, dict):
      raise ValueError(f"summary.episodes[{position}] must be an object")
    index = _require_int(row.get("episode_index"), f"summary.episodes[{position}]")
    if index in indexes:
      raise ValueError(f"duplicate summary episode_index {index}")
    indexes.add(index)
    parsed_rows.append(row)

  physical: dict[int, Path] = {}
  for path in input_dir.glob("usb_*.h5"):
    match = CAPTURE_NAME.fullmatch(path.name)
    if match is None:
      raise ValueError(f"unexpected USB HDF5 filename: {path.name}")
    index = int(match.group("episode"))
    if index in physical:
      raise ValueError(f"duplicate USB HDF5 episode index {index}")
    physical[index] = path
  if set(physical) != indexes:
    raise ValueError(
      "summary/HDF5 episode sets differ: "
      f"summary_only={sorted(indexes - set(physical))}, "
      f"hdf5_only={sorted(set(physical) - indexes)}"
    )

  capture_indexes: set[int] = set()
  result_indexes: set[int] = set()
  for path in input_dir.glob("usb_*.json"):
    capture_match = CAPTURE_SIDECAR_NAME.fullmatch(path.name)
    result_match = RESULT_SIDECAR_NAME.fullmatch(path.name)
    if capture_match is not None:
      capture_indexes.add(int(capture_match.group("episode")))
    elif result_match is not None:
      result_indexes.add(int(result_match.group("episode")))
    else:
      raise ValueError(f"unexpected USB JSON filename: {path.name}")
  for label, observed in (
    ("capture sidecar", capture_indexes),
    ("result sidecar", result_indexes),
  ):
    if observed != indexes:
      raise ValueError(
        f"summary/{label} episode sets differ: "
        f"summary_only={sorted(indexes - observed)}, "
        f"sidecar_only={sorted(observed - indexes)}"
      )

  episodes = tuple(
    _validate_flat_pair(input_dir, row)
    for row in sorted(parsed_rows, key=lambda item: item["episode_index"])
  )
  return summary, episodes


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


def _load_open_episode(
  source: RawEpisode,
  file: h5py.File,
  *,
  source_hash_verified: bool,
  camera_views: tuple[str, ...] = ("head",),
) -> _OpenEpisode:
  """Validate all selected-camera, clock, outcome and pose contracts."""

  capture = _read_json(source.capture_path, "capture sidecar")
  result = _read_json(source.result_path, "result sidecar")
  if text(file.attrs.get("schema_version", "")) != RAW_SCHEMA:
    raise ValueError(f"{source.hdf5_path}: unsupported raw schema")
  if int(file.attrs.get("camera_hz", -1)) != EGOSTEER_FPS:
    raise ValueError(f"{source.hdf5_path}: head camera must be exactly 30 Hz")
  physics_hz = int(file.attrs.get("physics_hz", -1))
  if physics_hz <= 0:
    raise ValueError(f"{source.hdf5_path}: physics_hz must be positive")
  metadata = _json_attr(file, "metadata_json")
  outcome = _json_attr(file, "outcome_json")
  validate_usb_outcome(metadata, outcome)
  if outcome != capture["outcome"] or outcome != result["outcome"]:
    raise ValueError(f"episode {source.episode_index}: HDF5/sidecar outcomes differ")
  if metadata.get("episode_index") != source.episode_index:
    raise ValueError(f"episode {source.episode_index}: HDF5 episode index mismatch")
  for field, expected in (
    ("object_seed", source.object_seed),
    ("noise_seed", source.noise_seed),
    ("motion_profile", source.motion_profile),
  ):
    if metadata.get(field) != expected or result.get(field) != expected:
      raise ValueError(f"episode {source.episode_index}: {field} provenance mismatch")

  model_sha256 = text(file.attrs.get("model_sha256", ""))
  if SHA256.fullmatch(model_sha256) is None:
    raise ValueError(f"episode {source.episode_index}: invalid model_sha256")
  model_path = text(file.attrs.get("model_path", ""))
  if not model_path:
    raise ValueError(f"episode {source.episode_index}: missing recorded model_path")

  required = [
    "state/timestamp",
    "commands/phase",
    "cameras/head/timestamp",
    "cameras/head/pose_timestamp",
    "cameras/head/state_index",
    "cameras/head/world_from_camera",
    "cameras/head/world_from_wrist",
    "cameras/head/world_from_fingertip",
    "cameras/head/intrinsic",
    "cameras/head/rgb",
  ]
  for camera_name in camera_views[1:]:
    required.extend(
      f"cameras/{camera_name}/{field}"
      for field in (
        "timestamp",
        "pose_timestamp",
        "state_index",
        "world_from_camera",
        "intrinsic",
        "rgb",
      )
    )
  missing = [name for name in required if name not in file]
  if missing:
    raise ValueError(
      f"episode {source.episode_index}: missing HDF5 datasets {missing}"
    )
  camera = file["cameras/head"]
  acquisition = np.asarray(camera["timestamp"], dtype=np.float64)
  prefix, maximum_error = _strict_30hz_prefix(acquisition, physics_hz)
  if prefix < len(acquisition) - 1:
    raise ValueError(
      f"episode {source.episode_index}: internal off-grid camera frame"
    )
  source_frames = len(acquisition)
  expected_frames = capture["camera_samples"]["head"]
  if source_frames != expected_frames:
    raise ValueError(f"episode {source.episode_index}: camera count mismatch")
  if len(file["state/timestamp"]) != capture["state_samples"]:
    raise ValueError(f"episode {source.episode_index}: state count mismatch")
  head_state_indices = np.asarray(camera["state_index"], dtype=np.int64)

  expected_shapes = {
    "pose_timestamp": (source_frames,),
    "state_index": (source_frames,),
    "world_from_camera": (source_frames, 4, 4),
    "world_from_wrist": (source_frames, 2, 4, 4),
    "world_from_fingertip": (source_frames, 2, 5, 4, 4),
  }
  for name, expected_shape in expected_shapes.items():
    if camera[name].shape != expected_shape:
      raise ValueError(
        f"episode {source.episode_index}: head/{name} shape "
        f"{camera[name].shape} != {expected_shape}"
      )
  rgb = camera["rgb"]
  if (
    rgb.ndim != 4
    or rgb.shape[0] != source_frames
    or rgb.shape[-1] != 3
    or rgb.dtype != np.dtype(np.uint8)
  ):
    raise ValueError(
      f"episode {source.episode_index}: head/rgb must be uint8 NxHxWx3"
    )
  height, width = int(rgb.shape[1]), int(rgb.shape[2])
  if height <= 0 or width <= 0:
    raise ValueError(f"episode {source.episode_index}: invalid RGB dimensions")
  for attr, expected in (("height", height), ("width", width)):
    if attr in camera.attrs and int(camera.attrs[attr]) != expected:
      raise ValueError(f"episode {source.episode_index}: camera {attr} mismatch")

  pose_times, wrists, hands = archived_motion(file, SITE_FROM_EGOSTEER_WRIST)
  if (
    pose_times.shape != (source_frames,)
    or wrists.shape != (source_frames, 18)
    or hands.shape != (source_frames, 30)
  ):
    raise ValueError(f"episode {source.episode_index}: archived pose shape mismatch")
  intrinsics = {
    "head": _intrinsic_vector(np.asarray(camera["intrinsic"], dtype=np.float64))
  }
  world_from_cameras = {
    "head": np.asarray(camera["world_from_camera"], dtype=np.float64)
  }
  camera_samples = capture.get("camera_samples")
  if not isinstance(camera_samples, dict):
    raise ValueError(
      f"episode {source.episode_index}: capture camera_samples must be an object"
    )
  for camera_name in camera_views[1:]:
    extra = file[f"cameras/{camera_name}"]
    if camera_samples.get(camera_name) != source_frames:
      raise ValueError(
        f"episode {source.episode_index}: {camera_name} camera count mismatch"
      )
    extra_times = np.asarray(extra["timestamp"], dtype=np.float64)
    extra_pose_times = np.asarray(extra["pose_timestamp"], dtype=np.float64)
    extra_state_indices = np.asarray(extra["state_index"], dtype=np.int64)
    if not np.array_equal(extra_times, acquisition):
      raise ValueError(
        f"episode {source.episode_index}: {camera_name}/head timestamps differ"
      )
    if not np.array_equal(extra_pose_times, pose_times):
      raise ValueError(
        f"episode {source.episode_index}: {camera_name}/head pose timestamps differ"
      )
    if not np.array_equal(extra_state_indices, head_state_indices):
      raise ValueError(
        f"episode {source.episode_index}: {camera_name}/head state indices differ"
      )
    extra_rgb = extra["rgb"]
    if extra_rgb.shape != rgb.shape or extra_rgb.dtype != np.dtype(np.uint8):
      raise ValueError(
        f"episode {source.episode_index}: {camera_name} RGB contract mismatch"
      )
    extra_poses = np.asarray(extra["world_from_camera"], dtype=np.float64)
    if extra_poses.shape != (source_frames, 4, 4):
      raise ValueError(
        f"episode {source.episode_index}: {camera_name} camera pose shape mismatch"
      )
    intrinsics[camera_name] = _intrinsic_vector(
      np.asarray(extra["intrinsic"], dtype=np.float64)
    )
    world_from_cameras[camera_name] = extra_poses
  if any(not np.isfinite(value).all() for value in world_from_cameras.values()):
    raise ValueError(f"episode {source.episode_index}: camera poses are non-finite")

  exported_samples = prefix - 1
  facts = EpisodeFacts(
    episode_index=source.episode_index,
    source_frames=source_frames,
    strict_prefix_frames=prefix,
    exported_samples=exported_samples,
    off_grid_frames_discarded=source_frames - prefix,
    maximum_grid_error_seconds=maximum_error,
    image_width=width,
    image_height=height,
    model_path=model_path,
    model_sha256=model_sha256,
    source_hash_verified=source_hash_verified,
  )
  return _OpenEpisode(
    facts=facts,
    acquisition=acquisition,
    pose_times=pose_times,
    wrists=wrists,
    hands=hands,
    intrinsics=intrinsics,
    world_from_cameras=world_from_cameras,
  )


def _check_source_unchanged(source: RawEpisode) -> None:
  stat = source.hdf5_path.stat()
  if (stat.st_size, stat.st_mtime_ns) != (source.size_bytes, source.mtime_ns):
    raise RuntimeError(f"source changed during conversion: {source.hdf5_path}")


def _verify_recorded_hash(source: RawEpisode, enabled: bool) -> bool:
  if not enabled:
    return False
  actual = _sha256_file(source.hdf5_path)
  if actual != source.recorded_hdf5_sha256:
    raise ValueError(
      f"episode {source.episode_index}: HDF5 SHA-256 differs from capture sidecar"
    )
  _check_source_unchanged(source)
  return True


def _preflight_episode(
  source: RawEpisode,
  verify_source_hash: bool,
  camera_views: tuple[str, ...] = ("head",),
) -> EpisodeFacts:
  """Process-pool entry point for read-only whole-batch validation."""

  verified = _verify_recorded_hash(source, verify_source_hash)
  with h5py.File(source.hdf5_path, "r") as file:
    opened = _load_open_episode(
      source,
      file,
      source_hash_verified=verified,
      camera_views=camera_views,
    )
  _check_source_unchanged(source)
  return opened.facts


def _split_for_episode(episode_index: int) -> str:
  # Preserve the established USB whole-episode split used by both formats.
  return "val" if episode_index % 10 == 4 else "train"


def _export_episode(
  source: RawEpisode,
  staging_dir: Path,
  instruction: str,
  dataset_name: str,
  jpeg_quality: int,
  rgb_block_size: int,
  verify_source_hash: bool,
  camera_views: tuple[str, ...] = ("head",),
) -> EpisodeExport:
  """Process-pool entry point that validates and writes one complete shard."""

  verified = _verify_recorded_hash(source, verify_source_hash)
  split = _split_for_episode(source.episode_index)
  shard_path = staging_dir / split / f"shard-{source.episode_index:06d}.tar"
  with h5py.File(source.hdf5_path, "r") as file:
    opened = _load_open_episode(
      source,
      file,
      source_hash_verified=verified,
      camera_views=camera_views,
    )
    camera_groups = {
      camera_name: file[f"cameras/{camera_name}"] for camera_name in camera_views
    }
    meta_base = {
      "cameras": list(camera_views),
      "dataset_name": dataset_name,
      "episode_index": source.episode_index,
      "high_quality": 1,
      "instruction": instruction,
      "instruction_num": 1,
      "source_control": "scripted_simulation",
    }
    with tarfile.open(shard_path, mode="x", format=tarfile.USTAR_FORMAT) as archive:
      count = opened.facts.exported_samples
      for start in range(0, count, rgb_block_size):
        stop = min(count, start + rgb_block_size)
        rgb_blocks = {
          camera_name: group["rgb"][start:stop]
          for camera_name, group in camera_groups.items()
        }
        for offset, rgb in enumerate(rgb_blocks["head"]):
          frame = start + offset
          calibration = []
          for camera_name in camera_views:
            camera_from_world = _camera_from_world_cv(
              opened.world_from_cameras[camera_name][frame]
            )
            calibration.extend(
              (camera_from_world.reshape(-1), opened.intrinsics[camera_name])
            )
          lowdim = np.concatenate(
            (
              opened.wrists[frame],
              opened.hands[frame],
              opened.wrists[frame + 1],
              opened.hands[frame + 1],
              *calibration,
            )
          ).astype(np.float32, copy=False)
          expected_lowdim = camera_lowdim_dim(camera_views)
          if lowdim.shape != (expected_lowdim,) or not np.isfinite(lowdim).all():
            raise ValueError(
              f"episode {source.episode_index} frame {frame}: "
              f"invalid {expected_lowdim}D lowdim"
            )
          key = f"episode_{source.episode_index:06d}_frame_{frame:06d}"
          meta = dict(
            meta_base,
            source_frame_index=frame,
            acquisition_time_s=float(opened.acquisition[frame]),
            pose_time_s=float(opened.pose_times[frame]),
            next_pose_time_s=float(opened.pose_times[frame + 1]),
          )
          # DatasetValidator deliberately enforces this exact member order.
          for camera_name in camera_views:
            member = camera_image_member(camera_name)
            _add_tar_member(
              archive,
              key + "." + member,
              _jpeg_bytes(rgb_blocks[camera_name][offset], jpeg_quality),
            )
          _add_tar_member(archive, key + ".lowdim.npy", _npy_bytes(lowdim))
          _add_tar_member(archive, key + ".meta.json", _json_bytes(meta))
  _check_source_unchanged(source)
  return EpisodeExport(
    facts=opened.facts,
    split=split,
    shard_path=shard_path.relative_to(staging_dir).as_posix(),
    shard_sha256=_sha256_file(shard_path),
    shard_size_bytes=shard_path.stat().st_size,
  )


def _run_pool(
  operation,
  episodes: Sequence[RawEpisode],
  workers: int,
  *args,
) -> list[Any]:
  """Run an episode operation in clean spawned processes with progress."""

  if not episodes:
    return []
  maximum = min(workers, len(episodes))
  context = multiprocessing.get_context("spawn")
  executor = ProcessPoolExecutor(max_workers=maximum, mp_context=context)
  futures = {
    executor.submit(operation, episode, *args): episode.episode_index
    for episode in episodes
  }
  results: list[Any] = []
  try:
    for completed, future in enumerate(as_completed(futures), start=1):
      result = future.result()
      results.append(result)
      print(
        f"[{completed}/{len(episodes)}] episode {futures[future]:06d}",
        flush=True,
      )
  except BaseException:
    for future in futures:
      future.cancel()
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  executor.shutdown(wait=True)
  return results


def _source_objects(
  episodes: Sequence[RawEpisode],
  exports: Sequence[EpisodeExport],
) -> tuple[SourceEpisode, ...]:
  by_index = {item.facts.episode_index: item for item in exports}
  result = []
  for source in episodes:
    export = by_index[source.episode_index]
    result.append(
      SourceEpisode(
        path=source.hdf5_path,
        sidecar_path=source.capture_path,
        episode_index=source.episode_index,
        hdf5_sha256=source.recorded_hdf5_sha256,
        sidecar_sha256=source.capture_sidecar_sha256,
        size_bytes=source.size_bytes,
        mtime_ns=source.mtime_ns,
        model_path=Path(export.facts.model_path),
        model_sha256=export.facts.model_sha256,
      )
    )
  return tuple(result)


def _write_release_metadata(
  staging_dir: Path,
  output_dir: Path,
  input_dir: Path,
  full_summary: dict[str, Any],
  selected: Sequence[RawEpisode],
  exports: Sequence[EpisodeExport],
  args: argparse.Namespace,
  created_utc: str,
  elapsed_seconds: float,
) -> dict[str, Any]:
  ordered_exports = sorted(exports, key=lambda item: item.facts.episode_index)
  sources = _source_objects(selected, ordered_exports)
  summary_sha256 = _sha256_file(input_dir / "summary.json")
  source_manifest = _source_manifest(input_dir, sources, created_utc)
  source_manifest.update(
    {
      "capture_summary": {
        "path": str(input_dir / "summary.json"),
        "sha256": summary_sha256,
        "schema_version": full_summary["schema_version"],
        "complete_batch_episodes": len(full_summary["episodes"]),
        "selected_episodes": len(selected),
      },
      "hash_policy": (
        "full_hdf5_sha256_verified"
        if args.verify_source_hash
        else "capture_sidecar_sha256_trusted"
      ),
      "result_sidecars": [
        {
          "episode_index": item.episode_index,
          "path": item.result_path.name,
          "sha256": item.result_sidecar_sha256,
        }
        for item in selected
      ],
    }
  )
  source_manifest_path = staging_dir / "source_snapshot_manifest.json"
  _write_json(source_manifest_path, source_manifest)

  shard_groups: dict[str, list[ShardSummary]] = {"train": [], "val": []}
  conversions = []
  for export in ordered_exports:
    facts = export.facts
    shard_groups[export.split].append(
      ShardSummary(
        path=export.shard_path,
        sha256=export.shard_sha256,
        size_bytes=export.shard_size_bytes,
        episodes=1,
        episode_indices=(facts.episode_index,),
        samples=facts.exported_samples,
      )
    )
    conversions.append(
      EpisodeConversion(
        episode_index=facts.episode_index,
        split=export.split,
        source_frames=facts.source_frames,
        strict_prefix_frames=facts.strict_prefix_frames,
        exported_samples=facts.exported_samples,
        off_grid_frames_discarded=facts.off_grid_frames_discarded,
        maximum_grid_error_seconds=facts.maximum_grid_error_seconds,
        image_width=facts.image_width,
        image_height=facts.image_height,
      )
    )
  val_count = sum(item.split == "val" for item in ordered_exports)
  manifest = _dataset_manifest(
    created_utc=created_utc,
    dataset_name=args.dataset_name.strip(),
    instruction=args.instruction.strip(),
    split_seed=0,
    val_fraction=val_count / len(ordered_exports),
    jpeg_quality=args.jpeg_quality,
    source_manifest_sha256=_sha256_file(source_manifest_path),
    sources=sources,
    shards=shard_groups,
    conversions=tuple(conversions),
    cameras=args.cameras,
  )
  manifest["split_config"] = {
    "method": "whole episode; episode_index % 10 == 4 is validation",
    "train_episodes": len(ordered_exports) - val_count,
    "val_episodes": val_count,
  }
  manifest["time_alignment"] = {
    "pose_source": "archived native site SE3 at cameras/head/pose_timestamp",
    "qpos_interpolation": False,
    "simulation_replay": False,
    "terminal_action": "last strict-grid frame is action-only; N-1 samples",
  }
  manifest["source_hash_policy"] = source_manifest["hash_policy"]
  manifest["one_episode_per_shard"] = True
  _write_json(staging_dir / "dataset_manifest.json", manifest)

  print("validating complete staged EgoSteer dataset", flush=True)
  validation = DatasetValidator(
    staging_dir,
    precomputed_sha256={item.shard_path: item.shard_sha256 for item in exports},
  ).validate()
  validation["dataset_root"] = str(output_dir)
  _write_json(staging_dir / "validation.json", validation)
  if not validation["valid"]:
    raise ValueError(
      f"EgoSteer validation failed with {validation['error_count']} errors"
    )
  print("complete staged EgoSteer validation passed", flush=True)

  report = {
    "schema_version": "usb_to_egosteer_conversion_v1",
    "valid": True,
    "input": str(input_dir),
    "output": str(output_dir),
    "complete_batch_episodes": len(full_summary["episodes"]),
    "converted_episodes": len(ordered_exports),
    "train_episodes": len(ordered_exports) - val_count,
    "val_episodes": val_count,
    "samples": sum(item.facts.exported_samples for item in ordered_exports),
    "workers": min(args.workers, len(ordered_exports)),
    "source_hash_policy": source_manifest["hash_policy"],
    "cameras": list(args.cameras),
    "lowdim_dim": camera_lowdim_dim(args.cameras),
    "elapsed_seconds": elapsed_seconds,
  }
  _write_json(staging_dir / "conversion_report.json", report)
  (staging_dir / "README.md").write_text(
    "# USB insertion — EgoSteer\n\n"
    f"{len(ordered_exports)} episodes converted from `{input_dir}`. Each complete "
    "episode is one shard under `train/` or `val/`; validation uses "
    "`episode_index % 10 == 4`. Every frame has one JPEG per declared camera, "
    f"a native float32 {camera_lowdim_dim(args.cameras)}D lowdim array, and "
    f"metadata JSON. Cameras are `{','.join(args.cameras)}` at 30 fps; tactile "
    "streams are intentionally not included. State/action values are not "
    "pre-normalized. Compute EgoSteer "
    "normalization statistics from `train/` only.\n",
    encoding="utf-8",
  )
  return report


def _validate_only(
  args: argparse.Namespace,
  episodes: Sequence[RawEpisode],
  complete_batch_episodes: int,
) -> None:
  print(
    f"read-only EgoSteer preflight: episodes={len(episodes)}, workers="
    f"{min(args.workers, len(episodes))}",
    flush=True,
  )
  facts = _run_pool(
    _preflight_episode,
    episodes,
    args.workers,
    args.verify_source_hash,
    args.cameras,
  )
  image_sizes = {(item.image_width, item.image_height) for item in facts}
  if len(image_sizes) != 1:
    raise ValueError(f"source image sizes differ: {sorted(image_sizes)}")
  print(
    json.dumps(
      {
        "valid": True,
        "mode": "validate-only",
        "complete_batch_episodes": complete_batch_episodes,
        "preflighted_episodes": len(facts),
        "samples": sum(item.exported_samples for item in facts),
        "image_size": list(image_sizes.pop()),
        "cameras": list(args.cameras),
        "lowdim_dim": camera_lowdim_dim(args.cameras),
        "source_hash_policy": (
          "full_hdf5_sha256_verified"
          if args.verify_source_hash
          else "capture_sidecar_sha256_trusted"
        ),
        "output_created": False,
      },
      ensure_ascii=False,
      sort_keys=True,
    ),
    flush=True,
  )


def _run(args: argparse.Namespace) -> None:
  input_dir = args.input_dir.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  _validate_path_isolation(input_dir, output_dir)
  if os.path.lexists(output_dir):
    raise FileExistsError(f"output already exists: {output_dir}")

  full_summary, episodes = _discover_batch(input_dir, args.expected_episodes)
  if args.limit is not None and args.limit > len(episodes):
    raise ValueError(
      f"--limit {args.limit} exceeds complete batch size {len(episodes)}"
    )
  if args.validate_only:
    preflight = episodes[: args.limit] if args.limit is not None else episodes
    _validate_only(args, preflight, len(episodes))
    return

  selected = episodes[: args.limit] if args.limit is not None else episodes
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  lock_path = output_dir.with_name(f".{output_dir.name}.egosteer-conversion.lock")
  try:
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
  except FileExistsError as error:
    raise RuntimeError(f"another EgoSteer conversion owns {lock_path}") from error

  staging_dir: Path | None = None
  started = time.monotonic()
  try:
    os.write(
      lock_fd,
      f"pid={os.getpid()} run={uuid.uuid4().hex}\n".encode("utf-8"),
    )
    staging_parent = (
      args.staging_root.expanduser().resolve()
      if args.staging_root is not None
      else output_dir.parent
    )
    staging_parent.mkdir(parents=True, exist_ok=True)
    if staging_parent == input_dir or staging_parent.is_relative_to(input_dir):
      raise ValueError("--staging-root must not be inside the raw input directory")
    staging_dir = Path(
      tempfile.mkdtemp(
        prefix=f".{output_dir.name}.egosteer.partial-",
        dir=staging_parent,
      )
    )
    (staging_dir / "train").mkdir()
    (staging_dir / "val").mkdir()
    print(
      f"EgoSteer staging={staging_dir} episodes={len(selected)} workers="
      f"{min(args.workers, len(selected))}",
      flush=True,
    )
    exports = _run_pool(
      _export_episode,
      selected,
      args.workers,
      staging_dir,
      args.instruction.strip(),
      args.dataset_name.strip(),
      args.jpeg_quality,
      args.rgb_block_size,
      args.verify_source_hash,
      args.cameras,
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    report = _write_release_metadata(
      staging_dir,
      output_dir,
      input_dir,
      full_summary,
      selected,
      exports,
      args,
      created_utc,
      time.monotonic() - started,
    )
    publish_validated_staging(staging_dir, output_dir)
    staging_dir = None
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
  except BaseException:
    if staging_dir is not None and staging_dir.exists():
      shutil.rmtree(staging_dir)
    raise
  finally:
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
  _run(_parse_args(argv))


if __name__ == "__main__":
  main()
