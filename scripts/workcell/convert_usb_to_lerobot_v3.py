#!/usr/bin/env python3
"""Convert unified KaiHand recordings to a canonical LeRobot v3 dataset.

The output clock is the 30 Hz camera clock.  Every observation uses the
camera-provided latest-nonfuture state index.  Every action is the next camera
waypoint, so the terminal camera frame (and an optional off-grid terminal
capture) is not exported.

This is deliberately not the pi0.5 converter.  It keeps a broad, model-neutral
right-arm/right-hand schema and lets downstream model adapters select and
transform the fields they need. Task-specific diagnostics are retained under
``auxiliary.task.<task>`` and genuinely unavailable modalities are omitted
instead of filled with synthetic zeros. Episodes are converted in parallel,
RGB is streamed directly from HDF5 to FFmpeg, and durable per-episode
checkpoints allow conversion to resume after interruption.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

FPS = 30
RAW_SCHEMA = "kaihand_tactile_episode_v1"
COLLECTION_SCHEMA = "task_collection_v2"
MERGE_MANIFEST_SCHEMA = "vase_wipe_pi05_clean_merge_v1"
OUTPUT_SCHEMA = "kaihand_right_lerobot_v3_v2"
CHECKPOINT_SCHEMA = "kaihand_multitask_lerobot_v3_checkpoint_v1"


@dataclass(frozen=True)
class TaskSpec:
  name: str
  key: str
  object_name: str
  instruction: str
  control_hz: int
  diagnostic_group: str | None = None


TASK_SPECS = {
  "usb-insert": TaskSpec(
    name="usb-insert",
    key="usb_insert",
    object_name="usb_plug",
    instruction=(
      "Grasp the USB plug with the right hand, align it with the socket, insert "
      "it until seated, then release it and withdraw the hand."
    ),
    control_hz=500,
    diagnostic_group="usb_insertion",
  ),
  "poker-draw": TaskSpec(
    name="poker-draw",
    key="poker_draw",
    object_name="card",
    instruction=(
      "Slide the face-down card toward the table edge, pinch it between the "
      "fingers and thumb, lift it, and turn its face toward the robot to look at it."
    ),
    control_hz=100,
  ),
  "install-ram": TaskSpec(
    name="install-ram",
    key="install_ram",
    object_name="ram",
    instruction=(
      "Pick up the RAM module with the right hand, align its keyed edge with "
      "the socket, press it straight down until seated, then release it."
    ),
    control_hz=100,
    diagnostic_group="install_ram",
  ),
  "bulb-screw": TaskSpec(
    name="bulb-screw",
    key="bulb_screw",
    object_name="bulb",
    instruction=(
      "Pick up the light bulb with the right hand, align it with the socket, "
      "screw it clockwise until mechanically seated, then release it."
    ),
    control_hz=100,
    diagnostic_group="bulb_screw",
  ),
  "pick-place": TaskSpec(
    name="pick-place",
    key="pick_place",
    object_name="cylinder",
    instruction=(
      "Pick up the cylinder with the right hand, move it into the target box, "
      "release it, then withdraw the hand."
    ),
    control_hz=100,
  ),
  "whiteboard-wipe": TaskSpec(
    name="whiteboard-wipe",
    key="whiteboard_wipe",
    object_name="eraser",
    instruction=(
      "Pick up the eraser with the right hand, wipe all ink from the whiteboard "
      "with loaded sliding contact, then return the eraser to the table and release it."
    ),
    control_hz=100,
    diagnostic_group="whiteboard_wipe",
  ),
  "vase-wipe": TaskSpec(
    name="vase-wipe",
    key="vase_wipe",
    object_name="sponge",
    instruction=(
      "Pick up the sponge with the right hand, wipe all stains from the far inner "
      "wall of the vase with loaded sliding contact, then lift the sponge clear."
    ),
    control_hz=500,
    diagnostic_group="vase_wipe",
  ),
  "sponge-grasp": TaskSpec(
    name="sponge-grasp",
    key="sponge_grasp",
    object_name="sponge",
    instruction=(
      "Pick up the upright sponge with the right hand, carry it to the plate "
      "on the robot's right, place it inside the plate, and release it."
    ),
    control_hz=100,
    diagnostic_group="sponge_grasp",
  ),
}

# Legacy names retained for callers/tests that imported the original USB-only
# module. Runtime conversion uses the task detected from collection.json or a
# supported merge_manifest.json.
TASK = "usb-insert"
TASK_INSTRUCTION = TASK_SPECS[TASK].instruction

RIGHT_ARM_JOINT_NAMES = tuple(
  f"right_arm_joint{joint}" for joint in range(1, 8)
)
RIGHT_HAND_JOINT_NAMES = (
  "hand_r_thumb_joint1",
  "hand_r_thumb_joint2",
  "hand_r_thumb_joint3",
  "hand_r_thumb_joint5",
  "hand_r_index_joint1",
  "hand_r_index_joint2",
  "hand_r_index_joint3",
  "hand_r_index_joint4",
  "hand_r_middle_joint1",
  "hand_r_middle_joint2",
  "hand_r_middle_joint3",
  "hand_r_middle_joint4",
  "hand_r_ring_joint1",
  "hand_r_ring_joint2",
  "hand_r_ring_joint3",
  "hand_r_ring_joint4",
  "hand_r_pinky_joint1",
  "hand_r_pinky_joint2",
  "hand_r_pinky_joint3",
  "hand_r_pinky_joint4",
)
RIGHT_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
WRIST_POSE_NAMES = (
  "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
)
WRENCH_NAMES = ("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")

HEAD_IMAGE_KEY = "observation.images.head"
RIGHT_WRIST_IMAGE_KEY = "observation.images.right_wrist"

# The right-wrist optical camera and wrist site share a rigid parent in the
# common robot MJCF.  Older pick-place captures saved world_from_camera but not
# world_from_wrist, so this calibrated rigid transform reconstructs the actual
# wrist pose without loading MuJoCo. It was cross-checked against all four task
# families that recorded both transforms.
RIGHT_WRIST_CAMERA_FROM_WRIST = np.asarray([
  [-1.0, 0.0, 0.0, 0.0],
  [0.0, -0.9798040587804069, -0.19996001199600144, 0.03929214235779708],
  [0.0, -0.19996001199600144, 0.9798040587804069, -0.017496501049650123],
  [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


@dataclass(frozen=True)
class DiagnosticStream:
  source: str
  output: str
  dtype: str
  shape: tuple[int, ...]


@dataclass(frozen=True)
class SourceEpisode:
  episode_index: int
  hdf5_path: Path
  sidecar_path: Path
  relative_hdf5_path: str
  hdf5_sha256: str
  hdf5_size_bytes: int
  state_samples: int
  camera_samples: int
  metadata: dict[str, Any]
  outcome: dict[str, Any]
  task: str = TASK


@dataclass(frozen=True)
class EpisodePlan:
  source: SourceEpisode
  camera_samples: int
  strict_prefix_frames: int
  exported_frames: int
  off_grid_terminal_frames: int
  maximum_grid_error_seconds: float
  image_height: int
  image_width: int
  source_timestamp_start_s: float
  source_timestamp_end_s: float
  phase_names: tuple[str, ...]
  has_recorded_wrist_pose: bool = True
  has_fingertip_pose: bool = True
  has_actuator_control: bool = True
  has_actuator_force: bool = True
  has_tactile_contact_force: bool = True
  aggregate_tactile_group: str | None = "tactile_genesis"
  has_tactile_probes: bool = True
  diagnostics: tuple[DiagnosticStream, ...] = ()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dir", type=Path, required=True,
    help=(
      "Unified collection directory (for example raw/0920_200), a supported "
      "merge-manifest directory, or a parent containing exactly one of them."
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--task", choices=tuple(TASK_SPECS),
    help="Task to select from the collection; required for a multi-task batch.",
  )
  parser.add_argument(
    "--repo-id",
    help=(
      "LeRobot repository identifier stored in metadata; no upload is done. "
      "Defaults to kaihand/<task>_<collection-directory>."
    ),
  )
  parser.add_argument(
    "--expected-episodes", type=int, default=0,
    help="Require this many successful source episodes; 0 accepts any count.",
  )
  parser.add_argument(
    "--limit", type=int,
    help="Convert only the first N successes after validating collection counts.",
  )
  parser.add_argument(
    "--verify-source-hash", action="store_true",
    help="Recompute each large HDF5 SHA-256 instead of trusting its capture sidecar.",
  )
  parser.add_argument(
    "--validate-only", action="store_true",
    help="Validate the selected raw episodes and print the plan without writing output.",
  )
  parser.add_argument(
    "--work-dir", type=Path,
    help=(
      "Persistent checkpoint directory. Defaults to .<output-name>.resume-work "
      "beside the output. Keep it on the same filesystem as the output when possible."
    ),
  )
  parser.add_argument(
    "--staging-root", type=Path,
    help="Deprecated parent for the default work directory; prefer --work-dir.",
  )
  parser.add_argument(
    "--resume", action="store_true",
    help="Reuse the saved preflight plan and completed per-episode checkpoints.",
  )
  parser.add_argument(
    "--workers", type=int, default=3,
    help="Episodes converted concurrently (default: 3).",
  )
  parser.add_argument(
    "--ffmpeg-threads", type=int, default=2,
    help="libx264 threads per episode worker (default: 2).",
  )
  parser.add_argument(
    "--ffmpeg-preset", default="veryfast",
    help="FFmpeg libx264 preset (default: veryfast).",
  )
  parser.add_argument(
    "--video-crf", type=int, default=23,
    help="FFmpeg libx264 CRF (default: 23).",
  )
  parser.add_argument(
    "--video-files-size-in-mb", type=int, default=32,
    help=(
      "Declared LeRobot maximum video file size (default: 32 MB). The "
      "converter writes one MP4 per camera per episode."
    ),
  )
  parser.add_argument(
    "--keep-work-dir", action="store_true",
    help="Keep checkpoint artifacts after successful publication (useful for audits).",
  )
  args = parser.parse_args(argv)
  if args.expected_episodes < 0:
    parser.error("--expected-episodes must be nonnegative")
  if args.limit is not None and args.limit <= 0:
    parser.error("--limit must be positive")
  if args.workers <= 0:
    parser.error("--workers must be positive")
  if args.ffmpeg_threads <= 0:
    parser.error("--ffmpeg-threads must be positive")
  if not 0 <= args.video_crf <= 51:
    parser.error("--video-crf must be between 0 and 51")
  if args.video_files_size_in_mb <= 0:
    parser.error("--video-files-size-in-mb must be positive")
  if args.work_dir is not None and args.staging_root is not None:
    parser.error("use only one of --work-dir and --staging-root")
  return args


def _load_object(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot read JSON object {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _resolve_collection_root(input_dir: Path) -> Path:
  root = input_dir.expanduser().resolve(strict=True)
  if (root / "summary.json").is_file() and (root / "collection.json").is_file():
    return root
  if (root / "merge_manifest.json").is_file():
    return root
  candidates = sorted(
    path.parent
    for path in root.glob("*/summary.json")
    if (path.parent / "collection.json").is_file()
  )
  candidates.extend(
    path.parent
    for path in root.glob("*/merge_manifest.json")
  )
  candidates = sorted(set(candidates))
  if len(candidates) != 1:
    raise ValueError(
      f"{root} must be a collection, merge manifest, or contain exactly one "
      f"supported source; "
      f"found {len(candidates)}: {[str(path) for path in candidates]}"
    )
  return candidates[0].resolve(strict=True)


def _inside(root: Path, relative: str, *, context: str) -> Path:
  path = (root / relative).resolve(strict=True)
  if not path.is_relative_to(root):
    raise ValueError(f"{context} escapes collection root: {relative}")
  return path


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _decode_names(dataset: h5py.Dataset, *, context: str) -> tuple[str, ...]:
  if dataset.ndim != 1:
    raise ValueError(f"{context} must be one-dimensional")
  names = tuple(
    value.decode("utf-8") if isinstance(value, bytes) else str(value)
    for value in dataset[:]
  )
  if not names or len(set(names)) != len(names):
    raise ValueError(f"{context} must contain unique names")
  return names


def _json_attr(group: h5py.Group, name: str) -> dict[str, Any]:
  value = group.attrs.get(name)
  if isinstance(value, bytes):
    value = value.decode("utf-8")
  if not isinstance(value, str):
    raise ValueError(f"{group.file.filename}: attribute {name} is not JSON text")
  parsed = json.loads(value)
  if not isinstance(parsed, dict):
    raise ValueError(f"{group.file.filename}: attribute {name} must contain an object")
  return parsed


def _require_dataset(file: h5py.File, name: str) -> h5py.Dataset:
  value = file.get(name)
  if not isinstance(value, h5py.Dataset):
    raise ValueError(f"{file.filename}: missing dataset {name}")
  return value


def _columns(names: Sequence[str], selected: Sequence[str], *, context: str) -> np.ndarray:
  lookup = {name: index for index, name in enumerate(names)}
  missing = [name for name in selected if name not in lookup]
  if missing:
    raise ValueError(f"{context}: missing names {missing}")
  return np.asarray([lookup[name] for name in selected], dtype=np.int64)


def _detect_task(collection_root: Path, requested: str | None) -> str:
  if (collection_root / "merge_manifest.json").is_file():
    manifest = _load_object(collection_root / "merge_manifest.json")
    if manifest.get("schema") != MERGE_MANIFEST_SCHEMA:
      raise ValueError(
        f"{collection_root}: unsupported merge manifest schema "
        f"{manifest.get('schema')!r}"
      )
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
      raise ValueError(f"{collection_root / 'merge_manifest.json'}: episodes must be a non-empty list")
    tasks = sorted({
      row.get("task")
      for row in rows
      if isinstance(row, dict) and isinstance(row.get("task"), str)
    })
  else:
    collection = _load_object(collection_root / "collection.json")
    tasks = collection.get("tasks")
  if (
    not isinstance(tasks, list) or not tasks
    or any(task not in TASK_SPECS for task in tasks)
    or len(set(tasks)) != len(tasks)
  ):
    raise ValueError(
      f"{collection_root}: invalid supported task list {tasks!r}"
    )
  if requested is None:
    if len(tasks) != 1:
      raise ValueError(
        f"{collection_root}: multi-task batch requires --task; found {tasks!r}"
      )
    return str(tasks[0])
  if requested not in tasks:
    raise ValueError(
      f"--task={requested!r} is not in collection tasks {tasks!r}"
    )
  return requested


def _validate_success_gate(
  path: Path,
  spec: TaskSpec,
  outcome: dict[str, Any],
) -> None:
  if outcome.get("success") is not True:
    raise ValueError(f"{path}: task outcome is not successful")
  object_name = outcome.get("object_name")
  if object_name is not None and object_name != spec.object_name:
    raise ValueError(
      f"{path}: outcome object {object_name!r} is not {spec.object_name!r}"
    )
  if spec.name == "usb-insert":
    insertion = outcome.get("insertion")
    if (
      outcome.get("released") is not True
      or outcome.get("grasp_verified") is not True
      or outcome.get("active_bottom_out_confirmed") is not True
      or not isinstance(insertion, dict)
      or insertion.get("success") is not True
      or insertion.get("seated") is not True
      or insertion.get("bottom_out_confirmed") is not True
    ):
      raise ValueError(f"{path}: USB success gate failed")
  elif spec.name == "poker-draw":
    if outcome.get("task_completed") is not True:
      raise ValueError(f"{path}: card draw completion gate failed")
  elif spec.name == "install-ram":
    final_state = outcome.get("final_state")
    if (
      not isinstance(final_state, dict)
      or final_state.get("success") is not True
      or final_state.get("seated") is not True
      or final_state.get("bottom_out_confirmed") is not True
    ):
      raise ValueError(f"{path}: RAM seating gate failed")
  elif spec.name == "bulb-screw":
    state = outcome.get("state")
    if (
      outcome.get("tightening_verified") is not True
      or not isinstance(state, dict)
      or state.get("success") is not True
      or state.get("seated") is not True
    ):
      raise ValueError(f"{path}: bulb tightening gate failed")
  elif spec.name == "pick-place" and outcome.get("placed_in_box") is not True:
    raise ValueError(f"{path}: pick-place placement gate failed")
  elif spec.name == "whiteboard-wipe":
    remaining = outcome.get("ink_remaining")
    if (
      outcome.get("pickup_verified") is not True
      or outcome.get("released_on_table") is not True
      or not isinstance(remaining, list)
      or not remaining
      or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value > 1.0e-9
        for value in remaining
      )
    ):
      raise ValueError(f"{path}: whiteboard cleaning/release gate failed")
  elif spec.name == "vase-wipe":
    pickup = outcome.get("pickup")
    cleaning = outcome.get("cleaning")
    cleaned = outcome.get("cleaned_patch_count")
    patch_count = outcome.get("patch_count")
    remaining = outcome.get("remaining_dirt")
    try:
      remaining_array = np.asarray(remaining, dtype=np.float64)
    except (TypeError, ValueError):
      remaining_array = np.asarray([], dtype=np.float64)
    mean_remaining = cleaning.get("mean_remaining") if isinstance(cleaning, dict) else None
    worst_remaining = cleaning.get("worst_remaining") if isinstance(cleaning, dict) else None
    if (
      outcome.get("motion_completed") is not True
      or not isinstance(pickup, dict)
      or pickup.get("success") is not True
      or not isinstance(cleaning, dict)
      or isinstance(mean_remaining, bool)
      or not isinstance(mean_remaining, (int, float))
      or not np.isfinite(mean_remaining)
      or float(mean_remaining) > 0.05
      or isinstance(worst_remaining, bool)
      or not isinstance(worst_remaining, (int, float))
      or not np.isfinite(worst_remaining)
      or float(worst_remaining) > 0.1
      or isinstance(cleaned, bool)
      or not isinstance(cleaned, int)
      or isinstance(patch_count, bool)
      or not isinstance(patch_count, int)
      or cleaned <= 0
      or cleaned != patch_count
      or remaining_array.size == 0
      or not np.all(np.isfinite(remaining_array))
      or float(np.max(remaining_array)) > 0.1
    ):
      raise ValueError(f"{path}: vase cleaning/pickup gate failed")
  elif spec.name == "sponge-grasp":
    integrity = outcome.get("contact_integrity")
    if (
      not isinstance(integrity, dict)
      or integrity.get("checked_every_physics_step") is not True
      or integrity.get("penetration_count") != 0
    ):
      raise ValueError(f"{path}: sponge contact-integrity gate failed")


def _source_from_files(
  collection_root: Path,
  *,
  index: int,
  hdf5_path: Path,
  sidecar_path: Path,
  task: str,
  spec: TaskSpec,
  verify_source_hash: bool,
) -> SourceEpisode:
  """Validate one raw HDF5/sidecar pair and build its source record."""
  hdf5_path = hdf5_path.resolve(strict=True)
  sidecar_path = sidecar_path.resolve(strict=True)
  if not hdf5_path.is_relative_to(collection_root):
    raise ValueError(f"episode {index} HDF5 escapes collection root")
  if not sidecar_path.is_relative_to(collection_root):
    raise ValueError(f"episode {index} sidecar escapes collection root")
  sidecar = _load_object(sidecar_path)
  declared_hash = sidecar.get("sha256")
  if (
    not isinstance(declared_hash, str)
    or len(declared_hash) != 64
    or any(character not in "0123456789abcdef" for character in declared_hash)
  ):
    raise ValueError(f"{sidecar_path}: invalid acquisition SHA-256")
  if sidecar.get("episode") != hdf5_path.name:
    raise ValueError(f"{sidecar_path}: episode does not name {hdf5_path.name}")
  if verify_source_hash and _sha256_file(hdf5_path) != declared_hash:
    raise ValueError(f"{hdf5_path}: SHA-256 differs from acquisition sidecar")
  camera_samples = sidecar.get("camera_samples", {})
  state_samples = sidecar.get("state_samples")
  if (
    isinstance(state_samples, bool)
    or not isinstance(state_samples, int)
    or state_samples <= 1
    or not isinstance(camera_samples, dict)
    or camera_samples.get("head") != camera_samples.get("right_wrist")
    or isinstance(camera_samples.get("head"), bool)
    or not isinstance(camera_samples.get("head"), int)
    or camera_samples["head"] <= 2
  ):
    raise ValueError(f"{sidecar_path}: invalid state/camera sample counts")
  if sidecar.get("schema_version") != RAW_SCHEMA:
    raise ValueError(f"{sidecar_path}: unsupported raw schema")
  if task == "sponge-grasp" and (
    sidecar.get("task_audit_passed") is not True
    or not isinstance(sidecar.get("validation"), dict)
    or sidecar["validation"].get("valid") is not True
  ):
    raise ValueError(f"{sidecar_path}: sponge recorded-task audit is missing or failed")
  outcome = sidecar.get("outcome")
  if not isinstance(outcome, dict):
    raise ValueError(f"{sidecar_path}: outcome must be an object")
  _validate_success_gate(hdf5_path, spec, outcome)
  return SourceEpisode(
    episode_index=index,
    hdf5_path=hdf5_path,
    sidecar_path=sidecar_path,
    relative_hdf5_path=hdf5_path.relative_to(collection_root).as_posix(),
    hdf5_sha256=declared_hash,
    hdf5_size_bytes=hdf5_path.stat().st_size,
    state_samples=state_samples,
    camera_samples=camera_samples["head"],
    metadata={},
    outcome=outcome,
    task=task,
  )


def _manifest_path(collection_root: Path, value: Any, *, field: str) -> Path:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{collection_root / 'merge_manifest.json'}: {field} must be a path")
  path = Path(value).expanduser()
  if not path.is_absolute():
    path = collection_root / path
  return path.resolve()


def _discover_merged_sources(
  collection_root: Path,
  *,
  task: str,
  expected_episodes: int,
  verify_source_hash: bool,
) -> tuple[SourceEpisode, ...]:
  manifest_path = collection_root / "merge_manifest.json"
  manifest = _load_object(manifest_path)
  if manifest.get("schema") != MERGE_MANIFEST_SCHEMA:
    raise ValueError(
      f"{manifest_path}: unsupported merge manifest schema {manifest.get('schema')!r}"
    )
  rows = manifest.get("episodes")
  if not isinstance(rows, list) or not rows:
    raise ValueError(f"{manifest_path}: episodes must be a non-empty list")
  declared_count = manifest.get("target_episode_count")
  if (
    isinstance(declared_count, bool)
    or not isinstance(declared_count, int)
    or declared_count != len(rows)
  ):
    raise ValueError(
      f"{manifest_path}: target_episode_count disagrees with episode records"
    )
  if expected_episodes and len(rows) != expected_episodes:
    raise ValueError(f"expected {expected_episodes} episodes, found {len(rows)}")

  spec = TASK_SPECS[task]
  sources: list[SourceEpisode] = []
  seen: set[int] = set()
  for row_number, row in enumerate(rows):
    if not isinstance(row, dict) or row.get("task") != task:
      raise ValueError(f"{manifest_path}: row {row_number} is not a {task} episode")
    index = row.get("output_episode_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
      raise ValueError(f"{manifest_path}: row {row_number} has invalid output_episode_index")
    if index in seen:
      raise ValueError(f"{manifest_path}: duplicate output_episode_index {index}")
    seen.add(index)

    source_dir = _manifest_path(
      collection_root, row.get("source_episode_dir"),
      field=f"row {row_number} source_episode_dir",
    )
    source_hdf5 = _manifest_path(
      collection_root, row.get("source_hdf5"),
      field=f"row {row_number} source_hdf5",
    )
    source_sidecar = _manifest_path(
      collection_root, row.get("source_sidecar"),
      field=f"row {row_number} source_sidecar",
    )
    try:
      hdf5_relative = source_hdf5.relative_to(source_dir)
      sidecar_relative = source_sidecar.relative_to(source_dir)
    except ValueError as error:
      raise ValueError(
        f"{manifest_path}: row {row_number} source files are outside source_episode_dir"
      ) from error
    output_dir = _manifest_path(
      collection_root, row.get("output_episode_dir"),
      field=f"row {row_number} output_episode_dir",
    ).resolve(strict=True)
    if not output_dir.is_relative_to(collection_root):
      raise ValueError(f"{manifest_path}: row {row_number} output directory escapes collection root")
    hdf5_path = (output_dir / hdf5_relative).resolve(strict=True)
    sidecar_path = (output_dir / sidecar_relative).resolve(strict=True)
    sources.append(_source_from_files(
      collection_root,
      index=index,
      hdf5_path=hdf5_path,
      sidecar_path=sidecar_path,
      task=task,
      spec=spec,
      verify_source_hash=verify_source_hash,
    ))
  sources.sort(key=lambda source: source.episode_index)
  return tuple(sources)


def _discover_sources(
  collection_root: Path,
  *,
  task: str,
  expected_episodes: int,
  verify_source_hash: bool,
) -> tuple[SourceEpisode, ...]:
  if (collection_root / "merge_manifest.json").is_file():
    return _discover_merged_sources(
      collection_root,
      task=task,
      expected_episodes=expected_episodes,
      verify_source_hash=verify_source_hash,
    )

  spec = TASK_SPECS[task]
  collection = _load_object(collection_root / "collection.json")
  if (
    collection.get("schema") != COLLECTION_SCHEMA
    or not isinstance(collection.get("tasks"), list)
    or task not in collection["tasks"]
    or collection.get("collection_mode") != "target-successes"
  ):
    raise ValueError(f"{collection_root}: unsupported collection contract")

  summary = _load_object(collection_root / "summary.json")
  rows = summary.get("episodes")
  if not isinstance(rows, list):
    raise ValueError(f"{collection_root / 'summary.json'}: episodes must be a list")
  all_successes = [
    row for row in rows
    if isinstance(row, dict) and row.get("status") == "success"
  ]
  successes = [row for row in all_successes if row.get("task") == task]
  declared = summary.get("success_count")
  by_task = summary.get("by_task", {})
  if declared != len(all_successes):
    raise ValueError(
      f"summary success_count={declared!r}, but {len(all_successes)} success rows exist"
    )
  if not isinstance(by_task, dict) or by_task.get(task, {}).get("success") != len(successes):
    raise ValueError("summary per-task success count disagrees with success rows")
  if summary.get("target_met", {}).get(task) is not True:
    raise ValueError(f"collection did not meet the {task} success target")
  if expected_episodes and len(successes) != expected_episodes:
    raise ValueError(
      f"expected {expected_episodes} successes, found {len(successes)}"
    )

  sources: list[SourceEpisode] = []
  seen: set[int] = set()
  for row_number, row in enumerate(successes):
    index = row.get("episode_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
      raise ValueError(f"success row {row_number} has invalid episode_index")
    if index in seen:
      raise ValueError(f"duplicate successful episode_index {index}")
    seen.add(index)
    listed = row.get("hdf5")
    if not isinstance(listed, list) or len(listed) != 1 or not isinstance(listed[0], str):
      raise ValueError(f"success episode {index} must name exactly one HDF5")
    hdf5_path = _inside(
      collection_root, listed[0], context=f"episode {index} HDF5",
    )
    sidecar_path = hdf5_path.with_suffix(".json").resolve(strict=True)
    sources.append(_source_from_files(
      collection_root,
      index=index,
      hdf5_path=hdf5_path,
      sidecar_path=sidecar_path,
      task=task,
      spec=spec,
      verify_source_hash=verify_source_hash,
    ))
  sources.sort(key=lambda source: source.episode_index)
  return tuple(sources)


def _strict_30hz_prefix(timestamps: np.ndarray, physics_hz: int) -> tuple[int, float]:
  if timestamps.ndim != 1 or timestamps.size < 2:
    raise ValueError("camera needs at least two timestamps")
  if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
    raise ValueError("camera timestamps must be finite and strictly increasing")
  if physics_hz <= 0:
    raise ValueError("physics_hz must be positive")
  expected = timestamps[0] + np.arange(timestamps.size, dtype=np.float64) / FPS
  error = np.abs(timestamps - expected)
  tolerance = 0.51 / physics_hz + 1.0e-12
  mismatches = np.flatnonzero(error > tolerance)
  prefix = int(mismatches[0]) if mismatches.size else int(timestamps.size)
  if prefix < 2:
    raise ValueError("camera has no two-frame strict 30 Hz prefix")
  if prefix < timestamps.size - 1:
    raise ValueError(
      "only one optional off-grid terminal camera frame is supported; "
      f"first mismatch is {prefix}/{timestamps.size}"
    )
  return prefix, float(np.max(error[:prefix]))


def _control_tick_30hz_prefix(
  timestamps: np.ndarray, physics_hz: int, control_hz: int,
) -> tuple[int, float]:
  """Validate 30 Hz camera deadlines rounded up to control ticks."""
  if timestamps.ndim != 1 or timestamps.size < 2:
    raise ValueError("camera needs at least two timestamps")
  if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
    raise ValueError("camera timestamps must be finite and strictly increasing")
  if physics_hz <= 0 or control_hz <= 0:
    raise ValueError("physics_hz and control_hz must be positive")
  frame = np.arange(timestamps.size, dtype=np.int64)
  control_ticks = (frame * control_hz + FPS - 1) // FPS
  expected = timestamps[0] + control_ticks / control_hz
  error = np.abs(timestamps - expected)
  tolerance = 0.51 / physics_hz + 1.0e-12
  mismatches = np.flatnonzero(error > tolerance)
  prefix = int(mismatches[0]) if mismatches.size else int(timestamps.size)
  if prefix < 2:
    raise ValueError("camera has no two-frame control-tick/30 Hz deadline prefix")
  if prefix < timestamps.size - 1:
    raise ValueError(
      "only one optional off-grid terminal camera frame is supported; "
      f"first mismatch is {prefix}/{timestamps.size}"
    )
  return prefix, float(np.max(error[:prefix]))


def _diagnostic_dtype(dataset: h5py.Dataset, *, context: str) -> str:
  kind = dataset.dtype.kind
  if kind == "b":
    return "bool"
  if kind in "iu":
    return "int64"
  if kind == "f":
    return "float32"
  raise ValueError(f"{context}: unsupported diagnostic dtype {dataset.dtype}")


def _discover_diagnostics(
  file: h5py.File,
  spec: TaskSpec,
  state_samples: int,
) -> tuple[DiagnosticStream, ...]:
  candidates: list[tuple[str, str]] = []
  for name in (
    "drive_actual_fx_n",
    "drive_budget_active",
    "drive_limit_n",
    "drive_requested_fx_n",
  ):
    source = f"commands/{name}"
    if source in file:
      candidates.append((source, f"commands.{name}"))
  if spec.diagnostic_group is not None:
    group = file.get(spec.diagnostic_group)
    if not isinstance(group, h5py.Group):
      raise ValueError(
        f"{file.filename}: missing task diagnostic group {spec.diagnostic_group}"
      )
    for name, value in sorted(group.items()):
      if isinstance(value, h5py.Dataset):
        candidates.append((f"{spec.diagnostic_group}/{name}", name))

  streams: list[DiagnosticStream] = []
  for source, suffix in candidates:
    dataset = _require_dataset(file, source)
    if not dataset.shape or dataset.shape[0] != state_samples:
      raise ValueError(f"{file.filename}: {source} is not on the state clock")
    dtype = _diagnostic_dtype(dataset, context=source)
    shape = tuple(map(int, dataset.shape[1:])) or (1,)
    streams.append(DiagnosticStream(
      source=source,
      output=f"auxiliary.task.{spec.key}.{suffix}",
      dtype=dtype,
      shape=shape,
    ))
  return tuple(streams)


def _preflight_episode(source: SourceEpisode) -> EpisodePlan:
  spec = TASK_SPECS[source.task]
  with h5py.File(source.hdf5_path, "r") as file:
    schema = file.attrs.get("schema_version")
    if isinstance(schema, bytes):
      schema = schema.decode("utf-8")
    if schema != RAW_SCHEMA:
      raise ValueError(f"{source.hdf5_path}: unsupported raw schema")
    metadata = _json_attr(file, "metadata_json")
    outcome = _json_attr(file, "outcome_json")
    if outcome != source.outcome:
      raise ValueError(f"{source.sidecar_path}: outcome differs from HDF5 outcome_json")
    _validate_success_gate(source.hdf5_path, spec, outcome)
    active_objects = metadata.get("active_objects", [])
    if not isinstance(active_objects, list):
      active_objects = []
    metadata_object = metadata.get("object_name", metadata.get("object"))
    if (
      metadata.get("scene") != spec.name
      or (
        metadata_object != spec.object_name
        and spec.object_name not in active_objects
      )
    ):
      raise ValueError(
        f"{source.hdf5_path}: source metadata does not match {spec.name}/{spec.object_name}"
      )
    side = metadata.get("side")
    if side is not None and side != "right":
      raise ValueError(f"{source.hdf5_path}: only right-hand captures are supported")
    validated_source = replace(source, metadata=metadata, outcome=outcome)
    if int(file.attrs.get("camera_hz", -1)) != FPS:
      raise ValueError(f"{source.hdf5_path}: camera_hz must be {FPS}")
    if int(file.attrs.get("control_hz", -1)) != spec.control_hz:
      raise ValueError(
        f"{source.hdf5_path}: control_hz must be {spec.control_hz} for {spec.name}"
      )
    physics_hz = int(file.attrs.get("physics_hz", -1))

    state_timestamps = np.asarray(
      _require_dataset(file, "state/timestamp")[:], dtype=np.float64,
    )
    if state_timestamps.shape != (source.state_samples,):
      raise ValueError(f"{source.hdf5_path}: state sample count mismatch")
    if not np.all(np.isfinite(state_timestamps)) or not np.all(np.diff(state_timestamps) > 0):
      raise ValueError(f"{source.hdf5_path}: state timestamps must increase")

    head_timestamps = np.asarray(
      _require_dataset(file, "cameras/head/timestamp")[:], dtype=np.float64,
    )
    wrist_timestamps = np.asarray(
      _require_dataset(file, "cameras/right_wrist/timestamp")[:], dtype=np.float64,
    )
    head_indices = np.asarray(
      _require_dataset(file, "cameras/head/state_index")[:], dtype=np.int64,
    )
    wrist_indices = np.asarray(
      _require_dataset(file, "cameras/right_wrist/state_index")[:], dtype=np.int64,
    )
    if (
      head_timestamps.shape != (source.camera_samples,)
      or not np.array_equal(head_timestamps, wrist_timestamps)
      or not np.array_equal(head_indices, wrist_indices)
    ):
      raise ValueError(f"{source.hdf5_path}: head/right-wrist camera clocks differ")
    expected_indices = np.searchsorted(
      state_timestamps, head_timestamps, side="right",
    ) - 1
    if not np.array_equal(head_indices, expected_indices):
      raise ValueError(
        f"{source.hdf5_path}: camera state_index is not latest-nonfuture"
      )
    if np.any(head_indices < 0) or not np.all(np.diff(head_indices) > 0):
      raise ValueError(f"{source.hdf5_path}: invalid camera state indices")
    if spec.name == "sponge-grasp":
      prefix, grid_error = _control_tick_30hz_prefix(
        head_timestamps, physics_hz, spec.control_hz,
      )
    else:
      prefix, grid_error = _strict_30hz_prefix(head_timestamps, physics_hz)

    head_rgb = _require_dataset(file, "cameras/head/rgb")
    wrist_rgb = _require_dataset(file, "cameras/right_wrist/rgb")
    if (
      head_rgb.dtype != np.uint8
      or wrist_rgb.dtype != np.uint8
      or head_rgb.ndim != 4
      or wrist_rgb.ndim != 4
      or head_rgb.shape[0] != source.camera_samples
      or wrist_rgb.shape != head_rgb.shape
      or head_rgb.shape[-1] != 3
    ):
      raise ValueError(f"{source.hdf5_path}: RGB arrays must be matching uint8 NHWC")

    state_names = _decode_names(
      _require_dataset(file, "state/joint_names"), context="state/joint_names",
    )
    arm_names = _decode_names(
      _require_dataset(file, "commands/arm_joint_names"),
      context="commands/arm_joint_names",
    )
    hand_names = _decode_names(
      _require_dataset(file, "commands/hand_joint_names"),
      context="commands/hand_joint_names",
    )
    _columns(state_names, RIGHT_JOINT_NAMES, context="state joints")
    _columns(arm_names, RIGHT_ARM_JOINT_NAMES, context="arm commands")
    _columns(hand_names, RIGHT_HAND_JOINT_NAMES, context="hand commands")

    has_actuator_control = (
      "commands/actuator_names" in file and "commands/actuator_control" in file
    )
    if has_actuator_control:
      actuator_names = _decode_names(
        _require_dataset(file, "commands/actuator_names"),
        context="commands/actuator_names",
      )
      _columns(actuator_names, RIGHT_JOINT_NAMES, context="actuators")
    elif "commands/actuator_names" in file or "commands/actuator_control" in file:
      raise ValueError(f"{source.hdf5_path}: incomplete actuator-control recording")
    has_actuator_force = "physics/actuator_force" in file
    if has_actuator_force and not has_actuator_control:
      raise ValueError(f"{source.hdf5_path}: actuator force has no actuator names")

    side_names = _decode_names(
      _require_dataset(file, "wrist_wrench/side_names"),
      context="wrist_wrench/side_names",
    )
    if "right" not in side_names:
      raise ValueError(f"{source.hdf5_path}: wrist wrench has no right side")

    has_tactile_contact_force = "tactile_contact_force" in file
    if has_tactile_contact_force:
      tactile_names = _decode_names(
        _require_dataset(file, "tactile_contact_force/link_names"),
        context="tactile_contact_force/link_names",
      )
      if sum(name.startswith("hand_r_") for name in tactile_names) != 5:
        raise ValueError(f"{source.hdf5_path}: expected five right tactile pads")

    aggregate_group = (
      "tactile_genesis" if "tactile_genesis" in file
      else "tactile_proxy" if "tactile_proxy" in file
      else None
    )
    if aggregate_group is None:
      raise ValueError(f"{source.hdf5_path}: no aggregate tactile stream")
    aggregate_names = _decode_names(
      _require_dataset(file, f"{aggregate_group}/link_names"),
      context=f"{aggregate_group}/link_names",
    )
    if sum(name.startswith("hand_r_") for name in aggregate_names) != 5:
      raise ValueError(f"{source.hdf5_path}: expected five right aggregate tactile pads")
    has_tactile_probes = aggregate_group == "tactile_genesis"
    if has_tactile_probes:
      probe_count = _require_dataset(file, f"{aggregate_group}/probe_contact").shape[1]
      if probe_count != len(aggregate_names) * 35:
        raise ValueError(f"{source.hdf5_path}: expected 35 probes per tactile link")

    required_state_streams = (
      "state/robot_joint_position",
      "state/robot_joint_velocity",
      "state/robot_joint_effort",
      "commands/arm_joint_target",
      "commands/hand_joint_target",
      "commands/phase",
      f"objects/{spec.object_name}/pose_wxyz",
      f"objects/{spec.object_name}/twist_linear_angular",
      "wrist_wrench/force_local_n",
      "wrist_wrench/force_world_n",
      "wrist_wrench/torque_local_nm",
      "wrist_wrench/torque_world_nm",
      "wrist_wrench/origin_world_m",
      "wrist_wrench/world_from_sensor_rotation",
    )
    optional_state_streams: list[str] = []
    if has_actuator_control:
      optional_state_streams.append("commands/actuator_control")
    if has_actuator_force:
      optional_state_streams.append("physics/actuator_force")
    if has_tactile_contact_force:
      optional_state_streams.extend(
        f"tactile_contact_force/{name}" for name in (
          "contact_count", "force_world_n", "normal_axis_world",
          "normal_force_n", "normal_taxel_force_n", "tangent_basis_world",
          "tangent_force_n", "tangent_taxel_force_n",
        )
      )
    optional_state_streams.extend(
      f"{aggregate_group}/{name}" for name in (
        "centroid_world", "contact", "contact_count", "force_local",
        "force_world", "normal_force", "torque_world",
      )
    )
    if has_tactile_probes:
      optional_state_streams.extend(
        f"{aggregate_group}/{name}" for name in (
          "probe_contact", "probe_contact_instantaneous", "probe_depth",
          "probe_target_geom_id",
        )
      )
    diagnostics = _discover_diagnostics(file, spec, source.state_samples)
    optional_state_streams.extend(stream.source for stream in diagnostics)
    for name in (*required_state_streams, *optional_state_streams):
      dataset = _require_dataset(file, name)
      if dataset.shape[0] != source.state_samples:
        raise ValueError(f"{source.hdf5_path}: {name} is off the state clock")

    has_recorded_wrist_pose = "cameras/head/world_from_wrist" in file
    has_fingertip_pose = "cameras/head/world_from_fingertip" in file
    camera_streams = [
      "cameras/head/world_from_camera",
      "cameras/right_wrist/world_from_camera",
    ]
    if has_recorded_wrist_pose:
      camera_streams.append("cameras/head/world_from_wrist")
    if has_fingertip_pose:
      camera_streams.append("cameras/head/world_from_fingertip")
    for name in camera_streams:
      dataset = _require_dataset(file, name)
      if dataset.shape[0] != source.camera_samples:
        raise ValueError(f"{source.hdf5_path}: {name} is off the camera clock")

    phases = _decode_string_vector(_require_dataset(file, "commands/phase")[:])
    phase_names = tuple(dict.fromkeys(phases))
    height, width = map(int, head_rgb.shape[1:3])
    return EpisodePlan(
      source=validated_source,
      camera_samples=source.camera_samples,
      strict_prefix_frames=prefix,
      exported_frames=prefix - 1,
      off_grid_terminal_frames=source.camera_samples - prefix,
      maximum_grid_error_seconds=grid_error,
      image_height=height,
      image_width=width,
      source_timestamp_start_s=float(head_timestamps[0]),
      source_timestamp_end_s=float(head_timestamps[prefix - 1]),
      phase_names=phase_names,
      has_recorded_wrist_pose=has_recorded_wrist_pose,
      has_fingertip_pose=has_fingertip_pose,
      has_actuator_control=has_actuator_control,
      has_actuator_force=has_actuator_force,
      has_tactile_contact_force=has_tactile_contact_force,
      aggregate_tactile_group=aggregate_group,
      has_tactile_probes=has_tactile_probes,
      diagnostics=diagnostics,
    )


def _decode_string_vector(values: Iterable[Any]) -> tuple[str, ...]:
  return tuple(
    value.decode("utf-8") if isinstance(value, bytes) else str(value)
    for value in values
  )


def _rotation_matrices_to_quaternions_xyzw(matrices: np.ndarray) -> np.ndarray:
  """Convert (..., 3, 3) rotation matrices to sign-continuous XYZW quaternions."""

  values = np.asarray(matrices, dtype=np.float64)
  if values.shape[-2:] != (3, 3):
    raise ValueError(f"rotation matrices need trailing shape (3, 3), got {values.shape}")
  flat = values.reshape(-1, 3, 3)
  result = np.empty((flat.shape[0], 4), dtype=np.float64)
  for index, matrix in enumerate(flat):
    trace = float(np.trace(matrix))
    if trace > 0.0:
      scale = np.sqrt(trace + 1.0) * 2.0
      quaternion = np.array([
        (matrix[2, 1] - matrix[1, 2]) / scale,
        (matrix[0, 2] - matrix[2, 0]) / scale,
        (matrix[1, 0] - matrix[0, 1]) / scale,
        0.25 * scale,
      ])
    else:
      diagonal = np.diag(matrix)
      axis = int(np.argmax(diagonal))
      if axis == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = np.array([
          0.25 * scale,
          (matrix[0, 1] + matrix[1, 0]) / scale,
          (matrix[0, 2] + matrix[2, 0]) / scale,
          (matrix[2, 1] - matrix[1, 2]) / scale,
        ])
      elif axis == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = np.array([
          (matrix[0, 1] + matrix[1, 0]) / scale,
          0.25 * scale,
          (matrix[1, 2] + matrix[2, 1]) / scale,
          (matrix[0, 2] - matrix[2, 0]) / scale,
        ])
      else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = np.array([
          (matrix[0, 2] + matrix[2, 0]) / scale,
          (matrix[1, 2] + matrix[2, 1]) / scale,
          0.25 * scale,
          (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-12:
      raise ValueError("invalid rotation matrix produced a zero/non-finite quaternion")
    result[index] = quaternion / norm
  shaped = result.reshape(values.shape[:-2] + (4,))
  # Make each temporal stream continuous.  For fingertip arrays, axis 0 is
  # time and all remaining leading axes identify independent streams.
  if shaped.ndim >= 2 and shaped.shape[0] > 1:
    continuous = shaped.reshape(shaped.shape[0], -1, 4)
    for time_index in range(1, continuous.shape[0]):
      flip = np.sum(continuous[time_index - 1] * continuous[time_index], axis=-1) < 0
      continuous[time_index, flip] *= -1.0
  return shaped


def _matrices_to_poses_xyzw(matrices: np.ndarray) -> np.ndarray:
  values = np.asarray(matrices, dtype=np.float64)
  if values.shape[-2:] != (4, 4):
    raise ValueError(f"SE(3) matrices need trailing shape (4, 4), got {values.shape}")
  rotations = values[..., :3, :3]
  identity = np.eye(3)
  error = np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - identity))
  determinant = np.linalg.det(rotations)
  if error > 1.0e-5 or np.max(np.abs(determinant - 1.0)) > 1.0e-5:
    raise ValueError("pose contains a non-rigid rotation matrix")
  quaternions = _rotation_matrices_to_quaternions_xyzw(rotations)
  return np.concatenate((values[..., :3, 3], quaternions), axis=-1)


def _feature(dtype: str, shape: tuple[int, ...], names: Any = None) -> dict[str, Any]:
  return {"dtype": dtype, "shape": shape, "names": names}


def _features(plan: EpisodePlan, *, use_videos: bool) -> dict[str, dict[str, Any]]:
  visual_dtype = "video" if use_videos else "image"
  image = _feature(
    visual_dtype, (plan.image_height, plan.image_width, 3),
    ["height", "width", "channels"],
  )
  scalar = ["value"]
  features = {
    HEAD_IMAGE_KEY: dict(image),
    RIGHT_WRIST_IMAGE_KEY: dict(image),
    "observation.state": _feature(
      "float32", (27,),
      list(WRIST_POSE_NAMES) + list(RIGHT_HAND_JOINT_NAMES),
    ),
    "observation.state.right_joint_position": _feature(
      "float32", (27,), list(RIGHT_JOINT_NAMES),
    ),
    "observation.state.right_arm_joint_position": _feature(
      "float32", (7,), list(RIGHT_ARM_JOINT_NAMES),
    ),
    "observation.state.right_arm_joint_velocity": _feature(
      "float32", (7,), list(RIGHT_ARM_JOINT_NAMES),
    ),
    "observation.state.right_arm_joint_effort": _feature(
      "float32", (7,), list(RIGHT_ARM_JOINT_NAMES),
    ),
    "observation.state.right_hand_joint_position": _feature(
      "float32", (20,), list(RIGHT_HAND_JOINT_NAMES),
    ),
    "observation.state.right_hand_joint_velocity": _feature(
      "float32", (20,), list(RIGHT_HAND_JOINT_NAMES),
    ),
    "observation.state.right_hand_joint_effort": _feature(
      "float32", (20,), list(RIGHT_HAND_JOINT_NAMES),
    ),
    "observation.state.right_wrist_pose": _feature(
      "float32", (7,), list(WRIST_POSE_NAMES),
    ),
    "observation.state.right_fingertip_pose": _feature("float32", (5, 7)),
    "observation.wrist_wrench.right.local": _feature(
      "float32", (6,), list(WRENCH_NAMES),
    ),
    "observation.wrist_wrench.right.world": _feature(
      "float32", (6,), list(WRENCH_NAMES),
    ),
    "observation.tactile.right.taxel_force": _feature("float32", (5, 7, 5, 3)),
    "observation.tactile.right.contact_count": _feature("int32", (5,), list(FINGER_NAMES)),
    "observation.tactile.right.force_world": _feature("float32", (5, 3)),
    "observation.tactile.right.normal_force": _feature(
      "float32", (5,), list(FINGER_NAMES),
    ),
    "observation.tactile.right.tangent_force": _feature("float32", (5, 2)),
    "observation.tactile.right.normal_axis_world": _feature("float32", (5, 3)),
    "observation.tactile.right.tangent_basis_world": _feature("float32", (5, 2, 3)),
    "action": _feature(
      "float32", (27,),
      list(WRIST_POSE_NAMES) + list(RIGHT_HAND_JOINT_NAMES),
    ),
    "action.right_wrist_pose": _feature("float32", (7,), list(WRIST_POSE_NAMES)),
    "action.right_hand_joint_position": _feature(
      "float32", (20,), list(RIGHT_HAND_JOINT_NAMES),
    ),
    "auxiliary.action.right_arm_joint_target": _feature(
      "float32", (7,), list(RIGHT_ARM_JOINT_NAMES),
    ),
    "auxiliary.action.right_joint_target": _feature(
      "float32", (27,), list(RIGHT_JOINT_NAMES),
    ),
    "auxiliary.observation.right_actuator_control": _feature(
      "float32", (27,), list(RIGHT_JOINT_NAMES),
    ),
    "auxiliary.observation.right_actuator_force": _feature(
      "float32", (27,), list(RIGHT_JOINT_NAMES),
    ),
    "auxiliary.wrist_wrench.right.origin_world": _feature("float32", (3,), ["x_m", "y_m", "z_m"]),
    "auxiliary.wrist_wrench.right.world_from_sensor_rotation": _feature("float32", (3, 3)),
    "auxiliary.tactile_genesis.right.contact": _feature("bool", (5,), list(FINGER_NAMES)),
    "auxiliary.tactile_genesis.right.contact_count": _feature("int32", (5,), list(FINGER_NAMES)),
    "auxiliary.tactile_genesis.right.force_local": _feature("float32", (5, 3)),
    "auxiliary.tactile_genesis.right.force_world": _feature("float32", (5, 3)),
    "auxiliary.tactile_genesis.right.normal_force": _feature("float32", (5,), list(FINGER_NAMES)),
    "auxiliary.tactile_genesis.right.torque_world": _feature("float32", (5, 3)),
    "auxiliary.tactile_genesis.right.centroid_world": _feature("float32", (5, 3)),
    "auxiliary.tactile_genesis.right.centroid_valid": _feature("bool", (5,), list(FINGER_NAMES)),
    "auxiliary.tactile_genesis.right.probe_contact": _feature("bool", (5, 35)),
    "auxiliary.tactile_genesis.right.probe_contact_instantaneous": _feature("bool", (5, 35)),
    "auxiliary.tactile_genesis.right.probe_depth": _feature("float32", (5, 35)),
    "auxiliary.tactile_genesis.right.probe_target_geom_id": _feature("int32", (5, 35)),
    "auxiliary.object.primary.pose": _feature("float32", (7,), list(WRIST_POSE_NAMES)),
    "auxiliary.object.primary.twist": _feature(
      "float32", (6,),
      ["vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"],
    ),
    "auxiliary.camera.head.world_from_camera": _feature("float32", (4, 4)),
    "auxiliary.camera.right_wrist.world_from_camera": _feature("float32", (4, 4)),
    "auxiliary.camera.head.pose_timestamp": _feature("float64", (1,), scalar),
    "auxiliary.camera.right_wrist.pose_timestamp": _feature("float64", (1,), scalar),
    "auxiliary.task.phase_index": _feature("int64", (1,), scalar),
    "provenance.source_episode_index": _feature("int64", (1,), scalar),
    "provenance.source_camera_frame_index": _feature("int64", (1,), scalar),
    "provenance.source_state_index": _feature("int64", (1,), scalar),
    "provenance.source_timestamp": _feature("float64", (1,), scalar),
    "provenance.action_source_camera_frame_index": _feature("int64", (1,), scalar),
    "provenance.action_source_state_index": _feature("int64", (1,), scalar),
    "provenance.action_source_timestamp": _feature("float64", (1,), scalar),
  }
  if not plan.has_fingertip_pose:
    features.pop("observation.state.right_fingertip_pose")
  if not plan.has_actuator_control:
    features.pop("auxiliary.observation.right_actuator_control")
  if not plan.has_actuator_force:
    features.pop("auxiliary.observation.right_actuator_force")
  if not plan.has_tactile_contact_force:
    for key in tuple(features):
      if key.startswith("observation.tactile.right."):
        features.pop(key)

  aggregate_prefix = "auxiliary.tactile_genesis.right."
  aggregate_keys = tuple(
    key for key in features if key.startswith(aggregate_prefix)
  )
  if plan.aggregate_tactile_group != "tactile_genesis":
    aggregate_features = {
      key.removeprefix(aggregate_prefix): features.pop(key)
      for key in aggregate_keys
    }
    if plan.aggregate_tactile_group == "tactile_proxy":
      for suffix, feature in aggregate_features.items():
        if not suffix.startswith("probe_"):
          features[f"auxiliary.tactile_proxy.right.{suffix}"] = feature
  elif not plan.has_tactile_probes:
    for key in aggregate_keys:
      if ".probe_" in key:
        features.pop(key)

  for stream in plan.diagnostics:
    features[stream.output] = _feature(stream.dtype, stream.shape)
  return features


def _take(dataset: h5py.Dataset, indices: np.ndarray, dtype: Any) -> np.ndarray:
  return np.asarray(dataset[indices], dtype=dtype)


def _aligned_arrays(
  file: h5py.File,
  plan: EpisodePlan,
  phase_lookup: dict[str, int],
) -> dict[str, np.ndarray]:
  prefix = plan.strict_prefix_frames
  camera_timestamps = np.asarray(file["cameras/head/timestamp"][:prefix], dtype=np.float64)
  state_indices = np.asarray(file["cameras/head/state_index"][:prefix], dtype=np.int64)
  action_indices = state_indices[1:]
  observation_indices = state_indices[:-1]

  state_names = _decode_names(file["state/joint_names"], context="state/joint_names")
  arm_names = _decode_names(file["commands/arm_joint_names"], context="commands/arm_joint_names")
  hand_names = _decode_names(file["commands/hand_joint_names"], context="commands/hand_joint_names")
  joint_columns = _columns(state_names, RIGHT_JOINT_NAMES, context="state joints")
  arm_columns = _columns(arm_names, RIGHT_ARM_JOINT_NAMES, context="arm commands")
  hand_columns = _columns(hand_names, RIGHT_HAND_JOINT_NAMES, context="hand commands")

  side_names = _decode_names(file["wrist_wrench/side_names"], context="wrist sides")
  right_side = side_names.index("right")

  joint_position = _take(file["state/robot_joint_position"], observation_indices, np.float32)[:, joint_columns]
  joint_velocity = _take(file["state/robot_joint_velocity"], observation_indices, np.float32)[:, joint_columns]
  joint_effort = _take(file["state/robot_joint_effort"], observation_indices, np.float32)[:, joint_columns]

  if plan.has_recorded_wrist_pose:
    wrist_matrices = np.asarray(
      file["cameras/head/world_from_wrist"][:prefix, right_side], dtype=np.float64,
    )
  else:
    wrist_camera_matrices = np.asarray(
      file["cameras/right_wrist/world_from_camera"][:prefix], dtype=np.float64,
    )
    wrist_matrices = wrist_camera_matrices @ RIGHT_WRIST_CAMERA_FROM_WRIST
  wrist_poses = _matrices_to_poses_xyzw(wrist_matrices).astype(np.float32)

  hand_targets = _take(file["commands/hand_joint_target"], action_indices, np.float32)[:, hand_columns]
  arm_targets = _take(file["commands/arm_joint_target"], action_indices, np.float32)[:, arm_columns]
  joint_targets = np.concatenate((arm_targets, hand_targets), axis=1)
  action = np.concatenate((wrist_poses[1:], hand_targets), axis=1)

  local_wrench = np.concatenate((
    _take(file["wrist_wrench/force_local_n"], observation_indices, np.float32)[:, right_side],
    _take(file["wrist_wrench/torque_local_nm"], observation_indices, np.float32)[:, right_side],
  ), axis=1)
  world_wrench = np.concatenate((
    _take(file["wrist_wrench/force_world_n"], observation_indices, np.float32)[:, right_side],
    _take(file["wrist_wrench/torque_world_nm"], observation_indices, np.float32)[:, right_side],
  ), axis=1)

  phases = _decode_string_vector(file["commands/phase"][observation_indices])
  phase_indices = np.asarray([phase_lookup[phase] for phase in phases], dtype=np.int64)

  arrays: dict[str, np.ndarray] = {
    "camera_timestamps": camera_timestamps,
    "state_indices": state_indices,
    "observation.state": np.concatenate((wrist_poses[:-1], joint_position[:, 7:]), axis=1),
    "observation.state.right_joint_position": joint_position,
    "observation.state.right_arm_joint_position": joint_position[:, :7],
    "observation.state.right_arm_joint_velocity": joint_velocity[:, :7],
    "observation.state.right_arm_joint_effort": joint_effort[:, :7],
    "observation.state.right_hand_joint_position": joint_position[:, 7:],
    "observation.state.right_hand_joint_velocity": joint_velocity[:, 7:],
    "observation.state.right_hand_joint_effort": joint_effort[:, 7:],
    "observation.state.right_wrist_pose": wrist_poses[:-1],
    "observation.wrist_wrench.right.local": local_wrench,
    "observation.wrist_wrench.right.world": world_wrench,
    "action": action,
    "action.right_wrist_pose": wrist_poses[1:],
    "action.right_hand_joint_position": hand_targets,
    "auxiliary.action.right_arm_joint_target": arm_targets,
    "auxiliary.action.right_joint_target": joint_targets,
    "auxiliary.wrist_wrench.right.origin_world": _take(
      file["wrist_wrench/origin_world_m"], observation_indices, np.float32,
    )[:, right_side],
    "auxiliary.wrist_wrench.right.world_from_sensor_rotation": _take(
      file["wrist_wrench/world_from_sensor_rotation"], observation_indices, np.float32,
    )[:, right_side],
    "auxiliary.object.primary.pose": _take(
      file[f"objects/{TASK_SPECS[plan.source.task].object_name}/pose_wxyz"],
      observation_indices, np.float32,
    )[:, [0, 1, 2, 4, 5, 6, 3]],
    "auxiliary.object.primary.twist": _take(
      file[f"objects/{TASK_SPECS[plan.source.task].object_name}/twist_linear_angular"],
      observation_indices, np.float32,
    ),
    "auxiliary.camera.head.world_from_camera": np.asarray(
      file["cameras/head/world_from_camera"][:plan.exported_frames], dtype=np.float32,
    ),
    "auxiliary.camera.right_wrist.world_from_camera": np.asarray(
      file["cameras/right_wrist/world_from_camera"][:plan.exported_frames], dtype=np.float32,
    ),
    "auxiliary.camera.head.pose_timestamp": np.asarray(
      (
        file["cameras/head/pose_timestamp"][:plan.exported_frames]
        if "cameras/head/pose_timestamp" in file
        else camera_timestamps[:plan.exported_frames]
      ), dtype=np.float64,
    ),
    "auxiliary.camera.right_wrist.pose_timestamp": np.asarray(
      (
        file["cameras/right_wrist/pose_timestamp"][:plan.exported_frames]
        if "cameras/right_wrist/pose_timestamp" in file
        else camera_timestamps[:plan.exported_frames]
      ), dtype=np.float64,
    ),
    "auxiliary.task.phase_index": phase_indices,
  }

  if plan.has_fingertip_pose:
    fingertip_matrices = np.asarray(
      file["cameras/head/world_from_fingertip"][:prefix, right_side],
      dtype=np.float64,
    )
    arrays["observation.state.right_fingertip_pose"] = (
      _matrices_to_poses_xyzw(fingertip_matrices).astype(np.float32)[:-1]
    )

  if plan.has_actuator_control:
    actuator_names = _decode_names(
      file["commands/actuator_names"], context="commands/actuator_names",
    )
    actuator_columns = _columns(
      actuator_names, RIGHT_JOINT_NAMES, context="actuators",
    )
    arrays["auxiliary.observation.right_actuator_control"] = _take(
      file["commands/actuator_control"], observation_indices, np.float32,
    )[:, actuator_columns]
    if plan.has_actuator_force:
      arrays["auxiliary.observation.right_actuator_force"] = _take(
        file["physics/actuator_force"], observation_indices, np.float32,
      )[:, actuator_columns]

  if plan.has_tactile_contact_force:
    tactile_names = _decode_names(
      file["tactile_contact_force/link_names"], context="tactile links",
    )
    tactile_columns = np.asarray(
      [i for i, name in enumerate(tactile_names) if name.startswith("hand_r_")],
      dtype=np.int64,
    )
    normal_taxel = _take(
      file["tactile_contact_force/normal_taxel_force_n"], observation_indices,
      np.float32,
    )[:, tactile_columns]
    tangent_taxel = _take(
      file["tactile_contact_force/tangent_taxel_force_n"], observation_indices,
      np.float32,
    )[:, tactile_columns]
    arrays.update({
      "observation.tactile.right.taxel_force": np.concatenate(
        (normal_taxel[..., None], tangent_taxel), axis=-1,
      ),
      "observation.tactile.right.contact_count": _take(
        file["tactile_contact_force/contact_count"], observation_indices, np.int32,
      )[:, tactile_columns],
      "observation.tactile.right.force_world": _take(
        file["tactile_contact_force/force_world_n"], observation_indices, np.float32,
      )[:, tactile_columns],
      "observation.tactile.right.normal_force": _take(
        file["tactile_contact_force/normal_force_n"], observation_indices, np.float32,
      )[:, tactile_columns],
      "observation.tactile.right.tangent_force": _take(
        file["tactile_contact_force/tangent_force_n"], observation_indices, np.float32,
      )[:, tactile_columns],
      "observation.tactile.right.normal_axis_world": _take(
        file["tactile_contact_force/normal_axis_world"], observation_indices,
        np.float32,
      )[:, tactile_columns],
      "observation.tactile.right.tangent_basis_world": _take(
        file["tactile_contact_force/tangent_basis_world"], observation_indices,
        np.float32,
      )[:, tactile_columns],
    })

  aggregate_group = plan.aggregate_tactile_group
  if aggregate_group is not None:
    aggregate_names = _decode_names(
      file[f"{aggregate_group}/link_names"], context=f"{aggregate_group} links",
    )
    aggregate_columns = np.asarray(
      [i for i, name in enumerate(aggregate_names) if name.startswith("hand_r_")],
      dtype=np.int64,
    )
    prefix_key = f"auxiliary.{aggregate_group}.right"
    centroid = _take(
      file[f"{aggregate_group}/centroid_world"], observation_indices, np.float32,
    )[:, aggregate_columns]
    centroid_valid = np.all(np.isfinite(centroid), axis=-1)
    centroid = np.where(np.isfinite(centroid), centroid, 0.0).astype(np.float32)
    arrays.update({
      f"{prefix_key}.contact": _take(
        file[f"{aggregate_group}/contact"], observation_indices, np.bool_,
      )[:, aggregate_columns],
      f"{prefix_key}.contact_count": _take(
        file[f"{aggregate_group}/contact_count"], observation_indices, np.int32,
      )[:, aggregate_columns],
      f"{prefix_key}.force_local": _take(
        file[f"{aggregate_group}/force_local"], observation_indices, np.float32,
      )[:, aggregate_columns],
      f"{prefix_key}.force_world": _take(
        file[f"{aggregate_group}/force_world"], observation_indices, np.float32,
      )[:, aggregate_columns],
      f"{prefix_key}.normal_force": _take(
        file[f"{aggregate_group}/normal_force"], observation_indices, np.float32,
      )[:, aggregate_columns],
      f"{prefix_key}.torque_world": _take(
        file[f"{aggregate_group}/torque_world"], observation_indices, np.float32,
      )[:, aggregate_columns],
      f"{prefix_key}.centroid_world": centroid,
      f"{prefix_key}.centroid_valid": centroid_valid,
    })
    if plan.has_tactile_probes:
      link_count = len(aggregate_names)
      for name, dtype in (
        ("probe_contact", np.bool_),
        ("probe_contact_instantaneous", np.bool_),
        ("probe_depth", np.float32),
        ("probe_target_geom_id", np.int32),
      ):
        values = _take(
          file[f"{aggregate_group}/{name}"], observation_indices, dtype,
        ).reshape(plan.exported_frames, link_count, 35)[:, aggregate_columns]
        arrays[f"{prefix_key}.{name}"] = values

  diagnostic_dtypes = {
    "bool": np.bool_,
    "int64": np.int64,
    "float32": np.float32,
  }
  for stream in plan.diagnostics:
    arrays[stream.output] = _take(
      file[stream.source], observation_indices, diagnostic_dtypes[stream.dtype],
    )
  return arrays


def _encode_hdf5_video(
  rgb: h5py.Dataset,
  output: Path,
  *,
  frames: int,
  width: int,
  height: int,
  ffmpeg_threads: int,
  ffmpeg_preset: str,
  video_crf: int,
) -> dict[str, np.ndarray]:
  """Stream HDF5 RGB batches directly to FFmpeg without staging PNG files."""

  from lerobot.datasets.compute_stats import get_feature_stats, sample_indices

  command = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
    "-f", "rawvideo", "-pixel_format", "rgb24",
    "-video_size", f"{width}x{height}", "-framerate", str(FPS),
    "-i", "pipe:0", "-an", "-frames:v", str(frames),
    "-c:v", "libx264", "-preset", ffmpeg_preset, "-crf", str(video_crf),
    "-threads", str(ffmpeg_threads), "-pix_fmt", "yuv420p",
    "-g", "2", "-keyint_min", "2", "-sc_threshold", "0",
    "-movflags", "+faststart", str(output),
  ]
  process = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )
  assert process.stdin is not None
  assert process.stderr is not None
  sampled_indices = np.asarray(sample_indices(frames), dtype=np.int64)
  sampled_images: list[np.ndarray] = []
  maximum_size = max(width, height)
  downsample_factor = (
    (int(width / 150) if width > height else int(height / 150))
    if maximum_size >= 300
    else 1
  )
  try:
    for start in range(0, frames, 32):
      stop = min(start + 32, frames)
      batch = np.ascontiguousarray(
        rgb[start:stop], dtype=np.uint8,
      )
      process.stdin.write(batch.tobytes(order="C"))
      selected = sampled_indices[
        (sampled_indices >= start) & (sampled_indices < stop)
      ]
      sampled_images.extend(
        np.moveaxis(batch[index - start], -1, 0)[
          :, ::downsample_factor, ::downsample_factor
        ].copy()
        for index in selected
      )
    process.stdin.close()
    error_text = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
  except BaseException:
    process.kill()
    process.wait()
    raise
  if return_code != 0:
    raise RuntimeError(
      f"FFmpeg failed for {output} with exit code {return_code}: {error_text[-4000:]}"
    )
  images = np.stack(sampled_images)
  # Cast before squaring inside LeRobot's running statistics; uint8 ** 2 would
  # overflow and incorrectly produce zero image standard deviations.
  stats = get_feature_stats(
    images.astype(np.float32), axis=(0, 2, 3), keepdims=True,
  )
  return {
    key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
    for key, value in stats.items()
  }


def _probe_video(path: Path, *, frames: int, width: int, height: int) -> dict[str, Any]:
  command = [
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries",
    "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration",
    "-of", "json", str(path),
  ]
  result = subprocess.run(command, check=True, capture_output=True, text=True)
  payload = json.loads(result.stdout)
  streams = payload.get("streams", [])
  if len(streams) != 1:
    raise ValueError(f"{path}: expected exactly one video stream")
  stream = streams[0]
  if (
    stream.get("codec_name") != "h264"
    or stream.get("pix_fmt") != "yuv420p"
    or int(stream.get("width", -1)) != width
    or int(stream.get("height", -1)) != height
    or stream.get("avg_frame_rate") != f"{FPS}/1"
    or int(stream.get("nb_frames", -1)) != frames
  ):
    raise ValueError(f"{path}: encoded video contract mismatch: {stream}")
  return stream


def _episode_arrays(
  file: h5py.File,
  plan: EpisodePlan,
  phase_lookup: dict[str, int],
  *,
  output_episode_index: int,
  global_start_index: int,
) -> dict[str, np.ndarray]:
  arrays = _aligned_arrays(file, plan, phase_lookup)
  timestamps = arrays.pop("camera_timestamps")
  state_indices = arrays.pop("state_indices")
  frames = plan.exported_frames
  local_indices = np.arange(frames, dtype=np.int64)
  arrays.update({
    "provenance.source_episode_index": np.full(
      frames, plan.source.episode_index, dtype=np.int64,
    ),
    "provenance.source_camera_frame_index": local_indices,
    "provenance.source_state_index": state_indices[:-1],
    "provenance.source_timestamp": timestamps[:-1],
    "provenance.action_source_camera_frame_index": local_indices + 1,
    "provenance.action_source_state_index": state_indices[1:],
    "provenance.action_source_timestamp": timestamps[1:],
    "timestamp": local_indices.astype(np.float32) / np.float32(FPS),
    "frame_index": local_indices,
    "episode_index": np.full(frames, output_episode_index, dtype=np.int64),
    "index": global_start_index + local_indices,
    "task_index": np.zeros(frames, dtype=np.int64),
  })
  features = _features(plan, use_videos=True)
  for key, feature in features.items():
    if feature["dtype"] == "video":
      continue
    if key not in arrays:
      raise KeyError(f"episode table is missing feature {key}")
    values = np.asarray(arrays[key])
    if feature["shape"] == (1,) and values.shape == (frames, 1):
      values = values[:, 0]
    if values.shape[0] != frames:
      raise ValueError(
        f"{plan.source.hdf5_path}: {key} has {values.shape[0]} rows, expected {frames}"
      )
    arrays[key] = values
  return arrays


def _write_episode_parquet(
  path: Path,
  arrays: dict[str, np.ndarray],
  features: dict[str, dict[str, Any]],
) -> None:
  import datasets
  import pyarrow.parquet as pq
  from lerobot.datasets.utils import DEFAULT_FEATURES, get_hf_features_from_features

  all_features = {**features, **DEFAULT_FEATURES}
  hf_features = get_hf_features_from_features(all_features)
  table_data = {key: arrays[key] for key in hf_features}
  episode = datasets.Dataset.from_dict(
    table_data, features=hf_features, split="train",
  )
  table = episode.with_format("arrow")[:]
  pq.write_table(table, path, compression="snappy", use_dictionary=True)


def _compute_numeric_stats(
  arrays: dict[str, np.ndarray],
  features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
  from lerobot.datasets.compute_stats import compute_episode_stats
  from lerobot.datasets.utils import DEFAULT_FEATURES

  all_features = {**features, **DEFAULT_FEATURES}
  numeric = {
    key: arrays[key]
    for key, feature in all_features.items()
    if feature["dtype"] not in {"video", "image", "string"}
  }
  return compute_episode_stats(numeric, all_features)


def _convert_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
  plan: EpisodePlan = job["plan"]
  output_index = int(job["output_index"])
  work_root = Path(job["work_root"])
  episodes_root = work_root / "episodes"
  episodes_root.mkdir(parents=True, exist_ok=True)
  committed = episodes_root / f"episode-{output_index:06d}"
  partial = episodes_root / (
    f".episode-{output_index:06d}.partial-{os.getpid()}-{uuid.uuid4().hex}"
  )
  partial.mkdir()
  try:
    features = _features(plan, use_videos=True)
    with h5py.File(plan.source.hdf5_path, "r") as file:
      arrays = _episode_arrays(
        file,
        plan,
        job["phase_lookup"],
        output_episode_index=output_index,
        global_start_index=int(job["global_start_index"]),
      )
      _write_episode_parquet(partial / "data.parquet", arrays, features)
      stats = _compute_numeric_stats(arrays, features)
      for video_key, source_name, destination in (
        (HEAD_IMAGE_KEY, "cameras/head/rgb", "head.mp4"),
        (RIGHT_WRIST_IMAGE_KEY, "cameras/right_wrist/rgb", "right_wrist.mp4"),
      ):
        video_path = partial / destination
        stats[video_key] = _encode_hdf5_video(
          file[source_name], video_path,
          frames=plan.exported_frames,
          width=plan.image_width,
          height=plan.image_height,
          ffmpeg_threads=int(job["ffmpeg_threads"]),
          ffmpeg_preset=str(job["ffmpeg_preset"]),
          video_crf=int(job["video_crf"]),
        )
        _probe_video(
          video_path,
          frames=plan.exported_frames,
          width=plan.image_width,
          height=plan.image_height,
        )
      _write_json(partial / "stats.json", stats)
    artifact_names = ("data.parquet", "stats.json", "head.mp4", "right_wrist.mp4")
    artifacts = {
      name: {
        "size_bytes": (partial / name).stat().st_size,
        "sha256": _sha256_file(partial / name),
      }
      for name in artifact_names
    }
    done = {
      "checkpoint_schema": CHECKPOINT_SCHEMA,
      "plan_fingerprint": job["plan_fingerprint"],
      "output_episode_index": output_index,
      "source_episode_index": plan.source.episode_index,
      "source_hdf5_sha256": plan.source.hdf5_sha256,
      "exported_frames": plan.exported_frames,
      "global_start_index": int(job["global_start_index"]),
      "artifacts": artifacts,
      "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(partial / "done.json", done)
    for name in artifact_names:
      _fsync_file(partial / name)
    _fsync_file(partial / "done.json")
    _fsync_directory(partial)
    os.replace(partial, committed)
    _fsync_directory(episodes_root)
    return done
  except BaseException:
    shutil.rmtree(partial, ignore_errors=True)
    raise


def _jsonable(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  return value


def _write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    stream.write(
      json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(temporary, path)
  _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
  try:
    try:
      os.fsync(descriptor)
    except OSError as error:
      if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
        raise
  finally:
    os.close(descriptor)


def _fsync_file(path: Path) -> None:
  with path.open("rb") as stream:
    try:
      os.fsync(stream.fileno())
    except OSError as error:
      if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
        raise


def _schema_metadata(
  first_plan: EpisodePlan,
  phase_names: Sequence[str],
) -> dict[str, Any]:
  spec = TASK_SPECS[first_plan.source.task]
  with h5py.File(first_plan.source.hdf5_path, "r") as file:
    calibration: dict[str, Any] = {
      "head_intrinsic": file["cameras/head/intrinsic"][:],
      "right_wrist_intrinsic": file["cameras/right_wrist/intrinsic"][:],
    }
    if first_plan.has_tactile_contact_force:
      tactile_names = _decode_names(
        file["tactile_contact_force/link_names"], context="tactile links",
      )
      tactile_indices = np.asarray([
        index for index, name in enumerate(tactile_names)
        if name.startswith("hand_r_")
      ])
      calibration.update({
        "right_tactile_link_names": [
          tactile_names[index] for index in tactile_indices
        ],
        "right_tactile_normal_axis_local": (
          file["tactile_contact_force/normal_axis_local"][:][tactile_indices]
        ),
        "right_tactile_tangent_basis_local": (
          file["tactile_contact_force/tangent_basis_local"][:][tactile_indices]
        ),
        "right_tactile_taxel_positions_local_m": (
          file["tactile_contact_force/taxel_positions_local_m"][:][tactile_indices]
        ),
      })
    aggregate_group = first_plan.aggregate_tactile_group
    if aggregate_group is not None:
      aggregate_names = _decode_names(
        file[f"{aggregate_group}/link_names"],
        context=f"{aggregate_group} links",
      )
      aggregate_indices = np.asarray([
        index for index, name in enumerate(aggregate_names)
        if name.startswith("hand_r_")
      ])
      calibration[f"right_{aggregate_group}_link_names"] = [
        aggregate_names[index] for index in aggregate_indices
      ]
      if first_plan.has_tactile_probes:
        # One link name is intentionally repeated for each of its 35 probes.
        probe_link_names = _decode_string_vector(
          file[f"{aggregate_group}/probe_link_names"][:]
        )
        probe_mask = np.asarray([
          name.startswith("hand_r_") for name in probe_link_names
        ])
        for name in ("probe_local_normal", "probe_local_pos", "probe_radius"):
          calibration[f"right_{aggregate_group}_{name}"] = (
            file[f"{aggregate_group}/{name}"][:][probe_mask]
          )

    tactile_semantics: dict[str, Any] = {
      "aggregate_source": aggregate_group,
      "finger_order": list(FINGER_NAMES),
      "missing_modalities_are_omitted_not_zero_filled": True,
    }
    if first_plan.has_tactile_contact_force:
      tactile_semantics["observation.tactile.right.taxel_force"] = {
        "shape": [5, 7, 5, 3],
        "last_axis": ["normal", "grid_col_tangent", "grid_row_tangent"],
        "unit": "N per taxel",
        "warning": "the three chart components are not an SO(3) basis on mirrored hands",
      }

    payload = {
      "schema_version": OUTPUT_SCHEMA,
      "task": {
        "name": spec.name,
        "key": spec.key,
        "instruction": spec.instruction,
        "primary_object": spec.object_name,
      },
      "clock": {
        "fps": FPS,
        "source_state_hz": spec.control_hz,
        "observation": "camera frame with latest-nonfuture source state sample",
        "action": (
          "next retained camera waypoint (30 Hz deadlines rounded up to "
          "100 Hz control ticks; 30/40 ms source intervals)"
          if spec.name == "sponge-grasp"
          else "next retained camera waypoint (1/30 s horizon)"
        ),
        "terminal_rule": "last retained camera frame is excluded because it has no next-frame action",
      },
      "scope": {
        "robot": "right arm and right hand only",
        "cameras": ["head", "right_wrist"],
        "raw_source_remains_authoritative": True,
      },
      "coordinate_conventions": {
        "pose": "[x_m, y_m, z_m, qx, qy, qz, qw]",
        "pose_reference_frame": "MuJoCo world",
        "wrist_wrench_local": "right wrist_ft_site frame",
        "wrist_wrench_world": "MuJoCo world; torque about wrist_ft_site origin",
        "object_source_quaternion": "raw WXYZ converted to output XYZW",
        "wrist_pose_source": (
          "recorded cameras/head/world_from_wrist"
          if first_plan.has_recorded_wrist_pose
          else "recorded cameras/right_wrist/world_from_camera times fixed camera_from_wrist calibration"
        ),
      },
      "action_semantics": {
        "observation.state": "current actual wrist world pose + current actual 20D right-hand joints",
        "observation.state.right_joint_position": "current actual 7D arm + 20D right-hand joints",
        "action": "next wrist actual world pose + next right-hand controller target",
        "action.right_wrist_pose": "next camera-time actual right wrist pose",
        "action.right_hand_joint_position": "right-hand controller target at next camera's latest-nonfuture state",
        "auxiliary.action.right_arm_joint_target": "right-arm controller target at the same action state",
        "auxiliary.action.right_joint_target": "7 arm + 20 hand controller targets at the same action state",
        "warning": "do not compute quaternion-relative actions by component-wise action minus state",
      },
      "tactile_semantics": tactile_semantics,
      "joint_names": {
        "right_arm": list(RIGHT_ARM_JOINT_NAMES),
        "right_hand_actuated": list(RIGHT_HAND_JOINT_NAMES),
      },
      "phase_names": list(phase_names),
      "available_modalities": {
        "recorded_wrist_pose": first_plan.has_recorded_wrist_pose,
        "fingertip_pose": first_plan.has_fingertip_pose,
        "actuator_control": first_plan.has_actuator_control,
        "actuator_force": first_plan.has_actuator_force,
        "tactile_contact_force": first_plan.has_tactile_contact_force,
        "aggregate_tactile_group": aggregate_group,
        "tactile_probes": first_plan.has_tactile_probes,
      },
      "task_diagnostic_streams": [
        {
          "source": stream.source,
          "output": stream.output,
          "dtype": stream.dtype,
          "shape": list(stream.shape),
        }
        for stream in first_plan.diagnostics
      ],
      "calibration": calibration,
      "source_only_not_resampled_into_frame_table": [
        "left arm/hand and left-wrist camera (excluded by requested single-arm scope)",
        "full-rate qpos/qvel and other non-policy physics arrays",
        "ragged per-contact solver events",
        "sparse precontact-noise controller trace",
        "static full-model tables",
      ],
      "training_input_warning": (
        "auxiliary.object.*, auxiliary.task.* and success/phase diagnostics are "
        "privileged simulation signals; do not feed them to a deployable policy unless available at inference"
      ),
    }
    if not first_plan.has_recorded_wrist_pose:
      calibration["right_wrist_camera_from_wrist"] = RIGHT_WRIST_CAMERA_FROM_WRIST
    return payload


def _write_source_manifest(
  path: Path,
  plans: Sequence[EpisodePlan],
  *,
  verify_source_hash: bool,
) -> None:
  with path.open("w", encoding="utf-8") as stream:
    for output_index, plan in enumerate(plans):
      source = plan.source
      record = {
        "output_episode_index": output_index,
        "source_episode_index": source.episode_index,
        "source_hdf5": str(source.hdf5_path),
        "source_relative_hdf5": source.relative_hdf5_path,
        "source_sidecar": str(source.sidecar_path),
        "source_hdf5_sha256": source.hdf5_sha256,
        "source_hash_verification": (
          "recomputed" if verify_source_hash else "trusted_acquisition_sidecar"
        ),
        "source_hdf5_size_bytes": source.hdf5_size_bytes,
        "task": source.task,
        "state_samples": source.state_samples,
        "source_state_hz": TASK_SPECS[source.task].control_hz,
        "camera_samples": source.camera_samples,
        "exported_frames_30hz": plan.exported_frames,
        "off_grid_terminal_frames": plan.off_grid_terminal_frames,
        "object_seed": source.metadata.get("object_seed"),
        "noise_seed": source.metadata.get("noise_seed"),
        "motion_profile": source.metadata.get("motion_profile"),
        "success": source.outcome.get("success"),
      }
      stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _fingerprint(value: Any) -> str:
  encoded = json.dumps(
    _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _plan_record(plan: EpisodePlan, collection_root: Path) -> dict[str, Any]:
  source = plan.source
  hdf5_stat = source.hdf5_path.stat()
  sidecar_stat = source.sidecar_path.stat()
  return {
    "source_episode_index": source.episode_index,
    "relative_hdf5_path": source.relative_hdf5_path,
    "relative_sidecar_path": source.sidecar_path.relative_to(collection_root).as_posix(),
    "hdf5_sha256": source.hdf5_sha256,
    "hdf5_content_sha256": _sha256_file(source.hdf5_path),
    "hdf5_size_bytes": source.hdf5_size_bytes,
    "hdf5_mtime_ns": hdf5_stat.st_mtime_ns,
    "sidecar_size_bytes": sidecar_stat.st_size,
    "sidecar_mtime_ns": sidecar_stat.st_mtime_ns,
    "sidecar_content_sha256": _sha256_file(source.sidecar_path),
    "state_samples": source.state_samples,
    "camera_samples": plan.camera_samples,
    "strict_prefix_frames": plan.strict_prefix_frames,
    "exported_frames": plan.exported_frames,
    "off_grid_terminal_frames": plan.off_grid_terminal_frames,
    "maximum_grid_error_seconds": plan.maximum_grid_error_seconds,
    "image_height": plan.image_height,
    "image_width": plan.image_width,
    "source_timestamp_start_s": plan.source_timestamp_start_s,
    "source_timestamp_end_s": plan.source_timestamp_end_s,
    "phase_names": list(plan.phase_names),
    "source_metadata": {
      key: source.metadata.get(key)
      for key in ("object_seed", "noise_seed", "motion_profile")
    },
    "source_success": source.outcome.get("success"),
    "task": source.task,
    "has_recorded_wrist_pose": plan.has_recorded_wrist_pose,
    "has_fingertip_pose": plan.has_fingertip_pose,
    "has_actuator_control": plan.has_actuator_control,
    "has_actuator_force": plan.has_actuator_force,
    "has_tactile_contact_force": plan.has_tactile_contact_force,
    "aggregate_tactile_group": plan.aggregate_tactile_group,
    "has_tactile_probes": plan.has_tactile_probes,
    "diagnostics": [
      {
        "source": stream.source,
        "output": stream.output,
        "dtype": stream.dtype,
        "shape": list(stream.shape),
      }
      for stream in plan.diagnostics
    ],
  }


def _checkpoint_configuration(
  args: argparse.Namespace,
  collection_root: Path,
) -> dict[str, Any]:
  configuration = {
    "output_schema": OUTPUT_SCHEMA,
    "collection_root": str(collection_root),
    "repo_id": args.repo_id,
    "task": getattr(args, "task", None) or TASK,
    "expected_episodes": args.expected_episodes,
    "limit": args.limit,
    "verify_source_hash": args.verify_source_hash,
    "fps": FPS,
    "video_encoder": {
      "codec": "libx264",
      "pixel_format": "yuv420p",
      "preset": args.ffmpeg_preset,
      "crf": args.video_crf,
      "gop": 2,
      "ffmpeg_threads": args.ffmpeg_threads,
    },
    "video_files_size_in_mb": args.video_files_size_in_mb,
  }
  if (collection_root / "merge_manifest.json").is_file():
    configuration["merge_manifest_json_sha256"] = _sha256_file(
      collection_root / "merge_manifest.json"
    )
  else:
    configuration["collection_json_sha256"] = _sha256_file(
      collection_root / "collection.json"
    )
    configuration["summary_json_sha256"] = _sha256_file(
      collection_root / "summary.json"
    )
  return configuration


def _save_preflight_checkpoint(
  path: Path,
  *,
  args: argparse.Namespace,
  collection_root: Path,
  plans: Sequence[EpisodePlan],
  phase_names: Sequence[str],
) -> tuple[dict[str, Any], str]:
  configuration = _checkpoint_configuration(args, collection_root)
  body = {
    "checkpoint_schema": CHECKPOINT_SCHEMA,
    "configuration": configuration,
    "phase_names": list(phase_names),
    "total_frames": sum(plan.exported_frames for plan in plans),
    "episodes": [_plan_record(plan, collection_root) for plan in plans],
  }
  plan_fingerprint = _fingerprint(body)
  payload = {
    **body,
    "plan_fingerprint": plan_fingerprint,
    "created_utc": datetime.now(timezone.utc).isoformat(),
  }
  _write_json(path, payload)
  return payload, plan_fingerprint


def _load_preflight_checkpoint(
  path: Path,
  *,
  args: argparse.Namespace,
  collection_root: Path,
) -> tuple[list[EpisodePlan], tuple[str, ...], str]:
  payload = _load_object(path)
  if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
    raise ValueError(f"{path}: incompatible checkpoint schema")
  expected_configuration = _checkpoint_configuration(args, collection_root)
  if payload.get("configuration") != expected_configuration:
    raise ValueError(
      f"{path}: conversion configuration changed; use the original arguments "
      "or a different --work-dir"
    )
  body = {
    key: payload[key]
    for key in (
      "checkpoint_schema", "configuration", "phase_names", "total_frames", "episodes",
    )
  }
  plan_fingerprint = _fingerprint(body)
  if payload.get("plan_fingerprint") != plan_fingerprint:
    raise ValueError(f"{path}: preflight checkpoint fingerprint is corrupt")
  records = payload.get("episodes")
  if not isinstance(records, list) or not records:
    raise ValueError(f"{path}: checkpoint has no episode plans")
  plans: list[EpisodePlan] = []
  for record in records:
    if "hdf5_content_sha256" not in record or "sidecar_content_sha256" not in record:
      raise ValueError(
        f"{path}: preflight checkpoint predates source content verification; "
        "start with a new --work-dir"
      )
    hdf5_path = _inside(
      collection_root, record["relative_hdf5_path"], context="checkpoint HDF5",
    )
    sidecar_path = _inside(
      collection_root, record["relative_sidecar_path"], context="checkpoint sidecar",
    )
    hdf5_stat = hdf5_path.stat()
    sidecar_stat = sidecar_path.stat()
    if (
      hdf5_stat.st_size != record["hdf5_size_bytes"]
      or hdf5_stat.st_mtime_ns != record["hdf5_mtime_ns"]
      or sidecar_stat.st_size != record["sidecar_size_bytes"]
      or sidecar_stat.st_mtime_ns != record["sidecar_mtime_ns"]
      or _sha256_file(hdf5_path) != record.get("hdf5_content_sha256")
      or _sha256_file(sidecar_path) != record.get("sidecar_content_sha256")
    ):
      raise ValueError(
        f"raw source changed since preflight: {record['relative_hdf5_path']}"
      )
    source = SourceEpisode(
      episode_index=int(record["source_episode_index"]),
      hdf5_path=hdf5_path,
      sidecar_path=sidecar_path,
      relative_hdf5_path=str(record["relative_hdf5_path"]),
      hdf5_sha256=str(record["hdf5_sha256"]),
      hdf5_size_bytes=int(record["hdf5_size_bytes"]),
      state_samples=int(record["state_samples"]),
      camera_samples=int(record["camera_samples"]),
      metadata=dict(record.get("source_metadata", {})),
      outcome={"success": record.get("source_success")},
      task=str(record["task"]),
    )
    plans.append(EpisodePlan(
      source=source,
      camera_samples=int(record["camera_samples"]),
      strict_prefix_frames=int(record["strict_prefix_frames"]),
      exported_frames=int(record["exported_frames"]),
      off_grid_terminal_frames=int(record["off_grid_terminal_frames"]),
      maximum_grid_error_seconds=float(record["maximum_grid_error_seconds"]),
      image_height=int(record["image_height"]),
      image_width=int(record["image_width"]),
      source_timestamp_start_s=float(record["source_timestamp_start_s"]),
      source_timestamp_end_s=float(record["source_timestamp_end_s"]),
      phase_names=tuple(record["phase_names"]),
      has_recorded_wrist_pose=bool(record["has_recorded_wrist_pose"]),
      has_fingertip_pose=bool(record["has_fingertip_pose"]),
      has_actuator_control=bool(record["has_actuator_control"]),
      has_actuator_force=bool(record["has_actuator_force"]),
      has_tactile_contact_force=bool(record["has_tactile_contact_force"]),
      aggregate_tactile_group=record.get("aggregate_tactile_group"),
      has_tactile_probes=bool(record["has_tactile_probes"]),
      diagnostics=tuple(
        DiagnosticStream(
          source=str(stream["source"]),
          output=str(stream["output"]),
          dtype=str(stream["dtype"]),
          shape=tuple(int(value) for value in stream["shape"]),
        )
        for stream in record.get("diagnostics", [])
      ),
    ))
  if sum(plan.exported_frames for plan in plans) != payload["total_frames"]:
    raise ValueError(f"{path}: total frame count is corrupt")
  return plans, tuple(payload["phase_names"]), plan_fingerprint


def _completed_episode(
  work_root: Path,
  plan: EpisodePlan,
  output_index: int,
  global_start_index: int,
  plan_fingerprint: str,
) -> dict[str, Any] | None:
  root = work_root / "episodes" / f"episode-{output_index:06d}"
  if not root.exists():
    return None
  if not root.is_dir():
    raise ValueError(f"checkpoint path is not a directory: {root}")
  done_path = root / "done.json"
  if not done_path.is_file():
    raise ValueError(f"incomplete committed checkpoint (missing done.json): {root}")
  done = _load_object(done_path)
  expected = {
    "checkpoint_schema": CHECKPOINT_SCHEMA,
    "plan_fingerprint": plan_fingerprint,
    "output_episode_index": output_index,
    "source_episode_index": plan.source.episode_index,
    "source_hdf5_sha256": plan.source.hdf5_sha256,
    "exported_frames": plan.exported_frames,
    "global_start_index": global_start_index,
  }
  for key, value in expected.items():
    if done.get(key) != value:
      raise ValueError(f"{done_path}: checkpoint field {key} does not match plan")
  artifacts = done.get("artifacts")
  if not isinstance(artifacts, dict):
    raise ValueError(f"{done_path}: missing artifact table")
  for name in ("data.parquet", "stats.json", "head.mp4", "right_wrist.mp4"):
    artifact = root / name
    metadata = artifacts.get(name, {})
    if (
      not artifact.is_file()
      or artifact.stat().st_size <= 0
      or artifact.stat().st_size != metadata.get("size_bytes")
      or _sha256_file(artifact) != metadata.get("sha256")
    ):
      raise ValueError(f"{done_path}: checkpoint artifact is invalid: {name}")
  return done


def _load_episode_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
  raw = _load_object(path)
  return {
    feature: {
      statistic: np.asarray(value)
      for statistic, value in feature_stats.items()
    }
    for feature, feature_stats in raw.items()
  }


def _link_or_copy(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  try:
    os.link(source, destination)
  except OSError as error:
    if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
      raise
    shutil.copy2(source, destination)


def _validate_output(root: Path, expected_episodes: int, expected_frames: int) -> None:
  info = _load_object(root / "meta/info.json")
  if info.get("codebase_version") != "v3.0":
    raise ValueError(f"{root}: output is not LeRobot v3.0")
  if info.get("total_episodes") != expected_episodes:
    raise ValueError(f"{root}: episode total mismatch")
  if info.get("total_frames") != expected_frames:
    raise ValueError(f"{root}: frame total mismatch")
  if info.get("fps") != FPS:
    raise ValueError(f"{root}: FPS mismatch")
  features = info.get("features", {})
  for key in (
    HEAD_IMAGE_KEY,
    RIGHT_WRIST_IMAGE_KEY,
    "observation.state",
    "observation.wrist_wrench.right.local",
    "action",
    "action.right_wrist_pose",
    "action.right_hand_joint_position",
  ):
    if key not in features:
      raise ValueError(f"{root}: missing output feature {key}")
  if not any((root / "data").rglob("*.parquet")):
    raise ValueError(f"{root}: no data parquet was produced")
  if info.get("video_path") is not None and not any((root / "videos").rglob("*.mp4")):
    raise ValueError(f"{root}: no MP4 video was produced")


def _publish(staging: Path, output: Path) -> None:
  if output.exists() or output.is_symlink():
    raise FileExistsError(f"output already exists: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  try:
    os.replace(staging, output)
    return
  except OSError as error:
    if error.errno != errno.EXDEV:
      raise
  transfer = output.parent / f".{output.name}.publish-{uuid.uuid4().hex}"
  shutil.copytree(staging, transfer)
  _validate_output(
    transfer,
    _load_object(staging / "meta/info.json")["total_episodes"],
    _load_object(staging / "meta/info.json")["total_frames"],
  )
  os.replace(transfer, output)
  shutil.rmtree(staging)


def _load_lerobot() -> tuple[Any, Any]:
  try:
    from lerobot.datasets.lerobot_dataset import (
      CODEBASE_VERSION,
      LeRobotDataset,
      LeRobotDatasetMetadata,
    )
  except ImportError as error:
    raise RuntimeError(
      "LeRobot is not importable. Run with an environment containing lerobot 0.4.x, "
      "for example /cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python"
    ) from error
  if CODEBASE_VERSION != "v3.0":
    raise RuntimeError(f"expected LeRobot codebase v3.0, imported {CODEBASE_VERSION!r}")
  return LeRobotDataset, LeRobotDatasetMetadata


def _preflight_plans(
  sources: Sequence[SourceEpisode],
  workers: int,
) -> list[EpisodePlan]:
  results: list[EpisodePlan | None] = [None] * len(sources)
  executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
  try:
    future_to_index = {
      executor.submit(_preflight_episode, source): index
      for index, source in enumerate(sources)
    }
    completed = 0
    for future in concurrent.futures.as_completed(future_to_index):
      index = future_to_index[future]
      results[index] = future.result()
      completed += 1
      source = sources[index]
      print(
        f"preflight [{completed}/{len(sources)}] "
        f"outer={source.episode_index:06d} {source.relative_hdf5_path}",
        flush=True,
      )
  except BaseException:
    for future in future_to_index:
      future.cancel()
    for process in getattr(executor, "_processes", {}).values():
      process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  else:
    executor.shutdown(wait=True)
  return [result for result in results if result is not None]


def _compact_plan(plan: EpisodePlan) -> EpisodePlan:
  source = plan.source
  compact_source = replace(
    source,
    metadata={
      key: source.metadata.get(key)
      for key in ("object_seed", "noise_seed", "motion_profile")
    },
    outcome={"success": source.outcome.get("success")},
  )
  return replace(plan, source=compact_source)


def _validate_plan_contracts(plans: Sequence[EpisodePlan]) -> None:
  if not plans:
    raise ValueError("conversion plan is empty")
  expected_task = plans[0].source.task
  expected_features = _features(plans[0], use_videos=True)
  expected_diagnostics = plans[0].diagnostics
  for plan in plans[1:]:
    if plan.source.task != expected_task:
      raise ValueError("one LeRobot dataset may contain only one source task")
    if _features(plan, use_videos=True) != expected_features:
      raise ValueError(
        f"{plan.source.hdf5_path}: available modality/feature contract differs "
        "from the first episode"
      )
    if plan.diagnostics != expected_diagnostics:
      raise ValueError(
        f"{plan.source.hdf5_path}: task diagnostic streams differ from the first episode"
      )


def _run_episode_workers(
  plans: Sequence[EpisodePlan],
  *,
  args: argparse.Namespace,
  work_root: Path,
  phase_lookup: dict[str, int],
  plan_fingerprint: str,
) -> None:
  offsets: list[int] = []
  running = 0
  for plan in plans:
    offsets.append(running)
    running += plan.exported_frames

  episodes_root = work_root / "episodes"
  episodes_root.mkdir(parents=True, exist_ok=True)
  for abandoned in episodes_root.glob(".episode-*.partial-*"):
    if abandoned.is_dir():
      shutil.rmtree(abandoned)

  pending: list[tuple[int, EpisodePlan, int]] = []
  for output_index, (plan, offset) in enumerate(zip(plans, offsets, strict=True)):
    if _completed_episode(
      work_root, plan, output_index, offset, plan_fingerprint,
    ) is None:
      pending.append((output_index, plan, offset))
  done_count = len(plans) - len(pending)
  print(
    f"resume: completed={done_count} pending={len(pending)} workers={args.workers} "
    f"ffmpeg_threads_per_worker={args.ffmpeg_threads}",
    flush=True,
  )
  if not pending:
    return

  executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
  try:
    futures = {}
    for output_index, plan, offset in pending:
      job = {
        "plan": plan,
        "output_index": output_index,
        "global_start_index": offset,
        "work_root": str(work_root),
        "phase_lookup": phase_lookup,
        "plan_fingerprint": plan_fingerprint,
        "ffmpeg_threads": args.ffmpeg_threads,
        "ffmpeg_preset": args.ffmpeg_preset,
        "video_crf": args.video_crf,
      }
      futures[executor.submit(_convert_episode_worker, job)] = (
        output_index, plan.source.episode_index,
      )
    converted = 0
    for future in concurrent.futures.as_completed(futures):
      output_index, source_index = futures[future]
      result = future.result()
      converted += 1
      print(
        f"convert [{done_count + converted}/{len(plans)}] "
        f"output={output_index:06d} outer={source_index:06d} "
        f"frames={result['exported_frames']}",
        flush=True,
      )
  except BaseException:
    for future in futures:
      future.cancel()
    for process in getattr(executor, "_processes", {}).values():
      process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  else:
    executor.shutdown(wait=True)


def _assemble_dataset(
  plans: Sequence[EpisodePlan],
  *,
  args: argparse.Namespace,
  collection_root: Path,
  work_root: Path,
  phase_names: Sequence[str],
  plan_fingerprint: str,
  staging: Path,
) -> None:
  spec = TASK_SPECS[plans[0].source.task]
  LeRobotDataset, LeRobotDatasetMetadata = _load_lerobot()
  features = _features(plans[0], use_videos=True)
  metadata = LeRobotDatasetMetadata.create(
    repo_id=args.repo_id,
    fps=FPS,
    root=staging,
    robot_type="kaihand_right",
    features=features,
    use_videos=True,
    chunks_size=1000,
    data_files_size_in_mb=100,
    video_files_size_in_mb=args.video_files_size_in_mb,
  )
  metadata.save_episode_tasks([spec.instruction])

  global_start = 0
  for output_index, plan in enumerate(plans):
    checkpoint = work_root / "episodes" / f"episode-{output_index:06d}"
    chunk_index, file_index = divmod(output_index, 1000)
    _link_or_copy(
      checkpoint / "data.parquet",
      staging / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    )
    for key, name in (
      (HEAD_IMAGE_KEY, "head.mp4"),
      (RIGHT_WRIST_IMAGE_KEY, "right_wrist.mp4"),
    ):
      _link_or_copy(
        checkpoint / name,
        staging / (
          f"videos/{key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
      )
    duration = plan.exported_frames / FPS
    episode_metadata = {
      "data/chunk_index": chunk_index,
      "data/file_index": file_index,
      "dataset_from_index": global_start,
      "dataset_to_index": global_start + plan.exported_frames,
    }
    for key in (HEAD_IMAGE_KEY, RIGHT_WRIST_IMAGE_KEY):
      episode_metadata.update({
        f"videos/{key}/chunk_index": chunk_index,
        f"videos/{key}/file_index": file_index,
        f"videos/{key}/from_timestamp": 0.0,
        f"videos/{key}/to_timestamp": duration,
      })
    metadata.save_episode(
      episode_index=output_index,
      episode_length=plan.exported_frames,
      episode_tasks=[spec.instruction],
      episode_stats=_load_episode_stats(checkpoint / "stats.json"),
      episode_metadata=episode_metadata,
    )
    global_start += plan.exported_frames
  metadata._close_writer()
  metadata.update_video_info()
  _write_json(staging / "meta/info.json", metadata.info)

  _write_json(staging / "meta/kaihand_schema.json", _schema_metadata(plans[0], phase_names))
  _write_source_manifest(
    staging / "meta/kaihand_source_episodes.jsonl",
    plans,
    verify_source_hash=args.verify_source_hash,
  )
  _write_json(
    staging / "meta/kaihand_conversion.json",
    {
      "schema_version": OUTPUT_SCHEMA,
      "checkpoint_schema": CHECKPOINT_SCHEMA,
      "plan_fingerprint": plan_fingerprint,
      "created_utc": datetime.now(timezone.utc).isoformat(),
      "source_collection": str(collection_root),
      "task": spec.name,
      "task_instruction": spec.instruction,
      "primary_object": spec.object_name,
      "repo_id": args.repo_id,
      "lerobot_codebase_version": "v3.0",
      "episodes": [
        {
          "output_episode_index": output_index,
          "source_episode_index": plan.source.episode_index,
          "exported_frames": plan.exported_frames,
        }
        for output_index, plan in enumerate(plans)
      ],
      "total_episodes": len(plans),
      "total_frames": global_start,
      "source_hash_verification": (
        "recomputed" if args.verify_source_hash else "trusted_acquisition_sidecar"
      ),
      "videos": True,
      "video_storage": "one MP4 per camera per episode; HDF5 RGB streamed directly to FFmpeg",
      "parallel_episode_workers": args.workers,
      "ffmpeg_threads_per_worker": args.ffmpeg_threads,
      "video_files_size_in_mb": args.video_files_size_in_mb,
    },
  )
  _validate_output(staging, len(plans), global_start)

  dataset = LeRobotDataset(
    repo_id=args.repo_id,
    root=staging,
    video_backend="pyav",
  )
  if dataset.num_episodes != len(plans) or dataset.num_frames != global_start:
    raise ValueError(f"{staging}: official LeRobot reader totals disagree")
  probe_indices = sorted({0, global_start // 2, global_start - 1})
  for index in probe_indices:
    sample = dataset[index]
    expected_shape = (3, plans[0].image_height, plans[0].image_width)
    for key in (HEAD_IMAGE_KEY, RIGHT_WRIST_IMAGE_KEY):
      if tuple(sample[key].shape) != expected_shape:
        raise ValueError(f"{staging}: official reader decoded an invalid {key} image")


def _existing_output_matches(
  output: Path,
  *,
  collection_root: Path,
  repo_id: str,
) -> bool:
  conversion_path = output / "meta/kaihand_conversion.json"
  if not conversion_path.is_file():
    return False
  conversion = _load_object(conversion_path)
  if (
    conversion.get("source_collection") != str(collection_root)
    or conversion.get("repo_id") != repo_id
    or conversion.get("schema_version") != OUTPUT_SCHEMA
  ):
    return False
  _validate_output(
    output,
    int(conversion["total_episodes"]),
    int(conversion["total_frames"]),
  )
  return True


def run(args: argparse.Namespace) -> Path | None:
  collection_root = _resolve_collection_root(args.input_dir)
  args.task = _detect_task(collection_root, getattr(args, "task", None))
  if args.repo_id is None:
    collection_name = collection_root.name.replace("-", "_")
    task_name = args.task.replace("-", "_")
    args.repo_id = f"kaihand/{task_name}_{collection_name}"
  output = args.output_dir.expanduser().resolve()
  if output == collection_root or output.is_relative_to(collection_root):
    raise ValueError("output must not be inside the immutable raw collection")
  if output.exists() or output.is_symlink():
    if args.resume and output.is_dir() and _existing_output_matches(
      output, collection_root=collection_root, repo_id=args.repo_id,
    ):
      print(f"already published and valid: {output}", flush=True)
      return output
    raise FileExistsError(f"output already exists: {output}")

  if args.validate_only:
    sources = _discover_sources(
      collection_root,
      task=args.task,
      expected_episodes=args.expected_episodes,
      verify_source_hash=args.verify_source_hash,
    )
    selected = sources if args.limit is None else sources[:args.limit]
    if not selected:
      raise ValueError("no successful source episodes selected")
    plans = _preflight_plans(selected, args.workers)
    _validate_plan_contracts(plans)
    phase_names = tuple(
      dict.fromkeys(phase for plan in plans for phase in plan.phase_names)
    )
    print(
      f"plan: collection={collection_root} episodes={len(plans)} "
      f"frames={sum(plan.exported_frames for plan in plans)} fps={FPS} "
      f"phases={len(phase_names)}",
      flush=True,
    )
    return None

  if args.work_dir is not None:
    work_root = args.work_dir.expanduser().resolve()
  elif args.staging_root is not None:
    work_root = (
      args.staging_root.expanduser().resolve() / f".{output.name}.resume-work"
    )
  else:
    work_root = output.parent / f".{output.name}.resume-work"
  if work_root == collection_root or work_root.is_relative_to(collection_root):
    raise ValueError("work directory must not be inside the immutable raw collection")

  checkpoint_path = work_root / "preflight_plan.json"
  if work_root.exists() and not args.resume:
    raise FileExistsError(
      f"work directory already exists: {work_root}; pass --resume to reuse it"
    )
  if checkpoint_path.is_file():
    if not args.resume:
      raise FileExistsError(f"preflight checkpoint already exists: {checkpoint_path}")
    print(f"resume: loading saved preflight plan {checkpoint_path}", flush=True)
    plans, phase_names, plan_fingerprint = _load_preflight_checkpoint(
      checkpoint_path,
      args=args,
      collection_root=collection_root,
    )
    print(
      "resume: raw control files and source content hashes match; "
      "skipping HDF5 structural preflight",
      flush=True,
    )
  else:
    if work_root.exists() and any(work_root.iterdir()):
      raise ValueError(
        f"{work_root}: non-empty work directory has no preflight checkpoint"
      )
    work_root.mkdir(parents=True, exist_ok=True)
    sources = _discover_sources(
      collection_root,
      task=args.task,
      expected_episodes=args.expected_episodes,
      verify_source_hash=args.verify_source_hash,
    )
    selected = sources if args.limit is None else sources[:args.limit]
    if not selected:
      raise ValueError("no successful source episodes selected")
    print(
      f"preflight: episodes={len(selected)} workers={args.workers}", flush=True,
    )
    plans = _preflight_plans(selected, args.workers)
    _validate_plan_contracts(plans)
    image_shapes = {(plan.image_height, plan.image_width) for plan in plans}
    if len(image_shapes) != 1:
      raise ValueError(f"source episodes have inconsistent image shapes: {image_shapes}")
    phase_names = tuple(
      dict.fromkeys(phase for plan in plans for phase in plan.phase_names)
    )
    _, plan_fingerprint = _save_preflight_checkpoint(
      checkpoint_path,
      args=args,
      collection_root=collection_root,
      plans=plans,
      phase_names=phase_names,
    )
    print(f"saved preflight checkpoint: {checkpoint_path}", flush=True)

  _validate_plan_contracts(plans)
  plans = [_compact_plan(plan) for plan in plans]
  total_frames = sum(plan.exported_frames for plan in plans)
  phase_lookup = {name: index for index, name in enumerate(phase_names)}
  print(
    f"plan: collection={collection_root} episodes={len(plans)} "
    f"frames={total_frames} fps={FPS} phases={len(phase_names)}",
    flush=True,
  )

  staging = work_root / f"final.partial-{os.getpid()}-{uuid.uuid4().hex}"
  try:
    _run_episode_workers(
      plans,
      args=args,
      work_root=work_root,
      phase_lookup=phase_lookup,
      plan_fingerprint=plan_fingerprint,
    )
    for abandoned in work_root.glob("final.partial-*"):
      if abandoned != staging and abandoned.is_dir():
        shutil.rmtree(abandoned)
    print("assemble: building standard LeRobot v3 metadata and layout", flush=True)
    _assemble_dataset(
      plans,
      args=args,
      collection_root=collection_root,
      work_root=work_root,
      phase_names=phase_names,
      plan_fingerprint=plan_fingerprint,
      staging=staging,
    )
    _publish(staging, output)
    _validate_output(output, len(plans), total_frames)
  except BaseException:
    print(
      f"conversion interrupted/failed; resumable checkpoints retained at {work_root}",
      file=sys.stderr,
    )
    raise
  if not args.keep_work_dir:
    shutil.rmtree(work_root)
  print(f"published LeRobot v3 dataset: {output}", flush=True)
  return output


def main(argv: Sequence[str] | None = None) -> int:
  run(_parse_args(argv))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
