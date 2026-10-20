#!/usr/bin/env python3
"""Convert canonical KaiHand Raw episodes to FTP-1 Zarr shards.

The converter supports explicitly registered right-hand collection contracts. It
uses the recorded 30 Hz head-camera clock, follows each camera frame's
latest-nonfuture ``state_index``, and writes time-major state/tactile arrays.
FTP-1 constructs its current-frame-plus-31-future-frame action chunks from
these trajectories at load time; this converter intentionally does not write
a precomputed action array.

Run this script with an environment that provides zarr 3.x and numcodecs.  The
FTP-1 repository's ``.venv`` is a suitable environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kaihand_tactile_env.shared.ftp1_policy_contract import (
  RIGHT_ARM_JOINT_NAMES,
  RIGHT_HAND_JOINT_NAMES,
)
from kaihand_tactile_env.shared.ftp1_policy_contract import (
  RIGHT_HAND_FAAS_INDICES as _RIGHT_HAND_FAAS_INDICES,
)

FPS = 30
FTP1_ACTION_HORIZON = 32
RAW_SCHEMA = "kaihand_tactile_episode_v1"
MANIFEST_SCHEMA = "kaihand_raw_to_ftp1_zarr_v2"
TASK = "poker-draw"
TASK_INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the "
  "fingers and thumb, lift it, and turn its face toward the robot to look at it."
)
TACTILE_SENSOR = "KaiHandSimTaxelV1"


@dataclass(frozen=True)
class TaskContract:
  instruction: str
  model_fingerprint: str | None = None
  mechanics_version: str | None = None
  completion_protocol: str | None = None


# These describe the published Raw datasets, not whichever task source happens
# to be installed at conversion time. In particular, the old USB demonstrations
# include returning/releasing the plug and must not be relabelled as held-clear.
TASK_CONTRACTS = {
  TASK: TaskContract(TASK_INSTRUCTION),
  "bulb-screw": TaskContract(
    "Grasp the light bulb with the right hand, lift it to the socket, screw it in clockwise, release it, and withdraw the hand.",
    "cac34f5fc061c4d854be34563a312e83b9352dd2629a4215908fcbf778eb1d46",
    "phase_aware_strict_completion_v15",
  ),
  "bulb-screw-inplace": TaskContract(
    "Grasp the light bulb already on the socket with the right hand, screw it in clockwise until seated, release it, and withdraw the hand.",
    "66b7cfdbee4d5dde4fdde8d4946e531e2e781d31b55da5baf9ccfdaf45d787bf",
    "bulb_screw_inplace_inherits_phase_aware_strict_completion_v15_v1",
  ),
  "bulb-unscrew": TaskContract(
    "Grasp the installed light bulb with the right hand, unscrew it, return it to the table, release it, and withdraw the hand.",
    "752deb9287d401d1810dc0d790992dd373e7a0679a7beece0650da171a1ce79c",
    "bulb_unscrew_inherits_phase_aware_strict_completion_v15_reverse_tabletop_v2",
  ),
  "install-ram": TaskContract(
    "Grasp the RAM module with the right hand, align it with the memory slot, insert it fully, release it, and withdraw the hand.",
    "cf48ce4d47c5385f9f52bcf7f82f596db299b497edc3a1c4347a87d5a9eb7c46",
    "aligned_axial_insertion_frozen_grip_v3",
  ),
  "usb-insert": TaskContract(
    "Grasp the USB plug with the right hand, align it with the socket, insert it fully, release it, and withdraw the hand.",
    "204874dc823489804fd9759e2fc34f86758ef31f7e8707d5e5e48096eebb80a9",
  ),
  "usb-unplug": TaskContract(
    "Grasp the already inserted USB plug with the right hand, pull it straight out of the socket, return it to the table, release it, and withdraw the hand.",
    "f65ca619353297c385bb618be78891ea4847c66b5e515a93b1f5fb31069602d4",
    "usb_unplug_passive_spring_v1",
    "usb-unplug-return-release-v1",
  ),
  "whiteboard-wipe": TaskContract(
    "Grasp the eraser with the right hand, wipe the marks off the inclined whiteboard, return the eraser to the table, release it, and withdraw the hand.",
    "4d50170be5dee72ef49e367504b0fa67d2d462d2907f646e8455718ac606599d",
  ),
  "whiteboard-wipe-horizontal": TaskContract(
    "Grasp the eraser with the right hand, wipe the marks off the horizontal whiteboard, lift the eraser, return it to the table, release it, and withdraw the hand.",
    "ab370f245a167f08b118bfa36815a62eab292355324ee2e75ea8f0458e6ecd02",
    "horizontal_fixed_board_z_contact_v1",
  ),
  "whiteboard-wipe-horizontal-line": TaskContract(
    "Grasp the eraser with the right hand, wipe the single horizontal line off the horizontal whiteboard, lift the eraser, return it to the table, release it, and withdraw the hand.",
    "88fa5c1593a958231ec30eb50ff3a43889460df75c47c6839a0922fce3478ffb",
    "horizontal_fixed_board_top_line_contact_v1",
  ),
}


def _task_contract(task: str) -> TaskContract:
  if task not in TASK_CONTRACTS:
    raise ValueError(f"Unsupported FTP-1 Raw task: {task!r}")
  return TASK_CONTRACTS[task]


def _instruction(task: str, instruction: str | None) -> str:
  expected = _task_contract(task).instruction
  value = expected if instruction is None else instruction.strip()
  if not value or len(value) > 255:
    raise ValueError("instruction must contain 1-255 characters")
  # Preserve the original poker CLI's custom instruction support. For newly
  # registered datasets, fail closed rather than silently changing task goals.
  if task != TASK and value != expected:
    raise ValueError(f"{task}: instruction must match the recorded task protocol")
  return value


# The KaiHand order above follows the same anatomical order as FTP-1's
# MANO-derived FAAS definition, without its terminal palm channel.  In
# particular, the four thumb actuators map to palm-adduction,
# thumb-adduction, thumb-MCP and thumb-IP slots respectively.
RIGHT_HAND_FAAS_INDICES = np.asarray(_RIGHT_HAND_FAAS_INDICES, dtype=np.int32)

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
RIGHT_TACTILE_LINK_NAMES = tuple(
  f"hand_r_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
  for finger in FINGER_NAMES
)

EXPECTED_DATA_KEYS = (
  "timestamps",
  "source_episode_index",
  "source_camera_index",
  "source_state_index",
  "camera_ego_rgb",
  "right_wrist_camera_rgb",
  "camera_ego_pose",
  "right_wrist_pose",
  "right_arm_joints",
  "right_hand_joints",
  "right_hand_joints_idx",
  "right_tactile_data_fingers",
  "right_tactile_area_fingers",
  "right_tactile_sensor_fingers",
  "right_tactile_type_fingers",
  "sub_task_instruction",
)


@dataclass(frozen=True)
class SourceEpisode:
  episode_index: int
  hdf5_path: Path
  relative_hdf5_path: str
  sidecar_path: Path
  relative_sidecar_path: str
  size_bytes: int
  mtime_ns: int
  declared_sha256: str
  source_environment: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConvertedEpisode:
  arrays: dict[str, np.ndarray]
  source: SourceEpisode
  strict_camera_frames: int
  omitted_terminal_camera_frames: int
  maximum_camera_grid_error_seconds: float


@dataclass(frozen=True)
class ShardResult:
  shard_index: int
  shard_record: dict[str, Any]
  episode_records: tuple[dict[str, Any], ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Raw collection root containing summary.json (for example raw/0920_200).",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="FTP-1 domain directory; Zarr shards are direct children of this path.",
  )
  parser.add_argument("--task", choices=tuple(TASK_CONTRACTS), default=TASK)
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--expected-episodes", type=int, default=None)
  parser.add_argument("--episodes-per-shard", type=int, default=20)
  parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of independent Zarr shards to convert concurrently.",
  )
  parser.add_argument("--instruction", default=None)
  parser.add_argument("--sensor-name", default=TACTILE_SENSOR)
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Validate and reuse already published shards with matching episode indices.",
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help=(
      "Recompute each HDF5 SHA-256 and compare it with the capture sidecar. "
      "This adds one complete source-file read before conversion."
    ),
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate all selected source metadata/clocks/shapes without reading RGB pixels or writing output.",
  )
  parser.add_argument(
    "--plan-json",
    action="store_true",
    help="Print the metadata-preflight plan as JSON (requires --dry-run).",
  )
  return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def _inside(root: Path, relative: str, *, context: str) -> Path:
  candidate = (root / relative).resolve()
  resolved_root = root.resolve()
  try:
    candidate.relative_to(resolved_root)
  except ValueError as exc:
    raise ValueError(f"{context}: path escapes collection root: {relative}") from exc
  return candidate


def discover_sources(
  collection_root: Path, task: str = TASK
) -> tuple[SourceEpisode, ...]:
  _task_contract(task)
  summary_path = collection_root / "summary.json"
  if not summary_path.is_file():
    raise FileNotFoundError(f"Raw collection summary not found: {summary_path}")
  summary = _load_json(summary_path)
  target_met = summary.get("target_met", False)
  if isinstance(target_met, bool):
    task_target_met = target_met
  elif isinstance(target_met, dict):
    task_target_met = target_met.get(task, False)
  else:
    raise ValueError(f"{summary_path}: target_met must be a bool or object")
  if task_target_met is not True:
    raise ValueError(f"{summary_path}: target_met[{task!r}] is not true")
  records = summary.get("episodes")
  if not isinstance(records, list):
    raise ValueError(f"{summary_path}: episodes must be a list")

  sources: list[SourceEpisode] = []
  seen_indices: set[int] = set()
  for record in records:
    if not isinstance(record, dict):
      raise ValueError(f"{summary_path}: episode record is not an object")
    if record.get("task") != task or record.get("status") != "success":
      continue
    episode_index = int(record["episode_index"])
    if episode_index in seen_indices:
      raise ValueError(f"{summary_path}: duplicate successful episode {episode_index}")
    hdf5_entries = record.get("hdf5")
    if not isinstance(hdf5_entries, list) or len(hdf5_entries) != 1:
      raise ValueError(
        f"{summary_path}: episode {episode_index} must publish exactly one HDF5 file"
      )
    relative = str(hdf5_entries[0])
    hdf5_path = _inside(
      collection_root,
      relative,
      context=f"episode {episode_index}",
    )
    if not hdf5_path.is_file():
      raise FileNotFoundError(f"Published HDF5 does not exist: {hdf5_path}")
    sidecar_path = _inside(
      collection_root,
      str(hdf5_path.with_suffix(".json").relative_to(collection_root.resolve())),
      context=f"episode {episode_index} sidecar",
    )
    if not sidecar_path.is_file():
      raise FileNotFoundError(
        f"Published capture sidecar does not exist: {sidecar_path}"
      )
    sidecar = _load_json(sidecar_path)
    if sidecar.get("episode") != hdf5_path.name:
      raise ValueError(f"{sidecar_path}: episode does not name {hdf5_path.name!r}")
    declared_sha256 = str(sidecar.get("sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None:
      raise ValueError(f"{sidecar_path}: missing or invalid sha256")
    source_environment = None
    if task != TASK:
      with h5py.File(hdf5_path, "r") as file:
        source_environment = _validate_source_metadata(file, task=task)
        if sidecar.get("outcome") != _json_attr(file, "outcome_json"):
          raise ValueError(f"{sidecar_path}: outcome differs from HDF5 outcome")
    stat = hdf5_path.stat()
    seen_indices.add(episode_index)
    sources.append(
      SourceEpisode(
        episode_index=episode_index,
        hdf5_path=hdf5_path,
        relative_hdf5_path=relative,
        sidecar_path=sidecar_path,
        relative_sidecar_path=str(sidecar_path.relative_to(collection_root)),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        declared_sha256=declared_sha256,
        source_environment=source_environment,
      )
    )

  sources.sort(key=lambda source: source.episode_index)
  if not sources:
    raise ValueError(f"{summary_path}: no successful {task} episodes")
  return tuple(sources)


def _decode_scalar(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return str(value)


def _decode_names(dataset: h5py.Dataset, *, context: str) -> tuple[str, ...]:
  values = tuple(_decode_scalar(value) for value in dataset[:])
  if len(values) != len(set(values)):
    raise ValueError(f"{context}: names must be unique")
  return values


def _columns(
  names: Sequence[str],
  selected: Sequence[str],
  *,
  context: str,
) -> np.ndarray:
  missing = [name for name in selected if name not in names]
  if missing:
    raise ValueError(f"{context}: missing names: {missing}")
  return np.asarray([names.index(name) for name in selected], dtype=np.int64)


def _require_dataset(file: h5py.File, name: str) -> h5py.Dataset:
  if name not in file or not isinstance(file[name], h5py.Dataset):
    raise ValueError(f"{file.filename}: missing required dataset {name}")
  return file[name]


def _json_attr(file: h5py.File | h5py.Group, name: str) -> dict[str, Any]:
  if name not in file.attrs:
    raise ValueError(f"{file.file.filename}: missing HDF5 attribute {name}")
  raw = file.attrs[name]
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8")
  value = json.loads(str(raw))
  if not isinstance(value, dict):
    raise ValueError(f"{file.file.filename}: {name} must encode a JSON object")
  return value


def _validate_source_metadata(file: h5py.File, *, task: str) -> dict[str, Any]:
  contract = _task_contract(task)
  metadata = _json_attr(file, "metadata_json")
  outcome = _json_attr(file, "outcome_json")
  if metadata.get("scene") != task:
    raise ValueError(f"{file.filename}: source scene does not match {task}")
  side = metadata.get("side")
  if side not in (None, "right") or (task in (TASK, "usb-insert") and side != "right"):
    raise ValueError(f"{file.filename}: expected right-hand {task} metadata")
  if outcome.get("success") is not True:
    raise ValueError(f"{file.filename}: episode outcome is not successful")
  if task == TASK:
    if outcome.get("task_completed") is not True:
      raise ValueError(f"{file.filename}: episode outcome is not successful")
    return {}
  fingerprint = _decode_scalar(file.attrs.get("model_fingerprint", ""))
  if fingerprint != contract.model_fingerprint:
    raise ValueError(f"{file.filename}: {task} model_fingerprint mismatch")
  if metadata.get("model_fingerprint", fingerprint) != fingerprint:
    raise ValueError(f"{file.filename}: inconsistent source model_fingerprint")
  mechanics = metadata.get("mechanics_version")
  if mechanics != contract.mechanics_version:
    raise ValueError(f"{file.filename}: {task} mechanics_version mismatch")
  # A successful native outcome is authoritative; shared task recorders do not
  # publish poker's task_completed key. Never manufacture that missing field.
  if task == "usb-unplug":
    required = ("fully_extracted", "placed_on_table", "released", "hand_retreated")
    if any(outcome.get(key) is not True for key in required):
      raise ValueError(f"{file.filename}: legacy USB return/release outcome mismatch")
    recorded = metadata.get("completion_protocol", outcome.get("completion_protocol"))
    if recorded not in (None, contract.completion_protocol):
      raise ValueError(f"{file.filename}: USB completion protocol mismatch")
  expected_physics_hz = 1000 if task.startswith("whiteboard-wipe") else 500
  expected_control_hz = 500 if task == "usb-insert" else 100
  if int(file.attrs.get("physics_hz", -1)) != expected_physics_hz:
    raise ValueError(f"{file.filename}: unexpected source physics_hz")
  if int(file.attrs.get("control_hz", -1)) != expected_control_hz:
    raise ValueError(f"{file.filename}: unexpected source control_hz")
  force_source = _decode_scalar(file.attrs.get("contact_force_source", ""))
  if force_source != "solver_contact_distributed_taxel_v1":
    raise ValueError(
      f"{file.filename}: requires solver-contact force taxels, not proxy depth"
    )
  if "tactile_contact_force" not in file:
    raise ValueError(f"{file.filename}: missing tactile_contact_force")
  if _decode_scalar(file["tactile_contact_force"].attrs.get("force_unit", "")) != "N":
    raise ValueError(f"{file.filename}: tactile force_unit must be N")
  tactile_group = file["tactile_contact_force"]
  if _decode_scalar(tactile_group.attrs.get("taxel_unit", "")) != "N_per_taxel":
    raise ValueError(f"{file.filename}: taxel_unit must be N_per_taxel")
  tactile_metadata = _json_attr(tactile_group, "metadata_json")
  if (
    tactile_metadata.get("source") != force_source
    or tactile_metadata.get("force_unit") != "N"
    or tactile_metadata.get("taxel_force_unit") != "N"
    or tactile_metadata.get("grid_shape") != [7, 5]
    or tactile_metadata.get("tangent_basis")
    != ["grid_col_positive", "grid_row_positive"]
    or tactile_metadata.get("is_spatial_estimate") is not True
    or not bool(tactile_group.attrs.get("is_spatial_estimate", False))
  ):
    raise ValueError(f"{file.filename}: tactile units/basis/source metadata mismatch")
  tactile_names = _decode_names(
    _require_dataset(file, "tactile_contact_force/link_names"), context="tactile links"
  )
  geometry = {"metadata": tactile_metadata, "link_names": tactile_names}
  for key, tail in (
    ("normal_axis_local", (3,)),
    ("tangent_basis_local", (2, 3)),
    ("taxel_positions_local_m", (35, 3)),
  ):
    values = np.asarray(
      _require_dataset(file, f"tactile_contact_force/{key}")[:], dtype=np.float64
    )
    if values.shape != (len(tactile_names), *tail) or not np.all(np.isfinite(values)):
      raise ValueError(f"{file.filename}: invalid tactile geometry {key}")
    geometry[key] = values.tolist()
  axes = np.concatenate(
    (
      np.asarray(geometry["normal_axis_local"])[:, None],
      np.asarray(geometry["tangent_basis_local"]),
    ),
    axis=1,
  )
  if not np.allclose(
    axes @ axes.transpose(0, 2, 1), np.eye(3)[None], atol=1e-6, rtol=0
  ):
    raise ValueError(
      f"{file.filename}: tactile normal/tangent axes must be orthonormal (either handedness)"
    )
  result = {
    "mechanics_version": mechanics,
    "model_fingerprint": fingerprint,
    "physics_hz": expected_physics_hz,
    "control_hz": expected_control_hz,
    "contact_force_source": force_source,
    "tactile_geometry_sha256": hashlib.sha256(
      json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    "tactile_clock_semantics": _decode_scalar(
      tactile_group.attrs.get("clock_semantics", "")
    ),
  }
  if contract.completion_protocol:
    result["completion_protocol"] = contract.completion_protocol
    result["completion_protocol_evidence"] = (
      "native return/release outcome and bound source scene"
    )
  return result


def _strict_30hz_prefix(
  timestamps: np.ndarray,
  physics_hz: int,
) -> tuple[int, float]:
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


def _rotations_to_quaternions_xyzw(matrices: np.ndarray) -> np.ndarray:
  """Convert a batch of proper rotation matrices to normalized xyzw quaternions."""
  rotations = np.asarray(matrices, dtype=np.float64)
  if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
    raise ValueError(
      f"rotation matrices must have shape (T,3,3), got {rotations.shape}"
    )
  quaternions = np.empty((rotations.shape[0], 4), dtype=np.float64)
  for index, matrix in enumerate(rotations):
    trace = float(np.trace(matrix))
    if trace > 0.0:
      scale = np.sqrt(trace + 1.0) * 2.0
      qw = 0.25 * scale
      qx = (matrix[2, 1] - matrix[1, 2]) / scale
      qy = (matrix[0, 2] - matrix[2, 0]) / scale
      qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
      diagonal = np.diag(matrix)
      axis = int(np.argmax(diagonal))
      if axis == 0:
        scale = (
          np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
        )
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
      elif axis == 1:
        scale = (
          np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
        )
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
      else:
        scale = (
          np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
        )
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
      raise ValueError(f"invalid rotation matrix at index {index}")
    quaternion /= norm
    if quaternion[3] < 0.0:
      quaternion = -quaternion
    quaternions[index] = quaternion
  return quaternions


def _matrices_to_pose6(matrices: np.ndarray, *, context: str) -> np.ndarray:
  transforms = np.asarray(matrices, dtype=np.float64)
  if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
    raise ValueError(f"{context}: transforms must have shape (T,4,4)")
  if not np.all(np.isfinite(transforms)):
    raise ValueError(f"{context}: transforms contain non-finite values")
  if not np.allclose(
    transforms[:, 3],
    np.asarray([0.0, 0.0, 0.0, 1.0]),
    atol=1.0e-7,
  ):
    raise ValueError(f"{context}: invalid homogeneous bottom row")
  rotations = transforms[:, :3, :3]
  identity = np.eye(3, dtype=np.float64)
  orthogonality = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
  if not np.allclose(orthogonality, identity[None], atol=1.0e-5):
    raise ValueError(f"{context}: rotation matrices are not orthogonal")
  if not np.allclose(np.linalg.det(rotations), 1.0, atol=1.0e-5):
    raise ValueError(f"{context}: rotation matrices are not proper")

  quaternion = _rotations_to_quaternions_xyzw(rotations)
  vector = quaternion[:, :3]
  scalar = np.clip(quaternion[:, 3], -1.0, 1.0)
  vector_norm = np.linalg.norm(vector, axis=1)
  angle = 2.0 * np.arctan2(vector_norm, scalar)
  scale = np.empty_like(angle)
  small = vector_norm < 1.0e-8
  scale[small] = 2.0
  scale[~small] = angle[~small] / vector_norm[~small]
  rotvec = vector * scale[:, None]
  return np.concatenate((transforms[:, :3, 3], rotvec), axis=1).astype(np.float32)


def _validate_rgb(dataset: h5py.Dataset, frames: int, *, context: str) -> None:
  if (
    dataset.dtype != np.uint8
    or dataset.ndim != 4
    or dataset.shape[0] < frames
    or dataset.shape[-1] != 3
  ):
    raise ValueError(
      f"{context}: expected uint8 (T,H,W,3) with at least {frames} frames, "
      f"got {dataset.shape} {dataset.dtype}"
    )


def convert_episode(
  source: SourceEpisode,
  *,
  instruction: str | None = None,
  sensor_name: str = TACTILE_SENSOR,
  task: str = TASK,
  metadata_only: bool = False,
) -> ConvertedEpisode | dict[str, Any]:
  instruction = _instruction(task, instruction)
  if not sensor_name.strip() or len(sensor_name) > 63:
    raise ValueError("sensor name must contain 1-63 characters")

  with h5py.File(source.hdf5_path, "r") as file:
    schema = _decode_scalar(file.attrs.get("schema_version", ""))
    if schema != RAW_SCHEMA:
      raise ValueError(f"{source.hdf5_path}: unsupported raw schema {schema!r}")
    if int(file.attrs.get("camera_hz", -1)) != FPS:
      raise ValueError(f"{source.hdf5_path}: camera_hz must be {FPS}")
    physics_hz = int(file.attrs.get("physics_hz", -1))
    source_environment = _validate_source_metadata(file, task=task)
    if (
      source.source_environment is not None
      and source_environment != source.source_environment
    ):
      raise ValueError(
        f"{source.hdf5_path}: source environment changed after discovery"
      )

    state_timestamps = np.asarray(
      _require_dataset(file, "state/timestamp")[:],
      dtype=np.float64,
    )
    if (
      state_timestamps.ndim != 1
      or not np.all(np.isfinite(state_timestamps))
      or not np.all(np.diff(state_timestamps) > 0)
    ):
      raise ValueError(f"{source.hdf5_path}: invalid state timestamps")

    camera_timestamps_all = np.asarray(
      _require_dataset(file, "cameras/head/timestamp")[:],
      dtype=np.float64,
    )
    frames, grid_error = _strict_30hz_prefix(camera_timestamps_all, physics_hz)
    camera_timestamps = camera_timestamps_all[:frames]
    head_state_index = _require_dataset(file, "cameras/head/state_index")
    if head_state_index.shape != camera_timestamps_all.shape:
      raise ValueError(f"{source.hdf5_path}: head state_index length mismatch")
    state_indices = np.asarray(head_state_index[:frames], dtype=np.int64)
    expected_indices = (
      np.searchsorted(
        state_timestamps,
        camera_timestamps,
        side="right",
      )
      - 1
    )
    if not np.array_equal(state_indices, expected_indices):
      raise ValueError(f"{source.hdf5_path}: head state_index is not latest-nonfuture")
    if np.any(state_indices < 0) or not np.all(np.diff(state_indices) > 0):
      raise ValueError(f"{source.hdf5_path}: invalid head state_index")

    wrist_timestamps_all = np.asarray(
      _require_dataset(file, "cameras/right_wrist/timestamp")[:],
      dtype=np.float64,
    )
    wrist_indices_all = np.asarray(
      _require_dataset(file, "cameras/right_wrist/state_index")[:],
      dtype=np.int64,
    )
    if wrist_timestamps_all.shape != camera_timestamps_all.shape:
      raise ValueError(f"{source.hdf5_path}: head/right-wrist camera lengths differ")
    if wrist_indices_all.shape != camera_timestamps_all.shape:
      raise ValueError(f"{source.hdf5_path}: right-wrist state_index length mismatch")
    if not np.array_equal(camera_timestamps_all, wrist_timestamps_all):
      raise ValueError(f"{source.hdf5_path}: head/right-wrist camera clocks differ")
    if not np.array_equal(head_state_index[:], wrist_indices_all):
      raise ValueError(f"{source.hdf5_path}: head/right-wrist state indices differ")

    head_rgb = _require_dataset(file, "cameras/head/rgb")
    wrist_rgb = _require_dataset(file, "cameras/right_wrist/rgb")
    _validate_rgb(head_rgb, frames, context="head RGB")
    _validate_rgb(wrist_rgb, frames, context="right-wrist RGB")
    if head_rgb.shape[0] != camera_timestamps_all.size:
      raise ValueError(f"{source.hdf5_path}: head RGB length differs from its clock")
    if wrist_rgb.shape[0] != camera_timestamps_all.size:
      raise ValueError(
        f"{source.hdf5_path}: right-wrist RGB length differs from its clock"
      )
    if head_rgb.shape[1:] != wrist_rgb.shape[1:]:
      raise ValueError(f"{source.hdf5_path}: camera RGB shapes differ")

    state_names = _decode_names(
      _require_dataset(file, "state/joint_names"),
      context="state/joint_names",
    )
    arm_columns = _columns(
      state_names,
      RIGHT_ARM_JOINT_NAMES,
      context="right arm joints",
    )
    hand_columns = _columns(
      state_names,
      RIGHT_HAND_JOINT_NAMES,
      context="right hand joints",
    )
    joint_position = _require_dataset(file, "state/robot_joint_position")
    if joint_position.shape != (state_timestamps.size, len(state_names)):
      raise ValueError(f"{source.hdf5_path}: joint state is off the state clock")
    right_arm = np.asarray(
      joint_position[state_indices][:, arm_columns],
      dtype=np.float32,
    )
    right_hand = np.asarray(
      joint_position[state_indices][:, hand_columns],
      dtype=np.float32,
    )
    if not np.all(np.isfinite(right_arm)) or not np.all(np.isfinite(right_hand)):
      raise ValueError(f"{source.hdf5_path}: joint states contain non-finite values")

    side_names = _decode_names(
      _require_dataset(file, "wrist_wrench/side_names"),
      context="wrist_wrench/side_names",
    )
    if "right" not in side_names:
      raise ValueError(f"{source.hdf5_path}: wrist pose has no right side")
    right_side = side_names.index("right")
    world_from_wrist = _require_dataset(file, "cameras/head/world_from_wrist")
    if world_from_wrist.shape != (camera_timestamps_all.size, len(side_names), 4, 4):
      raise ValueError(f"{source.hdf5_path}: world_from_wrist shape mismatch")
    right_wrist_pose = _matrices_to_pose6(
      world_from_wrist[:frames, right_side],
      context="right wrist pose",
    )
    world_from_camera = _require_dataset(file, "cameras/head/world_from_camera")
    if world_from_camera.shape != (camera_timestamps_all.size, 4, 4):
      raise ValueError(f"{source.hdf5_path}: world_from_camera shape mismatch")
    camera_ego_pose = _matrices_to_pose6(
      world_from_camera[:frames],
      context="head camera pose",
    )

    tactile_group = "tactile_contact_force"
    tactile_names = _decode_names(
      _require_dataset(file, f"{tactile_group}/link_names"),
      context=f"{tactile_group}/link_names",
    )
    tactile_columns = _columns(
      tactile_names,
      RIGHT_TACTILE_LINK_NAMES,
      context="right tactile links",
    )
    tactile_timestamps = np.asarray(
      _require_dataset(file, f"{tactile_group}/timestamp")[:],
      dtype=np.float64,
    )
    if tactile_timestamps.shape != state_timestamps.shape:
      raise ValueError(f"{source.hdf5_path}: tactile stream is off the state clock")
    if not np.all(np.isfinite(tactile_timestamps)):
      raise ValueError(f"{source.hdf5_path}: tactile timestamps are non-finite")
    if np.any(tactile_timestamps - state_timestamps > 1.0e-9):
      raise ValueError(
        f"{source.hdf5_path}: tactile samples occur after paired state rows"
      )
    maximum_tactile_age = float(np.max(state_timestamps - tactile_timestamps))
    if maximum_tactile_age > 1.01 / physics_hz:
      raise ValueError(
        f"{source.hdf5_path}: tactile/state row age {maximum_tactile_age:.6f}s "
        f"exceeds one physics step"
      )

    normal_dataset = _require_dataset(
      file,
      f"{tactile_group}/normal_taxel_force_n",
    )
    tangent_dataset = _require_dataset(
      file,
      f"{tactile_group}/tangent_taxel_force_n",
    )
    expected_normal_shape = (state_timestamps.size, len(tactile_names), 7, 5)
    expected_tangent_shape = (*expected_normal_shape, 2)
    if normal_dataset.shape != expected_normal_shape:
      raise ValueError(
        f"{source.hdf5_path}: normal taxels have shape {normal_dataset.shape}, "
        f"expected {expected_normal_shape}"
      )
    if tangent_dataset.shape != expected_tangent_shape:
      raise ValueError(
        f"{source.hdf5_path}: tangent taxels have shape {tangent_dataset.shape}, "
        f"expected {expected_tangent_shape}"
      )
    normal_force_dataset = _require_dataset(file, f"{tactile_group}/normal_force_n")
    if normal_force_dataset.shape != expected_normal_shape[:2]:
      raise ValueError(f"{source.hdf5_path}: normal_force_n shape mismatch")
    tangent_force_dataset = None
    if task != TASK:
      tangent_force_dataset = _require_dataset(file, f"{tactile_group}/tangent_force_n")
      if tangent_force_dataset.shape != (*expected_normal_shape[:2], 2):
        raise ValueError(f"{source.hdf5_path}: tangent_force_n shape mismatch")
    for dataset in (
      normal_dataset,
      tangent_dataset,
      normal_force_dataset,
      tangent_force_dataset,
    ):
      if dataset is not None and dataset.dtype.kind != "f":
        raise ValueError(
          f"{source.hdf5_path}: tactile force datasets must be floating-point"
        )
    if metadata_only:
      return {
        **_source_record(source),
        "task": task,
        "frames": frames,
        "omitted_terminal_camera_frames": int(camera_timestamps_all.size - frames),
        "maximum_camera_grid_error_seconds": grid_error,
        "source_environment": source_environment,
        "image_pixels_read": False,
        "full_taxel_values_read": False,
        "source_hash_recomputed": False,
      }
    normal = np.asarray(
      normal_dataset[state_indices][:, tactile_columns],
      dtype=np.float32,
    )
    tangent = np.asarray(
      tangent_dataset[state_indices][:, tactile_columns],
      dtype=np.float32,
    )
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(tangent)):
      raise ValueError(f"{source.hdf5_path}: tactile taxels contain non-finite values")
    if np.any(normal < -1.0e-7):
      raise ValueError(f"{source.hdf5_path}: normal tactile force is negative")
    normal = np.maximum(normal, 0.0)
    tactile = np.concatenate((normal[..., None], tangent), axis=-1)

    normal_force = np.asarray(
      _require_dataset(file, f"{tactile_group}/normal_force_n")[state_indices][
        :, tactile_columns
      ],
      dtype=np.float32,
    )
    if not np.allclose(normal.sum(axis=(2, 3)), normal_force, atol=2.0e-5, rtol=2.0e-5):
      raise ValueError(f"{source.hdf5_path}: normal taxels do not conserve force")
    if tangent_force_dataset is not None:
      tangent_force = np.asarray(
        tangent_force_dataset[state_indices][:, tactile_columns], dtype=np.float32
      )
      if not np.allclose(
        tangent.sum(axis=(2, 3)), tangent_force, atol=2.0e-5, rtol=2.0e-5
      ):
        raise ValueError(
          f"{source.hdf5_path}: signed tangent taxels do not conserve force"
        )

    arrays: dict[str, np.ndarray] = {
      "timestamps": camera_timestamps.astype(np.float64),
      "source_episode_index": np.full(frames, source.episode_index, dtype=np.int32),
      "source_camera_index": np.arange(frames, dtype=np.int32),
      "source_state_index": state_indices.astype(np.int64),
      "camera_ego_rgb": np.asarray(head_rgb[:frames], dtype=np.uint8),
      "right_wrist_camera_rgb": np.asarray(wrist_rgb[:frames], dtype=np.uint8),
      "camera_ego_pose": camera_ego_pose,
      "right_wrist_pose": right_wrist_pose,
      "right_arm_joints": right_arm,
      "right_hand_joints": right_hand,
      "right_hand_joints_idx": np.broadcast_to(
        RIGHT_HAND_FAAS_INDICES,
        (frames, RIGHT_HAND_FAAS_INDICES.size),
      ).copy(),
      "right_tactile_data_fingers": tactile.astype(np.float32, copy=False),
      "right_tactile_area_fingers": np.broadcast_to(
        np.arange(5, dtype=np.int32),
        (frames, 5),
      ).copy(),
      "right_tactile_sensor_fingers": np.full(
        frames,
        sensor_name,
        dtype="<U64",
      ),
      "right_tactile_type_fingers": np.full(frames, "matrix", dtype="<U8"),
      "sub_task_instruction": np.full(frames, instruction.strip(), dtype="<U256"),
    }

  validate_converted_arrays(arrays)
  return ConvertedEpisode(
    arrays=arrays,
    source=source,
    strict_camera_frames=frames,
    omitted_terminal_camera_frames=int(camera_timestamps_all.size - frames),
    maximum_camera_grid_error_seconds=grid_error,
  )


def preflight_episode(source: SourceEpisode, *, task: str = TASK) -> dict[str, Any]:
  """Read small state/clock metadata and dataset layouts, never RGB/taxel payloads."""
  result = convert_episode(source, task=task, metadata_only=True)
  assert isinstance(result, dict)
  return result


def validate_converted_arrays(arrays: dict[str, np.ndarray]) -> None:
  if tuple(arrays) != EXPECTED_DATA_KEYS:
    raise ValueError(f"converted keys differ from FTP-1 contract: {tuple(arrays)}")
  length = arrays["timestamps"].shape[0]
  if length < 2:
    raise ValueError("converted episode needs at least two frames")
  for key, value in arrays.items():
    if not isinstance(value, np.ndarray) or value.shape[0] != length:
      raise ValueError(f"{key}: first dimension must equal {length}")
  expected_shapes = {
    "camera_ego_rgb": (length, *arrays["camera_ego_rgb"].shape[1:]),
    "right_wrist_camera_rgb": (length, *arrays["camera_ego_rgb"].shape[1:]),
    "camera_ego_pose": (length, 6),
    "right_wrist_pose": (length, 6),
    "right_arm_joints": (length, 7),
    "right_hand_joints": (length, 20),
    "right_hand_joints_idx": (length, 20),
    "right_tactile_data_fingers": (length, 5, 7, 5, 3),
    "right_tactile_area_fingers": (length, 5),
  }
  for key, expected in expected_shapes.items():
    if arrays[key].shape != expected:
      raise ValueError(f"{key}: expected shape {expected}, got {arrays[key].shape}")
  if arrays["camera_ego_rgb"].dtype != np.uint8:
    raise ValueError("camera_ego_rgb must be uint8")
  if arrays["right_wrist_camera_rgb"].dtype != np.uint8:
    raise ValueError("right_wrist_camera_rgb must be uint8")
  if not np.array_equal(arrays["right_hand_joints_idx"][0], RIGHT_HAND_FAAS_INDICES):
    raise ValueError("right_hand_joints_idx does not match the KaiHand FAAS mapping")
  if not np.array_equal(arrays["right_tactile_area_fingers"][0], np.arange(5)):
    raise ValueError("right tactile areas must be [0,1,2,3,4]")
  for key, value in arrays.items():
    if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
      raise ValueError(f"{key}: contains non-finite values")


def _zarr_modules() -> tuple[Any, Any]:
  try:
    import numcodecs
    import zarr
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "This converter requires zarr and numcodecs. Run it with the FTP-1 "
      "repository's .venv/bin/python or install its locked dependencies."
    ) from exc
  return zarr, numcodecs


def _optimal_chunks(
  array: np.ndarray, target_bytes: int = 2_000_000
) -> tuple[int, ...]:
  if array.ndim < 1:
    raise ValueError("time-major arrays must have at least one dimension")
  if array.ndim == 4 and array.shape[-1] == 3 and array.dtype == np.uint8:
    return (1, *array.shape[1:])
  tail_items = max(1, int(np.prod(array.shape[1:], dtype=np.int64)))
  row_bytes = max(1, tail_items * array.dtype.itemsize)
  time_chunk = max(1, min(array.shape[0], target_bytes // row_bytes))
  return (int(time_chunk), *array.shape[1:])


class _ShardWriter:
  def __init__(self, path: Path, attributes: dict[str, Any]):
    zarr, numcodecs = _zarr_modules()
    self.path = path
    self.root = zarr.open_group(store=str(path), mode="w", zarr_format=2)
    self.data = self.root.create_group("data")
    self.meta = self.root.create_group("meta")
    self.episode_ends = self.meta.create_array(
      name="episode_ends",
      shape=(0,),
      dtype=np.int64,
      chunks=(64,),
      compressor=None,
    )
    self.compressor = numcodecs.Blosc(
      cname="zstd",
      clevel=5,
      shuffle=numcodecs.Blosc.BITSHUFFLE,
    )
    self.root.attrs.update(attributes)
    self.length = 0
    self.episode_count = 0

  def add_episode(self, episode: ConvertedEpisode) -> None:
    arrays = episode.arrays
    validate_converted_arrays(arrays)
    episode_length = arrays["timestamps"].shape[0]
    new_length = self.length + episode_length
    for key, value in arrays.items():
      if key not in self.data:
        array = self.data.create_array(
          name=key,
          shape=(0, *value.shape[1:]),
          dtype=value.dtype,
          chunks=_optimal_chunks(value),
          compressor=self.compressor,
        )
      else:
        array = self.data[key]
        if array.shape[1:] != value.shape[1:] or array.dtype != value.dtype:
          raise ValueError(
            f"{key}: shard schema changed from {array.shape[1:]} {array.dtype} "
            f"to {value.shape[1:]} {value.dtype}"
          )
      array.resize((new_length, *value.shape[1:]))
      array[self.length : new_length] = value
    self.episode_ends.resize(self.episode_count + 1)
    self.episode_ends[self.episode_count] = new_length
    self.length = new_length
    self.episode_count += 1


def validate_zarr_shard(
  path: Path,
  *,
  expected_episode_indices: Sequence[int] | None = None,
  expected_fingerprint: str | None = None,
  task: str = TASK,
  expected_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
  zarr, _ = _zarr_modules()
  root = zarr.open_group(store=str(path), mode="r")
  if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
    raise ValueError(f"{path}: missing data/meta/episode_ends")
  ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
  if ends.ndim != 1 or ends.size == 0 or np.any(np.diff(ends) <= 0):
    raise ValueError(f"{path}: invalid episode_ends")
  keys = tuple(root["data"].keys())
  if set(keys) != set(EXPECTED_DATA_KEYS):
    raise ValueError(f"{path}: data keys differ from contract: {keys}")
  total = int(ends[-1])
  for key in keys:
    if root[f"data/{key}"].shape[0] != total:
      raise ValueError(f"{path}: {key} length differs from episode_ends")
  indices = list(root.attrs.get("source_episode_indices", []))
  if expected_episode_indices is not None and indices != list(expected_episode_indices):
    raise ValueError(
      f"{path}: source episodes {indices} != {list(expected_episode_indices)}"
    )
  if len(indices) != ends.size:
    raise ValueError(f"{path}: source episode count differs from episode_ends")
  if root.attrs.get("schema") != MANIFEST_SCHEMA:
    raise ValueError(f"{path}: wrong or missing conversion schema")
  if root.attrs.get("task") != task or int(root.attrs.get("fps", -1)) != FPS:
    raise ValueError(f"{path}: wrong task or fps attributes")
  if int(root.attrs.get("ftp1_action_horizon", -1)) != FTP1_ACTION_HORIZON:
    raise ValueError(f"{path}: wrong FTP-1 action horizon attribute")
  if expected_fingerprint is not None:
    actual_fingerprint = root.attrs.get("conversion_fingerprint")
    if actual_fingerprint != expected_fingerprint:
      raise ValueError(
        f"{path}: conversion fingerprint {actual_fingerprint!r} does not match "
        f"{expected_fingerprint!r}"
      )
  if expected_attributes is not None:
    for key, expected in expected_attributes.items():
      if root.attrs.get(key) != expected:
        raise ValueError(f"{path}: changed conversion/source attribute {key}")

  data = root["data"]
  expected_dtypes = {
    "timestamps": np.dtype(np.float64),
    "source_episode_index": np.dtype(np.int32),
    "source_camera_index": np.dtype(np.int32),
    "source_state_index": np.dtype(np.int64),
    "camera_ego_rgb": np.dtype(np.uint8),
    "right_wrist_camera_rgb": np.dtype(np.uint8),
    "camera_ego_pose": np.dtype(np.float32),
    "right_wrist_pose": np.dtype(np.float32),
    "right_arm_joints": np.dtype(np.float32),
    "right_hand_joints": np.dtype(np.float32),
    "right_hand_joints_idx": np.dtype(np.int32),
    "right_tactile_data_fingers": np.dtype(np.float32),
    "right_tactile_area_fingers": np.dtype(np.int32),
  }
  for key, expected_dtype in expected_dtypes.items():
    if np.dtype(data[key].dtype) != expected_dtype:
      raise ValueError(f"{path}: {key} dtype {data[key].dtype} != {expected_dtype}")
  expected_tails = {
    "camera_ego_pose": (6,),
    "right_wrist_pose": (6,),
    "right_arm_joints": (7,),
    "right_hand_joints": (20,),
    "right_hand_joints_idx": (20,),
    "right_tactile_data_fingers": (5, 7, 5, 3),
    "right_tactile_area_fingers": (5,),
  }
  for key, tail in expected_tails.items():
    if data[key].shape[1:] != tail:
      raise ValueError(f"{path}: {key} trailing shape is {data[key].shape[1:]}")
  head_shape = data["camera_ego_rgb"].shape
  wrist_shape = data["right_wrist_camera_rgb"].shape
  if len(head_shape) != 4 or head_shape[-1] != 3 or wrist_shape != head_shape:
    raise ValueError(f"{path}: incompatible RGB shapes {head_shape} and {wrist_shape}")

  starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
  for position, episode_index in zip(starts, indices, strict=True):
    if int(data["source_episode_index"][int(position)]) != int(episode_index):
      raise ValueError(f"{path}: source episode index is wrong at {position}")
  hand_idx = np.asarray(root["data/right_hand_joints_idx"][0], dtype=np.int32)
  if not np.array_equal(hand_idx, RIGHT_HAND_FAAS_INDICES):
    raise ValueError(f"{path}: wrong KaiHand FAAS mapping")
  tactile_shape = root["data/right_tactile_data_fingers"].shape
  if tactile_shape[1:] != (5, 7, 5, 3):
    raise ValueError(f"{path}: wrong tactile shape {tactile_shape}")
  areas = np.asarray(data["right_tactile_area_fingers"][0], dtype=np.int32)
  if not np.array_equal(areas, np.arange(5, dtype=np.int32)):
    raise ValueError(f"{path}: wrong tactile function areas")
  return {
    "path": str(path),
    "episodes": int(ends.size),
    "frames": total,
    "episode_ends": ends.tolist(),
    "source_episode_indices": indices,
  }


def _atomic_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.",
    suffix=".tmp",
    dir=path.parent,
  )
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
      json.dump(value, file, indent=2, ensure_ascii=False)
      file.write("\n")
      file.flush()
      os.fsync(file.fileno())
    os.replace(temporary, path)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def _chunks(
  items: Sequence[SourceEpisode], size: int
) -> list[tuple[SourceEpisode, ...]]:
  if size <= 0:
    raise ValueError("episodes-per-shard must be positive")
  return [tuple(items[start : start + size]) for start in range(0, len(items), size)]


def _source_record(source: SourceEpisode) -> dict[str, Any]:
  result = {
    "episode_index": source.episode_index,
    "relative_hdf5_path": source.relative_hdf5_path,
    "relative_sidecar_path": source.relative_sidecar_path,
    "size_bytes": source.size_bytes,
    "mtime_ns": source.mtime_ns,
    "declared_sha256": source.declared_sha256,
  }
  if source.source_environment is not None:
    result["source_environment"] = source.source_environment
  return result


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as file:
    while block := file.read(8 * 1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def _verify_source(source: SourceEpisode) -> None:
  stat = source.hdf5_path.stat()
  if stat.st_size != source.size_bytes or stat.st_mtime_ns != source.mtime_ns:
    raise ValueError(f"{source.hdf5_path}: source changed after discovery")
  actual_sha256 = _sha256_file(source.hdf5_path)
  if actual_sha256 != source.declared_sha256:
    raise ValueError(
      f"{source.hdf5_path}: SHA-256 {actual_sha256} does not match capture "
      f"sidecar {source.declared_sha256}"
    )


def _conversion_contract(
  *, instruction: str, sensor_name: str, task: str = TASK
) -> dict[str, Any]:
  instruction = _instruction(task, instruction)
  result = {
    "schema": MANIFEST_SCHEMA,
    "task": task,
    "fps": FPS,
    "ftp1_action_horizon": FTP1_ACTION_HORIZON,
    "ftp1_action_down_sample_steps": 1,
    "action_storage": "none; derived by FTP-1 loader from trajectory states",
    "instruction": instruction.strip(),
    "right_arm_joint_names": list(RIGHT_ARM_JOINT_NAMES),
    "right_hand_joint_names": list(RIGHT_HAND_JOINT_NAMES),
    "right_hand_faas_indices": RIGHT_HAND_FAAS_INDICES.tolist(),
    "right_tactile_link_names": list(RIGHT_TACTILE_LINK_NAMES),
    "right_tactile_areas": list(range(5)),
    "tactile_sensor": sensor_name,
    "tactile_type": "matrix",
    "tactile_shape_per_area": [7, 5, 3],
    "tactile_channels": [
      "normal_n",
      "grid_col_tangent_n",
      "grid_row_tangent_n",
    ],
  }
  if task != TASK:
    contract = _task_contract(task)
    result.update(
      {
        "action_supervision": "ftp1-native-future-measured-state-trajectory-v1",
        "controller_target_actions_used": False,
        "pi05_controller_target_contract_compatible": False,
        "online_deployment_ready": False,
        "source_model_fingerprint": contract.model_fingerprint,
        "source_mechanics_version": contract.mechanics_version,
        "source_completion_protocol": contract.completion_protocol,
        "tactile_force_semantics": "solver-force Gaussian spatial estimate in N per taxel; not pressure or flexible-skin ground truth",
      }
    )
  return result


def _shard_fingerprint(
  sources: Sequence[SourceEpisode],
  *,
  instruction: str,
  sensor_name: str,
  task: str = TASK,
) -> str:
  payload = {
    "contract": _conversion_contract(
      instruction=instruction,
      sensor_name=sensor_name,
      task=task,
    ),
    "sources": [_source_record(source) for source in sources],
  }
  encoded = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _shard_attributes(
  sources: Sequence[SourceEpisode],
  *,
  instruction: str,
  sensor_name: str,
  task: str = TASK,
) -> dict[str, Any]:
  contract = _conversion_contract(
    instruction=instruction,
    sensor_name=sensor_name,
    task=task,
  )
  return {
    **contract,
    "conversion_fingerprint": _shard_fingerprint(
      sources,
      instruction=instruction,
      sensor_name=sensor_name,
      task=task,
    ),
    "source_episode_indices": [source.episode_index for source in sources],
    "source_records": [_source_record(source) for source in sources],
  }


def _write_shard(
  *,
  shard_index: int,
  shard_count: int,
  shard_sources: tuple[SourceEpisode, ...],
  final_path: Path,
  instruction: str,
  sensor_name: str,
  verify_source_hash: bool,
  task: str = TASK,
) -> ShardResult:
  episode_indices = [source.episode_index for source in shard_sources]
  fingerprint = _shard_fingerprint(
    shard_sources,
    instruction=instruction,
    sensor_name=sensor_name,
    task=task,
  )
  temporary_path = final_path.parent / (
    f".{final_path.name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
  )
  writer = _ShardWriter(
    temporary_path,
    _shard_attributes(
      shard_sources,
      instruction=instruction,
      sensor_name=sensor_name,
      task=task,
    ),
  )
  episode_records: list[dict[str, Any]] = []
  try:
    for inside_index, source in enumerate(shard_sources):
      if verify_source_hash:
        _verify_source(source)
      else:
        stat = source.hdf5_path.stat()
        if stat.st_size != source.size_bytes or stat.st_mtime_ns != source.mtime_ns:
          raise ValueError(f"{source.hdf5_path}: source changed after discovery")
      episode = convert_episode(
        source,
        instruction=instruction,
        sensor_name=sensor_name,
        task=task,
      )
      # Hashing and conversion are separate reads. Refuse publication if the
      # source changed while its arrays were being materialized.
      stat = source.hdf5_path.stat()
      if stat.st_size != source.size_bytes or stat.st_mtime_ns != source.mtime_ns:
        raise ValueError(f"{source.hdf5_path}: source changed during conversion")
      writer.add_episode(episode)
      episode_records.append(
        {
          **_source_record(source),
          "source_hash_verified": verify_source_hash,
          "frames": episode.strict_camera_frames,
          "omitted_terminal_camera_frames": episode.omitted_terminal_camera_frames,
          "maximum_camera_grid_error_seconds": (
            episode.maximum_camera_grid_error_seconds
          ),
          "shard": final_path.name,
        }
      )
      print(
        f"write shard={shard_index + 1}/{shard_count} "
        f"episode={inside_index + 1}/{len(shard_sources)} "
        f"source_index={source.episode_index} frames={episode.strict_camera_frames}",
        flush=True,
      )
    writer.root.attrs["frames"] = writer.length
    writer.root.attrs["episodes"] = writer.episode_count
    del writer
    record = validate_zarr_shard(
      temporary_path,
      expected_episode_indices=episode_indices,
      expected_fingerprint=fingerprint,
      task=task,
      expected_attributes=_shard_attributes(
        shard_sources, instruction=instruction, sensor_name=sensor_name, task=task
      ),
    )
    os.replace(temporary_path, final_path)
    record["path"] = str(final_path)
    print(
      f"publish shard={shard_index + 1}/{shard_count} "
      f"episodes={record['episodes']} frames={record['frames']} path={final_path}",
      flush=True,
    )
    return ShardResult(
      shard_index=shard_index,
      shard_record=record,
      episode_records=tuple(episode_records),
    )
  except BaseException:
    if temporary_path.exists():
      shutil.rmtree(temporary_path)
    raise


def _dataset_configuration(output_dir: Path, task: str) -> dict[str, Any]:
  return {
    "datasets": [
      {
        "name": "KaiHandSim" + "".join(part.title() for part in task.split("-")),
        "path": str(output_dir),
        "use_trajectory_ratio": 1.0,
        "enabled": True,
      }
    ],
    "default_use_trajectory_ratio": 1.0,
    "description": f"KaiHand {task} Raw converted to FTP-1 Zarr with 5-finger matrix tactile.",
    "recommended_training": {
      "action_horizon": FTP1_ACTION_HORIZON,
      "action_down_sample_steps": 1,
      "trajectory_fps": FPS,
    },
  }


def run(args: argparse.Namespace) -> Path | None:
  collection_root = args.input_dir.resolve()
  unresolved_output = Path(os.path.abspath(args.output_dir.expanduser()))
  if any(path.is_symlink() for path in (unresolved_output, *unresolved_output.parents)):
    raise ValueError("output directory and its ancestors must not be symbolic links")
  output_dir = args.output_dir.resolve()
  if (
    output_dir == collection_root
    or collection_root in output_dir.parents
    or output_dir in collection_root.parents
  ):
    raise ValueError("output directory must not overlap the Raw collection")
  args.instruction = _instruction(args.task, args.instruction)
  if getattr(args, "plan_json", False) and not args.dry_run:
    raise ValueError("--plan-json requires --dry-run")
  sources = discover_sources(collection_root, args.task)
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError("limit must be positive")
    sources = sources[: args.limit]
  expected_episodes = getattr(args, "expected_episodes", None)
  if expected_episodes is not None and (
    expected_episodes <= 0 or len(sources) != expected_episodes
  ):
    raise ValueError(
      f"expected {expected_episodes} published episodes, found {len(sources)}"
    )
  if args.workers <= 0:
    raise ValueError("workers must be positive")
  if not args.instruction.strip():
    raise ValueError("instruction must not be empty")
  if len(args.instruction) > 255:
    raise ValueError("instruction must contain at most 255 characters")
  if not args.sensor_name.strip() or len(args.sensor_name) > 63:
    raise ValueError("sensor name must contain 1-63 characters")
  shards = _chunks(sources, args.episodes_per_shard)
  preflight_records = [preflight_episode(source, task=args.task) for source in sources]
  prefix = args.task.replace("-", "_")
  expected_names = {
    f"{prefix}_{shard_index:05d}.zarr" for shard_index in range(len(shards))
  }
  extras = {
    path.name for path in output_dir.glob("*.zarr") if path.name not in expected_names
  }
  if any(path.is_symlink() for path in output_dir.glob("*.zarr")):
    raise ValueError("published Zarr stores must not be symbolic links")
  if extras:
    raise FileExistsError(
      f"{output_dir}: unexpected Zarr stores would be loaded as this domain: {sorted(extras)}"
    )
  manifest_path = output_dir / "conversion_manifest.json"
  config_path = output_dir / "dataset_config.json"
  if manifest_path.is_symlink() or config_path.is_symlink():
    raise ValueError("published dataset metadata must not be symbolic links")
  if config_path.exists() and not args.resume:
    raise FileExistsError(
      f"{config_path} exists; pass --resume to validate existing output"
    )
  if config_path.exists() and _load_json(config_path) != _dataset_configuration(
    output_dir, args.task
  ):
    raise ValueError(f"{config_path}: existing dataset configuration identity mismatch")
  if manifest_path.exists():
    if not args.resume:
      raise FileExistsError(
        f"{manifest_path} exists; pass --resume to validate existing output"
      )
    previous = _load_json(manifest_path)
    for key, expected in (
      ("schema", MANIFEST_SCHEMA),
      ("task", args.task),
      ("instruction", args.instruction),
      ("source_root", str(collection_root)),
      ("output_root", str(output_dir)),
      ("episodes", len(sources)),
      ("frames", sum(record["frames"] for record in preflight_records)),
      ("fps", FPS),
      ("ftp1_action_horizon", FTP1_ACTION_HORIZON),
      ("ftp1_action_down_sample_steps", 1),
    ):
      if previous.get(key) != expected:
        raise ValueError(f"{manifest_path}: resume identity differs for {key}")
    previous_sources = previous.get("source_episodes", [])
    if len(previous_sources) != len(sources):
      raise ValueError(f"{manifest_path}: resume source episode count mismatch")
    for source, old in zip(sources, previous_sources, strict=True):
      if any(old.get(key) != value for key, value in _source_record(source).items()):
        raise ValueError(f"{manifest_path}: resume source identity mismatch")

  shard_results: list[ShardResult] = []
  episode_records: list[dict[str, Any]] = []
  pending: list[dict[str, Any]] = []
  for shard_index, shard_sources in enumerate(shards):
    final_path = output_dir / f"{prefix}_{shard_index:05d}.zarr"
    episode_indices = [source.episode_index for source in shard_sources]
    fingerprint = _shard_fingerprint(
      shard_sources,
      instruction=args.instruction,
      sensor_name=args.sensor_name,
      task=args.task,
    )
    if final_path.exists():
      if not args.resume:
        raise FileExistsError(
          f"{final_path} already exists; pass --resume to validate and reuse it"
        )
      record = validate_zarr_shard(
        final_path,
        expected_episode_indices=episode_indices,
        expected_fingerprint=fingerprint,
        task=args.task,
        expected_attributes=_shard_attributes(
          shard_sources,
          instruction=args.instruction,
          sensor_name=args.sensor_name,
          task=args.task,
        ),
      )
      if args.verify_source_hash and not args.dry_run:
        for source in shard_sources:
          _verify_source(source)
      shard_results.append(
        ShardResult(
          shard_index=shard_index,
          shard_record=record,
          episode_records=tuple(),
        )
      )
      continue
    pending.append(
      {
        "shard_index": shard_index,
        "shard_count": len(shards),
        "shard_sources": shard_sources,
        "final_path": final_path,
        "instruction": args.instruction,
        "sensor_name": args.sensor_name,
        "verify_source_hash": args.verify_source_hash,
        "task": args.task,
      }
    )

  published_complete = manifest_path.is_file() and config_path.is_file() and not pending
  if published_complete:
    actual_shards = [
      result.shard_record
      for result in sorted(shard_results, key=lambda result: result.shard_index)
    ]
    if previous.get("shards") != actual_shards:
      raise ValueError(
        f"{manifest_path}: published shard records differ from verified stores"
      )

  plan = {
    "task": args.task,
    "source_root": str(collection_root),
    "output_root": str(output_dir),
    "episodes": len(sources),
    "shards": len(shards),
    "fps": FPS,
    "ftp1_action_horizon": FTP1_ACTION_HORIZON,
    "metadata_preflight_valid": True,
    "image_pixels_read": False,
    "full_taxel_values_read": False,
    "source_hash_recomputed": False,
    "outputs_written": False,
    "published_complete": published_complete,
    "source_episodes": preflight_records,
  }
  if args.dry_run:
    if getattr(args, "plan_json", False):
      print(json.dumps(plan, sort_keys=True))
    else:
      print(
        f"plan task={args.task} episodes={len(sources)} shards={len(shards)} fps={FPS} metadata_preflight=valid (no RGB reads, no source hash, no output writes)"
      )
      print(f"source={collection_root}\noutput={output_dir}")
    return None

  print(
    f"plan task={args.task} episodes={len(sources)} shards={len(shards)} workers={min(args.workers, len(shards))} fps={FPS} ftp1_horizon={FTP1_ACTION_HORIZON}"
  )
  print(f"source={collection_root}\noutput={output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  if not pending and manifest_path.exists():
    if not config_path.is_file():
      raise ValueError(
        f"{config_path}: complete published dataset is missing its configuration"
      )
    print(
      "complete published dataset validated; preserving manifest/configuration bytes",
      flush=True,
    )
    return output_dir

  if len(pending) == 1 or args.workers == 1:
    for job in pending:
      shard_results.append(_write_shard(**job))
  elif pending:
    worker_count = min(args.workers, len(pending))
    print(f"parallel shards={len(pending)} workers={worker_count}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(
      max_workers=worker_count,
    ) as executor:
      futures = [executor.submit(_write_shard, **job) for job in pending]
      for future in concurrent.futures.as_completed(futures):
        shard_results.append(future.result())

  shard_results.sort(key=lambda result: result.shard_index)
  shard_records = [result.shard_record for result in shard_results]
  for result in shard_results:
    episode_records.extend(result.episode_records)

  # Reconstruct episode records when a complete shard was reused.
  recorded_indices = {record["episode_index"] for record in episode_records}
  for source in sources:
    if source.episode_index not in recorded_indices:
      episode_records.append(
        {
          **_source_record(source),
          "source_hash_verified": args.verify_source_hash,
          "reused": True,
        }
      )
  episode_records.sort(key=lambda record: int(record["episode_index"]))

  total_frames = sum(int(record["frames"]) for record in shard_records)
  manifest = {
    "schema": MANIFEST_SCHEMA,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "task": args.task,
    "source_root": str(collection_root),
    "output_root": str(output_dir),
    "episodes": len(sources),
    "frames": total_frames,
    "fps": FPS,
    "ftp1_action_horizon": FTP1_ACTION_HORIZON,
    "ftp1_action_down_sample_steps": 1,
    "horizon_semantics": (
      "training-time loader uses the current frame plus 31 later 30 Hz frames; "
      "no precomputed action chunks are stored"
    ),
    "action_window_seconds": (FTP1_ACTION_HORIZON - 1) / FPS,
    "clock_alignment": (
      "30 Hz head camera; camera state_index is latest-nonfuture; tactile uses "
      "the same source row and may precede post-integration state time by one physics step"
    ),
    "images": {
      "camera_ego_rgb": "cameras/head/rgb",
      "right_wrist_camera_rgb": "cameras/right_wrist/rgb",
    },
    "right_hand_joint_names": list(RIGHT_HAND_JOINT_NAMES),
    "right_hand_faas_indices": RIGHT_HAND_FAAS_INDICES.tolist(),
    "tactile": {
      "key": "right_tactile_data_fingers",
      "source_links": list(RIGHT_TACTILE_LINK_NAMES),
      "areas": list(range(5)),
      "sensor": args.sensor_name,
      "type": "matrix",
      "shape_per_area": [7, 5, 3],
      "channels": ["normal_n", "grid_col_tangent_n", "grid_row_tangent_n"],
      "unit": "N per taxel",
    },
    "instruction": args.instruction.strip(),
    "action_supervision": {
      "contract": "ftp1-native-future-measured-state-trajectory-v1",
      "source": "measured wrist poses and hand joint positions; actions are built by the native FTP-1 loader",
      "controller_targets_used": False,
      "raw_controller_targets": "may be present in Raw; not used as this export's action supervision",
      "pi05_recorded_controller_target_equivalent": False,
      "online_deployment_ready": False,
    },
    "source_hash_policy": (
      "recomputed-and-matched-capture-sidecar"
      if args.verify_source_hash
      else "capture-sidecar-sha256-plus-size-and-mtime"
    ),
    "shards": shard_records,
    "source_episodes": episode_records,
  }
  dataset_config = _dataset_configuration(output_dir, args.task)
  _atomic_json(output_dir / "conversion_manifest.json", manifest)
  _atomic_json(output_dir / "dataset_config.json", dataset_config)
  print(
    f"complete episodes={len(sources)} frames={total_frames} "
    f"dataset_config={output_dir / 'dataset_config.json'}",
    flush=True,
  )
  return output_dir


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  run(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
