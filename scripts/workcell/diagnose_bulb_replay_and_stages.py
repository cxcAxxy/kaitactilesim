#!/usr/bin/env python3
"""Replay bulb demonstrations and start pi0.5 from reference task stages.

The replay path applies the published 30 Hz absolute actions through the same
joint-limit and actuator-goal functions used by policy evaluation.  The staged
path restores a reference physics state only once, before policy inference, so
pickup/transport/insertion can be diagnosed independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from kaihand_tactile_env.shared.config import model_fingerprint  # noqa: E402
from kaihand_tactile_env.shared.policy_tasks import (  # noqa: E402
  TaskPolicyAdapter,
  create_task_policy_adapter,
)
from kaihand_tactile_env.tasks.bulb_screw import config as bulb_config  # noqa: E402
from replay_episode import _load_replay_states  # noqa: E402
from run_usb_pi05_policy import (  # noqa: E402
  apply_action,
  joint_limits,
  right_joint_state,
)

CONTROL_HZ = 30
STAGES = (
  "stable_grasp",
  "socket_hover",
  "threaded",
  "late_threaded_stable",
  "seated_loaded",
)
ACTION_SOURCES = ("published", "recorded-controller-target")
CONTROL_INTERPOLATIONS = ("zoh", "linear")
CONTROL_RATES_HZ = (30, 100)
INITIAL_OBJECT_POSITION_TOLERANCE_M = 1.0e-9
INITIAL_OBJECT_ORIENTATION_TOLERANCE_RAD = 1.0e-8
INITIAL_ROBOT_QPOS_TOLERANCE_RAD = 1.0e-9
INITIAL_ROBOT_QVEL_TOLERANCE_RAD_S = 1.0e-9


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  return value


def _write_new_json(path: Path, value: Any) -> None:
  resolved = path.expanduser().resolve()
  resolved.parent.mkdir(parents=True, exist_ok=True)
  with resolved.open("x", encoding="utf-8") as stream:
    json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2)
    stream.write("\n")


def _decode_names(dataset: h5py.Dataset) -> tuple[str, ...]:
  names = tuple(
    value.decode("utf-8") if isinstance(value, bytes) else str(value)
    for value in dataset
  )
  if not names or not all(names) or len(names) != len(set(names)):
    raise ValueError(f"{dataset.name} must contain unique, nonempty names")
  return names


def _metadata(file: h5py.File) -> dict[str, Any]:
  raw = file.attrs.get("metadata_json", "{}")
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8")
  value = json.loads(str(raw))
  if not isinstance(value, dict):
    raise ValueError("metadata_json must contain a JSON object")
  return value


def _source_position_seed(source_hdf5: Path) -> int:
  with h5py.File(source_hdf5, "r") as file:
    layout = _metadata(file).get("initial_position_randomization")
  if not isinstance(layout, dict) or not isinstance(layout.get("seed"), int):
    raise ValueError(f"source episode has no integer pickup seed: {source_hdf5}")
  return int(layout["seed"])


def _feature_names(info: dict[str, Any], key: str) -> list[str]:
  try:
    names = info["features"][key]["names"]
  except (KeyError, TypeError) as error:
    raise ValueError(f"dataset info has no {key} feature names") from error
  if len(names) == 1 and isinstance(names[0], list):
    names = names[0]
  if len(names) != 27 or len(set(names)) != 27 or not all(map(str, names)):
    raise ValueError(f"dataset {key} must declare 27 unique joint names")
  return [str(name) for name in names]


def _dataset_episode_paths(
  dataset_root: Path, episode_index: int
) -> tuple[Path, Path, dict[str, Any]]:
  root = dataset_root.expanduser().resolve(strict=True)
  info_path = root / "meta/info.json"
  info = json.loads(info_path.read_text(encoding="utf-8"))
  chunk_size = int(info.get("chunks_size", 1000))
  if episode_index < 0 or chunk_size <= 0:
    raise ValueError("episode index must be nonnegative and chunk size positive")
  relative = str(info["data_path"]).format(
    episode_chunk=episode_index // chunk_size,
    episode_index=episode_index,
  )
  parquet = (root / relative).resolve(strict=True)
  source_manifest = root / "meta/kaihand_source_episodes.jsonl"
  source_hdf5 = None
  source_row = None
  with source_manifest.open(encoding="utf-8") as stream:
    for line in stream:
      row = json.loads(line)
      if int(row.get("output_episode_index", -1)) == episode_index:
        source_hdf5 = Path(row["source_hdf5"]).expanduser().resolve(strict=True)
        source_row = row
        break
  if source_hdf5 is None or source_row is None:
    raise ValueError(f"source manifest has no output episode {episode_index}")
  return parquet, source_hdf5, source_row


def _file_identity(path: Path) -> dict[str, Any]:
  """Hash one immutable diagnostic input and reject concurrent replacement."""
  before = path.stat()
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(16 * 1024 * 1024):
      digest.update(block)
  after = path.stat()
  before_identity = (
    before.st_dev,
    before.st_ino,
    before.st_size,
    before.st_mtime_ns,
  )
  after_identity = (
    after.st_dev,
    after.st_ino,
    after.st_size,
    after.st_mtime_ns,
  )
  if before_identity != after_identity:
    raise RuntimeError(f"diagnostic input changed while hashing: {path}")
  return {
    "path": str(path),
    "size": after.st_size,
    "device": after.st_dev,
    "inode": after.st_ino,
    "mtime_ns": after.st_mtime_ns,
    "sha256": digest.hexdigest(),
  }


def _assert_file_identity_unchanged(identity: dict[str, Any]) -> None:
  path = Path(identity["path"])
  current = _file_identity(path)
  observed = (
    current["device"],
    current["inode"],
    current["size"],
    current["mtime_ns"],
    current["sha256"],
  )
  expected = (
    identity["device"],
    identity["inode"],
    identity["size"],
    identity["mtime_ns"],
    identity["sha256"],
  )
  if observed != expected:
    raise RuntimeError(f"diagnostic input changed during replay: {path}")


def _parquet_vector(column: Any, count: int, width: int, name: str) -> np.ndarray:
  values = np.asarray([column[index].as_py() for index in range(count)], dtype=float)
  if values.shape != (count, width) or not np.isfinite(values).all():
    raise ValueError(f"{name} must have finite shape {(count, width)}")
  return values


def _parquet_scalar_indices(column: Any, count: int, name: str) -> np.ndarray:
  values = []
  for index in range(count):
    value = column[index].as_py()
    if isinstance(value, list):
      if len(value) != 1:
        raise ValueError(f"{name} row {index} must contain one index")
      value = value[0]
    values.append(value)
  result = np.asarray(values, dtype=np.int64)
  if result.shape != (count,) or np.any(result < 0):
    raise ValueError(f"{name} must contain nonnegative scalar indices")
  return result


def _validated_published_parquet(
  parquet_path: Path,
  reconstructed_actions: np.ndarray,
  reconstructed_observed: np.ndarray,
  reconstructed_action_source_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
  """Load the real training rows and prove the HDF5 reconstruction matches."""
  import pyarrow.parquet as parquet

  required = (
    "action",
    "observation.state",
    "provenance.action_source_state_index",
  )
  table = parquet.read_table(parquet_path, columns=list(required))
  count = len(reconstructed_actions)
  if table.num_rows != count:
    raise ValueError(
      f"parquet row count {table.num_rows} differs from reconstructed count {count}"
    )
  actions = _parquet_vector(table.column("action"), count, 27, "parquet action")
  observed = _parquet_vector(
    table.column("observation.state"), count, 27, "parquet observation.state"
  )
  action_source_indices = _parquet_scalar_indices(
    table.column("provenance.action_source_state_index"),
    count,
    "parquet action source state index",
  )
  actions32 = actions.astype(np.float32)
  observed32 = observed.astype(np.float32)
  reconstructed_actions32 = np.asarray(reconstructed_actions, dtype=np.float32)
  reconstructed_observed32 = np.asarray(reconstructed_observed, dtype=np.float32)
  if not np.array_equal(actions32, reconstructed_actions32):
    difference = float(np.max(np.abs(actions32 - reconstructed_actions32)))
    raise ValueError(
      "published parquet actions differ from the HDF5 converter reconstruction "
      f"after float32 conversion (maximum absolute difference {difference})"
    )
  if not np.array_equal(observed32, reconstructed_observed32):
    difference = float(np.max(np.abs(observed32 - reconstructed_observed32)))
    raise ValueError(
      "published parquet observations differ from the HDF5 converter "
      f"reconstruction after float32 conversion (maximum absolute difference {difference})"
    )
  if not np.array_equal(action_source_indices, reconstructed_action_source_indices):
    raise ValueError(
      "published parquet action provenance differs from HDF5 camera state indices"
    )
  return (
    actions32,
    observed32,
    action_source_indices,
    {
      "status": "float32_exact",
      "rows": count,
      "action_float32_exact": True,
      "observation_state_float32_exact": True,
      "action_source_state_index_exact": True,
    },
  )


def _dataset_episode_length(dataset_root: Path, episode_index: int) -> int:
  manifest = dataset_root.expanduser().resolve() / "meta/episodes.jsonl"
  with manifest.open(encoding="utf-8") as stream:
    for line in stream:
      row = json.loads(line)
      if int(row.get("episode_index", -1)) == episode_index:
        length = int(row["length"])
        if length <= 0:
          raise ValueError("dataset episode length must be positive")
        return length
  raise ValueError(f"episode manifest has no episode {episode_index}")


def _interpolate_positions(
  source_times: np.ndarray,
  source_positions: np.ndarray,
  target_times: np.ndarray,
) -> np.ndarray:
  tolerance = 1.0e-12
  right = np.searchsorted(source_times, target_times, side="left")
  right = np.clip(right, 0, len(source_times) - 1)
  exact = np.isclose(source_times[right], target_times, rtol=0.0, atol=tolerance)
  left = np.maximum(right - 1, 0)
  left[exact] = right[exact]
  denominator = source_times[right] - source_times[left]
  alpha = np.zeros_like(target_times, dtype=float)
  bracketed = denominator > 0.0
  alpha[bracketed] = (
    target_times[bracketed] - source_times[left[bracketed]]
  ) / denominator[bracketed]
  return source_positions[left] + alpha[:, None] * (
    source_positions[right] - source_positions[left]
  )


def _reconstruct_published_episode(
  source_hdf5: Path,
  joint_names: list[str],
  total_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Rebuild the converter's next-camera-frame hybrid action exactly."""

  with h5py.File(source_hdf5, "r") as file:
    state_times = np.asarray(file["state/timestamp"], dtype=float)
    state_names = _decode_names(file["state/joint_names"])
    state_positions = np.asarray(file["state/robot_joint_position"], dtype=float)
    camera_times = np.asarray(file["cameras/head/timestamp"], dtype=float)
    camera_indices = np.asarray(file["cameras/head/state_index"], dtype=np.int64)
    if len(camera_times) < total_actions + 1:
      raise ValueError("source camera stream is shorter than the published episode")
    selected_camera_times = camera_times[: total_actions + 1]
    state_columns = []
    for name in joint_names:
      try:
        state_columns.append(state_names.index(name))
      except ValueError as error:
        raise ValueError(f"source state has no dataset joint {name!r}") from error
    aligned = _interpolate_positions(
      state_times,
      state_positions[:, state_columns],
      selected_camera_times,
    )
    observed = aligned[:-1].copy()
    actions = aligned[1:].copy()
    hand_names = _decode_names(file["commands/hand_joint_names"])
    hand_targets = np.asarray(file["commands/hand_joint_target"], dtype=float)
    hand_columns = []
    for name in joint_names[7:]:
      try:
        hand_columns.append(hand_names.index(name))
      except ValueError as error:
        raise ValueError(f"source commands have no dataset joint {name!r}") from error
    action_source_indices = camera_indices[1 : total_actions + 1]
    actions[:, 7:] = hand_targets[action_source_indices][:, hand_columns]
  if (
    actions.shape != (total_actions, 27)
    or observed.shape != (total_actions, 27)
    or action_source_indices.shape != (total_actions,)
    or not np.isfinite(actions).all()
    or not np.isfinite(observed).all()
  ):
    raise ValueError("reconstructed published episode is malformed")
  return actions, observed, action_source_indices


def _recorded_controller_target_actions(
  source_hdf5: Path,
  joint_names: list[str],
  action_source_indices: np.ndarray,
) -> np.ndarray:
  """Load recorded right-arm goals and hand targets at action provenance rows."""

  if len(joint_names) != 27 or len(set(joint_names)) != 27:
    raise ValueError("controller-target replay requires 27 unique dataset joints")
  expected_arm_names = tuple(f"right_arm_joint{index}" for index in range(1, 8))
  if tuple(joint_names[:7]) != expected_arm_names:
    raise ValueError(
      "dataset action arm order must be right_arm_joint1 through right_arm_joint7"
    )
  indices = np.asarray(action_source_indices)
  if (
    indices.ndim != 1
    or not np.issubdtype(indices.dtype, np.integer)
    or np.any(indices < 0)
  ):
    raise ValueError("action provenance indices must be nonnegative integers")

  with h5py.File(source_hdf5, "r") as file:
    controller_targets, _ = _controller_target_stream(file, joint_names)
    maximum_index = int(indices.max(initial=0))
    if maximum_index >= len(controller_targets):
      raise ValueError("action provenance index exceeds recorded controller streams")
    actions = controller_targets[indices]
  if actions.shape != (len(indices), 27) or not np.isfinite(actions).all():
    raise ValueError("recorded controller-target actions must be finite with width 27")
  return actions


def _controller_target_stream(
  file: h5py.File, joint_names: list[str]
) -> tuple[np.ndarray, dict[str, Any]]:
  """Return the generic recorded command stream in policy joint order."""

  expected_arm_names = tuple(f"right_arm_joint{index}" for index in range(1, 8))
  if tuple(joint_names[:7]) != expected_arm_names:
    raise ValueError(
      "dataset action arm order must be right_arm_joint1 through right_arm_joint7"
    )
  arm_names = _decode_names(file["commands/arm_joint_names"])
  hand_names = _decode_names(file["commands/hand_joint_names"])
  try:
    arm_columns = [arm_names.index(name) for name in joint_names[:7]]
    hand_columns = [hand_names.index(name) for name in joint_names[7:]]
  except ValueError as error:
    raise ValueError(
      f"source commands have no policy joint {error.args[0]!r}"
    ) from error
  arm_targets = np.asarray(file["commands/arm_joint_target"], dtype=float)
  hand_targets = np.asarray(file["commands/hand_joint_target"], dtype=float)
  if arm_targets.ndim != 2 or arm_targets.shape[1] != len(arm_names):
    raise ValueError("arm_joint_target width must match arm_joint_names")
  if hand_targets.ndim != 2 or hand_targets.shape[1] != len(hand_names):
    raise ValueError("hand_joint_target width must match hand_joint_names")
  if len(arm_targets) != len(hand_targets):
    raise ValueError("arm and hand controller-target streams must have equal length")
  targets = np.concatenate(
    (arm_targets[:, arm_columns], hand_targets[:, hand_columns]), axis=1
  )
  if targets.shape != (len(arm_targets), 27) or not np.isfinite(targets).all():
    raise ValueError("recorded controller-target stream must be finite with width 27")

  task_local_exact = None
  task_local_field = "bulb_screw/right_arm_joint_goal"
  if task_local_field in file:
    task_local = np.asarray(file[task_local_field], dtype=float)
    if task_local.shape != (len(targets), 7) or not np.isfinite(task_local).all():
      raise ValueError("right_arm_joint_goal must be finite with shape (N, 7)")
    task_local_exact = bool(np.array_equal(task_local, targets[:, :7]))
    if not task_local_exact:
      raise ValueError(
        "generic right-arm command targets differ from bulb task-local arm goals"
      )
  return targets, {
    "arm_field": "commands/arm_joint_target",
    "hand_field": "commands/hand_joint_target",
    "joint_order": list(joint_names),
    "task_local_arm_field": task_local_field if task_local_exact is not None else None,
    "task_local_arm_field_float64_exact": task_local_exact,
  }


def _source_rate_controller_target_episode(
  source_hdf5: Path, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
  """Build the recorded 100 Hz controller-target transition stream."""

  with h5py.File(source_hdf5, "r") as file:
    if int(file.attrs.get("control_hz", -1)) != 100:
      raise ValueError("source-rate replay requires a 100 Hz recorded control stream")
    state_times = np.asarray(file["state/timestamp"], dtype=float)
    state_names = _decode_names(file["state/joint_names"])
    state_positions = np.asarray(file["state/robot_joint_position"], dtype=float)
    try:
      state_columns = [state_names.index(name) for name in joint_names]
    except ValueError as error:
      raise ValueError(f"source state has no policy joint {error.args[0]!r}") from error
    targets, contract = _controller_target_stream(file, joint_names)
  if (
    state_times.ndim != 1
    or len(state_times) < 2
    or state_positions.shape != (len(state_times), len(state_names))
    or len(targets) != len(state_times)
    or not np.isfinite(state_times).all()
    or not np.all(np.diff(state_times) > 0)
  ):
    raise ValueError("source-rate state/controller streams are malformed")
  observed = state_positions[:-1, state_columns]
  actions = targets[1:]
  source_indices = np.arange(1, len(state_times), dtype=np.int64)
  deadlines = state_times[1:] - state_times[0]
  return actions, observed, source_indices, targets[0], deadlines, contract


def _initial_30hz_target(
  source_hdf5: Path,
  joint_names: list[str],
  observed: np.ndarray,
  action_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Return the controller target at the observation preceding action zero."""

  with h5py.File(source_hdf5, "r") as file:
    camera_indices = np.asarray(file["cameras/head/state_index"], dtype=np.int64)
    targets, contract = _controller_target_stream(file, joint_names)
  if camera_indices.ndim != 1 or not len(camera_indices):
    raise ValueError("source head camera has no state indices")
  index = int(camera_indices[0])
  if index < 0 or index >= len(targets):
    raise ValueError("initial camera state index exceeds controller-target stream")
  initial = targets[index].copy()
  if action_source == "published":
    initial[:7] = observed[0, :7]
  return initial, contract


def _linear_substep_action(
  start_action: np.ndarray,
  end_action: np.ndarray,
  *,
  interval_start_s: float,
  interval_end_s: float,
  current_time_s: float,
  timestep_s: float,
) -> np.ndarray:
  """Interpolate an endpoint target for the next physics integration step."""

  duration = interval_end_s - interval_start_s
  if duration <= 0 or timestep_s <= 0:
    raise ValueError("control interval and physics timestep must be positive")
  next_time = min(current_time_s + timestep_s, interval_end_s)
  alpha = float(np.clip((next_time - interval_start_s) / duration, 0.0, 1.0))
  return start_action + alpha * (end_action - start_action)


def _quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  if (
    a.shape != (4,)
    or b.shape != (4,)
    or not np.isfinite(a).all()
    or not np.isfinite(b).all()
  ):
    raise ValueError("quaternions must be finite four-vectors")
  a_norm = float(np.linalg.norm(a))
  b_norm = float(np.linalg.norm(b))
  if a_norm <= 0.0 or b_norm <= 0.0:
    raise ValueError("quaternions must have nonzero norm")
  a = a / a_norm
  b = b / b_norm
  return float(2.0 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0)))


def _initial_replay_state_validation(
  simulation: Any, source: h5py.File
) -> dict[str, Any]:
  """Prove that the deterministic reset reproduces the recorded initial state."""

  source_pose = np.asarray(source["objects/bulb/pose_wxyz"][0], dtype=float)
  actual_pose = np.asarray(simulation.object_pose("bulb"), dtype=float)
  if (
    source_pose.shape != (7,)
    or actual_pose.shape != (7,)
    or not np.isfinite(source_pose).all()
    or not np.isfinite(actual_pose).all()
    or np.linalg.norm(source_pose[3:]) <= 0.0
    or np.linalg.norm(actual_pose[3:]) <= 0.0
  ):
    raise ValueError("initial bulb poses must be finite xyz-wxyz vectors")

  source_joint_names = _decode_names(source["state/joint_names"])
  source_qpos = np.asarray(source["state/robot_joint_position"][0], dtype=float)
  source_qvel = np.asarray(source["state/robot_joint_velocity"][0], dtype=float)
  actual_joint_names, actual_qpos, actual_qvel = simulation.joint_state()
  actual_joint_names = tuple(actual_joint_names)
  actual_qpos = np.asarray(actual_qpos, dtype=float)
  actual_qvel = np.asarray(actual_qvel, dtype=float)
  if set(source_joint_names) != set(actual_joint_names):
    missing = sorted(set(actual_joint_names) - set(source_joint_names))
    extra = sorted(set(source_joint_names) - set(actual_joint_names))
    raise ValueError(
      "source and simulation initial robot joint sets differ: "
      f"missing={missing}, extra={extra}"
    )
  expected_shape = (len(source_joint_names),)
  if source_qpos.shape != expected_shape or source_qvel.shape != expected_shape:
    raise ValueError(
      "source initial robot qpos/qvel width must match state/joint_names"
    )
  if (
    actual_qpos.shape != (len(actual_joint_names),)
    or actual_qvel.shape != (len(actual_joint_names),)
    or not np.isfinite(source_qpos).all()
    or not np.isfinite(source_qvel).all()
    or not np.isfinite(actual_qpos).all()
    or not np.isfinite(actual_qvel).all()
  ):
    raise ValueError("initial robot qpos/qvel must be finite aligned vectors")
  source_columns = [source_joint_names.index(name) for name in actual_joint_names]
  qpos_error = actual_qpos - source_qpos[source_columns]
  qvel_error = actual_qvel - source_qvel[source_columns]
  errors = {
    "object_position_m": float(np.linalg.norm(actual_pose[:3] - source_pose[:3])),
    "object_orientation_rad": _quaternion_angle(actual_pose[3:], source_pose[3:]),
    "robot_qpos_rms_rad": float(np.sqrt(np.mean(qpos_error**2))),
    "robot_qpos_maximum_abs_rad": float(np.max(np.abs(qpos_error))),
    "robot_qvel_rms_rad_s": float(np.sqrt(np.mean(qvel_error**2))),
    "robot_qvel_maximum_abs_rad_s": float(np.max(np.abs(qvel_error))),
  }
  tolerances = {
    "object_position_m": INITIAL_OBJECT_POSITION_TOLERANCE_M,
    "object_orientation_rad": INITIAL_OBJECT_ORIENTATION_TOLERANCE_RAD,
    "robot_qpos_maximum_abs_rad": INITIAL_ROBOT_QPOS_TOLERANCE_RAD,
    "robot_qvel_maximum_abs_rad_s": INITIAL_ROBOT_QVEL_TOLERANCE_RAD_S,
  }
  failed = [name for name, tolerance in tolerances.items() if errors[name] > tolerance]
  report = {
    "status": "verified" if not failed else "mismatch",
    "joint_count": len(actual_joint_names),
    "errors": errors,
    "tolerances": tolerances,
    "failed_checks": failed,
  }
  if failed:
    detail = ", ".join(
      f"{name}={errors[name]:.9g} (limit {tolerances[name]:.9g})" for name in failed
    )
    raise RuntimeError(
      "source pickup seed did not reproduce the demonstration initial state: " + detail
    )
  return report


def replay_demo_actions(
  dataset_root: Path,
  episode_index: int,
  *,
  maximum_actions: int = 0,
  action_source: str = "published",
  control_interpolation: str = "zoh",
  control_rate_hz: int = CONTROL_HZ,
) -> dict[str, Any]:
  """Execute one demonstration action stream with the evaluation control path."""

  if action_source not in ACTION_SOURCES:
    raise ValueError(
      f"unknown action source {action_source!r}; expected one of {ACTION_SOURCES}"
    )
  if control_interpolation not in CONTROL_INTERPOLATIONS:
    raise ValueError(
      "unknown control interpolation "
      f"{control_interpolation!r}; expected one of {CONTROL_INTERPOLATIONS}"
    )
  if control_rate_hz not in CONTROL_RATES_HZ:
    raise ValueError(f"control_rate_hz must be one of {CONTROL_RATES_HZ}")
  if control_rate_hz == 100 and action_source != "recorded-controller-target":
    raise ValueError("100 Hz replay is only defined for recorded controller targets")
  parquet_path, source_hdf5, source_row = _dataset_episode_paths(
    dataset_root, episode_index
  )
  dataset_root = dataset_root.expanduser().resolve()
  info_path = dataset_root / "meta/info.json"
  source_manifest_path = dataset_root / "meta/kaihand_source_episodes.jsonl"
  input_files = {
    "info": _file_identity(info_path),
    "source_manifest": _file_identity(source_manifest_path),
    "parquet": _file_identity(parquet_path),
    "source_hdf5": _file_identity(source_hdf5),
  }
  expected_source_hash = source_row.get("source_hdf5_sha256")
  if (
    not isinstance(expected_source_hash, str)
    or len(expected_source_hash) != 64
    or any(character not in "0123456789abcdef" for character in expected_source_hash)
  ):
    raise ValueError("dataset source manifest has no valid source HDF5 SHA-256")
  if input_files["source_hdf5"]["sha256"] != expected_source_hash:
    raise ValueError("source HDF5 SHA-256 differs from the dataset source manifest")
  info = json.loads(info_path.read_text(encoding="utf-8"))
  joint_names = _feature_names(info, "action")
  if joint_names != _feature_names(info, "observation.state"):
    raise ValueError("action and observation state joint order differs")
  dataset_total_actions = _dataset_episode_length(dataset_root, episode_index)
  (
    reconstructed_actions,
    reconstructed_observed,
    reconstructed_action_source_indices,
  ) = _reconstruct_published_episode(source_hdf5, joint_names, dataset_total_actions)
  (
    published_actions,
    observed,
    action_source_indices,
    parquet_validation,
  ) = _validated_published_parquet(
    parquet_path,
    reconstructed_actions,
    reconstructed_observed,
    reconstructed_action_source_indices,
  )
  action_comparison = None
  if control_rate_hz == 100:
    (
      actions,
      observed,
      action_source_indices,
      initial_action,
      action_deadlines,
      controller_target_contract,
    ) = _source_rate_controller_target_episode(source_hdf5, joint_names)
    action_load_source = "source_hdf5_100hz_recorded_controller_targets"
    action_semantics = (
      "next-state-sample absolute controller targets; arm coordinates are "
      "commands/arm_joint_target and hand coordinates are "
      "commands/hand_joint_target"
    )
  else:
    if action_source == "published":
      actions = published_actions
      action_load_source = "published_parquet_float32_exact_hdf5_validation"
      action_semantics = (
        "published next-camera-frame absolute target; arm coordinates are measured "
        "qpos and hand coordinates are recorded controller targets"
      )
    else:
      actions = _recorded_controller_target_actions(
        source_hdf5, joint_names, action_source_indices
      )
      action_delta = actions - published_actions
      action_comparison = {
        "arm_rms_rad": float(np.sqrt(np.mean(action_delta[:, :7] ** 2))),
        "arm_maximum_abs_rad": float(np.max(np.abs(action_delta[:, :7]))),
        "hand_rms_rad": float(np.sqrt(np.mean(action_delta[:, 7:] ** 2))),
        "hand_maximum_abs_rad": float(np.max(np.abs(action_delta[:, 7:]))),
      }
      action_load_source = "source_hdf5_30hz_recorded_controller_targets"
      action_semantics = (
        "recorded controller target at each published action provenance row; arm "
        "coordinates are commands/arm_joint_target and hand coordinates are "
        "commands/hand_joint_target"
      )
    initial_action, controller_target_contract = _initial_30hz_target(
      source_hdf5, joint_names, observed, action_source
    )
    action_deadlines = np.arange(1, len(actions) + 1, dtype=float) / control_rate_hz
  total_actions = len(actions)
  count = total_actions if maximum_actions == 0 else min(total_actions, maximum_actions)
  if count <= 0:
    raise ValueError("demonstration has no actions to replay")
  actions = actions[:count]
  observed = observed[:count]
  action_source_indices = action_source_indices[:count]
  action_deadlines = action_deadlines[:count]

  position_seed = _source_position_seed(source_hdf5)
  adapter = create_task_policy_adapter("bulb-screw", position_seed)
  simulation = adapter.simulation
  lower, upper = joint_limits(simulation, joint_names)
  with h5py.File(source_hdf5, "r") as source:
    expected_fingerprint = str(source.attrs.get("model_fingerprint", ""))
    actual_fingerprint = model_fingerprint(simulation.model_path)
    if expected_fingerprint != actual_fingerprint:
      raise ValueError("source HDF5 model fingerprint differs from current bulb model")
    source_pose = np.asarray(source["objects/bulb/pose_wxyz"], dtype=float)
    source_time = np.asarray(source["state/timestamp"], dtype=float)
    source_engaged = np.asarray(source["bulb_screw/engaged"], dtype=bool)
    source_recording_contract = {
      "schema_version": str(source.attrs.get("schema_version", "")),
      "physics_hz": int(source.attrs.get("physics_hz", -1)),
      "state_and_command_hz": int(source.attrs.get("control_hz", -1)),
      "camera_and_published_action_hz": int(source.attrs.get("camera_hz", -1)),
      "measured_state_field": "state/robot_joint_position",
      "controller_targets": controller_target_contract,
    }
    if action_source_indices.max(initial=0) >= len(source_pose):
      raise ValueError("action provenance index exceeds source state stream")
    initial_state_validation = _initial_replay_state_validation(simulation, source)
    initial_errors = initial_state_validation["errors"]

    before_errors = []
    next_state_errors = []
    object_position_errors = []
    object_orientation_errors = []
    clipped_values = 0
    source_engagement_ticks = np.flatnonzero(source_engaged[action_source_indices])
    source_engagement_tick = (
      int(source_engagement_ticks[0]) if source_engagement_ticks.size else None
    )
    at_source_thread_engagement = None
    events: dict[str, Any] = {
      "thread_engaged": None,
      "seated": None,
      "evaluation_success": None,
    }
    previous_action = np.clip(initial_action, lower, upper)
    for tick, action in enumerate(actions):
      before_errors.append(right_joint_state(simulation, joint_names) - observed[tick])
      target_time = float(action_deadlines[tick])
      interval_start = float(simulation.data.time)
      clipped_action = np.clip(action, lower, upper)
      endpoint_clips = int(np.count_nonzero(np.abs(clipped_action - action) > 1.0e-9))
      clipped_values += endpoint_clips
      if control_interpolation == "zoh":
        applied_clips = apply_action(simulation, joint_names, action, lower, upper)
        if applied_clips != endpoint_clips:
          raise RuntimeError(
            "endpoint clipping count differs from evaluation controller"
          )
      while simulation.data.time < target_time - 1.0e-12:
        if control_interpolation == "linear":
          substep_action = _linear_substep_action(
            previous_action,
            clipped_action,
            interval_start_s=interval_start,
            interval_end_s=target_time,
            current_time_s=float(simulation.data.time),
            timestep_s=simulation.timestep,
          )
          if apply_action(simulation, joint_names, substep_action, lower, upper):
            raise RuntimeError("interpolation between clipped endpoints was clipped")
        adapter.step()
      previous_action = clipped_action
      current = right_joint_state(simulation, joint_names)
      if tick + 1 < len(observed):
        next_state_errors.append(current - observed[tick + 1])
      source_index = int(action_source_indices[tick])
      actual_pose = simulation.object_pose("bulb")
      expected_pose = source_pose[source_index]
      object_position_errors.append(
        float(np.linalg.norm(actual_pose[:3] - expected_pose[:3]))
      )
      object_orientation_errors.append(
        _quaternion_angle(actual_pose[3:], expected_pose[3:])
      )
      if tick == source_engagement_tick:
        source_engagement_state = {
          key: _jsonable(source[f"bulb_screw/{key}"][source_index])
          for key in (
            "engaged",
            "clockwise_turns",
            "insertion_depth_m",
            "lateral_error_m",
            "tilt_rad",
            "thread_error_m",
            "linear_speed_m_s",
            "angular_speed_rad_s",
            "thread_contact_load_n",
            "seated",
            "success",
            "bulb_lit",
          )
        }
        at_source_thread_engagement = {
          "action_index": tick,
          "replay_time_s": float(simulation.data.time),
          "source_index": source_index,
          "source_time_s": float(source_time[source_index]),
          "object_position_error_m": object_position_errors[-1],
          "object_orientation_error_rad": object_orientation_errors[-1],
          "source_state": source_engagement_state,
          "replay_state": asdict(adapter.state),
        }
      for name, condition in (
        ("thread_engaged", adapter.state.engaged),
        ("seated", adapter.state.seated),
        ("evaluation_success", adapter.success),
      ):
        if events[name] is None and condition:
          events[name] = {
            "action_index": tick,
            "replay_time_s": float(simulation.data.time),
            "source_time_s": float(source_time[source_index]),
          }

    before = np.asarray(before_errors)
    next_error = np.asarray(next_state_errors)
    object_position = np.asarray(object_position_errors)
    object_orientation = np.asarray(object_orientation_errors)
    source_final_index = int(action_source_indices[-1])
    source_final_state = {
      key: _jsonable(source[f"bulb_screw/{key}"][source_final_index])
      for key in (
        "engaged",
        "clockwise_turns",
        "insertion_depth_m",
        "lateral_error_m",
        "tilt_rad",
        "seated",
        "success",
        "bulb_lit",
      )
    }

  for identity in input_files.values():
    _assert_file_identity_unchanged(identity)

  def error_summary(error: np.ndarray) -> dict[str, float | None]:
    if not error.size:
      return {"rms_rad": None, "maximum_abs_rad": None}
    return {
      "rms_rad": float(np.sqrt(np.mean(error**2))),
      "maximum_abs_rad": float(np.max(np.abs(error))),
    }

  return {
    "schema": "bulb_demo_evaluation_controller_replay_v3",
    "diagnostic_only": True,
    "dataset_root": str(dataset_root.expanduser().resolve()),
    "episode_index": episode_index,
    "parquet": str(parquet_path),
    "source_hdf5": str(source_hdf5),
    "input_files": input_files,
    "source_manifest_hdf5_sha256": expected_source_hash,
    "published_parquet_validation": parquet_validation,
    "model_fingerprint": {
      "source_hdf5": expected_fingerprint,
      "current_simulation": actual_fingerprint,
      "exact": True,
    },
    "action_load_source": action_load_source,
    "source_position_seed": position_seed,
    "control_hz": control_rate_hz,
    "control_interpolation": control_interpolation,
    "physics_timestep_s": simulation.timestep,
    "physics_hz": int(round(1.0 / simulation.timestep)),
    "control_execution_contract": (
      "zero-order hold: apply each endpoint once, then hold through the control interval"
      if control_interpolation == "zoh"
      else "linear: interpolate the previous and current clipped 30/100 Hz endpoints "
      "at every 500 Hz physics step"
    ),
    "source_recording_contract": source_recording_contract,
    "action_semantics": action_semantics,
    "published_vs_recorded_controller_target": action_comparison,
    "dataset_30hz_actions_available": dataset_total_actions,
    "actions_available": total_actions,
    "actions_executed": count,
    "complete_episode": count == total_actions,
    "clipped_action_values": clipped_values,
    "initial_state_validation": initial_state_validation,
    "initial_object_position_error_m": initial_errors["object_position_m"],
    "initial_object_orientation_error_rad": initial_errors["object_orientation_rad"],
    "initial_robot_qpos_maximum_abs_error_rad": initial_errors[
      "robot_qpos_maximum_abs_rad"
    ],
    "initial_robot_qvel_maximum_abs_error_rad_s": initial_errors[
      "robot_qvel_maximum_abs_rad_s"
    ],
    "joint_error_at_action_start": error_summary(before),
    "joint_error_at_next_observation": error_summary(next_error),
    "object_trajectory_error": {
      "position_rms_m": float(np.sqrt(np.mean(object_position**2))),
      "position_maximum_m": float(np.max(object_position)),
      "orientation_rms_rad": float(np.sqrt(np.mean(object_orientation**2))),
      "orientation_maximum_rad": float(np.max(object_orientation)),
    },
    "at_source_thread_engagement": at_source_thread_engagement,
    "events": events,
    "source_final_state": source_final_state,
    "replay_final_state": asdict(adapter.state),
    "replay_final_object_pose_wxyz": simulation.object_pose("bulb").tolist(),
  }


def select_stage_index(file: h5py.File, stage: str) -> int:
  """Choose a stable source row for one diagnostic start condition."""

  if stage not in STAGES:
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
  timestamps = np.asarray(file["state/timestamp"], dtype=float)
  phases = np.asarray(file["commands/phase"].asstr(), dtype=object)
  engaged = np.asarray(file["bulb_screw/engaged"], dtype=bool)
  control_hz = file.attrs.get("control_hz")
  if not isinstance(control_hz, (int, np.integer)) or int(control_hz) <= 0:
    raise ValueError("source HDF5 must bind a positive integer control_hz")
  maximum_sample_gap_s = 1.5 / int(control_hz)
  if (
    timestamps.ndim != 1
    or phases.shape != timestamps.shape
    or engaged.shape != timestamps.shape
  ):
    raise ValueError("source phase/engagement streams must align with state/timestamp")
  if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
    raise ValueError("source state/timestamp must be finite and strictly increasing")

  def continuous(start: int, end: int) -> bool:
    return bool(
      end >= start
      and np.all(np.diff(timestamps[start : end + 1]) <= maximum_sample_gap_s)
    )

  if stage == "stable_grasp":
    candidates = np.flatnonzero(phases == "lift")
    if not candidates.size:
      raise ValueError("source episode has no lift phase")
    return int(candidates[-1])
  if stage == "socket_hover":
    candidates = np.flatnonzero(phases == "transfer")
    if not candidates.size:
      raise ValueError("source episode has no transfer phase")
    return int(candidates[-1])
  if stage in {"late_threaded_stable", "seated_loaded"}:
    fingertip = np.asarray(file["bulb_screw/fingertip_normal_force_n"], dtype=float)
    nonpad = np.asarray(file["bulb_screw/right_hand_nonpad_contact_count"], dtype=int)
    robot_contacts = np.asarray(file["bulb_screw/robot_contact_count"], dtype=int)
    hand_contacts = np.asarray(file["bulb_screw/right_hand_contact_count"], dtype=int)
    if fingertip.shape != (len(timestamps), 5):
      raise ValueError("source fingertip force stream must have shape [states, 5]")
    clean = (nonpad == 0) & (robot_contacts == hand_contacts)
    if stage == "late_threaded_stable":
      turns = np.asarray(file["bulb_screw/clockwise_turns"], dtype=float)
      mask = (
        engaged
        & (phases == "turn")
        & (np.min(fingertip, axis=1) >= 2.0)
        & (turns >= 0.20)
        & (turns <= bulb_config.TARGET_TURNS - 0.01)
        & clean
      )
      duration = 0.20
    else:
      seated = np.asarray(file["bulb_screw/seated"], dtype=bool)
      lit = np.asarray(file["bulb_screw/bulb_lit"], dtype=bool)
      torque = np.asarray(file["bulb_screw/hand_clockwise_torque_nm"], dtype=float)
      mask = (
        engaged
        & seated
        & ~lit
        & (phases == "tighten")
        & (np.min(fingertip, axis=1) >= 4.0)
        & (torque >= bulb_config.TIGHTENING_MIN_TORQUE_NM)
        & clean
      )
      duration = 0.10
    valid_ends = []
    for start in np.flatnonzero(mask):
      end = int(np.searchsorted(timestamps, timestamps[start] + duration - 1.0e-12))
      if (
        end < len(timestamps)
        and np.all(mask[start : end + 1])
        and continuous(start, end)
      ):
        valid_ends.append(end)
    if valid_ends:
      if stage == "late_threaded_stable":
        # Avoid a source state on the trailing edge of a stable contact block:
        # restoring qpos/qvel cannot restore MuJoCo's contact solver memory.
        # Require another full stable window after the candidate, then prefer
        # the candidate with the largest instantaneous all-finger margin.  The
        # post-settle contract below still verifies the real policy initial
        # state rather than trusting this source-only score.
        interior_ends = []
        for candidate in valid_ends:
          guard_end = int(
            np.searchsorted(timestamps, timestamps[candidate] + duration - 1.0e-12)
          )
          if (
            guard_end < len(timestamps)
            and np.all(mask[candidate : guard_end + 1])
            and continuous(candidate, guard_end)
          ):
            interior_ends.append(candidate)
        if interior_ends:
          return max(
            interior_ends,
            key=lambda candidate: np.min(fingertip[candidate]),
          )
        raise ValueError(
          "source episode has no late_threaded_stable state with 0.20s stable "
          "history and 0.20s stable forward margin"
        )
      return valid_ends[0]
    raise ValueError(f"source episode has no continuous {duration:.2f}s {stage} window")
  candidates = np.flatnonzero(engaged)
  if not candidates.size:
    raise ValueError("source episode never engages the bulb thread")
  first_engaged = int(candidates[0])
  target_time = timestamps[first_engaged] + 0.20
  index = int(np.searchsorted(timestamps, target_time - 1.0e-12, side="left"))
  if index >= len(timestamps):
    raise ValueError("source engagement is not retained for 0.20 seconds")
  retained = engaged[first_engaged : index + 1]
  if not np.all(retained) or not continuous(first_engaged, index):
    if np.all(retained):
      raise ValueError(
        "source engagement samples are too sparse to prove a continuous "
        "0.20 second window"
      )
    first_gap = first_engaged + int(np.flatnonzero(~retained)[0])
    raise ValueError(
      "source engagement is not continuous for 0.20 seconds; "
      f"first disengaged sample is index {first_gap} at {timestamps[first_gap]:.6f}s"
    )
  return index


def _stage_contract_validation(stage: str, state: Any) -> dict[str, Any]:
  """Validate the state actually handed to a staged policy rollout."""

  fingertip = tuple(float(value) for value in state.fingertip_normal_force_n)
  minimum_force = min(fingertip, default=0.0)
  scalar_values = {
    "clockwise_turns": float(state.clockwise_turns),
    "insertion_depth_m": float(state.insertion_depth_m),
    "lateral_error_m": float(state.lateral_error_m),
    "tilt_rad": float(state.tilt_rad),
    "linear_speed_m_s": float(state.linear_speed_m_s),
    "angular_speed_rad_s": float(state.angular_speed_rad_s),
    "hand_clockwise_torque_nm": float(state.hand_clockwise_torque_nm),
  }
  finite = bool(
    all(np.isfinite(value) for value in scalar_values.values())
    and len(fingertip) == 5
    and np.isfinite(fingertip).all()
  )
  clean_contact = (
    state.right_hand_nonpad_contact_count == 0
    and state.robot_contact_count == state.right_hand_contact_count
  )
  checks: dict[str, bool] = {"finite_state": finite}
  if stage in {"stable_grasp", "socket_hover"}:
    checks |= {
      "not_thread_engaged": not state.engaged,
      "not_seated": not state.seated,
      "five_fingertips_loaded_ge_2n": len(fingertip) == 5 and minimum_force >= 2.0,
      "clean_right_hand_contact": clean_contact,
    }
    if stage == "stable_grasp":
      checks |= {
        "away_from_socket": state.lateral_error_m >= 0.05,
        "above_thread_entrance": state.insertion_depth_m <= -0.03,
      }
    else:
      checks |= {
        "centered_over_socket": state.lateral_error_m <= 0.002,
        "above_thread_entrance": state.insertion_depth_m <= -0.03,
      }
  elif stage == "threaded":
    checks |= {
      "thread_engaged": state.engaged,
      "not_seated": not state.seated,
      "early_thread_progress": -0.03 <= state.clockwise_turns <= 0.03,
      "capture_aligned": state.lateral_error_m <= bulb_config.CAPTURE_LATERAL_M,
    }
  elif stage == "late_threaded_stable":
    checks |= {
      "thread_engaged": state.engaged,
      "not_seated": not state.seated,
      "late_thread_progress": (
        0.20 <= state.clockwise_turns <= bulb_config.TARGET_TURNS - 0.01
      ),
      "five_fingertips_loaded_ge_2n": len(fingertip) == 5 and minimum_force >= 2.0,
      "clean_right_hand_contact": clean_contact,
    }
  elif stage == "seated_loaded":
    checks |= {
      "thread_engaged": state.engaged,
      "seated": state.seated,
      "at_target_turns": state.clockwise_turns >= bulb_config.TARGET_TURNS - 0.01,
      "not_already_lit": not state.bulb_lit,
      "five_fingertips_loaded_ge_4n": len(fingertip) == 5 and minimum_force >= 4.0,
      "clockwise_torque_ge_threshold": (
        state.hand_clockwise_torque_nm >= bulb_config.TIGHTENING_MIN_TORQUE_NM
      ),
      "clean_right_hand_contact": clean_contact,
    }
  else:
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
  failed = [name for name, passed in checks.items() if not passed]
  return {
    "status": "verified" if not failed else "mismatch",
    "checks": checks,
    "failed_checks": failed,
    "minimum_fingertip_force_n": minimum_force,
    "hand_clockwise_torque_nm": float(state.hand_clockwise_torque_nm),
    "scalar_values": scalar_values,
  }


def _stage_diagnostic_scope(stage: str, state: Any) -> dict[str, Any]:
  common = {
    "inherits_expert_physics_state": True,
    "inherits_expert_hand_controller_targets": True,
    "formal_capability_claim": False,
  }
  scopes = {
    "stable_grasp": {
      "capabilities_bypassed": ["approach", "grasp_closure"],
      "remaining_work": [
        "transport",
        "socket_alignment",
        "thread_engagement",
        "thread_turning",
        "loaded_tightening",
        "clean_release",
      ],
    },
    "socket_hover": {
      "capabilities_bypassed": ["approach", "grasp_closure", "transport"],
      "remaining_work": [
        "descent_alignment",
        "thread_engagement",
        "thread_turning",
        "loaded_tightening",
        "clean_release",
      ],
    },
    "threaded": {
      "capabilities_bypassed": [
        "approach",
        "grasp_closure",
        "transport",
        "socket_alignment",
        "thread_capture",
      ],
      "remaining_work": [
        "thread_turning_from_early_regrasp",
        "loaded_tightening",
        "clean_release",
      ],
    },
    "late_threaded_stable": {
      "capabilities_bypassed": [
        "approach",
        "grasp_closure",
        "transport",
        "socket_alignment",
        "thread_capture",
        "most_thread_turning",
      ],
      "remaining_work": [
        "final_thread_turning",
        "loaded_tightening",
        "clean_release",
      ],
    },
    "seated_loaded": {
      "capabilities_bypassed": [
        "approach",
        "grasp_closure",
        "transport",
        "socket_alignment",
        "thread_capture",
        "thread_turning",
        "establish_terminal_preload",
      ],
      "remaining_work": [
        "maintain_clean_terminal_preload",
        "loaded_tightening_confirmation",
        "clean_release",
      ],
    },
  }
  return {
    **common,
    **scopes[stage],
    "initial_clockwise_turns": float(state.clockwise_turns),
    "target_clockwise_turns": float(bulb_config.TARGET_TURNS),
    "remaining_clockwise_turns": max(
      0.0, float(bulb_config.TARGET_TURNS - state.clockwise_turns)
    ),
  }


def restore_reference_stage(
  adapter: TaskPolicyAdapter,
  source_hdf5: Path,
  stage: str,
  *,
  settle_seconds: float = 0.10,
) -> dict[str, Any]:
  """Restore a source state and make controller memory consistent with it."""

  if adapter.task != "bulb-screw":
    raise ValueError("reference bulb stages require a bulb-screw adapter")
  if not np.isfinite(settle_seconds) or settle_seconds < 0:
    raise ValueError("settle_seconds must be finite and nonnegative")
  source_hdf5 = source_hdf5.expanduser().resolve(strict=True)
  source_position_seed = _source_position_seed(source_hdf5)
  simulation = adapter.simulation
  adapter_seed = simulation.initial_position_randomization.get("seed")
  if adapter_seed != source_position_seed:
    raise ValueError(
      "adapter pickup seed differs from the source episode: "
      f"{adapter_seed!r} != {source_position_seed!r}"
    )
  with h5py.File(source_hdf5, "r") as file:
    metadata = _metadata(file)
    if metadata.get("scene") != "bulb-screw":
      raise ValueError("source HDF5 is not a bulb-screw episode")
    expected_fingerprint = str(file.attrs.get("model_fingerprint", ""))
    actual_fingerprint = model_fingerprint(simulation.model_path)
    if expected_fingerprint != actual_fingerprint:
      raise ValueError("source HDF5 model fingerprint differs from current bulb model")
    expected_mechanics = str(metadata.get("mechanics_version", ""))
    if expected_mechanics != bulb_config.MECHANICS_VERSION:
      raise ValueError(
        "source HDF5 mechanics version differs from current bulb task: "
        f"{expected_mechanics!r} != {bulb_config.MECHANICS_VERSION!r}"
      )
    state_count = len(file["state/timestamp"])
    aligned_streams = (
      "commands/phase",
      "commands/hand_joint_target",
      "objects/bulb/pose_wxyz",
      "bulb_screw/eq_data",
      "bulb_screw/eq_active",
      "bulb_screw/engaged",
      "bulb_screw/clockwise_turns",
      "bulb_screw/seated",
      "bulb_screw/bulb_lit",
    )
    misaligned = [
      name
      for name in aligned_streams
      if file[name].ndim < 1 or file[name].shape[0] != state_count
    ]
    if misaligned:
      raise ValueError(
        f"source stage streams do not align with state/timestamp: {misaligned}"
      )
    index = select_stage_index(file, stage)
    qpos, qvel = _load_replay_states(file, simulation)
    if (
      qpos.shape[0] != state_count
      or qvel.shape[0] != state_count
      or not np.isfinite(qpos).all()
      or not np.isfinite(qvel).all()
    ):
      raise ValueError("source qpos/qvel streams must be finite and state-aligned")
    simulation.restore_full_state(qpos[index], qvel[index])
    eq_data = np.asarray(file["bulb_screw/eq_data"][index], dtype=float)
    eq_active = np.asarray(file["bulb_screw/eq_active"][index], dtype=bool)
    if eq_data.shape != simulation.model.eq_data.shape:
      raise ValueError("source equality data shape differs from current bulb model")
    if eq_active.shape != simulation.data.eq_active.shape:
      raise ValueError(
        "source equality activation shape differs from current bulb model"
      )
    if not np.isfinite(eq_data).all():
      raise ValueError("source equality data must be finite")
    simulation.model.eq_data[:] = eq_data
    simulation.data.eq_active[:] = eq_active
    simulation._set_bulb_lit(bool(file["bulb_screw/bulb_lit"][index]))
    hand_names = _decode_names(file["commands/hand_joint_names"])
    hand_targets = np.asarray(file["commands/hand_joint_target"][index], dtype=float)
    if hand_targets.shape != (len(hand_names),) or not np.isfinite(hand_targets).all():
      raise ValueError("source hand targets must be finite and name-aligned")
    accepted = simulation.set_hand_joint_targets(hand_names, hand_targets)
    if accepted != len(hand_names):
      raise ValueError(
        f"simulator accepted only {accepted}/{len(hand_names)} hand targets"
      )
    source_time = float(file["state/timestamp"][index])
    source_phase = str(file["commands/phase"].asstr()[index])
    source_pose = np.asarray(file["objects/bulb/pose_wxyz"][index], dtype=float)
    if (
      not np.isfinite(source_time)
      or source_pose.shape != (7,)
      or not np.isfinite(source_pose).all()
      or np.linalg.norm(source_pose[3:]) <= 0.0
    ):
      raise ValueError("selected source time and bulb pose must be finite and valid")
    source_state = {
      key: _jsonable(file[f"bulb_screw/{key}"][index])
      for key in (
        "engaged",
        "clockwise_turns",
        "insertion_depth_m",
        "lateral_error_m",
        "tilt_rad",
        "thread_contact_load_n",
        "seated",
        "bulb_lit",
      )
    }
    source_state["source_recorded_success"] = bool(file["bulb_screw/success"][index])
    source_state["source_recorded_success_semantics"] = (
      "expert recorder signal; not treated as formal policy success"
    )
    for key in (
      "fingertip_normal_force_n",
      "hand_clockwise_torque_nm",
      "right_hand_nonpad_contact_count",
      "right_hand_contact_count",
      "robot_contact_count",
      "tightening_verified",
      "released",
    ):
      dataset = f"bulb_screw/{key}"
      if dataset in file:
        source_state[key] = _jsonable(file[dataset][index])

  simulation.data.time = 0.0
  simulation.data.qfrc_applied.fill(0.0)
  simulation.data.xfrc_applied.fill(0.0)
  simulation.hold_current_arm_position("left")
  simulation.hold_current_arm_position("right")
  mujoco.mj_forward(simulation.model, simulation.data)
  restored_pose = simulation.object_pose("bulb")
  restore_position_error = float(np.linalg.norm(restored_pose[:3] - source_pose[:3]))
  restore_orientation_error = _quaternion_angle(restored_pose[3:], source_pose[3:])
  if restore_position_error > 1.0e-9 or restore_orientation_error > 1.0e-8:
    raise RuntimeError("restored bulb pose differs from selected source state")

  adapter.monitor.reset()
  adapter.state = adapter.monitor.measure()
  adapter._success = False
  initial_pose = restored_pose.copy()
  target_settle_time = settle_seconds
  while simulation.data.time < target_settle_time - 1.0e-12:
    adapter.step()
  settled_pose = simulation.object_pose("bulb")
  settled_state = asdict(adapter.state)
  settled_time = float(simulation.data.time)
  simulation.data.time = 0.0
  adapter.monitor.reset()
  adapter.state = adapter.monitor.measure()
  adapter._success = False
  adapter.initial_object_pose = simulation.object_pose("bulb").copy()
  policy_initial_state = asdict(adapter.state)
  stage_contract = _stage_contract_validation(stage, adapter.state)
  if stage_contract["failed_checks"]:
    failures = ", ".join(stage_contract["failed_checks"])
    raise RuntimeError(
      f"restored {stage} state violates its post-settle contract: {failures}"
    )
  return {
    "schema": "bulb_reference_stage_initialization_v4",
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "source_hdf5": str(source_hdf5),
    "model_fingerprint": actual_fingerprint,
    "mechanics_version": bulb_config.MECHANICS_VERSION,
    "stage": stage,
    "source_index": index,
    "source_time_s": source_time,
    "source_phase": source_phase,
    "source_state": source_state,
    "settle_seconds_requested": settle_seconds,
    "settle_seconds_executed": settled_time,
    "restore_position_error_m": restore_position_error,
    "restore_orientation_error_rad": restore_orientation_error,
    "settle_object_translation_m": float(
      np.linalg.norm(settled_pose[:3] - initial_pose[:3])
    ),
    "settle_object_rotation_rad": _quaternion_angle(settled_pose[3:], initial_pose[3:]),
    "settled_state": settled_state,
    "stage_contract_validation": stage_contract,
    "diagnostic_scope": _stage_diagnostic_scope(stage, adapter.state),
    "policy_initial_state": policy_initial_state,
    "policy_initial_object_pose_wxyz": adapter.initial_object_pose.tolist(),
  }


def summarize_stages(source_hdf5: Path, settle_seconds: float) -> dict[str, Any]:
  reports = {}
  resolved_source = source_hdf5.expanduser().resolve(strict=True)
  source_identity = _file_identity(resolved_source)
  try:
    position_seed = _source_position_seed(resolved_source)
    for stage in STAGES:
      adapter = create_task_policy_adapter("bulb-screw", position_seed)
      reports[stage] = restore_reference_stage(
        adapter, resolved_source, stage, settle_seconds=settle_seconds
      )
  finally:
    _assert_file_identity_unchanged(source_identity)
  return {
    "schema": "bulb_reference_stage_summary_v4",
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "source_hdf5": str(resolved_source),
    "source_hdf5_identity": source_identity,
    "source_position_seed": position_seed,
    "model_fingerprint": next(iter(reports.values()))["model_fingerprint"],
    "mechanics_version": bulb_config.MECHANICS_VERSION,
    "stages": reports,
  }


def _run_staged_policy(argv: list[str]) -> dict[str, Any]:
  parser = argparse.ArgumentParser(
    prog=f"{Path(sys.argv[0]).name} run-stage",
    description="Run the existing pi0.5 evaluator from one reference bulb stage.",
    add_help=False,
  )
  parser.add_argument("--source-hdf5", type=Path, required=True)
  parser.add_argument("--start-stage", choices=STAGES, required=True)
  parser.add_argument("--stage-settle-seconds", type=float, default=0.10)
  stage_args, runner_argv = parser.parse_known_args(argv)

  import run_shared_task_pi05_policy as runner

  runner_args = runner.parse_args(runner_argv)
  if runner_args.task != "bulb-screw":
    raise ValueError("run-stage requires --task bulb-screw")
  real_factory = runner.create_task_policy_adapter
  source_hdf5 = stage_args.source_hdf5.expanduser().resolve(strict=True)
  source_identity = _file_identity(source_hdf5)
  source_position_seed = _source_position_seed(source_hdf5)
  captured: dict[str, Any] = {}

  def staged_factory(task: str, seed: int, **kwargs: Any) -> TaskPolicyAdapter:
    adapter = real_factory(task, source_position_seed, **kwargs)
    captured["requested_evaluation_seed"] = seed
    captured["source_position_seed"] = source_position_seed
    captured["initialization"] = restore_reference_stage(
      adapter,
      source_hdf5,
      stage_args.start_stage,
      settle_seconds=stage_args.stage_settle_seconds,
    )
    captured["initialization"]["source_hdf5_sha256"] = source_identity["sha256"]
    return adapter

  runner.create_task_policy_adapter = staged_factory
  try:
    report = runner.run(
      runner_args,
      diagnostic_context="reference_stage_initialization",
    )
  finally:
    runner.create_task_policy_adapter = real_factory
    _assert_file_identity_unchanged(source_identity)
  captured["source_hdf5_identity"] = source_identity
  report["staged_initialization"] = captured
  summary = runner_args.output_dir.expanduser().resolve() / "summary.json"
  summary.write_text(
    json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  return report


def _replay_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Replay one published bulb demo")
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--episode-index", type=int, default=0)
  parser.add_argument("--maximum-actions", type=int, default=0)
  parser.add_argument(
    "--action-source",
    choices=ACTION_SOURCES,
    default="published",
    help="published converter action (default) or the source controller targets",
  )
  parser.add_argument(
    "--control-interpolation",
    choices=CONTROL_INTERPOLATIONS,
    default="zoh",
    help=(
      "hold each control endpoint (default), or linearly interpolate adjacent "
      "endpoints at every physics step"
    ),
  )
  parser.add_argument(
    "--control-rate-hz",
    type=int,
    choices=CONTROL_RATES_HZ,
    default=CONTROL_HZ,
    help=(
      "30 replays dataset-frame endpoints; 100 is a diagnostic replay of the "
      "raw recorded controller-target stream and requires recorded-controller-target"
    ),
  )
  parser.add_argument("--output", type=Path, required=True)
  return parser


def _stage_summary_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Validate five bulb stage presets")
  parser.add_argument("--source-hdf5", type=Path, required=True)
  parser.add_argument("--stage-settle-seconds", type=float, default=0.10)
  parser.add_argument("--output", type=Path, required=True)
  return parser


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if not argv or argv[0] not in {"replay-demo", "stage-summary", "run-stage"}:
    raise SystemExit(
      "usage: diagnose_bulb_replay_and_stages.py "
      "{replay-demo,stage-summary,run-stage} ..."
    )
  command, remaining = argv[0], argv[1:]
  if command == "replay-demo":
    args = _replay_parser().parse_args(remaining)
    if args.episode_index < 0 or args.maximum_actions < 0:
      raise SystemExit("episode-index and maximum-actions must be nonnegative")
    report = replay_demo_actions(
      args.dataset_root,
      args.episode_index,
      maximum_actions=args.maximum_actions,
      action_source=args.action_source,
      control_interpolation=args.control_interpolation,
      control_rate_hz=args.control_rate_hz,
    )
    _write_new_json(args.output, report)
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), flush=True)
    return 0
  if command == "stage-summary":
    args = _stage_summary_parser().parse_args(remaining)
    report = summarize_stages(args.source_hdf5, args.stage_settle_seconds)
    _write_new_json(args.output, report)
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), flush=True)
    return 0
  report = _run_staged_policy(remaining)
  print(
    json.dumps(
      {
        "status": report.get("status"),
        "output_dir": report.get("output_dir"),
        "stage": report["staged_initialization"]["initialization"]["stage"],
      },
      ensure_ascii=False,
    ),
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
