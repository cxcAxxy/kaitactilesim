"""Static rollout/reference comparison plots.

The reference is reconstructed from the raw HDF5 episode named by either the
canonical ``meta/kaihand_source_episodes.jsonl`` manifest or the provenance in
a fast pi0.5 conversion manifest.  This intentionally follows the converter's
observation alignment instead of reading the converted Parquet file: the first
``exported_frames_30hz`` head-camera rows are observations and
``cameras/head/state_index`` selects their state-clock data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SCHEMA_VERSION = "kaihand-openwam-evaluation-plots-v1"
SOURCE_MANIFEST = Path("meta/kaihand_source_episodes.jsonl")
FAST_CONVERSION_MANIFEST = Path("meta/kaihand_fast_pi05_conversion.json")
FAST_CONVERSION_SCHEMA = "kaihand_fast_pi05_video_v1"

RIGHT_HAND_ACTUATED_JOINT_NAMES = (
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
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
RIGHT_FINGERTIP_LINK_NAMES = tuple(
  f"hand_r_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
  for finger in FINGER_NAMES
)
NATIVE_RIGHT_WRIST_SITE = "hand_r_base_link_site"

WRIST_COMPONENT_NAMES = (
  "x",
  "y",
  "z",
  "rot6d c1.x",
  "rot6d c1.y",
  "rot6d c1.z",
  "rot6d c2.x",
  "rot6d c2.y",
  "rot6d c2.z",
)

_REFERENCE_COLOR = "#1f77b4"
_ROLLOUT_COLOR = "#d62728"


@dataclass(frozen=True)
class OpenWAMReference:
  """Camera-aligned reference episode data used by all three plots."""

  dataset_root: Path
  output_episode_index: int
  source_hdf5: Path
  exported_frames_30hz: int
  source_start_time_s: float
  time_s: np.ndarray
  wrist_state: np.ndarray
  hand_joint_position: np.ndarray
  fingertip_normal_force_n: np.ndarray
  fingertip_tangent_force_n: np.ndarray
  joint_names: tuple[str, ...]

  @property
  def state_29(self) -> np.ndarray:
    """Return ``[native wrist xyz, rot6d, 20 actuated hand DoF]``."""
    return np.concatenate((self.wrist_state, self.hand_joint_position), axis=1)


def _validate_joint_names(joint_names: Sequence[str]) -> tuple[str, ...]:
  names = tuple(joint_names)
  if len(names) != 20:
    raise ValueError(f"OpenWAM requires 20 right-hand actuated joint names, got {len(names)}")
  if any(not isinstance(name, str) or not name for name in names):
    raise ValueError("right-hand actuated joint names must be non-empty strings")
  if len(set(names)) != len(names):
    raise ValueError("right-hand actuated joint names must be distinct")
  return names


def _decode_names(dataset: h5py.Dataset, *, context: str) -> tuple[str, ...]:
  values = np.asarray(dataset[:])
  if values.ndim != 1:
    raise ValueError(f"{context} must be one-dimensional, got {values.shape}")
  names: list[str] = []
  for value in values:
    if isinstance(value, bytes):
      try:
        name = value.decode("utf-8")
      except UnicodeDecodeError as error:
        raise ValueError(f"{context} contains invalid UTF-8") from error
    elif isinstance(value, str):
      name = value
    else:
      raise ValueError(f"{context} contains a non-string value")
    if not name:
      raise ValueError(f"{context} contains an empty name")
    names.append(name)
  if len(names) != len(set(names)):
    raise ValueError(f"{context} contains duplicate names")
  return tuple(names)


def _json_string_list_attribute(
  group: h5py.Group,
  name: str,
  *,
  expected_length: int,
) -> tuple[str, ...]:
  if name not in group.attrs:
    raise ValueError(f"{group.name} is missing required attribute {name}")
  encoded = group.attrs[name]
  if isinstance(encoded, bytes):
    encoded = encoded.decode("utf-8")
  if not isinstance(encoded, str):
    raise ValueError(f"{group.name} attribute {name} must be JSON text")
  try:
    values = json.loads(encoded)
  except json.JSONDecodeError as error:
    raise ValueError(f"{group.name} attribute {name} is invalid JSON") from error
  if (
    not isinstance(values, list)
    or len(values) != expected_length
    or any(not isinstance(value, str) or not value for value in values)
  ):
    raise ValueError(
      f"{group.name} attribute {name} must contain {expected_length} string names"
    )
  if len(values) != len(set(values)):
    raise ValueError(f"{group.name} attribute {name} contains duplicate names")
  return tuple(values)


def _require_dataset(file: h5py.File, path: str) -> h5py.Dataset:
  if path not in file or not isinstance(file[path], h5py.Dataset):
    raise ValueError(f"{file.filename}: missing required dataset {path}")
  return file[path]


def _columns(
  available: Sequence[str],
  requested: Sequence[str],
  *,
  context: str,
) -> np.ndarray:
  missing = [name for name in requested if name not in available]
  if missing:
    raise ValueError(f"{context} is missing names: {missing}")
  return np.asarray([available.index(name) for name in requested], dtype=np.intp)


def _read_manifest_record(
  dataset_root: Path,
  output_episode_index: int,
) -> tuple[Path, int]:
  if (
    isinstance(output_episode_index, bool)
    or not isinstance(output_episode_index, int)
    or output_episode_index < 0
  ):
    raise ValueError("output_episode_index must be a nonnegative integer")
  manifest = dataset_root / SOURCE_MANIFEST
  fast_manifest = dataset_root / FAST_CONVERSION_MANIFEST
  selected: Mapping[str, Any] | None = None
  context: Path = manifest
  if manifest.is_file():
    seen: set[int] = set()
    with manifest.open("r", encoding="utf-8") as stream:
      for line_number, line in enumerate(stream, start=1):
        if not line.strip():
          continue
        try:
          record = json.loads(line)
        except json.JSONDecodeError as error:
          raise ValueError(f"{manifest}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict):
          raise ValueError(f"{manifest}:{line_number}: row must be a JSON object")
        index = record.get("output_episode_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
          raise ValueError(
            f"{manifest}:{line_number}: output_episode_index must be a nonnegative integer"
          )
        if index in seen:
          raise ValueError(
            f"{manifest}:{line_number}: duplicate output_episode_index {index}"
          )
        seen.add(index)
        if index == output_episode_index:
          selected = record
  elif fast_manifest.is_file():
    context = fast_manifest
    try:
      payload = json.loads(fast_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ValueError(f"cannot read fast pi0.5 conversion manifest: {fast_manifest}") from error
    if not isinstance(payload, dict):
      raise ValueError(f"{fast_manifest}: expected a JSON object")
    if payload.get("schema") != FAST_CONVERSION_SCHEMA:
      raise ValueError(
        f"{fast_manifest}: expected schema {FAST_CONVERSION_SCHEMA!r}"
      )
    records = payload.get("episodes")
    if not isinstance(records, list) or not records:
      raise ValueError(f"{fast_manifest}: episodes must be a nonempty list")
    seen = set()
    for position, record in enumerate(records):
      if not isinstance(record, dict):
        raise ValueError(
          f"{fast_manifest}: episodes[{position}] must be a JSON object"
        )
      index = record.get("output_episode_index")
      if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(
          f"{fast_manifest}: episodes[{position}].output_episode_index "
          "must be a nonnegative integer"
        )
      if index in seen:
        raise ValueError(
          f"{fast_manifest}: duplicate output_episode_index {index}"
        )
      seen.add(index)
      if index == output_episode_index:
        selected = record
    if selected is not None and not selected.get("source_hdf5"):
      source_index = selected.get("source_episode_index")
      if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index < 0
      ):
        raise ValueError(
          f"{fast_manifest}: episode {output_episode_index} has invalid "
          "source_episode_index"
        )
      input_value = payload.get("input_dir")
      task = payload.get("task")
      if not isinstance(input_value, str) or not input_value:
        raise ValueError(f"{fast_manifest}: input_dir must be a nonempty string")
      if not isinstance(task, str) or not task or Path(task).name != task:
        raise ValueError(f"{fast_manifest}: task must be one directory name")
      input_dir = Path(input_value).expanduser()
      if not input_dir.is_absolute():
        input_dir = dataset_root / input_dir
      input_dir = input_dir.resolve(strict=True)
      if not input_dir.is_dir():
        raise ValueError(f"{fast_manifest}: input_dir is not a directory: {input_dir}")
      summary_path = input_dir / "summary.json"
      try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
      except FileNotFoundError as error:
        raise FileNotFoundError(
          f"fast pi0.5 raw collection summary is missing: {summary_path}"
        ) from error
      except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read raw collection summary: {summary_path}") from error
      if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
        raise ValueError(f"{summary_path}: episodes must be a list")
      matches = []
      for position, record in enumerate(summary["episodes"]):
        if not isinstance(record, dict):
          raise ValueError(
            f"{summary_path}: episodes[{position}] must be a JSON object"
          )
        if (
          record.get("task") == task
          and record.get("status") == "success"
          and record.get("episode_index") == source_index
        ):
          matches.append(record)
      if not matches:
        raise FileNotFoundError(
          f"{summary_path}: no successful raw source record for task={task!r}, "
          f"episode_index={source_index}"
        )
      if len(matches) != 1:
        raise ValueError(
          f"{summary_path}: task={task!r}, episode_index={source_index} maps to "
          f"{len(matches)} successful raw records"
        )
      hdf5_values = matches[0].get("hdf5")
      if (
        not isinstance(hdf5_values, list)
        or len(hdf5_values) != 1
        or not isinstance(hdf5_values[0], str)
        or not hdf5_values[0]
      ):
        raise ValueError(
          f"{summary_path}: successful source record must name exactly one HDF5 file"
        )
      relative_hdf5 = Path(hdf5_values[0])
      if relative_hdf5.is_absolute():
        raise ValueError(f"{summary_path}: source HDF5 path must be relative")
      try:
        source_hdf5 = (input_dir / relative_hdf5).resolve(strict=True)
      except FileNotFoundError as error:
        raise FileNotFoundError(
          f"source_hdf5 for output episode {output_episode_index} does not exist: "
          f"{input_dir / relative_hdf5}"
        ) from error
      if not source_hdf5.is_relative_to(input_dir):
        raise ValueError(f"{summary_path}: source HDF5 escapes input_dir")
      if not source_hdf5.is_file():
        raise ValueError(f"source_hdf5 is not a file: {source_hdf5}")
      selected = {**selected, "source_hdf5": str(source_hdf5)}
  else:
    raise FileNotFoundError(
      "missing reference provenance manifest; expected either "
      f"{manifest} or {fast_manifest}"
    )

  if selected is None:
    raise ValueError(
      f"{context}: no record for output_episode_index {output_episode_index}"
    )
  source_value = selected.get("source_hdf5")
  if not isinstance(source_value, str) or not source_value:
    raise ValueError(
      f"{context}: episode {output_episode_index} has invalid source_hdf5"
    )
  exported_frames = selected.get(
    "exported_frames_30hz", selected.get("frames")
  )
  if (
    isinstance(exported_frames, bool)
    or not isinstance(exported_frames, int)
    or exported_frames < 1
  ):
    raise ValueError(
      f"{context}: episode {output_episode_index} has invalid exported_frames_30hz"
    )
  source_hdf5 = Path(source_value).expanduser()
  if not source_hdf5.is_absolute():
    source_hdf5 = dataset_root / source_hdf5
  try:
    source_hdf5 = source_hdf5.resolve(strict=True)
  except FileNotFoundError as error:
    raise FileNotFoundError(
      f"source_hdf5 for output episode {output_episode_index} does not exist: "
      f"{source_hdf5}"
    ) from error
  if not source_hdf5.is_file():
    raise ValueError(f"source_hdf5 is not a file: {source_hdf5}")
  return source_hdf5, exported_frames


def aggregate_fingertip_forces(
  normal_taxel_force_n: np.ndarray,
  tangent_taxel_force_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return per-finger ``Fn`` and ``|Ft|`` with vector-sum cancellation.

  Inputs may contain leading dimensions but must end in five canonical right
  fingertips and their 7x5 grids.  ``Fn`` is the scalar taxel sum.  ``|Ft|``
  is computed by first summing each signed tangent component over all taxels,
  then taking ``sqrt(sum(Ft_col)^2 + sum(Ft_row)^2)``.
  """
  normal = np.asarray(normal_taxel_force_n, dtype=np.float64)
  tangent = np.asarray(tangent_taxel_force_n, dtype=np.float64)
  if normal.ndim < 3 or normal.shape[-3:] != (5, 7, 5):
    raise ValueError(
      "normal tactile force must end in canonical shape (5, 7, 5), "
      f"got {normal.shape}"
    )
  if tangent.shape != (*normal.shape, 2):
    raise ValueError(
      f"tangent tactile force must have shape {(*normal.shape, 2)}, got {tangent.shape}"
    )
  if not np.isfinite(normal).all() or not np.isfinite(tangent).all():
    raise ValueError("tactile force must be finite")
  if np.any(normal < -1.0e-12):
    raise ValueError("normal tactile force must be nonnegative")
  normal = np.maximum(normal, 0.0)
  normal_sum = normal.sum(axis=(-2, -1))
  tangent_vector_sum = tangent.sum(axis=(-3, -2))
  tangent_resultant = np.linalg.norm(tangent_vector_sum, axis=-1)
  return normal_sum, tangent_resultant


def _validate_world_from_wrist(matrices: np.ndarray, *, context: str) -> None:
  if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
    raise ValueError(f"{context} must have shape (T, 4, 4), got {matrices.shape}")
  if not np.isfinite(matrices).all():
    raise ValueError(f"{context} must be finite")
  expected_bottom = np.broadcast_to(
    np.asarray([0.0, 0.0, 0.0, 1.0]), matrices[:, 3, :].shape
  )
  if not np.allclose(matrices[:, 3, :], expected_bottom, atol=1.0e-7):
    raise ValueError(f"{context} has invalid homogeneous bottom rows")
  rotation = matrices[:, :3, :3]
  identity = np.broadcast_to(np.eye(3), rotation.shape)
  if not np.allclose(np.swapaxes(rotation, 1, 2) @ rotation, identity, atol=1.0e-5):
    raise ValueError(f"{context} rotations are not orthonormal")
  if not np.allclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
    raise ValueError(f"{context} rotations are not proper")


def _rot6d_from_matrix(rotation: np.ndarray) -> np.ndarray:
  return np.concatenate((rotation[:, :, 0], rotation[:, :, 1]), axis=1)


def load_openwam_reference(
  dataset_root: str | Path,
  output_episode_index: int = 0,
  *,
  joint_names: Sequence[str] = RIGHT_HAND_ACTUATED_JOINT_NAMES,
) -> OpenWAMReference:
  """Load a raw-HDF5 reference using the LeRobot v3 conversion contract."""
  root = Path(dataset_root).expanduser().resolve()
  if not root.is_dir():
    raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
  requested_joints = _validate_joint_names(joint_names)
  source_hdf5, frame_count = _read_manifest_record(root, output_episode_index)

  with h5py.File(source_hdf5, "r") as file:
    head = file.get("cameras/head")
    if not isinstance(head, h5py.Group):
      raise ValueError(f"{source_hdf5}: missing required group cameras/head")

    camera_times_dataset = _require_dataset(file, "cameras/head/timestamp")
    state_indices_dataset = _require_dataset(file, "cameras/head/state_index")
    if camera_times_dataset.ndim != 1 or state_indices_dataset.ndim != 1:
      raise ValueError("head timestamp and state_index must be one-dimensional")
    if camera_times_dataset.shape != state_indices_dataset.shape:
      raise ValueError("head timestamp and state_index must have matching shapes")
    if state_indices_dataset.dtype.kind not in "iu":
      raise ValueError("cameras/head/state_index must have integer dtype")
    required_camera_rows = frame_count + 1
    if (
      camera_times_dataset.shape[0] < required_camera_rows
      or state_indices_dataset.shape[0] < required_camera_rows
    ):
      raise ValueError(
        "head camera streams must contain exported_frames_30hz + 1 rows "
        "for observation/action alignment"
      )
    camera_times = np.asarray(
      camera_times_dataset[:required_camera_rows], dtype=np.float64
    )
    state_indices = np.asarray(
      state_indices_dataset[:required_camera_rows], dtype=np.int64
    )
    if not np.isfinite(camera_times).all() or not np.all(np.diff(camera_times) > 0):
      raise ValueError("head camera timestamps must be finite and strictly increasing")
    if np.any(state_indices < 0) or not np.all(np.diff(state_indices) > 0):
      raise ValueError("head camera state_index must be nonnegative and strictly increasing")

    state_times_dataset = _require_dataset(file, "state/timestamp")
    state_times = np.asarray(state_times_dataset[:], dtype=np.float64)
    if state_times.ndim != 1 or state_times.shape[0] < 1:
      raise ValueError("state/timestamp must be a nonempty vector")
    if not np.isfinite(state_times).all() or not np.all(np.diff(state_times) > 0):
      raise ValueError("state/timestamp must be finite and strictly increasing")
    if state_indices[-1] >= len(state_times):
      raise ValueError("head camera state_index is outside state/timestamp")
    expected_indices = np.searchsorted(state_times, camera_times, side="right") - 1
    if not np.array_equal(state_indices, expected_indices):
      raise ValueError(
        "cameras/head/state_index must select the latest nonfuture state sample"
      )
    observation_indices = state_indices[:-1]

    joint_name_dataset = _require_dataset(file, "state/joint_names")
    available_joints = _decode_names(
      joint_name_dataset, context="state/joint_names"
    )
    joint_columns = _columns(
      available_joints, requested_joints, context="state/joint_names"
    )
    joint_dataset = _require_dataset(file, "state/robot_joint_position")
    if joint_dataset.shape != (len(state_times), len(available_joints)):
      raise ValueError(
        "state/robot_joint_position must have shape "
        f"({len(state_times)}, {len(available_joints)}), got {joint_dataset.shape}"
      )
    if observation_indices[-1] >= joint_dataset.shape[0]:
      raise ValueError("observation state_index is outside robot_joint_position")
    joint_position = np.asarray(
      joint_dataset[observation_indices][:, joint_columns], dtype=np.float64
    )
    if not np.isfinite(joint_position).all():
      raise ValueError("reference right-hand joint positions must be finite")

    wrist_dataset = _require_dataset(file, "cameras/head/world_from_wrist")
    if (
      wrist_dataset.ndim != 4
      or wrist_dataset.shape[0] != camera_times_dataset.shape[0]
      or wrist_dataset.shape[2:] != (4, 4)
    ):
      raise ValueError(
        "cameras/head/world_from_wrist must have shape (camera, side, 4, 4)"
      )
    side_count = wrist_dataset.shape[1]
    side_names = _json_string_list_attribute(
      head, "side_names_json", expected_length=side_count
    )
    wrist_sites = _json_string_list_attribute(
      head, "wrist_sites_json", expected_length=side_count
    )
    if NATIVE_RIGHT_WRIST_SITE not in wrist_sites:
      raise ValueError(
        f"cameras/head does not record native site {NATIVE_RIGHT_WRIST_SITE}"
      )
    right_side = wrist_sites.index(NATIVE_RIGHT_WRIST_SITE)
    if side_names[right_side] != "right":
      raise ValueError(
        f"native site {NATIVE_RIGHT_WRIST_SITE} is not labeled as the right side"
      )
    wrist_matrices = np.asarray(
      wrist_dataset[:frame_count, right_side], dtype=np.float64
    )
    _validate_world_from_wrist(
      wrist_matrices, context=f"{NATIVE_RIGHT_WRIST_SITE} world pose"
    )
    wrist_state = np.concatenate(
      (
        wrist_matrices[:, :3, 3],
        _rot6d_from_matrix(wrist_matrices[:, :3, :3]),
      ),
      axis=1,
    )

    tactile_names = _decode_names(
      _require_dataset(file, "tactile_contact_force/link_names"),
      context="tactile_contact_force/link_names",
    )
    tactile_columns = _columns(
      tactile_names,
      RIGHT_FINGERTIP_LINK_NAMES,
      context="tactile_contact_force/link_names",
    )
    normal_dataset = _require_dataset(
      file, "tactile_contact_force/normal_taxel_force_n"
    )
    tangent_dataset = _require_dataset(
      file, "tactile_contact_force/tangent_taxel_force_n"
    )
    expected_normal_tail = (len(tactile_names), 7, 5)
    if normal_dataset.shape != (len(state_times), *expected_normal_tail):
      raise ValueError(
        "normal_taxel_force_n must have shape "
        f"({len(state_times)}, {len(tactile_names)}, 7, 5), "
        f"got {normal_dataset.shape}"
      )
    if tangent_dataset.shape != (*normal_dataset.shape, 2):
      raise ValueError(
        "tangent_taxel_force_n must have shape "
        f"{(*normal_dataset.shape, 2)}, got {tangent_dataset.shape}"
      )
    if observation_indices[-1] >= normal_dataset.shape[0]:
      raise ValueError("observation state_index is outside tactile force streams")
    normal = np.asarray(
      normal_dataset[observation_indices][:, tactile_columns], dtype=np.float64
    )
    tangent = np.asarray(
      tangent_dataset[observation_indices][:, tactile_columns], dtype=np.float64
    )
    normal_force, tangent_force = aggregate_fingertip_forces(normal, tangent)

  reference_time = camera_times[:-1] - camera_times[0]
  return OpenWAMReference(
    dataset_root=root,
    output_episode_index=output_episode_index,
    source_hdf5=source_hdf5,
    exported_frames_30hz=frame_count,
    source_start_time_s=float(camera_times[0]),
    time_s=reference_time,
    wrist_state=wrist_state,
    hand_joint_position=joint_position,
    fingertip_normal_force_n=normal_force,
    fingertip_tangent_force_n=tangent_force,
    joint_names=requested_joints,
  )


def _sample_field(sample: Any, name: str) -> Any:
  if isinstance(sample, Mapping):
    if name not in sample:
      raise ValueError(f"tactile sample is missing {name}")
    return sample[name]
  if not hasattr(sample, name):
    raise ValueError(f"tactile sample is missing {name}")
  return getattr(sample, name)


def _canonical_right_sample(sample: Any) -> tuple[np.ndarray, np.ndarray]:
  links = tuple(_sample_field(sample, "link_names"))
  if any(not isinstance(name, str) or not name for name in links):
    raise ValueError("tactile sample link_names must be non-empty strings")
  if len(links) != len(set(links)):
    raise ValueError("tactile sample link_names must be distinct")
  indices = _columns(
    links, RIGHT_FINGERTIP_LINK_NAMES, context="tactile sample link_names"
  )
  normal = np.asarray(_sample_field(sample, "normal_taxel_force_n"), dtype=np.float64)
  tangent = np.asarray(
    _sample_field(sample, "tangent_taxel_force_n"), dtype=np.float64
  )
  if normal.shape != (len(links), 7, 5):
    raise ValueError(
      f"sample normal tactile force must have shape ({len(links)}, 7, 5), "
      f"got {normal.shape}"
    )
  if tangent.shape != (len(links), 7, 5, 2):
    raise ValueError(
      f"sample tangent tactile force must have shape ({len(links)}, 7, 5, 2), "
      f"got {tangent.shape}"
    )
  return normal[indices], tangent[indices]


def _draw_series(
  axes: np.ndarray,
  reference_time: np.ndarray,
  reference_values: np.ndarray,
  rollout_time: np.ndarray,
  rollout_values: np.ndarray,
  titles: Sequence[str],
  *,
  y_label: str,
  figure_title: str,
  reference_episode_index: int,
  output: Path,
) -> None:
  figure = axes.flat[0].figure
  reference_line = rollout_line = None
  for index, (axis, title) in enumerate(zip(axes.flat, titles, strict=True)):
    (reference_line,) = axis.plot(
      reference_time,
      reference_values[:, index],
      color=_REFERENCE_COLOR,
      linestyle="-",
      linewidth=1.5,
      label="Reference episode",
    )
    (rollout_line,) = axis.plot(
      rollout_time,
      rollout_values[:, index],
      color=_ROLLOUT_COLOR,
      linestyle="--",
      linewidth=1.3,
      label="Evaluation rollout",
    )
    axis.set_title(title, fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)
    if index % axes.shape[1] == 0:
      axis.set_ylabel(y_label, fontsize=8)
    if index // axes.shape[1] == axes.shape[0] - 1:
      axis.set_xlabel("elapsed time [s]", fontsize=8)
    axis.tick_params(labelsize=7)
  assert reference_line is not None and rollout_line is not None
  figure.suptitle(figure_title, fontsize=14, y=0.988)
  figure.legend(
    (reference_line, rollout_line),
    (
      f"Reference (dataset episode {reference_episode_index:02d})",
      "Evaluation rollout",
    ),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncols=2,
    frameon=False,
  )
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925), pad=1.2)
  FigureCanvasAgg(figure).print_png(str(output))
  figure.clear()


class OpenWAMEvaluationPlots:
  """Accumulate one OpenWAM rollout and export three static comparisons."""

  def __init__(
    self,
    reference: OpenWAMReference,
    tactile_provider: Any | None = None,
    *,
    artifact_prefix: str = "openwam",
  ) -> None:
    if reference.state_29.shape != (reference.exported_frames_30hz, 29):
      raise ValueError(
        "reference state must have shape "
        f"({reference.exported_frames_30hz}, 29), got {reference.state_29.shape}"
      )
    self.reference = reference
    self.tactile_provider = tactile_provider
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", artifact_prefix):
      raise ValueError("artifact_prefix must be a nonempty filename-safe token")
    self.artifact_prefix = artifact_prefix
    self._times: list[float] = []
    self._states: list[np.ndarray] = []
    self._normal: list[np.ndarray] = []
    self._tangent: list[np.ndarray] = []
    self._finished = False

  @classmethod
  def from_dataset(
    cls,
    dataset_root: str | Path,
    episode_index: int = 0,
    tactile_provider: Any | None = None,
    *,
    joint_names: Sequence[str] = RIGHT_HAND_ACTUATED_JOINT_NAMES,
    artifact_prefix: str = "openwam",
  ) -> OpenWAMEvaluationPlots:
    """Create a recorder from a LeRobot v3 dataset reference episode."""
    reference = load_openwam_reference(
      dataset_root,
      output_episode_index=episode_index,
      joint_names=joint_names,
    )
    return cls(
      reference,
      tactile_provider=tactile_provider,
      artifact_prefix=artifact_prefix,
    )

  @property
  def sample_count(self) -> int:
    return len(self._times)

  def capture(
    self,
    simulation_time_s: float,
    state_29: np.ndarray,
    sample: Any | None = None,
    *,
    normal_taxel_force_n: np.ndarray | None = None,
    tangent_taxel_force_n: np.ndarray | None = None,
    simulation: Any | None = None,
  ) -> None:
    """Capture one state/tactile sample without stepping the simulation.

    Tactile input can be supplied either as canonical five-finger ``normal``
    and ``tangent`` arrays, as a provider sample with ``link_names``, or by
    passing ``simulation=...`` when this recorder has a tactile provider.
    """
    if self._finished:
      raise RuntimeError("OpenWAM evaluation plots have already been finished")
    if isinstance(simulation_time_s, bool):
      raise ValueError("simulation_time_s must be a finite scalar")
    try:
      timestamp = float(simulation_time_s)
    except (TypeError, ValueError) as error:
      raise ValueError("simulation_time_s must be a finite scalar") from error
    if not np.isfinite(timestamp):
      raise ValueError("simulation_time_s must be a finite scalar")
    if self._times and timestamp <= self._times[-1]:
      raise ValueError("rollout simulation times must be strictly increasing")

    state = np.asarray(state_29, dtype=np.float64)
    if state.shape != (29,):
      raise ValueError(f"OpenWAM state must have shape (29,), got {state.shape}")
    if not np.isfinite(state).all():
      raise ValueError("OpenWAM state must be finite")

    direct_normal = normal_taxel_force_n is not None
    direct_tangent = tangent_taxel_force_n is not None
    if direct_normal != direct_tangent:
      raise ValueError("normal and tangent tactile arrays must be supplied together")
    if sample is not None and direct_normal:
      raise ValueError("pass either a tactile sample or direct tactile arrays, not both")
    if sample is not None and simulation is not None:
      raise ValueError("simulation is only used when obtaining a provider sample")
    if sample is None and not direct_normal:
      if self.tactile_provider is None:
        raise ValueError("capture needs tactile arrays, a sample, or a tactile provider")
      if simulation is None:
        raise ValueError("simulation is required to read the configured tactile provider")
      data = getattr(simulation, "data", simulation)
      sample = self.tactile_provider.read(data)

    if sample is not None:
      normal, tangent = _canonical_right_sample(sample)
    else:
      normal = np.asarray(normal_taxel_force_n, dtype=np.float64)
      tangent = np.asarray(tangent_taxel_force_n, dtype=np.float64)
      if normal.shape != (5, 7, 5):
        raise ValueError(
          f"direct normal tactile force must have shape (5, 7, 5), got {normal.shape}"
        )
      if tangent.shape != (5, 7, 5, 2):
        raise ValueError(
          "direct tangent tactile force must have shape (5, 7, 5, 2), "
          f"got {tangent.shape}"
        )
    normal_force, tangent_force = aggregate_fingertip_forces(normal, tangent)

    self._times.append(timestamp)
    self._states.append(state.copy())
    self._normal.append(normal_force.copy())
    self._tangent.append(tangent_force.copy())

  def finish(self, output_dir: str | Path) -> dict[str, Any]:
    """Write three PNGs and their metadata into an existing directory."""
    if self._finished:
      raise RuntimeError("OpenWAM evaluation plots have already been finished")
    if not self._times:
      raise RuntimeError("cannot finish OpenWAM evaluation plots without captures")
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
      raise FileNotFoundError(f"plot output directory must already exist: {output}")

    rollout_time = np.asarray(self._times, dtype=np.float64)
    rollout_time = rollout_time - rollout_time[0]
    rollout_state = np.stack(self._states)
    rollout_normal = np.stack(self._normal)
    rollout_tangent = np.stack(self._tangent)
    reference = self.reference

    prefix = self.artifact_prefix
    wrist_path = output / f"{prefix}_right_wrist_state.png"
    hand_path = output / f"{prefix}_right_hand_actuated_dof.png"
    tactile_path = output / f"{prefix}_right_fingertip_tactile.png"
    metadata_path = output / f"{prefix}_evaluation_plots.json"

    wrist_figure = Figure(figsize=(13, 10), dpi=120)
    wrist_axes = np.asarray(wrist_figure.subplots(3, 3), dtype=object)
    _draw_series(
      wrist_axes,
      reference.time_s,
      reference.wrist_state,
      rollout_time,
      rollout_state[:, :9],
      WRIST_COMPONENT_NAMES,
      y_label="m / unitless",
      figure_title="Right wrist state: native hand_r_base_link_site (xyz + rot6d)",
      reference_episode_index=reference.output_episode_index,
      output=wrist_path,
    )

    hand_figure = Figure(figsize=(16, 16), dpi=120)
    hand_axes = np.asarray(hand_figure.subplots(5, 4), dtype=object)
    short_joint_names = tuple(
      name.removeprefix("hand_r_") for name in reference.joint_names
    )
    _draw_series(
      hand_axes,
      reference.time_s,
      reference.hand_joint_position,
      rollout_time,
      rollout_state[:, 9:],
      short_joint_names,
      y_label="angle [rad]",
      figure_title="Right hand: 20 actuated DoF joint angles",
      reference_episode_index=reference.output_episode_index,
      output=hand_path,
    )

    tactile_figure = Figure(figsize=(13, 15), dpi=120)
    tactile_axes = np.asarray(tactile_figure.subplots(5, 2), dtype=object)
    tactile_reference = np.empty((len(reference.time_s), 10), dtype=np.float64)
    tactile_rollout = np.empty((len(rollout_time), 10), dtype=np.float64)
    tactile_titles: list[str] = []
    for finger_index, finger in enumerate(FINGER_NAMES):
      column = 2 * finger_index
      tactile_reference[:, column] = reference.fingertip_normal_force_n[:, finger_index]
      tactile_reference[:, column + 1] = reference.fingertip_tangent_force_n[:, finger_index]
      tactile_rollout[:, column] = rollout_normal[:, finger_index]
      tactile_rollout[:, column + 1] = rollout_tangent[:, finger_index]
      tactile_titles.extend((f"{finger} Fn", f"{finger} |Ft|"))
    _draw_series(
      tactile_axes,
      reference.time_s,
      tactile_reference,
      rollout_time,
      tactile_rollout,
      tactile_titles,
      y_label="force [N]",
      figure_title="Right fingertip tactile force (5 canonical fingertips)",
      reference_episode_index=reference.output_episode_index,
      output=tactile_path,
    )

    metadata: dict[str, Any] = {
      "schema": SCHEMA_VERSION,
      "reference": {
        "dataset_root": str(reference.dataset_root),
        "output_episode_index": reference.output_episode_index,
        "source_hdf5": str(reference.source_hdf5),
        "exported_frames_30hz": reference.exported_frames_30hz,
        "samples": len(reference.time_s),
        "source_start_time_s": reference.source_start_time_s,
        "alignment": (
          "first exported_frames_30hz head-camera rows; "
          "cameras/head/state_index selects observation state samples"
        ),
      },
      "rollout": {
        "samples": len(rollout_time),
        "source_start_time_s": self._times[0],
        "duration_s": float(rollout_time[-1]),
      },
      "state": {
        "layout": "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof",
        "dimension": 29,
        "wrist_site": NATIVE_RIGHT_WRIST_SITE,
        "rot6d": "first rotation column followed by second rotation column",
        "right_hand_actuated_dof": 20,
        "joint_names": list(reference.joint_names),
      },
      "tactile": {
        "finger_order": list(FINGER_NAMES),
        "link_names": list(RIGHT_FINGERTIP_LINK_NAMES),
        "taxel_grid": [7, 5],
        "Fn": "sum of all 7x5 normal taxel forces",
        "Ft": (
          "sqrt(sum(Ft_col over 7x5)^2 + sum(Ft_row over 7x5)^2); "
          "signed taxels cancel before magnitude"
        ),
        "unit": "N",
      },
      "styles": {
        "reference": "solid",
        "evaluation_rollout": "dashed",
        "time_axis": "elapsed seconds; each sequence starts at zero",
      },
      "artifacts": {
        "right_wrist_state": str(wrist_path),
        "right_hand_actuated_dof": str(hand_path),
        "right_fingertip_tactile": str(tactile_path),
        "metadata": str(metadata_path),
      },
    }
    metadata_path.write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    self._finished = True
    return metadata


__all__ = [
  "FINGER_NAMES",
  "NATIVE_RIGHT_WRIST_SITE",
  "OpenWAMEvaluationPlots",
  "OpenWAMReference",
  "RIGHT_FINGERTIP_LINK_NAMES",
  "RIGHT_HAND_ACTUATED_JOINT_NAMES",
  "SCHEMA_VERSION",
  "aggregate_fingertip_forces",
  "load_openwam_reference",
]
