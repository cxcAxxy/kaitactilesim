"""Reconstruct PickPlace episode-00 reference from its recorded solver contacts.

The 0920 PickPlace raw episode predates the ``tactile_contact_force`` and
``world_from_wrist`` streams used by the generic reference loader.  It does,
however, retain synchronized full qpos and exact per-contact solver wrenches.
The current scene's matching kinematics recover the wrist site, while the
recorded pad/cylinder contacts recover force-conserving 7x5-grid totals:
``Fn = sum(abs(contact normal))`` and ``|Ft|`` is the norm of the signed net
force in the stable pad tangent basis.  No Genesis depth-force proxy is used.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  NATIVE_RIGHT_WRIST_SITE,
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  OpenWAMReference,
  _columns,
  _decode_names,
  _read_manifest_record,
  _require_dataset,
)
from kaihand_tactile_env.shared.recording import _qpos_names

REFERENCE_TACTILE_SOURCE = "recorded_solver_contact_wrench_conservative_taxel_sum_v1"


def _recorded_geom_ids(file: h5py.File, names: tuple[str, ...]) -> dict[str, int]:
  raw = np.asarray(_require_dataset(file, "model/geom_names")[:])
  if raw.ndim != 1:
    raise ValueError("model/geom_names must be one-dimensional")
  decoded = tuple(
    name.decode("utf-8") if isinstance(name, bytes) else str(name)
    for name in raw
  )
  result: dict[str, int] = {}
  for name in names:
    matches = [index for index, candidate in enumerate(decoded) if candidate == name]
    if len(matches) != 1:
      raise ValueError(f"recorded model must contain exactly one geom {name!r}")
    result[name] = matches[0]
  return result


def _event_arrays(file: h5py.File, state_count: int) -> dict[str, np.ndarray]:
  required = {
    "start": ("contacts/frame_start", (state_count,)),
    "count": ("contacts/frame_count", (state_count,)),
    "geom1": ("contacts/events/geom1_id", None),
    "geom2": ("contacts/events/geom2_id", None),
    "frame": ("contacts/events/frame_world", None),
    "wrench": ("contacts/events/wrench_contact_on_geom2", None),
  }
  arrays = {
    key: np.asarray(_require_dataset(file, name)[:])
    for key, (name, _) in required.items()
  }
  event_count = len(arrays["geom1"])
  expected = {
    "start": (state_count,),
    "count": (state_count,),
    "geom1": (event_count,),
    "geom2": (event_count,),
    "frame": (event_count, 3, 3),
    "wrench": (event_count, 6),
  }
  for key, shape in expected.items():
    if arrays[key].shape != shape:
      raise ValueError(f"recorded contacts {key} must have shape {shape}")
  for key in ("start", "count", "geom1", "geom2"):
    if arrays[key].dtype.kind not in "iu":
      raise ValueError(f"recorded contacts {key} must contain integers")
  if np.any(arrays["start"] < 0) or np.any(arrays["count"] < 0):
    raise ValueError("recorded contact frame ranges must be nonnegative")
  if np.any(arrays["start"] + arrays["count"] > event_count):
    raise ValueError("recorded contact frame ranges exceed event arrays")
  if not np.isfinite(arrays["frame"]).all() or not np.isfinite(arrays["wrench"]).all():
    raise ValueError("recorded contact frames and wrenches must be finite")
  return arrays


def _contact_force_totals(
  arrays: dict[str, np.ndarray],
  state_index: int,
  pad_to_finger: dict[int, int],
  target_ids: frozenset[int],
  tangent_basis_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return five Fn and |Ft| values with exact signed vector cancellation."""
  normal = np.zeros(5, dtype=np.float64)
  tangent = np.zeros((5, 2), dtype=np.float64)
  first = int(arrays["start"][state_index])
  last = first + int(arrays["count"][state_index])
  for event in range(first, last):
    geom1 = int(arrays["geom1"][event])
    geom2 = int(arrays["geom2"][event])
    if geom1 in pad_to_finger and geom2 in target_ids:
      finger, sign = pad_to_finger[geom1], -1.0
    elif geom2 in pad_to_finger and geom1 in target_ids:
      finger, sign = pad_to_finger[geom2], 1.0
    else:
      continue
    wrench = arrays["wrench"][event]
    # MuJoCo contact-frame wrench acts on geom2.  The rows of frame_world
    # are its world-space axes, so frame.T converts contact components to world.
    tangent_world = sign * (
      arrays["frame"][event].T @ np.asarray((0.0, wrench[1], wrench[2]))
    )
    normal[finger] += abs(float(wrench[0]))
    tangent[finger] += tangent_basis_world[finger] @ tangent_world
  return normal, np.linalg.norm(tangent, axis=1)


def load_pickplace_reference(
  dataset_root: str | Path,
  output_episode_index: int = 0,
) -> OpenWAMReference:
  """Load camera-aligned wrist, hand, and real solver-force totals for PickPlace."""
  root = Path(dataset_root).expanduser().resolve(strict=True)
  if not root.is_dir():
    raise ValueError(f"reference dataset is not a directory: {root}")
  source, frame_count = _read_manifest_record(root, output_episode_index)
  model = mujoco.MjModel.from_xml_path(str(default_model_path("pick-place")))
  tactile = SolverDistributedTactileProvider(model)
  if tuple(tactile.link_names[5:]) != RIGHT_FINGERTIP_LINK_NAMES:
    raise ValueError("current PickPlace tactile link order differs from reference")
  site_id = int(model.site(NATIVE_RIGHT_WRIST_SITE).id)
  body_ids = tactile._body_ids[5:]
  tangent_basis_local = tactile.tangent_basis_local[5:]
  data = mujoco.MjData(model)

  with h5py.File(source, "r") as file:
    state_times = np.asarray(_require_dataset(file, "state/timestamp")[:], dtype=np.float64)
    if state_times.ndim != 1 or len(state_times) < 2:
      raise ValueError("reference state timestamps must be a nonempty vector")
    if not np.isfinite(state_times).all() or np.any(np.diff(state_times) <= 0):
      raise ValueError("reference state timestamps must be finite and increasing")
    camera_times = np.asarray(
      _require_dataset(file, "cameras/head/timestamp")[: frame_count + 1],
      dtype=np.float64,
    )
    state_indices = np.asarray(
      _require_dataset(file, "cameras/head/state_index")[: frame_count + 1],
      dtype=np.int64,
    )
    if camera_times.shape != (frame_count + 1,) or state_indices.shape != camera_times.shape:
      raise ValueError("head camera stream must contain exported frames plus next row")
    if not np.isfinite(camera_times).all() or np.any(np.diff(camera_times) <= 0):
      raise ValueError("head camera timestamps must be finite and increasing")
    if np.any(state_indices < 0) or np.any(np.diff(state_indices) <= 0):
      raise ValueError("head camera state indices must be nonnegative and increasing")
    if state_indices[-1] >= len(state_times):
      raise ValueError("head camera state index exceeds recorded states")
    expected_indices = np.searchsorted(state_times, camera_times, side="right") - 1
    if not np.array_equal(state_indices, expected_indices):
      raise ValueError("head camera state indices are not latest nonfuture samples")

    raw_qpos_names = _decode_names(
      _require_dataset(file, "state/full_qpos_names"),
      context="state/full_qpos_names",
    )
    if raw_qpos_names != tuple(_qpos_names(model)):
      raise ValueError("current PickPlace model qpos order differs from recorded model")
    qpos = _require_dataset(file, "state/qpos")
    if qpos.shape != (len(state_times), model.nq):
      raise ValueError("reference qpos shape differs from the current PickPlace model")
    available_joints = _decode_names(
      _require_dataset(file, "state/joint_names"), context="state/joint_names"
    )
    hand_columns = _columns(
      available_joints,
      RIGHT_HAND_ACTUATED_JOINT_NAMES,
      context="state/joint_names",
    )
    joints = _require_dataset(file, "state/robot_joint_position")
    if joints.shape != (len(state_times), len(available_joints)):
      raise ValueError("reference robot_joint_position has an invalid shape")

    requested_geoms = (*tactile.pad_geom_names[5:], *tactile.target_geom_names)
    raw_geom_ids = _recorded_geom_ids(file, requested_geoms)
    pad_to_finger = {
      raw_geom_ids[name]: index
      for index, name in enumerate(tactile.pad_geom_names[5:])
    }
    target_ids = frozenset(raw_geom_ids[name] for name in tactile.target_geom_names)
    events = _event_arrays(file, len(state_times))

    wrist_state = np.empty((frame_count, 9), dtype=np.float64)
    hand_position = np.empty((frame_count, 20), dtype=np.float64)
    normal_force = np.empty((frame_count, 5), dtype=np.float64)
    tangent_force = np.empty((frame_count, 5), dtype=np.float64)
    for frame_index, state_index in enumerate(state_indices[:-1]):
      state_index = int(state_index)
      data.qpos[:] = np.asarray(qpos[state_index], dtype=np.float64)
      mujoco.mj_forward(model, data)
      rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
      wrist_state[frame_index] = np.concatenate(
        (data.site_xpos[site_id], rotation[:, 0], rotation[:, 1])
      )
      hand_position[frame_index] = np.asarray(
        joints[state_index, hand_columns], dtype=np.float64
      )
      body_rotations = np.stack(
        [np.asarray(data.xmat[body_id]).reshape(3, 3) for body_id in body_ids]
      )
      tangent_basis_world = np.einsum(
        "fij,fkj->fki", body_rotations, tangent_basis_local
      )
      normal_force[frame_index], tangent_force[frame_index] = _contact_force_totals(
        events, state_index, pad_to_finger, target_ids, tangent_basis_world
      )

  if not all(
    np.isfinite(values).all()
    for values in (wrist_state, hand_position, normal_force, tangent_force)
  ):
    raise ValueError("reconstructed PickPlace reference contains nonfinite values")
  return OpenWAMReference(
    dataset_root=root,
    output_episode_index=output_episode_index,
    source_hdf5=source,
    exported_frames_30hz=frame_count,
    source_start_time_s=float(camera_times[0]),
    time_s=camera_times[:-1] - camera_times[0],
    wrist_state=wrist_state,
    hand_joint_position=hand_position,
    fingertip_normal_force_n=normal_force,
    fingertip_tangent_force_n=tangent_force,
    joint_names=RIGHT_HAND_ACTUATED_JOINT_NAMES,
  )


__all__ = ["REFERENCE_TACTILE_SOURCE", "load_pickplace_reference"]
