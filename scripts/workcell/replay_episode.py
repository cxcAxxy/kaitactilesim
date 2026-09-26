#!/usr/bin/env python3
"""Replay native motion or export synchronized RGB, tactile maps, and curves."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from kaihand_tactile_env.shared.config import (
  SCENE_NAMES,
  default_model_path,
  legacy_model_path,
)
from kaihand_tactile_env.shared.recording import (
  LEGACY_COMBINED_MODEL_LAYOUT,
  TASK_ISOLATED_MODEL_LAYOUT,
  _qpos_names,
  _qvel_names,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("episode", type=Path)
  parser.add_argument("--viewer", choices=("native", "none"), default="native")
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--export-rgb", type=Path)
  parser.add_argument(
    "--camera",
    default="head",
    help="Saved camera used by the legacy --export-rgb PNG export (default: head)",
  )
  parser.add_argument(
    "--review-dir",
    type=Path,
    help="New directory for review.mp4, tactile curves, frame log, and summary",
  )
  parser.add_argument(
    "--cameras",
    nargs="+",
    help=(
      "Saved RGB cameras displayed in the multimodal review; default selects "
      "available head/left_wrist/right_wrist streams"
    ),
  )
  parser.add_argument("--fps", type=float, default=10.0)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
  parser.add_argument(
    "--max-tactile-age-ms",
    type=float,
    default=50.0,
    help="Maximum causal tactile age accepted for a displayed RGB frame",
  )
  args = parser.parse_args()
  if args.speed <= 0.0:
    parser.error("--speed must be positive")
  if args.fps <= 0.0:
    parser.error("--fps must be positive")
  if args.width <= 0 or args.height <= 0:
    parser.error("--width and --height must be positive")
  if not 0.0 <= args.max_tactile_age_ms <= 1000.0:
    parser.error("--max-tactile-age-ms must be between 0 and 1000")

  if args.review_dir is not None:
    from kaihand_tactile_env.shared.offline_replay import export_multimodal_replay

    report = export_multimodal_replay(
      args.episode,
      args.review_dir,
      cameras=None if args.cameras is None else tuple(args.cameras),
      fps=args.fps,
      width=args.width,
      height=args.height,
      maximum_tactile_age_s=args.max_tactile_age_ms / 1000.0,
    )
    print(
      f"[replay] wrote {report['output_frame_count']} multimodal frame(s) "
      f"for {report['task']} to {args.review_dir}",
      flush=True,
    )

  with h5py.File(args.episode, "r") as file:
    simulation, timestamps, qpos, qvel = _load_validated_replay(file)
    if args.export_rgb is not None:
      _export_rgb(file, args.camera, args.export_rgb)
    if args.viewer == "none":
      _restore_terminal_state(simulation, qpos, qvel)
      print(
        f"[replay] validated {len(qpos)} frame(s) with {simulation.model_path}; "
        "restored terminal state (viewer disabled)",
        flush=True,
      )
      return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
      wall_start = time.monotonic()
      for index, timestamp in enumerate(timestamps):
        if not viewer.is_running():
          break
        simulation.restore_full_state(qpos[index], qvel[index])
        viewer.sync()
        target = wall_start + (timestamp - timestamps[0]) / args.speed
        delay = target - time.monotonic()
        if delay > 0.0:
          time.sleep(delay)


def _load_validated_replay(
  file: h5py.File,
) -> tuple[ArmHandSimulation, np.ndarray, np.ndarray, np.ndarray]:
  """Load a trusted model and validate the complete state clock headlessly."""

  timestamps = np.asarray(file["state/timestamp"], dtype=float)
  simulation = _simulation_for_episode(file)
  qpos, qvel = _load_replay_states(file, simulation)
  if timestamps.ndim != 1 or len(timestamps) != len(qpos) or not len(timestamps):
    raise ValueError(
      "state/timestamp must be a non-empty vector aligned with qpos/qvel"
    )
  if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
    raise ValueError("state/timestamp must be finite and strictly increasing")
  return simulation, timestamps, qpos, qvel


def _restore_terminal_state(
  simulation: ArmHandSimulation,
  qpos: np.ndarray,
  qvel: np.ndarray,
) -> None:
  """Exercise MuJoCo state restoration for a viewer-free compatibility check."""

  simulation.restore_full_state(qpos[-1], qvel[-1])


def _simulation_for_episode(file: h5py.File) -> ArmHandSimulation:
  """Construct a replay model exclusively from trusted repository paths."""

  scene, model_path = _replay_model_selection(file)
  if scene == "whiteboard-wipe":
    # Whiteboard ink locations are model geometry, not qpos. Restore the saved
    # randomized layout before applying recorded state so native replay shows
    # the same task instance as the RGB/cleaning streams in the episode.
    from kaihand_tactile_env.tasks.whiteboard_wipe.task import (
      WhiteboardWipeSimulation,
    )

    simulation = WhiteboardWipeSimulation()
    metadata = _episode_metadata(file)
    layout = metadata.get("ink_randomization")
    if isinstance(layout, dict):
      positions = layout.get("geom_positions_board_m")
      if positions is not None:
        simulation.set_ink_layout(positions, seed=layout.get("seed"))
    return simulation
  return ArmHandSimulation(model_path=model_path, scene=scene)


def _replay_model_selection(file: h5py.File) -> tuple[str, Path]:
  """Select an isolated task model or the frozen combined compatibility model.

  The recorded ``model_path`` attribute is intentionally never opened: an HDF5
  file may come from another machine and is not authority to load arbitrary
  local XML.  New recordings identify the isolated layout explicitly.  Any
  recording without that marker predates the split and therefore prefers the
  frozen combined model, including the still older cylinder-only coordinate
  layout handled by :func:`_load_replay_states`.
  """

  metadata = _episode_metadata(file)
  scene = metadata.get("scene", "pick-place")
  if not isinstance(scene, str) or scene not in SCENE_NAMES:
    raise ValueError(f"episode has unsupported scene {scene!r}")
  layout = metadata.get("model_layout")
  if layout == TASK_ISOLATED_MODEL_LAYOUT:
    return scene, default_model_path(scene)
  # Early Vase/Sponge recordings were mislabeled by a duplicate global object
  # name. The frozen combined model never contained a sponge, so their recorded
  # freejoint coordinates can only be replayed with the isolated task model.
  if (
    scene in {"vase-wipe", "sponge-grasp"}
    and layout == LEGACY_COMBINED_MODEL_LAYOUT
    and metadata.get("active_objects") == ["sponge"]
  ):
    return scene, default_model_path(scene)
  if layout in (None, LEGACY_COMBINED_MODEL_LAYOUT, "combined-v1", "legacy"):
    return scene, legacy_model_path()
  raise ValueError(f"episode has unsupported model_layout {layout!r}")


def _episode_metadata(file: h5py.File) -> dict[str, object]:
  raw = file.attrs.get("metadata_json", "{}")
  try:
    if isinstance(raw, bytes):
      raw = raw.decode("utf-8")
    metadata = json.loads(str(raw))
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
    return {}
  return metadata if isinstance(metadata, dict) else {}


def _load_replay_states(
  file: h5py.File, simulation: ArmHandSimulation
) -> tuple[np.ndarray, np.ndarray]:
  """Load full state, mapping legacy coordinate names into the current model."""
  qpos = np.asarray(file["state/qpos"])
  qvel = np.asarray(file["state/qvel"])
  if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape[0] != qvel.shape[0]:
    raise ValueError("recorded qpos/qvel must be aligned two-dimensional arrays")

  current_qpos_names = tuple(_qpos_names(simulation.model))
  current_qvel_names = tuple(_qvel_names(simulation.model))
  recorded_qpos_names = _read_coordinate_names(file, "full_qpos_names", qpos.shape[1])
  recorded_qvel_names = _read_coordinate_names(file, "full_qvel_names", qvel.shape[1])
  exact_dimensions = (
    qpos.shape[1] == simulation.model.nq and qvel.shape[1] == simulation.model.nv
  )
  exact_names = recorded_qpos_names in (
    None,
    current_qpos_names,
  ) and recorded_qvel_names in (None, current_qvel_names)
  if exact_dimensions and exact_names:
    return qpos, qvel

  if recorded_qpos_names is None or recorded_qvel_names is None:
    raise ValueError(
      "recorded state dimensions do not match the current model and the episode "
      "has no state/full_qpos_names and state/full_qvel_names compatibility metadata"
    )

  mapped_qpos, qpos_current_only = _map_coordinates(
    qpos,
    recorded_qpos_names,
    current_qpos_names,
    simulation.data.qpos,
    label="qpos",
  )
  mapped_qvel, qvel_current_only = _map_coordinates(
    qvel,
    recorded_qvel_names,
    current_qvel_names,
    simulation.data.qvel,
    label="qvel",
  )
  current_only_joints = sorted(
    {
      name.split("/", maxsplit=1)[0]
      for name in (*qpos_current_only, *qvel_current_only)
    }
  )
  suffix = (
    f"; current-only joints kept at scene defaults: {', '.join(current_only_joints)}"
    if current_only_joints
    else ""
  )
  print(
    "[replay] mapped recorded state "
    f"qpos {qpos.shape[1]}->{simulation.model.nq}, "
    f"qvel {qvel.shape[1]}->{simulation.model.nv}{suffix}",
    flush=True,
  )
  return mapped_qpos, mapped_qvel


def _read_coordinate_names(
  file: h5py.File, dataset_name: str, expected_count: int
) -> tuple[str, ...] | None:
  path = f"state/{dataset_name}"
  if path not in file:
    return None
  names = tuple(
    value.decode("utf-8") if isinstance(value, bytes) else str(value)
    for value in file[path]
  )
  if len(names) != expected_count or not all(names) or len(set(names)) != len(names):
    raise ValueError(
      f"{path} must contain {expected_count} unique, non-empty coordinate names"
    )
  return names


def _map_coordinates(
  recorded: np.ndarray,
  recorded_names: tuple[str, ...],
  current_names: tuple[str, ...],
  current_default: np.ndarray,
  *,
  label: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
  current_index = {name: index for index, name in enumerate(current_names)}
  unknown = tuple(name for name in recorded_names if name not in current_index)
  if unknown:
    preview = ", ".join(unknown[:5])
    if len(unknown) > 5:
      preview += f", ... ({len(unknown)} total)"
    raise ValueError(
      f"recorded {label} contains coordinates absent from the current model: {preview}"
    )

  mapped = np.broadcast_to(
    current_default, (recorded.shape[0], len(current_names))
  ).copy()
  destination = np.array([current_index[name] for name in recorded_names])
  mapped[:, destination] = recorded
  recorded_name_set = set(recorded_names)
  current_only = tuple(name for name in current_names if name not in recorded_name_set)
  return mapped, current_only


def _export_rgb(file: h5py.File, camera: str, output_dir: Path) -> None:
  from PIL import Image

  key = f"cameras/{camera}/rgb"
  if key not in file:
    raise RuntimeError(f"episode has no RGB stream {key}")
  output_dir.mkdir(parents=True, exist_ok=True)
  for index, image in enumerate(file[key]):
    Image.fromarray(np.asarray(image)).save(output_dir / f"{camera}_{index:06d}.png")


if __name__ == "__main__":
  main()
