"""Compare available legacy poker observations with reference episode 00.

The tactile comparison uses only recorded 10 Hz video frames.  Hand joints
and wrist FK use sparse measured policy-request states.  None of these plots
is a substitute for the new 30 Hz control-step comparison protocol.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import (
  FINGERTIP_LINK_NAMES,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  FINGER_NAMES,
  NATIVE_RIGHT_WRIST_SITE,
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  WRIST_COMPONENT_NAMES,
  aggregate_fingertip_forces,
  load_openwam_reference,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SCHEMA_VERSION = "kaihand-legacy-poker-sparse-comparison-v1"
PLOT_NAME = "legacy_right_fingertip_tactile_10hz.png"
WRIST_PLOT_NAME = "legacy_right_wrist_state_request_samples.png"
HAND_PLOT_NAME = "legacy_right_hand_actuated_dof_request_samples.png"
METADATA_NAME = "legacy_comparison.json"
RIGHT_ARM_JOINT_NAMES = tuple(f"right_arm_joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class LegacyTactileFrames:
  time_s: np.ndarray
  control_tick: np.ndarray
  normal_force_n: np.ndarray
  tangent_force_n: np.ndarray
  frame_path: Path
  review_path: Path
  review: dict[str, Any]


@dataclass(frozen=True)
class LegacyRequestStates:
  time_s: np.ndarray
  measured_state_27: np.ndarray
  request_path: Path
  summary_path: Path
  execute_steps: int
  control_hz: int
  joint_names: tuple[str, ...]


def load_legacy_tactile_frames(seed_dir: str | Path) -> LegacyTactileFrames:
  """Read only the recorded video frames, never synthesize control-step data."""
  seed_path = Path(seed_dir).expanduser().resolve()
  frame_path = seed_path / "review/frames.jsonl"
  review_path = seed_path / "review/review.json"
  review = json.loads(review_path.read_text(encoding="utf-8"))
  fps = int(review["fps"])
  control_hz = int(review["control_hz"])
  if fps != 10 or control_hz != 30:
    raise ValueError(
      f"this converter requires a 10 fps, 30 Hz legacy review, got {fps} fps and {control_hz} Hz"
    )
  if review.get("task") != "poker-draw":
    raise ValueError(f"expected poker-draw review, got {review.get('task')!r}")

  expected_order = tuple(FINGERTIP_LINK_NAMES)
  if len(expected_order) != 10 or tuple(expected_order[5:]) != RIGHT_FINGERTIP_LINK_NAMES:
    raise ValueError("the repository's canonical fingertip order has changed")
  times: list[float] = []
  ticks: list[int] = []
  normal: list[np.ndarray] = []
  tangent: list[np.ndarray] = []
  with frame_path.open(encoding="utf-8") as stream:
    for index, line in enumerate(stream):
      frame = json.loads(line)
      if int(frame["frame"]) != index:
        raise ValueError(f"legacy frame index is discontinuous at line {index + 1}")
      if tuple(frame["fingertip_order"]) != expected_order:
        raise ValueError(f"legacy fingertip order differs from canonical order at frame {index}")
      timestamp = float(frame["simulation_time_s"])
      tick = int(frame["control_tick"])
      if not np.isfinite(timestamp) or tick < 0:
        raise ValueError(f"legacy frame {index} has an invalid timestamp or control tick")
      all_normal = np.asarray(frame["normal_taxel_force_n"], dtype=np.float64)
      all_tangent = np.asarray(frame["tangent_taxel_force_n"], dtype=np.float64)
      if all_normal.shape != (10, 7, 5) or all_tangent.shape != (10, 7, 5, 2):
        raise ValueError(f"legacy frame {index} does not contain ten 7x5 fingertip grids")
      normal_sum, tangent_resultant = aggregate_fingertip_forces(
        all_normal[5:], all_tangent[5:]
      )
      times.append(timestamp)
      ticks.append(tick)
      normal.append(normal_sum)
      tangent.append(tangent_resultant)

  if len(times) < 2:
    raise ValueError("legacy review must contain at least two recorded tactile frames")
  if len(times) != int(review["frame_count"]):
    raise ValueError("legacy frame count differs from review.json")
  time_array = np.asarray(times, dtype=np.float64)
  tick_array = np.asarray(ticks, dtype=np.int64)
  if np.any(np.diff(time_array) <= 0) or np.any(np.diff(tick_array) != 3):
    raise ValueError("legacy frames are not consecutive 10 Hz samples")
  expected_elapsed = (tick_array - tick_array[0]) / control_hz
  if not np.allclose(time_array - time_array[0], expected_elapsed, rtol=0.0, atol=0.002):
    raise ValueError("legacy frame timestamps do not match their 30 Hz control ticks")

  return LegacyTactileFrames(
    time_s=time_array - time_array[0],
    control_tick=tick_array,
    normal_force_n=np.stack(normal),
    tangent_force_n=np.stack(tangent),
    frame_path=frame_path,
    review_path=review_path,
    review=review,
  )


def load_legacy_request_states(seed_dir: str | Path) -> LegacyRequestStates:
  """Load actual measured state at policy requests, not predicted actions."""
  seed_path = Path(seed_dir).expanduser().resolve()
  request_path = seed_path / "requests.jsonl"
  summary_path = seed_path / "summary.json"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  if summary.get("task") != "poker-draw":
    raise ValueError(f"expected poker-draw summary, got {summary.get('task')!r}")
  joint_names = tuple(summary["server_metadata"]["joint_names"])
  expected_names = (*RIGHT_ARM_JOINT_NAMES, *RIGHT_HAND_ACTUATED_JOINT_NAMES)
  if joint_names != expected_names:
    raise ValueError("policy request state is not in the expected 7-arm + 20-hand order")
  execute_steps = int(summary["execute_steps"])
  control_hz = int(summary["control_hz"])
  if execute_steps < 1 or control_hz < 1:
    raise ValueError("summary has an invalid execute_steps or control_hz")

  times: list[float] = []
  states: list[np.ndarray] = []
  with request_path.open(encoding="utf-8") as stream:
    for index, line in enumerate(stream):
      request = json.loads(line)
      if int(request["request"]) != index + 1:
        raise ValueError(f"policy request index is discontinuous at line {index + 1}")
      timestamp = float(request["sim_time"])
      state = np.asarray(request["state"], dtype=np.float64)
      if not np.isfinite(timestamp) or state.shape != (27,) or not np.isfinite(state).all():
        raise ValueError(f"policy request {index + 1} has an invalid state or timestamp")
      times.append(timestamp)
      states.append(state)
  if len(times) < 2:
    raise ValueError("at least two measured policy-request states are required")
  time_array = np.asarray(times, dtype=np.float64)
  intervals = np.diff(time_array)
  expected_interval = execute_steps / control_hz
  if np.any(intervals <= 0) or not np.allclose(
    intervals, expected_interval, rtol=0.0, atol=0.005
  ):
    raise ValueError("policy-request timestamps do not match the execution interval")
  if len(times) != int(summary["stats"]["requests"]):
    raise ValueError("policy-request count differs from summary.json")
  return LegacyRequestStates(
    time_s=time_array - time_array[0],
    measured_state_27=np.stack(states),
    request_path=request_path,
    summary_path=summary_path,
    execute_steps=execute_steps,
    control_hz=control_hz,
    joint_names=joint_names,
  )


def wrist_fk_from_measured_states(
  requests: LegacyRequestStates,
  reference_initial_wrist: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
  """Compute site FK only if the current scene reproduces the known home pose.

  The sparse arm/hand joint values are measurements in ``requests.jsonl``.
  The wrist site coordinates are derived by FK, not a saved wrist observation.
  """
  scene_path = default_model_path("poker-draw")
  details: dict[str, Any] = {
    "method": "MuJoCo mj_forward from each measured 27D request state",
    "scene_xml": str(scene_path),
    "wrist_site": NATIVE_RIGHT_WRIST_SITE,
    "status": "unavailable",
  }
  try:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    details["scene_fingerprint"] = model_fingerprint(scene_path)
    site_id = int(model.site(NATIVE_RIGHT_WRIST_SITE).id)
    addresses: list[int] = []
    for name in requests.joint_names:
      joint = model.joint(name)
      if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"FK joint {name} is not a hinge")
      addresses.append(int(model.jnt_qposadr[joint.id]))
    data = mujoco.MjData(model)
    wrist = np.empty((len(requests.time_s), 9), dtype=np.float64)
    for index, state in enumerate(requests.measured_state_27):
      data.qpos[:] = model.qpos0
      data.qpos[addresses] = state
      mujoco.mj_forward(model, data)
      rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
      wrist[index] = np.concatenate((data.site_xpos[site_id], rotation[:, 0], rotation[:, 1]))
    if not np.isfinite(wrist).all():
      raise ValueError("computed wrist FK contains nonfinite values")
    initial_reference = np.asarray(reference_initial_wrist, dtype=np.float64)
    if initial_reference.shape != (9,) or not np.isfinite(initial_reference).all():
      raise ValueError("reference initial wrist is invalid")
    initial_error = float(np.linalg.norm(wrist[0] - initial_reference))
    details["initial_reference_rot6d_error_norm"] = initial_error
    details["verification"] = "first request FK compared with episode 00 first wrist pose"
    if initial_error > 1.0e-3:
      details["reason"] = "current scene FK does not reproduce the reference home wrist pose"
      return None, details
    details["status"] = "verified_against_reference_home_pose"
    return wrist, details
  except Exception as error:
    details["reason"] = f"{type(error).__name__}: {error}"
    return None, details


def _plot_tactile(
  reference_time: np.ndarray,
  reference_normal: np.ndarray,
  reference_tangent: np.ndarray,
  rollout: LegacyTactileFrames,
  output: Path,
) -> None:
  figure = Figure(figsize=(13, 15), dpi=120)
  axes = np.asarray(figure.subplots(5, 2), dtype=object)
  reference_line = None
  rollout_line = None
  for finger_index, finger in enumerate(FINGER_NAMES):
    for column, (label, ref_values, actual_values) in enumerate((
      ("Fn", reference_normal[:, finger_index], rollout.normal_force_n[:, finger_index]),
      ("|Ft|", reference_tangent[:, finger_index], rollout.tangent_force_n[:, finger_index]),
    )):
      axis = axes[finger_index, column]
      (reference_line,) = axis.plot(
        reference_time, ref_values, color="#1f77b4", linestyle="-", linewidth=1.4
      )
      (rollout_line,) = axis.plot(
        rollout.time_s,
        actual_values,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        marker=".",
        markersize=2,
      )
      axis.set_title(f"{finger} {label}", fontsize=9)
      axis.grid(alpha=0.25, linewidth=0.5)
      if column == 0:
        axis.set_ylabel("force [N]", fontsize=8)
      if finger_index == len(FINGER_NAMES) - 1:
        axis.set_xlabel("elapsed time [s]", fontsize=8)
      axis.tick_params(labelsize=7)
  figure.suptitle(
    "Right fingertip tactile: episode 00 reference 30 Hz vs legacy review 10 Hz",
    fontsize=13,
    y=0.99,
  )
  figure.legend(
    (reference_line, rollout_line),
    ("Reference ep 00 (30 Hz, solid)", "Legacy rollout (10 Hz samples, dashed)"),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncols=2,
    frameon=False,
  )
  figure.text(
    0.5,
    0.934,
    "No temporal resampling or invented samples; dashed segments join recorded frames.",
    ha="center",
    fontsize=9,
  )
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.915), pad=1.2)
  FigureCanvasAgg(figure).print_png(str(output))
  figure.clear()


def _plot_sparse_series(
  reference_time: np.ndarray,
  reference_values: np.ndarray,
  request_time: np.ndarray,
  request_values: np.ndarray,
  names: tuple[str, ...],
  *,
  columns: int,
  request_hz: float,
  title: str,
  ylabel: str,
  output: Path,
) -> None:
  rows = len(names) // columns
  figure = Figure(figsize=(14, 3.0 * rows), dpi=120)
  axes = np.asarray(figure.subplots(rows, columns), dtype=object)
  reference_line = None
  rollout_line = None
  for index, (axis, name) in enumerate(zip(axes.flat, names, strict=True)):
    (reference_line,) = axis.plot(
      reference_time,
      reference_values[:, index],
      color="#1f77b4",
      linestyle="-",
      linewidth=1.4,
    )
    (rollout_line,) = axis.plot(
      request_time,
      request_values[:, index],
      color="#d62728",
      linestyle="--",
      linewidth=1.2,
      marker="o",
      markersize=2.5,
    )
    axis.set_title(name, fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)
    if index % columns == 0:
      axis.set_ylabel(ylabel, fontsize=8)
    if index // columns == rows - 1:
      axis.set_xlabel("elapsed time [s]", fontsize=8)
    axis.tick_params(labelsize=7)
  figure.suptitle(
    f"{title}: reference 30 Hz vs legacy request snapshots (~{request_hz:.3f} Hz)",
    fontsize=13,
    y=0.99,
  )
  figure.legend(
    (reference_line, rollout_line),
    ("Reference ep 00 (solid)", "Legacy request samples (dashed with markers)"),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncols=2,
    frameon=False,
  )
  figure.text(
    0.5,
    0.905,
    "Not 30 Hz rollout data. No interpolated values; dashed segments join sparse observations.",
    ha="center",
    fontsize=9,
  )
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.87), pad=1.2)
  FigureCanvasAgg(figure).print_png(str(output))
  figure.clear()


def convert_legacy_poker_comparison(
  seed_dir: str | Path,
  reference_dataset: str | Path,
  output_dir: str | Path | None = None,
) -> dict[str, Any]:
  """Write honestly labeled old-review comparisons into a new subdirectory."""
  seed_path = Path(seed_dir).expanduser().resolve()
  output = (
    Path(output_dir).expanduser().resolve()
    if output_dir is not None else seed_path / "review/legacy_sparse_comparison"
  )
  plot_path = output / PLOT_NAME
  wrist_plot_path = output / WRIST_PLOT_NAME
  hand_plot_path = output / HAND_PLOT_NAME
  metadata_path = output / METADATA_NAME
  if any(path.exists() for path in (plot_path, wrist_plot_path, hand_plot_path, metadata_path)):
    raise FileExistsError(f"legacy comparison artifacts already exist in {output}")

  rollout = load_legacy_tactile_frames(seed_path)
  requests = load_legacy_request_states(seed_path)
  reference = load_openwam_reference(reference_dataset, output_episode_index=0)
  if reference.time_s.ndim != 1 or len(reference.time_s) < 2:
    raise ValueError("episode 00 reference has no usable time series")
  if np.any(np.diff(reference.time_s) <= 0):
    raise ValueError("episode 00 reference timestamps are not strictly increasing")
  wrist, wrist_fk = wrist_fk_from_measured_states(requests, reference.wrist_state[0])
  request_hz = float(1.0 / np.median(np.diff(requests.time_s)))
  output.mkdir(parents=True, exist_ok=True)
  _plot_tactile(
    reference.time_s,
    reference.fingertip_normal_force_n,
    reference.fingertip_tangent_force_n,
    rollout,
    plot_path,
  )
  hand_short_names = tuple(
    name.removeprefix("hand_r_").replace("_joint", " j")
    for name in RIGHT_HAND_ACTUATED_JOINT_NAMES
  )
  _plot_sparse_series(
    reference.time_s,
    reference.hand_joint_position,
    requests.time_s,
    requests.measured_state_27[:, 7:],
    hand_short_names,
    columns=4,
    request_hz=request_hz,
    title="Right-hand 20 measured joint angles",
    ylabel="angle [rad]",
    output=hand_plot_path,
  )
  if wrist is not None:
    _plot_sparse_series(
      reference.time_s,
      reference.wrist_state,
      requests.time_s,
      wrist,
      WRIST_COMPONENT_NAMES,
      columns=3,
      request_hz=request_hz,
      title="Right wrist xyz + rot6d (request-state FK)",
      ylabel="m or unitless",
      output=wrist_plot_path,
    )
  metadata: dict[str, Any] = {
    "schema": SCHEMA_VERSION,
    "status": "ok_legacy_sparse_comparison",
    "reference": {
      "dataset_root": str(reference.dataset_root),
      "output_episode_index": 0,
      "source_hdf5": str(reference.source_hdf5),
      "samples": len(reference.time_s),
      "sampling_hz": 30,
      "duration_s": float(reference.time_s[-1] - reference.time_s[0]),
    },
    "rollout": {
      "seed_dir": str(seed_path),
      "frames_jsonl": str(rollout.frame_path),
      "review_json": str(rollout.review_path),
      "samples": len(rollout.time_s),
      "sampling_hz": 10,
      "control_hz": 30,
      "control_tick_stride": 3,
      "duration_s": float(rollout.time_s[-1]),
      "tactile_source": rollout.review.get("tactile_source"),
      "note": "Only actual video-frame tactile measurements were saved by this legacy run.",
    },
    "request_states": {
      "requests_jsonl": str(requests.request_path),
      "summary_json": str(requests.summary_path),
      "samples": len(requests.time_s),
      "measured_joint_dimension": 27,
      "arm_joint_names": list(RIGHT_ARM_JOINT_NAMES),
      "hand_joint_names": list(RIGHT_HAND_ACTUATED_JOINT_NAMES),
      "execute_steps": requests.execute_steps,
      "control_hz": requests.control_hz,
      "nominal_sample_interval_s": requests.execute_steps / requests.control_hz,
      "observed_median_sample_interval_s": float(np.median(np.diff(requests.time_s))),
      "observed_sampling_hz": request_hz,
      "duration_s": float(requests.time_s[-1]),
      "sampling_note": "Actual measured joints were logged only when the model was asked for a new action chunk, not every control step.",
    },
    "wrist_fk": wrist_fk,
    "tactile": {
      "fingers": list(FINGER_NAMES),
      "right_fingertip_order": list(RIGHT_FINGERTIP_LINK_NAMES),
      "taxel_grid": [7, 5],
      "Fn": "sum of 35 normal taxel forces per fingertip",
      "Ft": "sqrt(sum(Ft_col over 35 taxels)^2 + sum(Ft_row over 35 taxels)^2)",
      "unit": "N",
      "calibration_note": "Reference and legacy rollout tactile providers may differ.",
    },
    "styles": {
      "reference": "solid, 30 Hz",
      "legacy_tactile": "dashed with sample markers, 10 Hz",
      "legacy_request_states": f"dashed with sample markers, approximately {request_hz:.3f} Hz",
      "time_axis": "elapsed seconds; each sequence starts at zero",
      "interpolation": "none; plot line segments only join independently recorded samples",
    },
    "limitations": [
      "These plots do not satisfy the new 30 Hz control-step comparison protocol.",
      "Legacy tactile was recorded only at 10 Hz video frames.",
      "Legacy measured joints were recorded only at sparse policy requests; wrist values are FK-derived from those measured joints.",
      "Line segments between samples are visual connectors, not measured or interpolated intermediate states.",
    ],
    "artifacts": {
      "right_fingertip_tactile": str(plot_path),
      "right_hand_actuated_dof": str(hand_plot_path),
      "metadata": str(metadata_path),
    },
  }
  if wrist is not None:
    metadata["artifacts"]["right_wrist_state"] = str(wrist_plot_path)
  metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  return metadata


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed-dir", type=Path, required=True)
  parser.add_argument("--reference-dataset", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path)
  args = parser.parse_args()
  result = convert_legacy_poker_comparison(args.seed_dir, args.reference_dataset, args.output_dir)
  print(json.dumps(result["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
