#!/usr/bin/env python3
"""Publish a compact USB-style poker example from one validated raw episode.

Head/wrist RGB and tactile are original observations. The separate global
video is a kinematic visualization of recorded qpos, never another rollout or
a replacement for training images. Only the two MP4 files and one PNG curve
are derived; the Raw HDF5/JSON stay untouched. Existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _key in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
):
  os.environ[_key] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", "2")

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.poker_review import (  # noqa: E402
  _force_layout,
  _probe_video,
  _sha256,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from kaihand_tactile_env.shared.posture import ARM_HOME  # noqa: E402
from kaihand_tactile_env.shared.recording import (  # noqa: E402
  _qpos_names,
  validate_episode,
)
from kaihand_tactile_env.shared.render_backend import (  # noqa: E402
  current_backend,
  prepare_render_backend,
)
from kaihand_tactile_env.shared.tactile import RIGHT_FINGERTIP_LINK_NAMES  # noqa: E402
from kaihand_tactile_env.shared.task_video import _FfmpegPipeWriter  # noqa: E402


def save_json(path, value):
  with path.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")


def right_force_values(file):
  group = file["tactile_contact_force"]
  links = tuple(group["link_names"].asstr()[:])
  indices = [links.index(name) for name in RIGHT_FINGERTIP_LINK_NAMES]
  # h5py fancy indexing requires increasing indices. Map by name after reading
  # so a valid source with reordered fingertips remains supported.
  normal = group["normal_taxel_force_n"][:][:, indices].sum(axis=(-2, -1))
  tangent_xy = group["tangent_taxel_force_n"][:][:, indices].sum(axis=(-3, -2))
  np.testing.assert_allclose(normal, group["normal_force_n"][:][:, indices], atol=1e-10)
  np.testing.assert_allclose(
    tangent_xy, group["tangent_force_n"][:][:, indices], atol=1e-10
  )
  if not np.isfinite(normal).all() or not np.isfinite(tangent_xy).all():
    raise ValueError("force samples must be finite")
  return group["timestamp"][:], normal, tangent_xy, np.linalg.norm(tangent_xy, axis=-1)


def export_curves(file, output):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output.mkdir()
  t, normal, xy, tangent = right_force_values(file)
  phases = file["commands/phase"].asstr()[:]
  fingers = ("thumb", "index", "middle", "ring", "little")
  if len(t) != len(phases):
    raise ValueError("force and phase streams differ in length")
  changes = np.r_[0, np.flatnonzero(phases[1:] != phases[:-1]) + 1]
  fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
  for j, finger in enumerate(fingers):
    for column, (values, label) in enumerate(
      ((normal, "Normal Fn"), (tangent, "Tangential |Ft|"))
    ):
      ax = axes[j, column]
      ax.plot(t, values[:, j], linewidth=0.7)
      ax.set_ylabel(f"{finger} (N)")
      ax.grid(alpha=0.25)
      for k in changes:
        ax.axvline(t[k], color="grey", linewidth=0.5, alpha=0.3)
      if j == 0:
        ax.set_title(label)
      if j == 4:
        ax.set_xlabel("Recorded solver time (s)")
  fig.suptitle("Poker draw | shared folded home | right fingertip forces (unfiltered)")
  fig.tight_layout()
  fig.savefig(output / "right_hand_force_curves.png", dpi=150)
  plt.close(fig)
  result = {
    "fingers": fingers,
    "sample_count": len(t),
    "sampling_hz": int(file.attrs["control_hz"]),
    "normal_peak_n": normal.max(axis=0).tolist(),
    "tangent_peak_n": tangent.max(axis=0).tolist(),
    "normal_definition": "sum of measured normal taxels for each fingertip",
    "tangent_definition": "norm of summed signed tangent XY taxels in the pad-local frame",
    "filter": "none; no clipping, smoothing, resampling, or synthetic forces",
    "phase_start_s": [{"phase": phases[k], "time_s": float(t[k])} for k in changes],
  }
  return result


def export_review(file, output, frames, *, fps=10):
  """Stream recorded head/right-wrist RGB and right tactile directly to MP4."""
  right = _force_layout(file)
  writer = _FfmpegPipeWriter(output / "review.mp4", fps=fps, width=1280, height=720)
  try:
    for frame in frames:
      group = file["tactile_contact_force"]
      tactile = frame.tactile_index
      composite, _ = compose_review_frame(
        file["cameras/head/rgb"][frame.camera_index],
        file["cameras/right_wrist/rgb"][frame.camera_index],
        group["normal_taxel_force_n"][tactile][right],
        group["tangent_taxel_force_n"][tactile][right],
        frame=frame,
        width=1280,
        height=720,
        phase=str(file["commands/phase"].asstr()[tactile]),
        caption="Recorded RGB + solver contact spatial estimates | signed Ft preserved in Raw HDF5",
        second_camera_label="RIGHT WRIST",
      )
      writer.write(np.asarray(composite))
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(output / "review.mp4")
  _validate_probe(probe, count=len(frames), fps=fps, width=1280, height=720)
  return probe


def export_global(file, output, frames, *, fps=10):
  """Restore recorded poses for display only; never call mj_step or mj_forward."""
  model_path = default_model_path("poker-draw")
  metadata = json.loads(file.attrs["metadata_json"])
  if metadata.get("base_model_fingerprint") != model_fingerprint(model_path):
    raise ValueError(
      "current scene differs from the raw episode; refuse a mislabeled replay"
    )
  model = mujoco.MjModel.from_xml_path(str(model_path))
  if tuple(file["state/full_qpos_names"].asstr()[:]) != tuple(_qpos_names(model)):
    raise ValueError("global replay joint mapping differs from raw")
  data = mujoco.MjData(model)
  initial = file["state/qpos"][0]
  for side in ("left", "right"):
    addresses = [
      int(model.joint(f"{side}_arm_joint{j}").qposadr[0]) for j in range(1, 8)
    ]
    np.testing.assert_allclose(initial[addresses], ARM_HOME[side], rtol=0, atol=1e-12)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (0.25, 0.0, 0.86)
  camera.distance, camera.azimuth, camera.elevation = 2.25, 135.0, -20.0
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[[0, 1, 2]] = 1
  timestamps = file["state/timestamp"][:]
  source_times = np.array([frame.camera_pose_timestamp_s for frame in frames])
  # Use nearest stored state; never interpolate qpos or imply sensor-perfect timing.
  upper = np.searchsorted(timestamps, source_times).clip(0, len(timestamps) - 1)
  lower = np.maximum(upper - 1, 0)
  indices = np.where(
    abs(timestamps[lower] - source_times) <= abs(timestamps[upper] - source_times),
    lower,
    upper,
  )
  prepare_render_backend()
  video = output / "robot_global_short_path.mp4"
  writer = _FfmpegPipeWriter(video, fps=fps, width=960, height=540)
  try:
    with mujoco.Renderer(model, width=960, height=540) as renderer:
      renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
      backend = current_backend()
      for frame, index in zip(frames, indices, strict=True):
        data.qpos[:] = file["state/qpos"][index]
        data.time = timestamps[index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera=camera, scene_option=option)
        rgb = renderer.render().copy()
        writer.write(rgb)
        if frame.output_index % 100 == 0:
          print(f"global video: {frame.output_index}/{len(frames)}", flush=True)
    writer.finish()
  except BaseException:
    writer.abort()
    raise
  probe = _probe_video(video)
  _validate_probe(probe, count=len(frames), fps=fps, width=960, height=540)
  result = {
    "mode": "saved_qpos_kinematic_visualization",
    "physics_steps": 0,
    "training_rgb_replaced": False,
    "camera": "display-only free camera; not a sensor",
    "lookat": camera.lookat.tolist(),
    "distance": camera.distance,
    "azimuth": camera.azimuth,
    "elevation": camera.elevation,
    "frame_count": len(frames),
    "fps": fps,
    "probe": probe,
    "renderer": backend,
    "maximum_state_vs_sensor_time_error_s": float(
      np.max(abs(timestamps[indices] - source_times))
    ),
  }
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Existing directory containing raw; review and curves must be new",
  )
  args = parser.parse_args(argv)
  source, root = args.source.resolve(), args.output_dir.resolve()
  for name in ("review", "curves"):
    if (root / name).exists():
      raise FileExistsError(f"refusing to overwrite {root / name}")
  root.mkdir(parents=True, exist_ok=True)
  source_sha = _sha256(source)
  validation = validate_episode(source)
  if not validation.valid:
    raise ValueError(validation.errors)
  (root / "review").mkdir()
  with h5py.File(source, "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    outcome = json.loads(file.attrs["outcome_json"])
    if metadata.get("scene") != "poker-draw" or outcome.get("success") is not True:
      raise ValueError("compact example requires one successful poker-draw episode")
    frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
    print("Exporting head/right-wrist/tactile review", flush=True)
    review = export_review(file, root / "review", frames)
    print("Exporting right-hand force curve", flush=True)
    statistics = export_curves(file, root / "curves")
    print("Exporting saved-qpos global review", flush=True)
    overview = export_global(file, root / "review", frames)
  if _sha256(source) != source_sha:
    raise RuntimeError("raw source changed during export")
  print(
    json.dumps(
      {
        "completed": True,
        "source_sha256": source_sha,
        "review": review,
        "global": overview["probe"],
        "force_samples": statistics["sample_count"],
      },
      indent=2,
    ),
    flush=True,
  )


if __name__ == "__main__":
  main()
