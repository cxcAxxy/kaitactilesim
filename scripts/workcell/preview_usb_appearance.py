#!/usr/bin/env python3
"""Render a USB appearance review from saved poses without stepping physics.

RGB is explicitly a re-render; tactile values come from the read-only source.
This does not replace captured RGB, a training dataset, or a physical rollout.
The optional reference MJB produces comparisons at identical poses/cameras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio_ffmpeg
import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import (
  CameraConfig,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.offline_replay import (
  compose_multimodal_replay_frame,
  read_tactile_sample,
  resolve_tactile_source,
  tactile_scales,
)
from kaihand_tactile_env.shared.recording import _qpos_names
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from PIL import Image, ImageDraw, ImageFont


def digest(path):
  with Path(path).open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def restore(model, data, file, index):
  data.qpos[:] = file["state/qpos"][index]
  data.qvel[:] = file["state/qvel"][index]
  data.time = float(file["state/timestamp"][index])
  mujoco.mj_forward(model, data)


def comparison(before, after, output, label):
  canvas = Image.new("RGB", (before.width * 2, before.height + 56), "#101820")
  canvas.paste(before, (0, 56))
  canvas.paste(after, (before.width, 56))
  draw = ImageDraw.Draw(canvas)
  font = ImageFont.truetype("DejaVuSans.ttf", 19)
  draw.text((20, 17), f"BEFORE / {label}", font=font, fill="#aeb9c5")
  draw.text((before.width + 20, 17), f"SILVER LAB / {label}", font=font, fill="white")
  canvas.save(output)


def overview(model, data):
  # Display-only free camera: never add or modify a named sensor camera.
  camera = mujoco.MjvCamera()
  camera.lookat[:] = (0.25, 0, 0.86)
  camera.distance = 2.25
  camera.azimuth = 135
  camera.elevation = -20
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[[0, 1, 2]] = 1
  with mujoco.Renderer(model, height=720, width=1280) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=option)
    return Image.fromarray(renderer.render())


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--reference-model", type=Path)
  parser.add_argument("--appearance-name", default="white-silver-lab-v2")
  parser.add_argument("--fps", type=int, default=10, choices=(5, 10, 15, 30))
  args = parser.parse_args()
  if args.output_dir.exists():
    parser.error("output directory must be new")
  source_hash = digest(args.source)
  model = mujoco.MjModel.from_xml_path(str(default_model_path("usb-insert")))
  reference = (
    mujoco.MjModel.from_binary_path(str(args.reference_model))
    if args.reference_model
    else None
  )
  data = mujoco.MjData(model)
  output = args.output_dir
  with h5py.File(args.source, "r") as file:
    if json.loads(file.attrs["metadata_json"])["scene"] != "usb-insert":
      raise ValueError("this preview only supports USB recordings")
    recorded_names = tuple(v.decode() for v in file["state/full_qpos_names"][:])
    if recorded_names != tuple(_qpos_names(model)):
      raise ValueError("recorded joints do not match the current model")
    cameras = tuple(
      CameraConfig(
        name,
        width=file[f"cameras/{name}/rgb"].shape[2],
        height=file[f"cameras/{name}/rgb"].shape[1],
        depth=False,
        segmentation=False,
      )
      for name in ("head", "right_wrist")
    )
    times = file["cameras/head/timestamp"][:]
    state_indices = file["cameras/head/state_index"][:]
    for camera in cameras:
      if not np.array_equal(times, file[f"cameras/{camera.name}/timestamp"][:]):
        raise ValueError("recorded cameras must have the same clock")
      if not np.array_equal(
        state_indices, file[f"cameras/{camera.name}/state_index"][:]
      ):
        raise ValueError("recorded cameras must have the same state indices")
    targets = np.arange(0, times[-1] + 1e-9, 1 / args.fps)
    selected = np.maximum(0, np.searchsorted(times, targets, side="right") - 1)
    tactile = resolve_tactile_source(file)
    tactile_indices = np.maximum(
      0, np.searchsorted(tactile.times, times[selected], side="right") - 1
    )
    scales = tactile_scales(file, tactile, tactile_indices)
    output.mkdir(parents=True)
    restore(model, data, file, int(state_indices[0]))
    hero = overview(model, data)
    hero.save(output / "overview.png")
    if reference is not None:
      if tuple(_qpos_names(reference)) != recorded_names:
        raise ValueError("reference model joints differ from the source")
      reference_data = mujoco.MjData(reference)
      restore(reference, reference_data, file, int(state_indices[0]))
      before_overview = overview(reference, reference_data)
      before_overview.save(output / "before_overview.png")
      comparison(
        before_overview,
        hero,
        output / "comparison_overview.png",
        "same pose",
      )
      with WorkcellRenderer(reference, cameras, visible_geom_groups=(0, 1, 2)) as old:
        for camera in cameras:
          pixels = old.capture(reference_data, camera)["rgb"]
          Image.fromarray(pixels).save(output / f"before_{camera.name}.png")
    history = []
    max_extrinsic_error = 0.0
    writer = imageio_ffmpeg.write_frames(
      str(output / "review.mp4"),
      (1280, 720),
      fps=args.fps,
      codec="libx264",
      pix_fmt_out="yuv420p",
      macro_block_size=1,
      output_params=["-crf", "19", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
      with WorkcellRenderer(model, cameras, visible_geom_groups=(0, 1, 2)) as renderer:
        with (output / "frames.jsonl").open("w") as log:
          for frame, (camera_index, tactile_index) in enumerate(
            zip(selected, tactile_indices, strict=True)
          ):
            index = int(state_indices[camera_index])
            restore(model, data, file, index)
            camera_frames = []
            for camera in cameras:
              calibration = renderer.calibration(data, camera)
              saved = file[f"cameras/{camera.name}/world_from_camera"][camera_index]
              error = float(np.max(np.abs(calibration.world_from_camera - saved)))
              max_extrinsic_error = max(max_extrinsic_error, error)
              np.testing.assert_allclose(
                calibration.world_from_camera, saved, atol=1e-8, rtol=0
              )
              np.testing.assert_allclose(
                calibration.intrinsic,
                file[f"cameras/{camera.name}/intrinsic"][:],
                atol=1e-8,
                rtol=0,
              )
              rgb = renderer.capture(data, camera)["rgb"]
              camera_frames.append((camera.name, rgb))
              if frame == 0:
                new = Image.fromarray(rgb)
                new.save(output / f"after_{camera.name}.png")
                if reference is not None:
                  with Image.open(output / f"before_{camera.name}.png") as old:
                    comparison(
                      old, new, output / f"comparison_{camera.name}.png", camera.name
                    )
            first, second, curves = read_tactile_sample(
              file, tactile, int(tactile_index)
            )
            history.append(curves)
            phase = file["commands/phase"][index].decode()
            composed = compose_multimodal_replay_frame(
              tuple(camera_frames),
              first,
              second,
              np.asarray(history),
              width=1280,
              height=720,
              timestamp_s=data.time,
              phase=phase,
              heading=args.appearance_name.upper()
              + " | RGB RE-RENDER / RECORDED TACTILE",
              first_label=tactile.first_label,
              second_label=tactile.second_label,
              first_unit=tactile.first_unit,
              second_unit=tactile.second_unit,
              curve_unit=tactile.curve_unit,
              curve_channels=tactile.curve_channels,
              first_maximum=scales[0],
              second_maximum=scales[1],
              curve_maximum=scales[2],
            )
            # The shared compositor labels archived RGB as RECORDED; this
            # utility renders new pixels, so explicitly replace those labels.
            draw = ImageDraw.Draw(composed)
            label_font = ImageFont.truetype("DejaVuSans.ttf", 18)
            for camera_number, camera in enumerate(cameras):
              top = 78 + camera_number * 320
              draw.rectangle((8, top, 424, top + 28), fill="black")
              draw.text(
                (12, top + 4),
                camera.name.replace("_", " ").upper() + " / RE-RENDERED",
                font=label_font,
                fill="white",
              )
            writer.send(np.asarray(composed))
            if frame == 0:
              composed.save(output / "first_frame.png")
            if frame == len(selected) // 2:
              composed.save(output / "mid_frame.png")
            log.write(
              json.dumps(
                {
                  "frame": frame,
                  "source_camera_index": int(camera_index),
                  "state_index": index,
                  "tactile_index": int(tactile_index),
                  "timestamp_s": data.time,
                  "phase": phase,
                }
              )
              + "\n"
            )
            if frame % 50 == 0:
              print(f"Rendered {frame + 1}/{len(selected)}", flush=True)
          composed.save(output / "last_frame.png")
    finally:
      writer.close()
  if digest(args.source) != source_hash:
    raise RuntimeError("source changed during preview")
  report = {
    "appearance": args.appearance_name,
    "mode": "saved_pose_rgb_rerender",
    "source": str(args.source.resolve()),
    "source_sha256": source_hash,
    "reference_model_sha256": digest(args.reference_model) if reference else None,
    "model_fingerprint": model_fingerprint(default_model_path("usb-insert")),
    "source_unmodified": True,
    "physics_stepped": False,
    "cameras": [c.name for c in cameras],
    "camera_intrinsics_match_source": True,
    "maximum_camera_extrinsic_error": max_extrinsic_error,
    "tactile": "unchanged source samples; no recomputation or filtering",
    "frames": len(selected),
    "fps": args.fps,
    "note": "Display-only overview uses a free camera. Named sensor cameras are unchanged.",
  }
  (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
