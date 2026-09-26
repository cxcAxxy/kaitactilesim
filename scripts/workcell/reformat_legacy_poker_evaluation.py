#!/usr/bin/env python3
"""Reformat a legacy Poker π0.5 review without pretending to replay its rollout.

Only the archived 10 fps composite video and its matching tactile frame log are
used.  Camera panels are cropped from lossy encoded pixels; joint and wrist
state at the 30 Hz control rate cannot be recovered from these artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from kaihand_tactile_env.shared.evaluation_video import compose_evaluation_frame
from kaihand_tactile_env.shared.task_video import (
  _FfmpegPipeWriter,
  _find_ffmpeg_executable,
)

SOURCE_SIZE = (1920, 1080)
SOURCE_RENDER_SIZE = (640, 480)
FPS = 10
SOURCE_SCHEMA = "kaihand-policy-evaluation-review-v2"
OUTPUT_SCHEMA = "kaihand-legacy-poker-evaluation-reformat-v1"

# The old v2 compositor used x=8..644 camera boxes.  Source RGB was letterboxed
# into x=108..543, and the overlaid labels obscured its top-left pixels.  Trim
# both edges rather than showing a second label in the new compositor.
CAMERA_CROPS_XYXY = {
  "head": (150, 105, 543, 402),
  "right_wrist": (150, 439, 543, 736),
  "global": (150, 773, 543, 1072),
}
_EMPTY_HISTORY = np.empty((0, 10, 3), dtype=np.float64)


def _load_source_review(path: Path) -> dict[str, Any]:
  review = json.loads(path.read_text(encoding="utf-8"))
  if review.get("schema") != SOURCE_SCHEMA or review.get("task") != "poker-draw":
    raise ValueError("source must be a legacy v2 poker-draw evaluation review")
  if review.get("fps") != FPS or review.get("output_size") != list(SOURCE_SIZE):
    raise ValueError("source review must be 1920x1080 at 10 fps")
  if review.get("review_render_size") != list(SOURCE_RENDER_SIZE):
    raise ValueError("source review render size differs from the known legacy layout")
  if review.get("second_camera") != "global":
    raise ValueError("source review must display the global third camera")
  if review.get("model_input_cameras_displayed") != ["head", "right_wrist"]:
    raise ValueError("source review must display head and model-input right wrist")
  if review.get("layout") != (
    "head/right-wrist/global-or-named RGB; bilateral Fn/|Ft| maps; "
    "bilateral 3-axis mean curves"
  ):
    raise ValueError("source review does not match the known v2 panel layout")
  if not isinstance(review.get("frame_count"), int) or review["frame_count"] < 1:
    raise ValueError("source review has no valid frame_count")
  names = review.get("fingertip_order")
  if not isinstance(names, list) or len(names) != 10 or len(set(names)) != 10:
    raise ValueError("source review must identify ten unique fingertips")
  if not all(isinstance(name, str) and name for name in names):
    raise ValueError("source review has invalid fingertip names")
  return review


def _load_frame_log(path: Path, review: dict[str, Any]) -> list[dict[str, Any]]:
  frames: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        raise ValueError(f"blank frame-log line {line_number}")
      row = json.loads(line)
      index = len(frames)
      if row.get("frame") != index or row.get("control_tick") != 3 * index:
        raise ValueError(f"frame {index} index/control_tick mismatch")
      for key in ("simulation_time_s", "camera_pose_time_s", "tactile_time_s"):
        value = row.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
          raise ValueError(f"frame {index} has invalid {key}")
      time_s = float(row["simulation_time_s"])
      if abs(time_s - index / FPS) > 0.005:
        raise ValueError(f"frame {index} does not follow the 10 fps video clock")
      for key in ("camera_pose_time_s", "tactile_time_s"):
        age = time_s - float(row[key])
        if not -0.005 <= age <= 1 / 30 + 0.005:
          raise ValueError(f"frame {index} has unsynchronized {key}")
      if row.get("fingertip_order") != review["fingertip_order"]:
        raise ValueError(f"frame {index} fingertip order differs from review.json")
      if not isinstance(row.get("phase"), str):
        raise ValueError(f"frame {index} has invalid phase")
      if not isinstance(row.get("task_metrics"), dict):
        raise ValueError(f"frame {index} has invalid task_metrics")
      normal = np.asarray(row.get("normal_taxel_force_n"), dtype=np.float64)
      tangent = np.asarray(row.get("tangent_taxel_force_n"), dtype=np.float64)
      if normal.shape != (10, 7, 5) or tangent.shape != (10, 7, 5, 2):
        raise ValueError(f"frame {index} has invalid tactile grid shape")
      if not np.isfinite(normal).all() or not np.isfinite(tangent).all():
        raise ValueError(f"frame {index} has nonfinite tactile force")
      if np.any(normal < -1.0e-12):
        raise ValueError(f"frame {index} has negative normal force")
      row["normal_taxel_force_n"] = normal
      row["tangent_taxel_force_n"] = tangent
      frames.append(row)
  if len(frames) != review["frame_count"]:
    raise ValueError(
      f"frame log has {len(frames)} frames, expected {review['frame_count']}"
    )
  return frames


def _probe_source_video(path: Path, expected_frames: int) -> None:
  executable = shutil.which("ffprobe")
  if executable is None:
    raise RuntimeError("ffprobe is required to validate the legacy video")
  result = subprocess.run(
    [
      executable,
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=width,height,avg_frame_rate,nb_frames",
      "-of",
      "json",
      str(path),
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  streams = json.loads(result.stdout).get("streams", [])
  if len(streams) != 1:
    raise ValueError("legacy video must contain exactly one selected video stream")
  stream = streams[0]
  if (stream.get("width"), stream.get("height")) != SOURCE_SIZE:
    raise ValueError("legacy video dimensions differ from review.json")
  if Fraction(stream.get("avg_frame_rate", "0")) != FPS:
    raise ValueError("legacy video frame rate differs from review.json")
  if int(stream.get("nb_frames", -1)) != expected_frames:
    raise ValueError("legacy video frame count differs from frames.jsonl")


def _crop_legacy_cameras(rgb: np.ndarray) -> dict[str, np.ndarray]:
  if rgb.dtype != np.uint8 or rgb.shape != (SOURCE_SIZE[1], SOURCE_SIZE[0], 3):
    raise ValueError("decoded legacy frame must be 1920x1080 RGB uint8")
  return {
    name: rgb[top:bottom, left:right].copy()
    for name, (left, top, right, bottom) in CAMERA_CROPS_XYXY.items()
  }


def _read_exact_frame(stream: Any, buffer: bytearray, index: int) -> np.ndarray:
  view = memoryview(buffer)
  position = 0
  while position < len(buffer):
    count = stream.readinto(view[position:])
    if count is None or count <= 0:
      raise ValueError(f"legacy video ended inside decoded frame {index}")
    position += count
  return np.frombuffer(buffer, dtype=np.uint8).reshape(
    SOURCE_SIZE[1], SOURCE_SIZE[0], 3
  )


def _compose_legacy_frame(rgb: np.ndarray, row: dict[str, Any]):
  cameras = _crop_legacy_cameras(rgb)
  return compose_evaluation_frame(
    cameras["head"],
    cameras["global"],
    row["normal_taxel_force_n"],
    row["tangent_taxel_force_n"],
    _EMPTY_HISTORY,
    width=SOURCE_SIZE[0],
    height=SOURCE_SIZE[1],
    simulation_time_s=float(row["simulation_time_s"]),
    phase=row["phase"],
    second_camera_label="GLOBAL / REVIEW ONLY",
    model_wrist_rgb=cameras["right_wrist"],
    head_is_model_input=True,
    metrics=row["task_metrics"],
    heading="POKER LEGACY REFORMAT / 10 FPS",
  )


def reformat(source_dir: Path, *, preview_only: bool = False) -> Path:
  source = source_dir.expanduser().resolve(strict=True)
  if not source.is_dir():
    raise ValueError("source must be a seed evaluation directory")
  legacy_review = source / "review"
  video_path = legacy_review / "review.mp4"
  frame_log_path = legacy_review / "frames.jsonl"
  review_path = legacy_review / "review.json"
  summary_path = source / "summary.json"
  for path in (video_path, frame_log_path, review_path, summary_path):
    if not path.is_file():
      raise FileNotFoundError(path)
  destination = source / (
    "legacy_reformatted_preview" if preview_only else "legacy_reformatted"
  )
  if destination.exists():
    raise FileExistsError(f"refusing to overwrite {destination}")

  review = _load_source_review(review_path)
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  for key in ("seed", "checkpoint_path", "execute_steps", "diagnostic_only"):
    if summary.get(key) != review.get(key):
      raise ValueError(f"summary.json and review.json disagree on {key}")
  if summary.get("penetration_guard_enabled") != review.get(
    "penetration_guard_enabled"
  ):
    raise ValueError("summary.json and review.json disagree on penetration guard")
  frames = _load_frame_log(frame_log_path, review)
  _probe_source_video(video_path, len(frames))
  ffmpeg = _find_ffmpeg_executable()
  destination.mkdir(exist_ok=False)
  writer: _FfmpegPipeWriter | None = None
  decoder: subprocess.Popen[bytes] | None = None
  try:
    if not preview_only:
      writer = _FfmpegPipeWriter(
        destination / "review.mp4",
        fps=FPS,
        width=SOURCE_SIZE[0],
        height=SOURCE_SIZE[1],
        executable=ffmpeg,
      )
    command = [
      ffmpeg,
      "-hide_banner",
      "-loglevel",
      "error",
      "-nostdin",
      "-i",
      str(video_path),
      "-f",
      "rawvideo",
      "-pix_fmt",
      "rgb24",
      "-vsync",
      "0",
    ]
    if preview_only:
      command.extend(("-frames:v", "1"))
    command.append("pipe:1")
    decoder = subprocess.Popen(
      command,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    if decoder.stdout is None or decoder.stderr is None:
      raise RuntimeError("ffmpeg decoder did not expose its output pipes")
    buffer = bytearray(SOURCE_SIZE[0] * SOURCE_SIZE[1] * 3)
    selected_frames = frames[:1] if preview_only else frames
    for index, row in enumerate(selected_frames):
      rgb = _read_exact_frame(decoder.stdout, buffer, index)
      image = _compose_legacy_frame(rgb, row)
      if writer is not None:
        writer.write(np.asarray(image, dtype=np.uint8))
      if index == 0:
        image.save(destination / "first_frame.png")
      if not preview_only and index == len(frames) - 1:
        image.save(destination / "last_frame.png")
    if decoder.stdout.read(1):
      raise ValueError("legacy video contains more decoded frames than frames.jsonl")
    return_code = decoder.wait(timeout=30)
    detail = decoder.stderr.read().decode("utf-8", errors="replace").strip()
    if return_code != 0:
      raise RuntimeError(f"ffmpeg failed to decode legacy video: {detail}")
    if writer is not None:
      writer.finish()
    if preview_only:
      return destination
    metadata = {
      "schema": OUTPUT_SCHEMA,
      "task": "poker-draw",
      "status": "completed",
      "source_seed_directory": str(source),
      "source_video": str(video_path),
      "source_frame_log": str(frame_log_path),
      "source_review_metadata": str(review_path),
      "source_summary": str(summary_path),
      "source_review_schema": review["schema"],
      "video": "review.mp4",
      "first_frame": "first_frame.png",
      "last_frame": "last_frame.png",
      "fps": FPS,
      "frame_count": len(frames),
      "output_size": list(SOURCE_SIZE),
      "layout": "head, right wrist, global; bilateral Fn and |Ft| heatmaps; no curves",
      "camera_source": (
        "Lossy crops of the archived H.264 composite video, not original RGB renders; "
        "legacy overlaid labels and nearby top/left camera pixels were removed."
      ),
      "camera_crop_boxes_xyxy": CAMERA_CROPS_XYXY,
      "tactile_source": (
        "Archived frames.jsonl 10 fps 7x5 taxel values, redrawn without smoothing"
      ),
      "sampling": (
        "Video and tactile are 10 fps only; no 30 Hz right-wrist pose, joint, "
        "or tactile trajectory is reconstructed."
      ),
      "comparison_plots": {
        "status": "30hz_not_available",
        "reason": (
          "30 Hz measured wrist and joint states were not archived; "
          "lower-rate descriptive plots may be provided separately."
        ),
      },
      "source_seed": review.get("seed"),
      "source_checkpoint_path": review.get("checkpoint_path"),
      "source_execute_steps": review.get("execute_steps"),
      "source_diagnostic_only": review.get("diagnostic_only"),
      "source_penetration_guard_enabled": summary.get("penetration_guard_enabled"),
      "source_formal_metrics_valid": summary.get("formal_metrics_valid"),
      "formal_metrics_valid": False,
      "evaluation_scope": (
        "Visualization of an existing diagnostic-only rollout; not a new formal "
        "evaluation and not a newly measured success-rate trial."
      ),
      "source_evaluation": review.get("evaluation"),
    }
    (destination / "review.json").write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
      encoding="utf-8",
    )
  except BaseException:
    if writer is not None:
      writer.abort()
    if decoder is not None and decoder.poll() is None:
      decoder.terminate()
      try:
        decoder.wait(timeout=3)
      except subprocess.TimeoutExpired:
        decoder.kill()
        decoder.wait(timeout=3)
    raise
  finally:
    if decoder is not None:
      if decoder.stdout is not None:
        decoder.stdout.close()
      if decoder.stderr is not None:
        decoder.stderr.close()
  return destination


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source_dir", type=Path, help="Legacy seed_000 evaluation directory")
  parser.add_argument(
    "--preview-only",
    action="store_true",
    help="Write only a first-frame preview in a separate new directory",
  )
  args = parser.parse_args()
  print(reformat(args.source_dir, preview_only=args.preview_only))


if __name__ == "__main__":
  main()
