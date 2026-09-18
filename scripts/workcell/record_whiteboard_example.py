#!/usr/bin/env python3
"""Record the whiteboard example with 500 Hz touch and native camera images.

Existing output is preserved by default. --overwrite replaces only the selected
output directory without creating a backup; the default is
datasets/erase_whiteboard_example.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/whiteboard_matplotlib")

from kaihand_tactile_env.tasks.whiteboard_wipe.execution import WhiteboardWipeExecutor
from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES, TRAINING_CAMERA_NAMES
from kaihand_tactile_env.tasks.whiteboard_wipe.recording import RawCapture, WhiteboardRecorder
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/erase_whiteboard_example"


def _validate_output(args):
  """Validate before any deletion; source and destination must stay separate."""
  output = args.output_dir
  if output.is_symlink():
    raise ValueError("--output-dir must not be a symbolic link")
  resolved = output.resolve()
  if args.render_existing is not None:
    source = args.render_existing.resolve()
    if resolved == source or resolved in source.parents or source in resolved.parents:
      raise ValueError("--output-dir and --render-existing must not overlap")
    if (
      not (source / "raw/episode.h5").is_file()
      or not (source / "result.json").is_file()
    ):
      raise ValueError("--render-existing must contain raw/episode.h5 and result.json")
  if not output.exists():
    return
  if not args.overwrite:
    raise ValueError(
      "--output-dir already exists; use --overwrite to replace it without a backup"
    )
  if not output.is_dir():
    raise ValueError("--overwrite only accepts an output directory, not a file")
  protected_roots = {
    PROJECT_ROOT,
    *PROJECT_ROOT.parents,
    PROJECT_ROOT / "datasets",
    Path(tempfile.gettempdir()).resolve(),
  }
  if resolved in protected_roots:
    raise ValueError(
      "--overwrite refuses the project root, datasets root or a filesystem root"
    )
  for name in ("src", "scripts", "tests", "docs", ".git", ".codex", ".agents"):
    protected = PROJECT_ROOT / name
    if resolved == protected or protected in resolved.parents:
      raise ValueError(
        "--overwrite refuses project source and configuration directories"
      )


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output directory (default: datasets/erase_whiteboard_example)",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete and replace only --output-dir; no backup is created",
  )
  parser.add_argument("--video-fps", type=int, choices=(2, 5, 10), default=10)
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument(
    "--buffer-rows",
    type=int,
    default=128,
    help="Raw-only lossless non-camera HDF5 append buffer, 0 disables; allowed 0..256",
  )
  parser.add_argument(
    "--cameras",
    nargs="+",
    choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
    help="Raw training cameras; the review-only board_overview camera is unchanged",
  )
  parser.add_argument(
    "--raw-only",
    action="store_true",
    help="Save only shared three-camera Raw HDF5/result/validation sidecars",
  )
  parser.add_argument(
    "--render-existing",
    type=Path,
    help="Render a legacy native 500 Hz capture; shared Raw uses replay_episode.py",
  )
  parser.add_argument(
    "--ink-seed",
    type=int,
    help="Randomize ink center reproducibly within the supported board region",
  )
  args = parser.parse_args(argv)
  if args.ink_seed is not None and args.ink_seed < 0:
    parser.error("--ink-seed must be nonnegative")
  if args.render_existing and args.ink_seed is not None:
    parser.error("--render-existing restores saved ink; do not pass --ink-seed")
  if args.raw_only and args.render_existing:
    parser.error("--raw-only and --render-existing are mutually exclusive")
  if not np.isfinite(args.camera_hz) or args.camera_hz <= 0:
    parser.error("--camera-hz must be finite and positive")
  if not 0 <= args.buffer_rows <= 256:
    parser.error("--buffer-rows must be in 0..256")
  if args.raw_only and tuple(args.cameras) != TRAINING_CAMERA_NAMES:
    parser.error("--cameras must be exactly: head left_wrist right_wrist")
  try:
    _validate_output(args)
  except ValueError as error:
    parser.error(str(error))
  # Absolute paths keep the deletion target stable throughout initialization.
  args.output_dir = args.output_dir.resolve()
  if args.render_existing is not None:
    args.render_existing = args.render_existing.resolve()
  return args


def main(argv=None):
  args = parse_args(argv)
  sim = WhiteboardWipeSimulation(ink_seed=args.ink_seed)
  _validate_output(args)
  if args.output_dir.exists():
    shutil.rmtree(args.output_dir)
  if args.render_existing is not None:
    result = WhiteboardRecorder.render_existing(
      sim, args.render_existing, args.output_dir, video_fps=args.video_fps
    )
    return 0 if result.get("success", False) else 1
  recorder = (
    RawCapture(
      sim,
      args.output_dir,
      camera_hz=args.camera_hz,
      cameras=tuple(args.cameras),
      buffer_rows=args.buffer_rows,
    )
    if args.raw_only
    else WhiteboardRecorder(sim, args.output_dir, video_fps=args.video_fps)
  )
  try:
    result = WhiteboardWipeExecutor(sim, recorder.observe).run()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.raw_only:
      recorder.finish(result)
    else:
      recorder.finish(result, render=True)
    return 0 if result.get("success", False) else 1
  finally:
    recorder.close()


if __name__ == "__main__":
  raise SystemExit(main())
