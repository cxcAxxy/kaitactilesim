#!/usr/bin/env python3
"""Export synchronized review video/PNGs from a successful saved poker HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaihand_tactile_env.shared.poker_review import export_poker_review


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "source", type=Path, help="Completed successful poker .h5 episode"
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="New review directory; existing or partial exports are never overwritten",
  )
  parser.add_argument("--fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument(
    "--second-camera",
    choices=("right_wrist", "overhead", "front"),
    help="Default: right_wrist when recorded, otherwise legacy overhead; never synthesized.",
  )
  return parser.parse_args(argv)


def main():
  args = _parse_args()
  result = export_poker_review(
    args.source,
    args.output_dir,
    fps=args.fps,
    width=args.width,
    height=args.height,
    second_camera=args.second_camera,
  )
  print(
    f"{args.output_dir}: {result['output_frame_count']} frames at {result['fps']} fps; "
    f"last image t={result['last_camera_pose_timestamp_s']:.3f}s; "
    f"maximum playback timing error={result['maximum_absolute_playback_time_error_s']:.6f}s"
  )


if __name__ == "__main__":
  main()
