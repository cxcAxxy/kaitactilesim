#!/usr/bin/env python3
"""Export a Card HDF5 replay with ten-finger heatmaps and force curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaihand_tactile_env.shared.usb_review_video import export_card_review_video


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path, help="Completed successful Card .h5 episode")
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="New output directory; existing exports are never overwritten",
  )
  parser.add_argument("--fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
  args = parser.parse_args()
  result = export_card_review_video(
    args.source,
    args.output_dir,
    fps=args.fps,
    width=args.width,
    height=args.height,
  )
  print(
    f"{args.output_dir}: {result['output_frame_count']} frames at "
    f"{result['fps']} fps; duration={result['constant_fps_duration_s']:.3f}s"
  )


if __name__ == "__main__":
  main()
