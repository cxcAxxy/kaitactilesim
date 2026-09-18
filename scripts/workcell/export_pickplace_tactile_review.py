#!/usr/bin/env python3
"""Export a pick-place HDF5 replay with Genesis probe maps and curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaihand_tactile_env.shared.pickplace_probe_review import export_pickplace_probe_review


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path, help="Completed pick-place .h5 episode")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
  args = parser.parse_args()
  report = export_pickplace_probe_review(
    args.source, args.output_dir, fps=args.fps, width=args.width, height=args.height
  )
  print(
    f"{args.output_dir}: {report['output_frame_count']} frames at {report['fps']} fps; "
    f"last image t={report['last_camera_timestamp_s']:.3f}s"
  )


if __name__ == "__main__":
  main()
