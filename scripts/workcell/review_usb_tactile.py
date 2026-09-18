#!/usr/bin/env python3
"""Build offline physical tactile / RGB review sheets from one USB HDF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# No simulation, renderer, GUI or bulk image load; keep offline work small too.
for _name in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
):
  os.environ[_name] = "1"


def main(argv: list[str] | None = None) -> None:
  from kaihand_tactile_env.shared.usb_tactile_review import review_usb_tactile

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="A new review directory; never overwritten",
  )
  parser.add_argument("--camera", default="head")
  parser.add_argument("--signal-threshold-n", type=float, default=1e-6)
  parser.add_argument("--practical-threshold-n", type=float, default=0.01)
  args = parser.parse_args(argv)
  report = review_usb_tactile(
    args.input,
    args.output_dir,
    camera_name=args.camera,
    signal_threshold_n=args.signal_threshold_n,
    practical_threshold_n=args.practical_threshold_n,
  )
  print(
    json.dumps(
      {
        "output": str(args.output_dir.resolve()),
        "review_status": report["review_status"],
        "selected_rgb_frames": report["selected_rgb_frames"],
        "warnings": report["warnings"],
        "thresholds": [
          {
            "name": row["name"],
            "threshold_n": row["threshold_n"],
            "first_active_index": row["global"]["first_active_index"],
            "last_active_index": row["global"]["last_active_index"],
          }
          for row in report["thresholds"]
        ],
      },
      ensure_ascii=False,
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
