#!/usr/bin/env python3
"""Record one validated sponge-to-plate Raw episode for batch collection."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES, TRAINING_CAMERA_NAMES
from kaihand_tactile_env.tasks.sponge_grasp.recording import record_raw_episode


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--seed", type=int,
    help="Deterministic sponge initial XY offset, currently +/-1 mm per axis",
  )
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument("--buffer-rows", type=int, default=128)
  parser.add_argument("--raw-only", action="store_true", help="Record Raw without review video")
  parser.add_argument(
    "--cameras", nargs="+", choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
  )
  args = parser.parse_args(argv)
  if args.seed is not None and args.seed < 0:
    parser.error("--seed must be nonnegative")
  if args.camera_hz <= 0:
    parser.error("--camera-hz must be positive")
  if not 0 <= args.buffer_rows <= 256:
    parser.error("--buffer-rows must be in 0..256")
  if tuple(args.cameras) != TRAINING_CAMERA_NAMES:
    parser.error("--cameras must be exactly: head left_wrist right_wrist")
  record_raw_episode(
    args.output_dir,
    seed=args.seed,
    camera_hz=args.camera_hz,
    cameras=tuple(args.cameras),
    buffer_rows=args.buffer_rows,
  )
  return 0


if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  raise SystemExit(main())
