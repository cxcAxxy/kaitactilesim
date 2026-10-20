#!/usr/bin/env python3
"""Record one new-task physical expert episode without overwriting output."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from kaihand_tactile_env.shared.config import TRAINING_CAMERA_NAMES
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_TASKS,
  create_variant_expert,
)
from kaihand_tactile_env.shared.variant_recording import record_variant_episode


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=VARIANT_TASKS, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument("--buffer-rows", type=int, default=128)
  parser.add_argument("--cameras", nargs="+", choices=TRAINING_CAMERA_NAMES,
                      default=TRAINING_CAMERA_NAMES)
  args = parser.parse_args(argv)
  if args.seed is not None and args.seed < 0:
    parser.error("--seed must be nonnegative")
  if args.camera_hz != 30:
    parser.error("production Raw requires --camera-hz 30")
  if not 0 <= args.buffer_rows <= 256:
    parser.error("--buffer-rows must be in 0..256")
  if tuple(args.cameras) != TRAINING_CAMERA_NAMES:
    parser.error("--cameras must be exactly: head left_wrist right_wrist")
  if args.output_dir.exists():
    parser.error("--output-dir must be new; existing episodes are never overwritten")
  sim, executor = create_variant_expert(args.task, args.seed)
  result = record_variant_episode(
    sim, executor, args.output_dir, seed=args.seed, camera_hz=args.camera_hz,
    cameras=tuple(args.cameras), buffer_rows=args.buffer_rows,
  )
  print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
  return 0 if result.get("success") is True else 1


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  raise SystemExit(main())
