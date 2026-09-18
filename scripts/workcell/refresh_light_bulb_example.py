"""Regenerate the maintained bulb example, replacing it only after validation."""

import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from kaihand_tactile_env.shared.cameras import (
  SHARED_CAMERA_NAMES,
  TRAINING_CAMERA_NAMES,
)
from kaihand_tactile_env.tasks.bulb_screw.example import (
  record_raw_episode,
  refresh_example,
)

if __name__ == "__main__":
  # Let the recorder's temporary-directory context clean up stopped captures,
  # including shell jobs that inherited an ignored SIGINT disposition.
  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output", type=Path, default=Path("datasets/light_bulb_example")
  )
  parser.add_argument(
    "--position-seed", type=int, help="Randomize pickup X/Y within +/-2 mm"
  )
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument(
    "--buffer-rows",
    type=int,
    default=128,
    help="Lossless non-camera HDF5 append buffer, 0 disables; allowed 0..256",
  )
  parser.add_argument(
    "--raw-only",
    action="store_true",
    help="Save only HDF5, task result and minimal validation summary",
  )
  parser.add_argument(
    "--cameras",
    nargs="+",
    choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
  )
  args = parser.parse_args()
  if args.position_seed is not None and args.position_seed < 0:
    parser.error("--position-seed must be nonnegative")
  if args.camera_hz <= 0:
    parser.error("--camera-hz must be positive")
  if tuple(args.cameras) != TRAINING_CAMERA_NAMES:
    parser.error("--cameras must be exactly: head left_wrist right_wrist")
  recorder = record_raw_episode if args.raw_only else refresh_example
  recorder_kwargs = dict(
    position_seed=args.position_seed,
    camera_hz=args.camera_hz,
    cameras=tuple(args.cameras),
  )
  if args.raw_only:
    recorder_kwargs["buffer_rows"] = args.buffer_rows
  recorder(args.output, **recorder_kwargs)
