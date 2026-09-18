"""Record a complete RAM installation with synchronized shared cameras and touch."""

import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from kaihand_tactile_env.shared.cameras import (
  SHARED_CAMERA_NAMES,
  TRAINING_CAMERA_NAMES,
)
from kaihand_tactile_env.tasks.install_ram.example import (
  record_example,
  record_raw_episode,
)

if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    "--output",
    dest="output",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "datasets/install_ram_example",
    help="Dataset directory; replace a prior RAM example only with --replace-existing",
  )
  parser.add_argument(
    "--replace-existing",
    action="store_true",
    help="Publish only after success; preserve the previous raw source and force curves in provenance",
  )
  parser.add_argument(
    "--buffer-rows",
    type=int,
    default=64,
    help="Lossless non-camera HDF5 append buffer, 0 disables; allowed 0..256",
  )
  parser.add_argument(
    "--diagnostics-dir",
    type=Path,
    help="Archive a manifest-verified directory of independent force ablations",
  )
  parser.add_argument(
    "--position-seed", type=int, help="Randomize RAM and support X/Y within +/-2 mm"
  )
  parser.add_argument("--camera-hz", type=int, default=30)
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
  if args.raw_only:
    if args.replace_existing or args.diagnostics_dir is not None:
      parser.error("--raw-only cannot use --replace-existing or --diagnostics-dir")
    record_raw_episode(
      args.output,
      position_seed=args.position_seed,
      buffer_rows=args.buffer_rows,
      camera_hz=args.camera_hz,
      cameras=tuple(args.cameras),
    )
  else:
    record_example(
      args.output,
      position_seed=args.position_seed,
      replace_existing=args.replace_existing,
      buffer_rows=args.buffer_rows,
      diagnostics_dir=args.diagnostics_dir,
      camera_hz=args.camera_hz,
      cameras=tuple(args.cameras),
    )
