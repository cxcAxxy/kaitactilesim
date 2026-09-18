"""Record one successful pick/place example with videos and force curves."""

import argparse
import os
import signal
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from kaihand_tactile_env.tasks.pick_place.example import record_example

if __name__ == "__main__":
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir", type=Path, default=Path("datasets/pick_place_example")
  )
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  record_example(args.output_dir, seed=args.seed)
