#!/usr/bin/env python3
"""Run the pinned OpenPI shared-task converter for locally registered tasks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from fast_pi05_conversion import install_fast_conversion

WHITEBOARD_INSTRUCTION = (
  "Pick up the eraser with the right hand, wipe all ink from the tilted "
  "whiteboard with loaded sliding contact, then return and release the eraser."
)
VASE_INSTRUCTION = (
  "Pick up the sponge with the right hand, wipe all stains from the far inner "
  "wall of the vase with loaded sliding contact, then lift the sponge clear."
)
SPONGE_GRASP_INSTRUCTION = (
  "Pick up the upright sponge with the right hand, carry it to the plate on "
  "the robot's right, place it inside the plate, and release it."
)


def _sponge_control_tick_prefix(timestamps: np.ndarray, physics_hz: int):
  """Check 30 Hz deadlines captured on sponge's exact 100 Hz control ticks."""
  if timestamps.ndim != 1 or timestamps.size < 2:
    raise ValueError("head camera needs at least two timestamps")
  if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
    raise ValueError("head camera timestamps must be finite and increasing")
  if physics_hz <= 0:
    raise ValueError("physics_hz must be positive")
  # The recorder observes the simulator after complete 10 ms control steps.
  # At 30 Hz the first frames therefore occur at 0, 40, 70, 100, 140 ms,
  # rather than on the physics clock's ideal 33.333 ms grid.
  frame = np.arange(timestamps.size, dtype=np.int64)
  control_steps = (frame * 100 + 29) // 30
  expected = timestamps[0] + control_steps / 100.0
  error = np.abs(timestamps - expected)
  tolerance = 0.51 / physics_hz + 1.0e-12
  mismatches = np.flatnonzero(error > tolerance)
  prefix = int(mismatches[0]) if mismatches.size else int(timestamps.size)
  if prefix < 2:
    raise ValueError(
      "head camera has no two-frame 100 Hz control-tick/30 Hz deadline prefix"
    )
  return prefix, float(np.max(error[:prefix]))


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--openpi-root", type=Path, required=True)
  known, remaining = parser.parse_known_args(argv)
  root = known.openpi_root.expanduser().resolve()
  converter = root / "examples/kaihand/convert_card_to_lerobot.py"
  if not converter.is_file():
    raise FileNotFoundError(f"OpenPI shared converter not found: {converter}")
  sys.path.insert(0, str(converter.parent))
  module_name = "kaihand_openpi_shared_converter"
  spec = importlib.util.spec_from_file_location(module_name, converter)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load OpenPI shared converter: {converter}")
  module = importlib.util.module_from_spec(spec)
  # dataclasses and other runtime annotation helpers resolve their defining
  # module through sys.modules while the module body is executing.
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  module.TASK_INSTRUCTIONS = {
    **module.TASK_INSTRUCTIONS,
    "vase-wipe": VASE_INSTRUCTION,
    "whiteboard-wipe": WHITEBOARD_INSTRUCTION,
    "sponge-grasp": SPONGE_GRASP_INSTRUCTION,
  }
  args = module._parse_args(remaining)
  # Vase intentionally records state/tactile at 500 Hz while the other shared
  # tasks use 100 Hz.  Camera frames and policy actions remain aligned at the
  # converter's 30 Hz FPS; retaining the real source rate makes its timing
  # validation and provenance accurate instead of downsampling Raw in place.
  if args.task == "vase-wipe":
    module.CONTROL_HZ = 500
  elif args.task == "sponge-grasp":
    module.common._strict_30hz_prefix = _sponge_control_tick_prefix
  install_fast_conversion(
    module, workers=args.fingerprint_workers, adapter_path=Path(__file__)
  )
  module.run(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
