#!/usr/bin/env python3
"""Run the pinned OpenPI shared-task converter for locally registered tasks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from fast_pi05_conversion import install_fast_conversion

WHITEBOARD_INSTRUCTION = (
  "Pick up the eraser with the right hand, wipe all ink from the tilted "
  "whiteboard with loaded sliding contact, then return and release the eraser."
)
VASE_INSTRUCTION = (
  "Pick up the sponge with the right hand, wipe all stains from the far inner "
  "wall of the vase with loaded sliding contact, then lift the sponge clear."
)


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
  }
  args = module._parse_args(remaining)
  # Vase intentionally records state/tactile at 500 Hz while the other shared
  # tasks use 100 Hz.  Camera frames and policy actions remain aligned at the
  # converter's 30 Hz FPS; retaining the real source rate makes its timing
  # validation and provenance accurate instead of downsampling Raw in place.
  if args.task == "vase-wipe":
    module.CONTROL_HZ = 500
  install_fast_conversion(module, workers=args.fingerprint_workers)
  module.run(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
