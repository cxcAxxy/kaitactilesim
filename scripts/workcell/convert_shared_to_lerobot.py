#!/usr/bin/env python3
"""Run the pinned OpenPI shared-task converter for locally registered tasks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

WHITEBOARD_INSTRUCTION = (
  "Pick up the eraser with the right hand, wipe all ink from the tilted "
  "whiteboard with loaded sliding contact, then return and release the eraser."
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
    "whiteboard-wipe": WHITEBOARD_INSTRUCTION,
  }
  module.run(module._parse_args(remaining))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
