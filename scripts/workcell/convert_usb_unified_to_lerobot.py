#!/usr/bin/env python3
"""Adapt unified USB collections to the pinned OpenPI LeRobot converter.

The dedicated OpenPI USB converter predates ``task_collection_v2`` and expects
one flat batch whose HDF5 filenames carry globally unique episode indices.
Unified collection instead runs one isolated recorder per outer episode, so
every successful artifact is named ``usb_000000.h5`` below its own attempt
directory.  This adapter preserves the original files and their hashes while
mapping the outer collection index to the LeRobot source episode index.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from unified_lerobot_collection import discover_unified_artifacts


def _unified_pairs(module, input_dir: Path, expected_episodes: int, limit: int | None):
  summary_path = input_dir / "summary.json"
  discovered = discover_unified_artifacts(
    input_dir,
    task="usb-insert",
    expected_episodes=expected_episodes,
    limit=limit,
  )
  if discovered is None:
    # Preserve compatibility with historical flat USB batches.
    return module._legacy_discover_pairs(input_dir, expected_episodes, limit)

  summary_snapshot = module.common._snapshot_file(summary_path)
  artifacts, available = discovered
  pairs = tuple(
    module.SourcePair(item.episode_index, item.hdf5_path, item.sidecar_path)
    for item in artifacts
  )
  module.common._assert_unchanged(summary_snapshot)
  return pairs, available, summary_snapshot


def _load_converter(openpi_root: Path):
  converter = openpi_root / "examples/kaihand/convert_usb_to_lerobot.py"
  if not converter.is_file():
    raise FileNotFoundError(f"OpenPI USB converter not found: {converter}")
  sys.path.insert(0, str(converter.parent))
  name = "kaihand_openpi_usb_converter"
  spec = importlib.util.spec_from_file_location(name, converter)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load OpenPI USB converter: {converter}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--openpi-root", type=Path, required=True)
  known, remaining = parser.parse_known_args(argv)
  module = _load_converter(known.openpi_root.expanduser().resolve())

  module._legacy_discover_pairs = module._discover_pairs
  module._discover_pairs = lambda input_dir, expected_episodes, limit: _unified_pairs(
    module, input_dir, expected_episodes, limit
  )
  legacy_validate_identity = module._validate_source_identity

  def validate_identity(pair, hdf5_snapshot, sidecar_snapshot):
    internal_index = module._parse_episode_index(pair.hdf5_path)
    internal_pair = module.SourcePair(
      internal_index, pair.hdf5_path, pair.sidecar_path
    )
    source = legacy_validate_identity(
      internal_pair, hdf5_snapshot, sidecar_snapshot
    )
    return replace(source, episode_index=pair.episode_index)

  module._validate_source_identity = validate_identity
  module.run(module._parse_args(remaining))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
