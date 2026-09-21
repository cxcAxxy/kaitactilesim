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
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys


def _object(path: Path) -> dict:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _inside(root: Path, value: str, *, context: str) -> Path:
  path = (root / value).resolve(strict=True)
  if not path.is_relative_to(root):
    raise ValueError(f"{context} escapes input directory: {value}")
  return path


def _unified_pairs(module, input_dir: Path, expected_episodes: int, limit: int | None):
  summary_path = input_dir / "summary.json"
  collection_path = input_dir / "collection.json"
  if not summary_path.is_file() or not collection_path.is_file():
    # Preserve compatibility with historical flat USB batches.
    return module._legacy_discover_pairs(input_dir, expected_episodes, limit)

  collection = _object(collection_path)
  if (
    collection.get("schema") != "task_collection_v2"
    or collection.get("tasks") != ["usb-insert"]
    or collection.get("collection_mode") != "target-successes"
  ):
    return module._legacy_discover_pairs(input_dir, expected_episodes, limit)

  summary_snapshot = module.common._snapshot_file(summary_path)
  summary = _object(summary_path)
  rows = summary.get("episodes")
  if not isinstance(rows, list):
    raise ValueError(f"{summary_path}: episodes must be a list")
  successful = [row for row in rows if isinstance(row, dict) and row.get("status") == "success"]
  declared_successes = summary.get("success_count")
  task_counts = summary.get("by_task", {}).get("usb-insert", {})
  if declared_successes != len(successful) or task_counts.get("success") != len(successful):
    raise ValueError(f"{summary_path}: successful episode counts disagree")
  if summary.get("target_met", {}).get("usb-insert") is not True:
    raise ValueError(f"{summary_path}: USB success target was not met")
  if expected_episodes and len(successful) != expected_episodes:
    raise ValueError(
      f"expected {expected_episodes} successful episodes, found {len(successful)}"
    )

  pairs = []
  seen_indices = set()
  for row_number, row in enumerate(successful):
    index = row.get("episode_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
      raise ValueError(f"{summary_path}: success row {row_number} has invalid episode_index")
    if index in seen_indices:
      raise ValueError(f"{summary_path}: duplicate successful episode index {index}")
    seen_indices.add(index)
    paths = row.get("hdf5")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
      raise ValueError(f"{summary_path}: episode {index} must name exactly one HDF5")
    hdf5_path = _inside(input_dir, paths[0], context=f"episode {index} HDF5")
    sidecar_path = hdf5_path.with_suffix(".json").resolve(strict=True)
    if not sidecar_path.is_relative_to(input_dir):
      raise ValueError(f"episode {index} sidecar escapes input directory")
    pairs.append(module.SourcePair(index, hdf5_path, sidecar_path))

  pairs.sort(key=lambda pair: pair.episode_index)
  available = len(pairs)
  if limit is not None:
    if limit > available:
      raise ValueError(f"--limit {limit} exceeds the {available} available episodes")
    pairs = pairs[:limit]
  module.common._assert_unchanged(summary_snapshot)
  return tuple(pairs), available, summary_snapshot


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
