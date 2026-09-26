#!/usr/bin/env python3
"""Adapt unified Poker collections to the pinned OpenPI LeRobot converter."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

from fast_pi05_conversion import install_fast_conversion
from unified_lerobot_collection import discover_unified_artifacts


def _load_converter(openpi_root: Path):
  converter = openpi_root / "examples/kaihand/convert_card_to_lerobot.py"
  if not converter.is_file():
    raise FileNotFoundError(f"OpenPI Poker converter not found: {converter}")
  sys.path.insert(0, str(converter.parent))
  name = "kaihand_openpi_poker_converter"
  spec = importlib.util.spec_from_file_location(name, converter)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load OpenPI Poker converter: {converter}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--openpi-root", type=Path, required=True)
  known, remaining = parser.parse_known_args(argv)
  module = _load_converter(known.openpi_root.expanduser().resolve())

  legacy_discover = module._discover_pairs

  def discover(input_dir, expected_episodes, limit, task="poker-draw"):
    if task != "poker-draw":
      return legacy_discover(input_dir, expected_episodes, limit, task)
    result = discover_unified_artifacts(
      input_dir,
      task="poker-draw",
      expected_episodes=expected_episodes,
      limit=limit,
    )
    if result is None:
      return legacy_discover(input_dir, expected_episodes, limit, task)
    artifacts, available = result
    return (
      tuple(
        module.SourcePair(
          artifact.episode_index,
          artifact.hdf5_path,
          artifact.sidecar_path,
        )
        for artifact in artifacts
      ),
      available,
    )

  module._discover_pairs = discover
  legacy_validate = module._validate_source_identity

  def validate(pair, hdf5_snapshot, sidecar_snapshot, **kwargs):
    task = kwargs.get("task", "poker-draw")
    if task != "poker-draw":
      return legacy_validate(pair, hdf5_snapshot, sidecar_snapshot, **kwargs)
    internal_pair = module.SourcePair(
      module._parse_episode_index(pair.hdf5_path),
      pair.hdf5_path,
      pair.sidecar_path,
    )
    source = legacy_validate(
      internal_pair,
      hdf5_snapshot,
      sidecar_snapshot,
      **kwargs,
    )
    return replace(source, episode_index=pair.episode_index)

  module._validate_source_identity = validate
  args = module._parse_args(remaining)
  install_fast_conversion(
    module, workers=args.fingerprint_workers, adapter_path=Path(__file__)
  )
  module.run(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
