#!/usr/bin/env python3
"""Adapt unified PickPlace collections to the pinned OpenPI converter."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from fast_pi05_conversion import install_fast_conversion
from unified_lerobot_collection import discover_unified_artifacts


def _load_converter(openpi_root: Path):
  converter = openpi_root / "examples/kaihand/convert_pickplace_to_lerobot.py"
  if not converter.is_file():
    raise FileNotFoundError(f"OpenPI PickPlace converter not found: {converter}")
  sys.path.insert(0, str(converter.parent))
  name = "kaihand_openpi_pickplace_converter"
  spec = importlib.util.spec_from_file_location(name, converter)
  if spec is None or spec.loader is None:
    raise ImportError(f"cannot load OpenPI PickPlace converter: {converter}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def _metadata_snapshot(module, path: Path, sha256: str):
  before = path.stat()
  if not path.is_file():
    raise ValueError(f"source is not a regular file: {path}")
  after = path.stat()
  before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
  after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
  if before_identity != after_identity:
    raise RuntimeError(f"source changed while taking metadata snapshot: {path}")
  return module.FileSnapshot(
    path=path,
    sha256=sha256,
    size_bytes=after.st_size,
    mtime_ns=after.st_mtime_ns,
    device=after.st_dev,
    inode=after.st_ino,
  )


def _unified_sources(
  module,
  input_dir: Path,
  expected_episodes: int,
  limit: int | None,
  *,
  verify_source_hash: bool,
  fingerprint_workers: int,
):
  result = discover_unified_artifacts(
    input_dir,
    task="pick-place",
    expected_episodes=expected_episodes,
    limit=limit,
  )
  if result is None:
    return module._legacy_discover_sources(
      input_dir,
      expected_episodes,
      limit,
      verify_source_hash=verify_source_hash,
      fingerprint_workers=fingerprint_workers,
    )
  artifacts, available = result

  def snapshot(artifact):
    sidecar_snapshot = module._snapshot_file(artifact.sidecar_path)
    sidecar = module._load_json_object(artifact.sidecar_path)
    declared_sha256 = sidecar.get("sha256")
    if (
      not isinstance(declared_sha256, str)
      or module.SHA256_PATTERN.fullmatch(declared_sha256) is None
    ):
      raise ValueError(f"{artifact.sidecar_path}: invalid HDF5 SHA-256")
    if verify_source_hash:
      hdf5_snapshot = module._snapshot_file(artifact.hdf5_path)
      if hdf5_snapshot.sha256 != declared_sha256:
        raise ValueError(f"{artifact.sidecar_path}: HDF5 SHA-256 mismatch")
    else:
      hdf5_snapshot = _metadata_snapshot(module, artifact.hdf5_path, declared_sha256)
    module._assert_unchanged(sidecar_snapshot)
    return artifact, hdf5_snapshot, sidecar_snapshot

  mode = "verify SHA-256" if verify_source_hash else "trust sidecar SHA-256"
  print(f"source snapshot: {mode}; workers={fingerprint_workers}", flush=True)
  with ThreadPoolExecutor(
    max_workers=fingerprint_workers,
    thread_name_prefix="pickplace-unified-fingerprint",
  ) as executor:
    snapshots = tuple(executor.map(snapshot, artifacts))

  sources = []
  for number, (artifact, hdf5_snapshot, sidecar_snapshot) in enumerate(
    snapshots, start=1
  ):
    print(
      f"validate source [{number}/{len(snapshots)}] {artifact.hdf5_path.name}",
      flush=True,
    )
    source = module._validate_source_identity(hdf5_snapshot, sidecar_snapshot)
    sources.append(replace(source, episode_index=artifact.episode_index))
  return tuple(sources), available


def main(argv=None):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--openpi-root", type=Path, required=True)
  known, remaining = parser.parse_known_args(argv)
  module = _load_converter(known.openpi_root.expanduser().resolve())
  module._legacy_discover_sources = module._discover_sources
  module._discover_sources = lambda input_dir, expected_episodes, limit, **kwargs: (
    _unified_sources(
      module,
      input_dir,
      expected_episodes,
      limit,
      **kwargs,
    )
  )
  args = module._parse_args(remaining)
  install_fast_conversion(
    module, workers=args.fingerprint_workers, adapter_path=Path(__file__)
  )
  module.run(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
