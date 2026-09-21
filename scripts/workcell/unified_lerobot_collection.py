"""Read successful episodes from a unified ``task_collection_v2`` batch.

The acquisition supervisor records globally meaningful episode indices in
``summary.json`` while each isolated task recorder may reuse an internal
filename index (usually zero).  LeRobot adapters use this module to select only
published successes and preserve the outer collection index as provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnifiedArtifact:
  episode_index: int
  hdf5_path: Path
  sidecar_path: Path


def _object(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except UnicodeDecodeError as error:
    raise ValueError(f"JSON is not UTF-8: {path}") from error
  except json.JSONDecodeError as error:
    raise ValueError(f"invalid JSON: {path}") from error
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
  return value


def _regular_inside(root: Path, value: str, *, context: str) -> Path:
  unresolved = root / value
  if unresolved.is_symlink():
    raise ValueError(f"{context} must not be a symlink: {value}")
  path = unresolved.resolve(strict=True)
  if not path.is_relative_to(root):
    raise ValueError(f"{context} escapes input directory: {value}")
  if not path.is_file():
    raise ValueError(f"{context} is not a regular file: {value}")
  return path


def discover_unified_artifacts(
  input_dir: Path,
  *,
  task: str,
  expected_episodes: int,
  limit: int | None,
) -> tuple[tuple[UnifiedArtifact, ...], int] | None:
  """Return selected successful artifacts, or ``None`` for a legacy batch."""

  root = input_dir.expanduser().resolve(strict=True)
  summary_path = root / "summary.json"
  collection_path = root / "collection.json"
  if not summary_path.is_file() or not collection_path.is_file():
    return None

  before = summary_path.stat()
  collection = _object(collection_path)
  if collection.get("schema") != "task_collection_v2":
    return None
  tasks = collection.get("tasks")
  if not isinstance(tasks, list) or task not in tasks:
    raise ValueError(f"{collection_path}: task {task!r} is not in the collection contract")

  summary = _object(summary_path)
  rows = summary.get("episodes")
  if not isinstance(rows, list):
    raise ValueError(f"{summary_path}: episodes must be a list")
  successful = [
    row for row in rows
    if isinstance(row, dict)
    and row.get("task") == task
    and row.get("status") == "success"
  ]
  by_task = summary.get("by_task")
  if not isinstance(by_task, dict):
    raise ValueError(f"{summary_path}: by_task must be an object")
  task_counts = by_task.get(task, {})
  if not isinstance(task_counts, dict) or task_counts.get("success") != len(successful):
    raise ValueError(f"{summary_path}: {task} successful episode counts disagree")
  if collection.get("collection_mode") == "target-successes":
    target_met = summary.get("target_met", {})
    if not isinstance(target_met, dict) or target_met.get(task) is not True:
      raise ValueError(f"{summary_path}: {task} success target was not met")
  if expected_episodes and len(successful) != expected_episodes:
    raise ValueError(
      f"expected {expected_episodes} successful {task} episodes, "
      f"found {len(successful)}"
    )

  artifacts: list[UnifiedArtifact] = []
  seen_indices: set[int] = set()
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
    hdf5_path = _regular_inside(root, paths[0], context=f"episode {index} HDF5")
    unresolved_sidecar = hdf5_path.with_suffix(".json")
    if unresolved_sidecar.is_symlink() or not unresolved_sidecar.is_file():
      raise ValueError(f"episode {index} has no regular capture sidecar: {unresolved_sidecar}")
    sidecar_path = unresolved_sidecar.resolve(strict=True)
    if not sidecar_path.is_relative_to(root):
      raise ValueError(f"episode {index} sidecar escapes input directory")
    artifacts.append(UnifiedArtifact(index, hdf5_path, sidecar_path))

  artifacts.sort(key=lambda artifact: artifact.episode_index)
  available = len(artifacts)
  if limit is not None:
    if limit > available:
      raise ValueError(f"--limit {limit} exceeds the {available} available episodes")
    artifacts = artifacts[:limit]
  after = summary_path.stat()
  if (
    before.st_dev,
    before.st_ino,
    before.st_size,
    before.st_mtime_ns,
  ) != (
    after.st_dev,
    after.st_ino,
    after.st_size,
    after.st_mtime_ns,
  ):
    raise RuntimeError(f"summary changed while discovering sources: {summary_path}")
  return tuple(artifacts), available
