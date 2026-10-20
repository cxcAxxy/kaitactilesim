"""Change only episode-level task attributes, with full HDF5 backups and audit log."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def digest(path):
  result = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      result.update(block)
  return result.hexdigest()


def verify_unchanged(original, updated, new_task):
  """Compare every dataset and attribute except the intended root task change."""
  with h5py.File(original, "r") as before, h5py.File(updated, "r") as after:
    assert after.attrs["task"] == new_task
    left, right = [], []
    before.visit(left.append)
    after.visit(right.append)
    assert left == right, "HDF5 object names changed"
    for name in ["/"] + left:
      a, b = before[name], after[name]
      assert type(a) is type(b), f"Object type changed: {name}"
      assert set(a.attrs) == set(b.attrs), f"Attribute names changed: {name}"
      for key in a.attrs:
        if name == "/" and key == "task":
          continue
        np.testing.assert_equal(a.attrs[key], b.attrs[key])
      if isinstance(a, h5py.Dataset):
        assert a.shape == b.shape and a.dtype == b.dtype, f"Dataset schema changed: {name}"
        np.testing.assert_equal(a[()], b[()])


def relabel(root, backup, new_task, expected_count=200):
  root, backup = root.resolve(strict=True), backup.resolve()
  if backup == root or root in backup.parents:
    raise ValueError("Backup must be outside the source dataset")
  paths = sorted(root.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
  if len(paths) != expected_count:
    raise ValueError(f"Expected {expected_count} episodes; found {len(paths)}")
  if not new_task.strip():
    raise ValueError("Task must not be empty")
  records = []
  for path in paths:
    if path.is_symlink():
      raise ValueError(f"Refusing symlink: {path}")
    with h5py.File(path, "r") as file:
      old_task = str(file.attrs["task"])
    records.append({"file": path.name, "old_task": old_task, "new_task": new_task,
                    "original_sha256": digest(path)})
  backup.mkdir(parents=True, exist_ok=False)
  for path, record in zip(paths, records, strict=True):
    target = backup / path.name
    shutil.copy2(path, target)
    if digest(target) != record["original_sha256"]:
      raise RuntimeError(f"Source changed during backup: {path}; no source writes performed")
  manifest = {"dataset": str(root), "utc": datetime.now(timezone.utc).isoformat(), "files": records}
  with (backup / "manifest.json").open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, ensure_ascii=False, indent=2)
  # Finish and close the manifest and ALL backups before any source mutation.
  with (backup / "changes.jsonl").open("x", encoding="utf-8") as log:
    for index, (path, record) in enumerate(zip(paths, records, strict=True), 1):
      if digest(path) != record["original_sha256"]:
        raise RuntimeError(f"Source changed since preflight: {path}; stopped, keep backups and log")
      with h5py.File(path, "r+") as file:
        # Replace rather than modify: fixed-length old strings must not truncate.
        file.attrs["task"] = new_task
        file.flush()
      verify_unchanged(backup / path.name, path, new_task)
      log.write(json.dumps({"file": path.name, "verified": True, "updated_sha256": digest(path)}) + "\n")
      log.flush()
      if index % 20 == 0 or index == len(paths):
        print(f"Relabeled and verified {index}/{len(paths)}", flush=True)
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", type=Path)
  parser.add_argument("--backup", required=True, type=Path)
  parser.add_argument("--task", required=True)
  parser.add_argument("--expected-count", type=int, default=200)
  args = parser.parse_args()
  relabel(args.dataset, args.backup, args.task, args.expected_count)
  print(f"Completed. Full backups and change log: {args.backup}")


if __name__ == "__main__":
  main()
