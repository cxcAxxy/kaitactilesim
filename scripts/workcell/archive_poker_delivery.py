#!/usr/bin/env python3
"""Serial level-1 gzip archives, verified byte-for-byte without extraction."""

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from poker_delivery_common import save, sha256


def verify_embedded_integrity(root, inventory):
  """Reuse the freshly hashed inventory to check stored release manifests."""
  checked = {}
  integrity_path = root / "transfer_integrity.json"
  if integrity_path.exists():
    integrity = json.loads(integrity_path.read_text())
    records = integrity["files"]
    paths = [root.name + "/" + row["path"] for row in records]
    exempt = {root.name + "/selector_manifest.json", root.name + "/transfer_integrity.json"}
    if len(paths) != len(set(paths)) or set(paths) != set(inventory) - exempt:
      raise ValueError("transfer integrity coverage mismatch")
    for row, path in zip(records, paths, strict=True):
      if inventory[path] != {"size_bytes": row["bytes"], "sha256": row["sha256"]}:
        raise ValueError(f"transfer integrity checksum mismatch: {path}")
    selector = json.loads((root / "selector_manifest.json").read_text())
    for row in selector["records"]:
      relative = Path(row["training_data_path"]).with_name("rgb.png")
      expected_path = root.name + "/" + relative.as_posix()
      if (row["rgb_path"] != str(root / relative)
          or inventory.get(expected_path) != {"size_bytes": row["rgb_bytes"], "sha256": row["rgb_sha256"]}):
        raise ValueError("published selector path/checksum mismatch")
    checked["egotouch_transfer_and_selector"] = True
  manifest_path = root / "dataset_manifest.json"
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    records = [row for split in manifest["splits"].values() for row in split["shards"]]
    records += manifest.get("tactile_sidecars", [])
    for row in records:
      if inventory.get(root.name + "/" + row["path"]) != {"size_bytes": row["size_bytes"], "sha256": row["sha256"]}:
        raise ValueError("EgoSteer shard/sidecar checksum mismatch")
    checked["egosteer_shards_and_tactile"] = True
  return checked


def archive_release(root):
  root = Path(root).resolve()
  archive_path = root.with_name(root.name + ".tar.gz")
  digest_path = archive_path.with_name(archive_path.name + ".sha256")
  receipt_path = archive_path.with_name(archive_path.name + ".verification.json")
  for path in (archive_path, digest_path, receipt_path):
    if path.exists():
      raise FileExistsError(path)
  if not root.is_dir() or root.is_symlink():
    raise ValueError("release must be a regular directory")
  started = time.monotonic()
  inventory = {}
  paths = sorted(root.rglob("*"))
  for index, path in enumerate(paths):
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
      raise ValueError(f"non-regular release member: {path}")
    if not path.is_file():
      continue
    relative = path.relative_to(root.parent).as_posix()
    inventory[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    if index % 10000 == 0:
      print(f"{root.name}: inventory {index}/{len(paths)}", flush=True)
  embedded = verify_embedded_integrity(root, inventory)
  fd, temporary_name = tempfile.mkstemp(prefix=f".{archive_path.name}.partial-", dir=root.parent)
  os.close(fd)
  temporary = Path(temporary_name)
  print(f"{root.name}: compressing {len(inventory)} files with single-worker gzip level 1", flush=True)
  # Do not delete partial work on failure; it may help diagnose disk/I/O errors.
  with tarfile.open(temporary, "w:gz", compresslevel=1, format=tarfile.PAX_FORMAT) as output:
    output.add(root, arcname=root.name, recursive=False)
    for index, path in enumerate(paths):
      output.add(path, arcname=path.relative_to(root.parent).as_posix(), recursive=False)
      if index % 20000 == 0:
        print(f"{root.name}: archived {index}/{len(paths)} entries", flush=True)
  seen = set()
  print(f"{root.name}: verifying compressed contents", flush=True)
  with tarfile.open(temporary, "r|gz") as archive:
    for member in archive:
      name = PurePosixPath(member.name)
      if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != root.name:
        raise ValueError(f"unsafe archive path: {name}")
      if member.isdir():
        continue
      if not member.isreg() or member.name in seen or member.name not in inventory:
        raise ValueError(f"unexpected archive member: {name}")
      digest = hashlib.sha256()
      stream = archive.extractfile(member)
      for chunk in iter(lambda stream=stream: stream.read(4 * 1024**2), b""):
        digest.update(chunk)
      expected = inventory[member.name]
      if member.size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"archive byte mismatch: {name}")
      seen.add(member.name)
      if len(seen) % 20000 == 0:
        print(f"{root.name}: verified {len(seen)}/{len(inventory)} files", flush=True)
  if seen != set(inventory):
    raise ValueError("missing archive files")
  if archive_path.exists():
    raise FileExistsError(archive_path)
  digest = sha256(temporary)
  os.rename(temporary, archive_path)
  with digest_path.open("x", encoding="ascii") as stream:
    stream.write(f"{digest}  {archive_path.name}\n")
  save(receipt_path, {"valid": True, "archive": archive_path.name, "sha256": digest,
                      "size_bytes": archive_path.stat().st_size, "file_count": len(inventory),
                      "elapsed_seconds": time.monotonic() - started,
                      "embedded_integrity_verified": embedded,
                      "verification": "every compressed regular file matched original size and SHA256; no extraction",
                      "files": inventory})
  print(f"Complete {archive_path}: {archive_path.stat().st_size / 1024**3:.3f} GiB; all {len(inventory)} files verified", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("directories", nargs="+", type=Path)
  args = parser.parse_args()
  for root in args.directories:
    archive_release(root)


if __name__ == "__main__":
  main()
