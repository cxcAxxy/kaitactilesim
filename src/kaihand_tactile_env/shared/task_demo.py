"""Resolve individually audited task-demo files without rewriting their provenance.

Only exact manifest entries are relocated. A broken mapped file never falls back
to its original NAS path. Ordinary legacy paths remain usable without a manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_DEMO_ROOT = Path(__file__).resolve().parents[3] / "task_demo"
MANIFEST_NAME = "migration_manifest.json"
SCHEMA = "kaihand-task-demo-migration-v1"
ORIGINAL_ROOT = "/nas/chenxianchi/datasets/sim/task_demo"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ENTRIES = 10_000
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _absolute(path: str | os.PathLike[str]) -> Path:
  path = Path(path)
  return path if path.is_absolute() else Path.cwd() / path


def _has_manifest(root: Path) -> bool:
  try:
    (root / MANIFEST_NAME).lstat()
  except (FileNotFoundError, NotADirectoryError):
    return False
  return True


def locate_demo_root(path: str | os.PathLike[str]) -> Path | None:
  """Find a manifest in the lexical parent chain of a replay file or directory."""
  path = _absolute(path)
  for candidate in (path, *path.parents):
    if candidate == Path(candidate.anchor):
      continue
    if _has_manifest(candidate):
      _no_symlinks(candidate)
      return candidate
  return None


def _no_symlinks(path: Path) -> None:
  if ".." in path.parts:
    raise ValueError(f"task-demo path contains '..': {path}")
  for component in (*reversed(path.parents), path):
    try:
      info = component.lstat()
    except FileNotFoundError:
      continue
    if stat.S_ISLNK(info.st_mode):
      raise ValueError(f"task-demo mapped path contains a symlink: {component}")


def _signature(info: os.stat_result) -> tuple[int, ...]:
  return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
  result = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(f"duplicate key in task-demo migration manifest: {key}")
    result[key] = value
  return result


def _reject_constant(value: str) -> None:
  raise ValueError(f"non-finite value in task-demo migration manifest: {value}")


def _digest(value: object, label: str) -> str:
  if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
    raise ValueError(f"{label} must be a 64-hex SHA-256")
  return value.lower()


def _stream_sha256(stream) -> str:
  digest = hashlib.sha256()
  for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
    digest.update(block)
  return digest.hexdigest()


def _relative(value: object) -> str:
  if (
    not isinstance(value, str) or not value or "\x00" in value or "\\" in value
    or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
    or any(part in {"", ".", ".."} for part in value.split("/"))
    or value == MANIFEST_NAME
  ):
    raise ValueError(f"invalid relative path in task-demo migration manifest: {value!r}")
  return value


def _original(value: object) -> str:
  if (
    not isinstance(value, str) or not value.startswith("/") or value == "/"
    or "\x00" in value or "\\" in value
    or any(part in {"", ".", ".."} for part in value[1:].split("/"))
  ):
    raise ValueError(f"invalid original_path in task-demo migration manifest: {value!r}")
  return value


def _manifest(root: Path) -> tuple[dict, dict, tuple[int, ...]]:
  if root == Path(root.anchor):
    raise ValueError("task-demo root must not be a filesystem root")
  _no_symlinks(root)
  manifest = root / MANIFEST_NAME
  _no_symlinks(manifest)
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
  with os.fdopen(os.open(manifest, flags), "rb") as stream:
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_MANIFEST_BYTES:
      raise ValueError("task-demo migration manifest must be a bounded regular JSON file")
    raw = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
      raise ValueError("task-demo migration manifest exceeds size limit")
    if _signature(before) != _signature(os.fstat(stream.fileno())):
      raise ValueError("task-demo migration manifest changed during read")
  signature = _signature(before)
  if signature != _signature(manifest.lstat()):
    raise ValueError("task-demo migration manifest changed during read")
  try:
    payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
  except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
    raise ValueError("invalid task-demo migration manifest JSON") from error
  if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
    raise ValueError("task-demo migration manifest schema mismatch")
  original_root = _original(payload.get("original_root"))
  if original_root == str(root):
    raise ValueError("task-demo original_root must differ from its local root")
  tables = [payload.get("files"), payload.get("raw_sources")]
  if any(not isinstance(table, list) for table in tables):
    raise ValueError("task-demo migration manifest files/raw_sources must be lists")
  if sum(map(len, tables)) > _MAX_ENTRIES:
    raise ValueError("task-demo migration manifest exceeds entry limit")
  by_original, by_relative = {}, {}
  for table_index, table in enumerate(tables):
    seen_original, seen_relative = set(), set()
    for row in table:
      if not isinstance(row, dict):
        raise ValueError("task-demo migration manifest entry must be an object")
      relative = _relative(row.get("path"))
      original = _original(row.get("original_path"))
      if original in {original_root, str(root)}:
        raise ValueError("task-demo mapping must name a file, not a demo root")
      if table_index == 0 and original != f"{original_root}/{relative}":
        raise ValueError("task-demo files entry does not match its original_root/path")
      size = row.get("size")
      if type(size) is not int or size < 0:
        raise ValueError("task-demo migration manifest size must be a nonnegative integer")
      entry = (relative, original, _digest(row.get("sha256"), "manifest sha256"), size)
      if original in seen_original or relative in seen_relative:
        raise ValueError("duplicate task-demo migration manifest entry")
      seen_original.add(original)
      seen_relative.add(relative)
      if (
        original in by_original and by_original[original] != entry
        or relative in by_relative and by_relative[relative] != entry
      ):
        raise ValueError("conflicting task-demo migration manifest mapping")
      by_original[original] = by_relative[relative] = entry
  return by_original, by_relative, signature


def _mapped_entry(path: Path, root: Path, by_original: dict, by_relative: dict):
  original = by_original.get(str(path))
  local = None
  absolute = _absolute(path)
  try:
    relative = absolute.relative_to(root).as_posix()
  except ValueError:
    relative = None
  if relative is not None:
    local = by_relative.get(relative)
  if not path.is_absolute() and local is None:
    local = by_relative.get(path.as_posix())
  if original is not None and local is not None and original != local:
    raise ValueError("ambiguous task-demo original/local mapping")
  return original if original is not None else local


def _checked_target(root: Path, entry: tuple, expected_sha256: str | None, verify_hash: bool) -> Path:
  relative, _, declared_hash, size = entry
  if expected_sha256 is not None and _digest(expected_sha256, "expected_sha256") != declared_hash:
    raise ValueError("task-demo source SHA-256 disagrees with migration manifest")
  target = root / relative
  _no_symlinks(target)
  info = target.lstat()
  if not stat.S_ISREG(info.st_mode) or info.st_size != size:
    raise ValueError(f"task-demo mapped file type/size mismatch: {target}")
  if verify_hash:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(target, flags), "rb") as stream:
      if _signature(info) != _signature(os.fstat(stream.fileno())):
        raise ValueError("task-demo mapped file changed before hashing")
      actual_hash = _stream_sha256(stream)
      if _signature(info) != _signature(os.fstat(stream.fileno())):
        raise ValueError("task-demo mapped file changed during hashing")
    if _signature(info) != _signature(target.lstat()):
      raise ValueError("task-demo mapped file changed during hashing")
    if actual_hash != declared_hash:
      raise ValueError(f"task-demo mapped file SHA-256 mismatch: {target}")
  return target


def resolve_demo_path(
  path: str | os.PathLike[str], *, demo_root: str | os.PathLike[str] | None = None,
  expected_sha256: str | None = None, verify_hash: bool = False,
) -> Path:
  """Resolve an exact migrated file, optionally hashing its complete contents.

  Size and declared/expected SHA identities are checked for every mapped file.
  Without a mapping, the original Path is returned unchanged, including legacy
  fixture checksum declarations. No content hashing is implicit.
  """
  path = Path(path)
  root = _absolute(DEFAULT_DEMO_ROOT if demo_root is None else demo_root)
  if root == Path(root.anchor):
    raise ValueError("task-demo root must not be a filesystem root")
  roots = [root]
  for candidate in roots:
    if _has_manifest(candidate):
      by_original, by_relative, signature = _manifest(candidate)
      entry = _mapped_entry(path, candidate, by_original, by_relative)
      if entry is not None:
        target = _checked_target(candidate, entry, expected_sha256, verify_hash)
        if signature != _signature((candidate / MANIFEST_NAME).lstat()):
          raise ValueError("task-demo migration manifest changed during resolution")
        _no_symlinks(target)
        return target
    if demo_root is None and len(roots) == 1:
      located = locate_demo_root(path)
      if located is not None and located != root:
        roots.append(located)
  return path
