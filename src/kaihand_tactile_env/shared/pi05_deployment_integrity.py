"""Verify frozen pi0.5 inference bytes before a policy server starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(16 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _canonical_hash(rows: list[dict]) -> str:
  encoded = json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def _verify_rows(root: Path, rows: object, aggregate: object, label: str) -> set[str]:
  if not isinstance(rows, list) or not rows:
    raise RuntimeError(f"{label} file snapshot must be a nonempty list")
  root = root.expanduser().resolve(strict=True)
  observed = set()
  for row in rows:
    if not isinstance(row, dict):
      raise RuntimeError(f"{label} file snapshot entries must be objects")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative:
      raise RuntimeError(f"{label} file path must be a relative string")
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts or name.as_posix() != relative:
      raise RuntimeError(f"unsafe {label} file path: {relative!r}")
    if relative in observed:
      raise RuntimeError(f"duplicate {label} file path: {relative!r}")
    observed.add(relative)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
      raise RuntimeError(f"{label} file escapes snapshot root: {relative!r}")
    if path.stat().st_size != row.get("size") or _sha256_file(path) != row.get("sha256"):
      raise RuntimeError(f"{label} content SHA-256 mismatch: {path}")
  if _canonical_hash(rows) != aggregate:
    raise RuntimeError(f"{label} aggregate SHA-256 mismatch")
  return observed


def verify_pi05_deployment(manifest: dict, *, evaluation_root: Path | None = None) -> None:
  """Reject changed checkpoint or source code before advertising an identity."""
  checkpoint = Path(manifest["checkpoint_path"])
  observed = _verify_rows(
    checkpoint, manifest.get("checkpoint_files"),
    manifest.get("checkpoint_sha256"), "checkpoint",
  )
  actual = {"_CHECKPOINT_METADATA"}
  for directory in (checkpoint / "params", checkpoint / "assets"):
    if not directory.is_dir():
      raise RuntimeError(f"missing checkpoint directory: {directory}")
    actual.update(
      path.relative_to(checkpoint).as_posix()
      for path in directory.rglob("*") if path.is_file()
    )
  if observed != actual:
    raise RuntimeError(
      f"checkpoint file inventory changed: missing={sorted(actual - observed)}, "
      f"unexpected={sorted(observed - actual)}"
    )

  project = Path(manifest["model_project_path"])
  if manifest.get("schema") == "pickplace_pi05_deployment_v1":
    source = project / "src/openpi/training/config.py"
    if _sha256_file(source) != manifest.get("model_source_sha256"):
      raise RuntimeError(f"OpenPI model source SHA-256 mismatch: {source}")
  else:
    _verify_rows(
      project, manifest.get("openpi_source_files"),
      manifest.get("openpi_source_sha256"), "OpenPI source",
    )
  if manifest.get("schema") == "shared_task_pi05_deployment_v1":
    if evaluation_root is None:
      raise RuntimeError("shared-task deployment requires evaluation_root")
    _verify_rows(
      evaluation_root, manifest.get("evaluation_source_files"),
      manifest.get("evaluation_source_sha256"), "evaluation source",
    )
