from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from kaihand_tactile_env.shared.pi05_deployment_integrity import verify_pi05_deployment


def _rows(root: Path, paths: list[Path]) -> tuple[list[dict], str]:
  rows = [
    {
      "path": path.relative_to(root).as_posix(),
      "size": path.stat().st_size,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in paths
  ]
  encoded = json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return rows, hashlib.sha256(encoded).hexdigest()


def _manifest(tmp_path: Path) -> tuple[dict, Path, Path]:
  checkpoint = tmp_path / "checkpoint"
  (checkpoint / "params").mkdir(parents=True)
  (checkpoint / "assets").mkdir()
  committed = checkpoint / "_CHECKPOINT_METADATA"
  weights = checkpoint / "params/weights.bin"
  stats = checkpoint / "assets/stats.json"
  committed.write_bytes(b"metadata")
  weights.write_bytes(b"weight-A")
  stats.write_bytes(b"stats")
  checkpoint_rows, checkpoint_hash = _rows(checkpoint, [
    committed, stats, weights,
  ])
  project = tmp_path / "openpi"
  source = project / "src/openpi/training/config.py"
  source.parent.mkdir(parents=True)
  source.write_bytes(b"config-A")
  source_rows, source_hash = _rows(project, [source])
  return {
    "schema": "usb_pi05_deployment_v1",
    "checkpoint_path": str(checkpoint),
    "checkpoint_files": checkpoint_rows,
    "checkpoint_sha256": checkpoint_hash,
    "model_project_path": str(project),
    "openpi_source_files": source_rows,
    "openpi_source_sha256": source_hash,
  }, weights, source


def test_pi05_deployment_rejects_same_size_checkpoint_change(tmp_path: Path) -> None:
  manifest, weights, _ = _manifest(tmp_path)
  verify_pi05_deployment(manifest)
  weights.write_bytes(b"weight-B")
  with pytest.raises(RuntimeError, match="checkpoint content SHA-256 mismatch"):
    verify_pi05_deployment(manifest)


def test_pi05_deployment_rejects_source_change_and_extra_file(tmp_path: Path) -> None:
  manifest, _, source = _manifest(tmp_path)
  source.write_bytes(b"config-B")
  with pytest.raises(RuntimeError, match="OpenPI source content SHA-256 mismatch"):
    verify_pi05_deployment(manifest)
  source.write_bytes(b"config-A")
  (Path(manifest["checkpoint_path"]) / "params/extra.bin").write_bytes(b"new")
  with pytest.raises(RuntimeError, match="checkpoint file inventory changed"):
    verify_pi05_deployment(manifest)
