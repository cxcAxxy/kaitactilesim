"""Content-addressed identity checks for the standalone LingBot deployment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/workcell"))

from prepare_pickplace_lingbot_vla2_deployment import (  # noqa: E402
  canonical_sha256,
  checkpoint_identity,
  main,
)


def test_checkpoint_identity_includes_every_file_and_content(tmp_path: Path) -> None:
  checkpoint = tmp_path / "hf_ckpt"
  checkpoint.mkdir()
  (checkpoint / "config.json").write_text("{}", encoding="utf-8")
  (checkpoint / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
  (checkpoint / "model-00001.safetensors").write_bytes(b"weights")
  digest, rows = checkpoint_identity(checkpoint, hash_workers=2)
  assert digest == canonical_sha256(rows)
  assert [row["path"] for row in rows] == sorted(row["path"] for row in rows)
  weights = next(row for row in rows if row["path"] == "model-00001.safetensors")
  assert weights["sha256"] == hashlib.sha256(b"weights").hexdigest()
  assert weights["size"] == len(b"weights")

  (checkpoint / "model-00001.safetensors").write_bytes(b"changed")
  mutated, _ = checkpoint_identity(checkpoint, hash_workers=1)
  assert mutated != digest


def test_checkpoint_identity_rejects_symlink_and_invalid_worker(tmp_path: Path) -> None:
  checkpoint = tmp_path / "hf_ckpt"
  checkpoint.mkdir()
  (checkpoint / "config.json").write_text("{}", encoding="utf-8")
  (checkpoint / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
  with pytest.raises(ValueError, match="hash_workers"):
    checkpoint_identity(checkpoint, hash_workers=0)
  (checkpoint / "escape").symlink_to(tmp_path)
  with pytest.raises(ValueError, match="symlink"):
    checkpoint_identity(checkpoint)


def test_prepare_refuses_to_overwrite_existing_deployment(tmp_path: Path) -> None:
  output = tmp_path / "existing"
  output.mkdir()
  sentinel = output / "deployment_manifest.json"
  sentinel.write_text(json.dumps({"owner": "user"}), encoding="utf-8")
  with pytest.raises(FileExistsError, match="refusing to overwrite"):
    main([
      "--checkpoint", str(tmp_path / "missing"),
      "--output-dir", str(output),
    ])
  assert json.loads(sentinel.read_text(encoding="utf-8")) == {"owner": "user"}
