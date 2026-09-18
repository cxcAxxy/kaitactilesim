import json
from pathlib import Path

import pytest


def test_reindexed_split_preserves_old_assignment_and_uses_original_new_index(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from poker_delivery_common import preserved_split
  assert preserved_split({"previous_split": "val", "source_episode_index": 3}) == "val"
  assert preserved_split({"previous_split": "train", "source_episode_index": 4}) == "train"
  assert preserved_split({"previous_split": None, "source_episode_index": 114, "episode_index": 148}) == "val"
  assert preserved_split({"previous_split": None, "source_episode_index": 115, "episode_index": 149}) == "train"
  with pytest.raises(ValueError):
    preserved_split({"previous_split": "unknown", "source_episode_index": 4})


def test_duplicate_cleanup_requires_published_hardlinks(tmp_path, monkeypatch):
  import os
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from finalize_poker_dual_delivery import remove_linked_intermediates
  processing, output = tmp_path / "processing", tmp_path / "output"
  session = processing / "tict_sessions/session"
  for branch in ("production", "tict_sidecars"):
    (session / branch).mkdir(parents=True)
    (output / branch).mkdir(parents=True)
    (session / branch / "data.bin").write_bytes(b"test")
  with pytest.raises(ValueError, match="not preserved"):
    remove_linked_intermediates(processing, output, ["session"])
  assert session.exists()
  for branch in ("production", "tict_sidecars"):
    os.link(session / branch / "data.bin", output / branch / "data.bin")
  report = remove_linked_intermediates(processing, output, ["session"])
  assert report["preserved_payload_files"] == 2
  assert not (processing / "tict_sessions").exists()
  assert (output / "production/data.bin").read_bytes() == b"test"


def test_archive_verifies_bytes_and_refuses_overwrite(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from archive_poker_delivery import archive_release
  from poker_delivery_common import sha256

  root = tmp_path / "release"
  (root / "nested").mkdir(parents=True)
  (root / "a.json").write_text('{"valid": true}\n')
  (root / "nested/b.bin").write_bytes(bytes(range(256)))
  archive_release(root)
  archive = tmp_path / "release.tar.gz"
  receipt = json.loads((tmp_path / "release.tar.gz.verification.json").read_text())
  assert receipt["valid"] and receipt["file_count"] == 2
  assert receipt["sha256"] == sha256(archive)
  assert receipt["files"]["release/nested/b.bin"]["sha256"] == sha256(root / "nested/b.bin")
  with pytest.raises(FileExistsError):
    archive_release(root)


def test_archive_refuses_symlinks(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from archive_poker_delivery import archive_release

  root = tmp_path / "release"
  root.mkdir()
  (tmp_path / "outside.txt").write_text("must not be packaged")
  (root / "link").symlink_to(tmp_path / "outside.txt")
  with pytest.raises(ValueError, match="non-regular"):
    archive_release(root)


def test_quality_relative_path_and_safe_resume(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from poker_delivery_common import attach_quality

  monkeypatch.chdir(tmp_path)
  Path("release").mkdir()
  Path("report.json").write_text('{"valid": true}\n')
  Path("release/transfer_integrity.json").write_text('{"files": []}')
  snapshot = {"episodes": [{"episode_index": 3, "cleaning_report": str(tmp_path / "report.json")}]}
  attach_quality(Path("release"), snapshot)
  attach_quality(Path("release"), snapshot)
  integrity = json.loads(Path("release/transfer_integrity.json").read_text())
  assert len(integrity["files"]) == 3
  assert len({record["path"] for record in integrity["files"]}) == 3
  with pytest.raises(ValueError, match="different source selection"):
    attach_quality(Path("release"), {"episodes": []})


def test_embedded_selector_must_point_to_published_root(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from archive_poker_delivery import verify_embedded_integrity

  root = tmp_path / "release"
  root.mkdir()
  relative = "production/session/00000/rgb.png"
  inventory = {"release/" + relative: {"size_bytes": 3, "sha256": "abc"},
               "release/selector_manifest.json": {}, "release/transfer_integrity.json": {}}
  (root / "transfer_integrity.json").write_text(json.dumps({"files": [
    {"path": relative, "bytes": 3, "sha256": "abc"}]}))
  row = {"training_data_path": "production/session/00000/training_data.json",
         "rgb_path": str(root / relative), "rgb_bytes": 3, "rgb_sha256": "abc"}
  (root / "selector_manifest.json").write_text(json.dumps({"records": [row]}))
  assert verify_embedded_integrity(root, inventory) == {"egotouch_transfer_and_selector": True}
  row["rgb_path"] = "/old/machine/rgb.png"
  (root / "selector_manifest.json").write_text(json.dumps({"records": [row]}))
  with pytest.raises(ValueError, match="selector"):
    verify_embedded_integrity(root, inventory)
