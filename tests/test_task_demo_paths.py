import hashlib
import json
from pathlib import Path

import pytest

from kaihand_tactile_env.shared import task_demo


def make_demo(tmp_path, *, original_root=task_demo.ORIGINAL_ROOT):
  root = tmp_path / "project/task_demo"
  root.mkdir(parents=True)
  relative = "_metadata/example/replay.json"
  target = root / relative
  target.parent.mkdir(parents=True)
  target.write_bytes(b"original replay bytes")
  digest = hashlib.sha256(target.read_bytes()).hexdigest()
  row = {
    "path": relative, "original_path": f"{original_root}/{relative}",
    "sha256": digest, "size": target.stat().st_size,
  }
  payload = {
    "schema": task_demo.SCHEMA, "original_root": original_root,
    "files": [row], "raw_sources": [],
  }
  manifest = root / task_demo.MANIFEST_NAME
  manifest.write_text(json.dumps(payload), encoding="utf-8")
  return root, target, manifest, payload


def test_mapped_original_prefers_local_without_accessing_nas(tmp_path, monkeypatch):
  root, target, _, payload = make_demo(tmp_path)
  row = payload["files"][0]
  original_lstat = Path.lstat

  def no_nas_access(path, *args, **kwargs):
    if str(path).startswith("/nas/"):
      raise AssertionError("resolution touched the unavailable original NAS")
    return original_lstat(path, *args, **kwargs)

  monkeypatch.setattr(Path, "lstat", no_nas_access)
  monkeypatch.setattr(task_demo, "DEFAULT_DEMO_ROOT", root)
  assert task_demo.resolve_demo_path(
    row["original_path"], expected_sha256=row["sha256"],
  ) == target


def test_absolute_and_root_relative_mapped_paths_use_the_same_identity(tmp_path):
  root, target, _, payload = make_demo(tmp_path)
  row = payload["files"][0]
  for source in (target, row["path"]):
    assert task_demo.resolve_demo_path(
      source, demo_root=root, expected_sha256=row["sha256"].upper(),
    ) == target


def test_arbitrary_archive_original_root_is_supported(tmp_path):
  root, target, _, payload = make_demo(tmp_path, original_root=str(tmp_path / "archive"))
  row = payload["files"][0]
  assert task_demo.resolve_demo_path(row["original_path"], demo_root=root) == target


def test_locate_root_and_implicit_nested_fixture_resolution(tmp_path):
  root, target, _, _ = make_demo(tmp_path)
  assert task_demo.locate_demo_root(target) == root
  assert task_demo.locate_demo_root(target.parent / "not-created-yet.json") == root
  assert task_demo.locate_demo_root(root) == root
  assert task_demo.locate_demo_root(tmp_path / "unrelated.json") is None
  assert task_demo.resolve_demo_path(target) == target


def test_unmigrated_fixtures_keep_paths_and_arbitrary_checksum_declarations(tmp_path):
  root, _, _, _ = make_demo(tmp_path)
  fixture = tmp_path / "old-fixture.h5"
  fixture.write_bytes(b"legacy fixture")
  for demo_root in (root, tmp_path / "without-manifest"):
    assert task_demo.resolve_demo_path(
      fixture, demo_root=demo_root, expected_sha256="declared", verify_hash=True,
    ) == fixture
  relative = Path("unmapped-legacy-fixture.h5")
  assert task_demo.resolve_demo_path(relative, demo_root=root) == relative


@pytest.mark.parametrize("expected", ["declared", "0" * 64, "a" * 63, 12])
def test_mapped_expected_checksum_must_be_valid_and_match(tmp_path, expected):
  root, _, _, payload = make_demo(tmp_path)
  with pytest.raises(ValueError, match="SHA-256"):
    task_demo.resolve_demo_path(
      payload["files"][0]["original_path"], demo_root=root, expected_sha256=expected,
    )


def test_missing_or_corrupt_local_mapping_never_falls_back_to_existing_source(tmp_path):
  root, target, manifest, payload = make_demo(tmp_path)
  original = tmp_path / "external-source.h5"
  original.write_bytes(target.read_bytes())
  row = dict(payload["files"][0], original_path=str(original))
  payload["files"], payload["raw_sources"] = [], [row]
  manifest.write_text(json.dumps(payload))
  target.unlink()
  with pytest.raises(FileNotFoundError):
    task_demo.resolve_demo_path(original, demo_root=root)
  target.write_bytes(b"wrong-size")
  with pytest.raises(ValueError, match="size"):
    task_demo.resolve_demo_path(original, demo_root=root)


def test_hashing_is_explicit_and_detects_same_size_corruption(tmp_path, monkeypatch):
  root, target, _, payload = make_demo(tmp_path)
  row = payload["files"][0]
  assert task_demo.resolve_demo_path(
    row["original_path"], demo_root=root, verify_hash=True,
  ) == target
  target.write_bytes(b"x" * row["size"])
  real_hash = task_demo._stream_sha256

  def unexpected_hash(_stream):
    raise AssertionError("normal resolution must not read/hash the payload")

  monkeypatch.setattr(task_demo, "_stream_sha256", unexpected_hash)
  assert task_demo.resolve_demo_path(row["original_path"], demo_root=root) == target
  monkeypatch.setattr(task_demo, "_stream_sha256", real_hash)
  with pytest.raises(ValueError, match="SHA-256 mismatch"):
    task_demo.resolve_demo_path(row["original_path"], demo_root=root, verify_hash=True)


def test_manifest_is_reloaded_and_expected_source_hash_blocks_rebinding(tmp_path):
  root, target, manifest, payload = make_demo(tmp_path)
  row = payload["files"][0]
  expected = row["sha256"]
  assert task_demo.resolve_demo_path(
    row["original_path"], demo_root=root, expected_sha256=expected,
  ) == target
  row["sha256"] = "0" * 64
  manifest.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="disagrees"):
    task_demo.resolve_demo_path(row["original_path"], demo_root=root, expected_sha256=expected)


def test_manifest_rewrite_during_hashing_is_rejected(tmp_path, monkeypatch):
  root, _, manifest, payload = make_demo(tmp_path)
  real_hash = task_demo._stream_sha256

  def hash_then_rewrite(stream):
    digest = real_hash(stream)
    manifest.write_text(manifest.read_text() + " ")
    return digest

  monkeypatch.setattr(task_demo, "_stream_sha256", hash_then_rewrite)
  with pytest.raises(ValueError, match="manifest changed"):
    task_demo.resolve_demo_path(
      payload["files"][0]["original_path"], demo_root=root, verify_hash=True,
    )


@pytest.mark.parametrize("relative", [
  "../escape.h5", "/tmp/escape.h5", "a/../escape.h5", "a/./file.h5",
  "", ".", "a//file.h5", "a/", "C:/escape.h5", "a\\file.h5",
  task_demo.MANIFEST_NAME,
])
def test_manifest_cannot_select_traversal_absolute_or_manifest_targets(tmp_path, relative):
  root, _, manifest, payload = make_demo(tmp_path)
  payload["files"][0]["path"] = relative
  manifest.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="relative path"):
    task_demo.resolve_demo_path("unrelated", demo_root=root)


@pytest.mark.parametrize("where", ["file", "parent", "manifest", "root"])
def test_symlinks_are_rejected_even_when_a_source_would_be_available(tmp_path, where):
  root, target, manifest, payload = make_demo(tmp_path)
  original = payload["files"][0]["original_path"]
  if where == "file":
    outside = tmp_path / "outside.h5"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
  elif where == "parent":
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    target.rename(outside / target.name)
    target.parent.rmdir()
    target.parent.symlink_to(outside, target_is_directory=True)
  elif where == "manifest":
    outside = tmp_path / "outside-manifest.json"
    manifest.rename(outside)
    manifest.symlink_to(outside)
  else:
    alias = tmp_path / "root-alias"
    alias.symlink_to(root, target_is_directory=True)
    root = alias
  with pytest.raises(ValueError, match="symlink"):
    task_demo.resolve_demo_path(original, demo_root=root)


@pytest.mark.parametrize("conflict", ["original", "target", "checksum", "duplicate"])
def test_mapping_must_be_unambiguous(tmp_path, conflict):
  root, _, manifest, payload = make_demo(tmp_path)
  row = dict(payload["files"][0])
  if conflict == "original":
    row["path"] = "other.h5"
  elif conflict == "target":
    row["original_path"] = "/external/other.h5"
  elif conflict == "checksum":
    row["sha256"] = "0" * 64
  if conflict == "duplicate":
    payload["files"].append(row)
  else:
    payload["raw_sources"].append(row)
  manifest.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="conflicting|duplicate"):
    task_demo.resolve_demo_path("unrelated", demo_root=root)


def test_identical_files_and_raw_sources_row_can_share_one_mapping(tmp_path):
  root, target, manifest, payload = make_demo(tmp_path)
  payload["raw_sources"].append(dict(payload["files"][0]))
  manifest.write_text(json.dumps(payload))
  assert task_demo.resolve_demo_path(target, demo_root=root) == target


@pytest.mark.parametrize("bad_root", ["/", "relative/archive", "/tmp/../archive", "local"])
def test_original_root_must_be_safe_absolute_and_distinct(tmp_path, bad_root):
  root, _, manifest, payload = make_demo(tmp_path)
  payload["original_root"] = str(root) if bad_root == "local" else bad_root
  manifest.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="original_path|original_root"):
    task_demo.resolve_demo_path("unrelated", demo_root=root)


def test_file_rows_cannot_implicitly_remap_another_source_tree(tmp_path):
  root, _, manifest, payload = make_demo(tmp_path)
  payload["files"][0]["original_path"] = "/somewhere-else/replay.json"
  manifest.write_text(json.dumps(payload))
  with pytest.raises(ValueError, match="original_root/path"):
    task_demo.resolve_demo_path("unrelated", demo_root=root)
  with pytest.raises(ValueError, match="filesystem root"):
    task_demo.resolve_demo_path("unrelated", demo_root="/")


@pytest.mark.parametrize("malformed", ["duplicate-key", "too-large", "too-many", "nan", "bad-size"])
def test_manifest_input_is_bounded_and_unambiguous(tmp_path, monkeypatch, malformed):
  root, _, manifest, payload = make_demo(tmp_path)
  text = json.dumps(payload)
  if malformed == "duplicate-key":
    text = text.replace('"files":', '"files": [], "files":', 1)
  elif malformed == "too-large":
    monkeypatch.setattr(task_demo, "_MAX_MANIFEST_BYTES", 20)
  elif malformed == "too-many":
    monkeypatch.setattr(task_demo, "_MAX_ENTRIES", 0)
  elif malformed == "nan":
    text = text[:-1] + ', "unexpected": NaN}'
  else:
    payload["files"][0]["size"] = True
    text = json.dumps(payload)
  manifest.write_text(text)
  with pytest.raises(ValueError):
    task_demo.resolve_demo_path("unrelated", demo_root=root)
