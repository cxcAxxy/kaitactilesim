"""Offline transfer regressions with small synthetic recordings; no physics."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_export as exporter
from kaihand_tactile_env.shared import tict_package as package
from kaihand_tactile_env.shared.tict_validation import validate_tict_release
from test_tict_contract import _source


def _json(path):
  return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
  path.write_text(json.dumps(value), encoding="utf-8")


def _release(base, session, shift=0.0):
  directory = base / session
  directory.mkdir()
  source = _source(directory)
  with h5py.File(source, "r+") as file:
    file.attrs["fixture_id"] = session
    poses = file["cameras/head/world_from_wrist"][:]
    poses[..., 0, 3] += shift
    file["cameras/head/world_from_wrist"][:] = poses
    fingers = file["cameras/head/world_from_fingertip"][:]
    fingers[..., 0, 3] += shift
    file["cameras/head/world_from_fingertip"][:] = fingers
    file["tactile_contact_force/normal_taxel_force_n"][:] += shift
  root = directory / "release"
  exporter.export_tict_episode(source, root, session_id=session)
  return root


def _manifest(base, assignments):
  manifest = base / "assignments.json"
  _write(manifest, {"schema_version": package.MANIFEST_SCHEMA, "sessions": assignments})
  return manifest


def _assignment(root, session, split="train"):
  return {"release_dir": str(root), "session_id": session, "split": split}


def _source_audit(root, session):
  audit = _json(root / "dataset_audit.json")
  return {
    "schema_version": "kaihand-tict-source-audit-v1",
    "session_id": session,
    "valid": True,
    "source": audit["source"],
    "release": {
      "sidecar_sha256": audit["sessions"][0]["sidecar_sha256"],
      "dataset_audit_sha256": exporter.sha256_file(root / "dataset_audit.json"),
    },
  }


def _payload_hashes(root):
  return {
    str(path.relative_to(root)): exporter.sha256_file(path)
    for subdirectory in ("production", "tict_sidecars")
    for path in (root / subdirectory).rglob("*")
    if path.is_file()
  }


@pytest.mark.parametrize("corrupt", [False, True])
def test_reuse_audits_still_validates_assembled_content(tmp_path, corrupt):
  source = _release(tmp_path, "session")
  assignment = _assignment(source, "session")
  audit_path = tmp_path / "source_audit.json"
  _write(audit_path, _source_audit(source, "session"))
  assignment["source_audit_path"] = str(audit_path)
  manifest = _manifest(tmp_path, [assignment])
  if corrupt:
    path = source / "production/session/09_humanego_adapter/preprocess/all_data/00000/training_data.json"
    value = _json(path)
    value["metadata"]["idx"] = 999
    _write(path, value)
    with pytest.raises(ValueError, match="assembled release failed validation"):
      package.package_tict_releases(manifest, tmp_path / "output", link_payloads=True, reuse_source_audits=True)
    assert not (tmp_path / "output").exists()
  else:
    result = package.package_tict_releases(manifest, tmp_path / "output", link_payloads=True, reuse_source_audits=True)
    assert result["valid"]
    assert validate_tict_release(tmp_path / "output")["valid"]
    assert _payload_hashes(source) == _payload_hashes(tmp_path / "output")


def test_reuse_requires_source_audit(tmp_path):
  source = _release(tmp_path, "session")
  manifest = _manifest(tmp_path, [_assignment(source, "session")])
  with pytest.raises(ValueError, match="requires a passing independent source audit"):
    package.package_tict_releases(manifest, tmp_path / "output", reuse_source_audits=True)


def test_package_assigns_whole_sessions_and_uses_train_statistics_only(tmp_path):
  train = _release(tmp_path, "usb_train", shift=0.0)
  held_out = _release(tmp_path, "usb_test", shift=100.0)
  source_hashes = _payload_hashes(train) | _payload_hashes(held_out)
  manifest = _manifest(
    tmp_path,
    [
      _assignment(train, "usb_train"),
      _assignment(held_out, "usb_test", "test"),
    ],
  )
  destination = tmp_path / "combined"
  result = package.package_tict_releases(manifest, destination)
  assert result["valid"]
  assert validate_tict_release(destination)["valid"]
  assert _payload_hashes(destination) == source_hashes
  assert _payload_hashes(train) | _payload_hashes(held_out) == source_hashes
  assert _json(destination / "split_manifest.json")["splits"] == {
    "train": ["usb_train"],
    "validation": [],
    "test": ["usb_test"],
  }
  assert _json(destination / "window_starts.json")["sessions"] == {
    "usb_train": [0],
    "usb_test": [0],
  }
  stats = _json(destination / "train_statistics.json")
  expected = _json(train / "train_statistics.json")
  assert stats["sessions"] == ["usb_train"]
  assert stats["rotation_6d_normalization"] == "none"
  for name in ("wrist_xyz", "shared_ten_finger_xyz"):
    assert stats[name]["count"] == expected[name]["count"]
    np.testing.assert_allclose(stats[name]["mean"], expected[name]["mean"], atol=1e-10)
    np.testing.assert_allclose(stats[name]["std"], expected[name]["std"], atol=1e-10)
  for channel in exporter.CHANNELS:
    np.testing.assert_allclose(
      stats["tactile_channels"][channel]["mean"],
      expected["tactile_channels"][channel]["mean"],
      atol=1e-10,
    )
  provenance = destination / "source_provenance" / "usb_train"
  assert (provenance / "dataset_audit.json").read_bytes() == (
    train / "dataset_audit.json"
  ).read_bytes()


def test_linked_assembly_survives_intermediate_cleanup(tmp_path):
  release = _release(tmp_path, "linked_session")
  manifest = _manifest(tmp_path, [_assignment(release, "linked_session")])
  output = tmp_path / "linked_package"
  report = package.package_tict_releases(manifest, output, link_payloads=True)
  assert report["valid"]
  relative = "production/linked_session/09_humanego_adapter/preprocess/all_data/00000/rgb.png"
  assert (release / relative).stat().st_ino == (output / relative).stat().st_ino
  expected = _payload_hashes(output)
  shutil.rmtree(release)
  assert not (output / relative).is_symlink()
  assert _payload_hashes(output) == expected
  assert validate_tict_release(output)["valid"]


def test_recomputes_population_statistics_instead_of_averaging_session_stds(tmp_path):
  train_a = _release(tmp_path, "a", shift=0.0)
  train_b = _release(tmp_path, "b", shift=1.0)
  manifest = _manifest(tmp_path, [_assignment(train_a, "a"), _assignment(train_b, "b")])
  # Input validator checks the statistics contract, not numeric truth. Assembly
  # must calculate from actual training frames even when stored stats are wrong.
  for root in (train_a, train_b):
    path = root / "train_statistics.json"
    fake = _json(path)
    fake["wrist_xyz"].update(mean=[999, 999, 999], std=[999, 999, 999])
    _write(path, fake)
  destination = tmp_path / "combined"
  package.package_tict_releases(manifest, destination)
  stats = _json(destination / "train_statistics.json")
  assert stats["wrist_xyz"]["count"] == 200
  assert max(stats["wrist_xyz"]["mean"]) < 10
  normal = stats["tactile_channels"]["normal"]
  expected = np.concatenate(
    [
      np.load(root / "tict_sidecars" / name / "fingertip_tactile_v1.npz")[
        "tactile_mean"
      ][..., 0].ravel()
      for root, name in ((train_a, "a"), (train_b, "b"))
    ]
  ).astype(np.float64)
  assert normal["count"] == len(expected)
  assert normal["mean"] == pytest.approx(expected.mean(), abs=1e-12)
  assert normal["std"] == pytest.approx(expected.std(), abs=1e-12)


def test_received_package_rebases_without_reading_old_absolute_paths(tmp_path):
  source = _release(tmp_path, "usb")
  manifest = _manifest(tmp_path, [_assignment(source, "usb")])
  package_root = tmp_path / "outgoing"
  package.package_tict_releases(manifest, package_root)
  original_hashes = _payload_hashes(package_root)
  received = tmp_path / "received on another computer"
  shutil.move(str(package_root), received)
  path = received / "selector_manifest.json"
  selector = _json(path)
  for record in selector["records"]:
    record["rgb_path"] = "/unavailable/other/computer/or/untrusted/path.png"
  _write(path, selector)
  old_selector = path.read_bytes()
  report = package.rebase_tict_release(received)
  assert report["transfer_integrity_verified"]
  assert not report["old_absolute_paths_used_for_io"]
  assert report["validation"]["valid"]
  assert _payload_hashes(received) == original_hashes
  assert all(
    record["rgb_path"].startswith(str(received)) for record in _json(path)["records"]
  )
  backups = list(
    (received / "relocation_history").glob("*/selector_manifest.before.json")
  )
  assert len(backups) == 1 and backups[0].read_bytes() == old_selector
  package.rebase_tict_release(received)
  assert len(list((received / "relocation_history").glob("*/rebase_record.json"))) == 2


@pytest.mark.parametrize("payload", ["rgb", "sidecar", "json"])
def test_rebase_rejects_corruption_before_replacing_selector(tmp_path, payload):
  source = _release(tmp_path, "usb")
  manifest = _manifest(tmp_path, [_assignment(source, "usb")])
  root = tmp_path / "outgoing"
  package.package_tict_releases(manifest, root)
  selector_path = root / "selector_manifest.json"
  original = selector_path.read_bytes()
  record = _json(selector_path)["records"][0]
  corrupt = {
    "rgb": root / record["training_data_path"].replace("training_data.json", "rgb.png"),
    "json": root / record["training_data_path"],
    "sidecar": root / "tict_sidecars/usb/fingertip_tactile_v1.npz",
  }[payload]
  with corrupt.open("ab") as stream:
    stream.write(b"corrupt")
  with pytest.raises(ValueError, match="hash|byte-count"):
    package.rebase_tict_release(root)
  assert selector_path.read_bytes() == original


@pytest.mark.parametrize("damage", ["parent", "absolute", "symlink", "session"])
def test_rebase_rejects_path_escape(tmp_path, damage):
  root = _release(tmp_path, "usb")
  selector_path = root / "selector_manifest.json"
  selector = _json(selector_path)
  record = selector["records"][0]
  if damage == "parent":
    record["training_data_path"] = "../outside/training_data.json"
  elif damage == "absolute":
    record["training_data_path"] = "/outside/training_data.json"
  elif damage == "session":
    record["session_id"] = "../outside"
  else:
    rgb = root / record["training_data_path"].replace("training_data.json", "rgb.png")
    outside = tmp_path / "outside.png"
    shutil.copyfile(rgb, outside)
    rgb.unlink()
    rgb.symlink_to(outside)
  _write(selector_path, selector)
  original = selector_path.read_bytes()
  with pytest.raises(ValueError, match="canonical|unsafe|safe|symlink"):
    package.rebase_tict_release(root)
  assert selector_path.read_bytes() == original


def test_failed_final_rebase_validation_restores_exact_selector(tmp_path):
  root = _release(tmp_path, "usb")
  selector_path = root / "selector_manifest.json"
  original = selector_path.read_bytes()
  record = _json(selector_path)["records"][0]
  frame_path = root / record["training_data_path"]
  frame = _json(frame_path)
  frame["metadata"]["is_finished"] = True
  _write(frame_path, frame)
  with pytest.raises(ValueError, match="final-frame terminal"):
    package.rebase_tict_release(root)
  assert selector_path.read_bytes() == original
  assert len(list((root / "relocation_history").glob("*/FAILED.json"))) == 1


@pytest.mark.parametrize(
  "kind", ["duplicate", "no_train", "bad_split", "renamed_source"]
)
def test_package_rejects_split_leakage_or_missing_train(tmp_path, kind):
  source = _release(tmp_path, "usb")
  assignments = [_assignment(source, "usb")]
  if kind == "duplicate":
    assignments.append(_assignment(source, "usb", "test"))
  elif kind == "no_train":
    assignments[0]["split"] = "test"
  elif kind == "bad_split":
    assignments[0]["split"] = "random_frames"
  else:
    alternate = tmp_path / "alternate"
    exporter.export_tict_episode(
      tmp_path / "usb/synthetic.h5", alternate, session_id="renamed"
    )
    assignments.append(_assignment(alternate, "renamed", "test"))
  manifest = _manifest(tmp_path, assignments)
  output = tmp_path / "must_not_publish"
  with pytest.raises(ValueError, match="split|train|duplicate source"):
    package.package_tict_releases(manifest, output)
  assert not output.exists()


def test_existing_destination_is_never_replaced_and_source_audit_is_preserved(tmp_path):
  source = _release(tmp_path, "usb")
  source_audit = tmp_path / "source_audit.json"
  _write(
    source_audit,
    _source_audit(source, "usb"),
  )
  item = _assignment(source, "usb")
  item["source_audit_path"] = "source_audit.json"
  manifest = _manifest(tmp_path, [item])
  output = tmp_path / "package"
  package.package_tict_releases(manifest, output)
  assert (
    output / "source_provenance/usb/source_audit.json"
  ).read_bytes() == source_audit.read_bytes()
  before = (output / "dataset_audit.json").read_bytes()
  with pytest.raises(FileExistsError):
    package.package_tict_releases(manifest, output)
  assert (output / "dataset_audit.json").read_bytes() == before


def test_received_script_works_without_site_packages_or_project_imports(tmp_path):
  source = _release(tmp_path, "usb")
  manifest = _manifest(tmp_path, [_assignment(source, "usb")])
  output = tmp_path / "published"
  package.package_tict_releases(manifest, output)
  received = tmp_path / "receiver"
  shutil.move(str(output), received)
  before = _payload_hashes(received)
  # -I -S isolates the interpreter from PYTHONPATH, site-packages and installed
  # project modules. The delivered script must use only Python's standard library.
  result = subprocess.run(
    [sys.executable, "-I", "-S", str(received / "rebase_paths.py")],
    cwd=tmp_path,
    text=True,
    capture_output=True,
    check=False,
    timeout=30,
  )
  assert result.returncode == 0, result.stderr
  report = json.loads(result.stdout)
  assert report["transfer_integrity_verified"]
  assert not report["validation"]["full_geometry_validation_rerun"]
  assert _payload_hashes(received) == before
  assert validate_tict_release(received)["valid"]


@pytest.mark.parametrize("field", ["valid", "session_id", "source", "release"])
def test_source_audit_must_identify_the_same_session_and_source(tmp_path, field):
  source = _release(tmp_path, "usb")
  audit = _source_audit(source, "usb")
  audit[field] = {
    "valid": False,
    "session_id": "another_session",
    "source": {"sha256": "wrong"},
    "release": {"sidecar_sha256": "wrong", "dataset_audit_sha256": "wrong"},
  }[field]
  path = tmp_path / "audit.json"
  _write(path, audit)
  item = _assignment(source, "usb")
  item["source_audit_path"] = str(path)
  manifest = _manifest(tmp_path, [item])
  output = tmp_path / "must_not_publish"
  with pytest.raises(ValueError, match="passing audit"):
    package.package_tict_releases(manifest, output)
  assert not output.exists()
