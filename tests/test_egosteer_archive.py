"""Small synthetic checks: no simulation, renderer or training dependencies."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.egosteer_archive import (
  archived_motion,
  observation_archive,
  validate_observation_archive,
)


@pytest.fixture
def recorded(tmp_path):
  with h5py.File(tmp_path / "source.h5", "w") as file:
    file.attrs["physics_hz"] = 500
    file.attrs["metadata_json"] = "{}"
    file.attrs["outcome_json"] = "{}"
    camera = file.create_group("cameras/head")
    camera.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
    camera.attrs["side_names_json"] = json.dumps(["left", "right"])
    camera.attrs["finger_names_json"] = json.dumps(["thumb", "index", "middle", "ring", "little"])
    camera["timestamp"] = [0., .034, .066, .100, .110]
    camera["pose_timestamp"] = [0., .032, .064, .098, .108]
    file["state/timestamp"] = [0., .010, .020, .030, .040, .050, .060, .070, .080, .090, .100, .110]
    camera["state_index"] = [0, 3, 6, 10, 11]
    file["commands/phase"] = np.asarray(["terminal_settle"], dtype=h5py.string_dtype())
    wrists = np.tile(np.eye(4), (5, 2, 1, 1))
    tips = np.tile(np.eye(4), (5, 2, 5, 1, 1))
    wrists[..., :3, 3] = np.arange(30).reshape(5, 2, 3) / 100
    tips[..., :3, 3] = np.arange(150).reshape(5, 2, 5, 3) / 100
    camera["world_from_wrist"] = wrists
    camera["world_from_fingertip"] = tips
    camera["world_from_camera"] = np.tile(np.eye(4), (5, 1, 1))
    camera["intrinsic"] = [[200., 0, 160], [0, 200, 120], [0, 0, 1]]
    group = file.create_group("tactile_contact_force")
    group.attrs["force_unit"] = "N"
    names = [f"hand_{hand}_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
             for hand in ("l", "r") for finger in ("thumb", "index", "middle", "ring", "pinky")][::-1]
    group["link_names"] = np.asarray(names, dtype=h5py.string_dtype())
    group["timestamp"] = [0., .02, .04, .06, .08, .10]
    normal = np.arange(6 * 10 * 7 * 5).reshape(6, 10, 7, 5) / 10000
    group["normal_taxel_force_n"] = normal
    group["tangent_taxel_force_n"] = np.stack((normal * -.5, normal * .25), axis=-1)
    yield file


def archive(file):
  return observation_archive(file, episode_index=3, prefix=4,
                             pose_times=file["cameras/head/pose_timestamp"][:],
                             source_sha256="abc", include_tactile=True)


def validate(values):
  validate_observation_archive(values, episode_index=3, samples=3, source_frames=5,
                                prefix=4, source_sha256="abc", include_tactile=True)


def test_causal_touch_axis_reordering_and_terminal(recorded):
  values = archive(recorded)
  validate(values)
  np.testing.assert_array_equal(values["tactile_source_index"], [0, 1, 3, 4, 5])
  np.testing.assert_array_equal(values["training_sample_mask"], [True, True, True, False, False])
  np.testing.assert_array_equal(values["strict_grid_mask"], [True, True, True, True, False])
  np.testing.assert_array_equal(values["terminal_mask"], [False, False, False, False, True])
  normal = recorded["tactile_contact_force/normal_taxel_force_n"][1, 9]
  np.testing.assert_array_equal(values["tactile_taxel_force_n"][1, 0, 0, ..., 0], normal)
  np.testing.assert_array_equal(values["tactile_taxel_force_n"][1, 0, 0, ..., 1], normal * -.5)
  np.testing.assert_array_equal(values["tactile_taxel_force_n"][1, 0, 0, ..., 2], normal * .25)


def test_reject_future_touch_and_wrong_sum(recorded):
  values = archive(recorded)
  values["tactile_source_index"][1] = 2
  with pytest.raises(ValueError, match="latest-not-future"):
    validate(values)
  values = archive(recorded)
  values["tactile_force_n"][1, 0, 0, 0] += 1
  with pytest.raises(ValueError, match="conservation"):
    validate(values)


def test_archived_pose_and_independent_116d_reference(recorded, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  # Normal imports share the same module implementations as the CLI.
  from convert_to_egosteer import SITE_FROM_EGOSTEER_WRIST
  from package_poker_egosteer import reference_lowdim
  times, wrists, hands = archived_motion(recorded, SITE_FROM_EGOSTEER_WRIST)
  np.testing.assert_array_equal(times, recorded["cameras/head/pose_timestamp"][:])
  reference = reference_lowdim(recorded["cameras/head"], 1)
  np.testing.assert_array_equal(reference[:48], np.concatenate((wrists[1], hands[1])))
  np.testing.assert_array_equal(reference[48:96], np.concatenate((wrists[2], hands[2])))
  np.testing.assert_array_equal(reference[96:112].reshape(4, 4), np.diag([1, -1, -1, 1]))
  np.testing.assert_array_equal(reference[112:], [200, 200, 160, 120])
  assert "package_poker_egosteer" in sys.modules


def test_reject_improper_pose(recorded):
  canonical = {side: np.eye(3) for side in ("left", "right")}
  recorded["cameras/head/world_from_wrist"][1, 0, 0, 0] = -1
  with pytest.raises(ValueError, match="improper"):
    archived_motion(recorded, canonical)


def test_buffered_rgb_preserves_every_frame_and_bounds_reads(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from package_poker_egosteer import rgb_frames
  data = np.arange(37 * 2 * 3 * 3, dtype=np.uint8).reshape(37, 2, 3, 3)
  class Source:
    def __init__(self):
      self.reads = []
    def __len__(self):
      return len(data)
    def __getitem__(self, item):
      self.reads.append(item)
      return data[item]
  source = Source()
  frames = list(rgb_frames(source, 35))
  assert [index for index, _ in frames] == list(range(35))
  np.testing.assert_array_equal(np.stack([rgb for _, rgb in frames]), data[:35])
  assert [(s.start, s.stop) for s in source.reads] == [(0, 16), (16, 32), (32, 35)]


@pytest.mark.parametrize("mode,source_calls", [("schema", 0), ("source", 1)])
def test_conversion_verification_mode_is_explicit(tmp_path, monkeypatch, mode, source_calls):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  import package_poker_egosteer as package
  calls = {"schema": 0, "source": 0}
  class Validator:
    def __init__(self, root):
      pass
    def validate(self):
      calls["schema"] += 1
      return {"valid": True}
  def audit(*args):
    calls["source"] += 1
    return {"valid": True}
  monkeypatch.setattr(package, "DatasetValidator", Validator)
  monkeypatch.setattr(package, "audit_source", audit)
  package.verify_release(tmp_path, (), (), {}, (), mode)
  assert calls == {"schema": 1, "source": source_calls}
  report = json.loads((tmp_path / "verification_scope.json").read_text())
  assert report["full_source_audit_executed"] is (mode == "source")
  assert (tmp_path / "source_audit.json").exists() is (mode == "source")


def test_schema_mode_still_rejects_bad_output(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  import package_poker_egosteer as package
  from types import SimpleNamespace
  monkeypatch.setattr(package, "DatasetValidator", lambda root: SimpleNamespace(validate=lambda: {"valid": False}))
  with pytest.raises(ValueError, match="validation failed"):
    package.verify_release(tmp_path, (), (), {}, (), "schema")
  assert not (tmp_path / "verification_scope.json").exists()


def test_buffered_conversion_matches_legacy_image_and_pose(recorded, tmp_path, monkeypatch):
  import io
  import tarfile
  from types import SimpleNamespace
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  from package_poker_egosteer import convert, reference_lowdim, _jpeg_bytes
  camera = recorded["cameras/head"]
  rgb = np.random.default_rng(9).integers(0, 256, (5, 240, 320, 3), dtype=np.uint8)
  camera["rgb"] = rgb
  recorded.flush()
  path = Path(recorded.filename)
  stat = path.stat()
  source = SimpleNamespace(path=path, episode_index=3, hdf5_sha256="a" * 64,
                           size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
  (tmp_path / "tactile_sidecars").mkdir()
  tar_path = tmp_path / "data.tar"
  with tarfile.open(tar_path, "w") as output:
    conversion, descriptor = convert(source, "train", output, tmp_path, "test")
  assert conversion.exported_samples == 3
  with tarfile.open(tar_path) as output:
    assert len(output.getmembers()) == 9
    for i in range(3):
      key = f"episode_000003_frame_{i:06d}"
      assert output.extractfile(key + ".image.jpg").read() == _jpeg_bytes(rgb[i], 95)
      values = np.load(io.BytesIO(output.extractfile(key + ".lowdim.npy").read()), allow_pickle=False)
      np.testing.assert_array_equal(values, reference_lowdim(camera, i))
  with np.load(tmp_path / descriptor["path"], allow_pickle=False) as values:
    validate_observation_archive(values, episode_index=3, samples=3, source_frames=5,
                                  prefix=4, source_sha256="a" * 64, include_tactile=True)
