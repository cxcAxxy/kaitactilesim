from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import convert_usb_to_egosteer as converter  # noqa: E402


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _contracts(index: int) -> tuple[dict, dict]:
  controller = {"record_usb_dataset.py": "c" * 64}
  metadata = {
    "scene": "usb-insert",
    "recording_contract": "usb_insert_taskspace_raw_v1",
    "observation_clock": "post_step_forward_v1",
    "episode_index": index,
    "object_seed": 1000 + index,
    "noise_seed": 2000 + index,
    "motion_profile": "fast",
    "controller_source_sha256": controller,
  }
  outcome = {
    "success": True,
    "released": True,
    "grasp_verified": True,
    "active_bottom_out_confirmed": True,
    "source_files_unchanged": True,
    "object_name": "usb_plug",
    "controller_source_sha256_at_end": controller,
    "insertion": {"success": True, "seated": True},
  }
  return metadata, outcome


def _write_episode(root: Path, index: int) -> dict:
  metadata, outcome = _contracts(index)
  source = root / f"usb_{index:06d}.h5"
  acquisition = np.array([0.0, 0.034, 0.066, 0.100, 0.110])
  pose_times = np.array([0.0, 0.032, 0.064, 0.098, 0.108])
  state_times = np.arange(12, dtype=np.float64) / 100.0
  state_indices = np.array([0, 3, 6, 10, 11], dtype=np.int64)
  wrists = np.tile(np.eye(4), (5, 2, 1, 1))
  tips = np.tile(np.eye(4), (5, 2, 5, 1, 1))
  cameras = np.tile(np.eye(4), (5, 1, 1))
  wrists[..., :3, 3] = np.arange(30).reshape(5, 2, 3) / 1000
  tips[..., :3, 3] = np.arange(150).reshape(5, 2, 5, 3) / 1000
  cameras[:, 0, 3] = np.arange(5) / 1000
  wrist_cameras = cameras.copy()
  wrist_cameras[:, 1, 3] = 0.2
  rng = np.random.default_rng(index)
  rgb = rng.integers(0, 256, (5, 12, 16, 3), dtype=np.uint8)
  wrist_rgb = rng.integers(0, 256, (5, 12, 16, 3), dtype=np.uint8)

  with h5py.File(source, "w") as file:
    file.attrs["schema_version"] = converter.RAW_SCHEMA
    file.attrs["camera_hz"] = 30
    file.attrs["physics_hz"] = 500
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["outcome_json"] = json.dumps(outcome)
    file.attrs["model_path"] = "/relocated/source/scene.xml"
    file.attrs["model_sha256"] = "b" * 64
    file["state/timestamp"] = state_times
    phases = ["motion"] * 11 + ["terminal_settle"]
    file["commands/phase"] = np.asarray(phases, dtype=h5py.string_dtype())
    camera = file.create_group("cameras/head")
    camera.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
    camera.attrs["side_names_json"] = json.dumps(["left", "right"])
    camera.attrs["finger_names_json"] = json.dumps(
      ["thumb", "index", "middle", "ring", "little"]
    )
    camera.attrs["height"] = 12
    camera.attrs["width"] = 16
    camera["timestamp"] = acquisition
    camera["pose_timestamp"] = pose_times
    camera["state_index"] = state_indices
    camera["world_from_camera"] = cameras
    camera["world_from_wrist"] = wrists
    camera["world_from_fingertip"] = tips
    camera["intrinsic"] = np.array(
      [[10.0, 0.0, 7.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]
    )
    camera["rgb"] = rgb
    wrist = file.create_group("cameras/right_wrist")
    wrist["timestamp"] = acquisition
    wrist["pose_timestamp"] = pose_times
    wrist["state_index"] = state_indices
    wrist["world_from_camera"] = wrist_cameras
    wrist["intrinsic"] = np.array(
      [[12.0, 0.0, 7.0], [0.0, 11.0, 5.0], [0.0, 0.0, 1.0]]
    )
    wrist["rgb"] = wrist_rgb

  capture = {
    "schema_version": converter.RAW_SCHEMA,
    "episode": source.name,
    "sha256": _sha256(source),
    "state_samples": len(state_times),
    "camera_samples": {
      "head": len(acquisition),
      "right_wrist": len(acquisition),
    },
    "outcome": outcome,
  }
  (root / f"usb_{index:06d}.json").write_text(json.dumps(capture))
  result = {
    "episode_index": index,
    "object_seed": metadata["object_seed"],
    "noise_seed": metadata["noise_seed"],
    "motion_profile": metadata["motion_profile"],
    "status": "success",
    "success": True,
    "raw_path": str(source),
    "failure_reason": None,
    "recording_errors": [],
    "validation": {
      "valid": True,
      "errors": [],
      "state_samples": len(state_times),
      "camera_samples": {
        "head": len(acquisition),
        "right_wrist": len(acquisition),
      },
    },
    "outcome": outcome,
  }
  (root / f"usb_{index:06d}.result.json").write_text(json.dumps(result))
  return {
    "episode_index": index,
    "status": "success",
    "success": True,
    "raw_path": str(source),
    "object_seed": metadata["object_seed"],
    "noise_seed": metadata["noise_seed"],
    "motion_profile": metadata["motion_profile"],
    "failure_reason": None,
  }


def _write_batch(root: Path, count: int) -> None:
  root.mkdir()
  rows = [_write_episode(root, index) for index in range(count)]
  (root / "summary.json").write_text(
    json.dumps(
      {
        "schema_version": converter.BATCH_SCHEMA,
        "complete": True,
        "planned_attempts": count,
        "recorded_attempts": count,
        "successful_episodes": count,
        "episodes": rows,
      }
    )
  )


def test_output_path_is_isolated_from_raw_and_pi05(tmp_path: Path) -> None:
  raw = tmp_path / "raw" / "0914_200"
  raw.mkdir(parents=True)
  with pytest.raises(ValueError, match="differ"):
    converter._validate_path_isolation(raw, raw)
  with pytest.raises(ValueError, match="contain"):
    converter._validate_path_isolation(raw, raw / "egosteer")
  with pytest.raises(ValueError, match="pi05"):
    converter._validate_path_isolation(raw, tmp_path / "pi05" / "0914_200")
  converter._validate_path_isolation(
    raw, tmp_path / "egosteer" / "0914_200"
  )


def test_existing_output_is_rejected_before_source_scan(tmp_path: Path) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "egosteer"
  source.mkdir()
  output.mkdir()
  args = converter._parse_args(
    [
      "--input-dir",
      str(source),
      "--output-dir",
      str(output),
      "--expected-episodes",
      "1",
    ]
  )
  with pytest.raises(FileExistsError, match="already exists"):
    converter._run(args)


def test_usb_success_gates_are_mandatory() -> None:
  _, valid = _contracts(0)
  converter._validate_outcome_shape(valid, "test")
  for path in (
    ("released",),
    ("active_bottom_out_confirmed",),
    ("insertion", "seated"),
  ):
    invalid = json.loads(json.dumps(valid))
    target = invalid
    for part in path[:-1]:
      target = target[part]
    target[path[-1]] = False
    with pytest.raises(ValueError, match="gate failed"):
      converter._validate_outcome_shape(invalid, "test")


def test_limit_still_checks_complete_batch_pairing(tmp_path: Path) -> None:
  source = tmp_path / "raw"
  _write_batch(source, 2)
  (source / "usb_000001.result.json").unlink()
  with pytest.raises(ValueError, match="result sidecar"):
    converter._discover_batch(source, expected_episodes=2)


def test_optional_full_source_hash_is_verified(tmp_path: Path) -> None:
  source = tmp_path / "raw"
  _write_batch(source, 1)
  _, episodes = converter._discover_batch(source, expected_episodes=1)
  facts = converter._preflight_episode(episodes[0], verify_source_hash=True)
  assert facts.source_hash_verified is True


def test_unified_collection_converts_nested_usb_with_outer_index(tmp_path: Path) -> None:
  root = tmp_path / "collection"
  data = root / "usb-insert/000042/attempt_001/data/raw"
  data.mkdir(parents=True)
  _write_episode(data, 0)
  source = data / "usb_000000.h5"
  (root / "collection.json").write_text(json.dumps({
    "schema": "task_collection_v2",
    "tasks": ["usb-insert"],
    "collection_mode": "attempts",
  }))
  (root / "summary.json").write_text(json.dumps({
    "episodes": [{
      "task": "usb-insert", "episode_index": 42,
      "status": "success", "hdf5": [source.relative_to(root).as_posix()],
    }],
    "by_task": {"usb-insert": {"success": 1}},
  }))
  output = tmp_path / "egosteer"
  converter.main([
    "--input-dir", str(root), "--output-dir", str(output),
    "--expected-episodes", "1", "--workers", "1",
    "--cameras", "head", "right_wrist",
  ])
  report = json.loads((output / "conversion_report.json").read_text())
  assert report["converted_episodes"] == 1
  source_manifest = json.loads((output / "source_snapshot_manifest.json").read_text())
  assert source_manifest["episodes"][0]["episode_index"] == 42
  assert source_manifest["episodes"][0]["hdf5"] == source.relative_to(root).as_posix()


def test_parallel_export_writes_one_valid_standard_shard_per_episode(
  tmp_path: Path,
) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "egosteer" / "0914_200"
  _write_batch(source, 2)

  converter.main(
    [
      "--input-dir",
      str(source),
      "--output-dir",
      str(output),
      "--expected-episodes",
      "2",
      "--workers",
      "2",
      "--dataset-name",
      "usb_insert_test",
    ]
  )

  validation = json.loads((output / "validation.json").read_text())
  manifest = json.loads((output / "dataset_manifest.json").read_text())
  report = json.loads((output / "conversion_report.json").read_text())
  assert validation["valid"] is True
  assert validation["episodes"] == 2
  assert report["complete_batch_episodes"] == 2
  assert report["converted_episodes"] == 2
  assert manifest["cameras"] == ["head"]
  assert manifest["lowdim_dim"] == 116
  assert manifest["tactile_included"] is False
  assert manifest["one_episode_per_shard"] is True
  assert not (output / "tactile_sidecars").exists()

  shards = sorted(output.rglob("shard-*.tar"))
  assert len(shards) == 2
  observed_episodes = set()
  for shard in shards:
    with tarfile.open(shard, "r:") as archive:
      names = archive.getnames()
    episode_prefixes = {name.split("_frame_", 1)[0] for name in names}
    assert len(episode_prefixes) == 1
    observed_episodes.add(next(iter(episode_prefixes)))
    assert len(names) == 9  # three frames, three members per frame
    assert not any("tactile" in name or "right_wrist" in name for name in names)
  assert observed_episodes == {"episode_000000", "episode_000001"}


def test_head_and_right_wrist_export_has_four_members_and_136d_lowdim(
  tmp_path: Path,
) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "egosteer" / "0914_200_head_right_wrist"
  _write_batch(source, 1)

  converter.main(
    [
      "--input-dir",
      str(source),
      "--output-dir",
      str(output),
      "--expected-episodes",
      "1",
      "--workers",
      "1",
      "--cameras",
      "head",
      "right_wrist",
    ]
  )

  manifest = json.loads((output / "dataset_manifest.json").read_text())
  validation = json.loads((output / "validation.json").read_text())
  assert validation["valid"] is True
  assert manifest["cameras"] == ["head", "right_wrist"]
  assert manifest["lowdim_dim"] == 136
  assert manifest["member_order"] == [
    "image.jpg",
    "right_wrist_image.jpg",
    "lowdim.npy",
    "meta.json",
  ]
  shard = next(output.rglob("shard-*.tar"))
  with tarfile.open(shard, "r:") as archive:
    names = archive.getnames()
    lowdim_member = next(name for name in names if name.endswith(".lowdim.npy"))
    stream = archive.extractfile(lowdim_member)
    assert stream is not None
    lowdim = np.load(io.BytesIO(stream.read()), allow_pickle=False)
  assert len(names) == 12  # three frames, four members per frame
  assert sum(name.endswith(".right_wrist_image.jpg") for name in names) == 3
  assert lowdim.shape == (136,)
  expected_world_from_wrist_camera = np.eye(4)
  expected_world_from_wrist_camera[1, 3] = 0.2
  np.testing.assert_allclose(
    lowdim[116:132],
    converter._camera_from_world_cv(expected_world_from_wrist_camera).reshape(-1),
  )
  np.testing.assert_allclose(lowdim[132:136], [12.0, 11.0, 7.0, 5.0])


def test_validate_only_limit_creates_no_output(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "missing-parent" / "egosteer"
  _write_batch(source, 2)
  seen = {}

  def fake_pool(operation, episodes, workers, *args):
    seen["episodes"] = [item.episode_index for item in episodes]
    return [
      converter.EpisodeFacts(
        episode_index=0,
        source_frames=5,
        strict_prefix_frames=4,
        exported_samples=3,
        off_grid_frames_discarded=1,
        maximum_grid_error_seconds=0.001,
        image_width=16,
        image_height=12,
        model_path="scene.xml",
        model_sha256="b" * 64,
        source_hash_verified=False,
      )
    ]

  monkeypatch.setattr(converter, "_run_pool", fake_pool)
  converter.main(
    [
      "--input-dir",
      str(source),
      "--output-dir",
      str(output),
      "--expected-episodes",
      "2",
      "--validate-only",
      "--limit",
      "1",
    ]
  )
  assert seen["episodes"] == [0]
  assert not output.exists()
  assert not output.parent.exists()
