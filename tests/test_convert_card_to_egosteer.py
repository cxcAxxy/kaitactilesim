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

import convert_card_to_egosteer as converter  # noqa: E402


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _contracts(index: int) -> tuple[dict, dict]:
  metadata = {
    "acceptance_policy": "task-completion-v1",
    "episode_index": index,
    "object": "card",
    "preset": "middle-force-precontact-v1",
    "recording_contract": "poker_draw_per_finger_press_shear_v5",
    "scene": "poker-draw",
    "side": "right",
  }
  outcome = {
    "acceptance_policy": "task-completion-v1",
    "draw_posture_version": "tip-pad-v7",
    "edge_outcome": {
      "full_slide_qualified": True,
      "held_at_edge": True,
      "slide_geometry_qualified": True,
      "slide_task_completed": True,
      "target_reached": True,
      "terminal_reason": "edge_reached",
    },
    "half_overhang_reached": True,
    "handoff_outcome": {"completed": True},
    "inspection_face_alignment": 0.9,
    "inspection_face_robot_alignment": 0.9,
    "inspection_position_error": 0.01,
    "maximum_slide_fingertip_plane_angle_degrees": 40.0,
    "minimum_supported_card_clearance": 0.0,
    "object_name": "card",
    "pressure_quality": "stable",
    "retained_at_end": True,
    "side": "right",
    "simultaneous_four_finger_contact": True,
    "slide_fingertip_plane_angle_limit_degrees": 50.0,
    "slide_press_control_qualified": True,
    "success": True,
    "sustained_pinch": True,
    "task_completed": True,
    "terminal_pinch": True,
  }
  return metadata, outcome


def _write_episode(root: Path, index: int) -> None:
  metadata, outcome = _contracts(index)
  source = root / f"episode_{index:06d}_card_right.h5"
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
  head_rgb = rng.integers(0, 256, (5, 12, 16, 3), dtype=np.uint8)
  wrist_rgb = rng.integers(0, 256, (5, 12, 16, 3), dtype=np.uint8)

  with h5py.File(source, "w") as file:
    file.attrs["schema_version"] = converter.RAW_SCHEMA
    file.attrs["camera_hz"] = 30
    file.attrs["control_hz"] = 100
    file.attrs["physics_hz"] = 500
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["outcome_json"] = json.dumps(outcome)
    file.attrs["model_path"] = "/relocated/source/scene.xml"
    file.attrs["model_sha256"] = "b" * 64
    file["state/timestamp"] = state_times
    file["commands/phase"] = np.asarray(
      ["motion"] * 11 + ["terminal_settle"],
      dtype=h5py.string_dtype(),
    )
    head = file.create_group("cameras/head")
    head.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
    head.attrs["side_names_json"] = json.dumps(["left", "right"])
    head.attrs["finger_names_json"] = json.dumps(
      ["thumb", "index", "middle", "ring", "little"]
    )
    head.attrs["height"] = 12
    head.attrs["width"] = 16
    head["timestamp"] = acquisition
    head["pose_timestamp"] = pose_times
    head["state_index"] = state_indices
    head["world_from_camera"] = cameras
    head["world_from_wrist"] = wrists
    head["world_from_fingertip"] = tips
    head["intrinsic"] = np.array(
      [[10.0, 0.0, 7.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]
    )
    head["rgb"] = head_rgb
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
  source.with_suffix(".json").write_text(json.dumps(capture))


def _write_batch(root: Path, indices: tuple[int, ...]) -> None:
  root.mkdir()
  for index in indices:
    _write_episode(root, index)


def test_output_path_is_isolated_from_raw_and_pi05(tmp_path: Path) -> None:
  raw = tmp_path / "raw" / "0914_200"
  raw.mkdir(parents=True)
  with pytest.raises(ValueError, match="differ"):
    converter._validate_path_isolation(raw, raw)
  with pytest.raises(ValueError, match="contain"):
    converter._validate_path_isolation(raw, raw / "egosteer")
  with pytest.raises(ValueError, match="pi05"):
    converter._validate_path_isolation(raw, tmp_path / "pi05" / "0914_200")
  converter._validate_path_isolation(raw, tmp_path / "egosteer" / "0914_200")


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


def test_card_success_gates_are_mandatory() -> None:
  _, valid = _contracts(0)
  converter._validate_outcome_shape(valid, "test")
  for path in (
    ("sustained_pinch",),
    ("retained_at_end",),
    ("edge_outcome", "target_reached"),
    ("handoff_outcome", "completed"),
  ):
    invalid = json.loads(json.dumps(valid))
    target = invalid
    for part in path[:-1]:
      target = target[part]
    target[path[-1]] = False
    with pytest.raises(ValueError, match="gate failed"):
      converter._validate_outcome_shape(invalid, "test")


def test_pairing_accepts_noncontiguous_ids_and_rejects_missing_sidecar(
  tmp_path: Path,
) -> None:
  source = tmp_path / "raw"
  _write_batch(source, (0, 44))
  episodes = converter._discover_batch(source, expected_episodes=2)
  assert [item.episode_index for item in episodes] == [0, 44]
  (source / "episode_000044_card_right.json").unlink()
  with pytest.raises(ValueError, match="episode sets differ"):
    converter._discover_batch(source, expected_episodes=2)


def test_optional_full_source_hash_is_verified(tmp_path: Path) -> None:
  source = tmp_path / "raw"
  _write_batch(source, (0,))
  episode = converter._discover_batch(source, expected_episodes=1)[0]
  facts = converter._preflight_episode(episode, verify_source_hash=True)
  assert facts.source_hash_verified is True


def test_parallel_export_writes_one_valid_standard_shard_per_episode(
  tmp_path: Path,
) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "egosteer" / "0914_200"
  _write_batch(source, (0, 44))

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
      "0914_200",
    ]
  )

  validation = json.loads((output / "validation.json").read_text())
  manifest = json.loads((output / "dataset_manifest.json").read_text())
  report = json.loads((output / "conversion_report.json").read_text())
  assert validation["valid"] is True
  assert validation["episodes"] == 2
  assert report["complete_batch_episodes"] == 2
  assert report["converted_episodes"] == 2
  assert report["samples"] == 6
  assert manifest["cameras"] == ["head"]
  assert manifest["lowdim_dim"] == 116
  assert manifest["tactile_included"] is False
  assert manifest["one_episode_per_shard"] is True

  shards = sorted(output.rglob("shard-*.tar"))
  assert len(shards) == 2
  assert {path.parent.name for path in shards} == {"train", "val"}
  for shard in shards:
    with tarfile.open(shard, "r:") as archive:
      names = archive.getnames()
    assert len({name.split("_frame_", 1)[0] for name in names}) == 1
    assert len(names) == 9
    assert not any("tactile" in name or "right_wrist" in name for name in names)


def test_head_and_right_wrist_export_has_four_members_and_136d_lowdim(
  tmp_path: Path,
) -> None:
  source = tmp_path / "raw"
  output = tmp_path / "egosteer" / "0914_200_head_right_wrist"
  _write_batch(source, (0,))

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
  _write_batch(source, (0, 44))
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


def test_limit_is_validation_only() -> None:
  with pytest.raises(SystemExit):
    converter._parse_args(["--limit", "1"])
