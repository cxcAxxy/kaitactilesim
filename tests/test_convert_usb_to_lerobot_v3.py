from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import convert_usb_to_lerobot_v3 as converter  # noqa: E402


def test_collection_parent_resolves_one_batch(tmp_path: Path) -> None:
  batch = tmp_path / "0920_200"
  batch.mkdir()
  (batch / "summary.json").write_text("{}")
  (batch / "collection.json").write_text("{}")
  assert converter._resolve_collection_root(tmp_path) == batch
  assert converter._resolve_collection_root(batch) == batch

  second = tmp_path / "second"
  second.mkdir()
  (second / "summary.json").write_text("{}")
  (second / "collection.json").write_text("{}")
  with pytest.raises(ValueError, match="exactly one"):
    converter._resolve_collection_root(tmp_path)


def test_merge_manifest_root_resolves(tmp_path: Path) -> None:
  merged = tmp_path / "raw_pi05_clean"
  merged.mkdir()
  (merged / "merge_manifest.json").write_text(
    '{"schema": "vase_wipe_pi05_clean_merge_v1", "episodes": []}'
  )
  assert converter._resolve_collection_root(merged) == merged
  assert converter._resolve_collection_root(tmp_path) == merged


def test_multi_task_collection_selects_only_requested_successes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  tasks = ["usb-insert", "poker-draw"]
  (tmp_path / "collection.json").write_text(json.dumps({
    "schema": converter.COLLECTION_SCHEMA,
    "tasks": tasks, "collection_mode": "target-successes",
  }))
  rows = []
  for index, task in enumerate(tasks):
    path = tmp_path / task / f"{index:06d}" / "episode.h5"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"raw")
    path.with_suffix(".json").write_text("{}")
    rows.append({
      "task": task, "status": "success", "episode_index": index,
      "hdf5": [path.relative_to(tmp_path).as_posix()],
    })
  (tmp_path / "summary.json").write_text(json.dumps({
    "episodes": rows, "success_count": 2,
    "by_task": {task: {"success": 1} for task in tasks},
    "target_met": {task: True for task in tasks},
  }))
  with pytest.raises(ValueError, match="requires --task"):
    converter._detect_task(tmp_path, None)
  assert converter._detect_task(tmp_path, "usb-insert") == "usb-insert"

  selected = []

  def fake_source(collection_root, *, index, hdf5_path, sidecar_path, task,
                  spec, verify_source_hash):
    selected.append((index, task, hdf5_path))
    return converter.SourceEpisode(
      episode_index=index, hdf5_path=hdf5_path, sidecar_path=sidecar_path,
      relative_hdf5_path=hdf5_path.relative_to(collection_root).as_posix(),
      hdf5_sha256="0" * 64, hdf5_size_bytes=3,
      state_samples=10, camera_samples=4, metadata={}, outcome={}, task=task,
    )

  monkeypatch.setattr(converter, "_source_from_files", fake_source)
  sources = converter._discover_sources(
    tmp_path, task="usb-insert", expected_episodes=1,
    verify_source_hash=False,
  )
  assert len(sources) == 1
  assert selected == [(0, "usb-insert", tmp_path / "usb-insert/000000/episode.h5")]


def test_strict_camera_grid_allows_only_one_terminal_frame() -> None:
  regular = np.arange(5, dtype=np.float64) / converter.FPS
  timestamps = np.concatenate((regular, [regular[-1] + 0.01]))
  prefix, error = converter._strict_30hz_prefix(timestamps, physics_hz=500)
  assert prefix == 5
  assert error < 1.0e-12

  internal = regular.copy()
  internal[3] += 0.01
  with pytest.raises(ValueError, match="optional off-grid terminal"):
    converter._strict_30hz_prefix(internal, physics_hz=500)


def test_sponge_camera_grid_uses_100hz_control_deadlines() -> None:
  frame = np.arange(682)
  ticks = (frame * 100 + 29) // 30
  timestamps = ticks / 100.0
  prefix, error = converter._control_tick_30hz_prefix(timestamps, 4000, 100)
  assert prefix == len(timestamps)
  assert error == 0.0
  np.testing.assert_allclose(timestamps[:5], [0, 0.04, 0.07, 0.10, 0.14])
  with pytest.raises(ValueError, match="no two-frame"):
    converter._strict_30hz_prefix(timestamps, 4000)

  internal = timestamps.copy()
  internal[3] += 0.01
  with pytest.raises(ValueError, match="optional off-grid terminal"):
    converter._control_tick_30hz_prefix(internal, 4000, 100)

  terminal = np.concatenate((timestamps[:5], [timestamps[4] + 0.01]))
  assert converter._control_tick_30hz_prefix(terminal, 4000, 100)[0] == 5


def test_sponge_source_requires_validated_task_audit(tmp_path: Path) -> None:
  source = tmp_path / "sponge_grasp_000000.h5"
  source.write_bytes(b"raw")
  sidecar = source.with_suffix(".json")
  outcome = {
    "success": True,
    "contact_integrity": {
      "checked_every_physics_step": True,
      "penetration_count": 0,
    },
  }
  manifest = {
    "episode": source.name,
    "sha256": "0" * 64,
    "schema_version": converter.RAW_SCHEMA,
    "state_samples": 10,
    "camera_samples": {"head": 4, "right_wrist": 4},
    "outcome": outcome,
    "validation": {"valid": True},
    "task_audit_passed": True,
  }
  sidecar.write_text(json.dumps(manifest))
  spec = converter.TASK_SPECS["sponge-grasp"]
  assert spec.object_name == "sponge"
  assert spec.control_hz == 100
  assert spec.diagnostic_group == "sponge_grasp"
  assert converter._source_from_files(
    tmp_path, index=0, hdf5_path=source, sidecar_path=sidecar,
    task="sponge-grasp", spec=spec, verify_source_hash=False,
  ).task == "sponge-grasp"

  manifest["task_audit_passed"] = False
  sidecar.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="recorded-task audit"):
    converter._source_from_files(
      tmp_path, index=0, hdf5_path=source, sidecar_path=sidecar,
      task="sponge-grasp", spec=spec, verify_source_hash=False,
    )
  outcome["contact_integrity"]["penetration_count"] = 1
  with pytest.raises(ValueError, match="contact-integrity gate"):
    converter._validate_success_gate(source, spec, outcome)


def test_se3_pose_is_xyzw_and_normalized() -> None:
  matrices = np.tile(np.eye(4), (2, 1, 1))
  matrices[0, :3, 3] = [1.0, 2.0, 3.0]
  matrices[1, :3, :3] = np.diag([1.0, -1.0, -1.0])
  poses = converter._matrices_to_poses_xyzw(matrices)
  np.testing.assert_allclose(poses[0], [1, 2, 3, 0, 0, 0, 1])
  np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0)
  np.testing.assert_allclose(np.abs(poses[1, 3:]), [1, 0, 0, 0])


def test_canonical_feature_contract_contains_model_neutral_views() -> None:
  source = converter.SourceEpisode(
    episode_index=0,
    hdf5_path=Path("episode.h5"),
    sidecar_path=Path("episode.json"),
    relative_hdf5_path="episode.h5",
    hdf5_sha256="0" * 64,
    hdf5_size_bytes=1,
    state_samples=10,
    camera_samples=4,
    metadata={},
    outcome={},
  )
  plan = converter.EpisodePlan(
    source=source,
    camera_samples=4,
    strict_prefix_frames=4,
    exported_frames=3,
    off_grid_terminal_frames=0,
    maximum_grid_error_seconds=0.0,
    image_height=240,
    image_width=320,
    source_timestamp_start_s=0.0,
    source_timestamp_end_s=0.1,
    phase_names=("approach",),
  )
  features = converter._features(plan, use_videos=True)
  assert features[converter.HEAD_IMAGE_KEY]["dtype"] == "video"
  assert features["observation.state"]["shape"] == (27,)
  assert features["observation.state"]["names"][:7] == list(
    converter.WRIST_POSE_NAMES
  )
  assert features["observation.state.right_joint_position"]["shape"] == (27,)
  assert features["action"]["shape"] == (27,)
  assert features["action.right_wrist_pose"]["shape"] == (7,)
  assert features["action.right_hand_joint_position"]["shape"] == (20,)
  assert features["observation.wrist_wrench.right.local"]["shape"] == (6,)
  assert features["observation.tactile.right.taxel_force"]["shape"] == (5, 7, 5, 3)
  assert "observation.images.left_wrist" not in features


def test_optional_modalities_are_omitted_and_proxy_is_named_truthfully() -> None:
  source = converter.SourceEpisode(
    episode_index=0,
    hdf5_path=Path("episode.h5"),
    sidecar_path=Path("episode.json"),
    relative_hdf5_path="episode.h5",
    hdf5_sha256="0" * 64,
    hdf5_size_bytes=1,
    state_samples=10,
    camera_samples=4,
    metadata={},
    outcome={},
    task="poker-draw",
  )
  plan = converter.EpisodePlan(
    source=source,
    camera_samples=4,
    strict_prefix_frames=4,
    exported_frames=3,
    off_grid_terminal_frames=0,
    maximum_grid_error_seconds=0.0,
    image_height=240,
    image_width=320,
    source_timestamp_start_s=0.0,
    source_timestamp_end_s=0.1,
    phase_names=("approach",),
    has_fingertip_pose=False,
    has_actuator_control=False,
    has_actuator_force=False,
    has_tactile_contact_force=False,
    aggregate_tactile_group="tactile_proxy",
    has_tactile_probes=False,
    diagnostics=(converter.DiagnosticStream(
      source="commands/drive_limit_n",
      output="auxiliary.task.poker_draw.commands.drive_limit_n",
      dtype="float32",
      shape=(1,),
    ),),
  )
  features = converter._features(plan, use_videos=True)
  assert "observation.state.right_fingertip_pose" not in features
  assert "observation.tactile.right.taxel_force" not in features
  assert "auxiliary.observation.right_actuator_control" not in features
  assert "auxiliary.tactile_proxy.right.force_world" in features
  assert "auxiliary.tactile_genesis.right.force_world" not in features
  assert not any("probe_" in key for key in features)
  assert "auxiliary.task.poker_draw.commands.drive_limit_n" in features


def test_pickplace_wrist_calibration_is_rigid() -> None:
  transform = converter.RIGHT_WRIST_CAMERA_FROM_WRIST
  np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-7)
  np.testing.assert_allclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-7)
  np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_whiteboard_task_contract_and_success_gate() -> None:
  spec = converter.TASK_SPECS["whiteboard-wipe"]
  assert spec.object_name == "eraser"
  assert spec.control_hz == 100
  assert spec.diagnostic_group == "whiteboard_wipe"
  outcome = {
    "success": True,
    "pickup_verified": True,
    "released_on_table": True,
    "ink_remaining": [0.0] * 25,
  }
  converter._validate_success_gate(Path("episode.h5"), spec, outcome)

  with pytest.raises(ValueError, match="whiteboard cleaning/release gate"):
    converter._validate_success_gate(
      Path("episode.h5"), spec, {**outcome, "ink_remaining": [0.0, 1.0e-3]},
    )


def test_vase_task_contract_and_success_gate() -> None:
  spec = converter.TASK_SPECS["vase-wipe"]
  assert spec.object_name == "sponge"
  assert spec.control_hz == 500
  assert spec.diagnostic_group == "vase_wipe"
  outcome = {
    "success": True,
    "motion_completed": True,
    "pickup": {"success": True},
    "cleaned_patch_count": 21,
    "patch_count": 21,
    "cleaning": {"mean_remaining": 0.002, "worst_remaining": 0.05},
    "remaining_dirt": [[0.05]],
  }
  converter._validate_success_gate(Path("episode.h5"), spec, outcome)

  with pytest.raises(ValueError, match="vase cleaning/pickup gate"):
    converter._validate_success_gate(
      Path("episode.h5"), spec,
      {**outcome, "cleaning": {"mean_remaining": 0.06, "worst_remaining": 0.05}},
    )


def _test_args() -> Namespace:
  return Namespace(
    repo_id="kaihand/test",
    expected_episodes=1,
    limit=1,
    verify_source_hash=False,
    ffmpeg_preset="veryfast",
    video_crf=23,
    ffmpeg_threads=2,
    video_files_size_in_mb=32,
  )


def test_preflight_checkpoint_round_trip_skips_hdf5_open(tmp_path: Path) -> None:
  collection = tmp_path / "raw"
  collection.mkdir()
  (collection / "collection.json").write_text('{"schema": "test"}')
  (collection / "summary.json").write_text('{"episodes": []}')
  hdf5 = collection / "episode.h5"
  hdf5.write_bytes(b"not actually HDF5")
  sidecar = collection / "episode.json"
  sidecar.write_text("{}")
  source = converter.SourceEpisode(
    episode_index=4,
    hdf5_path=hdf5,
    sidecar_path=sidecar,
    relative_hdf5_path="episode.h5",
    hdf5_sha256="1" * 64,
    hdf5_size_bytes=hdf5.stat().st_size,
    state_samples=10,
    camera_samples=4,
    metadata={"object_seed": 1},
    outcome={"success": True},
  )
  plan = converter.EpisodePlan(
    source=source,
    camera_samples=4,
    strict_prefix_frames=4,
    exported_frames=3,
    off_grid_terminal_frames=0,
    maximum_grid_error_seconds=0.0,
    image_height=240,
    image_width=320,
    source_timestamp_start_s=0.0,
    source_timestamp_end_s=0.1,
    phase_names=("approach",),
  )
  checkpoint = tmp_path / "work" / "preflight_plan.json"
  _, fingerprint = converter._save_preflight_checkpoint(
    checkpoint,
    args=_test_args(),
    collection_root=collection,
    plans=[plan],
    phase_names=("approach",),
  )
  restored, phases, restored_fingerprint = converter._load_preflight_checkpoint(
    checkpoint,
    args=_test_args(),
    collection_root=collection,
  )
  assert restored_fingerprint == fingerprint
  assert phases == ("approach",)
  assert restored[0].source.hdf5_path == hdf5
  assert restored[0].exported_frames == 3

  hdf5_stat = hdf5.stat()
  hdf5.write_bytes(b"not actually HDF6")
  os.utime(hdf5, ns=(hdf5_stat.st_atime_ns, hdf5_stat.st_mtime_ns))
  with pytest.raises(ValueError, match="raw source changed"):
    converter._load_preflight_checkpoint(
      checkpoint, args=_test_args(), collection_root=collection,
    )
  hdf5.write_bytes(b"not actually HDF5")
  os.utime(hdf5, ns=(hdf5_stat.st_atime_ns, hdf5_stat.st_mtime_ns))

  sidecar_stat = sidecar.stat()
  sidecar.write_text("[]")
  os.utime(sidecar, ns=(sidecar_stat.st_atime_ns, sidecar_stat.st_mtime_ns))
  with pytest.raises(ValueError, match="raw source changed"):
    converter._load_preflight_checkpoint(
      checkpoint, args=_test_args(), collection_root=collection,
    )


def test_completed_episode_checkpoint_uses_atomic_artifact_contract(tmp_path: Path) -> None:
  source = converter.SourceEpisode(
    episode_index=7,
    hdf5_path=Path("episode.h5"),
    sidecar_path=Path("episode.json"),
    relative_hdf5_path="episode.h5",
    hdf5_sha256="2" * 64,
    hdf5_size_bytes=1,
    state_samples=10,
    camera_samples=4,
    metadata={},
    outcome={},
  )
  plan = converter.EpisodePlan(
    source=source,
    camera_samples=4,
    strict_prefix_frames=4,
    exported_frames=3,
    off_grid_terminal_frames=0,
    maximum_grid_error_seconds=0.0,
    image_height=240,
    image_width=320,
    source_timestamp_start_s=0.0,
    source_timestamp_end_s=0.1,
    phase_names=("approach",),
  )
  root = tmp_path / "episodes" / "episode-000000"
  root.mkdir(parents=True)
  names = ("data.parquet", "stats.json", "head.mp4", "right_wrist.mp4")
  for name in names:
    (root / name).write_bytes(name.encode())
  done = {
    "checkpoint_schema": converter.CHECKPOINT_SCHEMA,
    "plan_fingerprint": "fingerprint",
    "output_episode_index": 0,
    "source_episode_index": 7,
    "source_hdf5_sha256": "2" * 64,
    "exported_frames": 3,
    "global_start_index": 0,
    "artifacts": {
      name: {
        "size_bytes": (root / name).stat().st_size,
        "sha256": converter._sha256_file(root / name),
      }
      for name in names
    },
  }
  converter._write_json(root / "done.json", done)
  assert converter._completed_episode(
    tmp_path, plan, 0, 0, "fingerprint",
  ) == done

  (root / "head.mp4").write_bytes(b"same-len")
  with pytest.raises(ValueError, match="artifact is invalid"):
    converter._completed_episode(tmp_path, plan, 0, 0, "fingerprint")

  (root / "head.mp4").write_bytes(b"changed-size")
  with pytest.raises(ValueError, match="artifact is invalid"):
    converter._completed_episode(tmp_path, plan, 0, 0, "fingerprint")
