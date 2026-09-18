"""Independent small-data T-ICT release and action-convention regressions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_export as exporter
from kaihand_tactile_env.shared import tict_validation as validator

SESSION = "synthetic_middle_001"


def _rot_z(angle):
  cosine, sine = np.cos(angle), np.sin(angle)
  return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def _poses(count=51):
  cameras = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
  wrists = np.broadcast_to(np.eye(4), (count, 2, 4, 4)).copy()
  relative = np.broadcast_to(np.eye(4), (count, 2, 5, 4, 4)).copy()
  for frame in range(count):
    cameras[frame, :3, :3] = _rot_z(0.2 + frame * 0.01)
    cameras[frame, :3, 3] = [0.1 + 0.01 * frame, -0.2, 1.0]
    for side in range(2):
      wrists[frame, side, :3, :3] = _rot_z(0.4 + frame * 0.02 + side * 0.3)
      wrists[frame, side, :3, 3] = [1 + 0.03 * frame, -0.1 + 0.2 * side, 0.5]
      for finger in range(5):
        relative[frame, side, finger, :3, :3] = _rot_z(0.1 * finger + 0.005 * frame)
        relative[frame, side, finger, :3, 3] = [
          0.01 * (finger + 1),
          0.002 * frame,
          0.02,
        ]
  return cameras, wrists, relative


def _source(tmp_path: Path, count=51):
  path = tmp_path / "synthetic.h5"
  cameras, wrists, relative = _poses(count)
  timestamps = np.arange(count, dtype=np.float64) * 0.1
  poststep = timestamps + 0.002
  poststep[0] = timestamps[0]
  with h5py.File(path, "w") as file:
    file.attrs["schema_version"] = "kaihand_tactile_episode_v1"
    file.attrs["physics_hz"] = 500
    file.attrs["control_hz"] = 100
    file.attrs["camera_hz"] = 10
    file.attrs["contact_force_source"] = "mujoco_solver_distributed_contact_v1"
    file.attrs["metadata_json"] = json.dumps(
      {"scene": "poker-draw", "preset": "middle-force-v1"}
    )
    file.attrs["outcome_json"] = json.dumps(
      {
        "success": True,
        "object_name": "card",
        "phases": ["slide_card", "inspect_card", "terminal_settle"],
        "terminal_stability": {
          "elapsed_seconds": 0.1,
          "stable_seconds": 0.1,
          "linear_speed": 0.0,
          "angular_speed": 0.0,
          "steps": 50,
        },
        "edge_outcome": {
          "target_reached": True,
          "held_at_edge": True,
          "full_slide_qualified": True,
        },
        "handoff_outcome": {"completed": True},
      }
    )
    state = file.create_group("state")
    state.create_dataset("timestamp", data=poststep)
    card = file.create_group("objects/card")
    card.create_dataset(
      "pose_wxyz", data=np.tile([0.58, -0.16, 0.84, 1.0, 0.0, 0.0, 0.0], (count, 1))
    )
    commands = file.create_group("commands")
    commands.create_dataset(
      "phase",
      data=["slide_card"] * (count - 1) + ["terminal_settle"],
      dtype=h5py.string_dtype(),
    )
    group = file.create_group("cameras/head")
    group.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
    group.attrs["side_names_json"] = json.dumps(["left", "right"])
    group.attrs["finger_names_json"] = json.dumps(
      ["thumb", "index", "middle", "ring", "little"]
    )
    group.attrs["wrist_sites_json"] = json.dumps(
      ["hand_l_base_link_site", "hand_r_base_link_site"]
    )
    group.attrs["fingertip_sites_json"] = json.dumps(
      [
        [
          f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}_site"
          for finger in ("thumb", "index", "middle", "ring", "pinky")
        ]
        for side in ("l", "r")
      ]
    )
    group.create_dataset(
      "rgb",
      shape=(count, 240, 320, 3),
      dtype=np.uint8,
      chunks=(1, 240, 320, 3),
      compression="gzip",
    )
    group.create_dataset("pose_timestamp", data=timestamps)
    group.create_dataset("timestamp", data=poststep)
    group.create_dataset("state_index", data=np.arange(count, dtype=np.int64))
    # Stored MuJoCo c2w becomes the intended CV convention after conversion.
    group.create_dataset(
      "world_from_camera", data=cameras @ exporter.CV_FROM_MUJOCO_CAMERA
    )
    group.create_dataset("world_from_wrist", data=wrists)
    group.create_dataset("world_from_fingertip", data=wrists[:, :, None] @ relative)
    group.create_dataset(
      "intrinsic", data=np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1]])
    )
    force = file.create_group("tactile_contact_force")
    force.attrs["force_unit"] = "N"
    force.attrs["is_spatial_estimate"] = True
    force.attrs["timestamp_reference"] = "/tactile_contact_force/timestamp"
    force.create_dataset("timestamp", data=timestamps)
    force.create_dataset(
      "link_names", data=exporter._finger_link_names(), dtype=h5py.string_dtype()
    )
    normal = np.full((count, 10, 7, 5), 0.01)
    # A real zero reading remains observed, not marked missing.
    normal[:, 0] = 0
    force.create_dataset("normal_taxel_force_n", data=normal)
    shear = np.zeros((count, 10, 7, 5, 2))
    shear[..., 0] = 0.002
    shear[..., 1] = -0.003
    force.create_dataset("tangent_taxel_force_n", data=shear)
    force.create_dataset("normal_axis_local", data=np.tile([0.0, 0.0, 1.0], (10, 1)))
    force.create_dataset(
      "tangent_basis_local",
      data=np.tile([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (10, 1, 1)),
    )
  return path


def _release(tmp_path):
  source = _source(tmp_path)
  destination = tmp_path / "release"
  exporter.export_tict_episode(source, destination, session_id=SESSION)
  return destination


def _frame(root, frame=0):
  return (
    root
    / "production"
    / SESSION
    / "09_humanego_adapter/preprocess/all_data"
    / f"{frame:05d}"
  )


def _read_json(path):
  return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
  path.write_text(json.dumps(payload), encoding="utf-8")


def test_synthetic_hdf5_exports_and_independent_release_validator_accepts(tmp_path):
  root = _release(tmp_path)
  report = validator.validate_tict_release(root)
  assert report["valid"], report["errors"]
  assert report["upstream_loader_verified"] is False
  assert report["sessions"][0]["frame_count"] == 51
  assert report["sessions"][0]["window_count"] == 1
  assert report["sessions"][0]["action_shape"] == [50, 108]
  assert _read_json(root / "split_manifest.json")["splits"] == {
    "train": [SESSION],
    "validation": [],
    "test": [],
  }
  assert _read_json(root / "window_starts.json")["sessions"][SESSION] == [0]
  assert not _read_json(_frame(root) / "training_data.json")["metadata"]["is_finished"]
  assert _read_json(_frame(root, 50) / "training_data.json")["metadata"]["is_finished"]
  with np.load(
    root / "tict_sidecars" / SESSION / "fingertip_tactile_v1.npz", allow_pickle=False
  ) as sidecar:
    assert sidecar["tactile_channel_mask"].all()
    np.testing.assert_allclose(sidecar["tactile_mean"][:, 0, 0, 0], 0)
    np.testing.assert_allclose(sidecar["tactile_mean"][..., 1], 0.002)
    np.testing.assert_allclose(sidecar["tactile_mean"][..., 2], -0.003)
    _, _, expected_relative = _poses()
    np.testing.assert_allclose(
      sidecar["T_fingertip_to_wrist"], expected_relative, atol=1e-12
    )


@pytest.mark.parametrize(
  "damage, expected",
  [
    ("rgb_hash", "hash"),
    ("reflection", "reflection"),
    ("future_tactile", "noncausal"),
    ("sidecar_hash", "sidecar digest"),
    ("tail_start", "tail"),
    ("missing_frame", "No such file"),
    ("session_leakage", "leakage"),
  ],
)
def test_release_validator_rejects_corruption(tmp_path, damage, expected):
  root = _release(tmp_path)
  if damage == "rgb_hash":
    path = _frame(root) / "rgb.png"
    path.write_bytes(path.read_bytes() + b"unexpected trailing bytes")
  elif damage == "reflection":
    path = _frame(root) / "training_data.json"
    value = _read_json(path)
    reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
    value["entities"]["hands_hawor_v3"]["right"]["T_hand_to_world"] = (
      reflection.tolist()
    )
    _write_json(path, value)
  elif damage in {"future_tactile", "sidecar_hash"}:
    path = root / "tict_sidecars" / SESSION / "fingertip_tactile_v1.npz"
    with np.load(path, allow_pickle=False) as sidecar:
      values = {name: sidecar[name] for name in sidecar.files}
    if damage == "future_tactile":
      values["tactile_sync_error_ns"][0] = -1
    else:
      values["tactile_mean"][0, 0, 0, 0] = 0.001
    np.savez_compressed(path, **values)
    if damage == "future_tactile":
      audit_path = root / "dataset_audit.json"
      audit = _read_json(audit_path)
      audit["sessions"][0]["sidecar_sha256"] = exporter.sha256_file(path)
      _write_json(audit_path, audit)
  elif damage == "tail_start":
    path = root / "window_starts.json"
    value = _read_json(path)
    value["sessions"][SESSION].append(1)
    _write_json(path, value)
  elif damage == "missing_frame":
    (_frame(root, 20) / "training_data.json").unlink()
  else:
    path = root / "split_manifest.json"
    value = _read_json(path)
    value["splits"]["validation"] = [SESSION]
    _write_json(path, value)
  report = validator.validate_tict_release(root)
  assert not report["valid"]
  assert any(expected.lower() in error.lower() for error in report["errors"]), report


def test_moving_camera_actions_freeze_current_camera_and_use_future_finger_frame():
  cameras, wrists, relative = _poses()
  action = exporter.build_action_window(wrists, relative, cameras, start=0)
  independent = validator.build_action_window(wrists[1:], relative[1:], cameras[0])
  assert action.shape == (50, 108)
  np.testing.assert_allclose(action, independent, atol=2e-7)
  slots = action.reshape(50, 2, 6, 9)
  expected_wrist = np.linalg.inv(cameras[0]) @ wrists[1:]
  np.testing.assert_allclose(slots[:, :, 0, :3], expected_wrist[..., :3, 3], atol=2e-7)
  wrong_future_camera = np.linalg.inv(cameras[1:])[:, None] @ wrists[1:]
  assert not np.allclose(slots[:, :, 0, :3], wrong_future_camera[..., :3, 3])
  np.testing.assert_allclose(slots[:, :, 1:, :3], relative[1:, ..., :3, 3], atol=2e-7)
  world_fingers = wrists[:, :, None] @ relative
  wrong_current_wrist = np.linalg.inv(wrists[0])[None, :, None] @ world_fingers[1:]
  assert not np.allclose(slots[:, :, 1:, :3], wrong_current_wrist[..., :3, 3])
  # Rotation6D is column 0 followed by column 1, not row flattening.
  np.testing.assert_allclose(slots[:, :, 0, 3:6], expected_wrist[..., :3, 0], atol=2e-7)
  np.testing.assert_allclose(slots[:, :, 0, 6:9], expected_wrist[..., :3, 1], atol=2e-7)


def test_exporter_rejects_legacy_missing_true_pose_stream_without_publishing(tmp_path):
  source = _source(tmp_path)
  with h5py.File(source, "r+") as file:
    del file["cameras/head/world_from_fingertip"]
  destination = tmp_path / "release"
  with pytest.raises(ValueError, match="full-pose"):
    exporter.export_tict_episode(source, destination, session_id=SESSION)
  assert not destination.exists()


def test_exporter_never_replaces_existing_release(tmp_path):
  root = _release(tmp_path)
  marker = root / "DATA_CONTRACT.md"
  before = marker.read_bytes()
  with pytest.raises(FileExistsError):
    exporter.export_tict_episode(tmp_path / "synthetic.h5", root, session_id=SESSION)
  assert marker.read_bytes() == before


@pytest.mark.parametrize(
  "damage, expected",
  [
    ("unknown_schema", "taskspace"),
    ("side_swap", "ordering"),
    ("unfinished", "terminal_settle"),
    ("reflection", "SO\\(3\\)"),
    ("no_terminal_frame", "terminal_settle"),
  ],
)
def test_source_preflight_rejects_ambiguous_pose_order_or_terminal_state(
  tmp_path, damage, expected
):
  source = _source(tmp_path)
  with h5py.File(source, "r+") as file:
    camera = file["cameras/head"]
    if damage == "unknown_schema":
      camera.attrs["taskspace_schema"] = "xyz_only"
    elif damage == "side_swap":
      camera.attrs["side_names_json"] = json.dumps(["right", "left"])
    elif damage == "unfinished":
      file["commands/phase"][-1] = "slide_card"
    elif damage == "no_terminal_frame":
      camera["state_index"][-1] = 49
    else:
      camera["world_from_wrist"][0, 0] = np.diag([-1.0, 1.0, 1.0, 1.0])
  destination = tmp_path / "release"
  with pytest.raises(ValueError, match=expected):
    exporter.export_tict_episode(source, destination, session_id=SESSION)
  assert not destination.exists()


def test_tactile_sync_never_uses_future_sample_and_masks_stale_observations(tmp_path):
  source = _source(tmp_path)
  with h5py.File(source, "r") as file:
    synchronized = exporter._sync_tactile(
      file["tactile_contact_force"],
      np.array([0, 50_000_000, 100_000_000, 150_000_000], dtype=np.int64),
      20_000_000,
    )
  np.testing.assert_array_equal(synchronized["tactile_source_index"], [0, 0, 1, 1])
  np.testing.assert_array_equal(
    synchronized["tactile_sync_error_ns"], [0, 50_000_000, 0, 50_000_000]
  )
  np.testing.assert_array_equal(
    synchronized["tactile_frame_valid"], [True, False, True, False]
  )
  assert not synchronized["tactile_channel_mask"][[1, 3]].any()
  assert not synchronized["tactile_mean"][[1, 3]].any()
