"""Synthetic USB release/provenance regressions; no simulator or renderer."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_export as exporter
from kaihand_tactile_env.shared import tict_source_audit as auditor
from kaihand_tactile_env.shared import tict_validation as validator

SESSION = "synthetic_usb_001"


@pytest.mark.parametrize(
  "declared,phase,depth,margin,accepted",
  [
    (True, "align", -0.10, 12.0, True),
    (True, "align", -0.10, 9.0, False),
    (True, "align", -0.03, 12.0, False),
    (True, "insert", -0.10, 12.0, False),
    (False, "align", -0.10, 12.0, False),
    (True, "align", -0.002, 16.0, True),
  ],
)
def test_usb_high_hover_margin_exception_is_limited_to_declared_high_alignment(
  usb_source, declared, phase, depth, margin, accepted
):
  with h5py.File(usb_source, "r+") as file:
    outcome = _metadata(file, "outcome_json")
    if declared:
      outcome["motion_parameters"] = {"align_clearance_m": 0.10, "align_hold_s": 0.6}
    count = len(file["state/timestamp"])
    depths = np.zeros(count)
    depths[30] = depth
    file.create_dataset("usb_insertion/insertion_depth_m", data=depths)
    file["commands/phase"][30] = phase
    for key in ("state/qpos", "commands/arm_joint_target"):
      file[key][30, 7] = np.pi - np.deg2rad(margin)
    errors = []
    auditor._usb_recorded_motion(
      file, np.rint(file["state/timestamp"][:] * 1e9).astype(np.int64), outcome, errors
    )
    margin_errors = [error for error in errors if "joint margin" in error]
    assert (not margin_errors) == accepted
    if not accepted:
      assert len(margin_errors) == 2


def _json_attribute(group, name, value):
  group.attrs[name] = json.dumps(value)


def _metadata(file, name):
  return json.loads(file.attrs[name])


@pytest.fixture
def usb_source(tmp_path, monkeypatch):
  # The independent source audit sees a fixed tiny code/scene snapshot. Real
  # source-list coverage is checked separately below; these are synthetic bytes.
  controller = tmp_path / "controller.py"
  controller.write_text("# synthetic controller\n", encoding="utf-8")
  scene = tmp_path / "scene.xml"
  scene.write_text('<mujoco><include file="robot.xml"/></mujoco>', encoding="utf-8")
  (tmp_path / "robot.xml").write_text("<mujoco/>", encoding="utf-8")
  monkeypatch.setattr(
    auditor, "_known_usb_source_paths", lambda: ({"controller.py": controller}, scene)
  )
  hashes = {"controller.py": exporter.sha256_file(controller)}
  count, frames = 101, 51
  state_times = np.arange(count) * 0.002
  solver_times = state_times.copy()
  camera_indices = np.arange(frames) * 2
  quaternion = [0, -np.cos(np.pi / 12), np.sin(np.pi / 12), 0]
  poses = np.tile([0.5, -0.18, 0.6865, *quaternion], (count, 1))
  poses[:, 0] += np.arange(count) * 0.0001
  names = [f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)]
  metadata = {
    "scene": "usb-insert",
    "recording_contract": "usb_insert_taskspace_raw_v1",
    "observation_clock": "post_step_forward_v1",
    "motion_profile": "fast",
    "controller_source_sha256": hashes,
    "base_model_sha256": exporter.sha256_file(scene),
    "base_model_fingerprint": auditor._mjcf_source_fingerprint(scene),
    "initial_pose_randomization": {"initial_pose_wxyz": poses[0].tolist()},
  }
  outcome = {
    "success": True,
    "object_name": "usb_plug",
    "released": True,
    "grasp_verified": True,
    "source_files_unchanged": True,
    "controller_source_sha256_at_end": hashes,
    "motion_profile": "fast",
    "insertion": {"success": True, "seated": True, "stable_duration_s": 0.12},
    "final_object_pose": poses[-1].tolist(),
    "final_object_twist": [0.0] * 6,
    "terminal_stability": {
      "stable_seconds": 0.1,
      "linear_speed": 0.0,
      "angular_speed": 0.0,
    },
  }
  source = tmp_path / "usb.h5"
  with h5py.File(source, "w") as file:
    file.attrs.update(
      schema_version="kaihand_tactile_episode_v1",
      physics_hz=500,
      control_hz=500,
      camera_hz=250,
      model_sha256="a" * 64,
      model_fingerprint="b" * 64,
      contact_force_source="solver_contact_distributed_taxel_v1",
    )
    _json_attribute(file, "metadata_json", metadata)
    _json_attribute(file, "outcome_json", outcome)
    state = file.create_group("state")
    state.create_dataset("timestamp", data=state_times)
    state.create_dataset("qpos", data=np.zeros((count, 14)))
    state.create_dataset("full_qpos_names", data=names, dtype=h5py.string_dtype())
    state.create_dataset(
      "full_qvel_names",
      data=names
      + [f"usb_plug_freejoint/{name}" for name in ("vx", "vy", "vz", "wx", "wy", "wz")],
      dtype=h5py.string_dtype(),
    )
    model = file.create_group("model")
    model.create_dataset("arm_joint_names", data=names, dtype=h5py.string_dtype())
    model.create_dataset("arm_joint_limits_rad", data=np.tile([-np.pi, np.pi], (14, 1)))
    model.create_dataset("arm_joint_qpos_indices", data=np.arange(14))
    model.create_dataset(
      "body_names",
      data=[
        "world",
        "table",
        "usb_plug",
        "hand_r_index_link4",
        "usb_fixture",
        "usb_socket",
      ],
      dtype=h5py.string_dtype(),
    )
    command = file.create_group("commands")
    command.create_dataset(
      "phase",
      data=["insert"] * (count - 1) + ["terminal_settle"],
      dtype=h5py.string_dtype(),
    )
    command.create_dataset("arm_joint_names", data=names, dtype=h5py.string_dtype())
    command.create_dataset("arm_joint_target", data=np.zeros((count, 14)))
    physics = file.create_group("physics")
    physics.create_dataset("solver_timestamp", data=solver_times)
    gravity_feedforward = np.zeros((count, 20))
    gravity_feedforward[:, :14] = 0.25
    physics.create_dataset("qfrc_applied", data=gravity_feedforward)
    physics.create_dataset("xfrc_applied", data=np.zeros((count, 6, 6)))
    plug = file.create_group("objects/usb_plug")
    plug.create_dataset("pose_wxyz", data=poses)
    plug.create_dataset("twist_linear_angular", data=np.zeros((count, 6)))
    contacts = file.create_group("contacts")
    contacts.create_dataset("frame_start", data=np.arange(count, dtype=np.int64))
    contacts.create_dataset("frame_count", data=np.ones(count, dtype=np.int32))
    events = contacts.create_group("events")
    events.create_dataset("state_index", data=np.arange(count, dtype=np.int64))
    events.create_dataset("body1_id", data=np.full(count, 1, dtype=np.int32))
    events.create_dataset("body2_id", data=np.full(count, 2, dtype=np.int32))
    camera = file.create_group("cameras/head")
    camera.attrs["taskspace_schema"] = "kaihand-native-site-se3-v1"
    _json_attribute(camera, "side_names_json", ["left", "right"])
    _json_attribute(
      camera, "finger_names_json", ["thumb", "index", "middle", "ring", "little"]
    )
    camera.create_dataset(
      "rgb", data=np.zeros((frames, 240, 320, 3), dtype=np.uint8), compression="gzip"
    )
    camera.create_dataset("timestamp", data=state_times[camera_indices])
    camera.create_dataset("pose_timestamp", data=solver_times[camera_indices])
    camera.create_dataset("state_index", data=camera_indices)
    cameras = np.tile(np.eye(4), (frames, 1, 1))
    cameras[:, 0, 3] = np.arange(frames) * 0.001
    camera.create_dataset(
      "world_from_camera", data=cameras @ exporter.CV_FROM_MUJOCO_CAMERA
    )
    wrists = np.tile(np.eye(4), (frames, 2, 1, 1))
    wrists[:, 1, 0, 3] = 0.4 + np.arange(frames) * 0.001
    fingers = np.tile(np.eye(4), (frames, 2, 5, 1, 1))
    fingers[..., 2, 3] = np.arange(1, 6) * 0.01
    camera.create_dataset("world_from_wrist", data=wrists)
    camera.create_dataset("world_from_fingertip", data=wrists[:, :, None] @ fingers)
    camera.create_dataset("intrinsic", data=[[200, 0, 160], [0, 200, 120], [0, 0, 1]])
    force = file.create_group("tactile_contact_force")
    force.attrs.update(force_unit="N", is_spatial_estimate=True)
    _json_attribute(force, "metadata_json", {"target_geom_names": ["usb_plug_handle"]})
    force.create_dataset("timestamp", data=solver_times)
    force.create_dataset(
      "link_names", data=exporter._finger_link_names(), dtype=h5py.string_dtype()
    )
    normal = np.zeros((count, 10, 7, 5))
    normal[:, 5:7] = 0.01  # Thumb and index; the remaining measured zeros are valid.
    tangent = np.zeros((*normal.shape, 2))
    tangent[:, 5:7, ..., 0] = 0.002
    tangent[:, 5:7, ..., 1] = -0.003
    force.create_dataset("normal_taxel_force_n", data=normal)
    force.create_dataset("tangent_taxel_force_n", data=tangent)
    force.create_dataset("normal_force_n", data=normal.sum(axis=(2, 3)))
    force.create_dataset("tangent_force_n", data=tangent.sum(axis=(2, 3)))
    force.create_dataset("normal_axis_local", data=np.tile([0, 0, 1], (10, 1)))
    force.create_dataset(
      "tangent_basis_local", data=np.tile([[1, 0, 0], [0, 1, 0]], (10, 1, 1))
    )
  return source


def _export(source, name="release"):
  root = source.parent / name
  exporter.export_tict_episode(source, root, session_id=SESSION)
  return root


def _frame(root, index):
  return (
    root
    / "production"
    / SESSION
    / "09_humanego_adapter/preprocess/all_data"
    / f"{index:05d}"
    / "training_data.json"
  )


def test_usb_export_and_independent_audit_preserve_signed_taxels_and_static_pose(
  usb_source,
):
  root = _export(usb_source)
  report = validator.validate_tict_release(root)
  assert report["valid"], report["errors"]
  report = auditor.audit_tict_source(usb_source, root, SESSION)
  assert report["valid"], report["errors"]
  assert report["recorded_motion"]["all_physics_steps_recorded"]
  assert report["recorded_motion"]["robot_obstacle_contact_events"] == 0
  physics = report["physical_gates_and_identity"]["insertion_physics"]
  assert physics["status"] == "legacy_physics_gate_not_available"
  assert physics["recorded_physics_gate"]["checked"] is False
  assert report["clock"]["observation_clock"] == "post_step_forward_v1"
  assert report["clock"]["max_state_minus_solver_ns"] == 0
  assert report["clock"]["max_capture_minus_render_ns"] == 0
  assert report["session_id"] == SESSION
  assert report["release"]["sidecar_sha256"] == exporter.sha256_file(
    root / report["release"]["sidecar_path"]
  )
  assert report["release"]["dataset_audit_sha256"] == exporter.sha256_file(
    root / "dataset_audit.json"
  )
  assert "drive" not in report
  assert report["phase_forces"]["insert"]["right_finger_order"] == [
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
  ]
  documents = [json.loads(_frame(root, index).read_text()) for index in (0, 50)]
  for document in documents:
    assert document["entities"]["objects"] == {}
    anchor = np.array(document["metadata"]["world_transforms"]["virtual_static_anchor"])
    np.testing.assert_allclose(anchor[:3, 3], [0.5, -0.18, 0.6865])
    np.testing.assert_allclose(
      anchor[:3, :3],
      [[np.sqrt(3) / 2, -0.5, 0], [-0.5, -np.sqrt(3) / 2, 0], [0, 0, -1]],
      atol=1e-12,
    )
  with np.load(
    root / "tict_sidecars" / SESSION / "fingertip_tactile_v1.npz", allow_pickle=False
  ) as archive:
    metadata = json.loads(archive["source_metadata_json"].item())
    assert metadata["object_pose_source"] == "objects/usb_plug/pose_wxyz"
    assert metadata["static_anchor_timestamp_ns"] == 0
    assert "card contact" not in metadata["force_action"]
    np.testing.assert_allclose(
      archive["tactile_mean"][:, 1, :2], np.tile([0.01, 0.002, -0.003], (51, 2, 1))
    )
    assert archive["tactile_channel_mask"].all()
  assert "initial usb_plug pose" in (root / "DATA_CONTRACT.md").read_text()


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("wrong_object", "object_name"),
    ("not_released", "release"),
    ("not_seated", "seating"),
    ("not_reset", "time-zero"),
  ],
)
def test_usb_export_rejects_ambiguous_task_or_initial_pose(
  usb_source, damage, expected
):
  with h5py.File(usb_source, "r+") as file:
    outcome = _metadata(file, "outcome_json")
    if damage == "wrong_object":
      outcome["object_name"] = "card"
    elif damage == "not_released":
      outcome["released"] = False
    elif damage == "not_seated":
      outcome["insertion"]["seated"] = False
    else:
      file["state/timestamp"][0] = 0.001
    _json_attribute(file, "outcome_json", outcome)
  with pytest.raises(ValueError, match=expected):
    _export(usb_source)
  assert not (usb_source.parent / "release").exists()


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("actual_limit", "actual joint margin"),
    ("goal_limit", "goal joint margin"),
    ("collision", "raw contact stream"),
    ("spring_collision", "raw contact stream"),
    ("moving_terminal", "velocity window"),
    ("final_pose", "final_object_pose"),
    ("initial_pose", "initial pose provenance"),
    ("contact_target", "target set"),
    ("external_force", "force assistance"),
    ("object_generalized_force", "force assistance"),
    ("qpos_mapping", "full state joint names"),
    ("aggregate", "conservation"),
    ("source_drift", "current-code drift"),
    ("include_drift", "fingerprint"),
  ],
)
def test_usb_source_audit_rejects_physical_or_provenance_corruption(
  usb_source, damage, expected
):
  with h5py.File(usb_source, "r+") as file:
    outcome = _metadata(file, "outcome_json")
    if damage in {"actual_limit", "goal_limit"}:
      file["state/qpos" if damage == "actual_limit" else "commands/arm_joint_target"][
        30, 7
      ] = np.pi - 0.01
    elif damage == "collision":
      file["contacts/events/body2_id"][20] = 3
    elif damage == "spring_collision":
      file["model/body_names"][5] = "usb_socket_spring_top"
      file["contacts/events/body1_id"][20] = 5
      file["contacts/events/body2_id"][20] = 3
    elif damage == "moving_terminal":
      file["objects/usb_plug/twist_linear_angular"][-10, 0] = 0.03
    elif damage == "final_pose":
      outcome["final_object_pose"][0] += 0.01
    elif damage == "initial_pose":
      metadata = _metadata(file, "metadata_json")
      metadata["initial_pose_randomization"]["initial_pose_wxyz"][0] += 0.01
      _json_attribute(file, "metadata_json", metadata)
    elif damage == "contact_target":
      _json_attribute(
        file["tactile_contact_force"],
        "metadata_json",
        {"target_geom_names": ["card_core_geom"]},
      )
    elif damage == "external_force":
      file["physics/xfrc_applied"][20, 2, 0] = 0.1
    elif damage == "object_generalized_force":
      file["physics/qfrc_applied"][20, 14] = 0.1
    elif damage == "qpos_mapping":
      file["model/arm_joint_qpos_indices"][:] = np.r_[np.arange(7, 14), np.arange(7)]
    elif damage == "aggregate":
      file["tactile_contact_force/normal_force_n"][20, 5] += 0.1
    _json_attribute(file, "outcome_json", outcome)
  root = _export(usb_source)
  if damage == "source_drift":
    (usb_source.parent / "controller.py").write_text("# changed controller\n")
  elif damage == "include_drift":
    (usb_source.parent / "robot.xml").write_text("<mujoco><!-- changed --></mujoco>")
  report = auditor.audit_tict_source(usb_source, root, SESSION)
  assert not report["valid"]
  assert any(expected in error for error in report["errors"]), report


def test_v2_source_audit_requires_recorded_bottom_evidence(usb_source):
  with h5py.File(usb_source, "r+") as file:
    metadata = _metadata(file, "metadata_json")
    metadata["contact_model_version"] = "passive_spring_shoes_bottom_out_v2"
    _json_attribute(file, "metadata_json", metadata)
    outcome = _metadata(file, "outcome_json")
    # A declared success flag alone cannot replace the every-step evidence.
    outcome["active_bottom_out_confirmed"] = True
    outcome["bottom_out_hold_s"] = 0.1
    _json_attribute(file, "outcome_json", outcome)
  root = _export(usb_source)
  report = auditor.audit_tict_source(usb_source, root, SESSION)
  assert not report["valid"]
  physics = report["physical_gates_and_identity"]["insertion_physics"]
  assert physics["pipeline_integrity"]["valid"] is False
  assert physics["recorded_physics_gate"]["checked"] is False
  assert any("USB insertion stream integrity" in error for error in report["errors"])


def test_usb_validator_rejects_dynamic_object_used_as_static_anchor(usb_source):
  root = _export(usb_source)
  path = _frame(root, 25)
  document = json.loads(path.read_text())
  document["metadata"]["world_transforms"]["virtual_static_anchor"][0][3] += 0.01
  path.write_text(json.dumps(document))
  report = validator.validate_tict_release(root)
  assert not report["valid"]
  assert any("remain fixed" in error for error in report["errors"])


def test_usb_source_identity_covers_motion_collector_and_shared_code():
  paths, scene = auditor._known_usb_source_paths()
  assert scene.name == "scene.xml" and scene.parent.name == "usb_insert"
  assert "scripts/workcell/record_usb_dataset.py" in paths
  for name in ("motion", "recording", "execution", "grasp", "setup", "task"):
    assert f"src/kaihand_tactile_env/tasks/usb_insert/{name}.py" in paths
  assert "src/kaihand_tactile_env/shared/contact_tactile.py" in paths


def _damage_usb_observation_clock(source, damage):
  with h5py.File(source, "r+") as file:
    metadata = _metadata(file, "metadata_json")
    if damage == "missing_clock_version":
      del metadata["observation_clock"]
      _json_attribute(file, "metadata_json", metadata)
    elif damage == "wrong_clock_version":
      metadata["observation_clock"] = "pre_step_cache_v1"
      _json_attribute(file, "metadata_json", metadata)
    elif damage == "physics_only":
      file["physics/solver_timestamp"][10] += 0.001
    elif damage == "camera_pose_only":
      camera = file["cameras/head"]
      camera["pose_timestamp"][:] = np.maximum(0, camera["timestamp"][:] - 0.002)
    elif damage == "camera_acquisition_only":
      file["cameras/head/timestamp"][20] += 0.001
    elif damage == "camera_state_index_only":
      file["cameras/head/state_index"][20] += 1
    elif damage == "legacy_labels_with_new_version":
      # A version string does not cure the old self-consistent 2-ms early labels.
      old = np.maximum(0, file["state/timestamp"][:] - 0.002)
      file["physics/solver_timestamp"][:] = old
      file["tactile_contact_force/timestamp"][:] = old
      file["cameras/head/pose_timestamp"][:] = old[file["cameras/head/state_index"][:]]
    elif damage == "tactile_only":
      file["tactile_contact_force/timestamp"][:] = np.maximum(
        0, file["state/timestamp"][:] - 0.002
      )
    else:
      raise AssertionError(damage)


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("missing_clock_version", "observation_clock"),
    ("wrong_clock_version", "observation_clock"),
    ("physics_only", "USB physics solver clock"),
    ("tactile_only", "USB tactile solver clock"),
    ("legacy_labels_with_new_version", "USB tactile solver clock"),
    ("camera_pose_only", "USB camera pose/acquisition"),
    ("camera_acquisition_only", "USB camera pose/acquisition"),
    ("camera_state_index_only", "USB camera pose/acquisition"),
  ],
)
def test_usb_export_rejects_missing_or_mislabeled_post_step_clock(
  usb_source, damage, expected
):
  _damage_usb_observation_clock(usb_source, damage)
  with pytest.raises(ValueError, match=expected):
    _export(usb_source)
  assert not (usb_source.parent / "release").exists()


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("missing_clock_version", "observation_clock"),
    ("wrong_clock_version", "observation_clock"),
    ("physics_only", "USB physics solver clock"),
    ("tactile_only", "USB tactile solver clock"),
    ("legacy_labels_with_new_version", "USB tactile solver clock"),
    ("camera_pose_only", "USB camera pose clock"),
    ("camera_acquisition_only", "USB camera pose clock"),
    ("camera_state_index_only", "indexed post-step state"),
  ],
)
def test_usb_independent_audit_rejects_mislabeled_clock(usb_source, damage, expected):
  root = _export(usb_source)
  _damage_usb_observation_clock(usb_source, damage)
  report = auditor.audit_tict_source(usb_source, root, SESSION)
  assert not report["valid"]
  assert any(expected in error for error in report["errors"]), report["errors"]
