"""Tiny array-only checks of randomized capture provenance; no robot loading."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_source_audit as audit


@pytest.fixture
def accepted_source(tmp_path, monkeypatch):
  controller = tmp_path / "controller.py"
  controller.write_text("# synthetic controller fixture\n", encoding="utf-8")
  scene = tmp_path / "scene.xml"
  scene.write_text("<mujoco/>\n", encoding="utf-8")
  monkeypatch.setattr(
    audit, "_known_source_paths", lambda preset: ({"controller.py": controller}, scene)
  )
  metadata = {
    "scene": "poker-draw",
    "preset": "middle-force-v1",
    "press_force_per_finger_n": 0.5,
    "object_xy_jitter": 0.0,
    "object_yaw_jitter": 0.0,
    "base_model_sha256": audit._sha256(scene),
    "preset_settings": {
      "press_force_per_finger_n": 0.5,
      "pressure_window": {
        "table_friction": 1.0,
        "drive_limit_n": 4.0,
        "slide_speed_m_s": 0.005,
      },
      "observation_noise": None,
      "action_noise": None,
      "controller_source_sha256": {"controller.py": audit._sha256(controller)},
    },
    "contact_model": {
      "physics_timestep_used_s": 0.002,
      "contact_friction_impedance_ratio_used": 100,
      "add_genesis_probes": False,
      "table_card_pair_friction": [1, 1, 0.005, 0.0005, 0.0005],
      "table_card_pair_solref_used": [0.01, 1],
      "card_geom_solref_used": [0.01, 1],
    },
  }
  outcome = {
    "success": True,
    "slide_press_control_qualified": True,
    "edge_outcome": {
      "target_reached": True,
      "held_at_edge": True,
      "full_slide_qualified": True,
    },
    "handoff_outcome": {
      "completed": True,
      "transition": {
        "maximum_qpos_change": 0.0,
        "maximum_qvel_change": 0.0,
        "control_jump_rad": 0.0,
        "object_state_modified": False,
        "physics_parameters_modified": False,
      },
    },
  }
  file = SimpleNamespace(
    attrs={"physics_hz": 500, "model_sha256": "a" * 64, "model_fingerprint": "b" * 64}
  )
  return file, metadata, outcome, controller


def test_existing_zero_preset_remains_accepted(accepted_source):
  file, metadata, outcome, _ = accepted_source
  errors = []
  report = audit._physics_and_identity(file, metadata, outcome, errors)
  assert errors == []
  assert report["controller_source_status"] == "matches_current_code"
  assert report["drift_source_names"] == []


def test_training_admission_keeps_failed_pressure_label_without_bypassing_task(accepted_source):
  file, metadata, outcome, _ = accepted_source
  metadata = deepcopy(metadata)
  outcome = deepcopy(outcome)
  metadata["acceptance_policy"] = "task-completion-v1"
  outcome.update(
    acceptance_policy="task-completion-v1", task_completed=True,
    terminal_pinch=True, sustained_pinch=True, retained_at_end=True,
    half_overhang_reached=True, simultaneous_four_finger_contact=True,
    slide_press_control_qualified=False, pressure_quality="completed_with_force_variation",
    maximum_slide_fingertip_plane_angle_degrees=14.5,
    inspection_face_alignment=.9, inspection_face_robot_alignment=.9,
    inspection_position_error=.01, minimum_supported_card_clearance=-.0001,
  )
  outcome["edge_outcome"].update(full_slide_qualified=False, slide_task_completed=True,
                                 slide_geometry_qualified=True, terminal_reason="edge_reached")
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert errors == []
  assert outcome["slide_press_control_qualified"] is False
  outcome["terminal_pinch"] = False
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert any("physical task" in item for item in errors)


@pytest.mark.parametrize("name", ["object_xy_jitter", "object_yaw_jitter"])
@pytest.mark.parametrize("value", [0.001, -0.001, float("nan"), float("inf")])
def test_existing_zero_preset_still_rejects_jitter(accepted_source, name, value):
  file, metadata, outcome, _ = accepted_source
  metadata = deepcopy(metadata)
  metadata[name] = value
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert any("jitter" in error for error in errors)


def test_source_drift_is_reported_separately_but_still_strict(accepted_source):
  file, metadata, outcome, controller = accepted_source
  controller.write_text("# later controller revision\n", encoding="utf-8")
  errors = []
  report = audit._physics_and_identity(file, metadata, outcome, errors)
  assert report["controller_source_status"] == "current_code_drift"
  assert report["drift_source_names"] == ["controller.py"]
  assert len(errors) == 1
  assert "current-code drift" in errors[0]


def test_randomized_source_list_adds_only_randomization_module():
  baseline, scene = audit._known_source_paths("middle-force-v1")
  randomized, randomized_scene = audit._known_source_paths("middle-force-randomized-v1")
  assert len(baseline) == 14
  assert "poker_draw/acceptance.py" in baseline
  assert randomized_scene == scene
  assert set(randomized) - set(baseline) == {"poker_draw/randomization.py"}
  assert all(randomized[name] == path for name, path in baseline.items())


def test_archival_audit_preserves_drift_without_rejecting_valid_data(accepted_source):
  file, metadata, outcome, controller = accepted_source
  controller.write_text("# later revision\n", encoding="utf-8")
  errors = []
  report = audit._physics_and_identity(file, metadata, outcome, errors, source_code_policy="recorded")
  assert errors == []
  assert report["controller_source_status"] == "current_code_drift"
  assert "not independently verified" in report["source_identity_scope"]
  outcome["handoff_outcome"]["completed"] = False
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors, source_code_policy="recorded")
  assert errors


@pytest.mark.parametrize("hashes", [{}, {"controller.py": "bad"}, {"unknown.py": "a" * 64}])
def test_archival_audit_rejects_invalid_provenance(accepted_source, hashes):
  file, metadata, outcome, _ = accepted_source
  metadata["preset_settings"]["controller_source_sha256"] = hashes
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors, source_code_policy="recorded")
  assert errors


def test_archival_legacy_hash_list_only_allows_missing_acceptance(accepted_source, monkeypatch):
  file, metadata, outcome, controller = accepted_source
  scene = controller.with_name("scene.xml")
  monkeypatch.setattr(audit, "_known_source_paths", lambda preset: (
    {"controller.py": controller, "poker_draw/acceptance.py": controller}, scene))
  errors = []
  report = audit._physics_and_identity(file, metadata, outcome, errors, source_code_policy="recorded")
  assert errors == []
  assert report["drift_source_names"] == ["poker_draw/acceptance.py"]
  metadata["acceptance_policy"] = "task-completion-v1"
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors, source_code_policy="recorded")
  assert "controller source hash list incomplete or unexpected" in errors


class _ArraySource(SimpleNamespace):
  def __getitem__(self, name):
    return self.streams[name]


@pytest.fixture
def randomized_source(accepted_source, tmp_path):
  file, metadata, outcome, controller = accepted_source
  scene = tmp_path / "scene.xml"
  known_scene = Path(audit.__file__).resolve().parents[1] / "tasks/poker_draw/scene.xml"
  scene.write_text(known_scene.read_text(encoding="utf-8"), encoding="utf-8")
  metadata = deepcopy(metadata)
  metadata.update(
    preset="middle-force-randomized-v1",
    seed=14,
    object_xy_jitter=0.002,
    object_yaw_jitter=float(np.pi / 180),
    base_model_sha256=audit._sha256(scene),
  )
  metadata["preset_settings"].update(
    object_xy_jitter_m=metadata["object_xy_jitter"],
    object_yaw_jitter_rad=metadata["object_yaw_jitter"],
    robot_initial_state_noise=None,
  )
  rng = np.random.default_rng(metadata["seed"])
  xy = rng.uniform(-0.002, 0.002, 2)
  yaw = rng.uniform(-np.pi / 180, np.pi / 180)
  nominal = np.array([0.58, -0.16, 0.84175, 1, 0, 0, 0])
  sampled = nominal.copy()
  sampled[:2] += xy
  sampled[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
  card_extent_x = abs(np.cos(yaw)) * 0.0315 + abs(np.sin(yaw)) * 0.044
  card_extent_y = abs(np.sin(yaw)) * 0.0315 + abs(np.cos(yaw)) * 0.044
  margin = min(
    0.115 - abs(sampled[0] - 0.58) - card_extent_x,
    0.070 - abs(sampled[1] + 0.18) - card_extent_y,
  )
  metadata["initial_card_randomization"] = {
    "schema_version": "poker-initial-card-randomization-v1",
    "mode": "seeded_uniform",
    "distribution": "independent_uniform",
    "seed": metadata["seed"],
    "xy_jitter_m": metadata["object_xy_jitter"],
    "yaw_jitter_rad": metadata["object_yaw_jitter"],
    "nominal_pose_wxyz": nominal.tolist(),
    "sampled_pose_wxyz": sampled.tolist(),
    "sampled_offset_xy_m": xy.tolist(),
    "sampled_yaw_offset_rad": float(yaw),
    "perturbation_scope": "reset_only_card_xy_yaw",
    "height_unchanged": True,
    "tilt_unchanged": True,
    "observation_noise": None,
    "action_noise": None,
    "table_support": {
      "valid": True,
      "minimum_corner_margin_xy_m": float(margin),
      "bottom_gap_m": 0.00020,
    },
  }
  file = _ArraySource(
    attrs=file.attrs,
    streams={
      "objects/card/pose_wxyz": sampled[None],
      "state/timestamp": np.array([0.0]),
    },
  )
  return file, metadata, outcome, controller


def test_randomized_capture_is_independently_reconstructed(randomized_source):
  file, metadata, outcome, _ = randomized_source
  errors = []
  report = audit._physics_and_identity(file, metadata, outcome, errors)
  assert errors == []
  variation = report["initial_card_randomization"]
  assert variation["first_raw_pose_verified"]
  assert variation["nominal_scene_pose_verified"]
  assert variation["seed"] == 14
  assert variation["minimum_corner_margin_xy_m"] > 0.00025


@pytest.mark.parametrize(
  "path,value,expected",
  [
    (["seed"], 15, "seed"),
    (["initial_card_randomization", "seed"], True, "seed"),
    (["initial_card_randomization", "seed"], -1, "seed"),
    (["initial_card_randomization", "seed"], 15, "seed"),
    (["initial_card_randomization", "mode"], "fixed_validation_offset", "mode"),
    (["initial_card_randomization", "distribution"], "gaussian", "distribution"),
    (["initial_card_randomization", "xy_jitter_m"], 0.006, "bounds"),
    (["initial_card_randomization", "yaw_jitter_rad"], 0.018, "bounds"),
    (["initial_card_randomization", "xy_jitter_m"], float("nan"), "bounds"),
    (["initial_card_randomization", "yaw_jitter_rad"], -0.01, "bounds"),
    (["initial_card_randomization", "sampled_offset_xy_m"], [0, 0], "XY offsets"),
    (["initial_card_randomization", "sampled_yaw_offset_rad"], 0, "yaw"),
    (["initial_card_randomization", "observation_noise"], {}, "observation_noise"),
    (["initial_card_randomization", "action_noise"], 0.01, "action_noise"),
    (["initial_card_randomization", "height_unchanged"], False, "height_unchanged"),
    (["initial_card_randomization", "tilt_unchanged"], False, "tilt_unchanged"),
    (["initial_card_randomization", "table_support", "valid"], False, "support"),
    (["initial_card_randomization", "table_support", "bottom_gap_m"], 0, "support"),
    (["preset_settings", "robot_initial_state_noise"], {}, "initial state noise"),
    (["preset_settings", "object_xy_jitter_m"], 0, "bounds"),
    (["preset_settings", "object_yaw_jitter_rad"], 0, "bounds"),
    (["object_xy_jitter"], 0, "bounds"),
    (["object_yaw_jitter"], 0, "bounds"),
  ],
)
def test_randomized_provenance_tampering_is_rejected(
  randomized_source, path, value, expected
):
  file, metadata, outcome, _ = randomized_source
  metadata = deepcopy(metadata)
  target = metadata
  for key in path[:-1]:
    target = target[key]
  target[path[-1]] = value
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
  "name,index", [("nominal_pose_wxyz", 0), ("sampled_pose_wxyz", 2)]
)
def test_randomized_pose_metadata_must_match_known_scene_and_sample(
  randomized_source, name, index
):
  file, metadata, outcome, _ = randomized_source
  metadata = deepcopy(metadata)
  metadata["initial_card_randomization"][name][index] += 0.0001
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert any("pose differs" in error for error in errors), errors


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6])
def test_first_raw_card_pose_cannot_disagree_with_randomization(
  randomized_source, index
):
  file, metadata, outcome, _ = randomized_source
  file.streams["objects/card/pose_wxyz"][0, index] += 0.0001
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert errors


def test_randomized_source_must_retain_reset_frame(randomized_source):
  file, metadata, outcome, _ = randomized_source
  file.streams["state/timestamp"][0] = 0.002
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert any("exact reset sample" in error for error in errors)


def test_quaternion_sign_does_not_change_physical_anchor(randomized_source):
  file, metadata, outcome, _ = randomized_source
  file.streams["objects/card/pose_wxyz"][0, 3:] *= -1
  errors = []
  audit._physics_and_identity(file, metadata, outcome, errors)
  assert errors == []


def test_randomized_anchor_changes_translation_and_rotation(randomized_source):
  _, metadata, _, _ = randomized_source
  variation = metadata["initial_card_randomization"]
  nominal = audit._card_pose_transform(variation["nominal_pose_wxyz"])
  sampled = audit._card_pose_transform(variation["sampled_pose_wxyz"])
  assert not np.allclose(nominal[:2, 3], sampled[:2, 3])
  assert not np.allclose(nominal[:3, :3], sampled[:3, :3])
  assert np.allclose(sampled[:3, :3] @ sampled[:3, :3].T, np.eye(3))
  assert sampled[2, 3] == nominal[2, 3]


def test_randomized_release_and_source_anchor_cross_check(tmp_path, randomized_source):
  # Reuse only the independent synthetic raw fixture, not any production helper.
  from kaihand_tactile_env.shared import tict_export, tict_validation
  from test_tict_contract import SESSION, _source

  source = _source(tmp_path)
  array_file, metadata, outcome, _ = randomized_source
  with h5py.File(source, "r+") as file:
    file.attrs.update(array_file.attrs)
    file.attrs["metadata_json"] = json.dumps(metadata)
    # Preserve terminal label metadata required by the converter.
    old_outcome = json.loads(file.attrs["outcome_json"])
    old_outcome.update(outcome)
    file.attrs["outcome_json"] = json.dumps(old_outcome)
    file["objects/card/pose_wxyz"][:] = np.tile(
      array_file.streams["objects/card/pose_wxyz"][0], (51, 1)
    )
    commands = file["commands"]
    commands["phase"][:] = (
      ["slide_card"] * 25
      + ["edge_hold"] * 6
      + ["inspect_card"] * 19
      + ["terminal_settle"]
    )
    active = np.arange(51) < 31
    commands.create_dataset("drive_budget_active", data=active)
    commands.create_dataset("drive_actual_fx_n", data=active.astype(float))
    commands.create_dataset("drive_requested_fx_n", data=active.astype(float))
    commands.create_dataset("drive_limit_n", data=4 * active.astype(float))
    commands.create_dataset("actuator_control", data=np.zeros((51, 1)))
    commands.create_dataset(
      "actuator_names", data=["synthetic"], dtype=h5py.string_dtype()
    )
    force = file["tactile_contact_force"]
    force.create_dataset(
      "normal_force_n", data=force["normal_taxel_force_n"][:].sum(axis=(2, 3))
    )
    force.create_dataset(
      "tangent_force_n", data=force["tangent_taxel_force_n"][:].sum(axis=(2, 3))
    )
  release = tmp_path / "release"
  tict_export.export_tict_episode(source, release, session_id=SESSION)
  local = tict_validation.validate_tict_release(release)
  assert local["valid"], local["errors"]
  report = audit.audit_tict_source(source, release, SESSION)
  assert report["valid"], report["errors"]
  assert report["cross_check"]["frames_checked"] == 51
  assert report["physical_gates_and_identity"]["initial_card_randomization"][
    "first_raw_pose_verified"
  ]
  frame = (
    release
    / "production"
    / SESSION
    / "09_humanego_adapter/preprocess/all_data/00000/training_data.json"
  )
  document = json.loads(frame.read_text(encoding="utf-8"))
  original = audit._card_pose_transform(
    metadata["initial_card_randomization"]["nominal_pose_wxyz"]
  )
  assert not np.allclose(
    document["metadata"]["world_transforms"]["virtual_static_anchor"], original
  )
  # A stale baseline anchor is still a legal SE(3); only the source cross-check
  # detects that it does not describe this randomized session's actual reset.
  document["metadata"]["world_transforms"]["virtual_static_anchor"] = original.tolist()
  frame.write_text(json.dumps(document), encoding="utf-8")
  report = audit.audit_tict_source(source, release, SESSION)
  assert not report["valid"]
  assert report["cross_check"]["mismatch_counts"]["json"] == 1
