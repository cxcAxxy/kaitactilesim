"""Small synthetic raw-stream corruption checks; no simulation or rendering."""

from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.usb_cleaning import (
  CleaningThresholds,
  audit_usb_cleaning,
  audit_usb_insertion_physics,
  write_transition_indices,
)


def _stream(file, path, values):
  values = np.asarray(values)
  if values.dtype.kind in "UO":
    return file.create_dataset(
      path,
      data=values.astype(object),
      dtype=h5py.string_dtype(),
      maxshape=(None, *values.shape[1:]),
    )
  return file.create_dataset(path, data=values, maxshape=(None, *values.shape[1:]))


@pytest.fixture
def raw(tmp_path):
  path = tmp_path / "usb.h5"
  n = 103
  time = np.arange(n) * 0.002
  cached_time = time.copy()
  qpos = np.zeros((n, 8))
  qpos[:, 0] = time * 0.01
  qpos[:, 1] = time * 0.001
  qpos[:, 4] = 1
  qvel = np.zeros((n, 7))
  qvel[:, 0] = 0.01
  qvel[:, 1] = 0.001
  pose = qpos[:, 1:].copy()
  camera_indices = np.array([0, 50, 100, 102])
  with h5py.File(path, "w") as file:
    file.attrs.update(
      physics_hz=500,
      control_hz=500,
      camera_hz=10,
      metadata_json=json.dumps(
        {
          "recording_contract": "usb_insert_taskspace_raw_v1",
          "observation_clock": "post_step_forward_v1",
          "episode_index": 0,
        }
      ),
      outcome_json=json.dumps({"success": True}),
    )
    _stream(file, "state/timestamp", time)
    _stream(file, "state/qpos", qpos)
    _stream(file, "state/qvel", qvel)
    _stream(file, "state/robot_joint_position", qpos[:, :1])
    _stream(file, "state/robot_joint_velocity", qvel[:, :1])
    _stream(file, "state/robot_joint_effort", np.zeros((n, 1)))
    file.create_dataset(
      "state/joint_names", data=["right_arm_joint1"], dtype=h5py.string_dtype()
    )
    file.create_dataset(
      "state/full_qpos_names",
      data=[
        "right_arm_joint1",
        *[
          f"usb_plug_freejoint/{suffix}"
          for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")
        ],
      ],
      dtype=h5py.string_dtype(),
    )
    file.create_dataset(
      "state/full_qvel_names",
      data=[
        "right_arm_joint1",
        *[
          f"usb_plug_freejoint/{suffix}"
          for suffix in ("vx", "vy", "vz", "wx", "wy", "wz")
        ],
      ],
      dtype=h5py.string_dtype(),
    )
    _stream(file, "commands/actuator_control", 0.1 + time[:, None])
    _stream(file, "commands/arm_joint_target", np.zeros((n, 1)))
    _stream(file, "commands/hand_joint_target", np.zeros((n, 1)))
    _stream(file, "commands/phase", np.full(n, "approach"))
    _stream(file, "physics/solver_timestamp", cached_time)
    _stream(file, "physics/qacc", np.zeros((n, 7)))
    _stream(file, "physics/qfrc_applied", np.zeros((n, 7)))
    _stream(file, "physics/xfrc_applied", np.zeros((n, 2, 6)))
    _stream(file, "physics/actuator_force", np.zeros((n, 1)))
    _stream(file, "physics/noslip_iterations", np.zeros(n, dtype=int))
    _stream(file, "tactile_contact_force/timestamp", cached_time)
    for key, shape in {
      "normal_force_n": (),
      "normal_force_world_n": (3,),
      "force_world_n": (3,),
      "tangent_force_world_n": (3,),
      "tangent_force_n": (2,),
      "tangent_load_n": (),
      "normal_taxel_force_n": (7, 5),
      "tangent_taxel_force_n": (7, 5, 2),
      "tangent_taxel_load_n": (7, 5),
      "tangent_basis_world": (2, 3),
      "normal_axis_world": (3,),
      "contact_count": (),
    }.items():
      _stream(file, f"tactile_contact_force/{key}", np.zeros((n, 10, *shape)))
    _stream(file, "objects/usb_plug/pose_wxyz", pose)
    _stream(file, "objects/usb_plug/twist_linear_angular", np.zeros((n, 6)))
    _stream(file, "contacts/frame_start", np.zeros(n, dtype=np.int64))
    _stream(file, "contacts/frame_count", np.zeros(n, dtype=np.int32))
    _stream(file, "contacts/events/state_index", np.zeros(0, dtype=np.int64))
    for key, values in {
      "time_s": time[[20, 30]],
      "state_index_at_or_before_issue": np.array([20, 30]),
      "first_future_state_index": np.array([21, 31]),
      "arm_goal_rad": np.zeros((2, 7)),
    }.items():
      _stream(file, f"control/precontact_noise/{key}", values)
    _stream(file, "cameras/head/timestamp", time[camera_indices])
    _stream(file, "cameras/head/state_index", camera_indices)
    _stream(file, "cameras/head/pose_timestamp", cached_time[camera_indices])
    _stream(
      file,
      "cameras/head/rgb",
      np.stack([np.full((4, 4, 3), k, dtype=np.uint8) for k in range(4)]),
    )
    for key, tail in (
      ("world_from_camera", ()),
      ("world_from_wrist", (2,)),
      ("world_from_fingertip", (2, 5)),
    ):
      values = np.broadcast_to(np.eye(4), (4, *tail, 4, 4)).copy()
      _stream(file, f"cameras/head/{key}", values)
  return path


def test_valid_complete_clock_and_terminal_short_frame(raw):
  report = audit_usb_cleaning(raw)
  assert report["valid"], report["errors"]
  assert report["warnings"] == []
  assert report["checks"]["transitions"]["count"] == 102
  camera = report["checks"]["cameras"]["head"]
  assert camera["frames"] == camera["expected_frames"] == 4
  assert camera["terminal_short_interval_is_expected"]
  assert camera["last_interval_s"] == pytest.approx(0.004)
  assert camera["maximum_capture_minus_render_s"] == 0
  assert not report["checks"]["clock"]["reset_solver_timestamp_duplicate_is_expected"]
  assert report["checks"]["clock"]["observation_clock"] == "post_step_forward_v1"
  legacy = report["checks"]["insertion_physics"]
  assert legacy["status"] == "legacy_physics_gate_not_available"
  assert legacy["recorded_physics_gate"]["checked"] is False
  assert legacy["recorded_physics_gate"]["valid"] is None


@pytest.fixture
def raw_30hz(raw):
  # Independently enumerated real 500 Hz observations plus 204 ms terminal.
  indices = np.array([0, 17, 33, 50, 67, 83, 100, 102])
  with h5py.File(raw, "r+") as file:
    file.attrs["camera_hz"] = 30
    group = file["cameras/head"]
    for dataset in group.values():
      original = dataset[:]
      dataset.resize(len(indices), axis=0)
      dataset[:] = original[0]
    group["state_index"][:] = indices
    group["timestamp"][:] = file["state/timestamp"][:][indices]
    group["pose_timestamp"][:] = group["timestamp"][:]
    group["rgb"][:] = np.arange(8, dtype=np.uint8)[:, None, None, None]
  return raw


def test_30hz_real_quantized_camera_grid_and_exact_terminal_pass(raw_30hz):
  report = audit_usb_cleaning(raw_30hz)
  assert report["valid"], report["errors"]
  camera = report["checks"]["cameras"]["head"]
  assert camera["frames"] == camera["expected_frames"] == 8
  assert camera["terminal_short_interval_is_expected"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_tick", "fake_clock"])
def test_30hz_camera_corruption_still_fails(raw_30hz, damage):
  with h5py.File(raw_30hz, "r+") as file:
    group = file["cameras/head"]
    if damage in ("missing", "duplicate"):
      for dataset in group.values():
        values = dataset[:]
        values = np.delete(values, 2, axis=0) if damage == "missing" else values
        if damage == "duplicate":
          values[2] = values[1]
        dataset.resize(len(values), axis=0)
        dataset[:] = values
    elif damage == "wrong_tick":
      group["state_index"][2] = 34
      group["timestamp"][2] = group["pose_timestamp"][2] = 0.068
    else:
      group["timestamp"][1] = 1 / 30
  report = audit_usb_cleaning(raw_30hz)
  assert not report["valid"]
  assert any("camera head" in error for error in report["errors"])


@pytest.fixture
def spring_raw(raw):
  """Known scalar evidence for a 50-sample bottom preload, without simulation."""
  with h5py.File(raw, "r+") as file:
    times = file["state/timestamp"][:]
    n = len(times)
    metadata = json.loads(file.attrs["metadata_json"])
    metadata["contact_model_version"] = "passive_spring_shoes_bottom_out_v2"
    file.attrs["metadata_json"] = json.dumps(metadata)
    group = file.create_group("usb_insertion")
    group.attrs.update(
      schema_version="usb_insertion_monitor_v1",
      contact_model_version=metadata["contact_model_version"],
      observation_clock="post_step_forward_v1",
      state_index_reference="/state/timestamp",
    )
    _stream(group, "timestamp", times)
    _stream(group, "state_index", np.arange(n))
    metrics = {
      "insertion_depth_m": 0.012,
      "axial_resistance_n": 0.18,
      "backstop_axial_resistance_n": 0.18,
      "spring_axial_resistance_n": 0.0,
      "spring_normal_load_n": 0.7,
      "wall_normal_load_n": 0.0,
      "backstop_normal_load_n": 0.18,
      "axial_speed_m_s": 0.0,
      "linear_speed_m_s": 0.0,
      "angular_speed_rad_s": 0.0,
      "maximum_socket_penetration_m": 0.00001,
      "orientation_error_rad": 0.0,
    }
    terminal = dict(metrics)
    for name, value in metrics.items():
      values = np.full(n, value)
      values[0] = 0
      if name in (
        "axial_resistance_n",
        "backstop_axial_resistance_n",
        "backstop_normal_load_n",
      ):
        values[1:51] = 0.9
      _stream(group, name, values)
    for name in (
      "seated",
      "success",
      "shell_fits_aperture",
      "backstop_contact",
      "bottom_out_confirmed",
    ):
      start = 51 if name == "success" else 50 if name == "bottom_out_confirmed" else 1
      values = np.arange(n) >= start
      _stream(group, name, values)
      terminal[name] = True
    phases = np.full(n, "verify", dtype=object)
    phases[0] = "insert"
    phases[1:51] = "bottom_out"
    phases[51] = "release"
    phases[-1] = "terminal_settle"
    file["commands/phase"][:] = phases
    file.attrs["outcome_json"] = json.dumps(
      {
        "success": True,
        "active_bottom_out_confirmed": True,
        "bottom_out_hold_s": 0.1,
        "insertion": terminal,
      }
    )
  return raw


def test_v2_integrity_and_physics_gates_are_distinct_and_count_each_sample(spring_raw):
  report = audit_usb_cleaning(spring_raw)
  assert report["valid"], report["errors"]
  gate = report["checks"]["insertion_physics"]
  assert gate["pipeline_integrity"]["valid"]
  assert gate["recorded_physics_gate"]["valid"]
  assert gate["recorded_physics_gate"][
    "maximum_continuous_bottom_hold_s"
  ] == pytest.approx(0.1)
  assert gate["recorded_physics_gate"]["first_release_state_index"] == 51
  assert gate["recorded_physics_gate"]["bottom_out_latch_reconstructed"]


def test_v2_verified_bottom_history_allows_static_unloaded_seating(spring_raw):
  with h5py.File(spring_raw, "r+") as file:
    group = file["usb_insertion"]
    outcome = json.loads(file.attrs["outcome_json"])
    for name in ("backstop_axial_resistance_n", "backstop_normal_load_n"):
      group[name][51:] = 0
      outcome["insertion"][name] = 0.0
    group["backstop_contact"][51:] = False
    outcome["insertion"]["backstop_contact"] = False
    file.attrs["outcome_json"] = json.dumps(outcome)
  report = audit_usb_cleaning(spring_raw)
  assert report["valid"], report["errors"]
  gate = report["checks"]["insertion_physics"]["recorded_physics_gate"]
  assert gate["bottom_out_latch_reconstructed"]
  assert gate["bottom_out_confirmation_retained_at_end"]
  assert gate["terminal_backstop_axial_resistance_n"] == 0


@pytest.mark.parametrize("damage", ["premature", "missing", "escaped_then_returned"])
def test_v2_bottom_history_is_independently_reconstructed(spring_raw, damage):
  with h5py.File(spring_raw, "r+") as file:
    group = file["usb_insertion"]
    if damage == "premature":
      group["bottom_out_confirmed"][49] = True
    elif damage == "missing":
      group["bottom_out_confirmed"][50] = False
    else:
      group["insertion_depth_m"][60] = 0.0118
      group["seated"][60] = False
      # The original bottom-out boolean is deliberately left latched after
      # leaving the valid depth region: source reconstruction must reject it.
  report = audit_usb_cleaning(spring_raw)
  assert not report["valid"]
  gate = report["checks"]["insertion_physics"]["recorded_physics_gate"]
  assert gate["bottom_out_latch_reconstructed"] is False
  assert any("reconstructed contact history" in error for error in report["errors"])


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("missing_group", "stream integrity"),
    ("missing_metric", "scalar rows"),
    ("short_stream", "scalar rows"),
    ("shifted_timestamp", "timestamps differ"),
    ("duplicate_index", "indices are missing"),
    ("nonfinite", "nonfinite"),
    ("wrong_group_version", "contact_model_version"),
  ],
)
def test_v2_insertion_stream_corruption_fails_integrity(spring_raw, damage, expected):
  with h5py.File(spring_raw, "r+") as file:
    group = file["usb_insertion"]
    if damage == "missing_group":
      del file["usb_insertion"]
    elif damage == "missing_metric":
      del group["linear_speed_m_s"]
    elif damage == "short_stream":
      group["spring_normal_load_n"].resize((102,))
    elif damage == "shifted_timestamp":
      group["timestamp"][25] += 0.002
    elif damage == "duplicate_index":
      group["state_index"][25] = 24
    elif damage == "nonfinite":
      group["backstop_axial_resistance_n"][25] = np.nan
    else:
      group.attrs["contact_model_version"] = "legacy"
  report = audit_usb_cleaning(spring_raw)
  assert not report["valid"]
  gate = report["checks"]["insertion_physics"]
  assert gate["pipeline_integrity"]["valid"] is False
  assert gate["recorded_physics_gate"]["checked"] is False
  assert any(expected in error for error in report["errors"]), report["errors"]


@pytest.mark.parametrize(
  "damage,expected",
  [
    ("low_force", "continuous pre-release"),
    ("interrupted_force", "continuous pre-release"),
    ("only_49_samples", "continuous pre-release"),
    ("early_release", "before the first release"),
    ("confirmation_missing", "active_bottom_out_confirmed"),
    ("unsupported_declared_hold", "bottom_out_hold_s"),
    ("velocity_mismatch", "recorded object velocity"),
    ("moving_but_seated", "physical seating criteria"),
    ("false_bottom_contact", "positive bottom load"),
    ("premature_success", "continuous seating dwell"),
    ("terminal_mismatch", "final monitor row"),
  ],
)
def test_v2_plausible_aligned_stream_can_still_fail_physics(
  spring_raw, damage, expected
):
  with h5py.File(spring_raw, "r+") as file:
    group = file["usb_insertion"]
    outcome = json.loads(file.attrs["outcome_json"])
    if damage == "low_force":
      group["backstop_axial_resistance_n"][1:51] = 0.79
    elif damage == "interrupted_force":
      group["backstop_axial_resistance_n"][25] = 0.79
    elif damage == "only_49_samples":
      file["commands/phase"][1] = "insert"
    elif damage == "early_release":
      file["commands/phase"][25] = "release"
    elif damage == "confirmation_missing":
      outcome.pop("active_bottom_out_confirmed")
    elif damage == "unsupported_declared_hold":
      outcome["bottom_out_hold_s"] = 0.2
    elif damage == "velocity_mismatch":
      group["linear_speed_m_s"][25] = 0.003
    elif damage == "moving_but_seated":
      file["objects/usb_plug/twist_linear_angular"][25, 2] = -0.003
      group["linear_speed_m_s"][25] = group["axial_speed_m_s"][25] = 0.003
    elif damage == "false_bottom_contact":
      group["backstop_axial_resistance_n"][25] = 0
    elif damage == "premature_success":
      group["success"][50] = True
    else:
      outcome["insertion"]["backstop_normal_load_n"] = 9.0
    file.attrs["outcome_json"] = json.dumps(outcome)
  report = audit_usb_cleaning(spring_raw)
  assert not report["valid"]
  gate = report["checks"]["insertion_physics"]
  assert gate["pipeline_integrity"]["valid"] is True
  assert gate["recorded_physics_gate"]["valid"] is False
  assert any(expected in error for error in report["errors"]), report["errors"]


def test_unknown_contact_model_never_silently_receives_legacy_acceptance(raw):
  with h5py.File(raw, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    metadata["contact_model_version"] = "unknown"
    report = audit_usb_insertion_physics(file, metadata)
  assert not report["valid"]
  assert "unsupported USB contact_model_version" in report["errors"][0]


def test_v2_contact_version_cannot_be_removed_to_skip_its_gate(spring_raw):
  with h5py.File(spring_raw, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    metadata.pop("contact_model_version")
    file.attrs["metadata_json"] = json.dumps(metadata)
  report = audit_usb_cleaning(spring_raw)
  assert not report["valid"]
  assert any("contact_model_version is missing" in error for error in report["errors"])


@pytest.mark.parametrize("clock", [None, "pre_integration_v1"])
def test_legacy_observation_clock_is_rejected_even_if_arrays_look_aligned(raw, clock):
  with h5py.File(raw, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    if clock is None:
      metadata.pop("observation_clock")
    else:
      metadata["observation_clock"] = clock
    file.attrs["metadata_json"] = json.dumps(metadata)
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any(
    "observation_clock must be post_step_forward_v1" in error
    for error in report["errors"]
  )


def test_post_step_clock_tag_does_not_allow_legacy_shifted_solver_times(raw):
  with h5py.File(raw, "r+") as file:
    time = file["state/timestamp"][:]
    legacy_time = np.r_[0, time[:-1]]
    file["physics/solver_timestamp"][:] = legacy_time
    file["tactile_contact_force/timestamp"][:] = legacy_time
    camera_indices = file["cameras/head/state_index"][:]
    file["cameras/head/pose_timestamp"][:] = legacy_time[camera_indices]
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("same-row post-step state" in error for error in report["errors"])


@pytest.mark.parametrize(
  "dataset,index,value,message",
  [
    ("state/timestamp", 10, 0.018, "advance exactly one"),
    ("state/timestamp", 10, 0.022, "advance exactly one"),
    ("physics/solver_timestamp", 10, 0.018, "same-row post-step"),
    ("tactile_contact_force/timestamp", 10, 0.018, "same-row solver"),
    ("cameras/head/timestamp", 1, 0.0, "duplicated or reversed"),
    ("cameras/head/pose_timestamp", 1, 0.098, "same-row solver"),
    ("cameras/head/state_index", 1, 49, "referenced state"),
    (
      "control/precontact_noise/first_future_state_index",
      0,
      20,
      "future state indices",
    ),
    ("control/precontact_noise/state_index_at_or_before_issue", 0, 19, "at-or-before"),
  ],
)
def test_timestamp_and_command_index_corruption(raw, dataset, index, value, message):
  with h5py.File(raw, "r+") as file:
    file[dataset][index] = value
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any(message in error for error in report["errors"])


def test_missing_rgb_frame_detected_even_with_remaining_valid_time_order(raw):
  with h5py.File(raw, "r+") as file:
    for dataset in file["cameras/head"].values():
      values = dataset[[0, 2, 3]]
      dataset.resize(3, axis=0)
      dataset[:] = values
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("frame missing/extra" in error for error in report["errors"])


def test_truncated_command_rows_detected(raw):
  with h5py.File(raw, "r+") as file:
    file["commands/actuator_control"].resize(102, axis=0)
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("expected 103" in error for error in report["errors"])


def test_missing_tactile_stream_does_not_silently_pass(raw):
  with h5py.File(raw, "r+") as file:
    del file["tactile_contact_force/tangent_taxel_force_n"]
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any(
    "missing required stream: tactile_contact_force/tangent_taxel_force_n" in error
    for error in report["errors"]
  )


def test_requested_arm_goal_is_not_mistaken_for_applied_control(raw):
  with h5py.File(raw, "r+") as file:
    file["commands/arm_joint_target"][50, 0] = 0.5
  report = audit_usb_cleaning(raw)
  assert report["valid"], report["errors"]
  assert (
    "post-step contact latch" in report["checks"]["transitions"]["goal_stream_note"]
  )


def test_position_jump_caught_independent_of_named_duplicate_streams(raw):
  with h5py.File(raw, "r+") as file:
    file["state/qpos"][35, 0] += 0.001
    file["state/robot_joint_position"][35, 0] += 0.001
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("destination velocity integration" in error for error in report["errors"])


def test_object_preintegration_cache_shift_detected(raw):
  with h5py.File(raw, "r+") as file:
    qpos = file["state/qpos"][:, 1:]
    file["objects/usb_plug/pose_wxyz"][:] = np.vstack((qpos[0], qpos[:-1]))
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("post-step epoch" in error for error in report["errors"])


def test_quaternion_sign_changes_do_not_create_false_jumps(raw):
  with h5py.File(raw, "r+") as file:
    dataset = file["objects/usb_plug/pose_wxyz"]
    values = dataset[:]
    values[::2, 3:] *= -1
    dataset[:] = values
  report = audit_usb_cleaning(raw)
  assert report["valid"], report["errors"]
  assert report["checks"]["trajectory"]["usb_orientation_step_rad"]["maximum"] == 0


def test_static_duplicate_pixels_are_reported_but_allowed(raw):
  with h5py.File(raw, "r+") as file:
    file["cameras/head/rgb"][:] = 0
  report = audit_usb_cleaning(raw)
  assert report["valid"], report["errors"]
  assert report["checks"]["cameras"]["head"]["identical_adjacent_rgb_count"] == 3
  assert not report["review_required"]


def test_duplicate_pixels_with_task_motion_require_review(raw):
  with h5py.File(raw, "r+") as file:
    file["cameras/head/rgb"][:] = 0
    pose = file["cameras/head/world_from_wrist"][:]
    pose[:, 1, 0, 3] = np.arange(4) * 0.003
    file["cameras/head/world_from_wrist"][:] = pose
  report = audit_usb_cleaning(raw)
  assert report["valid"], report["errors"]
  assert report["review_required"]
  assert len(report["checks"]["cameras"]["head"]["suspicious_identical_rgb_pairs"]) == 3


def test_nonfinite_state_fails_closed_and_json_remains_serializable(raw):
  with h5py.File(raw, "r+") as file:
    file["state/qpos"][20, 0] = np.nan
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  json.dumps(report, allow_nan=False)


def test_nonfinite_taxel_fails_even_when_aggregate_force_is_finite(raw):
  with h5py.File(raw, "r+") as file:
    file["tactile_contact_force/tangent_taxel_force_n"][25, 0, 2, 3, 0] = np.nan
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("tangent_taxel_force_n: nonfinite" in error for error in report["errors"])
  json.dumps(report, allow_nan=False)


def test_bad_camera_rotation_fails_closed(raw):
  with h5py.File(raw, "r+") as file:
    file["cameras/head/world_from_camera"][1, 0, 0] = 2
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("not SO(3)" in error for error in report["errors"])


def test_failed_episode_is_not_counted_as_pipeline_acceptance(raw):
  with h5py.File(raw, "r+") as file:
    file.attrs["outcome_json"] = json.dumps({"success": False})
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert "episode is not marked successfully completed" in report["errors"]


def test_bad_contact_event_ranges_are_detected(raw):
  with h5py.File(raw, "r+") as file:
    file["contacts/frame_count"][50] = 1
  report = audit_usb_cleaning(raw)
  assert not report["valid"]
  assert any("noncontiguous" in error for error in report["errors"])


def test_transition_sidecar_uses_destination_control_rows_and_latest_logging_time(
  raw, tmp_path
):
  output = tmp_path / "transitions.npz"
  write_transition_indices(raw, output)
  with np.load(output) as indices:
    assert np.array_equal(
      indices["actuator_control_index"], indices["state_t_index"] + 1
    )
    assert np.array_equal(
      indices["actuator_control_index"], indices["state_t_plus_1_index"]
    )
    assert np.array_equal(
      indices["actuator_control_logged_timestamp_ns"],
      indices["state_t_plus_1_timestamp_ns"],
    )
    assert np.all(
      indices["state_t_timestamp_ns"] < indices["state_t_plus_1_timestamp_ns"]
    )
  with pytest.raises(FileExistsError):
    write_transition_indices(raw, output)


def test_thresholds_must_be_positive_and_finite(raw):
  with pytest.raises(ValueError, match="finite and positive"):
    audit_usb_cleaning(raw, CleaningThresholds(joint_speed_rad_s=float("nan")))


def test_batch_cli_rejects_duplicate_content_and_never_overwrites(raw, tmp_path):
  script = (
    Path(__file__).resolve().parents[1] / "scripts/workcell/check_usb_cleaning.py"
  )
  main = run_path(str(script))["main"]
  copy = tmp_path / "copy.h5"
  copy.write_bytes(raw.read_bytes())
  output = tmp_path / "checks"
  with pytest.raises(SystemExit) as error:
    main(["--input", str(raw), "--input", str(copy), "--output-dir", str(output)])
  assert error.value.code == 1
  summary = json.loads((output / "summary.json").read_text())
  assert not summary["valid"]
  assert "duplicate episode content" in summary["errors"][0]
  with pytest.raises(SystemExit) as error:
    main(["--input", str(raw), "--output-dir", str(output)])
  assert error.value.code == 2


@pytest.mark.parametrize("preload_samples,valid", [(50, False), (75, True)])
def test_v3_gentle_preload_requires_longer_continuous_dwell(
  spring_raw, preload_samples, valid
):
  with h5py.File(spring_raw, "r+") as file:
    version = "passive_spring_shoes_bottom_out_v3"
    metadata = json.loads(file.attrs["metadata_json"])
    metadata["contact_model_version"] = version
    file.attrs["metadata_json"] = json.dumps(metadata)
    group = file["usb_insertion"]
    group.attrs["contact_model_version"] = version
    for key in (
      "axial_resistance_n",
      "backstop_axial_resistance_n",
      "backstop_normal_load_n",
    ):
      values = group[key][:]
      values[1:] = 0.18
      values[1 : preload_samples + 1] = 0.65
      group[key][:] = values
    n = len(group["timestamp"])
    group["bottom_out_confirmed"][:] = np.arange(n) >= 75
    phases = np.full(n, "verify", dtype=object)
    phases[0] = "insert"
    phases[1 : preload_samples + 1] = "bottom_out"
    phases[preload_samples + 1] = "release"
    phases[-1] = "terminal_settle"
    file["commands/phase"][:] = phases
    outcome = json.loads(file.attrs["outcome_json"])
    outcome["bottom_out_hold_s"] = preload_samples * 0.002
    file.attrs["outcome_json"] = json.dumps(outcome)
  report = audit_usb_cleaning(spring_raw)
  assert report["valid"] is valid, report["errors"]
  gate = report["checks"]["insertion_physics"]["recorded_physics_gate"]
  assert gate["bottom_force_threshold_n"] == 0.6
  assert gate["required_bottom_hold_s"] == 0.15
