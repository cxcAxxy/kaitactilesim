"""Temporal corruption tests for offline force response checks; no simulation."""

import numpy as np
import pytest
from kaihand_tactile_env.shared.usb_action_audit import (
  _joint_qpos_labels,
  affine_actuator_force,
  arm_slew_alignment,
  audit_usb_actions,
  force_alignment,
  pose_alignment,
)


@pytest.mark.parametrize(
  "kind,expected",
  [
    (np.int32(3), ["joint"]),
    (np.int32(2), ["joint"]),
    (np.int32(1), ["joint/0", "joint/1", "joint/2", "joint/3"]),
    (
      np.int32(0),
      ["joint/x", "joint/y", "joint/z", "joint/qw", "joint/qx", "joint/qy", "joint/qz"],
    ),
  ],
)
def test_numpy_joint_type_scalar_uses_integer_enum_comparisons(kind, expected):
  # Real np.int32 vs pybind enum tuple membership used to reject hinge joints.
  # Importing enums allocates no model, data, renderer or simulation.
  assert _joint_qpos_labels("joint", kind) == expected


def example():
  # One position and one velocity actuator, with an intentionally saturated row.
  ctrl = np.array([[0.0, 0.0], [0.2, 0.4], [0.3, -0.2], [-0.1, 0.1]])
  pos = np.array([[0.0, 0.0], [0.01, 0.03], [0.04, 0.05], [0.02, 0.04]])
  vel = np.array([[0.0, 0.0], [0.2, 0.4], [0.3, -0.2], [-0.1, 0.1]])
  gain = np.array([10.0, 2.0])
  bias = np.array([[0.0, -10.0, -1.0], [0.0, 0.0, -2.0]])
  ranges = np.array([[-100.0, 100.0], [-0.5, 0.5]])
  force = np.zeros_like(ctrl)
  force[1:] = [[2.0, 0.5], [2.7, -0.5], [-1.7, 0.5]]
  return [ctrl, pos, vel, force, gain, bias, ranges]


def test_arm_slew_uses_initial_home_position_instead_of_reset_ctrl_placeholder():
  ctrl = np.array([[0.0], [2.09], [2.095]])
  report = arm_slew_alignment(ctrl, [2.09], [0.0, 0.002, 0.004])
  assert report["valid"]
  assert report["transitions"] == 2
  assert report["maximum_slew_rad_s"] == pytest.approx(2.5)


@pytest.mark.parametrize("row", [1, 2])
def test_arm_slew_rejects_large_first_or_later_real_command_jump(row):
  ctrl = np.array([[0.0], [2.09], [2.095]])
  ctrl[row, 0] += 0.1
  report = arm_slew_alignment(ctrl, [2.09], [0.0, 0.002, 0.004])
  assert not report["valid"]
  assert report["maximum_delta_excess_rad"] > 0.08


def test_control_uses_preceding_state_and_retains_force_saturation():
  report = force_alignment(*example())
  assert report["valid"]
  assert report["transitions"] == 3
  assert report["incorrect_post_step_state_maximum_residual"] > 0.1


def test_refreshed_post_step_force_requires_explicit_epoch_and_is_not_integration_force():
  arrays = example()
  # Same held control reevaluated after state integration by mj_forward.
  arrays[3][1:] = [[1.7, 0.0], [2.3, 0.0], [-1.1, 0.0]]
  assert not force_alignment(*arrays)["valid"]
  report = force_alignment(*arrays, state_epoch="post")
  assert report["valid"]
  assert report["state_epoch"] == "post"
  assert report["incorrect_pre_step_state_maximum_residual"] > 0.1
  assert "integration force was not archived" in report["force_semantics"]


def test_force_epoch_is_not_inferred_from_data():
  with pytest.raises(ValueError, match="never inferred"):
    force_alignment(*example(), state_epoch="auto")


def test_legacy_clock_is_rejected_before_model_compilation(tmp_path, monkeypatch):
  import json
  from types import SimpleNamespace

  import h5py
  import mujoco
  from kaihand_tactile_env.shared import tict_source_audit

  monkeypatch.setattr(
    tict_source_audit, "_known_usb_source_paths", lambda: ({}, tmp_path / "scene.xml")
  )
  monkeypatch.setattr(tict_source_audit, "_mjcf_source_fingerprint", lambda _: "model")

  def forbidden_compile(*args, **kwargs):
    raise AssertionError("legacy clock must fail before any model compilation")

  monkeypatch.setattr(
    mujoco, "MjModel", SimpleNamespace(from_xml_path=forbidden_compile)
  )
  path = tmp_path / "legacy.h5"
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps(
      {
        "controller_source_sha256": {},
        "base_model_fingerprint": "model",
      }
    )
  report = audit_usb_actions(path)
  assert not report["valid"]
  assert len(report["errors"]) == 1
  assert "post_step_forward_v1" in report["errors"][0]


@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_one_frame_shift_in_action_state_or_force_is_detected(field):
  arrays = example()
  arrays[field] = np.roll(arrays[field], 1, axis=0)
  assert not force_alignment(*arrays)["valid"]


def test_missing_frame_or_nonfinite_control_is_rejected():
  arrays = example()
  arrays[0] = arrays[0][:-1]
  with pytest.raises(ValueError, match="matching finite"):
    force_alignment(*arrays)
  arrays = example()
  arrays[0][1, 0] = np.nan
  with pytest.raises(ValueError, match="matching finite"):
    force_alignment(*arrays)


def test_force_equation_applies_gear_coordinate_bias_and_limits():
  result = affine_actuator_force(
    np.array([[1.0, 2.0]]),
    np.array([[0.2, 0.3]]),
    np.array([[0.1, 0.2]]),
    np.array([5.0, 6.0]),
    np.array([[0.1, -2.0, -0.3], [0.0, 0.0, -4.0]]),
    np.array([[-4.0, 4.0], [-100.0, 100.0]]),
  )
  np.testing.assert_allclose(result, [[4.0, 11.2]])


def test_engine_control_clamping_is_applied_before_force_equation():
  arrays = example()
  arrays[0][2, 0] = 30.0
  assert not force_alignment(*arrays)["valid"]
  report = force_alignment(
    *arrays,
    control_range=np.array([[-0.3, 0.3], [-np.inf, np.inf]]),
  )
  assert report["valid"]


@pytest.mark.parametrize(
  "field,replacement",
  [
    (4, np.array([np.nan, 2.0])),
    (4, np.array([10.0])),
    (5, np.zeros((2, 2))),
    (5, np.full((2, 3), np.inf)),
    (6, np.array([[-100.0, 100.0], [np.nan, 0.5]])),
    (6, np.array([[-100.0, 100.0], [1.0, -1.0]])),
  ],
)
def test_invalid_actuator_coefficients_do_not_silently_pass(field, replacement):
  arrays = example()
  arrays[field] = replacement
  with pytest.raises(ValueError):
    force_alignment(*arrays)


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -1.0])
def test_invalid_tolerance_is_rejected(tolerance):
  with pytest.raises(ValueError, match="tolerance"):
    force_alignment(*example(), tolerance=tolerance)


def test_empty_actuator_axis_is_rejected():
  arrays = example()
  arrays[:4] = [a[:, :0] for a in arrays[:4]]
  with pytest.raises(ValueError, match="matching finite"):
    force_alignment(*arrays)


def moving_pose_movie():
  # Three valid SE(3) frames: both hands translate 3 mm and turn 0.01 rad/frame.
  poses = np.broadcast_to(np.eye(4), (3, 2, 5, 4, 4)).copy()
  for frame, angle in enumerate((0.0, 0.01, 0.02)):
    c, s = np.cos(angle), np.sin(angle)
    poses[frame, ..., :3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    poses[frame, ..., 0, 3] = 0.003 * frame
  return poses


def test_pose_movie_one_frame_delay_is_detected_despite_valid_individual_se3():
  expected = moving_pose_movie()
  assert pose_alignment(expected.copy(), expected)["valid"]
  delayed = np.concatenate((expected[:1], expected[:-1]), axis=0)
  report = pose_alignment(delayed, expected)
  assert not report["valid"]
  assert report["frames_checked"] == 3
  assert report["poses_checked"] == 30
  assert report["mismatch_count"] == 20
  assert report["first_mismatches_frame_pose_indices"][0] == [1, 0, 0]
  assert report["maximum_translation_error_m"] == pytest.approx(0.003)
  assert report["maximum_rotation_matrix_error"] > 0.009


def test_single_fingertip_pose_corruption_is_localized():
  expected = moving_pose_movie()
  recorded = expected.copy()
  recorded[2, 1, 4, 2, 3] += 0.0001
  report = pose_alignment(recorded, expected)
  assert not report["valid"]
  assert report["mismatch_count"] == 1
  assert report["first_mismatches_frame_pose_indices"] == [[2, 1, 4]]
  assert report["maximum_translation_error_m"] == pytest.approx(0.0001)
  assert report["maximum_rotation_matrix_error"] == 0.0


def test_camera_pose_shape_and_homogeneous_row_are_checked():
  expected = moving_pose_movie()[:, 0, 0]
  recorded = expected.copy()
  recorded[1, 3, 3] = 0.0
  report = pose_alignment(recorded, expected)
  assert not report["valid"]
  assert report["first_mismatches_frame_pose_indices"] == [[1]]
  assert report["maximum_homogeneous_row_error"] == 1.0
  with pytest.raises(ValueError, match="matching nonempty finite"):
    pose_alignment(recorded[:-1], expected)
  recorded[0, 0, 0] = np.nan
  with pytest.raises(ValueError, match="matching nonempty finite"):
    pose_alignment(recorded, expected)
