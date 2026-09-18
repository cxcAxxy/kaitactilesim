"""Array/tiny HDF5 checks only; no simulator or rendering."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/workcell/poker_cleaning_checks.py"
spec = importlib.util.spec_from_file_location("poker_checks_test", path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_30hz_physics_quantized_and_short_terminal():
  result = checks.clock_check([0, .034, .066, .1, .104], 1e9 / 30, 104_000_000,
                              physics_step_ns=2_000_000)
  assert result["valid"] and result["off_grid_terminal_frame_expected"]


@pytest.mark.parametrize("times", [[0, .066, .1], [0, .034, .034, .066, .1],
                                  [0, .066, .034, .1], [0, np.nan]])
def test_missing_duplicate_reversed_and_nonfinite_clocks_fail(times):
  assert not checks.clock_check(times, 1e9 / 30, 100_000_000,
                                physics_step_ns=2_000_000)["valid"]


def fixture_file(tmp_path):
  f = h5py.File(tmp_path / "synthetic.h5", "w")
  f.attrs.update(physics_hz=500, control_hz=100, camera_hz=30)
  state = np.r_[np.arange(11) * .01, .104]
  trace_time = np.arange(52) * .002
  actual = np.arange(52 * 7).reshape(52, 7) * .0001
  f["state/timestamp"] = state
  f["control/precontact_noise/time_s"] = trace_time
  f["control/precontact_noise/actual_ctrl_rad"] = actual
  f["tactile_contact_force/timestamp"] = np.maximum(state - .002, 0)
  capture = np.array([0, .034, .066, .1, .104])
  f["cameras/head/timestamp"] = capture
  f["cameras/head/pose_timestamp"] = np.maximum(capture - .002, 0)
  f["cameras/head/state_index"] = np.searchsorted(checks.ns(state), checks.ns(capture), side="right") - 1
  f["commands/actuator_names"] = np.array([f"right_arm_joint{i}" for i in range(1, 8)], dtype=h5py.string_dtype())
  f["commands/actuator_control"] = np.vstack([np.zeros(7), actual[np.rint(state[1:] / .002).astype(int) - 1]])
  f["commands/phase"] = np.array(["slide_card"] * len(state), dtype=h5py.string_dtype())
  output = tmp_path / "report"
  output.mkdir()
  return f, output


def test_mixed_camera_clock_and_actual_substep_mapping(tmp_path):
  f, out = fixture_file(tmp_path)
  with f:
    report = checks.check_clocks_and_controls(f, out)
  assert report["valid_for_declared_clock_semantics"]
  assert report["cameras"]["head"]["maximum_indexed_force_age_ns"] == 6_000_000
  assert report["right_arm_transition_mapping"]["substep_count_distribution"] == {"2": 1, "5": 10}


def test_shifted_action_row_is_detected(tmp_path):
  f, out = fixture_file(tmp_path)
  with f:
    f["commands/actuator_control"][3] = f["commands/actuator_control"][2]
    report = checks.check_clocks_and_controls(f, out)
  assert not report["right_arm_transition_mapping"]["valid"]


def test_wrong_camera_index_is_detected(tmp_path):
  f, out = fixture_file(tmp_path)
  with f:
    f["cameras/head/state_index"][2] = 5
    report = checks.check_clocks_and_controls(f, out)
  assert not report["cameras"]["head"]["valid"]
