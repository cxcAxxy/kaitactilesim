"""Bounded checks for the headless bulb capture-basin diagnostic."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation


def _script():
  return run_path(
    str(Path(__file__).parents[1] / "scripts/workcell/experiment_bulb_capture_basin.py")
  )


def test_default_gate_grid_is_bounded_and_resolves_yaw_phase():
  args = _script()["_parse_args"]([])
  assert args.protocol == "gate"
  assert args.yaw_degrees == tuple(float(value) for value in range(0, 360, 5))
  assert args.depth_errors_mm[0] == pytest.approx(-2.0)
  assert args.depth_errors_mm[-1] == pytest.approx(5.0)
  assert args.trial_count == 72 * 71


def test_approach_defaults_sample_funnel_sector_centre_and_boundary():
  args = _script()["_parse_args"](["--protocol", "approach"])
  assert args.azimuth_degrees == (0.0, 7.5)
  assert args.lateral_offsets_mm == (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)
  assert args.trial_count == 4 * 6 * 4 * 2


@pytest.mark.parametrize(
  "options",
  [
    ["--yaw-degrees", "nan"],
    ["--tilt-degrees", "-1"],
    ["--protocol", "approach", "--depth-errors-mm", "0"],
    ["--protocol", "approach", "--approach-duration-s", "0"],
    ["--protocol", "approach", "--lateral-stiffness-n-m", "-1"],
    ["--protocol", "approach", "--yaw-degrees", *map(str, range(1001))],
  ],
)
def test_invalid_or_unbounded_grids_fail_early(options):
  with pytest.raises(SystemExit) as error:
    _script()["_parse_args"](options)
  assert error.value.code == 2


def test_gate_row_reports_each_predicate_without_changing_default_physics():
  script = _script()
  simulation = BulbScrewSimulation()
  row = script["_gate_row"](
    simulation,
    yaw_deg=0,
    depth_error_mm=0,
    lateral_mm=0,
    tilt_deg=0,
    azimuth_deg=0,
  )
  assert row["depth_ok"]
  assert row["lateral_ok"]
  assert row["tilt_ok"]
  assert row["speed_ok"]
  assert row["angular_speed_ok"]
  assert row["contact_ok"]
  assert row["captured"]
  assert row["thread_contact_load_n"] > config.THREAD_CAPTURE_LOAD_N


def test_pose_uses_positive_depth_error_for_deeper_insertion():
  position, quaternion = _script()["_pose"](2, 3, 0, 0, 90)
  np.testing.assert_allclose(
    position,
    config.THREAD_ENTRY_POSITION_M + np.array([0, 0.002, -0.003]),
    atol=1e-12,
  )
  np.testing.assert_allclose(quaternion, [1, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize(
  "captured,released,cleared,contact,speed,z,radial,tilt,outcome",
  [
    (True, True, True, True, 0, 0.73, 0, 0, "captured_and_cleared"),
    (True, False, False, True, 0, 0.73, 0, 0, "captured_release_jam"),
    (True, True, False, True, 0, 0.73, 0, 0, "captured_extraction_jam"),
    (False, False, False, True, 0.001, 0.73, 0, 0.1, "tilted_jam"),
    (False, False, False, True, 0.001, 0.73, 0, 0, "straight_entry_jam"),
    (False, False, False, False, 0.001, 0.68, 0.1, 0, "missed_fixture"),
    (False, False, False, False, 0.1, 0.73, 0, 0, "unsettled_timeout"),
  ],
)
def test_terminal_classifier_separates_capture_jam_miss_and_motion(
  captured, released, cleared, contact, speed, z, radial, tilt, outcome
):
  classify = _script()["_classify_terminal"]
  assert (
    classify(
      captured=captured,
      released=released,
      cleared=cleared,
      fixture_contact_recent=contact,
      linear_speed_m_s=speed,
      base_z_m=z,
      radial_m=radial,
      tilt_rad=tilt,
    )
    == outcome
  )


def test_output_reservation_refuses_overwrite(tmp_path):
  script = _script()
  output = tmp_path / "existing.json"
  output.write_text("keep", encoding="utf-8")
  args = script["_parse_args"](["--json", str(output)])
  with pytest.raises(FileExistsError):
    script["_reserve_outputs"](args)
  assert output.read_text(encoding="utf-8") == "keep"


def test_gate_summary_allows_grid_without_a_production_depth_sample():
  summary = _script()["_summary"](
    "gate",
    [
      {
        "captured": False,
        "yaw_deg": 90.0,
        "depth_error_mm": 2.0,
        "depth_ok": False,
        "thread_contact_load_n": 0.0,
        "contact_ok": False,
      }
    ],
  )
  phase = summary["yaw_phase"][0]
  assert phase["peak_gate_contact_load_n"] is None
  assert phase["peak_gate_contact_depth_error_mm"] is None


def test_gate_summary_keeps_lateral_tilt_and_azimuth_conditions_separate():
  summarize = _script()["_summary"]

  def row(lateral, tilt, azimuth):
    return {
      "captured": False,
      "yaw_deg": 90.0,
      "lateral_offset_mm": lateral,
      "initial_tilt_deg": tilt,
      "azimuth_deg": azimuth,
      "depth_error_mm": 0.0,
      "depth_ok": True,
      "thread_contact_load_n": 0.0,
      "contact_ok": False,
    }

  summary = summarize(
    "gate",
    [row(0.0, 0.0, 0.0), row(1.0, 2.0, 7.5)],
  )

  assert summary["gate_grouping_dimensions"] == [
    "yaw_deg",
    "lateral_offset_mm",
    "initial_tilt_deg",
    "azimuth_deg",
  ]
  assert len(summary["yaw_phase"]) == 2
  assert {
    (
      condition["lateral_offset_mm"],
      condition["initial_tilt_deg"],
      condition["azimuth_deg"],
    )
    for condition in summary["yaw_phase"]
  } == {(0.0, 0.0, 0.0), (1.0, 2.0, 7.5)}


def test_dynamic_trial_marks_mujoco_warning_and_nonfinite_state_invalid(monkeypatch):
  script = _script()
  args = script["_parse_args"](
    [
      "--protocol",
      "approach",
      "--yaw-degrees",
      "0",
      "--lateral-offsets-mm",
      "0",
      "--tilt-degrees",
      "0",
      "--azimuth-degrees",
      "0",
      "--approach-duration-s",
      "0.002",
      "--settle-duration-s",
      "0",
    ]
  )
  original_step = BulbScrewSimulation.step

  def unstable_step(self):
    original_step(self)
    script["mujoco"].get_mju_user_warning()("synthetic numerical warning")
    self.data.qpos[0] = np.nan

  monkeypatch.setattr(BulbScrewSimulation, "step", unstable_step)
  row = script["_run_approach"](BulbScrewSimulation(), args)[0]
  summary = script["_summary"]("approach", [row])

  assert row["outcome"] == "invalid"
  assert row["valid"] is False
  assert row["invalid_reasons"] == ["mujoco_warning", "nonfinite_state"]
  assert row["mujoco_warning_messages"] == [
    {"message": "synthetic numerical warning", "count": 1}
  ]
  assert {entry["field"] for entry in row["nonfinite_fields"]} == {"qpos"}
  assert summary["valid_trial_count"] == 0
  assert summary["invalid_trial_count"] == 1
  assert summary["capture_count"] == 0
  assert summary["capture_fraction"] is None


def test_centered_dynamic_capture_records_the_enabling_crest_load():
  script = _script()
  args = script["_parse_args"](
    [
      "--protocol",
      "approach",
      "--yaw-degrees",
      "0",
      "--lateral-offsets-mm",
      "0",
      "--tilt-degrees",
      "0",
      "--azimuth-degrees",
      "0",
    ]
  )
  row = script["_run_approach"](BulbScrewSimulation(), args)[0]
  assert row["valid"]
  assert row["mujoco_warning_messages"] == []
  assert row["mujoco_warning_stats"] == []
  assert row["nonfinite_fields"] == []
  assert row["captured"] and row["released"] and row["cleared"]
  assert row["first_thread_contact_s"] is not None
  assert row["peak_thread_contact_n"] > config.THREAD_CAPTURE_LOAD_N
