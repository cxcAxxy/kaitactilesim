"""Admission and safety regressions with no MuJoCo model or rendering."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.acceptance import (
  STRICT_FORCE_POLICY,
  TASK_COMPLETION_POLICY,
  accept_edge,
  accept_task,
  pressure_quality_label,
  validate_recorded_acceptance,
)
from kaihand_tactile_env.tasks.poker_draw.mid_full import MidForcePokerExecutor
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawExecutor


def edge():
  return {"target_reached": True, "held_at_edge": True,
          "slide_geometry_qualified": True, "slide_task_completed": True,
          "terminal_reason": "edge_reached", "full_slide_qualified": False}


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("quality", [False, True])
def test_only_force_quality_is_decoupled(completed, quality):
  assert accept_task(completed, quality, STRICT_FORCE_POLICY) == (completed and quality)
  assert accept_task(completed, quality, TASK_COMPLETION_POLICY) == completed
  if not completed:
    assert pressure_quality_label(completed, quality) == "incomplete"
  else:
    assert pressure_quality_label(completed, quality) == (
      "stable" if quality else "completed_with_force_variation")


def test_edge_success_cannot_be_faked_by_relaxing_pressure():
  outcome = edge()
  assert not accept_edge(outcome, STRICT_FORCE_POLICY)
  assert accept_edge(outcome, TASK_COMPLETION_POLICY)
  for name in ("target_reached", "held_at_edge", "slide_geometry_qualified", "slide_task_completed"):
    assert not accept_edge({**outcome, name: False}, TASK_COMPLETION_POLICY)
  assert not accept_edge({**outcome, "terminal_reason": "time_limit"}, TASK_COMPLETION_POLICY)
  with pytest.raises(ValueError):
    accept_edge(outcome, "anything-goes")


def valid_record():
  return {"acceptance_policy": TASK_COMPLETION_POLICY, "success": True,
          "task_completed": True, "terminal_pinch": True, "sustained_pinch": True,
          "retained_at_end": True, "half_overhang_reached": True,
          "simultaneous_four_finger_contact": True,
          "slide_press_control_qualified": False,
          "pressure_quality": "completed_with_force_variation",
          "edge_outcome": edge(), "handoff_outcome": {"completed": True},
          "maximum_slide_fingertip_plane_angle_degrees": 10.,
          "inspection_face_alignment": .9, "inspection_face_robot_alignment": .9,
          "inspection_position_error": .01, "minimum_supported_card_clearance": -.0001}


def test_export_validation_accepts_explicit_policy_not_legacy_relabeling():
  metadata = {"acceptance_policy": TASK_COMPLETION_POLICY}
  outcome = valid_record()
  validate_recorded_acceptance(metadata, outcome)
  assert outcome["slide_press_control_qualified"] is False
  with pytest.raises(ValueError, match="mismatch"):
    validate_recorded_acceptance({}, outcome)
  with pytest.raises(ValueError, match="strict source"):
    validate_recorded_acceptance({}, {"slide_press_control_qualified": False})
  for name in ("success", "task_completed", "terminal_pinch", "retained_at_end"):
    with pytest.raises(ValueError):
      validate_recorded_acceptance(metadata, {**outcome, name: False})
  with pytest.raises(ValueError, match="physical gate"):
    validate_recorded_acceptance(metadata, {**outcome, "minimum_supported_card_clearance": -.001})
  with pytest.raises(ValueError, match="inconsistent"):
    validate_recorded_acceptance(metadata, {**outcome, "pressure_quality": "stable"})


def test_egosteer_accepts_complete_training_episode_with_quality_warning():
  from kaihand_tactile_env.shared.egosteer_archive import validate_poker_outcome
  metadata = {"acceptance_policy": TASK_COMPLETION_POLICY, "scene": "poker-draw",
              "object": "card", "preset": "middle-force-precontact-v1"}
  outcome = {**valid_record(), "object_name": "card"}
  validate_poker_outcome(metadata, outcome)
  with pytest.raises(ValueError):
    validate_poker_outcome(metadata, {**outcome, "task_completed": False})


def test_tip_contact_angle_is_versioned_without_relaxing_historical_records():
  metadata = {"acceptance_policy": TASK_COMPLETION_POLICY}
  outcome = {**valid_record(), "maximum_slide_fingertip_plane_angle_degrees": 40.0}
  # Historical captures have no posture marker and keep their 21 degree gate.
  with pytest.raises(ValueError, match="physical gate"):
    validate_recorded_acceptance(metadata, outcome)
  tip_outcome = {**outcome, "draw_posture_version": "tip-pad-v7",
                 "slide_fingertip_plane_angle_limit_degrees": 50.0}
  validate_recorded_acceptance(metadata, tip_outcome)
  with pytest.raises(ValueError, match="physical gate"):
    validate_recorded_acceptance(metadata, {
      **tip_outcome, "maximum_slide_fingertip_plane_angle_degrees": 50.01})


@pytest.mark.parametrize("declaration", [
  {"slide_fingertip_plane_angle_limit_degrees": 50.0},
  {"draw_posture_version": "tip-pad-v7"},
  {"draw_posture_version": "tip-pad-v7", "slide_fingertip_plane_angle_limit_degrees": 70.0},
  {"draw_posture_version": "tip-pad-v7", "slide_fingertip_plane_angle_limit_degrees": float("nan")},
  {"draw_posture_version": "tip-pad-v7", "slide_fingertip_plane_angle_limit_degrees": True},
  {"draw_posture_version": "unknown"},
  {"draw_posture_version": []},
])
def test_recorded_posture_must_declare_a_known_matching_angle_limit(declaration):
  with pytest.raises(ValueError, match="posture"):
    validate_recorded_acceptance(
      {"acceptance_policy": TASK_COMPLETION_POLICY}, {**valid_record(), **declaration})


def guard(monkeypatch):
  executor = object.__new__(MidForcePokerExecutor)
  executor.sim = SimpleNamespace(timestep=.002)
  executor.acceptance_policy = TASK_COMPLETION_POLICY
  executor.press_force_per_finger_n = .5
  executor._multi_low_load_duration_s = 0.
  executor._maximum_multi_low_load_duration_s = 0.
  executor.forces = np.full(4, .5)
  executor._current_card_finger_normal_forces = lambda: executor.forces
  monkeypatch.setattr(PokerDrawExecutor, "_step", lambda *args: None)
  return executor


def test_short_multifinger_load_dips_are_allowed_and_recovery_resets(monkeypatch):
  executor = guard(monkeypatch)
  for _ in range(10):
    executor.forces[:] = [.01, .01, .5, .5]
    for _ in range(17):
      executor._step("slide_card", .465)
    executor.forces[:] = .5
    executor._step("slide_card", .465)
  assert executor._maximum_multi_low_load_duration_s == pytest.approx(.034)
  assert executor._multi_low_load_duration_s == 0


def test_sustained_multiple_low_load_still_stops(monkeypatch):
  executor = guard(monkeypatch)
  executor.forces[:] = [.01, .01, .5, .5]
  for _ in range(149):
    executor._step("edge_hold", .465)
  with pytest.raises(RuntimeError, match="0.30s"):
    executor._step("edge_hold", .465)


def test_one_finger_low_load_does_not_trigger_multifinger_guard(monkeypatch):
  executor = guard(monkeypatch)
  executor.forces[:] = [.01, .5, .5, .5]
  for _ in range(500):
    executor._step("slide_card", .465)
  assert executor._maximum_multi_low_load_duration_s == 0


def test_strict_mode_does_not_change_control_or_add_new_guard(monkeypatch):
  executor = guard(monkeypatch)
  executor.acceptance_policy = STRICT_FORCE_POLICY
  executor._current_card_finger_normal_forces = lambda: pytest.fail("new guard used in legacy mode")
  executor._step("slide_card", .465)


def test_capture_cli_policy_is_opt_in_and_reaches_executor(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  import collect_poker_batch as batch
  import record_dataset as recorder

  args = recorder._parse_args(["--scene", "poker-draw", "--preset", "middle-force-precontact-v1",
                              "--acceptance-policy", TASK_COMPLETION_POLICY,
                              "--output-dir", str(tmp_path), "--no-cameras"])
  job = recorder._build_jobs(args)[0]
  assert job.acceptance_policy == TASK_COMPLETION_POLICY
  assert recorder._preset_settings_for_job(job)["acceptance_policy"] == TASK_COMPLETION_POLICY
  assert "poker_draw/acceptance.py" in recorder._middle_source_hashes(job.preset)
  monkeypatch.setattr(recorder, "MidForcePokerExecutor", lambda sim, **kwargs: kwargs)
  assert recorder._poker_executor_for_job(job, object(), None)["acceptance_policy"] == TASK_COMPLETION_POLICY
  defaults = batch.parse_args(["--output-dir", str(tmp_path)])
  assert defaults.acceptance_policy == STRICT_FORCE_POLICY
  chosen = batch.parse_args(["--output-dir", str(tmp_path), "--acceptance-policy", TASK_COMPLETION_POLICY])
  command = batch.episode_command(chosen, 1)
  assert command[command.index("--acceptance-policy") + 1] == TASK_COMPLETION_POLICY
  for extra in (["--acceptance-policy", "unknown"], ["--acceptance-policy", TASK_COMPLETION_POLICY]):
    with pytest.raises(SystemExit):
      recorder._parse_args(["--scene", "pick-place", *extra])
