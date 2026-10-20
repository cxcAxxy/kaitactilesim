from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.whiteboard_contact_quality import build_contact_quality


def _trace(
  board, *, durations=None, times=None, fingers=None, clean=None, metadata=None
):
  size = len(board)
  channels = {"board_normal_n": np.asarray(board, dtype=float)}
  channels.update(
    {name: np.asarray(values, dtype=float) for name, values in (fingers or {}).items()}
  )
  return SimpleNamespace(
    time_s=np.asarray(times if times is not None else np.arange(size), dtype=float),
    duration_s=np.asarray(
      durations if durations is not None else np.ones(size), dtype=float
    ),
    channels=channels,
    phases=np.full(size, "board_contact"),
    metadata=metadata or {},
    signals={}
    if clean is None
    else {"cleaned_fraction": np.asarray(clean, dtype=float)},
  )


def test_finger_mean_includes_detached_intervals_and_variation_is_time_weighted():
  trace = _trace(
    [2, 2, 2],
    times=[0, 1, 4],
    durations=[1, 3, 1],
    fingers={"finger_thumb_normal_n": [0, 2, 4]},
  )
  result = build_contact_quality(trace)
  finger = result["continuity"]["rollout"]["finger_board_contact"]["thumb"]
  assert finger["contact_fraction"] == pytest.approx(0.8)
  assert finger["mean_n"] == 2
  assert finger["contact_mean_n"] == 2.5
  assert finger["std_n"] == pytest.approx(np.sqrt(1.6))
  assert finger["contact_std_n"] == pytest.approx(np.sqrt(0.75))
  phase = result["phase_variability"]["rollout"]["board_contact"][
    "finger_thumb_normal_n"
  ]
  assert phase["cv"] == pytest.approx(np.sqrt(1.6) / 2)
  assert phase["contact_cv"] == pytest.approx(np.sqrt(0.75) / 2.5)


def test_missing_collision_timeline_has_physical_peak_but_no_fake_exposure():
  trace = _trace([2, 3], metadata={"reported_peaks_n": {"hand_board_normal_n": 7.47}})
  safety = build_contact_quality(trace)["safety"]["rollout"]["hand_board_normal_n"]
  assert safety["reported_peak_n"] == 7.47
  assert safety["sampled_peak_n"] is None
  assert safety["exceeded"] is True
  assert safety["sampled_exceeded"] is None
  assert safety["exceeded_duration_s"] is None
  assert safety["coverage_fraction"] is None


def test_runtime_threshold_and_terminal_peak_do_not_fabricate_duration():
  trace = _trace(
    [1, 5, 7],
    durations=[1, 1, 0],
    metadata={"safety_thresholds_n": {"board_normal_n": 4}},
  )
  safety = build_contact_quality(trace)["safety"]["rollout"]["board_normal_n"]
  assert safety["threshold_n"] == 4
  assert safety["sampled_peak_n"] == 7
  assert safety["exceeded_duration_s"] == 1
  assert safety["exceeded_fraction"] == 0.5


def test_unknown_samples_and_clock_gaps_split_contact_without_becoming_known_breaks():
  trace = _trace(
    [2, np.nan, 2, 0, 2, 2],
    times=[0, 1, 2, 3, 4, 8],
    durations=[1] * 6,
  )
  continuity = build_contact_quality(trace)["continuity"]["rollout"]
  assert continuity["board_contact_episode_count"] == 4
  assert continuity["internal_break_count"] == 1
  assert continuity["internal_break_duration_s"] == 1
  assert continuity["unknown_duration_s"] == 4
  assert continuity["observed_span_s"] == 9
  assert continuity["board_contact_duration_s"] == 4
  assert continuity["clock_discontinuity_count"] == 1


def test_zero_duration_endpoint_alone_cannot_create_contact_episode():
  trace = _trace([0, 0, 2], durations=[1, 1, 0])
  continuity = build_contact_quality(trace)["continuity"]["rollout"]
  assert continuity["board_contact_episode_count"] == 0
  assert continuity["board_contact_duration_s"] == 0
  assert continuity["internal_break_count"] == 0
  assert continuity["finger_board_contact"]["thumb"]["contact_fraction"] is None


def test_sample_clock_progress_excludes_missing_and_nonmonotonic_intervals():
  trace = _trace(
    [2] * 6, durations=[1, 1, 1, 1, 1, 0], clean=[0, 0, 0.2, np.nan, 0.1, 0]
  )
  effect = build_contact_quality(trace)["effect"]["rollout"]
  assert effect["final_cleaned_fraction"] == 0
  assert effect["contact_progress_observed_duration_s"] == 2
  assert effect["contact_without_progress_duration_s"] == 1
  assert effect["contact_without_progress_fraction"] == 0.5
  assert effect["unknown_contact_progress_duration_s"] == 3
  assert effect["signal_coverage_fraction"] == 0.8


def test_interval_end_progress_uses_initial_state_and_never_crosses_missing_time():
  trace = _trace(
    [2, 2, 2],
    times=[0, 1, 4],
    clean=[0, 0.2, 0.2],
    metadata={"cleaned_fraction_clock": "interval_end", "initial_cleaned_fraction": 0},
  )
  effect = build_contact_quality(trace)["effect"]["rollout"]
  assert effect["contact_progress_observed_duration_s"] == 2
  assert effect["contact_without_progress_duration_s"] == 1
  assert effect["unknown_contact_progress_duration_s"] == 1


def test_missing_progress_preserves_real_final_zero_and_unknown_fault_final():
  without_signal = _trace([2, 2], metadata={"final_cleaned_fraction": 0})
  effect = build_contact_quality(without_signal)["effect"]["rollout"]
  assert effect["final_cleaned_fraction"] == 0
  assert effect["availability"] == "signal_unavailable"
  assert effect["contact_without_progress_duration_s"] is None
  assert effect["contact_without_progress_fraction"] is None
  fault = _trace(
    [2, 2],
    clean=[0.4, np.nan],
    metadata={
      "final_cleaned_fraction": None,
      "physics_valid": False,
      "stop_reason": "numerical_failure",
    },
  )
  result = build_contact_quality(fault)
  assert result["effect"]["rollout"]["final_cleaned_fraction"] is None
  assert result["effect"]["rollout"]["last_observed_cleaned_fraction"] == 0.4
  assert result["trial"]["rollout"]["physics_valid"] is False
  assert result["trial"]["rollout"]["stop_reason"] == "numerical_failure"


def test_no_contact_is_observed_zero_but_zero_force_cv_is_undefined():
  trace = _trace([0, 0], fingers={"finger_thumb_normal_n": [0, 0]}, clean=[0, 0])
  result = build_contact_quality(trace)
  assert result["effect"]["rollout"]["board_contact_observed_duration_s"] == 0
  assert result["effect"]["rollout"]["contact_without_progress_fraction"] is None
  stats = result["phase_variability"]["rollout"]["board_contact"]["board_normal_n"]
  assert stats["std_n"] == 0
  assert stats["cv"] is None
  assert stats["contact_std_n"] is None
  assert result["effect"]["reference"] is None


def test_unknowns_break_known_noncontact_runs_and_missing_board_is_unavailable():
  trace = _trace([2, 0, np.nan, 0, 2])
  continuity = build_contact_quality(trace)["continuity"]["rollout"]
  assert continuity["board_contact_episode_count"] == 2
  assert continuity["internal_break_count"] == 2
  assert continuity["internal_break_duration_s"] == 2
  trace.channels.clear()
  result = build_contact_quality(trace)
  assert result["continuity"]["rollout"]["board_contact_episode_count"] is None
  assert result["effect"]["rollout"]["board_contact_observed_duration_s"] is None
  assert result["safety"]["rollout"]["board_normal_n"]["exceeded"] is None


def test_invalid_duration_preserves_unknown_elapsed_time_and_splits_episodes():
  trace = _trace([2, 2, 2], durations=[1, np.nan, 1])
  continuity = build_contact_quality(trace)["continuity"]["rollout"]
  assert continuity["board_contact_episode_count"] == 2
  assert continuity["unknown_duration_s"] == 1
  assert continuity["internal_break_count"] == 0


def test_progress_cannot_jump_across_missing_clock_interval():
  trace = _trace([2, 2], times=[0, 4], durations=[1, 0], clean=[0, 0.8])
  effect = build_contact_quality(trace)["effect"]["rollout"]
  assert effect["final_cleaned_fraction"] == 0.8
  assert effect["contact_progress_observed_duration_s"] == 0
  assert effect["contact_without_progress_duration_s"] is None
  assert effect["unknown_contact_progress_duration_s"] == 1


def test_partial_finger_coverage_does_not_turn_unknown_samples_into_detachment():
  trace = _trace(
    [2, 2, 2, 2],
    fingers={"finger_thumb_normal_n": [0, 2, np.nan, np.nan]},
  )
  finger = build_contact_quality(trace)["continuity"]["rollout"][
    "finger_board_contact"
  ]["thumb"]
  assert finger["coverage_fraction"] == 0.5
  assert finger["contact_fraction"] == 0.5
  assert finger["fraction_scope"] == "valid_finger_samples_during_board_contact"
  assert finger["mean_n"] == 1
  assert finger["contact_mean_n"] == 2
  assert finger["valid_duration_s"] == 2
  assert finger["contact_duration_s"] == 1
