import json

import numpy as np
import pytest
from kaihand_tactile_env.shared.whiteboard_contact_intensity import (
  ContactTrace,
  _channel_stats,
  _comparison,
  _plot_indices,
  _sampling_label,
  contact_phases,
  export_contact_intensity,
)
from PIL import Image


def _trace(channels, duration=None, phases=None, metadata=None):
  channels = {
    name: np.asarray(values, dtype=float) for name, values in channels.items()
  }
  count = len(next(iter(channels.values()))) if channels else 0
  return ContactTrace(
    time_s=np.arange(count, dtype=float),
    duration_s=np.asarray(
      duration if duration is not None else np.ones(count), dtype=float
    ),
    channels=channels,
    phases=np.asarray(phases) if phases is not None else contact_phases(channels),
    metadata=metadata or {},
  )


def test_duration_weighting_avoids_sample_count_bias_and_separates_exposures():
  trace = _trace({"board_normal_n": [0, 2, 10]}, duration=[1, 98, 1])
  stats = _channel_stats(
    trace, "board_normal_n", safety_threshold=8, expert_threshold=3
  )
  assert stats["mean_n"] == pytest.approx(2.06)
  assert stats["force_time_integral_n_s"] == pytest.approx(206)
  assert stats["contact_mean_n"] == pytest.approx(206 / 99)
  assert stats["p95_n"] == stats["contact_p95_n"] == 2
  assert stats["contact_duration_s"] == 99
  assert stats["sampled_peak_n"] == 10
  assert stats["sampled_peak_time_s"] == 2
  assert stats["safety_exceedance"]["duration_s"] == 1
  assert stats["safety_exceedance"]["fraction_of_valid_time"] == pytest.approx(0.01)
  assert stats["safety_exceedance"]["fraction_of_contact_time"] == pytest.approx(1 / 99)
  assert stats["expert_p95_exceedance"]["threshold_n"] == 3


def test_zero_contact_is_not_missing_and_zero_reference_ratio_is_null():
  trace = _trace({"board_normal_n": [0, 0.05, 0]})
  stats = _channel_stats(trace, "board_normal_n")
  assert stats["contact_duration_s"] == 0
  assert stats["contact_mean_n"] is None
  assert stats["contact_p95_n"] is None
  assert stats["mean_n"] == pytest.approx(0.05 / 3)
  expert = _channel_stats(_trace({"board_normal_n": [0, 0, 0]}), "board_normal_n")
  comparison = _comparison(stats, expert)
  assert comparison["sampled_peak_n"] == {
    "rollout": 0.05,
    "reference": 0.0,
    "delta": 0.05,
    "ratio": None,
  }
  assert comparison["contact_p95_n"]["delta"] is None


def test_invalid_samples_are_excluded_with_duration_coverage():
  trace = ContactTrace(
    time_s=np.array([0, 1, 2, np.nan, 4, 5]),
    duration_s=np.array([1, 2, 3, 4, -1, 0]),
    channels={"board_normal_n": np.array([2, np.nan, -3, 4, 5, 12])},
    phases=np.full(6, "board_contact"),
    metadata={},
  )
  stats = _channel_stats(trace, "board_normal_n")
  assert stats["valid_samples"] == 2
  assert stats["excluded_samples"] == 4
  assert stats["input_duration_s"] == 10
  assert stats["valid_duration_s"] == 1
  assert stats["coverage_fraction"] == pytest.approx(0.1)
  assert stats["mean_n"] == 2
  # A valid instantaneous endpoint can supply a sampled peak but has no
  # positive integration weight and does not distort duration statistics.
  assert stats["sampled_peak_n"] == 12
  assert stats["contact_p95_n"] == 2


def test_reported_peak_without_timeline_stays_distinct_and_unavailable():
  trace = _trace(
    {"grip_normal_n": [1, 2]},
    metadata={"reported_peaks_n": {"hand_board_normal_n": 7.47}},
  )
  stats = _channel_stats(trace, "hand_board_normal_n", safety_threshold=0.5)
  assert stats["availability"] == "channel_unavailable"
  assert stats["reported_peak_n"] == 7.47
  assert stats["sampled_peak_n"] is None
  assert stats["contact_duration_s"] is None
  assert stats["safety_exceedance"] is None


def test_phase_inference_preserves_unknowns_and_board_contact_recurrence():
  phases = contact_phases(
    {
      "grip_normal_n": np.array([0, 1, 1, 1, 0, 1, 1]),
      "board_normal_n": np.array([0, 0, 2, np.nan, 0, 3, 0]),
    }
  )
  assert phases.tolist() == [
    "pre_contact",
    "grasp_transport",
    "board_contact",
    "unknown",
    "post_board",
    "board_contact",
    "post_board",
  ]
  assert contact_phases({"grip_normal_n": np.array([0, 1])}).tolist() == [
    "unknown",
    "unknown",
  ]
  assert contact_phases({}).size == 0


def test_shape_and_time_errors_are_rejected():
  with pytest.raises(ValueError, match="same shape"):
    ContactTrace(
      np.array([0, 1]), np.array([1]), {}, np.array(["unknown", "unknown"]), {}
    )
  with pytest.raises(ValueError, match="channel"):
    ContactTrace(
      np.array([0, 1]),
      np.ones(2),
      {"board_normal_n": np.ones(3)},
      np.full(2, "unknown"),
      {},
    )
  with pytest.raises(ValueError, match="nondecreasing"):
    ContactTrace(np.array([1, 0]), np.ones(2), {}, np.full(2, "unknown"), {})


def test_export_writes_independent_artifacts_and_phase_relative_metrics(tmp_path):
  rollout = _trace(
    {
      "grip_normal_n": [0, 1, 2, 3],
      "board_normal_n": [0, 0, 4, 10],
      "finger_thumb_normal_n": [0, 0.5, 1, 1.5],
      "finger_index_normal_n": [0, 0.5, 1, 1.5],
    },
    metadata={
      "sampling": "control-step archived samples",
      "reported_peaks_n": {"hand_board_normal_n": 7.47},
    },
  )
  rollout.signals["cleaned_fraction"] = np.zeros(4)
  rollout.metadata.update(
    task_success=False,
    trial_sim_seconds=4.0,
    termination_reason="RuntimeError: hand struck the whiteboard",
    final_cleaned_fraction=0.0,
  )
  expert = _trace(
    {
      "grip_normal_n": [0, 1, 2, 2, 0],
      "board_normal_n": [0, 0, 2, 3, 0],
      "hand_board_normal_n": [0, 0, 0, 0, 0],
    }
  )
  expert.signals["cleaned_fraction"] = np.array([0.0, 0.0, 0.3, 1.0, 1.0])
  expert.metadata.update(task_success=True, final_cleaned_fraction=1.0)
  result = export_contact_intensity(tmp_path, rollout, expert)
  assert result["language"] == "zh-CN"
  assert result["quality"]["effect"]["rollout"]["final_cleaned_fraction"] == 0
  assert result["quality"]["effect"]["reference"]["final_cleaned_fraction"] == 1
  assert (
    result["quality"]["safety"]["rollout"]["hand_board_normal_n"]["exceeded"] is True
  )
  assert (
    result["quality"]["safety"]["rollout"]["hand_board_normal_n"]["exceeded_duration_s"]
    is None
  )
  assert result["status"] == "ok"
  assert result["common_phases"] == ["pre_contact", "grasp_transport", "board_contact"]
  board = result["channels"]["board_normal_n"]
  assert board["thresholds"] == {"safety_n": 8.0, "expert_contact_p95_n": 3.0}
  assert (
    board["phases"]["board_contact"]["rollout"]["expert_p95_exceedance"]["duration_s"]
    == 2
  )
  assert "post_board" not in board["phases"]
  assert result["channels"]["hand_board_normal_n"]["phases"] == {}
  assert result["channels"]["grip_normal_n"]["thresholds"]["safety_n"] is None
  loaded = json.loads((tmp_path / "contact_intensity.json").read_text())
  assert loaded == result
  with np.load(tmp_path / "contact_intensity.npz", allow_pickle=False) as archive:
    assert "rollout_hand_board_normal_n" not in archive
    np.testing.assert_array_equal(archive["rollout_board_normal_n"], [0, 0, 4, 10])
    np.testing.assert_array_equal(
      archive["rollout_signal_cleaned_fraction"], np.zeros(4)
    )
    assert json.loads(str(archive["metadata_json"])) == result
  assert result["artifacts"]["phases_plot"] == str(
    tmp_path / "contact_intensity_phases.png"
  )
  for name in (
    "contact_intensity.png",
    "contact_intensity_fingers.png",
    "contact_intensity_phases.png",
  ):
    with Image.open(tmp_path / name) as image:
      assert image.width >= 1000
      assert image.height >= 600


def test_empty_rollout_and_missing_reference_still_export(tmp_path):
  result = export_contact_intensity(
    tmp_path, _trace({}), reference_error="expert not found"
  )
  assert result["status"] == "no_valid_samples"
  assert result["reference_status"] == "unavailable"
  assert result["reference_error"] == "expert not found"
  assert result["channels"]["board_normal_n"]["rollout"]["sampled_peak_n"] is None
  assert (tmp_path / "contact_intensity.png").is_file()
  assert "phases_plot" not in result["artifacts"]
  assert not (tmp_path / "contact_intensity_phases.png").exists()


def test_partial_capture_is_reported_and_never_fabricates_missing_force(tmp_path):
  trace = _trace(
    {"board_normal_n": [1, np.nan, 3]},
    metadata={
      "attempted_steps": 5,
      "valid_steps": 3,
      "rejected_samples": 2,
      "capture_error": "nonfinite state",
      "physics_valid": False,
    },
  )
  result = export_contact_intensity(tmp_path, trace)
  assert result["status"] == "partial"
  assert result["reference_status"] == "unavailable"
  assert result["rollout"]["capture_coverage_fraction"] == 0.6
  assert result["channels"]["board_normal_n"]["rollout"][
    "coverage_fraction"
  ] == pytest.approx(2 / 3)
  with np.load(tmp_path / "contact_intensity.npz", allow_pickle=False) as archive:
    assert np.isnan(archive["rollout_board_normal_n"][1])


@pytest.mark.parametrize(
  "name", ["contact_intensity.png", "contact_intensity_phases.png"]
)
def test_preexisting_artifact_prevents_any_write(tmp_path, name):
  artifact = tmp_path / name
  artifact.write_bytes(b"existing-user-artifact")
  with pytest.raises(FileExistsError, match="already exists"):
    export_contact_intensity(tmp_path, _trace({"board_normal_n": [1]}))
  assert artifact.read_bytes() == b"existing-user-artifact"
  assert sorted(path.name for path in tmp_path.iterdir()) == [name]


def test_sampling_labels_distinguish_recorded_frequency_from_physics_rate():
  model = _trace(
    {"board_normal_n": [1, 2]},
    metadata={"sampling": "physics_step", "physics_timestep_s": 0.001},
  )
  expert = _trace(
    {"board_normal_n": [1, 2]}, metadata={"observed_hz": 100, "physics_hz": 1000}
  )
  archived = _trace({"board_normal_n": [1, 2]}, metadata={"observed_hz": 30})
  assert _sampling_label(model) == "1000 Hz 物理步采样"
  assert _sampling_label(expert) == "100 Hz 已记录样本"
  assert _sampling_label(archived) == "30 Hz 已记录样本"
  assert _sampling_label(None) == "参考缺失"


def test_plot_reduction_preserves_extremes_and_missing_gaps():
  values = np.ones(90000)
  values[41713] = 123
  values[41714] = 0
  values[50004:50019] = np.nan
  indices = _plot_indices(values)
  assert 41713 in indices
  assert 41714 in indices
  assert 50003 in indices and 50004 in indices and 50018 in indices and 50019 in indices
  assert len(indices) < 6100
