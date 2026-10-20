import json

import numpy as np
import pytest
from kaihand_tactile_env.shared.whiteboard_tactile_comparison import (
  TactileTrace,
  _model_end,
  build_tactile_comparison,
  derive_tactile_trace,
  export_tactile_comparison,
)


def _trace(times=(0, 1, 2), force=None, tangent=None, metadata=None):
  times = np.asarray(times, dtype=float)
  normal = np.zeros((len(times), 5, 7, 5))
  normal[:, :, 3, 2] = 1
  if force is not None:
    normal = np.asarray(force, dtype=float)
  if tangent is None:
    tangent = np.zeros((*normal.shape, 2))
  columns, rows = np.meshgrid(np.arange(5), np.arange(7))
  positions = (
    np.stack((columns, rows, np.zeros_like(rows)), axis=-1).reshape(35, 3) / 1000
  )
  return TactileTrace(
    times,
    normal,
    tangent,
    np.tile(positions[None], (5, 1, 1)),
    np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]])[None], (5, 1, 1)),
    metadata or {},
  )


def test_identical_grids_have_zero_errors_and_endpoint_has_no_duration():
  trace = _trace()
  summary, arrays = build_tactile_comparison(trace, trace)
  finger = summary["fingers"]["thumb"]
  assert finger["rollout"]["contact_duration_s"] == 2
  assert finger["comparison"]["center_distance_mm"] == {"mean": 0.0, "p95": 0.0}
  assert finger["comparison"]["normalized_spatial_difference"] == {
    "mean": 0.0,
    "p95": 0.0,
  }
  assert finger["comparison"]["coverage_of_reference_valid_time"] == 1
  np.testing.assert_allclose(arrays["rollout_center_local_mm"][:, 0], [[2, 3]] * 3)
  np.testing.assert_array_equal(arrays["rollout_duration_s"], [1, 1, 0])


def test_force_scaling_does_not_change_center_or_normalized_distribution():
  reference = _trace()
  model = _trace(force=reference.normal_taxel_force_n * 4)
  summary, _ = build_tactile_comparison(model, reference)
  metrics = summary["fingers"]["thumb"]["comparison"]
  assert metrics["center_distance_mm"]["mean"] == 0
  assert metrics["normalized_spatial_difference"]["mean"] == 0
  assert metrics["normal_force_absolute_error_n"]["mean"] == 3


def test_shifted_contact_has_physical_center_offset_and_maximal_map_difference():
  reference = _trace()
  force = reference.normal_taxel_force_n.copy()
  force[:, :, 3, 2] = 0
  force[:, :, 3, 3] = 1
  summary, _ = build_tactile_comparison(_trace(force=force), reference)
  metrics = summary["fingers"]["thumb"]["comparison"]
  assert metrics["center_distance_mm"]["mean"] == pytest.approx(1)
  assert metrics["normalized_spatial_difference"]["mean"] == 1


def test_same_center_can_have_different_spatial_distribution():
  reference = _trace()
  force = reference.normal_taxel_force_n.copy()
  force[:, :, 3, 2] = 0
  force[:, :, 3, 1] = 0.5
  force[:, :, 3, 3] = 0.5
  summary, _ = build_tactile_comparison(_trace(force=force), reference)
  metrics = summary["fingers"]["thumb"]["comparison"]
  assert metrics["center_distance_mm"]["mean"] == pytest.approx(0)
  assert metrics["normalized_spatial_difference"]["mean"] == 1


def test_no_contact_centers_and_distributions_are_nan_not_origin():
  trace = _trace()
  trace.normal_taxel_force_n[:] = 0
  derived = derive_tactile_trace(trace)
  assert np.all(np.isnan(derived["center_local_mm"]))
  assert np.all(np.isnan(derived["normalized_normal_distribution"]))
  assert not np.any(derived["contact"])
  assert np.all(derived["normal_valid"])
  summary, _ = build_tactile_comparison(trace, trace)
  assert summary["fingers"]["thumb"]["comparison"]["center_distance_mm"]["mean"] is None


def test_missed_and_extra_contact_are_separate_from_missing_and_map_distance():
  reference = _trace(times=(0, 1, 2, 3))
  model = _trace(times=(0, 1, 2, 3))
  reference.normal_taxel_force_n[1] = 0
  model.normal_taxel_force_n[0] = 0
  model.normal_taxel_force_n[2, 0, 0, 0] = np.nan
  summary, arrays = build_tactile_comparison(model, reference)
  metrics = summary["fingers"]["thumb"]["comparison"]
  assert metrics["missed_contact_duration_s"] == 1
  assert metrics["extra_contact_duration_s"] == 1
  assert metrics["missed_contact_fraction_of_reference_contact_in_overlap"] == 1
  assert metrics["extra_contact_fraction_of_reference_no_contact_in_overlap"] == 1
  assert metrics["paired_valid_duration_s"] == 2
  assert metrics["coverage_of_reference_valid_time"] == pytest.approx(2 / 3)
  assert np.isnan(arrays["normalized_spatial_difference"][:3, 0]).all()
  assert not arrays["missed_contact"][2, 0]


def test_early_end_is_not_zero_padded_or_stretched():
  reference = _trace(times=(0, 1, 2, 3, 4))
  model = _trace(times=(0, 1, 2))
  summary, arrays = build_tactile_comparison(model, reference)
  metrics = summary["fingers"]["thumb"]["comparison"]
  assert metrics["paired_valid_duration_s"] == 2
  assert metrics["coverage_of_reference_valid_time"] == 0.5
  assert metrics["missed_contact_duration_s"] == 0
  np.testing.assert_array_equal(arrays["matched_rollout_indices"], [0, 1, 2, -1, -1])
  assert np.isnan(arrays["center_distance_mm"][3:]).all()


def test_model_stop_time_is_projected_from_source_clock_onto_trace_axis():
  metadata = {"trial_sim_seconds": 12, "source_start_time_s": 2}
  assert _model_end(_trace(times=(0, 5, 10), metadata=metadata)) == 10
  assert _model_end(_trace(times=(3, 8, 13), metadata=metadata)) == 13
  assert _model_end(_trace(times=(0, 5, 10))) == 10


def test_left_hold_does_not_use_future_sample_and_weights_are_native():
  reference = _trace(times=(0, 0.25, 0.5, 0.75, 1))
  model = _trace(times=(0, 0.5, 1))
  model.normal_taxel_force_n[1] *= 3
  summary, arrays = build_tactile_comparison(model, reference)
  np.testing.assert_array_equal(arrays["matched_rollout_indices"], [0, 0, 1, 1, 2])
  assert summary["fingers"]["thumb"]["rollout"]["normal_mean_n"] == 2


def test_sparse_gap_does_not_hold_old_force_through_whole_gap():
  reference = _trace(times=np.arange(8))
  model = _trace(times=(0, 1, 2, 6, 7), metadata={"nominal_interval_s": 1})
  summary, arrays = build_tactile_comparison(model, reference)
  np.testing.assert_array_equal(
    arrays["matched_rollout_indices"], [0, 1, 2, 2, -1, -1, 3, 4]
  )
  assert arrays["paired_duration_s"][3] == pytest.approx(0.5)
  assert summary["fingers"]["thumb"]["rollout"]["contact_duration_s"] == 3
  assert summary["fingers"]["thumb"]["comparison"]["paired_valid_duration_s"] == 4.5


def test_signed_tangent_is_net_force_not_noncancelling_load():
  trace = _trace()
  trace.tangent_taxel_force_n[:, :, 3, 1, 0] = 2
  trace.tangent_taxel_force_n[:, :, 3, 3, 0] = -2
  derived = derive_tactile_trace(trace)
  np.testing.assert_array_equal(derived["tangent_net_force_n"], np.zeros((3, 5)))
  trace.tangent_taxel_force_n[1, 0, 0, 0, 0] = np.nan
  derived = derive_tactile_trace(trace)
  assert np.isnan(derived["tangent_net_force_n"][1, 0])
  assert derived["normal_valid"][1, 0]


def test_negative_force_is_unknown_per_finger():
  trace = _trace()
  trace.normal_taxel_force_n[0, 0, 0, 0] = -1
  derived = derive_tactile_trace(trace)
  assert np.isnan(derived["normal_force_n"][0, 0])
  assert not derived["normal_valid"][0, 0]
  assert derived["normal_valid"][0, 1]


def test_coordinate_mismatch_is_rejected():
  reference, model = _trace(), _trace()
  model.taxel_positions_local_m[0, 0, 0] += 0.001
  with pytest.raises(ValueError, match="不一致"):
    build_tactile_comparison(model, reference)


def test_invalid_time_or_basis_is_rejected():
  with pytest.raises(ValueError, match="严格递增"):
    _trace(times=(0, 1, 1))
  trace = _trace()
  trace.tangent_basis_local[0, 0] *= 2
  with pytest.raises(ValueError, match="正交"):
    TactileTrace(
      trace.time_s,
      trace.normal_taxel_force_n,
      trace.tangent_taxel_force_n,
      trace.taxel_positions_local_m,
      trace.tangent_basis_local,
      {},
    )


def test_export_chinese_json_raw_arrays_and_never_overwrites(tmp_path):
  reference = _trace(metadata={"snapshot_times_s": [0, 1, 2], "source": "task_demo"})
  model = _trace(metadata={"trial_sim_seconds": 2, "missing": np.nan})
  result = export_tactile_comparison(model, reference, tmp_path)
  assert len(result["artifacts"]) == 5
  summary = json.loads((tmp_path / "tactile_contact_comparison.json").read_text())
  assert summary["language"] == "zh-CN"
  assert summary["rollout"]["metadata"]["missing"] is None
  assert summary["heatmap_snapshots"][1] == {
    "reference_time_s": 1.0,
    "rollout_time_s": 1.0,
  }
  with np.load(tmp_path / "tactile_contact_comparison.npz") as archive:
    np.testing.assert_array_equal(
      archive["rollout_normal_taxel_force_n"], model.normal_taxel_force_n
    )
  with pytest.raises(FileExistsError):
    export_tactile_comparison(model, reference, tmp_path)


@pytest.mark.parametrize("snapshot_times", [[-0.01], [2.01], [np.nan], [[1]]])
def test_invalid_snapshot_times_fail_before_any_artifact_is_written(
  tmp_path, snapshot_times
):
  reference = _trace(metadata={"snapshot_times_s": snapshot_times})
  with pytest.raises(ValueError, match="关键时刻"):
    export_tactile_comparison(_trace(), reference, tmp_path)
  assert not list(tmp_path.iterdir())


def test_custom_threshold_is_applied_to_contact_metrics():
  trace = _trace()
  summary, arrays = build_tactile_comparison(trace, trace, contact_threshold_n=2)
  assert summary["contact_threshold_n"] == 2
  assert summary["fingers"]["thumb"]["rollout"]["contact_duration_s"] == 0
  assert not arrays["both_contact"].any()
