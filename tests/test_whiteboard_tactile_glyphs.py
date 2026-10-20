import json
from unittest.mock import Mock

import numpy as np
import pytest
from kaihand_tactile_env.shared.whiteboard_tactile_comparison import TactileTrace
from kaihand_tactile_env.shared.whiteboard_tactile_glyphs import (
  _draw_sample,
  build_glyph_samples,
  export_tactile_glyphs,
)


def _trace(times=(0, 0.1, 0.2, 0.3), *, force=None, tangent=None, metadata=None):
  times = np.asarray(times, dtype=float)
  normal = np.zeros((len(times), 5, 7, 5))
  values = np.ones(len(times)) if force is None else np.asarray(force, dtype=float)
  normal[:, :, 3, 2] = values[:, None]
  tangential = np.zeros((*normal.shape, 2))
  if tangent is not None:
    tangential[:, :, 3, 2] = np.asarray(tangent, dtype=float)[:, None]
  columns, rows = np.meshgrid(np.arange(5), np.arange(7))
  positions = (
    np.stack((columns, rows, np.zeros_like(rows)), axis=-1).reshape(35, 3) / 1000
  )
  return TactileTrace(
    times,
    normal,
    tangential,
    np.tile(positions[None], (5, 1, 1)),
    np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]])[None], (5, 1, 1)),
    metadata or {},
  )


def test_signed_vector_change_is_not_change_in_magnitude():
  trace = _trace(force=[1, 2, 3, 4], tangent=[[1, 0], [0, 1], [-1, 0], [0, -1]])
  summary, archive = build_glyph_samples(trace, trace, [0.1, 0.2], window_s=0.1)
  np.testing.assert_allclose(
    archive["rollout_tangent_change_local_n"][:, 0], [[-1, 1], [-1, -1]]
  )
  np.testing.assert_allclose(archive["rollout_normal_change_n"][:, 0], [1, 1])
  assert summary["scales"]["tangent_change_max_n"] == pytest.approx(np.sqrt(2))
  assert "不是当前力" in summary["definitions"]["arrow"]


def test_tangent_taxel_vectors_cancel_before_computing_increment():
  trace = _trace()
  trace.tangent_taxel_force_n[:, :, 3, 1, 0] = np.arange(1, 5)[:, None]
  trace.tangent_taxel_force_n[:, :, 3, 3, 0] = -np.arange(1, 5)[:, None]
  _, archive = build_glyph_samples(trace, trace, [0.1, 0.2])
  np.testing.assert_array_equal(
    archive["rollout_tangent_change_local_n"], np.zeros((2, 5, 2))
  )


def test_fixed_time_window_matches_different_native_sample_rates():
  fast_time = np.arange(61) / 100
  slow_time = np.arange(19) / 30
  expert = _trace(
    fast_time, force=fast_time + 1, tangent=np.column_stack((fast_time, -fast_time))
  )
  model = _trace(
    slow_time, force=slow_time + 1, tangent=np.column_stack((slow_time, -slow_time))
  )
  summary, archive = build_glyph_samples(model, expert, [0.25, 0.3], window_s=0.1)
  for source in ("rollout", "reference"):
    np.testing.assert_allclose(archive[f"{source}_normal_change_n"][:, 0], [0.1, 0.1])
    np.testing.assert_allclose(
      archive[f"{source}_tangent_change_local_n"][:, 0], [[0.1, -0.1], [0.1, -0.1]]
    )
  assert archive["rollout_sample_time_s"][0] == pytest.approx(7 / 30)
  assert archive["reference_sample_time_s"][0] == 0.25
  assert summary["window_s"] == 0.1


def test_clock_tolerance_only_corrects_roundoff_not_future_samples():
  trace = _trace(times=np.arange(4) / 10 + 1e-12, force=[1, 2, 3, 4])
  _, archive = build_glyph_samples(trace, trace, [0.2, 0.25])
  np.testing.assert_allclose(archive["rollout_sample_time_s"], [0.2, 0.2], atol=1e-9)
  np.testing.assert_allclose(archive["rollout_normal_change_n"][:, 0], [1, 1])


def test_force_gain_from_no_contact_has_valid_delta_without_past_center():
  trace = _trace(force=[0, 0, 2, 0], tangent=[[0, 0], [0, 0], [0.5, -0.25], [0, 0]])
  _, archive = build_glyph_samples(trace, trace, [0.2, 0.3])
  np.testing.assert_allclose(archive["rollout_normal_change_n"][:, 0], [2, -2])
  assert archive["rollout_contact"][0, 0]
  assert not archive["rollout_contact"][1, 0]
  assert np.isnan(archive["rollout_center_local_mm"][1]).all()
  np.testing.assert_allclose(
    archive["rollout_tangent_change_local_n"][:, 0], [[0.5, -0.25], [-0.5, 0.25]]
  )


def test_window_before_first_observation_is_unknown_not_zero():
  trace = _trace()
  _, archive = build_glyph_samples(trace, trace, [0.0, 0.1])
  assert archive["rollout_normal_valid"][0].all()
  assert np.isnan(archive["rollout_normal_change_n"][0]).all()
  assert np.isnan(archive["rollout_tangent_change_local_n"][0]).all()
  assert archive["rollout_normal_change_valid"][1].all()


def test_gap_inside_window_invalidates_change_even_with_valid_endpoints():
  trace = _trace(
    times=[0, 0.01, 0.02, 0.12, 0.13], metadata={"nominal_interval_s": 0.01}
  )
  _, archive = build_glyph_samples(trace, trace, [0.13], window_s=0.13)
  assert archive["rollout_normal_valid"].all()
  assert archive["rollout_sample_time_s"][0] == 0.13
  assert archive["rollout_previous_sample_time_s"][0] == 0
  assert np.isnan(archive["rollout_normal_change_n"]).all()
  assert not archive["rollout_window_continuous"].any()


def test_invalid_interior_sample_masks_only_affected_finger_and_channel():
  trace = _trace()
  trace.normal_taxel_force_n[1, 0, 0, 0] = np.nan
  trace.tangent_taxel_force_n[1, 1, 0, 0, 0] = np.nan
  _, archive = build_glyph_samples(trace, trace, [0.3], window_s=0.3)
  assert not archive["rollout_normal_change_valid"][0, 0]
  assert archive["rollout_normal_change_valid"][0, 1]
  assert not archive["rollout_tangent_change_valid"][0, 1]
  assert archive["rollout_tangent_change_valid"][0, 2:].all()


def test_early_stop_missing_is_not_no_contact_or_extrapolation():
  expert = _trace()
  model = _trace(times=[0, 0.1])
  _, archive = build_glyph_samples(model, expert, [0.1, 0.2, 0.3])
  assert archive["rollout_contact"][0].all()
  assert not archive["rollout_normal_valid"][1:].any()
  assert np.isnan(archive["rollout_normal_force_n"][1:]).all()
  assert (archive["rollout_missing_reason"][1:] == 2).all()
  assert archive["reference_normal_valid"].all()


def test_known_zero_is_no_contact_not_missing_and_threshold_is_respected():
  trace = _trace(force=[0, 0.01, 0.04, 1])
  summary, archive = build_glyph_samples(
    trace, trace, [0, 0.1, 0.2], contact_threshold_n=0.05
  )
  assert archive["rollout_normal_valid"].all()
  assert not archive["rollout_contact"].any()
  assert (archive["rollout_missing_reason"] == 0).all()
  assert np.isnan(archive["rollout_center_local_mm"]).all()
  assert summary["contact_threshold_n"] == 0.05


def test_shared_circle_scale_includes_native_peaks_between_video_frames():
  trace = _trace(times=[0, 0.05, 0.1, 0.15, 0.2], force=[1, 9, 1, 1, 1])
  summary, archive = build_glyph_samples(trace, trace, [0, 0.1, 0.2])
  assert archive["rollout_normal_force_n"].max() == 1
  assert summary["scales"]["normal_force_max_n"] == 9
  scale = summary["scales"]["circle_radius_squared_mm2_per_n"]
  assert 9 * scale == pytest.approx(summary["scales"]["max_circle_radius_mm"] ** 2)


def test_arrow_visual_cap_does_not_modify_actual_force_vectors():
  trace = _trace(tangent=[[0, 0], [6, 8], [0, 0], [0, 0]])
  summary, archive = build_glyph_samples(trace, trace, [0.1], arrow_display_cap_n=2)
  np.testing.assert_allclose(archive["rollout_tangent_change_local_n"][0, 0], [6, 8])
  assert summary["scales"]["tangent_change_max_n"] == 10
  assert summary["scales"]["arrow_display_cap_n"] == 2
  assert "虚线" in summary["scales"]["arrow_clipping"]


def test_rendered_circle_area_and_arrow_direction_have_shared_physical_scale():
  trace = _trace(force=[0, 2, 4, 0], tangent=[[0, 0], [6, 8], [0, 0], [0, 0]])
  summary, archive = build_glyph_samples(trace, trace, [0.1, 0.2, 0.3])
  panel = tuple(Mock() for _ in range(4))
  _draw_sample(panel, "rollout", 0, 0, archive, summary, lambda _: "red", lambda x: x)
  circle, arrow, status, label = panel
  np.testing.assert_allclose(circle.center, [0, 0])
  assert circle.radius**2 == pytest.approx(
    2 * summary["scales"]["circle_radius_squared_mm2_per_n"]
  )
  start, stop = arrow.set_positions.call_args.args
  np.testing.assert_allclose(start, [0, 0])
  np.testing.assert_allclose(
    stop, np.array([6, 8]) / 10 * 2 * summary["scales"]["arrow_mm_per_n"]
  )
  arrow.set_linestyle.assert_called_once_with("--")
  assert "10.000 N" in label.set_text.call_args.args[0]
  np.testing.assert_allclose(archive["rollout_center_local_mm"][0, 0], [2, 3])
  # A later release never gets a fabricated center or an arrow anchored at zero.
  _draw_sample(panel, "rollout", 0, 2, archive, summary, lambda _: "red", lambda x: x)
  circle.set_visible.assert_called_with(False)
  arrow.set_visible.assert_called_with(False)
  status.set_text.assert_called_with("无接触")
  assert "-4.000 N" in label.set_text.call_args.args[0]


def test_current_contact_with_unknown_change_renders_gray_not_zero_delta():
  trace = _trace()
  summary, archive = build_glyph_samples(trace, trace, [0])
  panel = tuple(Mock() for _ in range(4))
  _draw_sample(panel, "rollout", 0, 0, archive, summary, lambda _: "red", lambda x: x)
  panel[0].set_facecolor.assert_called_with("#94a3b8")
  panel[1].set_visible.assert_called_with(False)
  assert "ΔFn 未知" in panel[3].set_text.call_args.args[0]


def test_tiny_force_increment_does_not_get_a_fixed_large_arrowhead():
  trace = _trace(tangent=[[0, 0], [0.005, 0], [0.505, 0], [0.505, 0]])
  summary, archive = build_glyph_samples(trace, trace, [0.1, 0.2, 0.3])
  panel = tuple(Mock() for _ in range(4))
  arrow = panel[1]
  _draw_sample(panel, "rollout", 0, 0, archive, summary, lambda _: "red", lambda x: x)
  small_head = arrow.set_mutation_scale.call_args.args[0]
  small_start, small_end = arrow.set_positions.call_args.args
  _draw_sample(panel, "rollout", 0, 1, archive, summary, lambda _: "red", lambda x: x)
  large_head = arrow.set_mutation_scale.call_args.args[0]
  large_start, large_end = arrow.set_positions.call_args.args
  assert small_head == pytest.approx(large_head / 100)
  assert np.linalg.norm(small_end - small_start) == pytest.approx(
    np.linalg.norm(large_end - large_start) / 100
  )
  _draw_sample(panel, "rollout", 0, 2, archive, summary, lambda _: "red", lambda x: x)
  arrow.set_visible.assert_called_with(False)


@pytest.mark.parametrize(
  "setting",
  [
    {"window_s": 0},
    {"window_s": np.nan},
    {"contact_threshold_n": -1},
    {"arrow_display_cap_n": 0},
  ],
)
def test_invalid_settings_fail_before_building(setting):
  with pytest.raises(ValueError):
    build_glyph_samples(_trace(), _trace(), [0.1], **setting)


@pytest.mark.parametrize(
  "times", [[], [0.1, 0.1], [0.2, 0.1], [np.nan], [-0.01], [0.31], [[0.1]]]
)
def test_invalid_display_clock(times):
  with pytest.raises(ValueError, match="展示时间"):
    build_glyph_samples(_trace(), _trace(), times)


def test_coordinate_mismatch_is_rejected():
  model, expert = _trace(), _trace()
  model.taxel_positions_local_m[0, 0, 0] += 0.001
  with pytest.raises(ValueError, match="不一致"):
    build_glyph_samples(model, expert, [0.1])


def test_export_chinese_snapshots_and_lossless_arrays_never_overwrites(tmp_path):
  trace = _trace(metadata={"source": "task_demo", "unknown": np.nan})
  result = export_tactile_glyphs(
    trace, trace, tmp_path, snapshot_times_s=[0.1, 0.2], include_video=False
  )
  assert len(result["artifacts"]) == 3
  assert (tmp_path / "tactile_overall_change.png").stat().st_size > 1000
  summary = json.loads((tmp_path / "tactile_overall_change.json").read_text())
  assert summary["language"] == "zh-CN"
  assert summary["reference"]["metadata"]["unknown"] is None
  assert not summary["video"]["generated"]
  with np.load(tmp_path / "tactile_overall_change.npz") as archive:
    np.testing.assert_allclose(archive["rollout_normal_force_n"], 1)
    np.testing.assert_allclose(archive["pad_center_local_mm"], [[2, 3]] * 5)
  with pytest.raises(FileExistsError):
    export_tactile_glyphs(
      trace, trace, tmp_path, snapshot_times_s=[0.1], include_video=False
    )


@pytest.mark.parametrize(
  "settings",
  [
    {"snapshot_times_s": [0.4]},
    {"snapshot_times_s": []},
    {"snapshot_times_s": [0.2, 0.1]},
    {"frame_interval_s": 0},
    {"video_fps": 0},
    {"window_s": 0},
  ],
)
def test_invalid_export_writes_nothing(tmp_path, settings):
  values = {"snapshot_times_s": [0.1], "include_video": False, **settings}
  with pytest.raises(ValueError):
    export_tactile_glyphs(_trace(), _trace(), tmp_path, **values)
  assert not list(tmp_path.iterdir())


def test_video_speed_mismatch_is_rejected_before_writing(tmp_path):
  with pytest.raises(ValueError, match="隐式加速"):
    export_tactile_glyphs(
      _trace(),
      _trace(),
      tmp_path,
      snapshot_times_s=[0.1],
      frame_interval_s=0.1,
      video_fps=20,
    )
  assert not list(tmp_path.iterdir())
