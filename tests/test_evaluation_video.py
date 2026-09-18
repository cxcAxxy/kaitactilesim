"""Pure layout and task-metric checks for the common evaluation artifact."""

from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.evaluation_video import compose_evaluation_frame
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics


def test_compositor_is_1080p_and_preserves_signed_curve_input():
  rgb = np.zeros((480, 640, 3), dtype=np.uint8)
  normal = np.zeros((10, 7, 5))
  tangent = np.zeros((10, 7, 5, 2))
  history = np.zeros((3, 10, 3))
  history[:, 0, 0] = [-0.01, 0.0, 0.01]
  history[:, 5, 1] = [0.01, 0.0, -0.01]

  frame = compose_evaluation_frame(
    rgb,
    rgb,
    normal,
    tangent,
    history,
    simulation_time_s=1.25,
    phase="policy",
    second_camera_label="GLOBAL",
  )

  assert frame.size == (1920, 1080)
  assert np.asarray(frame).shape == (1080, 1920, 3)


def test_compositor_rejects_right_only_tactile():
  rgb = np.zeros((10, 10, 3), dtype=np.uint8)
  with pytest.raises(ValueError, match="10"):
    compose_evaluation_frame(
      rgb,
      rgb,
      np.zeros((5, 7, 5)),
      np.zeros((5, 7, 5, 2)),
      np.zeros((1, 10, 3)),
      simulation_time_s=0.0,
      phase="test",
      second_camera_label="GLOBAL",
    )


def test_compositor_can_show_both_model_cameras_and_global():
  rgb = np.zeros((240, 320, 3), dtype=np.uint8)
  frame = compose_evaluation_frame(
    rgb,
    rgb,
    np.zeros((10, 7, 5)),
    np.zeros((10, 7, 5, 2)),
    np.zeros((1, 10, 3)),
    simulation_time_s=0.0,
    phase="test",
    second_camera_label="GLOBAL",
    model_wrist_rgb=rgb,
  )

  assert frame.size == (1920, 1080)


def test_poker_metrics_preserve_negative_current_and_physics_peak():
  pose = np.array([0.40, 0.0, 0.8])
  simulation = SimpleNamespace(object_pose=lambda _: pose.copy())
  metrics = PokerReviewMetrics(0.40)

  pose[0] = 0.34
  metrics.update(simulation)
  pose[0] = 0.42
  metrics.update(simulation)
  snapshot = metrics.snapshot(simulation)

  assert snapshot["card_displacement_now_mm"] == pytest.approx(-20.0)
  assert snapshot["card_displacement_peak_mm"] == pytest.approx(60.0)
  assert snapshot["edge_distance_reached"] is True
  assert metrics.metadata()["edge_threshold_m"] == pytest.approx(0.05)


def test_poker_metrics_record_outcome_stages_per_frame():
  pose = np.array([0.40, 0.0, 0.8])
  simulation = SimpleNamespace(object_pose=lambda _: pose.copy())
  monitor = SimpleNamespace(
    success=False,
    lifted=False,
    edge_reached=False,
    contacted=True,
    failure_stage=lambda: "did_not_reach_edge",
  )
  metrics = PokerReviewMetrics(0.40, outcome_monitor=monitor)

  snapshot = metrics.snapshot(simulation)

  assert snapshot["success"] is False
  assert snapshot["success_stage"] == "contacted"
  assert snapshot["failure_stage"] == "did_not_reach_edge"
