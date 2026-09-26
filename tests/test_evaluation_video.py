"""Layout, metadata, and control-rate sampling checks for evaluation video."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared import evaluation_video
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.evaluation_video import compose_evaluation_frame
from kaihand_tactile_env.shared.policy_cameras import (
  include_model_right_wrist_panel,
)
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from PIL import Image


def test_compositor_is_1080p_and_accepts_legacy_signed_force_history():
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


def test_compositor_enlarges_camera_and_heatmaps_without_drawing_curves(
  monkeypatch,
):
  camera_boxes = []
  heatmap_boxes = []
  draw_camera = evaluation_video._draw_camera
  draw_heatmap = evaluation_video._draw_heatmap

  def record_camera(image, draw, pixels, box, label):
    camera_boxes.append(box)
    draw_camera(image, draw, pixels, box, label)

  def record_heatmap(image, draw, values, box, label, maximum):
    heatmap_boxes.append(box)
    draw_heatmap(image, draw, values, box, label, maximum)

  def reject_curve(*args, **kwargs):
    raise AssertionError("evaluation video must not draw time-series curves")

  monkeypatch.setattr(evaluation_video, "_draw_camera", record_camera)
  monkeypatch.setattr(evaluation_video, "_draw_heatmap", record_heatmap)
  monkeypatch.setattr(evaluation_video, "_draw_curve", reject_curve)
  rgb = np.zeros((480, 640, 3), dtype=np.uint8)

  compose_evaluation_frame(
    rgb,
    rgb,
    np.zeros((10, 7, 5)),
    np.zeros((10, 7, 5, 2)),
    np.zeros((1, 10, 3)),
    simulation_time_s=0.0,
    phase="test",
    second_camera_label="GLOBAL",
  )

  assert len(camera_boxes) == 2
  assert len(heatmap_boxes) == 20
  assert camera_boxes[0][2] - camera_boxes[0][0] > 800
  assert min(box[0] for box in heatmap_boxes) == int(1920 * 0.45) + 8
  assert max(box[2] for box in heatmap_boxes) == 1920 - 8
  assert min(box[2] - box[0] for box in heatmap_boxes) > 200


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


def test_compositor_labels_review_only_wrist_without_claiming_model_input(monkeypatch):
  labels = []
  original = evaluation_video._draw_camera

  def record_camera(image, draw, pixels, box, label):
    labels.append(label)
    original(image, draw, pixels, box, label)

  monkeypatch.setattr(evaluation_video, "_draw_camera", record_camera)
  rgb = np.zeros((24, 32, 3), dtype=np.uint8)
  compose_evaluation_frame(
    rgb,
    rgb,
    np.zeros((10, 7, 5)),
    np.zeros((10, 7, 5, 2)),
    np.zeros((1, 10, 3)),
    simulation_time_s=0.0,
    phase="test",
    second_camera_label="GLOBAL",
    review_wrist_rgb=rgb,
  )

  assert labels == [
    "HEAD / MODEL VIEW",
    "RIGHT WRIST / REVIEW ONLY",
    "GLOBAL",
  ]


def test_comparison_samples_every_control_tick_even_when_video_skips(tmp_path, monkeypatch):
  render_cameras = []
  composed = []
  comparison_instances = []
  provider_reads = []

  class FakeRenderer:
    def __init__(self, model, *, height, width):
      self.scene = SimpleNamespace(flags={})
      self.height, self.width = height, width

    def update_scene(self, data, *, camera):
      render_cameras.append(camera)

    def render(self):
      return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
      pass

  class FakeWriter:
    def __init__(self, path, *, fps, width, height):
      self.frames = []

    def write(self, frame):
      self.frames.append(frame)

    def finish(self):
      pass

  class FakeComparisonTrace:
    def __init__(self, *, tactile_provider):
      self.timestamps = []
      self.provider = tactile_provider
      self.reference_dataset = None
      comparison_instances.append(self)

    def capture(
      self,
      simulation,
      *,
      timestamp=None,
      normal_taxel_force_n=None,
      tangent_taxel_force_n=None,
    ):
      if normal_taxel_force_n is None or tangent_taxel_force_n is None:
        self.provider.read(simulation.data)
      self.timestamps.append(simulation.data.time if timestamp is None else timestamp)
      return True

    def finish(self, output_dir, *, reference_dataset, reference_episode_index):
      self.reference_dataset = reference_dataset
      assert reference_episode_index == 0
      return {"status": "ok", "rollout": {"samples": len(self.timestamps)}}

  def record_compose(*args, **kwargs):
    composed.append(kwargs)
    return Image.new("RGB", (320, 240))

  monkeypatch.setattr(evaluation_video.mujoco, "Renderer", FakeRenderer)
  monkeypatch.setattr(evaluation_video, "_FfmpegPipeWriter", FakeWriter)
  monkeypatch.setattr(evaluation_video, "current_backend", lambda: "test")
  monkeypatch.setattr(evaluation_video, "compose_evaluation_frame", record_compose)
  monkeypatch.setitem(
    sys.modules,
    "kaihand_tactile_env.shared.evaluation_comparison",
    SimpleNamespace(EvaluationComparisonTrace=FakeComparisonTrace),
  )
  def read_provider(data):
    provider_reads.append(data.time)
    return SimpleNamespace(
      link_names=FINGERTIP_LINK_NAMES,
      normal_taxel_force_n=np.zeros((10, 7, 5)),
      tangent_taxel_force_n=np.zeros((10, 7, 5, 2)),
    )

  provider = SimpleNamespace(
    link_names=FINGERTIP_LINK_NAMES,
    source="test",
    taxel_force_semantics="test",
    read=read_provider,
  )
  simulation = SimpleNamespace(
    model=object(), data=SimpleNamespace(time=0.0), scene="test"
  )
  output = tmp_path / "review"
  recorder = evaluation_video.EvaluationVideo(
    simulation,
    output,
    width=320,
    height=240,
    render_width=32,
    render_height=24,
    include_review_wrist=True,
    comparison=True,
    tactile_provider=provider,
    metadata={
      "observation_contract": {"cameras": ["head"]},
      "reference_dataset": "/data/reference",
    },
  )
  for tick in range(4):
    simulation.data.time = tick / 30.0
    recorder.capture(tick, "policy")
  simulation.data.time = 4 / 30.0
  recorder.capture_due("policy")
  result = recorder.finish(status="completed", evaluation={})

  assert comparison_instances[0].timestamps == pytest.approx(
    [0.0, 1 / 30, 2 / 30, 3 / 30, 4 / 30]
  )
  assert provider_reads == pytest.approx([0.0, 1 / 30, 2 / 30, 3 / 30, 4 / 30])
  assert comparison_instances[0].reference_dataset == Path("/data/reference")
  assert len(composed) == 2
  assert render_cameras[0::3] == ["head", "head"]
  assert all(isinstance(camera, evaluation_video.mujoco.MjvCamera) for camera in render_cameras[1::3])
  assert render_cameras[2::3] == ["right_wrist", "right_wrist"]
  assert all(item["review_wrist_rgb"] is not None for item in composed)
  assert all(item["model_wrist_rgb"] is None for item in composed)
  assert result["frame_count"] == 2
  assert result["model_input_cameras_displayed"] == ["head"]
  assert result["right_wrist_camera_role"] == "review_only"
  assert result["comparison_plots"]["rollout"]["samples"] == 5
  assert json.loads((output / "review.json").read_text())["comparison_plots"]["status"] == "ok"

  camera_cases = (
    ("global", True, {"observation_contract": {"cameras": ["head", "right_wrist"]}}),
    ("right_wrist", False, {"observation_contract": {"cameras": ["head", "right_wrist"]}}),
    ("right_wrist", False, {"server_metadata": {"cameras": ["head", "right_wrist"]}}),
  )
  for case_index, (second_camera, include_model_wrist, camera_metadata) in enumerate(camera_cases):
    model_recorder = evaluation_video.EvaluationVideo(
      simulation,
      tmp_path / f"model_{case_index}",
      width=320,
      height=240,
      render_width=32,
      render_height=24,
      second_camera=second_camera,
      include_model_wrist=include_model_wrist,
      include_review_wrist=True,
      tactile_provider=provider,
      metadata=camera_metadata,
    )
    assert model_recorder.metadata["model_input_cameras_displayed"] == [
      "head", "right_wrist"
    ]
    assert model_recorder.metadata["right_wrist_camera_role"] == "model_input"
    model_recorder.finish(status="completed", evaluation={})

  def failed_finish(self, output_dir, *, reference_dataset, reference_episode_index):
    return {"status": "error", "reason": "plot export failed"}

  monkeypatch.setattr(FakeComparisonTrace, "finish", failed_finish)
  failed_recorder = evaluation_video.EvaluationVideo(
    simulation,
    tmp_path / "failed_comparison",
    width=320,
    height=240,
    render_width=32,
    render_height=24,
    comparison=True,
    tactile_provider=provider,
  )
  failed_recorder.capture(0, "policy")
  with pytest.raises(RuntimeError, match="comparison artifact failed"):
    failed_recorder.finish(status="completed", evaluation={})
  failed_metadata = json.loads(
    (tmp_path / "failed_comparison/review.json").read_text()
  )
  assert failed_metadata["comparison_plots"]["status"] == "error"


def test_right_wrist_model_input_gets_a_dedicated_review_panel():
  assert include_model_right_wrist_panel(("head", "right_wrist"), "global")
  assert include_model_right_wrist_panel(
    ("head", "left_wrist", "right_wrist"), "overhead"
  )
  assert not include_model_right_wrist_panel(("head",), "global")
  assert not include_model_right_wrist_panel(
    ("head", "right_wrist"), "right_wrist"
  )


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
