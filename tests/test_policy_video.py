"""Recorder checks without loading a robot, OpenGL or encoder."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared import evaluation_video
from kaihand_tactile_env.shared import policy_video as video
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES


@pytest.fixture
def setup(monkeypatch, tmp_path):
  normal = np.full((10, 7, 5), 0.001)
  tangent = np.zeros((10, 7, 5, 2))
  tangent[..., 0] = -0.0003
  tangent[..., 1] = 0.0004
  provider = SimpleNamespace(
    source="test",
    taxel_force_semantics="test",
    link_names=FINGERTIP_LINK_NAMES,
    read=lambda data: SimpleNamespace(
      link_names=FINGERTIP_LINK_NAMES,
      normal_taxel_force_n=normal,
      tangent_taxel_force_n=tangent,
    ),
  )
  state = SimpleNamespace(frames=[], cameras=[], closed=False, fail=False)

  class Renderer:
    def __init__(self, *args, **kwargs):
      self.scene = SimpleNamespace(flags=np.ones(32))

    def update_scene(self, data, *, camera):
      state.cameras.append(camera if isinstance(camera, str) else "global")

    def render(self):
      return np.full((480, 640, 3), 70, dtype=np.uint8)

    def close(self):
      state.closed = True

  class Writer:
    def __init__(self, *args, **kwargs):
      pass

    def write(self, rgb):
      state.frames.append(rgb.copy())

    def finish(self):
      if state.fail:
        raise RuntimeError("encoder failed")

    def abort(self):
      pass

  monkeypatch.setattr(
    evaluation_video, "_create_evaluation_spatial_provider", lambda sim: provider
  )
  monkeypatch.setattr(evaluation_video.mujoco, "Renderer", Renderer)
  monkeypatch.setattr(evaluation_video, "_FfmpegPipeWriter", Writer)
  monkeypatch.setattr(
    evaluation_video, "current_backend", lambda: {"software": False}
  )
  sim = SimpleNamespace(
    model=object(), data=SimpleNamespace(time=0), observation_time=0,
    scene="poker-draw",
  )
  return sim, state, tmp_path / "review"


def test_schedule_sync_signed_forces_and_failure_status(setup):
  sim, state, path = setup
  recorder = video.PokerPolicyVideo(sim, path)
  for tick in range(7):
    sim.data.time = tick / 30
    sim.observation_time = max(0, sim.data.time - 0.002)
    recorder.capture(tick, "approach")
  recorder.capture(6, "approach", force=True)
  metadata = recorder.finish(status="task_not_completed", evaluation={"success": False})
  assert len(state.frames) == 3
  assert state.frames[0].shape == (1080, 1920, 3)
  assert state.cameras == ["head", "global"] * 3
  assert state.closed and metadata["completed"]
  assert metadata["schema"] == "kaihand-policy-evaluation-review-v3"
  assert metadata["time_series_displayed"] is False
  assert "curves" not in metadata["layout"]
  assert metadata["task_status"] == "task_not_completed"
  assert metadata["evaluation"]["success"] is False
  rows = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
  assert [row["control_tick"] for row in rows] == [0, 3, 6]
  assert all(row["camera_pose_time_s"] == row["tactile_time_s"] for row in rows)
  normal = np.asarray(rows[-1]["normal_taxel_force_n"])
  assert normal.shape == (10, 7, 5)
  np.testing.assert_allclose(normal, 0.001)
  tangent = np.asarray(rows[-1]["tangent_taxel_force_n"])
  assert tangent.shape == (10, 7, 5, 2)
  np.testing.assert_allclose(np.linalg.norm(tangent, axis=-1), 0.0005)
  assert np.all(tangent[..., 0] < 0)
  means = np.asarray(rows[-1]["force_mean_n_per_taxel"])
  assert means.shape == (10, 3)
  np.testing.assert_allclose(means[0], [-0.0003, 0.0004, 0.001])
  assert recorder.finish(status="ignored", evaluation=None) == metadata


def test_off_grid_terminal_frame_and_no_overwrite(setup):
  sim, state, path = setup
  recorder = video.PokerPolicyVideo(sim, path, fps=5)
  with pytest.raises(FileExistsError):
    video.PokerPolicyVideo(sim, path)
  recorder.capture(0, "approach")
  sim.data.time = 0.034
  sim.observation_time = 0.032
  recorder.capture(1, "approach", force=True)
  recorder.finish(status="error", evaluation=None, error="IK guard")
  assert len(state.frames) == 2


def test_encoder_failure_closes_resources_and_marks_incomplete(setup):
  sim, state, path = setup
  recorder = video.PokerPolicyVideo(sim, path)
  recorder.capture(0, "approach")
  state.fail = True
  with pytest.raises(RuntimeError, match="encoder failed"):
    recorder.finish(status="task_not_completed", evaluation=None)
  assert state.closed
  assert json.loads((path / "review.json").read_text())["completed"] is False
