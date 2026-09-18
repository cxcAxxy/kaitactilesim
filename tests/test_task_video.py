from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared import task_video
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from PIL import Image, ImageDraw

RIGHT_FINGERTIP_LINK_NAMES = tuple(
  name for name in FINGERTIP_LINK_NAMES if name.startswith("hand_r_")
)


class _FakeScene:
  def __init__(self) -> None:
    self.flags = np.ones(int(mujoco.mjtRndFlag.mjNRNDFLAG), dtype=np.uint8)


class _FakeRenderer:
  instances: list[_FakeRenderer] = []

  def __init__(self, _model: object, *, height: int, width: int) -> None:
    self.height = height
    self.width = width
    self.scene = _FakeScene()
    self.cameras: list[object] = []
    self.depth_disabled = False
    self.segmentation_disabled = False
    self.closed = False
    self.instances.append(self)

  def disable_depth_rendering(self) -> None:
    self.depth_disabled = True

  def disable_segmentation_rendering(self) -> None:
    self.segmentation_disabled = True

  def update_scene(self, _data: object, *, camera: object) -> None:
    self.cameras.append(camera)

  def render(self) -> np.ndarray:
    value = 40 if self.cameras[-1] == "front" else 120
    return np.full((self.height, self.width, 3), value, dtype=np.uint8)

  def close(self) -> None:
    self.closed = True


class _FakeWriter:
  instances: list[_FakeWriter] = []

  def __init__(
    self,
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    executable: str,
  ) -> None:
    self.path = path
    self.fps = fps
    self.width = width
    self.height = height
    self.executable = executable
    self.frames: list[np.ndarray] = []
    self.finished = False
    self.instances.append(self)

  def write(self, frame: np.ndarray) -> None:
    self.frames.append(frame.copy())

  def finish(self) -> None:
    if not self.finished:
      self.finished = True
      self.path.touch(exist_ok=False)

  def close(self) -> None:
    self.finish()


class _FakePreview:
  instances: list[_FakePreview] = []

  def __init__(self, *, width: int, height: int) -> None:
    self.width = width
    self.height = height
    self.frames: list[np.ndarray] = []
    self.closed = False
    self.instances.append(self)

  def update(self, frame: np.ndarray) -> None:
    self.frames.append(frame.copy())

  def close(self) -> None:
    self.closed = True


class _FakeLayout:
  def __init__(self) -> None:
    self.body_names = tuple(name for name in FINGERTIP_LINK_NAMES for _ in range(35))
    self.grid_shape = tuple((7, 5) for _ in self.body_names)
    self.count = len(self.body_names)


class _FakeTactile:
  source = "fake_genesis"
  force_unit = "genesis_depth_force_proxy"
  available = True

  def __init__(self) -> None:
    self.layout = _FakeLayout()
    self.probe_depth = np.zeros(self.layout.count, dtype=np.float64)
    self.read_count = 0

  def read(self, data: object) -> SimpleNamespace:
    self.read_count += 1
    self.probe_depth[:] = self.read_count * 1.0e-4
    return SimpleNamespace(
      timestamp=float(data.time),
      link_names=FINGERTIP_LINK_NAMES,
      normal_force=np.arange(1.0, 11.0),
    )


class _FakePokerSpatialTactile:
  source = "solver_contact_distributed_taxel_v1"
  force_unit = "N"
  taxel_force_unit = "N"
  available = True

  def __init__(self) -> None:
    self.link_names = RIGHT_FINGERTIP_LINK_NAMES
    self.kernel_sigma_m = 0.003
    self.read_count = 0

  def metadata(self) -> dict[str, object]:
    return {
      "algorithm": "normalized_gaussian_3d_taxel_distance",
      "kernel_sigma_m": self.kernel_sigma_m,
      "tangent_basis": ["grid_col_positive", "grid_row_positive"],
    }

  def read(self, data: object) -> SimpleNamespace:
    self.read_count += 1
    normal_maps = np.zeros((5, 7, 5), dtype=np.float64)
    tangent_components = np.zeros((5, 7, 5, 2), dtype=np.float64)
    for finger_index in range(5):
      normal_maps[finger_index, finger_index, 0] = finger_index + 0.25
      tangent_components[finger_index, 0, finger_index, 0] = finger_index + 3.0
      tangent_components[finger_index, 0, finger_index, 1] = 4.0
    return SimpleNamespace(
      timestamp=float(data.time),
      link_names=self.link_names,
      normal_force_n=np.arange(1.0, 6.0),
      tangent_force_n=np.column_stack((np.arange(3.0, 8.0), np.full(5, 4.0))),
      tangent_load_n=np.arange(5.0),
      normal_taxel_force_n=normal_maps,
      tangent_taxel_force_n=tangent_components,
      tangent_taxel_load_n=np.linalg.norm(tangent_components, axis=-1),
    )


class _FakeSimulation:
  def __init__(self, *, scene: str = "pick-place") -> None:
    self.model = object()
    self.data = SimpleNamespace(time=0.0)
    self.genesis_probe_layout = _FakeLayout()
    self.scene = scene


def test_video_sidecar_keeps_measured_force_outcome(tmp_path, fake_backends):
  simulation = _FakeSimulation()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "force_outcome.mp4",
    tactile_provider=_FakeTactile(),
    preview=False,
  )
  with recorder:
    recorder.observe(simulation, "initial")
    recorder.set_outcome(
      {
        "press_force_target_per_finger_n": 0.35,
        "slide_finger_normal_force_means_n": (0.34, 0.35, 0.36, 0.35),
        "final_card_pose": np.zeros(7),
      }
    )
    _, sidecar = recorder.finish(success=True)
  outcome = json.loads(sidecar.read_text())["task_outcome"]
  assert outcome["slide_finger_normal_force_means_n"] == [0.34, 0.35, 0.36, 0.35]
  assert outcome["final_card_pose"] == [0] * 7
  with pytest.raises(RuntimeError, match="finished"):
    recorder.set_outcome({})


@pytest.fixture
def fake_backends(monkeypatch: Any) -> None:
  _FakeRenderer.instances.clear()
  _FakeWriter.instances.clear()
  _FakePreview.instances.clear()
  monkeypatch.setattr(task_video.mujoco, "Renderer", _FakeRenderer)
  monkeypatch.setattr(task_video, "_FfmpegPipeWriter", _FakeWriter)
  monkeypatch.setattr(task_video, "_TkPreview", _FakePreview)
  monkeypatch.setattr(task_video, "_find_ffmpeg_executable", lambda: "/fake/ffmpeg")


def test_recorder_uses_one_rgb_renderer_and_streams_same_preview_frame(
  tmp_path: Path,
  fake_backends: None,
) -> None:
  simulation = _FakeSimulation()
  tactile = _FakeTactile()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "episode.mp4",
    fps=10,
    width=320,
    height=240,
    tactile_provider=tactile,
    preview=True,
  )

  recorder.observe(simulation, "initial")

  assert len(_FakeRenderer.instances) == 1
  renderer = _FakeRenderer.instances[0]
  assert renderer.depth_disabled
  assert renderer.segmentation_disabled
  assert not renderer.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)]
  assert renderer.cameras == ["front", "overhead"]
  assert len(_FakeWriter.instances[0].frames) == 1
  assert _FakeWriter.instances[0].frames[0].shape == (240, 320, 3)
  assert _FakeWriter.instances[0].frames[0].dtype == np.uint8
  np.testing.assert_array_equal(
    _FakeWriter.instances[0].frames[0], _FakePreview.instances[0].frames[0]
  )
  recorder.close()


def test_poker_video_uses_independent_right_hand_fn_ft_solver_fields(
  tmp_path: Path,
  fake_backends: None,
  monkeypatch: Any,
) -> None:
  simulation = _FakeSimulation(scene="poker-draw")
  legacy_tactile = _FakeTactile()
  poker_tactile = _FakePokerSpatialTactile()
  monkeypatch.setattr(
    task_video,
    "_create_poker_spatial_provider",
    lambda _simulation: poker_tactile,
  )
  captured: dict[str, np.ndarray] = {}

  def fake_compose(
    _main: np.ndarray,
    _overhead: np.ndarray,
    normal_maps: np.ndarray,
    tangent_maps: np.ndarray,
    normal_forces: np.ndarray,
    tangent_forces: np.ndarray,
    **_kwargs: object,
  ) -> np.ndarray:
    captured["normal_maps"] = normal_maps.copy()
    captured["tangent_maps"] = tangent_maps.copy()
    captured["normal_forces"] = normal_forces.copy()
    captured["tangent_forces"] = tangent_forces.copy()
    return np.zeros((240, 320, 3), dtype=np.uint8)

  monkeypatch.setattr(task_video, "_compose_poker_dashboard", fake_compose)
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "poker.mp4",
    width=320,
    height=240,
    tactile_provider=legacy_tactile,
    preview=False,
  )
  recorder.observe(simulation, "pinch")

  assert legacy_tactile.read_count == 0
  assert poker_tactile.read_count == 1
  assert captured["normal_maps"].shape == (5, 7, 5)
  assert captured["tangent_maps"].shape == (5, 7, 5)
  assert captured["normal_forces"].tolist() == [1, 2, 3, 4, 5]
  np.testing.assert_allclose(
    captured["tangent_forces"],
    np.sqrt(np.arange(3.0, 8.0) ** 2 + 4.0**2),
  )
  assert captured["tangent_maps"][0, 0, 0] == pytest.approx(5.0)
  assert _FakeRenderer.instances[0].cameras == ["head", "right_wrist"]
  # Explicit legacy diagnostic close-ups keep their global/overhead pair;
  # resetting the override restores the new named head/right-wrist views.
  recorder.follow_viewer_camera(mujoco.MjvCamera())
  assert recorder._secondary_camera == "overhead"
  recorder.follow_viewer_camera(None)
  assert recorder._main_camera == "head"
  assert recorder._secondary_camera == "right_wrist"

  recorder.finish(success=True)
  metadata = json.loads((tmp_path / "poker.json").read_text(encoding="utf-8"))
  assert metadata["cameras"] == {"main": "fixed:head", "secondary": "right_wrist"}
  tactile_metadata = metadata["tactile"]
  assert tactile_metadata["source"] == "solver_contact_distributed_taxel_v1"
  assert tactile_metadata["hand"] == "right"
  assert tactile_metadata["link_names"] == list(RIGHT_FINGERTIP_LINK_NAMES)
  assert tactile_metadata["spatial_estimate"] is True
  assert tactile_metadata["taxel_force_unit"] == "N"
  assert tactile_metadata["display_taxel_color_unit"] == "N/taxel"
  assert "depth_unit" not in tactile_metadata
  assert tactile_metadata["display_rows"] == [
    "normal_taxel_force_n",
    "norm(tangent_taxel_force_n, axis=-1)",
  ]
  assert tactile_metadata["color_scales"]["normal_taxel_force_n"] == {
    "fixed": True,
    "maximum": 0.1,
    "minimum": 0.0,
    "saturates_above_maximum": True,
    "unit": "N/taxel",
  }
  assert (
    tactile_metadata["spatial_distribution"]["algorithm"]
    == "normalized_gaussian_3d_taxel_distance"
  )


def test_poker_dashboard_draws_two_rows_for_only_five_right_fingers(
  monkeypatch: Any,
) -> None:
  calls: list[tuple[str, str, float, np.ndarray]] = []

  def fake_cell(
    _image: object,
    _draw: object,
    taxel_map: np.ndarray,
    force: float,
    _box: object,
    label: str,
    *,
    quantity: str,
    scale_maximum: float,
  ) -> None:
    calls.append((quantity, label, scale_maximum, taxel_map.copy()))
    assert force >= 0.0

  monkeypatch.setattr(task_video, "_draw_force_tactile_cell", fake_cell)
  main = np.zeros((60, 80, 3), dtype=np.uint8)
  maps = np.arange(5 * 7 * 5, dtype=np.float64).reshape(5, 7, 5)
  frame = task_video._compose_poker_dashboard(
    main,
    main,
    maps,
    maps + 1000.0,
    np.arange(5.0),
    np.arange(5.0),
    width=320,
    height=240,
    sim_time=1.0,
    phase="look",
    status=("RECORDING", None),
    error=None,
    tactile_source="solver_contact_distributed_taxel_v1",
  )

  assert frame.shape == (240, 320, 3)
  assert [call[:3] for call in calls] == [
    *(("Fn", label, 0.1) for label in ("Th", "In", "Mi", "Ri", "Pi")),
    *(("Ft", label, 0.02) for label in ("Th", "In", "Mi", "Ri", "Pi")),
  ]
  for index in range(5):
    np.testing.assert_array_equal(calls[index][3], maps[index])
    np.testing.assert_array_equal(calls[index + 5][3], maps[index] + 1000.0)


def test_poker_fixed_normal_scale_makes_gentle_taxel_load_visible() -> None:
  image = Image.new("RGB", (50, 100), (0, 0, 0))
  taxels = np.zeros((7, 5), dtype=np.float64)
  taxels[3, 2] = 0.02
  task_video._draw_force_tactile_cell(
    image,
    ImageDraw.Draw(image),
    taxels,
    0.02,
    (0, 0, 50, 100),
    "In",
    quantity="Fn",
    scale_maximum=task_video._MAX_NORMAL_TAXEL_FORCE_N,
  )

  expected_color = task_video._HEATMAP_LUT[int(round(0.02 / 0.10 * 255.0))]
  pixels = np.asarray(image)
  assert np.any(np.all(pixels == expected_color, axis=-1))
  assert not np.array_equal(expected_color, task_video._HEATMAP_LUT[0])


def test_simulation_clock_throttles_without_render_catchup(
  tmp_path: Path,
  fake_backends: None,
) -> None:
  simulation = _FakeSimulation()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "episode.mp4",
    fps=10,
    width=320,
    height=240,
    tactile_provider=_FakeTactile(),
    preview=False,
  )

  recorder.observe(simulation, "initial")
  simulation.data.time = 0.05
  recorder.observe(simulation, "approach")
  simulation.data.time = 0.10
  recorder.observe(simulation, "approach")
  simulation.data.time = 0.35
  recorder.observe(simulation, "pinch")
  recorder.finish(success=True)

  # One initial, two due samples and one explicit terminal frame.  A large
  # timestamp jump never causes a burst of duplicate catch-up renders.
  assert len(_FakeWriter.instances[0].frames) == 4
  metadata = json.loads((tmp_path / "episode.json").read_text(encoding="utf-8"))
  assert metadata["success"] is True
  assert metadata["recording_complete"] is True
  assert metadata["tactile"]["normal_force_unit"] == "genesis_depth_force_proxy"
  assert metadata["tactile"]["link_names"] == list(FINGERTIP_LINK_NAMES)
  assert metadata["tactile"]["depth_unit"] == "mm"
  assert metadata["frame_count"] == 4
  assert [event["phase"] for event in metadata["phases"]] == [
    "initial",
    "approach",
    "pinch",
  ]
  assert metadata["first_simulation_time"] == pytest.approx(0.0)
  assert metadata["last_simulation_time"] == pytest.approx(0.35)
  assert recorder._next_sample_time == pytest.approx(0.4)


def test_failure_is_marked_and_context_manager_preserves_video(
  tmp_path: Path,
  fake_backends: None,
) -> None:
  simulation = _FakeSimulation()
  with pytest.raises(RuntimeError, match="task failed"):
    with task_video.TaskVideoRecorder(
      simulation,
      tmp_path / "failed.mp4",
      width=320,
      height=240,
      tactile_provider=_FakeTactile(),
      preview=False,
    ) as recorder:
      recorder.observe(simulation, "slide")
      raise RuntimeError("task failed")

  assert (tmp_path / "failed.mp4").is_file()
  metadata = json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))
  assert metadata["success"] is False
  assert metadata["episode_completed"] is False
  assert metadata["recording_complete"] is False
  assert metadata["incomplete"] is True
  assert metadata["error"] == "task failed"
  assert metadata["frame_count"] == 2
  final_frame = _FakeWriter.instances[0].frames[-1]
  # FAILED has a red header badge and the error banner is also red.
  assert final_frame[5, -5, 0] > final_frame[5, -5, 1]
  assert _FakeRenderer.instances[0].closed


def test_15fps_clock_has_no_accumulated_500hz_quantization_drift(
  tmp_path: Path,
  fake_backends: None,
  monkeypatch: Any,
) -> None:
  simulation = _FakeSimulation()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "clock.mp4",
    fps=15,
    tactile_provider=_FakeTactile(),
    preview=False,
  )
  timestamps = []
  monkeypatch.setattr(recorder, "_capture", lambda t, phase: timestamps.append(t))
  for step in range(6501):
    simulation.data.time = 10.0 + step / 500.0
    recorder.observe(simulation, "clock_only")
  assert len(timestamps) == 196
  quantization = np.asarray(timestamps) - (10.0 + np.arange(196) / 15.0)
  assert np.min(quantization) >= -1.0e-10
  assert np.max(quantization) < 0.002 + 1.0e-10
  assert _FakeRenderer.instances[0].cameras == []
  recorder.close()


@pytest.mark.parametrize("conflict", ["episode.mp4", "episode.json"])
def test_recorder_never_overwrites_output_or_sidecar(
  tmp_path: Path,
  fake_backends: None,
  conflict: str,
) -> None:
  (tmp_path / conflict).write_bytes(b"existing")

  with pytest.raises(FileExistsError, match="never overwrites"):
    task_video.TaskVideoRecorder(
      _FakeSimulation(),
      tmp_path / "episode.mp4",
      tactile_provider=_FakeTactile(),
      preview=False,
    )

  assert not _FakeRenderer.instances
  assert not _FakeWriter.instances


def test_missing_ffmpeg_has_actionable_error(monkeypatch: Any) -> None:
  monkeypatch.setattr(task_video.shutil, "which", lambda _: None)
  monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

  with pytest.raises(RuntimeError, match="pixi install"):
    task_video._find_ffmpeg_executable()


class _FakeInput(io.BytesIO):
  closed_by_writer = False
  write_calls = 0

  def write(self, value: Any) -> int:
    self.write_calls += 1
    return super().write(value[:5])

  def close(self) -> None:
    self.closed_by_writer = True
    super().close()


class _FakeProcess:
  def __init__(self) -> None:
    self.stdin = _FakeInput()
    self.stderr = io.BytesIO(b"")
    self.returncode: int | None = None
    self.terminated = False
    self.killed = False

  def poll(self) -> int | None:
    return self.returncode

  def wait(self, timeout: float) -> int:
    self.returncode = 0
    return 0

  def terminate(self) -> None:
    self.terminated = True
    self.returncode = -15

  def kill(self) -> None:
    self.killed = True
    self.returncode = -9


def test_ffmpeg_writer_requests_single_thread_and_closes_process(
  tmp_path: Path,
  monkeypatch: Any,
) -> None:
  process = _FakeProcess()
  commands: list[list[str]] = []

  def fake_popen(command: list[str], **_: Any) -> _FakeProcess:
    commands.append(command)
    return process

  monkeypatch.setattr(task_video.subprocess, "Popen", fake_popen)
  writer = task_video._FfmpegPipeWriter(
    tmp_path / "tiny.mp4",
    fps=15,
    width=4,
    height=2,
    executable="/fake/ffmpeg",
  )
  writer.write(np.zeros((2, 4, 3), dtype=np.uint8))
  writer.finish()
  writer.close()

  assert commands[0][commands[0].index("-threads") + 1] == "1"
  assert commands[0][commands[0].index("-filter_threads") + 1] == "1"
  assert process.stdin.closed_by_writer
  assert process.stdin.write_calls == 5
  assert process.returncode == 0


def test_encoding_failure_is_separate_from_task_success(
  tmp_path: Path,
  fake_backends: None,
  monkeypatch: Any,
) -> None:
  class FailingWriter(_FakeWriter):
    def finish(self) -> None:
      if not self.finished:
        self.finished = True
        self.path.touch(exist_ok=False)
      raise RuntimeError("encoder stopped")

    def close(self) -> None:
      if not self.finished:
        self.finish()

  monkeypatch.setattr(task_video, "_FfmpegPipeWriter", FailingWriter)
  simulation = _FakeSimulation()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "encoding_failed.mp4",
    width=320,
    height=240,
    tactile_provider=_FakeTactile(),
    preview=False,
  )
  recorder.observe(simulation, "initial")

  with pytest.raises(RuntimeError, match="task video is incomplete"):
    recorder.finish(success=True)

  metadata = json.loads((tmp_path / "encoding_failed.json").read_text(encoding="utf-8"))
  assert metadata["task_success"] is True
  assert metadata["success"] is False
  assert metadata["encoding_complete"] is False
  assert metadata["recording_complete"] is False
  assert "encoder stopped" in metadata["encoding_error"]


def test_solver_proxy_records_aggregate_newtons_without_fake_taxel_maps(
  tmp_path: Path,
  fake_backends: None,
) -> None:
  class SolverTactile:
    source = "solver_contact_proxy_v1"
    force_unit = "N"
    available = True

    def read(self, data: object) -> SimpleNamespace:
      return SimpleNamespace(
        timestamp=float(data.time),
        link_names=FINGERTIP_LINK_NAMES,
        normal_force=np.full(10, 0.5),
      )

  simulation = _FakeSimulation()
  recorder = task_video.TaskVideoRecorder(
    simulation,
    tmp_path / "solver.mp4",
    width=320,
    height=240,
    tactile_provider=SolverTactile(),
    preview=False,
  )
  recorder.observe(simulation, "initial")
  recorder.finish(success=True)

  metadata = json.loads((tmp_path / "solver.json").read_text(encoding="utf-8"))
  assert metadata["tactile"]["normal_force_unit"] == "N"
  assert metadata["tactile"]["spatial_taxel_maps"] is False
