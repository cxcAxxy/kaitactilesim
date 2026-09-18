"""Backend policy checks; no GL context, rendering or physics steps."""

import importlib.util
import sys
from pathlib import Path

import pytest
from kaihand_tactile_env.shared.render_backend import describe_backend


@pytest.mark.parametrize("renderer", ["llvmpipe (LLVM 15)", "softpipe", "Software Rasterizer"])
def test_hardware_never_silently_falls_back(renderer):
  with pytest.raises(RuntimeError, match="Hardware rendering required"):
    describe_backend(renderer, "Mesa", "4.6", "hardware")


def test_intel_gpu_is_hardware():
  info = describe_backend("Mesa Intel(R) Graphics (MTL)", "Intel", "4.6", "hardware")
  assert info["software"] is False


def test_software_mode_must_really_be_software():
  with pytest.raises(RuntimeError, match="Software rendering requested"):
    describe_backend("Mesa Intel(R) Graphics", "Intel", "4.6", "software")


@pytest.mark.parametrize("mode", ["hardware", "software"])
def test_strict_mode_rejects_unknown_renderer(mode):
  with pytest.raises(RuntimeError, match="Cannot identify"):
    describe_backend("", "", "", mode)


def test_auto_reports_unknown_without_claiming_hardware():
  assert describe_backend("", "", "", "auto")["software"] is None


@pytest.fixture
def batch():
  path = Path(__file__).resolve().parents[1] / "scripts/workcell/collect_poker_batch.py"
  spec = importlib.util.spec_from_file_location("poker_batch_backend_test", path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  yield module
  sys.modules.pop(spec.name, None)


def test_batch_defaults_to_single_hardware_instance(batch, tmp_path):
  args = batch.parse_args(["--output-dir", str(tmp_path)])
  assert args.render_backend == "hardware"
  assert args.workers == 1
  assert args.hdf5_buffer_rows == 64
  assert tuple(args.cameras) == ("head", "left_wrist", "right_wrist")
  assert args.nice_increment == 0
  assert args.verify_output_hash is False


@pytest.mark.parametrize("wrist", ["right_wrist", "left_wrist"])
def test_shared_camera_selection_reaches_poker_child(batch, tmp_path, wrist):
  args = batch.parse_args(["--output-dir", str(tmp_path), "--cameras", "head", wrist])
  command = batch.episode_command(args, 7)
  start = command.index("--cameras") + 1
  assert command[start:command.index("--width")] == ["head", wrist]
  assert command[command.index("--workers") + 1] == "1"


@pytest.mark.parametrize("names", [[], ["right_wrist"], ["head", "head"], ["head", "unknown_camera"]])
def test_poker_batch_rejects_invalid_camera_sets(batch, tmp_path, names):
  with pytest.raises(SystemExit):
    batch.parse_args(["--output-dir", str(tmp_path), "--cameras", *names])


@pytest.mark.parametrize("problem", [None, "camera_set", "wrist_shape", "wrist_clock"])
def test_poker_success_checks_all_requested_cameras(batch, tmp_path, monkeypatch, problem):
  import hashlib
  import json
  from types import SimpleNamespace

  import h5py
  import numpy as np
  from kaihand_tactile_env.shared import recording

  args = batch.parse_args(["--output-dir", str(tmp_path), "--cameras", "head", "right_wrist"])
  path = tmp_path / "episode.h5"
  monkeypatch.setattr(recording, "validate_episode", lambda _path: SimpleNamespace(valid=True, state_samples=2))
  monkeypatch.setattr(batch, "validate_recorded_acceptance", lambda *_args: None)
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps({"preset": batch.PRESET, "seed": 0, "episode_index": 0})
    file.attrs["outcome_json"] = '{"success": true}'
    file.attrs.update(physics_hz=500, control_hz=100, camera_hz=30,
                      render_backend_json='{"renderer": "test GPU", "software": false}')
    file.create_dataset("state/timestamp", data=[0, 0.1])
    for name in ("head", "right_wrist"):
      shape = (2, 120, 320, 3) if name == "right_wrist" and problem == "wrist_shape" else (2, 240, 320, 3)
      file.create_dataset(f"cameras/{name}/rgb", shape=shape, dtype="u1")
      time = np.array([0, 0.1])
      if name == "right_wrist" and problem == "wrist_clock":
        time += 0.01
      file.create_dataset(f"cameras/{name}/timestamp", data=time)
  path.with_suffix(".json").write_text(json.dumps({"episode": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
  if problem == "camera_set":
    args.cameras = ("head",)
  if problem is None:
    assert batch.validate_success(path, args, 0)["camera_frames"] == 2
  else:
    with pytest.raises(ValueError, match={"camera_set": "recorded cameras", "wrist_shape": "image dimensions", "wrist_clock": "synchronized"}[problem]):
      batch.validate_success(path, args, 0)


def test_success_target_requires_positive_count(batch, tmp_path):
  with pytest.raises(SystemExit):
    batch.parse_args(["--output-dir", str(tmp_path), "--target-successes", "0"])


def test_resume_preserves_failed_partial_indices_and_counts_verified_success(batch, tmp_path, monkeypatch):
  import json
  args = batch.parse_args(["--output-dir", str(tmp_path), "--target-successes", "150"])
  raw = tmp_path / "raw"
  raw.mkdir()
  (raw / "episode_000012_card_right.h5.partial").touch()
  for index, completed in ((0, True), (10, True), (11, False)):
    log = tmp_path / "logs" / batch.episode_name(index)
    log.mkdir(parents=True)
    (log / "execution.json").write_text(json.dumps({"episode_index": index, "completed": completed}))
  checks = []
  monkeypatch.setattr(batch, "validate_success", lambda path, config, index:
                      checks.append((index, config.render_backend)))
  assert batch.resume_inventory(args) == ([0, 10], 13)
  assert checks == [(0, "auto"), (10, "auto")]
  assert args.render_backend == "hardware"


def test_resume_rejects_corrupt_previously_successful_data(batch, tmp_path, monkeypatch):
  log = tmp_path / "logs" / "episode_000000_card_right"
  log.mkdir(parents=True)
  (log / "execution.json").write_text('{"episode_index": 0, "completed": true}')
  args = batch.parse_args(["--output-dir", str(tmp_path), "--target-successes", "150"])
  def reject(*args):
    raise ValueError("bad digest")
  monkeypatch.setattr(batch, "validate_success", reject)
  with pytest.raises(ValueError, match="bad digest"):
    batch.resume_inventory(args)


def test_success_target_reserves_inflight_slots_and_does_not_count_failures(batch, tmp_path):
  args = batch.parse_args(["--output-dir", str(tmp_path), "--target-successes", "7"])
  results = [{"completed": False}, {"completed": True}]
  assert batch.remaining_success_slots(args, 5, results) == 1
  assert batch.remaining_success_slots(args, 5, results, active_count=1) == 0
  assert batch.remaining_success_slots(args, 5, results + [{"completed": True}]) == 0


def test_target_loop_continues_after_failure_and_stops_exactly_at_total(batch, tmp_path, monkeypatch):
  import io
  from contextlib import nullcontext
  from types import SimpleNamespace
  args = batch.parse_args(["--output-dir", str(tmp_path), "--target-successes", "7", "--episodes", "20"])
  monkeypatch.setattr(batch, "wrapper_lock", nullcontext)
  monkeypatch.setattr(batch, "resume_inventory", lambda args: ([0, 1, 2, 3, 10], 11))
  monkeypatch.setattr(batch, "resource_snapshot", lambda path: {})
  monkeypatch.setattr(batch, "proc_table", lambda: {})
  monkeypatch.setattr(batch, "process_tree_metrics", lambda *args: {})
  monkeypatch.setattr(batch, "critical_reason", lambda *args: None)
  monkeypatch.setattr(batch, "launch_allowed", lambda *args: True)
  monkeypatch.setattr(batch.time, "sleep", lambda *args: None)
  launched = []
  def launch(args, index):
    launched.append(index)
    return SimpleNamespace(index=index, process=SimpleNamespace(pid=0, poll=lambda: 0),
                           peaks={}, resources=io.StringIO(), started=batch.time.monotonic(),
                           term_sent=None, kill_sent=False)
  monkeypatch.setattr(batch, "start_episode", launch)
  monkeypatch.setattr(batch, "finish_episode", lambda item, args:
                      {"episode_index": item.index, "completed": item.index != 11})
  report = batch.run_batch(args)
  assert launched == [11, 12, 13]
  assert report["completed"] is True
  assert report["successful_episodes"] == 2
  assert report["total_successful_episodes"] == 7
  assert len(report["not_started_indices"]) == 17


def test_hardware_child_clears_forced_software_but_keeps_cpu_bounded(batch, monkeypatch):
  monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
  monkeypatch.setenv("GALLIUM_DRIVER", "llvmpipe")
  env = batch.child_environment("hardware")
  assert "LIBGL_ALWAYS_SOFTWARE" not in env
  assert "GALLIUM_DRIVER" not in env
  assert env["KAIHAND_RENDER_BACKEND"] == "hardware"
  for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
    assert env[name] == "1"
  assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
  assert env["KAIHAND_POKER_HDF5_BUFFER_ROWS"] == "64"


def test_explicit_software_is_auditable_and_single_threaded(batch):
  env = batch.child_environment("software")
  assert env["GALLIUM_DRIVER"] == "llvmpipe"
  assert env["LP_NUM_THREADS"] == "1"
  assert env["KAIHAND_RENDER_BACKEND"] == "software"


def test_buffer_can_be_disabled_for_paired_comparison(batch):
  assert batch.child_environment("hardware", 0)["KAIHAND_POKER_HDF5_BUFFER_ROWS"] == "0"


def test_backend_rejection_closes_the_allocated_renderer(monkeypatch):
  from kaihand_tactile_env.shared import rendering
  from kaihand_tactile_env.shared.config import CameraConfig

  instances = []

  class FakeRenderer:
    def __init__(self, *args, **kwargs):
      self.closed = False
      instances.append(self)

    def close(self):
      self.closed = True

  def reject():
    raise RuntimeError("hardware unavailable")

  monkeypatch.setattr(rendering.mujoco, "Renderer", FakeRenderer)
  monkeypatch.setattr(rendering, "_require_camera", lambda *args: 0)
  monkeypatch.setattr(rendering, "current_backend", reject)
  with pytest.raises(RuntimeError, match="hardware unavailable"):
    rendering.WorkcellRenderer(object(), (CameraConfig("head"),))
  assert len(instances) == 1 and instances[0].closed


@pytest.mark.parametrize('extensions,index', [
  (['EGL_EXT_device_drm', 'EGL_MESA_device_software EGL_EXT_device_drm_render_node'], 1),
  (['EGL_MESA_device_software'], 0),
])
def test_software_device_is_found_by_extension_not_fixed_index(extensions, index):
  from kaihand_tactile_env.shared.render_backend import _software_device_index
  assert _software_device_index(extensions) == index


def test_missing_software_device_never_falls_back_to_gpu():
  from kaihand_tactile_env.shared.render_backend import _software_device_index
  with pytest.raises(RuntimeError, match='no software device'):
    _software_device_index(['EGL_EXT_device_drm', 'EGL_MESA_device_software_not_real'])


def test_prepare_overrides_gpu_index_and_reuses_software_display(monkeypatch):
  from types import SimpleNamespace

  import mujoco
  from kaihand_tactile_env.shared import render_backend as backend
  fake = SimpleNamespace(EGL_DISPLAY=None, EGL=object())
  monkeypatch.setattr(mujoco, 'egl', fake, raising=False)
  monkeypatch.setattr(backend, '_SOFTWARE_EGL_DEVICE_ID', None)
  monkeypatch.setattr(backend, '_egl_device_extensions', lambda _: ['EGL_EXT_device_drm', 'EGL_MESA_device_software'])
  monkeypatch.setenv('MUJOCO_GL', 'egl')
  monkeypatch.setenv('KAIHAND_RENDER_BACKEND', 'software')
  monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
  backend.prepare_render_backend()
  import os
  assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '1'
  fake.EGL_DISPLAY = object()
  backend.prepare_render_backend()
  monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
  with pytest.raises(RuntimeError, match='before the first GL context'):
    backend.prepare_render_backend()


def test_preinitialized_unmanaged_display_rejected(monkeypatch):
  from types import SimpleNamespace

  import mujoco
  from kaihand_tactile_env.shared import render_backend as backend
  monkeypatch.setattr(mujoco, 'egl', SimpleNamespace(EGL_DISPLAY=object()), raising=False)
  monkeypatch.setattr(backend, '_SOFTWARE_EGL_DEVICE_ID', None)
  monkeypatch.setenv('MUJOCO_GL', 'egl')
  monkeypatch.setenv('KAIHAND_RENDER_BACKEND', 'software')
  with pytest.raises(RuntimeError, match='fresh process'):
    backend.prepare_render_backend()


@pytest.mark.parametrize('mode', ['hardware', 'auto'])
def test_prepare_does_not_change_hardware_selection(monkeypatch, mode):
  import os

  from kaihand_tactile_env.shared import render_backend as backend
  monkeypatch.setenv('MUJOCO_GL', 'egl')
  monkeypatch.setenv('KAIHAND_RENDER_BACKEND', mode)
  monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
  backend.prepare_render_backend()
  assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '0'
