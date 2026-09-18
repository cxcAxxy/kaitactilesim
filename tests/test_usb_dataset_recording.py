"""USB-specific capture wiring and preservation checks; no task rollout/rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.recording import EpisodeRecorder
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import recording

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/workcell/record_usb_dataset.py"


def test_capture_defaults_are_usb_only_500_hz_and_three_camera_rgb():
  config = recording.recording_config()
  assert config.model_path.name == "scene.xml"
  assert config.model_path.parent.name == "usb_insert"
  assert config.physics_hz == config.control_hz == 500
  assert config.camera_hz == 30
  assert [camera.name for camera in config.cameras] == [
    "head", "left_wrist", "right_wrist",
  ]
  for camera in config.cameras:
    assert (camera.width, camera.height) == (320, 240)
    assert camera.rgb and not camera.depth and not camera.segmentation


def test_overhead_is_opt_in_and_shares_head_clock_and_rgb_dimensions(tmp_path):
  config = recording.recording_config(10, include_overhead=True)
  assert [camera.name for camera in config.cameras] == [
    "head", "left_wrist", "right_wrist", "overhead",
  ]
  assert config.physics_hz == config.control_hz == 500
  assert config.camera_hz == 10
  for camera in config.cameras:
    assert (camera.width, camera.height) == (320, 240)
    assert camera.rgb and not camera.depth and not camera.segmentation
  job = recording.UsbRecordJob(tmp_path / "usb.h5", 0, 7, include_overhead=True)
  assert job.include_overhead is True
  with pytest.raises(ValueError, match="boolean"):
    recording.recording_config(include_overhead="yes")


@pytest.mark.parametrize("wrist", ["right_wrist", "left_wrist"])
def test_shared_wrist_camera_selection_preserves_usb_clocks(tmp_path, wrist):
  parse = run_path(str(SCRIPT))["parse_args"]
  args = parse(["--output-dir", str(tmp_path / "new"),
                "--cameras", "head", wrist])
  job = recording.UsbRecordJob(tmp_path / "usb.h5", 0, 7, cameras=tuple(args.cameras))
  config = recording.recording_config(job.camera_hz, job.include_overhead, job.cameras)
  assert tuple(camera.name for camera in config.cameras) == ("head", wrist)
  assert config.physics_hz == config.control_hz == 500
  assert config.camera_hz == 30
  assert all(camera.rgb and not camera.depth and not camera.segmentation for camera in config.cameras)
  extra = recording.recording_config(cameras=job.cameras, include_overhead=True)
  assert tuple(camera.name for camera in extra.cameras) == ("head", wrist, "overhead")
  extra = recording.recording_config(cameras=("head", "overhead"), include_overhead=True)
  assert tuple(camera.name for camera in extra.cameras) == ("head", "overhead")


@pytest.mark.parametrize("names", [(), ("right_wrist",), ("head", "head"), ("head", "unknown_camera")])
def test_usb_rejects_invalid_camera_sets(names, tmp_path):
  with pytest.raises(ValueError, match="unique shared names"):
    recording.recording_config(cameras=names)
  parse = run_path(str(SCRIPT))["parse_args"]
  with pytest.raises(SystemExit):
    parse(["--output-dir", str(tmp_path / "new"), "--cameras", *names])


def test_separate_episode_seeds_reproduce_without_start_index_dependence():
  expected = recording.episode_seeds(123, 7)
  assert expected == recording.episode_seeds(123, 7)
  assert expected[0] != expected[1]
  assert expected != recording.episode_seeds(123, 8)
  assert expected != recording.episode_seeds(124, 7)
  assert [recording.episode_seeds(123, index) for index in range(7, 10)][0] == expected
  with pytest.raises(ValueError):
    recording.episode_seeds(123, -1)


def test_cli_help_and_validation_need_no_simulation(tmp_path):
  parse = run_path(str(SCRIPT))["parse_args"]
  args = parse(["--output-dir", str(tmp_path / "new")])
  assert args.motion_profile == "fast" and args.camera_hz == 30
  assert tuple(args.cameras) == ("head", "left_wrist", "right_wrist")
  assert args.include_overhead is False
  assert args.workers == 1
  batch = parse(
    ["--output-dir", str(tmp_path / "new"), "--workers", "2", "--camera-hz", "30"]
  )
  assert (batch.workers, batch.camera_hz) == (2, 30)
  assert recording.recording_config(30).camera_hz == 30
  assert (
    parse(
      ["--output-dir", str(tmp_path / "new"), "--include-overhead"]
    ).include_overhead
    is True
  )
  assert (args.xy_jitter_mm, args.yaw_jitter_deg, args.precontact_noise_mm) == (
    10,
    5,
    0.5,
  )
  for extra in (
    ["--workers", "3"],
    ["--workers", "0"],
    ["--seed", "-1"],
    ["--camera-hz", "31"],
    ["--precontact-noise-mm", "nan"],
  ):
    with pytest.raises(SystemExit) as error:
      parse(["--output-dir", str(tmp_path / "new"), *extra])
    assert error.value.code == 2


def test_camera_cli_keeps_numerical_imports_after_thread_setup(tmp_path):
  import subprocess
  import sys

  code = (
    "import runpy, sys; "
    f"cli = runpy.run_path({str(SCRIPT)!r}); "
    f"cli['parse_args'](['--output-dir', {str(tmp_path / 'new')!r}, "
    "'--cameras', 'head', 'right_wrist']); "
    "assert 'numpy' not in sys.modules; assert 'mujoco' not in sys.modules"
  )
  subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=15)


def test_lazy_package_exports_preserve_the_existing_api():
  import kaihand_tactile_env as package
  from kaihand_tactile_env.shared.config import CameraConfig, WorkcellConfig
  from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider

  assert package.CameraConfig is CameraConfig
  assert package.WorkcellConfig is WorkcellConfig
  assert package.ArmHandSimulation is ArmHandSimulation
  assert package.GenesisProbeTactileProvider is GenesisProbeTactileProvider
  assert set(package.__all__).issubset(dir(package))
  with pytest.raises(AttributeError):
    _ = package.nonexistent_camera_attribute


@pytest.mark.parametrize("rate", [True, 0, -1, 30.5, 31, 501])
def test_camera_rate_rejects_invalid_or_unsupported_grid(rate):
  with pytest.raises(ValueError):
    recording.recording_config(rate)


def test_usb_clock_adapter_preserves_original_step_call(monkeypatch):
  sim = object.__new__(recording.UsbRecordingSimulation)
  sim.data = SimpleNamespace(time=0.0)
  sim.timestep = 0.002
  calls = []

  def original_step(current, steps=1):
    assert current is sim
    calls.append(steps)
    current.data.time += steps * current.timestep

  monkeypatch.setattr(ArmHandSimulation, "step", original_step)
  assert sim.observation_time == 0.0
  sim.step(3)
  assert calls == [3]
  assert sim.data.time == pytest.approx(0.006)
  assert sim.observation_time == pytest.approx(0.004)
  sim.note_observation_refresh()
  assert sim.observation_time == pytest.approx(0.006)
  assert calls == [3]  # Clock bookkeeping performs no extra simulation step.
  assert sim.drive_limit_n is None
  sim.data.time = 0.0
  assert sim.observation_time == 0.0


def test_noise_trace_maps_command_and_contact_clocks_without_relabeling():
  noise = {
    "enabled": True,
    "first_contact": {"time_s": 0.004, "normal_force_n": 0.05},
    "commands": [
      {
        "time_s": time,
        "phase": "approach",
        "nominal_wrist_position_m": [1, 2, 3],
        "applied_wrist_position_m": [1.001, 2, 3],
        "random_offset_m": [0.001, 0, 0],
        "recovery_offset_m": [0, 0, 0],
        "arm_goal_rad": list(range(7)),
        "gaussian_knot_draws": 2,
      }
      for time in (0.0, 0.004, 0.006)
    ],
  }
  original = json.loads(json.dumps(noise))
  arrays, metadata = recording.noise_trace_arrays(
    noise, np.array([0, 0.002, 0.004, 0.006])
  )
  np.testing.assert_array_equal(arrays["state_index_at_or_before_issue"], [0, 2, 3])
  np.testing.assert_array_equal(arrays["first_future_state_index"], [1, 3, -1])
  assert arrays["arm_goal_rad"].shape == (3, 7)
  assert metadata["first_contact_state_index"] == 2
  assert metadata["first_contact"]["time_s"] == 0.004
  assert noise == original
  empty, _ = recording.noise_trace_arrays({"commands": []}, np.array([0]))
  assert empty["arm_goal_rad"].shape == (0, 7)


def test_terminal_stability_comes_only_from_actual_post_release_samples():
  evidence = recording.TerminalEvidence()
  twist = np.zeros(6)
  sim = SimpleNamespace(object_twist=lambda _: twist)
  for _ in range(100):
    evidence.observe(sim, "insert")
  assert evidence.report(0.002)["stable_seconds"] == 0
  for _ in range(50):
    evidence.observe(sim, "verify")
  assert evidence.report(0.002)["stable_seconds"] == pytest.approx(0.1)
  twist[0] = 0.03
  evidence.observe(sim, "verify")
  assert evidence.report(0.002)["stable_seconds"] == 0


def test_usb_recorder_selects_only_plug_and_writes_real_model_limits(
  tmp_path, monkeypatch
):
  model = SimpleNamespace(
    body=lambda name: SimpleNamespace(id={"usb_plug": 1}[name]),
    ngeom=4,
    geom_bodyid=np.array([1, 2, 1, 1]),
    geom_contype=np.array([1, 1, 1, 0]),
    geom_conaffinity=np.array([1, 1, 1, 0]),
    geom=lambda index: SimpleNamespace(
      name=("plug_handle", "tabletop", "plug_shell", "plug_visual_mark")[index]
    ),
    joint=lambda index: SimpleNamespace(name=f"joint{index}"),
    jnt_range=np.tile([-2.0, 2.0], (14, 1)),
    body_mass=np.array([0, 0.018, 1]),
    geom_friction=np.ones((3, 3)),
    nv=14,
    nbody=3,
    nu=14,
  )
  sim = SimpleNamespace(
    model=model,
    genesis_probe_layout=object(),
    _arm_joint_ids={"left": np.arange(7), "right": np.arange(7, 14)},
    _arm_qpos={"left": np.arange(7), "right": np.arange(7, 14)},
    _joint_id={f"joint{i}": i for i in range(14)},
    joint_names=tuple(f"joint{i}" for i in range(14)),
  )
  selected = []

  def provider(current_model, layout, *, target_geom_names):
    assert current_model is model and layout is sim.genesis_probe_layout
    selected.append(target_geom_names)
    return object()

  def shared_initialize(recorder, _metadata):
    assert recorder.contact_force_provider is not None
    recorder._file.create_group("model")
    recorder._file.create_group("tactile_contact_force")

  monkeypatch.setattr(recording, "SolverDistributedTactileProvider", provider)
  monkeypatch.setattr(EpisodeRecorder, "_initialize", shared_initialize)
  recorder = object.__new__(recording.UsbEpisodeRecorder)
  recorder.sim, recorder.h5py = sim, h5py
  recorder.config = SimpleNamespace(cameras=())
  with h5py.File(tmp_path / "schema.h5", "w") as file:
    recorder._file = file
    recorder._initialize({})
    assert selected == [("plug_handle", "plug_shell")]
    np.testing.assert_array_equal(file["model/arm_joint_qpos_indices"], np.arange(14))
    np.testing.assert_array_equal(file["model/arm_joint_limits_rad"], model.jnt_range)
    assert file["physics/xfrc_applied"].shape == (0, 3, 6)
    assert file["physics/qfrc_applied"].shape == (0, 14)
    assert "post-step" in file["physics"].attrs["solver_clock"]
    assert (
      file["usb_insertion"].attrs["contact_model_version"]
      == recording.usb_config.CONTACT_MODEL_VERSION
    )
    for name in (
      "timestamp",
      "state_index",
      *recording.INSERTION_METRIC_FIELDS,
      *recording.INSERTION_FLAG_FIELDS,
    ):
      assert file[f"usb_insertion/{name}"].shape == (0,)
    assert file["usb_insertion/state_index"].dtype == np.dtype(np.int64)
    assert file["usb_insertion/seated"].dtype == np.dtype(np.bool_)


def _insertion_snapshot(timestamp, **changes):
  values = {
    "timestamp": timestamp,
    **{name: 0.0 for name in recording.INSERTION_METRIC_FIELDS},
    "seated": False,
    "success": False,
    "shell_fits_aperture": False,
    "backstop_contact": False,
    "bottom_out_confirmed": False,
    **changes,
  }
  return SimpleNamespace(**values)


def test_insertion_metrics_copy_controller_snapshots_without_sampling_or_terminal_duplicate(
  tmp_path, monkeypatch
):
  sim = SimpleNamespace(
    data=SimpleNamespace(
      time=0.0,
      qfrc_applied=np.zeros(4),
      xfrc_applied=np.zeros((3, 6)),
      qacc=np.zeros(4),
      actuator_force=np.zeros(2),
    ),
    observation_time=0.0,
    model=SimpleNamespace(opt=SimpleNamespace(noslip_iterations=0)),
  )
  recorder = object.__new__(recording.UsbEpisodeRecorder)
  recorder.sim = sim
  recorder._insertion_state = None
  recorder._state_samples = 0
  recorder._closed = False
  recorder._last_state_time = None
  recorder._last_camera_time = None
  recorder.config = SimpleNamespace(control_hz=500, camera_hz=10, cameras=())

  def shared_record_state(current, phase):
    recording._append(current._file["state/timestamp"], sim.data.time)
    recording._append(current._file["commands/phase"], phase)
    current._state_samples += 1
    current._last_state_time = sim.data.time

  monkeypatch.setattr(EpisodeRecorder, "_record_state", shared_record_state)
  with h5py.File(tmp_path / "metrics.h5", "w") as file:
    recorder._file = recording.BufferedH5File(file)
    state = file.create_group("state")
    recording._stream(state, "timestamp", (), np.float64)
    commands = file.create_group("commands")
    recording._stream(commands, "phase", (), h5py.string_dtype())
    physics = file.create_group("physics")
    for name, shape in (
      ("solver_timestamp", ()),
      ("qfrc_applied", (4,)),
      ("xfrc_applied", (3, 6)),
      ("qacc", (4,)),
      ("actuator_force", (2,)),
      ("noslip_iterations", ()),
    ):
      recording._stream(physics, name, shape, np.float64)
    metrics = file.create_group("usb_insertion")
    for name in ("timestamp", *recording.INSERTION_METRIC_FIELDS):
      recording._stream(metrics, name, (), np.float64)
    recording._stream(metrics, "state_index", (), np.int64)
    for name in recording.INSERTION_FLAG_FIELDS:
      recording._stream(metrics, name, (), np.bool_)
    recorder.set_insertion_state(_insertion_snapshot(0.0))
    recorder._record_state("initial")
    sim.data.time = sim.observation_time = 0.002
    snapshot = _insertion_snapshot(
      0.002,
      insertion_depth_m=0.0115,
      backstop_axial_resistance_n=0.6,
      spring_normal_load_n=0.8,
      linear_speed_m_s=0.001,
      angular_speed_rad_s=0.002,
      maximum_socket_penetration_m=0.00002,
      orientation_error_rad=0.003,
      seated=True,
      shell_fits_aperture=True,
      backstop_contact=True,
      bottom_out_confirmed=True,
    )
    recorder.set_insertion_state(snapshot)
    assert recorder._insertion_state is snapshot
    recorder._record_state("bottom_out")
    # Terminal at an already-recorded camera/state time only updates the phase.
    recorder._last_camera_time = 0.002
    recorder.record_terminal("terminal_settle")
    report = recorder.insertion_stream_report()
    assert report["valid"] and report["terminal_monitor_row_recorded"]
    assert report["state_samples"] == 2
    recorder._file.flush()
    np.testing.assert_array_equal(file["usb_insertion/state_index"], [0, 1])
    np.testing.assert_array_equal(file["usb_insertion/timestamp"], [0.0, 0.002])
    np.testing.assert_allclose(
      file["usb_insertion/backstop_axial_resistance_n"], [0, 0.6]
    )
    for name in (
      "linear_speed_m_s",
      "angular_speed_rad_s",
      "maximum_socket_penetration_m",
      "orientation_error_rad",
    ):
      np.testing.assert_allclose(
        file[f"usb_insertion/{name}"], [0, getattr(snapshot, name)]
      )
    for name in ("shell_fits_aperture", "backstop_contact", "bottom_out_confirmed"):
      np.testing.assert_array_equal(file[f"usb_insertion/{name}"], [False, True])
    assert file["commands/phase"].asstr()[-1] == "terminal_settle"
    sim.data.time = 0.004
    with pytest.raises(ValueError, match="timestamp differs"):
      recorder._record_state("stale")
    assert file["state/timestamp"].shape[0] == 2
    assert recorder.insertion_stream_report()["missing_or_partial_tail"]


def test_insertion_snapshot_rejects_nonfinite_metrics_and_missing_clock():
  recorder = object.__new__(recording.UsbEpisodeRecorder)
  recorder.sim = SimpleNamespace(data=SimpleNamespace(time=0.0))
  with pytest.raises(ValueError, match="nonfinite"):
    recorder.set_insertion_state(
      _insertion_snapshot(0.0, spring_axial_resistance_n=np.nan)
    )
  with pytest.raises(ValueError, match="timestamp differs"):
    recorder.set_insertion_state(_insertion_snapshot(0.002))


@pytest.mark.parametrize("suffix", (".h5", ".h5.partial", ".json", ".result.json"))
def test_existing_episode_evidence_is_never_deleted_or_overwritten(tmp_path, suffix):
  output = tmp_path / "usb_000000.h5"
  existing = output.with_suffix(suffix)
  existing.write_bytes(b"previous evidence")
  with pytest.raises(FileExistsError):
    recording.ensure_new_episode(output)
  assert existing.read_bytes() == b"previous evidence"


@pytest.mark.parametrize("status", ("failed", "cancelled", "exception", "unconfirmed"))
def test_unsuccessful_executor_keeps_raw_and_noise_without_finalizing(
  tmp_path, monkeypatch, status
):
  noise = {"enabled": True, "seed": 12, "commands": []}
  instances = []
  model_path = tmp_path / "scene.xml"
  model_path.write_text("<mujoco/>")

  class Simulation:
    timestep = 0.002

    def __init__(self, **_kwargs):
      self.model_path = model_path
      self.data = SimpleNamespace(time=0.0)

    def reset(self, *, seed):
      self.seed = seed

    def note_observation_refresh(self):
      self.observation_time = self.data.time

    def object_pose(self, _name):
      return np.array([0.5, -0.18, 0.6865, 1, 0, 0, 0])

    def object_twist(self, _name):
      return np.zeros(6)

  class Recorder:
    def __init__(self, output, simulation, _config, **_kwargs):
      assert _kwargs["metadata"]["observation_clock"] == "post_step_forward_v1"
      assert (
        _kwargs["metadata"]["contact_model_version"]
        == recording.usb_config.CONTACT_MODEL_VERSION
      )
      self.partial_path = output.with_suffix(".h5.partial")
      self._file = {"state/timestamp": np.array([0.0])}
      self.sim = simulation
      self.trace = None
      instances.append(self)

    def record_initial(self):
      assert self.snapshot.timestamp == 0.0

    def set_insertion_state(self, snapshot):
      assert snapshot.timestamp == self.sim.data.time
      self.snapshot = snapshot

    def insertion_stream_report(self):
      return {"valid": True, "terminal_monitor_row_recorded": True}

    def observe(self, _sim, _phase):
      assert _sim.observation_time == _sim.data.time == 0.002
      assert self.snapshot.timestamp == 0.002

    def record_terminal(self, phase):
      self.terminal_phase = phase

    def write_precontact_noise_trace(self, trace, *, metadata):
      self.trace = trace, metadata

    def set_outcome(self, outcome):
      self.outcome = outcome

    def close(self, *, finalize):
      assert not finalize
      self.partial_path.write_bytes(b"actual captured raw fixture")

  @dataclass
  class Result:
    success: bool = status == "unconfirmed"
    released: bool = status == "unconfirmed"
    grasp_verified: bool = status == "unconfirmed"
    active_bottom_out_confirmed: bool = False
    failure_reason: str | None = (
      None
      if status == "unconfirmed"
      else "cancelled"
      if status == "cancelled"
      else "grip lost"
    )
    precontact_noise: dict | None = None
    insertion: dict | None = None

  class Executor:
    def __init__(self, _sim, **options):
      assert options["motion_profile"] == "fast"
      assert options["precontact_noise_std_m"] == 0.0005
      self._noise = SimpleNamespace(report=lambda: noise)
      self.sim, self.observe = _sim, options["observer"]
      self.monitor = SimpleNamespace(
        measure=lambda: _insertion_snapshot(self.sim.data.time)
      )

    def execute(self):
      self.sim.data.time = 0.002
      self._state = _insertion_snapshot(0.002)
      self.observe(self.sim, "approach")
      if status == "exception":
        raise RuntimeError("unexpected control failure")
      return Result(precontact_noise=noise, insertion={"success": True, "seated": True})

  monkeypatch.setattr(recording, "UsbRecordingSimulation", Simulation)
  monkeypatch.setattr(recording, "UsbEpisodeRecorder", Recorder)
  monkeypatch.setattr(recording, "UsbInsertionExecutor", Executor)
  monkeypatch.setattr(
    recording, "initialize_for_insertion", lambda *_args, **kwargs: kwargs
  )
  monkeypatch.setattr(
    recording, "controller_source_hashes", lambda: {"source": "fixed"}
  )
  if status == "unconfirmed":
    monkeypatch.setattr(
      recording,
      "TerminalEvidence",
      lambda: SimpleNamespace(
        stable_steps=100,
        observe=lambda *_args: None,
        report=lambda *_args: {"stable_seconds": 0.2},
      ),
    )
  job = recording.UsbRecordJob(tmp_path / "usb_000000.h5", 0, 7)
  report = recording.record_episode(job)
  expected_status = "failed" if status == "unconfirmed" else status
  assert report["status"] == expected_status and not report["success"]
  if status == "unconfirmed":
    assert report["outcome"]["executor_success"] is True
    assert report["outcome"]["active_bottom_out_confirmed"] is False
  assert not report["outcome"]["success"]
  assert not job.output.exists() and not job.output.with_suffix(".json").exists()
  assert (
    job.output.with_suffix(".h5.partial").read_bytes() == b"actual captured raw fixture"
  )
  assert instances[0].trace[1]["seed"] == 12
  assert (
    json.loads(job.output.with_suffix(".result.json").read_text())["status"]
    == expected_status
  )
