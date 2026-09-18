from __future__ import annotations

import json
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import ModuleType, SimpleNamespace

import mujoco.viewer
import numpy as np
import pytest
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config, setup

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/workcell/view_usb_insert.py"
EXECUTION_MODULE = "kaihand_tactile_env.tasks.usb_insert.execution"


@dataclass(frozen=True)
class TaskResult:
  success: bool
  phases: tuple[str, ...]
  failure_reason: str | None
  observer_calls: int
  precontact_noise: dict | None = None


@pytest.fixture(scope="module")
def simulation():
  return ArmHandSimulation(scene="usb-insert")


@pytest.fixture
def command(monkeypatch, simulation):
  script = run_path(str(SCRIPT))
  globals_ = script["main"].__globals__
  monkeypatch.setitem(globals_, "ArmHandSimulation", lambda **_kwargs: simulation)

  def run(*arguments):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *map(str, arguments)])
    return script["main"]()

  return SimpleNamespace(run=run, globals=globals_)


def _install_executor(
  monkeypatch,
  simulation,
  *,
  success=True,
  interrupt=False,
  output_race=None,
  repeats=1,
  noise_record=None,
):
  calls = SimpleNamespace(
    count=0, steps=0, active=False, finished=False, executor_options=[]
  )
  actual_step = simulation.step

  def step(*args, **kwargs):
    assert calls.active, "CLI advanced physics outside the automatic executor"
    calls.steps += 1
    return actual_step(*args, **kwargs)

  monkeypatch.setattr(simulation, "step", step)

  class Executor:
    def __init__(self, observed_simulation, observer=None, should_stop=None, **kwargs):
      assert observed_simulation is simulation
      assert observer is not None
      assert should_stop is not None
      calls.executor_options.append(kwargs)
      self.observer = observer
      self.should_stop = should_stop

    def execute(self):
      calls.count += 1
      calls.active = True
      phases = []
      try:
        for phase in ("grasp", "lift", "align", "insert") * repeats:
          if self.should_stop():
            break
          simulation.step()
          phases.append(phase)
          if interrupt and phase == "lift":
            signal.raise_signal(signal.SIGINT)
          self.observer(simulation, phase)
        cancelled = self.should_stop()
        if output_race is not None:
          output_race.write_text("created while the task was running")
        return TaskResult(
          success=success and not cancelled,
          phases=tuple(phases),
          failure_reason="cancelled" if cancelled else (None if success else "no fit"),
          observer_calls=len(phases),
          precontact_noise=noise_record,
        )
      finally:
        calls.active = False
        calls.finished = True

  module = ModuleType(EXECUTION_MODULE)
  module.UsbInsertionExecutor = Executor
  monkeypatch.setitem(sys.modules, EXECUTION_MODULE, module)
  return calls


def _install_viewer(monkeypatch, calls, *, cancel=None, result_json=None):
  class Viewer:
    def __init__(self):
      self.opt = mujoco.MjvOption()
      self.cam = SimpleNamespace(lookat=np.zeros(3))
      self.running = True
      self.closed = False
      self.sync_count = 0
      self.finished_syncs = 0
      self.callback = None

    def __enter__(self):
      return self

    def __exit__(self, exc_type, *_args):
      assert exc_type is None
      self.closed = True
      self.running = False

    def is_running(self):
      return self.running

    def sync(self):
      self.sync_count += 1
      if calls.finished:
        self.finished_syncs += 1
        # Results must be available while the final state is still displayed.
        assert result_json is not None and result_json.exists()
        self.running = False
      elif cancel == "window":
        self.running = False
      elif cancel in ("q", "escape"):
        self.callback(ord("Q") if cancel == "q" else 256)
      elif cancel == "sigint":
        signal.raise_signal(signal.SIGINT)

  viewer = Viewer()

  def launch(_model, _data, *, key_callback):
    viewer.callback = key_callback
    return viewer

  monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
  return viewer


@pytest.mark.parametrize(
  "noise_flags",
  ((), ("--precontact-noise-mm", "3", "--noise-seed", "27")),
)
def test_idle_does_not_apply_noise_or_import_executor_and_writes_without_rendering(
  command,
  monkeypatch,
  tmp_path,
  noise_flags,
):
  monkeypatch.setitem(sys.modules, EXECUTION_MODULE, None)

  def render_unexpected(*_args, **_kwargs):
    pytest.fail("--result-json unexpectedly tried to render a snapshot")

  monkeypatch.setitem(command.globals, "snapshot", render_unexpected)
  output = tmp_path / "nested" / "idle.json"
  command.run(
    "--headless", "--duration", "0.002", "--result-json", output, *noise_flags
  )
  report = json.loads(output.read_text())
  assert "task_result" not in report
  assert report["plug_initialization"] == "scene-default"
  assert report["finite_state"]
  assert report["insertion"]["timestamp"] == pytest.approx(0.002)


@pytest.mark.parametrize(
  "flags,expected",
  [
    ((), {"precontact_noise_std_m": 0.0005, "noise_seed": None}),
    (
      ("--precontact-noise-mm", "1.25", "--noise-seed", "43"),
      {"precontact_noise_std_m": 0.00125, "noise_seed": 43},
    ),
    (("--precontact-noise-mm", "0"), {}),
    (
      ("--precontact-noise-mm", "0", "--motion-profile", "baseline"),
      {"motion_profile": "baseline"},
    ),
    (
      ("--motion-profile", "baseline"),
      {
        "motion_profile": "baseline",
        "precontact_noise_std_m": 0.0005,
        "noise_seed": None,
      },
    ),
    (
      ("--precontact-noise-mm", "0", "--noise-seed", "0"),
      {"precontact_noise_std_m": 0.0, "noise_seed": 0},
    ),
  ],
)
def test_auto_converts_noise_units_keeps_seed_independent_and_allows_zero(
  command, monkeypatch, simulation, tmp_path, flags, expected
):
  calls = _install_executor(monkeypatch, simulation)
  command.run(
    "--headless",
    "--run-task",
    "--seed",
    "91",
    "--result-json",
    tmp_path / "noise.json",
    *flags,
  )
  assert calls.executor_options == [expected]


def test_auto_report_preserves_actual_executor_noise_seed_and_samples(
  command, monkeypatch, simulation, tmp_path
):
  # A generated seed and realized sample come from the executor result. The
  # CLI must not replace them with object seed 91 or merely the requested sigma.
  record = {"seed": 987654, "std_m": 0.0005, "offset_xy_m": [0.0002, -0.0001]}
  calls = _install_executor(monkeypatch, simulation, noise_record=record)
  output = tmp_path / "actual-noise.json"
  command.run("--headless", "--run-task", "--seed", "91", "--result-json", output)
  report = json.loads(output.read_text())
  assert calls.executor_options[0]["noise_seed"] is None
  assert report["seed"] == 91
  assert report["task_result"]["precontact_noise"] == record


@pytest.mark.parametrize("command_count", (0, 3))
def test_publish_keeps_full_noise_trace_in_files_and_only_summarizes_console(
  monkeypatch, tmp_path, capsys, command_count
):
  # Exercise publication directly: no simulation fixture, model or physics.
  script = run_path(str(SCRIPT))
  publish = script["_publish_report"]
  simulation = object()
  args = SimpleNamespace(
    result_json=tmp_path / "result.json", snapshot_dir=tmp_path / "snapshot"
  )
  report = {
    "finite_state": True,
    "task_result": {
      "success": True,
      "precontact_noise": {
        "seed": 37,
        "std_m": 0.0005,
        "commands": [
          {
            "timestamp_s": i * 0.02,
            "phase": "approach",
            "offset_xy_m": [0.0002, -0.0001],
            "target_position_m": [0.4, -0.2, 0.8],
          }
          for i in range(command_count)
        ],
      },
    },
  }
  before = json.loads(json.dumps(report))
  snapshot_inputs = []

  def save_snapshot(observed_simulation, directory, full_report):
    assert observed_simulation is simulation
    snapshot_inputs.append(full_report)
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps(full_report), encoding="utf-8")

  monkeypatch.setitem(publish.__globals__, "snapshot", save_snapshot)
  publish(simulation, args, report)
  captured = capsys.readouterr()
  console = json.loads(captured.out)
  assert not captured.err
  assert json.loads(args.result_json.read_text()) == before
  assert json.loads((args.snapshot_dir / "state.json").read_text()) == before
  assert report == before
  assert snapshot_inputs == [before]
  if command_count:
    noise = console["task_result"]["precontact_noise"]
    assert "commands" not in noise
    assert noise["seed"] == 37 and noise["std_m"] == 0.0005
    assert noise["command_count"] == command_count
    assert noise["control_trace_in_console"] is False
    assert len(captured.out) < len(args.result_json.read_text())
  else:
    assert console == before


@pytest.mark.parametrize(
  "flags",
  [
    ("--precontact-noise-mm", "-1"),
    ("--precontact-noise-mm", "nan"),
    ("--precontact-noise-mm", "inf"),
    ("--precontact-noise-mm=-inf",),
    ("--noise-seed", "-1"),
    ("--noise-seed", "1.5"),
    ("--noise-seed", "nan"),
  ],
)
def test_cli_rejects_invalid_noise_before_model_creation(command, monkeypatch, flags):
  def create_unexpected(**_kwargs):
    pytest.fail("invalid noise was not rejected before model creation")

  monkeypatch.setitem(command.globals, "ArmHandSimulation", create_unexpected)
  with pytest.raises(SystemExit) as error:
    command.run("--headless", "--run-task", *flags)
  assert error.value.code == 2


@pytest.mark.parametrize(
  "flag,initializer,mode",
  (
    ("--plug-for-insertion", "initialize_for_insertion", "pinky-forward-grasp"),
    ("--plug-face-down", "initialize_face_down", "mark-down"),
  ),
)
def test_cli_initializes_only_orientation_before_the_first_automatic_step(
  command, monkeypatch, simulation, tmp_path, flag, initializer, mode
):
  calls = _install_executor(monkeypatch, simulation)
  actual_initialize = getattr(setup, initializer)
  snapshots = []

  def initialize(observed_simulation, **kwargs):
    assert observed_simulation is simulation
    assert simulation.data.time == 0
    assert calls.count == calls.steps == 0
    before_qpos = simulation.data.qpos.copy()
    before_qvel = simulation.data.qvel.copy()
    goals = simulation.arm_goal
    record = actual_initialize(simulation, **kwargs)
    snapshots.append((before_qpos, simulation.data.qpos.copy()))
    np.testing.assert_array_equal(simulation.data.qvel, before_qvel)
    for side, goal in goals.items():
      np.testing.assert_array_equal(simulation.arm_goal[side], goal)
    assert simulation.data.time == 0
    return record

  monkeypatch.setattr(setup, initializer, initialize)
  output = tmp_path / "initialized.json"
  command.run("--headless", "--run-task", flag, "--result-json", output)

  assert len(snapshots) == 1
  before, after = snapshots[0]
  address = int(simulation.model.joint("usb_plug_freejoint").qposadr[0])
  expected = before.copy()
  expected[address + 3 : address + 7] = (
    config.AUTO_PLUG_QUATERNION_WXYZ
    if initializer == "initialize_for_insertion"
    else [0.0, 1.0, 0.0, 0.0]
  )
  np.testing.assert_array_equal(after, expected)
  assert calls.count == 1 and calls.steps == 4
  assert json.loads(output.read_text())["plug_initialization"] == mode


def test_cli_records_requested_ranges_and_actual_time_zero_pose(
  command, monkeypatch, simulation, tmp_path
):
  calls = _install_executor(monkeypatch, simulation)
  actual_initialize = setup.initialize_for_insertion
  initialized = []

  def initialize(observed_simulation, **kwargs):
    assert observed_simulation is simulation
    assert simulation.data.time == 0.0 and calls.steps == 0
    record = actual_initialize(simulation, **kwargs)
    initialized.append(record)
    return record

  monkeypatch.setattr(setup, "initialize_for_insertion", initialize)
  reports = []
  for attempt in range(2):
    output = tmp_path / f"randomized-{attempt}.json"
    command.run(
      "--headless",
      "--run-task",
      "--plug-for-insertion",
      "--seed",
      "27",
      "--xy-jitter-mm",
      "5",
      "--yaw-jitter-deg",
      "3",
      "--result-json",
      output,
    )
    reports.append(json.loads(output.read_text()))
    # The installed executor is reused; its counters do not represent reset time.
    calls.count = calls.steps = 0
  record = reports[0]["initial_pose_randomization"]
  assert record == initialized[0] == reports[1]["initial_pose_randomization"]
  assert record["seed"] == 27
  assert record["xy_jitter_m"] == pytest.approx(0.005)
  assert record["yaw_jitter_rad"] == pytest.approx(np.deg2rad(3))
  assert np.all(np.abs(record["offset_xy_m"]) <= 0.005)
  assert abs(record["yaw_offset_rad"]) <= np.deg2rad(3)
  np.testing.assert_allclose(
    record["initial_pose_wxyz"][:2],
    config.PLUG_INITIAL_POSITION_M[:2] + record["offset_xy_m"],
  )
  assert record["initial_pose_wxyz"][2] == config.PLUG_INITIAL_POSITION_M[2]
  assert reports[0]["plug_pose_wxyz"][2] != record["initial_pose_wxyz"][2]


@pytest.mark.parametrize(
  "arguments",
  [
    ("--xy-jitter-mm", "1"),
    ("--yaw-jitter-deg", "0"),
    ("--plug-face-down", "--yaw-jitter-deg", "1"),
    ("--plug-for-insertion", "--xy-jitter-mm", "-1"),
    ("--plug-for-insertion", "--xy-jitter-mm", "nan"),
    ("--plug-for-insertion", "--yaw-jitter-deg", "inf"),
    ("--plug-for-insertion", "--yaw-jitter-deg", "-2"),
    ("--plug-for-insertion", "--seed", "-1"),
  ],
)
def test_cli_rejects_invalid_randomization_before_model_creation(
  command, monkeypatch, arguments
):
  def create_unexpected(**_kwargs):
    pytest.fail("invalid randomization was not rejected before model creation")

  monkeypatch.setitem(command.globals, "ArmHandSimulation", create_unexpected)
  with pytest.raises(SystemExit) as error:
    command.run("--headless", *arguments)
  assert error.value.code == 2


def test_reset_helper_reapplies_identical_randomization(
  command, monkeypatch, simulation
):
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT),
      "--plug-for-insertion",
      "--seed",
      "19",
      "--xy-jitter-mm",
      "8",
      "--yaw-jitter-deg",
      "4",
    ],
  )
  args = command.globals["parse_args"]()
  reset = command.globals["_reset_episode"]
  first = reset(simulation, args)
  simulation.step()
  second = reset(simulation, args)
  assert first == second
  assert simulation.data.time == 0.0
  np.testing.assert_array_equal(
    simulation.object_pose("usb_plug"), first["initial_pose_wxyz"]
  )


@pytest.mark.parametrize(
  "flags",
  (
    ("--plug-for-insertion", "--plug-face-down"),
    ("--plug-face-down", "--plug-for-insertion"),
  ),
)
def test_cli_rejects_conflicting_initialization_before_model_creation(
  command, monkeypatch, flags
):
  def create_unexpected(**_kwargs):
    pytest.fail("conflicting initializations were not rejected before model creation")

  monkeypatch.setitem(command.globals, "ArmHandSimulation", create_unexpected)
  with pytest.raises(SystemExit) as error:
    command.run("--headless", *flags)
  assert error.value.code == 2


@pytest.mark.parametrize("success", (True, False))
def test_headless_auto_runs_once_ignores_idle_duration_and_saves_result(
  command,
  monkeypatch,
  simulation,
  tmp_path,
  success,
):
  calls = _install_executor(monkeypatch, simulation, success=success)
  output = tmp_path / "automatic.json"
  arguments = (
    "--headless",
    "--run-task",
    "--duration",
    "0.0001",
    "--result-json",
    output,
  )
  if success:
    command.run(*arguments)
  else:
    with pytest.raises(SystemExit) as error:
      command.run(*arguments)
    assert error.value.code == 1
  report = json.loads(output.read_text())
  assert calls.count == 1
  assert calls.steps == 4
  assert report["insertion"]["timestamp"] == pytest.approx(0.008)
  assert report["task_result"] == {
    "success": success,
    "phases": ["grasp", "lift", "align", "insert"],
    "failure_reason": None if success else "no fit",
    "observer_calls": 4,
    "precontact_noise": None,
  }


@pytest.mark.parametrize("success", (True, False))
def test_gui_auto_plays_once_then_displays_final_state_without_extra_steps(
  command,
  monkeypatch,
  simulation,
  tmp_path,
  success,
):
  calls = _install_executor(monkeypatch, simulation, success=success)
  output = tmp_path / "viewer.json"
  viewer = _install_viewer(monkeypatch, calls, result_json=output)
  command.run("--run-task", "--duration", "0.0001", "--result-json", output)
  assert calls.count == 1
  assert calls.steps == 4
  assert calls.executor_options == [
    {"precontact_noise_std_m": 0.0005, "noise_seed": None}
  ]
  assert viewer.sync_count >= 2
  assert viewer.finished_syncs == 1
  assert viewer.closed
  assert json.loads(output.read_text())["task_result"]["success"] is success


def test_gui_observer_paces_simulation_time_and_limits_live_sync_to_60_hz(
  command,
  monkeypatch,
  simulation,
  tmp_path,
):
  class Clock:
    now = 0.0

    def monotonic(self):
      return self.now

    def sleep(self, duration):
      assert 0 < duration <= 1.0 / 60.0
      self.now += duration

  clock = Clock()
  monkeypatch.setitem(command.globals, "time", clock)
  calls = _install_executor(monkeypatch, simulation, repeats=10)
  output = tmp_path / "paced.json"
  viewer = _install_viewer(monkeypatch, calls, result_json=output)
  actual_sync = viewer.sync
  live_sync_times = []
  completed_at = []

  def sync():
    if calls.active:
      live_sync_times.append(clock.now)
    else:
      completed_at.append(clock.now)
    actual_sync()

  viewer.sync = sync
  command.run("--run-task", "--result-json", output)
  assert calls.steps == 40
  assert completed_at[0] == pytest.approx(40 * simulation.timestep)
  assert len(live_sync_times) >= 3
  assert np.all(np.diff(live_sync_times) >= 1.0 / 60.0 - 1e-12)


@pytest.mark.parametrize("cancel", ("sigint", "q", "escape", "window"))
def test_gui_auto_stops_for_signal_keys_or_window_close_and_saves_cancelled_result(
  command,
  monkeypatch,
  simulation,
  tmp_path,
  cancel,
):
  calls = _install_executor(monkeypatch, simulation)
  output = tmp_path / "cancelled.json"
  viewer = _install_viewer(monkeypatch, calls, cancel=cancel, result_json=output)
  original_handler = signal.getsignal(signal.SIGINT)
  command.run("--run-task", "--result-json", output)
  assert signal.getsignal(signal.SIGINT) is original_handler
  assert calls.count == 1
  assert calls.steps == 1
  assert viewer.closed
  result = json.loads(output.read_text())["task_result"]
  assert not result["success"]
  assert result["failure_reason"] == "cancelled"
  assert result["phases"] == ["grasp"]


def test_headless_auto_ctrl_c_keeps_clean_exit_and_records_cancellation(
  command,
  monkeypatch,
  simulation,
  tmp_path,
  capsys,
):
  calls = _install_executor(monkeypatch, simulation, interrupt=True)
  output = tmp_path / "cancelled.json"
  original_handler = signal.getsignal(signal.SIGINT)
  command.run("--headless", "--run-task", "--result-json", output)
  assert calls.steps == 2
  assert signal.getsignal(signal.SIGINT) is original_handler
  assert json.loads(output.read_text())["task_result"]["failure_reason"] == "cancelled"
  captured = capsys.readouterr()
  assert "USB scene stopped." in captured.out
  assert not captured.err


def test_result_json_rejects_existing_file_before_model_creation(
  command,
  monkeypatch,
  tmp_path,
):
  output = tmp_path / "existing.json"
  output.write_text("previous experiment")

  def create_unexpected(**_kwargs):
    pytest.fail("existing output was not rejected before model creation")

  monkeypatch.setitem(command.globals, "ArmHandSimulation", create_unexpected)
  with pytest.raises(SystemExit) as error:
    command.run("--headless", "--run-task", "--result-json", output)
  assert error.value.code == 2
  assert output.read_text() == "previous experiment"


def test_result_json_does_not_overwrite_file_created_during_execution(
  command,
  monkeypatch,
  simulation,
  tmp_path,
):
  output = tmp_path / "race.json"
  _install_executor(monkeypatch, simulation, output_race=output)
  with pytest.raises(FileExistsError):
    command.run("--headless", "--run-task", "--result-json", output)
  assert output.read_text() == "created while the task was running"
