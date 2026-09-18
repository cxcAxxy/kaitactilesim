"""Friction protocol/CLI checks; no rendering or task execution."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.config import DEFAULT_PRESS_FORCE_PER_FINGER_N


def _parse(options):
  script = run_path(
    str(Path(__file__).parents[1] / "scripts/workcell/experiment_poker_friction.py")
  )
  return script["_parse_args"](options)


def test_default_friction_scan_uses_production_force_baseline():
  args = _parse([])
  assert args.protocol == "force"
  assert args.friction == (0.1, 0.45, 0.7)
  assert args.press_forces_n == (DEFAULT_PRESS_FORCE_PER_FINGER_N,)
  assert args.press_offset_degrees is None
  assert args.diagnostics_dir is None
  assert args.record is False
  assert args.record_fps == 10.0


def test_force_scan_accepts_pairwise_newton_targets():
  args = _parse(["--friction", "0.1", "0.45", "--press-force", "0.2", "0.35"])
  assert args.press_forces_n == [0.2, 0.35]


def test_legacy_scan_requires_explicit_protocol():
  args = _parse(["--protocol", "legacy-preload"])
  assert args.press_offset_degrees == (0.0, 2.0)
  assert args.friction == (0.1, 0.9, 1.15, 1.3)
  assert args.slide_distance == 0.135


@pytest.mark.parametrize(
  "options",
  [
    ["--press-force", "nan"],
    ["--press-force", "0"],
    ["--friction", "-1"],
    ["--friction", "inf"],
    ["--press-offset-degrees", "2"],
    ["--slide-speed", "0.05"],
    ["--protocol", "legacy-preload", "--press-force", "0.35"],
    ["--protocol", "legacy-preload", "--diagnostics-dir", "unused"],
    ["--record"],
    ["--record-fps", "nan"],
    ["--record-fps", "0"],
    ["--record-fps", "30.1"],
    ["--protocol", "legacy-preload", "--record", "--diagnostics-dir", "unused"],
    ["--press-force", *["0.35"] * 4],  # 3 friction values x 4 targets > 10.
  ],
)
def test_invalid_or_ambiguous_scan_arguments_fail_early(options):
  with pytest.raises(SystemExit) as error:
    _parse(options)
  assert error.value.code == 2


def _script():
  return run_path(
    str(Path(__file__).parents[1] / "scripts/workcell/experiment_poker_friction.py")
  )


def test_diagnostics_reserves_new_directory_and_default_summary(tmp_path):
  script = _script()
  directory = tmp_path / "baseline"
  args = script["_parse_args"](
    ["--friction", "0.1", "--press-force", "0.35", "--diagnostics-dir", str(directory)]
  )
  script["_prepare_outputs"](args)
  assert args.json == directory / "summary.json"
  assert directory.is_dir()
  assert list(directory.iterdir()) == []
  with pytest.raises(FileExistsError):
    script["_prepare_outputs"](args)


@pytest.mark.parametrize(
  "filename", ["trial_001.json", "trial_001_timeseries.csv", "trial_001_curves.png"]
)
def test_diagnostics_rejects_summary_filename_collision_before_creation(
  tmp_path, filename
):
  script = _script()
  directory = tmp_path / "baseline"
  args = script["_parse_args"](
    [
      "--friction",
      "0.1",
      "--diagnostics-dir",
      str(directory),
      "--json",
      str(directory / filename),
    ]
  )
  with pytest.raises(ValueError, match="different files"):
    script["_prepare_outputs"](args)
  assert not directory.exists()


def test_diagnostics_refuses_existing_summary_without_mutation(tmp_path):
  script = _script()
  summary = tmp_path / "old.json"
  summary.write_text("keep this result", encoding="utf-8")
  directory = tmp_path / "baseline"
  args = script["_parse_args"](
    ["--friction", "0.1", "--diagnostics-dir", str(directory), "--json", str(summary)]
  )
  with pytest.raises(FileExistsError):
    script["_prepare_outputs"](args)
  assert summary.read_text() == "keep this result"
  assert not directory.exists()


@pytest.mark.parametrize("filename", ["trial_001_video.mp4", "trial_001_video.json"])
def test_recording_rejects_video_summary_collision_before_creation(tmp_path, filename):
  script = _script()
  directory = tmp_path / "baseline"
  args = script["_parse_args"](
    [
      "--friction",
      "0.1",
      "--diagnostics-dir",
      str(directory),
      "--record",
      "--json",
      str(directory / filename),
    ]
  )

  with pytest.raises(ValueError, match="different files"):
    script["_prepare_outputs"](args)

  assert not directory.exists()


def test_stage_report_does_not_call_initial_press_failure_a_friction_failure():
  result = _script()["_stage_outcome"](
    {
      "success": False,
      "error": "force did not stabilize",
      "established_press_normal_forces_n": (0.0,) * 4,
      "maximum_overhang_fraction": 0.0,
      "slide_press_control_qualified": False,
    },
    {"four_finger_press": 10},
    "four_finger_press",
  )
  assert not result["pressure_established"]
  assert not result["slide_started"]
  assert result["failure_stage"] == "press"
  assert result["failure_reason"] == "force did not stabilize"


def test_stage_report_separates_edge_reached_from_force_quality():
  result = _script()["_stage_outcome"](
    {
      "success": False,
      "error": "force quality",
      "established_press_normal_forces_n": (0.35,) * 4,
      "maximum_overhang_fraction": 0.50,
      "slide_press_control_qualified": False,
    },
    {"slide_card": 100, "edge_hold": 10},
    "edge_hold",
  )
  assert result["pressure_established"]
  assert result["half_overhang_reached"]
  assert not result["force_tracking_qualified"]
  assert result["failure_stage"] == "edge_hold"


def test_observer_streams_press_slide_edge_and_unobserved_terminal(tmp_path):
  import csv

  class FakeTelemetry:
    def sample(self, phase):
      return {"time_s": simulation.data.time, "phase": phase, "card_x_m": 0.58}

  simulation = SimpleNamespace(
    data=SimpleNamespace(time=0.0, geom_xpos=np.tile([0.6, 0, 0], (4, 1))),
  )
  observer_class = _script()["_ForceTrialObserver"]
  target = tmp_path / "trace.csv"
  with target.open("x", newline="") as file:
    observer = observer_class(FakeTelemetry(), (0, 1, 2, 3), file)
    for phase in (
      "four_finger_press",
      "slide_card",
      "edge_hold",
      "terminal_unobserved",
    ):
      simulation.data.time += 0.002
      observer(simulation, phase)
  with target.open(newline="") as file:
    rows = list(csv.DictReader(file))
  assert [row["phase"] for row in rows] == [
    "four_finger_press",
    "slide_card",
    "edge_hold",
    "terminal_unobserved",
  ]
  assert observer.last_task_phase == "edge_hold"
  assert observer.last_time_s == pytest.approx(0.008)
  assert len(observer.slide_positions) == 2


def test_observer_preserves_telemetry_and_distinguishes_video_failure(tmp_path):
  """An encoder failure must escape the executor's physical RuntimeError path."""
  import csv

  script = _script()

  class FakeTelemetry:
    def sample(self, phase):
      return {"time_s": simulation.data.time, "phase": phase, "card_x_m": 0.58}

  class BrokenVideo:
    def observe(self, _sim, _phase):
      raise RuntimeError("encoder stopped")

  simulation = SimpleNamespace(
    data=SimpleNamespace(time=0.002, geom_xpos=np.tile([0.6, 0, 0], (4, 1))),
  )
  target = tmp_path / "trace.csv"
  with target.open("x", newline="") as file:
    observer = script["_ForceTrialObserver"](
      FakeTelemetry(), (0, 1, 2, 3), file, video=BrokenVideo()
    )
    with pytest.raises(script["FrictionVideoError"], match="encoder stopped"):
      observer(simulation, "slide_card")

  with target.open(newline="") as file:
    rows = list(csv.DictReader(file))
  assert [row["phase"] for row in rows] == ["slide_card"]
  assert observer.last_time_s == pytest.approx(0.002)
  assert not issubclass(script["FrictionVideoError"], RuntimeError)


@pytest.mark.parametrize("record", (False, True))
@pytest.mark.parametrize("trial_success", (False, True))
def test_force_trial_outputs_streamed_artifacts_without_a_robot(
  tmp_path, monkeypatch, record, trial_success
):
  """Exercise the complete CLI output path without a physical rollout or GL."""
  import json

  from kaihand_tactile_env.shared import task_video
  from kaihand_tactile_env.tasks.poker_draw import telemetry, telemetry_plot

  script = _script()
  globals_ = script["_run_force_trials"].__globals__
  directory = tmp_path / "baseline"
  options = ["--friction", "0.1", "--diagnostics-dir", str(directory)]
  if record:
    options.extend(("--record", "--record-fps", "12"))
  args = script["_parse_args"](options)
  script["_prepare_outputs"](args)
  execute_calls = 0
  reset_calls = 0
  simulation_instances = []
  video_instances = []

  class FakeSimulation:
    def __init__(self, *_args, **_kwargs):
      self.data = SimpleNamespace(time=0.0, geom_xpos=np.tile([0.6, 0, 0], (4, 1)))
      self.model = SimpleNamespace(geom=lambda _name: SimpleNamespace(id=0))
      simulation_instances.append(self)

    def reset(self, **kwargs):
      nonlocal reset_calls
      reset_calls += 1
      assert kwargs == {"seed": 0, "object_xy_jitter": 0.0, "object_yaw_jitter": 0.0}

  class FakeTelemetry:
    def __init__(self, sim, *, target_force_n):
      self.sim = sim
      assert target_force_n == 0.35

    def sample(self, phase):
      return {"time_s": self.sim.data.time, "phase": phase, "card_x_m": 0.58}

    def summary(self):
      return {
        "slide_and_edge": {
          "fingers": {
            name: {"first_sustained_slip": None}
            for name in ("index", "middle", "ring", "pinky")
          }
        }
      }

    def metadata(self):
      return {"source": "fake", "target_force_per_finger_n": 0.35}

  @dataclass
  class FakeResult:
    success: bool
    error: str | None
    initial_card_pose: np.ndarray
    edge_card_pose: np.ndarray
    established_press_normal_forces_n: tuple
    slide_finger_normal_force_means_n: tuple
    slide_finger_contact_fractions: tuple
    maximum_overhang_fraction: float
    slide_press_control_qualified: bool

  class FakeExecutor:
    def __init__(self, sim, *, press_force_per_finger_n, observer):
      assert press_force_per_finger_n == 0.35
      self.sim, self.observer = sim, observer

    def execute_slide_only(self, _plan):
      nonlocal execute_calls
      execute_calls += 1
      for phase in ("four_finger_press", "slide_card", "edge_hold"):
        self.sim.data.time += 0.002
        self.observer(self.sim, phase)
      return FakeResult(
        trial_success,
        None if trial_success else "physical force-quality failure",
        np.zeros(7),
        np.zeros(7),
        (0.35,) * 4,
        (0.35,) * 4,
        (1.0,) * 4,
        0.50 if trial_success else 0.40,
        trial_success,
      )

  class FakeVideoRecorder:
    def __init__(
      self,
      sim,
      output_path,
      *,
      fps,
      preview,
      metadata,
    ):
      assert sim is simulation_instances[0]
      assert fps == 12.0
      assert preview is False
      assert metadata["scope"] == "press_slide_edge_only_not_full_draw"
      self.output_path = Path(output_path)
      self.metadata_path = self.output_path.with_suffix(".json")
      self.metadata = metadata
      self.observe_calls = []
      self.outcomes = []
      self.finish_calls = []
      self.enter_count = 0
      self.exit_count = 0
      video_instances.append(self)

    def __enter__(self):
      self.enter_count += 1
      return self

    def __exit__(self, exception_type, exception, _traceback):
      self.exit_count += 1
      if not self.finish_calls:
        self.finish(success=False, error=exception or "context closed before finish")

    def observe(self, sim, phase):
      assert sim is simulation_instances[0]
      self.observe_calls.append((float(sim.data.time), phase))

    def set_outcome(self, outcome):
      self.outcomes.append(outcome)

    def finish(self, success, error=None):
      self.finish_calls.append((success, error))
      self.output_path.write_bytes(b"fake MP4")
      self.metadata_path.write_text("{}\n", encoding="utf-8")
      return self.output_path, self.metadata_path

  def fake_plot(source, destination, *, target_force_n):
    assert target_force_n == 0.35
    assert source.exists()
    assert destination.with_name("trial_001.json").exists()
    with destination.open("xb") as stream:
      stream.write(b"fake PNG")
    return destination

  for key, value in {
    "ArmHandSimulation": FakeSimulation,
    "model_with_table_card_friction": lambda *_: nullcontext("fake-model"),
    "set_table_card_friction": lambda *_: None,
    "PokerDrawPlanner": lambda _: SimpleNamespace(plan=lambda: object()),
    "PokerDrawExecutor": FakeExecutor,
    "model_fingerprint": lambda _: "unchanged-scene",
    "press_control_metadata": lambda: {"unchanged_controller": True},
    "_measurement_source_hashes": lambda: {"telemetry.py": "fake-hash"},
    "TaskVideoRecorder": FakeVideoRecorder,
  }.items():
    monkeypatch.setitem(globals_, key, value)
  monkeypatch.setattr(task_video, "TaskVideoRecorder", FakeVideoRecorder)
  monkeypatch.setattr(telemetry, "PokerFrictionTelemetry", FakeTelemetry)
  monkeypatch.setattr(telemetry_plot, "plot_friction_trace", fake_plot)
  script["_run_force_trials"](args)
  report = json.loads((directory / "summary.json").read_text())
  assert report["schema_version"] == 4
  assert report["scope"] == "press_slide_edge_only_not_full_draw"
  assert report["state_noise"] is False
  assert len(report["trials"]) == 1
  row = report["trials"][0]
  assert row["success"] is trial_success
  assert row["stage_outcome"]["half_overhang_reached"] is trial_success
  assert row["stage_outcome"]["observed_phase_sample_counts"]["edge_hold"] == 1
  assert row["diagnostic_sustained_slip_fingers"] == []
  assert execute_calls == 1
  assert reset_calls == 1
  expected_files = {
    "summary.json",
    "trial_001.json",
    "trial_001_timeseries.csv",
    "trial_001_curves.png",
  }
  if record:
    expected_files.update(("trial_001_video.mp4", "trial_001_video.json"))
    assert row["video_artifacts"] == {
      "mp4": str(directory / "trial_001_video.mp4"),
      "json": str(directory / "trial_001_video.json"),
    }
    assert len(video_instances) == 1
    video = video_instances[0]
    assert video.enter_count == 1
    assert video.exit_count == 1
    assert [phase for _time, phase in video.observe_calls] == [
      "initial",
      "four_finger_press",
      "slide_card",
      "edge_hold",
    ]
    assert video.finish_calls == [(trial_success, None)]
    assert len(video.outcomes) == 1
    assert video.outcomes[0]["success"] is trial_success
    assert video.outcomes[0]["error"] == (
      None if trial_success else "physical force-quality failure"
    )
    assert "stage_outcome" in video.outcomes[0]
    assert "telemetry" in video.outcomes[0]
  else:
    assert row["video_artifacts"] is None
    assert video_instances == []
  assert {p.name for p in directory.iterdir()} == expected_files
