"""CLI/lifecycle coverage without opening a window or executing a full task."""

from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import mujoco.viewer
import pytest


def _script():
  return run_path(str(Path(__file__).parents[1] / "scripts/workcell/view_workcell.py"))


@pytest.mark.parametrize("scene", ["pick-place", "poker-draw", "usb-insert", "bulb-screw"])
@pytest.mark.parametrize("camera", ["right_wrist", "left_wrist"])
def test_shared_wrist_camera_is_selectable_in_each_task(scene, camera, monkeypatch):
  root = Path(__file__).parents[1] / "scripts/workcell"
  if scene in ("usb-insert", "bulb-screw"):
    path = root / f"view_{scene.replace('-', '_')}.py"
    options = ["--camera", camera]
  else:
    path = root / "view_workcell.py"
    options = ["--task", scene, "--camera", camera]
  script = run_path(str(path))
  monkeypatch.setattr(sys, "argv", [str(path), *options])
  assert script["parse_args"]().camera == camera


@pytest.mark.parametrize(
  "options",
  (
    ["--record"],
    ["--task", "pick-place", "--table-card-friction", "0.8"],
    ["--task", "poker-draw", "--table-card-friction", "nan"],
    ["--task", "poker-draw", "--table-card-friction", "-1"],
    ["--task", "poker-draw", "--record-fps", "31"],
    ["--task", "poker-draw", "--record", "bad.avi"],
    ["--viewer-hz", "inf"],
    ["--task", "pick-place", "--press-force", "0.35"],
    ["--task", "poker-draw", "--press-force", "nan"],
    ["--task", "poker-draw", "--press-force", "0"],
    ["--task", "poker-draw", "--press-force", "-0.1"],
  ),
)
def test_invalid_record_options_fail_before_simulation(options, monkeypatch):
  script = _script()
  monkeypatch.setattr(sys, "argv", ["view", *options])
  with pytest.raises(SystemExit) as error:
    script["parse_args"]()
  assert error.value.code == 2


def test_record_options_preserve_task_defaults(monkeypatch):
  from kaihand_tactile_env.tasks.poker_draw.config import (
    DEFAULT_PRESS_FORCE_PER_FINGER_N,
  )

  script = _script()
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "view",
      "--task",
      "poker-draw",
      "--viewer-hz",
      "30",
      "--record",
      "--table-card-friction",
      "0.8",
    ],
  )
  args = script["parse_args"]()
  assert args.record == ""
  assert args.record_fps == 15.0
  assert args.playback_speed == 1.25
  assert args.table_card_friction == 0.8
  assert args.press_force_per_finger_n == DEFAULT_PRESS_FORCE_PER_FINGER_N
  assert args.object_xy_jitter == args.object_yaw_jitter == 0.0


def test_explicit_press_force_uses_newtons_per_finger(monkeypatch):
  script = _script()
  monkeypatch.setattr(
    sys, "argv", ["view", "--task", "poker-draw", "--press-force", "0.25"]
  )
  assert script["parse_args"]().press_force_per_finger_n == 0.25


@pytest.mark.parametrize("ending", ("success", "unsuccessful", "failure", "closed"))
def test_record_one_task_lifecycle(ending, monkeypatch, tmp_path):
  from kaihand_tactile_env.shared import task_video

  script = _script()
  runtime = script["main"].__globals__
  target = tmp_path / "task.mp4"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "view",
      "--task",
      "pick-place",
      "--record",
      str(target),
      "--record-no-preview",
      "--playback-speed",
      "1000",
    ],
  )
  events = []

  class Viewer:
    running = True
    opt = SimpleNamespace(geomgroup=[False] * 6)

    def __enter__(self):
      return self

    def __exit__(self, *args):
      events.append("viewer_close")

    def is_running(self):
      return self.running

    def sync(self):
      pass

  viewer = Viewer()
  monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *args: viewer)

  class Video:
    def __init__(self, simulation, path, **kwargs):
      assert path == target
      assert kwargs["preview"] is False
      assert kwargs["metadata"]["table_card_friction_override"] is None

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc, tb):
      if exc is not None:
        events.append(("interrupted", type(exc).__name__))
      events.append("video_close")

    def observe(self, simulation, phase):
      events.append(phase)

    def finish(self, *, success):
      events.append(("finish", success))
      return target, target.with_suffix(".json")

  class Executor:
    def __init__(self, simulation, *, observer):
      self.simulation = simulation
      self.observer = observer

    def execute(self, plan):
      if ending == "closed":
        viewer.running = False
      self.simulation.step()
      self.observer(self.simulation, "move")
      if ending == "failure":
        raise RuntimeError("test task failure")
      viewer.running = False
      return SimpleNamespace(
        success=ending == "success", placed_in_box=True, phases=("move",)
      )

  monkeypatch.setattr(task_video, "TaskVideoRecorder", Video)
  monkeypatch.setitem(runtime, "PickPlaceExecutor", Executor)
  monkeypatch.setitem(
    runtime,
    "KnownStateGraspPlanner",
    lambda simulation: SimpleNamespace(plan_pick_and_place=lambda side: None),
  )
  if ending == "failure":
    with pytest.raises(RuntimeError, match="test task failure"):
      script["main"]()
  else:
    script["main"]()
  assert events[0] == "initial"
  assert events[-2:] == ["video_close", "viewer_close"]
  assert (("finish", True) in events) == (ending == "success")
  assert (("finish", False) in events) == (ending == "unsuccessful")
  if ending == "closed":
    assert ("interrupted", "ViewerClosed") in events
    assert "move" not in events


def test_friction_view_builds_pair_override_and_cleans_wrapper(monkeypatch):
  from kaihand_tactile_env.tasks.poker_draw.friction import TABLE_CARD_PAIR_NAME

  script = _script()
  runtime = script["main"].__globals__
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "view",
      "--task",
      "poker-draw",
      "--table-card-friction",
      "0.8",
    ],
  )
  inspected = []

  def inspect_without_window(args, simulation):
    assert args.table_card_friction == 0.8
    assert simulation.model.pair(TABLE_CARD_PAIR_NAME).friction[:2] == pytest.approx(
      [0.8, 0.8]
    )
    assert simulation.model.geom("card_core_geom").friction[0] == pytest.approx(1.4)
    assert simulation.model_path.is_file()
    inspected.append(simulation.model_path)

  monkeypatch.setitem(runtime, "run_view", inspect_without_window)
  script["main"]()
  assert len(inspected) == 1
  assert not inspected[0].exists()
