#!/usr/bin/env python3
"""Inspect or execute the independent DIMM installation task."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES
from kaihand_tactile_env.tasks.install_ram.task import (
  RamInstallationMonitor,
  RamInstallSimulation,
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--run-task", action="store_true")
  parser.add_argument(
    "--duration",
    type=float,
    default=2.0,
    help="Idle simulation seconds; ignored by --run-task",
  )
  parser.add_argument(
    "--camera", choices=("free", *SHARED_CAMERA_NAMES, "ram_closeup"), default="free"
  )
  parser.add_argument("--result-json", type=Path)
  parser.add_argument(
    "--evaluation-review-dir",
    type=Path,
    help="New common review directory with RGB, ten-finger heatmaps, and force curves",
  )
  parser.add_argument(
    "--evaluation-video-fps", type=int, choices=(5, 10), default=10
  )
  args = parser.parse_args(argv)
  if not math.isfinite(args.duration) or args.duration <= 0:
    parser.error("--duration must be finite and positive")
  if args.result_json is not None and args.result_json.exists():
    parser.error("--result-json must be a new file")
  if args.evaluation_review_dir is not None:
    if not args.run_task:
      parser.error("--evaluation-review-dir requires --run-task")
    if args.evaluation_review_dir.exists():
      parser.error("--evaluation-review-dir must be a new directory")
  return args


def run(args):
  sim = RamInstallSimulation()
  executor = None
  if args.run_task:
    from kaihand_tactile_env.tasks.install_ram.execution import RamInstallExecutor

    executor = RamInstallExecutor(sim)
    if hasattr(executor, "prepare"):
      executor.prepare()
  monitor = RamInstallationMonitor(sim)
  viewer = evaluation_video = None
  outcome = None
  evaluation_error = None
  wall_start = time.monotonic()
  last_sync = 0.0

  def observe(current, phase):
    nonlocal last_sync
    if evaluation_video is not None:
      evaluation_video.capture_due(phase)
    if viewer is None:
      return
    if not viewer.is_running():
      raise KeyboardInterrupt
    now = float(current.data.time)
    if now - last_sync >= 1 / 30:
      delay = now - (time.monotonic() - wall_start)
      if delay > 0:
        time.sleep(min(delay, 0.04))
      viewer.sync()
      last_sync = now

  try:
    if not args.headless:
      import mujoco.viewer as mj_viewer

      viewer = mj_viewer.launch_passive(sim.model, sim.data)
      viewer.opt.geomgroup[3:] = 0
      if args.camera == "free":
        viewer.cam.lookat[:] = (0.49, -0.15, 0.77)
        viewer.cam.distance = 0.72
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
      else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera(args.camera).id
    if args.evaluation_review_dir is not None:
      from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

      evaluation_video = EvaluationVideo(
        sim,
        args.evaluation_review_dir,
        fps=args.evaluation_video_fps,
        second_camera="global",
        heading="RAM INSTALLATION EVALUATION",
        metadata={"controller": "known_state_task_executor"},
      )
      evaluation_video.capture_due("initial", force=True)
    if executor is not None:
      outcome = asdict(executor.run(observer=observe))
      status = 0 if outcome["success"] else 1
    else:
      state = monitor.measure()
      for _ in range(round(args.duration / sim.timestep)):
        sim.step()
        state = monitor.update()
        observe(sim, "idle")
      finite = bool(
        np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
      )
      outcome = {"scene": sim.scene, "finite_state": finite, "state": asdict(state)}
      status = 0 if finite else 1
  except BaseException as error:
    evaluation_error = f"{type(error).__name__}: {error}"
    raise
  finally:
    if evaluation_video is not None:
      evaluation_video.capture_due("terminal", force=True)
      evaluation_video.finish(
        status=(
          "success"
          if outcome is not None and outcome.get("success") is True
          else "task_not_completed"
        ),
        evaluation=outcome,
        error=evaluation_error,
      )
    if viewer is not None:
      viewer.close()
  report = json.dumps(outcome, indent=2, ensure_ascii=False, allow_nan=False)
  print(report)
  if args.result_json is not None:
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("x") as output:
      output.write(report + "\n")
  return status


if __name__ == "__main__":
  try:
    raise SystemExit(run(parse_args()))
  except KeyboardInterrupt:
    raise SystemExit(130) from None
