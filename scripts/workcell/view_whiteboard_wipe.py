#!/usr/bin/env python3
"""Inspect, execute, or record the independent 45-degree whiteboard task."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES
from kaihand_tactile_env.tasks.whiteboard_wipe.execution import WhiteboardWipeExecutor
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--headless", action="store_true")
  p.add_argument("--run-task", action="store_true")
  p.add_argument("--video-fps", type=int, choices=(2, 5, 10), default=10)
  p.add_argument("--duration", type=float, default=2.0)
  p.add_argument(
    "--camera", choices=("free", *SHARED_CAMERA_NAMES, "board_overview"), default="free"
  )
  p.add_argument(
    "--output-dir",
    type=Path,
    help="New directory for result, synchronized video, HDF5 and force curves",
  )
  p.add_argument(
    "--ink-seed",
    type=int,
    help="Randomize ink center reproducibly within the supported board region",
  )
  args = p.parse_args(argv)
  if args.ink_seed is not None and args.ink_seed < 0:
    p.error("--ink-seed must be nonnegative")
  if not math.isfinite(args.duration) or args.duration <= 0:
    p.error("--duration must be finite and positive")
  if args.output_dir is not None:
    if not args.run_task:
      p.error("--output-dir requires --run-task")
    if args.output_dir.exists():
      p.error("--output-dir must not already exist")
  return args


def run(args):
  sim = WhiteboardWipeSimulation(ink_seed=args.ink_seed)
  viewer = recorder = None
  wall_start = time.monotonic()
  last_sync = -1.0
  try:
    if args.output_dir is not None:
      from kaihand_tactile_env.tasks.whiteboard_wipe.recording import WhiteboardRecorder

      recorder = WhiteboardRecorder(sim, args.output_dir, video_fps=args.video_fps)
    if not args.headless:
      import mujoco.viewer as mj_viewer

      viewer = mj_viewer.launch_passive(sim.model, sim.data)
      viewer.opt.geomgroup[3:] = 0
      if args.camera == "free":
        viewer.cam.lookat[:] = (0.55, -0.18, 0.84)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -30
      else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera(args.camera).id

    def observe(current, phase):
      nonlocal last_sync
      if recorder is not None:
        recorder.observe(current, phase)
      if viewer is not None:
        if not viewer.is_running():
          raise KeyboardInterrupt
        if current.data.time - last_sync >= 1 / 30:
          delay = current.data.time - (time.monotonic() - wall_start)
          if delay > 0:
            time.sleep(min(delay, 0.04))
          viewer.sync()
          last_sync = float(current.data.time)

    if args.run_task:
      result = WhiteboardWipeExecutor(sim, observe).run()
    else:
      for _ in range(round(args.duration / 0.01)):
        sim.step(round(0.01 / sim.timestep))
        observe(sim, "idle")
      result = dict(
        scene=sim.scene,
        elapsed_s=float(sim.data.time),
        success=not any(w.number for w in sim.data.warning),
      )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if recorder is not None:
      recorder.finish(result)
    return 0 if result["success"] else 1
  finally:
    if recorder is not None:
      recorder.close()
    if viewer is not None:
      viewer.close()


if __name__ == "__main__":
  try:
    raise SystemExit(run(parse_args()))
  except KeyboardInterrupt:
    raise SystemExit(130) from None
