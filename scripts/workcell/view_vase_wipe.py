#!/usr/bin/env python3
"""Inspect or execute the independent shallow-vase wiping prototype."""

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  TRAINING_CAMERA_NAMES,
)
from kaihand_tactile_env.tasks.vase_wipe.review import RawCapture, Review
from kaihand_tactile_env.tasks.vase_wipe.task import (
  VaseWipeExecutor,
  VaseWipeSimulation,
)

CAMERAS = (*SHARED_CAMERA_NAMES, "vase_closeup", "vase_inside")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-task", action="store_true")
  parser.add_argument(
    "--stain-seed", type=int, help="Enable mild seeded stain randomization"
  )
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--duration", type=float, default=0.0, help="Idle seconds; 0 keeps the viewer open"
  )
  parser.add_argument("--camera", choices=("free", *CAMERAS), default="vase_closeup")
  parser.add_argument(
    "--output-dir",
    type=Path,
    help="New directory for MP4, preview, force curves and result",
  )
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument(
    "--buffer-rows",
    type=int,
    default=128,
    help="Raw-only lossless non-camera HDF5 append buffer, 0 disables; allowed 0..256",
  )
  parser.add_argument(
    "--raw-only",
    action="store_true",
    help="Save only HDF5, task result and minimal validation summary",
  )
  parser.add_argument(
    "--evaluation-review-dir",
    type=Path,
    help="New common review directory with RGB, ten-finger heatmaps, and force curves",
  )
  parser.add_argument("--evaluation-video-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument(
    "--cameras",
    nargs="+",
    choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
    help="Raw training cameras; review-only close-up cameras are unchanged",
  )
  args = parser.parse_args()
  if not np.isfinite(args.duration) or args.duration < 0:
    parser.error("--duration must be finite and nonnegative")
  if args.output_dir is not None:
    if not args.run_task:
      parser.error("--output-dir requires --run-task")
    if args.output_dir.exists():
      parser.error("--output-dir must be a new directory")
  if args.raw_only and args.output_dir is None:
    parser.error("--raw-only requires --output-dir")
  if args.evaluation_review_dir is not None:
    if not args.run_task:
      parser.error("--evaluation-review-dir requires --run-task")
    if args.evaluation_review_dir.exists():
      parser.error("--evaluation-review-dir must be a new directory")
  if args.headless and args.duration == 0:
    args.duration = 2.0
  if args.stain_seed is not None and args.stain_seed < 0:
    parser.error("--stain-seed must be nonnegative")
  if args.camera_hz <= 0:
    parser.error("--camera-hz must be positive")
  if tuple(args.cameras) != TRAINING_CAMERA_NAMES:
    parser.error("--cameras must be exactly: head left_wrist right_wrist")
  sim = VaseWipeSimulation(stain_seed=args.stain_seed)
  viewer = review = evaluation_video = None
  start = time.monotonic()
  last_sync = 0.0
  last_phase = None
  last_report_time = -1.0
  with ExitStack() as stack:
    if not args.headless:
      import mujoco.viewer

      viewer = stack.enter_context(mujoco.viewer.launch_passive(sim.model, sim.data))
      viewer.opt.geomgroup[:] = (1, 1, 1, 0, 0, 0)
      if args.camera != "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera(args.camera).id
      else:
        viewer.cam.lookat[:] = [0.54, -0.18, 0.87]
        viewer.cam.distance = 0.65
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -28
    if args.output_dir is not None:
      recorder_type = RawCapture if args.raw_only else Review
      recorder_kwargs = dict(
        camera_hz=args.camera_hz,
        cameras=tuple(args.cameras),
      )
      if args.raw_only:
        recorder_kwargs["buffer_rows"] = args.buffer_rows
      review = recorder_type(sim, args.output_dir, **recorder_kwargs)
      stack.callback(review.close)
    if args.evaluation_review_dir is not None:
      from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

      evaluation_video = EvaluationVideo(
        sim,
        args.evaluation_review_dir,
        fps=args.evaluation_video_fps,
        second_camera="global",
        heading="VASE WIPE EVALUATION",
        metadata={"controller": "known_state_task_executor"},
      )
      evaluation_video.capture_due("initial", force=True)

    def observe(sim, phase, state):
      nonlocal last_sync, last_phase, last_report_time
      if phase != last_phase or state.timestamp - last_report_time >= 1.0:
        print(
          f"{state.timestamp:5.2f}s {phase}: cleaned={state.cleaned_fraction:.1%}, "
          f"wall_peak={sim.peak_wall_force_n:.3f} N",
          flush=True,
        )
        last_phase = phase
        last_report_time = state.timestamp
      if review is not None:
        review.observe(sim, phase, state)
      if evaluation_video is not None:
        evaluation_video.capture_due(phase)
      if viewer is not None:
        if not viewer.is_running():
          raise KeyboardInterrupt
        if sim.data.time - last_sync >= 1 / 30:
          viewer.sync()
          last_sync = sim.data.time
        delay = start + sim.data.time - time.monotonic()
        if delay > 0:
          time.sleep(min(delay, 0.02))

    try:
      if args.run_task:
        result = VaseWipeExecutor(sim, observe).execute()
      else:
        while args.duration == 0 or sim.data.time < args.duration:
          sim.step(5)
          observe(sim, "tabletop_ready", sim.measure())
        result = {"task": "vase-wipe", "state": asdict(sim.measure())}
    except (KeyboardInterrupt, RuntimeError) as error:
      result = {
        "task": "vase-wipe",
        "success": False,
        "motion_completed": False,
        "cleaned_patch_count": int((sim.dirt <= 0.10).sum()),
        "patch_count": len(sim.dirt),
        "error": str(error) or "Viewer closed",
        "state": asdict(sim.measure()),
        "cleaning": sim.cleaning.report(),
        "remaining_dirt": sim.dirt.tolist(),
        "physics_warning_count": int(sum(w.number for w in sim.data.warning)),
      }
    result["stain_randomization"] = sim.stain_randomization
    if evaluation_video is not None:
      evaluation_video.capture_due("terminal", force=True)
      evaluation_video.finish(
        status="success" if result.get("success") is True else "task_not_completed",
        evaluation=result,
        error=result.get("error"),
      )
    if review is not None:
      review.finish(result)
    print(json.dumps(result, indent=2, allow_nan=False))
    if viewer is not None and args.run_task and result.get("success"):
      end = time.monotonic() + 3
      while viewer.is_running() and time.monotonic() < end:
        viewer.sync()
        time.sleep(0.03)
  return 1 if result.get("success") is False else 0


if __name__ == "__main__":
  raise SystemExit(main())
