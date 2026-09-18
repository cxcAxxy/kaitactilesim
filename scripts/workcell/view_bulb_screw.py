#!/usr/bin/env python3
"""Inspect the bulb scene, execute the robot task, or run a mechanics demo."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  CameraConfig,
  model_fingerprint,
)
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import (
  BulbScrewMonitor,
  BulbScrewSimulation,
)

CAMERAS = (*SHARED_CAMERA_NAMES, "bulb_closeup")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--run-task",
    action="store_true",
    help="Run the automatic robot pick-and-screw task",
  )
  parser.add_argument("--result-json", type=Path, help="New file for the task outcome")
  parser.add_argument(
    "--grasp",
    choices=("five-finger", "pinch"),
    default="five-finger",
    help="Five-finger drive with fixed wrist (default); pinch explicitly selects the legacy wrist-driven controller",
  )
  parser.add_argument(
    "--speed",
    choices=("fast", "normal"),
    default="fast",
    help="Robot motion timing; normal retains the original conservative timing",
  )
  parser.add_argument(
    "--video",
    type=Path,
    help="New MP4 with scene/overhead/tactile panels; requires --run-task",
  )
  parser.add_argument("--video-fps", type=float, default=10.0)
  parser.add_argument(
    "--evaluation-review-dir",
    type=Path,
    help="New common review directory with RGB, ten-finger heatmaps, and force curves",
  )
  parser.add_argument(
    "--evaluation-video-fps", type=int, choices=(5, 10), default=10
  )
  parser.add_argument(
    "--threaded", action="store_true", help="Start at the thread entrance"
  )
  parser.add_argument(
    "--mechanics-demo",
    action="store_true",
    help="Start threaded and apply 30 mNm to the bulb; this is not a robot policy",
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Idle/demo seconds; 0 means until closed; ignored with --run-task",
  )
  parser.add_argument("--camera", choices=("free", *CAMERAS), default="free")
  parser.add_argument("--show-probes", action="store_true")
  parser.add_argument(
    "--snapshot-dir", type=Path, help="New directory for camera PNGs and state.json"
  )
  args = parser.parse_args(argv)
  if not math.isfinite(args.duration) or args.duration < 0:
    parser.error("--duration must be finite and nonnegative")
  if not math.isfinite(args.video_fps) or args.video_fps <= 0:
    parser.error("--video-fps must be finite and positive")
  if args.run_task and (args.threaded or args.mechanics_demo):
    parser.error(
      "--run-task starts on the tabletop and cannot use --threaded or --mechanics-demo"
    )
  if args.video is not None:
    if not args.run_task:
      parser.error("--video requires --run-task")
    if args.video.suffix.lower() != ".mp4":
      parser.error("--video must end in .mp4")
    if args.video.exists() or args.video.with_suffix(".json").exists():
      parser.error("--video and its JSON sidecar must be new files")
  if args.evaluation_review_dir is not None:
    if not args.run_task:
      parser.error("--evaluation-review-dir requires --run-task")
    if args.evaluation_review_dir.exists():
      parser.error("--evaluation-review-dir must be a new directory")
  if args.result_json is not None:
    if args.result_json.exists():
      parser.error("--result-json must be a new file")
    conflicts = []
    if args.video is not None:
      conflicts.extend(
        [args.video.resolve(), args.video.with_suffix(".json").resolve()]
      )
    if args.snapshot_dir is not None:
      conflicts.append((args.snapshot_dir / "state.json").resolve())
    if args.result_json.resolve() in conflicts:
      parser.error("--result-json must differ from video and snapshot output paths")
  if args.snapshot_dir is not None and args.snapshot_dir.exists():
    parser.error("--snapshot-dir must be a new directory")
  if args.snapshot_dir is not None and args.video is not None:
    if args.video.resolve().is_relative_to(args.snapshot_dir.resolve()):
      parser.error("--video must be outside --snapshot-dir")
  if args.duration == 0 and (args.headless or args.mechanics_demo):
    args.duration = 12.0 if args.mechanics_demo else 2.0
  return args


def snapshot(simulation, directory, report):
  from kaihand_tactile_env.shared.rendering import WorkcellRenderer
  from PIL import Image

  directory.mkdir(parents=True, exist_ok=False)
  cameras = tuple(
    CameraConfig(name, width=960, height=720, depth=False, segmentation=False)
    for name in CAMERAS
  )
  with WorkcellRenderer(
    simulation.model, cameras, visible_geom_groups=(0, 1, 2)
  ) as renderer:
    for camera in cameras:
      Image.fromarray(renderer.capture(simulation.data, camera)["rgb"]).save(
        directory / f"{camera.name}.png"
      )
  (directory / "state.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
  )


def run(args):
  simulation = BulbScrewSimulation()
  if args.threaded or args.mechanics_demo:
    simulation.initialize_threaded()
  monitor = BulbScrewMonitor(simulation)
  probes = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  forces = SolverDistributedTactileProvider(simulation.model)
  stopped = False

  def stop(*_):
    nonlocal stopped
    stopped = True

  def key(code):
    if code in (ord("Q"), ord("q"), 256):
      stop()

  state = monitor.update()
  bottom_contact_seen = False
  task_result = None
  previous_handler = signal.signal(signal.SIGINT, stop)

  def advance():
    nonlocal state, bottom_contact_seen
    if simulation.data.time < state.timestamp - 1e-12:
      # Native viewer reset also resets robot qpos; restore the task home pose.
      if args.threaded or args.mechanics_demo:
        simulation.initialize_threaded()
      else:
        simulation.reset()
      monitor.reset()
      bottom_contact_seen = False
    if args.mechanics_demo:
      simulation.data.xfrc_applied[simulation.model.body("bulb").id, 5] = (
        0.0 if bottom_contact_seen else -config.MECHANICS_DEMO_TORQUE_NM
      )
    simulation.step()
    state = monitor.update()
    bottom_contact_seen |= state.backstop_load_n > 0.2

  def running():
    return not stopped and (args.duration == 0 or simulation.data.time < args.duration)

  def execute_task(viewer=None):
    from contextlib import ExitStack

    from kaihand_tactile_env.tasks.bulb_screw.execution import BulbScrewExecutor

    nonlocal state, task_result
    last_sync = time.monotonic()
    last_time = float(simulation.data.time)
    last_phase = None
    with ExitStack() as resources:
      video = None
      evaluation_video = None
      if args.video is not None:
        from kaihand_tactile_env.shared.task_video import TaskVideoRecorder

        video = resources.enter_context(
          TaskVideoRecorder(
            simulation,
            args.video,
            fps=args.video_fps,
            preview=False,
            tactile_provider=probes,
            metadata={
              "scene": config.SCENE_NAME,
              "policy": config.ROTATION_DRIVER
              if args.grasp == "five-finger"
              else "legacy_pinch_wrist",
              "grasp_mode": args.grasp,
              "speed": args.speed,
              "main_camera": "bulb_closeup",
              "model_fingerprint": model_fingerprint(simulation.model_path),
              "external_bulb_wrench": False,
              "hand_bulb_weld": False,
            },
          )
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = simulation.model.camera("bulb_closeup").id
        video.follow_viewer_camera(camera)
      if args.evaluation_review_dir is not None:
        from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

        evaluation_video = EvaluationVideo(
          simulation,
          args.evaluation_review_dir,
          fps=args.evaluation_video_fps,
          second_camera="global",
          heading="BULB SCREW EVALUATION",
          metadata={"controller": "known_state_task_executor"},
        )
        evaluation_video.capture_due("initial", force=True)

      def observe(sim, phase):
        nonlocal last_sync, last_time, last_phase
        if phase != last_phase:
          if phase in {"lift", "align_thread", "turn", "release", "verify_seated"}:
            print(
              f"t={sim.data.time:.2f}s {phase}; turns={executor._state.clockwise_turns:.3f}",
              flush=True,
            )
          last_phase = phase
        if video is not None:
          video.observe(sim, phase)
        if evaluation_video is not None:
          evaluation_video.capture_due(phase)
        if viewer is not None and sim.data.time - last_time >= 1 / 60:
          time.sleep(max(0, sim.data.time - last_time - (time.monotonic() - last_sync)))
          viewer.sync()
          last_sync, last_time = time.monotonic(), float(sim.data.time)

      executor = BulbScrewExecutor(
        simulation,
        observer=observe,
        should_stop=lambda: stopped or (viewer is not None and not viewer.is_running()),
        grasp_mode=args.grasp,
        speed=args.speed,
      )
      evaluation_error = None
      try:
        task_result = executor.execute()
        state = task_result.state
        if video is not None:
          video.set_outcome(asdict(task_result))
          video.finish(
            task_result.success, None if task_result.success else task_result.reason
          )
      except BaseException as error:
        evaluation_error = f"{type(error).__name__}: {error}"
        raise
      finally:
        if evaluation_video is not None:
          evaluation_video.capture_due("terminal", force=True)
          evaluation_video.finish(
            status=(
              "success"
              if task_result is not None and task_result.success
              else "task_not_completed"
            ),
            evaluation=(asdict(task_result) if task_result is not None else None),
            error=evaluation_error,
          )

  try:
    if args.headless:
      if args.run_task:
        execute_task()
      else:
        while running():
          advance()
    else:
      from mujoco import viewer as mjviewer

      with mjviewer.launch_passive(
        simulation.model, simulation.data, key_callback=key
      ) as viewer:
        viewer.opt.geomgroup[4] = 0
        viewer.opt.geomgroup[5] = args.show_probes
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
        if args.show_probes:
          ids = [
            i
            for i in range(simulation.model.ngeom)
            if simulation.model.geom(i).name.startswith("workcell_genesis_probe_")
          ]
          simulation.model.geom_rgba[ids] = (0.1, 0.5, 1.0, 0.8)
        if args.camera == "free":
          viewer.cam.lookat[:] = (0.40, -0.06, 0.93)
          viewer.cam.distance = 1.4
          viewer.cam.azimuth = 145
          viewer.cam.elevation = -28
        else:
          viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
          viewer.cam.fixedcamid = simulation.model.camera(args.camera).id
        print("Bulb scene: clockwise from above screws down; Q/Esc/Ctrl+C exits.")
        if args.mechanics_demo:
          print("Mechanics demo: external bulb torque; robot remains at home.")
        if args.run_task:
          execute_task(viewer)
        else:
          while running() and viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
              for _ in range(8):
                if running():
                  advance()
            viewer.sync()
            time.sleep(max(0, 8 * simulation.timestep - (time.monotonic() - started)))
  finally:
    simulation.data.xfrc_applied[simulation.model.body("bulb").id] = 0
    signal.signal(signal.SIGINT, previous_handler)
  depth_sample = probes.read(simulation.data)
  force_sample = forces.read(simulation.data)
  report = {
    "scene": config.SCENE_NAME,
    "mechanics_version": config.MECHANICS_VERSION,
    "model_fingerprint": model_fingerprint(simulation.model_path),
    "initialization": "threaded"
    if args.threaded or args.mechanics_demo
    else "tabletop",
    "external_torque_demo": args.mechanics_demo,
    "task_result": asdict(task_result) if task_result is not None else None,
    "interrupted": stopped,
    "state": asdict(state),
    "probe_count": simulation.genesis_probe_layout.count,
    "tactile_contact_fingers": int(np.count_nonzero(depth_sample.contact)),
    "fingertip_normal_force_n": force_sample.normal_force_n.tolist(),
    "bulb_pose_wxyz": simulation.object_pose("bulb").tolist(),
  }
  if args.snapshot_dir is not None and not stopped:
    snapshot(simulation, args.snapshot_dir, report)
  if args.result_json is not None:
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("x", encoding="utf-8") as stream:
      stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps(report, indent=2, allow_nan=False))
  failed = (task_result is not None and not task_result.success) or (
    args.mechanics_demo and not state.success
  )
  return 130 if stopped else 1 if failed else 0


def main():
  raise SystemExit(run(parse_args()))


if __name__ == "__main__":
  main()
