#!/usr/bin/env python3
"""Open the dual-arm workcell and print live fingertip contact aggregates."""

from __future__ import annotations

import argparse
import math
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mujoco.viewer
from kaihand_tactile_env.shared.config import (
  SCENE_NAMES,
  SHARED_CAMERA_NAMES,
  default_model_path,
  model_fingerprint,
  task_config,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import (
  GENESIS_PROBE_GEOM_PREFIX,
  GenesisProbeTactileProvider,
  SolverContactTactileProvider,
)
from kaihand_tactile_env.tasks.pick_place.task import (
  KnownStateGraspPlanner,
  PickPlaceExecutor,
)
from kaihand_tactile_env.tasks.poker_draw.acceptance import TASK_COMPLETION_POLICY
from kaihand_tactile_env.tasks.poker_draw.task import (
  PokerDrawExecutor,
  PokerDrawPlanner,
)


class ViewerClosed(RuntimeError):
  """End the scripted episode when the user closes its viewer."""


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--duration", type=float, default=0.0, help="0 waits until closed"
  )
  parser.add_argument(
    "--task",
    choices=("scene", "pick-place", "poker-draw"),
    default="scene",
    help="Show the idle scene or execute one complete task episode.",
  )
  parser.add_argument(
    "--scene",
    choices=SCENE_NAMES,
    help="Scene shown for --task scene; task selections imply their own scene.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--camera", choices=("free", *SHARED_CAMERA_NAMES), default="free")
  parser.add_argument("--object-xy-jitter", type=float)
  parser.add_argument("--object-yaw-jitter", type=float)
  parser.add_argument(
    "--playback-speed",
    type=float,
    help=(
      "Task playback multiplier; defaults to 1.25 for poker-draw and 1.0 "
      "otherwise. Physics still advances at the configured 500 Hz."
    ),
  )
  parser.add_argument(
    "--viewer-hz",
    type=float,
    default=60.0,
    help="Maximum native-viewer sync frequency (default: 60 Hz).",
  )
  parser.add_argument(
    "--tactile-source",
    choices=(SolverContactTactileProvider.source, GenesisProbeTactileProvider.source),
    default=GenesisProbeTactileProvider.source,
  )
  parser.add_argument(
    "--show-probes",
    action="store_true",
    help="Render the loaded Genesis probe spheres and live activation when available.",
  )
  parser.add_argument(
    "--probe-layout",
    type=Path,
    help="Optional bilateral layout used for visual inspection.",
  )
  parser.add_argument(
    "--record",
    nargs="?",
    const="",
    metavar="VIDEO.mp4",
    help="Save one task as a synchronized scene/overhead/tactile video; optional path.",
  )
  parser.add_argument(
    "--record-fps",
    type=float,
    default=15.0,
    help="Composite video frame rate in simulation time (default: 15, maximum: 30).",
  )
  parser.add_argument(
    "--record-no-preview",
    action="store_true",
    help="Write the composite video without its additional live preview window.",
  )
  parser.add_argument(
    "--table-card-friction",
    type=float,
    help="Experimental sliding friction for table/card only; poker-draw scenes only.",
  )
  parser.add_argument(
    "--press-force-per-finger",
    "--press-force",
    dest="press_force_per_finger_n",
    type=float,
    help="Poker slide normal-force target in N for EACH of the four fingers.",
  )
  args = parser.parse_args()
  if args.playback_speed is not None and (
    not math.isfinite(args.playback_speed) or args.playback_speed <= 0.0
  ):
    parser.error("--playback-speed must be positive")
  if not math.isfinite(args.viewer_hz) or args.viewer_hz <= 0.0:
    parser.error("--viewer-hz must be positive")
  if not math.isfinite(args.record_fps) or not 0.0 < args.record_fps <= 30.0:
    parser.error("--record-fps must be in (0, 30]")
  if args.record is not None and args.task == "scene":
    parser.error("--record requires --task poker-draw or --task pick-place")
  if args.record and Path(args.record).suffix.lower() != ".mp4":
    parser.error("--record output path must end in .mp4")
  if args.task != "scene" and args.scene is not None and args.scene != args.task:
    parser.error("--scene must match an explicitly selected --task")
  scene = args.scene or ("poker-draw" if args.task == "poker-draw" else "pick-place")
  args.scene = scene
  if args.table_card_friction is not None:
    if scene != "poker-draw":
      parser.error("--table-card-friction requires a poker-draw scene")
    if not math.isfinite(args.table_card_friction) or args.table_card_friction < 0.0:
      parser.error("--table-card-friction must be finite and nonnegative")
  settings = task_config(scene)
  if args.press_force_per_finger_n is not None:
    if scene != "poker-draw" or args.task != "poker-draw":
      parser.error("--press-force requires --task poker-draw")
    if (
      not math.isfinite(args.press_force_per_finger_n)
      or args.press_force_per_finger_n <= 0
    ):
      parser.error("--press-force must be finite and positive (N per finger)")
  if args.task == "poker-draw" and args.press_force_per_finger_n is None:
    args.press_force_per_finger_n = settings.DEFAULT_PRESS_FORCE_PER_FINGER_N
  if args.object_xy_jitter is None:
    args.object_xy_jitter = settings.DEFAULT_XY_JITTER
  if args.object_yaw_jitter is None:
    args.object_yaw_jitter = settings.DEFAULT_YAW_JITTER
  if args.playback_speed is None:
    args.playback_speed = settings.DEFAULT_PLAYBACK_SPEED
  return args


def main() -> None:
  args = parse_args()
  with ExitStack() as resources:
    model_path = default_model_path(args.scene)
    if args.table_card_friction is not None:
      from kaihand_tactile_env.tasks.poker_draw.friction import (
        model_with_table_card_friction,
      )

      model_path = resources.enter_context(
        model_with_table_card_friction(model_path, args.table_card_friction)
      )
    if args.scene == "bulb-screw":
      from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation

      simulation = BulbScrewSimulation(
        model_path=model_path, probe_layout_path=args.probe_layout,
      )
    else:
      simulation = ArmHandSimulation(
        model_path,
        probe_layout_path=args.probe_layout,
        scene=args.scene,
      )
    try:
      run_view(args, simulation)
    except ViewerClosed:
      print(
        "Viewer closed: task interrupted; any partial recording is marked incomplete."
      )


def run_view(args: argparse.Namespace, simulation: ArmHandSimulation) -> None:
  simulation.reset(
    seed=args.seed,
    object_xy_jitter=args.object_xy_jitter,
    object_yaw_jitter=args.object_yaw_jitter,
  )

  def restore_task_initial_state() -> None:
    simulation.reset(
      seed=args.seed,
      object_xy_jitter=args.object_xy_jitter,
      object_yaw_jitter=args.object_yaw_jitter,
    )

  tactile = (
    GenesisProbeTactileProvider(simulation.model, simulation.genesis_probe_layout)
    if args.tactile_source == GenesisProbeTactileProvider.source
    else SolverContactTactileProvider(simulation.model)
  )
  probe_geom_ids = [
    index
    for index in range(simulation.model.ngeom)
    if simulation.model.geom(index).name.startswith(GENESIS_PROBE_GEOM_PREFIX)
  ]
  if args.show_probes:
    simulation.model.geom_rgba[probe_geom_ids] = (0.9, 0.15, 0.05, 0.75)
  wall_start = time.monotonic()
  last_print = -1.0
  with (
    mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer,
    ExitStack() as resources,
  ):
    viewer.opt.geomgroup[5] = args.show_probes
    if args.camera != "free":
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = simulation.model.camera(args.camera).id
    video = None
    if args.record is not None:
      from kaihand_tactile_env.shared.task_video import TaskVideoRecorder
      from kaihand_tactile_env.tasks.poker_draw.friction import press_control_metadata

      output_path = (
        Path(args.record)
        if args.record
        else Path("datasets")
        / args.scene.replace("-", "_")
        / "videos"
        / f"{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
      )
      video = resources.enter_context(
        TaskVideoRecorder(
          simulation,
          output_path,
          fps=args.record_fps,
          tactile_provider=tactile,
          preview=not args.record_no_preview,
          metadata={
            "scene": args.scene,
            "base_model_fingerprint": model_fingerprint(default_model_path(args.scene)),
            "mujoco_version": mujoco.__version__,
            "physics_hz": 1.0 / simulation.timestep,
            "seed": args.seed,
            "object_xy_jitter": args.object_xy_jitter,
            "object_yaw_jitter": args.object_yaw_jitter,
            "playback_speed": args.playback_speed,
            "table_card_friction_override": args.table_card_friction,
            "press_force_per_finger_n": args.press_force_per_finger_n,
            "press_control": (
              press_control_metadata() if args.scene == "poker-draw" else None
            ),
          },
        )
      )
      # Fixed front camera for reproducible recordings; overhead and task tactile
      # pad images are shown beside it in the composite, not extra renderers.
      video.observe(simulation, "initial")
      print(f"Recording one task to {output_path}; video time is simulation time.")

    def task_observer(playback_speed: float):
      task_wall_start = time.monotonic()
      task_sim_start = float(simulation.data.time)
      next_sync_time = task_wall_start

      def observe(_: ArmHandSimulation, phase: str) -> None:
        nonlocal next_sync_time
        if not viewer.is_running():
          raise ViewerClosed("Native viewer was closed before the task completed")
        if video is not None:
          video.observe(simulation, phase)
        if time.monotonic() >= next_sync_time:
          viewer.sync()
          next_sync_time = time.monotonic() + 1.0 / args.viewer_hz
        target_wall_time = (
          task_wall_start + (simulation.data.time - task_sim_start) / playback_speed
        )
        delay = target_wall_time - time.monotonic()
        if delay > 0.0:
          time.sleep(delay)

      return observe

    if args.task == "pick-place":
      plan = KnownStateGraspPlanner(simulation).plan_pick_and_place("right")
      observe = task_observer(args.playback_speed)
      result = PickPlaceExecutor(simulation, observer=observe).execute(plan)
      print(
        f"pick-place finished: success={result.success}, "
        f"placed_in_box={result.placed_in_box}, phases={result.phases}"
      )
      # Restart the wall-clock reference so the post-task idle loop remains
      # real-time instead of trying to catch up with the completed episode.
      wall_start = time.monotonic() - simulation.data.time
    elif args.task == "poker-draw":
      plan = PokerDrawPlanner(simulation).plan("right")
      observe_poker = task_observer(args.playback_speed)
      result = PokerDrawExecutor(
        simulation,
        observer=observe_poker,
        press_force_per_finger_n=args.press_force_per_finger_n,
        # Tip-side contact permits short pressure fluctuations; still report
        # strict force quality separately from physical task completion.
        acceptance_policy=TASK_COMPLETION_POLICY,
      ).execute(plan)
      if video is not None:
        video.set_outcome(asdict(result))
      print(
        f"poker-draw finished: success={result.success}, "
        f"acceptance_policy={result.acceptance_policy}, "
        f"pressure_quality={result.pressure_quality}, "
        f"slide_force_mean_n={result.slide_finger_normal_force_means_n}, "
        f"slide_contact_fraction={result.slide_finger_contact_fractions}, "
        f"slide_pressure_qualified={result.slide_press_control_qualified}, "
        f"overhang={result.maximum_overhang_fraction:.3f}, "
        f"thumb_face_contact={result.thumb_face_contact}, "
        f"sustained_pinch={result.sustained_pinch}, "
        f"lift_opposition={result.lift_opposition_fraction:.3f}, "
        f"lift_four_fingers={result.lift_four_finger_fraction:.3f}, "
        f"inspection_opposition={result.inspection_opposition_fraction:.3f}, "
        f"inspection_four_fingers={result.inspection_four_finger_fraction:.3f}, "
        f"terminal_grip={result.terminal_grip_fingers}, "
        f"finger_pad_alignment="
        f"{result.minimum_terminal_finger_pad_alignment:.3f}, "
        f"thumb_pad_alignment={result.terminal_thumb_pad_alignment:.3f}, "
        f"fingertip_plane_angle="
        f"{result.maximum_terminal_fingertip_angle_to_card_plane_degrees:.1f} deg, "
        f"wrist_inward_turn={result.wrist_inward_turn_degrees:.1f} deg, "
        f"face_to_head={result.inspection_face_alignment:.3f}, "
        f"face_to_robot={result.inspection_face_robot_alignment:.3f}, "
        f"inspection_rotation={result.inspection_rotation_degrees:.1f} deg, "
        f"phases={result.phases}"
      )
      wall_start = time.monotonic() - simulation.data.time
    if video is not None:
      video_path, metadata_path = video.finish(success=bool(result.success))
      print(f"Saved task video: {video_path}\nVideo metadata: {metadata_path}")
    previous_sim_time = float(simulation.data.time)
    next_idle_sync = time.monotonic()
    while viewer.is_running():
      # The native Reset button calls mj_resetData directly.  Detect its time
      # rewind and reapply the same scripted initial state and actuator goals.
      if simulation.data.time + simulation.timestep < previous_sim_time:
        restore_task_initial_state()
        wall_start = time.monotonic()
      simulation.step()
      previous_sim_time = float(simulation.data.time)
      elapsed = time.monotonic() - wall_start
      if elapsed - last_print >= 0.5:
        sample = tactile.read(simulation.data)
        if args.show_probes and isinstance(tactile, GenesisProbeTactileProvider):
          colors = simulation.model.geom_rgba[probe_geom_ids]
          colors[:] = (0.9, 0.15, 0.05, 0.75)
          colors[tactile.probe_contact] = (0.1, 1.0, 0.15, 0.95)
        active = [
          (
            f"{name}: count={sample.contact_count[index]}, "
            f"aggregate={sample.normal_force[index]:.3f} "
            f"[{tactile.force_unit}]"
          )
          for index, name in enumerate(sample.link_names)
          if sample.contact_count[index] > 0
        ]
        print(f"t={simulation.data.time:.2f}s contacts: {', '.join(active) or 'none'}")
        last_print = elapsed
      if time.monotonic() >= next_idle_sync:
        viewer.sync()
        next_idle_sync = time.monotonic() + 1.0 / args.viewer_hz
      if args.duration > 0.0 and elapsed >= args.duration:
        break
      delay = simulation.timestep - (
        time.monotonic() - wall_start - simulation.data.time
      )
      if delay > 0.0:
        time.sleep(delay)


if __name__ == "__main__":
  main()
