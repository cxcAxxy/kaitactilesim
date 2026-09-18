#!/usr/bin/env python3
"""Inspect the USB scene or run one automatic grasp-and-insertion episode."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from contextlib import contextmanager
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
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider
from kaihand_tactile_env.tasks.usb_insert.task import UsbInsertionMonitor

CAMERAS = (*SHARED_CAMERA_NAMES, "usb_closeup")


class _StopRequest:
  """Callbacks request shutdown; the main thread owns viewer cleanup."""

  def __init__(self) -> None:
    self.requested = False

  def on_sigint(self, _signum: int, _frame: object) -> None:
    # Do not raise into MuJoCo or acquire locks from a Python signal handler.
    self.requested = True

  def on_key(self, keycode: int) -> None:
    # GLFW uses 256 for Escape and uppercase ASCII codes for letter keys.
    # This callback runs on the viewer thread, so it must not close the viewer.
    if keycode in (ord("Q"), ord("q"), 256):
      self.requested = True


@contextmanager
def _stop_requests():
  stop = _StopRequest()
  previous = signal.getsignal(signal.SIGINT)
  signal.signal(signal.SIGINT, stop.on_sigint)
  try:
    yield stop
  finally:
    # Keep handling repeated Ctrl+C throughout the viewer's __exit__/close.
    signal.signal(signal.SIGINT, previous)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true", help="Run without a window")
  parser.add_argument(
    "--run-task",
    action="store_true",
    help="Execute one automatic grasp, alignment and insertion; default: idle scene",
  )
  initialization = parser.add_mutually_exclusive_group()
  initialization.add_argument(
    "--plug-face-down",
    action="store_true",
    help="Initialize the legacy USB mark-down orientation at reset",
  )
  initialization.add_argument(
    "--plug-for-insertion",
    action="store_true",
    help="Initialize the USB for pinky-forward pickup and palm-down insertion",
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Idle scene seconds only; ignored with --run-task; 0 = until closed (headless: 2 s)",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--motion-profile",
    choices=("fast", "baseline"),
    default="fast",
    help="USB motion schedule: fast (default) or the previous approximately 21 s baseline",
  )
  parser.add_argument(
    "--precontact-noise-mm",
    type=float,
    default=0.5,
    help="Pre-contact control noise: Gaussian XY sigma per axis in mm (default: 0.5; 0 disables; --run-task only)",
  )
  parser.add_argument(
    "--noise-seed",
    type=int,
    default=None,
    help="Reproduce control noise with this nonnegative seed; otherwise generate a fresh seed independently of --seed (--run-task only)",
  )
  parser.add_argument(
    "--xy-jitter-mm",
    type=float,
    default=None,
    help="Uniform initial X/Y offsets within +/- this many mm (default: 0; --plug-for-insertion only)",
  )
  parser.add_argument(
    "--yaw-jitter-deg",
    type=float,
    default=None,
    help="Uniform initial world-Z yaw offset within +/- these degrees (default: 0; --plug-for-insertion only)",
  )
  parser.add_argument("--camera", choices=("free", *CAMERAS), default="free")
  parser.add_argument("--show-probes", action="store_true")
  parser.add_argument(
    "--hide-camera-markers",
    action="store_true",
    help="Compatibility option: camera bodies, names and axes are now always hidden",
  )
  parser.add_argument(
    "--snapshot-dir",
    type=Path,
    help="New directory for all shared camera PNGs, USB closeup and state.json",
  )
  parser.add_argument(
    "--result-json",
    type=Path,
    help="New JSON file for the complete report; no rendering required",
  )
  args = parser.parse_args()
  randomization_requested = (
    args.xy_jitter_mm is not None or args.yaw_jitter_deg is not None
  )
  if randomization_requested and not args.plug_for_insertion:
    parser.error("USB jitter options require --plug-for-insertion")
  args.xy_jitter_mm = 0.0 if args.xy_jitter_mm is None else args.xy_jitter_mm
  args.yaw_jitter_deg = 0.0 if args.yaw_jitter_deg is None else args.yaw_jitter_deg
  for name in ("xy_jitter_mm", "yaw_jitter_deg", "precontact_noise_mm"):
    value = getattr(args, name)
    if not math.isfinite(value) or value < 0:
      parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
  if args.seed < 0:
    parser.error("--seed must be nonnegative")
  if args.noise_seed is not None and args.noise_seed < 0:
    parser.error("--noise-seed must be nonnegative")
  if not math.isfinite(args.duration) or args.duration < 0:
    parser.error("--duration must be finite and nonnegative")
  if args.snapshot_dir is not None and args.snapshot_dir.exists():
    parser.error("--snapshot-dir must be a new directory")
  if args.result_json is not None:
    if args.result_json.exists() or args.result_json.is_symlink():
      parser.error("--result-json must be a new file")
    if (
      args.snapshot_dir is not None
      and args.result_json.resolve() == (args.snapshot_dir / "state.json").resolve()
    ):
      parser.error("--result-json must differ from the snapshot's state.json")
  if args.headless and not args.run_task and args.duration == 0:
    args.duration = 2.0
  return args


def snapshot(simulation: ArmHandSimulation, directory: Path, report: dict) -> None:
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
    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
  )


def _execute_automatic(
  simulation,
  monitor,
  stop,
  viewer=None,
  *,
  precontact_noise_std_m=0.0,
  noise_seed=None,
  motion_profile="fast",
):
  # Idle inspection remains usable without importing the robot task executor.
  from kaihand_tactile_env.tasks.usb_insert.execution import UsbInsertionExecutor

  monitor.update()
  wall_start = time.monotonic()
  simulation_start = float(simulation.data.time)
  next_sync = wall_start
  previous_phase = None

  def should_stop():
    return stop.requested or (viewer is not None and not viewer.is_running())

  def observer(observed_simulation, phase):
    nonlocal next_sync, previous_phase
    # The executor owns every physics step. This observer only measures and
    # presents those states, including the monitor's continuous success dwell.
    monitor.update()
    if viewer is None or should_stop():
      return
    if phase != previous_phase:
      print(f"USB task phase: {phase}", flush=True)
      previous_phase = phase
    now = time.monotonic()
    if now >= next_sync:
      viewer.sync()
      next_sync = now + 1.0 / 60.0
    target_time = wall_start + float(observed_simulation.data.time) - simulation_start
    while not should_stop():
      delay = target_time - time.monotonic()
      if delay <= 0:
        break
      time.sleep(min(delay, 0.01))

  noise_options = {}
  if motion_profile != "fast":
    noise_options["motion_profile"] = motion_profile
  if precontact_noise_std_m != 0.0 or noise_seed is not None:
    # The executor owns seed generation and reports the actual realized noise.
    # Do not derive its seed from the independent initial-object pose seed.
    noise_options.update(
      {
        "precontact_noise_std_m": precontact_noise_std_m,
        "noise_seed": noise_seed,
      }
    )
  result = UsbInsertionExecutor(
    simulation, observer=observer, should_stop=should_stop, **noise_options
  ).execute()
  return result, monitor.update()


def _build_report(
  simulation, args, state, probes, forces, task_result, initialization_record=None
):
  probe_sample = probes.read(simulation.data)
  force_sample = forces.read(simulation.data)
  report = {
    "scene": simulation.scene,
    "mujoco_version": mujoco.__version__,
    "model_fingerprint": model_fingerprint(simulation.model_path),
    "seed": args.seed,
    "plug_initialization": (
      "pinky-forward-grasp"
      if args.plug_for_insertion
      else "mark-down"
      if args.plug_face_down
      else "scene-default"
    ),
    "noslip_iterations": int(simulation.model.opt.noslip_iterations),
    "physics_hz": 1.0 / simulation.timestep,
    "finite_state": bool(
      np.isfinite(simulation.data.qpos).all()
      and np.isfinite(simulation.data.qvel).all()
    ),
    "plug_pose_wxyz": simulation.object_pose("usb_plug").tolist(),
    "insertion": asdict(state),
    "probe_count": simulation.genesis_probe_layout.count,
    "tactile_contact_count": probe_sample.contact_count.tolist(),
    "fingertip_normal_force_n": force_sample.normal_force_n.tolist(),
    "tactile_targets": list(forces.target_geom_names),
    "task_scope": (
      "known-state robot grasp, alignment and mechanical insertion baseline"
      if args.run_task
      else "idle scene and mechanical insertion monitor"
    ),
  }
  if task_result is not None:
    report["task_result"] = asdict(task_result)
  if initialization_record is not None:
    report["initial_pose_randomization"] = initialization_record
  return report


def _publish_report(simulation, args, report, *, write_snapshot=True):
  content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
  if args.snapshot_dir is not None and write_snapshot:
    snapshot(simulation, args.snapshot_dir, report)
  if args.result_json is not None:
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwriting a file created after parse_args.
    with args.result_json.open("x", encoding="utf-8") as output:
      output.write(content)
  console_content = content
  noise = (report.get("task_result") or {}).get("precontact_noise") or {}
  if noise.get("commands"):
    # Clone the serialized report deeply: neither the caller's nested data nor
    # the full JSON/snapshot evidence should lose the realized control trace.
    console_report = json.loads(content)
    console_noise = console_report["task_result"]["precontact_noise"]
    commands = console_noise.pop("commands")
    console_noise["command_count"] = len(commands)
    console_noise["control_trace_in_console"] = False
    console_content = (
      json.dumps(console_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
  print(console_content, end="", flush=True)
  if not report["finite_state"]:
    raise SystemExit("USB scene produced a non-finite state")


def _reset_episode(simulation, args, stop=None):
  """Apply the selected pose after reset, including viewer-triggered resets."""
  simulation.reset(seed=args.seed)
  if stop is not None and stop.requested:
    return None
  if args.plug_face_down:
    from kaihand_tactile_env.tasks.usb_insert.setup import initialize_face_down

    initialize_face_down(simulation)
  elif args.plug_for_insertion:
    from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion

    return initialize_for_insertion(
      simulation,
      seed=args.seed,
      xy_jitter_m=args.xy_jitter_mm / 1000.0,
      yaw_jitter_rad=math.radians(args.yaw_jitter_deg),
    )
  return None


def _run(args: argparse.Namespace, stop: _StopRequest) -> None:
  simulation = ArmHandSimulation(scene="usb-insert")
  if stop.requested:
    return
  initialization_record = _reset_episode(simulation, args, stop)
  if stop.requested:
    return
  monitor = UsbInsertionMonitor(simulation)
  probes = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  forces = SolverDistributedTactileProvider(simulation.model)
  state = monitor.update()
  if stop.requested:
    return
  task_result = None
  if args.headless:
    if args.run_task:
      task_result, state = _execute_automatic(
        simulation,
        monitor,
        stop,
        precontact_noise_std_m=args.precontact_noise_mm / 1000.0,
        noise_seed=args.noise_seed,
        motion_profile=args.motion_profile,
      )
    else:
      while not stop.requested and simulation.data.time < args.duration:
        simulation.step()
        state = monitor.update()
  else:
    from mujoco import viewer as mjviewer

    with mjviewer.launch_passive(
      simulation.model, simulation.data, key_callback=stop.on_key
    ) as viewer:
      if stop.requested:
        return
      viewer.opt.geomgroup[4] = 0
      viewer.opt.geomgroup[5] = args.show_probes
      # Camera glyphs/axes are viewer decorations, not scene objects.
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
      viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
      viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
      if args.show_probes:
        ids = [
          i
          for i in range(simulation.model.ngeom)
          if simulation.model.geom(i).name.startswith("workcell_genesis_probe_")
        ]
        simulation.model.geom_rgba[ids] = (0.1, 0.5, 1.0, 0.8)
      if args.camera == "free":
        viewer.cam.lookat[:] = (0.38, -0.06, 0.92)
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -28.0
      else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = simulation.model.camera(args.camera).id
      print("USB-A scene: socket faces upward; insert downward along world -Z.")
      for camera_name in ("head", "overhead"):
        camera_id = simulation.model.camera(camera_name).id
        camera_position = simulation.data.cam_xpos[camera_id]
        print(
          f"Camera {camera_name}: world XYZ={np.round(camera_position, 3).tolist()} m; "
          f"select with --camera {camera_name}."
        )
      if args.run_task:
        print("Executing one automatic USB insertion; --duration does not limit it.")
      else:
        print("Idle inspection. Add --run-task to execute automatic insertion.")
      print("Exit: Ctrl+C in the terminal, Q/Esc in the window, or close the window.")
      if args.run_task:
        task_result, state = _execute_automatic(
          simulation,
          monitor,
          stop,
          viewer,
          precontact_noise_std_m=args.precontact_noise_mm / 1000.0,
          noise_seed=args.noise_seed,
          motion_profile=args.motion_profile,
        )
        report = _build_report(
          simulation, args, state, probes, forces, task_result, initialization_record
        )
        _publish_report(
          simulation,
          args,
          report,
          write_snapshot=not stop.requested and viewer.is_running(),
        )
        if not stop.requested and viewer.is_running():
          print("USB task finished. Final state is paused; close the window to exit.")
        while not stop.requested and viewer.is_running():
          viewer.sync()
          time.sleep(1.0 / 60.0)
        return
      wall_start = time.monotonic()
      next_sync = wall_start
      next_print = 0.0
      previous_insertion = None
      previous_time = float(simulation.data.time)
      while not stop.requested and viewer.is_running():
        if simulation.data.time < previous_time - simulation.timestep:
          initialization_record = _reset_episode(simulation, args, stop)
          monitor.reset()
          wall_start = time.monotonic()
          next_print = 0.0
        simulation.step()
        previous_time = float(simulation.data.time)
        state = monitor.update()
        if stop.requested:
          break
        insertion = (state.seated, state.success)
        if state.timestamp >= next_print or insertion != previous_insertion:
          force = forces.read(simulation.data)
          print(
            f"t={state.timestamp:.2f}s depth={state.insertion_depth_m * 1000:.2f}mm "
            f"socket_load={state.socket_normal_load_n:.3f}N "
            f"finger_load={float(force.normal_force_n.sum()):.3f}N "
            f"inserted={state.success}"
          )
          next_print = state.timestamp + 5.0
          previous_insertion = insertion
        now = time.monotonic()
        if now >= next_sync:
          viewer.sync()
          next_sync = now + 1.0 / 60.0
        if args.duration and state.timestamp >= args.duration:
          break
        delay = wall_start + state.timestamp - time.monotonic()
        if delay > 0:
          time.sleep(delay)
  if stop.requested and task_result is None:
    return
  report = _build_report(
    simulation, args, state, probes, forces, task_result, initialization_record
  )
  _publish_report(simulation, args, report, write_snapshot=not stop.requested)
  if args.headless and task_result is not None and not task_result.success:
    if not stop.requested:
      raise SystemExit(1)


def main() -> None:
  args = parse_args()
  with _stop_requests() as stop:
    _run(args, stop)
    if stop.requested:
      print("USB scene stopped.", flush=True)


if __name__ == "__main__":
  main()
