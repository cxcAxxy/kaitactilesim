#!/usr/bin/env python3
"""Inspect or run the independently named, socket-preengaged screw task.

No renderer or display is needed for --headless --run-task.  This entry point
never offers a torque-demo mode: the expert drives robot actuators only.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Prefer this checkout even when an older simulator is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kaihand_tactile_env.tasks.bulb_screw_inplace.execution import (  # noqa: E402
  BulbScrewInplaceExecutor,
)
from kaihand_tactile_env.tasks.bulb_screw_inplace.task import (  # noqa: E402
  BulbScrewInplaceMonitor,
  BulbScrewInplaceSimulation,
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--run-task", action="store_true")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--speed", choices=("fast", "normal"), default="fast")
  parser.add_argument("--max-strokes", type=int, default=20)
  parser.add_argument("--duration", type=float, default=2.0)
  parser.add_argument("--result-json", type=Path)
  args = parser.parse_args(argv)
  if args.seed < 0:
    parser.error("--seed must be nonnegative")
  if args.max_strokes < 1:
    parser.error("--max-strokes must be positive")
  if not math.isfinite(args.duration) or args.duration <= 0:
    parser.error("--duration must be finite and positive")
  if args.result_json is not None and args.result_json.exists():
    parser.error("--result-json must name a new file")
  return args


def run(args):
  simulation = BulbScrewInplaceSimulation(position_seed=args.seed)
  stopped = False

  def stop(*_):
    nonlocal stopped
    stopped = True

  previous_handler = signal.signal(signal.SIGINT, stop)
  task_result = None
  monitor = BulbScrewInplaceMonitor(simulation)
  state = monitor.measure()

  def execute(viewer=None):
    nonlocal state, task_result
    if args.run_task:
      last_sync = time.monotonic()
      last_time = float(simulation.data.time)

      def observe(sim, _phase):
        nonlocal last_sync, last_time
        if viewer is not None and sim.data.time - last_time >= 1 / 60:
          time.sleep(
            max(0, sim.data.time - last_time - (time.monotonic() - last_sync))
          )
          viewer.sync()
          last_sync, last_time = time.monotonic(), float(sim.data.time)

      task_result = BulbScrewInplaceExecutor(
        simulation,
        speed=args.speed,
        max_strokes=args.max_strokes,
        observer=observe,
        should_stop=lambda: stopped or (viewer is not None and not viewer.is_running()),
      ).execute()
      state = task_result.state
    else:
      start = float(simulation.data.time)
      while not stopped and simulation.data.time - start < args.duration:
        if viewer is not None and not viewer.is_running():
          break
        simulation.step()
        state = monitor.update()
        if viewer is not None:
          viewer.sync()
          time.sleep(simulation.timestep)

  try:
    if args.headless:
      execute()
    else:
      import mujoco
      from mujoco import viewer as mjviewer

      with mjviewer.launch_passive(simulation.model, simulation.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = simulation.model.camera("bulb_closeup").id
        execute(viewer)
  finally:
    signal.signal(signal.SIGINT, previous_handler)
  report = {
    "scene": "bulb-screw-inplace",
    "initialization_mode": "preengaged_at_socket_entrance",
    "seed": args.seed,
    "initialization": simulation.initialization,
    "external_object_wrench": False,
    "object_pose_control": False,
    "task_result": asdict(task_result) if task_result is not None else None,
    "state": asdict(state),
    "interrupted": stopped,
  }
  serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
  if args.result_json is not None:
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("x", encoding="utf-8") as stream:
      stream.write(serialized)
  print(serialized, end="")
  return 130 if stopped else 1 if task_result is not None and not task_result.success else 0


def main():
  raise SystemExit(run(parse_args()))


if __name__ == "__main__":
  main()
