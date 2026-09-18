#!/usr/bin/env python3
"""Plan and execute one simulator-truth grasp."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.pick_place.task import (
  GraspExecutor,
  KnownStateGraspPlanner,
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--object", choices=("cylinder",), default="cylinder")
  parser.add_argument("--side", choices=("left", "right"))
  parser.add_argument("--viewer", choices=("none", "native"), default="native")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  side = args.side or "right"
  simulation = ArmHandSimulation()
  simulation.reset(seed=args.seed)
  plan = KnownStateGraspPlanner(simulation).plan(args.object, side)

  viewer = None
  if args.viewer == "native":
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)

  def observe(_: ArmHandSimulation, phase: str) -> None:
    del phase
    if viewer is not None and viewer.is_running():
      viewer.sync()

  try:
    result = GraspExecutor(simulation, observer=observe).execute(plan)
  finally:
    if viewer is not None:
      viewer.close()
  print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
