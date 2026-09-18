#!/usr/bin/env python3
"""Render workcell RGB/depth/segmentation snapshots for visual acceptance."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES, CameraConfig
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.pick_place.task import (
  GraspExecutor,
  KnownStateGraspPlanner,
)
from kaihand_tactile_env.tasks.poker_draw.task import (
  PokerDrawExecutor,
  PokerDrawPlanner,
)
from PIL import Image


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output-dir", type=Path, help="Defaults to artifacts/<task>/render."
  )
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument(
    "--scene",
    choices=("pick-place", "poker-draw"),
    default="pick-place",
  )
  parser.add_argument("--grasp", choices=("none", "cylinder"), default="none")
  parser.add_argument(
    "--draw-card",
    action="store_true",
    help="Execute poker-draw before capturing the selected scene.",
  )
  args = parser.parse_args()
  if args.grasp != "none" and args.scene != "pick-place":
    parser.error("--grasp is only available in --scene pick-place")
  if args.draw_card and args.scene != "poker-draw":
    parser.error("--draw-card requires --scene poker-draw")
  if args.output_dir is None:
    args.output_dir = Path("artifacts") / args.scene.replace("-", "_") / "render"
  args.output_dir.mkdir(parents=True, exist_ok=True)
  simulation = ArmHandSimulation(scene=args.scene)
  if args.grasp != "none":
    side = "right"
    plan = KnownStateGraspPlanner(simulation).plan(args.grasp, side)
    GraspExecutor(simulation).execute(plan)
  elif args.draw_card:
    plan = PokerDrawPlanner(simulation).plan("right")
    PokerDrawExecutor(simulation).execute(plan)
  cameras = tuple(
    CameraConfig(name, args.width, args.height)
    for name in SHARED_CAMERA_NAMES
  )
  with WorkcellRenderer(simulation.model, cameras) as renderer:
    for camera in cameras:
      capture = renderer.capture(simulation.data, camera)
      Image.fromarray(capture["rgb"]).save(args.output_dir / f"{camera.name}_rgb.png")
      Image.fromarray(_colorize_depth(capture["depth"])).save(
        args.output_dir / f"{camera.name}_depth.png"
      )
      Image.fromarray(_colorize_segmentation(capture["segmentation"])).save(
        args.output_dir / f"{camera.name}_segmentation.png"
      )
  print(args.output_dir.resolve())


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
  finite = np.isfinite(depth)
  output = np.zeros((*depth.shape, 3), dtype=np.uint8)
  if not finite.any():
    return output
  low, high = np.percentile(depth[finite], (2.0, 98.0))
  normalized = np.clip((depth - low) / max(high - low, 1.0e-6), 0.0, 1.0)
  output[..., 0] = (255.0 * normalized).astype(np.uint8)
  output[..., 1] = (255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
  output[..., 2] = (255.0 * (1.0 - normalized)).astype(np.uint8)
  return output


def _colorize_segmentation(segmentation: np.ndarray) -> np.ndarray:
  object_type = segmentation[..., 0].astype(np.int64)
  object_id = segmentation[..., 1].astype(np.int64)
  valid = np.logical_and(object_type >= 0, object_id >= 0)
  code = (object_type + 1) * 1009 + (object_id + 1)
  color = np.zeros((*object_id.shape, 3), dtype=np.uint8)
  color[..., 0] = np.mod(53 * code + 97, 255)
  color[..., 1] = np.mod(97 * code + 53, 255)
  color[..., 2] = np.mod(193 * code + 17, 255)
  color[~valid] = 0
  return color


if __name__ == "__main__":
  main()
