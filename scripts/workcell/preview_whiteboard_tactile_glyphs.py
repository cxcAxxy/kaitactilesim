#!/usr/bin/env python3
"""从已有白板评估生成中文整体触觉变化示意图和视频，不重跑仿真。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.whiteboard_tactile_glyphs import export_tactile_glyphs
from kaihand_tactile_env.shared.whiteboard_tactile_reference import (
  DEFAULT_DEMO_ROOT,
  choose_snapshot_times,
  load_historical_whiteboard_tactile_trial,
  load_whiteboard_tactile_reference,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--trial-dir", required=True, type=Path, help="已有单次白板评估目录"
  )
  parser.add_argument(
    "--output-dir", type=Path, help="默认 trial 根目录；同名产物不覆盖"
  )
  parser.add_argument(
    "--demo-root", type=Path, default=DEFAULT_DEMO_ROOT, help="task_demo 专家目录"
  )
  parser.add_argument(
    "--reference-hdf5", type=Path, help="显式指定同任务成功专家的原始 HDF5"
  )
  parser.add_argument(
    "--snapshot-times-s",
    nargs=3,
    type=float,
    help="静态图中的三个专家仿真时间，默认按专家阶段选择",
  )
  parser.add_argument(
    "--window-s", type=float, default=0.1, help="力变化的固定仿真时间窗口，默认 0.1 秒"
  )
  parser.add_argument(
    "--frame-interval-s",
    type=float,
    default=0.1,
    help="视频显示帧间隔，默认 0.1 仿真秒；不提高原始采样率",
  )
  parser.add_argument(
    "--contact-threshold-n",
    type=float,
    default=0.05,
    help="每指接触统计门限，默认 0.05 N",
  )
  parser.add_argument(
    "--no-video", action="store_true", help="只生成静态图和 JSON/NPZ，不生成视频"
  )
  args = parser.parse_args()
  for label, value, positive in (
    ("window-s", args.window_s, True),
    ("frame-interval-s", args.frame_interval_s, True),
    ("contact-threshold-n", args.contact_threshold_n, False),
  ):
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
      parser.error(f"--{label} 必须有限且{'大于零' if positive else '非负'}")
  output = args.output_dir or args.trial_dir
  existing = [
    str(output / f"tactile_overall_change.{suffix}")
    for suffix in ("png", "mp4", "json", "npz")
    if (output / f"tactile_overall_change.{suffix}").exists()
  ]
  if existing:
    parser.error(f"已有产物不会覆盖：{existing}；请指定新的 --output-dir")
  summary = json.loads((args.trial_dir / "summary.json").read_text(encoding="utf-8"))
  reference = load_whiteboard_tactile_reference(
    task=summary["task"], reference_hdf5=args.reference_hdf5, demo_root=args.demo_root
  )
  rollout = load_historical_whiteboard_tactile_trial(args.trial_dir, reference)
  snapshots = args.snapshot_times_s or choose_snapshot_times(rollout, reference)
  result = export_tactile_glyphs(
    rollout,
    reference,
    output,
    snapshot_times_s=snapshots,
    window_s=args.window_s,
    contact_threshold_n=args.contact_threshold_n,
    frame_interval_s=args.frame_interval_s,
    include_video=not args.no_video,
    video_fps=1.0 / args.frame_interval_s,
  )
  print(json.dumps(result["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
