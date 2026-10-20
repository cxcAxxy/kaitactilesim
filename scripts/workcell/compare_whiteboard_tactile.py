#!/usr/bin/env python3
"""按 task_demo 专家原始时间线生成中文指尖触觉接触对比；不重跑仿真。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.whiteboard_tactile_comparison import (
  export_tactile_comparison,
)
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
    "--output-dir", type=Path, help="默认评估目录；已有同名报告拒绝覆盖"
  )
  parser.add_argument(
    "--demo-root", type=Path, default=DEFAULT_DEMO_ROOT, help="task_demo 专家示例目录"
  )
  parser.add_argument(
    "--reference-hdf5", type=Path, help="显式指定同任务的成功专家原始 HDF5"
  )
  parser.add_argument(
    "--contact-threshold-n",
    type=float,
    default=0.05,
    help="每指接触判定门限，默认 0.05 N",
  )
  parser.add_argument(
    "--snapshot-times-s",
    nargs=3,
    type=float,
    help="三个专家仿真时间快照；默认根据专家接触阶段选择",
  )
  args = parser.parse_args()
  output = args.output_dir or args.trial_dir
  filenames = (
    "tactile_contact_comparison.json",
    "tactile_contact_comparison.npz",
    "tactile_contact_timeline.png",
    "tactile_contact_centers.png",
    "tactile_contact_heatmaps.png",
  )
  existing = [str(output / name) for name in filenames if (output / name).exists()]
  if existing:
    parser.error(f"已有报告不会覆盖：{existing}；请指定新的 --output-dir")
  task = json.loads((args.trial_dir / "summary.json").read_text(encoding="utf-8"))[
    "task"
  ]
  reference = load_whiteboard_tactile_reference(
    task=task, reference_hdf5=args.reference_hdf5, demo_root=args.demo_root
  )
  rollout = load_historical_whiteboard_tactile_trial(args.trial_dir, reference)
  reference.metadata["snapshot_times_s"] = (
    args.snapshot_times_s or choose_snapshot_times(rollout, reference)
  )
  result = export_tactile_comparison(
    rollout, reference, output, contact_threshold_n=args.contact_threshold_n
  )
  print(json.dumps(result.get("artifacts", result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
