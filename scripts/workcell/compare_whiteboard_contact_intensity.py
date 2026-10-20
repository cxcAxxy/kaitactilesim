#!/usr/bin/env python3
"""使用已有白板评估日志生成中文接触强度与清洁效果预览。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.whiteboard_contact_intensity import (
  export_contact_intensity,
)
from kaihand_tactile_env.shared.whiteboard_contact_reference import (
  DEFAULT_DEMO_ROOT,
  load_historical_whiteboard_contact_trial,
  load_whiteboard_contact_reference,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--trial-dir", required=True, type=Path, help="已有单次评估的目录"
  )
  parser.add_argument(
    "--output-dir", type=Path, help="默认输出到评估目录；已有接触产物不会覆盖"
  )
  parser.add_argument("--reference-hdf5", type=Path, help="指定成功专家轨迹的 Raw HDF5")
  parser.add_argument(
    "--demo-root", type=Path, default=DEFAULT_DEMO_ROOT, help="专家示例目录"
  )
  parser.add_argument(
    "--preview", action="store_true", help="标记为历史日志补算预览（不会重新运行模型）"
  )
  args = parser.parse_args()
  output = args.output_dir or args.trial_dir
  for filename in (
    "contact_intensity.json",
    "contact_intensity.npz",
    "contact_intensity.png",
    "contact_intensity_fingers.png",
    "contact_intensity_phases.png",
  ):
    if (output / filename).exists():
      parser.error(f"已有产物 {output / filename}；请用 --output-dir 指定新目录")
  rollout = load_historical_whiteboard_contact_trial(args.trial_dir)
  reference = load_whiteboard_contact_reference(
    task=rollout.metadata["task"],
    reference_hdf5=args.reference_hdf5,
    demo_root=args.demo_root,
  )
  result = export_contact_intensity(output, rollout, reference=reference)
  print(
    json.dumps(
      {
        "artifacts": result.get("artifacts"),
        "mode": "historical_control_step_preview",
        "original_artifacts_modified": False,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
