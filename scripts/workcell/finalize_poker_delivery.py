#!/usr/bin/env python3
"""Verify cross-format membership and write the completed delivery receipt."""

import json
from pathlib import Path

from poker_delivery_common import save


def read(path):
  return json.loads(path.read_text(encoding="utf-8"))


def main():
  parent = Path("datasets").resolve()
  raw = parent / "poker_draw_0909"
  processing = raw / "processing/dual_format_20260909"
  touch = parent / "poker_draw_0909_egotouch"
  steer = parent / "poker_draw_0909_egosteer"
  snapshot = read(processing / "source_selection.json")
  assert snapshot == read(touch / "SOURCE_SELECTION.json") == read(steer / "SOURCE_SELECTION.json")
  assert len(snapshot["episodes"]) == 50
  touch_audit = read(touch / "dataset_audit.json")
  touch_validation = read(touch / "package_validation.json")
  steer_audit = read(steer / "source_audit.json")
  steer_validation = read(steer / "validation.json")
  assert touch_audit["passed"] and touch_audit["all_source_audits_verified"]
  assert touch_validation["valid"] and steer_audit["valid"] and steer_validation["valid"]
  assert len(touch_audit["sessions"]) == len(steer_audit["episodes"]) == steer_validation["episodes"] == 50
  touch_splits = read(touch / "split_manifest.json")["splits"]
  steer_manifest = read(steer / "dataset_manifest.json")
  for split, tict_split in (("train", "train"), ("val", "validation")):
    rows = [row for row in snapshot["episodes"] if row["split"] == split]
    assert set(touch_splits[tict_split]) == {row["session_id"] for row in rows}
    assert set(steer_manifest["splits"][split]["episode_indices"]) == {row["episode_index"] for row in rows}
  archives = []
  for root in (touch, steer):
    path = root.with_name(root.name + ".tar.gz")
    receipt = read(path.with_name(path.name + ".verification.json"))
    assert receipt["valid"] and receipt["size_bytes"] == path.stat().st_size
    assert path.with_name(path.name + ".sha256").read_text().split()[0] == receipt["sha256"]
    archives.append({key: receipt[key] for key in ("archive", "sha256", "size_bytes", "file_count")})
  result = {"valid": True, "episodes": 50, "split": {"train": 43, "validation": 7},
            "same_frozen_source_selection": True, "same_whole_episode_split": True,
            "egotouch_frames": sum(item["frame_count"] for item in touch_audit["sessions"]),
            "egotouch_h50_windows": sum(item["window_count"] for item in touch_audit["sessions"]),
            "egosteer_samples": steer_validation["samples"], "archives": archives,
            "raw_modified": False, "upstream_training_executed": False,
            "cleaning": "50 offline first4 checked; no hard errors; review_required velocity warnings retained"}
  save(processing / "delivery_summary.json", result)
  content = f"""# 50条成功摸牌数据：双格式交付

已完成同一批50条成功轨迹的两种转换和压缩；原始HDF5未修改，失败轨迹未入包。
整条episode划分：43条train、7条validation，两种格式的划分完全一致。
逐条源文件SHA和split见`processing/dual_format_20260909/source_selection.json`。

| 格式 | 目录 | 压缩包 | 大小 |
| --- | --- | --- | --- |
| EgoTouch | [poker_draw_0909_egotouch](../poker_draw_0909_egotouch/) | [tar.gz](../poker_draw_0909_egotouch.tar.gz) | {archives[0]['size_bytes'] / 1024**3:.3f} GiB |
| EgoSteer | [poker_draw_0909_egosteer](../poker_draw_0909_egosteer/) | [tar.gz](../poker_draw_0909_egosteer.tar.gz) | {archives[1]['size_bytes'] / 1024**3:.3f} GiB |

EgoTouch按test.pdf中数据字段要求导出，忽略外机路径、Part11。共{result['egotouch_frames']:,}帧
head RGB/JSON，{result['egotouch_h50_windows']:,}个H50有效窗口；完整腕/指尖SE3、三通道触觉NPZ。
归一化统计只用train。搬到新路径、解压后先在该包目录执行`python rebase_paths.py`。
本地manifest容器的外部DataLoader接通情况见包内DATA_CONTRACT，不声称远端训练已运行。

EgoSteer按本机`egosteer`项目的116D、head-only WebDataset要求导出，共
{result['egosteer_samples']:,}个样本；触觉额外保存在`tactile_sidecars/`，不修改标准48D动作。
记录实际render/FK时刻、采集时刻、触觉时刻及终帧mask。训练前仍须在目标项目配置
`motion_type=fingertips`及数据路径，并仅用train建立归一化统计；未运行外部训练。

## 验证与质量边界

- 50条均补齐Cleaning前四项离线检查，无硬错误；第4项接触前快速运动仍为
  `review_required`，不等同无条件人工通过。第2项不是全机器人动力学重放认证。
- EgoTouch全部逐条源审计和整包校验通过；EgoSteer全量格式校验、每帧RGB/位姿/
  下一状态和触觉taxel对源HDF5核验通过。没有平滑、补帧或裁剪力值。
- 两个压缩包均逐文件校验解压内容的大小和SHA256，完整性记录为同名
  `.tar.gz.verification.json`；传输后在压缩包所在目录执行：

```bash
sha256sum -c poker_draw_0909_egotouch.tar.gz.sha256
sha256sum -c poker_draw_0909_egosteer.tar.gz.sha256
```

汇总：[delivery_summary.json](processing/dual_format_20260909/delivery_summary.json)。
本次只做串行离线转换、校验和单worker gzip压缩，没有启动仿真或新增视频。
"""
  with (raw / "DELIVERY.md").open("x", encoding="utf-8") as stream:
    stream.write(content)
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
