"""Read-only quality audit for the real aloha_mobile_dummy HDF5 + MP4 dataset.

Uses h5py/numpy and ffmpeg/ffprobe. Writes only a new, external report directory.
Indices are zero-based; transition index is the destination frame. Flags are
review candidates, NOT deletion decisions or certified hardware limit violations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
TRAJECTORIES = ("observations/qpos", "action", "action_actual_shift")
ARM = list(range(7)) + list(range(13, 20))
HAND = list(range(7, 13)) + list(range(20, 26))


def plain(value):
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  if isinstance(value, np.ndarray):
    return plain(value.tolist())
  if isinstance(value, np.generic):
    return plain(value.item())
  if isinstance(value, (list, tuple)):
    return [plain(v) for v in value]
  if isinstance(value, dict):
    return {str(k): plain(v) for k, v in value.items()}
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def save(path, value):
  with path.open("x", encoding="utf-8") as stream:
    json.dump(plain(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")


def indices(mask):
  return np.flatnonzero(mask).tolist()


def clock_check(times, period):
  times = np.asarray(times)
  if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
    return {"valid": False, "error": "empty/nonfinite/nonvector clock"}
  delta = np.diff(times)
  gaps = indices(delta > 1.5 * period)
  return {
    "valid": True, "samples": len(times),
    "duplicate_transitions": (np.flatnonzero(delta == 0) + 1).tolist(),
    "reversed_transitions": (np.flatnonzero(delta < 0) + 1).tolist(),
    "gap_transitions": [i + 1 for i in gaps],
    "estimated_missing_samples": int(sum(max(0, int(round(delta[i] / period)) - 1) for i in gaps)),
    "interval_min": float(delta.min()) if len(delta) else None,
    "interval_median": float(np.median(delta)) if len(delta) else None,
    "interval_max": float(delta.max()) if len(delta) else None,
  }


def duplicate_runs(hashes):
  runs = []
  start = 0
  for end in range(1, len(hashes) + 1):
    if end == len(hashes) or hashes[end] != hashes[start]:
      if end - start > 1:
        runs.append({"start": start, "end": end - 1, "length": end - start})
      start = end
  return runs


def trajectory_check(values, stamps, arm_step, hand_step):
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 2 or values.shape[1] != 26 or len(values) != len(stamps):
    return {"valid": False, "shape": list(values.shape), "error": "expected (timestamp_count,26)"}
  finite = np.isfinite(values)
  delta = np.diff(values, axis=0)
  magnitude = np.abs(delta)
  # Do not unwrap angles: no joint topology/continuous-joint metadata is saved.
  absolute = np.full(26, arm_step)
  absolute[HAND] = hand_step
  thresholds = absolute.copy()
  for j in range(26):
    good = magnitude[:, j][np.isfinite(magnitude[:, j])]
    if len(good):
      median = np.median(good)
      thresholds[j] = max(absolute[j], median + 8 * 1.4826 * np.median(np.abs(good - median)))
  suspect = np.isfinite(magnitude) & (magnitude > thresholds)
  dt = np.diff(stamps).astype(float) / 1e9
  events = []
  for i, j in np.argwhere(suspect):
    returns = i + 1 < len(delta) and np.isfinite(delta[i + 1, j]) and (
      delta[i, j] * delta[i + 1, j] < 0
      and abs(values[i + 2, j] - values[i, j]) < 0.25 * magnitude[i, j]
    )
    events.append({"frame": int(i + 1), "dimension": int(j),
                   "group": "hand_normalized" if j in HAND else "arm_raw_position",
                   "before": values[i, j], "after": values[i + 1, j],
                   "step": magnitude[i, j], "threshold": thresholds[j],
                   "dt_seconds": dt[i],
                   "speed_raw_per_second": magnitude[i, j] / dt[i] if dt[i] > 0 else None,
                   "one_frame_return_spike": bool(returns)})
  return plain({"valid": True, "nonfinite_cells": np.argwhere(~finite).tolist(),
                "max_step_per_dimension": [float(v[np.isfinite(v)].max()) if np.isfinite(v).any() else None for v in magnitude.T],
                "threshold_per_dimension": thresholds,
                "jump_events": events,
                "identical_row_transitions": (np.flatnonzero(np.all(delta == 0, axis=1)) + 1).tolist(),
                "constant_dimensions": [j for j in range(26) if len(values) and finite[:, j].all() and np.all(values[:, j] == values[0, j])],
                "hand_out_of_0_1_cells": [[int(i), HAND[int(j)]] for i, j in np.argwhere(
                  finite[:, HAND] & ((values[:, HAND] < -1e-6) | (values[:, HAND] > 1 + 1e-6))) ]})


def relation(file, first, second, shift=0):
  if first not in file or second not in file:
    return {"available": False}
  x, y = file[first][:], file[second][:]
  if shift:
    x, y = x[:-shift], y[shift:]
  return {"available": True, "equal": bool(np.array_equal(x, y)),
          "max_abs_diff": float(np.max(np.abs(x - y))) if x.shape == y.shape and x.size else None}


def audit_h5(path, args):
  result = {"episode": path.stem, "path": str(path), "errors": []}
  with h5py.File(path, "r") as file:
    attrs = plain(dict(file.attrs))
    result["attributes"] = attrs
    schema = {}
    def visit(name, obj):
      schema[name] = {"attributes": plain(dict(obj.attrs))}
      if isinstance(obj, h5py.Dataset):
        schema[name].update(shape=list(obj.shape), dtype=str(obj.dtype))
    file.visititems(visit)
    result["schema"] = schema
    result["text_fields"] = {name: plain(file[name][()]) for name in schema
                             if isinstance(file[name], h5py.Dataset) and file[name].dtype.kind in "SUO"}
    stamps = file["timestamp_ros_nsec"][:]
    result["frames"] = len(stamps)
    fps = float(attrs.get("fps", 30))
    result["clock_nsec"] = clock_check(stamps, 1e9 / fps)
    result["aligned_length_mismatches"] = [name for name, info in schema.items()
      if "shape" in info and info["shape"] and not name.startswith("raw/") and info["shape"][0] != len(stamps)]
    result["nonfinite_dataset_counts"] = {}
    for name, info in schema.items():
      if "dtype" in info and file[name].dtype.kind in "fc":
        count = int(np.count_nonzero(~np.isfinite(file[name][:])))
        if count:
          result["nonfinite_dataset_counts"][name] = count
    result["relations"] = {
      "action_equals_master_action": relation(file, "action", "master_action"),
      "qpos_equals_slave_state": relation(file, "observations/qpos", "slave_state"),
      "action_equals_qpos": relation(file, "action", "observations/qpos"),
      "actual_shift_equals_next_qpos_except_last": relation(file, "action_actual_shift", "observations/qpos", 1),
    }
    if "action_actual_shift" in file and len(stamps):
      result["actual_shift_last_equals_last_qpos"] = bool(np.array_equal(file["action_actual_shift"][-1], file["observations/qpos"][-1]))
    known_layout = attrs.get("qpos_layout") == "interleaved_LhRh" and attrs.get("arm_dof") == 14 and attrs.get("hand_dof_per_side") == 6
    result["trajectory"] = {name: trajectory_check(file[name][:], stamps, args.arm_step, args.hand_step)
                            for name in TRAJECTORIES if name in file and known_layout}
    if not known_layout:
      result["errors"].append("Unknown joint layout: group-specific trajectory checks skipped")
    result["tactile"] = {}
    for side in ("left", "right"):
      name = f"raw/revo2_touch/{side}"
      if name in file:
        ts = file[name + "/stamp_nsec"][:]
        period = 1e9 / float(file[name].attrs.get("rate_hz", 50))
        result["tactile"][side] = {"clock_nsec": clock_check(ts, period),
          "sample_count_matches": len(ts) == len(file[name + "/data"]),
          "start_offset_ms": float((int(ts[0]) - int(stamps[0])) / 1e6) if len(ts) and len(stamps) else None,
          "end_offset_ms": float((int(ts[-1]) - int(stamps[-1])) / 1e6) if len(ts) and len(stamps) else None}
    result["lowdim_sha256"] = {name: hashlib.sha256(file[name][:].tobytes()).hexdigest() for name in ("action", "observations/qpos") if name in file}
  return result


def run(command, timeout=180):
  proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
  if proc.returncode:
    raise RuntimeError(proc.stderr[-2000:] or f"exit={proc.returncode}")
  return proc


def audit_video(path, expected_frames, fps):
  result = {"path": str(path), "expected_frames": expected_frames}
  try:
    probe = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
      "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate:packet=pts_time",
      "-of", "json", str(path)]).stdout)
    result["stream"] = probe["streams"][0]
    # Presentation order, not packet decode order (B-frames can reorder packets).
    pts = sorted(float(p["pts_time"]) for p in probe.get("packets", []) if "pts_time" in p)
    result["presentation_clock_seconds"] = clock_check(pts, 1 / fps)
    result["packets_without_pts"] = sum("pts_time" not in p for p in probe.get("packets", []))
    decoded = run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
      "-map", "0:v:0", "-an", "-vsync", "0", "-threads", "1", "-f", "framemd5", "-"])
    hashes = [line.rsplit(",", 1)[-1].strip() for line in decoded.stdout.splitlines() if line and not line.startswith("#")]
    result.update(decoded_frames=len(hashes), frame_count_matches=len(hashes) == expected_frames,
                  decode_errors=decoded.stderr.strip(), duplicate_runs=duplicate_runs(hashes),
                  decoded_sequence_sha256=hashlib.sha256("\n".join(hashes).encode()).hexdigest())
    result["adjacent_duplicate_frames"] = sum(r["length"] - 1 for r in result["duplicate_runs"])
  except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
  return result


def summarize(episodes):
  videos = [v for e in episodes for v in e.get("videos", {}).values()]
  tasks = Counter(str(e.get("attributes", {}).get("task")) for e in episodes)
  def groups(key):
    seen = {}
    for e in episodes:
      digest = e.get("lowdim_sha256", {}).get(key)
      if digest:
        seen.setdefault(digest, []).append(e["episode"])
    return [v for v in seen.values() if len(v) > 1]
  return {
    "episodes": len(episodes), "frames": sum(e.get("frames", 0) for e in episodes),
    "videos_checked": len(videos), "task_counts": dict(tasks),
    "h5_errors": [e["episode"] for e in episodes if e.get("errors")],
    "video_errors": [v["path"] for v in videos if v.get("error") or v.get("decode_errors")],
    "video_frame_count_mismatches": [v["path"] for v in videos if v.get("frame_count_matches") is False],
    "adjacent_duplicate_video_frames": sum(v.get("adjacent_duplicate_frames", 0) for v in videos),
    "videos_with_adjacent_duplicates": sum(bool(v.get("duplicate_runs")) for v in videos),
    "video_estimated_missing_pts": sum(v.get("presentation_clock_seconds", {}).get("estimated_missing_samples", 0) for v in videos),
    "main_clock_duplicate_transitions": sum(len(e.get("clock_nsec", {}).get("duplicate_transitions", [])) for e in episodes),
    "main_clock_reversed_transitions": sum(len(e.get("clock_nsec", {}).get("reversed_transitions", [])) for e in episodes),
    "main_clock_estimated_missing_samples": sum(e.get("clock_nsec", {}).get("estimated_missing_samples", 0) for e in episodes),
    "nonfinite_episodes": [e["episode"] for e in episodes if e.get("nonfinite_dataset_counts")],
    "jump_events": {name: sum(len(e.get("trajectory", {}).get(name, {}).get("jump_events", [])) for e in episodes) for name in TRAJECTORIES},
    "episodes_with_jumps": {name: [e["episode"] for e in episodes if e.get("trajectory", {}).get(name, {}).get("jump_events")] for name in TRAJECTORIES},
    "relation_equal_episode_counts": {key: sum(e.get("relations", {}).get(key, {}).get("equal", False) for e in episodes)
      for key in ("action_equals_master_action", "qpos_equals_slave_state", "action_equals_qpos", "actual_shift_equals_next_qpos_except_last")},
    "duplicate_lowdim_episodes": {key: groups(key) for key in ("action", "observations/qpos")},
  }


def write_semantics(output, episodes):
  """Summarize observed evidence, not semantics borrowed from sim exporters."""
  attrs = [e.get("attributes", {}) for e in episodes]
  fields = sorted({name for e in episodes for name in e.get("schema", {})})
  summary = summarize(episodes)
  relations = summary["relation_equal_episode_counts"]
  sources = dict(Counter(str(a.get("action_source")) for a in attrs))
  qpos_sources = dict(Counter(str((a.get("qpos_arm_source"), a.get("qpos_hand_source"))) for a in attrs))
  subtask_fields = [name for name in fields if any(s in name.lower() for s in ("subtask", "stage", "instruction", "language", "segment"))]
  tip_fields = [name for name in fields if any(s in name.lower() for s in ("fingertip", "finger_tip", "world_from", "eef", "pose", "extrinsic", "intrinsic"))]
  shifted_terminal = sum(e.get("actual_shift_last_equals_last_qpos", False) for e in episodes)
  command_coverage = []
  for episode in episodes:
    raw_stats = episode.get("attributes", {}).get("stream_stats_json", "{}")
    try:
      stats = json.loads(raw_stats)
      matched = stats.get("arm_cmd_matched_frames")
      total = episode.get("frames", 0)
      if matched is not None and matched != total:
        command_coverage.append({"episode": episode["episode"], "frames": total,
                                 "arm_cmd_matched_frames": matched, "unmatched_frames": total - matched})
    except (TypeError, ValueError):
      command_coverage.append({"episode": episode["episode"], "error": "invalid stream_stats_json"})
  state_jumps = [(e["episode"], event) for e in episodes
                 for event in e.get("trajectory", {}).get("observations/qpos", {}).get("jump_events", [])]
  evidence = {
    "action_source_counts": sources, "qpos_source_counts": qpos_sources,
    "task_counts": summary["task_counts"], "subtask_candidate_fields": subtask_fields,
    "fingertip_pose_or_calibration_candidate_fields": tip_fields,
    "relation_equal_episode_counts": relations, "actual_shift_terminal_repeat_count": shifted_terminal,
    "command_matching_coverage_warnings": command_coverage,
    "dataset_fields": fields,
  }
  save(output / "semantics.json", evidence)
  lines = ["# 字段语义与清洗结论", "",
    "## 动作空间与监督来源", "",
    f"- action_source 统计：`{sources}`。这是命令来源的关节控制量，不是 EEF 位姿。",
    "- 本批保存 26 维 action/qpos；属性 arm_dof=14、hand_dof_per_side=6、hand_normalized=True。",
    "- 依 qpos_layout=interleaved_LhRh 解释：左臂 7 + 左手 6 + 右臂 7 + 右手 6。手部六维不是五根手指的 XYZ。",
    f"- qpos 实测来源统计：`{qpos_sources}`。",
    f"- action 与 master_action 完全相等：{relations['action_equals_master_action']}/{len(episodes)} 条。",
    f"- qpos 与 slave_state 完全相等：{relations['qpos_equals_slave_state']}/{len(episodes)} 条。",
    f"- action 与同帧 qpos 完全相等：{relations['action_equals_qpos']}/{len(episodes)} 条。",
    f"- action_actual_shift[t] == qpos[t+1]（除最后帧）：{relations['actual_shift_equals_next_qpos_except_last']}/{len(episodes)} 条；末帧重复 qpos 末帧：{shifted_terminal}/{len(episodes)} 条。",
    "- 学习下发控制命令时应显式读 action；若学习下一帧状态，可显式选 action_actual_shift，但它改变了学习目标，不能混用。",
    "- command 并不证明机器人精确执行了该值。逐关节名称、确切单位、command/state 原始异步时间戳未保存，不能凭相等关系推断控制延迟。",
    f"- 采集器记录的 arm command 匹配覆盖异常：`{command_coverage}`。这些帧的具体填充/回退策略需要采集端源码确认；文件没有逐帧匹配掩码，不宜宣称每一帧都是成功匹配的 command。",
    "", "## 文本与坐标系", "",
    f"- task 属性分布：`{summary['task_counts']}`。`pick up the cup` 是整段任务级文本，不是分段子任务描述，而且与扑克任务不符。",
    f"- 子任务/阶段/语言分段候选字段：`{subtask_fields}`。没有分段时间边界就无法恢复可靠子任务标签。",
    f"- 指尖位姿/EEF/坐标变换/标定候选字段：`{tip_fields}`。",
    "- 原始数据未存指尖 XYZ/pose；raw/revo2_touch 是每指的触觉通道，不是坐标。因此手腕/世界/头部坐标系三者均不能选定。",
    "- 若要生成指尖坐标，需要与实机一致的运动学、关节映射和标定。手腕系可由手部 FK 定义；世界系还需腕到世界变换；头部相机系还需相机外参。仿真仓库中的 world_from_fingertip 不代表这批真机文件也有该字段。",
    "", "## 需人工复核的 state 候选", "",
    "帧号/维度均从 0 开始；dt 过长时位置增量会变大，不能直接判定电机瞬跳。", "",
    "| Episode | 帧 | 维度 | 位置增量 | dt (ms) | 原始单位/秒 |",
    "|---|---:|---:|---:|---:|---:|"]
  for episode, event in state_jumps:
    speed = event["speed_raw_per_second"]
    lines.append(f"| {episode} | {event['frame']} | {event['dimension']} | {event['step']:.6f} | {event['dt_seconds'] * 1000:.3f} | {speed if speed is None else round(speed, 4)} |")
  lines.extend(["", "## 安全清洗建议", "",
    "1. 先修正任务文本标注；保留原始值与改标记录，不要直接覆盖来源证据。",
    "2. 解码失败/长度不一致优先隔离复核；时钟大间隔、重复画面、命令跳变只作为复核候选，不自动删除整条。",
    "3. 当前没有硬件速度/加速度极限，不能将筛查阈值称为安全限值；也不应先平滑 command 再声称与原控制一致。",
    "4. 视频帧数相等且 PTS 连续，不等于原始传感器无丢帧；恒定帧率封装可能隐藏实际采集时序。",
    "5. 使用 report.json/episodes.csv/jump_events.csv 定位异常；本次没有改动源数据。", ""])
  with (output / "SEMANTICS.md").open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines))


def write_report(output, episodes, args, complete):
  summary = summarize(episodes)
  save(output / ("report.json" if complete else "lowdim_report.json"),
       {"dataset": str(args.dataset), "video_scan_complete": complete and not args.skip_video,
        "thresholds": {"arm_step_raw": args.arm_step, "hand_step_normalized": args.hand_step},
        "summary": summary, "episodes": episodes})
  if not complete:
    return
  write_semantics(output, episodes)
  with (output / "episodes.csv").open("x", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["episode", "frames", "state_jump_events", "command_jump_events", "clock_estimated_missing", "video_duplicate_frames", "video_errors"])
    for e in episodes:
      writer.writerow([e["episode"], e.get("frames"), len(e.get("trajectory", {}).get("observations/qpos", {}).get("jump_events", [])),
        len(e.get("trajectory", {}).get("action", {}).get("jump_events", [])), e.get("clock_nsec", {}).get("estimated_missing_samples"),
        sum(v.get("adjacent_duplicate_frames", 0) for v in e.get("videos", {}).values()),
        sum(bool(v.get("error") or v.get("decode_errors")) for v in e.get("videos", {}).values())])
  with (output / "jump_events.csv").open("x", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["episode", "dataset", "frame", "dimension", "group", "before", "after", "step", "threshold", "dt_seconds", "speed_raw_per_second", "one_frame_return_spike"])
    for e in episodes:
      for name, result in e.get("trajectory", {}).items():
        for event in result.get("jump_events", []):
          writer.writerow([e["episode"], name, *event.values()])
  text = f"""# 真机扑克数据只读检查

- 数据目录：`{args.dataset}`
- 检查 {summary['episodes']} 条，{summary['frames']} 个主时钟帧；视频检查 {summary['videos_checked']} 路。
- 视频解码/读取异常：{len(summary['video_errors'])} 路；视频与 HDF5 帧数不符：{len(summary['video_frame_count_mismatches'])} 路。
- 相邻完全相同解码帧：{summary['adjacent_duplicate_video_frames']} 个，分布于 {summary['videos_with_adjacent_duplicates']} 路视频。
- 主时钟重复/倒序：{summary['main_clock_duplicate_transitions']}/{summary['main_clock_reversed_transitions']}；大间隔估算缺帧：{summary['main_clock_estimated_missing_samples']}。
- MP4 PTS 大间隔估算缺帧：{summary['video_estimated_missing_pts']}。
- state / command 跳变候选：{summary['jump_events']['observations/qpos']} / {summary['jump_events']['action']} 个（按帧×维度计数）。
- 非有限数值 episode：{len(summary['nonfinite_episodes'])}。
- task 值统计：`{summary['task_counts']}`。

## 字段解释与限制

本脚本针对 `qpos_layout=interleaved_LhRh, arm_dof=14, hand_dof_per_side=6`。
26 维布局解释为左臂 0:7、左手 7:13、右臂 13:20、右手 20:26；手为归一化 [0,1] 控制量。
没有保存逐关节名称、拓扑或正式单位表；臂的角度单位需要采集端确认，脚本用原始位置数值而非强行声明弧度。
`action_source=armcmd_handcmd` 表示 action 来源为 command；`qpos_*_source=measured` 表示 qpos 来源为实测。
实际相等关系见 summary.relation_equal_episode_counts；action_actual_shift 不应与 command action 混用。
任务语义仅按原文件记录；`pick up the cup` 与扑克任务不符，需要人工提供正确标签，不能静默覆盖。
逐 episode schema/text_fields 保存字段证据；没有指尖 XYZ/pose 字段就不能判断其坐标系，更不能套用仿真 world_from_fingertip。

## 判定方法

1. 完整解码原始 MP4，以每帧全分辨率解码像素 MD5 检测相邻完全重复帧，不是压缩包比较或抽帧。
   相同画面可能是静止或复用帧；有损重编码的近似重复不在此规则内。不会自动判坏/删除。
2. 主时钟、独立触觉时钟和视频 PTS 检查重复、倒序及 >1.5 倍周期的间隔；缺失数为周期估算。
   MP4 恒定帧率可能掩盖采集端补帧；未保存逐相机原始时间戳，不能据此证明传感器绝对无丢帧。
3. 视频解码帧数与主时钟长度比较，所有非 raw 时序长度核对。raw 触觉约 50 Hz，本来就不应等于 30 Hz 主帧数。
4. state/action 分开检查 NaN/Inf、手部越界、相邻整行相同及逐维突变。
   每维阈值 = max(臂 {args.arm_step} / 手 {args.hand_step}, median(|差分|) + 8×1.4826×MAD)。
   这些是筛查阈值，不是机器人硬件极限；跳变附真实 dt、维度、原值和一帧回跳特征，应结合视频复核。
5. `report.json` 含每个异常的零基帧号；`jump_events.csv` 可直接筛选轨迹异常，`episodes.csv` 用于选片复核。

仅写报告，不修改、平滑、重采样或删除任何原始数据。视频跳过状态：{args.skip_video}。

## 重跑

在 kaitactilesim 仓库根目录执行（输出目录必须不存在且在数据集以外）：

```bash
python scripts/workcell/audit_real_poker_dataset.py \\
  {args.dataset} \\
  --output /tmp/poker-audit-new --workers 4
```

依赖 h5py、numpy、ffmpeg 和 ffprobe。`--skip-video` 仅检查低维数据；
`--arm-step` / `--hand-step` 可修改突变筛查下限。详细字段语义见 `SEMANTICS.md`。
"""
  with (output / "README.md").open("x", encoding="utf-8") as stream:
    stream.write(text)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", type=Path)
  parser.add_argument("--output", type=Path, required=True, help="New report directory outside dataset")
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--arm-step", type=float, default=0.15)
  parser.add_argument("--hand-step", type=float, default=0.20)
  parser.add_argument("--skip-video", action="store_true")
  parser.add_argument("--limit", type=int)
  args = parser.parse_args()
  args.dataset = args.dataset.resolve(strict=True)
  output = args.output.resolve()
  if output == args.dataset or args.dataset in output.parents:
    parser.error("Report directory must be outside source dataset")
  if args.workers < 1 or args.arm_step <= 0 or args.hand_step <= 0:
    parser.error("workers and thresholds must be positive")
  files = sorted(args.dataset.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
  if args.limit is not None:
    files = files[:args.limit]
  if not files:
    parser.error("No episode HDF5 files found")
  output.mkdir(parents=True, exist_ok=False)
  episodes = []
  for path in files:
    try:
      episodes.append(audit_h5(path, args))
    except Exception as error:
      episodes.append({"episode": path.stem, "errors": [f"{type(error).__name__}: {error}"]})
  write_report(output, episodes, args, complete=False)
  print(f"HDF5 complete: {len(episodes)} episodes; {output / 'lowdim_report.json'}", flush=True)
  if not args.skip_video:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      jobs = {}
      for episode in episodes:
        if "frames" not in episode:
          continue
        episode["videos"] = {}
        for camera in CAMERAS:
          path = args.dataset / "video" / camera / (episode["episode"] + ".mp4")
          jobs[pool.submit(audit_video, path, episode["frames"], float(episode["attributes"].get("fps", 30)))] = (episode, camera)
      for n, future in enumerate(as_completed(jobs), 1):
        episode, camera = jobs[future]
        episode["videos"][camera] = future.result()
        if n % 10 == 0 or n == len(jobs):
          print(f"Videos decoded: {n}/{len(jobs)}", flush=True)
  write_report(output, episodes, args, complete=True)
  print(json.dumps(summarize(episodes), ensure_ascii=False, indent=2), flush=True)
  print(f"Report: {output / 'README.md'}", flush=True)


if __name__ == "__main__":
  main()
