#!/usr/bin/env python3
"""Incremental, serial, read-only Cleaning 1–4 for completed poker captures."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ[name] = "1"

import h5py  # noqa: E402
from kaihand_tactile_env.shared.recording import validate_episode  # noqa: E402
from poker_cleaning_checks import (  # noqa: E402
  check_clocks_and_controls,
  check_duplicate_images,
  digest,
)
from poker_trajectory_checks import check_trajectory  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = [Path(__file__), Path(__file__).with_name("poker_cleaning_checks.py"),
           Path(__file__).with_name("poker_trajectory_checks.py")]


def save(path, payload):
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
  os.replace(temporary, path)


def capture_active():
  path = PROJECT / "datasets/.poker_batch.lock"
  if not path.exists():
    return False
  with path.open("r") as stream:
    try:
      fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
      return True
    fcntl.flock(stream, fcntl.LOCK_UN)
  return False


def completed_sources(root):
  result = []
  for log in sorted((root / "logs").glob("episode_*/execution.json")):
    execution = json.loads(log.read_text())
    if execution.get("completed") is not True:
      continue
    source = root / "raw" / (log.parent.name + ".h5")
    if not source.is_file() or not source.with_suffix(".json").is_file():
      continue
    if source.with_suffix(".h5.lock").exists():
      continue
    manifest = json.loads(source.with_suffix(".json").read_text())
    if manifest.get("outcome", {}).get("success") is True:
      result.append((source, execution, manifest))
  return result


def inspect(source, execution, manifest, output, source_hashes):
  output.mkdir(parents=True, exist_ok=False)
  started = time.monotonic()
  initial = source.stat()
  report = {
    "schema_version": "poker-cleaning-first4-v2", "source": str(source),
    "episode_index": execution["episode_index"], "seed": execution["episode_seed"],
    "original_task_success": True, "audit_complete": False,
    "status": "error", "auditor_source_sha256": source_hashes,
    "new_simulations": 0, "source_modified_by_audit": False,
    "frames_deleted_imputed_or_smoothed": 0,
    "criterion5_or_conversion_performed": False,
    "errors": [], "warnings": [],
  }
  try:
    sha = digest(source)
    report["source_sha256"] = sha
    identity = sha == manifest.get("sha256") == execution.get("validation", {}).get("sha256")
    report["identity_valid"] = identity
    schema = validate_episode(source)
    report["save_schema_validation"] = {
      "valid": schema.valid, "errors": list(schema.errors), "warnings": list(schema.warnings)}
    with h5py.File(source, "r") as file:
      outcome = json.loads(file.attrs["outcome_json"])
      report["original_task_success"] = outcome.get("success") is True
      timing = check_clocks_and_controls(file, output)
      trajectory = check_trajectory(file)
      duplicates = check_duplicate_images(file)
      report.update(timing_and_action_mapping=timing, trajectory=trajectory,
                    duplicate_images=duplicates)
    final = source.stat()
    stable = (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
    report["source_size_mtime_unchanged"] = stable
    clocks_ok = (timing["state"]["valid"] and timing["physics_trace"]["valid"]
                 and timing["force_cache_clock_matches"]
                 and all(c["valid"] for c in timing["cameras"].values()))
    mapping_ok = timing["right_arm_transition_mapping"]["valid"]
    suspect_images = sum(c["suspect_count"] for c in duplicates.values())
    candidates = trajectory.get("candidate_count", 0)
    report["criteria"] = {
      "1_timestamps": "pass" if clocks_ok else "fail",
      "2_state_action_next_state": "pass_with_scope_limit" if mapping_ok and trajectory["valid"] else "fail",
      "3_missing_duplicate_frames": "fail" if not clocks_ok or not schema.valid else "review_required" if suspect_images else "pass",
      "4_state_discontinuities": "fail" if not trajectory["valid"] else "review_required" if trajectory.get("review_required") else "pass",
    }
    if not identity or not stable:
      report["errors"].append("source identity mismatch or source changed during audit")
    if not schema.valid:
      report["errors"].extend(schema.errors)
    if not report["original_task_success"]:
      report["errors"].append("source outcome does not declare task success")
    if not clocks_ok or not mapping_ok:
      report["errors"].append("clock or executed action interval mapping failed")
    report["errors"].extend(trajectory.get("errors", []))
    report["warnings"] = list(schema.warnings) + trajectory.get("warnings", [])
    report["review_candidate_count"] = candidates
    report["suspected_stale_rgb_pairs"] = suspect_images
    report["status"] = ("fail" if report["errors"] else "review_required"
                        if trajectory.get("review_required") or suspect_images else "pass_with_scope_limit")
    report["full_robot_dynamics_replay_verified"] = False
    report["criterion2_scope"] = (
      "All saved right-arm 500Hz executed controls indexed within 100Hz state transitions; "
      "next-state row ctrl checked against final executed substep; sampled joint/card "
      "kinematic consistency checked. Full-robot substep dynamics cannot be certified "
      "from this archive. Same-row ctrl is not the next interval's held action.")
    report["audit_complete"] = True
  except Exception as error:
    report["errors"].append(f"{type(error).__name__}: {error}")
  report["wall_seconds"] = time.monotonic() - started
  report["finished_utc"] = datetime.now(timezone.utc).isoformat()
  save(output / "cleaning.json", report)
  print(f"{source.stem}: {report['status']} ({report['wall_seconds']:.1f}s)", flush=True)
  return report


def summarize(root, output, rows, active, complete):
  counts = {s: sum(r["status"] == s for r in rows)
            for s in ("pass_with_scope_limit", "review_required", "fail", "error")}
  brief = [{k: r.get(k) for k in ("episode_index", "seed", "status", "criteria",
            "review_candidate_count", "suspected_stale_rgb_pairs", "source_sha256", "errors")}
           for r in rows]
  summary = {
    "schema_version": "poker-cleaning-first4-batch-v2", "root": str(root),
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "inspected_episodes": len(rows), "counts": counts,
    "capture_wrapper_active": active, "caught_up_at_exit": complete,
    "new_simulations": 0, "source_data_modified": False,
    "automatic_results_are_not_manual_review_clearance": True,
    "criteria_2_scope": "right-arm executed control timing plus sampled kinematic consistency; no full-robot dynamics replay",
    "episodes": brief,
  }
  save(output / "summary.json", summary)
  lines = ["# 摸牌原始数据 Cleaning 前四项", "",
           f"已检查 {len(rows)} 条成功记录；自动结果：{counts}。",
           f"采集仍运行：{active}；退出时已追平完成记录：{complete}。", "",
           "本检查串行离线运行，没有启动仿真、删帧、补帧、修改或平滑原始数据。",
           "每条目录包含 cleaning.json 与 state_action_intervals.csv。", "",
           "pass_with_scope_limit：前四项自动检查通过，但第2项仅覆盖已归档的右臂500Hz实际控制与采样状态运动学一致性，不宣称全机器人动力学重放。",
           "review_required：结构检查没有硬错误，但存在轨迹/图像候选，必须结合阶段解释；不能直接作为无条件通过名单。",
           "fail/error：检测到错误或检查不完整。任务成功与Cleaning通过单独记录。", "",
           "30Hz相机按500Hz物理时钟量化，允许32/34ms间隔；额外终帧按实际终态时刻检查。",
           "相机索引对应最近非未来100Hz状态，RGB/触觉按各自求值时刻核对，不要求不同时钟采样行强行相等。", "",
           "本轮只做前四项，不进行第5项触觉起止视觉人工验收，也不转换格式。", "",
           "重新运行相同命令只检查新增已成功关闭的文件。已检报告只在源SHA和检查器源码都匹配时复用。"]
  temporary = output / "README.md.tmp"
  temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
  os.replace(temporary, output / "README.md")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--watch-until-idle", action="store_true")
  parser.add_argument("--max-watch-seconds", type=float, default=3600)
  args = parser.parse_args()
  root = args.dataset_root.resolve()
  output = (args.output_dir or root / "cleaning/first4_v2").resolve()
  output.mkdir(parents=True, exist_ok=True)
  source_hashes = {p.name: digest(p) for p in SOURCES}
  fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:12]
  started = time.monotonic()
  idle_scans = 0
  with (output / ".audit.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
      rows = []
      completed = completed_sources(root)
      for source, execution, manifest in completed:
        directory = output / f"{source.stem}_{manifest['sha256'][:12]}_{fingerprint}"
        report = directory / "cleaning.json"
        if report.exists():
          row = json.loads(report.read_text())
          if row.get("source_sha256") != manifest["sha256"] or row["auditor_source_sha256"] != source_hashes:
            raise RuntimeError("audit cache identity mismatch")
        else:
          print(f"Checking {source.stem} (read-only, one file at a time)", flush=True)
          row = inspect(source, execution, manifest, directory, source_hashes)
        rows.append(row)
        summarize(root, output, rows, capture_active(), False)
      active = capture_active()
      fresh_count = len(completed_sources(root))
      idle_scans = idle_scans + 1 if not active and fresh_count == len(rows) else 0
      finished = not args.watch_until_idle or idle_scans >= 2
      summarize(root, output, rows, active, not active and fresh_count == len(rows) and finished)
      if finished:
        break
      if time.monotonic() - started > args.max_watch_seconds:
        print("Watch time limit reached; rerun to audit later completed successes", flush=True)
        return 2
      time.sleep(10)
  print(json.dumps({"inspected": len(rows), "summary": str(output / "summary.json")}), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
