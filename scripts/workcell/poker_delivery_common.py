"""Frozen source selection and portable quality notes for the two poker releases."""

import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
  result = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
      result.update(chunk)
  return result.hexdigest()


def save(path, value):
  with Path(path).open("x", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")


def selection(root, processing, expected_count=50):
  root, processing = Path(root).resolve(), Path(processing).resolve()
  processing.mkdir(parents=True, exist_ok=True)
  target = processing / "source_selection.json"
  if target.exists():
    result = json.loads(target.read_text())
    if result["source_root"] != str(root) or result["episode_count"] != expected_count:
      raise ValueError("frozen source root/count mismatch")
    return result
  merged = {}
  if (root / "manifest.json").exists():
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") == "poker-merged-successes-v1":
      verified = json.loads((root / "final_verification.json").read_text())
      if not manifest.get("completed") or not verified.get("valid") or verified.get("episode_count") != expected_count:
        raise ValueError("merged source requires completed final verification")
      merged = {r["episode_index"]: r for r in manifest["episodes"]}
      if len(merged) != expected_count:
        raise ValueError("merged manifest count mismatch")
  cleaning = root / "cleaning/first4_v2"
  summary = json.loads((cleaning / "summary.json").read_text())
  audits = {}
  for path in cleaning.glob("episode_*/cleaning.json"):
    report = json.loads(path.read_text())
    audits[(report["episode_index"], report.get("source_sha256"))] = (path, report)
  rows = []
  for path in sorted((root / "logs").glob("episode_*/execution.json")):
    execution = json.loads(path.read_text())
    if execution.get("completed") is not True:
      continue
    source = root / "raw" / (path.parent.name + ".h5")
    manifest = json.loads(source.with_suffix(".json").read_text())
    index = execution["episode_index"]
    digest = manifest["sha256"]
    audit_path, audit = audits[(index, digest)]
    if (manifest.get("outcome", {}).get("success") is not True
        or execution["validation"]["sha256"] != digest
        or not audit.get("audit_complete") or audit.get("errors")
        or audit["status"] not in ("review_required", "pass_with_scope_limit")):
      raise ValueError(f"source has failed or incomplete checks: {source}")
    provenance = merged.get(index)
    if merged and (provenance is None or provenance["sha256"] != digest):
      raise ValueError("merged source identity mismatch")
    if provenance is not None:
      split = preserved_split(provenance)
      session_id = f"{root.name}_{index:06d}"
    else:
      split = "val" if index % 10 == 4 else "train"
      session_id = f"poker_0909_{index:06d}"
    rows.append({"episode_index": index, "seed": execution["episode_seed"],
                 "session_id": session_id,
                 "split": split,
                 "source": str(source), "sha256": digest,
                 "cleaning_report": str(audit_path), "cleaning_status": audit["status"],
                 "provenance": provenance})
  if len(rows) != expected_count or summary["inspected_episodes"] != expected_count:
    raise ValueError(f"expected exactly {expected_count} completed and inspected successes, got {len(rows)}")
  result = {"schema_version": "poker-dual-release-selection-v1", "source_root": str(root),
            "episodes": rows, "episode_count": len(rows),
            "split_rule": ("whole episode; preserve previous_split when present; otherwise ORIGINAL source_episode_index % 10 == 4 is validation; shared by both formats" if merged else
                           "whole episode; index % 10 == 4 is validation; shared by both formats"),
            "cleaning_scope": "first four offline checks; velocity review warnings retained; no unconditional manual or hardware safety approval",
            "failures_included": False}
  save(target, result)
  return result


def preserved_split(provenance):
  previous = provenance.get("previous_split")
  if previous is not None:
    if previous not in ("train", "val"):
      raise ValueError(f"unknown historical split: {previous}")
    return previous
  return "val" if int(provenance["source_episode_index"]) % 10 == 4 else "train"


QUALITY_TEXT = """# 数据来源与质量说明

本包只包含同一批50条任务成功的摸牌数据，失败和partial不作为训练演示。
两种格式共享完整episode划分：原始编号%10==4为val，其余train，不按相邻帧拆分。
成功样本存在筛选偏差，不能用本包推断全部随机初态100%成功。

Cleaning前四项已逐条离线检查；无时间/动作索引、缺帧或状态结构硬错误。
腕/指尖接触前快速运动仍为review_required警告，不把它写成无条件通过；
保留逐条报告，没有删帧、平滑或裁剪力值。第2项只验证已归档右臂500Hz实际
输入索引与采样运动学一致性，未完成全机器人动力学重放或硬件安全认证。

视觉只使用head顶部相机，320x240。名义30Hz在500Hz物理时钟上量化，正常
相邻采样间隔为32/34ms；真实求值时刻与采集时刻分别保留，不伪造插值图像。
原始HDF5未包含在本训练包中，仍保留在poker_draw_0909/raw，使用SHA256追溯。
旧机器的源路径只作provenance，不需要在接收机器上重现。

EgoTouch使用PDF的完整腕/指尖SE3、三通道触觉与H50窗口；忽略PDF的外机路径、
Part11及外部launcher/训练开发指令。搬机器后先执行包内rebase_paths.py。
EgoSteer使用本机项目的标准116D WebDataset；触觉是独立NPZ，不改变48D动作。
两者均未运行外部完整训练，不把本地格式验证当作远端训练成功。
"""


def attach_quality(destination, snapshot):
  destination = Path(destination).resolve()
  directory = destination / "quality"
  directory.mkdir(exist_ok=True)
  selection_path = destination / "SOURCE_SELECTION.json"
  if selection_path.exists():
    if json.loads(selection_path.read_text()) != snapshot:
      raise ValueError("refusing to replace a different source selection")
  else:
    save(selection_path, snapshot)
  notes_path = destination / "QUALITY_NOTES.md"
  quality_text = QUALITY_TEXT.replace("同一批50条", f"同一批{snapshot.get('episode_count', len(snapshot['episodes']))}条").replace(
    "原始编号%10==4为val，其余train", snapshot.get("split_rule", "原始编号%10==4为val，其余train")).replace(
    "poker_draw_0909/raw", Path(snapshot.get("source_root", "poker_draw_0909")).name + "/raw")
  if notes_path.exists():
    if notes_path.read_text(encoding="utf-8") != quality_text:
      raise ValueError("refusing to replace different quality notes")
  else:
    with notes_path.open("x", encoding="utf-8") as stream:
      stream.write(quality_text)
  for row in snapshot["episodes"]:
    target = directory / f"episode_{row['episode_index']:06d}.json"
    if target.exists():
      if sha256(target) != sha256(row["cleaning_report"]):
        raise ValueError("refusing to replace a different cleaning report")
    else:
      shutil.copyfile(row["cleaning_report"], target)
  integrity_path = destination / "transfer_integrity.json"
  if integrity_path.exists():
    from kaihand_tactile_env.shared.tict_relocation import _digest
    integrity = json.loads(integrity_path.read_text())
    for path in [destination / "SOURCE_SELECTION.json", destination / "QUALITY_NOTES.md", *sorted(directory.glob("*.json"))]:
      record = _digest(destination, path.relative_to(destination).as_posix())
      existing = [item for item in integrity["files"] if item["path"] == record["path"]]
      if existing:
        if existing != [record]:
          raise ValueError("existing quality integrity record differs")
      else:
        integrity["files"].append(record)
    temporary = integrity_path.with_suffix(".tmp")
    save(temporary, integrity)
    temporary.replace(integrity_path)
