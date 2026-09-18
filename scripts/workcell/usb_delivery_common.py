"""USB-only frozen selection and quality provenance for two head-camera releases."""

import hashlib
import json
import shutil
from pathlib import Path

import h5py


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


def validate_usb_outcome(metadata, outcome):
  if (
    metadata.get("scene") != "usb-insert"
    or metadata.get("recording_contract") != "usb_insert_taskspace_raw_v1"
    or metadata.get("observation_clock") != "post_step_forward_v1"
    or outcome.get("object_name") != "usb_plug"
  ):
    raise ValueError("USB export requires the USB task and post-step raw contract")
  for name in (
    "success",
    "released",
    "grasp_verified",
    "active_bottom_out_confirmed",
    "source_files_unchanged",
  ):
    if outcome.get(name) is not True:
      raise ValueError(f"USB source gate failed: {name}")
  for name in ("success", "seated"):
    if outcome.get("insertion", {}).get(name) is not True:
      raise ValueError(f"USB insertion gate failed: {name}")
  if outcome.get("controller_source_sha256_at_end") != metadata.get(
    "controller_source_sha256"
  ):
    raise ValueError("USB controller changed during recording")


def selection(root, processing):
  root, processing = Path(root).resolve(), Path(processing).resolve()
  processing.mkdir(parents=True, exist_ok=True)
  target = processing / "source_selection.json"
  if target.exists():
    result = json.loads(target.read_text())
    if result["source_root"] != str(root):
      raise ValueError("frozen USB source root mismatch")
    return result
  capture = json.loads((root / "summary.json").read_text())
  cleaning = root / "cleaning/dual_format_0910"
  summary = json.loads((cleaning / "summary.json").read_text())
  if not summary["valid"] or not summary["actuator_response_audit_requested"]:
    raise ValueError("USB cleaning and action response audit must pass first")
  audits = {}
  for row in summary["episodes"]:
    path = cleaning / row["report"]
    report = json.loads(path.read_text())
    audits[report["episode"]["episode_index"]] = (path, report)
  rows, digests = [], set()
  for result in sorted(capture["episodes"], key=lambda item: item["episode_index"]):
    if result["status"] != "success" or result["success"] is not True:
      continue
    index = result["episode_index"]
    source = root / "raw" / f"usb_{index:06d}.h5"
    manifest = json.loads(source.with_suffix(".json").read_text())
    digest = sha256(source)
    audit_path, audit = audits[index]
    if (
      digest != manifest["sha256"]
      or digest in digests
      or audit["valid"] is not True
      or audit["errors"]
      or audit["actuator_response_audit"]["valid"] is not True
      or Path(audit["actuator_response_audit"]["source"]).resolve() != source
      or audit["actuator_response_audit"]["source_identity_matches"] is not True
    ):
      raise ValueError(f"USB source identity or cleaning failed: {source}")
    with h5py.File(source, "r") as file:
      metadata = json.loads(file.attrs["metadata_json"])
      outcome = json.loads(file.attrs["outcome_json"])
      validate_usb_outcome(metadata, outcome)
      if (
        file.attrs["camera_hz"] != 30
        or metadata["episode_index"] != index
        or outcome != manifest["outcome"]
      ):
        raise ValueError(
          "USB requires native 30 Hz RGB and matching episode provenance"
        )
    digests.add(digest)
    rows.append(
      {
        "episode_index": index,
        "seed": metadata["object_seed"],
        "session_id": f"usb_0910_{index:06d}",
        "split": "val" if index % 10 == 4 else "train",
        "source": str(source),
        "sha256": digest,
        "cleaning_report": str(audit_path),
        "cleaning_status": "review_required" if audit["review_required"] else "pass",
      }
    )
  if len(rows) != 50 or summary["episodes_checked"] != 50:
    raise ValueError(f"expected 50 completed and inspected successes, got {len(rows)}")
  result = {
    "schema_version": "usb-dual-release-selection-v1",
    "source_root": str(root),
    "episodes": rows,
    "episode_count": len(rows),
    "split_rule": "whole episode; index % 10 == 4 is validation; shared by both formats",
    "cleaning_scope": "all four offline checks and independent actuator/FK checks; event-boundary image review documented separately",
    "failures_included": False,
  }
  save(target, result)
  return result


QUALITY_TEXT = """# USB 数据来源与质量说明

本包包含本批 50 条成功 USB 插入轨迹。失败或 partial 不进入训练演示。
两种格式使用同一批来源和完整 episode 划分：编号 %10==4 为验证集，其余为训练集，
即 45 条 train、5 条 validation；不把相邻帧拆入不同集合。

Cleaning 前四项逐条核验，另核对实际执行器响应方程及图像时刻的腕/指尖 FK。
逐条原始报告随 quality 保存。未平滑、裁剪力值或删除尖峰；检查不是完整动力学重放。
接触起止图像抽查结果随 quality/visual_tactile_review 保存，不宣称全部图像经人工检查。

训练视觉仅使用 head 头部相机，320x240、原生 30 Hz；500 Hz 物理时钟使常规间隔
为 32/34 ms，保留真实时间戳。精确终态帧可能不在严格 30 Hz 网格上，EgoSteer
不将其作为严格网格动作标签，但保留在 NPZ 观察时间轴；EgoTouch 保留真实时间轴。

原始 HDF5 保留在 usb_insert_0910/raw，不在发布包内重复复制；SHA256 标识来源。
EgoTouch 为用户 PDF 的完整腕/指尖 SE3、三通道触觉及 H50（50x108）动作窗口；
忽略外机路径和拧螺丝部分。搬机器后执行包内 rebase_paths.py 重建路径。
EgoSteer 为标准 116D WebDataset、48D 动作，触觉另存 tactile_sidecars NPZ。
两种格式均未预先归一化原始观测；统计只取训练集，不使用验证集。
本包未执行对方的训练程序或硬件力标定。任务标签为 USB 插入。
"""


def attach_quality(destination, snapshot):
  destination = Path(destination).resolve()
  directory = destination / "quality"
  directory.mkdir(exist_ok=True)
  review_source = (
    Path(snapshot["source_root"]) / "cleaning/dual_format_0910/visual_tactile_review"
  )
  review_target = directory / "visual_tactile_review"
  if not (review_source / "inspection.json").is_file():
    raise ValueError("USB tactile boundary image inspection must be documented first")
  if not review_target.exists():
    shutil.copytree(review_source, review_target)
  selection_path = destination / "SOURCE_SELECTION.json"
  if selection_path.exists():
    if json.loads(selection_path.read_text()) != snapshot:
      raise ValueError("refusing to replace a different source selection")
  else:
    save(selection_path, snapshot)
  notes_path = destination / "QUALITY_NOTES.md"
  if notes_path.exists():
    if notes_path.read_text(encoding="utf-8") != QUALITY_TEXT:
      raise ValueError("refusing to replace different quality notes")
  else:
    with notes_path.open("x", encoding="utf-8") as stream:
      stream.write(QUALITY_TEXT)
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
    for path in [
      destination / "SOURCE_SELECTION.json",
      destination / "QUALITY_NOTES.md",
      *sorted(path for path in directory.rglob("*") if path.is_file()),
    ]:
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
