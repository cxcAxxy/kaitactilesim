"""Attach existing PNG diagnostics through declared archive metadata, never redraw."""
from __future__ import annotations

import math
from pathlib import Path
import re
import struct
import zlib

from evaluation_progress_experiment.adapters import _path, _read
from evaluation_v2.horizon import inspect_prediction_horizon

from .export import _digest, _document, _rows, _verify


_PLOTS = (
    ("tactile", "right_fingertip_tactile", "五指法向 / 切向触觉", "evaluation_right_fingertip_tactile.png",
     "右手五指 Fn 与 |Ft| 汇总曲线；每指先累加有符号切向分量，再取切向合力大小。"),
    ("hand-dof", "right_hand_actuated_dof", "右手 20 个驱动关节", "evaluation_right_hand_actuated_dof.png",
     "右手 20 个驱动关节的实际角度（rad）。"),
    ("wrist", "right_wrist_state", "右腕位置与姿态分量", "evaluation_right_wrist_state.png",
     "右腕世界坐标 xyz（m）与 rot6d 姿态分量（无量纲）。"),
)


def _declared_path(value, expected):
    if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).is_symlink():
        raise ValueError("Diagnostic artifact needs an explicit non-leaf-symlink absolute path")
    resolved = Path(value).resolve(strict=True)
    if resolved != expected:
        raise ValueError("Diagnostic artifact/reference path escapes its declared trial layout")
    return _path(resolved)


def _png(path):
    raw, sha, size, _ = _read(path, limit=20_000_000, collect=True)
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Diagnostic artifact is not PNG")
    offset, dimensions, ended, image_data = 8, None, False, False
    while offset + 12 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        stop = offset + 12 + length
        if stop > len(raw):
            raise ValueError("Truncated diagnostic PNG")
        data = raw[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", raw[offset + 8 + length:stop])[0]
        if zlib.crc32(kind + data) & 0xffffffff != crc:
            raise ValueError("Diagnostic PNG chunk checksum mismatch")
        if dimensions is None:
            if kind != b"IHDR" or length != 13:
                raise ValueError("Diagnostic PNG needs a leading IHDR")
            dimensions = struct.unpack(">II", data[:8])
            if not all(0 < value <= 12000 for value in dimensions) or math.prod(dimensions) > 80_000_000:
                raise ValueError("Diagnostic PNG dimensions are oversized")
        elif kind == b"IHDR":
            raise ValueError("Duplicate diagnostic PNG header")
        if kind == b"IDAT":
            image_data = True
        if kind == b"IEND":
            ended = length == 0 and stop == len(raw)
            break
        offset = stop
    if not ended or not image_data:
        raise ValueError("Incomplete diagnostic PNG")
    return dimensions, {"path": str(path), "sha256": sha, "size": size, "role": "diagnostic_png"}


def load_diagnostic_plots(trial_dir, *, expected_task, expected_summary_sha=None, url_prefix="plots"):
    """Return display config and exact source/relative-target copy records.

    Metadata binds image paths and the recorded reference identity. These PNGs
    do not carry historical hashes; hashes here pin the bytes being attached.
    Reference HDF/NPZ arrays are not loaded and no metric is recomputed.
    """
    if not isinstance(url_prefix, str) or not re.fullmatch(r"plots(?:/[A-Za-z0-9_-]+)?", url_prefix):
        raise ValueError("Unsafe diagnostic plot URL prefix")
    trial = _path(Path(trial_dir).resolve(strict=True))
    summary_path = trial / "summary.json"
    summary_row = _digest(summary_path)
    if expected_summary_sha is not None and summary_row["sha256"] != expected_summary_sha:
        raise ValueError("Diagnostic summary SHA256 does not match the selected model trial")
    summary = _document(summary_path)
    horizon = inspect_prediction_horizon(trial)
    if horizon["status"] != "verified" or horizon["prediction_horizon"] != 50:
        raise ValueError("Diagnostic trial must explicitly verify prediction horizon H50")
    task, seed, checkpoint = summary.get("task"), summary.get("seed"), summary.get("checkpoint_path")
    if (task != expected_task or not isinstance(task, str) or not task.strip()
            or type(seed) is not int or seed < 0 or not isinstance(checkpoint, str) or not checkpoint
            or not Path(checkpoint).is_absolute()):
        raise ValueError("Diagnostic task/seed/checkpoint identity is missing or disagrees")
    def missing(reason, provenance=None):
        rows = _rows([summary_row] + horizon["provenance"] + (provenance or []))
        _verify(rows)
        return {"status": "missing", "task": task, "seed": seed, "checkpoint": Path(checkpoint).name,
                "prediction_horizon": 50, "reason": reason, "plots": [], "provenance": rows}, []
    video = summary.get("video")
    if not isinstance(video, dict):
        return missing("No recorded comparison images; no other trial's images substituted")
    if not isinstance(video, dict) or video.get("completed") is not True or video.get("task") != task or video.get("seed") != seed:
        raise ValueError("Diagnostic source has no completed same-trial review declaration")
    comparison = video.get("comparison_plots")
    if not isinstance(comparison, dict) or comparison.get("status") != "ok":
        return missing("Recorded comparison images are unavailable; reference/images were not generated or substituted")
    if (not isinstance(comparison, dict) or comparison.get("schema") != "kaihand-evaluation-comparison-v1"
            or comparison.get("status") != "ok" or not isinstance(comparison.get("artifacts"), dict)):
        raise ValueError("Diagnostic comparison metadata is unavailable or unsupported")
    declared = comparison["artifacts"].get("metadata")
    if not isinstance(declared, str):
        raise ValueError("Missing declared comparison metadata path")
    if Path(declared).is_symlink():
        raise ValueError("Diagnostic metadata leaf symlink is unsupported")
    location = Path(declared).resolve(strict=False)
    if location not in {trial / "evaluation_comparison.json", trial / "review/evaluation_comparison.json"}:
        raise ValueError("Diagnostic comparison metadata belongs to another trial")
    if not location.is_file():
        return missing("The declared comparison metadata is unavailable")
    location = _declared_path(declared, location)
    directory = location.parent
    metadata_paths = [location, directory / "review.json", directory / "evaluation_evaluation_plots.json"]
    if any(not path.is_file() for path in metadata_paths) or any(not (directory / row[3]).is_file() for row in _PLOTS):
        return missing("The complete three-image diagnostic bundle is not available")
    provenance = [summary_row] + [_digest(path) for path in metadata_paths]
    saved, review, plots = [_document(path) for path in metadata_paths]
    if (saved != comparison or review.get("comparison_plots") != comparison
            or review.get("task") != task or review.get("seed") != seed or review.get("completed") is not True
            or review.get("checkpoint_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError("Summary/comparison/review diagnostic identities disagree")
    styles = {"reference": "solid", "evaluation_rollout": "dashed", "time_axis": "elapsed seconds; each sequence starts at zero"}
    if plots.get("schema") != "kaihand-openwam-evaluation-plots-v1" or comparison.get("styles") != styles or plots.get("styles") != styles:
        raise ValueError("Unsupported diagnostic plot schema or reference/rollout line semantics")
    _declared_path(plots.get("artifacts", {}).get("metadata"), directory / "evaluation_evaluation_plots.json")
    reference, plot_reference = comparison.get("reference", {}), plots.get("reference", {})
    for key in ("dataset_root", "source_hdf5", "output_episode_index", "samples"):
        if key not in reference or reference[key] != plot_reference.get(key):
            raise ValueError("Diagnostic plot reference dataset/HDF/episode binding disagrees")
    if (reference["dataset_root"] != video.get("reference_dataset")
            or reference["output_episode_index"] != video.get("reference_episode_index")
            or type(reference["output_episode_index"]) is not int or reference["output_episode_index"] < 0
            or not isinstance(reference["source_hdf5"], str) or not Path(reference["source_hdf5"]).is_absolute()):
        raise ValueError("Diagnostic review/reference selection disagrees")
    for key in ("samples", "source_start_time_s", "duration_s"):
        value = comparison.get("rollout", {}).get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or plots.get("rollout", {}).get(key) != value:
            raise ValueError("Diagnostic rollout clocks disagree")
    for key in ("layout", "dimension", "wrist_site", "rot6d", "joint_names"):
        if comparison.get("state", {}).get(key) != plots.get("state", {}).get(key):
            raise ValueError("Diagnostic state layout disagrees")
    state = plots.get("state", {})
    if (state.get("layout") != "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof" or state.get("dimension") != 29
            or state.get("wrist_site") != "hand_r_base_link_site" or state.get("right_hand_actuated_dof") != 20
            or not isinstance(state.get("joint_names"), list) or len(state["joint_names"]) != 20):
        raise ValueError("Unsupported diagnostic wrist/hand state semantics")
    for key in ("finger_order", "link_names", "taxel_grid", "Fn", "unit"):
        if comparison.get("tactile", {}).get(key) != plots.get("tactile", {}).get(key):
            raise ValueError("Diagnostic tactile schema disagrees")
    if plots.get("tactile", {}).get("unit") != "N":
        raise ValueError("Diagnostic tactile units must be recorded N")
    config_plots, copies = [], []
    for identifier, artifact, label, name, description in _PLOTS:
        path = directory / name
        for metadata in (comparison, plots):
            _declared_path(metadata.get("artifacts", {}).get(artifact), path)
        dimensions, row = _png(_path(path))
        provenance.append(row)
        url = f"{url_prefix}/{name}"
        config_plots.append({"id": identifier, "label": label, "description": description, "url": url,
                             "sha256": row["sha256"], "width": dimensions[0], "height": dimensions[1]})
        copies.append({"source": path, "url": url, "row": row})
    provenance = _rows(provenance + horizon["provenance"])
    _verify(provenance)
    config = {
        "status": "available", "task": task, "seed": seed, "checkpoint": Path(checkpoint).name,
        "checkpoint_path": checkpoint, "checkpoint_sha256": summary.get("checkpoint_sha256"), "prediction_horizon": 50,
        "source_trial": str(trial), "source_label": f"{task} · seed {seed:03d} · checkpoint {Path(checkpoint).name} · H50",
        "comparison_note": "横轴是每条轨迹各自开始后的时间；参考与模型可能处于不同动作阶段。原图用于全程观察，不跟随视频游标。",
        "reference_label": f"蓝色实线 · 参考 episode {reference['output_episode_index']}",
        "rollout_label": f"红色虚线 · 模型 seed {seed:03d}",
        "reference": reference, "plots": config_plots, "provenance": provenance,
        "tactile_note": comparison.get("tactile", {}).get("comparison_note", "参考与模型触觉来源/校准可不同，量值可比性未核验。"),
        "binding_note": "归档 summary/review/比较元数据绑定原图路径及参考身份；PNG 字节在本次挂载固定 SHA256，未宣称存在历史 PNG 摘要，也未重读参考 HDF。",
        "warnings": horizon["warnings"],
    }
    return config, copies
