"""Checksum-bound native wrist positions, never inferred eraser motion.

The supported comparison writer reads ``data.site_xpos[hand_r_base_link_site]``
into wrist_state[:, :3]; its companion plot labels xyz in metres.  The remaining
six columns are rotation columns, not Euler angles or commanded joint targets.
Only the SHA-bound summary's embedded comparison metadata is trusted here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import zipfile

import numpy as np

from evaluation_progress_experiment.adapters import (
    MAX_JSON_BYTES, MAX_SAMPLES, MAX_SOURCE_BYTES, _SHA, _clock, _parse, _path, _read,
)


_LAYOUT = "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof"
_SITE = "hand_r_base_link_site"
_LIMITATIONS = [
    "手腕位置不是板擦位置；不假定刚性抓持，不能据此判定滑移、板擦速度或有效擦拭。",
    "关节指令不等于动作意图；这里读取物理手腕 site 状态，不读取指令或推断意图。",
    "只精确匹配原生记录时间，不插值、外推或近邻补齐；缺测保留 null。",
    "同记录关联，非瞬时同相位验证；不将手腕运动与接触力的关联解释为因果。",
    "米及世界坐标来自已支持记录 schema 的源码约定，并非 NPZ 中独立标注的单位。",
]


class _InvalidSource(ValueError):
    pass


def _rows(value):
    if not isinstance(value, list) or len(value) > 150:
        raise ValueError("Missing or oversized motion source provenance")
    for row in value:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str) or not _SHA.fullmatch(row["sha256"])
                or not Path(row["path"]).is_absolute()):
            raise ValueError("Malformed motion source provenance")
    return value


def _binding(rows, path):
    matching = [row for row in rows if row["path"] == str(path)]
    if not matching:
        raise ValueError(f"Source is not checksum-bound in episode provenance: {path}")
    hashes = {row["sha256"].lower() for row in matching}
    sizes = {row["size"] for row in matching if "size" in row and type(row["size"]) is int}
    if (len(hashes) != 1 or len(sizes) > 1
            or any("size" in row and (type(row["size"]) is not int or row["size"] < 0) for row in matching)):
        raise ValueError("Contradictory motion source SHA256 or size bindings")
    return matching[0]


def _bound_read(path, row, role, provenance, snapshots, *, collect=False):
    path = _path(path)
    raw, digest, size, signature = _read(path, limit=MAX_JSON_BYTES if collect else MAX_SOURCE_BYTES,
                                        collect=collect)
    if digest != row["sha256"].lower() or ("size" in row and size != row["size"]):
        raise ValueError(f"Motion source SHA256 or size mismatch: {path}")
    provenance.append({"path": str(path), "sha256": digest, "size": size, "role": role})
    snapshots[path] = (digest, size, signature)
    return raw


def _unchanged(snapshots):
    # Rehash after decoding, not merely stat: mutation is an export-stopping error.
    for path, before in snapshots.items():
        try:
            _, digest, size, signature = _read(_path(path), limit=MAX_SOURCE_BYTES)
        except (OSError, ValueError) as error:
            raise ValueError(f"Motion source changed during read: {path}") from error
        if (digest, size, signature) != before:
            raise ValueError(f"Motion source changed during read: {path}")


def _numeric_clock(values, label):
    if not isinstance(values, (list, np.ndarray)):
        raise ValueError(f"Invalid {label} clock")
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.ndim != 1:
        raise ValueError(f"Invalid {label} clock")
    return _clock(values, label)


def _array(archive, name, width=None):
    """Read only a bounded numeric NPY member; never unpickle or load taxels."""
    member = f"{name}.npy"
    if member not in archive.namelist():
        raise _InvalidSource(f"Missing native array: {name}")
    info = archive.getinfo(member)
    if info.file_size > MAX_SAMPLES * (width or 1) * 8 + 16_384:
        raise _InvalidSource(f"Oversized native array: {name}")
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise _InvalidSource("Unsupported NPY version")
        shape, _, dtype = reader(stream)
        expected_ndim = 2 if width else 1
        if (len(shape) != expected_ndim or not 0 < shape[0] <= MAX_SAMPLES
                or (width and shape[1] != width) or dtype.kind not in "fiu"
                or dtype.itemsize > 8 or dtype.hasobject
                or info.file_size != stream.tell() + math.prod(shape) * dtype.itemsize):
            raise _InvalidSource(f"Invalid dimensions, dtype or length for {name}")
    with archive.open(info) as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)


def _decode(path):
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) > 64 or len(set(names)) != len(names):
                raise _InvalidSource("Duplicate or excessive NPZ entries")
            times = _numeric_clock(_array(archive, "time_s"), "native elapsed")
            simulation = _numeric_clock(_array(archive, "simulation_time_s"), "native simulation")
            wrist = _array(archive, "wrist_state", width=9)
    except (ValueError, OSError, EOFError, zipfile.BadZipFile, RuntimeError) as error:
        raise _InvalidSource(f"Invalid native motion arrays: {error}") from error
    if len(times) != len(simulation) or wrist.shape[0] != len(times):
        raise _InvalidSource("Native wrist and clock sample counts disagree")
    if not np.array_equal(times, simulation - simulation[0]):
        raise _InvalidSource("Native elapsed and simulation clocks disagree")
    return times, simulation, wrist


def _extract(episode, result, snapshots):
    rows = _rows(episode.get("provenance"))
    report_path = Path(episode.get("source_report", ""))
    report_row = _binding(rows, report_path)
    report = _parse(_bound_read(report_path, report_row, "motion_source_report", result["provenance"], snapshots, collect=True))
    trial = report.get("trial", {})
    if (not isinstance(trial, dict) or trial.get("trial_id") != episode.get("id")
            or trial.get("task") != "whiteboard-wipe" or type(trial.get("prediction_horizon")) is not int
            or trial["prediction_horizon"] != 50):
        raise ValueError("Motion episode identity or H50 report binding disagrees")
    source = _path(trial.get("source"))
    archived = _rows(report.get("provenance"))
    summary_path = source / "summary.json"
    summary_row = _binding(archived, summary_path)
    if _binding(rows, summary_path)["sha256"].lower() != summary_row["sha256"].lower():
        raise ValueError("Episode and report summary SHA256 bindings disagree")
    summary = _parse(_bound_read(summary_path, summary_row, "motion_source_summary", result["provenance"], snapshots, collect=True))
    trace_rows = [row for row in archived if row.get("role") == "native_control_cadence_tactile"
                  and Path(row["path"]).name == "evaluation_rollout_trace.npz"]
    if not trace_rows:
        result["warnings"].append("No archived checksum-bound native rollout trace; motion remains null")
        return
    if len(trace_rows) != 1:
        raise ValueError("Ambiguous archived motion trace identity")
    row = trace_rows[0]
    path = Path(row["path"])
    if path not in {source / "evaluation_rollout_trace.npz", source / "review/evaluation_rollout_trace.npz"}:
        raise ValueError("Motion trace is outside the declared trial's supported layouts")
    if _binding(rows, path)["sha256"].lower() != row["sha256"].lower():
        raise ValueError("Episode and report trace SHA256 bindings disagree")
    _bound_read(path, row, "motion_native_rollout_trace", result["provenance"], snapshots)
    video = summary.get("video")
    comparison = video.get("comparison_plots") if isinstance(video, dict) else None
    if not isinstance(comparison, dict):
        raise _InvalidSource("Missing checksum-bound comparison schema in summary.video.comparison_plots")
    state, rollout, artifacts = (comparison.get(key) for key in ("state", "rollout", "artifacts"))
    if (comparison.get("schema") != "kaihand-evaluation-comparison-v1"
            or not all(isinstance(value, dict) for value in (state, rollout, artifacts))
            or state.get("layout") != _LAYOUT or state.get("wrist_site") != _SITE
            or type(state.get("dimension")) is not int or state["dimension"] != 29
            or artifacts.get("rollout_trace") != str(path)):
        raise _InvalidSource("Unsupported wrist layout/site or comparison trace identity")
    # V1 has implicit metre/world conventions; reject explicit contradictory additions.
    if any(key in state and state[key] != expected for key, expected in
           (("position_unit", "m"), ("position_units", "m"), ("length_unit", "m"),
            ("unit", "m"), ("coordinate_frame", "world"), ("position_frame", "world"))):
        raise _InvalidSource("Contradictory wrist position unit or coordinate frame")
    times, simulation, wrist = _decode(path)
    origin = episode.get("metadata", {}).get("source_time_origin_s")
    start = rollout.get("source_start_time_s")
    if (type(origin) not in (int, float) or not math.isfinite(origin)
            or type(start) not in (int, float) or not math.isfinite(start)
            or origin != start or start != float(simulation[0])
            or type(rollout.get("samples")) is not int or rollout["samples"] != len(times)):
        raise _InvalidSource("Episode origin or bound rollout clock/sample count disagrees")
    valid = np.isfinite(wrist[:, :3]).all(axis=1)
    by_time = {float(time): position.tolist() if ok else None
               for time, position, ok in zip(times, wrist[:, :3], valid)}
    positions = [by_time.get(float(time)) for time in result["time_s"]]
    matched = sum(position is not None for position in positions)
    result.update(position_m=positions, matched_count=matched, source_sample_count=len(times),
                  status="available" if matched == len(positions) else "partial" if matched else "missing",
                  source_clock="time_s", source_time_origin_s=float(origin),
                  schema_evidence="SHA-bound summary.video.comparison_plots.state",
                  record_clock_association="Exact native elapsed timestamps; elapsed equals simulation_time_s minus verified origin")
    if matched != len(positions):
        result["warnings"].append("Unmatched native times or nonfinite xyz rows remain null; no interpolation or zero fill")


def load_wrist_motion(episode: dict) -> dict:
    """Augment an already-adapted H50 policy case without changing its clock.

    Missing/invalid native motion produces nulls. SHA mismatches, inconsistent
    bindings and changes during reading raise ValueError to stop the export.
    Only the report, bound summary and three small NPZ arrays are decoded.
    """
    if (not isinstance(episode, dict) or episode.get("role") != "policy"
            or episode.get("task") != "whiteboard-wipe"
            or not isinstance(episode.get("metadata"), dict)
            or type(episode["metadata"].get("prediction_horizon")) is not int
            or episode["metadata"]["prediction_horizon"] != 50):
        raise ValueError("Wrist motion requires an adapted whiteboard H50 policy episode")
    _numeric_clock(episode.get("time_s"), "episode")
    result = {"status": "missing", "time_s": deepcopy(episode["time_s"]),
              "position_m": [None] * len(episode["time_s"]), "unit": "m", "coordinate_frame": "world",
              "frame": "world", "object": "wrist", "label": "Recorded right wrist site position",
              "site": _SITE, "matched_count": 0, "source_sample_count": None,
              "provenance": [], "limitations": list(_LIMITATIONS), "warnings": []}
    snapshots = {}
    try:
        _extract(episode, result, snapshots)
    except FileNotFoundError as error:
        result["warnings"].append(f"Bound source is unavailable; wrist positions remain null: {error.filename}")
    except _InvalidSource as error:
        result["status"] = "invalid"
        result["warnings"].append(str(error))
    finally:
        _unchanged(snapshots)
    return result
