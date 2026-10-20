"""Additive five-finger load series; no changes to archived reports or metrics."""
from __future__ import annotations

import math
from pathlib import Path
import zipfile

import numpy as np

from evaluation_progress_experiment.adapters import (
    MAX_SAMPLES, _SHA, _clock, _gaps, _json, _path, _stable, _verify_rows,
    load_policy_episode,
)


_LABELS = {
    "hand_r_thumb_link6": "右拇指", "hand_r_index_link4": "右食指",
    "hand_r_middle_link4": "右中指", "hand_r_ring_link4": "右无名指",
    "hand_r_pinky_link4": "右小指",
}
_FT_FORMULAS = {
    "sqrt(sum(Ft_x over 7x5)^2 + sum(Ft_y over 7x5)^2)",
    "sqrt(sum(Ft_col over 7x5)^2 + sum(Ft_row over 7x5)^2); signed taxels cancel before magnitude",
}
_SEMANTICS = {
    "normal_force_n": "各指原始记录 Fn；每指 7×5 触觉单元的法向载荷标量和。有效记录值不改动，扩展模块不截断负值为零。",
    "normal_total_n": "同一原生时刻右手五指 Fn 的标量和，不是世界坐标净力或板擦—白板法向力。",
    "tangent_force_n": "各指 |Ft|：先分别求和 7×5 触觉单元的有符号切向分量，再取二维向量长度；不是触觉单元切向力大小之和。",
    "tangent_total_n": "同一原生时刻五个指尖切向合力大小的标量和 Σ_i |Ft_i|；不在不同指尖坐标基之间直接求向量和，不是全手全局净切向力。",
    "unknown": "任一指未知则该时刻五指总和为 null；不把部分指的和、负值或非有限值当作完整总量。",
    "clock": "保留完整原生时间点，不插值、不重采样、不外推；缺测及声明的时间间隙必须断线。",
    "tangent_change": "不接受 contact_glyph.tangent_change_n 作为 Ft；它是时间端点变化，不是切向力本身。",
    "contact_role": "hand_object_unresolved",
    "physical_scope": "指尖—接触对象负载（对象身份未逐接触核验），不等同于稳定抓持、滑移真值或有效擦拭。",
    "source_normal_roundoff": "支持的原始记录器拒绝小于 -1e-12 N 的单元法向载荷，允许的微小负舍入值先截为零再求和；该扩展不重复截断。",
    "units": "N，来自已支持的归档记录 schema；不是压力 Pa，也不是已校准柔性皮肤真值。",
}


def _times(values):
    if (not isinstance(values, (list, np.ndarray)) or np.asarray(values).ndim != 1
            or np.asarray(values).dtype.kind not in "fiu"):
        raise ValueError("Native clock must be numeric")
    return _clock(values, "native tactile")


def _channels(names):
    if (not isinstance(names, list) or len(names) != 5 or any(not isinstance(name, str) for name in names)
            or len(set(names)) != 5 or set(names) != set(_LABELS)):
        raise ValueError("Exactly five distinct explicitly identified right fingertips are required")
    return [{"name": name, "label": _LABELS[name]} for name in names]


def _clean(rows, count, label):
    if rows is None:
        return [[None] * 5 for _ in range(count)]
    if not isinstance(rows, (list, np.ndarray)) or len(rows) != count:
        raise ValueError(f"{label} sample count disagrees with native clock")
    result = []
    for row in rows:
        if row is None:
            result.append([None] * 5)
            continue
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 5:
            raise ValueError(f"{label} requires five scalar values per sample, not vectors or changes")
        clean = []
        for value in row:
            if isinstance(value, (list, tuple, np.ndarray, dict)):
                raise ValueError(f"{label} requires scalar magnitudes, not component vectors or changes")
            valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            try:
                valid = valid and math.isfinite(value) and value >= 0
            except (OverflowError, TypeError):
                valid = False
            clean.append(value.item() if valid and isinstance(value, np.generic) else value if valid else None)
        result.append(clean)
    return result


def _totals(rows):
    result = []
    for row in rows:
        total = sum(row) if all(value is not None for value in row) else None
        try:
            result.append(total if total is not None and math.isfinite(total) else None)
        except OverflowError:
            result.append(None)
    return result


def _status(rows):
    count = sum(value is not None for row in rows for value in row)
    return "available" if count == len(rows) * 5 else "partial" if count else "missing"


def build_load_series(time_s, normal_rows, tangent_rows, channel_names, gap_after, nominal_interval_s) -> dict:
    """Pure scalar-load sums; unknown channels never become partial hand totals."""
    times = _times(time_s)
    channels = _channels(channel_names)
    normal, tangent = _clean(normal_rows, len(times), "Fn"), _clean(tangent_rows, len(times), "|Ft|")
    measured_interval = float(np.median(np.diff(times))) if len(times) > 1 else None
    if nominal_interval_s is not None:
        if (type(nominal_interval_s) not in (int, float) or not math.isfinite(nominal_interval_s)
                or nominal_interval_s <= 0):
            raise ValueError("Nominal interval must be finite and positive or unknown")
    interval = nominal_interval_s if nominal_interval_s is not None else measured_interval
    normal_status, tangent_status = _status(normal), _status(tangent)
    warnings = []
    if normal_status != "available":
        warnings.append("Missing/negative/nonfinite Fn stays null; incomplete five-finger normal totals are unknown")
    if tangent_status != "available":
        warnings.append("Missing/negative/nonfinite |Ft| stays null; incomplete five-finger tangent-magnitude totals are unknown")
    return {"schema_version": "evaluation-five-finger-load-extension-v1", "unit": "N",
            "time_s": times.tolist(), "channels": channels,
            "normal_force_n": normal, "normal_total_n": _totals(normal),
            "tangent_force_n": tangent, "tangent_total_n": _totals(tangent),
            "normal_status": normal_status, "tangent_status": tangent_status,
            "gap_after": _gaps(times, gap_after), "nominal_interval_s": interval,
            "provenance": [], "semantics": dict(_SEMANTICS), "warnings": warnings}


def _unique_rows(rows):
    if not isinstance(rows, list) or len(rows) > 150:
        raise ValueError("Malformed or oversized source provenance")
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str) or not _SHA.fullmatch(row["sha256"])
                or ("size" in row and (type(row["size"]) is not int or row["size"] < 0))):
            raise ValueError("Malformed source SHA256/size provenance")
        previous = result.get(row["path"])
        if previous and (previous["sha256"].lower() != row["sha256"].lower()
                         or ("size" in previous and "size" in row and previous["size"] != row["size"])):
            raise ValueError("Contradictory source SHA256/size provenance")
        result[row["path"]] = row
    return result


def _member(archive, name, *, shape_tail=(), strings=False):
    key = name + ".npy"
    if key not in archive.namelist():
        return None
    info = archive.getinfo(key)
    max_bytes = 4096 if strings else MAX_SAMPLES * max(1, math.prod(shape_tail)) * 8 + 16_384
    if info.file_size > max_bytes:
        raise ValueError(f"Oversized NPZ member: {name}")
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError("Unsupported NPY version")
        shape, _, dtype = reader(stream)
        if strings:
            valid = shape == (5,) and dtype.kind in "SU" and 0 < dtype.itemsize <= 512
        else:
            valid = (len(shape) == 1 + len(shape_tail) and 0 < shape[0] <= MAX_SAMPLES
                     and shape[1:] == shape_tail and dtype.kind in "fiu" and dtype.itemsize <= 8)
        if not valid or dtype.hasobject or info.file_size != stream.tell() + math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Invalid shape, dtype or byte length of NPZ member: {name}")
    with archive.open(info) as stream:
        return np.lib.format.read_array(stream, allow_pickle=False)


def _read_trace(path):
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) > 64 or len(set(names)) != len(names):
            raise ValueError("Duplicate or excessive NPZ entries")
        time = _member(archive, "simulation_time_s")
        elapsed = _member(archive, "time_s")
        fingers = _member(archive, "fingertip_link_names", strings=True)
        normal = _member(archive, "fingertip_normal_force_n", shape_tail=(5,))
        tangent = _member(archive, "fingertip_tangent_force_n", shape_tail=(5,))
    if any(value is None for value in (time, elapsed, fingers, normal)):
        raise ValueError("Native trace lacks clock, channel identities or normal-force binding")
    time, elapsed = _times(time), _times(elapsed)
    names = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in fingers]
    _channels(names)
    if (len(time) != len(elapsed) or normal.shape != (len(time), 5)
            or (tangent is not None and tangent.shape != normal.shape)
            or not np.array_equal(elapsed, time - time[0])):
        raise ValueError("Native force arrays and elapsed/simulation clocks disagree")
    return time, elapsed, names, normal, tangent


def load_extension(source_report: Path, h50_report: Path) -> dict:
    """Derive display-only loads from the identical old/H50 report evidence.

    The old visualization must byte-match the H50 visualization. All original
    source rows must match the freshly verified H50 source bindings. Rehashes
    after decoding stop export on mutation. Only small named NPZ arrays load.
    """
    episode = load_policy_episode(h50_report)
    verified = _unique_rows(episode["provenance"])
    provenance, signatures = [], {}
    source_report, h50_report = _path(source_report), _path(h50_report)
    old = _json(source_report, provenance, signatures, "tactile_extension_source_report")
    stable = _json(h50_report, provenance, signatures, "tactile_extension_h50_report")
    if any(report.get("schema_version") != "task-tactile-evaluation-v1"
           or report.get("suite_schema_version") != "task-tactile-suite-v2" for report in (old, stable)):
        raise ValueError("Unsupported old/H50 report schema")
    if provenance[-1]["sha256"] != verified[str(h50_report)]["sha256"]:
        raise ValueError("H50 report changed after validation")
    trial, stable_trial = old.get("trial"), stable.get("trial")
    if (not isinstance(trial, dict) or not isinstance(stable_trial, dict)
            or any(trial.get(key) != stable_trial.get(key) for key in ("trial_id", "task", "source"))
            or trial.get("trial_id") != episode["id"]
            or trial.get("task") != "whiteboard-wipe"):
        raise ValueError("Old and H50 reports are not the same whiteboard trial/source identity")
    if "prediction_horizon" in trial and (type(trial["prediction_horizon"]) is not int or trial["prediction_horizon"] != 50):
        raise ValueError("Old report contradicts verified H50 identity")
    source = _path(trial.get("source"))
    old_rows = _unique_rows(old.get("provenance"))
    for path, row in old_rows.items():
        if (path not in verified or row["sha256"].lower() != verified[path]["sha256"].lower()
                or ("size" in row and row["size"] != verified[path].get("size"))):
            raise ValueError("Old report source SHA256/size is not bound to the verified H50 report")
    if str(source / "summary.json") not in old_rows:
        raise ValueError("Old report lacks bound source summary")
    visual = _json(source_report.parent / "visualization.json", provenance, signatures, "tactile_extension_source_visualization")
    old_visual_sha = provenance[-1]["sha256"]
    stable_visual_path = h50_report.parent / "visualization.json"
    if str(stable_visual_path) not in verified or old_visual_sha != verified[str(stable_visual_path)]["sha256"]:
        raise ValueError("Old and H50 visualizations differ; refusing silent mixing")
    if (visual.get("schema_version") != "evaluation-visualization-v1" or visual.get("clock_aligned") is not True
            or visual.get("task_summary", {}).get("task") != "whiteboard-wipe"):
        raise ValueError("Unsupported or unaligned source visualization schema")
    tactile = visual.get("tactile_series", {})
    names = [entry.get("name") for entry in visual.get("channels", []) if isinstance(entry, dict)]
    result = build_load_series(tactile.get("time_s"), tactile.get("normal_force_n"), None, names,
                               tactile.get("gap_after"), tactile.get("nominal_interval_s"))
    traces = [row for row in old_rows.values() if row.get("role") == "native_control_cadence_tactile"
              and Path(row["path"]).name == "evaluation_rollout_trace.npz"]
    if len(traces) > 1:
        raise ValueError("Ambiguous native trace source")
    native_trace = None
    extra_warnings = []
    if traces:
        trace_row = traces[0]
        native_trace = _path(trace_row["path"])
        if native_trace not in {source / "evaluation_rollout_trace.npz", source / "review/evaluation_rollout_trace.npz"}:
            raise ValueError("Native trace escapes supported trial layouts")
        _verify_rows([trace_row], lambda path, role: path == native_trace, provenance, signatures)
        summary = _json(source / "summary.json", provenance, signatures, "tactile_extension_source_summary")
        if provenance[-1]["sha256"] != old_rows[str(source / "summary.json")]["sha256"].lower():
            raise ValueError("Source summary SHA256 mismatch")
        video = summary.get("video")
        comparison = video.get("comparison_plots") if isinstance(video, dict) else None
        if not isinstance(comparison, dict):
            raise ValueError("Missing bound native comparison schema")
        schema = comparison.get("tactile", {})
        if (comparison.get("schema") != "kaihand-evaluation-comparison-v1" or not isinstance(schema, dict)
                or comparison.get("artifacts", {}).get("rollout_trace") != str(native_trace)
                or schema.get("unit") != "N" or schema.get("taxel_grid") != [7, 5]
                or schema.get("link_names") != names or schema.get("Fn") != "sum of all 7x5 normal taxel forces"):
            raise ValueError("Unsupported or unbound native tactile schema/unit/channel definition")
        if "Ft" in schema and schema["Ft"] not in _FT_FORMULAS:
            raise ValueError("Unsupported |Ft| definition; signed changes or non-cancelling loads are not this field")
        time, elapsed, native_names, normal, tangent = _read_trace(native_trace)
        rollout = comparison.get("rollout", {})
        origin = visual.get("time_origin_s")
        if (native_names != names or not np.array_equal(elapsed, result["time_s"])
                or type(origin) not in (int, float) or not math.isfinite(origin) or origin != float(time[0])
                or type(rollout.get("samples")) is not int or rollout["samples"] != len(time)
                or type(rollout.get("source_start_time_s")) not in (int, float)
                or rollout["source_start_time_s"] != origin
                or _clean(normal, len(time), "native Fn") != result["normal_force_n"]):
            raise ValueError("Native trace and source visualization normal/clock/channel values disagree")
        if schema.get("Ft") not in _FT_FORMULAS:
            tangent = None
            extra_warnings.append("No bound |Ft| formula; tangent magnitudes remain unknown")
        if tangent is None:
            extra_warnings.append("No native fingertip_tangent_force_n source; never substitute tangent_change_n")
        result = build_load_series(tactile["time_s"], tactile["normal_force_n"], tangent, names,
                                   tactile.get("gap_after"), tactile.get("nominal_interval_s"))
    else:
        extra_warnings.append("No checksum-bound native |Ft| source; tangent magnitudes remain unknown")
    combined = _unique_rows(list(verified.values()) + provenance)
    # Re-read every original source against its pinned digest after all decoding.
    _verify_rows(list(combined.values()), lambda path, role: True, [], {})
    _stable(signatures)
    result.update(trial_id=episode["id"], task=episode["task"], prediction_horizon=50,
                  source_report=str(source_report), h50_report=str(h50_report),
                  native_trace=str(native_trace) if native_trace else None,
                  provenance=list(combined.values()))
    result["warnings"].extend(extra_warnings)
    return result
