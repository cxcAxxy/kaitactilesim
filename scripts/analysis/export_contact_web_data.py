#!/usr/bin/env python3
"""Export existing whiteboard contact reports as a portable, path-free JS asset.

The presentation clock is the existing 10 Hz glyph clock. Native tactile grids
are matched to the glyph's recorded sample timestamps, never interpolated.
Force-change masks and the original report's duration-weighted summary are
preserved. This script requires only NumPy and does not load the simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


FINGERS = tuple(zip(
    ("thumb", "index", "middle", "ring", "pinky"),
    ("拇指", "食指", "中指", "无名指", "小指"),
))
SOURCES = ("reference", "rollout")
TOLERANCE_S = 1e-7


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files if name != "metadata_json"}


def numeric(value):
    """Make strict JSON, retaining contact booleans separately from rounded force."""
    if isinstance(value, np.ndarray):
        return numeric(value.tolist())
    if isinstance(value, dict):
        return {key: numeric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numeric(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 5) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def masked_vectors(values: np.ndarray, valid: np.ndarray) -> list:
    valid = valid & np.all(np.isfinite(values), axis=-1)
    return [[values[i, finger].tolist() if valid[i, finger] else None
             for finger in range(5)] for i in range(len(values))]


def exact_indices(native: np.ndarray, sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(native)) or np.any(np.diff(native) <= 0):
        raise ValueError("原生触觉时间必须有限且严格递增")
    right = np.clip(np.searchsorted(native, sampled), 0, len(native) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(native[left] - sampled) < np.abs(native[right] - sampled), left, right)
    matched = np.isfinite(sampled) & (np.abs(native[indices] - sampled) <= TOLERANCE_S)
    return indices, matched


def left_hold(native: np.ndarray, values: np.ndarray, display: np.ndarray) -> np.ndarray:
    """Use an observed left sample, without crossing gaps or extending the end."""
    differences = np.diff(native)
    if np.any(~np.isfinite(native)) or np.any(differences <= 0):
        raise ValueError("接触强度时间必须有限且严格递增")
    max_age = 1.5 * float(np.median(differences))
    indices = np.clip(np.searchsorted(native, display + TOLERANCE_S, side="right") - 1,
                      0, len(native) - 1)
    valid = ((display >= native[0] - TOLERANCE_S)
             & (display <= native[-1] + TOLERANCE_S)
             & (display - native[indices] <= max_age + TOLERANCE_S)
             & np.isfinite(values[indices]))
    return np.where(valid, values[indices], np.nan)


def source_payload(source, glyph, spatial, intensity, display, pad_center):
    normal = glyph[f"{source}_normal_force_n"]
    normal_valid = glyph[f"{source}_normal_valid"] & np.isfinite(normal)
    contact = glyph[f"{source}_contact"]
    continuous = glyph[f"{source}_window_continuous"][:, None]
    normal_change_valid = glyph[f"{source}_normal_change_valid"] & continuous
    tangent_change_valid = glyph[f"{source}_tangent_change_valid"] & continuous
    sample_time = glyph[f"{source}_sample_time_s"]
    indices, matched = exact_indices(spatial[f"{source}_time_s"], sample_time)
    # A finite glyph sample always has to resolve to the exact recorded row.
    if np.any(np.isfinite(sample_time) & ~matched):
        raise ValueError(f"{source}: 展示样本不能精确对应原生触觉时间")
    grids = spatial[f"{source}_normal_taxel_force_n"][indices].reshape(len(display), 5, 35)
    grid_valid = normal_valid & matched[:, None] & spatial[f"{source}_normal_valid"][indices]
    # Force agreement catches accidental reference/model ordering mistakes.
    native_force = spatial[f"{source}_normal_force_n"][indices]
    if not np.allclose(native_force[grid_valid], normal[grid_valid], atol=1e-8, rtol=1e-8):
        raise ValueError(f"{source}: 触觉图与合力的原生样本不一致")
    intensity_time = intensity[f"{source}_time_s"]
    return {
        "label": "专家" if source == "reference" else "模型",
        "sampleTime": sample_time,
        "normal": np.where(normal_valid, normal, np.nan),
        "center": masked_vectors(glyph[f"{source}_center_local_mm"] - pad_center,
                                 normal_valid & contact),
        "contact": [[bool(contact[i, finger]) if normal_valid[i, finger] else None
                     for finger in range(5)] for i in range(len(display))],
        "normalChange": np.where(normal_change_valid,
                                 glyph[f"{source}_normal_change_n"], np.nan),
        "tangentChange": masked_vectors(glyph[f"{source}_tangent_change_local_n"],
                                        tangent_change_valid),
        "grids": masked_vectors(grids, grid_valid),
        "board": left_hold(intensity_time, intensity[f"{source}_board_normal_n"], display),
        "cleaned": left_hold(intensity_time, intensity[f"{source}_signal_cleaned_fraction"], display),
    }


def comparison_payload(report: dict) -> dict:
    window = report["common_window"]
    result = {"startS": window["start_s"], "endS": window["end_s"]}
    for source in SOURCES:
        values = window[source]
        result[source] = {
            "gripMean": values["grip_normal_n"]["mean_n"],
            "gripStd": values["grip_normal_n"]["std_n"],
            "pinkyContact": values["fingers"]["小指"]["contact_fraction"],
            "boardMean": values["board_normal_n"]["mean_n"],
            "cleaned": values["last_cleaned_fraction"],
        }
    return result


def events_payload(report: dict, display: np.ndarray) -> list[dict]:
    def nearest(time):
        return float(display[np.argmin(np.abs(display - time))])

    peak = report["native_finger_peaks"].get("rollout_0")
    lost = report.get("final_model_pinky_no_contact")
    events = []
    if report["rollout_end_s"] >= 20:
        events.append({"id": "baseline", "time": 20.0, "displayTime": nearest(20),
                       "label": "20 秒快照", "detail": "查看该时刻的五指受力；具体接触阶段由本次轨迹记录确定。"})
    if peak is not None:
        events.append({"id": "peak", "time": peak["time_s"], "displayTime": nearest(peak["time_s"]),
                       "label": "拇指峰值", "detail": f"本次轨迹原生样本中模型拇指最高达到 {peak['force_n']:.2f} N；"
                       "10 Hz 展示帧可能不包含峰值本身，峰值不必然表示冲击。"})
    if lost:
        events.append({"id": "contact-loss", "time": lost["start_s"],
                       "displayTime": float(display[min(len(display)-1, np.searchsorted(display, lost["start_s"]))]),
                       "label": "小指低于门限", "detail": f"本次模型小指连续 {lost['duration_s']:.3f} 秒未超过接触门限，直至模型结束；不据此推断打滑原因。"})
    if report["rollout_end_s"] >= 34:
        events.append({"id": "snapshot", "time": 34.0, "displayTime": nearest(34),
                       "label": "34 秒快照", "detail": "查看当前法向力、受力中心及过去 0.1 秒的力变化，不预设是否断触。"})
    for source, label in (("reference", "专家"), ("rollout", "模型")):
        end = report[f"{source}_end_s"]
        after = min(int(np.searchsorted(display, end)), len(display) - 1)
        events.append({"id": "end" if source == "rollout" else "reference-end", "time": end,
                       "displayTime": float(display[after]), "label": f"{label}结束",
                       "detail": f"{label}观测到此结束；之后该轨迹显示无观测，另一轨迹按其真实记录继续。"})
    return sorted(events, key=lambda event: event["time"])


def export_data(trial: Path) -> dict:
    glyph = read_npz(trial / "tactile_overall_change.npz")
    spatial = read_npz(trial / "tactile_contact_comparison.npz")
    intensity = read_npz(trial / "contact_intensity.npz")
    overview = read_npz(trial / "contact_strength_overview.npz")
    glyph_report = json.loads((trial / "tactile_overall_change.json").read_text())
    report = json.loads((trial / "contact_strength_analysis.json").read_text())
    summary_path = trial / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    checkpoint_name = Path(summary.get("checkpoint_path", "")).name
    display = glyph["display_time_s"]
    pad_center = glyph["pad_center_local_mm"]
    if not np.allclose(glyph["reference_taxel_xy_mm"], glyph["rollout_taxel_xy_mm"], atol=1e-6):
        raise ValueError("专家与模型的指尖局部网格不一致")
    heatmap = {"edges": overview["bin_edges_s"]}
    for source in SOURCES:
        coverage = overview[f"{source}_coverage_fraction"]
        mean = overview[f"{source}_mean_normal_force_n"]
        # Retain partially covered means with their explicit coverage. Zero
        # coverage is missing, not a zero-force observation.
        heatmap[source] = np.where(coverage > 0, mean, np.nan)
        heatmap[f"{source}Coverage"] = coverage
        heatmap[f"{source}Contact"] = np.where(coverage > 0,
            overview[f"{source}_contact_fraction"], np.nan)
    scales = glyph_report["scales"]
    taxel_max = max(float(np.nanmax(spatial[f"{source}_normal_taxel_force_n"][
        spatial[f"{source}_normal_valid"]])) for source in SOURCES)
    native_peaks = []
    for source in SOURCES:
        for finger, (identifier, label) in enumerate(FINGERS):
            peak = report["native_finger_peaks"][f"{source}_{finger}"]
            native_peaks.append({"source": source, "finger": finger, "fingerId": identifier,
                                 "label": label, "forceN": peak["force_n"], "time": peak["time_s"]})
    payload = {
        "schemaVersion": "contact-web-v1",
        "meta": {
            "title": "接触的语言", "task": "白板擦拭", "seed": trial.name,
            "checkpointStep": int(checkpoint_name) if checkpoint_name.isdigit() else None,
            "taskId": summary.get("task"),
            "timelineEndS": max(report["reference_end_s"], report["rollout_end_s"]),
            "thresholdN": glyph_report["contact_threshold_n"], "windowS": glyph_report["window_s"],
            "referenceEndS": report["reference_end_s"], "rolloutEndS": report["rollout_end_s"],
            "forceMaxN": scales["normal_force_max_n"], "taxelMaxN": taxel_max,
            "deltaMaxN": scales["tangent_change_max_n"], "arrowCapN": scales["arrow_display_cap_n"],
            "normalDeltaMaxN": scales["normal_change_abs_max_n"],
            "referenceHz": glyph_report["reference"]["metadata"]["observed_hz"],
            "rolloutHz": glyph_report["rollout"]["metadata"]["observed_hz"], "displayHz": 10,
        },
        "fingers": [{"id": identifier, "label": label} for identifier, label in FINGERS],
        "time": display,
        "padXY": glyph["reference_taxel_xy_mm"] - pad_center[:, None, :],
        "sources": {source: source_payload(source, glyph, spatial, intensity, display, pad_center)
                    for source in SOURCES},
        "heatmap": heatmap,
        "events": events_payload(report, display),
        "comparison": comparison_payload(report),
        "nativePeaks": native_peaks,
    }
    return numeric(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="输出 JS 路径；文件已存在时拒绝覆盖")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出已存在，拒绝覆盖: {args.output}")
    payload = export_data(args.trial_dir)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write("window.CONTACT_DATA = " + serialized + ";\n")
    print(f"导出 {len(payload['time'])} 个展示时刻，{args.output.stat().st_size:,} 字节：{args.output}")


if __name__ == "__main__":
    main()
