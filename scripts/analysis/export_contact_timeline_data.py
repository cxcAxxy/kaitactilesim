#!/usr/bin/env python3
"""Export real tactile, contact-phase and measured-state data for one shared clock.

No simulator is loaded. The existing glyph clock and source records are read
without alteration. Native traces preserve sampled peaks; the 10 Hz readouts
use bounded left holds, never interpolate or extend a terminated trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from export_contact_web_data import SOURCES, exact_indices, numeric, read_npz

TOLERANCE_S = 1e-7
PHASES = (
    ("pre_contact", "接触前"),
    ("grasp_transport", "抓持／搬运"),
    ("board_contact", "板面接触"),
    ("post_board", "离板后"),
    ("unknown", "接触状态未知"),
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WRIST_CHANNELS = [
    {"id": label, "label": label, "unit": "m" if i < 3 else "无量纲"}
    for i, label in enumerate(("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32"))
]


def held_indices(time: np.ndarray, display: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return past-only indices and a bounded age/extent validity mask."""
    time = np.asarray(time, dtype=float)
    display = np.asarray(display, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("时间戳必须有限、严格递增，且至少包含两个样本")
    indices = np.clip(np.searchsorted(time, display + TOLERANCE_S, side="right") - 1, 0, len(time) - 1)
    max_age = 1.5 * float(np.median(np.diff(time)))
    valid = (np.isfinite(display) & (display >= time[0] - TOLERANCE_S)
             & (display <= time[-1] + TOLERANCE_S)
             & (display - time[indices] <= max_age + TOLERANCE_S))
    return indices, valid


def hold(time: np.ndarray, values: np.ndarray, display: np.ndarray) -> np.ndarray:
    indices, valid = held_indices(time, display)
    values = np.asarray(values, dtype=float)
    if len(values) != len(time):
        raise ValueError("时间戳与数值长度不一致")
    sampled = values[indices]
    valid = valid.reshape((-1,) + (1,) * (sampled.ndim - 1)) & np.isfinite(sampled)
    return np.where(valid, sampled, np.nan)


def phase_intervals(time: np.ndarray, duration: np.ndarray, phases: np.ndarray) -> list[dict]:
    """Preserve recorded interval weights; do not bridge missing clock coverage."""
    held_indices(time, time)
    if len(duration) != len(time) or len(phases) != len(time):
        raise ValueError("阶段数组长度不一致")
    max_age = 1.5 * float(np.median(np.diff(time)))
    result = []
    for i, (start, width, phase) in enumerate(zip(time, duration, phases)):
        if not np.isfinite(width) or width < 0:
            raise ValueError("阶段时长必须非负且有限")
        if width <= 0:
            continue
        end = min(float(start + width), float(time[-1]), float(start + max_age),
                  float(time[i + 1]) if i + 1 < len(time) else float(time[-1]))
        if end <= start:
            continue
        if str(phase) not in dict(PHASES):
            raise ValueError(f"未知阶段: {phase}")
        if result and result[-1]["id"] == str(phase) and abs(result[-1]["endS"] - start) <= TOLERANCE_S:
            result[-1]["endS"] = end
        else:
            result.append({"startS": float(start), "endS": end, "id": str(phase)})
    return result


def read_reference_state(path: Path, frame_count: int, joint_names: list[str]) -> dict:
    """Match the original review's camera-clock pose/measured-joint contract."""
    if frame_count < 2 or len(joint_names) != 20 or len(set(joint_names)) != 20:
        raise ValueError("专家状态必须包含至少两帧及 20 个唯一关节")
    with h5py.File(path, "r") as file:
        head = file["cameras/head"]
        if len(head["timestamp"]) < frame_count:
            raise ValueError("专家 HDF5 相机行数不足")
        time = np.asarray(head["timestamp"][:frame_count], dtype=float)
        indices = np.asarray(head["state_index"][:frame_count])
        state_time = np.asarray(file["state/timestamp"][:], dtype=float)
        held_indices(time, time)
        held_indices(state_time, state_time)
        expected = np.searchsorted(state_time, time, side="right") - 1
        if indices.dtype.kind not in "iu" or not np.array_equal(indices, expected) or np.any(indices < 0):
            raise ValueError("专家 state_index 必须选择相机时刻最新的非未来状态")
        sites = json.loads(head.attrs["wrist_sites_json"])
        sides = json.loads(head.attrs["side_names_json"])
        right = sites.index("hand_r_base_link_site")
        if sides[right] != "right":
            raise ValueError("专家腕部坐标不是右腕")
        transform = np.asarray(head["world_from_wrist"][:frame_count, right], dtype=float)
        if transform.shape != (frame_count, 4, 4) or not np.isfinite(transform).all():
            raise ValueError("专家腕部变换应为有限的 4×4 矩阵")
        rotation = transform[:, :3, :3]
        if (not np.allclose(transform[:, 3], [0, 0, 0, 1], atol=1e-7)
                or not np.allclose(rotation.swapaxes(1, 2) @ rotation, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("专家腕部旋转矩阵无效")
        wrist = np.concatenate((transform[:, :3, 3], rotation[:, :, 0], rotation[:, :, 1]), axis=1)
        available = [item.decode() if isinstance(item, bytes) else str(item)
                     for item in file["state/joint_names"][:]]
        columns = [available.index(name) for name in joint_names]
        joints = np.asarray(file["state/robot_joint_position"][indices][:, columns], dtype=float)
        if not np.isfinite(joints).all():
            raise ValueError("专家关节状态含非有限值")
    return {"time": time - time[0], "wrist": wrist, "joints": joints,
            "jointStateTime": state_time[indices] - time[0]}


def comparison_at_display(glyph: dict, spatial: dict) -> dict:
    """Compare exactly the samples used by CONTACT_DATA, only with both contacts."""
    valid = np.ones_like(glyph["reference_contact"], dtype=bool)
    distributions = {}
    for source in SOURCES:
        indices, matched = exact_indices(spatial[f"{source}_time_s"], glyph[f"{source}_sample_time_s"])
        valid &= (glyph[f"{source}_normal_valid"] & glyph[f"{source}_contact"]
                  & matched[:, None] & spatial[f"{source}_normal_valid"][indices])
        distributions[source] = spatial[f"{source}_normalized_normal_distribution"][indices]
    distance = np.linalg.norm(glyph["reference_center_local_mm"] - glyph["rollout_center_local_mm"], axis=-1)
    difference = 0.5 * np.sum(np.abs(distributions["reference"] - distributions["rollout"]), axis=(-2, -1))
    valid &= np.isfinite(distance) & np.isfinite(difference)
    return {"centerDistanceMm": np.where(valid, distance, np.nan),
            "distributionDifference": np.where(valid, difference, np.nan)}


def phase_payload(report: dict, intensity: dict) -> dict:
    result = {"definitions": [{"id": key, "label": label} for key, label in PHASES],
              "common": report["common_phases"], "sources": {},
              "semantics": "按物理接触状态分段，不表示有效擦拭或成功；表格为整段原生时长加权统计，不随播放累积。"}
    for source in SOURCES:
        duration = report[source]["phase_duration_s"]
        statistics = {}
        for phase, values in report["quality"]["phase_variability"][source].items():
            grip = values.get("grip_normal_n", {})
            statistics[phase] = {
                "durationS": duration.get(phase), "gripMeanN": grip.get("mean_n"),
                "gripStdN": grip.get("std_n"), "boardMeanN": values.get("board_normal_n", {}).get("mean_n"),
                "fingerContactFraction": [values.get(f"finger_{finger}_normal_n", {}).get("contact_fraction") for finger in FINGERS],
            }
        result["sources"][source] = {
            "intervals": phase_intervals(intensity[f"{source}_time_s"], intensity[f"{source}_duration_s"], intensity[f"{source}_phases"]),
            "durationS": duration, "statistics": statistics,
        }
    return result


def safety_payload(report: dict) -> dict:
    result = {}
    for source in SOURCES:
        result[source] = {}
        for name, key in (("board", "board_normal_n"), ("handBoard", "hand_board_normal_n")):
            recorded = report["quality"]["safety"][source][key]
            result[source][name] = {
                "reportedPeakN": recorded.get("reported_peak_n"),
                "sampledPeakN": recorded.get("sampled_peak_n"),
                "thresholdN": recorded.get("threshold_n"),
                "seriesAvailable": recorded.get("availability") == "available",
                "exceededDurationS": recorded.get("exceeded_duration_s"),
                "reportedPeakTimeS": None,
            }
    return result


def source_payload(source, glyph, spatial, intensity, state, display):
    time = spatial[f"{source}_time_s"]
    if not np.allclose(time, intensity[f"{source}_time_s"], atol=TOLERANCE_S, rtol=0):
        raise ValueError(f"{source}: 力与接触状态不在同一原生时钟")
    sampled_time = glyph[f"{source}_sample_time_s"]
    indices, matched = exact_indices(time, sampled_time)
    if np.any(np.isfinite(sampled_time) & ~matched):
        raise ValueError(f"{source}: 展示帧缺少对应原生触觉样本")
    tangent = glyph[f"{source}_tangent_force_local_n"]
    tangent_valid = glyph[f"{source}_tangent_valid"] & matched[:, None] & np.isfinite(tangent).all(axis=-1)
    normal_valid = spatial[f"{source}_normal_valid"]
    native_tangent_valid = spatial[f"{source}_tangent_valid"]
    center = spatial[f"{source}_center_local_mm"] - glyph["pad_center_local_mm"]
    contact = spatial[f"{source}_contact"]
    native_center_valid = normal_valid & contact & np.isfinite(center).all(axis=-1)
    hand_board = intensity.get(f"{source}_hand_board_normal_n", np.full(len(time), np.nan))
    phase_index, phase_valid = held_indices(time, display)
    state_index, state_valid = held_indices(state["time"], display)
    return {
        "sampleTime": sampled_time,
        "tangentMagnitude": np.where(tangent_valid, np.linalg.norm(tangent, axis=-1), np.nan),
        "tangentVector": [[tangent[i, finger].tolist() if tangent_valid[i, finger] else None
                           for finger in range(5)] for i in range(len(display))],
        "handBoard": hold(time, hand_board, display),
        "wrist": hold(state["time"], state["wrist"], display),
        "joints": hold(state["time"], state["joints"], display),
        "stateSampleTime": np.where(state_valid, state["time"][state_index], np.nan),
        "jointStateTime": hold(state["time"], state["jointStateTime"], display),
        "phase": [str(intensity[f"{source}_phases"][index]) if valid else None
                  for index, valid in zip(phase_index, phase_valid)],
        "native": {
            "time": time,
            "normal": np.where(normal_valid, spatial[f"{source}_normal_force_n"], np.nan),
            "tangentMagnitude": np.where(native_tangent_valid, spatial[f"{source}_tangent_net_force_n"], np.nan),
            "center": [[center[i, finger].tolist() if native_center_valid[i, finger] else None
                        for finger in range(5)] for i in range(len(time))],
            "contact": [[bool(contact[i, finger]) if normal_valid[i, finger] else None
                         for finger in range(5)] for i in range(len(time))],
            "grip": intensity[f"{source}_grip_normal_n"],
            "board": intensity[f"{source}_board_normal_n"], "handBoard": hand_board,
            "cleaned": intensity[f"{source}_signal_cleaned_fraction"],
        },
        "nativeState": state,
    }


def export_data(trial: Path) -> dict:
    glyph = read_npz(trial / "tactile_overall_change.npz")
    spatial = read_npz(trial / "tactile_contact_comparison.npz")
    intensity = read_npz(trial / "contact_intensity.npz")
    trace = read_npz(trial / "review/evaluation_rollout_trace.npz")
    report = json.loads((trial / "contact_intensity.json").read_text())
    review = json.loads((trial / "review/evaluation_evaluation_plots.json").read_text())
    display = glyph["display_time_s"]
    joint_names = trace["joint_names"].tolist()
    if joint_names != review["state"]["joint_names"]:
        raise ValueError("原始图表与模型 trace 关节顺序不一致")
    reference_meta = review["reference"]
    reference_path = Path(reference_meta["source_hdf5"])
    if reference_path.resolve() != Path(report["reference"]["metadata"]["source_hdf5"]).resolve():
        raise ValueError("接触报告与原始图表使用了不同专家 HDF5")
    states = {
        "reference": read_reference_state(reference_path, reference_meta["exported_frames_30hz"], joint_names),
        "rollout": {"time": trace["simulation_time_s"] - trace["simulation_time_s"][0],
                    "wrist": trace["wrist_state"], "joints": trace["hand_joint_position"],
                    "jointStateTime": trace["simulation_time_s"] - trace["simulation_time_s"][0]},
    }
    payload = {
        "version": 1, "time": display, "jointNames": joint_names, "wristChannels": WRIST_CHANNELS,
        "sources": {source: source_payload(source, glyph, spatial, intensity, states[source], display) for source in SOURCES},
        "comparison": comparison_at_display(glyph, spatial),
        "phases": phase_payload(report, intensity), "safety": safety_payload(report),
        "notes": [
            "展示读数沿用 10 Hz 共同时间轴；完整曲线使用原生记录，短峰不因降采样被抹去。",
            "时间配对为最近非未来样本，最大年龄为 1.5 倍中位采样间隔，不跨缺口或观测终点补值。",
            "空间中心距离与归一化分布差异仅在双方当前均接触时定义；差异包含动作进度不同的影响。",
            "切向力是有符号网格力求和后的净向量模长，不是不会相互抵消的接触载荷之和。",
            "模型没有手—板碰撞力曲线，仅有任务维护的峰值；不虚构峰值时间或把缺失曲线补零。",
            "腕部前三维为位置（m），后六维为旋转矩阵前两列（无量纲），不是欧拉角；手关节单位 rad。",
            f"专家腕部／关节与原始评估图使用同一批相机行；这组状态止于 {states['reference']['time'][-1]:.3f} s，触觉记录止于 {spatial['reference_time_s'][-1]:.3f} s。",
            "阶段表格是原生样本按时长加权的整段描述统计，不代表播放至当前时刻的累计指标。",
            "单条专家仅作参考，初始墨迹等条件不同；不生成综合评分或跨种子结论。",
        ],
    }
    return numeric(payload)


def write_asset(payload: dict, output: Path) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("window.CONTACT_DETAILS = " + serialized + ";\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出已存在，拒绝覆盖: {args.output}")
    payload = export_data(args.trial_dir)
    write_asset(payload, args.output)
    print(f"导出 {len(payload['time'])} 个同步时刻；{args.output.stat().st_size:,} 字节：{args.output}")


if __name__ == "__main__":
    main()
