#!/usr/bin/env python3
"""从既有白板评估生成三套离线接触报告，双方完整仿真时间都保留。

旧分析库仅用于读取归档、核验坐标和绘制报告，不启动仿真或模型。
所有产物先在临时目录生成、校验，再以 exclusive-create 复制至评估目录；
不覆盖任何既有文件，不修改原始 summary、控制日志、review 或专家轨迹。
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


DEFAULT_ANALYSIS_SOURCE = Path("/cpfs_infra/user/chenxianchi/code/kaitactilesim/src")
ARTIFACTS = (
    "contact_intensity.json", "contact_intensity.npz", "contact_intensity.png",
    "contact_intensity_fingers.png", "contact_intensity_phases.png",
    "tactile_contact_comparison.json", "tactile_contact_comparison.npz",
    "tactile_contact_timeline.png", "tactile_contact_centers.png",
    "tactile_contact_heatmaps.png", "tactile_overall_change.json",
    "tactile_overall_change.npz", "tactile_overall_change.png",
    "contact_reports_preparation.json",
)
TOLERANCE_S = 1e-9


def analysis_modules(source_root: Path) -> dict:
    root = source_root.resolve(strict=True)
    if not (root / "kaihand_tactile_env/shared/whiteboard_tactile_glyphs.py").is_file():
        raise ValueError(f"分析库不存在：{root}")
    sys.path.insert(0, str(root))
    names = ("whiteboard_contact_intensity", "whiteboard_contact_reference",
             "whiteboard_tactile_comparison", "whiteboard_tactile_reference",
             "whiteboard_tactile_glyphs")
    modules = {name: importlib.import_module(f"kaihand_tactile_env.shared.{name}") for name in names}
    for module in modules.values():
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("当前进程已导入其他版本分析库，请使用独立进程运行")
    return modules


def display_clock(reference_time, rollout_time, snapshots, interval_s=0.1):
    """10 Hz 公共时轴，额外保留双方真实末端与指定快照；不拉伸轨迹。"""
    if not np.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("展示间隔必须有限且大于零")
    native = [np.asarray(time, dtype=float) for time in (reference_time, rollout_time)]
    for time in native:
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("原生时间必须包含至少两个严格递增的有限样本")
        if abs(time[0]) > TOLERANCE_S:
            raise ValueError("轨迹必须使用从零开始的仿真时间")
    end = max(time[-1] for time in native)
    snapshots = np.asarray(snapshots, dtype=float)
    if (snapshots.ndim != 1 or not len(snapshots) or not np.isfinite(snapshots).all()
            or np.any(np.diff(snapshots) <= 0) or np.any(snapshots < 0) or np.any(snapshots > end)):
        raise ValueError("快照必须在完整时域内且严格递增")
    anchors = np.sort(np.r_[snapshots, native[0][-1], native[1][-1]])
    # Merge float-clock roundoff, retaining the true final observation rather
    # than introducing two visually identical 90.0000000000 s display rows.
    retained = []
    for value in anchors:
        if retained and value - retained[-1] <= TOLERANCE_S:
            retained[-1] = value
        else:
            retained.append(value)
    regular = np.arange(0, end + TOLERANCE_S, interval_s)
    regular = regular[regular <= end + TOLERANCE_S]
    for anchor in retained:
        regular = regular[np.abs(regular - anchor) > TOLERANCE_S]
    return np.sort(np.r_[regular, retained])


def build_full_glyph_samples(rollout, reference, glyphs, *, snapshot_times_s,
                            frame_interval_s=0.1, window_s=0.1, contact_threshold_n=0.05):
    """Reuse the vetted past-only sampler, not the expert-only display limit.

    The original builder supplies unchanged scales/provenance/definitions from
    the shared observed interval. Every display-dependent array is then built
    afresh for the union clock and both change scales are recomputed on it.
    """
    times = display_clock(reference.time_s, rollout.time_s, snapshot_times_s, frame_interval_s)
    common = times[(times >= reference.time_s[0]) & (times <= reference.time_s[-1])]
    summary, _ = glyphs.build_glyph_samples(
        rollout, reference, common, window_s=window_s,
        contact_threshold_n=contact_threshold_n,
    )
    archive = {"display_time_s": times}
    sampled = {
        source: glyphs._sample_trace(trace, times, window_s, contact_threshold_n)
        for source, trace in (("reference", reference), ("rollout", rollout))
    }
    for source, values in sampled.items():
        archive.update({f"{source}_{key}": value for key, value in values.items()})
    archive["pad_center_local_mm"] = sampled["reference"]["taxel_xy_mm"].mean(axis=1)
    archive["video_frame_time_s"] = np.array([], dtype=float)
    summary["scales"]["normal_change_abs_max_n"] = glyphs._finite_max(
        [values["normal_change_n"] for values in sampled.values()], absolute=True)
    summary["scales"]["tangent_change_max_n"] = glyphs._finite_max(
        [np.linalg.norm(values["tangent_change_local_n"], axis=-1) for values in sampled.values()])
    summary["display_samples"] = len(times)
    summary["display_end_time_s"] = float(times[-1])
    summary["snapshot_times_s"] = [float(value) for value in snapshot_times_s]
    summary["definitions"]["clock"] = (
        "双方从零开始的实际仿真秒数，展示到双方较晚的真实观测终点；保留双方末端。"
        "不拉伸、不时间规整；任一方结束后该方观测为空，另一方继续。")
    summary["video"] = {"generated": False, "frame_interval_s": frame_interval_s,
                        "fps": 1.0 / frame_interval_s, "frames": 0, "last_frame_time_s": None,
                        "speed": "不生成额外触觉视频；网页按原始场景视频帧记录同步。"}
    validate_glyph_archive(archive, {"reference": reference.time_s, "rollout": rollout.time_s})
    return summary, archive


def validate_glyph_archive(archive: dict, native_times: dict):
    display = archive["display_time_s"]
    if np.any(np.diff(display) <= 0):
        raise ValueError("完整展示时轴必须严格递增")
    if abs(display[-1] - max(time[-1] for time in native_times.values())) > TOLERANCE_S:
        raise ValueError("完整展示时轴未保留较晚终点")
    for source, times in native_times.items():
        sampled = archive[f"{source}_sample_time_s"]
        after = display > times[-1] + TOLERANCE_S
        if (np.isfinite(sampled[after]).any() or archive[f"{source}_normal_valid"][after].any()
                or np.isfinite(archive[f"{source}_normal_force_n"][after]).any()
                or archive[f"{source}_contact"][after].any()):
            raise ValueError(f"{source} 终点之后不可填充观测")
        known = np.isfinite(sampled)
        if np.any(sampled[known] - display[known] > TOLERANCE_S):
            raise ValueError("展示数据不可使用未来样本")
        if len(archive[f"{source}_native_normal_force_n"]) != len(times):
            raise ValueError("原生样本被改变或裁短")
        endpoint = np.flatnonzero(np.isclose(display, times[-1], atol=TOLERANCE_S, rtol=0))
        if len(endpoint) != 1 or not np.isclose(sampled[endpoint[0]], times[-1], atol=TOLERANCE_S, rtol=0):
            raise ValueError(f"{source} 原生末端没有对应展示样本")


def relocate(value, source: Path, target: Path):
    if isinstance(value, dict):
        return {key: relocate(item, source, target) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, source, target) for item in value]
    if isinstance(value, str) and value.startswith(str(source) + "/"):
        return str(target / Path(value).relative_to(source))
    return value


def reject_existing(target: Path):
    existing = [str(target / name) for name in ARTIFACTS if (target / name).exists()]
    if existing:
        raise FileExistsError("不覆盖既有报告：" + ", ".join(existing))


def prepare(trial: Path, analysis_source_root: Path, snapshots=(20., 28.5, 34.)) -> dict:
    trial = trial.resolve(strict=True)
    reject_existing(trial)
    modules = analysis_modules(analysis_source_root)
    contact_reference = modules["whiteboard_contact_reference"]
    tactile_reference = modules["whiteboard_tactile_reference"]
    glyphs = modules["whiteboard_tactile_glyphs"]
    metadata = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
    reference_contact = contact_reference.load_whiteboard_contact_reference(task=metadata["task"])
    rollout_contact = contact_reference.load_historical_whiteboard_contact_trial(trial)
    reference = tactile_reference.load_whiteboard_tactile_reference(task=metadata["task"])
    rollout = tactile_reference.load_historical_whiteboard_tactile_trial(trial, reference)
    reference.metadata["snapshot_times_s"] = list(snapshots)
    if any(time < reference.time_s[0] or time > reference.time_s[-1] for time in snapshots):
        raise ValueError("静态快照必须处于专家实际观测范围；完整动态时轴不受此限制")
    summary, archive = build_full_glyph_samples(
        rollout, reference, glyphs, snapshot_times_s=snapshots)
    with tempfile.TemporaryDirectory(prefix="contact-reports-") as directory:
        stage = Path(directory)
        modules["whiteboard_contact_intensity"].export_contact_intensity(
            stage, rollout_contact, reference_contact)
        modules["whiteboard_tactile_comparison"].export_tactile_comparison(rollout, reference, stage)
        summary["artifacts"] = {key: str(trial / f"tactile_overall_change.{suffix}")
                                for key, suffix in (("metrics", "json"), ("trace", "npz"), ("snapshot_plot", "png"))}
        with glyphs._chinese_plot_style():
            glyphs._plot_snapshots(stage / "tactile_overall_change.png", archive, summary, snapshots)
        np.savez_compressed(stage / "tactile_overall_change.npz", **archive)
        (stage / "tactile_overall_change.json").write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        for path in stage.glob("*.json"):
            data = relocate(json.loads(path.read_text(encoding="utf-8")), stage, trial)
            path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        # The contact NPZ contains a second copy of its report metadata.
        contact_npz = stage / "contact_intensity.npz"
        with np.load(contact_npz, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["metadata_json"] = np.array((stage / "contact_intensity.json").read_text(encoding="utf-8"))
        np.savez_compressed(contact_npz, **arrays)
        report = {
            "schema_version": "contact-reports-preparation-v1", "task": metadata["task"],
            "seed": metadata["seed"], "checkpoint_path": metadata.get("checkpoint_path"),
            "mode": "offline_existing_observations_only", "simulation_rerun": False,
            "original_artifacts_modified": False, "analysis_source_root": str(analysis_source_root.resolve()),
            "display_samples": len(archive["display_time_s"]), "display_end_s": float(archive["display_time_s"][-1]),
            "reference": {"samples": len(reference.time_s), "end_s": float(reference.time_s[-1])},
            "rollout": {"samples": len(rollout.time_s), "end_s": float(rollout.time_s[-1])},
            "snapshot_times_s": list(snapshots),
            "notes": ["完整动态触觉归档保留双方全部时域；较短轨迹结束后缺失，不补零。",
                      "原始空间比较静态图采用专家参照时域；NPZ 中双方原生数据均完整保存。"],
            "artifacts": [str(trial / path.name) for path in sorted(stage.iterdir())],
        }
        (stage / "contact_reports_preparation.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        # Recheck all names after generation, then use exclusive-create for each
        # copy so even a competing writer cannot be overwritten.
        reject_existing(trial)
        for path in sorted(stage.iterdir()):
            if path.name not in ARTIFACTS:
                raise ValueError(f"意外产物：{path.name}")
            with path.open("rb") as source, (trial / path.name).open("xb") as destination:
                shutil.copyfileobj(source, destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--analysis-source-root", type=Path, default=DEFAULT_ANALYSIS_SOURCE)
    parser.add_argument("--snapshot-times-s", nargs=3, type=float, default=[20., 28.5, 34.])
    args = parser.parse_args()
    print(json.dumps(prepare(args.trial_dir, args.analysis_source_root, args.snapshot_times_s),
                     ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
