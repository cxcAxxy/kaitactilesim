"""从既有接触诊断归档生成中文总览与快照，无需导入仿真环境。

示例：python scripts/analysis/preview_contact_strength.py --trial-dir /path/to/seed_001
依赖 numpy、matplotlib；输入为 contact_intensity 与两份 tactile 诊断归档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from contact_snapshot_preview import FORCE_CMAP, render_snapshot
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle

FINGERS = ("拇指", "食指", "中指", "无名指", "小指")
SOURCES = (("reference", "专家", "#2776b7"), ("rollout", "模型", "#db7029"))


def setup_font() -> None:
    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if Path(path).exists():
            fontManager.addfont(path)
            plt.rcParams["font.family"] = FontProperties(fname=path).get_name()
            break
    else:
        raise RuntimeError("需要可用的中文字体（WenQuanYi Zen Hei / Micro Hei）。")
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "savefig.dpi": 180})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key != "metadata_json"}


def interval_ends(time: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """只保持至下一样本或 1.5 个名义间隔；不跨大缺口，不外推终态。"""
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("时间轴需包含至少两个严格递增的有限样本。")
    nominal = float(np.median(np.diff(time)))
    ends = time + np.minimum(np.maximum(duration, 0), nominal * 1.5)
    return np.minimum(ends, np.r_[time[1:], time[-1]])


def bin_force(time, duration, force, valid, edges, threshold):
    """原生样本区间与时间箱精确求交；均值与接触率均按时长加权。"""
    ends = interval_ends(time, duration)
    shape = (len(edges) - 1, force.shape[1])
    weighted, covered, contact = (np.zeros(shape) for _ in range(3))
    for i, (start, stop) in enumerate(zip(time, ends)):
        if stop <= start:
            continue
        good = valid[i] & np.isfinite(force[i]) & (force[i] >= 0)
        lo = max(0, int(np.searchsorted(edges, start, side="right")) - 1)
        hi = min(len(edges) - 1, int(np.searchsorted(edges, stop, side="left")))
        for b in range(lo, hi):
            overlap = max(0.0, min(stop, edges[b + 1]) - max(start, edges[b]))
            covered[b, good] += overlap
            weighted[b, good] += overlap * force[i, good]
            contact[b, good] += overlap * (force[i, good] > threshold)
    mean = np.divide(weighted, covered, out=np.full(shape, np.nan), where=covered > 0)
    fraction = np.divide(contact, covered, out=np.full(shape, np.nan), where=covered > 0)
    coverage = np.clip(covered / np.diff(edges)[:, None], 0, 1)
    return mean, fraction, coverage


def window_stats(time, duration, values, start, stop, threshold=0.05):
    weights = np.maximum(0, np.minimum(interval_ends(time, duration), stop) - np.maximum(time, start))
    good = np.isfinite(values) & (values >= 0) & (weights > 0)
    if not np.any(good):
        return {"known_duration_s": 0, "mean_n": None, "std_n": None, "p95_n": None}
    v, w = values[good], weights[good]
    mean = float(np.average(v, weights=w))
    order = np.argsort(v)
    p95 = v[order[np.searchsorted(np.cumsum(w[order]), w.sum() * 0.95)]]
    return {
        "known_duration_s": float(w.sum()),
        "coverage_fraction": float(w.sum() / (stop - start)),
        "mean_n": mean,
        "std_n": float(np.sqrt(np.average((v - mean) ** 2, weights=w))),
        "p95_n": float(p95),
        "contact_fraction": float(w[v > threshold].sum() / w.sum()),
    }


def summarize(spatial, contact, start, stop, threshold):
    result = {"start_s": start, "end_s": stop, "definition": "共同物理时间窗，原生样本左保持、时长加权；含已知零力。"}
    for source, _, _ in SOURCES:
        t, dt = contact[f"{source}_time_s"], contact[f"{source}_duration_s"]
        stats = {}
        for key in ("grip_normal_n", "board_normal_n"):
            stats[key] = window_stats(t, dt, contact[f"{source}_{key}"], start, stop, threshold)
        ft, fd = spatial[f"{source}_time_s"], spatial[f"{source}_duration_s"]
        force = np.where(spatial[f"{source}_normal_valid"], spatial[f"{source}_normal_force_n"], np.nan)
        stats["fingers"] = {
            finger: window_stats(ft, fd, force[:, i], start, stop, threshold)
            for i, finger in enumerate(FINGERS)
        }
        clean = contact[f"{source}_signal_cleaned_fraction"]
        available = np.flatnonzero((t <= stop + 1e-9) & np.isfinite(clean))
        last = int(available[-1]) if len(available) else None
        stats["last_cleaned_fraction"] = float(clean[last]) if last is not None else None
        stats["last_cleaning_sample_time_s"] = float(t[last]) if last is not None else None
        result[source] = stats
    return result


def unknown_runs(mask):
    padded = np.r_[False, mask, False]
    return zip(np.flatnonzero(np.diff(padded.astype(int)) == 1), np.flatnonzero(np.diff(padded.astype(int)) == -1))


def final_pinky_no_contact(spatial, threshold):
    time = spatial["rollout_time_s"]
    force = spatial["rollout_normal_force_n"][:, 4]
    valid = spatial["rollout_normal_valid"][:, 4]
    absent = valid & np.isfinite(force) & (force >= 0) & (force <= threshold)
    if not absent[-1]:
        return None
    ends = interval_ends(time, spatial["rollout_duration_s"])
    first = len(time) - 1
    while first > 0 and absent[first - 1] and ends[first - 1] >= time[first] - 1e-9:
        first -= 1
    return {"start_s": float(time[first]), "end_s": float(time[-1]), "duration_s": float(time[-1] - time[first])}


def draw_overview(spatial, contact, binned, edges, report, output):
    fig = plt.figure(figsize=(15.8, 11.2), facecolor="white")
    ax = fig.add_axes((0.115, 0.405, 0.725, 0.435))
    board = fig.add_axes((0.115, 0.245, 0.725, 0.12), sharex=ax)
    cleaned = fig.add_axes((0.115, 0.09, 0.725, 0.11), sharex=ax)
    cax = fig.add_axes((0.115, 0.885, 0.53, 0.013))
    norm = Normalize(0, report["force_color_max_n"])
    fig.text(0.115, 0.975, "五指接触强度：专家与模型的完整时间对照", fontsize=21, weight="bold", va="top")
    fig.text(0.115, 0.933, f"颜色＝每 {report['bin_width_s']:g} 秒内的平均法向力；细带深浅＝该时间窗的接触占比；★＝原生采样峰值", fontsize=11.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FORCE_CMAP), cax=cax, orientation="horizontal")
    cb.set_label("五指、双方统一力值色标 / N", fontsize=10, labelpad=2)
    fig.text(0.68, 0.881, "白色：全窗未超过 0.05 N\n灰斜纹：观测结束或有效覆盖不足", fontsize=10, va="bottom", color="#475569")
    ticks, labels, peaks = [], [], {}
    for finger in range(5):
        for side, (source, name, color) in enumerate(SOURCES):
            row = finger * 2 + side
            mean, fraction, coverage = binned[source]
            missing = coverage[:, finger] < 1 - 1e-6
            values = mean[:, finger].copy()
            values[fraction[:, finger] == 0] = 0
            masked = np.ma.array(values[None, :], mask=missing[None, :])
            ax.pcolormesh(edges, [row + 0.06, row + 0.84], masked, cmap=FORCE_CMAP, norm=norm, shading="flat", rasterized=True)
            stripe = np.ma.array(fraction[None, :, finger], mask=missing[None, :])
            ax.pcolormesh(edges, [row + 0.86, row + 0.96], stripe, cmap="Greens", vmin=0, vmax=1, shading="flat", rasterized=True)
            for first, last in unknown_runs(missing):
                ax.add_patch(Rectangle((edges[first], row + 0.06), edges[last] - edges[first], 0.9, facecolor="#e5e8ec", edgecolor="#b7c0ca", hatch="////", linewidth=0))
            raw = np.where(spatial[f"{source}_normal_valid"][:, finger], spatial[f"{source}_normal_force_n"][:, finger], np.nan)
            if np.isfinite(raw).any():
                i = int(np.nanargmax(raw))
                when, peak = float(spatial[f"{source}_time_s"][i]), float(raw[i])
                ax.scatter(when, row + 0.46, marker="*", s=64, facecolor="#152c42", edgecolor="white", linewidth=0.5, zorder=4)
                ax.text(1.02, row + 0.45, f"{peak:.2f} N · {when:.2f} s", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=color)
                peaks[f"{source}_{finger}"] = {"force_n": peak, "time_s": when}
            ticks.append(row + 0.47)
            labels.append(f"{FINGERS[finger]} · {name}")
        ax.axhline(finger * 2 + 2, color="#ccd3da", linewidth=0.65)
    ax.text(1.02, 1.03, "原生采样峰值 / 时刻", transform=ax.transAxes, fontsize=10, color="#475569")
    ax.set_yticks(ticks, labels)
    for label, (_, _, color) in zip(ax.get_yticklabels(), SOURCES * 5):
        label.set_color(color)
    ax.set_ylim(10.02, -0.05)
    ax.set_xlim(edges[0], edges[-1])
    ax.tick_params(axis="both", length=0)
    no_contact = report["final_model_pinky_no_contact"]
    if no_contact is not None and no_contact["duration_s"] >= 1:
        ax.text((no_contact["start_s"] + no_contact["end_s"]) / 2, 9.48,
                f"连续无接触\n{no_contact['duration_s']:.2f} s", ha="center", va="center",
                fontsize=9, color="#6c4b24", linespacing=1.1)
    for source, name, color in SOURCES:
        time = contact[f"{source}_time_s"]
        board.plot(time, contact[f"{source}_board_normal_n"], color=color, linewidth=1.0, label=name)
        cleaned.plot(time, 100 * contact[f"{source}_signal_cleaned_fraction"], color=color, linewidth=1.5, label=name)
    board.set_title("板擦—白板法向力：是否出现较强载荷或短峰", loc="left", fontsize=12, pad=7)
    board.set_ylabel("力 / N")
    board.legend(loc="upper right", ncol=2, fontsize=10, frameon=False)
    board.set_ylim(bottom=0)
    board_values = contact["rollout_board_normal_n"]
    peak_index = int(np.nanargmax(board_values))
    peak_time = float(contact["rollout_time_s"][peak_index])
    board.annotate(f"采样峰值 {board_values[peak_index]:.2f} N",
                   xy=(peak_time, board_values[peak_index]), xytext=(peak_time + 8, board_values[peak_index] * 0.87),
                   fontsize=9, color="#a65226", arrowprops={"arrowstyle": "->", "color": "#b87951", "lw": 0.8})
    cleaned.set_title("清洁进度：接触是否伴随真实任务进展", loc="left", fontsize=12, pad=7)
    cleaned.set_ylabel("已清洁 / %")
    cleaned.set_ylim(-3, 105)
    cleaned.set_xlabel("共享物理时间 / s（双方使用自身同一秒数，分别保留实际终点）", labelpad=6)
    focus = report["common_window"]
    end = report["rollout_end_s"]
    for panel in (ax, board, cleaned):
        panel.axvline(end, color="#c84548", linestyle="--", linewidth=1.15)
        panel.axvline(report["reference_end_s"], color="#2776b7", linestyle="--", linewidth=0.9)
        panel.axvline(focus["start_s"], color="#596f87", linestyle=":", linewidth=0.8)
        for spine in ("top", "right"):
            panel.spines[spine].set_visible(False)
    for panel in (board, cleaned):
        panel.grid(alpha=0.13)
    ax.text(end + 0.6, -0.24, f"模型结束 {end:.2f} s", color="#c84548", fontsize=10)
    r, e = focus["rollout"], focus["reference"]
    fig.text(0.855, 0.350, f"共同区间 {focus['start_s']:g}–{focus['end_s']:.2f} s\n\n板擦平均力\n模型 {r['board_normal_n']['mean_n']:.2f} N\n专家 {e['board_normal_n']['mean_n']:.2f} N", fontsize=10.5, va="top", linespacing=1.65)
    fig.text(0.855, 0.194, f"窗口末清洁进度\n模型 {100*r['last_cleaned_fraction']:.1f}%\n专家 {100*e['last_cleaned_fraction']:.1f}%", fontsize=10.5, va="top", linespacing=1.7)
    fig.text(0.115, 0.019, "参考：task_demo 的单条成功轨迹。专家 100 Hz，模型约 30.3 Hz；峰值仍可能受采样限制。\n同一秒不保证相同动作进度；初始墨迹位置不同。热图用于诊断，不据此判定打滑或失败因果。", fontsize=9.5, color="#5f6e80", va="bottom")
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    report["native_finger_peaks"] = peaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-times-s", nargs="+", type=float, default=[20.0, 28.5, 34.0])
    parser.add_argument("--bin-s", type=float, default=0.1)
    parser.add_argument("--focus-start-s", type=float, default=28.0)
    args = parser.parse_args()
    if not np.isfinite(args.bin_s) or args.bin_s <= 0:
        parser.error("--bin-s 必须为有限正数。")
    trial = args.trial_dir.resolve()
    output = (args.output_dir or trial).resolve()
    sources = [trial / f"{name}.{suffix}" for name in ("tactile_contact_comparison", "tactile_overall_change", "contact_intensity") for suffix in ("npz", "json")]
    spatial = load_npz(trial / "tactile_contact_comparison.npz")
    contact = load_npz(trial / "contact_intensity.npz")
    glyph = load_npz(trial / "tactile_overall_change.npz")
    meta = json.loads((trial / "tactile_overall_change.json").read_text())
    end = float(spatial["rollout_time_s"][-1])
    expert_end = float(spatial["reference_time_s"][-1])
    stop = min(end, expert_end)
    if not np.isfinite(args.focus_start_s) or not 0 <= args.focus_start_s < stop:
        parser.error("关注窗口起点必须位于双方共同观测区间内。")
    for when in args.snapshot_times_s:
        if np.count_nonzero(np.isclose(glyph["display_time_s"], when, rtol=0, atol=1e-8)) != 1:
            parser.error(f"快照时间 {when} 必须唯一匹配既有触觉变化归档的 display_time_s。")
    filenames = ["contact_strength_overview.png", "contact_strength_analysis.json", "contact_strength_overview.npz"]
    snapshot_names = [f"contact_strength_snapshot_{format(t, 'g').replace('.', 'p')}s.png" for t in args.snapshot_times_s]
    filenames.extend(snapshot_names)
    if len(set(filenames)) != len(filenames):
        parser.error("快照时间不得重复。")
    existing = [str(output / name) for name in filenames if (output / name).exists()]
    if existing:
        raise FileExistsError("拒绝覆盖已有产物：" + ", ".join(existing))
    setup_font()
    threshold = float(meta["contact_threshold_n"])
    timeline_end = max(expert_end, end)
    uniform = np.arange(0, timeline_end, args.bin_s)
    edges = np.unique(np.r_[uniform, expert_end, end, timeline_end])
    binned = {}
    for source, _, _ in SOURCES:
        binned[source] = bin_force(spatial[f"{source}_time_s"], spatial[f"{source}_duration_s"], spatial[f"{source}_normal_force_n"], spatial[f"{source}_normal_valid"], edges, threshold)
    maximum = max(float(np.max(spatial[f"{s}_normal_force_n"][spatial[f"{s}_normal_valid"]])) for s, _, _ in SOURCES)
    report = {
        "schema_version": "contact-strength-readable-preview-v1", "language": "zh-CN",
        "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "bin_width_s": args.bin_s, "force_color_max_n": maximum, "contact_threshold_n": threshold,
        "rollout_end_s": end, "reference_end_s": expert_end, "timeline_end_s": timeline_end,
        "definitions": {
            "heatmap": "每个物理时间箱按已知覆盖时长加权的平均 Fn；终止时刻单独切箱，未知覆盖不涂成零力。",
            "contact_stripe": "每箱 Fn > 门限的时间占比；白色力块表示全箱未超过门限，绿色细带为接触占比。",
            "missing": "有效覆盖不足 100% 的时间箱绘灰斜纹，原始均值及覆盖率仍归档。",
            "peak": "直接在原始有效样本中取峰值，与时间箱平均值分别表达。",
            "comparison": "共享物理时间覆盖双方全部观测，各自结束后不外推、不拉伸；比较窗口仅取共同观测段。单示例且初态不同，仅作描述性诊断。",
        },
        "common_window": summarize(spatial, contact, args.focus_start_s, stop, threshold),
        "final_model_pinky_no_contact": final_pinky_no_contact(spatial, threshold),
        "artifacts": {name: str(output / name) for name in filenames},
    }
    output.mkdir(parents=True, exist_ok=True)
    draw_overview(spatial, contact, binned, edges, report, output / filenames[0])
    report["snapshots"] = [render_snapshot(glyph, meta, when, output / name) for when, name in zip(args.snapshot_times_s, snapshot_names)]
    arrays = {"bin_edges_s": edges}
    for source, values in binned.items():
        arrays.update({f"{source}_{key}": value for key, value in zip(("mean_normal_force_n", "contact_fraction", "coverage_fraction"), values)})
    np.savez_compressed(output / filenames[2], **arrays)
    (output / filenames[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"artifacts": report["artifacts"], "common_window": report["common_window"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
