"""Render a Chinese, single-time fingertip contact preview from saved arrays.

The caller owns the matplotlib backend/font configuration.  This module uses
archived measurements only and has no dependency on the simulation package.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

_FINGERS = ("拇指", "食指", "中指", "无名指", "小指")
_SOURCES = (("reference", "专家"), ("rollout", "模型"))
_MISSING = {0: "观测无效", 1: "尚未观测", 2: "观测已结束", 3: "采样缺口", 4: "该指数据无效"}
FORCE_CMAP = LinearSegmentedColormap.from_list(
    "contact_force", ["#ffffff", "#f8dfa0", "#efaa40", "#d66d26", "#a12d22", "#57131e"]
)
FORCE_CMAP.set_bad("#e5e8ec")


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def render_snapshot(
    data: dict[str, np.ndarray],
    metadata: dict,
    time_s: float,
    output_path: Path,
) -> dict:
    """Draw one exact archived display time, refusing to replace an output.

    Circle position is translated, never rotated, relative to each fingertip's
    archived pad origin.  Both force encodings share the complete native-data
    maximum.  The separate change panel uses signed local force increments in
    N, with a fixed 2 N arrow display cap and unclipped numerical annotations.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    times = np.asarray(data["display_time_s"], dtype=float)
    matches = np.flatnonzero(np.isclose(times, time_s, atol=1e-8, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"需要唯一且精确的展示时刻；{time_s:g} s 匹配到 {len(matches)} 个样本")
    idx = int(matches[0])
    pad_centers = np.asarray(data["pad_center_local_mm"], dtype=float)
    if pad_centers.shape != (5, 2) or not np.all(np.isfinite(pad_centers)):
        raise ValueError("pad_center_local_mm 必须是有效的 (5, 2) 毫米坐标")

    force_values = []
    grid_values = []
    for source, _ in _SOURCES:
        native = np.asarray(data[f"{source}_native_normal_force_n"], dtype=float)
        force_values.extend(native[np.isfinite(native) & (native >= 0)].tolist())
        grid = np.asarray(data[f"{source}_taxel_xy_mm"], dtype=float)
        if grid.ndim != 3 or grid.shape[0] != 5 or grid.shape[2] != 2:
            raise ValueError(f"{source}_taxel_xy_mm 的维度应为 (5, 单元数, 2)")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{source}_taxel_xy_mm 含无效坐标")
        grid_values.append(grid - pad_centers[:, None, :])
    if not force_values:
        raise ValueError("原始法向力没有有效样本，无法建立统一色标")
    normal_max = max(force_values)
    # A zero-only archive still needs a well-defined plotting scale.
    scale_max = normal_max if normal_max > 0 else 1.0
    all_grid = np.concatenate(grid_values, axis=1)
    grid_span = float(np.max(np.ptp(all_grid, axis=1)))
    max_radius = max(grid_span * 0.21, 0.1)
    radius_squared_per_n = max_radius**2 / scale_max
    axis_limit = float(np.max(np.abs(all_grid))) + max_radius + 0.6
    threshold = float(metadata.get("contact_threshold_n", 0.05))
    window_s = float(metadata["window_s"])
    arrow_cap = 2.0
    cmap = FORCE_CMAP
    norm = Normalize(vmin=0.0, vmax=scale_max, clip=False)

    fig = plt.figure(figsize=(19.6, 11.0), facecolor="white")
    layout = fig.add_gridspec(
        4, 5, left=0.105, right=0.925, bottom=0.17, top=0.865,
        height_ratios=(2.35, 1.15, 2.35, 1.15), hspace=0.35, wspace=0.12,
    )
    fig.suptitle(f"五指接触强度对照｜专家时间线 {times[idx]:.2f} 秒", fontsize=23, y=0.969)
    fig.text(
        0.515, 0.924,
        "上部看当前接触：圆的位置＝受力中心，面积与颜色＝当前法向力；下部独立展示力的变化",
        ha="center", fontsize=13, color="#344054",
    )
    result: dict = {
        "display_time_s": float(times[idx]),
        "window_s": window_s,
        "normal_force_scale_max_n": scale_max,
        "native_normal_force_max_n": normal_max,
        "circle_radius_squared_mm2_per_n": radius_squared_per_n,
        "arrow_display_cap_n": arrow_cap,
        "sources": {},
    }
    try:
        for source_index, (source, label) in enumerate(_SOURCES):
            row = 2 * source_index
            sample_time = float(data[f"{source}_sample_time_s"][idx])
            previous_time = float(data[f"{source}_previous_sample_time_s"][idx])
            actual_time = f"实采 {sample_time:.3f} s" if np.isfinite(sample_time) else "当前无观测"
            source_result = {
                "sample_time_s": _finite_number(sample_time),
                "previous_sample_time_s": _finite_number(previous_time),
                "fingers": [],
            }
            result["sources"][source] = source_result
            normal = np.asarray(data[f"{source}_normal_force_n"][idx], dtype=float)
            normal_valid = np.asarray(data[f"{source}_normal_valid"][idx], dtype=bool)
            contact = np.asarray(data[f"{source}_contact"][idx], dtype=bool)
            center = np.asarray(data[f"{source}_center_local_mm"][idx], dtype=float)
            normal_delta = np.asarray(data[f"{source}_normal_change_n"][idx], dtype=float)
            normal_delta_valid = np.asarray(data[f"{source}_normal_change_valid"][idx], dtype=bool)
            tangent_delta = np.asarray(data[f"{source}_tangent_change_local_n"][idx], dtype=float)
            tangent_delta_valid = np.asarray(data[f"{source}_tangent_change_valid"][idx], dtype=bool)
            continuous = bool(data[f"{source}_window_continuous"][idx])
            missing_reason = np.asarray(data[f"{source}_missing_reason"][idx], dtype=int)
            first_ax = None
            for finger_index, finger in enumerate(_FINGERS):
                ax = fig.add_subplot(layout[row, finger_index])
                if first_ax is None:
                    first_ax = ax
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-axis_limit, axis_limit)
                ax.set_ylim(-axis_limit, axis_limit)
                ax.tick_params(labelsize=8, length=2, color="#c3c9d2")
                for spine in ax.spines.values():
                    spine.set_color("#d0d5dd")
                if finger_index == 0:
                    ax.set_ylabel("指尖局部 y（mm）", fontsize=10)
                else:
                    ax.tick_params(labelleft=False)
                if source_index == 0:
                    ax.set_title(finger, fontsize=16, pad=11)
                ax.axhline(0, color="#edf0f4", lw=0.8, zorder=0)
                ax.axvline(0, color="#edf0f4", lw=0.8, zorder=0)
                grid = grid_values[source_index][finger_index]
                ax.scatter(grid[:, 0], grid[:, 1], s=9, color="#ced3da", zorder=1)
                fn = float(normal[finger_index])
                observed = bool(normal_valid[finger_index] and np.isfinite(fn) and fn >= 0)
                in_contact = bool(observed and contact[finger_index])
                fn_delta_ok = bool(
                    observed and continuous and normal_delta_valid[finger_index]
                    and np.isfinite(normal_delta[finger_index])
                )
                ft_delta_ok = bool(
                    observed and continuous and tangent_delta_valid[finger_index]
                    and np.all(np.isfinite(tangent_delta[finger_index]))
                )
                dft_norm = float(np.linalg.norm(tangent_delta[finger_index])) if ft_delta_ok else None
                if not observed:
                    ax.add_patch(Rectangle(
                        (0, 0), 1, 1, transform=ax.transAxes, facecolor="#e9edf2",
                        edgecolor="#c5ccd6", hatch="///", linewidth=0.0, zorder=3,
                    ))
                    ax.text(0.5, 0.53, _MISSING.get(int(missing_reason[finger_index]), "观测未知"),
                            ha="center", va="center", transform=ax.transAxes, fontsize=14,
                            color="#667085", zorder=4)
                elif not in_contact:
                    ax.text(0.5, 0.50, "无接触", ha="center", va="center", transform=ax.transAxes,
                            fontsize=16, color="#667085", bbox={"facecolor": "white", "edgecolor": "none"})
                elif np.all(np.isfinite(center[finger_index])):
                    xy = center[finger_index] - pad_centers[finger_index]
                    ax.add_patch(Circle(
                        xy, np.sqrt(fn * radius_squared_per_n), facecolor=cmap(norm(fn)),
                        edgecolor="#9f3a12", linewidth=1.15, zorder=2,
                    ))
                    ax.plot(xy[0], xy[1], "+", color="#263244", ms=6, mew=1.2, zorder=3)
                else:
                    ax.text(0.5, 0.5, "受力中心未知", ha="center", transform=ax.transAxes,
                            fontsize=12, color="#667085")
                if observed:
                    ax.text(0.04, 0.95, f"{fn:.2f} N", ha="left", va="top", transform=ax.transAxes,
                            fontsize=17, weight="bold", color="#202936", zorder=5,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.5})
                ax.text(0.98, 0.02, "x（mm）", ha="right", va="bottom", transform=ax.transAxes,
                        fontsize=8, color="#667085", zorder=5)

                change_ax = fig.add_subplot(layout[row + 1, finger_index])
                change_ax.set_facecolor("#f8fafc" if observed else "#e9edf2")
                change_ax.set_xticks([])
                change_ax.set_yticks([])
                for spine in change_ax.spines.values():
                    spine.set_color("#e4e7ec")
                change_ax.text(0.045, 0.88, f"过去 {window_s:g} s", transform=change_ax.transAxes,
                               fontsize=10, color="#667085")
                delta_fn_text = f"ΔFn {normal_delta[finger_index]:+.2f} N" if fn_delta_ok else "ΔFn 未知"
                delta_fn_color = "#b42318" if fn_delta_ok and normal_delta[finger_index] > 0 else "#175cd3"
                if not fn_delta_ok or abs(normal_delta[finger_index]) < 0.005:
                    delta_fn_color = "#475467"
                change_ax.text(0.045, 0.57, delta_fn_text, transform=change_ax.transAxes,
                               fontsize=12, color=delta_fn_color, weight="bold")
                change_ax.text(
                    0.045, 0.27, f"|ΔFt| {dft_norm:.2f} N" if ft_delta_ok else "ΔFt 未知",
                    transform=change_ax.transAxes, fontsize=11, color="#344054",
                )
                arrow_ax = change_ax.inset_axes([0.62, 0.08, 0.36, 0.85])
                arrow_ax.set_aspect("equal", adjustable="box")
                arrow_ax.set_xlim(-2.35, 2.35)
                arrow_ax.set_ylim(-2.35, 2.35)
                arrow_ax.set_xticks([])
                arrow_ax.set_yticks([])
                for spine in arrow_ax.spines.values():
                    spine.set_visible(False)
                arrow_ax.set_facecolor("none")
                if in_contact and ft_delta_ok:
                    arrow_ax.axhline(0, lw=0.8, color="#d0d5dd")
                    arrow_ax.axvline(0, lw=0.8, color="#d0d5dd")
                    arrow_ax.add_patch(Circle((0, 0), arrow_cap, facecolor="none", edgecolor="#d0d5dd", lw=0.65))
                    arrow_ax.text(1.95, -0.17, "2 N", fontsize=6, ha="right", va="top", color="#98a2b3")
                    if dft_norm is not None and dft_norm > 1e-12:
                        clipped = dft_norm > arrow_cap
                        vector = tangent_delta[finger_index] * min(1.0, arrow_cap / dft_norm)
                        arrow_ax.add_patch(FancyArrowPatch(
                            (0, 0), vector, arrowstyle="-|>", color="#233a59", shrinkA=0, shrinkB=0,
                            mutation_scale=min(15.0, max(0.5, 20.0 * dft_norm)),
                            linewidth=min(1.7, max(0.35, 4.0 * dft_norm)),
                            linestyle="--" if clipped else "-",
                        ))
                        if clipped:
                            arrow_ax.text(0.5, -0.05, "超限虚线", transform=arrow_ax.transAxes,
                                          ha="center", fontsize=7, color="#667085")
                    else:
                        arrow_ax.plot(0, 0, ".", ms=3, color="#233a59")
                else:
                    status = "无接触\n不画箭头" if observed and not in_contact else "变化未知"
                    arrow_ax.text(0.5, 0.5, status, transform=arrow_ax.transAxes,
                                  ha="center", va="center", fontsize=9, color="#667085")
                source_result["fingers"].append({
                    "finger": finger,
                    "normal_force_n": fn if observed else None,
                    "contact": in_contact if observed else None,
                    "normal_change_n": float(normal_delta[finger_index]) if fn_delta_ok else None,
                    "tangent_change_local_n": tangent_delta[finger_index].tolist() if ft_delta_ok else None,
                    "tangent_change_norm_n": dft_norm,
                    "tangent_arrow_clipped": bool(in_contact and ft_delta_ok and dft_norm > arrow_cap),
                })
            position = first_ax.get_position()
            fig.text(0.028, position.y0 + 0.58 * position.height, label, fontsize=24, weight="bold", color="#202936")
            fig.text(0.018, position.y0 + 0.38 * position.height, actual_time, fontsize=10, color="#667085")

        color_ax = fig.add_axes([0.945, 0.43, 0.013, 0.31])
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax)
        colorbar.set_label("当前法向力 Fn（N）", fontsize=11, labelpad=9)
        colorbar.ax.tick_params(labelsize=9)
        fig.text(0.105, 0.115,
                 f"统一尺度：圆面积 ∝ Fn，填色越深当前力越大（双方原始最大值 {normal_max:.2f} N）；圆面积是力值示意，不是实际接触面积。",
                 fontsize=11, color="#344054")
        fig.text(0.105, 0.085,
                 "箭头表示局部切向力增量 ΔFt，单位 N；方向不是中心位移。全图共用 2 N 显示上限，超限虚线，数字保留完整值。",
                 fontsize=11, color="#344054")
        fig.text(0.105, 0.055,
                 f"专家原始时间，不拉伸、不规整；同一秒可处于不同任务进度。Fn ≤ {threshold:g} N 记为无接触；灰色斜纹表示未知，未知不补零。",
                 fontsize=11, color="#667085")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("xb") as output:
            fig.savefig(output, format="png", dpi=150, facecolor="white")
    finally:
        plt.close(fig)
    return result
