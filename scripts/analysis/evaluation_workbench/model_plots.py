"""Generate explicitly model-only native-state PNGs in a new report directory.

This is an additive display fallback, not a comparison reconstruction or a
metric update. Original simulator records and archived reports are read-only.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from evaluation_progress_experiment.adapters import _path
from evaluation_tactile_load_extension.data import _read_trace, _member, _FT_FORMULAS
from evaluation_tactile_load_extension.export import _copy_checked
from evaluation_v1.visualization import _gap_flags
from evaluation_v2.horizon import inspect_prediction_horizon
from .diagnostic_plots import _png
from .export import _digest, _document, _rows, _verify


_PLOTS = (
    ("tactile", "model_right_fingertip_tactile.png", "模型五指法向 / 切向载荷", "五指 Fn 与 |Ft|（N）；仅模型原生记录，无参考线。"),
    ("hand-dof", "model_right_hand_actuated_dof.png", "模型右手 20 个驱动关节", "实际关节角（rad），不是目标或指令。仅模型记录。"),
    ("wrist", "model_right_wrist_state.png", "模型右腕位置与姿态", "世界坐标 xyz（m）与 rot6d 姿态分量（无量纲），不是欧拉角。仅模型记录。"),
)
_RED = "#b73639"


def _joint_names(archive):
    info = archive.getinfo("joint_names.npy")
    if info.file_size > 16_384:
        raise ValueError("Oversized native joint names")
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError("Unsupported native joint-name NPY format")
        shape, _, dtype = reader(stream)
        if (shape != (20,) or dtype.kind not in "SU" or dtype.hasobject or not 0 < dtype.itemsize <= 512
                or info.file_size != stream.tell() + 20 * dtype.itemsize):
            raise ValueError("Expected twenty bounded explicitly named native joints")
    with archive.open(info) as stream:
        values = np.lib.format.read_array(stream, allow_pickle=False)
    names = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    if len(set(names)) != 20 or any(not name or len(name) > 128 for name in names):
        raise ValueError("Missing or duplicate native joint identities")
    return names


def _bound_inputs(trial_dir, report):
    trial = _path(Path(trial_dir).resolve(strict=True))
    identity = report.get("trial", {})
    if (report.get("schema_version") != "task-tactile-evaluation-v1"
            or identity.get("source") != str(trial) or identity.get("prediction_horizon") != 50
            or type(identity.get("prediction_horizon")) is not int):
        raise ValueError("Model plots require the same explicitly H50 source report")
    provenance = _rows(report.get("provenance"))
    # Verify the entire already-declared report snapshot before decoding any
    # member. Output plots never become a replacement for these original rows.
    _verify(provenance)
    by_path = {row.get("resolved_path", row["path"]): row for row in provenance}
    summary_path = trial / "summary.json"
    if str(summary_path) not in by_path:
        raise ValueError("Model plot summary is not SHA-bound to the report")
    summary = _document(summary_path)
    horizon = inspect_prediction_horizon(trial)
    if horizon["status"] != "verified" or horizon["prediction_horizon"] != 50:
        raise ValueError("Model plot source no longer verifies H50")
    for row in horizon["provenance"]:
        if row["path"] not in by_path or by_path[row["path"]]["sha256"] != row["sha256"]:
            raise ValueError("Model plot H50 evidence is not bound to its report")
    if summary.get("task") != identity.get("task") or summary.get("seed") != identity.get("seed"):
        raise ValueError("Model plot task/seed source identity disagrees")
    traces = [row for row in provenance if row.get("role") == "native_control_cadence_tactile"
              and Path(row.get("resolved_path", row["path"])).name == "evaluation_rollout_trace.npz"]
    if len(traces) != 1:
        raise ValueError("Expected one source-SHA-bound native rollout trace")
    trace_row = traces[0]
    trace = _path(trace_row.get("resolved_path", trace_row["path"]))
    if trace not in {trial / "evaluation_rollout_trace.npz", trial / "review/evaluation_rollout_trace.npz"}:
        raise ValueError("Model plot trace is outside the selected trial layout")
    if trace.stat().st_size > 256 * 1024**2:
        raise ValueError("Model plot trace exceeds bounded input size")
    video = summary.get("video")
    comparison = video.get("comparison_plots") if isinstance(video, dict) else None
    if not isinstance(comparison, dict) or comparison.get("schema") != "kaihand-evaluation-comparison-v1":
        raise ValueError("Missing native state/tactile schema binding")
    declared = comparison.get("artifacts", {}).get("rollout_trace")
    if (not isinstance(declared, str) or not Path(declared).is_absolute() or Path(declared).is_symlink()
            or Path(declared).resolve(strict=True) != trace):
        raise ValueError("Declared native trace does not identify the report-bound source")
    review_path = trace.parent / "review.json"
    if str(review_path) not in by_path:
        raise ValueError("Model plot review metadata is not SHA-bound to the report")
    review = _document(review_path)
    if (review.get("comparison_plots") != comparison or review.get("completed") is not True
            or review.get("task") != identity.get("task") or review.get("seed") != identity.get("seed")
            or review.get("checkpoint_sha256") != summary.get("checkpoint_sha256")):
        raise ValueError("Model plot recorder identity/schema differs from the summary")
    return trial, summary, trace, trace_row, comparison, provenance


def _native_arrays(trace, comparison):
    simulation, elapsed, fingers, normal, tangent = _read_trace(trace)
    with zipfile.ZipFile(trace) as archive:
        state = _member(archive, "state_29", shape_tail=(29,))
        wrist = _member(archive, "wrist_state", shape_tail=(9,))
        hand = _member(archive, "hand_joint_position", shape_tail=(20,))
        joints = _joint_names(archive)
    if any(value is None for value in (state, wrist, hand, tangent)):
        raise ValueError("Model-only three-plot fallback requires explicit native state and Fn/Ft members")
    if (len(state) != len(elapsed) or len(wrist) != len(elapsed) or len(hand) != len(elapsed)
            or not np.array_equal(state[:, :9], wrist, equal_nan=True)
            or not np.array_equal(state[:, 9:], hand, equal_nan=True)):
        raise ValueError("Native state29/wrist/hand arrays or clocks disagree")
    schema, tactile = comparison.get("state", {}), comparison.get("tactile", {})
    if (schema.get("layout") != "native_right_wrist_xyz_rot6d_then_20_actuated_hand_dof"
            or schema.get("dimension") != 29 or schema.get("wrist_site") != "hand_r_base_link_site"
            or schema.get("rot6d") != "first rotation column followed by second rotation column"
            or schema.get("joint_names") != joints):
        raise ValueError("Unsupported actual-joint/world-xyz/rotation-column state contract")
    if (tactile.get("unit") != "N" or tactile.get("taxel_grid") != [7, 5]
            or tactile.get("link_names") != fingers or tactile.get("Fn") != "sum of all 7x5 normal taxel forces"
            or tactile.get("Ft") not in _FT_FORMULAS):
        raise ValueError("Unsupported native Fn/Ft formula, unit, or channel contract")
    rollout = comparison.get("rollout", {})
    if (type(rollout.get("samples")) is not int or rollout["samples"] != len(elapsed)
            or rollout.get("source_start_time_s") != float(simulation[0])
            or rollout.get("duration_s") != float(elapsed[-1])):
        raise ValueError("Native trace and recorded rollout clock disagree")
    # NaN creates a plotted gap. Invalid magnitudes are never clamped to zero.
    normal = np.where(np.isfinite(normal) & (normal >= 0), normal, np.nan)
    tangent = np.where(np.isfinite(tangent) & (tangent >= 0), tangent, np.nan)
    wrist = np.where(np.isfinite(wrist), wrist, np.nan)
    hand = np.where(np.isfinite(hand), hand, np.nan)
    return simulation, elapsed, fingers, joints, normal, tangent, wrist, hand


def _render(stage, arrays, gap_after, title):
    # Lazy imports keep generic suite assembly independent of Matplotlib.
    os.environ.setdefault("MPLCONFIGDIR", str(stage / "matplotlib-cache"))
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    _, time, fingers, joints, normal, tangent, wrist, hand = arrays
    starts = [0] + [i + 1 for i, gap in enumerate(gap_after[:-1]) if gap]
    stops = starts[1:] + [len(time)]
    def curve(axis, values, label, unit):
        for start, stop in zip(starts, stops):
            axis.plot(time[start:stop], values[start:stop], color=_RED, linewidth=1.05, linestyle="-")
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("Recorded elapsed time (s)")
        axis.set_ylabel(unit)
        axis.grid(True, alpha=.22)
        if len(time) > 1:
            axis.set_xlim(time[0], time[-1])
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "path.simplify": False}):
        for identifier, name, _, _ in _PLOTS:
            shape, size = {"tactile": ((5, 2), (13, 15)), "hand-dof": ((5, 4), (16, 16)), "wrist": ((3, 3), (13, 10))}[identifier]
            figure = Figure(figsize=size, dpi=120, layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(*shape, squeeze=False)
            figure.suptitle(title + "\nMODEL ONLY - red solid lines; no reference trajectory", fontsize=14)
            if identifier == "tactile":
                for i, finger in enumerate(fingers):
                    label = finger.replace("hand_r_", "").replace("_link", " / link ")
                    curve(axes[i, 0], normal[:, i], label + " / Fn", "N")
                    curve(axes[i, 1], tangent[:, i], label + " / |Ft|", "N")
            elif identifier == "hand-dof":
                for i, joint in enumerate(joints):
                    curve(axes.flat[i], hand[:, i], joint, "rad")
            else:
                labels = ("world x", "world y", "world z", "R11", "R21", "R31", "R12", "R22", "R32")
                for i, label in enumerate(labels):
                    curve(axes.flat[i], wrist[:, i], label, "m" if i < 3 else "dimensionless")
            figure.savefig(stage / name, dpi=120, format="png", metadata={"Title": title, "Description": "Model-only native records. No reference data or resampling."})
            figure.clear()


def generate_model_plots(trial_dir, output_dir, report):
    """Write only three new PNG files under output_dir/plots; never overwrite."""
    trial, summary, trace, trace_row, comparison, provenance = _bound_inputs(trial_dir, report)
    output = Path(output_dir).absolute()
    if output.resolve() != output or output == trial or output.is_relative_to(trial):
        raise ValueError("Model plots must be generated outside immutable source trials without symlinks")
    if (output / "plots").resolve() != output / "plots":
        raise ValueError("Model plot destination directory must not follow a symlink")
    if not output.parent.is_dir():
        raise ValueError("Model plot output parent must already exist")
    targets = [output / "plots" / row[1] for row in _PLOTS]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("Refusing to overwrite existing model plots")
    arrays = _native_arrays(trace, comparison)
    factor = report.get("protocol", {}).get("max_gap_factor", 1.5)
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor < 1:
        raise ValueError("Invalid recorded gap policy")
    gaps, interval = _gap_flags(arrays[1], factor)
    checkpoint = Path(summary["checkpoint_path"]).name
    title = f"{summary['task']} / seed {summary['seed']:03d} / checkpoint {checkpoint} / H50"
    descriptors = []
    with tempfile.TemporaryDirectory(prefix=".model-plots-", dir=output.parent) as temporary:
        stage = Path(temporary)
        _render(stage, arrays, gaps, title)
        generated = []
        for identifier, name, label, description in _PLOTS:
            dimensions, row = _png(stage / name)
            generated.append((stage / name, output / "plots" / name, row))
            descriptors.append({"id": identifier, "label": label, "description": description,
                                "url": "plots/" + name, "sha256": row["sha256"],
                                "width": dimensions[0], "height": dimensions[1]})
        _verify(provenance)
        for source, destination, row in generated:
            _copy_checked(source, destination, row)
        _verify(provenance)
        for descriptor in descriptors:
            if _digest(output / descriptor["url"])["sha256"] != descriptor["sha256"]:
                raise ValueError("Generated model PNG SHA256 changed before publication")
    unknown = int(sum(np.count_nonzero(~np.isfinite(array)) for array in arrays[4:]))
    return {"status": "available", "comparison_kind": "model_only", "task": summary["task"], "seed": summary["seed"],
            "checkpoint": checkpoint, "prediction_horizon": 50, "source_label": title,
            "comparison_note": "仅模型自身的原生记录；没有专家参考线，不用于专家相似度、排名或重新判定成功。",
            "reference_label": "无参考轨迹", "rollout_label": "红色实线 · 仅模型",
            "tactile_note": "每指 |Ft| 是先累加有符号切向单元分量后取长度，不是 ΔFt，也不是全手世界坐标净力。",
            "plots": descriptors, "provenance": provenance,
            "generated": {"kind": "model_only_native_record_plots", "source_trace_sha256": trace_row["sha256"],
                          "source_trace": str(trace), "sample_count": len(arrays[1]), "clock": "native time_s = simulation_time_s - first simulation timestamp",
                          "time_start_s": float(arrays[1][0]), "time_end_s": float(arrays[1][-1]),
                          "nominal_interval_s": interval, "gap_count": sum(gaps), "invalid_values_as_gaps": unknown,
                          "state_layout": comparison["state"]["layout"], "state_dimension": 29,
                          "units": {"joint": "rad", "wrist_xyz": "m", "wrist_rot6d": "dimensionless", "Fn_Ft": "N"},
                          "reference_loaded": False, "resampled": False, "metrics_recomputed": False},
            "warnings": ["Missing/nonfinite values and native time gaps break curves; no values are filled"] if unknown or any(gaps) else []}
