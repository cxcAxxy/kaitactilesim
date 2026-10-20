"""Build finite, source-clock-aware visualization data without any file I/O.

Formal metrics are read from the report, never recomputed from display frames.
Every normal-force sample is retained. Only taxel heatmaps have a frame budget,
and they retain every channel's first native total-normal-force maximum when
that is possible within the budget. Sensor gaps are not interpolated.
"""
from __future__ import annotations

from copy import deepcopy
import math
import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adapters import TrialData


SCHEMA_VERSION = "evaluation-visualization-v1"
MAX_SPATIAL_FRAMES = 180
CONTACT_GLYPH_WINDOW_S = 0.1

# Explicit task contracts, not an arbitrary numeric-field guessing heuristic.
# The last two entries are the permitted range, or None for a signed quantity.
_FIELDS = {
    "cleaned_fraction": ("清洁进度（记录值）", "fraction", 0.0, 1.0),
    "board_normal_force_n": ("板擦—白板法向力", "N", 0.0, None),
    "board_tangent_force_n": ("板擦—白板切向力（记录值）", "N", 0.0, None),
    "table_support_force_n": ("台面支撑力", "N", 0.0, None),
    "maximum_lift_m": ("累计最大抬升高度", "m", 0.0, None),
    "insertion_depth_m": ("插入深度（有符号记录值）", "m", None, None),
    "orientation_error_rad": ("姿态误差", "rad", 0.0, None),
    "axial_resistance_n": ("轴向阻力（记录值）", "N", None, None),
    "sponge_plate_center_offset_m": ("海绵与盘中心偏移", "m", 0.0, None),
    "sponge_maximum_plate_radius_m": ("海绵相对盘中心最大半径", "m", 0.0, None),
    "plate_base_mean_support_force_n": ("盘底平均支撑力", "N", 0.0, None),
    "maximum_lift_clearance_m": ("累计最大抬升间隙", "m", None, None),
    "minimum_carry_clearance_m": ("搬运最小间隙", "m", None, None),
    "recent_plate_base_mean_support_force_n": ("近期盘底平均支撑力", "N", 0.0, None),
    "settle_stable_s": ("记录的稳定持续时间", "s", 0.0, None),
}


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _gap_flags(times, factor):
    gaps = np.zeros(len(times), dtype=bool)
    nominal = float(np.median(np.diff(times))) if len(times) > 1 else None
    if nominal is not None:
        tolerance = max(1e-10, float(np.max(np.abs(times))) * np.finfo(float).eps * 8)
        gaps[:-1] = np.diff(times) > factor * nominal + tolerance
    return gaps.tolist(), nominal


def _task_keys(task):
    if task.startswith("whiteboard-wipe"):
        return ("cleaned_fraction", "board_normal_force_n", "board_tangent_force_n",
                "table_support_force_n", "maximum_lift_m")
    if task in ("install-ram", "usb-insert"):
        return ("insertion_depth_m", "orientation_error_rad", "axial_resistance_n")
    if task == "sponge-grasp":
        return ("sponge_plate_center_offset_m", "sponge_maximum_plate_radius_m",
                "plate_base_mean_support_force_n", "maximum_lift_clearance_m",
                "minimum_carry_clearance_m", "recent_plate_base_mean_support_force_n",
                "settle_stable_s")
    return ()


def _task_series(trial, origin, factor, warnings):
    rows = trial.control_rows
    if not rows:
        return []
    source_times = [_number(row.get("sim_time")) if isinstance(row, dict) else None for row in rows]
    if any(value is None for value in source_times):
        warnings.append("Task curve clock contains an invalid/missing sim_time; no rows were sorted, dropped, or interpolated to invent an alignment.")
        return []
    clock = np.asarray(source_times)
    if len(clock) > 1 and np.any(np.diff(clock) <= 0):
        warnings.append("Task curve clock is not strictly increasing; synchronized task curves are unavailable.")
        return []
    gaps, nominal = _gap_flags(clock, factor)
    result = []
    for key in _task_keys(trial.task):
        tasks = [row.get("task") if isinstance(row.get("task"), dict) else {} for row in rows]
        if not any(key in task for task in tasks):
            continue
        label, unit, minimum, maximum = _FIELDS[key]
        values = []
        for task in tasks:
            value = _number(task.get(key))
            if value is not None and ((minimum is not None and value < minimum) or
                                      (maximum is not None and value > maximum)):
                value = None
            values.append(value)
        result.append({
            "key": key, "label": label, "unit": unit,
            "time_s": (clock - origin).tolist(), "values": values,
            "gap_after": gaps, "nominal_interval_s": nominal,
            "status": "available" if all(value is not None for value in values) else
                      "partial" if any(value is not None for value in values) else "missing",
            "source": f"control_rows.task.{key}",
            "sampling": "native recorded task samples; no resampling, terminal-value backfill, or stage-based inference",
        })
    if any(gaps):
        warnings.append("Task curves contain long sampling gaps; lines must break at gap_after and null values.")
    return result


def _annotations(report, origin, warnings):
    scopes, events = [], []
    for row in report.get("scope_intervals", []):
        if not isinstance(row, dict):
            warnings.append("Malformed scope annotation omitted from the visualization; the source report is unchanged.")
            continue
        start, end = _number(row.get("start_s")), _number(row.get("end_s"))
        if start is None or end is None or end < start:
            warnings.append("Scope annotation with an invalid time interval omitted from the visualization.")
            continue
        copied = deepcopy(row)
        copied.update(start_s=start - origin, end_s=end - origin)
        scopes.append(copied)
    for row in report.get("events", []):
        time = _number(row.get("time_s")) if isinstance(row, dict) else None
        if time is None:
            warnings.append("Event annotation without a finite timestamp omitted from the visualization.")
            continue
        copied = deepcopy(row)
        copied["time_s"] = time - origin
        events.append(copied)
    return scopes, events


def _spatial(trial, times, force, budget, warnings):
    result = {
        "status": "missing", "time_s": [], "source_sample_index": [],
        "normal_taxel_force_n": [], "scale_max_n": None,
        "selection": "native sample-index coverage plus each channel's first valid total-normal-force maximum; no interpolation; display selection never changes formal metrics",
        "native_frame_count": len(times), "selected_frame_count": 0,
        "frame_budget": budget, "peak_sample_indices": [],
        "geometry_status": "missing", "taxel_positions_local_m": None,
    }
    raw = trial.normal_taxel_force_n
    if raw is None:
        return result, None
    try:
        grid = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        warnings.append("Spatial force arrays are not numeric; heatmaps are unavailable.")
        return result, None
    if grid.ndim != 4 or grid.shape[:2] != force.shape or min(grid.shape[2:], default=0) < 1:
        warnings.append("Spatial force dimensions do not match [native sample, channel, row, column]; heatmaps are unavailable.")
        return result, None
    valid = np.isfinite(grid) & (grid >= 0)
    clean_grid = np.where(valid, grid, np.nan)
    result["scale_max_n"] = float(grid[valid].max()) if np.any(valid) else None
    if np.any(~valid):
        warnings.append("Missing, nonfinite or negative taxel forces are null; heatmap gaps must not be painted as zero force.")
    coordinates = trial.taxel_positions_local_m
    if coordinates is not None:
        try:
            positions = np.asarray(coordinates, dtype=float)
            expected = (*grid.shape[1:], 3)
            if positions.shape == expected and np.isfinite(positions).all():
                result.update(geometry_status="available", taxel_positions_local_m=positions.tolist())
            else:
                warnings.append("Supplied geometry is incompatible or invalid; no physical coordinates were inferred.")
        except (TypeError, ValueError):
            warnings.append("Supplied geometry is not numeric; no physical coordinates were inferred.")
    if result["geometry_status"] == "missing":
        warnings.append("Heatmaps use taxel row/column indices only: physical geometry is missing, so no millimetre coordinates or physical contact area are inferred.")
    mandatory = set()
    for channel in range(force.shape[1]):
        values = force[:, channel]
        known = np.flatnonzero(np.isfinite(values) & (values >= 0))
        if len(known):
            mandatory.add(int(known[np.argmax(values[known])]))
    result["peak_sample_indices"] = sorted(mandatory)
    if budget == 0:
        result["status"] = "disabled"
        return result, list(grid.shape[2:])
    if len(mandatory) > budget:
        result["status"] = "insufficient_frame_budget"
        warnings.append(f"Spatial budget {budget} cannot retain {len(mandatory)} distinct channel-peak frames; heatmaps are omitted rather than silently dropping required peaks. Full force curves remain available.")
        return result, list(grid.shape[2:])
    selected = set(mandatory)
    # Include a global taxel peak when possible; scale_max_n always covers the
    # whole native grid, even when a display frame cannot fit in the budget.
    extra = []
    if np.any(valid):
        extra.append(int(np.unravel_index(np.argmax(np.where(valid, grid, -np.inf)), grid.shape)[0]))
    extra.extend([0, len(times) - 1])
    extra.extend(np.rint(np.linspace(0, len(times) - 1, min(budget, len(times)))).astype(int).tolist())
    for index in extra:
        if len(selected) >= budget:
            break
        selected.add(index)
    indices = sorted(selected)
    result.update(
        status="available" if np.all(valid) else "partial" if np.any(valid) else "missing",
        time_s=times[indices].tolist(), source_sample_index=indices,
        normal_taxel_force_n=clean_grid[indices].tolist(), selected_frame_count=len(indices),
    )
    return result, list(grid.shape[2:])


def _empty_contact_glyph():
    return {
        "status": "missing", "time_s": [], "gap_after": [],
        "nominal_interval_s": None, "window_s": CONTACT_GLYPH_WINDOW_S,
        "channels": [], "normal_force_n": [], "center_xy": [],
        "normal_change_n": [], "tangent_change_n": [],
        "center_status": "missing", "tangent_status": "missing",
        "tangent_coordinate_status": "recorded_components", "tangent_axis_labels": ["t₁", "t₂"],
        "force_scale_max_n": None, "normal_change_scale_max_n": None,
        "tangent_change_scale_max_n": None,
        "semantics": {
            "force": "Adapter-provided source-cadence normal force in declared N, unchanged by this view: explicit recorded totals take precedence; the adapter may sum recorded taxels when totals are absent. Source cadence is not necessarily physics cadence. Zero force has no contact center.",
            "center": "Normal-taxel-force-weighted centroid at every native sample; all taxels and positive native total force are required. Not contact area or slip truth.",
            "coordinates": "Recorded local XY is used in mm only with a finite nondegenerate two-dimensional projection; otherwise column/row indices, not inferred millimetres.",
            "change": "Signed difference at exact native t and t-0.1 s endpoints within floating-point tolerance; no interpolation or fixed-frame lag. Any gap or missing channel sample inside the closed window makes the change unknown.",
            "tangent": "Sum signed recorded two-component taxel forces first, then subtract endpoint resultants. Components retain their recorded basis, not an inferred taxel-XY basis; not a slip estimate.",
            "display": "All native samples retained independently of the heatmap frame budget; display quantities do not change formal metrics. Null is unknown, not zero.",
        },
    }


def _native_lag_indices(times, gaps, window):
    """Match actual clock endpoints, excluding any gap inside each window."""
    tolerance = max(1e-10, float(np.max(np.abs(times))) * np.finfo(float).eps * 8)
    targets = times - window
    right = np.searchsorted(times, targets)
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    nearest = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    matched = (nearest < np.arange(len(times))) & (np.abs((times - times[nearest]) - window) <= tolerance)
    gap_prefix = np.concatenate(([0], np.cumsum(np.asarray(gaps, dtype=int))))
    matched &= gap_prefix[np.arange(len(times))] == gap_prefix[nearest]
    return np.where(matched, nearest, -1)


def _native_changes(values, lag_indices):
    """A missing interior sample cannot be hidden by known endpoints."""
    known = np.isfinite(values)
    if values.ndim == 3:
        known = known.all(axis=-1)
    missing_prefix = np.concatenate((np.zeros((1, values.shape[1]), dtype=int),
                                     np.cumsum(~known, axis=0)), axis=0)
    result = np.full(values.shape, np.nan)
    ends = np.flatnonzero(lag_indices >= 0)
    starts = lag_indices[ends]
    complete = missing_prefix[ends + 1] == missing_prefix[starts]
    with np.errstate(over="ignore", invalid="ignore"):
        difference = values[ends] - values[starts]
    if values.ndim == 3:
        complete = complete[..., None]
    result[ends] = np.where(complete & np.isfinite(difference), difference, np.nan)
    return result


def _vector_rows(values):
    return [[pair.tolist() if np.isfinite(pair).all() else None for pair in row] for row in values]


def _status(known):
    return "available" if np.all(known) else "partial" if np.any(known) else "missing"


def _contact_glyph(trial, source_time, force, names, gaps, nominal, warnings):
    result = _empty_contact_glyph()
    count, channels = force.shape
    valid_force = np.isfinite(force)
    result.update(status=_status(valid_force), time_s=(source_time - source_time[0]).tolist(),
                  gap_after=list(gaps), nominal_interval_s=nominal, normal_force_n=force.tolist(),
                  force_scale_max_n=float(force[valid_force].max()) if np.any(valid_force) else None)
    lag = _native_lag_indices(source_time, gaps, CONTACT_GLYPH_WINDOW_S)
    delta = _native_changes(force, lag)
    known_delta = np.isfinite(delta)
    result["normal_change_n"] = delta.tolist()
    result["normal_change_scale_max_n"] = float(np.abs(delta[known_delta]).max()) if np.any(known_delta) else None
    centers = np.full((count, channels, 2), np.nan)
    grid = None
    if trial.normal_taxel_force_n is not None:
        try:
            candidate = np.asarray(trial.normal_taxel_force_n, dtype=float)
            if candidate.ndim == 4 and candidate.shape[:2] == force.shape and min(candidate.shape[2:]) > 0:
                grid = candidate
        except (TypeError, ValueError, OverflowError):
            pass
    positions = None
    if grid is not None and trial.taxel_positions_local_m is not None:
        try:
            candidate = np.asarray(trial.taxel_positions_local_m, dtype=float)
            if candidate.shape == (*grid.shape[1:], 3):
                positions = candidate
        except (TypeError, ValueError, OverflowError):
            pass
    for channel, name in enumerate(names):
        description = {"name": name, "coordinate_status": "index", "unit": "index",
                       "axis_labels": ["column", "row"], "taxel_xy": []}
        if grid is not None:
            rows, columns = grid.shape[2:]
            yy, xx = np.indices((rows, columns))
            xy = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
            if positions is not None and np.isfinite(positions[channel]).all():
                with np.errstate(over="ignore", invalid="ignore"):
                    physical = positions[channel, ..., :2].reshape(-1, 2) * 1000.0
                    centered = physical - physical[0]
                if np.isfinite(centered).all():
                    singular = np.linalg.svd(centered, compute_uv=False)
                    if len(singular) == 2 and singular[1] > max(1e-9, singular[0] * 1e-8):
                        xy = physical
                        description.update(coordinate_status="physical", unit="mm",
                                           axis_labels=["local x (mm)", "local y (mm)"])
            description["taxel_xy"] = xy.tolist()
            weights = grid[:, channel].reshape(count, -1)
            complete = (np.isfinite(weights) & (weights >= 0)).all(axis=1)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                total = weights.sum(axis=1)
                good = complete & np.isfinite(total) & (total > 0) & valid_force[:, channel] & (force[:, channel] > 0)
                centers[good, channel] = (weights[good] / total[good, None]) @ xy
        result["channels"].append(description)
    result["center_xy"] = _vector_rows(centers)
    result["center_status"] = _status(np.isfinite(centers).all(axis=-1))
    tangent_resultants = np.full((count, channels, 2), np.nan)
    if trial.tangent_taxel_force_n is not None:
        try:
            tangent = np.asarray(trial.tangent_taxel_force_n, dtype=float)
            shape_ok = (tangent.ndim == 5 and tangent.shape[:2] == force.shape and
                        tangent.shape[-1] == 2 and min(tangent.shape[2:4]) > 0 and
                        (grid is None or tangent.shape[:-1] == grid.shape))
            if shape_ok:
                complete = np.isfinite(tangent).all(axis=(2, 3, 4))
                with np.errstate(over="ignore", invalid="ignore"):
                    sums = tangent.sum(axis=(2, 3))
                tangent_resultants = np.where(complete[..., None] & np.isfinite(sums), sums, np.nan)
            else:
                warnings.append("Contact glyph tangent dimensions are incompatible; tangent changes are unknown, never inferred from normal force.")
        except (TypeError, ValueError, OverflowError):
            warnings.append("Contact glyph tangent arrays are not numeric; tangent changes are unavailable.")
    result["tangent_status"] = _status(np.isfinite(tangent_resultants).all(axis=-1))
    tangent_delta = _native_changes(tangent_resultants, lag)
    result["tangent_change_n"] = _vector_rows(tangent_delta)
    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = np.hypot(tangent_delta[..., 0], tangent_delta[..., 1])
    known_magnitudes = magnitudes[np.isfinite(magnitudes)]
    result["tangent_change_scale_max_n"] = float(known_magnitudes.max()) if len(known_magnitudes) else None
    return result


def build_visualization(trial: TrialData, report: dict, max_spatial_frames=180) -> dict:
    """Return a finite JSON payload from in-memory evidence, without modifying it.

    Invalid optional arrays do not erase the independently recorded task
    summary. Curves preserve real source timestamps relative to the tactile
    origin. Task curves and event annotations require verified clock alignment;
    they are not aligned merely by subtracting separate starting timestamps.
    """
    if isinstance(max_spatial_frames, (bool, np.bool_)):
        raise ValueError("max_spatial_frames must be a nonnegative integer")
    try:
        budget = operator.index(max_spatial_frames)
    except TypeError as error:
        raise ValueError("max_spatial_frames must be a nonnegative integer") from error
    if budget < 0:
        raise ValueError("max_spatial_frames must be nonnegative")
    warnings = [str(value) for value in list(trial.warnings) + list(report.get("warnings", []))]
    if budget > MAX_SPATIAL_FRAMES:
        warnings.append(f"Spatial frame budget capped at {MAX_SPATIAL_FRAMES}; native force curves are not downsampled.")
    budget = min(budget, MAX_SPATIAL_FRAMES)
    protocol = report.get("protocol") or {}
    factor = _number(protocol.get("max_gap_factor"))
    if factor is None or factor < 1:
        factor = 1.5
        if "max_gap_factor" in protocol:
            warnings.append("Invalid visualization gap factor; conservative default 1.5 used for line breaks only.")
    payload = {
        "schema_version": SCHEMA_VERSION, "time_origin_s": None,
        "channels": [], "tactile_series": {
            "status": "missing", "time_s": [], "normal_force_n": [],
            "gap_after": [], "nominal_interval_s": None,
        },
        "spatial_frames": {
            "status": "missing", "time_s": [], "source_sample_index": [],
            "normal_taxel_force_n": [], "selection": "native frames only; no interpolation",
            "scale_max_n": None, "geometry_status": "missing",
            "taxel_positions_local_m": None, "selected_frame_count": 0,
            "native_frame_count": 0, "frame_budget": budget, "peak_sample_indices": [],
        },
        "contact_glyph": _empty_contact_glyph(),
        "task_series": [], "scope_intervals": [], "events": [],
        "clock_aligned": False,
        "task_summary": {"task": trial.task, **{key: deepcopy(report.get(key)) for key in
                          ("outcome", "safety", "efficiency", "task_specific")}},
        "semantics": {
            "force": "All native normal-force samples in declared newtons; calibration is not independently verified. Null is unknown, not zero.",
            "line_breaks": "Break each channel at null samples and after gap_after=true. Gap detection uses the native median interval and cannot discover regular hidden subsampling.",
            "time": "One recorded tactile origin is subtracted from verified common-clock observations. No time stretching, expert alignment, or separate-origin alignment.",
            "spatial": "Native taxel allocations, possibly solver-kernel estimates; not flexible-skin pressure, validated contact area, or slip truth. Display selection does not alter report metrics.",
            "annotations": "Original report event evidence and ground_truth flags are preserved. Declared stages are not certified task phases; contact episodes are not inferred retries.",
            "missing_tactile": "Missing optional tactile evidence does not remove the recorded task summary or change its outcome.",
        },
        "warnings": warnings,
    }

    def finish():
        payload["warnings"] = list(dict.fromkeys(warnings))
        return _finite_json(payload)

    if trial.time_s is None or trial.normal_force_n is None:
        warnings.append("No native tactile series available; the task summary is preserved without invented curves.")
        return finish()
    try:
        source_time = np.asarray(trial.time_s, dtype=float)
        force = np.asarray(trial.normal_force_n, dtype=float)
    except (TypeError, ValueError):
        warnings.append("Tactile arrays are not numeric; visualization curves are unavailable, while the task summary remains available.")
        return finish()
    if (source_time.ndim != 1 or not len(source_time) or not np.isfinite(source_time).all()
            or np.any(np.diff(source_time) <= 0) or force.ndim != 2
            or force.shape[0] != len(source_time) or force.shape[1] < 1):
        warnings.append("Tactile clock/dimensions are invalid; no samples were silently sorted or interpolated.")
        return finish()
    origin = float(source_time[0])
    relative = source_time - origin
    valid = np.isfinite(force) & (force >= 0)
    clean_force = np.where(valid, force, np.nan)
    names = list(trial.channel_names)
    if len(names) != force.shape[1] or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        names = [f"channel_{index:03d}" for index in range(force.shape[1])]
        warnings.append("Channel names unavailable or inconsistent; display indices are not inferred anatomical identities.")
    gaps, nominal = _gap_flags(source_time, factor)
    payload.update(time_origin_s=origin, clock_aligned=bool(trial.task_tactile_clock_aligned))
    payload["tactile_series"].update(
        status="available" if np.all(valid) else "partial" if np.any(valid) else "missing",
        time_s=relative.tolist(), normal_force_n=clean_force.tolist(), gap_after=gaps,
        nominal_interval_s=nominal, max_gap_factor=factor,
    )
    if np.any(~valid):
        warnings.append("Missing, nonfinite or negative native normal forces are null; per-channel curves must break at these samples.")
    if any(gaps):
        warnings.append("Native tactile clock contains long gaps; connecting those samples would imply unobserved behavior.")
    payload["spatial_frames"], shape = _spatial(trial, relative, clean_force, budget, warnings)
    payload["channels"] = [{"name": name, "grid_shape": shape} for name in names]
    payload["contact_glyph"] = _contact_glyph(trial, source_time, clean_force, names, gaps, nominal, warnings)
    if trial.task_tactile_clock_aligned:
        payload["task_series"] = _task_series(trial, origin, factor, warnings)
        payload["scope_intervals"], payload["events"] = _annotations(report, origin, warnings)
    else:
        warnings.append("Task/tactile clock mapping is unverified; task curves, scopes and events are not overlaid. Their summary evidence remains in the report.")
    return finish()
