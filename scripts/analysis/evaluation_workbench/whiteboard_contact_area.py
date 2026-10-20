"""Display SHA-bound recorded whiteboard area estimates, without replaying physics.

The rigid felt geometry estimate is not a deformable true-contact-area sensor.
Neither tactile forces nor ink coverage are used to invent contact square metres.
Contact-weighted averages are copied from the archived physics-dt integrator;
they must not be recomputed from this lower-rate display sample clock.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from statistics import median

from evaluation_unified.staged import load_staged_result
from .goal_references import (
    _Unavailable, _check_identity, _one_evidence, _read_bound_row, _read_bytes,
    _source_inventory,
)


SCHEMA_VERSION = "whiteboard-contact-area-v1"
_TASKS = {"whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}
_GEOMETRY_PATH = "src/kaihand_tactile_env/shared/whiteboard_geometry_metrics.py"
_GEOMETRY_SHA = "125e9e6607f04ac01d063eec2b0afc0af5ba2f99d21f91953c9152d7e2eb8354"
_METRIC_SCHEMA = "kaihand-whiteboard-geometric-area-v1"
_DENOMINATOR = "compiled_felt_mesh_flat_bottom_polygon_area"
_INTEGRATION = "physics_dt_weighted_zero_order_hold_of_sampled_geometry_with_physics_rate_loaded_contact_gate"
_MANIFEST_ROLES = {"staged_deployment_manifest", "deployment_manifest", "model_crop_deployment_manifest"}
_NOTE = (
    "刚性毡底几何贴合估计，不是真实形变接触面积，也不是墨迹累计覆盖率。"
    "以编译后的完整毡底平面多边形面积为分母；真实板面承载接触、法线、表面间隙及有限板内重叠共同筛选。"
    "未接触时瞬时曲线留空；有接触但几何不贴合时保留真实 0。贴合比没有通用成功阈值。"
)
_MEAN_NOTE = (
    "截至当前的接触期间时间加权平均：∫有效贴合面积 dt /（完整毡底面积 × 板面承载接触时长）。"
    "直接展示原物理 dt 积分记录，不从 30 Hz 曲线重新平均；从未发生板面承载接触时不适用。"
)


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fraction(value):
    value = _number(value)
    return value if value is not None and 0 <= value <= 1 else None


def _metric_snapshot(metrics, *, denominator=None):
    if metrics is None or metrics == {}:
        return None
    if (not isinstance(metrics, dict) or metrics.get("schema_version") != _METRIC_SCHEMA
            or metrics.get("measurement_type") != "loaded_rigid_mesh_geometric_area_estimate"
            or metrics.get("is_true_deformable_contact_area") is not False
            or metrics.get("area_denominator") != _DENOMINATOR
            or metrics.get("area_integration") != _INTEGRATION):
        raise _Unavailable("Unreviewed or conflicting recorded contact-area measurement semantics")
    full = _number(metrics.get("full_pad_bottom_area_m2"))
    if full is None or full <= 0:
        raise _Unavailable("Recorded contact-area denominator is absent or invalid")
    if denominator is not None and not math.isclose(full, denominator, rel_tol=1e-12, abs_tol=1e-15):
        raise _Unavailable("Recorded felt-bottom denominator changed within the same trial")
    return metrics


def _row_snapshot(metrics):
    """Missing contact states stay unknown; an unladen source zero is not quality."""
    if metrics is None:
        return None, None
    instantaneous = _fraction(metrics.get("effective_pad_contact_area_ratio"))
    if metrics.get("pad_board_loaded_contact") is not True:
        instantaneous = None
    area = _number(metrics.get("effective_pad_contact_area_m2"))
    full = _number(metrics.get("full_pad_bottom_area_m2"))
    if instantaneous is not None and (
            area is None or area < 0 or not math.isclose(instantaneous, area / full,
                                                       rel_tol=1e-10, abs_tol=1e-12)):
        instantaneous = None
    duration = _number(metrics.get("pad_board_contact_duration_s"))
    average = (_fraction(metrics.get("contact_time_weighted_mean_effective_area_ratio"))
               if duration is not None and duration > 0 else None)
    return instantaneous, average


def _series(key, label, times, values, gaps, interval, source, note):
    return {"key": key, "label": label, "unit": "fraction", "time_s": list(times),
            "values": values, "gap_after": list(gaps), "nominal_interval_s": interval,
            "status": "available" if all(v is not None for v in values) else (
                "partial" if any(v is not None for v in values) else "missing"),
            "source": source, "sampling": "Archived native staged observer clock; no sorting, resampling, interpolation or final-state backfill.",
            "note": note, "presentation_extension": SCHEMA_VERSION}


def _summary(key, label, unit, value, source, note, *, status=None, recorded_value=None):
    result = {"key": key, "label": label, "unit": unit, "value": value,
              "status": status or ("recorded" if value is not None else "missing"),
              "source": source, "semantics": note, "presentation_extension": SCHEMA_VERSION}
    if recorded_value is not None:
        result["recorded_value"] = recorded_value
    return result


def _bound_result(task, report, provenance):
    """Recheck archived import and pin every large-log byte to the parent report."""
    source = report.get("trial", {}).get("source")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise _Unavailable("Recorded trial has no explicit source directory")
    loaded = load_staged_result(Path(source), include_state_log=True)
    aliases = {
        "staged_summary": {"staged_summary", "prediction_horizon_summary"},
        "staged_result": {"staged_result"},
        "staged_deployment_manifest": _MANIFEST_ROLES,
        "staged_state_log": {"staged_state_log"},
    }
    for witness in loaded["provenance"]:
        expected = _one_evidence(report, aliases[witness["role"]])
        if any(witness.get(key) != expected.get(key) for key in ("sha256", "size")):
            raise _Unavailable("Archived contact-area evidence differs from its report SHA binding: " + witness["role"])
        provenance.append({**witness, "role": "contact_area_" + witness["role"]})
    if loaded["staged_result"] != report.get("recorded_staged_evaluation", {}).get("staged_result"):
        raise _Unavailable("Archived staged measurements differ from the preserved report")
    declared_clock = report.get("recorded_staged_evaluation", {}).get("data_evidence", {})
    for key in ("status", "clock", "sample_count", "time_start_s", "time_end_s"):
        if declared_clock.get(key) != loaded["data_evidence"].get(key):
            raise _Unavailable("Recorded staged clock evidence differs from the preserved report: " + key)
    manifest_row = _one_evidence(report, _MANIFEST_ROLES)
    manifest = _read_bound_row(manifest_row, "contact_area_deployment_manifest", provenance)
    summary_row = _one_evidence(report, {"staged_summary", "prediction_horizon_summary"})
    summary = _read_bound_row(summary_row, "contact_area_trial_summary", provenance)
    inventory = _source_inventory(manifest)
    _check_identity(task, report, summary, manifest, manifest["evaluation_source_sha256"])
    root_value = summary["deployment_integrity"]["evaluation_source"].get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise _Unavailable("No SHA-bound archived geometry source root; current defaults are not a fallback")
    root = Path(root_value).resolve(strict=True)
    geometry = inventory.get(_GEOMETRY_PATH)
    if geometry is None or geometry.get("sha256") != _GEOMETRY_SHA:
        raise _Unavailable("Archived whiteboard geometry measurement source has not been reviewed")
    path = root / _GEOMETRY_PATH
    if not path.resolve(strict=True).is_relative_to(root):
        raise _Unavailable("Archived whiteboard geometry path escapes the frozen source root")
    _, witness = _read_bytes(path, geometry)
    provenance.append({**witness, "role": "contact_area_reviewed_geometry_source"})
    return loaded, summary


def build_whiteboard_contact_area(task: str, report: dict, *, time_origin_s: float | None = None) -> dict:
    """Return additive series/summary, or explicit unavailable without fake zeros.

    ``time_origin_s`` must be the already verified visualization's original
    simulation origin. The function copies recorded values at their own native
    timestamps, not a new aligned/interpolated task or tactile sampling clock.
    """
    result = {"schema_version": SCHEMA_VERSION, "task": task, "status": "not_applicable",
              "series": [], "summary": [], "provenance": [], "warnings": []}
    if task not in _TASKS:
        return result
    result["status"] = "unavailable"
    try:
        if not isinstance(report, dict) or _number(time_origin_s) is None:
            raise _Unavailable("Contact-area display requires a bound report and finite native clock origin")
        loaded, summary = _bound_result(task, report, result["provenance"])
        final = _metric_snapshot(loaded["staged_result"].get("continuous_metrics"))
        if final is None:
            raise _Unavailable("No recorded whiteboard contact-area measurements")
        denominator = final["full_pad_bottom_area_m2"]
        rows = loaded["state_rows"]
        if not rows:
            raise _Unavailable("No recorded whiteboard contact-area samples")
        times = [row["simulation_time_s"] - time_origin_s for row in rows]
        deltas = [right - left for left, right in zip(times, times[1:])]
        nominal = median(deltas) if deltas else None
        gaps = [i < len(deltas) and deltas[i] > nominal * 1.5 + 1e-8 for i in range(len(times))]
        metrics = [_metric_snapshot(row.get("continuous_metrics"), denominator=denominator) for row in rows]
        snapshots = [_row_snapshot(row) for row in metrics]
        source = str(Path(summary["deployment_integrity"]["evaluation_source"]["root"]))
        log_source = str(Path(report["trial"]["source"]) / "staged_states.jsonl")
        result["series"] = [
            _series("effective_pad_contact_area_ratio", "接触时刻贴合面积比（刚性几何估计）", times,
                    [row[0] for row in snapshots], gaps, nominal,
                    log_source + ":continuous_metrics.effective_pad_contact_area_ratio", _NOTE),
            _series("contact_time_weighted_mean_effective_area_ratio", "接触期间累计平均贴合面积比（刚性几何估计）", times,
                    [row[1] for row in snapshots], gaps, nominal,
                    log_source + ":continuous_metrics.contact_time_weighted_mean_effective_area_ratio", _MEAN_NOTE),
        ]
        # This independent native cleaning gate is already recorded in the same
        # stream; never infer the maximum patch residual from the mean cleaning.
        residuals = [((row.get("measurements") or {}).get("native_task_state") or {}).get("maximum_ink_remaining")
                     for row in rows]
        if any(value is not None for value in residuals):
            result["series"].append(_series(
                "maximum_ink_remaining", "最大单元墨迹余量（原生擦净条件）", times,
                [_fraction(value) for value in residuals], gaps, nominal,
                log_source + ":measurements.native_task_state.maximum_ink_remaining",
                "直接展示原监视器记录的最大单元墨迹余量，不由平均清洁进度反推；达到擦净条件仍不等于取放、释放和完整任务成功。"))
        duration = _number(final.get("pad_board_contact_duration_s"))
        duration = duration if duration is not None and duration >= 0 else None
        _, average = _row_snapshot(final)
        final_source = str(Path(report["trial"]["source"]) / "staged_evaluation.json") + ":continuous_metrics."
        result["summary"] = [
            _summary("contact_time_weighted_mean_effective_area_ratio", "终态接触期间平均贴合面积比（几何估计）", "fraction",
                     average, final_source + "contact_time_weighted_mean_effective_area_ratio", _MEAN_NOTE,
                     status="no_contact" if duration == 0 else None,
                     recorded_value=final.get("contact_time_weighted_mean_effective_area_ratio") if duration == 0 else None),
            _summary("pad_board_contact_duration_s", "累计板面承载接触时长", "s", duration,
                     final_source + "pad_board_contact_duration_s", "原物理 dt 积分的板擦—白板承载接触时长，不等于全轨迹时长或有效贴合时长。"),
        ]
        # Status describes provenance verification, not whether the eraser ever
        # touched the board. Per-series availability and summary.no_contact
        # preserve that separate fact, even when every contact sample is null.
        result.update(status="verified",
                      binding={"evaluation_source_sha256": loaded["rule_identity"]["evaluation_source_sha256"],
                               "source_root": source, "geometry_source_sha256": _GEOMETRY_SHA,
                               "source_code_executed": False, "rules_reexecuted": False,
                               "native_values_recomputed": False, "origin_simulation_time_s": time_origin_s,
                               "denominator_m2": denominator, "instantaneous_contact_filter": "pad_board_loaded_contact is true",
                               "average_formula": "recorded integral(A_effective dt)/(A_pad*loaded_contact_duration)",
                               "recorded_geometry_thresholds": deepcopy(final.get("thresholds")),
                               "geometry_sample_hz_requested": final.get("geometry_sample_hz_requested"),
                               "sampling_limitations": final.get("sampling_limitations"),
                               "measurement_type": "loaded_rigid_mesh_geometric_area_estimate",
                               "is_true_deformable_contact_area": False},
                      warnings=[_NOTE, _MEAN_NOTE,
                                "白板归档阶段仅定义拾取；贴合与清洁是附加诊断，不新增完整成功判定。"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
        result.update(status="unavailable", series=[], summary=[
            _summary("contact_time_weighted_mean_effective_area_ratio", "终态接触期间平均贴合面积比（几何估计）", "fraction",
                     None, None, "缺少或冲突的来源证据；不显示伪造的零或百分比。", status="unavailable")],
                      warnings=["未显示白板贴合估计：" + str(error), _NOTE])
    return result


__all__ = ["build_whiteboard_contact_area"]
