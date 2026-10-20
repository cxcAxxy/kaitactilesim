"""Add archived staged facts to an existing tactile workbench case.

This never imports or replays the archived evaluator. The native outcome and
recorded cumulative milestones remain separate, and use the same recorded
simulation clock rather than expert/model time alignment.
"""
from __future__ import annotations

import math
from statistics import median

from evaluation_v1.protocol import content_hash
from evaluation_workbench.export import _rows
from .staged import load_staged_result


STAGE_LABELS = {
    "picked_up": "拾取并稳定持有", "held_near_socket": "持有并接近插口",
    "threaded_and_seated": "螺纹旋入并就位", "fully_extracted": "完全取出",
    "released_stably_on_table": "稳定放回台面并释放",
    "extracted_and_held_2s": "完全拔出并连续手持 2 秒",
    "picked_up_and_held_near_slot": "拾取并接近内存槽", "inserted_and_seated": "插入并就位",
    "dragged_on_table": "沿台面有效拖动", "stable_grasp": "稳定抓持",
    "terminal_hold_over_3s": "终点稳定持有超过 3 秒",
    "stable_inplace_grasp": "原位稳定抓持", "tightening_verified": "承力停转并确认拧紧",
    "released_stably": "无机器人支撑下稳定就位",
}
SIGNALS = (
    ("lift_from_initial_m", "相对初始位置抬升", "m"),
    ("near_mouth_distance_m", "工作端至插口距离", "m"),
    ("table_clearance_m", "物体台面间隙", "m"),
    ("hand_normal_load_n", "手—物体接触法向载荷", "N"),
    ("table_normal_load_n", "物体—台面接触法向载荷", "N"),
    ("socket_normal_load_n", "物体—插口接触法向载荷", "N"),
    ("linear_speed_m_s", "物体线速度", "m/s"),
    ("angular_speed_rad_s", "物体角速度", "rad/s"),
)
CONDITIONS = (("opposed_grasp_now", "当前对向抓持"), ("support_free_now", "当前脱离支撑"),
              ("seated_now", "当前机械就位"))


def projection(result, origin_s):
    if type(origin_s) not in (int, float) or not math.isfinite(origin_s):
        raise ValueError("Native simulation clock origin must be finite")
    stage = result["staged_result"]
    definitions = stage.get("stage_definitions", {})
    flags = stage.get("stage_flags", stage.get("stages", {}))
    first = stage.get("first_times", {})
    cards = []
    for key, definition in definitions.items():
        name = definition if isinstance(definition, str) else str(definition)
        timestamp = first.get(key)
        timestamp = timestamp - origin_s if type(timestamp) in (float, int) and math.isfinite(timestamp) else None
        cards.append({"key": key, "name": name, "label": STAGE_LABELS.get(name, name),
                      "reached": flags.get(key), "first_time_s": timestamp,
                      "source": "archived staged_evaluation.json; no re-evaluation"})
    rows = result.get("state_rows", [])
    times, measurements, conditions = [], [], []
    for row in rows:
        timestamp = row.get("simulation_time_s", row.get("time_s"))
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("Staged projection requires a finite recorded simulation clock")
        if times and timestamp - origin_s <= times[-1]:
            raise ValueError("Staged projection clock must increase without sorting")
        times.append(timestamp - origin_s)
        values = row.get("measurements") or {}
        measurements.append([values.get(key) for key, _, _ in SIGNALS])
        conditions.append([values.get(key) for key, _ in CONDITIONS])
    interval = median(b - a for a, b in zip(times, times[1:])) if len(times) > 1 else None
    gaps = [bool(i + 1 < len(times) and interval and times[i + 1] - t > interval * 1.5 + 1e-8)
            for i, t in enumerate(times)]
    return {"schema_version": "unified-staged-display-v1", "native_result": result["native_result"],
            "staged_result": stage, "stages": cards, "rule_identity": result["rule_identity"],
            "data_evidence": result["data_evidence"], "warnings": result["warnings"],
            "timeline": {"time_s": times, "signals": [{"key": k, "label": label, "unit": unit} for k, label, unit in SIGNALS],
                         "values": measurements, "conditions": [{"key": k, "label": label} for k, label in CONDITIONS],
                         "condition_values": conditions, "gap_after": gaps, "nominal_interval_s": interval,
                         "clock": "recorded simulation_time_s minus native tactile origin",
                         "origin_simulation_time_s": origin_s, "sampling": "Recorded control-endpoint observations; no interpolation"},
            "semantics": "Native outcome and staged diagnostics are separate. Archived rule/source identity is authoritative; current rules do not rescore old trials."}


def augment_staged_case(case, trial, selection=None):
    result = load_staged_result(trial.root, include_state_log=True)
    clock = getattr(trial, "time_s", None)
    shared_clock = bool(getattr(trial, "task_tactile_clock_aligned", False) and clock is not None and len(clock))
    origin = float(clock[0]) if shared_clock else 0.0
    display = projection(result, origin)
    display["timeline"]["model_clock_mapping"] = "mapped" if shared_clock else "unmapped"
    if not shared_clock:
        display["timeline"].update(clock="independent archived simulation_time_s", origin_simulation_time_s=None)
        display["warnings"] = [*display["warnings"], "阶段记录未与模型触觉游标核验同一时钟；保留独立时间，不联动或推测对齐。"]
    if selection:
        selected = selection.get("selection", {})
        display["source_selection"] = {"kind": selected.get("kind"),
            "group_status": (selected.get("group_status") or {}).get("status"),
            "warnings": selection.get("warnings", [])}
        result["provenance"] = _rows(selection.get("provenance", []) + result["provenance"])
    # Reader witnesses use canonical_path; the web exporter pins aliases with
    # resolved_path. Preserve the reader's bound target rather than resolving
    # an alias anew and silently accepting a different file.
    result["provenance"] = [{**row, "resolved_path": row["canonical_path"]}
                            if row.get("canonical_path") else row for row in result["provenance"]]
    case["config"]["staged_evaluation"] = display
    case["unified_staged"] = {**display, "provenance": result["provenance"]}
    report = case["report"]
    report["recorded_staged_evaluation"] = {k: v for k, v in display.items() if k != "timeline"}
    # The row merger rejects conflicting hashes; the native witness retains
    # its original role/clock fields if the staged reader saw the same file.
    report["provenance"] = _rows(result["provenance"] + report["provenance"])
    case["tactile_loads"]["provenance"] = report["provenance"]
    report["visualization"]["sha256"] = content_hash(case["visualization"])
    return case
