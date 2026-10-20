"""Explicit task display fields for the additive H50 workbench.

Pure display adaptation: no file I/O, simulator imports, acceptance thresholds,
stage inference, force calibration, or expert/model alignment. The caller must
bind every supplied record to the same trial before calling this module.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from statistics import median
import sys


_RECORD_NOTE = "原监视器记录，不是独立复核；缺测不补零，不将位移、深度或角度归一化为任务完成率。"
_SERIES_NOTE = "同条轨迹的原始仿真时钟；不重采样、不拼接不同来源、不用终态回填曲线。"


def _field(key, label, unit, *, minimum=None, maximum=None, source_key=None,
           component=None, vector_length=None, scale=None, source_unit=None,
           note=None):
    result = {"key": key, "label": label, "unit": unit,
              "minimum": minimum, "maximum": maximum}
    if source_key is not None:
        result["source_key"] = source_key
    if component is not None:
        result.update(component=component, vector_length=vector_length)
    if scale is not None:
        result["conversion"] = {"operation": "multiply", "factor": scale,
                                "source_unit": source_unit, "output_unit": unit}
    if note:
        result["note"] = note
    return result


def _final(key, label, unit="", *, paths=None, flag=False, minimum=None, maximum=None,
           scale=None, source_unit=None):
    result = {"key": key, "label": label, "unit": unit,
              "source_paths": list(paths or (f"evaluation.state.{key}",)),
              "kind": "recorded_boolean" if flag else "recorded_number",
              "minimum": minimum, "maximum": maximum}
    if scale is not None:
        result["conversion"] = {"operation": "multiply", "factor": scale,
                                "source_unit": source_unit, "output_unit": unit}
    return result


_BOARD_FIELDS = (
    _field("cleaned_fraction", "清洁进度（仿真墨迹记录值）", "fraction", minimum=0, maximum=1),
    _field("board_normal_force_n", "板擦—白板法向载荷", "N", minimum=0),
    _field("board_tangent_force_n", "板擦—白板切向载荷（记录值）", "N", minimum=0),
    _field("table_support_force_n", "板擦—桌面支撑载荷", "N", minimum=0),
    _field("maximum_lift_m", "累计最大抬升高度", "m", minimum=0),
)
_BOARD_FINAL = (
    _final("cleaned_fraction", "终态清洁进度", "fraction", minimum=0, maximum=1),
    _final("maximum_ink_remaining", "终态最大单元墨迹余量", "fraction", minimum=0, maximum=1),
    _final("maximum_lift_m", "累计最大抬升高度", "m", minimum=0),
    _final("released_on_table", "记录的桌面释放标志", flag=True),
)


def _entry(title, note, fields, primary, finals=()):
    return {"title": title, "task_note": note + " " + _RECORD_NOTE,
            "fields": tuple(fields), "primary_task_keys": tuple(primary),
            "secondary_task_keys": tuple(row["key"] for row in fields if row["key"] not in primary),
            "final_metrics": tuple(finals)}


_CATALOG = {
    "whiteboard-wipe": _entry("倾斜白板擦拭", "清洁值来自仿真墨迹单元，不是视觉面积；板面载荷与手指载荷属于不同接触位置。",
        _BOARD_FIELDS, ("cleaned_fraction", "board_normal_force_n"), _BOARD_FINAL),
    "install-ram": _entry("内存条安装", "插深必须结合姿态与孔口匹配观察；超过目标深度不等于安装完成。机械就位不证明电连接或卡扣锁定。", (
        _field("insertion_depth_m", "内存条插入深度（有符号几何记录）", "m"),
        _field("orientation_error_rad", "内存条—插槽姿态误差", "rad", minimum=0),
        _field("axial_resistance_n", "内存条—插槽轴向阻力（记录值）", "N"),
    ), ("insertion_depth_m", "orientation_error_rad", "axial_resistance_n"), (
        _final("insertion_depth_m", "终态插入深度", "m"),
        _final("orientation_error_rad", "终态姿态误差", "rad", minimum=0),
        _final("aperture_fits", "记录的孔口匹配标志", flag=True),
        _final("bottom_out_confirmed", "记录的触底确认标志", flag=True),
        _final("seated", "记录的机械就位标志", flag=True),
    )),
    "usb-insert": _entry("USB 插入", "insertion_depth_mm 在未对准时被原监视器记为 0，并非原始几何深度为零；机械就位不证明电连接或已松手。", (
        _field("insertion_depth_mm", "对准条件下的插入深度（未对准记 0）", "mm",
               note="aligned_with_socket=false 时源程序写入 0，不是缺测替代，也不是物理插深为零。"),
        _field("orientation_error_deg", "插头—插座姿态误差", "deg", minimum=0),
        _field("axial_resistance_n", "插头—插座轴向阻力（记录值）", "N"),
        _field("socket_normal_load_n", "插头—插座法向接触载荷", "N", minimum=0),
        _field("lateral_error_mm", "插头—插座横向偏移", "mm", minimum=0),
        _field("peak_insertion_depth_mm", "累计最大对准插入深度", "mm", minimum=0),
        _field("socket_penetration_mm", "插头—插座最大穿透量（记录值）", "mm", minimum=0),
        # Older explicitly named records have different units and semantics.
        _field("insertion_depth_m", "插入深度（有符号原始几何记录）", "m"),
        _field("orientation_error_rad", "插头—插座姿态误差", "rad", minimum=0),
    ), ("insertion_depth_mm", "orientation_error_deg", "axial_resistance_n", "socket_normal_load_n",
        "insertion_depth_m", "orientation_error_rad"), (
        _final("insertion_depth_m", "终态轴向位置（原始几何）", "m", paths=("evaluation.insertion.insertion_depth_m", "evaluation.state.insertion_depth_m")),
        _final("maximum_insertion_depth_m", "累计最大对准插入深度", "m", paths=("evaluation.maximum_insertion_depth_m",), minimum=0),
        _final("shell_fits_aperture", "记录的孔口匹配标志", flag=True, paths=("evaluation.insertion.shell_fits_aperture", "evaluation.state.shell_fits_aperture")),
        _final("bottom_out_confirmed", "记录的触底确认标志", flag=True, paths=("evaluation.insertion.bottom_out_confirmed", "evaluation.state.bottom_out_confirmed")),
        _final("seated", "记录的机械就位标志", flag=True, paths=("evaluation.insertion.seated", "evaluation.state.seated")),
    )),
    "usb-unplug": _entry("USB 拔出", "轴向位置、拔出阻力与夹持载荷分别来自插头—插座及拇指／食指—插头。"
        "当前新版只要求完全拔出后连续手持至少 2 秒，无需放回、释放或退手；历史严格放回规则以各条归档协议为准，不用新规则重判旧记录。", (
        _field("insertion_depth_m", "插头有符号轴向位置", "m"),
        _field("extraction_hold_duration_s", "完全拔出后当前连续手持时长", "s", minimum=0,
               note="拔出、无插座接触、右手接触且无桌面接触的当前连续窗口；中断后重新计时，不是累计或历史最大时长。"),
        _field("withdrawal_resistance_n", "插头—插座拔出阻力", "N", minimum=0),
        _field("socket_normal_load_n", "插头—插座法向接触载荷", "N", minimum=0),
        _field("signed_axial_force_n", "插座作用于插头的有符号轴向力", "N"),
        _field("grip_thumb_force_n", "拇指—插头夹持载荷", "N", minimum=0,
               source_key="grip_force_n", component=0, vector_length=2),
        _field("grip_index_force_n", "食指—插头夹持载荷", "N", minimum=0,
               source_key="grip_force_n", component=1, vector_length=2),
    ), ("insertion_depth_m", "extraction_hold_duration_s", "withdrawal_resistance_n", "socket_normal_load_n"), (
        _final("extraction_hold_duration_s", "终态当前连续手持时长", "s", minimum=0),
        _final("extraction_hold_now", "终态当前完全拔出并手持", flag=True),
        _final("extraction_hold_verified", "终态当前连续手持满 2 秒", flag=True),
        *(_final(key, label, flag=True) for key, label in (
            ("initially_seated_verified", "记录的初始就位确认"), ("grasp_verified", "记录的抓握确认"),
            ("fully_extracted", "记录的完全拔出标志"), ("placed_on_table", "记录的桌面放回标志"),
            ("released", "记录的释放标志"), ("hand_retreated", "记录的退手标志"))))),
    "bulb-unscrew": _entry("灯泡旋出与放回", "净旋出角度由原始移除圈数换算；角度、轴向位置与螺纹偏差不是完成百分比；灯泡亮灭和里程碑均为仿真记录。没有连续记录的灯座载荷或扭矩不由终态回填。", (
        _field("angle_removed_deg", "灯泡净旋出角度（逆时针）", "deg",
               source_key="turns_removed", scale=360.0, source_unit="turn"),
        _field("insertion_depth_m", "灯泡有符号轴向位置", "m"),
        _field("thread_error_m", "螺纹约束偏差（有符号）", "m"),
        _field("hand_counterclockwise_torque_nm", "手对灯泡的逆时针扭矩", "N·m"),
        _field("socket_contact_load_n", "灯泡—灯座接触载荷", "N", minimum=0),
        _field("table_contact_load_n", "灯泡—桌面接触载荷", "N", minimum=0),
    ), ("angle_removed_deg", "insertion_depth_m", "thread_error_m"), (
        _final("angle_removed_deg", "终态灯泡净旋出角度（逆时针）", "deg",
               paths=("evaluation.state.turns_removed",), scale=360.0, source_unit="turn"),
        *(_final(key, label, flag=True) for key, label in (
            ("initially_installed_verified", "记录的初始安装确认"), ("grasp_verified", "记录的抓握确认"),
            ("breakaway_verified", "记录的旋松确认"), ("thread_disengaged", "记录的螺纹脱离标志"),
            ("disengagement_valid", "记录的脱离有效性标志"), ("fully_extracted", "记录的完全取出标志"),
            ("placed_on_table", "记录的桌面放回标志"), ("released", "记录的释放标志"),
            ("hand_retreated", "记录的回位退手标志"))))),
    "bulb-screw": _entry("灯泡旋入与释放", "未接合时轴向几何坐标仍可很大，不能据此判定插入完成；接合、就位、拧紧与释放应分别观察。灯泡亮灭不是电气认证。", (
        _field("clockwise_angle_deg", "灯泡净旋入角度（顺时针）", "deg",
               source_key="clockwise_turns", scale=360.0, source_unit="turn"),
        _field("insertion_depth_m", "灯泡有符号轴向位置", "m"),
        _field("hand_clockwise_torque_nm", "手对灯泡的顺时针扭矩", "N·m"),
        _field("thread_contact_load_n", "灯泡—灯座螺纹接触载荷", "N", minimum=0),
        _field("thread_error_m", "螺纹约束偏差（有符号）", "m"),
        _field("lateral_error_m", "灯泡横向偏移", "m", minimum=0),
        _field("tilt_rad", "灯泡倾斜角", "rad", minimum=0),
        _field("backstop_load_n", "灯泡—灯座止挡载荷", "N", minimum=0),
        _field("shoulder_contact_load_n", "灯泡—灯座肩部接触载荷", "N", minimum=0),
        _field("cushion_contact_load_n", "灯泡—灯座缓冲接触载荷", "N", minimum=0),
    ), ("clockwise_angle_deg", "insertion_depth_m", "hand_clockwise_torque_nm", "thread_contact_load_n"), (
        _final("clockwise_angle_deg", "终态灯泡净旋入角度（顺时针）", "deg",
               paths=("evaluation.state.clockwise_turns",), scale=360.0, source_unit="turn"),
        *(_final(key, label, flag=True) for key, label in (
            ("engaged", "记录的螺纹接合状态"), ("seated", "记录的机械就位标志"),
            ("tightening_verified", "记录的拧紧确认"), ("released", "记录的释放标志"),
            ("bulb_lit", "仿真灯泡亮起状态"))))),
    "poker-draw": _entry("卡片抽取与举起", "50 mm 位移阈值不等于到达桌边：原任务还要求桌面支持、平放及足够悬出，随后需对向夹持抬起和朝向保持。", (
        _field("card_displacement_now_mm", "卡片当前朝机器人位移", "mm"),
        _field("card_displacement_peak_mm", "卡片累计最大朝机器人位移", "mm", minimum=0),
        _field("edge_threshold_mm", "位移诊断阈值（不是完整到边条件）", "mm", minimum=0),
    ), ("card_displacement_now_mm", "card_displacement_peak_mm"), (
        _final("contacted", "原任务监视器：曾接触卡片", flag=True, paths=("evaluation.contacted",)),
        _final("edge_reached", "原任务监视器：到达受支撑桌边", flag=True, paths=("evaluation.edge_reached",)),
        _final("lifted", "原任务监视器：对向夹持抬起", flag=True, paths=("evaluation.lifted",)),
        _final("hold_seconds", "终态合格保持时长", "s", paths=("evaluation.hold_seconds",), minimum=0),
        _final("maximum_supported_overhang_fraction", "累计最大受支撑悬出比例", "fraction", paths=("evaluation.maximum_supported_overhang_fraction",), minimum=0, maximum=1),
    )),
    "pick-place": _entry("抓取放置", "仅展示确实记录的任务状态；若 task_metrics 为空，任务响应显示未记录，不由视频、手指载荷或腕部运动推断物体放置进度。",
        (), (), (_final("placed_in_box", "记录的放入盒内标志", flag=True, paths=("evaluation.placed_in_box",)),)),
}
_CATALOG["bulb-screw-inplace"] = _entry(
    "灯泡原位旋入",
    "灯泡初始已在灯座入口啮合，原位抓持后旋入；没有桌面拾取或搬运前置。"
    "实际旋入、承力停转与无支撑释放分别记录；亮灯和回原位均不代替物理完成。",
    (*_CATALOG["bulb-screw"]["fields"],
     _field("rotation_from_initial_deg", "相对实际初态的净旋入角度", "deg",
            source_key="rotation_from_initial_turns", scale=360.0, source_unit="turn"),
     _field("depth_from_initial_m", "相对实际初态的轴向进给", "m")),
    ("rotation_from_initial_deg", "insertion_depth_m", "hand_clockwise_torque_nm"),
    (_final("rotation_from_initial_deg", "终态相对实际初态的净旋入角度", "deg",
            paths=("evaluation.state.rotation_from_initial_turns",), scale=360.0, source_unit="turn"),
     *(_final(key, label, flag=True) for key, label in (
        ("initially_threaded_verified", "初始未到座入牙确认"),
        ("grasp_verified", "原位稳定抓持确认"),
        ("effective_rotation_verified", "有效旋入确认"),
        ("seated", "机械就位"), ("tightening_verified", "承力停转拧紧确认"),
        ("released_stably", "无机器人支撑下稳定就位"),
        ("physical_success", "物理任务完成"),
        ("home_return_verified", "回原位收尾（独立记录）")))))
for _task, _title in (("whiteboard-wipe-horizontal", "水平白板擦拭"),
                      ("whiteboard-wipe-horizontal-line", "水平白板线迹擦拭")):
    _CATALOG[_task] = _entry(_title, "清洁值来自仿真墨迹单元；本任务还要求合格拾取、放回释放及手臂／手部回位，不能直接套用倾斜白板的完成条件。",
        _BOARD_FIELDS, ("cleaned_fraction", "board_normal_force_n"), (*_BOARD_FINAL,
        _final("pickup_verified", "记录的拾取确认标志", flag=True),
        _final("home_return_verified", "记录的回位确认标志", flag=True)))


def task_presentation(task):
    """Return JSON-safe, caller-owned labels; unknown tasks get no guessed fields."""
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a nonempty string")
    entry = _CATALOG.get(task)
    if entry is None:
        return {"title": task, "task_note": "尚无本任务的明确字段映射；保留原记录结果，不套用其他任务语义。",
                "primary_task_keys": [], "secondary_task_keys": [], "final_metrics": [], "supported": False}
    return {key: deepcopy(list(value) if isinstance(value, tuple) else value)
            for key, value in entry.items() if key != "fields"} | {"supported": True}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantity(value, descriptor):
    number = _number(value)
    conversion = descriptor.get("conversion")
    if number is not None and conversion is not None:
        number *= conversion["factor"]
        if not math.isfinite(number):
            number = None
    if number is None or (descriptor.get("minimum") is not None and number < descriptor["minimum"]) or (
            descriptor.get("maximum") is not None and number > descriptor["maximum"]):
        return None
    return number


def _identity(trial, report):
    task = trial.task
    if not isinstance(trial.summary, dict) or not isinstance(report, dict):
        raise ValueError("Task display requires dictionary summary and report")
    declared = trial.summary.get("task")
    reported = (report.get("trial") or {}).get("task")
    if any(value is not None and value != task for value in (declared, reported)):
        raise ValueError("Task display inputs identify different tasks")
    evaluation = trial.summary.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("Summary evaluation must be an object")
    evaluation_task = evaluation.get("task")
    if evaluation_task is not None and evaluation_task != task:
        raise ValueError("Summary evaluation task differs from the trial")
    return task


def _lookup(container, path):
    for key in path.split("."):
        if not isinstance(container, dict) or key not in container:
            return False, None
        container = container[key]
    return True, container


def build_task_summary(trial, report=None):
    """Read true final fields only; a missing final is never a last-known value."""
    task = _identity(trial, report or {})
    rows = []
    for descriptor in task_presentation(task)["final_metrics"]:
        candidates = [(path, value) for path in descriptor["source_paths"]
                      for present, value in [_lookup(trial.summary, path)] if present]
        values = [(value if type(value) is bool else None) if descriptor["kind"] == "recorded_boolean"
                  else _quantity(value, descriptor) for _, value in candidates]
        conflict = len(values) > 1 and any(value != values[0] for value in values[1:])
        value = values[0] if values and not conflict else None
        row = {"key": descriptor["key"], "label": descriptor["label"], "unit": descriptor["unit"],
               "value": value, "status": "conflicting" if conflict else "recorded" if value is not None else "missing",
               "source": "; ".join("summary.json:" + path for path, _ in candidates) or None,
               "semantics": "原监视器终态记录；不是当前游标值，也不是重新认证的任务判定。"}
        if descriptor.get("conversion"):
            row["conversion"] = deepcopy(descriptor["conversion"])
        rows.append(row)
    return rows


def _clock_source(rows, *, frames, factor, warnings):
    name = "frames" if frames else "control_rows"
    if rows is None or (isinstance(rows, (list, tuple)) and not rows):
        return None
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{name} must be a sequence of record objects")
    key = "simulation_time_s" if frames else "sim_time"
    times = [_number(row.get(key)) for row in rows]
    if any(time is None for time in times) or any(b <= a for a, b in zip(times, times[1:])):
        warnings.append(f"{name}: invalid/non-increasing native task clock; no sorting, row dropping or fallback replacement.")
        return {"valid": False, "rows": rows}
    explicit = [row.get("gap_after", False) for row in rows]
    if any(type(gap) is not bool for gap in explicit):
        warnings.append(f"{name}: invalid gap declaration; task series unavailable.")
        return {"valid": False, "rows": rows}
    deltas = [b - a for a, b in zip(times, times[1:])]
    nominal = median(deltas) if deltas else None
    tolerance = max(1e-10, max(abs(t) for t in times) * sys.float_info.epsilon * 8)
    gaps = [explicit[i] or (i < len(deltas) and deltas[i] > factor * nominal + tolerance)
            for i in range(len(times))]
    return {"valid": True, "rows": rows, "times": times, "gaps": gaps,
            "nominal": nominal, "name": name}


def _tasks(source, *, frames):
    return [row.get("task_metrics" if frames else "task", {}) for row in source["rows"]]


def augment_task_visualization(trial, report, visual, *, frame_rows=None):
    """Replace task_series using the task's whitelist, keeping each native clock.

    ``frame_rows`` are raw same-episode frames (simulation_time_s/task_metrics).
    They may supply fields wholly absent from control_rows, never fill holes in
    an existing field or replace a present field with an invalid control clock.
    The return value is workbench config metadata; report and trial are untouched.
    """
    task = _identity(trial, report)
    presentation = task_presentation(task)
    result = {"presentation": presentation, "task_summary": build_task_summary(trial, report),
              "review_status": "not_verified", "limitations": [_RECORD_NOTE, _SERIES_NOTE], "warnings": []}
    warnings = result["warnings"]
    origin = _number(visual.get("time_origin_s"))
    visual["task_series"] = []
    if task not in _CATALOG:
        result["task_series_status"] = "unsupported"
        return result
    if not bool(getattr(trial, "task_tactile_clock_aligned", False)) or visual.get("clock_aligned") is False or origin is None:
        warnings.append("Task/tactile common simulation clock or display origin is unverified; synchronized task curves unavailable.")
        result["task_series_status"] = "unverified_clock"
        return result
    factor = _number((report.get("protocol") or {}).get("max_gap_factor", 1.5))
    if factor is None or factor < 1:
        raise ValueError("max_gap_factor must be finite and >= 1")
    control = _clock_source(trial.control_rows, frames=False, factor=factor, warnings=warnings)
    frames = _clock_source(frame_rows, frames=True, factor=factor, warnings=warnings)
    for descriptor in _CATALOG[task]["fields"]:
        source_key = descriptor.get("source_key", descriptor["key"])
        chosen, values, from_frames = None, None, False
        for source, is_frame in ((control, False), (frames, True)):
            if source is None:
                continue
            records = _tasks(source, frames=is_frame)
            if any(isinstance(row, dict) and source_key in row for row in records):
                chosen, values, from_frames = source, records, is_frame
                break
        if chosen is None or not chosen["valid"]:
            continue
        numbers = []
        for row in values:
            value = row.get(source_key) if isinstance(row, dict) else None
            if "component" in descriptor:
                value = value[descriptor["component"]] if isinstance(value, (list, tuple)) and len(value) == descriptor["vector_length"] else None
            numbers.append(_quantity(value, descriptor))
        source_suffix = source_key + (f"[{descriptor['component']}]" if "component" in descriptor else "")
        series = {"key": descriptor["key"], "label": descriptor["label"], "unit": descriptor["unit"],
                  "time_s": [time - origin for time in chosen["times"]], "values": numbers,
                  "gap_after": list(chosen["gaps"]), "nominal_interval_s": chosen["nominal"],
                  "status": "available" if all(v is not None for v in numbers) else "partial" if any(v is not None for v in numbers) else "missing",
                  "source": ("frames.task_metrics." if from_frames else "control_rows.task.") + source_suffix,
                  "sampling": _SERIES_NOTE,
                  "source_records": [chosen["rows"][i].get("_source") for i in (0, -1)]}
        if descriptor.get("conversion"):
            series["conversion"] = deepcopy(descriptor["conversion"])
        if descriptor.get("note"):
            series["note"] = descriptor["note"]
        visual["task_series"].append(series)
    series_statuses = [row["status"] for row in visual["task_series"]]
    result["task_series_status"] = (
        "not_recorded" if not series_statuses else "available" if all(status == "available" for status in series_statuses)
        else "missing" if all(status == "missing" for status in series_statuses) else "partial")
    if not visual["task_series"]:
        result["limitations"].append("此轨迹没有可同步的任务响应数值记录；保留终态，不绘制零曲线或推断阶段。")
    return result


__all__ = ["task_presentation", "build_task_summary", "augment_task_visualization"]
