"""Fixed-rule factual windows on one native episode clock, not diagnoses.

All candidates are retained. Preview selection is separate from event discovery.
The caller certifies signal clocks; optional motion must exactly match time_s.
"""

from bisect import bisect_left, bisect_right
import hashlib
import math
from numbers import Real

from evaluation_progress_experiment.metrics import analyze_episode


RULE_VERSION = "action-review-exploration-v1"
DEFAULT_CONFIG = {"min_plateau_s": 2.0, "equality_tolerance": 1e-12, "context_s": 2.0}
LIMITATIONS = [
    "候选片段是固定探索规则产生的事实索引，不是失效、正常、因果或动作质量判定。",
    "清洁比例记录未变化可能包含量化停顿；不证明物理上完全没有清洁。",
    "0% 可能包含接近或尚未开始，100% 可能包含收尾；二者不构成无效动作率。",
    "正力值只是记录值，不认证有效接触；零力值不证明几何分离。",
    "手腕位置变化不等于板擦运动，更不能证明移动经过剩余墨迹区域。",
    "未标记时段不是正常时段；全部候选及缺口保留，预览不改变全程覆盖分母。",
    "同轨迹进度增长片段仅供人工参照，不自动匹配条件或比较优劣。",
    "未使用专家时间对齐、成功筛选、评分或反事实判断。",
]


def _real(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _rules(config):
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    if config and set(config) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown configuration key")
    result = {**DEFAULT_CONFIG, **(config or {})}
    for key, value in result.items():
        result[key] = _real(value, key, nonnegative=True)
    if result["min_plateau_s"] <= 0 or result["equality_tolerance"] >= 1:
        raise ValueError("min_plateau_s must be positive and equality_tolerance must be below 1")
    return {"version": RULE_VERSION, **result,
            "plateau_definition": "Consecutive valid samples with total progress range within equality_tolerance; no gap crossing.",
            "growth_definition": "Consecutive valid adjacent intervals with progress increase greater than equality_tolerance; no joining over flat, unknown, regressing, or gap intervals.",
            "configuration_source": "explicit_caller_config" if config else "fixed_defaults",
            "data_adaptive_thresholds": False}


def _motion(motion, times):
    if motion is None:
        return None
    if not isinstance(motion, dict):
        raise ValueError("motion must be a dictionary")
    supplied = motion.get("time_s")
    positions = motion.get("position_m")
    if not isinstance(supplied, (list, tuple)) or len(supplied) != len(times):
        raise ValueError("Motion timestamps must exactly match episode timestamps")
    if [_real(value, "motion.time_s") for value in supplied] != times:
        raise ValueError("Motion timestamps must exactly match episode timestamps; no interpolation")
    if not isinstance(positions, (list, tuple)) or len(positions) != len(times):
        raise ValueError("motion.position_m must have one position per native timestamp")
    normalized = []
    for point in positions:
        if point is None:
            normalized.append(None)
            continue
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError("Each motion position must be a three-vector or null")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in point):
            raise ValueError("Motion position coordinates must be numeric")
        vector = [float(value) for value in point]
        normalized.append(vector if all(math.isfinite(value) for value in vector) else None)
    gaps = motion.get("gap_after", [False] * len(times))
    if not isinstance(gaps, (list, tuple)) or len(gaps) != len(times) or any(type(value) is not bool for value in gaps):
        raise ValueError("motion.gap_after must contain one boolean per native timestamp")
    return {"position_m": normalized, "gap_after": list(gaps),
            "label": str(motion.get("label", "Recorded wrist position")),
            "object": str(motion.get("object", "wrist")),
            "frame": str(motion.get("frame", "unverified"))}


def _force_stats(series, start, end, key):
    values = [point[key] for point in series[start:end + 1]]
    known = [value for value in values if value is not None]
    weighted = [(series[i][key], series[i + 1]["time_s"] - series[i]["time_s"])
                for i in range(start, end) if not series[i]["gap_after"] and series[i][key] is not None]
    duration = math.fsum(dt for _, dt in weighted)
    mean = math.fsum(value * (dt / duration) for value, dt in weighted) if duration else None
    observation = ("missing" if not known else "partial" if len(known) != len(values) else
                   "all_positive" if min(known) > 0 else "all_zero" if max(known) == 0 else "mixed_zero_and_positive")
    return {"sample_count": len(values), "known_sample_count": len(known),
            "min_n": min(known) if known else None, "max_n": max(known) if known else None,
            "time_weighted_mean_n": mean, "known_duration_s": duration,
            "coverage_fraction": duration / (series[end]["time_s"] - series[start]["time_s"]),
            "positive_sample_count": sum(value > 0 for value in known),
            "zero_sample_count": sum(value == 0 for value in known), "observation": observation,
            "semantics": "Sample extrema include both endpoints; time-weighted mean uses left hold, terminal duration is zero."}


def _motion_stats(series, start, end, motion):
    if motion is None:
        return {"status": "unavailable", "wrist_path_m": None,
                "wrist_displacement_m": None, "known_duration_s": 0.0, "coverage_fraction": 0.0}
    positions = motion["position_m"]
    segments = []
    for i in range(start, end):
        if (series[i]["gap_after"] or motion["gap_after"][i]
                or positions[i] is None or positions[i + 1] is None):
            continue
        distance = math.dist(positions[i], positions[i + 1])
        if math.isfinite(distance):
            segments.append((distance, series[i + 1]["time_s"] - series[i]["time_s"]))
    known_duration = math.fsum(dt for _, dt in segments)
    try:
        path = math.fsum(distance for distance, _ in segments) if segments else None
    except OverflowError:
        path = None
    complete = len(segments) == end - start
    endpoint = math.dist(positions[start], positions[end]) if complete else None
    if endpoint is not None and not math.isfinite(endpoint):
        endpoint = None
    return {"status": "available" if complete else "partial" if segments else "unavailable",
            "label": motion["label"], "object": motion["object"], "frame": motion["frame"],
            "wrist_path_m": path, "wrist_displacement_m": endpoint,
            "known_duration_s": known_duration,
            "coverage_fraction": known_duration / (series[end]["time_s"] - series[start]["time_s"]),
            "semantics": "Wrist path sums known adjacent position displacements only; missing/gap intervals are excluded and partial path is a lower bound. Net displacement is withheld unless the entire window has valid motion intervals. Neither measures object-surface coverage or proves effective wiping."}


def _plateaus(series, tolerance, minimum):
    start = None
    low = high = None
    for i, point in enumerate(series):
        value = point["progress"]
        extends = (start is not None and value is not None and not series[i - 1]["gap_after"]
                   and max(high, value) - min(low, value) <= tolerance)
        if not extends:
            if start is not None and i - 1 > start and series[i - 1]["time_s"] - series[start]["time_s"] >= minimum:
                yield "plateau", start, i - 1
            start = i if value is not None else None
            low = high = value
        else:
            low, high = min(low, value), max(high, value)
    if start is not None and len(series) - 1 > start and series[-1]["time_s"] - series[start]["time_s"] >= minimum:
        yield "plateau", start, len(series) - 1


def _growth(series, tolerance):
    start = None
    for i in range(len(series) - 1):
        left, right = series[i], series[i + 1]
        growing = (not left["gap_after"] and left["progress"] is not None and right["progress"] is not None
                   and right["progress"] - left["progress"] > tolerance)
        if growing and start is None:
            start = i
        if not growing and start is not None:
            yield "growth", start, i
            start = None
    if start is not None:
        yield "growth", start, len(series) - 1


def build_review(episode, motion=None, config=None):
    """Return full factual candidate index, native series, and conserved coverage.

    Optional motion schema: ``{time_s, position_m, gap_after?, label?, object?,
    frame?}``. Coordinates are wrist positions in metres in one unchanged frame;
    matching clock values are checked here, semantic validation is the adapter's
    responsibility. No source is opened and no supplied object is mutated.
    """
    rules = _rules(config)
    normalized = analyze_episode(episode)
    points = normalized["points"]
    times = [point["time_s"] for point in points]
    motion = _motion(motion, times)
    series = [{"time_s": point["time_s"], "progress": point["progress"],
               "board_normal": point["signals"].get("board_normal"),
               "hand_normal_sum": point["signals"].get("hand_normal_sum"),
               "gap_after": point["gap_after"],
               **({"position_m": motion["position_m"][i], "motion_gap_after": motion["gap_after"][i]} if motion else {})}
              for i, point in enumerate(points)]
    candidates = sorted([*_plateaus(series, rules["equality_tolerance"], rules["min_plateau_s"]),
                         *_growth(series, rules["equality_tolerance"])], key=lambda item: (item[1], item[2], item[0]))
    marked = [False] * max(0, len(series) - 1)
    events = []
    for kind, start, end in candidates:
        first, last = series[start], series[end]
        identity = f"{episode['id']}|{kind}|{first['time_s'].hex()}|{last['time_s'].hex()}"
        contact = _force_stats(series, start, end, "board_normal")
        region = "initial_zero" if kind == "plateau" and max(point["progress"] for point in series[start:end + 1]) <= rules["equality_tolerance"] else (
            "terminal_one" if kind == "plateau" and min(point["progress"] for point in series[start:end + 1]) >= 1 - rules["equality_tolerance"] else "intermediate")
        label = "连续清洁比例记录未变化" if kind == "plateau" else "清洁比例记录增长"
        if kind == "plateau" and contact["observation"] == "all_positive":
            label += "，板面力采样均为正"
        if region == "initial_zero":
            label += "（0% 可含接近或尚未开始）"
        elif region == "terminal_one":
            label += "（100% 可含收尾）"
        events.append({
            "id": "event-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            "kind": kind, "label": label, "start_s": first["time_s"], "end_s": last["time_s"],
            "duration_s": last["time_s"] - first["time_s"], "start_index": start, "end_index": end,
            "context_start_s": times[bisect_left(times, first["time_s"] - rules["context_s"])],
            "context_end_s": times[bisect_right(times, last["time_s"] + rules["context_s"]) - 1],
            "progress_start": first["progress"], "progress_end": last["progress"],
            "progress_delta": last["progress"] - first["progress"], "progress_region": region,
            "contact_stats": contact, "hand_stats": _force_stats(series, start, end, "hand_normal_sum"),
            "motion_stats": _motion_stats(series, start, end, motion),
            "interpretation": "Observation only; no task intent, cleaning opportunity, failure, or causal conclusion inferred.",
        })
        for i in range(start, end):
            marked[i] = True
    interval_duration = [right - left for left, right in zip(times, times[1:])]
    gaps = [point["gap_after"] for point in series[:-1]]
    total = math.fsum(interval_duration)
    gap_duration = math.fsum(dt for dt, gap in zip(interval_duration, gaps) if gap)
    marked_duration = math.fsum(dt for dt, selected, gap in zip(interval_duration, marked, gaps) if selected and not gap)
    unmarked_duration = math.fsum(dt for dt, selected, gap in zip(interval_duration, marked, gaps) if not selected and not gap)
    observed = total - gap_duration
    for event in events:
        event["fraction_of_observed_duration"] = event["duration_s"] / observed if observed else None
    ranked = sorted(events, key=lambda event: (-event["duration_s"], event["start_s"], event["id"]))
    positive_plateaus = [event for event in ranked if event["kind"] == "plateau" and event["contact_stats"]["observation"] == "all_positive"]
    plateaus = [event for event in ranked if event["kind"] == "plateau"]
    growth = [event for event in ranked if event["kind"] == "growth"]
    preferred = next(iter(positive_plateaus or plateaus or growth), None)
    previews = []
    for event in ([preferred] if preferred else []) + growth[:1] + ranked:
        if event["id"] not in previews and len(previews) < 3:
            previews.append(event["id"])
    return {
        **{key: normalized[key] for key in ("id", "role", "task", "outcome")},
        "schema_version": "evaluation-action-review-v1", "rules": rules,
        "events": events, "series": series, "preferred_event_id": preferred["id"] if preferred else None,
        "preview_event_ids": previews,
        "preview_rule": "Longest plateau with every board-force sample known and positive; otherwise longest plateau, then growth. Also preview longest growth and longest remaining candidate, at most three. Full event index is never filtered.",
        "coverage": {"total_duration_s": total, "observed_duration_s": observed,
                     "gap_duration_s": gap_duration, "marked_duration_s": marked_duration,
                     "unmarked_duration_s": unmarked_duration,
                     "unknown_progress_duration_s": math.fsum(dt for i, dt in enumerate(interval_duration)
                        if not gaps[i] and (series[i]["progress"] is None or series[i + 1]["progress"] is None)),
                     "event_count": len(events), "sample_count": len(series),
                     "semantics": "marked + unmarked + gap = total, with each native interval counted once; unknown progress is a subset of unmarked, not an additional duration. Context windows do not increase coverage."},
        "limitations": list(LIMITATIONS),
    }
