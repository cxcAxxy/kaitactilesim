"""Experimental progress-conditioned summaries; no alignment or resampling.

The adapter must establish a common native clock for all fields in one episode.
Progress bins are assigned by observed sample values, never by elapsed-time
normalization. Durations use left hold; the terminal sample has zero duration.
"""

from bisect import bisect_right
from collections import Counter
import math
from numbers import Real


DEFAULT_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SEMANTICS = {
    "clock": "Within-episode native timestamps only; no expert-policy time alignment or resampling.",
    "bins": "Observed left-sample progress assigns an interval to a bin; no interpolation across bins.",
    "not_task_phases": "Progress bins are not task phases: 0% may include approach, 100% may include finishing; unchanged progress is not a stall label.",
    "progress_delta": "Signed adjacent observed progress difference, assigned to the start bin; gaps excluded.",
    "duration": "Left hold on non-gap intervals; terminal samples contribute no duration.",
    "not_observed": "No sample in this bin; does not prove it was never reached, especially across gaps.",
    "milestones": "First observed sample at or above the threshold, not an interpolated crossing time.",
    "load_integral": "Force-time load integral in N s; not energy or a collision-impulse estimate.",
    "interpretation": "Load and unchanged progress are descriptive associations, not slip, failure, or causality labels.",
    "population": "Per-episode summaries grouped by role and task; no pooled frames or success-only selection.",
}


def _number(value, *, low=None, high=None):
    """Invalid numeric measurements become unknown; incorrect types are errors."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Measurements must be numbers or null, not booleans or strings")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (low is not None and result < low) or (high is not None and result > high):
        return None
    return result


def _sequence(value, name, length=None):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} length must equal time_s length")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _safe_sum(values):
    try:
        result = math.fsum(values)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _weighted_p95(samples, total_duration):
    accumulated = 0.0
    for value, duration in sorted(samples):
        accumulated += duration
        if accumulated >= 0.95 * total_duration:
            return value
    return max(value for value, _ in samples)


def _signal_summary(definition, samples, point_values, bin_duration):
    duration = math.fsum(dt for _, dt in samples)
    mean = _safe_sum(value * (dt / duration) for value, dt in samples) if duration else None
    integral = _number(mean * duration) if mean is not None else None
    return {
        "label": definition["label"], "unit": "N", "contact_role": definition["contact_role"],
        "known_duration_s": duration,
        "coverage_fraction": duration / bin_duration if bin_duration else None,
        "mean_n": mean,
        "p95_n": _weighted_p95(samples, duration) if duration else None,
        "sampled_peak_n": max(point_values) if point_values else None,
        "load_integral_ns": integral,
    }


def analyze_episode(episode, edges=DEFAULT_EDGES):
    """Summarize one episode without inventing progress, clocks, or outcomes.

    ``gap_after[i]`` invalidates the entire interval from sample i to i+1.
    Out-of-range/nonfinite measurements become null; malformed dimensions,
    types, timestamps, units, and identity fields raise ``ValueError``.
    """
    if not isinstance(episode, dict):
        raise ValueError("episode must be a dictionary")
    identity = {key: _text(episode.get(key), key) for key in ("id", "role", "task", "outcome")}
    if identity["role"] not in ("policy", "expert_reference"):
        raise ValueError("role must be policy or expert_reference")
    if identity["outcome"] not in ("pass", "fail", "unknown"):
        raise ValueError("outcome must be pass, fail, or unknown")
    edges = [_number(value, low=0.0, high=1.0) for value in _sequence(edges, "edges")]
    if len(edges) < 2 or any(value is None for value in edges) or edges[0] != 0 or edges[-1] != 1:
        raise ValueError("edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("edges must be strictly increasing")
    raw_times = _sequence(episode.get("time_s"), "time_s")
    times = [_number(value) for value in raw_times]
    if any(value is None for value in times) or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("time_s must contain finite strictly increasing timestamps")
    duration = times[-1] - times[0] if times else 0.0
    if not math.isfinite(duration):
        raise ValueError("time_s duration must be finite")
    count = len(times)
    raw_progress = _sequence(episode.get("progress"), "progress", count)
    progress = [_number(value, low=0.0, high=1.0) for value in raw_progress]
    gaps = list(_sequence(episode.get("gap_after"), "gap_after", count))
    if any(type(value) is not bool for value in gaps):
        raise ValueError("gap_after must contain booleans")
    raw_signals = episode.get("signals")
    if not isinstance(raw_signals, dict):
        raise ValueError("signals must be a dictionary")
    signals = {}
    for key, definition in raw_signals.items():
        _text(key, "signal key")
        if not isinstance(definition, dict) or definition.get("unit") != "N":
            raise ValueError("Each signal must be a dictionary with unit N")
        signals[key] = {
            "label": _text(definition.get("label"), "signal label"), "unit": "N",
            "contact_role": _text(definition.get("contact_role"), "contact_role"),
            "values": [_number(value, low=0.0) for value in
                       _sequence(definition.get("values"), f"signals.{key}.values", count)],
        }
    assignments = [min(bisect_right(edges, value) - 1, len(edges) - 2) if value is not None else None
                   for value in progress]
    bins = []
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        bins.append({
            "index": index, "low": low, "high": high, "status": "not_observed", "first_time_s": None,
            "sample_count": 0, "visit_count": 0, "observed_duration_s": 0.0,
            "known_transition_duration_s": 0.0, "unchanged_progress_duration_s": 0.0,
            "progress_delta": None, "signals": {},
        })
    weighted = [{key: [] for key in signals} for _ in bins]
    peaks = [{key: [] for key in signals} for _ in bins]
    observed_duration = gap_duration = unknown_progress_duration = 0.0
    regression_count = 0
    for i, (time, index) in enumerate(zip(times, assignments)):
        if index is not None:
            item = bins[index]
            item["status"] = "observed"
            item["sample_count"] += 1
            if item["first_time_s"] is None:
                item["first_time_s"] = time
            if i == 0 or assignments[i - 1] != index or gaps[i - 1]:
                item["visit_count"] += 1
            for key, signal in signals.items():
                if signal["values"][i] is not None:
                    peaks[index][key].append(signal["values"][i])
        if i + 1 == count:
            continue
        dt = times[i + 1] - time
        if gaps[i]:
            gap_duration += dt
            continue
        observed_duration += dt
        if index is None:
            unknown_progress_duration += dt
            continue
        item["observed_duration_s"] += dt
        if progress[i + 1] is not None:
            delta = progress[i + 1] - progress[i]
            item["known_transition_duration_s"] += dt
            item["progress_delta"] = (item["progress_delta"] or 0.0) + delta
            if abs(delta) <= 1e-12:
                item["unchanged_progress_duration_s"] += dt
            if delta < -1e-12:
                regression_count += 1
        for key, signal in signals.items():
            value = signal["values"][i]
            if value is not None:
                weighted[index][key].append((value, dt))
    for index, item in enumerate(bins):
        item["signals"] = {key: _signal_summary(definition, weighted[index][key], peaks[index][key],
                                                item["observed_duration_s"])
                           for key, definition in signals.items()}
    known_progress = [value for value in progress if value is not None]
    missing_signals = sum(value is None for signal in signals.values() for value in signal["values"])
    partial = len(known_progress) != count or missing_signals > 0 or gap_duration > 0
    return {
        **identity, "status": "unavailable" if not known_progress else "partial" if partial else "available",
        "edges": edges, "duration_s": duration, "observed_duration_s": observed_duration,
        "gap_duration_s": gap_duration, "unknown_progress_duration_s": unknown_progress_duration,
        "initial_progress": progress[0] if progress else None,
        "final_progress": progress[-1] if progress else None,
        "last_observed_progress": known_progress[-1] if known_progress else None,
        "max_progress": max(known_progress) if known_progress else None,
        "regression_count": regression_count, "bins": bins,
        "milestones": [{
            "progress": edge,
            "first_observed_time_s": next((time for time, value in zip(times, progress)
                                            if value is not None and value >= edge), None),
            "status": "observed" if any(value >= edge for value in known_progress) else "not_observed",
        } for edge in edges],
        "points": [{"time_s": time, "progress": progress[i], "gap_after": gaps[i],
                    "signals": {key: signal["values"][i] for key, signal in signals.items()}}
                   for i, time in enumerate(times)],
        "missing_progress_samples": count - len(known_progress),
        "missing_signal_samples": missing_signals, "semantics": dict(SEMANTICS),
    }


def _distribution(values):
    """One equally weighted value per episode; missing values remain counted."""
    known = sorted(value for value in values if value is not None)
    count = len(known)
    median = (known[count // 2] if count % 2 else
              known[count // 2 - 1] / 2 + known[count // 2] / 2) if count else None
    return {"episode_count": len(values), "known_episode_count": count,
            "unknown_episode_count": len(values) - count,
            "mean": _safe_sum(value / count for value in known) if count else None,
            "median": median, "min": known[0] if count else None, "max": known[-1] if count else None}


def summarize_population(analyses):
    """Keep every episode and group descriptive summaries by role and task.

    No cross-role score/ranking is produced. Signal keys must retain identical
    unit/contact-role meanings within a task; mismatched binning is rejected.
    """
    analyses = list(_sequence(analyses, "analyses"))
    groups = {}
    seen = set()
    edges = None
    definitions = {}
    for analysis in analyses:
        if not isinstance(analysis, dict):
            raise ValueError("Each analysis must be a dictionary")
        identity = (analysis["role"], analysis["task"], analysis["id"])
        if identity in seen:
            raise ValueError("Duplicate episode identity")
        seen.add(identity)
        if edges is None:
            edges = analysis["edges"]
        if analysis["edges"] != edges:
            raise ValueError("Population analyses must use identical progress edges")
        for item in analysis["bins"]:
            for key, definition in item["signals"].items():
                semantic_key = (analysis["task"], key)
                semantics = (definition["unit"], definition["contact_role"])
                if semantic_key in definitions and definitions[semantic_key] != semantics:
                    raise ValueError("Signal units/contact roles differ for the same task and signal key")
                definitions[semantic_key] = semantics
        groups.setdefault(analysis["role"], {}).setdefault(analysis["task"], []).append(analysis)
    output = {}
    scalar_keys = ("duration_s", "observed_duration_s", "gap_duration_s", "unknown_progress_duration_s",
                   "initial_progress", "final_progress", "max_progress", "regression_count")
    bin_keys = ("observed_duration_s", "known_transition_duration_s", "unchanged_progress_duration_s",
                "progress_delta", "visit_count")
    signal_keys = ("known_duration_s", "coverage_fraction", "mean_n", "p95_n", "sampled_peak_n", "load_integral_ns")
    for role, tasks in groups.items():
        output[role] = {}
        for task, episodes in tasks.items():
            outcomes = Counter(item["outcome"] for item in episodes)
            result = {
                "episode_count": len(episodes), "outcome_counts": {key: outcomes[key] for key in ("pass", "fail", "unknown")},
                "episodes": [{key: episode[key] for key in ("id", "outcome", "status", *scalar_keys)} for episode in episodes],
                "summary": {key: _distribution([episode[key] for episode in episodes]) for key in scalar_keys}, "bins": [],
            }
            for i, (low, high) in enumerate(zip(edges, edges[1:])):
                items = [episode["bins"][i] for episode in episodes]
                keys = sorted({key for item in items for key in item["signals"]})
                result["bins"].append({
                    "index": i, "low": low, "high": high,
                    "observed_episode_count": sum(item["status"] == "observed" for item in items),
                    "not_observed_episode_count": sum(item["status"] != "observed" for item in items),
                    "summary": {key: _distribution([item[key] if item["status"] == "observed" else None for item in items])
                                for key in bin_keys},
                    "signals": {key: {
                        "unit": definitions[(task, key)][0], "contact_role": definitions[(task, key)][1],
                        **{metric: _distribution([item["signals"].get(key, {}).get(metric)
                                                   if item["status"] == "observed" else None for item in items])
                           for metric in signal_keys},
                    } for key in keys},
                })
            output[role][task] = result
    return {"episode_count": len(analyses), "edges": edges or list(DEFAULT_EDGES),
            "groups": output, "semantics": dict(SEMANTICS)}
