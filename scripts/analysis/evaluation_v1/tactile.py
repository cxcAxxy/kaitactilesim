"""Reference-free descriptive tactile metrics using only NumPy and the stdlib.

Inputs named ``*_n`` must already be calibrated forces in newtons. This module
does not calibrate raw sensor counts, infer skin pressure, or establish slip.
Per-channel tangent vectors and taxel positions must use a stable, consistent
pad-local coordinate system. No vectors from different pads are added together.

All intervals are native left holds [t[i], t[i+1]); the terminal sample has zero
duration, but still participates in native sampled peaks. Long gaps, missing
forces and negative normal forces never become zero-force observations.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


DEFAULT_CONFIG = {
    "contact_on_n": 0.05,
    "contact_off_n": 0.03,
    "min_contact_duration_s": 0.1,
    "analysis_bin_s": 0.1,
    "change_lag_s": 0.1,
    "max_gap_factor": 1.5,
}


def _configuration(config):
    result = dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"unsupported tactile configuration: {sorted(unknown)}")
        result.update(config)
    for key, value in result.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a finite number")
        try:
            result[key] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite number") from error
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= result["contact_off_n"] <= result["contact_on_n"]:
        raise ValueError("require 0 <= contact_off_n <= contact_on_n")
    if result["min_contact_duration_s"] < 0:
        raise ValueError("min_contact_duration_s must be nonnegative")
    if result["analysis_bin_s"] <= 0 or result["change_lag_s"] <= 0:
        raise ValueError("analysis_bin_s and change_lag_s must be positive")
    if result["max_gap_factor"] < 1:
        raise ValueError("max_gap_factor must be at least 1")
    ratio = result["change_lag_s"] / result["analysis_bin_s"]
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9, rel_tol=0):
        raise ValueError("change_lag_s must be an integer multiple of analysis_bin_s")
    return result


def _stats(values, weights, *, unit="n"):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = np.asarray(values)[valid], np.asarray(weights)[valid]
    result = {f"{name}_{unit}": None for name in ("mean", "p50", "p95", "std")}
    result["duration_s"] = float(weights.sum())
    if not len(values):
        return result
    mean = float(np.average(values, weights=weights))
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    result[f"mean_{unit}"] = mean
    result[f"std_{unit}"] = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    for q, label in ((0.5, "p50"), (0.95, "p95")):
        index = min(int(np.searchsorted(cumulative, q * cumulative[-1])), len(order) - 1)
        result[f"{label}_{unit}"] = float(values[order[index]])
    return result


def _availability(known_duration, requested_duration, tolerance):
    if requested_duration <= tolerance:
        return "insufficient_duration"
    if known_duration <= tolerance:
        return "missing"
    return "available" if known_duration >= requested_duration - tolerance else "partial"


def _peak(values, selected, times):
    indices = np.flatnonzero(selected & np.isfinite(values))
    if not len(indices):
        return None, None
    index = int(indices[np.argmax(values[indices])])
    return float(values[index]), float(times[index])


def _bin_values(times, durations, selected, values, edges):
    """Time integrals into episode-anchored bins, excluding unknown intervals."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    sums = np.zeros((len(edges) - 1, values.shape[1]), dtype=float)
    coverage = np.zeros(len(edges) - 1, dtype=float)
    for index in np.flatnonzero(selected & (durations > 0)):
        left, right = times[index], times[index] + durations[index]
        first = max(0, int(np.searchsorted(edges, left, side="right")) - 1)
        stop = min(len(coverage), int(np.searchsorted(edges, right, side="left")))
        for bin_index in range(first, stop):
            duration = max(0.0, min(right, edges[bin_index + 1]) - max(left, edges[bin_index]))
            coverage[bin_index] += duration
            sums[bin_index] += duration * values[index]
    means = np.full(sums.shape, np.nan)
    np.divide(sums, coverage[:, None], out=means, where=coverage[:, None] > 0)
    return means, coverage


def _bin_pairs(coverage, config, tolerance):
    complete = coverage >= config["analysis_bin_s"] - tolerance
    lag = int(round(config["change_lag_s"] / config["analysis_bin_s"]))
    if lag >= len(complete):
        return complete, np.array([], dtype=int), np.array([], dtype=int)
    # Every intermediate bin must be complete too: no jumping over data gaps,
    # noncontact intervals (for spatial metrics), or an out-of-scope stage.
    bad_prefix = np.r_[0, np.cumsum(~complete)]
    start = np.arange(len(complete) - lag)
    start = start[(bad_prefix[start + lag + 1] - bad_prefix[start]) == 0]
    return complete, start, start + lag


def _change(times, durations, selected, values, edges, config, tolerance):
    means, coverage = _bin_values(times, durations, selected, values, edges)
    complete, first, last = _bin_pairs(coverage, config, tolerance)
    changes = np.linalg.norm(means[last] - means[first], axis=1)
    stats = _stats(changes, np.full(len(changes), config["analysis_bin_s"]))
    return {
        "status": "available" if len(changes) else "insufficient_complete_bins",
        "analysis_bin_s": config["analysis_bin_s"],
        "change_lag_s": config["change_lag_s"],
        "complete_bin_count": int(complete.sum()),
        "pair_count": len(changes),
        "mean_absolute_change_n": stats["mean_n"],
        "p95_absolute_change_n": stats["p95_n"],
        "rms_change_n": float(np.sqrt(np.mean(changes ** 2))) if len(changes) else None,
        "max_absolute_change_n": float(changes.max()) if len(changes) else None,
        "semantics": "magnitude of fixed-lag change between complete time-bin means; not a derivative, unified sensor bandwidth, or slip measurement",
    }


def _hysteresis(force, valid, durations, config):
    """State -1 means unknown; a new segment in the hysteresis band is unknown."""
    states = np.full(len(force), -1, dtype=np.int8)
    state = -1
    for index, value in enumerate(force):
        if not valid[index] or durations[index] <= 0:
            state = -1
            continue
        if value > config["contact_on_n"]:
            state = 1
        elif value <= config["contact_off_n"]:
            state = 0
        states[index] = state
    return states


def _runs(selected, times, durations, tolerance):
    indices = np.flatnonzero(selected & (durations > 0))
    runs = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1 and abs(times[index] - times[runs[-1][-1]] - durations[runs[-1][-1]]) <= tolerance:
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    return runs


def _continuity(times, raw_durations, durations, mask, states, config, tolerance):
    known = mask & (states >= 0) & (durations > 0)
    contact = known & (states == 1)
    candidates = _runs(contact, times, durations, tolerance)
    confirmed = [run for run in candidates if durations[run].sum() + tolerance >= config["min_contact_duration_s"]]
    episodes = []
    for run in confirmed:
        first, last = run[0], run[-1]
        episodes.append({
            "start_s": float(times[first]),
            "end_s": float(times[last] + durations[last]),
            "duration_s": float(durations[run].sum()),
            "left_censored": bool(first == 0 or not known[first - 1]),
            "right_censored": bool(last + 1 == len(times) or not known[last + 1]),
        })
    internal = np.zeros(len(times), dtype=bool)
    # All raw observed contact candidates delimit the descriptive span; short
    # contacts are retained separately, rather than silently turned into zeros.
    if candidates:
        internal[candidates[0][0]:candidates[-1][-1] + 1] = True
    absent = internal & known & (states == 0)
    breaks = _runs(absent, times, durations, tolerance)
    break_lengths = [float(durations[run].sum()) for run in breaks]
    requested = float(raw_durations[mask].sum())
    known_duration = float(durations[known].sum())
    return {
        "status": _availability(known_duration, requested, tolerance),
        "semantics": "observed contact and internal known absences only; no expected-contact contract or grasp-failure conclusion",
        "contact_on_n": config["contact_on_n"],
        "contact_off_n": config["contact_off_n"],
        "min_contact_duration_s": config["min_contact_duration_s"],
        "observed_contact_duration_s": float(durations[contact].sum()),
        "confirmed_contact_duration_s": sum(item["duration_s"] for item in episodes),
        "contact_fraction_of_known_state_time": float(durations[contact].sum() / known_duration) if known_duration > 0 else None,
        "known_state_duration_s": known_duration,
        "unknown_duration_s": max(0.0, requested - known_duration),
        "contact_episode_count": len(confirmed) if known_duration > 0 else None,
        "candidate_episode_count": len(candidates) if known_duration > 0 else None,
        "short_contact_episode_count": len(candidates) - len(confirmed) if known_duration > 0 else None,
        "observed_internal_break_count": len(breaks) if candidates else None,
        "observed_internal_break_duration_s": sum(break_lengths) if candidates else None,
        "longest_internal_break_s": max(break_lengths, default=0.0) if candidates else None,
        "episodes": episodes,
    }


def _spatial(times, durations, mask, force, grid, coordinates, edges, config, tolerance):
    center = {
        "status": "missing", "mean_local_mm": None,
        "mean_displacement_mm": None, "p95_displacement_mm": None,
    }
    distribution = {"status": "missing", "pair_count": None,
                    "mean_total_variation": None, "p95_total_variation": None}
    result = {
        "status": "missing", "distribution_change": distribution,
        "center_of_pressure": center,
        "semantics": "normalized force allocation and supplied pad-local force-weighted taxel center; not skin pressure, contact area ground truth, or slip",
    }
    if grid is None:
        return result
    grid_valid = np.all(np.isfinite(grid) & (grid >= 0), axis=1)
    sums = np.sum(np.where(np.isfinite(grid), grid, 0), axis=1)
    consistent = np.isfinite(force) & (force >= 0) & np.isclose(sums, force, rtol=1e-5, atol=1e-8)
    valid = mask & grid_valid & consistent & (force > config["contact_on_n"])
    distribution_values = np.full_like(grid, np.nan)
    np.divide(grid, sums[:, None], out=distribution_values, where=valid[:, None] & (sums[:, None] > 0))
    means, coverage = _bin_values(times, durations, valid, distribution_values, edges)
    complete, first, last = _bin_pairs(coverage, config, tolerance)
    tv = 0.5 * np.abs(means[last] - means[first]).sum(axis=1)
    stats = _stats(tv, np.full(len(tv), config["analysis_bin_s"]), unit="fraction")
    known_duration = float(durations[valid].sum())
    result.update(
        status="available" if known_duration > 0 else "no_valid_contact",
        contact_observed_duration_s=known_duration,
        inconsistent_force_sum_sample_count=int(np.count_nonzero(mask & grid_valid & ~consistent)),
    )
    distribution.update(
        status="available" if len(tv) else "insufficient_complete_contact_bins",
        pair_count=len(tv), complete_bin_count=int(complete.sum()),
        mean_total_variation=stats["mean_fraction"], p95_total_variation=stats["p95_fraction"],
    )
    if coordinates is not None:
        centers = distribution_values @ coordinates * 1000.0
        eligible = valid & (durations > 0)
        center_means = means @ coordinates * 1000.0
        displacement = np.linalg.norm(center_means[last] - center_means[first], axis=1)
        displacement_stats = _stats(displacement, np.full(len(displacement), config["analysis_bin_s"]), unit="mm")
        center.update(
            status="available" if known_duration > 0 else "no_valid_contact",
            mean_local_mm=np.average(centers[eligible], axis=0, weights=durations[eligible]).tolist() if np.any(eligible) else None,
            mean_displacement_mm=displacement_stats["mean_mm"],
            p95_displacement_mm=displacement_stats["p95_mm"],
            displacement_pair_count=len(displacement),
            change_status="available" if len(displacement) else "insufficient_complete_contact_bins",
        )
    return result


def _tangent(times, durations, mask, tangent, requested, edges, config, tolerance):
    result = {
        "status": "missing", "net_magnitude": None,
        "sampled_peak_n": None, "sampled_peak_time_s": None,
        "coverage_fraction": None, "change": None,
        "semantics": "sum signed per-taxel vectors in the supplied stable pad-local basis, then take magnitude; non-cancelling contact shear load is unavailable",
    }
    if tangent is None:
        return result
    valid = np.all(np.isfinite(tangent), axis=(1, 2))
    net = np.sum(tangent, axis=1)
    magnitude = np.linalg.norm(net, axis=1)
    selected = mask & valid
    stats = _stats(magnitude, durations * selected)
    peak, peak_time = _peak(magnitude, selected, times)
    result.update(
        status=_availability(stats["duration_s"], requested, tolerance),
        net_magnitude=stats, sampled_peak_n=peak, sampled_peak_time_s=peak_time,
        coverage_fraction=stats["duration_s"] / requested if requested > 0 else None,
        change=_change(times, durations, selected, net, edges, config, tolerance),
    )
    return result


def _shares(force, durations, mask, names, requested, threshold, tolerance):
    valid = mask & np.all(np.isfinite(force) & (force >= 0), axis=1)
    total = np.sum(np.where(np.isfinite(force), force, 0), axis=1)
    eligible = valid & (total > threshold)
    observed = float(durations[valid].sum())
    contact_duration = float(durations[eligible].sum())
    shares = np.full_like(force, np.nan)
    np.divide(force, total[:, None], out=shares, where=eligible[:, None])
    channels = {}
    for index, name in enumerate(names):
        stats = _stats(shares[:, index], durations * eligible, unit="fraction")
        channels[name] = {"mean_fraction": stats["mean_fraction"], "p95_fraction": stats["p95_fraction"]}
    status = _availability(observed, requested, tolerance)
    if observed > tolerance and contact_duration <= tolerance:
        status = "no_valid_contact"
    return {
        "status": status,
        "semantics": "time-weighted instantaneous shares of total normal scalar load; all channels must be valid and total load must exceed contact_on_n",
        "observed_duration_s": observed, "contact_duration_s": contact_duration,
        "coverage_fraction": observed / requested if requested > 0 else None,
        "channels": channels,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_tactile(time_s, normal_force_n, *, channel_names=None,
                    normal_taxel_force_n=None, tangent_taxel_force_n=None,
                    taxel_positions_local_m=None, scope_masks=None, config=None):
    """Analyze arbitrary tactile channels, with no expert or simulation import.

    ``normal_force_n`` is (N, C), normal grids (N, C, ...), and tangent grids
    (N, C, ..., 2). Coordinates may be (C, P, 3), (C, *grid_shape, 3), or shared
    (P, 3), where P is the number of taxels. Scope masks must be boolean (N,)
    arrays; an all-false mask is ``not_reached`` rather than a zero-valued score.
    All event and peak times in scopes are relative to the first input sample.

    Cadence is inferred from the median native interval. A duration exceeding
    ``max_gap_factor`` times this cadence is wholly unknown. The inference is
    descriptive: regular low-rate acquisition cannot reveal hidden missing
    samples. Scope membership is held from the interval's left sample.
    """
    config = _configuration(config)
    source_time = np.asarray(time_s, dtype=float)
    force = np.asarray(normal_force_n, dtype=float)
    if source_time.ndim != 1 or not len(source_time) or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
        raise ValueError("time_s must be a nonempty, finite, strictly increasing vector")
    if force.ndim != 2 or force.shape[0] != len(source_time) or force.shape[1] < 1:
        raise ValueError("normal_force_n must have shape (N, C), with C >= 1")
    count, channels = force.shape
    names = list(channel_names) if channel_names is not None else [f"channel_{index}" for index in range(channels)]
    if len(names) != channels or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != channels:
        raise ValueError("channel_names must contain C unique nonempty strings")
    times = source_time - source_time[0]
    tolerance = max(1e-10, float(np.max(np.abs(source_time))) * np.finfo(float).eps * 8)
    if tolerance >= config["analysis_bin_s"] * 1e-3:
        raise ValueError("timestamp precision is insufficient for analysis_bin_s; provide relative times")
    raw_durations = np.r_[np.diff(times), 0.0]
    cadence = float(np.median(np.diff(times))) if count > 1 else None
    gap = raw_durations > cadence * config["max_gap_factor"] + tolerance if cadence is not None else np.zeros(count, dtype=bool)
    durations = np.where(gap, 0.0, raw_durations)
    scopes = {"overall": np.ones(count, dtype=bool)}
    if scope_masks is not None:
        if not isinstance(scope_masks, Mapping):
            raise ValueError("scope_masks must be a mapping")
        for name, value in scope_masks.items():
            mask = np.asarray(value)
            if not isinstance(name, str) or not name or name == "overall":
                raise ValueError("scope names must be nonempty strings other than reserved 'overall'")
            if mask.shape != (count,) or mask.dtype.kind != "b":
                raise ValueError(f"scope mask {name!r} must be a boolean vector of length N")
            scopes[name] = mask
    grid, coordinates, tangent = None, None, None
    if normal_taxel_force_n is not None:
        supplied_grid = np.asarray(normal_taxel_force_n, dtype=float)
        if supplied_grid.ndim < 3 or supplied_grid.shape[:2] != force.shape or any(size == 0 for size in supplied_grid.shape[2:]):
            raise ValueError("normal_taxel_force_n must have shape (N, C, ...), with nonempty taxel axes")
        grid = supplied_grid.reshape(count, channels, -1)
        if taxel_positions_local_m is not None:
            coordinates = np.asarray(taxel_positions_local_m, dtype=float)
            if coordinates.shape == (grid.shape[2], 3):
                coordinates = np.broadcast_to(coordinates, (channels, grid.shape[2], 3))
            elif coordinates.shape not in ((channels, grid.shape[2], 3), (channels, *supplied_grid.shape[2:], 3)):
                raise ValueError("taxel_positions_local_m must match channels and taxels with a final xyz axis")
            coordinates = coordinates.reshape(channels, grid.shape[2], 3)
            if not np.isfinite(coordinates).all():
                raise ValueError("taxel_positions_local_m must be finite")
    if tangent_taxel_force_n is not None:
        supplied_tangent = np.asarray(tangent_taxel_force_n, dtype=float)
        if supplied_tangent.ndim < 3 or supplied_tangent.shape[:2] != force.shape or supplied_tangent.shape[-1] != 2 or any(size == 0 for size in supplied_tangent.shape[2:]):
            raise ValueError("tangent_taxel_force_n must have shape (N, C, ..., 2)")
        if grid is not None and supplied_tangent.shape != (*supplied_grid.shape, 2):
            raise ValueError("normal and tangent taxel grid shapes must agree")
        tangent = supplied_tangent.reshape(count, channels, -1, 2)
    span = float(times[-1])
    bin_count = max(0, int(math.ceil((span - tolerance) / config["analysis_bin_s"])))
    edges = np.arange(bin_count + 1) * config["analysis_bin_s"]
    valid = np.isfinite(force) & (force >= 0)
    states = [_hysteresis(force[:, index], valid[:, index], durations, config) for index in range(channels)]
    result_scopes = {}
    for scope_name, mask in scopes.items():
        requested = float(raw_durations[mask].sum())
        scope = {"status": "available" if requested > tolerance else "insufficient_duration",
                 "duration_s": requested, "sample_count": int(mask.sum()),
                 "channels": {}, "load_shares": None}
        result_scopes[scope_name] = scope
        if not np.any(mask):
            scope.update(status="not_reached", duration_s=None, channels=None)
            continue
        for index, name in enumerate(names):
            values = force[:, index]
            selected = mask & valid[:, index]
            weights = durations * selected
            observed = float(weights.sum())
            contact_weights = weights * (values > config["contact_on_n"])
            peak, peak_time = _peak(values, selected, times)
            scope["channels"][name] = {
                "status": _availability(observed, requested, tolerance),
                "intensity": {
                    "scope_duration_s": requested, "observed_duration_s": observed,
                    "coverage_fraction": observed / requested if requested > 0 else None,
                    "invalid_sample_count": int(np.count_nonzero(mask & ~valid[:, index])),
                    "all_observed": _stats(values, weights),
                    "contact": _stats(values, contact_weights),
                    "contact_definition": "normal force strictly greater than contact_on_n; independent of hysteresis event qualification",
                    "sampled_peak_n": peak, "sampled_peak_time_s": peak_time,
                },
                "change": _change(times, durations, selected, values, edges, config, tolerance),
                "continuity": _continuity(times, raw_durations, durations, mask, states[index], config, tolerance),
                "spatial": _spatial(times, durations, mask, values, grid[:, index] if grid is not None else None,
                                    coordinates[index] if coordinates is not None else None, edges, config, tolerance),
                "tangent": _tangent(times, durations, mask, tangent[:, index] if tangent is not None else None,
                                    requested, edges, config, tolerance),
            }
        scope["load_shares"] = _shares(force, durations, mask, names, requested, config["contact_on_n"], tolerance)
    return _json_safe({
        "schema_version": "evaluation-v1-tactile-1", "config": config,
        "timebase": {
            "source_start_time_s": float(source_time[0]), "duration_s": span,
            "sample_count": count, "nominal_interval_s": cadence,
            "nominal_hz": 1.0 / cadence if cadence is not None else None,
            "gap_interval_count": int(gap.sum()), "gap_duration_s": float(raw_durations[gap].sum()),
            "sampling_observed_duration_s": float(durations.sum()),
            "nominal_interval_source": "median of native differences; no declared sensor cadence supplied",
            "clock": "scope times are relative to the first input sample; native left holds with zero-duration terminal sample",
        },
        "input_evidence": {
            "force_unit": "N", "calibration": "assumed by input contract; not verified here",
            "normal_taxels_supplied": grid is not None, "tangent_taxels_supplied": tangent is not None,
            "physical_taxel_geometry_available": coordinates is not None,
            "spatial_source": "caller-supplied; physical skin-pressure validity not established",
        },
        "definitions": {
            "scope": "native left-sample membership; no expert-clock alignment; supplied stages must be independently justified by caller",
            "coverage": "finite nonnegative native normal observations divided by selected physical interval duration; gaps remain unknown",
            "change": "complete episode-anchored physical time-bin means and integer-bin lag; all intermediate bins must be complete; binning is not sensor-bandwidth matching",
            "contact": "hysteresis enters above on and exits at or below off; an uninitialized hysteresis-band sample is unknown; minimum duration filters reported episodes only, retaining short candidates",
            "peak": "maximum valid native observation, including terminal and short spikes; not a physics-step or between-sample bound",
            "interpretation": "descriptive contact metrics only; no universal safety threshold, success score, slip truth, or proof of policy tactile utilization",
        },
        "scopes": result_scopes,
    })
