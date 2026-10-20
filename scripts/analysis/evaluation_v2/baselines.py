"""Explicit, episode-first conditional reference distributions (no clock matching).

``compare_to_reference(reports, reference_reports, reference_manifest=...,
compatibility=...)`` accepts v1 reports and additive v2 reports. It never reads
files, fits a time warp, infers expert membership from success, or changes task
outcomes. Compatibility may be supplied as ``report.reference_compatibility``
or a sidecar mapping trial IDs to the same object; conflicting declarations
are rejected rather than silently overridden.

Reference manifest contract::

    {"schema_version": "conditional-reference-manifest-v1",
     "reference_id": "curated-set", "role": "expert_reference",
     "declared_by": "curator", "evidence": "selection provenance",
     "selection_rule": "explicit population and inclusion rule",
     "members": [{"trial_id": "...", "source": "/absolute/trial",
                  "observation_sha256": "64 hex characters"}]}

Compatibility contract (all fields are required, not guessed)::

    {"status": "verified", "task_definition_id": "versioned task contract",
     "sensor_cadence_hz": 30, "force_unit": "N",
     "sensor_semantics_id": "versioned sensor/calibration/bandwidth contract",
     "channels": {"finger": {"status": "verified", "geometry_sha256": "..."}},
     "phase_conditions": {"scope-name": {
         "status": "verified", "condition_id": "independent condition",
         "definition_sha256": "...", "independent_of_tactile": True,
         "evidence": ["source of independently identified phase"]}}}

The context is an explicit caller assertion, not an independent calibration
certificate. The caller must audit that evidence; this pure function can only
check completeness, consistency, population membership and provenance overlap.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import posixpath
import re


SCHEMA_VERSION = "conditional-reference-comparison-v1"
DEFAULT_METRICS = (
    {"name": "normal_mean_n", "path": ["intensity", "all_observed", "mean_n"], "unit": "N"},
    {"name": "normal_p95_n", "path": ["intensity", "all_observed", "p95_n"], "unit": "N"},
    {"name": "normal_sampled_peak_n", "path": ["intensity", "sampled_peak_n"], "unit": "N"},
    {"name": "normal_contact_mean_n", "path": ["intensity", "contact", "mean_n"], "unit": "N"},
    {"name": "normal_rms_change_n", "path": ["change", "rms_change_n"], "unit": "N"},
    {"name": "observed_contact_duration_s", "path": ["continuity", "observed_contact_duration_s"], "unit": "s"},
)
_OBSERVATION_ROLES = {
    "native_control_cadence_tactile", "native_tactile", "expert_hdf5",
    "native_hdf5_observations", "reference_tactile",
}
_LIMITATIONS = [
    "Descriptive episode distributions, not a universal quality score, safety threshold, or expert acceptance band.",
    "No cross-episode timestamp alignment, dynamic time warping, frame pooling, or inferred reference membership is used.",
    "Reference selection is declared by the caller; expert competence and calibration are not independently certified here.",
    "Compatibility requires exact declared task, condition, channel geometry, units, sensor semantics, cadence, metric schema and protocol.",
    "Matching cadence does not itself establish matching physical sensor bandwidth; that must be covered by the verified sensor semantics declaration.",
    "All failed/unknown episodes and missing/unreached conditions remain in population counts; available scalar counts are reported separately.",
    "Quantiles and empirical rank intervals describe the supplied finite reference set, not population confidence intervals or pass/fail rules.",
    "Identity checks use archived observation hashes and normalized absolute paths; unrecorded aliases or preprocessing ancestry cannot be discovered by a pure report comparator.",
]


def _object(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _source(value):
    if not _text(value) or not posixpath.isabs(value) or "\x00" in value:
        return None
    return posixpath.normpath(value)


def _identity(report):
    trial = _object(report.get("trial"))
    paths, hashes = set(), set()
    provenance = report.get("provenance")
    for row in provenance if isinstance(provenance, list) else []:
        if not isinstance(row, dict) or row.get("role") not in _OBSERVATION_ROLES:
            continue
        path = _source(row.get("path"))
        if path and path.lower().endswith((".npz", ".h5", ".hdf5")) and _sha(row.get("sha256")):
            paths.add(path)
            hashes.add(row["sha256"].lower())
    return {"trial_id": trial.get("trial_id"), "source": _source(trial.get("source")),
            "paths": paths, "hashes": hashes}


def _overlap(left, right):
    return (bool(left["source"] and left["source"] == right["source"])
            or bool(left["paths"] & right["paths"])
            or bool(left["hashes"] & right["hashes"]))


def _outcome(report):
    status = _object(report.get("outcome")).get("status")
    return status if status in {"pass", "fail", "unknown"} else "unknown"


def _counts(reports):
    counts = Counter(_outcome(report) for report in reports)
    return {key: counts[key] for key in ("pass", "fail", "unknown")}


def _context(report, compatibility):
    embedded = report.get("reference_compatibility")
    sidecar = compatibility.get(_object(report.get("trial")).get("trial_id"))
    if embedded is not None and sidecar is not None and embedded != sidecar:
        return {}, ["conflicting_compatibility_declarations"]
    context = deepcopy(_object(sidecar if sidecar is not None else embedded))
    reasons = []
    if context.get("status") != "verified":
        reasons.append("compatibility_unverified")
    for key in ("task_definition_id", "sensor_semantics_id"):
        if not _text(context.get(key)):
            reasons.append(f"missing_{key}")
    if context.get("force_unit") != "N":
        reasons.append("force_unit_not_verified_N")
    cadence = context.get("sensor_cadence_hz")
    if not _number(cadence) or cadence <= 0:
        reasons.append("sensor_cadence_unverified")
    cohort = _object(_object(report.get("trial")).get("cohort"))
    interval = cohort.get("tactile_sample_interval_s")
    if _number(interval) and interval > 0 and _number(cadence) and not math.isclose(
            cadence, 1 / interval, rel_tol=2e-5, abs_tol=1e-8):
        reasons.append("sensor_cadence_conflicts_with_report")
    if not _text(_object(report.get("trial")).get("task")):
        reasons.append("task_missing")
    if not _text(_object(report.get("tactile")).get("schema_version")):
        reasons.append("metric_schema_missing")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or not _sha(report.get("protocol_sha256")):
        reasons.append("metric_protocol_unverified")
    else:
        try:
            if _hash(protocol) != report["protocol_sha256"].lower():
                reasons.append("metric_protocol_hash_mismatch")
        except (TypeError, ValueError):
            reasons.append("metric_protocol_invalid")
    if not _object(context.get("channels")):
        reasons.append("channel_geometry_unverified")
    if not _object(context.get("phase_conditions")):
        reasons.append("phase_conditions_unverified")
    return context, reasons


def _group_context(report, context, scope, channel, metric):
    geometry = _object(_object(context.get("channels")).get(channel))
    phase = _object(_object(context.get("phase_conditions")).get(scope))
    reasons = []
    if geometry.get("status") != "verified" or not _sha(geometry.get("geometry_sha256")):
        reasons.append("channel_geometry_unverified")
    if (phase.get("status") != "verified" or phase.get("independent_of_tactile") is not True
            or not _text(phase.get("condition_id")) or not _sha(phase.get("definition_sha256"))
            or not isinstance(phase.get("evidence"), list)
            or not phase["evidence"] or not all(_text(row) for row in phase["evidence"])):
        reasons.append("independent_phase_condition_unverified")
    if reasons:
        return None, reasons
    outcome = _object(report.get("outcome"))
    mode = "diagnostic" if outcome.get("diagnostic_only") is True or outcome.get("formal_metrics_valid") is False else (
        "formal" if outcome.get("formal_metrics_valid") is True and outcome.get("diagnostic_only") is False else "unverified")
    return {
        "task": report["trial"]["task"], "task_definition_id": context["task_definition_id"],
        "scope": scope, "channel": channel, "geometry_sha256": geometry["geometry_sha256"].lower(),
        "force_unit": "N", "sensor_semantics_id": context["sensor_semantics_id"],
        "sensor_cadence_hz": round(float(context["sensor_cadence_hz"]), 9),
        "condition_id": phase["condition_id"], "condition_definition_sha256": phase["definition_sha256"].lower(),
        "metric_schema": report["tactile"]["schema_version"], "protocol_sha256": report["protocol_sha256"].lower(),
        "formal_use_class": mode, "metric": metric,
    }, []


def _metric_value(report, scope_name, channel, spec):
    scope = _object(_object(_object(report.get("tactile")).get("scopes")).get(scope_name))
    if scope.get("status") != "available":
        return None, "scope_" + str(scope.get("status", "missing"))
    value = _object(_object(scope.get("channels")).get(channel))
    if value.get("status") != "available":
        return None, "channel_" + str(value.get("status", "missing"))
    for name in spec["path"]:
        if not isinstance(value, dict):
            return None, "metric_missing"
        # A finite stale scalar must not override an explicit unavailable block.
        if value.get("status") not in (None, "available"):
            return None, "metric_" + str(value["status"])
        value = value.get(name)
    return (float(value), None) if _number(value) else (None, "metric_missing")


def _specs(metric_specs):
    specs = deepcopy(list(DEFAULT_METRICS if metric_specs is None else metric_specs))
    names = set()
    for spec in specs:
        if (not isinstance(spec, dict) or set(spec) != {"name", "path", "unit"}
                or not _text(spec["name"]) or spec["name"] in names
                or not _text(spec["unit"]) or not isinstance(spec["path"], (list, tuple))
                or not spec["path"] or not all(_text(key) for key in spec["path"])):
            raise ValueError("Metric specs require unique name, nonempty dictionary-key path, and unit")
        spec["path"] = list(spec["path"])
        names.add(spec["name"])
    if not specs:
        raise ValueError("At least one metric is required")
    return specs


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def _statistics(values):
    ordered = sorted(values)
    if not ordered:
        return None
    median = _quantile(ordered, .5)
    return {"p05": _quantile(ordered, .05), "p25": _quantile(ordered, .25),
            "median": median, "p75": _quantile(ordered, .75), "p95": _quantile(ordered, .95),
            "mad": _quantile(sorted(abs(value - median) for value in values), .5),
            "min": ordered[0], "max": ordered[-1]}


def _manifest(manifest):
    reasons = []
    if not isinstance(manifest, dict):
        return {}, ["reference_manifest_missing"]
    try:
        _hash(manifest)
    except (TypeError, ValueError):
        reasons.append("reference_manifest_not_canonical_json")
    if manifest.get("schema_version") != "conditional-reference-manifest-v1":
        reasons.append("reference_manifest_schema_unsupported")
    if manifest.get("role") != "expert_reference":
        reasons.append("expert_reference_membership_not_declared")
    for name in ("reference_id", "declared_by", "evidence", "selection_rule"):
        if not _text(manifest.get(name)):
            reasons.append(f"reference_manifest_missing_{name}")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        reasons.append("reference_manifest_members_missing")
        return {}, reasons
    index = {}
    for row in members:
        if (not isinstance(row, dict) or not _text(row.get("trial_id"))
                or not _source(row.get("source")) or not _sha(row.get("observation_sha256"))):
            reasons.append("reference_manifest_member_invalid")
            continue
        if row["trial_id"] in index:
            reasons.append("reference_manifest_duplicate_trial_id")
        index[row["trial_id"]] = row
    return index, sorted(set(reasons))


def compare_to_reference(reports, reference_reports, *, reference_manifest=None,
                         compatibility=None, metric_specs=None, minimum_reference_episodes=3):
    """Return conditional finite-population summaries, preserving all input trials.

    Evidence problems are returned as explicit unavailable rows. Invalid API
    arguments (e.g. non-report input or metric path) raise ``ValueError``. A
    missing manifest never activates an automatic success-derived reference.
    Every metric is one channel/scope scalar per episode, with equal episode
    weight. Only fully available scope/channel/metric blocks enter a group.
    """
    reports, reference_reports = list(reports), list(reference_reports)
    if not all(isinstance(report, dict) for report in reports + reference_reports):
        raise ValueError("Inputs must be report dictionaries")
    if (not isinstance(minimum_reference_episodes, int) or isinstance(minimum_reference_episodes, bool)
            or minimum_reference_episodes < 2):
        raise ValueError("minimum_reference_episodes must be an integer at least 2")
    if compatibility is not None and not isinstance(compatibility, dict):
        raise ValueError("compatibility must map trial IDs to declarations")
    compatibility = compatibility or {}
    specs = _specs(metric_specs)
    members, manifest_reasons = _manifest(reference_manifest)
    ref_ids = [_identity(report) for report in reference_reports]
    candidate_ids = [_identity(report) for report in reports]
    groups, ref_rows, candidate_rows = {}, [], []
    accepted_references = []
    for i, (report, identity) in enumerate(zip(reference_reports, ref_ids)):
        reasons = list(manifest_reasons)
        member = members.get(identity["trial_id"])
        if member is None:
            reasons.append("not_a_declared_reference_member")
        elif (_source(member["source"]) != identity["source"]
              or member["observation_sha256"].lower() not in identity["hashes"]):
            reasons.append("reference_membership_provenance_mismatch")
        if not identity["source"] or not identity["hashes"]:
            reasons.append("observation_identity_unverified")
        if any(i != j and (_overlap(identity, other) or identity["trial_id"] == other["trial_id"])
               for j, other in enumerate(ref_ids)):
            reasons.append("duplicate_reference_observation")
        context, context_reasons = _context(report, compatibility)
        reasons.extend(context_reasons)
        row = {"trial_id": identity["trial_id"], "source": identity["source"],
               "outcome": deepcopy(report.get("outcome", {"status": "unknown"})),
               "status": "rejected" if reasons else "accepted", "reasons": sorted(set(reasons)),
               "condition_rejections": []}
        ref_rows.append(row)
        if reasons:
            continue
        accepted_references.append(report)
        for scope in sorted(_object(context.get("phase_conditions"))):
            for channel in sorted(_object(context.get("channels"))):
                for spec in specs:
                    group_context, group_reasons = _group_context(report, context, scope, channel, spec)
                    if group_reasons:
                        row["condition_rejections"].append({"scope": scope, "channel": channel,
                                                             "metric": spec["name"], "reasons": group_reasons})
                        continue
                    key = _hash(group_context)
                    group = groups.setdefault(key, {"compatibility": group_context, "metric": spec,
                        "compatible_episode_count": 0, "reference_episode_count": 0,
                        "outcome_counts": {"pass": 0, "fail": 0, "unknown": 0},
                        "metric_outcome_counts": {"pass": 0, "fail": 0, "unknown": 0},
                        "missing_reasons": {}, "trial_ids": [], "observed_trial_ids": [], "_values": []})
                    group["compatible_episode_count"] += 1
                    group["outcome_counts"][_outcome(report)] += 1
                    group["trial_ids"].append(identity["trial_id"])
                    value, reason = _metric_value(report, scope, channel, spec)
                    if reason:
                        group["missing_reasons"][reason] = group["missing_reasons"].get(reason, 0) + 1
                    else:
                        group["_values"].append(value)
                        group["observed_trial_ids"].append(identity["trial_id"])
                        group["metric_outcome_counts"][_outcome(report)] += 1
    for group in groups.values():
        values = group["_values"]
        group["reference_episode_count"] = len(values)
        group["status"] = "available" if len(values) >= minimum_reference_episodes else "insufficient_reference_episodes"
        group["statistics"] = _statistics(values) if group["status"] == "available" else None
        group["small_reference_warning"] = len(values) < 20
    for i, (report, identity) in enumerate(zip(reports, candidate_ids)):
        reasons = list(manifest_reasons)
        if not _text(identity["trial_id"]) or not identity["source"] or not identity["hashes"]:
            reasons.append("observation_identity_unverified")
        if any(_overlap(identity, other) or identity["trial_id"] == other["trial_id"] for other in ref_ids):
            reasons.append("candidate_reference_source_overlap")
        if any(i != j and (_overlap(identity, other) or identity["trial_id"] == other["trial_id"])
               for j, other in enumerate(candidate_ids)):
            reasons.append("duplicate_candidate_observation")
        context, context_reasons = _context(report, compatibility)
        reasons.extend(context_reasons)
        row = {"trial_id": identity["trial_id"], "task": _object(report.get("trial")).get("task"),
               "source": identity["source"], "outcome": deepcopy(report.get("outcome", {"status": "unknown"})),
               "status": "unavailable", "reasons": sorted(set(reasons)), "metrics": []}
        candidate_rows.append(row)
        if reasons:
            continue
        for scope in sorted(_object(context.get("phase_conditions"))):
            for channel in sorted(_object(context.get("channels"))):
                for spec in specs:
                    group_context, group_reasons = _group_context(report, context, scope, channel, spec)
                    value, missing_reason = _metric_value(report, scope, channel, spec)
                    item = {"scope": scope, "channel": channel, "metric": spec["name"], "unit": spec["unit"],
                            "value": value, "status": "unavailable", "reason": None,
                            "reference_group_id": _hash(group_context) if group_context else None,
                            "reference_episode_count": 0, "percentile_rank_interval": None}
                    row["metrics"].append(item)
                    group = groups.get(item["reference_group_id"])
                    if group_reasons:
                        item["reason"] = ";".join(group_reasons)
                    elif missing_reason:
                        item["reason"] = missing_reason
                    elif group is None:
                        item["reason"] = "no_compatible_reference_group"
                    else:
                        item["reference_episode_count"] = group["reference_episode_count"]
                        if group["status"] != "available":
                            item["reason"] = group["status"]
                        else:
                            values = group["_values"]
                            item["percentile_rank_interval"] = [sum(v < value for v in values) / len(values),
                                                                 sum(v <= value for v in values) / len(values)]
                            item["status"] = "available"
        available = sum(item["status"] == "available" for item in row["metrics"])
        row["status"] = "available" if available and available == len(row["metrics"]) else "partial" if available else "unavailable"
        if not available:
            row["reasons"] = sorted({item["reason"] for item in row["metrics"] if item["reason"]}) or ["no_declared_scope_channel_metrics"]
    for group in groups.values():
        del group["_values"]
    available_trials = sum(row["status"] in {"available", "partial"} for row in candidate_rows)
    manifest = _object(reference_manifest)
    try:
        manifest_hash = _hash(manifest) if manifest else None
    except (TypeError, ValueError):
        manifest_hash = None
    return {"schema_version": SCHEMA_VERSION,
            "status": "available" if available_trials and all(row["status"] == "available" for row in candidate_rows) else "partial" if available_trials else "unavailable",
            "reference_id": manifest.get("reference_id"), "reference_manifest_sha256": manifest_hash,
            "aggregation_unit": "episode", "time_alignment": "none",
            "minimum_reference_episodes": minimum_reference_episodes, "metric_specs": specs,
            "reference_population": {"declared_member_count": len(members), "supplied_report_count": len(reference_reports),
                "accepted_report_count": len(accepted_references), "outcome_counts": _counts(reference_reports),
                "accepted_outcome_counts": _counts(accepted_references), "selection_rule": manifest.get("selection_rule"),
                "declared_by": manifest.get("declared_by"), "evidence": manifest.get("evidence"),
                "missing_member_ids": sorted(set(members) - {identity["trial_id"] for identity in ref_ids}),
                "reasons": manifest_reasons},
            "candidate_population": {"input_trial_count": len(reports), "outcome_counts": _counts(reports),
                                     "comparable_trial_count": available_trials},
            "groups": groups, "reference_trials": ref_rows, "trials": candidate_rows,
            "limitations": list(_LIMITATIONS)}
