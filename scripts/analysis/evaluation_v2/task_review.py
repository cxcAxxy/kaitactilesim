"""Evidence-aware task review, separate from the original rollout's outcome.

No simulator imports, reruns or writes.  A recorded success flag is never fed to
the independent predicates.  The ``review`` result describes the explicitly
selected observable review contract, not an independent validation of physics.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contracts import get_contract

_KINDS = {"measurement", "simulation_oracle"}
_NONTERMINAL = {"running", "pending", "started", "initializing", "in_progress", "queued"}


def _number(value):
    return float(value) if isinstance(value, (float, int, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)) else None


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _status(value):
    return "pass" if value is True else "fail" if value is False else "unknown"


def _and(values):
    values = list(values)
    return False if False in values else None if None in values else True


def _or(values):
    values = list(values)
    return True if True in values else None if None in values else False


def _recorded(summary):
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    labels = [{"source": source, "value": container[key]} for source, container, key in (
        ("summary.evaluation.success", evaluation, "success"), ("summary.success", summary, "success"),
        ("summary.task_success", summary, "task_success")) if type(container.get(key)) is bool]
    conflict = len({row["value"] for row in labels}) > 1
    running = summary.get("status") in _NONTERMINAL or evaluation.get("status") in _NONTERMINAL
    invalid = summary.get("physics_valid") is False or evaluation.get("physics_valid") is False
    value = labels[0]["value"] if labels and not conflict and not running and not invalid else None
    return {"status": _status(value), "labels": labels, "conflict": conflict, "finalized": not running,
            "physics_invalid_recorded": invalid, "source": "recorded label only; not recomputed",
            "termination": summary.get("termination_reason", summary.get("status"))}


def _manifest(trial):
    """Inspect only a recorded manifest within the local experiment ancestry.

    The smallest matching ancestor is bounded at three parents.  This allows
    seed/evaluation/deployment layouts without following arbitrary data paths.
    A source hash match is evidence about rule version, not evidence of success.
    """
    summary = trial.summary
    path = summary.get("deployment_manifest")
    if not isinstance(path, str):
        return {}, [], []
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = trial.root / candidate
        candidate = candidate.resolve()
        ancestors = list(Path(trial.root).resolve().parents)[:3]
        if not any(candidate.is_relative_to(parent) for parent in ancestors) or candidate.suffix != ".json":
            return {}, [], ["Recorded source manifest is outside the bounded experiment ancestry; it was not read."]
        before = candidate.stat()
        if before.st_size > 10_000_000:
            return {}, [], ["Recorded source manifest exceeds the small-metadata size limit."]
        raw = candidate.read_bytes()
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            return {}, [], ["Source manifest changed while reading; source version is unknown."]
        digest = hashlib.sha256(raw).hexdigest()
        integrity = summary.get("deployment_integrity", {})
        expected = integrity.get("manifest", {}).get("expected_sha256") if isinstance(integrity, dict) else None
        if expected is not None and expected != digest:
            return {}, [], ["Recorded source manifest checksum mismatch; source version is unknown."]
        value = json.loads(raw)
        rows = value.get("evaluation_source_files", []) if isinstance(value, dict) else []
        hashes = {row["path"]: row["sha256"] for row in rows if isinstance(row, dict) and isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str)}
        return hashes, [{"path": str(candidate), "size": len(raw), "sha256": digest, "role": "task_contract_source_manifest"}], []
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
        return {}, [], [f"Task rule source manifest unavailable: {type(exc).__name__}"]


def _vector(value, length=None):
    if not isinstance(value, (list, tuple, np.ndarray)) or not len(value) or (length and len(value) != length):
        return None
    result = [_number(v) for v in value]
    return np.array(result) if all(v is not None for v in result) else None


def _derived(values):
    values = deepcopy(values)
    loads = _vector(values.get("fingertip_normal_force_n"), 5)
    if loads is not None:
        values.setdefault("thumb_normal_force_n", float(loads[0]))
        values.setdefault("other_fingers_normal_force_n", float(loads[1:].sum()))
    a, b = _number(values.get("robot_contact_count")), _number(values.get("right_hand_contact_count"))
    if a is not None and b is not None and a >= b:
        values.setdefault("other_robot_contact_count", a - b)
    pose = _vector(values.get("bulb_pose_wxyz"), 7)
    if pose is not None:
        w, x, y, z = pose[3:]
        if abs(float(np.linalg.norm(pose[3:])) - 1) <= 1e-5:
            values.setdefault("bulb_yaw_rad", math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return values


def _predicate(values, rule, *, diagnostic=False):
    field, op = rule["field"], rule["op"]
    value = values.get(field)
    alias = None
    if diagnostic and value is None:
        if field == "ink_remaining" and rule.get("reduce") == "max":
            value, alias = values.get("maximum_ink_remaining"), "maximum_ink_remaining (monitor summary, diagnostic only)"
        elif field == "eraser_lift_m":
            value, alias = values.get("maximum_lift_m"), "maximum_lift_m (history summary, diagnostic only)"
    if op == "is":
        return (value is rule["value"] if type(value) is bool else None), alias
    if rule.get("reduce") and not alias:
        vector = _vector(value, rule.get("length"))
        if vector is None:
            return None, alias
        if field in {"dirt_remaining", "ink_remaining"} and np.any((vector < 0) | (vector > 1)):
            return None, alias
        if field.endswith("force_n") and np.any(vector < 0):
            return None, alias
        reducer = rule["reduce"]
        value = float({"min": np.min, "max": np.max, "sum": np.sum, "mean": np.mean, "norm": np.linalg.norm}[reducer](vector))
    else:
        value = _number(value)
    if value is None or not math.isfinite(value):
        return None, alias
    # Counts/speeds/normal loads have explicit nonnegative physical domains.
    if (field.endswith("contact_count") or field.endswith("speed_m_s") or field.endswith("speed_rad_s") or field in {
            "table_support_force_n", "backstop_load_n", "maximum_socket_penetration_m", "orientation_error_rad",
            "robot_contact_load_n", "right_hand_contact_load_n", "socket_contact_load_n"}) and value < 0:
        return None, alias
    target = rule["value"]
    if op.startswith("abs_"):
        value, op = abs(value - rule.get("center", 0)), op[4:]
    return bool({"lt": value < target, "le": value <= target, "gt": value > target,
                 "ge": value >= target, "eq": value == target}[op]), alias


def _conditions(samples, contract, diagnostic):
    states, reports = {}, []
    for name, predicates in contract["conditions"].items():
        result, aliases = [], set()
        for row in samples:
            checks = [_predicate(row["values"], rule, diagnostic=diagnostic) for rule in predicates] if row["valid"] else [(None, None)]
            result.append(_and(value for value, _ in checks))
            aliases.update(alias for _, alias in checks if alias)
        states[name] = result
        known = sum(v is not None for v in result)
        reports.append({"name": name, "status": "observed" if True in result else "not_observed" if known else "missing",
            "last_status": _status(result[-1]) if result else "unknown", "sample_count": len(result),
            "known_sample_count": known, "positive_sample_count": result.count(True),
            "coverage_fraction": known / len(result) if result else None,
            "predicates": deepcopy(predicates), "diagnostic_aliases": sorted(aliases),
            "evidence_kind": "recorded_numeric_predicate_recomputation", "ground_truth": False})
    return states, reports


def _intervals(times, states, gaps):
    result = []
    for name, values in states.items():
        start = None
        for i, value in enumerate(values):
            if value is True and start is None:
                start = i
            boundary = i == len(values) - 1 or gaps[i] or values[i + 1] is not True
            if start is not None and boundary:
                result.append({"name": name, "start_s": times[start], "end_s": times[i],
                    "sample_count": i - start + 1, "evidence_kind": "sampled_primitive_predicate", "ground_truth": False,
                    "interpretation": "Both endpoints are observed; no extrapolation after the last true sample."})
                start = None
    return result


def _run_program(samples, states, contract, gaps):
    """Conservative ordered witness search; timers never bridge unknown samples."""
    phases, engine = contract["phases"], contract["engine"]
    events, acquired, first = [], 0, None
    terminal_seen = False
    failure = None
    bottom_since = seated_since = None
    bottom_confirmed = False
    bulb_yaw = []
    last_engaged = None
    maximum_removed = maximum_torque = 0.0
    settling = []
    for i, row in enumerate(samples):
        t, values = row["time_s"], row["values"]
        if i and gaps[i - 1]:
            first = bottom_since = seated_since = None
            bottom_confirmed = False
            bulb_yaw = []
            settling = []
        if engine in {"ram", "usb"}:
            near, stationary = states["near_bottom"][i], states["stationary"][i]
            if near is not True:
                bottom_confirmed, bottom_since = False, None
            loaded = _and([near, stationary, states["bottom_load"][i]] + ([states["current_bottom_support"][i]] if engine == "usb" else []))
            if loaded is True:
                bottom_since = t if bottom_since is None else bottom_since
                if t - bottom_since >= phases[0]["dwell_s"] - 1e-12:
                    if not bottom_confirmed:
                        events.append({"name": "bottom_out", "time_s": t, "ground_truth": False})
                    bottom_confirmed = True
            elif not bottom_confirmed:
                bottom_since = None
            seated = _and([near, stationary, True if bottom_confirmed else states["current_bottom_support"][i]])
            seated_since = (t if seated_since is None else seated_since) if seated is True else None
            success = seated_since is not None and t - seated_since >= phases[1]["dwell_s"] - 1e-12 and (bottom_confirmed or engine == "usb")
            terminal_seen = bool(success)
            acquired = 2 if success else 1 if bottom_confirmed else 0
            continue
        if engine == "bulb_unscrew":
            engaged = values.get("engaged")
            turns = _number(values.get("clockwise_turns"))
            torque = _number(values.get("hand_counterclockwise_torque_nm"))
            if engaged is True and turns is not None:
                maximum_removed = max(maximum_removed, 100 / 360 - turns)
            if torque is not None:
                maximum_torque = max(maximum_torque, torque)
            if last_engaged is True and engaged is False:
                required = 100 / 360 + .00035 / .004 - 10 / 360
                if acquired < 3 or maximum_removed < required:
                    failure = "premature_thread_disengagement"
            if type(engaged) is bool:
                last_engaged = engaged
        if engine == "sponge" and acquired in {1, 2}:
            if states["grasp"][i] is False:
                failure = "five_finger_contact_lost_before_placement"
            placing = _or([states["placement_zone"][i], states["placement_support"][i]])
            if acquired == 2 and placing is False and states["carry"][i] is False:
                failure = "carry_clearance_or_support_failed"
        if acquired >= len(phases):
            terminal_seen = True
            continue
        phase = phases[acquired]
        condition = _and(states[name][i] for name in phase["conditions"])
        if phase.get("any_conditions"):
            condition = _and([condition, _or(states[name][i] for name in phase["any_conditions"])])
        if engine == "bulb_screw" and acquired == 0:
            yaw = _number(values.get("bulb_yaw_rad"))
            condition = _and([condition, None if yaw is None else True])
            if condition is True:
                bulb_yaw.append((t, yaw))
                while len(bulb_yaw) > 1 and t - bulb_yaw[1][0] >= .3:
                    bulb_yaw.pop(0)
                if float(np.ptp(np.unwrap([v for _, v in bulb_yaw]))) > math.radians(.3):
                    first = t
                    bulb_yaw = [(t, yaw)]
            else:
                bulb_yaw = []
        if engine == "sponge" and phase["name"] == "settled":
            height = [_number(values.get(key)) for key in ("sponge_pin_z_m", "sponge_centroid_z_m", "sponge_bottom_z_m")]
            impulse = _number(values.get("plate_base_support_impulse_ns"))
            condition = _and([condition, None if None in height or impulse is None else True])
            if condition is True:
                settling.append((t, height, impulse))
                if np.any(np.ptp([v[1] for v in settling], axis=0) > .003):
                    first, settling = t, [(t, height, impulse)]
                elapsed = t - settling[0][0]
                if elapsed >= 1 and (impulse - settling[0][2]) / elapsed < .2:
                    condition = False
            else:
                settling = []
        if condition is not True:
            first = None
            continue
        first = t if first is None else first
        if t - first >= phase["dwell_s"] - 1e-12:
            events.append({"name": phase["name"], "time_s": t, "start_s": first,
                           "evidence_kind": "sampled_primitive_predicate", "ground_truth": False})
            acquired += 1
            first = None
            terminal_seen = acquired == len(phases)
    final = _and(states[name][-1] for name in contract["terminal_conditions"]) if samples else None
    if failure:
        value = False
    elif terminal_seen and final is True:
        value = True
    else:
        value = False
    return value, events, {"completed_phase_count": acquired, "phase_count": len(phases),
        "last_terminal_conditions": _status(final), "observed_failure": failure,
        "next_phase": phases[acquired]["name"] if acquired < len(phases) else None}


def review_task(trial, config=None, *, evidence=None):
    """Review one task without replacing its recorded success label.

    ``evidence`` accepts EvidenceBundle or its ``task_evidence()`` mapping.
    ``config`` can select ``contract_id`` + ``contract_sha256``; thresholds are
    immutable here, so arbitrary overrides cannot silently change the protocol.
    """
    config = deepcopy(config or {})
    if not isinstance(config, dict):
        raise ValueError("task review config must be a mapping")
    if hasattr(evidence, "task_evidence"):
        evidence = evidence.task_evidence()
    evidence = evidence if isinstance(evidence, dict) else {}
    recorded = _recorded(trial.summary)
    contract = get_contract(trial.task)
    result = {"schema_version": "evaluation-task-review-v2", "task": trial.task,
        "supported_task": contract is not None, "recorded_outcome": recorded,
        "review": {"status": "unknown", "availability": "missing", "basis": "none"},
        "required_evidence": [], "missing_evidence": [], "rule_provenance": [],
        "diagnostic_predicates": [], "condition_intervals": [], "events": [],
        "warnings": [], "provenance": [], "original_outcome_preserved": True,
        "limitations": ["This does not rerun physics or validate sensor calibration.",
            "Controller stages and recorded success/phase booleans are never independent task ground truth.",
            "Sampled conditions do not prove continuous behavior between samples; independent means recomputation from declared primitives, not independently authenticated sensors."]}
    if contract is None:
        result["review"]["reason"] = "No explicit task contract; the recorded outcome is preserved without guessing from field names."
        return result
    result.update(contract_id=contract["contract_id"], contract_sha256=contract["contract_sha256"],
                  required_evidence=deepcopy(contract["required_evidence"]), rule_provenance=deepcopy(contract["rule_provenance"]),
                  phase_definitions=deepcopy(contract["phases"]), event_definitions=deepcopy(contract["conditions"]))
    hashes, provenance, warnings = _manifest(trial)
    result["provenance"].extend(provenance)
    result["warnings"].extend(warnings)
    source_rows = [{"path": row["path"], "expected_sha256": row["sha256"], "recorded_sha256": hashes.get(row["path"]),
                    "matches": hashes.get(row["path"]) == row["sha256"]} for row in contract["rule_provenance"]]
    result["source_match"] = {"status": "matched" if source_rows and all(row["matches"] for row in source_rows) else "unverified",
                              "files": source_rows, "scope": "source identity only; not task evidence"}
    selection = config if config.get("contract_id") else evidence.get("contract", {})
    selection = selection if isinstance(selection, dict) else {}
    selected = selection.get("contract_id") == contract["contract_id"] and selection.get("contract_sha256") == contract["contract_sha256"]
    result["contract_selection"] = {"status": "explicit_match" if selected else "unverified", "contract_id": selection.get("contract_id"),
        "reason": "This observable review protocol is explicit and separate from the original monitor; source matching alone does not silently select a new acceptance policy."}
    sidecar = bool(evidence.get("task_samples"))
    raw = evidence.get("task_samples", []) if sidecar else [{"time_s": row.get("sim_time"), "values": row.get("task", {}),
        "source": row.get("_source", "control_steps.jsonl"), "valid": True} for row in trial.control_rows if isinstance(row, dict)]
    if sidecar and evidence.get("task") != trial.task:
        result["review"]["reason"] = "Evidence task differs from trial task; cross-task evidence was rejected."
        result["missing_evidence"] = result["required_evidence"]
        return result
    samples = []
    bad_clock = False
    for row in raw:
        if not isinstance(row, dict) or _number(row.get("time_s")) is None:
            bad_clock = True
            break
        values = row.get("values") if isinstance(row.get("values"), dict) else {}
        samples.append({"time_s": _number(row["time_s"]), "values": _derived(values),
                        "valid": row.get("valid") is True, "source": row.get("source")})
    times = [row["time_s"] for row in samples]
    bad_clock |= any(b <= a for a, b in zip(times, times[1:]))
    if bad_clock or not samples:
        result["review"]["reason"] = "Missing or non-increasing task evidence clock; rows are not silently dropped, sorted or interpolated."
        result["missing_evidence"] = result["required_evidence"]
        return result
    differences = np.diff(times)
    nominal = float(np.median(differences)) if len(differences) else None
    factor = _number(config.get("max_gap_factor", 1.5))
    if factor is None or factor < 1:
        raise ValueError("max_gap_factor must be finite and >= 1")
    gaps = [bool(dt > factor * nominal + 1e-10) for dt in differences] + [False] if nominal is not None else [False]
    field_sources = evidence.get("field_sources", {}) if sidecar else {}
    field_sources = field_sources if isinstance(field_sources, dict) else {}
    rejected_fields = []
    independent = []
    derivation_inputs = {"thumb_normal_force_n": ["fingertip_normal_force_n"], "other_fingers_normal_force_n": ["fingertip_normal_force_n"],
                         "other_robot_contact_count": ["robot_contact_count", "right_hand_contact_count"], "bulb_yaw_rad": ["bulb_pose_wxyz"]}
    for key in {key for row in samples for key in row["values"]}:
        declaration = field_sources.get(key, {})
        kind = declaration.get("kind") if isinstance(declaration, dict) else None
        derived_ok = key in derivation_inputs and all(isinstance(field_sources.get(k), dict) and field_sources[k].get("kind") in _KINDS for k in derivation_inputs[key])
        if kind in _KINDS or derived_ok:
            independent.append(key)
        elif sidecar:
            rejected_fields.append(key)
    diagnostic, predicates = _conditions(samples, contract, True)
    result["diagnostic_predicates"] = predicates
    result["monitor_observations"] = {key: _finite(samples[-1]["values"][key]) for key in samples[-1]["values"]
        if type(samples[-1]["values"][key]) is bool and key not in independent}
    clean_samples = [{**row, "values": {key: value for key, value in row["values"].items() if key in independent}} for row in samples]
    if contract["engine"] == "sponge":
        # The original monitor's recent support mean is a derived history, not
        # primitive evidence. Recompute it from recorded cumulative impulse.
        for i, row in enumerate(clean_samples):
            j = max(0, int(np.searchsorted(times, times[i] - .25, side="right")) - 1)
            impulses = [_number(sample["values"].get("plate_base_support_impulse_ns")) for sample in clean_samples[j:i + 1]]
            valid_impulses = (bool(impulses) and None not in impulses and
                              all(b >= a for a, b in zip(impulses, impulses[1:])) and not any(gaps[j:i]))
            elapsed = times[i] - times[j]
            row["values"]["recent_plate_base_mean_support_force_n"] = (
                (impulses[-1] - impulses[0]) / elapsed if valid_impulses and elapsed >= .24 else
                0.0 if valid_impulses else None)
        if all(row["values"].get("recent_plate_base_mean_support_force_n") is not None for row in clean_samples):
            independent.append("recent_plate_base_mean_support_force_n")
    states, verified_predicates = _conditions(clean_samples, contract, False)
    result["predicate_review"] = verified_predicates
    result["condition_intervals"] = _intervals(times, states if sidecar else diagnostic, gaps)
    result["missing_evidence"] = sorted({field for field in result["required_evidence"] if not any(field in row["values"] for row in clean_samples)})
    if contract["engine"] == "bulb_screw":
        if not any("bulb_yaw_rad" in row["values"] for row in clean_samples):
            result["missing_evidence"] = sorted(set(result["missing_evidence"] + ["bulb_yaw_rad"]))
    if contract["engine"] == "sponge":
        extra = ["plate_base_support_impulse_ns", "sponge_pin_z_m", "sponge_centroid_z_m"]
        result["missing_evidence"] = sorted(set(result["missing_evidence"] + [field for field in extra if not any(field in row["values"] for row in clean_samples)]))
    extra_fields = {"bulb_screw": ["bulb_yaw_rad"], "sponge": ["plate_base_support_impulse_ns", "sponge_pin_z_m", "sponge_centroid_z_m"]}.get(contract["engine"], [])
    coverage = (all(row["valid"] for row in clean_samples) and
                all(v is not None for series in states.values() for v in series) and not any(gaps) and
                all(_number(row["values"].get(field)) is not None for row in clean_samples for field in extra_fields))
    complete = evidence.get("complete") is True and coverage
    value, events, progress = _run_program(clean_samples, states, contract, gaps)
    result["events"] = events
    result["progress"] = progress
    result["coverage"] = {"sample_count": len(samples), "start_s": times[0], "end_s": times[-1],
        "nominal_interval_s": nominal, "gap_count": sum(gaps), "complete_declared": evidence.get("complete") is True,
        "all_required_predicate_samples_known": coverage, "sampling": evidence.get("sampling"),
        "rejected_fields": sorted(rejected_fields), "independent_field_count": len(independent)}
    reasons = []
    if not selected:
        reasons.append("No explicitly selected matching contract id and hash; current source thresholds are reference diagnostics only.")
    if not sidecar:
        reasons.append("Only original monitor measurements are archived; no independently declared primitive evidence sidecar.")
    if not complete:
        reasons.append("Primitive evidence is missing, invalid, gapped, or not declared complete; absence of a witnessed success is not a certified failure.")
    if result["missing_evidence"]:
        reasons.append("Required primitive fields are missing.")
    if not recorded["finalized"]:
        reasons.append("Trial is not finalized; provisional false labels cannot establish failure.")
    if recorded["physics_invalid_recorded"]:
        reasons.append("Recorded physics is invalid; task outcome cannot be independently certified.")
    if recorded["conflict"]:
        reasons.append("Recorded outcome labels conflict; resolve provenance before assigning a reviewed verdict.")
    known = selected and sidecar and complete and not result["missing_evidence"] and recorded["finalized"] and not recorded["physics_invalid_recorded"] and not recorded["conflict"]
    status = _status(value) if known else "unknown"
    kinds = {field_sources[k].get("kind") for k in independent if isinstance(field_sources.get(k), dict)}
    result["review"] = {"status": status, "availability": "available" if known else "partial" if independent else "missing",
        "basis": "simulation_oracle_predicate_recomputation" if "simulation_oracle" in kinds else "declared_primitive_recomputation" if independent else "original_monitor_only",
        "reason": " ".join(reasons) if reasons else "Recomputed the selected observable contract from complete declared primitive samples; no original success booleans were used.",
        "matches_recorded_outcome": status == recorded["status"] if status != "unknown" and recorded["status"] != "unknown" else None,
        "independent_physics_validation": False}
    result["provenance"].extend(deepcopy(evidence.get("provenance", [])))
    return _finite(result)


__all__ = ["review_task"]
