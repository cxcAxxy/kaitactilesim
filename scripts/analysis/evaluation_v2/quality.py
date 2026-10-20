"""Contact-pair quality from explicitly recorded, independently scoped evidence.

No expert trajectory, universal safety force, policy-declared phase, or normal
force fluctuation is converted into a slip/expected-contact ground truth.
Durations are sampled left-hold estimates; gaps and unknowns retain coverage.
"""
from __future__ import annotations

import math

from .evidence import EvidenceBundle


def _coverage(known, weights, span):
    duration = sum(weight for valid, weight in zip(known, weights) if valid)
    fraction = min(1., max(0., duration / span)) if span > 0 else None
    return {"status": "available" if fraction is not None and fraction >= 1 - 1e-9 else
                      "partial" if duration > 0 else "missing",
            "known_duration_s": duration, "unknown_duration_s": max(0., span - duration),
            "coverage_fraction": fraction, "known_sample_count": sum(known)}


def _duration(values, weights, span):
    known = [value is not None for value in values]
    result = _coverage(known, weights, span)
    result.update(observed_duration_s=sum(weight for value, weight in zip(values, weights) if value is True),
                  positive_sample_count=sum(value is True for value in values))
    return result


def _independent_label(contact, key):
    label = contact.get(key)
    if isinstance(label, dict) and label.get("independent") is True and type(label.get("value")) is bool:
        return label["value"]
    return None


def _episodes(values, times, weights):
    episodes, active = [], None
    for index, value in enumerate(values):
        if value is not True or weights[index] <= 0:
            active = None
            continue
        if active is None:
            active = {"start_s": times[index], "end_s": times[index] + weights[index], "duration_s": weights[index]}
            episodes.append(active)
        else:
            active["end_s"] = times[index] + weights[index]
            active["duration_s"] += weights[index]
    return episodes


def _load_report(values, threshold, weights, span, threshold_source, unit="N"):
    known = [value is not None for value in values]
    coverage = _coverage(known, weights, span)
    native = [value for value in values if value is not None]
    coverage.update(sampled_peak=max(native) if native else None, unit=unit,
                    duration_weighted_mean=(sum(value * weight for value, weight in zip(values, weights) if value is not None)
                                            / coverage["known_duration_s"] if coverage["known_duration_s"] > 0 else None))
    if threshold is None:
        coverage["overload"] = {"status": "unknown", "reason": "No explicit task-specific threshold supplied",
                                "threshold_n": None, "observed_duration_s": None, "excess_impulse_ns": None}
    else:
        exceeded = [value > threshold if value is not None else None for value in values]
        overload = _duration(exceeded, weights, span)
        overload.update(threshold_n=threshold, threshold_source=threshold_source,
                        criterion_status="fail" if any(value is True for value in exceeded) else
                                         "pass" if coverage["status"] == "available" and all(known) else "unknown",
                        excess_impulse_ns=sum(max(0., value - threshold) * weight for value, weight in zip(values, weights) if value is not None),
                        semantics="Observed excess above the supplied threshold only; not a universal safety certificate. Native final peaks retain zero duration.")
        coverage["overload"] = overload
    return coverage


def _pair_report(pair, records, name, times, weights, span):
    contacts = [row.get("contacts", {}).get(name, {}) for row in records]
    valid = [row["valid"] and contact.get("valid") is True for row, contact in zip(records, contacts)]
    forces, tangent, presence, expected, slip_labels, impact_labels, speeds, impulses = [], [], [], [], [], [], [], []
    thresholds = pair.get("thresholds", {})
    for row, contact, okay in zip(records, contacts, valid):
        load_ok = okay and contact.get("saturated") is not True
        force = contact.get("normal_force_n") if load_ok else None
        forces.append(force)
        vector = contact.get("tangent_force_n") if load_ok else None
        tangent.append(math.hypot(*vector) if vector is not None else None)
        present = contact.get("contact_present") if okay else None
        if present is None and force is not None and "contact_on_n" in thresholds:
            present = force > thresholds["contact_on_n"]
        presence.append(present)
        # Expected contact is independent of whether the contact sensor worked.
        expected.append(_independent_label(contact, "expected_contact") if row["valid"] else None)
        slip_labels.append(_independent_label(contact, "slip") if okay else None)
        impact_labels.append(_independent_label(contact, "impact") if okay else None)
        velocity = contact.get("tangential_relative_velocity_m_s") if okay else None
        speeds.append(math.hypot(*velocity) if velocity is not None else None)
        impulses.append(contact.get("normal_impulse_ns") if load_ok else None)
    contact_duration = _duration(presence, weights, span)
    contact_duration["semantics"] = "Recorded contact predicate, or normal force strictly above the explicitly supplied diagnostic threshold; sampled left-hold duration, not continuous contact certification."
    expected_duration = _duration(expected, weights, span)
    interruptions = [not present if wanted is True and present is not None else
                     False if wanted is False else None for wanted, present in zip(expected, presence)]
    observed_expected = sum(weight for wanted, present, weight in zip(expected, presence, weights) if wanted is True and present is not None)
    interrupted_duration = sum(weight for value, weight in zip(interruptions, weights) if value is True)
    expected_positive = expected_duration["observed_duration_s"]
    interruption = {
        "status": "available" if expected_duration["status"] == "available" and expected_positive > 0 and math.isclose(observed_expected, expected_positive, abs_tol=1e-9) else
                  "partial" if observed_expected > 0 else "missing",
        "expected_duration_s": expected_positive if expected_duration["known_duration_s"] > 0 else None,
        "evaluated_expected_duration_s": observed_expected,
        "unknown_expected_duration_s": max(0., expected_positive - observed_expected),
        "coverage_fraction": observed_expected / expected_positive if expected_positive > 0 else None,
        "observed_interruption_duration_s": interrupted_duration if observed_expected > 0 else None,
        "episodes": _episodes(interruptions, times, weights),
        "criterion_status": "fail" if interrupted_duration > 0 else "pass" if expected_duration["status"] == "available" and expected_positive > 0 and
                            math.isclose(observed_expected, expected_positive, abs_tol=1e-9) else "unknown",
        "semantics": "Only independently declared expected-contact masks are evaluated; controller phases, force presence, and observed contact are never used to manufacture expectation. Separate episodes are not assumed task retries.",
    }
    if expected_positive == 0 and expected_duration["status"] == "available":
        interruption.update(status="not_applicable", reason="No expected-contact interval in the fully observed sampled window")
    allowed = pair.get("allowed_relative_motion", "unknown")
    slip_threshold = thresholds.get("slip_speed_m_s")
    if allowed in {"sliding", "rolling", "any"}:
        relative_slip = {"status": "not_applicable", "reason": "Relative motion is permitted for this contact pair; motion is not classified as unintended slip"}
    elif allowed != "none" or slip_threshold is None:
        relative_slip = {"status": "missing", "reason": "A no-relative-motion contact contract and an explicit speed threshold are required"}
    else:
        slipping = [speed > slip_threshold if present is True and speed is not None else
                    False if present is False else None for present, speed in zip(presence, speeds)]
        relative_slip = _duration(slipping, weights, span)
        relative_slip.update(threshold_m_s=slip_threshold, threshold_source=pair.get("threshold_source"),
                             episodes=_episodes(slipping, times, weights),
                             semantics="Kinematic evidence of tangential contact-point relative motion during a no-motion contact contract; not a calibrated incipient-slip detector, object drop label, or normal-force-change heuristic.")
    return {
        "body_a": pair["body_a"], "body_b": pair["body_b"], "target": pair["target"],
        "role": pair["role"], "allowed_relative_motion": allowed,
        "measurement_source": pair["measurement_source"], "contact": contact_duration,
        "correct_target_contact": contact_duration if pair["target"] else {"status": "not_applicable", "reason": "Pair is not declared a task target"},
        "non_target_contact": contact_duration if not pair["target"] else {"status": "not_applicable"},
        "expected_contact": expected_duration, "expected_contact_interruptions": interruption,
        "normal_load": _load_report(forces, thresholds.get("max_normal_force_n"), weights, span, pair.get("threshold_source")),
        "tangent_load": _load_report(tangent, thresholds.get("max_tangent_force_n"), weights, span, pair.get("threshold_source")),
        "independently_labeled_slip": {**_duration(slip_labels, weights, span), "semantics": "Only explicitly independent recorded slip labels; their validation is not independently authenticated here."},
        "relative_motion_supported_slip": relative_slip,
        "tangential_relative_speed": {**_coverage([value is not None for value in speeds], weights, span),
                                      "sampled_peak_m_s": max((value for value in speeds if value is not None), default=None)},
        "independently_labeled_impact": {**_duration(impact_labels, weights, span), "semantics": "Sampled independent impact labels; positive samples are not assumed distinct impact events."},
        "sampled_peak_recorded_normal_impulse_ns": max((value for value in impulses if value is not None), default=None),
        "saturation": _duration([contact.get("saturated") if okay else None for contact, okay in zip(contacts, valid)], weights, span),
    }


def _effects(bundle, times, weights, span):
    result = {}
    for name, descriptor in bundle.metadata.get("effects", {}).items():
        if not descriptor["independent"]:
            result[name] = {"status": "missing", "reason": "Effect source is not declared independent of the scored behavior"}
            continue
        values = [row.get("effects", {}).get(name) if row["valid"] else None for row in bundle.records]
        known = [value is not None for value in values]
        row = _coverage(known, weights, span)
        changes = [values[index + 1] - values[index] for index in range(len(values) - 1)
                   if known[index] and known[index + 1] and weights[index] > 0]
        final = values[-1] if values else None
        has_target = "target_min" in descriptor or "target_max" in descriptor
        target = (descriptor.get("target_min", -math.inf) <= final <= descriptor.get("target_max", math.inf)) if final is not None and has_target else None
        row.update(source=descriptor["source"], unit=descriptor["unit"], contact_pair_ids=descriptor.get("contact_pairs", []),
                   contiguous_observed_change=sum(changes) if changes else None,
                   observed_final_value=final, target_status="pass" if target is True else "fail" if target is False else "unknown",
                   semantics="Independent recorded task effect; only adjacent observed changes are summed, never filled across gaps. Association with contact is not causal attribution or an expert similarity score.")
        result[name] = row
    return result


def assess_contact_quality(bundle: EvidenceBundle) -> dict:
    """Return finite descriptive/conditional results, retaining missing evidence."""
    result = {"schema_version": "contact-quality-v2", "status": bundle.status,
              "source_kind": bundle.metadata.get("source_kind"), "contact_pairs": {}, "effects": {},
              "limitations": [
                  "All durations are source-cadence left-hold estimates; last sample has zero duration and long gaps are unknown.",
                  "No universal task safety threshold or force-derived expected-contact/slip label is introduced.",
                  "Calibration and independent-label declarations are input claims, not authenticated by this analysis.",
                  "Simultaneous pair durations are not added into a global duration; contact identities retain their distinct roles.",
              ], "warnings": list(bundle.warnings)}
    if not bundle.records:
        result.update(status="missing", reason="No recorded versioned or explicitly adapted contact evidence",
                      duration_s=None, sampling=bundle.sampling)
        return result
    times = bundle.time_s
    span = times[-1] - times[0]
    sampling = bundle.sampling
    weights = [(times[index + 1] - time) if index + 1 < len(times) and not sampling["gap_after"][index] else 0.
               for index, time in enumerate(times)]
    result.update(duration_s=span, sampling=sampling,
                  global_validity=_coverage([row["valid"] for row in bundle.records], weights, span),
                  declared_calibration=bundle.metadata.get("calibration", {}))
    result["contact_pairs"] = {name: _pair_report(pair, bundle.records, name, times, weights, span)
                               for name, pair in bundle.metadata.get("contact_pairs", {}).items()}
    result["effects"] = _effects(bundle, times, weights, span)
    overflow = False
    def finite(value):
        nonlocal overflow
        if isinstance(value, dict): return {key: finite(item) for key, item in value.items()}
        if isinstance(value, list): return [finite(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            overflow = True
            return None
        return value
    result = finite(result)
    if overflow: result["warnings"].append("A derived arithmetic result overflowed; retained as unknown null, never an infinite metric.")
    return result
