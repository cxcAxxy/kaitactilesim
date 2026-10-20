"""Conservative task plugins: preserve outcomes, expose the evidence limitations."""
from __future__ import annotations

from typing import Any
import numpy as np

from .adapters import TrialData


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        return float(value)
    return None


def _status(value: Any) -> str:
    return "pass" if value is True else "fail" if value is False else "unknown"


def _predicate(name: str, state: dict, key: str, *, test=None, rule=None) -> dict:
    value = state.get(key)
    verdict = value if type(value) is bool else None
    if test is not None:
        number = _number(value)
        verdict = bool(test(number)) if number is not None else None
    return {"name": name, "status": _status(verdict), "value": value,
            "rule": rule or "explicit recorded monitor predicate; not independently re-certified",
            "evidence": f"summary.json:evaluation.state.{key}"}


def _termination(summary: dict, evaluation: dict) -> tuple[Any, str]:
    for source, container in (("summary.json", summary), ("summary.json:evaluation", evaluation)):
        for key in ("termination_reason", "stop_reason", "failure_reason"):
            if container.get(key) is not None:
                return container[key], f"{source}.{key}"
    duration, limit = _number(summary.get("sim_seconds")), _number(summary.get("max_sim_seconds"))
    if (summary.get("status") == "task_not_completed" and duration is not None
            and limit is not None and duration >= limit - 1e-6):
        return "time_limit", "inferred from task_not_completed and recorded simulation time budget"
    return summary.get("status"), "summary.json.status (not an independently inferred failure cause)"


def _safety(summary: dict, evaluation: dict) -> dict:
    diagnostics = evaluation.get("contact_guard_diagnostics") or {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    mode = diagnostics.get("mode", evaluation.get("guard_mode", summary.get("guard_mode")))
    enforcement = "enforce" if mode in ("enforce", "strict") else "observe" if mode in ("observe", "observe_only") else "unknown"
    checks = []
    for name, exceeded, peak, threshold in (
        ("board_normal_force", "board_normal_force_exceeded", "peak_board_normal_force_n", "board_normal_force_guard_threshold_n"),
        ("direct_hand_board_contact", "direct_hand_board_contact_observed", "peak_direct_hand_board_force_n", "direct_hand_board_force_guard_threshold_n"),
    ):
        violation = diagnostics.get(exceeded)
        if type(violation) is bool:
            checks.append({"name": name, "status": "fail" if violation else "pass",
                "value": diagnostics.get(peak), "threshold": diagnostics.get(threshold),
                "source": f"summary.json:evaluation.contact_guard_diagnostics.{exceeded}"})
    events = evaluation.get("guard_events", summary.get("guard_events", []))
    events = events if isinstance(events, list) else []
    def occurred(event):
        if not isinstance(event, dict):
            return False
        if type(event.get("triggered")) is bool:
            return event["triggered"]
        if "count" in event:
            return (_number(event["count"]) or 0) > 0
        # guard_events is an occurrence log, unlike a guard configuration list.
        return event.get("kind", event.get("type")) in {
            "force_guard", "collision_guard", "object_dropped", "action_rejected",
            "nonfinite_state", "solver_warning", "joint_limit_guard",
        }
    violations = [event for event in events if occurred(event)]
    failed = bool(violations) or any(check["status"] == "fail" for check in checks)
    return {"status": "fail" if failed else "pass" if checks else "unknown",
        "scope": "recorded_checks_only", "enforcement": enforcement,
        "checks": checks, "recorded_guard_events": events,
        "limitations": ["absence of a recorded guard is not evidence that every interaction was safe",
                        "observe mode does not enforce stopping; force thresholds are task-specific"]}


def _event_report(trial: TrialData, config: dict) -> tuple[list, list, list]:
    rows = []
    invalid_clock = False
    for row in trial.control_rows:
        time = _number(row.get("sim_time"))
        task = row.get("task")
        if time is None:
            invalid_clock = True
        else:
            rows.append((time, task if isinstance(task, dict) else {}, row))
    times = np.array([row[0] for row in rows])
    invalid_clock |= bool(len(times) > 1 and np.any(np.diff(times) <= 0))
    if invalid_clock:
        trial.warnings.append("task evidence clock is missing or not strictly increasing; event scopes unavailable")
    median = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    factor = float(config.get("max_gap_factor", 1.5))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("max_gap_factor must be finite and positive")
    gap = factor * median
    events, scopes = [], []
    previous_declared = None
    observed_conditions: dict[str, list] = {}
    if trial.task.startswith("whiteboard-wipe"):
        threshold = float(config.get("board_contact_threshold_n", 0.05))
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("board_contact_threshold_n must be finite and nonnegative")
        observed_conditions["board_contact"] = [
            None if _number(task.get("board_normal_force_n")) is None else task["board_normal_force_n"] > threshold
            for _, task, _ in rows]
    elif trial.task in ("install-ram", "usb-insert"):
        observed_conditions["seated_observation"] = [
            task.get("seated") if type(task.get("seated")) is bool else None for _, task, _ in rows]
    elif trial.task == "sponge-grasp":
        observed_conditions["released_observation"] = [
            bool(task["released"] and task["lift_established"])
            if type(task.get("released")) is bool and type(task.get("lift_established")) is bool else None
            for _, task, _ in rows]
    availability = []
    durations = np.append(np.diff(times), 0.0) if len(times) else np.array([])
    supported = (durations > 0) & (durations <= gap + 1e-12)
    span = float(times[-1] - times[0]) if len(times) > 1 and not invalid_clock else None
    for name, conditions in observed_conditions.items():
        known = np.array([value is not None for value in conditions], dtype=bool)
        positive = np.array([value is True for value in conditions], dtype=bool)
        if invalid_clock:
            known[:] = False
            positive[:] = False
        known_count, positive_count = int(known.sum()), int(positive.sum())
        known_duration = float(durations[supported & known].sum())
        coverage = float(np.clip(known_duration / span, 0., 1.)) if span is not None and span > 0 else None
        availability.append({
            "name": name, "status": "observed" if positive_count else "not_observed" if known_count else "missing",
            "sample_count": len(rows), "known_sample_count": known_count,
            "positive_sample_count": positive_count,
            "sample_coverage_fraction": known_count / len(rows) if rows else None,
            "known_duration_s": known_duration,
            "observed_duration_s": float(durations[supported & positive].sum()),
            "start_s": float(times[0]) if len(times) and not invalid_clock else None,
            "end_s": float(times[-1]) if len(times) and not invalid_clock else None,
            "span_s": span, "coverage_fraction": coverage,
            "fully_observed": bool(known_count == len(rows) and len(rows) > 1 and coverage is not None and coverage >= 1 - 1e-9),
            "evidence_kind": "sampled_task_monitor_predicate",
            "evidence": [rows[0][2].get("_source"), rows[-1][2].get("_source")] if rows else [],
            "interpretation": "not_observed means the predicate was not seen in available samples; it does not establish that the physical task phase was never reached",
            "reason": "invalid task clock" if invalid_clock else "no known predicate samples" if not known_count else None,
        })
    if invalid_clock:
        return [], [], availability
    for time, task, row in rows:
        declared = task.get("stage", row.get("stage_after_action"))
        if declared is not None and declared != previous_declared:
            events.append({"type": "declared_stage", "name": declared, "time_s": time,
                "evidence": row.get("_source"), "evidence_kind": "recorded_monitor_or_controller_declaration",
                "ground_truth": False})
        previous_declared = declared
    for name, conditions in observed_conditions.items():
        active = None
        episode = 0
        for index, (time, task, row) in enumerate(rows):
            duration = times[index + 1] - time if index + 1 < len(times) else 0.0
            # An unknown sample, gap or final sample does not establish an interval.
            valid = conditions[index] is True and 0 < duration <= gap + 1e-12
            if not valid:
                active = None
                continue
            if active is None:
                episode += 1
                active = {"name": name, "start_s": time, "end_s": time + duration,
                    "episode_index": episode, "evidence": row.get("_source"),
                    "scope": "conditional_on_recorded_sample_predicate",
                    "timing": "sampled left-hold; no sub-sample contact boundary claim",
                    "sampling": row.get("_sampling", "control_steps"),
                    "attempt_id": None, "attempt_semantics": "contact episodes are not inferred task retries"}
                scopes.append(active)
                events.append({"type": "observed_predicate_start", "name": name, "time_s": time,
                    "evidence": row.get("_source"), "evidence_kind": "sampled_task_monitor_measurement",
                    "episode_index": episode})
            else:
                active["end_s"] = time + duration
    return sorted(events, key=lambda event: event["time_s"]), scopes, availability


def build_task_report(trial: TrialData, config: dict) -> dict:
    """Do not turn tactile availability, similarity or controller stages into success."""
    summary = trial.summary
    evaluation = summary.get("evaluation") or {}
    state = evaluation.get("state") or {}
    state = state if isinstance(state, dict) else {}
    success = evaluation.get("success")
    source = "summary.json:evaluation.success"
    if type(success) is not bool and type(summary.get("success")) is bool:
        success, source = summary["success"], "summary.json:success"
    if type(success) is not bool:
        success = None
    status = _status(success)
    reason = "recorded task monitor outcome; no independent outcome certification"
    if summary.get("status") in {"running", "queued", "pending", "starting", "in_progress"} or summary.get("completed") is False:
        status, reason = "unknown", "trial is explicitly nonterminal; provisional success label is not a final outcome"
    if type(state.get("success")) is bool and success is not None and state["success"] != success:
        status, reason = "unknown", "conflicting explicit success labels in summary"
    if (summary.get("evaluation_physics_invalid") is True or evaluation.get("evaluation_physics_invalid") is True
            or summary.get("physics_valid") is False or evaluation.get("physics_valid") is False):
        status, reason = "unknown", "recorded invalid simulation physics; label retained but outcome cannot be certified"
    termination, termination_source = _termination(summary, evaluation)
    specific = {"plugin": "generic_recorded_outcome", "predicates": [], "partial_state": state,
                "limitations": ["task semantics are not inferred from arbitrary field names"]}
    predicates = specific["predicates"]
    if trial.task.startswith("whiteboard-wipe"):
        specific["plugin"] = "whiteboard_wipe"
        predicates.extend([
            _predicate("recorded_maximum_ink_remaining", state, "maximum_ink_remaining",
                       rule="quantity only; task-version completion tolerance is not re-certified"),
            _predicate("recorded_lift_clearance", state, "maximum_lift_m",
                       rule="quantity only; task-version lift threshold and grasp evidence are not inferred"),
            _predicate("recorded_table_release", state, "released_on_table"),
        ])
        specific["cleaned_fraction"] = state.get("cleaned_fraction")
        specific["limitations"] = ["cleaned_fraction is simulated ink-cell progress, not measured image area or physical ink mass",
            "released_on_table can be true before pickup; it alone is not a return-and-release event",
            "holding, loaded sliding and valid release events require additional synchronized evidence",
            "board contact uses a measurement threshold, not the controller's wiping stage"]
    elif trial.task in ("install-ram", "usb-insert"):
        specific["plugin"] = "insertion"
        predicates.append(_predicate("recorded_seated", state, "seated"))
        if trial.task == "install-ram":
            predicates.extend([_predicate("recorded_aperture_alignment", state, "aperture_fits"),
                               _predicate("recorded_bottom_out", state, "bottom_out_confirmed")])
        specific["limitations"] = ["insertion depth alone does not prove alignment, seating, release, or electrical function",
                                   "recorded monitor success is not re-certified with guessed task thresholds"]
    elif trial.task == "sponge-grasp":
        specific["plugin"] = "sponge_grasp"
        for key in ("grasp_established", "lift_established", "transport_contact_valid", "release_allowed", "released"):
            predicate = _predicate(f"recorded_{key}", state, key)
            if key in ("transport_contact_valid", "release_allowed", "released") and state.get("lift_established") is False:
                predicate.update(status="unknown", applicability="not_reached",
                    reason="recorded lift was not established; a later-phase flag is not a measured phase outcome")
            predicates.append(predicate)
        specific["limitations"] = ["unreached transport or release phases do not have zero force or perfect stability",
            "a retained transport_contact_valid flag before grasp is not evidence of successful transport",
            "soft-object placement requires task-specific support and geometry, not one rigid-body pose"]
    events, scopes, availability = _event_report(trial, config)
    return {
        "outcome": {"status": status, "recorded_success": success, "source": source,
            "scope": "recorded_task_monitor_result_not_independent_recertification", "reason": reason,
            "termination_reason": termination, "termination_source": termination_source,
            "formal_metrics_valid": summary.get("formal_metrics_valid", evaluation.get("formal_metrics_valid")),
            "diagnostic_only": summary.get("diagnostic_only", evaluation.get("diagnostic_only"))},
        "safety": _safety(summary, evaluation),
        "efficiency": {"simulation_duration_s": _number(summary.get("sim_seconds")),
            "wall_duration_s": _number(summary.get("wall_seconds")),
            "recorded_time_budget_s": _number(summary.get("max_sim_seconds")),
            "termination_reason": termination, "scope": "simulation_and_wall_clocks_kept_separate"},
        "task_specific": specific, "events": events, "scope_intervals": scopes,
        "scope_availability": availability,
    }
