"""Frozen, content-addressed definitions for offline evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy


DEFAULT_PROTOCOL = {
    "schema_version": "task-tactile-protocol-v1",
    "contact_on_n": 0.05,
    "contact_off_n": 0.03,
    "board_contact_threshold_n": 0.05,
    "min_contact_duration_s": 0.1,
    "analysis_bin_s": 0.1,
    "change_lag_s": 0.1,
    "max_gap_factor": 1.5,
    "reference_required": False,
    "cross_trial_time_alignment": "none",
    "task_scope_boundary_policy": "require_native_sample_boundaries_otherwise_unavailable",
    "aggregation_unit": "episode",
    "cross_task_weighting": "equal_task",
    "unknown_outcomes": "retain_in_all_trial_denominator_and_report_separately",
    "force_threshold_semantics": "contact_detection_only_not_safety_or_effectiveness",
    "safety_threshold_source": "recorded_task_contract_only",
    "change_semantics": "complete_fixed_physical_time_bins_not_sensor_bandwidth_equalization",
    "slip_semantics": "unavailable_without_validated_relative_motion_evidence",
    "expert_reference": "not_implemented_in_stage1",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_protocol(overrides: dict | None = None) -> dict:
    """Reject unknown and semantic overrides rather than silently changing a run."""
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Protocol must be a JSON object")
    result = deepcopy(DEFAULT_PROTOCOL)
    tunable = {
        "contact_on_n", "contact_off_n", "board_contact_threshold_n", "min_contact_duration_s",
        "analysis_bin_s", "change_lag_s", "max_gap_factor",
    }
    if overrides:
        unknown = set(overrides) - set(DEFAULT_PROTOCOL)
        if unknown:
            raise ValueError(f"Unknown protocol fields: {sorted(unknown)}")
        for key, value in overrides.items():
            if key not in tunable and value != DEFAULT_PROTOCOL[key]:
                raise ValueError(f"Unsupported protocol semantics: {key}")
            result[key] = value
    for key in tunable:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite and numeric")
    if not 0 <= result["contact_off_n"] < result["contact_on_n"]:
        raise ValueError("Require 0 <= contact_off_n < contact_on_n")
    if result["min_contact_duration_s"] < 0:
        raise ValueError("min_contact_duration_s must be nonnegative")
    if result["board_contact_threshold_n"] < 0:
        raise ValueError("board_contact_threshold_n must be nonnegative")
    if result["analysis_bin_s"] <= 0 or result["change_lag_s"] <= 0:
        raise ValueError("Bin width and lag must be positive")
    if result["max_gap_factor"] < 1:
        raise ValueError("max_gap_factor must be at least 1")
    ratio = result["change_lag_s"] / result["analysis_bin_s"]
    if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8) or round(ratio) < 1:
        raise ValueError("change_lag_s must be a positive integer multiple of analysis_bin_s")
    return result
