"""Describe observed validation coverage without claiming unrun experiments."""
from __future__ import annotations

from collections import Counter


def coverage_report(reports: list[dict], catalog: list[dict]) -> dict:
    sources = [report["trial"].get("source") for report in reports if report["trial"].get("source")]
    if len(sources) != len(set(sources)):
        raise ValueError("Repeated source trials must not be counted as independent validation")
    tasks = {entry["task"]: entry for entry in catalog}
    tasks.update({report["trial"]["task"]: tasks.get(report["trial"]["task"], {}) for report in reports})
    rows = []
    for task in sorted(tasks):
        selected = [report for report in reports if report["trial"]["task"] == task]
        outcomes = Counter(report.get("outcome", {}).get("status", "unknown") for report in selected)
        evidence = Counter(report.get("evidence_v2", {}).get("status", "missing") for report in selected)
        review = Counter((report.get("task_review", {}).get("review") or {}).get("status", "unknown") for report in selected)
        seeds = sorted({report["trial"]["seed"] for report in selected
                        if isinstance(report["trial"].get("seed"), int) and not isinstance(report["trial"].get("seed"), bool)})
        gaps = []
        if not selected:
            gaps.append("No archived trial from this task was supplied")
        if not outcomes["pass"]:
            gaps.append("No recorded successful trial in this supplied set")
        if not outcomes["fail"]:
            gaps.append("No recorded failed trial in this supplied set")
        if len(seeds) < 2:
            gaps.append("Multiple distinct recorded seeds are not represented")
        if any(report.get("evidence_v2", {}).get("source") != "sidecar" or
               report.get("evidence_v2", {}).get("sampled_complete") is not True or
               report.get("evidence_v2", {}).get("recorded_complete") is not True or
               report.get("evidence_v2", {}).get("status") != "available"
               for report in selected) or not selected:
            gaps.append("Complete versioned evidence sidecars are not available for every trial")
        rows.append({"task": task, "contract_available": bool(tasks[task]),
                     "trial_count": len(selected), "recorded_outcome_counts": dict(outcomes),
                     "review_counts": dict(review), "evidence_counts": dict(evidence),
                     "recorded_seeds": seeds, "coverage_gaps": gaps})
    return {"schema_version": "evaluation-validation-coverage-v1", "status": "descriptive_only",
            "trial_count": len(reports), "task_count_with_trials": sum(row["trial_count"] > 0 for row in rows),
            "tasks": rows,
            "not_established": ["Independent manual video/physics annotation agreement",
                                "Controlled perturbation robustness across tasks and configurations",
                                "Causal benefit from the policy actually using tactile observations",
                                "Real-hardware calibration and transfer"],
            "semantics": "Counts describe only supplied independent trial sources. Re-exported versions are not new trials. Unit tests are not new simulation experiments; no sample-count threshold certifies benchmark completeness."}
