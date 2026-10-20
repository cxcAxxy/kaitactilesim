"""Orchestration and episode-first aggregation; no simulation or expert replay."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import platform

import numpy as np

from . import SCHEMA_VERSION
from .adapters import load_trial
from .protocol import content_hash, make_protocol
from .tasks import build_task_report
from .tactile import DEFAULT_CONFIG as TACTILE_CONFIG, analyze_tactile


def implementation_manifest() -> dict:
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py")) + sorted((package / "web").glob("*.js")) + sorted((package / "web").glob("*.css")) + [package.parent / "evaluate_trials.py"]
    rows = [{"path": str(path.relative_to(package.parent)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths if path.is_file()]
    return {"files": rows, "sha256": content_hash(rows),
            "python": platform.python_version(), "numpy": np.__version__}


def _scope_masks(time: np.ndarray, intervals: list[dict]) -> dict:
    masks = {}
    for row in intervals:
        name = row["name"]
        start, end = float(row["start_s"]), float(row["end_s"])
        if not math.isfinite(start + end) or end < start:
            raise ValueError(f"Invalid evidence interval: {row}")
        masks.setdefault(name, np.zeros(len(time), dtype=bool))
        masks[name] |= (time >= start - 1e-9) & (time < end - 1e-9)
    return masks


def _aligned_scopes(trial, task_report: dict) -> tuple[dict, dict]:
    """Do not silently quantize task boundaries onto a different sample clock.

    V1 supports native-sample-aligned intervals only. Fractional interval
    weighting is deferred; the whole affected scope is unavailable, rather
    than selectively retaining intervals with convenient boundaries.
    """
    grouped = defaultdict(list)
    for row in task_report.get("scope_intervals", []):
        grouped[row["name"]].append(row)
    evidence = {}
    for row in task_report.get("scope_availability", []):
        evidence[row["name"]] = {"status": row["status"],
                                  "reason": "No positive-duration interval supported by sampled task evidence"}
    masks = {}
    for name, intervals in grouped.items():
        if not trial.task_tactile_clock_aligned:
            evidence[name] = {"status": "unavailable", "reason": "Task and tactile clock mapping is unverified"}
            continue
        boundary_ok = all(np.any(np.isclose(trial.time_s, row[key], rtol=0, atol=1e-9))
                          for row in intervals for key in ("start_s", "end_s"))
        if not boundary_ok:
            evidence[name] = {"status": "unavailable", "reason": "Task boundary is outside tactile coverage or between native samples; fractional interval weighting not implemented"}
            trial.warnings.append(f"{name}: native tactile boundaries do not match task evidence; conditional metrics unavailable")
            continue
        masks.update(_scope_masks(trial.time_s, intervals))
        evidence[name] = {"status": "available", "reason": "Recorded task and tactile simulation clocks share native sample boundaries"}
    return masks, evidence


def _verify_provenance(rows: list[dict]) -> None:
    """A live writer must not silently change the observations being evaluated."""
    for row in rows:
        if not row.get("sha256") or not row.get("path"):
            continue
        path = Path(row["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Input changed during evaluation: {path}")


def evaluate_trial(path: str | Path, protocol: dict | None = None) -> dict:
    protocol = make_protocol(protocol)
    implementation = implementation_manifest()
    trial = load_trial(Path(path))
    task_report = build_task_report(trial, protocol)
    if trial.time_s is None or trial.normal_force_n is None:
        tactile = {"schema_version": "tactile-metrics-v1", "status": "missing",
                   "reason": "No recorded, interpretable normal-force series",
                   "scopes": {}}
    else:
        scopes, scope_evidence = _aligned_scopes(trial, task_report)
        tactile = analyze_tactile(
            trial.time_s, trial.normal_force_n, channel_names=trial.channel_names,
            normal_taxel_force_n=trial.normal_taxel_force_n,
            tangent_taxel_force_n=trial.tangent_taxel_force_n,
            taxel_positions_local_m=trial.taxel_positions_local_m,
            scope_masks=scopes, config={key: protocol[key] for key in TACTILE_CONFIG},
        )
        tactile["task_scope_evidence"] = scope_evidence
        for name, evidence in scope_evidence.items():
            if name not in tactile["scopes"]:
                tactile["scopes"][name] = {**evidence, "duration_s": None,
                                             "channels": None, "load_shares": None}
    _verify_provenance(trial.provenance)
    if implementation != implementation_manifest():
        raise ValueError("Evaluator source changed during evaluation; rerun on a stable version")
    summary = trial.summary
    cohort = {
        "task": trial.task,
        "checkpoint": summary.get("checkpoint_sha256") or summary.get("checkpoint_path"),
        "model_family": summary.get("model_family"),
        "controller": summary.get("controller"),
        "control_hz": summary.get("control_hz"),
        "execute_steps": summary.get("execute_steps"),
        "guard_mode": task_report["safety"]["enforcement"],
        "diagnostic_only": task_report["outcome"]["diagnostic_only"],
        "formal_metrics_valid": task_report["outcome"]["formal_metrics_valid"],
        "tactile_channels": trial.channel_names,
        "tactile_sample_interval_s": (round(float(np.median(np.diff(trial.time_s))), 6)
                                      if trial.time_s is not None and len(trial.time_s) > 1 else None),
    }
    source = str(trial.root.resolve())
    video_metadata = summary.get("video") or {}
    if not isinstance(video_metadata, dict):
        video_metadata = {}
    trial_id = f"{trial.task}:{trial.root.parent.name}:{trial.root.name}:{hashlib.sha256(source.encode()).hexdigest()[:8]}"
    report = {
        "schema_version": SCHEMA_VERSION,
        "trial": {"task": trial.task, "trial_id": trial_id, "source": source,
                  "seed": summary.get("seed"), "cohort": cohort},
        "protocol": protocol, "protocol_sha256": content_hash(protocol),
        "implementation": implementation,
        "provenance": trial.provenance,
        "warnings": list(trial.warnings),
        "evidence_quality": {
            "tactile_source": video_metadata.get("tactile_source"),
            "tactile_semantics": video_metadata.get("tactile_semantics"),
            "force_unit": "N as declared by archived force fields; not independently calibrated here",
            "taxel_geometry_status": "available" if trial.taxel_positions_local_m is not None else "missing",
            "calibration_verification": "not_performed",
            "task_tactile_clock_mapping": "recorded_common_simulation_clock" if trial.task_tactile_clock_aligned else "unverified",
            "sensor_saturation": {"status": "missing", "reason": "No independent saturation mask in legacy inputs"},
            "independent_slip": {"status": "missing", "reason": "No validated object-hand relative motion or slip labels"},
            "expected_contact_compliance": {"status": "missing", "reason": "Legacy monitor phases alone do not define independent expected-contact intervals"},
            "tactile_sent_to_model": (summary.get("observation_contract") or {}).get("tactile_sent_to_model"),
        },
        **task_report,
        "tactile": tactile,
        "limitations": [
            "Stage 1 reads archived observations; archived task success is not independently re-certified.",
            "Declared controller stages are not task ground truth; contact-conditioned scopes are explicitly conditional.",
            "Force maps may be solver kernel allocations, not calibrated flexible-skin pressure.",
            "Binned changes do not recover unsampled impacts or equalize physical sensor bandwidth.",
            "No expert temporal alignment, universal tactile score, or inferred slip truth is used.",
            "Expected-contact compliance requires independent phase evidence; observed contact is not that evidence.",
            "Real hardware calibration, validated slip detection, and tactile-input ablation are not performed.",
        ],
    }
    # Validate the actual export contract, including any nested plugin fields.
    content_hash(report)
    return report


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_leaves(value: object, prefix: str = "") -> dict[str, float]:
    """Collect per-episode scalars only, never pool source frames across episodes."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_numeric_leaves(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: float(value)} if _finite_number(value) else {}


def _distribution(values: list[float], total: int) -> dict:
    return {"available_episode_count": len(values), "total_episode_count": total,
            "median": float(np.median(values)), "p25": float(np.quantile(values, .25)),
            "p75": float(np.quantile(values, .75)),
            "min": min(values), "max": max(values)}


def _group(reports: list[dict], *, include_metrics: bool) -> dict:
    counts = Counter(report["outcome"]["status"] for report in reports)
    if set(counts) - {"pass", "fail", "unknown"}:
        raise ValueError("Unexpected task outcome status")
    total = len(reports)
    passed, failed, unknown = (counts[key] for key in ("pass", "fail", "unknown"))
    resolved = passed + failed
    result = {
        "trial_count": total,
        "outcome_counts": {"pass": passed, "fail": failed, "unknown": unknown},
        "confirmed_success_fraction": passed / total,
        "possible_success_fraction": (passed + unknown) / total,
        "resolved_success_rate": passed / resolved if resolved else None,
        "unknown_count": unknown,
        "interpretation": "Confirmed/possible fractions bound unknown outcomes; resolved-only rate excludes them explicitly.",
    }
    configuration_count = len({content_hash(report["trial"]["cohort"]) for report in reports})
    result["configuration_count"] = configuration_count
    if unknown == 0 and configuration_count == 1:
        z = 1.959963984540054
        rate = passed / total
        denominator = 1 + z * z / total
        center = (rate + z * z / (2 * total)) / denominator
        half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
        result["wilson95_interval"] = [max(0., center - half), min(1., center + half)]
        result["interval_assumption"] = "Independent Bernoulli trials under this cohort; not a guarantee under correlated scenes."
    else:
        result["wilson95_interval"] = None
        result["interval_assumption"] = "Not reported for unknown outcomes or mixed configurations."
    if include_metrics:
        scalars = defaultdict(list)
        for report in reports:
            for key, value in _numeric_leaves({"efficiency": report.get("efficiency", {}),
                                              "tactile": report.get("tactile", {}).get("scopes", {})}).items():
                scalars[key].append(value)
        result["metrics"] = {key: _distribution(values, total) for key, values in sorted(scalars.items())}
    return result


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("At least one trial is required")
    protocols = {report["protocol_sha256"] for report in reports}
    implementations = {report["implementation"]["sha256"] for report in reports}
    if len(protocols) != 1 or len(implementations) != 1:
        raise ValueError("Cannot aggregate different evaluator protocols or implementations")
    sources = [report["trial"]["source"] for report in reports]
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate trial sources would inflate the sample count")
    by_task, cohorts = defaultdict(list), defaultdict(list)
    for report in reports:
        by_task[report["trial"]["task"]].append(report)
        cohorts[content_hash(report["trial"]["cohort"])].append(report)
    task_results = {task: _group(group, include_metrics=False) for task, group in sorted(by_task.items())}
    cohort_results = {}
    for key, group in sorted(cohorts.items()):
        cohort_results[key] = {"configuration": group[0]["trial"]["cohort"],
                               **_group(group, include_metrics=True)}
    return {
        "schema_version": "task-tactile-batch-v1",
        "protocol": reports[0]["protocol"],
        "protocol_sha256": reports[0]["protocol_sha256"],
        "implementation": reports[0]["implementation"],
        "trial_count": len(reports), "by_task": task_results, "by_cohort": cohort_results,
        "macro_confirmed_success_fraction": float(np.mean([group["confirmed_success_fraction"] for group in task_results.values()])),
        "aggregation_semantics": "Equal-task descriptive macro average of supplied trials, not a cross-policy ranking; per-cohort distributions use one scalar per episode.",
        "trials": [{"trial_id": report["trial"]["trial_id"], "task": report["trial"]["task"],
                    "source": report["trial"]["source"], "outcome": report["outcome"],
                    "report_path": report.get("report_path")} for report in reports],
    }
