"""Source-bound display annotations for an already published rich workbench.

This adapter never imports archived evaluators, loads weights, or changes a
score.  Mechanical references use the existing reviewed, static AST reader.
Recorded samples stay byte-for-byte equivalent as JSON values; bulb terminal
angles come only from the archived report's terminal state, never its curves.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
from pathlib import Path

from evaluation_v1.protocol import content_hash
from evaluation_workbench import goal_references as goals

from .unified_workbench_export import _family

SCHEMA = "kaihand-rich-reference-display-v1"
_ANGLES = {
    "bulb-screw": ("clockwise_turns", "clockwise_angle_deg", "终态灯泡净旋入角度（顺时针）"),
    "bulb-unscrew": ("turns_removed", "angle_removed_deg", "终态灯泡净旋出角度（逆时针）"),
    "bulb-screw-inplace": ("rotation_from_initial_turns", "rotation_from_initial_deg", "终态相对实际初态的净旋入角度"),
}
_ANGLE_KEYS = {value[0]: value[1] for value in _ANGLES.values()}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bound_document(row, root, role, provenance):
    path = Path(row.get("path", ""))
    if not path.is_absolute() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Bound evidence is outside its selected publication/evaluation root")
    raw, actual = goals._read_bytes(path, row)
    provenance.append({**actual, "role": role})
    return goals._json(raw), actual


def _equal_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _summary(report, group, trial, reports_root, evaluation_root, provenance):
    binding = report.get("source_binding")
    if not isinstance(binding, dict):
        raise ValueError("Missing published/producer summary binding")
    published = binding.get("published_summary", {})
    producer = binding.get("producer_summary", {})
    if published.get("sha256") != trial.get("summary_sha256"):
        raise ValueError("Summary SHA does not match the published member")
    authority, _ = _bound_document(published, reports_root, "goal_reference_published_summary", provenance)
    observed, producer_row = _bound_document(producer, evaluation_root, "goal_reference_trial_summary", provenance)
    if not _equal_json(authority, observed):
        raise ValueError("Published and producer summary JSON objects differ")
    expected = {"task": group["task"], "seed": trial["seed"],
                "prediction_horizon": group["prediction_horizon"], "execute_steps": group["execute_steps"],
                "checkpoint_sha256": group["checkpoint_sha256"]}
    for key, value in expected.items():
        if observed.get(key) != value or type(observed.get(key)) is not type(value):
            raise ValueError("Summary/catalog reference identity mismatch: " + key)
    if _family(observed.get("model_family")) != group["model"]:
        raise ValueError("Summary/catalog reference identity mismatch: model_family")
    return observed, producer_row


def _manifest(summary, evaluation_root, provenance):
    path = summary.get("deployment_manifest")
    integrity = summary.get("deployment_integrity") or {}
    entry = integrity.get("manifest") or {}
    expected = summary.get("deployment_manifest_sha256") or entry.get("expected_sha256") or integrity.get("manifest_sha256")
    if not goals._sha(expected) or not isinstance(path, str):
        raise ValueError("Summary has no explicit SHA-bound deployment manifest")
    # Some tactile manifests bind SHA but do not separately repeat byte size.
    # The expected hash still binds every byte; stat is not a substitute hash.
    size = entry.get("size")
    if size is None:
        size = Path(path).stat().st_size
    value, row = _bound_document({"path": path, "sha256": expected, "size": size},
                                evaluation_root, "goal_reference_deployment_manifest", provenance)
    return value, row


def _tactile_references(summary, manifest, manifest_row, group, provenance):
    """Verify the tactile schema honestly instead of inventing legacy integrity."""
    if manifest.get("schema") != "poker_tactile_deployment_v1" or group["task"] != "poker-draw":
        raise ValueError("Unsupported tactile deployment reference schema/task")
    identity = {key: value for key, value in manifest.items() if key != "deployment_id"}
    if hashlib.sha256(goals._canonical(identity)).hexdigest() != manifest.get("deployment_id"):
        raise ValueError("Tactile deployment identity digest mismatch")
    for key in ("task", "model_family", "checkpoint_sha256", "prediction_horizon", "execute_steps", "deployment_id"):
        if summary.get(key) != manifest.get(key) or type(summary.get(key)) is not type(manifest.get(key)):
            raise ValueError("Tactile summary/deployment identity mismatch: " + key)
    if summary.get("deployment_manifest_sha256") != manifest_row["sha256"]:
        raise ValueError("Tactile summary does not bind the deployment bytes")
    inventory = manifest.get("files")
    if not isinstance(inventory, dict):
        raise ValueError("Tactile deployment lacks its source file inventory")
    trees = {}
    for name in goals._TASK_FILES[group["task"]]:
        relative = goals._source_relative(name)
        matches = [(path, digest) for path, digest in inventory.items()
                   if isinstance(path, str) and path.endswith("/" + relative)]
        if len(matches) != 1:
            raise ValueError("Tactile reviewed source binding is absent or ambiguous: " + relative)
        path, expected = matches[0]
        if not Path(path).is_absolute() or expected not in {goals._REVIEWED[name], *goals._REVIEWED_ALTERNATES.get(name, ())}:
            raise ValueError("Tactile mechanics source version has not been reviewed: " + relative)
        raw, row = goals._read_bytes(path, {"sha256": expected, "size": Path(path).stat().st_size})
        provenance.append({**row, "role": "goal_reference_reviewed_source", "inventory_path": relative})
        trees[name] = ast.parse(raw, filename=path)
    return {"schema_version": goals.SCHEMA_VERSION, "task": group["task"], "status": "verified",
            "references": goals._references(group["task"], trees), "warnings": [goals._DISCLAIMER],
            "provenance": copy.deepcopy(provenance),
            "binding": {"kind": "tactile_summary_manifest_reviewed_measurement_bytes",
                        "deployment_manifest_sha256": manifest_row["sha256"],
                        "prediction_horizon": manifest["prediction_horizon"],
                        "execute_steps": manifest["execute_steps"],
                        "scope": "Mechanical chart references only; no model-weight or stage recertification",
                        "source_code_executed": False, "rules_reexecuted": False}}


def _recover_goals(report, group, trial, reports_root, evaluation_root, provenance):
    result = {"schema_version": goals.SCHEMA_VERSION, "task": group["task"],
              "status": "unavailable", "references": [], "warnings": [], "provenance": []}
    if group["task"] not in goals._TASKS:
        return {**result, "status": "not_applicable", "reason": "本任务未归档可绑定机械目标的物体指标，不从动作或触觉推断目标。"}
    try:
        summary, summary_row = _summary(report, group, trial, reports_root, evaluation_root, provenance)
        manifest, manifest_row = _manifest(summary, evaluation_root, provenance)
        if manifest.get("schema") == "poker_tactile_deployment_v1":
            result = _tactile_references(summary, manifest, manifest_row, group, provenance)
        else:
            # The static reader already checks summary/source identity, reviewed
            # bytes and safe arithmetic. Only evidence-role names are adapted.
            bound_report = copy.deepcopy(report)
            bound_report["provenance"] = [
                {**summary_row, "role": "prediction_horizon_summary"},
                {**manifest_row, "role": "deployment_manifest"},
            ]
            declared_root = (summary.get("deployment_integrity", {}).get("evaluation_source") or {}).get("root")
            # USB's archived inventory omits its root, but includes every source
            # SHA and size. Supply this package's explicit byte-copy location;
            # the reader still requires each exact inventoried/reviewed byte.
            source_copy = None if declared_root else Path(goals.__file__).resolve().parents[3]
            result = goals.build_goal_references(group["task"], bound_report, source_root=source_copy)
            provenance.extend(result.get("provenance", []))
        if result["status"] != "verified":
            result["reason"] = "该条归档的机械参考来源未能完成校验；保留实测曲线，不套用其他版本的目标值。"
        if trial.get("staged_evidence_status") == "invalid":
            result.setdefault("warnings", []).append("机械参考校验不恢复阶段成绩：阶段已记录，但来源校验未通过，仍不计分。")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        result.update(status="unavailable", references=[],
                      reason="该条归档的机械参考来源未能完成校验；保留实测曲线，不套用其他版本的目标值。",
                      warnings=["未显示机械目标参考：" + str(error), goals._DISCLAIMER],
                      provenance=copy.deepcopy(provenance))
    return result


def _for_series(references, series):
    result = []
    for reference in references:
        source, target = (reference.get("key"), reference.get("unit")), (series.get("key"), series.get("unit"))
        factor = 1.0 if source == target else None
        for turns, degrees in _ANGLE_KEYS.items():
            if source == (degrees, "deg") and target == (turns, "turn"):
                factor = 1 / 360.0
            elif source == (turns, "turn") and target == (degrees, "deg"):
                factor = 360.0
        if factor is None:
            continue
        item = copy.deepcopy(reference)
        if factor != 1:
            item.update(key=target[0], unit=target[1])
            for field in ("target", "lower", "upper"):
                if field in item:
                    item[field] *= factor
        result.append(item)
    return result


def _terminal_angle(report, config, task):
    if task not in _ANGLES:
        return None
    source, key, label = _ANGLES[task]
    state = report.get("task_specific", {}).get("partial_state", {})
    value = state.get(source)
    degree = value * 360.0 if _finite(value) and _finite(value * 360.0) else None
    rows = config.setdefault("task_summary", [])
    matches = [row for row in rows if row.get("key") == key]
    if len(matches) > 1:
        raise ValueError("Duplicate terminal bulb angle fields")
    if matches:
        row = matches[0]
        if row.get("unit") != "deg" or row.get("value") != degree:
            raise ValueError("Published terminal angle contradicts archived final state")
        added = False
    else:
        row = {"key": key, "label": label, "unit": "deg", "value": degree,
               "status": "recorded" if degree is not None else "unknown",
               "source": "report.json:task_specific.partial_state." + source,
               "semantics": "原监视器终态净角度；反向回拧抵消，不是累计旋转路程，不取曲线末样本，不重判任务成功。",
               "conversion": {"operation": "multiply", "factor": 360.0, "source_unit": "turn", "output_unit": "deg"}}
        if degree is None:
            row["reason"] = "归档终态未记录有限净圈数，终态角度保持未知；不以曲线末帧替代。"
        rows.insert(0, row)
        added = True
    return {"key": key, "source_key": source, "value": degree,
            "status": row.get("status"), "added": added,
            "basis": "archived_report_terminal_partial_state", "curve_endpoint_used": False}


def enrich_case(report, visualization, config, group, trial, *, reports_root, evaluation_root):
    """Return caller-owned metadata enrichments, with native evidence untouched."""
    report, visual, config = copy.deepcopy(report), copy.deepcopy(visualization), copy.deepcopy(config)
    if report.get("trial", {}).get("task") != group["task"] or config.get("task") != group["task"]:
        raise ValueError("Reference presentation belongs to another task")
    if config.get("unified_identity", {}).get("trial_id") not in (None, trial["id"]):
        raise ValueError("Reference presentation belongs to another unified trial")
    provenance = []
    existing = config.get("goal_references")
    preserved = isinstance(existing, dict) and existing.get("status") == "verified"
    if preserved:
        references = existing
    else:
        references = _recover_goals(report, group, trial, reports_root, evaluation_root, provenance)
        config["goal_references"] = references
    attached = []
    if references.get("status") == "verified":
        for series in visual.get("task_series", []):
            matching = _for_series(references.get("references", []), series)
            if matching:
                if series.get("goal_references") not in (None, matching):
                    raise ValueError("Existing chart reference contradicts its archived reference bundle")
                series["goal_references"] = matching
                attached.append(series["key"])
    terminal = _terminal_angle(report, config, group["task"])
    coverage = {"trial_id": trial["id"], "task": group["task"], "model": group["model"],
                "checkpoint": group["checkpoint"], "reference_status": references["status"],
                "reference_basis": "preserved_published" if preserved else "recovered_bound_sources",
                "reference_count": len(references.get("references", [])), "series_with_reference": attached,
                "reason": references.get("reason"), "terminal_angle": terminal}
    report["goal_reference_extension"] = {"schema_version": SCHEMA, "scope": "Display-only annotation; native/staged admissions unchanged",
                                           "reference_status": references["status"], "terminal_angle": terminal,
                                           "source_code_executed": False, "rules_reexecuted": False}
    report.setdefault("visualization", {})["sha256"] = content_hash(visual)
    return {"report": report, "visualization": visual, "config": config,
            "provenance": provenance, "coverage": coverage}
