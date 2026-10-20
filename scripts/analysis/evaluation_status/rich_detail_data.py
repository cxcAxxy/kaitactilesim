"""Restore rich displays from recorded artifacts without rerunning a policy.

Published admissions, native outcomes, and stage labels remain authoritative.
This adapter supports H30/H32/H50 explicitly; it never passes non-H50 inputs
through the historical workbench's H50 admission or rescoring entry points.
"""

from __future__ import annotations

import copy
import math
import re
import zipfile
from pathlib import Path

import numpy as np
from evaluation_tactile_load_extension.data import _LABELS, _member, build_load_series
from evaluation_unified.report import projection
from evaluation_v1.adapters import TrialData
from evaluation_v1.protocol import content_hash, make_protocol
from evaluation_v1.tactile import DEFAULT_CONFIG, analyze_tactile
from evaluation_v1.visualization import build_visualization
from evaluation_workbench.data import (
    MAX_TRACE_BYTES,
    _frame_trial,
    _json_bound,
    _jsonl,
    _layout_file,
    _load_native,
    _media,
)
from evaluation_workbench.diagnostic_plots import _PLOTS, _png
from evaluation_workbench.experts import model_views_for_case
from evaluation_workbench.export import _rows, _verify
from evaluation_workbench.task_fields import (
    augment_task_visualization,
    task_presentation,
)
from evaluation_workbench.whiteboard_contact_area import (
    _MEAN_NOTE,
    _NOTE,
    _fraction,
    _metric_snapshot,
    _row_snapshot,
    _series,
    _summary,
)

from .unified_workbench_export import _Artifacts, _family, _read, _sha

SCHEMA = "kaihand-published-rich-detail-v1"
_ACTIVE = {"running", "starting", "queued", "pending", "in_progress"}
_SLUG = re.compile(r"[A-Za-z0-9_-]+\Z")


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _safe_source(path, root):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Rich source must be an explicit non-leaf-symlink absolute directory")
    actual, root = path.resolve(strict=True), Path(root).resolve(strict=True)
    if not actual.is_relative_to(root) or not actual.is_dir():
        raise ValueError("Rich source escapes the selected evaluation root")
    return actual


def resolve_rich_source(catalog_root, group, trial, *, evaluation_root):
    """Resolve only a published member's exact accepted source, never discover jobs.

    Existing rich/quarantine pages are returned as such. Build bindings verify
    the original catalog's copied summary bytes and carry that authority into
    ``build_rich_case``. The producer may differ only in JSON serialization.
    """
    catalog_root = Path(catalog_root).resolve(strict=True)
    store = _Artifacts(catalog_root.parent, catalog_root)
    detail = store.target(catalog_root, trial.get("detail_url"))
    if trial.get("publication_status") == "quarantined_replay_only":
        return {"kind": "quarantine", "trial_id": trial["id"], "published_detail": str(detail),
                "provenance": list(store.inputs.values())}
    if detail.name == "report.html":
        return {"kind": "existing_rich", "trial_id": trial["id"], "published_detail": str(detail),
                "provenance": list(store.inputs.values())}
    published_trial = store.document(detail.parent / "trial.json", "published_member_admission")
    if published_trial.get("id") != trial["id"] or published_trial.get("seed") != trial["seed"]:
        raise ValueError("Published detail belongs to another trial")
    summary = store.artifact(detail.parent / "summary.json", "published_summary_authority")
    if store.inputs[str(summary)]["sha256"] != trial.get("summary_sha256"):
        raise ValueError("Published summary bytes disagree with unified member identity")
    if group.get("source_snapshot") == "tactile":
        relative = detail.relative_to(catalog_root.parent)
        snapshot = catalog_root.parent / relative.parts[0]
        catalog = store.document(snapshot / "catalog.json", "published_tactile_catalog")
        verification = store.document(snapshot / "verification.json", "published_tactile_source_binding")
        matches = [row for row in catalog.get("groups", []) if row.get("label") == group.get("source_group_id")]
        if len(matches) != 1:
            raise ValueError("Tactile member has no unique published deployment group")
        source_group = matches[0]
        label, attempt = source_group.get("label"), source_group.get("accepted_attempt")
        if not all(isinstance(value, str) and _SLUG.fullmatch(value) for value in (label, attempt)):
            raise ValueError("Tactile accepted-attempt binding is missing or unsafe")
        if not any(row == published_trial for row in source_group.get("trials", [])):
            raise ValueError("Tactile detail differs from its published group admission")
        source = Path(verification["job_root"]) / "groups" / label / attempt / "evaluation" / f"seed_{trial['seed']:04d}"
    else:
        if published_trial.get("summary_sha256") != trial.get("summary_sha256"):
            raise ValueError("Native detail and unified summary identities disagree")
        source = Path(published_trial.get("source", ""))
        if Path(published_trial.get("summary_path", "")) != source / "summary.json":
            raise ValueError("Published native summary/source binding is inconsistent")
        if published_trial.get("valid") is not trial.get("valid") or published_trial.get("native_success") is not trial.get("native_success"):
            raise ValueError("Native detail admission differs from unified catalog")
    source = _safe_source(source, evaluation_root)
    return {"kind": "build", "trial_id": trial["id"], "source_dir": str(source),
            "published_detail": str(detail), "published_summary_path": str(summary),
            "expected_summary_sha256": trial["summary_sha256"], "provenance": list(store.inputs.values())}


def _identity(summary, group, trial):
    expected = {"task": group["task"], "seed": trial["seed"],
                "prediction_horizon": group["prediction_horizon"], "execute_steps": group["execute_steps"],
                "checkpoint_sha256": group["checkpoint_sha256"]}
    for key, value in expected.items():
        observed = summary.get(key)
        if observed != value or type(observed) is not type(value):
            raise ValueError(f"Rich source identity mismatch: {key}")
    if _family(summary.get("model_family")) != group["model"]:
        raise ValueError("Rich source model family mismatch")
    if not _sha(expected["checkpoint_sha256"]) or not _sha(trial.get("summary_sha256")):
        raise ValueError("Rich member lacks published SHA-256 identities")
    if summary.get("status") in _ACTIVE or summary.get("completed") is False:
        raise ValueError("Rich source is not terminal")
    if trial.get("publication_status") == "quarantined_replay_only":
        raise ValueError("Quarantined historical members retain their original replay page")
    recorded = summary.get("task_success")
    if type(recorded) is not bool:
        recorded = (summary.get("evaluation") or {}).get("success")
    if trial.get("valid") is True and recorded is not trial.get("native_success"):
        raise ValueError("Producer native outcome differs from published admission")


def _summary_binding(source, published, expected, evidence):
    source_value = _json_bound(source, evidence, "producer_summary_observations")
    source_row = evidence.provenance[-1]
    if not _sha(expected):
        raise ValueError("An exact published summary SHA-256 is required")
    if published is None:
        if source_row["sha256"] != expected:
            raise ValueError("Producer summary SHA differs from published authority")
        authority, authority_row = source_value, source_row
    else:
        authority, authority_row = _read(published)
        if authority_row["sha256"] != expected:
            raise ValueError("Published summary SHA differs from catalog authority")
        authority_row["role"] = "published_summary_authority"
        evidence.provenance = _rows(evidence.provenance + [authority_row])
        if not _same_json(authority, source_value):
            raise ValueError("Producer and published summary JSON objects differ")
    basis = "exact_bytes" if source_row["sha256"] == authority_row["sha256"] else "json_object_equal"
    return authority, {"binding_basis": basis, "producer_summary_path": str(source),
                       "published_summary_path": authority_row["path"],
                       "producer_summary": copy.deepcopy(source_row), "published_summary": copy.deepcopy(authority_row),
                       "semantics": "Only byte-equivalent or strictly equal parsed JSON objects are admitted; no field mutation is permitted."}


def _report(evidence, group, trial):
    summary = evidence.summary
    native = trial.get("native_success") if trial.get("valid") is True else None
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    state = evaluation.get("state") if isinstance(evaluation.get("state"), dict) else {}
    protocol = make_protocol()
    outcome = {"status": "pass" if native is True else "fail" if native is False else "unknown",
               "recorded_success": native, "task_success": native,
               "reported_status": trial.get("status"), "source": "published catalog admission",
               "diagnostic_only": group.get("mode") == "diagnostic", "formal_metrics_valid": group.get("mode") == "formal",
               "termination_reason": summary.get("stop_reason", summary.get("status")),
               "reason": "Previously published native outcome, not rescored by rich display export"}
    tactile = {"schema_version": "tactile-metrics-v1", "status": "missing", "scopes": {},
               "reason": "No recorded interpretable normal-force trace"}
    if evidence.time_s is not None and evidence.normal_force_n is not None:
        tactile = analyze_tactile(evidence.time_s, evidence.normal_force_n, channel_names=evidence.channel_names,
                                  normal_taxel_force_n=evidence.normal_taxel_force_n,
                                  tangent_taxel_force_n=evidence.tangent_taxel_force_n,
                                  config={key: protocol[key] for key in DEFAULT_CONFIG})
        tactile["semantics"] = "Descriptive recorded-force analysis only; no native/staged success re-evaluation"
    cohort = {"task": group["task"], "model_family": group["model"], "checkpoint": group["checkpoint_sha256"],
              "checkpoint_label": group["checkpoint"], "prediction_horizon": group["prediction_horizon"],
              "execute_steps": group["execute_steps"], **copy.deepcopy(group.get("protocol", {}))}
    return {"schema_version": "task-tactile-evaluation-v1", "suite_schema_version": "task-tactile-suite-v2",
            "trial": {"task": group["task"], "trial_id": trial["id"], "source": str(evidence.root),
                      "seed": trial["seed"], "prediction_horizon": group["prediction_horizon"], "cohort": cohort},
            "outcome": outcome, "protocol": protocol, "protocol_sha256": content_hash(protocol),
            "implementation": {"schema": SCHEMA, "semantics": "Offline display adapter; published outcomes unchanged"},
            "provenance": evidence.provenance, "warnings": evidence.warnings,
            "safety": {"status": "unknown", "enforcement": group.get("protocol", {}).get("guard_mode", "unknown"),
                       "checks": [], "recorded_guard_events": copy.deepcopy(summary.get("guard_events", [])),
                       "limitations": ["Only recorded guard events are shown; safety is not re-certified"]},
            "efficiency": {"simulation_duration_s": _number(summary.get("sim_seconds")),
                           "wall_duration_s": _number(summary.get("wall_seconds")),
                           "recorded_time_budget_s": _number(summary.get("max_sim_seconds"))},
            "task_specific": {"plugin": "published_recorded_outcome", "predicates": [], "partial_state": copy.deepcopy(state),
                              **({"cleaned_fraction": state["cleaned_fraction"]} if "cleaned_fraction" in state else {})},
            "events": [], "scope_intervals": [], "scope_availability": [], "tactile": tactile,
            "task_review": {"review": {"status": "unknown", "availability": "not_verified", "basis": "published_record"},
                            "original_outcome_preserved": True},
            "evidence_v2": {"status": "not_verified", "source": "recorded artifacts; display-only recovery"},
            "contact_quality": {"status": "unknown"}, "reference_comparison": {"status": "not_configured"},
            "evidence_quality": {"task_tactile_clock_mapping": "recorded_common_simulation_clock" if evidence.task_tactile_clock_aligned else "unverified",
                                 "force_unit": "recorded N; no renewed calibration", "tactile_sent_to_model": summary.get("tactile_sent_to_model")},
            "limitations": ["Published native and staged admissions are not recomputed.",
                            "Missing measurements, commands, expert references, and unknown outcomes remain missing.",
                            "No simulator, model, checkpoint weight, or original evaluator is executed."]}


def _plots(evidence):
    """Copy only explicit same-trial existing diagnostics; never create an expert."""
    video = evidence.summary.get("video") or {}
    comparison = video.get("comparison_plots") if isinstance(video, dict) else None
    result = {"status": "missing", "plots": [], "reason": "No complete, declared same-trial PNG diagnostic bundle",
              "comparison_kind": "model_only", "source_label": f"{evidence.task} · recorded model trace"}
    if not isinstance(comparison, dict) or not isinstance(comparison.get("artifacts"), dict):
        return result, []
    plots, copies = [], []
    reference = comparison.get("reference")
    reference_present = (isinstance(reference, dict) and isinstance(reference.get("dataset_root"), str)
                         and isinstance(reference.get("source_hdf5"), str)
                         and _number(reference.get("samples")) is not None and reference["samples"] > 0)
    for identifier, artifact, label, _, description in _PLOTS:
        declared = comparison["artifacts"].get(artifact)
        if not isinstance(declared, str) or not Path(declared).is_absolute() or Path(declared).is_symlink():
            return result, []
        path = Path(declared).resolve(strict=False)
        if path.parent not in (evidence.root, evidence.root / "review"):
            raise ValueError("Declared diagnostic image escapes its trial")
        if not path.is_file():
            return result, []
        dimensions, row = _png(path)
        evidence.provenance = _rows(evidence.provenance + [row])
        url = f"plots/{path.name}"
        plots.append({"id": identifier, "label": label, "description": description, "url": url,
                      "sha256": row["sha256"], "width": dimensions[0], "height": dimensions[1]})
        copies.append({"source": path, "url": url, "sha256": row["sha256"], "size": row["size"]})
    result.update(status="available", plots=plots,
                  comparison_kind="recorded_reference" if reference_present else "model_only",
                  comparison_note="原记录诊断图；每条曲线保留自身时间轴，不做专家/模型阶段对齐。",
                  reference_label="原图已记录参考曲线；未生成或替代专家", rollout_label="原图模型记录",
                  reference=copy.deepcopy(comparison.get("reference")), provenance=evidence.provenance)
    return result, copies


def _staged(evidence, group, member):
    stage_valid = member.get("valid") is True and member.get("staged_evidence_status") == "valid"
    path = evidence.root / "staged_evaluation.json"
    raw = _json_bound(path, evidence, "recorded_stage_observations") if path.is_file() else {}
    payload = raw.get("staged_result", raw)
    if not isinstance(payload, dict):
        raise ValueError("Recorded stage payload must be an object")
    admitted = {row["key"]: row.get("reached") if stage_valid else None for row in member.get("stages", [])}
    definitions = {row["key"]: row["name"] for row in member.get("stages", [])}
    if stage_valid and (payload.get("stage_flags") != admitted or payload.get("stage_definitions") != definitions):
        raise ValueError("Recorded stage flags/definitions differ from published admission")
    # The copied published summary already freezes the complete stage payload
    # for current native runners. Authenticate first-times and thresholds there,
    # rather than attributing new live-file values to an archived rule identity.
    authority = evidence.summary.get("staged_evaluation")
    if isinstance(authority, dict):
        authority = authority.get("staged_result", authority)
    stage_payload_bound = stage_valid and isinstance(authority, dict)
    if stage_payload_bound and not _same_json(payload, authority):
        raise ValueError("Recorded stage payload differs from published summary authority")
    if not definitions and not raw:
        return None
    if not definitions and raw:
        definitions = copy.deepcopy(payload.get("stage_definitions", {}))
        admitted = dict.fromkeys(definitions)
    # Unverified CPM payloads may be inspected in audit only. Never allow their
    # raw flags/first-times/complete-success to populate scored stage badges.
    displayed = {"version": payload.get("version", "staged_v1"), "stage_definitions": definitions,
                 "stage_flags": admitted, "complete_success": member.get("complete_stage_success") if stage_valid else None,
                 "first_times": copy.deepcopy(payload.get("first_times", {})) if stage_payload_bound else dict.fromkeys(definitions),
                 "thresholds": copy.deepcopy(payload.get("thresholds", {})) if stage_payload_bound else {},
                 "stage_achievement_policy": payload.get("stage_achievement_policy"), "time_basis": payload.get("time_basis")}
    states = evidence.root / "staged_states.jsonl"
    rows = _jsonl(states, evidence, "recorded_stage_state_observations") if states.is_file() else []
    origin = float(evidence.time_s[0]) if evidence.time_s is not None and len(evidence.time_s) else 0.0
    result = {"staged_result": displayed, "native_result": {"task_success": member.get("native_success"), "valid_trial": member.get("valid")},
              "rule_identity": copy.deepcopy(group.get("rule_identity")),
              "data_evidence": {"basis": member.get("staged_evidence_basis"),
                                "stage_payload_binding": "published_summary_embedded_payload" if stage_payload_bound else "recorded_observations_only",
                                "rules_reexecuted": False},
              "warnings": list(member.get("notes", [])), "state_rows": rows}
    if not stage_valid:
        result["warnings"].append("阶段已记录，但已发布来源校验未通过或未定义，暂不计分；原始标志仅保留在审计字段。")
    display = projection(result, origin)
    display["timeline"]["model_clock_mapping"] = "mapped" if evidence.task_tactile_clock_aligned else "unmapped"
    display.update(evidence_status=member.get("staged_evidence_status"),
                   evidence_basis=member.get("staged_evidence_basis"),
                   recorded_raw_stage_flags=copy.deepcopy(payload.get("stage_flags")),
                   recorded_raw_thresholds=copy.deepcopy(payload.get("thresholds")),
                   provenance=evidence.provenance)
    if evidence.task.startswith("whiteboard-wipe"):
        display["whiteboard_contact_area"] = _whiteboard_area(evidence, payload, rows, origin)
    return display


def _whiteboard_area(evidence, final_payload, rows, origin):
    """Copy already-integrated geometry values; do not reintegrate display rows."""
    result = {"schema_version": "whiteboard-contact-area-v1", "task": evidence.task, "status": "unavailable",
              "series": [], "summary": [], "provenance": evidence.provenance, "warnings": [_NOTE, _MEAN_NOTE]}
    try:
        final = _metric_snapshot(final_payload.get("continuous_metrics"))
        if final is None or not rows:
            return result
        full = final["full_pad_bottom_area_m2"]
        metrics = [_metric_snapshot(row.get("continuous_metrics"), denominator=full) for row in rows]
        snapshots = [_row_snapshot(row) for row in metrics]
        times = [row.get("simulation_time_s", row.get("time_s")) - origin for row in rows]
        if any(not math.isfinite(time) for time in times) or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Recorded whiteboard geometry clock is invalid")
        nominal = float(np.median(np.diff(times))) if len(times) > 1 else None
        gaps = [bool(i + 1 < len(times) and nominal and times[i + 1] - time > nominal * 1.5 + 1e-8)
                for i, time in enumerate(times)]
        source = str(evidence.root / "staged_states.jsonl") + ":continuous_metrics."
        result["series"] = [
            _series("effective_pad_contact_area_ratio", "接触时刻贴合面积比（刚性几何估计）", times,
                    [row[0] for row in snapshots], gaps, nominal, source + "effective_pad_contact_area_ratio", _NOTE),
            _series("contact_time_weighted_mean_effective_area_ratio", "接触期间累计平均贴合面积比（刚性几何估计）", times,
                    [row[1] for row in snapshots], gaps, nominal, source + "contact_time_weighted_mean_effective_area_ratio", _MEAN_NOTE),
        ]
        residuals = [((row.get("measurements") or {}).get("native_task_state") or {}).get("maximum_ink_remaining")
                     for row in rows]
        if any(value is not None for value in residuals):
            result["series"].append(_series(
                "maximum_ink_remaining", "最大单元墨迹余量（原监视器记录）", times,
                [_fraction(value) for value in residuals], gaps, nominal,
                str(evidence.root / "staged_states.jsonl") + ":measurements.native_task_state.maximum_ink_remaining",
                "直接展示原监视器记录，不由平均清洁度或阶段状态推断。"))
        duration = _number(final.get("pad_board_contact_duration_s"))
        duration = duration if duration is not None and duration >= 0 else None
        _, mean = _row_snapshot(final)
        result["summary"] = [
            _summary("contact_time_weighted_mean_effective_area_ratio", "终态接触期间平均贴合面积比（几何估计）", "fraction", mean,
                     source + "contact_time_weighted_mean_effective_area_ratio", _MEAN_NOTE,
                     status="no_contact" if duration == 0 else None),
            _summary("pad_board_contact_duration_s", "累计板面承载接触时长", "s", duration,
                     source + "pad_board_contact_duration_s", "原物理积分记录值，不由展示样本重新计算。"),
        ]
        result.update(status="recorded", binding={"basis": "published_archive_plus_SHA_bound_recorded_samples",
                                                   "native_values_recomputed": False, "rules_reexecuted": False,
                                                   "geometry_source_recertified": False, "denominator_m2": full})
    except (ValueError, TypeError, KeyError) as error:
        result.update(status="unavailable", series=[], summary=[], warnings=[_NOTE, str(error)])
    return result


def _state_trace(evidence, trace):
    result = {"status": "missing", "time_s": [], "wrist_state": [], "hand_joint_position": [],
              "reason": "No explicit recorded wrist/hand state arrays; measured state never substitutes for commands"}
    if trace is None:
        return result
    with zipfile.ZipFile(trace) as archive:
        time = _member(archive, "simulation_time_s")
        wrist = _member(archive, "wrist_state", shape_tail=(9,))
        hand = _member(archive, "hand_joint_position", shape_tail=(20,))
    if time is None or wrist is None or hand is None:
        return result
    if len(time) != len(wrist) or len(time) != len(hand):
        raise ValueError("Recorded state array lengths disagree")
    def finite_rows(values):
        return [[float(value) if np.isfinite(value) else None for value in row] for row in values]
    result.update(status="available", time_s=(time - time[0]).tolist(), wrist_state=finite_rows(wrist),
                  hand_joint_position=finite_rows(hand), source=str(trace),
                  semantics="Actual recorded right-wrist xyz/rot6d and twenty hand joints, not predicted/applied action targets")
    return result


def build_rich_case(source_dir, group, trial, *, published_summary_path=None,
                    expected_summary_sha256=None, suite_url="../index.html"):
    """Return a complete workbench-compatible case plus copy jobs; write nothing."""
    source = Path(source_dir)
    if not source.is_absolute() or source.is_symlink():
        raise ValueError("Select an explicit, non-leaf-symlink source directory")
    source = source.resolve(strict=True)
    evidence = TrialData(root=source, summary={}, task=group["task"])
    expected = expected_summary_sha256 or trial.get("summary_sha256")
    if expected != trial.get("summary_sha256"):
        raise ValueError("Explicit published summary SHA differs from catalog member")
    summary, binding = _summary_binding(source / "summary.json", published_summary_path, expected, evidence)
    _identity(summary, group, trial)
    evidence.summary = summary
    trace, tangent = _load_native(evidence)
    if trace is not None and trace.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("Native trace exceeds bounded source size")
    controls = source / "control_steps.jsonl"
    if controls.is_file():
        evidence.control_rows = _jsonl(controls, evidence, "recorded_control_step_observations")
    frames_path = _layout_file(source, "frames.jsonl")
    frames = _jsonl(frames_path, evidence, "recorded_review_frames", frames=True,
                    keep_taxels=evidence.normal_taxel_force_n is None) if frames_path else []
    if not evidence.control_rows and frames:
        evidence.control_rows = [{"sim_time": row.get("simulation_time_s"), "task": row.get("task_metrics", {}),
                                  "stage_after_action": row.get("phase"), "_source": row.get("_source"),
                                  "_sampling": "recorded_review_frame_cadence"} for row in frames]
    if evidence.time_s is None:
        fallback = _frame_trial(evidence, frames)
        if fallback is not None:
            for key in ("time_s", "normal_force_n", "normal_taxel_force_n", "tangent_taxel_force_n", "channel_names", "task_tactile_clock_aligned"):
                setattr(evidence, key, getattr(fallback, key))
    report = _report(evidence, group, trial)
    visual = build_visualization(evidence, report)
    presentation = augment_task_visualization(evidence, report, visual, frame_rows=frames)
    review = _layout_file(source, "review.json")
    manifest = _json_bound(review, evidence, "recorded_review_manifest") if review else None
    visual["media"], copies = _media(evidence, frames, manifest, review.parent if review else frames_path.parent if frames_path else None)
    if evidence.time_s is not None and set(evidence.channel_names) == set(_LABELS) and len(evidence.channel_names) == 5:
        values = visual["tactile_series"]
        loads = build_load_series(values["time_s"], values["normal_force_n"], tangent, evidence.channel_names,
                                  values["gap_after"], values["nominal_interval_s"])
        loads["status"] = "available" if loads["normal_status"] == "available" else "partial"
    else:
        loads = {"schema_version": "evaluation-five-finger-load-extension-v1", "status": "missing", "unit": "N",
                 "time_s": [], "channels": [], "normal_force_n": [], "normal_total_n": [], "tangent_force_n": [],
                 "tangent_total_n": [], "gap_after": [], "normal_status": "missing", "tangent_status": "missing",
                 "nominal_interval_s": None, "warnings": ["No explicit recorded right-five-finger trace"]}
    loads.update(trial_id=trial["id"], task=group["task"], prediction_horizon=group["prediction_horizon"])
    plots, plot_jobs = _plots(evidence)
    copies.extend(plot_jobs)
    crop = model_views_for_case(summary, visual["media"].get("sha256"))
    evidence.provenance = _rows(evidence.provenance + crop.get("provenance", []))
    stage = _staged(evidence, group, trial)
    config = {"schema_version": "evaluation-workbench-v1", "trial_id": trial["id"], "task": group["task"],
              "prediction_horizon": group["prediction_horizon"], "checkpoint": group["checkpoint"], "seed": trial["seed"],
              "generic": True, "suite_url": suite_url, "task_presentation": task_presentation(group["task"]),
              "media_views": crop,
              "expert_video": {"status": "missing", "reason": "No independently bound same-task expert media attached"},
              "diagnostic_plots": plots, "load_status": loads["status"], **presentation}
    if stage is not None:
        area = stage.pop("whiteboard_contact_area", None)
        if area is not None:
            config["whiteboard_contact_area"] = area
            visual["task_series"].extend(copy.deepcopy(area["series"]))
        config["staged_evaluation"] = stage
        report["recorded_staged_evaluation"] = {key: value for key, value in stage.items() if key != "timeline"}
    report["source_binding"] = binding
    provenance = _rows(evidence.provenance)
    report["provenance"], loads["provenance"] = provenance, provenance
    report["visualization"] = {"schema_version": visual["schema_version"], "path": "visualization.json", "sha256": content_hash(visual)}
    availability = {"native_samples": len(evidence.time_s) if evidence.time_s is not None else 0,
                    "review_frames": len(frames), "task_series": len(visual["task_series"]),
                    "stage_evidence_status": trial.get("staged_evidence_status"),
                    "applied_actions": "recorded" if controls.is_file() else "not_recorded",
                    "expert": "not_attached", "diagnostic_plots": plots["status"]}
    result = {"report": report, "visualization": visual, "tactile_loads": loads, "config": config, "copies": copies,
              "provenance": provenance, "source_binding": binding, "availability": availability,
              "state_trace": _state_trace(evidence, trace)}
    if stage is not None:
        result["unified_staged"] = stage
    for key in ("report", "visualization", "tactile_loads", "config", "state_trace", "source_binding", "availability"):
        content_hash(result[key])
    _verify(provenance)
    return result


__all__ = ["SCHEMA", "resolve_rich_source", "build_rich_case"]
