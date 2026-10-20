"""Publish one identity-bound poker tactile-policy experiment snapshot.

The producer job is read-only here.  A stopped job may be incomplete, but all
planned seed slots remain visible and only valid, identity-matched evidence
enters native or staged denominators.  Existing H50 and EgoTouch snapshots are
linked, never rewritten or pooled with this experiment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .export import (
    PACKAGE,
    _bound_document,
    _copy,
    _inside,
    _publish_portal,
    _write,
    digest,
    document,
    encoded,
    inert,
    verify_row,
)

SCHEMA = "kaihand-tactile-policy-experiment-snapshot-v1"
CATALOG_SCHEMA = "kaihand-tactile-policy-experiment-v1"
PLAN_SCHEMA = "kaihand-poker-tactile-experiment-plan-v1"
STATUS_SCHEMA = "kaihand-poker-tactile-experiment-status-v1"
BATCH_SCHEMA = "poker_tactile_batch_v1"
GROUPS = (
    "cpm_best",
    "ftp1_step10000",
    "ftp1_step15000",
    "ftp1_step20000",
    "ftp1_step24999",
)
TITLES = {
    "cpm_best": "CPM best",
    "ftp1_step10000": "FTP-1 step 10000",
    "ftp1_step15000": "FTP-1 step 15000",
    "ftp1_step20000": "FTP-1 step 20000",
    "ftp1_step24999": "FTP-1 step 24999",
}
TERMINAL_JOB_STATES = frozenset({"complete", "incomplete", "failed", "interrupted"})
ACTIVE_GROUP_STATES = frozenset({"pending", "preparing", "prepared", "running", "starting"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _content_identity(value):
    raw = json.dumps(
        {key: item for key, item in value.items() if key != "deployment_id"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _add_input(inputs, row, *, role=None):
    row = dict(row)
    row["path"] = str(Path(row["path"]).resolve(strict=True))
    if role:
        row["role"] = role
    previous = inputs.get(row["path"])
    if previous and (previous["size"], previous["sha256"]) != (row["size"], row["sha256"]):
        raise ValueError(f"Conflicting source identity: {row['path']}")
    inputs[row["path"]] = previous or row


def _bound(path, inputs, role):
    value, row = _bound_document(path)
    _add_input(inputs, row, role=role)
    return value


def _safe_source(root, path, label, *, required=True):
    """Resolve one producer artifact without accepting aliases outside ``root``."""
    root = Path(root).resolve(strict=True)
    raw = Path(os.path.abspath(path))
    _require(raw.is_relative_to(root), f"{label} escapes its source root")
    relative = raw.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        _require(not cursor.is_symlink(), f"{label} must not use symlinks")
    resolved = raw.resolve(strict=False)
    _require(resolved.is_relative_to(root), f"{label} escapes its source root")
    if required:
        resolved = raw.resolve(strict=True)
        _require(resolved.is_file(), f"{label} is not a regular file")
    return resolved


def _bound_source(root, path, inputs, role):
    path = _safe_source(root, path, role)
    return _bound(path, inputs, role)


def _bind_raw_source(root, path, inputs, role):
    """Bind malformed evidence bytes without attempting to parse them."""
    path = _safe_source(root, path, role)
    row = digest(path)
    _add_input(inputs, row, role=role)


def _safe_expected(job, value, expected, label):
    _require(isinstance(value, str) and value, f"Missing {label}")
    path = Path(value)
    _require(path.is_absolute(), f"{label} must be absolute")
    resolved = _safe_source(job, path, label, required=False)
    _require(resolved == expected.resolve(strict=False), f"Unexpected {label}: {path}")
    _require(resolved.is_relative_to(job), f"{label} escapes job root")
    return resolved


def _validate_protocol(plan):
    protocol = plan.get("protocol")
    _require(isinstance(protocol, dict), "Missing experiment protocol")
    required = {
        "guard_mode": "observe",
        "action_interpolation": "zoh",
        "execute_steps": 5,
        "control_hz": 30,
        "record_fps": 10,
        "staged_evaluation_version": "staged_v1",
        "diagnostic_only": True,
        "formal_metrics_valid": False,
    }
    for key, expected in required.items():
        _require(type(protocol.get(key)) is type(expected) and protocol.get(key) == expected,
                 f"Unsupported experiment protocol: {key}")
    _require(_finite(protocol.get("max_sim_seconds"))
             and float(protocol["max_sim_seconds"]) == 90.0,
             "Experiment max_sim_seconds must be exactly 90")
    return dict(protocol)


def _validate_plan(plan, job):
    _require(plan.get("schema") == PLAN_SCHEMA, "Unsupported tactile experiment plan")
    _require(plan.get("task") == "poker-draw", "Experiment task must be poker-draw")
    _safe_expected(job, plan.get("job_root"), job, "plan.job_root")
    seeds = plan.get("seeds")
    _require(isinstance(seeds, list) and seeds, "Plan needs at least one seed")
    _require(all(type(seed) is int and seed >= 0 for seed in seeds), "Invalid plan seed")
    _require(len(seeds) == len(set(seeds)), "Duplicate plan seed")
    groups = plan.get("groups")
    _require(isinstance(groups, list), "Missing plan groups")
    _require([row.get("label") for row in groups if isinstance(row, dict)] == list(GROUPS),
             "Plan must contain the fixed five groups in declared order")
    _require(plan.get("planned_trials") == len(seeds) * len(GROUPS),
             "planned_trials disagrees with groups and seeds")
    protocol = _validate_protocol(plan)
    validated = []
    for row in groups:
        label = row["label"]
        expected_family = "pi05+cpm" if label == "cpm_best" else "ftp1"
        expected_horizon = 30 if label == "cpm_best" else 32
        expected_start = 0 if label == "cpm_best" else 1
        _require(row.get("task") == "poker-draw", f"Wrong task for {label}")
        _require(row.get("model_family") == expected_family, f"Wrong model family for {label}")
        _require(row.get("prediction_horizon") == expected_horizon, f"Wrong horizon for {label}")
        _require(row.get("action_start_index") == expected_start, f"Wrong action start for {label}")
        _require(row.get("execute_steps") == 5, f"Wrong execute_steps for {label}")
        expected_step = "best" if label == "cpm_best" else label.removeprefix("ftp1_step")
        _require(str(row.get("checkpoint_step")) == expected_step,
                 f"Wrong checkpoint step for {label}")
        _require(isinstance(row.get("checkpoint"), str) and row["checkpoint"],
                 f"Missing checkpoint for {label}")
        manifest = job / "deployments" / label / "deployment_manifest.json"
        evaluation = job / "groups" / label / "attempt_001" / "evaluation"
        _safe_expected(job, row.get("deployment_manifest"), manifest, f"{label} manifest")
        _safe_expected(job, row.get("evaluation_path"), evaluation, f"{label} evaluation")
        validated.append({**row, "manifest_path": manifest, "evaluation_path_obj": evaluation})
    return seeds, validated, protocol


def _manifest_guidance(manifest, family):
    if family == "ftp1":
        _require(manifest.get("guidance") in (None, {}), "FTP-1 manifest unexpectedly enables guidance")
        return {"enabled": False, "steps": 0, "scale": 0.0, "variant": "not_applicable"}
    guidance = manifest.get("guidance")
    _require(isinstance(guidance, dict), "CPM manifest is missing guidance")
    steps = guidance.get("guidance_steps", guidance.get("steps"))
    scale = guidance.get("scale")
    variant = guidance.get("variant")
    _require(type(steps) is int and steps == 3, "CPM guidance steps must be 3")
    _require(_finite(scale) and float(scale) == 1.0, "CPM guidance scale must be 1")
    _require(variant == "pi05+touchguidance", "Invalid CPM guidance variant")
    return {"enabled": True, "steps": steps, "scale": float(scale), "variant": variant}


def _validate_manifest(manifest, identity, manifest_sha):
    label = identity["label"]
    expected = {
        "task": "poker-draw",
        "model_family": identity["model_family"],
        "checkpoint_path": identity["checkpoint"],
        "prediction_horizon": identity["prediction_horizon"],
        "action_start_index": identity["action_start_index"],
        "execute_steps": 5,
        "control_hz": 30,
        "controller_version": "poker-tactile-direct-joints-v1",
        "required_action_interpolation": "zoh",
        "diagnostic_only": True,
        "formal_metrics_valid": False,
    }
    for key, value in expected.items():
        _require(type(manifest.get(key)) is type(value) and manifest.get(key) == value,
                 f"Manifest mismatch for {label}: {key}")
    _require(manifest.get("schema") in {"poker_tactile_deployment_v1", "tactile_deployment_v2"},
             f"Unsupported deployment manifest for {label}")
    _require(_sha(manifest.get("deployment_id")), f"Invalid deployment_id for {label}")
    _require(manifest["deployment_id"] == _content_identity(manifest),
             f"Manifest deployment_id does not bind its content for {label}")
    _require(_sha(manifest.get("checkpoint_sha256")), f"Invalid checkpoint SHA for {label}")
    _require(str(manifest.get("checkpoint_step")) == str(identity.get("checkpoint_step")),
             f"Checkpoint step mismatch for {label}")
    observation = manifest.get("observation_contract")
    _require(isinstance(observation, dict) and observation.get("tactile_sent_to_model") is True,
             f"Tactile input is not frozen for {label}")
    guidance = _manifest_guidance(manifest, identity["model_family"])
    return {
        "deployment_id": manifest["deployment_id"],
        "deployment_manifest_sha256": manifest_sha,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "controller": manifest["controller_version"],
        "tactile_sent_to_model": True,
        "guidance": guidance,
    }


def _validate_batch(batch, expected):
    label = expected["label"]
    fixed = {
        "schema": BATCH_SCHEMA,
        "task": "poker-draw",
        "model_family": expected["model_family"],
        "prediction_horizon": expected["prediction_horizon"],
        "action_start_index": expected["action_start_index"],
        "execute_steps": 5,
        "deployment_id": expected["deployment_id"],
        "deployment_manifest_sha256": expected["deployment_manifest_sha256"],
    }
    for key, value in fixed.items():
        _require(type(batch.get(key)) is type(value) and batch.get(key) == value,
                 f"Batch identity mismatch for {label}: {key}")
    _require(type(batch.get("complete")) is bool, f"Batch complete flag missing for {label}")
    for key, value in {"diagnostic_only": True, "formal_metrics_valid": False}.items():
        # A stopped partial producer may retain its incremental batch before
        # these finalization fields are appended.  If present they are still
        # identity fields; a complete batch must always carry them.
        if batch["complete"] is True or key in batch:
            _require(type(batch.get(key)) is type(value) and batch.get(key) == value,
                     f"Batch identity mismatch for {label}: {key}")
    _require(isinstance(batch.get("trials"), list), f"Batch trials missing for {label}")


def _batch_gate(row):
    reasons = []
    if type(row.get("valid")) is not bool:
        reasons.append("invalid_batch_valid")
    elif row["valid"] is not True:
        reasons.append("batch_declared_invalid")
    if type(row.get("timed_out")) is not bool:
        reasons.append("invalid_batch_timed_out")
    elif row["timed_out"] is True:
        reasons.append("batch_timed_out")
    if type(row.get("returncode")) is not int:
        reasons.append("invalid_batch_returncode")
    elif row["returncode"] != 0:
        reasons.append("batch_nonzero_returncode")
    if row.get("identity_matches") is not True:
        reasons.append("batch_identity_not_matched")
    return reasons


def _validate_status_group(job, status_row, plan_group):
    label = plan_group["label"]
    _require(isinstance(status_row.get("state"), str) and status_row["state"],
             f"Status state missing for {label}")
    accepted = status_row.get("accepted_attempt")
    _require(accepted in (None, "attempt_001"), f"Unexpected accepted attempt for {label}")
    _safe_expected(
        job,
        status_row.get("evaluation_path"),
        plan_group["evaluation_path_obj"],
        f"status {label} evaluation_path",
    )
    batch_reference = status_row.get("batch_summary")
    if batch_reference is not None:
        _safe_expected(
            job,
            batch_reference,
            plan_group["evaluation_path_obj"] / "batch_summary.json",
            f"status {label} batch_summary",
        )
    if accepted is not None:
        _require(status_row.get("batch_complete") is True,
                 f"Accepted attempt has no complete batch for {label}")


def _expected_identity(group, protocol):
    return {
        "task": "poker-draw",
        "model_family": group["model_family"],
        "deployment_manifest_sha256": group["deployment_manifest_sha256"],
        "prediction_horizon": group["prediction_horizon"],
        "action_start_index": group["action_start_index"],
        "execute_steps": 5,
        "tactile_sent_to_model": True,
        "guidance": group["guidance"],
        "protocol": {
            "max_sim_seconds": float(protocol["max_sim_seconds"]),
            "guard_mode": protocol["guard_mode"],
            "controller": group["controller"],
            "action_interpolation": protocol["action_interpolation"],
        },
    }


def _compare_summary_identity(classified, summary, expected, group):
    reasons = []
    actual = classified.identity
    if actual is None:
        return [*classified.reasons]
    if classified.mode != "diagnostic":
        reasons.append("summary_mode_is_not_diagnostic")
    for key in (
        "task", "model_family", "deployment_manifest_sha256", "prediction_horizon",
        "action_start_index", "execute_steps", "tactile_sent_to_model", "guidance",
    ):
        if actual.get(key) != expected.get(key):
            reasons.append(f"summary_identity_mismatch:{key}")
    for key, value in expected["protocol"].items():
        if actual["protocol"].get(key) != value:
            reasons.append(f"summary_protocol_mismatch:{key}")
    direct = {
        "deployment_id": group["deployment_id"],
        "checkpoint_sha256": group["checkpoint_sha256"],
        "checkpoint_path": group["checkpoint"],
    }
    for key, value in direct.items():
        if summary.get(key) != value:
            reasons.append(f"summary_identity_mismatch:{key}")
    expected_expert = group["model_family"] == "pi05+cpm"
    if summary.get("expert_used") is not expected_expert:
        reasons.append("summary_identity_mismatch:expert_used")
    return reasons


def _safe_summary(job, evaluation, row, seed):
    expected_dir = evaluation / f"seed_{seed:04d}"
    if row is None:
        path = _safe_source(
            job, expected_dir / "summary.json", f"seed {seed} summary", required=False
        )
        return path if path.is_file() else None, expected_dir
    reference = row.get("summary")
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ValueError(f"Invalid summary reference for seed {seed}")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe summary reference for seed {seed}")
    path = _safe_source(
        job, evaluation / relative, f"seed {seed} summary", required=False
    )
    if path.parent != expected_dir.resolve(strict=False) or path.name != "summary.json":
        raise ValueError(f"Unexpected summary location for seed {seed}")
    return (path if path.is_file() else None), expected_dir


def _staged_projection(source_root, trial_dir, horizon, native_valid, inputs):
    sidecar = _safe_source(
        source_root,
        trial_dir / "staged_evaluation.json",
        "staged sidecar",
        required=False,
    )
    if not sidecar.is_file():
        return None, "missing", [], None
    try:
        from evaluation_unified.staged import load_staged_result

        loaded = load_staged_result(
            trial_dir,
            include_state_log=False,
            required_prediction_horizon=horizon,
            integrity_profile="tactile_manifest_inventory_v1",
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Preserve the exact failed evidence as a verification dependency.  A
        # later repair or replacement must invalidate this snapshot rather
        # than silently changing why the record was excluded.
        for path, role in (
            (sidecar, "failed_staged_result"),
            (trial_dir / "staged_states.jsonl", "failed_staged_state_log"),
        ):
            candidate = _safe_source(source_root, path, role, required=False)
            if candidate.is_file():
                _bind_raw_source(source_root, candidate, inputs, role)
        return None, "invalid", [f"strict_staged_import_failed:{error}"], None
    for row in loaded["provenance"]:
        _add_input(inputs, row)
    projected = loaded["staged_result"]
    complete = projected.get("complete_success")
    if complete is not None and type(complete) is not bool:
        raise ValueError("Strict staged result returned a nonboolean completion")
    status = "verified" if native_valid else "verified_excluded_invalid_native"
    rule = loaded["rule_identity"]
    rule_contract = {
        key: rule.get(key) for key in (
            "observer_sha256", "recording_sha256", "definitions_thresholds_sha256",
        )
    }
    rule_contract.update(
        full_rule_identity_sha256=rule.get("sha256"),
        binding_basis=rule.get("binding_basis", rule.get("binding")),
    )
    _require(all(_sha(rule_contract[key]) for key in (
        "observer_sha256", "recording_sha256", "definitions_thresholds_sha256",
        "full_rule_identity_sha256",
    )), "Strict staged loader returned an invalid rule identity")
    return loaded, status, [], rule_contract


def _rate(successes, known):
    return None if not known else successes / known


def _counts(trials):
    values = {"planned": len(trials), "observed": 0, "valid": 0, "invalid": 0,
              "missing": 0}
    for trial in trials:
        values["observed"] += trial["observed"] is True
        status = trial["evidence_status"]
        values[status if status in {"valid", "invalid"} else "missing"] += 1
    return values


def _staged_stats(trials):
    verified = [trial for trial in trials
                if trial["evidence_status"] == "valid"
                and trial["staged_evidence_status"] == "verified"]
    complete = [trial for trial in verified if type(trial["staged_complete_success"]) is bool]
    definitions = None
    stage_rows = {}
    for trial in verified:
        result = trial.get("_staged_result") or {}
        current = result.get("stage_definitions")
        flags = result.get("stage_flags")
        if not isinstance(current, dict) or not isinstance(flags, dict):
            continue
        if definitions is None:
            definitions = current
        elif definitions != current:
            raise ValueError("Verified staged definitions differ within one experiment")
        for key, name in current.items():
            row = stage_rows.setdefault(key, {"key": key, "name": name, "known": 0, "reached": 0})
            _require(row["name"] == name, "Staged label changed within experiment")
            if type(flags.get(key)) is bool:
                row["known"] += 1
                row["reached"] += flags[key] is True
    successes = sum(trial["staged_complete_success"] is True for trial in complete)
    evidence_counts = {}
    for trial in trials:
        status = trial["staged_evidence_status"]
        evidence_counts[status] = evidence_counts.get(status, 0) + 1
    return {"known": len(complete), "successes": successes,
            "success_rate": _rate(successes, len(complete)),
            "evidence_counts": evidence_counts,
            "stages": list(stage_rows.values())}


def _detail_page(group, trial, summary):
    validity = {"valid": "有效（进入原生分母）", "invalid": "无效（不计分）",
                "missing": "缺失（不计分）"}[trial["evidence_status"]]
    native = "成功" if trial["native_success"] is True else (
        "未完成" if trial["native_success"] is False else "未知")
    staged = "完整阶段达成" if trial["staged_complete_success"] is True else (
        "完整阶段未达成" if trial["staged_complete_success"] is False else "阶段未知")
    media = ('<video controls preload="metadata" playsinline src="review.mp4"></video>'
             if trial.get("video_url") else "<p>本条没有可核验的视频产物。</p>")
    facts = {
        "seed": trial["seed"],
        "evidence_status": trial["evidence_status"],
        "observed_in_batch": trial["observed"],
        "producer_status": trial["status"],
        "native_success": trial["native_success"],
        "staged_evidence_status": trial["staged_evidence_status"],
        "staged_complete_success": trial["staged_complete_success"],
        "reasons": trial["reasons"],
    }
    if isinstance(summary, dict):
        facts.update(sim_seconds=summary.get("sim_seconds"), stop_reason=summary.get("stop_reason"),
                     physics_valid=summary.get("physics_valid"), error=summary.get("error"))
    staged_link = (' · <a href="staged_result.json">严格 staged 投影</a>'
                   if trial.get("staged_result_url") else "")
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(group["title"])} · seed {trial["seed"]}</title>'
            '<style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:15px/1.65 system-ui;color:#17241f}'
            'a{color:#176b4d}video{width:100%;max-height:72vh;background:#111;border-radius:10px}'
            '.badge{display:inline-block;margin:3px 7px 3px 0;padding:4px 10px;border-radius:999px;background:#edf5f1}'
            '.warning{padding:14px;border:1px solid #d8e3dd;border-radius:10px;background:#f8faf9}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>'
            '<a href="../../../index.html">← 返回 Poker 触觉模型对比</a>'
            f'<h1>{escape(group["title"])} · seed {trial["seed"]}</h1>'
            f'<span class="badge">{validity}</span><span class="badge">原生：{native}</span>'
            f'<span class="badge">staged：{staged}</span>'
            '<p class="warning">原生成功与累计 staged 完成是两层证据；invalid、missing 和未严格校验的 staged '
            '记录均不进入成功率分母。</p>'
            f'{media}<h2>身份与判定</h2><pre>{escape(json.dumps(facts, ensure_ascii=False, indent=2))}</pre>'
            + ('<p><a href="summary.json">冻结 summary.json</a>' if summary is not None else '<p>无 summary.json')
            + staged_link + ' · <a href="trial.json">发布记录</a></p></html>')


def _page(catalog):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Poker 触觉模型对比 | KaiTactileSim</title>'
            '<link rel="stylesheet" href="assets/tactile_experiment.css">'
            '<script defer src="assets/tactile_experiment.js"></script>'
            '<main id="tactile-experiment"></main>'
            f'<script type="application/json" id="tactile-experiment-data">{inert(catalog)}</script>'
            '<noscript>请启用 JavaScript，或查看 <a href="catalog.json">JSON 清单</a>。</noscript></html>')


def _portal(output, h50):
    tactile = output.name + "/index.html"
    h50_link = h50.name + "/index.html"
    ego = h50.name + "/egotouch.html"
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>KaiTactileSim 评估平台</title><style>body{max-width:900px;margin:48px auto;padding:20px;'
            'font:17px/1.7 system-ui;color:#183229}a{color:#176b4d}.card{display:block;margin:14px 0;padding:18px;'
            'border:1px solid #d8e3dd;border-radius:12px;text-decoration:none}</style>'
            '<h1>KaiTactileSim 评估平台</h1>'
            f'<a class="card" href="{escape(tactile, quote=True)}"><strong>Poker 触觉模型对比</strong><br>'
            'CPM best 与 FTP-1 四个 checkpoint；native / staged 分开。</a>'
            f'<a class="card" href="{escape(h50_link, quote=True)}"><strong>π0.5 H50 同实验</strong><br>'
            '450 条成员，430 条计分、20 条隔离回放。</a>'
            f'<a class="card" href="{escape(ego, quote=True)}"><strong>EgoTouch 独立实验</strong><br>'
            '不与 H50 或 Poker 触觉实验合并统计。</a></html>').encode()


def _bind_related_snapshot(h50, inputs):
    manifest = _bound_source(
        h50, h50 / "artifact_manifest.json", inputs, "related_snapshot_manifest"
    )
    rows = {Path(row["path"]).as_posix(): row for row in manifest.get("files", [])
            if isinstance(row, dict) and isinstance(row.get("path"), str)}
    for name in ("index.html", "egotouch.html"):
        row = rows.get(name)
        _require(row is not None, f"Related snapshot manifest does not bind {name}")
        path = _safe_source(h50, h50 / name, f"related snapshot {name}")
        verify_row(row, path)
        _add_input(inputs, {**row, "path": str(path)}, role="related_snapshot_entry")


def _read_group(
    job, plan_group, status_row, seeds, protocol, output, artifacts, inputs,
):
    from tactile_evaluation_summary import classify_trial

    label = plan_group["label"]
    manifest_path = plan_group["manifest_path"]
    evaluation = plan_group["evaluation_path_obj"]
    manifest = None
    identity = {
        key: plan_group[key] for key in (
            "label", "model_family", "checkpoint", "checkpoint_step", "prediction_horizon",
            "action_start_index", "execute_steps",
        )
    }
    manifest_path = _safe_source(
        job, manifest_path, f"{label} deployment manifest", required=False
    )
    if manifest_path.is_file():
        manifest = _bound_source(
            job, manifest_path, inputs, f"{label}_deployment_manifest"
        )
        manifest_sha = inputs[str(manifest_path.resolve(strict=True))]["sha256"]
        identity.update(_validate_manifest(manifest, identity, manifest_sha))
    batch_path = _safe_source(
        job,
        evaluation / "batch_summary.json",
        f"{label} batch summary",
        required=False,
    )
    batch = None
    if batch_path.is_file():
        _require(manifest is not None, f"Batch exists without manifest for {label}")
        batch = _bound_source(job, batch_path, inputs, f"{label}_batch_summary")
        _validate_batch(batch, identity)
    _write(output, Path("source/deployments") / f"{label}.json",
           encoded(manifest if manifest is not None else {"status": "missing"}), artifacts)
    _write(output, Path("source/batches") / f"{label}.json",
           encoded(batch if batch is not None else {"status": "missing"}), artifacts)
    rows = {}
    if batch is not None:
        for row in batch["trials"]:
            _require(isinstance(row, dict), f"Invalid batch trial for {label}")
            seed = row.get("seed")
            _require(type(seed) is int and seed in seeds, f"Unexpected batch seed for {label}")
            _require(seed not in rows, f"Duplicate batch seed for {label}: {seed}")
            rows[seed] = row
    expected = _expected_identity(identity, protocol) if manifest is not None else None
    trials, hold_values, rule_keys, full_rule_contracts = [], set(), set(), {}
    for seed in seeds:
        row = rows.get(seed)
        summary_path, trial_dir = _safe_summary(job, evaluation, row, seed)
        observed = row is not None
        reasons = [] if row is not None else ["missing_batch_row"]
        summary = None
        classified = None
        evidence = "missing"
        native = None
        producer_status = "missing"
        if summary_path is not None:
            summary = _bound_source(
                job, summary_path, inputs, f"{label}_seed_{seed}_summary"
            )
            classified = classify_trial(summary, source=str(summary_path))
            producer_status = classified.status or "unknown"
            reasons.extend(classified.reasons)
            if row is None:
                reasons.append("unbatched_summary_excluded")
            else:
                reasons.extend(_batch_gate(row))
                reasons.extend(_compare_summary_identity(classified, summary, expected, identity))
                if classified.seed != seed:
                    reasons.append("summary_seed_mismatch")
                evidence = classified.evidence_status
                if reasons:
                    if any(reason.startswith("missing_") for reason in reasons) and not any(
                        reason.startswith(("invalid_", "batch_", "summary_"))
                        for reason in reasons
                    ):
                        evidence = "missing"
                    else:
                        evidence = "invalid"
                native = classified.native_success if evidence == "valid" else None
            hold = classified.identity["protocol"].get("success_hold_seconds") \
                if classified.identity else None
            if hold is not None:
                hold_values.add(hold)
        elif row is not None:
            reasons.extend([*_batch_gate(row), "missing_referenced_summary"])
            evidence = "invalid" if _batch_gate(row) else "missing"
            producer_status = str(row.get("status") or evidence)
        staged_loaded, staged_status, staged_reasons, rule_contract = _staged_projection(
            job, trial_dir, identity["prediction_horizon"], evidence == "valid", inputs
        )
        reasons.extend(staged_reasons)
        if rule_contract is not None:
            rule_key = tuple(rule_contract[key] for key in (
                "observer_sha256", "recording_sha256", "definitions_thresholds_sha256",
            ))
            rule_keys.add(rule_key)
            full_sha = rule_contract["full_rule_identity_sha256"]
            previous = full_rule_contracts.setdefault(full_sha, rule_contract)
            _require(previous == rule_contract, "Contradictory strict staged rule identity")
        staged_result = staged_loaded["staged_result"] if staged_loaded else None
        staged_complete = (staged_result.get("complete_success")
                           if staged_status == "verified" else None)
        relative = Path("trials") / label / f"seed_{seed:04d}"
        trial = {
            "id": f"{label}_seed_{seed:04d}", "seed": seed, "observed": observed,
            "status": producer_status, "evidence_status": evidence,
            "native_success": native, "reasons": list(dict.fromkeys(reasons)),
            "tactile_sent_to_model": (
                summary.get("tactile_sent_to_model") if summary is not None else None
            ),
            "expert_used": summary.get("expert_used") if summary is not None else None,
            "guidance": (
                classified.identity.get("guidance")
                if classified is not None and classified.identity is not None else None
            ),
            "staged_evidence_status": staged_status,
            "staged_complete_success": staged_complete,
            "recorded_staged_complete_success": (
                staged_result.get("complete_success") if staged_result else None
            ),
            "detail_url": (relative / "index.html").as_posix(),
            "video_url": None, "staged_result_url": None,
            "_staged_result": staged_result,
        }
        if summary is not None:
            _write(output, relative / "summary.json", encoded(summary), artifacts)
        if staged_loaded is not None:
            _write(output, relative / "staged_result.json", encoded(staged_loaded), artifacts)
            trial["staged_result_url"] = (relative / "staged_result.json").as_posix()
        video = _safe_source(
            job, trial_dir / "review.mp4", f"{label} seed {seed} video", required=False
        )
        if video.is_file():
            _copy(output, relative / "review.mp4", video, artifacts, inputs)
            trial["video_url"] = (relative / "review.mp4").as_posix()
        public_trial = {key: value for key, value in trial.items() if key != "_staged_result"}
        _write(output, relative / "trial.json", encoded(public_trial), artifacts)
        _write(output, relative / "index.html", _detail_page(
            {"title": TITLES[label]}, public_trial, summary
        ).encode(), artifacts)
        trials.append(trial)
    _require(len(hold_values) <= 1, f"success_hold_seconds changed within {label}")
    _require(len(rule_keys) <= 1, f"Strict staged rule identity changed within {label}")
    _require(len(full_rule_contracts) <= 1,
             f"Full strict staged source contract changed within {label}")
    counts = _counts(trials)
    native_known = sum(trial["evidence_status"] == "valid" for trial in trials)
    native_successes = sum(trial["native_success"] is True for trial in trials
                           if trial["evidence_status"] == "valid")
    staged = _staged_stats(trials)
    staged["rule_contract"] = (
        next(iter(full_rule_contracts.values())) if full_rule_contracts else None
    )
    for trial in trials:
        trial.pop("_staged_result", None)
    source_state = status_row.get("state", "pending")
    group_status = source_state
    if source_state in ACTIVE_GROUP_STATES:
        group_status = "partial" if counts["observed"] else "pending"
    protocol_view = {
        "max_sim_seconds": float(protocol["max_sim_seconds"]),
        "guard_mode": protocol["guard_mode"],
        "controller": identity.get("controller"),
        "action_interpolation": protocol["action_interpolation"],
        "success_hold_seconds": next(iter(hold_values), None),
        "staged_evaluation_version": protocol["staged_evaluation_version"],
    }
    group_identity = {
        **{key: identity.get(key) for key in (
            "model_family", "checkpoint", "checkpoint_step", "checkpoint_sha256",
            "deployment_id", "deployment_manifest_sha256", "prediction_horizon",
            "action_start_index", "execute_steps", "tactile_sent_to_model", "guidance",
        )},
        "checkpoint_label": ("best" if label == "cpm_best" else f"step {identity['checkpoint_step']}"),
        "evaluation_mode": "完整时长诊断评估",
        "diagnostic_only": True,
        "formal_metrics_valid": False,
        "protocol": protocol_view,
    }
    return {
        "label": label, "title": TITLES[label], "status": group_status,
        "source_status": source_state, "accepted_attempt": status_row.get("accepted_attempt"),
        "batch_complete": bool(batch and batch.get("complete") is True),
        "identity": group_identity, "counts": counts,
        "native": {"known": native_known, "successes": native_successes,
                   "success_rate": _rate(native_successes, native_known)},
        "staged": staged, "trials": trials,
    }, hold_values, rule_keys


def export_tactile_experiment(
    job_root,
    h50_snapshot,
    output,
    *,
    publish_index=False,
    require_complete=False,
):
    job = Path(job_root).resolve(strict=True)
    h50 = Path(h50_snapshot).resolve(strict=True)
    output = Path(output).absolute()
    _require(job.is_dir(), "Job root is not a directory")
    _require(output.resolve() == output and not output.exists() and not output.is_symlink(),
             "Export into a fresh canonical output directory")
    _require(output.parent == h50.parent, "Output must be a sibling of the frozen H50 snapshot")
    _require(not output.is_relative_to(job), "Output must be outside the producer job")
    inputs, artifacts = {}, []
    plan = _bound_source(job, job / "plan.json", inputs, "experiment_plan")
    seeds, plan_groups, protocol = _validate_plan(plan, job)
    status = _bound_source(job, job / "status.json", inputs, "terminal_experiment_status")
    _require(status.get("schema") == STATUS_SCHEMA, "Unsupported experiment status")
    _require(status.get("state") in TERMINAL_JOB_STATES,
             "Producer is still active; publish only a stopped job")
    status_groups = status.get("groups")
    _require(isinstance(status_groups, dict) and set(status_groups) == set(GROUPS),
             "Status groups disagree with the plan")
    _require(all(isinstance(status_groups[label], dict) for label in GROUPS),
             "Status group rows must be objects")
    for plan_group in plan_groups:
        _validate_status_group(job, status_groups[plan_group["label"]], plan_group)
    status_complete = status["state"] == "complete"
    if status_complete:
        _require(all(status_groups[label].get("state") == "complete"
                     and status_groups[label].get("accepted_attempt") == "attempt_001"
                     and status_groups[label].get("batch_complete") is True
                     for label in GROUPS),
                 "Complete producer status contradicts its group states")
    if require_complete:
        _require(status_complete, "Complete publication requires complete producer status")
        _require(all(status_groups[label].get("accepted_attempt") == "attempt_001"
                     and status_groups[label].get("batch_complete") is True for label in GROUPS),
                 "Complete publication requires accepted complete batches")
    _bind_related_snapshot(h50, inputs)
    portal = output.parent / "index.html"
    portal_before = portal.read_bytes() if portal.is_file() else None
    output.mkdir()
    _write(output, Path("source/plan.json"), encoded(plan), artifacts)
    _write(output, Path("source/status.json"), encoded(status), artifacts)
    groups, all_holds, all_rules = [], set(), set()
    for plan_group in plan_groups:
        label = plan_group["label"]
        _require(isinstance(status_groups[label], dict), f"Invalid status group: {label}")
        group, holds, rules = _read_group(
            job, plan_group, status_groups[label], seeds, protocol, output, artifacts, inputs
        )
        groups.append(group)
        all_holds.update(holds)
        all_rules.update(rules)
    _require(len(all_holds) <= 1, "success_hold_seconds differs across model groups")
    _require(len(all_rules) <= 1, "Strict staged rule contract differs across model groups")
    totals = {key: sum(group["counts"][key] for group in groups)
              for key in ("planned", "observed", "valid", "invalid", "missing")}
    _require(totals["planned"] == plan["planned_trials"], "Published slots disagree with plan")
    if status_complete or require_complete:
        _require(all(group["batch_complete"] and group["accepted_attempt"] == "attempt_001"
                     for group in groups), "Complete publication requires accepted complete batches")
        _require(totals["observed"] == totals["planned"] and totals["missing"] == 0,
                 "Complete publication has unobserved or missing evidence")
    relative_h50 = "../" + h50.name + "/index.html"
    relative_ego = "../" + h50.name + "/egotouch.html"
    generated = datetime.now(timezone.utc).isoformat()
    catalog = {
        "schema": CATALOG_SCHEMA,
        "generated_utc": generated,
        "experiment": {
            "id": job.name, "title": "Poker 触觉模型对比",
            "status": status["state"], "task": "poker-draw",
            "seeds": seeds, "counts": totals,
            "semantics": {
                "membership": "Five predeclared deployment identities times plan.seeds.",
                "native": "Native successes divided only by valid identity-matched trials.",
                "staged": "Complete success divided only by valid trials with strict state-log verification.",
                "separation": "H30/H32 and CPM/FTP-1 remain distinct groups; H50 and EgoTouch are links only.",
                "missing": "Invalid, missing and unverified records never become failures or zero percent.",
            },
        },
        "protocol": protocol,
        "groups": groups,
        "links": [
            {"label": "π0.5 H50 同实验 · 450 条", "url": relative_h50},
            {"label": "EgoTouch 独立实验 · 120 条", "url": relative_ego},
        ],
        "publication": {"source_writes": 0, "producer_runs_started": 0,
                        "existing_snapshot_writes": 0},
    }
    for name in ("tactile_experiment.js", "tactile_experiment.css"):
        _copy(output, Path("assets") / name, PACKAGE / "web" / name, artifacts, inputs)
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    _write(output, Path("index.html"), _page(catalog).encode(), artifacts)
    if portal_before is not None:
        _write(output, Path("previous_root_index.html"), portal_before, artifacts)
    verification = {
        "schema": SCHEMA, "generated_utc": generated,
        "job_root": str(job), "job_terminal_state": status["state"],
        "scope": "Frozen producer outputs; native and strict staged evidence remain separate.",
        "source_files": list(inputs.values()), "source_writes": 0,
        "producer_runs_started": 0, "existing_snapshot_writes": 0,
    }
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SCHEMA, "files": artifacts}), [])
    result = verify_tactile_experiment(output)
    if publish_index:
        _publish_portal(output, portal, portal_before, _portal(output, h50))
        result = verify_tactile_experiment(output)
    return {**result, "output": str(output), "published": publish_index}


def _local_target(output, value):
    _require(isinstance(value, str) and value and "\\" not in value, "Invalid snapshot URL")
    path = Path(value)
    _require(not path.is_absolute(), "Snapshot URL must be relative")
    target = (output / path).resolve(strict=True)
    _require(target.is_relative_to(output.parent) and target.is_file(), "Snapshot URL escapes reports")
    return target


def verify_tactile_experiment(output):
    output = Path(output).resolve(strict=True)
    manifest = document(output / "artifact_manifest.json")
    _require(manifest.get("schema") == SCHEMA, "Unsupported tactile experiment snapshot")
    artifact_paths = set()
    for row in manifest.get("files", []):
        relative = Path(row["path"])
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "Unsafe tactile snapshot artifact")
        _require(relative.as_posix() not in artifact_paths, "Duplicate tactile snapshot artifact")
        artifact_paths.add(relative.as_posix())
        verify_row(row, _inside(output, output / relative))
    verification = document(output / "verification.json")
    _require(verification.get("schema") == SCHEMA, "Verification schema mismatch")
    for row in verification.get("source_files", []):
        verify_row(row)
    catalog = document(output / "catalog.json")
    _require(catalog.get("schema") == CATALOG_SCHEMA, "Catalog schema mismatch")
    groups = catalog.get("groups")
    _require(isinstance(groups, list) and [group.get("label") for group in groups] == list(GROUPS),
             "Catalog groups changed")
    seeds = catalog["experiment"].get("seeds")
    _require(isinstance(seeds, list), "Catalog seeds missing")
    for group in groups:
        trials = group.get("trials")
        _require(isinstance(trials, list) and [trial.get("seed") for trial in trials] == seeds,
                 f"Catalog trial slots changed for {group.get('label')}")
        counts = _counts(trials)
        _require(counts == group.get("counts"), f"Catalog counts disagree for {group['label']}")
        native_trials = [trial for trial in trials if trial["evidence_status"] == "valid"]
        successes = sum(trial.get("native_success") is True for trial in native_trials)
        _require(group["native"] == {"known": len(native_trials), "successes": successes,
                                    "success_rate": _rate(successes, len(native_trials))},
                 f"Native denominator disagrees for {group['label']}")
        staged_trials = [trial for trial in native_trials
                         if trial.get("staged_evidence_status") == "verified"
                         and type(trial.get("staged_complete_success")) is bool]
        staged_successes = sum(trial["staged_complete_success"] for trial in staged_trials)
        _require(group["staged"]["known"] == len(staged_trials)
                 and group["staged"]["successes"] == staged_successes
                 and group["staged"]["success_rate"] == _rate(staged_successes, len(staged_trials)),
                 f"Staged denominator disagrees for {group['label']}")
        for trial in trials:
            _local_target(output, trial["detail_url"])
            if trial.get("video_url"):
                _local_target(output, trial["video_url"])
    totals = {key: sum(group["counts"][key] for group in groups)
              for key in ("planned", "observed", "valid", "invalid", "missing")}
    _require(totals == catalog["experiment"].get("counts"), "Experiment totals disagree")
    for link in catalog.get("links", []):
        _local_target(output, link["url"])
    pending = output / "portal_publication.pending.json"
    _require(not pending.exists(), "Incomplete tactile portal publication")
    portal_matches = None
    audit_path = output / "portal_publication.json"
    if audit_path.exists():
        audit = document(audit_path)
        portal = output.parent / "index.html"
        _require(audit.get("state") == "committed" and audit.get("root_index") == str(portal),
                 "Invalid tactile portal publication audit")
        portal_matches = portal.is_file() and digest(portal)["sha256"] == audit.get("new_sha256")
    return {"status": "passed", "groups": len(groups), "trials": totals["planned"],
            "valid_trials": totals["valid"], "invalid_trials": totals["invalid"],
            "missing_trials": totals["missing"], "artifacts": len(manifest["files"]),
            "root_portal_matches_this_snapshot": portal_matches}
