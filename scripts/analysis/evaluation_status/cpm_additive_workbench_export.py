"""Append the requested, independently checked CPM best n=10 experiment.

The 595-member r5 publication is an immutable input, including its invalid old
CPM stages. No policy, weight, simulator, or retired evaluator is executed by
this display exporter. A new SHA-qualified checkpoint label prevents pooling
the current best.pt with the differently weighted historical ``best`` batch.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .additive_workbench_export import _REPLACED, _REQUIRED, _reviewed_source
from .export import PACKAGE, _copy, _publish_portal, _write, digest, encoded, verify_row
from .reference_workbench_export import verify_reference_workbench
from .rich_workbench_export import _audit_page, _identity, _merge_inputs, _model_plots
from .tactile_experiment_export import (
    PLAN_SCHEMA, STATUS_SCHEMA, _batch_gate, _compare_summary_identity,
    _expected_identity, _safe_source, _staged_projection, _validate_batch,
    _validate_manifest, _validate_protocol, _validate_status_group,
)
from .unified_workbench_export import (
    _Artifacts, _base_group, _counts, _metric_means, _read, _sha, _stages,
    _stats, _validate_catalog,
)

SCHEMA = "kaihand-cpm-additive-workbench-snapshot-v1"
QA_SCHEMA = "kaihand-cpm-poker-independent-qa-v1"
JOB_NAME = "cpm_poker_best_n10_20261021_r2"
CPM_SHA = "e5f859a5755254e55e5fec660e8d11e8b5f27b7400cc4243ad48a133951186f3"
CHECKPOINT_LABEL = "best@" + CPM_SHA[:12]
# Compared to the previously reviewed c31e renderer: the only differences are
# _metrics_text's degree labels. _draw_camera, compose geometry and camera-role
# declarations are identical. This explicit allowlist is local to the new H30
# adapter and does not broaden historical H50 source admission.
CPM_RENDERER_SHA = "323abd4a321affd6a95c82644f772b4aedf1ab0b7ba304105b0fd61fde9fb69a"
_RENDERER_SUFFIX = "/src/kaihand_tactile_env/shared/evaluation_video.py"
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SCOPE_NOTE = (
    "本批次为当前 best.pt 的 10 次新闭环评估，按独立权重 SHA 与部署清单归档；"
    "不与旧 best 权重合并，不恢复旧 CPM 的失效阶段成绩。"
    "使用归档的随机初态设置，10 次样本仅作本批次诊断统计，不代表广泛泛化结论。"
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _bind(job, path, inputs, role):
    actual = _safe_source(job, path, role)
    value, row = _read(actual)
    _merge_inputs(inputs, [{**row, "role": role}])
    return value, row


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _bind_stage_projection(projected, sidecar, embedded):
    """Compare equivalent fields without confusing strict projection with raw.

    The strict importer authenticates the complete raw sidecar, summary and
    runtime inventory before returning a deliberately smaller projection.
    Scores continue to come exclusively from that verified projection.
    """
    _need(isinstance(sidecar, dict) and isinstance(embedded, dict) and _same_json(sidecar, embedded),
          "CPM sidecar differs from summary stage authority")
    fields = ("version", "stage_definitions", "stage_flags", "complete_success", "first_times",
              "thresholds", "continuous_metrics", "stage_achievement_policy", "time_basis")
    _need(isinstance(projected, dict) and all(key in projected and key in sidecar
          and _same_json(projected[key], sidecar[key]) for key in fields),
          "CPM verified stage projection differs from raw stage authority")


def _expected_terminal_state(summary):
    """Mirror rich_detail_data._report's existing display-only normalization.

    Poker's native evaluation stores its outcome fields directly, without a
    nested ``state`` object. The rich adapter intentionally emits an empty
    partial_state in that case; no measurements may be inferred or backfilled.
    """
    evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    state = evaluation.get("state")
    return state if isinstance(state, dict) else {}


def _selected_job(job_root, evaluation_root, qa_receipt):
    """Admit exactly the declared new CPM group, using strict existing checks."""
    from tactile_evaluation_summary import classify_trial

    evaluation_root = Path(evaluation_root).resolve(strict=True)
    requested = Path(job_root).absolute()
    _need(not requested.is_symlink(), "Selected CPM job must not be a symlink")
    job = requested.resolve(strict=True)
    _need(job.parent == evaluation_root / "_jobs" and job.name == JOB_NAME,
          "Select only the requested CPM n10 job")
    inputs = {}
    plan, _ = _bind(job, job / "plan.json", inputs, "cpm_n10_plan")
    status, _ = _bind(job, job / "status.json", inputs, "cpm_n10_complete_status")
    _need(plan.get("schema") == PLAN_SCHEMA and status.get("schema") == STATUS_SCHEMA,
          "Unsupported CPM plan/status schema")
    _need(plan.get("job_root") == str(job) and plan.get("task") == "poker-draw",
          "CPM plan task/job identity mismatch")
    _need(plan.get("seeds") == list(range(10)) and plan.get("planned_trials") == 10,
          "Only the requested seeds 0..9 may be appended")
    _need(plan.get("initial_state_randomization") is True
          and plan.get("trial_scope") == "randomized_initial_state_trials",
          "CPM plan must explicitly preserve randomized-initial-state trial scope")
    groups = plan.get("groups")
    _need(isinstance(groups, list) and len(groups) == 1 and groups[0].get("label") == "cpm_best",
          "CPM plan must contain only cpm_best")
    protocol = _validate_protocol(plan)
    identity = copy.deepcopy(groups[0])
    fixed = {"task": "poker-draw", "model_family": "pi05+cpm", "checkpoint_step": "best",
             "prediction_horizon": 30, "action_start_index": 0, "execute_steps": 5,
             "checkpoint": str(job / "assets/cpm/best.pt")}
    for key, value in fixed.items():
        _need(identity.get(key) == value and type(identity.get(key)) is type(value),
              "CPM group identity mismatch: " + key)
    manifest_path = job / "deployments/cpm_best/deployment_manifest.json"
    evaluation = job / "groups/cpm_best/attempt_001/evaluation"
    _need(identity.get("deployment_manifest") == str(manifest_path)
          and identity.get("evaluation_path") == str(evaluation), "CPM plan paths differ from requested attempt")
    identity.update(manifest_path=manifest_path, evaluation_path_obj=evaluation)
    states = status.get("groups")
    _need(status.get("state") == "complete" and isinstance(states, dict) and set(states) == {"cpm_best"},
          "CPM job must be complete with only the selected group")
    state = states["cpm_best"]
    _validate_status_group(job, state, identity)
    _need(state.get("state") == "complete" and state.get("accepted_attempt") == "attempt_001"
          and state.get("batch_complete") is True, "CPM accepted attempt is not complete")
    manifest, manifest_row = _bind(job, manifest_path, inputs, "cpm_n10_deployment_manifest")
    identity.update(_validate_manifest(manifest, identity, manifest_row["sha256"]))
    _need(identity["checkpoint_sha256"] == CPM_SHA, "CPM best weight SHA differs from requested frozen version")
    batch, _ = _bind(job, evaluation / "batch_summary.json", inputs, "cpm_n10_batch_summary")
    _validate_batch(batch, identity)
    _need(batch.get("complete") is True and len(batch["trials"]) == 10, "CPM batch needs ten completed trial rows")
    rows = batch["trials"]
    _need(all(isinstance(row, dict) and type(row.get("seed")) is int for row in rows)
          and sorted(row["seed"] for row in rows) == list(range(10)), "CPM batch contains missing/duplicate seeds")
    qa, qa_row = _bind(job, Path(qa_receipt), inputs, "cpm_n10_independent_raw_evidence_qa")
    _need(qa.get("schema") == QA_SCHEMA and qa.get("passed") is True
          and qa.get("job_root") == str(job) and qa.get("evaluation_root") == str(evaluation),
          "Independent CPM raw-evidence QA did not pass for this job")
    for key, value in {"deployment_manifest_sha256": manifest_row["sha256"],
                       "checkpoint_sha256": CPM_SHA, "cpm_checkpoint_sha256": CPM_SHA,
                       "planned_trials": 10, "native_valid_trials": 10, "stage_valid_trials": 10}.items():
        _need(qa.get(key) == value and type(qa.get(key)) is type(value), "Independent CPM QA identity mismatch: " + key)
    qa_trials = qa.get("trials")
    _need(isinstance(qa_trials, list) and len(qa_trials) == 10
          and all(isinstance(row, dict) and type(row.get("seed")) is int for row in qa_trials)
          and sorted(row["seed"] for row in qa_trials) == list(range(10)), "Independent CPM QA has duplicate/missing seeds")
    qa_by_seed = {row["seed"]: row for row in qa_trials}
    expected = _expected_identity(identity, protocol)
    trials, native_sources, rule = [], {}, None
    for row in sorted(rows, key=lambda value: value["seed"]):
        seed = row["seed"]
        source = evaluation / f"seed_{seed:04d}"
        _need(row.get("summary") == f"seed_{seed:04d}/summary.json", "CPM batch summary path is not the accepted seed")
        summary, summary_row = _bind(job, source / "summary.json", inputs, "cpm_n10_trial_summary")
        classified = classify_trial(summary, source=str(source / "summary.json"))
        reasons = [*classified.reasons, *_batch_gate(row), *_compare_summary_identity(classified, summary, expected, identity)]
        _need(classified.evidence_status == "valid" and not reasons and classified.seed == seed,
              "CPM trial native/identity checks failed: " + repr(reasons))
        seconds = summary.get("sim_seconds")
        _need(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 90 - 1e-8
              and summary.get("full_duration_evaluation") is True,
              "New CPM trials must retain the complete 90-second evaluation")
        loaded, staged_status, reasons, current_rule = _staged_projection(job, source, 30, True, inputs)
        _need(staged_status == "verified" and not reasons and isinstance(loaded, dict),
              "CPM trial strict staged evidence did not verify: " + repr(reasons))
        if rule is None:
            rule = current_rule
        _need(rule == current_rule, "CPM stage source identities changed within the ten trials")
        staged = loaded["staged_result"]
        sidecar, _ = _bind(job, source / "staged_evaluation.json", inputs, "cpm_n10_raw_stage_authority")
        _bind_stage_projection(staged, sidecar, summary.get("staged_evaluation"))
        witness = qa_by_seed[seed]
        _need(witness.get("source") == str(source) and witness.get("summary_sha256") == summary_row["sha256"]
              and witness.get("native_success") is classified.native_success
              and witness.get("stages") == staged.get("stage_flags"),
              "Independent CPM QA trial differs from strict admissions")
        video = _safe_source(job, source / "review.mp4", "cpm_n10_recorded_video")
        _need(video.stat().st_size > 0 and summary.get("video", {}).get("completed") is True,
              "New CPM trial has no completed recorded video")
        identifier = "cpm-n10-" + hashlib.sha256((job.name + ":" + str(seed) + ":" + summary_row["sha256"]).encode()).hexdigest()[:24]
        trial = {"id": identifier, "seed": seed, "valid": True, "status": classified.status,
                 "native_success": classified.native_success, "recorded_native_success": classified.native_success,
                 "publication_status": "eligible", "staged_evidence_status": "valid",
                 "staged_evidence_basis": "tactile_manifest_inventory_v1", "stages": _stages(staged),
                 "complete_stage_success": staged.get("complete_success"),
                 "continuous_metrics": copy.deepcopy(staged.get("continuous_metrics", {})),
                 "detail_url": identifier + "/report.html", "video_url": None,
                 "summary_sha256": summary_row["sha256"], "sim_seconds": seconds,
                 "notes": [_SCOPE_NOTE], "validity_basis": "explicit_summary_physics_valid",
                 "source_trial_id": identifier}
        trials.append(trial)
        native_sources[identifier] = str(source)
    source_group = {"id": "cpm_best_" + manifest_row["sha256"][:16], "task": "poker-draw",
                    "title": "卡片抽取与举起", "checkpoint_sha256": CPM_SHA,
                    "mode": "diagnostic", "batch_id": job.name,
                    "protocol": {**protocol, "controller": identity["controller"],
                                 "rule_sha256": rule["full_rule_identity_sha256"]}}
    group = _base_group(source_group, origin="additive_" + job.name, model="pi05+cpm",
                        checkpoint=CHECKPOINT_LABEL, horizon=30, execute_steps=5)
    group.update(trials=trials, initial_state_randomization=True, trial_scope=plan["trial_scope"],
                 cpm_checkpoint_sha256=CPM_SHA, cpm_sha256=CPM_SHA, deployment_manifest_sha256=manifest_row["sha256"],
                 deployment_id=manifest["deployment_id"], action_start_index=0, tactile_sent_to_model=True,
                 guidance=identity["guidance"], rule_identity=rule, warnings=[_SCOPE_NOTE])
    group.update(_stats(trials))
    group["metrics"] = _metric_means(trials)
    _need(group["native_success_count"] == qa.get("native_success_count"), "Independent CPM QA native count differs")
    _need({row["key"]: row["reached"] for row in group["stages"]} == qa.get("stage_counts"),
          "Independent CPM QA stage counts differ")
    return group, native_sources, list(inputs.values()), qa_row


def _assert_old_preserved(original, catalog):
    old = {group["id"]: group for group in original["groups"]}
    current = {group["id"]: group for group in catalog["groups"]}
    _need(len(current) == len(catalog["groups"]) and all(current.get(key) == value for key, value in old.items()),
          "An existing group, score, stage admission, or detail identity changed")
    for key in ("schema", "title", "historical_summary", "links", "duplicates", "tasks"):
        _need(catalog.get(key) == original.get(key), "Existing catalog metadata changed: " + key)
    added = [group for group in catalog["groups"] if group["id"] not in old]
    _need(len(added) == 1 and len(added[0]["trials"]) == 10, "Expected one new CPM group with ten trials")
    group = added[0]
    _need(group.get("model") == "pi05+cpm" and group.get("task") == "poker-draw"
          and group.get("checkpoint") == CHECKPOINT_LABEL and group.get("checkpoint_sha256") == CPM_SHA,
          "New CPM group lost its separate weight-qualified checkpoint identity")
    old_ids = {trial["id"] for row in original["groups"] for trial in row["trials"]}
    new_ids = [trial["id"] for trial in group["trials"]]
    _need(len(set(new_ids)) == 10 and not old_ids.intersection(new_ids)
          and all(_ID.fullmatch(value) for value in new_ids), "New CPM member identities conflict or are unsafe")
    _need(sorted(trial["seed"] for trial in group["trials"]) == list(range(10)), "New CPM seeds differ from 0..9")
    return group


def _cpm_media_views(summary, media, evaluation_root):
    """Verify spatial crops against this H30 deployment's exact renderer bytes."""
    from evaluation_workbench.experts import _MODEL_LAYOUT, _MODEL_VIEWS
    from .tactile_experiment_export import _content_identity

    video = summary.get("video", {})
    expected = {"schema": "kaihand-policy-evaluation-review-v3", "layout": _MODEL_LAYOUT,
                "output_size": [1920, 1080], "review_render_size": [640, 480],
                "model_input_cameras_displayed": ["head", "right_wrist"],
                "review_only_cameras_displayed": ["global"], "right_wrist_camera_role": "model_input",
                "completed": True}
    _need(summary.get("model_family") == "pi05+cpm" and summary.get("prediction_horizon") == 30
          and summary.get("execute_steps") == 5 and summary.get("checkpoint_sha256") == CPM_SHA,
          "CPM camera crop has another model identity")
    _need(all(video.get(key) == value and type(video.get(key)) is type(value) for key, value in expected.items())
          and _sha(media.get("sha256")), "CPM recorded camera layout/model roles do not match reviewed geometry")
    manifest_path = _safe_source(evaluation_root, Path(summary["deployment_manifest"]), "CPM crop deployment")
    manifest, manifest_row = _read(manifest_path)
    _need(manifest_row["sha256"] == summary.get("deployment_manifest_sha256")
          and manifest.get("deployment_id") == summary.get("deployment_id") == _content_identity(manifest)
          and manifest.get("checkpoint_sha256") == CPM_SHA and manifest.get("prediction_horizon") == 30,
          "CPM camera crop manifest identity differs")
    inventory = manifest.get("files", {})
    renderers = [(path, value) for path, value in inventory.items()
                 if isinstance(path, str) and path.endswith(_RENDERER_SUFFIX)]
    _need(len(renderers) == 1 and renderers[0][1] == CPM_RENDERER_SHA, "CPM camera renderer was not reviewed")
    renderer = Path(renderers[0][0])
    _need(renderer.is_absolute() and not renderer.is_symlink(), "CPM camera renderer path is unsafe")
    renderer_row = digest(renderer)
    _need(renderer_row["sha256"] == CPM_RENDERER_SHA, "CPM camera renderer bytes changed")
    return {"status": "verified", "sha256": media["sha256"], "source_width": 1920, "source_height": 1080,
            "views": copy.deepcopy(_MODEL_VIEWS), "renderer_sha256": CPM_RENDERER_SHA,
            "deployment_manifest_sha256": manifest_row["sha256"],
            "reason": "本 H30 CPM 部署已核验同一三相机渲染器几何及模型输入角色；腕部裁去标题覆盖条。仅空间裁剪，不声明额外时间对齐。",
            "provenance": [{**manifest_row, "role": "cpm_model_crop_deployment_manifest"},
                           {**renderer_row, "role": "cpm_model_crop_audited_renderer_bytes"}]}


def _export_one(arguments):
    from evaluation_workbench.suite import _page
    from .rich_detail_data import build_rich_case
    from .rich_reference_data import enrich_case

    output, evaluation_root, group, trial, source = arguments
    output, prefix = Path(output), Path(trial["id"])
    artifacts, inputs = [], {}
    summary = output / prefix / "summary.json"
    _copy(output, prefix / "summary.json", Path(source) / "summary.json", artifacts, inputs,
          expected=trial["summary_sha256"])
    case = build_rich_case(source, group, trial, published_summary_path=summary,
                           expected_summary_sha256=trial["summary_sha256"])
    _merge_inputs(inputs, case["provenance"])
    for item in case["copies"]:
        _copy(output, prefix / item["url"], item["source"], artifacts, inputs, expected=item["sha256"])
    if case["config"].get("diagnostic_plots", {}).get("status") != "available":
        _model_plots(case, group, trial, output, artifacts)
    config = case["config"]
    config["unified_identity"] = _identity(group, trial, True)
    config["unified_navigation"] = {"catalog_url": "../catalog.json", "overview_url": "../index.html"}
    config["suite_url"] = "../index.html"
    config["trial_scope"] = {"initial_state_randomization": True, "kind": group["trial_scope"], "note": _SCOPE_NOTE}
    recorded_summary, _ = _read(summary)
    config["media_views"] = _cpm_media_views(recorded_summary, case["visualization"]["media"], evaluation_root)
    _merge_inputs(inputs, config["media_views"]["provenance"])
    annotated = enrich_case(case["report"], case["visualization"], config, group, trial,
                           reports_root=output.parent, evaluation_root=Path(evaluation_root))
    _merge_inputs(inputs, annotated["provenance"])
    case.update(annotated)
    refs = case["config"]["goal_references"]
    _need(refs.get("status") == "verified", "New CPM detail lacks source-bound task reference values")
    for name, key in (("report.json", "report"), ("visualization.json", "visualization"),
                      ("tactile_loads.json", "tactile_loads"), ("workbench.json", "config")):
        _write(output, prefix / name, encoded(case[key]), artifacts)
    _write(output, prefix / "goal_references.json", encoded(refs), artifacts)
    _write(output, prefix / "report.html", _page(case["report"], case["visualization"], case["tactile_loads"], case["config"]).encode(), artifacts)
    _write(output, prefix / "audit.html", _audit_page(case["report"], group, trial).encode(), artifacts)
    video = case["visualization"].get("media", {}).get("url")
    _need(isinstance(video, str) and video, "New CPM detail lacks recorded video")
    return {"id": trial["id"], "video_url": (prefix / video).as_posix(),
            "availability": case["availability"], "coverage": case["coverage"],
            "artifacts": artifacts, "inputs": list(inputs.values())}


def export_cpm_additive_workbench(baseline, output, *, job_root, evaluation_root, qa_receipt, workers=2):
    """Create 605-member snapshot only; never publish the common landing page."""
    from .unified_workbench_page import render_page

    baseline, output = Path(baseline).resolve(strict=True), Path(output).absolute()
    evaluation_root = Path(evaluation_root).resolve(strict=True)
    _need(not output.exists() and not output.is_symlink() and output.parent.resolve(strict=True) == baseline.parent,
          "CPM additive output must be a fresh canonical sibling snapshot")
    _need(type(workers) is int and 1 <= workers <= 8, "Choose 1..8 detail workers")
    verify_reference_workbench(baseline, verify_sources=False)
    store = _Artifacts(baseline.parent, output)
    original = store.document(baseline / "catalog.json", "immutable_595_baseline_catalog")
    _validate_catalog(original)
    _need(original["counts"].get("members") == 595 and original["counts"].get("groups") == 47,
          "Only the accepted 595-member/47-group baseline may be extended")
    group, native_sources, sources, qa_row = _selected_job(job_root, evaluation_root, qa_receipt)
    catalog = copy.deepcopy(original)
    catalog["groups"].append(group)
    catalog["counts"] = _counts(catalog["groups"])
    _assert_old_preserved(original, catalog)
    inventory = store.inventory(baseline)
    previous = (output.parent / "index.html").read_bytes() if (output.parent / "index.html").exists() else None
    output.mkdir()
    artifacts, inputs, preserved = [], dict(store.inputs), []
    _merge_inputs(inputs, sources)
    for relative, row in inventory.items():
        if relative not in _REPLACED:
            _copy(output, Path(relative), baseline / relative, artifacts, inputs, expected=row["sha256"])
            preserved.append({"path": relative, "size": row["size"], "sha256": row["sha256"]})
    arguments = [(str(output), str(evaluation_root), group, trial, native_sources[trial["id"]]) for trial in group["trials"]]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_export_one, arguments, chunksize=1), 1):
            artifacts.extend(result.pop("artifacts"))
            _merge_inputs(inputs, result.pop("inputs"))
            results[result["id"]] = result
            print(f"cpm-rich-details {index}/10", flush=True)
    for trial in group["trials"]:
        trial.update(video_url=results[trial["id"]]["video_url"], rich_detail=results[trial["id"]]["availability"])
    coverage = copy.deepcopy(store.document(baseline / "reference_coverage.json", "immutable_reference_coverage"))
    _merge_inputs(inputs, list(store.inputs.values()))
    coverage["schema"] = SCHEMA
    coverage["trials"].extend(results[trial["id"]]["coverage"] for trial in group["trials"])
    catalog["generated_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["rich_details"]["count"] = 605
    catalog["additive_evaluation"] = {"schema": SCHEMA, "baseline": baseline.name,
        "added_members": 10, "added_groups": 1, "job": str(Path(job_root).resolve(strict=True)),
        "initial_state_randomization": True, "trial_scope": group["trial_scope"], "note": _SCOPE_NOTE,
        "checkpoint_label": CHECKPOINT_LABEL, "checkpoint_sha256": CPM_SHA,
        "deployment_manifest_sha256": group["deployment_manifest_sha256"]}
    _assert_old_preserved(original, catalog)
    _validate_catalog(catalog)
    for relative, value in (("catalog.json", catalog), ("reference_coverage.json", coverage),
                            ("preserved_artifacts.json", {"schema": SCHEMA, "files": preserved})):
        _write(output, Path(relative), encoded(value), artifacts)
    _write(output, Path("index.html"), render_page(catalog).encode(), artifacts)
    if previous is not None:
        _write(output, Path("previous_root_index.html"), previous, artifacts)
    for source in (Path(__file__), PACKAGE.parent / "export_cpm_additive_evaluation_workbench.py"):
        _merge_inputs(inputs, [{**digest(source), "role": "cpm_additive_export_implementation"}])
    verification = {"schema": SCHEMA, "baseline_catalog": digest(baseline / "catalog.json"),
        "baseline_manifest": digest(baseline / "artifact_manifest.json"),
        "evaluation_root": str(evaluation_root), "job_root": str(Path(job_root).resolve(strict=True)),
        "independent_qa": qa_row, "added_trial_ids": [trial["id"] for trial in group["trials"]],
        "added_group_id": group["id"], "native_outcomes_rescored": False, "staged_rules_reexecuted": False,
        "source_writes": 0, "existing_snapshot_writes": 0, "simulation_runs_started": 0,
        "model_weights_loaded": 0, "deployment_inventory_files_sha256_verified": True,
        "source_files": sorted(inputs.values(), key=lambda row: row["path"])}
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SCHEMA, "files": artifacts}), [])
    return verify_cpm_additive_workbench(output)


def verify_cpm_additive_workbench(output, *, verify_sources=True):
    """Check exact old bytes, new identities, archived scores and displayed data."""
    from evaluation_v1.protocol import content_hash
    from .rich_reference_data import _for_series

    output = Path(output).resolve(strict=True)
    manifest, _ = _read(output / "artifact_manifest.json")
    verification, _ = _read(output / "verification.json")
    _need(manifest.get("schema") == verification.get("schema") == SCHEMA, "Unsupported CPM additive snapshot schema")
    paths = {}
    for row in manifest.get("files", []):
        relative = Path(row["path"])
        path = output / relative
        _need(not relative.is_absolute() and ".." not in relative.parts and row["path"] not in paths
              and not path.is_symlink() and path.resolve(strict=True).is_relative_to(output), "Unsafe CPM snapshot artifact")
        verify_row(row, path)
        paths[row["path"]] = row
    for name in ("catalog.json", "verification.json", "index.html", "reference_coverage.json", "preserved_artifacts.json"):
        _need(name in paths, "Missing CPM snapshot artifact: " + name)
    baseline = Path(verification["baseline_catalog"]["path"]).parent
    _need(baseline.parent == output.parent and baseline != output and not baseline.is_symlink(), "Unsafe CPM baseline")
    verify_row(verification["baseline_catalog"])
    verify_row(verification["baseline_manifest"])
    original, _ = _read(baseline / "catalog.json")
    catalog, _ = _read(output / "catalog.json")
    _validate_catalog(original)
    _validate_catalog(catalog)
    _need(original["counts"].get("members") == 595 and original["counts"].get("groups") == 47
          and catalog["counts"].get("members") == 605 and catalog["counts"].get("groups") == 48,
          "CPM snapshot must be precisely 595 old + 10 new members and 48 groups")
    added = _assert_old_preserved(original, catalog)
    _need(verification.get("added_group_id") == added["id"]
          and verification.get("added_trial_ids") == [trial["id"] for trial in added["trials"]], "CPM additional identity inventory changed")
    baseline_inventory, _ = _read(baseline / "artifact_manifest.json")
    expected_preserved = {row["path"]: row for row in baseline_inventory["files"] if row["path"] not in _REPLACED}
    preserved, _ = _read(output / "preserved_artifacts.json")
    recorded_preserved = {row["path"]: row for row in preserved.get("files", [])}
    _need(recorded_preserved == expected_preserved and len(recorded_preserved) == len(preserved.get("files", [])),
          "Preserved-byte inventory differs from the baseline")
    for relative, row in expected_preserved.items():
        _need(paths.get(relative) == row, "Existing episode or UI asset bytes changed: " + relative)
    coverage, _ = _read(output / "reference_coverage.json")
    baseline_coverage, _ = _read(baseline / "reference_coverage.json")
    _need(coverage.get("trials", [])[:len(baseline_coverage["trials"])] == baseline_coverage["trials"], "Old reference coverage changed")
    covered = [row["trial_id"] for row in coverage.get("trials", [])]
    ids = [trial["id"] for group in catalog["groups"] for trial in group["trials"]]
    _need(len(covered) == len(set(covered)) and set(covered) == set(ids), "Reference coverage must exactly match all 605 members")
    for trial in added["trials"]:
        prefix = Path(trial["id"])
        _need(all((prefix / name).as_posix() in paths for name in _REQUIRED), "New CPM detail lacks rich/reference payloads")
        _need(trial["detail_url"] == (prefix / "report.html").as_posix() and trial.get("video_url") in paths,
              "New CPM detail/video link is not inventoried")
        config, _ = _read(output / prefix / "workbench.json")
        refs, _ = _read(output / prefix / "goal_references.json")
        report, _ = _read(output / prefix / "report.json")
        summary, summary_row = _read(output / prefix / "summary.json")
        visual, _ = _read(output / prefix / "visualization.json")
        _need(summary_row["sha256"] == trial["summary_sha256"], "New CPM summary bytes differ from admission")
        _need(config.get("unified_identity") == _identity(added, trial, True), "New CPM displayed identity changed")
        _need(refs == config.get("goal_references") and refs.get("status") == "verified", "New CPM reference download differs")
        _need(report.get("visualization", {}).get("sha256") == content_hash(visual), "New CPM visual samples are not report-bound")
        for series in visual.get("task_series", []):
            _need((series.get("goal_references") or []) == _for_series(refs.get("references", []), series), "CPM chart references differ")
        _need(report.get("outcome", {}).get("recorded_success") is trial["native_success"]
              and summary.get("task_success") is trial["native_success"], "New CPM displayed native result changed")
        _need(_same_json(report.get("task_specific", {}).get("partial_state"), _expected_terminal_state(summary)),
              "New CPM terminal measurements changed")
        displayed = (config.get("staged_evaluation") or {}).get("staged_result", {})
        recorded = summary.get("staged_evaluation", {})
        _need(displayed.get("stage_flags") == {stage["key"]: stage["reached"] for stage in trial["stages"]}
              and displayed.get("complete_success") is trial["complete_stage_success"]
              and all(displayed.get(key) == recorded.get(key) for key in ("first_times", "thresholds")), "New CPM displayed stage values changed")
        _need(summary.get("checkpoint_sha256") == CPM_SHA
              and summary.get("deployment_manifest_sha256") == added.get("deployment_manifest_sha256"), "New CPM detail has another deployment identity")
        _need(config.get("media_views", {}).get("sha256") == visual.get("media", {}).get("sha256")
              and config.get("media_views", {}).get("status") == "verified"
              and {row.get("id") for row in config["media_views"].get("views", [])} == {"head", "wrist", "global"},
              "New CPM model-input camera views are not bound to its video")
    if verify_sources:
        evaluation_root = Path(verification["evaluation_root"]).resolve(strict=True)
        roots = (output.parent, evaluation_root, PACKAGE.parent.resolve(strict=True))
        job_manifest, _ = _read(Path(verification["job_root"]) / "deployments/cpm_best/deployment_manifest.json")
        manifest_inventory = job_manifest.get("files", {})
        for row in verification.get("source_files", []):
            path = Path(row["path"])
            _need(path.is_absolute() and not path.is_symlink(), "CPM source must be an absolute regular leaf")
            camera_renderer = (row.get("role") == "cpm_model_crop_audited_renderer_bytes"
                               and path.as_posix().endswith(_RENDERER_SUFFIX) and row.get("sha256") == CPM_RENDERER_SHA)
            # The strict tactile importer explicitly hashes every frozen
            # inventory file, including the base checkpoint and tokenizer.
            # Permit only that exact manifest-bound path+SHA, never a broad
            # external directory or model deserialization.
            inventory_witness = (row.get("role") == "tactile_inventory_file"
                                 and manifest_inventory.get(str(path)) == row.get("sha256"))
            _need(any(path.resolve(strict=True).is_relative_to(root) for root in roots) or camera_renderer
                  or inventory_witness or _reviewed_source(row, path),
                  "Unadmitted CPM display source path: " + str(path))
            verify_row(row)
        actual, _, _, qa_row = _selected_job(verification["job_root"], evaluation_root, verification["independent_qa"]["path"])
        expected = copy.deepcopy(added)
        for trial in expected["trials"]:
            trial.pop("rich_detail", None)
            trial["video_url"] = None
        _need(actual == expected and qa_row == verification["independent_qa"], "New CPM scores no longer match strict selected-job admissions")
        for trial in added["trials"]:
            prefix = output / trial["id"]
            summary, _ = _read(prefix / "summary.json")
            visual, _ = _read(prefix / "visualization.json")
            config, _ = _read(prefix / "workbench.json")
            _need(config.get("media_views") == _cpm_media_views(summary, visual["media"], evaluation_root),
                  "New CPM camera crop differs from source-bound reviewed renderer")
    _need(not (output / "portal_publication.pending.json").exists(), "Incomplete CPM landing-page publication")
    published = None
    if (output / "portal_publication.json").exists():
        publication, _ = _read(output / "portal_publication.json")
        _need(publication.get("state") == "committed" and publication.get("root_index") == str(output.parent / "index.html"), "Invalid CPM publication audit")
        published = digest(output.parent / "index.html")["sha256"] == publication.get("new_sha256")
    return {"status": "passed", **catalog["counts"], "artifacts": len(paths), "added_members": 10, "added_groups": 1,
            "checkpoint": CHECKPOINT_LABEL, "native_success_count": added["native_success_count"], "native_known_count": added["native_known_count"],
            "stages": added["stages"], "existing_score_admissions_preserved": True, "existing_episode_bytes_preserved": True,
            "simulation_runs_started": 0, "model_weights_loaded": 0,
            "deployment_inventory_files_sha256_verified": verify_sources, "root_portal_matches_this_snapshot": published}


def publish_cpm_additive_workbench(output):
    """Explicit publication only; require successful browser QA for this snapshot."""
    output = Path(output).resolve(strict=True)
    result = verify_cpm_additive_workbench(output)
    browser, _ = _read(output / "browser_qa/cpm_additive_browser_verification.json")
    _need(browser.get("status") == "passed" and browser.get("members") == 605
          and browser.get("new_dom_trials") == 10 and browser.get("snapshot_directory") == str(output)
          and browser.get("catalog_sha256") == digest(output / "catalog.json")["sha256"], "CPM publication requires successful snapshot-bound browser QA")
    previous = (output / "previous_root_index.html").read_bytes() if (output / "previous_root_index.html").exists() else None
    relative = quote(output.name, safe="._-") + "/index.html"
    landing = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={relative}">'
               f'<title>KaiTactileSim 官方评估平台</title><p><a href="{relative}">进入 KaiTactileSim 统一评估工作台</a></p></html>').encode()
    _publish_portal(output, output.parent / "index.html", previous, landing)
    result["root_portal_matches_this_snapshot"] = True
    return result
