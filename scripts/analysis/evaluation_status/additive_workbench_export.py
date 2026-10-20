"""Append one explicitly selected, completed bulb-inplace batch to a frozen site.

The existing 595 admissions and episode artifacts are immutable inputs. Only
the requested five H50/K50 checkpoints, each with seeds 0..4, may be admitted.
This offline publisher never runs a policy, task rule, or simulator.
"""
from __future__ import annotations

import copy
import math
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .data import _Builder, _load_new_job, _read_json
from .export import PACKAGE, _copy, _publish_portal, _write, digest, encoded, verify_row
from .reference_workbench_export import verify_reference_workbench
from .rich_workbench_export import (
    _audit_page,
    _identity,
    _merge_inputs,
    _model_plots,
    score_projection,
)
from .unified_workbench_export import (
    _Artifacts,
    _base_group,
    _compare_stats,
    _counts,
    _metric_means,
    _read,
    _stages,
    _stats,
    _validate_catalog,
)

SCHEMA = "kaihand-additive-rich-workbench-snapshot-v1"
JOB_SCHEMA = "kaihand-pi05-bulb-inplace-stage-batch-v1"
TASK = "bulb-screw-inplace"
STEPS = (15000, 18000, 21000, 24000, 24999)
CHECKPOINT_ROOT = Path(
    "/nas/chenxianchi/datasets/sim/bulb-screw-inplace/pi05/1018_200/checkpoints/"
    "pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1/bulb-screw-inplace_bs128_25k"
)
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_REPLACED = {
    "catalog.json", "index.html", "verification.json", "artifact_manifest.json",
    "previous_root_index.html", "reference_coverage.json",
}
_REQUIRED = {
    "summary.json", "report.json", "visualization.json", "tactile_loads.json",
    "workbench.json", "goal_references.json", "report.html", "audit.html",
}
_SCOPE_NOTE = (
    "固定初态的 5 次重复实验；seed 不代表随机物体初态，不能据此声称初态泛化。"
    "相对实际初态的净旋入角度与绝对到座角度分开显示，不把 100° 直接作为相对旋转终点。"
)


def _selected_job(job_root, evaluation_root):
    """Reuse native/staged strict admission, never scan unrelated jobs."""
    evaluation_root = Path(evaluation_root).resolve(strict=True)
    requested = Path(job_root).absolute()
    if requested.is_symlink():
        raise ValueError("Selected job must not be a symlink")
    job = requested.resolve(strict=True)
    if job.parent != evaluation_root / "_jobs":
        raise ValueError("Selected job must be a direct evaluation-root/_jobs child")
    batch, batch_row = _read_json(job / "batch.json")
    status, status_row = _read_json(job / "status.json")
    if batch.get("schema") != JOB_SCHEMA or status.get("schema") != JOB_SCHEMA:
        raise ValueError("Unsupported dedicated bulb-inplace batch schema")
    if status.get("state", status.get("status")) != "complete":
        raise ValueError("Selected job is not complete")
    if batch.get("initial_state_randomization") is not False or batch.get("trial_scope") != "fixed_initial_state_repeats":
        raise ValueError("Batch must explicitly describe fixed-initial-state repetitions")
    settings = batch.get("settings", {})
    if (settings.get("prediction_horizon"), settings.get("execute_steps"), settings.get("num_trials")) != (50, 50, 5):
        raise ValueError("Only requested H50/K50, five-trial groups may be appended")
    groups, states = batch.get("groups"), status.get("groups")
    if not isinstance(groups, list) or len(groups) != len(STEPS) or not isinstance(states, list) or len(states) != len(STEPS):
        raise ValueError("Expected exactly five checkpoint groups and accepted-attempt states")
    observed = []
    for group, state in zip(groups, states, strict=True):
        if not isinstance(group, dict) or not isinstance(state, dict):
            raise ValueError("Malformed group or accepted-attempt state")
        step = group.get("step")
        if type(step) is not int or step not in STEPS or group.get("task") != TASK:
            raise ValueError("Batch contains an unrequested task/checkpoint")
        if group.get("checkpoint") != str(CHECKPOINT_ROOT / str(step)):
            raise ValueError("Checkpoint path differs from the explicitly requested training run")
        if group.get("group_dir") != f"groups/{TASK}/step{step}" or state.get("state") != "complete":
            raise ValueError("Batch group path or accepted state is inconsistent")
        if type(state.get("attempt")) is not int or state["attempt"] <= 0:
            raise ValueError("Completed group has no explicit positive accepted attempt")
        observed.append(step)
    if set(observed) != set(STEPS) or len(set(observed)) != len(STEPS):
        raise ValueError("Checkpoint groups are duplicated or missing")
    builder = _Builder(evaluation_root)
    builder.provenance(batch_row, "additive_completed_batch_plan")
    builder.provenance(status_row, "additive_completed_status_attempt_selection")
    # This internal loader consumes groups/settings and preserves its strict
    # native/staged checks; its public old-schema discovery is not modified.
    _load_new_job(builder, job, batch, status)
    if builder.excluded or len(builder.groups) != len(STEPS):
        raise ValueError("Strict admission rejected or split a requested group: " + repr(builder.excluded))
    for group in builder.groups:
        trials = group["trials"]
        if len(trials) != 5 or {trial["seed"] for trial in trials} != set(range(5)):
            raise ValueError("Each checkpoint must contain exactly seeds 0..4")
        for trial in trials:
            if trial.get("valid") is not True or not isinstance(trial.get("staged_result"), dict):
                raise ValueError("Every new trial needs verified native and staged evidence")
            summary, _ = _read_json(Path(trial["summary_path"]))
            stage = summary.get("staged_evaluation")
            if (summary.get("full_duration_evaluation") is not True
                    or not isinstance(stage, dict) or stage.get("full_duration_evaluation") is not True
                    or type(summary.get("sim_seconds")) not in (int, float)
                    or not math.isfinite(summary["sim_seconds"]) or summary["sim_seconds"] < 90 - 1e-8):
                raise ValueError("New trials must preserve a complete recorded 90-second evaluation")
            source = Path(trial["source"])
            if source.is_symlink() or not source.resolve(strict=True).is_relative_to(job):
                raise ValueError("Accepted trial escapes its selected job")
    return builder.groups, builder.provenance_rows, batch


def _unified_groups(source_groups, job_name):
    """Project already verified native admissions without rerunning rules."""
    result = []
    for source in source_groups:
        group = _base_group(source, origin="additive_" + job_name, model="pi05",
                            checkpoint=source["checkpoint"], horizon=50, execute_steps=50)
        group.update(initial_state_randomization=False, trial_scope="fixed_initial_state_repeats")
        group["warnings"] = list(dict.fromkeys([*group["warnings"], _SCOPE_NOTE]))
        for trial in source["trials"]:
            staged = trial["staged_result"]
            group["trials"].append({
                "id": trial["id"], "seed": trial["seed"], "valid": True,
                "status": trial["status"], "native_success": trial["native_success"],
                "publication_status": "eligible", "staged_evidence_status": "valid",
                "staged_evidence_basis": "strict_completed_job_archived_rule_inventory",
                "stages": _stages(staged), "complete_stage_success": staged.get("complete_success"),
                "continuous_metrics": copy.deepcopy(staged.get("continuous_metrics", {})),
                "detail_url": trial["id"] + "/report.html", "video_url": None,
                "notes": list(dict.fromkeys([*trial.get("warnings", []), _SCOPE_NOTE])),
                "summary_sha256": trial["summary_sha256"], "sim_seconds": trial["sim_seconds"],
                "recorded_native_success": trial["native_success"], "validity_basis": trial.get("validity_basis"),
                "source_trial_id": trial["id"],
            })
        group.update(_stats(group["trials"]))
        _compare_stats(group, source)
        group["recorded_statistics"] = {key: copy.deepcopy(source[key]) for key in (
            "valid_trials", "native_success_count", "native_known_count", "stages",
            "complete_stage_success_count", "complete_stage_known_count")}
        group["metrics"] = _metric_means(group["trials"])
        result.append(group)
    return result


def _assert_old_preserved(original, catalog):
    old_groups = {group["id"]: group for group in original["groups"]}
    current = {group["id"]: group for group in catalog["groups"]}
    if len(current) != len(catalog["groups"]) or any(current.get(key) != group for key, group in old_groups.items()):
        raise ValueError("An existing group, member, score or evidence admission changed")
    for key in ("schema", "title", "historical_summary", "links", "duplicates"):
        if catalog.get(key) != original.get(key):
            raise ValueError("Existing catalog metadata changed: " + key)
    old_tasks = {row["task"]: row for row in original["tasks"]}
    tasks = {row["task"]: row for row in catalog["tasks"]}
    if any(tasks.get(key) != row for key, row in old_tasks.items()):
        raise ValueError("Existing task descriptions changed")
    old_trials = {trial["id"] for group in original["groups"] for trial in group["trials"]}
    added = [trial for group in catalog["groups"] if group["id"] not in old_groups for trial in group["trials"]]
    if len(added) != 25 or len({trial["id"] for trial in added}) != 25 or old_trials & {trial["id"] for trial in added}:
        raise ValueError("Additional membership is not exactly 25 distinct new trials")


def _new_projection(groups):
    return score_projection({"groups": groups})["groups"]


def _reviewed_source(row, path):
    """Allow only explicit audited Python witnesses outside report/data roots."""
    actual = path.resolve(strict=True)
    if actual.suffix != ".py" or not actual.is_relative_to(Path("/cpfs_infra/user/chenxianchi")):
        return False
    if row.get("role") == "goal_reference_reviewed_source":
        return "/src/kaihand_tactile_env/" in actual.as_posix()
    if row.get("role") == "model_crop_audited_renderer_bytes":
        from evaluation_workbench.experts import (
            _MODEL_RENDERER_AUDIT,
            _MODEL_RENDERER_PATH,
        )

        return actual.as_posix().endswith("/" + _MODEL_RENDERER_PATH) and row.get("sha256") in _MODEL_RENDERER_AUDIT
    if row.get("role") == "audited_camera_pose_clock_source":
        from evaluation_workbench.data import _POSE_CLOCK_SOURCE_SETS

        return any(actual.as_posix().endswith("/" + relative) and row.get("sha256") == expected
                   for audit in _POSE_CLOCK_SOURCE_SETS for relative, expected in audit.items())
    return False


def _relative_reference(case, published_summary, group, trial):
    """Shift an authenticated absolute goal by this trial's recorded reset angle.

    The reset angle is measured after initial physics settling, not the nominal
    reset config. Its relation to the two terminal angle measurements is checked
    independently before any relative chart target is emitted.
    """
    from evaluation_v1.protocol import content_hash

    summary, row = _read(published_summary)
    if row["sha256"] != trial["summary_sha256"]:
        raise ValueError("Relative reference summary differs from admitted bytes")
    evaluation = summary.get("evaluation", {})
    initial = evaluation.get("threaded_initialization", {})
    turns = initial.get("initial_clockwise_turns")
    if (group["task"] != TASK or initial.get("schema") != "bulb_screw_inplace_threaded_reset_v1"
            or initial.get("position_randomization_applied") is not False
            or type(turns) not in (int, float) or not math.isfinite(turns)):
        raise ValueError("Per-trial net-angle reference lacks an explicit measured fixed reset angle")
    state = case["report"].get("task_specific", {}).get("partial_state", {})
    absolute, relative = state.get("clockwise_turns"), state.get("rotation_from_initial_turns")
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (absolute, relative))
            or not math.isclose(absolute - relative, turns, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("Recorded reset angle contradicts terminal absolute-minus-relative measurements")
    bundle = case["config"]["goal_references"]
    goals = [item for item in bundle.get("references", []) if item.get("key") == "clockwise_angle_deg" and item.get("unit") == "deg"]
    if bundle.get("status") != "verified" or len(goals) != 1:
        raise ValueError("Relative reference requires one verified absolute mechanical angle goal")
    initial_deg = turns * 360.0
    reference = copy.deepcopy(goals[0])
    reference.update(key="rotation_from_initial_deg", label="相对本轨迹实际初态的名义到座净旋入角度",
                     note="逐条以原始记录的实际初始角度扣除绝对到座角度；不是累计旋转路程，达到参考线也不等于完成拧紧或松手稳定。")
    for key in ("target", "lower", "upper"):
        if key in reference:
            reference[key] -= initial_deg
    reference["binding"] = {
        "summary": row, "initial_angle_source": "evaluation.threaded_initialization.initial_clockwise_turns",
        "initial_angle_turns": turns, "initial_angle_deg": initial_deg,
        "absolute_target_deg": goals[0]["target"], "operation": "absolute_goal_minus_actual_initial_angle",
        "fixed_shared_relative_target": False,
    }
    bundle["references"].append(reference)
    bundle.setdefault("warnings", []).append("本次新增逐条净角度参考，已扣除本轨迹实际初始角度；原提示指不能给所有轨迹套用同一个固定净旋转终点。")
    bundle.setdefault("provenance", []).append({**row, "role": "additive_actual_initial_angle_summary"})
    attached = False
    for series in case["visualization"].get("task_series", []):
        if (series.get("key"), series.get("unit")) == ("rotation_from_initial_deg", "deg"):
            series.setdefault("goal_references", []).append(copy.deepcopy(reference))
            attached = True
    if not attached:
        raise ValueError("New rich detail lacks the relative net-angle task curve")
    case["coverage"]["series_with_reference"] = list(dict.fromkeys([
        *case["coverage"]["series_with_reference"], "rotation_from_initial_deg"]))
    case["coverage"]["reference_count"] = len(bundle["references"])
    case["coverage"]["relative_angle_reference"] = copy.deepcopy(reference["binding"])
    case["report"]["goal_reference_extension"]["relative_angle_reference"] = copy.deepcopy(reference["binding"])
    case["report"]["visualization"]["sha256"] = content_hash(case["visualization"])


def _export_one(arguments):
    from evaluation_workbench.suite import _page

    from .rich_detail_data import build_rich_case
    from .rich_reference_data import enrich_case

    output, evaluation_root, group, trial, native = arguments
    output = Path(output)
    prefix = Path(trial["id"])
    artifacts, inputs = [], {}
    published_summary = output / prefix / "summary.json"
    _copy(output, prefix / "summary.json", Path(native["summary_path"]), artifacts, inputs, expected=trial["summary_sha256"])
    case = build_rich_case(native["source"], group, trial,
                           published_summary_path=published_summary, expected_summary_sha256=trial["summary_sha256"])
    _merge_inputs(inputs, case["provenance"])
    for job in case["copies"]:
        _copy(output, prefix / job["url"], job["source"], artifacts, inputs, expected=job["sha256"])
    if case["config"].get("diagnostic_plots", {}).get("status") != "available":
        _model_plots(case, group, trial, output, artifacts)
    config = case["config"]
    config["unified_identity"] = _identity(group, trial, isinstance(config.get("staged_evaluation"), dict))
    config["unified_navigation"] = {"catalog_url": "../catalog.json", "overview_url": "../index.html"}
    config["suite_url"] = "../index.html"
    config["trial_scope"] = {"initial_state_randomization": False, "kind": "fixed_initial_state_repeats", "note": _SCOPE_NOTE}
    annotated = enrich_case(case["report"], case["visualization"], config, group, trial,
                           reports_root=output.parent, evaluation_root=Path(evaluation_root))
    _merge_inputs(inputs, annotated["provenance"])
    case.update(annotated)
    references = case["config"]["goal_references"]
    if references.get("status") != "verified" or case["coverage"]["terminal_angle"]["status"] != "recorded":
        raise ValueError("New bulb detail lacks source-bound mechanical references or terminal net angle")
    _relative_reference(case, published_summary, group, trial)
    for name, key in (("report.json", "report"), ("visualization.json", "visualization"),
                      ("tactile_loads.json", "tactile_loads"), ("workbench.json", "config")):
        _write(output, prefix / name, encoded(case[key]), artifacts)
    _write(output, prefix / "goal_references.json", encoded(references), artifacts)
    _write(output, prefix / "report.html", _page(case["report"], case["visualization"], case["tactile_loads"], case["config"]).encode(), artifacts)
    _write(output, prefix / "audit.html", _audit_page(case["report"], group, trial).encode(), artifacts)
    media = case["visualization"].get("media", {})
    if not media.get("url"):
        raise ValueError("New admitted detail has no recorded video")
    return {"id": trial["id"], "video_url": (prefix / media["url"]).as_posix(),
            "availability": case["availability"], "coverage": case["coverage"],
            "artifacts": artifacts, "inputs": list(inputs.values())}


def export_additive_workbench(baseline, output, *, job_root, evaluation_root, workers=2):
    """Write a fresh additive snapshot; the shared portal is never changed here."""
    from evaluation_workbench.task_fields import task_presentation

    from .unified_workbench_page import render_page

    baseline = Path(baseline).resolve(strict=True)
    output = Path(output).absolute()
    evaluation_root = Path(evaluation_root).resolve(strict=True)
    if output.exists() or output.is_symlink() or output.parent.resolve(strict=True) != baseline.parent:
        raise ValueError("Additive output must be a fresh sibling report snapshot")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Choose 1..8 export workers")
    verify_reference_workbench(baseline, verify_sources=False)
    store = _Artifacts(baseline.parent, output)
    original = store.document(baseline / "catalog.json", "immutable_baseline_catalog")
    _validate_catalog(original)
    if original["counts"].get("members") != 595 or original["counts"].get("groups") != 47:
        raise ValueError("This requested extension requires the accepted 595-member/47-group baseline")
    source_groups, sources, batch = _selected_job(job_root, evaluation_root)
    additions = _unified_groups(source_groups, Path(job_root).name)
    catalog = copy.deepcopy(original)
    catalog["groups"].extend(additions)
    if TASK not in {row["task"] for row in catalog["tasks"]}:
        presentation = task_presentation(TASK)
        catalog["tasks"].append({"task": TASK, "title": presentation["title"], "note": presentation["task_note"]})
    catalog["counts"] = _counts(catalog["groups"])
    _assert_old_preserved(original, catalog)
    old_ids = {trial["id"] for group in original["groups"] for trial in group["trials"]}
    new_ids = [trial["id"] for group in additions for trial in group["trials"]]
    if any(not _ID.fullmatch(identifier) for identifier in new_ids) or old_ids & set(new_ids):
        raise ValueError("Unsafe or conflicting new trial IDs")
    inventory = store.inventory(baseline)
    previous = (output.parent / "index.html").read_bytes() if (output.parent / "index.html").exists() else None
    output.mkdir()
    artifacts, inputs = [], dict(store.inputs)
    _merge_inputs(inputs, sources)
    preserved = []
    for relative, row in inventory.items():
        if relative in _REPLACED:
            continue
        # Preserve every existing episode artifact and UI asset byte-for-byte.
        _copy(output, Path(relative), baseline / relative, artifacts, inputs, expected=row["sha256"])
        preserved.append({"path": relative, "size": row["size"], "sha256": row["sha256"]})
    native = {trial["id"]: trial for group in source_groups for trial in group["trials"]}
    arguments = [(str(output), str(evaluation_root), group, trial, native[trial["id"]])
                 for group in additions for trial in group["trials"]]
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_export_one, arguments, chunksize=1), 1):
            artifacts.extend(result.pop("artifacts"))
            _merge_inputs(inputs, result.pop("inputs"))
            results[result["id"]] = result
            print(f"additive-rich-details {index}/{len(arguments)}", flush=True)
    for group in additions:
        for trial in group["trials"]:
            item = results[trial["id"]]
            trial.update(video_url=item["video_url"], rich_detail=item["availability"])
    prior_coverage = store.document(baseline / "reference_coverage.json", "immutable_reference_coverage")
    _merge_inputs(inputs, list(store.inputs.values()))
    coverage = copy.deepcopy(prior_coverage)
    coverage["schema"] = SCHEMA
    coverage["trials"].extend(results[identifier]["coverage"] for identifier in new_ids)
    catalog["generated_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["rich_details"]["count"] = catalog["counts"]["members"]
    catalog["additive_evaluation"] = {
        "schema": SCHEMA, "baseline": baseline.name, "added_members": len(new_ids), "added_groups": len(additions),
        "job": str(Path(job_root).resolve(strict=True)), "initial_state_randomization": False,
        "trial_scope": batch["trial_scope"], "note": _SCOPE_NOTE,
    }
    _assert_old_preserved(original, catalog)
    _validate_catalog(catalog)
    for relative, value in (("catalog.json", catalog), ("reference_coverage.json", coverage),
                            ("additional_admissions.json", {"schema": SCHEMA, "groups": _new_projection(additions),
                                                          "native_groups": source_groups}),
                            ("preserved_artifacts.json", {"schema": SCHEMA, "files": preserved})):
        _write(output, Path(relative), encoded(value), artifacts)
    _write(output, Path("index.html"), render_page(catalog).encode(), artifacts)
    if previous is not None:
        _write(output, Path("previous_root_index.html"), previous, artifacts)
    for source in (Path(__file__), PACKAGE.parent / "export_additive_evaluation_workbench.py"):
        _merge_inputs(inputs, [{**digest(source), "role": "additive_export_implementation"}])
    verification = {
        "schema": SCHEMA, "baseline_catalog": digest(baseline / "catalog.json"),
        "baseline_manifest": digest(baseline / "artifact_manifest.json"),
        "evaluation_root": str(evaluation_root), "job_root": str(Path(job_root).resolve(strict=True)),
        "added_trial_ids": new_ids, "added_group_ids": [group["id"] for group in additions],
        "existing_score_admissions_preserved": True, "existing_episode_bytes_preserved": True,
        "native_outcomes_rescored": False, "staged_rules_reexecuted": False,
        "source_writes": 0, "existing_snapshot_writes": 0, "simulation_runs_started": 0, "model_weights_read": 0,
        "source_files": sorted(inputs.values(), key=lambda row: row["path"]),
    }
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SCHEMA, "files": artifacts}), [])
    return verify_additive_workbench(output)


def verify_additive_workbench(output, *, verify_sources=True):
    """Bind new admissions to the exact selected job and old episodes to r5."""
    from evaluation_v1.protocol import content_hash

    from .rich_reference_data import _for_series

    output = Path(output).resolve(strict=True)
    manifest, _ = _read(output / "artifact_manifest.json")
    verification, _ = _read(output / "verification.json")
    if manifest.get("schema") != SCHEMA or verification.get("schema") != SCHEMA:
        raise ValueError("Unsupported additive workbench schema")
    paths = {}
    for row in manifest.get("files", []):
        relative = Path(row["path"])
        path = output / relative
        if (relative.is_absolute() or ".." in relative.parts or row["path"] in paths
                or path.is_symlink() or not path.resolve(strict=True).is_relative_to(output)):
            raise ValueError("Unsafe or duplicate additive artifact")
        verify_row(row, path)
        paths[row["path"]] = row
    for name in ("catalog.json", "verification.json", "index.html", "reference_coverage.json",
                 "additional_admissions.json", "preserved_artifacts.json"):
        if name not in paths:
            raise ValueError("Missing additive core artifact: " + name)
    baseline = Path(verification["baseline_catalog"]["path"]).parent
    if baseline.parent != output.parent or baseline == output or baseline.is_symlink():
        raise ValueError("Baseline must be a different canonical sibling snapshot")
    verify_row(verification["baseline_catalog"])
    verify_row(verification["baseline_manifest"])
    original, _ = _read(baseline / "catalog.json")
    catalog, _ = _read(output / "catalog.json")
    _validate_catalog(original)
    _validate_catalog(catalog)
    _assert_old_preserved(original, catalog)
    old_ids = {group["id"] for group in original["groups"]}
    added = [group for group in catalog["groups"] if group["id"] not in old_ids]
    admissions, _ = _read(output / "additional_admissions.json")
    if _new_projection(added) != admissions.get("groups"):
        raise ValueError("New catalog scores differ from frozen strict admissions")
    if len(added) != 5 or catalog["counts"].get("members") != 620 or catalog["counts"].get("groups") != 52:
        raise ValueError("Additive publication is not exactly 620 members and 52 groups")
    added_ids = [trial["id"] for group in added for trial in group["trials"]]
    if added_ids != verification.get("added_trial_ids") or [group["id"] for group in added] != verification.get("added_group_ids"):
        raise ValueError("Additional identity inventory differs from catalog")
    baseline_inventory, _ = _read(baseline / "artifact_manifest.json")
    baseline_rows = {row["path"]: row for row in baseline_inventory["files"]}
    verify_row(baseline_rows["reference_coverage.json"], baseline / "reference_coverage.json")
    expected_preserved = {row["path"]: row for row in baseline_inventory["files"] if row["path"] not in _REPLACED}
    preserved, _ = _read(output / "preserved_artifacts.json")
    recorded_preserved = {row["path"]: row for row in preserved.get("files", [])}
    if recorded_preserved != expected_preserved or len(recorded_preserved) != len(preserved.get("files", [])):
        raise ValueError("Preserved-byte inventory is not exactly the baseline episode/assets inventory")
    for relative, row in expected_preserved.items():
        if paths.get(relative) != row:
            raise ValueError("Existing episode artifact bytes changed or were omitted: " + relative)
    coverage, _ = _read(output / "reference_coverage.json")
    baseline_coverage, _ = _read(baseline / "reference_coverage.json")
    if coverage.get("trials", [])[:len(baseline_coverage["trials"])] != baseline_coverage["trials"]:
        raise ValueError("Existing reference coverage changed")
    covered = [row["trial_id"] for row in coverage.get("trials", [])]
    all_ids = [trial["id"] for group in catalog["groups"] for trial in group["trials"]]
    if len(covered) != len(set(covered)) or set(covered) != set(all_ids):
        raise ValueError("Reference coverage does not exactly cover all members")
    for group in added:
        for trial in group["trials"]:
            prefix = Path(trial["id"])
            if not all((prefix / name).as_posix() in paths for name in _REQUIRED):
                raise ValueError("New detail lacks a required rich payload or reference download")
            if trial["detail_url"] != (prefix / "report.html").as_posix() or trial.get("video_url") not in paths:
                raise ValueError("New detail/video link is not inventoried locally")
            config, _ = _read(output / prefix / "workbench.json")
            if config.get("unified_identity") != _identity(group, trial, True):
                raise ValueError("New rich detail identity differs from catalog")
            refs, _ = _read(output / prefix / "goal_references.json")
            if refs != config.get("goal_references") or refs.get("status") != "verified":
                raise ValueError("New reference download is missing or differs from displayed configuration")
            verify_row({"sha256": trial["summary_sha256"], "size": paths[(prefix / "summary.json").as_posix()]["size"]}, output / prefix / "summary.json")
            report, _ = _read(output / prefix / "report.json")
            summary, _ = _read(output / prefix / "summary.json")
            visual, _ = _read(output / prefix / "visualization.json")
            if report.get("visualization", {}).get("sha256") != content_hash(visual):
                raise ValueError("New rich report does not bind displayed visual samples")
            for series in visual.get("task_series", []):
                if (series.get("goal_references") or []) != _for_series(refs.get("references", []), series):
                    raise ValueError("New task chart reference differs from its source-bound reference bundle")
            native = report.get("outcome", {}).get("recorded_success")
            if native is not trial["native_success"] or summary.get("task_success") is not trial["native_success"]:
                raise ValueError("New rich report native result differs from admission")
            if report.get("task_specific", {}).get("partial_state") != summary.get("evaluation", {}).get("state"):
                raise ValueError("New rich terminal measurements differ from copied summary")
            displayed_stage = (config.get("staged_evaluation") or {}).get("staged_result", {})
            recorded_stage = summary.get("staged_evaluation", {})
            if (displayed_stage.get("stage_flags") != {stage["key"]: stage["reached"] for stage in trial["stages"]}
                    or displayed_stage.get("complete_success") is not trial["complete_stage_success"]
                    or any(displayed_stage.get(key) != recorded_stage.get(key) for key in ("first_times", "thresholds"))):
                raise ValueError("New displayed stages differ from admitted flags or bound stage metadata")
            terminal = report.get("task_specific", {}).get("partial_state", {}).get("rotation_from_initial_turns")
            angles = [row for row in config.get("task_summary", []) if row.get("key") == "rotation_from_initial_deg"]
            if (type(terminal) not in (int, float) or not math.isfinite(terminal) or len(angles) != 1
                    or angles[0].get("value") != terminal * 360.0 or angles[0].get("unit") != "deg"):
                raise ValueError("New terminal net angle is not the archived terminal state in degrees")
    if verify_sources:
        evaluation_root = Path(verification["evaluation_root"]).resolve(strict=True)
        roots = (output.parent, evaluation_root, PACKAGE.parent.resolve(strict=True))
        for row in verification.get("source_files", []):
            path = Path(row["path"])
            if not path.is_absolute() or path.is_symlink():
                raise ValueError("Additive source must be an absolute regular-file leaf: " + str(path))
            if not any(path.resolve(strict=True).is_relative_to(root) for root in roots):
                if not _reviewed_source(row, path):
                    raise ValueError("Unadmitted additive source path: " + str(path))
            verify_row(row)
        source_groups, _, _ = _selected_job(verification["job_root"], evaluation_root)
        if source_groups != admissions.get("native_groups"):
            raise ValueError("Selected job's strict native/staged admissions changed")
        if _new_projection(_unified_groups(source_groups, Path(verification["job_root"]).name)) != admissions["groups"]:
            raise ValueError("Additional catalog no longer matches strict selected-job admissions")
        from .rich_reference_data import _recover_goals

        for group in added:
            for trial in group["trials"]:
                prefix = output / trial["id"]
                report, _ = _read(prefix / "report.json")
                visual, _ = _read(prefix / "visualization.json")
                config, _ = _read(prefix / "workbench.json")
                recovered = _recover_goals(report, group, trial, output.parent, evaluation_root, [])
                reconstructed = {
                    "report": copy.deepcopy(report), "visualization": copy.deepcopy(visual),
                    "config": {"goal_references": recovered},
                    "coverage": {"series_with_reference": []},
                }
                _relative_reference(reconstructed, prefix / "summary.json", group, trial)
                if config.get("goal_references") != reconstructed["config"]["goal_references"]:
                    raise ValueError("New mechanical or per-trial relative reference differs from authenticated sources")
    if (output / "portal_publication.pending.json").exists():
        raise ValueError("Incomplete additive root publication")
    published = None
    if (output / "portal_publication.json").exists():
        publication, _ = _read(output / "portal_publication.json")
        if publication.get("state") != "committed" or publication.get("root_index") != str(output.parent / "index.html"):
            raise ValueError("Invalid additive root publication audit")
        published = digest(output.parent / "index.html")["sha256"] == publication.get("new_sha256")
    return {"status": "passed", **catalog["counts"], "artifacts": len(paths),
            "existing_score_admissions_preserved": True, "existing_episode_bytes_preserved": True,
            "added_members": len(added_ids), "added_groups": len(added),
            "simulation_runs_started": 0, "root_portal_matches_this_snapshot": published}


def publish_additive_workbench(output):
    """Explicit publication opt-in, after source and immutable-byte verification."""
    output = Path(output).resolve(strict=True)
    result = verify_additive_workbench(output)
    previous = (output / "previous_root_index.html").read_bytes() if (output / "previous_root_index.html").exists() else None
    relative = quote(output.name, safe="._-") + "/index.html"
    landing = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={relative}">'
               f'<title>KaiTactileSim 官方评估平台</title><p><a href="{relative}">进入 KaiTactileSim 统一评估工作台</a></p></html>').encode()
    _publish_portal(output, output.parent / "index.html", previous, landing)
    result["root_portal_matches_this_snapshot"] = True
    return result
