"""Fail-closed CPM append contracts; no simulator, policy, or weight is loaded."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from evaluation_status import cpm_additive_workbench_export as cpm  # noqa: E402
from evaluation_status.rich_workbench_export import _identity  # noqa: E402
from evaluation_status.unified_workbench_export import (  # noqa: E402
    SCHEMA as CATALOG_SCHEMA, _counts, _stats,
)
from evaluation_v1.protocol import content_hash  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n")
    return digest(path)


def digest(path, root=None):
    return {"path": str(path) if root is None else path.relative_to(root).as_posix(),
            "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def inventory(output):
    write(output / "artifact_manifest.json", {"schema": cpm.SCHEMA, "files": [
        digest(path, output) for path in sorted(output.rglob("*")) if path.is_file()
        and path.name not in {"artifact_manifest.json", "portal_publication.json", "portal_publication.pending.json"}]})


def stages(seed):
    definitions = {"stage1": "dragged_on_table", "stage2": "stable_grasp", "stage3": "terminal_hold_over_3s"}
    return {"version": "staged_v1", "stage_definitions": definitions,
            "stage_flags": {"stage1": seed < 7, "stage2": seed < 2, "stage3": False},
            "complete_success": False, "first_times": {"stage1": 1.0 if seed < 7 else None,
                "stage2": 2.0 if seed < 2 else None, "stage3": None},
            "thresholds": {"poker_terminal_hold_s": 3.0}, "continuous_metrics": {},
            "stage_achievement_policy": "ordered_cumulative_ever_reached", "time_basis": "mujoco_data_time_seconds",
            "seed": seed, "task": "poker-draw", "measurements": {"object_position_xyz_m": [0, 0, 0]},
            "state_log_samples": 2701, "physics_valid": True}


def projection(raw):
    return {**{key: copy.deepcopy(raw[key]) for key in ("version", "stage_definitions", "stage_flags",
        "complete_success", "first_times", "thresholds", "continuous_metrics", "stage_achievement_policy", "time_basis")},
        "final_facts": copy.deepcopy(raw["measurements"]), "final_time_s": 90.0}


@pytest.fixture
def job(tmp_path, monkeypatch):
    evaluations = tmp_path / "evaluations"
    job = evaluations / "_jobs" / cpm.JOB_NAME
    manifest_path = job / "deployments/cpm_best/deployment_manifest.json"
    evaluation = job / "groups/cpm_best/attempt_001/evaluation"
    checkpoint = str(job / "assets/cpm/best.pt")
    protocol = {"max_sim_seconds": 90.0, "guard_mode": "observe", "action_interpolation": "zoh",
                "execute_steps": 5, "control_hz": 30, "record_fps": 10,
                "staged_evaluation_version": "staged_v1", "diagnostic_only": True, "formal_metrics_valid": False}
    group = {"label": "cpm_best", "task": "poker-draw", "model_family": "pi05+cpm",
             "checkpoint": checkpoint, "checkpoint_step": "best", "prediction_horizon": 30,
             "action_start_index": 0, "execute_steps": 5,
             "deployment_manifest": str(manifest_path), "evaluation_path": str(evaluation)}
    manifest = {"schema": "poker_tactile_deployment_v1", "task": "poker-draw", "model_family": "pi05+cpm",
                "checkpoint_path": checkpoint, "checkpoint_step": "best", "checkpoint_sha256": cpm.CPM_SHA,
                "prediction_horizon": 30, "action_start_index": 0, "execute_steps": 5, "control_hz": 30,
                "controller_version": "poker-tactile-direct-joints-v1", "required_action_interpolation": "zoh",
                "diagnostic_only": True, "formal_metrics_valid": False,
                "observation_contract": {"tactile_sent_to_model": True},
                "guidance": {"num_steps": 10, "guidance_steps": 3, "scale": 1.0, "variant": "pi05+touchguidance"}}
    manifest["deployment_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    manifest_row = write(manifest_path, manifest)
    write(job / "plan.json", {"schema": cpm.PLAN_SCHEMA, "job_root": str(job), "task": "poker-draw",
          "seeds": list(range(10)), "planned_trials": 10, "initial_state_randomization": True,
          "trial_scope": "randomized_initial_state_trials", "groups": [group], "protocol": protocol})
    write(job / "status.json", {"schema": cpm.STATUS_SCHEMA, "state": "complete", "groups": {"cpm_best": {
        "state": "complete", "accepted_attempt": "attempt_001", "batch_complete": True,
        "evaluation_path": str(evaluation), "batch_summary": str(evaluation / "batch_summary.json")}}})
    rows, witnesses = [], []
    for seed in range(10):
        source = evaluation / f"seed_{seed:04d}"
        summary = {"task": "poker-draw", "model_family": "pi05+cpm", "seed": seed,
                   "deployment_id": manifest["deployment_id"], "deployment_manifest_sha256": manifest_row["sha256"],
                   "checkpoint_path": checkpoint, "checkpoint_sha256": cpm.CPM_SHA,
                   "prediction_horizon": 30, "action_start_index": 0, "execute_steps": 5,
                   "tactile_sent_to_model": True, "expert_used": True,
                   "guidance": {"enabled": True, "steps": 3, "scale": 1.0, "variant": "pi05+touchguidance"},
                   "max_sim_seconds": 90.0, "guard_mode": "observe", "controller": manifest["controller_version"],
                   "action_interpolation": "zoh", "success_hold_seconds": 0.1,
                   "diagnostic_only": True, "formal_metrics_valid": False,
                   "status": "task_not_completed", "task_success": False, "physics_valid": True,
                   "sim_seconds": 90.0, "full_duration_evaluation": True,
                   "staged_evaluation": stages(seed), "video": {"completed": True},
                   "evaluation": {"state": {"card_displacement_m": seed / 1000}}}
        summary_row = write(source / "summary.json", summary)
        write(source / "staged_evaluation.json", stages(seed))
        (source / "review.mp4").write_bytes(b"recorded-video")
        rows.append({"seed": seed, "summary": f"seed_{seed:04d}/summary.json", "returncode": 0,
                     "timed_out": False, "valid": True, "identity_matches": True})
        witnesses.append({"seed": seed, "source": str(source), "summary_sha256": summary_row["sha256"],
                          "native_success": False, "stages": stages(seed)["stage_flags"]})
    write(evaluation / "batch_summary.json", {"schema": "poker_tactile_batch_v1", "task": "poker-draw",
        "model_family": "pi05+cpm", "prediction_horizon": 30, "action_start_index": 0, "execute_steps": 5,
        "deployment_id": manifest["deployment_id"], "deployment_manifest_sha256": manifest_row["sha256"],
        "complete": True, "diagnostic_only": True, "formal_metrics_valid": False, "trials": rows})
    qa = job / "independent_qa.json"
    write(qa, {"schema": cpm.QA_SCHEMA, "passed": True, "job_root": str(job), "evaluation_root": str(evaluation),
        "deployment_manifest_sha256": manifest_row["sha256"], "checkpoint_sha256": cpm.CPM_SHA,
        "cpm_checkpoint_sha256": cpm.CPM_SHA, "planned_trials": 10, "native_valid_trials": 10,
        "native_success_count": 0, "stage_valid_trials": 10,
        "stage_counts": {"stage1": 7, "stage2": 2, "stage3": 0}, "trials": witnesses})
    calls = []

    def strict_projection(root, source, horizon, valid, inputs):
        assert root == job and horizon == 30 and valid is True
        calls.append(source)
        return {"staged_result": projection(json.loads((source / "staged_evaluation.json").read_text()))}, "verified", [], {
            "full_rule_identity_sha256": "a" * 64, "observer_sha256": "b" * 64,
            "recording_sha256": "c" * 64, "definitions_thresholds_sha256": "d" * 64}

    monkeypatch.setattr(cpm, "_staged_projection", strict_projection)
    return job, evaluations, qa, calls


def test_selected_current_cpm_has_separate_sha_checkpoint_and_real_failures_in_denominator(job):
    path, evaluations, qa, calls = job
    group, sources, inputs, bound_qa = cpm._selected_job(path, evaluations, qa)
    assert len(calls) == len(sources) == 10
    assert group["checkpoint"] == "best@e5f859a57552" and group["checkpoint_sha256"] == cpm.CPM_SHA
    assert group["model"] == "pi05+cpm" and group["native_known_count"] == 10
    assert group["native_success_count"] == 0
    assert {row["key"]: row["reached"] for row in group["stages"]} == {"stage1": 7, "stage2": 2, "stage3": 0}
    assert all(trial["valid"] and trial["staged_evidence_status"] == "valid" for trial in group["trials"])
    assert group["initial_state_randomization"] is True
    assert bound_qa == digest(qa)
    assert any(row["role"] == "cpm_n10_independent_raw_evidence_qa" for row in inputs)


def test_real_projection_shape_is_not_mistaken_for_the_complete_raw_sidecar():
    raw = stages(0)
    checked = projection(raw)
    assert "measurements" not in checked and "final_facts" not in raw and raw != checked
    cpm._bind_stage_projection(checked, raw, copy.deepcopy(raw))


@pytest.mark.parametrize("change", [
    lambda d: d["stage_flags"].update(stage1=False),
    lambda d: d["first_times"].update(stage1=99.0),
    lambda d: d["thresholds"].update(poker_terminal_hold_s=1.0),
    lambda d: d.update(complete_success=True),
    lambda d: d.update(complete_success=0),
    lambda d: d.pop("continuous_metrics"),
])
def test_verified_projection_must_match_every_typed_display_score_field(change):
    raw = stages(0)
    checked = projection(raw)
    change(checked)
    with pytest.raises(ValueError, match="verified stage projection differs"):
        cpm._bind_stage_projection(checked, raw, copy.deepcopy(raw))


@pytest.mark.parametrize("summary", [
    {}, {"evaluation": None}, {"evaluation": []}, {"evaluation": "unrecorded"},
    {"evaluation": {"success": False, "contacted": True}},
    {"evaluation": {"state": None}}, {"evaluation": {"state": []}},
    {"evaluation": {"state": False}}, {"evaluation": {"state": 0}},
])
def test_missing_or_nondict_terminal_state_matches_existing_rich_adapter_empty_projection(summary):
    frozen = copy.deepcopy(summary)
    assert cpm._expected_terminal_state(summary) == {}
    assert summary == frozen
    assert not cpm._same_json(None, cpm._expected_terminal_state(summary))
    assert not cpm._same_json({"invented_measurement": 0}, cpm._expected_terminal_state(summary))


def test_recorded_terminal_state_is_preserved_without_coercion_or_invented_fields():
    summary = {"evaluation": {"state": {"measured": 0.0, "contacted": False}}}
    expected = cpm._expected_terminal_state(summary)
    assert cpm._same_json({"measured": 0.0, "contacted": False}, expected)
    assert not cpm._same_json({"measured": 0, "contacted": False}, expected)
    assert not cpm._same_json({"measured": 0.0, "contacted": 0}, expected)
    assert not cpm._same_json({"measured": 0.0}, expected)


@pytest.mark.parametrize("filename,change,message", [
    ("plan.json", lambda d: d.update(seeds=list(range(5))), "seeds 0..9"),
    ("plan.json", lambda d: d.update(initial_state_randomization=False), "randomized-initial"),
    ("plan.json", lambda d: d["groups"].append(copy.deepcopy(d["groups"][0])), "only cpm_best"),
    ("plan.json", lambda d: d["groups"][0].update(checkpoint="/nas/old/best.pt"), "checkpoint"),
    ("plan.json", lambda d: d["protocol"].update(execute_steps=30), "protocol"),
    ("status.json", lambda d: d.update(state="running"), "must be complete"),
    ("status.json", lambda d: d["groups"]["cpm_best"].update(accepted_attempt="attempt_002"), "Unexpected accepted"),
    ("independent_qa.json", lambda d: d.update(passed=False), "QA did not pass"),
    ("independent_qa.json", lambda d: d.update(checkpoint_sha256="a" * 64), "QA identity"),
    ("independent_qa.json", lambda d: d["trials"][0].update(summary_sha256="b" * 64), "QA trial differs"),
    ("independent_qa.json", lambda d: d["trials"][0].update(seed=1), "duplicate/missing"),
    ("independent_qa.json", lambda d: d["stage_counts"].update(stage1=8), "stage counts"),
])
def test_rejects_contract_drift(job, filename, change, message):
    path, evaluations, qa, _ = job
    edit(path / filename, change)
    with pytest.raises(ValueError, match=message):
        cpm._selected_job(path, evaluations, qa)


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(physics_valid=False), "native/identity"),
    (lambda d: d.update(task_success=None), "native/identity"),
    (lambda d: d.update(expert_used=False), "native/identity"),
    (lambda d: d.update(sim_seconds=20.0), "90-second"),
    (lambda d: d.update(full_duration_evaluation=False), "90-second"),
    (lambda d: d["staged_evaluation"]["stage_flags"].update(stage1=False), "sidecar differs"),
])
def test_no_silent_admission_of_invalid_native_or_stage_data(job, change, message):
    path, evaluations, qa, _ = job
    summary = path / "groups/cpm_best/attempt_001/evaluation/seed_0000/summary.json"
    edit(summary, change)
    with pytest.raises(ValueError, match=message):
        cpm._selected_job(path, evaluations, qa)


def baseline_catalog():
    groups = []
    index = 0
    for ordinal in range(47):
        trials = []
        for _ in range(13 if ordinal < 31 else 12):
            quarantine = index >= 575
            trials.append({"id": f"old-{index:04d}", "seed": index, "valid": not quarantine,
                "native_success": None if quarantine else False,
                "publication_status": "quarantined_replay_only" if quarantine else "eligible",
                "staged_evidence_status": "quarantined" if quarantine else "invalid",
                "stages": [], "complete_stage_success": None, "continuous_metrics": {},
                "detail_url": f"old-{index:04d}/report.html", "video_url": None})
            index += 1
        group = {"id": f"old-group-{ordinal}", "task": "poker-draw" if ordinal == 0 else f"task-{ordinal % 10}",
                 "model": "pi05+cpm" if ordinal == 0 else "pi05", "checkpoint": "best" if ordinal == 0 else str(ordinal),
                 "prediction_horizon": 30 if ordinal == 0 else 50, "execute_steps": 5 if ordinal == 0 else 50,
                 "trials": trials}
        group.update(_stats(trials))
        groups.append(group)
    assert index == 595
    return {"schema": CATALOG_SCHEMA, "title": "Frozen platform", "groups": groups, "counts": _counts(groups),
            "tasks": [{"task": "poker-draw", "title": "卡片"}], "historical_summary": {"not_counted_again": True},
            "links": [], "duplicates": [], "rich_details": {"count": 595}}


@pytest.fixture
def snapshot(job, tmp_path, monkeypatch):
    path, evaluations, qa, _ = job
    added, sources, _, qa_row = cpm._selected_job(path, evaluations, qa)
    original = baseline_catalog()
    baseline, output = tmp_path / "reports/baseline", tmp_path / "reports/cpm"
    write(baseline / "catalog.json", original)
    old_coverage = {"trials": [{"trial_id": trial["id"], "reference_status": "unchanged"}
                             for group in original["groups"] for trial in group["trials"]]}
    write(baseline / "reference_coverage.json", old_coverage)
    (baseline / "old-0000").mkdir()
    (baseline / "old-0000/report.html").write_text("old cpm remains invalid")
    inventory(baseline)
    preserved = [row for row in json.loads((baseline / "artifact_manifest.json").read_text())["files"]
                 if row["path"] not in cpm._REPLACED]
    for row in preserved:
        target = output / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline / row["path"], target)
    coverage = copy.deepcopy(old_coverage)
    for trial in added["trials"]:
        prefix = output / trial["id"]
        prefix.mkdir()
        shutil.copyfile(Path(sources[trial["id"]]) / "summary.json", prefix / "summary.json")
        summary = json.loads((prefix / "summary.json").read_text())
        refs = {"status": "verified", "references": []}
        visual = {"task_series": [], "media": {"url": "review.mp4", "sha256": "f" * 64}}
        report = {"outcome": {"recorded_success": False}, "visualization": {"sha256": content_hash(visual)},
                  "task_specific": {"partial_state": summary["evaluation"]["state"]}}
        config = {"unified_identity": _identity(added, trial, True), "goal_references": refs,
                  "staged_evaluation": {"staged_result": summary["staged_evaluation"]},
                  "media_views": {"status": "verified", "sha256": "f" * 64,
                                  "views": [{"id": key} for key in ("head", "wrist", "global")]}}
        for name, value in (("report.json", report), ("visualization.json", visual), ("workbench.json", config),
                            ("goal_references.json", refs), ("tactile_loads.json", {})):
            write(prefix / name, value)
        (prefix / "report.html").write_text("rich cpm detail")
        (prefix / "audit.html").write_text("audit")
        (prefix / "review.mp4").write_bytes(b"recorded-video")
        trial["video_url"] = trial["id"] + "/review.mp4"
        trial["rich_detail"] = {"native_samples": 2701}
        coverage["trials"].append({"trial_id": trial["id"], "reference_status": "verified"})
    catalog = copy.deepcopy(original)
    catalog["groups"].append(added)
    catalog["counts"] = _counts(catalog["groups"])
    write(output / "catalog.json", catalog)
    write(output / "reference_coverage.json", coverage)
    write(output / "preserved_artifacts.json", {"schema": cpm.SCHEMA, "files": preserved})
    write(output / "verification.json", {"schema": cpm.SCHEMA, "baseline_catalog": digest(baseline / "catalog.json"),
        "baseline_manifest": digest(baseline / "artifact_manifest.json"), "evaluation_root": str(evaluations),
        "job_root": str(path), "independent_qa": qa_row,
        "added_group_id": added["id"], "added_trial_ids": [trial["id"] for trial in added["trials"]], "source_files": []})
    (output / "index.html").write_text("605 members")
    inventory(output)
    monkeypatch.setattr(cpm, "_cpm_media_views", lambda summary, media, root: {
        "status": "verified", "sha256": "f" * 64, "views": [{"id": key} for key in ("head", "wrist", "global")]})
    return baseline, output


def test_snapshot_preserves_old_595_and_separate_new_group_with_source_recheck(snapshot):
    baseline, output = snapshot
    result = cpm.verify_cpm_additive_workbench(output)
    assert result["status"] == "passed" and result["members"] == 605
    assert result["groups"] == 48 and result["eligible"] == 585 and result["quarantined"] == 20
    assert result["native_known_count"] == 10 and result["native_success_count"] == 0
    assert (baseline / "old-0000/report.html").read_bytes() == (output / "old-0000/report.html").read_bytes()
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["groups"][0]["checkpoint"] == "best"
    assert catalog["groups"][-1]["checkpoint"] == cpm.CHECKPOINT_LABEL
    assert all(trial["staged_evidence_status"] == "invalid" for trial in catalog["groups"][0]["trials"])


@pytest.mark.parametrize("relative,change,message", [
    ("catalog.json", lambda d: d["groups"][0]["trials"][0].update(detail_url="new/report.html"), "existing group"),
    ("catalog.json", lambda d: d["groups"][-1].update(checkpoint="best"), "separate weight"),
    ("reference_coverage.json", lambda d: d["trials"].pop(), "exactly match"),
    ("preserved_artifacts.json", lambda d: d["files"].clear(), "Preserved-byte"),
])
def test_refreshed_manifest_cannot_hide_old_identity_or_membership_mutation(snapshot, relative, change, message):
    _, output = snapshot
    edit(output / relative, change)
    inventory(output)
    with pytest.raises(ValueError, match=message):
        cpm.verify_cpm_additive_workbench(output, verify_sources=False)


@pytest.mark.parametrize("filename,change,message", [
    ("workbench.json", lambda d: d["unified_identity"].update(checkpoint="best"), "displayed identity"),
    ("report.json", lambda d: d["outcome"].update(recorded_success=True), "native result"),
    ("workbench.json", lambda d: d["staged_evaluation"]["staged_result"]["first_times"].update(stage1=99), "stage values"),
    ("goal_references.json", lambda d: d.update(status="unavailable"), "reference download"),
    ("visualization.json", lambda d: d.update(forged_samples=[0]), "report-bound"),
])
def test_refreshed_manifest_cannot_hide_display_semantic_drift(snapshot, filename, change, message):
    _, output = snapshot
    trial = json.loads((output / "catalog.json").read_text())["groups"][-1]["trials"][0]
    edit(output / trial["id"] / filename, change)
    inventory(output)
    with pytest.raises(ValueError, match=message):
        cpm.verify_cpm_additive_workbench(output, verify_sources=False)


def test_real_poker_missing_state_projection_verifies_without_inventing_terminal_values(snapshot):
    _, output = snapshot
    catalog = json.loads((output / "catalog.json").read_text())
    trial = catalog["groups"][-1]["trials"][0]
    prefix = output / trial["id"]
    edit(prefix / "summary.json", lambda value: value.update(evaluation={"success": False, "contacted": True}))
    trial["summary_sha256"] = digest(prefix / "summary.json")["sha256"]
    write(output / "catalog.json", catalog)
    edit(prefix / "report.json", lambda value: value["task_specific"].update(partial_state={}))
    inventory(output)
    assert cpm.verify_cpm_additive_workbench(output, verify_sources=False)["status"] == "passed"
    edit(prefix / "report.json", lambda value: value["task_specific"].update(partial_state={"contacted": True}))
    inventory(output)
    with pytest.raises(ValueError, match="terminal measurements changed"):
        cpm.verify_cpm_additive_workbench(output, verify_sources=False)


def test_forged_browser_receipt_cannot_publish_another_catalog(snapshot, monkeypatch):
    _, output = snapshot
    write(output / "browser_qa/cpm_additive_browser_verification.json", {"status": "passed", "members": 605,
        "new_dom_trials": 10, "snapshot_directory": str(output), "catalog_sha256": "0" * 64})
    monkeypatch.setattr(cpm, "verify_cpm_additive_workbench", lambda path: {"status": "passed"})
    with pytest.raises(ValueError, match="snapshot-bound browser QA"):
        cpm.publish_cpm_additive_workbench(output)


@pytest.fixture
def crop(tmp_path, monkeypatch):
    from evaluation_workbench.experts import _MODEL_LAYOUT
    from evaluation_status.tactile_experiment_export import _content_identity

    renderer = tmp_path / "source/src/kaihand_tactile_env/shared/evaluation_video.py"
    renderer.parent.mkdir(parents=True)
    renderer.write_text("reviewed renderer bytes")
    renderer_sha = digest(renderer)["sha256"]
    monkeypatch.setattr(cpm, "CPM_RENDERER_SHA", renderer_sha)
    manifest = {"task": "poker-draw", "checkpoint_sha256": cpm.CPM_SHA, "prediction_horizon": 30,
                "files": {str(renderer): renderer_sha}}
    manifest["deployment_id"] = _content_identity(manifest)
    manifest_path = tmp_path / "evaluations/_jobs/current/deployment_manifest.json"
    row = write(manifest_path, manifest)
    summary = {"model_family": "pi05+cpm", "checkpoint_sha256": cpm.CPM_SHA,
               "prediction_horizon": 30, "execute_steps": 5, "deployment_manifest": str(manifest_path),
               "deployment_manifest_sha256": row["sha256"], "deployment_id": manifest["deployment_id"],
               "video": {"schema": "kaihand-policy-evaluation-review-v3", "layout": _MODEL_LAYOUT,
                    "output_size": [1920, 1080], "review_render_size": [640, 480],
                    "model_input_cameras_displayed": ["head", "right_wrist"],
                    "review_only_cameras_displayed": ["global"], "right_wrist_camera_role": "model_input",
                    "completed": True}}
    return summary, {"sha256": "f" * 64}, tmp_path / "evaluations", renderer


def test_h30_crop_binds_only_current_cpm_manifest_renderer_and_displayed_video(crop):
    summary, media, evaluations, renderer = crop
    views = cpm._cpm_media_views(summary, media, evaluations)
    assert views["status"] == "verified" and views["sha256"] == media["sha256"]
    assert [view["id"] for view in views["views"]] == ["head", "wrist", "global"]
    assert views["provenance"][-1]["sha256"] == digest(renderer)["sha256"]
    assert views["views"][1]["height"] == 294


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(prediction_horizon=50), "another model"),
    (lambda d: d["video"].update(right_wrist_camera_role="review_only"), "model roles"),
    (lambda d: d["video"].update(model_input_cameras_displayed=["head"]), "model roles"),
    (lambda d: d.update(deployment_manifest_sha256="0" * 64), "manifest identity"),
])
def test_h30_crop_never_borrows_another_identity_or_unverified_wrist(crop, change, message):
    summary, media, evaluations, _ = crop
    change(summary)
    with pytest.raises(ValueError, match=message):
        cpm._cpm_media_views(summary, media, evaluations)


def test_h30_crop_refuses_renderer_bytes_drift(crop):
    summary, media, evaluations, renderer = crop
    renderer.write_text("modified renderer")
    with pytest.raises(ValueError, match="renderer bytes changed"):
        cpm._cpm_media_views(summary, media, evaluations)


def test_full_export_builds_fresh_605_snapshot_and_does_not_publish(snapshot, job, monkeypatch):
    baseline, prepared = snapshot
    job_root, evaluations, qa, _ = job
    target = baseline.parent / "new-export"
    (baseline.parent / "index.html").write_text("unchanged root platform")
    monkeypatch.setattr(cpm, "verify_reference_workbench", lambda *args, **kwargs: {"status": "passed"})

    class InlinePool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def map(self, function, arguments, **kwargs):
            return map(function, arguments)

    def export_one(arguments):
        output, _, _, trial, _ = arguments
        output, artifacts, inputs = Path(output), [], {}
        for source in sorted((prepared / trial["id"]).rglob("*")):
            if source.is_file():
                cpm._copy(output, source.relative_to(prepared), source, artifacts, inputs)
        return {"id": trial["id"], "video_url": trial["id"] + "/review.mp4",
                "availability": {"native_samples": 2701}, "coverage": {"trial_id": trial["id"]},
                "artifacts": artifacts, "inputs": list(inputs.values())}

    monkeypatch.setattr(cpm, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(cpm, "_export_one", export_one)
    result = cpm.export_cpm_additive_workbench(baseline, target, job_root=job_root,
        evaluation_root=evaluations, qa_receipt=qa, workers=2)
    assert result["status"] == "passed" and result["members"] == 605 and result["groups"] == 48
    assert (baseline.parent / "index.html").read_text() == "unchanged root platform"
    assert not (target / "portal_publication.json").exists()
    assert "605" in (target / "index.html").read_text()
    assert cpm.verify_cpm_additive_workbench(target)["added_members"] == 10
