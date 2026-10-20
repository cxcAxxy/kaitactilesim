"""Small offline append guards: no simulator, policy, or checkpoint is loaded."""
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

from evaluation_status import additive_workbench_export as additive  # noqa: E402
from evaluation_status import data  # noqa: E402
from evaluation_status.rich_workbench_export import _identity  # noqa: E402
from evaluation_status.unified_workbench_export import (  # noqa: E402
    SCHEMA as CATALOG_SCHEMA,
)
from evaluation_status.unified_workbench_export import _counts, _stats  # noqa: E402


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n")
    return _digest(path)


def _digest(path, root=None):
    return {"path": str(path) if root is None else path.relative_to(root).as_posix(),
            "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _edit(path, function):
    value = json.loads(path.read_text())
    function(value)
    _json(path, value)


def _inventory(output, schema=additive.SCHEMA):
    _json(output / "artifact_manifest.json", {"schema": schema, "files": [
        _digest(path, output) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "portal_publication.json", "portal_publication.pending.json"}]})


def _stages():
    definitions = {f"s{i}": label for i, label in enumerate(("原位抓稳", "旋入就位", "承力拧紧", "松手稳定"), 1)}
    return {"version": "staged_v1", "full_duration_evaluation": True,
            "stage_definitions": definitions, "stage_flags": dict.fromkeys(definitions, False),
            "stages": dict.fromkeys(definitions, False), "complete_success": False,
            "first_times": dict.fromkeys(definitions), "thresholds": {"hold_s": 0.2}, "continuous_metrics": {}}


@pytest.fixture
def job(tmp_path, monkeypatch):
    evaluations = tmp_path / "evaluations"
    job_root = evaluations / "_jobs" / "requested-inplace"
    plan_groups = []
    for step in additive.STEPS:
        group_dir = f"groups/{additive.TASK}/step{step}"
        plan_groups.append({"task": additive.TASK, "step": step,
                            "checkpoint": str(additive.CHECKPOINT_ROOT / str(step)), "group_dir": group_dir})
        for seed in range(5):
            source = job_root / group_dir / "attempt_001/run/evaluation" / f"seed_{seed:03d}"
            initial_turns = 0.0004
            state = {"clockwise_turns": initial_turns + seed / 100, "rotation_from_initial_turns": seed / 100}
            summary = {"task": additive.TASK, "seed": seed, "model_family": "pi0.5",
                       "checkpoint_path": str(additive.CHECKPOINT_ROOT / str(step)),
                       "checkpoint_sha256": hashlib.sha256(str(step).encode()).hexdigest(),
                       "prediction_horizon": 50, "execute_steps": 50, "sim_seconds": 90.0,
                       "full_duration_evaluation": True, "physics_valid": True,
                       "diagnostic_only": True, "formal_metrics_valid": False,
                       "status": "task_not_completed", "task_success": False,
                       "staged_evaluation": _stages(), "video": {"completed": True},
                       "evaluation": {"success": False, "state": state,
                           "threaded_initialization": {"schema": "bulb_screw_inplace_threaded_reset_v1",
                               "initial_clockwise_turns": initial_turns, "position_randomization_applied": False}}}
            _json(source / "summary.json", summary)
            _json(source / "staged_evaluation.json", _stages())
            (source / "review.mp4").write_bytes(b"recorded-video")
    _json(job_root / "batch.json", {"schema": additive.JOB_SCHEMA, "initial_state_randomization": False,
        "trial_scope": "fixed_initial_state_repeats", "groups": plan_groups,
        "settings": {"prediction_horizon": 50, "execute_steps": 50, "num_trials": 5}})
    _json(job_root / "status.json", {"schema": additive.JOB_SCHEMA, "state": "complete",
        "groups": [{"state": "complete", "attempt": 1} for _ in additive.STEPS]})
    calls = []

    def staged_loader(source, *, include_state_log):
        assert include_state_log is False
        calls.append(source)
        return {"staged_result": _stages(), "rule_identity": {"sha256": "b" * 64}, "provenance": []}

    monkeypatch.setattr(data, "load_staged_result", staged_loader)
    return job_root, evaluations, calls


def test_exact_selected_job_uses_strict_loader_and_keeps_real_failures_in_denominator(job):
    job_root, evaluations, calls = job
    groups, sources, batch = additive._selected_job(job_root, evaluations)
    assert len(calls) == 25 and len(groups) == 5
    assert all(group["valid_trials"] == group["native_known_count"] == 5 for group in groups)
    assert all(group["native_success_count"] == 0 for group in groups)
    assert batch["initial_state_randomization"] is False
    assert any(row["role"] == "additive_completed_status_attempt_selection" for row in sources)
    converted = additive._unified_groups(groups, job_root.name)
    assert all(group["trial_scope"] == "fixed_initial_state_repeats" for group in converted)
    assert additive._counts(converted)["staged_verified"] == 25


@pytest.mark.parametrize("filename,change,message", [
    ("status.json", lambda d: d.update(state="running"), "not complete"),
    ("status.json", lambda d: d.update(schema="kaihand-pi05-stage-batch-v1"), "schema"),
    ("batch.json", lambda d: d.update(initial_state_randomization=True), "fixed-initial"),
    ("batch.json", lambda d: d["settings"].update(execute_steps=5), "H50/K50"),
    ("batch.json", lambda d: d["groups"][0].update(step=12000), "unrequested"),
    ("batch.json", lambda d: d["groups"][0].update(checkpoint="/other/15000"), "path differs"),
    ("batch.json", lambda d: d["groups"][0].update(group_dir="../escape"), "path or accepted"),
    ("status.json", lambda d: d["groups"][0].update(attempt=True), "positive accepted"),
])
def test_selected_job_contract_is_not_broadened(job, filename, change, message):
    job_root, evaluations, _ = job
    _edit(job_root / filename, change)
    with pytest.raises(ValueError, match=message):
        additive._selected_job(job_root, evaluations)


@pytest.mark.parametrize("change,message", [
    (lambda d: d.update(physics_valid=False), "verified native"),
    (lambda d: d.update(full_duration_evaluation=False), "90-second"),
    (lambda d: d.update(sim_seconds=30.0), "90-second"),
    (lambda d: d.update(record_error="camera failed"), "verified native"),
])
def test_infrastructure_failures_are_not_silently_admitted_as_zero_successes(job, change, message):
    job_root, evaluations, _ = job
    source = next(job_root.glob("groups/*/*/attempt_001/run/evaluation/seed_000/summary.json"))
    _edit(source, change)
    with pytest.raises(ValueError, match=message):
        additive._selected_job(job_root, evaluations)


def _reference_case(summary_path):
    summary = json.loads(summary_path.read_text())
    absolute = {"key": "clockwise_angle_deg", "unit": "deg", "target": 100.0,
                "lower": 97.84, "upper": 102.16, "relation": "target", "label": "绝对到座角"}
    return {"report": {"task_specific": {"partial_state": copy.deepcopy(summary["evaluation"]["state"])},
                       "goal_reference_extension": {}, "visualization": {}},
            "visualization": {"task_series": [{"key": "rotation_from_initial_deg", "unit": "deg",
                                                  "time_s": [0.0, 90.0], "values": [0.0, 1.0]}]},
            "config": {"goal_references": {"status": "verified", "references": [absolute], "provenance": []}},
            "coverage": {"series_with_reference": []}}


def test_relative_goal_uses_actual_measured_reset_not_nominal_100_degrees(job):
    job_root, _, _ = job
    summary = next(job_root.glob("groups/*/*/attempt_001/run/evaluation/seed_000/summary.json"))
    case = _reference_case(summary)
    trial = {"summary_sha256": _digest(summary)["sha256"]}
    additive._relative_reference(case, summary, {"task": additive.TASK}, trial)
    relative = case["config"]["goal_references"]["references"][-1]
    assert relative["target"] == pytest.approx(99.856)
    assert relative["lower"] == pytest.approx(97.696)
    assert relative["binding"]["summary"]["sha256"] == trial["summary_sha256"]
    assert case["visualization"]["task_series"][0]["goal_references"] == [relative]
    assert case["report"]["task_specific"]["partial_state"] == json.loads(summary.read_text())["evaluation"]["state"]


def test_relative_goal_rejects_inconsistent_terminal_measurements(job):
    job_root, _, _ = job
    summary = next(job_root.glob("groups/*/*/attempt_001/run/evaluation/seed_000/summary.json"))
    case = _reference_case(summary)
    case["report"]["task_specific"]["partial_state"]["rotation_from_initial_turns"] = 2.0
    with pytest.raises(ValueError, match="contradicts terminal"):
        additive._relative_reference(case, summary, {"task": additive.TASK}, {"summary_sha256": _digest(summary)["sha256"]})


def _original_catalog():
    groups = []
    index = 0
    for ordinal in range(47):
        trials = []
        model, horizon, execute = ("pi05+cpm", 30, 5) if ordinal == 0 else ("pi05", 50, 50)
        for _ in range(13 if ordinal < 31 else 12):
            quarantine = index >= 575
            stage_valid = 1 <= index <= 450
            trials.append({"id": f"old-{index:04d}", "seed": index, "valid": not quarantine,
                "native_success": None if quarantine else False,
                "publication_status": "quarantined_replay_only" if quarantine else "eligible",
                "staged_evidence_status": "quarantined" if quarantine else "valid" if stage_valid else "invalid",
                "stages": [], "complete_stage_success": False if stage_valid else None,
                "continuous_metrics": {}, "detail_url": f"old-{index:04d}/report.html", "video_url": None})
            index += 1
        group = {"id": f"old-group-{ordinal}", "task": f"old-task-{ordinal % 10}", "model": model,
                 "checkpoint": str(ordinal), "prediction_horizon": horizon, "execute_steps": execute, "trials": trials}
        group.update(_stats(trials))
        groups.append(group)
    assert index == 595
    return {"schema": CATALOG_SCHEMA, "title": "Frozen workbench", "groups": groups, "counts": _counts(groups),
            "tasks": [{"task": f"old-task-{i}", "title": str(i)} for i in range(10)],
            "historical_summary": {"not_counted_again": True}, "links": [], "duplicates": [], "rich_details": {"count": 595}}


@pytest.fixture
def snapshot(job, tmp_path):
    job_root, evaluations, _ = job
    native_groups, _, _ = additive._selected_job(job_root, evaluations)
    additions = additive._unified_groups(native_groups, job_root.name)
    original = _original_catalog()
    baseline, output = tmp_path / "reports/baseline", tmp_path / "reports/additive"
    _json(baseline / "catalog.json", original)
    old_coverage = {"schema": "old", "trials": [
        {"trial_id": trial["id"], "reference_status": "unchanged"}
        for group in original["groups"] for trial in group["trials"]]}
    _json(baseline / "reference_coverage.json", old_coverage)
    (baseline / "old-0000").mkdir()
    (baseline / "old-0000/report.html").write_text("unchanged old rich bytes")
    _inventory(baseline, "baseline")
    baseline_inventory = json.loads((baseline / "artifact_manifest.json").read_text())["files"]
    preserved = [row for row in baseline_inventory if row["path"] not in additive._REPLACED]
    for row in preserved:
        target = output / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline / row["path"], target)
    by_id = {trial["id"]: trial for group in native_groups for trial in group["trials"]}
    coverage = copy.deepcopy(old_coverage)
    for group in additions:
        for trial in group["trials"]:
            prefix = output / trial["id"]
            prefix.mkdir()
            native = by_id[trial["id"]]
            shutil.copyfile(native["summary_path"], prefix / "summary.json")
            summary = json.loads((prefix / "summary.json").read_text())
            case = _reference_case(prefix / "summary.json")
            additive._relative_reference(case, prefix / "summary.json", group, trial)
            case["report"]["outcome"] = {"recorded_success": False}
            config = case["config"]
            config["unified_identity"] = _identity(group, trial, True)
            config["staged_evaluation"] = {"staged_result": summary["staged_evaluation"]}
            config["task_summary"] = [{"key": "rotation_from_initial_deg", "value": summary["evaluation"]["state"]["rotation_from_initial_turns"] * 360,
                                        "unit": "deg"}]
            for filename, value in (("report.json", case["report"]), ("visualization.json", case["visualization"]),
                                    ("workbench.json", config), ("goal_references.json", config["goal_references"]),
                                    ("tactile_loads.json", {})):
                _json(prefix / filename, value)
            (prefix / "report.html").write_text("rich detail")
            (prefix / "audit.html").write_text("audit")
            (prefix / "review.mp4").write_bytes(b"recorded-video")
            trial["video_url"] = f"{trial['id']}/review.mp4"
            coverage["trials"].append({"trial_id": trial["id"]})
    catalog = copy.deepcopy(original)
    catalog["groups"].extend(additions)
    catalog["tasks"].append({"task": additive.TASK, "title": "原位拧入"})
    catalog["counts"] = _counts(catalog["groups"])
    _json(output / "catalog.json", catalog)
    _json(output / "reference_coverage.json", coverage)
    _json(output / "preserved_artifacts.json", {"schema": additive.SCHEMA, "files": preserved})
    _json(output / "additional_admissions.json", {"schema": additive.SCHEMA,
                                                "groups": additive._new_projection(additions), "native_groups": native_groups})
    _json(output / "verification.json", {"schema": additive.SCHEMA, "baseline_catalog": _digest(baseline / "catalog.json"),
        "baseline_manifest": _digest(baseline / "artifact_manifest.json"), "evaluation_root": str(evaluations),
        "job_root": str(job_root), "source_files": [],
        "added_trial_ids": [trial["id"] for group in additions for trial in group["trials"]],
        "added_group_ids": [group["id"] for group in additions]})
    (output / "index.html").write_text("overview")
    _inventory(output)
    return output


def test_additive_verification_preserves_595_and_admits_only_new_25(snapshot):
    result = additive.verify_additive_workbench(snapshot, verify_sources=False)
    assert result["status"] == "passed"
    assert (result["members"], result["groups"], result["tasks"]) == (620, 52, 11)
    assert (result["eligible"], result["quarantined"], result["staged_verified"]) == (600, 20, 475)
    assert result["existing_score_admissions_preserved"] is True


def _mock_mechanical_reader(snapshot, monkeypatch):
    from evaluation_status import rich_reference_data

    def recover(report, group, trial, reports_root, evaluation_root, provenance):
        return _reference_case(snapshot / trial["id"] / "summary.json")["config"]["goal_references"]

    monkeypatch.setattr(rich_reference_data, "_recover_goals", recover)


def test_source_verification_reloads_exact_selected_job_and_reconstructs_references(snapshot, monkeypatch):
    _mock_mechanical_reader(snapshot, monkeypatch)
    assert additive.verify_additive_workbench(snapshot)["status"] == "passed"


def test_self_consistent_but_wrong_relative_reference_is_rejected_against_source(snapshot, monkeypatch):
    from evaluation_v1.protocol import content_hash

    _mock_mechanical_reader(snapshot, monkeypatch)
    catalog = json.loads((snapshot / "catalog.json").read_text())
    prefix = snapshot / catalog["groups"][-1]["trials"][0]["id"]
    _edit(prefix / "workbench.json", lambda d: d["goal_references"]["references"][-1].update(target=100.0))
    _edit(prefix / "goal_references.json", lambda d: d["references"][-1].update(target=100.0))
    _edit(prefix / "visualization.json", lambda d: d["task_series"][0]["goal_references"][0].update(target=100.0))
    visual = json.loads((prefix / "visualization.json").read_text())
    _edit(prefix / "report.json", lambda d: d["visualization"].update(sha256=content_hash(visual)))
    _inventory(snapshot)
    with pytest.raises(ValueError, match="differs from authenticated sources"):
        additive.verify_additive_workbench(snapshot)


@pytest.mark.parametrize("change", [
    lambda c: c["groups"][0]["trials"][0].update(staged_evidence_status="valid"),
    lambda c: c["groups"][0].update(warnings=["silently changed"]),
    lambda c: c["historical_summary"].update(note="rewritten"),
])
def test_refreshed_inventory_cannot_hide_old_admission_or_metadata_changes(snapshot, change):
    _edit(snapshot / "catalog.json", change)
    _inventory(snapshot)
    with pytest.raises(ValueError):
        additive.verify_additive_workbench(snapshot, verify_sources=False)


def test_refreshed_inventory_cannot_hide_old_episode_byte_changes(snapshot):
    (snapshot / "old-0000/report.html").write_text("changed old evidence")
    _inventory(snapshot)
    with pytest.raises(ValueError, match="Existing episode artifact bytes"):
        additive.verify_additive_workbench(snapshot, verify_sources=False)


@pytest.mark.parametrize("filename,change,message", [
    ("report.json", lambda d: d["task_specific"]["partial_state"].update(rotation_from_initial_turns=9.0), "terminal measurements"),
    ("workbench.json", lambda d: d["staged_evaluation"]["staged_result"]["stage_flags"].update(s1=True), "displayed stages"),
    ("workbench.json", lambda d: d["task_summary"][0].update(value=3600.0), "terminal net angle"),
    ("workbench.json", lambda d: d["unified_identity"].update(native_success=True), "identity"),
    ("visualization.json", lambda d: d["task_series"][0]["values"].__setitem__(0, 8.0), "bind displayed visual"),
])
def test_refreshed_inventory_cannot_hide_new_detail_semantic_drift(snapshot, filename, change, message):
    catalog = json.loads((snapshot / "catalog.json").read_text())
    identifier = catalog["groups"][-1]["trials"][0]["id"]
    _edit(snapshot / identifier / filename, change)
    _inventory(snapshot)
    with pytest.raises(ValueError, match=message):
        additive.verify_additive_workbench(snapshot, verify_sources=False)


def test_wrong_chart_reference_is_rejected_even_if_visualization_hash_refreshed(snapshot):
    from evaluation_v1.protocol import content_hash

    catalog = json.loads((snapshot / "catalog.json").read_text())
    prefix = snapshot / catalog["groups"][-1]["trials"][0]["id"]
    _edit(prefix / "visualization.json", lambda d: d["task_series"][0]["goal_references"][0].update(target=100.0))
    visual = json.loads((prefix / "visualization.json").read_text())
    _edit(prefix / "report.json", lambda d: d["visualization"].update(sha256=content_hash(visual)))
    _inventory(snapshot)
    with pytest.raises(ValueError, match="chart reference differs"):
        additive.verify_additive_workbench(snapshot, verify_sources=False)


def test_missing_new_reference_download_is_rejected(snapshot):
    catalog = json.loads((snapshot / "catalog.json").read_text())
    identifier = catalog["groups"][-1]["trials"][0]["id"]
    _edit(snapshot / "artifact_manifest.json", lambda d: d.update(files=[
        row for row in d["files"] if row["path"] != identifier + "/goal_references.json"]))
    with pytest.raises(ValueError, match="required rich payload"):
        additive.verify_additive_workbench(snapshot, verify_sources=False)


def test_append_export_refuses_existing_output_before_loading_any_source(snapshot):
    with pytest.raises(ValueError, match="fresh sibling"):
        additive.export_additive_workbench(snapshot.parent / "baseline", snapshot,
            job_root=snapshot, evaluation_root=snapshot.parent)
