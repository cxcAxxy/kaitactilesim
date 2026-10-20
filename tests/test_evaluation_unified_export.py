"""Synthetic boundaries for staged export; never open real evaluation outputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_unified import export, publish, report
from evaluation_workbench.export import _digest, _document, _verify
from evaluation_workbench.suite import SCHEMA


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def entry(*, rule="a", mode="diagnostic", cohort="run1", checkpoint_sha="b"):
    return {
        "task": "usb-insert", "title": "USB 插入", "model": "pi05", "checkpoint": "step20000",
        "mode": mode, "outcome": "fail", "cohort": {"checkpoint": checkpoint_sha * 64},
        "staged": {"rule_identity": {"sha256": rule * 64}, "source_cohort": cohort,
                   "complete_success": False, "native_result": {"task_success": False},
                   "stages": [{"key": "stage1", "name": "picked_up", "label": "拾取", "reached": True},
                              {"key": "stage2", "name": "inserted", "label": "插入", "reached": False}]},
    }


def test_grouping_is_rule_cohort_and_mode_specific_and_retains_failures():
    rows = [entry(), entry(), entry(rule="c"), entry(mode="formal"), entry(cohort="run2")]
    groups = export.stage_groups(rows)
    assert len(groups) == 4
    assert sum(group["count"] for group in groups) == len(rows)
    together = next(group for group in groups if group["count"] == 2)
    assert together["native_success_count"] == 0
    assert together["complete_success_count"] == 0
    assert together["stages"][0]["reached"] == 2
    assert together["stages"][1]["reached"] == 0


def test_grouping_matches_stage_keys_not_json_insertion_order():
    first, second = entry(), entry()
    second["staged"]["stages"].reverse()
    group, = export.stage_groups([first, second])
    counts = {stage["key"]: stage["reached"] for stage in group["stages"]}
    assert counts == {"stage1": 2, "stage2": 0}


def test_equal_checkpoint_folder_names_do_not_merge_different_weights():
    groups = export.stage_groups([entry(checkpoint_sha="b"), entry(checkpoint_sha="c")])
    assert len(groups) == 2


def test_conflicting_stage_definition_under_same_identity_rejected():
    first, second = entry(), entry()
    second["staged"]["stages"][0]["name"] = "different_recorded_condition"
    with pytest.raises(ValueError, match="conflicting stage definitions"):
        export.stage_groups([first, second])


def test_undefined_completion_and_unknown_stage_not_counted_as_failure():
    row = entry()
    row["staged"]["complete_success"] = None
    row["staged"]["stages"][0]["reached"] = None
    group, = export.stage_groups([row])
    assert group["complete_success_defined"] is False
    assert group["stages"][0]["unknown"] == 1
    assert group["stages"][0]["reached"] == 0


def test_native_success_count_uses_raw_label_not_independent_audit_status():
    row = entry()
    row["outcome"] = "unknown"
    row["staged"]["native_result"] = {"task_success": True, "physics_valid": False}
    group, = export.stage_groups([row])
    assert group["native_success_count"] == 1
    assert row["outcome"] == "unknown"


def staged_result():
    return {
        "native_result": {"task_success": False, "status": "task_not_completed", "diagnostic_only": True},
        "staged_result": {"version": "staged_v1", "stage_definitions": {"stage1": "archived_label"},
                          "stage_flags": {"stage1": True}, "first_times": {"stage1": 101.0},
                          "complete_success": True},
        "rule_identity": {"sha256": "f" * 64}, "data_evidence": {"status": "verified"}, "warnings": [],
        "state_rows": [{"simulation_time_s": time, "measurements": {"hand_normal_load_n": value}}
                       for time, value in ((100.0, 0), (101.0, None), (102.0, 2), (105.0, 3))],
    }


def test_projection_preserves_native_outcome_missing_values_clock_and_archived_labels():
    result = staged_result()
    before = deepcopy(result)
    projected = report.projection(result, 100.0)
    assert result == before
    assert projected["native_result"]["task_success"] is False
    assert projected["staged_result"]["complete_success"] is True
    assert projected["stages"][0]["name"] == "archived_label"
    assert projected["stages"][0]["first_time_s"] == 1.0
    timeline = projected["timeline"]
    index = next(i for i, item in enumerate(timeline["signals"]) if item["key"] == "hand_normal_load_n")
    assert [row[index] for row in timeline["values"]] == [0, None, 2, 3]
    assert timeline["time_s"] == [0, 1, 2, 5]
    assert timeline["gap_after"] == [False, False, True, False]


def test_projection_rejects_reordered_clock_instead_of_sorting():
    result = staged_result()
    result["state_rows"][1], result["state_rows"][2] = result["state_rows"][2], result["state_rows"][1]
    with pytest.raises(ValueError, match="increase"):
        report.projection(result, 100.0)


def test_augmentation_adds_evidence_without_replacing_native_metrics(tmp_path, monkeypatch):
    witness = tmp_path / "summary.json"
    write_json(witness, {"task_success": False})
    result = staged_result()
    result["provenance"] = [_digest(witness)]
    monkeypatch.setattr(report, "load_staged_result", lambda *_args, **_kwargs: result)
    trial = SimpleNamespace(root=tmp_path, time_s=[100, 101, 102, 105], task_tactile_clock_aligned=True)
    case = {"config": {}, "report": {"outcome": {"status": "fail", "diagnostic_only": True},
            "provenance": [_digest(witness)], "visualization": {"sha256": "old"},
            "task_specific": {"original": 7}}, "tactile_loads": {"normal_total_n": [0, None]},
            "visualization": {"time_origin_s": 100, "clock_aligned": True, "tactile_series": {"time_s": [0, 1]}}}
    original_outcome = deepcopy(case["report"]["outcome"])
    report.augment_staged_case(case, trial)
    assert case["report"]["outcome"] == original_outcome
    assert case["report"]["task_specific"] == {"original": 7}
    assert case["tactile_loads"]["normal_total_n"] == [0, None]
    assert case["config"]["staged_evaluation"]["native_result"]["task_success"] is False
    assert case["unified_staged"]["staged_result"]["complete_success"] is True


@pytest.mark.parametrize("clock,clock_aligned", [(None, False), ([7, 8, 9], False), ([], True)])
def test_unmapped_or_missing_tactile_keeps_independent_staged_clock(tmp_path, monkeypatch, clock, clock_aligned):
    witness = tmp_path / "summary.json"
    write_json(witness, {"task_success": False})
    result = staged_result()
    result["provenance"] = [_digest(witness)]
    monkeypatch.setattr(report, "load_staged_result", lambda *_args, **_kwargs: result)
    trial = SimpleNamespace(root=tmp_path, time_s=clock, task_tactile_clock_aligned=clock_aligned)
    case = {"config": {}, "report": {"outcome": {"status": "fail"}, "provenance": [_digest(witness)],
            "visualization": {"sha256": "old"}}, "tactile_loads": {"normal_total_n": []},
            "visualization": {"clock_aligned": False, "time_origin_s": None}}
    report.augment_staged_case(case, trial)
    timeline = case["config"]["staged_evaluation"]["timeline"]
    assert timeline["model_clock_mapping"] == "unmapped"
    assert timeline["time_s"] == [100, 101, 102, 105]
    assert timeline["origin_simulation_time_s"] is None
    assert case["report"]["outcome"]["status"] == "fail"


def test_original_snapshot_guard_detects_content_and_symlink_changes(tmp_path):
    source = tmp_path / "original.py"
    source.write_text("original bytes", encoding="utf-8")
    alias = tmp_path / "asset"
    alias.symlink_to("original.py")
    project = tmp_path / "unified"
    row = _digest(source)
    write_json(project / "integration_manifest.json", {"files": [
        {"source": str(source), "kind": "file", "size": row["size"], "sha256": row["sha256"]},
        {"source": str(alias), "kind": "symlink", "target": "original.py"},
    ]})
    assert export.verify_original_sources(project)["verified_original_files"] == 2
    source.write_text("changed bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        export.verify_original_sources(project)
    source.write_text("original bytes", encoding="utf-8")
    alias.unlink()
    alias.symlink_to("another.py")
    with pytest.raises(ValueError, match="link changed"):
        export.verify_original_sources(project)


@pytest.mark.parametrize("target", ["original_a", "original_b", "source", "previous", "extra", "alias"])
def test_forbidden_destinations_rejected_before_anchor_or_export(tmp_path, monkeypatch, target):
    project = tmp_path / "kaitactilesim_unified"
    source, previous, extra = [tmp_path / name for name in ("evaluation_source", "preserved", "extra_source")]
    original_a, original_b = tmp_path / "kaitactilesim", tmp_path / "kaitactilesim_laest"
    for path in (project, source, previous, extra, original_a, original_b):
        path.mkdir()
    alias = tmp_path / "alias_to_original"
    alias.symlink_to(original_a, target_is_directory=True)
    parents = {"original_a": original_a, "original_b": original_b, "source": source,
               "previous": previous, "extra": extra, "alias": alias}
    monkeypatch.setattr(export, "PROJECT", project)
    monkeypatch.setattr(export, "discover_staged_trials", lambda *_args, **_kwargs: {
        "included": [{"source": str(source / "seed_000"), "identity": {"task": "usb-insert"}}]})
    monkeypatch.setattr(export, "verify_original_sources", lambda: {})
    called = []

    def must_not_write(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("Unsafe output reached a filesystem-writing phase")

    monkeypatch.setattr(export, "baseline_anchor", must_not_write)
    monkeypatch.setattr(export, "export_suite", must_not_write)
    with pytest.raises(ValueError):
        export.export_staged(source, parents[target] / "new-report", extra_roots=[extra], previous=previous,
                             generate_model_plots=False)
    assert called == []
    assert not (parents[target] / "new-report").exists()
    assert not (parents[target] / "_preserved_h50").exists()


def make_publish_fixture(tmp_path, monkeypatch):
    output = tmp_path / "new-staged-site"
    output.mkdir()
    original = tmp_path / "untouched-original"
    original.write_text("original", encoding="utf-8")
    implementation = tmp_path / "current_source.py"
    implementation.write_text("implementation", encoding="utf-8")
    values = {"batch.json": {"schema_version": SCHEMA, "evaluation_profile": "recorded-staged",
                            "trial_count": 1, "task_count": 1, "trials": [entry()]},
              "verification.json": {"implementation": [_digest(implementation)]},
              "rollback.json": {"schema_version": "evaluation-experiment-rollback-v1",
                                "protected_files": [_digest(original)], "protected_roots": []}}
    for name, value in values.items():
        write_json(output / name, value)
    (output / "index.html").write_text("original index", encoding="utf-8")
    write_json(output / "artifact_manifest.json", {"files": [_digest(output / name) for name in
               ("batch.json", "verification.json", "rollback.json", "index.html")]})
    monkeypatch.setattr(publish, "implementation", lambda: [_digest(implementation)])
    monkeypatch.setattr(publish, "_portal", lambda _batch: "new index")
    return output, original


def test_finalization_updates_bound_metadata_and_preserves_sources(tmp_path, monkeypatch):
    output, original = make_publish_fixture(tmp_path, monkeypatch)
    groups = export.stage_groups([entry()])
    publish.publish_unified_metadata(output, groups, {"original_source_writes": 0})
    assert _document(output / "batch.json")["stage_groups"] == groups
    assert _document(output / "verification.json")["unified_import"]["native_outcome_preserved"] is True
    _verify(_document(output / "artifact_manifest.json")["files"])
    assert original.read_text() == "original"
    assert not (output / ".complete-model-plots.lock").exists()


def test_finalization_restores_all_bytes_on_partial_replace_failure(tmp_path, monkeypatch):
    output, original = make_publish_fixture(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    replace = publish.os.replace
    calls = []

    def fail_second(source, destination):
        calls.append((source, destination))
        if len(calls) == 2:
            raise OSError("synthetic replace failure")
        return replace(source, destination)

    monkeypatch.setattr(publish.os, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic replace failure"):
        publish.publish_unified_metadata(output, [], {})
    assert {path: path.read_bytes() for path in output.iterdir() if path.is_file()} == before
    assert original.read_text() == "original"


def make_relocated_alias_fixture(tmp_path):
    previous = tmp_path / "old-site"
    previous.mkdir()
    (previous / "assets").mkdir()
    (previous / "assets" / "site.js").write_text("original asset", encoding="utf-8")
    for name in ("batch.json", "artifact_manifest.json", "verification.json"):
        write_json(previous / name, {})
    (previous / "index.html").write_text("original old site", encoding="utf-8")
    old_storage, new_storage = tmp_path / "old-storage", tmp_path / "retired-storage"
    old_storage.mkdir()
    new_storage.mkdir()
    for directory in (old_storage, new_storage):
        (directory / "summary.json").write_bytes(b"old metadata")
    alias = tmp_path / "evaluation-alias"
    alias.symlink_to(new_storage, target_is_directory=True)
    old_file, new_file = old_storage / "summary.json", new_storage / "summary.json"
    row = {**_digest(old_file), "path": str(alias / "summary.json"), "resolved_path": str(old_file)}
    protected = tmp_path / "protected-source"
    protected.mkdir()
    (protected / "original.py").write_text("preserved", encoding="utf-8")
    rollback = {"schema_version": "evaluation-experiment-rollback-v1",
                "protected_files": [row, _digest(protected / "original.py")],
                "protected_roots": [{"path": str(protected), "files": [str(protected / "original.py")]}]}
    write_json(previous / "rollback.json", rollback)
    return previous, tmp_path / "new-output-parent", row, new_file, protected


def test_anchor_rebinds_identical_relocated_alias_with_audit_without_old_site_writes(tmp_path):
    previous, parent, row, relocated, _ = make_relocated_alias_fixture(tmp_path)
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    anchor = export.baseline_anchor(parent, previous)
    document = _document(anchor / "rollback.json")
    rebound = next(item for item in document["protected_files"] if item["path"] == row["path"])
    assert rebound["resolved_path"] == str(relocated)
    assert (rebound["sha256"], rebound["size"]) == (row["sha256"], row["size"])
    audit = document["prior_rollback_alias_rebindings"]
    assert len(audit) == 1
    audit_json = json.dumps(audit)
    assert row["path"] in audit_json and row["resolved_path"] in audit_json
    assert str(relocated) in audit_json and row["sha256"] in audit_json
    bound_original = next(item for item in document["protected_files"] if item["path"] == str(previous / "rollback.json"))
    assert bound_original == _digest(previous / "rollback.json")
    assert {path: path.read_bytes() for path in before} == before
    assert export.verify_rollback(anchor / "rollback.json")["verified"] is True


def test_anchor_rejects_changed_relocated_contents_even_with_identical_size(tmp_path):
    previous, parent, row, relocated, _ = make_relocated_alias_fixture(tmp_path)
    before = (previous / "rollback.json").read_bytes()
    relocated.write_bytes(b"new metadata")
    assert relocated.stat().st_size == row["size"]
    with pytest.raises(ValueError):
        export.baseline_anchor(parent, previous)
    assert (previous / "rollback.json").read_bytes() == before
    assert not (parent / "_preserved_h50").exists()


@pytest.mark.parametrize("bad_binding", ["no_resolved_path", "wrong_size", "missing_size"])
def test_anchor_cannot_rebind_without_complete_matching_old_evidence(tmp_path, bad_binding):
    previous, parent, _, _, _ = make_relocated_alias_fixture(tmp_path)
    rollback = _document(previous / "rollback.json")
    row = rollback["protected_files"][0]
    if bad_binding == "no_resolved_path":
        del row["resolved_path"]
    elif bad_binding == "missing_size":
        del row["size"]
    else:
        row["size"] += 1
    write_json(previous / "rollback.json", rollback)
    before = (previous / "rollback.json").read_bytes()
    with pytest.raises(ValueError):
        export.baseline_anchor(parent, previous)
    assert (previous / "rollback.json").read_bytes() == before
    assert not (parent / "_preserved_h50").exists()


def test_alias_rebinding_does_not_bypass_original_protected_inventory(tmp_path):
    previous, parent, _, _, protected = make_relocated_alias_fixture(tmp_path)
    (protected / "unexpected.py").write_text("changed inventory", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        export.baseline_anchor(parent, previous)
    assert not (parent / "_preserved_h50").exists()


def make_retired_source_fixture(tmp_path, monkeypatch):
    previous, parent, _, _, _ = make_relocated_alias_fixture(tmp_path)
    project = tmp_path / "workspace" / "code" / "kaitactilesim_unified"
    monkeypatch.setattr(export, "PROJECT", project)
    missing = project.parents[1] / "kaitactilesim_evaluations/_legacy_analysis/retired/seed_000/summary.json"
    rollback = _document(previous / "rollback.json")
    rollback["protected_files"].append({"path": str(missing), "sha256": "a" * 64, "size": 17})
    write_json(previous / "rollback.json", rollback)
    report_path = previous / "case_000" / "report.json"
    write_json(report_path, {"outcome": {"status": "fail"}, "original": True})
    (report_path.parent / "report.html").write_text("old generated report", encoding="utf-8")
    write_json(previous / "artifact_manifest.json", {"files": [
        _digest(path) for path in sorted(previous.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]})
    return previous, parent, missing, report_path


def test_retired_source_exception_is_explicit_and_old_site_is_fully_bound(tmp_path, monkeypatch):
    previous, parent, missing, _ = make_retired_source_fixture(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in previous.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, OSError)):
        export.baseline_anchor(parent, previous)
    assert not (parent / "_preserved_h50").exists()
    anchor = export.baseline_anchor(parent, previous, allow_retired_sources=True)
    assert anchor.name == "_preserved_h50_site_bound"
    document = _document(anchor / "rollback.json")
    assert document["prior_rollback_status"] == "historical_inputs_unavailable"
    missing_row, = document["prior_rollback_unavailable_sources"]
    assert missing_row["path"] == str(missing)
    assert missing_row["verification"] == "unavailable_before_integration"
    assert missing_row["path"] not in {row["path"] for row in document["protected_files"]}
    assert _digest(previous / "artifact_manifest.json") in document["protected_artifact_manifests"]
    assert {path: path.read_bytes() for path in before} == before
    assert export.verify_rollback(anchor / "rollback.json")["verified"] is True


@pytest.mark.parametrize("changed_target", ["artifact", "manifest"])
def test_protected_artifact_manifests_reject_content_tampering(tmp_path, monkeypatch, changed_target):
    previous, parent, _, report_path = make_retired_source_fixture(tmp_path, monkeypatch)
    anchor = export.baseline_anchor(parent, previous, allow_retired_sources=True)
    if changed_target == "artifact":
        report_path.write_text('{"outcome":{"status":"pass"}}', encoding="utf-8")
    else:
        write_json(previous / "artifact_manifest.json", {"files": []})
    with pytest.raises(ValueError, match="SHA256"):
        export.verify_rollback(anchor / "rollback.json")


def test_retired_source_exception_never_allows_missing_generated_report(tmp_path, monkeypatch):
    previous, parent, _, report_path = make_retired_source_fixture(tmp_path, monkeypatch)
    original_rollback = (previous / "rollback.json").read_bytes()
    report_path.unlink()
    with pytest.raises((ValueError, OSError)):
        export.baseline_anchor(parent, previous, allow_retired_sources=True)
    assert not (parent / "_preserved_h50_site_bound").exists()
    assert (previous / "rollback.json").read_bytes() == original_rollback


def test_retired_source_exception_does_not_cover_missing_resolved_alias(tmp_path, monkeypatch):
    previous, parent, missing, _ = make_retired_source_fixture(tmp_path, monkeypatch)
    rollback = _document(previous / "rollback.json")
    rollback["protected_files"][-1]["resolved_path"] = str(missing)
    write_json(previous / "rollback.json", rollback)
    with pytest.raises((ValueError, OSError)):
        export.baseline_anchor(parent, previous, allow_retired_sources=True)
    assert not (parent / "_preserved_h50_site_bound").exists()


def test_reader_alias_target_is_pinned_without_reresolving_during_projection(tmp_path, monkeypatch):
    source = tmp_path / "bound-source.json"
    write_json(source, {"task_success": False})
    source_row = _digest(source)
    result = staged_result()
    alias_path = tmp_path / "reader-alias" / "bound-source.json"
    result["provenance"] = [{**source_row, "path": str(alias_path), "canonical_path": str(source)}]
    monkeypatch.setattr(report, "load_staged_result", lambda *_args, **_kwargs: result)
    trial = SimpleNamespace(root=tmp_path, time_s=[100, 101], task_tactile_clock_aligned=True)
    case = {"config": {}, "report": {"outcome": {"status": "fail"}, "provenance": [source_row],
            "visualization": {"sha256": "old"}}, "tactile_loads": {},
            "visualization": {"clock_aligned": True, "time_origin_s": 100}}
    # There is deliberately no alias to resolve here: retain the reader's
    # captured target and let the later source verification check its validity.
    report.augment_staged_case(case, trial)
    row = next(item for item in case["report"]["provenance"] if item["path"] == str(alias_path))
    assert row["resolved_path"] == row["canonical_path"] == str(source)
