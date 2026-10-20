"""Small artifact-only regression tests; no simulator or model is imported."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from evaluation_status.rich_workbench_export import (  # noqa: E402
    SCHEMA,
    _audit_page,
    _identity,
    _merge_inputs,
    score_projection,
    verify_rich_workbench,
)
from evaluation_status.unified_workbench_export import (  # noqa: E402
    SCHEMA as CATALOG_SCHEMA,
)
from evaluation_status.unified_workbench_export import (  # noqa: E402
    _counts,
    _stats,
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _digest(path, *, root=None):
    return {"path": path.relative_to(root).as_posix() if root else str(path),
            "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _inventory(output):
    rows = [_digest(path, root=output) for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"]
    _json(output / "artifact_manifest.json", {"schema": SCHEMA, "files": rows})


def _catalog():
    groups = []
    for model, ident, task, horizon, execution, status in (
        ("ftp1", "ftp", "poker-draw", 32, 5, "valid"),
        ("pi05+cpm", "cpm", "poker-draw", 30, 5, "invalid"),
        ("egotouch", "ego", "pick-place", 50, 50, "unknown"),
        ("pi05", "replay", "whiteboard-wipe-horizontal-line", 50, 50, "quarantined"),
    ):
        quarantine = ident == "replay"
        stage_valid = status == "valid"
        trial = {"id": ident, "seed": 0, "valid": not quarantine,
                 "native_success": None if quarantine else model == "egotouch",
                 "publication_status": "quarantined_replay_only" if quarantine else "eligible",
                 "staged_evidence_status": status,
                 "staged_evidence_basis": "tactile_manifest_inventory_v1" if stage_valid else None,
                 "stages": [{"key": "stage1", "name": "dragged_on_table", "reached": True},
                            {"key": "stage2", "name": "stable_grasp", "reached": False}] if stage_valid else [],
                 "complete_stage_success": False if stage_valid else None,
                 "continuous_metrics": {"a_recorded_quantity": 0.25} if stage_valid else {},
                 "summary_sha256": hashlib.sha256(ident.encode()).hexdigest(),
                 "notes": ["recorded source warning"] if model == "pi05+cpm" else [],
                 "detail_url": f"../original/{ident}/report.html", "video_url": f"../original/{ident}/review.mp4"}
        group = {"id": "group-" + ident, "task": task, "model": model, "checkpoint": "best" if ident == "cpm" else "10000",
                 "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
                 "prediction_horizon": horizon, "execute_steps": execution, "mode": "diagnostic",
                 "protocol": {"max_sim_seconds": 90.0, "guard_mode": "observe"},
                 "rule_identity": {"sha256": "a" * 64}, "trials": [trial]}
        group.update(_stats(group["trials"]))
        groups.append(group)
    return {"schema": CATALOG_SCHEMA, "generated_utc": "old-time", "groups": groups,
            "counts": _counts(groups), "historical_summary": {"not_counted_again": True, "rows": [{"count": 20}]},
            "provenance": [{"path": "retired-source", "sha256": "b" * 64}], "verification_scope": {"source_writes": 0},
            "links": [{"url": "../original/index.html"}]}


def _fixture(tmp_path):
    reports = tmp_path / "reports"
    baseline, output, evaluations = reports / "baseline", reports / "rich", tmp_path / "evaluations"
    evaluations.mkdir()
    original = _catalog()
    _json(baseline / "catalog.json", original)
    catalog = copy.deepcopy(original)
    for group in catalog["groups"]:
        for trial in group["trials"]:
            trial.update(detail_url=f"{trial['id']}/report.html", video_url=f"{trial['id']}/review.mp4",
                         rich_detail={"native_samples": 3})
            case = output / trial["id"]
            _json(case / "workbench.json", {"unified_identity": _identity(group, trial, group["model"] != "egotouch")})
            (case / "report.html").write_text("<!doctype html><title>fixture detail</title>")
            (case / "review.mp4").write_bytes(b"fixture-video")
    catalog.update(generated_utc="new-time", rich_details={"count": 4, "simulation_runs_started": 0})
    _json(output / "catalog.json", catalog)
    _json(output / "verification.json", {"schema": SCHEMA, "baseline_catalog": str(baseline / "catalog.json"),
                                         "evaluation_root": str(evaluations),
                                         "source_files": [_digest(baseline / "catalog.json")]})
    _inventory(output)
    return baseline, output


def _edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    _json(path, value)


def test_score_projection_ignores_only_presentation_fields_without_mutating_input():
    original = _catalog()
    before = copy.deepcopy(original)
    rich = copy.deepcopy(original)
    rich.update(generated_utc="new-time", provenance=[], verification_scope={"new": True},
                links=[], rich_details={"count": 4})
    for group in rich["groups"]:
        for trial in group["trials"]:
            trial.update(detail_url="new/report.html", video_url="new/video.mp4", rich_detail={"native_samples": 3})
    assert score_projection(original) == score_projection(rich)
    assert original == before


@pytest.mark.parametrize("change", [
    lambda c: c["counts"].update(eligible=99),
    lambda c: c["historical_summary"]["rows"][0].update(count=21),
    lambda c: c["groups"][0].update(model="pi05"),
    lambda c: c["groups"][0].update(checkpoint_sha256="c" * 64),
    lambda c: c["groups"][0].update(prediction_horizon=50),
    lambda c: c["groups"][0]["protocol"].update(guard_mode="strict"),
    lambda c: c["groups"][0]["rule_identity"].update(sha256="d" * 64),
    lambda c: c["groups"][0]["trials"][0].update(valid=False),
    lambda c: c["groups"][0]["trials"][0].update(native_success=True),
    lambda c: c["groups"][0]["trials"][0].update(summary_sha256="e" * 64),
    lambda c: c["groups"][0]["trials"][0].update(staged_evidence_basis="new-basis"),
    lambda c: c["groups"][0]["trials"][0].update(complete_stage_success=True),
    lambda c: c["groups"][0]["trials"][0]["stages"][0].update(reached=False),
    lambda c: c["groups"][0]["trials"][0]["continuous_metrics"].update(a_recorded_quantity=0.75),
    lambda c: c["groups"][1]["trials"][0].update(staged_evidence_status="valid"),
    lambda c: c["groups"][3]["trials"][0].update(publication_status="eligible"),
    lambda c: c["groups"][0]["trials"][0].update(future_non_display_fact=True),
])
def test_score_projection_preserves_every_admission_protocol_and_recorded_fact(change):
    original = _catalog()
    edited = copy.deepcopy(original)
    change(edited)
    assert score_projection(edited) != score_projection(original)


def test_merge_inputs_deduplicates_only_identical_file_identity():
    first = {"path": "/explicit/artifact", "size": 5, "sha256": "a" * 64, "role": "first"}
    inputs = {}
    _merge_inputs(inputs, [first, {**first, "role": "another-display-role"}])
    assert inputs == {first["path"]: first}


@pytest.mark.parametrize("model,expected", [
    ("pi05", "当前记录已隔离"),
    ("pi05+cpm", "来源校验未通过"),
    ("ftp1", "未重跑或重判历史结果"),
])
def test_original_audit_has_visible_current_admission_context_without_mutating_raw_report(model, expected):
    group = next(group for group in _catalog()["groups"] if group["model"] == model)
    group["title"] = "<script>untrusted-title()</script>"
    report = {"trial": {"task": group["task"], "trial_id": "raw-record-id"},
              "outcome": {"status": "pass", "recorded_success": True}}
    before = copy.deepcopy(report)
    html = _audit_page(report, group, group["trials"][0])
    assert 'class="rich-audit-context"' in html
    assert expected in html and "原始记录与诊断审计" in html
    assert 'href="report.html"' in html and 'href="../index.html"' in html
    assert "&lt;script&gt;untrusted-title()&lt;/script&gt;" in html
    assert "<script>untrusted-title()" not in html
    assert report == before and report["outcome"]["recorded_success"] is True


@pytest.mark.parametrize("change", [{"size": 6}, {"sha256": "b" * 64}])
def test_merge_inputs_rejects_conflicting_size_or_sha(change):
    first = {"path": "/explicit/artifact", "size": 5, "sha256": "a" * 64}
    with pytest.raises(ValueError, match="identity changed"):
        _merge_inputs({first["path"]: first}, [{**first, **change}])


def test_minimal_rich_snapshot_verifies_without_model_or_simulator(tmp_path):
    _, output = _fixture(tmp_path)
    result = verify_rich_workbench(output)
    assert result["status"] == "passed"
    assert result["members"] == 4 and result["eligible"] == 3 and result["quarantined"] == 1
    assert result["score_admissions_preserved"] is True and result["simulation_runs_started"] == 0


@pytest.mark.parametrize("relative", ["../outside.json", "/tmp/outside.json"])
def test_verification_rejects_inventory_escape(tmp_path, relative):
    _, output = _fixture(tmp_path)
    _edit(output / "artifact_manifest.json", lambda d: d["files"].append({"path": relative, "size": 1, "sha256": "a" * 64}))
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        verify_rich_workbench(output)


def test_verification_rejects_leaf_symlink(tmp_path):
    _, output = _fixture(tmp_path)
    original = output / "ftp/report.html"
    outside = tmp_path / "outside.html"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        verify_rich_workbench(output)


@pytest.mark.parametrize("change", [
    lambda d: d["unified_identity"].update(model="pi05"),
    lambda d: d["unified_identity"].update(prediction_horizon=50),
    lambda d: d["unified_identity"].update(native_success=True),
    lambda d: d["unified_identity"].update(staged_evidence_status="invalid"),
    lambda d: d["unified_identity"].update(group_id="another-group"),
])
def test_verification_rejects_detail_identity_drift_even_with_fresh_inventory(tmp_path, change):
    _, output = _fixture(tmp_path)
    _edit(output / "ftp/workbench.json", change)
    _inventory(output)
    with pytest.raises(ValueError, match="detail identity differs"):
        verify_rich_workbench(output)


def test_verification_rejects_native_result_change_with_consistent_counters(tmp_path):
    _, output = _fixture(tmp_path)
    def promote(catalog):
        group = catalog["groups"][0]
        group["trials"][0]["native_success"] = True
        group.update(_stats(group["trials"]))
        catalog["counts"] = _counts(catalog["groups"])
    _edit(output / "catalog.json", promote)
    _inventory(output)
    with pytest.raises(ValueError, match="baseline score admissions"):
        verify_rich_workbench(output)


def test_verification_rejects_counter_drift(tmp_path):
    _, output = _fixture(tmp_path)
    _edit(output / "catalog.json", lambda c: c["counts"].update(eligible=4))
    _inventory(output)
    with pytest.raises(ValueError, match="membership counts"):
        verify_rich_workbench(output)


def test_verification_rejects_source_path_outside_admitted_roots(tmp_path):
    _, output = _fixture(tmp_path)
    unrelated = tmp_path / "unrelated.json"
    _json(unrelated, {"not": "an admitted source"})
    _edit(output / "verification.json", lambda v: v["source_files"].append(_digest(unrelated)))
    _inventory(output)
    with pytest.raises(ValueError, match="escapes admitted"):
        verify_rich_workbench(output)


def test_verification_detects_changed_source_and_partial_publication(tmp_path):
    baseline, output = _fixture(tmp_path)
    _edit(baseline / "catalog.json", lambda c: c.update(generated_utc="changed"))
    with pytest.raises(ValueError, match="Archived bytes changed"):
        verify_rich_workbench(output)
    # Restore the source hash without changing any scored fact, then simulate
    # an interrupted root-index transaction; it must never verify as complete.
    _edit(output / "verification.json", lambda v: v.update(source_files=[_digest(baseline / "catalog.json")]))
    _inventory(output)
    _json(output / "portal_publication.pending.json", {"state": "pending"})
    with pytest.raises(ValueError, match="Incomplete rich root publication"):
        verify_rich_workbench(output)
