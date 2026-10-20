import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from evaluation_status.unified_workbench_export import (  # noqa: E402
    SCHEMA,
    build_unified_catalog,
    export_unified_workbench,
    verify_unified_workbench,
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _file(path, value="fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root, *, absolute=False):
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append({"path": str(Path("/retired/v1/reports") / root.name / path.relative_to(root))
                         if absolute else path.relative_to(root).as_posix(),
                         "size": path.stat().st_size, "sha256": _hash(path)})
    _json(root / "artifact_manifest.json", {"schema": "fixture-published-artifacts", "files": rows})


def _result(*, complete=False):
    return {"stage_definitions": {"stage1": "dragged_on_table", "stage2": "stable_grasp"},
            "stage_flags": {"stage1": True, "stage2": False}, "complete_success": complete,
            "continuous_metrics": {"cleaned_fraction": 0.25,
                                   "contact_time_weighted_mean_effective_area_ratio": 0.5}}


def _archive_group(model="pi05", *, quarantined=False):
    ident = "quarantine" if quarantined else model
    valid = not quarantined
    trial = {"id": ident, "seed": 0, "valid": valid, "native_success": False if model == "pi05" else True,
             "status": "task_not_completed", "summary_sha256": hashlib.sha256(ident.encode()).hexdigest(),
             "detail_url": ("quarantine/index.html" if quarantined else "../native_status/trials/pi05/index.html")
             if model == "pi05" else "../native_status/trials/egotouch/index.html",
             "video_url": "trials/pi05/review.mp4" if model == "pi05" and not quarantined else None,
             "publication_status": "quarantined_replay_only" if quarantined else "eligible",
             "staged_result": _result() if valid and model == "pi05" else None,
             "summary_path": "/retired/producer/summary.json", "video_path": "/retired/producer/review.mp4"}
    if quarantined:
        trial["historical_video_url"] = "../original360/old/media/rollout.mp4"
        trial["recorded_native_success"] = True
    return {"id": ident, "task": "poker-draw" if model == "pi05" else "pick-place", "title": "card" if model == "pi05" else "pick",
            "model": model, "checkpoint": "20000" if quarantined else "24999", "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
            "prediction_horizon": 50, "execute_steps": 50, "mode": "diagnostic", "batch_id": "legacy-batch",
            "protocol": {"max_sim_seconds": 90 if model == "pi05" else 50,
                         "action_interpolation": "linear", "rule_sha256": "a" * 64 if model == "pi05" else None},
            "planned_trials": 1, "valid_trials": int(valid), "invalid_trials": int(not valid),
            "native_success_count": int(valid and model == "egotouch"), "native_known_count": int(valid),
            "complete_stage_success_count": 0, "complete_stage_known_count": int(valid and model == "pi05"),
            "stages": [{"key": "stage1", "name": "dragged_on_table", "reached": 1, "known": 1},
                       {"key": "stage2", "name": "stable_grasp", "reached": 0, "known": 1}]
            if valid and model == "pi05" else [], "warnings": [], "trials": [trial]}


def _fixture(tmp_path):
    reports = tmp_path / "reports"
    old, h50, tactile, status = [reports / name for name in ("original360", "h50", "tactile", "native_status")]
    output = reports / "unified"
    historical = [{"task": "poker-draw", "title": "card", "model": "pi0.5", "checkpoint": "20000", "count": 1,
                   "native_success_count": 1, "complete_success_count": 1, "complete_success_defined": True,
                   "stages": [{"key": "stage1", "label": "drag", "reached": 1, "unknown": 0}]}]
    _file(reports / "index.html", "original root")
    _file(old / "index.html", "old workbench")
    _file(old / "old/media/rollout.mp4", "old video")
    _json(old / "batch.json", {"stage_groups": historical})
    _inventory(old, absolute=True)
    for model in ("pi05", "egotouch"):
        _file(status / f"trials/{model}/index.html", f"{model} detail")
        _file(status / f"trials/{model}/review.mp4", f"{model} video")
    _inventory(status)
    _file(h50 / "index.html", "h50 original")
    _file(h50 / "egotouch.html", "ego original")
    _file(h50 / "quarantine/index.html", "quarantined old replay")
    tasks = [{"task": "poker-draw", "title": "card"}, {"task": "pick-place", "title": "pick"}]
    hcat = {"schema": "kaihand-evaluation-status-v1", "tasks": tasks,
            "groups": [_archive_group(), _archive_group(quarantined=True)],
            "historical_url": "../original360/index.html",
            "experiment": {"schema": "kaihand-pi05-h50-checkpoint-experiment-v1",
                           "member_trials": 2, "eligible_trials": 1, "quarantined_trials": 1}}
    ecat = {"schema": "kaihand-evaluation-status-v1", "tasks": tasks, "groups": [_archive_group("egotouch")]}
    _json(h50 / "catalog.json", hcat)
    _json(h50 / "egotouch_catalog.json", ecat)
    _inventory(h50)
    _file(tactile / "index.html", "tactile original")
    groups = []
    for model, label in (("ftp1", "ftp1_step10000"), ("pi05+cpm", "cpm_best")):
        cpm = model == "pi05+cpm"
        path = tactile / f"trials/{label}/seed_0000"
        _file(path / "index.html", f"{model} detail")
        _file(path / "review.mp4", f"{model} video")
        _json(path / "summary.json", {"model_family": model, "seed": 0, "task_success": False})
        projection = {"staged_result": _result()}
        if not cpm:
            _json(path / "staged_result.json", projection)
        prefix = f"trials/{label}/seed_0000/"
        trial = {"id": label, "seed": 0, "observed": True, "status": "task_not_completed",
                 "evidence_status": "valid", "native_success": False, "reasons": ["checkpoint changed"] if cpm else [],
                 "staged_evidence_status": "invalid" if cpm else "verified", "staged_complete_success": None if cpm else False,
                 "recorded_staged_complete_success": True if cpm else False,
                 "detail_url": prefix + "index.html", "video_url": prefix + "review.mp4",
                 "staged_result_url": None if cpm else prefix + "staged_result.json",
                 "tactile_sent_to_model": True, "expert_used": cpm}
        groups.append({"label": label, "title": label,
                       "identity": {"model_family": model, "checkpoint_step": "best" if cpm else "10000",
                                    "checkpoint_sha256": hashlib.sha256(model.encode()).hexdigest(),
                                    "prediction_horizon": 30 if cpm else 32, "execute_steps": 5,
                                    "diagnostic_only": True, "formal_metrics_valid": False,
                                    "protocol": {"max_sim_seconds": 90, "action_interpolation": "zoh"}},
                       "counts": {"planned": 1, "observed": 1, "valid": 1, "invalid": 0, "missing": 0},
                       "native": {"successes": 0, "known": 1},
                       "staged": {"successes": 0, "known": 0 if cpm else 1, "rule_contract": None if cpm else {"binding_basis": "tactile_manifest_inventory_v1"},
                                  "stages": [] if cpm else [{"key": "stage1", "name": "dragged_on_table", "reached": 1, "known": 1},
                                                           {"key": "stage2", "name": "stable_grasp", "reached": 0, "known": 1}]},
                       "trials": [trial]})
    tcat = {"schema": "kaihand-tactile-policy-experiment-v1", "groups": groups,
            "experiment": {"id": "poker-n5", "task": "poker-draw", "counts": {"planned": 2, "valid": 2}}}
    _json(tactile / "catalog.json", tcat)
    _inventory(tactile)
    return h50, tactile, output


def _edit_catalog(root, name, edit):
    path = root / name
    value = json.loads(path.read_text())
    edit(value)
    _json(path, value)
    _inventory(root)


def test_unified_keeps_native_stage_protocol_and_archive_semantics(tmp_path):
    h50, tactile, output = _fixture(tmp_path)
    catalog = build_unified_catalog(h50, tactile, output)
    assert catalog["schema"] == SCHEMA
    assert catalog["counts"] == {"members": 5, "eligible": 4, "quarantined": 1, "staged_verified": 2,
                                  "staged_unknown": 3, "groups": 5, "tasks": 2, "models": 4, "invalid": 0}
    groups = {group["model"]: group for group in catalog["groups"] if group["valid_trials"]}
    assert [(groups[m]["prediction_horizon"], groups[m]["execute_steps"]) for m in ("pi05", "egotouch", "ftp1", "pi05+cpm")] == [(50, 50), (50, 50), (32, 5), (30, 5)]
    cpm = groups["pi05+cpm"]
    assert cpm["native_known_count"] == 1 and cpm["complete_stage_known_count"] == 0
    assert cpm["trials"][0]["stages"] == []
    assert cpm["trials"][0]["complete_stage_success"] is None
    assert cpm["trials"][0]["expert_used"] is True
    assert groups["egotouch"]["stages"] == []
    assert groups["pi05"]["trials"][0]["staged_evidence_basis"] == "recorded_archive"
    assert groups["ftp1"]["trials"][0]["staged_evidence_basis"] == "tactile_manifest_inventory_v1"
    assert groups["pi05"]["metrics"][0]["value"] == 25.0
    assert groups["pi05"]["metrics"][0]["display"] == "25.00"
    assert groups["pi05"]["metrics"][0]["unit"] == "%"
    assert groups["pi05"]["trials"][0]["video_url"] == "../native_status/trials/pi05/review.mp4"
    assert catalog["historical_summary"]["not_counted_again"] is True
    assert catalog["historical_summary"]["rows"][0]["native_success_count"] == 1
    assert not any("producer" in row["path"] for row in catalog["provenance"])
    assert catalog["verification_scope"]["live_model_weights_verified"] is False
    assert not output.exists()


@pytest.mark.parametrize("source,name,edit,match", [
    ("h50", "catalog.json", lambda c: c.update(schema="wrong"), "Unsupported"),
    ("h50", "catalog.json", lambda c: c["groups"][0].update(prediction_horizon=32), "H50"),
    ("h50", "catalog.json", lambda c: c["groups"][0].update(execute_steps=True), "H50"),
    ("h50", "catalog.json", lambda c: c["groups"][0].update(native_success_count=1), "native_success_count"),
    ("h50", "catalog.json", lambda c: c["groups"][0]["trials"][0].update(valid="true"), "admission"),
    ("h50", "catalog.json", lambda c: c["groups"][1]["trials"][0].update(valid=True), "Quarantined"),
    ("h50", "catalog.json", lambda c: c["experiment"].update(member_trials=3), "member_trials"),
    ("tactile", "catalog.json", lambda c: c.update(schema="wrong"), "Unsupported tactile"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["identity"].update(prediction_horizon=50), "horizon"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["identity"].update(execute_steps=50), "K5"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["identity"].update(formal_metrics_valid=True), "formal"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["native"].update(known=0), "native_known_count"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["counts"].update(observed=0), "observed count"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["counts"].update(invalid=1), "invalid count"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["staged"].update(known=0), "complete_stage_known_count"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["trials"][0].update(evidence_status="maybe"), "admission"),
    ("tactile", "catalog.json", lambda c: c["groups"][0]["trials"][0].update(staged_complete_success=True), "projection"),
])
def test_unified_rejects_changed_identity_admissions_and_counts(tmp_path, source, name, edit, match):
    h50, tactile, output = _fixture(tmp_path)
    _edit_catalog(h50 if source == "h50" else tactile, name, edit)
    with pytest.raises(ValueError, match=match):
        build_unified_catalog(h50, tactile, output)
    assert not output.exists()


@pytest.mark.parametrize("href", ["https://example.com/report.html", "/etc/passwd", "../../../escape.html",
                                  "..%2f..%2f..%2fetc/passwd", "..\\outside.html", "//example.com/report.html"])
def test_unified_rejects_unsafe_cross_snapshot_urls(tmp_path, href):
    h50, tactile, output = _fixture(tmp_path)
    _edit_catalog(h50, "catalog.json", lambda c: c["groups"][0]["trials"][0].update(detail_url=href))
    with pytest.raises((ValueError, FileNotFoundError)):
        build_unified_catalog(h50, tactile, output)


def test_unified_refuses_tampered_inventoried_media(tmp_path):
    h50, tactile, output = _fixture(tmp_path)
    (h50.parent / "native_status/trials/pi05/review.mp4").write_text("tampered")
    with pytest.raises(ValueError, match="Archived bytes changed"):
        build_unified_catalog(h50, tactile, output)


def test_unified_deduplicates_replica_without_recounting_old_360(tmp_path):
    h50, tactile, output = _fixture(tmp_path)
    def duplicate(catalog):
        group = copy.deepcopy(catalog["groups"][0])
        group["id"] += "-replica"
        catalog["groups"].append(group)
        catalog["experiment"].update(member_trials=3, eligible_trials=2)
    _edit_catalog(h50, "catalog.json", duplicate)
    catalog = build_unified_catalog(h50, tactile, output)
    assert catalog["counts"]["members"] == 5
    assert catalog["counts"]["eligible"] == 4
    assert len(catalog["duplicates"]) == 1
    assert len(catalog["historical_summary"]["rows"]) == 1


@pytest.mark.parametrize("change,match", [
    (lambda group: group["protocol"].update(guard_mode="enforce"), "identities"),
    (lambda group: group["protocol"].update(rule_sha256="b" * 64), "identities"),
    (lambda group: group["trials"][0]["staged_result"]["continuous_metrics"].update(cleaned_fraction=0.75), "admissions"),
])
def test_unified_rejects_duplicate_summary_protocol_rule_or_metrics_conflict(tmp_path, change, match):
    h50, tactile, output = _fixture(tmp_path)
    def duplicate(catalog):
        group = copy.deepcopy(catalog["groups"][0])
        group["id"] += "-replica"
        change(group)
        catalog["groups"].append(group)
        catalog["experiment"].update(member_trials=3, eligible_trials=2)
    _edit_catalog(h50, "catalog.json", duplicate)
    with pytest.raises(ValueError, match=match):
        build_unified_catalog(h50, tactile, output)


def test_unified_preserves_unknown_metric_denominator(tmp_path):
    h50, tactile, output = _fixture(tmp_path)
    def remove_metric(catalog):
        catalog["groups"][0]["trials"][0]["staged_result"]["continuous_metrics"].pop("cleaned_fraction")
    _edit_catalog(h50, "catalog.json", remove_metric)
    catalog = build_unified_catalog(h50, tactile, output)
    group = next(group for group in catalog["groups"] if group["id"] == "pi05_h50:pi05")
    assert [metric["key"] for metric in group["metrics"]] == ["contact_time_weighted_mean_effective_area_ratio"]
    assert group["metrics"][0]["known"] == 1


def test_unified_export_verifies_before_publishing_and_preserves_sources(tmp_path):
    h50, tactile, output = _fixture(tmp_path)
    before = {path: path.read_bytes() for root in (h50, tactile) for path in root.rglob("*") if path.is_file()}
    result = export_unified_workbench(h50, tactile, output, publish_index=True)
    assert result["status"] == "passed"
    assert result["members"] == 5
    assert result["root_portal_matches_this_snapshot"] is True
    assert (output / "previous_root_index.html").read_text() == "original root"
    root_text = (output.parent / "index.html").read_text()
    assert "unified/index.html" in root_text and "三个" not in root_text
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert verify_unified_workbench(output)["eligible"] == 4
    with pytest.raises(ValueError, match="fresh directory"):
        export_unified_workbench(h50, tactile, output)
    (tactile / "trials/cpm_best/seed_0000/review.mp4").write_text("modified")
    with pytest.raises(ValueError, match="Archived bytes changed"):
        verify_unified_workbench(output)
