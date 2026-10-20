from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
exporter = importlib.import_module("evaluation_status.export")


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture(tmp_path, monkeypatch, *, reused=False):
    source = tmp_path / "archive"
    trial_path = source / "bulb-screw/pi05/group/seed_000"
    summary = trial_path / "summary.json"
    _json(summary, dict(task="bulb-screw", task_success=False, evaluation=dict(state=dict(clockwise_turns=0.25))))
    (trial_path / "review.mp4").write_bytes(b"synthetic video bytes")
    site = tmp_path / "reports"
    historical = site / "preserved"
    historical.mkdir(parents=True)
    (site / "index.html").write_bytes(b"original root portal")
    _json(historical / "batch.json", dict(trials=[dict(trial_id="old", report_path="case/report.html")]))
    summary_hash = exporter.digest(summary)["sha256"]
    _json(historical / "source_manifest.json", dict(trials=[dict(trial_id="old",
        provenance=[dict(path="/old/seed_000/summary.json", sha256=summary_hash if reused else "f" * 64)])]))
    (historical / "index.html").write_text("old preserved index")
    (historical / "case").mkdir()
    (historical / "case/report.html").write_text("old preserved detailed workbench")
    rows = []
    for item in historical.rglob("*"):
        if item.is_file():
            row = exporter.digest(item)
            # Simulate the v1 absolute inventory copied intact into v2.
            row["path"] = str(Path("/retired/v1/reports/preserved") / item.relative_to(historical))
            rows.append(row)
    _json(historical / "artifact_manifest.json", dict(files=rows))
    trial = dict(id="trial_000", seed=0, status="task_not_completed", valid=True,
        native_success=False, sim_seconds=90, summary_path=str(summary), summary_sha256=summary_hash,
        video_path=str(trial_path / "review.mp4"), source=str(trial_path), staged_result=None,
        mode="diagnostic", detail_url=None)
    group = dict(id="group_000", task="bulb-screw", title="旋入灯泡", model="pi05", checkpoint="10000",
        prediction_horizon=50, execute_steps=50, mode="diagnostic", protocol=dict(guard_mode="observe"),
        valid_trials=1, planned_trials=1, trials=[trial], warnings=[])
    catalog = dict(schema="kaihand-evaluation-status-v1", tasks=[], groups=[group], jobs=[], excluded=[],
        provenance=[exporter.digest(summary)])
    monkeypatch.setitem(sys.modules, "evaluation_status.data", SimpleNamespace(build_catalog=lambda root: copy.deepcopy(catalog)))
    package = tmp_path / "package"
    (package / "web").mkdir(parents=True)
    (package / "web/status.js").write_text("// synthetic status UI")
    (package / "web/status.css").write_text("/* synthetic status style */")
    monkeypatch.setattr(exporter, "PACKAGE", package)
    return source, site, historical, catalog


@pytest.mark.parametrize("reused", [False, True])
def test_fresh_status_snapshot_and_portal_preserve_old_site(tmp_path, monkeypatch, reused):
    source, site, historical, _ = fixture(tmp_path, monkeypatch, reused=reused)
    previous = {str(path): path.read_bytes() for path in historical.rglob("*") if path.is_file()}
    output = site / "latest"
    result = exporter.export_status(source, output, historical, publish_index=True)
    assert result["trial_count"] == 1
    assert result["reused_preserved_details"] == int(reused)
    assert result["new_native_detail_pages"] == int(not reused)
    assert exporter.verify_status(output)["status"] == "passed"
    assert {str(path):path.read_bytes() for path in historical.rglob("*") if path.is_file()} == previous
    assert (output / "previous_root_index.html").read_bytes() == b"original root portal"
    assert b"latest/index.html" in (site / "index.html").read_bytes()
    assert json.loads((output / "verification.json").read_text())["retired_original_sources_verified"] is False
    catalog = json.loads((output / "catalog.json").read_text())
    if reused:
        assert catalog["groups"][0]["trials"][0]["detail_url"] == "../preserved/case/report.html"
    else:
        assert (output / "trials/trial_000/review.mp4").read_bytes() == b"synthetic video bytes"
        html = (output / "trials/trial_000/index.html").read_text()
        assert "90 °" in html and "原生任务：未完成" in html


def test_historical_artifact_change_prevents_publication(tmp_path, monkeypatch):
    source, site, historical, _ = fixture(tmp_path, monkeypatch)
    (historical / "case/report.html").write_text("changed old report")
    with pytest.raises(ValueError, match="Archived bytes changed"):
        exporter.export_status(source, site / "latest", historical, publish_index=True)
    assert (site / "index.html").read_bytes() == b"original root portal"
    assert not (site / "latest").exists()


def test_source_summary_change_cannot_reuse_or_publish_a_stale_identity(tmp_path, monkeypatch):
    source, site, historical, catalog = fixture(tmp_path, monkeypatch)
    Path(catalog["groups"][0]["trials"][0]["summary_path"]).write_text("{}")
    with pytest.raises(ValueError, match="Source changed during copy"):
        exporter.export_status(source, site / "latest", historical, publish_index=True)
    assert (site / "index.html").read_bytes() == b"original root portal"


def test_snapshot_manifest_path_cannot_escape(tmp_path, monkeypatch):
    source, site, historical, _ = fixture(tmp_path, monkeypatch)
    output = site / "latest"
    exporter.export_status(source, output, historical)
    _json(output / "artifact_manifest.json", dict(schema=exporter.SCHEMA,
        files=[dict(path="../index.html", size=0, sha256="0" * 64)]))
    with pytest.raises(ValueError, match="Unsafe snapshot manifest"):
        exporter.verify_status(output)


def test_export_never_overwrites_an_existing_snapshot(tmp_path, monkeypatch):
    source, site, historical, _ = fixture(tmp_path, monkeypatch)
    output = site / "latest"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh canonical"):
        exporter.export_status(source, output, historical)


def test_inert_json_and_native_facts_are_not_html_injection():
    assert "</script>" not in exporter.inert(dict(text="</script><img src=x>"))
    assert exporter._facts(dict(evaluation=dict(state=dict(turns_removed=0.5)))) == [
        ("旋出角度", "180 °")]


def test_native_degree_values_are_not_converted_again():
    assert exporter._facts(dict(evaluation=dict(state=dict(clockwise_angle_deg=90)))) == [("旋入角度", "90 °")]


def test_portal_audit_commit_failure_rolls_back_root(tmp_path, monkeypatch):
    output = tmp_path / "snapshot"
    output.mkdir()
    portal = tmp_path / "index.html"
    portal.write_bytes(b"before")
    original = exporter.os.replace

    def fail_audit(source, target):
        if Path(target).name == "portal_publication.json":
            raise OSError("synthetic audit commit failure")
        return original(source, target)

    monkeypatch.setattr(exporter.os, "replace", fail_audit)
    with pytest.raises(OSError, match="audit commit failure"):
        exporter._publish_portal(output, portal, b"before", b"after")
    assert portal.read_bytes() == b"before"
    assert not (tmp_path / ".evaluation-status-publish.lock").exists()


def test_portal_publish_lock_does_not_remove_another_publisher_lock(tmp_path):
    output = tmp_path / "snapshot"
    output.mkdir()
    lock = tmp_path / ".evaluation-status-publish.lock"
    lock.write_bytes(b"other publisher")
    with pytest.raises(FileExistsError):
        exporter._publish_portal(output, tmp_path / "index.html", None, b"after")
    assert lock.read_bytes() == b"other publisher"
