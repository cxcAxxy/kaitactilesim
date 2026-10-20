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
exporter = importlib.import_module("evaluation_status.experiment_export")
parent_exporter = importlib.import_module("evaluation_status.export")


def fixture(tmp_path, monkeypatch):
    from test_evaluation_status_export import fixture as parent_fixture

    source, site, historical, _ = parent_fixture(tmp_path, monkeypatch)
    status = site / "status"
    parent_exporter.export_status(source, status, historical)
    catalog = json.loads((status / "catalog.json").read_text())
    good = catalog["groups"][0]
    good["trials"][0]["detail_url"] = "../status/" + good["trials"][0]["detail_url"]
    good.update(native_known_count=1, native_success_count=0)
    bad = copy.deepcopy(good)
    bad.update(id="quarantine-group", checkpoint="20000", mode="quarantined",
               publication_status="quarantined_replay_only", valid_trials=0, invalid_trials=1,
               native_known_count=0, native_success_count=0, stages=[], complete_stage_known_count=0)
    bad["trials"] = [dict(id="quarantined-trial", seed=0, valid=False, native_success=None,
        summary_sha256="f" * 64, publication_status="quarantined_replay_only",
        historical_detail_url="../preserved/case/report.html", quarantine_reason="<unsafe> publication failure")]
    catalog["groups"].append(bad)
    catalog["experiment"] = dict(member_trials=2, eligible_trials=1, quarantined_trials=1)
    monkeypatch.setitem(sys.modules, "evaluation_status.experiment",
                        SimpleNamespace(build_experiment_catalog=lambda *_: copy.deepcopy(catalog)))
    package = tmp_path / "experiment-package"
    (package / "web").mkdir(parents=True)
    (package / "web/experiment.js").write_text("// synthetic experiment UI")
    (package / "web/experiment.css").write_text("/* synthetic experiment style */")
    monkeypatch.setattr(exporter, "PACKAGE", package)
    return site, status, historical


def test_checkpoint_view_preserves_both_sites_and_quarantines_replay(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    previous = {str(p):p.read_bytes() for root in (status, historical) for p in root.rglob("*") if p.is_file()}
    output = site / "experiment"
    result = exporter.export_experiment(status, historical, output, publish_index=True)
    assert result["trials"] == 2 and result["eligible_trials"] == 1 and result["quarantined_trials"] == 1
    assert exporter.verify_experiment(output)["root_portal_matches_this_snapshot"] is True
    assert {str(p):p.read_bytes() for root in (status, historical) for p in root.rglob("*") if p.is_file()} == previous
    catalog = json.loads((output / "catalog.json").read_text())
    bad = catalog["groups"][1]["trials"][0]
    wrapper = output / bad["detail_url"]
    assert "仅回看，不计成绩" in wrapper.read_text()
    assert "&lt;unsafe&gt;" in wrapper.read_text() and "<unsafe>" not in wrapper.read_text()
    assert (output / "egotouch.html").is_file()
    assert catalog["publication"]["media_copies"] == 0


def test_corrupted_parent_blocks_experiment_publication(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    (status / "index.html").write_text("changed parent")
    before = (site / "index.html").read_bytes()
    with pytest.raises(ValueError, match="Archived bytes changed"):
        exporter.export_experiment(status, historical, site / "experiment", publish_index=True)
    assert (site / "index.html").read_bytes() == before
    assert not (site / "experiment").exists()


def test_detail_must_be_a_verified_dependency(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    rogue = site / "rogue.html"
    rogue.write_text("not inventoried")
    build = sys.modules["evaluation_status.experiment"].build_experiment_catalog

    def unsafe(*args):
        catalog = build(*args)
        catalog["groups"][0]["trials"][0]["detail_url"] = "../rogue.html"
        return catalog

    monkeypatch.setattr(sys.modules["evaluation_status.experiment"], "build_experiment_catalog", unsafe)
    with pytest.raises(ValueError, match="not in a verified snapshot"):
        exporter.export_experiment(status, historical, site / "experiment", publish_index=True)
    assert (site / "index.html").read_bytes() == b"original root portal"


def test_experiment_export_does_not_overwrite_existing_directory(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    output = site / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh canonical"):
        exporter.export_experiment(status, historical, output)


def test_membership_count_mismatch_never_switches_portal(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    build = sys.modules["evaluation_status.experiment"].build_experiment_catalog

    def mismatch(*args):
        catalog = build(*args)
        catalog["experiment"]["eligible_trials"] = 2
        return catalog

    monkeypatch.setattr(sys.modules["evaluation_status.experiment"], "build_experiment_catalog", mismatch)
    with pytest.raises(ValueError, match="counters disagree"):
        exporter.export_experiment(status, historical, site / "experiment", publish_index=True)
    assert (site / "index.html").read_bytes() == b"original root portal"


def test_detail_path_rejects_outside_or_active_content(tmp_path):
    output = tmp_path / "reports/experiment"
    output.mkdir(parents=True)
    for href in ("javascript:alert(1)", "https://example.com/", "/absolute/path", "../x?query", "x\\y"):
        with pytest.raises(ValueError):
            exporter._target(output, href)


def test_independent_experiment_links_must_also_be_verified(tmp_path, monkeypatch):
    site, status, historical = fixture(tmp_path, monkeypatch)
    (site / "rogue.html").write_text("not inventoried")

    def rogue_ego(*_):
        return dict(groups=[dict(trials=[dict(detail_url="../rogue.html")])]), exporter.digest(status / "catalog.json")

    monkeypatch.setattr(exporter, "_ego_catalog", rogue_ego)
    with pytest.raises(ValueError, match="Independent experiment target"):
        exporter.export_experiment(status, historical, site / "experiment", publish_index=True)
    assert (site / "index.html").read_bytes() == b"original root portal"
