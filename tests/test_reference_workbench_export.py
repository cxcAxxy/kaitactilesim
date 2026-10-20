"""Artifact-only reference migration guards; never imports simulation or policy code."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from evaluation_status.reference_workbench_export import (  # noqa: E402
    SCHEMA,
    _validate_delta,
    verify_reference_workbench,
)
from evaluation_status.rich_workbench_export import _identity  # noqa: E402
from evaluation_status.unified_workbench_export import (  # noqa: E402
    SCHEMA as CATALOG_SCHEMA,
)
from evaluation_status.unified_workbench_export import _counts, _stats  # noqa: E402


def _case():
    return {
        "report": {"outcome": {"recorded_success": False}, "visualization": {"sha256": "a" * 64},
                   "recorded_staged_evaluation": {"success": False}, "provenance": [{"path": "/raw/summary"}]},
        "visualization": {"task_series": [{"key": "clockwise_turns", "unit": "turn", "time_s": [0, 1],
                                            "values": [0, .2], "goal_references": []}],
                          "media": {"url": "media/rollout.mp4"}},
        "config": {"prediction_horizon": 50, "unified_identity": {"native_success": False},
                   "task_summary": [{"key": "engaged", "value": True, "source": "raw"}],
                   "goal_references": {"status": "unavailable", "references": []}},
    }


def test_additive_reference_and_degree_metadata_preserves_original_payload():
    before = _case()
    before["config"].pop("goal_references")
    frozen = copy.deepcopy(before)
    after = copy.deepcopy(before)
    after["report"]["goal_reference_extension"] = {"status": "verified"}
    after["report"]["visualization"]["sha256"] = "b" * 64
    after["visualization"]["task_series"][0]["goal_references"] = [{"key": "clockwise_turns", "target": .3}]
    after["config"]["goal_references"] = {"status": "verified"}
    after["config"]["task_summary"].insert(0, {"key": "clockwise_angle_deg", "value": 72.0, "unit": "deg"})
    after["config"]["reference_extension"] = {"no_rescore": True}
    _validate_delta(before, after)
    assert before == frozen


@pytest.mark.parametrize("change", [
    lambda c: c["config"]["goal_references"].update(status="verified"),
    lambda c: c["visualization"]["task_series"][0]["goal_references"].append({"target": .3}),
])
def test_previously_published_goal_evidence_is_not_rewritten(change):
    before = _case()
    after = copy.deepcopy(before)
    change(after)
    with pytest.raises(ValueError, match="previously published"):
        _validate_delta(before, after)


@pytest.mark.parametrize("change,match", [
    (lambda c: c["report"]["outcome"].update(recorded_success=True), "native report"),
    (lambda c: c["report"]["recorded_staged_evaluation"].update(success=True), "native report"),
    (lambda c: c["report"]["provenance"].clear(), "native report"),
    (lambda c: c["visualization"]["task_series"][0]["values"].__setitem__(1, 0), "recorded series"),
    (lambda c: c["visualization"]["task_series"][0]["time_s"].__setitem__(1, 2), "recorded series"),
    (lambda c: c["visualization"]["task_series"][0].update(key="clockwise_angle_deg", unit="deg"), "recorded series"),
    (lambda c: c["visualization"]["media"].update(url="other.mp4"), "recorded series"),
    (lambda c: c["config"].update(prediction_horizon=30), "unrelated workbench"),
    (lambda c: c["config"]["unified_identity"].update(native_success=True), "unrelated workbench"),
    (lambda c: c["config"]["task_summary"][0].update(value=False), "non-angle terminal"),
    (lambda c: c["config"]["task_summary"].clear(), "non-angle terminal"),
])
def test_reference_extension_cannot_change_existing_measurement_or_result(change, match):
    before = _case()
    after = copy.deepcopy(before)
    change(after)
    with pytest.raises(ValueError, match=match):
        _validate_delta(before, after)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _digest(path, root=None):
    return {"path": str(path) if root is None else path.relative_to(root).as_posix(),
            "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _inventory(output):
    _write_json(output / "artifact_manifest.json", {"schema": SCHEMA, "files": [
        _digest(path, output) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"]})


def _edit(path, change):
    value = json.loads(path.read_text())
    change(value)
    _write_json(path, value)


def _fixture(tmp_path):
    parent, output, evaluations = tmp_path / "reports/baseline", tmp_path / "reports/references", tmp_path / "evaluations"
    evaluations.mkdir()
    trial = {"id": "trial-one", "seed": 0, "valid": True, "native_success": False,
             "publication_status": "eligible", "staged_evidence_status": "invalid", "stages": [],
             "complete_stage_success": None, "continuous_metrics": {}, "summary_sha256": "a" * 64,
             "detail_url": "trial-one/report.html", "video_url": "trial-one/media/rollout.mp4"}
    group = {"id": "group-one", "task": "poker-draw", "model": "pi05+cpm", "checkpoint": "best",
             "prediction_horizon": 30, "execute_steps": 5, "trials": [trial]}
    group.update(_stats(group["trials"]))
    catalog = {"schema": CATALOG_SCHEMA, "groups": [group], "counts": _counts([group]),
               "historical_summary": {"not_counted_again": True}, "rich_details": {"count": 1}}
    case = _case()
    case["config"]["unified_identity"] = _identity(group, trial, True)
    case["config"]["prediction_horizon"] = 30
    for directory in (parent, output):
        _write_json(directory / "catalog.json", catalog)
        for key, name in (("report", "report.json"), ("visualization", "visualization.json"), ("config", "workbench.json")):
            _write_json(directory / "trial-one" / name, case[key])
        _write_json(directory / "trial-one/goal_references.json", case["config"]["goal_references"])
        _write_json(directory / "trial-one/tactile_loads.json", {"trial_id": "trial-one", "time_s": [0, 1]})
        (directory / "trial-one/report.html").write_text("<title>Fixture</title>")
        (directory / "trial-one/audit.html").write_text("<title>Recorded audit</title>")
        (directory / "trial-one/media").mkdir()
        (directory / "trial-one/media/rollout.mp4").write_bytes(b"recorded-video")
    _inventory(parent)
    _write_json(output / "reference_coverage.json", {"schema": SCHEMA, "trials": [{"trial_id": trial["id"]}]})
    _write_json(output / "verification.json", {"schema": SCHEMA, "baseline_catalog": str(parent / "catalog.json"),
                "baseline_manifest_sha256": _digest(parent / "artifact_manifest.json")["sha256"],
                "evaluation_root": str(evaluations), "source_files": [_digest(parent / "catalog.json")]})
    _inventory(output)
    return parent, output


def test_reference_snapshot_verifies_without_simulator_or_live_checkpoint(tmp_path):
    _, output = _fixture(tmp_path)
    result = verify_reference_workbench(output)
    assert result["status"] == "passed" and result["members"] == 1
    assert result["recorded_samples_preserved"] is True and result["simulation_runs_started"] == 0


@pytest.mark.parametrize("relative,change,error", [
    ("trial-one/workbench.json", lambda d: d["unified_identity"].update(native_success=True), "unrelated workbench"),
    ("trial-one/visualization.json", lambda d: d["task_series"][0]["values"].__setitem__(1, 2), "recorded series"),
    ("trial-one/report.json", lambda d: d["outcome"].update(recorded_success=True), "native report"),
    ("trial-one/workbench.json", lambda d: d["task_summary"][0].update(value=False), "non-angle terminal"),
    ("trial-one/goal_references.json", lambda d: d.update(status="verified"), "download differs"),
    ("reference_coverage.json", lambda d: d["trials"].append({"trial_id": "trial-one"}), "exactly the admitted"),
    ("reference_coverage.json", lambda d: d["trials"].clear(), "exactly the admitted"),
])
def test_fresh_inventory_cannot_hide_semantic_drift(tmp_path, relative, change, error):
    _, output = _fixture(tmp_path)
    _edit(output / relative, change)
    _inventory(output)
    with pytest.raises(ValueError, match=error):
        verify_reference_workbench(output)


def test_changed_parent_inventory_is_rejected(tmp_path):
    parent, output = _fixture(tmp_path)
    _edit(parent / "artifact_manifest.json", lambda d: d.update(changed=True))
    with pytest.raises(ValueError, match="Parent publication inventory changed"):
        verify_reference_workbench(output)


def test_partial_publication_is_rejected(tmp_path):
    _, output = _fixture(tmp_path)
    _write_json(output / "portal_publication.pending.json", {"state": "pending"})
    with pytest.raises(ValueError, match="Incomplete reference root publication"):
        verify_reference_workbench(output)


@pytest.mark.parametrize("name", ["workbench.json", "report.json", "visualization.json", "goal_references.json", "tactile_loads.json", "audit.html"])
def test_core_payload_cannot_be_omitted_from_manifest(tmp_path, name):
    _, output = _fixture(tmp_path)
    _edit(output / "artifact_manifest.json", lambda d: d.update(files=[row for row in d["files"] if row["path"] != "trial-one/" + name]))
    with pytest.raises(ValueError, match="core payload or download"):
        verify_reference_workbench(output)


@pytest.mark.parametrize("relative", ["../outside.json", "/absolute.json"])
def test_manifest_paths_cannot_escape_snapshot(tmp_path, relative):
    _, output = _fixture(tmp_path)
    _edit(output / "artifact_manifest.json", lambda d: d["files"].append({"path": relative, "sha256": "a" * 64, "size": 1}))
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        verify_reference_workbench(output)
