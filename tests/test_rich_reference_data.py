"""Reference recovery is annotation, never a policy or outcome re-evaluation."""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_status.rich_reference_data import enrich_case
from evaluation_v1.protocol import content_hash
from evaluation_workbench import goal_references as goals

PROJECT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    raw = path.read_bytes()
    return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def case(tmp_path):
    reports, evaluations = tmp_path / "reports", tmp_path / "evaluations"
    files = {}
    for name in goals._TASK_FILES["poker-draw"]:
        relative = goals._source_relative(name)
        path = tmp_path / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, path)
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": "poker_tactile_deployment_v1", "task": "poker-draw", "model_family": "ftp1",
                "checkpoint_sha256": "a" * 64, "prediction_horizon": 32, "execute_steps": 5, "files": files}
    manifest["deployment_id"] = hashlib.sha256(goals._canonical(manifest)).hexdigest()
    manifest_row = _write(evaluations / "deployment_manifest.json", manifest)
    summary = {key: manifest[key] for key in ("task", "model_family", "checkpoint_sha256", "prediction_horizon", "execute_steps", "deployment_id")}
    summary.update(seed=0, deployment_manifest=manifest_row["path"], deployment_manifest_sha256=manifest_row["sha256"], task_success=False)
    producer = _write(evaluations / "seed_0000" / "summary.json", summary)
    published = _write(reports / "seed_0000" / "summary.json", summary)
    report = {"trial": {"task": "poker-draw", "seed": 0, "prediction_horizon": 32},
              "outcome": {"task_success": False}, "recorded_staged_evaluation": {"status": "invalid"},
              "task_specific": {"partial_state": {}},
              "source_binding": {"published_summary": published, "producer_summary": producer}, "provenance": []}
    visual = {"task_series": [{"key": "card_displacement_now_mm", "unit": "mm", "values": [0, 62, 8], "time_s": [0, 1, 2]}]}
    config = {"task": "poker-draw", "task_summary": [], "unified_identity": {"trial_id": "trial-0", "staged_evidence_status": "invalid"}}
    group = {"task": "poker-draw", "model": "ftp1", "prediction_horizon": 32, "execute_steps": 5, "checkpoint_sha256": "a" * 64, "checkpoint": "15000"}
    trial = {"id": "trial-0", "seed": 0, "summary_sha256": published["sha256"], "staged_evidence_status": "invalid"}
    return {"report": report, "visualization": visual, "config": config, "group": group, "trial": trial,
            "reports_root": reports, "evaluation_root": evaluations}


def _rewrite_summary(case, edit):
    binding = case["report"]["source_binding"]
    summary = json.loads(Path(binding["producer_summary"]["path"]).read_text())
    edit(summary)
    for name in ("producer_summary", "published_summary"):
        binding[name] = _write(Path(binding[name]["path"]), summary)
    case["trial"]["summary_sha256"] = binding["published_summary"]["sha256"]


def test_tactile_h32_references_use_reviewed_mechanics_and_do_not_change_scores(case):
    before = copy.deepcopy(case)
    out = enrich_case(**case)
    assert case == before
    assert out["coverage"]["reference_status"] == "verified"
    assert out["config"]["goal_references"]["references"][0]["target"] == 50
    assert out["visualization"]["task_series"][0]["values"] == [0, 62, 8]
    assert out["report"]["outcome"] == before["report"]["outcome"]
    assert out["report"]["recorded_staged_evaluation"] == {"status": "invalid"}
    assert out["config"]["goal_references"]["binding"]["prediction_horizon"] == 32
    assert any("仍不计分" in item for item in out["config"]["goal_references"]["warnings"])
    assert out["report"]["visualization"]["sha256"] == content_hash(out["visualization"])
    assert out["report"]["visualization"]["sha256"] != content_hash(case["visualization"])


def test_updated_visual_binding_keeps_original_schema_path_and_semantics(case):
    original = {"schema_version": "evaluation-visualization-v1", "path": "visualization.json",
                "semantics": "Native display data; no resampling"}
    case["report"]["visualization"] = {**original, "sha256": content_hash(case["visualization"])}
    out = enrich_case(**case)
    assert out["report"]["visualization"] == {**original, "sha256": content_hash(out["visualization"])}


def test_tactile_h30_is_truthful_not_faked_h50(case):
    path = case["evaluation_root"] / "deployment_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(model_family="pi05+cpm", prediction_horizon=30)
    manifest.pop("deployment_id")
    manifest["deployment_id"] = hashlib.sha256(goals._canonical(manifest)).hexdigest()
    row = _write(path, manifest)
    _rewrite_summary(case, lambda s: s.update(model_family="pi05+cpm", prediction_horizon=30,
                     deployment_id=manifest["deployment_id"], deployment_manifest_sha256=row["sha256"]))
    case["group"].update(model="pi05+cpm", prediction_horizon=30)
    case["report"]["trial"]["prediction_horizon"] = 30
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "verified"
    assert out["config"]["goal_references"]["binding"]["prediction_horizon"] == 30


def test_legacy_source_inventory_and_pi05_alias_remain_source_bound(case):
    path = case["evaluation_root"] / "deployment_manifest.json"
    tactile = json.loads(path.read_text())
    inventory = [{"path": goals._source_relative(name), "size": Path(source).stat().st_size, "sha256": digest}
                 for name in goals._TASK_FILES["poker-draw"]
                 for source, digest in tactile["files"].items() if source.endswith("/" + goals._source_relative(name))]
    inventory_sha = hashlib.sha256(goals._canonical(inventory)).hexdigest()
    manifest = {"task": "poker-draw", "model_family": "pi0.5", "prediction_horizon": 50,
                "execute_steps": 50, "checkpoint_sha256": "a" * 64, "deployment_id": "b" * 64,
                "evaluation_source_files": inventory, "evaluation_source_sha256": inventory_sha}
    row = _write(path, manifest)
    source_root = Path(next(iter(tactile["files"]))).parents[4]
    _rewrite_summary(case, lambda s: s.update(model_family="pi0.5", prediction_horizon=50, execute_steps=50,
                     deployment_id="b" * 64, deployment_manifest_sha256=row["sha256"],
                     deployment_integrity={"status": "verified", "errors": [], "manifest_sha256": row["sha256"],
                        "evaluation_source": {"status": "verified", "root": str(source_root),
                                              "expected_sha256": inventory_sha, "actual_sha256": inventory_sha}}))
    case["group"].update(model="pi05", prediction_horizon=50, execute_steps=50)
    case["report"]["trial"]["prediction_horizon"] = 50
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "verified", out["config"]["goal_references"]
    assert out["config"]["goal_references"]["binding"]["evaluation_source_sha256"] == inventory_sha


@pytest.mark.parametrize("field,value", [("prediction_horizon", 50), ("execute_steps", 50), ("task", "usb-insert"), ("seed", 1), ("checkpoint_sha256", "b" * 64)])
def test_summary_identity_drift_has_no_reference_lines(case, field, value):
    _rewrite_summary(case, lambda s: s.update({field: value}))
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "unavailable"
    assert out["config"]["goal_references"]["references"] == []
    assert "goal_references" not in out["visualization"]["task_series"][0]


def test_manifest_bytes_cannot_change_even_if_task_constants_match(case):
    path = case["evaluation_root"] / "deployment_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "unavailable"


def test_rebound_manifest_must_have_valid_content_identity(case):
    path = case["evaluation_root"] / "deployment_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["execute_steps"] = 7
    row = _write(path, manifest)
    _rewrite_summary(case, lambda s: s.update(deployment_manifest_sha256=row["sha256"]))
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "unavailable"
    assert "identity digest" in out["config"]["goal_references"]["warnings"][0]


def test_reviewed_source_drift_is_not_current_default_fallback(case):
    manifest = json.loads((case["evaluation_root"] / "deployment_manifest.json").read_text())
    source = Path(next(iter(manifest["files"])))
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "unavailable"
    assert out["config"]["goal_references"]["references"] == []


def test_reformatted_equal_json_summary_is_allowed_but_field_drift_is_not(case):
    binding = case["report"]["source_binding"]
    path = Path(binding["published_summary"]["path"])
    raw = json.dumps(json.loads(path.read_text()), indent=2).encode()
    path.write_bytes(raw)
    binding["published_summary"].update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    case["trial"]["summary_sha256"] = binding["published_summary"]["sha256"]
    assert enrich_case(**case)["coverage"]["reference_status"] == "verified"
    value = json.loads(path.read_text())
    value["task_success"] = True
    binding["published_summary"] = _write(path, value)
    case["trial"]["summary_sha256"] = binding["published_summary"]["sha256"]
    assert enrich_case(**case)["coverage"]["reference_status"] == "unavailable"


def _bulb(case, task="bulb-screw", value=-0.1):
    source, key = {"bulb-screw": ("clockwise_turns", "clockwise_angle_deg"),
                   "bulb-unscrew": ("turns_removed", "angle_removed_deg"),
                   "bulb-screw-inplace": ("rotation_from_initial_turns", "rotation_from_initial_deg")}[task]
    case["group"]["task"] = case["report"]["trial"]["task"] = case["config"]["task"] = task
    case["report"]["task_specific"]["partial_state"] = {source: value}
    # Frozen original metadata is authoritative and never reread/replaced.
    case["config"]["goal_references"] = {"status": "verified", "references": [
        {"key": "clockwise_turns", "unit": "turn", "target": 100 / 360, "lower": 100 / 360 - .08,
         "upper": 100 / 360 + .08, "relation": "target", "label": "归档目标"}]}
    case["visualization"]["task_series"] = [{"key": source, "unit": "turn", "time_s": [0, 1], "values": [0, 1234]}]
    return source, key


@pytest.mark.parametrize("task", ["bulb-screw", "bulb-unscrew", "bulb-screw-inplace"])
def test_terminal_angle_uses_recorded_final_state_not_curve_endpoint(case, task):
    source, key = _bulb(case, task)
    before = copy.deepcopy(case)
    out = enrich_case(**case)
    row = next(row for row in out["config"]["task_summary"] if row["key"] == key)
    assert row["value"] == -36.0
    assert row["source"].endswith("partial_state." + source)
    assert out["visualization"]["task_series"][0]["values"] == [0, 1234]
    assert out["config"]["goal_references"] == before["config"]["goal_references"]
    assert out["report"]["task_specific"] == before["report"]["task_specific"]
    assert case == before
    if task == "bulb-screw-inplace":
        assert all(ref["key"] != "rotation_from_initial_deg" for ref in out["config"]["goal_references"]["references"])


@pytest.mark.parametrize("value", [None, True, "0.1", float("inf"), float("nan")])
def test_unknown_terminal_is_not_filled_from_last_curve_sample(case, value):
    _bulb(case, value=value)
    out = enrich_case(**case)
    row = out["config"]["task_summary"][0]
    assert row["value"] is None and row["status"] == "unknown"


def test_terminal_conflict_fails_instead_of_silently_replacing(case):
    _bulb(case)
    case["config"]["task_summary"] = [{"key": "clockwise_angle_deg", "unit": "deg", "value": 99}]
    with pytest.raises(ValueError, match="contradicts"):
        enrich_case(**case)


def test_ego_object_reference_not_applicable_and_reads_no_missing_source(case):
    case["group"].update(task="pick-place", model="egotouch")
    case["report"]["trial"]["task"] = case["config"]["task"] = "pick-place"
    case["report"].pop("source_binding")
    out = enrich_case(**case)
    assert out["coverage"]["reference_status"] == "not_applicable"
    assert out["provenance"] == []
    assert out["config"]["goal_references"]["reason"]
