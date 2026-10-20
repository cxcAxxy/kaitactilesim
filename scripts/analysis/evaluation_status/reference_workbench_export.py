"""Publish verified chart references as an additive immutable rich snapshot.

The parent is verified as published bytes, not against mutable live source code.
Only chart-reference annotations, terminal angle display metadata and UI assets
change. Recorded samples, outcomes, stages, video and plots remain unchanged.
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .export import PACKAGE, _copy, _publish_portal, _write, digest, encoded, verify_row
from .rich_workbench_export import (
    _audit_page,
    _identity,
    _merge_inputs,
    score_projection,
    verify_rich_workbench,
)
from .unified_workbench_export import _Artifacts, _read, _validate_catalog

SCHEMA = "kaihand-reference-rich-workbench-snapshot-v1"
_DETAIL_GENERATED = {"report.json", "visualization.json", "workbench.json", "report.html", "audit.html", "goal_references.json"}
_COMMON_GENERATED = {"catalog.json", "index.html", "previous_root_index.html", "verification.json", "artifact_manifest.json"}
_UI = {"assets/workbench.js": PACKAGE.parent / "evaluation_workbench/web/workbench.js",
       "assets/workbench.css": PACKAGE.parent / "evaluation_workbench/web/workbench.css"}


def _strip_report(value):
    result = copy.deepcopy(value)
    result.pop("goal_reference_extension", None)
    result.get("visualization", {}).pop("sha256", None)
    return result


def _strip_visual(value):
    result = copy.deepcopy(value)
    for series in result.get("task_series", []):
        series.pop("goal_references", None)
    return result


def _validate_delta(before, after):
    if _strip_report(before["report"]) != _strip_report(after["report"]):
        raise ValueError("Reference extension changed native report evidence")
    if _strip_visual(before["visualization"]) != _strip_visual(after["visualization"]):
        raise ValueError("Reference extension changed recorded series or media")
    if before["config"].get("goal_references"):
        if before["config"]["goal_references"] != after["config"].get("goal_references"):
            raise ValueError("Reference extension rewrote previously published reference metadata")
        for old_series, new_series in zip(before["visualization"].get("task_series", []), after["visualization"].get("task_series", []), strict=True):
            if old_series.get("goal_references") != new_series.get("goal_references"):
                raise ValueError("Reference extension rewrote previously published reference curves")
    a, b = copy.deepcopy(before["config"]), copy.deepcopy(after["config"])
    for config in (a, b):
        for key in ("goal_references", "task_summary", "reference_extension"):
            config.pop(key, None)
    if a != b:
        raise ValueError("Reference extension changed unrelated workbench configuration")
    # All original non-angle summaries must survive exactly, in order.
    angle_keys = {"clockwise_angle_deg", "angle_removed_deg", "rotation_from_initial_deg"}
    for key in ("clockwise_turns", "turns_removed", "rotation_from_initial_turns"):
        angle_keys.add(key)
    old = [row for row in before["config"].get("task_summary", []) if row.get("key") not in angle_keys]
    new = [row for row in after["config"].get("task_summary", []) if row.get("key") not in angle_keys]
    if old != new:
        raise ValueError("Reference extension changed non-angle terminal facts")


def _export_one(arguments):
    from evaluation_workbench.suite import _page

    from .rich_reference_data import enrich_case

    baseline, output, evaluation_root, group, trial, rows = arguments
    baseline, output = Path(baseline), Path(output)
    store = _Artifacts(baseline.parent, output)
    prefix = Path(trial["id"])
    source = baseline / prefix
    before = {key: store.document(source / name, "published_parent_payload") for key, name in (
        ("report", "report.json"), ("visualization", "visualization.json"), ("config", "workbench.json"))}
    loads = store.document(source / "tactile_loads.json", "published_parent_loads")
    if before["config"].get("unified_identity") != _identity(group, trial, before["config"].get("unified_identity", {}).get("stage_recorded")):
        raise ValueError("Parent rich identity differs from admitted catalog")
    case = enrich_case(before["report"], before["visualization"], before["config"], group, trial,
                       reports_root=baseline.parent, evaluation_root=Path(evaluation_root))
    _validate_delta(before, case)
    artifacts, inputs = [], dict(store.inputs)
    _merge_inputs(inputs, case["provenance"])
    for row in rows:
        relative = Path(row["path"])
        if relative.parent == prefix and relative.name in _DETAIL_GENERATED:
            continue
        _copy(output, relative, baseline / relative, artifacts, inputs, expected=row["sha256"])
    for name, key in (("report.json", "report"), ("visualization.json", "visualization"), ("workbench.json", "config")):
        _write(output, prefix / name, encoded(case[key]), artifacts)
    _write(output, prefix / "goal_references.json", encoded(case["config"]["goal_references"]), artifacts)
    _write(output, prefix / "report.html", _page(case["report"], case["visualization"], loads, case["config"]).encode(), artifacts)
    _write(output, prefix / "audit.html", _audit_page(case["report"], group, trial).encode(), artifacts)
    return {"trial_id": trial["id"], "artifacts": artifacts, "inputs": list(inputs.values()), "coverage": case["coverage"]}


def export_reference_workbench(baseline, output, *, evaluation_root, workers=4):
    from .unified_workbench_page import render_page

    baseline, output = Path(baseline).resolve(strict=True), Path(output).absolute()
    evaluation_root = Path(evaluation_root).resolve(strict=True)
    if output.exists() or output.is_symlink() or output.parent.resolve(strict=True) != baseline.parent:
        raise ValueError("Reference output must be a fresh sibling snapshot")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Choose 1..16 export workers")
    # Frozen bytes remain authoritative when the implementation has advanced.
    verify_rich_workbench(baseline, verify_sources=False)
    store = _Artifacts(baseline.parent, output)
    original = store.document(baseline / "catalog.json", "published_baseline_catalog")
    catalog = copy.deepcopy(original)
    inventory = store.inventory(baseline)
    rows_by_trial = {}
    for group in catalog["groups"]:
        for trial in group["trials"]:
            identifier = trial["id"]
            if trial["detail_url"] != identifier + "/report.html" or identifier in rows_by_trial or "/" in identifier or ".." in identifier:
                raise ValueError("Parent must have unique canonical episode-local detail paths")
            rows_by_trial[identifier] = []
    common = []
    for relative, row in inventory.items():
        parts = Path(relative).parts
        projected = {**row, "path": relative}
        if len(parts) > 1 and parts[0] in rows_by_trial:
            rows_by_trial[parts[0]].append(projected)
        else:
            common.append(projected)
    previous = (output.parent / "index.html").read_bytes() if (output.parent / "index.html").exists() else None
    output.mkdir()
    artifacts, inputs, coverage = [], dict(store.inputs), []
    arguments = [(str(baseline), str(output), str(evaluation_root), group, trial, rows_by_trial[trial["id"]])
                 for group in original["groups"] for trial in group["trials"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_export_one, arguments, chunksize=1), 1):
            artifacts.extend(result["artifacts"])
            _merge_inputs(inputs, result["inputs"])
            coverage.append({"trial_id": result["trial_id"], **result["coverage"]})
            if index % 25 == 0 or index == len(arguments):
                print(f"reference-details {index}/{len(arguments)}", flush=True)
    for row in common:
        relative = row["path"]
        if relative in _COMMON_GENERATED or relative in _UI:
            continue
        _copy(output, Path(relative), baseline / relative, artifacts, inputs, expected=row["sha256"])
    for relative, source in _UI.items():
        _write(output, Path(relative), source.read_bytes(), artifacts)
        _merge_inputs(inputs, [{**digest(source), "role": "reference_display_implementation"}])
    catalog["generated_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["rich_details"]["reference_extension"] = {"schema": SCHEMA, "baseline": baseline.name,
                                                   "members": len(coverage), "score_admissions_preserved": True}
    if score_projection(catalog) != score_projection(original):
        raise ValueError("Reference publication changed admitted catalog facts")
    _validate_catalog(catalog)
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    _write(output, Path("index.html"), render_page(catalog).encode(), artifacts)
    _write(output, Path("reference_coverage.json"), encoded({"schema": SCHEMA, "trials": coverage}), artifacts)
    if previous is not None:
        _write(output, Path("previous_root_index.html"), previous, artifacts)
    for source in (Path(__file__), PACKAGE / "rich_reference_data.py", PACKAGE.parent / "evaluation_workbench/goal_references.py",
                   PACKAGE.parent / "export_reference_evaluation_workbench.py"):
        _merge_inputs(inputs, [{**digest(source), "role": "reference_export_implementation"}])
    verification = {"schema": SCHEMA, "baseline_catalog": str(baseline / "catalog.json"),
                    "baseline_manifest_sha256": digest(baseline / "artifact_manifest.json")["sha256"],
                    "evaluation_root": str(evaluation_root), "score_admissions_preserved": True,
                    "recorded_samples_preserved": True, "simulation_runs_started": 0,
                    "native_outcomes_rescored": False, "staged_rules_reexecuted": False,
                    "model_weights_read": 0, "parent_verified_as": "immutable_published_bytes_not_mutable_live_implementation",
                    "source_files": sorted(inputs.values(), key=lambda row: row["path"])}
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SCHEMA, "files": artifacts}), [])
    return verify_reference_workbench(output)


def _safe_input(row, output, evaluation_root):
    path = Path(row["path"])
    if path.is_symlink() or not path.is_absolute():
        return False
    actual = path.resolve(strict=True)
    if any(actual.is_relative_to(root) for root in (output.parent, evaluation_root, PACKAGE.parent.resolve(strict=True))):
        return True
    # Only exact SHA-bound reviewed Python mechanics may be read elsewhere.
    # No checkpoint or simulator execution is part of this annotation export.
    return (row.get("role") == "goal_reference_reviewed_source" and actual.suffix == ".py"
            and "/src/kaihand_tactile_env/" in actual.as_posix()
            and actual.is_relative_to(Path("/cpfs_infra/user/chenxianchi")))


def verify_reference_workbench(output, *, verify_sources=True):
    output = Path(output).resolve(strict=True)
    manifest, _ = _read(output / "artifact_manifest.json")
    verification, _ = _read(output / "verification.json")
    if manifest.get("schema") != SCHEMA or verification.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference snapshot")
    paths = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        path = output / relative
        if relative.is_absolute() or ".." in relative.parts or row["path"] in paths or path.is_symlink() or not path.resolve(strict=True).is_relative_to(output):
            raise ValueError("Unsafe or duplicate reference artifact")
        paths.add(row["path"])
        verify_row(row, path)
    if verify_sources:
        for row in verification["source_files"]:
            if not _safe_input(row, output, Path(verification["evaluation_root"]).resolve(strict=True)):
                raise ValueError(f"Unadmitted reference source input: {row['path']}")
            verify_row(row)
    catalog, _ = _read(output / "catalog.json")
    original, _ = _read(verification["baseline_catalog"])
    baseline = Path(verification["baseline_catalog"]).parent
    if digest(baseline / "artifact_manifest.json")["sha256"] != verification["baseline_manifest_sha256"]:
        raise ValueError("Parent publication inventory changed")
    _validate_catalog(catalog)
    if score_projection(catalog) != score_projection(original):
        raise ValueError("Reference display changed original scores or membership")
    coverage, _ = _read(output / "reference_coverage.json")
    covered = [row["trial_id"] for row in coverage["trials"]]
    members = [trial["id"] for group in catalog["groups"] for trial in group["trials"]]
    if len(set(covered)) != len(covered) or set(covered) != set(members):
        raise ValueError("Reference coverage is not exactly the admitted membership")
    for group in catalog["groups"]:
        for trial in group["trials"]:
            prefix = Path(trial["id"])
            if trial["detail_url"] != (prefix / "report.html").as_posix() or trial["detail_url"] not in paths:
                raise ValueError("Reference member detail is not inventoried locally")
            if trial.get("video_url") and trial["video_url"] not in paths:
                raise ValueError("Reference member video is not inventoried locally")
            required = _DETAIL_GENERATED | {"tactile_loads.json"}
            if not all((prefix / name).as_posix() in paths for name in required):
                raise ValueError("Reference detail core payload or download is not inventoried")
            before, after = {}, {}
            for key, name in (("report", "report.json"), ("visualization", "visualization.json"), ("config", "workbench.json")):
                before[key], _ = _read(baseline / prefix / name)
                after[key], _ = _read(output / prefix / name)
            _validate_delta(before, after)
            if after["config"].get("unified_identity") != _identity(group, trial, after["config"].get("unified_identity", {}).get("stage_recorded")):
                raise ValueError("Reference member identity differs from catalog")
            references, _ = _read(output / prefix / "goal_references.json")
            if references != after["config"].get("goal_references"):
                raise ValueError("Reference download differs from displayed configuration")
    if (output / "portal_publication.pending.json").exists():
        raise ValueError("Incomplete reference root publication")
    published = None
    if (output / "portal_publication.json").exists():
        audit, _ = _read(output / "portal_publication.json")
        if audit.get("state") != "committed" or audit.get("root_index") != str(output.parent / "index.html"):
            raise ValueError("Invalid reference root publication audit")
        published = digest(output.parent / "index.html")["sha256"] == audit.get("new_sha256")
    return {"status": "passed", **catalog["counts"], "artifacts": len(paths), "dependencies": len(verification["source_files"]),
            "score_admissions_preserved": True, "recorded_samples_preserved": True,
            "root_portal_matches_this_snapshot": published, "simulation_runs_started": 0}


def publish_reference_workbench(output):
    output = Path(output).resolve(strict=True)
    result = verify_reference_workbench(output)
    previous_path = output / "previous_root_index.html"
    previous = previous_path.read_bytes() if previous_path.exists() else None
    relative = quote(output.name, safe="._-") + "/index.html"
    landing = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={relative}">'
               f'<title>KaiTactileSim 官方评估平台</title><p><a href="{relative}">进入 KaiTactileSim 统一评估工作台</a></p></html>').encode()
    _publish_portal(output, output.parent / "index.html", previous, landing)
    result["root_portal_matches_this_snapshot"] = digest(output.parent / "index.html")["sha256"] == digest_bytes(landing)
    if not result["root_portal_matches_this_snapshot"]:
        raise ValueError("Published root differs from the verified reference snapshot landing")
    return result


def digest_bytes(raw):
    import hashlib

    return hashlib.sha256(raw).hexdigest()

