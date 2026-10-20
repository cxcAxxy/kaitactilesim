"""Export recorded staged evaluations into a separate, source-bound website."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from evaluation_workbench.export import _digest, _document, _rows, _verify, _verify_inventory, verify_rollback
from evaluation_workbench.suite import export_suite, verify_suite
from .staged import discover_staged_trials


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_PREVIOUS = PROJECT.parent / "kaitactilesim_laest/reports/evaluation_v1_stage1_20261013/workbench_suite_h50_20261014"


def verify_original_sources(project=PROJECT):
    manifest = _document(Path(project) / "integration_manifest.json")
    checked = 0
    for row in manifest["files"]:
        source = Path(row["source"])
        if row["kind"] == "symlink":
            if not source.is_symlink() or str(source.readlink()) != row["target"]:
                raise ValueError(f"Original source link changed: {source}")
        else:
            _verify([{"path": str(source), "sha256": row["sha256"], "size": row["size"]}])
        checked += 1
    return {"verified_original_files": checked, "original_source_writes": 0}


def baseline_anchor(parent, previous=DEFAULT_PREVIOUS, *, allow_retired_sources=False):
    """Small local rollback anchor; never copy or rewrite the preserved site."""
    parent, previous = Path(parent).resolve(), Path(previous).resolve(strict=True)
    anchor = parent / ("_preserved_h50_site_bound" if allow_retired_sources else "_preserved_h50")
    if anchor.exists():
        verify_rollback(anchor / "rollback.json")
        if _document(anchor / "rollback.json").get("preserved_site") != str(previous):
            raise ValueError("Existing rollback anchor points to another preserved site")
        return anchor
    inherited = _document(previous / "rollback.json")
    if inherited.get("schema_version") != "evaluation-experiment-rollback-v1":
        raise ValueError("Not a supported previous rollback manifest")
    # Historical data aliases may have been moved after the old report was
    # frozen. Accept only identical SHA/size bytes, with the relocation fully
    # recorded in this NEW anchor; never rewrite the original rollback file.
    inherited_rows, rebindings, unavailable = [], [], []
    for original in _rows(inherited["protected_files"]):
        row = dict(original)
        if not Path(row["path"]).exists() and allow_retired_sources:
            # A previous independent cleanup can retire historical INPUTS.
            # Never pretend to have verified their missing bytes. The new
            # anchor still protects all persisted report trees and metadata.
            archive = PROJECT.parents[1] / "kaitactilesim_evaluations"
            if Path(row["path"]).is_relative_to(archive) and "resolved_path" not in row:
                unavailable.append({**row, "verification": "unavailable_before_integration"})
                continue
        if "resolved_path" in row:
            current = str(Path(row["path"]).resolve(strict=True))
            if current != row["resolved_path"]:
                if type(row.get("size")) is not int or row["size"] < 0:
                    raise ValueError("Relocating a historical alias requires SHA256 and size")
                rebindings.append({"path": row["path"], "previous_resolved_path": row["resolved_path"],
                                   "resolved_path": current, "sha256": row["sha256"], "size": row["size"]})
                row["resolved_path"] = current
        _verify([row])
        inherited_rows.append(row)
    _verify_inventory(inherited.get("protected_roots", []))
    artifact_manifests = inherited.get("protected_artifact_manifests", [])
    if allow_retired_sources:
        artifact_manifests = artifact_manifests + [_digest(previous / "artifact_manifest.json")]
        for row in _document(previous / "artifact_manifest.json")["files"]:
            if not Path(row["path"]).is_relative_to(previous):
                raise ValueError("Preserved-site artifact escapes the old report directory")
            _verify([row])
    watched = [previous / name for name in ("index.html", "batch.json", "artifact_manifest.json", "rollback.json", "verification.json")]
    watched += [p for p in (previous / "assets").iterdir() if p.is_file()]
    rows = _rows(inherited_rows + [_digest(p) for p in watched])
    _verify(rows)
    anchor.mkdir(parents=True)
    (anchor / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>保留的 H50 稳定版</title>'
        '<h1>原 H50 站点保持不变</h1><p>整合版在独立目录导入 staged 数据，不覆盖旧版。</p>'
        '<a href="http://127.0.0.1:8876/workbench_suite_h50_20261014/index.html">打开原 H50 多任务站点（8876）</a></html>', encoding="utf-8")
    rollback = {"schema_version": "evaluation-experiment-rollback-v1", "experiment": "unified-baseline-anchor-v1",
                "preserved_site": str(previous), "protected_files": rows,
                "protected_roots": inherited.get("protected_roots", []),
                "protected_artifact_manifests": artifact_manifests,
                "prior_rollback_alias_rebindings": rebindings,
                "prior_rollback_unavailable_sources": unavailable,
                "prior_rollback_status": "historical_inputs_unavailable" if unavailable else "content_verified_alias_rebound" if rebindings else "verified",
                "preservation_scope": "Persisted prior report trees and available SHA-bound historical sources; missing old inputs explicitly not verified",
                "strategy": "Separate integration project and new output; never write original simulation, analysis or H50 sites."}
    (anchor / "rollback.json").write_text(json.dumps(rollback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return anchor


def stage_groups(entries):
    groups = defaultdict(list)
    for trial in entries:
        staged = trial["staged"]
        checkpoint_identity = trial.get("cohort", {}).get("checkpoint")
        if not checkpoint_identity:
            raise ValueError("Stage grouping requires checkpoint content identity, not its display name")
        key = (trial["task"], trial["model"], checkpoint_identity, trial["mode"],
               staged["rule_identity"]["sha256"], staged["source_cohort"])
        groups[key].append(trial)
    output = []
    for key, trials in groups.items():
        example = trials[0]
        definitions = {s["key"]: s["name"] for s in example["staged"]["stages"]}
        by_key = []
        for trial in trials:
            cards = trial["staged"]["stages"]
            if len(cards) != len(definitions) or {s["key"]: s["name"] for s in cards} != definitions:
                raise ValueError("The same archived rule identity contains conflicting stage definitions")
            by_key.append({s["key"]: s for s in cards})
        stages = [{"key": s["key"], "label": s["label"],
                   "reached": sum(t[s["key"]]["reached"] is True for t in by_key),
                   "unknown": sum(t[s["key"]]["reached"] is None for t in by_key)}
                  for s in example["staged"]["stages"]]
        complete = [t["staged"]["complete_success"] for t in trials]
        output.append({"task": key[0], "title": example["title"], "model": key[1],
                       "checkpoint": example["checkpoint"], "checkpoint_identity": key[2],
                       "mode": key[3], "rule_sha256": key[4], "cohort": key[5], "count": len(trials),
                       "stages": stages, "complete_success_count": sum(x is True for x in complete),
                       "complete_success_defined": any(x is not None for x in complete),
                       "native_success_count": sum(t["staged"].get("native_result", {}).get("task_success") is True for t in trials),
                       "scope": "Recorded diagnostic counts within one task/model/checkpoint/cohort/rule identity; not a formal model ranking"})
    return output


def export_staged(source_root, output, *, extra_roots=(), previous=DEFAULT_PREVIOUS,
                  task_demo_root=Path("/nas/chenxianchi/datasets/sim/task_demo"), workers=4,
                  generate_model_plots=True, trial_limit=None):
    output = Path(output).absolute()
    extra_roots = tuple(extra_roots)
    if output.resolve() != output:
        raise ValueError("Unified report output must not use symlinks or noncanonical parent paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Export a fresh staged snapshot, never overwrite an existing report")
    if any(output == Path(p).resolve() or output.is_relative_to(Path(p).resolve()) for p in
           (source_root, *extra_roots, previous, PROJECT.parent / "kaitactilesim", PROJECT.parent / "kaitactilesim_laest")):
        raise ValueError("Unified reports cannot be written inside original projects or source evaluations")
    selection = discover_staged_trials(source_root, extra_roots=extra_roots)
    included = selection["included"]
    if trial_limit is not None:
        if type(trial_limit) is not int or trial_limit <= 0:
            raise ValueError("trial_limit must be positive")
        # A bounded preflight still shows all task types before taking repeats.
        first, later, seen = [], [], set()
        for item in included:
            task = item["identity"]["task"]
            (later if task in seen else first).append(item)
            seen.add(task)
        included = (first + later)[:trial_limit]
        selection["export_limit"] = trial_limit
    if not included:
        raise ValueError("No completed, identity-verified H50 staged trials were selected")
    source_guard = verify_original_sources()
    output.parent.mkdir(parents=True, exist_ok=True)
    anchor = baseline_anchor(output.parent, previous, allow_retired_sources=True)
    batch = export_suite([Path(row["source"]) for row in included], output, previous_workbench=anchor,
                         task_demo_root=task_demo_root, selection=selection, workers=workers, profile="staged")
    if generate_model_plots:
        from evaluation_workbench.suite_figures import complete_missing_model_plots
        complete_missing_model_plots(output, workers=workers)
        batch = _document(output / "batch.json")
    # Attach source and diagnostic grouping without rewriting any trial payload.
    from .publish import publish_unified_metadata
    publish_unified_metadata(output, stage_groups(batch["trials"]), source_guard)
    verify_original_sources()
    return _document(output / "batch.json")


def verify_unified(output):
    result = verify_suite(output)
    verification = _document(Path(output) / "verification.json")
    if verification.get("unified_import", {}).get("profile") != "recorded-staged":
        raise ValueError("This is not a unified staged report")
    return {**result, **verify_original_sources(), "profile": "recorded-staged"}
