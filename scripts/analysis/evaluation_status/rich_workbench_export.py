"""Restore rich detail views without changing any published score admission.

Old rich payloads are consumed from their immutable publication inventories.
New details use exactly the producer selected by the published catalog. This is
an offline display migration, not a simulation, rescore, or model certification.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

from .export import PACKAGE, _copy, _publish_portal, _write, digest, encoded, verify_row
from .unified_workbench_export import (
    _Artifacts,
    _read,
    _validate_catalog,
    verify_unified_workbench,
)

SCHEMA = "kaihand-rich-unified-workbench-snapshot-v1"
_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_PAYLOADS = {"report.json", "visualization.json", "tactile_loads.json", "workbench.json", "report.html", "audit.html"}
# Exact recorded review-renderer leaves only. These are read to validate crop
# metadata, never imported/executed; this does not certify retired evaluators.
_RECORDED_RENDERERS = frozenset({
    Path("/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/src/kaihand_tactile_env/shared/evaluation_video.py"),
    Path("/cpfs_infra/user/chenxianchi/runtime_snapshots/pi05_serial_20261013_111821/kaitactilesim/src/kaihand_tactile_env/shared/evaluation_video.py"),
})


def score_projection(catalog):
    """All published membership and statistical facts, excluding display links."""
    result = copy.deepcopy(catalog)
    for key in ("generated_utc", "provenance", "verification_scope", "links", "rich_details"):
        result.pop(key, None)
    for group in result["groups"]:
        for trial in group["trials"]:
            for key in ("detail_url", "video_url", "rich_detail"):
                trial.pop(key, None)
    return result


def _merge_inputs(target, rows):
    for row in rows:
        previous = target.get(row["path"])
        if previous and (previous["sha256"], previous["size"]) != (row["sha256"], row["size"]):
            raise ValueError(f"Source identity changed within export: {row['path']}")
        target.setdefault(row["path"], row)


def _identity(group, trial, stage_recorded):
    return {"group_id": group["id"], "trial_id": trial["id"], "task": group["task"],
            "model": group["model"], "checkpoint": group["checkpoint"], "seed": trial["seed"],
            "prediction_horizon": group["prediction_horizon"], "execute_steps": group["execute_steps"],
            **{key: copy.deepcopy(trial.get(key)) for key in (
                "valid", "native_success", "publication_status", "staged_evidence_status", "staged_evidence_basis",
                "notes", "stages", "complete_stage_success")}, "stage_recorded": stage_recorded}


def _audit_page(report, group, trial):
    from evaluation_v1.render import render_report

    warning = ("当前记录已隔离，仅供回看，不计入成功率、阶段成功率或任何有效样本分母。下方保留历史字段，不代表恢复为有效成绩。"
               if not trial["valid"] else
               "阶段记录已保存，但来源校验未通过，阶段达成和首次时间不计入当前统计。下方为原记录的诊断审计。"
               if trial["staged_evidence_status"] == "invalid" else
               "此页保留原记录与离线诊断字段。当前是否计分以统一工作台的模型、checkpoint 和数据有效性为准；未重跑或重判历史结果。")
    context = f"{group['title']} · {group['model']} · checkpoint {group['checkpoint']} · seed {trial['seed']}"
    banner = (f'<aside class="rich-audit-context" style="margin:20px;padding:20px;border:2px solid #b77c24;background:#fff8e9;color:#593c17;border-radius:12px">'
              f'<a href="report.html">← 返回轨迹分析</a> · <a href="../index.html">统一工作台</a>'
              f'<h2>原始记录与诊断审计</h2><p>{escape(context)}</p><p>{escape(warning)}</p></aside>')
    return render_report(report).replace("<body>", "<body>" + banner, 1)


def _legacy_case(store, detail, trial):
    if trial["publication_status"] == "quarantined_replay_only":
        matches = re.findall(r'class="historical-replay-link" href="([^"]+)"', detail.read_text())
        if len(matches) != 1:
            raise ValueError("Quarantine wrapper needs one inventoried historical replay")
        detail = store.target(detail.parent, matches[0])
    source = detail.parent
    data = {key: store.document(source / name, "published_rich_payload") for key, name in (
        ("report", "report.json"), ("visualization", "visualization.json"),
        ("tactile_loads", "tactile_loads.json"), ("config", "workbench.json"))}
    internal = data["report"]["trial"]["trial_id"]
    if data["config"]["trial_id"] != internal or data["tactile_loads"]["trial_id"] != internal:
        raise ValueError("Historical rich payload identities disagree")
    root = store.reports / source.relative_to(store.reports).parts[0]
    prefix = source.relative_to(root).as_posix() + "/"
    copies = []
    for relative, row in store.inventory(root).items():
        if relative.startswith(prefix):
            suffix = relative[len(prefix):]
            if suffix in _PAYLOADS:
                continue
            path = store.artifact(row["path"], "published_rich_companion")
            copies.append({"source": path, "url": suffix, "sha256": row["sha256"], "size": row["size"]})
    data.update(copies=copies, provenance=list(store.inputs.values()),
                availability={"kind": "preserved_rich", "native_samples": len(data["tactile_loads"].get("time_s", [])),
                              "task_series": len(data["visualization"].get("task_series", [])),
                              "stage_evidence_status": trial["staged_evidence_status"],
                              "diagnostic_plots": data["config"].get("diagnostic_plots", {}).get("status")})
    return data


def _model_plots(case, group, trial, output, artifacts):
    """Draw recorded actual states only, with the catalog's genuine H/K identity."""
    from evaluation_v1.visualization import _gap_flags
    from evaluation_workbench.diagnostic_plots import _png
    from evaluation_workbench.model_plots import _PLOTS, _native_arrays, _render

    source = Path(case["source_binding"]["producer_summary_path"])
    summary, row = _read(source)
    verify_row(case["source_binding"]["producer_summary"], source)
    comparison = (summary.get("video") or {}).get("comparison_plots")
    if not isinstance(comparison, dict):
        return
    traces = [item for item in case["provenance"] if Path(item["path"]).name == "evaluation_rollout_trace.npz"]
    if len(traces) != 1:
        return
    trace = traces[0]
    verify_row(trace)
    arrays = _native_arrays(Path(trace["path"]), comparison)
    gaps, interval = _gap_flags(arrays[1], case["report"]["protocol"].get("max_gap_factor", 1.5))
    title = f"{group['task']} / {group['model']} / checkpoint {group['checkpoint']} / seed {trial['seed']} / H{group['prediction_horizon']} K{group['execute_steps']}"
    plots = []
    with tempfile.TemporaryDirectory(prefix="rich-model-plots-", dir=output.parent) as temporary:
        stage = Path(temporary)
        _render(stage, arrays, gaps, title)
        for identifier, name, label, description in _PLOTS:
            dimensions, image_row = _png(stage / name)
            relative = Path(trial["id"]) / "plots" / name
            _write(output, relative, (stage / name).read_bytes(), artifacts)
            plots.append({"id": identifier, "label": label, "description": description,
                          "url": "plots/" + name, "sha256": image_row["sha256"],
                          "width": dimensions[0], "height": dimensions[1]})
    verify_row(trace)
    case["config"]["diagnostic_plots"] = {
        "status": "available", "comparison_kind": "model_only", "task": group["task"], "seed": trial["seed"],
        "checkpoint": group["checkpoint"], "prediction_horizon": group["prediction_horizon"],
        "source_label": title, "reference_label": "无参考轨迹", "rollout_label": "红色实线 · 仅模型",
        "comparison_note": "模型实际记录的离线诊断图；没有专家参考线，不用于重新判定成功或专家相似度。",
        "plots": plots, "provenance": [trace, row],
        "generated": {"kind": "model_only_native_record_plots", "source_trace_sha256": trace["sha256"],
                      "sample_count": len(arrays[1]), "reference_loaded": False, "resampled": False,
                      "metrics_recomputed": False, "nominal_interval_s": interval, "gap_count": sum(gaps)}}
    case["availability"]["diagnostic_plots"] = "model_only_generated"


def _export_one(arguments):
    from evaluation_workbench.suite import _page

    from evaluation_status.rich_detail_data import build_rich_case, resolve_rich_source

    baseline, evaluation_root, output, group, trial = arguments
    baseline, output = Path(baseline), Path(output)
    binding = resolve_rich_source(baseline, group, trial, evaluation_root=evaluation_root)
    inputs = {}
    _merge_inputs(inputs, binding["provenance"])
    if binding["kind"] in ("existing_rich", "quarantine"):
        case = _legacy_case(_Artifacts(baseline.parent, output), Path(binding["published_detail"]), trial)
    else:
        case = build_rich_case(binding["source_dir"], group, trial,
                               published_summary_path=binding["published_summary_path"],
                               expected_summary_sha256=binding["expected_summary_sha256"])
    _merge_inputs(inputs, case["provenance"])
    artifacts = []
    prefix = Path(trial["id"])
    for job in case["copies"]:
        _copy(output, prefix / job["url"], job["source"], artifacts, inputs, expected=job["sha256"])
    if binding["kind"] == "build" and case["config"].get("diagnostic_plots", {}).get("status") != "available":
        _model_plots(case, group, trial, output, artifacts)
    config = case["config"]
    config["unified_identity"] = _identity(group, trial, isinstance(config.get("staged_evaluation"), dict))
    config["unified_navigation"] = {"catalog_url": "../catalog.json", "overview_url": "../index.html"}
    config["suite_url"] = "../index.html"
    config["audit_links"] = [{"label": "查看原始归档详情", "url": Path(os.path.relpath(binding["published_detail"], output / prefix)).as_posix()}]
    for name, key in (("report.json", "report"), ("visualization.json", "visualization"),
                      ("tactile_loads.json", "tactile_loads"), ("workbench.json", "config")):
        _write(output, prefix / name, encoded(case[key]), artifacts)
    _write(output, prefix / "report.html", _page(case["report"], case["visualization"], case["tactile_loads"], config).encode(), artifacts)
    _write(output, prefix / "audit.html", _audit_page(case["report"], group, trial).encode(), artifacts)
    media = case["visualization"].get("media", {})
    video = (prefix / media["url"]).as_posix() if media.get("url") else None
    return {"id": trial["id"], "detail_url": (prefix / "report.html").as_posix(), "video_url": video,
            "availability": case["availability"], "artifacts": artifacts, "inputs": list(inputs.values()), "kind": binding["kind"]}


def export_rich_workbench(baseline, output, *, evaluation_root, workers=4, publish_index=False):
    from evaluation_v1.dashboard import read_assets

    from .unified_workbench_page import render_page

    baseline, output = Path(baseline).resolve(strict=True), Path(output).absolute()
    evaluation_root = Path(evaluation_root).resolve(strict=True)
    if output.exists() or output.parent.resolve(strict=True) != baseline.parent or output.is_symlink():
        raise ValueError("Rich output must be a fresh sibling report snapshot")
    verify_unified_workbench(baseline)
    store = _Artifacts(baseline.parent, output)
    original = store.document(baseline / "catalog.json")
    _validate_catalog(original)
    catalog = copy.deepcopy(original)
    seen = set()
    tasks = []
    for group in original["groups"]:
        for trial in group["trials"]:
            if not _ID.fullmatch(trial["id"]) or trial["id"] in seen:
                raise ValueError("Canonical trial IDs must be unique safe path segments")
            seen.add(trial["id"])
            tasks.append((str(baseline), str(evaluation_root), str(output), group, trial))
    previous = (output.parent / "index.html").read_bytes() if (output.parent / "index.html").exists() else None
    output.mkdir()
    artifacts, inputs, results = [], dict(store.inputs), {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_export_one, tasks, chunksize=1), 1):
            artifacts.extend(result.pop("artifacts"))
            _merge_inputs(inputs, result.pop("inputs"))
            results[result["id"]] = result
            if index % 10 == 0 or index == len(tasks):
                print(f"rich-details {index}/{len(tasks)}", flush=True)
    for group in catalog["groups"]:
        for trial in group["trials"]:
            item = results[trial["id"]]
            trial.update(detail_url=item["detail_url"], video_url=item["video_url"], rich_detail=item["availability"])
    if score_projection(original) != score_projection(catalog):
        raise ValueError("Rich display migration changed an admission or statistical fact")
    catalog["generated_utc"] = datetime.now(timezone.utc).isoformat()
    catalog["rich_details"] = {"schema": SCHEMA, "count": len(results), "baseline": baseline.name,
                               "simulation_runs_started": 0, "native_outcomes_rescored": False,
                               "staged_rules_reexecuted": False}
    _validate_catalog(catalog)
    for name, raw in read_assets().items():
        _write(output, Path("assets") / name, raw, artifacts)
    for source, target in (
        (PACKAGE.parent / "evaluation_workbench/web/workbench.js", "workbench.js"),
        (PACKAGE.parent / "evaluation_workbench/web/workbench.css", "workbench.css"),
        (PACKAGE.parent / "evaluation_workbench/web/suite.css", "suite.css"),
        (PACKAGE / "web/unified_workbench.js", "unified_workbench.js"),
        (PACKAGE / "web/unified_workbench.css", "unified_workbench.css"),
    ):
        _write(output, Path("assets") / target, source.read_bytes(), artifacts)
        _merge_inputs(inputs, [{**digest(source), "role": "display_implementation"}])
    for source in (Path(__file__), PACKAGE / "rich_detail_data.py", PACKAGE.parent / "export_rich_evaluation_workbench.py"):
        _merge_inputs(inputs, [{**digest(source), "role": "offline_export_implementation"}])
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    _write(output, Path("index.html"), render_page(catalog).encode(), artifacts)
    if previous is not None:
        _write(output, Path("previous_root_index.html"), previous, artifacts)
    verification = {"schema": SCHEMA, "baseline_catalog": str(baseline / "catalog.json"),
                    "evaluation_root": str(evaluation_root), "score_admissions_preserved": True,
                    "source_writes": 0, "existing_snapshot_writes": 0, "simulation_runs_started": 0,
                    "native_outcomes_rescored": False, "staged_rules_reexecuted": False,
                    "live_model_weights_verified": False,
                    "source_files": sorted(inputs.values(), key=lambda row: row["path"])}
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SCHEMA, "files": artifacts}), [])
    result = verify_rich_workbench(output)
    if publish_index:
        publish_rich_workbench(output)
        result = verify_rich_workbench(output)
    return result


def verify_rich_workbench(output, *, verify_sources=True):
    output = Path(output).resolve(strict=True)
    manifest, _ = _read(output / "artifact_manifest.json")
    verification, _ = _read(output / "verification.json")
    if manifest.get("schema") != SCHEMA or verification.get("schema") != SCHEMA:
        raise ValueError("Unsupported rich-workbench snapshot")
    paths = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        path = output / relative
        if relative.is_absolute() or ".." in relative.parts or row["path"] in paths or path.is_symlink() or not path.resolve(strict=True).is_relative_to(output):
            raise ValueError("Unsafe or duplicate rich artifact")
        paths.add(row["path"])
        verify_row(row, path)
    roots = (output.parent, Path(verification["evaluation_root"]).resolve(strict=True), PACKAGE.parent.resolve(strict=True))
    if verify_sources:
        for row in verification["source_files"]:
            path = Path(row["path"])
            actual = path.resolve(strict=True)
            if path.is_symlink() or (actual not in _RECORDED_RENDERERS and not any(actual.is_relative_to(root) for root in roots)):
                raise ValueError("Rich verification input escapes admitted publication/record/implementation roots")
            verify_row(row, path)
    catalog, _ = _read(output / "catalog.json")
    baseline, _ = _read(verification["baseline_catalog"])
    _validate_catalog(catalog)
    if score_projection(catalog) != score_projection(baseline):
        raise ValueError("Rich catalog changed baseline score admissions")
    for group in catalog["groups"]:
        for trial in group["trials"]:
            if trial["detail_url"] != f"{trial['id']}/report.html" or trial["detail_url"] not in paths:
                raise ValueError("Rich member detail link is not locally inventoried")
            if trial.get("video_url") and trial["video_url"] not in paths:
                raise ValueError("Rich member video link is not locally inventoried")
            config, _ = _read(output / trial["id"] / "workbench.json")
            if config.get("unified_identity") != _identity(group, trial, config.get("unified_identity", {}).get("stage_recorded")):
                raise ValueError("Rich detail identity differs from catalog")
    if (output / "portal_publication.pending.json").exists():
        raise ValueError("Incomplete rich root publication")
    published = None
    if (output / "portal_publication.json").exists():
        audit, _ = _read(output / "portal_publication.json")
        if audit.get("state") != "committed" or audit.get("root_index") != str(output.parent / "index.html"):
            raise ValueError("Invalid rich root publication audit")
        published = digest(output.parent / "index.html")["sha256"] == audit.get("new_sha256")
    return {"status": "passed", **catalog["counts"], "artifacts": len(paths),
            "dependencies": len(verification["source_files"]), "score_admissions_preserved": True,
            "simulation_runs_started": 0, "root_portal_matches_this_snapshot": published}


def publish_rich_workbench(output):
    output = Path(output).resolve(strict=True)
    verify_rich_workbench(output)
    previous = (output / "previous_root_index.html").read_bytes() if (output / "previous_root_index.html").exists() else None
    relative = quote(output.name, safe="._-") + "/index.html"
    landing = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={relative}">'
               f'<title>KaiTactileSim 官方评估平台</title><p><a href="{relative}">进入 KaiTactileSim 统一评估工作台</a></p></html>').encode()
    _publish_portal(output, output.parent / "index.html", previous, landing)

