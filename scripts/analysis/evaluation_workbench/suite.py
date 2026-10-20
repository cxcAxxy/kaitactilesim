"""Export a local H50 workbench for every completed, explicitly selected trial."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re

from evaluation_v1.dashboard import read_assets, render_dashboard
from evaluation_v1.render import render_report
from evaluation_v1.protocol import make_protocol, content_hash
from evaluation_v2.horizon import inspect_prediction_horizon

from .export import _digest, _document, _files, _inventory, _rows, _verify, _verify_inventory, verify_rollback

SCHEMA = "evaluation-workbench-suite-v1"
WEB = Path(__file__).parent / "web"


def discover_trials(source_root, extra_roots=()):
    """Only seed directories, canonical deduplication, completed recorder, verified H50."""
    root = Path(source_root).resolve(strict=True)
    candidates = sorted(root.glob("*/*/*/seed_*/summary.json"))
    for directory in extra_roots:
        directory = Path(directory).resolve(strict=True)
        candidates.extend(sorted(directory.rglob("seed_*/summary.json")))
    if len(candidates) > 5000:
        raise ValueError("Trial discovery exceeds 5000 explicit seed summaries")
    admitted, excluded, aliases, seen = [], [], [], set()
    for summary_path in candidates:
        if summary_path.is_relative_to(root) and summary_path.relative_to(root).parts[0].startswith("_") and not any(summary_path.is_relative_to(Path(p).resolve()) for p in extra_roots):
            continue
        path = summary_path.parent.resolve(strict=True)
        if not re.fullmatch(r"seed_\d+", path.name):
            continue
        if path in seen:
            aliases.append({"path": str(summary_path.parent), "canonical": str(path)})
            continue
        seen.add(path)
        try:
            summary = _document(path / "summary.json")
            if summary.get("status") in {"running", "starting", "queued", "pending", "in_progress"}:
                excluded.append({"source": str(path), "reason": "in_progress"}); continue
            if not isinstance(summary.get("video"), dict) or summary["video"].get("completed") is not True:
                excluded.append({"source": str(path), "reason": "recording_not_complete"}); continue
            horizon = inspect_prediction_horizon(path)
            if horizon.get("status") != "verified" or horizon.get("prediction_horizon") != 50:
                excluded.append({"source": str(path), "reason": "prediction_horizon_not_verified_H50", "inspection": horizon}); continue
            admitted.append(path)
        except (ValueError, OSError, KeyError) as error:
            excluded.append({"source": str(path), "reason": str(error)})
    admitted.sort(key=lambda p: (str(_document(p / "summary.json").get("task")), str(p.parent), p.name, str(p)))
    return admitted, {"source_root": str(root), "extra_roots": [str(Path(p).resolve()) for p in extra_roots],
                      "snapshot_utc": datetime.now(timezone.utc).isoformat(), "candidate_count": len(candidates),
                      "included_count": len(admitted), "excluded": excluded, "aliases": aliases,
                      "required_prediction_horizon": 50,
                      "semantics": "Completed seed snapshots only; canonical aliases counted once; active controller status is not outcome evidence."}


def _json(path, value, artifacts=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    _bytes(path, raw, artifacts)


def _bytes(path, raw, artifacts=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    if artifacts is not None:
        artifacts.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})


def _copy(job, target, artifacts):
    source = Path(job["source"]).resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    before = source.stat()
    with source.open("rb") as original, target.open("xb") as copied:
        for chunk in iter(lambda: original.read(1024 * 1024), b""):
            copied.write(chunk); digest.update(chunk); size += len(chunk)
    after = source.stat()
    if ((before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns)
            or digest.hexdigest() != job["sha256"] or size != job.get("size", size)):
        raise ValueError(f"Copy source changed: {source}")
    artifacts.append({"path": str(target), "sha256": digest.hexdigest(), "size": size})


def _mode(outcome):
    if outcome.get("formal_metrics_valid") is True and outcome.get("diagnostic_only") is False:
        return "formal"
    if outcome.get("diagnostic_only") is True and outcome.get("formal_metrics_valid") is False:
        return "diagnostic"
    return "unverified"


def _page(report, visual, loads, config):
    html = render_dashboard(report, visual)
    def inert(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = html.replace("</head>", '<link rel="stylesheet" href="../assets/workbench.css">\n</head>')
    return html.replace("</body>", f'<script type="application/json" id="kai-tactile-load-data">{inert(loads)}</script>'
                        f'<script type="application/json" id="kai-workbench-config">{inert(config)}</script>'
                        '<script src="../assets/workbench.js" defer></script></body>')


def _portal(batch):
    data = json.dumps(batch, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    staged = batch.get("evaluation_profile") == "recorded-staged"
    title = "统一评估 · staged 观察台" if staged else "多任务轨迹观察台"
    stage_filter = ('<label>阶段记录<select id="ks-stage"><option value="">全部</option><option value="complete">完整阶段通过</option>'
                    '<option value="partial">已到达部分阶段</option><option value="none">未到达首阶段</option></select></label>') if staged else ""
    stage_summary = ('<aside class="ks-import-note">原任务判定与 staged 阶段判定分别保留；'
                     '阶段卡表示整条轨迹曾到达，不代表当前接触。白板目前只定义拿起阶段，不等于擦拭完成。</aside>'
                     '<details class="ks-stage-groups"><summary>同规则、同权重的阶段统计</summary><p>只在相同任务、权重内容、批次和归档规则内计数，不合并不同规则，不重算旧判定。</p>'
                     '<div class="ks-group-scroll"><table><thead><tr><th>任务 / 权重</th><th>轨迹</th><th>曾到达阶段</th><th>完整阶段</th><th>原任务成功</th><th>规则身份</th></tr></thead>'
                     '<tbody id="ks-stage-groups"></tbody></table></div></details>') if staged else ""
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>H50 · {title}</title><link rel="stylesheet" href="assets/suite.css"><script src="assets/suite.js" defer></script>'
            f'</head><body><main id="kai-suite"><header><p class="ks-eyebrow">TACTILE EVALUATION / H50</p><h1>{title}</h1>'
            '<p>从任务记录、模型动作与专家示范，到五指触觉、关节角和右腕状态。</p>'
            f'<a class="ks-old" href="{escape(batch["previous_url"], quote=True)}">{"原 H50 稳定版（保留）" if staged else "返回上一版白板观察台"}</a></header>'
            '<div id="ks-counts" class="ks-counts"></div><section class="ks-controls"><div id="ks-task-tabs" class="ks-task-tabs" role="group" aria-label="选择任务"></div>'
            '<div class="ks-filter-row"><label>模型<select id="ks-model"><option value="">全部</option></select></label>'
            '<label>Checkpoint<select id="ks-checkpoint"><option value="">全部</option></select></label>'
            '<label>原记录结果<select id="ks-outcome"><option value="">全部</option><option value="pass">通过</option><option value="fail">未通过</option><option value="unknown">未知</option></select></label>'
            f'{stage_filter}'
            '<label class="ks-search">查找 seed / 轨迹<input id="ks-search" type="search" placeholder="例如 seed_005"></label></div></section>'
            f'{stage_summary}'
            '<section id="ks-task-note" class="ks-task-note"></section><p id="ks-visible" class="ks-visible" role="status"></p>'
            '<div id="ks-trials" class="ks-trials"></div><p class="ks-footnote">失败与未知记录均保留。本页只列原记录结果，不合并成模型成功率排名。</p>'
            f'</main><script type="application/json" id="kai-suite-data">{data}</script></body></html>')


def implementation():
    package = Path(__file__).parent
    paths = list(package.glob("*.py")) + list(WEB.glob("*.js")) + list(WEB.glob("*.css"))
    paths += list((package.parent / "evaluation_unified").glob("*.py"))
    paths += [path for path in (
        package.parent / "evaluate_unified.py",
        package.parents[2] / "src/kaihand_tactile_env/shared/evaluation_rules.py",
        package.parents[2] / "src/kaihand_tactile_env/shared/staged_evaluation.py",
        package.parents[2] / "src/kaihand_tactile_env/shared/staged_recording.py",
    ) if path.is_file()]
    return [_digest(path) for path in sorted(set(paths))]


def _assemble_trial(arguments):
    from .data import build_case
    from .task_fields import augment_task_visualization, task_presentation
    path, protocol, *profiles = arguments
    case = build_case(path, protocol=protocol)
    trial = case.pop("trial")
    presentation = augment_task_visualization(trial, case["report"], case["visualization"], frame_rows=case.pop("frame_rows", None))
    case["config"].update(generic=True, suite_url="../index.html", task_presentation=task_presentation(trial.task))
    case["config"].update(presentation)
    if profiles and profiles[0] == "staged":
        from evaluation_unified.report import augment_staged_case
        augment_staged_case(case, trial, profiles[1] if len(profiles) > 1 else None)
    case["report"]["visualization"]["sha256"] = content_hash(case["visualization"])
    case["summary"] = trial.summary
    return case


def _prepared_cases(trials, protocol, workers, profile=None, selected=None):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        def arguments(index):
            if profile:
                return (trials[index], protocol, profile, (selected or {}).get(str(trials[index])))
            return (trials[index], protocol)
        pending = {index: executor.submit(_assemble_trial, arguments(index))
                   for index in range(min(workers, len(trials)))}
        try:
            for index in range(len(trials)):
                case = pending.pop(index).result()
                next_index = index + workers
                if next_index < len(trials):
                    pending[next_index] = executor.submit(_assemble_trial, arguments(next_index))
                yield case
        finally:
            for future in pending.values():
                future.cancel()


def export_suite(trials, output, *, previous_workbench, task_demo_root, selection=None, workers=4, profile=None):
    from .experts import prepare_expert_bindings, expert_for_case, model_views_for_case
    from .task_fields import augment_task_visualization, task_presentation

    output = Path(output).absolute()
    previous = Path(previous_workbench).resolve(strict=True)
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError("Use a new non-symlink output directory")
    if output.parent != previous.parent:
        raise ValueError("Suite must be a new sibling of the preserved previous workbench")
    trials = [Path(p).resolve(strict=True) for p in trials]
    if not trials or len(set(trials)) != len(trials) or len(trials) > 5000:
        raise ValueError("Select distinct, bounded completed H50 trials")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use 1 to 8 offline CPU workers")
    if profile not in (None, "staged"):
        raise ValueError("Unknown recorded evaluation profile")
    for trial in trials:
        if output == trial or output.is_relative_to(trial):
            raise ValueError("Output cannot be inside original trials")
        summary = _document(trial / "summary.json")
        if (summary.get("status") in {"running", "starting", "queued", "pending", "in_progress"}
                or summary.get("completed") is False or not isinstance(summary.get("video"), dict)
                or summary["video"].get("completed") is not True):
            raise ValueError(f"Trial is not a completed immutable snapshot: {trial}")
        horizon = inspect_prediction_horizon(trial)
        if horizon.get("status") != "verified" or horizon.get("prediction_horizon") != 50:
            raise ValueError(f"Only verified H50 trials may be exported: {trial}")
    verify_rollback(previous / "rollback.json")
    inherited = _document(previous / "rollback.json")
    previous_inventory = _inventory([previous])
    protected = _rows(inherited["protected_files"] + [_digest(path) for path in _files(previous)])
    protected_inventory = inherited.get("protected_roots", []) + previous_inventory
    _verify(protected); _verify_inventory(protected_inventory)
    source_implementation = implementation()
    bindings = prepare_expert_bindings(trials, task_demo_root)
    protocol = make_protocol()
    output.mkdir()
    artifacts, entries, sources = [], [], []
    for name, raw in read_assets().items():
        _bytes(output / "assets" / name, raw, artifacts)
    for name in ("workbench.js", "workbench.css", "suite.js", "suite.css"):
        _bytes(output / "assets" / name, (WEB / name).read_bytes(), artifacts)
    previous_url = f"../{previous.name}/index.html"
    selected = {str(Path(row["source"]).resolve()): row for row in (selection or {}).get("included", [])}
    prepared = _prepared_cases(trials, protocol, workers, profile, selected) if profile else _prepared_cases(trials, protocol, workers)
    for index, case in enumerate(prepared):
        trial = trials[index]
        report, visual, loads, config = [case[key] for key in ("report", "visualization", "tactile_loads", "config")]
        expert = expert_for_case(case["summary"], bindings)
        expert_jobs = expert.pop("copy_jobs", []) if isinstance(expert, dict) else []
        if expert:
            config["expert_video"] = expert
        config["media_views"] = model_views_for_case(case["summary"], visual.get("media", {}).get("sha256"))
        report["provenance"] = _rows(report["provenance"] + (expert.get("provenance", []) if expert else []) + config["media_views"].get("provenance", []))
        loads["provenance"] = report["provenance"]
        config["previous_workbench_url"] = f"../../{previous.name}/index.html"
        task = report["trial"]["task"]
        model = str(report["trial"].get("cohort", {}).get("model_family") or case["summary"].get("model_family") or "unknown")
        checkpoint = str(config.get("checkpoint") or Path(str(case["summary"].get("checkpoint_path", "unknown"))).name)
        seed = report["trial"].get("seed")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{index:03d}_{task}_{model}_step{checkpoint}_seed_{seed:03d}" if type(seed) is int else f"{index:03d}_{task}_{model}_unknown_seed")
        directory = output / slug
        report["report_path"] = slug + "/report.html"
        for name, value in (("report.json", report), ("visualization.json", visual), ("tactile_loads.json", loads), ("workbench.json", config)):
            _json(directory / name, value, artifacts)
        if "unified_staged" in case:
            _json(directory / "staged_evaluation.json", case["unified_staged"], artifacts)
        _bytes(directory / "report.html", _page(report, visual, loads, config).encode("utf-8"), artifacts)
        _bytes(directory / "audit.html", render_report(report).encode("utf-8"), artifacts)
        for job in case["copies"] + expert_jobs:
            relative = Path(job["url"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe copied case asset URL")
            _copy(job, directory / relative, artifacts)
        _verify(report["provenance"])
        sources.append({"trial_id": report["trial"]["trial_id"], "source": str(trial), "provenance": report["provenance"]})
        outcome = report.get("outcome", {})
        entry = {"trial_id": report["trial"]["trial_id"], "task": task, "title": config["task_presentation"].get("title", task),
                 "seed": seed, "model": model, "checkpoint": checkpoint, "prediction_horizon": 50, "mode": _mode(outcome),
                 "outcome": outcome.get("status", "unknown"), "report_path": report["report_path"], "source": str(trial),
                 "tactile": loads.get("normal_status", "missing"), "tangent": loads.get("tangent_status", "missing"),
                 "plots": config.get("diagnostic_plots", {}).get("status", "missing"),
                 "expert": expert.get("status", "missing") if expert else "missing",
                 "expert_relation": expert.get("reference_relation") if expert else None,
                 "task_curve_count": len(visual.get("task_series", [])), "cohort": report["trial"].get("cohort", {})}
        entries.append(entry)
        if profile == "staged":
            staged = config["staged_evaluation"]
            entry["staged"] = {"version": staged["staged_result"].get("version"),
                "stages": staged["stages"], "complete_success": staged["staged_result"].get("complete_success"),
                "rule_identity": staged["rule_identity"], "source_cohort": trial.parent.name,
                "data_evidence": staged["data_evidence"], "native_result": staged["native_result"],
                "source_selection": staged.get("source_selection", {})}
        print(json.dumps({"completed": index + 1, "total": len(trials), "task": task, "seed": seed, "outcome": entry["outcome"]}), flush=True)
    counts = Counter(row["mode"] for row in entries)
    for source in sources:
        _verify(source["provenance"])
    catalog = {task: {**task_presentation(task), "trial_count": sum(row["task"] == task for row in entries)} for task in sorted({row["task"] for row in entries})}
    batch = {"schema_version": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(), "prediction_horizon": 50,
             "trial_count": len(entries), "task_count": len(catalog), "mode_counts": dict(counts), "tasks": catalog,
             "trials": entries, "previous_url": previous_url,
             "semantics": "Original outcome records; diagnostic/formal/unverified remain separate. No cross-task or cross-model success-rate ranking.",
             "coverage": {"normal": sum(row["tactile"] == "available" for row in entries), "tangent": sum(row["tangent"] == "available" for row in entries),
                          "plots": sum(row["plots"] == "available" for row in entries), "expert": sum(row["expert"] == "available" for row in entries)}}
    if profile == "staged":
        batch["evaluation_profile"] = "recorded-staged"
        batch["rule_semantics"] = "Group only identical archived rule identities; no current-rule rescoring. Native and staged diagnostics stay separate."
    for name, value in (("batch.json", batch), ("selection.json", selection or {}), ("source_manifest.json", {"trials": sources}), ("protocol.json", protocol)):
        _json(output / name, value, artifacts)
    _verify(source_implementation)
    _verify(protected); _verify_inventory(protected_inventory)
    rollback = {"schema_version": "evaluation-experiment-rollback-v1", "experiment": SCHEMA, "previous_workbench": str(previous),
                "protected_files": protected, "protected_roots": protected_inventory,
                "protected_artifact_manifests": inherited.get("protected_artifact_manifests", []),
                "strategy": "All earlier files preserved; return to previous_workbench. New suite is an independent H50 snapshot."}
    _json(output / "rollback.json", rollback, artifacts)
    verification = {"status": "passed", "trial_count": len(entries), "task_count": len(catalog), "protected_file_count": len(protected),
                    "previous_workbench_unchanged": True, "h50_only": True, "simulation_runs_started": 0,
                    "reference_time_alignment": "none", "original_source_writes": 0, "implementation": source_implementation,
                    "coverage": batch["coverage"], "mode_counts": batch["mode_counts"]}
    if profile == "staged":
        verification["prior_site_preservation"] = {key: inherited.get(key) for key in
            ("preserved_site", "prior_rollback_status", "prior_rollback_alias_rebindings", "prior_rollback_unavailable_sources", "preservation_scope")}
    _json(output / "verification.json", verification, artifacts)
    _bytes(output / "index.html", _portal(batch).encode("utf-8"), artifacts)
    _json(output / "artifact_manifest.json", {"files": artifacts})
    return batch


def verify_suite(output):
    output = Path(output).resolve(strict=True)
    result = verify_rollback(output / "rollback.json")
    for row in _document(output / "artifact_manifest.json")["files"]:
        path = Path(row["path"])
        if not path.is_relative_to(output):
            raise ValueError("Artifact manifest escapes the exported suite")
        _verify([row])
    for case in _document(output / "source_manifest.json")["trials"]:
        _verify(case["provenance"])
    return {**result, "suite_verified": True, "trial_count": _document(output / "batch.json")["trial_count"]}
