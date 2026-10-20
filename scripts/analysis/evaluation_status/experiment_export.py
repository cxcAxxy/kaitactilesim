"""Present checkpoints of one experiment together, preserving quarantine.

This publishes an identity-bound view of existing snapshots, not a new native
evaluation or a repair of the canonical archive's unpublished records.
"""
from __future__ import annotations

import copy
import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .export import (
    PACKAGE,
    _bound_document,
    _copy,
    _inside,
    _portal,
    _publish_portal,
    _write,
    digest,
    document,
    encoded,
    inert,
    verify_row,
    verify_status,
)

SCHEMA = "kaihand-evaluation-experiment-snapshot-v1"


def _identities(status, historical, catalog):
    """Bind current snapshot, its original dependencies, and builder metadata."""
    manifest, identity = _bound_document(status / "artifact_manifest.json")
    rows = [identity]
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe parent snapshot artifact")
        rows.append({**row, "path": str(_inside(status, status / relative))})
    verification, identity = _bound_document(status / "verification.json")
    if Path(verification["historical_root"]).resolve(strict=True) != historical:
        raise ValueError("Preserved site does not match the verified parent")
    rows.extend([identity, *verification["historical_artifacts"],
                 *verification["historical_metadata"], *verification["source_files"],
                 *catalog.get("provenance", []), *catalog.get("dependencies", [])])
    unique = {}
    for row in rows:
        if row.get("mutable", False):
            continue
        key = str(Path(row["path"]).resolve(strict=True))
        row = {**row, "path": key}
        previous = unique.get(key)
        if previous and (previous["size"], previous["sha256"]) != (row["size"], row["sha256"]):
            raise ValueError(f"Conflicting dependency identity: {key}")
        unique[key] = row
    return unique


def _target(output, href):
    """Only link verified files within this read-only report collection."""
    if not isinstance(href, str) or "\\" in href or "\x00" in href:
        raise ValueError("Invalid experiment detail URL")
    url = urlsplit(href)
    if url.scheme or url.netloc or url.query or url.fragment or url.path.startswith("/"):
        raise ValueError("Experiment detail URL must be a relative report link")
    target = (output / unquote(url.path)).resolve(strict=True)
    if not target.is_relative_to(output.parent) or not target.is_file():
        raise ValueError("Experiment detail escapes report collection")
    return target


def _quarantine_page(group, trial):
    reason = str(trial.get("quarantine_reason") or "Canonical archive publication validation failed; this record remains replay-only.")
    # Wrappers live two levels below the main experiment page.
    original = "../../" + trial["historical_detail_url"]
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>隔离回放 · 不计成绩 | KaiTactileSim</title>'
            '<style>body{max-width:850px;margin:40px auto;padding:20px;font:16px/1.7 system-ui;color:#243c32}'
            '.warning{padding:20px;border:2px solid #c88c27;border-radius:12px;background:#fff8e7}'
            'a{color:#176b4d}code{overflow-wrap:anywhere}</style>'
            '<a href="../../index.html">← 返回同实验 checkpoint 对比</a>'
            f'<h1>{escape(str(group["title"]))} · step {escape(str(group["checkpoint"]))} · seed {trial["seed"]}</h1>'
            '<section class="warning"><h2>历史隔离记录：仅回看，不计成绩</h2>'
            '<p>此条属于原 360 条工作台，也属于本实验 450 条成员，但规范归档没有通过发布校验。'
            '本次合并没有修复原始数据、重跑仿真、重新判定或解除隔离。</p>'
            f'<p>原隔离原因：{escape(reason)}</p>'
            '<p>原生成功、阶段完成及有效试验分母均不纳入本次统计。原工作台仍保持历史页面，'
            '其显示不代表此条已恢复为有效成绩。</p></section>'
            f'<p><a class="historical-replay-link" href="{escape(original, quote=True)}">查看保留的历史回放与诊断图 →</a></p>'
            f'<p>摘要 SHA-256：<code>{escape(str(trial["summary_sha256"]))}</code></p>'
            '</html>')


def _experiment_page(catalog):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>同实验四个 Checkpoint · 450 条 | KaiTactileSim</title>'
            '<link rel="stylesheet" href="assets/experiment.css">'
            '<script defer src="assets/experiment.js"></script>'
            '<main id="evaluation-status"></main>'
            f'<script type="application/json" id="evaluation-status-data">{inert(catalog)}</script>'
            '<noscript>请启用 JavaScript，或查看 <a href="catalog.json">实验清单</a>。</noscript></html>')


def _ego_catalog(status, output):
    parent, identity = _bound_document(status / "catalog.json")
    catalog = copy.deepcopy(parent)
    catalog["groups"] = [group for group in catalog["groups"] if group["model"] == "egotouch"]
    tasks = {group["task"] for group in catalog["groups"]}
    catalog["tasks"] = [task for task in catalog["tasks"] if task["task"] in tasks]
    catalog["jobs"], catalog["excluded"] = [], []
    catalog["links"] = [{"label": "返回 π0.5 H50 同实验四个 checkpoint（450 条）", "url": "index.html"}]
    catalog.pop("experiment", None)
    catalog["independent_experiment"] = "EgoTouch; not pooled with the pi05 H50 experiment."
    catalog["publication"] = dict(presentation_only=True, simulation_runs_started=0,
                                  member_records=sum(len(g["trials"]) for g in catalog["groups"]))
    for group in catalog["groups"]:
        for trial in group["trials"]:
            for key in ("detail_url", "video_url"):
                href = trial.get(key)
                if href:
                    trial[key] = Path(os.path.relpath(_target(status, href), output)).as_posix()
    return catalog, identity


def export_experiment(status_snapshot, historical_report, output, *, publish_index=False):
    from .experiment import build_experiment_catalog

    status = Path(status_snapshot).resolve(strict=True)
    historical = Path(historical_report).resolve(strict=True)
    output = Path(output).absolute()
    if output.resolve() != output or output.exists() or output.is_symlink():
        raise ValueError("Export experiment into a fresh canonical output directory")
    if output.parent != status.parent or output.parent != historical.parent:
        raise ValueError("Experiment must be a sibling of both preserved snapshots")
    portal = output.parent / "index.html"
    before = portal.read_bytes() if portal.exists() else None
    verify_status(status)
    catalog = build_experiment_catalog(status, historical)
    inputs = _identities(status, historical, catalog)
    # Explicit aliases allow independent frontends to display the same counts.
    exp = catalog["experiment"]
    exp.update(total_records=exp["member_trials"], eligible_records=exp["eligible_trials"],
               quarantined_records=exp["quarantined_trials"])
    catalog["links"] = [
        {"label": "EgoTouch：独立实验（120 条）", "url": "egotouch.html"},
        {"label": "保留的原 360 条工作台", "url": "../" + historical.name + "/index.html"},
        {"label": "CPM / FTP-1 扑克诊断链路", "url": "../tactile_integration_20261019/accepted_summary.html"},
    ]
    output.mkdir()
    artifacts, wrappers = [], 0
    for group in catalog["groups"]:
        for trial in group["trials"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trial["id"]):
                raise ValueError("Unsafe experiment trial ID")
            if trial.get("publication_status") == "quarantined_replay_only":
                if str(_target(output, trial["historical_detail_url"])) not in inputs:
                    raise ValueError("Historical quarantine detail is not verified")
                relative = Path("quarantine") / trial["id"] / "index.html"
                trial["detail_url"] = relative.as_posix()
                trial["detail_type"] = "quarantine_notice_then_preserved_replay"
                _write(output, relative, _quarantine_page(group, trial).encode(), artifacts)
                wrappers += 1
            elif str(_target(output, trial["detail_url"])) not in inputs:
                raise ValueError("Experiment detail is not in a verified snapshot")
    ego, identity = _ego_catalog(status, output)
    if identity["path"] in inputs:
        verify_row(inputs[identity["path"]])
    inputs[identity["path"]] = identity
    for group in ego["groups"]:
        for trial in group["trials"]:
            for key in ("detail_url", "video_url"):
                if trial.get(key) and str(_target(output, trial[key])) not in inputs:
                    raise ValueError("Independent experiment target is not in a verified snapshot")
    for name in ("experiment.js", "experiment.css"):
        _copy(output, Path("assets") / name, PACKAGE / "web" / name, artifacts, inputs)
    for name in ("status.js", "status.css"):
        _copy(output, Path("assets") / name, status / "assets" / name, artifacts, inputs)
    catalog["publication"] = dict(simulation_runs_started=0, source_writes=0, preserved_site_writes=0,
                                  new_quarantine_notice_pages=wrappers, media_copies=0)
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    _write(output, Path("index.html"), _experiment_page(catalog).encode(), artifacts)
    _write(output, Path("egotouch_catalog.json"), encoded(ego), artifacts)
    _write(output, Path("egotouch.html"), _portal(ego).encode(), artifacts)
    if before is not None:
        _write(output, Path("previous_root_index.html"), before, artifacts)
    verification = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Presentation-only experiment membership; original native/staged protocol groups and archive quarantine preserved.",
        parent_status=str(status), historical_root=str(historical),
        source_files=list(inputs.values()), retired_original_sources_verified=False,
        simulation_runs_started=0, source_writes=0, preserved_site_writes=0)
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded(dict(schema=SCHEMA, files=artifacts)), [])
    result = verify_experiment(output)
    if publish_index:
        relative = output.name + "/index.html"
        raw = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<meta http-equiv="refresh" content="0;url={escape(relative, quote=True)}">'
               '<title>KaiTactileSim · 同实验四个 checkpoint</title>'
               f'<p><a href="{escape(relative, quote=True)}">π0.5 H50 同实验 · 450 条</a></p></html>').encode()
        _publish_portal(output, portal, before, raw)
    return {**result, "output": str(output), "published": publish_index}


def verify_experiment(output):
    output = Path(output).resolve(strict=True)
    manifest = document(output / "artifact_manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported experiment snapshot schema")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe experiment snapshot artifact")
        verify_row(row, _inside(output, output / relative))
    verification = document(output / "verification.json")
    for row in verification["source_files"]:
        verify_row(row)
    catalog = document(output / "catalog.json")
    groups = catalog["groups"]
    count = sum(len(group["trials"]) for group in groups)
    eligible = sum(group["valid_trials"] for group in groups)
    quarantined = sum(trial.get("publication_status") == "quarantined_replay_only"
                      for group in groups for trial in group["trials"])
    exp = catalog["experiment"]
    if (count, eligible, quarantined) != (exp["member_trials"], exp["eligible_trials"], exp["quarantined_trials"]):
        raise ValueError("Experiment membership counters disagree")
    if (output / "portal_publication.pending.json").exists():
        raise ValueError("Incomplete experiment portal publication")
    current = None
    if (output / "portal_publication.json").exists():
        audit = document(output / "portal_publication.json")
        root = output.parent / "index.html"
        if audit.get("state") != "committed" or audit.get("root_index") != str(root):
            raise ValueError("Invalid experiment publication audit")
        current = root.is_file() and digest(root)["sha256"] == audit.get("new_sha256")
    return dict(status="passed", groups=len(groups), trials=count, eligible_trials=eligible,
                quarantined_trials=quarantined, artifacts=len(manifest["files"]),
                root_portal_matches_this_snapshot=current, retired_original_sources_verified=False)
