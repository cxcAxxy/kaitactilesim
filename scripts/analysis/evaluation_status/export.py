"""Publish archived native facts without rewriting or rescoring prior sites.

This is not the historical full-H50 exporter: its retired original-source
guards remain untouched. Only currently readable archive facts and copied
artifacts are verified here; that narrower scope is written into the report.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote

PACKAGE = Path(__file__).resolve().parent
SCHEMA = "kaihand-evaluation-status-snapshot-v1"


def digest(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during hash: {path}")
    return {"path": str(path), "size": after.st_size, "sha256": value}


def verify_row(row, path=None):
    actual = digest(path or row["path"])
    if actual["sha256"] != row["sha256"] or actual["size"] != row["size"]:
        raise ValueError(f"Archived bytes changed: {actual['path']}")


def document(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _bound_document(path):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Metadata changed during read: {path}")
    value = json.loads(raw, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    return value, dict(path=str(path), size=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def inert(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _inside(root, path):
    root, path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Artifact escapes its snapshot: {path}")
    return path


def _write(root, relative, raw, artifacts):
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe output artifact path")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    artifacts.append(dict(path=relative.as_posix(), size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))


def _copy(root, relative, source, artifacts, inputs, *, expected=None):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe output artifact path")
    source = Path(source).resolve(strict=True)
    before = source.stat()
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum, size = hashlib.sha256(), 0
    with source.open("rb") as original, path.open("xb") as target:
        for block in iter(lambda: original.read(1024 * 1024), b""):
            target.write(block)
            checksum.update(block)
            size += len(block)
    after = source.stat()
    value = checksum.hexdigest()
    if ((before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns)
            or size != before.st_size or expected is not None and value != expected):
        raise ValueError(f"Source changed during copy: {source}")
    inputs[str(source)] = dict(path=str(source), size=size, sha256=value, role="copied_native_artifact")
    artifacts.append(dict(path=relative.as_posix(), size=size, sha256=value))


def _historical_lookup(historical, output):
    """Verify a copied site's artifact bytes, explicitly recording relocation.

    Old absolute inventories may still name the v1 location after the user's
    v2 copy. Original manifests are never edited. This verifies the actual
    served copy against those same hashes, not the unavailable original code.
    """
    historical = Path(historical).resolve(strict=True)
    artifact_path = historical / "artifact_manifest.json"
    manifest, manifest_identity = _bound_document(artifact_path)
    rows = manifest["files"]
    if not rows:
        raise ValueError("Historical report has no artifact inventory")
    recorded_roots = [p for p in Path(rows[0]["path"]).parents if p.name == historical.name]
    if len(recorded_roots) != 1:
        raise ValueError("Cannot identify historical artifact root")
    recorded_root = recorded_roots[0]
    bound = []
    for row in rows:
        original = Path(row["path"])
        if not original.is_relative_to(recorded_root):
            raise ValueError("Historical artifact inventory has mixed roots")
        actual = _inside(historical, historical / original.relative_to(recorded_root))
        verify_row(row, actual)
        bound.append({**row, "recorded_path": str(original), "path": str(actual)})
    batch_path, sources_path = historical / "batch.json", historical / "source_manifest.json"
    batch, batch_identity = _bound_document(batch_path)
    sources, sources_identity = _bound_document(sources_path)
    verified_paths = {row["path"] for row in bound}
    if str(batch_path) not in verified_paths or str(sources_path) not in verified_paths:
        raise ValueError("Historical identity metadata is not in its artifact inventory")
    by_id = {row["trial_id"]: row for row in batch["trials"]}
    lookup = {}
    for source in sources["trials"]:
        entry = by_id.get(source["trial_id"])
        if entry is None:
            continue
        summaries = [row for row in source["provenance"] if Path(row["path"]).name == "summary.json"]
        if len(summaries) != 1:
            continue
        report = _inside(historical, historical / entry["report_path"])
        if str(report) not in verified_paths:
            raise ValueError("Historical report link is not in its artifact inventory")
        key = summaries[0]["sha256"]
        href = quote(Path(os.path.relpath(report, output)).as_posix(), safe="/._-")
        if key in lookup and lookup[key] != href:
            raise ValueError("Ambiguous preserved report summary identity")
        lookup[key] = href
    metadata = [manifest_identity, batch_identity, sources_identity, digest(historical / "index.html")]
    return lookup, bound, metadata


_DETAIL_CSS = """
body{margin:0;background:#f3f6f8;color:#142b37;font:15px/1.6 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:28px}
a{color:#096c79}header{margin-bottom:22px}h1{font-size:26px;margin:8px 0}h2{font-size:18px}code{overflow-wrap:anywhere}
.detail-grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,1fr);gap:18px}.panel{background:#fff;border:1px solid #dce5e9;border-radius:14px;padding:18px;margin-bottom:18px;min-width:0}
video{width:100%;max-height:76vh;background:#10212b;border-radius:10px}.badge{display:inline-block;padding:3px 10px;margin:3px 7px 3px 0;border-radius:30px;background:#e4f0f1}.note{color:#5d7079;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px;border-bottom:1px solid #edf1f3;overflow-wrap:anywhere}th{width:54%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.plots{display:grid;gap:14px}.plots img{width:100%;height:auto}
@media(max-width:850px){main{padding:15px}.detail-grid{grid-template-columns:1fr}h1{font-size:22px}}
"""


def _facts(summary):
    evaluation = summary.get("evaluation") or {}
    state = evaluation.get("state", evaluation) if isinstance(evaluation, dict) else {}
    if not isinstance(state, dict):
        state = {}
    facts = []
    angles = {"clockwise_turns": ("旋入角度", 360), "clockwise_angle_deg": ("旋入角度", 1),
              "turns_removed": ("旋出角度", 360), "angle_removed_deg": ("旋出角度", 1),
              "rotation_from_initial_turns": ("相对初态旋入角度", 360),
              "rotation_from_initial_deg": ("相对初态旋入角度", 1)}
    displayed_angles = set()
    for key, value in state.items():
        if isinstance(value, (dict, list)) or value is None:
            continue
        label = key
        if key in angles and type(value) in (float, int):
            label, scale = angles[key]
            if label in displayed_angles:
                continue
            displayed_angles.add(label)
            value = f"{value * scale:.6g} °"
        facts.append((label, value))
    return facts[:100]


def _detail(trial, group, summary, plots):
    label = "成功" if trial.get("native_success") is True else "未完成" if trial.get("native_success") is False else "未知"
    validity = "有效诊断记录" if trial.get("valid") and trial.get("mode") == "diagnostic" else "有效记录" if trial.get("valid") else "无效记录（不计成绩）"
    media = ('<video controls preload="metadata" playsinline src="review.mp4"></video>'
             if trial.get("video_url") else '<p>本条没有完整、可核验的视频。</p>')
    facts = "".join(f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>" for key, value in _facts(summary))
    staged = trial.get("staged_result")
    stage_html = '<p class="note">没有可核验的 staged 阶段记录；不补造阶段结果。</p>'
    if isinstance(staged, dict):
        projected = staged.get("staged_result", staged)
        definitions = projected.get("stage_definitions") or {}
        flags = projected.get("stage_flags") or {}
        stage_html = "".join(f'<span class="badge">{"✓" if flags.get(key) is True else "—" if flags.get(key) is False else "?"} {escape(str(name))}</span>'
                             for key, name in definitions.items())
        stage_html += '<p class="note">阶段表示累计曾达到，与原生任务成功分开；白板拾取阶段不等于擦净。</p>'
    plot_html = "".join(f'<figure><img loading="lazy" src="{escape(name, quote=True)}" alt="{escape(name)}"><figcaption>{escape(name)}</figcaption></figure>' for name in plots)
    protocol = json.dumps(group.get("protocol", {}), ensure_ascii=False, indent=2)
    warnings = " ".join(str(value) for value in (*group.get("warnings", []), *trial.get("warnings", [])))
    validity_basis = trial.get("validity_basis", "archived_native_evidence")
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(group["title"])} · seed {trial["seed"]} | KaiTactileSim</title><style>{_DETAIL_CSS}</style></head><body><main>'
            '<header><a href="../../index.html">← 返回各任务最新评估</a>'
            f'<h1>{escape(group["title"])} · {escape(group["model"])} · step {escape(str(group["checkpoint"]))} · seed {trial["seed"]}</h1>'
            f'<span class="badge">原生任务：{label}</span><span class="badge">{validity}</span>'
            f'<span class="badge">H{group.get("prediction_horizon")} / K{group.get("execute_steps")}</span>'
            '<p class="note">原始归档回放和终态事实；没有重跑模型、重判历史规则或对齐不同时间线。</p></header>'
            f'<p class="note">有效性依据：{escape(str(validity_basis))}。{escape(warnings)}</p>'
            f'<div class="detail-grid"><section class="panel">{media}<p class="note">视频保持原始字节与时钟；曲线原图见下方，不与播放器强制同步。</p></section>'
            f'<aside><section class="panel"><h2>累计阶段记录</h2>{stage_html}</section><section class="panel"><h2>原生终态指标</h2><table>{facts or "<tr><td>无可展示的标量终态指标；请查看原始记录。</td></tr>"}</table>'
            '<p class="note">灯泡旋拧量统一显示角度（°）；展示换算不更改原始记录或成功条件。</p></section></aside></div>'
            f'<section class="panel"><h2>归档诊断图</h2><div class="plots">{plot_html or "<p>未归档诊断图。</p>"}</div></section>'
            f'<details class="panel"><summary>评估协议与原始记录</summary><pre>{escape(protocol)}</pre>'
            '<a href="summary.json">下载原始 summary.json</a> · <a href="trial.json">查看导入身份与校验信息</a></details>'
            '</main></body></html>')


def _portal(catalog):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>各任务最新评估 | KaiTactileSim</title><link rel="stylesheet" href="assets/status.css"><script defer src="assets/status.js"></script></head>'
            f'<body><main id="evaluation-status"></main><script type="application/json" id="evaluation-status-data">{inert(catalog)}</script>'
            '<noscript>请启用 JavaScript，或查看 <a href="catalog.json">JSON 状态清单</a>。</noscript></body></html>')


def export_status(source_root, output, historical, *, publish_index=False):
    from .data import build_catalog

    source_root, output, historical = Path(source_root).resolve(strict=True), Path(output).absolute(), Path(historical).resolve(strict=True)
    if output.resolve() != output or output.exists() or output.is_symlink():
        raise ValueError("Export into a fresh canonical output directory")
    if output.is_relative_to(source_root) or output.parent != historical.parent:
        raise ValueError("Output must be a sibling of the preserved site, outside source evaluations")
    old_index = output.parent / "index.html"
    index_before = old_index.read_bytes() if old_index.is_file() else None
    lookup, historical_rows, historical_metadata = _historical_lookup(historical, output)
    catalog = build_catalog(source_root)
    catalog["historical_url"] = quote(Path(os.path.relpath(historical / "index.html", output)).as_posix(), safe="/._-")
    catalog["verification_scope"] = "Current archive metadata and copied artifact bytes; historical artifact-only verification. Retired original source trees are not claimed verified."
    catalog["links"] = [
        {"label": "保留的 360 条 H50 工作台", "url": catalog["historical_url"]},
        {"label": "CPM / FTP-1 扑克诊断链路", "url": "../tactile_integration_20261019/accepted_summary.html"},
        {"label": "各任务触觉模型接入状态", "url": "../tactile_multitask_20261019/readiness.json"},
    ]
    output.mkdir()
    artifacts, inputs = [], {row["path"]: row for row in catalog.get("provenance", [])}
    reused = copied = 0
    for group in catalog["groups"]:
        for trial in group["trials"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trial["id"]):
                raise ValueError("Unsafe trial id")
            if trial["summary_sha256"] in lookup:
                trial.update(detail_url=lookup[trial["summary_sha256"]], detail_type="preserved_workbench")
                reused += 1
                continue
            relative = Path("trials") / trial["id"]
            trial["detail_url"] = (relative / "index.html").as_posix()
            trial["detail_type"] = "native_status_snapshot"
            summary_path = _inside(source_root, trial["summary_path"])
            _copy(output, relative / "summary.json", summary_path, artifacts, inputs, expected=trial["summary_sha256"])
            # Detail facts are parsed from the exact frozen bytes just copied,
            # not a second potentially changed read of the source summary.
            summary = document(output / relative / "summary.json")
            trial["video_url"] = None
            if trial.get("video_path") and trial.get("valid"):
                video = _inside(source_root, trial["video_path"])
                _copy(output, relative / "review.mp4", video, artifacts, inputs)
                trial["video_url"] = (relative / "review.mp4").as_posix()
            plots = []
            source = summary_path.parent
            for directory in (source, source / "review"):
                if not directory.is_dir():
                    continue
                for image in sorted(directory.glob("evaluation_*.png"))[:12]:
                    image = _inside(source, image)
                    name = ("review_" if directory.name == "review" else "") + image.name
                    _copy(output, relative / name, image, artifacts, inputs)
                    plots.append(name)
            _write(output, relative / "index.html", _detail(trial, group, summary, plots).encode(), artifacts)
            _write(output, relative / "trial.json", encoded(trial), artifacts)
            copied += 1
        print(json.dumps({"group": group["id"], "task": group["task"], "valid": group["valid_trials"], "reused": reused, "new_details": copied}), flush=True)
    for name in ("status.js", "status.css"):
        _copy(output, Path("assets") / name, PACKAGE / "web" / name, artifacts, inputs)
    # Mutable job status/log observations are point-in-time UI facts, never
    # silently presented as immutable native trial evidence.
    for row in inputs.values():
        if not row.get("mutable", False):
            verify_row(row)
    for row in historical_metadata:
        verify_row(row)
    catalog["publication"] = dict(reused_preserved_details=reused, new_native_detail_pages=copied,
                                  simulation_runs_started=0, source_writes=0)
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    _write(output, Path("index.html"), _portal(catalog).encode(), artifacts)
    if index_before is not None:
        _write(output, Path("previous_root_index.html"), index_before, artifacts)
    verification = dict(schema=SCHEMA, status="passed", generated_utc=datetime.now(timezone.utc).isoformat(),
        scope=catalog["verification_scope"], retired_original_sources_verified=False,
        source_writes=0, simulation_runs_started=0, historical_site_writes=0,
        historical_artifact_count=len(historical_rows), historical_root=str(historical),
        historical_metadata=historical_metadata, historical_artifact_relocation_verified=True,
        historical_artifacts=historical_rows, source_files=list(inputs.values()),
        publication=catalog["publication"])
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded(dict(schema=SCHEMA, files=artifacts)), [])
    verify_status(output)
    if publish_index:
        relative = output.name + "/index.html"
        raw = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<meta http-equiv="refresh" content="0;url={escape(relative, quote=True)}"><title>KaiTactileSim · 各任务最新评估</title>'
               f'<h1>KaiTactileSim 官方评估平台</h1><p><a href="{escape(relative, quote=True)}">各任务最新评估</a></p>'
               f'<p><a href="{escape(historical.name, quote=True)}/index.html">保留的 360 条 H50 工作台</a></p></html>').encode()
        _publish_portal(output, old_index, index_before, raw)
    return dict(output=str(output), **catalog["publication"], group_count=len(catalog["groups"]),
                trial_count=sum(len(group["trials"]) for group in catalog["groups"]))


def _publish_portal(output, old_index, index_before, raw):
    """Serialize publishers, journal first, and roll back any commit failure."""
    lock = output.parent / ".evaluation-status-publish.lock"
    # Never remove another publisher's lock. A stale lock needs investigation.
    with lock.open("xb") as stream:
        stream.write(encoded(dict(pid=os.getpid(), output=str(output))))
    temporary = output.parent / ("." + output.name + ".portal.tmp")
    rollback = output.parent / ("." + output.name + ".portal.rollback.tmp")
    pending = output / "portal_publication.pending.json"
    audit = output / "portal_publication.json"
    changed = False
    try:
        if (old_index.read_bytes() if old_index.exists() else None) != index_before:
            raise ValueError("Root portal changed during export; snapshot ready but portal not replaced")
        publication = dict(state="committed", root_index=str(old_index), new_sha256=hashlib.sha256(raw).hexdigest(),
            previous_sha256=hashlib.sha256(index_before).hexdigest() if index_before else None,
            backup="previous_root_index.html" if index_before else None)
        with pending.open("xb") as stream:
            stream.write(encoded(publication))
        with temporary.open("xb") as stream:
            stream.write(raw)
        if index_before is not None:
            with rollback.open("xb") as stream:
                stream.write(index_before)
        try:
            os.replace(temporary, old_index)
            changed = True
            os.replace(pending, audit)
            if old_index.read_bytes() != raw or document(audit) != publication:
                raise ValueError("Root portal publication verification failed")
        except BaseException:
            if changed and old_index.read_bytes() == raw:
                if index_before is None:
                    old_index.unlink()
                else:
                    os.replace(rollback, old_index)
            if audit.exists():
                audit.unlink()
            raise
    finally:
        for path in (temporary, rollback, pending):
            if path.exists():
                path.unlink()
        lock.unlink()


def verify_status(output):
    output = Path(output).resolve(strict=True)
    manifest = document(output / "artifact_manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported evaluation status snapshot")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe snapshot manifest path")
        verify_row(row, _inside(output, output / relative))
    verification = document(output / "verification.json")
    for row in verification["historical_artifacts"] + verification["historical_metadata"]:
        verify_row(row)
    for row in verification["source_files"]:
        if not row.get("mutable", False):
            verify_row(row)
    catalog = document(output / "catalog.json")
    portal_current = None
    pending = output / "portal_publication.pending.json"
    if pending.exists():
        raise ValueError("Root portal publication journal is incomplete")
    audit = output / "portal_publication.json"
    if audit.exists():
        publication = document(audit)
        expected_root = output.parent / "index.html"
        if publication.get("state") != "committed" or publication.get("root_index") != str(expected_root):
            raise ValueError("Invalid root portal publication audit")
        portal_current = expected_root.is_file() and digest(expected_root)["sha256"] == publication.get("new_sha256")
    return dict(status="passed", artifacts=len(manifest["files"]), groups=len(catalog["groups"]),
                trials=sum(len(group["trials"]) for group in catalog["groups"]),
                root_portal_matches_this_snapshot=portal_current,
                retired_original_sources_verified=False)
