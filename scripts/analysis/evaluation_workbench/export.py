"""Publish a new layout without recomputing loads or editing baseline artifacts."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tempfile
from urllib.parse import quote

_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from kaihand_tactile_env.shared.task_demo import locate_demo_root, resolve_demo_path

from evaluation_progress_experiment.adapters import MAX_JSON_BYTES, MAX_SOURCE_BYTES, _SHA, _parse, _path, _read
from evaluation_tactile_load_extension.export import _copy_checked
from evaluation_v2.horizon import inspect_prediction_horizon

from . import SCHEMA_VERSION


_WEB_DIR = Path(__file__).parent / "web"
_MEDIA_SHA = "e28542122e5e14da975ecac87a0638bbbcd2598ff17843ce625208717d92f784"
_CROPS = [
    {"id": "head", "label": "头部视角", "x": 214, "y": 78, "width": 434, "height": 324},
    {"id": "wrist", "label": "腕部视角", "x": 214, "y": 440, "width": 434, "height": 294},
    {"id": "global", "label": "全局视角", "x": 214, "y": 746, "width": 434, "height": 324},
]
# Explicit audit record, not a filename-derived pairing. The replay itself has
# no video path/hash; only this audited replay/video/role binding is supported.
_EXPERT_AUDIT = {
    "manifest": {"path": "/nas/chenxianchi/datasets/sim/task_demo/_metadata/whiteboard-wipe/replay.json",
                 "sha256": "6276f686f675ae99fc2f79c4158a12606d4543310ab7041c6ffac8113c901fdb"},
    "video": {"path": "/nas/chenxianchi/datasets/sim/task_demo/whiteboard-wipe.mp4",
              "sha256": "8b8bf394d02eaf949584092af17c5117968ce6f1f6abc34c6b942b2045846423"},
    "membership": {"path": "/cpfs_infra/user/chenxianchi/code/kaitactilesim_laest/reports/evaluation_v1_stage1_20261013/expert_reference_v2_20261013/reference_manifest.json",
                   "sha256": "0f2c2e476f1a2182b5a240a2aede26b8b694e5ed1d2ea3511b9c877c837d4c1d"},
    "report": {"path": "/cpfs_infra/user/chenxianchi/code/kaitactilesim_laest/reports/evaluation_v1_stage1_20261013/expert_reference_v2_20261013/000_whiteboard-wipe_6fa506a8bb6e/report.json",
               "sha256": "417a0401adf513e8f5600ec9f917123b932948ba6fae2c74dad956a6cfe07608"},
    "source_hdf5": "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/0929_200/whiteboard-wipe/000000/attempt_001/data/raw/episode.h5",
    "source_sha256": "6fa506a8bb6ebd0950c1502df178202b8c602ee26533b85ae022987ae5a4bdd6",
    "trial_id": "expert:whiteboard-wipe:6fa506a8bb6ebd0950c1",
}


def _digest(path):
    path = _path(path)
    _, digest, size, _ = _read(path, limit=MAX_SOURCE_BYTES)
    return {"path": str(path), "sha256": digest, "size": size}


def _document(path, *, html=False):
    path = _path(path)
    raw, _, _, _ = _read(path, limit=50_000_000 if html else MAX_JSON_BYTES, collect=True)
    return raw.decode("utf-8") if html else _parse(raw)


def _rows(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 5000:
        raise ValueError("Expected bounded nonempty SHA256 provenance")
    unique = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not Path(row["path"]).is_absolute() or not isinstance(row.get("sha256"), str)
                or not _SHA.fullmatch(row["sha256"])
                or ("size" in row and (type(row["size"]) is not int or row["size"] < 0))):
            raise ValueError("Invalid source provenance row")
        previous = unique.get(row["path"])
        if previous and (previous["sha256"].lower() != row["sha256"].lower()
                         or ("size" in row and "size" in previous and row["size"] != previous["size"])):
            raise ValueError("Conflicting SHA256 or size provenance")
        unique[row["path"]] = row
    return list(unique.values())


def _verify(rows):
    for row in _rows(rows):
        canonical = row.get("resolved_path")
        if canonical is not None and str(Path(row["path"]).resolve(strict=True)) != canonical:
            raise ValueError(f"Bound source alias target changed: {row['path']}")
        actual = _digest(canonical or row["path"])
        if actual["sha256"] != row["sha256"].lower() or ("size" in row and actual["size"] != row["size"]):
            raise ValueError(f"Protected source SHA256/size changed: {row['path']}")
        if canonical is not None and str(Path(row["path"]).resolve(strict=True)) != canonical:
            raise ValueError(f"Bound source alias target changed while reading: {row['path']}")


def _pin_bound_aliases(rows):
    """Resolve only pre-existing SHA-bound provenance, never report-tree links."""
    pinned = []
    for original in _rows(rows):
        row = dict(original)
        if Path(row["path"]).is_symlink():
            raise ValueError(f"Leaf symlink in SHA-bound source is unsupported: {row['path']}")
        target = str(Path(row["path"]).resolve(strict=True))
        if target != row["path"]:
            row["resolved_path"] = target
        pinned.append(row)
    _verify(pinned)
    return pinned


def _files(root):
    root = _path(root)
    if not root.is_dir():
        raise ValueError(f"Expected an artifact directory: {root}")
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in protected artifacts: {path}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        mode = path.stat().st_mode
        if stat.S_ISREG(mode):
            result.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"Non-regular protected artifact: {path}")
        if len(result) > 5000:
            raise ValueError("Protected artifact inventory is oversized")
    return result


def _old_code_roots():
    analysis = Path(__file__).resolve().parents[1]
    return [analysis / name for name in ("evaluation_v1", "evaluation_v2", "evaluation_progress_experiment",
                                        "evaluation_action_review", "evaluation_tactile_load_extension")]


def _old_code_files():
    analysis = Path(__file__).resolve().parents[1]
    names = ("evaluate_trials.py", "evaluate_suite.py", "export_expert_reference.py", "serve_evaluation.py",
             "compare_task_progress.py", "review_action_evidence.py", "extend_tactile_loads.py")
    return [analysis / name for name in names if (analysis / name).is_file()]


def _implementation():
    package = Path(__file__).parent
    paths = list(package.glob("*.py")) + [_WEB_DIR / "workbench.js", _WEB_DIR / "workbench.css",
                                          package.parent / "export_evaluation_workbench.py"]
    return [_digest(path) for path in sorted(paths)]


def _inventory(roots):
    return [{"path": str(_path(root)), "files": [str(path) for path in _files(root)]} for root in roots]


def _verify_inventory(inventory):
    for entry in inventory:
        if [str(path) for path in _files(entry["path"])] != entry["files"]:
            raise ValueError(f"Protected file inventory changed: {entry['path']}")


def _embedded(html, identifier):
    pattern = rf'<script\b[^>]*\bid=["\']{re.escape(identifier)}["\'][^>]*>(.*?)</script>'
    found = re.findall(pattern, html, flags=re.S | re.I)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one embedded payload: {identifier}")
    return found[0], _parse(found[0])


def _relative_media(report_dir, media):
    url = media.get("url")
    if (media.get("status") != "available" or media.get("sync_status") != "mapped"
            or not isinstance(url, str) or not re.fullmatch(r"media/[A-Za-z0-9_.-]+\.mp4", url)):
        raise ValueError("A mapped episode-local media file is required")
    path = _path(report_dir / url)
    actual = _digest(path)
    if media.get("sha256") != _MEDIA_SHA or actual["sha256"] != _MEDIA_SHA:
        raise ValueError("Video SHA256 does not match the independently audited crop layout")
    return path


def _validate_payloads(report_path, stable_site):
    report = _document(report_path)
    visual = _document(report_path.parent / "visualization.json")
    loads = _document(report_path.parent / "tactile_loads.json")
    html = _document(report_path.with_suffix(".html"), html=True)
    _, embedded = _embedded(html, "kai-evaluation-data")
    _, embedded_loads = _embedded(html, "kai-tactile-load-data")
    if embedded.get("report") != report or embedded.get("visualization") != visual or embedded_loads != loads:
        raise ValueError("Embedded report/visualization/load payload differs from its JSON file")
    if (report.get("schema_version") != "task-tactile-evaluation-v1"
            or report.get("suite_schema_version") != "task-tactile-suite-v2"
            or visual.get("schema_version") != "evaluation-visualization-v1"
            or loads.get("schema_version") != "evaluation-five-finger-load-extension-v1"
            or type(loads.get("prediction_horizon")) is not int or loads["prediction_horizon"] != 50):
        raise ValueError("Unsupported report/visualization/load schema or H50 declaration")
    provenance = _pin_bound_aliases(loads.get("provenance"))
    bindings = {row["path"]: row for row in provenance}
    _verify(provenance)
    _verify(loads.get("implementation"))
    original = _path(loads.get("source_report"))
    h50 = _path(loads.get("h50_report"))
    if h50.parent.parent != stable_site:
        raise ValueError("H50 report is outside the inherited stable site")
    for path in (original, original.parent / "visualization.json", h50, h50.parent / "visualization.json"):
        if str(path) not in bindings:
            raise ValueError("Archived report/visualization is not bound in load provenance")
    if (_digest(report_path)["sha256"] != bindings[str(original)]["sha256"]
            or _digest(report_path.parent / "visualization.json")["sha256"] != bindings[str(original.parent / "visualization.json")]["sha256"]
            or bindings[str(original.parent / "visualization.json")]["sha256"] != bindings[str(h50.parent / "visualization.json")]["sha256"]):
        raise ValueError("Copied report/visualization SHA256 differs from its archived inputs")
    stable_report = _document(h50)
    trial, stable_trial = report.get("trial", {}), stable_report.get("trial", {})
    if (trial.get("trial_id") != loads.get("trial_id") or trial.get("task") != "whiteboard-wipe"
            or loads.get("task") != "whiteboard-wipe"
            or any(trial.get(key) != stable_trial.get(key) for key in ("trial_id", "task", "source"))
            or type(stable_trial.get("prediction_horizon")) is not int or stable_trial["prediction_horizon"] != 50):
        raise ValueError("Trial/source identity or H50 report differs")
    source = Path(trial.get("source", ""))
    source_summary = bindings.get(str(source / "summary.json"))
    if source_summary is None:
        raise ValueError("Source summary has no pinned identity")
    canonical_summary = Path(source_summary.get("resolved_path", source_summary["path"]))
    horizon = inspect_prediction_horizon(canonical_summary.parent)
    if horizon["status"] != "verified" or horizon["prediction_horizon"] != 50:
        raise ValueError("Source summary/deployment does not verify H50")
    for row in horizon["provenance"]:
        matching = [bound for bound in provenance if bound.get("resolved_path", bound["path"]) == row["path"]]
        if not matching or any(row["sha256"] != bound["sha256"] for bound in matching):
            raise ValueError("Source H50 evidence is not bound in load provenance")
    batch = _document(stable_site / "batch.json")
    if batch.get("horizon_selection", {}).get("required_prediction_horizon") != 50:
        raise ValueError("Stable cohort is not explicitly H50-only")
    member = False
    for entry in batch.get("trials", []):
        relative = Path(entry.get("report_path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe stable report membership path")
        member |= entry.get("prediction_horizon") == 50 and (stable_site / relative).with_suffix(".json") == h50
    if not member:
        raise ValueError("H50 report is not a member of the stable cohort")
    media = visual.get("media", {})
    _relative_media(report_path.parent, media)
    return report, visual, loads, html, provenance


def _expert_attachment(manifest_path):
    """Validate the audited video and explicit reference membership, not HDF data."""
    audit = _EXPERT_AUDIT
    demo_root = locate_demo_root(manifest_path)
    manifest_path = _path(resolve_demo_path(manifest_path, demo_root=demo_root,
                                          expected_sha256=audit["manifest"]["sha256"]))
    demo_root = locate_demo_root(manifest_path)
    audited_manifest = resolve_demo_path(audit["manifest"]["path"], demo_root=demo_root,
                                        expected_sha256=audit["manifest"]["sha256"])
    if manifest_path != audited_manifest:
        raise ValueError("Expert manifest path is not an explicitly audited media binding")
    provenance = []
    if demo_root is not None:
        provenance.append({**_digest(demo_root / "migration_manifest.json"),
                           "role": "expert_video_migration_manifest"})
    selected = {}
    for key in ("manifest", "video", "membership", "report"):
        # Only migrated task-demo artifacts are relocated. The archived
        # reference report/membership remain mandatory original audit inputs.
        selected[key] = (resolve_demo_path(audit[key]["path"], demo_root=demo_root,
                                          expected_sha256=audit[key]["sha256"])
                         if key in {"manifest", "video"} else Path(audit[key]["path"]))
        if demo_root is not None and key in {"manifest", "video"} and not selected[key].is_relative_to(demo_root):
            raise ValueError("Audited media has no exact local task-demo mapping; refusing a NAS fallback")
        actual = _digest(selected[key])
        if actual["sha256"] != audit[key]["sha256"]:
            raise ValueError(f"Expert {key} SHA256 differs from the audited binding")
        row = {**actual, "role": "expert_video_" + key}
        if actual["path"] != audit[key]["path"]:
            row["original_path"] = audit[key]["path"]
        provenance.append(row)
    replay = _document(manifest_path)
    if (replay.get("schema_version") != "kaihand-offline-multimodal-replay-v1"
            or replay.get("task") != "whiteboard-wipe" or replay.get("completed") is not True
            or replay.get("source_modified") is not False or replay.get("physics_rerun") is not False
            or replay.get("source_hdf5") != audit["source_hdf5"]
            or replay.get("source_sha256") != audit["source_sha256"]
            or replay.get("output_size") != [1920, 1080] or replay.get("output_frame_count") != 669
            or replay.get("fps") != 10.0 or replay.get("video_duration_s") != 66.9):
        raise ValueError("Expert replay task/completion/HDF/video-clock declaration is unsupported")
    reference = _document(audit["report"]["path"])
    trial, imported = reference.get("trial", {}), reference.get("reference_import", {})
    if (trial.get("trial_id") != audit["trial_id"] or trial.get("task") != "whiteboard-wipe"
            or trial.get("source") != audit["source_hdf5"] or imported.get("status") != "imported"
            or imported.get("source_schema") != "kaihand_tactile_episode_v1"
            or imported.get("source_sha256_verified") is not True):
        raise ValueError("Expert reference report does not declare the audited recording identity")
    # HDF identity is checked against its already verified archived import. The
    # 607 MB HDF is not reread merely to attach its existing offline video.
    reference_rows = _rows(reference.get("provenance"))
    for path, sha, role in ((audit["source_hdf5"], audit["source_sha256"], "expert_hdf5"),
                           (audit["manifest"]["path"], audit["manifest"]["sha256"], "explicit_task_demo_replay_selection")):
        matching = [row for row in reference_rows if row["path"] == path and row.get("role") == role]
        if len(matching) != 1 or matching[0]["sha256"] != sha:
            raise ValueError("Expert reference HDF/replay provenance binding disagrees")
    membership = _document(audit["membership"]["path"])
    members = membership.get("members")
    if (membership.get("schema_version") != "conditional-reference-manifest-v1"
            or membership.get("role") != "expert_reference" or not isinstance(members, list) or len(members) > 10000
            or any(not isinstance(membership.get(key), str) or not membership[key].strip()
                   for key in ("declared_by", "selection_rule", "evidence"))):
        raise ValueError("Explicit expert-reference role and selection declaration are required")
    matches = [row for row in members if isinstance(row, dict) and row.get("trial_id") == audit["trial_id"]]
    if (len(matches) != 1 or matches[0].get("source") != audit["source_hdf5"]
            or matches[0].get("observation_sha256") != audit["source_sha256"]):
        raise ValueError("Expert is not a unique source-SHA-bound reference member")
    _verify(provenance)
    expert = {
        "status": "available", "url": "media/expert.mp4", "task": "whiteboard-wipe",
        "source_label": "白板专家示范 · 0929_200 / 000000 / attempt_001",
        "description": "由记录的脚本示范轨迹离线重放，不是真机人工示教。",
        "duration_s": 66.9, "width": 1920, "height": 1080, "fps": 10,
        "sha256": audit["video"]["sha256"], "clock": "independent_video_seconds",
        "comparison_note": "同任务的单条独立示范，初始墨迹与模型不同，不证明场景及动力学完全相同；仅使用独立视频秒，不与模型时间对齐，不参与当前指标计算。",
        "provenance": provenance, "source_hdf5": audit["source_hdf5"], "source_sha256": audit["source_sha256"],
        "source_verification": "Existing SHA-bound reference import certifies the archived HDF identity; HDF bytes are not reloaded/rehashed for this video-only attachment.",
        "media_binding": "Explicit audited manifest/video SHA pair; replay has no video path/hash, and no pairing is inferred from names.",
        "selection_bias": membership["selection_rule"],
        "media_views": {"status": "verified", "sha256": audit["video"]["sha256"], "source_width": 1920, "source_height": 1080,
                        "views": [{"id": "head", "label": "头部视角", "x": 10, "y": 108, "width": 632, "height": 456}]},
    }
    return expert, _path(selected["video"])


def _render(html, config):
    if html.count("</head>") != 1 or html.count("</body>") != 1 or "kai-workbench-config" in html:
        raise ValueError("Unsupported or already-workbench page structure")
    original_payloads = {name: _embedded(html, name)[0] for name in ("kai-evaluation-data", "kai-tactile-load-data")}
    patterns = (
        r'<link\b[^>]*\bhref=["\']\.\./assets/tactile_loads\.css["\'][^>]*>',
        r'<script\b[^>]*\bsrc=["\']\.\./assets/tactile_loads\.js["\'][^>]*>\s*</script>',
    )
    for pattern in patterns:
        html, count = re.subn(pattern, "", html, flags=re.I)
        if count != 1:
            raise ValueError("Expected exactly one legacy tactile addon script/style reference")
    for asset in ("dashboard.js", "bootstrap.js"):
        if len(re.findall(rf'<script\b[^>]*\bsrc=["\']\.\./assets/{re.escape(asset)}["\']', html, flags=re.I)) != 1:
            raise ValueError(f"Required base dashboard script is missing or duplicated: {asset}")
    payload = json.dumps(config, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = html.replace("</head>", '<link rel="stylesheet" href="../assets/workbench.css">\n</head>')
    html = html.replace("</body>", f'<script type="application/json" id="kai-workbench-config">{payload}</script>\n'
                                    '<script src="../assets/workbench.js" defer></script>\n</body>')
    if any(_embedded(html, key)[0] != value for key, value in original_payloads.items()):
        raise ValueError("Original embedded payload bytes changed during layout rendering")
    return html


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def _diagnostic_page(config):
    payload = json.dumps(config, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>灯泡任务 · 全程诊断原图</title><link rel="stylesheet" href="../assets/workbench.css">'
            '</head><body><main id="kai-diagnostic-plots" class="kai-eval wb-root"></main>'
            f'<script type="application/json" id="kai-diagnostic-config">{payload}</script>'
            '<script src="../assets/workbench.js" defer></script></body></html>')


def _verify_diagnostic_copies(config, directory):
    if config.get("status") != "available" or not isinstance(config.get("plots"), list) or len(config["plots"]) != 3:
        raise ValueError("Unsupported attached diagnostic configuration")
    for plot in config["plots"]:
        url = plot.get("url")
        if not isinstance(url, str) or not re.fullmatch(r"plots/(?:[A-Za-z0-9_-]+/)?[A-Za-z0-9_-]+\.png", url):
            raise ValueError("Unsafe copied diagnostic URL")
        if _digest(directory / url)["sha256"] != plot.get("sha256"):
            raise ValueError("Copied diagnostic PNG SHA256 differs before publication")


def verify_rollback(path):
    manifest = _document(path)
    if manifest.get("schema_version") != "evaluation-experiment-rollback-v1":
        raise ValueError("Not a supported rollback manifest")
    rows = _rows(manifest.get("protected_files"))
    _verify(rows)
    _verify_inventory(manifest.get("protected_roots", []))
    for witness in manifest.get("protected_artifact_manifests", []):
        _verify([witness])
        root = Path(witness["path"]).parent
        for row in _document(witness["path"])["files"]:
            if not Path(row["path"]).is_relative_to(root):
                raise ValueError("Protected artifact manifest escapes its preserved report site")
            _verify([row])
    return {"verified": True, "protected_file_count": len(rows)}


def refresh_web(output: Path) -> dict:
    """Explicitly refresh only this workbench's two assets and implementation audit.

    Archived inputs and every other output file are immutable. Stage complete
    copies before replacement and restore the original allowlisted files if a
    commit or final verification fails; neither data nor HTML is regenerated.
    """
    output = _path(output)
    if not output.is_dir():
        raise ValueError("Web refresh requires an existing workbench output directory")
    rollback = _document(output / "rollback.json")
    if rollback.get("experiment") != SCHEMA_VERSION:
        raise ValueError("Web refresh accepts only an explicitly exported workbench")
    checked = verify_rollback(output / "rollback.json")
    baseline, stable = _path(rollback.get("baseline_site")), _path(rollback.get("stable_site"))
    if (output in {baseline, stable} or output.parent != baseline.parent or output.parent != stable.parent
            or not isinstance(rollback.get("protected_roots"), list)
            or not any(entry.get("path") == str(baseline) for entry in rollback["protected_roots"])):
        raise ValueError("Web refresh output must be the new sibling, never an archived/protected site")
    source_html = _path(rollback.get("source_page"))
    if source_html.name != "report.html" or source_html.parent.parent != baseline:
        raise ValueError("Workbench baseline report path is not bound to its source site")
    config = _document(output / "workbench.json")
    report, _, loads, original_html, _ = _validate_payloads(source_html.with_suffix(".json"), stable)
    if (config.get("schema_version") != SCHEMA_VERSION or config.get("trial_id") != loads["trial_id"]
            or type(config.get("prediction_horizon")) is not int or config["prediction_horizon"] != 50
            or config.get("baseline_url") != f"../../{quote(baseline.name, safe='')}/{quote(source_html.parent.name, safe='')}/report.html"
            or config.get("stable_url") != f"../../{quote(stable.name, safe='')}/index.html"):
        raise ValueError("Workbench config identity/H50/rollback links disagree")
    report_dir = output / source_html.parent.name
    if "expert_video" in config:
        if not rollback.get("expert_manifest"):
            raise ValueError("Expert attachment is not declared by this workbench export")
        expert, _ = _expert_attachment(rollback["expert_manifest"])
        if config["expert_video"] != expert or _digest(report_dir / "media/expert.mp4")["sha256"] != expert["sha256"]:
            raise ValueError("Expert video/config differs from its exported binding")
    elif rollback.get("expert_manifest"):
        raise ValueError("Exported expert attachment is missing from workbench config")
    if "diagnostic_plots" in config:
        _verify_diagnostic_copies(config["diagnostic_plots"], report_dir)
    if "diagnostic_example_url" in config:
        if config["diagnostic_example_url"] != "../diagnostics/bulb-unscrew_step20000_seed_000.html":
            raise ValueError("Unsupported diagnostic example URL")
        example_page = _document(output / "diagnostics/bulb-unscrew_step20000_seed_000.html", html=True)
        _, example_config = _embedded(example_page, "kai-diagnostic-config")
        _verify_diagnostic_copies(example_config, output / "diagnostics")
    page = _document(report_dir / "report.html", html=True)
    if page != _render(original_html, config):
        raise ValueError("Workbench HTML/config or original embedded payload bytes changed")
    for name in ("report.json", "visualization.json", "tactile_loads.json"):
        if _digest(report_dir / name)["sha256"] != _digest(source_html.parent / name)["sha256"]:
            raise ValueError("Workbench report/data payload differs from immutable baseline")
    inventory = _files(output)
    before = [_digest(path) for path in inventory]
    by_path = {row["path"]: row for row in before}
    assets = [output / "assets" / name for name in ("workbench.js", "workbench.css")]
    verification_path = output / "verification.json"
    allowlist = assets + [verification_path]
    if any(str(path) not in by_path for path in allowlist):
        raise ValueError("Expected workbench assets/audit do not already exist")
    protected = _rows(rollback["protected_files"])
    protected_paths = {row.get("resolved_path", row["path"]) for row in protected}
    if any(str(path) in protected_paths for path in allowlist):
        raise ValueError("Refresh target is protected; refusing to replace it")
    immutable = [row for row in before if Path(row["path"]) not in allowlist]
    implementation = _implementation()
    new_sources = {row["path"]: row for row in implementation}
    verification = _document(verification_path)
    if verification.get("status") != "passed" or verification.get("fn_ft_recomputed") is not False:
        raise ValueError("Workbench has no passing layout-only export verification")
    count = verification.get("web_refresh_count", 0)
    if type(count) is not int or count < 0:
        raise ValueError("Invalid web refresh audit counter")
    verification.update(implementation=implementation, web_refresh_count=count + 1,
                        web_refresh={"allowlist": [str(path.relative_to(output)) for path in allowlist],
                                     "protected_sources_rechecked": True, "other_output_bytes_unchanged": True})
    changed = []
    with tempfile.TemporaryDirectory(prefix=".workbench-refresh-", dir=output) as temporary:
        stage = Path(temporary)
        for destination in allowlist:
            _copy_checked(destination, stage / "backup" / destination.name, by_path[str(destination)])
        for destination in assets:
            source = _path(_WEB_DIR / destination.name)
            if str(source) not in new_sources:
                raise ValueError("Current workbench asset lacks an implementation digest")
            _copy_checked(source, stage / "next" / destination.name, new_sources[str(source)])
        _write_json(stage / "next" / verification_path.name, verification)
        _verify(before)
        _verify(implementation)
        verify_rollback(output / "rollback.json")
        try:
            for destination in allowlist:
                os.replace(stage / "next" / destination.name, destination)
                changed.append(destination)
            _verify(immutable)
            _verify(implementation)
            verify_rollback(output / "rollback.json")
            for destination in assets:
                if _digest(destination)["sha256"] != new_sources[str((_WEB_DIR / destination.name).resolve())]["sha256"]:
                    raise ValueError("Refreshed workbench asset hash mismatch")
        except BaseException:
            for destination in reversed(changed):
                os.replace(stage / "backup" / destination.name, destination)
            raise
    if _files(output) != inventory:
        raise ValueError("Workbench file inventory changed during web refresh")
    _verify(immutable)
    return {"refreshed": True, "output": str(output), "updated": [str(path.relative_to(output)) for path in allowlist],
            "protected_file_count": checked["protected_file_count"], "web_refresh_count": count + 1,
            "payloads_unchanged": True, "fn_ft_recomputed": False}


def export_workbench(source_report: Path, output: Path, *, expert_manifest=None, previous_workbench=None,
                     diagnostic_plots=False, diagnostic_example=None) -> dict:
    if (expert_manifest is None) != (previous_workbench is None):
        raise ValueError("Expert video and previous workbench must be explicitly supplied together")
    source = _path(source_report)
    if source.name not in {"report.html", "report.json"}:
        raise ValueError("Source must be an archived report.html or report.json")
    source = _path(source.with_suffix(".json"))
    site = source.parent.parent
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Use a new output directory: {output}")
    if output.parent != _path(site.parent) or output.resolve() != output:
        raise ValueError("Output must be a new, non-symlink sibling of the source site")
    rollback_path = site / "rollback.json"
    inherited = _document(rollback_path)
    verify_rollback(rollback_path)
    stable = _path(inherited.get("stable_site"))
    if stable.parent != site.parent:
        raise ValueError("Stable H50 site must be a sibling of the source site")
    previous, expert, expert_source = None, None, None
    extra_protected = []
    if expert_manifest is not None:
        previous = _path(previous_workbench)
        if not previous.is_dir() or previous.parent != site.parent or previous in {site, stable, output}:
            raise ValueError("Previous workbench must be a distinct existing sibling")
        previous_rollback = _document(previous / "rollback.json")
        previous_config = _document(previous / "workbench.json")
        if (previous_rollback.get("experiment") != SCHEMA_VERSION or previous_config.get("schema_version") != SCHEMA_VERSION
                or previous_rollback.get("source_page") != str(source.with_suffix(".html"))):
            raise ValueError("Previous workbench does not bind the same model baseline")
        verify_rollback(previous / "rollback.json")
        extra_protected = previous_rollback["protected_files"]
        expert, expert_source = _expert_attachment(expert_manifest)
        if (source.parent / "media/expert.mp4").exists():
            raise ValueError("Baseline already contains an expert media file; no overwrite allowed")
    protected_roots = _inventory([site] + _old_code_roots() + ([previous] if previous else []))
    source_files = [Path(path) for path in protected_roots[0]["files"]]
    snapshots = [_digest(path) for root in protected_roots for path in root["files"]]
    snapshots.extend(_digest(path) for path in _old_code_files())
    report, visual, loads, html, provenance = _validate_payloads(source, stable)
    if previous and previous_config.get("trial_id") != loads["trial_id"]:
        raise ValueError("Previous workbench trial identity differs")
    diagnostic, example, diagnostic_copies, diagnostic_provenance = None, None, [], []
    example_name = "bulb-unscrew_step20000_seed_000.html"
    if diagnostic_plots or diagnostic_example is not None:
        from .diagnostic_plots import load_diagnostic_plots
        if diagnostic_plots:
            summary_path = str(Path(report["trial"]["source"]) / "summary.json")
            bound_summary = next(row for row in provenance if row["path"] == summary_path)
            trial = Path(bound_summary.get("resolved_path", summary_path)).parent
            diagnostic, rows = load_diagnostic_plots(trial, expected_task=report["trial"]["task"],
                                                     expected_summary_sha=bound_summary["sha256"])
            diagnostic_copies.extend((row, Path(source.parent.name) / row["url"]) for row in rows)
            diagnostic_provenance.extend(diagnostic["provenance"])
        if diagnostic_example is not None:
            example, rows = load_diagnostic_plots(diagnostic_example, expected_task="bulb-unscrew", url_prefix="plots/bulb")
            if example["seed"] != 0 or example["checkpoint"] != "20000":
                raise ValueError("Diagnostic bulb example must bind explicit seed 0 and checkpoint 20000")
            example["workbench_url"] = f"../{quote(source.parent.name, safe='')}/report.html"
            diagnostic_copies.extend((row, Path("diagnostics") / row["url"]) for row in rows)
            diagnostic_provenance.extend(example["provenance"])
    protected = _rows(inherited["protected_files"] + extra_protected + snapshots + provenance + loads["implementation"]
                      + (expert["provenance"] if expert else []) + diagnostic_provenance)
    source_rows = {row["path"]: row for row in protected}
    implementation = _implementation()
    config = {
        "schema_version": SCHEMA_VERSION, "trial_id": loads["trial_id"], "prediction_horizon": 50,
        "baseline_url": f"../../{quote(site.name, safe='')}/{quote(source.parent.name, safe='')}/report.html",
        "stable_url": f"../../{quote(stable.name, safe='')}/index.html",
        "media_views": {"status": "verified", "sha256": _MEDIA_SHA, "source_width": 1920, "source_height": 1080,
                        "views": [dict(crop) for crop in _CROPS]},
        "semantics": "Layout-only workbench; original report, native Fn/Ft data, clocks and video mapping remain unchanged.",
    }
    if expert:
        config["expert_video"] = expert
        config["previous_workbench_url"] = f"../../{quote(previous.name, safe='')}/{quote(source.parent.name, safe='')}/report.html"
    if diagnostic:
        config["diagnostic_plots"] = diagnostic
    if example:
        config["diagnostic_example_url"] = "../diagnostics/" + example_name
    page = _render(html, config)
    source_html = source.with_suffix(".html")
    copies = [path for path in source_files if path.is_relative_to(source.parent)
              or path.is_relative_to(site / "assets") or path.is_relative_to(site / "data")
              or path.is_relative_to(site / "media")]
    if any((site / "assets" / name) not in copies for name in ("dashboard.js", "dashboard.css", "bootstrap.js")):
        raise ValueError("Archived base dashboard assets missing")
    for name in ("workbench.js", "workbench.css"):
        if (site / "assets" / name).exists():
            raise ValueError("Source already includes workbench assets")
    rollback = {
        "schema_version": "evaluation-experiment-rollback-v1", "experiment": SCHEMA_VERSION,
        "stable_site": str(stable), "stable_url": f"../{quote(stable.name, safe='')}/index.html",
        "baseline_site": str(site), "baseline_url": f"../{quote(site.name, safe='')}/{quote(source.parent.name, safe='')}/report.html",
        "source_page": str(source_html), "protected_files": protected, "protected_roots": protected_roots,
        "bound_source_aliases": [{"path": row["path"], "resolved_path": row["resolved_path"], "sha256": row["sha256"]}
                                 for row in provenance if "resolved_path" in row],
        "inherited_protected_file_count": len(inherited["protected_files"]),
        "strategy": "New sibling only; return to baseline_url or stable_url. No original file restoration or deletion required.",
        "verification_command": "python -B scripts/analysis/export_evaluation_workbench.py --verify-rollback "
                                + shlex.quote(str(output / "rollback.json")),
    }
    if expert:
        selected_manifest = next(row["path"] for row in expert["provenance"]
                                 if row.get("role") == "expert_video_manifest")
        rollback.update(expert_manifest=str(expert_manifest), previous_workbench=str(previous),
                        previous_workbench_url=f"../{quote(previous.name, safe='')}/{quote(source.parent.name, safe='')}/report.html")
        if str(expert_manifest) != selected_manifest:
            rollback["expert_manifest_local"] = selected_manifest
    if diagnostic or example:
        rollback["diagnostic_sources"] = diagnostic_provenance
    for rows in (protected, implementation):
        _verify(rows)
    _verify_inventory(protected_roots)
    output.mkdir(exist_ok=False)
    for path in copies:
        if path != source_html:
            _copy_checked(path, output / path.relative_to(site), source_rows[str(path)])
    for name in ("workbench.js", "workbench.css"):
        path = _WEB_DIR / name
        row = next(row for row in implementation if row["path"] == str(path.resolve()))
        _copy_checked(path, output / "assets" / name, row)
    if expert:
        _copy_checked(expert_source, output / source.parent.name / "media/expert.mp4", source_rows[str(expert_source)])
    for row, relative in diagnostic_copies:
        _copy_checked(row["source"], output / relative, row["row"])
    _write_json(output / "rollback.json", rollback)
    _write_json(output / "workbench.json", config)
    for rows in (protected, implementation):
        _verify(rows)
    _verify_inventory(protected_roots)
    # Verify copied data, media and base assets before publishing either entry.
    for path in copies:
        if path != source_html and _digest(output / path.relative_to(site))["sha256"] != source_rows[str(path)]["sha256"]:
            raise ValueError("Copied report/data/media differs before publication")
    if expert and _digest(output / source.parent.name / "media/expert.mp4")["sha256"] != expert["sha256"]:
        raise ValueError("Copied expert video differs before publication")
    if diagnostic:
        _verify_diagnostic_copies(diagnostic, output / source.parent.name)
    if example:
        _verify_diagnostic_copies(example, output / "diagnostics")
    _write_json(output / "verification.json", {
        "status": "passed", "protected_file_count": len(protected),
        "inherited_protected_file_count": len(inherited["protected_files"]),
        "source_site_file_count": len(source_files), "original_and_stable_h50_unchanged": True,
        "embedded_payloads_unchanged": True, "evaluation_metrics_recomputed": False,
        "fn_ft_recomputed": False, "new_simulation_trials": 0, "implementation": implementation,
        "expert_video_attached": expert is not None, "expert_clock": "independent_video_seconds" if expert else None,
        "previous_workbench_unchanged": True if previous else None,
        "diagnostic_png_count": len(diagnostic_copies), "diagnostic_plots_redrawn": False,
    })
    if example:
        with (output / "diagnostics" / example_name).open("x", encoding="utf-8") as stream:
            stream.write(_diagnostic_page(example))
    destination = output / source.parent.name
    with (destination / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    link = quote(source.parent.name, safe="") + "/report.html"
    title = "观察台 · 独立专家回放" if expert else "H50 · 触觉工作台"
    heading = "观察台 · 独立专家回放" if expert else "H50 触觉工作台 · 独立布局实验"
    description = ("模型指标和时间线保持不变；新增独立脚本示范视频，不按时间与模型对齐。"
                   if expert else "原始 Fn/Ft、报告、视频与时钟保持不变；本版只重排界面。")
    example_link = (f'<p><a href="diagnostics/{example_name}">灯泡任务 · 三类全程诊断原图 →</a></p>' if example else "")
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                     f'<title>{title}</title><main style="max-width:800px;margin:4rem auto;padding:1rem;font:16px/1.8 system-ui">'
                     f'<h1>{heading}</h1><p>{description}</p>'
                     f'<p><a href="{escape(link, quote=True)}">打开工作台 →</a></p>'
                     f'{example_link}'
                     f'<p><a href="{escape(rollback["baseline_url"], quote=True)}">回到原五指载荷报告</a> · '
                     f'<a href="{escape(rollback["stable_url"], quote=True)}">回到 H50 稳定版</a></p></main></html>')
    return config
