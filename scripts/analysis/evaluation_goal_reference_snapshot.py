#!/usr/bin/env python3
"""Add verified mechanical reference lines to a new, self-contained H50 site.

Only presentation metadata and explicitly selected UI assets change. Archived samples,
native outcomes, stages, videos and the complete parent report remain intact.
This is not a rollout, re-export of raw evaluations, or bypass of their guards.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote

from evaluation_v1.protocol import content_hash
from evaluation_workbench.export import _digest, _document, _embedded, _files, _verify
from evaluation_workbench.goal_references import build_goal_references
from evaluation_workbench.whiteboard_contact_area import (
    SCHEMA_VERSION as AREA_SCHEMA, build_whiteboard_contact_area,
)
from evaluation_workbench.suite import _copy, _page, _portal


SCHEMA = "evaluation-goal-reference-snapshot-v1"
BOARD_TASKS = {"whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}
AREA_KEYS = {"effective_pad_contact_area_ratio", "contact_time_weighted_mean_effective_area_ratio",
             "maximum_ink_remaining"}
PRESENTATION_FILES = {"index.html", "batch.json", "assets/suite.js", "assets/suite.css",
                      "presentation_extension.json", "verification.json", "parent_verification.json",
                      "artifact_manifest.json"}
DETAIL_NAVIGATION_FILES = {"assets/workbench.js", "assets/workbench.css", "trajectory_index.json"}
NAVIGATION_SCHEMA = "evaluation-task-navigation-v1"
PLATFORM_TITLE = "KaiTactileSim 官方评估平台"
PLATFORM_ENGLISH = "KaiTactileSim Evaluation Platform"
PLATFORM_SUBTITLE = "机器人操作任务评估 · 动作回放 · 触觉分析"
TARGET_TASKS = {"bulb-screw", "bulb-unscrew", "bulb-screw-inplace", "install-ram", "poker-draw",
                "usb-insert", "usb-unplug"} | BOARD_TASKS
ANALYSIS = Path(__file__).resolve().parent


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def _trial_directory(root, row):
    value = row.get("report_path")
    if not isinstance(value, str):
        raise ValueError("Each trial requires its explicit relative report path")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2
            or relative.name != "report.html" or "\\" in value):
        raise ValueError("Invalid episode-local report path")
    path = root / relative.parent
    if path.is_symlink() or path.resolve(strict=True).parent != root:
        raise ValueError("Trial directory escapes the source snapshot")
    return path


def _same_native_evidence(before, after):
    previous, current = deepcopy(before), deepcopy(after)
    previous.pop("goal_reference_extension", None)
    current.pop("goal_reference_extension", None)
    previous.pop("whiteboard_contact_area_extension", None)
    current.pop("whiteboard_contact_area_extension", None)
    previous.get("visualization", {}).pop("sha256", None)
    current.get("visualization", {}).pop("sha256", None)
    if previous != current:
        raise ValueError("Reference annotations changed native report evidence")


def _strip_references(visual):
    visual = deepcopy(visual)
    for series in visual.get("task_series", []):
        series.pop("goal_references", None)
    return visual


def _references_for_series(references, series):
    """Match display units without changing an archived series' native values."""
    angle_keys = {"clockwise_turns": "clockwise_angle_deg",
                  "turns_removed": "angle_removed_deg",
                  "rotation_from_initial_turns": "rotation_from_initial_deg"}
    result = []
    for reference in references:
        source = (reference.get("key"), reference.get("unit"))
        target = (series.get("key"), series.get("unit"))
        factor = 1.0 if source == target else None
        for turns, degrees in angle_keys.items():
            if source == (degrees, "deg") and target == (turns, "turn"):
                factor = 1 / 360.0
            elif source == (turns, "turn") and target == (degrees, "deg"):
                factor = 360.0
        if factor is None:
            continue
        converted = deepcopy(reference)
        if factor != 1:
            converted.update(key=target[0], unit=target[1])
            for key in ("target", "lower", "upper"):
                if key in converted:
                    converted[key] *= factor
        result.append(converted)
    return result


def _strip_display_extensions(visual):
    visual = _strip_references(visual)
    visual["task_series"] = [series for series in visual.get("task_series", [])
                             if not (series.get("presentation_extension") == AREA_SCHEMA
                                     and series.get("key") in AREA_KEYS)]
    return visual


def _same_archived_batch(before, after):
    previous, current = deepcopy(before), deepcopy(after)
    previous.pop("presentation_extension", None)
    current.pop("presentation_extension", None)
    if previous != current:
        raise ValueError("Portal presentation changed archived batch evidence")


def _task_navigation_index(trials):
    """Project only archived public trial identities, never audit/provenance data.

    The native task result is deliberately independent of the importer outcome
    and of staged completion (which can be null for pickup-only whiteboard).
    No stages are inferred or recomputed for this lightweight navigation list.
    """
    rows, trial_ids, report_paths = [], set(), set()
    for trial in trials:
        if type(trial.get("prediction_horizon")) is not int or trial["prediction_horizon"] != 50:
            raise ValueError("Navigation accepts only explicit H50 episodes")
        row = {}
        for key in ("task", "trial_id", "model", "checkpoint"):
            value = trial.get(key)
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError("Navigation requires a bounded nonempty " + key)
            row[key] = value
        path = trial.get("report_path")
        if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/report\.html", path):
            raise ValueError("Navigation requires a safe episode-local report path")
        if row["trial_id"] in trial_ids or path in report_paths:
            raise ValueError("Duplicate navigation trial identity or report path")
        trial_ids.add(row["trial_id"])
        report_paths.add(path)
        seed = trial.get("seed")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Navigation seed must be a nonnegative integer or null")
        staged = trial.get("staged")
        native = staged.get("native_result") if isinstance(staged, dict) else None
        native_success = native.get("task_success") if isinstance(native, dict) else None
        if native_success is not None and type(native_success) is not bool:
            raise ValueError("Navigation native task success must be boolean or null")
        row.update(seed=seed, prediction_horizon=50, report_path=path, native_success=native_success)
        rows.append(row)

    def ordered(row):
        checkpoint = row["checkpoint"]
        checkpoint_key = (0, int(checkpoint)) if checkpoint.isascii() and checkpoint.isdigit() else (1, checkpoint.casefold())
        return (row["task"].casefold(), row["model"].casefold(), checkpoint_key,
                row["seed"] is None, row["seed"] if row["seed"] is not None else 0,
                row["trial_id"], row["report_path"])

    return {"schema_version": NAVIGATION_SCHEMA, "prediction_horizon": 50,
            "trials": sorted(rows, key=ordered)}


def _attach_area(visual, config, area):
    """Expose recorded fields without modifying any pre-existing native series."""
    if area.get("status") != "verified" and area.get("series"):
        raise ValueError("Unverified whiteboard sources must not create display curves")
    existing = {row.get("key") for row in visual.get("task_series", [])}
    for series in area.get("series", []):
        if (series.get("presentation_extension") != AREA_SCHEMA
                or series.get("key") not in AREA_KEYS or series.get("key") in existing):
            raise ValueError("Invalid or colliding whiteboard display extension")
        existing.add(series["key"])
        visual.setdefault("task_series", []).append(deepcopy(series))
    summary = config.get("task_summary", [])
    if not isinstance(summary, list):
        raise ValueError("Whiteboard task summaries must be an explicit list")
    config["task_summary"] = [row for row in summary if row.get("presentation_extension") != AREA_SCHEMA]
    for row in area.get("summary", []):
        if row.get("presentation_extension") != AREA_SCHEMA:
            raise ValueError("Whiteboard display summaries require their extension identity")
        config["task_summary"].append(deepcopy(row))
    presentation = config.setdefault("task_presentation", {})
    primary = list(presentation.get("primary_task_keys", []))
    for series in area.get("series", []):
        if series["key"] not in primary:
            primary.append(series["key"])
    presentation["primary_task_keys"] = primary
    config["whiteboard_contact_area"] = {key: deepcopy(value) for key, value in area.items() if key != "series"}


def export_snapshot(source_report, output_dir, *, verified_usb_source_copy=None, presentation_only=False,
                    detail_navigation=False, visual_refresh=False):
    if visual_refresh and not (presentation_only and detail_navigation):
        raise ValueError("Visual refresh requires presentation-only and detail-navigation modes")
    if detail_navigation and not presentation_only:
        raise ValueError("Detail navigation requires presentation-only mode")
    source = Path(source_report).resolve(strict=True)
    requested = Path(output_dir).absolute()
    if (requested.exists() or requested.is_symlink()
            or requested.parent.resolve(strict=True) != requested.parent):
        raise FileExistsError("Choose a fresh canonical report directory")
    if requested.is_relative_to(source) or source.is_relative_to(requested):
        raise ValueError("The new snapshot must be separate from its preserved parent")
    batch = _document(source / "batch.json")
    if batch.get("evaluation_profile") != "recorded-staged" or batch.get("prediction_horizon") != 50:
        raise ValueError("Only an explicit recorded-staged H50 parent is supported")
    trials = batch.get("trials")
    if not isinstance(trials, list) or not trials or len(trials) > 1000:
        raise ValueError("A bounded, nonempty trial inventory is required")
    for trial in trials:
        if trial.get("prediction_horizon") != 50:
            raise ValueError("Never admit H30 or unknown-horizon episodes")
        _trial_directory(source, trial)
    navigation = _task_navigation_index(trials) if detail_navigation else None

    # Verify the preserved report, not unavailable original A/B directories.
    # Missing retired sources must never be presented as a full unified verify.
    manifest = _document(source / "artifact_manifest.json")
    original_paths = _files(source)
    declared = set()
    for row in manifest["files"]:
        path = Path(row["path"])
        if not path.is_absolute() or not path.resolve(strict=True).is_relative_to(source):
            raise ValueError("Parent artifact manifest escapes its report directory")
        if path in declared:
            raise ValueError("Duplicate parent artifact identity")
        declared.add(path)
    if declared != set(original_paths) - {source / "artifact_manifest.json"}:
        raise ValueError("Parent artifact inventory differs from its manifest")
    _verify(manifest["files"])
    parent_rows = [_digest(path) for path in original_paths]
    source_manifest_sha = _digest(source / "artifact_manifest.json")
    statuses = {"verified": 0, "unavailable": 0, "not_applicable": 0}
    area_statuses = {"verified": 0, "unavailable": 0, "not_applicable": 0}
    annotations = []
    if presentation_only:
        # This mode intentionally does not call goal/geometry importers or read
        # retired raw sources: it preserves the already-verified archived view.
        previous_extension = batch.get("goal_reference_extension", {})
        statuses = deepcopy(previous_extension.get("status_counts", {
            "verified": 0, "unavailable": 0, "not_applicable": len(trials)}))
        area_statuses = deepcopy(previous_extension.get("whiteboard_contact_area_status_counts", {
            "verified": 0, "unavailable": 0, "not_applicable": len(trials)}))
        for counts in (statuses, area_statuses):
            if (not isinstance(counts, dict) or set(counts) != {"verified", "unavailable", "not_applicable"}
                    or any(type(value) is not int or value < 0 for value in counts.values())
                    or sum(counts.values()) != len(trials)):
                raise ValueError("Invalid archived reference-status counts")

    with tempfile.TemporaryDirectory(prefix=f".{requested.name}-", dir=requested.parent) as temp:
        stage = Path(temp) / "site"
        stage.mkdir()
        copied = []
        for row in parent_rows:
            _copy({"source": row["path"], **row}, stage / Path(row["path"]).relative_to(source), copied)

        for trial in (() if presentation_only else trials):
            if trial.get("task") not in TARGET_TASKS:
                statuses["not_applicable"] += 1
                area_statuses["not_applicable"] += 1
                continue
            original = _trial_directory(source, trial)
            target = stage / original.relative_to(source)
            report = _document(original / "report.json")
            visual = _document(original / "visualization.json")
            loads = _document(original / "tactile_loads.json")
            config = _document(original / "workbench.json")
            html = _document(original / "report.html", html=True)
            _, embedded = _embedded(html, "kai-evaluation-data")
            _, embedded_config = _embedded(html, "kai-workbench-config")
            _, embedded_loads = _embedded(html, "kai-tactile-load-data")
            if (embedded.get("report") != report or embedded.get("visualization") != visual
                    or embedded_config != config or embedded_loads != loads):
                raise ValueError("Parent embedded data differs from the archived JSON")
            if report.get("trial", {}).get("task") != trial["task"]:
                raise ValueError("Trial task identity mismatch")
            source_options = ({"source_root": Path(verified_usb_source_copy)}
                              if trial["task"] == "usb-insert" and verified_usb_source_copy is not None else {})
            goals = build_goal_references(trial["task"], report, **source_options)
            if goals["status"] not in statuses:
                raise ValueError("Unknown reference verification status")
            if goals["status"] != "verified" and goals["references"]:
                raise ValueError("Unverified goals must not contain reference values")
            statuses[goals["status"]] += 1
            # A previous reference snapshot can itself be the preserved parent.
            # Rebind every line to verified source evidence; never retain a
            # stale annotation if its source is now unavailable.
            new_visual, new_report, new_config = _strip_display_extensions(visual), deepcopy(report), deepcopy(config)
            if trial["task"] in BOARD_TASKS:
                # This is a projection of already-recorded geometry/integrator
                # fields, never a physics replay or average of video frames.
                area = build_whiteboard_contact_area(trial["task"], report,
                                                      time_origin_s=visual.get("time_origin_s")
                                                      if visual.get("clock_aligned") is True else None)
                if area.get("status") not in area_statuses:
                    raise ValueError("Unknown whiteboard display verification status")
                _attach_area(new_visual, new_config, area)
                area_statuses[area["status"]] += 1
                new_report["whiteboard_contact_area_extension"] = {
                    "schema_version": area["schema_version"], "status": area["status"],
                    "path": "whiteboard_contact_area.json", "sha256": content_hash(area),
                    "native_outcomes_rescored": False,
                }
                _json(target / "whiteboard_contact_area.json", area)
            else:
                area_statuses["not_applicable"] += 1
            for series in new_visual.get("task_series", []):
                references = _references_for_series(goals["references"], series)
                if references:
                    series["goal_references"] = deepcopy(references)
            if _strip_display_extensions(new_visual) != _strip_display_extensions(visual):
                raise ValueError("Reference annotations changed native samples")
            new_report["visualization"]["sha256"] = content_hash(new_visual)
            new_report["goal_reference_extension"] = {
                "schema_version": goals["schema_version"], "status": goals["status"],
                "path": "goal_references.json", "sha256": content_hash(goals),
                "native_outcomes_rescored": False,
            }
            _same_native_evidence(report, new_report)
            new_config["goal_references"] = goals
            _json(target / "report.json", new_report)
            _json(target / "visualization.json", new_visual)
            _json(target / "workbench.json", new_config)
            _json(target / "goal_references.json", goals)
            (target / "report.html").write_text(_page(new_report, new_visual, loads, new_config), encoding="utf-8")
            annotations.append({"report_path": trial["report_path"], "task": trial["task"],
                                "status": goals["status"], "references": len(goals["references"])})

        # A portal-only update keeps every episode and reference byte-identical.
        # Detail navigation may additionally replace only the workbench UI assets
        # and create the explicitly limited public identity index.
        if presentation_only:
            for name in ("suite.js", "suite.css"):
                (stage / "assets" / name).write_bytes((ANALYSIS / "evaluation_workbench/web" / name).read_bytes())
            if detail_navigation:
                for name in ("workbench.js", "workbench.css"):
                    (stage / "assets" / name).write_bytes((ANALYSIS / "evaluation_workbench/web" / name).read_bytes())
                _json(stage / "trajectory_index.json", navigation)
        else:
            for name in ("dashboard.js", "dashboard.css"):
                (stage / "assets" / name).write_bytes((ANALYSIS / "evaluation_v1/web" / name).read_bytes())
            (stage / "assets/workbench.js").write_bytes((ANALYSIS / "evaluation_workbench/web/workbench.js").read_bytes())
            # New exports must use the portal implementation matching _portal.
            for name in ("suite.js", "suite.css"):
                (stage / "assets" / name).write_bytes((ANALYSIS / "evaluation_workbench/web" / name).read_bytes())
        extension = {
            "schema_version": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_report": str(source), "parent_artifact_manifest": source_manifest_sha,
            "trial_count": len(trials), "status_counts": statuses, "annotations": annotations,
            "whiteboard_contact_area_status_counts": area_statuses,
            "verified_usb_source_copy": str(Path(verified_usb_source_copy).resolve()) if verified_usb_source_copy else None,
            "native_samples_changed": False, "native_outcomes_rescored": False,
            "previously_recorded_whiteboard_fields_exposed": True,
            "media_independently_copied": True,
            "semantics": "Verified mechanical references, not task success or elapsed-time endpoints.",
        }
        new_batch = deepcopy(batch)
        if presentation_only:
            extension = {
                "schema_version": "evaluation-suite-presentation-v1",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "source_report": str(source), "parent_artifact_manifest": source_manifest_sha,
                "show_run_mode": False, "show_coverage_and_sources": False,
                "internal_audit_metadata_preserved": True, "trial_files_unchanged": True,
                "native_outcomes_rescored": False, "media_independently_copied": True,
            }
            if detail_navigation:
                extension.update(detail_navigation=True, show_task_stages=True,
                                 navigation_schema=NAVIGATION_SCHEMA,
                                 navigation_index="trajectory_index.json")
            if visual_refresh:
                extension.update(visual_refresh=True, layout_version="evaluation-studio-v1",
                                 model_first=True, expert_clock_independent=True,
                                 overview_views=["cards", "list"],
                                 branding={"project": "KaiTactileSim", "title": PLATFORM_TITLE,
                                           "english_name": PLATFORM_ENGLISH,
                                           "scope": "project_owned_interface_identity_only",
                                           "evaluation_status_changed": False})
            new_batch["presentation_extension"] = extension
            _same_archived_batch(batch, new_batch)
        else:
            new_batch["goal_reference_extension"] = extension
        _json(stage / "batch.json", new_batch)
        portal = _portal(new_batch)
        if visual_refresh:
            portal = re.sub(r"<title>[^<]*</title>", f"<title>{PLATFORM_TITLE} | H50</title>", portal, count=1)
            portal = re.sub(r"<h1>[^<]*</h1>", f"<h1>{PLATFORM_TITLE}</h1>", portal, count=1)
            portal = re.sub(r'<p class="ks-eyebrow">[^<]*</p>', f'<p class="ks-eyebrow">{PLATFORM_ENGLISH}</p>', portal, count=1)
            portal = portal.replace('从任务记录、模型动作与专家示范，到五指触觉、关节角和右腕状态。', PLATFORM_SUBTITLE, 1)
            portal = portal.replace('原 H50 稳定版（保留）</a>', '历史 H50 版本</a>', 1)
            portal = portal.replace('</main>', f'<p class="ks-brand-footer">{PLATFORM_TITLE} · 评估规则与结果以各批次归档为准</p></main>', 1)
        else:
            portal = portal.replace("</h1>", " · 目标参考线</h1>", 1)
        if not presentation_only:
            portal = portal.replace('</header>', '</header><aside class="ks-import-note">'
                '目标参考版：灯泡、RAM、抽卡、USB 与三类白板增加来源核验的单项参考线；'
                '白板补充接触期间几何贴合比与最大墨迹残留。贴合比不是真实形变接触面积，平均清洁 100% 是理想参考。'
                '越线不等于完整任务成功。原始轨迹、原生结果与阶段统计均未重算。'
                f'<a href="../{escape(quote(source.name), quote=True)}/index.html">打开保留的原 H50 版</a></aside>', 1)
        (stage / "index.html").write_text(portal, encoding="utf-8")
        _json(stage / ("presentation_extension.json" if presentation_only else "goal_reference_extension.json"), extension)
        (stage / "parent_verification.json").write_bytes((source / "verification.json").read_bytes())
        _json(stage / "verification.json", {
            "schema_version": SCHEMA, "parent_report": str(source),
            "preserved_parent_files": parent_rows, "native_data_preserved": True,
            "source_projects_reverified": False, "reference_status_counts": statuses,
            "presentation_only": presentation_only, "trial_files_unchanged": presentation_only,
            "detail_navigation": detail_navigation,
            "visual_refresh": visual_refresh,
            "verification_scope": "Archived parent artifacts plus presentation-only annotations; not unified source verification.",
        })
        artifact_rows = []
        allowed_presentation_files = PRESENTATION_FILES | (DETAIL_NAVIGATION_FILES if detail_navigation else set())
        parent_by_relative = {str(Path(row["path"]).relative_to(source)): row for row in parent_rows}
        for path in _files(stage):
            if path.name == "artifact_manifest.json" and path.parent == stage:
                continue
            row = _digest(path)
            if presentation_only:
                relative = str(path.relative_to(stage))
                previous = parent_by_relative.get(relative)
                if relative not in allowed_presentation_files and (
                        previous is None or any(row[key] != previous[key] for key in ("sha256", "size"))):
                    raise ValueError("Portal-only update changed a protected archived artifact: " + relative)
            row["path"] = str(requested / path.relative_to(stage))
            artifact_rows.append(row)
        _json(stage / "artifact_manifest.json", {"schema_version": SCHEMA, "files": artifact_rows})
        if _files(source) != original_paths:
            raise ValueError("Preserved parent inventory changed during export")
        _verify(parent_rows)
        if requested.exists() or requested.is_symlink():
            raise FileExistsError("Output appeared while building; it will not be overwritten")
        stage.rename(requested)
    return {"output": str(requested), "trial_count": len(trials), "status_counts": statuses,
            "whiteboard_contact_area_status_counts": area_statuses,
            "presentation_only": presentation_only,
            "detail_navigation": detail_navigation,
            "visual_refresh": visual_refresh,
            "parent_unchanged": True, "native_outcomes_rescored": False}


def verify_snapshot(output_dir):
    root = Path(output_dir).resolve(strict=True)
    manifest = _document(root / "artifact_manifest.json")
    proof = _document(root / "verification.json")
    if manifest.get("schema_version") != SCHEMA or proof.get("schema_version") != SCHEMA:
        raise ValueError("Not a goal-reference snapshot")
    expected = set()
    for row in manifest["files"]:
        path = Path(row["path"])
        if path.is_symlink() or not path.is_relative_to(root):
            raise ValueError("Snapshot artifact escapes the report directory")
        expected.add(path)
    actual = set(_files(root)) - {root / "artifact_manifest.json"}
    if actual != expected:
        raise ValueError("Snapshot inventory differs from its manifest")
    _verify(manifest["files"])
    parent = Path(proof["parent_report"]).resolve(strict=True)
    if set(_files(parent)) != {Path(row["path"]) for row in proof["preserved_parent_files"]}:
        raise ValueError("Preserved parent inventory changed")
    _verify(proof["preserved_parent_files"])
    return {"verified_snapshot_files": len(expected),
            "preserved_parent_files": len(proof["preserved_parent_files"]), "parent_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verified-usb-source-copy", type=Path,
                        help="Explicit byte-identical USB insertion source copy; each required file must match its frozen inventory")
    parser.add_argument("--presentation-only", action="store_true",
                        help="Update only the portal UI; preserve all archived trial files and references without rereading raw sources")
    parser.add_argument("--detail-navigation", action="store_true",
                        help="With --presentation-only, add archived stage display and same-task navigation assets/index")
    parser.add_argument("--visual-refresh", action="store_true",
                        help="With --presentation-only --detail-navigation, mark the model-first studio visual edition; no rescoring")
    args = parser.parse_args(argv)
    if args.visual_refresh and not (args.presentation_only and args.detail_navigation):
        parser.error("--visual-refresh requires --presentation-only --detail-navigation")
    if args.detail_navigation and not args.presentation_only:
        parser.error("--detail-navigation requires --presentation-only")
    if args.verify:
        result = verify_snapshot(args.verify)
    else:
        if args.source_report is None or args.output_dir is None:
            parser.error("--source-report and --output-dir are required")
        result = export_snapshot(args.source_report, args.output_dir,
                                 verified_usb_source_copy=args.verified_usb_source_copy,
                                 presentation_only=args.presentation_only,
                                 detail_navigation=args.detail_navigation,
                                 visual_refresh=args.visual_refresh)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
