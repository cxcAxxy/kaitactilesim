"""Transactional, model-only PNG completion for an existing additive H50 suite.

Only missing USB-insert galleries are completed. Numerical payloads, source
records, videos, previous sites, and the source manifest are never rewritten.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from evaluation_progress_experiment.adapters import _path
from evaluation_v1.protocol import content_hash
from .export import _digest, _document, _rows, _verify, verify_rollback
from .model_plots import _PLOTS
from .suite import SCHEMA, WEB, _page, _portal, implementation


_ASSETS = ("workbench.js", "workbench.css", "suite.js", "suite.css")
_PAYLOADS = ("report.json", "visualization.json", "tactile_loads.json")
_CASE_FILES = (*_PAYLOADS, "workbench.json", "report.html")


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _bytes_row(path, raw):
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _generate_one(arguments):
    from .model_plots import generate_model_plots
    return generate_model_plots(*arguments)


def _generate_all(jobs, workers):
    # The context waits for running work even on failure, before caller cleanup.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_generate_one, job) for job in jobs]
        try:
            results = []
            for index, future in enumerate(futures):
                results.append(future.result())
                print(json.dumps({"model_plot_trials_completed": index + 1, "total": len(jobs)}), flush=True)
            return results
        finally:
            for future in futures:
                future.cancel()


def _bound_artifacts(output, manifest):
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) > 100_000:
        raise ValueError("Expected a bounded suite artifact manifest")
    bound = {}
    for row in files:
        _rows([row])
        path = Path(row["path"])
        if not path.is_relative_to(output) or path.resolve(strict=True) != path or path.is_symlink():
            raise ValueError("Artifact target is outside the current non-symlink suite")
        if str(path) in bound:
            raise ValueError("Duplicate suite artifact target")
        bound[str(path)] = row
    return bound


def _report_provenance_is_preserved(report_rows, source_rows):
    """Permit only independently audited crop witnesses beyond original rows."""
    original = {row["path"]: row for row in _rows(report_rows)}
    current = {row["path"]: row for row in _rows(source_rows)}
    _rows(report_rows + source_rows)  # Reject duplicate-path SHA/size conflicts.
    for path, row in original.items():
        bound = current.get(path)
        if bound is None or (bound["sha256"].lower(), bound.get("size")) != (row["sha256"].lower(), row.get("size")):
            return False
    return all(path in original or str(row.get("role", "")).startswith("model_crop_")
               for path, row in current.items())


def _model_view_descriptor(value, video_sha):
    if not isinstance(value, dict) or value.get("sha256") != video_sha:
        raise ValueError("Model crop descriptor does not bind the displayed video SHA")
    views = value.get("views")
    if value.get("status") == "verified" and isinstance(views, list) and any(
            isinstance(view, dict) and view.get("id") == "wrist" for view in views):
        width, height = value.get("source_width"), value.get("source_height")
        if any(type(number) is not int or number <= 0 for number in (width, height)) or not 1 <= len(views) <= 8:
            raise ValueError("Invalid verified model crop frame geometry")
        identities = set()
        for view in views:
            if not isinstance(view, dict) or not isinstance(view.get("id"), str) or view["id"] in identities:
                raise ValueError("Invalid or duplicate model view identity")
            identities.add(view["id"])
            numbers = [view.get(key) for key in ("x", "y", "width", "height")]
            if any(type(number) not in (int, float) or not math.isfinite(number) for number in numbers):
                raise ValueError("Invalid model crop rectangle")
            x, y, w, h = numbers
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError("Model crop rectangle escapes its video frame")
        if not value.get("provenance"):
            raise ValueError("Verified wrist crop has no independent source witnesses")
        return value
    reason = value.get("reason") or "本条录制缺少已核验的右腕裁剪区域；保留完整拼图，不推测腕部位置。"
    if value.get("status") == "verified":
        reason += " 未提供已核验的右腕区域，本次保留完整拼图。"
    return {**value, "status": "full_frame_only", "views": [], "reason": reason}


def refresh_suite_model_views(output):
    """Refresh model crop configs/pages only, retaining all recorded payloads.

    New crop witnesses extend the per-case source manifest; original report
    provenance and metrics remain byte-identical. Copied videos are hashed once
    and stat-checked again around the commit, never copied or re-encoded.
    """
    from .experts import model_views_for_case
    output = _path(Path(output).absolute())
    rollback = _document(output / "rollback.json")
    batch = _document(output / "batch.json")
    if (rollback.get("experiment") != SCHEMA or batch.get("schema_version") != SCHEMA
            or type(batch.get("prediction_horizon")) is not int or batch["prediction_horizon"] != 50):
        raise ValueError("Model-view refresh only accepts the new H50 multi-task suite")
    previous = _path(rollback.get("previous_workbench"))
    if previous == output or previous.parent != output.parent:
        raise ValueError("Suite must remain a separate sibling of its preserved previous site")
    if any(output.is_relative_to(Path(root["path"]).resolve(strict=True)) for root in rollback.get("protected_roots", [])):
        raise ValueError("Cannot refresh model views inside a protected old site")
    verify_rollback(output / "rollback.json")
    artifact_path = output / "artifact_manifest.json"
    manifest, manifest_snapshot = _document(artifact_path), _digest(artifact_path)
    bound = _bound_artifacts(output, manifest)
    root_files = [output / name for name in ("rollback.json", "batch.json", "source_manifest.json", "verification.json", "index.html")]
    if any(str(path) not in bound for path in root_files):
        raise ValueError("Model-view refresh metadata lacks artifact bindings")
    checked = [bound[str(path)] for path in root_files] + [manifest_snapshot]
    _verify(checked)
    verification = _document(output / "verification.json")
    if verification.get("status") != "passed":
        raise ValueError("Suite has no passing export verification")
    source_manifest = _document(output / "source_manifest.json")
    source_entries, entries = source_manifest.get("trials"), batch.get("trials")
    if (not isinstance(source_entries, list) or not isinstance(entries, list)
            or not 0 < len(entries) <= 5000 or len(entries) != batch.get("trial_count")):
        raise ValueError("Invalid bounded suite/source trial lists")
    sources = {row["trial_id"]: row for row in source_entries}
    if len(sources) != len(source_entries) or len({row["trial_id"] for row in entries}) != len(entries):
        raise ValueError("Duplicate suite/source trial identities")
    protected = {row.get("resolved_path", row["path"]) for row in rollback["protected_files"]}
    implementation_rows = implementation()
    updated, new_rows, source_checks, video_checks = [], {}, {}, {}
    verified_count, fallback = 0, []
    lock = output / ".complete-model-plots.lock"
    with lock.open("xb"):
        pass

    def signature(path):
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    def verify_many(rows):
        for start in range(0, len(rows), 1000):
            _verify(rows[start:start + 1000])

    def verify_videos():
        for path, expected in video_checks.items():
            if path.is_symlink() or path.resolve(strict=True) != path or signature(path) != expected:
                raise ValueError("Copied model video changed during crop refresh")

    def remember_sources(rows):
        for row in rows:
            previous = source_checks.get(row["path"])
            if previous and (previous["sha256"], previous.get("size")) != (row["sha256"], row.get("size")):
                raise ValueError("Model crop source witnesses conflict across cases")
            source_checks[row["path"]] = row

    try:
        _verify(checked)
        with tempfile.TemporaryDirectory(prefix=".suite-model-views-", dir=output) as temporary:
            stage = Path(temporary)

            def stage_replace(path, raw):
                if str(path) in protected or (path != artifact_path and str(path) not in bound):
                    raise ValueError("Model-view refresh target is protected or unbound")
                relative = path.relative_to(output)
                for name, data in (("next", raw), ("before", path.read_bytes())):
                    destination = stage / name / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                updated.append(path)
                new_rows[str(path)] = _bytes_row(path, raw)

            for entry in entries:
                relative = entry.get("report_path")
                if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/report\.html", relative):
                    raise ValueError("Unsupported model-view case report path")
                directory = _path(output / Path(relative).parent)
                paths = [directory / name for name in _CASE_FILES]
                if any(str(path) not in bound for path in paths):
                    raise ValueError("Model-view case payload/config/page is not manifest-bound")
                case_rows = [bound[str(path)] for path in paths]
                _verify(case_rows)
                checked.extend(case_rows)
                report, visual, loads, config = [_document(directory / name) for name in (*_PAYLOADS, "workbench.json")]
                identity = report.get("trial", {})
                source = _path(entry["source"])
                source_entry = sources.get(entry["trial_id"], {})
                if (identity.get("trial_id") != entry["trial_id"] or identity.get("task") != entry["task"]
                        or identity.get("source") != str(source) or identity.get("prediction_horizon") != 50
                        or config.get("trial_id") != entry["trial_id"] or config.get("task") != entry["task"]
                        or config.get("prediction_horizon") != 50 or loads.get("trial_id") != entry["trial_id"]
                        or source_entry.get("source") != str(source)
                        or not _report_provenance_is_preserved(report.get("provenance"), source_entry.get("provenance"))
                        or report.get("visualization", {}).get("sha256") != content_hash(visual)):
                    raise ValueError("Model-view case task/source/H50/payload binding disagrees")
                provenance = {row["path"]: row for row in _rows(report["provenance"])}
                summary_path = source / "summary.json"
                summary_row = provenance.get(str(summary_path))
                if summary_row is None:
                    raise ValueError("Model crop summary is not bound to the original report")
                _verify([summary_row])
                remember_sources([summary_row])
                summary = _document(summary_path)
                if summary.get("task") != identity["task"] or summary.get("seed") != identity.get("seed"):
                    raise ValueError("Model crop source summary identifies another trial")
                media = visual.get("media") or {}
                if media.get("status") == "available":
                    if media.get("url") != "media/rollout.mp4":
                        raise ValueError("Model crop video must use the bound suite rollout URL")
                    video = _path(directory / media["url"])
                    video_row = bound.get(str(video))
                    witnesses = [row for row in provenance.values() if row.get("role") == "copied_review_video"]
                    if (video_row is None or len(witnesses) != 1
                            or witnesses[0]["sha256"] != video_row["sha256"] or witnesses[0].get("size") != video_row.get("size")
                            or (media.get("sha256") is not None and media["sha256"] != video_row["sha256"])):
                        raise ValueError("Copied model video differs from original report/visual bindings")
                    witness_path = Path(witnesses[0].get("resolved_path", witnesses[0]["path"]))
                    if witness_path not in {source / "review.mp4", source / "review/review.mp4"}:
                        raise ValueError("Model video witness is outside the selected source trial")
                    before_video = signature(video)
                    _verify([video_row])
                    if signature(video) != before_video:
                        raise ValueError("Copied model video changed while hashing")
                    video_checks[video] = before_video
                    descriptor = _model_view_descriptor(model_views_for_case(summary, video_row["sha256"]), video_row["sha256"])
                else:
                    descriptor = {"status": "full_frame_only", "views": [], "sha256": None,
                                  "reason": "本条没有已绑定的模型回放视频，无法提供已核验腕部裁剪；不替换其他轨迹的视频。"}
                witnesses = descriptor.get("provenance", [])
                if witnesses:
                    _verify(witnesses)
                    if any(not str(row.get("role", "")).startswith("model_crop_") for row in witnesses):
                        raise ValueError("Unexpected non-crop provider witness")
                    _rows(source_entry["provenance"] + witnesses)
                    old_paths = {row["path"] for row in source_entry["provenance"]}
                    source_entry["provenance"].extend(row for row in witnesses if row["path"] not in old_paths)
                    remember_sources(witnesses)
                if descriptor["status"] == "verified":
                    verified_count += 1
                else:
                    fallback.append({"trial_id": entry["trial_id"], "task": entry["task"], "reason": descriptor["reason"]})
                if config.get("media_views") != descriptor:
                    config["media_views"] = descriptor
                    stage_replace(directory / "workbench.json", _encoded(config))
                    stage_replace(directory / "report.html", _page(report, visual, loads, config).encode("utf-8"))
            changed_cases = len(updated) // 2
            verification.update(implementation=implementation_rows, model_wrist_view_trial_count=verified_count,
                                model_view_refresh={"completed_utc": datetime.now(timezone.utc).isoformat(),
                                    "trial_count": len(entries), "updated_case_count": changed_cases,
                                    "verified_wrist_count": verified_count, "full_frame_only_count": len(fallback),
                                    "fallbacks": fallback, "copied_videos_hashed": len(video_checks),
                                    "video_pixels_unchanged": True, "metrics_recomputed": False,
                                    "report_visualization_tactile_unchanged": True, "reference_time_alignment": "none"})
            stage_replace(output / "source_manifest.json", _encoded(source_manifest))
            stage_replace(output / "verification.json", _encoded(verification))
            revised = {**bound, **new_rows}
            stage_replace(artifact_path, _encoded({**manifest, "files": list(revised.values())}))
            verify_many(checked)
            verify_many(list(source_checks.values()))
            _verify(implementation_rows)
            verify_videos()
            verify_rollback(output / "rollback.json")
            committed = []
            try:
                for path in updated:
                    os.replace(stage / "next" / path.relative_to(output), path)
                    committed.append(path)
                changed = set(updated)
                verify_many([row for row in checked if Path(row["path"]) not in changed])
                verify_many(list(source_checks.values()))
                verify_many(list(new_rows.values()))
                _verify(implementation_rows)
                verify_videos()
                verify_rollback(output / "rollback.json")
            except BaseException:
                for path in reversed(committed):
                    os.replace(stage / "before" / path.relative_to(output), path)
                raise
        return {"refreshed": True, "output": str(output), "trial_count": len(entries),
                "updated_case_count": changed_cases, "verified_wrist_count": verified_count,
                "full_frame_only_count": len(fallback), "updated": [str(path.relative_to(output)) for path in updated],
                "payloads_unchanged": True, "video_pixels_unchanged": True, "metrics_recomputed": False}
    finally:
        lock.unlink()


def refresh_suite_assets(output):
    """Refresh only four assets and their audit hashes in an additive H50 suite.

    No HTML, case config, numerical payload, source manifest, video or PNG is
    regenerated. Each replaced file has a staged backup for failure rollback.
    """
    output = _path(Path(output).absolute())
    if not output.is_dir():
        raise ValueError("Asset refresh requires an existing H50 suite directory")
    rollback = _document(output / "rollback.json")
    batch = _document(output / "batch.json")
    if (rollback.get("experiment") != SCHEMA or batch.get("schema_version") != SCHEMA
            or type(batch.get("prediction_horizon")) is not int or batch["prediction_horizon"] != 50):
        raise ValueError("Asset refresh only accepts the new H50 multi-task suite")
    previous = _path(rollback.get("previous_workbench"))
    if previous == output or previous.parent != output.parent:
        raise ValueError("Suite must remain a separate sibling of its preserved previous site")
    for root in rollback.get("protected_roots", []):
        if output.is_relative_to(Path(root["path"]).resolve(strict=True)):
            raise ValueError("Cannot refresh a protected old site")
    verify_rollback(output / "rollback.json")
    artifact_path = output / "artifact_manifest.json"
    manifest = _document(artifact_path)
    manifest_snapshot = _digest(artifact_path)
    bound = _bound_artifacts(output, manifest)
    assets = [output / "assets" / name for name in _ASSETS]
    verification_path = output / "verification.json"
    immutable_metadata = [output / name for name in ("rollback.json", "batch.json", "source_manifest.json", "index.html")]
    required = assets + [verification_path] + immutable_metadata
    if any(str(path) not in bound for path in required):
        raise ValueError("Refresh assets and suite metadata must be artifact-manifest bound")
    checked_rows = [bound[str(path)] for path in required] + [manifest_snapshot]
    _verify(checked_rows)
    verification = _document(verification_path)
    if verification.get("status") != "passed":
        raise ValueError("Suite has no passing export verification")
    # Parse the unchanged source binding as an object, without re-reading every
    # multi-gigabyte source; full verification remains available to the caller.
    if not isinstance(_document(output / "source_manifest.json").get("trials"), list):
        raise ValueError("Suite source manifest is not a bound trial list")
    allowlist = assets + [verification_path, artifact_path]
    protected = {row.get("resolved_path", row["path"]) for row in rollback["protected_files"]}
    if any(str(path) in protected for path in allowlist):
        raise ValueError("Asset refresh target is protected by an older site")
    source_implementation = implementation()
    sources = {row["path"]: row for row in source_implementation}
    replacements = {}
    for name, destination in zip(_ASSETS, assets):
        source = _path(WEB / name)
        if str(source) not in sources:
            raise ValueError("Current web asset lacks an implementation binding")
        raw = source.read_bytes()
        if _bytes_row(source, raw)["sha256"] != sources[str(source)]["sha256"]:
            raise ValueError("Current web asset changed during refresh preparation")
        replacements[destination] = raw
    verification["implementation"] = source_implementation
    replacements[verification_path] = _encoded(verification)
    revised = dict(bound)
    for path, raw in replacements.items():
        revised[str(path)] = _bytes_row(path, raw)
    replacements[artifact_path] = _encoded({**manifest, "files": list(revised.values())})
    if set(replacements) != set(allowlist):
        raise ValueError("Asset refresh replacement set differs from its narrow allowlist")
    # Use the same lock as model completion so these two in-place transactions
    # cannot interleave their artifact manifests.
    lock = output / ".complete-model-plots.lock"
    with lock.open("xb"):
        pass
    try:
        _verify(checked_rows)
        before = {path: path.read_bytes() for path in allowlist}
        committed = []
        with tempfile.TemporaryDirectory(prefix=".suite-assets-", dir=output) as temporary:
            stage = Path(temporary)
            for path in allowlist:
                relative = path.relative_to(output)
                for name, raw in (("next", replacements[path]), ("before", before[path])):
                    staged = stage / name / relative
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(raw)
            _verify(checked_rows)
            _verify(source_implementation)
            verify_rollback(output / "rollback.json")
            try:
                for path in allowlist:
                    os.replace(stage / "next" / path.relative_to(output), path)
                    committed.append(path)
                _verify([bound[str(path)] for path in immutable_metadata])
                _verify(source_implementation)
                verify_rollback(output / "rollback.json")
                for path in allowlist:
                    _verify([_bytes_row(path, replacements[path])])
            except BaseException:
                for path in reversed(committed):
                    os.replace(stage / "before" / path.relative_to(output), path)
                raise
        return {"refreshed": True, "output": str(output),
                "updated": [str(path.relative_to(output)) for path in allowlist],
                "payloads_unchanged": True, "html_unchanged": True,
                "source_manifest_unchanged": True, "metrics_recomputed": False}
    finally:
        lock.unlink()


def complete_missing_model_plots(output, workers=4):
    """Complete USB model-only galleries, atomically replacing allowlisted files.

    A failure restores every replaced byte and removes only this call's new PNG
    targets. The caller can run suite.verify_suite afterwards for a full audit.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use 1 to 8 offline CPU workers")
    output = _path(Path(output).absolute())
    if not output.is_dir():
        raise ValueError("Model plot completion requires an existing suite directory")
    rollback = _document(output / "rollback.json")
    batch = _document(output / "batch.json")
    if (rollback.get("experiment") != SCHEMA or batch.get("schema_version") != SCHEMA
            or type(batch.get("prediction_horizon")) is not int or batch["prediction_horizon"] != 50):
        raise ValueError("Only the new H50 multi-task suite may be completed; archived sites are read-only")
    previous = _path(rollback.get("previous_workbench"))
    if previous == output or previous.parent != output.parent:
        raise ValueError("Suite must remain a separate sibling of its preserved previous site")
    for root in rollback.get("protected_roots", []):
        if output.is_relative_to(Path(root["path"]).resolve(strict=True)):
            raise ValueError("Cannot complete plots inside a protected old site")
    verify_rollback(output / "rollback.json")
    artifact_path = output / "artifact_manifest.json"
    original_manifest = _document(artifact_path)
    artifact_snapshot = _digest(artifact_path)
    bound = _bound_artifacts(output, original_manifest)
    protected_paths = {row.get("resolved_path", row["path"]) for row in rollback["protected_files"]}
    root_files = [output / name for name in ("rollback.json", "batch.json", "verification.json", "index.html", "source_manifest.json")]
    root_files.extend(output / "assets" / name for name in _ASSETS)
    if any(str(path) not in bound for path in root_files):
        raise ValueError("Suite metadata/assets are not bound by its artifact manifest")
    _verify([bound[str(path)] for path in root_files])
    verification = _document(output / "verification.json")
    if verification.get("status") != "passed":
        raise ValueError("Suite has no passing export verification")
    source_manifest = _document(output / "source_manifest.json")
    source_cases = {row["trial_id"]: row for row in source_manifest["trials"]}
    entries = batch.get("trials")
    if not isinstance(entries, list) or len(entries) != batch.get("trial_count"):
        raise ValueError("Suite trial count differs from its entries")
    selected = [entry for entry in entries if entry.get("task") == "usb-insert" and entry.get("plots") == "missing"]
    if not selected:
        return {"completed": True, "output": str(output), "generated_count": 0,
                "generated_model_plot_trial_count": sum(row.get("plot_kind") == "model_only" for row in entries),
                "updated": [], "metrics_recomputed": False, "payloads_unchanged": True}

    contexts, jobs, png_targets, new_plot_dirs, checked = [], [], [], [], list(root_files)
    for entry in selected:
        relative = entry.get("report_path")
        if not isinstance(relative, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/report\.html", relative):
            raise ValueError("Unsupported case report path")
        directory = _path(output / Path(relative).parent)
        paths = [directory / name for name in _CASE_FILES]
        if any(str(path) not in bound for path in paths):
            raise ValueError("Case payload/config/page is not bound by the suite manifest")
        _verify([bound[str(path)] for path in paths])
        report, visual, loads, config = [_document(directory / name) for name in (*_PAYLOADS, "workbench.json")]
        source = _path(entry["source"])
        identity = report.get("trial", {})
        recorded_source = source_cases.get(entry["trial_id"], {})
        if (identity.get("trial_id") != entry["trial_id"] or identity.get("task") != "usb-insert"
                or identity.get("source") != str(source) or identity.get("prediction_horizon") != 50
                or config.get("trial_id") != entry["trial_id"] or config.get("task") != "usb-insert"
                or config.get("prediction_horizon") != 50 or loads.get("trial_id") != entry["trial_id"]
                or config.get("diagnostic_plots", {}).get("status") != "missing"
                or recorded_source.get("source") != str(source)
                or not _report_provenance_is_preserved(report.get("provenance"), recorded_source.get("provenance"))
                or report.get("visualization", {}).get("sha256") != content_hash(visual)):
            raise ValueError("Case task/source/H50/payload identity differs from the exported suite")
        if output == source or output.is_relative_to(source):
            raise ValueError("Plot completion cannot write inside original trials")
        plot_dir = directory / "plots"
        if plot_dir.resolve() != plot_dir or plot_dir.is_symlink():
            raise ValueError("Model plot directory cannot follow a symlink")
        targets = [plot_dir / row[1] for row in _PLOTS]
        if any(path.exists() or path.is_symlink() or str(path) in bound for path in targets):
            raise FileExistsError("Model plot completion never overwrites existing PNGs")
        if not plot_dir.exists():
            new_plot_dirs.append(plot_dir)
        checked.extend(paths)
        png_targets.extend(targets)
        contexts.append((entry, directory, report, visual, loads, config))
        jobs.append((source, directory, report))
    if len(set(png_targets)) != len(png_targets):
        raise ValueError("Duplicate selected suite cases")
    allowlist = [output / "assets" / name for name in _ASSETS]
    allowlist += [path for _, directory, *_ in contexts for path in (directory / "workbench.json", directory / "report.html")]
    allowlist += [output / name for name in ("batch.json", "verification.json", "index.html", "artifact_manifest.json")]
    if any(str(path) in protected_paths for path in allowlist + png_targets):
        raise ValueError("Completion target is protected by the previous site manifest")
    checked_rows = [bound[str(path)] for path in dict.fromkeys(checked)] + [artifact_snapshot]
    source_implementation = implementation()
    original_bytes = {path: path.read_bytes() for path in allowlist}
    committed = []
    lock = output / ".complete-model-plots.lock"
    with lock.open("xb"):
        pass
    generation_started = False
    try:
        # Snapshots were prepared before acquiring the lock. A prior completion
        # must not let this call mistake its files for newly owned PNGs.
        _verify(checked_rows)
        if any(path.exists() or path.is_symlink() for path in png_targets):
            raise FileExistsError("Model plot targets appeared before generation")
        generation_started = True
        descriptors = _generate_all(jobs, workers)
        if len(descriptors) != len(contexts):
            raise ValueError("Not every selected model plot job completed")
        replacements = {}
        generated_rows = []
        for context, descriptor in zip(contexts, descriptors):
            entry, directory, report, visual, loads, config = context
            if (descriptor.get("status") != "available" or descriptor.get("comparison_kind") != "model_only"
                    or descriptor.get("task") != "usb-insert" or descriptor.get("seed") != report["trial"].get("seed")
                    or descriptor.get("prediction_horizon") != 50 or descriptor.get("provenance") != report["provenance"]
                    or len(descriptor.get("plots", [])) != len(_PLOTS)):
                raise ValueError("Generated gallery does not identify this model-only native record")
            _verify(descriptor["provenance"])
            expected = {"plots/" + row[1] for row in _PLOTS}
            if {plot.get("url") for plot in descriptor["plots"]} != expected:
                raise ValueError("Generated model PNG target names differ from the allowlist")
            for plot in descriptor["plots"]:
                row = _digest(directory / plot["url"])
                if row["sha256"] != plot["sha256"]:
                    raise ValueError("Generated model PNG SHA256 differs")
                generated_rows.append(row)
            config["diagnostic_plots"] = descriptor
            replacements[directory / "workbench.json"] = _encoded(config)
            replacements[directory / "report.html"] = _page(report, visual, loads, config).encode("utf-8")
            entry.update(plots="available", plot_kind="model_only")
        for name in _ASSETS:
            replacements[output / "assets" / name] = (WEB / name).read_bytes()
        batch["coverage"]["plots"] = sum(entry.get("plots") == "available" for entry in entries)
        total = sum(entry.get("plot_kind") == "model_only" for entry in entries)
        verification.update(implementation=source_implementation, coverage=batch["coverage"],
                            generated_model_plot_trial_count=total,
                            model_plot_completion={"completed_utc": datetime.now(timezone.utc).isoformat(),
                                "new_trial_count": len(selected), "kind": "model_only", "metrics_recomputed": False,
                                "source_manifest_unchanged": True, "report_visualization_tactile_unchanged": True})
        replacements[output / "batch.json"] = _encoded(batch)
        replacements[output / "verification.json"] = _encoded(verification)
        replacements[output / "index.html"] = _portal(batch).encode("utf-8")
        revised_rows = dict(bound)
        for path, raw in replacements.items():
            revised_rows[str(path)] = _bytes_row(path, raw)
        for row in generated_rows:
            revised_rows[row["path"]] = row
        replacements[artifact_path] = _encoded({**original_manifest, "files": list(revised_rows.values())})
        if set(replacements) != set(allowlist):
            raise ValueError("Completion replacement set differs from the explicit allowlist")
        with tempfile.TemporaryDirectory(prefix=".suite-figures-", dir=output) as temporary:
            stage = Path(temporary)
            for path, raw in replacements.items():
                relative = path.relative_to(output)
                for name, content in (("next", raw), ("before", original_bytes[path])):
                    staged = stage / name / relative
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(content)
            _verify(checked_rows)
            _verify(source_implementation)
            for descriptor in descriptors:
                _verify(descriptor["provenance"])
            verify_rollback(output / "rollback.json")
            try:
                for path in allowlist:
                    os.replace(stage / "next" / path.relative_to(output), path)
                    committed.append(path)
                _verify([row for row in checked_rows if Path(row["path"]) not in allowlist])
                _verify(source_implementation)
                _verify(generated_rows)
                for descriptor in descriptors:
                    _verify(descriptor["provenance"])
                verify_rollback(output / "rollback.json")
                for path in allowlist:
                    _verify([_bytes_row(path, replacements[path])])
            except BaseException:
                for path in reversed(committed):
                    os.replace(stage / "before" / path.relative_to(output), path)
                raise
        return {"completed": True, "output": str(output), "generated_count": len(selected),
                "generated_model_plot_trial_count": total, "updated": [str(path.relative_to(output)) for path in allowlist],
                "png_count": len(generated_rows), "metrics_recomputed": False, "payloads_unchanged": True}
    except BaseException:
        # Workers have exited before this point; remove only names absent before
        # this call, never a pre-existing gallery or any model/expert video.
        if generation_started:
            for path in png_targets:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            for directory in new_plot_dirs:
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
        raise
    finally:
        lock.unlink()
