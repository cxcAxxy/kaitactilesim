"""Read-only whiteboard case-study inputs; no resampling or compatibility claims.

Every archived source checksum is revalidated. A reference is admitted only
through an explicit expert-reference membership manifest and replay/HDF binding.
Matching recorded timestamps establishes association, not equal physical phase.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import h5py
import numpy as np

from evaluation_v2.expert import _node
from evaluation_v2.horizon import inspect_prediction_horizon
from kaihand_tactile_env.shared.task_demo import DEFAULT_DEMO_ROOT, locate_demo_root, resolve_demo_path


MAX_JSON_BYTES = 20_000_000
MAX_SOURCE_BYTES = 2 * 1024**3
MAX_SAMPLES = 250_000
MAX_HDF_VALUES = 1_000_000
_SHA = re.compile(r"[a-fA-F0-9]{64}\Z")
_RIGHT_NAMES = [f"hand_r_{finger}_link{link}" for finger, link in
                (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4))]
_PHASE_WARNING = "同记录关联，非瞬时同相位验证；专家力记录的积分前求解相位与状态积分后相位不能视为已严格对齐。"
_COMPARISON_WARNING = "专家约100 Hz、模型约30 Hz；仅作独立时钟的描述性案例参照，不生成跨运行分数、排名或安全带。"


def _path(value):
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise ValueError("Expected a nonempty local path")
    absolute = Path(os.path.abspath(value))
    if absolute.resolve(strict=True) != absolute:
        raise ValueError("Symlinked evidence paths are unsupported")
    return absolute


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(path, *, limit, collect=False):
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    chunks, digest, size = [], hashlib.sha256(), 0
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("Unsupported or oversized evidence file")
        while chunk := stream.read(min(1024 * 1024, limit - size + 1)):
            size += len(chunk)
            if size > limit:
                raise ValueError("Evidence exceeds bounded read limit")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    if size != before.st_size or _signature(before) != _signature(after) or _signature(after) != _signature(path.stat()):
        raise ValueError("Evidence changed while reading")
    return b"".join(chunks), digest.hexdigest(), size, _signature(after)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON metadata key")
        result[key] = value
    return result


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def _constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _parse(raw):
    value = json.loads(raw, object_pairs_hook=_pairs, parse_float=_float, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object metadata")
    return value


def _json(path, provenance, signatures, role):
    path = _path(path)
    raw, digest, size, signature = _read(path, limit=MAX_JSON_BYTES, collect=True)
    provenance.append({"path": str(path), "sha256": digest, "size": size, "role": role})
    signatures[path] = signature
    return _parse(raw)


def _verify_rows(rows, allowed, provenance, signatures):
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("Archived report requires bounded nonempty source provenance")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not _SHA.fullmatch(row["sha256"]):
            raise ValueError("Malformed archived source SHA256")
        path = _path(row.get("path"))
        if path in seen or not allowed(path, row.get("role")):
            raise ValueError("Duplicate or out-of-scope archived source provenance")
        seen.add(path)
        _, digest, size, signature = _read(path, limit=MAX_SOURCE_BYTES)
        if digest != row["sha256"].lower() or ("size" in row and row["size"] != size):
            raise ValueError(f"Archived source SHA256 or size mismatch: {path}")
        provenance.append({"path": str(path), "sha256": digest, "size": size, "role": row.get("role")})
        signatures[path] = signature


def _stable(signatures):
    if any(_signature(path.stat()) != signature for path, signature in signatures.items()):
        raise ValueError("Evidence changed during episode adaptation")


def _clock(values, label):
    if not isinstance(values, (list, np.ndarray)) or len(values) > MAX_SAMPLES:
        raise ValueError(f"Invalid or oversized {label} clock")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"Invalid {label} clock")
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid {label} clock; no sorting, dropping or interpolation is allowed")
    return times


def _values(values, count, label, *, progress=False):
    if not isinstance(values, (list, np.ndarray)) or len(values) != count:
        raise ValueError(f"{label} sample count does not match its clock")
    clean = []
    for value in values:
        valid = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
        number = float(value) if valid else None
        clean.append(number if number is not None and math.isfinite(number) and number >= 0
                     and (not progress or number <= 1) else None)
    return clean


def _gaps(times, declared=None):
    if declared is not None and (not isinstance(declared, list) or len(declared) != len(times)
                                 or any(type(value) is not bool for value in declared)):
        raise ValueError("Invalid native gap flags")
    differences = np.diff(times)
    flags = (differences > 1.5 * np.median(differences) + 1e-10).tolist() + [False] if len(differences) else [False]
    return [flag or bool(declared and declared[index]) for index, flag in enumerate(flags)]


def _exact_match(target_time, source_time, values):
    # Float equality is intentional. Nearby timestamps never substitute for a
    # recorded simultaneous sample; unequal clocks retain explicit nulls.
    by_time = {float(time): value for time, value in zip(source_time, values)}
    return [by_time.get(float(time)) for time in target_time]


def _merge_gaps(target_time, flags, source_time, source_flags):
    """Mark target intervals overlapping recorded gaps; never fill samples."""
    for index, missing in enumerate(source_flags[:-1]):
        if not missing:
            continue
        first = max(0, int(np.searchsorted(target_time, source_time[index], side="right")) - 1)
        stop = min(len(target_time) - 1, int(np.searchsorted(target_time, source_time[index + 1], side="left")))
        for target_index in range(first, stop):
            flags[target_index] = True


def _hand_sum(force, names, count, warnings):
    if (force is None or not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or not all(name in names for name in _RIGHT_NAMES)):
        warnings.append("Missing or ambiguous right-hand fingertip channels; hand force remains unknown")
        return [None] * count
    array = np.asarray(force, dtype=object)
    if array.shape != (count, len(names)):
        raise ValueError("Fingertip force dimensions do not match its native clock/channels")
    columns = [_values(array[:, names.index(name)], count, "fingertip force") for name in _RIGHT_NAMES]
    return [sum(values) if all(value is not None for value in values) else None for values in zip(*columns)]


def _base(report, role, times, progress, gaps, board, hand, provenance, warnings, source_report):
    trial = report["trial"]
    recorded = report.get("outcome")
    outcome = recorded.get("status", "unknown") if isinstance(recorded, dict) else "unknown"
    if outcome not in {"pass", "fail", "unknown"}:
        outcome = "unknown"
    dt = float(np.median(np.diff(times))) if len(times) > 1 else None
    return {"id": trial["trial_id"], "role": role, "task": "whiteboard-wipe", "outcome": outcome,
            "time_s": times.tolist(), "progress": progress, "gap_after": gaps,
            "signals": {
                "board_normal": {"label": "板擦—白板法向力", "unit": "N", "contact_role": "eraser_board", "values": board},
                "hand_normal_sum": {"label": "右手五指法向力总和（接触对象未独立核验）", "unit": "N",
                                    "contact_role": "hand_object_unresolved", "values": hand}},
            "metadata": {"sample_count": len(times), "nominal_interval_s": dt,
                         "nominal_cadence_hz": 1 / dt if dt else None,
                         "sampling_hz": 1 / dt if dt else None,
                         "sampling_rate_semantics": "Reciprocal median recorded interval; not a sensor-bandwidth equivalence claim",
                         "compatibility": "unverified", "quantitative_comparison": "unverified",
                         "physical_phase_alignment": "unverified_or_offset",
                         "progress_definition_comparability": "unverified",
                         "sampling": "Native recorded samples only; no resampling, nearest-neighbour fill or cross-run clock alignment",
                         "selection_bias": "Selected archived examples; not an unbiased population",
                         "progress_is_causal_evidence": False},
            "provenance": provenance, "warnings": warnings, "source_report": str(source_report),
            "media": {"status": "missing", "reason": "No verified episode-local video mapping"}}


def _report(path, provenance, signatures):
    path = _path(path)
    report = _json(path, provenance, signatures, "progress_source_report")
    trial = report.get("trial")
    if not isinstance(trial, dict) or trial.get("task") != "whiteboard-wipe" or not isinstance(trial.get("trial_id"), str) or not trial["trial_id"]:
        raise ValueError("This experiment accepts explicitly identified tilted whiteboard-wipe trials only")
    return path, report


def _series(visual):
    rows = visual.get("task_series", [])
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("Invalid task-series metadata")
    series = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("key") in series:
            raise ValueError("Malformed or duplicate task series")
        key = row.get("key")
        if not isinstance(key, str):
            raise ValueError("Task-series key must be a string")
        times = _clock(row.get("time_s"), key)
        series[key] = (times, _values(row.get("values"), len(times), key, progress=key == "cleaned_fraction"),
                       _gaps(times, row.get("gap_after")))
    return series


def load_policy_episode(report_path: Path) -> dict:
    provenance, signatures, warnings = [], {}, [_PHASE_WARNING, _COMPARISON_WARNING]
    report_path, report = _report(report_path, provenance, signatures)
    trial = report["trial"]
    source = _path(trial.get("source"))
    horizon = inspect_prediction_horizon(source)
    archived = report.get("prediction_horizon_evidence", {})
    if (not isinstance(archived, dict) or type(trial.get("prediction_horizon")) is not int
            or trial.get("prediction_horizon") != 50 or archived.get("status") != "verified"
            or archived.get("prediction_horizon") != 50 or horizon["status"] != "verified"
            or horizon["prediction_horizon"] != 50):
        raise ValueError("Policy input must be H50 in both archived report and rechecked source summary")
    allowed_manifest = {Path(row["path"]) for row in horizon["provenance"]}
    _verify_rows(report.get("provenance"), lambda path, role: path.is_relative_to(source) or path in allowed_manifest,
                 provenance, signatures)
    summary_path = source / "summary.json"
    expected = next((row["sha256"] for row in report["provenance"] if row.get("path") == str(summary_path)), None)
    if expected != next(row["sha256"] for row in horizon["provenance"] if row["path"] == str(summary_path)):
        raise ValueError("Archived report is not bound to the inspected source summary")
    summary = _json(summary_path, provenance, signatures, "progress_source_summary")
    if summary.get("task") != "whiteboard-wipe" or provenance[-1]["sha256"] != expected:
        raise ValueError("Source summary task or SHA256 disagrees with the archived policy report")
    visual = _json(report_path.parent / "visualization.json", provenance, signatures, "progress_source_visualization")
    if visual.get("schema_version") != "evaluation-visualization-v1" or visual.get("task_summary", {}).get("task") != "whiteboard-wipe":
        raise ValueError("Unsupported or cross-task visualization")
    series = _series(visual) if visual.get("clock_aligned") is True else {}
    if visual.get("clock_aligned") is not True:
        warnings.append("Task/tactile clock association is unverified; task progress and board force remain unknown")
    tactile = visual.get("tactile_series", {})
    tactile_time = _clock(tactile.get("time_s"), "tactile") if tactile.get("time_s") else None
    base = series.get("cleaned_fraction") or series.get("board_normal_force_n")
    if base:
        times, _, gaps = base
    elif tactile_time is not None:
        times, gaps = tactile_time, _gaps(tactile_time, tactile.get("gap_after"))
    else:
        raise ValueError("No valid native clock remains for the archived episode")
    def matched(key):
        row = series.get(key)
        return _exact_match(times, row[0], row[1]) if row else [None] * len(times)
    progress, board = matched("cleaned_fraction"), matched("board_normal_force_n")
    board_series = series.get("board_normal_force_n")
    if board_series is not None and any(value is not None for value in board):
        _merge_gaps(times, gaps, board_series[0], board_series[2])
    if all(value is None for value in progress):
        warnings.append("Missing native cleaned_fraction; episode retained with unknown progress")
    hand = [None] * len(times)
    if tactile_time is not None:
        names = [row.get("name") for row in visual.get("channels", []) if isinstance(row, dict)]
        totals = _hand_sum(tactile.get("normal_force_n"), names, len(tactile_time), warnings)
        hand = _exact_match(times, tactile_time, totals)
        tactile_gaps = _gaps(tactile_time, tactile.get("gap_after"))
        if any(value is not None for value in hand):
            _merge_gaps(times, gaps, tactile_time, tactile_gaps)
        if any(value is None for value in hand):
            warnings.append("Hand force uses exact native timestamps only; unmatched/invalid samples remain null")
    episode = _base(report, "policy", times, progress, gaps, board, hand, provenance, warnings, report_path)
    episode["metadata"].update(prediction_horizon=50,
        record_clock_association="Archived visualization task/tactile common simulation-clock declaration; exact timestamps only",
        progress_definition="Archived control_steps.task.cleaned_fraction; no reconstruction or terminal-value fill",
        progress_definition_note="Current source defines mean(1 - ink_remaining); archived policy-source bytes differ, so historical formula binding is not certified",
        gap_semantics="Conservative union of native base-clock gaps and adopted board/hand signal gaps on overlapping recorded intervals; no resampling",
        source_time_origin_s=visual.get("time_origin_s"),
        selection_bias="Explicitly selected H50 model cases; failed and missing-progress episodes retained")
    media = visual.get("media", {})
    if isinstance(media, dict) and media.get("status") == "available" and media.get("sync_status") == "mapped":
        url = media.get("url")
        if not isinstance(url, str) or not re.fullmatch(r"media/[A-Za-z0-9_.-]+\.mp4", url):
            raise ValueError("Unsupported episode-local media URL")
        mapped_time = _clock(media.get("time_s"), "media simulation")
        video_time = _clock(media.get("video_time_s"), "video")
        if len(mapped_time) != len(video_time) or len(video_time) != media.get("frame_count"):
            raise ValueError("Media mapping dimensions disagree")
        path = _path(report_path.parent / url)
        _verify_rows([{"path": str(path), "sha256": media.get("sha256"), "role": "progress_local_media"}],
                     lambda candidate, role: candidate.is_relative_to(report_path.parent), provenance, signatures)
        episode["media"] = {**deepcopy(media), "source_path": str(path), "source_report": str(report_path)}
    warnings.extend(horizon["warnings"])
    _stable(signatures)
    return episode


def _hdf_array(file, path):
    dataset = _node(file, path)
    if dataset is None:
        return None
    if not isinstance(dataset, h5py.Dataset) or dataset.size > MAX_HDF_VALUES or dataset.dtype.kind not in "fiu":
        raise ValueError(f"Unsupported or oversized numeric HDF dataset: {path}")
    if dataset.ndim not in (1, 2) or dataset.shape[0] > MAX_SAMPLES:
        raise ValueError(f"Unsupported HDF dimensions: {path}")
    return np.asarray(dataset[...], dtype=float)


def _attr_json(file, key):
    raw = file.attrs.get(key)
    if not isinstance(raw, (str, bytes)) or len(raw) > 1_000_000:
        raise ValueError(f"Missing or oversized HDF metadata: {key}")
    return _parse(raw)


def load_reference_episode(reference_report_path: Path) -> dict:
    provenance, signatures, warnings = [], {}, [_PHASE_WARNING, _COMPARISON_WARNING]
    report_path, report = _report(reference_report_path, provenance, signatures)
    trial = report["trial"]
    imported = report.get("reference_import", {})
    if not isinstance(imported, dict) or imported.get("status") != "imported" or imported.get("source_schema") != "kaihand_tactile_episode_v1":
        raise ValueError("Reference must be an explicitly imported task-demo HDF report")
    rows = report.get("provenance", [])
    if not isinstance(rows, list):
        raise ValueError("Reference source provenance must be an array")
    hdf_rows = [row for row in rows if isinstance(row, dict) and row.get("role") == "expert_hdf5"]
    replay_rows = [row for row in rows if isinstance(row, dict) and row.get("role") == "explicit_task_demo_replay_selection"]
    migration_rows = [row for row in rows if isinstance(row, dict) and row.get("role") == "task_demo_migration_manifest"]
    if len(rows) != 2 + len(migration_rows) or len(migration_rows) > 1 or len(hdf_rows) != 1 or len(replay_rows) != 1:
        raise ValueError("Reference requires exactly one bound HDF and task-demo replay")
    source_name, replay_name = hdf_rows[0].get("path"), replay_rows[0].get("path")
    if (not isinstance(source_name, str) or not Path(source_name).is_absolute()
            or source_name != trial.get("source") or Path(source_name).suffix.lower() not in {".h5", ".hdf5"}):
        raise ValueError("Reference source identity or file type mismatch")
    replay_path = _path(resolve_demo_path(replay_name, expected_sha256=replay_rows[0].get("sha256")))
    demo_root = locate_demo_root(replay_path)
    if demo_root is None and replay_path.is_relative_to(DEFAULT_DEMO_ROOT):
        raise ValueError("Project task_demo requires migration_manifest.json; refusing a NAS fallback")
    source = _path(resolve_demo_path(source_name, demo_root=demo_root,
                                    expected_sha256=hdf_rows[0].get("sha256")))
    if replay_path.name != "replay.json" or (demo_root is not None and not source.is_relative_to(demo_root)):
        raise ValueError("Reference replay/source has no exact local task-demo mapping")
    if ("local_path" in hdf_rows[0] and (demo_root is None or hdf_rows[0]["local_path"] != str(source))
            or "resolved_source" in trial and trial["resolved_source"] != str(source)):
        raise ValueError("Reference local source differs from its SHA-bound migration mapping")
    migration_path = demo_root / "migration_manifest.json" if demo_root is not None else None
    if migration_rows and (migration_path is None or migration_rows[0].get("path") != str(migration_path)):
        raise ValueError("Reference migration manifest is not bound to the selected local replay")
    normalized = [{**hdf_rows[0], "path": str(source)}, {**replay_rows[0], "path": str(replay_path)}, *migration_rows]
    start = len(provenance)
    _verify_rows(normalized, lambda path, role: (path == source and role == "expert_hdf5") or
                 (path == replay_path and role == "explicit_task_demo_replay_selection") or
                 (path == migration_path and role == "task_demo_migration_manifest"), provenance, signatures)
    for row, original in zip(provenance[start:], (source_name, replay_name)):
        if row["path"] != original:
            row["original_path"] = original
    if migration_path is not None and not migration_rows:
        _json(migration_path, provenance, signatures, "task_demo_migration_manifest")
    replay = _json(replay_path, provenance, signatures, "progress_replay_metadata")
    if (replay.get("schema_version") != "kaihand-offline-multimodal-replay-v1" or replay.get("completed") is not True
            or replay.get("task") != "whiteboard-wipe" or replay.get("source_hdf5") != source_name
            or replay.get("source_sha256") != hdf_rows[0]["sha256"]):
        raise ValueError("Replay task/completion/HDF SHA256 binding mismatch")
    manifest = _json(report_path.parent.parent / "reference_manifest.json", provenance, signatures, "progress_reference_membership")
    if (manifest.get("schema_version") != "conditional-reference-manifest-v1" or manifest.get("role") != "expert_reference"
            or any(not isinstance(manifest.get(key), str) or not manifest[key].strip()
                   for key in ("declared_by", "selection_rule", "evidence"))):
        raise ValueError("Explicit expert-reference membership and selection evidence are required")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) > 10000:
        raise ValueError("Invalid reference membership list")
    matching = [row for row in members if isinstance(row, dict) and row.get("trial_id") == trial["trial_id"]]
    if len(matching) != 1 or matching[0].get("source") != source_name or matching[0].get("observation_sha256") != hdf_rows[0]["sha256"]:
        raise ValueError("Reference report is not a unique checksum-bound declared member")
    with h5py.File(source, "r") as file:
        if file.attrs.get("schema_version") != "kaihand_tactile_episode_v1":
            raise ValueError("Unsupported native HDF schema")
        metadata = _attr_json(file, "metadata_json")
        if metadata.get("scene") != "whiteboard-wipe" or metadata.get("recording_contract") != "whiteboard_wipe_shared_raw_v1":
            raise ValueError("HDF does not declare the audited whiteboard recording contract")
        times = _clock(_hdf_array(file, "/state/timestamp"), "native state")
        ink = _hdf_array(file, "/whiteboard_wipe/ink_remaining")
        progress = [None] * len(times)
        if ink is None:
            warnings.append("Missing native ink_remaining; reference retained with unknown progress")
        else:
            if ink.ndim != 2 or ink.shape[0] != len(times) or not 1 <= ink.shape[1] <= 1024:
                raise ValueError("Ink-patch dimensions do not match native state samples")
            valid = np.all(np.isfinite(ink) & (ink >= 0) & (ink <= 1), axis=1)
            progress = [float(value) if ok else None for value, ok in zip(np.mean(1 - ink, axis=1), valid)]
            if not valid.all():
                warnings.append("Invalid ink-patch rows retain unknown progress; no clipping or fill")
        board_raw = _hdf_array(file, "/whiteboard_wipe/board_force_n")
        if board_raw is not None and board_raw.ndim != 1:
            raise ValueError("Board normal force must be a scalar per native state sample")
        board = _values(board_raw, len(times), "board normal force") if board_raw is not None else [None] * len(times)
        force = _hdf_array(file, "/tactile_contact_force/normal_force_n")
        force_clock = _hdf_array(file, "/tactile_contact_force/timestamp")
        names_node = _node(file, "/tactile_contact_force/link_names")
        names = []
        if names_node is not None:
            if not isinstance(names_node, h5py.Dataset) or names_node.shape not in {(10,), (5,)}:
                raise ValueError("Unexpected fingertip channel names")
            names = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in names_node[...]]
        hand = [None] * len(times)
        if force_clock is not None:
            tactile_time = _clock(force_clock, "native tactile")
            hand = _exact_match(times, tactile_time, _hand_sum(force, names, len(tactile_time), warnings))
            if any(value is None for value in hand):
                warnings.append("Native state/tactile timestamps do not all match exactly; unmatched hand forces remain null")
        force_group = _node(file, "/tactile_contact_force")
        phase = str(force_group.attrs.get("clock_semantics", "unverified")) if force_group is not None else "unverified"
        force_unit = str(force_group.attrs.get("force_unit", "unverified")) if force_group is not None else "unverified"
        if force_unit != "N":
            hand = [None] * len(times)
            warnings.append("Native fingertip force unit is not declared N; hand force omitted")
    episode = _base(report, "expert_reference", times, progress, _gaps(times), board, hand,
                    provenance, warnings, report_path)
    episode["metadata"].update(
        record_clock_association="whiteboard_wipe arrays share state recording indices; fingertip forces require exact numeric timestamps",
        source_clock="/state/timestamp", force_clock_semantics=phase,
        progress_definition="Derived mean(1 - /whiteboard_wipe/ink_remaining, axis=1); not an archived cleaned_fraction field",
        progress_definition_note="Formula audited against current whiteboard source; historical policy-source binding and cross-scene equivalence remain unverified",
        selection_bias=manifest["selection_rule"], expert_role_evidence=manifest["declared_by"],
        reference_id=manifest.get("reference_id"), source_hdf5=source_name,
        membership_semantics="Explicit selection only; no expertise, outcome or compatibility inferred")
    if str(source) != source_name:
        episode["metadata"]["local_source_hdf5"] = str(source)
    warnings.append("单条成功筛选专家示范不构成专家总体分布或无偏成功率；进度定义关联不证明同场景、同动力学或因果关系。")
    _stable(signatures)
    return episode
