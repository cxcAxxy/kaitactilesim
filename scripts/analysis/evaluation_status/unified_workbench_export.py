"""Compose a single workbench from immutable, already-published snapshots.

This module does not discover jobs, import simulators, reread producer summaries,
or verify live model weights.  Recorded validity and stage admissions are copied
from their published catalogs; artifact hashes bind what is actually displayed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from .export import PACKAGE, _publish_portal, _write, digest, encoded, verify_row

SCHEMA = "kaihand-unified-evaluation-workbench-v1"
SNAPSHOT_SCHEMA = "kaihand-unified-evaluation-workbench-snapshot-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON = 64 * 1024 * 1024
_METRICS = (
    ("cleaned_fraction", "终态清洁度", "fraction"),
    ("contact_time_weighted_mean_effective_area_ratio", "接触期间平均贴合面积比（几何估计）", "fraction"),
    ("pad_board_contact_duration_s", "板擦与板面接触时长", "s"),
)


def _object_pairs(rows):
    value = {}
    for key, item in rows:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_JSON:
        raise ValueError(f"Expected bounded regular JSON artifact: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed while reading: {path}")
    value = json.loads(raw, object_pairs_hook=_object_pairs,
                      parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _int(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"Expected nonnegative integer for {label}")
    return value


def _bool(value):
    return value if type(value) is bool else None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _rows(value, label):
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected object list for {label}")
    return value


def _family(value):
    if not isinstance(value, str):
        raise ValueError("Missing model family")
    return {"pi0.5": "pi05", "pi05": "pi05", "egotouch": "egotouch",
            "ftp-1": "ftp1", "ftp1": "ftp1", "pi05+cpm": "pi05+cpm"}.get(value.lower(), value.lower())


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class _Artifacts:
    """Verify only catalog/snapshot bytes, never their retired original sources."""

    def __init__(self, reports, output):
        self.reports = Path(reports).resolve(strict=True)
        self.output = Path(output).absolute()
        self.inputs = {}
        self.inventories = {}

    def record(self, row, role):
        row = {**row, "role": role}
        previous = self.inputs.get(row["path"])
        if previous and (previous["size"], previous["sha256"]) != (row["size"], row["sha256"]):
            raise ValueError(f"Artifact identity changed within build: {row['path']}")
        self.inputs.setdefault(row["path"], row)

    def inventory(self, root):
        root = Path(root).resolve(strict=True)
        if root.parent != self.reports:
            raise ValueError("Source snapshots must be siblings inside the report collection")
        if root in self.inventories:
            return self.inventories[root]
        value, identity = _read(root / "artifact_manifest.json")
        self.record(identity, "published_snapshot_artifact_manifest")
        result = {}
        for row in _rows(value.get("files"), "artifact inventory"):
            path = row.get("path")
            if not isinstance(path, str) or "\\" in path or "\x00" in path:
                raise ValueError("Unsafe artifact inventory path")
            pure = PurePosixPath(path)
            if pure.is_absolute():
                # Historical copies retain their original absolute v1 inventory.
                matches = [i for i, part in enumerate(pure.parts) if part == root.name]
                if len(matches) != 1:
                    raise ValueError("Cannot relocate historical artifact inventory")
                pure = PurePosixPath(*pure.parts[matches[0] + 1:])
            if not pure.parts or ".." in pure.parts or pure.as_posix() in result:
                raise ValueError("Unsafe or duplicate artifact inventory entry")
            if not _sha(row.get("sha256")) or type(row.get("size")) is not int or row["size"] <= 0:
                raise ValueError("Invalid published artifact identity")
            result[pure.as_posix()] = {"path": str(root / Path(*pure.parts)),
                                      "size": row["size"], "sha256": row["sha256"]}
        if not result:
            raise ValueError("Empty published artifact inventory")
        self.inventories[root] = result
        return result

    def artifact(self, path, role="linked_published_artifact"):
        # The output is intentionally absent until all inputs pass. Collapse its
        # lexical sibling links before strict resolution of the existing target.
        path = Path(os.path.abspath(path))
        if path.is_symlink():
            raise ValueError("Published artifact leaf cannot be a symlink")
        actual = path.resolve(strict=True)
        if not actual.is_relative_to(self.reports) or not actual.is_file():
            raise ValueError("Artifact escapes report collection")
        parts = actual.relative_to(self.reports).parts
        if len(parts) < 2:
            raise ValueError("Artifact must belong to an inventoried report snapshot")
        root = self.reports / parts[0]
        row = self.inventory(root).get(PurePosixPath(*parts[1:]).as_posix())
        if row is None:
            raise ValueError(f"Linked artifact is not inventoried: {actual}")
        if str(actual) not in self.inputs:
            verify_row(row, actual)
            self.record({**row, "path": str(actual)}, role)
        return actual

    def document(self, path, role="published_catalog"):
        path = self.artifact(path, role)
        value, identity = _read(path)
        row = self.inputs[str(path)]
        if (identity["size"], identity["sha256"]) != (row["size"], row["sha256"]):
            raise ValueError("Published JSON changed after artifact verification")
        return value

    def target(self, source, href):
        if (not isinstance(href, str) or not href or any(c in href for c in ("\\", "\x00", "\r", "\n"))):
            raise ValueError("Invalid published relative URL")
        url = urlsplit(href)
        if url.scheme or url.netloc or url.query or url.fragment or url.path.startswith("/"):
            raise ValueError("Only relative snapshot URLs are accepted")
        decoded = unquote(url.path)
        if any(c in decoded for c in ("\\", "\x00", "\r", "\n")):
            raise ValueError("Unsafe encoded artifact URL")
        return self.artifact(Path(source) / decoded)

    def href(self, path):
        target = self.artifact(path)
        return quote(Path(os.path.relpath(target, self.output)).as_posix(), safe="/._-")

    def link(self, source, href):
        return self.href(self.target(source, href))

    def trial_links(self, source, trial):
        detail = self.target(source, trial.get("detail_url"))
        video = None
        # Some H50 composed catalogs retained the status snapshot's local video
        # URL while correctly rebasing its detail URL. Resolve via that verified
        # detail directory instead of following the producer's live video_path.
        declared = trial.get("historical_video_url") or trial.get("video_url")
        if declared:
            try:
                video = self.target(source, declared)
            except FileNotFoundError:
                # Only this known local-file omission permits the two archived
                # media conventions below. Unsafe URLs and hash errors fail.
                pass
        if video is None:
            candidates = (detail.parent / "review.mp4", detail.parent / "media" / "rollout.mp4")
            for candidate in candidates:
                if candidate.exists():
                    video = self.artifact(candidate)
                    break
        return self.href(detail), self.href(video) if video is not None else None


def _stages(result):
    if not isinstance(result, dict):
        return []
    definitions, flags = result.get("stage_definitions"), result.get("stage_flags")
    if not isinstance(definitions, dict) or not isinstance(flags, dict):
        raise ValueError("Published staged projection is missing its definitions/flags")
    return [{"key": key, "name": name, "reached": _bool(flags.get(key))}
            for key, name in definitions.items()]


def _stats(trials):
    eligible = [trial for trial in trials if trial["valid"] is True]
    native = [trial["native_success"] for trial in eligible if type(trial["native_success"]) is bool]
    stage_trials = [trial for trial in eligible if trial["staged_evidence_status"] == "valid"]
    complete = [trial["complete_stage_success"] for trial in stage_trials
                if type(trial["complete_stage_success"]) is bool]
    definitions = {}
    stages = {}
    for trial in stage_trials:
        for stage in trial["stages"]:
            key = stage["key"]
            if key in definitions and definitions[key] != stage["name"]:
                raise ValueError("One protocol group contains different recorded stage definitions")
            definitions[key] = stage["name"]
            row = stages.setdefault(key, {"key": key, "name": stage["name"], "reached": 0, "known": 0})
            if type(stage["reached"]) is bool:
                row["known"] += 1
                row["reached"] += stage["reached"] is True
    return {
        "planned_trials": len(trials), "valid_trials": len(eligible),
        "invalid_trials": sum(trial["valid"] is False for trial in trials),
        "native_success_count": native.count(True), "native_known_count": len(native),
        "complete_stage_success_count": complete.count(True), "complete_stage_known_count": len(complete),
        "stages": list(stages.values()),
        "counts": {"members": len(trials), "eligible": len(eligible),
                   "quarantined": sum(trial["publication_status"] == "quarantined_replay_only" for trial in trials),
                   "staged_verified": len(stage_trials),
                   "staged_unknown": len(trials) - len(stage_trials)},
    }


def _compare_stats(actual, source, *, tactile=False):
    if tactile:
        expected = {"planned_trials": source["counts"]["planned"],
                    "valid_trials": source["counts"]["valid"],
                    "native_success_count": source["native"]["successes"],
                    "native_known_count": source["native"]["known"],
                    "complete_stage_success_count": source["staged"]["successes"],
                    "complete_stage_known_count": source["staged"]["known"]}
        stage_rows = source["staged"].get("stages", [])
    else:
        keys = ("planned_trials", "valid_trials", "invalid_trials", "native_success_count",
                "native_known_count", "complete_stage_success_count", "complete_stage_known_count")
        expected = {key: source[key] for key in keys}
        stage_rows = source.get("stages", [])
    for key, value in expected.items():
        if actual[key] != _int(value, key):
            raise ValueError(f"Published {key} disagrees with its recorded trial admissions")
    expected_stages = {row["key"]: (row["name"], row["reached"], row["known"]) for row in stage_rows}
    actual_stages = {row["key"]: (row["name"], row["reached"], row["known"]) for row in actual["stages"]}
    if expected_stages != actual_stages:
        raise ValueError("Published stage counters disagree with their recorded trial admissions")


def _metric_means(trials):
    result = []
    for key, label, unit in _METRICS:
        values = [trial.get("continuous_metrics", {}).get(key) for trial in trials
                  if trial["valid"] is True and trial["staged_evidence_status"] == "valid"]
        values = [float(value) for value in values if _finite(value)]
        if values:
            mean = sum(values) / len(values)
            display_value = mean * 100 if unit == "fraction" else mean
            result.append({"key": key, "name": label, "label": label, "unit": "%" if unit == "fraction" else unit,
                           "mean": mean, "value": display_value, "display": f"{display_value:.2f}", "known": len(values),
                           "aggregation": "arithmetic_mean_of_recorded_trial_values",
                           "note": "仅已记录数值；缺测不补零。贴合面积是刚性几何估计，不是真实可变形接触面积。"})
    return result


def _base_group(source, *, origin, model, checkpoint, horizon, execute_steps):
    for key in ("id", "task", "title"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Published group is missing {key}")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("Published group is missing its checkpoint label")
    if not _sha(source.get("checkpoint_sha256")):
        raise ValueError("Published group is missing a typed checkpoint SHA-256")
    if type(horizon) is not int or type(execute_steps) is not int or horizon <= 0 or execute_steps <= 0:
        raise ValueError("Published group has invalid horizon or execution length")
    protocol = source.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Published group is missing a protocol object")
    return {"id": f"{origin}:{source['id']}", "task": source["task"], "title": source["title"],
            "model": model, "checkpoint": str(checkpoint), "checkpoint_sha256": source["checkpoint_sha256"],
            "prediction_horizon": horizon, "execute_steps": execute_steps,
            "cohort_id": str(source.get("batch_id", origin)), "cohort_label": str(source.get("batch_id", origin)),
            "mode": source.get("mode", "unverified"), "protocol": copy.deepcopy(protocol),
            "rule_identity": {"sha256": protocol.get("rule_sha256"), "basis": "recorded_archive"},
            "warnings": copy.deepcopy(source.get("warnings", [])), "source_snapshot": origin,
            "source_group_id": source["id"], "trials": []}


def _archive_groups(store, root, catalog, origin, model):
    groups = []
    if catalog.get("schema") != "kaihand-evaluation-status-v1":
        raise ValueError("Unsupported published status catalog schema")
    for source in _rows(catalog.get("groups"), "published status groups"):
        if _family(source.get("model")) != model:
            raise ValueError("Archive model family does not match its declared source")
        if (type(source.get("prediction_horizon")) is not int or source["prediction_horizon"] != 50
                or type(source.get("execute_steps")) is not int or source["execute_steps"] != 50):
            raise ValueError("Archived H50 source cannot admit another horizon/execution protocol")
        group = _base_group(source, origin=origin, model=model, checkpoint=source.get("checkpoint"),
                            horizon=source["prediction_horizon"], execute_steps=source["execute_steps"])
        for trial in _rows(source.get("trials"), "published archive trials"):
            valid = _bool(trial.get("valid"))
            if valid is None or valid and _bool(trial.get("native_success")) is None:
                raise ValueError("Invalid published native admission/result")
            quarantined = trial.get("publication_status") == "quarantined_replay_only"
            if quarantined and valid:
                raise ValueError("Quarantined replay cannot be admitted as a valid trial")
            if not _sha(trial.get("summary_sha256")):
                raise ValueError("Archive trial is missing its published summary identity")
            result = trial.get("staged_result") if valid and not quarantined else None
            detail, video = store.trial_links(root, trial)
            group["trials"].append({
                "id": trial["id"], "seed": _int(trial.get("seed"), "seed"), "valid": valid,
                "status": trial.get("status", "unknown"), "native_success": _bool(trial.get("native_success")) if valid else None,
                "publication_status": "quarantined_replay_only" if quarantined else "eligible" if valid else "invalid",
                "staged_evidence_status": "quarantined" if quarantined else "valid" if isinstance(result, dict) else "unknown",
                "staged_evidence_basis": "recorded_archive" if isinstance(result, dict) else None,
                "stages": _stages(result), "complete_stage_success": _bool(result.get("complete_success")) if isinstance(result, dict) else None,
                "continuous_metrics": copy.deepcopy(result.get("continuous_metrics", {})) if isinstance(result, dict) else {},
                "detail_url": detail, "video_url": video, "notes": copy.deepcopy(trial.get("warnings", [])),
                "summary_sha256": trial["summary_sha256"], "sim_seconds": trial.get("sim_seconds"),
                "recorded_native_success": trial.get("recorded_native_success"),
                "validity_basis": trial.get("validity_basis"), "source_trial_id": trial["id"],
            })
        group.update(_stats(group["trials"]))
        _compare_stats(group, source)
        group["recorded_statistics"] = {key: copy.deepcopy(source[key]) for key in
                                        ("valid_trials", "native_success_count", "native_known_count", "stages",
                                         "complete_stage_success_count", "complete_stage_known_count")}
        group["metrics"] = _metric_means(group["trials"])
        groups.append(group)
    return groups


def _tactile_groups(store, root, catalog):
    if catalog.get("schema") != "kaihand-tactile-policy-experiment-v1":
        raise ValueError("Unsupported tactile experiment catalog schema")
    groups = []
    for source in _rows(catalog.get("groups"), "published tactile groups"):
        source_trials = _rows(source.get("trials"), "published tactile trials")
        expected_counts = {"planned": len(source_trials),
                           "observed": sum(trial.get("observed") is True for trial in source_trials),
                           "valid": sum(trial.get("evidence_status") == "valid" for trial in source_trials),
                           "invalid": sum(trial.get("evidence_status") == "invalid" for trial in source_trials),
                           "missing": sum(trial.get("evidence_status") == "missing" for trial in source_trials)}
        for key, value in expected_counts.items():
            if _int(source.get("counts", {}).get(key), f"tactile {key}") != value:
                raise ValueError(f"Published tactile {key} count disagrees with recorded admissions")
        identity = source.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Tactile group is missing its published identity")
        model = _family(identity.get("model_family"))
        horizon = identity.get("prediction_horizon")
        if model not in ("ftp1", "pi05+cpm") or type(horizon) is not int or horizon != (30 if model == "pi05+cpm" else 32):
            raise ValueError("Tactile source model/horizon identity is unsupported")
        if type(identity.get("execute_steps")) is not int or identity["execute_steps"] != 5:
            raise ValueError("Tactile source must preserve its recorded K5 protocol")
        if identity.get("diagnostic_only") is not True or identity.get("formal_metrics_valid") is not False:
            raise ValueError("Tactile diagnostic source cannot become a formal evaluation")
        label = source["label"]
        group_source = {"id": label, "task": catalog["experiment"]["task"], "title": source["title"],
                        "checkpoint_sha256": identity.get("checkpoint_sha256"), "protocol": identity.get("protocol"),
                        "mode": "diagnostic", "batch_id": catalog["experiment"]["id"]}
        group = _base_group(group_source, origin="tactile", model=model, checkpoint=identity.get("checkpoint_step"),
                            horizon=horizon, execute_steps=identity["execute_steps"])
        group["rule_identity"] = copy.deepcopy(source["staged"].get("rule_contract"))
        group["guidance"] = copy.deepcopy(identity.get("guidance"))
        group["action_start_index"] = identity.get("action_start_index")
        group["tactile_sent_to_model"] = identity.get("tactile_sent_to_model")
        for trial in _rows(source.get("trials"), "published tactile trials"):
            valid = trial.get("evidence_status") == "valid"
            if trial.get("evidence_status") not in ("valid", "invalid", "missing"):
                raise ValueError("Unsupported published tactile trial admission")
            if valid and (_bool(trial.get("native_success")) is None or trial.get("observed") is not True):
                raise ValueError("Valid tactile native trial must be observed with a boolean result")
            detail, video = store.trial_links(root, trial) if trial.get("detail_url") else (None, None)
            result = None
            staged_valid = valid and trial.get("staged_evidence_status") == "verified"
            if staged_valid:
                projection = store.document(store.target(root, trial.get("staged_result_url")), "published_staged_projection")
                result = projection.get("staged_result")
                if not isinstance(result, dict) or _bool(result.get("complete_success")) != trial.get("staged_complete_success"):
                    raise ValueError("Published tactile staged projection disagrees with its admitted result")
            elif valid and trial.get("staged_evidence_status") not in ("invalid", "missing", "unavailable", "unknown"):
                raise ValueError("Unsupported published staged admission")
            summary_sha = None
            if detail is not None:
                detail_path = store.target(store.output, detail)
                summary_path = detail_path.parent / "summary.json"
                if summary_path.exists():
                    store.artifact(summary_path, "copied_published_native_summary")
                    summary_sha = store.inputs[str(summary_path.resolve())]["sha256"]
            if valid and not _sha(summary_sha):
                raise ValueError("Valid tactile native trial has no inventoried copied summary")
            stage_status = "valid" if staged_valid else "invalid" if trial.get("staged_evidence_status") == "invalid" else "unknown"
            group["trials"].append({
                "id": trial["id"], "seed": _int(trial.get("seed"), "seed"), "valid": valid,
                "status": trial.get("status", "unknown"), "native_success": _bool(trial.get("native_success")) if valid else None,
                "publication_status": "eligible" if valid else trial.get("evidence_status", "invalid"),
                "staged_evidence_status": stage_status,
                "staged_evidence_basis": "tactile_manifest_inventory_v1" if staged_valid else None,
                "stages": _stages(result), "complete_stage_success": _bool(result.get("complete_success")) if result else None,
                "continuous_metrics": copy.deepcopy(result.get("continuous_metrics", {})) if result else {},
                "detail_url": detail, "video_url": video, "notes": copy.deepcopy(trial.get("reasons", [])),
                "summary_sha256": summary_sha, "source_trial_id": trial["id"],
                "tactile_sent_to_model": trial.get("tactile_sent_to_model"), "expert_used": trial.get("expert_used"),
                "guidance": copy.deepcopy(trial.get("guidance")),
            })
        group.update(_stats(group["trials"]))
        _compare_stats(group, source, tactile=True)
        group["recorded_statistics"] = copy.deepcopy({"counts": source["counts"], "native": source["native"], "staged": source["staged"]})
        group["metrics"] = _metric_means(group["trials"])
        groups.append(group)
    return groups


def _deduplicate(groups):
    seen, ids, duplicates = {}, {}, []
    for group in groups:
        retained = []
        for trial in group["trials"]:
            sha = trial.get("summary_sha256")
            identity = (group["task"], group["model"], group["checkpoint_sha256"], group["prediction_horizon"],
                        group["execute_steps"], trial["seed"], group["mode"],
                        _canonical(group["protocol"]), _canonical(group["rule_identity"]))
            if trial["id"] in ids and ids[trial["id"]] != sha:
                raise ValueError("Published trial ID refers to conflicting summary bytes")
            ids[trial["id"]] = sha
            if sha is not None and sha in seen:
                previous = seen[sha]
                if previous["identity"] != identity:
                    raise ValueError("Identical summary bytes carry conflicting published identities")
                # Evidence/admission differences must not be silently resolved.
                signature = (trial["valid"], trial["native_success"], trial["publication_status"],
                             trial["staged_evidence_status"], trial["stages"], trial["complete_stage_success"],
                             _canonical(trial["continuous_metrics"]))
                if signature != previous["signature"]:
                    raise ValueError("Duplicate summary has conflicting published admissions")
                duplicates.append({"source_group": group["id"], "trial_id": trial["id"],
                                   "canonical_trial_id": previous["id"], "summary_sha256": sha})
                continue
            if sha is not None:
                seen[sha] = {"identity": identity, "id": trial["id"],
                             "signature": (trial["valid"], trial["native_success"], trial["publication_status"],
                                           trial["staged_evidence_status"], trial["stages"], trial["complete_stage_success"],
                                           _canonical(trial["continuous_metrics"]))}
            retained.append(trial)
        group["trials"] = retained
        group.update(_stats(retained))
        group["metrics"] = _metric_means(retained)
    return [group for group in groups if group["trials"]], duplicates


def _counts(groups):
    trials = [trial for group in groups for trial in group["trials"]]
    staged = sum(trial["valid"] is True and trial["staged_evidence_status"] == "valid" for trial in trials)
    counts = {"members": len(trials), "eligible": sum(trial["valid"] is True for trial in trials),
              "quarantined": sum(trial["publication_status"] == "quarantined_replay_only" for trial in trials),
              "staged_verified": staged, "staged_unknown": len(trials) - staged}
    counts.update(groups=len(groups), tasks=len({group["task"] for group in groups}),
                  models=len({group["model"] for group in groups}),
                  invalid=sum(trial["valid"] is False and trial["publication_status"] != "quarantined_replay_only" for trial in trials))
    return counts


def build_unified_catalog(h50_snapshot, tactile_snapshot, output):
    """Read published evidence and return one task/model/checkpoint catalog."""
    h50 = Path(h50_snapshot).resolve(strict=True)
    tactile = Path(tactile_snapshot).resolve(strict=True)
    output = Path(output).absolute()
    if h50.parent != tactile.parent or output.parent.resolve(strict=True) != h50.parent:
        raise ValueError("Unified output and source snapshots must share one reports root")
    if output.resolve(strict=False) in (h50, tactile):
        raise ValueError("Unified output cannot overwrite a source snapshot")
    store = _Artifacts(h50.parent, output)
    archived = store.document(h50 / "catalog.json")
    ego = store.document(h50 / "egotouch_catalog.json")
    touch = store.document(tactile / "catalog.json")
    experiment = archived.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("schema") != "kaihand-pi05-h50-checkpoint-experiment-v1":
        raise ValueError("H50 source must be the published checkpoint experiment")
    groups = _archive_groups(store, h50, archived, "pi05_h50", "pi05")
    h50_counts = _counts(groups)
    for key, name in (("members", "member_trials"), ("eligible", "eligible_trials"), ("quarantined", "quarantined_trials")):
        if h50_counts[key] != _int(experiment.get(name), name):
            raise ValueError(f"H50 experiment {name} disagrees with its catalog")
    groups += _archive_groups(store, h50, ego, "egotouch", "egotouch")
    tactile_groups = _tactile_groups(store, tactile, touch)
    expected_touch = touch.get("experiment", {}).get("counts", {})
    if (sum(len(group["trials"]) for group in tactile_groups) != expected_touch.get("planned")
            or sum(group["valid_trials"] for group in tactile_groups) != expected_touch.get("valid")):
        raise ValueError("Tactile experiment totals disagree with its catalog")
    groups += tactile_groups
    groups, duplicates = _deduplicate(groups)
    if len({group["id"] for group in groups}) != len(groups):
        raise ValueError("Duplicate published protocol group ID")
    tasks = {}
    for value in (archived, ego):
        for row in _rows(value.get("tasks"), "published task catalog"):
            tasks.setdefault(row["task"], {"task": row["task"], "title": row["title"],
                                           "note": row.get("note", row.get("task_note", ""))})
    for group in groups:
        tasks.setdefault(group["task"], {"task": group["task"], "title": group["title"], "note": ""})
    old_index = store.target(h50, archived.get("historical_url"))
    old_root = old_index.parent
    old_batch = store.document(old_root / "batch.json", "preserved_historical_summary")
    historical_summary = {"title": "原 360 条 H50 成功率与阶段汇总（历史保留）",
                          "note": "原归档表原样保留，不作为新增试验再次计数；其中隔离记录仅供历史回看，不进入当前成绩。白板拾取阶段不等于完整清洁成功。",
                          "rows": copy.deepcopy(_rows(old_batch.get("stage_groups"), "historical stage groups")),
                          "not_counted_again": True, "source_url": store.href(old_index)}
    links = [{"label": "原 H50 四个 checkpoint · 450 条", "url": store.href(h50 / "index.html")},
             {"label": "原 360 条完整工作台", "url": store.href(old_index)},
             {"label": "EgoTouch · 120 条", "url": store.href(h50 / "egotouch.html")},
             {"label": "触觉模型 · 25 条", "url": store.href(tactile / "index.html")}]
    scope = {"published_catalogs_and_linked_artifacts_verified": True,
             "retired_original_sources_verified": False, "live_model_weights_verified": False,
             "native_outcomes_rescored": False, "staged_rules_reexecuted": False,
             "simulation_runs_started": 0, "source_writes": 0,
             "semantics": "Published admissions and recorded rules are preserved. Snapshot-byte verification is not renewed simulator/source/model certification."}
    return {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "title": "KaiTactileSim 官方评估平台", "tasks": list(tasks.values()), "groups": groups,
            "counts": _counts(groups), "historical_summary": historical_summary, "links": links,
            "duplicates": duplicates, "verification_scope": scope,
            "provenance": sorted(store.inputs.values(), key=lambda row: row["path"])}


def _validate_catalog(catalog):
    if catalog.get("schema") != SCHEMA:
        raise ValueError("Unsupported unified workbench catalog")
    groups = _rows(catalog.get("groups"), "unified groups")
    for group in groups:
        expected = {"pi05": (50, 50), "egotouch": (50, 50), "ftp1": (32, 5), "pi05+cpm": (30, 5)}.get(group.get("model"))
        if expected is None or (group.get("prediction_horizon"), group.get("execute_steps")) != expected:
            raise ValueError("Unified model/horizon/execution identity is inconsistent")
        actual = _stats(_rows(group.get("trials"), "unified trials"))
        for key in actual:
            if group.get(key) != actual[key]:
                raise ValueError(f"Unified group {key} is inconsistent")
        for trial in group["trials"]:
            if type(trial.get("valid")) is not bool:
                raise ValueError("Unified validity must be a recorded boolean")
            if trial["valid"] and _bool(trial.get("native_success")) is None:
                raise ValueError("Valid unified native trial needs a recorded boolean result")
            if trial["publication_status"] == "quarantined_replay_only" and trial["valid"]:
                raise ValueError("Quarantine cannot enter unified scores")
    if catalog.get("counts") != _counts(groups):
        raise ValueError("Unified membership counts are inconsistent")
    if catalog.get("historical_summary", {}).get("not_counted_again") is not True:
        raise ValueError("Historical summary cannot be admitted a second time")


def export_unified_workbench(h50_snapshot, tactile_snapshot, output, *, publish_index=False):
    """Write a fresh unified snapshot; publish only after independent verification."""
    from .unified_workbench_page import render_page

    output = Path(output).absolute()
    if output.exists():
        raise ValueError("Unified workbench output must be a fresh directory")
    root_index = output.parent / "index.html"
    previous = root_index.read_bytes() if root_index.exists() else None
    catalog = build_unified_catalog(h50_snapshot, tactile_snapshot, output)
    _validate_catalog(catalog)
    output.mkdir()
    artifacts = []
    _write(output, Path("catalog.json"), encoded(catalog), artifacts)
    for source, target in (
        (PACKAGE.parent / "evaluation_workbench/web/suite.css", "assets/suite.css"),
        (PACKAGE / "web/unified_workbench.js", "assets/unified_workbench.js"),
        (PACKAGE / "web/unified_workbench.css", "assets/unified_workbench.css"),
    ):
        _write(output, Path(target), source.read_bytes(), artifacts)
    page = render_page(catalog)
    _write(output, Path("index.html"), page.encode() if isinstance(page, str) else page, artifacts)
    if previous is not None:
        _write(output, Path("previous_root_index.html"), previous, artifacts)
    verification = {"schema": SNAPSHOT_SCHEMA, "generated_utc": catalog["generated_utc"],
                    "verification_scope": catalog["verification_scope"], "source_files": catalog["provenance"],
                    "source_writes": 0, "existing_snapshot_writes": 0, "simulation_runs_started": 0}
    _write(output, Path("verification.json"), encoded(verification), artifacts)
    _write(output, Path("artifact_manifest.json"), encoded({"schema": SNAPSHOT_SCHEMA, "files": artifacts}), [])
    result = verify_unified_workbench(output)
    if publish_index:
        relative = quote(output.name, safe="._-") + "/index.html"
        landing = ("<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
                   f"<meta http-equiv=\"refresh\" content=\"0; url={relative}\">"
                   "<title>KaiTactileSim 官方评估平台</title>"
                   f"<p><a href=\"{relative}\">进入 KaiTactileSim 统一评估工作台</a></p></html>").encode()
        _publish_portal(output, root_index, previous, landing)
        result = verify_unified_workbench(output)
    return result


def verify_unified_workbench(output):
    """Verify this snapshot and all inventoried publication dependencies only."""
    output = Path(output).resolve(strict=True)
    manifest, _ = _read(output / "artifact_manifest.json")
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported unified workbench snapshot schema")
    for row in _rows(manifest.get("files"), "unified snapshot inventory"):
        relative = Path(row["path"])
        path = output / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.resolve(strict=True).is_relative_to(output):
            raise ValueError("Unsafe unified snapshot artifact")
        verify_row(row, path)
    verification, _ = _read(output / "verification.json")
    if verification.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported unified verification schema")
    for row in _rows(verification.get("source_files"), "unified source artifact inventory"):
        path = Path(row["path"])
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(output.parent):
            raise ValueError("Unified verification cannot read sources outside published reports")
        verify_row(row, path)
    catalog, _ = _read(output / "catalog.json")
    _validate_catalog(catalog)
    if (output / "portal_publication.pending.json").exists():
        raise ValueError("Incomplete unified root publication")
    current = None
    audit_path = output / "portal_publication.json"
    if audit_path.exists():
        audit, _ = _read(audit_path)
        root = output.parent / "index.html"
        if audit.get("state") != "committed" or audit.get("root_index") != str(root):
            raise ValueError("Invalid unified root publication audit")
        current = root.is_file() and digest(root)["sha256"] == audit.get("new_sha256")
    return {"status": "passed", **catalog["counts"], "artifacts": len(manifest["files"]),
            "dependencies": len(verification["source_files"]), "root_portal_matches_this_snapshot": current,
            "retired_original_sources_verified": False, "native_outcomes_rescored": False}


__all__ = ["SCHEMA", "SNAPSHOT_SCHEMA", "build_unified_catalog", "export_unified_workbench", "verify_unified_workbench"]
