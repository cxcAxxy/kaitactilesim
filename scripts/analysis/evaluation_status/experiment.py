"""Compose the four-checkpoint pi05 H50 experiment without rescoring evidence.

The 430 eligible members are copied from a verified evaluation-status snapshot.
The separately archived, failed-publication whiteboard-line cohort is exposed as
20 replay-only members.  Its recorded native facts remain visible, but it never
contributes to validity, native-success, or staged denominators.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

SCHEMA = "kaihand-evaluation-status-v1"
EXPERIMENT_SCHEMA = "kaihand-pi05-h50-checkpoint-experiment-v1"
MAX_METADATA_BYTES = 16 * 1024 * 1024
CHECKPOINT_COUNTS = {"10000": 5, "15000": 5, "20000": 20, "24999": 20}
TASKS = (
  "poker-draw",
  "usb-insert",
  "bulb-screw",
  "install-ram",
  "whiteboard-wipe",
  "usb-unplug",
  "whiteboard-wipe-horizontal",
  "whiteboard-wipe-horizontal-line",
  "bulb-unscrew",
)
QUARANTINED_TASK = "whiteboard-wipe-horizontal-line"
QUARANTINED_CHECKPOINT = "20000"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _pairs(rows):
  value = {}
  for key, item in rows:
    if key in value:
      raise ValueError(f"duplicate JSON key: {key}")
    value[key] = item
  return value


def _constant(value):
  raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
  path = path.absolute()
  if path.is_symlink():
    raise ValueError(f"metadata leaf must not be a symlink: {path}")
  before = path.stat()
  if not path.is_file() or not 0 < before.st_size <= MAX_METADATA_BYTES:
    raise ValueError(f"metadata must be a bounded nonempty regular file: {path}")
  raw = path.read_bytes()
  after = path.stat()
  if (
    len(raw) != before.st_size
    or (before.st_ino, before.st_size, before.st_mtime_ns)
    != (after.st_ino, after.st_size, after.st_mtime_ns)
  ):
    raise ValueError(f"metadata changed while reading: {path}")
  value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value, {
    "path": str(path),
    "size": len(raw),
    "sha256": hashlib.sha256(raw).hexdigest(),
  }


def _sha(value: object) -> bool:
  return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _artifact_row(row: object, *, label: str) -> dict[str, Any]:
  if not isinstance(row, Mapping):
    raise ValueError(f"{label} artifact row is not an object")
  size = row.get("size")
  sha256 = row.get("sha256")
  if type(size) is not int or size <= 0 or not _sha(sha256):
    raise ValueError(f"{label} artifact identity is invalid")
  return {"size": size, "sha256": sha256}


def _status_artifacts(manifest: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
  rows = manifest.get("files")
  if manifest.get("schema") != "kaihand-evaluation-status-snapshot-v1" or not isinstance(rows, list):
    raise ValueError("unsupported status snapshot artifact manifest")
  result = {}
  for source in rows:
    identity = _artifact_row(source, label="status snapshot")
    relative = source.get("path") if isinstance(source, Mapping) else None
    if not isinstance(relative, str) or not relative:
      raise ValueError("status artifact path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or relative in result:
      raise ValueError(f"unsafe or duplicate status artifact path: {relative!r}")
    result[relative] = {
      **identity,
      "path": str(root / Path(*pure.parts)),
      "relative_path": relative,
      "snapshot": "status_snapshot",
    }
  return result


def _historical_artifacts(
  manifest: Mapping[str, Any],
  root: Path,
) -> dict[str, dict[str, Any]]:
  rows = manifest.get("files")
  if not isinstance(rows, list):
    raise ValueError("historical artifact manifest has no file inventory")
  result = {}
  for source in rows:
    identity = _artifact_row(source, label="historical snapshot")
    recorded = source.get("path") if isinstance(source, Mapping) else None
    if not isinstance(recorded, str) or not recorded:
      raise ValueError("historical artifact path is invalid")
    pure = PurePosixPath(recorded)
    occurrences = [index for index, part in enumerate(pure.parts) if part == root.name]
    if len(occurrences) != 1:
      raise ValueError(f"cannot bind historical artifact root: {recorded}")
    relative = PurePosixPath(*pure.parts[occurrences[0] + 1 :])
    if not relative.parts or ".." in relative.parts:
      raise ValueError(f"unsafe historical artifact path: {recorded}")
    key = relative.as_posix()
    if key in result:
      raise ValueError(f"duplicate historical artifact path: {key}")
    result[key] = {
      **identity,
      "path": str(root / Path(*relative.parts)),
      "recorded_path": recorded,
      "relative_path": key,
      "snapshot": "historical_report",
    }
  return result


def _verify_metadata(identity: Mapping[str, Any], artifact: Mapping[str, Any], label: str) -> None:
  if identity["size"] != artifact["size"] or identity["sha256"] != artifact["sha256"]:
    raise ValueError(f"{label} differs from its snapshot artifact manifest")


def _dependency(
  rows: Mapping[str, Mapping[str, Any]],
  relative: str,
  role: str,
) -> dict[str, Any]:
  row = rows.get(relative)
  if row is None:
    raise ValueError(f"required snapshot artifact is not inventoried: {relative}")
  path = Path(row["path"])
  if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"]:
    raise ValueError(f"required snapshot artifact is missing or changed in size: {path}")
  return {**row, "role": role}


def _add_dependency(
  dependencies: dict[str, dict[str, Any]],
  row: Mapping[str, Any],
) -> None:
  path = str(row["path"])
  previous = dependencies.get(path)
  if previous is not None and (
    previous["size"] != row["size"] or previous["sha256"] != row["sha256"]
  ):
    raise ValueError(f"conflicting dependency identity: {path}")
  dependencies.setdefault(path, dict(row))


def _metadata_row(identity: Mapping[str, Any], role: str) -> dict[str, Any]:
  return {**identity, "role": role}


def _normalized_model(value: object) -> str | None:
  if not isinstance(value, str):
    return None
  return {"pi0.5": "pi05", "pi05": "pi05"}.get(value.strip().lower())


def _safe_relative(value: object, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a nonempty relative path")
  pure = PurePosixPath(value)
  if pure.is_absolute() or ".." in pure.parts:
    raise ValueError(f"unsafe {label}: {value!r}")
  return pure.as_posix()


def _relocate_detail(
  value: object,
  *,
  status_name: str,
  historical_name: str,
  status_artifacts: Mapping[str, Mapping[str, Any]],
  historical_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
  if not isinstance(value, str) or not value:
    raise ValueError("eligible member has no detail URL")
  pure = PurePosixPath(value)
  if pure.is_absolute():
    raise ValueError(f"eligible detail URL is absolute: {value}")
  if pure.parts and pure.parts[0] == "trials" and ".." not in pure.parts:
    relative = pure.as_posix()
    return (
      f"../{quote(status_name, safe='._-')}/{relative}",
      _dependency(status_artifacts, relative, "eligible_member_detail"),
    )
  if len(pure.parts) >= 3 and pure.parts[:2] == ("..", historical_name):
    relative = PurePosixPath(*pure.parts[2:]).as_posix()
    return value, _dependency(historical_artifacts, relative, "eligible_member_detail")
  raise ValueError(f"eligible detail URL is outside its bound snapshots: {value}")


def _summary_binding(source: Mapping[str, Any]) -> dict[str, Any]:
  provenance = source.get("provenance")
  if not isinstance(provenance, list):
    raise ValueError("historical source member has no provenance")
  rows = [
    row
    for row in provenance
    if isinstance(row, Mapping)
    and row.get("role") == "prediction_horizon_summary"
    and PurePosixPath(str(row.get("path", ""))).name == "summary.json"
  ]
  if len(rows) != 1:
    raise ValueError("historical member must bind exactly one prediction summary")
  identity = _artifact_row(rows[0], label="historical summary")
  path = rows[0].get("path")
  if not isinstance(path, str) or not path:
    raise ValueError("historical summary path is missing")
  return {**identity, "path": path, "role": "prediction_horizon_summary"}


def _cohort_protocol(trial: Mapping[str, Any]) -> dict[str, Any]:
  cohort = trial.get("cohort")
  staged = trial.get("staged")
  if not isinstance(cohort, Mapping) or not isinstance(staged, Mapping):
    raise ValueError("quarantined member has no recorded cohort/staged identity")
  rule = staged.get("rule_identity")
  rule_sha = rule.get("sha256") if isinstance(rule, Mapping) else None
  if not _sha(rule_sha):
    raise ValueError("quarantined member has no recorded staged rule SHA-256")
  return {
    "max_sim_seconds": None,
    "guard_mode": cohort.get("guard_mode") if isinstance(cohort.get("guard_mode"), str) else None,
    "controller": cohort.get("controller") if isinstance(cohort.get("controller"), str) else None,
    "action_interpolation": None,
    "control_hz": float(cohort["control_hz"])
    if type(cohort.get("control_hz")) in (int, float) and math.isfinite(cohort["control_hz"])
    else None,
    "success_hold": None,
    "completion_protocol_recorded": False,
    "completion_protocol": None,
    "rule_sha256": rule_sha,
    "validity_basis": ["quarantined_failed_source_group_not_eligible"],
  }


def build_experiment_catalog(
  status_snapshot: str | Path,
  historical_report: str | Path,
) -> dict[str, Any]:
  """Return the 450-member pi05 H50 experiment catalog.

  This only reads bounded snapshot metadata and stats selected artifact leaves.
  Artifact content dependencies are returned for the publisher to hash.
  """

  status_root = Path(status_snapshot).expanduser().absolute()
  historical_root = Path(historical_report).expanduser().absolute()
  if not status_root.is_dir() or not historical_root.is_dir():
    raise ValueError("status_snapshot and historical_report must be existing directories")

  status_manifest, status_manifest_id = _read_json(status_root / "artifact_manifest.json")
  status_rows = _status_artifacts(status_manifest, status_root)
  status_catalog, status_catalog_id = _read_json(status_root / "catalog.json")
  status_verification, status_verification_id = _read_json(status_root / "verification.json")
  _verify_metadata(status_catalog_id, status_rows["catalog.json"], "status catalog")
  _verify_metadata(status_verification_id, status_rows["verification.json"], "status verification")
  if status_catalog.get("schema") != SCHEMA:
    raise ValueError("unsupported status catalog schema")
  if (
    status_verification.get("schema") != "kaihand-evaluation-status-snapshot-v1"
    or status_verification.get("status") != "passed"
  ):
    raise ValueError("parent status snapshot is not verified")

  historical_manifest, historical_manifest_id = _read_json(
    historical_root / "artifact_manifest.json"
  )
  historical_rows = _historical_artifacts(historical_manifest, historical_root)
  historical_batch, historical_batch_id = _read_json(historical_root / "batch.json")
  historical_sources, historical_sources_id = _read_json(
    historical_root / "source_manifest.json"
  )
  _verify_metadata(historical_batch_id, historical_rows["batch.json"], "historical batch")
  _verify_metadata(
    historical_sources_id,
    historical_rows["source_manifest.json"],
    "historical source manifest",
  )
  if historical_batch.get("schema_version") != "evaluation-workbench-suite-v1":
    raise ValueError("unsupported historical workbench batch")

  dependencies: dict[str, dict[str, Any]] = {}
  groups = []
  summary_shas = set()
  checkpoint_parents: dict[str, set[str]] = {task: set() for task in TASKS}
  checkpoint_paths: dict[str, dict[str, str]] = {task: {} for task in TASKS}
  checkpoint_identities: dict[str, dict[str, str]] = {task: {} for task in TASKS}
  existing_pairs = set()

  source_groups = status_catalog.get("groups")
  if not isinstance(source_groups, list):
    raise ValueError("status catalog has no groups")
  for source_group in source_groups:
    if not isinstance(source_group, Mapping) or _normalized_model(source_group.get("model")) != "pi05":
      continue
    group = copy.deepcopy(dict(source_group))
    task = group.get("task")
    checkpoint = str(group.get("checkpoint"))
    if task not in TASKS or checkpoint not in CHECKPOINT_COUNTS:
      raise ValueError(f"unexpected pi05 experiment group: {task}/{checkpoint}")
    pair = (task, checkpoint)
    if pair in existing_pairs:
      raise ValueError(f"protocol-split group cannot be merged: {task}/{checkpoint}")
    existing_pairs.add(pair)
    if group.get("prediction_horizon") != 50 or group.get("execute_steps") != 50:
      raise ValueError(f"non-H50/K50 group in experiment: {task}/{checkpoint}")
    trials = group.get("trials")
    expected_count = CHECKPOINT_COUNTS[checkpoint]
    if not isinstance(trials, list) or len(trials) != expected_count:
      raise ValueError(f"unexpected member count for {task}/{checkpoint}")
    if (
      group.get("valid_trials") != expected_count
      or group.get("invalid_trials") != 0
      or any(not isinstance(trial, Mapping) or trial.get("valid") is not True for trial in trials)
    ):
      raise ValueError(f"eligible source group is not wholly valid: {task}/{checkpoint}")
    checkpoint_path = group.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or PurePosixPath(checkpoint_path).name != checkpoint:
      raise ValueError(f"checkpoint path does not bind its label: {task}/{checkpoint}")
    parent = str(PurePosixPath(checkpoint_path).parent)
    checkpoint_parents[task].add(parent)
    checkpoint_paths[task][checkpoint] = checkpoint_path
    checkpoint_sha = group.get("checkpoint_sha256")
    if not _sha(checkpoint_sha):
      raise ValueError(f"missing checkpoint identity: {task}/{checkpoint}")
    checkpoint_identities[task][checkpoint] = checkpoint_sha

    group["publication_status"] = "eligible"
    for trial in trials:
      summary_sha = trial.get("summary_sha256")
      if not _sha(summary_sha) or summary_sha in summary_shas:
        raise ValueError("eligible members have missing or duplicate summary SHA-256")
      summary_shas.add(summary_sha)
      relocated, dependency = _relocate_detail(
        trial.get("detail_url"),
        status_name=status_root.name,
        historical_name=historical_root.name,
        status_artifacts=status_rows,
        historical_artifacts=historical_rows,
      )
      trial["detail_url"] = relocated
      trial["publication_status"] = "eligible"
      _add_dependency(dependencies, dependency)
    groups.append(group)

  expected_existing = {
    (task, checkpoint)
    for task in TASKS
    for checkpoint in CHECKPOINT_COUNTS
    if (task, checkpoint) != (QUARANTINED_TASK, QUARANTINED_CHECKPOINT)
  }
  if existing_pairs != expected_existing or sum(len(group["trials"]) for group in groups) != 430:
    raise ValueError("parent status snapshot is not the expected 430-member pi05 experiment")
  for task, parents in checkpoint_parents.items():
    if len(parents) != 1:
      raise ValueError(f"checkpoints do not share one training-run parent for {task}")

  batch_trials = historical_batch.get("trials")
  source_trials = historical_sources.get("trials")
  if not isinstance(batch_trials, list) or not isinstance(source_trials, list):
    raise ValueError("historical snapshot has no trial inventories")
  source_by_id = {}
  for row in source_trials:
    trial_id = row.get("trial_id") if isinstance(row, Mapping) else None
    if not isinstance(trial_id, str) or not trial_id or trial_id in source_by_id:
      raise ValueError("historical source trial ids are missing or duplicate")
    source_by_id[trial_id] = row
  quarantined = [
    row
    for row in batch_trials
    if isinstance(row, Mapping)
    and row.get("task") == QUARANTINED_TASK
    and str(row.get("checkpoint")) == QUARANTINED_CHECKPOINT
    and _normalized_model(row.get("model")) == "pi05"
    and row.get("prediction_horizon") == 50
  ]
  if len(quarantined) != 20:
    raise ValueError("historical snapshot does not contain exactly 20 quarantined members")

  quarantine_trials = []
  quarantine_protocol = None
  quarantine_checkpoint_sha = None
  recorded_native_successes = 0
  for batch_trial in quarantined:
    trial_id = batch_trial.get("trial_id")
    source = source_by_id.get(trial_id)
    if not isinstance(source, Mapping):
      raise ValueError(f"historical source manifest has no trial-id match: {trial_id}")
    if source.get("source") != batch_trial.get("source"):
      raise ValueError(f"historical batch/source path mismatch: {trial_id}")
    binding = _summary_binding(source)
    summary_sha = binding["sha256"]
    if summary_sha in summary_shas:
      raise ValueError("quarantined summary SHA-256 is missing or duplicates another member")
    summary_shas.add(summary_sha)

    cohort = batch_trial.get("cohort")
    staged = batch_trial.get("staged")
    if not isinstance(cohort, Mapping) or cohort.get("execute_steps") != 50:
      raise ValueError(f"quarantined member is not H50/K50: {trial_id}")
    if (
      cohort.get("diagnostic_only") is not True
      or cohort.get("formal_metrics_valid") is not False
      or _normalized_model(cohort.get("model_family")) != "pi05"
    ):
      raise ValueError(f"quarantined member has incompatible mode/model: {trial_id}")
    selection = staged.get("source_selection") if isinstance(staged, Mapping) else None
    if not isinstance(selection, Mapping) or selection.get("group_status") != "failed":
      raise ValueError(f"quarantined member is not bound to the failed source group: {trial_id}")
    protocol = _cohort_protocol(batch_trial)
    if quarantine_protocol is None:
      quarantine_protocol = protocol
    elif quarantine_protocol != protocol:
      raise ValueError("quarantined members do not share one recorded protocol")
    checkpoint_sha = cohort.get("checkpoint")
    if not _sha(checkpoint_sha):
      raise ValueError(f"quarantined member has no checkpoint SHA-256: {trial_id}")
    if quarantine_checkpoint_sha is None:
      quarantine_checkpoint_sha = checkpoint_sha
    elif checkpoint_sha != quarantine_checkpoint_sha:
      raise ValueError("quarantined members do not share one checkpoint identity")

    report_relative = _safe_relative(batch_trial.get("report_path"), "historical report path")
    report_dependency = _dependency(
      historical_rows,
      report_relative,
      "quarantined_member_detail",
    )
    video_relative = (
      PurePosixPath(report_relative).parent / "media" / "rollout.mp4"
    ).as_posix()
    video_dependency = _dependency(
      historical_rows,
      video_relative,
      "quarantined_member_video",
    )
    _add_dependency(dependencies, report_dependency)
    _add_dependency(dependencies, video_dependency)
    native = staged.get("native_result") if isinstance(staged, Mapping) else None
    if not isinstance(native, Mapping) or type(native.get("task_success")) is not bool:
      raise ValueError(f"quarantined member has no typed recorded native result: {trial_id}")
    recorded_native_successes += int(native["task_success"])
    warning = (
      "Replay-only quarantined member: its source group failed publication validation. "
      "Recorded native facts are retained separately, but this member contributes to no "
      "valid, native-success, or staged denominator."
    )
    member_id = f"quarantine-{summary_sha[:24]}"
    quarantine_trials.append({
      "id": member_id,
      "seed": batch_trial.get("seed") if type(batch_trial.get("seed")) is int else None,
      "status": "quarantined_replay_only",
      "publication_status": "quarantined_replay_only",
      "valid": False,
      "native_success": None,
      "recorded_native_success": native["task_success"],
      "recorded_native_status": native.get("status"),
      "recorded_native_result": copy.deepcopy(dict(native)),
      "recorded_outcome": batch_trial.get("outcome"),
      "sim_seconds": None,
      "summary_path": binding["path"],
      "summary_sha256": summary_sha,
      "summary_binding": binding,
      "video_path": None,
      "source": source.get("source"),
      "staged_result": None,
      "mode": "diagnostic",
      "detail_url": None,
      "historical_detail_url": (
        f"../{quote(historical_root.name, safe='._-')}/{report_relative}"
      ),
      "historical_video_url": (
        f"../{quote(historical_root.name, safe='._-')}/{video_relative}"
      ),
      "historical_report_binding": {
        "size": report_dependency["size"],
        "sha256": report_dependency["sha256"],
        "relative_path": report_relative,
      },
      "historical_video_binding": {
        "size": video_dependency["size"],
        "sha256": video_dependency["sha256"],
        "relative_path": video_relative,
      },
      "physics_valid": None,
      "video_completed": None,
      "native_source": None,
      "validity_basis": "quarantined_failed_source_group_not_eligible",
      "control_protocol": {
        key: protocol[key]
        for key in (
          "max_sim_seconds",
          "guard_mode",
          "controller",
          "action_interpolation",
          "control_hz",
          "success_hold",
        )
      },
      "warnings": [warning],
      "source_kind": "archive",
      "batch_id": "historical_failed_group_quarantine",
    })

  quarantine_trials.sort(key=lambda row: (row["seed"] is None, row["seed"], row["summary_sha256"]))
  title = quarantined[0].get("title")
  if not isinstance(title, str) or not title:
    raise ValueError("quarantined group has no task title")
  inferred_parent = next(iter(checkpoint_parents[QUARANTINED_TASK]))
  inferred_path = str(PurePosixPath(inferred_parent) / QUARANTINED_CHECKPOINT)
  checkpoint_paths[QUARANTINED_TASK][QUARANTINED_CHECKPOINT] = inferred_path
  checkpoint_identities[QUARANTINED_TASK][QUARANTINED_CHECKPOINT] = quarantine_checkpoint_sha
  quarantine_warning = (
    "This 20-member cohort is replay-only and quarantined because its source group failed "
    "publication validation. Recorded native results are shown separately; scoring and staged "
    "denominators are zero."
  )
  quarantine_id = hashlib.sha256(
    f"{QUARANTINED_TASK}:{QUARANTINED_CHECKPOINT}:{quarantine_checkpoint_sha}:quarantine".encode()
  ).hexdigest()[:24]
  groups.append({
    "id": f"group-{quarantine_id}",
    "task": QUARANTINED_TASK,
    "title": title,
    "model": "pi05",
    "checkpoint": QUARANTINED_CHECKPOINT,
    "checkpoint_path": inferred_path,
    "checkpoint_path_basis": "inferred_from_same-task_shared_training-run_parent",
    "checkpoint_path_independently_verified": False,
    "checkpoint_sha256": quarantine_checkpoint_sha,
    "prediction_horizon": 50,
    "execute_steps": 50,
    "batch_id": "historical_failed_group_quarantine",
    "source_kind": "archive",
    "mode": "diagnostic",
    "publication_status": "quarantined_replay_only",
    "protocol": quarantine_protocol,
    "planned_trials": 20,
    "valid_trials": 0,
    "invalid_trials": 20,
    "native_success_count": 0,
    "native_known_count": 0,
    "recorded_native_success_count": recorded_native_successes,
    "recorded_native_known_count": 20,
    "stages": [],
    "complete_stage_success_count": 0,
    "complete_stage_known_count": 0,
    "warnings": [quarantine_warning],
    "trials": quarantine_trials,
  })

  task_order = {task: index for index, task in enumerate(TASKS)}
  checkpoint_order = {key: index for index, key in enumerate(CHECKPOINT_COUNTS)}
  groups.sort(key=lambda row: (task_order[row["task"]], checkpoint_order[row["checkpoint"]]))
  pairs = {(group["task"], group["checkpoint"]) for group in groups}
  expected_pairs = {(task, checkpoint) for task in TASKS for checkpoint in CHECKPOINT_COUNTS}
  if len(groups) != 36 or pairs != expected_pairs:
    raise ValueError("experiment does not contain exactly 9 tasks x 4 checkpoints")
  if len(summary_shas) != 450 or sum(len(group["trials"]) for group in groups) != 450:
    raise ValueError("experiment does not contain 450 unique summary-bound members")

  source_tasks = status_catalog.get("tasks")
  if not isinstance(source_tasks, list):
    raise ValueError("status catalog has no task metadata")
  task_metadata = {
    row.get("task"): copy.deepcopy(dict(row))
    for row in source_tasks
    if isinstance(row, Mapping) and row.get("task") in TASKS
  }
  if set(task_metadata) != set(TASKS):
    raise ValueError("status catalog does not describe all nine experiment tasks")
  tasks = [task_metadata[task] for task in TASKS]

  lineage = []
  for task in TASKS:
    paths = checkpoint_paths[task]
    if set(paths) != set(CHECKPOINT_COUNTS):
      raise ValueError(f"incomplete checkpoint-path evidence for {task}")
    inferred = task == QUARANTINED_TASK
    lineage.append({
      "task": task,
      "checkpoint_parent": next(iter(checkpoint_parents[task])),
      "checkpoint_paths": dict(sorted(paths.items(), key=lambda row: checkpoint_order[row[0]])),
      "checkpoint_sha256": dict(
        sorted(checkpoint_identities[task].items(), key=lambda row: checkpoint_order[row[0]])
      ),
      "relationship_status": (
        "shared_parent_for_three_recorded_paths; step20000 path inferred and its SHA is bound "
        "by historical metadata"
        if inferred
        else "all_four_recorded_checkpoint_paths_share_one_training-run parent"
      ),
      "all_checkpoint_paths_independently_recorded": not inferred,
    })

  provenance = [
    _metadata_row(status_catalog_id, "parent_status_catalog"),
    _metadata_row(status_verification_id, "parent_status_verification"),
    _metadata_row(status_manifest_id, "parent_status_artifact_manifest"),
    _metadata_row(historical_batch_id, "historical_workbench_batch"),
    _metadata_row(historical_sources_id, "historical_source_manifest"),
    _metadata_row(historical_manifest_id, "historical_artifact_manifest"),
  ]
  experiment = {
    "schema": EXPERIMENT_SCHEMA,
    "id": "pi05-h50-k50-checkpoints-10000-15000-20000-24999",
    "status": "430_eligible_plus_20_quarantined_replay_only",
    "model": "pi05",
    "prediction_horizon": 50,
    "execute_steps": 50,
    "task_count": 9,
    "group_count": 36,
    "checkpoint_labels": list(CHECKPOINT_COUNTS),
    "member_trials": 450,
    "eligible_trials": 430,
    "quarantined_trials": 20,
    "score_denominator": 430,
    "checkpoint_member_counts": {"10000": 45, "15000": 45, "20000": 180, "24999": 180},
    "checkpoint_eligible_counts": {"10000": 45, "15000": 45, "20000": 160, "24999": 180},
    "training_lineage": lineage,
    "semantics": (
      "One user-declared pi05 H50/K50 checkpoint experiment. Metrics remain per checkpoint and "
      "per recorded protocol; native and staged results are never pooled. The 20 quarantined "
      "members are replay-only and excluded from every score denominator. EgoTouch is excluded."
    ),
  }
  return {
    "schema": SCHEMA,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(status_root),
    "tasks": tasks,
    "groups": groups,
    "jobs": [],
    "excluded": [],
    "provenance": provenance,
    "dependencies": [dependencies[path] for path in sorted(dependencies)],
    "historical_url": f"../{quote(historical_root.name, safe='._-')}/index.html",
    "parent_status_url": f"../{quote(status_root.name, safe='._-')}/index.html",
    "experiment": experiment,
    "verification_scope": (
      "Metadata identities are read and checked here. Listed artifact dependencies must be "
      "content-hashed by the publisher; parent snapshots remain immutable."
    ),
  }


__all__ = ["EXPERIMENT_SCHEMA", "build_experiment_catalog"]
