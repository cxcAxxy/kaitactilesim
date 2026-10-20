"""Build a fail-closed, read-only catalog of recorded evaluation outcomes.

The canonical archive remains authoritative for published cohorts. Completed
new-style jobs are additive; legacy, interrupted, superseded, and
normalization-only jobs are described but never promoted to trials. No model,
simulator, frame log, or weight file is imported or read here. Staged sidecars
are delegated to the existing strict archived-rule loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation_unified.staged import load_staged_result
from evaluation_workbench.task_fields import task_presentation
from kaihand_tactile_env.pipeline.catalog import TASKS

SCHEMA = "kaihand-evaluation-status-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_GROUP_SUMMARY_JSON_BYTES = 64 * 1024 * 1024
MAX_JOBS = 256
MAX_GROUPS = 2_048
MAX_TRIALS = 20_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_STATUSES = frozenset({"success", "task_not_completed", "policy_action_rejected"})


def _pairs(rows):
  value = {}
  for key, item in rows:
    if key in value:
      raise ValueError(f"duplicate JSON key: {key}")
    value[key] = item
  return value


def _constant(value):
  raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(
  path: Path,
  *,
  max_bytes: int = MAX_JSON_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
  path = Path(path).absolute()
  if path.is_symlink():
    raise ValueError(f"JSON leaf must not be a symlink: {path}")
  before = path.stat()
  if not path.is_file() or not 0 < before.st_size <= max_bytes:
    raise ValueError(f"JSON must be a bounded nonempty regular file: {path}")
  raw = path.read_bytes()
  after = path.stat()
  if (
    len(raw) != before.st_size
    or (before.st_ino, before.st_size, before.st_mtime_ns)
    != (after.st_ino, after.st_size, after.st_mtime_ns)
  ):
    raise ValueError(f"source changed while reading: {path}")
  value = json.loads(
    raw,
    object_pairs_hook=_pairs,
    parse_constant=_constant,
  )
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value, {
    "path": str(path),
    "size": len(raw),
    "sha256": hashlib.sha256(raw).hexdigest(),
  }


def _safe_relative(root: Path, value: object) -> Path:
  if not isinstance(value, str) or not value:
    raise ValueError("relative artifact path must be a nonempty string")
  relative = PurePosixPath(value)
  if relative.is_absolute() or ".." in relative.parts:
    raise ValueError(f"unsafe relative artifact path: {value!r}")
  path = (root / Path(*relative.parts)).absolute()
  if not path.is_relative_to(root.absolute()):
    raise ValueError(f"artifact escapes source root: {value!r}")
  return path


def _sha(value: object) -> bool:
  return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _integer(value: object) -> int | None:
  return value if type(value) is int else None


def _number(value: object) -> float | None:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  result = float(value)
  return result if math.isfinite(result) else None


def _mode(payload: Mapping[str, Any]) -> str:
  diagnostic = payload.get("diagnostic_only")
  formal = payload.get("formal_metrics_valid")
  if diagnostic is True and formal is False:
    return "diagnostic"
  if diagnostic is False and formal is True:
    return "formal"
  return "unverified"


def _model_family(value: object) -> str | None:
  if not isinstance(value, str) or not value.strip():
    return None
  normalized = value.strip().lower().replace("_", "-")
  aliases = {
    "pi0.5": "pi05",
    "pi0-5": "pi05",
    "pi05": "pi05",
    "egotouch": "egotouch",
    "ftp-1": "ftp1",
    "ftp1": "ftp1",
    "pi05+cpm": "pi05+cpm",
  }
  return aliases.get(normalized, normalized)


def _control_protocol(summary: Mapping[str, Any]) -> dict[str, Any]:
  success_hold = None
  for key in ("success_hold", "success_hold_s", "success_hold_seconds"):
    if key in summary:
      success_hold = _number(summary[key])
      break
  return {
    "max_sim_seconds": _number(summary.get("max_sim_seconds")),
    "guard_mode": summary.get("guard_mode") if isinstance(summary.get("guard_mode"), str) else None,
    "controller": summary.get("controller") if isinstance(summary.get("controller"), str) else None,
    "action_interpolation": (
      summary.get("action_interpolation")
      if isinstance(summary.get("action_interpolation"), str)
      else None
    ),
    "control_hz": _number(summary.get("control_hz")),
    "success_hold": success_hold,
  }


def _native(payload: Mapping[str, Any]) -> tuple[bool | None, str | None]:
  value = payload.get("task_success")
  if type(value) is bool:
    return value, "task_success"
  evaluation = payload.get("evaluation")
  if isinstance(evaluation, Mapping) and type(evaluation.get("success")) is bool:
    return evaluation["success"], "evaluation.success"
  return None, None


def _completion_protocol(
  summary: Mapping[str, Any],
  staged: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
  if staged is not None and "completion_protocol" in staged:
    value = staged["completion_protocol"]
    return True, value if isinstance(value, str) else None
  for container in (
    summary,
    summary.get("evaluation"),
    summary.get("staged_evaluation"),
  ):
    if isinstance(container, Mapping) and "completion_protocol" in container:
      value = container["completion_protocol"]
      return True, value if isinstance(value, str) else None
  transfer = summary.get("completion_protocol_transfer")
  if isinstance(transfer, Mapping) and "evaluation_completion_protocol" in transfer:
    value = transfer["evaluation_completion_protocol"]
    return True, value if isinstance(value, str) else None
  return False, None


def _no_recorded_errors(summary: Mapping[str, Any]) -> bool:
  for key in ("error", "record_error", "task_error", "errors"):
    value = summary.get(key)
    if value not in (None, "", []):
      return False
  video = summary.get("video")
  return not isinstance(video, Mapping) or video.get("task_error") in (None, "")


class _Builder:
  def __init__(self, root: Path):
    self.root = root.absolute()
    self.groups: list[dict[str, Any]] = []
    self.jobs: list[dict[str, Any]] = []
    self.excluded: list[dict[str, Any]] = []
    self._provenance: dict[str, dict[str, Any]] = {}
    self._fingerprints: dict[tuple[str, str | None], str] = {}
    self.trial_count = 0

  def provenance(self, row: Mapping[str, Any], role: str) -> None:
    path = str(row["path"])
    item = {
      "path": path,
      "size": int(row["size"]),
      "sha256": str(row["sha256"]),
      "role": role,
    }
    previous = self._provenance.get(path)
    if previous is not None and (
      previous["size"] != item["size"]
      or previous["sha256"] != item["sha256"]
    ):
      raise ValueError(f"source identity changed within catalog build: {path}")
    self._provenance.setdefault(path, item)

  def exclude(self, source: object, reason: str, **extra: Any) -> None:
    self.excluded.append({"source": str(source), "reason": reason, **extra})

  def register(self, trial: dict[str, Any]) -> bool:
    self.trial_count += 1
    if self.trial_count > MAX_TRIALS:
      raise ValueError("evaluation trial bound exceeded")
    fingerprint = trial.pop("_fingerprint")
    source = trial["source"]
    canonical = self._fingerprints.get(fingerprint)
    if canonical is not None:
      self.exclude(
        source,
        "duplicate summary and staged-sidecar content; canonical source retained",
        canonical_source=canonical,
      )
      return False
    self._fingerprints[fingerprint] = source
    return True

  @property
  def provenance_rows(self) -> list[dict[str, Any]]:
    return [self._provenance[key] for key in sorted(self._provenance)]


def _expected_file(
  manifest_files: Mapping[str, Mapping[str, Any]],
  group_root: Path,
  path: Path,
) -> Mapping[str, Any] | None:
  try:
    relative = path.relative_to(group_root).as_posix()
  except ValueError:
    return None
  return manifest_files.get(relative)


def _validate_expected(
  row: Mapping[str, Any],
  expected: Mapping[str, Any] | None,
  label: str,
) -> None:
  if expected is None:
    return
  if expected.get("sha256") != row["sha256"] or expected.get("size") != row["size"]:
    raise ValueError(f"{label} differs from its archive manifest")


def _regular_video(path: Path) -> bool:
  try:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0
  except OSError:
    return False


def _load_trial(
  builder: _Builder,
  *,
  summary_path: Path,
  video_path: Path,
  source_kind: str,
  batch_id: str,
  expected: Mapping[str, Any],
  seed_declaration: Mapping[str, Any] | None = None,
  legacy_archive: bool = False,
  group_root: Path | None = None,
  manifest_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
  summary, summary_row = _read_json(summary_path)
  expected_summary = (
    _expected_file(manifest_files or {}, group_root, summary_path)
    if group_root is not None
    else None
  )
  _validate_expected(summary_row, expected_summary, "trial summary")
  builder.provenance(summary_row, "trial_summary")

  sidecar_path = summary_path.parent / "staged_evaluation.json"
  sidecar_row = None
  if sidecar_path.exists():
    _, sidecar_row = _read_json(sidecar_path)
    expected_sidecar = (
      _expected_file(manifest_files or {}, group_root, sidecar_path)
      if group_root is not None
      else None
    )
    _validate_expected(sidecar_row, expected_sidecar, "staged sidecar")
    builder.provenance(sidecar_row, "staged_sidecar")

  warnings: list[str] = []
  identity_ok = True
  for key in (
    "task",
    "checkpoint_path",
    "checkpoint_sha256",
    "prediction_horizon",
    "execute_steps",
  ):
    required = expected.get(key)
    if required is not None and (
      summary.get(key) != required or type(summary.get(key)) is not type(required)
    ):
      identity_ok = False
      warnings.append(f"summary identity mismatch: {key}")
  observed_model = _model_family(summary.get("model_family"))
  expected_model = _model_family(expected.get("model"))
  if observed_model is None or expected_model is None or observed_model != expected_model:
    identity_ok = False
    warnings.append("summary identity mismatch: model_family")
  seed = _integer(summary.get("seed"))
  expected_seed = expected.get("seed")
  if seed is None or (expected_seed is not None and seed != expected_seed):
    identity_ok = False
    warnings.append("summary identity mismatch: seed")
  checkpoint_sha = summary.get("checkpoint_sha256")
  if not _sha(checkpoint_sha):
    identity_ok = False
    warnings.append("missing typed checkpoint SHA-256")

  native_success, native_source = _native(summary)
  physics = summary.get("physics_valid")
  physics_valid = physics if type(physics) is bool else None
  mode = _mode(summary)
  video = summary.get("video")
  video_completed = (
    video.get("completed") if isinstance(video, Mapping) and type(video.get("completed")) is bool else None
  )
  error_free = _no_recorded_errors(summary)
  status = summary.get("status") if isinstance(summary.get("status"), str) else "unknown"
  terminal = status in _TERMINAL_STATUSES
  video_present = _regular_video(video_path)

  legacy_valid = (
    legacy_archive
    and physics_valid is None
    and isinstance(seed_declaration, Mapping)
    and seed_declaration.get("native_valid_trial") is True
    and mode in {"formal", "diagnostic"}
  )
  if legacy_valid:
    validity_basis = "legacy_archive_native_valid_declaration"
    warnings.append(
      "Legacy archive has no explicit physics_valid; validity uses the exact "
      "native_valid_trial declaration and does not claim renewed physics verification."
    )
  else:
    validity_basis = "explicit_summary_physics_valid"
  valid = all((
    identity_ok,
    type(native_success) is bool,
    physics_valid is True or legacy_valid,
    video_completed is True,
    error_free,
    terminal,
    video_present,
    mode != "unverified",
  ))
  if not valid:
    if type(native_success) is not bool:
      warnings.append("native success is missing or not a boolean")
    if physics_valid is not True and not legacy_valid:
      warnings.append("physics_valid is not explicitly true")
    if video_completed is not True or not video_present:
      warnings.append("video is missing or not explicitly completed")
    if not error_free:
      warnings.append("summary records an execution or recording error")
    if not terminal:
      warnings.append("summary has no supported terminal status")
    if mode == "unverified":
      warnings.append("diagnostic/formal source labels are not self-consistent")

  staged_result = None
  rule_sha = None
  if sidecar_row is not None:
    try:
      loaded = load_staged_result(summary_path.parent, include_state_log=False)
      staged_result = loaded.get("staged_result")
      if not isinstance(staged_result, dict):
        raise ValueError("strict staged loader returned no projection")
      rule = loaded.get("rule_identity")
      rule_sha = rule.get("sha256") if isinstance(rule, Mapping) else None
      if not _sha(rule_sha):
        raise ValueError("strict staged loader returned no rule SHA-256")
      for row in loaded.get("provenance", []):
        if isinstance(row, Mapping):
          builder.provenance(row, str(row.get("role", "staged_evidence")))
    except (OSError, ValueError, TypeError, KeyError) as error:
      staged_result = None
      rule_sha = None
      warnings.append(f"staged evidence unavailable: {type(error).__name__}: {error}")
      builder.exclude(
        summary_path.parent,
        "strict staged validation failed; native result retained",
        error=f"{type(error).__name__}: {error}",
      )
  elif isinstance(summary.get("staged_evaluation"), Mapping):
    warnings.append("embedded staged payload has no hashed sidecar; staged result unavailable")
    builder.exclude(
      summary_path.parent,
      "embedded staged payload is missing its sidecar; native result retained",
    )

  protocol_recorded, protocol = _completion_protocol(summary, staged_result)
  control_protocol = _control_protocol(summary)
  trial_id = hashlib.sha256(
    f"{summary_row['sha256']}:{seed}:{batch_id}".encode()
  ).hexdigest()[:24]
  return {
    "id": f"trial-{trial_id}",
    "seed": seed,
    "status": status,
    "valid": valid,
    "native_success": native_success,
    "sim_seconds": _number(summary.get("sim_seconds")),
    "summary_path": str(summary_path),
    "summary_sha256": summary_row["sha256"],
    "video_path": str(video_path),
    "source": str(summary_path.parent),
    "staged_result": staged_result,
    "mode": mode,
    "detail_url": None,
    "physics_valid": physics_valid,
    "video_completed": video_completed,
    "native_source": native_source,
    "validity_basis": validity_basis,
    "control_protocol": control_protocol,
    "warnings": warnings,
    "source_kind": source_kind,
    "batch_id": batch_id,
    "checkpoint_sha256": checkpoint_sha if _sha(checkpoint_sha) else None,
    "completion_protocol": protocol,
    "completion_protocol_recorded": protocol_recorded,
    "rule_sha256": rule_sha,
    "staged_sidecar_sha256": sidecar_row["sha256"] if sidecar_row else None,
    "_fingerprint": (
      summary_row["sha256"],
      sidecar_row["sha256"] if sidecar_row else None,
    ),
  }


def _stage_counts(trials: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
  definitions: dict[str, str] = {}
  reached: defaultdict[str, int] = defaultdict(int)
  known: defaultdict[str, int] = defaultdict(int)
  complete_reached = 0
  complete_known = 0
  for trial in trials:
    if not trial["valid"] or not isinstance(trial["staged_result"], Mapping):
      continue
    staged = trial["staged_result"]
    trial_definitions = staged.get("stage_definitions")
    flags = staged.get("stage_flags")
    if not isinstance(trial_definitions, Mapping) or not isinstance(flags, Mapping):
      continue
    for key, name in trial_definitions.items():
      if isinstance(key, str) and isinstance(name, str) and type(flags.get(key)) is bool:
        definitions.setdefault(key, name)
        known[key] += 1
        reached[key] += int(flags[key])
    complete = staged.get("complete_success")
    if type(complete) is bool:
      complete_known += 1
      complete_reached += int(complete)
  return (
    [
      {"key": key, "name": definitions[key], "reached": reached[key], "known": known[key]}
      for key in sorted(definitions)
    ],
    complete_reached,
    complete_known,
  )


def _publish_group(
  builder: _Builder,
  metadata: Mapping[str, Any],
  trials: list[dict[str, Any]],
) -> None:
  accepted = [trial for trial in trials if builder.register(trial)]
  if not accepted:
    builder.exclude(metadata.get("source", metadata["batch_id"]), "group has no unique trials")
    return

  buckets: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
  for trial in accepted:
    buckets[(
      trial["checkpoint_sha256"],
      trial["mode"],
      trial["completion_protocol_recorded"],
      trial["completion_protocol"],
      trial["rule_sha256"],
      json.dumps(
        trial["control_protocol"],
        sort_keys=True,
        separators=(",", ":"),
      ),
    )].append(trial)

  for key, rows in buckets.items():
    checkpoint_sha, mode, protocol_recorded, protocol, rule_sha, _ = key
    stages, complete_count, complete_known = _stage_counts(rows)
    valid_rows = [trial for trial in rows if trial["valid"]]
    native_rows = [
      trial for trial in valid_rows if type(trial["native_success"]) is bool
    ]
    warnings = list(metadata.get("warnings", []))
    for trial in rows:
      warnings.extend(trial["warnings"])
    warnings = list(dict.fromkeys(warnings))
    identity = {
      "task": metadata["task"],
      "model": metadata["model"],
      "checkpoint_sha256": checkpoint_sha,
      "prediction_horizon": metadata["prediction_horizon"],
      "execute_steps": metadata["execute_steps"],
      "batch_id": metadata["batch_id"],
      "completion_protocol_recorded": protocol_recorded,
      "completion_protocol": protocol,
      "rule_sha256": rule_sha,
      "control_protocol": rows[0]["control_protocol"],
      "mode": mode,
    }
    digest = hashlib.sha256(
      json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    step = metadata.get("step")
    checkpoint_path = metadata.get("checkpoint")
    checkpoint_label = (
      str(step)
      if step is not None
      else Path(checkpoint_path).name if isinstance(checkpoint_path, str) else "unknown"
    )
    for trial in rows:
      for internal in (
        "_fingerprint",
        "checkpoint_sha256",
        "completion_protocol",
        "completion_protocol_recorded",
        "rule_sha256",
        "staged_sidecar_sha256",
      ):
        trial.pop(internal, None)
    builder.groups.append({
      "id": f"group-{digest}",
      "task": metadata["task"],
      "title": task_presentation(metadata["task"])["title"],
      "model": metadata["model"],
      "checkpoint": checkpoint_label,
      "checkpoint_path": checkpoint_path,
      "checkpoint_sha256": checkpoint_sha,
      "prediction_horizon": metadata["prediction_horizon"],
      "execute_steps": metadata["execute_steps"],
      "batch_id": metadata["batch_id"],
      "source_kind": metadata["source_kind"],
      "mode": mode,
      "protocol": {
        **rows[0]["control_protocol"],
        "completion_protocol_recorded": protocol_recorded,
        "completion_protocol": protocol,
        "rule_sha256": rule_sha,
        "validity_basis": sorted({trial["validity_basis"] for trial in rows}),
      },
      "planned_trials": metadata.get("planned_trials"),
      "valid_trials": len(valid_rows),
      "invalid_trials": len(rows) - len(valid_rows),
      "native_success_count": sum(trial["native_success"] is True for trial in native_rows),
      "native_known_count": len(native_rows),
      "stages": stages,
      "complete_stage_success_count": complete_count,
      "complete_stage_known_count": complete_known,
      "warnings": warnings,
      "trials": sorted(rows, key=lambda trial: (trial["seed"] is None, trial["seed"])),
    })


def _manifest_files(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
  rows = manifest.get("files")
  if not isinstance(rows, list):
    return {}
  result = {}
  for row in rows:
    if not isinstance(row, Mapping):
      continue
    relative = row.get("relative_path")
    if isinstance(relative, str) and relative and relative not in result:
      result[relative] = row
  return result


def _load_archive(builder: _Builder) -> None:
  index_path = builder.root / "index.json"
  if not index_path.is_file():
    builder.exclude(index_path, "canonical archive index is missing")
    return
  index, index_row = _read_json(index_path)
  builder.provenance(index_row, "archive_index")
  groups = index.get("groups")
  if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
    raise ValueError("canonical archive index has invalid group list")

  for excluded in index.get("excluded_completed_unpublished_groups", []):
    if isinstance(excluded, Mapping):
      source = excluded.get("directory", "canonical-index-exclusion")
      builder.exclude(source, str(excluded.get("reason", "excluded by canonical index")), details=dict(excluded))

  for group in groups:
    if not isinstance(group, Mapping):
      builder.exclude(index_path, "canonical group entry is not an object")
      continue
    try:
      manifest_path = _safe_relative(builder.root, group["manifest"])
      manifest, manifest_row = _read_json(manifest_path)
      if manifest_row["sha256"] != group.get("manifest_sha256"):
        raise ValueError("archive manifest SHA-256 differs from index")
      builder.provenance(manifest_row, "archive_group_manifest")
      group_summary_path = _safe_relative(builder.root, group["summary"])
      _, group_summary_row = _read_json(
        group_summary_path,
        max_bytes=MAX_GROUP_SUMMARY_JSON_BYTES,
      )
      if group_summary_row["sha256"] != group.get("summary_sha256"):
        raise ValueError("archive group summary SHA-256 differs from index")
      builder.provenance(group_summary_row, "archive_group_summary")
      group_root = _safe_relative(builder.root, group["relative_dir"])
      files = _manifest_files(manifest)
      seeds = group.get("seeds")
      if not isinstance(seeds, list) or len(seeds) > MAX_TRIALS:
        raise ValueError("archive group has invalid seed list")
      expected = {
        "task": group.get("task"),
        "model": group.get("model"),
        "checkpoint_path": group.get("checkpoint"),
        "checkpoint_sha256": group.get("checkpoint_sha256"),
        "prediction_horizon": group.get("prediction_horizon"),
        "execute_steps": group.get("execute_steps"),
      }
      legacy = group.get("archive_schema") == "local-h50-archive-manifest-v1"
      trials = []
      for seed_row in seeds:
        if not isinstance(seed_row, Mapping):
          builder.exclude(group_root, "archive seed entry is not an object")
          continue
        seed_expected = {**expected, "seed": seed_row.get("seed")}
        summary_path = _safe_relative(builder.root, seed_row.get("summary"))
        video_path = _safe_relative(builder.root, seed_row.get("video"))
        trials.append(_load_trial(
          builder,
          summary_path=summary_path,
          video_path=video_path,
          source_kind="archive",
          batch_id=str(group.get("run_id") or manifest.get("batch_id") or index.get("archive_id")),
          expected=seed_expected,
          seed_declaration=seed_row,
          legacy_archive=legacy,
          group_root=group_root,
          manifest_files=files,
        ))
      _publish_group(builder, {
        "task": group["task"],
        "model": group["model"],
        "checkpoint": group.get("checkpoint"),
        "prediction_horizon": group.get("prediction_horizon"),
        "execute_steps": group.get("execute_steps"),
        "batch_id": str(group.get("run_id") or manifest.get("batch_id") or index.get("archive_id")),
        "source_kind": "archive",
        "planned_trials": len(seeds),
        "step": group.get("step"),
        "source": group_root,
      }, trials)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
      builder.exclude(
        group.get("relative_dir", index_path),
        f"canonical group rejected: {type(error).__name__}: {error}",
      )


def _job_progress(status: Mapping[str, Any]) -> str | None:
  completed = _integer(status.get("completed_trials"))
  total = _integer(status.get("total_trials"))
  if completed is not None and total is not None:
    return f"{completed}/{total} trials"
  groups = status.get("groups")
  if isinstance(groups, list):
    done = sum(
      isinstance(group, Mapping)
      and group.get("state", group.get("status")) == "complete"
      for group in groups
    )
    return f"{done}/{len(groups)} groups"
  return None


def _load_new_job(
  builder: _Builder,
  job: Path,
  batch: Mapping[str, Any],
  status: Mapping[str, Any],
) -> None:
  batch_groups = batch.get("groups")
  status_groups = status.get("groups")
  settings = batch.get("settings")
  if (
    not isinstance(batch_groups, list)
    or not isinstance(status_groups, list)
    or len(batch_groups) != len(status_groups)
    or not isinstance(settings, Mapping)
  ):
    builder.exclude(job, "new-style batch/status group alignment is invalid")
    return
  for group, state in zip(batch_groups, status_groups, strict=True):
    if not isinstance(group, Mapping) or not isinstance(state, Mapping):
      builder.exclude(job, "new-style batch group entry is invalid")
      continue
    source = job / str(group.get("group_dir", "invalid-group"))
    if state.get("state") != "complete":
      builder.exclude(source, f"job group is not complete: {state.get('state')!r}")
      continue
    attempt = _integer(state.get("attempt"))
    if attempt is None or attempt <= 0:
      builder.exclude(source, "completed job group has no accepted attempt")
      continue
    try:
      group_dir = _safe_relative(job, group["group_dir"])
      evaluation = group_dir / f"attempt_{attempt:03d}" / "run" / "evaluation"
      summaries = sorted(evaluation.glob("seed_*/summary.json"))
      if not summaries or len(summaries) > MAX_TRIALS:
        raise ValueError("accepted attempt has no bounded seed summary set")
      expected = {
        "task": group.get("task"),
        "model": "pi05",
        "checkpoint_path": group.get("checkpoint"),
        "prediction_horizon": settings.get("prediction_horizon"),
        "execute_steps": settings.get("execute_steps"),
      }
      trials = []
      for summary_path in summaries:
        match = re.fullmatch(r"seed_(\d+)", summary_path.parent.name)
        if match is None:
          builder.exclude(summary_path, "unsupported seed directory name")
          continue
        video_path = summary_path.parent / "review.mp4"
        trials.append(_load_trial(
          builder,
          summary_path=summary_path,
          video_path=video_path,
          source_kind="job",
          batch_id=job.name,
          expected={**expected, "seed": int(match.group(1))},
        ))
      warnings = []
      if state.get("error"):
        warnings.append(
          "Controller status retains an earlier retry error; the explicit accepted attempt was used."
        )
      _publish_group(builder, {
        "task": group["task"],
        "model": "pi05",
        "checkpoint": group.get("checkpoint"),
        "prediction_horizon": settings.get("prediction_horizon"),
        "execute_steps": settings.get("execute_steps"),
        "batch_id": job.name,
        "source_kind": "job",
        "planned_trials": settings.get("num_trials"),
        "step": group.get("step"),
        "source": evaluation,
        "warnings": warnings,
      }, trials)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
      builder.exclude(source, f"accepted job attempt rejected: {type(error).__name__}: {error}")


def _load_jobs(builder: _Builder) -> None:
  jobs_root = builder.root / "_jobs"
  if not jobs_root.is_dir():
    return
  jobs = sorted(path for path in jobs_root.iterdir() if path.is_dir())
  if len(jobs) > MAX_JOBS:
    raise ValueError("evaluation job directory bound exceeded")
  for job in jobs:
    status_path = job / "status.json"
    batch_path = job / "batch.json"
    if not status_path.is_file():
      kind = "normalization" if "norm" in job.name.lower() else "unknown"
      builder.jobs.append({
        "name": job.name,
        "kind": kind,
        "state": "unknown",
        "description": (
          "Normalization/training artifact only; never counted as an evaluation trial."
          if kind == "normalization"
          else "No bounded status.json was present; not counted as evaluation evidence."
        ),
        "progress": None,
      })
      builder.exclude(job, "normalization/non-evaluation job has no trial evidence")
      continue
    try:
      status, status_row = _read_json(status_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
      builder.jobs.append({
        "name": job.name,
        "kind": "unknown",
        "state": "invalid_status",
        "description": f"Status could not be read: {type(error).__name__}: {error}",
        "progress": None,
      })
      builder.exclude(job, "job status is invalid")
      continue
    state = status.get("state", status.get("status", "unknown"))
    schema = status.get("schema")
    new_style = schema == "kaihand-pi05-stage-batch-v1" and batch_path.is_file()
    builder.jobs.append({
      "name": job.name,
      "kind": "staged_evaluation" if new_style else "historical_evaluation",
      "state": str(state),
      "description": (
        "Accepted attempts are selected from status.json; no score-based selection."
        if new_style
        else "Historical/superseded job status retained; canonical index is authoritative."
      ),
      "progress": _job_progress(status),
    })
    if not new_style:
      builder.exclude(
        job,
        "historical, failed, or superseded job is not imported outside the canonical index",
      )
      continue
    try:
      batch, batch_row = _read_json(batch_path)
      if state == "complete":
        builder.provenance(status_row, "completed_job_status_attempt_selection")
        builder.provenance(batch_row, "completed_job_batch_plan")
      _load_new_job(builder, job, batch, status)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
      builder.exclude(job, f"new-style job rejected: {type(error).__name__}: {error}")


def build_catalog(source_root: str | Path) -> dict[str, Any]:
  """Return normalized native/staged facts without mutating the source tree."""

  root = Path(source_root).expanduser().absolute()
  if not root.is_dir():
    raise ValueError(f"evaluation source root does not exist: {root}")
  builder = _Builder(root)
  _load_archive(builder)
  _load_jobs(builder)
  if len(builder.groups) > MAX_GROUPS:
    raise ValueError("normalized evaluation group bound exceeded")
  groups = sorted(
    builder.groups,
    key=lambda group: (
      TASKS.index(group["task"]) if group["task"] in TASKS else len(TASKS),
      group["model"],
      group["checkpoint_sha256"] or "",
      group["batch_id"],
      group["id"],
    ),
  )
  tasks = []
  for task in TASKS:
    presentation = task_presentation(task)
    tasks.append({
      "task": task,
      "title": presentation["title"],
      "note": presentation["task_note"],
    })
  return {
    "schema": SCHEMA,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_root": str(root),
    "tasks": tasks,
    "groups": groups,
    "jobs": builder.jobs,
    "excluded": builder.excluded,
    "provenance": builder.provenance_rows,
  }


__all__ = ["SCHEMA", "build_catalog"]
