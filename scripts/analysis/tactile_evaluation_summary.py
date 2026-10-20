#!/usr/bin/env python3
"""Build horizon-aware JSON and HTML summaries for tactile policy trials.

This is intentionally independent of the historical H50-only workbench.  It
groups only explicit trial metadata and never infers a prediction horizon from
the execution length, control rate, directory name, or model action width.

Each input JSON document may be one trial object, a list of trial objects, or
an object whose ``trials`` or ``results`` member is a list.  The canonical
trial identity fields are::

    task, model_family, deployment_manifest_sha256, prediction_horizon,
    action_start_index, execute_steps, tactile_sent_to_model, guidance

``guidance`` is an explicit object with ``enabled``, ``steps``, ``scale`` and
an optional ``variant``.  An unguided trial must still declare
``{"enabled": false}``; missing guidance is not silently treated as disabled.

Outcome evidence uses ``seed``, ``status``, ``native_success`` and
``formal_metrics_valid``.  The native runner's ``task_success`` is accepted as
an alias for ``native_success``; declaring both with different values is an
error.  ``trial_validity`` (valid/invalid/missing) is the preferred explicit
evidence classification.  For compatibility, ``valid_trial`` is accepted
next, followed by a deliberately small status mapping.  Invalid and missing
trials are reported but never enter a success rate denominator.

``poker_tactile_batch_v1`` inputs resolve each relative ``trial.summary``
reference for identity and outcome evidence.  The batch row is an additional
validity gate: only child-valid evidence with ``valid=true``,
``timed_out=false`` and ``returncode=0`` can enter a success-rate denominator.
Batch evidence can downgrade child evidence but can never upgrade it.
Directory discovery prefers a parent ``batch_summary.json`` over its child
summaries so batch-level failures cannot be hidden by reading a successful
child in isolation.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "tactile-evaluation-summary-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALID_STATUSES = frozenset({"success", "task_not_completed", "completed", "failure"})
_INVALID_STATUSES = frozenset(
  {"error", "invalid", "infrastructure_error", "fatal_error"}
)
_MISSING_STATUSES = frozenset({"missing", "incomplete", "unknown"})


@dataclass(frozen=True)
class ClassifiedTrial:
  source: str
  identity: dict[str, Any] | None
  mode: str | None
  evidence_status: str
  native_success: bool | None
  reasons: tuple[str, ...]
  seed: int | None
  status: str | None


def _positive_integer(value: object) -> bool:
  return type(value) is int and value > 0


def _nonnegative_integer(value: object) -> bool:
  return type(value) is int and value >= 0


def _finite_number(value: object) -> bool:
  return (
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
  )


def _missing(record: Mapping[str, Any], key: str) -> bool:
  return key not in record or record[key] is None


def _guidance(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
  reasons: list[str] = []
  nested = record.get("guidance")
  flat_present = any(
    key in record
    for key in (
      "guidance_enabled",
      "guidance_steps",
      "guidance_scale",
      "guidance_variant",
    )
  )
  if nested is None and not flat_present:
    return None, ["missing_guidance"]
  if nested is not None and not isinstance(nested, Mapping):
    return None, ["invalid_guidance"]

  nested = nested or {}
  values = {
    "enabled": nested.get("enabled", record.get("guidance_enabled")),
    "steps": nested.get("steps", record.get("guidance_steps")),
    "scale": nested.get("scale", record.get("guidance_scale")),
    "variant": nested.get("variant", record.get("guidance_variant")),
  }
  for nested_key, flat_key in (
    ("enabled", "guidance_enabled"),
    ("steps", "guidance_steps"),
    ("scale", "guidance_scale"),
    ("variant", "guidance_variant"),
  ):
    if (
      nested_key in nested
      and flat_key in record
      and nested[nested_key] != record[flat_key]
    ):
      reasons.append(f"conflicting_{flat_key}")

  enabled = values["enabled"]
  if type(enabled) is not bool:
    reasons.append(
      "missing_guidance_enabled" if enabled is None else "invalid_guidance_enabled"
    )
  steps = values["steps"]
  scale = values["scale"]
  variant = values["variant"]
  if enabled is True:
    if not _positive_integer(steps):
      reasons.append(
        "missing_guidance_steps" if steps is None else "invalid_guidance_steps"
      )
    if not _finite_number(scale) or float(scale) <= 0:
      reasons.append(
        "missing_guidance_scale" if scale is None else "invalid_guidance_scale"
      )
  elif enabled is False:
    if steps not in (None, 0):
      reasons.append("unguided_trial_has_guidance_steps")
    if scale is not None and (not _finite_number(scale) or float(scale) != 0.0):
      reasons.append("unguided_trial_has_guidance_scale")
    steps, scale = 0, 0.0
  if variant is not None and (not isinstance(variant, str) or not variant.strip()):
    reasons.append("invalid_guidance_variant")

  if reasons:
    return None, reasons
  return {
    "enabled": enabled,
    "steps": int(steps),
    "scale": float(scale),
    "variant": variant.strip() if isinstance(variant, str) else None,
  }, []


def _protocol(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
  protocol: dict[str, Any] = {
    "max_sim_seconds": None,
    "guard_mode": None,
    "controller": None,
    "action_interpolation": None,
    "success_hold_seconds": None,
  }
  reasons: list[str] = []
  for key in ("max_sim_seconds", "success_hold_seconds"):
    value = record.get(key)
    if value is None:
      continue
    if not _finite_number(value) or float(value) <= 0:
      reasons.append(f"invalid_{key}")
    else:
      protocol[key] = float(value)

  guard_mode = record.get("guard_mode")
  if guard_mode is not None:
    if guard_mode not in {"observe", "strict"}:
      reasons.append("invalid_guard_mode")
    else:
      protocol["guard_mode"] = guard_mode

  for key in ("controller", "action_interpolation"):
    value = record.get(key)
    if value is None:
      continue
    if not isinstance(value, str) or not value.strip():
      reasons.append(f"invalid_{key}")
    else:
      protocol[key] = value.strip()
  return protocol, reasons


def _identity(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
  reasons: list[str] = []
  for key in ("task", "model_family"):
    value = record.get(key)
    if value is None:
      reasons.append(f"missing_{key}")
    elif not isinstance(value, str) or not value.strip():
      reasons.append(f"invalid_{key}")

  digest = record.get("deployment_manifest_sha256")
  if digest is None:
    reasons.append("missing_deployment_manifest_sha256")
  elif not isinstance(digest, str) or _SHA256.fullmatch(digest.lower()) is None:
    reasons.append("invalid_deployment_manifest_sha256")

  horizon = record.get("prediction_horizon")
  start = record.get("action_start_index")
  execute = record.get("execute_steps")
  for key, value, predicate in (
    ("prediction_horizon", horizon, _positive_integer),
    ("action_start_index", start, _nonnegative_integer),
    ("execute_steps", execute, _positive_integer),
  ):
    if value is None:
      reasons.append(f"missing_{key}")
    elif not predicate(value):
      reasons.append(f"invalid_{key}")
  if all(type(value) is int for value in (horizon, start, execute)) and (
    horizon <= 0 or start < 0 or execute <= 0 or start + execute > horizon
  ):
    reasons.append("action_window_exceeds_prediction_horizon")

  tactile = record.get("tactile_sent_to_model")
  if tactile is None:
    reasons.append("missing_tactile_sent_to_model")
  elif type(tactile) is not bool:
    reasons.append("invalid_tactile_sent_to_model")

  guidance, guidance_reasons = _guidance(record)
  reasons.extend(guidance_reasons)
  protocol, protocol_reasons = _protocol(record)
  reasons.extend(protocol_reasons)
  if reasons:
    return None, reasons
  return {
    "task": record["task"].strip(),
    "model_family": record["model_family"].strip(),
    "deployment_manifest_sha256": digest.lower(),
    "prediction_horizon": horizon,
    "action_start_index": start,
    "execute_steps": execute,
    "tactile_sent_to_model": tactile,
    "guidance": guidance,
    "protocol": protocol,
  }, []


def _mode(record: Mapping[str, Any]) -> tuple[str | None, list[str]]:
  formal = record.get("formal_metrics_valid")
  diagnostic = record.get("diagnostic_only")
  if type(formal) is not bool:
    return None, [
      "missing_formal_metrics_valid"
      if formal is None
      else "invalid_formal_metrics_valid"
    ]
  if diagnostic is not None and type(diagnostic) is not bool:
    return None, ["invalid_diagnostic_only"]
  if diagnostic is not None and diagnostic == formal:
    return None, ["conflicting_formal_and_diagnostic_flags"]
  return ("formal" if formal else "diagnostic"), []


def _evidence_status(record: Mapping[str, Any]) -> tuple[str, list[str]]:
  declared = record.get("trial_validity")
  if declared is not None:
    if declared not in {"valid", "invalid", "missing"}:
      return "invalid", ["invalid_trial_validity"]
    valid_trial = record.get("valid_trial")
    if valid_trial is not None:
      if type(valid_trial) is not bool:
        return "invalid", ["invalid_valid_trial"]
      if (declared == "valid") != valid_trial:
        return "invalid", ["conflicting_trial_validity"]
    return str(declared), []
  valid_trial = record.get("valid_trial")
  if valid_trial is not None:
    if type(valid_trial) is not bool:
      return "invalid", ["invalid_valid_trial"]
    return ("valid" if valid_trial else "invalid"), []
  status = record.get("status")
  if status in _VALID_STATUSES:
    return "valid", []
  if status in _INVALID_STATUSES:
    return "invalid", []
  if status is None or status in _MISSING_STATUSES:
    return "missing", []
  return "invalid", ["unrecognized_status"]


def classify_trial(record: object, *, source: str) -> ClassifiedTrial:
  if not isinstance(record, Mapping):
    return ClassifiedTrial(
      source, None, None, "invalid", None, ("trial_is_not_an_object",), None, None
    )

  identity, identity_reasons = _identity(record)
  mode, mode_reasons = _mode(record)
  evidence_status, evidence_reasons = _evidence_status(record)
  reasons = [*identity_reasons, *mode_reasons, *evidence_reasons]
  loader_reasons = record.get("_tactile_summary_loader_reasons", [])
  if isinstance(loader_reasons, list) and all(
    isinstance(reason, str) and reason for reason in loader_reasons
  ):
    reasons.extend(loader_reasons)
  loader_status = record.get("_tactile_summary_loader_status")
  if loader_status == "invalid":
    evidence_status = "invalid"
  elif loader_status == "missing" and evidence_status != "invalid":
    evidence_status = "missing"

  seed = record.get("seed")
  if seed is None:
    reasons.append("missing_seed")
  elif not _nonnegative_integer(seed):
    reasons.append("invalid_seed")
    seed = None

  status = record.get("status")
  if status is None:
    reasons.append("missing_status")
    if evidence_status != "invalid":
      evidence_status = "missing"
  elif not isinstance(status, str) or not status.strip():
    reasons.append("invalid_status")
    status = None
    evidence_status = "invalid"

  native_declared = "native_success" in record
  task_declared = "task_success" in record
  success = (
    record.get("native_success")
    if native_declared
    else record.get("task_success")
  )
  if (
    native_declared
    and task_declared
    and record.get("native_success") != record.get("task_success")
  ):
    reasons.append("conflicting_native_and_task_success")
    evidence_status = "invalid"

  physics = record.get("physics_valid")
  if "physics_valid" in record and type(physics) is not bool:
    reasons.append("invalid_physics_valid")
    evidence_status = "invalid"
  elif physics is False and evidence_status == "valid":
    reasons.append("physics_invalid_trial_cannot_be_valid")
    evidence_status = "invalid"

  stop_reason = record.get("stop_reason")
  if stop_reason is not None and (
    not isinstance(stop_reason, str) or not stop_reason.strip()
  ):
    reasons.append("invalid_stop_reason")
    evidence_status = "invalid"

  if evidence_status == "valid":
    if type(success) is not bool:
      reasons.append(
        "missing_native_success" if success is None else "invalid_native_success"
      )
      evidence_status = "missing" if success is None else "invalid"
      success = None
    elif status in _INVALID_STATUSES or status in _MISSING_STATUSES:
      reasons.append("status_conflicts_with_valid_trial")
      evidence_status = "invalid"
    elif status == "success" and success is not True:
      reasons.append("success_status_conflicts_with_native_success")
      evidence_status = "invalid"
  elif success is not None and type(success) is not bool:
    reasons.append("invalid_native_success")
    success = None

  if identity is None or mode is None or seed is None:
    # Missing required values remain missing; malformed/conflicting values
    # are invalid.  Neither class contributes to a success denominator.
    evidence_status = (
      "invalid"
      if any(not reason.startswith("missing_") for reason in reasons)
      else "missing"
    )
  return ClassifiedTrial(
    source=source,
    identity=identity,
    mode=mode,
    evidence_status=evidence_status,
    native_success=success if type(success) is bool else None,
    reasons=tuple(dict.fromkeys(reasons)),
    seed=seed if type(seed) is int else None,
    status=status if isinstance(status, str) else None,
  )


def _group_key(identity: Mapping[str, Any]) -> str:
  return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _empty_counts() -> dict[str, Any]:
  return {
    "total": 0,
    "valid": 0,
    "invalid": 0,
    "missing": 0,
    "successes": 0,
    "failures": 0,
    "success_rate": None,
  }


def _add(counts: dict[str, Any], trial: ClassifiedTrial) -> None:
  counts["total"] += 1
  counts[trial.evidence_status] += 1
  if trial.evidence_status == "valid":
    if trial.native_success:
      counts["successes"] += 1
    else:
      counts["failures"] += 1


def _finish(counts: dict[str, Any]) -> None:
  counts["success_rate"] = (
    counts["successes"] / counts["valid"] if counts["valid"] else None
  )


def summarize_trials(
  records: Iterable[object], *, sources: Sequence[str] | None = None
) -> dict[str, Any]:
  values = list(records)
  if sources is None:
    sources = [f"record[{index}]" for index in range(len(values))]
  if len(sources) != len(values):
    raise ValueError("sources must have exactly one entry per trial")

  classified = [
    classify_trial(record, source=source)
    for record, source in zip(values, sources, strict=True)
  ]
  groups: dict[str, dict[str, Any]] = {}
  ungrouped = []
  for trial in classified:
    if trial.identity is None:
      ungrouped.append(
        {
          "source": trial.source,
          "evidence_status": trial.evidence_status,
          "reasons": list(trial.reasons),
        }
      )
      continue
    key = _group_key(trial.identity)
    group = groups.setdefault(
      key,
      {
        "identity": trial.identity,
        "overall": _empty_counts(),
        "formal": _empty_counts(),
        "diagnostic": _empty_counts(),
        "trials": [],
      },
    )
    _add(group["overall"], trial)
    if trial.mode is not None:
      _add(group[trial.mode], trial)
    group["trials"].append(
      {
        "source": trial.source,
        "seed": trial.seed,
        "status": trial.status,
        "mode": trial.mode,
        "evidence_status": trial.evidence_status,
        "native_success": trial.native_success,
        "reasons": list(trial.reasons),
      }
    )

  ordered_groups = sorted(
    groups.values(),
    key=lambda group: (
      group["identity"]["task"],
      group["identity"]["model_family"],
      group["identity"]["deployment_manifest_sha256"],
      group["identity"]["prediction_horizon"],
      group["identity"]["action_start_index"],
      group["identity"]["execute_steps"],
      _group_key(group["identity"]["guidance"]),
      _group_key(group["identity"]["protocol"]),
    ),
  )
  for group in ordered_groups:
    for name in ("overall", "formal", "diagnostic"):
      _finish(group[name])
    group["trials"].sort(
      key=lambda row: (row["seed"] is None, row["seed"], row["source"])
    )

  return {
    "schema_version": SCHEMA_VERSION,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "semantics": {
      "grouping": (
        "Explicit task, model, deployment manifest SHA-256, prediction horizon, "
        "action start, execute steps, tactile flag, guidance configuration, "
        "and explicitly declared evaluation protocol"
      ),
      "horizon": (
        "prediction_horizon is never inferred from execute_steps, action width, "
        "control rate, paths, or names"
      ),
      "success_rate": "native successes divided only by valid trials",
      "modes": "formal and diagnostic evidence are counted separately",
    },
    "input_trials": len(classified),
    "grouped_trials": sum(group["overall"]["total"] for group in ordered_groups),
    "ungrouped_trials": len(ungrouped),
    "groups": ordered_groups,
    "ungrouped": ungrouped,
  }


def _document_trials(
  document: object, *, source: Path
) -> tuple[list[object], list[str]]:
  if isinstance(document, list):
    records = document
  elif isinstance(document, Mapping):
    containers = [key for key in ("trials", "results") if key in document]
    if len(containers) > 1:
      raise ValueError(f"ambiguous trial containers in {source}: {containers}")
    if containers and not isinstance(document[containers[0]], list):
      raise ValueError(f"{source}.{containers[0]} must be a list")
    if containers and containers[0] == "trials":
      records = document["trials"]
    elif containers:
      records = document["results"]
    else:
      records = [document]
  else:
    records = [document]
  labels = [f"{source}[{index}]" for index in range(len(records))]
  return list(records), labels


def _read_json(path: Path) -> object:
  try:
    return json.loads(
      path.read_text(encoding="utf-8"),
      parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant {value}")
      ),
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid JSON summary: {path}") from error


def _batch_validity(row: Mapping[str, Any]) -> tuple[str, list[str]]:
  reasons: list[str] = []
  valid = row.get("valid")
  timed_out = row.get("timed_out")
  returncode = row.get("returncode")
  missing = []
  for key, value in (
    ("valid", valid),
    ("timed_out", timed_out),
    ("returncode", returncode),
  ):
    if value is None:
      missing.append(key)
  if valid is not None and type(valid) is not bool:
    reasons.append("invalid_batch_valid")
  if timed_out is not None and type(timed_out) is not bool:
    reasons.append("invalid_batch_timed_out")
  if returncode is not None and type(returncode) is not int:
    reasons.append("invalid_batch_returncode")
  if reasons:
    return "invalid", reasons
  if valid is False:
    reasons.append("batch_declared_invalid")
  if timed_out is True:
    reasons.append("batch_timed_out")
  if type(returncode) is int and returncode != 0:
    reasons.append("batch_nonzero_returncode")
  if reasons:
    return "invalid", reasons
  if missing:
    return "missing", [f"missing_batch_{key}" for key in missing]
  return "valid", []


def _batch_trials(
  document: Mapping[str, Any], *, source: Path
) -> tuple[list[object], list[str]]:
  rows = document.get("trials")
  if not isinstance(rows, list):
    raise ValueError(f"{source}.trials must be a list")
  records: list[object] = []
  labels: list[str] = []
  batch_root = source.parent.resolve()
  for index, row in enumerate(rows):
    batch_label = f"{source}[{index}]"
    if not isinstance(row, Mapping):
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": ["batch_trial_is_not_an_object"],
      })
      labels.append(batch_label)
      continue
    reference = row.get("summary")
    if reference is None:
      records.append({
        "_tactile_summary_loader_status": "missing",
        "_tactile_summary_loader_reasons": ["missing_batch_summary_ref"],
      })
      labels.append(batch_label)
      continue
    if not isinstance(reference, str) or not reference.strip():
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": ["invalid_batch_summary_ref"],
      })
      labels.append(batch_label)
      continue
    child = Path(reference)
    if child.is_absolute():
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": ["absolute_batch_summary_ref"],
      })
      labels.append(batch_label)
      continue
    child = (batch_root / child).resolve()
    child_label = f"{child} (via {batch_label})"
    if not child.is_relative_to(batch_root):
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": ["batch_summary_ref_escapes_root"],
      })
      labels.append(child_label)
      continue
    if not child.is_file():
      records.append({
        "_tactile_summary_loader_status": "missing",
        "_tactile_summary_loader_reasons": ["missing_referenced_summary"],
      })
      labels.append(child_label)
      continue
    try:
      child_document = _read_json(child)
    except ValueError:
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": ["invalid_referenced_summary"],
      })
      labels.append(child_label)
      continue
    if not isinstance(child_document, Mapping):
      records.append({
        "_tactile_summary_loader_status": "invalid",
        "_tactile_summary_loader_reasons": [
          "referenced_summary_is_not_an_object"
        ],
      })
      labels.append(child_label)
      continue

    record = dict(child_document)
    batch_status, batch_reasons = _batch_validity(row)
    child_status, _ = _evidence_status(record)
    severity = {"valid": 0, "missing": 1, "invalid": 2}
    effective_status = max(
      (child_status, batch_status), key=severity.__getitem__
    )
    if child_status != batch_status:
      batch_reasons.append("batch_child_validity_conflict")
      if severity[batch_status] > severity[child_status]:
        batch_reasons.append("batch_downgraded_child_validity")
    record["_tactile_summary_loader_status"] = effective_status
    record["_tactile_summary_loader_reasons"] = batch_reasons
    if effective_status == "missing":
      record.pop("trial_validity", None)
      record.pop("valid_trial", None)
    else:
      record["trial_validity"] = effective_status
      record["valid_trial"] = effective_status == "valid"
    records.append(record)
    labels.append(child_label)
  return records, labels


def _directory_candidates(root: Path) -> list[Path]:
  batches = sorted(root.rglob("batch_summary.json"))
  batch_roots = {path.parent.resolve() for path in batches}
  summaries = []
  for summary in sorted(root.rglob("summary.json")):
    resolved = summary.resolve()
    if any(batch_root == resolved.parent or batch_root in resolved.parents
           for batch_root in batch_roots):
      continue
    summaries.append(summary)
  return [*batches, *summaries]


def load_trial_summaries(paths: Sequence[Path]) -> tuple[list[object], list[str]]:
  records: list[object] = []
  sources: list[str] = []
  files: list[Path] = []
  seen_files: set[Path] = set()
  for path in paths:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
      candidates = _directory_candidates(resolved)
    elif resolved.is_file():
      candidates = [resolved]
    else:
      raise FileNotFoundError(resolved)
    for candidate in candidates:
      candidate = candidate.resolve()
      if candidate not in seen_files:
        files.append(candidate)
        seen_files.add(candidate)
  if not files:
    raise ValueError("no JSON summary files found")
  for path in files:
    document = _read_json(path)
    if (
      path.name == "batch_summary.json"
      or isinstance(document, Mapping)
      and document.get("schema") in {"poker_tactile_batch_v1", "task_tactile_batch_v2"}
    ):
      if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a batch object")
      rows, labels = _batch_trials(document, source=path)
    else:
      rows, labels = _document_trials(document, source=path)
    records.extend(rows)
    sources.extend(labels)
  return records, sources


def _rate(value: object) -> str:
  return "—" if value is None else f"{100 * float(value):.1f}%"


def render_html(report: Mapping[str, Any]) -> str:
  rows = []
  for group in report.get("groups", []):
    identity = group["identity"]
    guidance = identity["guidance"]
    protocol = identity["protocol"]
    guidance_label = (
      f"{guidance.get('variant') or 'guided'} / steps={guidance['steps']} / "
      f"scale={guidance['scale']:g}"
      if guidance["enabled"]
      else "disabled"
    )
    protocol_label = "; ".join((
      "max=" + (
        "—" if protocol["max_sim_seconds"] is None
        else f"{protocol['max_sim_seconds']:g}s"
      ),
      f"guard={protocol['guard_mode'] or '—'}",
      f"controller={protocol['controller'] or '—'}",
      f"interpolation={protocol['action_interpolation'] or '—'}",
      "success-hold=" + (
        "—" if protocol["success_hold_seconds"] is None
        else f"{protocol['success_hold_seconds']:g}s"
      ),
    ))
    for mode in ("formal", "diagnostic"):
      counts = group[mode]
      rows.append(
        "<tr>"
        + "".join(
          f"<td>{html.escape(str(value))}</td>"
          for value in (
            identity["task"],
            identity["model_family"],
            identity["deployment_manifest_sha256"][:12],
            f"H{identity['prediction_horizon']}",
            identity["action_start_index"],
            identity["execute_steps"],
            guidance_label,
            protocol_label,
            mode,
            counts["total"],
            counts["valid"],
            counts["invalid"],
            counts["missing"],
            counts["successes"],
            counts["failures"],
            _rate(counts["success_rate"]),
          )
        )
        + "</tr>"
      )
  ungrouped_rows = "".join(
    "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
      html.escape(str(row.get("source"))),
      html.escape(str(row.get("evidence_status"))),
      html.escape(", ".join(row.get("reasons", []))),
    )
    for row in report.get("ungrouped", [])
  )
  return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tactile evaluation summary</title>
<style>
body{{font:14px system-ui,sans-serif;margin:2rem;color:#182331}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #cbd3dc;padding:.45rem;text-align:left;vertical-align:top}}th{{background:#eef2f6}}
code{{background:#eef2f6;padding:.1rem .25rem}}.note{{max-width:80rem;color:#465567}}
</style></head><body>
<h1>Tactile evaluation summary</h1>
<p class="note">Horizon-aware grouping uses explicit metadata only. H30, H32, and H50 are never combined. Invalid and missing trials are shown but excluded from success-rate denominators. Formal and diagnostic evidence are separate.</p>
<p>Inputs: {int(report.get("input_trials", 0))}; grouped: {int(report.get("grouped_trials", 0))}; ungrouped: {int(report.get("ungrouped_trials", 0))}.</p>
<table><thead><tr><th>Task</th><th>Model</th><th>Manifest SHA</th><th>Horizon</th><th>Action start</th><th>K</th><th>Guidance</th><th>Protocol</th><th>Mode</th><th>Total</th><th>Valid</th><th>Invalid</th><th>Missing</th><th>Success</th><th>Failure</th><th>Success rate</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Ungrouped evidence</h2>
<table><thead><tr><th>Source</th><th>Evidence status</th><th>Reasons</th></tr></thead><tbody>{ungrouped_rows}</tbody></table>
</body></html>
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "inputs",
    nargs="+",
    type=Path,
    help="JSON files or directories containing summary.json",
  )
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-html", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  json_path = args.output_json.expanduser().resolve()
  html_path = args.output_html.expanduser().resolve()
  if json_path == html_path:
    raise ValueError("JSON and HTML outputs must be different files")
  for output in (json_path, html_path):
    if output.exists():
      raise FileExistsError(f"refusing to overwrite existing output: {output}")
  records, sources = load_trial_summaries(args.inputs)
  report = summarize_trials(records, sources=sources)
  encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
  rendered = render_html(report)
  json_path.parent.mkdir(parents=True, exist_ok=True)
  html_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(encoded, encoding="utf-8")
  html_path.write_text(rendered, encoding="utf-8")
  print(
    json.dumps(
      {
        "output_json": str(json_path),
        "output_html": str(html_path),
        "groups": len(report["groups"]),
        "input_trials": report["input_trials"],
      },
      ensure_ascii=False,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
