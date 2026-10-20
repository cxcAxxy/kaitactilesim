"""Select prediction horizons from bounded, immutable metadata reads only.

Only the explicit ``prediction_horizon`` field in summary.json and its bound
deployment manifest is evidence. Execution steps, control frequency, model
names and directory names never establish a prediction horizon. No simulator,
checkpoint, NPZ, video or source-code files are opened by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_ANCESTORS = 3
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_MANIFEST_NAME = re.compile(r"deployment_manifest(?:\.([0-9a-fA-F]{64}))?\.json\Z")


class _UnavailableManifest(ValueError):
    """A bound manifest cannot safely be read; direct summary evidence remains."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def _read_metadata(path: Path, role: str, provenance: list) -> dict:
    """Bound reads even when an input grows or is replaced while reading."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Metadata must be a regular file")
        if before.st_size > MAX_METADATA_BYTES:
            raise ValueError("Metadata exceeds bounded size limit")
        raw = handle.read(MAX_METADATA_BYTES + 1)
        after = os.fstat(handle.fileno())
    final = path.stat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if len(raw) > MAX_METADATA_BYTES or len(raw) != before.st_size:
        raise ValueError("Metadata exceeds size limit or changed while reading")
    if identity(before) != identity(after) or identity(after) != identity(final):
        raise ValueError("Metadata changed while reading")
    provenance.append({"path": str(path), "size": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest(), "role": role})
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant,
                       parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("Metadata JSON must be an object")
    return value


def _manifest_path(raw_path: str, trial: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Manifest path must be a nonempty string")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = trial / candidate
    candidate = Path(os.path.abspath(candidate))
    resolved = candidate.resolve()
    if candidate != resolved:
        raise _UnavailableManifest("Manifest symlinks are not accepted")
    if not _MANIFEST_NAME.fullmatch(candidate.name):
        raise ValueError("Manifest filename is not an allowed deployment metadata filename")
    # Only fixed deployment layouts inside at most three experiment ancestors.
    # This also prevents an unrelated JSON elsewhere under a broad parent being read.
    for ancestor in [trial, *list(trial.parents)[:MAX_MANIFEST_ANCESTORS]]:
        if ancestor == Path(ancestor.anchor):
            continue
        try:
            parts = candidate.relative_to(ancestor).parts
        except ValueError:
            continue
        if len(parts) == 1:
            return candidate
        if len(parts) == 2 and (parts[0] in {"deployment", "deployment_snapshot"}
                                or parts[0].endswith("_deployment")):
            return candidate
    raise _UnavailableManifest("Manifest is outside the bounded experiment deployment paths")


def _declaration(document: dict, path: Path, role: str, evidence: list, warnings: list) -> bool:
    if "prediction_horizon" not in document:
        return False
    value = document["prediction_horizon"]
    valid = type(value) is int and value > 0
    evidence.append({"source": str(path), "field": "prediction_horizon", "value": value,
                     "valid": valid, "role": role})
    if not valid:
        warnings.append(f"{role}.prediction_horizon must be a positive integer")
    return not valid


def inspect_prediction_horizon(trial_dir) -> dict:
    """Return verified/unknown/conflict/invalid without reading trajectory arrays.

    An explicit summary declaration is sufficient when no manifest is declared.
    An unavailable or out-of-bounds manifest adds a warning without erasing an
    explicit summary declaration. Every readable manifest's recorded checksum
    and shared deployment identity must agree. ``verified`` means explicit
    metadata agreement only, not deployment, model-weight or rollout validation.
    """
    result = {"status": "unknown", "prediction_horizon": None, "evidence": [],
              "provenance": [], "warnings": [],
              "verification_scope": "Explicit prediction_horizon metadata only; not deployment or model-weight certification"}
    evidence, provenance, warnings = (result[key] for key in ("evidence", "provenance", "warnings"))
    try:
        trial = Path(trial_dir).resolve()
        summary_path = trial / "summary.json"
        if summary_path.resolve() != summary_path:
            raise ValueError("Summary symlinks outside the trial are not accepted")
        summary = _read_metadata(summary_path, "prediction_horizon_summary", provenance)
    except FileNotFoundError:
        warnings.append("summary.json is missing; prediction horizon is unknown")
        return result
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        result["status"] = "invalid"
        warnings.append(f"Summary metadata rejected: {exc}")
        return result

    invalid = _declaration(summary, summary_path, "summary", evidence, warnings)
    integrity = summary.get("deployment_integrity", {})
    if not isinstance(integrity, dict):
        integrity = {}
        invalid = True
        warnings.append("deployment_integrity must be an object")
    integrity_manifest = integrity.get("manifest", {})
    if not isinstance(integrity_manifest, dict):
        integrity_manifest = {}
        invalid = True
        warnings.append("deployment_integrity.manifest must be an object")
    paths = [container[key] for container, key in (
        (summary, "deployment_manifest"), (integrity_manifest, "path")) if key in container]
    hashes = [container[key] for container, key in (
        (summary, "deployment_manifest_sha256"), (integrity_manifest, "expected_sha256"),
        (integrity_manifest, "actual_sha256")) if key in container]
    # Check facts already present in summary before resolving an optional file.
    # An unavailable file cannot turn a recorded failure or contradictory hashes
    # into an unverified-but-acceptable metadata declaration.
    binding_valid = True
    try:
        if integrity.get("status", "verified") != "verified":
            raise ValueError("Recorded deployment integrity is not verified")
        if "errors" in integrity and not isinstance(integrity["errors"], list):
            raise ValueError("deployment_integrity.errors must be an array")
        if integrity.get("errors"):
            raise ValueError("Recorded deployment integrity contains errors")
        for value in hashes:
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError("Manifest SHA256 declaration is malformed")
        if len({value.lower() for value in hashes}) > 1:
            raise ValueError("Recorded manifest SHA256 declarations disagree")
        if hashes and not paths:
            raise ValueError("Manifest checksum is declared without a bound manifest path")
        normalized_paths = []
        for value in paths:
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError("Manifest path must be a nonempty string without NUL bytes")
            path = Path(value)
            normalized_paths.append(os.path.abspath(path if path.is_absolute() else trial / path))
        if len(set(normalized_paths)) > 1:
            raise ValueError("Summary and integrity manifest paths disagree")
    except ValueError as exc:
        invalid, binding_valid = True, False
        warnings.append(f"Summary deployment binding rejected: {exc}")
    if paths and binding_valid:
        try:
            candidates = [_manifest_path(path, trial) for path in paths]
            if len(set(candidates)) != 1:
                raise ValueError("Summary and integrity manifest paths disagree")
            manifest_path = candidates[0]
            manifest = _read_metadata(manifest_path, "prediction_horizon_deployment_manifest", provenance)
            digest = provenance[-1]["sha256"]
            filename_hash = _MANIFEST_NAME.fullmatch(manifest_path.name).group(1)
            expected_hashes = [*hashes, *([filename_hash] if filename_hash else [])]
            if any(value.lower() != digest for value in expected_hashes):
                raise ValueError("Bound deployment manifest SHA256 mismatch")
            for key in ("deployment_id", "checkpoint_sha256", "task", "model_family"):
                if key in summary and key in manifest and summary[key] != manifest[key]:
                    raise ValueError(f"Summary and bound deployment manifest {key} disagree")
            if not expected_hashes:
                warnings.append("Legacy manifest has no recorded SHA256 binding; its bytes are hashed for export revalidation")
            invalid |= _declaration(manifest, manifest_path, "deployment_manifest", evidence, warnings)
        except (_UnavailableManifest, OSError) as exc:
            warnings.append(f"Bound deployment manifest unavailable and not verified: {exc}; only explicit summary metadata can establish the horizon")
        except (ValueError, TypeError, RuntimeError) as exc:
            invalid = True
            warnings.append(f"Bound deployment manifest rejected: {exc}")

    values = {row["value"] for row in evidence if row["valid"]}
    if invalid:
        result["status"] = "invalid"
    elif len(values) > 1:
        result["status"] = "conflict"
        warnings.append("Explicit prediction_horizon declarations conflict")
    elif values:
        result.update(status="verified", prediction_horizon=next(iter(values)))
    else:
        warnings.append("No explicit prediction_horizon declaration; execute_steps, rates and names are not evidence")
    return result


def filter_prediction_horizon(trial_dirs, required=50):
    """Return included absolute Paths and a complete JSON-serializable audit."""
    if type(required) is not int or required <= 0:
        raise ValueError("required prediction horizon must be a positive integer")
    included_paths = []
    audit = {"required_prediction_horizon": required, "input_count": 0, "included_count": 0,
             "included": [], "excluded": [], "provenance": [],
             "semantics": "Explicit prediction_horizon metadata only; execution steps, sampling frequency and names are never inferred"}
    for trial_dir in trial_dirs:
        source = Path(trial_dir).resolve()
        inspection = inspect_prediction_horizon(source)
        accepted = inspection["status"] == "verified" and inspection["prediction_horizon"] == required
        reason = (f"verified_prediction_horizon_{required}" if accepted else
                  f"prediction_horizon_mismatch_required_{required}_observed_{inspection['prediction_horizon']}"
                  if inspection["status"] == "verified" else f"prediction_horizon_{inspection['status']}")
        audit["included" if accepted else "excluded"].append(
            {"source": str(source), "inspection": inspection, "reason": reason})
        audit["input_count"] += 1
        if accepted:
            included_paths.append(source)
        for row in inspection["provenance"]:
            # Keep distinct hashes of the same path so later verification also
            # catches a shared manifest replaced between two inspections.
            if row not in audit["provenance"]:
                audit["provenance"].append(row)
    audit["included_count"] = len(included_paths)
    return included_paths, audit
