"""Versioned, opt-in evaluation observations; independent of simulator imports.

Only ``evaluation_evidence.meta.json`` and ``evaluation_evidence.jsonl`` inside
the selected trial are read. Missing files are not reconstructed from policy
stages. Recorder helpers only serialize observations supplied by their caller;
they do not certify their accuracy, independence, calibration, or completeness.

Metadata declares task, source_kind, clock, coordinate_frames, bodies,
contact_pairs, task_fields, task_config, geometry, effects, and completeness.
Each JSONL record has time_s and valid; optional contacts, objects, goals,
task_values and effects use the declared names. SI units appear in field names.
Contact relative velocity means tangential velocity *at the contact point* in
the pair's declared basis, not an arbitrary difference of body-center speeds.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "evaluation-evidence-v1"
META_NAME = "evaluation_evidence.meta.json"
RECORDS_NAME = "evaluation_evidence.jsonl"
SOURCE_KINDS = {"simulation_oracle", "hardware_observation", "synthetic_test", "legacy_simulation_record"}
FIELD_KINDS = {"measurement", "simulation_oracle", "controller_declaration", "monitor_predicate"}


def _finite_json(value):
    # Round-trip also removes aliases to caller-owned mutable structures.
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Evidence must contain only finite JSON values") from error


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _keys(value, allowed, where):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")


def _text(value, where):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{where} must be a nonempty string")
    return value


def _number(value, where, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} must be finite and numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return float(value)


def _boolean(value, where, *, nullable=False):
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"{where} must be boolean" + (" or null" if nullable else ""))


def _vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must have {length} components")
    for component in value:
        _number(component, where)


def _label(value, where):
    _object(value, where)
    _keys(value, {"value", "source", "independent"}, where)
    _boolean(value.get("value"), where + ".value", nullable=True)
    _boolean(value.get("independent"), where + ".independent")
    _text(value.get("source"), where + ".source")


def make_metadata(task: str, *, clock_id: str, source_kind: str,
                  nominal_interval_s: float | None = None) -> dict:
    """An explicit empty contract for a future recorder; no invented channels."""
    return {
        "schema_version": SCHEMA_VERSION, "task": task, "source_kind": source_kind,
        "clock": {"id": clock_id, "unit": "s", "sampling_mode": "recorded_samples",
                  "nominal_interval_s": nominal_interval_s, "max_gap_factor": 1.5,
                  "aligned_to_trial": False},
        "coordinate_frames": {}, "bodies": {}, "contact_pairs": {},
        "task_fields": {}, "task_config": {}, "geometry": {}, "effects": {},
        "complete": False,
        "calibration": {"status": "unverified", "source": "not supplied"},
    }


def _validate_metadata(meta):
    _object(meta, "metadata")
    _keys(meta, {"schema_version", "task", "source_kind", "clock", "coordinate_frames", "bodies",
                 "contact_pairs", "task_fields", "task_config", "geometry", "effects", "complete",
                 "calibration", "producer", "validity_source"}, "metadata")
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation evidence schema_version")
    _text(meta.get("task"), "metadata.task")
    if meta.get("source_kind") not in SOURCE_KINDS:
        raise ValueError("Unknown evidence source_kind")
    _boolean(meta.get("complete"), "metadata.complete")
    clock = _object(meta.get("clock"), "clock")
    _keys(clock, {"id", "unit", "sampling_mode", "nominal_interval_s", "max_gap_factor",
                  "aligned_to_trial", "alignment_evidence"}, "clock")
    _text(clock.get("id"), "clock.id")
    _text(clock.get("sampling_mode"), "clock.sampling_mode")
    if clock.get("unit") != "s":
        raise ValueError("Evidence clocks must be declared in seconds")
    if clock.get("nominal_interval_s") is not None and _number(clock["nominal_interval_s"], "clock.nominal_interval_s", minimum=0) <= 0:
        raise ValueError("nominal_interval_s must be positive")
    _number(clock.get("max_gap_factor", 1.5), "clock.max_gap_factor", minimum=1)
    _boolean(clock.get("aligned_to_trial", False), "clock.aligned_to_trial")
    if clock.get("aligned_to_trial"):
        _text(clock.get("alignment_evidence"), "clock.alignment_evidence")
    for name in ("coordinate_frames", "bodies", "contact_pairs", "task_fields", "task_config", "geometry", "effects"):
        _object(meta.get(name, {}), name)
    for name, description in meta.get("coordinate_frames", {}).items():
        _text(name, "coordinate frame name"); _text(description, "coordinate frame description")
    for name, description in meta.get("bodies", {}).items():
        _text(name, "body name"); _text(description, "body description")
    calibration = _object(meta.get("calibration", {}), "calibration")
    _keys(calibration, {"status", "source"}, "calibration")
    if calibration.get("status", "unverified") not in {"verified", "unverified", "not_applicable"}:
        raise ValueError("Unknown calibration status")
    if calibration.get("status") == "verified":
        _text(calibration.get("source"), "calibration.source")
    for name, descriptor in meta.get("task_fields", {}).items():
        _text(name, "task field name"); _object(descriptor, "task field descriptor")
        _keys(descriptor, {"source", "kind", "unit"}, "task field descriptor")
        _text(descriptor.get("source"), "task field source")
        _text(descriptor.get("unit"), "task field unit")
        if descriptor.get("kind") not in FIELD_KINDS:
            raise ValueError("Unknown task field evidence kind")
    for name, pair in meta.get("contact_pairs", {}).items():
        _text(name, "contact pair name"); _object(pair, "contact pair")
        _keys(pair, {"body_a", "body_b", "target", "role", "measurement_source", "allowed_relative_motion",
                     "tangent_basis", "thresholds", "threshold_source", "contact_semantics"}, "contact pair")
        for side in ("body_a", "body_b"):
            if pair.get(side) not in meta.get("bodies", {}):
                raise ValueError(f"Contact pair {name} has an undeclared {side}")
        if pair["body_a"] == pair["body_b"]:
            raise ValueError("A contact pair must identify two different bodies")
        _boolean(pair.get("target"), "contact pair target")
        _text(pair.get("measurement_source"), "contact measurement_source")
        if pair.get("role") not in {"retention", "working_contact", "support", "other"}:
            raise ValueError("Unknown contact pair role")
        if pair.get("allowed_relative_motion", "unknown") not in {"none", "sliding", "rolling", "any", "unknown"}:
            raise ValueError("Unknown allowed_relative_motion")
        if "tangent_basis" in pair:
            _text(pair["tangent_basis"], "pair tangent_basis")
        thresholds = _object(pair.get("thresholds", {}), "thresholds")
        _keys(thresholds, {"contact_on_n", "max_normal_force_n", "max_tangent_force_n", "slip_speed_m_s"}, "thresholds")
        for key, value in thresholds.items():
            _number(value, "thresholds." + key, minimum=0)
        if thresholds:
            _text(pair.get("threshold_source"), "pair threshold_source")
    for name, descriptor in meta.get("effects", {}).items():
        _text(name, "effect name"); _object(descriptor, "effect descriptor")
        _keys(descriptor, {"source", "unit", "independent", "target_min", "target_max", "contact_pairs"}, "effect descriptor")
        _text(descriptor.get("source"), "effect source"); _text(descriptor.get("unit"), "effect unit")
        _boolean(descriptor.get("independent"), "effect independent")
        for boundary in ("target_min", "target_max"):
            if boundary in descriptor:
                _number(descriptor[boundary], "effect " + boundary)
        if descriptor.get("target_min", -math.inf) > descriptor.get("target_max", math.inf):
            raise ValueError("Effect target_min exceeds target_max")
        associations = descriptor.get("contact_pairs", [])
        if not isinstance(associations, list) or any(pair not in meta.get("contact_pairs", {}) for pair in associations):
            raise ValueError("Effect associations must name declared contact pairs")


def _validate_record(row, meta, index):
    where = f"record {index}"
    _object(row, where)
    _keys(row, {"time_s", "valid", "validity_reason", "contacts", "objects", "goals", "task_values", "effects"}, where)
    _number(row.get("time_s"), where + ".time_s")
    _boolean(row.get("valid"), where + ".valid")
    for name in ("contacts", "objects", "goals", "task_values", "effects"):
        _object(row.get(name, {}), where + "." + name)
    for kind in ("objects", "goals"):
        for name, pose in row.get(kind, {}).items():
            if kind == "objects" and name not in meta.get("bodies", {}):
                raise ValueError("Object pose names an undeclared body")
            _text(name, "pose name"); _object(pose, "pose")
            _keys(pose, {"position_m", "quaternion_wxyz", "frame", "valid", "source"}, "pose")
            _boolean(pose.get("valid"), "pose.valid")
            _text(pose.get("source"), "pose.source")
            if pose.get("frame") not in meta.get("coordinate_frames", {}):
                raise ValueError("Pose coordinate frame is undeclared")
            for key, size in (("position_m", 3), ("quaternion_wxyz", 4)):
                if pose.get(key) is not None:
                    _vector(pose[key], size, "pose." + key)
            if pose.get("quaternion_wxyz") is not None:
                norm = math.sqrt(sum(value * value for value in pose["quaternion_wxyz"]))
                if not math.isclose(norm, 1., rel_tol=0, abs_tol=1e-5):
                    raise ValueError("Pose quaternion must already be normalized, not silently corrected")
    for name, value in row.get("task_values", {}).items():
        if name not in meta.get("task_fields", {}):
            raise ValueError(f"Undeclared task primitive: {name}")
        def primitive(item):
            if isinstance(item, list):
                for part in item: primitive(part)
            elif item is not None and type(item) is not bool:
                _number(item, "task primitive")
        primitive(value)
    for name, value in row.get("effects", {}).items():
        if name not in meta.get("effects", {}):
            raise ValueError(f"Undeclared effect: {name}")
        if value is not None: _number(value, "effect value")
    for name, contact in row.get("contacts", {}).items():
        if name not in meta.get("contact_pairs", {}):
            raise ValueError(f"Undeclared contact pair: {name}")
        _object(contact, "contact")
        _keys(contact, {"valid", "contact_present", "normal_force_n", "tangent_force_n",
                        "tangential_relative_velocity_m_s", "expected_contact", "slip", "impact",
                        "normal_impulse_ns", "saturated"}, "contact")
        _boolean(contact.get("valid"), "contact.valid")
        for key in ("contact_present", "saturated"):
            if key in contact: _boolean(contact[key], "contact." + key, nullable=True)
        for key in ("normal_force_n", "normal_impulse_ns"):
            if contact.get(key) is not None: _number(contact[key], "contact." + key, minimum=0)
        for key in ("tangent_force_n", "tangential_relative_velocity_m_s"):
            if contact.get(key) is not None:
                _vector(contact[key], 2, "contact." + key)
                if not meta["contact_pairs"][name].get("tangent_basis"):
                    raise ValueError("Tangential vectors require a declared contact-pair basis")
        for key in ("expected_contact", "slip", "impact"):
            if key in contact: _label(contact[key], "contact." + key)


def _sampling(meta, records):
    intervals = [b["time_s"] - a["time_s"] for a, b in zip(records, records[1:])]
    ordered = sorted(intervals)
    median = ((ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2) if ordered else None
    declared = meta.get("clock", {}).get("nominal_interval_s")
    nominal = declared if declared is not None else median
    limit = nominal * meta.get("clock", {}).get("max_gap_factor", 1.5) if nominal else None
    gaps = [duration > limit + 1e-10 for duration in intervals] if limit is not None else []
    return {"mode": meta.get("clock", {}).get("sampling_mode"), "nominal_interval_s": nominal,
            "nominal_interval_source": "declared" if declared is not None else "observed_median",
            "observed_median_interval_s": median, "max_gap_s": max(intervals) if intervals else None,
            "gap_limit_s": limit, "gap_after": gaps + ([False] if records else []),
            "resolution": "recorded sample cadence; completeness does not establish continuous physics or hidden-event coverage"}


@dataclass
class EvidenceBundle:
    status: str
    metadata: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def time_s(self):
        return [row["time_s"] for row in self.records]

    @property
    def sampling(self):
        return _sampling(self.metadata, self.records)

    def task_evidence(self):
        source = next((row["path"] for row in self.provenance if row.get("role") == "evaluation_evidence_records"), "in_memory_evidence")
        return {"task": self.metadata.get("task"), "task_samples": [
                    {"time_s": row["time_s"], "values": deepcopy(row.get("task_values", {})),
                     "valid": row["valid"], "source": f"{source}:{index + 1}"}
                    for index, row in enumerate(self.records)],
                "sampling": self.sampling,
                "complete": bool(self.records and self.metadata.get("complete") and all(row["valid"] for row in self.records)
                                 and not any(self.sampling["gap_after"])),
                "contract": deepcopy(self.metadata.get("task_config", {})),
                "task_config": deepcopy(self.metadata.get("task_config", {})),
                "field_sources": deepcopy(self.metadata.get("task_fields", {})),
                "provenance": deepcopy(self.provenance), "source_kind": self.metadata.get("source_kind")}

    @property
    def summary(self):
        return describe_evidence(self)


def validate_evidence(metadata: dict, records: list[dict]) -> EvidenceBundle:
    meta, rows = _finite_json(metadata), _finite_json(records)
    _validate_metadata(meta)
    if not isinstance(rows, list):
        raise ValueError("Evidence records must be a list")
    for index, row in enumerate(rows):
        _validate_record(row, meta, index)
        if index and (row["time_s"] <= rows[index - 1]["time_s"] or
                      not math.isfinite(row["time_s"] - rows[index - 1]["time_s"])):
            raise ValueError("Evidence time_s must be strictly increasing; no sorting or duplicate dropping")
    warnings = ["Evidence labels and calibration assertions are recorded claims, not independently authenticated."]
    sampling = _sampling(meta, rows)
    if any(sampling["gap_after"]): warnings.append("Sampling gaps remain unknown and are never interpolated.")
    return EvidenceBundle("available" if rows else "empty", meta, rows, warnings=warnings)


def describe_evidence(bundle: EvidenceBundle) -> dict:
    return {"schema_version": SCHEMA_VERSION, "status": bundle.status,
            "task": bundle.metadata.get("task"), "source_kind": bundle.metadata.get("source_kind"),
            "sample_count": len(bundle.records), "time_start_s": bundle.time_s[0] if bundle.records else None,
            "time_end_s": bundle.time_s[-1] if bundle.records else None,
            "clock": deepcopy(bundle.metadata.get("clock", {})), "sampling": bundle.sampling,
            "contact_pair_ids": list(bundle.metadata.get("contact_pairs", {})),
            "recorded_complete": bundle.metadata.get("complete"),
            "calibration": deepcopy(bundle.metadata.get("calibration", {})),
            "provenance": deepcopy(bundle.provenance), "warnings": list(bundle.warnings)}


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _paths(trial_dir):
    root = Path(trial_dir).resolve(strict=True)
    if not root.is_dir(): raise ValueError("Evidence trial root must be a directory")
    paths = [root / META_NAME, root / RECORDS_NAME]
    for path in paths:
        if path.is_symlink(): raise ValueError("Evidence sidecar symlinks are not permitted")
        if path.exists() and (not path.is_file() or not path.resolve().is_relative_to(root)):
            raise ValueError("Evidence sidecars must be regular files inside the selected trial")
    return paths


def load_evidence(trial_dir) -> EvidenceBundle:
    meta_path, records_path = _paths(trial_dir)
    exists = [path.exists() for path in (meta_path, records_path)]
    if not any(exists):
        return EvidenceBundle("missing", warnings=["No versioned independent evidence sidecar; missing contact identity, expected contact, or slip is not reconstructed from tactile changes."])
    if not all(exists): raise ValueError("Evidence sidecar is incomplete: metadata and JSONL are both required")
    signatures, contents = [], []
    for path in (meta_path, records_path):
        before = _signature(path); data = path.read_bytes()
        if before != _signature(path) or len(data) != before[2]: raise ValueError("Evidence changed during reading")
        signatures.append(before); contents.append(data)
    def reject_constant(value): raise ValueError(f"Nonfinite JSON literal: {value}")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError(f"Duplicate evidence JSON key: {key}")
            result[key] = value
        return result
    meta = json.loads(contents[0], parse_constant=reject_constant, object_pairs_hook=unique_object)
    records = []
    for index, line in enumerate(contents[1].decode("utf-8").splitlines(), 1):
        if not line.strip(): raise ValueError(f"Blank evidence record at line {index}; record indices must remain auditable")
        records.append(json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object))
    bundle = validate_evidence(meta, records)
    for path, data, signature, role in zip((meta_path, records_path), contents, signatures,
                                         ("evaluation_evidence_metadata", "evaluation_evidence_records")):
        if _signature(path) != signature: raise ValueError("Evidence changed during validation")
        bundle.provenance.append({"path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "role": role})
    return bundle


def write_evidence(trial_dir, metadata, records) -> EvidenceBundle:
    """Serialize a completed supplied capture; never replace an existing sidecar."""
    bundle = validate_evidence(metadata, list(records))
    paths = _paths(trial_dir)
    if any(path.exists() for path in paths): raise FileExistsError("Refusing to overwrite evidence sidecars")
    contents = [json.dumps(bundle.metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in bundle.records)]
    # Exclusive creates mean a failed capture remains visibly incomplete and
    # can never overwrite an original. No cleanup deletes user-owned files.
    for path, content in zip(paths, contents):
        with path.open("x", encoding="utf-8") as stream: stream.write(content)
    return load_evidence(trial_dir)


class EvidenceRecorder:
    """Validate caller-supplied observations, optionally streaming to new files.

    Without trial_dir, records are buffered and ``write`` exports them. With a
    trial_dir, each ``record`` is flushed immediately and metadata.complete is
    false until an explicit ``close(complete=True)``. No simulator is imported,
    no fields are fabricated, and existing sidecars are never replaced.
    """
    def __init__(self, metadata, *, trial_dir=None):
        self.metadata = validate_evidence(metadata, []).metadata
        self.records = []
        self._last_time = None
        self._count = 0
        self._streams = None
        self._closed = False
        self._root = None
        if trial_dir is not None:
            paths = _paths(trial_dir)
            if any(path.exists() for path in paths): raise FileExistsError("Refusing to overwrite evidence sidecars")
            self.metadata["complete"] = False
            meta_stream = paths[0].open("x", encoding="utf-8")
            try:
                json.dump(self.metadata, meta_stream, ensure_ascii=False, allow_nan=False)
                meta_stream.write("\n"); meta_stream.flush()
                record_stream = paths[1].open("x", encoding="utf-8")
            except BaseException:
                meta_stream.close()
                raise
            self._streams = (meta_stream, record_stream)
            self._root = paths[0].parent

    def record(self, row):
        if self._closed: raise ValueError("Recorder is closed")
        copied = _finite_json(row)
        _validate_record(copied, self.metadata, self._count)
        if self._last_time is not None and (copied["time_s"] <= self._last_time or
                                            not math.isfinite(copied["time_s"] - self._last_time)):
            raise ValueError("Recorder timestamps must be strictly increasing")
        if self._streams:
            stream = self._streams[1]
            stream.write(json.dumps(copied, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
        else:
            self.records.append(copied)
        self._last_time = copied["time_s"]
        self._count += 1

    def write(self, trial_dir):
        if self._streams: raise ValueError("Streaming recorder already owns its output; use close")
        return write_evidence(trial_dir, self.metadata, self.records)

    def close(self, *, complete=False):
        _boolean(complete, "recorder complete")
        if self._closed: return None
        self._closed = True
        if self._streams:
            metadata_stream, record_stream = self._streams
            try:
                record_stream.flush(); record_stream.close()
                # This is the same exclusively-created open file descriptor,
                # never an arbitrary path reopened for overwrite.
                self.metadata["complete"] = complete
                metadata_stream.seek(0)
                json.dump(self.metadata, metadata_stream, ensure_ascii=False, allow_nan=False)
                metadata_stream.write("\n"); metadata_stream.truncate(); metadata_stream.flush()
            finally:
                record_stream.close(); metadata_stream.close()
            return load_evidence(self._root)
        self.metadata["complete"] = complete
        return validate_evidence(self.metadata, self.records)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(complete=False)


def adapt_legacy_evidence(trial, *, contact_threshold_n=None) -> EvidenceBundle:
    """Optional narrow adapter for genuinely named whiteboard measurements.

    Other archived tasks do not name the bodies under each fingertip, so their
    taxel forces cannot establish target identity. No legacy controller phase
    becomes an expected-contact mask. The caller must explicitly supply any
    desired diagnostic contact threshold; it is not a safety threshold.
    """
    if not trial.task.startswith("whiteboard-wipe") or not trial.control_rows:
        return EvidenceBundle("missing", warnings=["Legacy archive does not provide a supported named contact-pair measurement."])
    if not trial.task_tactile_clock_aligned:
        return EvidenceBundle("missing", warnings=["Legacy task clock is unverified; no contact-pair adapter was aligned."])
    meta = make_metadata(trial.task, clock_id="archived_simulation_time", source_kind="legacy_simulation_record")
    meta["clock"].update(aligned_to_trial=True, alignment_evidence="V1 adapter verified common simulation clock",
                         sampling_mode="archived_task_records_not_physics_steps")
    meta["bodies"] = {"eraser": "whiteboard task eraser", "board": "whiteboard task board"}
    pair = {"body_a": "eraser", "body_b": "board", "target": True, "role": "working_contact",
            "allowed_relative_motion": "sliding", "measurement_source": "archived task.board_normal_force_n"}
    if contact_threshold_n is not None:
        _number(contact_threshold_n, "contact_threshold_n", minimum=0)
        pair.update(thresholds={"contact_on_n": contact_threshold_n}, threshold_source="explicit caller diagnostic threshold; not safety/effectiveness")
    meta["contact_pairs"] = {"eraser_board": pair}
    meta["effects"] = {"cleaned_fraction": {"source": "archived simulated ink-cell progress", "unit": "fraction",
                                             "independent": True, "contact_pairs": ["eraser_board"]}}
    records = []
    for raw in trial.control_rows:
        task = raw.get("task") if isinstance(raw.get("task"), dict) else {}
        time = raw.get("sim_time")
        try: _number(time, "legacy sim_time")
        except ValueError:
            return EvidenceBundle("missing", warnings=["Legacy task timestamps are invalid; no samples were silently dropped."])
        contact = {"valid": True}
        force = task.get("board_normal_force_n")
        if type(force) in (int, float) and math.isfinite(force) and force >= 0: contact["normal_force_n"] = force
        effect = task.get("cleaned_fraction")
        effect = effect if type(effect) in (int, float) and math.isfinite(effect) and 0 <= effect <= 1 else None
        records.append({"time_s": time, "valid": True, "contacts": {"eraser_board": contact},
                        "effects": {"cleaned_fraction": effect}})
    bundle = validate_evidence(meta, records)
    bundle.status = "legacy_partial"
    bundle.provenance = deepcopy(trial.provenance)
    bundle.warnings.append("Narrow legacy adapter exposes named force/progress only; independent expectation, slip, geometry and calibrated hardware remain unavailable.")
    return bundle
