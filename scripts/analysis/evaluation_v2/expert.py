"""Read explicit task-demo replay sources into real, reference-ready reports.

Only the single-recording ``kaihand_tactile_episode_v1`` HDF5 schema is
supported. No simulator import, resampling, temporal alignment, phase inference
or expert-quality inference is performed. Reference membership is a separate
explicit caller declaration. Missing raw tactile remains missing, even when
the archived task outcome is successful.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

_PROJECT = Path(__file__).resolve().parents[3]
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from kaihand_tactile_env.shared.task_demo import DEFAULT_DEMO_ROOT, locate_demo_root, resolve_demo_path

import h5py
import numpy as np

from evaluation_v1 import SCHEMA_VERSION
from evaluation_v1.protocol import content_hash, make_protocol
from evaluation_v1.render import render_report
from evaluation_v1.tactile import DEFAULT_CONFIG, analyze_tactile


REPLAY_SCHEMA = "kaihand-offline-multimodal-replay-v1"
HDF_SCHEMA = "kaihand_tactile_episode_v1"
_LIMITATIONS = [
    "Reference membership is an explicit curator declaration, not inferred from successful outcomes or independently certified expert competence.",
    "Task-demo examples may be success-selected; one example per task is not an expert population or an unbiased policy benchmark.",
    "Only native overall tactile scalars are exported. Command phases are not converted into independent expected-contact conditions.",
    "Native sampling rates, missing samples and source semantics are retained; no resampling, clock warping or camera-clock substitution is used.",
    "Archived SI units and geometry are recorded claims, not independent calibration or equal-bandwidth verification.",
    "Original task outcomes are not recomputed. Missing reference compatibility remains unverified until supported by separate audited declarations.",
]


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_hash(path):
    before = _signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if before != _signature(path):
        raise ValueError(f"Reference source changed while hashing: {path}")
    return digest.hexdigest(), before


def _text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value) if isinstance(value, (str, np.str_)) else None


def _json_object(raw, where):
    if not isinstance(raw, (str, bytes)) or len(raw) > 8_000_000:
        raise ValueError(f"{where} must be a bounded JSON object")
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {where}: {value}")
    value = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _node(file, path):
    """No external/soft/virtual datasets outside the hashed observation file."""
    current = file
    for part in path.strip("/").split("/"):
        if not isinstance(current, (h5py.Group, h5py.File)):
            raise ValueError(f"Expected HDF5 group in path: {path}")
        if part not in current:
            return None
        if not isinstance(current.get(part, getlink=True), h5py.HardLink):
            raise ValueError(f"Linked HDF5 evidence is unsupported: {path}")
        current = current[part]
    if isinstance(current, h5py.Dataset) and (current.is_virtual or current.external):
        raise ValueError(f"External/virtual HDF5 evidence is unsupported: {path}")
    return current


def _array(file, path, *, numeric=True):
    dataset = _node(file, path)
    if dataset is None:
        return None
    if not isinstance(dataset, h5py.Dataset) or dataset.size > 30_000_000:
        raise ValueError(f"Unsupported or oversized dataset: {path}")
    if numeric and dataset.dtype.kind not in "fiu":
        raise ValueError(f"Expected numeric dataset: {path}")
    return np.asarray(dataset[...], dtype=float if numeric else None)


def _clock(values, label):
    if values is None or values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{label} must be a finite strictly increasing native clock")
    return values


def _optional_array(file, path, warnings):
    try:
        return _array(file, path)
    except ValueError as error:
        # Do not downgrade source-link provenance failures to a usable metric.
        if "HDF5 evidence" in str(error):
            raise
        warnings.append(str(error))
        return None


def _implementation():
    root = Path(__file__).resolve().parent.parent
    paths = [Path(__file__), root / "export_expert_reference.py",
             root / "evaluation_v1/tactile.py", root / "evaluation_v1/protocol.py"]
    files = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in paths if path.is_file()]
    resolver = _PROJECT / "src/kaihand_tactile_env/shared/task_demo.py"
    files.append({"path": str(resolver), "sha256": hashlib.sha256(resolver.read_bytes()).hexdigest()})
    return {"sha256": content_hash(files), "files": files, "numpy": np.__version__, "h5py": h5py.__version__}


def load_expert_report(replay_json: str | Path, protocol: dict | None = None) -> dict:
    """Read exactly one replay-declared, checksum-verified source recording.

    ``source_episode_index`` is an identity cross-check against the single
    recording's metadata, never an index used to slice or duplicate a file.
    Reports remain unusable for conditional comparison without separate,
    evidence-backed compatibility declarations.
    """
    protocol = make_protocol(protocol)
    implementation = _implementation()
    replay_path = resolve_demo_path(replay_json).resolve(strict=True)
    if replay_path.is_relative_to(DEFAULT_DEMO_ROOT) and locate_demo_root(replay_path) is None:
        raise ValueError("Project task_demo requires migration_manifest.json; refusing a NAS fallback")
    replay_before = _signature(replay_path)
    if replay_before[2] > 2_000_000:
        raise ValueError("Replay metadata exceeds the small-input size limit")
    replay_bytes = replay_path.read_bytes()
    if replay_before != _signature(replay_path):
        raise ValueError("Replay metadata changed during read")
    replay = _json_object(replay_bytes, "replay metadata")
    if replay.get("schema_version") != REPLAY_SCHEMA or replay.get("completed") is not True:
        raise ValueError("Only completed explicit task-demo replay metadata is supported")
    task = replay.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Replay task identity is required")
    source_name = replay.get("source_hdf5")
    if not isinstance(source_name, str) or not Path(source_name).is_absolute():
        raise ValueError("Replay source_hdf5 must be an explicit absolute local path")
    expected = replay.get("source_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Replay must declare the original HDF5 SHA-256")
    demo_root = locate_demo_root(replay_path)
    migration_row = None
    if demo_root is not None:
        resolve_demo_path(replay_path, demo_root=demo_root,
                          expected_sha256=hashlib.sha256(replay_bytes).hexdigest())
        migration_path = demo_root / "migration_manifest.json"
        migration_hash, migration_before = _file_hash(migration_path)
        migration_row = {"path": str(migration_path), "sha256": migration_hash,
                         "size": migration_before[2], "role": "task_demo_migration_manifest"}
    source = resolve_demo_path(source_name, demo_root=demo_root, expected_sha256=expected)
    if demo_root is not None and not source.is_relative_to(demo_root):
        raise ValueError("Migrated replay source has no local task-demo mapping; refusing a NAS fallback")
    source = source.resolve(strict=True)
    source_hash, source_before = _file_hash(source)
    if source_hash != expected.lower():
        raise ValueError("Replay source SHA-256 mismatch; refusing an unbound reference")
    provenance = [
        {"path": str(replay_path), "sha256": hashlib.sha256(replay_bytes).hexdigest(),
         "size": len(replay_bytes), "role": "explicit_task_demo_replay_selection"},
        {"path": source_name, "sha256": source_hash, "size": source_before[2], "role": "expert_hdf5"},
    ]
    if source != Path(source_name):
        # A migration is not a filesystem alias: the original NAS copy can
        # still exist. Keep this separate from workbench's symlink guard.
        provenance[-1]["local_path"] = str(source)
    if migration_row is not None:
        provenance.append(migration_row)
    warnings, geometry_hashes, names = [], {}, []
    force_time, state_time, normal, grid, tangent, geometry = (None,) * 6
    spatial_meta = {}
    tactile = {"schema_version": "tactile-metrics-v1", "status": "missing",
               "reason": "No interpretable native tactile_contact_force series in this recording", "scopes": {}}
    with h5py.File(source, "r") as file:
        if _text(file.attrs.get("schema_version")) != HDF_SCHEMA:
            raise ValueError("Unsupported HDF5 schema; no speculative schema conversion is performed")
        metadata = _json_object(_text(file.attrs.get("metadata_json", "{}")), "HDF5 metadata_json")
        if metadata.get("scene") != task or (metadata.get("task") is not None and metadata["task"] != task):
            raise ValueError("Replay task and HDF5 scene identity disagree or cannot be verified")
        episode_index = replay.get("source_episode_index")
        if episode_index is not None and (isinstance(episode_index, bool) or not isinstance(episode_index, int)
                                          or episode_index < 0 or metadata.get("episode_index") != episode_index):
            raise ValueError("Replay source_episode_index does not identify this single recorded episode")
        outcome_raw = _json_object(_text(file.attrs.get("outcome_json", "{}")), "HDF5 outcome_json")
        model_hash = _text(file.attrs.get("model_sha256"))
        declared_control_hz = file.attrs.get("control_hz")
        if not isinstance(declared_control_hz, (int, float, np.number)) or not np.isfinite(declared_control_hz):
            declared_control_hz = None
        else:
            declared_control_hz = float(declared_control_hz)
        raw_state_time = _array(file, "state/timestamp")
        if raw_state_time is not None:
            try:
                state_time = _clock(raw_state_time, "state/timestamp")
            except ValueError as error:
                warnings.append(str(error))
        group = _node(file, "tactile_contact_force")
        if group is not None and not isinstance(group, h5py.Group):
            raise ValueError("tactile_contact_force is not the declared HDF5 group")
        declared_force_unit = _text(group.attrs.get("force_unit")) if group else None
        declared_taxel_unit = _text(group.attrs.get("taxel_unit")) if group else None
        clock_semantics = _text(group.attrs.get("clock_semantics")) if group else None
        if group is not None:
            spatial_meta = _json_object(_text(group.attrs.get("metadata_json", "{}")), "tactile metadata_json")
            if declared_force_unit != "N":
                warnings.append("Native tactile force_unit is not explicitly N; force metrics unavailable, not converted by guessing.")
            else:
                force_time = _array(file, "tactile_contact_force/timestamp")
                normal = _array(file, "tactile_contact_force/normal_force_n")
                links = _array(file, "tactile_contact_force/link_names", numeric=False)
                try:
                    _clock(force_time, "tactile_contact_force/timestamp")
                    if normal is None or normal.ndim != 2 or normal.shape[0] != len(force_time):
                        raise ValueError("Native normal_force_n shape does not match its own clock")
                    if links is None or links.ndim != 1:
                        raise ValueError("Native link_names missing; channel identity is not invented")
                    names = [_text(value) for value in links.tolist()]
                    if len(names) != normal.shape[1] or not all(names) or len(set(names)) != len(names):
                        raise ValueError("Native link_names must identify every channel uniquely")
                    if declared_taxel_unit == "N_per_taxel" and spatial_meta.get("force_unit") == "N":
                        grid = _optional_array(file, "tactile_contact_force/normal_taxel_force_n", warnings)
                        if grid is not None and (grid.ndim != 4 or grid.shape[:2] != normal.shape or min(grid.shape[2:]) < 1):
                            warnings.append("Taxel grid shape invalid; total native normal force remains available")
                            grid = None
                        tangent = _optional_array(file, "tactile_contact_force/tangent_taxel_force_n", warnings)
                        if tangent is not None and (grid is None or tangent.shape != (*grid.shape, 2)
                                or spatial_meta.get("tangent_basis") != ["grid_col_positive", "grid_row_positive"]):
                            warnings.append("Tangent grid or recorded component basis unverified; tangent metrics unavailable")
                            tangent = None
                        geometry = _optional_array(file, "tactile_contact_force/taxel_positions_local_m", warnings)
                        if geometry is not None:
                            expected_shapes = () if grid is None else ((normal.shape[1], grid.shape[2] * grid.shape[3], 3), (*grid.shape[1:], 3))
                            if geometry.shape not in expected_shapes or not np.isfinite(geometry).all():
                                warnings.append("Archived geometry does not match taxels; spatial centers unavailable")
                                geometry = None
                    else:
                        warnings.append("Taxel force units unverified; no force-per-taxel or geometry metrics are fabricated")
                    tactile = analyze_tactile(force_time, normal, channel_names=names,
                        normal_taxel_force_n=grid, tangent_taxel_force_n=tangent, taxel_positions_local_m=geometry,
                        config={key: protocol[key] for key in DEFAULT_CONFIG})
                    if geometry is not None:
                        geometry_hashes = {name: content_hash({"coordinates_local_m": geometry[index].reshape(-1, 3).tolist(),
                                            "grid_shape": list(grid.shape[2:]), "channel": name}) for index, name in enumerate(names)}
                except ValueError as error:
                    if "HDF5 evidence" in str(error):
                        raise
                    warnings.append(str(error))
                    force_time = normal = grid = tangent = geometry = None
                    names = []
        source_schema = _text(file.attrs.get("schema_version"))
    if source_before != _signature(source) or replay_before != _signature(replay_path):
        raise ValueError("Reference source changed while evaluating; no report is valid")
    if migration_row is not None and _file_hash(migration_path)[0] != migration_row["sha256"]:
        raise ValueError("Task-demo migration manifest changed while evaluating")
    if _implementation() != implementation:
        raise ValueError("Expert evaluator source changed during evaluation")
    labels = [{"field": key, "value": outcome_raw[key]} for key in ("success", "task_success") if type(outcome_raw.get(key)) is bool]
    conflict = len({row["value"] for row in labels}) > 1
    recorded_success = labels[0]["value"] if labels and not conflict else None
    status = "pass" if recorded_success is True else "fail" if recorded_success is False else "unknown"
    cadence = float(np.median(np.diff(force_time))) if force_time is not None and len(force_time) > 1 else None
    trial_id = f"expert:{task}:{source_hash[:20]}"
    state_span = float(state_time[-1] - state_time[0]) if state_time is not None else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "trial": {"trial_id": trial_id, "task": task, "source": source_name, "seed": metadata.get("seed"),
                  "cohort": {"task": task, "controller": "recorded_task_demo_not_independently_certified",
                             "control_hz": declared_control_hz, "tactile_sample_interval_s": cadence,
                             "tactile_channels": names, "formal_metrics_valid": None, "diagnostic_only": None}},
        "protocol": protocol, "protocol_sha256": content_hash(protocol), "implementation": implementation,
        "provenance": provenance, "warnings": warnings,
        "outcome": {"status": status, "recorded_success": recorded_success, "source": "HDF5 root attrs.outcome_json",
                    "scope": "recorded_task_demo_monitor_result_not_independent_recertification",
                    "reason": "Conflicting success labels" if conflict else "No explicit boolean task outcome" if not labels else "Archived explicit outcome label",
                    "labels": labels, "formal_metrics_valid": None, "diagnostic_only": None,
                    "termination_reason": outcome_raw.get("termination_reason", outcome_raw.get("stop_reason"))},
        "efficiency": {"simulation_duration_s": state_span, "wall_duration_s": None,
                       "recorded_time_budget_s": None, "scope": "observed native state-clock span; no inferred wall clock"},
        "safety": {"status": "unknown", "scope": "no independent safety recertification", "checks": [], "enforcement": "unknown"},
        "task_specific": {"plugin": "explicit_task_demo_import", "partial_state": {"archived_outcome": outcome_raw},
                          "predicates": [], "limitations": ["Archived terminal fields are not recomputed predicates"]},
        "events": [], "scope_intervals": [], "tactile": tactile,
        "evidence_quality": {"force_unit": declared_force_unit, "taxel_geometry_status": "archived" if geometry is not None else "missing",
                             "calibration_verification": "not_performed", "task_tactile_clock_mapping": "not_inferred"},
        "reference_import": {"status": "imported", "source_schema": source_schema, "source_sha256_verified": True,
            "source_episode_index": episode_index, "recording_metadata": metadata, "model_sha256": model_hash,
            "sensor_metadata": spatial_meta, "geometry_sha256_by_channel": geometry_hashes,
            "native_tactile_sample_count": len(force_time) if force_time is not None else 0,
            "native_tactile_cadence_hz": 1 / cadence if cadence else None, "clock_semantics": clock_semantics},
        "compatibility_hints": {"status": "unverified", "task_definition_id": None,
            "sensor_cadence_hz": 1 / cadence if cadence else None, "force_unit": declared_force_unit,
            "sensor_semantics_id": None,
            "channels": {name: {"status": "archived" if name in geometry_hashes else "missing",
                                "geometry_sha256": geometry_hashes.get(name)} for name in names},
            "phase_conditions": {}, "reason": "Independent task/phase and calibration/bandwidth compatibility must be explicitly audited; overall is not auto-certified"},
        "limitations": list(_LIMITATIONS),
    }
    if source != Path(source_name):
        report["trial"]["resolved_source"] = str(source)
    content_hash(report)
    return report


def export_reference(replay_jsons, output, *, reference_id, declared_by, selection_rule,
                     selection_evidence, declare_expert_reference=False, protocol=None):
    """Create a new reference package from distinct explicitly selected recordings."""
    if declare_expert_reference is not True:
        raise ValueError("Explicit --declare-expert-reference is required; success does not imply expert membership")
    for name, value in (("reference_id", reference_id), ("declared_by", declared_by),
                        ("selection_rule", selection_rule), ("selection_evidence", selection_evidence)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Explicit {name} is required")
    requested = list(replay_jsons)
    paths = [resolve_demo_path(path).resolve(strict=True) for path in requested]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply distinct task-demo replay metadata paths")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Reference output already exists: {destination}")
    reports = [load_expert_report(path, protocol) for path in requested]
    sources = [report["trial"]["source"] for report in reports]
    hashes = [next(row["sha256"] for row in report["provenance"] if row["role"] == "expert_hdf5") for report in reports]
    if len(set(sources)) != len(sources) or len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate source recording or observation SHA cannot be counted as independent reference episodes")
    original_dirs = ([path.parent for path in paths] + [Path(source).parent for source in sources]
                     + [Path(report["trial"].get("resolved_source", report["trial"]["source"])).parent
                        for report in reports])
    if any(destination == parent or destination.is_relative_to(parent) for parent in original_dirs):
        raise ValueError("Reference output must be outside original metadata/HDF5 directories")
    manifest = {"schema_version": "conditional-reference-manifest-v1", "reference_id": reference_id,
                "role": "expert_reference", "declared_by": declared_by, "selection_rule": selection_rule,
                "evidence": selection_evidence, "membership_semantics": "Explicit selection, not quality certification; no outcome filtering by this exporter",
                "members": [{"trial_id": report["trial"]["trial_id"], "source": report["trial"]["source"],
                             "observation_sha256": digest} for report, digest in zip(reports, hashes)]}
    rows = []
    for index, report in enumerate(reports):
        slug = f"{index:03d}_{re.sub(r'[^A-Za-z0-9_-]', '_', report['trial']['task'])}_{hashes[index][:12]}"
        report["report_path"] = f"{slug}/report.json"
        rows.append({"trial_id": report["trial"]["trial_id"], "task": report["trial"]["task"],
                     "source": report["trial"]["source"], "outcome": report["outcome"]["status"],
                     "report_path": report["report_path"], "audit_path": f"{slug}/audit.html",
                     "tactile_status": report["tactile"].get("status") or report["tactile"].get("scopes", {}).get("overall", {}).get("status", "missing"),
                     "compatibility_status": report["compatibility_hints"]["status"]})
    counts = Counter(row["outcome"] for row in rows)
    result = {"schema_version": "expert-reference-export-v1", "status": "exported",
              "reference_id": reference_id, "reference_manifest_sha256": content_hash(manifest),
              "independent_recording_count": len(reports), "outcome_counts": {key: counts[key] for key in ("pass", "fail", "unknown")},
              "recording_count_semantics": "Distinct source files and SHA-256 values; statistical independence is not independently certified",
              "trials": rows, "comparison_status": "unavailable_until_compatibility_is_audited", "limitations": list(_LIMITATIONS)}
    # All inputs and pages are checked before creating the new destination.
    pages = [render_report(report) for report in reports]
    for report in reports:
        for row in report["provenance"]:
            resolved = Path(row.get("local_path", row["path"]))
            actual, _ = _file_hash(resolved)
            if actual != row["sha256"]:
                raise ValueError("Reference input changed before export")
        if report["implementation"] != _implementation():
            raise ValueError("Expert evaluator implementation changed before export")
    destination.mkdir(parents=True, exist_ok=False)
    def write_json(path, value):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    for report, page in zip(reports, pages):
        path = destination / report["report_path"]
        path.parent.mkdir()
        write_json(path, report)
        with (path.parent / "audit.html").open("x", encoding="utf-8") as stream:
            stream.write(page)
    write_json(destination / "reference_manifest.json", manifest)
    write_json(destination / "reference_index.json", result)
    return result
