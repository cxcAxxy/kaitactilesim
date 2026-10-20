"""Read-only generic H50 case assembly using existing numerical definitions.

Root and review/ recorder layouts are explicit alternatives. No simulation,
interpolation, copied-source staging, or monkeypatch of the older importers.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np

from evaluation_progress_experiment.adapters import _read, _parse, _path, _clock, _SHA
from evaluation_tactile_load_extension.data import build_load_series, _read_trace, _member, _FT_FORMULAS, _LABELS
from evaluation_v1.adapters import TrialData, InputChangedError, _assign_force_arrays
from evaluation_v1.core import _aligned_scopes
from evaluation_v1.protocol import make_protocol, content_hash
from evaluation_v1.tasks import build_task_report
from evaluation_v1.tactile import analyze_tactile, DEFAULT_CONFIG
from evaluation_v1.visualization import build_visualization, _spatial
from evaluation_v2.horizon import inspect_prediction_horizon
from . import SCHEMA_VERSION
from .export import _implementation, _rows, _verify
from .diagnostic_plots import load_diagnostic_plots


MAX_LOG_BYTES = 128 * 1024**2
MAX_LOG_ROWS = 250_000
MAX_TRACE_BYTES = 256 * 1024**2
_ACTIVE = {"running", "queued", "pending", "starting", "in_progress"}
_REVIEW_SCHEMAS = {"kaihand-policy-evaluation-review-v2", "kaihand-policy-evaluation-review-v3"}
_CACHED_TIMING = "same cached FK and solver state for both cameras and tactile; recorder never steps/forwards physics"
# Read-only source audit: recorder renders cached data and labels it with
# observation_time; poker's observation_time is the absolute pre-step time.
_POSE_CLOCK_SOURCES = {
    "src/kaihand_tactile_env/shared/evaluation_video.py": "c31e5590abfa93e0cb6052b39db4b92e334771c55759f054e474ac75060a862b",
    "src/kaihand_tactile_env/tasks/poker_draw/mid_full.py": "96ab0062316e336c3ea6d822032ac0c9c83e1dc9c0c757a18bd0ff7246ac46a1",
    "src/kaihand_tactile_env/shared/evaluation_comparison.py": "bc78a63a95965b8ca5aa61a899827bbd8eb9ee4e248863febe20fcf417d16752",
    "scripts/workcell/run_poker_pi05_policy.py": "d114a95bf81ea9241bc1a5f49ba2edcd1a5ede003b620b20ead202a9625d52af",
}
# Audited 2026-10-15 against the frozen poker_fix_20261014_152500 sources.
# The staged runner adds linear actions and a read-only observer; the unchanged
# MidForcePokerSimulation still records pre-step observation_time and the
# unchanged recorder renders that cached state. StagedMetricView only snapshots
# the observer. Require this COMPLETE combination, never mix independent SHA
# allowlists. This certifies recorded RGB timestamps, not physical Fn/RGB phase.
_STAGED_POSE_CLOCK_SOURCES = {
    **_POSE_CLOCK_SOURCES,
    "scripts/workcell/run_poker_pi05_policy.py": "f1115d6e681563e152896590a2c1f81724c66fd7f99f589f82940f00303b83e6",
    "src/kaihand_tactile_env/shared/staged_evaluation.py": "b3388ac64cda03f792f0c9726e3b7bf4a80de60f8ec935aa62ba28e343b90497",
    "src/kaihand_tactile_env/shared/staged_recording.py": "5fb14c8dfbf7d415b7262c27ced1f38fed6fbbb96917a55448be67a96c6bf190",
}
_POSE_CLOCK_SOURCE_SETS = (_POSE_CLOCK_SOURCES, _STAGED_POSE_CLOCK_SOURCES)


def _read_bound(path, trial, role, *, limit=MAX_LOG_BYTES, collect=False):
    path = _path(path)
    try:
        raw, sha, size, _ = _read(path, limit=limit, collect=collect)
    except ValueError as error:
        if "changed while reading" in str(error):
            raise InputChangedError(str(error)) from error
        raise
    previous = next((r for r in trial.provenance if r["path"] == str(path)), None)
    row = {"path": str(path), "sha256": sha, "size": size, "role": role}
    if previous and (previous["sha256"] != sha or previous.get("size", size) != size):
        raise InputChangedError("Source changed between case assembly reads")
    if previous is None:
        trial.provenance.append(row)
    return raw, row


def _json_bound(path, trial, role):
    return _parse(_read_bound(path, trial, role, limit=2 * 1024**2, collect=True)[0])


def _jsonl(path, trial, role, *, frames=False, keep_taxels=False):
    raw, _ = _read_bound(path, trial, role, collect=True)
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        if len(rows) >= MAX_LOG_ROWS or len(line) > 2 * 1024**2:
            raise ValueError("Oversized recorder row collection")
        row = _parse(line)
        row["_source"] = f"{path}:{index}"
        if frames:
            keys = ("frame", "simulation_time_s", "camera_pose_time_s", "tactile_time_s", "video_time_s",
                    "task_metrics", "phase", "fingertip_order", "_source")
            if keep_taxels:
                keys += ("normal_taxel_force_n", "tangent_taxel_force_n")
            row = {key: row[key] for key in keys if key in row}
            row.update(sim_time=row.get("simulation_time_s"), task=row.get("task_metrics", {}))
        rows.append(row)
    return rows


def _layout_file(root, name):
    found = [path for path in (root / name, root / "review" / name) if path.exists()]
    if len(found) > 1:
        raise ValueError(f"Ambiguous root/review recorder artifact: {name}")
    return _path(found[0]) if found else None


def _comparison_binding(trial, trace):
    video = trial.summary.get("video")
    comparison = video.get("comparison_plots") if isinstance(video, dict) else None
    if not isinstance(comparison, dict):
        return None
    declared = comparison.get("artifacts", {}).get("rollout_trace")
    if declared is not None:
        if (not isinstance(declared, str) or not Path(declared).is_absolute()
                or Path(declared).is_symlink() or Path(declared).resolve(strict=True) != trace):
            raise ValueError("Declared native rollout trace belongs to another source")
    if comparison.get("schema") != "kaihand-evaluation-comparison-v1" or declared is None:
        return None
    return comparison


def _load_native(trial):
    trace = _layout_file(trial.root, "evaluation_rollout_trace.npz")
    if trace is None:
        return None, None
    _read_bound(trace, trial, "native_control_cadence_tactile", limit=MAX_TRACE_BYTES)
    comparison = _comparison_binding(trial, trace)
    try:
        time, elapsed, names, normal, tangent = _read_trace(trace)
        with zipfile.ZipFile(trace) as archive:
            taxels = _member(archive, "normal_taxel_force_n", shape_tail=(5, 7, 5))
            tangential = _member(archive, "tangent_taxel_force_n", shape_tail=(5, 7, 5, 2))
        if taxels is not None and len(taxels) != len(time):
            trial.warnings.append("NPZ taxel sample count differs from native Fn; no cadence guessed, padded, or resampled")
            taxels, tangential = None, None
        if tangential is not None and (taxels is None or tangential.shape != taxels.shape + (2,)):
            trial.warnings.append("Native tangential taxel shape is inconsistent; tangential spatial evidence unavailable")
            tangential = None
        _assign_force_arrays(trial, time, normal, taxels, tangential, names)
        trial.task_tactile_clock_aligned = True
        native = next(row for row in trial.provenance if row["path"] == str(trace))
        native.update(clock="simulation_time_s", task_tactile_clock_alignment="shared_recorded_simulation_clock",
                      sampling="saved native samples; no interpolation or simulation rerun", sample_count=len(time))
        schema = comparison.get("tactile", {}) if comparison else {}
        if (schema.get("unit") != "N" or schema.get("taxel_grid") != [7, 5] or schema.get("link_names") != names
                or schema.get("Fn") != "sum of all 7x5 normal taxel forces" or schema.get("Ft") not in _FT_FORMULAS):
            tangent = None
            trial.warnings.append("No exact declared native |Ft| schema/unit/channel binding; |Ft| remains unknown, never replaced by ΔFt")
        return trace, tangent
    except (ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        trial.warnings.append(f"Native tactile unavailable: {error}; recorded outcome retained")
        return trace, None


def _frame_trial(trial, frames):
    """Explicit right-hand identities and the frame tactile clock, never row count alignment."""
    if not frames or "normal_taxel_force_n" not in frames[0]:
        return None
    names = frames[0].get("fingertip_order")
    if (not isinstance(names, list) or len(set(names)) != len(names)
            or not set(_LABELS).issubset(names) or any(row.get("fingertip_order") != names for row in frames)):
        trial.warnings.append("Frame right-fingertip identities unavailable; no channels guessed")
        return None
    indices = [names.index(name) for name in _LABELS]
    try:
        values = np.asarray([row.get("normal_taxel_force_n") for row in frames], dtype=float)
        if values.shape != (len(frames), len(names), 7, 5):
            raise ValueError("unsupported frame taxel shape")
        frame_time = _clock([row.get("tactile_time_s", row.get("simulation_time_s")) for row in frames], "frame tactile")
        # A differing tactile clock remains its own source clock. Task/media use
        # the explicit simulation clock and are not separately zero-aligned.
        same_clock = all(row.get("tactile_time_s", row.get("simulation_time_s")) == row.get("simulation_time_s") for row in frames)
        tangent = None
        if all(row.get("tangent_taxel_force_n") is not None for row in frames):
            candidate = np.asarray([row["tangent_taxel_force_n"] for row in frames], dtype=float)
            if candidate.shape == values.shape + (2,):
                tangent = candidate[:, indices]
        result = TrialData(root=trial.root, summary=trial.summary, task=trial.task,
                           task_tactile_clock_aligned=same_clock)
        _assign_force_arrays(result, frame_time, taxels=values[:, indices], tangent=tangent, names=list(_LABELS))
        return result
    except (ValueError, TypeError) as error:
        trial.warnings.append(f"Frame spatial evidence unavailable: {error}")
        return None


def _pose_clock_evidence(trial, records):
    """Permit the audited poker pose clock only through its bound source files."""
    if trial.task != "poker-draw" or any(row.get("timing") != _CACHED_TIMING for row in records):
        raise ValueError("No audited cached-pose recorder contract")
    integrity = trial.summary.get("deployment_integrity", {})
    binding, source = integrity.get("manifest", {}), integrity.get("evaluation_source", {})
    if (integrity.get("status") != "verified" or source.get("status") != "verified"
            or binding.get("path") != trial.summary.get("deployment_manifest")
            or not isinstance(binding.get("path"), str) or not Path(binding["path"]).is_absolute()
            or Path(binding["path"]).is_symlink()
            or not isinstance(binding.get("expected_sha256"), str) or not _SHA.fullmatch(binding["expected_sha256"])
            or binding.get("actual_sha256") != binding["expected_sha256"]):
        raise ValueError("No checksum-bound deployment evidence for camera pose clock")
    path = Path(binding["path"]).resolve(strict=True)
    raw, manifest_row = _read_bound(path, trial, "camera_pose_clock_deployment_binding", limit=2 * 1024**2, collect=True)
    if manifest_row["sha256"] != binding["expected_sha256"]:
        raise InputChangedError("Pose-clock deployment manifest SHA256 mismatch")
    deployment = _parse(raw)
    if any(deployment.get(key) != trial.summary.get(key) for key in ("task", "deployment_id", "checkpoint_sha256")):
        raise ValueError("Pose-clock deployment identity mismatch")
    project = deployment.get("evaluation_project_path")
    if (not isinstance(project, str) or not Path(project).is_absolute() or project != source.get("root")
            or deployment.get("evaluation_source_sha256") != source.get("expected_sha256")
            or source.get("actual_sha256") != source.get("expected_sha256")):
        raise ValueError("Pose-clock source snapshot binding mismatch")
    project = _path(project)
    files = deployment.get("evaluation_source_files")
    if (not isinstance(files, list) or len(files) > 1000
            or not all(isinstance(row, dict) for row in files)):
        raise ValueError("Invalid pose-clock source inventory")
    if content_hash(files) != source["expected_sha256"]:
        raise ValueError("Pose-clock source inventory SHA256 mismatch")
    selected = next((audit for audit in _POSE_CLOCK_SOURCE_SETS if all(
        len(rows := [row for row in files if row.get("path") == relative]) == 1
        and rows[0].get("sha256") == expected
        for relative, expected in audit.items())), None)
    if selected is None:
        raise ValueError("Recorder/simulation source does not match an audited pose-clock combination")
    provenance = [manifest_row]
    for relative, expected in selected.items():
        rows = [row for row in files if isinstance(row, dict) and row.get("path") == relative]
        if len(rows) != 1 or rows[0].get("sha256") != expected:
            raise ValueError("Recorder/simulation source does not match the audited pose-clock implementation")
        _, actual = _read_bound(project / relative, trial, "audited_camera_pose_clock_source", limit=2 * 1024**2)
        if actual["sha256"] != expected or actual["size"] != rows[0].get("size"):
            raise InputChangedError("Pose-clock source SHA256/size mismatch")
        provenance.append(actual)
    return provenance


def _media(trial, frames, manifest, directory):
    result = {"status": "missing", "sync_status": "unverified", "url": None,
              "time_s": [], "video_time_s": [], "reason": "No recorded local video",
              "video_validation": "Local MP4 header and recorder metadata, not independently decoded PTS",
              "mapping_interpolation": "past-only frame hold; no extrapolation beyond recorded frames"}
    path = directory / "review.mp4" if directory else _layout_file(trial.root, "review.mp4")
    if path is None or not path.is_file():
        return result, []
    path = _path(path)
    with path.open("rb") as stream:
        if b"ftyp" not in stream.read(64):
            result["reason"] = "Invalid MP4 header; no video substituted"
            return result, []
    _, row = _read_bound(path, trial, "copied_review_video", limit=2 * 1024**3)
    result.update(status="available", url="media/rollout.mp4", sha256=row["sha256"],
                  reason="Independent playback; recording-to-trial clock unverified")
    copies = [{"source": path, "url": result["url"], "sha256": row["sha256"], "size": row["size"]}]
    video = trial.summary.get("video")
    records = [record for record in (manifest, video) if isinstance(record, dict)]
    try:
        if not records or any(record.get("schema") not in _REVIEW_SCHEMAS for record in records):
            raise ValueError("No supported review schema")
        for record in records:
            if record.get("completed") is not True or record.get("task") != trial.task:
                raise ValueError("Review completion/task identity disagreement")
            for key in ("seed", "deployment_id", "checkpoint_sha256", "prediction_horizon"):
                if key in record and key in trial.summary and record[key] != trial.summary[key]:
                    raise ValueError("Review trial identity disagreement")
        if trial.summary.get("status") in _ACTIVE or trial.summary.get("completed") is False:
            raise ValueError("Trial is not terminal")
        if not frames or not trial.task_tactile_clock_aligned or trial.time_s is None:
            raise ValueError("No shared recorded simulation clock")
        if any(type(row.get("frame")) is not int or row["frame"] != i for i, row in enumerate(frames)):
            raise ValueError("Frame indices are not contiguous and zero-based")
        simulation = _clock([row.get("simulation_time_s") for row in frames], "video simulation")
        pose = _clock([row.get("camera_pose_time_s", row.get("simulation_time_s")) for row in frames], "camera pose")
        offset = pose - simulation
        pose_mapping = bool(np.any(np.abs(offset) > 1e-6))
        pose_provenance = []
        if pose_mapping:
            if (any("camera_pose_time_s" not in row or "tactile_time_s" not in row for row in frames)
                    or not np.array_equal(_clock([row["tactile_time_s"] for row in frames], "frame tactile"), pose)):
                raise ValueError("Camera pose offset lacks an explicit matching cached tactile clock")
            pose_provenance = _pose_clock_evidence(trial, records)
            if pose[0] < trial.time_s[0] - 1e-9 or pose[-1] > trial.time_s[-1] + 1e-9:
                raise ValueError("Recorded camera pose clock is outside the native simulation-clock range")
        mapping_time = pose if pose_mapping else simulation
        for record in records:
            if type(record.get("frame_count")) is not int or record["frame_count"] != len(frames):
                raise ValueError("Recorded frame count disagrees")
            if pose_mapping and any(key not in record for key in ("first_pose_time_s", "last_pose_time_s")):
                raise ValueError("Pose-clock mapping needs explicit absolute manifest endpoints")
            for key, actual in (("first_pose_time_s", mapping_time[0]), ("last_pose_time_s", mapping_time[-1])):
                if key in record and (type(record[key]) not in (int, float) or not math.isfinite(record[key]) or abs(record[key] - actual) > 1e-6):
                    raise ValueError("Manifest/frame timestamps disagree")
        if any("video_time_s" in frame for frame in frames):
            playback = _clock([frame.get("video_time_s") for frame in frames], "explicit video")
            method = "explicit_per_frame_video_time_s_to_simulation_time_s"
        else:
            fps = [record.get("fps") for record in records]
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in fps)
                    or any(value != fps[0] for value in fps)
                    or any(not str(record.get("video_clock", "")).startswith("simulation time") for record in records)):
                raise ValueError("Recorder FPS/simulation-clock declaration missing or inconsistent")
            playback = np.arange(len(frames), dtype=float) / fps[0]
            method = "recorded_frame_index_divided_by_manifest_fps_to_simulation_time_s"
        if np.any(playback < 0):
            raise ValueError("Negative video timestamp")
        origin = float(trial.time_s[0])
        result.update(sync_status="mapped", time_s=(mapping_time - origin).tolist(), video_time_s=playback.tolist(),
                      mapping_method="recorded_camera_pose_time_s" if pose_mapping else method,
                      playback_clock_method=method, trial_time_origin_s=origin, frame_count=len(frames),
                      reason="Same-trial recorder metadata and native frame timestamps; no duration stretching")
        if pose_mapping:
            result.update(camera_pose_clock_provenance=pose_provenance,
                          camera_pose_minus_frame_simulation_s={"minimum": float(np.min(offset)), "maximum": float(np.max(offset))},
                          physical_phase_alignment="not_certified",
                          reason="Video uses its explicitly recorded cached RGB pose time; frame simulation timestamps and native Fn labels are unchanged. No physical same-phase certification or interpolation.")
    except InputChangedError:
        raise
    except (ValueError, TypeError, OverflowError, OSError) as error:
        result["reason"] = f"{error}; independent playback only"
    return result, copies


def _report(trial, protocol, horizon):
    task = build_task_report(trial, protocol)
    if trial.time_s is None or trial.normal_force_n is None:
        tactile = {"schema_version": "tactile-metrics-v1", "status": "missing", "scopes": {},
                   "reason": "No recorded interpretable normal-force series"}
    else:
        scopes, availability = _aligned_scopes(trial, task)
        tactile = analyze_tactile(trial.time_s, trial.normal_force_n, channel_names=trial.channel_names,
                                 normal_taxel_force_n=trial.normal_taxel_force_n, tangent_taxel_force_n=trial.tangent_taxel_force_n,
                                 taxel_positions_local_m=trial.taxel_positions_local_m, scope_masks=scopes,
                                 config={key: protocol[key] for key in DEFAULT_CONFIG})
        tactile["task_scope_evidence"] = availability
        for key, status in availability.items():
            if key not in tactile["scopes"]:
                tactile["scopes"][key] = {**status, "duration_s": None, "channels": None, "load_shares": None}
    summary = trial.summary
    video_metadata = summary.get("video")
    video_metadata = video_metadata if isinstance(video_metadata, dict) else {}
    cohort = {"task": trial.task, "checkpoint": summary.get("checkpoint_sha256") or summary.get("checkpoint_path"),
              **{key: summary.get(key) for key in ("model_family", "controller", "control_hz", "execute_steps")},
              "guard_mode": task["safety"]["enforcement"], "diagnostic_only": task["outcome"]["diagnostic_only"],
              "formal_metrics_valid": task["outcome"]["formal_metrics_valid"], "prediction_horizon": 50,
              "tactile_channels": trial.channel_names,
              "tactile_sample_interval_s": round(float(np.median(np.diff(trial.time_s))), 6)
              if trial.time_s is not None and len(trial.time_s) > 1 else None}
    implementation_rows = _implementation()
    # Include reused evaluator sources; hashes identify the actual formulas.
    analysis = Path(__file__).resolve().parents[1]
    for folder in ("evaluation_v1", "evaluation_v2", "evaluation_tactile_load_extension"):
        for path in sorted((analysis / folder).glob("*.py")):
            _, sha, size, _ = _read(path, limit=2 * 1024**2)
            implementation_rows.append({"path": str(path), "sha256": sha, "size": size})
    implementation = {"files": implementation_rows, "sha256": content_hash(implementation_rows),
                      "semantics": "Existing numerical definitions; generic immutable recorder-layout adapter"}
    trial_id = f"{trial.task}:{trial.root.parent.name}:{trial.root.name}:{hashlib.sha256(str(trial.root).encode()).hexdigest()[:8]}"
    return {"schema_version": "task-tactile-evaluation-v1", "suite_schema_version": "task-tactile-suite-v2",
            "trial": {"task": trial.task, "trial_id": trial_id, "source": str(trial.root), "seed": summary.get("seed"),
                      "prediction_horizon": 50, "cohort": cohort},
            "protocol": protocol, "protocol_sha256": content_hash(protocol), "implementation": implementation,
            "prediction_horizon_evidence": horizon, "provenance": trial.provenance, "warnings": trial.warnings,
            **task, "tactile": tactile,
            "task_review": {"schema_version": "evaluation-task-review-v2", "task": trial.task,
                            "review": {"status": "unknown", "availability": "not_verified", "basis": "none"},
                            "original_outcome_preserved": True, "reason": "No independent contract/evidence certification performed"},
            "evidence_v2": {"status": "not_verified", "source": "recorded_summary_and_native_samples"},
            "contact_quality": {"status": "unknown", "reason": "No independently verified contact-pair evidence"},
            "reference_comparison": {"status": "not_configured"},
            "evidence_quality": {"tactile_source": video_metadata.get("tactile_source"),
                                 "force_unit": "N as recorded; calibration not independently verified",
                                 "task_tactile_clock_mapping": "recorded_common_simulation_clock" if trial.task_tactile_clock_aligned else "unverified"},
            "limitations": ["Recorded task outcome retained, not independently certified.",
                            "No expert time alignment, universal score, slip truth or tactile causal claim.",
                            "Missing fields and time gaps are not replaced with zero or interpolated."]}


def _missing_loads(report, reason):
    return {"schema_version": "evaluation-five-finger-load-extension-v1", "status": "missing",
            "trial_id": report["trial"]["trial_id"], "task": report["trial"]["task"], "prediction_horizon": 50,
            "time_s": [], "channels": [], "normal_force_n": [], "normal_total_n": [], "tangent_force_n": [],
            "tangent_total_n": [], "normal_status": "missing", "tangent_status": "missing", "gap_after": [],
            "nominal_interval_s": None, "unit": "N", "provenance": report["provenance"], "warnings": [reason],
            "semantics": {"unknown": "No supported right-five-fingertip native series; trial retained, no zero fill"}}


def build_case(trial_dir, protocol=None):
    """Build one H50 case entirely in memory; copy jobs are not executed here."""
    root = _path(Path(trial_dir).resolve(strict=True))
    horizon = inspect_prediction_horizon(root)
    if horizon["status"] != "verified" or horizon["prediction_horizon"] != 50:
        raise ValueError("Only explicitly verified H50 source trials may be assembled")
    trial = TrialData(root=root, summary={}, task="unknown")
    summary = _json_bound(root / "summary.json", trial, "recorded_trial_outcome")
    evaluation = summary.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("Summary evaluation must be a mapping")
    task = summary.get("task", evaluation.get("task"))
    if not isinstance(task, str) or not task.strip() or isinstance(summary.get("trials"), list):
        raise ValueError("An explicitly identified single trial is required")
    if evaluation.get("task", task) != task:
        raise ValueError("Summary/evaluation task identity mismatch")
    trial.summary, trial.task = summary, task
    trial.provenance = _rows(trial.provenance + horizon["provenance"])
    trial.warnings.extend(horizon["warnings"])
    trace, tangent = _load_native(trial)
    controls = root / "control_steps.jsonl"
    if controls.is_file():
        try:
            trial.control_rows = _jsonl(controls, trial, "control_step_task_evidence")
        except InputChangedError:
            raise
        except (ValueError, TypeError) as error:
            trial.warnings.append(f"Control evidence unavailable: {error}")
    frames_path = _layout_file(root, "frames.jsonl")
    frames = []
    if frames_path:
        try:
            frames = _jsonl(frames_path, trial, "review_frame_evidence", frames=True,
                            keep_taxels=trial.normal_taxel_force_n is None)
        except InputChangedError:
            raise
        except (ValueError, TypeError) as error:
            trial.warnings.append(f"Review-frame evidence unavailable: {error}")
    if not trial.control_rows and frames:
        trial.control_rows = [{"sim_time": row.get("simulation_time_s"), "task": row.get("task_metrics", {}),
                               "stage_after_action": row.get("phase"), "_source": row["_source"],
                               "_sampling": "native_review_frames"} for row in frames]
        trial.warnings.append("Task samples use recorded review-frame cadence, not the native tactile/control cadence")
    spatial_trial = _frame_trial(trial, frames) if trial.normal_taxel_force_n is None else None
    if trial.time_s is None and spatial_trial is not None:
        for key in ("time_s", "normal_force_n", "normal_taxel_force_n", "tangent_taxel_force_n", "channel_names", "task_tactile_clock_aligned"):
            setattr(trial, key, getattr(spatial_trial, key))
        tangent = None
        trial.warnings.append("Tactile series falls back to explicit right-hand review-frame samples; native trace unavailable")
    protocol = make_protocol(protocol)
    report = _report(trial, protocol, horizon)
    visual = build_visualization(trial, report)
    if spatial_trial is not None and trial.normal_taxel_force_n is None and trial.time_s is not None:
        if spatial_trial.task_tactile_clock_aligned:
            spatial, _ = _spatial(spatial_trial, spatial_trial.time_s - trial.time_s[0], spatial_trial.normal_force_n,
                                  180, visual["warnings"])
            spatial["sampling"] = "Separate native review-frame tactile clock; not resampled to the Fn clock"
            visual["spatial_frames"] = spatial
    manifest_path = _layout_file(root, "review.json")
    manifest = _json_bound(manifest_path, trial, "video_recording_manifest") if manifest_path else None
    directory = manifest_path.parent if manifest_path else frames_path.parent if frames_path else None
    visual["media"], copies = _media(trial, frames, manifest, directory)
    tactile = visual["tactile_series"]
    if trial.time_s is not None and set(trial.channel_names) == set(_LABELS) and len(trial.channel_names) == 5:
        loads = build_load_series(tactile["time_s"], tactile["normal_force_n"], tangent, trial.channel_names,
                                  tactile["gap_after"], tactile["nominal_interval_s"])
        loads.update(status="available" if loads["normal_status"] == "available" else "partial",
                     trial_id=report["trial"]["trial_id"], task=task, prediction_horizon=50,
                     native_trace=str(trace) if trace else None)
    else:
        loads = _missing_loads(report, "No explicitly identified native right-five-finger series")
    plots, plot_jobs = load_diagnostic_plots(root, expected_task=task,
        expected_summary_sha=next(row["sha256"] for row in trial.provenance if row["path"] == str(root / "summary.json")))
    for job in plot_jobs:
        copies.append({"source": job["source"], "url": job["url"], "sha256": job["row"]["sha256"], "size": job["row"]["size"]})
    trial.provenance = _rows(trial.provenance + plots["provenance"])
    report["provenance"], loads["provenance"] = trial.provenance, trial.provenance
    report["visualization"] = {"schema_version": visual["schema_version"], "path": "visualization.json",
                               "sha256": content_hash(visual), "semantics": "Native display data; no resampling"}
    checkpoint = summary.get("checkpoint_path")
    config = {"schema_version": SCHEMA_VERSION, "trial_id": report["trial"]["trial_id"], "task": task,
              "prediction_horizon": 50, "checkpoint": Path(checkpoint).name if isinstance(checkpoint, str) else None,
              "seed": summary.get("seed"), "media_views": {"status": "unverified", "views": []},
              "diagnostic_plots": plots, "load_status": loads["status"]}
    _verify(trial.provenance)
    _verify(report["implementation"]["files"])
    content_hash(report); content_hash(visual); content_hash(loads); content_hash(config)
    return {"trial": trial, "report": report, "visualization": visual, "tactile_loads": loads, "config": config,
            "copies": copies, "frame_rows": frames}
