"""Read immutable snapshots of existing trial artifacts without simulator imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np


@dataclass
class TrialData:
    root: Path
    summary: dict
    task: str
    time_s: np.ndarray | None = None
    normal_force_n: np.ndarray | None = None
    normal_taxel_force_n: np.ndarray | None = None
    tangent_taxel_force_n: np.ndarray | None = None
    taxel_positions_local_m: np.ndarray | None = None
    channel_names: list[str] = field(default_factory=list)
    control_rows: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    task_tactile_clock_aligned: bool = False


class InputChangedError(ValueError):
    """An input changed while reading; do not publish a mixed trial snapshot."""


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _read(path: Path, trial: TrialData, *, role: str) -> bytes:
    before = _signature(path)
    data = path.read_bytes()
    if _signature(path) != before or len(data) != before[2]:
        raise InputChangedError(f"input changed while reading: {path}")
    trial.provenance.append({
        "path": str(path.resolve()), "size": len(data),
        "sha256": sha256(data).hexdigest(), "role": role,
    })
    return data


def _rows(path: Path, trial: TrialData, *, role: str) -> list[dict]:
    payload = _read(path, trial, role=role)
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = _json_loads(line, trial)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        row = dict(row)
        row["_source"] = f"{path.resolve()}:{line_number}"
        rows.append(row)
    return rows


def _json_loads(payload: bytes | str, trial: TrialData) -> Any:
    def unknown_constant(value):
        message = f"nonfinite JSON constant {value} retained as unknown (null); raw source hash preserved"
        if message not in trial.warnings:
            trial.warnings.append(message)
        return None
    return json.loads(payload, parse_constant=unknown_constant)


def _time(values: Any) -> np.ndarray:
    time = np.asarray(values, dtype=float)
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all():
        raise ValueError("sample clock must be a nonempty finite 1D array")
    if len(time) > 1 and np.any(np.diff(time) <= 0):
        raise ValueError("sample clock must be strictly increasing; no silent sorting")
    return time


def _numeric_force(values: Any, trial: TrialData, *, normal: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(array)
    if normal:
        invalid |= array < 0
    if invalid.any():
        trial.warnings.append(
            f"{'normal' if normal else 'tangent'} force: {int(invalid.sum())} invalid values retained as unknown (NaN)"
        )
        array[invalid] = np.nan
    return array


def _assign_force_arrays(trial: TrialData, time: Any, normal: Any = None,
                         taxels: Any = None, tangent: Any = None,
                         names: Any = None, positions: Any = None) -> None:
    clock = _time(time)
    n = len(clock)
    taxel_array = None
    if taxels is not None:
        taxel_array = _numeric_force(taxels, trial, normal=True)
        if taxel_array.ndim != 4 or taxel_array.shape[0] != n:
            raise ValueError("normal_taxel_force_n must have shape [N, C, rows, cols]")
    if normal is None and taxel_array is not None:
        normal = np.sum(taxel_array, axis=(-2, -1))  # NaN does not become zero.
    if normal is None:
        raise ValueError("no measured normal force array in artifact")
    force = _numeric_force(normal, trial, normal=True)
    if force.ndim != 2 or force.shape[0] != n:
        raise ValueError("normal force must have shape [N, C]")
    if taxel_array is not None and taxel_array.shape[:2] != force.shape:
        raise ValueError("normal force and taxel channel dimensions disagree")
    if taxel_array is not None:
        summed = np.sum(taxel_array, axis=(-2, -1))
        known = np.isfinite(summed) & np.isfinite(force)
        if np.any(~np.isclose(summed[known], force[known], rtol=1e-5, atol=1e-7)):
            trial.warnings.append("recorded fingertip normal force differs from taxel sum; recorded force retained")
    tangent_array = None
    if tangent is not None:
        candidate = _numeric_force(tangent, trial, normal=False)
        expected = None if taxel_array is None else taxel_array.shape + (2,)
        if expected is not None and candidate.shape == expected:
            tangent_array = candidate
        else:
            trial.warnings.append("tangent taxel dimensions unavailable or inconsistent; tangent metrics unknown")
    c = force.shape[1]
    labels = [str(value.decode() if isinstance(value, bytes) else value)
              for value in names] if names is not None else []
    if len(labels) != c or len(set(labels)) != c:
        labels = [f"channel_{index:03d}" for index in range(c)]
        trial.warnings.append("anatomical channel names unavailable; channel indices are not finger identities")
    geometry = None
    if positions is not None and taxel_array is not None:
        candidate = np.asarray(positions, dtype=float)
        if (candidate.shape[:-1] == taxel_array.shape[1:] and
                candidate.shape[-1] == 3 and np.isfinite(candidate).all()):
            geometry = candidate
        elif (candidate.shape[:-1] == taxel_array.shape[1:] and
              candidate.shape[-1] == 2 and np.isfinite(candidate).all()):
            trial.warnings.append("only 2D taxel coordinates recorded; 3D physical geometry left unknown rather than assuming a planar z=0 surface")
        else:
            trial.warnings.append("taxel geometry has incompatible dimensions; physical geometry left unknown")
    trial.time_s, trial.normal_force_n = clock, force
    trial.normal_taxel_force_n, trial.tangent_taxel_force_n = taxel_array, tangent_array
    trial.taxel_positions_local_m, trial.channel_names = geometry, labels


def _from_npz(path: Path, trial: TrialData) -> None:
    payload = _read(path, trial, role="native_control_cadence_tactile")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        get = lambda key: archive[key] if key in archive else None
        clock = get("simulation_time_s")
        clock_name = "simulation_time_s"
        if clock is None:
            clock, clock_name = get("time_s"), "time_s"
            trial.warnings.append("NPZ has elapsed time_s only; absolute simulation clock alignment is unverified")
        _assign_force_arrays(trial, clock, get("fingertip_normal_force_n"),
                             get("normal_taxel_force_n"), get("tangent_taxel_force_n"),
                             get("fingertip_link_names"), get("taxel_positions_local_m"))
        trial.task_tactile_clock_aligned = clock_name == "simulation_time_s"
        trial.provenance[-1].update({"clock": clock_name,
            "task_tactile_clock_alignment": "shared_recorded_simulation_clock" if trial.task_tactile_clock_aligned else "unverified_elapsed_clock",
            "sampling": "saved native samples; no interpolation or simulation rerun",
            "sample_count": len(trial.time_s)})


def _from_frames(rows: list[dict], trial: TrialData) -> None:
    if not rows:
        raise ValueError("frames.jsonl contains no samples")
    names = rows[0].get("fingertip_order")
    if any(row.get("fingertip_order") != names for row in rows):
        raise ValueError("fingertip channel order changes between review frames")
    taxels = [row.get("normal_taxel_force_n") for row in rows]
    array = np.asarray(taxels, dtype=float)
    tangent = None
    if all(row.get("tangent_taxel_force_n") is not None for row in rows):
        tangent = np.asarray([row["tangent_taxel_force_n"] for row in rows], dtype=float)
    # Preserve every recorded channel. A fallback may contain ten fingers while
    # a native rollout NPZ contains five; task handedness is never guessed here.
    _assign_force_arrays(trial, [row.get("simulation_time_s") for row in rows],
                         taxels=array, tangent=tangent, names=names)
    trial.task_tactile_clock_aligned = True
    trial.warnings.append("tactile fallback uses all recorded review-frame channels (typically 10 Hz); native NPZ channel coverage may differ; these are not physics samples or physics peaks")


def load_trial(path: str | Path) -> TrialData:
    """Load only original small data artifacts, with content-addressed provenance.

    Optional missing or malformed tactile data does not erase the recorded task
    outcome. Input files that change during the read are rejected outright.
    """
    root = Path(path).expanduser().resolve()
    trial = TrialData(root=root, summary={}, task="unknown")
    summary_path = root / "summary.json"
    signature = _signature(summary_path)
    summary = _json_loads(_read(summary_path, trial, role="recorded_trial_outcome"), trial)
    if not isinstance(summary, dict):
        raise ValueError("summary.json must be an object")
    if isinstance(summary.get("trials"), list) and not isinstance(summary.get("evaluation"), dict):
        raise ValueError("batch summary is not a trial; select one trial directory")
    evaluation = summary.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("summary.evaluation must be an object")
    task = summary.get("task", evaluation.get("task", "unknown"))
    if evaluation.get("task") is not None and task != evaluation["task"]:
        raise ValueError("summary.task and evaluation.task disagree")
    trial.summary, trial.task = summary, str(task)
    control = root / "control_steps.jsonl"
    if control.exists():
        try:
            trial.control_rows = _rows(control, trial, role="control_step_task_evidence")
        except InputChangedError:
            raise
        except (ValueError, OSError) as exc:
            trial.warnings.append(f"control evidence unavailable: {exc}")
    npz = root / "review/evaluation_rollout_trace.npz"
    if npz.exists():
        try:
            _from_npz(npz, trial)
        except InputChangedError:
            raise
        except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
            trial.warnings.append(f"native tactile unavailable: {exc}")
    frames = root / "review/frames.jsonl"
    if frames.exists() and (not trial.control_rows or trial.time_s is None):
        try:
            rows = _rows(frames, trial, role="review_frame_evidence")
            if not trial.control_rows:
                trial.control_rows = [{
                    "sim_time": row.get("simulation_time_s"),
                    "task": row.get("task_metrics", {}),
                    "stage_after_action": row.get("phase"),
                    "_source": row["_source"], "_sampling": "review_frames_typically_10_hz",
                } for row in rows]
                trial.warnings.append("task event evidence falls back to review frames (typically 10 Hz), not control-step records")
            if trial.time_s is None:
                _from_frames(rows, trial)
        except InputChangedError:
            raise
        except (ValueError, OSError, TypeError) as exc:
            trial.warnings.append(f"review fallback unavailable: {exc}")
    if trial.time_s is None:
        trial.warnings.append("no usable tactile time series; outcome report remains available")
    elif len(trial.time_s) > 1:
        differences = np.diff(trial.time_s)
        if np.any(differences > 1.5 * np.median(differences)):
            trial.warnings.append("tactile clock contains gaps; continuity must not bridge missing samples")
    if trial.taxel_positions_local_m is None:
        trial.warnings.append("taxel physical geometry unavailable; no inferred millimetre coordinates")
    if _signature(summary_path) != signature:
        raise InputChangedError("summary changed during trial loading")
    return trial
