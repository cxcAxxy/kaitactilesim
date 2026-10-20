"""Inspect optional local trial video and explicit recording clocks, without writes."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .adapters import InputChangedError, TrialData, _read


_REVIEW_SCHEMAS = {"kaihand-policy-evaluation-review-v2", "kaihand-policy-evaluation-review-v3"}


def _inside(root: Path, value: object) -> Path | None:
    """Never follow a URL or a symlink out of the explicitly selected trial."""
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    value = str(value)
    try:
        if urlsplit(value).scheme or "\x00" in value:
            return None
    except ValueError:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _read_record(path: Path, trial: TrialData, role: str) -> bytes:
    """Reuse prior provenance only after verifying these are the same bytes."""
    existing = next((row for row in trial.provenance if row.get("path") == str(path)), None)
    data = _read(path, trial, role=role)
    fresh = trial.provenance[-1]
    if existing is not None:
        trial.provenance.pop()
        if existing.get("sha256") != fresh["sha256"]:
            raise InputChangedError(f"media input changed since trial loading: {path}")
    return data


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        return float(value)
    return None


def _valid_mp4(path: Path) -> bool:
    # Header validation is not a video decode or independent validation of frame count.
    before = path.stat()
    with path.open("rb") as file:
        header = file.read(64)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise InputChangedError(f"video changed during header inspection: {path}")
    return path.suffix.lower() == ".mp4" and before.st_size >= 12 and b"ftyp" in header


def inspect_media(trial: TrialData) -> dict:
    """Return an internal media descriptor; caller must export only a relative URL.

    Mapping is supported for the known review recorder bundle, whose manifest
    defines frame indices and frame rate. Other local MP4s can play independently.
    No duration stretching, filename-only synchronization, or guessed time offset
    is used. Video bytes are not hashed here; the copying caller should stream-hash
    and verify the copied media before publishing a bundle.
    """
    root = trial.root.resolve()
    initial_provenance = len(trial.provenance)
    result = {"status": "missing", "source_path": None, "export_name": "rollout.mp4",
              "sync_status": "unverified", "reason": "no valid local trial MP4 found",
              "video_time_s": [], "time_s": [], "mapping_method": None,
              "trial_time_origin_s": None, "provenance": [],
              "video_validation": "local MP4 header only; encoded stream not decoded",
              "mapping_interpolation": "past-only frame hold; no extrapolation beyond recorded frames"}
    video_metadata = trial.summary.get("video")
    metadata = video_metadata if isinstance(video_metadata, dict) else {}
    candidates = []
    if isinstance(video_metadata, str):
        candidates.append(video_metadata)
    for key in ("path", "video_path", "mp4_path", "output_path"):
        candidates.append(metadata.get(key))
    for key in ("video_path", "review_video_path"):
        candidates.append(trial.summary.get(key))
    candidates.extend(("review/review.mp4", "review.mp4", "rollout.mp4"))
    selected = None
    for candidate in candidates:
        local = _inside(root, candidate)
        if local is not None and local.suffix.lower() == ".mp4":
            try:
                if _valid_mp4(local):
                    selected = local
                    break
            except InputChangedError:
                raise
            except OSError:
                continue
    if selected is None:
        return result
    result.update(status="available", source_path=str(selected),
                  reason="video available for independent playback; recording-to-trial clock is unverified")

    def finish(reason: str | None = None):
        if reason is not None:
            result["reason"] = reason
        result["provenance"] = [dict(row) for row in trial.provenance[initial_provenance:]]
        return result

    review_video = _inside(root, "review/review.mp4")
    if selected != review_video:
        return finish("local video is not the recognized review recorder bundle; independent playback only")
    manifest_path = _inside(root, "review/review.json")
    manifest = None
    if manifest_path is not None:
        try:
            manifest = json.loads(_read_record(manifest_path, trial, "video_recording_manifest"))
            if not isinstance(manifest, dict):
                raise ValueError("recording manifest is not an object")
        except InputChangedError:
            raise
        except (OSError, ValueError, TypeError):
            return finish("recording manifest is unreadable; independent playback only")
    # The recognized schema establishes review.mp4 / frames.jsonl association.
    records = [record for record in (manifest, metadata)
               if isinstance(record, dict) and record.get("schema") in _REVIEW_SCHEMAS]
    if not records:
        return finish("no recognized recording manifest binds the local video to frames.jsonl")
    for record in records:
        if record.get("task") is not None and record["task"] != trial.task:
            return finish("recording manifest task does not match the trial; synchronization is unverified")
        for key in ("seed", "deployment_id", "checkpoint_sha256"):
            if record.get(key) is not None and trial.summary.get(key) is not None and record[key] != trial.summary[key]:
                return finish("recording manifest identity does not match the trial; synchronization is unverified")
    if any(record.get("completed") is not True for record in records):
        return finish("recording is not explicitly complete; synchronization is unverified")
    if trial.summary.get("status") in {"running", "queued", "pending", "starting", "in_progress"}:
        return finish("trial is still in progress; synchronization is unverified")
    if not trial.task_tactile_clock_aligned or trial.time_s is None or not len(trial.time_s):
        return finish("trial series has no verified common simulation clock; independent playback only")
    origin = _number(float(trial.time_s[0]))
    if origin is None:
        return finish("trial time origin is not finite; synchronization is unverified")
    frames_path = _inside(root, "review/frames.jsonl")
    if frames_path is None:
        return finish("recorded per-frame timestamps are missing; independent playback only")
    try:
        payload = _read_record(frames_path, trial, "video_frame_clock_mapping")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError("no usable frame records")
        indices = [row.get("frame") for row in rows]
        if any(type(index) is not int for index in indices) or indices != list(range(len(rows))):
            raise ValueError("frame indices must be contiguous and zero-based")
        simulation = [_number(row.get("simulation_time_s")) for row in rows]
        if any(value is None for value in simulation):
            raise ValueError("frame simulation clock is missing or nonfinite")
        simulation = np.asarray(simulation, dtype=float)
        if len(rows) > 1 and np.any(np.diff(simulation) <= 0):
            raise ValueError("frame simulation clock is not strictly increasing")
        for row, time in zip(rows, simulation):
            if "camera_pose_time_s" in row:
                camera_time = _number(row["camera_pose_time_s"])
                if camera_time is None or abs(camera_time - time) > 1e-6:
                    raise ValueError("camera pose and simulation frame clocks disagree")
        for record in records:
            count = record.get("frame_count")
            if type(count) is not int or count != len(rows):
                raise ValueError("recorded frame count is missing or does not match frames.jsonl")
            for key, actual in (("first_pose_time_s", simulation[0]), ("last_pose_time_s", simulation[-1])):
                if key in record:
                    recorded = _number(record[key])
                    if recorded is None or abs(recorded - actual) > 1e-6:
                        raise ValueError("recording manifest and frame simulation clocks disagree")
        explicit = [_number(row.get("video_time_s")) for row in rows]
        if all(value is not None for value in explicit):
            video_time = np.asarray(explicit, dtype=float)
            method = "explicit_per_frame_video_time_s_to_simulation_time_s"
        elif any("video_time_s" in row for row in rows):
            raise ValueError("explicit video timestamps are incomplete; no guessed replacements")
        else:
            rates = [_number(record.get("fps")) for record in records]
            if any(rate is None or rate <= 0 for rate in rates):
                raise ValueError("positive recording FPS is not explicitly available")
            if any(abs(rate - rates[0]) > 1e-9 for rate in rates):
                raise ValueError("recording manifest and summary FPS disagree")
            if not all(str(record.get("video_clock", "")).startswith("simulation time") for record in records):
                raise ValueError("recorder does not explicitly declare its simulation clock")
            video_time = np.asarray(indices, dtype=float) / rates[0]
            method = "recorded_frame_index_divided_by_manifest_fps_to_simulation_time_s"
        if np.any(video_time < 0) or (len(rows) > 1 and np.any(np.diff(video_time) <= 0)):
            raise ValueError("video clock must be nonnegative and strictly increasing")
    except InputChangedError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return finish(f"frame mapping could not be verified: {exc}; independent playback only")
    result.update(sync_status="mapped", video_time_s=video_time.tolist(),
                  time_s=(simulation - origin).tolist(), mapping_method=method,
                  trial_time_origin_s=origin, frame_count=len(rows),
                  reason="recording manifest and per-frame clocks establish this mapping; video stream PTS were not independently decoded")
    return finish()
