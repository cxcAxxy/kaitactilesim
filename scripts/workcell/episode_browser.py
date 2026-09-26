#!/usr/bin/env python3
"""Browse, play, soft-delete, and restore external-video HDF5 episodes."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import secrets
import threading
import time
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import parse_qs, quote, unquote, urlsplit

import h5py


EPISODE_PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")
CAMERA_PATTERN = re.compile(r"^[^/\\\x00]+$")
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CHANNELS = ("normal", "tangential", "proximity")
CHANNEL_INDICES = (1, 2, 4)
EXPECTED_EPISODES = 200
MAX_IDENTIFIER_LENGTH = 64
MAX_JSON_BODY_BYTES = 64 * 1024
TRASH_DIRECTORY = ".episode_browser_trash"
STATIC_FILES = {
  "/": ("index.html", "text/html; charset=utf-8"),
  "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
  "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ApiError(Exception):
  """An expected request failure with a stable JSON representation."""

  def __init__(
    self,
    status: int,
    code: str,
    message: str,
    *,
    details: Any | None = None,
    headers: dict[str, str] | None = None,
  ) -> None:
    super().__init__(message)
    self.status = status
    self.code = code
    self.message = message
    self.details = details
    self.headers = headers or {}

  def payload(self) -> dict[str, Any]:
    error: dict[str, Any] = {"code": self.code, "message": self.message}
    if self.details is not None:
      error["details"] = self.details
    return {"error": error}


def _json_value(value: Any) -> Any:
  """Convert h5py/numpy attribute values to strict JSON-compatible values."""

  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  if isinstance(value, str) or value is None or isinstance(value, bool):
    return value
  if isinstance(value, int):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {str(key): _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if hasattr(value, "tolist"):
    return _json_value(value.tolist())
  if hasattr(value, "item"):
    return _json_value(value.item())
  return str(value)


def _attrs(group: h5py.Group | h5py.Dataset) -> dict[str, Any]:
  return {str(key): _json_value(value) for key, value in group.attrs.items()}


def _strict_json_bytes(payload: Any) -> bytes:
  return json.dumps(
    payload,
    ensure_ascii=False,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def _finite_float(value: Any) -> float | None:
  result = float(value)
  return result if math.isfinite(result) else None


def _percentile_scale(values: Any) -> float:
  finite = sorted(abs(float(value)) for value in values if value is not None)
  if not finite:
    return 1.0
  position = 0.99 * (len(finite) - 1)
  lower = int(math.floor(position))
  upper = int(math.ceil(position))
  percentile = finite[lower]
  if upper != lower:
    percentile += (finite[upper] - finite[lower]) * (position - lower)
  return max(1.0, percentile)


def _identifier(value: str) -> str:
  if not value.isascii() or not value.isdigit() or not value:
    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_episode", "Invalid episode ID")
  if len(value) > MAX_IDENTIFIER_LENGTH:
    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_episode", "Episode ID is too long")
  return value


def _camera(value: str) -> str:
  if value in {".", ".."} or CAMERA_PATTERN.fullmatch(value) is None:
    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_camera", "Invalid camera name")
  return value


def _is_within(path: Path, directory: Path) -> bool:
  try:
    path.relative_to(directory)
  except ValueError:
    return False
  return True


def _select_video_frames(timestamps: list[int], fps: int) -> list[int]:
  """Select nearest existing frames on a regular grid and retain the terminal frame."""

  if not timestamps:
    raise ValueError("timestamp_ros_nsec must be non-empty")
  if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
    raise ValueError("timestamp_ros_nsec must be strictly increasing")
  first = timestamps[0]
  duration = timestamps[-1] - first
  target_count = duration * fps // 1_000_000_000
  selected = [0]
  for output_index in range(int(target_count) + 1):
    target = first + output_index * 1_000_000_000 // fps
    right = bisect.bisect_left(timestamps, target)
    if right >= len(timestamps):
      candidate = len(timestamps) - 1
    elif right == 0:
      candidate = 0
    else:
      left = right - 1
      candidate = (
        left
        if target - timestamps[left] <= timestamps[right] - target
        else right
      )
    if candidate != selected[-1]:
      selected.append(candidate)
  terminal = len(timestamps) - 1
  if terminal != selected[-1]:
    selected.append(terminal)
  return selected


def parse_single_range(header: str, size: int) -> tuple[int, int]:
  """Parse one RFC 7233 byte range and return an inclusive interval."""

  if size <= 0 or not header.startswith("bytes="):
    raise ValueError("invalid byte range")
  value = header[6:].strip()
  if not value or "," in value or value.count("-") != 1:
    raise ValueError("only one byte range is supported")
  start_text, end_text = (part.strip() for part in value.split("-", 1))
  if not start_text:
    if not end_text.isascii() or not end_text.isdigit():
      raise ValueError("invalid suffix byte range")
    suffix = int(end_text)
    if suffix <= 0:
      raise ValueError("invalid suffix byte range")
    start = max(0, size - suffix)
    return start, size - 1
  if not start_text.isascii() or not start_text.isdigit():
    raise ValueError("invalid byte range start")
  start = int(start_text)
  if start >= size:
    raise ValueError("byte range starts beyond the resource")
  if not end_text:
    return start, size - 1
  if not end_text.isascii() or not end_text.isdigit():
    raise ValueError("invalid byte range end")
  end = int(end_text)
  if end < start:
    raise ValueError("byte range end precedes its start")
  return start, min(end, size - 1)


class EpisodeStore:
  """Filesystem and HDF5 operations serialized against delete/restore races."""

  def __init__(self, root: str | Path) -> None:
    candidate = Path(root).expanduser()
    try:
      self.root = candidate.resolve(strict=True)
    except OSError as error:
      raise ValueError(f"dataset directory does not exist: {candidate}") from error
    if not self.root.is_dir():
      raise ValueError(f"dataset path is not a directory: {self.root}")
    self.lock = threading.RLock()

  @property
  def trash_root(self) -> Path:
    return self.root / TRASH_DIRECTORY

  def _episode_path(self, identifier: str) -> Path:
    identifier = _identifier(identifier)
    path = self.root / f"episode_{identifier}.hdf5"
    if path.is_symlink() or not path.is_file():
      raise ApiError(
        HTTPStatus.NOT_FOUND,
        "episode_not_found",
        f"Episode {identifier} was not found",
      )
    try:
      resolved = path.resolve(strict=True)
    except OSError as error:
      raise ApiError(
        HTTPStatus.NOT_FOUND,
        "episode_not_found",
        f"Episode {identifier} was not found",
      ) from error
    if not _is_within(resolved, self.root):
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe episode path")
    return path

  def _video_files(self, identifier: str) -> list[tuple[str, Path]]:
    identifier = _identifier(identifier)
    video_root = self.root / "video"
    if video_root.is_symlink() or not video_root.is_dir():
      return []
    result: list[tuple[str, Path]] = []
    try:
      camera_directories = sorted(video_root.iterdir(), key=lambda path: path.name)
    except OSError:
      return []
    for directory in camera_directories:
      if directory.is_symlink() or not directory.is_dir():
        continue
      try:
        camera = _camera(directory.name)
      except ApiError:
        continue
      path = directory / f"episode_{identifier}.mp4"
      if path.is_symlink() or not path.is_file():
        continue
      try:
        resolved = path.resolve(strict=True)
      except OSError:
        continue
      if _is_within(resolved, self.root):
        result.append((camera, path))
    return result

  def _summary(self, identifier: str, path: Path) -> dict[str, Any]:
    videos = []
    for camera, video_path in self._video_files(identifier):
      try:
        size = video_path.stat().st_size
      except OSError:
        continue
      videos.append(
        {
          "name": camera,
          "size_bytes": size,
          "url": f"/media/{identifier}/{quote(camera, safe='')}",
        }
      )
    try:
      hdf5_size = path.stat().st_size
    except OSError as error:
      raise ApiError(
        HTTPStatus.NOT_FOUND,
        "episode_not_found",
        f"Episode {identifier} was not found",
      ) from error
    total_size = hdf5_size + sum(video["size_bytes"] for video in videos)
    metadata = self._episode_metadata(path)
    return {
      "id": int(identifier),
      "id_text": identifier,
      "name": path.name,
      "hdf5_size_bytes": hdf5_size,
      "size_bytes": total_size,
      "cameras": videos,
      "media": {video["name"]: video["url"] for video in videos},
      "videos": {video["name"]: video["url"] for video in videos},
      "total_size_bytes": total_size,
      **metadata,
    }

  @staticmethod
  def _episode_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
      "frame_count": None,
      "fps": None,
      "duration_s": None,
      "task": None,
      "collect_time_utc": None,
      "hdf5_readable": False,
    }
    try:
      with h5py.File(path, "r") as file:
        timestamps = file.get("timestamp_ros_nsec")
        if isinstance(timestamps, h5py.Dataset) and timestamps.ndim == 1:
          frame_count = int(timestamps.shape[0])
          result["frame_count"] = frame_count
        else:
          frame_count = 0
        fps = _finite_float(file.attrs.get("fps", 0))
        if fps is not None and fps > 0:
          result["fps"] = fps
          result["duration_s"] = frame_count / fps
        result["task"] = _json_value(file.attrs.get("task"))
        result["collect_time_utc"] = _json_value(file.attrs.get("collect_time_utc"))
        result["hdf5_readable"] = True
    except (OSError, TypeError, ValueError):
      pass
    return result

  def dataset(self) -> dict[str, Any]:
    with self.lock:
      episodes: list[tuple[int, str, Path]] = []
      try:
        children = tuple(self.root.iterdir())
      except OSError as error:
        raise ApiError(
          HTTPStatus.INTERNAL_SERVER_ERROR,
          "dataset_unavailable",
          "Could not scan the dataset directory",
        ) from error
      for path in children:
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
          continue
        identifier = match.group(1)
        episodes.append((int(identifier), identifier, path))
      episodes.sort(key=lambda item: (item[0], item[1]))
      summaries = [self._summary(identifier, path) for _, identifier, path in episodes]
      cameras = sorted(
        {video["name"] for episode in summaries for video in episode["cameras"]}
      )
      return {
        "name": self.root.name,
        "root": str(self.root),
        "count": len(summaries),
        "expected_count": EXPECTED_EPISODES,
        "is_expected_count": len(summaries) == EXPECTED_EPISODES,
        "cameras": cameras,
        "total_size_bytes": sum(episode["total_size_bytes"] for episode in summaries),
        "episodes": summaries,
      }

  def detail(self, identifier: str) -> dict[str, Any]:
    with self.lock:
      path = self._episode_path(identifier)
      try:
        with h5py.File(path, "r") as file:
          schema: list[dict[str, Any]] = []

          def visit(name: str, item: h5py.Group | h5py.Dataset) -> None:
            entry: dict[str, Any] = {
              "path": name,
              "kind": "dataset" if isinstance(item, h5py.Dataset) else "group",
              "attrs": _attrs(item),
            }
            if isinstance(item, h5py.Dataset):
              entry["shape"] = list(item.shape)
              entry["dtype"] = str(item.dtype)
            schema.append(entry)

          file.visititems(visit)
          attrs = _attrs(file)
      except (OSError, ValueError) as error:
        raise ApiError(
          HTTPStatus.UNPROCESSABLE_ENTITY,
          "invalid_hdf5",
          f"Could not inspect episode {identifier}",
        ) from error
      return {
        "episode": self._summary(identifier, path),
        "attrs": attrs,
        "schema": schema,
      }

  def telemetry(self, identifier: str, fps: int) -> dict[str, Any]:
    if fps not in {5, 10}:
      raise ApiError(
        HTTPStatus.BAD_REQUEST,
        "invalid_fps",
        "fps must be either 5 or 10",
      )
    with self.lock:
      path = self._episode_path(identifier)
      try:
        with h5py.File(path, "r") as file:
          timestamps = self._read_video_timestamps(file)
          selected = _select_video_frames(timestamps, fps)
          side_streams = {
            side: self._read_tactile_stream(file, side) for side in ("left", "right")
          }
          source_fps = _finite_float(file.attrs.get("fps", 0))
      except KeyError as error:
        raise ApiError(
          HTTPStatus.UNPROCESSABLE_ENTITY,
          "missing_telemetry",
          f"Episode {identifier} does not contain the required tactile telemetry",
          details={"missing": str(error)},
        ) from error
      except (OSError, TypeError, ValueError) as error:
        raise ApiError(
          HTTPStatus.UNPROCESSABLE_ENTITY,
          "invalid_telemetry",
          f"Episode {identifier} has invalid tactile telemetry",
          details={"reason": str(error)},
        ) from error

      frames = []
      compact: dict[str, list[Any]] = {"left": [], "right": []}
      first_timestamp = timestamps[0]
      for output_index, frame_index in enumerate(selected):
        timestamp = timestamps[frame_index]
        playback_time = (
          frame_index / source_fps
          if source_fps is not None and source_fps > 0
          else output_index / fps
        )
        frame: dict[str, Any] = {
          "output_index": output_index,
          "video_frame_index": frame_index,
          "timestamp_nsec": timestamp,
          "time_s": playback_time,
          "timestamp_offset_s": (timestamp - first_timestamp) / 1_000_000_000,
        }
        for side in ("left", "right"):
          stamps, data = side_streams[side]
          tactile_index = bisect.bisect_right(stamps, timestamp) - 1
          if tactile_index < 0:
            channel_values = [[None, None, None] for _ in FINGERS]
            frame[side] = {
              "sample_index": None,
              "stamp_nsec": None,
              "age_ms": None,
              "normal": [None] * len(FINGERS),
              "tangential": [None] * len(FINGERS),
              "proximity": [None] * len(FINGERS),
            }
          else:
            values = data[tactile_index]
            channel_values = [
              [_finite_float(values[finger][channel]) for channel in CHANNEL_INDICES]
              for finger in range(len(FINGERS))
            ]
            frame[side] = {
              "sample_index": tactile_index,
              "stamp_nsec": stamps[tactile_index],
              "age_ms": (timestamp - stamps[tactile_index]) / 1_000_000,
              "normal": [values[0] for values in channel_values],
              "tangential": [values[1] for values in channel_values],
              "proximity": [values[2] for values in channel_values],
            }
          compact[side].append(channel_values)
        frames.append(frame)
      scales = {
        channel: _percentile_scale(
          compact[side][frame][finger][channel_index]
          for side in ("left", "right")
          for frame in range(len(frames))
          for finger in range(len(FINGERS))
        )
        for channel_index, channel in enumerate(CHANNELS)
      }
      return {
        "episode_id": int(identifier),
        "episode_id_text": identifier,
        "fps": fps,
        "source_fps": source_fps,
        "source_frame_count": len(timestamps),
        "fingers": list(FINGERS),
        "channels": list(CHANNELS),
        "times": [frame["time_s"] for frame in frames],
        "timestamp_times_s": [frame["timestamp_offset_s"] for frame in frames],
        "left": compact["left"],
        "right": compact["right"],
        "scales": scales,
        "alignment": "latest-nonfuture tactile sample at each selected video frame",
        "frames": frames,
      }

  @staticmethod
  def _read_video_timestamps(file: h5py.File) -> list[int]:
    dataset = file["timestamp_ros_nsec"]
    if dataset.ndim != 1 or dataset.dtype.kind not in {"i", "u"}:
      raise ValueError("timestamp_ros_nsec must be a one-dimensional integer dataset")
    timestamps = [int(value) for value in dataset[:]]
    if not timestamps:
      raise ValueError("timestamp_ros_nsec must be non-empty")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
      raise ValueError("timestamp_ros_nsec must be strictly increasing")
    return timestamps

  @staticmethod
  def _read_tactile_stream(file: h5py.File, side: str) -> tuple[list[int], Any]:
    group = file[f"raw/revo2_touch/{side}"]
    stamp_dataset = group["stamp_nsec"]
    data_dataset = group["data"]
    if stamp_dataset.ndim != 1 or stamp_dataset.dtype.kind not in {"i", "u"}:
      raise ValueError(f"{side} stamp_nsec must be a one-dimensional integer dataset")
    if data_dataset.ndim != 3 or data_dataset.shape[1:] != (5, 5):
      raise ValueError(f"{side} tactile data must have shape (N, 5, 5)")
    if data_dataset.shape[0] != stamp_dataset.shape[0] or not stamp_dataset.shape[0]:
      raise ValueError(f"{side} tactile timestamps and samples must be non-empty and aligned")
    stamps = [int(value) for value in stamp_dataset[:]]
    if any(current < previous for previous, current in zip(stamps, stamps[1:])):
      raise ValueError(f"{side} tactile timestamps must be nondecreasing")
    return stamps, data_dataset[:]

  def open_media(self, identifier: str, camera: str) -> tuple[BinaryIO, int]:
    """Open a media file while holding the mutation lock across pathname lookup."""

    identifier = _identifier(identifier)
    camera = _camera(camera)
    with self.lock:
      self._episode_path(identifier)
      matches = dict(self._video_files(identifier))
      path = matches.get(camera)
      if path is None:
        raise ApiError(
          HTTPStatus.NOT_FOUND,
          "media_not_found",
          f"Camera {camera!r} was not found for episode {identifier}",
        )
      try:
        stream = path.open("rb")
        size = os.fstat(stream.fileno()).st_size
      except OSError as error:
        raise ApiError(
          HTTPStatus.NOT_FOUND,
          "media_not_found",
          f"Camera {camera!r} was not found for episode {identifier}",
        ) from error
      return stream, size

  def delete(self, identifier: str, confirm: Any) -> dict[str, Any]:
    identifier = _identifier(identifier)
    expected = f"episode_{identifier}"
    if confirm != expected:
      raise ApiError(
        HTTPStatus.BAD_REQUEST,
        "confirmation_mismatch",
        f"confirm must exactly equal {expected!r}",
      )
    with self.lock:
      episode_path = self._episode_path(identifier)
      sources = [episode_path, *(path for _, path in self._video_files(identifier))]
      relative_paths = [path.relative_to(self.root).as_posix() for path in sources]
      trash_root = self._ensure_trash_root()
      token, entry = self._new_trash_entry(trash_root, identifier)
      manifest = {
        "version": 1,
        "token": token,
        "episode": expected,
        "identifier": identifier,
        "created_at": datetime.now(UTC).isoformat(),
        "state": "moving",
        "files": relative_paths,
      }
      self._atomic_json(entry / "manifest.json", manifest)
      moved: list[tuple[Path, Path]] = []
      try:
        for source, relative in zip(sources, relative_paths, strict=True):
          destination = entry / "files" / PurePosixPath(relative)
          destination.parent.mkdir(parents=True, exist_ok=True)
          os.replace(source, destination)
          moved.append((source, destination))
        manifest["state"] = "deleted"
        self._atomic_json(entry / "manifest.json", manifest)
      except Exception as error:
        rollback_errors = self._rollback_moves(moved)
        if not rollback_errors:
          self._cleanup_entry(entry)
          raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "delete_failed",
            f"Could not delete {expected}; all moved files were rolled back",
          ) from error
        raise ApiError(
          HTTPStatus.INTERNAL_SERVER_ERROR,
          "rollback_failed",
          f"Could not fully roll back {expected}; preserve trash token {token}",
          details={"token": token, "errors": rollback_errors},
        ) from error
      return {
        "deleted": True,
        "token": token,
        "episode": expected,
        "files": relative_paths,
      }

  def restore(self, token: str) -> dict[str, Any]:
    if TOKEN_PATTERN.fullmatch(token) is None:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_token", "Invalid trash token")
    with self.lock:
      trash_root = self._existing_trash_root()
      entry = trash_root / token
      if entry.is_symlink() or not entry.is_dir():
        raise ApiError(HTTPStatus.NOT_FOUND, "trash_not_found", "Trash entry not found")
      resolved_entry = entry.resolve(strict=True)
      if resolved_entry.parent != trash_root:
        raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe trash entry")
      manifest = self._read_manifest(entry, token)
      identifier = _identifier(manifest.get("identifier", ""))
      episode = f"episode_{identifier}"
      if manifest.get("episode") != episode or manifest.get("state") != "deleted":
        raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Trash entry is incomplete")
      relative_paths = self._validated_manifest_paths(manifest, identifier)
      moves: list[tuple[Path, Path]] = []
      conflicts = []
      for relative in relative_paths:
        source = entry / "files" / PurePosixPath(relative)
        destination = self.root / PurePosixPath(relative)
        if source.is_symlink() or not source.is_file():
          raise ApiError(
            HTTPStatus.CONFLICT,
            "incomplete_trash",
            f"Trash entry is missing {relative}",
          )
        if not _is_within(source.resolve(strict=True), resolved_entry):
          raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe trash file path")
        if destination.exists() or destination.is_symlink():
          conflicts.append(relative)
        moves.append((source, destination))
      if conflicts:
        raise ApiError(
          HTTPStatus.CONFLICT,
          "restore_conflict",
          "Restore would overwrite existing files",
          details={"files": conflicts},
        )
      moved: list[tuple[Path, Path]] = []
      try:
        for source, destination in moves:
          self._validate_restore_parent(destination.parent)
          destination.parent.mkdir(parents=True, exist_ok=True)
          if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
          os.replace(source, destination)
          moved.append((source, destination))
      except Exception as error:
        rollback_errors = self._rollback_restore(moved)
        code = "restore_failed" if not rollback_errors else "restore_rollback_failed"
        raise ApiError(
          HTTPStatus.INTERNAL_SERVER_ERROR,
          code,
          f"Could not restore {episode}; restored files were moved back to trash",
          details={"errors": rollback_errors} if rollback_errors else None,
        ) from error
      self._cleanup_entry(entry)
      return {
        "restored": True,
        "token": token,
        "episode": episode,
        "files": relative_paths,
      }

  def _ensure_trash_root(self) -> Path:
    path = self.trash_root
    if path.is_symlink():
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Trash directory is a symlink")
    try:
      path.mkdir(mode=0o700, exist_ok=True)
      resolved = path.resolve(strict=True)
    except OSError as error:
      raise ApiError(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "trash_unavailable",
        "Could not create the trash directory",
      ) from error
    if resolved.parent != self.root:
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe trash directory")
    return resolved

  def _existing_trash_root(self) -> Path:
    path = self.trash_root
    if path.is_symlink() or not path.is_dir():
      raise ApiError(HTTPStatus.NOT_FOUND, "trash_not_found", "Trash entry not found")
    resolved = path.resolve(strict=True)
    if resolved.parent != self.root:
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe trash directory")
    return resolved

  @staticmethod
  def _new_trash_entry(trash_root: Path, identifier: str) -> tuple[str, Path]:
    for _ in range(10):
      token = f"episode_{identifier}_{time.time_ns()}_{secrets.token_hex(6)}"
      entry = trash_root / token
      try:
        entry.mkdir(mode=0o700)
      except FileExistsError:
        continue
      return token, entry
    raise ApiError(
      HTTPStatus.INTERNAL_SERVER_ERROR,
      "trash_unavailable",
      "Could not allocate a unique trash entry",
    )

  @staticmethod
  def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
      with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
      os.replace(temporary, path)
    finally:
      if temporary.exists():
        temporary.unlink()

  @staticmethod
  def _rollback_moves(moved: list[tuple[Path, Path]]) -> list[str]:
    errors = []
    for source, destination in reversed(moved):
      try:
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() or source.is_symlink():
          raise FileExistsError(source)
        os.replace(destination, source)
      except Exception as error:
        errors.append(f"{destination}: {error}")
    return errors

  @staticmethod
  def _rollback_restore(moved: list[tuple[Path, Path]]) -> list[str]:
    errors = []
    for source, destination in reversed(moved):
      try:
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() or source.is_symlink():
          raise FileExistsError(source)
        os.replace(destination, source)
      except Exception as error:
        errors.append(f"{destination}: {error}")
    return errors

  @staticmethod
  def _cleanup_entry(entry: Path) -> None:
    if not entry.exists():
      return
    paths = sorted(entry.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
      if path.is_symlink() or path.is_file():
        path.unlink()
      elif path.is_dir():
        path.rmdir()
    entry.rmdir()

  @staticmethod
  def _read_manifest(entry: Path, token: str) -> dict[str, Any]:
    path = entry / "manifest.json"
    try:
      if path.stat().st_size > MAX_JSON_BODY_BYTES:
        raise ValueError("manifest is too large")
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
      raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest") from error
    if not isinstance(payload, dict) or payload.get("token") != token:
      raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
    return payload

  @staticmethod
  def _validated_manifest_paths(manifest: dict[str, Any], identifier: str) -> list[str]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
      raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
    expected_hdf5 = f"episode_{identifier}.hdf5"
    expected_video = f"episode_{identifier}.mp4"
    result = []
    for value in files:
      if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
      path = PurePosixPath(value)
      if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
      valid = value == expected_hdf5 or (
        len(path.parts) == 3
        and path.parts[0] == "video"
        and _valid_manifest_camera(path.parts[1])
        and path.parts[2] == expected_video
      )
      if not valid or value in result:
        raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
      result.append(value)
    if expected_hdf5 not in result:
      raise ApiError(HTTPStatus.CONFLICT, "invalid_manifest", "Invalid trash manifest")
    return result

  def _validate_restore_parent(self, parent: Path) -> None:
    existing = parent
    while not existing.exists():
      if existing == self.root:
        break
      existing = existing.parent
    if existing.is_symlink():
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe restore path")
    resolved = existing.resolve(strict=True)
    if not _is_within(resolved, self.root):
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe restore path")


def _valid_manifest_camera(value: str) -> bool:
  return value not in {".", ".."} and CAMERA_PATTERN.fullmatch(value) is not None


class EpisodeBrowserServer(ThreadingHTTPServer):
  """Threaded HTTP server carrying immutable roots and a synchronized store."""

  daemon_threads = True

  def __init__(
    self,
    address: tuple[str, int],
    dataset_root: str | Path,
    static_root: str | Path | None = None,
  ) -> None:
    self.store = EpisodeStore(dataset_root)
    default_static = Path(__file__).with_name("episode_browser_web")
    self.static_root = Path(static_root or default_static).expanduser().resolve()
    super().__init__(address, EpisodeBrowserHandler)


class EpisodeBrowserHandler(BaseHTTPRequestHandler):
  """Same-origin JSON API, fixed static assets, and seekable MP4 media."""

  protocol_version = "HTTP/1.1"
  server_version = "EpisodeBrowser/1.0"

  @property
  def browser_server(self) -> EpisodeBrowserServer:
    return self.server  # type: ignore[return-value]

  def do_GET(self) -> None:
    self._handle("GET")

  def do_HEAD(self) -> None:
    self._handle("HEAD")

  def do_DELETE(self) -> None:
    self._handle("DELETE")

  def do_POST(self) -> None:
    self._handle("POST")

  def do_PUT(self) -> None:
    self._handle("PUT")

  def do_PATCH(self) -> None:
    self._handle("PATCH")

  def do_OPTIONS(self) -> None:
    self._handle("OPTIONS")

  def _handle(self, method: str) -> None:
    try:
      split = urlsplit(self.path)
      try:
        path = unquote(split.path, errors="strict")
      except UnicodeDecodeError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_path", "Invalid URL encoding") from error
      if "\x00" in path:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_path", "Invalid URL path")
      if method in {"GET", "HEAD"}:
        self._route_read(path, split.query, head=method == "HEAD")
      elif method == "DELETE":
        self._route_delete(path, split.query)
      elif method == "POST":
        self._route_post(path, split.query)
      else:
        self._method_not_allowed()
    except ApiError as error:
      self._send_json(error.status, error.payload(), head=method == "HEAD", headers=error.headers)
    except Exception:
      self.log_error("Unhandled request failure")
      self._send_json(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        {"error": {"code": "internal_error", "message": "Internal server error"}},
        head=method == "HEAD",
      )

  def _route_read(self, path: str, query: str, *, head: bool) -> None:
    if path in STATIC_FILES:
      if query:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
      self._send_static(path, head=head)
      return
    if path == "/api/dataset":
      if head:
        self._method_not_allowed(allow="GET")
        return
      if query:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
      self._send_json(HTTPStatus.OK, self.browser_server.store.dataset())
      return
    match = re.fullmatch(r"/api/episodes/(\d{1,64})", path)
    if match is not None:
      if head:
        self._method_not_allowed(allow="GET")
        return
      if query:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
      self._send_json(HTTPStatus.OK, self.browser_server.store.detail(match.group(1)))
      return
    match = re.fullmatch(r"/api/episodes/(\d{1,64})/telemetry", path)
    if match is not None:
      if head:
        self._method_not_allowed(allow="GET")
        return
      parameters = parse_qs(query, keep_blank_values=True, strict_parsing=False)
      if set(parameters) != {"fps"} or len(parameters["fps"]) != 1:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_fps", "fps must be either 5 or 10")
      try:
        fps = int(parameters["fps"][0])
      except ValueError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_fps", "fps must be either 5 or 10") from error
      self._send_json(
        HTTPStatus.OK,
        self.browser_server.store.telemetry(match.group(1), fps),
      )
      return
    match = re.fullmatch(r"/media/(\d{1,64})/([^/]+)", path)
    if match is not None:
      if query:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
      self._send_media(match.group(1), match.group(2), head=head)
      return
    raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "Route not found")

  def _route_delete(self, path: str, query: str) -> None:
    match = re.fullmatch(r"/api/episodes/(\d{1,64})", path)
    if match is None:
      raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "Route not found")
    if query:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
    body = self._read_json_object()
    result = self.browser_server.store.delete(match.group(1), body.get("confirm"))
    self._send_json(HTTPStatus.OK, result)

  def _route_post(self, path: str, query: str) -> None:
    match = re.fullmatch(r"/api/trash/([A-Za-z0-9_-]{1,200})/restore", path)
    if match is None:
      raise ApiError(HTTPStatus.NOT_FOUND, "not_found", "Route not found")
    if query:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_query", "Unexpected query string")
    self._require_empty_body()
    result = self.browser_server.store.restore(match.group(1))
    self._send_json(HTTPStatus.OK, result)

  def _read_json_object(self) -> dict[str, Any]:
    if self.headers.get("Transfer-Encoding"):
      raise ApiError(
        HTTPStatus.BAD_REQUEST,
        "invalid_body",
        "Transfer-Encoding request bodies are not supported",
      )
    raw_length = self.headers.get("Content-Length")
    try:
      length = int(raw_length) if raw_length is not None else 0
    except ValueError as error:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_body", "Invalid Content-Length") from error
    if length <= 0 or length > MAX_JSON_BODY_BYTES:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_body", "A small JSON body is required")
    try:
      payload = json.loads(self.rfile.read(length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json", "Request body must be valid JSON") from error
    if not isinstance(payload, dict):
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json", "Request body must be a JSON object")
    return payload

  def _require_empty_body(self) -> None:
    raw_length = self.headers.get("Content-Length")
    if raw_length is None:
      return
    try:
      length = int(raw_length)
    except ValueError as error:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_body", "Invalid Content-Length") from error
    if length != 0:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_body", "This endpoint takes no body")

  def _send_static(self, route: str, *, head: bool) -> None:
    filename, content_type = STATIC_FILES[route]
    root = self.browser_server.static_root
    path = root / filename
    try:
      resolved = path.resolve(strict=True)
    except OSError as error:
      raise ApiError(
        HTTPStatus.SERVICE_UNAVAILABLE,
        "static_unavailable",
        "Episode browser assets are not installed",
      ) from error
    if not _is_within(resolved, root) or not resolved.is_file():
      raise ApiError(HTTPStatus.FORBIDDEN, "unsafe_path", "Unsafe static asset path")
    try:
      content = resolved.read_bytes()
    except OSError as error:
      raise ApiError(
        HTTPStatus.SERVICE_UNAVAILABLE,
        "static_unavailable",
        "Could not read episode browser assets",
      ) from error
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(content)))
    self.send_header("Cache-Control", "no-cache")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    if not head:
      self.wfile.write(content)

  def _send_media(self, identifier: str, camera: str, *, head: bool) -> None:
    stream, size = self.browser_server.store.open_media(identifier, camera)
    with stream:
      ranges = self.headers.get_all("Range", failobj=[])
      if len(ranges) > 1:
        raise self._range_error(size)
      if ranges:
        try:
          start, end = parse_single_range(ranges[0], size)
        except ValueError as error:
          raise self._range_error(size) from error
        status = HTTPStatus.PARTIAL_CONTENT
      else:
        start, end = 0, size - 1
        status = HTTPStatus.OK
      length = 0 if size == 0 else end - start + 1
      self.send_response(status)
      self.send_header("Content-Type", "video/mp4")
      self.send_header("Accept-Ranges", "bytes")
      self.send_header("Content-Length", str(length))
      self.send_header("Cache-Control", "no-cache")
      self.send_header("X-Content-Type-Options", "nosniff")
      if status == HTTPStatus.PARTIAL_CONTENT:
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
      self.end_headers()
      if head or not length:
        return
      stream.seek(start)
      remaining = length
      try:
        while remaining:
          chunk = stream.read(min(1024 * 1024, remaining))
          if not chunk:
            break
          self.wfile.write(chunk)
          remaining -= len(chunk)
      except (BrokenPipeError, ConnectionResetError):
        return

  @staticmethod
  def _range_error(size: int) -> ApiError:
    return ApiError(
      HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
      "invalid_range",
      "The requested byte range is not satisfiable",
      headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
    )

  def _send_json(
    self,
    status: int,
    payload: Any,
    *,
    head: bool = False,
    headers: dict[str, str] | None = None,
  ) -> None:
    content = _strict_json_bytes(payload)
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.send_header("Content-Length", str(len(content)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Content-Type-Options", "nosniff")
    for key, value in (headers or {}).items():
      self.send_header(key, value)
    self.end_headers()
    if not head:
      self.wfile.write(content)

  def _method_not_allowed(self, *, allow: str = "GET, HEAD, DELETE, POST") -> None:
    raise ApiError(
      HTTPStatus.METHOD_NOT_ALLOWED,
      "method_not_allowed",
      "Method not allowed",
      headers={"Allow": allow},
    )


def create_server(
  dataset_root: str | Path,
  host: str = "127.0.0.1",
  port: int = 8000,
  *,
  static_root: str | Path | None = None,
) -> EpisodeBrowserServer:
  return EpisodeBrowserServer((host, port), dataset_root, static_root)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Browse and safely curate external-video HDF5 episodes in a web UI."
  )
  parser.add_argument("dataset", type=Path, help="Dataset directory containing episode_*.hdf5")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
  args = parser.parse_args(argv)
  if not 0 <= args.port <= 65535:
    parser.error("--port must be between 0 and 65535")
  return args


def _browser_url(host: str, port: int) -> str:
  display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
  if ":" in display_host and not display_host.startswith("["):
    display_host = f"[{display_host}]"
  return f"http://{display_host}:{port}/"


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv)
  try:
    server = create_server(args.dataset, args.host, args.port)
  except (OSError, ValueError) as error:
    raise SystemExit(str(error)) from error
  host, port = server.server_address[:2]
  url = _browser_url(str(host), int(port))
  summary = server.store.dataset()
  print(
    f"[episode-browser] {summary['count']} episode(s) in {server.store.root}",
    flush=True,
  )
  print(f"[episode-browser] serving {url}", flush=True)
  if not args.no_browser:
    opener = threading.Timer(0.25, webbrowser.open, args=(url,))
    opener.daemon = True
    opener.start()
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\n[episode-browser] stopping", flush=True)
  finally:
    server.server_close()


if __name__ == "__main__":
  main()
