"""Parallel, resumable and PNG-free LeRobot v2.1 publication.

The pinned OpenPI task converters remain the authority for source validation,
clock alignment and action semantics.  This module replaces only their output
stage: independent episodes are converted in forked worker processes, HDF5 RGB
arrays are streamed directly to FFmpeg, and the committed episode artifacts
are assembled into a standard video-backed LeRobot v2.1 dataset.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA = "kaihand_fast_pi05_video_v1"
FPS = 30
ROBOT_TYPE = "kaihand_right"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = 18
FFMPEG_THREADS_PER_WORKER = 1
FRAME_BATCH = 32

_CONVERTER: Any | None = None
_ADAPTER_PATH: Path | None = None


def _jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  return value


def _write_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
  try:
    with temporary.open("x", encoding="utf-8") as stream:
      json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _write_json_lines(path: Path, values: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
  try:
    with temporary.open("x", encoding="utf-8") as stream:
      for value in values:
        stream.write(
          json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n"
        )
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"cannot read JSON object: {path}") from error
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _fingerprint(value: Any) -> str:
  payload = json.dumps(
    _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def _conversion_source_hashes() -> dict[str, str]:
  if _CONVERTER is None or not getattr(_CONVERTER, "__file__", None):
    raise RuntimeError("fast converter has no source file to freeze")
  sources = {
    Path(__file__).resolve(),
    Path(_CONVERTER.__file__).resolve(),
    Path(__file__).with_name("unified_lerobot_collection.py").resolve(),
  }
  common = getattr(_CONVERTER, "common", None)
  if common is not None and getattr(common, "__file__", None):
    sources.add(Path(common.__file__).resolve())
  if _ADAPTER_PATH is not None:
    sources.add(_ADAPTER_PATH.resolve())
  return {str(path): _sha256_file(path) for path in sorted(sources)}


def _fsync_file(path: Path) -> None:
  with path.open("rb") as stream:
    os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
  try:
    with contextlib.suppress(OSError):
      os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _is_path(path: Path) -> bool:
  return os.path.lexists(path)


def _source_identity(source: Any) -> dict[str, Any]:
  return {
    "episode_index": source.episode_index,
    "hdf5": {
      "path": str(source.hdf5.path),
      "sha256": source.hdf5.sha256,
      "size_bytes": source.hdf5.size_bytes,
      "mtime_ns": source.hdf5.mtime_ns,
      "device": source.hdf5.device,
      "inode": source.hdf5.inode,
    },
    "sidecar": {
      "path": str(source.sidecar.path),
      "sha256": source.sidecar.sha256,
      "size_bytes": source.sidecar.size_bytes,
      "mtime_ns": source.sidecar.mtime_ns,
      "device": source.sidecar.device,
      "inode": source.sidecar.inode,
    },
    "task": getattr(source, "task", "pick-place"),
    "cameras": list(getattr(source, "camera_names", ("head",))),
    "instruction": getattr(source, "instruction", None),
    "repo_id": getattr(source, "repo_id", None),
  }


def _plan_identity(plan: Any) -> dict[str, Any]:
  scalar_fields = {
    key: value
    for key, value in vars(plan).items()
    if key != "source"
  }
  return {
    "source": _source_identity(plan.source),
    "plan": _jsonable(scalar_fields),
  }


def _source_manifest_rows(
  plans: tuple[Any, ...], *, verify_source_hash: bool
) -> list[dict[str, Any]]:
  rows = []
  for output_index, plan in enumerate(plans):
    source = _source_identity(plan.source)
    rows.append({
      "output_episode_index": output_index,
      "source_episode_index": source["episode_index"],
      "source_hdf5": source["hdf5"]["path"],
      "source_hdf5_sha256": source["hdf5"]["sha256"],
      "source_hdf5_size_bytes": source["hdf5"]["size_bytes"],
      "source_sidecar": source["sidecar"]["path"],
      "source_hash_verification": (
        "recomputed" if verify_source_hash else "trusted_acquisition_sidecar"
      ),
      "task": source["task"],
      "exported_frames_30hz": int(plan.exported_frames),
    })
  return rows


def _camera_names(plan: Any) -> tuple[str, ...]:
  return tuple(getattr(plan.source, "camera_names", ("head",)))


def _instruction(plan: Any) -> str:
  if getattr(plan.source, "instruction", None):
    return plan.source.instruction
  if _CONVERTER is None:
    raise RuntimeError("fast converter is not installed")
  return _CONVERTER.TASK_INSTRUCTION


def _repo_id(plan: Any) -> str:
  value = getattr(plan.source, "repo_id", None)
  if value:
    return value
  if _CONVERTER is None:
    raise RuntimeError("fast converter is not installed")
  return _CONVERTER.REPO_ID


def _base_features(plan: Any) -> dict[str, dict[str, Any]]:
  if _CONVERTER is None:
    raise RuntimeError("fast converter is not installed")
  if hasattr(plan, "camera_shapes"):
    features = _CONVERTER._lerobot_features(plan)
  else:
    features = _CONVERTER._lerobot_features(plan.image_height, plan.image_width)
  result = {key: dict(value) for key, value in features.items()}
  for name in _camera_names(plan):
    key = f"observation.images.{name}"
    if key not in result:
      raise ValueError(f"converter omitted selected camera feature {key}")
    result[key]["dtype"] = "video"
  return result


def _camera_shape(plan: Any, name: str) -> tuple[int, int]:
  if hasattr(plan, "camera_shapes"):
    shapes = {
      camera: (int(height), int(width))
      for camera, height, width in plan.camera_shapes
    }
    return shapes[name]
  if name != "head":
    raise ValueError(f"pick-place only supports head, got {name}")
  return int(plan.image_height), int(plan.image_width)


def _episode_arrays(
  plan: Any,
  loaded: tuple[Any, ...],
  *,
  output_episode_index: int,
  global_start_index: int,
) -> dict[str, np.ndarray]:
  if len(loaded) == 5:
    observed, camera_timestamps, camera_state_indices, positions, actions = loaded
  elif len(loaded) == 6:
    (
      observed,
      camera_timestamps,
      camera_state_indices,
      _,
      positions,
      actions,
    ) = loaded
  else:
    raise RuntimeError(f"unsupported converter load result length: {len(loaded)}")
  if observed != plan:
    raise RuntimeError(f"preflight changed for {plan.source.hdf5.path}")
  frames = int(plan.exported_frames)
  local = np.arange(frames, dtype=np.int64)
  return {
    "observation.state": np.asarray(positions[:frames], dtype=np.float32),
    "action": np.asarray(actions[1 : frames + 1], dtype=np.float32),
    "provenance.source_episode_index": np.full(
      frames, plan.source.episode_index, dtype=np.int64
    ),
    "provenance.source_camera_frame_index": local,
    "provenance.source_camera_timestamp_s": np.asarray(
      camera_timestamps[:frames], dtype=np.float64
    ),
    "provenance.source_state_index": np.asarray(
      camera_state_indices[:frames], dtype=np.int64
    ),
    "provenance.action_source_camera_frame_index": local + 1,
    "provenance.action_source_camera_timestamp_s": np.asarray(
      camera_timestamps[1 : frames + 1], dtype=np.float64
    ),
    "provenance.action_source_state_index": np.asarray(
      camera_state_indices[1 : frames + 1], dtype=np.int64
    ),
    "timestamp": local.astype(np.float32) / np.float32(FPS),
    "frame_index": local,
    "episode_index": np.full(frames, output_episode_index, dtype=np.int64),
    "index": global_start_index + local,
    "task_index": np.zeros(frames, dtype=np.int64),
  }


def _write_episode_parquet(
  path: Path,
  arrays: dict[str, np.ndarray],
  features: dict[str, dict[str, Any]],
) -> None:
  import datasets
  from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    get_hf_features_from_features,
  )

  all_features = {**features, **DEFAULT_FEATURES}
  hf_features = get_hf_features_from_features(all_features)
  values: dict[str, np.ndarray] = {}
  for key in hf_features:
    value = np.asarray(arrays[key])
    if all_features[key]["shape"] == (1,) and value.ndim == 2:
      value = value[:, 0]
    values[key] = value
  episode = datasets.Dataset.from_dict(values, features=hf_features, split="train")
  episode.to_parquet(path)


def _numeric_stats(
  arrays: dict[str, np.ndarray],
  features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
  from lerobot.common.datasets.compute_stats import compute_episode_stats
  from lerobot.common.datasets.utils import DEFAULT_FEATURES

  all_features = {**features, **DEFAULT_FEATURES}
  numeric = {
    key: value
    for key, value in arrays.items()
    if all_features[key]["dtype"] not in {"image", "video", "string"}
  }
  return compute_episode_stats(numeric, all_features)


def _encode_hdf5_video(
  rgb: h5py.Dataset,
  output: Path,
  *,
  frames: int,
  width: int,
  height: int,
) -> dict[str, np.ndarray]:
  """Stream RGB directly to H.264 and compute LeRobot-compatible stats."""

  from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices

  command = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
    "-f", "rawvideo", "-pixel_format", "rgb24",
    "-video_size", f"{width}x{height}", "-framerate", str(FPS),
    "-i", "pipe:0", "-an", "-frames:v", str(frames),
    "-c:v", "libx264", "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
    "-threads", str(FFMPEG_THREADS_PER_WORKER), "-pix_fmt", "yuv420p",
    "-g", "2", "-keyint_min", "2", "-sc_threshold", "0",
    "-movflags", "+faststart", str(output),
  ]
  process = subprocess.Popen(
    command,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )
  assert process.stdin is not None
  assert process.stderr is not None
  selected_indices = np.asarray(sample_indices(frames), dtype=np.int64)
  sampled_images: list[np.ndarray] = []
  maximum_size = max(width, height)
  downsample = (
    int(width / 150) if width > height else int(height / 150)
  ) if maximum_size >= 300 else 1
  try:
    for start in range(0, frames, FRAME_BATCH):
      stop = min(start + FRAME_BATCH, frames)
      batch = np.ascontiguousarray(rgb[start:stop], dtype=np.uint8)
      if batch.shape != (stop - start, height, width, 3):
        raise ValueError(f"{rgb.name}: RGB shape changed during conversion")
      process.stdin.write(batch.tobytes(order="C"))
      selected = selected_indices[
        (selected_indices >= start) & (selected_indices < stop)
      ]
      sampled_images.extend(
        np.moveaxis(batch[index - start], -1, 0)[
          :, ::downsample, ::downsample
        ].copy()
        for index in selected
      )
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
  except BaseException:
    process.kill()
    process.wait()
    raise
  if return_code != 0:
    raise RuntimeError(
      f"FFmpeg failed for {output} with exit code {return_code}: {stderr[-4000:]}"
    )
  images = np.stack(sampled_images).astype(np.float32)
  stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
  return {
    key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
    for key, value in stats.items()
  }


def _probe_video(
  path: Path,
  *,
  frames: int,
  width: int,
  height: int,
) -> None:
  command = [
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
    "-of", "json", str(path),
  ]
  result = subprocess.run(command, check=True, capture_output=True, text=True)
  streams = json.loads(result.stdout).get("streams", [])
  if len(streams) != 1:
    raise ValueError(f"{path}: expected exactly one video stream")
  stream = streams[0]
  if (
    stream.get("codec_name") != "h264"
    or stream.get("pix_fmt") != "yuv420p"
    or int(stream.get("width", -1)) != width
    or int(stream.get("height", -1)) != height
    or stream.get("avg_frame_rate") != f"{FPS}/1"
    or int(stream.get("nb_frames", -1)) != frames
  ):
    raise ValueError(f"{path}: encoded video contract mismatch: {stream}")


def _worker_episode(job: dict[str, Any]) -> dict[str, Any]:
  if _CONVERTER is None:
    raise RuntimeError("worker did not inherit the installed task converter")
  plan = job["plan"]
  output_index = int(job["output_index"])
  partial = Path(job["partial"])
  committed = Path(job["committed"])
  partial.mkdir(parents=True)
  try:
    _assert_source_unchanged(plan.source)
    with h5py.File(plan.source.hdf5.path, "r") as file:
      loaded = _CONVERTER._load_episode_timing_and_positions(file, plan.source)
      arrays = _episode_arrays(
        plan,
        loaded,
        output_episode_index=output_index,
        global_start_index=int(job["global_start_index"]),
      )
      features = _base_features(plan)
      _write_episode_parquet(partial / "data.parquet", arrays, features)
      stats = _numeric_stats(arrays, features)
      for camera in _camera_names(plan):
        height, width = _camera_shape(plan, camera)
        video = partial / f"{camera}.mp4"
        stats[f"observation.images.{camera}"] = _encode_hdf5_video(
          file[f"cameras/{camera}/rgb"],
          video,
          frames=plan.exported_frames,
          width=width,
          height=height,
        )
        _probe_video(
          video,
          frames=plan.exported_frames,
          width=width,
          height=height,
        )
      _write_json(partial / "stats.json", stats)
    _assert_source_unchanged(plan.source)

    artifacts = {
      path.name: {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
      }
      for path in sorted(partial.iterdir())
      if path.is_file()
    }
    done = {
      "schema": SCHEMA,
      "plan_fingerprint": job["plan_fingerprint"],
      "output_episode_index": output_index,
      "source_episode_index": plan.source.episode_index,
      "source_hdf5_sha256": plan.source.hdf5.sha256,
      "exported_frames": plan.exported_frames,
      "global_start_index": int(job["global_start_index"]),
      "artifacts": artifacts,
      "completed_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(partial / "done.json", done)
    for path in partial.iterdir():
      if path.is_file():
        _fsync_file(path)
    _fsync_directory(partial)
    os.replace(partial, committed)
    _fsync_directory(committed.parent)
    return done
  except BaseException:
    shutil.rmtree(partial, ignore_errors=True)
    raise


def _completed_episode(
  path: Path,
  *,
  plan: Any,
  output_index: int,
  global_start_index: int,
  plan_fingerprint: str,
) -> dict[str, Any] | None:
  done_path = path / "done.json"
  if not done_path.is_file():
    return None
  try:
    done = _load_object(done_path)
  except ValueError:
    return None
  if (
    done.get("schema") != SCHEMA
    or done.get("plan_fingerprint") != plan_fingerprint
    or done.get("output_episode_index") != output_index
    or done.get("source_episode_index") != plan.source.episode_index
    or done.get("source_hdf5_sha256") != plan.source.hdf5.sha256
    or done.get("exported_frames") != plan.exported_frames
    or done.get("global_start_index") != global_start_index
  ):
    return None
  artifacts = done.get("artifacts")
  expected = {"data.parquet", "stats.json", *{
    f"{camera}.mp4" for camera in _camera_names(plan)
  }}
  if not isinstance(artifacts, dict) or set(artifacts) != expected:
    return None
  for name, identity in artifacts.items():
    artifact = path / name
    if (
      not isinstance(identity, dict)
      or not artifact.is_file()
      or artifact.stat().st_size != identity.get("size_bytes")
      or _sha256_file(artifact) != identity.get("sha256")
    ):
      return None
  return done


def _run_workers(
  plans: tuple[Any, ...],
  *,
  work_root: Path,
  workers: int,
  plan_fingerprint: str,
) -> tuple[dict[str, Any], ...]:
  episodes = work_root / "episodes"
  episodes.mkdir(parents=True, exist_ok=True)
  for partial in episodes.glob(".episode-*.partial-*"):
    if partial.is_dir():
      shutil.rmtree(partial)

  offsets: list[int] = []
  running = 0
  for plan in plans:
    offsets.append(running)
    running += int(plan.exported_frames)

  results: list[dict[str, Any] | None] = [None] * len(plans)
  pending: list[dict[str, Any]] = []
  for output_index, (plan, offset) in enumerate(zip(plans, offsets, strict=True)):
    committed = episodes / f"episode-{output_index:06d}"
    completed = _completed_episode(
      committed,
      plan=plan,
      output_index=output_index,
      global_start_index=offset,
      plan_fingerprint=plan_fingerprint,
    )
    if completed is not None:
      results[output_index] = completed
      continue
    if committed.exists():
      raise RuntimeError(
        f"committed checkpoint does not match the conversion plan: {committed}"
      )
    pending.append({
      "plan": plan,
      "output_index": output_index,
      "global_start_index": offset,
      "committed": str(committed),
      "partial": str(
        episodes / f".episode-{output_index:06d}.partial-{uuid.uuid4().hex}"
      ),
      "plan_fingerprint": plan_fingerprint,
    })
  print(
    f"fast pi0.5 resume: completed={len(plans) - len(pending)} "
    f"pending={len(pending)} episode_workers={workers} "
    f"ffmpeg_threads_per_worker={FFMPEG_THREADS_PER_WORKER}",
    flush=True,
  )
  if pending:
    executor = concurrent.futures.ProcessPoolExecutor(
      max_workers=workers,
      mp_context=multiprocessing.get_context("fork"),
    )
    futures = {
      executor.submit(_worker_episode, job): job
      for job in pending
    }
    try:
      converted = 0
      for future in concurrent.futures.as_completed(futures):
        job = futures[future]
        result = future.result()
        output_index = int(job["output_index"])
        results[output_index] = result
        converted += 1
        print(
          f"convert [{len(plans) - len(pending) + converted}/{len(plans)}] "
          f"output={output_index:06d} "
          f"source={result['source_episode_index']:06d} "
          f"frames={result['exported_frames']}",
          flush=True,
        )
    except BaseException:
      for future in futures:
        future.cancel()
      for process in getattr(executor, "_processes", {}).values():
        process.terminate()
      executor.shutdown(wait=True, cancel_futures=True)
      raise
    else:
      executor.shutdown(wait=True)
  if any(result is None for result in results):
    raise RuntimeError("conversion worker result set is incomplete")
  return tuple(result for result in results if result is not None)


def _stats_arrays(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _stats_arrays(item) for key, item in value.items()}
  if isinstance(value, list):
    return np.asarray(value)
  return value


def _link_or_copy(source: Path, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  try:
    os.link(source, destination)
  except OSError:
    shutil.copy2(source, destination)


def _validate_dataset(root: Path, plans: tuple[Any, ...]) -> None:
  from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
  )

  metadata = LeRobotDatasetMetadata(repo_id=_repo_id(plans[0]), root=root)
  expected_frames = sum(int(plan.exported_frames) for plan in plans)
  if CODEBASE_VERSION != "v2.1" or metadata.info.get("codebase_version") != "v2.1":
    raise RuntimeError(f"{root}: expected LeRobot v2.1")
  if metadata.total_episodes != len(plans) or metadata.total_frames != expected_frames:
    raise RuntimeError(f"{root}: metadata totals disagree with conversion plan")
  if metadata.info.get("total_videos") != sum(len(_camera_names(plan)) for plan in plans):
    raise RuntimeError(f"{root}: video count disagrees with conversion plan")
  for output_index, plan in enumerate(plans):
    parquet = root / metadata.get_data_file_path(output_index)
    if not parquet.is_file():
      raise RuntimeError(f"{root}: missing episode parquet {parquet}")
    for camera in _camera_names(plan):
      video = root / metadata.get_video_file_path(
        output_index, f"observation.images.{camera}"
      )
      if not video.is_file():
        raise RuntimeError(f"{root}: missing episode video {video}")

  dataset = LeRobotDataset(
    repo_id=_repo_id(plans[0]),
    root=root,
    video_backend="pyav",
  )
  if dataset.num_episodes != len(plans) or dataset.num_frames != expected_frames:
    raise RuntimeError(f"{root}: official reader totals disagree")
  probes = sorted({0, expected_frames // 2, expected_frames - 1})
  for index in probes:
    sample = dataset[index]
    for camera in _camera_names(plans[0]):
      key = f"observation.images.{camera}"
      height, width = _camera_shape(plans[0], camera)
      if tuple(sample[key].shape) != (3, height, width):
        raise RuntimeError(f"{root}: official reader decoded invalid {key}")


def _assemble(
  input_dir: Path,
  output_dir: Path,
  plans: tuple[Any, ...],
  results: tuple[dict[str, Any], ...],
  *,
  work_root: Path,
  plan_fingerprint: str,
  available_source_episodes: int,
  expected_episodes: int,
  selection_limit: int | None,
  verify_source_hash: bool,
  workers: int,
) -> Path:
  from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

  staging = work_root / "dataset"
  if staging.exists():
    shutil.rmtree(staging)
  features = _base_features(plans[0])
  metadata = LeRobotDatasetMetadata.create(
    repo_id=_repo_id(plans[0]),
    fps=FPS,
    root=staging,
    robot_type=ROBOT_TYPE,
    features=features,
    use_videos=True,
  )
  instruction = _instruction(plans[0])
  metadata.add_task(instruction)
  for output_index, (plan, _result) in enumerate(zip(plans, results, strict=True)):
    checkpoint = work_root / "episodes" / f"episode-{output_index:06d}"
    _link_or_copy(
      checkpoint / "data.parquet",
      staging / metadata.get_data_file_path(output_index),
    )
    for camera in _camera_names(plan):
      key = f"observation.images.{camera}"
      _link_or_copy(
        checkpoint / f"{camera}.mp4",
        staging / metadata.get_video_file_path(output_index, key),
      )
    stats = _stats_arrays(_load_object(checkpoint / "stats.json"))
    metadata.save_episode(
      episode_index=output_index,
      episode_length=int(plan.exported_frames),
      episode_tasks=[instruction],
      episode_stats=stats,
    )

  manifest = {
    "schema": SCHEMA,
    "created_utc": datetime.now(UTC).isoformat(),
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "repo_id": _repo_id(plans[0]),
    "task": getattr(plans[0].source, "task", "pick-place"),
    "instruction": instruction,
    "cameras": list(_camera_names(plans[0])),
    "plan_fingerprint": plan_fingerprint,
    "available_source_episodes": available_source_episodes,
    "expected_episodes": expected_episodes,
    "selection_limit": selection_limit,
    "total_episodes": len(plans),
    "total_frames": sum(int(plan.exported_frames) for plan in plans),
    "source_hash_verification": (
      "recomputed" if verify_source_hash else "trusted_acquisition_sidecar"
    ),
    "episode_workers": workers,
    "ffmpeg_threads_per_worker": FFMPEG_THREADS_PER_WORKER,
    "video": {
      "codec": "h264",
      "pixel_format": "yuv420p",
      "fps": FPS,
      "preset": VIDEO_PRESET,
      "crf": VIDEO_CRF,
      "temporary_png_files": False,
    },
    "episodes": [
      {
        "output_episode_index": output_index,
        "source_episode_index": plan.source.episode_index,
        "frames": plan.exported_frames,
        "exported_frames_30hz": plan.exported_frames,
        "source_hdf5": str(plan.source.hdf5.path),
        "artifacts": result["artifacts"],
      }
      for output_index, (plan, result) in enumerate(zip(plans, results, strict=True))
    ],
  }
  _write_json_lines(
    staging / "meta" / "kaihand_source_episodes.jsonl",
    _source_manifest_rows(plans, verify_source_hash=verify_source_hash),
  )
  _write_json(staging / "meta" / "kaihand_fast_pi05_conversion.json", manifest)
  _validate_dataset(staging, plans)

  if _is_path(output_dir):
    raise FileExistsError(f"output appeared during conversion: {output_dir}")
  if staging.stat().st_dev == output_dir.parent.stat().st_dev:
    os.replace(staging, output_dir)
  else:
    upload = output_dir.with_name(
      f".{output_dir.name}.upload-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
      print(f"copy validated dataset to output filesystem: {upload}", flush=True)
      shutil.copytree(staging, upload, copy_function=shutil.copy2)
      _validate_dataset(upload, plans)
      if _is_path(output_dir):
        raise FileExistsError(f"output appeared during publication: {output_dir}")
      os.replace(upload, output_dir)
    finally:
      if upload.is_dir():
        shutil.rmtree(upload)
  return output_dir


def _fast_convert(
  input_dir: Path,
  output_dir: Path,
  plans: tuple[Any, ...],
  *,
  available_source_episodes: int,
  expected_episodes: int,
  selection_limit: int | None,
  verify_source_hash: bool,
  fingerprint_workers: int,
  image_writer_processes: int,
  image_writer_threads: int,
  staging_root: Path | None,
) -> None:
  del image_writer_processes, image_writer_threads
  if not plans:
    raise ValueError("conversion plan is empty")
  input_dir = input_dir.expanduser().resolve(strict=True)
  output_dir = output_dir.expanduser().resolve()
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  if _is_path(output_dir):
    raise FileExistsError(f"output path already exists: {output_dir}")

  lock_path = output_dir.with_name(f".{output_dir.name}.conversion.lock")
  try:
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
  except FileExistsError as error:
    raise RuntimeError(
      f"conversion lock exists: {lock_path}; verify no converter is running"
    ) from error
  lock_stat = os.fstat(lock_descriptor)
  lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
  staging_parent = (
    output_dir.parent if staging_root is None else staging_root.expanduser().resolve()
  )
  staging_parent.mkdir(parents=True, exist_ok=True)
  work_root = staging_parent / f".{output_dir.name}.fast-pi05-work"
  plan_payload = {
    "schema": SCHEMA,
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "features": _base_features(plans[0]),
    "instruction": _instruction(plans[0]),
    "repo_id": _repo_id(plans[0]),
    "video_preset": VIDEO_PRESET,
    "video_crf": VIDEO_CRF,
    "conversion_sources": _conversion_source_hashes(),
    "plans": [_plan_identity(plan) for plan in plans],
  }
  plan_fingerprint = _fingerprint(plan_payload)
  try:
    os.write(
      lock_descriptor,
      f"host={socket.gethostname()} pid={os.getpid()}\n".encode("utf-8"),
    )
    os.fsync(lock_descriptor)
    if work_root.exists():
      checkpoint = _load_object(work_root / "plan.json")
      if checkpoint.get("plan_fingerprint") != plan_fingerprint:
        raise RuntimeError(
          f"resume work does not match this conversion plan: {work_root}"
        )
    else:
      work_root.mkdir(parents=True)
      _write_json(
        work_root / "plan.json",
        {**plan_payload, "plan_fingerprint": plan_fingerprint},
      )
    results = _run_workers(
      plans,
      work_root=work_root,
      workers=fingerprint_workers,
      plan_fingerprint=plan_fingerprint,
    )
    published = _assemble(
      input_dir,
      output_dir,
      plans,
      results,
      work_root=work_root,
      plan_fingerprint=plan_fingerprint,
      available_source_episodes=available_source_episodes,
      expected_episodes=expected_episodes,
      selection_limit=selection_limit,
      verify_source_hash=verify_source_hash,
      workers=fingerprint_workers,
    )
    shutil.rmtree(work_root)
    print(
      f"published atomically: {published} "
      f"({len(plans)} episodes, "
      f"{sum(int(plan.exported_frames) for plan in plans)} frames)",
      flush=True,
    )
  finally:
    with contextlib.suppress(OSError):
      os.close(lock_descriptor)
    try:
      observed = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
      pass
    else:
      if (observed.st_dev, observed.st_ino) == lock_identity:
        lock_path.unlink()


def _assert_source_unchanged(source: Any) -> None:
  if _CONVERTER is None:
    raise RuntimeError("fast converter is not installed")
  authority = _CONVERTER.common if hasattr(_CONVERTER, "common") else _CONVERTER
  authority._assert_unchanged(source.hdf5)
  authority._assert_unchanged(source.sidecar)


def _parallel_preflight(sources: tuple[Any, ...], workers: int) -> tuple[Any, ...]:
  if _CONVERTER is None:
    raise RuntimeError("fast converter is not installed")
  plans: list[Any | None] = [None] * len(sources)
  with concurrent.futures.ThreadPoolExecutor(
    max_workers=workers,
    thread_name_prefix="fast-pi05-preflight",
  ) as executor:
    futures = {
      executor.submit(_CONVERTER._preflight_episode, source): index
      for index, source in enumerate(sources)
    }
    completed = 0
    for future in concurrent.futures.as_completed(futures):
      index = futures[future]
      plans[index] = future.result()
      completed += 1
      print(
        f"preflight [{completed}/{len(sources)}] {sources[index].hdf5.path.name}",
        flush=True,
      )
  resolved = tuple(plan for plan in plans if plan is not None)
  if len(resolved) != len(sources) or not resolved:
    raise ValueError("preflight produced no episode plans")
  expected_shape: Any | None = None
  for plan in resolved:
    shape = (
      tuple(plan.camera_shapes)
      if hasattr(plan, "camera_shapes")
      else (int(plan.image_height), int(plan.image_width))
    )
    if expected_shape is None:
      expected_shape = shape
    elif shape != expected_shape:
      raise ValueError(
        f"{plan.source.hdf5.path}: image sizes {shape} differ from {expected_shape}"
      )
  if sum(int(plan.exported_frames) for plan in resolved) <= 0:
    raise ValueError("preflight produced no training frames")
  return resolved


def install_fast_conversion(
  converter: Any, *, workers: int = 1, adapter_path: Path | None = None
) -> None:
  """Install the high-throughput output backend on an OpenPI converter."""

  if workers <= 0:
    raise ValueError("fast conversion workers must be positive")
  global _CONVERTER, _ADAPTER_PATH
  _CONVERTER = converter
  _ADAPTER_PATH = adapter_path
  if hasattr(converter, "_preflight_episode"):
    converter._preflight = lambda sources: _parallel_preflight(sources, workers)
  converter._convert = _fast_convert
