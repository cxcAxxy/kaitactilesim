#!/usr/bin/env python3
"""Convert the fixed PickPlace 0914_200 raw batch to EgoSteer quickly.

The source and final destination are intentionally fixed so an EgoSteer
release cannot accidentally be published under ``pi05``.  Episode conversion
is process-parallel, JPEGs are encoded directly into one tar shard per episode
(there are no temporary PNG files), and the raw HDF5 digest recorded at
capture time is trusted unless ``--verify-source-hash`` is requested.

Workers build and strictly validate the complete release on CPFS first.  A
cross-filesystem publication is copied to a hidden NAS directory and renamed
into place only when complete.  The expensive dataset validator runs once,
after all 200 episodes have been converted.
"""

# Parallelism is at episode granularity.  Keep native libraries from creating
# a second layer of threads in every worker process.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

for _thread_variable in (
  "BLIS_NUM_THREADS",
  "LP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
):
  os.environ.setdefault(_thread_variable, "1")

import h5py
import numpy as np
from convert_to_egosteer import (
  EPISODE_NAME_PATTERN,
  SOURCE_SCHEMA_VERSION,
  EpisodeConversion,
  Kinematics,
  ShardSummary,
  SourceEpisode,
  _convert_episode,
  _dataset_manifest,
  _json_attr,
  _make_kinematics,
  _sha256_file,
  _source_manifest,
  _split_sources,
  _strict_30hz_prefix,
  _validate_camera_datasets,
  _write_json,
)
from kaihand_tactile_env.shared.config import model_fingerprint
from validate_egosteer_dataset import DatasetValidator


DATASET_ROOT = Path("/nas/chenxianchi/datasets/sim/pickplace")
RAW_DIR = DATASET_ROOT / "raw/0914_200"
EGOSTEER_OUTPUT = DATASET_ROOT / "egosteer/0914_200"
DEFAULT_SCRATCH_ROOT = Path(
  "/cpfs_infra/user/chenxianchi/.conversion_staging/pickplace_0914_200_egosteer"
)
EXPECTED_EPISODES = 200
AUTO_WORKER_CAP = 8
DEFAULT_DATASET_NAME = "pickplace_0914_200"
DEFAULT_INSTRUCTION = "Put the red cylinder into the blue box."
SHA256 = re.compile(r"^[0-9a-f]{64}$")

SENSITIVE_PROCESS_NAMES = frozenset(
  {
    "record_dataset.py",
    "record_usb_dataset.py",
    "collect_poker_batch.py",
    "train_usb_tict_ddp.py",
    "train.py",
    "train_pytorch.py",
  }
)


@dataclass(frozen=True)
class ScannedSource:
  """Read-only source facts collected without loading camera payloads."""

  path: Path
  sidecar_path: Path
  episode_index: int
  hdf5_sha256: str
  sidecar_sha256: str
  size_bytes: int
  mtime_ns: int
  recorded_model_path: Path
  model_sha256: str
  model_fingerprint: str


@dataclass(frozen=True)
class EpisodeExport:
  """One completed episode shard and the facts required by the manifest."""

  conversion: EpisodeConversion
  shard_path: str
  shard_sha256: str
  shard_size_bytes: int


@dataclass(frozen=True)
class PreflightResult:
  """Cheap structural preflight result for one source episode."""

  episode_index: int
  exported_samples: int
  image_width: int
  image_height: int


_WORKER_KINEMATICS: dict[str, Kinematics] = {}
_T = TypeVar("_T")
_R = TypeVar("_R")


def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError("must be a positive integer")
  return parsed


def _fraction(value: str) -> float:
  parsed = float(value)
  if not 0.0 <= parsed < 1.0:
    raise argparse.ArgumentTypeError("must be in [0, 1)")
  return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--workers",
    type=_positive_int,
    help=(
      "Override adaptive parallelism. Automatic mode uses up to 8 workers and "
      "drops to one low-priority worker while collection/training is active."
    ),
  )
  parser.add_argument(
    "--scratch-root",
    type=Path,
    default=DEFAULT_SCRATCH_ROOT,
    help="Fast CPFS staging root used before atomic NAS publication.",
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help=(
      "Re-hash all 2.4 GiB of raw HDF5 data instead of trusting the capture "
      "sidecars. This is slower and is not needed for a normal conversion."
    ),
  )
  parser.add_argument(
    "--val-fraction",
    type=_fraction,
    default=0.0,
    help="Whole-episode validation fraction; default 0 keeps all 200 for training.",
  )
  parser.add_argument("--split-seed", type=int, default=0)
  parser.add_argument("--jpeg-quality", type=_positive_int, default=95)
  parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
  parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Run read-only source/model/header preflight without creating output.",
  )
  parser.add_argument(
    "--limit",
    type=_positive_int,
    help="Preflight only the first N episodes; valid only with --validate-only.",
  )
  parser.add_argument(
    "--low-priority",
    action="store_true",
    help="Force nice=10 and lowest best-effort ionice priority.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print paths and resource plan without reading HDF5 payloads or writing output.",
  )
  args = parser.parse_args(argv)
  if args.jpeg_quality > 100:
    parser.error("--jpeg-quality must be in [1, 100]")
  if not args.dataset_name.strip():
    parser.error("--dataset-name cannot be empty")
  if not args.instruction.strip():
    parser.error("--instruction cannot be empty")
  if args.limit is not None and not args.validate_only:
    parser.error("--limit is only allowed with --validate-only")
  return args


def validate_layout(raw: Path, output: Path) -> None:
  raw = raw.resolve()
  output = output.resolve()
  if raw.name != "0914_200" or raw.parent.name != "raw":
    raise ValueError(f"unexpected raw dataset path: {raw}")
  expected = raw.parent.parent / "egosteer/0914_200"
  if output != expected:
    raise ValueError(f"EgoSteer output must be {expected}, got {output}")
  if "pi05" in output.parts:
    raise ValueError(f"refusing to put EgoSteer data under pi05: {output}")
  if raw == output or raw in output.parents or output in raw.parents:
    raise ValueError(f"input/output paths overlap: {raw} and {output}")


def _affinity() -> tuple[int, ...]:
  if hasattr(os, "sched_getaffinity"):
    return tuple(sorted(os.sched_getaffinity(0)))
  return tuple(range(os.cpu_count() or 1))


def active_sensitive_processes(
  proc_root: Path = Path("/proc"),
) -> tuple[tuple[int, str], ...]:
  active = []
  for directory in proc_root.iterdir():
    if not directory.name.isdigit():
      continue
    try:
      state = (directory / "stat").read_text().split(")", 1)[1].split()[0]
      arguments = [
        item.decode(errors="replace")
        for item in (directory / "cmdline").read_bytes().split(b"\0")
        if item
      ]
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
      continue
    if state in {"X", "Z"}:
      continue
    matches = SENSITIVE_PROCESS_NAMES.intersection(
      Path(item).name for item in arguments
    )
    if matches:
      active.append((int(directory.name), sorted(matches)[0]))
  return tuple(sorted(active))


def choose_workers(
  requested: int | None,
  *,
  affinity_cpus: int,
  load_1m: float,
  active_sensitive: int,
) -> int:
  if affinity_cpus < 1:
    raise ValueError("CPU affinity must expose at least one CPU")
  if requested is not None:
    if requested > affinity_cpus:
      raise ValueError(
        f"requested workers ({requested}) exceed CPU affinity ({affinity_cpus})"
      )
    return requested
  if active_sensitive:
    return 1
  reserve = min(max(0, affinity_cpus - 1), max(2, math.ceil(affinity_cpus * 0.20)))
  load_spare = math.floor(affinity_cpus - max(0.0, load_1m) - reserve)
  return max(1, min(AUTO_WORKER_CAP, load_spare, affinity_cpus - reserve))


def _lower_current_priority() -> None:
  with contextlib.suppress(OSError):
    os.nice(10)
  ionice = shutil.which("ionice")
  if ionice is not None:
    subprocess.run(
      (ionice, "-c", "2", "-n", "7", "-p", str(os.getpid())),
      check=False,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"{label} is not readable JSON: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{label} must contain a JSON object: {path}")
  return value


def _scan_source(task: tuple[Path, bool]) -> ScannedSource:
  path, verify_source_hash = task
  path = path.resolve()
  match = EPISODE_NAME_PATTERN.fullmatch(path.name)
  if match is None:
    raise ValueError(f"unsupported episode filename: {path.name}")
  filename_index = int(match.group(1))
  sidecar_path = path.with_suffix(".json")
  sidecar = _read_json(sidecar_path, "capture sidecar")
  if sidecar.get("episode") != path.name:
    raise ValueError(f"source sidecar episode mismatch: {sidecar_path}")
  recorded_digest = sidecar.get("sha256")
  if not isinstance(recorded_digest, str) or SHA256.fullmatch(recorded_digest) is None:
    raise ValueError(f"source sidecar SHA-256 is invalid: {sidecar_path}")

  before = path.stat()
  if verify_source_hash and _sha256_file(path) != recorded_digest:
    raise ValueError(f"source sidecar SHA-256 mismatch: {sidecar_path}")
  with h5py.File(path, "r") as file:
    if str(file.attrs.get("schema_version", "")) != SOURCE_SCHEMA_VERSION:
      raise ValueError(f"{path}: unsupported source schema_version")
    metadata = _json_attr(file, "metadata_json")
    outcome = _json_attr(file, "outcome_json")
    if outcome.get("success") is not True or outcome.get("placed_in_box") is not True:
      raise ValueError(f"{path}: episode is not a successful pick-and-place")
    episode_index = metadata.get("episode_index")
    if not isinstance(episode_index, int) or isinstance(episode_index, bool):
      raise ValueError(f"{path}: metadata episode_index must be an integer")
    if episode_index != filename_index:
      raise ValueError(f"{path}: filename and metadata episode indices differ")
    model_sha256 = str(file.attrs.get("model_sha256", ""))
    if SHA256.fullmatch(model_sha256) is None:
      raise ValueError(f"{path}: missing or invalid model_sha256")
    recorded_model_path = Path(str(file.attrs.get("model_path", "")))
    recorded_model_fingerprint = str(file.attrs.get("model_fingerprint", ""))
    if SHA256.fullmatch(recorded_model_fingerprint) is None:
      raise ValueError(f"{path}: missing or invalid model_fingerprint")
  after = path.stat()
  if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
    raise RuntimeError(f"source changed during preflight: {path}")
  return ScannedSource(
    path=path,
    sidecar_path=sidecar_path.resolve(),
    episode_index=episode_index,
    hdf5_sha256=recorded_digest,
    sidecar_sha256=_sha256_file(sidecar_path),
    size_bytes=after.st_size,
    mtime_ns=after.st_mtime_ns,
    recorded_model_path=recorded_model_path,
    model_sha256=model_sha256,
    model_fingerprint=recorded_model_fingerprint,
  )


def _run_pool(
  operation: Callable[[_T], _R],
  items: Sequence[_T],
  workers: int,
  label: str,
) -> list[_R]:
  if not items:
    return []
  maximum = min(workers, len(items))
  context = multiprocessing.get_context("spawn")
  started = time.monotonic()
  executor = ProcessPoolExecutor(max_workers=maximum, mp_context=context)
  futures = {executor.submit(operation, item): item for item in items}
  results: list[_R] = []
  try:
    for completed, future in enumerate(as_completed(futures), start=1):
      results.append(future.result())
      elapsed = max(1.0e-9, time.monotonic() - started)
      eta = elapsed / completed * (len(items) - completed)
      print(
        f"{label} [{completed}/{len(items)}] elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
      )
  except BaseException:
    for future in futures:
      future.cancel()
    executor.shutdown(wait=True, cancel_futures=True)
    raise
  executor.shutdown(wait=True)
  return results


def _discover_sources(
  input_dir: Path,
  workers: int,
  verify_source_hash: bool,
) -> tuple[SourceEpisode, ...]:
  paths = sorted(input_dir.glob("episode_*.h5"))
  if len(paths) != EXPECTED_EPISODES:
    raise ValueError(
      f"expected exactly {EXPECTED_EPISODES} source HDF5 files, found {len(paths)}"
    )
  scans = _run_pool(
    _scan_source,
    tuple((path, verify_source_hash) for path in paths),
    workers,
    "source preflight",
  )
  scans.sort(key=lambda item: item.episode_index)
  indices = [item.episode_index for item in scans]
  if indices != list(range(EXPECTED_EPISODES)):
    raise ValueError("source episode indices must be exactly 0..199")

  model_cache: dict[Path, tuple[str, str]] = {}
  sources = []
  for item in scans:
    model_path = item.recorded_model_path.expanduser().resolve()
    if not model_path.is_file():
      raise FileNotFoundError(f"recorded model is unavailable: {model_path}")
    if model_path not in model_cache:
      model_cache[model_path] = (
        _sha256_file(model_path),
        model_fingerprint(model_path),
      )
    model_sha256, fingerprint = model_cache[model_path]
    if model_sha256 != item.model_sha256:
      raise ValueError(f"episode {item.episode_index}: recorded model SHA mismatch")
    if fingerprint != item.model_fingerprint:
      raise ValueError(f"episode {item.episode_index}: recorded model fingerprint mismatch")
    sources.append(
      SourceEpisode(
        path=item.path,
        sidecar_path=item.sidecar_path,
        episode_index=item.episode_index,
        hdf5_sha256=item.hdf5_sha256,
        sidecar_sha256=item.sidecar_sha256,
        size_bytes=item.size_bytes,
        mtime_ns=item.mtime_ns,
        model_path=model_path,
        model_sha256=item.model_sha256,
      )
    )
  return tuple(sources)


def _preflight_source(source: SourceEpisode) -> PreflightResult:
  with h5py.File(source.path, "r") as file:
    camera = _validate_camera_datasets(file, source)
    timestamps = np.asarray(camera["timestamp"], dtype=np.float64)
    prefix, _ = _strict_30hz_prefix(timestamps, int(file.attrs.get("physics_hz", -1)))
    rgb = camera["rgb"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
      raise ValueError(f"{source.path}: head RGB must be uint8 NxHxWx3")
    if file["state/qpos"].shape[1] != _worker_kinematics(source).model.nq:
      raise ValueError(f"{source.path}: qpos dimension does not match recorded model")
    return PreflightResult(
      episode_index=source.episode_index,
      exported_samples=prefix - 1,
      image_width=int(rgb.shape[2]),
      image_height=int(rgb.shape[1]),
    )


def _worker_kinematics(source: SourceEpisode) -> Kinematics:
  if source.model_sha256 not in _WORKER_KINEMATICS:
    _WORKER_KINEMATICS[source.model_sha256] = _make_kinematics(source)
  return _WORKER_KINEMATICS[source.model_sha256]


def _export_episode(
  task: tuple[SourceEpisode, str, Path, str, str, int],
) -> EpisodeExport:
  source, split, staging_dir, instruction, dataset_name, jpeg_quality = task
  shard_path = staging_dir / split / f"shard-{source.episode_index:06d}.tar"
  with tarfile.open(shard_path, mode="x", format=tarfile.USTAR_FORMAT) as archive:
    conversion = _convert_episode(
      source,
      split,
      archive,
      _WORKER_KINEMATICS,
      instruction=instruction,
      dataset_name=dataset_name,
      jpeg_quality=jpeg_quality,
    )
  return EpisodeExport(
    conversion=conversion,
    shard_path=shard_path.relative_to(staging_dir).as_posix(),
    shard_sha256=_sha256_file(shard_path),
    shard_size_bytes=shard_path.stat().st_size,
  )


def _write_release_metadata(
  staging_dir: Path,
  sources: tuple[SourceEpisode, ...],
  exports: Sequence[EpisodeExport],
  args: argparse.Namespace,
  created_utc: str,
  elapsed_seconds: float,
) -> dict[str, Any]:
  ordered = sorted(exports, key=lambda item: item.conversion.episode_index)
  source_manifest = _source_manifest(RAW_DIR, sources, created_utc)
  source_manifest.update(
    {
      "hash_policy": (
        "full_hdf5_sha256_verified"
        if args.verify_source_hash
        else "capture_sidecar_sha256_trusted"
      ),
      "source_selection": {
        "complete_source_episodes": len(sources),
        "method": "all successful paired PickPlace HDF5/sidecar files, IDs 0..199",
      },
    }
  )
  source_manifest_path = staging_dir / "source_snapshot_manifest.json"
  _write_json(source_manifest_path, source_manifest)

  shard_groups: dict[str, list[ShardSummary]] = {"train": [], "val": []}
  conversions = []
  for item in ordered:
    conversion = item.conversion
    shard_groups[conversion.split].append(
      ShardSummary(
        path=item.shard_path,
        sha256=item.shard_sha256,
        size_bytes=item.shard_size_bytes,
        episodes=1,
        episode_indices=(conversion.episode_index,),
        samples=conversion.exported_samples,
      )
    )
    conversions.append(conversion)

  manifest = _dataset_manifest(
    created_utc=created_utc,
    dataset_name=args.dataset_name.strip(),
    instruction=args.instruction.strip(),
    split_seed=args.split_seed,
    val_fraction=args.val_fraction,
    jpeg_quality=args.jpeg_quality,
    source_manifest_sha256=_sha256_file(source_manifest_path),
    sources=sources,
    shards=shard_groups,
    conversions=tuple(conversions),
  )
  manifest["one_episode_per_shard"] = True
  manifest["source_hash_policy"] = source_manifest["hash_policy"]
  if args.val_fraction == 0.0:
    manifest["split_config"] = {
      "method": "all episodes assigned to train",
      "split_seed": args.split_seed,
      "train_episodes": len(sources),
      "val_episodes": 0,
    }
  _write_json(staging_dir / "dataset_manifest.json", manifest)

  print("final validation [1/1]: checking all shards and samples", flush=True)
  validation = DatasetValidator(staging_dir).validate()
  validation["dataset_root"] = str(EGOSTEER_OUTPUT)
  _write_json(staging_dir / "validation.json", validation)
  if not validation["valid"]:
    raise ValueError(
      f"EgoSteer validation failed with {validation['error_count']} errors"
    )

  train = manifest["splits"]["train"]
  val = manifest["splits"]["val"]
  report = {
    "schema_version": "pickplace_0914_200_to_egosteer_conversion_v1",
    "valid": True,
    "input": str(RAW_DIR),
    "output": str(EGOSTEER_OUTPUT),
    "converted_episodes": len(sources),
    "train_episodes": train["episodes"],
    "val_episodes": val["episodes"],
    "samples": train["samples"] + val["samples"],
    "workers": min(args.resolved_workers, len(sources)),
    "source_hash_policy": source_manifest["hash_policy"],
    "elapsed_seconds_before_publication": elapsed_seconds,
    "temporary_png_files": False,
    "full_dataset_validation_passes": 1,
  }
  _write_json(staging_dir / "conversion_report.json", report)
  (staging_dir / "README.md").write_text(
    "# PickPlace 0914_200 — EgoSteer\n\n"
    f"{len(sources)} successful PickPlace episodes converted from `{RAW_DIR}`. "
    "Each episode is one head-only WebDataset tar shard. RGB is JPEG encoded "
    "directly in memory; each sample also contains a float32 116D lowdim array "
    "and metadata JSON. Tactile data is intentionally excluded. All episodes "
    "are in `train/` by default, as requested.\n",
    encoding="utf-8",
  )
  return report


def _tree_signature(root: Path) -> tuple[tuple[str, int], ...]:
  return tuple(
    sorted(
      (path.relative_to(root).as_posix(), path.stat().st_size)
      for path in root.rglob("*")
      if path.is_file()
    )
  )


def _publish(staging_dir: Path, output_dir: Path) -> None:
  output_dir.parent.mkdir(parents=True, exist_ok=True)
  if os.path.lexists(output_dir):
    raise FileExistsError(f"output appeared during conversion: {output_dir}")
  if staging_dir.stat().st_dev == output_dir.parent.stat().st_dev:
    os.rename(staging_dir, output_dir)
    return

  upload_dir = output_dir.with_name(
    f".{output_dir.name}.upload-{os.getpid()}-{uuid.uuid4().hex}"
  )
  try:
    print(f"publish: copying validated CPFS release to {upload_dir}", flush=True)
    shutil.copytree(staging_dir, upload_dir, copy_function=shutil.copy2)
    if _tree_signature(upload_dir) != _tree_signature(staging_dir):
      raise RuntimeError("published tree file/size verification failed")
    if os.path.lexists(output_dir):
      raise FileExistsError(f"output appeared during publication: {output_dir}")
    os.rename(upload_dir, output_dir)
  except BaseException:
    if upload_dir.is_dir():
      shutil.rmtree(upload_dir)
    raise
  shutil.rmtree(staging_dir)


def _release_lock(lock_path: Path, descriptor: int, identity: tuple[int, int]) -> None:
  with contextlib.suppress(OSError):
    os.close(descriptor)
  try:
    observed = lock_path.stat(follow_symlinks=False)
  except FileNotFoundError:
    return
  if (observed.st_dev, observed.st_ino) == identity:
    lock_path.unlink()


def _validate_only(
  sources: tuple[SourceEpisode, ...],
  args: argparse.Namespace,
) -> None:
  selected = sources[: args.limit] if args.limit is not None else sources
  results = _run_pool(
    _preflight_source,
    selected,
    args.resolved_workers,
    "episode header preflight",
  )
  image_sizes = {(item.image_width, item.image_height) for item in results}
  if len(image_sizes) != 1:
    raise ValueError(f"source image sizes differ: {sorted(image_sizes)}")
  print(
    json.dumps(
      {
        "valid": True,
        "mode": "validate-only",
        "complete_batch_episodes": len(sources),
        "preflighted_episodes": len(results),
        "samples": sum(item.exported_samples for item in results),
        "image_size": list(image_sizes.pop()),
        "source_hash_policy": (
          "full_hdf5_sha256_verified"
          if args.verify_source_hash
          else "capture_sidecar_sha256_trusted"
        ),
        "output_created": False,
      },
      ensure_ascii=False,
      sort_keys=True,
    ),
    flush=True,
  )


def _run_conversion(
  sources: tuple[SourceEpisode, ...],
  args: argparse.Namespace,
) -> None:
  splits = _split_sources(sources, args.val_fraction, args.split_seed)
  split_by_index = {
    source.episode_index: split
    for split, split_sources in splits.items()
    for source in split_sources
  }

  EGOSTEER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  lock_path = EGOSTEER_OUTPUT.with_name(
    f".{EGOSTEER_OUTPUT.name}.egosteer-conversion.lock"
  )
  try:
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
  except FileExistsError as error:
    raise RuntimeError(f"another conversion owns {lock_path}") from error
  lock_stat = os.fstat(lock_fd)
  lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
  staging_dir: Path | None = None
  started = time.monotonic()
  try:
    os.write(lock_fd, f"pid={os.getpid()} run={uuid.uuid4().hex}\n".encode())
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
      tempfile.mkdtemp(
        prefix=f".{EGOSTEER_OUTPUT.name}.partial-",
        dir=args.scratch_root,
      )
    )
    (staging_dir / "train").mkdir()
    (staging_dir / "val").mkdir()
    print(
      f"conversion staging={staging_dir} episodes={len(sources)} "
      f"workers={args.resolved_workers}",
      flush=True,
    )
    tasks = tuple(
      (
        source,
        split_by_index[source.episode_index],
        staging_dir,
        args.instruction.strip(),
        args.dataset_name.strip(),
        args.jpeg_quality,
      )
      for source in sources
    )
    exports = _run_pool(
      _export_episode,
      tasks,
      args.resolved_workers,
      "convert",
    )
    report = _write_release_metadata(
      staging_dir,
      sources,
      exports,
      args,
      datetime.now(timezone.utc).isoformat(),
      time.monotonic() - started,
    )
    _publish(staging_dir, EGOSTEER_OUTPUT)
    staging_dir = None
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"published atomically: {EGOSTEER_OUTPUT}", flush=True)
  except BaseException:
    if staging_dir is not None and staging_dir.exists():
      shutil.rmtree(staging_dir)
    raise
  finally:
    _release_lock(lock_path, lock_fd, lock_identity)


def preflight_paths(*, no_write: bool) -> None:
  validate_layout(RAW_DIR, EGOSTEER_OUTPUT)
  if not RAW_DIR.is_dir():
    raise FileNotFoundError(f"raw dataset does not exist: {RAW_DIR}")
  if os.path.lexists(EGOSTEER_OUTPUT):
    raise FileExistsError(
      f"refusing to overwrite existing EgoSteer output: {EGOSTEER_OUTPUT}"
    )
  if no_write:
    return
  EGOSTEER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  args.scratch_root = args.scratch_root.expanduser().resolve(strict=False)
  preflight_paths(no_write=args.dry_run or args.validate_only)

  affinity = _affinity()
  active = active_sensitive_processes()
  load_1m = os.getloadavg()[0]
  args.resolved_workers = choose_workers(
    args.workers,
    affinity_cpus=len(affinity),
    load_1m=load_1m,
    active_sensitive=len(active),
  )
  low_priority = args.low_priority or bool(active)
  active_text = ",".join(f"{name}:{pid}" for pid, name in active) or "none"
  print(
    "resource plan: "
    f"affinity_cpus={len(affinity)} load_1m={load_1m:.2f} "
    f"active_sensitive={active_text} priority={'low' if low_priority else 'normal'} "
    f"workers={args.resolved_workers}",
    flush=True,
  )
  print(f"input:  {RAW_DIR}", flush=True)
  print(f"output: {EGOSTEER_OUTPUT}", flush=True)
  print(f"scratch: {args.scratch_root}", flush=True)
  print(
    f"split: train={100.0 * (1.0 - args.val_fraction):.1f}% "
    f"val={100.0 * args.val_fraction:.1f}%",
    flush=True,
  )
  if args.dry_run:
    return 0
  if low_priority:
    _lower_current_priority()

  sources = _discover_sources(
    RAW_DIR,
    args.resolved_workers,
    args.verify_source_hash,
  )
  if args.validate_only:
    _validate_only(sources, args)
  else:
    _run_conversion(sources, args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
