#!/usr/bin/env python3
"""Bounded, audited poker capture orchestration; never changes the controller.

Run inside Pixi: pixi run python scripts/workcell/collect_poker_batch.py ...
Each child invokes the existing recorder with --workers 1. This supervisor's
lock excludes other instances of this wrapper, NOT independently launched
simulations. Resource checks are soft safeguards, not a hard memory guarantee.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES, TRAINING_CAMERA_NAMES
from kaihand_tactile_env.tasks.poker_draw.acceptance import (
  ACCEPTANCE_POLICIES,
  STRICT_FORCE_POLICY,
  validate_recorded_acceptance,
)

PROJECT = Path(__file__).resolve().parents[2]
GLOBAL_LOCK = PROJECT / "datasets/.poker_batch.lock"
PRESET = "middle-force-precontact-v1"
MIB = 1024**2
GIB = 1024**3


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument("--target-successes", type=int,
                      help="Stop at this total of validated successes in output-dir; episodes is the new-attempt budget. Automatically continue after occupied indices.")
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument("--cameras", nargs="+", choices=SHARED_CAMERA_NAMES,
                      default=TRAINING_CAMERA_NAMES,
                      help="Recorded shared cameras; defaults to head/left_wrist/right_wrist")
  parser.add_argument("--acceptance-policy", choices=ACCEPTANCE_POLICIES,
                      default=STRICT_FORCE_POLICY)
  parser.add_argument("--render-backend", choices=("hardware", "software", "auto"),
                      default="hardware", help="Verify actual GL backend; hardware never silently falls back")
  parser.add_argument("--hdf5-buffer-rows", type=int, choices=(0, 32, 64, 128), default=64,
                      help="Bounded lossless non-camera append batch; 0 disables")
  parser.add_argument("--min-available-memory-mib", type=float, default=3500)
  parser.add_argument("--critical-available-memory-mib", type=float, default=2000)
  parser.add_argument("--min-free-disk-gib", type=float, default=15)
  parser.add_argument("--episode-timeout", type=float, default=1200)
  parser.add_argument("--critical-duration-seconds", type=float, default=5)
  parser.add_argument("--poll-seconds", type=float, default=1)
  parser.add_argument("--term-grace-seconds", type=float, default=15)
  parser.add_argument(
    "--nice-increment",
    type=int,
    choices=range(20),
    default=0,
    help=(
      "Lower child CPU priority by 0..19. The speed-oriented default is 0; "
      "use 10 when collection must yield to another important workload."
    ),
  )
  parser.add_argument(
    "--verify-output-hash",
    action="store_true",
    help=(
      "Re-read each completed HDF5 and verify its SHA-256. By default the "
      "supervisor trusts the SHA-256 sidecar written atomically by the recorder "
      "and avoids a redundant full-file read."
    ),
  )
  args = parser.parse_args(argv)
  if args.episodes <= 0 or args.start_index < 0 or args.seed < 0:
    parser.error("episodes must be positive; start-index and seed nonnegative")
  if args.target_successes is not None and args.target_successes <= 0:
    parser.error("target-successes must be positive")
  if len(set(args.cameras)) != len(args.cameras) or "head" not in args.cameras:
    parser.error("--cameras must contain head and must not contain duplicates")
  if args.camera_hz <= 0 or args.camera_hz > 100:
    parser.error("camera-hz must be in 1..100")
  if len(set(args.cameras)) != len(args.cameras) or "head" not in args.cameras:
    parser.error("--cameras must contain head and must not contain duplicates")
  for name in (
    "min_available_memory_mib", "critical_available_memory_mib",
    "min_free_disk_gib", "episode_timeout", "critical_duration_seconds",
    "poll_seconds", "term_grace_seconds",
  ):
    value = getattr(args, name)
    if not math.isfinite(value) or value <= 0:
      parser.error(f"--{name.replace('_', '-')} must be finite and positive")
  if args.critical_available_memory_mib >= args.min_available_memory_mib:
    parser.error("critical memory threshold must be lower than launch threshold")
  if args.poll_seconds > 10:
    parser.error("poll-seconds must not exceed 10")
  args.output_dir = args.output_dir.resolve()
  return args


def utc_now():
  return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload):
  temporary = path.with_name(path.name + ".tmp")
  with temporary.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
  os.replace(temporary, path)


@contextmanager
def wrapper_lock(path: Path = GLOBAL_LOCK):
  """Persistent lock inode avoids unlink/recreation races between supervisors."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a+", encoding="utf-8") as stream:
    try:
      fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError(f"another poker batch supervisor owns {path}") from error
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps({"pid": os.getpid(), "started_utc": utc_now()}))
    stream.flush()
    try:
      yield
    finally:
      fcntl.flock(stream, fcntl.LOCK_UN)


def child_environment(render_backend="hardware", hdf5_buffer_rows=64):
  if render_backend not in ("hardware", "software", "auto"):
    raise ValueError("invalid render backend")
  if hdf5_buffer_rows not in (0, 32, 64, 128):
    raise ValueError("invalid HDF5 buffer row count")
  result = dict(os.environ)
  result.update({
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "LP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1",
    "KAIHAND_RENDER_BACKEND": render_backend,
    "KAIHAND_POKER_HDF5_BUFFER_ROWS": str(hdf5_buffer_rows),
  })
  if render_backend == "software":
    result.update({"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe"})
  elif render_backend == "hardware":
    result.pop("LIBGL_ALWAYS_SOFTWARE", None)
    result.pop("GALLIUM_DRIVER", None)
  return result


def episode_command(args, index):
  return [
    sys.executable, str(PROJECT / "scripts/workcell/record_dataset.py"),
    "--scene", "poker-draw", "--preset", PRESET,
    "--episodes", "1", "--seed", str(args.seed),
    "--start-index", str(index), "--workers", "1",
    "--cameras", *args.cameras, "--width", "320", "--height", "240",
    "--rgb-only", "--camera-hz", str(args.camera_hz),
    "--output-dir", str(args.output_dir / "raw"), "--fail-fast",
    "--acceptance-policy", args.acceptance_policy,
  ]


def episode_name(index):
  return f"episode_{index:06d}_card_right"


def check_output_conflicts(args):
  conflicts = []
  for index in range(args.start_index, args.start_index + args.episodes):
    stem = episode_name(index)
    for name in (stem + ".h5", stem + ".json", stem + ".h5.partial",
                 stem + ".failure.json", stem + ".h5.lock"):
      path = args.output_dir / "raw" / name
      if path.exists():
        conflicts.append(str(path))
    logs = args.output_dir / "logs" / stem
    if logs.exists():
      conflicts.append(str(logs))
  if conflicts:
    raise FileExistsError("existing attempts are never replaced: " + ", ".join(conflicts))


def proc_table(proc_root: Path = Path("/proc")):
  """One lightweight /proc snapshot; RSS sums can double-count shared pages."""
  result = {}
  for entry in proc_root.iterdir():
    if not entry.name.isdecimal():
      continue
    try:
      raw = (entry / "stat").read_text()
      fields = raw[raw.rfind(")") + 2:].split()
      result[int(entry.name)] = {
        "ppid": int(fields[1]), "pgrp": int(fields[2]),
        "major_faults": int(fields[9]),
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "rss_mib": int(fields[21]) * os.sysconf("SC_PAGE_SIZE") / MIB,
      }
    except (OSError, ValueError, IndexError):
      continue
  return result


def process_tree_metrics(table, root_pid):
  selected = {root_pid}
  while True:
    children = {pid for pid, item in table.items() if item["ppid"] in selected}
    expanded = selected | children
    if expanded == selected:
      break
    selected = expanded
  entries = [table[pid] for pid in selected if pid in table]
  return {
    "pids": sorted(pid for pid in selected if pid in table),
    "rss_mib": sum(item["rss_mib"] for item in entries),
    "major_faults": sum(item["major_faults"] for item in entries),
    "cpu_seconds": sum(item["cpu_seconds"] for item in entries),
  }


def resource_snapshot(output: Path, proc_root: Path = Path("/proc")):
  memory = {}
  for line in (proc_root / "meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    memory[key] = int(value.split()[0])
  vm = {}
  for line in (proc_root / "vmstat").read_text().splitlines():
    key, value = line.split()
    if key in ("pswpin", "pswpout", "pgmajfault"):
      vm[key] = int(value)
  return {
    "utc": utc_now(), "monotonic_s": time.monotonic(),
    "mem_available_mib": memory["MemAvailable"] / 1024,
    "swap_used_mib": (memory["SwapTotal"] - memory["SwapFree"]) / 1024,
    "disk_free_gib": shutil.disk_usage(output).free / GIB,
    "global_vmstat": vm,
  }


def launch_allowed(snapshot, args):
  return (snapshot["mem_available_mib"] >= args.min_available_memory_mib
          and snapshot["disk_free_gib"] >= args.min_free_disk_gib)


def critical_reason(snapshot, previous, args):
  if snapshot["mem_available_mib"] < args.critical_available_memory_mib:
    return "critical_available_memory"
  if snapshot["disk_free_gib"] < args.min_free_disk_gib:
    return "critical_free_disk"
  if previous is not None:
    swap_delta = (snapshot["global_vmstat"].get("pswpout", 0)
                  - previous["global_vmstat"].get("pswpout", 0))
    seconds = max(snapshot["monotonic_s"] - previous["monotonic_s"], 1e-9)
    if swap_delta / seconds > 256 and snapshot["mem_available_mib"] < args.min_available_memory_mib:
      return "global_swap_pressure_with_low_available_memory"
  return None


def validate_success(path, args, index):
  """Read-only schema, sidecar identity and requested-camera/seed verification."""
  import h5py
  import numpy as np
  from kaihand_tactile_env.shared.recording import validate_episode

  report = validate_episode(path)
  if not report.valid:
    raise ValueError(f"recording schema validation failed: {report.errors}")
  manifest = json.loads(path.with_suffix(".json").read_text())
  sha = manifest.get("sha256")
  if (
    manifest.get("episode") != path.name
    or not isinstance(sha, str)
    or re.fullmatch(r"[0-9a-f]{64}", sha) is None
  ):
    raise ValueError("episode sidecar identity/SHA-256 is invalid")
  hash_verification = "trusted_atomic_recorder_sidecar"
  if args.verify_output_hash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
      for block in iter(lambda: stream.read(MIB), b""):
        digest.update(block)
    if digest.hexdigest() != sha:
      raise ValueError("episode sidecar SHA-256 does not match HDF5")
    hash_verification = "recomputed"
  with h5py.File(path) as file:
    metadata = json.loads(file.attrs["metadata_json"])
    outcome = json.loads(file.attrs["outcome_json"])
    if (metadata.get("preset") != PRESET
        or metadata.get("seed") != args.seed + index
        or metadata.get("episode_index") != index):
      raise ValueError("recorded preset/seed/index mismatch")
    if not outcome.get("success"):
      raise ValueError("recorded task did not succeed")
    if metadata.get("acceptance_policy", STRICT_FORCE_POLICY) != args.acceptance_policy:
      raise ValueError("recorded acceptance policy differs from requested collection")
    validate_recorded_acceptance(metadata, outcome)
    if (int(file.attrs["physics_hz"]) != 500
        or int(file.attrs["control_hz"]) != 100
        or int(file.attrs["camera_hz"]) != args.camera_hz):
      raise ValueError("recorded clocks differ from requested collection")
    if set(file["cameras"]) != set(args.cameras):
      raise ValueError("recorded cameras differ from requested collection")
    for name in args.cameras:
      if file[f"cameras/{name}/rgb"].shape[1:] != (240, 320, 3):
        raise ValueError(f"{name} image dimensions are not 320x240 RGB")
      if (len(file[f"cameras/{name}/rgb"]) != len(file["cameras/head/rgb"])
          or not np.array_equal(file[f"cameras/{name}/timestamp"][:],
                                file["cameras/head/timestamp"][:])):
        raise ValueError(f"{name} camera frames are not synchronized with head")
    frame_count = len(file["cameras/head/rgb"])
    duration = float(file["state/timestamp"][-1])
    backend = json.loads(file.attrs.get("render_backend_json", "{}"))
    if args.render_backend == "hardware" and (not backend.get("renderer") or backend.get("software") is not False):
      raise ValueError("hardware rendering was not verified in the actual capture")
  return {"valid": True, "sha256": sha,
          "sha256_verification": hash_verification, "camera_frames": frame_count,
          "duration_seconds": duration, "state_samples": report.state_samples,
          "render_backend": backend}


def resume_inventory(args):
  """Called under the wrapper lock. Preserve every old attempt, count verified data.

  Historical software captures can count toward a hardware continuation. All
  other collection requirements are checked again. The acquisition sidecar
  digest is trusted unless --verify-output-hash requests a full file re-read.
  An invalid previously successful episode stops resume instead of silently
  accepting it or changing the old success label.
  """
  occupied = []
  for folder in (args.output_dir / "raw", args.output_dir / "logs"):
    if folder.exists():
      for path in folder.iterdir():
        match = re.match(r"episode_(\d+)_card_right(?:\.|$)", path.name)
        if match:
          occupied.append(int(match.group(1)))
  successes = []
  historical_args = argparse.Namespace(**{**vars(args), "render_backend": "auto"})
  for path in sorted((args.output_dir / "logs").glob("episode_*_card_right/execution.json")):
    report = json.loads(path.read_text())
    if report.get("completed") is not True:
      continue
    index = int(report["episode_index"])
    if path.parent.name != episode_name(index):
      raise ValueError(f"episode index/log path mismatch: {path}")
    validate_success(args.output_dir / "raw" / (episode_name(index) + ".h5"),
                     historical_args, index)
    successes.append(index)
  return successes, max(args.start_index, max(occupied, default=-1) + 1)


def remaining_success_slots(args, existing_count, results, active_count=0):
  if args.target_successes is None:
    return args.workers
  return max(0, args.target_successes - existing_count
             - sum(r["completed"] for r in results) - active_count)


@dataclass
class ActiveEpisode:
  index: int
  command: list[str]
  process: Any
  logs: Path
  stdout: Any
  stderr: Any
  resources: Any
  started: float
  started_utc: str
  termination_reason: str | None = None
  term_sent: float | None = None
  kill_sent: bool = False
  peaks: dict[str, float] = field(default_factory=lambda: {
    "rss_mib": 0., "major_faults": 0., "cpu_seconds": 0.})


def start_episode(args, index):
  logs = args.output_dir / "logs" / episode_name(index)
  logs.mkdir(parents=True, exist_ok=False)
  command = episode_command(args, index)
  environment = child_environment(args.render_backend, args.hdf5_buffer_rows)
  save_json(logs / "command.json", {
    "argv": command, "working_directory": str(PROJECT),
    "supervisor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "episode_index": index, "episode_seed": args.seed + index,
    "environment_overrides": {key: environment.get(key) for key in (
      "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
      "LP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MUJOCO_GL", "PYOPENGL_PLATFORM",
      "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "KAIHAND_RENDER_BACKEND",
      "KAIHAND_POKER_HDF5_BUFFER_ROWS")},
    "nice_increment": args.nice_increment,
  })
  stdout = (logs / "stdout.log").open("xb")
  stderr = (logs / "stderr.log").open("xb")
  resources = (logs / "resources.jsonl").open("x", encoding="utf-8")
  try:
    launch_command = (
      ["nice", "-n", str(args.nice_increment), *command]
      if args.nice_increment
      else command
    )
    process = subprocess.Popen(
      launch_command, cwd=PROJECT, env=environment,
      stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
      start_new_session=True,
    )
  except BaseException:
    stdout.close()
    stderr.close()
    resources.close()
    raise
  print(f"started index={index} seed={args.seed + index} pid={process.pid}", flush=True)
  return ActiveEpisode(index, command, process, logs, stdout, stderr, resources,
                       time.monotonic(), utc_now())


def signal_group(active, signum):
  try:
    os.killpg(active.process.pid, signum)
  except ProcessLookupError:
    pass


def terminate_episode(active, reason, now):
  if active.term_sent is None:
    active.termination_reason = reason
    active.term_sent = now
    signal_group(active, signal.SIGTERM)


def finish_episode(active, args):
  active.stdout.close()
  active.stderr.close()
  active.resources.close()
  report = {
    "episode_index": active.index, "episode_seed": args.seed + active.index,
    "argv": active.command, "pid": active.process.pid,
    "started_utc": active.started_utc, "ended_utc": utc_now(),
    "wall_seconds": time.monotonic() - active.started,
    "exit_code": active.process.returncode,
    "termination_reason": active.termination_reason,
    "sigkill_sent": active.kill_sent, "sampled_peak_process_tree": active.peaks,
    "resource_scope": "sampled own child process tree RSS sum; shared pages may be double counted; vmstat is host-global",
    "completed": False,
  }
  if active.process.returncode == 0 and active.termination_reason is None:
    try:
      report["validation"] = validate_success(
        args.output_dir / "raw" / (episode_name(active.index) + ".h5"), args, active.index)
      report["completed"] = True
    except Exception as error:
      report["validation_error"] = f"{type(error).__name__}: {error}"
  save_json(active.logs / "execution.json", report)
  print(f"finished index={active.index} completed={report['completed']} wall={report['wall_seconds']:.1f}s", flush=True)
  return report


def run_batch(args):
  """Bounded poll loop; no retry and no new launch after resource abort."""
  with wrapper_lock():
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "REORGANIZED.json").exists():
      raise ValueError("This dataset has been reindexed/frozen; collect into a new output directory, not a consolidated dataset or source view")
    existing_successes = []
    if args.target_successes is not None:
      existing_successes, args.start_index = resume_inventory(args)
      print(f"existing_successes={len(existing_successes)} target={args.target_successes} "
            f"next_index={args.start_index}", flush=True)
      if len(existing_successes) >= args.target_successes:
        return {"completed": True, "successful_episodes": 0,
                "total_successful_episodes": len(existing_successes),
                "target_successes": args.target_successes, "abort_reason": None}
    check_output_conflicts(args)
    (args.output_dir / "raw").mkdir(exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    run_name = f"batch_{args.start_index:06d}_{args.start_index + args.episodes - 1:06d}"
    if args.target_successes is not None:
      # A resource stop before the first launch must not prevent resuming with
      # the identical command. Episode paths themselves remain non-overwriting.
      run_name += "_target_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = args.output_dir / "logs" / (run_name + ".json")
    monitor_path = args.output_dir / "logs" / (run_name + "_resources.jsonl")
    if report_path.exists() or monitor_path.exists():
      raise FileExistsError("batch attempt already exists; use a new index range or directory")
    waiting = list(range(args.start_index, args.start_index + args.episodes))
    active: list[ActiveEpisode] = []
    results = []
    interrupted = []
    abort_reason = None
    critical_since = None
    previous = None
    started = time.monotonic()
    old_handlers = {}

    def request_stop(signum, frame):
      del frame
      interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
      old_handlers[signum] = signal.signal(signum, request_stop)
    try:
      with monitor_path.open("x", encoding="utf-8") as monitor:
        while waiting or active:
          now = time.monotonic()
          snapshot = resource_snapshot(args.output_dir)
          table = proc_table()
          snapshot["supervisor_tree"] = process_tree_metrics(table, os.getpid())
          monitor.write(json.dumps(snapshot) + "\n")
          monitor.flush()
          reason = critical_reason(snapshot, previous, args)
          previous = snapshot
          if reason:
            critical_since = now if critical_since is None else critical_since
            if now - critical_since >= args.critical_duration_seconds:
              abort_reason = reason
          else:
            critical_since = None
          if interrupted:
            abort_reason = f"supervisor_signal_{interrupted[0]}"
          for item in list(active):
            metrics = process_tree_metrics(table, item.process.pid)
            for key in item.peaks:
              item.peaks[key] = max(item.peaks[key], metrics[key])
            item.resources.write(json.dumps({**snapshot, "child_tree": metrics}) + "\n")
            item.resources.flush()
            if abort_reason:
              terminate_episode(item, abort_reason, now)
            if now - item.started > args.episode_timeout:
              terminate_episode(item, "episode_timeout", now)
            if item.term_sent is not None and now - item.term_sent >= args.term_grace_seconds and not item.kill_sent:
              signal_group(item, signal.SIGKILL)
              item.kill_sent = True
            if item.process.poll() is not None:
              results.append(finish_episode(item, args))
              active.remove(item)
              if args.target_successes is not None:
                total = len(existing_successes) + sum(r["completed"] for r in results)
                print(f"success_progress={total}/{args.target_successes}", flush=True)
          if abort_reason:
            if not active:
              break
          elif remaining_success_slots(args, len(existing_successes), results) == 0 and not active:
            break
          elif (waiting and len(active) < args.workers
                and remaining_success_slots(args, len(existing_successes), results, len(active)) > 0):
            # Launch at most one per poll, rechecking host resources each time.
            if launch_allowed(snapshot, args):
              index = waiting.pop(0)
              try:
                active.append(start_episode(args, index))
              except Exception as error:
                abort_reason = f"child_launch_error: {type(error).__name__}: {error}"
                results.append({"episode_index": index, "completed": False,
                                "error": abort_reason})
            elif not active:
              abort_reason = "insufficient_launch_resources"
              break
          if waiting or active:
            time.sleep(args.poll_seconds)
    except BaseException as error:
      abort_reason = f"supervisor_error: {type(error).__name__}: {error}"
    finally:
      # Gracefully preserve partials. Escalation targets only our session groups.
      for item in active:
        terminate_episode(item, abort_reason or "supervisor_cleanup", time.monotonic())
      cleanup_deadline = time.monotonic() + args.term_grace_seconds
      while active:
        for item in list(active):
          if item.process.poll() is not None:
            results.append(finish_episode(item, args))
            active.remove(item)
          elif time.monotonic() >= cleanup_deadline and not item.kill_sent:
            signal_group(item, signal.SIGKILL)
            item.kill_sent = True
        if active:
          time.sleep(min(args.poll_seconds, .1))
      for signum, handler in old_handlers.items():
        signal.signal(signum, handler)
    report = {
      "schema_version": "poker-controlled-batch-v1", "started_indices": [r["episode_index"] for r in results],
      "requested_episodes": args.episodes, "base_seed": args.seed,
      "start_index": args.start_index, "workers": args.workers,
      "camera_hz": args.camera_hz, "camera": "head",
      "cameras": list(args.cameras), "width": 320, "height": 240,
      "render_backend_requested": args.render_backend,
      "hdf5_buffer_rows": args.hdf5_buffer_rows,
      "physics_hz": 500, "control_hz": 100, "preset": PRESET,
      "acceptance_policy": args.acceptance_policy,
      "nice_increment": args.nice_increment,
      "verify_output_hash": args.verify_output_hash,
      "completed": (len(results) == args.episodes and all(r["completed"] for r in results)
                    if args.target_successes is None else
                    len(existing_successes) + sum(r["completed"] for r in results) >= args.target_successes),
      "successful_episodes": sum(r["completed"] for r in results),
      "existing_successful_indices": existing_successes,
      "total_successful_episodes": len(existing_successes) + sum(r["completed"] for r in results),
      "target_successes": args.target_successes,
      "not_started_indices": waiting, "abort_reason": abort_reason,
      "wall_seconds": time.monotonic() - started, "episodes": results,
      "soft_safeguards_only": True,
      "lock_scope": "this project wrapper only; manual simulations are not controlled",
      "failure_policy": "one attempt per index; never overwrite or automatically retry",
      "limits": {name: getattr(args, name) for name in (
        "min_available_memory_mib", "critical_available_memory_mib", "min_free_disk_gib",
        "episode_timeout", "critical_duration_seconds", "term_grace_seconds")},
    }
    save_json(report_path, report)
    return report


def main():
  args = parse_args()
  report = run_batch(args)
  print(json.dumps({"completed": report["completed"],
                    "successful_episodes": report["successful_episodes"],
                    "total_successful_episodes": report["total_successful_episodes"],
                    "target_successes": report["target_successes"],
                    "abort_reason": report["abort_reason"]}), flush=True)
  return 0 if report["completed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
