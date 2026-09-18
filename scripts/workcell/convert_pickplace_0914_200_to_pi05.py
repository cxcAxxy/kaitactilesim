#!/usr/bin/env python3
"""Convert the fixed PickPlace 0914_200 batch to the OpenPI/LeRobot format.

The raw and published paths are intentionally fixed. Temporary PNG/Parquet
work is placed on CPFS and copied to NAS only after validation. The launcher
uses an adaptive worker budget and drops to one low-priority worker while a
recorder or model-training process is active.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


SIM_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = SIM_ROOT.parent / "openpi"

DATASET_ROOT = Path("/nas/chenxianchi/datasets/sim/pickplace")
RAW_DIR = DATASET_ROOT / "raw/0914_200"
PI05_OUTPUT = DATASET_ROOT / "pi05/0914_200"
DEFAULT_SCRATCH_ROOT = Path(
  "/cpfs_infra/user/chenxianchi/.conversion_staging/pickplace_0914_200"
)

OPENPI_PYTHON = OPENPI_ROOT / ".venv-pi05/bin/python"
PI05_CONVERTER = OPENPI_ROOT / "examples/kaihand/convert_pickplace_to_lerobot.py"
EXPECTED_EPISODES = 200
AUTO_WORKER_CAP = 8

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
THREAD_LIMITS = {
  "BLIS_NUM_THREADS": "1",
  "LP_NUM_THREADS": "1",
  "MKL_NUM_THREADS": "1",
  "NUMEXPR_NUM_THREADS": "1",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--workers",
    type=int,
    help=(
      "Override the adaptive fingerprint/PNG worker count. The automatic "
      "budget is capped at 8 and becomes 1 while collection/training is active."
    ),
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Validate the selected source episodes without creating output.",
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help=(
      "Re-hash all raw HDF5 payloads instead of trusting the acquisition "
      "sidecars. This is slower and normally unnecessary."
    ),
  )
  parser.add_argument(
    "--limit",
    type=int,
    help="Validate only the first N episodes; accepted only with --validate-only.",
  )
  parser.add_argument(
    "--scratch-root",
    type=Path,
    default=DEFAULT_SCRATCH_ROOT,
    help="Fast CPFS/local staging directory used before NAS publication.",
  )
  parser.add_argument(
    "--low-priority",
    action="store_true",
    help="Force nice=10 and lowest best-effort ionice priority.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the resolved plan and command without starting conversion.",
  )
  args = parser.parse_args(argv)
  if args.workers is not None and args.workers <= 0:
    parser.error("--workers must be positive")
  if args.limit is not None and args.limit <= 0:
    parser.error("--limit must be positive")
  if args.limit is not None and not args.validate_only:
    parser.error("--limit is only allowed with --validate-only")
  return args


def validate_layout(raw: Path, output: Path) -> None:
  raw = raw.resolve()
  output = output.resolve()
  if raw.name != "0914_200" or raw.parent.name != "raw":
    raise ValueError(f"unexpected raw dataset path: {raw}")
  expected = raw.parent.parent / "pi05/0914_200"
  if output != expected:
    raise ValueError(f"Pi0.5 output must be {expected}, got {output}")
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


def _priority_prefix(enabled: bool) -> tuple[str, ...]:
  if not enabled:
    return ()
  return (
    "/usr/bin/ionice",
    "-c",
    "2",
    "-n",
    "7",
    "/usr/bin/nice",
    "-n",
    "10",
  )


def build_command(
  *,
  workers: int,
  scratch_root: Path,
  validate_only: bool,
  verify_source_hash: bool,
  limit: int | None,
  low_priority: bool,
) -> tuple[str, ...]:
  command = (
    *_priority_prefix(low_priority),
    str(OPENPI_PYTHON),
    str(PI05_CONVERTER),
    "--input-dir",
    str(RAW_DIR),
    "--output-dir",
    str(PI05_OUTPUT),
    "--expected-episodes",
    str(EXPECTED_EPISODES),
    "--staging-root",
    str(scratch_root),
    "--fingerprint-workers",
    str(workers),
    "--image-writer-processes",
    "0",
    "--image-writer-threads",
    str(workers),
  )
  if validate_only:
    command += ("--validate-only",)
  if verify_source_hash:
    command += ("--verify-source-hash",)
  if limit is not None:
    command += ("--limit", str(limit))
  return command


def _nearest_existing(path: Path) -> Path:
  current = path
  while not current.exists() and current != current.parent:
    current = current.parent
  return current


def preflight(*, no_write: bool) -> None:
  if not RAW_DIR.is_dir():
    raise FileNotFoundError(f"raw dataset does not exist: {RAW_DIR}")
  if os.path.lexists(PI05_OUTPUT):
    raise FileExistsError(f"refusing to overwrite existing Pi0.5 output: {PI05_OUTPUT}")
  if not OPENPI_ROOT.is_dir():
    raise FileNotFoundError(f"missing OpenPI root: {OPENPI_ROOT}")
  if not OPENPI_PYTHON.is_file():
    raise FileNotFoundError(f"missing OpenPI Python: {OPENPI_PYTHON}")
  if not PI05_CONVERTER.is_file():
    raise FileNotFoundError(f"missing PickPlace converter: {PI05_CONVERTER}")
  if no_write:
    return
  parent = _nearest_existing(PI05_OUTPUT.parent)
  readonly = bool(os.statvfs(parent).f_flag & getattr(os, "ST_RDONLY", 1))
  if readonly or not os.access(parent, os.W_OK):
    raise PermissionError(f"output filesystem is not writable: {parent}")
  PI05_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  validate_layout(RAW_DIR, PI05_OUTPUT)
  affinity = _affinity()
  active = active_sensitive_processes()
  load_1m = os.getloadavg()[0]
  workers = choose_workers(
    args.workers,
    affinity_cpus=len(affinity),
    load_1m=load_1m,
    active_sensitive=len(active),
  )
  low_priority = args.low_priority or bool(active)
  scratch_root = args.scratch_root.expanduser().resolve(strict=False)
  command = build_command(
    workers=workers,
    scratch_root=scratch_root,
    validate_only=args.validate_only,
    verify_source_hash=args.verify_source_hash,
    limit=args.limit,
    low_priority=low_priority,
  )
  preflight(no_write=args.dry_run or args.validate_only)

  active_text = ",".join(f"{name}:{pid}" for pid, name in active) or "none"
  print(
    "resource plan: "
    f"affinity_cpus={len(affinity)} load_1m={load_1m:.2f} "
    f"active_sensitive={active_text} priority={'low' if low_priority else 'normal'} "
    f"workers={workers}",
    flush=True,
  )
  environment_text = " ".join(f"{key}={value}" for key, value in THREAD_LIMITS.items())
  print(
    f"cd {shlex.quote(str(OPENPI_ROOT))} && {environment_text} {shlex.join(command)}",
    flush=True,
  )
  if args.dry_run:
    return 0

  environment = os.environ.copy()
  environment.update(THREAD_LIMITS)
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  environment["PYTHONUNBUFFERED"] = "1"
  environment.pop("LEROBOT_HOME", None)
  completed = subprocess.run(command, cwd=OPENPI_ROOT, env=environment, check=False)
  if completed.returncode:
    print(f"PickPlace Pi0.5 conversion failed: exit={completed.returncode}", file=sys.stderr)
  return completed.returncode


if __name__ == "__main__":
  raise SystemExit(main())
