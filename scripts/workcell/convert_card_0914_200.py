#!/usr/bin/env python3
"""Launch the 0914_200 card conversions with a shared adaptive CPU budget.

The two output paths are intentionally fixed.  This launcher only coordinates
the format-specific converters; each converter remains responsible for its own
transactional output and dataset validation.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SIM_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = SIM_ROOT.parent / "openpi"

DATASET_ROOT = Path("/nas/chenxianchi/datasets/sim/card")
RAW_DIR = DATASET_ROOT / "raw/0914_200"
PI05_OUTPUT = DATASET_ROOT / "pi05/0914_200"
EGOSTEER_OUTPUT = DATASET_ROOT / "egosteer/0914_200"
DEFAULT_SCRATCH_ROOT = Path(
  "/cpfs_infra/user/chenxianchi/.conversion_staging/card_0914_200"
)

SIM_PYTHON = SIM_ROOT / ".venv/bin/python"
OPENPI_PYTHON = OPENPI_ROOT / ".venv-pi05/bin/python"
PI05_CONVERTER = OPENPI_ROOT / "examples/kaihand/convert_card_to_lerobot.py"
EGOSTEER_CONVERTER = SIM_ROOT / "scripts/workcell/convert_card_to_egosteer.py"

EXPECTED_EPISODES = 200
AUTO_WORKER_CAP = 8
CAPTURE_NAMES = frozenset(
  {"record_dataset.py", "record_usb_dataset.py", "collect_poker_batch.py"}
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


@dataclass(frozen=True)
class ResourcePlan:
  affinity_cpus: tuple[int, ...]
  load_1m: float
  active_captures: tuple[tuple[int, str], ...]
  total_workers: int
  pi05_workers: int
  egosteer_workers: int


@dataclass(frozen=True)
class Conversion:
  name: str
  cwd: Path
  output: Path
  command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--only",
    choices=("both", "pi05", "egosteer"),
    default="both",
    help="Run both conversions (default) or only one named format.",
  )
  parser.add_argument(
    "--total-workers",
    type=int,
    help=(
      "Override the shared worker budget. With --only=both it is split between "
      "the converters; otherwise it is assigned to the selected converter."
    ),
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help=(
      "Validate raw inputs in both converters without creating output; "
      "combine with --limit for a focused HDF5 preflight."
    ),
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help=(
      "Re-hash every raw HDF5 instead of trusting its capture sidecar. This "
      "adds a full 39 GiB source read per selected converter."
    ),
  )
  parser.add_argument(
    "--limit",
    type=int,
    help=(
      "Limit source validation for a focused smoke check. For safety this is "
      "only accepted together with --validate-only."
    ),
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the resource decision and final commands without starting them.",
  )
  parser.add_argument(
    "--low-priority",
    action="store_true",
    help=(
      "Run converters with nice=10 and lowest best-effort ionice. This is "
      "enabled automatically while a dataset recorder is active; idle-machine "
      "conversions otherwise run at normal priority for higher throughput."
    ),
  )
  parser.add_argument(
    "--scratch-root",
    type=Path,
    default=DEFAULT_SCRATCH_ROOT,
    help=(
      "Fast local/CPFS work directory for Pi0.5 temporary PNG files. The "
      "validated final Parquet dataset is copied to NAS and checked again."
    ),
  )
  args = parser.parse_args(argv)
  selected = 2 if args.only == "both" else 1
  if args.total_workers is not None and args.total_workers < selected:
    parser.error(
      f"--total-workers must be at least {selected} with --only={args.only}"
    )
  if args.limit is not None and args.limit < 1:
    parser.error("--limit must be positive")
  if args.limit is not None and not args.validate_only:
    parser.error(
      "--limit is only allowed with --validate-only; an incomplete conversion "
      "must never occupy the fixed production output"
    )
  return args


def _is_relative_to(path: Path, parent: Path) -> bool:
  try:
    path.relative_to(parent)
  except ValueError:
    return False
  return True


def validate_layout(raw: Path, pi05: Path, egosteer: Path) -> None:
  """Reject swapped, nested, or non-canonical format destinations."""
  raw = raw.resolve()
  pi05 = pi05.resolve()
  egosteer = egosteer.resolve()
  if raw.name != "0914_200" or raw.parent.name != "raw":
    raise ValueError(f"unexpected raw dataset path: {raw}")
  common = raw.parent.parent
  expected_pi05 = common / "pi05/0914_200"
  expected_egosteer = common / "egosteer/0914_200"
  if pi05 != expected_pi05:
    raise ValueError(f"Pi0.5 output must be {expected_pi05}, got {pi05}")
  if egosteer != expected_egosteer:
    raise ValueError(f"EgoSteer output must be {expected_egosteer}, got {egosteer}")
  paths = (raw, pi05, egosteer)
  for index, first in enumerate(paths):
    for second in paths[index + 1 :]:
      if first == second or _is_relative_to(first, second) or _is_relative_to(
        second, first
      ):
        raise ValueError(f"dataset paths overlap: {first} and {second}")


def _affinity() -> tuple[int, ...]:
  if hasattr(os, "sched_getaffinity"):
    return tuple(sorted(os.sched_getaffinity(0)))
  return tuple(range(os.cpu_count() or 1))


def active_capture_processes(
  proc_root: Path = Path("/proc"),
) -> tuple[tuple[int, str], ...]:
  """Return live simulation recorders without requiring psutil."""
  captures = []
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
    matches = CAPTURE_NAMES.intersection(Path(item).name for item in arguments)
    if matches:
      captures.append((int(directory.name), sorted(matches)[0]))
  return tuple(sorted(captures))


def choose_total_workers(
  requested: int | None,
  *,
  selected_conversions: int,
  affinity_cpus: int,
  load_1m: float,
  active_captures: int,
) -> int:
  """Choose a conservative shared budget from CPUs visible to this process."""
  if selected_conversions not in (1, 2):
    raise ValueError("selected_conversions must be one or two")
  if affinity_cpus < 1:
    raise ValueError("affinity_cpus must be positive")
  if affinity_cpus < selected_conversions:
    raise ValueError(
      "CPU affinity cannot give every selected conversion one worker"
    )
  if requested is not None:
    if requested < selected_conversions:
      raise ValueError("requested workers cannot give every conversion one worker")
    if requested > affinity_cpus:
      raise ValueError(
        f"requested workers ({requested}) exceed CPU affinity ({affinity_cpus})"
      )
    return requested

  # Preserve at least 20% (and two CPUs where available) for the IDE, inference,
  # filesystem work, and other jobs. Load average includes runnable and D-state
  # tasks, which is desirable here because the source lives on network NAS.
  reserve = min(
    max(0, affinity_cpus - selected_conversions),
    max(2, math.ceil(affinity_cpus * 0.20)),
  )
  load_spare = math.floor(affinity_cpus - max(0.0, load_1m) - reserve)
  budget = max(selected_conversions, load_spare)
  budget = min(
    budget,
    AUTO_WORKER_CAP,
    max(selected_conversions, affinity_cpus - reserve),
  )

  # Simulation capture is latency-sensitive and also performs HDF5/image I/O.
  # While any recorder is live, permit only one worker per selected converter.
  if active_captures:
    budget = min(budget, selected_conversions)
  return max(selected_conversions, budget)


def split_workers(total: int, only: str) -> tuple[int, int]:
  if only == "pi05":
    return total, 0
  if only == "egosteer":
    return 0, total
  if only != "both" or total < 2:
    raise ValueError("both conversions require at least two total workers")
  return (total + 1) // 2, total // 2


def resource_plan(
  only: str, requested: int | None, *, load_1m: float | None = None
) -> ResourcePlan:
  affinity = _affinity()
  captures = active_capture_processes()
  load = os.getloadavg()[0] if load_1m is None else load_1m
  selected = 2 if only == "both" else 1
  total = choose_total_workers(
    requested,
    selected_conversions=selected,
    affinity_cpus=len(affinity),
    load_1m=load,
    active_captures=len(captures),
  )
  pi05, egosteer = split_workers(total, only)
  return ResourcePlan(affinity, load, captures, total, pi05, egosteer)


def _priority_prefix(enabled: bool = False) -> tuple[str, ...]:
  if not enabled:
    return ()
  return ("/usr/bin/ionice", "-c", "2", "-n", "7", "/usr/bin/nice", "-n", "10")


def build_conversions(
  plan: ResourcePlan,
  *,
  validate_only: bool,
  verify_source_hash: bool,
  limit: int | None,
  low_priority: bool = False,
  scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Conversion, ...]:
  result = []
  common = ("--input-dir", str(RAW_DIR), "--expected-episodes", str(EXPECTED_EPISODES))
  optional = tuple(
    argument
    for enabled, argument in (
      (validate_only, "--validate-only"),
      (verify_source_hash, "--verify-source-hash"),
    )
    if enabled
  )
  if limit is not None:
    optional += ("--limit", str(limit))
  if plan.pi05_workers:
    workers = plan.pi05_workers
    # Fingerprinting and image writing are separate phases. Threads let Pillow
    # encode concurrently without copying every two-camera frame through a
    # multiprocessing queue.
    image_processes = 0
    image_threads = workers
    command = (
      *_priority_prefix(low_priority),
      str(OPENPI_PYTHON),
      str(PI05_CONVERTER),
      *common,
      "--output-dir",
      str(PI05_OUTPUT),
      "--staging-root",
      str(scratch_root),
      "--fingerprint-workers",
      str(workers),
      "--image-writer-processes",
      str(image_processes),
      "--image-writer-threads",
      str(image_threads),
      *optional,
    )
    result.append(Conversion("pi05", OPENPI_ROOT, PI05_OUTPUT, command))
  if plan.egosteer_workers:
    command = (
      *_priority_prefix(low_priority),
      str(SIM_PYTHON),
      str(EGOSTEER_CONVERTER),
      *common,
      "--output-dir",
      str(EGOSTEER_OUTPUT),
      "--dataset-name",
      "0914_200",
      "--workers",
      str(plan.egosteer_workers),
      *optional,
    )
    result.append(Conversion("egosteer", SIM_ROOT, EGOSTEER_OUTPUT, command))
  return tuple(result)


def _nearest_existing(path: Path) -> Path:
  current = path
  while not current.exists() and current != current.parent:
    current = current.parent
  return current


def _require_writable_parent(output: Path) -> None:
  parent = _nearest_existing(output.parent)
  readonly = bool(os.statvfs(parent).f_flag & getattr(os, "ST_RDONLY", 1))
  if readonly or not os.access(parent, os.W_OK):
    raise PermissionError(f"output filesystem is not writable: {parent}")


def preflight(conversions: Sequence[Conversion], *, no_write: bool) -> None:
  if not RAW_DIR.is_dir():
    raise FileNotFoundError(f"raw dataset does not exist: {RAW_DIR}")
  for conversion in conversions:
    if os.path.lexists(conversion.output):
      raise FileExistsError(
        f"refusing to overwrite existing {conversion.name} output: {conversion.output}"
      )
    if not conversion.cwd.is_dir():
      raise FileNotFoundError(f"missing project root: {conversion.cwd}")
    if conversion.name == "pi05":
      executable, script = OPENPI_PYTHON, PI05_CONVERTER
    else:
      executable, script = SIM_PYTHON, EGOSTEER_CONVERTER
    if not executable.is_file():
      raise FileNotFoundError(f"missing Python interpreter: {executable}")
    if not script.is_file():
      raise FileNotFoundError(f"missing converter: {script}")
    if not no_write:
      _require_writable_parent(conversion.output)
  if not no_write:
    for conversion in conversions:
      conversion.output.parent.mkdir(parents=True, exist_ok=True)


def _print_plan(
  plan: ResourcePlan, conversions: Sequence[Conversion], *, low_priority: bool
) -> None:
  capture_counts = {
    name: sum(item_name == name for _pid, item_name in plan.active_captures)
    for _pid, name in plan.active_captures
  }
  captures = ", ".join(
    f"{name}x{count}" for name, count in sorted(capture_counts.items())
  )
  print(
    "resource plan: "
    f"affinity_cpus={len(plan.affinity_cpus)} "
    f"load_1m={plan.load_1m:.2f} "
    f"active_captures={captures or 'none'} "
    f"priority={'low' if low_priority else 'normal'} "
    f"total_workers={plan.total_workers} "
    f"pi05={plan.pi05_workers} egosteer={plan.egosteer_workers}",
    flush=True,
  )
  environment = " ".join(f"{key}={value}" for key, value in THREAD_LIMITS.items())
  for conversion in conversions:
    print(
      f"[{conversion.name}] cd {shlex.quote(str(conversion.cwd))} && "
      f"{environment} {shlex.join(conversion.command)}",
      flush=True,
    )


def _stop_children(children: Sequence[tuple[Conversion, subprocess.Popen]]) -> None:
  for _conversion, process in children:
    if process.poll() is None:
      try:
        os.killpg(process.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
  deadline = time.monotonic() + 30.0
  for _conversion, process in children:
    remaining = max(0.0, deadline - time.monotonic())
    try:
      process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass


def launch(conversions: Sequence[Conversion]) -> int:
  environment = os.environ.copy()
  environment.update(THREAD_LIMITS)
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  environment["PYTHONUNBUFFERED"] = "1"
  environment.pop("LEROBOT_HOME", None)
  children: list[tuple[Conversion, subprocess.Popen]] = []
  try:
    for conversion in conversions:
      process = subprocess.Popen(
        conversion.command,
        cwd=conversion.cwd,
        env=environment,
        start_new_session=True,
      )
      children.append((conversion, process))
      print(f"started {conversion.name}: pid={process.pid}", flush=True)
  except BaseException:
    _stop_children(children)
    raise

  try:
    while any(process.poll() is None for _conversion, process in children):
      time.sleep(0.5)
  except KeyboardInterrupt:
    print("interrupt received; forwarding SIGINT to both converters", file=sys.stderr)
    _stop_children(children)
    return 130

  failed = []
  for conversion, process in children:
    code = process.returncode
    print(f"finished {conversion.name}: exit={code}", flush=True)
    if code:
      failed.append((conversion.name, code))
  if failed:
    print(
      "conversion failure(s): "
      + ", ".join(f"{name}=exit {code}" for name, code in failed),
      file=sys.stderr,
    )
    return 1
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  validate_layout(RAW_DIR, PI05_OUTPUT, EGOSTEER_OUTPUT)
  plan = resource_plan(args.only, args.total_workers)
  conversions = build_conversions(
    plan,
    validate_only=args.validate_only,
    verify_source_hash=args.verify_source_hash,
    limit=args.limit,
    low_priority=args.low_priority or bool(plan.active_captures),
    scratch_root=args.scratch_root.expanduser().resolve(strict=False),
  )
  preflight(conversions, no_write=args.dry_run or args.validate_only)
  _print_plan(
    plan,
    conversions,
    low_priority=args.low_priority or bool(plan.active_captures),
  )
  if args.dry_run:
    return 0
  return launch(conversions)


if __name__ == "__main__":
  raise SystemExit(main())
