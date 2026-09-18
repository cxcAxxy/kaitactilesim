#!/usr/bin/env python3
"""Convert Card and USB 0914_200 to head+right-wrist EgoSteer datasets.

Both conversions run concurrently under one adaptive CPU budget.  Encoding and
the single full validation pass happen on CPFS; each validated tree is then
published atomically to its distinct NAS destination.  Existing head-only
datasets are never reused or overwritten.
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
SIM_PYTHON = SIM_ROOT / ".venv/bin/python"

CARD_RAW = Path("/nas/chenxianchi/datasets/sim/card/raw/0914_200")
USB_RAW = Path("/nas/chenxianchi/datasets/sim/usb_insert/raw/0914_200")
CARD_OUTPUT = Path(
  "/nas/chenxianchi/datasets/sim/card/egosteer/0914_200_head_right_wrist"
)
USB_OUTPUT = Path(
  "/nas/chenxianchi/datasets/sim/usb_insert/egosteer/0914_200_head_right_wrist"
)
DEFAULT_SCRATCH_ROOT = Path(
  "/cpfs_infra/user/chenxianchi/.conversion_staging/"
  "card_usb_0914_200_egosteer_wrist"
)
DEFAULT_LOG_DIR = SIM_ROOT / "logs/card_usb_0914_200_egosteer_wrist"

CARD_CONVERTER = SIM_ROOT / "scripts/workcell/convert_card_to_egosteer.py"
USB_CONVERTER = SIM_ROOT / "scripts/workcell/convert_usb_to_egosteer.py"
EXPECTED_EPISODES = 200
AUTO_WORKER_CAP = 8

# Workloads whose latency or NAS traffic should take precedence over conversion.
SENSITIVE_NAMES = frozenset(
  {
    "collect_poker_batch.py",
    "record_dataset.py",
    "record_usb_dataset.py",
    "train.py",
    "train_pytorch.py",
    "train_usb_tict_ddp.py",
    "evaluate_pickplace_policy_batch.py",
    "evaluate_poker_policy_batch.py",
    "evaluate_poker_pi05_policy_batch.py",
    "evaluate_usb_pi05_policy_batch.py",
    "evaluate_usb_policy_batch.py",
    "convert_card_to_egosteer.py",
    "convert_usb_to_egosteer.py",
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


@dataclass(frozen=True)
class ResourcePlan:
  affinity_cpus: tuple[int, ...]
  load_1m: float
  active_workloads: tuple[tuple[int, str], ...]
  total_workers: int
  card_workers: int
  usb_workers: int


@dataclass(frozen=True)
class Conversion:
  name: str
  raw: Path
  output: Path
  staging: Path
  converter: Path
  command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--only",
    choices=("both", "card", "usb"),
    default="both",
    help="Run both datasets (default), Card only, or USB only.",
  )
  parser.add_argument(
    "--total-workers",
    type=int,
    help=(
      "Override the shared budget; default is load-aware, capped at 8, and "
      "drops to one worker per dataset while sensitive jobs are active."
    ),
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Read-only source preflight; creates no dataset output.",
  )
  parser.add_argument(
    "--limit",
    type=int,
    help="Preflight only N HDF5 files; accepted only with --validate-only.",
  )
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help="Rehash every raw HDF5 (slow); normally sidecar hashes are trusted.",
  )
  parser.add_argument(
    "--low-priority",
    action="store_true",
    help="Force nice=10 and lowest best-effort ionice.",
  )
  parser.add_argument(
    "--scratch-root",
    type=Path,
    default=DEFAULT_SCRATCH_ROOT,
    help="CPFS/local staging root used before NAS publication.",
  )
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=DEFAULT_LOG_DIR,
    help="Directory for separate card.log and usb.log progress logs.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print resource decisions and commands without starting conversion.",
  )
  args = parser.parse_args(argv)
  selected = 2 if args.only == "both" else 1
  if args.total_workers is not None and args.total_workers < selected:
    parser.error(f"--total-workers must be at least {selected}")
  if args.limit is not None and args.limit < 1:
    parser.error("--limit must be positive")
  if args.limit is not None and not args.validate_only:
    parser.error("--limit is only allowed with --validate-only")
  return args


def _affinity() -> tuple[int, ...]:
  if hasattr(os, "sched_getaffinity"):
    return tuple(sorted(os.sched_getaffinity(0)))
  return tuple(range(os.cpu_count() or 1))


def active_sensitive_processes(
  proc_root: Path = Path("/proc"),
) -> tuple[tuple[int, str], ...]:
  result = []
  for directory in proc_root.iterdir():
    if not directory.name.isdigit() or int(directory.name) == os.getpid():
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
    matches = SENSITIVE_NAMES.intersection(Path(item).name for item in arguments)
    if matches:
      result.append((int(directory.name), sorted(matches)[0]))
  return tuple(sorted(result))


def choose_total_workers(
  requested: int | None,
  *,
  selected: int,
  affinity_cpus: int,
  load_1m: float,
  active_workloads: int,
) -> int:
  if selected not in (1, 2) or affinity_cpus < selected:
    raise ValueError("CPU affinity cannot supply every selected dataset")
  if requested is not None:
    if requested < selected or requested > affinity_cpus:
      raise ValueError(
        f"requested workers must be in [{selected}, {affinity_cpus}]"
      )
    return requested
  reserve = min(
    max(0, affinity_cpus - selected),
    max(2, math.ceil(affinity_cpus * 0.20)),
  )
  spare = math.floor(affinity_cpus - max(0.0, load_1m) - reserve)
  budget = min(
    AUTO_WORKER_CAP,
    max(selected, affinity_cpus - reserve),
    max(selected, spare),
  )
  if active_workloads:
    budget = selected
  return max(selected, budget)


def resource_plan(
  only: str, requested: int | None, *, load_1m: float | None = None
) -> ResourcePlan:
  affinity = _affinity()
  active = active_sensitive_processes()
  load = os.getloadavg()[0] if load_1m is None else load_1m
  selected = 2 if only == "both" else 1
  total = choose_total_workers(
    requested,
    selected=selected,
    affinity_cpus=len(affinity),
    load_1m=load,
    active_workloads=len(active),
  )
  if only == "card":
    card_workers, usb_workers = total, 0
  elif only == "usb":
    card_workers, usb_workers = 0, total
  else:
    card_workers, usb_workers = (total + 1) // 2, total // 2
  return ResourcePlan(
    affinity, load, active, total, card_workers, usb_workers
  )


def _priority_prefix(enabled: bool) -> tuple[str, ...]:
  if not enabled:
    return ()
  return ("/usr/bin/ionice", "-c", "2", "-n", "7", "/usr/bin/nice", "-n", "10")


def build_conversions(
  plan: ResourcePlan,
  *,
  validate_only: bool,
  limit: int | None,
  verify_source_hash: bool,
  scratch_root: Path,
  low_priority: bool,
) -> tuple[Conversion, ...]:
  optional = tuple(
    flag
    for enabled, flag in (
      (validate_only, "--validate-only"),
      (verify_source_hash, "--verify-source-hash"),
    )
    if enabled
  )
  if limit is not None:
    optional += ("--limit", str(limit))
  specs = (
    (
      "card",
      plan.card_workers,
      CARD_RAW,
      CARD_OUTPUT,
      scratch_root / "card",
      CARD_CONVERTER,
      "card_0914_200_head_right_wrist",
    ),
    (
      "usb",
      plan.usb_workers,
      USB_RAW,
      USB_OUTPUT,
      scratch_root / "usb",
      USB_CONVERTER,
      "usb_insert_0914_200_head_right_wrist",
    ),
  )
  result = []
  for name, workers, raw, output, staging, converter, dataset_name in specs:
    if not workers:
      continue
    command = (
      *_priority_prefix(low_priority),
      str(SIM_PYTHON),
      str(converter),
      "--input-dir",
      str(raw),
      "--output-dir",
      str(output),
      "--expected-episodes",
      str(EXPECTED_EPISODES),
      "--dataset-name",
      dataset_name,
      "--workers",
      str(workers),
      "--cameras",
      "head",
      "right_wrist",
      "--staging-root",
      str(staging),
      *optional,
    )
    result.append(Conversion(name, raw, output, staging, converter, command))
  return tuple(result)


def _nearest_existing(path: Path) -> Path:
  current = path
  while not current.exists() and current != current.parent:
    current = current.parent
  return current


def _require_writable_parent(path: Path) -> None:
  parent = _nearest_existing(path.parent)
  readonly = bool(os.statvfs(parent).f_flag & getattr(os, "ST_RDONLY", 1))
  if readonly or not os.access(parent, os.W_OK):
    raise PermissionError(f"filesystem is not writable: {parent}")


def preflight(
  conversions: Sequence[Conversion],
  *,
  dry_run: bool,
  validate_only: bool,
  log_dir: Path,
) -> None:
  if not SIM_PYTHON.is_file():
    raise FileNotFoundError(f"missing Python interpreter: {SIM_PYTHON}")
  for conversion in conversions:
    if not conversion.raw.is_dir():
      raise FileNotFoundError(f"missing raw dataset: {conversion.raw}")
    if not conversion.converter.is_file():
      raise FileNotFoundError(f"missing converter: {conversion.converter}")
    if os.path.lexists(conversion.output):
      raise FileExistsError(
        f"refusing to overwrite {conversion.name} output: {conversion.output}"
      )
    if not dry_run and not validate_only:
      _require_writable_parent(conversion.output)
      _require_writable_parent(conversion.staging)
  if not dry_run:
    log_dir.mkdir(parents=True, exist_ok=True)


def _print_plan(
  plan: ResourcePlan,
  conversions: Sequence[Conversion],
  *,
  low_priority: bool,
  log_dir: Path,
) -> None:
  active = ", ".join(f"{pid}:{name}" for pid, name in plan.active_workloads)
  print(
    "resource plan: "
    f"affinity_cpus={len(plan.affinity_cpus)} load_1m={plan.load_1m:.2f} "
    f"active={active or 'none'} priority={'low' if low_priority else 'normal'} "
    f"total_workers={plan.total_workers} card={plan.card_workers} "
    f"usb={plan.usb_workers}",
    flush=True,
  )
  environment = " ".join(f"{key}={value}" for key, value in THREAD_LIMITS.items())
  for conversion in conversions:
    print(
      f"[{conversion.name}] log={log_dir / (conversion.name + '.log')} "
      f"cd {shlex.quote(str(SIM_ROOT))} && {environment} "
      f"{shlex.join(conversion.command)}",
      flush=True,
    )


def _stop_children(
  children: Sequence[tuple[Conversion, subprocess.Popen, object]],
) -> None:
  for _conversion, process, _stream in children:
    if process.poll() is None:
      try:
        os.killpg(process.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
  deadline = time.monotonic() + 30.0
  for _conversion, process, _stream in children:
    try:
      process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
      try:
        os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass


def launch(conversions: Sequence[Conversion], log_dir: Path) -> int:
  environment = os.environ.copy()
  environment.update(THREAD_LIMITS)
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  environment["PYTHONUNBUFFERED"] = "1"
  environment.pop("LEROBOT_HOME", None)
  children = []
  try:
    for conversion in conversions:
      log_path = log_dir / f"{conversion.name}.log"
      stream = log_path.open("w", encoding="utf-8", buffering=1)
      try:
        process = subprocess.Popen(
          conversion.command,
          cwd=SIM_ROOT,
          env=environment,
          stdout=stream,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
      except BaseException:
        stream.close()
        raise
      children.append((conversion, process, stream))
      print(
        f"started {conversion.name}: pid={process.pid} log={log_path}",
        flush=True,
      )
  except BaseException:
    _stop_children(children)
    for _conversion, _process, stream in children:
      stream.close()
    raise

  try:
    while any(process.poll() is None for _conversion, process, _stream in children):
      time.sleep(0.5)
  except KeyboardInterrupt:
    print("interrupt received; stopping both converters", file=sys.stderr)
    _stop_children(children)
    return_code = 130
  else:
    return_code = 0
  finally:
    for _conversion, _process, stream in children:
      stream.close()

  for conversion, process, _stream in children:
    print(f"finished {conversion.name}: exit={process.returncode}", flush=True)
    if process.returncode:
      return_code = 1
  return return_code


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  plan = resource_plan(args.only, args.total_workers)
  low_priority = args.low_priority or bool(plan.active_workloads)
  scratch_root = args.scratch_root.expanduser().resolve(strict=False)
  log_dir = args.log_dir.expanduser().resolve(strict=False)
  conversions = build_conversions(
    plan,
    validate_only=args.validate_only,
    limit=args.limit,
    verify_source_hash=args.verify_source_hash,
    scratch_root=scratch_root,
    low_priority=low_priority,
  )
  preflight(
    conversions,
    dry_run=args.dry_run,
    validate_only=args.validate_only,
    log_dir=log_dir,
  )
  _print_plan(
    plan,
    conversions,
    low_priority=low_priority,
    log_dir=log_dir,
  )
  if args.dry_run:
    return 0
  return launch(conversions, log_dir)


if __name__ == "__main__":
  raise SystemExit(main())
