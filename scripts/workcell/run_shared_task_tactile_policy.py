#!/usr/bin/env python3
"""Run a task-bound FTP-1 worker through the native shared-task evaluator."""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KAIHAND_RENDER_BACKEND", "hardware")
vendor = ROOT / "scripts/collect/nvidia_egl_vendor.json"
if vendor.is_file():
  os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))

from kaihand_tactile_env.shared.tactile_task_contracts import SHARED_TACTILE_TASKS
from run_shared_task_pi05_policy import parse_args, run
from tactile_policy_binding import TaskTactileBinding


def parse_tactile_args(argv=None):
  values = list(sys.argv[1:] if argv is None else argv)
  if any(value == "--save-raw" or value.startswith("--save-raw=") for value in values):
    raise SystemExit(
      "--save-raw is not supported by the shared-task tactile runner; "
      "requests, control steps, first request, summary, and optional video are saved"
    )
  if not any(
    value == "--server" or value.startswith("--server=") for value in values
  ):
    values.extend(("--server", "local-worker"))
  if not any(
    value == "--action-interpolation"
    or value.startswith("--action-interpolation=")
    for value in values
  ):
    values.extend(("--action-interpolation", "zoh"))
  args = parse_args(values, task_choices=SHARED_TACTILE_TASKS)
  if args.task not in SHARED_TACTILE_TASKS:
    raise SystemExit(f"tactile closed loop is not registered for task {args.task!r}")
  if args.pre_roll_steps or args.pre_roll_actions is not None:
    raise SystemExit("tactile policy trials do not permit expert pre-roll actions")
  return args


def main(argv=None):
  def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGTERM, interrupted)
  args = parse_tactile_args(argv)
  try:
    return run(args, policy_binding=TaskTactileBinding())
  except Exception:
    # A native task guard is an evaluated diagnostic trial, not an IO/model error.
    summary = args.output_dir / "summary.json"
    if summary.is_file():
      report = json.loads(summary.read_text(encoding="utf-8"))
      if report.get("valid_trial") is True and report.get("stop_reason") == "guard":
        return report
    raise


if __name__ == "__main__":
  main()
