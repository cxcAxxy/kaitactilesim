#!/usr/bin/env python3
"""Replace the empty four-capacity queue without touching running evaluations."""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
queue = importlib.import_module("run_pi05_new_tasks_queue")

OLD_ROOT = (
  queue.worker.EVALUATION_PARENT / "pi05_h50_k50_new_tasks_20261013_180300_frozen_q4"
)
OLD_BUNDLE = queue.worker.SNAPSHOT_PARENT / "pi05_new_tasks_20261013_175500"
OLD_PID = 2805803


def command_line(pid: int) -> list[str]:
  return Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")


def children(pid: int) -> list[int]:
  return [
    int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
  ]


def validate_empty_queue() -> dict:
  status = queue.read_json(OLD_ROOT / "status.json")
  process = queue.process_info(OLD_PID)
  expected_command = [
    str(queue.CONTROLLER_PYTHON),
    str(OLD_BUNDLE / "kaitactilesim/scripts/deploy/run_pi05_new_tasks_queue.py"),
    "--job-root",
    str(OLD_ROOT),
    "--simulation-root",
    str(OLD_BUNDLE / "kaitactilesim"),
    "--openpi-root",
    str(OLD_BUNDLE / "openpi"),
  ]
  groups = status.get("groups", [])
  expected_groups = [
    (index, task, step, str(checkpoint))
    for index, (task, step, checkpoint) in enumerate(queue.worker.NEW_GROUPS)
  ]
  if (
    status.get("pid") != OLD_PID
    or status.get("status") != "running"
    or status.get("stage") != "waiting_for_capacity"
    or status.get("active_new_groups") != 0
    or status.get("completed_trials") != 0
    or status.get("settings", {}).get("total_checkpoint_worker_cap") != 4
    or [
      (
        group.get("group_index"),
        group.get("task"),
        group.get("step"),
        group.get("checkpoint"),
      )
      for group in groups
    ]
    != expected_groups
    or any(
      group.get("status") != "queued" or group.get("worker_pid") is not None
      for group in groups
    )
    or any(Path(group["output_dir"]).exists() for group in groups)
    or process is None
    or process["state"] == "Z"
    or process["ppid"] != 1
    or command_line(OLD_PID) != expected_command
    or children(OLD_PID)
  ):
    raise RuntimeError(
      "old new-task queue is no longer empty or its process identity changed"
    )
  return {
    "pid": OLD_PID,
    "starttime": process["starttime"],
    "command": expected_command,
    "status": status,
  }


def stop_empty_queue() -> dict:
  if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise RuntimeError("queue upgrade requires system Python with Linux pidfd support")
  descriptor = os.pidfd_open(OLD_PID)
  paused = False
  try:
    original = validate_empty_queue()
    signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
    paused = True
    deadline = time.monotonic() + 10
    while True:
      actual = queue.process_info(OLD_PID)
      if actual is None or actual["starttime"] != original["starttime"]:
        raise RuntimeError("owned queue disappeared or PID was reused")
      if actual["state"] in ("T", "t"):
        break
      if time.monotonic() >= deadline:
        raise TimeoutError("queue did not stop at its allocation boundary")
      time.sleep(0.05)
    # Recheck after pausing; no fork can race this check and the final termination.
    validate_empty_queue()
    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    signal.pidfd_send_signal(descriptor, signal.SIGCONT)
    paused = False
    deadline = time.monotonic() + 30
    while True:
      actual = queue.process_info(OLD_PID)
      if actual is None or actual["state"] == "Z":
        break
      if actual["starttime"] != original["starttime"]:
        raise RuntimeError("queue PID changed while waiting for exit")
      if time.monotonic() >= deadline:
        raise TimeoutError("empty queue termination did not complete")
      time.sleep(0.1)
    final = queue.read_json(OLD_ROOT / "status.json")
    if final.get("status") != "interrupted" or final.get("active_new_groups") != 0:
      raise RuntimeError("empty queue did not publish its interrupted status")
    return {
      "previous_job_root": str(OLD_ROOT),
      "previous_controller_pid": OLD_PID,
      "previous_controller_starttime": original["starttime"],
      "previous_controller_stopped": True,
      "previous_queue_was_empty": True,
      "previous_results_deleted": False,
      "existing_evaluation_processes_untouched": True,
      "completed_utc": queue.now(),
    }
  finally:
    try:
      if paused:
        try:
          signal.pidfd_send_signal(descriptor, signal.SIGCONT)
        except ProcessLookupError:
          pass
    finally:
      os.close(descriptor)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  args = parser.parse_args()
  job_root = args.job_root.absolute()
  simulation_root = Path(__file__).resolve().parents[2]
  openpi_root = simulation_root.parent / "openpi"
  if queue.CAPACITY != 6 or queue.EXISTING_GROUP_COUNT != 4:
    raise RuntimeError("upgrade must retain four old groups and set total capacity six")
  queue.validate_configuration(job_root, simulation_root, openpi_root)
  if job_root.exists():
    raise FileExistsError(f"refusing to overwrite {job_root}")
  for index in range(len(queue.worker.NEW_GROUPS)):
    if not queue.port_free(queue.FIRST_PORT + index):
      raise RuntimeError(f"new evaluation port is not free: {queue.FIRST_PORT + index}")
  transition = stop_empty_queue()
  code = queue.start(job_root, simulation_root, openpi_root)
  if code:
    raise RuntimeError(f"six-capacity queue start failed: {code}")
  queue.write_json(
    job_root / "queue_upgrade_record.json",
    {**transition, "new_job_root": str(job_root), "total_checkpoint_worker_cap": 6},
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
