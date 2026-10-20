#!/usr/bin/env python3
"""Retry only the two zero-valid USB groups without disturbing primary workers."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
queue = importlib.import_module("run_pi05_new_tasks_queue")
resume = importlib.import_module("resume_pi05_new_tasks_queue")
PRIMARY_ROOT = (
  queue.worker.EVALUATION_PARENT
  / "pi05_h50_k50_new_tasks_20261013_201900_resume56_frozen_q6"
)
PRIMARY_CONTROLLER_PID = 2852241
USB_INDICES = (4, 5)
LOCK_NAME = ".pi05_h50_usb_retry_scheduler.lock"


def read_primary() -> dict:
  if PRIMARY_ROOT.is_symlink():
    raise ValueError("primary queue directory must not be a symlink")
  primary = queue.read_json(PRIMARY_ROOT / "status.json")
  if primary.get("pid") != PRIMARY_CONTROLLER_PID:
    raise ValueError("primary controller identity changed")
  if primary.get("status") not in ("running", "complete", "failed"):
    raise RuntimeError("primary queue is not healthy or terminal")
  groups = primary.get("groups", [])
  if [(g.get("task"), g.get("step"), g.get("checkpoint")) for g in groups] != [
    (task, step, str(checkpoint)) for task, step, checkpoint in queue.worker.NEW_GROUPS
  ] or [g.get("group_index") for g in groups] != list(range(6)):
    raise ValueError("primary checkpoint identities changed")
  expected_settings = {
    "total_checkpoint_worker_cap": 6,
    "seed_workers_per_checkpoint": 1,
    "prediction_horizon": 50,
    "execute_steps": "horizon",
    "max_sim_seconds": 90,
    "guard_mode": "observe",
  }
  if any(primary.get("settings", {}).get(k) != v for k, v in expected_settings.items()):
    raise ValueError("primary execution contract changed")
  for index, group in enumerate(groups):
    task, step, _ = queue.worker.NEW_GROUPS[index]
    source = resume.SOURCE_ROOT if index == 0 else PRIMARY_ROOT
    if group.get("output_dir") != str(source / task / f"step{step}_N20_video20"):
      raise ValueError("primary output directory is outside its approved queue")
    if group.get("status") not in ("running", "complete", "failed", "interrupted"):
      raise ValueError("unexpected primary checkpoint status")
    count = group.get("completed_trials")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 20:
      raise ValueError("invalid primary completed trial count")
    if group.get("valid_trials") != count:
      raise ValueError("primary valid count differs from completed count")
  return primary


def build_retry_plan() -> dict:
  primary = read_primary()
  if any(g["status"] == "running" for g in primary["groups"][:4]):
    actual = queue.process_info(PRIMARY_CONTROLLER_PID)
    if actual is None or actual["state"] == "Z":
      raise RuntimeError("primary controller exited while non-USB workers remain")
  usb_groups = []
  for index in USB_INDICES:
    group = primary["groups"][index]
    if (
      group["status"] != "failed"
      or group["completed_trials"] != 0
      or group.get("owned_cleanup_confirmed") is not True
    ):
      raise RuntimeError(
        "USB retry requires failed zero-valid groups and confirmed cleanup"
      )
    resume.require_exited(group.get("worker_pid"))
    output = Path(group["output_dir"])
    if output.is_symlink():
      raise ValueError("prior USB output must not be a symlink")
    detail = queue.read_json(output / "group_status.json")
    record = queue.read_json(output / "run/run_manifest.json")
    if detail.get("pid") != group.get("worker_pid") or detail.get("status") != "failed":
      raise ValueError("prior USB worker identity changed")
    if any(
      detail.get(key) is not True or record.get(key) is not True
      for key in ("managed_server_stopped", "evaluation_process_stopped")
    ):
      raise RuntimeError("prior USB owned process cleanup is not confirmed")
    for key in ("server_pid", "evaluation_pid"):
      resume.require_exited(record.get(key))
    if any(
      record.get(key) != value
      for key, value in {
        "num_trials": 20,
        "seed_start": 0,
        "execute_steps": "horizon",
        "max_sim_seconds": 90,
        "guard_mode": "observe",
      }.items()
    ):
      raise ValueError("prior USB run contract changed")
    manifest = queue.read_json(output / "deployment/deployment_manifest.json")
    if (
      manifest.get("checkpoint_path") != group["checkpoint"]
      or manifest.get("task") != "usb-insert"
    ):
      raise ValueError("prior USB deployment checkpoint changed")
    summary_path = output / "run/evaluation/summary.json"
    summary = queue.read_json(summary_path)
    rows = summary.get("results", summary.get("trials", []))
    if (
      not isinstance(rows, list)
      or summary.get("valid_trials") != 0
      or any(row.get("valid_trial") is True for row in rows)
    ):
      raise ValueError("prior USB valid trials exist; refusing to repeat seeds")
    port = queue.FIRST_PORT + index
    if not queue.port_free(port):
      raise RuntimeError(f"USB retry service port is occupied: {port}")
    usb_groups.append(
      {
        "group_index": index,
        "task": "usb-insert",
        "step": group["step"],
        "checkpoint": group["checkpoint"],
        "prior_failed_output_dir": str(output),
        "prior_failed_summary_sha256": hashlib.sha256(
          summary_path.read_bytes()
        ).hexdigest(),
        "prior_failure_errors": detail.get("errors", []),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "seed_start": 0,
        "num_trials": 20,
        "pending_seeds": list(range(20)),
        "preserved_completed_trials": 0,
        "preserved_diagnostic_successes": 0,
        "port": port,
        "endpoint": f"ws://127.0.0.1:{port}",
      }
    )
  return {
    "primary_job_root": str(PRIMARY_ROOT),
    "primary_controller_pid": PRIMARY_CONTROLLER_PID,
    "created_utc": queue.now(),
    "prior_results_modified": False,
    "primary_workers_signalled": False,
    "total_trials": 120,
    "retry_trials_planned": 40,
    "usb_groups": usb_groups,
  }


def refresh(status: dict, job_root: Path) -> None:
  primary = read_primary()
  status["groups"][:4] = copy.deepcopy(primary["groups"][:4])
  for group in status["groups"][:4]:
    group["queue_owner"] = "primary_read_only"
  for group in status["groups"][4:]:
    output = Path(group["output_dir"])
    detail_path = output / "group_status.json"
    if detail_path.is_file():
      group["worker_status"] = queue.read_json(detail_path)
      if group["status"] == "running":
        group["stage"] = group["worker_status"].get("stage")
    summary_path = output / "run/evaluation/summary.json"
    if summary_path.is_file():
      try:
        summary = queue.read_json(summary_path)
      except (OSError, ValueError):
        summary = None
      if summary is not None:
        rows = summary.get("results", summary.get("trials", []))
        if (
          not isinstance(rows, list)
          or len(rows) > 20
          or [r.get("seed") for r in rows] != list(range(len(rows)))
        ):
          raise ValueError("new USB trials must be a zero-based, nonoverlapping prefix")
        count = sum(
          r.get("valid_trial") is True and r.get("returncode") == 0 for r in rows
        )
        if summary.get("valid_trials") != count:
          raise ValueError("new USB native validity count differs from results")
        group.update(
          completed_trials=count,
          valid_trials=count,
          fresh_completed_trials=count,
          diagnostic_successes=sum(
            r.get("success") is True for r in rows if r.get("valid_trial") is True
          ),
          native_attempted_trials=len(rows),
        )
    if group["status"] == "complete" and (
      group.get("valid_trials") != 20 or group.get("native_attempted_trials") != 20
    ):
      group["status"] = "failed"
      status["usb_errors"].append(
        f"usb-insert step{group['step']}: incomplete valid seed results"
      )
  primary_errors = [
    error
    for error in primary.get("errors", [])
    if not str(error).startswith(("usb-insert step20000:", "usb-insert step24999:"))
  ]
  for group in status["groups"][:4]:
    if group["status"] in ("failed", "interrupted"):
      primary_errors.append(
        f"primary {group['task']} step{group['step']}: {group['status']}"
      )
    elif group["status"] == "complete" and group["valid_trials"] != 20:
      group["status"] = "failed"
      primary_errors.append(
        f"primary {group['task']} step{group['step']}: incomplete valid results"
      )
  active_primary = sum(g["status"] == "running" for g in status["groups"][:4])
  active_usb = sum(g["status"] == "running" for g in status["groups"][4:])
  if active_primary + active_usb > queue.CAPACITY:
    raise RuntimeError("combined primary and USB retry capacity exceeded")
  status.update(
    errors=list(dict.fromkeys(primary_errors + status["usb_errors"])),
    completed_trials=sum(g.get("completed_trials", 0) for g in status["groups"]),
    valid_trials=sum(g.get("valid_trials", 0) for g in status["groups"]),
    completed_checkpoint_count=sum(g["status"] == "complete" for g in status["groups"]),
    active_primary_groups=active_primary,
    active_usb_retry_groups=active_usb,
    active_new_groups=active_primary + active_usb,
    retry_completed_trials=sum(
      g.get("completed_trials", 0) for g in status["groups"][4:]
    ),
    primary_status=primary["status"],
    primary_updated_utc=primary.get("updated_utc"),
    updated_utc=queue.now(),
  )
  queue.write_json(job_root / "status.json", status)


def run(job_root: Path, simulation_root: Path, openpi_root: Path) -> int:
  queue.validate_configuration(job_root, simulation_root, openpi_root)
  plan = queue.read_json(job_root / "usb_retry_plan.json")
  status = {
    "status": "running",
    "stage": "parallel_evaluations",
    "pid": os.getpid(),
    "started_utc": queue.now(),
    "simulation_root": str(simulation_root),
    "openpi_root": str(openpi_root),
    "primary_job_root": str(PRIMARY_ROOT),
    "total_trials": 120,
    "retry_trials_planned": 40,
    "settings": {
      "total_checkpoint_worker_cap": queue.CAPACITY,
      "seed_workers_per_checkpoint": 1,
      "prediction_horizon": 50,
      "execute_steps": "horizon",
      "max_sim_seconds": 90,
      "guard_mode": "observe",
      "gpu_device": 0,
      "hardware_rendering": True,
      "diagnostic_only": True,
      "formal_metrics_valid": False,
      "training_started": False,
    },
    "groups": [None] * 4 + copy.deepcopy(plan["usb_groups"]),
    "usb_errors": [],
    "errors": [],
  }
  for group in status["groups"][4:]:
    group.update(
      status="queued",
      completed_trials=0,
      valid_trials=0,
      fresh_completed_trials=0,
      queue_owner="usb_retry",
      output_dir=str(job_root / "usb-insert" / f"step{group['step']}_N20_video20"),
    )
  processes = {}

  def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGINT, interrupt)
  signal.signal(signal.SIGTERM, interrupt)
  with (queue.worker.EVALUATION_PARENT / LOCK_NAME).open("a+") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      checked = build_retry_plan()
      for key in ("primary_job_root", "primary_controller_pid", "usb_groups"):
        if plan.get(key) != checked[key]:
          raise ValueError("USB retry plan no longer matches prior failed outputs")
      while True:
        for index, process in list(processes.items()):
          code = process.poll()
          if code is None:
            continue
          group = status["groups"][index]
          if not queue.finalize_worker(group, code):
            raise RuntimeError(f"owned USB cleanup unconfirmed for step{group['step']}")
          if group["status"] == "failed":
            status["usb_errors"].append(
              f"usb-insert step{group['step']}: worker exit {code}"
            )
          del processes[index]
        refresh(status, job_root)
        active_primary = status["active_primary_groups"]
        free = max(0, queue.CAPACITY - active_primary - len(processes))
        queued = [g for g in status["groups"][4:] if g["status"] == "queued"]
        for group in queued[:free]:
          processes[group["group_index"]] = resume.launch_group(
            group, job_root, simulation_root, openpi_root
          )
        refresh(status, job_root)
        if not processes and all(
          g["status"] in ("complete", "failed", "interrupted") for g in status["groups"]
        ):
          status.update(
            status="failed" if status["errors"] else "complete",
            stage="complete",
            finished_utc=queue.now(),
          )
          refresh(status, job_root)
          return int(bool(status["errors"]))
        time.sleep(15)
    except BaseException as error:
      # These are exclusively the two children created by this retry controller.
      for process in processes.values():
        if process.poll() is None:
          process.terminate()
      for index, process in processes.items():
        try:
          queue.finalize_worker(status["groups"][index], process.wait(timeout=90))
        except (subprocess.TimeoutExpired, OSError, ValueError) as cleanup_error:
          status["usb_errors"].append(
            f"USB worker {process.pid} cleanup: {cleanup_error}"
          )
      status.update(
        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        stage="queue_error",
        finished_utc=queue.now(),
      )
      status["usb_errors"].append(f"{type(error).__name__}: {error}")
      try:
        refresh(status, job_root)
      except (OSError, ValueError, RuntimeError) as refresh_error:
        status["errors"] = status["usb_errors"] + [f"status refresh: {refresh_error}"]
        queue.write_json(job_root / "status.json", status)
      raise


def start(job_root: Path, simulation_root: Path, openpi_root: Path) -> int:
  queue.validate_configuration(job_root, simulation_root, openpi_root)
  if job_root.exists():
    raise FileExistsError(f"refusing to overwrite {job_root}")
  plan = build_retry_plan()
  if not queue.TOKENIZER_SOURCE.is_file():
    raise FileNotFoundError(queue.TOKENIZER_SOURCE)
  job_root.mkdir(exist_ok=False)
  cache = job_root / ".cache/openpi/big_vision"
  cache.mkdir(parents=True)
  shutil.copyfile(queue.TOKENIZER_SOURCE, cache / queue.TOKENIZER_SOURCE.name)
  for name in ("preflight_report.json", "snapshot_record.json"):
    shutil.copyfile(simulation_root.parent / name, job_root / name)
  queue.write_json(job_root / "usb_retry_plan.json", plan)
  queue.write_json(
    job_root / "status.json",
    {"status": "starting", "stage": "launching", "started_utc": queue.now()},
  )
  with (job_root / "queue.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(queue.CONTROLLER_PYTHON),
        str(Path(__file__).absolute()),
        "--job-root",
        str(job_root),
        "--simulation-root",
        str(simulation_root),
        "--openpi-root",
        str(openpi_root),
      ],
      cwd=simulation_root,
      env=queue.environment(job_root, simulation_root, openpi_root),
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  queue.write_json(
    job_root / "queue_process.json",
    {"pid": process.pid, "started_utc": queue.now(), "job_root": str(job_root)},
  )
  print(
    json.dumps(
      {
        "pid": process.pid,
        "job_root": str(job_root),
        "status": str(job_root / "status.json"),
        "retry_trials_planned": 40,
      },
      indent=2,
    )
  )
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  parser.add_argument("--simulation-root", type=Path, required=True)
  parser.add_argument("--openpi-root", type=Path, required=True)
  parser.add_argument("--start", action="store_true")
  args = parser.parse_args()
  paths = [
    p.absolute() for p in (args.job_root, args.simulation_root, args.openpi_root)
  ]
  return start(*paths) if args.start else run(*paths)


if __name__ == "__main__":
  raise SystemExit(main())
