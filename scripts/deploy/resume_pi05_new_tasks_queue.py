#!/usr/bin/env python3
"""Run only missing seeds in a fresh job, preserving the failed queue's results."""

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
SOURCE_ROOT = (
  queue.worker.EVALUATION_PARENT / "pi05_h50_k50_new_tasks_20261013_181900_frozen_q6"
)


def require_exited(pid: int | None) -> None:
  if pid is None:
    return
  if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
    raise ValueError("invalid recorded process identity")
  actual = queue.process_info(pid)
  if actual is not None and actual["state"] != "Z":
    raise RuntimeError(f"prior process {pid} is still alive; refusing duplicate launch")


def validate_rows(rows: list[dict], task: str, manifest: dict, output: Path) -> None:
  if not isinstance(rows, list) or len(rows) > 20:
    raise ValueError("invalid prior trial rows")
  for seed, row in enumerate(rows):
    if (
      row.get("seed") != seed
      or row.get("valid_trial") is not True
      or row.get("returncode") != 0
      or not isinstance(row.get("success"), bool)
      or any(row.get(key) for key in ("error", "record_error", "raw_record_error"))
      or row.get("identity_errors")
      or row.get("summary") != f"seed_{seed:03d}/summary.json"
    ):
      raise ValueError("prior results are not a valid, completed seed prefix")
    detail_path = output / "run/evaluation" / row["summary"]
    if detail_path.is_symlink():
      raise ValueError("prior trial summary must not be a symlink")
    detail = queue.read_json(detail_path)
    expected = {
      "task": task,
      "seed": seed,
      "prediction_horizon": 50,
      "execute_steps": 50,
      "max_sim_seconds": 90,
      "guard_mode": "observe",
      "action_interpolation": "linear",
      "physics_valid": True,
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "deployment_id": manifest["deployment_id"],
      "status": "success" if row["success"] else "task_not_completed",
    }
    if any(detail.get(key) != value for key, value in expected.items()):
      raise ValueError(
        f"prior trial identity or execution contract changed: {detail_path}"
      )
    integrity = detail.get("deployment_integrity", {})
    if integrity.get("status") != "verified" or integrity.get("errors"):
      raise ValueError("prior trial deployment integrity is not verified")


def build_resume_plan(source_root: Path) -> dict:
  if source_root != SOURCE_ROOT or source_root.is_symlink():
    raise ValueError("resume is restricted to the explicitly approved failed queue")
  old = queue.read_json(source_root / "status.json")
  if old.get("status") != "failed" or old.get("active_new_groups") != 0:
    raise RuntimeError("prior queue is not stopped")
  require_exited(old.get("pid"))
  groups = old.get("groups", [])
  if [(g.get("task"), g.get("step"), g.get("checkpoint")) for g in groups] != [
    (task, step, str(checkpoint)) for task, step, checkpoint in queue.worker.NEW_GROUPS
  ]:
    raise ValueError("prior checkpoint identities changed")
  preserved = []
  for index, group in enumerate(groups):
    task, step, checkpoint = queue.worker.NEW_GROUPS[index]
    output = source_root / task / f"step{step}_N20_video20"
    if group.get("output_dir") != str(output) or output.is_symlink():
      raise ValueError("prior output is outside the approved job")
    require_exited(group.get("worker_pid"))
    rows = []
    fingerprint = None
    if group.get("status") == "queued":
      if output.exists():
        raise ValueError("queued prior group unexpectedly has outputs")
    else:
      record = queue.read_json(output / "run/run_manifest.json")
      for key in ("server_pid", "evaluation_pid"):
        require_exited(record.get(key))
      if any(
        record.get(key) is not True
        for key in ("managed_server_stopped", "evaluation_process_stopped")
      ):
        raise RuntimeError("prior owned process cleanup is not confirmed")
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
        raise ValueError("prior evaluation settings changed")
      manifest = queue.read_json(output / "deployment/deployment_manifest.json")
      if (
        manifest.get("checkpoint_path") != str(checkpoint)
        or manifest.get("task") != task
      ):
        raise ValueError("prior deployment checkpoint changed")
      summary_path = output / "run/evaluation/summary.json"
      summary = queue.read_json(summary_path)
      rows = summary.get("results", summary.get("trials", []))
      validate_rows(rows, task, manifest, output)
      if summary.get("valid_trials") != len(rows):
        raise ValueError("prior aggregate validity count does not match rows")
      fingerprint = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    preserved.append(
      {
        "group_index": index,
        "task": task,
        "step": step,
        "checkpoint": str(checkpoint),
        "preserved_output_dir": str(output),
        "preserved_summary_sha256": fingerprint,
        "preserved_completed_trials": len(rows),
        "preserved_diagnostic_successes": sum(row["success"] for row in rows),
        "preserved_seeds": list(range(len(rows))),
        "pending_seeds": list(range(len(rows), 20)),
        "seed_start": len(rows),
        "num_trials": 20 - len(rows),
      }
    )
  return {
    "source_job_root": str(source_root),
    "created_utc": queue.now(),
    "prior_results_modified": False,
    "total_trials": 120,
    "preserved_trials": sum(g["preserved_completed_trials"] for g in preserved),
    "remaining_trials": sum(g["num_trials"] for g in preserved),
    "groups": preserved,
  }


def refresh(status: dict, job_root: Path) -> None:
  for group in status["groups"]:
    fresh_count = fresh_successes = 0
    output = Path(group["output_dir"])
    if group["num_trials"]:
      detail = output / "group_status.json"
      if detail.is_file():
        group["worker_status"] = queue.read_json(detail)
        if group["status"] == "running":
          group["stage"] = group["worker_status"].get("stage")
      summary_path = output / "run/evaluation/summary.json"
      if summary_path.is_file():
        try:
          summary = queue.read_json(summary_path)
        except (OSError, ValueError):
          summary = None  # Native batch writers may currently be replacing this file.
        if summary is not None:
          rows = summary.get("results", summary.get("trials", []))
          if [r.get("seed") for r in rows] != group["pending_seeds"][: len(rows)]:
            raise ValueError(
              "fresh trial seeds overlap preserved results or exceed plan"
            )
          fresh_count = sum(
            r.get("valid_trial") is True and r.get("returncode") == 0 for r in rows
          )
          fresh_successes = sum(
            r.get("success") is True for r in rows if r.get("valid_trial") is True
          )
          group["fresh_completed_trials"] = fresh_count
          group["fresh_diagnostic_successes"] = fresh_successes
    group["completed_trials"] = group["preserved_completed_trials"] + group.get(
      "fresh_completed_trials", fresh_count
    )
    group["valid_trials"] = group["completed_trials"]
    group["diagnostic_successes"] = group["preserved_diagnostic_successes"] + group.get(
      "fresh_diagnostic_successes", fresh_successes
    )
  status.update(
    completed_trials=sum(g["completed_trials"] for g in status["groups"]),
    valid_trials=sum(g["valid_trials"] for g in status["groups"]),
    resumed_completed_trials=sum(
      g.get("fresh_completed_trials", 0) for g in status["groups"]
    ),
    active_new_groups=sum(g["status"] == "running" for g in status["groups"]),
    completed_checkpoint_count=sum(g["status"] == "complete" for g in status["groups"]),
    updated_utc=queue.now(),
  )
  queue.write_json(job_root / "status.json", status)


def launch_group(group: dict, job_root: Path, simulation_root: Path, openpi_root: Path):
  if not group["num_trials"] or not queue.port_free(group["port"]):
    raise RuntimeError("refusing duplicate seed range or occupied service port")
  with (job_root / f"group_{group['group_index']}.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(simulation_root / ".pixi/envs/default/bin/python"),
        str(Path(queue.worker.__file__).absolute()),
        "--job-root",
        str(job_root),
        "--simulation-root",
        str(simulation_root),
        "--openpi-root",
        str(openpi_root),
        "--group-index",
        str(group["group_index"]),
        "--endpoint",
        group["endpoint"],
        "--new-tasks",
        "--seed-start",
        str(group["seed_start"]),
        "--num-trials",
        str(group["num_trials"]),
      ],
      cwd=simulation_root,
      env=queue.environment(job_root, simulation_root, openpi_root),
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  group.update(
    status="running",
    stage="launching",
    worker_pid=process.pid,
    worker_identity=queue.process_info(process.pid),
    started_utc=queue.now(),
  )
  print(
    f"[{queue.now()}] LAUNCH {group['task']} step{group['step']} seeds={group['pending_seeds']} pid={process.pid}",
    flush=True,
  )
  return process


def run(job_root: Path, simulation_root: Path, openpi_root: Path) -> int:
  queue.validate_configuration(job_root, simulation_root, openpi_root)
  plan = queue.read_json(job_root / "resume_plan.json")
  checked = build_resume_plan(SOURCE_ROOT)
  for key in (
    "source_job_root",
    "total_trials",
    "preserved_trials",
    "remaining_trials",
    "groups",
  ):
    if plan.get(key) != checked[key]:
      raise ValueError("resume plan no longer matches validated prior results")
  status = {
    "status": "running",
    "stage": "parallel_evaluations",
    "pid": os.getpid(),
    "started_utc": queue.now(),
    "simulation_root": str(simulation_root),
    "openpi_root": str(openpi_root),
    "total_trials": 120,
    "resumed_trials_planned": plan["remaining_trials"],
    "preserved_trials": plan["preserved_trials"],
    "source_job_root": plan["source_job_root"],
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
    "groups": copy.deepcopy(plan["groups"]),
    "errors": [],
  }
  for group in status["groups"]:
    group.update(
      status="queued" if group["num_trials"] else "complete",
      output_dir=str(job_root / group["task"] / f"step{group['step']}_N20_video20")
      if group["num_trials"]
      else group["preserved_output_dir"],
      endpoint=f"ws://127.0.0.1:{queue.FIRST_PORT + group['group_index']}",
      port=queue.FIRST_PORT + group["group_index"],
    )
  processes = {}

  def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGINT, interrupt)
  signal.signal(signal.SIGTERM, interrupt)
  with (queue.worker.EVALUATION_PARENT / ".pi05_h50_new_tasks_scheduler.lock").open(
    "a+"
  ) as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      while True:
        for index, process in list(processes.items()):
          code = process.poll()
          if code is None:
            continue
          group = status["groups"][index]
          if not queue.finalize_worker(group, code):
            raise RuntimeError(
              f"owned cleanup unconfirmed for {group['task']} step{group['step']}"
            )
          if group["status"] == "failed":
            status["errors"].append(
              f"{group['task']} step{group['step']}: worker exit {code}"
            )
          del processes[index]
        reserved, evidence = queue.old_reservations(queue.EXISTING_ROOT)
        status["existing_capacity_evidence"] = evidence
        free = max(0, queue.CAPACITY - reserved - len(processes))
        queued = [g for g in status["groups"] if g["status"] == "queued"]
        for group in queued[:free]:
          processes[group["group_index"]] = launch_group(
            group, job_root, simulation_root, openpi_root
          )
        refresh(status, job_root)
        if not processes and all(
          g["status"] in ("complete", "failed") for g in status["groups"]
        ):
          for group in status["groups"]:
            if group["status"] == "complete" and group["completed_trials"] != 20:
              group["status"] = "failed"
              status["errors"].append(
                f"{group['task']} step{group['step']}: incomplete valid seed results"
              )
          status.update(
            status="failed" if status["errors"] else "complete",
            stage="complete",
            finished_utc=queue.now(),
          )
          refresh(status, job_root)
          return int(bool(status["errors"]))
        time.sleep(15)
    except BaseException as error:
      for process in processes.values():
        if process.poll() is None:
          process.terminate()
      for index, process in processes.items():
        try:
          queue.finalize_worker(status["groups"][index], process.wait(timeout=90))
        except (subprocess.TimeoutExpired, OSError, ValueError) as cleanup_error:
          status["errors"].append(f"worker {process.pid} cleanup: {cleanup_error}")
      status.update(
        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        stage="queue_error",
        finished_utc=queue.now(),
      )
      status["errors"].append(f"{type(error).__name__}: {error}")
      refresh(status, job_root)
      raise


def start(
  job_root: Path, simulation_root: Path, openpi_root: Path, source_root: Path
) -> int:
  queue.validate_configuration(job_root, simulation_root, openpi_root)
  if job_root.exists():
    raise FileExistsError(f"refusing to overwrite {job_root}")
  plan = build_resume_plan(source_root)
  for group in plan["groups"]:
    if group["num_trials"] and not queue.port_free(
      queue.FIRST_PORT + group["group_index"]
    ):
      raise RuntimeError("resume service port is occupied")
  if not queue.TOKENIZER_SOURCE.is_file():
    raise FileNotFoundError(queue.TOKENIZER_SOURCE)
  job_root.mkdir(exist_ok=False)
  cache = job_root / ".cache/openpi/big_vision"
  cache.mkdir(parents=True)
  shutil.copyfile(queue.TOKENIZER_SOURCE, cache / queue.TOKENIZER_SOURCE.name)
  for name in ("preflight_report.json", "snapshot_record.json"):
    shutil.copyfile(simulation_root.parent / name, job_root / name)
  queue.write_json(job_root / "resume_plan.json", plan)
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
        "preserved_trials": plan["preserved_trials"],
        "remaining_trials": plan["remaining_trials"],
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
  parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
  parser.add_argument("--start", action="store_true")
  args = parser.parse_args()
  paths = [
    p.absolute() for p in (args.job_root, args.simulation_root, args.openpi_root)
  ]
  return start(*paths, args.source_root.absolute()) if args.start else run(*paths)


if __name__ == "__main__":
  raise SystemExit(main())
