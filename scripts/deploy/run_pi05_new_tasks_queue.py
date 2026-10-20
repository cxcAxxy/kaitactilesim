#!/usr/bin/env python3
"""Queue six frozen H50 checkpoint evaluations under a shared six-group cap.

The existing handoff owns four reserved slots until its transition has finished.
This controller only reads that job; it never signals or rewrites old processes.
Each newly launched group owns one model service, running its seeds serially.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Prevent a privileged interpreter from adding caches to immutable source copies.
sys.dont_write_bytecode = True
worker = importlib.import_module("run_pi05_checkpoint_worker")

EXISTING_ROOT = (
  worker.EVALUATION_PARENT / "pi05_h50_k50_parallel_20261013_173007_frozen_p4"
)
TOKENIZER_SOURCE = (
  worker.EVALUATION_PARENT
  / "pi05_h50_k50_serial_20261013_111821_frozen_r2"
  / ".cache/openpi/big_vision/paligemma_tokenizer.model"
)
CONTROLLER_PYTHON = Path("/usr/local/bin/python3.11")
CAPACITY = 6
EXISTING_GROUP_COUNT = len(worker.GROUPS)
FIRST_PORT = 18800


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def write_json(path: Path, value: dict) -> None:
  fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  temporary = Path(name)
  try:
    with os.fdopen(fd, "w") as stream:
      json.dump(value, stream, ensure_ascii=False, indent=2)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


def process_info(pid: int) -> dict | None:
  try:
    raw = Path(f"/proc/{pid}/stat").read_text()
  except FileNotFoundError:
    return None
  fields = raw[raw.rindex(")") + 2 :].split()
  return {
    "state": fields[0],
    "ppid": int(fields[1]),
    "starttime": int(fields[19]),
    "exit_status": int(fields[49]),
  }


def port_free(port: int) -> bool:
  """Probe service availability without treating closed TIME_WAIT sockets as live."""
  try:
    with socket.socket() as listener:
      # Match the reusable IPv4 listener used by the asyncio model service.
      # This permits stale TCP TIME_WAIT records, not an existing listener.
      listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      listener.bind(("127.0.0.1", port))
    return True
  except OSError:
    return False


def old_reservations(existing_root: Path) -> tuple[int, dict]:
  """Release old slots only with independent cleanup and worker-exit evidence."""
  evidence: dict = {"job_root": str(existing_root), "reserved_slots": CAPACITY}
  try:
    old = read_json(existing_root / "status.json")
    evidence.update(status=old.get("status"), stage=old.get("stage"))
    if old.get("stage") not in ("parallel_evaluations", "complete") or old.get(
      "status"
    ) not in ("running", "complete"):
      evidence["reason"] = (
        "old handoff or unknown/failed coordinator reserves all slots"
      )
      return CAPACITY, evidence
    handoff = old.get("handoff", {})
    if any(
      handoff.get(key) is not True
      for key in (
        "managed_server_stopped",
        "evaluation_process_stopped",
        "no_remaining_group_started_by_serial",
      )
    ):
      evidence["reason"] = "old serial handoff cleanup is not confirmed"
      return CAPACITY, evidence
    groups = old.get("groups", [])[2:]
    expected = [
      (task, step) for task in ("usb-unplug", "bulb-unscrew") for step in (20000, 24999)
    ]
    if (
      len(groups) != EXISTING_GROUP_COUNT
      or [(group.get("task"), group.get("step")) for group in groups] != expected
    ):
      raise ValueError("unexpected old checkpoint groups")
    released = []
    for index, group in enumerate(groups):
      output = existing_root / group["task"] / f"step{group['step']}_N20_video20"
      proof = {"task": group["task"], "step": group["step"], "port": 18784 + index}
      if Path(group.get("output_dir", "")) != output:
        raise ValueError("old output directory is not inside the owned job")
      proof["status"] = group.get("status")
      pid = group.get("worker_pid")
      proof["worker_pid"] = pid
      if group.get("status") != "complete" or not isinstance(pid, int) or pid <= 1:
        proof["released"] = False
        released.append(proof)
        continue
      detail = read_json(output / "group_status.json")
      record = read_json(output / "run/run_manifest.json")
      cleaned = (
        detail.get("pid") == pid
        and detail.get("status") == "complete"
        and detail.get("evaluation_returncode") == 0
        and all(
          detail.get(key) is True and record.get(key) is True
          for key in ("managed_server_stopped", "evaluation_process_stopped")
        )
      )
      actual = process_info(pid)
      exited = actual is None or (
        actual["state"] == "Z" and os.waitstatus_to_exitcode(actual["exit_status"]) == 0
      )
      proof.update(
        cleaned=cleaned,
        worker_process=actual,
        worker_exited=exited,
        port_free=port_free(18784 + index),
      )
      proof["released"] = cleaned and exited and proof["port_free"]
      released.append(proof)
    reserved = EXISTING_GROUP_COUNT - sum(bool(proof["released"]) for proof in released)
    evidence.update(reserved_slots=reserved, groups=released)
    return reserved, evidence
  except (OSError, ValueError, KeyError, TypeError) as error:
    evidence["reason"] = f"fail-safe reservation: {type(error).__name__}: {error}"
    return CAPACITY, evidence


def environment(job_root: Path, simulation_root: Path, openpi_root: Path) -> dict:
  result = dict(os.environ)
  result.update(
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUNBUFFERED="1",
    PYTHONPATH=os.pathsep.join(
      (
        str(simulation_root / "src"),
        str(openpi_root / "src"),
        str(openpi_root / "packages/openpi-client/src"),
      )
    ),
    CUDA_VISIBLE_DEVICES="0",
    MUJOCO_EGL_DEVICE_ID="0",
    MUJOCO_GL="egl",
    PYOPENGL_PLATFORM="egl",
    KAIHAND_RENDER_BACKEND="hardware",
    LIBGL_ALWAYS_SOFTWARE="0",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    LP_NUM_THREADS="1",
    __EGL_VENDOR_LIBRARY_FILENAMES="/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json",
    OPENPI_DATA_HOME=str(job_root / ".cache/openpi"),
    HF_HOME="/nas/chenxianchi/datasets/sim/_cache/huggingface",
    MPLCONFIGDIR=str(job_root / ".cache/matplotlib"),
    XLA_PYTHON_CLIENT_PREALLOCATE="false",
  )
  result.pop("JAX_PLATFORMS", None)
  result.pop("JAX_PLATFORM_NAME", None)
  return result


def validate_configuration(
  job_root: Path, simulation_root: Path, openpi_root: Path
) -> None:
  if job_root.parent != worker.EVALUATION_PARENT or not job_root.name.startswith(
    "pi05_h50_k50_new_tasks_"
  ):
    raise ValueError("new job must be directly inside the scoped NAS evaluation parent")
  if job_root.is_symlink():
    raise ValueError("job directory must not be a symlink")
  if (
    simulation_root.parent != openpi_root.parent
    or simulation_root.parent.parent != worker.SNAPSHOT_PARENT
    or simulation_root.name != "kaitactilesim"
    or openpi_root.name != "openpi"
    or simulation_root.is_symlink()
    or openpi_root.is_symlink()
  ):
    raise ValueError("source roots must be sibling frozen copies in runtime_snapshots")
  for _, _, checkpoint in worker.NEW_GROUPS:
    worker.validate_assets(simulation_root, openpi_root, checkpoint)
  for script in (Path(__file__), Path(worker.__file__)):
    if not script.resolve().is_relative_to(simulation_root / "scripts"):
      raise ValueError("controller and worker must run from the new frozen bundle")
  report = read_json(simulation_root.parent / "preflight_report.json")
  rows = report.get("checkpoints", [])
  if (
    report.get("all_six_valid") is not True
    or report.get("simulation_root") != str(simulation_root)
    or report.get("model_project_root") != str(openpi_root)
    or [(row.get("task"), row.get("step"), row.get("checkpoint")) for row in rows]
    != [(task, step, str(checkpoint)) for task, step, checkpoint in worker.NEW_GROUPS]
    or any(
      row.get("valid") is not True
      or row.get("prediction_horizon") != 50
      or row.get("execute_steps") != 50
      or row.get("required_action_interpolation") != "linear"
      or row.get("binding_status") != "bound"
      for row in rows
    )
  ):
    raise ValueError("all six new checkpoints must pass frozen-source CPU preflight")


def start(job_root: Path, simulation_root: Path, openpi_root: Path) -> int:
  validate_configuration(job_root, simulation_root, openpi_root)
  if not TOKENIZER_SOURCE.is_file():
    raise FileNotFoundError(TOKENIZER_SOURCE)
  if not (EXISTING_ROOT / "status.json").is_file():
    raise FileNotFoundError(EXISTING_ROOT / "status.json")
  job_root.mkdir(exist_ok=False)
  cache = job_root / ".cache/openpi/big_vision"
  cache.mkdir(parents=True)
  shutil.copyfile(TOKENIZER_SOURCE, cache / TOKENIZER_SOURCE.name)
  for name in ("preflight_report.json", "snapshot_record.json"):
    shutil.copyfile(simulation_root.parent / name, job_root / name)
  write_json(
    job_root / "status.json",
    {"status": "starting", "stage": "launching", "started_utc": now()},
  )
  with (job_root / "queue.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(CONTROLLER_PYTHON),
        str(Path(__file__).absolute()),
        "--job-root",
        str(job_root),
        "--simulation-root",
        str(simulation_root),
        "--openpi-root",
        str(openpi_root),
      ],
      cwd=simulation_root,
      env=environment(job_root, simulation_root, openpi_root),
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  write_json(
    job_root / "queue_process.json",
    {"pid": process.pid, "started_utc": now(), "job_root": str(job_root)},
  )
  print(
    json.dumps(
      {
        "pid": process.pid,
        "job_root": str(job_root),
        "status": str(job_root / "status.json"),
      },
      indent=2,
    )
  )
  return 0


def refresh(status: dict, job_root: Path) -> None:
  for group in status["groups"]:
    output = Path(group["output_dir"])
    for filename, kind in (
      ("group_status.json", "worker"),
      ("run/evaluation/summary.json", "batch"),
    ):
      path = output / filename
      if not path.is_file():
        continue
      try:
        content = read_json(path)
        if kind == "worker":
          # Keep controller-owned identity and final exit status authoritative.
          group["worker_status"] = content
          if group["status"] not in ("complete", "failed", "interrupted"):
            group["stage"] = content.get("stage")
        else:
          group.update(
            completed_trials=len(content.get("results", content.get("trials", []))),
            valid_trials=content.get("valid_trials", 0),
            diagnostic_successes=content.get("successes", 0),
          )
      except (OSError, ValueError, TypeError):
        continue
  status.update(
    completed_trials=sum(
      group.get("completed_trials", 0) for group in status["groups"]
    ),
    valid_trials=sum(group.get("valid_trials", 0) for group in status["groups"]),
    completed_checkpoint_count=sum(
      group["status"] == "complete" for group in status["groups"]
    ),
    active_new_groups=sum(group["status"] == "running" for group in status["groups"]),
    updated_utc=now(),
  )
  write_json(job_root / "status.json", status)


def finalize_worker(group: dict, returncode: int) -> bool:
  """A terminated group releases capacity only if its owned children are gone."""
  output = Path(group["output_dir"])
  detail = (
    read_json(output / "group_status.json")
    if (output / "group_status.json").is_file()
    else {}
  )
  record = (
    read_json(output / "run/run_manifest.json")
    if (output / "run/run_manifest.json").is_file()
    else {}
  )
  clean = detail.get("stage") in ("preflight", "prepare", "verify") or all(
    record.get(key) is True
    for key in ("managed_server_stopped", "evaluation_process_stopped")
  )
  clean = bool(clean and port_free(group["port"]))
  complete = returncode == 0 and detail.get("status") == "complete" and clean
  group.update(
    status="complete" if complete else "failed",
    worker_returncode=returncode,
    owned_cleanup_confirmed=clean,
    finished_utc=now(),
    worker_status=detail,
  )
  return clean


def launch_group(
  group: dict, job_root: Path, simulation_root: Path, openpi_root: Path
) -> subprocess.Popen:
  if not port_free(group["port"]):
    raise RuntimeError(f"new group port is already occupied: {group['port']}")
  with (job_root / f"group_{group['group_index']}.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(simulation_root / ".pixi/envs/default/bin/python"),
        str(Path(worker.__file__).absolute()),
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
      ],
      cwd=simulation_root,
      env=environment(job_root, simulation_root, openpi_root),
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  group.update(
    status="running",
    stage="launching",
    worker_pid=process.pid,
    worker_identity=process_info(process.pid),
    started_utc=now(),
  )
  print(
    f"[{now()}] LAUNCH {group['task']} step{group['step']} pid={process.pid} {group['endpoint']}",
    flush=True,
  )
  return process


def run(job_root: Path, simulation_root: Path, openpi_root: Path) -> int:
  validate_configuration(job_root, simulation_root, openpi_root)
  status = {
    "status": "running",
    "stage": "waiting_for_capacity",
    "pid": os.getpid(),
    "started_utc": now(),
    "simulation_root": str(simulation_root),
    "openpi_root": str(openpi_root),
    "existing_job_root": str(EXISTING_ROOT),
    "total_trials": 120,
    "total_checkpoint_count": 6,
    "settings": {
      "total_checkpoint_worker_cap": CAPACITY,
      "seed_workers_per_checkpoint": 1,
      "independent_model_services": True,
      "prediction_horizon": 50,
      "execute_steps": "horizon",
      "num_trials": 20,
      "video_count": 20,
      "seed_start": 0,
      "guard_mode": "observe",
      "max_sim_seconds": 90,
      "gpu_device": 0,
      "hardware_rendering": True,
      "diagnostic_only": True,
      "formal_metrics_valid": False,
      "training_started": False,
    },
    "groups": [
      {
        "group_index": index,
        "task": task,
        "step": step,
        "checkpoint": str(checkpoint),
        "output_dir": str(job_root / task / f"step{step}_N20_video20"),
        "endpoint": f"ws://127.0.0.1:{FIRST_PORT + index}",
        "port": FIRST_PORT + index,
        "status": "queued",
      }
      for index, (task, step, checkpoint) in enumerate(worker.NEW_GROUPS)
    ],
    "errors": [],
  }
  processes: dict[int, subprocess.Popen] = {}

  def interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGINT, interrupt)
  signal.signal(signal.SIGTERM, interrupt)
  with (worker.EVALUATION_PARENT / ".pi05_h50_new_tasks_scheduler.lock").open(
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
          if not finalize_worker(group, code):
            raise RuntimeError(
              f"owned cleanup unconfirmed for {group['task']} step{group['step']}; refusing additional launches"
            )
          if group["status"] == "failed":
            status["errors"].append(
              f"{group['task']} step{group['step']}: worker exit {code}"
            )
          del processes[index]
        reserved, evidence = old_reservations(EXISTING_ROOT)
        status["existing_capacity_evidence"] = evidence
        free = max(0, CAPACITY - reserved - len(processes))
        queued = [group for group in status["groups"] if group["status"] == "queued"]
        for group in queued[:free]:
          processes[group["group_index"]] = launch_group(
            group, job_root, simulation_root, openpi_root
          )
        status["stage"] = (
          "parallel_evaluations" if processes else "waiting_for_capacity"
        )
        refresh(status, job_root)
        if not processes and all(
          group["status"] in ("complete", "failed") for group in status["groups"]
        ):
          status.update(
            status="failed" if status["errors"] else "complete",
            stage="complete",
            finished_utc=now(),
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
          code = process.wait(timeout=90)
          finalize_worker(status["groups"][index], code)
        except (subprocess.TimeoutExpired, OSError, ValueError) as cleanup_error:
          status["errors"].append(f"worker {process.pid} cleanup: {cleanup_error}")
      status.update(
        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        stage="queue_error",
        finished_utc=now(),
      )
      status["errors"].append(f"{type(error).__name__}: {error}")
      refresh(status, job_root)
      raise


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  parser.add_argument("--simulation-root", type=Path, required=True)
  parser.add_argument("--openpi-root", type=Path, required=True)
  parser.add_argument("--start", action="store_true")
  args = parser.parse_args()
  paths = [
    path.absolute() for path in (args.job_root, args.simulation_root, args.openpi_root)
  ]
  return start(*paths) if args.start else run(*paths)


if __name__ == "__main__":
  raise SystemExit(main())
