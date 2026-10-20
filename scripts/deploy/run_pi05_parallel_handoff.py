#!/usr/bin/env python3
"""Finish the current serial checkpoint, then run four independent groups."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

EVALUATION_PARENT = Path("/nas/chenxianchi/datasets/sim/_evaluations")
OLD_ROOT = EVALUATION_PARENT / "pi05_h50_k50_serial_20261013_111821_frozen_r2"
FROZEN_BUNDLE = Path(
  "/cpfs_infra/user/chenxianchi/runtime_snapshots/pi05_serial_20261013_111821"
)
SIM_ROOT = FROZEN_BUNDLE / "kaitactilesim"
OPENPI_ROOT = FROZEN_BUNDLE / "openpi"
SIM_PYTHON = SIM_ROOT / ".pixi/envs/default/bin/python"
CONTROLLER_PYTHON = Path("/usr/local/bin/python3.11")
MODEL_PYTHON = Path("/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python")
SERIAL_PID = 2690228
PORTS = (18784, 18785, 18786, 18787)


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"expected a JSON object: {path}")
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


def process_info(pid: int) -> dict:
  raw = Path(f"/proc/{pid}/stat").read_text()
  fields = raw[raw.rindex(")") + 2 :].split()
  return {
    "state": fields[0],
    "ppid": int(fields[1]),
    "starttime": int(fields[19]),
    "exit_status": int(fields[49]),
  }


def command_line(pid: int) -> list[str]:
  return Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")


def assert_same_process(pid: int, identity: dict) -> dict:
  actual = process_info(pid)
  if actual["starttime"] != identity["starttime"]:
    raise RuntimeError(f"refusing to act on a reused PID: {pid}")
  return actual


def validate_serial() -> tuple[dict, dict, dict]:
  old = read_json(OLD_ROOT / "status.json")
  if (
    old.get("pid") != SERIAL_PID
    or old.get("status") != "running"
    or old.get("task") != "whiteboard-wipe-horizontal"
    or old.get("checkpoint_step") != 24999
    or old.get("stage") != "evaluate"
    or len(old.get("groups", [])) != 6
    or old["groups"][0].get("status") != "complete"
    or any(g.get("status") != "pending" for g in old["groups"][2:])
  ):
    raise RuntimeError("serial job is not at the authorized whiteboard boundary")
  command = command_line(SERIAL_PID)
  script = str(SIM_ROOT / "scripts/deploy/run_pi05_checkpoint_serial.py")
  if script not in command or str(OLD_ROOT) not in command:
    raise RuntimeError("serial PID does not match the owned frozen job")
  info = process_info(SERIAL_PID)
  if info["state"] in ("Z", "T", "t"):
    raise RuntimeError("serial orchestrator is not running normally")
  run_root = Path(old["groups"][1]["output_dir"]) / "run"
  record = read_json(run_root / "run_manifest.json")
  if record.get("status") != "evaluating":
    raise RuntimeError("current whiteboard evaluator has not started")
  # The frozen API only persists evaluation_pid in its final cleanup record.
  # Resolve the still-running evaluator from the orchestrator's direct children.
  children = Path(f"/proc/{SERIAL_PID}/task/{SERIAL_PID}/children").read_text().split()
  candidates = []
  evaluator_script = str(SIM_ROOT / "scripts/evaluate/evaluate.py")
  for candidate in children:
    try:
      candidate_pid = int(candidate)
      candidate_command = command_line(candidate_pid)
      if (
        evaluator_script in candidate_command
        and str(run_root / "evaluation") in candidate_command
        and process_info(candidate_pid)["ppid"] == SERIAL_PID
      ):
        candidates.append(candidate_pid)
    except FileNotFoundError:
      continue
  if len(candidates) != 1:
    raise RuntimeError(f"expected one owned whiteboard evaluator, got {candidates}")
  child_pid = candidates[0]
  child_info = process_info(child_pid)
  if child_info["ppid"] != SERIAL_PID or str(
    run_root / "evaluation"
  ) not in command_line(child_pid):
    raise RuntimeError("evaluation PID does not belong to this whiteboard group")
  if (
    record.get("task") != "whiteboard-wipe-horizontal"
    or record.get("num_trials") != 20
    or record.get("video_count") != 20
    or record.get("execute_steps") != "horizon"
    or record.get("guard_mode") != "observe"
    or record.get("max_sim_seconds") != 90
  ):
    raise RuntimeError("current evaluation settings differ from the authorized plan")
  return old, {"pid": SERIAL_PID, **info}, {"pid": child_pid, **child_info}


def validated_batch(path: Path, group: dict) -> dict:
  batch = read_json(path)
  deployment = read_json(
    Path(group["output_dir"]) / "deployment/deployment_manifest.json"
  )
  rows = batch.get("results", [])
  if (
    batch.get("planned") != 20
    or batch.get("attempted") != 20
    or batch.get("valid_trials") != 20
    or len(rows) != 20
    or sorted(row.get("seed", -1) for row in rows) != list(range(20))
    or batch.get("task") != "whiteboard-wipe-horizontal"
    or batch.get("deployment_id") != deployment["deployment_id"]
    or batch.get("checkpoint_sha256") != deployment["checkpoint_sha256"]
    or any(
      row.get("returncode") != 0 or row.get("valid_trial") is not True for row in rows
    )
    or any(
      row.get("deployment_integrity", {}).get("status") != "verified" for row in rows
    )
  ):
    raise RuntimeError(f"whiteboard group is not fully and validly evaluated: {path}")
  return batch


def environment(job_root: Path) -> dict[str, str]:
  result = dict(os.environ)
  result.update(
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONUNBUFFERED="1",
    PYTHONPATH=os.pathsep.join(
      (
        str(SIM_ROOT / "src"),
        str(OPENPI_ROOT / "src"),
        str(OPENPI_ROOT / "packages/openpi-client/src"),
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


def assert_ports_free() -> None:
  for port in PORTS:
    with socket.socket() as listener:
      listener.bind(("127.0.0.1", port))


def start(job_root: Path) -> int:
  validate_serial()
  assert_ports_free()
  worker = Path(__file__).with_name("run_pi05_checkpoint_worker.py")
  for required in (SIM_PYTHON, CONTROLLER_PYTHON, MODEL_PYTHON, worker):
    if not required.is_file():
      raise FileNotFoundError(required)
  if job_root.parent != EVALUATION_PARENT or not job_root.name.startswith(
    "pi05_h50_k50_parallel_"
  ):
    raise ValueError("job-root must be a new parallel evaluation directory")
  for group in read_json(OLD_ROOT / "status.json")["groups"][2:]:
    if not (Path(group["checkpoint"]) / "_CHECKPOINT_METADATA").is_file():
      raise RuntimeError(f"checkpoint is not committed: {group['checkpoint']}")
  source = OLD_ROOT / ".cache/openpi/big_vision/paligemma_tokenizer.model"
  if not source.is_file():
    raise FileNotFoundError(source)
  job_root.mkdir(exist_ok=False)
  cache = job_root / ".cache/openpi/big_vision"
  cache.mkdir(parents=True)
  shutil.copyfile(source, cache / source.name)
  write_json(
    job_root / "status.json",
    {"status": "starting", "stage": "launching", "started_utc": now()},
  )
  with (job_root / "parallel.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(CONTROLLER_PYTHON),
        str(Path(__file__).absolute()),
        "--job-root",
        str(job_root),
      ],
      cwd=SIM_ROOT,
      env=environment(job_root),
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  (job_root / "parallel.pid").write_text(f"{process.pid}\n")
  print(
    json.dumps(
      {
        "pid": process.pid,
        "job_root": str(job_root),
        "log": str(job_root / "parallel.log"),
      },
      indent=2,
    )
  )
  return 0


def refresh(status: dict, job_root: Path) -> None:
  for group in status["groups"]:
    worker_status = Path(group["output_dir"]) / "group_status.json"
    if not group.get("preserved_results") and worker_status.is_file():
      try:
        group.update(read_json(worker_status))
      except (OSError, ValueError):
        pass
    summary = Path(group["output_dir"]) / "run/evaluation/summary.json"
    if summary.is_file():
      try:
        batch = read_json(summary)
        group.update(
          completed_trials=len(batch.get("results", [])),
          valid_trials=batch.get("valid_trials", 0),
          successes=batch.get("successes", 0),
        )
      except (OSError, ValueError):
        pass
  status["completed_trials"] = sum(
    group.get("completed_trials", 0) for group in status["groups"]
  )
  status["valid_trials"] = sum(
    group.get("valid_trials", 0) for group in status["groups"]
  )
  status["diagnostic_successes"] = sum(
    group.get("successes", 0) for group in status["groups"]
  )
  status["completed_checkpoint_count"] = sum(
    group.get("status") == "complete" for group in status["groups"]
  )
  status["updated_utc"] = now()
  write_json(job_root / "status.json", status)


def run(job_root: Path) -> int:
  if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise RuntimeError("handoff controller requires Python with Linux pidfd support")
  old, serial, evaluator = validate_serial()
  groups = copy.deepcopy(old["groups"])
  for group in groups[:2]:
    group["preserved_results"] = True
  for index, group in enumerate(groups[2:]):
    group.update(
      status="waiting_for_whiteboard",
      endpoint=f"ws://127.0.0.1:{PORTS[index]}",
      output_dir=str(job_root / group["task"] / f"step{group['step']}_N20_video20"),
    )
  status = {
    "status": "running",
    "stage": "waiting_for_whiteboard",
    "pid": os.getpid(),
    "started_utc": now(),
    "old_job_root": str(OLD_ROOT),
    "total_trials": 120,
    "total_checkpoint_count": 6,
    "settings": {
      **old["settings"],
      "checkpoint_workers": 4,
      "seed_workers_per_checkpoint": 1,
      "independent_model_services": True,
      "formal_metrics_valid": False,
    },
    "groups": groups,
    "errors": [],
  }
  status["settings"].pop("endpoint", None)
  workers: list[subprocess.Popen] = []
  paused = False
  serial_termination_requested = False
  serial_fd = os.pidfd_open(SERIAL_PID)
  evaluator_fd = os.pidfd_open(evaluator["pid"])

  def interrupted(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGTERM, interrupted)
  signal.signal(signal.SIGINT, interrupted)
  try:
    with (OLD_ROOT / ".parallel_handoff.lock").open("a+") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      assert_same_process(SERIAL_PID, serial)
      assert_same_process(evaluator["pid"], evaluator)
      paused = True
      signal.pidfd_send_signal(serial_fd, signal.SIGSTOP)
      deadline = time.monotonic() + 10
      while assert_same_process(SERIAL_PID, serial)["state"] not in ("T", "t"):
        if time.monotonic() >= deadline:
          raise RuntimeError("serial orchestrator did not stop at the current group")
        time.sleep(0.1)
      stopped_old = read_json(OLD_ROOT / "status.json")
      if (
        stopped_old.get("task") != "whiteboard-wipe-horizontal"
        or stopped_old.get("checkpoint_step") != 24999
        or stopped_old.get("stage") != "evaluate"
        or any(group.get("status") != "pending" for group in stopped_old["groups"][2:])
      ):
        raise RuntimeError(
          "serial runner crossed the group boundary before being stopped"
        )
      status["serial_orchestrator_paused"] = True
      status["handoff"] = {
        "serial": serial,
        "evaluator": evaluator,
        "children_continue_running": True,
      }
      refresh(status, job_root)
      print(
        f"[{now()}] WAIT whiteboard; only orchestrator {SERIAL_PID} is stopped",
        flush=True,
      )
      deadline = time.monotonic() + 4 * 3600
      next_refresh = 0.0
      while True:
        child = assert_same_process(evaluator["pid"], evaluator)
        if child["ppid"] != SERIAL_PID:
          raise RuntimeError(
            "current evaluator no longer belongs to the old orchestrator"
          )
        if child["state"] == "Z":
          code = os.waitstatus_to_exitcode(child["exit_status"])
          if code != 0:
            raise RuntimeError(f"whiteboard evaluation failed with exit code {code}")
          status["handoff"]["evaluation_exit_code"] = code
          break
        if time.monotonic() >= deadline:
          raise TimeoutError(
            "current whiteboard group did not finish within four hours"
          )
        if time.monotonic() >= next_refresh:
          refresh(status, job_root)
          next_refresh = time.monotonic() + 15
        time.sleep(2)
      batch = validated_batch(
        Path(groups[1]["output_dir"]) / "run/evaluation/summary.json", groups[1]
      )
      validated_batch(
        Path(groups[0]["output_dir"]) / "run/evaluation/summary.json", groups[0]
      )
      status["stage"] = "cleaning_up_serial_handoff"
      refresh(status, job_root)
      serial_termination_requested = True
      signal.pidfd_send_signal(serial_fd, signal.SIGTERM)
      signal.pidfd_send_signal(serial_fd, signal.SIGCONT)
      paused = False
      deadline = time.monotonic() + 90
      while True:
        try:
          state = assert_same_process(SERIAL_PID, serial)["state"]
        except FileNotFoundError:
          break
        if state == "Z":
          break
        if time.monotonic() >= deadline:
          raise TimeoutError("old orchestrator cleanup did not complete")
        time.sleep(1)
      run_record = read_json(Path(groups[1]["output_dir"]) / "run/run_manifest.json")
      if (
        run_record.get("managed_server_stopped") is not True
        or run_record.get("evaluation_process_stopped") is not True
      ):
        raise RuntimeError("old managed service/evaluator cleanup is not confirmed")
      final_old = read_json(OLD_ROOT / "status.json")
      if any(group.get("status") != "pending" for group in final_old["groups"][2:]):
        raise RuntimeError(
          "old serial runner unexpectedly began a remaining checkpoint"
        )
      groups[1].update(
        status="complete",
        completion_basis="evaluation_exit_0_all_20_valid_and_managed_cleanup",
        finished_utc=now(),
        serial_handoff_interruption=True,
      )
      status.update(stage="parallel_evaluations", serial_orchestrator_paused=False)
      status["handoff"].update(
        completed_utc=now(),
        whiteboard_attempted=batch["attempted"],
        whiteboard_valid_trials=batch["valid_trials"],
        original_serial_status=final_old["status"],
        original_run_status=run_record["status"],
        managed_server_stopped=True,
        evaluation_process_stopped=True,
        no_remaining_group_started_by_serial=True,
      )
      write_json(job_root / "handoff_record.json", status["handoff"])
      print(f"[{now()}] HANDOFF complete; 40 whiteboard results retained", flush=True)
      assert_ports_free()
      worker = Path(__file__).with_name("run_pi05_checkpoint_worker.py")
      for index, group in enumerate(groups[2:]):
        group.update(status="launching", started_utc=now())
        output = Path(group["output_dir"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with (job_root / f"group_{index}.log").open("xb", buffering=0) as log:
          process = subprocess.Popen(
            [
              str(SIM_PYTHON),
              str(worker),
              "--job-root",
              str(job_root),
              "--simulation-root",
              str(SIM_ROOT),
              "--openpi-root",
              str(OPENPI_ROOT),
              "--group-index",
              str(index),
              "--endpoint",
              group["endpoint"],
            ],
            cwd=SIM_ROOT,
            env=environment(job_root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
          )
        workers.append(process)
        group["worker_pid"] = process.pid
        refresh(status, job_root)
        print(
          f"[{now()}] LAUNCH {group['task']} step{group['step']} pid={process.pid} {group['endpoint']}",
          flush=True,
        )
      while any(process.poll() is None for process in workers):
        refresh(status, job_root)
        time.sleep(15)
      refresh(status, job_root)
      for group, process in zip(groups[2:], workers, strict=True):
        group["worker_returncode"] = process.returncode
        if process.returncode != 0 or group.get("status") != "complete":
          group["status"] = "failed"
          status["errors"].append(
            f"{group['task']} step{group['step']}: worker returncode={process.returncode}"
          )
      status.update(
        status="failed" if status["errors"] else "complete",
        stage="complete",
        finished_utc=now(),
      )
      refresh(status, job_root)
      return 1 if status["errors"] else 0
  except BaseException as error:
    if paused and not serial_termination_requested:
      signal.pidfd_send_signal(serial_fd, signal.SIGCONT)
      paused = False
      status["serial_resumed_after_handoff_failure"] = True
    if paused and serial_termination_requested:
      signal.pidfd_send_signal(serial_fd, signal.SIGCONT)
      paused = False
    for process in workers:
      if process.poll() is None:
        process.terminate()
    for process in workers:
      if process.poll() is None:
        try:
          process.wait(timeout=90)
        except subprocess.TimeoutExpired:
          status["errors"].append(
            f"worker {process.pid} cleanup timed out; manual inspection required"
          )
    status.update(
      status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
      stage="handoff_or_parallel_error",
      serial_orchestrator_paused=False,
      finished_utc=now(),
    )
    status["errors"].append(f"{type(error).__name__}: {error}")
    refresh(status, job_root)
    raise
  finally:
    os.close(serial_fd)
    os.close(evaluator_fd)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  parser.add_argument("--start", action="store_true")
  args = parser.parse_args()
  job_root = args.job_root.resolve()
  return start(job_root) if args.start else run(job_root)


if __name__ == "__main__":
  raise SystemExit(main())
