#!/usr/bin/env python3
"""Run the six requested H50 checkpoints sequentially with durable progress."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENPI = Path("/cpfs_infra/user/chenxianchi/code/openpi")
MODEL_PYTHON = OPENPI / ".venv-pi05/bin/python"
SIM_PYTHON = ROOT / ".pixi/envs/default/bin/python"
EVALUATION_PARENT = Path("/nas/chenxianchi/datasets/sim/_evaluations")
CHECKPOINT_RUNS = (
  (
    "whiteboard-wipe-horizontal",
    "/nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal/pi05/1010_200/checkpoints/pi05_kaihand_h50/whiteboard_wipe_horizontal_1010_h50_bs128_25k_fsdp8_20261011_222102",
  ),
  (
    "usb-unplug",
    "/nas/chenxianchi/datasets/sim/usb-unplug/pi05/1010_200/checkpoints/pi05_kaihand_h50/usb_unplug_1010_h50_bs128_25k_fsdp8_20261011_222315",
  ),
  (
    "bulb-unscrew",
    "/nas/chenxianchi/datasets/sim/bulb-unscrew/pi05/1010_200/checkpoints/pi05_kaihand_h50/bulb_unscrew_tabletop_1010_h50_bs128_25k_fsdp8_20261011_222530",
  ),
)


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def write_status(job_root: Path, status: dict) -> None:
  status["updated_utc"] = now()
  fd, name = tempfile.mkstemp(prefix=".status.", dir=job_root)
  temporary = Path(name)
  try:
    with os.fdopen(fd, "w") as output:
      json.dump(status, output, ensure_ascii=False, indent=2)
      output.write("\n")
      output.flush()
      os.fsync(output.fileno())
    temporary.replace(job_root / "status.json")
  finally:
    temporary.unlink(missing_ok=True)


def start(job_root: Path, openpi_root: Path = OPENPI) -> int:
  if job_root.parent != EVALUATION_PARENT or not job_root.name.startswith(
    "pi05_h50_k50_serial_"
  ):
    raise ValueError(
      "job-root must be a new pi05_h50_k50_serial_* directory under _evaluations"
    )
  for python in (MODEL_PYTHON, SIM_PYTHON):
    if not python.is_file() or not os.access(python, os.X_OK):
      raise ValueError(f"Python is not executable: {python}")
  if not (openpi_root / "src/openpi/training/config.py").is_file():
    raise ValueError(f"OpenPI source root is incomplete: {openpi_root}")
  for _task, run in CHECKPOINT_RUNS:
    for step in (20000, 24999):
      if not (Path(run) / str(step) / "_CHECKPOINT_METADATA").is_file():
        raise ValueError(f"checkpoint is not committed: {run}/{step}")
  tokenizer_source = Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model")
  if not tokenizer_source.is_file():
    raise FileNotFoundError(tokenizer_source)

  job_root.mkdir(parents=True, exist_ok=False)
  cache = job_root / ".cache/openpi/big_vision"
  cache.mkdir(parents=True)
  shutil.copyfile(tokenizer_source, cache / tokenizer_source.name)
  environment = dict(os.environ)
  environment.update(
    {
      "PYTHONDONTWRITEBYTECODE": "1",
      "PYTHONUNBUFFERED": "1",
      "CUDA_VISIBLE_DEVICES": "0",
      "MUJOCO_EGL_DEVICE_ID": "0",
      "MUJOCO_GL": "egl",
      "PYOPENGL_PLATFORM": "egl",
      "LIBGL_ALWAYS_SOFTWARE": "0",
      "OPENBLAS_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
      "MKL_NUM_THREADS": "1",
      "__EGL_VENDOR_LIBRARY_FILENAMES": "/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json",
      "OPENPI_DATA_HOME": str(job_root / ".cache/openpi"),
      "HF_HOME": "/nas/chenxianchi/datasets/sim/_cache/huggingface",
      "MPLCONFIGDIR": str(job_root / ".cache/matplotlib"),
      "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
  )
  environment.pop("JAX_PLATFORMS", None)
  environment.pop("JAX_PLATFORM_NAME", None)
  write_status(
    job_root, {"status": "starting", "stage": "launching", "started_utc": now()}
  )
  with (job_root / "serial.log").open("xb", buffering=0) as log:
    process = subprocess.Popen(
      [
        str(SIM_PYTHON),
        str(Path(__file__).absolute()),
        "--job-root",
        str(job_root),
        "--openpi-root",
        str(openpi_root),
      ],
      cwd=ROOT,
      env=environment,
      stdin=subprocess.DEVNULL,
      stdout=log,
      stderr=subprocess.STDOUT,
      start_new_session=True,
    )
  (job_root / "serial.pid").write_text(f"{process.pid}\n")
  print(
    json.dumps(
      {
        "pid": process.pid,
        "job_root": str(job_root),
        "log": str(job_root / "serial.log"),
      },
      indent=2,
    )
  )
  return 0


def run(job_root: Path, openpi_root: Path = OPENPI) -> int:
  sys.path.insert(0, str(ROOT / "src"))
  from kaihand_tactile_env.pipeline.deployment import (
    build_prepare_plan,
    execute_prepare,
    run_deployment,
    verify_deployment,
    write_pipeline_deployment,
  )
  from kaihand_tactile_env.pipeline.runtime import load_runtime_profile

  profile = dataclasses.replace(
    load_runtime_profile(ROOT / "configs/runtimes.example.toml", "pi05"),
    python=MODEL_PYTHON,
    project_root=openpi_root,
  )
  profile.validate(require_python=True, require_project=True)
  groups = [
    {
      "task": task,
      "step": step,
      "checkpoint": str(Path(checkpoint_run) / str(step)),
      "status": "pending",
    }
    for task, checkpoint_run in CHECKPOINT_RUNS
    for step in (20000, 24999)
  ]
  status = {
    "status": "running",
    "stage": "starting",
    "pid": os.getpid(),
    "started_utc": now(),
    "completed_checkpoint_count": 0,
    "total_checkpoint_count": 6,
    "settings": {
      "gpu_device": 0,
      "prediction_horizon": 50,
      "execute_steps": "horizon",
      "num_trials": 20,
      "video_count": 20,
      "seed_start": 0,
      "max_sim_seconds": 90,
      "guard_mode": "observe",
      "service_mode": "managed",
      "endpoint": "ws://127.0.0.1:18783",
      "training_started": False,
      "simulation_root": str(ROOT),
      "model_project_root": str(openpi_root),
      "model_python": str(MODEL_PYTHON),
    },
    "groups": groups,
    "errors": [],
  }

  def interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")

  signal.signal(signal.SIGINT, interrupt)
  signal.signal(signal.SIGTERM, interrupt)
  with (job_root / ".serial.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_status(job_root, status)
    current = None
    try:
      for group in groups:
        current = group
        task, step = group["task"], group["step"]
        output = job_root / task / f"step{step}_N20_video20"
        deployment_dir = output / "deployment"
        manifest = deployment_dir / "deployment_manifest.json"
        group["started_utc"] = now()
        group["output_dir"] = str(output)
        status.update({"task": task, "checkpoint_step": step, "stage": "prepare"})
        group["status"] = "preparing"
        write_status(job_root, status)
        print(f"[{now()}] START {task} step{step}", flush=True)
        plan = build_prepare_plan(
          task,
          "pi05",
          checkpoint=Path(group["checkpoint"]),
          output_dir=deployment_dir,
          profile=profile,
        )
        code = execute_prepare(plan)
        if code:
          raise RuntimeError(f"prepare exited with {code}")
        status["stage"] = "verify"
        group["status"] = "verifying"
        write_status(job_root, status)
        verification = verify_deployment(manifest, profile=profile)
        write_pipeline_deployment(
          deployment_dir,
          task=task,
          model_family="pi05",
          prepare_plan=plan,
          profile=profile,
          verification=verification,
        )
        status["stage"] = "evaluate"
        group["status"] = "evaluating"
        write_status(job_root, status)
        record = run_deployment(
          manifest,
          output_dir=output / "run",
          profile=profile,
          service_mode="managed",
          endpoint="ws://127.0.0.1:18783",
          num_trials=20,
          seed_start=0,
          video_count=20,
          execute_steps="horizon",
          max_sim_seconds=90,
          guard_mode="observe",
          server_timeout=600,
        )
        code = int(record.get("evaluation_returncode", 1))
        group["evaluation_returncode"] = code
        group["run_manifest"] = str(output / "run/run_manifest.json")
        group["managed_server_stopped"] = record.get("managed_server_stopped")
        group["evaluation_process_stopped"] = record.get("evaluation_process_stopped")
        if (
          code
          or group["managed_server_stopped"] is not True
          or group["evaluation_process_stopped"] is not True
        ):
          raise RuntimeError(f"evaluation or service cleanup failed: returncode={code}")
        group.update({"status": "complete", "finished_utc": now()})
        status["completed_checkpoint_count"] += 1
        write_status(job_root, status)
        print(f"[{now()}] COMPLETE {task} step{step}", flush=True)
      status.update({"status": "complete", "stage": "complete", "finished_utc": now()})
      write_status(job_root, status)
      return 0
    except BaseException as error:
      interrupted = isinstance(error, KeyboardInterrupt) or hasattr(error, "signum")
      status.update(
        {"status": "interrupted" if interrupted else "failed", "finished_utc": now()}
      )
      status["errors"].append(f"{type(error).__name__}: {error}")
      if current is not None:
        current["status"] = status["status"]
      write_status(job_root, status)
      raise


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  parser.add_argument("--start", action="store_true")
  parser.add_argument("--openpi-root", type=Path, default=OPENPI)
  args = parser.parse_args()
  job_root = args.job_root.expanduser().resolve()
  openpi_root = args.openpi_root.expanduser().resolve()
  return start(job_root, openpi_root) if args.start else run(job_root, openpi_root)


if __name__ == "__main__":
  raise SystemExit(main())
