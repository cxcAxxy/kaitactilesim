#!/usr/bin/env python3
"""Evaluate a fixed H50 checkpoint using isolated services and frozen sources."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import signal
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

EVALUATION_PARENT = Path("/nas/chenxianchi/datasets/sim/_evaluations")
SNAPSHOT_PARENT = Path("/cpfs_infra/user/chenxianchi/runtime_snapshots")
# Do not resolve this symlink: its venv entry point selects the model dependencies.
MODEL_PYTHON = Path("/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python")
CHECKPOINT_RUNS = (
  (
    "usb-unplug",
    "/nas/chenxianchi/datasets/sim/usb-unplug/pi05/1010_200/checkpoints/pi05_kaihand_h50/usb_unplug_1010_h50_bs128_25k_fsdp8_20261011_222315",
  ),
  (
    "bulb-unscrew",
    "/nas/chenxianchi/datasets/sim/bulb-unscrew/pi05/1010_200/checkpoints/pi05_kaihand_h50/bulb_unscrew_tabletop_1010_h50_bs128_25k_fsdp8_20261011_222530",
  ),
)
GROUPS = tuple(
  (task, step, Path(checkpoint_run) / str(step))
  for task, checkpoint_run in CHECKPOINT_RUNS
  for step in (20000, 24999)
)
NEW_CHECKPOINT_RUNS = (
  (
    "install-ram",
    "/nas/chenxianchi/datasets/sim/install-ram/pi05/0920_200/checkpoints/pi05_kaihand_h50/install_ram_0920_h50_bs128_25k",
  ),
  (
    "poker-draw",
    "/nas/chenxianchi/datasets/sim/poker-draw/pi05/0920_200/checkpoints/pi05_kaihand_h50/poker_draw_0920_h50_bs128_25k",
  ),
  (
    "usb-insert",
    "/nas/chenxianchi/datasets/sim/usb_insert/pi05/0921_200/checkpoints/pi05_kaihand_h50/usb_insert_0921_h50_hand_tail_v1_bs128_25k",
  ),
)
NEW_GROUPS = tuple(
  (task, step, Path(checkpoint_run) / str(step))
  for task, checkpoint_run in NEW_CHECKPOINT_RUNS
  for step in (20000, 24999)
)


def now() -> str:
  return datetime.now(timezone.utc).isoformat()


def write_status(output_dir: Path, status: dict) -> None:
  """Atomically publish only this worker's progress."""
  status["updated_utc"] = now()
  descriptor, name = tempfile.mkstemp(prefix=".group_status.", dir=output_dir)
  temporary = Path(name)
  try:
    with os.fdopen(descriptor, "w") as stream:
      json.dump(status, stream, ensure_ascii=False, indent=2)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(output_dir / "group_status.json")
  finally:
    temporary.unlink(missing_ok=True)


def validate_scope(
  job_root: Path,
  simulation_root: Path,
  openpi_root: Path,
  group_index: int,
  endpoint: str,
  new_tasks: bool = False,
) -> None:
  prefix = "pi05_h50_k50_new_tasks_" if new_tasks else "pi05_h50_k50_parallel_"
  if (
    job_root.parent != EVALUATION_PARENT
    or not job_root.name.startswith(prefix)
    or not job_root.is_dir()
    or job_root.is_symlink()
  ):
    raise ValueError(f"job-root must be an existing {prefix}* directory")
  groups = NEW_GROUPS if new_tasks else GROUPS
  if group_index not in range(len(groups)):
    raise ValueError(f"group-index must be between 0 and {len(groups) - 1}")
  expected_endpoint = f"ws://127.0.0.1:{(18800 if new_tasks else 18784) + group_index}"
  if endpoint != expected_endpoint:
    raise ValueError(f"group {group_index} requires endpoint {expected_endpoint}")
  if (
    simulation_root.name != "kaitactilesim"
    or openpi_root.name != "openpi"
    or simulation_root.parent != openpi_root.parent
    or simulation_root.parent.parent != SNAPSHOT_PARENT
    or simulation_root.is_symlink()
    or openpi_root.is_symlink()
  ):
    raise ValueError("source roots must be sibling copies inside runtime_snapshots")


def assert_readonly_tree(path: Path) -> None:
  if not path.is_dir() or path.is_symlink():
    raise ValueError(f"frozen source directory is missing or linked: {path}")
  for directory, directories, files in os.walk(path, followlinks=False):
    for entry in [
      Path(directory),
      *(Path(directory) / name for name in directories + files),
    ]:
      if entry.is_symlink() or entry.stat().st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
      ):
        raise ValueError(f"source must be a read-only copy: {entry}")


def validate_assets(simulation_root: Path, openpi_root: Path, checkpoint: Path) -> None:
  for source in (
    simulation_root / "src",
    simulation_root / "scripts",
    simulation_root / "configs",
    openpi_root / "src",
    openpi_root / "packages",
    openpi_root / "scripts",
  ):
    assert_readonly_tree(source)
  for entry in (
    simulation_root / "configs/runtimes.example.toml",
    openpi_root / "src/openpi/training/config.py",
    openpi_root / "packages/openpi-client/src/openpi_client/__init__.py",
    checkpoint / "_CHECKPOINT_METADATA",
    checkpoint / "params/_METADATA",
  ):
    if not entry.is_file():
      raise ValueError(
        f"required source or committed checkpoint metadata missing: {entry}"
      )
  for python in (MODEL_PYTHON, simulation_root / ".pixi/envs/default/bin/python"):
    if not python.is_file() or not os.access(python, os.X_OK):
      raise ValueError(f"Python is not executable: {python}")


def load_api(simulation_root: Path, openpi_root: Path) -> SimpleNamespace:
  sources = [
    str(simulation_root / "src"),
    str(openpi_root / "src"),
    str(openpi_root / "packages/openpi-client/src"),
  ]
  sys.path[:0] = sources
  # Children inherit only the frozen project roots, not the mutable worktree.
  os.environ["PYTHONPATH"] = os.pathsep.join(sources)
  os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
  sys.dont_write_bytecode = True
  from kaihand_tactile_env.pipeline import deployment, runtime

  if not Path(deployment.__file__).resolve().is_relative_to(simulation_root):
    raise RuntimeError(
      "deployment module was not imported from frozen simulation source"
    )
  if not Path(runtime.__file__).resolve().is_relative_to(simulation_root):
    raise RuntimeError("runtime module was not imported from frozen simulation source")
  profile = dataclasses.replace(
    runtime.load_runtime_profile(
      simulation_root / "configs/runtimes.example.toml", "pi05"
    ),
    python=MODEL_PYTHON,
    project_root=openpi_root,
  )
  profile.validate(require_python=True, require_project=True)
  return SimpleNamespace(deployment=deployment, profile=profile)


def run_worker(
  job_root: Path,
  simulation_root: Path,
  openpi_root: Path,
  group_index: int,
  endpoint: str,
  new_tasks: bool = False,
  num_trials: int = 20,
  seed_start: int = 0,
) -> int:
  # Validate before creating a lock, output directory, or progress record.
  if (
    type(num_trials) is not int
    or type(seed_start) is not int
    or not 1 <= num_trials <= 20
    or not 0 <= seed_start < 20
    or seed_start + num_trials > 20
  ):
    raise ValueError(
      "trial range requires 1 <= num-trials <= 20, 0 <= seed-start < 20, "
      "and seed-start + num-trials <= 20"
    )
  if not new_tasks and (num_trials != 20 or seed_start != 0):
    raise ValueError("non-default trial ranges require --new-tasks")
  validate_scope(
    job_root, simulation_root, openpi_root, group_index, endpoint, new_tasks
  )
  task, step, checkpoint = (NEW_GROUPS if new_tasks else GROUPS)[group_index]
  output_dir = job_root / task / f"step{step}_N20_video20"
  lock_path = job_root / f".group_{group_index}.lock"
  with lock_path.open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Never overwrite deployment, run, or progress from an earlier invocation.
    output_dir.parent.mkdir(exist_ok=True)
    if output_dir.parent.is_symlink() or output_dir.parent.resolve().parent != job_root:
      raise ValueError("group parent must be an ordinary directory inside this job")
    output_dir.mkdir(exist_ok=False)
    status = {
      "status": "running",
      "stage": "preflight",
      "pid": os.getpid(),
      "started_utc": now(),
      "group_index": group_index,
      "task": task,
      "checkpoint_step": step,
      "checkpoint": str(checkpoint),
      "output_dir": str(output_dir),
      "simulation_root": str(simulation_root),
      "model_project_root": str(openpi_root),
      "model_python": str(MODEL_PYTHON),
      "endpoint": endpoint,
      "port": (18800 if new_tasks else 18784) + group_index,
      "settings": {
        "gpu_device": 0,
        "prediction_horizon": 50,
        "execute_steps": "horizon",
        "num_trials": num_trials,
        "video_count": num_trials,
        "seed_start": seed_start,
        "max_sim_seconds": 90,
        "guard_mode": "observe",
        "service_mode": "managed",
        "server_timeout": 600,
        "training_started": False,
        "independent_model_service": True,
        "formal_metrics_valid": False,
      },
      "errors": [],
    }
    write_status(output_dir, status)

    def interrupt(signum: int, _frame: object) -> None:
      raise KeyboardInterrupt(f"received signal {signum}")

    previous_handlers = {
      signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
      signal.signal(signum, interrupt)
    try:
      validate_assets(simulation_root, openpi_root, checkpoint)
      api = load_api(simulation_root, openpi_root)
      deployment = api.deployment
      profile = api.profile
      deployment_dir = output_dir / "deployment"
      manifest = deployment_dir / "deployment_manifest.json"
      status["stage"] = "prepare"
      write_status(output_dir, status)
      print(f"[{now()}] START {task} step{step} endpoint={endpoint}", flush=True)
      plan = deployment.build_prepare_plan(
        task,
        "pi05",
        checkpoint=checkpoint,
        output_dir=deployment_dir,
        profile=profile,
      )
      code = deployment.execute_prepare(plan)
      if code:
        raise RuntimeError(f"prepare exited with {code}")
      status["stage"] = "verify"
      write_status(output_dir, status)
      verification = deployment.verify_deployment(manifest, profile=profile)
      deployment.write_pipeline_deployment(
        deployment_dir,
        task=task,
        model_family="pi05",
        prepare_plan=plan,
        profile=profile,
        verification=verification,
      )
      status["stage"] = "evaluate"
      status["run_manifest"] = str(output_dir / "run/run_manifest.json")
      write_status(output_dir, status)
      record = deployment.run_deployment(
        manifest,
        output_dir=output_dir / "run",
        profile=profile,
        service_mode="managed",
        endpoint=endpoint,
        num_trials=num_trials,
        seed_start=seed_start,
        video_count=num_trials,
        execute_steps="horizon",
        max_sim_seconds=90,
        guard_mode="observe",
        server_timeout=600,
      )
      code = int(record.get("evaluation_returncode", 1))
      status["evaluation_returncode"] = code
      status["managed_server_stopped"] = record.get("managed_server_stopped")
      status["evaluation_process_stopped"] = record.get("evaluation_process_stopped")
      if (
        code
        or status["managed_server_stopped"] is not True
        or status["evaluation_process_stopped"] is not True
      ):
        raise RuntimeError(
          f"evaluation or owned service cleanup failed: returncode={code}"
        )
      status.update({"status": "complete", "stage": "complete", "finished_utc": now()})
      write_status(output_dir, status)
      print(f"[{now()}] COMPLETE {task} step{step}", flush=True)
      return 0
    except BaseException as error:
      interrupted = isinstance(error, KeyboardInterrupt) or hasattr(error, "signum")
      status.update(
        {"status": "interrupted" if interrupted else "failed", "finished_utc": now()}
      )
      status["errors"].append(f"{type(error).__name__}: {error}")
      write_status(output_dir, status)
      raise
    finally:
      for signum, previous_handler in previous_handlers.items():
        signal.signal(signum, previous_handler)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--job-root", type=Path, required=True)
  parser.add_argument("--simulation-root", type=Path, required=True)
  parser.add_argument("--openpi-root", type=Path, required=True)
  parser.add_argument("--group-index", type=int, choices=range(6), required=True)
  parser.add_argument("--endpoint", required=True)
  parser.add_argument("--new-tasks", action="store_true")
  parser.add_argument("--num-trials", type=int, default=20)
  parser.add_argument("--seed-start", type=int, default=0)
  args = parser.parse_args()
  return run_worker(
    args.job_root.expanduser().resolve(strict=True),
    args.simulation_root.expanduser().resolve(strict=True),
    args.openpi_root.expanduser().resolve(strict=True),
    args.group_index,
    args.endpoint,
    args.new_tasks,
    args.num_trials,
    args.seed_start,
  )


if __name__ == "__main__":
  raise SystemExit(main())
