#!/usr/bin/env python3
"""Server collection supervisor. Run --help; see README.md beside this file."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = (
  "pick-place",
  "poker-draw",
  "usb-insert",
  "bulb-screw",
  "vase-wipe",
  "install-ram",
  "whiteboard-wipe",
  "sponge-grasp",
)
TERMINAL = {"success", "failed", "timeout"}
COLLECTION_CAMERAS = ("head", "left_wrist", "right_wrist")
COLLECTION_CAMERA_HZ = 30
POKER_ACCEPTANCE_POLICY = "task-completion-v1"
DEFAULT_STAGING_ROOT = Path("/cpfs_infra/user/chenxianchi/.capture_staging")
PUBLISH_POLICY = "cpfs-stage-verified-copy-nas-atomic-v1"


def save(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
  temporary.replace(path)


def read(path):
  return json.loads(path.read_text())


def summary_record(record):
  # Per-frame hashes stay in status.json instead of being copied into a
  # progressively larger batch summary after every episode.
  return {
    key: value
    for key, value in record.items()
    if key not in ("artifact_bytes", "artifact_sha256", "command")
  }


def collection_summary(records, *, target_successes=None, max_attempts=None):
  by_task = {}
  for record in records:
    task = record["task"]
    counts = by_task.setdefault(
      task,
      {"attempted": 0, "success": 0, "failed": 0, "timeout": 0},
    )
    counts["attempted"] += 1
    status = record["status"]
    if status in counts:
      counts[status] += 1
  payload = {
    "episodes": records,
    "success_count": sum(r["status"] == "success" for r in records),
    "by_task": by_task,
  }
  if target_successes is not None:
    payload.update(
      target_successes=target_successes,
      max_attempts_per_task=max_attempts,
      target_met={
        task: counts["success"] >= target_successes for task, counts in by_task.items()
      },
    )
  return payload


def episode_seed(seed, task, index):
  digest = hashlib.sha256(f"{seed}:{task}:{index}".encode()).digest()
  return int.from_bytes(digest[:4], "little")


def environment(backend):
  env = os.environ.copy()
  env.update(
    MUJOCO_GL="osmesa" if backend == "software" else "egl",
    PYOPENGL_PLATFORM="osmesa" if backend == "software" else "egl",
    PYTHONUNBUFFERED="1",
    KAIHAND_RENDER_BACKEND=backend,
    LIBGL_ALWAYS_SOFTWARE="1" if backend == "software" else "0",
  )
  for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LP_NUM_THREADS",
  ):
    env[key] = "1"
  if backend == "software":
    env["GALLIUM_DRIVER"] = "llvmpipe"
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
  else:
    env.pop("GALLIUM_DRIVER", None)
  return env


def command(task, output, seed, fixed_stains=False):
  base = [sys.executable]
  scripts = ROOT / "scripts/workcell"
  if task in ("pick-place", "poker-draw"):
    cmd = base + [
      str(scripts / "record_dataset.py"),
      "--scene",
      task,
      "--output-dir",
      str(output),
      "--episodes",
      "1",
      "--workers",
      "1",
      "--seed",
      str(seed),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--cameras",
      *COLLECTION_CAMERAS,
      "--rgb-only",
    ]
    if task == "pick-place":
      cmd += ["--object", "cylinder", "--side", "right"]
    else:
      cmd += [
        "--preset",
        "middle-force-precontact-v1",
        "--tactile-links",
        "right",
        "--acceptance-policy",
        POKER_ACCEPTANCE_POLICY,
      ]
    return cmd
  if task == "usb-insert":
    return base + [
      str(scripts / "record_usb_dataset.py"),
      "--output-dir",
      str(output),
      "--episodes",
      "1",
      "--workers",
      "1",
      "--seed",
      str(seed),
      "--motion-profile",
      "fast",
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--cameras",
      *COLLECTION_CAMERAS,
      "--xy-jitter-mm",
      "10",
      "--yaw-jitter-deg",
      "5",
      "--precontact-noise-mm",
      "0.5",
    ]
  if task == "bulb-screw":
    return base + [
      str(scripts / "refresh_light_bulb_example.py"),
      "--position-seed",
      str(seed),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--buffer-rows",
      "128",
      "--cameras",
      *COLLECTION_CAMERAS,
      "--raw-only",
      "--output",
      str(output),
    ]
  if task == "install-ram":
    return base + [
      str(scripts / "record_install_ram_example.py"),
      "--position-seed",
      str(seed),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--buffer-rows",
      "64",
      "--cameras",
      *COLLECTION_CAMERAS,
      "--raw-only",
      "--output-dir",
      str(output),
    ]
  if task == "whiteboard-wipe":
    cmd = base + [
      str(scripts / "record_whiteboard_example.py"),
      "--output-dir",
      str(output),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--buffer-rows",
      "128",
      "--cameras",
      *COLLECTION_CAMERAS,
      "--raw-only",
    ]
    return cmd if fixed_stains else cmd + ["--ink-seed", str(seed)]
  if task == "vase-wipe":
    cmd = base + [
      str(scripts / "view_vase_wipe.py"),
      "--headless",
      "--run-task",
      "--output-dir",
      str(output),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--buffer-rows",
      "128",
      "--cameras",
      *COLLECTION_CAMERAS,
      "--raw-only",
    ]
    return cmd if fixed_stains else cmd + ["--stain-seed", str(seed)]
  if task == "sponge-grasp":
    return base + [
      str(scripts / "record_sponge_grasp.py"),
      "--output-dir",
      str(output),
      "--seed",
      str(seed),
      "--camera-hz",
      str(COLLECTION_CAMERA_HZ),
      "--buffer-rows",
      "128",
      "--raw-only",
      "--cameras",
      *COLLECTION_CAMERAS,
    ]
  raise ValueError(task)


def outcome(task, output):
  """Require the recorder's actual outcome and a published raw file."""
  if not any(output.rglob("*.h5")):
    return False
  if task in ("pick-place", "poker-draw"):
    manifests = [p.with_suffix(".json") for p in output.glob("*.h5")]
    return len(manifests) == 1 and read(manifests[0])["outcome"].get("success") is True
  if task == "sponge-grasp":
    raw = output / "raw/sponge_grasp_000000.h5"
    result = read(raw.with_suffix(".result.json"))
    manifest = read(raw.with_suffix(".json"))
    return (
      raw.is_file()
      and result.get("success") is True
      and manifest.get("validation", {}).get("valid") is True
      and manifest.get("task_audit_passed") is True
    )
  if task == "vase-wipe":
    result = read(output / "result.json")
    validation = read(output / "raw/episode.json").get("validation", {})
    return result.get("success") is True and validation.get("valid") is True
  if task == "whiteboard-wipe":
    result = read(output / "result.json")
    validation = read(output / "raw/episode.json").get("validation", {})
    return result.get("success") is True and validation.get("valid") is True
  if task == "usb-insert":
    report = read(output / "summary.json")
    return report.get("complete") is True and report.get("successful_episodes") == 1
  raw = next(output.glob("raw/*.h5"))
  result = read(raw.with_suffix(".result.json"))
  validation = read(raw.with_suffix(".json")).get("validation", {})
  return result.get("success") is True and validation.get("valid") is True


def source_hashes():
  paths = [ROOT / n for n in ("pyproject.toml", "pixi.toml", "pixi.lock")]
  for directory in ("src", "scripts"):
    paths.extend(
      p
      for p in (ROOT / directory).rglob("*")
      if p.is_file()
      and p.suffix in (".py", ".xml", ".png", ".stl", ".STL", ".json", ".obj")
    )
  return {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(paths)
    if p.is_file()
  }


def artifact_hashes(directory):
  """Hash actual artifact bytes and cross-check each recorder HDF5 digest."""
  result = actual_artifact_hashes(directory)
  for relative, digest in result.items():
    path = directory / relative
    if path.suffix != ".h5":
      continue
    recorder_sidecar = path.with_suffix(".json")
    if not recorder_sidecar.is_file():
      continue
    manifest = read(recorder_sidecar)
    if manifest.get("episode") != path.name or manifest.get("sha256") != digest:
      raise ValueError(f"recorder sidecar SHA-256 differs from HDF5: {path}")
  return result


def artifact_sizes(directory):
  return {
    str(path.relative_to(directory)): path.stat().st_size
    for path in sorted(directory.rglob("*"))
    if path.is_file()
  }


def actual_artifact_hashes(directory):
  """Hash every copied byte, including HDF5, for one-time publication checks."""
  result = {}
  for path in sorted(directory.rglob("*")):
    if not path.is_file():
      continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
      for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
    result[str(path.relative_to(directory))] = digest.hexdigest()
  return result


def publish_staged_data(staged, target):
  """Copy a validated episode to NAS, verify bytes, then expose it atomically."""
  if target.exists() or target.is_symlink():
    raise FileExistsError(f"published data already exists: {target}")
  expected_sizes = artifact_sizes(staged)
  expected_hashes = artifact_hashes(staged)
  if not expected_sizes or not any(name.endswith(".h5") for name in expected_sizes):
    raise ValueError(f"staged episode has no HDF5 artifact: {staged}")
  upload = target.parent / (
    f".{target.name}.publishing-{os.getpid()}-{threading.get_ident()}"
  )
  if upload.exists() or upload.is_symlink():
    raise FileExistsError(f"publication workspace already exists: {upload}")
  try:
    shutil.copytree(staged, upload, copy_function=shutil.copyfile)
    if artifact_sizes(upload) != expected_sizes:
      raise OSError("published artifact paths or sizes differ from staging")
    if actual_artifact_hashes(upload) != expected_hashes:
      raise OSError("published artifact SHA-256 differs from staging")
    upload.replace(target)
  finally:
    if upload.exists() and not upload.is_symlink():
      shutil.rmtree(upload)
  return expected_hashes, expected_sizes


def staging_episode_directory(staging_root, output, task, index):
  identity = hashlib.sha256(str(output).encode()).hexdigest()[:16]
  batch = f"{output.name}-{identity}"
  return staging_root / batch / task / f"{index:06d}"


def artifacts_match(directory, record):
  """Check actual bytes against the published artifact inventory."""
  expected_sizes = record.get("artifact_bytes")
  if expected_sizes is not None and artifact_sizes(directory) != expected_sizes:
    return False
  try:
    return artifact_hashes(directory) == record.get("artifact_sha256")
  except (OSError, ValueError):
    return False


def check(tasks, backend):
  env = environment(backend)
  os.environ.clear()
  os.environ.update(env)
  # Heavy imports only after the backend and thread settings are established.
  import mujoco
  from kaihand_tactile_env.shared.config import CameraConfig
  from kaihand_tactile_env.shared.rendering import WorkcellRenderer

  reports = {}
  for task in tasks:
    path = ROOT / "src/kaihand_tactile_env/tasks" / task.replace("-", "_") / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cameras = tuple(
      CameraConfig(name, width=160, height=120, depth=False, segmentation=False)
      for name in COLLECTION_CAMERAS
    )
    with WorkcellRenderer(model, cameras) as renderer:
      for camera in cameras:
        renderer.capture(data, camera)
      reports[task] = renderer.backend_info
    print(f"CHECK OK: {task}", flush=True)
  return {"mujoco": mujoco.__version__, "python": sys.version, "renderers": reports}


def stop_child(process):
  if process.poll() is not None:
    return
  os.killpg(process.pid, signal.SIGINT)
  try:
    process.wait(timeout=20)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_episode(cmd, directory, env, timeout, stop_event=None):
  started = time.monotonic()
  process = None
  status = "failed"
  with (directory / "collector.log").open("w") as log:
    try:
      process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
      )
      save(directory / "process.json", {"pid": process.pid, "command": cmd})
      while process.poll() is None:
        if stop_event is not None and stop_event.is_set():
          status = "interrupted"
          stop_child(process)
          break
        if time.monotonic() - started >= timeout:
          status = "timeout"
          stop_child(process)
          break
        time.sleep(0.5)
      else:
        status = "completed" if process.returncode == 0 else "failed"
    except KeyboardInterrupt:
      status = "interrupted"
      if process is not None:
        stop_child(process)
    except OSError as error:
      print(f"Cannot run collector: {error}", file=log)
    finally:
      if process is not None and process.poll() is None:
        stop_child(process)
  return {
    "status": status,
    "returncode": process.returncode if process else None,
    "wall_seconds": time.monotonic() - started,
  }


def collect_episode(task, index, output, args, env, stop_event=None):
  """Collect or verify one isolated episode index; safe to call from a thread."""
  episode = output / task / f"{index:06d}"
  episode.mkdir(parents=True, exist_ok=True)
  attempts = sorted(
    episode.glob("attempt_*"), key=lambda path: int(path.name.split("_")[-1])
  )
  previous = attempts[-1] / "status.json" if attempts else None
  staging_episode = staging_episode_directory(args.staging_root, output, task, index)
  if previous and previous.exists():
    record = read(previous)
    process_file = attempts[-1] / "process.json"
    if record["status"] == "running" and process_file.exists():
      pid = read(process_file)["pid"]
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        pass
      else:
        raise RuntimeError(
          f"Previous collector PID {pid} may still be active; "
          f"stop it before resuming {episode}"
        )
    if record["status"] in TERMINAL:
      if record["status"] == "success" and not artifacts_match(
        attempts[-1] / "data", record
      ):
        raise RuntimeError(
          "Successful episode is missing/corrupt; preserve it and use a new "
          f"output: {episode}"
        )
      print(f"SKIP {task}/{index:06d}: {record['status']}", flush=True)
      return summary_record(record)
    if record["status"] == "publish_failed":
      staged_data = Path(record["staging_dir"])
      published_data = output / record["data_dir"]
      if staged_data.is_dir() and outcome(task, staged_data):
        hashes, sizes = publish_staged_data(staged_data, published_data)
        record.update(
          status="success",
          artifact_sha256=hashes,
          artifact_bytes=sizes,
          hdf5=[
            str(path.relative_to(output))
            for path in sorted(published_data.rglob("*.h5"))
          ],
          published_unix_s=time.time(),
        )
        save(previous, record)
        shutil.rmtree(staging_episode)
        print(f"PUBLISH RECOVERED {task}/{index:06d}", flush=True)
        return summary_record(record)
  if stop_event is not None and stop_event.is_set():
    return {
      "task": task,
      "episode_index": index,
      "status": "interrupted",
      "returncode": None,
      "wall_seconds": 0.0,
    }
  minimum_free = args.min_free_gb * 1024**3
  if shutil.disk_usage(output).free < minimum_free:
    raise RuntimeError("Insufficient NAS free disk; no new episode started")
  if shutil.disk_usage(args.staging_root).free < minimum_free:
    raise RuntimeError("Insufficient CPFS staging free disk; no new episode started")
  if staging_episode.exists():
    shutil.rmtree(staging_episode)
  attempt_index = len(attempts) + 1
  directory = episode / f"attempt_{attempt_index:03d}"
  directory.mkdir()
  staging_attempt = staging_episode / f"attempt_{attempt_index:03d}"
  staging_attempt.mkdir(parents=True)
  seed = episode_seed(args.seed, task, index)
  data = directory / "data"
  staged_data = staging_attempt / "data"
  cmd = command(task, staged_data, seed, args.fixed_stains)
  record = {
    "task": task,
    "episode_index": index,
    "attempt_index": attempt_index,
    "seed": seed,
    "command": cmd,
    "data_dir": str(data.relative_to(output)),
    "staging_dir": str(staged_data),
    "status": "running",
    "started_unix_s": time.time(),
  }
  save(directory / "status.json", record)
  print(
    f"START {task}/{index:06d}, seed={seed}; log={directory / 'collector.log'}",
    flush=True,
  )
  record.update(run_episode(cmd, directory, env, args.timeout_s, stop_event))
  if record["status"] == "completed":
    try:
      record["status"] = "success" if outcome(task, staged_data) else "failed"
    except (OSError, ValueError, KeyError) as error:
      record.update(status="failed", outcome_error=str(error))
  if record["status"] == "success":
    try:
      hashes, sizes = publish_staged_data(staged_data, data)
      record["hdf5"] = [
        str(path.relative_to(output)) for path in sorted(data.rglob("*.h5"))
      ]
      record["artifact_sha256"] = hashes
      record["artifact_bytes"] = sizes
      record["published_unix_s"] = time.time()
    except (OSError, ValueError) as error:
      record.update(status="publish_failed", publish_error=str(error))
  save(directory / "status.json", record)
  if record["status"] != "publish_failed" and staging_episode.exists():
    shutil.rmtree(staging_episode)
  print(f"END {task}/{index:06d}: {record['status']}", flush=True)
  return summary_record(record)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", nargs="+", choices=(*TASKS, "all"), required=True)
  parser.add_argument("--output-dir", type=Path)
  count = parser.add_mutually_exclusive_group()
  count.add_argument(
    "--episodes",
    type=int,
    help="Attempt count per task (default: 1 when --target-successes is absent)",
  )
  count.add_argument(
    "--target-successes",
    type=int,
    help="Continue distinct episode attempts until each task reaches this many successes",
  )
  parser.add_argument(
    "--max-attempts",
    type=int,
    help="Safety limit per task; required with --target-successes and may be raised on resume",
  )
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--render-backend", choices=("hardware", "software"), default="hardware"
  )
  parser.add_argument("--timeout-s", type=float, default=7200)
  parser.add_argument("--min-free-gb", type=float, default=10)
  parser.add_argument(
    "--staging-root",
    type=Path,
    default=DEFAULT_STAGING_ROOT,
    help="Fast local/CPFS workspace; successful data is verified before NAS publication",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="Independent episode subprocesses to run in parallel (1..8)",
  )
  parser.add_argument(
    "--fixed-stains",
    action="store_true",
    help="Use fixed vase stains and whiteboard ink positions",
  )
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument(
    "--check", action="store_true", help="Compile and render chosen scenes only"
  )
  args = parser.parse_args()
  import math

  if args.episodes is None and args.target_successes is None:
    args.episodes = 1
  if args.target_successes is not None and args.max_attempts is None:
    parser.error("--max-attempts is required with --target-successes")
  if args.target_successes is None and args.max_attempts is not None:
    parser.error("--max-attempts requires --target-successes")

  if (
    (args.episodes is not None and args.episodes < 1)
    or (args.target_successes is not None and args.target_successes < 1)
    or (args.max_attempts is not None and args.max_attempts < args.target_successes)
    or min(args.start_index, args.seed) < 0
    or not math.isfinite(args.timeout_s)
    or args.timeout_s <= 0
    or not math.isfinite(args.min_free_gb)
    or args.min_free_gb < 0
    or not 1 <= args.workers <= 8
  ):
    parser.error(
      "counts and seed must be nonnegative; attempt/target/timeout counts must "
      "be positive; max-attempts must cover target-successes; workers must be 1..8"
    )
  attempt_limit = args.episodes if args.episodes is not None else args.max_attempts
  tasks = list(TASKS) if "all" in args.task else list(dict.fromkeys(args.task))
  if args.check:
    print(json.dumps(check(tasks, args.render_backend), indent=2))
    return 0
  if args.output_dir is None:
    parser.error("--output-dir is required unless --check")
  output = args.output_dir.resolve()
  args.staging_root = args.staging_root.expanduser().resolve()
  if args.dry_run:
    for task in tasks:
      index = args.start_index
      print(
        json.dumps(
          {
            "task": task,
            "episodes": args.episodes,
            "target_successes": args.target_successes,
            "max_attempts": args.max_attempts,
            "workers": args.workers,
            "staging_root": str(args.staging_root),
            "first_command": command(
              task,
              staging_episode_directory(
                args.staging_root, output, task, index
              ) / "attempt_001/data",
              episode_seed(args.seed, task, index),
              args.fixed_stains,
            ),
          }
        )
      )
    return 0
  output.mkdir(parents=True, exist_ok=True)
  args.staging_root.mkdir(parents=True, exist_ok=True)
  with (output / ".collection.lock").open("a+") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      parser.error("another collector holds this output directory")
    contract = {
      "schema": "task_collection_v2",
      "tasks": tasks,
      "collection_mode": (
        "target-successes" if args.target_successes is not None else "attempts"
      ),
      "target_successes": args.target_successes,
      "seed": args.seed,
      "start_index": args.start_index,
      "render_backend": args.render_backend,
      "fixed_stains": args.fixed_stains,
      "staging_root": str(args.staging_root),
      "publish_policy": PUBLISH_POLICY,
      "source_sha256": source_hashes(),
    }
    manifest = output / "collection.json"
    if manifest.exists():
      if not args.resume or read(manifest) != contract:
        parser.error(
          "existing collection requires --resume and identical code/settings; otherwise use a new directory"
        )
    elif args.resume or any(p.name != ".collection.lock" for p in output.iterdir()):
      parser.error(
        "cannot resume an unknown directory or initialize a nonempty directory"
      )
    else:
      save(manifest, contract)
    # Preflight once per invocation, before spending time on an episode.
    preflight = subprocess.run(
      [
        sys.executable,
        str(Path(__file__).resolve()),
        "--check",
        "--task",
        *tasks,
        "--render-backend",
        args.render_backend,
      ],
      cwd=ROOT,
      env=environment(args.render_backend),
      capture_output=True,
      text=True,
    )
    (output / "preflight.log").write_text(preflight.stdout + preflight.stderr)
    if preflight.returncode:
      print(f"Preflight failed: {output / 'preflight.log'}", file=sys.stderr)
      return 1
    results = []
    targets_met = {}
    task_order = {task: index for index, task in enumerate(tasks)}

    def ordered_results():
      return sorted(
        results,
        key=lambda record: (
          task_order[record["task"]],
          record["episode_index"],
        ),
      )

    def write_summary():
      save(
        output / "summary.json",
        collection_summary(
          ordered_results(),
          target_successes=args.target_successes,
          max_attempts=args.max_attempts,
        ),
      )

    for task in tasks:
      task_successes = 0
      stop_event = threading.Event()
      next_index = args.start_index
      end_index = args.start_index + attempt_limit
      task_env = environment(args.render_backend)
      interrupted_task = False
      in_flight = {}
      with ThreadPoolExecutor(max_workers=args.workers) as executor:
        try:
          while True:
            while (
              not interrupted_task
              and next_index < end_index
              and len(in_flight) < args.workers
              and (
                args.target_successes is None
                or task_successes + len(in_flight) < args.target_successes
              )
            ):
              future = executor.submit(
                collect_episode,
                task,
                next_index,
                output,
                args,
                task_env,
                stop_event,
              )
              in_flight[future] = next_index
              next_index += 1
            if not in_flight:
              break
            completed, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in completed:
              in_flight.pop(future)
              record = future.result()
              results.append(record)
              task_successes += record["status"] == "success"
              if record["status"] == "interrupted":
                interrupted_task = True
                stop_event.set()
            write_summary()
        except BaseException:
          stop_event.set()
          for future in in_flight:
            future.cancel()
          raise
      if interrupted_task:
        write_summary()
        return 130
      if args.target_successes is not None and task_successes >= args.target_successes:
        print(
          f"TARGET {task}: {task_successes}/{args.target_successes} successes",
          flush=True,
        )
      targets_met[task] = (
        args.target_successes is None or task_successes >= args.target_successes
      )
      if args.target_successes is not None and not targets_met[task]:
        print(
          f"TARGET NOT MET {task}: {task_successes}/{args.target_successes} "
          f"successes after {attempt_limit} indexed attempts",
          file=sys.stderr,
          flush=True,
        )
    write_summary()
    if args.target_successes is not None:
      return 0 if all(targets_met.values()) else 1
    return 0 if all(r["status"] == "success" for r in ordered_results()) else 1


if __name__ == "__main__":

  def interrupted(_signum, _frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, interrupted)
  try:
    raise SystemExit(main())
  except KeyboardInterrupt:
    raise SystemExit(130) from None
