"""Continue the requested Poker evaluation matrix as a durable serial job."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "scripts/workcell/evaluate_poker_policy_batch.py"
MODEL_PROJECT = Path("/cpfs_infra/user/chenxianchi/code/egosteer")
MODEL_PYTHON = Path(
  "/cpfs_infra/user/chenxianchi/miniconda3/envs/egosteer_touch/bin/python"
)
PORT = 18780
SERVER = f"ws://127.0.0.1:{PORT}"


def read_json(path: Path) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (FileNotFoundError, json.JSONDecodeError):
    return {}
  return value if isinstance(value, dict) else {}


def process_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


def wait_process(pid: int, *, poll_s: float = 10.0) -> None:
  while process_alive(pid):
    time.sleep(poll_s)


def wait_port(*, open_expected: bool, timeout_s: float) -> None:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    with socket.socket() as sock:
      sock.settimeout(1.0)
      is_open = sock.connect_ex(("127.0.0.1", PORT)) == 0
    if is_open == open_expected:
      return
    time.sleep(2.0)
  state = "open" if open_expected else "closed"
  raise TimeoutError(f"port {PORT} did not become {state}")


def choose_output(base: Path) -> Path:
  if not base.exists():
    return base
  if read_json(base / "summary.json").get("complete") is True:
    return base
  for index in range(1, 100):
    candidate = base.with_name(f"{base.name}_retry{index:02d}")
    if not candidate.exists():
      return candidate
  raise RuntimeError(f"too many retry directories for {base}")


def run_batch(
  output_root: Path,
  manifest: Path,
  *,
  step: int,
  execute_steps: int,
  count: int,
) -> Path:
  base = output_root / f"step{step:04d}_exec{execute_steps}_N{count}"
  if read_json(base / "summary.json").get("complete") is True:
    print(f"[skip] complete batch {base}", flush=True)
    return base
  output = choose_output(base)
  command = [
    sys.executable,
    str(BATCH),
    "--server",
    SERVER,
    "--deployment-manifest",
    str(manifest),
    "--seeds",
    *[str(seed) for seed in range(count)],
    "--execute-steps",
    str(execute_steps),
    "--max-sim-seconds",
    "50",
    "--trial-wall-limit",
    "1800",
    "--video-count",
    "3",
    "--record-fps",
    "10",
    "--output-dir",
    str(output),
  ]
  print(f"[batch] start {output}", flush=True)
  completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
  if completed.returncode != 0:
    raise RuntimeError(f"batch failed with return code {completed.returncode}: {output}")
  summary = read_json(output / "summary.json")
  if summary.get("complete") is not True:
    raise RuntimeError(f"batch did not produce a complete summary: {output}")
  print(f"[batch] complete {output}", flush=True)
  return output


def stop_server(pid: int) -> None:
  if process_alive(pid):
    print(f"[server] stopping pid={pid}", flush=True)
    os.kill(pid, signal.SIGTERM)
  wait_port(open_expected=False, timeout_s=120.0)


def start_server(
  config: Path, manifest: Path, log_path: Path
) -> subprocess.Popen:
  cameras = read_json(manifest).get("observation_contract", {}).get(
    "cameras", ["head"]
  )
  if cameras not in (["head"], ["head", "right_wrist"]):
    raise RuntimeError(f"unsupported EgoSteer cameras in {manifest}: {cameras}")
  env = {
    **os.environ,
    "EGOSTEER_INFERENCE_CONFIG": str(config),
    "EGOSTEER_CAMERA_VIEWS": ",".join(cameras),
    "HF_HOME": "/cpfs_infra/user/chenxianchi/.cache/hf",
    "HUGGINGFACE_HUB_CACHE": "/cpfs_infra/user/chenxianchi/.cache/hf/hub",
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
  }
  log = log_path.open("a", encoding="utf-8")
  process = subprocess.Popen(
    [str(MODEL_PYTHON), "-m", "src.serving.serve_policy"],
    cwd=MODEL_PROJECT,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
  )
  log.close()
  print(f"[server] starting pid={process.pid} config={config}", flush=True)
  try:
    wait_port(open_expected=True, timeout_s=600.0)
  except BaseException:
    if process.poll() is None:
      process.terminate()
    raise
  if process.poll() is not None:
    raise RuntimeError(f"server exited during startup: {config}")
  return process


def write_state(path: Path, **values) -> None:
  state = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    **read_json(path),
    **values,
  }
  path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--current-batch-pid", type=int, required=True)
  parser.add_argument("--current-server-pid", type=int, required=True)
  args = parser.parse_args()
  output_root = args.output_root.expanduser().resolve()
  state_path = output_root / "supervisor_state.json"
  manifest1200 = output_root / "deployment_frozen_step1200/deployment_manifest.json"
  manifest1400 = output_root / "deployment_frozen_step1400/deployment_manifest.json"
  config1400 = output_root / "deployment_frozen_step1400/inference.yaml"
  for path in (manifest1200, manifest1400, config1400):
    if not path.is_file():
      raise FileNotFoundError(path)

  server_pid = args.current_server_pid
  try:
    write_state(
      state_path,
      status="waiting_current_step1200_exec5_N50",
      blocked_checkpoint_0800=(
        "original top-k directory was deleted after step1600; exact bytes required"
      ),
    )
    wait_process(args.current_batch_pid)
    current = output_root / "step1200_exec5_N50"
    if read_json(current / "summary.json").get("complete") is not True:
      run_batch(output_root, manifest1200, step=1200, execute_steps=5, count=50)

    write_state(state_path, status="running_step1200_horizon_batches")
    run_batch(output_root, manifest1200, step=1200, execute_steps=32, count=20)
    run_batch(output_root, manifest1200, step=1200, execute_steps=32, count=50)

    stop_server(server_pid)
    process = start_server(
      config1400,
      manifest1400,
      output_root / "deployment_frozen_step1400/server.log",
    )
    server_pid = process.pid
    write_state(
      state_path,
      status="running_step1400_batches",
      current_server_pid=server_pid,
    )
    for execute_steps in (5, 32):
      for count in (20, 50):
        run_batch(
          output_root,
          manifest1400,
          step=1400,
          execute_steps=execute_steps,
          count=count,
        )
    stop_server(server_pid)
    server_pid = 0
    write_state(
      state_path,
      status="available_batches_complete",
      completed_checkpoints=[1200, 1400],
      blocked_checkpoints=[800],
    )
  except BaseException as error:
    write_state(
      state_path,
      status="failed",
      error=f"{type(error).__name__}: {error}",
      current_server_pid=server_pid,
    )
    raise


if __name__ == "__main__":
  main()
