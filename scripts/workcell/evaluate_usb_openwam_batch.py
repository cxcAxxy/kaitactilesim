#!/usr/bin/env python3
"""Run the two-group OpenWAM USB-insert evaluation protocol.

Each group starts a fresh OpenWAM server with a different receding-horizon
length, then runs the requested seeds serially in kaitactilesim.  Every
recorded trial uses ``EvaluationVideo`` plus reference-versus-rollout state,
joint and fingertip-force plots.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/workcell/run_usb_openwam_policy.py"
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"


def _text_tail(value: str | bytes | None, limit: int = 4000) -> str:
  if isinstance(value, bytes):
    value = value.decode("utf-8", errors="replace")
  return (value or "")[-limit:]


def _validate_reference_dataset(dataset: Path) -> Path:
  dataset = dataset.expanduser().resolve(strict=True)
  required = (
    dataset / "meta/info.json",
    dataset / "meta/kaihand_source_episodes.jsonl",
  )
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise ValueError(
      f"OpenWAM reference dataset is missing required metadata: {missing}"
    )
  return dataset


def _checkpoint_reference_dataset(checkpoint_dir: Path) -> Path:
  """Read the frozen training dataset root from an OpenWAM config.yaml."""
  config = checkpoint_dir / "config.yaml"
  if not config.is_file():
    raise FileNotFoundError(f"OpenWAM checkpoint config is missing: {config}")
  in_dataloader = False
  value = None
  for line in config.read_text(encoding="utf-8").splitlines():
    if line and not line[0].isspace():
      in_dataloader = line.strip() == "dataloader:"
      continue
    if in_dataloader and line.startswith("  dataset_dir:"):
      value = line.split(":", 1)[1].strip().split(" #", 1)[0].strip()
      break
  if not value:
    raise ValueError(f"OpenWAM checkpoint config has no dataloader.dataset_dir: {config}")
  if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
    value = value[1:-1]
  return _validate_reference_dataset(Path(value))


def _wait_port(host: str, port: int, timeout: float) -> None:
  deadline = time.monotonic() + timeout
  last_error = None
  while time.monotonic() < deadline:
    try:
      with socket.create_connection((host, port), timeout=1.0):
        return
    except OSError as error:
      last_error = error
      time.sleep(1.0)
  raise TimeoutError(f"OpenWAM server did not open {host}:{port}: {last_error}")


def _start_server(args, horizon: int, log_path: Path):
  log = log_path.open("w", encoding="utf-8")
  command = [
    str(args.openwam_python),
    str(args.openwam_root / "scripts/deploy.py"),
    "--ckpt-dir", str(args.ckpt_dir),
    "--ckpt-name", args.ckpt_name,
    "--device", args.device,
    "--host", args.host,
    "--port", str(args.port),
    "--compile-enabled", "false",
    "--inference-horizon", str(horizon),
  ]
  environment = dict(os.environ)
  environment["PYTHONUNBUFFERED"] = "1"
  process = subprocess.Popen(
    command,
    cwd=args.openwam_root,
    stdout=log,
    stderr=subprocess.STDOUT,
    env=environment,
  )
  try:
    deadline = time.monotonic() + args.server_start_timeout
    last_error = None
    while time.monotonic() < deadline:
      if process.poll() is not None:
        raise RuntimeError(
          f"OpenWAM server exited before opening {args.host}:{args.port} "
          f"(returncode={process.returncode}); see {log_path}"
        )
      try:
        with socket.create_connection((args.host, args.port), timeout=1.0):
          break
      except OSError as error:
        last_error = error
        time.sleep(1.0)
    else:
      raise TimeoutError(
        f"OpenWAM server did not open {args.host}:{args.port}: {last_error}"
      )
  except Exception:
    process.terminate()
    process.wait(timeout=20)
    log.close()
    raise
  return process, log


def _stop_server(process, log):
  if process.poll() is None:
    process.terminate()
    try:
      process.wait(timeout=30)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait(timeout=30)
  log.close()


def _run_trial(args, group_dir: Path, seed: int, execute_steps: int, record: bool) -> dict:
  trial_dir = group_dir / f"seed_{seed:03d}"
  command = [
    str(args.kaitactilesim_python), str(RUNNER),
    "--server", f"ws://127.0.0.1:{args.port}",
    "--seed", str(seed),
    "--execute-steps", str(execute_steps),
    "--max-sim-seconds", str(args.max_sim_seconds),
    "--response-timeout", str(args.response_timeout),
    "--output-dir", str(trial_dir),
  ]
  if record:
    if args.reference_dataset is None:
      raise RuntimeError("recorded trials require a reference dataset")
    command.extend(
      (
        "--reference-dataset", str(args.reference_dataset),
        "--reference-episode-index", str(args.reference_episode_index),
      )
    )
  else:
    command.append("--no-record")
  env = dict(os.environ)
  env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
  env["OPENBLAS_NUM_THREADS"] = "1"
  env["OMP_NUM_THREADS"] = "1"
  env["MKL_NUM_THREADS"] = "1"
  env["LP_NUM_THREADS"] = "1"
  env["MPLCONFIGDIR"] = str(group_dir / ".matplotlib-cache")
  env["MUJOCO_GL"] = "egl"
  env["KAIHAND_RENDER_BACKEND"] = "hardware"
  env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(EGL_VENDOR)
  env["NO_PROXY"] = "127.0.0.1,localhost"
  env["no_proxy"] = "127.0.0.1,localhost"
  env.pop("LIBGL_ALWAYS_SOFTWARE", None)
  for key in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
  ):
    env.pop(key, None)
  try:
    result = subprocess.run(
      command, cwd=ROOT, env=env, text=True,
      capture_output=True, timeout=args.trial_wall_limit,
    )
  except subprocess.TimeoutExpired as error:
    return {
      "seed": seed,
      "status": "trial_timeout",
      "timeout_seconds": args.trial_wall_limit,
      "stdout_tail": _text_tail(error.stdout),
      "stderr_tail": _text_tail(error.stderr),
    }
  summary_path = trial_dir / "summary.json"
  if not summary_path.is_file():
    payload = None
  else:
    try:
      payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      return {
        "seed": seed,
        "status": "invalid_summary",
        "returncode": result.returncode,
        "summary": str(summary_path),
        "error": f"{type(error).__name__}: {error}",
      }
  if payload is None:
    return {
      "seed": seed,
      "status": (
        "missing_summary"
        if result.returncode == 0
        else "infrastructure_or_unclassified_error"
      ),
      "returncode": result.returncode,
      "stdout_tail": _text_tail(result.stdout),
      "stderr_tail": _text_tail(result.stderr),
    }
  evaluation = payload.get("evaluation", {})
  trial = {
    "seed": seed,
    "status": payload.get("status"),
    "success": (
      result.returncode == 0
      and payload.get("status") == "success"
      and bool((evaluation or {}).get("success", False))
      and not payload.get("artifact_errors")
    ),
    "failure_stage": (evaluation or {}).get("failure_stage"),
    "error": payload.get("error"),
    "artifact_errors": payload.get("artifact_errors"),
    "comparison_plots": payload.get("comparison_plots"),
    "video": payload.get("video"),
    "summary": str(summary_path),
  }
  if result.returncode != 0:
    trial.update(
      returncode=result.returncode,
      stdout_tail=_text_tail(result.stdout),
      stderr_tail=_text_tail(result.stderr),
    )
  return trial


def _summarize_trials(results: list[dict]) -> dict:
  valid = [
    row for row in results
    if row.get("returncode", 0) == 0
    and not row.get("error")
    and not row.get("artifact_errors")
    and (
      (row.get("status") == "success" and row.get("success") is True)
      or (row.get("status") == "task_not_completed" and row.get("success") is False)
    )
  ]
  successes = sum(row["success"] for row in valid)
  return {
    "valid_trials": len(valid),
    "invalid_trials": len(results) - len(valid),
    "successes": successes,
    "success_rate": successes / len(valid) if valid else None,
  }


def _group(args, execute_steps: int, name: str) -> dict:
  group_dir = args.output_root / f"execute_steps_{name}"
  group_dir.mkdir(parents=False, exist_ok=False)
  server, log = _start_server(args, execute_steps, group_dir / "openwam_server.log")
  try:
    results = []
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.num_trials)):
      results.append(_run_trial(args, group_dir, seed, execute_steps, index < args.video_count))
      print(f"[{name}] trial={index + 1}/{args.num_trials} seed={seed} "
            f"status={results[-1].get('status')} success={results[-1].get('success')}", flush=True)
    payload = {
      "execute_steps": execute_steps,
      "num_trials": args.num_trials,
      "seed_start": args.seed_start,
      "video_count": args.video_count,
      **_summarize_trials(results),
      "results": results,
    }
    (group_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["invalid_trials"]:
      raise RuntimeError(
        f"OpenWAM group has {payload['invalid_trials']} invalid trials; "
        f"see {group_dir / 'summary.json'}"
      )
    return payload
  finally:
    _stop_server(server, log)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt-dir", type=Path, required=True)
  parser.add_argument("--ckpt-name", required=True)
  parser.add_argument("--openwam-root", type=Path, required=True)
  parser.add_argument("--openwam-python", type=Path, required=True)
  parser.add_argument("--kaitactilesim-python", type=Path, required=True)
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8848)
  parser.add_argument("--num-trials", type=int, default=20)
  parser.add_argument("--seed-start", type=int, default=0)
  parser.add_argument("--video-count", type=int, default=20)
  parser.add_argument("--horizon", type=int, default=32)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--response-timeout", type=float, default=600.0)
  parser.add_argument("--trial-wall-limit", type=float, default=1800.0)
  parser.add_argument("--server-start-timeout", type=float, default=900.0)
  parser.add_argument(
    "--reference-dataset",
    type=Path,
    help=(
      "LeRobot v3 training dataset root (default: dataloader.dataset_dir in "
      "the checkpoint config)"
    ),
  )
  parser.add_argument("--reference-episode-index", type=int, default=0)
  args = parser.parse_args(argv)
  if args.num_trials <= 0 or args.seed_start < 0:
    parser.error("num-trials must be positive and seed-start nonnegative")
  if not 0 <= args.video_count <= args.num_trials:
    parser.error("video-count must be in [0, num-trials]")
  if args.horizon <= 16:
    parser.error("horizon must be greater than 16")
  if args.reference_episode_index < 0:
    parser.error("reference-episode-index must be nonnegative")
  args.openwam_root = args.openwam_root.expanduser().resolve()
  args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
  args.openwam_python = args.openwam_python.expanduser().resolve()
  args.kaitactilesim_python = args.kaitactilesim_python.expanduser().resolve()
  args.output_root = args.output_root.expanduser().resolve()
  if args.reference_dataset is not None:
    args.reference_dataset = _validate_reference_dataset(args.reference_dataset)
  elif args.video_count > 0:
    args.reference_dataset = _checkpoint_reference_dataset(args.ckpt_dir)
  checkpoint = args.ckpt_dir / args.ckpt_name
  if not checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {checkpoint}")
  if not EGL_VENDOR.is_file():
    parser.error(f"NVIDIA EGL vendor configuration is missing: {EGL_VENDOR}")
  args.output_root.mkdir(parents=True, exist_ok=False)
  groups = {
    "16": _group(args, 16, "16"),
    f"horizon_{args.horizon}": _group(args, args.horizon, f"horizon_{args.horizon}"),
  }
  summary = {
    "task": "usb-insert",
    "checkpoint": str(args.ckpt_dir / args.ckpt_name),
    "reference_dataset": (
      None if args.reference_dataset is None else str(args.reference_dataset)
    ),
    "reference_episode_index": args.reference_episode_index,
    "groups": groups,
  }
  (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps({name: payload["success_rate"] for name, payload in groups.items()}, indent=2))


if __name__ == "__main__":
  main()
