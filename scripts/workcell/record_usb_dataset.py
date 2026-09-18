#!/usr/bin/env python3
"""Record USB raw HDF5 with 1--2 isolated workers and full 500 Hz tactile."""

from __future__ import annotations

import argparse
import math
import signal
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from kaihand_tactile_env.shared.cameras import SHARED_CAMERA_NAMES, TRAINING_CAMERA_NAMES


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir",
    required=True,
    type=Path,
    help="New directory; raw evidence is never overwritten",
  )
  parser.add_argument(
    "--episodes",
    type=int,
    default=1,
    help="Number of attempts; failed or cancelled attempts are retained",
  )
  parser.add_argument(
    "--workers",
    type=int,
    choices=(1, 2),
    default=1,
    help="Independent USB processes (default 1, maximum 2)",
  )
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Root seed; each index deterministically derives independent object/noise seeds",
  )
  parser.add_argument("--motion-profile", choices=("fast", "baseline"), default="fast")
  parser.add_argument(
    "--camera-hz",
    type=int,
    default=30,
    help="RGB at 320x240; 30 Hz or a divisor of 500; no depth/segmentation",
  )
  parser.add_argument(
    "--include-overhead",
    action="store_true",
    help="Also record synchronized overhead RGB at 320x240; default is head only",
  )
  parser.add_argument(
    "--cameras", nargs="+", choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
    help="Recorded shared cameras; defaults to head/left_wrist/right_wrist",
  )
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  parser.add_argument(
    "--precontact-noise-mm",
    type=float,
    default=0.5,
    help="Precontact XY Gaussian knot sigma per axis; 0 disables",
  )
  args = parser.parse_args(argv)
  if len(set(args.cameras)) != len(args.cameras) or "head" not in args.cameras:
    parser.error("--cameras must contain head and must not contain duplicates")
  if args.episodes <= 0 or args.start_index < 0 or args.seed < 0:
    parser.error("episodes must be positive; start-index and seed must be nonnegative")
  if args.camera_hz <= 0 or (args.camera_hz != 30 and 500 % args.camera_hz):
    parser.error("camera-hz must be 30 or a positive divisor of 500")
  for name in ("xy_jitter_mm", "yaw_jitter_deg", "precontact_noise_mm"):
    value = getattr(args, name)
    if not math.isfinite(value) or value < 0:
      parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
  if args.output_dir.exists() or args.output_dir.is_symlink():
    parser.error("--output-dir must be a new directory")
  return args


@contextmanager
def stop_requests():
  requested = [False]

  def stop(_signal, _frame):
    requested[0] = True

  previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
  for sig in previous:
    signal.signal(sig, stop)
  try:
    yield lambda: requested[0]
  finally:
    for sig, handler in previous.items():
      signal.signal(sig, handler)


def main(argv=None):
  args = parse_args(argv)
  from kaihand_tactile_env.tasks.usb_insert.batch import configure_environment, run_jobs

  configure_environment()
  from kaihand_tactile_env.tasks.usb_insert.recording import (
    UsbRecordJob,
    controller_source_hashes,
    episode_seeds,
    write_json_new,
  )

  args.output_dir.mkdir(parents=True, exist_ok=False)
  raw_dir = args.output_dir / "raw"
  raw_dir.mkdir()
  jobs = [
    UsbRecordJob(
      output=raw_dir / f"usb_{index:06d}.h5",
      episode_index=index,
      root_seed=args.seed,
      xy_jitter_m=args.xy_jitter_mm / 1000,
      yaw_jitter_rad=math.radians(args.yaw_jitter_deg),
      precontact_noise_std_m=args.precontact_noise_mm / 1000,
      motion_profile=args.motion_profile,
      camera_hz=args.camera_hz,
      include_overhead=args.include_overhead,
      cameras=tuple(args.cameras),
      worker_count=min(args.workers, args.episodes),
    )
    for index in range(args.start_index, args.start_index + args.episodes)
  ]
  write_json_new(
    args.output_dir / "run_manifest.json",
    {
      "schema_version": "usb_insert_raw_batch_v1",
      "worker_count": min(args.workers, args.episodes),
      "requested_workers": args.workers,
      "camera_hz": args.camera_hz,
      "root_seed": args.seed,
      "source_sha256": controller_source_hashes(),
      "jobs": [
        {
          **asdict(job),
          "output": str(job.output),
          "object_seed": episode_seeds(job.root_seed, job.episode_index)[0],
          "noise_seed": episode_seeds(job.root_seed, job.episode_index)[1],
        }
        for job in jobs
      ],
    },
  )
  rows = []
  with stop_requests() as should_stop:
    try:
      print(
        f"USB capture: {len(jobs)} attempts, {min(args.workers, args.episodes)} workers, "
        f"{args.camera_hz} Hz RGB / 500 Hz state and tactile",
        flush=True,
      )
      for report in run_jobs(jobs, min(args.workers, args.episodes), should_stop):
        row = {
          key: report[key]
          for key in (
            "episode_index",
            "status",
            "success",
            "raw_path",
            "object_seed",
            "noise_seed",
            "motion_profile",
            "failure_reason",
            "wall_duration_s",
          )
        }
        row["elapsed_simulation_s"] = report["outcome"].get("elapsed_simulation_s")
        rows.append(row)
        write_json_new(
          args.output_dir / f"progress_{len(rows):04d}.json", {"episodes": rows}
        )
        print(f"  status={row['status']} raw={row['raw_path']}", flush=True)
    finally:
      complete = len(rows) == len(jobs) and all(
        row["status"] != "cancelled" for row in rows
      )
      summary = {
        "schema_version": "usb_insert_raw_batch_v1",
        "complete": complete,
        "planned_attempts": len(jobs),
        "recorded_attempts": len(rows),
        "successful_episodes": sum(row["success"] for row in rows),
        "episodes": sorted(rows, key=lambda row: row["episode_index"]),
        "worker_count": min(args.workers, args.episodes),
        "not_started_indices": sorted(
          set(job.episode_index for job in jobs)
          - {row["episode_index"] for row in rows}
        ),
        "export_policy": "Only finalized .h5 with verified success and valid terminal capture; .h5.partial is retained diagnostic raw data, never training input",
      }
      write_json_new(args.output_dir / "summary.json", summary)
  if should_stop() or any(row["status"] == "cancelled" for row in rows):
    return 130
  return 0 if complete and all(row["success"] for row in rows) else 1


if __name__ == "__main__":
  raise SystemExit(main())
