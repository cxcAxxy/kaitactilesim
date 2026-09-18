#!/usr/bin/env python3
"""Serial, renderer-free full-task checks of bounded initial card randomization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

for _name in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "LP_NUM_THREADS",
):
  os.environ[_name] = "1"

import numpy as np  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.recording import wait_until_object_stable  # noqa: E402
from kaihand_tactile_env.tasks.poker_draw.acceptance import (  # noqa: E402
  ACCEPTANCE_POLICIES,
  STRICT_FORCE_POLICY,
  accept_edge,
)
from kaihand_tactile_env.tasks.poker_draw.mid_full import (  # noqa: E402
  MID_FORCE_PER_FINGER_N,
  MID_FORCE_SETTINGS,
  MidForcePokerExecutor,
  middle_force_simulation,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (  # noqa: E402
  DEFAULT_PRECONTACT_STD_RAD,
  PRECONTACT_PRESET,
  precontact_force_simulation,
  precontact_noise_settings,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (  # noqa: E402
  DEFAULT_XY_JITTER_M as PRECONTACT_XY_JITTER_M,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (  # noqa: E402
  DEFAULT_YAW_JITTER_RAD as PRECONTACT_YAW_JITTER_RAD,
)
from kaihand_tactile_env.tasks.poker_draw.randomization import (  # noqa: E402
  RANDOMIZED_PRESET,
  reset_randomized_card,
  validate_randomization_bounds,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawPlanner  # noqa: E402


def _json(path, payload):
  def convert(value):
    if isinstance(value, np.ndarray):
      return value.tolist()
    if isinstance(value, np.generic):
      return value.item()
    raise TypeError(type(value).__name__)

  with path.open("x", encoding="utf-8") as stream:
    json.dump(
      payload,
      stream,
      ensure_ascii=False,
      indent=2,
      sort_keys=True,
      default=convert,
      allow_nan=False,
    )
    stream.write("\n")


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--xy-mm", type=float)
  parser.add_argument("--yaw-deg", type=float)
  parser.add_argument(
    "--precontact-noise",
    action="store_true",
    help="Opt in to tiny right-arm command noise, permanently off after first tactile contact",
  )
  parser.add_argument(
    "--precontact-noise-std-deg",
    type=float,
    help="Precontact Gaussian process sigma in degrees per arm joint, [0,0.05]; zero for paired control",
  )
  parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
  parser.add_argument("--acceptance-policy", choices=ACCEPTANCE_POLICIES,
                      default=STRICT_FORCE_POLICY)
  parser.add_argument(
    "--fixed-offset-mm-deg",
    type=float,
    nargs=3,
    action="append",
    metavar=("DX_MM", "DY_MM", "YAW_DEG"),
    help="Named deterministic boundary checks instead of RNG draws, repeatable; never dataset samples",
  )
  parser.add_argument("--trial-wall-limit", type=float, default=180.0)
  args = parser.parse_args(argv)
  if args.xy_mm is None:
    args.xy_mm = PRECONTACT_XY_JITTER_M * 1000 if args.precontact_noise else 2.0
  if args.yaw_deg is None:
    args.yaw_deg = (
      math.degrees(PRECONTACT_YAW_JITTER_RAD) if args.precontact_noise else 0.0
    )
  if args.precontact_noise_std_deg is not None and not args.precontact_noise:
    parser.error("--precontact-noise-std-deg requires --precontact-noise")
  if args.precontact_noise_std_deg is None:
    args.precontact_noise_std_deg = math.degrees(DEFAULT_PRECONTACT_STD_RAD)
  try:
    validate_randomization_bounds(args.xy_mm / 1000, math.radians(args.yaw_deg))
    if args.precontact_noise:
      precontact_noise_settings(math.radians(args.precontact_noise_std_deg))
  except ValueError as error:
    parser.error(str(error))
  if (
    not args.seeds
    or len(set(args.seeds)) != len(args.seeds)
    or min(args.seeds) < 0
    or len(args.seeds) > 6
  ):
    parser.error("use 1 to 6 distinct nonnegative seeds for a bounded serial check")
  if not math.isfinite(args.trial_wall_limit) or not 10 <= args.trial_wall_limit <= 300:
    parser.error("trial wall limit must be in [10,300] seconds")
  if args.fixed_offset_mm_deg:
    if len(args.fixed_offset_mm_deg) > 6:
      parser.error("at most 6 boundary cases per invocation")
    for dx, dy, yaw in args.fixed_offset_mm_deg:
      if (
        not np.all(np.isfinite([dx, dy, yaw]))
        or max(abs(dx), abs(dy)) > args.xy_mm + 1e-12
        or abs(yaw) > args.yaw_deg + 1e-12
      ):
        parser.error("fixed boundary offsets must be finite and within declared bounds")
  return args


class _Observer:
  def __init__(self, simulation, stream, trial_index, wall_limit):
    self.sim = simulation
    self.writer = csv.writer(stream)
    self.writer.writerow(
      [
        "state_time_s",
        "pose_time_s",
        "phase",
        "card_x_m",
        "card_y_m",
        "card_z_m",
        "card_qw",
        "card_qx",
        "card_qy",
        "card_qz",
        "tool_x_m",
        "tool_y_m",
        "tool_z_m",
        "drive_active",
        "drive_actual_fx_n",
      ]
    )
    self.trial_index = trial_index
    self.started = time.monotonic()
    self.wall_limit = wall_limit
    self.next_sample = 0.0
    self.next_progress = 5.0
    self.last_phase = "reset"

  def __call__(self, simulation, phase):
    self.last_phase = phase
    elapsed = time.monotonic() - self.started
    if elapsed > self.wall_limit:
      raise TimeoutError(f"trial exceeded bounded wall time {self.wall_limit:g}s")
    if simulation.data.time + 1e-9 >= self.next_sample:
      pose = simulation.object_pose("card")
      tool, _ = simulation.current_pose_matrix("right")
      active = simulation.drive_limit_n is not None
      drive = simulation.drive_state()
      self.writer.writerow(
        [
          float(simulation.data.time),
          simulation.observation_time,
          phase,
          *pose,
          *tool,
          int(active),
          drive["drive_actual_fx_n"] if active else 0.0,
        ]
      )
      self.next_sample = float(simulation.data.time) + 0.05
    if simulation.data.time >= self.next_progress:
      print(
        f"trial={self.trial_index} sim={simulation.data.time:.3f}s phase={phase} wall={elapsed:.1f}s",
        flush=True,
      )
      self.next_progress = float(simulation.data.time) + 5.0


def _accepted(result, executor):
  edge, handoff = executor.edge_outcome, executor.handoff_outcome
  return bool(
    result.success
    and edge
    and all(
      edge.get(k) is True
      for k in ("target_reached", "held_at_edge")
    )
    and accept_edge(edge, getattr(executor, "acceptance_policy", STRICT_FORCE_POLICY))
    and (result.slide_press_control_qualified
         or getattr(executor, "acceptance_policy", STRICT_FORCE_POLICY) != STRICT_FORCE_POLICY)
    and handoff
    and handoff.get("completed") is True
  )


@contextmanager
def _termination_guard():
  """Let SIGTERM archive the current diagnostic without calling it success."""

  def interrupted(signum, frame):
    del signum, frame
    raise KeyboardInterrupt("randomization screen terminated before completion")

  previous = signal.signal(signal.SIGTERM, interrupted)
  try:
    yield
  finally:
    signal.signal(signal.SIGTERM, previous)


def _run_case(sim, args, index, seed, fixed):
  started = time.monotonic()
  row = {
    "trial_index": index,
    "seed": seed,
    "success": False,
    "error": None,
    "initial_card_randomization": None,
    "task_result": None,
  }
  executor = None
  noise_configured = False
  interrupted = None
  trace_path = args.output_dir / f"trial_{index:03d}_trajectory.csv"
  with trace_path.open("x", encoding="utf-8", newline="") as stream:
    observer = _Observer(sim, stream, index, args.trial_wall_limit)
    try:
      reset = reset_randomized_card(
        sim, seed, args.xy_mm / 1000, math.radians(args.yaw_deg), fixed_offset=fixed
      )
      row["initial_card_randomization"] = reset
      print(
        f"starting trial={index} seed={seed} xy_mm={np.asarray(reset['sampled_offset_xy_m']) * 1000} yaw_deg={math.degrees(reset['sampled_yaw_offset_rad']):.5f}",
        flush=True,
      )
      observer(sim, "reset")
      plan = PokerDrawPlanner(sim).plan()
      if args.precontact_noise:
        sim.configure_precontact_noise(
          seed=seed, std_rad=math.radians(args.precontact_noise_std_deg)
        )
        noise_configured = True
      executor = MidForcePokerExecutor(sim, observer=observer, acceptance_policy=args.acceptance_policy)
      result = executor.execute(plan)
      stability = wait_until_object_stable(sim, "card", observer=observer)
      result = executor.refresh_terminal_result(result)
      row.update(
        success=_accepted(result, executor),
        task_result=asdict(result),
        terminal_stability=asdict(stability),
      )
    except (RuntimeError, ValueError, TimeoutError) as error:
      row.update(error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt as error:
      interrupted = error
      row.update(error=f"KeyboardInterrupt: {error}", interrupted=True)
    row.update(
      edge_outcome=getattr(executor, "edge_outcome", None),
      handoff_outcome=getattr(executor, "handoff_outcome", None),
      experimental_control=executor.control_metadata()
      if executor is not None
      else None,
      last_phase=observer.last_phase,
      simulation_duration_s=float(sim.data.time),
      wall_seconds=time.monotonic() - started,
      final_card_pose_wxyz=sim.object_pose("card"),
      trajectory_csv=str(trace_path),
    )
    if noise_configured:
      noise_trace_path = args.output_dir / f"trial_{index:03d}_precontact_trace.npz"
      with noise_trace_path.open("xb") as noise_stream:
        np.savez(noise_stream, **sim.precontact_noise_trace())
      row.update(
        precontact_noise=sim.precontact_noise_metadata(),
        precontact_trace_npz=str(noise_trace_path),
      )
  _json(args.output_dir / f"trial_{index:03d}.json", row)
  print(
    f"finished trial={index} success={row['success']} error={row['error']}", flush=True
  )
  if interrupted is not None:
    raise interrupted
  return row


def main(argv=None):
  args = _parse_args(argv)
  args.output_dir = args.output_dir.expanduser().resolve()
  args.output_dir.mkdir(parents=True, exist_ok=False)
  import kaihand_tactile_env.tasks.poker_draw.randomization as module

  source_dir = Path(module.__file__).parent
  sources = [
    Path(__file__),
    *(
      source_dir / name
      for name in (
        "randomization.py",
        "mid_full.py",
        "pressure_window.py",
        "task.py",
        "config.py",
        "press_control.py",
        "acceptance.py",
      )
    ),
  ]
  shared_dir = source_dir.parents[1] / "shared"
  sources.extend(shared_dir / name for name in ("simulation.py", "tactile.py"))
  if args.precontact_noise:
    sources.append(source_dir / "precontact_noise.py")
  protocol = {
    "schema_version": "poker-randomization-screen-v1",
    "created_utc": datetime.now(UTC).isoformat(),
    "preset": PRECONTACT_PRESET if args.precontact_noise else RANDOMIZED_PRESET,
    "acceptance_policy": args.acceptance_policy,
    "xy_jitter_m": args.xy_mm / 1000,
    "yaw_jitter_rad": math.radians(args.yaw_deg),
    "seeds": args.seeds,
    "fixed_offsets_mm_deg": args.fixed_offset_mm_deg,
    "pressure_window": asdict(MID_FORCE_SETTINGS),
    "press_force_per_finger_n": MID_FORCE_PER_FINGER_N,
    "observation_noise": None,
    "action_noise": precontact_noise_settings(
      math.radians(args.precontact_noise_std_deg)
    )
    if args.precontact_noise
    else None,
    "headless": True,
    "rendering": False,
    "serial": True,
    "model_fingerprint": model_fingerprint(default_model_path("poker-draw")),
    "source_sha256": {
      str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    },
    "success_semantics": "versioned acceptance policy; original physical/terminal guards; separate strict pressure quality; no retries or force retuning",
    "coverage_scope": "finite tested cases only, not a probability estimate or guarantee over the continuous range",
  }
  _json(args.output_dir / "protocol.json", protocol)
  cases = [(seed, None) for seed in args.seeds]
  if args.fixed_offset_mm_deg:
    cases = [
      (0, (dx / 1000, dy / 1000, math.radians(yaw)))
      for dx, dy, yaw in args.fixed_offset_mm_deg
    ]
  rows = []
  factory = (
    precontact_force_simulation if args.precontact_noise else middle_force_simulation
  )
  with (
    _termination_guard(),
    factory(default_model_path("poker-draw")) as (sim, contacts),
  ):
    for index, (seed, fixed) in enumerate(cases, 1):
      rows.append(_run_case(sim, args, index, seed, fixed))
  summary = {
    **protocol,
    "contact_model": contacts,
    "trials": rows,
    "passed_count": sum(row["success"] for row in rows),
    "trial_count": len(rows),
    "all_passed": all(row["success"] for row in rows),
  }
  _json(args.output_dir / "summary.json", summary)
  print(
    f"summary={args.output_dir / 'summary.json'} passed={summary['passed_count']}/{len(rows)}",
    flush=True,
  )
  return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
