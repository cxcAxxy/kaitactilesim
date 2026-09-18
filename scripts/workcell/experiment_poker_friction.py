#!/usr/bin/env python3
"""Run serial table-card friction trials using the production force controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

# Keep numerical libraries single-threaded before importing NumPy/MuJoCo.
for variable in (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "LP_NUM_THREADS",
):
  os.environ.setdefault(variable, "1")
# This entry point never opens a viewer. Use bounded CPU offscreen rendering
# by default when --record is selected; explicit environment overrides remain
# available. No GL context is created for numeric-only trials.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation  # noqa: E402
from kaihand_tactile_env.tasks.poker_draw.config import (  # noqa: E402
  DEFAULT_PRESS_FORCE_PER_FINGER_N,
)
from kaihand_tactile_env.tasks.poker_draw.friction import (  # noqa: E402
  BASELINE_TABLE_CARD_FRICTION,
  TABLE_CARD_PAIR_NAME,
  PokerFrictionExperiment,
  PokerFrictionTrial,
  model_with_table_card_friction,
  press_control_metadata,
  set_table_card_friction,
)
from kaihand_tactile_env.tasks.poker_draw.task import (  # noqa: E402
  PokerDrawExecutor,
  PokerDrawPlanner,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--protocol",
    choices=("force", "legacy-preload"),
    default="force",
    help="force uses the same per-finger feedback as view/record; legacy is historical.",
  )
  parser.add_argument(
    "--friction",
    type=float,
    nargs="+",
    default=None,
    help="Pair-local table-card sliding coefficients.",
  )
  parser.add_argument(
    "--press-offset-degrees",
    type=float,
    nargs="+",
    default=None,
    help="Only for --protocol legacy-preload; joint offsets, not a force setpoint.",
  )
  parser.add_argument(
    "--press-force-per-finger",
    "--press-force",
    dest="press_forces_n",
    type=float,
    nargs="+",
    help="Positive normal-force targets in N per finger (force protocol only).",
  )
  parser.add_argument("--slide-distance", type=float, help="Legacy protocol only.")
  parser.add_argument("--slide-step", type=float, help="Legacy protocol only.")
  parser.add_argument("--slide-speed", type=float, help="Legacy protocol only.")
  parser.add_argument("--settle-seconds", type=float, help="Legacy protocol only.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--scene-path",
    type=Path,
    default=default_model_path("poker-draw"),
  )
  parser.add_argument("--json", type=Path, help="Optional summary JSON output.")
  parser.add_argument("--csv", type=Path, help="Optional summary CSV output.")
  parser.add_argument(
    "--diagnostics-dir",
    type=Path,
    help=(
      "New output directory for per-step CSV, force/slip PNG curves and per-trial "
      "JSON (force protocol only). Defaults --json to DIRECTORY/summary.json."
    ),
  )
  parser.add_argument(
    "--record",
    action="store_true",
    help="Save a front/overhead/right Fn-Ft MP4 for each force trial; requires --diagnostics-dir.",
  )
  parser.add_argument(
    "--record-fps",
    type=float,
    default=10.0,
    help="Simulation-time video frame rate, >0 and <=30 (default: 10). No preview window.",
  )
  args = parser.parse_args(argv)
  if not math.isfinite(args.record_fps) or not 0 < args.record_fps <= 30:
    parser.error("--record-fps must be finite and in (0, 30]")
  if args.record and (args.protocol != "force" or args.diagnostics_dir is None):
    parser.error("--record requires --protocol force and --diagnostics-dir")
  legacy_defaults = {
    "press_offset_degrees": (0.0, 2.0),
    "slide_distance": 0.135,
    "slide_step": 0.004,
    "slide_speed": 0.055,
    "settle_seconds": 0.08,
  }
  if args.protocol == "force":
    if any(getattr(args, name) is not None for name in legacy_defaults):
      parser.error("joint-offset/slide overrides require --protocol legacy-preload")
    args.press_forces_n = args.press_forces_n or (DEFAULT_PRESS_FORCE_PER_FINGER_N,)
    if any(not math.isfinite(force) or force <= 0 for force in args.press_forces_n):
      parser.error("--press-force-per-finger values must be finite and positive")
    args.friction = args.friction or (0.10, 0.45, 0.70)
    trial_count = len(args.friction) * len(args.press_forces_n)
  else:
    if args.diagnostics_dir is not None:
      parser.error("--diagnostics-dir requires --protocol force")
    if args.press_forces_n is not None:
      parser.error("--press-force-per-finger requires --protocol force")
    for name, default in legacy_defaults.items():
      if getattr(args, name) is None:
        setattr(args, name, default)
    args.friction = args.friction or (0.10, 0.90, 1.15, 1.30)
    trial_count = len(args.friction) * len(args.press_offset_degrees)
  if any(not math.isfinite(mu) or mu < 0 for mu in args.friction):
    parser.error("--friction values must be finite and non-negative")
  if trial_count < 1 or trial_count > 10:
    parser.error("the bounded experiment requires between 1 and 10 trials")
  return args


def _json_safe(value: Any) -> Any:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def _write_json(path: Path, document: dict[str, Any]) -> None:
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("x", encoding="utf-8") as file:
    file.write(json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("x", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(
      {
        key: json.dumps(_json_safe(value), separators=(",", ":"))
        if isinstance(value, (dict, tuple, list))
        else value
        for key, value in row.items()
      }
      for row in rows
    )


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  _prepare_outputs(args)
  if args.protocol == "force":
    _run_force_trials(args)
    return
  _run_legacy_trials(args)


def _prepare_outputs(args: argparse.Namespace) -> None:
  """Reserve a new diagnostics directory before loading any simulation model."""
  diagnostic_outputs: tuple[Path, ...] = ()
  if args.diagnostics_dir is not None:
    directory = args.diagnostics_dir.expanduser().resolve()
    if directory.exists():
      raise FileExistsError(f"diagnostics directory must be new: {directory}")
    args.diagnostics_dir = directory
    if args.json is None:
      args.json = directory / "summary.json"
    suffixes = ("_timeseries.csv", "_curves.png", ".json")
    if args.record:
      suffixes += ("_video.mp4", "_video.json")
    diagnostic_outputs = tuple(
      directory / f"trial_{index:03d}{suffix}"
      for index in range(1, len(args.friction) * len(args.press_forces_n) + 1)
      for suffix in suffixes
    )
  outputs = tuple(path.expanduser().resolve() for path in (args.json, args.csv) if path)
  all_outputs = (*outputs, *diagnostic_outputs)
  if len(set(all_outputs)) != len(all_outputs):
    raise ValueError("summary and diagnostic outputs must resolve to different files")
  if args.diagnostics_dir in outputs:
    raise ValueError("an output file cannot also be the diagnostics directory")
  for path in all_outputs:
    if any(path in other.parents for other in all_outputs if other != path):
      raise ValueError("an output file cannot be another output's parent directory")
  existing = tuple(path for path in outputs if path.exists())
  if existing:
    raise FileExistsError(f"refusing to overwrite experiment output: {existing[0]}")
  if args.diagnostics_dir is not None:
    args.diagnostics_dir.mkdir(parents=True, exist_ok=False)


def _run_legacy_trials(args: argparse.Namespace) -> None:
  trials = tuple(
    PokerFrictionTrial(
      table_card_friction=friction,
      press_distal_offset_degrees=offset,
      slide_distance_m=args.slide_distance,
      slide_step_m=args.slide_step,
      slide_speed_m_s=args.slide_speed,
      settle_seconds=args.settle_seconds,
      seed=args.seed,
    )
    for friction in args.friction
    for offset in args.press_offset_degrees
  )

  with model_with_table_card_friction(
    args.scene_path, trials[0].table_card_friction
  ) as model:
    # Exactly one model/data pair is reused serially for every condition.
    simulation = ArmHandSimulation(model, scene="poker-draw")
    experiment = PokerFrictionExperiment(simulation)
    rows: list[dict[str, object]] = []
    for index, trial in enumerate(trials, start=1):
      print(
        f"starting trial={index}/{len(trials)} "
        f"mu={trial.table_card_friction:.3f} "
        f"preload={trial.press_distal_offset_degrees:.2f}deg",
        flush=True,
      )
      row = experiment.run_trial(trial).as_dict()
      rows.append(row)
      print(
        f"finished trial={index}/{len(trials)} "
        f"press={row['initial_normal_force_mean_n']:.4f}N "
        f"card_dx={1000.0 * row['card_displacement_m']:.1f}mm "
        f"slip={1000.0 * row['fingertip_card_relative_slip_m']:.1f}mm "
        f"overhang={row['maximum_overhang_fraction']:.3f} "
        f"tactile_depth={1.0e3 * row['maximum_genesis_probe_depth_m']:.3f}mm "
        f"outcome={row['outcome']} failure={row['failure']}",
        flush=True,
      )

  document = {
    "schema_version": 2,
    "protocol": "legacy-preload",
    "created_utc": datetime.now(UTC).isoformat(),
    "scene": "poker-draw",
    "scene_path": str(args.scene_path.expanduser().resolve()),
    "scene_model_fingerprint": model_fingerprint(args.scene_path),
    "mujoco_version": mujoco.__version__,
    "model_initializations": 1,
    "headless": True,
    "state_noise": False,
    "object_xy_jitter_m": 0.0,
    "object_yaw_jitter_rad": 0.0,
    "seed": args.seed,
    "slide_distance_m": args.slide_distance,
    "slide_step_m": args.slide_step,
    "slide_speed_m_s": args.slide_speed,
    "settle_seconds": args.settle_seconds,
    "pair_name": TABLE_CARD_PAIR_NAME,
    "baseline_pair_friction": BASELINE_TABLE_CARD_FRICTION,
    "pressure_control": (
      "distal_joint4_offset_degrees; actual normal force is measured, not commanded"
    ),
    "trials": rows,
  }
  if args.json is not None:
    _write_json(args.json, document)
    print(f"wrote JSON: {args.json.expanduser().resolve()}")
  if args.csv is not None:
    _write_csv(args.csv, rows)
    print(f"wrote CSV: {args.csv.expanduser().resolve()}")


def _run_force_trials(args: argparse.Namespace) -> None:
  """Use exactly the production approach, force establishment and guarded slide."""
  from kaihand_tactile_env.tasks.poker_draw.telemetry import PokerFrictionTelemetry

  trials = tuple((mu, force) for mu in args.friction for force in args.press_forces_n)
  rows: list[dict[str, object]] = []
  measurement_sources = _measurement_source_hashes()
  scene_fingerprint = model_fingerprint(args.scene_path)
  with model_with_table_card_friction(args.scene_path, trials[0][0]) as model:
    simulation = ArmHandSimulation(model, scene="poker-draw")
    pad_geom_ids = tuple(
      simulation.model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id
      for finger in ("index", "middle", "ring", "pinky")
    )
    for index, (mu, force) in enumerate(trials, start=1):
      print(
        f"starting trial={index}/{len(trials)} mu={mu:.3f} "
        f"target={force:.3f}N/finger protocol=production-force",
        flush=True,
      )
      simulation.reset(seed=args.seed, object_xy_jitter=0.0, object_yaw_jitter=0.0)
      set_table_card_friction(simulation.model, mu)
      plan = PokerDrawPlanner(simulation).plan()
      telemetry = PokerFrictionTelemetry(simulation, target_force_n=force)
      trace_path = (
        args.diagnostics_dir / f"trial_{index:03d}_timeseries.csv"
        if args.diagnostics_dir is not None
        else None
      )
      # Buffered streaming: retain partial numeric observations if interrupted,
      # never retain a video or the whole trace in memory while simulating.
      video_artifacts = None
      with ExitStack() as resources:
        trace_file = (
          resources.enter_context(trace_path.open("x", encoding="utf-8", newline=""))
          if trace_path is not None
          else None
        )
        video = None
        if args.record:
          from kaihand_tactile_env.shared.task_video import TaskVideoRecorder

          assert args.diagnostics_dir is not None
          video_path = args.diagnostics_dir / f"trial_{index:03d}_video.mp4"
          video = resources.enter_context(
            TaskVideoRecorder(
              simulation,
              video_path,
              fps=args.record_fps,
              preview=False,
              metadata={
                "scene": "poker-draw",
                "scope": "press_slide_edge_only_not_full_draw",
                "protocol": "production-per-finger-force",
                "table_card_friction_override": mu,
                "press_force_per_finger_n": force,
                "seed": args.seed,
                "object_xy_jitter": 0.0,
                "object_yaw_jitter": 0.0,
                "state_noise": False,
                "scene_model_fingerprint": scene_fingerprint,
                "press_control": press_control_metadata(),
                "measurement_source_sha256": measurement_sources,
                "mujoco_version": mujoco.__version__,
                "rendering_backend": os.environ.get("MUJOCO_GL", "default"),
                "software_rendering_requested": os.environ.get("LIBGL_ALWAYS_SOFTWARE")
                == "1",
                "completion_semantics": "returned task failure is a completed experiment; task_outcome retains the reason",
              },
            )
          )
          video.observe(simulation, "initial")
        observer = _ForceTrialObserver(telemetry, pad_geom_ids, trace_file, video=video)
        executor = PokerDrawExecutor(
          simulation, press_force_per_finger_n=force, observer=observer
        )
        result = executor.execute_slide_only(plan)
        # Task safety checks may raise before calling the observer.  Preserve
        # that last solver sample under an explicit label, not a guessed phase.
        if simulation.data.time > observer.last_time_s + 1e-9:
          observer(simulation, "terminal_unobserved")
        if video is not None:
          video_result = asdict(result)
          video_result["stage_outcome"] = _stage_outcome(
            video_result, observer.phase_counts, observer.last_task_phase
          )
          video_result["telemetry"] = telemetry.summary()
          video.set_outcome(video_result)
          # A returned physical failure is complete experimental evidence,
          # unlike an interrupted recorder/renderer. The reason stays in outcome.
          video_path, sidecar_path = video.finish(success=bool(result.success))
          video_artifacts = {"mp4": str(video_path), "json": str(sidecar_path)}
      row = asdict(result)
      for name in ("initial_card_pose", "edge_card_pose"):
        row[name] = row[name].tolist()
      row["table_card_friction"] = mu
      row["video_artifacts"] = video_artifacts
      row["simulation_duration_s"] = float(simulation.data.time)
      pad_displacement = card_displacement = 0.0
      slide_positions = observer.slide_positions
      if slide_positions:
        pad_displacement = slide_positions[0][0] - slide_positions[-1][0]
        card_displacement = slide_positions[0][1] - slide_positions[-1][1]
      row["slide_fingertip_toward_robot_displacement_m"] = pad_displacement
      row["slide_card_toward_robot_displacement_m"] = card_displacement
      row["slide_fingertip_card_relative_slip_m"] = pad_displacement - card_displacement
      row["stage_outcome"] = _stage_outcome(
        row, observer.phase_counts, observer.last_task_phase
      )
      row["telemetry"] = telemetry.summary()
      row["measurement"] = telemetry.metadata()
      row["diagnostic_sustained_slip_fingers"] = [
        finger
        for finger, measured in row["telemetry"]["slide_and_edge"]["fingers"].items()
        if measured["first_sustained_slip"] is not None
      ]
      row["diagnostic_artifacts"] = None
      if trace_path is not None:
        curve_path = (
          trace_path.with_name(f"trial_{index:03d}_curves.png")
          if observer.phase_counts
          else None
        )
        trial_json = trace_path.with_name(f"trial_{index:03d}.json")
        row["diagnostic_artifacts"] = {
          "timeseries_csv": str(trace_path),
          "curves_png": str(curve_path) if curve_path is not None else None,
          "curves_omission_reason": "no completed physics samples"
          if curve_path is None
          else None,
          "trial_json": str(trial_json),
        }
        # Write the measured result before plotting: plotting errors must not
        # erase the physical trial or require running it again.
        _write_json(
          trial_json,
          {
            "schema_version": 1,
            "scope": "press_slide_edge_only_not_full_draw",
            "scene_model_fingerprint": scene_fingerprint,
            "mujoco_version": mujoco.__version__,
            "seed": args.seed,
            "object_xy_jitter_m": 0.0,
            "object_yaw_jitter_rad": 0.0,
            "state_noise": False,
            "press_control": press_control_metadata(),
            "measurement_source_sha256": measurement_sources,
            "measurement": telemetry.metadata(),
            "result": row,
          },
        )
        from kaihand_tactile_env.tasks.poker_draw.telemetry_plot import (
          plot_friction_trace,
        )

        if curve_path is not None:
          plot_friction_trace(trace_path, curve_path, target_force_n=force)
      rows.append(row)
      print(
        f"finished trial={index}/{len(trials)} success={result.success} "
        f"Fn={result.slide_finger_normal_force_means_n}N "
        f"contact={result.slide_finger_contact_fractions} "
        f"overhang={result.maximum_overhang_fraction:.3f} error={result.error}",
        flush=True,
      )
  document = {
    "schema_version": 4,
    "protocol": "production-per-finger-force",
    "created_utc": datetime.now(UTC).isoformat(),
    "scene": "poker-draw",
    "scene_path": str(args.scene_path.expanduser().resolve()),
    "scene_model_fingerprint": scene_fingerprint,
    "mujoco_version": mujoco.__version__,
    "model_initializations": 1,
    "headless": True,
    "record_video": args.record,
    "record_fps": args.record_fps if args.record else None,
    "state_noise": False,
    "object_xy_jitter_m": 0.0,
    "object_yaw_jitter_rad": 0.0,
    "seed": args.seed,
    "pair_name": TABLE_CARD_PAIR_NAME,
    "baseline_pair_friction": BASELINE_TABLE_CARD_FRICTION,
    "pressure_control": "per-finger pad-card solver normal force feedback, N/finger",
    "controller": "PokerDrawExecutor.execute_slide_only (shared with full task)",
    "press_control": press_control_metadata(),
    "measurement_source_sha256": measurement_sources,
    "scope": "press_slide_edge_only_not_full_draw",
    "legacy_slip_proxy_semantics": (
      "slide_fingertip_card_relative_slip_m is only mean pad-center world-X "
      "displacement minus card-center displacement over slide+edge; it is not "
      "contact-point slip. Use telemetry for per-finger contact slip."
    ),
    "trials": rows,
  }
  if args.json is not None:
    _write_json(args.json, document)
    print(f"wrote JSON: {args.json.expanduser().resolve()}")
  if args.csv is not None:
    _write_csv(args.csv, rows)
    print(f"wrote CSV: {args.csv.expanduser().resolve()}")


class _ForceTrialObserver:
  """Read-only, bounded-memory instrumentation of one synchronous trial."""

  def __init__(
    self,
    telemetry: Any,
    pad_geom_ids: tuple[int, ...],
    trace_file: TextIO | None,
    *,
    video: Any | None = None,
  ) -> None:
    self.telemetry = telemetry
    self.pad_geom_ids = pad_geom_ids
    self.trace_file = trace_file
    self.video = video
    self.writer: csv.DictWriter | None = None
    self.phase_counts: dict[str, int] = {}
    self.last_task_phase: str | None = None
    self.last_time_s = 0.0
    self.slide_positions: list[tuple[float, float]] = []

  def __call__(self, sim: ArmHandSimulation, phase: str) -> None:
    sample = self.telemetry.sample(phase)
    self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
    if self.trace_file is not None:
      if self.writer is None:
        self.writer = csv.DictWriter(self.trace_file, fieldnames=tuple(sample))
        self.writer.writeheader()
      self.writer.writerow(sample)
      if self.last_task_phase != phase:
        self.trace_file.flush()
    self.last_time_s = float(sim.data.time)
    if self.video is not None:
      try:
        self.video.observe(sim, phase)
      except Exception as error:
        raise FrictionVideoError(f"friction video recording failed: {error}") from error
    if phase != "terminal_unobserved":
      self.last_task_phase = phase
    if phase not in ("slide_card", "edge_hold"):
      return
    positions = (
      sum(float(sim.data.geom_xpos[gid, 0]) for gid in self.pad_geom_ids) / 4.0,
      # Keep both quantities at the same cached solver kinematics timestamp.
      float(sample["card_x_m"]),
    )
    if len(self.slide_positions) < 2:
      self.slide_positions.append(positions)
    else:
      self.slide_positions[-1] = positions


class FrictionVideoError(Exception):
  """Recording failure, intentionally not caught as a task RuntimeError."""


def _stage_outcome(
  row: dict[str, Any], phase_counts: dict[str, int], last_phase: str | None
) -> dict[str, Any]:
  """Report independent stage indicators, never infer a friction cause from failure."""
  forces = row["established_press_normal_forces_n"]
  established = bool(len(forces) == 4 and all(force > 0 for force in forces))
  started = phase_counts.get("slide_card", 0) > 0
  if row["success"]:
    failure_stage = None
  elif not established:
    failure_stage = "press" if phase_counts.get("four_finger_press", 0) else "approach"
  elif last_phase == "edge_hold":
    failure_stage = "edge_hold"
  else:
    failure_stage = "slide"
  return {
    "pressure_established": established,
    "slide_started": started,
    "half_overhang_reached": row["maximum_overhang_fraction"] >= 0.49,
    "force_tracking_qualified": row["slide_press_control_qualified"],
    "force_protocol_success": row["success"],
    "failure_stage": failure_stage,
    "failure_reason": row["error"],
    "observed_phase_sample_counts": dict(phase_counts),
    "last_observed_phase": last_phase,
    "causal_warning": "stage flags and slip diagnostics do not establish a friction-limit cause",
  }


def _measurement_source_hashes() -> dict[str, str]:
  import kaihand_tactile_env.tasks.poker_draw.telemetry as measurement

  module = Path(measurement.__file__)
  shared = module.parents[2] / "shared"
  paths = (
    Path(__file__),
    module,
    module.with_name("telemetry_plot.py"),
    shared / "contact_tactile.py",
    shared / "simulation.py",
    shared / "task_video.py",
  )
  return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


if __name__ == "__main__":
  main()
