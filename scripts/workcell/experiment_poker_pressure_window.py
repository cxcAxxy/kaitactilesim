#!/usr/bin/env python3
"""Serial, opt-in pressure-window calibration: short stroke or full table-edge draw.

This candidate protocol limits the right-arm servo's world-X wrench. It never
forces the card to move, locks it, or uses measured slip as a control input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

for _variable in (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "LP_NUM_THREADS",
):
  os.environ.setdefault(_variable, "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.tasks.poker_draw.friction import (  # noqa: E402
  TABLE_CARD_PAIR_NAME,
  model_with_table_card_friction,
  press_control_metadata,
)
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (  # noqa: E402
  ForceLimitedPokerSimulation,
  PressureWindowExecutor,
  PressureWindowSettings,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawPlanner  # noqa: E402
from kaihand_tactile_env.tasks.poker_draw.telemetry import (  # noqa: E402
  FINGERS,
  PokerFrictionTelemetry,
  cached_point_velocity,
)

SCOPE = "short_pressure_window_not_full_draw"
TABLE_EDGE_SCOPE = "pressure_window_to_table_edge_not_pickup"
_DRAW_PHASES = frozenset(("slide_card", "edge_hold", "final_hold"))
_DRIVE_KEYS = (
  "drive_requested_fx_n",
  "drive_limited_fx_n",
  "drive_actual_fx_n",
  "drive_saturated",
  "drive_cap_error_n",
  "drive_jacobian_condition",
  "drive_ee_y_m",
  "drive_ee_z_m",
  "drive_pose_error_y_m",
  "drive_pose_error_z_m",
  "drive_rotation_error_rad",
  "drive_requested_fy_n",
  "drive_requested_fz_n",
  "drive_requested_tx_nm",
  "drive_requested_ty_nm",
  "drive_requested_tz_nm",
  "drive_actual_fy_n",
  "drive_actual_fz_n",
  "drive_actual_tx_nm",
  "drive_actual_ty_nm",
  "drive_actual_tz_nm",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  defaults = PressureWindowSettings()
  parser.add_argument(
    "--press-force", type=float, nargs="+", default=[0.15, 0.35, 1.20]
  )
  parser.add_argument(
    "--goal",
    choices=("short", "table-edge"),
    default="short",
    help="short: legacy 30 mm probe; table-edge: draw to measured table edge, without pickup.",
  )
  parser.add_argument(
    "--max-slide-time",
    type=float,
    default=30.0,
    help="Safety bound for table-edge slide, at most 45 simulation seconds (not wall time).",
  )
  parser.add_argument(
    "--max-slide-travel",
    type=float,
    default=0.16,
    help="Safety bound for commanded table-edge hand travel, at most 0.20 m.",
  )
  parser.add_argument("--table-friction", type=float, default=defaults.table_friction)
  parser.add_argument(
    "--drive-limit",
    type=float,
    default=defaults.drive_limit_n,
    help="Servo world-X wrench limit, N (not a contact force cap).",
  )
  parser.add_argument(
    "--slide-distance", type=float, default=0.030, help="m, at most 0.060."
  )
  parser.add_argument(
    "--slide-speed",
    type=float,
    default=defaults.slide_speed_m_s,
    help="Nominal path speed, m/s.",
  )
  parser.add_argument("--hold-seconds", type=float, default=0.5)
  parser.add_argument(
    "--contact-time-constant", type=float, default=defaults.contact_time_constant_s
  )
  parser.add_argument(
    "--timestep",
    type=float,
    default=None,
    help="Experiment-only physics step, 0.0005 to 0.002 s; smaller uses more CPU.",
  )
  parser.add_argument(
    "--finger-servo-gain",
    type=float,
    default=None,
    help="Opt-in velocity gain of right non-thumb q2/q3 servos; all original assets unchanged.",
  )
  parser.add_argument(
    "--friction-impedance-ratio",
    type=float,
    default=defaults.contact_friction_impedance_ratio,
    help="Experiment contact tangential/normal impedance ratio (MuJoCo impratio).",
  )
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="A new directory; never overwritten."
  )
  parser.add_argument("--record", action="store_true")
  parser.add_argument("--record-fps", type=float, default=5.0)
  args = parser.parse_args(argv)
  if not 1 <= len(args.press_force) <= 10:
    parser.error("the bounded experiment requires between 1 and 10 pressure trials")
  if any(not math.isfinite(value) or value <= 0 for value in args.press_force):
    parser.error("--press-force values must be finite and positive")
  if not math.isfinite(args.record_fps) or not 0 < args.record_fps <= 10:
    parser.error("--record-fps must be finite and in (0, 10]")
  try:
    args.settings = PressureWindowSettings(
      table_friction=args.table_friction,
      drive_limit_n=args.drive_limit,
      slide_distance_m=args.slide_distance,
      slide_speed_m_s=args.slide_speed,
      hold_seconds=args.hold_seconds,
      contact_time_constant_s=args.contact_time_constant,
      contact_friction_impedance_ratio=args.friction_impedance_ratio,
      finger_servo_velocity_gain=args.finger_servo_gain,
      physics_timestep_s=args.timestep,
      goal=args.goal,
      max_slide_time_s=args.max_slide_time,
      max_slide_travel_m=args.max_slide_travel,
    )
  except ValueError as error:
    parser.error(str(error))
  args.output_dir = args.output_dir.expanduser().resolve()
  return args


def _json_safe(value: Any) -> Any:
  if isinstance(value, (float, np.floating)):
    return float(value) if math.isfinite(value) else None
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def _write_json(path: Path, document: dict[str, Any]) -> None:
  with path.open("x", encoding="utf-8") as stream:
    json.dump(_json_safe(document), stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")


def _classification(
  goal: str, outcome: dict[str, Any] | None, error: str | None
) -> str:
  """Classify measured completion only, never infer slip/stall from target force."""
  if error is not None:
    return "control_error"
  if goal != "table-edge" or outcome is None:
    return "unclassified"
  if not outcome.get("target_reached", False):
    return "edge_not_reached"
  if outcome.get("full_slide_qualified", False):
    return "edge_reached_contact_qualified"
  return "edge_reached_contact_unqualified"


def _video_success(
  goal: str, outcome: dict[str, Any] | None, error: str | None
) -> bool:
  if error is not None:
    return False
  if goal == "short":
    return True  # Legacy protocol-completion semantics, explicitly documented.
  return bool(
    outcome is not None
    and outcome.get("target_reached", False)
    and outcome.get("full_slide_qualified", False)
  )


def _source_hashes() -> dict[str, str]:
  import kaihand_tactile_env.tasks.poker_draw.pressure_window as controller

  directory = Path(controller.__file__).parent
  shared = directory.parents[1] / "shared"
  sources = [Path(__file__)]
  sources.extend(
    directory / name
    for name in (
      "pressure_window.py",
      "pressure_video.py",
      "task.py",
      "config.py",
      "press_control.py",
      "friction.py",
      "telemetry.py",
    )
  )
  sources.extend(
    shared / name
    for name in (
      "simulation.py",
      "tactile.py",
      "contact_tactile.py",
      "task_video.py",
    )
  )
  return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


def _observed_fields() -> tuple[str, ...]:
  return (
    *_DRIVE_KEYS,
    "card_vx_m_s",
    "mean_pad_vx_m_s",
    "ee_vx_m_s",
    "four_fingers_loaded",
    "four_fingers_pressure_qualified",
    *(
      f"{finger}_{suffix}"
      for finger in FINGERS
      for suffix in (
        "fn_n",
        "ft_n",
        "contact",
        "pressure_qualified",
      )
    ),
  )


class _SampleStatistics:
  """Time-weighted online sums, independent of trace size."""

  def __init__(self) -> None:
    self.count = 0
    self.duration_s = 0.0
    self.integrals = {key: 0.0 for key in _observed_fields()}
    self.maximum_abs = dict(self.integrals)

  def add(self, row: dict[str, Any], *, dt_s: float | None = None) -> None:
    duration = float(row["sample_dt_s"] if dt_s is None else dt_s)
    self.count += 1
    self.duration_s += duration
    for key in self.integrals:
      value = float(row[key])
      self.integrals[key] += value * duration
      self.maximum_abs[key] = max(self.maximum_abs[key], abs(value))

  def summary(self) -> dict[str, Any]:
    means = {
      key: value / self.duration_s if self.duration_s else None
      for key, value in self.integrals.items()
    }
    return {
      "sample_count": self.count,
      "duration_s": self.duration_s,
      "means": means,
      "maximum_absolute": self.maximum_abs,
    }


def _tail_summary(
  rows: deque[dict[str, Any]], duration_s: float = 0.5
) -> dict[str, Any]:
  """Right-endpoint weighted final window, clipping the boundary sample."""
  summary = _SampleStatistics()
  if rows:
    cutoff = float(rows[-1]["time_s"]) - duration_s
    for row in rows:
      dt_s = min(float(row["sample_dt_s"]), max(0.0, float(row["time_s"]) - cutoff))
      if dt_s > 0:
        summary.add(row, dt_s=dt_s)
  return summary.summary()


class PressureWindowVideoError(Exception):
  """Recording failures must not be mistaken for physical/control RuntimeErrors."""


class _Observer:
  def __init__(
    self, simulation: Any, telemetry: Any, stream: TextIO, video: Any = None
  ):
    self.simulation = simulation
    self.telemetry = telemetry
    self.stream = stream
    self.video = video
    self.writer: csv.DictWriter | None = None
    self.last_time_s = 0.0
    self.last_phase: str | None = None
    self.previous: dict[str, Any] | None = None
    self.slide_start: dict[str, Any] | None = None
    self.slide_end: dict[str, Any] | None = None
    self.active_slide_end: dict[str, Any] | None = None
    self.slide_stats = _SampleStatistics()
    self.draw_stats = _SampleStatistics()
    self.tail: deque[dict[str, Any]] = deque()
    self.slide_tail: deque[dict[str, Any]] = deque()
    self.pad_ids = tuple(
      simulation.model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id
      for finger in FINGERS
    )
    self.ee_id = simulation._site_id["right"]
    self.ee_body_id = int(simulation.model.site_bodyid[self.ee_id])

  @staticmethod
  def _append_tail(window: deque[dict[str, Any]], row: dict[str, Any]) -> None:
    window.append(row)
    cutoff = float(row["time_s"]) - 0.5
    while len(window) > 1 and float(window[0]["time_s"]) <= cutoff:
      window.popleft()

  def __call__(self, sim: Any, phase: str) -> None:
    row = self.telemetry.sample(phase)
    active = phase in _DRAW_PHASES
    state = sim.drive_state()
    budget_active = sim.drive_limit_n is not None
    # Preserve a safety-check failure's last applied wrench even if the task
    # raised before its usual slide observer callback. Do not leak stale drive
    # samples into the next reset's approach measurements.
    row.update(
      {
        key: float(state.get(key, 0.0)) if active or budget_active else 0.0
        for key in _DRIVE_KEYS
      }
    )
    row["drive_budget_active"] = int(budget_active)
    pad_positions = [sim.data.geom_xpos[gid] for gid in self.pad_ids]
    pad_velocities = [
      cached_point_velocity(sim.model, sim.data, int(sim.model.geom_bodyid[gid]), point)
      for gid, point in zip(self.pad_ids, pad_positions, strict=True)
    ]
    ee_position = sim.data.site_xpos[self.ee_id]
    ee_velocity = cached_point_velocity(
      sim.model, sim.data, self.ee_body_id, ee_position
    )
    row["mean_pad_x_m"] = float(np.mean([point[0] for point in pad_positions]))
    row["mean_pad_vx_m_s"] = float(
      np.mean([velocity[0] for velocity in pad_velocities])
    )
    row["ee_x_m"] = float(ee_position[0])
    row["ee_vx_m_s"] = float(ee_velocity[0])
    row["four_fingers_loaded"] = int(
      all(row[f"{finger}_contact"] for finger in FINGERS)
    )
    row["four_fingers_pressure_qualified"] = int(
      all(row[f"{finger}_pressure_qualified"] for finger in FINGERS)
    )
    if self.writer is None:
      self.writer = csv.DictWriter(self.stream, fieldnames=tuple(row))
      self.writer.writeheader()
    self.writer.writerow(row)
    if self.last_phase != phase:
      self.stream.flush()
    self.last_time_s = float(sim.data.time)
    self.last_phase = phase
    if active:
      if self.slide_start is None:
        # Prefer the immediately preceding setup sample: both positions are
        # cached solver-stage poses, before the first commanded slide step.
        self.slide_start = self.previous or row
      self.slide_end = row
      self.draw_stats.add(row)
      self._append_tail(self.tail, row)
      if phase == "slide_card":
        self.active_slide_end = row
        self.slide_stats.add(row)
        self._append_tail(self.slide_tail, row)
    self.previous = row
    if self.video is not None:
      try:
        self.video.observe(sim, phase)
      except Exception as error:
        raise PressureWindowVideoError(
          f"pressure-window video failed: {error}"
        ) from error

  def summary(self) -> dict[str, Any]:
    positions = ("time_s", "solver_time_s", "card_x_m", "mean_pad_x_m", "ee_x_m")

    def snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
      return {key: row[key] for key in positions} if row is not None else None

    def displacement(end: dict[str, Any] | None) -> dict[str, float] | None:
      if self.slide_start is None or end is None:
        return None
      start = self.slide_start
      result = {
        "card_toward_robot_m": float(start["card_x_m"] - end["card_x_m"]),
        "mean_pad_toward_robot_m": float(start["mean_pad_x_m"] - end["mean_pad_x_m"]),
        "ee_toward_robot_m": float(start["ee_x_m"] - end["ee_x_m"]),
      }
      result["pad_center_relative_displacement_m"] = (
        result["mean_pad_toward_robot_m"] - result["card_toward_robot_m"]
      )
      return result

    return {
      "slide_start": snapshot(self.slide_start),
      "active_slide_end": snapshot(self.active_slide_end),
      "slide_and_hold_end": snapshot(self.slide_end),
      "active_slide_displacement": displacement(self.active_slide_end),
      "slide_and_hold_displacement": displacement(self.slide_end),
      "active_slide": self.slide_stats.summary(),
      "slide_and_hold": self.draw_stats.summary(),
      "active_slide_final_0_5s": _tail_summary(self.slide_tail),
      "slide_and_hold_final_0_5s": _tail_summary(self.tail),
      "velocity_sign": "world X; negative means toward robot",
      "pad_center_relative_displacement_semantics": "kinematic proxy, not integrated contact-point slip",
    }


def _configure_contact_model(
  sim: Any, settings: PressureWindowSettings
) -> dict[str, Any]:
  model = sim.model
  pair_id = model.pair(TABLE_CARD_PAIR_NAME).id
  card_id = model.geom("card_core_geom").id
  original_pair_solref = model.pair_solref[pair_id].copy()
  original_card_solref = model.geom_solref[card_id].copy()
  original_impratio = float(model.opt.impratio)
  original_timestep = float(model.opt.timestep)
  if settings.physics_timestep_s is not None:
    model.opt.timestep = settings.physics_timestep_s
    sim.timestep = settings.physics_timestep_s
  model.opt.impratio = settings.contact_friction_impedance_ratio
  finger_gains = {}
  if settings.finger_servo_velocity_gain is not None:
    for finger in FINGERS:
      for joint in (2, 3):
        name = f"hand_r_{finger}_joint{joint}"
        actuator = model.actuator(name).id
        finger_gains[name] = {
          "original": float(model.actuator_gainprm[actuator, 0]),
          "used": settings.finger_servo_velocity_gain,
        }
        model.actuator_gainprm[actuator, 0] = settings.finger_servo_velocity_gain
        model.actuator_biasprm[actuator, 2] = -settings.finger_servo_velocity_gain
  if settings.contact_time_constant_s is not None:
    # Local experiment model only. Card priority selects its solref for pads;
    # the explicit table-card pair has its own separately stored solref.
    model.pair_solref[pair_id, 0] = settings.contact_time_constant_s
    model.geom_solref[card_id, 0] = settings.contact_time_constant_s
  if settings.contact_damping_ratio is not None:
    model.pair_solref[pair_id, 1] = settings.contact_damping_ratio
    model.geom_solref[card_id, 1] = settings.contact_damping_ratio
  return {
    "table_card_pair_friction": model.pair_friction[pair_id].tolist(),
    "physics_timestep_original_s": original_timestep,
    "physics_timestep_used_s": float(model.opt.timestep),
    "experimental_right_finger_servo_gains": finger_gains,
    "contact_friction_impedance_ratio_original": original_impratio,
    "contact_friction_impedance_ratio_used": float(model.opt.impratio),
    "table_card_pair_solref_original": original_pair_solref.tolist(),
    "table_card_pair_solref_used": model.pair_solref[pair_id].tolist(),
    "card_geom_solref_original": original_card_solref.tolist(),
    "card_geom_solref_used": model.geom_solref[card_id].tolist(),
    "card_geom_friction": model.geom_friction[card_id].tolist(),
    "card_geom_priority": int(model.geom_priority[card_id]),
    "pad_geom_priorities": [
      int(model.geom_priority[model.geom(f"hand_r_{finger}_link4_tactile_pad_col").id])
      for finger in FINGERS
    ],
    "finger_card_nominal_sliding_friction": float(model.geom_friction[card_id, 0]),
    "finger_card_friction_source": "card_core_geom friction with higher contact priority; model parameter, not measured friction",
    "contact_solref_scope": "explicit table-card pair and priority-selected card geom only; no persistent scene changes",
  }


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  # Reserve all trial names by requiring an entirely new directory before any
  # model allocation. Individual artifacts also use exclusive creation.
  args.output_dir.mkdir(parents=True, exist_ok=False)
  settings: PressureWindowSettings = args.settings
  table_edge = settings.goal == "table-edge"
  scene = default_model_path("poker-draw")
  metadata = {
    "schema_version": 1,
    "created_utc": datetime.now(UTC).isoformat(),
    "scope": TABLE_EDGE_SCOPE if table_edge else SCOPE,
    "validation_status": "candidate_not_validated",
    "scene": "poker-draw",
    "scene_path": str(Path(scene).resolve()),
    "scene_model_fingerprint": model_fingerprint(scene),
    "settings": asdict(settings),
    "press_forces_per_finger_n": args.press_force,
    "seed": 0,
    "state_noise": False,
    "object_xy_jitter_m": 0.0,
    "object_yaw_jitter_rad": 0.0,
    "mujoco_version": mujoco.__version__,
    "controller_source_sha256": _source_hashes(),
    "production_reference_press_control": press_control_metadata(),
    "controller": "PressureWindowExecutor.draw; continuous bounded drive, no low-Fn pause",
    "goal_semantics": (
      "measured card/table geometry controls endpoint speed, stop and dwell; bounded hand travel/time; no pickup, no slip-feedback pressure adjustment"
      if table_edge
      else "fixed short hand stroke; not a full table-edge draw or pickup"
    ),
    "force_limit_semantics": "right-arm active servo wrench world-X only; not a cap on transient contact impulses or bias feedforward",
    "classification_semantics": (
      "measured edge completion and full-slide contact quality only; no light-slip or heavy-stall label inferred from force setting; RuntimeError is control_error"
      if table_edge
      else "unclassified until measured slip/contact/velocity/drive evidence is reviewed; RuntimeError is control_error, never inferred heavy_stall"
    ),
    "video_success_semantics": (
      "measured edge target reached AND full_slide_qualified AND no control error; not pickup/full-task success or validated pressure window"
      if table_edge
      else "protocol returned without control error, not classification, full-task success or validated pressure window"
    ),
    "model_initializations": 1,
    "add_genesis_probes": False,
    "serial": True,
    "headless": True,
    "record_video": args.record,
    "record_fps": args.record_fps if args.record else None,
    "rendering_backend": os.environ.get("MUJOCO_GL"),
    "software_rendering_requested": os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
    "thread_environment": {
      key: os.environ.get(key)
      for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LP_NUM_THREADS",
      )
    },
  }
  _write_json(args.output_dir / "protocol.json", metadata)
  trials = []
  with model_with_table_card_friction(scene, settings.table_friction) as model_path:
    simulation = ForceLimitedPokerSimulation(
      model_path, scene="poker-draw", add_genesis_probes=False
    )
    metadata["contact_model"] = _configure_contact_model(simulation, settings)
    for index, force in enumerate(args.press_force, 1):
      print(
        f"starting trial={index}/{len(args.press_force)} mu={settings.table_friction:g} target={force:g}N/finger drive_cap={settings.drive_limit_n:g}N",
        flush=True,
      )
      simulation.drive_limit_n = None
      simulation.reset(seed=0, object_xy_jitter=0.0, object_yaw_jitter=0.0)
      telemetry = PokerFrictionTelemetry(simulation, target_force_n=force)
      prefix = args.output_dir / f"trial_{index:03d}"
      trace_path = prefix.with_name(prefix.name + "_timeseries.csv")
      video_artifacts = None
      video_camera = None
      error = None
      experimental_control = None
      goal_outcome = None
      executor = None
      with ExitStack() as resources:
        stream = resources.enter_context(
          trace_path.open("x", encoding="utf-8", newline="")
        )
        video = None
        if args.record:
          from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider
          from kaihand_tactile_env.shared.task_video import TaskVideoRecorder

          # This lightweight model deliberately has no Genesis probe geoms.
          # Supply a solver provider explicitly so the recorder does not try
          # to construct its default Genesis provider. Poker's visible Fn/Ft
          # maps still use the recorder's independent right-hand spatial
          # solver provider; no probe geometry or physics settings are added.
          video = resources.enter_context(
            TaskVideoRecorder(
              simulation,
              prefix.with_name(prefix.name + "_video.mp4"),
              fps=args.record_fps,
              preview=False,
              tactile_provider=SolverContactTactileProvider(simulation.model),
              metadata={**metadata, "press_force_per_finger_n": force},
            )
          )
          if table_edge:
            from kaihand_tactile_env.tasks.poker_draw.pressure_video import (
              configure_pressure_window_video,
            )

            video_camera = configure_pressure_window_video(video, simulation)
          video.observe(simulation, "initial")
        observer = _Observer(simulation, telemetry, stream, video)
        try:
          plan = PokerDrawPlanner(simulation).plan()
          executor = PressureWindowExecutor(
            simulation, press_force_per_finger_n=force, observer=observer
          )
          experimental_control = executor.control_metadata()
          if table_edge:
            experimental_control = {
              **experimental_control,
              "object_motion_or_slip_used_for_control": True,
              "geometry_usage": "endpoint speed/stop/dwell only; no slip-based pressure adjustment",
              "slip_feedback_used_for_pressure_adjustment": False,
            }
          goal_outcome = executor.draw(plan, settings)
        except RuntimeError as caught:
          error = str(caught)
          # A controller/safety exception can happen after partial progress.
          # Keep that measured evidence rather than fabricating a failed
          # edge result or losing the last observed overhang/force quality.
          goal_outcome = getattr(executor, "edge_outcome", None)
        if float(simulation.data.time) > observer.last_time_s + 1e-9:
          observer(simulation, "terminal_unobserved")
        row = {
          "trial_index": index,
          "press_force_per_finger_n": force,
          "protocol_completed": error is None,
          "goal_outcome": goal_outcome,
          "video_camera": video_camera,
          "experimental_control": experimental_control,
          "classification": _classification(settings.goal, goal_outcome, error),
          "error": error,
          "simulation_duration_s": float(simulation.data.time),
          "terminal_drive_state": simulation.drive_state(),
          "observed": observer.summary(),
          "telemetry": telemetry.summary(),
          "measurement": telemetry.metadata(),
          "timeseries_csv": str(trace_path),
          "trial_json": str(prefix.with_suffix(".json")),
        }
        # Preserve numeric evidence before encoding finalization; recorder
        # failures propagate and must not silently become physical outcomes.
        _write_json(
          prefix.with_name(prefix.name + "_measurement.json"),
          {**metadata, "result": row},
        )
        if video is not None:
          video.set_outcome(row)
          mp4, sidecar = video.finish(
            success=_video_success(settings.goal, goal_outcome, error)
          )
          video_artifacts = {"mp4": str(mp4), "json": str(sidecar)}
      row["video_artifacts"] = video_artifacts
      _write_json(prefix.with_suffix(".json"), {**metadata, "result": row})
      trials.append(row)
      observed = row["observed"]
      displacement = observed["slide_and_hold_displacement"]
      card_mm = 1000 * displacement["card_toward_robot_m"] if displacement else None
      print(
        f"finished trial={index}/{len(args.press_force)} classification={row['classification']} card_dx_mm={card_mm} saturation={observed['active_slide']['means']['drive_saturated']} error={error}",
        flush=True,
      )
  _write_json(args.output_dir / "summary.json", {**metadata, "trials": trials})
  print(f"wrote summary: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
  main()
