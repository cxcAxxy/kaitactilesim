#!/usr/bin/env python3
"""Measure the bulb thread gate and passive funnel capture basin headlessly.

The diagnostic never edits the scene or task constants.  ``gate`` teleports the
free bulb through a deterministic pose grid and reports every capture predicate.
``approach`` drives the bulb downward through a compliant virtual holder, then
classifies capture, tilted/straight jams, misses, and reversible extraction.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation


def _finite(values: Iterable[float], name: str) -> tuple[float, ...]:
  result = tuple(float(value) for value in values)
  if not result or not all(math.isfinite(value) for value in result):
    raise ValueError(f"{name} must contain finite values")
  return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--protocol", choices=("gate", "approach"), default="gate")
  parser.add_argument("--yaw-degrees", type=float, nargs="+")
  parser.add_argument("--depth-errors-mm", type=float, nargs="+")
  parser.add_argument("--lateral-offsets-mm", type=float, nargs="+")
  parser.add_argument("--tilt-degrees", type=float, nargs="+")
  parser.add_argument("--azimuth-degrees", type=float, nargs="+")
  parser.add_argument("--lateral-stiffness-n-m", type=float, nargs="+")
  parser.add_argument("--angular-stiffness-nm-rad", type=float, nargs="+")
  parser.add_argument("--approach-speed-mm-s", type=float, default=20.0)
  parser.add_argument("--start-clearance-mm", type=float, default=2.0)
  parser.add_argument("--approach-duration-s", type=float, default=2.0)
  parser.add_argument("--settle-duration-s", type=float, default=0.25)
  parser.add_argument("--reverse-torque-nm", type=float, default=0.06)
  parser.add_argument("--release-timeout-s", type=float, default=1.5)
  parser.add_argument("--extract-force-n", type=float, default=1.2)
  parser.add_argument("--extract-timeout-s", type=float, default=1.0)
  parser.add_argument("--json", type=Path, help="New JSON result path.")
  parser.add_argument("--csv", type=Path, help="New flat trial CSV result path.")
  args = parser.parse_args(argv)

  if args.protocol == "gate":
    args.yaw_degrees = args.yaw_degrees or tuple(range(0, 360, 5))
    args.depth_errors_mm = args.depth_errors_mm or tuple(
      float(value) for value in np.linspace(-2.0, 5.0, 71)
    )
    args.lateral_offsets_mm = args.lateral_offsets_mm or (0.0,)
    args.tilt_degrees = args.tilt_degrees or (0.0,)
    args.azimuth_degrees = args.azimuth_degrees or (0.0,)
    args.lateral_stiffness_n_m = (0.0,)
    args.angular_stiffness_nm_rad = (0.0,)
  else:
    if args.depth_errors_mm is not None:
      parser.error("--depth-errors-mm is only valid with --protocol gate")
    args.yaw_degrees = args.yaw_degrees or (0.0, 90.0, 180.0, 270.0)
    args.lateral_offsets_mm = args.lateral_offsets_mm or (0.0, 1.0, 2.0, 4.0, 8.0, 12.0)
    args.tilt_degrees = args.tilt_degrees or (0.0, 2.0, 5.0, 10.0)
    # The 24 funnel sectors repeat every 15 degrees.  These sample a sector
    # centre and boundary without multiplying equivalent cases.
    args.azimuth_degrees = args.azimuth_degrees or (0.0, 7.5)
    args.lateral_stiffness_n_m = args.lateral_stiffness_n_m or (50.0,)
    args.angular_stiffness_nm_rad = args.angular_stiffness_nm_rad or (0.01,)

  try:
    for name in (
      "yaw_degrees",
      "lateral_offsets_mm",
      "tilt_degrees",
      "azimuth_degrees",
      "lateral_stiffness_n_m",
      "angular_stiffness_nm_rad",
    ):
      setattr(args, name, _finite(getattr(args, name), name))
    if args.protocol == "gate":
      args.depth_errors_mm = _finite(args.depth_errors_mm, "depth_errors_mm")
  except ValueError as error:
    parser.error(str(error))

  nonnegative = (
    *args.lateral_offsets_mm,
    *args.tilt_degrees,
    *args.lateral_stiffness_n_m,
    *args.angular_stiffness_nm_rad,
    args.approach_speed_mm_s,
    args.start_clearance_mm,
    args.approach_duration_s,
    args.settle_duration_s,
    args.release_timeout_s,
    args.extract_force_n,
    args.extract_timeout_s,
  )
  if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
    parser.error(
      "offsets, tilts, stiffnesses, speeds, forces and durations must be finite and non-negative"
    )
  if not math.isfinite(args.reverse_torque_nm) or args.reverse_torque_nm <= 0:
    parser.error("--reverse-torque-nm must be finite and positive")
  if (
    args.approach_duration_s <= 0
    or args.release_timeout_s <= 0
    or args.extract_timeout_s <= 0
  ):
    parser.error("approach, release and extract timeouts must be positive")

  varying = (
    args.yaw_degrees,
    args.lateral_offsets_mm,
    args.tilt_degrees,
    args.azimuth_degrees,
  )
  if args.protocol == "gate":
    trial_count = math.prod(len(values) for values in (*varying, args.depth_errors_mm))
    limit = 10_000
  else:
    trial_count = math.prod(
      len(values)
      for values in (
        *varying,
        args.lateral_stiffness_n_m,
        args.angular_stiffness_nm_rad,
      )
    )
    limit = 1_000
  if trial_count > limit:
    parser.error(
      f"{trial_count} trials exceed the bounded {args.protocol} limit of {limit}"
    )
  args.trial_count = trial_count
  return args


def _quaternion(yaw_deg: float, tilt_deg: float, azimuth_deg: float) -> np.ndarray:
  """Return object-to-world yaw followed by a world-radial lean."""
  yaw = math.radians(yaw_deg)
  tilt = math.radians(tilt_deg)
  azimuth = math.radians(azimuth_deg)
  q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
  q_tilt = np.concatenate(([math.cos(tilt / 2)], axis * math.sin(tilt / 2)))
  result = np.empty(4)
  mujoco.mju_mulQuat(result, q_tilt, q_yaw)
  return result


def _pose(
  lateral_mm: float,
  depth_error_mm: float,
  tilt_deg: float,
  yaw_deg: float,
  azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
  azimuth = math.radians(azimuth_deg)
  position = config.THREAD_ENTRY_POSITION_M.copy()
  position[:2] += 0.001 * lateral_mm * np.array([math.cos(azimuth), math.sin(azimuth)])
  # Positive depth error means deeper insertion, hence a lower free-body origin.
  position[2] -= 0.001 * depth_error_mm
  return position, _quaternion(yaw_deg, tilt_deg, azimuth_deg)


def _gate_row(
  simulation: BulbScrewSimulation,
  *,
  yaw_deg: float,
  depth_error_mm: float,
  lateral_mm: float,
  tilt_deg: float,
  azimuth_deg: float,
) -> dict[str, Any]:
  simulation.reset()
  position, quaternion = _pose(
    lateral_mm, depth_error_mm, tilt_deg, yaw_deg, azimuth_deg
  )
  simulation.set_object_pose("bulb", position, quaternion)
  depth, lateral, tilt = simulation._alignment()
  load = simulation.thread_contact_load()
  twist = simulation.object_twist("bulb")
  speed = float(np.linalg.norm(twist[:3]))
  angular_speed = float(np.linalg.norm(twist[3:]))
  row = {
    "yaw_deg": yaw_deg,
    "depth_error_mm": 1000.0 * (depth - config.THREAD_ENTRY_DEPTH_M),
    "lateral_offset_mm": lateral_mm,
    "initial_tilt_deg": tilt_deg,
    "lateral_mm": 1000.0 * lateral,
    "tilt_deg": math.degrees(tilt),
    "azimuth_deg": azimuth_deg,
    "thread_contact_load_n": load,
    "linear_speed_m_s": speed,
    "angular_speed_rad_s": angular_speed,
    "depth_ok": abs(depth - config.THREAD_ENTRY_DEPTH_M) <= config.CAPTURE_DEPTH_M,
    "contact_ok": load > config.THREAD_CAPTURE_LOAD_N,
    "lateral_ok": lateral <= config.CAPTURE_LATERAL_M,
    "tilt_ok": tilt <= config.CAPTURE_TILT_RAD,
    "speed_ok": speed <= config.CAPTURE_SPEED_M_S,
    "angular_speed_ok": angular_speed <= config.CAPTURE_ANGULAR_SPEED_RAD_S,
    "depth_margin_mm": 1000.0
    * (config.CAPTURE_DEPTH_M - abs(depth - config.THREAD_ENTRY_DEPTH_M)),
    "contact_margin_n": load - config.THREAD_CAPTURE_LOAD_N,
    "lateral_margin_mm": 1000.0 * (config.CAPTURE_LATERAL_M - lateral),
    "tilt_margin_deg": math.degrees(config.CAPTURE_TILT_RAD - tilt),
    "angular_speed_margin_rad_s": config.CAPTURE_ANGULAR_SPEED_RAD_S - angular_speed,
  }
  row["captured"] = simulation.try_engage_thread()
  return row


def _run_gate(
  simulation: BulbScrewSimulation, args: argparse.Namespace
) -> list[dict[str, Any]]:
  rows = []
  conditions = itertools.product(
    args.yaw_degrees,
    args.depth_errors_mm,
    args.lateral_offsets_mm,
    args.tilt_degrees,
    args.azimuth_degrees,
  )
  for yaw, depth, lateral, tilt, azimuth in conditions:
    rows.append(
      _gate_row(
        simulation,
        yaw_deg=yaw,
        depth_error_mm=depth,
        lateral_mm=lateral,
        tilt_deg=tilt,
        azimuth_deg=azimuth,
      )
    )
  return rows


def _contact_load(simulation: BulbScrewSimulation, target_geoms: set[int]) -> float:
  bulb = simulation.model.body("bulb").id
  wrench = np.empty(6)
  total = 0.0
  for index, contact in enumerate(simulation.data.contact):
    geom1, geom2 = int(contact.geom1), int(contact.geom2)
    body1 = int(simulation.model.geom_bodyid[geom1])
    body2 = int(simulation.model.geom_bodyid[geom2])
    if not (
      (body1 == bulb and geom2 in target_geoms)
      or (body2 == bulb and geom1 in target_geoms)
    ):
      continue
    mujoco.mj_contactForce(simulation.model, simulation.data, index, wrench)
    total += max(0.0, float(wrench[0]))
  return total


def _orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
  inverse = current.copy()
  inverse[1:] *= -1
  error = np.empty(4)
  mujoco.mju_mulQuat(error, desired, inverse)
  if error[0] < 0:
    error *= -1
  length = float(np.linalg.norm(error[1:]))
  if length < 1.0e-12:
    return np.zeros(3)
  angle = 2.0 * math.atan2(length, float(np.clip(error[0], -1.0, 1.0)))
  return error[1:] * (angle / length)


def _apply_holder(
  simulation: BulbScrewSimulation,
  origin_xy: np.ndarray,
  desired_quaternion: np.ndarray,
  downward_speed_m_s: float,
  lateral_stiffness_n_m: float,
  angular_stiffness_nm_rad: float,
) -> None:
  model, data = simulation.model, simulation.data
  body = model.body("bulb").id
  position = data.xpos[body]
  quaternion = data.xquat[body]
  twist = simulation.object_twist("bulb")
  mass = float(model.body_subtreemass[body])
  inertia = float(np.max(model.body_inertia[body]))
  lateral_damping = 2.0 * math.sqrt(lateral_stiffness_n_m * mass)
  angular_damping = 2.0 * math.sqrt(angular_stiffness_nm_rad * inertia)
  force = np.zeros(3)
  force[:2] = (
    lateral_stiffness_n_m * (origin_xy - position[:2]) - lateral_damping * twist[:2]
  )
  axial_damping = 2.0
  gravity_compensation = -mass * float(model.opt.gravity[2])
  force[2] = gravity_compensation + axial_damping * (-downward_speed_m_s - twist[2])
  torque = (
    angular_stiffness_nm_rad * _orientation_error(desired_quaternion, quaternion)
    - angular_damping * twist[3:]
  )
  data.xfrc_applied[body, :3] = np.clip(force, -3.0, 3.0)
  data.xfrc_applied[body, 3:] = np.clip(torque, -0.2, 0.2)


def _fixture_sets(simulation: BulbScrewSimulation) -> tuple[set[int], set[int]]:
  funnel = {
    simulation.model.geom(f"bulb_socket_funnel_{index:02d}").id for index in range(24)
  }
  socket_body = simulation.model.body("bulb_socket").id
  fixture = set(np.flatnonzero(simulation.model.geom_bodyid == socket_body))
  return funnel, fixture


@contextmanager
def _capture_mujoco_warnings() -> Iterator[Counter[str]]:
  """Capture process-global MuJoCo warning callbacks for one serial trial."""

  messages: Counter[str] = Counter()
  previous = mujoco.get_mju_user_warning()

  def capture(message: str) -> None:
    messages[str(message)] += 1
    if previous is not None:
      previous(message)

  mujoco.set_mju_user_warning(capture)
  try:
    yield messages
  finally:
    mujoco.set_mju_user_warning(previous)


def _mujoco_warning_stats(
  simulation: BulbScrewSimulation,
) -> list[dict[str, int | str]]:
  return [
    {
      "warning": mujoco.mjtWarning(index).name,
      "count": int(stat.number),
      "last_info": int(stat.lastinfo),
    }
    for index, stat in enumerate(simulation.data.warning)
    if stat.number
  ]


_DYNAMIC_STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "xpos",
  "xquat",
  "cvel",
  "sensordata",
)


def _record_nonfinite_state(
  simulation: BulbScrewSimulation,
  first_occurrence: dict[str, str],
  checkpoint: str,
) -> bool:
  """Record the first checkpoint at which each dynamic array became nonfinite."""

  if not math.isfinite(float(simulation.data.time)):
    first_occurrence.setdefault("time", checkpoint)
  for name in _DYNAMIC_STATE_FIELDS:
    values = np.asarray(getattr(simulation.data, name))
    if not np.isfinite(values).all():
      first_occurrence.setdefault(name, checkpoint)
  return bool(first_occurrence)


def _exit_trial(
  simulation: BulbScrewSimulation,
  args: argparse.Namespace,
  *,
  nonfinite: dict[str, str],
) -> dict[str, Any]:
  body = simulation.model.body("bulb").id
  simulation.data.xfrc_applied[body] = 0.0
  released_at_s = None
  start = float(simulation.data.time)
  release_steps = math.ceil(args.release_timeout_s / simulation.timestep)
  for _ in range(release_steps):
    simulation.data.xfrc_applied[body] = (0, 0, 0, 0, 0, args.reverse_torque_nm)
    simulation.step()
    if _record_nonfinite_state(simulation, nonfinite, "reverse_release"):
      break
    if not simulation.thread_engaged:
      released_at_s = float(simulation.data.time - start)
      break

  cleared_at_s = None
  if released_at_s is not None:
    extraction_start = float(simulation.data.time)
    extract_steps = math.ceil(args.extract_timeout_s / simulation.timestep)
    for _ in range(extract_steps):
      simulation.data.xfrc_applied[body] = (0, 0, args.extract_force_n, 0, 0, 0)
      simulation.step()
      if _record_nonfinite_state(simulation, nonfinite, "extraction"):
        break
      depth, _, _ = simulation._alignment()
      if depth < -0.003:
        # One further step lets the production re-arm condition observe clear air.
        simulation.step()
        _record_nonfinite_state(simulation, nonfinite, "extraction_clearance")
        cleared_at_s = float(simulation.data.time - extraction_start)
        break
  simulation.data.xfrc_applied[body] = 0.0
  return {
    "released": released_at_s is not None,
    "release_time_s": released_at_s,
    "cleared": cleared_at_s is not None,
    "clear_time_s": cleared_at_s,
    "capture_rearmed_after_clear": bool(simulation._capture_armed),
  }


def _classify_terminal(
  *,
  captured: bool,
  released: bool,
  cleared: bool,
  fixture_contact_recent: bool,
  linear_speed_m_s: float,
  base_z_m: float,
  radial_m: float,
  tilt_rad: float,
) -> str:
  if captured:
    if not released:
      return "captured_release_jam"
    if not cleared:
      return "captured_extraction_jam"
    return "captured_and_cleared"
  fell_clear = base_z_m < config.SOCKET_MOUTH_POSITION_M[2] - 0.026 or radial_m > 0.050
  if fell_clear:
    return "missed_fixture"
  settled = linear_speed_m_s <= 0.02
  if fixture_contact_recent and settled:
    if tilt_rad > config.CAPTURE_TILT_RAD:
      return "tilted_jam"
    return "straight_entry_jam"
  return "unsettled_timeout"


def _approach_row_impl(
  simulation: BulbScrewSimulation,
  args: argparse.Namespace,
  *,
  yaw_deg: float,
  lateral_mm: float,
  tilt_deg: float,
  azimuth_deg: float,
  lateral_stiffness_n_m: float,
  angular_stiffness_nm_rad: float,
  nonfinite: dict[str, str],
) -> dict[str, Any]:
  simulation.reset()
  azimuth = math.radians(azimuth_deg)
  position = np.array(
    [
      config.SOCKET_MOUTH_POSITION_M[0] + 0.001 * lateral_mm * math.cos(azimuth),
      config.SOCKET_MOUTH_POSITION_M[1] + 0.001 * lateral_mm * math.sin(azimuth),
      config.SOCKET_FUNNEL_TOP_Z_M + 0.001 * args.start_clearance_mm,
    ]
  )
  quaternion = _quaternion(yaw_deg, tilt_deg, azimuth_deg)
  simulation.set_object_pose("bulb", position, quaternion)
  _record_nonfinite_state(simulation, nonfinite, "initial_pose")
  origin_xy = position[:2].copy()
  funnel_geoms, fixture_geoms = _fixture_sets(simulation)
  start = float(simulation.data.time)
  first_funnel_s = None
  first_thread_s = None
  capture_time_s = None
  last_fixture_s = None
  peak_funnel_n = 0.0
  peak_fixture_n = 0.0
  peak_thread_n = 0.0
  minimum_lateral_m = math.inf
  maximum_depth_m = -math.inf
  maximum_tilt_rad = 0.0

  approach_steps = math.ceil(args.approach_duration_s / simulation.timestep)
  settle_steps = math.ceil(args.settle_duration_s / simulation.timestep)
  for index in range(approach_steps + settle_steps):
    if nonfinite:
      break
    speed = 0.001 * args.approach_speed_mm_s if index < approach_steps else 0.0
    _apply_holder(
      simulation,
      origin_xy,
      quaternion,
      speed,
      lateral_stiffness_n_m,
      angular_stiffness_nm_rad,
    )
    # Match the production pre-step capture check and retain the real crest
    # load that enables the weld; the guide may unload that contact in mj_step.
    mujoco.mj_forward(simulation.model, simulation.data)
    if _record_nonfinite_state(simulation, nonfinite, "approach_pre_step"):
      break
    pre_elapsed = float(simulation.data.time - start)
    pre_funnel = _contact_load(simulation, funnel_geoms)
    pre_fixture = _contact_load(simulation, fixture_geoms)
    pre_thread = simulation.thread_contact_load()
    engaged_before_step = simulation.try_engage_thread()
    simulation.step()
    if _record_nonfinite_state(simulation, nonfinite, "approach_post_step"):
      break
    elapsed = float(simulation.data.time - start)
    depth, lateral, tilt = simulation._alignment()
    funnel_load = max(pre_funnel, _contact_load(simulation, funnel_geoms))
    fixture_load = max(pre_fixture, _contact_load(simulation, fixture_geoms))
    thread_load = max(pre_thread, simulation.thread_contact_load())
    peak_funnel_n = max(peak_funnel_n, funnel_load)
    peak_fixture_n = max(peak_fixture_n, fixture_load)
    peak_thread_n = max(peak_thread_n, thread_load)
    minimum_lateral_m = min(minimum_lateral_m, lateral)
    maximum_depth_m = max(maximum_depth_m, depth)
    maximum_tilt_rad = max(maximum_tilt_rad, tilt)
    if funnel_load > 0 and first_funnel_s is None:
      first_funnel_s = pre_elapsed if pre_funnel > 0 else elapsed
    if thread_load > 0 and first_thread_s is None:
      first_thread_s = pre_elapsed if pre_thread > 0 else elapsed
    if fixture_load > 0:
      last_fixture_s = elapsed
    if simulation.thread_engaged:
      capture_time_s = pre_elapsed if engaged_before_step else elapsed
      break

  captured = capture_time_s is not None
  exit_result = (
    _exit_trial(simulation, args, nonfinite=nonfinite)
    if captured and not nonfinite
    else {
      "released": False,
      "release_time_s": None,
      "cleared": False,
      "clear_time_s": None,
      "capture_rearmed_after_clear": False,
    }
  )
  pose = simulation.object_pose("bulb")
  depth, lateral, tilt = simulation._alignment()
  twist = simulation.object_twist("bulb")
  radial = float(np.linalg.norm(pose[:2] - config.SOCKET_MOUTH_POSITION_M[:2]))
  elapsed = float(simulation.data.time - start)
  recent_window = max(0.05, args.settle_duration_s)
  contact_recent = (
    last_fixture_s is not None and elapsed - last_fixture_s <= recent_window
  )
  outcome = (
    "invalid"
    if nonfinite
    else _classify_terminal(
      captured=captured,
      released=bool(exit_result["released"]),
      cleared=bool(exit_result["cleared"]),
      fixture_contact_recent=contact_recent,
      linear_speed_m_s=float(np.linalg.norm(twist[:3])),
      base_z_m=float(pose[2]),
      radial_m=radial,
      tilt_rad=tilt,
    )
  )
  return {
    "yaw_deg": yaw_deg,
    "lateral_offset_mm": lateral_mm,
    "initial_tilt_deg": tilt_deg,
    "azimuth_deg": azimuth_deg,
    "lateral_stiffness_n_m": lateral_stiffness_n_m,
    "angular_stiffness_nm_rad": angular_stiffness_nm_rad,
    "outcome": outcome,
    "captured": captured,
    "capture_time_s": capture_time_s,
    **exit_result,
    "first_funnel_contact_s": first_funnel_s,
    "first_thread_contact_s": first_thread_s,
    "peak_funnel_contact_n": peak_funnel_n,
    "peak_fixture_contact_n": peak_fixture_n,
    "peak_thread_contact_n": peak_thread_n,
    "minimum_lateral_error_mm": 1000.0 * minimum_lateral_m,
    "maximum_insertion_depth_mm": 1000.0 * maximum_depth_m,
    "maximum_tilt_deg": math.degrees(maximum_tilt_rad),
    "terminal_lateral_error_mm": 1000.0 * lateral,
    "terminal_insertion_depth_mm": 1000.0 * depth,
    "terminal_tilt_deg": math.degrees(tilt),
    "terminal_linear_speed_m_s": float(np.linalg.norm(twist[:3])),
    "terminal_radial_mm": 1000.0 * radial,
    "fixture_contact_recent": contact_recent,
  }


def _approach_row(
  simulation: BulbScrewSimulation,
  args: argparse.Namespace,
  *,
  yaw_deg: float,
  lateral_mm: float,
  tilt_deg: float,
  azimuth_deg: float,
  lateral_stiffness_n_m: float,
  angular_stiffness_nm_rad: float,
) -> dict[str, Any]:
  nonfinite: dict[str, str] = {}
  with _capture_mujoco_warnings() as callback_messages:
    row = _approach_row_impl(
      simulation,
      args,
      yaw_deg=yaw_deg,
      lateral_mm=lateral_mm,
      tilt_deg=tilt_deg,
      azimuth_deg=azimuth_deg,
      lateral_stiffness_n_m=lateral_stiffness_n_m,
      angular_stiffness_nm_rad=angular_stiffness_nm_rad,
      nonfinite=nonfinite,
    )
  warning_stats = _mujoco_warning_stats(simulation)
  warnings = [
    {"message": message, "count": count}
    for message, count in sorted(callback_messages.items())
  ]
  invalid_reasons = []
  if warnings or warning_stats:
    invalid_reasons.append("mujoco_warning")
  if nonfinite:
    invalid_reasons.append("nonfinite_state")
  row.update(
    {
      "valid": not invalid_reasons,
      "invalid_reasons": invalid_reasons,
      "mujoco_warning_messages": warnings,
      "mujoco_warning_stats": warning_stats,
      "nonfinite_fields": [
        {"field": field, "first_checkpoint": checkpoint}
        for field, checkpoint in sorted(nonfinite.items())
      ],
    }
  )
  if invalid_reasons:
    row["outcome"] = "invalid"
  return row


def _run_approach(
  simulation: BulbScrewSimulation, args: argparse.Namespace
) -> list[dict[str, Any]]:
  rows = []
  conditions = itertools.product(
    args.yaw_degrees,
    args.lateral_offsets_mm,
    args.tilt_degrees,
    args.azimuth_degrees,
    args.lateral_stiffness_n_m,
    args.angular_stiffness_nm_rad,
  )
  for index, (yaw, lateral, tilt, azimuth, linear_k, angular_k) in enumerate(
    conditions, start=1
  ):
    row = _approach_row(
      simulation,
      args,
      yaw_deg=yaw,
      lateral_mm=lateral,
      tilt_deg=tilt,
      azimuth_deg=azimuth,
      lateral_stiffness_n_m=linear_k,
      angular_stiffness_nm_rad=angular_k,
    )
    rows.append(row)
    print(
      f"trial={index}/{args.trial_count} yaw={yaw:g}deg offset={lateral:g}mm "
      f"tilt={tilt:g}deg azimuth={azimuth:g}deg outcome={row['outcome']}",
      flush=True,
    )
  return rows


def _summary(protocol: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
  outcomes: dict[str, int] = {}
  for row in rows:
    name = row.get("outcome", "captured" if row["captured"] else "rejected")
    outcomes[name] = outcomes.get(name, 0) + 1
  valid_rows = [row for row in rows if row.get("valid", True)]
  capture_count = sum(bool(row["captured"]) for row in valid_rows)
  summary: dict[str, Any] = {
    "trial_count": len(rows),
    "valid_trial_count": len(valid_rows),
    "invalid_trial_count": len(rows) - len(valid_rows),
    "capture_count": capture_count,
    "capture_fraction": capture_count / len(valid_rows) if valid_rows else None,
    "outcomes": outcomes,
  }
  if protocol == "gate":
    grouped: dict[tuple[float, float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
      condition = (
        float(row["yaw_deg"]),
        float(row.get("lateral_offset_mm", row.get("lateral_mm", 0.0))),
        float(row.get("initial_tilt_deg", row.get("tilt_deg", 0.0))),
        float(row.get("azimuth_deg", 0.0)),
      )
      grouped.setdefault(condition, []).append(row)
    phase_rows = []
    for (yaw, lateral, tilt, azimuth), group in sorted(grouped.items()):
      fixed_entry = min(group, key=lambda row: abs(float(row["depth_error_mm"])))
      gate_group = [row for row in group if row["depth_ok"]]
      peak_gate = (
        max(gate_group, key=lambda row: float(row["thread_contact_load_n"]))
        if gate_group
        else None
      )
      peak_scanned = max(group, key=lambda row: float(row["thread_contact_load_n"]))
      physical_contacts = [row for row in group if row["contact_ok"]]
      nearest_contact = (
        min(physical_contacts, key=lambda row: abs(float(row["depth_error_mm"])))
        if physical_contacts
        else None
      )
      phase_rows.append(
        {
          "yaw_deg": yaw,
          "lateral_offset_mm": lateral,
          "initial_tilt_deg": tilt,
          "azimuth_deg": azimuth,
          "fixed_entry_contact_load_n": fixed_entry["thread_contact_load_n"],
          "fixed_entry_contact_ok": fixed_entry["contact_ok"],
          "fixed_entry_captured": fixed_entry["captured"],
          "peak_gate_contact_load_n": (
            peak_gate["thread_contact_load_n"] if peak_gate else None
          ),
          "peak_gate_contact_depth_error_mm": (
            peak_gate["depth_error_mm"] if peak_gate else None
          ),
          "peak_scanned_contact_load_n": peak_scanned["thread_contact_load_n"],
          "peak_scanned_contact_depth_error_mm": peak_scanned["depth_error_mm"],
          "nearest_contact_depth_error_mm": (
            nearest_contact["depth_error_mm"] if nearest_contact else None
          ),
          "nearest_contact_load_n": (
            nearest_contact["thread_contact_load_n"] if nearest_contact else None
          ),
          "contact_depth_errors_mm": [
            row["depth_error_mm"] for row in physical_contacts
          ],
          "captured_depth_errors_mm": [
            row["depth_error_mm"] for row in group if row["captured"]
          ],
        }
      )
    summary["gate_grouping_dimensions"] = [
      "yaw_deg",
      "lateral_offset_mm",
      "initial_tilt_deg",
      "azimuth_deg",
    ]
    # Retain the established key while making each row a complete non-depth
    # condition slice instead of incorrectly coalescing everything by yaw.
    summary["yaw_phase"] = phase_rows
  else:
    summary["release_count"] = sum(bool(row["released"]) for row in valid_rows)
    summary["clear_count"] = sum(bool(row["cleared"]) for row in valid_rows)
  return summary


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def _reserve_outputs(args: argparse.Namespace) -> None:
  outputs = tuple(path.expanduser().resolve() for path in (args.json, args.csv) if path)
  if len(set(outputs)) != len(outputs):
    raise ValueError("--json and --csv must be different files")
  for path in outputs:
    if path.exists():
      raise FileExistsError(f"refusing to overwrite result: {path}")
  args.json = args.json.expanduser().resolve() if args.json else None
  args.csv = args.csv.expanduser().resolve() if args.csv else None


def _write_outputs(
  args: argparse.Namespace, document: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x", encoding="utf-8") as file:
      file.write(json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n")
  if args.csv:
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("x", encoding="utf-8", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
      writer.writeheader()
      writer.writerows(_json_safe(rows))


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  _reserve_outputs(args)
  scene_path = default_model_path(config.SCENE_NAME)
  simulation = BulbScrewSimulation()
  rows = (
    _run_gate(simulation, args)
    if args.protocol == "gate"
    else _run_approach(simulation, args)
  )
  document = {
    "schema_version": 2,
    "created_utc": datetime.now(UTC).isoformat(),
    "protocol": args.protocol,
    "scene": config.SCENE_NAME,
    "scene_path": str(scene_path),
    "scene_model_fingerprint": model_fingerprint(scene_path),
    "mechanics_version": config.MECHANICS_VERSION,
    "mujoco_version": mujoco.__version__,
    "capture_thresholds": {
      "thread_entry_depth_mm": 1000.0 * config.THREAD_ENTRY_DEPTH_M,
      "depth_tolerance_mm": 1000.0 * config.CAPTURE_DEPTH_M,
      "lateral_tolerance_mm": 1000.0 * config.CAPTURE_LATERAL_M,
      "tilt_tolerance_deg": math.degrees(config.CAPTURE_TILT_RAD),
      "thread_contact_load_n": config.THREAD_CAPTURE_LOAD_N,
      "speed_m_s": config.CAPTURE_SPEED_M_S,
      "angular_speed_rad_s": config.CAPTURE_ANGULAR_SPEED_RAD_S,
    },
    "arguments": {
      key: value for key, value in vars(args).items() if key not in {"json", "csv"}
    },
    "summary": _summary(args.protocol, rows),
    "trials": rows,
  }
  _write_outputs(args, document, rows)
  print(json.dumps(_json_safe(document["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
