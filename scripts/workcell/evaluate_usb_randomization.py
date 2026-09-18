#!/usr/bin/env python3
"""Serial USB robustness evaluation; writes each case before starting the next.

Example (51 episodes, use --list-cases to inspect without loading MuJoCo):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
    .pixi/envs/default/bin/python scripts/workcell/evaluate_usb_randomization.py \
    --samples 8 --seed 0 --output-dir artifacts/usb_randomization/run01

Uniform samples, the nominal pose and eight box corners are summarized
separately. Case indices select a subset; the script never launches workers.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

LEVELS = {
  "small": (0.002, math.radians(1.0)),
  "medium": (0.005, math.radians(3.0)),
  "large": (0.010, math.radians(5.0)),
}
KINDS = ("nominal", "corner", "uniform")
PLACEMENT_PHASES = {"align", "insert", "release", "retreat", "verify"}
QUALITY_LIMITS = {
  "minimum_actual_joint_margin_deg": 15.0,
  "minimum_goal_joint_margin_deg": 15.0,
  "maximum_robot_obstacle_contact_observations": 0,
  "insertion_maximum_palm_normal_z_exclusive": -0.2,
  "insertion_minimum_index_above_pinky_m_exclusive": 0.03,
  "insertion_minimum_thumb_above_pinky_m_exclusive": 0.015,
  "placement_minimum_elbow_below_shoulder_m": 0.12,
  "placement_maximum_elbow_world_y_m": -0.15,
  "maximum_elapsed_simulation_s": 22.0,
  "maximum_lift_m_exclusive": 0.15,
}


def generate_cases(
  levels: tuple[str, ...] | list[str], samples: int, seed: int
) -> list[dict]:
  """Stable per-level samples, independent of filtering or selected case order."""
  if samples < 0 or seed < 0:
    raise ValueError("samples and seed must be nonnegative")
  if not levels or len(set(levels)) != len(levels) or set(levels) - LEVELS.keys():
    raise ValueError("levels must be distinct names from small, medium, large")
  cases = []
  for level_index, (level, (xy_range, yaw_range)) in enumerate(LEVELS.items()):
    if level not in levels:
      continue
    # Explicit poses have zero jitter. Keeping nominal and corners in every
    # level makes partial runs self-contained without inflating uniform rates.
    requests = [("nominal", (0.0, 0.0), 0.0)]
    requests += [
      ("corner", (sx * xy_range, sy * xy_range), sz * yaw_range)
      for sx, sy, sz in product((-1, 1), repeat=3)
    ]
    requests += [("uniform", None, None)] * samples
    for local_index, (kind, offset, yaw) in enumerate(requests):
      # SeedSequence avoids both process-global RNG state and Python's salted
      # hash. A selected case has the same seed in a full run or subset run.
      case_seed = int(
        np.random.SeedSequence([seed, level_index, local_index]).generate_state(1)[0]
      )
      kwargs = {
        "seed": case_seed,
        "xy_jitter_m": xy_range if kind == "uniform" else 0.0,
        "yaw_jitter_rad": yaw_range if kind == "uniform" else 0.0,
      }
      if kind != "uniform":
        kwargs.update(offset_xy_m=list(offset), yaw_offset_rad=yaw)
      ordinal = local_index - 9 if kind == "uniform" else max(0, local_index - 1)
      cases.append(
        {
          "index": level_index * (samples + 9) + local_index,
          "id": f"{level}_{kind}_{ordinal:04d}",
          "level": level,
          "kind": kind,
          "seed": case_seed,
          "xy_range_m": xy_range,
          "yaw_range_rad": yaw_range,
          "initialization_kwargs": kwargs,
        }
      )
  return cases


def _finite(value) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(value)


def _numeric_summary(values) -> dict:
  values = [float(value) for value in values if _finite(value)]
  return {
    "count": len(values),
    "minimum": min(values) if values else None,
    "mean": sum(values) / len(values) if values else None,
    "maximum": max(values) if values else None,
  }


def summarize_cases(reports: list[dict], planned_cases: list[dict]) -> dict:
  """Report independent denominators; interrupted cases do not count as failures.

  An unexpected exception counts as a failed evaluated case and is also listed
  separately. Missing and interrupted cases remain visible, so a partial run
  cannot be mistaken for completion of the planned matrix.
  """
  levels = [
    level for level in LEVELS if any(c["level"] == level for c in planned_cases)
  ]
  summary = {
    "planned_cases": len(planned_cases),
    "reported_cases": len(reports),
    "uniform_rate_denominator": (
      "completed or exception uniform cases only; nominal, corners, interrupted "
      "and unstarted cases are excluded"
    ),
    "by_level": {},
  }
  for level in levels:
    groups = {}
    for kind in KINDS:
      planned = [c for c in planned_cases if c["level"] == level and c["kind"] == kind]
      rows = [
        r for r in reports if r["case"]["level"] == level and r["case"]["kind"] == kind
      ]
      evaluated = [r for r in rows if r["status"] != "interrupted"]
      count = len(evaluated)
      task_count = sum(bool(r["task_success"]) for r in evaluated)
      quality_count = sum(bool(r["quality_pass"]) for r in evaluated)
      qualified_count = sum(
        bool(r["task_success"] and r["quality_pass"]) for r in evaluated
      )
      results = [r["result"] for r in evaluated if r.get("result") is not None]
      groups[kind] = {
        "planned_count": len(planned),
        "reported_count": len(rows),
        "evaluated_count": count,
        "unstarted_count": len(planned) - len(rows),
        "interrupted_count": sum(r["status"] == "interrupted" for r in rows),
        "exception_count": sum(r["status"] == "exception" for r in rows),
        "task_success_count": task_count,
        "task_success_rate": task_count / count if count else None,
        "quality_pass_count": quality_count,
        "quality_pass_rate": quality_count / count if count else None,
        "qualified_success_count": qualified_count,
        "qualified_success_rate": qualified_count / count if count else None,
        "failure_stages": dict(
          Counter(r["failure_stage"] for r in evaluated if not r["task_success"])
        ),
        "failure_reasons": dict(
          Counter(r["failure_reason"] for r in evaluated if not r["task_success"])
        ),
        "quality_failures": dict(
          Counter(reason for r in evaluated for reason in r["quality"]["violations"])
        ),
        "insertion_depth_m": _numeric_summary(
          r["insertion"]["insertion_depth_m"] for r in results
        ),
        "elapsed_simulation_s": _numeric_summary(
          r["elapsed_simulation_s"] for r in results
        ),
        "peak_socket_normal_load_n": _numeric_summary(
          r["peak_socket_normal_load_n"] for r in results
        ),
        "peak_axial_resistance_n": _numeric_summary(
          r["peak_axial_resistance_n"] for r in results
        ),
        "wall_duration_s": _numeric_summary(r["wall_duration_s"] for r in evaluated),
      }
    summary["by_level"][level] = groups
  return summary


class QualityObserver:
  """Read-only audit at every controller-observed MuJoCo step, without rendering."""

  def __init__(self, simulation):
    model = simulation.model
    self.limits = model.jnt_range[simulation._arm_joint_ids["right"]].copy()
    self.arm_indices = simulation._arm_qpos["right"]
    self.shoulder = model.body("right_arm_link1").id
    self.elbow = model.body("right_arm_link4").id
    self.palm = model.body("hand_r_base_link").id
    self.fingers = [
      model.body(name).id
      for name in ("hand_r_index_link2", "hand_r_pinky_link2", "hand_r_thumb_link1")
    ]
    bodies = [model.body(int(i)).name for i in model.geom_bodyid]
    self.geom_names = [model.geom(i).name or f"geom_{i}" for i in range(model.ngeom)]
    self.robot = np.array(
      [
        name.startswith(("right_arm_", "left_arm_", "hand_r_", "hand_l_"))
        or name == "torso"
        for name in bodies
      ]
    )
    self.obstacle = np.array(
      [name in {"table", "usb_fixture", "usb_socket"} for name in bodies]
    )
    self.obstacle_categories = [
      {
        "table": "robot_table",
        "usb_fixture": "robot_fixture",
        "usb_socket": "robot_socket",
      }.get(name)
      for name in bodies
    ]
    self.phase_stats = {}
    self.collisions = {}
    self.steps = 0
    self.last_phase = "initialization"
    self.wrench = np.empty(6)
    self.previous_wrist_degrees = None

  def __call__(self, simulation, phase: str) -> None:
    import mujoco

    data = simulation.data
    self.steps += 1
    self.last_phase = phase
    q = data.qpos[self.arm_indices]
    wrist_degrees = np.rad2deg(q[5:7])
    goal = simulation.arm_goal["right"]
    actual_margin = np.rad2deg(np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q))
    goal_margin = np.rad2deg(
      np.minimum(goal - self.limits[:, 0], self.limits[:, 1] - goal)
    )
    relative = data.xpos[self.elbow] - data.xpos[self.shoulder]
    palm_z = float(data.xmat[self.palm].reshape(3, 3)[2, 1])
    index, pinky, thumb = data.xpos[self.fingers]
    stat = self.phase_stats.setdefault(
      phase,
      {
        "steps": 0,
        "first_time_s": float(data.time),
        "last_time_s": float(data.time),
        "minimum_actual_joint_margin_deg": np.full(7, np.inf),
        "minimum_goal_joint_margin_deg": np.full(7, np.inf),
        "maximum_palm_normal_z": -np.inf,
        "minimum_index_above_pinky_m": np.inf,
        "minimum_thumb_above_pinky_m": np.inf,
        "minimum_elbow_below_shoulder_m": np.inf,
        "minimum_elbow_outside_shoulder_m": np.inf,
        "maximum_elbow_world_y_m": -np.inf,
        "first_wrist_degrees": wrist_degrees.copy(),
        "last_wrist_degrees": wrist_degrees.copy(),
        "wrist_travel_deg": np.zeros(2),
      },
    )
    stat["steps"] += 1
    stat["last_time_s"] = float(data.time)
    if self.previous_wrist_degrees is not None:
      stat["wrist_travel_deg"] += np.abs(wrist_degrees - self.previous_wrist_degrees)
    stat["last_wrist_degrees"] = wrist_degrees.copy()
    self.previous_wrist_degrees = wrist_degrees.copy()
    stat["minimum_actual_joint_margin_deg"] = np.minimum(
      stat["minimum_actual_joint_margin_deg"], actual_margin
    )
    stat["minimum_goal_joint_margin_deg"] = np.minimum(
      stat["minimum_goal_joint_margin_deg"], goal_margin
    )
    for name, value in (
      ("minimum_index_above_pinky_m", index[2] - pinky[2]),
      ("minimum_thumb_above_pinky_m", thumb[2] - pinky[2]),
      ("minimum_elbow_below_shoulder_m", -relative[2]),
      ("minimum_elbow_outside_shoulder_m", -relative[1]),
    ):
      stat[name] = min(stat[name], float(value))
    stat["maximum_palm_normal_z"] = max(stat["maximum_palm_normal_z"], palm_z)
    stat["maximum_elbow_world_y_m"] = max(
      stat["maximum_elbow_world_y_m"], float(data.xpos[self.elbow, 1])
    )
    if phase == "close":
      displacement = pinky - index
      stat["final_pinky_minus_index_x_m"] = float(displacement[0])
      stat["final_pinky_minus_index_x_cosine"] = float(
        displacement[0] / np.linalg.norm(displacement)
      )
      stat["final_pinky_minus_thumb_x_m"] = float(pinky[0] - thumb[0])
    pairs = np.column_stack((data.contact.geom1, data.contact.geom2))
    first, second = pairs[:, 0], pairs[:, 1]
    illegal = (self.robot[first] & self.obstacle[second]) | (
      self.robot[second] & self.obstacle[first]
    )
    for contact_index in np.flatnonzero(illegal):
      a, b = map(int, pairs[contact_index])
      obstacle = b if self.robot[a] else a
      key = (
        self.obstacle_categories[obstacle],
        *sorted((self.geom_names[a], self.geom_names[b])),
      )
      row = self.collisions.setdefault(
        key,
        {
          "category": key[0],
          "geoms": list(key[1:]),
          "observations": 0,
          "first_time_s": float(data.time),
          "last_time_s": float(data.time),
          "phases": [],
          "maximum_normal_force_n": 0.0,
          "maximum_penetration_m": 0.0,
        },
      )
      row["observations"] += 1
      row["last_time_s"] = float(data.time)
      if phase not in row["phases"]:
        row["phases"].append(phase)
      mujoco.mj_contactForce(simulation.model, data, int(contact_index), self.wrench)
      row["maximum_normal_force_n"] = max(
        row["maximum_normal_force_n"], abs(float(self.wrench[0]))
      )
      row["maximum_penetration_m"] = max(
        row["maximum_penetration_m"], -float(data.contact[contact_index].dist)
      )

  def report(self, result: dict | None) -> dict:
    violations = []
    actual = np.full(7, np.inf)
    goal = np.full(7, np.inf)
    for stat in self.phase_stats.values():
      actual = np.minimum(actual, stat["minimum_actual_joint_margin_deg"])
      goal = np.minimum(goal, stat["minimum_goal_joint_margin_deg"])
    if not self.steps:
      violations.append("no_physics_observations")
    for name, margin in (("actual", actual), ("goal", goal)):
      if (
        not np.isfinite(margin).all()
        or np.min(margin) < QUALITY_LIMITS[f"minimum_{name}_joint_margin_deg"]
      ):
        violations.append(f"{name}_joint_margin")
    if self.collisions:
      violations.append("robot_obstacle_contact")
    insertion = self.phase_stats.get("insert")
    if insertion is None:
      violations.append("insertion_not_observed")
    else:
      for metric, limit, sense in (
        ("maximum_palm_normal_z", "insertion_maximum_palm_normal_z_exclusive", "less"),
        (
          "minimum_index_above_pinky_m",
          "insertion_minimum_index_above_pinky_m_exclusive",
          "greater",
        ),
        (
          "minimum_thumb_above_pinky_m",
          "insertion_minimum_thumb_above_pinky_m_exclusive",
          "greater",
        ),
      ):
        value = insertion[metric]
        valid = (
          value < QUALITY_LIMITS[limit]
          if sense == "less"
          else value > QUALITY_LIMITS[limit]
        )
        if not math.isfinite(value) or not valid:
          violations.append(f"insertion_{metric}")
    placement = [
      stat for phase, stat in self.phase_stats.items() if phase in PLACEMENT_PHASES
    ]
    if not placement:
      violations.append("placement_not_observed")
    else:
      if (
        min(s["minimum_elbow_below_shoulder_m"] for s in placement)
        < QUALITY_LIMITS["placement_minimum_elbow_below_shoulder_m"]
      ):
        violations.append("placement_elbow_height")
      if (
        max(s["maximum_elbow_world_y_m"] for s in placement)
        > QUALITY_LIMITS["placement_maximum_elbow_world_y_m"]
      ):
        violations.append("placement_elbow_inward")
    if result is None:
      violations.append("execution_result_unavailable")
    else:
      if (
        not _finite(result["elapsed_simulation_s"])
        or result["elapsed_simulation_s"]
        > QUALITY_LIMITS["maximum_elapsed_simulation_s"]
      ):
        violations.append("elapsed_simulation_time")
      if (
        not _finite(result["maximum_lift_m"])
        or result["maximum_lift_m"] >= QUALITY_LIMITS["maximum_lift_m_exclusive"]
      ):
        violations.append("maximum_lift")
    return {
      "pass": not violations,
      "violations": violations,
      "limits": QUALITY_LIMITS,
      "scope": "Every executor observer call after a physics step; robot self-contact is not classified as an obstacle collision.",
      "observed_steps": self.steps,
      "minimum_actual_joint_margin_deg": actual,
      "minimum_goal_joint_margin_deg": goal,
      "robot_obstacle_contact_observations": sum(
        row["observations"] for row in self.collisions.values()
      ),
      "robot_obstacle_contacts": list(self.collisions.values()),
      "phase_stats": self.phase_stats,
    }


def _json_value(value):
  """Strict JSON: missing/non-finite measurements become null, never NaN/Infinity."""
  if isinstance(value, np.ndarray):
    return _json_value(value.tolist())
  if isinstance(value, np.generic):
    return _json_value(value.item())
  if isinstance(value, dict):
    return {key: _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def write_json_new(path: Path, payload: dict) -> None:
  with path.open("x", encoding="utf-8") as handle:
    json.dump(_json_value(payload), handle, indent=2, allow_nan=False)
    handle.write("\n")


@contextmanager
def stop_requests():
  requested = [False]

  def request(_signum, _frame):
    requested[0] = True

  previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
  for sig in previous:
    signal.signal(sig, request)
  try:
    yield lambda: requested[0]
  finally:
    for sig, handler in previous.items():
      signal.signal(sig, handler)


def run_case(case: dict, should_stop) -> dict:
  started = time.monotonic()
  initialization = result = observer = None
  status, error, failure_stage = "completed", None, "initialization"
  try:
    from kaihand_tactile_env.shared.simulation import ArmHandSimulation
    from kaihand_tactile_env.tasks.usb_insert.execution import UsbInsertionExecutor
    from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion

    simulation = ArmHandSimulation(scene="usb-insert", add_genesis_probes=True)
    initialization = initialize_for_insertion(
      simulation, **case["initialization_kwargs"]
    )
    observer = QualityObserver(simulation)
    execution = UsbInsertionExecutor(simulation, observer, should_stop)
    result = asdict(execution.execute())
    if result["failure_reason"] == "cancelled":
      status = "interrupted"
    failure_stage = result["phases"][-1] if result["phases"] else "initialization"
  except KeyboardInterrupt:
    status, error = "interrupted", "KeyboardInterrupt"
  except Exception:
    status, error = "exception", traceback.format_exc()
  if observer is not None:
    quality = observer.report(result)
    if result is None:
      failure_stage = observer.last_phase
  else:
    quality = {
      "pass": False,
      "violations": ["initialization_failed"],
      "observed_steps": 0,
    }
  success = bool(result and result["success"])
  reason = None if success else error or (result or {}).get("failure_reason") or status
  return {
    "case": case,
    "initialization": initialization,
    "status": status,
    "task_success": success,
    "quality_pass": quality["pass"],
    "qualified_success": success and quality["pass"],
    "failure_stage": None if success else failure_stage,
    "failure_reason": reason,
    "wall_duration_s": time.monotonic() - started,
    "result": result,
    "quality": quality,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--levels", nargs="+", choices=tuple(LEVELS), default=list(LEVELS)
  )
  parser.add_argument(
    "--samples",
    type=int,
    default=8,
    help="Uniform samples per level, plus nominal and eight corners",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--case-indices",
    nargs="+",
    type=int,
    help="Only run these stable zero-based indices (see --list-cases)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    help="New output directory; existing files are never overwritten",
  )
  parser.add_argument(
    "--list-cases",
    action="store_true",
    help="Print cases without importing MuJoCo or running physics",
  )
  args = parser.parse_args(argv)
  try:
    cases = generate_cases(args.levels, args.samples, args.seed)
  except ValueError as error:
    parser.error(str(error))
  if args.case_indices is not None:
    selected = set(args.case_indices)
    if len(selected) != len(args.case_indices) or selected - {
      case["index"] for case in cases
    }:
      parser.error(
        "--case-indices must be distinct indices present in the selected levels"
      )
    cases = [case for case in cases if case["index"] in selected]
  if args.list_cases:
    print(json.dumps(cases, indent=2))
    return 0
  if args.output_dir is None:
    parser.error("--output-dir is required unless --list-cases is used")
  try:
    args.output_dir.mkdir(parents=True, exist_ok=False)
  except FileExistsError:
    parser.error("--output-dir must be a new directory")
  manifest = {
    "seed": args.seed,
    "samples_per_level": args.samples,
    "execution": "serial",
    "genesis_probes": True,
    "quality_limits": QUALITY_LIMITS,
    "cases": cases,
  }
  write_json_new(args.output_dir / "manifest.json", manifest)
  reports = []
  with stop_requests() as should_stop:
    try:
      for case in cases:
        if should_stop():
          break
        directory = args.output_dir / f"{case['index']:04d}_{case['id']}"
        directory.mkdir()
        write_json_new(directory / "case.json", case)
        print(f"[{len(reports) + 1}/{len(cases)}] {case['id']}", flush=True)
        report = run_case(case, should_stop)
        write_json_new(directory / "result.json", report)
        reports.append(report)
        # Immutable progress snapshots survive interruption even before final
        # summary.json can be written, and never overwrite earlier evidence.
        write_json_new(
          args.output_dir / f"summary_{len(reports):04d}.json",
          summarize_cases(reports, cases),
        )
        print(
          f"  task={report['task_success']} quality={report['quality_pass']} stage={report['failure_stage']}",
          flush=True,
        )
        if report["status"] == "interrupted":
          break
    finally:
      summary = summarize_cases(reports, cases)
      summary["complete"] = len(reports) == len(cases) and all(
        r["status"] != "interrupted" for r in reports
      )
      write_json_new(args.output_dir / "summary.json", summary)
  if not summary["complete"]:
    return 130
  return 0 if all(r["qualified_success"] for r in reports) else 1


if __name__ == "__main__":
  raise SystemExit(main())
