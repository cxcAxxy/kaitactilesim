#!/usr/bin/env python3
"""Renderer-free, resumable multi-seed audit of the v15 bulb expert."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

for _name in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
):
  os.environ.setdefault(_name, "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.tasks.bulb_screw import config  # noqa: E402
from kaihand_tactile_env.tasks.bulb_screw.execution import (  # noqa: E402
  BulbScrewExecutor,
)
from kaihand_tactile_env.tasks.bulb_screw.task import (  # noqa: E402
  BulbScrewSimulation,
)

SEED_SCHEMA = "bulb_expert_seed_audit_v1"
AGGREGATE_SCHEMA = "bulb_expert_multiseed_audit_v1"
DEFAULT_SEEDS = tuple(range(20))


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
  task_root = Path(config.__file__).resolve().parent
  package_root = task_root.parents[1]
  shared_root = package_root / "shared"
  paths = {
    "audit_bulb_expert_multiseed.py": Path(__file__).resolve(),
    "bulb_screw/config.py": task_root / "config.py",
    "bulb_screw/execution.py": task_root / "execution.py",
    "bulb_screw/grasp.py": task_root / "grasp.py",
    "bulb_screw/task.py": task_root / "task.py",
    "bulb_screw/scene.xml": task_root / "scene.xml",
    "shared/config.py": shared_root / "config.py",
    "shared/pickup_randomization.py": shared_root / "pickup_randomization.py",
    "shared/posture.py": shared_root / "posture.py",
    "shared/simulation.py": shared_root / "simulation.py",
    "shared/tactile.py": shared_root / "tactile.py",
    "tactile/layout.py": package_root / "tactile/layout.py",
    "tactile/taxels/kaihand_bimanual_fingertip_genesis.json": (
      package_root / "tactile/taxels/kaihand_bimanual_fingertip_genesis.json"
    ),
    "assets/workcell/ASSET_MANIFEST.json": (
      package_root / "assets/workcell/ASSET_MANIFEST.json"
    ),
  }
  return {name: _sha256(path) for name, path in paths.items()}


def _directory_content_identity(root: Path) -> dict:
  root = root.resolve(strict=True)
  paths = sorted(path for path in root.rglob("*") if path.is_file())
  if not paths:
    raise RuntimeError(f"asset directory is empty: {root}")
  rows = []
  total_bytes = 0
  for path in paths:
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
      raise RuntimeError(f"asset changed while hashing: {path}")
    size = after.st_size
    total_bytes += size
    rows.append(
      {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": size,
        "sha256": digest,
      }
    )
  aggregate = hashlib.sha256(
    json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  return {
    "root": str(root),
    "file_count": len(rows),
    "total_bytes": total_bytes,
    "aggregate_sha256": aggregate,
  }


def _environment_identity() -> dict:
  model_path = default_model_path(config.SCENE_NAME).resolve()
  package_root = Path(config.__file__).resolve().parents[2]
  return {
    "scene": config.SCENE_NAME,
    "mechanics_version": config.MECHANICS_VERSION,
    "rotation_driver": config.ROTATION_DRIVER,
    "model_path": str(model_path),
    "model_fingerprint": model_fingerprint(model_path),
    "physics_hz": 500,
    "runtime": {
      "python": platform.python_version(),
      "numpy": np.__version__,
      "mujoco": mujoco.__version__,
      "thread_environment": {
        name: os.environ[name]
        for name in (
          "OPENBLAS_NUM_THREADS",
          "OMP_NUM_THREADS",
          "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS",
        )
      },
    },
    "thresholds": {
      "minimum_all_finger_load_n": 4.0,
      "minimum_clockwise_torque_nm": config.TIGHTENING_MIN_TORQUE_NM,
      "maximum_rotation_span_deg": float(
        np.rad2deg(config.TIGHTENING_MAX_ROTATION_RAD)
      ),
      "tightening_hold_s": config.TIGHTENING_HOLD_S,
      "release_dwell_s": config.SEATED_DWELL_S,
      "maximum_release_fingertip_load_n": config.RELEASE_FINGERTIP_LOAD_N,
      "maximum_release_robot_load_n": config.RELEASE_HAND_LOAD_N,
    },
    "source_sha256": _source_hashes(),
    "mesh_assets": _directory_content_identity(package_root / "assets/workcell/meshes"),
  }


def _assert_environment_unchanged(expected: dict) -> None:
  if _environment_identity() != expected:
    raise RuntimeError(
      "Bulb expert audit environment changed during the multi-seed run"
    )


def _json_bytes(payload: dict) -> bytes:
  def json_default(value):
    if isinstance(value, np.generic):
      return value.item()
    if isinstance(value, np.ndarray):
      return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

  return (
    json.dumps(
      payload,
      ensure_ascii=False,
      indent=2,
      sort_keys=True,
      allow_nan=False,
      default=json_default,
    )
    + "\n"
  ).encode()


def _write_json_atomic(path: Path, payload: dict, *, overwrite: bool) -> None:
  """Publish complete JSON atomically; optionally require a new destination."""
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
  )
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "wb") as stream:
      stream.write(_json_bytes(payload))
      stream.flush()
      os.fsync(stream.fileno())
    if overwrite:
      os.replace(temporary, path)
    else:
      # link(2) makes the complete file visible without overwriting a result
      # another process or an earlier invocation already published.
      os.link(temporary, path)
      temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    if temporary.exists():
      temporary.unlink()


def _seed_path(output_dir: Path, seed: int) -> Path:
  return output_dir / f"seed_{seed:06d}.json"


class StrictTighteningObserver:
  """Freeze the executor's first exact 500 Hz certified tightening window."""

  def __init__(self, wall_limit_s: float):
    self.wall_limit_s = wall_limit_s
    self.started = time.monotonic()
    self.executor = None
    self.certified_window = None
    self.tighten_steps = 0
    self.tighten_clean_steps = 0
    self.tighten_hidden_support_steps = 0
    self.maximum_nonpad_contact_count = 0
    self.maximum_other_robot_contact_count = 0
    self.maximum_release_stable_duration_s = 0.0
    self.last_phase = "initialization"

  def bind(self, executor) -> None:
    self.executor = executor

  def __call__(self, simulation, phase: str) -> None:
    del simulation
    if time.monotonic() - self.started > self.wall_limit_s:
      raise TimeoutError(f"trial exceeded {self.wall_limit_s:g} wall seconds")
    executor = self.executor
    if executor is None:
      raise RuntimeError("strict tightening observer was not bound to its executor")
    self.last_phase = phase
    state = executor._state
    clean = bool(
      state.right_hand_nonpad_contact_count == 0
      and state.robot_contact_count == state.right_hand_contact_count
    )
    if phase == "tighten":
      self.tighten_steps += 1
      self.tighten_clean_steps += int(clean)
      self.tighten_hidden_support_steps += int(not clean)
      self.maximum_nonpad_contact_count = max(
        self.maximum_nonpad_contact_count,
        int(state.right_hand_nonpad_contact_count),
      )
      self.maximum_other_robot_contact_count = max(
        self.maximum_other_robot_contact_count,
        max(0, int(state.robot_contact_count - state.right_hand_contact_count)),
      )
    self.maximum_release_stable_duration_s = max(
      self.maximum_release_stable_duration_s,
      float(state.release_stable_duration_s),
    )
    if self.certified_window is None and executor._tightening_verified:
      self.certified_window = self._freeze_history(executor._tightening_history)

  @staticmethod
  def _freeze_history(history) -> dict:
    samples = tuple(history)
    if not samples:
      raise RuntimeError("tightening verified without a recorded history window")
    times = np.asarray([sample[0] for sample in samples], dtype=float)
    angles = np.asarray([sample[1] for sample in samples], dtype=float)
    torques = np.asarray([sample[2] for sample in samples], dtype=float)
    minimum_loads = np.asarray([sample[3] for sample in samples], dtype=float)
    seated = tuple(bool(sample[4]) for sample in samples)
    clean = tuple(bool(sample[5]) for sample in samples)
    duration = float(times[-1] - times[0])
    rotation_span_rad = float(np.ptp(angles))
    minimum_torque = float(np.min(torques))
    minimum_all_finger_load = float(np.min(minimum_loads))
    rotation_span_deg = float(np.rad2deg(rotation_span_rad))
    contract = {
      "duration": duration >= config.TIGHTENING_HOLD_S - 1e-9,
      "rotation": rotation_span_rad <= config.TIGHTENING_MAX_ROTATION_RAD,
      "torque": minimum_torque >= config.TIGHTENING_MIN_TORQUE_NM,
      "all_five_loaded": minimum_all_finger_load >= 4.0,
      "seated": all(seated),
      "clean_contact": all(clean),
    }
    return {
      "found": True,
      "sample_rate_hz": 500,
      "samples": len(samples),
      "start_s": float(times[0]),
      "end_s": float(times[-1]),
      "duration_s": duration,
      "minimum_all_finger_load_n": minimum_all_finger_load,
      "minimum_clockwise_torque_nm": minimum_torque,
      "rotation_span_deg": rotation_span_deg,
      "all_samples_seated": all(seated),
      "all_samples_clean_contact": all(clean),
      "contract_checks": contract,
      "contract_satisfied": all(contract.values()),
      "margins": {
        "duration_s": duration - config.TIGHTENING_HOLD_S,
        "all_finger_load_n": minimum_all_finger_load - 4.0,
        "clockwise_torque_nm": minimum_torque - config.TIGHTENING_MIN_TORQUE_NM,
        "rotation_deg": float(np.rad2deg(config.TIGHTENING_MAX_ROTATION_RAD))
        - rotation_span_deg,
      },
    }

  def diagnostics(self) -> dict:
    return {
      "certified_window": self.certified_window,
      "tighten_steps": self.tighten_steps,
      "tighten_clean_steps": self.tighten_clean_steps,
      "tighten_clean_fraction": (
        self.tighten_clean_steps / self.tighten_steps if self.tighten_steps else None
      ),
      "tighten_hidden_support_steps": self.tighten_hidden_support_steps,
      "maximum_nonpad_contact_count": self.maximum_nonpad_contact_count,
      "maximum_other_robot_contact_count": self.maximum_other_robot_contact_count,
      "maximum_release_stable_duration_s": (self.maximum_release_stable_duration_s),
      "last_phase": self.last_phase,
    }


def _qualification(
  result, observer: StrictTighteningObserver
) -> tuple[dict, list[str]]:
  state = result.state
  window = observer.certified_window
  release_duration = float(state.release_stable_duration_s)
  final_fingertip_loads = np.asarray(result.final_fingertip_load_n, dtype=float)
  final_fingertip_load_valid = bool(
    final_fingertip_loads.shape == (5,)
    and np.isfinite(final_fingertip_loads).all()
    and np.all(final_fingertip_loads >= 0.0)
  )
  right_hand_contact_load = float(state.right_hand_contact_load_n)
  robot_contact_load = float(state.robot_contact_load_n)
  checks = {
    "task_result_success": bool(result.success),
    "bulb_lit": bool(state.bulb_lit),
    "tightening_verified": bool(result.tightening_verified),
    "reported_tightening_stall_duration": (
      result.tightening_stall_duration_s >= config.TIGHTENING_HOLD_S - 1e-9
    ),
    "reported_tightening_peak_torque": (
      result.tightening_peak_torque_nm >= config.TIGHTENING_MIN_TORQUE_NM
    ),
    "strict_window_captured": window is not None,
    "strict_window_contract_satisfied": bool(
      window is not None and window["contract_satisfied"]
    ),
    "seated": bool(state.seated),
    "thread_engaged": bool(state.engaged),
    "released": bool(state.released),
    "release_dwell": release_duration >= config.SEATED_DWELL_S - 1e-9,
    "final_fingertip_load": (
      final_fingertip_load_valid
      and float(np.max(final_fingertip_loads)) <= config.RELEASE_FINGERTIP_LOAD_N
    ),
    "final_right_hand_contact_count_zero": state.right_hand_contact_count == 0,
    "final_right_hand_contact_load": (
      np.isfinite(right_hand_contact_load)
      and 0.0 <= right_hand_contact_load <= config.RELEASE_HAND_LOAD_N
    ),
    "final_robot_contact_count_zero": state.robot_contact_count == 0,
    "final_robot_contact_load": (
      np.isfinite(robot_contact_load)
      and 0.0 <= robot_contact_load <= config.RELEASE_HAND_LOAD_N
    ),
    "five_finger_mode": result.grasp_mode == "five-finger",
    "finger_rotation_driver": result.rotation_driver == config.ROTATION_DRIVER,
  }
  failures = [name for name, passed in checks.items() if not passed]
  return (
    {
      "checks": checks,
      "release_stable_duration_s": release_duration,
      "maximum_observed_release_stable_duration_s": (
        observer.maximum_release_stable_duration_s
      ),
      "release_dwell_margin_s": release_duration - config.SEATED_DWELL_S,
    },
    failures,
  )


def _run_seed(seed: int, environment: dict, wall_limit_s: float) -> dict:
  started = time.monotonic()
  simulation = executor = None
  observer = StrictTighteningObserver(wall_limit_s)
  row = {
    "schema": SEED_SCHEMA,
    "seed": seed,
    "environment": environment,
    "status": "exception",
    "qualified_success": False,
    "qualification_failures": ["trial_exception"],
    "initial_position_randomization": None,
    "strict_tightening": observer.diagnostics(),
    "qualification": None,
    "task_result": None,
    "error": None,
  }
  try:
    simulation = BulbScrewSimulation(position_seed=seed)
    row["initial_position_randomization"] = simulation.initial_position_randomization
    executor = BulbScrewExecutor(
      simulation,
      observer=observer,
      grasp_mode="five-finger",
      speed="fast",
    )
    observer.bind(executor)
    result = executor.execute()
    qualification, failures = _qualification(result, observer)
    row.update(
      status="completed",
      qualified_success=not failures,
      qualification_failures=failures,
      strict_tightening=observer.diagnostics(),
      qualification=qualification,
      task_result=asdict(result),
    )
  except Exception as error:  # Preserve one failed seed and continue the audit.
    row.update(
      strict_tightening=observer.diagnostics(),
      error={
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      },
    )
  finally:
    row["wall_duration_s"] = time.monotonic() - started
    del executor, simulation
  return row


def _numeric_summary(values) -> dict:
  finite = [
    float(value) for value in values if value is not None and math.isfinite(value)
  ]
  return {
    "count": len(finite),
    "minimum": min(finite) if finite else None,
    "mean": sum(finite) / len(finite) if finite else None,
    "maximum": max(finite) if finite else None,
  }


def _row_summary(row: dict, path: Path) -> dict:
  result = row.get("task_result") or {}
  state = result.get("state") or {}
  strict = row.get("strict_tightening") or {}
  window = (
    strict.get("certified_window") or {} if row.get("status") == "completed" else {}
  )
  qualification = row.get("qualification") or {}
  return {
    "seed": row["seed"],
    "path": path.name,
    "sha256": _sha256(path),
    "status": row["status"],
    "task_success": bool(result.get("success", False)),
    "qualified_success": bool(row["qualified_success"]),
    "reason": result.get("reason"),
    "qualification_failures": row["qualification_failures"],
    "wall_duration_s": row.get("wall_duration_s"),
    "elapsed_simulation_s": result.get("elapsed_s"),
    "certified_window": window or None,
    "tightening_peak_torque_nm": result.get("tightening_peak_torque_nm"),
    "release_stable_duration_s": qualification.get(
      "release_stable_duration_s", state.get("release_stable_duration_s")
    ),
    "final_robot_contact_count": state.get("robot_contact_count"),
    "final_robot_contact_load_n": state.get("robot_contact_load_n"),
    "error": row.get("error"),
  }


def _aggregate(
  rows: list[dict], paths: list[Path], seeds: list[int], environment: dict
) -> dict:
  summaries = [_row_summary(row, path) for row, path in zip(rows, paths, strict=True)]
  task_success_count = sum(row["task_success"] for row in summaries)
  qualified_count = sum(row["qualified_success"] for row in summaries)
  count = len(summaries)
  windows = [row["certified_window"] for row in summaries if row["certified_window"]]

  def window_values(name):
    return [window[name] for window in windows]

  margins = [window["margins"] for window in windows]
  release_durations = [row["release_stable_duration_s"] for row in summaries]
  release_margins = [
    value - config.SEATED_DWELL_S for value in release_durations if value is not None
  ]
  failure_reasons = Counter(
    row["reason"] or (row["error"] or {}).get("type") or "unknown"
    for row in summaries
    if not row["task_success"]
  )
  qualification_failures = Counter(
    failure for row in summaries for failure in row["qualification_failures"]
  )
  return {
    "schema": AGGREGATE_SCHEMA,
    "created_utc": datetime.now(UTC).isoformat(),
    "mechanics_version": environment["mechanics_version"],
    "model_fingerprint": environment["model_fingerprint"],
    "environment": environment,
    "requested_seeds": seeds,
    "planned_count": len(seeds),
    "reported_count": count,
    "completed_count": sum(row["status"] == "completed" for row in summaries),
    "exception_count": sum(row["status"] == "exception" for row in summaries),
    "task_success_count": task_success_count,
    "task_success_rate": task_success_count / count if count else None,
    "qualified_success_count": qualified_count,
    "qualified_success_rate": qualified_count / count if count else None,
    "failure_reasons": dict(sorted(failure_reasons.items())),
    "qualification_failures": dict(sorted(qualification_failures.items())),
    "certified_window_count": len(windows),
    "certified_window_rate": len(windows) / count if count else None,
    "certified_clean_window_count": sum(
      bool(window["all_samples_clean_contact"]) for window in windows
    ),
    "certified_clean_window_rate": (
      sum(bool(window["all_samples_clean_contact"]) for window in windows) / count
      if count
      else None
    ),
    "certified_clean_rate_among_captured_windows": (
      sum(bool(window["all_samples_clean_contact"]) for window in windows)
      / len(windows)
      if windows
      else None
    ),
    "metrics": {
      "elapsed_simulation_s": _numeric_summary(
        row["elapsed_simulation_s"] for row in summaries
      ),
      "wall_duration_s": _numeric_summary(row["wall_duration_s"] for row in summaries),
      "certified_window_duration_s": _numeric_summary(window_values("duration_s")),
      "certified_minimum_all_finger_load_n": _numeric_summary(
        window_values("minimum_all_finger_load_n")
      ),
      "certified_minimum_clockwise_torque_nm": _numeric_summary(
        window_values("minimum_clockwise_torque_nm")
      ),
      "certified_rotation_span_deg": _numeric_summary(
        window_values("rotation_span_deg")
      ),
      "tightening_peak_torque_nm": _numeric_summary(
        row["tightening_peak_torque_nm"] for row in summaries
      ),
      "release_stable_duration_s": _numeric_summary(release_durations),
    },
    "margin_metrics": {
      "tightening_duration_s": _numeric_summary(
        margin["duration_s"] for margin in margins
      ),
      "all_finger_load_n": _numeric_summary(
        margin["all_finger_load_n"] for margin in margins
      ),
      "clockwise_torque_nm": _numeric_summary(
        margin["clockwise_torque_nm"] for margin in margins
      ),
      "rotation_deg": _numeric_summary(margin["rotation_deg"] for margin in margins),
      "release_dwell_s": _numeric_summary(release_margins),
    },
    "seeds": summaries,
  }


def _require_json_boolean(value, field: str) -> bool:
  if type(value) is not bool:
    raise TypeError(f"{field} must be a JSON boolean")
  return value


def _require_finite_number(value, field: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{field} must be a finite JSON number")
  number = float(value)
  if not math.isfinite(number):
    raise ValueError(f"{field} must be a finite JSON number")
  return number


def _require_nonnegative_integer(value, field: str) -> int:
  if type(value) is not int or value < 0:
    raise TypeError(f"{field} must be a nonnegative JSON integer")
  return value


def _validate_completed_semantics(row: dict, path: Path) -> None:
  """Recompute every qualification bit before trusting a resumed result."""
  try:
    result = row["task_result"]
    if not isinstance(result, dict):
      raise TypeError("task_result must be an object")
    state = result["state"]
    if not isinstance(state, dict):
      raise TypeError("task_result.state must be an object")
    strict = row["strict_tightening"]
    window = strict.get("certified_window")
    qualification = row["qualification"]
    if not isinstance(qualification, dict):
      raise TypeError("qualification must be an object")
    if window is None:
      strict_contract = False
    else:
      if not isinstance(window, dict):
        raise TypeError("certified_window must be an object or null")
      duration = _require_finite_number(
        window["duration_s"], "certified_window.duration_s"
      )
      rotation = _require_finite_number(
        window["rotation_span_deg"], "certified_window.rotation_span_deg"
      )
      minimum_torque = _require_finite_number(
        window["minimum_clockwise_torque_nm"],
        "certified_window.minimum_clockwise_torque_nm",
      )
      minimum_load = _require_finite_number(
        window["minimum_all_finger_load_n"],
        "certified_window.minimum_all_finger_load_n",
      )
      start = _require_finite_number(window["start_s"], "certified_window.start_s")
      end = _require_finite_number(window["end_s"], "certified_window.end_s")
      sample_rate = _require_nonnegative_integer(
        window["sample_rate_hz"], "certified_window.sample_rate_hz"
      )
      samples = _require_nonnegative_integer(
        window["samples"], "certified_window.samples"
      )
      found = _require_json_boolean(window["found"], "certified_window.found")
      if (
        not found
        or sample_rate != 500
        or samples < 1
        or min(start, end, duration, rotation, minimum_load) < 0.0
        or end < start
        or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(
          duration,
          (samples - 1) / sample_rate,
          rel_tol=0.0,
          abs_tol=1e-9,
        )
      ):
        raise ValueError("certified window timing/load domain is invalid")
      all_seated = _require_json_boolean(
        window["all_samples_seated"], "certified_window.all_samples_seated"
      )
      all_clean = _require_json_boolean(
        window["all_samples_clean_contact"],
        "certified_window.all_samples_clean_contact",
      )
      expected_contract = {
        "duration": duration >= config.TIGHTENING_HOLD_S - 1e-9,
        "rotation": rotation <= float(np.rad2deg(config.TIGHTENING_MAX_ROTATION_RAD)),
        "torque": minimum_torque >= config.TIGHTENING_MIN_TORQUE_NM,
        "all_five_loaded": minimum_load >= 4.0,
        "seated": all_seated,
        "clean_contact": all_clean,
      }
      strict_contract = all(expected_contract.values())
      contract_checks = window.get("contract_checks")
      if not isinstance(contract_checks, dict) or set(contract_checks) != set(
        expected_contract
      ):
        raise TypeError("certified window contract checks have an invalid shape")
      for name, value in contract_checks.items():
        _require_json_boolean(value, f"certified_window.contract_checks.{name}")
      if contract_checks != expected_contract:
        raise ValueError("certified window contract checks do not match its metrics")
      if (
        _require_json_boolean(
          window.get("contract_satisfied"), "certified_window.contract_satisfied"
        )
        is not strict_contract
      ):
        raise ValueError("certified window contract result does not match its metrics")
      expected_margins = {
        "duration_s": duration - config.TIGHTENING_HOLD_S,
        "all_finger_load_n": minimum_load - 4.0,
        "clockwise_torque_nm": minimum_torque - config.TIGHTENING_MIN_TORQUE_NM,
        "rotation_deg": float(np.rad2deg(config.TIGHTENING_MAX_ROTATION_RAD))
        - rotation,
      }
      margins = window["margins"]
      if not isinstance(margins, dict) or set(margins) != set(expected_margins):
        raise TypeError("certified window margins have an invalid shape")
      if any(
        not math.isclose(
          _require_finite_number(margins[name], f"certified_window.margins.{name}"),
          value,
          rel_tol=0.0,
          abs_tol=1e-12,
        )
        for name, value in expected_margins.items()
      ):
        raise ValueError("certified window margins do not match its metrics")

    result_success = _require_json_boolean(result["success"], "task_result.success")
    tightening_verified = _require_json_boolean(
      result["tightening_verified"], "task_result.tightening_verified"
    )
    bulb_lit = _require_json_boolean(state["bulb_lit"], "state.bulb_lit")
    seated = _require_json_boolean(state["seated"], "state.seated")
    engaged = _require_json_boolean(state["engaged"], "state.engaged")
    released = _require_json_boolean(state["released"], "state.released")
    tightening_stall_duration = _require_finite_number(
      result["tightening_stall_duration_s"],
      "task_result.tightening_stall_duration_s",
    )
    tightening_peak_torque = _require_finite_number(
      result["tightening_peak_torque_nm"],
      "task_result.tightening_peak_torque_nm",
    )
    release_duration = _require_finite_number(
      state["release_stable_duration_s"], "state.release_stable_duration_s"
    )
    final_load_values = result["final_fingertip_load_n"]
    if not isinstance(final_load_values, list):
      raise TypeError("task_result.final_fingertip_load_n must be an array")
    final_loads = [
      _require_finite_number(value, "task_result.final_fingertip_load_n[]")
      for value in final_load_values
    ]
    if len(final_loads) != 5 or any(value < 0.0 for value in final_loads):
      raise ValueError(
        "task_result.final_fingertip_load_n must contain five nonnegative values"
      )
    right_hand_contact_count = _require_nonnegative_integer(
      state["right_hand_contact_count"], "state.right_hand_contact_count"
    )
    right_hand_contact_load = _require_finite_number(
      state["right_hand_contact_load_n"], "state.right_hand_contact_load_n"
    )
    robot_contact_count = _require_nonnegative_integer(
      state["robot_contact_count"], "state.robot_contact_count"
    )
    robot_contact_load = _require_finite_number(
      state["robot_contact_load_n"], "state.robot_contact_load_n"
    )
    if min(right_hand_contact_load, robot_contact_load, tightening_peak_torque) < 0.0:
      raise ValueError("reported contact loads and peak torque must be nonnegative")
    expected_checks = {
      "task_result_success": result_success,
      "bulb_lit": bulb_lit,
      "tightening_verified": tightening_verified,
      "reported_tightening_stall_duration": (
        tightening_stall_duration >= config.TIGHTENING_HOLD_S - 1e-9
      ),
      "reported_tightening_peak_torque": (
        tightening_peak_torque >= config.TIGHTENING_MIN_TORQUE_NM
      ),
      "strict_window_captured": window is not None,
      "strict_window_contract_satisfied": strict_contract,
      "seated": seated,
      "thread_engaged": engaged,
      "released": released,
      "release_dwell": release_duration >= config.SEATED_DWELL_S - 1e-9,
      "final_fingertip_load": (
        max(final_loads, default=0.0) <= config.RELEASE_FINGERTIP_LOAD_N
      ),
      "final_right_hand_contact_count_zero": right_hand_contact_count == 0,
      "final_right_hand_contact_load": (
        right_hand_contact_load <= config.RELEASE_HAND_LOAD_N
      ),
      "final_robot_contact_count_zero": robot_contact_count == 0,
      "final_robot_contact_load": robot_contact_load <= config.RELEASE_HAND_LOAD_N,
      "five_finger_mode": result["grasp_mode"] == "five-finger",
      "finger_rotation_driver": result["rotation_driver"] == config.ROTATION_DRIVER,
    }
    stored_checks = qualification.get("checks")
    if not isinstance(stored_checks, dict) or set(stored_checks) != set(
      expected_checks
    ):
      raise TypeError("stored qualification checks have an invalid shape")
    for name, value in stored_checks.items():
      _require_json_boolean(value, f"qualification.checks.{name}")
    if stored_checks != expected_checks:
      raise ValueError("stored qualification checks do not match the task result")
    stored_release_duration = _require_finite_number(
      qualification["release_stable_duration_s"],
      "qualification.release_stable_duration_s",
    )
    if not math.isclose(
      stored_release_duration,
      release_duration,
      rel_tol=0.0,
      abs_tol=1e-12,
    ):
      raise ValueError("stored release duration does not match the terminal state")
    stored_release_margin = _require_finite_number(
      qualification["release_dwell_margin_s"],
      "qualification.release_dwell_margin_s",
    )
    if not math.isclose(
      stored_release_margin,
      release_duration - config.SEATED_DWELL_S,
      rel_tol=0.0,
      abs_tol=1e-12,
    ):
      raise ValueError("stored release margin does not match the terminal state")
    maximum_release_duration = _require_finite_number(
      qualification["maximum_observed_release_stable_duration_s"],
      "qualification.maximum_observed_release_stable_duration_s",
    )
    if maximum_release_duration + 1e-12 < release_duration:
      raise ValueError("maximum observed release duration is below the terminal value")
    elapsed = _require_finite_number(result["elapsed_s"], "task_result.elapsed_s")
    wall_duration = _require_finite_number(row["wall_duration_s"], "wall_duration_s")
    if min(elapsed, wall_duration, release_duration, tightening_stall_duration) < 0:
      raise ValueError("durations must be nonnegative")
    if row.get("error") is not None:
      raise ValueError("completed result error must be null")
  except (KeyError, TypeError, ValueError) as error:
    raise ValueError(
      f"inconsistent completed resume seed result {path}: {error}"
    ) from error
  failures = [name for name, passed in expected_checks.items() if not passed]
  if row["qualification_failures"] != failures:
    raise ValueError(f"resume qualification failures are inconsistent: {path}")
  if row["qualified_success"] is not (not failures):
    raise ValueError(f"resume qualified_success is inconsistent: {path}")


def _read_seed_result(path: Path, seed: int, environment: dict) -> dict:
  try:
    row = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid resume seed result {path}: {error}") from error
  if row.get("schema") != SEED_SCHEMA:
    raise ValueError(f"resume seed result has wrong schema: {path}")
  if type(row.get("seed")) is not int or row["seed"] != seed:
    raise ValueError(f"resume seed result has wrong seed: {path}")
  randomization = row.get("initial_position_randomization")
  if row.get("status") == "completed" and (
    not isinstance(randomization, dict)
    or randomization.get("seed") != seed
    or isinstance(randomization.get("seed"), bool)
  ):
    raise ValueError(f"resume seed result randomization seed is inconsistent: {path}")
  if row.get("environment") != environment:
    raise ValueError(f"resume seed result has stale environment provenance: {path}")
  if row.get("status") not in {"completed", "exception"}:
    raise ValueError(f"resume seed result has invalid status: {path}")
  if not isinstance(row.get("qualified_success"), bool):
    raise ValueError(f"resume seed result lacks qualified_success: {path}")
  if not isinstance(row.get("qualification_failures"), list):
    raise ValueError(f"resume seed result lacks qualification failures: {path}")
  if not isinstance(row.get("strict_tightening"), dict):
    raise ValueError(f"resume seed result lacks strict tightening diagnostics: {path}")
  if row["status"] == "completed" and not isinstance(row.get("task_result"), dict):
    raise ValueError(f"completed resume seed result lacks task_result: {path}")
  if row["status"] == "exception" and not isinstance(row.get("error"), dict):
    raise ValueError(f"exception resume seed result lacks error details: {path}")
  if row["status"] == "completed":
    _validate_completed_semantics(row, path)
  else:
    error = row["error"]
    if row.get("task_result") is not None or row.get("qualification") is not None:
      raise ValueError(
        f"exception resume seed result task_result and qualification must be null: {path}"
      )
    if (
      not isinstance(error.get("type"), str)
      or not error["type"]
      or not isinstance(error.get("message"), str)
      or not isinstance(error.get("traceback"), str)
    ):
      raise ValueError(
        f"exception resume seed result has invalid error details: {path}"
      )
    try:
      wall_duration = _require_finite_number(row["wall_duration_s"], "wall_duration_s")
    except (KeyError, TypeError, ValueError) as validation_error:
      raise ValueError(
        f"exception resume seed result has invalid duration {path}: {validation_error}"
      ) from validation_error
    if wall_duration < 0:
      raise ValueError(f"exception resume seed result has negative duration: {path}")
    if row["qualified_success"] or row["qualification_failures"] != ["trial_exception"]:
      raise ValueError(f"exception resume seed result has inconsistent outcome: {path}")
  return row


def _validate_existing_aggregate(
  path: Path, seeds: list[int], environment: dict
) -> dict[int, str]:
  if not path.exists():
    return {}
  try:
    aggregate = json.loads(path.read_text())
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid existing aggregate {path}: {error}") from error
  if (
    aggregate.get("schema") != AGGREGATE_SCHEMA
    or aggregate.get("requested_seeds") != seeds
    or aggregate.get("environment") != environment
  ):
    raise ValueError(f"existing aggregate does not match this resumed audit: {path}")
  entries = aggregate.get("seeds")
  if not isinstance(entries, list) or len(entries) != len(seeds):
    raise ValueError(f"existing aggregate has an invalid seed inventory: {path}")
  inventory = {}
  for entry in entries:
    if (
      not isinstance(entry, dict)
      or type(entry.get("seed")) is not int
      or entry["seed"] not in seeds
    ):
      raise ValueError(f"existing aggregate has an invalid seed entry: {path}")
    seed = entry["seed"]
    digest = entry.get("sha256")
    if (
      seed in inventory
      or entry.get("path") != _seed_path(path.parent, seed).name
      or not isinstance(digest, str)
      or len(digest) != 64
    ):
      raise ValueError(f"existing aggregate has an invalid seed identity: {path}")
    inventory[seed] = digest
  return inventory


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
  parser.add_argument(
    "--trial-wall-limit-s",
    type=float,
    default=180.0,
    help="Per-seed wall timeout checked at every observed physics step",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Reuse only complete seed files with identical environment provenance",
  )
  args = parser.parse_args(argv)
  if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
    parser.error("--seeds must be distinct nonnegative integers")
  if not math.isfinite(args.trial_wall_limit_s) or args.trial_wall_limit_s <= 0:
    parser.error("--trial-wall-limit-s must be finite and positive")
  return args


def main(argv=None) -> int:
  args = _parse_args(argv)
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  aggregate_path = output_dir / "aggregate.json"
  environment = _environment_identity()
  seeds = list(args.seeds)
  seed_paths = [_seed_path(output_dir, seed) for seed in seeds]
  expected_hashes = {}
  if not args.resume:
    existing = [path for path in (*seed_paths, aggregate_path) if path.exists()]
    if existing:
      raise FileExistsError(
        "audit outputs already exist; use a new directory or --resume: "
        + ", ".join(str(path) for path in existing)
      )
  else:
    expected_hashes = _validate_existing_aggregate(aggregate_path, seeds, environment)

  rows = []
  for index, (seed, path) in enumerate(zip(seeds, seed_paths, strict=True), start=1):
    if expected_hashes and not path.exists():
      raise ValueError(f"resume seed result listed by aggregate is missing: {path}")
    if path.exists():
      if expected_hashes and expected_hashes.get(seed) != _sha256(path):
        raise ValueError(f"resume seed result hash does not match aggregate: {path}")
      row = _read_seed_result(path, seed, environment)
      print(f"[{index}/{len(seeds)}] seed={seed} resume {row['status']}", flush=True)
    else:
      _assert_environment_unchanged(environment)
      print(f"[{index}/{len(seeds)}] seed={seed} start", flush=True)
      row = _run_seed(seed, environment, args.trial_wall_limit_s)
      _write_json_atomic(path, row, overwrite=False)
      print(
        f"[{index}/{len(seeds)}] seed={seed} {row['status']} "
        f"qualified={row['qualified_success']} wall={row['wall_duration_s']:.1f}s",
        flush=True,
      )
      gc.collect()
    rows.append(row)

  _assert_environment_unchanged(environment)
  aggregate = _aggregate(rows, seed_paths, seeds, environment)
  _write_json_atomic(aggregate_path, aggregate, overwrite=args.resume)
  print(_json_bytes(aggregate).decode(), end="")
  return 0 if aggregate["qualified_success_count"] == len(seeds) else 1


if __name__ == "__main__":
  raise SystemExit(main())
