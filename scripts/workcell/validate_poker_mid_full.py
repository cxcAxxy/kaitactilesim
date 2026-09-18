#!/usr/bin/env python3
"""One serial, zero-perturbation acceptance run of the middle-force full task.

This opt-in entry point does not change the regular viewer or dataset defaults.
It saves diagnostic CSV/JSON, not a training dataset. Optional software-rendered
video covers the original full scene, including pickup and inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

_THREAD_VARIABLES = (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "LP_NUM_THREADS",
)
for _variable in _THREAD_VARIABLES:
  # This bounded validation subprocess deliberately overrides inherited BLAS
  # parallelism before importing NumPy/MuJoCo; the parent shell is unchanged.
  os.environ[_variable] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco  # noqa: E402
from kaihand_tactile_env.shared.config import (  # noqa: E402
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.tasks.poker_draw.friction import (  # noqa: E402
  model_with_table_card_friction,
)
from kaihand_tactile_env.tasks.poker_draw.mid_full import (  # noqa: E402
  MID_FORCE_PER_FINGER_N,
  MID_FORCE_SETTINGS,
  MidForcePokerExecutor,
  MidForcePokerSimulation,
  configure_middle_contact_impedance,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawPlanner  # noqa: E402
from kaihand_tactile_env.tasks.poker_draw.telemetry import (  # noqa: E402
  PokerFrictionTelemetry,
)


def _pressure_helpers() -> Any:
  """Reuse bounded telemetry without depending on the shell's import path."""
  path = Path(__file__).with_name("experiment_poker_pressure_window.py")
  spec = importlib.util.spec_from_file_location("_poker_pressure_window_cli", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_helpers = _pressure_helpers()
_Observer = _helpers._Observer
_configure_contact_model = _helpers._configure_contact_model
_write_json = _helpers._write_json
PressureWindowVideoError = _helpers.PressureWindowVideoError
SCOPE = "middle_force_full_task_zero_perturbation_acceptance"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="A new directory; never overwritten."
  )
  parser.add_argument("--record", action="store_true")
  parser.add_argument("--record-fps", type=float, default=5.0)
  args = parser.parse_args(argv)
  if not math.isfinite(args.record_fps) or not 0 < args.record_fps <= 10:
    parser.error("--record-fps must be finite and in (0, 10]")
  args.output_dir = args.output_dir.expanduser().resolve()
  return args


def _source_hashes() -> dict[str, str]:
  import kaihand_tactile_env.tasks.poker_draw.mid_full as controller

  result = _helpers._source_hashes()
  for path in (Path(__file__), Path(controller.__file__)):
    result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
  return result


def _accepted(
  result: dict[str, Any] | None,
  edge: dict[str, Any] | None,
  handoff: dict[str, Any] | None,
  error: str | None,
) -> bool:
  """Never promote protocol completion, edge-only success, or partial metrics."""
  return bool(
    error is None
    and result is not None
    and result.get("success", False)
    and edge is not None
    and edge.get("target_reached", False)
    and edge.get("held_at_edge", False)
    and edge.get("full_slide_qualified", False)
    and handoff is not None
    and handoff.get("completed", False)
  )


def _make_video(simulation: Any, output: Path, args: Any, metadata: Any) -> Any:
  from kaihand_tactile_env.shared.tactile import SolverContactTactileProvider
  from kaihand_tactile_env.shared.task_video import TaskVideoRecorder

  video = TaskVideoRecorder(
    simulation,
    output,
    fps=args.record_fps,
    preview=False,
    tactile_provider=SolverContactTactileProvider(simulation.model),
    metadata=metadata,
  )
  try:
    # An oblique, fixed free camera exposes the drag direction and keeps the
    # full table-to-head path in frame. This is a visualization object only;
    # the model's calibrated cameras and all physical arrays are untouched.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.49, -0.135, 0.955]
    camera.distance = 0.70
    camera.azimuth = 135.0
    camera.elevation = -25.0
    video.follow_viewer_camera(camera)
  except Exception:
    video.close()
    raise
  return video


def _run_once(args: Any, metadata: dict[str, Any], simulation: Any) -> dict[str, Any]:
  prefix = args.output_dir / "trial_001"
  trace_path = prefix.with_name(prefix.name + "_timeseries.csv")
  video_path = prefix.with_name(prefix.name + "_video.mp4")
  telemetry = PokerFrictionTelemetry(simulation, target_force_n=MID_FORCE_PER_FINGER_N)
  video = None
  recording_error = None
  task_error = None
  outcome = None
  executor = None
  experimental_control = None
  video_artifacts = None
  if args.record:
    try:
      video = _make_video(simulation, video_path, args, metadata)
      # The oblique main camera remains fixed for the whole task. The
      # original calibrated overhead camera is unchanged.
      video.observe(simulation, "initial")
    except Exception as caught:
      recording_error = f"video initialization failed: {caught}"
      if video is not None:
        video.close()
      video = None

  try:
    with trace_path.open("x", encoding="utf-8", newline="") as stream:
      observer = _Observer(simulation, telemetry, stream, video)
      try:
        plan = PokerDrawPlanner(simulation).plan()
        executor = MidForcePokerExecutor(simulation, observer=observer)
        outcome = asdict(executor.execute(plan))
      except PressureWindowVideoError as caught:
        recording_error = str(caught)
        # The physical run was interrupted, not a measured control failure.
        observer.video = None
      except (RuntimeError, ValueError) as caught:
        task_error = str(caught)
      if float(simulation.data.time) > observer.last_time_s + 1e-9:
        try:
          observer(simulation, "terminal_unobserved")
        except PressureWindowVideoError as caught:
          recording_error = str(caught)
          observer.video = None
      # Read the controller's authoritative metadata after execution (also on
      # failure). Episode initialization may replace dynamic diagnostic lists;
      # an earlier snapshot would miss the measured lift endpoint corrections.
      control_metadata = getattr(executor, "control_metadata", None)
      if control_metadata is not None:
        experimental_control = control_metadata()
      edge = getattr(executor, "edge_outcome", None)
      handoff = getattr(executor, "handoff_outcome", None)
      accepted = _accepted(outcome, edge, handoff, task_error)
      row = {
        "trial_index": 1,
        "success": accepted,
        "protocol_completed": outcome is not None and task_error is None,
        "classification": (
          "full_task_accepted"
          if accepted
          else "control_error"
          if task_error is not None
          else "recording_interrupted"
          if recording_error is not None and outcome is None
          else "full_task_not_accepted"
        ),
        "error": task_error,
        "recording_error": recording_error,
        "task_result": outcome,
        "edge_outcome": edge,
        "handoff_outcome": handoff,
        "experimental_control": experimental_control,
        "last_observed_phase": observer.last_phase,
        "simulation_duration_s": float(simulation.data.time),
        "terminal_drive_state": simulation.drive_state(),
        "observed": observer.summary(),
        "telemetry": telemetry.summary(),
        "measurement": telemetry.metadata(),
        "timeseries_csv": str(trace_path),
        "trial_json": str(prefix.with_suffix(".json")),
      }
      stream.flush()
      # Numeric evidence exists even if final MP4 encoding/sidecar creation fails.
      _write_json(
        prefix.with_name(prefix.name + "_measurement.json"),
        {**metadata, "result": row},
      )
      if video is not None:
        try:
          video.set_outcome(row)
          mp4, sidecar = video.finish(
            success=accepted, error=task_error or recording_error
          )
          video_artifacts = {"mp4": str(mp4), "json": str(sidecar)}
        except Exception as caught:
          row["recording_error"] = f"video finalization failed: {caught}"
      row["video_artifacts"] = video_artifacts
      _write_json(prefix.with_suffix(".json"), {**metadata, "result": row})
      return row
  finally:
    if video is not None:
      video.close()


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  args.output_dir.mkdir(parents=True, exist_ok=False)
  scene = default_model_path("poker-draw")
  metadata = {
    "schema_version": 1,
    "created_utc": datetime.now(UTC).isoformat(),
    "scope": SCOPE,
    "scene": "poker-draw",
    "scene_path": str(Path(scene).resolve()),
    "scene_model_fingerprint": model_fingerprint(scene),
    "settings": asdict(MID_FORCE_SETTINGS),
    "press_force_per_finger_n": MID_FORCE_PER_FINGER_N,
    "seed": 0,
    "state_noise": False,
    "action_noise": False,
    "object_xy_jitter_m": 0.0,
    "object_yaw_jitter_rad": 0.0,
    "mujoco_version": mujoco.__version__,
    "controller_source_sha256": _source_hashes(),
    "success_semantics": "original full PokerDrawResult.success AND measured edge target/hold/full_slide_qualified AND completed supported handoff AND no control error",
    "force_limit_semantics": "4 N active world-X servo wrench budget during table-edge draw; handoff uses explicit staged controller transition; not a transient contact force cap",
    "dataset_semantics": "diagnostic acceptance CSV/JSON, not formal HDF5 training collection",
    "model_initializations": 1,
    "trial_count": 1,
    "add_genesis_probes": False,
    "serial": True,
    "headless": True,
    "record_video": args.record,
    "record_fps": args.record_fps if args.record else None,
    "video_cameras": {
      "main": "fixed_oblique_free_camera",
      "main_lookat_m": [0.49, -0.135, 0.955],
      "main_distance_m": 0.70,
      "main_azimuth_degrees": 135.0,
      "main_elevation_degrees": -25.0,
      "overhead": "fixed:overhead",
      "scope": "fixed full-task oblique MjvCamera main view, original overhead; no model camera arrays or physical state modified; recorder mode 'viewer' denotes a free camera, not a GUI",
    },
    "rendering_backend": os.environ.get("MUJOCO_GL"),
    "software_rendering_requested": os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
    "thread_environment": {key: os.environ.get(key) for key in _THREAD_VARIABLES},
  }
  _write_json(args.output_dir / "protocol.json", metadata)
  print(
    "starting one zero-noise full-task trial: mu=1, target=0.5 N/finger, slide servo budget=4 N",
    flush=True,
  )
  with model_with_table_card_friction(
    scene, MID_FORCE_SETTINGS.table_friction
  ) as model_path:
    simulation = MidForcePokerSimulation(
      model_path, scene="poker-draw", add_genesis_probes=False
    )
    metadata["contact_model"] = _configure_contact_model(simulation, MID_FORCE_SETTINGS)
    metadata["contact_model"].update(configure_middle_contact_impedance(simulation))
    simulation.reset(seed=0, object_xy_jitter=0.0, object_yaw_jitter=0.0)
    row = _run_once(args, metadata, simulation)
  _write_json(args.output_dir / "summary.json", {**metadata, "trials": [row]})
  print(
    f"finished: {row['classification']}; task_error={row['error']}; recording_error={row['recording_error']}",
    flush=True,
  )
  print(f"wrote summary: {args.output_dir / 'summary.json'}", flush=True)
  return 0 if row["success"] and row["recording_error"] is None else 1


if __name__ == "__main__":
  raise SystemExit(main())
