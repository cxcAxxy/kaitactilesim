#!/usr/bin/env python3
"""Record synchronized grasp episodes to HDF5 plus JSON manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import (
  SHARED_CAMERA_NAMES,
  TRAINING_CAMERA_NAMES,
  CameraConfig,
  WorkcellConfig,
  default_model_path,
  model_fingerprint,
  task_config,
)
from kaihand_tactile_env.shared.recording import (
  POKER_FORCE_RECORDING_CONTRACT,
  TASK_ISOLATED_MODEL_LAYOUT,
  EpisodeRecorder,
  validate_episode,
  wait_until_object_stable,
)
from kaihand_tactile_env.shared.rendering import set_fixed_camera_lookat
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import (
  RIGHT_FINGERTIP_LINK_NAMES,
  SolverContactTactileProvider,
)
from kaihand_tactile_env.tasks.pick_place.task import (
  KnownStateGraspPlanner,
  PickPlaceExecutor,
  cylinder_is_in_box,
)
from kaihand_tactile_env.tasks.poker_draw.acceptance import (
  ACCEPTANCE_POLICIES,
  STRICT_FORCE_POLICY,
  accept_edge,
  check_policy,
)
from kaihand_tactile_env.tasks.poker_draw.mid_full import (
  MID_FORCE_PER_FINGER_N,
  MID_FORCE_SETTINGS,
  MidForcePokerExecutor,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (
  DEFAULT_PRECONTACT_STD_RAD,
  PRECONTACT_PRESET,
  precontact_noise_settings,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (
  DEFAULT_XY_JITTER_M as PRECONTACT_XY_JITTER_M,
)
from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (
  DEFAULT_YAW_JITTER_RAD as PRECONTACT_YAW_JITTER_RAD,
)
from kaihand_tactile_env.tasks.poker_draw.randomization import (
  DEFAULT_XY_JITTER_M,
  RANDOMIZED_PRESET,
  reset_randomized_card,
  validate_randomization_bounds,
  validate_recorded_randomization,
)
from kaihand_tactile_env.tasks.poker_draw.task import (
  PokerDrawExecutor,
  PokerDrawPlanner,
)

RECORDING_CONTRACT_VERSION = "pick_place_stable_terminal_v2"
POKER_RECORDING_CONTRACT_VERSION = POKER_FORCE_RECORDING_CONTRACT
PRODUCTION_PRESET = "production"
MIDDLE_FORCE_PRESET = "middle-force-v1"
MIDDLE_FORCE_PRESETS = (MIDDLE_FORCE_PRESET, RANDOMIZED_PRESET, PRECONTACT_PRESET)
INITIAL_RANDOMIZATION_PRESETS = (RANDOMIZED_PRESET, PRECONTACT_PRESET)


@dataclass(frozen=True)
class RecordJob:
  """One independently reproducible episode assigned to a worker process."""

  episode_index: int
  output: Path
  scene: str
  object_name: str
  side: str
  episode_seed: int
  config: WorkcellConfig
  tactile_links: str
  object_xy_jitter: float
  object_yaw_jitter: float
  overhead_pos: tuple[float, float, float] | None
  overhead_lookat: tuple[float, float, float]
  overhead_fovy: float | None
  front_pos: tuple[float, float, float] | None
  front_lookat: tuple[float, float, float]
  front_fovy: float | None
  overwrite: bool
  press_force_per_finger_n: float | None = None
  preset: str = PRODUCTION_PRESET
  precontact_noise_std_rad: float | None = None
  acceptance_policy: str = STRICT_FORCE_POLICY


@dataclass(frozen=True)
class EpisodeSummary:
  """Small process-safe result returned after an episode is validated."""

  episode_index: int
  output: Path
  success: bool
  placed_in_box: bool | None
  state_samples: int
  camera_samples: dict[str, int]


@dataclass(frozen=True)
class EpisodeFailure:
  """Machine-readable record of one failed job that did not stop the batch."""

  episode_index: int
  output: Path
  error_type: str
  error_message: str
  partial_path: Path | None


EpisodeResult = EpisodeSummary | EpisodeFailure


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output-dir",
    type=Path,
    help=(
      "Episode directory. Defaults to datasets/pick_place or "
      "datasets/poker_draw for the selected scene."
    ),
  )
  parser.add_argument("--episodes", type=int, default=1)
  parser.add_argument(
    "--start-index",
    type=int,
    default=0,
    help="First global episode index; useful when resuming a dataset shard.",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help=(
      "Number of independent spawned MuJoCo processes. Each worker records "
      "one episode at a time into its own HDF5 file."
    ),
  )
  parser.add_argument(
    "--scene",
    choices=("pick-place", "poker-draw"),
    default="pick-place",
  )
  parser.add_argument(
    "--preset",
    choices=(PRODUCTION_PRESET, *MIDDLE_FORCE_PRESETS),
    default=PRODUCTION_PRESET,
    help=(
      "Opt in to middle-force, bounded initial card-pose randomization, or "
      "precontact arm-control noise; production preserves original settings."
    ),
  )
  parser.add_argument("--acceptance-policy", choices=ACCEPTANCE_POLICIES,
                      default=STRICT_FORCE_POLICY,
                      help="Opt-in poker training admission; force control and raw signals are unchanged")
  parser.add_argument("--object", choices=("cylinder", "card"))
  parser.add_argument(
    "--tactile-links",
    choices=("all", "right"),
    default="all",
    help="Optionally restrict the solver proxy baseline to right fingertips.",
  )
  parser.add_argument("--side", choices=("left", "right"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--camera-hz", type=int, default=30)
  parser.add_argument(
    "--cameras",
    nargs="+",
    choices=SHARED_CAMERA_NAMES,
    default=TRAINING_CAMERA_NAMES,
    help="Camera streams to save; defaults to head/left_wrist/right_wrist.",
  )
  parser.add_argument(
    "--rgb-only",
    action="store_true",
    help="Save RGB only; omit depth and segmentation to reduce rendering load.",
  )
  parser.add_argument("--no-cameras", action="store_true")
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help=(
      "Replace completed, partial, or failure artifacts for the selected "
      "episode indices; active worker locks are never replaced."
    ),
  )
  parser.add_argument(
    "--fail-fast",
    action="store_true",
    help=(
      "Stop the batch at the first failed episode. By default a failed episode "
      "gets a .failure.json report and independent jobs continue."
    ),
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help=(
      "Keep completed episodes whose HDF5 validation and manifest hash pass, "
      "and retry missing, partial, failed, or invalid episode indices."
    ),
  )
  parser.add_argument("--object-xy-jitter", type=float)
  parser.add_argument("--object-yaw-jitter", type=float)
  parser.add_argument(
    "--precontact-noise-std-deg",
    type=float,
    help=(
      "Gaussian right-arm control-target standard deviation in degrees, only "
      "for middle-force-precontact-v1 (default 0.03, range 0 through 0.05). "
      "Zero provides a paired control with identical initial-pose sampling."
    ),
  )
  parser.add_argument(
    "--press-force-per-finger",
    "--press-force",
    dest="press_force_per_finger_n",
    type=float,
    help="Poker slide normal-force target in N per finger (four fingers).",
  )
  parser.add_argument(
    "--tactile-source",
    choices=("solver_contact_proxy_v1", "genesis_probe_bimanual_clean_v1"),
    default=None,
    help=(
      "Choose the solver proxy or accepted bilateral Genesis probes. Defaults "
      "to Genesis for production, solver proxy for the middle-force presets."
    ),
  )
  parser.add_argument("--overhead-pos", type=float, nargs=3)
  parser.add_argument(
    "--overhead-lookat", type=float, nargs=3, default=(0.56, 0.0, 0.72)
  )
  parser.add_argument("--overhead-fovy", type=float)
  parser.add_argument("--front-pos", type=float, nargs=3)
  parser.add_argument("--front-lookat", type=float, nargs=3, default=(0.56, 0.0, 0.78))
  parser.add_argument("--front-fovy", type=float)
  args = parser.parse_args(argv)
  if args.episodes <= 0:
    parser.error("--episodes must be positive")
  if args.workers <= 0:
    parser.error("--workers must be positive")
  if args.start_index < 0:
    parser.error("--start-index cannot be negative")
  if len(args.cameras) != len(set(args.cameras)):
    parser.error("--cameras cannot contain duplicates")
  defaults = task_config(args.scene)
  if args.acceptance_policy != STRICT_FORCE_POLICY and args.preset not in MIDDLE_FORCE_PRESETS:
    parser.error("--acceptance-policy task-completion-v1 requires a middle-force poker preset")
  if args.preset in MIDDLE_FORCE_PRESETS:
    if args.scene != "poker-draw":
      parser.error(f"--preset {args.preset} requires --scene poker-draw")
    if args.workers != 1:
      parser.error(f"--preset {args.preset} currently requires --workers 1")
    if args.press_force_per_finger_n is None:
      args.press_force_per_finger_n = MID_FORCE_PER_FINGER_N
    elif args.press_force_per_finger_n != MID_FORCE_PER_FINGER_N:
      parser.error(f"--preset {args.preset} fixes --press-force to 0.50 N")
    if args.tactile_source not in (None, SolverContactTactileProvider.source):
      parser.error(
        f"--preset {args.preset} requires --tactile-source solver_contact_proxy_v1"
      )
  if args.tactile_source is None:
    args.tactile_source = (
      SolverContactTactileProvider.source
      if args.preset in MIDDLE_FORCE_PRESETS
      else "genesis_probe_bimanual_clean_v1"
    )
  if args.press_force_per_finger_n is not None:
    if args.scene != "poker-draw":
      parser.error("--press-force requires --scene poker-draw")
    if (
      not math.isfinite(args.press_force_per_finger_n)
      or args.press_force_per_finger_n <= 0
    ):
      parser.error("--press-force must be finite and positive (N per finger)")
  elif args.scene == "poker-draw":
    args.press_force_per_finger_n = defaults.DEFAULT_PRESS_FORCE_PER_FINGER_N
  if args.object_xy_jitter is None:
    args.object_xy_jitter = (
      PRECONTACT_XY_JITTER_M
      if args.preset == PRECONTACT_PRESET
      else (
        DEFAULT_XY_JITTER_M
        if args.preset == RANDOMIZED_PRESET
        else defaults.DEFAULT_XY_JITTER
      )
    )
  if args.object_yaw_jitter is None:
    args.object_yaw_jitter = (
      PRECONTACT_YAW_JITTER_RAD
      if args.preset == PRECONTACT_PRESET
      else defaults.DEFAULT_YAW_JITTER
    )
  args.precontact_noise_std_rad = None
  if args.preset == PRECONTACT_PRESET:
    args.precontact_noise_std_rad = (
      DEFAULT_PRECONTACT_STD_RAD
      if args.precontact_noise_std_deg is None
      else math.radians(args.precontact_noise_std_deg)
    )
    try:
      precontact_noise_settings(args.precontact_noise_std_rad)
    except ValueError as error:
      parser.error(str(error))
  elif args.precontact_noise_std_deg is not None:
    parser.error(
      "--precontact-noise-std-deg requires --preset middle-force-precontact-v1"
    )
  if (
    not math.isfinite(args.object_xy_jitter)
    or not math.isfinite(args.object_yaw_jitter)
    or args.object_xy_jitter < 0.0
    or args.object_yaw_jitter < 0.0
  ):
    parser.error("object jitter must be finite and non-negative")
  if args.preset == MIDDLE_FORCE_PRESET and (
    args.object_xy_jitter != 0.0 or args.object_yaw_jitter != 0.0
  ):
    parser.error("--preset middle-force-v1 currently requires zero object jitter")
  if args.preset in INITIAL_RANDOMIZATION_PRESETS:
    if args.seed < 0:
      parser.error(f"--preset {args.preset} requires a non-negative --seed")
    try:
      validate_randomization_bounds(args.object_xy_jitter, args.object_yaw_jitter)
    except ValueError as error:
      parser.error(str(error))
  if args.resume and args.overwrite:
    parser.error("--resume and --overwrite are mutually exclusive")
  expected_object = "card" if args.scene == "poker-draw" else "cylinder"
  if args.object is not None and args.object != expected_object:
    parser.error(f"--scene {args.scene} requires --object {expected_object}")
  args.object = expected_object
  if args.scene == "poker-draw" and args.side not in (None, "right"):
    parser.error("--scene poker-draw currently requires --side right")
  if args.output_dir is None:
    args.output_dir = Path("datasets") / args.scene.replace("-", "_")
  return args


def _build_jobs(args: argparse.Namespace) -> tuple[RecordJob, ...]:
  """Resolve CLI options into deterministic, independently picklable jobs."""

  cameras = ()
  if not args.no_cameras:
    cameras = tuple(
      CameraConfig(
        name,
        args.width,
        args.height,
        depth=not args.rgb_only,
        segmentation=not args.rgb_only,
      )
      for name in args.cameras
    )
  config = WorkcellConfig(
    model_path=default_model_path(args.scene),
    camera_hz=args.camera_hz,
    cameras=cameras,
    tactile_provider=args.tactile_source,
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)

  jobs = []
  for offset in range(args.episodes):
    episode_index = args.start_index + offset
    object_name = _object_for_episode(args.object, episode_index)
    side = args.side or "right"
    episode_seed = args.seed + episode_index
    output = args.output_dir / f"episode_{episode_index:06d}_{object_name}_{side}.h5"
    jobs.append(
      RecordJob(
        episode_index=episode_index,
        output=output,
        scene=args.scene,
        object_name=object_name,
        side=side,
        episode_seed=episode_seed,
        config=config,
        tactile_links=args.tactile_links,
        object_xy_jitter=args.object_xy_jitter,
        object_yaw_jitter=args.object_yaw_jitter,
        overhead_pos=_triplet_or_none(args.overhead_pos),
        overhead_lookat=_triplet(args.overhead_lookat),
        overhead_fovy=args.overhead_fovy,
        front_pos=_triplet_or_none(args.front_pos),
        front_lookat=_triplet(args.front_lookat),
        front_fovy=args.front_fovy,
        overwrite=args.overwrite or args.resume,
        press_force_per_finger_n=args.press_force_per_finger_n,
        preset=args.preset,
        precontact_noise_std_rad=args.precontact_noise_std_rad,
        acceptance_policy=args.acceptance_policy,
      )
    )
  resolved_jobs = tuple(jobs)
  outputs = [job.output for job in resolved_jobs]
  if len(set(outputs)) != len(outputs):
    raise ValueError("episode output paths must be unique")
  if args.resume:
    return _select_resume_jobs(resolved_jobs)
  _check_output_conflicts(resolved_jobs, overwrite=args.overwrite)
  return resolved_jobs


def _record_episode(job: RecordJob) -> EpisodeSummary:
  """Record one scene entirely inside the process that owns its MuJoCo state."""
  lock_path = job.output.with_suffix(job.output.suffix + ".lock")
  try:
    lock_descriptor = os.open(
      lock_path,
      os.O_CREAT | os.O_EXCL | os.O_WRONLY,
      0o644,
    )
  except FileExistsError as error:
    raise RuntimeError(
      f"episode {job.episode_index} is already claimed: {lock_path}"
    ) from error
  try:
    os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
    with _recording_termination_guard(job):
      return _record_claimed_episode(job)
  finally:
    os.close(lock_descriptor)
    lock_path.unlink(missing_ok=True)


@contextmanager
def _recording_termination_guard(job: RecordJob):
  """Let middle-preset timeout/SIGTERM unwind HDF5; never finalize a partial."""
  if job.preset not in MIDDLE_FORCE_PRESETS:
    yield
    return

  def interrupted(signum, frame):
    del signum, frame
    raise KeyboardInterrupt(
      "middle-force recording terminated; partial is not a successful episode"
    )

  previous = signal.signal(signal.SIGTERM, interrupted)
  try:
    yield
  finally:
    signal.signal(signal.SIGTERM, previous)


def _record_claimed_episode(job: RecordJob) -> EpisodeSummary:
  """Record an episode after its output path has been claimed atomically."""
  conflicts = [path for path in _episode_artifacts(job.output) if path.exists()]
  if conflicts and not job.overwrite:
    joined = ", ".join(str(path) for path in conflicts)
    raise FileExistsError(
      f"episode {job.episode_index} output already exists: {joined}"
    )
  if job.overwrite:
    # --overwrite explicitly replaces this one resolved episode only. Clear old
    # completed artifacts before planning so a new failure can never coexist
    # with a stale success file under the same episode index.
    for path in _episode_artifacts(job.output):
      path.unlink(missing_ok=True)
  with _simulation_for_job(job) as (simulation, contact_model):
    return _record_simulation_episode(job, simulation, contact_model)


@contextmanager
def _simulation_for_job(job: RecordJob):
  """Keep the opt-in temporary model alive throughout planning and recording."""
  if job.preset in MIDDLE_FORCE_PRESETS:
    _validate_middle_job(job)
    if job.preset == PRECONTACT_PRESET:
      from kaihand_tactile_env.tasks.poker_draw.precontact_noise import (
        precontact_force_simulation as factory,
      )
    else:
      from kaihand_tactile_env.tasks.poker_draw.mid_full import (
        middle_force_simulation as factory,
      )
    with factory(job.config.model_path) as configured:
      yield configured
  elif job.preset == PRODUCTION_PRESET:
    if job.precontact_noise_std_rad is not None:
      raise ValueError("precontact noise requires middle-force-precontact-v1")
    yield ArmHandSimulation(job.config.model_path, scene=job.scene), None
  else:
    raise ValueError(f"unknown recording preset: {job.preset!r}")


def _validate_middle_job(job: RecordJob) -> None:
  check_policy(job.acceptance_policy)
  if job.scene != "poker-draw" or job.object_name != "card" or job.side != "right":
    raise ValueError(f"{job.preset} requires the right-hand poker-draw card task")
  if job.press_force_per_finger_n not in (None, MID_FORCE_PER_FINGER_N):
    raise ValueError(f"{job.preset} fixes pressure to 0.50 N per finger")
  if job.preset in INITIAL_RANDOMIZATION_PRESETS:
    if job.episode_seed < 0:
      raise ValueError(f"{job.preset} requires a non-negative episode seed")
    validate_randomization_bounds(job.object_xy_jitter, job.object_yaw_jitter)
  elif job.object_xy_jitter != 0.0 or job.object_yaw_jitter != 0.0:
    raise ValueError("middle-force-v1 currently requires zero object jitter")
  if job.preset == PRECONTACT_PRESET:
    if isinstance(job.episode_seed, (bool, np.bool_)) or not isinstance(
      job.episode_seed, (int, np.integer)
    ):
      raise ValueError(
        "middle-force-precontact-v1 requires a non-negative integer seed"
      )
    if job.precontact_noise_std_rad is None:
      raise ValueError("middle-force-precontact-v1 requires precontact noise settings")
    precontact_noise_settings(job.precontact_noise_std_rad)
  elif job.precontact_noise_std_rad is not None:
    raise ValueError("precontact noise requires middle-force-precontact-v1")
  if job.config.tactile_provider != SolverContactTactileProvider.source:
    raise ValueError(f"{job.preset} requires solver-contact tactile without probes")
  if job.config.physics_hz != 500:
    raise ValueError(f"{job.preset} requires the validated 500 Hz physics clock")


def _record_simulation_episode(
  job: RecordJob,
  simulation: ArmHandSimulation,
  contact_model: dict[str, object] | None,
) -> EpisodeSummary:
  """Run the common recorder while the owning preset context remains live."""
  _apply_camera_override(
    simulation,
    "overhead",
    job.overhead_pos,
    job.overhead_lookat,
    job.overhead_fovy,
  )
  _apply_camera_override(
    simulation,
    "front",
    job.front_pos,
    job.front_lookat,
    job.front_fovy,
  )
  initial_card_randomization = None
  if job.preset in INITIAL_RANDOMIZATION_PRESETS:
    initial_card_randomization = reset_randomized_card(
      simulation,
      seed=job.episode_seed,
      xy_jitter_m=job.object_xy_jitter,
      yaw_jitter_rad=job.object_yaw_jitter,
    )
  else:
    simulation.reset(
      seed=job.episode_seed,
      object_xy_jitter=job.object_xy_jitter,
      object_yaw_jitter=job.object_yaw_jitter,
    )
  # Plan before constructing camera renderers or opening HDF5. Predictable IK or
  # collision-planning failures therefore do not leave a one-frame partial file.
  if job.scene == "pick-place":
    plan = KnownStateGraspPlanner(simulation).plan_pick_and_place(job.side)
    recording_contract = RECORDING_CONTRACT_VERSION
  else:
    plan = PokerDrawPlanner(simulation).plan(job.side)
    recording_contract = POKER_RECORDING_CONTRACT_VERSION
  if job.preset == PRECONTACT_PRESET:
    simulation.configure_precontact_noise(
      seed=job.episode_seed, std_rad=job.precontact_noise_std_rad
    )
  tactile_provider = None
  if (
    job.config.tactile_provider == SolverContactTactileProvider.source
    and job.tactile_links == "right"
  ):
    tactile_provider = SolverContactTactileProvider(
      simulation.model, RIGHT_FINGERTIP_LINK_NAMES
    )
  execution_context: dict[str, object] = {}
  with (
    EpisodeRecorder(
      job.output,
      simulation,
      job.config,
      tactile_provider=tactile_provider,
      **({"capture_taskspace": True} if job.preset in MIDDLE_FORCE_PRESETS else {}),
      metadata={
        "recording_contract": recording_contract,
        "episode_index": job.episode_index,
        "seed": job.episode_seed,
        "scene": job.scene,
        "object": job.object_name,
        "side": job.side,
        "tactile_links": job.tactile_links,
        "object_xy_jitter": job.object_xy_jitter,
        "object_yaw_jitter": job.object_yaw_jitter,
        **(
          {"initial_card_randomization": initial_card_randomization}
          if job.preset in INITIAL_RANDOMIZATION_PRESETS
          else {}
        ),
        **(
          {"precontact_noise": simulation.precontact_noise_metadata()}
          if job.preset == PRECONTACT_PRESET
          else {}
        ),
        "press_force_per_finger_n": _press_force_for_job(job),
        "press_control": _press_control_for_job(job),
        "preset": job.preset,
        "preset_settings": _preset_settings_for_job(job),
        "acceptance_policy": job.acceptance_policy,
        "contact_model": contact_model,
        "base_model_sha256": _sha256_file(job.config.model_path),
        "base_model_fingerprint": model_fingerprint(job.config.model_path),
        "overhead_pos": job.overhead_pos,
        "overhead_lookat": job.overhead_lookat,
        "overhead_fovy": job.overhead_fovy,
        "front_pos": job.front_pos,
        "front_lookat": job.front_lookat,
        "front_fovy": job.front_fovy,
      },
    ) as recorder,
    _record_poker_failure(job, simulation, recorder, execution_context),
    _record_precontact_trace(job, simulation, recorder),
  ):
    recorder.record_initial()
    if job.scene == "pick-place":
      result = PickPlaceExecutor(simulation, observer=recorder.observe).execute(plan)
      stability = wait_until_object_stable(
        simulation,
        result.object_name,
        observer=recorder.observe,
      )
      recorder.record_terminal()
      final_pose = simulation.object_pose(result.object_name)
      placed_in_box: bool | None = cylinder_is_in_box(simulation, final_pose)
      result = replace(
        result,
        success=placed_in_box,
        final_object_pose=final_pose,
        placed_in_box=placed_in_box,
        phases=(*result.phases, "terminal_settle"),
      )
      outcome = asdict(result)
      outcome["final_object_pose"] = result.final_object_pose.tolist()
      outcome["box_center"] = result.box_center.tolist()
      if not result.success or not result.placed_in_box:
        raise RuntimeError(
          f"episode {job.episode_index} completed motion but cylinder was not "
          "placed in the box"
        )
    else:
      executor = _poker_executor_for_job(job, simulation, recorder.observe)
      execution_context["executor"] = executor
      result = executor.execute(plan)
      stability = wait_until_object_stable(
        simulation,
        result.object_name,
        observer=recorder.observe,
      )
      recorder.record_terminal()
      result = executor.refresh_terminal_result(result)
      result = replace(result, phases=(*result.phases, "terminal_settle"))
      placed_in_box = None
      outcome = asdict(result)
      for key in (
        "initial_card_pose",
        "edge_card_pose",
        "preinspection_card_pose",
        "final_card_pose",
        "inspection_target_card_position",
      ):
        outcome[key] = getattr(result, key).tolist()
      outcome["final_object_pose"] = result.final_card_pose.tolist()
      if job.preset in MIDDLE_FORCE_PRESETS:
        outcome.update(
          preset=job.preset,
          preset_settings=_preset_settings_for_job(job),
          contact_model=contact_model,
          edge_outcome=executor.edge_outcome,
          handoff_outcome=executor.handoff_outcome,
          experimental_control=executor.control_metadata(),
        )
        if not (
          executor.edge_outcome
          and executor.edge_outcome.get("target_reached")
          and executor.edge_outcome.get("held_at_edge")
          and accept_edge(executor.edge_outcome, job.acceptance_policy)
          and executor.handoff_outcome
          and executor.handoff_outcome.get("completed")
        ):
          raise RuntimeError("middle-force task lacks a qualified slide and handoff")
      if not result.success:
        raise RuntimeError(
          f"episode {job.episode_index} completed motion but the card was not "
          "retained after the thumb pinch or the four-finger slide did not qualify"
        )
    outcome["final_object_twist"] = simulation.object_twist(result.object_name).tolist()
    outcome["terminal_stability"] = asdict(stability)
    if job.preset == PRECONTACT_PRESET:
      outcome["precontact_noise"] = simulation.precontact_noise_metadata()
    recorder.set_outcome(outcome)
  try:
    report = validate_episode(job.output)
  except Exception:
    _demote_completed_episode(job.output)
    raise
  if not report.valid:
    _demote_completed_episode(job.output)
    raise RuntimeError(
      f"episode {job.episode_index} failed validation: {report.errors}"
    )
  if job.preset == PRECONTACT_PRESET:
    try:
      _validate_precontact_archive(job.output)
    except BaseException:
      _demote_completed_episode(job.output)
      raise
  _failure_path(job.output).unlink(missing_ok=True)
  return EpisodeSummary(
    episode_index=job.episode_index,
    output=job.output,
    success=result.success,
    placed_in_box=placed_in_box,
    state_samples=report.state_samples,
    camera_samples=report.camera_samples,
  )


def _poker_executor_for_job(job: RecordJob, simulation, observer):
  check_policy(job.acceptance_policy)
  if job.preset in MIDDLE_FORCE_PRESETS:
    return MidForcePokerExecutor(simulation, observer=observer, acceptance_policy=job.acceptance_policy)
  if job.acceptance_policy != STRICT_FORCE_POLICY:
    raise ValueError("training acceptance requires a middle-force preset")
  return PokerDrawExecutor(
    simulation,
    observer=observer,
    press_force_per_finger_n=_press_force_for_job(job),
  )


def _validate_precontact_archive(output: Path) -> None:
  """Require archived noise to agree with raw controls and tactile before success."""
  import h5py
  from kaihand_tactile_env.shared.tict_source_audit import audit_precontact_noise

  with h5py.File(output, "r") as file:
    report = audit_precontact_noise(file)
  if not report["valid"]:
    raise RuntimeError(
      f"precontact control archive failed validation: {report['errors']}"
    )


@contextmanager
def _record_poker_failure(job: RecordJob, simulation, recorder, execution_context):
  """Keep controller gates in the partial archive without changing success rules.

  This context exits after the precontact trace context but before the recorder
  closes. set_outcome REPLACES its payload, so explicitly retain the previously
  archived outcome (including precontact metadata). Diagnostic failures must not
  replace the original task/interruption.
  """
  try:
    yield
  except BaseException as error:
    if job.scene == "poker-draw":
      try:
        failure = {
          "error_type": type(error).__name__,
          "error_message": str(error),
          "episode_index": job.episode_index,
          "episode_seed": job.episode_seed,
          "interrupted": isinstance(error, (KeyboardInterrupt, SystemExit)),
          "diagnostic_scope": "runtime failure before completed episode validation",
        }
        simulation_time = getattr(getattr(simulation, "data", None), "time", None)
        if simulation_time is not None:
          failure["simulation_time_s"] = float(simulation_time)
        observation_time = getattr(simulation, "observation_time", None)
        if observation_time is not None:
          failure["observation_time_s"] = float(observation_time)
        outcome = dict(getattr(recorder, "_outcome", {}))
        outcome.update(success=False, task_failure=failure,
                       acceptance_policy=job.acceptance_policy,
                       task_completed=False, pressure_quality="incomplete")
        executor = execution_context.get("executor")
        if executor is not None:
          for name in ("edge_outcome", "handoff_outcome"):
            value = getattr(executor, name, None)
            if value is not None:
              outcome[name] = value
        recorder.set_outcome(outcome)
      except BaseException as archive_error:
        error.add_note(
          f"poker failure diagnostics could not be archived: {archive_error}"
        )
    raise


@contextmanager
def _record_precontact_trace(job: RecordJob, simulation, recorder):
  """Archive every executed physics step before the recorder closes, even on failure."""
  if job.preset != PRECONTACT_PRESET:
    yield
    return
  try:
    yield
  except BaseException as error:
    # Trace/metadata failures must never replace the task error or interruption.
    for operation in (
      lambda: recorder.write_precontact_noise_trace(
        simulation.precontact_noise_trace(),
        metadata=simulation.precontact_noise_metadata(),
      ),
      lambda: recorder.set_outcome(
        {"precontact_noise": simulation.precontact_noise_metadata()}
      ),
    ):
      try:
        operation()
      except BaseException as archive_error:
        error.add_note(f"precontact partial archive failed: {archive_error}")
    raise
  else:
    recorder.write_precontact_noise_trace(
      simulation.precontact_noise_trace(),
      metadata=simulation.precontact_noise_metadata(),
    )


def _failure_path(output: Path) -> Path:
  return output.with_suffix(".failure.json")


def _demote_completed_episode(output: Path) -> None:
  """Make a just-written invalid episode impossible to glob as completed."""
  partial = output.with_suffix(output.suffix + ".partial")
  if output.exists():
    os.replace(output, partial)
  output.with_suffix(".json").unlink(missing_ok=True)


def _episode_artifacts(output: Path) -> tuple[Path, Path, Path, Path]:
  return (
    output,
    output.with_suffix(".json"),
    output.with_suffix(output.suffix + ".partial"),
    _failure_path(output),
  )


def _check_output_conflicts(jobs: tuple[RecordJob, ...], *, overwrite: bool) -> None:
  active_locks = [
    job.output.with_suffix(job.output.suffix + ".lock")
    for job in jobs
    if job.output.with_suffix(job.output.suffix + ".lock").exists()
  ]
  if active_locks:
    raise FileExistsError(
      "episode worker lock already exists: "
      + ", ".join(str(path) for path in active_locks)
    )
  if overwrite:
    return
  conflicts = [
    path for job in jobs for path in _episode_artifacts(job.output) if path.exists()
  ]
  if conflicts:
    raise FileExistsError(
      "episode output already exists; choose --start-index or explicitly pass "
      "--overwrite: " + ", ".join(str(path) for path in conflicts)
    )


def _select_resume_jobs(jobs: tuple[RecordJob, ...]) -> tuple[RecordJob, ...]:
  """Keep only indices that do not already have one verified episode pair."""
  active_locks = [
    job.output.with_suffix(job.output.suffix + ".lock")
    for job in jobs
    if job.output.with_suffix(job.output.suffix + ".lock").exists()
  ]
  if active_locks:
    raise FileExistsError(
      "cannot resume while episode worker locks exist: "
      + ", ".join(str(path) for path in active_locks)
    )

  pending: list[RecordJob] = []
  completed = 0
  for job in jobs:
    output = job.output
    manifest = output.with_suffix(".json")
    partial = output.with_suffix(output.suffix + ".partial")
    failure = _failure_path(output)
    unambiguous_complete = (
      output.exists()
      and manifest.exists()
      and not partial.exists()
      and not failure.exists()
    )
    if unambiguous_complete:
      if _completed_episode_is_valid(output, manifest, job):
        completed += 1
        continue
      if _completed_episode_is_valid(output, manifest):
        raise FileExistsError(
          f"cannot resume over validated episode {output}: its task, model layout, "
          "model dependencies, or collection settings differ from this job. "
          "Use a new --output-dir. To replace it intentionally, rerun without "
          "--resume and pass --overwrite."
        )
    pending.append(replace(job, overwrite=True))
  print(
    f"resume: keeping {completed} validated completed episode(s); "
    f"scheduling {len(pending)} episode(s)"
  )
  return tuple(pending)


def _completed_episode_is_valid(
  output: Path, manifest: Path, job: RecordJob | None = None
) -> bool:
  """Verify both schema and sidecar digest before resume skips an episode."""
  try:
    report = validate_episode(output)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
  except (OSError, ValueError, json.JSONDecodeError):
    return False
  if not report.valid or not isinstance(payload, dict):
    return False
  if payload.get("episode") != output.name:
    return False
  expected_digest = payload.get("sha256")
  if not isinstance(expected_digest, str) or expected_digest != _sha256_file(output):
    return False
  return job is None or _episode_matches_job(output, job)


def _press_force_for_job(job: RecordJob) -> float | None:
  if job.scene != "poker-draw":
    return None
  if job.preset in MIDDLE_FORCE_PRESETS:
    return MID_FORCE_PER_FINGER_N
  if job.press_force_per_finger_n is not None:
    return float(job.press_force_per_finger_n)
  return float(task_config(job.scene).DEFAULT_PRESS_FORCE_PER_FINGER_N)


def _press_control_for_job(job: RecordJob) -> dict[str, object] | None:
  if job.scene != "poker-draw":
    return None
  from kaihand_tactile_env.tasks.poker_draw.friction import press_control_metadata

  if job.preset in MIDDLE_FORCE_PRESETS:
    return {
      "preset": job.preset,
      "parameters": _preset_settings_for_job(job),
      "controller_source_sha256": _middle_source_hashes(job.preset),
    }
  return press_control_metadata()


def _preset_settings_for_job(job: RecordJob) -> dict[str, object] | None:
  if job.preset not in MIDDLE_FORCE_PRESETS:
    return None
  return {
    "preset": job.preset,
    "acceptance_policy": job.acceptance_policy,
    "press_force_per_finger_n": MID_FORCE_PER_FINGER_N,
    "pressure_window": asdict(MID_FORCE_SETTINGS),
    "object_xy_jitter_m": job.object_xy_jitter,
    "object_yaw_jitter_rad": job.object_yaw_jitter,
    **(
      {
        "initial_pose_distribution": "independent uniform world XY and yaw",
        "robot_initial_state_noise": None,
        "randomization_scope": (
          "initial card pose and precontact right-arm control targets"
          if job.preset == PRECONTACT_PRESET
          else "initial card pose only; no online disturbance"
        ),
        "randomization_bounds_are_success_guarantee": False,
      }
      if job.preset in INITIAL_RANDOMIZATION_PRESETS
      else {}
    ),
    "observation_noise": None,
    "action_noise": (
      precontact_noise_settings(job.precontact_noise_std_rad)
      if job.preset == PRECONTACT_PRESET
      else None
    ),
    "force_limit_scope": "slide_card and edge_hold; explicit handoff before pickup",
    "contact_model_scope": "unchanged throughout the whole episode",
    "reconstruction": (
      (
        "requires precontact_force_simulation factory, configure_precontact_noise "
        "after reset/planning, and MidForcePokerExecutor; "
        if job.preset == PRECONTACT_PRESET
        else "requires middle_force_simulation factory and MidForcePokerExecutor; "
      )
      + "legacy replay does not reconstruct the temporary wrapper/contact overrides "
      "or the bounded-servo handoff automatically"
    ),
    "controller_source_sha256": _middle_source_hashes(job.preset),
  }


def _middle_source_hashes(preset: str = MIDDLE_FORCE_PRESET) -> dict[str, str]:
  """Fingerprint controller and recording code separately from scene assets."""
  import kaihand_tactile_env.shared.recording as recording_module
  import kaihand_tactile_env.tasks.poker_draw.mid_full as controller_module

  task_directory = Path(controller_module.__file__).parent
  shared_directory = Path(recording_module.__file__).parent
  paths = {
    "record_dataset.py": Path(__file__),
    **{
      f"poker_draw/{name}": task_directory / name
      for name in (
        "mid_full.py",
        "pressure_window.py",
        "friction.py",
        "task.py",
        "config.py",
        "press_control.py",
        "acceptance.py",
      )
    },
    **{
      f"shared/{name}": shared_directory / name
      for name in (
        "recording.py",
        "taskspace_recording.py",
        "simulation.py",
        "config.py",
        "cameras.py",
        "tactile.py",
        "contact_tactile.py",
      )
    },
  }
  if preset in INITIAL_RANDOMIZATION_PRESETS:
    paths["poker_draw/randomization.py"] = task_directory / "randomization.py"
  if preset == PRECONTACT_PRESET:
    paths["poker_draw/precontact_noise.py"] = task_directory / "precontact_noise.py"
  return {name: _sha256_file(path) for name, path in paths.items()}


def _model_identity_for_job(job: RecordJob) -> tuple[str, str]:
  if job.preset in MIDDLE_FORCE_PRESETS:
    from kaihand_tactile_env.tasks.poker_draw.friction import (
      model_with_table_card_friction,
    )

    # This creates only a tiny XML wrapper; no robot, MuJoCo data or renderers.
    with model_with_table_card_friction(
      job.config.model_path, MID_FORCE_SETTINGS.table_friction
    ) as model_path:
      return _sha256_file(model_path), model_fingerprint(model_path)
  return _sha256_file(job.config.model_path), model_fingerprint(job.config.model_path)


def _episode_matches_job(output: Path, job: RecordJob) -> bool:
  """Prevent --resume from silently mixing incompatible collection settings."""
  try:
    import h5py

    with h5py.File(output, "r") as file:
      metadata = json.loads(str(file.attrs.get("metadata_json", "{}")))
      if not isinstance(metadata, dict):
        return False
      expected_metadata = {
        "recording_contract": (
          POKER_RECORDING_CONTRACT_VERSION
          if job.scene == "poker-draw"
          else RECORDING_CONTRACT_VERSION
        ),
        "episode_index": job.episode_index,
        "seed": job.episode_seed,
        "object": job.object_name,
        "side": job.side,
        "object_xy_jitter": job.object_xy_jitter,
        "object_yaw_jitter": job.object_yaw_jitter,
        "model_layout": TASK_ISOLATED_MODEL_LAYOUT,
        "active_objects": [job.object_name],
      }
      if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return False
      if metadata.get("scene", "pick-place") != job.scene:
        return False
      if metadata.get("preset", PRODUCTION_PRESET) != job.preset:
        return False
      if metadata.get("acceptance_policy", STRICT_FORCE_POLICY) != job.acceptance_policy:
        return False
      if metadata.get("preset_settings") != _preset_settings_for_job(job):
        return False
      if job.preset in INITIAL_RANDOMIZATION_PRESETS:
        randomization = metadata.get("initial_card_randomization")
        if not validate_recorded_randomization(
          randomization,
          job.episode_seed,
          job.object_xy_jitter,
          job.object_yaw_jitter,
        ):
          return False
        # Verify the realized reset against raw evidence, not metadata alone.
        if "objects/card/pose_wxyz" not in file or "state/timestamp" not in file:
          return False
        poses, timestamps = file["objects/card/pose_wxyz"], file["state/timestamp"]
        if not len(poses) or not len(timestamps) or timestamps[0] != 0.0:
          return False
        actual = np.asarray(poses[0], dtype=float)
        expected = np.asarray(randomization["sampled_pose_wxyz"], dtype=float)
        if (
          actual.shape != (7,)
          or not np.all(np.isfinite(actual))
          or not np.allclose(actual[:3], expected[:3], atol=1e-12, rtol=0)
          or min(
            np.linalg.norm(actual[3:] - expected[3:]),
            np.linalg.norm(actual[3:] + expected[3:]),
          )
          > 1e-10
        ):
          return False
      if job.preset == PRECONTACT_PRESET:
        from kaihand_tactile_env.shared.tict_source_audit import audit_precontact_noise

        initial_noise = metadata.get("precontact_noise")
        if not isinstance(initial_noise, dict):
          return False
        expected_initial_noise = {
          "schema_version": "poker-precontact-arm-noise-v1",
          "settings": precontact_noise_settings(job.precontact_noise_std_rad),
          "seed": job.episode_seed,
          "configured": True,
          "random_sample_count": 0,
          "physics_step_count": 0,
          "postcontact_random_sample_count": 0,
          "command_handoff_count": 0,
        }
        if any(
          initial_noise.get(key) != value
          for key, value in expected_initial_noise.items()
        ):
          return False
        # The reset itself may already expose tactile contact. The audit checks
        # the initial latch against raw t=0 tactile instead of assuming no contact.
        if not audit_precontact_noise(file, metadata=metadata)["valid"]:
          return False
      if metadata.get("press_force_per_finger_n") != _press_force_for_job(job):
        return False
      if job.scene == "poker-draw" and "tactile_contact_force" not in file:
        return False
      if metadata.get("press_control") != _press_control_for_job(job):
        return False
      if metadata.get("tactile_links", "all") != job.tactile_links:
        return False
      camera_overrides = {
        "overhead_pos": job.overhead_pos,
        "overhead_lookat": job.overhead_lookat,
        "overhead_fovy": job.overhead_fovy,
        "front_pos": job.front_pos,
        "front_lookat": job.front_lookat,
        "front_fovy": job.front_fovy,
      }
      for key, value in camera_overrides.items():
        if key not in metadata:
          # Older v1 files had no override metadata. They are compatible only
          # when no position/fovy override was requested; lookat alone is inert.
          if key.endswith(("_pos", "_fovy")) and value is not None:
            return False
          continue
        expected_value = list(value) if isinstance(value, tuple) else value
        if metadata[key] != expected_value:
          return False
      if int(file.attrs.get("physics_hz", -1)) != job.config.physics_hz:
        return False
      if int(file.attrs.get("control_hz", -1)) != job.config.control_hz:
        return False
      if int(file.attrs.get("camera_hz", -1)) != job.config.camera_hz:
        return False
      if str(file.attrs.get("tactile_source", "")) != job.config.tactile_provider:
        return False
      expected_model_hash, expected_model_fingerprint = _model_identity_for_job(job)
      if str(file.attrs.get("model_sha256", "")) != expected_model_hash:
        return False
      if str(file.attrs.get("model_fingerprint", "")) != expected_model_fingerprint:
        return False
      if "cameras" not in file:
        return not job.config.cameras
      recorded_names = set(file["cameras"].keys())
      if recorded_names != {camera.name for camera in job.config.cameras}:
        return False
      for camera in job.config.cameras:
        group = file[f"cameras/{camera.name}"]
        if int(group.attrs.get("width", -1)) != camera.width:
          return False
        if int(group.attrs.get("height", -1)) != camera.height:
          return False
        for modality, enabled in (
          ("rgb", camera.rgb),
          ("depth", camera.depth),
          ("segmentation", camera.segmentation),
        ):
          if (modality in group) != enabled:
            return False
  except (OSError, TypeError, ValueError, json.JSONDecodeError):
    return False
  return True


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _run_jobs(
  jobs: tuple[RecordJob, ...], workers: int, *, continue_on_error: bool = True
) -> Iterator[EpisodeResult]:
  """Yield completed episodes, using spawn so GL/HDF5 state is never inherited."""
  if workers <= 0:
    raise ValueError("workers must be positive")
  if not jobs:
    return
  if workers == 1:
    for job in jobs:
      try:
        yield _record_episode(job)
      except Exception as error:
        if not continue_on_error:
          raise
        yield _record_failure(job, error)
    return

  context = multiprocessing.get_context("spawn")
  worker_count = min(workers, len(jobs))
  with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
    future_jobs = {executor.submit(_record_episode, job): job for job in jobs}
    pending_results: dict[int, EpisodeResult] = {}
    next_result = 0
    try:
      for future in as_completed(future_jobs):
        job = future_jobs[future]
        try:
          result: EpisodeResult = future.result()
        except Exception as error:
          if not continue_on_error:
            raise
          result = _record_failure(job, error)
        if result.episode_index != job.episode_index:
          raise RuntimeError(
            f"worker returned episode {result.episode_index} for "
            f"job {job.episode_index}"
          )
        pending_results[result.episode_index] = result
        while (
          next_result < len(jobs) and jobs[next_result].episode_index in pending_results
        ):
          yield pending_results.pop(jobs[next_result].episode_index)
          next_result += 1
    except BaseException:
      for future in future_jobs:
        future.cancel()
      raise


def _record_failure(job: RecordJob, error: Exception) -> EpisodeFailure:
  """Persist an explicit failure marker while keeping partial HDF5 unusable."""
  partial = job.output.with_suffix(job.output.suffix + ".partial")
  result = EpisodeFailure(
    episode_index=job.episode_index,
    output=job.output,
    error_type=type(error).__name__,
    error_message=str(error),
    partial_path=partial if partial.exists() else None,
  )
  report_path = _failure_path(job.output)
  temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
  payload = {
    "completed": False,
    "episode_index": job.episode_index,
    "episode_seed": job.episode_seed,
    "scene": job.scene,
    "object": job.object_name,
    "side": job.side,
    "intended_output": job.output.name,
    "recording_contract": (
      POKER_RECORDING_CONTRACT_VERSION
      if job.scene == "poker-draw"
      else RECORDING_CONTRACT_VERSION
    ),
    "press_force_per_finger_n": _press_force_for_job(job),
    "press_control": _press_control_for_job(job),
    "preset": job.preset,
    "preset_settings": _preset_settings_for_job(job),
    "object_xy_jitter": job.object_xy_jitter,
    "object_yaw_jitter": job.object_yaw_jitter,
    "tactile_source": job.config.tactile_provider,
    "partial": result.partial_path.name if result.partial_path else None,
    "error_type": result.error_type,
    "error_message": result.error_message,
  }
  temporary_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary_path, report_path)
  return result


def main() -> None:
  args = _parse_args()
  jobs = _build_jobs(args)
  failures: list[EpisodeFailure] = []
  for result in _run_jobs(jobs, args.workers, continue_on_error=not args.fail_fast):
    if isinstance(result, EpisodeFailure):
      failures.append(result)
      print(f"{result.output}: FAILED {result.error_type}: {result.error_message}")
    else:
      task_outcome = (
        f"placed_in_box={result.placed_in_box}"
        if result.placed_in_box is not None
        else "card_retained=True"
      )
      print(
        f"{result.output}: success={result.success}, {task_outcome}, "
        f"states={result.state_samples}, cameras={result.camera_samples}"
      )
  if failures:
    print(
      f"completed batch with {len(failures)} failed episode(s); "
      "see the corresponding .failure.json files"
    )
    raise SystemExit(1)


def _triplet(values: Sequence[float]) -> tuple[float, float, float]:
  return (float(values[0]), float(values[1]), float(values[2]))


def _triplet_or_none(
  values: Sequence[float] | None,
) -> tuple[float, float, float] | None:
  return None if values is None else _triplet(values)


def _object_for_episode(selection: str, index: int) -> str:
  del index
  return selection


def _apply_camera_override(
  simulation: ArmHandSimulation,
  name: str,
  position: tuple[float, float, float] | None,
  lookat: tuple[float, float, float],
  fovy: float | None,
) -> None:
  if position is None and fovy is None:
    return
  if position is None:
    camera_id = simulation.model.camera(name).id
    simulation.model.cam_fovy[camera_id] = fovy
    return
  set_fixed_camera_lookat(
    simulation.model,
    name,
    position=np.asarray(position),
    lookat=np.asarray(lookat),
    fovy_degrees=fovy,
  )
  mujoco.mj_forward(simulation.model, simulation.data)


if __name__ == "__main__":
  main()
