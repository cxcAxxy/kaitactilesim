#!/usr/bin/env python3
"""Run a remote EgoSteer policy in the local KaiHand MuJoCo workcell."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.egosteer_adapter import (
  camera_from_world_opencv,
  decode_action_targets,
  model_state_history,
  raw_unified_from_simulation,
  relative_to_absolute,
)
from kaihand_tactile_env.shared.egosteer_client import (
  EgoSteerPolicyClient,
  ObservationHistory,
)
from kaihand_tactile_env.shared.inference_dashboard import (
  InferenceSensorDashboard,
)
from kaihand_tactile_env.shared.policy_cameras import policy_camera_names
from kaihand_tactile_env.shared.recording import (
  TERMINAL_ANGULAR_SPEED_THRESHOLD,
  TERMINAL_LINEAR_SPEED_THRESHOLD,
  TERMINAL_STABLE_DURATION,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider
from kaihand_tactile_env.tasks.pick_place.task import cylinder_is_in_box

DEFAULT_INSTRUCTION = "Put the red cylinder into the blue box."
CONTROL_HZ = 30
FINGERTIP_LINKS = (
  ("thumb", "link6"),
  ("index", "link4"),
  ("middle", "link4"),
  ("ring", "link4"),
  ("pinky", "link4"),
)
DEFAULT_FREE_CLOSE_FORCE_LIMIT = 0.82
DEFAULT_FREE_CLOSE_CUTOFF_M = 0.002
DEFAULT_FREE_CLOSE_MIN_CLOSURE = 0.35


class ViewerClosed(RuntimeError):
  """The user closed the native viewer during a policy request."""


@dataclass
class RolloutStats:
  requests: int = 0
  action_steps: int = 0
  arm_ik_failures: int = 0
  hand_ik_failures: int = 0
  maximum_arm_position_error: float = 0.0
  maximum_arm_orientation_error: float = 0.0
  maximum_hand_error: float = 0.0
  maximum_object_height: float = 0.0
  stable_success: bool = False


@dataclass(frozen=True)
class ModelObservation:
  images: Any
  raw_states: np.ndarray
  camera_intrinsics: Any
  selected_images: dict[str, np.ndarray]


class StablePlacementDetector:
  """Apply the dataset collection terminal-stability rule during rollout."""

  def __init__(
    self,
    *,
    timestep: float,
    linear_speed_threshold: float = TERMINAL_LINEAR_SPEED_THRESHOLD,
    angular_speed_threshold: float = TERMINAL_ANGULAR_SPEED_THRESHOLD,
    stable_duration: float = TERMINAL_STABLE_DURATION,
  ) -> None:
    if timestep <= 0.0:
      raise ValueError("timestep must be positive")
    self.linear_speed_threshold = linear_speed_threshold
    self.angular_speed_threshold = angular_speed_threshold
    self.required_steps = max(1, int(np.ceil(stable_duration / timestep)))
    self.stable_steps = 0
    self.succeeded = False
    self.linear_speed = float("inf")
    self.angular_speed = float("inf")

  def update(
    self,
    *,
    linear_speed: float,
    angular_speed: float,
    placed_in_box: bool,
    grasp_active: bool,
  ) -> bool:
    """Return true once a released cylinder is stably inside the box."""
    self.linear_speed = linear_speed
    self.angular_speed = angular_speed
    eligible = (
      placed_in_box
      and not grasp_active
      and linear_speed < self.linear_speed_threshold
      and angular_speed < self.angular_speed_threshold
    )
    self.stable_steps = self.stable_steps + 1 if eligible else 0
    self.succeeded |= self.stable_steps >= self.required_steps
    return self.succeeded


class AutomaticGraspStabilizer:
  """Mirror the environment-side grasp constraint used to collect the data."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    *,
    enabled: bool,
    contact_steps: int,
    release_closure: float,
  ) -> None:
    self.simulation = simulation
    self.enabled = enabled
    self.required_contact_steps = contact_steps
    self.release_closure = release_closure
    self.contact_streak = 0
    self.active = False
    self._cylinder_geom = simulation.model.geom("cylinder_geom").id
    self._pad_to_index = {
      simulation.model.geom(
        f"hand_r_{finger}_{'link6' if finger == 'thumb' else 'link4'}_tactile_pad_col"
      ).id: index
      for index, (finger, _link) in enumerate(FINGERTIP_LINKS)
    }
    self._joint_names = tuple(simulation._hand_targets["right"])
    self._open_targets = self._command_targets()
    simulation.set_hand_closure("right", 1.0)
    self._closed_targets = self._command_targets()
    simulation.set_hand_joint_targets(self._joint_names, self._open_targets)
    direction = self._closed_targets - self._open_targets
    self._closure_denominator = max(float(direction @ direction), 1.0e-12)

  def _command_targets(self) -> np.ndarray:
    return np.array(
      [self.simulation._hand_targets["right"][name] for name in self._joint_names]
    )

  def closure(self) -> float:
    direction = self._closed_targets - self._open_targets
    value = self._command_targets() - self._open_targets
    return float(np.clip((value @ direction) / self._closure_denominator, 0.0, 1.0))

  def after_command(self) -> None:
    if self.enabled and self.active and self.closure() < self.release_closure:
      self.simulation.set_grasp_stabilizer(False)
      self.active = False
      self.contact_streak = 0
      print(
        f"[sim] grasp stabilizer released at t={self.simulation.data.time:.3f}s",
        flush=True,
      )

  def after_physics_step(self) -> None:
    if not self.enabled or self.active:
      return
    touching = np.zeros(5, dtype=bool)
    for contact_index in range(self.simulation.data.ncon):
      contact = self.simulation.data.contact[contact_index]
      if self._cylinder_geom not in (int(contact.geom1), int(contact.geom2)):
        continue
      other = int(
        contact.geom2 if int(contact.geom1) == self._cylinder_geom else contact.geom1
      )
      finger_index = self._pad_to_index.get(other)
      if finger_index is not None:
        touching[finger_index] = True
    self.contact_streak = self.contact_streak + 1 if np.all(touching) else 0
    if self.contact_streak >= self.required_contact_steps:
      self.simulation.set_grasp_stabilizer(True)
      self.active = True
      print(
        f"[sim] grasp stabilizer activated at t={self.simulation.data.time:.3f}s",
        flush=True,
      )

  def close(self) -> None:
    if self.active:
      self.simulation.set_grasp_stabilizer(False)
      self.active = False


class AdaptiveFreeCloseForce:
  """Apply the free-space finger force used by demonstration collection.

  The demonstration executor temporarily raises each finger actuator's force
  limit while that fingertip is still more than ``distance_cutoff`` from the
  cylinder.  Without the same controller behavior, a policy can predict the
  correct fingertip trajectory but fail to overcome hand joint friction in the
  final millimetres before contact.
  """

  def __init__(
    self,
    simulation: ArmHandSimulation,
    *,
    enabled: bool,
    force_limit: float,
    distance_cutoff: float,
    minimum_closure: float,
  ) -> None:
    self.simulation = simulation
    self.enabled = enabled
    self.force_limit = force_limit
    self.distance_cutoff = distance_cutoff
    self.minimum_closure = minimum_closure
    self.active = False
    self._cylinder_geom = simulation.model.geom("cylinder_geom").id
    self._pad_ids = tuple(
      simulation.model.geom(f"hand_r_{finger}_{link}_tactile_pad_col").id
      for finger, link in FINGERTIP_LINKS
    )
    self._actuator_ids_by_finger = tuple(
      tuple(
        actuator_id
        for name, actuator_id in simulation._hand_actuators["right"].items()
        if f"_{finger}_" in name
      )
      for finger, _link in FINGERTIP_LINKS
    )
    self._baseline_ranges = {
      actuator_id: simulation.model.actuator_forcerange[actuator_id].copy()
      for actuator_ids in self._actuator_ids_by_finger
      for actuator_id in actuator_ids
    }
    self._distance_workspace = np.zeros(6, dtype=float)

  def before_physics_step(self, *, closure: float, grasp_active: bool) -> None:
    should_assist = (
      self.enabled and not grasp_active and closure >= self.minimum_closure
    )
    if not should_assist:
      self._restore_baseline()
      return

    boosted = False
    for pad_id, actuator_ids in zip(
      self._pad_ids, self._actuator_ids_by_finger, strict=True
    ):
      gap = mujoco.mj_geomDistance(
        self.simulation.model,
        self.simulation.data,
        int(pad_id),
        self._cylinder_geom,
        0.1,
        self._distance_workspace,
      )
      for actuator_id in actuator_ids:
        baseline = self._baseline_ranges[actuator_id]
        baseline_force = float(np.max(np.abs(baseline)))
        force = (
          max(baseline_force, self.force_limit)
          if gap > self.distance_cutoff
          else baseline_force
        )
        self.simulation.model.actuator_forcerange[actuator_id] = (-force, force)
        boosted |= force > baseline_force
    self.active = boosted

  def _restore_baseline(self) -> None:
    if not self.active:
      return
    for actuator_id, force_range in self._baseline_ranges.items():
      self.simulation.model.actuator_forcerange[actuator_id] = force_range
    self.active = False

  def close(self) -> None:
    self._restore_baseline()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--server", default="ws://127.0.0.1:8765")
  parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--object-xy-jitter", type=float, default=0.01)
  parser.add_argument("--object-yaw-jitter", type=float, default=0.05)
  parser.add_argument("--viewer", choices=("native", "none"), default="native")
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--record",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save the common 1080p bilateral tactile evaluation video",
  )
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  parser.add_argument("--review-render-width", type=int, default=640)
  parser.add_argument("--review-render-height", type=int, default=480)
  parser.add_argument(
    "--review-second-camera",
    choices=("global", "right_wrist", "overhead"),
    default="global",
  )
  parser.add_argument(
    "--sensor-dashboard",
    action="store_true",
    help="Open overhead RGB and bilateral 7x5 fingertip tactile views",
  )
  parser.add_argument("--tactile-max-depth-mm", type=float, default=3.0)
  parser.add_argument("--tactile-refresh-hz", type=float, default=10.0)
  parser.add_argument("--overhead-refresh-hz", type=float, default=5.0)
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--max-sim-seconds", type=float, default=8.0)
  parser.add_argument(
    "--max-requests",
    type=int,
    default=0,
    help="0 uses only max-sim-seconds; 1 is useful for a connection test",
  )
  parser.add_argument(
    "--execute-steps",
    type=int,
    default=5,
    help="Number of 30 Hz action targets executed before replanning",
  )
  parser.add_argument("--control-side", choices=("right", "both"), default="right")
  parser.add_argument("--image-format", choices=("jpeg", "raw"), default="jpeg")
  parser.add_argument("--jpeg-quality", type=int, default=95)
  parser.add_argument("--open-timeout", type=float, default=30.0)
  parser.add_argument("--response-timeout", type=float, default=600.0)
  parser.add_argument("--max-wrist-jump", type=float, default=0.35)
  parser.add_argument("--max-arm-position-error", type=float, default=0.03)
  parser.add_argument("--max-arm-orientation-error", type=float, default=0.35)
  parser.add_argument("--hand-ik-tolerance", type=float, default=0.0015)
  parser.add_argument("--max-hand-error", type=float, default=0.02)
  parser.add_argument(
    "--strict-ik",
    action="store_true",
    help="Abort on any IK miss instead of accepting bounded best-effort results",
  )
  parser.add_argument(
    "--auto-grasp-stabilizer",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Mirror the grasp/release constraint used by the demonstration executor",
  )
  parser.add_argument("--stabilizer-contact-steps", type=int, default=3)
  parser.add_argument("--stabilizer-release-closure", type=float, default=0.35)
  parser.add_argument(
    "--adaptive-grasp-force",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Mirror the free-space finger force boost used during data collection",
  )
  parser.add_argument(
    "--free-close-force-limit",
    type=float,
    default=DEFAULT_FREE_CLOSE_FORCE_LIMIT,
  )
  parser.add_argument(
    "--free-close-cutoff-m",
    type=float,
    default=DEFAULT_FREE_CLOSE_CUTOFF_M,
  )
  parser.add_argument(
    "--free-close-min-closure",
    type=float,
    default=DEFAULT_FREE_CLOSE_MIN_CLOSURE,
  )
  parser.add_argument(
    "--stop-on-success", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
      "Pace action execution at wall-clock 30 Hz; defaults on for native viewer "
      "or sensor dashboard"
    ),
  )
  parser.add_argument(
    "--hold-final-seconds",
    type=float,
    default=3.0,
    help="Keep the final viewer frame visible this long after success",
  )
  args = parser.parse_args(argv)
  if args.object_xy_jitter < 0.0 or args.object_yaw_jitter < 0.0:
    parser.error("object jitter must be non-negative")
  if args.width <= 0 or args.height <= 0:
    parser.error("image dimensions must be positive")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  if args.tactile_max_depth_mm <= 0.0:
    parser.error("--tactile-max-depth-mm must be positive")
  if args.tactile_refresh_hz <= 0.0 or args.overhead_refresh_hz <= 0.0:
    parser.error("dashboard refresh rates must be positive")
  if args.max_sim_seconds <= 0.0:
    parser.error("--max-sim-seconds must be positive")
  if args.max_requests < 0:
    parser.error("--max-requests cannot be negative")
  if args.execute_steps <= 0:
    parser.error("--execute-steps must be positive")
  if not 1 <= args.jpeg_quality <= 100:
    parser.error("--jpeg-quality must be in [1, 100]")
  if args.max_wrist_jump <= 0.0:
    parser.error("--max-wrist-jump must be positive")
  if args.max_arm_position_error <= 0.0 or args.max_arm_orientation_error <= 0.0:
    parser.error("arm error limits must be positive")
  if args.hand_ik_tolerance <= 0.0 or args.max_hand_error <= 0.0:
    parser.error("hand error limits must be positive")
  if args.hand_ik_tolerance > args.max_hand_error:
    parser.error("--hand-ik-tolerance cannot exceed --max-hand-error")
  if args.stabilizer_contact_steps <= 0:
    parser.error("--stabilizer-contact-steps must be positive")
  if not 0.0 <= args.stabilizer_release_closure <= 1.0:
    parser.error("--stabilizer-release-closure must be in [0, 1]")
  if args.free_close_force_limit <= 0.0:
    parser.error("--free-close-force-limit must be positive")
  if args.free_close_cutoff_m < 0.0:
    parser.error("--free-close-cutoff-m cannot be negative")
  if not 0.0 <= args.free_close_min_closure <= 1.0:
    parser.error("--free-close-min-closure must be in [0, 1]")
  if args.hold_final_seconds < 0.0:
    parser.error("--hold-final-seconds cannot be negative")
  if args.real_time is None:
    args.real_time = args.viewer == "native" or args.sensor_dashboard
  return args


@contextlib.contextmanager
def _viewer_context(simulation: ArmHandSimulation, mode: str) -> Iterator[Any | None]:
  if mode == "none":
    yield None
    return
  import mujoco.viewer

  with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
    viewer.opt.geomgroup[5] = 0
    yield viewer


@contextlib.contextmanager
def _sensor_dashboard_context(
  simulation: ArmHandSimulation,
  tactile: GenesisProbeTactileProvider | None,
  renderer: WorkcellRenderer,
  overhead_camera: CameraConfig,
  args: argparse.Namespace,
) -> Iterator[InferenceSensorDashboard | None]:
  if not args.sensor_dashboard:
    yield None
    return
  if tactile is None:
    raise RuntimeError("sensor dashboard requires the Genesis tactile provider")
  dashboard = InferenceSensorDashboard(
    simulation,
    tactile,
    renderer,
    overhead_camera,
    max_depth_mm=args.tactile_max_depth_mm,
    tactile_refresh_hz=args.tactile_refresh_hz,
    camera_refresh_hz=args.overhead_refresh_hz,
  )
  try:
    yield dashboard
  finally:
    dashboard.close()


def _append_observation(
  simulation: ArmHandSimulation,
  renderer: WorkcellRenderer,
  camera: CameraConfig,
  history: ObservationHistory,
) -> Any:
  calibration = renderer.calibration(simulation.data, camera)
  image = renderer.capture(simulation.data, camera)["rgb"]
  history.append(image, raw_unified_from_simulation(simulation))
  return calibration


def _model_camera_names(metadata: dict[str, Any]) -> tuple[str, ...]:
  """Resolve the model-owned camera contract with a legacy head-only fallback."""
  contract = dict(metadata.get("observation_contract", {}))
  if metadata.get("cameras") is not None:
    contract["cameras"] = metadata["cameras"]
  return policy_camera_names(contract)


def _append_model_observations(
  simulation: ArmHandSimulation,
  renderer: WorkcellRenderer,
  cameras: dict[str, CameraConfig],
  histories: dict[str, ObservationHistory],
) -> dict[str, Any]:
  """Capture one synchronized frame for every model-declared camera."""
  if tuple(cameras) != tuple(histories):
    raise ValueError("camera and history keys must match in canonical order")
  raw_state = raw_unified_from_simulation(simulation)
  calibrations = {}
  for name, camera in cameras.items():
    calibrations[name] = renderer.calibration(simulation.data, camera)
    image = renderer.capture(simulation.data, camera)["rgb"]
    histories[name].append(image, raw_state)
  return calibrations


def _select_model_observation(
  camera_names: tuple[str, ...],
  histories: dict[str, ObservationHistory],
  calibrations: dict[str, Any],
) -> ModelObservation:
  """Preserve the legacy scalar payload for head-only checkpoints."""
  selected = {name: histories[name].select() for name in camera_names}
  head_states = selected["head"].raw_states
  for name in camera_names[1:]:
    if not np.array_equal(selected[name].raw_states, head_states):
      raise RuntimeError(f"{name} history is not synchronized with head history")
  images_by_camera = {name: selected[name].images for name in camera_names}
  intrinsics_by_camera = {
    name: calibrations[name].intrinsic for name in camera_names
  }
  if camera_names == ("head",):
    images: Any = images_by_camera["head"]
    intrinsics: Any = intrinsics_by_camera["head"]
  else:
    images = images_by_camera
    intrinsics = intrinsics_by_camera
  return ModelObservation(
    images=images,
    raw_states=head_states,
    camera_intrinsics=intrinsics,
    selected_images=images_by_camera,
  )


def _model_action_horizon(metadata: dict[str, Any]) -> int:
  """Read a positive model-owned prediction horizon from server metadata."""
  horizon = metadata.get("action_horizon")
  if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
    raise RuntimeError(
      f"Server must declare a positive integer action_horizon: {metadata}"
    )
  return horizon


def _validated_prediction(
  actions: Any,
  *,
  horizon: int,
  action_dim: int | None = None,
) -> np.ndarray:
  """Reject response tensors that disagree with the advertised model contract."""
  array = np.asarray(actions)
  expected = (horizon, action_dim) if action_dim is not None else None
  valid = (
    array.ndim == 2
    and array.shape[0] == horizon
    and (action_dim is None or array.shape[1] == action_dim)
    and np.all(np.isfinite(array))
  )
  if not valid:
    expectation = f"({horizon}, {action_dim})" if expected else f"({horizon}, action_dim)"
    raise RuntimeError(
      f"Model prediction must be finite with shape {expectation}, got {array.shape}"
    )
  return array


async def _infer_while_servicing_viewer(
  client: EgoSteerPolicyClient,
  viewer: Any | None,
  sensor_dashboard: InferenceSensorDashboard | None,
  **request: Any,
) -> Any:
  task = asyncio.create_task(client.infer(**request))
  try:
    while not task.done():
      if viewer is not None:
        if not viewer.is_running():
          raise ViewerClosed("native viewer was closed")
        viewer.sync()
      if sensor_dashboard is not None:
        if not sensor_dashboard.is_running():
          raise ViewerClosed("sensor dashboard was closed")
        sensor_dashboard.sync()
      await asyncio.sleep(0.05)
    return await task
  finally:
    if not task.done():
      task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await task


def _command_action_step(
  simulation: ArmHandSimulation,
  decoded: Any,
  action_index: int,
  args: argparse.Namespace,
  stats: RolloutStats,
) -> None:
  sides = ("left", "right") if args.control_side == "both" else ("right",)
  for side in sides:
    targets = decoded.for_side(side)
    current_position, _ = simulation.current_pose_matrix(side)
    wrist_jump = float(
      np.linalg.norm(targets.site_positions_world[action_index] - current_position)
    )
    if wrist_jump > args.max_wrist_jump:
      raise RuntimeError(
        f"{side} wrist target jump {wrist_jump:.3f}m exceeds "
        f"--max-wrist-jump={args.max_wrist_jump:.3f}m"
      )

    arm_result = simulation.set_pose_target(
      side,
      targets.site_positions_world[action_index],
      targets.site_quaternions_wxyz[action_index],
    )
    stats.maximum_arm_position_error = max(
      stats.maximum_arm_position_error, arm_result.position_error
    )
    stats.maximum_arm_orientation_error = max(
      stats.maximum_arm_orientation_error, arm_result.orientation_error
    )
    if not arm_result.success:
      stats.arm_ik_failures += 1
      bounded = (
        arm_result.position_error <= args.max_arm_position_error
        and arm_result.orientation_error <= args.max_arm_orientation_error
      )
      if args.strict_ik or not bounded:
        raise RuntimeError(
          f"{side} arm IK failed: position_error={arm_result.position_error:.4f}m, "
          f"orientation_error={arm_result.orientation_error:.4f}rad"
        )
      simulation.set_arm_joint_goal(side, arm_result.joint_positions)

    hand_result = simulation.solve_hand_ik(
      side,
      targets.fingertips_world[action_index],
      arm_joint_positions=arm_result.joint_positions,
      position_tolerance=args.hand_ik_tolerance,
    )
    stats.maximum_hand_error = max(stats.maximum_hand_error, hand_result.position_error)
    if not hand_result.success:
      stats.hand_ik_failures += 1
    if args.strict_ik and not hand_result.success:
      raise RuntimeError(
        f"{side} hand IK failed: maximum fingertip error="
        f"{hand_result.position_error:.4f}m"
      )
    if hand_result.position_error > args.max_hand_error:
      raise RuntimeError(
        f"{side} hand target is too far from the reachable manifold: "
        f"error={hand_result.position_error:.4f}m > {args.max_hand_error:.4f}m"
      )
    accepted = simulation.set_hand_joint_targets(
      hand_result.joint_names, hand_result.joint_positions
    )
    if accepted != len(hand_result.joint_names):
      raise RuntimeError(
        f"{side} hand controller accepted {accepted}/{len(hand_result.joint_names)} targets"
      )


def _hold_final_view(
  viewer: Any | None,
  sensor_dashboard: InferenceSensorDashboard | None,
  seconds: float,
) -> None:
  if (viewer is None and sensor_dashboard is None) or seconds <= 0.0:
    return
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    active = False
    if viewer is not None and viewer.is_running():
      viewer.sync()
      active = True
    if sensor_dashboard is not None and sensor_dashboard.is_running():
      sensor_dashboard.sync(force=True)
      active = True
    if not active:
      break
    time.sleep(1.0 / 60.0)


async def _run(args: argparse.Namespace) -> RolloutStats:
  output = args.output_dir or Path("artifacts") / time.strftime(
    "pick_place_policy_%Y%m%d_%H%M%S"
  )
  if args.record:
    output.mkdir(parents=True, exist_ok=False)
  simulation = ArmHandSimulation(
    add_genesis_probes=args.sensor_dashboard, scene="pick-place"
  )
  simulation.reset(
    seed=args.seed,
    object_xy_jitter=args.object_xy_jitter,
    object_yaw_jitter=args.object_yaw_jitter,
    randomized_objects=("cylinder",),
  )
  head_camera = CameraConfig(
    "head",
    width=args.width,
    height=args.height,
    rgb=True,
    depth=False,
    segmentation=False,
  )
  left_wrist_camera = CameraConfig(
    "left_wrist",
    width=args.width,
    height=args.height,
    rgb=True,
    depth=False,
    segmentation=False,
  )
  wrist_camera = CameraConfig(
    "right_wrist",
    width=args.width,
    height=args.height,
    rgb=True,
    depth=False,
    segmentation=False,
  )
  overhead_camera = CameraConfig(
    "overhead",
    width=args.width,
    height=args.height,
    rgb=True,
    depth=False,
    segmentation=False,
  )
  cameras = (
    (head_camera, left_wrist_camera, wrist_camera, overhead_camera)
    if args.sensor_dashboard
    else (head_camera, left_wrist_camera, wrist_camera)
  )
  tactile = (
    GenesisProbeTactileProvider(simulation.model, simulation.genesis_probe_layout)
    if args.sensor_dashboard
    else None
  )
  stats = RolloutStats(
    maximum_object_height=float(simulation.object_pose("cylinder")[2])
  )
  stabilizer = AutomaticGraspStabilizer(
    simulation,
    enabled=args.auto_grasp_stabilizer,
    contact_steps=args.stabilizer_contact_steps,
    release_closure=args.stabilizer_release_closure,
  )
  free_close_force = AdaptiveFreeCloseForce(
    simulation,
    enabled=args.adaptive_grasp_force,
    force_limit=args.free_close_force_limit,
    distance_cutoff=args.free_close_cutoff_m,
    minimum_closure=args.free_close_min_closure,
  )
  placement_detector = StablePlacementDetector(timestep=simulation.timestep)
  episode_start_sim = float(simulation.data.time)
  episode_start_wall = time.monotonic()
  control_tick = 0
  stop_requested = False
  recorder = None
  run_status = "task_not_completed"
  run_error = None

  try:
    with (
      WorkcellRenderer(
        simulation.model,
        cameras,
        visible_geom_groups=(0, 1, 2, 3, 4),
      ) as renderer,
      _viewer_context(simulation, args.viewer) as viewer,
      _sensor_dashboard_context(
        simulation, tactile, renderer, overhead_camera, args
      ) as sensor_dashboard,
    ):
      async with EgoSteerPolicyClient(
        args.server,
        open_timeout=args.open_timeout,
        response_timeout=args.response_timeout,
      ) as client:
        metadata = client.metadata or {}
        prediction_horizon = _model_action_horizon(metadata)
        camera_names = _model_camera_names(metadata)
        observation_contract = metadata.get("observation_contract", {})
        image_horizon = int(observation_contract.get("image_history", 6))
        image_stride = int(observation_contract.get("image_stride", 30))
        model_cameras = {
          name: {
            "head": head_camera,
            "left_wrist": left_wrist_camera,
            "right_wrist": wrist_camera,
          }[name]
          for name in camera_names
        }
        histories = {
          name: ObservationHistory(horizon=image_horizon, stride=image_stride)
          for name in camera_names
        }
        calibrations = _append_model_observations(
          simulation, renderer, model_cameras, histories
        )
        if viewer is not None:
          viewer.sync()
        if sensor_dashboard is not None:
          sensor_dashboard.sample_physics()
          sensor_dashboard.sync(force=True)
        if args.execute_steps > prediction_horizon:
          raise RuntimeError(
            f"--execute-steps={args.execute_steps} exceeds server horizon "
            f"{prediction_horizon}"
          )
        if args.record:
          from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo

          recorder = EvaluationVideo(
            simulation,
            output / "review",
            fps=args.record_fps,
            width=args.review_width,
            height=args.review_height,
            render_width=args.review_render_width,
            render_height=args.review_render_height,
            second_camera=args.review_second_camera,
            metadata={
              "server": args.server,
              "server_metadata": metadata,
              "instruction": args.instruction,
              "prediction_horizon": prediction_horizon,
              "execute_steps": args.execute_steps,
              "control_hz": CONTROL_HZ,
              "replan_period_s": args.execute_steps / CONTROL_HZ,
            },
          )
          recorder.capture(0, "initial")
        print(f"[policy] connected to {args.server}: {metadata}", flush=True)
        if sensor_dashboard is not None:
          sensor_dashboard.set_policy_status(f"policy connected: {metadata}")
          sensor_dashboard.sync(force=True)

        while float(simulation.data.time) - episode_start_sim < args.max_sim_seconds:
          if viewer is not None and not viewer.is_running():
            break
          if sensor_dashboard is not None and not sensor_dashboard.is_running():
            break
          if args.max_requests and stats.requests >= args.max_requests:
            break

          selected = _select_model_observation(
            camera_names, histories, calibrations
          )
          camera_from_world = camera_from_world_opencv(
            calibrations["head"].world_from_camera
          )
          states = model_state_history(selected.raw_states, camera_from_world)
          request_start = time.monotonic()
          if sensor_dashboard is not None:
            sensor_dashboard.set_policy_status(
              f"request {stats.requests + 1}: waiting for remote inference"
            )
            sensor_dashboard.sync()
          response = await _infer_while_servicing_viewer(
            client,
            viewer,
            sensor_dashboard,
            images=selected.images,
            states=states,
            camera_intrinsics=selected.camera_intrinsics,
            instruction=args.instruction,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
          )
          request_wall = time.monotonic() - request_start
          stats.requests += 1
          infer_ms = response.server_timing.get("infer_ms")
          infer_label = "?" if infer_ms is None else f"{infer_ms:.1f}ms"
          print(
            f"[policy] request={stats.requests} sim_t={simulation.data.time:.3f}s "
            f"server_infer={infer_label} round_trip={request_wall:.2f}s",
            flush=True,
          )
          if sensor_dashboard is not None:
            sensor_dashboard.set_policy_status(
              f"request {stats.requests}: executing actions | "
              f"server={infer_label} | round trip={request_wall:.2f}s"
            )

          predicted_actions = _validated_prediction(
            response.pred_actions, horizon=prediction_horizon
          )
          absolute_actions = relative_to_absolute(states[-1], predicted_actions)
          decoded = decode_action_targets(
            simulation, absolute_actions, camera_from_world
          )

          for action_index in range(args.execute_steps):
            if float(simulation.data.time) - episode_start_sim >= args.max_sim_seconds:
              break
            if viewer is not None and not viewer.is_running():
              raise ViewerClosed("native viewer was closed")
            if sensor_dashboard is not None and not sensor_dashboard.is_running():
              raise ViewerClosed("sensor dashboard was closed")
            step_wall_start = time.monotonic()
            _command_action_step(simulation, decoded, action_index, args, stats)
            stabilizer.after_command()

            control_tick += 1
            target_sim_time = episode_start_sim + control_tick / CONTROL_HZ
            while simulation.data.time < target_sim_time - 1.0e-12:
              free_close_force.before_physics_step(
                closure=stabilizer.closure(),
                grasp_active=stabilizer.active,
              )
              simulation.step()
              stabilizer.after_physics_step()
              object_twist = simulation.object_twist("cylinder")
              stable_success = placement_detector.update(
                linear_speed=float(np.linalg.norm(object_twist[:3])),
                angular_speed=float(np.linalg.norm(object_twist[3:])),
                placed_in_box=cylinder_is_in_box(simulation),
                grasp_active=stabilizer.active,
              )
              if stable_success and not stats.stable_success:
                stats.stable_success = True
                print(
                  "[sim] cylinder stably placed in box at "
                  f"t={simulation.data.time:.3f}s "
                  f"linear_speed={placement_detector.linear_speed:.4f}m/s "
                  f"angular_speed={placement_detector.angular_speed:.4f}rad/s",
                  flush=True,
                )
              if args.stop_on_success and stats.stable_success:
                stop_requested = True
                break
              if sensor_dashboard is not None:
                sensor_dashboard.sample_physics()
            stats.action_steps += 1
            stats.maximum_object_height = max(
              stats.maximum_object_height,
              float(simulation.object_pose("cylinder")[2]),
            )
            calibrations = _append_model_observations(
              simulation, renderer, model_cameras, histories
            )
            if recorder is not None:
              recorder.capture(
                control_tick, "placed" if stats.stable_success else "policy"
              )
            if viewer is not None:
              viewer.sync()
            if sensor_dashboard is not None:
              sensor_dashboard.sync()
            if args.real_time:
              delay = 1.0 / CONTROL_HZ - (time.monotonic() - step_wall_start)
              if delay > 0.0:
                time.sleep(delay)

            if stop_requested:
              break

          if stop_requested:
            break

      if stop_requested:
        _hold_final_view(viewer, sensor_dashboard, args.hold_final_seconds)
      run_status = "success" if stats.stable_success else "task_not_completed"
  except ViewerClosed as error:
    run_status = "viewer_closed"
    run_error = str(error)
    print(f"[viewer] {error}", flush=True)
  except BaseException as error:
    run_status = "error"
    run_error = f"{type(error).__name__}: {error}"
    raise
  finally:
    try:
      if recorder is not None:
        evaluation = {
          "success": stats.stable_success,
          "placed_in_box": cylinder_is_in_box(simulation),
        }
        record_error = None
        try:
          recorder.capture(
            control_tick,
            "placed" if stats.stable_success else "terminal",
            force=True,
          )
        except Exception as error:
          record_error = f"{type(error).__name__}: {error}"
        try:
          recorder.finish(
            status=run_status,
            evaluation=evaluation,
            error=run_error or record_error,
          )
        except Exception:
          if run_error is None:
            raise
    finally:
      free_close_force.close()
      stabilizer.close()

  wall_elapsed = time.monotonic() - episode_start_wall
  placed = cylinder_is_in_box(simulation)
  print(
    "[summary] "
    f"success={stats.stable_success} placed_in_box={placed} "
    f"requests={stats.requests} "
    f"action_steps={stats.action_steps} sim_elapsed_s="
    f"{simulation.data.time - episode_start_sim:.3f} "
    f"wall_elapsed_s={wall_elapsed:.3f} "
    f"max_object_height={stats.maximum_object_height:.3f} "
    f"arm_ik_failures={stats.arm_ik_failures} "
    f"hand_ik_failures={stats.hand_ik_failures} "
    f"max_hand_error={stats.maximum_hand_error:.4f}m",
    flush=True,
  )
  return stats


def main() -> None:
  args = _parse_args()
  asyncio.run(_run(args))


if __name__ == "__main__":
  main()
