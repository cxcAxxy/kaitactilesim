"""Middle-pressure draw with compliant contact and bounded tactile pinch.

Neither the shared robot nor production task defaults are changed. The world-X
servo budget applies through the supported edge handover, not during pickup.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from .acceptance import (
  MAXIMUM_MULTI_FINGER_LOW_LOAD_S,
  MINIMUM_LOADED_FINGERS,
  STRICT_FORCE_POLICY,
  TASK_COMPLETION_POLICY,
  accept_edge,
  check_policy,
)
from .config import (
  _FINGERS,
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _PRESS_FORCE_CONTACT_RATIO,
)
from .pressure_window import (
  ForceLimitedPokerSimulation,
  PressureWindowExecutor,
  PressureWindowSettings,
)
from .task import PokerDrawPlan

MID_FORCE_PER_FINGER_N = 0.50
MID_FORCE_SETTINGS = PressureWindowSettings(
  goal="table-edge",
  table_friction=1.0,
  drive_limit_n=4.0,
  slide_speed_m_s=0.005,
)

CONTACT_MODEL_VERSION = "poker-compliant-contact-v2"
# Distinct compliant interfaces: the felt support is softer than the tactile
# pads. The old near-rigid .98/.995 impedance made contact-manifold changes
# unload three fingers in a single 2 ms step. Preserve friction, margins,
# solref, masses and solver clocks; never change parameters mid-episode.
MID_TABLE_CONTACT_IMPEDANCE = (0.90, 0.95, 0.0005, 0.5, 2.0)
MID_CARD_CONTACT_IMPEDANCE = (0.95, 0.98, 0.0005, 0.5, 2.0)


def configure_middle_contact_impedance(
  simulation: MidForcePokerSimulation,
) -> dict[str, object]:
  """Apply and record only the two poker contact interfaces at construction."""
  from .friction import TABLE_CARD_PAIR_NAME

  model = simulation.model
  pair = model.pair(TABLE_CARD_PAIR_NAME).id
  card = model.geom("card_core_geom").id
  metadata = {
    "contact_model_version": CONTACT_MODEL_VERSION,
    "table_card_pair_solimp_original": model.pair_solimp[pair].tolist(),
    "card_geom_solimp_original": model.geom_solimp[card].tolist(),
  }
  model.pair_solimp[pair] = MID_TABLE_CONTACT_IMPEDANCE
  model.geom_solimp[card] = MID_CARD_CONTACT_IMPEDANCE
  metadata.update(
    table_card_pair_solimp_used=model.pair_solimp[pair].tolist(),
    card_geom_solimp_used=model.geom_solimp[card].tolist(),
    contact_impedance_scope="table-card pair and priority-selected card-pad contact; fixed for entire episode",
    tactile_temporal_smoothing=False,
  )
  return metadata


@contextmanager
def middle_force_simulation(
  model_path: str | Path,
  *,
  simulation_type: type[MidForcePokerSimulation] | None = None,
) -> Iterator[tuple[MidForcePokerSimulation, dict]]:
  """Reconstruct the accepted preset without changing the production scene.

  The include wrapper lives through recording. Runtime contact overrides are
  deliberately explicit because the wrapper hash alone cannot describe them.
  """
  from .friction import TABLE_CARD_PAIR_NAME, model_with_table_card_friction

  with model_with_table_card_friction(
    model_path, MID_FORCE_SETTINGS.table_friction
  ) as path:
    constructor = (
      MidForcePokerSimulation if simulation_type is None else simulation_type
    )
    sim = constructor(path, scene="poker-draw", add_genesis_probes=False)
    model = sim.model
    pair = model.pair(TABLE_CARD_PAIR_NAME).id
    card = model.geom("card_core_geom").id
    metadata = {
      "physics_timestep_original_s": float(model.opt.timestep),
      "contact_friction_impedance_ratio_original": float(model.opt.impratio),
      "table_card_pair_solref_original": model.pair_solref[pair].tolist(),
      "card_geom_solref_original": model.geom_solref[card].tolist(),
    }
    settings = MID_FORCE_SETTINGS
    if settings.physics_timestep_s is not None:
      model.opt.timestep = settings.physics_timestep_s
      sim.timestep = settings.physics_timestep_s
    model.opt.impratio = settings.contact_friction_impedance_ratio
    if settings.finger_servo_velocity_gain is not None:
      raise RuntimeError("middle-force-v1 does not modify finger servo gains")
    if settings.contact_time_constant_s is not None:
      model.pair_solref[pair, 0] = settings.contact_time_constant_s
      model.geom_solref[card, 0] = settings.contact_time_constant_s
    metadata.update(
      table_card_pair_friction=model.pair_friction[pair].tolist(),
      physics_timestep_used_s=float(model.opt.timestep),
      contact_friction_impedance_ratio_used=float(model.opt.impratio),
      table_card_pair_solref_used=model.pair_solref[pair].tolist(),
      card_geom_solref_used=model.geom_solref[card].tolist(),
      card_geom_friction=model.geom_friction[card].tolist(),
      card_geom_priority=int(model.geom_priority[card]),
      experimental_right_finger_servo_gains={},
      add_genesis_probes=False,
    )
    metadata.update(configure_middle_contact_impedance(sim))
    yield sim, metadata


class MidForcePokerSimulation(ForceLimitedPokerSimulation):
  """Allow an explicit, command-continuous transition after a qualified draw."""

  @property
  def observation_time(self) -> float:
    """Time of cached FK/render/solver data, not post-integration qpos."""
    return min(
      float(self.data.time), getattr(self, "_observation_time", float(self.data.time))
    )

  def step(self, steps: int = 1) -> None:
    evaluated_at = float(self.data.time)
    super().step(steps)
    self._observation_time = max(evaluated_at, float(self.data.time) - self.timestep)

  def end_cartesian_drive(self) -> dict[str, object]:
    if self.drive_limit_n is None or getattr(self, "_cartesian_drive", None) is None:
      raise RuntimeError("middle-pressure handover requires an active Cartesian drive")
    ids = self._arm_actuators["right"]
    control = self.data.ctrl[ids].copy()
    qpos_before = self.data.qpos.copy()
    qvel_before = self.data.qvel.copy()
    limits = self.model.actuator_ctrlrange[ids]
    if (
      not np.all(np.isfinite(control))
      or np.any(control < limits[:, 0])
      or np.any(control > limits[:, 1])
    ):
      raise RuntimeError("invalid bounded servo command at handover")
    outcome = {
      "time_s": float(self.data.time),
      "from": "bounded_cartesian_world_x",
      "to": "original_joint_position_servo",
      "previous_drive_limit_n": float(self.drive_limit_n),
      "previous_drive_state": self.drive_state(),
      "discarded_arm_goal_rad": self._arm_goal["right"].tolist(),
      "preserved_actuator_control_rad": control.tolist(),
      "control_jump_rad": 0.0,
      "object_state_modified": False,
      "physics_parameters_modified": False,
    }
    # Carry the actual executed control, NOT the stale IK target or measured q.
    # This preserves the settled PD preload and removes accumulated path error.
    self._arm_goal["right"] = control.copy()
    self._arm_command["right"] = control.copy()
    self.__dict__.pop("_cartesian_drive", None)
    self.__dict__.pop("_drive_state", None)
    self.drive_limit_n = None
    outcome.update(
      qpos_before=qpos_before.tolist(),
      qvel_before=qvel_before.tolist(),
      maximum_qpos_change=float(np.max(np.abs(self.data.qpos - qpos_before))),
      maximum_qvel_change=float(np.max(np.abs(self.data.qvel - qvel_before))),
      control_jump_rad=float(np.max(np.abs(self.data.ctrl[ids] - control))),
    )
    return outcome


class MidForcePokerExecutor(PressureWindowExecutor):
  """Reuse pickup/view paths and success gates, with load-based jaw feedback."""

  def __init__(
    self,
    simulation: MidForcePokerSimulation,
    *,
    acceptance_policy: str = STRICT_FORCE_POLICY,
    **kwargs: object,
  ) -> None:
    if not isinstance(simulation, MidForcePokerSimulation):
      raise TypeError("middle full task requires MidForcePokerSimulation")
    if "press_force_per_finger_n" in kwargs:
      raise ValueError("middle full preset fixes pressure to 0.50 N per finger")
    super().__init__(
      simulation, press_force_per_finger_n=MID_FORCE_PER_FINGER_N, **kwargs
    )
    self.acceptance_policy = check_policy(acceptance_policy)
    self._multi_low_load_duration_s = 0.0
    self._maximum_multi_low_load_duration_s = 0.0
    self.handoff_outcome: dict[str, object] | None = None
    self._lift_reference: tuple[np.ndarray, np.ndarray] | None = None
    self._lift_compensation: list[dict[str, object]] = []
    self._pinch_force_filtered: np.ndarray | None = None
    self._pinch_force_reference: np.ndarray | None = None
    self._pinch_force_steps = 0

  def control_metadata(self) -> dict[str, object]:
    return {
      **super().control_metadata(),
      "preset": "middle_force_full_task_v1",
      "contact_model_version": CONTACT_MODEL_VERSION,
      "acceptance_policy": getattr(self, "acceptance_policy", STRICT_FORCE_POLICY),
      "training_safety": {
        "minimum_loaded_fingers": MINIMUM_LOADED_FINGERS,
        "maximum_multi_finger_low_load_s": MAXIMUM_MULTI_FINGER_LOW_LOAD_S,
        "maximum_observed_multi_low_load_s": getattr(
          self, "_maximum_multi_low_load_duration_s", 0.0
        ),
        "scope": "task-completion-v1 slide/edge only; other physical guards unchanged",
      },
      "object_motion_or_slip_used_for_control": True,
      "geometry_usage": (
        "draw endpoint, supported edge retreat, and measured card/tool lever-arm "
        "compensation of pickup waypoints; never object-state writes"
      ),
      "slip_feedback_used_for_pressure_adjustment": False,
      "raise_card_to_view_duration_scale": 2.0,
      "lift_card_duration_scale": 2.0,
      "turn_card_inward_duration_scale": 2.0,
      "pickup_controller": "bounded tactile normal-force pinch v2",
      "pinch_force_control": {
        "target_normal_n": [0.25, 0.25, 0.25, 0.25, 1.0],
        "finger_order": [*_FINGERS, "thumb"],
        "update_period_s": 0.010,
        "feedback_filter_s": 0.020,
        "maximum_joint_rate_deg_s": 1.5,
        "maximum_offset_deg": 1.0,
        "prelift_settle_s": 0.35,
        "raw_tactile_filtered": False,
      },
      "force_limit_scope": "slide_card and edge_hold; disabled by explicit handover",
      "contact_model_scope": "unchanged throughout the whole episode",
      "lift_waypoint_compensation": self._lift_compensation,
    }

  def _prepare_and_slide(
    self, plan: PokerDrawPlan, phases: list[str]
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    self.handoff_outcome = None
    self._lift_reference = None
    self._lift_compensation = []
    self._pinch_force_filtered = None
    self._pinch_force_reference = None
    self._pinch_force_steps = 0
    self._multi_low_load_duration_s = 0.0
    self._maximum_multi_low_load_duration_s = 0.0
    outcome = self.draw(plan, MID_FORCE_SETTINGS)
    policy = getattr(self, "acceptance_policy", STRICT_FORCE_POLICY)
    if not accept_edge(outcome, policy):
      raise RuntimeError("middle-pressure draw did not qualify; pickup is prohibited")
    phases.extend(waypoint.phase for waypoint in plan.waypoints)
    phases.extend(("four_finger_press", "slide_card", "edge_hold"))
    self.handoff_outcome = {"completed": False, "stage": "supported_edge_retreat"}
    # The original pickup releases/reforms the hand at the edge. Restore its
    # support margin by physically pushing the card inward, still at 4 N and
    # with the same per-finger Fn feedback. Never move the card state directly.
    self._supported_edge_retreat(plan)
    quality = self._slide_force_monitor.summary()
    if policy == STRICT_FORCE_POLICY and not self._slide_press_control_qualified(
      quality
    ):
      raise RuntimeError("middle-pressure edge handover lost continuous force quality")
    if (
      policy == TASK_COMPLETION_POLICY
      and self._maximum_slide_fingertip_plane_angle_degrees
      > _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
    ):
      raise RuntimeError("middle-pressure edge handover exceeded draw contact angle")
    edge_pose = self.sim.object_pose("card").copy()
    slide_position, _ = self.sim.current_pose_matrix(plan.side)
    transition = self.sim.end_cartesian_drive()
    self._press_controller_active = False
    self.handoff_outcome.update(
      completed=True,
      stage="joint_servo_ready",
      transition=transition,
      edge_card_pose=edge_pose.tolist(),
      slide_and_handoff_force_quality=asdict(quality),
      limit_scope="slide_card and edge_hold only; original servo during pickup/view",
    )
    return self.sim.arm_goal[plan.side], slide_position, edge_pose

  def _step(self, phase: str, table_edge_x: float) -> None:
    super()._step(phase, table_edge_x)
    if (
      getattr(self, "acceptance_policy", STRICT_FORCE_POLICY) != TASK_COMPLETION_POLICY
    ):
      return
    if phase not in {"slide_card", "edge_hold"}:
      self._multi_low_load_duration_s = 0.0
      return
    forces = self._current_card_finger_normal_forces()
    loaded = np.count_nonzero(
      forces >= self.press_force_per_finger_n * _PRESS_FORCE_CONTACT_RATIO
    )
    self._multi_low_load_duration_s = (
      self._multi_low_load_duration_s + self.sim.timestep
      if loaded < MINIMUM_LOADED_FINGERS
      else 0.0
    )
    self._maximum_multi_low_load_duration_s = max(
      self._maximum_multi_low_load_duration_s, self._multi_low_load_duration_s
    )
    if self._multi_low_load_duration_s >= MAXIMUM_MULTI_FINGER_LOW_LOAD_S - 1e-10:
      raise RuntimeError(
        "multiple draw fingers remained below load threshold for 0.30s"
      )

  def _supported_edge_retreat(self, plan: PokerDrawPlan) -> None:
    sim = self.sim
    start, _ = sim.current_pose_matrix(plan.side)
    commanded = 0.0
    # A tip-side draw stores more elastic deflection than the old flat pads.
    # At 800 N/m the unchanged 4 N budget can consume 5 mm of reference error
    # before the card moves; leave room for the ~1 mm supported-edge retreat.
    # This is a command bound, never extra force or a card-state correction.
    maximum_travel = 0.008
    target = 0.475
    started = float(sim.data.time)
    self._record_slide_force_metrics = True
    try:
      while self._overhang_fraction(plan.table_edge_x) > target:
        if float(sim.data.time) - started >= 2.0 or commanded >= maximum_travel:
          raise RuntimeError("card could not regain supported edge margin under 4 N")
        commanded += min(0.005 * sim.timestep, maximum_travel - commanded)
        reference = start.copy()
        reference[0] += commanded
        sim.set_cartesian_drive_position(reference)
        self._step("edge_hold", plan.table_edge_x)
      current, _ = sim.current_pose_matrix(plan.side)
      reference = start.copy()
      reference[0] = current[0]
      sim.set_cartesian_drive_position(reference)
      self._advance_fixed(0.10, "edge_hold", plan.table_edge_x)
      overhang = self._overhang_fraction(plan.table_edge_x)
      if not 0.44 <= overhang <= 0.48:
        raise RuntimeError("card left the supported edge handover geometry")
      self.handoff_outcome.update(
        retreat_duration_s=float(sim.data.time) - started,
        retreat_commanded_travel_m=commanded,
        overhang_fraction=overhang,
      )
    finally:
      self._record_slide_force_metrics = False

  def _move_pose_with_guarded_pinch(
    self,
    plan: PokerDrawPlan,
    position: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
    end_effector_rotation: np.ndarray,
  ) -> np.ndarray:
    if phase == "lift_card":
      # Tilting the wrist also translates a card ~15 cm ahead of the tool.
      # Compensate that lever arm so a 4/25 mm rise means a CARD rise, instead
      # of rotating the fingertips down into the still-supporting tabletop.
      current, rotation = self.sim.current_pose_matrix(plan.side)
      card = self.sim.object_pose("card")[:3].copy()
      if self._lift_reference is None:
        self._lift_reference = (current.copy(), card.copy())
      origin_tool, origin_card = self._lift_reference
      desired_card = origin_card + (position - origin_tool)
      local_offset = rotation.T @ (card - current)
      compensated = desired_card - end_effector_rotation @ local_offset
      if np.linalg.norm(compensated - position) > 0.04:
        raise RuntimeError("pickup lever-arm compensation exceeded 40 mm bound")
      self._lift_compensation.append(
        {
          "time_s": float(self.sim.data.time),
          "original_tool_target_m": np.asarray(position).tolist(),
          "compensated_tool_target_m": compensated.tolist(),
          "desired_card_center_m": desired_card.tolist(),
          "measured_local_card_offset_m": local_offset.tolist(),
        }
      )
      position = compensated
    # The softer experimental contacts need time to redistribute the pinch
    # when accelerating the card toward the head. Keep the original path.
    return super()._move_pose_with_guarded_pinch(
      plan,
      position,
      seed,
      duration=duration
      * (2.0 if phase in {"lift_card", "raise_card_to_view"} else 1.0),
      phase=phase,
      end_effector_rotation=end_effector_rotation,
    )

  def _approach_flat_pinch(
    self,
    plan: PokerDrawPlan,
    seed: np.ndarray,
  ) -> np.ndarray:
    seed = super()._approach_flat_pinch(plan, seed)
    # Establish force-supported opposition before accelerating the card.
    # Geometry-only contacts can still carry zero load with compliant pads.
    for _ in range(max(1, round(0.35 / self.sim.timestep))):
      self._step("thumb_face_press", plan.table_edge_x)
      self._maintain_dynamic_pinch(*self._current_card_face_contact_state())
    return seed

  def _maintain_dynamic_pinch(
    self,
    top_contacts: set[str],
    bottom_contacts: set[str],
    top_force: float,
    bottom_force: float,
  ) -> None:
    """Bounded low-bandwidth jaw force feedback, never modifying sensor data."""
    del top_contacts, bottom_contacts, top_force, bottom_force
    if self._pinch_flexion_targets is None:
      return
    _, _, forces = self._current_card_face_contact_details()
    measured = np.asarray([forces[f] for f in (*_FINGERS, "thumb")])
    if not np.all(np.isfinite(measured)) or np.any(measured < 0):
      raise RuntimeError("invalid pinch contact force feedback")
    if self._pinch_force_filtered is None:
      self._pinch_force_filtered = measured.copy()
      self._pinch_force_reference = np.r_[
        self._pinch_flexion_targets, self._pinch_thumb_joint5_target
      ]
    alpha = self.sim.timestep / (0.020 + self.sim.timestep)
    self._pinch_force_filtered += alpha * (measured - self._pinch_force_filtered)
    self._pinch_force_steps += 1
    period = max(1, round(0.010 / self.sim.timestep))
    if self._pinch_force_steps % period:
      return
    elapsed = period * self.sim.timestep
    target = np.asarray([0.25, 0.25, 0.25, 0.25, 1.0])
    error = target - self._pinch_force_filtered
    error[np.abs(error) <= 0.05] = 0.0
    rate = np.deg2rad(1.5) * elapsed
    delta = np.clip(0.10 * error * elapsed, -rate, rate)
    current = np.r_[self._pinch_flexion_targets, self._pinch_thumb_joint5_target]
    reference = self._pinch_force_reference
    updated = np.clip(
      current + delta, reference - np.deg2rad(1.0), reference + np.deg2rad(1.0)
    )
    self._pinch_flexion_targets[:] = updated[:4]
    self._pinch_thumb_joint5_target = float(updated[4])
    self.sim.set_hand_joint_targets(
      tuple(f"hand_r_{f}_joint2" for f in _FINGERS), updated[:4]
    )
    self.sim.set_hand_joint_targets(("hand_r_thumb_joint5",), (updated[4],))

  def _move_arm_joints_linear(
    self,
    plan: PokerDrawPlan,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    duration: float,
    phase: str,
  ) -> np.ndarray:
    return super()._move_arm_joints_linear(
      plan,
      target,
      seed,
      duration=duration * (2.0 if phase == "turn_card_inward" else 1.0),
      phase=phase,
    )
