"""Reset-only threaded preset and physical completion for in-place screwing.

This reuses the v15 passive pitch guide, contact pairs and actuator mechanics.
The sole object-pose intervention is the documented initial-condition reset.
Success never follows from a light, a commanded stroke count or a home pose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from kaihand_tactile_env.shared.pickup_randomization import PickupPositionRandomization
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.bulb_screw.task import (
  BulbScrewMonitor,
  BulbScrewSimulation,
  BulbScrewState,
)

from . import config


@dataclass(frozen=True)
class BulbScrewInplaceState(BulbScrewState):
  initially_threaded_verified: bool = False
  grasp_verified: bool = False
  effective_rotation_verified: bool = False
  released_stably: bool = False
  physical_success: bool = False
  home_return_verified: bool = False
  hand_home_now: bool = False
  arm_home_error_rad: float = 0.0
  hand_home_error_rad: float = 0.0
  home_joint_speed_rad_s: float = 0.0
  grasp_stable_duration_s: float = 0.0
  rotation_from_initial_turns: float = 0.0
  depth_from_initial_m: float = 0.0


class BulbScrewInplaceSimulation(BulbScrewSimulation):
  """An unlit bulb at the thread entry, with both hands initially at home."""

  def __init__(
    self,
    *,
    position_seed: int | None = None,
    physics_observer: Callable[["BulbScrewInplaceSimulation"], None] | None = None,
    **kwargs,
  ) -> None:
    if "scene" in kwargs:
      raise TypeError("BulbScrewInplaceSimulation owns scene='bulb-screw-inplace'")
    self.position_seed = position_seed
    self._pickup_randomization = None
    self.physics_observer = physics_observer
    self.initialization: dict[str, Any] | None = None
    self.reset_serial = 0
    self._right_hand_home_qpos: np.ndarray | None = None
    model_path = kwargs.pop("model_path", Path(__file__).with_name("scene.xml"))
    ArmHandSimulation.__init__(
      self, model_path=model_path, scene=config.SCENE_NAME, **kwargs
    )

  def reset(self, *, position_seed: int | None = None, **kwargs) -> None:
    self.reset_serial += 1
    if position_seed is not None:
      self.position_seed = int(position_seed)
    self.initialization = None
    # A no-support instance prevents the inherited tabletop reset from looking
    # up a cradle which this independent scene deliberately does not contain.
    if self._pickup_randomization is None:
      self._pickup_randomization = PickupPositionRandomization(self, "bulb")
    BulbScrewSimulation.reset(self, **kwargs)
    self.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
    if not self.try_engage_thread():
      raise RuntimeError("in-place reset could not engage the physical thread entry")
    entry_angle = float(self.data.qpos[self._hinge])
    entry_depth = float(self._alignment()[0])
    # Step the actual v15 model. No pose correction, external force, contact
    # change or bulb gravity compensation is used to pass this validation.
    ArmHandSimulation.step(
      self, max(1, round(config.THREADED_SETTLE_S / self.timestep))
    )
    before_zero = BulbScrewMonitor(self, auto_confirm_tightening=False).measure()
    angle = float(self.data.qpos[self._hinge])
    finite = bool(
      np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
    )
    valid = bool(
      finite
      and before_zero.engaged
      and not before_zero.seated
      and not before_zero.bulb_lit
      and before_zero.robot_contact_count == 0
      and before_zero.linear_speed_m_s <= config.SETTLED_LINEAR_SPEED_M_S
      and before_zero.angular_speed_rad_s <= config.SETTLED_ANGULAR_SPEED_RAD_S
      and abs(angle - entry_angle) <= config.INITIAL_MAX_ROTATION_RAD
      and abs(before_zero.insertion_depth_m - entry_depth)
      <= config.INITIAL_MAX_DEPTH_CHANGE_M
      and before_zero.insertion_depth_m
      < config.SEATED_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
      and abs(before_zero.thread_error_m) <= config.MAX_THREAD_ERROR_M
      and before_zero.lateral_error_m <= config.CAPTURE_LATERAL_M
      and before_zero.tilt_rad <= config.CAPTURE_TILT_RAD
    )
    if not valid:
      raise RuntimeError(
        "in-place thread entry failed physical settling: "
        f"engaged={before_zero.engaged}, seated={before_zero.seated}, "
        f"angle_drift_rad={angle - entry_angle:.5g}, "
        f"depth_drift_m={before_zero.insertion_depth_m - entry_depth:.5g}, "
        f"linear={before_zero.linear_speed_m_s:.5g}, "
        f"angular={before_zero.angular_speed_rad_s:.5g}, "
        f"robot_contacts={before_zero.robot_contact_count}"
      )
    # Validate first, then zero only velocities/bookkeeping for a reproducible
    # episode clock. The actual physically settled angle/depth are retained.
    self.data.time = 0.0
    self.data.qvel.fill(0.0)
    self.data.qacc.fill(0.0)
    self.data.qacc_warmstart.fill(0.0)
    self.data.qfrc_applied.fill(0.0)
    self.data.xfrc_applied.fill(0.0)
    self._set_bulb_lit(False)
    mujoco.mj_forward(self.model, self.data)
    self._right_hand_home_qpos = self.data.qpos[self._hand_qpos["right"]].copy()
    self.initialization = {
      "schema": "bulb_screw_inplace_threaded_reset_v1",
      "mechanics_version": config.MECHANICS_VERSION,
      "time_s": 0.0,
      "position_seed": self.position_seed,
      "position_randomization_applied": False,
      "thread_capture_active": self.thread_engaged,
      "thread_angle_qpos_rad": float(self.data.qpos[self._hinge]),
      "thread_depth_qpos_m": float(self.data.qpos[self._slide]),
      "initial_clockwise_turns": float(self.data.qpos[self._hinge] / (2 * np.pi)),
      "initial_insertion_depth_m": float(self._alignment()[0]),
      "bulb_pose_wxyz": self.object_pose("bulb").tolist(),
      "settled_under_physics_s": config.THREADED_SETTLE_S,
      "prezero_linear_speed_m_s": before_zero.linear_speed_m_s,
      "prezero_angular_speed_rad_s": before_zero.angular_speed_rad_s,
      "entry_angle_drift_rad": angle - entry_angle,
      "entry_depth_drift_m": before_zero.insertion_depth_m - entry_depth,
      "prezero_physically_settled": True,
      "physically_validated": True,
      "initially_threaded_verified": True,
      "initially_seated": False,
      "initially_lit": False,
      "velocity_zeroed": True,
      "model_parameters_changed": False,
    }
    self.initial_position_randomization = {
      "version": "bulb_screw_inplace_fixed_thread_entry_v1",
      "enabled": False,
      "seed": self.position_seed,
      "object": "bulb",
      "xy_half_range_m": 0.0,
      "offset_xyz_m": [0.0, 0.0, 0.0],
      "initial_pose_xyz_wxyz": self.object_pose("bulb").tolist(),
      "support_body": None,
      "support_body_position_m": None,
    }

  def initialize_threaded(self, *, position_seed: int | None = None) -> None:
    self.reset(position_seed=position_seed)

  def step(self, steps: int = 1) -> None:
    for _ in range(max(1, int(steps))):
      super().step(1)
      mujoco.mj_forward(self.model, self.data)
      if self.physics_observer is not None:
        self.physics_observer(self)


class BulbScrewInplaceMonitor(BulbScrewMonitor):
  """Ordered physical proof, independent from visual light and cleanup home.

With ``auto_confirm_tightening=False`` the same measured loaded-stall proof is
still accumulated; :meth:`verify_tightening` enables the light after that proof.
It never accepts the already-lit state as evidence of successful tightening.
"""

  def __init__(
    self,
    simulation: BulbScrewInplaceSimulation,
    *,
    auto_confirm_tightening: bool = True,
  ) -> None:
    if not isinstance(simulation, BulbScrewInplaceSimulation):
      raise ValueError("BulbScrewInplaceMonitor requires BulbScrewInplaceSimulation")
    super().__init__(simulation, auto_confirm_tightening=auto_confirm_tightening)

  def reset(self) -> None:
    super().reset()
    self._grasp_since: float | None = None
    self._grasp_verified = False
    self._grasp_start_turns: float | None = None
    self._grasp_start_depth_m: float | None = None
    self._effective_rotation_verified = False
    self._tightening_certified = False
    self._release_since: float | None = None
    self._released_stably = False
    self._home_since: float | None = None
    self._home_return_verified = False
    self._last_inplace_time: float | None = None
    self._latest_state: BulbScrewInplaceState | None = None
    self._reset_serial = self.simulation.reset_serial

  def _extra(self, state: BulbScrewState) -> BulbScrewInplaceState:
    sim = self.simulation
    initial = sim.initialization or {}
    arm_error = float(
      np.max(np.abs(sim.data.qpos[sim._arm_qpos["right"]] - config.ARM_HOME["right"]))
    )
    hand_error = float(
      np.inf
      if sim._right_hand_home_qpos is None
      else np.max(
        np.abs(sim.data.qpos[sim._hand_qpos["right"]] - sim._right_hand_home_qpos)
      )
    )
    speed = float(
      np.max(
        np.abs(
          sim.data.qvel[
            np.r_[sim._arm_dofs["right"], sim._hand_dofs["right"]]
          ]
        )
      )
    )
    home = bool(
      arm_error <= config.ARM_HOME_TOLERANCE_RAD
      and hand_error <= config.HAND_HOME_TOLERANCE_RAD
      and speed <= config.HOME_SPEED_RAD_S
      and state.robot_contact_count == 0
    )
    values = {field.name: getattr(state, field.name) for field in fields(BulbScrewState)}
    physical = bool(
      initial.get("initially_threaded_verified")
      and self._grasp_verified
      and self._effective_rotation_verified
      and self._tightening_certified
      and self._released_stably
      and state.seated
      and state.released
    )
    values["success"] = physical
    values["tightening_verified"] = self._tightening_certified
    return BulbScrewInplaceState(
      **values,
      initially_threaded_verified=bool(initial.get("initially_threaded_verified")),
      grasp_verified=self._grasp_verified,
      effective_rotation_verified=self._effective_rotation_verified,
      released_stably=self._released_stably,
      physical_success=physical,
      home_return_verified=self._home_return_verified,
      hand_home_now=home,
      arm_home_error_rad=arm_error,
      hand_home_error_rad=hand_error,
      home_joint_speed_rad_s=speed,
      rotation_from_initial_turns=state.clockwise_turns
      - float(initial.get("initial_clockwise_turns", state.clockwise_turns)),
      depth_from_initial_m=state.insertion_depth_m
      - float(initial.get("initial_insertion_depth_m", state.insertion_depth_m)),
    )

  def measure(self) -> BulbScrewInplaceState:
    return self._extra(super().measure())

  def _update_tightening(self, state: BulbScrewState) -> BulbScrewState:
    # Never take the parent monitor's bulb_lit shortcut. Even an externally
    # enabled visual light cannot fabricate the independent physical proof.
    # A former loaded stall cannot certify an object which was subsequently
    # disengaged or backed substantially out of its seated depth. In contrast,
    # a short speed/contact fluctuation does not erase a genuine earlier stall.
    backed_out = state.insertion_depth_m < (
      config.SEATED_DEPTH_M - 2 * config.SEATED_DEPTH_TOLERANCE_M
    )
    if not state.engaged or backed_out:
      self._tightening_certified = False
      self._tightening_history.clear()
      self._tightening_stall_duration_s = 0.0
      return replace(
        state, tightening_verified=False, tightening_stall_duration_s=0.0
      )
    if self._tightening_certified:
      return replace(
        state,
        tightening_verified=True,
        tightening_stall_duration_s=self._tightening_stall_duration_s,
        tightening_peak_torque_nm=self._tightening_peak_torque_nm,
      )
    if not (self._grasp_verified and self._effective_rotation_verified and state.seated):
      self._tightening_history.clear()
      self._tightening_stall_duration_s = 0.0
      return replace(state, tightening_verified=False)
    # Same numerical loaded-stall gate as v15, but without the visual-latch
    # shortcut. Certification must remain independent even when another
    # controller directly enabled the completion material.
    rotation = self.simulation.data.xmat[self.simulation._bulb_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    history = self._tightening_history
    history.append(
      (
        state.timestamp,
        yaw,
        state.hand_clockwise_torque_nm,
        min(state.fingertip_normal_force_n, default=0.0),
        state.right_hand_nonpad_contact_count == 0
        and state.robot_contact_count == state.right_hand_contact_count,
      )
    )
    while len(history) > 1 and state.timestamp - history[1][0] >= config.TIGHTENING_HOLD_S:
      history.popleft()
    self._tightening_peak_torque_nm = max(
      self._tightening_peak_torque_nm, state.hand_clockwise_torque_nm
    )
    duration = state.timestamp - history[0][0]
    displacement = float(np.ptp(np.unwrap(np.asarray([sample[1] for sample in history]))))
    self._tightening_certified = bool(
      duration >= config.TIGHTENING_HOLD_S - 1e-9
      and displacement <= config.TIGHTENING_MAX_ROTATION_RAD
      and min(sample[2] for sample in history) >= config.TIGHTENING_MIN_TORQUE_NM
      and min(sample[3] for sample in history) >= 4.0
      and all(sample[4] for sample in history)
    )
    self._tightening_stall_duration_s = duration if self._tightening_certified else 0.0
    if self._tightening_certified and self.auto_confirm_tightening:
      self.simulation.confirm_tightening()
    return replace(
      state,
      bulb_lit=self.simulation.bulb_lit,
      tightening_verified=self._tightening_certified,
      tightening_stall_duration_s=self._tightening_stall_duration_s,
      tightening_peak_torque_nm=self._tightening_peak_torque_nm,
    )

  def verify_tightening(self) -> BulbScrewInplaceState:
    """Idempotent confirmation using the most recently measured physics step."""
    if (
      self._reset_serial != self.simulation.reset_serial
      or self._latest_state is None
      or self._latest_state.timestamp != self.simulation.data.time
    ):
      self.update()
    if self._tightening_certified and not self.simulation.bulb_lit:
      self.simulation.confirm_tightening()
      self._latest_state = replace(self._latest_state, bulb_lit=True)
    return self._latest_state

  def update(self) -> BulbScrewInplaceState:
    time = float(self.simulation.data.time)
    if self._reset_serial != self.simulation.reset_serial:
      self.reset()
    if self._last_inplace_time is not None and time == self._last_inplace_time:
      return self._latest_state
    if self._last_inplace_time is not None and time < self._last_inplace_time:
      self.reset()
    discontinuity = bool(
      self._last_inplace_time is not None
      and time - self._last_inplace_time > 1.5 * self.simulation.timestep
    )
    if discontinuity:
      self._grasp_since = self._release_since = self._home_since = None
      self._released_stably = False
      # Also enforced by the base monitor, but keep this class's independent
      # loaded-stall proof explicitly fail-closed across missing physics ticks.
      self._tightening_history.clear()
      if not self._tightening_certified:
        self._tightening_stall_duration_s = 0.0
    state = super().update()
    clean_grasp = bool(
      state.initially_threaded_verified
      and state.engaged
      and len(state.fingertip_normal_force_n) == 5
      and min(state.fingertip_normal_force_n) >= config.GRASP_MIN_FORCE_N
      and state.right_hand_nonpad_contact_count == 0
      and state.robot_contact_count == state.right_hand_contact_count
    )
    if not clean_grasp:
      self._grasp_since = None
    elif self._grasp_since is None:
      self._grasp_since = time
      if not self._grasp_verified:
        self._grasp_start_turns = state.clockwise_turns
        self._grasp_start_depth_m = state.insertion_depth_m
    grasp_duration = 0.0 if self._grasp_since is None else time - self._grasp_since
    self._grasp_verified |= grasp_duration >= config.GRASP_STABLE_DWELL_S - 1e-12
    self._effective_rotation_verified |= bool(
      self._grasp_verified
      and clean_grasp
      and state.engaged
      and self._grasp_start_turns is not None
      and self._grasp_start_depth_m is not None
      and state.clockwise_turns - self._grasp_start_turns
      >= config.EFFECTIVE_ROTATION_MIN_TURNS
      and state.insertion_depth_m - self._grasp_start_depth_m
      >= config.EFFECTIVE_DEPTH_MIN_M
      and state.rotation_from_initial_turns >= config.EFFECTIVE_ROTATION_MIN_TURNS
      and state.depth_from_initial_m >= config.EFFECTIVE_DEPTH_MIN_M
    )
    physical_candidate = bool(
      state.initially_threaded_verified
      and self._grasp_verified
      and self._effective_rotation_verified
      and self._tightening_certified
      and state.seated
      and state.released
    )
    if not physical_candidate:
      self._release_since = None
      self._released_stably = False
    elif self._release_since is None:
      self._release_since = time
    release_duration = 0.0 if self._release_since is None else time - self._release_since
    self._released_stably = bool(
      physical_candidate and release_duration >= config.RELEASE_DWELL_S - 1e-12
    )
    if not (self._released_stably and state.hand_home_now):
      self._home_since = None
    elif self._home_since is None:
      self._home_since = time
    home_duration = 0.0 if self._home_since is None else time - self._home_since
    self._home_return_verified |= bool(
      self._released_stably and home_duration >= config.HOME_DWELL_S - 1e-12
    )
    self._last_inplace_time = time
    self._latest_state = replace(
      state,
      grasp_verified=self._grasp_verified,
      effective_rotation_verified=self._effective_rotation_verified,
      tightening_verified=self._tightening_certified,
      grasp_stable_duration_s=grasp_duration,
      released_stably=self._released_stably,
      release_stable_duration_s=release_duration,
      stable_duration_s=release_duration,
      physical_success=self._released_stably,
      success=self._released_stably,
      home_return_verified=self._home_return_verified,
    )
    return self._latest_state
