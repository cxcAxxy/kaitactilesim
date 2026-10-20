"""Actuator-only expert for a bulb already engaged at the socket entrance.

The original pickup/transfer/insertion policy remains untouched.  This policy
shares its force-regulated finger gait but starts with a stationary threaded
bulb.  Neither bulb state nor passive thread coordinates are written here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import numpy as np

from kaihand_tactile_env.tasks.bulb_screw import config as screw_config
from kaihand_tactile_env.tasks.bulb_screw.execution import (
  BulbScrewExecutor,
  BulbScrewResult,
  _TaskFailure,
)
from kaihand_tactile_env.tasks.bulb_screw.grasp import calibrated_grasp

from .task import BulbScrewInplaceMonitor, BulbScrewInplaceSimulation


@dataclass(frozen=True)
class BulbScrewInplaceResult(BulbScrewResult):
  """Physical facts plus expert execution outcome.

  ``success`` means the whole expert workflow finished. ``physical_success``
  retains a completed installation even if subsequent home cleanup failed.
  The native monitor's own ``state.success`` is physical completion only.
  """

  task: str = "bulb-screw-inplace"
  initially_threaded_verified: bool = False
  grasp_verified: bool = False
  effective_rotation_verified: bool = False
  released_stably: bool = False
  physical_success: bool = False
  home_return_verified: bool = False


class BulbScrewInplaceExecutor(BulbScrewExecutor):
  """Approach, grasp, turn, confirm loaded stall, release, verify, then home.

  Stroke count is diagnostic only.  Rotation stops using measured bulb angle;
  all completion gates are independently checked by the inplace monitor.
  """

  def __init__(
    self,
    simulation: BulbScrewInplaceSimulation,
    observer: Callable[[BulbScrewInplaceSimulation, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    *,
    max_strokes: int = 20,
    speed: str = "fast",
    grasp_mode: str = "five-finger",
  ) -> None:
    if not isinstance(simulation, BulbScrewInplaceSimulation):
      raise ValueError("BulbScrewInplaceExecutor requires BulbScrewInplaceSimulation")
    if grasp_mode != "five-finger":
      raise ValueError("inplace screwing requires five-finger actuation")
    super().__init__(
      simulation,
      observer=observer,
      should_stop=should_stop,
      max_strokes=max_strokes,
      speed=speed,
      grasp_mode=grasp_mode,
    )
    # Measurement is always enabled.  Illumination is requested only from the
    # strict monitor certificate, never directly from a successful hand gesture.
    self.monitor = BulbScrewInplaceMonitor(
      simulation, auto_confirm_tightening=False
    )
    self._state = self.monitor.measure()

  def _step(self, phase, **checks):
    # Common collectors observe simulation.step() internally, before the expert
    # observer callback.  Publish the real phase before crossing that boundary.
    self.sim.task_phase = phase
    super()._step(phase, **checks)
    failure = getattr(self._state, "failure_reason", None)
    if failure is not None:
      raise _TaskFailure(failure)

  def _check_tightening_stall(self):
    """Use the same independent physical certificate as policy evaluation."""
    self._state = self.monitor.verify_tightening()
    verified = bool(self._state.tightening_verified)
    self._tightening_verified = self._tightening_verified or verified
    self._tightening_peak_torque = max(
      self._tightening_peak_torque,
      float(self._state.tightening_peak_torque_nm),
    )
    if verified:
      self._tightening_stall_s = max(
        self._tightening_stall_s,
        float(self._state.tightening_stall_duration_s),
      )

  def _motion(self):
    if (
      not self.sim.thread_engaged
      or not self._state.initially_threaded_verified
      or self._state.seated
      or self._state.bulb_lit
    ):
      raise _TaskFailure("task requires a stable, unlit bulb at thread entrance")
    self.grasp = calibrated_grasp(self.sim, five_finger=True)
    approach = self.timing.pickup_scale
    retreat = self.timing.retreat_scale
    self.sim.set_hand_joint_targets(self._names, self.grasp.open_hand)
    self._move_ee(
      self.grasp.wrist_position + [0, 0, 0.12],
      self.grasp.wrist_rotation,
      3.0 * approach,
      "hover",
      require_thread=True,
    )
    self._move_ee(
      self.grasp.wrist_position,
      self.grasp.wrist_rotation,
      3.0 * approach,
      "approach",
      require_thread=True,
    )
    self._advance(0.5 * approach, "approach", require_thread=True)
    self._hand_motion(
      self.grasp.closed_hand,
      2.0 * approach,
      "grasp",
      require_thread=True,
    )
    self._preload(1.0 * approach, "preload", require_thread=True)
    self._advance(
      0.35, "grasp", require_grip=True, require_thread=True
    )
    if not self._state.grasp_verified:
      raise _TaskFailure("stable five-finger grasp was not established")

    # There is deliberately no lift, transfer, alignment or thread-entry move.
    # The inherited gait retains a fixed arm goal and measures actual progress.
    self._screw_with_fingers()
    if not self._state.effective_rotation_verified:
      raise _TaskFailure("finger motion did not establish effective bulb rotation")
    if not self._tightening_verified or not self._state.tightening_verified:
      raise _TaskFailure("physical loaded-stall certificate is missing")

    self._hand_motion(
      self.grasp.open_hand,
      1.0 * retreat,
      "release",
      require_thread=True,
    )
    wrist, rotation = self.sim.current_pose_matrix("right")
    self._move_ee(
      wrist + [0, 0, 0.10],
      rotation,
      1.5 * retreat,
      "retreat",
      require_thread=True,
    )
    self._advance(0.5, "verify_seated", require_thread=True)
    if (
      not self._state.physical_success
      or not self._state.released_stably
      or not self._state.released
      or np.max(self._grip) > screw_config.RELEASE_FINGERTIP_LOAD_N
    ):
      raise _TaskFailure("bulb did not remain independently seated after release")

    # Home is a separate cleanup gate after physical release was demonstrated.
    # Follow the tested high return path to avoid sweeping across the left hand.
    self._move_ee(
      self.grasp.wrist_position + [0, 0, 0.18],
      self.grasp.wrist_rotation,
      2.0 * retreat,
      "clear_for_home",
      require_thread=True,
    )
    self.sim.set_hand_home("right")
    self._move_ee(
      self._home_wrist_position + [0, 0, 0.18],
      self._home_wrist_rotation,
      2.0 * retreat,
      "return_home",
      require_thread=True,
    )
    self._move_ee(
      self._home_wrist_position,
      self._home_wrist_rotation,
      2.0 * retreat,
      "return_home",
      require_thread=True,
    )
    start = self.sim.data.qpos[self.sim._arm_qpos["right"]].copy()
    for i in range(1, 151):
      u = i / 150
      alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
      self.sim.set_arm_joint_goal(
        "right", start + alpha * (screw_config.ARM_HOME["right"] - start)
      )
      self._advance(0.02, "return_home", require_thread=True)
    self._advance(0.5, "return_home", require_thread=True)
    self._advance(0.5, "verify_complete", require_thread=True)
    if not self._state.success or not self._state.home_return_verified:
      raise _TaskFailure("physical completion or home return was not verified")

  def execute(self) -> BulbScrewInplaceResult:
    result = super().execute()
    state = result.state
    values = {field.name: getattr(result, field.name) for field in fields(result)}
    if state.physical_success and not result.success:
      values["reason"] = f"cleanup_failed: {result.reason}"
    return BulbScrewInplaceResult(
      **values,
      initially_threaded_verified=bool(state.initially_threaded_verified),
      grasp_verified=bool(state.grasp_verified),
      effective_rotation_verified=bool(state.effective_rotation_verified),
      released_stably=bool(state.released_stably),
      physical_success=bool(state.physical_success),
      home_return_verified=bool(state.home_return_verified),
    )

  def run(self) -> BulbScrewInplaceResult:
    """Common variant-recorder interface."""
    return self.execute()


Executor = BulbScrewInplaceExecutor

__all__ = ["BulbScrewInplaceExecutor", "BulbScrewInplaceResult", "Executor"]
