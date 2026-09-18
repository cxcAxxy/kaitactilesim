"""Task adapters for model-only closed-loop evaluation of the newer tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

TASK_INSTRUCTIONS = {
  "bulb-screw": (
    "Pick up the light bulb with the right hand, align it with the socket, "
    "screw it clockwise until mechanically seated, then release it."
  ),
  "install-ram": (
    "Pick up the RAM module with the right hand, align its keyed edge with "
    "the socket, press it straight down until seated, then release it."
  ),
  "vase-wipe": (
    "Pick up the sponge with the right hand and wipe the marked dirt from "
    "the inside wall of the vase until it is clean."
  ),
  "whiteboard-wipe": (
    "Pick up the eraser with the right hand, wipe all ink from the tilted "
    "whiteboard with loaded sliding contact, then return and release the eraser."
  ),
}


@dataclass(frozen=True)
class WhiteboardPolicyState:
  cleaned_fraction: float
  maximum_ink_remaining: float
  board_normal_force_n: float
  board_tangent_force_n: float
  table_support_force_n: float
  fingertip_normal_force_n: tuple[float, ...]
  eraser_linear_speed_m_s: float
  maximum_lift_m: float
  released_on_table: bool
  success: bool


class TaskPolicyAdapter:
  """Own task reset, per-physics-step metrics, guards, and success semantics."""

  def __init__(self, task: str, seed: int):
    if task not in TASK_INSTRUCTIONS:
      raise ValueError(f"unsupported shared policy task: {task!r}")
    if seed < 0:
      raise ValueError("seed must be nonnegative")
    self.task = task
    self.seed = seed
    self.instruction = TASK_INSTRUCTIONS[task]
    self._success = False
    if task == "bulb-screw":
      from kaihand_tactile_env.tasks.bulb_screw.task import (
        BulbScrewMonitor,
        BulbScrewSimulation,
      )

      self.simulation = BulbScrewSimulation(position_seed=seed)
      self.monitor = BulbScrewMonitor(self.simulation)
      self.state = self.monitor.measure()
      self.object_name = "bulb"
    elif task == "install-ram":
      from kaihand_tactile_env.tasks.install_ram.task import (
        RamInstallationMonitor,
        RamInstallSimulation,
      )

      self.simulation = RamInstallSimulation(position_seed=seed)
      self.monitor = RamInstallationMonitor(self.simulation)
      self.state = self.monitor.measure()
      self.object_name = "ram"
    elif task == "vase-wipe":
      from kaihand_tactile_env.tasks.vase_wipe.task import VaseWipeSimulation

      self.simulation = VaseWipeSimulation(stain_seed=seed)
      self.monitor = None
      self.state = self.simulation.measure()
      self.object_name = "sponge"
    else:
      from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation

      self.simulation = WhiteboardWipeSimulation(ink_seed=seed)
      self.monitor = None
      self.object_name = "eraser"
      self.maximum_lift_m = 0.0
      self.release_stable_s = 0.0
    self.initial_object_pose = self.simulation.object_pose(self.object_name).copy()
    if task == "whiteboard-wipe":
      self.state = self._whiteboard_state()

  @property
  def success(self) -> bool:
    return self._success

  def stage(self) -> str:
    if self._success:
      return "success"
    if self.task == "bulb-screw":
      if self.state.seated:
        return "seated"
      if self.state.engaged:
        return "thread_engaged"
      return "pick_align_screw"
    if self.task == "install-ram":
      if self.state.seated:
        return "seated"
      if self.state.insertion_depth_m > 0:
        return "inserting"
      if self.state.aperture_fits:
        return "aligned"
      return "pick_and_align"
    if self.task == "whiteboard-wipe":
      if self.state.maximum_ink_remaining <= 1e-9:
        return "return_and_release"
      if self.state.board_normal_force_n > 0.05:
        return "wiping"
      if self.state.maximum_lift_m > 0.05:
        return "carry_eraser"
      if max(self.state.fingertip_normal_force_n, default=0.0) > 0.05:
        return "eraser_grasped"
      return "pick_eraser"
    if self.state.wall_normal_force_n > 0:
      return "wiping"
    if max(self.state.fingertip_normal_force_n, default=0.0) > 0.05:
      return "sponge_grasped"
    return "pick_sponge"

  def step(self) -> None:
    if self.task == "whiteboard-wipe":
      self.simulation.phase = "policy"
    self.simulation.step()
    if self.task == "whiteboard-wipe":
      self.maximum_lift_m = max(
        self.maximum_lift_m,
        float(
          self.simulation.object_pose("eraser")[2]
          - self.initial_object_pose[2]
        ),
      )
      instant_release = self._whiteboard_release_condition()
      self.release_stable_s = (
        self.release_stable_s + self.simulation.timestep
        if instant_release
        else 0.0
      )
      self.state = self._whiteboard_state()
      self._success = self.state.success
    elif self.monitor is None:
      self.state = self.simulation.measure()
      self._success = bool(self.simulation.cleaning.success)
    else:
      self.state = self.monitor.update()
      self._success = bool(self.state.success)
    self._guard()

  def _whiteboard_state(self) -> WhiteboardPolicyState:
    sim = self.simulation
    remaining = np.asarray(sim.remaining, dtype=float)
    fingertip = np.asarray(sim.forces.read(sim.data).normal_force_n[5:], dtype=float)
    linear_speed = float(np.linalg.norm(sim.object_twist("eraser")[:3]))
    maximum_lift = float(getattr(self, "maximum_lift_m", 0.0))
    released = bool(getattr(self, "release_stable_s", 0.0) >= 0.1)
    maximum_remaining = float(remaining.max(initial=0.0))
    return WhiteboardPolicyState(
      cleaned_fraction=float(np.mean(1.0 - remaining)),
      maximum_ink_remaining=maximum_remaining,
      board_normal_force_n=float(sim.board_force),
      board_tangent_force_n=float(sim.board_tangent_force),
      table_support_force_n=float(sim.table_force),
      fingertip_normal_force_n=tuple(float(value) for value in fingertip),
      eraser_linear_speed_m_s=linear_speed,
      maximum_lift_m=maximum_lift,
      released_on_table=released,
      success=bool(
        maximum_lift > 0.12 and maximum_remaining <= 1e-9 and released
      ),
    )

  def _whiteboard_release_condition(self) -> bool:
    sim = self.simulation
    fingertip = np.asarray(sim.forces.read(sim.data).normal_force_n[5:], dtype=float)
    linear_speed = float(np.linalg.norm(sim.object_twist("eraser")[:3]))
    return bool(
      sim.table_force > 0.2
      and linear_speed < 0.01
      and fingertip.sum() < 0.05
    )

  def _guard(self) -> None:
    sim = self.simulation
    if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
      raise RuntimeError("nonfinite simulation state")
    if self.task == "bulb-screw":
      if float(sim.object_pose("bulb")[2]) < 0.5:
        raise RuntimeError("bulb fell below the work surface")
    elif self.task == "install-ram":
      from kaihand_tactile_env.tasks.install_ram import config

      if self.state.axial_resistance_n > config.INSERTION_FORCE_LIMIT_N:
        raise RuntimeError("RAM insertion resistance exceeded task limit")
      if self.state.maximum_socket_penetration_m > config.MAX_SOCKET_PENETRATION_M:
        raise RuntimeError("RAM socket penetration exceeded task limit")
      if float(sim.object_pose("ram")[2]) < 0.5:
        raise RuntimeError("RAM fell below the work surface")
    elif self.task == "vase-wipe":
      from kaihand_tactile_env.tasks.vase_wipe import config

      if self.state.wall_normal_force_n > config.MAX_CLEAN_NORMAL_N:
        raise RuntimeError("vase wall normal force exceeded cleaning limit")
      if float(sim.object_pose("sponge")[2]) < 0.4:
        raise RuntimeError("sponge fell below the work surface")
    else:
      from kaihand_tactile_env.tasks.whiteboard_wipe import cleaning

      if self.state.board_normal_force_n > cleaning.MAX_NORMAL_FORCE_N:
        raise RuntimeError("whiteboard normal force exceeded cleaning limit")
      if sim.direct_hand_board_force > 0.5:
        raise RuntimeError("hand struck the whiteboard")
      if float(sim.object_pose("eraser")[2]) < 0.5:
        raise RuntimeError("eraser fell below the work surface")
    if any(warning.number for warning in sim.data.warning):
      raise RuntimeError("simulation solver warning")

  def snapshot(self, _simulation: Any = None) -> dict[str, Any]:
    values = asdict(self.state)
    compact: dict[str, Any] = {"success": self._success, "stage": self.stage()}
    preferred = {
      "bulb-screw": (
        "clockwise_turns",
        "insertion_depth_m",
        "lateral_error_m",
        "seated",
      ),
      "install-ram": (
        "insertion_depth_m",
        "orientation_error_rad",
        "axial_resistance_n",
        "seated",
      ),
      "vase-wipe": (
        "cleaned_fraction",
        "wall_normal_force_n",
        "wall_tangent_force_n",
        "deformation_mm",
      ),
      "whiteboard-wipe": (
        "cleaned_fraction",
        "maximum_ink_remaining",
        "board_normal_force_n",
        "table_support_force_n",
        "maximum_lift_m",
        "released_on_table",
      ),
    }[self.task]
    compact.update({key: values[key] for key in preferred})
    return compact

  def metadata(self) -> dict[str, Any]:
    return {
      "name": f"{self.task}-policy-outcome",
      "success_source": (
        "whiteboard cleaned plus physical lift and stable table release"
        if self.task == "whiteboard-wipe"
        else (
          "task monitor stable success"
          if self.monitor is not None
          else "ground-truth cleaning completion"
        )
      ),
    }

  def report(self) -> dict[str, Any]:
    return {
      "task": self.task,
      "success": self._success,
      "stage": self.stage(),
      "state": asdict(self.state),
      "initial_object_pose_wxyz": self.initial_object_pose.tolist(),
      "final_object_pose_wxyz": self.simulation.object_pose(
        self.object_name
      ).tolist(),
      "initial_randomization": getattr(
        self.simulation, "initial_position_randomization", None
      ),
      "stain_randomization": getattr(
        self.simulation, "stain_randomization", None
      ),
      "ink_randomization": getattr(
        self.simulation, "ink_randomization", None
      ),
    }


def create_task_policy_adapter(task: str, seed: int) -> TaskPolicyAdapter:
  return TaskPolicyAdapter(task, seed)
