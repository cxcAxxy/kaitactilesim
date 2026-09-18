"""Explicit episode initialization, separate from robot execution."""

import mujoco
import numpy as np

from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config


def _initialization_address(simulation: ArmHandSimulation) -> int:
  if simulation.scene != "usb-insert":
    raise ValueError("USB initialization requires scene='usb-insert'")
  if simulation.data.time != 0.0:
    raise ValueError("USB initialization is only allowed immediately after reset")
  joint = simulation.model.joint("usb_plug_freejoint").id
  return int(simulation.model.jnt_qposadr[joint])


def _initialize_orientation(simulation: ArmHandSimulation, quaternion) -> None:
  address = _initialization_address(simulation)
  simulation.data.qpos[address + 3 : address + 7] = quaternion
  mujoco.mj_forward(simulation.model, simulation.data)


def initialize_face_down(simulation: ArmHandSimulation) -> None:
  """Place the USB mark down at time zero, before any robot action or physics."""
  _initialize_orientation(simulation, [0.0, 1.0, 0.0, 0.0])


def initialize_for_insertion(
  simulation: ArmHandSimulation,
  *,
  seed: int = 0,
  xy_jitter_m: float = 0.0,
  yaw_jitter_rad: float = 0.0,
  offset_xy_m=None,
  yaw_offset_rad: float | None = None,
) -> dict:
  """Initialize once after reset with bounded XY and world-Z yaw perturbations.

  Explicit offsets support reproducible boundary cases and replace the random
  range for that component. Robot state, USB height and velocities are retained.
  """
  address = _initialization_address(simulation)
  xy_jitter_m, yaw_jitter_rad = float(xy_jitter_m), float(yaw_jitter_rad)
  if (
    not np.isfinite([xy_jitter_m, yaw_jitter_rad]).all()
    or min(xy_jitter_m, yaw_jitter_rad) < 0
  ):
    raise ValueError("USB jitter ranges must be finite and nonnegative")
  if not isinstance(seed, (int, np.integer)) or seed < 0:
    raise ValueError("USB initialization seed must be a nonnegative integer")
  if offset_xy_m is not None:
    offset_xy_m = np.array(offset_xy_m, dtype=float, copy=True)
    if offset_xy_m.shape != (2,) or not np.isfinite(offset_xy_m).all():
      raise ValueError("offset_xy_m must contain two finite coordinates")
    if xy_jitter_m != 0.0:
      raise ValueError("offset_xy_m and xy_jitter_m are mutually exclusive")
  if yaw_offset_rad is not None:
    yaw_offset_rad = float(yaw_offset_rad)
    if not np.isfinite(yaw_offset_rad):
      raise ValueError("yaw_offset_rad must be finite")
    if yaw_jitter_rad != 0.0:
      raise ValueError("yaw_offset_rad and yaw_jitter_rad are mutually exclusive")
  rng = np.random.default_rng(seed)
  if offset_xy_m is None:
    offset_xy_m = (
      rng.uniform(-xy_jitter_m, xy_jitter_m, size=2) if xy_jitter_m else np.zeros(2)
    )
  if yaw_offset_rad is None:
    yaw_offset_rad = (
      float(rng.uniform(-yaw_jitter_rad, yaw_jitter_rad)) if yaw_jitter_rad else 0.0
    )
  quaternion = config.AUTO_PLUG_QUATERNION_WXYZ.copy()
  if yaw_offset_rad != 0.0:
    yaw_quaternion = np.array(
      [np.cos(yaw_offset_rad / 2), 0.0, 0.0, np.sin(yaw_offset_rad / 2)]
    )
    mujoco.mju_mulQuat(quaternion, yaw_quaternion, config.AUTO_PLUG_QUATERNION_WXYZ)
  simulation.data.qpos[address : address + 2] += offset_xy_m
  simulation.data.qpos[address + 3 : address + 7] = quaternion
  mujoco.mj_forward(simulation.model, simulation.data)
  return {
    "seed": int(seed),
    "xy_jitter_m": xy_jitter_m,
    "yaw_jitter_rad": yaw_jitter_rad,
    "offset_xy_m": offset_xy_m.tolist(),
    "yaw_offset_rad": yaw_offset_rad,
    "initial_pose_wxyz": simulation.data.qpos[address : address + 7].tolist(),
  }
