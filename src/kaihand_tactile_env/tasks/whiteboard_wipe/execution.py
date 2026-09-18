"""Known-state actuator policy: grasp, lift, wipe, replace, and release.

No welds, object pose writes, or forces applied to the eraser during execution.
"""

from __future__ import annotations

import numpy as np

from ...shared.simulation import HAND_JOINT_NAMES, _rotation_vector_world
from . import cleaning
from . import config as C


def turn(vector):
  angle = np.linalg.norm(vector)
  if angle < 1e-12:
    return np.eye(3)
  x, y, z = vector / angle
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
  return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


class WhiteboardWipeExecutor:
  def __init__(self, simulation, observer=None):
    self.sim = simulation
    self.observer = observer
    self.position, self.rotation = simulation.current_pose_matrix("right")
    self.progress = np.zeros(5)
    self.holding = False
    self.closing = False
    self.phases = []
    self.peak_tactile = np.zeros(5)
    self.max_lift = 0.0
    self.pickup_verified = False

  def tick(self, phase):
    s = self.sim
    if self.closing or self.holding:
      loads = s.forces.read(s.data).normal_force_n[5:]
      rate = np.clip(
        1.2 * (np.array([2.0, 1.5, 1.2, 1.2, 0.8]) - loads),
        -0.1 if self.closing else 0,
        0.35 if self.closing else 0.06,
      )
      self.progress = np.clip(self.progress + 0.01 * rate, 0, 1.2)
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"],
        s.open_grip + np.repeat(self.progress, 4) * (s.grip - s.open_grip),
      )
    ik = s.solve_ik(
      "right",
      self.position,
      self.rotation,
      seed=s.arm_goal["right"],
      max_iterations=140,
      position_tolerance=0.00001 if phase in ("wipe", "support", "release") else 0.0003,
      orientation_tolerance=0.004,
      posture_weight=0,
    )
    if ik.position_error > 0.005 or ik.orientation_error > 0.05:
      raise RuntimeError(f"{phase}: unreachable wrist ({ik.position_error:.4f} m)")
    s.set_arm_joint_goal("right", ik.joint_positions)
    s.phase = phase
    s.step(round(0.01 / s.timestep))
    self.max_lift = max(
      self.max_lift, float(s.object_pose("eraser")[2] - s.pickup_center[2])
    )
    self.peak_tactile = np.maximum(
      self.peak_tactile, s.forces.read(s.data).normal_force_n[5:]
    )
    if not self.phases or self.phases[-1] != phase:
      self.phases.append(phase)
    if not np.isfinite(s.data.qpos).all() or any(w.number for w in s.data.warning):
      raise RuntimeError(f"{phase}: unstable physics")
    if self.holding:
      wrist, rot = s.current_pose_matrix("right")
      expected = wrist + rot @ self.tool_offset
      if np.linalg.norm(expected - s.object_pose("eraser")[:3]) > 0.045:
        raise RuntimeError(f"{phase}: eraser slipped from fingers")
    if self.observer:
      self.observer(s, phase)

  def move_wrist(self, target, seconds, phase, rotation=None):
    start, rot = self.position.copy(), self.rotation.copy()
    vector = (
      _rotation_vector_world(rotation, rot) if rotation is not None else np.zeros(3)
    )
    for i in range(round(seconds / 0.01)):
      u = (i + 1) / round(seconds / 0.01)
      u = u * u * (3 - 2 * u)
      self.position = start + u * (np.asarray(target) - start)
      self.rotation = turn(u * vector) @ rot
      self.tick(phase)

  def servo_tool(self, target, rotation, phase):
    s = self.sim
    actual = s.data.xpos[s.eraser_body]
    actual_rot = s.data.xmat[s.eraser_body].reshape(3, 3)
    wrist, wrist_rot = s.current_pose_matrix("right")
    gain = 0.06 if phase == "wipe" else 0.25
    delta = np.clip(gain * (target - actual), -0.001, 0.001)
    angular = gain * _rotation_vector_world(rotation, actual_rot)
    length = np.linalg.norm(angular)
    if length > 0.015:
      angular *= 0.015 / length
    change = turn(angular)
    self.position += delta + (change - np.eye(3)) @ (wrist - actual)
    self.rotation = change @ self.rotation
    lead = self.position - wrist
    if np.linalg.norm(lead) > 0.006:
      self.position = wrist + 0.006 * lead / np.linalg.norm(lead)
    self.tick(phase)

  def move_tool(self, target, seconds, phase, rotation):
    s = self.sim
    start = s.object_pose("eraser")[:3].copy()
    start_rot = s.data.xmat[s.eraser_body].reshape(3, 3).copy()
    vector = _rotation_vector_world(rotation, start_rot)
    for i in range(round(seconds / 0.01)):
      u = (i + 1) / round(seconds / 0.01)
      u = u * u * (3 - 2 * u)
      self.servo_tool(
        start + u * (np.asarray(target) - start), turn(u * vector) @ start_rot, phase
      )

  def place_eraser(self):
    """Transfer weight to the table, then unload opposing finger forces."""
    s = self.sim
    self.move_tool(
      s.pickup_center + [0, 0, 0.003], 3.0, "lower", C.INITIAL_ERASER_ROTATION
    )
    weight = float(s.model.body_mass[s.eraser_body] * -s.model.opt.gravity[2])
    for _ in range(500):
      if s.table_force >= 0.20:
        break
      self.position[2] -= 0.00002
      self.tick("support")
    else:
      raise RuntimeError("Table contact was not established before release")
    self.holding = False

    def unload(phase):
      self.position[2] += float(
        np.clip(0.00003 * (s.table_force - weight), -0.000015, 0.000015)
      )
      self.tick(phase)

    for _ in range(100):
      unload("support")
    if abs(s.table_force - weight) > 0.2:
      raise RuntimeError("Table has not taken the eraser weight before release")
    initial_loads = s.forces.read(s.data).normal_force_n[5:].copy()
    # Reduce all five measured normal loads toward zero before opening fully.
    for i in range(250):
      u = min((i + 1) / 150, 1.0)
      u = u * u * (3 - 2 * u)
      loads = s.forces.read(s.data).normal_force_n[5:]
      goal = (1 - u) * initial_loads
      rate = np.clip(0.08 * (goal - loads), -0.15, 0.04)
      self.progress = np.clip(self.progress + 0.01 * rate, 0, 1.2)
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"],
        s.open_grip + np.repeat(self.progress, 4) * (s.grip - s.open_grip),
      )
      unload("release")
    hand = np.array(
      [s._hand_targets["right"][name] for name in HAND_JOINT_NAMES["right"]]
    )
    for i in range(100):
      u = (i + 1) / 100
      s.set_hand_joint_targets(
        HAND_JOINT_NAMES["right"], hand + u * (s.open_grip - hand)
      )
      unload("release")

  def run(self):
    s = self.sim
    reason = "completed"
    released = False
    try:
      self.move_wrist(self.position, 0.4, "observe")
      if s.table_force < 0.2 or s.forces.read(s.data).normal_force_n[5:].sum() > 0.05:
        raise RuntimeError(
          "Eraser must initially rest on the table with no finger contact"
        )
      self.move_wrist(s.pickup_center + s.wrist_offset, 2.0, "approach")
      self.closing = True
      self.move_wrist(self.position, 4.0, "grasp")
      loads = s.forces.read(s.data).normal_force_n[5:]
      if loads[0] < 0.05 or loads[1:].sum() < 0.15:
        raise RuntimeError(f"No opposed tactile grasp: {loads}")
      self.closing = False
      wrist, rot = s.current_pose_matrix("right")
      self.tool_offset = rot.T @ (s.object_pose("eraser")[:3] - wrist)
      self.holding = True
      self.move_tool(
        s.pickup_center + [0, 0, 0.15], 2.5, "lift", C.INITIAL_ERASER_ROTATION
      )
      if s.table_force > 0.01 or self.max_lift < 0.12:
        raise RuntimeError("Eraser did not lift off the table")
      self.pickup_verified = True
      # Tool bottom faces the board; the handle remains on the robot side.
      center = s.ink_surface_center - C.BOARD_ROTATION @ C.PAD_BOTTOM
      self.move_tool(
        center + 0.055 * C.BOARD_NORMAL, 3.0, "orient_and_transfer", C.WIPE_ROTATION
      )
      self.move_tool(
        center + 0.002 * C.BOARD_NORMAL, 1.5, "approach_board", C.WIPE_ROTATION
      )
      wipe_offset = self.position - s.object_pose("eraser")[:3]
      wipe_rotation = self.rotation.copy()
      depth = 0.002
      for i in range(4500):
        # Use the last 10 ms of actual load directly: the extra EMA delayed
        # unloading, while coarse IK held small depth corrections until they
        # accumulated into a sudden motion. Keep the 2.5 N wiping target.
        depth = float(
          np.clip(depth + 0.000006 * (s.mean_board_force - 2.5), -0.012, 0.005)
        )
        target = (
          center
          + depth * C.BOARD_NORMAL
          + np.array([0, 0.045 * np.sin(2 * np.pi * (i * 0.01) / 4.0), 0])
          + 0.015 * np.sin(2 * np.pi * (i * 0.01) / 7.0) * C.BOARD_ROTATION[:, 0]
        )
        self.position = target + wipe_offset
        self.rotation = wipe_rotation
        self.tick("wipe")
        if i > 300 and np.max(s.remaining) <= 1e-9:
          break
      self.move_tool(
        center + 0.07 * C.BOARD_NORMAL, 1.5, "leave_board", C.WIPE_ROTATION
      )
      self.move_tool(
        s.pickup_center + [0, 0, 0.15], 3.0, "return", C.INITIAL_ERASER_ROTATION
      )
      self.place_eraser()
      self.move_wrist(self.position + [0, 0, 0.16], 2.0, "retreat")
      self.move_wrist(self.position, 0.8, "verify")
      released = bool(
        s.table_force > 0.2
        and np.linalg.norm(s.object_twist("eraser")[:3]) < 0.01
        and s.forces.read(s.data).normal_force_n[5:].sum() < 0.05
      )
      if not released:
        reason = "Eraser did not settle freely on table"
      elif np.max(s.remaining) > 1e-9:
        reason = "Ink remains after bounded wiping"
    except RuntimeError as exc:
      reason = str(exc)
    return dict(
      success=bool(
        self.pickup_verified
        and released
        and np.max(s.remaining) <= 1e-9
        and reason == "completed"
      ),
      reason=reason,
      ink_randomization=s.ink_randomization,
      elapsed_s=float(s.data.time),
      pickup_verified=self.pickup_verified,
      released_on_table=released,
      ink_remaining=s.remaining.tolist(),
      maximum_lift_m=self.max_lift,
      peak_board_force_n=s.peak_board_force,
      peak_board_tangent_force_n=s.peak_board_tangent_force,
      peak_direct_hand_board_force_n=s.peak_direct_hand_board_force,
      cleaning=dict(
        minimum_normal_force_n=cleaning.MIN_NORMAL_FORCE_N,
        maximum_normal_force_n=cleaning.MAX_NORMAL_FORCE_N,
        minimum_patch_tangent_load_n=cleaning.MIN_TANGENT_LOAD_N,
        minimum_sliding_speed_m_s=cleaning.MIN_SLIDING_SPEED_M_S,
        work_required_j=cleaning.REQUIRED_WORK_J,
        stroke_required_m=cleaning.REQUIRED_STROKE_M,
        loaded_time_required_s=cleaning.REQUIRED_LOADED_TIME_S,
        patch_work_j=s.cleaning.work_j.tolist(),
        patch_stroke_m=s.cleaning.stroke_m.tolist(),
        patch_loaded_time_s=s.cleaning.loaded_time_s.tolist(),
        remaining=s.remaining.tolist(),
      ),
      eraser_board_friction=s.model.pair_friction[
        s.model.pair("eraser_board_contact").id
      ].tolist(),
      eraser_board_solimp=s.model.pair_solimp[
        s.model.pair("eraser_board_contact").id
      ].tolist(),
      friction_impedance_ratio=float(s.model.opt.impratio),
      hand_velocity_gain=C.HAND_VELOCITY_GAIN,
      board_target_force_n=2.5,
      board_depth_gain_m_per_n_tick=0.000006,
      contact_control_ik_tolerance_m=0.00001,
      table_approach_speed_m_s=0.002,
      table_target_support_n=float(
        s.model.body_mass[s.eraser_body] * -s.model.opt.gravity[2]
      ),
      release_force_ramp_s=1.5,
      command_interpolation_period_s=C.CONTROL_PERIOD_S,
      ccd_tolerance=float(s.model.opt.ccd_tolerance),
      ccd_iterations=int(s.model.opt.ccd_iterations),
      eraser_contact_margins_m={
        "handle": float(s.model.geom_margin[s.handle_id]),
        "felt": float(s.model.geom_margin[s.pad_id]),
        "board_pair": float(
          s.model.pair_margin[s.model.pair("eraser_board_contact").id]
        ),
      },
      peak_fingertip_force_n=self.peak_tactile.tolist(),
      phases=self.phases,
      solver_warnings=[int(w.number) for w in s.data.warning],
    )
