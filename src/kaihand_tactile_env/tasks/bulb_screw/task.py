"""Contact-gated guide for colliding rounded internal/external helical crests.

The free bulb must enter the socket and load a thread crest before a passive
helical guide is enabled. The guide constrains ideal alignment/pitch; it is
not an electrically or mechanically calibrated E27 thread model.
"""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from kaihand_tactile_env.shared.pickup_randomization import PickupPositionRandomization
from kaihand_tactile_env.shared.simulation import ArmHandSimulation

from . import config


@dataclass(frozen=True)
class BulbScrewState:
  timestamp: float
  engaged: bool
  clockwise_turns: float
  insertion_depth_m: float
  lateral_error_m: float
  tilt_rad: float
  thread_error_m: float
  linear_speed_m_s: float
  angular_speed_rad_s: float
  backstop_load_n: float
  seated: bool
  stable_duration_s: float = 0.0
  success: bool = False
  axial_travel_m: float = 0.0
  thread_contact_load_n: float = 0.0
  exposed_thread_m: float = 0.0
  shoulder_gap_m: float = 0.0
  shoulder_contact_load_n: float = 0.0
  cushion_contact_load_n: float = 0.0
  bulb_lit: bool = False


class BulbScrewSimulation(ArmHandSimulation):
  """Free tabletop bulb with automatically gated, reversible thread capture."""

  def __init__(self, *, position_seed=None, **kwargs):
    self.position_seed = position_seed
    self._pickup_randomization = None
    super().__init__(scene=config.SCENE_NAME, **kwargs)

  def reset(self, **kwargs) -> None:
    super().reset(**kwargs)
    if self._pickup_randomization is None:
      self._pickup_randomization = PickupPositionRandomization(self, "bulb")
    self.initial_position_randomization = self._pickup_randomization.apply(
      self, self.position_seed
    )
    self._weld = self.model.equality("bulb_thread_capture").id
    self._slide = int(self.model.joint("bulb_thread_depth").qposadr[0])
    self._hinge = int(self.model.joint("bulb_thread_angle").qposadr[0])
    self._slide_dof = int(self.model.joint("bulb_thread_depth").dofadr[0])
    self._hinge_dof = int(self.model.joint("bulb_thread_angle").dofadr[0])
    self._bulb_id = self.model.body("bulb").id
    self._mouth_id = self.model.site("bulb_socket_mouth").id
    self._external_thread_geoms = {
      self.model.geom(f"bulb_thread_crest_{i:03d}").id
      for i in range(config.EXTERNAL_THREAD_SEGMENTS)
    }
    self._internal_thread_geoms = {
      self.model.geom(f"bulb_socket_thread_{i:03d}").id
      for i in range(config.INTERNAL_THREAD_SEGMENTS)
    }
    self._socket_walls = {
      self.model.geom(f"bulb_socket_wall_{i:02d}").id for i in range(16)
    }
    self._socket_cushions = {
      self.model.geom(f"bulb_socket_cushion_{i:02d}").id for i in range(16)
    }
    self._capture_armed = True
    self._set_bulb_lit(False)
    self.model.eq_data[self._weld, 3:10] = (0, 0, 0, 1, 0, 0, 0)
    self.data.eq_active[self._weld] = False
    mujoco.mj_forward(self.model, self.data)

  @property
  def thread_engaged(self) -> bool:
    return bool(self.data.eq_active[self._weld])

  @property
  def bulb_lit(self) -> bool:
    return bool(self.model.light_active[self.model.light("bulb_glow").id])

  def _set_bulb_lit(self, lit: bool) -> None:
    """Switch visual material and local illumination, without changing physics."""
    material = "bulb_glowing" if lit else "bulb_frosted"
    self.model.geom_matid[self.model.geom("bulb_globe").id] = self.model.material(
      material
    ).id
    self.model.light_active[self.model.light("bulb_glow").id] = lit

  def confirm_tightening(self) -> None:
    """Latch the completion light after the controller verifies loaded stall."""
    if not BulbScrewMonitor(self).measure().seated:
      raise ValueError("cannot light a bulb that is not seated")
    self._set_bulb_lit(True)

  def initialize_threaded(self) -> None:
    """Explicit reset preset: insert the lead-in and touch the first crest."""
    self.reset()
    self.set_object_pose("bulb", config.THREAD_ENTRY_POSITION_M)
    if not self.try_engage_thread():
      raise RuntimeError("thread preset failed to align with the socket")

  def _alignment(self):
    position = self.data.xpos[self._bulb_id] - self.data.site_xpos[self._mouth_id]
    rotation = self.data.xmat[self._bulb_id].reshape(3, 3)
    tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    return -float(position[2]), float(np.linalg.norm(position[:2])), tilt

  def thread_contact_load(self) -> float:
    """Sum actual internal/external crest contact loads, excluding the guide."""
    load = 0.0
    wrench = np.zeros(6)
    for i, contact in enumerate(self.data.contact):
      a, b = int(contact.geom1), int(contact.geom2)
      if (a in self._external_thread_geoms and b in self._internal_thread_geoms) or (
        b in self._external_thread_geoms and a in self._internal_thread_geoms
      ):
        mujoco.mj_contactForce(self.model, self.data, i, wrench)
        load += max(0.0, float(wrench[0]))
    return load

  def try_engage_thread(self) -> bool:
    """Capture only after insertion and physical thread contact.

    Never overwrite the free bulb's pose/velocity.

    The solver corrects the bounded residual alignment error. Initial yaw is
    retained in the weld so turn zero is relative to each capture event.
    """
    if self.thread_engaged:
      return True
    mujoco.mj_forward(self.model, self.data)
    depth, lateral, tilt = self._alignment()
    if not (
      self._capture_armed
      and abs(depth - config.THREAD_ENTRY_DEPTH_M) <= config.CAPTURE_DEPTH_M
      and self.thread_contact_load() > config.THREAD_CAPTURE_LOAD_N
      and lateral <= config.CAPTURE_LATERAL_M
      and tilt <= config.CAPTURE_TILT_RAD
      and np.linalg.norm(self.object_twist("bulb")[:3]) <= config.CAPTURE_SPEED_M_S
    ):
      return False
    rotation = self.data.xmat[self._bulb_id].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    self.model.eq_data[self._weld, 3:10] = (
      0,
      0,
      0,
      np.cos(yaw / 2),
      0,
      0,
      np.sin(yaw / 2),
    )
    self.data.qpos[[self._slide, self._hinge]] = 0
    self.data.qvel[[self._slide_dof, self._hinge_dof]] = 0
    self.data.eq_active[self._weld] = True
    mujoco.mj_forward(self.model, self.data)
    return True

  def _before_physics_step(self) -> None:
    # Refresh geometry because mj_step's caches precede its final integration.
    mujoco.mj_forward(self.model, self.data)
    if self.thread_engaged:
      if self.data.qpos[self._hinge] < -0.08:
        self.data.eq_active[self._weld] = False
        self._capture_armed = False
    else:
      depth, _, _ = self._alignment()
      if depth < -0.002:
        self._capture_armed = True
      self.try_engage_thread()
    if self.bulb_lit:
      depth, _, _ = self._alignment()
      if not self.thread_engaged or depth < (
        config.SEATED_DEPTH_M - config.SEATED_DEPTH_TOLERANCE_M
      ):
        self._set_bulb_lit(False)


class BulbScrewMonitor:
  """Require buried threads, shoulder/stop contact and a stable seated dwell."""

  def __init__(self, simulation: BulbScrewSimulation):
    self.simulation = simulation
    self.reset()

  def reset(self) -> None:
    self._last_time = None
    self._seated_since = None

  def measure(self) -> BulbScrewState:
    sim = self.simulation
    model, data = sim.model, sim.data
    mujoco.mj_forward(model, data)
    depth, lateral, tilt = sim._alignment()
    turns = float(data.qpos[sim._hinge] / (2 * np.pi)) if sim.thread_engaged else 0.0
    thread_error = depth - config.THREAD_ENTRY_DEPTH_M - config.THREAD_PITCH_M * turns
    twist = sim.object_twist("bulb")
    linear = float(np.linalg.norm(twist[:3]))
    angular = float(np.linalg.norm(twist[3:]))
    stop = model.geom("bulb_socket_backstop").id
    shoulder = model.geom("bulb_neck").id
    mouth_z = data.site_xpos[sim._mouth_id, 2]
    # A capsule's endpoint extent along world Z plus its radius. The highest
    # external crest must be below the physical rim, not just near turn one.
    crests = np.array(sorted(sim._external_thread_geoms))
    crest_z = (
      data.geom_xpos[crests, 2]
      + np.abs(data.geom_xmat[crests, 8]) * model.geom_size[crests, 1]
      + model.geom_size[crests, 0]
    )
    exposed_thread = max(0.0, float(np.max(crest_z) - mouth_z))
    # Highest edge of the shoulder's bottom face also detects tilted seating.
    axis_z = data.geom_xmat[shoulder, 8]
    shoulder_gap = float(
      data.geom_xpos[shoulder, 2]
      - model.geom_size[shoulder, 1] * axis_z
      + model.geom_size[shoulder, 0] * np.sqrt(max(0.0, 1 - axis_z**2))
      - mouth_z
    )
    load = 0.0
    shoulder_load = 0.0
    cushion_load = 0.0
    wrench = np.zeros(6)
    for i in range(data.ncon):
      contact = data.contact[i]
      ids = (int(contact.geom1), int(contact.geom2))
      if stop in ids and any(model.geom_bodyid[g] == sim._bulb_id for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        load += max(0.0, float(wrench[0]))
      if shoulder in ids and any(g in sim._socket_walls for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        shoulder_load += max(0.0, float(wrench[0]))
      if shoulder in ids and any(g in sim._socket_cushions for g in ids):
        mujoco.mj_contactForce(model, data, i, wrench)
        cushion_load += max(0.0, float(wrench[0]))
    seated = bool(
      sim.thread_engaged
      and abs(depth - config.SEATED_DEPTH_M) <= config.SEATED_DEPTH_TOLERANCE_M
      and abs(turns - config.TARGET_TURNS) <= 0.08
      and abs(thread_error) <= config.SEATED_DEPTH_TOLERANCE_M
      and lateral <= config.CAPTURE_LATERAL_M
      and tilt <= config.CAPTURE_TILT_RAD
      and linear <= 0.001
      and angular <= 0.05
      and load >= 0.01
      and exposed_thread <= config.SEATED_SHOULDER_GAP_M
      and abs(shoulder_gap) <= config.SEATED_SHOULDER_GAP_M
      and shoulder_load >= 0.01
    )
    return BulbScrewState(
      float(data.time),
      sim.thread_engaged,
      turns,
      depth,
      lateral,
      tilt,
      thread_error,
      linear,
      angular,
      load,
      seated,
      axial_travel_m=depth - config.THREAD_ENTRY_DEPTH_M,
      thread_contact_load_n=sim.thread_contact_load(),
      exposed_thread_m=exposed_thread,
      shoulder_gap_m=shoulder_gap,
      shoulder_contact_load_n=shoulder_load,
      cushion_contact_load_n=cushion_load,
      bulb_lit=sim.bulb_lit,
    )

  def update(self) -> BulbScrewState:
    state = self.measure()
    if self._last_time is not None:
      elapsed = state.timestamp - self._last_time
      if elapsed < 0 or elapsed > 1.5 * self.simulation.timestep:
        self._seated_since = None
    self._last_time = state.timestamp
    if not state.seated:
      self._seated_since = None
    elif self._seated_since is None:
      self._seated_since = state.timestamp
    duration = (
      0.0 if self._seated_since is None else state.timestamp - self._seated_since
    )
    return replace(
      state,
      stable_duration_s=duration,
      success=duration >= config.SEATED_DWELL_S - 1e-12,
    )
