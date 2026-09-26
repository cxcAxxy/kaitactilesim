from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.execution import BulbScrewExecutor
from kaihand_tactile_env.tasks.bulb_screw.grasp import calibrated_grasp
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation


@pytest.fixture(scope="module")
def simulation():
  sim = BulbScrewSimulation()
  yield sim
  del sim
  gc.collect()


@pytest.mark.parametrize("five_finger", [False, True])
def test_grasp_calibration_does_not_move_robot_or_bulb(simulation, five_finger):
  simulation.reset()
  qpos, qvel = simulation.full_state()
  goals = simulation.arm_goal["right"].copy()
  grasp = calibrated_grasp(simulation, five_finger=five_finger)
  np.testing.assert_array_equal(simulation.data.qpos, qpos)
  np.testing.assert_array_equal(simulation.data.qvel, qvel)
  np.testing.assert_array_equal(simulation.arm_goal["right"], goals)
  assert grasp.open_hand.shape == grasp.closed_hand.shape == (20,)
  assert grasp.wrist_position[2] > simulation.object_pose("bulb")[2] + 0.15
  np.testing.assert_allclose(
    grasp.wrist_rotation.T @ grasp.wrist_rotation, np.eye(3), atol=1e-12
  )


@pytest.mark.parametrize(
  "grasp_mode,speed,time_limit",
  [("pinch", "normal", 150), ("pinch", "fast", 110), ("five-finger", "fast", 60)],
)
def test_full_robot_episode_without_object_teleports_or_external_wrenches(
  simulation, monkeypatch, grasp_mode, speed, time_limit
):
  sim = simulation
  sim.reset()
  initial = sim.object_pose("bulb")[:3].copy()

  def forbidden(*args, **kwargs):
    raise AssertionError("automatic task attempted an object pose/reset shortcut")

  monkeypatch.setattr(sim, "set_object_pose", forbidden)
  monkeypatch.setattr(sim, "initialize_threaded", forbidden)
  grip_seen = False
  released_strokes = 0
  previous_phase = None
  previous_position = initial.copy()
  capture_seen = False
  maximum_pitch_error = 0.0
  data = sim.data
  left_robot_bodies = {
    i
    for i in range(sim.model.nbody)
    if sim.model.body(i).name.startswith(("left_arm", "left_hand", "hand_l"))
  }
  right_robot_bodies = {
    i
    for i in range(sim.model.nbody)
    if sim.model.body(i).name.startswith(("right_arm", "right_hand", "hand_r"))
  }
  executor = None
  entry_contact_load = 0.0
  reset_started_unloaded = False
  lit_tighten_times = []
  turn_samples = []
  transport_samples = []
  aligned_above_socket = False
  phase_started = 0.0
  reset_group = None
  read_thread_contact = sim.thread_contact_load

  def observe_entry_contact():
    nonlocal entry_contact_load
    load = read_thread_contact()
    if not sim.thread_engaged:
      entry_contact_load = load
    return load

  monkeypatch.setattr(sim, "thread_contact_load", observe_entry_contact)

  def observe(simulation, phase):
    nonlocal grip_seen, released_strokes, previous_phase, previous_position
    nonlocal capture_seen, maximum_pitch_error
    nonlocal phase_started
    nonlocal aligned_above_socket
    if phase != previous_phase:
      phase_started = float(data.time)
    assert not np.any(data.xfrc_applied)
    # Only the native thread weld may become active; no hand/bulb stabilizer.
    assert simulation.model.neq == 4  # Two shared thumb equalities, pitch, capture.
    position = simulation.object_pose("bulb")[:3]
    assert np.linalg.norm(position - previous_position) < 0.002
    previous_position = position
    state = executor._state
    if grasp_mode == "five-finger":
      if phase in {"clear_for_home", "return_home", "verify_seated"}:
        for contact in data.contact:
          a = int(sim.model.geom_bodyid[contact.geom1])
          b = int(sim.model.geom_bodyid[contact.geom2])
          assert not (
            (a in left_robot_bodies and b in right_robot_bodies)
            or (a in right_robot_bodies and b in left_robot_bodies)
          ), "right hand or arm contacted the left side during return home"
      if phase == "transfer" or (
        phase == "align_thread"
        and position[2] > config.SOCKET_MOUTH_POSITION_M[2] + 0.003
      ):
        bulb_body = sim.model.body("bulb").id
        socket_body = sim.model.body("bulb_socket").id
        for contact in data.contact:
          bodies = {
            int(sim.model.geom_bodyid[contact.geom1]),
            int(sim.model.geom_bodyid[contact.geom2]),
          }
          assert not {bulb_body, socket_body} <= bodies
      if phase == "transfer":
        transport_samples.append((executor._grip.copy(), executor._grip_tangent.copy()))
        if np.linalg.norm(position[:2] - config.SOCKET_MOUTH_POSITION_M[:2]) < 0.03:
          assert (
            position[2]
            >= config.SOCKET_MOUTH_POSITION_M[2]
            + config.SOCKET_APPROACH_CLEARANCE_M
            - 0.002
          )
      if phase == "align_thread" and previous_phase != phase:
        assert np.linalg.norm(position[:2] - config.SOCKET_MOUTH_POSITION_M[:2]) < 0.001
        assert position[2] == pytest.approx(
          config.SOCKET_MOUTH_POSITION_M[2] + config.SOCKET_APPROACH_CLEARANCE_M,
          abs=0.002,
        )
        aligned_above_socket = True
      if phase == "turn":
        turn_samples.append(
          (
            float(data.time),
            phase_started,
            state.clockwise_turns,
            executor._grip.copy(),
            executor._grip_tangent.copy(),
          )
        )
      if (
        phase == "reset_fingers"
        and reset_group == (0, 4)
        and 0.15 * config.TARGET_TURNS
        < state.clockwise_turns
        < 0.8 * config.TARGET_TURNS
      ):
        # The middle three support fingers must regulate load while thumb and
        # pinky reset; holding their old angles caused a roughly 9 N ring spike.
        assert np.all(executor._grip[1:4] > 0.75)
        assert np.all(executor._grip[1:4] < 4.5)
    if simulation.bulb_lit:
      assert executor._tightening_verified
      if phase in {"tighten", "hold_tight"}:
        assert state.seated
        if not lit_tighten_times:
          assert min(executor._grip) >= 4.0
          assert executor._hand_torque_nm >= config.TIGHTENING_MIN_TORQUE_NM
        # The legacy wrist grasp can unload its torque at fixed targets;
        # the default finger-driven grasp must keep applying tightening effort.
        assert min(executor._grip) > 0.025
        if grasp_mode == "five-finger":
          assert min(executor._grip) >= 4.0
          assert executor._hand_torque_nm >= config.TIGHTENING_MIN_TORQUE_NM
        lit_tighten_times.append(float(data.time))
    if phase == "release":
      assert lit_tighten_times
    elif phase in {"hover", "approach", "grasp", "turn"}:
      assert not state.bulb_lit
    if executor._finger_drive_active:
      np.testing.assert_array_equal(
        simulation.arm_goal["right"], executor._fixed_arm_goal
      )
    if state.engaged and not capture_seen:
      assert state.insertion_depth_m > 0.0055
      # Capture is checked before integration. The first post-capture physics
      # step can unload a crest; inspect the force used by the actual gate.
      assert entry_contact_load > config.THREAD_CAPTURE_LOAD_N
      capture_seen = True
    if phase in {"turn", "regrasp", "open_for_regrasp", "reset_fingers"}:
      maximum_pitch_error = max(maximum_pitch_error, abs(state.thread_error_m))
    if phase == "lift" and min(executor._grip) > 0.1:
      grip_seen = True
    if phase == "reset_fingers" and phase != previous_phase:
      # Check immediately before commanding reset. A returning pad can already
      # touch again in the first post-integration observation.
      assert reset_started_unloaded
      released_strokes += 1
    if phase == "recover_wrist" and phase != previous_phase:
      assert max(executor._grip) < 0.1
      released_strokes += 1
    previous_phase = phase

  executor = BulbScrewExecutor(
    sim, observer=observe, grasp_mode=grasp_mode, speed=speed
  )
  hand_motion = executor._hand_motion

  def observe_reset(target, duration, phase, **checks):
    nonlocal reset_started_unloaded, reset_group
    if phase == "reset_fingers":
      reset_group = checks.get("reset_group")
      reset_started_unloaded = bool(
        min(executor._grip) < 0.1 and max(executor._grip) > 0.1
      )
    return hand_motion(target, duration, phase, **checks)

  monkeypatch.setattr(executor, "_hand_motion", observe_reset)
  result = executor.execute()
  assert result.success, result.reason
  if grasp_mode == "five-finger":
    assert aligned_above_socket
  assert result.tightening_verified
  assert sim.bulb_lit and result.state.bulb_lit
  assert (
    lit_tighten_times[-1] - lit_tighten_times[0]
    >= config.TIGHTENING_LIGHT_HOLD_S - sim.timestep - 1e-8
  )
  assert result.tightening_peak_torque_nm >= config.TIGHTENING_MIN_TORQUE_NM
  assert result.tightening_stall_duration_s >= config.TIGHTENING_HOLD_S - 1e-8
  assert result.maximum_lift_m > 0.09
  assert capture_seen and maximum_pitch_error < config.MAX_THREAD_ERROR_M
  assert result.state.axial_travel_m == pytest.approx(
    config.THREAD_TRAVEL_M, abs=0.0003
  )
  assert grip_seen
  if grasp_mode == "five-finger":
    carried = np.asarray(transport_samples)
    assert len(carried) > 1000
    for column, native_limit, recorded_limit in ((0, 0.03, 0.06), (1, 0.04, 0.08)):
      forces = carried[:, column]
      # Check both physics-rate contact forces and the 100 Hz curve cadence;
      # neither check filters away chatter at the transport/alignment transition.
      for stride, limit in ((1, native_limit), (5, recorded_limit)):
        jumps = np.diff(forces[::stride], axis=0)
        assert np.all(np.sqrt(np.mean(jumps**2, axis=0)) < limit)
    times, starts, turns = np.array([s[:3] for s in turn_samples]).T
    core = (times - starts > 0.2) & (turns > 0.075) & (turns < 0.4)
    adjacent = core[1:] & core[:-1] & (starts[1:] == starts[:-1])
    assert adjacent.sum() > 500
    for column in (3, 4):
      forces = np.array([s[column] for s in turn_samples])
      jumps = np.diff(forces, axis=0)[adjacent]
      # Native 500 Hz contact forces, with no smoothing or resampling. Normal
      # forces with the old 20 ms step commands had 0.38–0.86 N RMS jumps.
      assert np.all(np.sqrt(np.mean(jumps**2, axis=0)) < 0.2)
    assert released_strokes >= 7
    assert result.strokes == 4
    assert result.rotation_driver == config.ROTATION_DRIVER
    assert result.maximum_wrist_rotation_deg < 1.0
    assert result.maximum_wrist_displacement_m < 0.004
  else:
    assert released_strokes == 2
    assert result.strokes == 3
    assert result.rotation_driver == "legacy_pinch_wrist"
  assert result.elapsed_s < time_limit
  assert result.grasp_mode == grasp_mode and result.speed == speed
  assert sum(result.phase_durations_s.values()) == pytest.approx(result.elapsed_s)
  count = 5 if grasp_mode == "five-finger" else 2
  assert len(result.peak_fingertip_load_n) == count
  assert min(result.peak_fingertip_load_n) > 0.1
  assert result.simultaneous_turn_contact_fraction > 0.98
  assert min(result.turn_contact_fraction) > 0.98
  assert result.state.success and result.state.backstop_load_n > 0.01
  assert result.state.exposed_thread_m == 0
  assert abs(result.state.shoulder_gap_m) < config.SEATED_SHOULDER_GAP_M
  assert result.state.shoulder_contact_load_n > 0.01
  assert result.state.clockwise_turns == pytest.approx(config.TARGET_TURNS, abs=0.08)
  assert result.state.insertion_depth_m == pytest.approx(
    config.SEATED_DEPTH_M, abs=0.0003
  )
  assert max(result.final_fingertip_load_n) < 0.01
  assert sim.arm_goal_error("right") < 0.02
  assert result.phases[-1] == "verify_seated"
  with pytest.raises(RuntimeError, match="fresh executor"):
    executor.execute()


def test_python_default_is_finger_driven_five_finger(simulation):
  executor = BulbScrewExecutor(simulation)
  assert executor.finger_count == 5 and executor.speed == "fast"


def test_tightening_rejects_unloading_between_control_ticks(simulation, monkeypatch):
  """A 2 ms loss of load must restart the full 0.3 s confirmation window."""
  from dataclasses import replace

  simulation.reset()
  executor = BulbScrewExecutor(simulation)
  executor._tightening_rotation = np.eye(3)
  executor._state = replace(executor._state, seated=True)
  executor._hand_torque_nm = 0.2
  monkeypatch.setattr(executor, "_object_pose", lambda: (np.zeros(3), np.eye(3)))
  confirmations = []
  monkeypatch.setattr(
    simulation, "confirm_tightening", lambda: confirmations.append(simulation.data.time)
  )
  for tick in range(231):
    simulation.data.time = tick * simulation.timestep
    executor._grip[:] = 6.0
    if tick == 79:  # 158 ms: missed by a 20 ms verification loop.
      executor._grip[3] = 3.9
    executor._check_tightening_stall()
    if tick < 230:
      assert not executor._tightening_verified
      assert not confirmations
  assert executor._tightening_verified
  assert confirmations == pytest.approx([0.46])


def test_cancellation_returns_failure_without_advancing_further(simulation):
  simulation.reset()
  calls = 0

  def stop():
    nonlocal calls
    calls += 1
    return calls > 20

  result = BulbScrewExecutor(simulation, should_stop=stop).execute()
  assert not result.success and result.reason == "cancelled"
  assert result.elapsed_s == pytest.approx(20 * simulation.timestep)
  assert result.strokes == 0


def test_example_requires_a_full_loaded_stall_in_actual_bulb_pose():
  """The 100 Hz export includes both endpoints of the required 0.3 s window."""
  from types import SimpleNamespace

  import h5py
  from kaihand_tactile_env.tasks.bulb_screw.example import _audit

  t = np.arange(121) / 100
  tight = (t >= 0.7) & (t <= 1.0)
  phase = np.where(
    tight, "tighten", np.where(t < 0.1, "align_thread", np.where(t < 0.7, "turn", "release"))
  )
  fn = np.where(tight, 6.0, np.where(t < 0.7, 3.0, 0.0))
  ft = np.where(tight, 2.0, np.where(t < 0.7, 1.0, 0.0))
  result = SimpleNamespace(success=True, tightening_verified=True, elapsed_s=1.2)
  with h5py.File("bulb-review-memory", "w", driver="core", backing_store=False) as f:
    force = f.create_group("tactile_contact_force")
    force["timestamp"] = t
    normal = np.broadcast_to(fn[:, None, None, None] / 35, (len(t), 5, 7, 5))
    tangent = np.zeros((len(t), 5, 7, 5, 2))
    tangent[..., 0] = ft[:, None, None, None] / 35
    force["normal_taxel_force_n"] = normal
    force["tangent_taxel_force_n"] = tangent
    force["normal_force_n"] = normal.sum(axis=(-2, -1))
    force["tangent_force_n"] = tangent.sum(axis=(-3, -2))
    f.create_group("commands").create_dataset(
      "phase", data=phase.astype(object), dtype=h5py.string_dtype()
    )
    state = f.create_group("state")
    state["qpos"] = np.zeros((len(t), 1))
    state["qvel"] = np.zeros((len(t), 1))
    task = f.create_group("bulb_screw")
    task["clockwise_turns"] = np.where(t < 0.7, 0.5, 1.0) * config.TARGET_TURNS
    task["engaged"] = t >= 0.1
    task["hand_clockwise_torque_nm"] = np.where(tight, 0.3, 0)
    task["xfrc_applied"] = np.zeros((len(t), 1, 6))
    task["seated"] = tight
    pose = np.zeros((len(t), 7))
    pose[:, :2] = config.SOCKET_MOUTH_POSITION_M[:2]
    pose[:, 2] = config.SOCKET_FUNNEL_TOP_Z_M - 0.002
    pose[:, 3] = 1
    bulb = f.create_group("objects").create_group("bulb")
    bulb["pose_wxyz"] = pose
    metrics = _audit(f, result)
    np.testing.assert_allclose(metrics["loaded_stall_window_s"], [0.7, 1.0])
    pose[phase == "align_thread", 0] += 0.001
    bulb["pose_wxyz"][:] = pose
    with pytest.raises(ValueError, match="laterally misaligned"):
      _audit(f, result)
    pose[phase == "align_thread", 0] -= 0.001
    bulb["pose_wxyz"][:] = pose
    # A stationary guide alone cannot pass if the actual bulb is still turning.
    angle = np.linspace(0, np.deg2rad(1), tight.sum())
    pose[tight, 3], pose[tight, 6] = np.cos(angle / 2), np.sin(angle / 2)
    bulb["pose_wxyz"][:] = pose
    with pytest.raises(ValueError, match="loaded angular stall"):
      _audit(f, result)
    pose[:, 3], pose[:, 6] = 1, 0
    bulb["pose_wxyz"][:] = pose
    task["hand_clockwise_torque_nm"][:] = 0
    with pytest.raises(ValueError, match="loaded angular stall"):
      _audit(f, result)


def test_example_rejects_contact_between_robot_sides():
  import h5py
  from kaihand_tactile_env.tasks.bulb_screw.example import _verify_no_interarm_contact

  with h5py.File("bulb-contact-memory", "w", driver="core", backing_store=False) as f:
    f.create_group("model").create_dataset(
      "body_names",
      data=["world", "hand_l_index_link4", "hand_r_index_link4"],
      dtype=h5py.string_dtype(),
    )
    events = f.create_group("contacts").create_group("events")
    events["body1_id"] = [1]
    events["body2_id"] = [2]
    with pytest.raises(ValueError, match="inter-arm contact"):
      _verify_no_interarm_contact(f)
    events["body2_id"][:] = 0
    _verify_no_interarm_contact(f)


@pytest.mark.parametrize("fault", [None, "arm", "wrist", "finger"])
def test_recorded_finger_motion_requires_fixed_arm_and_all_fingers(fault):
  import h5py
  from kaihand_tactile_env.tasks.bulb_screw.example import _finger_motion

  with h5py.File("finger-motion-memory", "w", driver="core", backing_store=False) as f:
    task = f.create_group("bulb_screw")
    task["finger_drive_active"] = [True, True, True]
    goals = np.zeros((3, 7))
    positions = np.zeros((3, 3))
    joints = np.arange(3)[:, None] * np.ones((3, 20)) * 0.1
    if fault == "arm":
      goals[-1, 0] = 0.01
    if fault == "wrist":
      positions[-1, 0] = 0.01
    if fault == "finger":
      joints[:, -4:] = 0
    task["right_arm_joint_goal"] = goals
    task["right_wrist_position_m"] = positions
    task["right_wrist_rotation"] = np.broadcast_to(np.eye(3), (3, 3, 3))
    task["right_finger_joint_position"] = joints
    if fault is None:
      metrics = _finger_motion(f)
      assert metrics["arm_goal_max_change_rad"] == 0
      assert min(metrics["finger_joint_range_deg"]) > 5
    else:
      with pytest.raises(ValueError):
        _finger_motion(f)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_stroke_limit(simulation, limit):
  with pytest.raises(ValueError, match="positive integer"):
    BulbScrewExecutor(simulation, max_strokes=limit)


@pytest.mark.parametrize("kwargs", [{"speed": "turbo"}, {"grasp_mode": "three"}])
def test_invalid_motion_mode(simulation, kwargs):
  with pytest.raises(ValueError):
    BulbScrewExecutor(simulation, **kwargs)


def test_cli_defaults_to_fast_five_finger_and_retains_legacy_mode():
  args = _cli().parse_args(["--run-task"])
  assert args.speed == "fast" and args.grasp == "five-finger"
  args = _cli().parse_args(["--run-task", "--grasp", "pinch", "--speed", "normal"])
  assert args.speed == "normal" and args.grasp == "pinch"


def _cli():
  path = Path(__file__).resolve().parents[1] / "scripts/workcell/view_bulb_screw.py"
  spec = importlib.util.spec_from_file_location("bulb_cli", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.mark.parametrize(
  "arguments",
  [
    ["--run-task", "--threaded"],
    ["--run-task", "--mechanics-demo"],
    ["--video", "unused.mp4"],
    ["--video-fps", "nan"],
    ["--run-task", "--video", "same.mp4", "--result-json", "same.json"],
  ],
)
def test_cli_rejects_incompatible_modes_and_outputs(arguments):
  with pytest.raises(SystemExit) as error:
    _cli().parse_args(arguments)
  assert error.value.code == 2


def test_tightening_recovery_is_bounded_and_cannot_replace_confirmation(
  simulation, monkeypatch
):
  simulation.reset()
  executor = BulbScrewExecutor(simulation)
  executor._command_position, executor._command_rotation = (
    simulation.current_pose_matrix("right")
  )
  steps = []
  monkeypatch.setattr(executor, "_finger_regrasp", lambda: None)
  monkeypatch.setattr(executor, "_advance", lambda *args, **kwargs: None)
  monkeypatch.setattr(
    executor, "_finger_step", lambda angle_step=0.0, **kwargs: steps.append(angle_step)
  )
  with pytest.raises(RuntimeError, match="loaded angular stall"):
    executor._tighten()
  recovery = steps[round(config.TIGHTENING_DURATION_S / 0.02) :]
  assert len(recovery) == 100
  assert recovery == [0.0] * 100  # Recover grip without commanding more rotation.
  assert not executor._tightening_verified
  assert not simulation.bulb_lit
