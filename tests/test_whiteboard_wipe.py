"""Independent model, physical erasure gates, and complete pickup/release."""

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES, default_model_path
from kaihand_tactile_env.tasks.whiteboard_wipe import config as C
from kaihand_tactile_env.tasks.whiteboard_wipe.execution import WhiteboardWipeExecutor
from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation


def test_scene_inherits_robot_cameras_and_has_45_degree_board():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  reference = mujoco.MjModel.from_xml_path(str(default_model_path("pick-place")))
  for name in SHARED_CAMERA_NAMES:
    for field in ("pos", "quat", "fovy"):
      np.testing.assert_array_equal(
        getattr(sim.model.camera(name), field), getattr(reference.camera(name), field)
      )
  rotation = sim.data.xmat[sim.model.body("whiteboard").id].reshape(3, 3)
  np.testing.assert_allclose(rotation, C.BOARD_ROTATION, atol=1e-12)
  assert np.arccos(rotation[2, 2]) == pytest.approx(np.pi / 4)
  assert sim.object_names == ("eraser",)
  assert (
    sim.model.neq == reference.neq - 1
  )  # Only shared finger couplings, no tool weld.
  assert all(
    sim.model.body(name).id >= 0 for name in ("hand_r_base_link", "hand_l_base_link")
  )
  sim.step(400)
  assert sim.table_force > 0.5
  assert sim.forces.read(sim.data).normal_force_n.sum() == 0
  np.testing.assert_array_equal(sim.remaining, 1)


def test_eraser_is_a_flat_white_shell_with_a_black_felt_base():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  model = sim.model
  bounds = []
  for name in ("eraser_handle", "eraser_pad"):
    geom = model.geom(name)
    assert model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_MESH
    mesh_id = model.geom_dataid[geom.id]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    assert count > 8  # Rounded surfaces, beyond the eight corners of a box.
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom.id])
    vertices = (
      model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
      + model.geom_pos[geom.id]
    )
    bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
  lower = np.minimum(bounds[0][0], bounds[1][0])
  upper = np.maximum(bounds[0][1], bounds[1][1])
  np.testing.assert_allclose(upper - lower, [0.115, 0.056, 0.030], atol=1e-8)
  assert bounds[1][1][2] == pytest.approx(bounds[0][0][2], abs=1e-8)
  assert bounds[1][0][2] < bounds[0][0][2]

  shell_material = model.geom_matid[sim.handle_id]
  assert np.all(model.mat_rgba[shell_material, :3] > 0.8)
  felt_material = model.material("eraser_black_felt").id
  assert model.geom_matid[sim.pad_id] == felt_material
  texture = model.texture("eraser_felt_texture").id
  assert texture in model.mat_texid[felt_material]
  start = model.tex_adr[texture]
  count = (
    model.tex_width[texture] * model.tex_height[texture] * model.tex_nchannel[texture]
  )
  pixels = model.tex_data[start : start + count]
  assert 0 < pixels.min() < pixels.max() < 26  # Dark, nonuniform felt texture.


def test_pad_board_pair_sets_its_own_friction():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  pair = sim.model.pair("eraser_board_contact").id
  assert {sim.model.pair_geom1[pair], sim.model.pair_geom2[pair]} == {
    sim.pad_id,
    sim.board_id,
  }
  assert sim.model.pair_dim[pair] == 6
  np.testing.assert_allclose(
    sim.model.pair_friction[pair], [0.65, 0.65, 0.003, 0.0003, 0.0003]
  )


def test_cleaning_requires_loaded_pad_and_actual_sliding():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  # Stationary loading must not erase ink, nor may the moving free eraser.
  center = C.BOARD_SURFACE - C.BOARD_ROTATION @ C.PAD_BOTTOM
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, C.BOARD_ROTATION.ravel())
  sim.set_object_pose("eraser", center - 0.0005 * C.BOARD_NORMAL, quat)
  mujoco.mj_forward(sim.model, sim.data)
  assert sim.data.ncon > 0
  sim._update_cleaning()
  assert sim.board_force > 0
  np.testing.assert_array_equal(sim.patch_speed, 0)
  np.testing.assert_array_equal(sim.remaining, 1)
  np.testing.assert_array_equal(sim.cleaning.work_j, 0)
  # Actual solver friction and slip at insufficient pressure must not erase ink.
  sim.data.qvel[sim._object_dofs["eraser"] + 1] = 0.05
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert 0 < sim.board_force < 1.5
  assert sim.board_tangent_force > 0
  assert sim.patch_speed.max() == pytest.approx(0.05)
  np.testing.assert_array_equal(sim.remaining, 1)
  np.testing.assert_array_equal(sim.cleaning.work_j, 0)
  sim.set_object_pose("eraser", center + 0.03 * C.BOARD_NORMAL, quat)
  sim.data.qvel[sim._object_dofs["eraser"] + 1] = 0.08
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert sim.board_force == 0
  np.testing.assert_array_equal(sim.remaining, 1)
  sim.remaining[:] = 0
  sim.model.geom_rgba[sim.ink_ids, 3] = 0
  sim.reset()
  np.testing.assert_array_equal(sim.remaining, 1)
  np.testing.assert_array_equal(sim.model.geom_rgba[sim.ink_ids, 3], 1)
  assert sim.data.time == 0


@pytest.mark.parametrize("friction", [0.0, 0.65])
def test_loaded_sliding_requires_actual_friction_work(friction):
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  center = C.BOARD_SURFACE - C.BOARD_ROTATION @ C.PAD_BOTTOM
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, C.BOARD_ROTATION.ravel())
  sim.set_object_pose("eraser", center - 0.0005 * C.BOARD_NORMAL, quat)
  pair = sim.model.pair("eraser_board_contact").id
  sim.model.pair_friction[pair, :2] = friction
  # Apply a real 3 N load to the free tool; MuJoCo computes the contact wrench.
  sim.data.xfrc_applied[sim.eraser_body, :3] = -3.0 * C.BOARD_NORMAL
  sim.data.qvel[sim._object_dofs["eraser"] + 1] = 0.05
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert 1.5 < sim.board_force < 8.0
  assert sim.patch_speed.max() == pytest.approx(0.05)
  if friction == 0:
    # MuJoCo retains a tiny numerical friction floor for a zero coefficient.
    assert sim.board_tangent_force < 0.001
    np.testing.assert_array_equal(sim.cleaning.work_j, 0)
    np.testing.assert_array_equal(sim.remaining, 1)
  else:
    assert sim.board_tangent_force > 0.1
    assert sim.cleaning.work_j.sum() > 0
    assert sim.remaining.min() < 1
    assert sim.remaining.min() > 0.99  # One valid step still leaves the ink.


def test_large_writing_stays_inside_board_at_randomization_limits():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  axes = sim.model.geom_quat[sim.ink_ids]
  rotations = np.empty((C.INK_COUNT, 9))
  for quat, rotation in zip(axes, rotations, strict=True):
    mujoco.mju_quat2Mat(rotation, quat)
  extent = (
    abs(rotations.reshape(-1, 3, 3)[:, :, 2])
    * sim.model.geom_size[sim.ink_ids, 1, None]
    + sim.model.geom_size[sim.ink_ids, 0, None]
  )
  centers = sim.model.geom_pos[sim.ink_ids]
  lower, upper = (centers - extent).min(axis=0), (centers + extent).max(axis=0)
  assert upper[0] - lower[0] > 0.16
  assert upper[1] - lower[1] > 0.20
  row_centers = [centers[:8, 0], centers[8:17, 0], centers[17:, 0]]
  np.testing.assert_allclose([np.mean(row) for row in row_centers], [-0.08, 0, 0.08], atol=0.003)
  assert all(np.ptp(row) < 0.005 for row in row_centers)
  for sign in (-1, 1):
    shifted = centers[:, :2] + sign * C.INK_CENTER_RANGE_M
    assert np.all(abs(shifted) + extent[:, :2] < [0.18, 0.20])
  assert len(sim.remaining) == 25  # Keep existing recording field dimensions.


def test_partially_covered_long_stroke_does_not_erase():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  center = C.BOARD_SURFACE - C.BOARD_ROTATION @ C.PAD_BOTTOM
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, C.WIPE_ROTATION.ravel())
  sim.set_object_pose("eraser", center - 0.0005 * C.BOARD_NORMAL, quat)
  sim.data.xfrc_applied[sim.eraser_body, :3] = -3.0 * C.BOARD_NORMAL
  sim.data.qvel[sim._object_dofs["eraser"] + 1] = 0.05
  mujoco.mj_forward(sim.model, sim.data)
  sim._update_cleaning()
  assert sim.board_force > 1.5 and sim.board_tangent_force > 0.1
  # The two segments two positions away have centers inside the 10.6 cm pad,
  # but their outer ends extend beyond it. Fully covered center segments fade.
  assert sim.remaining[12] < 1
  np.testing.assert_array_equal(sim.remaining[[10, 14]], 1)
  np.testing.assert_array_equal(sim.cleaning.work_j[[10, 14]], 0)


def test_full_actuator_episode_cleans_and_releases_eraser():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  wipe_force = []
  wipe_center = []
  wipe_torque = []
  wipe_contact_count = []
  phase_forces = {
    phase: []
    for phase in (
      "grasp",
      "lift",
      "orient_and_transfer",
      "unload_board",
      "reposition",
      "load_board",
      "wipe",
      "lower",
      "support",
      "release",
    )
  }
  pad_ids = [sim.model.geom(name).id for name in sim.forces.pad_geom_names[5:]]
  pad_to_finger = {geom_id: index for index, geom_id in enumerate(pad_ids)}
  pad_bodies = sim.model.geom_bodyid[pad_ids]
  target_ids = {sim.handle_id, sim.pad_id}
  tangent_basis_local = sim.forces.tangent_basis_local[5:]
  checked_solver_totals = False
  lower_relative = []
  table_peak = 0.0
  placement_peak = 0.0
  held_angular_speed = 0.0
  before_force = None
  update_cleaning = sim._update_cleaning

  def observe_step(current):
    nonlocal before_force, table_peak, held_angular_speed, checked_solver_totals
    nonlocal placement_peak
    before_force = current.board_force
    np.testing.assert_array_equal(current.data.xfrc_applied[current.eraser_body], 0)
    table_peak = max(table_peak, current.table_force)
    if current.phase in ("lower", "support", "release"):
      placement_peak = max(placement_peak, current.table_force)
    if current.phase in phase_forces:
      # Aggregate each physical step directly, avoiding the unnecessary 35-taxel
      # spatial allocation while retaining the provider's signed pad basis.
      normal = np.zeros(5)
      tangent_world = np.zeros((5, 3))
      wrench = np.zeros(6)
      for contact_id, contact in enumerate(current.data.contact):
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in pad_to_finger and geom2 in target_ids:
          finger, sign = pad_to_finger[geom1], -1
        elif geom2 in pad_to_finger and geom1 in target_ids:
          finger, sign = pad_to_finger[geom2], 1
        else:
          continue
        mujoco.mj_contactForce(current.model, current.data, contact_id, wrench)
        frame = np.asarray(contact.frame).reshape(3, 3)
        normal[finger] += abs(wrench[0])
        tangent_world[finger] += sign * (frame[1:].T @ wrench[1:3])
      rotations = current.data.xmat[pad_bodies].reshape(5, 3, 3)
      basis = np.einsum("lij,lkj->lki", rotations, tangent_basis_local)
      tangent = np.einsum("lki,li->lk", basis, tangent_world)
      phase_forces[current.phase].append(
        np.concatenate((normal, np.linalg.norm(tangent, axis=1), tangent.ravel()))
      )
      if not checked_solver_totals and normal.max() > 0.5:
        reference = current.forces.read(current.data)
        np.testing.assert_allclose(normal, reference.normal_force_n[5:], atol=1e-12)
        np.testing.assert_allclose(tangent, reference.tangent_force_n[5:], atol=1e-12)
        checked_solver_totals = True
    if current.phase == "wipe":
      wipe_force.append(current.board_force)
      wipe_center.append(current.board_contact_center.copy())
      wipe_torque.append(current.board_contact_torque.copy())
      wipe_contact_count.append(current.board_contact_count)
    if current.phase in ("return", "lower", "support"):
      held_angular_speed = max(
        held_angular_speed, np.linalg.norm(current.object_twist("eraser")[3:])
      )
    if current.phase == "lower":
      wrist, rotation = current.current_pose_matrix("right")
      lower_relative.append(
        rotation.T @ (current.data.xpos[current.eraser_body] - wrist)
      )

  def verify_integrated_contact(*, integrate=True):
    update_cleaning(integrate=integrate)
    if integrate:
      # The sampled pre-step wrench must be the one consumed by mj_step,
      # rather than a different solution recomputed after integration.
      assert sim.board_force == pytest.approx(before_force, abs=1e-8)

  sim.physics_observer = observe_step
  sim._update_cleaning = verify_integrated_contact

  def forbid_pose_write(*args, **kwargs):
    raise AssertionError("Executor must never teleport its tool")

  sim.set_object_pose = forbid_pose_write
  outcome = WhiteboardWipeExecutor(sim).run()
  assert outcome["success"], outcome
  assert outcome["pickup_verified"] and outcome["released_on_table"]
  assert outcome["maximum_lift_m"] > 0.12
  assert max(outcome["ink_remaining"]) < 0.01
  assert not any(outcome["solver_warnings"])
  assert checked_solver_totals
  # Check raw 1 ms changes separately for each action: low episode peaks alone
  # miss sustained chatter, and a low |Ft| can hide a changing force direction.
  for phase, samples in phase_forces.items():
    assert len(samples) > 1, phase
    differences = np.diff(np.asarray(samples), axis=0)
    rms = np.sqrt(np.mean(differences**2, axis=0))
    limit = 0.08 if phase == "load_board" else 0.05
    assert np.all(rms[:5] < limit), (phase, "Fn", rms[:5])
    assert np.all(rms[5:10] < limit), (phase, "|Ft|", rms[5:10])
    assert np.all(rms[10:] < limit), (phase, "signed Ft", rms[10:])
  active_force = np.asarray(wipe_force)
  contact_count = np.asarray(wipe_contact_count)
  loaded = (active_force > 0.5) & (contact_count > 0)
  consecutive_loaded = loaded[:-1] & loaded[1:]
  assert consecutive_loaded.sum() > 1000
  # A smooth total force can still conceal jumping contact lever arms. Check
  # the normal-load center and the complete contact torque about the tool COM.
  for quantity, samples, limit in (
    ("board contact center", wipe_center, 0.001),
    ("board contact torque", wipe_torque, 0.002),
  ):
    differences = np.diff(np.asarray(samples), axis=0)[consecutive_loaded]
    rms = np.sqrt(np.mean(np.sum(differences**2, axis=1)))
    assert rms < limit, (quantity, rms)
  active_force = active_force[np.flatnonzero(active_force > 0.5)[0] :]
  assert np.mean(active_force > 0.01) > 0.99
  assert np.sqrt(np.mean(np.diff(active_force) ** 2)) < 0.25
  assert max(wipe_force) < 3.0  # Bound physical overshoot around the 2.2 N target.
  for phase in ("wipe",):
    assert np.asarray(phase_forces[phase])[:, 5].max() < 1.3, phase
  # Bound short transients as well as episode peaks and long-phase RMS.
  board = np.asarray(wipe_force)
  thumb = np.asarray(phase_forces["wipe"])[:, 5]
  assert np.max(np.abs(board[20:] - board[:-20])) < 0.4
  assert np.max(np.abs(thumb[20:] - thumb[:-20])) < 0.25
  # Weight is 0.785 N: reject renewed downward preload during placement/release.
  assert table_peak < 2.0
  assert placement_peak < 1.3
  for phase in ("support", "release"):
    assert np.asarray(phase_forces[phase])[:, 5].max() < 0.5, phase
  assert held_angular_speed < 0.8  # Reject dropping/tumbling near placement.
  relative = np.asarray(lower_relative)
  assert np.linalg.norm(relative - relative[0], axis=1).max() < 0.0025
  assert max(outcome["peak_fingertip_force_n"]) < 10.0
  assert sim.table_force > 0.2
  assert np.linalg.norm(sim.object_pose("eraser")[:2] - sim.pickup_center[:2]) < 0.02
  sample = sim.forces.read(sim.data)
  np.testing.assert_allclose(
    sample.normal_taxel_force_n.sum(axis=(1, 2)), sample.normal_force_n
  )


def test_ink_randomization_is_reproducible_local_and_resettable():
  sim = WhiteboardWipeSimulation(ink_seed=123, add_genesis_probes=False)
  initial = sim.model.geom_pos[sim.ink_ids].copy()
  untouched = np.ones(sim.model.ngeom, dtype=bool)
  untouched[sim.ink_ids] = False
  other_positions = sim.model.geom_pos[untouched].copy()
  friction = sim.model.geom_friction.copy()
  for seed in range(50):
    sim.reset(ink_seed=seed)
    layout = sim.ink_randomization
    offset = np.asarray(layout["center_offset_board_m"])
    assert np.all(abs(offset) <= C.INK_CENTER_RANGE_M)
    np.testing.assert_allclose(
      sim.model.geom_pos[sim.ink_ids, :2], sim._initial_ink_positions[:, :2] + offset
    )
    np.testing.assert_allclose(
      sim.ink_surface_center, C.BOARD_SURFACE + C.BOARD_ROTATION[:, :2] @ offset
    )
    np.testing.assert_array_equal(sim.model.geom_pos[untouched], other_positions)
    np.testing.assert_array_equal(sim.model.geom_friction, friction)
    np.testing.assert_array_equal(sim.remaining, 1)
  sim.reset(ink_seed=123)
  np.testing.assert_array_equal(sim.model.geom_pos[sim.ink_ids], initial)
  sim.reset()
  np.testing.assert_array_equal(sim.model.geom_pos[sim.ink_ids], initial)
  sim.reset(ink_seed=None)
  np.testing.assert_array_equal(
    sim.model.geom_pos[sim.ink_ids], sim._initial_ink_positions
  )
  assert not sim.ink_randomization["enabled"]


def test_random_ink_capture_and_replay_restore_actual_geometry(tmp_path, monkeypatch):
  import json

  import h5py
  from kaihand_tactile_env.tasks.whiteboard_wipe.recording import WhiteboardRecorder

  sim = WhiteboardWipeSimulation(ink_seed=123, add_genesis_probes=False)
  source, destination = tmp_path / "source", tmp_path / "replay"
  recorder = WhiteboardRecorder(sim, source)
  sim.step(10)
  recorder.finish({"success": False}, render=False)
  with h5py.File(source / "raw/episode.h5") as h:
    layout = json.loads(h.attrs["ink_randomization_json"])
    wrist = h["wrist_wrench"]
    np.testing.assert_array_equal(wrist["timestamp"], h["state/timestamp"])
    assert wrist["force_local_n"].shape == (len(h["time_s"]), 2, 3)
    assert wrist["torque_local_nm"].shape == (len(h["time_s"]), 2, 3)
    assert np.isfinite(wrist["force_world_n"][:]).all()
    assert np.isfinite(wrist["torque_world_nm"][:]).all()
  assert layout == sim.ink_randomization
  assert (
    json.loads((source / "manifest.json").read_text())["ink_randomization"] == layout
  )
  # Replay must restore model geometry before it creates any camera frames.
  replay = WhiteboardWipeSimulation(add_genesis_probes=False)
  seen = []

  def check_geometry(self):
    np.testing.assert_array_equal(
      self.sim.model.geom_pos[self.sim.ink_ids], layout["geom_positions_board_m"]
    )
    seen.append(True)

  monkeypatch.setattr(WhiteboardRecorder, "_render", check_geometry)
  monkeypatch.setattr(WhiteboardRecorder, "_curves", lambda self: None)
  monkeypatch.setattr(WhiteboardRecorder, "_documents", lambda *args: None)
  WhiteboardRecorder.render_existing(replay, source, destination)
  assert seen == [True]
  np.testing.assert_array_equal(
    sim.model.geom_pos[sim.ink_ids], replay.model.geom_pos[replay.ink_ids]
  )


def test_random_region_corner_episode_tracks_ink_and_completes():
  sim = WhiteboardWipeSimulation(add_genesis_probes=False)
  positions = sim._initial_ink_positions.copy()
  positions[:, :2] += C.INK_CENTER_RANGE_M
  sim.set_ink_layout(positions)
  result = WhiteboardWipeExecutor(sim).run()
  assert result["success"], result
  assert result["pickup_verified"] and result["released_on_table"]
  assert max(result["ink_remaining"]) == 0
  assert not any(result["solver_warnings"])
  assert result["peak_direct_hand_board_force_n"] == 0
  assert result["peak_board_force_n"] < 3.6
  np.testing.assert_array_equal(sim.data.xfrc_applied[sim.eraser_body], 0)
  np.testing.assert_allclose(
    result["ink_randomization"]["center_offset_board_m"], C.INK_CENTER_RANGE_M
  )
