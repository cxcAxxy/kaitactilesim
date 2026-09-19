from __future__ import annotations

import gc
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import (
  FINGERTIP_LINK_NAMES,
  SCENE_OBJECTS,
  default_model_path,
  task_config,
)
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import (
  GENESIS_PROBE_GEOM_PREFIX,
  GenesisProbeTactileProvider,
)


def _body_name(model: mujoco.MjModel, body_id: int) -> str | None:
  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))


def _shared_snapshot(simulation: ArmHandSimulation) -> dict:
  """Compare physical properties by name, never by task-dependent numeric IDs."""
  model = simulation.model
  task_bodies = {
    "card",
    "poker_table",
    "usb_plug",
    "usb_socket",
    "usb_socket_spring_top",
    "usb_socket_spring_bottom",
  }
  fields = {
    "body": (
      "body_pos",
      "body_quat",
      "body_ipos",
      "body_iquat",
      "body_mass",
      "body_inertia",
      "body_gravcomp",
    ),
    "joint": (
      "jnt_type",
      "jnt_limited",
      "jnt_solref",
      "jnt_solimp",
      "jnt_pos",
      "jnt_axis",
      "jnt_stiffness",
      "jnt_range",
      "jnt_margin",
    ),
    "geom": (
      "geom_type",
      "geom_contype",
      "geom_conaffinity",
      "geom_condim",
      "geom_group",
      "geom_priority",
      "geom_size",
      "geom_pos",
      "geom_quat",
      "geom_friction",
      "geom_solmix",
      "geom_solref",
      "geom_solimp",
      "geom_margin",
      "geom_gap",
      "geom_rgba",
    ),
    "actuator": (
      "actuator_trntype",
      "actuator_dyntype",
      "actuator_gaintype",
      "actuator_biastype",
      "actuator_ctrllimited",
      "actuator_forcelimited",
      "actuator_ctrlrange",
      "actuator_forcerange",
      "actuator_gear",
      "actuator_dynprm",
      "actuator_gainprm",
      "actuator_biasprm",
    ),
    "camera": ("cam_mode", "cam_pos", "cam_quat", "cam_fovy", "cam_ipd"),
  }
  snapshot = {}
  for kind, count in (
    ("body", model.nbody),
    ("joint", model.njnt),
    ("geom", model.ngeom),
    ("actuator", model.nu),
    ("camera", model.ncam),
  ):
    rows = snapshot[kind] = {}
    for index in range(count):
      item = getattr(model, kind)(index)
      if kind == "body":
        owner = item.name
      elif kind in {"joint", "geom", "camera"}:
        owner = _body_name(model, item.bodyid[0])
      else:
        owner = None
      if owner in task_bodies:
        continue
      row = {
        field: np.asarray(getattr(model, field)[index]).copy() for field in fields[kind]
      }
      rows[item.name] = row
      if kind == "body":
        row["parent_name"] = _body_name(model, item.parentid[0])
        row["is_mocap"] = bool(item.mocapid[0] >= 0)
      elif kind in {"joint", "geom", "camera"}:
        row["body_name"] = owner
      if kind == "joint":
        # Shared robot joints are scalar hinges. Offsets can change when a
        # scene inserts its free joint earlier in the model.
        assert item.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        qpos_address = int(item.qposadr[0])
        dof_address = int(item.dofadr[0])
        row["initial_qpos"] = simulation.data.qpos[qpos_address].copy()
        row["qpos0"] = model.qpos0[qpos_address].copy()
        row["qpos_spring"] = model.qpos_spring[qpos_address].copy()
        for field in (
          "dof_armature",
          "dof_damping",
          "dof_frictionloss",
          "dof_solref",
          "dof_solimp",
        ):
          row[field] = np.asarray(getattr(model, field)[dof_address]).copy()
      elif kind == "geom":
        row["material_name"] = mujoco.mj_id2name(
          model, mujoco.mjtObj.mjOBJ_MATERIAL, int(item.matid[0])
        )
        if item.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
          row["mesh_name"] = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, int(item.dataid[0])
          )
      elif kind == "actuator":
        assert item.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
        row["joint_names"] = tuple(
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
          for joint_id in item.trnid
        )
      elif kind == "camera":
        row["target_body_name"] = _body_name(model, item.targetbodyid[0])
  return snapshot


@pytest.fixture(scope="module")
def poker_snapshot() -> dict:
  simulation = ArmHandSimulation(scene="poker-draw")
  snapshot = _shared_snapshot(simulation)
  for name in ("usb_plug", "usb_socket"):
    assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
  del simulation
  gc.collect()
  return snapshot


@pytest.fixture(scope="module")
def usb_simulation(poker_snapshot: dict):
  # The large shared mesh model is compiled serially with the reference.
  simulation = ArmHandSimulation(scene="usb-insert")
  yield simulation
  del simulation
  gc.collect()


def test_usb_preserves_poker_robot_table_control_cameras_and_350_probes(
  poker_snapshot: dict,
  usb_simulation: ArmHandSimulation,
) -> None:
  actual = _shared_snapshot(usb_simulation)
  assert {"overhead", "front", "head"}.issubset(poker_snapshot["camera"])
  assert (
    sum(name.startswith(GENESIS_PROBE_GEOM_PREFIX) for name in poker_snapshot["geom"])
    == 350
  )
  assert usb_simulation.genesis_probe_layout.count == 350
  assert tuple(dict.fromkeys(usb_simulation.genesis_probe_layout.body_names)) == (
    FINGERTIP_LINK_NAMES
  )
  for kind, expected_rows in poker_snapshot.items():
    for name, expected_fields in expected_rows.items():
      assert name in actual[kind], (kind, name)
      for field, expected in expected_fields.items():
        np.testing.assert_array_equal(
          actual[kind][name][field],
          expected,
          err_msg=f"shared {kind} {name}: {field}",
        )


def test_usb_uses_shared_head_camera(usb_simulation):
  model = usb_simulation.model
  camera = model.camera("head").id
  np.testing.assert_allclose(model.cam_pos[camera], [0.11, 0, 1.28], atol=1e-12)
  assert model.cam_fovy[camera] == 70.0


def test_usb_is_independent_and_preserves_historical_default(
  usb_simulation: ArmHandSimulation,
) -> None:
  simulation = usb_simulation
  assert default_model_path() == default_model_path("pick-place")
  assert default_model_path("usb-insert") not in {
    default_model_path("pick-place"),
    default_model_path("poker-draw"),
  }
  assert SCENE_OBJECTS["pick-place"] == ("cylinder",)
  assert SCENE_OBJECTS["poker-draw"] == ("card",)
  assert simulation.object_names == simulation.model_object_names == ("usb_plug",)
  assert simulation.model.body("usb_socket").jntnum[0] == 0
  assert (
    simulation.model.joint("usb_plug_freejoint").type[0] == mujoco.mjtJoint.mjJNT_FREE
  )
  for name in ("card", "cylinder", "box", "poker_table"):
    assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
  for scene in ("pick-place", "poker-draw"):
    with pytest.raises(ValueError, match="independent task models"):
      simulation.set_scene(scene)
  usb_home = task_config("usb-insert").ARM_HOME
  poker_home = task_config("poker-draw").ARM_HOME
  assert usb_home is poker_home


def test_usb_idle_left_arm_folds_acute_and_holds_clear_of_table():
  simulation = ArmHandSimulation(scene="usb-insert")
  model, data = simulation.model, simulation.data
  previous = mujoco.MjData(model)
  previous.qpos[:] = data.qpos
  previous.qpos[simulation._arm_qpos["left"]] = np.deg2rad(
    [55, -65, -70, -60, -120, 0, 0]
  )
  mujoco.mj_forward(model, previous)
  site = model.site("left_ee_site").id
  initial_position = data.site_xpos[site].copy()
  assert initial_position[0] < previous.site_xpos[site, 0] - 0.15
  shoulder = data.body("left_arm_link2").xpos
  elbow = data.body("left_arm_link4").xpos
  wrist = data.body("left_arm_link5").xpos
  upper, forearm = shoulder - elbow, wrist - elbow
  angle = np.degrees(
    np.arccos(upper @ forearm / np.linalg.norm(upper) / np.linalg.norm(forearm))
  )
  assert 50 < angle < 85
  elevation = np.degrees(np.arctan2(forearm[2], np.linalg.norm(forearm[:2])))
  assert abs(elevation) < 5.0
  # The upper arm hangs close to the body instead of abducting at the shoulder.
  assert 0.03 < elbow[1] - shoulder[1] < 0.10
  assert shoulder[2] - elbow[2] > 0.25
  np.testing.assert_allclose(data.qpos[simulation._arm_qpos["left"][-2:]], 0)
  left_bodies = [
    i for i in range(model.nbody) if (model.body(i).name or "").startswith("hand_l_")
  ]
  simulation.step(250)
  mujoco.mj_forward(model, data)
  assert np.isfinite(data.qpos).all()
  np.testing.assert_allclose(data.site_xpos[site], initial_position, atol=0.002)
  assert data.xpos[left_bodies, 2].min() > 0.80
  for contact in data.contact:
    touches_left = [model.geom_bodyid[g] in left_bodies for g in contact.geom]
    # Existing palm/thumb self-contact is unrelated to shoulder placement.
    assert not (any(touches_left) and not all(touches_left))


@pytest.mark.parametrize("scene", ("pick-place", "poker-draw"))
def test_existing_scenes_do_not_gain_usb_objects(scene: str) -> None:
  model = mujoco.MjModel.from_xml_path(str(default_model_path(scene)))
  for kind, count in (
    ("body", model.nbody),
    ("geom", model.ngeom),
    ("joint", model.njnt),
  ):
    assert not any(
      getattr(model, kind)(index).name.startswith("usb_") for index in range(count)
    )
  for name in SCENE_OBJECTS[scene]:
    assert model.body(name).jntnum[0] == 1
  del model
  gc.collect()


def _plug_collision_ids(model: mujoco.MjModel) -> set[int]:
  return {
    index
    for index in range(model.ngeom)
    if model.geom_bodyid[index] == model.body("usb_plug").id
    and (model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
  }


def test_usb_handle_shoulder_clears_mouth_when_tip_reaches_backstop(
  usb_simulation: ArmHandSimulation,
) -> None:
  """The grip shoulder must not create a false wall load before bottom contact."""
  model = usb_simulation.model
  config = task_config("usb-insert")
  handle = model.geom("usb_plug_handle").id
  backstop = model.geom("usb_socket_backstop").id
  tip = model.site("usb_plug_tip").id
  np.testing.assert_allclose(
    model.geom_pos[handle], config.PLUG_HANDLE_CENTER_LOCAL_M, rtol=0, atol=1e-12
  )
  np.testing.assert_allclose(
    model.geom_size[handle], config.PLUG_HANDLE_HALF_SIZE_M, rtol=0, atol=1e-12
  )
  np.testing.assert_allclose(
    model.site_pos[tip], config.PLUG_TIP_LOCAL_M, rtol=0, atol=1e-12
  )
  np.testing.assert_allclose(
    model.site_pos[model.site("usb_plug_grasp").id],
    config.PLUG_GRASP_LOCAL_M,
    rtol=0,
    atol=1e-12,
  )
  assert model.body_mass[model.body("usb_plug").id] == pytest.approx(
    config.PLUG_MASS_KG
  )
  for geom in (handle, backstop):
    assert int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    np.testing.assert_allclose(model.geom_quat[geom], [1, 0, 0, 0], rtol=0, atol=1e-12)
  shoulder_x = model.geom_pos[handle, 0] + model.geom_size[handle, 0]
  tip_to_shoulder = model.site_pos[tip, 0] - shoulder_x
  backstop_depth = model.geom_pos[backstop, 0] - model.geom_size[backstop, 0]
  assert backstop_depth == pytest.approx(config.BACKSTOP_DEPTH_M, abs=1e-12)
  # Old geometry yielded exactly zero; the narrow neck leaves 1 mm at seating.
  assert tip_to_shoulder - backstop_depth == pytest.approx(0.001, abs=1e-12)
  # Conservative bound over any tilt axis within the monitor's 3 degree limit.
  # Include the allowed bottom penetration and both geoms' contact margins.
  tilt = np.deg2rad(3.0)
  shoulder_radius = float(np.linalg.norm(model.geom_size[handle, 1:]))
  mouth_margin = max(
    model.geom_margin[model.geom(f"usb_socket_wall_{side}").id]
    for side in ("left", "right", "top", "bottom")
  )
  worst_clearance = (
    tip_to_shoulder * np.cos(tilt)
    - shoulder_radius * np.sin(tilt)
    - backstop_depth
    - config.MAX_SOCKET_PENETRATION_M
    - model.geom_margin[handle]
    - mouth_margin
  )
  assert worst_clearance > 0.00025


def test_usb_neck_bridges_grip_and_shell_without_filling_socket_aperture(
  usb_simulation: ArmHandSimulation,
) -> None:
  model = usb_simulation.model
  config = task_config("usb-insert")
  handle = model.geom("usb_plug_handle").id
  neck = model.geom("usb_plug_neck").id
  shell = model.geom("usb_plug_shell_top").id
  assert "usb_plug_neck" in config.SCENE_GEOM_NAMES
  assert neck in _plug_collision_ids(model)
  assert int(model.geom_type[neck]) == int(mujoco.mjtGeom.mjGEOM_BOX)
  np.testing.assert_allclose(model.geom_quat[neck], [1, 0, 0, 0], rtol=0, atol=1e-12)
  neck_rear = model.geom_pos[neck, 0] - model.geom_size[neck, 0]
  neck_front = model.geom_pos[neck, 0] + model.geom_size[neck, 0]
  assert neck_rear == pytest.approx(
    model.geom_pos[handle, 0] + model.geom_size[handle, 0], abs=1e-12
  )
  assert neck_front == pytest.approx(
    model.geom_pos[shell, 0] - model.geom_size[shell, 0], abs=1e-12
  )
  assert 2 * model.geom_size[shell, 0] == pytest.approx(
    config.PLUG_SHELL_LENGTH_M, abs=1e-12
  )
  # Derive the aperture from the compiled collision walls, then check config.
  left = model.geom("usb_socket_wall_left").id
  top = model.geom("usb_socket_wall_top").id
  half_width = model.geom_pos[left, 1] - model.geom_size[left, 1]
  half_height = model.geom_pos[top, 2] - model.geom_size[top, 2]
  assert half_width == pytest.approx(config.SOCKET_HALF_WIDTH_M, abs=1e-12)
  assert half_height == pytest.approx(config.SOCKET_HALF_HEIGHT_M, abs=1e-12)
  neck_extent = np.abs(model.geom_pos[neck, 1:]) + model.geom_size[neck, 1:]
  margins = model.geom_margin[neck] + model.geom_margin[[left, top]]
  assert np.all(neck_extent + margins < [half_width, half_height])
  assert np.all(model.geom_size[neck, 1:] < model.geom_size[handle, 1:])


def test_usb_springs_are_unactuated_outward_slide_compliances(
  poker_snapshot: dict,
  usb_simulation: ArmHandSimulation,
) -> None:
  model = usb_simulation.model
  # Exact actuator-name equality disallows an added force or position motor on
  # either spring, including transmissions other than a direct joint actuator.
  assert {model.actuator(i).name for i in range(model.nu)} == set(
    poker_snapshot["actuator"]
  )
  socket = model.body("usb_socket").id
  spring_joints = {
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).name.startswith("usb_socket_spring_")
  }
  assert spring_joints == {
    "usb_socket_spring_top_slide",
    "usb_socket_spring_bottom_slide",
  }
  for side, sign in (("top", 1), ("bottom", -1)):
    body = model.body(f"usb_socket_spring_{side}").id
    joint = model.joint(f"usb_socket_spring_{side}_slide").id
    qpos = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    assert int(model.body_parentid[body]) == socket
    assert int(model.body_jntnum[body]) == 1
    assert int(model.jnt_bodyid[joint]) == body
    assert int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
    np.testing.assert_allclose(model.jnt_axis[joint], [0, 0, sign], rtol=0, atol=1e-12)
    assert model.jnt_limited[joint]
    assert model.jnt_range[joint, 0] == 0
    assert model.jnt_range[joint, 1] >= 0.001
    assert model.qpos_spring[qpos] == 0
    assert model.jnt_stiffness[joint] > 0
    assert model.dof_damping[dof] > 0
    assert model.body_mass[body] > 0
    assert not np.any(
      (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
      & (model.actuator_trnid[:, 0] == joint)
    )


def test_usb_spring_collision_masks_admit_only_the_two_shell_faces(
  usb_simulation: ArmHandSimulation,
) -> None:
  model = usb_simulation.model
  expected = {"usb_plug_shell_top", "usb_plug_shell_bottom"}
  for side in ("top_pad_left", "top_pad_right", "bottom_pad_left", "bottom_pad_right"):
    spring = model.geom(f"usb_socket_spring_{side}").id
    assert int(model.geom_contype[spring]) != 0
    assert int(model.geom_conaffinity[spring]) == 0
    admitted = {
      model.geom(other).name
      for other in range(model.ngeom)
      if (int(model.geom_contype[spring]) & int(model.geom_conaffinity[other]))
      or (int(model.geom_contype[other]) & int(model.geom_conaffinity[spring]))
    }
    # This includes all robot meshes, tactile pads/probes, the table, socket
    # walls, the handle/neck and the opposite spring in the exclusion check.
    assert admitted == expected
    assert all(
      spring not in (int(first), int(second))
      for first, second in zip(model.pair_geom1, model.pair_geom2, strict=True)
    ), "explicit contact pairs must not bypass the spring collision mask"


def test_usb_tactile_selects_plug_and_excludes_socket_and_table(
  usb_simulation: ArmHandSimulation,
) -> None:
  simulation = usb_simulation
  simulation.reset()
  model = simulation.model
  genesis = GenesisProbeTactileProvider(model, simulation.genesis_probe_layout)
  solver = SolverDistributedTactileProvider(model, simulation.genesis_probe_layout)
  expected = _plug_collision_ids(model)
  assert expected
  assert set(genesis._target_geom_ids) == expected
  assert set(solver._target_geom_ids) == expected
  assert {model.geom(index).name for index in expected} == set(solver.target_geom_names)

  furniture_ids = [
    index
    for index in range(model.ngeom)
    if _body_name(model, model.geom_bodyid[index]) in {"table", "usb_socket"}
    and (model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0)
  ]
  assert any(_body_name(model, model.geom_bodyid[i]) == "table" for i in furniture_ids)
  assert any(
    _body_name(model, model.geom_bodyid[i]) == "usb_socket" for i in furniture_ids
  )
  pad_id = model.geom("hand_r_index_link4_tactile_pad_col").id
  data = SimpleNamespace(
    time=simulation.data.time,
    xpos=simulation.data.xpos,
    xmat=simulation.data.xmat,
    ncon=len(furniture_ids),
    contact=[SimpleNamespace(geom1=pad_id, geom2=index) for index in furniture_ids],
  )
  # Even explicit pad/furniture contact candidates must not enter either
  # task-object tactile stream; no synthetic force or distance is installed.
  assert not genesis.read(data).contact.any()
  assert not genesis.probe_depth.any()
  assert not solver.read(data).contact_count.any()
  for geom_id in furniture_ids:
    with pytest.raises(ValueError, match="target geoms must belong"):
      SolverDistributedTactileProvider(
        model,
        simulation.genesis_probe_layout,
        target_geom_names=(model.geom(geom_id).name,),
      )


def test_real_usb_pad_contact_produces_depth_and_conserved_solver_force(
  usb_simulation: ArmHandSimulation,
) -> None:
  simulation = usb_simulation
  simulation.reset()
  model, data = simulation.model, simulation.data
  layout = simulation.genesis_probe_layout
  genesis = GenesisProbeTactileProvider(model, layout)
  solver = SolverDistributedTactileProvider(model, layout)
  link_name = "hand_r_index_link4"
  link_id = model.body(link_name).id
  link_index = FINGERTIP_LINK_NAMES.index(link_name)
  indices = np.flatnonzero(
    (np.asarray(layout.body_names) == link_name) & (layout.probe_radius > 0.0)
  )
  body_rotation = data.xmat[link_id].reshape(3, 3)
  normal = body_rotation @ layout.local_normal[indices].mean(axis=0)
  normal /= np.linalg.norm(normal)
  center = data.xpos[link_id] + body_rotation @ layout.local_pos[indices].mean(axis=0)
  boxes = [
    index
    for index in _plug_collision_ids(model)
    if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  assert boxes
  grip_id = max(boxes, key=lambda index: float(np.prod(model.geom_size[index])))
  short_axis = int(np.argmin(model.geom_size[grip_id]))
  reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
  tangent = reference - np.dot(reference, normal) * normal
  tangent /= np.linalg.norm(tangent)
  target_rotation = np.empty((3, 3))
  target_rotation[:, short_axis] = normal
  target_rotation[:, (short_axis + 1) % 3] = tangent
  target_rotation[:, (short_axis + 2) % 3] = np.cross(normal, tangent)
  geom_rotation = np.empty(9)
  mujoco.mju_quat2Mat(geom_rotation, model.geom_quat[grip_id])
  object_rotation = target_rotation @ geom_rotation.reshape(3, 3).T
  quaternion = np.empty(4)
  mujoco.mju_mat2Quat(quaternion, object_rotation.reshape(9))
  # A small, deliberate indentation at the real pad establishes a solver
  # constraint without moving the hand or changing collision settings.
  grip_center = center + normal * (model.geom_size[grip_id, short_axis] - 0.0005)
  position = grip_center - object_rotation @ model.geom_pos[grip_id]
  try:
    simulation.set_object_pose("usb_plug", position, quaternion)
    depth_sample = genesis.read(data)
    force_sample = solver.read(data)
    assert depth_sample.contact[link_index]
    assert depth_sample.centroid_valid[link_index]
    assert genesis.probe_contact_instantaneous[indices].any()
    assert genesis.probe_depth[indices].max() > genesis.contact_threshold_m
    assert force_sample.contact_count[link_index] > 0
    assert force_sample.normal_force_n[link_index] > 0.0
    assert force_sample.normal_taxel_force_n[link_index].max() > 0.0
    np.testing.assert_allclose(
      force_sample.normal_taxel_force_n.sum(axis=(1, 2)),
      force_sample.normal_force_n,
      rtol=1.0e-12,
      atol=1.0e-12,
    )
    assert all(
      assignment.target_geom_id in _plug_collision_ids(model)
      for assignment in force_sample.assignments
    )
  finally:
    simulation.reset()
