from __future__ import annotations

import gc
import xml.etree.ElementTree as ET
from pathlib import Path

import kaihand_tactile_env
import mujoco
import numpy as np
import pytest

PACKAGE_DIR = Path(kaihand_tactile_env.__file__).resolve().parent
LEGACY_MODEL = PACKAGE_DIR / "assets/workcell/mjcf/kaihand_dual_arm_workcell.xml"
SCENE_MODELS = {
  "pick_place": PACKAGE_DIR / "tasks/pick_place/scene.xml",
  "poker_draw": PACKAGE_DIR / "tasks/poker_draw/scene.xml",
}

BODY_FIELDS = (
  "body_parentid",
  "body_pos",
  "body_quat",
  "body_ipos",
  "body_iquat",
  "body_mass",
  "body_inertia",
  "body_gravcomp",
  "body_mocapid",
)
JOINT_FIELDS = (
  "jnt_type",
  "jnt_bodyid",
  "jnt_qposadr",
  "jnt_dofadr",
  "jnt_limited",
  "jnt_solref",
  "jnt_solimp",
  "jnt_pos",
  "jnt_axis",
  "jnt_stiffness",
  "jnt_range",
  "jnt_margin",
)
DOF_FIELDS = (
  "dof_armature",
  "dof_damping",
  "dof_frictionloss",
  "dof_solref",
  "dof_solimp",
)
GEOM_FIELDS = (
  "geom_type",
  "geom_bodyid",
  "geom_contype",
  "geom_conaffinity",
  "geom_condim",
  "geom_dataid",
  "geom_matid",
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
)
ACTUATOR_FIELDS = (
  "actuator_trntype",
  "actuator_dyntype",
  "actuator_gaintype",
  "actuator_biastype",
  "actuator_trnid",
  "actuator_ctrllimited",
  "actuator_forcelimited",
  "actuator_ctrlrange",
  "actuator_forcerange",
  "actuator_gear",
  "actuator_dynprm",
  "actuator_gainprm",
  "actuator_biasprm",
)
TASK_BODIES = {
  "pick_place": ("cylinder", "box"),
  "poker_draw": ("poker_table", "card"),
}
TASK_GEOMS = {
  "pick_place": (
    "cylinder_geom",
    "box_bottom",
    "box_wall_x_pos",
    "box_wall_x_neg",
    "box_wall_y_pos",
    "box_wall_y_neg",
  ),
  "poker_draw": (
    "poker_table_base",
    "poker_table_top",
    "card_core_geom",
    "card_back_visual",
    "card_face_visual",
  ),
}
TASK_MATERIALS = {
  "pick_place": ("cylinder", "box"),
  "poker_draw": ("poker_felt", "card_back", "card_face"),
}


@pytest.mark.parametrize(
  "task", ("pick_place", "poker_draw", "usb_insert", "bulb_screw", "vase_wipe", "install_ram")
)
def test_all_tasks_inherit_shared_head_and_symmetric_wrist_cameras(task: str) -> None:
  scene_path = PACKAGE_DIR / "tasks" / task / "scene.xml"
  scene = ET.parse(scene_path).getroot()
  assert not scene.findall(".//default/camera"), "task-local camera defaults drift"
  assert scene.find(".//camera[@name='head']") is None
  assert scene.find(".//camera[@name='right_wrist']") is None
  assert scene.find(".//camera[@name='left_wrist']") is None
  robot = ET.parse(PACKAGE_DIR / "shared/mjcf/robot.xml").getroot()
  head = robot.find(".//camera[@name='head']")
  assert head is not None
  assert head.attrib == {"name": "head", "class": "robot_head_camera"}
  wrist_node = robot.find(".//camera[@name='right_wrist']")
  assert wrist_node is not None
  assert wrist_node.attrib == {"name": "right_wrist", "class": "robot_right_wrist_camera"}
  left_node = robot.find(".//camera[@name='left_wrist']")
  assert left_node is not None
  assert left_node.attrib == {"name": "left_wrist", "class": "robot_left_wrist_camera"}
  from kaihand_tactile_env.shared.config import SHARED_CAMERA_NAMES

  # Compile one model at a time; no rendering, probes, or simulation rollouts.
  model = mujoco.MjModel.from_xml_path(str(scene_path))
  for name in SHARED_CAMERA_NAMES:
    assert model.camera(name).id >= 0
  camera = model.camera("head")
  assert model.body(int(camera.bodyid[0])).name == "torso"
  np.testing.assert_allclose(camera.pos, [0.11, 0.0, 1.28], atol=1e-12)
  assert camera.fovy[0] == 70.0
  matrix = np.empty(9)
  mujoco.mju_quat2Mat(matrix, camera.quat)
  x_axis = np.array([0.0, -1.0, 0.0])
  y_axis = np.array([0.76229408, 0.0, 0.64723082])
  y_axis /= np.linalg.norm(y_axis)
  np.testing.assert_allclose(
    matrix.reshape(3, 3),
    np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis))),
    atol=1e-12,
  )
  for name, position, fovy in (
    ("front", [1.45, 0.0, 1.02], 50.0),
    ("overhead", [0.56, 0.0, 1.75], 52.0),
  ):
    np.testing.assert_allclose(model.camera(name).pos, position, atol=1e-12)
    assert model.camera(name).fovy[0] == fovy
  wrist = model.camera("right_wrist")
  assert model.body(int(wrist.bodyid[0])).name == "right_hand_mount"
  assert int(wrist.mode[0]) == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
  np.testing.assert_allclose(wrist.pos, [0.0, 0.035, 0.025], atol=1e-12)
  assert wrist.fovy[0] == 85.0
  left = model.camera("left_wrist")
  mirror = np.diag([1.0, -1.0, 1.0])
  right_rotation, left_rotation = np.empty(9), np.empty(9)
  mujoco.mju_quat2Mat(right_rotation, wrist.quat)
  mujoco.mju_quat2Mat(left_rotation, left.quat)
  np.testing.assert_allclose(left.pos, mirror @ wrist.pos, atol=1e-12)
  np.testing.assert_allclose(
    left_rotation.reshape(3, 3),
    mirror @ right_rotation.reshape(3, 3) @ np.diag([-1.0, 1.0, 1.0]),
    atol=1e-12,
  )
  assert np.linalg.det(left_rotation.reshape(3, 3)) == pytest.approx(1.0)
  assert left.fovy[0] == wrist.fovy[0]
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  initial_wrist = data.cam_xpos[wrist.id].copy()
  initial_wrist_rotation = data.cam_xmat[wrist.id].copy()
  initial_head = data.cam_xpos[camera.id].copy()
  data.qpos[int(model.joint("right_arm_joint7").qposadr[0])] += 0.2
  mujoco.mj_forward(model, data)
  assert np.linalg.norm(data.cam_xpos[wrist.id] - initial_wrist) > 0.001
  assert np.linalg.norm(data.cam_xmat[wrist.id] - initial_wrist_rotation) > 0.01
  np.testing.assert_array_equal(data.cam_xpos[camera.id], initial_head)
  parent = int(wrist.bodyid[0])
  np.testing.assert_allclose(
    data.xmat[parent].reshape(3, 3).T @ (data.cam_xpos[wrist.id] - data.xpos[parent]),
    wrist.pos,
    atol=1e-12,
  )
  local_rotation = np.empty(9)
  mujoco.mju_quat2Mat(local_rotation, wrist.quat)
  np.testing.assert_allclose(
    data.xmat[parent].reshape(3, 3).T @ data.cam_xmat[wrist.id].reshape(3, 3),
    local_rotation.reshape(3, 3),
    atol=1e-12,
  )
  assert model.body(int(left.bodyid[0])).name == "left_hand_mount"
  assert int(left.mode[0]) == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
  initial_left = data.cam_xpos[left.id].copy()
  initial_left_rotation = data.cam_xmat[left.id].copy()
  right_position = data.cam_xpos[wrist.id].copy()
  right_orientation = data.cam_xmat[wrist.id].copy()
  data.qpos[int(model.joint("left_arm_joint7").qposadr[0])] += 0.2
  mujoco.mj_forward(model, data)
  assert np.linalg.norm(data.cam_xpos[left.id] - initial_left) > 0.001
  assert np.linalg.norm(data.cam_xmat[left.id] - initial_left_rotation) > 0.01
  np.testing.assert_array_equal(data.cam_xpos[wrist.id], right_position)
  np.testing.assert_array_equal(data.cam_xmat[wrist.id], right_orientation)
  np.testing.assert_array_equal(data.cam_xpos[camera.id], initial_head)
  parent = int(left.bodyid[0])
  np.testing.assert_allclose(
    data.xmat[parent].reshape(3, 3).T @ (data.cam_xpos[left.id] - data.xpos[parent]),
    left.pos,
    atol=1e-12,
  )
  np.testing.assert_allclose(
    data.xmat[parent].reshape(3, 3).T @ data.cam_xmat[left.id].reshape(3, 3),
    left_rotation.reshape(3, 3),
    atol=1e-12,
  )
  del data
  del model
  gc.collect()


def _names(
  model: mujoco.MjModel,
  object_type: mujoco.mjtObj,
  count: int,
) -> tuple[str | None, ...]:
  return tuple(mujoco.mj_id2name(model, object_type, index) for index in range(count))


def _copy_fields(
  model: mujoco.MjModel,
  fields: tuple[str, ...],
  count: int,
) -> dict[str, np.ndarray]:
  return {field: np.asarray(getattr(model, field)[:count]).copy() for field in fields}


def _named_rows(
  model: mujoco.MjModel,
  object_type: mujoco.mjtObj,
  names: tuple[str, ...],
  fields: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
  rows: dict[str, dict[str, np.ndarray]] = {}
  for name in names:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0
    rows[name] = {
      field: np.asarray(getattr(model, field)[object_id]).copy() for field in fields
    }
  return rows


def _assert_fields_equal(
  actual: mujoco.MjModel,
  expected: dict[str, np.ndarray],
) -> None:
  for field, values in expected.items():
    np.testing.assert_array_equal(
      np.asarray(getattr(actual, field))[: len(values)], values
    )


def _assert_named_rows_equal(
  model: mujoco.MjModel,
  object_type: mujoco.mjtObj,
  expected: dict[str, dict[str, np.ndarray]],
) -> None:
  for name, fields in expected.items():
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0
    for field, values in fields.items():
      np.testing.assert_array_equal(getattr(model, field)[object_id], values)


def test_task_models_preserve_legacy_robot_and_keep_objects_isolated() -> None:
  """The split may remove objects, but must not silently alter the robot."""
  legacy = mujoco.MjModel.from_xml_path(str(LEGACY_MODEL))
  common_body_count = legacy.body("cylinder").id
  common_joint_count = legacy.joint("cylinder_freejoint").id
  common_dof_count = int(legacy.jnt_dofadr[common_joint_count])
  common_qpos_count = int(legacy.jnt_qposadr[common_joint_count])
  common_geom_count = legacy.geom("cylinder_geom").id

  common_names = {
    "body": _names(legacy, mujoco.mjtObj.mjOBJ_BODY, common_body_count),
    "joint": _names(legacy, mujoco.mjtObj.mjOBJ_JOINT, common_joint_count),
    "geom": _names(legacy, mujoco.mjtObj.mjOBJ_GEOM, common_geom_count),
    "actuator": _names(legacy, mujoco.mjtObj.mjOBJ_ACTUATOR, legacy.nu),
    "camera": _names(legacy, mujoco.mjtObj.mjOBJ_CAMERA, legacy.ncam),
  }
  common_fields = {
    "body": _copy_fields(legacy, BODY_FIELDS, common_body_count),
    "joint": _copy_fields(legacy, JOINT_FIELDS, common_joint_count),
    "dof": _copy_fields(legacy, DOF_FIELDS, common_dof_count),
    "geom": _copy_fields(legacy, GEOM_FIELDS, common_geom_count),
    "actuator": _copy_fields(legacy, ACTUATOR_FIELDS, legacy.nu),
  }
  common_qpos0 = legacy.qpos0[:common_qpos_count].copy()
  common_qpos_spring = legacy.qpos_spring[:common_qpos_count].copy()
  option_contract = (
    legacy.opt.timestep,
    legacy.opt.gravity.copy(),
    int(legacy.opt.integrator),
    legacy.opt.iterations,
    int(legacy.opt.cone),
  )
  task_bodies = {
    task: _named_rows(
      legacy,
      mujoco.mjtObj.mjOBJ_BODY,
      names,
      ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia"),
    )
    for task, names in TASK_BODIES.items()
  }
  # Material ids differ after removing the other task, so compare the physical
  # geom values and the referenced material's RGBA independently.
  task_geoms = {
    task: _named_rows(
      legacy,
      mujoco.mjtObj.mjOBJ_GEOM,
      names,
      tuple(
        field for field in GEOM_FIELDS if field not in {"geom_bodyid", "geom_matid"}
      ),
    )
    for task, names in TASK_GEOMS.items()
  }
  task_material_rgba = {
    task: {name: legacy.material(name).rgba.copy() for name in names}
    for task, names in TASK_MATERIALS.items()
  }
  legacy_equalities = {
    field: np.asarray(getattr(legacy, field)).copy()
    for field in (
      "eq_type",
      "eq_obj1id",
      "eq_obj2id",
      "eq_active0",
      "eq_solref",
      "eq_solimp",
      "eq_data",
    )
  }
  del legacy
  gc.collect()

  for task, model_path in SCENE_MODELS.items():
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nbody == common_body_count + len(TASK_BODIES[task])
    assert model.njnt == common_joint_count + 1
    assert model.ngeom == common_geom_count + len(TASK_GEOMS[task])

    assert (
      _names(model, mujoco.mjtObj.mjOBJ_BODY, common_body_count) == common_names["body"]
    )
    assert (
      _names(model, mujoco.mjtObj.mjOBJ_JOINT, common_joint_count)
      == common_names["joint"]
    )
    assert (
      _names(model, mujoco.mjtObj.mjOBJ_GEOM, common_geom_count) == common_names["geom"]
    )
    assert (
      _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu) == common_names["actuator"]
    )
    assert _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam) == (
      *common_names["camera"],
      "left_wrist",
      "right_wrist",
    )

    _assert_fields_equal(model, common_fields["body"])
    _assert_fields_equal(model, common_fields["joint"])
    _assert_fields_equal(model, common_fields["dof"])
    _assert_fields_equal(model, common_fields["geom"])
    _assert_fields_equal(model, common_fields["actuator"])
    np.testing.assert_array_equal(model.qpos0[:common_qpos_count], common_qpos0)
    np.testing.assert_array_equal(
      model.qpos_spring[:common_qpos_count], common_qpos_spring
    )
    assert model.opt.timestep == option_contract[0]
    np.testing.assert_array_equal(model.opt.gravity, option_contract[1])
    assert int(model.opt.integrator) == option_contract[2]
    assert model.opt.iterations == option_contract[3]
    assert int(model.opt.cone) == option_contract[4]

    _assert_named_rows_equal(model, mujoco.mjtObj.mjOBJ_BODY, task_bodies[task])
    _assert_named_rows_equal(model, mujoco.mjtObj.mjOBJ_GEOM, task_geoms[task])
    for name, rgba in task_material_rgba[task].items():
      np.testing.assert_array_equal(model.material(name).rgba, rgba)

    other_task = "poker_draw" if task == "pick_place" else "pick_place"
    for name in TASK_BODIES[other_task]:
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    for name in TASK_GEOMS[other_task]:
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1

    equality_slice = slice(None) if task == "pick_place" else slice(1, None)
    for field, legacy_values in legacy_equalities.items():
      np.testing.assert_array_equal(
        getattr(model, field), legacy_values[equality_slice]
      )

    del model
    gc.collect()
