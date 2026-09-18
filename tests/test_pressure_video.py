"""Camera-only pressure-video tests on tiny models; no robot or rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.pressure_video import (
  configure_pressure_window_video,
)

_XML = """
<mujoco>
  <worldbody>
    <camera name="front" pos="1.45 0 1.02" xyaxes="0 1 0 0 0 1" fovy="50"/>
    <camera name="overhead" pos="0.56 0 1.75" fovy="52"/>
    <body name="poker_table" pos="0.58 -0.18 0.76">
      <geom name="poker_table_top" type="box" pos="0 0 0.0805"
            size="0.115 0.070 0.0005" friction="1.30 0.005 0.0005"/>
    </body>
    <body name="card" pos="0.58 -0.16 0.84175">
      <freejoint/>
      <geom name="card_core_geom" type="box" size="0.0315 0.044 0.00055"
            mass="0.004" friction="1.4 0.02 0.002"/>
    </body>
    <body pos="0 0 2">
      <joint name="test_joint"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
  <actuator><position joint="test_joint" kp="30"/></actuator>
</mujoco>
"""


class _Video:
  def __init__(self, simulation):
    self.simulation = simulation
    self._metadata = {"existing": {"keep": True}}
    self.camera = None

  def follow_viewer_camera(self, camera):
    self.camera = camera


def _simulation(xml=_XML):
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)
  data.qvel[:] = 0.001
  data.ctrl[:] = 0.1
  data.qfrc_applied[:] = 0.002
  mujoco.mj_forward(model, data)
  return SimpleNamespace(scene="poker-draw", model=model, data=data)


def _snapshot(simulation):
  result = {}
  for owner_name, names in (
    (
      "model",
      (
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "body_mass",
        "body_pos",
        "geom_pos",
        "geom_size",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "jnt_range",
        "qpos0",
      ),
    ),
    (
      "data",
      (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "qfrc_applied",
        "xfrc_applied",
        "qfrc_actuator",
        "qfrc_constraint",
        "qfrc_bias",
        "geom_xpos",
        "geom_xmat",
        "efc_force",
        "efc_pos",
      ),
    ),
  ):
    owner = getattr(simulation, owner_name)
    for name in names:
      result[f"{owner_name}.{name}"] = getattr(owner, name).copy()
  result["time"] = simulation.data.time
  result["ncon"] = simulation.data.ncon
  result["timestep"] = simulation.model.opt.timestep
  return result


def test_cameras_frame_complete_robot_side_drag_and_metadata_is_json_safe():
  simulation = _simulation()
  video = _Video(simulation)

  metadata = configure_pressure_window_video(video, simulation)

  assert video.camera.type == mujoco.mjtCamera.mjCAMERA_FREE
  np.testing.assert_allclose(video.camera.lookat, [0.5225, -0.16, 0.841])
  assert video.camera.distance == pytest.approx(0.50)
  assert video.camera.azimuth == 135.0
  assert video.camera.elevation == -35.0
  overhead = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
  np.testing.assert_allclose(simulation.model.cam_pos[overhead], [0.5225, -0.16, 1.341])
  np.testing.assert_allclose(simulation.data.cam_xpos[overhead], [0.5225, -0.16, 1.341])
  np.testing.assert_allclose(
    simulation.data.cam_xmat[overhead].reshape(3, 3), np.eye(3)
  )
  assert simulation.model.cam_fovy[overhead] == 40.0

  # The complete start-to-half-overhang card path fits the top view with margin.
  half_view_height = 0.5 * np.tan(np.deg2rad(40.0 / 2))
  half_view_width = half_view_height * (4.0 / 3.0)
  assert 0.58 + 0.0315 < 0.5225 + half_view_width
  assert 0.465 - 0.0315 > 0.5225 - half_view_width
  assert 0.044 < half_view_height
  assert metadata["main"]["native_viewer_opened"] is False
  assert metadata["main"]["recorder_mode_label"] == "viewer"
  assert metadata["overhead"]["original_before_first_configuration"] == {
    "position_m": [0.56, 0.0, 1.75],
    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    "fovy_deg": 52.0,
  }
  assert video._metadata["pressure_window_camera"] == metadata
  assert video._metadata["existing"] == {"keep": True}
  assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
  metadata["main"]["lookat_world_m"][0] = 100.0
  assert (
    video._metadata["pressure_window_camera"]["main"]["lookat_world_m"][0] == 0.5225
  )


def test_no_live_forward_step_or_physics_state_change(monkeypatch):
  simulation = _simulation()
  before = _snapshot(simulation)
  model = simulation.model
  front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
  front_before = [
    model.cam_pos[front].copy(),
    model.cam_quat[front].copy(),
    model.cam_fovy[front],
  ]

  def forbidden(*_args, **_kwargs):
    pytest.fail("camera setup must not run forward dynamics or step physics")

  for name in ("mj_forward", "mj_step", "mj_step1", "mj_step2", "mj_kinematics"):
    monkeypatch.setattr(mujoco, name, forbidden)
  configure_pressure_window_video(_Video(simulation), simulation)

  after = _snapshot(simulation)
  for name, value in before.items():
    np.testing.assert_array_equal(after[name], value, err_msg=name)
  for observed, expected in zip(
    [model.cam_pos[front], model.cam_quat[front], model.cam_fovy[front]],
    front_before,
    strict=True,
  ):
    np.testing.assert_array_equal(observed, expected)


def test_geometry_translation_moves_both_views_and_leaves_other_instance_untouched():
  simulation = _simulation()
  other = _simulation()
  model = simulation.model
  translation = np.asarray([0.2, 0.1, 0.3])
  table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "poker_table")
  model.body_pos[table_id] += translation
  simulation.data.qpos[:3] += translation
  mujoco.mj_forward(model, simulation.data)

  video = _Video(simulation)
  configure_pressure_window_video(video, simulation)

  np.testing.assert_allclose(video.camera.lookat, [0.7225, -0.06, 1.141])
  overhead = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
  np.testing.assert_allclose(model.cam_pos[overhead], [0.7225, -0.06, 1.641])
  np.testing.assert_allclose(other.model.cam_pos[overhead], [0.56, 0.0, 1.75])
  assert other.model.cam_fovy[overhead] == 52.0
  assert not hasattr(other, "_pressure_window_original_overhead_camera")


def test_repeated_trials_preserve_camera_original_before_first_configuration():
  simulation = _simulation()
  first = configure_pressure_window_video(_Video(simulation), simulation)
  second = configure_pressure_window_video(_Video(simulation), simulation)

  assert (
    second["overhead"]["original_before_first_configuration"]
    == first["overhead"]["original_before_first_configuration"]
  )
  assert second["overhead"]["previous_before_this_configuration"]["fovy_deg"] == 40.0


@pytest.mark.parametrize(
  "original,replacement,message",
  [
    ('name="poker_table_top"', 'name="other_table"', "poker_table_top"),
    ('name="card_core_geom"', 'name="other_card"', "card_core_geom"),
    ('name="overhead"', 'name="other_overhead"', "overhead"),
    ('name="overhead" pos=', 'name="overhead" mode="track" pos=', "fixed camera mode"),
    (
      'name="overhead" pos=',
      'name="overhead" projection="orthographic" pos=',
      "perspective",
    ),
    ('pos="0.58 -0.18 0.76"', 'pos="0.58 -0.18 0.76" euler="0 0 15"', "axis-aligned"),
    (
      'name="poker_table_top" type="box"',
      'name="poker_table_top" type="ellipsoid"',
      "must be a box",
    ),
  ],
)
def test_unsupported_scene_geometry_is_rejected_without_mutation(
  original, replacement, message
):
  simulation = _simulation(_XML.replace(original, replacement))
  video = _Video(simulation)
  cameras_before = simulation.model.cam_pos.copy()
  with pytest.raises(ValueError, match=message):
    configure_pressure_window_video(video, simulation)
  np.testing.assert_array_equal(simulation.model.cam_pos, cameras_before)
  assert video.camera is None
  assert "pressure_window_camera" not in video._metadata


def test_body_attached_overhead_is_rejected():
  overhead = '<camera name="overhead" pos="0.56 0 1.75" fovy="52"/>'
  xml = _XML.replace(overhead, "").replace(
    '<body pos="0 0 2">', '<body pos="0 0 2">' + overhead
  )
  simulation = _simulation(xml)
  with pytest.raises(ValueError, match="world body"):
    configure_pressure_window_video(_Video(simulation), simulation)


def test_wrong_task_and_mismatched_recorder_are_rejected():
  simulation = _simulation()
  video = _Video(simulation)
  simulation.scene = "pick-place"
  with pytest.raises(ValueError, match="only for the poker-draw"):
    configure_pressure_window_video(video, simulation)
  simulation.scene = "poker-draw"
  with pytest.raises(ValueError, match="this simulation instance"):
    configure_pressure_window_video(_Video(_simulation()), simulation)


def test_closed_recorder_does_not_partially_mutate_model_cameras():
  simulation = _simulation()
  video = _Video(simulation)
  cameras_before = simulation.model.cam_pos.copy()

  def closed(_camera):
    raise RuntimeError("closed")

  video.follow_viewer_camera = closed
  with pytest.raises(RuntimeError, match="closed"):
    configure_pressure_window_video(video, simulation)
  np.testing.assert_array_equal(simulation.model.cam_pos, cameras_before)
  assert not hasattr(simulation, "_pressure_window_original_overhead_camera")
