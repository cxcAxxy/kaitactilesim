"""Native-pose recording checks using tiny MuJoCo models, never the robot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared import contact_tactile, recording, taskspace_recording
from kaihand_tactile_env.shared.config import CameraConfig, WorkcellConfig
from kaihand_tactile_env.shared.recording import (
  _CONTACT_FORCE_SHAPES,
  EpisodeRecorder,
)
from kaihand_tactile_env.shared.taskspace_recording import TaskspaceCapture
from kaihand_tactile_env.tasks.poker_draw import friction, mid_full


def _xml() -> str:
  hands = []
  for side, offset in (("l", -0.1), ("r", 0.1)):
    sites = [f'<site name="hand_{side}_base_link_site" size=".001"/>']
    for index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
      link = 6 if finger == "thumb" else 4
      sites.append(
        f'<site name="hand_{side}_{finger}_link{link}_site" '
        f'pos="{0.01 * (index + 1)} .002 .003" quat=".70710678 .70710678 0 0" size=".001"/>'
      )
      if side == "r" and finger != "thumb":
        sites.append(
          f'<geom name="hand_r_{finger}_link4_tactile_pad_col" '
          f'type="sphere" size=".001" pos="{index * 0.01} 0 .1" contype="0" conaffinity="0"/>'
        )
    hands.append(
      f'<body name="hand_{side}" pos="{offset} 0 .5" quat=".70710678 0 0 .70710678">'
      f'<joint name="joint_{side}" type="hinge" axis="0 1 0"/>'
      '<geom type="sphere" size=".005" mass=".1" contype="0" conaffinity="0"/>'
      + "".join(sites)
      + "</body>"
    )
  return (
    '<mujoco model="tiny_taskspace"><option timestep=".002" impratio="1"/>'
    '<worldbody><geom name="poker_table_top" type="plane" size="1 1 .1"/>'
    + "".join(hands)
    + '<body name="card" pos="0 0 .2"><freejoint name="card_free"/>'
    '<geom name="card_core_geom" type="box" size=".03 .04 .001" mass=".006" '
    'friction="1.4 .005 .0005" priority="1" solref=".002 1"/></body>'
    '<camera name="head" pos="0 0 1"/></worldbody>'
    '<contact><pair name="poker_table_card_friction_pair" geom1="card_core_geom" '
    'geom2="poker_table_top" friction="1 1 .005 .0005 .0005" solref=".002 1"/></contact>'
    '<actuator><motor name="motor_l" joint="joint_l"/>'
    '<motor name="motor_r" joint="joint_r"/></actuator></mujoco>'
  )


def _tiny_sim(tmp_path):
  path = tmp_path / "tiny.xml"
  path.write_text(_xml(), encoding="utf-8")
  model = mujoco.MjModel.from_xml_path(str(path))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  sim = SimpleNamespace(model=model, data=data, model_path=path, observation_time=0.0)
  return sim


def test_native_capture_preserves_real_site_rotation_and_does_not_forward(
  tmp_path, monkeypatch
):
  sim = _tiny_sim(tmp_path)
  capture = TaskspaceCapture(sim)
  # FK stays at the last solver time even when integrated qpos is newer.
  sim.data.qpos[0] = 0.3
  sim.data.time = 0.002
  sim.observation_time = 0.0
  arrays = {
    name: getattr(sim.data, name).copy()
    for name in (
      "qpos",
      "qvel",
      "ctrl",
      "qacc_warmstart",
      "site_xpos",
      "site_xmat",
    )
  }

  def forbidden(*_):
    raise AssertionError("capture must not recompute FK or dynamics")

  monkeypatch.setattr(mujoco, "mj_forward", forbidden)
  stamp, wrists, fingertips = capture.read()
  assert stamp == 0.0
  assert wrists.shape == (2, 4, 4)
  assert fingertips.shape == (2, 5, 4, 4)
  for poses, ids in ((wrists, capture.wrist_ids), (fingertips, capture.finger_ids)):
    np.testing.assert_allclose(poses[..., :3, 3], arrays["site_xpos"][ids])
    np.testing.assert_allclose(
      poses[..., :3, :3], arrays["site_xmat"][ids].reshape(*ids.shape, 3, 3)
    )
    np.testing.assert_allclose(
      poses[..., 3, :], np.broadcast_to([0, 0, 0, 1], (*ids.shape, 4))
    )
    assert not np.allclose(poses[..., :3, :3], np.eye(3))
    np.testing.assert_allclose(np.linalg.det(poses[..., :3, :3]), 1.0)
  assert capture.finger_names[0][-1] == "hand_l_pinky_link4_site"
  assert capture.finger_names[1][0] == "hand_r_thumb_link6_site"
  fingertips.fill(0)  # Returned matrices cannot alias the simulation cache.
  for name, expected in arrays.items():
    np.testing.assert_array_equal(getattr(sim.data, name), expected)


def test_native_capture_requires_a_clock_not_an_assumed_state_timestamp(tmp_path):
  sim = _tiny_sim(tmp_path)
  del sim.observation_time
  with pytest.raises(ValueError, match="clock"):
    TaskspaceCapture(sim)


def test_middle_step_labels_last_preintegration_solver_time(monkeypatch):
  simulation = object.__new__(mid_full.MidForcePokerSimulation)
  simulation.data = SimpleNamespace(time=1.0)
  simulation.timestep = 0.002
  calls = []

  def step(sim, count):
    calls.append((sim.data.time, count))
    sim.data.time += count * sim.timestep

  monkeypatch.setattr(mid_full.ForceLimitedPokerSimulation, "step", step)
  assert simulation.observation_time == 1.0
  simulation.step(3)
  assert [count for _, count in calls] == [3]
  assert simulation.data.time == pytest.approx(1.006)
  assert simulation.observation_time == pytest.approx(1.004)
  simulation.data.time = 0.0  # reset cannot retain a future cache timestamp
  assert simulation.observation_time == 0.0


def test_shared_middle_factory_matches_accepted_contact_setup(tmp_path, monkeypatch):
  sim = _tiny_sim(tmp_path)
  sim.timestep = 0.002
  reference = _tiny_sim(tmp_path)
  reference.timestep = 0.002
  events = []

  @contextmanager
  def wrapper(base, coefficient):
    assert base == sim.model_path and coefficient == 1.0
    events.append("enter")
    try:
      yield sim.model_path
    finally:
      events.append("exit")

  def construct(path, **kwargs):
    assert path == sim.model_path
    assert kwargs == {"scene": "poker-draw", "add_genesis_probes": False}
    return sim

  monkeypatch.setattr(friction, "model_with_table_card_friction", wrapper)
  monkeypatch.setattr(mid_full, "MidForcePokerSimulation", construct)
  path = (
    Path(__file__).resolve().parents[1]
    / "scripts/workcell/experiment_poker_pressure_window.py"
  )
  spec = importlib.util.spec_from_file_location("_taskspace_reference_contact", path)
  assert spec is not None and spec.loader is not None
  helper = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(helper)
  reference_metadata = helper._configure_contact_model(
    reference, mid_full.MID_FORCE_SETTINGS
  )
  before = sim.data.qpos.copy()
  with mid_full.middle_force_simulation(sim.model_path) as (actual, metadata):
    assert actual is sim
    assert events == ["enter"]
    for key in metadata.keys() & reference_metadata.keys():
      assert metadata[key] == reference_metadata[key]
    np.testing.assert_array_equal(sim.model.pair_solref, reference.model.pair_solref)
    np.testing.assert_array_equal(sim.model.geom_solref, reference.model.geom_solref)
    np.testing.assert_array_equal(
      sim.model.pair_friction, reference.model.pair_friction
    )
    np.testing.assert_array_equal(
      sim.model.actuator_gainprm, reference.model.actuator_gainprm
    )
    assert sim.model.opt.impratio == reference.model.opt.impratio == 100
    assert not metadata["add_genesis_probes"]
    np.testing.assert_array_equal(sim.data.qpos, before)
  assert events == ["enter", "exit"]


class _Proxy:
  source = "solver_contact_proxy_v1"
  available = True
  link_names = ("tiny_right_pad",)

  def read(self, _data):
    return SimpleNamespace(
      contact=np.zeros(1, dtype=bool),
      normal_force=np.zeros(1),
      contact_count=np.zeros(1, dtype=np.int32),
      **{
        name: np.zeros((1, 3))
        for name in ("force_world", "torque_world", "force_local", "centroid_world")
      },
    )


class _Contact:
  source = "tiny_contact_force"
  link_names = ("tiny_right_pad",)
  taxel_positions_local_m = np.zeros((1, 35, 3))
  normal_axis_local = np.array([[0, 0, 1]])
  tangent_basis_local = np.array([[[1, 0, 0], [0, 1, 0]]])

  def __init__(self, *_args, **_kwargs):
    pass

  def metadata(self):
    return {"source": self.source}

  def read(self, _data):
    return SimpleNamespace(
      **{name: np.zeros((1, *shape)) for name, shape in _CONTACT_FORCE_SHAPES.items()}
    )


class _Renderer:
  def calibration(self, _data, _camera):
    return SimpleNamespace(
      intrinsic=np.eye(3), fovy_degrees=45, world_from_camera=np.eye(4)
    )

  def capture(self, _data, camera):
    return {"rgb": np.zeros((camera.height, camera.width, 3), dtype=np.uint8)}


@pytest.mark.parametrize("capture_taskspace", [True, False])
@pytest.mark.parametrize("automatic_renderer", [True, False])
def test_recorder_writes_explicit_clocks_actual_actuators_and_budget_flags(
  tmp_path, monkeypatch, capture_taskspace, automatic_renderer
):
  sim = _tiny_sim(tmp_path)
  sim.scene = "poker-draw"
  sim.genesis_probe_layout = None
  sim.object_names = sim.model_object_names = ("card",)
  sim.joint_names = ("joint_l", "joint_r")
  sim._qvel_address = {"joint_l": 0, "joint_r": 1}
  sim.command_state = lambda: (np.ones(14), np.ones(2), ("joint_l", "joint_r"))
  sim.joint_state = lambda: (sim.joint_names, sim.data.qpos[:2], sim.data.qvel[:2])
  sim.object_pose = lambda _: np.array([0, 0, 0.2, 1, 0, 0, 0])
  sim.object_twist = lambda _: np.zeros(6)
  sim.drive_limit_n = 4.0
  sim.drive_state = lambda: {"drive_requested_fx_n": -5.0, "drive_actual_fx_n": -4.0}
  sim.timestep = 0.002
  monkeypatch.setattr(contact_tactile, "SolverDistributedTactileProvider", _Contact)
  camera = CameraConfig("head", 4, 3, depth=False, segmentation=False)
  config = WorkcellConfig(
    model_path=sim.model_path, cameras=(camera,), tactile_provider=_Proxy.source
  )
  output = tmp_path / "episode.h5"
  renderer_calls = []

  class AutomaticRenderer(_Renderer):
    def __init__(self, model, cameras, *, shadows=True):
      assert model is sim.model and cameras == (camera,)
      renderer_calls.append(shadows)

    def close(self):
      pass

  monkeypatch.setattr(recording, "WorkcellRenderer", AutomaticRenderer)
  with EpisodeRecorder(
    output,
    sim,
    config,
    tactile_provider=_Proxy(),
    renderer=None if automatic_renderer else _Renderer(),
    capture_taskspace=capture_taskspace,
  ) as recorder:
    sim.data.ctrl[:] = [0.3, -0.7]
    recorder.record_initial()
    sim.data.time = 0.012
    sim.observation_time = 0.010
    sim.data.ctrl[:] = [0.2, -0.6]
    sim.drive_limit_n = None
    recorder._record_state("inspection")
    recorder._capture_cameras()
  with h5py.File(output, "r") as file:
    assert renderer_calls == ([not capture_taskspace] if automatic_renderer else [])
    np.testing.assert_allclose(file["state/timestamp"], [0, 0.012])
    np.testing.assert_allclose(file["cameras/head/timestamp"], [0, 0.012])
    if not capture_taskspace:
      assert "render_shadows" not in file.attrs
      assert "taskspace_capture_source_sha256" not in file.attrs
      assert "commands/actuator_control" not in file
      assert "cameras/head/pose_timestamp" not in file
      assert "tactile_contact_force/timestamp" not in file
      assert (
        file["tactile_contact_force"].attrs["timestamp_reference"] == "/state/timestamp"
      )
      return
    assert not file.attrs["render_shadows"]
    expected_hash = hashlib.sha256(
      Path(taskspace_recording.__file__).read_bytes()
    ).hexdigest()
    assert file.attrs["taskspace_capture_source_sha256"] == expected_hash
    np.testing.assert_allclose(
      file["commands/actuator_control"], [[0.3, -0.7], [0.2, -0.6]]
    )
    assert list(file["commands/actuator_names"].asstr()[:]) == ["motor_l", "motor_r"]
    np.testing.assert_array_equal(file["commands/drive_budget_active"], [True, False])
    np.testing.assert_allclose(file["commands/drive_requested_fx_n"], [-5, 0])
    np.testing.assert_allclose(file["commands/drive_actual_fx_n"], [-4, 0])
    np.testing.assert_allclose(file["commands/drive_limit_n"], [4, 0])
    np.testing.assert_allclose(file["tactile_contact_force/timestamp"], [0, 0.010])
    np.testing.assert_allclose(file["cameras/head/pose_timestamp"], [0, 0.010])
    group = file["cameras/head"]
    assert group["world_from_wrist"].shape == (2, 2, 4, 4)
    assert group["world_from_fingertip"].shape == (2, 2, 5, 4, 4)
    assert json.loads(group.attrs["finger_names_json"])[-1] == "little"
    assert group.attrs["taskspace_schema"] == "kaihand-native-site-se3-v1"
    assert not np.allclose(group["world_from_fingertip"][0, 0, 0, :3, :3], np.eye(3))


@pytest.mark.parametrize("capture_taskspace", [True, False])
def test_progress_flush_is_middle_only_once_per_five_simulated_seconds(
  monkeypatch, capsys, capture_taskspace
):
  recorder = object.__new__(EpisodeRecorder)
  sim = SimpleNamespace(data=SimpleNamespace(time=4.999), timestep=0.002)
  recorder.sim = sim
  recorder.config = SimpleNamespace(cameras=())
  recorder.taskspace_capture = object() if capture_taskspace else None
  recorder._next_state_time = float("inf")
  recorder._next_camera_time = float("inf")
  recorder._next_progress_time = 5.0
  recorder._recording_started = 0.0
  recorder._state_write_seconds = 1.0
  recorder._camera_write_seconds = 2.0
  recorder._state_samples = 500
  recorder._camera_samples = {"head": 50}
  flushed = []
  recorder._file = SimpleNamespace(flush=lambda: flushed.append(sim.data.time))
  monkeypatch.setattr(recording.time, "monotonic", lambda: 10.0)
  recorder.observe(sim, "slide_card")
  assert not flushed and not capsys.readouterr().out
  sim.data.time = 5.0
  recorder.observe(sim, "slide_card")
  assert flushed == ([5.0] if capture_taskspace else [])
  first = capsys.readouterr().out
  assert bool(first) == capture_taskspace
  if capture_taskspace:
    assert "sim=5.000s" in first and "phase=slide_card" in first
    assert "states=500" in first and "camera_write=2.0s" in first
    assert recorder._next_progress_time == 10.0
  sim.data.time = 5.1
  recorder.observe(sim, "slide_card")
  assert not capsys.readouterr().out
  sim.data.time = 10.0
  recorder.observe(sim, "edge_hold")
  assert flushed == ([5.0, 10.0] if capture_taskspace else [])


@pytest.mark.parametrize("shadows", [True, False])
def test_renderer_shadow_flag_is_visual_only_and_legacy_default_is_on(
  tmp_path, monkeypatch, shadows
):
  from kaihand_tactile_env.shared.rendering import WorkcellRenderer

  sim = _tiny_sim(tmp_path)
  flags = np.ones(int(mujoco.mjtRndFlag.mjNRNDFLAG), dtype=np.uint8)
  camera = CameraConfig("head", 4, 3)
  dummy = SimpleNamespace(scene=SimpleNamespace(flags=flags), close=lambda: None)
  monkeypatch.setattr(mujoco, "Renderer", lambda *args, **kwargs: dummy)
  before = sim.data.qpos.copy()
  arguments = {} if shadows else {"shadows": False}
  with WorkcellRenderer(sim.model, (camera,), **arguments):
    assert flags[mujoco.mjtRndFlag.mjRND_SHADOW] == int(shadows)
    np.testing.assert_array_equal(sim.data.qpos, before)
