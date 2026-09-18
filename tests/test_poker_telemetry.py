from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.tasks.poker_draw import telemetry
from kaihand_tactile_env.tasks.poker_draw.telemetry import (
  FINGERS,
  PokerFrictionTelemetry,
  cached_point_velocity,
)


def _small_model() -> mujoco.MjModel:
  fingers = "".join(
    f'<body name="hand_r_{finger}_link4" pos="{index + 1} 0 1">'
    "<freejoint/>"
    f'<geom name="hand_r_{finger}_link4_tactile_pad_col" '
    'type="box" size=".01 .01 .002" mass=".01"/></body>'
    for index, finger in enumerate(FINGERS)
  )
  return mujoco.MjModel.from_xml_string(
    '<mujoco><compiler fusestatic="false"/>'
    '<option timestep=".002" integrator="implicitfast"/><worldbody>'
    f'{fingers}<body name="card" pos="0 0 1"><freejoint/>'
    '<geom name="card_core_geom" type="box" size=".03 .04 .001" mass=".006"/>'
    '</body><geom name="poker_table_top" type="box" size=".1 .1 .01"/>'
    '<body name="root" pos="0 0 2"><freejoint/>'
    '<geom size=".04" pos=".2 0 0"/><body name="child" pos=".5 .1 0">'
    '<joint axis="0 1 0"/><geom size=".03" pos="0 .2 .3"/>'
    "</body></body></worldbody></mujoco>"
  )


def _fixture(monkeypatch: pytest.MonkeyPatch, *, flat_basis: bool = True):
  model = _small_model()
  data = SimpleNamespace(
    time=0.002,
    contact=[],
    ncon=0,
    xpos=np.zeros((model.nbody, 3)),
    xmat=np.tile(np.eye(3).reshape(1, 9), (model.nbody, 1)),
    xquat=np.tile([1.0, 0.0, 0.0, 0.0], (model.nbody, 1)),
    cvel=np.zeros((model.nbody, 6)),
    subtree_com=np.zeros((model.nbody, 3)),
  )
  if flat_basis:
    monkeypatch.setattr(
      telemetry,
      "SolverDistributedTactileProvider",
      lambda *args, **kwargs: SimpleNamespace(
        tangent_basis_local=np.tile(np.eye(3)[:2], (4, 1, 1))
      ),
    )
  wrenches: list[np.ndarray] = []

  def contact_force(_model, _data, contact_id, output):
    output[:] = wrenches[contact_id]

  monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
  sim = SimpleNamespace(model=model, data=data, timestep=0.002)
  monitor = PokerFrictionTelemetry(sim, target_force_n=0.35)
  return model, data, wrenches, monitor


def _add_contact(
  model,
  data,
  wrenches,
  *,
  finger="index",
  force=(0.35, 0.02, -0.03),
  position=(0.0, 0.0, 0.0),
  swap=False,
  table=False,
):
  card = model.geom("card_core_geom").id
  other = model.geom(
    "poker_table_top" if table else f"hand_r_{finger}_link4_tactile_pad_col"
  ).id
  data.contact.append(
    SimpleNamespace(
      geom1=other if swap else card,
      geom2=card if swap else other,
      pos=np.asarray(position, dtype=float),
      frame=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
  )
  data.ncon = len(data.contact)
  wrenches.append(np.asarray([*force, 0.0, 0.0, 0.0]))


def test_cached_velocity_matches_jacobian_including_root_subtree_com() -> None:
  model = _small_model()
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-0.4, 0.5, model.nv)
  mujoco.mj_forward(model, data)
  for name in ("card", "hand_r_index_link4", "child"):
    body_id = model.body(name).id
    point = np.asarray(data.xpos[body_id]) + np.array([0.017, -0.023, 0.011])
    jacobian = np.zeros((3, model.nv))
    rotation_jacobian = np.zeros_like(jacobian)
    mujoco.mj_jac(model, data, jacobian, rotation_jacobian, point, body_id)
    np.testing.assert_allclose(
      cached_point_velocity(model, data, body_id, point),
      jacobian @ data.qvel,
      atol=1.0e-12,
    )


def test_cached_velocity_does_not_use_post_integration_qvel() -> None:
  model = _small_model()
  data = mujoco.MjData(model)
  data.qvel[:] = np.linspace(-0.4, 0.5, model.nv)
  mujoco.mj_forward(model, data)
  body_id = model.body("child").id
  point = data.xpos[body_id].copy()
  original = cached_point_velocity(model, data, body_id, point)
  data.qvel[:] = 100.0
  np.testing.assert_array_equal(
    cached_point_velocity(model, data, body_id, point), original
  )


def test_force_sign_table_action_and_zero_contact_nan_velocity(monkeypatch) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches)
  _add_contact(model, data, wrenches, finger="middle", swap=True)
  _add_contact(model, data, wrenches, table=True, swap=True, force=(1.0, -0.1, 0.2))
  row = monitor.sample("slide_card")
  assert row["index_fn_n"] == pytest.approx(0.35)
  assert row["index_ft_x_n"] == pytest.approx(0.02)
  assert row["index_ft_y_n"] == pytest.approx(-0.03)
  assert row["index_ft_n"] == pytest.approx(np.hypot(0.02, 0.03))
  assert row["index_force_world_x_n"] == pytest.approx(0.02)
  assert row["index_force_world_y_n"] == pytest.approx(-0.03)
  assert row["index_force_world_z_n"] == pytest.approx(0.35)
  assert row["middle_ft_x_n"] == pytest.approx(-0.02)
  assert row["middle_ft_y_n"] == pytest.approx(0.03)
  assert row["table_normal_force_n"] == pytest.approx(1.0)
  assert row["table_force_on_card_x_n"] == pytest.approx(-0.1)
  assert row["table_force_on_card_y_n"] == pytest.approx(0.2)
  assert row["table_force_on_card_z_n"] == pytest.approx(1.0)
  assert row["ring_contact"] == 0
  assert np.isnan(row["ring_slip_speed_m_s"])
  assert row["ring_cumulative_slip_m"] == 0.0
  assert row["solver_time_s"] == pytest.approx(0.0)
  assert monitor.metadata()["card_mass_kg"] == pytest.approx(0.006)
  assert monitor.metadata()["gravity_world_m_s2"] == [0.0, 0.0, -9.81]
  json.dumps(monitor.summary(), allow_nan=False)
  json.dumps(monitor.metadata(), allow_nan=False)


def test_basis_and_signed_shear_match_existing_tactile_provider(monkeypatch) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch, flat_basis=False)
  angle = 0.47
  rotation = np.array(
    [
      [np.cos(angle), -np.sin(angle), 0.0],
      [np.sin(angle), np.cos(angle), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  data.xmat[model.body("hand_r_index_link4").id] = rotation.reshape(9)
  _add_contact(model, data, wrenches, force=(0.3, 0.07, -0.05))
  provider = SolverDistributedTactileProvider(
    model, link_names=monitor.link_names, target_geom_names=("card_core_geom",)
  )
  sample = provider.read(data)
  row = monitor.sample("slide_card")
  np.testing.assert_allclose(
    [row["index_ft_x_n"], row["index_ft_y_n"]], sample.tangent_force_n[0]
  )
  assert row["index_fn_n"] == pytest.approx(sample.normal_force_n[0])
  np.testing.assert_array_equal(
    monitor.metadata()["tangent_basis_local"], provider.tangent_basis_local
  )


def test_slip_uses_both_angular_velocities_and_excludes_normal_motion(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, position=(0.03, -0.02, 0.0))
  finger = model.body("hand_r_index_link4").id
  card = model.body("card").id
  data.cvel[finger] = [0.0, 0.0, 2.0, 0.1, 0.2, 0.9]
  data.cvel[card] = [0.0, 0.0, 1.0, 0.1, 0.2, -0.1]
  row = monitor.sample("slide_card")
  assert row["index_slip_vx_m_s"] == pytest.approx(0.02)
  assert row["index_slip_vy_m_s"] == pytest.approx(0.03)
  assert row["index_slip_speed_m_s"] == pytest.approx(np.hypot(0.02, 0.03))
  assert row["index_slip_world_vz_m_s"] == pytest.approx(0.0)


def test_co_moving_bodies_are_not_slipping_while_table_motion_exists(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, position=(0.04, 0.0, 0.0))
  for name in ("hand_r_index_link4", "card"):
    data.cvel[model.body(name).id] = [0.0, 0.0, 1.0, -0.015, 0.0, 0.0]
  row = monitor.sample("slide_card")
  assert row["index_slip_speed_m_s"] == pytest.approx(0.0)
  assert row["card_vx_m_s"] == pytest.approx(-0.015)


def test_speed_and_shear_load_do_not_cancel_across_multiple_contacts(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, position=(0.01, 0.0, 0.0), force=(0.1, 0.02, 0.0))
  _add_contact(
    model, data, wrenches, position=(-0.01, 0.0, 0.0), force=(0.1, -0.02, 0.0)
  )
  data.cvel[model.body("hand_r_index_link4").id, 2] = 1.0
  row = monitor.sample("slide_card")
  assert row["index_slip_vy_m_s"] == pytest.approx(0.0)
  assert row["index_slip_speed_m_s"] == pytest.approx(0.01)
  assert row["index_slip_max_speed_m_s"] == pytest.approx(0.01)
  assert row["index_ft_n"] == pytest.approx(0.0)
  assert row["index_ft_load_n"] == pytest.approx(0.04)


def test_slip_integrates_slide_and_edge_but_never_across_contact_loss(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, force=(0.35, 0.0, 0.0))
  data.cvel[model.body("hand_r_index_link4").id, 3] = 0.01
  monitor.sample("four_finger_press")
  data.time += 0.002
  row = monitor.sample("slide_card")
  assert row["index_cumulative_slip_m"] == 0.0
  data.time += 0.002
  row = monitor.sample("slide_card")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00002)
  data.time += 0.002
  row = monitor.sample("edge_hold")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00004)
  assert monitor.summary()["edge_hold"]["fingers"]["index"]["cumulative_slip_m"] == 0.0
  wrenches[0][:] = 0.0
  data.time += 0.002
  row = monitor.sample("edge_hold")
  assert np.isnan(row["index_slip_speed_m_s"])
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00004)
  assert row["index_contact_gap_s"] == pytest.approx(0.002)
  data.time += 0.002
  row = monitor.sample("edge_hold")
  assert row["index_contact_gap_s"] == pytest.approx(0.004)
  wrenches[0][0] = 0.35
  data.contact[0].pos[:] = 10.0  # Recontact location changes must not be integrated.
  data.time += 0.002
  row = monitor.sample("edge_hold")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00004)
  assert row["index_contact_gap_s"] == 0.0
  data.time += 0.002
  row = monitor.sample("edge_hold")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00006)
  assert row["index_cumulative_slip_x_m"] == pytest.approx(0.00006)
  summary = monitor.summary()
  combined = summary["slide_and_edge"]["fingers"]["index"]
  assert combined["contact_loss_events"] == 1
  assert combined["maximum_contact_gap_s"] == pytest.approx(0.004)
  assert combined["unloaded_time_s"] == pytest.approx(0.004)
  assert summary["press"]["sample_count"] == 1
  assert summary["slide_card"]["sample_count"] == 2
  assert summary["edge_hold"]["sample_count"] == 5
  assert summary["slide_and_edge"]["sample_count"] == 7
  data.time += 0.002
  row = monitor.sample("lift_card")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00006)
  assert row["index_contact_gap_s"] == 0.0


def test_load_threshold_and_pressure_quality_are_distinct(monkeypatch) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, force=(0.01, 0.0, 0.0))
  row = monitor.sample("slide_card")
  assert row["index_contact"] == 1
  assert row["index_pressure_qualified"] == 0
  assert row["index_contact_gap_s"] == 0.0
  assert row["index_pressure_gap_s"] == pytest.approx(0.002)
  wrenches[0][0] = telemetry.CONTACT_FORCE_THRESHOLD_N
  data.time += 0.002
  row = monitor.sample("slide_card")
  assert row["index_contact"] == 0
  assert np.isnan(row["index_slip_speed_m_s"])


def test_missing_samples_do_not_bridge_slip_or_classify_unknown_load(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, force=(0.35, 0.0, 0.0))
  data.cvel[model.body("hand_r_index_link4").id, 3] = 0.01
  monitor.sample("slide_card")
  data.time += 0.1
  row = monitor.sample("slide_card")
  assert row["index_cumulative_slip_m"] == 0.0
  assert row["sampling_gap_s"] == pytest.approx(0.098)
  summary = monitor.summary()
  assert not summary["physics_step_sampling_complete"]
  assert summary["unobserved_time_s"] == pytest.approx(0.098)
  assert summary["slide_and_edge"]["duration_s"] == pytest.approx(0.004)
  assert summary["slide_and_edge"]["fingers"]["index"]["first_sustained_slip"] is None
  data.time += 0.002
  row = monitor.sample("slide_card")
  assert row["index_cumulative_slip_m"] == pytest.approx(0.00002)


def test_sustained_slip_records_onset_not_detection_and_resets_on_separation(
  monkeypatch,
) -> None:
  model, data, wrenches, monitor = _fixture(monkeypatch)
  _add_contact(model, data, wrenches, force=(0.35, 0.0, 0.0))
  data.cvel[model.body("hand_r_index_link4").id, 3] = 0.002
  card = model.body("card").id
  for _ in range(5):
    monitor.sample("slide_card")
    data.time += 0.002
  wrenches[0][0] = 0.0
  monitor.sample("slide_card")
  data.time += 0.002
  wrenches[0][0] = 0.35
  data.xpos[card] = [0.52, -0.16, 0.84]
  onset = data.time
  for _ in range(10):
    monitor.sample("slide_card")
    data.time += 0.002
  stats = monitor.summary()["slide_and_edge"]["fingers"]["index"]
  assert stats["first_sustained_slip"] is None
  data.xpos[card, 0] -= 0.01
  monitor.sample("edge_hold")
  stats = monitor.summary()["slide_and_edge"]["fingers"]["index"]
  assert stats["first_sustained_slip"] == {
    "time_s": onset,
    "card_position_m": [0.52, -0.16, 0.84],
    "detected_time_s": data.time,
  }
  assert (
    monitor.summary()["edge_hold"]["fingers"]["index"]["first_sustained_slip"] is None
  )


def test_read_only_sampling_and_monotonic_time(monkeypatch) -> None:
  model = _small_model()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  data.time = 0.002
  monitor = PokerFrictionTelemetry(
    SimpleNamespace(model=model, data=data), target_force_n=0.35
  )
  arrays = (
    data.qpos,
    data.qvel,
    data.ctrl,
    data.xpos,
    data.xquat,
    data.cvel,
    data.qacc,
  )
  before = [array.copy() for array in arrays]

  def fail_forward(*args):
    raise AssertionError("telemetry must not call mj_forward")

  monkeypatch.setattr(mujoco, "mj_forward", fail_forward)
  row = monitor.sample("approach")
  assert row["phase"] == "approach"
  for array, original in zip(arrays, before, strict=True):
    np.testing.assert_array_equal(array, original)
  with pytest.raises(ValueError, match="increasing"):
    monitor.sample("approach")
  data.time += 0.006
  monitor.sample("approach")
  assert not monitor.summary()["physics_step_sampling_complete"]


@pytest.mark.parametrize("target", [0.0, -1.0, np.nan, np.inf])
def test_invalid_target_rejected_before_accessing_simulation(target) -> None:
  with pytest.raises(ValueError, match="finite and positive"):
    PokerFrictionTelemetry(None, target_force_n=target)
