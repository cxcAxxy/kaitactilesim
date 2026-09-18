from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared import contact_tactile
from kaihand_tactile_env.shared.contact_tactile import (
  GRID_SHAPE,
  SolverDistributedTactileProvider,
)
from kaihand_tactile_env.tactile.layout import TaxelLayout

LINK_NAME = "hand_r_index_link4"
PAD_NAME = f"{LINK_NAME}_tactile_pad_col"


def _small_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(
    f"""
    <mujoco>
      <compiler fusestatic="false"/>
      <worldbody>
        <body name="{LINK_NAME}">
          <geom name="{PAD_NAME}" type="box" size=".01 .01 .002"/>
          <geom name="finger_shell_col" type="box" size=".012 .012 .004"/>
        </body>
        <body name="card">
          <geom name="card_core_geom" type="box" size=".03 .04 .001"/>
        </body>
        <body name="table">
          <geom name="table_geom" type="box" size=".1 .1 .01"/>
        </body>
      </worldbody>
    </mujoco>
    """
  )


def _flat_layout() -> TaxelLayout:
  positions = np.asarray(
    [
      [(column - 2) * 0.002, (row - 3) * 0.002, 0.0]
      for row in range(GRID_SHAPE[0])
      for column in range(GRID_SHAPE[1])
    ],
    dtype=np.float32,
  )
  count = GRID_SHAPE[0] * GRID_SHAPE[1]
  return TaxelLayout(
    body_names=(LINK_NAME,) * count,
    local_pos=positions,
    local_normal=np.tile([0.0, 0.0, 1.0], (count, 1)).astype(np.float32),
    probe_radius=np.full(count, 0.001, dtype=np.float32),
    grid_shape=(GRID_SHAPE,) * count,
  )


class _FakeData:
  def __init__(
    self,
    model: mujoco.MjModel,
    contacts: list[SimpleNamespace],
    *,
    body_rotation: np.ndarray | None = None,
    body_position: np.ndarray | None = None,
  ) -> None:
    self.time = 0.125
    self.contact = contacts
    self.ncon = len(contacts)
    self.xmat = np.tile(np.eye(3).reshape(1, 9), (model.nbody, 1))
    self.xpos = np.zeros((model.nbody, 3), dtype=np.float64)
    body_id = model.body(LINK_NAME).id
    if body_rotation is not None:
      self.xmat[body_id] = np.asarray(body_rotation).reshape(9)
    if body_position is not None:
      self.xpos[body_id] = body_position


def _contact(
  geom1: int,
  geom2: int,
  position: np.ndarray,
  frame: np.ndarray,
) -> SimpleNamespace:
  return SimpleNamespace(
    geom1=geom1,
    geom2=geom2,
    pos=np.asarray(position, dtype=np.float64),
    frame=np.asarray(frame, dtype=np.float64),
  )


def _install_wrenches(
  monkeypatch: pytest.MonkeyPatch, wrenches: list[np.ndarray]
) -> None:
  def fake_contact_force(
    _model: mujoco.MjModel,
    _data: object,
    contact_id: int,
    output: np.ndarray,
  ) -> None:
    output[:] = wrenches[contact_id]

  monkeypatch.setattr(contact_tactile.mujoco, "mj_contactForce", fake_contact_force)


def test_empty_frame_has_stable_layout_basis_and_zero_newtons() -> None:
  model = _small_model()
  provider = SolverDistributedTactileProvider(
    model, _flat_layout(), link_names=(LINK_NAME,)
  )

  sample = provider.read(_FakeData(model, []))

  assert sample.link_names == (LINK_NAME,)
  assert sample.normal_taxel_force_n.shape == (1, 7, 5)
  assert sample.tangent_taxel_force_n.shape == (1, 7, 5, 2)
  assert sample.tangent_basis_local.shape == (1, 2, 3)
  assert sample.normal_axis_local.shape == (1, 3)
  np.testing.assert_allclose(sample.normal_axis_local[0], [0.0, 0.0, 1.0])
  np.testing.assert_allclose(sample.tangent_basis_local[0, 0], [1.0, 0.0, 0.0])
  np.testing.assert_allclose(sample.tangent_basis_local[0, 1], [0.0, 1.0, 0.0])
  assert not sample.contact_count.any()
  assert not sample.normal_force_n.any()
  assert not sample.force_world_n.any()
  assert not sample.tangent_force_n.any()
  assert not sample.normal_taxel_force_n.any()
  assert not sample.tangent_taxel_force_n.any()
  assert sample.assignments == ()
  metadata = provider.metadata()
  assert metadata["force_unit"] == "N"
  assert metadata["taxel_force_unit"] == "N"
  assert metadata["pressure_unit"] is None
  assert "not Pa" in metadata["taxel_force_semantics"]
  json.dumps(metadata)


def test_contact_force_is_sign_corrected_and_distributed_conservatively(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  model = _small_model()
  provider = SolverDistributedTactileProvider(
    model, _flat_layout(), link_names=(LINK_NAME,), kernel_sigma_m=0.002
  )
  card_id = model.geom("card_core_geom").id
  pad_id = model.geom(PAD_NAME).id
  # Rows are contact-frame normal, tangent-1 and tangent-2 axes in world.
  frame = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  )
  contacts = [_contact(card_id, pad_id, np.zeros(3), frame)]
  _install_wrenches(
    monkeypatch, [np.asarray([4.0, 3.0, -2.0, 0.0, 0.0, 0.0])]
  )

  sample = provider.read(_FakeData(model, contacts))

  np.testing.assert_allclose(sample.normal_force_n, [4.0])
  np.testing.assert_allclose(sample.normal_force_world_n[0], [0.0, 0.0, 4.0])
  np.testing.assert_allclose(sample.tangent_force_world_n[0], [3.0, -2.0, 0.0])
  np.testing.assert_allclose(sample.force_world_n[0], [3.0, -2.0, 4.0])
  np.testing.assert_allclose(sample.tangent_force_n[0], [3.0, -2.0])
  assert sample.tangent_load_n[0] == pytest.approx(np.sqrt(13.0))
  assert sample.normal_taxel_force_n.sum() == pytest.approx(4.0)
  np.testing.assert_allclose(
    sample.tangent_taxel_force_n.sum(axis=(1, 2))[0], [3.0, -2.0]
  )
  assert sample.tangent_taxel_load_n.sum() == pytest.approx(np.sqrt(13.0))
  assignment = sample.assignments[0]
  assert assignment.nearest_taxel_flat_index == 17
  assert assignment.nearest_taxel_row_col == (3, 2)
  assert assignment.nearest_taxel_distance_m == pytest.approx(0.0)
  assert assignment.kernel_weights.sum() == pytest.approx(1.0)
  assert assignment.kernel_weights[3, 2] == assignment.kernel_weights.max()


def test_pad_as_geom1_reverses_force_to_report_action_on_finger(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  model = _small_model()
  provider = SolverDistributedTactileProvider(
    model, _flat_layout(), link_names=(LINK_NAME,)
  )
  pad_id = model.geom(PAD_NAME).id
  card_id = model.geom("card_core_geom").id
  frame = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
  )
  _install_wrenches(
    monkeypatch, [np.asarray([4.0, 3.0, -2.0, 0.0, 0.0, 0.0])]
  )

  sample = provider.read(
    _FakeData(model, [_contact(pad_id, card_id, np.zeros(3), frame)])
  )

  np.testing.assert_allclose(sample.force_world_n[0], [-3.0, 2.0, -4.0])
  np.testing.assert_allclose(sample.tangent_force_n[0], [-3.0, 2.0])
  assert sample.normal_force_n[0] == pytest.approx(4.0)


def test_only_exact_pad_to_active_card_contacts_are_admitted(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  model = _small_model()
  provider = SolverDistributedTactileProvider(
    model, _flat_layout(), link_names=(LINK_NAME,)
  )
  pad_id = model.geom(PAD_NAME).id
  shell_id = model.geom("finger_shell_col").id
  card_id = model.geom("card_core_geom").id
  table_id = model.geom("table_geom").id
  frame = np.eye(3)
  contacts = [
    _contact(shell_id, card_id, np.zeros(3), frame),
    _contact(table_id, pad_id, np.zeros(3), frame),
    _contact(card_id, pad_id, np.zeros(3), frame),
  ]
  _install_wrenches(
    monkeypatch,
    [
      np.asarray([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      np.asarray([200.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
  )

  sample = provider.read(_FakeData(model, contacts))

  assert sample.contact_count.tolist() == [1]
  assert sample.normal_force_n.tolist() == [2.0]
  assert [assignment.contact_id for assignment in sample.assignments] == [2]
  with pytest.raises(ValueError, match="enabled card or cylinder"):
    SolverDistributedTactileProvider(
      model,
      _flat_layout(),
      link_names=(LINK_NAME,),
      target_geom_names=("table_geom",),
    )


def test_world_to_link_mapping_and_basis_are_rotation_invariant(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  model = _small_model()
  provider = SolverDistributedTactileProvider(
    model, _flat_layout(), link_names=(LINK_NAME,)
  )
  rotation = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
  )
  translation = np.asarray([0.4, -0.2, 0.7])
  local_taxel = provider.taxel_positions_local_m[0, 4 * 5 + 3]
  position_world = translation + rotation @ local_taxel
  # normal=z, tangent-1=world y and tangent-2=-world x.
  frame = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
  )
  card_id = model.geom("card_core_geom").id
  pad_id = model.geom(PAD_NAME).id
  _install_wrenches(
    monkeypatch, [np.asarray([2.0, 5.0, 0.0, 0.0, 0.0, 0.0])]
  )

  sample = provider.read(
    _FakeData(
      model,
      [_contact(card_id, pad_id, position_world, frame)],
      body_rotation=rotation,
      body_position=translation,
    )
  )

  np.testing.assert_allclose(sample.tangent_basis_world[0, 0], [0.0, 1.0, 0.0])
  np.testing.assert_allclose(sample.tangent_basis_world[0, 1], [-1.0, 0.0, 0.0])
  np.testing.assert_allclose(sample.tangent_force_n[0], [5.0, 0.0])
  assignment = sample.assignments[0]
  assert assignment.nearest_taxel_row_col == (4, 3)
  assert assignment.nearest_taxel_distance_m == pytest.approx(0.0, abs=1.0e-14)

