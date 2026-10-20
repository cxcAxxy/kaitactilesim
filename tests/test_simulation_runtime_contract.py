import json
from types import SimpleNamespace

import mujoco
import pytest
from kaihand_tactile_env.shared.runtime_contract import simulation_runtime_contract


def test_contract_freezes_actual_compiled_contact_settings_and_provider_units():
  model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.002"/><worldbody>
      <geom name="floor" type="plane" size="1 1 0.1"/>
      <body pos="0 0 0.2"><freejoint/><geom name="ball" size="0.1"/></body>
    </worldbody><contact><pair name="pad_contact" geom1="floor" geom2="ball"
      condim="6" friction="0.8 0.7 0.01 0.001 0.002"/></contact></mujoco>
  ''')
  # These modifications are absent from the XML; the recorder must see them.
  model.opt.timestep = 0.001
  model.opt.noslip_iterations = 8
  model.opt.impratio = 100
  model.pair_friction[0, 0] = 0.9
  simulation = SimpleNamespace(model=model, scene="usb-insert")
  probe = SimpleNamespace(
    source="genesis_probe_v1", force_unit="genesis_agg_force_depth_x_1e4",
    contact_threshold_m=5e-5, release_debounce_steps=10,
  )
  spatial = SimpleNamespace(
    source="solver_distributed", force_unit="N", kernel_sigma_m=0.003,
    metadata=lambda: {"kernel": "normalized_gaussian_3d_taxel_distance"},
  )
  contract = simulation_runtime_contract(
    simulation, tactile_provider=probe, contact_force_provider=spatial
  )
  assert json.loads(json.dumps(contract, allow_nan=False)) == contract
  assert contract["physics"]["timestep_s"] == 0.001
  assert contract["physics"]["noslip_iterations"] == 8
  assert contract["physics"]["impratio"] == 100
  assert contract["explicit_contact_pairs"][0]["friction"][0] == pytest.approx(0.9)
  assert contract["explicit_contact_pairs"][0]["geom2"] == "ball"
  assert contract["tactile"]["force_unit"] == "genesis_agg_force_depth_x_1e4"
  assert contract["tactile"]["release_debounce_seconds"] == pytest.approx(0.01)
  assert contract["contact_force"]["force_unit"] == "N"
  assert contract["contact_force"]["kernel_sigma_m"] == 0.003


def test_unspecified_provider_properties_remain_unknown():
  model = mujoco.MjModel.from_xml_string("<mujoco/>")
  snapshot = simulation_runtime_contract(
    SimpleNamespace(model=model, scene="pick-place"),
    tactile_provider=SimpleNamespace(source="test"),
  )
  assert snapshot["tactile"]["force_unit"] is None
  assert "release_debounce_steps" not in snapshot["tactile"]
  assert snapshot["explicit_contact_pairs"] == []
