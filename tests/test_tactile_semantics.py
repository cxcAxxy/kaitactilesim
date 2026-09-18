from __future__ import annotations

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.workcell.config import WorkcellConfig
from kaihand_tactile_env.workcell.recording import SCHEMA_VERSION, EpisodeRecorder
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation
from kaihand_tactile_env.workcell.tactile import (
  GenesisProbeTactileProvider,
  LinkTactileSample,
)


def test_release_debounce_does_not_extend_instantaneous_numeric_streams() -> None:
  simulation = ArmHandSimulation()
  provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  probe_index = int(np.flatnonzero(provider.layout.probe_radius > 0.0)[0])
  link_index = int(provider._probe_link_index[probe_index])
  target_id = simulation.model.geom("cylinder_geom").id

  provider.probe_contact[probe_index] = True
  provider.probe_depth[probe_index] = 1.0e-3
  provider.probe_target_geom_id[probe_index] = target_id
  provider._release_until[probe_index] = (
    float(simulation.data.time) + simulation.timestep
  )

  decaying = provider.read(simulation.data)
  assert provider.probe_contact[probe_index]
  assert not provider.probe_contact_instantaneous[probe_index]
  assert provider.probe_depth[probe_index] == 0.0
  assert decaying.contact[link_index]
  assert decaying.normal_force[link_index] == 0.0
  assert not decaying.centroid_valid[link_index]
  assert np.isnan(decaying.centroid_world[link_index]).all()
  assert provider.probe_target_geom_id[probe_index] == -1

  # Exactly at expiry, the debounced Boolean stream also returns to false.
  provider._release_until[probe_index] = float(simulation.data.time)
  expired = provider.read(simulation.data)
  assert not provider.probe_contact[probe_index]
  assert not provider.probe_contact_instantaneous[probe_index]
  assert provider.probe_depth[probe_index] == 0.0
  assert not expired.contact[link_index]
  assert expired.normal_force[link_index] == 0.0
  assert not expired.centroid_valid[link_index]
  assert np.isnan(expired.centroid_world[link_index]).all()


def test_instantaneous_contact_is_the_documented_depth_threshold(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  simulation = ArmHandSimulation()
  provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  link_index = 5
  target_id = simulation.model.geom("cylinder_geom").id
  candidates = [set() for _ in provider.link_names]
  candidates[link_index].add(target_id)
  depth = 2.0 * provider.contact_threshold_m
  distance = float(simulation.model.geom_margin[target_id]) - depth

  monkeypatch.setattr(provider, "_candidate_geoms", lambda _data: candidates)
  monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *_args: distance)

  sample = provider.read(simulation.data)
  expected = (provider.probe_depth >= provider.contact_threshold_m) & (
    provider.layout.probe_radius > 0.0
  )
  np.testing.assert_array_equal(provider.probe_contact_instantaneous, expected)
  assert expected[provider._probe_indices_by_link[link_index]].any()
  assert sample.centroid_valid[link_index]


def test_centroid_valid_is_independent_of_contact_threshold() -> None:
  simulation = ArmHandSimulation()
  provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  probe_index = int(np.flatnonzero(provider.layout.probe_radius > 0.0)[0])
  link_index = int(provider._probe_link_index[probe_index])

  # Positive sub-threshold support defines a centroid even though the
  # thresholded contact classification remains false.
  local = provider.layout.local_pos[probe_index]
  body_id = provider._body_ids[link_index]
  rotation = simulation.data.xmat[body_id].reshape(3, 3)
  centroid = simulation.data.xpos[body_id] + rotation @ local
  link_count = len(provider.link_names)
  centroid_world = np.full((link_count, 3), np.nan)
  centroid_world[link_index] = centroid

  # The public helper is intentionally a finite-data mask, not contact.copy().
  sample = LinkTactileSample(
    timestamp=float(simulation.data.time),
    source=provider.source,
    link_names=provider.link_names,
    contact=np.zeros(link_count, dtype=bool),
    normal_force=np.zeros(link_count),
    contact_count=np.zeros(link_count, dtype=np.int32),
    force_world=np.zeros((link_count, 3)),
    torque_world=np.zeros((link_count, 3)),
    force_local=np.zeros((link_count, 3)),
    centroid_world=centroid_world,
  )
  assert not sample.contact[link_index]
  assert sample.centroid_valid[link_index]


def test_genesis_hdf5_records_tactile_semantics_without_changing_old_shapes(
  tmp_path,
) -> None:
  simulation = ArmHandSimulation()
  config = WorkcellConfig(
    cameras=(), tactile_provider=GenesisProbeTactileProvider.source
  )
  output = tmp_path / "semantics.h5"

  with EpisodeRecorder(output, simulation, config) as recorder:
    recorder.record_initial()

  with h5py.File(output, "r") as file:
    assert file.attrs["schema_version"] == SCHEMA_VERSION
    tactile = file["tactile_genesis"]
    assert tactile.attrs["contact_threshold_m"] == pytest.approx(5.0e-5)
    assert tactile.attrs["release_threshold_m"] == pytest.approx(2.5e-5)
    assert tactile.attrs["release_debounce_seconds"] == pytest.approx(0.02)
    assert tactile.attrs["bool_count_threshold"] == 0
    assert "debounced" in tactile.attrs["contact_semantics"]
    assert "release debounce" in tactile.attrs["probe_contact_semantics"]
    assert (
      "probe_depth >= contact_threshold_m"
      in tactile.attrs["probe_contact_instantaneous_semantics"]
    )
    assert tactile.attrs["centroid_world_missing_value"] == "NaN"
    assert tactile.attrs["centroid_world_validity"] == "all three coordinates finite"

    # Existing v1 arrays retain their dimensions; the instantaneous mask is an
    # additive stream aligned with the same state/probe axes.
    assert tactile["contact"].shape == (1, 10)
    assert tactile["probe_contact"].shape == (1, 350)
    assert tactile["probe_depth"].shape == (1, 350)
    assert tactile["probe_contact_instantaneous"].shape == (1, 350)
    expected = (tactile["probe_depth"][0] >= tactile.attrs["contact_threshold_m"]) & (
      tactile["probe_radius"][:] > 0.0
    )
    np.testing.assert_array_equal(tactile["probe_contact_instantaneous"][0], expected)
