from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import (
  WorkcellConfig,
  default_model_path,
  model_fingerprint,
)
from kaihand_tactile_env.shared.recording import (
  TASK_ISOLATED_MODEL_LAYOUT,
  EpisodeRecorder,
  validate_episode,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import (
  GenesisProbeTactileProvider,
  SolverContactTactileProvider,
)


@pytest.mark.parametrize(
  "tactile_source",
  (GenesisProbeTactileProvider.source, SolverContactTactileProvider.source),
)
@pytest.mark.parametrize(
  ("scene", "active_object"),
  (("pick-place", "cylinder"), ("poker-draw", "card")),
)
def test_task_recording_contains_only_active_object_and_model_identity(
  tmp_path: Path,
  scene: str,
  active_object: str,
  tactile_source: str,
) -> None:
  model_path = default_model_path(scene)
  simulation = ArmHandSimulation(
    model_path,
    scene=scene,
    add_genesis_probes=tactile_source == GenesisProbeTactileProvider.source,
  )
  config = WorkcellConfig(
    model_path=model_path,
    cameras=(),
    tactile_provider=tactile_source,
  )
  output = tmp_path / f"{scene}.h5"

  with EpisodeRecorder(
    output, simulation, config, metadata={"scene": "wrong"}
  ) as recorder:
    recorder.record_initial()
    if tactile_source == GenesisProbeTactileProvider.source:
      expected_geom = "card_core_geom" if active_object == "card" else "cylinder_geom"
      assert recorder.tactile_provider._target_geom_set == {
        simulation.model.geom(expected_geom).id
      }

  with h5py.File(output, "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    assert set(file["objects"].keys()) == {active_object}
    assert metadata["scene"] == scene
    assert metadata["active_objects"] == [active_object]
    assert metadata["model_layout"] == TASK_ISOLATED_MODEL_LAYOUT
    assert isinstance(metadata["model_id"], str) and metadata["model_id"]
    assert (
      file.attrs["model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    assert file.attrs["model_fingerprint"] == model_fingerprint(model_path)
    assert file.attrs["tactile_source"] == tactile_source
    if tactile_source == GenesisProbeTactileProvider.source:
      assert file["tactile_genesis/probe_depth"].shape == (1, 350)
    assert ("tactile_contact_force" in file) == (scene == "poker-draw")
    if scene == "poker-draw":
      forces = file["tactile_contact_force"]
      assert forces.attrs["force_unit"] == "N"
      assert forces.attrs["taxel_unit"] == "N_per_taxel"
      assert forces.attrs["is_spatial_estimate"]
      assert forces.attrs["timestamp_reference"] == "/state/timestamp"
      assert len(forces["link_names"]) == 10  # Left hand is hidden only in video.
      assert forces["taxel_positions_local_m"].shape == (10, 35, 3)
      assert forces["tangent_basis_local"].shape == (10, 2, 3)
      assert forces["normal_taxel_force_n"].shape == (1, 10, 7, 5)
      assert forces["tangent_taxel_force_n"].shape == (1, 10, 7, 5, 2)
      for name in ("normal_force_n", "tangent_force_n", "force_world_n"):
        assert np.isfinite(forces[name][:]).all()
      np.testing.assert_allclose(
        forces["normal_taxel_force_n"][:].sum(axis=(2, 3)),
        forces["normal_force_n"][:],
      )
      np.testing.assert_allclose(
        forces["tangent_taxel_force_n"][:].sum(axis=(2, 3)),
        forces["tangent_force_n"][:],
      )

  assert validate_episode(output).valid
  if scene == "poker-draw":
    with h5py.File(output, "r+") as file:
      basis = file["tactile_contact_force/tangent_basis_local"][:]
      del file["tactile_contact_force/tangent_basis_local"]
    assert any("invalid layout" in error for error in validate_episode(output).errors)
    with h5py.File(output, "r+") as file:
      file["tactile_contact_force"].create_dataset("tangent_basis_local", data=basis)
      file["tactile_contact_force/tangent_taxel_load_n"][0, 0, 0, 0] = 0.5
    assert any(
      "does not conserve" in error for error in validate_episode(output).errors
    )
    with h5py.File(output, "r+") as file:
      file["tactile_contact_force/tangent_taxel_load_n"][0, 0, 0, 0] = 0.0
    with h5py.File(output, "r+") as file:
      file["tactile_contact_force/tangent_taxel_force_n"][0, 0, 0, 0, 0] = 0.5
    report = validate_episode(output)
    assert not report.valid
    assert any("does not conserve" in error for error in report.errors)
    with h5py.File(output, "r+") as file:
      del file["tactile_contact_force"]
    report = validate_episode(output)
    assert not report.valid
    assert any("missing tactile_contact_force" in error for error in report.errors)
