from __future__ import annotations

from pathlib import Path
from runpy import run_path

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation

_MODULE = run_path(
  str(Path(__file__).parents[1] / "scripts/workcell/experiment_bulb_funnel_geometry.py")
)
_funnel_vertices = _MODULE["_funnel_vertices"]
_parse_args = _MODULE["_parse_args"]
_temporary_scene = _MODULE["_temporary_scene"]


def test_funnel_vertex_generator_sets_requested_throat_and_collar():
  values = np.fromstring(_funnel_vertices(0.0165, 0.008), sep=" ").reshape(12, 3)
  np.testing.assert_allclose(np.linalg.norm(values[[0, 1, 4, 5], :2], axis=1), 0.0165)
  np.testing.assert_allclose(values[:4, 2], -0.012)
  np.testing.assert_allclose(values[4:8, 2], 0.008)
  np.testing.assert_allclose(values[8:, 2], 0.028)


def test_temporary_variant_loads_without_mutating_production_scene():
  source = default_model_path("bulb-screw")
  before = source.read_bytes()
  with _temporary_scene(
    source,
    throat_radius_m=0.0165,
    collar_top_z_m=0.008,
    friction=0.15,
  ) as variant:
    simulation = BulbScrewSimulation(model_path=variant)
    ids = np.array(
      [
        simulation.model.geom(f"bulb_socket_funnel_{index:02d}").id
        for index in range(24)
      ]
    )
    assert np.all(simulation.model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_MESH)
    np.testing.assert_allclose(simulation.model.geom_friction[ids, 0], 0.15)
  assert not variant.exists()
  assert source.read_bytes() == before


def test_geometry_sweep_rejects_overwrite_and_unsafe_throat(tmp_path: Path):
  output = tmp_path / "result.json"
  output.write_text("occupied")
  with pytest.raises(SystemExit):
    _parse_args(["--output", str(output)])
  with pytest.raises(SystemExit):
    _parse_args(["--output", str(tmp_path / "new.json"), "--throat-radii-mm", "15.9"])
