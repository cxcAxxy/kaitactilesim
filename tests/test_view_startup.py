from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path

import mujoco
import mujoco.viewer
import pytest
from kaihand_tactile_env.shared.config import legacy_model_path
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import GenesisProbeTactileProvider


@pytest.mark.parametrize("scene", ("pick-place", "poker-draw"))
def test_default_view_command_reaches_window_creation(
  scene: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Exercise real startup up to the window boundary without creating OpenGL."""
  script = run_path(
    str(Path(__file__).parents[1] / "scripts/workcell/view_workcell.py")
  )
  monkeypatch.setattr(
    sys, "argv", ["view_workcell.py", "--task", scene, "--viewer-hz", "30"]
  )

  class WindowCreationReached(Exception):
    pass

  def launch(model: mujoco.MjModel, data: mujoco.MjData):
    assert model.nq == len(data.qpos)
    raise WindowCreationReached

  monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
  with pytest.raises(WindowCreationReached):
    script["main"]()


@pytest.mark.parametrize(
  ("scene", "expected_geom"),
  (("pick-place", "cylinder_geom"), ("poker-draw", "card_core_geom")),
)
def test_legacy_default_tactile_ignores_inactive_object(
  scene: str, expected_geom: str
) -> None:
  simulation = ArmHandSimulation(legacy_model_path(), scene=scene)
  provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  assert provider._target_geom_set == {simulation.model.geom(expected_geom).id}
  sample = provider.read(simulation.data)
  assert sample.normal_force.shape == (10,)


def test_explicit_missing_tactile_target_is_still_rejected() -> None:
  simulation = ArmHandSimulation(scene="poker-draw")
  with pytest.raises(RuntimeError, match="cylinder_geom"):
    GenesisProbeTactileProvider(
      simulation.model,
      simulation.genesis_probe_layout,
      target_geom_names=("cylinder_geom",),
    )
