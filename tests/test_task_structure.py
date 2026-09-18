from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.simulation import ArmHandSimulation


@pytest.mark.parametrize(
  ("old_module", "new_module"),
  (
    ("workcell.simulation", "shared.simulation"),
    ("workcell.recording", "shared.recording"),
    ("workcell.tactile", "shared.tactile"),
    ("workcell.planning", "tasks.pick_place.task"),
    ("workcell.poker", "tasks.poker_draw.task"),
  ),
)
def test_existing_imports_keep_module_identity(
  old_module: str, new_module: str
) -> None:
  old = importlib.import_module(f"kaihand_tactile_env.{old_module}")
  new = importlib.import_module(f"kaihand_tactile_env.{new_module}")
  assert old is new


def test_model_fingerprint_tracks_shared_includes_and_is_relocatable(
  tmp_path: Path,
) -> None:
  for directory in (tmp_path / "first", tmp_path / "second"):
    directory.mkdir()
    (directory / "scene.xml").write_text(
      '<mujoco><include file="robot.xml"/></mujoco>', encoding="utf-8"
    )
    (directory / "robot.xml").write_text(
      '<mujoco><option timestep="0.002"/></mujoco>', encoding="utf-8"
    )
  original = model_fingerprint(tmp_path / "first/scene.xml")
  assert original == model_fingerprint(tmp_path / "second/scene.xml")
  (tmp_path / "second/robot.xml").write_text(
    '<mujoco><option timestep="0.004"/></mujoco>', encoding="utf-8"
  )
  assert original != model_fingerprint(tmp_path / "second/scene.xml")


def test_independent_model_cannot_be_switched_to_missing_task() -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False, scene="pick-place")
  with pytest.raises(ValueError, match="create ArmHandSimulation"):
    simulation.set_scene("poker-draw")
  assert simulation.scene == "pick-place"
  assert simulation.object_names == ("cylinder",)
  with pytest.raises(ValueError, match="missing task objects"):
    ArmHandSimulation(
      default_model_path("pick-place"),
      add_genesis_probes=False,
      scene="poker-draw",
    )
