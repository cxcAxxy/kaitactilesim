from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.tasks.poker_draw.friction import (
  BASELINE_TABLE_CARD_FRICTION,
  TABLE_CARD_PAIR_NAME,
  default_friction_trials,
  model_with_table_card_friction,
  set_table_card_friction,
  theoretical_minimum_press_force,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawExecutor


def test_theoretical_press_threshold_matches_coulomb_balance() -> None:
  threshold = theoretical_minimum_press_force(0.8, 1.4, 0.004, 9.81)
  assert threshold == pytest.approx(0.05232)
  assert np.isinf(theoretical_minimum_press_force(1.4, 1.4, 0.004, 9.81))


def test_temporary_profile_is_pair_local_and_leaves_scene_unchanged() -> None:
  scene = default_model_path("poker-draw")
  original = scene.read_bytes()
  with model_with_table_card_friction(scene, 0.8) as wrapper:
    assert wrapper.is_file()
    root = ET.parse(wrapper).getroot()
    include = root.find("include")
    pair = root.find("contact/pair")
    assert include is not None
    assert Path(include.attrib["file"]).is_absolute()
    assert Path(include.attrib["file"]).resolve() == scene.resolve()
    assert pair is not None
    assert pair.attrib["name"] == TABLE_CARD_PAIR_NAME
    assert pair.attrib["geom1"] == "card_core_geom"
    assert pair.attrib["geom2"] == "poker_table_top"
    friction = tuple(float(value) for value in pair.attrib["friction"].split())
    assert friction == pytest.approx(
      (0.8, 0.8, *BASELINE_TABLE_CARD_FRICTION[2:])
    )
  assert scene.read_bytes() == original
  assert not wrapper.exists()


def test_runtime_update_changes_only_pair_sliding_terms(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair_friction = np.asarray([BASELINE_TABLE_CARD_FRICTION], dtype=float)
  model = SimpleNamespace(pair_friction=pair_friction)
  monkeypatch.setattr(mujoco, "mj_name2id", lambda *_args: 0)

  set_table_card_friction(model, 0.8)  # type: ignore[arg-type]

  np.testing.assert_allclose(pair_friction[0], (0.8, 0.8, 0.005, 0.0005, 0.0005))


def test_legacy_sweep_is_bounded_but_production_defaults_to_force_control() -> None:
  trials = default_friction_trials()
  assert len(trials) == 8
  assert {trial.table_card_friction for trial in trials} == {0.1, 0.9, 1.15, 1.3}
  assert {trial.press_distal_offset_degrees for trial in trials} == {0.0, 2.0}
  parameter = inspect.signature(
    PokerDrawExecutor._press_until_four_contacts
  ).parameters["distal_offset_degrees"]
  # A numeric value explicitly selects historical position preload.  The
  # production path must not silently retain the old +2 degree joint curl.
  assert parameter.default is None
