from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import (
  SCENE_NAMES,
  default_model_path,
  legacy_model_path,
)
from kaihand_tactile_env.shared.recording import (
  LEGACY_COMBINED_MODEL_LAYOUT,
  TASK_ISOLATED_MODEL_LAYOUT,
  _qpos_names,
  _qvel_names,
)
from kaihand_tactile_env.shared.simulation import ArmHandSimulation

_REPLAY = run_path(
  str(Path(__file__).parents[1] / "scripts" / "workcell" / "replay_episode.py")
)
_load_replay_states = _REPLAY["_load_replay_states"]
_load_validated_replay = _REPLAY["_load_validated_replay"]
_replay_model_selection = _REPLAY["_replay_model_selection"]
_restore_terminal_state = _REPLAY["_restore_terminal_state"]


def _write_state(
  path: Path,
  qpos: np.ndarray,
  qvel: np.ndarray,
  qpos_names: tuple[str, ...] | None,
  qvel_names: tuple[str, ...] | None,
  *,
  metadata: dict[str, object] | None = None,
  recorded_model_path: Path | None = None,
  timestamps: np.ndarray | None = None,
) -> None:
  with h5py.File(path, "w") as file:
    if metadata is not None:
      file.attrs["metadata_json"] = json.dumps(metadata)
    if recorded_model_path is not None:
      file.attrs["model_path"] = str(recorded_model_path)
    state = file.create_group("state")
    state.create_dataset("qpos", data=qpos)
    state.create_dataset("qvel", data=qvel)
    if timestamps is not None:
      state.create_dataset("timestamp", data=timestamps)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if qpos_names is not None:
      state.create_dataset("full_qpos_names", data=qpos_names, dtype=string_dtype)
    if qvel_names is not None:
      state.create_dataset("full_qvel_names", data=qvel_names, dtype=string_dtype)


def test_legacy_pick_place_state_keeps_current_card_hidden(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(
    legacy_model_path(), add_genesis_probes=False, scene="pick-place"
  )
  qpos_names = tuple(_qpos_names(simulation.model))
  qvel_names = tuple(_qvel_names(simulation.model))
  legacy_qpos_indices = tuple(
    index for index, name in enumerate(qpos_names) if not name.startswith("card_")
  )
  legacy_qvel_indices = tuple(
    index for index, name in enumerate(qvel_names) if not name.startswith("card_")
  )
  recorded_qpos = np.stack(
    (
      simulation.data.qpos[list(legacy_qpos_indices)],
      simulation.data.qpos[list(legacy_qpos_indices)] + 0.01,
    )
  )
  recorded_qvel = np.stack(
    (
      simulation.data.qvel[list(legacy_qvel_indices)],
      simulation.data.qvel[list(legacy_qvel_indices)] + 0.02,
    )
  )
  episode = tmp_path / "legacy.h5"
  _write_state(
    episode,
    recorded_qpos,
    recorded_qvel,
    tuple(qpos_names[index] for index in legacy_qpos_indices),
    tuple(qvel_names[index] for index in legacy_qvel_indices),
  )

  with h5py.File(episode, "r") as file:
    mapped_qpos, mapped_qvel = _load_replay_states(file, simulation)

  assert mapped_qpos.shape == (2, simulation.model.nq)
  assert mapped_qvel.shape == (2, simulation.model.nv)
  np.testing.assert_allclose(mapped_qpos[:, legacy_qpos_indices], recorded_qpos)
  np.testing.assert_allclose(mapped_qvel[:, legacy_qvel_indices], recorded_qvel)
  card_qpos_indices = tuple(
    index for index, name in enumerate(qpos_names) if name.startswith("card_")
  )
  card_qvel_indices = tuple(
    index for index, name in enumerate(qvel_names) if name.startswith("card_")
  )
  np.testing.assert_allclose(
    mapped_qpos[:, card_qpos_indices],
    np.broadcast_to(
      simulation.data.qpos[list(card_qpos_indices)],
      (mapped_qpos.shape[0], len(card_qpos_indices)),
    ),
  )
  np.testing.assert_allclose(
    mapped_qvel[:, card_qvel_indices],
    np.broadcast_to(
      simulation.data.qvel[list(card_qvel_indices)],
      (mapped_qvel.shape[0], len(card_qvel_indices)),
    ),
  )
  assert np.all(mapped_qpos[:, card_qpos_indices[:3]] == (0.0, 0.0, -10.0))


def test_matching_current_state_is_unchanged(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False, scene="poker-draw")
  qpos = np.stack((simulation.data.qpos, simulation.data.qpos + 0.01))
  qvel = np.stack((simulation.data.qvel, simulation.data.qvel + 0.02))
  episode = tmp_path / "current.h5"
  _write_state(episode, qpos, qvel, None, None)

  with h5py.File(episode, "r") as file:
    loaded_qpos, loaded_qvel = _load_replay_states(file, simulation)

  np.testing.assert_array_equal(loaded_qpos, qpos)
  np.testing.assert_array_equal(loaded_qvel, qvel)


def test_new_layout_selects_trusted_scene_model_not_recorded_path(
  tmp_path: Path,
) -> None:
  episode = tmp_path / "isolated.h5"
  untrusted_path = tmp_path / "untrusted.xml"
  _write_state(
    episode,
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    ("placeholder",),
    ("placeholder",),
    metadata={
      "scene": "poker-draw",
      "model_layout": TASK_ISOLATED_MODEL_LAYOUT,
    },
    recorded_model_path=untrusted_path,
  )

  with h5py.File(episode, "r") as file:
    scene, selected = _replay_model_selection(file)

  assert scene == "poker-draw"
  assert selected == default_model_path("poker-draw")
  assert selected != untrusted_path


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_new_layout_selects_each_task_model(
  tmp_path: Path, scene: str
) -> None:
  episode = tmp_path / f"{scene}.h5"
  _write_state(
    episode,
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    ("placeholder",),
    ("placeholder",),
    metadata={"scene": scene, "model_layout": TASK_ISOLATED_MODEL_LAYOUT},
  )

  with h5py.File(episode, "r") as file:
    selected_scene, selected_path = _replay_model_selection(file)

  assert selected_scene == scene
  assert selected_path == default_model_path(scene)
  assert selected_path.is_file()


def test_unversioned_recording_prefers_frozen_combined_model(tmp_path: Path) -> None:
  episode = tmp_path / "combined.h5"
  _write_state(
    episode,
    np.zeros((1, 2)),
    np.zeros((1, 2)),
    ("cylinder_freejoint/x", "card_freejoint/x"),
    ("cylinder_freejoint/x", "card_freejoint/x"),
    metadata={"scene": "poker-draw"},
  )

  with h5py.File(episode, "r") as file:
    scene, selected = _replay_model_selection(file)

  assert scene == "poker-draw"
  assert selected == legacy_model_path()


@pytest.mark.parametrize("scene", ("vase-wipe", "sponge-grasp"))
def test_mislabeled_sponge_layout_uses_isolated_model(
  tmp_path: Path, scene: str
) -> None:
  episode = tmp_path / f"{scene}.h5"
  _write_state(
    episode,
    np.zeros((1, 1)), np.zeros((1, 1)),
    ("sponge_freejoint/x",), ("sponge_freejoint/x",),
    metadata={
      "scene": scene, "model_layout": LEGACY_COMBINED_MODEL_LAYOUT,
      "active_objects": ["sponge"],
    },
  )
  with h5py.File(episode, "r") as file:
    _, selected = _replay_model_selection(file)
  assert selected == default_model_path(scene)


def test_unknown_layout_is_rejected_instead_of_opening_recorded_path(
  tmp_path: Path,
) -> None:
  episode = tmp_path / "unknown_layout.h5"
  _write_state(
    episode,
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    ("placeholder",),
    ("placeholder",),
    metadata={"scene": "pick-place", "model_layout": "external-v9"},
    recorded_model_path=tmp_path / "external.xml",
  )

  with h5py.File(episode, "r") as file:
    with pytest.raises(ValueError, match="unsupported model_layout"):
      _replay_model_selection(file)


def test_headless_replay_loads_and_restores_terminal_state(tmp_path: Path) -> None:
  source = ArmHandSimulation(add_genesis_probes=False, scene="pick-place")
  qpos = np.stack((source.data.qpos, source.data.qpos)).copy()
  qvel = np.stack((source.data.qvel, source.data.qvel)).copy()
  qpos[1, source._arm_qpos["right"][0]] += 0.001
  qvel[1, source._arm_dofs["right"][0]] += 0.002
  episode = tmp_path / "headless.h5"
  _write_state(
    episode,
    qpos,
    qvel,
    tuple(_qpos_names(source.model)),
    tuple(_qvel_names(source.model)),
    metadata={
      "scene": "pick-place",
      "model_layout": TASK_ISOLATED_MODEL_LAYOUT,
    },
    timestamps=np.array([0.0, 0.01]),
  )

  with h5py.File(episode, "r") as file:
    simulation, timestamps, loaded_qpos, loaded_qvel = _load_validated_replay(file)
  _restore_terminal_state(simulation, loaded_qpos, loaded_qvel)

  np.testing.assert_array_equal(timestamps, (0.0, 0.01))
  np.testing.assert_allclose(simulation.data.qpos, qpos[-1])
  np.testing.assert_allclose(simulation.data.qvel, qvel[-1])


def test_legacy_mismatch_requires_coordinate_names(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  episode = tmp_path / "unnamed.h5"
  _write_state(
    episode,
    np.zeros((1, simulation.model.nq - 1)),
    np.zeros((1, simulation.model.nv - 1)),
    None,
    None,
  )

  with h5py.File(episode, "r") as file:
    with pytest.raises(ValueError, match="compatibility metadata"):
      _load_replay_states(file, simulation)


def test_removed_coordinate_is_rejected(tmp_path: Path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  episode = tmp_path / "removed.h5"
  _write_state(
    episode,
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    ("removed_joint",),
    ("removed_joint",),
  )

  with h5py.File(episode, "r") as file:
    with pytest.raises(ValueError, match="absent from the current model"):
      _load_replay_states(file, simulation)
