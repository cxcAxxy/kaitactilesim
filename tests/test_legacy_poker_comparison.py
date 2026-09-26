"""Small contract tests for honest conversion of an old poker review."""

import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
)

SCRIPT = run_path(
  str(Path(__file__).resolve().parents[1] / "scripts/workcell/plot_legacy_poker_tactile.py")
)


def _legacy_seed(tmp_path):
  seed = tmp_path / "seed_000"
  review = seed / "review"
  review.mkdir(parents=True)
  (review / "review.json").write_text(json.dumps({
    "task": "poker-draw",
    "fps": 10,
    "control_hz": 30,
    "frame_count": 2,
    "tactile_source": "solver_contact_distributed_taxel_v1",
  }))
  (seed / "summary.json").write_text(json.dumps({
    "task": "poker-draw",
    "execute_steps": 16,
    "control_hz": 30,
    "server_metadata": {"joint_names": [
      *(f"right_arm_joint{i}" for i in range(1, 8)),
      *RIGHT_HAND_ACTUATED_JOINT_NAMES,
    ]},
    "stats": {"requests": 2},
  }))
  rows = []
  for index in range(2):
    normal = np.zeros((10, 7, 5))
    tangent = np.zeros((10, 7, 5, 2))
    normal[5] = 2 + index
    tangent[5, 0, 0] = [3, 4]
    tangent[5, 0, 1] = [-1, -2]
    rows.append({
      "frame": index,
      "control_tick": index * 3,
      "simulation_time_s": index / 10,
      "fingertip_order": FINGERTIP_LINK_NAMES,
      "normal_taxel_force_n": normal.tolist(),
      "tangent_taxel_force_n": tangent.tolist(),
    })
  (review / "frames.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows)
  )
  requests = [
    {"request": index + 1, "sim_time": index * 16 / 30,
     "state": (np.arange(27) + index).tolist()}
    for index in range(2)
  ]
  (seed / "requests.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in requests)
  )
  return seed, rows


def test_legacy_tactile_uses_only_recorded_frames_and_vector_sum(tmp_path):
  seed, _ = _legacy_seed(tmp_path)
  trace = SCRIPT["load_legacy_tactile_frames"](seed)
  np.testing.assert_allclose(trace.time_s, [0, 0.1])
  np.testing.assert_allclose(trace.normal_force_n[:, 0], [70, 105])
  np.testing.assert_allclose(trace.tangent_force_n[:, 0], [np.sqrt(8)] * 2)
  np.testing.assert_allclose(trace.normal_force_n[:, 1:], 0)


def test_legacy_tactile_rejects_wrong_finger_order(tmp_path):
  seed, rows = _legacy_seed(tmp_path)
  rows[1]["fingertip_order"] = list(reversed(FINGERTIP_LINK_NAMES))
  (seed / "review/frames.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows)
  )
  with pytest.raises(ValueError, match="fingertip order"):
    SCRIPT["load_legacy_tactile_frames"](seed)


def test_legacy_request_states_are_sparse_measured_joint_values(tmp_path):
  seed, _ = _legacy_seed(tmp_path)
  requests = SCRIPT["load_legacy_request_states"](seed)
  np.testing.assert_allclose(requests.time_s, [0, 16 / 30])
  np.testing.assert_allclose(requests.measured_state_27[1, 7:], np.arange(7, 27) + 1)
  assert requests.execute_steps == 16


def test_metadata_identifies_different_sample_rates_and_fk(tmp_path, monkeypatch):
  seed, _ = _legacy_seed(tmp_path)
  reference = SimpleNamespace(
    dataset_root=tmp_path / "reference",
    output_episode_index=0,
    source_hdf5=tmp_path / "episode_00.h5",
    time_s=np.arange(3) / 30,
    wrist_state=np.zeros((3, 9)),
    hand_joint_position=np.zeros((3, 20)),
    fingertip_normal_force_n=np.zeros((3, 5)),
    fingertip_tangent_force_n=np.zeros((3, 5)),
  )
  monkeypatch.setitem(SCRIPT["convert_legacy_poker_comparison"].__globals__,
                      "load_openwam_reference", lambda *_args, **_kwargs: reference)
  monkeypatch.setitem(SCRIPT["convert_legacy_poker_comparison"].__globals__,
                      "wrist_fk_from_measured_states",
                      lambda requests, _initial: (np.zeros((len(requests.time_s), 9)),
                                                  {"status": "verified_against_reference_home_pose"}))
  monkeypatch.setitem(SCRIPT["convert_legacy_poker_comparison"].__globals__,
                      "_plot_tactile", lambda *_args: None)
  monkeypatch.setitem(SCRIPT["convert_legacy_poker_comparison"].__globals__,
                      "_plot_sparse_series", lambda *_args, **_kwargs: None)
  metadata = SCRIPT["convert_legacy_poker_comparison"](seed, tmp_path / "reference")
  assert metadata["rollout"]["sampling_hz"] == 10
  assert metadata["request_states"]["execute_steps"] == 16
  assert metadata["request_states"]["observed_sampling_hz"] == pytest.approx(1.875)
  assert metadata["reference"]["sampling_hz"] == 30
  assert "right_wrist_state" in metadata["artifacts"]
  assert "none" in metadata["styles"]["interpolation"]
