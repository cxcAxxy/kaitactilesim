"""Lightweight diagnostic-reference tests, no simulation or network."""

import io
import tarfile
from pathlib import Path
from runpy import run_path

import numpy as np


def test_reference_relative_roundtrip(monkeypatch):
  path = Path(__file__).parents[1] / "scripts/workcell"
  monkeypatch.syspath_prepend(str(path))
  module = run_path(str(path / "audit_poker_execution.py"))
  state = np.zeros(48, dtype=np.float32)
  state[6:18] = [1, 0, 0, 0, 1, 0] * 2
  state[:6] = [0.1, 0.2, 0.3, -0.1, 0.1, 0.2]
  absolute = np.repeat(state[None], 32, axis=0)
  absolute[:, 3] += np.linspace(0, 0.1, 32)
  absolute[:, 18:] += np.linspace(0, 0.01, 32)[:, None]
  relative = module["encode_reference_relative"](state, absolute)
  restored = module["relative_to_absolute"](state, relative)
  np.testing.assert_allclose(restored, absolute, atol=1e-7)


def test_load_only_selected_contiguous_episode(monkeypatch, tmp_path):
  path = Path(__file__).parents[1] / "scripts/workcell"
  monkeypatch.syspath_prepend(str(path))
  module = run_path(str(path / "audit_poker_execution.py"))
  shard = tmp_path / "shard.tar"
  with tarfile.open(shard, "w") as tar:
    for episode, index in [(3, 0), (3, 1), (4, 0)]:
      buf = io.BytesIO()
      np.save(buf, np.full(116, index, dtype=np.float32), allow_pickle=False)
      content = buf.getvalue()
      member = tarfile.TarInfo(f"episode_{episode:06d}_frame_{index:06d}.lowdim.npy")
      member.size = len(content)
      tar.addfile(member, io.BytesIO(content))
  rows = module["load_episode"](shard, 3)
  assert rows.shape == (2, 116)
  assert np.all(rows[0] == 0) and np.all(rows[1] == 1)
