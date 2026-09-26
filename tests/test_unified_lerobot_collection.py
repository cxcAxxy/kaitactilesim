from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

from convert_usb_unified_to_lerobot import _unified_pairs  # noqa: E402
from unified_lerobot_collection import discover_unified_artifacts  # noqa: E402


def _collection(tmp_path: Path, *, task: str, status: str = "success") -> Path:
  root = tmp_path / "raw"
  data = root / task / "000042/attempt_001/data/raw"
  data.mkdir(parents=True)
  hdf5 = data / "episode_000000.h5"
  hdf5.write_bytes(b"raw")
  hdf5.with_suffix(".json").write_text("{}", encoding="utf-8")
  relative = str(hdf5.relative_to(root))
  (root / "collection.json").write_text(json.dumps({
    "schema": "task_collection_v2",
    "tasks": [task],
    "collection_mode": "target-successes",
  }), encoding="utf-8")
  (root / "summary.json").write_text(json.dumps({
    "episodes": [{
      "task": task,
      "episode_index": 42,
      "status": status,
      "hdf5": [relative],
    }],
    "by_task": {task: {"success": int(status == "success")}},
    "target_met": {task: status == "success"},
  }), encoding="utf-8")
  return root


def test_discovers_success_and_preserves_outer_episode_index(tmp_path: Path) -> None:
  root = _collection(tmp_path, task="poker-draw")
  result = discover_unified_artifacts(
    root,
    task="poker-draw",
    expected_episodes=1,
    limit=None,
  )
  assert result is not None
  artifacts, available = result
  assert available == 1
  assert artifacts[0].episode_index == 42
  assert artifacts[0].hdf5_path.name == "episode_000000.h5"


def test_rejects_incomplete_success_target(tmp_path: Path) -> None:
  root = _collection(tmp_path, task="pick-place", status="failed")
  with pytest.raises(ValueError, match="success target was not met"):
    discover_unified_artifacts(
      root,
      task="pick-place",
      expected_episodes=1,
      limit=None,
    )


def test_returns_none_for_legacy_flat_batch(tmp_path: Path) -> None:
  assert discover_unified_artifacts(
    tmp_path,
    task="poker-draw",
    expected_episodes=0,
    limit=None,
  ) is None


def test_usb_wrapper_accepts_attempts_in_mixed_collection(tmp_path: Path) -> None:
  root = _collection(tmp_path, task="usb-insert")
  contract_path = root / "collection.json"
  contract = json.loads(contract_path.read_text())
  contract["tasks"] = ["usb-insert", "poker-draw"]
  contract["collection_mode"] = "attempts"
  contract_path.write_text(json.dumps(contract))
  module = SimpleNamespace(
    SourcePair=lambda index, hdf5, sidecar: (index, hdf5, sidecar),
    _legacy_discover_pairs=lambda *_: pytest.fail("legacy discovery was selected"),
    common=SimpleNamespace(
      _snapshot_file=lambda path: path,
      _assert_unchanged=lambda *_: None,
    ),
  )
  pairs, available, _ = _unified_pairs(module, root, 1, None)
  assert available == 1
  assert pairs[0][0] == 42
