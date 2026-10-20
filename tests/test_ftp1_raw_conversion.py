from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py"
)
SPEC = importlib.util.spec_from_file_location("ftp1_raw_conversion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def test_ftp1_contract_uses_official_horizon_and_kaihand_tactile() -> None:
  contract = converter._conversion_contract(
    instruction="draw a card",
    sensor_name="KaiHandSimTaxelV1",
  )

  assert contract["ftp1_action_horizon"] == 32
  assert contract["ftp1_action_down_sample_steps"] == 1
  assert contract["tactile_type"] == "matrix"
  assert contract["tactile_shape_per_area"] == [7, 5, 3]
  assert contract["right_tactile_areas"] == [0, 1, 2, 3, 4]


def test_strict_30hz_prefix_allows_only_one_off_grid_terminal_frame() -> None:
  exact = np.arange(5, dtype=np.float64) / 30.0
  frames, error = converter._strict_30hz_prefix(exact, physics_hz=500)
  assert frames == 5
  assert error == pytest.approx(0.0)

  terminal = np.concatenate((exact, np.asarray([0.25])))
  frames, _ = converter._strict_30hz_prefix(terminal, physics_hz=500)
  assert frames == 5

  interior = exact.copy()
  interior[2] += 0.01
  with pytest.raises(ValueError, match="optional off-grid terminal"):
    converter._strict_30hz_prefix(interior, physics_hz=500)


def test_discovery_rejects_false_per_task_target(tmp_path: Path) -> None:
  (tmp_path / "summary.json").write_text(
    json.dumps({"target_met": {"poker-draw": False}, "episodes": []}),
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match=r"target_met\['poker-draw'\] is not true"):
    converter.discover_sources(tmp_path)


def test_resume_fingerprint_covers_instruction_and_source_identity(
  tmp_path: Path,
) -> None:
  source = converter.SourceEpisode(
    episode_index=7,
    hdf5_path=tmp_path / "episode.h5",
    relative_hdf5_path="task/000007/episode.h5",
    sidecar_path=tmp_path / "episode.json",
    relative_sidecar_path="task/000007/episode.json",
    size_bytes=123,
    mtime_ns=456,
    declared_sha256="a" * 64,
  )

  baseline = converter._shard_fingerprint(
    [source],
    instruction="draw a card",
    sensor_name="KaiHandSimTaxelV1",
  )
  changed_instruction = converter._shard_fingerprint(
    [source],
    instruction="pick up a card",
    sensor_name="KaiHandSimTaxelV1",
  )
  changed_source = converter._shard_fingerprint(
    [converter.SourceEpisode(**{**source.__dict__, "mtime_ns": 457})],
    instruction="draw a card",
    sensor_name="KaiHandSimTaxelV1",
  )

  assert baseline != changed_instruction
  assert baseline != changed_source
