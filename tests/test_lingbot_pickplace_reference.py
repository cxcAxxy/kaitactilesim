"""Recorded-contact reconstruction for LingBot's PickPlace reference."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/workcell"))

from lingbot_pickplace_reference import (  # noqa: E402
  _contact_force_totals,
  _recorded_geom_ids,
)


def test_recorded_solver_tangents_cancel_before_magnitude() -> None:
  arrays = {
    "start": np.asarray([0]),
    "count": np.asarray([4]),
    "geom1": np.asarray([10, 99, 99, 10]),
    "geom2": np.asarray([99, 10, 11, 20]),
    "frame": np.repeat(np.eye(3)[None], 4, axis=0),
    "wrench": np.asarray([
      [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [4.0, 0.0, 2.0, 0.0, 0.0, 0.0],
      [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # non-target contact is excluded
    ]),
  }
  basis = np.repeat(np.eye(3)[None, :2], 5, axis=0)
  basis[1] = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
  normal, tangent = _contact_force_totals(
    arrays, 0, {10: 0, 11: 1}, frozenset({99}), basis
  )
  np.testing.assert_allclose(normal, [5.0, 4.0, 0.0, 0.0, 0.0])
  np.testing.assert_allclose(tangent, [0.0, 2.0, 0.0, 0.0, 0.0])


def test_recorded_geom_lookup_requires_unique_named_geoms(tmp_path: Path) -> None:
  path = tmp_path / "geoms.h5"
  with h5py.File(path, "w") as file:
    file.create_dataset(
      "model/geom_names",
      data=np.asarray([b"pad", b"target", b"pad"], dtype="S8"),
    )
  with h5py.File(path) as file:
    assert _recorded_geom_ids(file, ("target",)) == {"target": 1}
    with pytest.raises(ValueError, match="exactly one geom"):
      _recorded_geom_ids(file, ("pad",))

