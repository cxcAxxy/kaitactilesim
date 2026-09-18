#!/usr/bin/env python3
"""Compare aligned spherical-probe and solver-contact tactile streams."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("reference", help="spherical-probe episode HDF5")
  parser.add_argument("candidate", help="solver-proxy episode HDF5")
  parser.add_argument("--reference-group", default="tactile_genesis")
  parser.add_argument("--candidate-group", default="tactile_proxy")
  args = parser.parse_args()
  with (
    h5py.File(args.reference, "r") as ref_file,
    h5py.File(args.candidate, "r") as cand_file,
  ):
    reference_force = np.asarray(ref_file[f"{args.reference_group}/normal_force"])
    candidate_force = np.asarray(cand_file[f"{args.candidate_group}/normal_force"])
    reference_contact = np.asarray(ref_file[f"{args.reference_group}/contact"])
    candidate_contact = np.asarray(cand_file[f"{args.candidate_group}/contact"])
    ref_links = tuple(_decode(ref_file[f"{args.reference_group}/link_names"][:]))
    cand_links = tuple(_decode(cand_file[f"{args.candidate_group}/link_names"][:]))
    reference_unit = str(
      ref_file[args.reference_group].attrs.get("force_unit", "unspecified")
    )
    candidate_unit = str(
      cand_file[args.candidate_group].attrs.get("force_unit", "unspecified")
    )
  if ref_links != cand_links:
    raise RuntimeError("link order differs between reference and candidate")
  if reference_force.shape != candidate_force.shape:
    raise RuntimeError("stream shapes differ; align timestamps before comparison")
  error = candidate_force - reference_force
  tp = np.logical_and(reference_contact, candidate_contact).sum()
  fp = np.logical_and(~reference_contact, candidate_contact).sum()
  fn = np.logical_and(reference_contact, ~candidate_contact).sum()
  precision = float(tp / max(tp + fp, 1))
  recall = float(tp / max(tp + fn, 1))
  correlation = 0.0
  if np.std(reference_force) > 0.0 and np.std(candidate_force) > 0.0:
    correlation = float(
      np.corrcoef(reference_force.ravel(), candidate_force.ravel())[0, 1]
    )
  candidate_energy = float(np.sum(candidate_force**2))
  fitted_scale = (
    float(np.sum(reference_force * candidate_force) / candidate_energy)
    if candidate_energy > 0.0
    else 0.0
  )
  scaled_error = fitted_scale * candidate_force - reference_force
  result = {
    "samples": int(reference_force.shape[0]),
    "links": list(ref_links),
    "reference_force_unit": reference_unit,
    "candidate_force_unit": candidate_unit,
    "force_units_comparable": reference_unit == candidate_unit,
    "raw_numeric_force_rmse": float(np.sqrt(np.mean(error**2))),
    "raw_numeric_force_mae": float(np.mean(np.abs(error))),
    "normal_force_correlation": correlation,
    "candidate_to_reference_l2_scale": fitted_scale,
    "scaled_force_rmse_in_reference_units": float(np.sqrt(np.mean(scaled_error**2))),
    "contact_precision": precision,
    "contact_recall": recall,
    "contact_f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
  }
  if reference_unit == candidate_unit == "N":
    result["normal_force_rmse_n"] = result["raw_numeric_force_rmse"]
    result["normal_force_mae_n"] = result["raw_numeric_force_mae"]
  print(json.dumps(result, indent=2, ensure_ascii=False))


def _decode(values: np.ndarray) -> list[str]:
  return [
    value.decode() if isinstance(value, bytes) else str(value) for value in values
  ]


if __name__ == "__main__":
  main()
