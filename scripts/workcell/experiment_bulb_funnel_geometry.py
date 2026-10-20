#!/usr/bin/env python3
"""Sweep diagnostic bulb-funnel geometry without editing the production scene."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import mujoco
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.tasks.bulb_screw import config
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation

_WORKCELL = Path(__file__).resolve().parent
if str(_WORKCELL) not in sys.path:
  sys.path.insert(0, str(_WORKCELL))
import experiment_bulb_capture_basin as basin  # noqa: E402

_SECTOR_HALF_ANGLE_RAD = math.radians(7.5)
_OUTER_RADIUS_M = 0.044
_TOP_INNER_RADIUS_M = 0.030
_BOTTOM_Z_M = -0.012
_TOP_Z_M = 0.028


def _finite(values: Sequence[float], name: str) -> tuple[float, ...]:
  result = tuple(float(value) for value in values)
  if not result or not all(math.isfinite(value) for value in result):
    raise ValueError(f"{name} must contain finite values")
  return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--throat-radii-mm", type=float, nargs="+", default=(16.35, 16.50, 16.65, 16.80)
  )
  parser.add_argument(
    "--collar-top-z-mm", type=float, nargs="+", default=(7.0, 8.0, 10.0)
  )
  parser.add_argument(
    "--funnel-friction", type=float, nargs="+", default=(0.05, 0.15, 0.30)
  )
  parser.add_argument("--yaw-degrees", type=float, nargs="+", default=(0.0,))
  parser.add_argument(
    "--lateral-offsets-mm", type=float, nargs="+", default=(0.0, 1.0, 2.0)
  )
  parser.add_argument("--tilt-degrees", type=float, nargs="+", default=(0.0, 2.0))
  parser.add_argument("--azimuth-degrees", type=float, nargs="+", default=(0.0, 7.5))
  parser.add_argument("--lateral-stiffness-n-m", type=float, default=50.0)
  parser.add_argument("--angular-stiffness-nm-rad", type=float, default=0.01)
  parser.add_argument("--approach-speed-mm-s", type=float, default=20.0)
  parser.add_argument("--start-clearance-mm", type=float, default=2.0)
  parser.add_argument("--approach-duration-s", type=float, default=2.0)
  parser.add_argument("--settle-duration-s", type=float, default=0.25)
  parser.add_argument("--reverse-torque-nm", type=float, default=0.06)
  parser.add_argument("--release-timeout-s", type=float, default=1.5)
  parser.add_argument("--extract-force-n", type=float, default=1.2)
  parser.add_argument("--extract-timeout-s", type=float, default=1.0)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  try:
    for name in (
      "throat_radii_mm",
      "collar_top_z_mm",
      "funnel_friction",
      "yaw_degrees",
      "lateral_offsets_mm",
      "tilt_degrees",
      "azimuth_degrees",
    ):
      setattr(args, name, _finite(getattr(args, name), name))
  except ValueError as error:
    parser.error(str(error))
  if not all(16.0 < value < 30.0 for value in args.throat_radii_mm):
    parser.error("throat radii must be between the 16 mm neck and 30 mm funnel top")
  if not all(-12.0 < value < 28.0 for value in args.collar_top_z_mm):
    parser.error("collar top z must lie within the funnel mesh")
  if not all(value > 0 for value in args.funnel_friction):
    parser.error("funnel friction must be positive")
  nonnegative = (
    *args.lateral_offsets_mm,
    *args.tilt_degrees,
    args.lateral_stiffness_n_m,
    args.angular_stiffness_nm_rad,
    args.approach_speed_mm_s,
    args.start_clearance_mm,
    args.approach_duration_s,
    args.settle_duration_s,
    args.release_timeout_s,
    args.extract_force_n,
    args.extract_timeout_s,
  )
  if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
    parser.error("offsets, tilts and dynamic parameters must be finite/nonnegative")
  if args.approach_duration_s <= 0 or args.reverse_torque_nm <= 0:
    parser.error("approach duration and reverse torque must be positive")
  configurations = (
    len(args.throat_radii_mm) * len(args.collar_top_z_mm) * len(args.funnel_friction)
  )
  trials_per_configuration = (
    len(args.yaw_degrees)
    * len(args.lateral_offsets_mm)
    * len(args.tilt_degrees)
    * len(args.azimuth_degrees)
  )
  if configurations * trials_per_configuration > 5000:
    parser.error("geometry sweep is limited to 5000 dynamic trials")
  args.configuration_count = configurations
  args.trials_per_configuration = trials_per_configuration
  args.trial_count = configurations * trials_per_configuration
  args.output = args.output.expanduser().resolve()
  if args.output.exists():
    parser.error(f"refusing to overwrite output: {args.output}")
  return args


def _point(radius_m: float, sign: float, z_m: float) -> tuple[float, float, float]:
  return (
    radius_m * math.cos(_SECTOR_HALF_ANGLE_RAD),
    sign * radius_m * math.sin(_SECTOR_HALF_ANGLE_RAD),
    z_m,
  )


def _funnel_vertices(throat_radius_m: float, collar_top_z_m: float) -> str:
  if not 0.016 < throat_radius_m < _TOP_INNER_RADIUS_M:
    raise ValueError("throat radius must clear the bulb neck and remain below the top")
  if not _BOTTOM_Z_M < collar_top_z_m < _TOP_Z_M:
    raise ValueError("collar top z must lie inside the funnel mesh")
  vertices = (
    _point(throat_radius_m, -1, _BOTTOM_Z_M),
    _point(throat_radius_m, +1, _BOTTOM_Z_M),
    _point(_OUTER_RADIUS_M, +1, _BOTTOM_Z_M),
    _point(_OUTER_RADIUS_M, -1, _BOTTOM_Z_M),
    _point(throat_radius_m, -1, collar_top_z_m),
    _point(throat_radius_m, +1, collar_top_z_m),
    _point(_OUTER_RADIUS_M, +1, collar_top_z_m),
    _point(_OUTER_RADIUS_M, -1, collar_top_z_m),
    _point(_TOP_INNER_RADIUS_M, -1, _TOP_Z_M),
    _point(_TOP_INNER_RADIUS_M, +1, _TOP_Z_M),
    _point(_OUTER_RADIUS_M, +1, _TOP_Z_M),
    _point(_OUTER_RADIUS_M, -1, _TOP_Z_M),
  )
  return " ".join(f"{coordinate:.10f}" for vertex in vertices for coordinate in vertex)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while block := stream.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


@contextmanager
def _temporary_scene(
  source_scene: Path,
  *,
  throat_radius_m: float,
  collar_top_z_m: float,
  friction: float,
) -> Iterator[Path]:
  tree = ET.parse(source_scene)
  root = tree.getroot()
  mesh = root.find("./asset/mesh[@name='bulb_funnel_sector']")
  if mesh is None:
    raise ValueError("production scene has no bulb_funnel_sector mesh")
  mesh.set("vertex", _funnel_vertices(throat_radius_m, collar_top_z_m))
  geoms = [
    geom
    for geom in root.findall(".//geom")
    if geom.get("name", "").startswith("bulb_socket_funnel_")
  ]
  if len(geoms) != 24:
    raise ValueError(f"expected 24 funnel sectors, found {len(geoms)}")
  for geom in geoms:
    geom.set("friction", f"{friction:.10g} 0.001 0.0001")
  descriptor, raw_path = tempfile.mkstemp(
    prefix=".bulb_funnel_sweep_", suffix=".xml", dir=source_scene.parent
  )
  os.close(descriptor)
  path = Path(raw_path)
  try:
    tree.write(path, encoding="utf-8", xml_declaration=True)
    yield path
  finally:
    path.unlink(missing_ok=True)


def _approach_args(args: argparse.Namespace) -> SimpleNamespace:
  return SimpleNamespace(
    approach_speed_mm_s=args.approach_speed_mm_s,
    start_clearance_mm=args.start_clearance_mm,
    approach_duration_s=args.approach_duration_s,
    settle_duration_s=args.settle_duration_s,
    reverse_torque_nm=args.reverse_torque_nm,
    release_timeout_s=args.release_timeout_s,
    extract_force_n=args.extract_force_n,
    extract_timeout_s=args.extract_timeout_s,
  )


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
  valid = [row for row in rows if row.get("valid", True)]
  captured = [row for row in valid if row["captured"]]
  cleared = [row for row in captured if row["released"] and row["cleared"]]
  return {
    "trial_count": len(rows),
    "valid_trial_count": len(valid),
    "invalid_trial_count": len(rows) - len(valid),
    "capture_count": len(captured),
    "capture_fraction": len(captured) / len(valid) if valid else None,
    "captured_and_cleared_count": len(cleared),
    "captured_and_cleared_fraction": len(cleared) / len(valid) if valid else None,
    "peak_funnel_contact_n": max(
      (float(row["peak_funnel_contact_n"]) for row in valid), default=None
    ),
    "peak_fixture_contact_n": max(
      (float(row["peak_fixture_contact_n"]) for row in valid), default=None
    ),
    "outcomes": {
      outcome: sum(row["outcome"] == outcome for row in rows)
      for outcome in sorted({str(row["outcome"]) for row in rows})
    },
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  source_scene = default_model_path(config.SCENE_NAME)
  source_sha256 = _sha256(source_scene)
  common_args = _approach_args(args)
  configurations = []
  all_rows = []
  number = 0
  for throat_mm, collar_mm, friction in itertools.product(
    args.throat_radii_mm,
    args.collar_top_z_mm,
    args.funnel_friction,
  ):
    rows = []
    with _temporary_scene(
      source_scene,
      throat_radius_m=0.001 * throat_mm,
      collar_top_z_m=0.001 * collar_mm,
      friction=friction,
    ) as scene:
      simulation = BulbScrewSimulation(model_path=scene)
      for yaw, lateral, tilt, azimuth in itertools.product(
        args.yaw_degrees,
        args.lateral_offsets_mm,
        args.tilt_degrees,
        args.azimuth_degrees,
      ):
        number += 1
        row = basin._approach_row(
          simulation,
          common_args,
          yaw_deg=yaw,
          lateral_mm=lateral,
          tilt_deg=tilt,
          azimuth_deg=azimuth,
          lateral_stiffness_n_m=args.lateral_stiffness_n_m,
          angular_stiffness_nm_rad=args.angular_stiffness_nm_rad,
        )
        row.update(
          throat_radius_mm=throat_mm,
          collar_top_z_mm=collar_mm,
          funnel_friction=friction,
        )
        rows.append(row)
        all_rows.append(row)
        print(
          f"trial={number}/{args.trial_count} throat={throat_mm:g}mm "
          f"collar={collar_mm:g}mm mu={friction:g} offset={lateral:g}mm "
          f"tilt={tilt:g}deg azimuth={azimuth:g}deg outcome={row['outcome']}",
          flush=True,
        )
    configurations.append(
      {
        "throat_radius_mm": throat_mm,
        "collar_top_z_mm": collar_mm,
        "funnel_friction": friction,
        "score": _score(rows),
      }
    )
  if _sha256(source_scene) != source_sha256:
    raise RuntimeError("production bulb scene changed during the geometry sweep")
  ranked = sorted(
    configurations,
    key=lambda item: (
      -(item["score"]["captured_and_cleared_count"]),
      -(item["score"]["capture_count"]),
      item["score"]["invalid_trial_count"],
      item["score"]["peak_fixture_contact_n"] or math.inf,
    ),
  )
  return {
    "schema": "bulb_funnel_geometry_sweep_v1",
    "diagnostic_only": True,
    "created_utc": datetime.now(UTC).isoformat(),
    "production_scene": str(source_scene),
    "production_scene_sha256": source_sha256,
    "production_model_fingerprint": model_fingerprint(source_scene),
    "mechanics_version": config.MECHANICS_VERSION,
    "mujoco_version": mujoco.__version__,
    "arguments": {
      key: value for key, value in vars(args).items() if key not in {"output"}
    },
    "ranking": ranked,
    "trials": all_rows,
  }


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  report = run(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(basin._json_safe(report), indent=2, sort_keys=True) + "\n")
  print(json.dumps(report["ranking"][:5], indent=2), flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
