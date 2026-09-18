#!/usr/bin/env python3
"""Publish one successful HDF5 episode in the PDF's T-ICT release layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.tict_export import export_tict_episode


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input", type=Path, required=True, help="Finalized successful .h5 episode"
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="New release directory; never overwritten",
  )
  parser.add_argument("--session-id", required=True)
  parser.add_argument("--camera", default="head")
  parser.add_argument("--horizon", type=int, default=50, choices=(50,))
  parser.add_argument("--max-sync-ms", type=float, default=20.0)
  args = parser.parse_args(argv)
  if not 0 <= args.max_sync_ms <= 20:
    parser.error("--max-sync-ms must be finite and in [0,20]")
  return args


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv)
  result = export_tict_episode(
    args.input,
    args.output_dir,
    session_id=args.session_id,
    camera=args.camera,
    horizon=args.horizon,
    max_sync_error_ns=int(round(args.max_sync_ms * 1_000_000)),
  )
  print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
