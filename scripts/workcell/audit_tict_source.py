#!/usr/bin/env python3
"""Cross-check a finalized HDF5 episode against every T-ICT release frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.tict_source_audit import audit_tict_source


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--release-root", type=Path, required=True)
  parser.add_argument("--session-id", required=True)
  parser.add_argument("--camera", default="head")
  parser.add_argument("--source-code-policy", choices=("current", "recorded"), default="current")
  parser.add_argument(
    "--output",
    type=Path,
    help="Optional new audit JSON; existing paths are never overwritten",
  )
  args = parser.parse_args(argv)
  if args.output is not None and args.output.exists():
    parser.error(f"--output already exists: {args.output}")
  report = audit_tict_source(
    args.input, args.release_root, args.session_id, args.camera,
    source_code_policy=args.source_code_policy
  )
  serialized = (
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    + "\n"
  )
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
      stream.write(serialized)
  print(serialized, end="")
  if not report["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
