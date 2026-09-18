#!/usr/bin/env python3
"""Validate one workcell HDF5 file against the v1 schema."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from kaihand_tactile_env.shared.recording import validate_episode


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("episode")
  args = parser.parse_args()
  report = validate_episode(args.episode)
  print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
  raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
  main()
