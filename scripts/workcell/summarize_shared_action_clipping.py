#!/usr/bin/env python3
"""Aggregate per-joint and per-stage action clipping from shared-task trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaihand_tactile_env.shared.action_diagnostics import (
  merge_action_clip_summaries,
)


def find_summaries(paths: list[Path]) -> list[Path]:
  found: set[Path] = set()
  for path in paths:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_file():
      if resolved.name != "action_clipping.json":
        raise ValueError(f"expected action_clipping.json, got {resolved}")
      found.add(resolved)
    elif (resolved / "action_clipping.json").is_file():
      found.add(resolved / "action_clipping.json")
    else:
      found.update(resolved.rglob("action_clipping.json"))
  if not found:
    raise FileNotFoundError("no action_clipping.json artifacts found")
  return sorted(found)


def summarize(paths: list[Path]) -> dict:
  artifacts = find_summaries(paths)
  summaries = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
  result = merge_action_clip_summaries(summaries)
  result["artifacts"] = [str(path) for path in artifacts]
  return result


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("paths", nargs="+", type=Path)
  parser.add_argument("--output", type=Path)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = parse_args(argv)
  result = summarize(args.paths)
  payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
  if args.output is None:
    print(payload, end="")
  else:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
