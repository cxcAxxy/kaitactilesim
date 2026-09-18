#!/usr/bin/env python3
"""Validate the local documented T-ICT release (without a training dependency)."""

import argparse
import json

from kaihand_tactile_env.shared.tict_validation import validate_tict_release


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("release_root")
  args = parser.parse_args()
  result = validate_tict_release(args.release_root)
  print(json.dumps(result, indent=2, ensure_ascii=False))
  if not result["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
