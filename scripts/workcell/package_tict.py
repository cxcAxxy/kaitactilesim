#!/usr/bin/env python3
"""Assemble complete T-ICT sessions or repair selector paths after transfer."""

import argparse
import json

from kaihand_tactile_env.shared.tict_package import (
  package_tict_releases,
  rebase_tict_release,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  actions = parser.add_subparsers(dest="action", required=True)
  build = actions.add_parser(
    "build", help="copy explicitly split sessions into a new package"
  )
  build.add_argument(
    "--manifest", required=True, help="kaihand-tict-package-input-v1 JSON"
  )
  build.add_argument(
    "--output",
    required=True,
    help="new output directory; existing data is never replaced",
  )
  rebase = actions.add_parser(
    "rebase", help="verify bytes and rebuild selector absolute paths"
  )
  rebase.add_argument(
    "--root", required=True, help="received release directory at its new location"
  )
  args = parser.parse_args()
  try:
    result = (
      package_tict_releases(args.manifest, args.output)
      if args.action == "build"
      else rebase_tict_release(args.root)
    )
  except (ValueError, OSError, KeyError, TypeError) as error:
    parser.exit(1, f"{type(error).__name__}: {error}\n")
  print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
