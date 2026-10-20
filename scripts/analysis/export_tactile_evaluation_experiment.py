#!/usr/bin/env python3
"""Export or verify the stopped five-model Poker tactile experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parents[1]
for path in (ROOT / "src", ANALYSIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation_status.tactile_experiment_export import (  # noqa: E402
    export_tactile_experiment,
    verify_tactile_experiment,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--h50-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-index", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        if any(value is not None for value in (args.job_root, args.h50_snapshot, args.output)):
            parser.error("--verify cannot be combined with export paths")
        if args.publish_index or args.require_complete:
            parser.error("publication flags do not apply to --verify")
        result = verify_tactile_experiment(args.verify)
    else:
        if None in (args.job_root, args.h50_snapshot, args.output):
            parser.error("export requires --job-root, --h50-snapshot and --output")
        result = export_tactile_experiment(
            args.job_root,
            args.h50_snapshot,
            args.output,
            publish_index=args.publish_index,
            require_complete=args.require_complete,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
