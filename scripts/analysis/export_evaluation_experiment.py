#!/usr/bin/env python3
"""Publish a unified checkpoint view of an existing H50 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-snapshot", type=Path)
    parser.add_argument("--historical-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-index", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    from evaluation_status.experiment_export import export_experiment, verify_experiment
    if args.verify:
        if args.status_snapshot or args.historical_report or args.output or args.publish_index:
            parser.error("--verify cannot be combined with export options")
        result = verify_experiment(args.verify)
    else:
        if not all((args.status_snapshot, args.historical_report, args.output)):
            parser.error("--status-snapshot, --historical-report and --output are required")
        result = export_experiment(args.status_snapshot, args.historical_report, args.output,
                                   publish_index=args.publish_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
