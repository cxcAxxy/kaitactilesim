#!/usr/bin/env python3
"""Append the explicit 25-trial bulb-inplace job to the accepted 595-trial site."""
from __future__ import annotations

import argparse
import json

from evaluation_status.additive_workbench_export import (
    export_additive_workbench,
    publish_additive_workbench,
    verify_additive_workbench,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-root")
    parser.add_argument("--evaluation-root")
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_additive_workbench(args.output)
    elif args.publish_only:
        result = publish_additive_workbench(args.output)
    else:
        if not args.baseline or not args.job_root or not args.evaluation_root:
            parser.error("Export requires --baseline, --job-root and --evaluation-root")
        result = export_additive_workbench(args.baseline, args.output, job_root=args.job_root,
                                           evaluation_root=args.evaluation_root, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
