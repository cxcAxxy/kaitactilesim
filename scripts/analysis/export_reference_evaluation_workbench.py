#!/usr/bin/env python3
"""Restore source-bound reference values and terminal bulb angles offline."""
import argparse
import json

from evaluation_status.reference_workbench_export import (
    export_reference_workbench,
    publish_reference_workbench,
    verify_reference_workbench,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-root")
    parser.add_argument("--workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.publish_only:
        result = publish_reference_workbench(args.output)
    elif args.verify_only:
        result = verify_reference_workbench(args.output)
    else:
        if not args.baseline or not args.evaluation_root:
            parser.error("Export requires --baseline and --evaluation-root")
        result = export_reference_workbench(args.baseline, args.output,
                                           evaluation_root=args.evaluation_root, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
