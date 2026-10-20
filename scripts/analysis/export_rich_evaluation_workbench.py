#!/usr/bin/env python3
"""Offline rich-detail migration of an accepted unified evaluation snapshot."""
import argparse
import json

from evaluation_status.rich_workbench_export import (
    export_rich_workbench,
    publish_rich_workbench,
    verify_rich_workbench,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-root")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--publish-index", action="store_true")
    args = parser.parse_args()
    if args.publish_only:
        publish_rich_workbench(args.output)
        result = verify_rich_workbench(args.output)
    elif args.verify_only:
        result = verify_rich_workbench(args.output)
    else:
        if not args.baseline or not args.evaluation_root or not 1 <= args.workers <= 16:
            parser.error("export requires --baseline, --evaluation-root, and 1..16 workers")
        result = export_rich_workbench(args.baseline, args.output, evaluation_root=args.evaluation_root,
                                       workers=args.workers, publish_index=args.publish_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
