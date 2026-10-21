#!/usr/bin/env python3
"""Add only the independently verified ten-trial current CPM batch to r5."""
from __future__ import annotations

import argparse
import json

from evaluation_status.cpm_additive_workbench_export import (
    export_cpm_additive_workbench, publish_cpm_additive_workbench,
    verify_cpm_additive_workbench,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-root")
    parser.add_argument("--evaluation-root")
    parser.add_argument("--qa-receipt")
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_cpm_additive_workbench(args.output)
    elif args.publish_only:
        result = publish_cpm_additive_workbench(args.output)
    else:
        if not all((args.baseline, args.job_root, args.evaluation_root, args.qa_receipt)):
            parser.error("Export requires --baseline, --job-root, --evaluation-root, --qa-receipt")
        result = export_cpm_additive_workbench(args.baseline, args.output, job_root=args.job_root,
            evaluation_root=args.evaluation_root, qa_receipt=args.qa_receipt, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
