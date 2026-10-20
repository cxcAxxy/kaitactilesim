#!/usr/bin/env python3
"""Publish the original-style workbench with all archived policy comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h50-snapshot", type=Path)
    parser.add_argument("--tactile-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish-index", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    from evaluation_status.unified_workbench_export import (
        export_unified_workbench,
        verify_unified_workbench,
    )
    if args.verify:
        if args.h50_snapshot or args.tactile_snapshot or args.output or args.publish_index:
            parser.error("--verify cannot be combined with export options")
        result = verify_unified_workbench(args.verify)
    else:
        if not all((args.h50_snapshot, args.tactile_snapshot, args.output)):
            parser.error("--h50-snapshot, --tactile-snapshot and --output are required")
        result = export_unified_workbench(
            args.h50_snapshot, args.tactile_snapshot, args.output,
            publish_index=args.publish_index,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
