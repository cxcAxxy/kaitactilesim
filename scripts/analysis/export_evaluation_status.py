#!/usr/bin/env python3
"""Publish or verify a fresh, read-only native evaluation status snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-report", type=Path)
    parser.add_argument("--publish-index", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    from evaluation_status.export import export_status, verify_status
    if args.verify:
        if args.source_root or args.output or args.historical_report or args.publish_index:
            parser.error("--verify cannot be combined with export options")
        result = verify_status(args.verify)
    else:
        if not all((args.source_root, args.output, args.historical_report)):
            parser.error("--source-root, --output and --historical-report are required")
        result = export_status(args.source_root, args.output, args.historical_report, publish_index=args.publish_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
