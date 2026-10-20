#!/usr/bin/env python3
"""Publish a new layout-only H50 workbench without changing prior reports."""
import argparse
import json
from pathlib import Path

from evaluation_workbench.export import export_workbench, refresh_web, verify_rollback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--verify-rollback", type=Path)
    parser.add_argument("--refresh-web", type=Path)
    parser.add_argument("--expert-manifest", type=Path)
    parser.add_argument("--previous-workbench", type=Path)
    parser.add_argument("--diagnostic-plots", action="store_true", help="Attach the selected model trial's existing three diagnostic PNGs")
    parser.add_argument("--diagnostic-example", type=Path, help="Explicit bulb H50 seed trial with existing diagnostic PNGs")
    args = parser.parse_args(argv)
    if ((args.verify_rollback or args.refresh_web) and (args.source_report or args.output or args.expert_manifest or args.previous_workbench
                                                     or args.diagnostic_plots or args.diagnostic_example)
            or args.verify_rollback and args.refresh_web):
        parser.error("--verify-rollback and --refresh-web are separate from each other and export arguments")
    try:
        if args.verify_rollback:
            result = verify_rollback(args.verify_rollback)
        elif args.refresh_web:
            result = refresh_web(args.refresh_web)
        else:
            if not args.source_report or not args.output:
                parser.error("--source-report and --output are required")
            config = export_workbench(args.source_report, args.output, expert_manifest=args.expert_manifest,
                                      previous_workbench=args.previous_workbench, diagnostic_plots=args.diagnostic_plots,
                                      diagnostic_example=args.diagnostic_example)
            result = {"output": str(args.output.resolve()), "trial_id": config["trial_id"],
                      "original_and_h50_unchanged": True, "fn_ft_recomputed": False}
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Workbench export stopped: {error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
