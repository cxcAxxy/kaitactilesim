#!/usr/bin/env python3
"""Unified local evaluation: inspect rules, import staged records, export, verify.

This entry point is offline only; actual model rollouts retain the existing
scripts/evaluate/evaluate.py interface. No simulation/GPU jobs are started here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rules", help="Print the current pure-Python shared rule catalog; does not rescore archives")
    discover = commands.add_parser("discover", help="List completed H50 staged trials and their archived rule identities")
    discover.add_argument("--source-root", type=Path, required=True)
    discover.add_argument("--extra-root", action="append", type=Path, default=[])
    export = commands.add_parser("export", help="Import existing records into a fresh standalone local report")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--extra-root", action="append", type=Path, default=[])
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--previous-workbench", type=Path)
    export.add_argument("--task-demo-root", type=Path, default=Path("/nas/chenxianchi/datasets/sim/task_demo"))
    export.add_argument("--workers", type=int, default=4)
    export.add_argument("--trial-limit", type=int)
    export.add_argument("--no-model-plots", action="store_true")
    verify = commands.add_parser("verify", help="Verify report/source hashes and preservation of both original projects")
    verify.add_argument("output", type=Path)
    commands.add_parser("verify-originals", help="Verify the original working trees remain unchanged")
    args = parser.parse_args(argv)
    if args.command == "rules":
        from kaihand_tactile_env.shared.evaluation_rules import export_rules
        result = export_rules()
    elif args.command == "discover":
        from evaluation_unified.staged import discover_staged_trials
        result = discover_staged_trials(args.source_root, extra_roots=args.extra_root)
    elif args.command == "verify-originals":
        from evaluation_unified.export import verify_original_sources
        result = verify_original_sources()
    elif args.command == "verify":
        from evaluation_unified.export import verify_unified
        result = verify_unified(args.output)
    else:
        from evaluation_unified.export import export_staged
        options = {"extra_roots": args.extra_root, "task_demo_root": args.task_demo_root,
                   "workers": args.workers, "generate_model_plots": not args.no_model_plots, "trial_limit": args.trial_limit}
        if args.previous_workbench:
            options["previous"] = args.previous_workbench
        batch = export_staged(args.source_root, args.output, **options)
        result = {"output": str(args.output), "trial_count": batch["trial_count"], "task_count": batch["task_count"],
                  "coverage": batch["coverage"], "mode_counts": batch["mode_counts"], "profile": batch["evaluation_profile"]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
