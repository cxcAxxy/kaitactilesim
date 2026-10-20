#!/usr/bin/env python3
"""Build the multi-task local H50 workbench from completed archived trials."""
import argparse
import json
from pathlib import Path

from evaluation_workbench.suite import discover_trials, export_suite, verify_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--trial-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous-workbench", type=Path)
    parser.add_argument("--task-demo-root", type=Path, default=Path("/nas/chenxianchi/datasets/sim/task_demo"))
    parser.add_argument("--workers", type=int, default=4, help="Offline CPU readers, not simulator/GPU jobs")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--complete-model-plots", type=Path, help="Fill missing USB figures in an existing new suite using model-only native records")
    parser.add_argument("--generate-missing-model-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.complete_model_plots:
        if args.verify or args.source_root or args.output or args.trial_dir:
            parser.error("--complete-model-plots is separate from export and verify arguments")
        from evaluation_workbench.suite_figures import complete_missing_model_plots
        print(json.dumps(complete_missing_model_plots(args.complete_model_plots, workers=args.workers), ensure_ascii=False)); return 0
    if args.verify:
        print(json.dumps(verify_suite(args.verify), ensure_ascii=False)); return 0
    if not args.output or not args.previous_workbench or not (args.source_root or args.trial_dir):
        parser.error("--output, --previous-workbench and --source-root or --trial-dir are required")
    if args.source_root:
        trials, selection = discover_trials(args.source_root, args.extra_root)
        trials.extend(path.resolve(strict=True) for path in args.trial_dir)
        trials = sorted(set(trials))
    else:
        trials, selection = [path.resolve(strict=True) for path in args.trial_dir], {"explicit_trials": [str(path) for path in args.trial_dir]}
    batch = export_suite(trials, args.output, previous_workbench=args.previous_workbench,
                         task_demo_root=args.task_demo_root, selection=selection, workers=args.workers)
    if args.generate_missing_model_plots:
        from evaluation_workbench.suite_figures import complete_missing_model_plots
        from evaluation_workbench.export import _document
        complete_missing_model_plots(args.output, workers=args.workers)
        batch = _document(args.output / "batch.json")
    print(json.dumps({"output": str(args.output), "trial_count": batch["trial_count"], "task_count": batch["task_count"],
                      "coverage": batch["coverage"], "mode_counts": batch["mode_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
