#!/usr/bin/env python3
"""Evidence-aware offline suite and local website; never edits simulator sources.

Examples (from kaitactilesim_laest):
  .pixi/envs/default/bin/python -B scripts/analysis/evaluate_suite.py \
      --trial-dir /path/seed_000 --output-dir reports/new_suite
  .pixi/envs/default/bin/python -B scripts/analysis/evaluate_suite.py \
      --status-json /path/status.json --output-dir reports/finished_snapshot
  .pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py \
      --directory reports/new_suite --port 8877

After a simulator batch exits successfully, call this CLI with --batch-dir.
--status-json reads only completed groups, never interrupts running groups.
References require explicit membership and compatibility evidence; they are
not inferred from success or matched by elapsed time. All outputs are new.
Default cohort: explicitly recorded H50 prediction horizon only; H30, unknown,
conflicting and invalid horizon declarations are excluded before computation.
Prediction horizon is not execute_steps or the sensor/control sample rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_trials import discover_trials
from evaluation_v2.status_input import completed_status_trials
from evaluation_v2.suite import export_suite


def read_config(path: Path | None, provenance: list):
    if path is None:
        return None
    path = path.resolve(strict=True)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Configuration changed during read: {path}")
    value = json.loads(raw)
    provenance.append({"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "role": "explicit_suite_configuration_or_reference_report"})
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-dir", type=Path, action="append", default=[])
    parser.add_argument("--batch-dir", type=Path, action="append", default=[])
    parser.add_argument("--status-json", type=Path, help="Read-only snapshot of explicitly completed groups")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--task-configs", type=Path, help="Task-keyed contracts; never implicit guessed safety thresholds")
    parser.add_argument("--reference-report", type=Path, action="append", default=[])
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-compatibility", type=Path, help="Trial-ID keyed evidence-backed compatibility declarations")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--prediction-horizon", type=int, default=50,
                        help="Require an explicitly evidenced prediction horizon (default: 50); never infer it from execution steps or Hz")
    args = parser.parse_args(argv)
    try:
        trials = discover_trials(args.trial_dir, args.batch_dir) if args.trial_dir or args.batch_dir else []
        selection = None
        if args.status_json:
            selected, selection = completed_status_trials(args.status_json)
            trials += selected
        if not trials:
            raise ValueError("No completed trials selected; specify trial/batch directories or a completed status group")
        provenance = []
        protocol = read_config(args.protocol, provenance)
        configs = read_config(args.task_configs, provenance)
        references = [read_config(path, provenance) for path in args.reference_report]
        manifest = read_config(args.reference_manifest, provenance)
        compatibility = read_config(args.reference_compatibility, provenance)
        batch = export_suite(trials, args.output_dir, protocol, include_video=not args.no_video,
                             task_configs=configs, reference_reports=references,
                             reference_manifest=manifest, compatibility=compatibility,
                             configuration_provenance=provenance, selection=selection,
                             required_prediction_horizon=args.prediction_horizon)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Suite export stopped: {exc}\n")
    print(json.dumps({"output": str(args.output_dir.resolve()), "trial_count": batch["trial_count"],
                      "task_count": len(batch["by_task"]), "validation": "coverage only; see validation.json",
                      "prediction_horizon": batch["horizon_selection"]["required_prediction_horizon"],
                      "excluded_trial_count": len(batch["horizon_selection"]["excluded"]),
                      "reference_status": batch["reference_comparison"]["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
