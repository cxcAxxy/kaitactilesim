#!/usr/bin/env python3
"""Export explicit task-demo HDF5 sources; originals are only read.

Example:
  python -B scripts/analysis/export_expert_reference.py \
    --replay-json /archive/task_demo/_metadata/task/replay.json \
    --output-dir /path/to/new/reference \
    --reference-id selected-task-demo --declared-by curator \
    --selection-rule 'One explicitly selected demonstration; success-selected, not a population' \
    --selection-evidence 'Archive task_demo replay metadata audited by curator' \
    --declare-expert-reference

Repeat --replay-json for distinct recordings. The exporter never invents
reference members, independent phase labels or sampling compatibility. Output
reports can be supplied to evaluate_suite.py --reference-report, with the
generated --reference-manifest; separate audited compatibility is still needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation_v2.expert import export_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--declared-by", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--selection-evidence", required=True)
    parser.add_argument("--declare-expert-reference", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_reference(args.replay_json, args.output_dir, reference_id=args.reference_id,
            declared_by=args.declared_by, selection_rule=args.selection_rule, selection_evidence=args.selection_evidence,
            declare_expert_reference=args.declare_expert_reference)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Expert reference export stopped: {error}\n")
    print(json.dumps({"output": str(args.output_dir.resolve()), "independent_recording_count": result["independent_recording_count"],
                      "comparison_status": result["comparison_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
