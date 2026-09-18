#!/usr/bin/env python3
"""Run read-only USB Cleaning checks serially on one or more raw HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _name in (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
):
  os.environ[_name] = "1"


def main(argv=None):
  from kaihand_tactile_env.shared.usb_cleaning import (
    audit_usb_cleaning,
    write_transition_indices,
  )

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    type=Path,
    required=True,
    action="append",
    help="Repeat for each episode; files are checked one at a time",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="New directory for per-episode reports and summary.json",
  )
  parser.add_argument(
    "--with-actuator-audit",
    action="store_true",
    help="Also independently reproduce affine actuator response from the static model (no physics stepping)",
  )
  args = parser.parse_args(argv)
  if args.output_dir.exists():
    parser.error(f"--output-dir already exists: {args.output_dir}")
  paths = [path.resolve() for path in args.input]
  if len(set(paths)) != len(paths):
    parser.error("the same input path was supplied more than once")
  args.output_dir.mkdir(parents=True)
  rows, seen, batch_errors = [], {}, []
  for index, path in enumerate(paths):
    report = audit_usb_cleaning(path)
    if args.with_actuator_audit:
      from kaihand_tactile_env.shared.usb_action_audit import audit_usb_actions

      try:
        action_report = audit_usb_actions(path)
      except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
        action_report = {
          "valid": False,
          "errors": [
            f"cannot complete actuator audit: {type(error).__name__}: {error}"
          ],
        }
      report["actuator_response_audit"] = action_report
      if not action_report["valid"]:
        report["errors"].extend(
          f"actuator response: {error}" for error in action_report["errors"]
        )
        report["valid"] = False
    if report["valid"]:
      sidecar_name = f"episode_{index:03d}_{path.stem}_transitions.npz"
      write_transition_indices(path, args.output_dir / sidecar_name)
      report["transition_index_sidecar"] = sidecar_name
    fingerprint = report.get("state_action_fingerprint_sha256")
    if fingerprint and fingerprint in seen:
      batch_errors.append(f"duplicate episode content: {path} and {seen[fingerprint]}")
    elif fingerprint:
      seen[fingerprint] = str(path)
    name = f"episode_{index:03d}_{path.stem}.json"
    with (args.output_dir / name).open("x", encoding="utf-8") as stream:
      json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
      stream.write("\n")
    row = {
      "input": str(path),
      "report": name,
      "valid": report["valid"],
      "review_required": report["review_required"],
      "errors": report["errors"],
      "warnings": report["warnings"],
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
  summary = {
    "schema": "usb_raw_cleaning_batch_v1",
    "valid": all(row["valid"] for row in rows) and not batch_errors,
    "episodes_checked": len(rows),
    "successful_clean_episodes": sum(row["valid"] for row in rows),
    "five_complete_raw_episodes_passed": len(rows) >= 5
    and all(row["valid"] for row in rows)
    and not batch_errors,
    "actuator_response_audit_requested": args.with_actuator_audit,
    "review_required": any(row["review_required"] for row in rows),
    "visual_tactile_review_complete": False,
    "errors": batch_errors,
    "episodes": rows,
    "note": "Raw clock, frame and trajectory checks are always run. Use --with-actuator-audit for independent action-response equations. Five passing episodes still require tactile boundary image review and do not guarantee future batches; continue checking later episodes.",
  }
  with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
  print(
    json.dumps(
      {key: value for key, value in summary.items() if key != "episodes"},
      ensure_ascii=False,
    ),
    flush=True,
  )
  if not summary["valid"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
