#!/usr/bin/env python3
"""Serial PDF-format export, full source audits, then session-level assembly."""

import argparse
import json
import time
from pathlib import Path

from kaihand_tactile_env.shared.tict_export import export_tict_episode
from kaihand_tactile_env.shared.tict_package import package_tict_releases
from kaihand_tactile_env.shared.tict_source_audit import audit_tict_source
from usb_delivery_common import attach_quality, save, selection


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source-root", type=Path, default=Path("datasets/usb_insert_0910")
  )
  parser.add_argument(
    "--processing",
    type=Path,
    default=Path("datasets/usb_insert_0910/processing/dual_format_0910"),
  )
  parser.add_argument(
    "--output", type=Path, default=Path("datasets/usb_inset_0910_egotouch")
  )
  parser.add_argument("--max-sessions", type=int, default=50)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  snapshot = selection(args.source_root, args.processing)
  processing = args.processing.resolve()
  inputs = []
  for row in snapshot["episodes"][: args.max_sessions]:
    started = time.monotonic()
    session = row["session_id"]
    release = processing / "tict_sessions" / session
    audit_path = processing / f"{session}_source_audit.json"
    print(f"EgoTouch {session}: export/verify", flush=True)
    if not release.exists():
      export_tict_episode(row["source"], release, session_id=session, camera="head")
    if audit_path.exists():
      audit = json.loads(audit_path.read_text())
    else:
      audit = audit_tict_source(row["source"], release, session)
      save(audit_path, audit)
    if not audit["valid"] or audit.get("source", {}).get("sha256") != row["sha256"]:
      raise RuntimeError(f"source audit failed: {audit_path}: {audit.get('errors')}")
    inputs.append(
      {
        "session_id": session,
        "release_dir": str(release),
        "split": "validation" if row["split"] == "val" else "train",
        "source_audit_path": str(audit_path),
      }
    )
    print(f"EgoTouch {session}: passed ({time.monotonic() - started:.1f}s)", flush=True)
  if len(inputs) != 50:
    print("Pilot finished; full package not assembled yet", flush=True)
    return
  manifest = processing / "tict_package_input.json"
  if not manifest.exists():
    save(
      manifest, {"schema_version": "kaihand-tict-package-input-v1", "sessions": inputs}
    )
  print("EgoTouch: assembling 50 sessions and train-only statistics", flush=True)
  report = package_tict_releases(manifest, args.output)
  attach_quality(args.output, snapshot)
  save(processing / "egotouch_package_result.json", report)
  print(f"EgoTouch complete: {args.output}", flush=True)


if __name__ == "__main__":
  main()
