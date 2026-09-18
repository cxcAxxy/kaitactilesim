#!/usr/bin/env python3
"""Serial PDF-format export, full source audits, then session-level assembly."""

import argparse
import json
import time
from pathlib import Path

from kaihand_tactile_env.shared.tict_export import export_tict_episode
from kaihand_tactile_env.shared.tict_package import package_tict_releases
from kaihand_tactile_env.shared.tict_source_audit import audit_tict_source
from poker_delivery_common import attach_quality, save, selection


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, default=Path("datasets/poker_draw_0909"))
  parser.add_argument("--processing", type=Path, default=Path("datasets/poker_draw_0909/processing/dual_format_20260909"))
  parser.add_argument("--output", type=Path, default=Path("datasets/poker_draw_0909_egotouch"))
  parser.add_argument("--max-sessions", type=int, default=50)
  parser.add_argument("--expected-episodes", type=int, default=50)
  parser.add_argument("--reuse-source-audits", action="store_true",
                      help="Reuse passing source audits; validate the assembled output once, without revalidating each input")
  parser.add_argument("--source-code-policy", choices=("current", "recorded"), default="current")
  parser.add_argument("--link-payloads", action="store_true",
                      help="Hard-link immutable intermediate payloads during assembly to avoid duplicate disk usage")
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  snapshot = selection(args.source_root, args.processing, args.expected_episodes)
  processing = args.processing.resolve()
  inputs = []
  for row in snapshot["episodes"][:args.max_sessions]:
    started = time.monotonic()
    session = row["session_id"]
    release = processing / "tict_sessions" / session
    audit_path = processing / f"{session}_source_audit_{args.source_code_policy}.json"
    print(f"EgoTouch {session}: export/verify", flush=True)
    if not release.exists():
      export_tict_episode(row["source"], release, session_id=session, camera="head")
    if audit_path.exists():
      audit = json.loads(audit_path.read_text())
    else:
      audit = audit_tict_source(row["source"], release, session,
                                source_code_policy=args.source_code_policy)
      save(audit_path, audit)
    if not audit["valid"] or audit.get("source", {}).get("sha256") != row["sha256"]:
      raise RuntimeError(f"source audit failed: {audit_path}: {audit.get('errors')}")
    inputs.append({"session_id": session, "release_dir": str(release),
                   "split": "validation" if row["split"] == "val" else "train",
                   "source_audit_path": str(audit_path)})
    print(f"EgoTouch {session}: passed ({time.monotonic()-started:.1f}s)", flush=True)
  if len(inputs) != snapshot["episode_count"]:
    print("Pilot finished; full package not assembled yet", flush=True)
    return
  manifest = processing / "tict_package_input.json"
  if not manifest.exists():
    save(manifest, {"schema_version": "kaihand-tict-package-input-v1", "sessions": inputs})
  print(f"EgoTouch: assembling {len(inputs)} sessions and train-only statistics", flush=True)
  report = package_tict_releases(manifest, args.output, link_payloads=args.link_payloads,
                                 reuse_source_audits=args.reuse_source_audits)
  attach_quality(args.output, snapshot)
  save(processing / "egotouch_package_result.json", report)
  print(f"EgoTouch complete: {args.output}", flush=True)


if __name__ == "__main__":
  main()
