#!/usr/bin/env python3
"""Wait for an existing EgoTouch export, then serially export and verify EgoSteer.

Never starts a second EgoTouch job. Durable status/logs survive client disconnects.
Intermediate cleanup is restricted to payloads proven hard-linked into a valid
published EgoTouch release; no original HDF5 or other dataset is deleted.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaihand_tactile_env.shared.tict_validation import validate_tict_release


def read(path):
    return json.loads(path.read_text())


def status(processing, phase, **fields):
    value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    temporary = processing / "delivery_status.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(processing / "delivery_status.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def remove_linked_intermediates(processing, output, sessions):
    intermediate = processing / "tict_sessions"
    if not intermediate.exists():
        return {"removed": False, "reason": "no intermediates"}
    if intermediate.is_symlink() or intermediate.resolve().parent != processing.resolve():
        raise ValueError("unsafe intermediate root")
    if {p.name for p in intermediate.iterdir()} != set(sessions):
        raise ValueError("unexpected intermediate entries; refusing recursive cleanup")
    count = 0
    for session in sessions:
        root = intermediate / session
        if root.is_symlink() or not root.is_dir():
            raise ValueError("unsafe intermediate session")
        for branch in ("production", "tict_sidecars"):
            if (root / branch).is_symlink() or not (root / branch).is_dir():
                raise ValueError("unsafe intermediate payload branch")
            for path in (root / branch).rglob("*"):
                if path.is_symlink():
                    raise ValueError("unexpected intermediate symlink")
                if path.is_file():
                    target = output / path.relative_to(root)
                    if target.is_symlink() or not target.is_file() or not os.path.samefile(path, target):
                        raise ValueError(f"payload not preserved in final release: {path}")
                    count += 1
    # Exact generated root, after every exported payload has been checked above.
    shutil.rmtree(intermediate)
    return {"removed": True, "path": str(intermediate), "preserved_payload_files": count,
            "recoverability": "identical payloads remain as ordinary files in final EgoTouch release"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--processing", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    parser.add_argument("--wait-seconds", type=int, default=14400)
    args = parser.parse_args()
    source, processing = args.source_root.resolve(), args.processing.resolve()
    if not processing.is_relative_to(source / "processing"):
        raise ValueError("processing must belong to this source dataset")
    touch = source.with_name(source.name + "_egotouch")
    steer = source.with_name(source.name + "_egosteer")
    with (processing / "delivery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            started = time.monotonic()
            result_path = processing / "egotouch_package_result.json"
            while True:
                try:
                    result = read(result_path)
                except (FileNotFoundError, json.JSONDecodeError):
                    result = None
                if result is not None:
                    if result.get("valid") is not True or result.get("output") != str(touch):
                        raise ValueError("EgoTouch result does not match the requested release")
                    break
                audits = list(processing.glob("*_source_audit_recorded.json"))
                finished = 0
                for path in audits:
                    try:
                        audit = read(path)
                    except json.JSONDecodeError:
                        continue
                    if audit.get("valid") is not True:
                        raise ValueError(f"EgoTouch source audit failed: {path}")
                    finished += 1
                status(processing, "waiting_for_egotouch", verified_sessions=finished,
                       expected_sessions=args.expected_episodes)
                if time.monotonic() - started > args.wait_seconds:
                    raise TimeoutError("EgoTouch has not published a completed result; no second conversion started")
                time.sleep(30)
            snapshot = read(processing / "source_selection.json")
            rows = snapshot["episodes"]
            if snapshot["episode_count"] != args.expected_episodes or len(rows) != args.expected_episodes:
                raise ValueError("selection count mismatch")
            status(processing, "converting_egosteer", expected_sessions=len(rows))
            if not steer.exists():
                command = [sys.executable, str(Path(__file__).with_name("package_poker_egosteer.py")),
                           "--source-root", str(source), "--processing", str(processing),
                           "--output", str(steer), "--expected-episodes", str(len(rows))]
                with (processing / "egosteer_conversion.log").open("a") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            status(processing, "final_verification")
            for output in (touch, steer):
                if read(output / "SOURCE_SELECTION.json") != snapshot:
                    raise ValueError("source selection or split differs between formats")
            splits = read(touch / "split_manifest.json")["splits"]
            expected_splits = {"train": [], "validation": [], "test": []}
            for row in rows:
                expected_splits["validation" if row["split"] == "val" else "train"].append(row["session_id"])
            if splits != expected_splits:
                raise ValueError("EgoTouch split mismatch")
            # Verify again at the published path, including absolute RGB selectors.
            touch_validation = validate_tict_release(touch)
            if not touch_validation["valid"] or len(touch_validation["sessions"]) != len(rows):
                raise ValueError(f"published EgoTouch invalid: {touch_validation['errors']}")
            steer_validation = read(steer / "validation.json")
            steer_audit = read(steer / "source_audit.json")
            if (not steer_validation["valid"] or steer_validation["episodes"] != len(rows)
                    or not steer_audit["valid"] or len(steer_audit["episodes"]) != len(rows)):
                raise ValueError("EgoSteer schema/source validation incomplete")
            manifest = read(steer / "dataset_manifest.json")
            conversions = manifest["sources"]
            if {r["episode_index"]: r["split"] for r in conversions} != {r["episode_index"]: r["split"] for r in rows}:
                raise ValueError("EgoSteer episode split mismatch")
            status(processing, "removing_verified_duplicate_intermediates")
            cleanup = remove_linked_intermediates(processing, touch, [r["session_id"] for r in rows])
            report = dict(valid=True, episodes=len(rows), train=len(splits["train"]),
                          validation=len(splits["validation"]), egotouch_output=str(touch),
                          egosteer_output=str(steer),
                          egotouch_frames=sum(r["frame_count"] for r in touch_validation["sessions"]),
                          egotouch_windows=sum(r["window_count"] for r in touch_validation["sessions"]),
                          egosteer_samples=steer_validation["samples"], cleanup=cleanup,
                          original_data_modified=False, upstream_training_executed=False,
                          cleaning_warnings="retained; no unconditional manual quality approval",
                          elapsed_seconds=time.monotonic() - started)
            path = processing / "delivery_verification.json"
            with path.open("x") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            status(processing, "complete", **report)
        except Exception as error:
            status(processing, "failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
