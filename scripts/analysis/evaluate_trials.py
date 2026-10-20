#!/usr/bin/env python3
"""Stage 1: evaluate archived trials without replay or expert-clock matching.

Run from the repository root with an interpreter that provides NumPy:
  .pixi/envs/default/bin/python -B scripts/analysis/evaluate_trials.py \
      --trial-dir /path/seed_000 --output-dir /path/new_report
  .pixi/envs/default/bin/python -B scripts/analysis/evaluate_trials.py \
      --batch-dir /path/checkpoint_trials --trial-dir /path/another_task/seed_000 \
      --output-dir /path/new_report

Inputs: summary.json; optional control_steps.jsonl and
review/evaluation_rollout_trace.npz; review/frames.jsonl is a labelled fallback.
Outputs: visual index.html, batch.json, frozen protocol.json, component assets,
integration.html, and per-trial report.html/report.json/visualization.json.
The original-style detailed tables remain in audit.html / batch_audit.html.
Recorded review videos are copied when available; --no-video omits media.
Every output directory must be new and outside the original trial directories.
No simulation is run, no original report is replaced, and no expert is needed.

Metrics: recorded outcome, recorded safety checks, elapsed times, normal-force
magnitude, physical-window changes, observed contact continuity, per-channel
load shares, normalized taxel redistribution, and geometry-supported centers.
Task evidence plugins: whiteboard variants, RAM/USB insertion, sponge grasp;
unknown tasks retain explicit archived outcome and generic tactile diagnostics.
No universal tactile score, inferred slip truth, or guessed completion rule.

Protocol: --protocol accepts JSON numeric overrides or a saved full protocol.
Defaults: contact on/off 0.05/0.03 N, minimum event duration 0.1 s, analysis bin
and change lag 0.1 s. These are diagnostic definitions, not validated safety or
hardware-calibration thresholds. Freeze them before comparing policy batches.
Only native-sample-aligned task intervals receive conditional tactile metrics;
unverified clocks and between-sample boundaries are explicitly unavailable.

Aggregation keeps failures/unknowns and uses one scalar per episode. Raw force
metrics stay within task/checkpoint/controller/sampling cohorts; task macro
averages describe supplied data and are not a cross-policy benchmark ranking.

Not implemented in Stage 1: independent outcome re-certification, calibrated
hardware ingestion, validated slip/expected-contact tests, expert distribution
baselines, perturbation sweeps, and causal tactile-input ablations.

Tests:
  .pixi/envs/default/bin/python -B -m pytest -q scripts/analysis/tests tests \
      -p no:cacheprovider
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from evaluation_v1.adapters import load_trial
from evaluation_v1.core import aggregate_reports, evaluate_trial, implementation_manifest, _verify_provenance
from evaluation_v1.dashboard import read_assets, render_dashboard, render_integration, render_visual_batch
from evaluation_v1.media import inspect_media
from evaluation_v1.protocol import content_hash, make_protocol
from evaluation_v1.render import render_batch, render_report
from evaluation_v1.visualization import build_visualization


def discover_trials(trial_dirs, batch_dirs) -> list[Path]:
    paths = [Path(path).resolve(strict=True) for path in trial_dirs]
    for directory in batch_dirs:
        root = Path(directory).resolve(strict=True)
        seeds = sorted(path.parent for path in root.glob("seed_*/summary.json"))
        if not seeds:
            raise ValueError(f"No completed seed summaries directly under {root}")
        paths.extend(seeds)
    if not paths:
        raise ValueError("Specify --trial-dir or --batch-dir")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate trial directories are not allowed")
    for path in paths:
        if not path.is_dir() or not (path / "summary.json").is_file():
            raise ValueError(f"Missing trial summary: {path}")
    return paths


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visual_package(report: dict, include_video: bool) -> tuple[dict, tuple | None]:
    """Read the same recorded snapshot again, never a silently newer version."""
    trial = load_trial(report["trial"]["source"])
    expected = {row["path"]: row["sha256"] for row in report["provenance"]}
    current = {row["path"]: row["sha256"] for row in trial.provenance}
    if set(current) != set(expected):
        raise ValueError("Primary evidence selection changed before visualization; optional inputs appeared or disappeared")
    for row in trial.provenance:
        if row["path"] in expected and expected[row["path"]] != row["sha256"]:
            raise ValueError(f"Input changed before visualization: {row['path']}")
    _verify_provenance(report["provenance"])
    visualization = build_visualization(trial, report)
    media = inspect_media(trial) if include_video else {
        "status": "missing", "sync_status": "unverified", "reason": "Video export disabled by --no-video",
        "time_s": [], "video_time_s": [],
    }
    media_copy = None
    if media.get("status") == "available":
        source = Path(media["source_path"]).resolve(strict=True)
        if not source.is_relative_to(trial.root) or source.suffix.lower() != ".mp4":
            raise ValueError("Only recorded MP4 files inside the selected trial may be exported")
        before = source.stat()
        digest = _file_digest(source)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Recorded video changed while hashing")
        record = {"path": str(source), "size": before.st_size, "sha256": digest,
                  "role": "copied_review_video"}
        trial.provenance.append(record)
        media_copy = (source, "media/rollout.mp4", digest)
    provenance = {row["path"]: row for row in report["provenance"]}
    for row in trial.provenance + media.get("provenance", []):
        if row["path"] in provenance and provenance[row["path"]]["sha256"] != row["sha256"]:
            raise ValueError(f"Input changed while preparing media: {row['path']}")
        provenance[row["path"]] = row
    report["provenance"] = list(provenance.values())
    visualization["media"] = {key: value for key, value in media.items()
                                if key not in ("source_path", "provenance", "export_name")}
    visualization["media"]["url"] = media_copy[1] if media_copy else None
    if media_copy:
        visualization["media"]["sha256"] = media_copy[2]
    report["visualization"] = {
        "schema_version": visualization["schema_version"], "path": "visualization.json",
        "sha256": content_hash(visualization),
        "semantics": "Display data only; native curves and selected native spatial frames do not recompute metrics",
    }
    _verify_provenance(report["provenance"])
    return visualization, media_copy


def export_reports(trials: list[Path], output: Path, protocol: dict, *, include_video: bool = True) -> dict:
    protocol = make_protocol(protocol)
    trials = [Path(trial).resolve(strict=True) for trial in trials]
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    for trial in trials:
        if output == trial or output.is_relative_to(trial):
            raise ValueError("Output must be outside original trial directories")
    # Validate and compute before creating any output artifacts.
    reports, visualizations, media_copies = [], [], []
    for i, trial in enumerate(trials):
        report = evaluate_trial(trial, protocol)
        task = re.sub(r"[^a-zA-Z0-9_-]", "_", report["trial"]["task"])
        slug = f"{i:03d}_{task}_{trial.name}"
        report["report_path"] = f"{slug}/report.html"
        visualization, media_copy = _visual_package(report, include_video)
        reports.append(report)
        visualizations.append(visualization)
        media_copies.append(media_copy)
    batch = aggregate_reports(reports)
    rendered_reports = [render_dashboard(report, visualization) for report, visualization in zip(reports, visualizations)]
    audit_reports = [render_report(report) for report in reports]
    rendered_batch = render_visual_batch(batch)
    audit_batch = render_batch(batch)
    assets = read_assets()
    if any(report["implementation"] != implementation_manifest() for report in reports):
        raise ValueError("Evaluator or dashboard source changed while exporting; rerun on a stable version")
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    for name, payload in assets.items():
        with (output / "assets" / name).open("xb") as file:
            file.write(payload)
    for report, visualization, media_copy, html, audit in zip(reports, visualizations, media_copies, rendered_reports, audit_reports):
        root = output / Path(report["report_path"]).parent
        root.mkdir()
        with (root / "report.json").open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write("\n")
        with (root / "report.html").open("x", encoding="utf-8") as file:
            file.write(html)
        with (root / "audit.html").open("x", encoding="utf-8") as file:
            file.write(audit)
        with (root / "visualization.json").open("x", encoding="utf-8") as file:
            json.dump(visualization, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            file.write("\n")
        if media_copy:
            source, relative_path, expected_digest = media_copy
            destination = root / relative_path
            destination.parent.mkdir()
            digest = hashlib.sha256()
            with source.open("rb") as original, destination.open("xb") as copied:
                for chunk in iter(lambda: original.read(1024 * 1024), b""):
                    copied.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise ValueError("Video changed while exporting; discard this incomplete output and rerun into a new directory")
    with (output / "batch.json").open("x", encoding="utf-8") as file:
        json.dump(batch, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    with (output / "protocol.json").open("x", encoding="utf-8") as file:
        json.dump(protocol, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    with (output / "index.html").open("x", encoding="utf-8") as file:
        file.write(rendered_batch)
    with (output / "batch_audit.html").open("x", encoding="utf-8") as file:
        file.write(audit_batch)
    with (output / "integration.html").open("x", encoding="utf-8") as file:
        file.write(render_integration())
    return batch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial-dir", action="append", type=Path, default=[],
                        help="Completed trial directory; may be repeated")
    parser.add_argument("--batch-dir", action="append", type=Path, default=[],
                        help="Directory with completed seed_*/summary.json files; may be repeated")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside original trials")
    parser.add_argument("--protocol", type=Path, help="JSON protocol or numeric overrides, validated before use")
    parser.add_argument("--no-video", action="store_true", help="Export charts without copying recorded MP4 media")
    args = parser.parse_args(argv)
    try:
        overrides = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else None
        protocol = make_protocol(overrides)
        trials = discover_trials(args.trial_dir, args.batch_dir)
        batch = export_reports(trials, args.output_dir, protocol, include_video=not args.no_video)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Evaluation stopped: {exc}\n")
    print(json.dumps({"output": str(args.output_dir.resolve()), "trial_count": batch["trial_count"],
                      "by_task": batch["by_task"], "protocol_sha256": batch["protocol_sha256"]},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
