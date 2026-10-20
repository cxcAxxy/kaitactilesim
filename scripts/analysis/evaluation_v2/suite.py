"""Export an evidence-aware local evaluation suite without mutating inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from evaluate_trials import _visual_package
from evaluation_v1.adapters import load_trial
from evaluation_v1.core import aggregate_reports, evaluate_trial, _verify_provenance
from evaluation_v1.dashboard import read_assets, render_dashboard, render_integration
from evaluation_v1.protocol import content_hash, make_protocol
from evaluation_v1.render import render_batch, render_report

from . import SCHEMA_VERSION
from .baselines import compare_to_reference
from .contracts import list_contracts
from .evidence import load_evidence, adapt_legacy_evidence
from .quality import assess_contact_quality
from .task_review import review_task
from .validation import coverage_report
from .horizon import filter_prediction_horizon
from .portal import render_portal, read_portal_assets, render_v2_integration, render_suite_report, render_suite_batch


def implementation_manifest() -> dict:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "evaluation_v1").glob("*.py"))
    paths += sorted((root / "evaluation_v1/web").glob("*.js")) + sorted((root / "evaluation_v1/web").glob("*.css"))
    paths += sorted((root / "evaluation_v2").glob("*.py"))
    paths += sorted((root / "evaluation_v2/web").glob("*.js")) + sorted((root / "evaluation_v2/web").glob("*.css"))
    paths += [root / "evaluate_trials.py", root / "evaluate_suite.py", root / "export_expert_reference.py"]
    rows = [{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths if path.is_file()]
    return {"files": rows, "sha256": content_hash(rows), "semantics": "Source and browser assets frozen together; V1 numerical metric definitions retained"}


def _json_write(path: Path, value: object):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def _text_write(path: Path, value: str):
    with path.open("x", encoding="utf-8") as file:
        file.write(value)


def _copy_video(source: Path, target: Path, digest: str):
    target.parent.mkdir()
    actual = hashlib.sha256()
    with source.open("rb") as original, target.open("xb") as copied:
        for chunk in iter(lambda: original.read(1024 * 1024), b""):
            actual.update(chunk)
            copied.write(chunk)
    if actual.hexdigest() != digest:
        raise ValueError("Source video changed while copying; output is incomplete, use a new directory")


def export_suite(trials: list[Path], output: Path, protocol: dict | None = None, *,
                 include_video=True, task_configs=None, reference_reports=None,
                 reference_manifest=None, compatibility=None, configuration_provenance=None,
                 selection=None, required_prediction_horizon=50) -> dict:
    """Calculate first, then create a new output; never overwrite prior reports."""
    protocol = make_protocol(protocol)
    trials = [Path(path).resolve(strict=True) for path in trials]
    output = Path(output).resolve()
    if not trials or len(set(trials)) != len(trials):
        raise ValueError("Supply one or more distinct trial directories")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if any(output == source or output.is_relative_to(source) for source in trials):
        raise ValueError("Output must be outside source trial directories")
    task_configs = {} if task_configs is None else task_configs
    if not isinstance(task_configs, dict):
        raise ValueError("task_configs must be a task-keyed object")
    if bool(reference_reports) != bool(reference_manifest):
        raise ValueError("Reference reports and an explicit reference manifest must be supplied together")
    config_provenance = configuration_provenance or []
    implementation = implementation_manifest()
    trials, horizon_selection = filter_prediction_horizon(trials, required=required_prediction_horizon)
    if not trials:
        raise ValueError(f"No verified H{required_prediction_horizon} trials; all {horizon_selection['input_count']} inputs excluded by prediction-horizon evidence")
    horizon_by_source = {row["source"]: row["inspection"] for row in horizon_selection["included"]}
    configuration = {"schema_version": SCHEMA_VERSION, "task_configs": task_configs,
                     "reference_manifest": reference_manifest, "compatibility": compatibility,
                     "required_prediction_horizon": required_prediction_horizon,
                     "tactile_protocol_sha256": content_hash(protocol)}
    config_hash = content_hash(configuration)
    _verify_provenance(config_provenance)
    _verify_provenance(horizon_selection["provenance"])
    reports, visualizations, media, sidecar_snapshots = [], [], [], []
    for index, source in enumerate(trials):
        report = evaluate_trial(source, protocol)
        primary = {row["path"]: row["sha256"] for row in report["provenance"]}
        visual, video = _visual_package(report, include_video)
        trial = load_trial(source)
        if {row["path"]: row["sha256"] for row in trial.provenance} != primary:
            raise ValueError("Primary evidence changed between metrics and task review")
        horizon_evidence = horizon_by_source[str(source)]
        report["prediction_horizon_evidence"] = horizon_evidence
        report["trial"]["prediction_horizon"] = horizon_evidence["prediction_horizon"]
        report["trial"]["cohort"]["prediction_horizon"] = horizon_evidence["prediction_horizon"]
        report["provenance"].extend(row for row in horizon_evidence["provenance"]
                                    if row["path"] not in {p["path"] for p in report["provenance"]})
        sidecar_paths = [source / "evaluation_evidence.meta.json", source / "evaluation_evidence.jsonl"]
        snapshot = [str(path.resolve()) for path in sidecar_paths if path.exists()]
        bundle = load_evidence(source)
        if bundle.status != "missing" and bundle.metadata.get("task") != trial.task:
            raise ValueError("Evidence task does not match the archived trial task")
        evidence_source = "sidecar"
        if bundle.status == "missing":
            bundle = adapt_legacy_evidence(trial, contact_threshold_n=protocol["board_contact_threshold_n"])
            evidence_source = "legacy" if bundle.status != "missing" else "missing"
        report["suite_schema_version"] = SCHEMA_VERSION
        report["suite_configuration_sha256"] = config_hash
        report["v1_implementation"] = report["implementation"]
        report["implementation"] = implementation
        task_evidence = bundle.task_evidence()
        report["evidence_v2"] = {**bundle.summary, "source": evidence_source,
                                 "sampled_complete": task_evidence.get("complete") is True}
        report["task_review"] = review_task(trial, task_configs.get(trial.task), evidence=task_evidence)
        report["provenance"].extend(row for row in report["task_review"].get("provenance", [])
                                    if row["path"] not in {p["path"] for p in report["provenance"]})
        report["contact_quality"] = assess_contact_quality(bundle)
        report["provenance"].extend(row for row in bundle.provenance if row["path"] not in {p["path"] for p in report["provenance"]})
        if compatibility and report["trial"]["trial_id"] in compatibility:
            report["reference_compatibility"] = compatibility[report["trial"]["trial_id"]]
        report["limitations"].append("V2 task review and contact quality are separate evidence-limited results, not replacements for the recorded outcome. Missing evidence does not establish safety or success.")
        slug = f"{index:03d}_{re.sub(r'[^a-zA-Z0-9_-]', '_', trial.task)}_{source.name}"
        report["report_path"] = f"{slug}/report.html"
        reports.append(report); visualizations.append(visual); media.append(video)
        sidecar_snapshots.append((sidecar_paths, snapshot))
    if reference_reports:
        reference = compare_to_reference(reports, reference_reports, reference_manifest=reference_manifest,
                                         compatibility=compatibility)
    else:
        reference = {"schema_version": "conditional-reference-comparison-v1", "status": "not_configured",
                     "reason": "No explicitly declared compatible expert reference population supplied",
                     "groups": {}, "trials": [], "limitations": ["No expert labels, geometry, sampling compatibility or reference membership were guessed"]}
    reference_trials = {row["trial_id"]: row for row in reference.get("trials", [])}
    for report in reports:
        report["reference_comparison"] = reference_trials.get(report["trial"]["trial_id"],
            {"status": "not_configured", "reasons": [reference.get("reason", "No matching reference comparison")]})
        group_ids = {metric.get("reference_group_id") for metric in report["reference_comparison"].get("metrics", [])}
        report["reference_distributions"] = {key: value for key, value in reference.get("groups", {}).items()
                                             if key in group_ids}
    catalog = list_contracts()
    validation = coverage_report(reports, catalog)
    batch = aggregate_reports(reports)
    batch.update(suite_schema_version=SCHEMA_VERSION, suite_configuration_sha256=config_hash,
                 task_catalog=catalog, validation=validation, reference_comparison=reference,
                 input_selection=selection, horizon_selection=horizon_selection)
    for entry, report in zip(batch["trials"], reports):
        entry.update(seed=report["trial"].get("seed"), cohort=report["trial"].get("cohort"),
                     prediction_horizon=report["trial"]["prediction_horizon"],
                     task_review=report["task_review"], evidence_v2=report["evidence_v2"],
                     contact_quality_status=report["contact_quality"].get("status", "unknown"),
                     reference_status=report["reference_comparison"].get("status", "unknown"))
    pages = [render_dashboard(report, visual) for report, visual in zip(reports, visualizations)]
    audit_pages = [render_suite_report(report) for report in reports]
    portal = render_portal(batch)
    assets = {**read_assets(), **read_portal_assets()}
    for report in reports:
        _verify_provenance(report["provenance"])
    _verify_provenance(config_provenance)
    _verify_provenance(horizon_selection["provenance"])
    for paths, before in sidecar_snapshots:
        if before != [str(path.resolve()) for path in paths if path.exists()]:
            raise ValueError("Optional evidence files appeared or disappeared during export")
    if implementation_manifest() != implementation:
        raise ValueError("Evaluation implementation changed during export; rerun on a stable version")
    output.mkdir(parents=True, exist_ok=False)
    (output / "assets").mkdir()
    for name, data in assets.items():
        with (output / "assets" / name).open("xb") as file:
            file.write(data)
    for report, visual, video, page, audit in zip(reports, visualizations, media, pages, audit_pages):
        directory = output / Path(report["report_path"]).parent
        directory.mkdir()
        _json_write(directory / "report.json", report)
        _json_write(directory / "visualization.json", visual)
        _text_write(directory / "report.html", page)
        _text_write(directory / "audit.html", audit)
        if video:
            _copy_video(video[0], directory / video[1], video[2])
    for name, value in (("batch.json", batch), ("protocol.json", protocol),
                        ("suite_configuration.json", configuration), ("task_catalog.json", catalog),
                        ("validation.json", validation), ("reference_comparison.json", reference),
                        ("horizon_selection.json", horizon_selection),
                        ("configuration_provenance.json", config_provenance)):
        _json_write(output / name, value)
    _text_write(output / "index.html", portal)
    _text_write(output / "batch_audit.html", render_suite_batch(batch))
    _text_write(output / "integration.html", render_v2_integration())
    _text_write(output / "component_integration.html", render_integration())
    return batch
