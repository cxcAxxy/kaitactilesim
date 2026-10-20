from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "scripts/analysis"
if str(ANALYSIS) not in sys.path:
  sys.path.insert(0, str(ANALYSIS))

from evaluation_status.experiment import TASKS, build_experiment_catalog  # noqa: E402


def _write(path: Path, value: object) -> dict:
  path.parent.mkdir(parents=True, exist_ok=True)
  raw = (json.dumps(value, sort_keys=True) + "\n").encode()
  path.write_bytes(raw)
  return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _leaf(path: Path, raw=b"x") -> dict:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(raw)
  return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(tmp_path: Path):
  status = tmp_path / "status-r2"
  historical = tmp_path / "historical"
  groups = []
  status_artifacts = []
  for task in TASKS:
    for checkpoint, count in (("10000", 5), ("15000", 5), ("20000", 20), ("24999", 20)):
      if task == "whiteboard-wipe-horizontal-line" and checkpoint == "20000":
        continue
      trials = []
      for seed in range(count):
        summary_sha = hashlib.sha256(f"eligible:{task}:{checkpoint}:{seed}".encode()).hexdigest()
        trial_id = f"trial-{summary_sha[:24]}"
        detail = f"trials/{trial_id}/index.html"
        artifact = _leaf(status / detail)
        status_artifacts.append({"path": detail, "size": artifact["size"], "sha256": artifact["sha256"]})
        trials.append({
          "id": trial_id,
          "seed": seed,
          "status": "task_not_completed",
          "valid": True,
          "native_success": False,
          "summary_sha256": summary_sha,
          "detail_url": detail,
          "staged_result": None,
        })
      groups.append({
        "id": f"group-{task}-{checkpoint}",
        "task": task,
        "title": task,
        "model": "pi05",
        "checkpoint": checkpoint,
        "checkpoint_path": f"/checkpoints/{task}/one-training-run/{checkpoint}",
        "checkpoint_sha256": hashlib.sha256(f"checkpoint:{task}:{checkpoint}".encode()).hexdigest(),
        "prediction_horizon": 50,
        "execute_steps": 50,
        "batch_id": "source",
        "source_kind": "archive",
        "mode": "diagnostic",
        "protocol": {"completion_protocol": f"protocol-{checkpoint}"},
        "planned_trials": count,
        "valid_trials": count,
        "invalid_trials": 0,
        "native_success_count": 0,
        "native_known_count": count,
        "stages": [],
        "complete_stage_success_count": 0,
        "complete_stage_known_count": 0,
        "warnings": [],
        "trials": trials,
      })
  catalog = {
    "schema": "kaihand-evaluation-status-v1",
    "tasks": [{"task": task, "title": task, "note": "note"} for task in TASKS]
    + [{"task": "egotouch-only", "title": "EgoTouch", "note": "excluded"}],
    "groups": groups + [{"model": "egotouch", "task": "egotouch-only", "trials": [{}]}],
  }
  catalog_id = _write(status / "catalog.json", catalog)
  verification_id = _write(status / "verification.json", {
    "schema": "kaihand-evaluation-status-snapshot-v1",
    "status": "passed",
  })
  status_artifacts += [
    {"path": "catalog.json", "size": catalog_id["size"], "sha256": catalog_id["sha256"]},
    {
      "path": "verification.json",
      "size": verification_id["size"],
      "sha256": verification_id["sha256"],
    },
  ]
  _write(status / "artifact_manifest.json", {
    "schema": "kaihand-evaluation-status-snapshot-v1",
    "files": status_artifacts,
  })

  batch_trials = []
  source_trials = []
  historical_artifacts = []
  checkpoint_sha = hashlib.sha256(b"quarantined-checkpoint").hexdigest()
  for seed in range(20):
    trial_id = f"quarantined:{seed}"
    report = f"case_{seed:03d}/report.html"
    video = f"case_{seed:03d}/media/rollout.mp4"
    for relative in (report, video):
      identity = _leaf(historical / relative, relative.encode())
      historical_artifacts.append({
        "path": f"/retired/reports/{historical.name}/{relative}",
        "size": identity["size"],
        "sha256": identity["sha256"],
      })
    summary_sha = hashlib.sha256(f"quarantined-summary:{seed}".encode()).hexdigest()
    source_path = f"/source/whiteboard-line/step20000/seed_{seed:03d}"
    batch_trials.append({
      "trial_id": trial_id,
      "task": "whiteboard-wipe-horizontal-line",
      "title": "水平白板线迹擦拭",
      "seed": seed,
      "model": "pi0.5",
      "checkpoint": "20000",
      "prediction_horizon": 50,
      "mode": "diagnostic",
      "outcome": "fail",
      "report_path": report,
      "source": source_path,
      "cohort": {
        "checkpoint": checkpoint_sha,
        "model_family": "pi0.5",
        "controller": "controller",
        "control_hz": 30,
        "execute_steps": 50,
        "guard_mode": "enforce",
        "diagnostic_only": True,
        "formal_metrics_valid": False,
      },
      "staged": {
        "rule_identity": {"sha256": "b" * 64},
        "native_result": {
          "task_success": seed == 0,
          "status": "success" if seed == 0 else "task_not_completed",
          "physics_valid": True,
        },
        "source_selection": {"group_status": "failed"},
      },
    })
    source_trials.append({
      "trial_id": trial_id,
      "source": source_path,
      "provenance": [{
        "path": f"{source_path}/summary.json",
        "size": 100 + seed,
        "sha256": summary_sha,
        "role": "prediction_horizon_summary",
      }],
    })
  batch_id = _write(historical / "batch.json", {
    "schema_version": "evaluation-workbench-suite-v1",
    "trials": batch_trials,
  })
  # Reverse order so successful construction cannot depend on seed/order zipping.
  sources_id = _write(historical / "source_manifest.json", {
    "trials": list(reversed(source_trials)),
  })
  for name, identity in (("batch.json", batch_id), ("source_manifest.json", sources_id)):
    historical_artifacts.append({
      "path": f"/retired/reports/{historical.name}/{name}",
      "size": identity["size"],
      "sha256": identity["sha256"],
    })
  _write(historical / "artifact_manifest.json", {
    "schema_version": "historical",
    "files": historical_artifacts,
  })
  return status, historical


def _rewrite_status_manifest(status: Path) -> None:
  manifest = json.loads((status / "artifact_manifest.json").read_text())
  catalog_id = _write(status / "catalog.json", json.loads((status / "catalog.json").read_text()))
  row = next(row for row in manifest["files"] if row["path"] == "catalog.json")
  row.update(size=catalog_id["size"], sha256=catalog_id["sha256"])
  _write(status / "artifact_manifest.json", manifest)


def _rewrite_historical_metadata(historical: Path, name: str, value: dict) -> None:
  identity = _write(historical / name, value)
  manifest = json.loads((historical / "artifact_manifest.json").read_text())
  row = next(row for row in manifest["files"] if row["path"].endswith(f"/{name}"))
  row.update(size=identity["size"], sha256=identity["sha256"])
  _write(historical / "artifact_manifest.json", manifest)


def test_builds_450_member_experiment_and_quarantines_failed_group(tmp_path):
  status, historical = _fixture(tmp_path)

  result = build_experiment_catalog(status, historical)

  assert result["schema"] == "kaihand-evaluation-status-v1"
  assert len(result["tasks"]) == 9
  assert len(result["groups"]) == 36
  assert result["experiment"]["member_trials"] == 450
  assert result["experiment"]["eligible_trials"] == 430
  assert result["experiment"]["quarantined_trials"] == 20
  assert result["experiment"]["score_denominator"] == 430
  assert {group["model"] for group in result["groups"]} == {"pi05"}

  eligible = next(
    group for group in result["groups"]
    if group["task"] == "poker-draw" and group["checkpoint"] == "10000"
  )
  assert eligible["valid_trials"] == 5
  assert eligible["protocol"] == {"completion_protocol": "protocol-10000"}
  assert eligible["trials"][0]["detail_url"].startswith(f"../{status.name}/trials/")

  quarantine = next(
    group for group in result["groups"]
    if group["task"] == "whiteboard-wipe-horizontal-line"
    and group["checkpoint"] == "20000"
  )
  assert quarantine["publication_status"] == "quarantined_replay_only"
  assert quarantine["valid_trials"] == 0
  assert quarantine["invalid_trials"] == 20
  assert quarantine["native_known_count"] == 0
  assert quarantine["native_success_count"] == 0
  assert quarantine["recorded_native_known_count"] == 20
  assert quarantine["recorded_native_success_count"] == 1
  assert quarantine["stages"] == []
  assert quarantine["complete_stage_known_count"] == 0
  assert all(trial["valid"] is False for trial in quarantine["trials"])
  assert all(trial["native_success"] is None for trial in quarantine["trials"])
  assert quarantine["trials"][0]["recorded_native_success"] is True
  assert quarantine["trials"][0]["historical_detail_url"].startswith(
    f"../{historical.name}/case_000/"
  )
  assert len(result["dependencies"]) == 470
  assert len(result["provenance"]) == 6
  lineage = next(
    row for row in result["experiment"]["training_lineage"]
    if row["task"] == "whiteboard-wipe-horizontal-line"
  )
  assert lineage["all_checkpoint_paths_independently_recorded"] is False
  assert lineage["checkpoint_paths"]["20000"].endswith("/one-training-run/20000")


def test_quarantine_join_is_trial_id_and_summary_sha_bound(tmp_path):
  status, historical = _fixture(tmp_path)
  sources = json.loads((historical / "source_manifest.json").read_text())
  sources["trials"][0]["trial_id"] = "not-in-batch"
  _rewrite_historical_metadata(historical, "source_manifest.json", sources)

  with pytest.raises(ValueError, match="no trial-id match"):
    build_experiment_catalog(status, historical)


def test_rejects_checkpoint_path_from_a_different_training_run(tmp_path):
  status, historical = _fixture(tmp_path)
  catalog = json.loads((status / "catalog.json").read_text())
  group = next(
    row for row in catalog["groups"]
    if row.get("task") == "poker-draw" and row.get("checkpoint") == "15000"
  )
  group["checkpoint_path"] = "/checkpoints/poker-draw/different-training-run/15000"
  _write(status / "catalog.json", catalog)
  _rewrite_status_manifest(status)

  with pytest.raises(ValueError, match="one training-run parent"):
    build_experiment_catalog(status, historical)


def test_rejects_metadata_bytes_not_bound_by_parent_manifest(tmp_path):
  status, historical = _fixture(tmp_path)
  catalog = json.loads((status / "catalog.json").read_text())
  catalog["groups"][0]["protocol"] = {"completion_protocol": "changed-without-manifest"}
  _write(status / "catalog.json", catalog)

  with pytest.raises(ValueError, match="differs from its snapshot artifact manifest"):
    build_experiment_catalog(status, historical)


def test_never_merges_distinct_protocols_for_one_checkpoint(tmp_path):
  import copy

  status, historical = _fixture(tmp_path)
  catalog = json.loads((status / "catalog.json").read_text())
  extra = copy.deepcopy(next(row for row in catalog["groups"] if row.get("task") == "poker-draw"))
  extra["id"] = "separate-protocol-not-poolable"
  extra["protocol"] = {"completion_protocol": "different-protocol"}
  catalog["groups"].append(extra)
  _write(status / "catalog.json", catalog)
  _rewrite_status_manifest(status)

  with pytest.raises(ValueError, match="duplicate|protocol|cohort"):
    build_experiment_catalog(status, historical)
