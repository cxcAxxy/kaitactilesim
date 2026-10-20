from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "scripts/analysis"
if str(ANALYSIS) not in sys.path:
  sys.path.insert(0, str(ANALYSIS))

from evaluation_status import data  # noqa: E402

SHA = "a" * 64


def _json(path: Path, value: object) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  raw = (json.dumps(value, sort_keys=True) + "\n").encode()
  path.write_bytes(raw)
  return hashlib.sha256(raw).hexdigest()


def _video(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"mp4")


def _summary(
  *,
  task="pick-place",
  seed=0,
  native=False,
  legacy=False,
  staged=False,
  checkpoint_sha=SHA,
):
  result = {
    "task": task,
    "seed": seed,
    "status": "task_not_completed",
    "checkpoint_path": f"/checkpoint/{task}",
    "checkpoint_sha256": checkpoint_sha,
    "model_family": "EgoTouch" if legacy else "pi0.5",
    "prediction_horizon": 50,
    "execute_steps": 50,
    "sim_seconds": 90.0,
    "diagnostic_only": False if legacy else True,
    "formal_metrics_valid": True if legacy else False,
    "evaluation": {"success": native},
    "video": {"completed": True, "task_error": None},
  }
  if not legacy:
    result.update(task_success=native, physics_valid=True)
  if staged:
    result["staged_evaluation"] = {"version": "staged_v1"}
  return result


def _empty_index(root: Path, *, groups=(), excluded=()) -> None:
  _json(
    root / "index.json",
    {
      "schema": "kaihand-evaluation-archive-index-v3",
      "archive_id": "archive",
      "groups": list(groups),
      "excluded_completed_unpublished_groups": list(excluded),
    },
  )


def _archive_group(root: Path, *, legacy=True, staged=False) -> dict:
  relative = Path("pick-place/egotouch/group")
  group_root = root / relative
  trial = group_root / "seed_000"
  summary = _summary(legacy=legacy, staged=staged)
  summary_sha = _json(trial / "summary.json", summary)
  _video(trial / "review/review.mp4")
  files = [
    {
      "relative_path": "seed_000/summary.json",
      "size": (trial / "summary.json").stat().st_size,
      "sha256": summary_sha,
    }
  ]
  if staged:
    sidecar_sha = _json(trial / "staged_evaluation.json", {"version": "staged_v1"})
    files.append({
      "relative_path": "seed_000/staged_evaluation.json",
      "size": (trial / "staged_evaluation.json").stat().st_size,
      "sha256": sidecar_sha,
    })
  manifest_sha = _json(group_root / "archive_manifest.json", {
    "schema": "local-h50-archive-manifest-v1" if legacy else "kaihand-staged-evaluation-archive-v1",
    "batch_id": "legacy-batch",
    "files": files,
  })
  aggregate_sha = _json(group_root / "summary.json", {"valid_trials": 1})
  return {
    "task": "pick-place",
    "model": "egotouch" if legacy else "pi05",
    "step": 10,
    "relative_dir": relative.as_posix(),
    "manifest": (relative / "archive_manifest.json").as_posix(),
    "summary": (relative / "summary.json").as_posix(),
    "manifest_sha256": manifest_sha,
    "summary_sha256": aggregate_sha,
    "checkpoint": "/checkpoint/pick-place",
    "checkpoint_sha256": SHA,
    "prediction_horizon": 50,
    "execute_steps": 50,
    "diagnostic_only": not legacy,
    "formal_metrics_valid": legacy,
    "archive_schema": "local-h50-archive-manifest-v1" if legacy else "kaihand-staged-evaluation-archive-v1",
    "run_id": "legacy-batch",
    "seeds": [{
      "seed": 0,
      "summary": (relative / "seed_000/summary.json").as_posix(),
      "video": (relative / "seed_000/review/review.mp4").as_posix(),
      "native_valid_trial": True,
      "native_returncode": 0,
    }],
  }


def _new_job(
  root: Path,
  *,
  name="new-batch",
  accepted_attempt=2,
  sidecar=False,
  summary_override=None,
) -> Path:
  job = root / "_jobs" / name
  group_dir = Path("groups/poker-draw/step10000")
  for attempt, native in ((1, True), (2, False)):
    trial = job / group_dir / f"attempt_{attempt:03d}/run/evaluation/seed_000"
    summary = _summary(task="poker-draw", native=native, staged=sidecar)
    if attempt == accepted_attempt and summary_override:
      summary.update(summary_override)
    _json(trial / "summary.json", summary)
    _video(trial / "review.mp4")
    if sidecar:
      _json(trial / "staged_evaluation.json", {"version": "staged_v1"})
  _json(job / "batch.json", {
    "schema": "kaihand-pi05-stage-batch-v1",
    "settings": {
      "prediction_horizon": 50,
      "execute_steps": 50,
      "num_trials": 1,
    },
    "groups": [{
      "task": "poker-draw",
      "step": 10000,
      "checkpoint": "/checkpoint/poker-draw",
      "group_dir": group_dir.as_posix(),
    }],
  })
  _json(job / "status.json", {
    "schema": "kaihand-pi05-stage-batch-v1",
    "state": "complete",
    "groups": [{
      "state": "complete",
      "attempt": accepted_attempt,
      "error": "earlier attempt failed",
    }],
  })
  return job


def test_legacy_egotouch_uses_explicit_archive_validity_without_faking_physics(tmp_path):
  group = _archive_group(tmp_path)
  _empty_index(
    tmp_path,
    groups=[group],
    excluded=[{"directory": "excluded/group", "reason": "publication rejected"}],
  )

  result = data.build_catalog(tmp_path)

  assert result["schema"] == data.SCHEMA
  assert len(result["tasks"]) == 13
  normalized = result["groups"][0]
  assert normalized["mode"] == "formal"
  assert normalized["valid_trials"] == 1
  assert normalized["native_known_count"] == 1
  trial = normalized["trials"][0]
  assert trial["native_source"] == "evaluation.success"
  assert trial["physics_valid"] is None
  assert trial["validity_basis"] == "legacy_archive_native_valid_declaration"
  assert any("does not claim renewed physics" in warning for warning in trial["warnings"])
  assert any(row["reason"] == "publication rejected" for row in result["excluded"])


def test_large_archive_group_summary_is_allowed_and_index_sha_is_enforced(tmp_path):
  group = _archive_group(tmp_path)
  aggregate = tmp_path / group["summary"]
  group["summary_sha256"] = _json(
    aggregate,
    {"valid_trials": 1, "unused_aggregate_payload": "x" * data.MAX_JSON_BYTES},
  )
  assert aggregate.stat().st_size > data.MAX_JSON_BYTES
  _empty_index(tmp_path, groups=[group])

  result = data.build_catalog(tmp_path)

  assert len(result["groups"]) == 1
  provenance = {
    row["path"]: row for row in result["provenance"]
  }
  assert provenance[str(aggregate)]["sha256"] == group["summary_sha256"]

  group["summary_sha256"] = "b" * 64
  _empty_index(tmp_path, groups=[group])
  mismatch = data.build_catalog(tmp_path)

  assert mismatch["groups"] == []
  assert any(
    "archive group summary SHA-256 differs from index" in row["reason"]
    for row in mismatch["excluded"]
  )


def test_new_job_uses_status_accepted_attempt_not_best_native_result(tmp_path):
  _empty_index(tmp_path)
  job = _new_job(tmp_path)
  norm = tmp_path / "_jobs/ftp1_norm_task"
  norm.mkdir(parents=True)
  (norm / "normalization.log").write_text("running\n", encoding="utf-8")

  result = data.build_catalog(tmp_path)

  assert len(result["groups"]) == 1
  group = result["groups"][0]
  assert group["source_kind"] == "job"
  assert group["valid_trials"] == 1
  assert group["native_success_count"] == 0
  assert group["title"] == "卡片抽取与举起"
  assert group["checkpoint"] == "10000"
  assert group["checkpoint_path"] == "/checkpoint/poker-draw"
  assert "attempt_002" in group["trials"][0]["summary_path"]
  assert "attempt_001" not in group["trials"][0]["summary_path"]
  assert any(row["name"] == norm.name and row["kind"] == "normalization" for row in result["jobs"])
  assert all(str(job / "groups") not in row["source"] for row in result["excluded"])
  roles = {row["role"] for row in result["provenance"]}
  assert "completed_job_status_attempt_selection" in roles
  assert "completed_job_batch_plan" in roles


def test_staged_validation_failure_keeps_native_trial_and_hashes_inputs(
  tmp_path, monkeypatch
):
  _empty_index(tmp_path)
  _new_job(tmp_path, accepted_attempt=2, sidecar=True)
  monkeypatch.setattr(
    data,
    "load_staged_result",
    lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad staged rule")),
  )

  result = data.build_catalog(tmp_path)

  trial = result["groups"][0]["trials"][0]
  assert trial["valid"] is True
  assert trial["staged_result"] is None
  assert any("staged evidence unavailable" in warning for warning in trial["warnings"])
  roles = {row["role"] for row in result["provenance"]}
  assert {"trial_summary", "staged_sidecar"}.issubset(roles)
  assert any("strict staged validation failed" in row["reason"] for row in result["excluded"])


def test_strict_staged_projection_is_aggregated_without_state_rows(tmp_path, monkeypatch):
  group = _archive_group(tmp_path, legacy=False, staged=True)
  _empty_index(tmp_path, groups=[group])

  def loaded(*args, **kwargs):
    assert kwargs == {"include_state_log": False}
    return {
      "staged_result": {
        "stage_definitions": {"stage1": "picked_up"},
        "stage_flags": {"stage1": True},
        "complete_success": False,
      },
      "rule_identity": {"sha256": "b" * 64},
      "provenance": [],
    }

  monkeypatch.setattr(data, "load_staged_result", loaded)
  result = data.build_catalog(tmp_path)

  normalized = result["groups"][0]
  assert normalized["stages"] == [
    {"key": "stage1", "name": "picked_up", "reached": 1, "known": 1}
  ]
  assert normalized["complete_stage_success_count"] == 0
  assert normalized["complete_stage_known_count"] == 1
  assert "state_rows" not in normalized["trials"][0]["staged_result"]


def test_current_trial_missing_physics_boolean_fails_closed(tmp_path):
  _empty_index(tmp_path)
  _new_job(
    tmp_path,
    accepted_attempt=2,
    summary_override={"physics_valid": None},
  )

  result = data.build_catalog(tmp_path)

  group = result["groups"][0]
  assert group["valid_trials"] == 0
  assert group["invalid_trials"] == 1
  assert group["native_known_count"] == 0
  assert any("physics_valid" in warning for warning in group["warnings"])


def test_control_protocol_is_part_of_group_identity(tmp_path):
  _empty_index(tmp_path)
  job = _new_job(tmp_path, accepted_attempt=2)
  batch = json.loads((job / "batch.json").read_text())
  batch["settings"]["num_trials"] = 2
  _json(job / "batch.json", batch)
  evaluation = job / "groups/poker-draw/step10000/attempt_002/run/evaluation"
  first = json.loads((evaluation / "seed_000/summary.json").read_text())
  first.update(max_sim_seconds=90, guard_mode="observe", control_hz=30)
  _json(evaluation / "seed_000/summary.json", first)
  second = {**first, "seed": 1, "max_sim_seconds": 60, "guard_mode": "strict"}
  _json(evaluation / "seed_001/summary.json", second)
  _video(evaluation / "seed_001/review.mp4")

  result = data.build_catalog(tmp_path)

  assert len(result["groups"]) == 2
  assert {group["protocol"]["max_sim_seconds"] for group in result["groups"]} == {
    60.0,
    90.0,
  }
  assert {group["protocol"]["guard_mode"] for group in result["groups"]} == {
    "observe",
    "strict",
  }
  assert len({group["id"] for group in result["groups"]}) == 2


def test_checkpoint_path_and_model_family_mismatch_fail_closed(tmp_path):
  _empty_index(tmp_path)
  _new_job(
    tmp_path,
    accepted_attempt=2,
    summary_override={
      "checkpoint_path": "/checkpoint/wrong-step",
      "model_family": "ftp1",
    },
  )

  result = data.build_catalog(tmp_path)

  group = result["groups"][0]
  assert group["valid_trials"] == 0
  assert any("checkpoint_path" in warning for warning in group["warnings"])
  assert any("model_family" in warning for warning in group["warnings"])


def test_duplicate_summary_and_sidecar_prefers_canonical_source(tmp_path, monkeypatch):
  group = _archive_group(tmp_path, legacy=False, staged=True)
  _empty_index(tmp_path, groups=[group])
  job = _new_job(tmp_path, accepted_attempt=2, sidecar=True)
  canonical = tmp_path / group["relative_dir"] / "seed_000"
  accepted = job / "groups/poker-draw/step10000/attempt_002/run/evaluation/seed_000"
  # Make the job an exact content alias, including task identity.
  accepted.joinpath("summary.json").write_bytes(canonical.joinpath("summary.json").read_bytes())
  accepted.joinpath("staged_evaluation.json").write_bytes(
    canonical.joinpath("staged_evaluation.json").read_bytes()
  )

  monkeypatch.setattr(data, "load_staged_result", lambda *args, **kwargs: {
    "staged_result": {
      "stage_definitions": {"stage1": "picked_up"},
      "stage_flags": {"stage1": False},
      "complete_success": False,
    },
    "rule_identity": {"sha256": "b" * 64},
    "provenance": [],
  })

  result = data.build_catalog(tmp_path)

  assert len(result["groups"]) == 1
  assert result["groups"][0]["source_kind"] == "archive"
  assert any(
    "duplicate summary and staged-sidecar" in row["reason"]
    for row in result["excluded"]
  )
