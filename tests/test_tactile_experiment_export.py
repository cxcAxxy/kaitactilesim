import hashlib
import json
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ANALYSIS))

from evaluation_status.tactile_experiment_export import (  # noqa: E402
    GROUPS,
    export_tactile_experiment,
    verify_tactile_experiment,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_h50(root):
    report = root / "reports/h50_checkpoint_experiment"
    report.mkdir(parents=True)
    (report / "index.html").write_text("h50", encoding="utf-8")
    (report / "egotouch.html").write_text("ego", encoding="utf-8")
    rows = []
    for name in ("index.html", "egotouch.html"):
        path = report / name
        rows.append({"path": name, "size": path.stat().st_size, "sha256": sha(path)})
    write_json(report / "artifact_manifest.json", {"schema": "fixture", "files": rows})
    (report.parent / "index.html").write_text("old landing", encoding="utf-8")
    return report


def build_job(root, *, seeds=(0, 1), state="complete"):
    job = root / "jobs/poker_tactile"
    job.mkdir(parents=True)
    protocol = {
        "max_sim_seconds": 90.0,
        "guard_mode": "observe",
        "action_interpolation": "zoh",
        "execute_steps": 5,
        "control_hz": 30,
        "record_fps": 10,
        "staged_evaluation_version": "staged_v1",
        "diagnostic_only": True,
        "formal_metrics_valid": False,
    }
    plan_groups = []
    status_groups = {}
    for label in GROUPS:
        cpm = label == "cpm_best"
        family, horizon, start = ("pi05+cpm", 30, 0) if cpm else ("ftp1", 32, 1)
        step = "best" if cpm else label.removeprefix("ftp1_step")
        checkpoint = root / "checkpoints" / label
        manifest_path = job / "deployments" / label / "deployment_manifest.json"
        evaluation = job / "groups" / label / "attempt_001" / "evaluation"
        plan_groups.append({
            "label": label,
            "task": "poker-draw",
            "model_family": family,
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "prediction_horizon": horizon,
            "action_start_index": start,
            "execute_steps": 5,
            "deployment_manifest": str(manifest_path),
            "evaluation_path": str(evaluation),
        })
        deployment_id = hashlib.sha256(f"deployment-{label}".encode()).hexdigest()
        checkpoint_sha = hashlib.sha256(f"checkpoint-{label}".encode()).hexdigest()
        guidance_manifest = ({"num_steps": 10, "guidance_steps": 3, "scale": 1.0,
                              "variant": "pi05+touchguidance"} if cpm else None)
        manifest = {
            "schema": "poker_tactile_deployment_v1",
            "task": "poker-draw",
            "model_family": family,
            "deployment_id": deployment_id,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": step,
            "prediction_horizon": horizon,
            "action_start_index": start,
            "execute_steps": 5,
            "control_hz": 30,
            "controller_version": "poker-tactile-direct-joints-v1",
            "required_action_interpolation": "zoh",
            "diagnostic_only": True,
            "formal_metrics_valid": False,
            "observation_contract": {"tactile_sent_to_model": True},
        }
        if guidance_manifest is not None:
            manifest["guidance"] = guidance_manifest
        manifest["deployment_id"] = hashlib.sha256(json.dumps(
            {key: value for key, value in manifest.items() if key != "deployment_id"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode()).hexdigest()
        deployment_id = manifest["deployment_id"]
        write_json(manifest_path, manifest)
        manifest_sha = sha(manifest_path)
        trials = []
        for seed in seeds:
            trial_dir = evaluation / f"seed_{seed:04d}"
            guidance = ({"enabled": True, "steps": 3, "scale": 1.0,
                         "variant": "pi05+touchguidance"} if cpm else
                        {"enabled": False, "steps": 0, "scale": 0.0,
                         "variant": "not_applicable"})
            summary = {
                "task": "poker-draw",
                "model_family": family,
                "deployment_id": deployment_id,
                "deployment_manifest_sha256": manifest_sha,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "prediction_horizon": horizon,
                "action_start_index": start,
                "execute_steps": 5,
                "tactile_sent_to_model": True,
                "expert_used": cpm,
                "guidance": guidance,
                "max_sim_seconds": 90.0,
                "guard_mode": "observe",
                "controller": "poker-tactile-direct-joints-v1",
                "action_interpolation": "zoh",
                "success_hold_seconds": 0.1,
                "diagnostic_only": True,
                "formal_metrics_valid": False,
                "seed": seed,
                "status": "success" if seed == 0 else "task_not_completed",
                "task_success": seed == 0,
                "physics_valid": True,
                "sim_seconds": 90.0,
                "stop_reason": "task_success" if seed == 0 else "time_limit",
            }
            write_json(trial_dir / "summary.json", summary)
            (trial_dir / "review.mp4").write_bytes(f"video-{label}-{seed}".encode())
            invalid = label == GROUPS[-1] and seed == seeds[-1]
            trials.append({
                "seed": seed,
                "summary": f"seed_{seed:04d}/summary.json",
                "returncode": 2 if invalid else 0,
                "timed_out": False,
                "valid": not invalid,
                "identity_matches": True,
                "task_success": seed == 0,
                "status": summary["status"],
                "formal_metrics_valid": False,
            })
        batch = {
            "schema": "poker_tactile_batch_v1",
            "task": "poker-draw",
            "model_family": family,
            "prediction_horizon": horizon,
            "action_start_index": start,
            "execute_steps": 5,
            "deployment_id": deployment_id,
            "deployment_manifest_sha256": manifest_sha,
            "complete": True,
            "valid_trials": len(seeds) - (1 if label == GROUPS[-1] else 0),
            "invalid_trials": 1 if label == GROUPS[-1] else 0,
            "diagnostic_only": True,
            "formal_metrics_valid": False,
            "trials": trials,
        }
        write_json(evaluation / "batch_summary.json", batch)
        status_groups[label] = {
            "state": "complete",
            "accepted_attempt": "attempt_001",
            "evaluation_path": str(evaluation),
            "observed_trials": len(seeds),
            "valid_trials": batch["valid_trials"],
            "invalid_trials": batch["invalid_trials"],
            "batch_complete": True,
            "batch_summary": str(evaluation / "batch_summary.json"),
        }
    plan = {
        "schema": "kaihand-poker-tactile-experiment-plan-v1",
        "job_root": str(job),
        "task": "poker-draw",
        "seeds": list(seeds),
        "planned_trials": len(seeds) * len(GROUPS),
        "groups": plan_groups,
        "protocol": protocol,
    }
    write_json(job / "plan.json", plan)
    write_json(job / "status.json", {
        "schema": "kaihand-poker-tactile-experiment-status-v1",
        "state": state,
        "groups": status_groups,
    })
    return job


def test_complete_export_keeps_invalid_out_of_denominators_and_publishes(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path)
    output = h50.parent / "poker_tactile_experiment"
    result = export_tactile_experiment(
        job, h50, output, publish_index=True, require_complete=True
    )
    assert result["status"] == "passed"
    assert result["trials"] == 10
    assert result["valid_trials"] == 9
    assert result["invalid_trials"] == 1
    assert result["missing_trials"] == 0
    assert result["root_portal_matches_this_snapshot"] is True
    catalog = json.loads((output / "catalog.json").read_text())
    assert catalog["experiment"]["counts"] == {
        "planned": 10, "observed": 10, "valid": 9, "invalid": 1, "missing": 0,
    }
    last = catalog["groups"][-1]
    assert last["native"] == {"known": 1, "successes": 1, "success_rate": 1.0}
    assert last["staged"]["known"] == 0
    assert last["staged"]["success_rate"] is None
    assert last["staged"]["evidence_counts"] == {"missing": 2}
    assert last["trials"][-1]["expert_used"] is False
    assert catalog["groups"][0]["trials"][0]["expert_used"] is True
    assert catalog["groups"][0]["identity"]["diagnostic_only"] is True
    assert catalog["groups"][0]["identity"]["formal_metrics_valid"] is False
    assert (output / "trials/cpm_best/seed_0000/review.mp4").is_file()
    assert "Poker 触觉模型对比" in (h50.parent / "index.html").read_text()
    assert verify_tactile_experiment(output)["root_portal_matches_this_snapshot"] is True


def test_stopped_partial_job_keeps_missing_slots_undefined(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0, 1, 2), state="failed")
    label = GROUPS[-1]
    evaluation = job / "groups" / label / "attempt_001" / "evaluation"
    (evaluation / "batch_summary.json").unlink()
    status = json.loads((job / "status.json").read_text())
    status["groups"][label].update(state="failed", accepted_attempt=None,
                                    observed_trials=0, valid_trials=0,
                                    invalid_trials=0, batch_complete=False)
    write_json(job / "status.json", status)
    output = h50.parent / "partial"
    result = export_tactile_experiment(job, h50, output)
    assert result["trials"] == 15
    assert result["missing_trials"] == 3
    catalog = json.loads((output / "catalog.json").read_text())
    group = catalog["groups"][-1]
    assert group["counts"] == {
        "planned": 3, "observed": 0, "valid": 0, "invalid": 0, "missing": 3,
    }
    assert group["native"]["success_rate"] is None
    assert all("unbatched_summary_excluded" in trial["reasons"] for trial in group["trials"])
    with pytest.raises(ValueError, match="Complete publication"):
        export_tactile_experiment(job, h50, h50.parent / "partial_complete", require_complete=True)


def test_active_job_and_path_escape_fail_closed(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, state="complete")
    status = json.loads((job / "status.json").read_text())
    status["state"] = "running"
    write_json(job / "status.json", status)
    with pytest.raises(ValueError, match="still active"):
        export_tactile_experiment(job, h50, h50.parent / "active")
    status["state"] = "complete"
    write_json(job / "status.json", status)
    plan = json.loads((job / "plan.json").read_text())
    plan["groups"][0]["evaluation_path"] = str(tmp_path / "escape")
    write_json(job / "plan.json", plan)
    with pytest.raises(ValueError, match="cpm_best evaluation"):
        export_tactile_experiment(job, h50, h50.parent / "escape")


def test_verify_detects_original_source_tamper(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0,))
    output = h50.parent / "snapshot"
    export_tactile_experiment(job, h50, output)
    summary = job / "groups/cpm_best/attempt_001/evaluation/seed_0000/summary.json"
    summary.write_text(summary.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Archived bytes changed"):
        verify_tactile_experiment(output)


def test_stopped_partial_batch_allows_unfinalized_top_level_flags(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0,), state="failed")
    label = GROUPS[1]
    evaluation = job / "groups" / label / "attempt_001" / "evaluation"
    batch = json.loads((evaluation / "batch_summary.json").read_text())
    batch["complete"] = False
    batch.pop("diagnostic_only")
    batch.pop("formal_metrics_valid")
    write_json(evaluation / "batch_summary.json", batch)
    status = json.loads((job / "status.json").read_text())
    status["groups"][label].update(
        state="failed", accepted_attempt=None, batch_complete=False
    )
    write_json(job / "status.json", status)
    result = export_tactile_experiment(job, h50, h50.parent / "partial_batch")
    assert result["status"] == "passed"
    assert result["valid_trials"] == 4


def test_source_symlink_and_status_identity_fail_closed(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0,))
    summary = job / "groups/cpm_best/attempt_001/evaluation/seed_0000/summary.json"
    outside = tmp_path / "outside-summary.json"
    outside.write_bytes(summary.read_bytes())
    summary.unlink()
    summary.symlink_to(outside)
    with pytest.raises(ValueError, match="must not use symlinks"):
        export_tactile_experiment(job, h50, h50.parent / "symlink")

    job = build_job(tmp_path / "status_case", seeds=(0,), state="failed")
    status = json.loads((job / "status.json").read_text())
    status["groups"][GROUPS[0]]["accepted_attempt"] = "attempt_002"
    write_json(job / "status.json", status)
    with pytest.raises(ValueError, match="Unexpected accepted attempt"):
        export_tactile_experiment(job, h50, h50.parent / "bad_attempt")


def test_failed_staged_bytes_remain_verification_dependencies(tmp_path):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0,))
    trial = job / "groups/cpm_best/attempt_001/evaluation/seed_0000"
    (trial / "staged_evaluation.json").write_text("{", encoding="utf-8")
    state_log = trial / "staged_states.jsonl"
    state_log.write_text("malformed\n", encoding="utf-8")
    output = h50.parent / "failed_staged"
    export_tactile_experiment(job, h50, output)
    catalog = json.loads((output / "catalog.json").read_text())
    record = catalog["groups"][0]["trials"][0]
    assert record["staged_evidence_status"] == "invalid"
    state_log.write_text("replaced\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Archived bytes changed"):
        verify_tactile_experiment(output)


def test_full_staged_rule_identity_must_be_stable_within_group(tmp_path, monkeypatch):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0, 1))
    evaluation = job / "groups/cpm_best/attempt_001/evaluation"
    for seed in (0, 1):
        write_json(evaluation / f"seed_{seed:04d}/staged_evaluation.json", {})

    def fake_load(path, **_kwargs):
        seed = int(Path(path).name.removeprefix("seed_"))
        return {
            "provenance": [],
            "staged_result": {
                "complete_success": False,
                "stage_definitions": {"stage1": "drag"},
                "stage_flags": {"stage1": False},
            },
            "rule_identity": {
                "observer_sha256": "a" * 64,
                "recording_sha256": "b" * 64,
                "definitions_thresholds_sha256": "c" * 64,
                "sha256": ("d" if seed == 0 else "e") * 64,
                "binding_basis": "tactile_manifest_inventory_v1",
            },
        }

    import evaluation_unified.staged as staged

    monkeypatch.setattr(staged, "load_staged_result", fake_load)
    with pytest.raises(ValueError, match="Full strict staged source contract changed"):
        export_tactile_experiment(job, h50, h50.parent / "mixed_rules")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duration", "exactly 90"),
        ("checkpoint", "Wrong checkpoint step"),
        ("guidance", "guidance steps must be 3"),
    ],
)
def test_fixed_experiment_identity_is_exact(tmp_path, mutation, message):
    h50 = frozen_h50(tmp_path)
    job = build_job(tmp_path, seeds=(0,))
    if mutation in {"duration", "checkpoint"}:
        plan = json.loads((job / "plan.json").read_text())
        if mutation == "duration":
            plan["protocol"]["max_sim_seconds"] = 91.0
        else:
            plan["groups"][0]["checkpoint_step"] = "latest"
        write_json(job / "plan.json", plan)
    else:
        path = job / "deployments/cpm_best/deployment_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["guidance"]["guidance_steps"] = 2
        manifest["deployment_id"] = hashlib.sha256(json.dumps(
            {key: value for key, value in manifest.items() if key != "deployment_id"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode()).hexdigest()
        write_json(path, manifest)
    with pytest.raises(ValueError, match=message):
        export_tactile_experiment(job, h50, h50.parent / f"bad_{mutation}")
