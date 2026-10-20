from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/tactile_evaluation_summary.py"


def load_module():
  spec = importlib.util.spec_from_file_location("tactile_evaluation_summary", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def trial(**changes):
  value = {
    "task": "poker-draw",
    "model_family": "pi05+touchguide",
    "prediction_horizon": 30,
    "action_start_index": 0,
    "execute_steps": 5,
    "seed": 0,
    "status": "success",
    "native_success": True,
    "formal_metrics_valid": False,
    "diagnostic_only": True,
    "tactile_sent_to_model": True,
    "deployment_manifest_sha256": "a" * 64,
    "guidance": {
      "enabled": True,
      "steps": 3,
      "scale": 1.0,
      "variant": "recovery-state-cpm",
    },
  }
  value.update(changes)
  return value


def test_horizon_manifest_k_and_guidance_are_never_mixed():
  module = load_module()
  records = [
    trial(seed=0),
    trial(seed=1, prediction_horizon=32),
    trial(seed=2, prediction_horizon=50),
    trial(seed=3, execute_steps=3),
    trial(seed=4, deployment_manifest_sha256="b" * 64),
    trial(seed=5, guidance={"enabled": True, "steps": 2, "scale": 1.0}),
    trial(seed=6, guidance={"enabled": False}),
  ]
  report = module.summarize_trials(records)
  assert len(report["groups"]) == 7
  assert {group["identity"]["prediction_horizon"] for group in report["groups"]} == {
    30,
    32,
    50,
  }
  assert report["grouped_trials"] == 7
  assert report["ungrouped_trials"] == 0


def test_explicit_evaluation_protocols_are_never_mixed():
  module = load_module()
  base_protocol = {
    "controller": "poker-tactile-direct-joints-v1",
    "action_interpolation": "zoh",
    "success_hold_seconds": 0.25,
  }
  records = [
    trial(seed=0, max_sim_seconds=2.0, guard_mode="observe", **base_protocol),
    trial(seed=1, max_sim_seconds=12.0, guard_mode="observe", **base_protocol),
    trial(seed=2, max_sim_seconds=2.0, guard_mode="strict", **base_protocol),
    trial(seed=3),
  ]
  report = module.summarize_trials(records)
  assert len(report["groups"]) == 4
  protocols = [group["identity"]["protocol"] for group in report["groups"]]
  assert {
    (protocol["max_sim_seconds"], protocol["guard_mode"])
    for protocol in protocols
  } == {(2.0, "observe"), (12.0, "observe"), (2.0, "strict"), (None, None)}
  assert next(
    protocol for protocol in protocols if protocol["max_sim_seconds"] == 2.0
    and protocol["guard_mode"] == "observe"
  ) == {
    "max_sim_seconds": 2.0,
    "guard_mode": "observe",
    **base_protocol,
  }


@pytest.mark.parametrize(
  "changes",
  (
    {"max_sim_seconds": 0},
    {"max_sim_seconds": float("inf")},
    {"success_hold_seconds": True},
    {"guard_mode": "sometimes"},
    {"controller": ""},
    {"action_interpolation": 3},
  ),
)
def test_invalid_optional_protocol_metadata_is_ungrouped(changes):
  module = load_module()
  report = module.summarize_trials([trial(**changes)])
  assert report["groups"] == []
  assert report["ungrouped"][0]["evidence_status"] == "invalid"
  assert any(reason.startswith("invalid_") for reason in report["ungrouped"][0]["reasons"])


def test_success_rate_excludes_invalid_and_missing_and_splits_modes():
  module = load_module()
  records = [
    trial(seed=0, formal_metrics_valid=True, diagnostic_only=False),
    trial(
      seed=1,
      status="task_not_completed",
      native_success=False,
      formal_metrics_valid=True,
      diagnostic_only=False,
    ),
    trial(
      seed=2,
      status="error",
      native_success=False,
      formal_metrics_valid=True,
      diagnostic_only=False,
    ),
    trial(seed=3, status="missing", native_success=None),
    trial(seed=4),
  ]
  group = module.summarize_trials(records)["groups"][0]
  assert group["overall"] == {
    "total": 5,
    "valid": 3,
    "invalid": 1,
    "missing": 1,
    "successes": 2,
    "failures": 1,
    "success_rate": pytest.approx(2 / 3),
  }
  assert group["formal"]["total"] == 3
  assert group["formal"]["valid"] == 2
  assert group["formal"]["success_rate"] == pytest.approx(0.5)
  assert group["diagnostic"]["total"] == 2
  assert group["diagnostic"]["valid"] == 1
  assert group["diagnostic"]["success_rate"] == 1.0


def test_missing_and_invalid_identity_stay_ungrouped():
  module = load_module()
  missing = trial()
  missing.pop("prediction_horizon")
  invalid = trial(prediction_horizon="50")
  impossible = trial(action_start_index=28, execute_steps=5)
  report = module.summarize_trials([missing, invalid, impossible])
  assert report["groups"] == []
  assert [row["evidence_status"] for row in report["ungrouped"]] == [
    "missing",
    "invalid",
    "invalid",
  ]
  assert "missing_prediction_horizon" in report["ungrouped"][0]["reasons"]
  assert "invalid_prediction_horizon" in report["ungrouped"][1]["reasons"]
  assert "action_window_exceeds_prediction_horizon" in report["ungrouped"][2]["reasons"]


def test_outcome_evidence_can_be_explicit_and_does_not_fill_missing_success():
  module = load_module()
  explicit_invalid = trial(
    seed=1,
    trial_validity="invalid",
    status="success",
    native_success=True,
  )
  missing_success = trial(seed=2, native_success=None)
  group = module.summarize_trials([explicit_invalid, missing_success])["groups"][0]
  assert group["overall"]["valid"] == 0
  assert group["overall"]["invalid"] == 1
  assert group["overall"]["missing"] == 1
  assert group["overall"]["success_rate"] is None
  assert "missing_native_success" in group["trials"][1]["reasons"]


def test_status_is_required_even_with_explicit_validity():
  module = load_module()
  record = trial(trial_validity="valid")
  record.pop("status")
  group = module.summarize_trials([record])["groups"][0]
  assert group["overall"]["valid"] == 0
  assert group["overall"]["missing"] == 1
  assert "missing_status" in group["trials"][0]["reasons"]


def test_native_runner_outcome_aliases_and_physics_are_audited():
  module = load_module()
  alias = trial(
    seed=1,
    native_success=None,
    task_success=False,
    status="task_not_completed",
    physics_valid=True,
    stop_reason="time_limit",
  )
  alias.pop("native_success")
  physics_invalid = trial(
    seed=2,
    task_success=True,
    physics_valid=False,
    stop_reason="fatal_error",
  )
  conflict = trial(seed=3, task_success=False, physics_valid=True)
  group = module.summarize_trials([alias, physics_invalid, conflict])["groups"][0]
  assert group["overall"] == {
    "total": 3,
    "valid": 1,
    "invalid": 2,
    "missing": 0,
    "successes": 0,
    "failures": 1,
    "success_rate": 0.0,
  }
  assert "physics_invalid_trial_cannot_be_valid" in group["trials"][1]["reasons"]
  assert "conflicting_native_and_task_success" in group["trials"][2]["reasons"]


def test_guidance_must_be_explicit_and_consistent():
  module = load_module()
  absent = trial()
  absent.pop("guidance")
  contradictory = trial(guidance={"enabled": False, "steps": 3, "scale": 0})
  flat = trial(
    guidance=None,
    guidance_enabled=True,
    guidance_steps=4,
    guidance_scale=0.5,
    guidance_variant="oa",
  )
  report = module.summarize_trials([absent, contradictory, flat])
  assert len(report["groups"]) == 1
  assert report["groups"][0]["identity"]["guidance"] == {
    "enabled": True,
    "steps": 4,
    "scale": 0.5,
    "variant": "oa",
  }
  assert report["ungrouped"][0]["evidence_status"] == "missing"
  assert report["ungrouped"][1]["evidence_status"] == "invalid"
  assert "unguided_trial_has_guidance_steps" in report["ungrouped"][1]["reasons"]


def test_html_is_simple_escaped_and_labels_modes_and_horizon():
  module = load_module()
  report = module.summarize_trials(
    [
      trial(
        model_family="<touch&guide>", formal_metrics_valid=True,
        diagnostic_only=False, max_sim_seconds=2.0, guard_mode="observe",
        controller="direct", action_interpolation="zoh",
        success_hold_seconds=0.25,
      ),
    ]
  )
  rendered = module.render_html(report)
  assert "&lt;touch&amp;guide&gt;" in rendered
  assert "H30" in rendered
  assert "Formal and diagnostic evidence are separate" in rendered
  assert "<th>Success rate</th>" in rendered
  assert "<th>Protocol</th>" in rendered
  assert "max=2s; guard=observe" in rendered
  assert "<touch&guide>" not in rendered


def test_loader_and_cli_write_new_json_and_html(tmp_path, capsys):
  module = load_module()
  source = tmp_path / "inputs"
  source.mkdir()
  (source / "summary.json").write_text(json.dumps({"trials": [trial()]}))
  output_json = tmp_path / "out" / "summary.json"
  output_html = tmp_path / "out" / "summary.html"
  assert (
    module.main(
      [
        str(source),
        "--output-json",
        str(output_json),
        "--output-html",
        str(output_html),
      ]
    )
    == 0
  )
  payload = json.loads(output_json.read_text())
  assert payload["schema_version"] == module.SCHEMA_VERSION
  assert payload["groups"][0]["diagnostic"]["success_rate"] == 1.0
  assert "Tactile evaluation summary" in output_html.read_text()
  assert json.loads(capsys.readouterr().out)["groups"] == 1
  with pytest.raises(FileExistsError, match="refusing to overwrite"):
    module.main(
      [
        str(source),
        "--output-json",
        str(output_json),
        "--output-html",
        str(tmp_path / "other.html"),
      ]
    )

  records, sources = module.load_trial_summaries([source, source / "summary.json"])
  assert len(records) == len(sources) == 1


def test_batch_refs_gate_validity_without_upgrading_child_evidence(
  tmp_path,
):
  module = load_module()
  first = tmp_path / "seed_0000" / "summary.json"
  second = tmp_path / "seed_0001" / "summary.json"
  first.parent.mkdir()
  second.parent.mkdir()
  first.write_text(json.dumps(trial(seed=0)), encoding="utf-8")
  second.write_text(json.dumps(trial(
    seed=1,
    status="task_not_completed",
    native_success=False,
    trial_validity="invalid",
    valid_trial=False,
  )), encoding="utf-8")
  batch = {
    "schema": "poker_tactile_batch_v1",
    "complete": False,
    "trials": [
      {
        "seed": 0,
        "summary": "seed_0000/summary.json",
        "valid": True,
        "timed_out": False,
        "returncode": 9,
      },
      {
        "seed": 1,
        "summary": "seed_0001/summary.json",
        "valid": True,
        "timed_out": False,
        "returncode": 0,
      },
      {
        "seed": 2,
        "summary": "seed_0002/summary.json",
        "valid": False,
        "timed_out": True,
        "returncode": -15,
      },
    ],
  }
  (tmp_path / "batch_summary.json").write_text(
    json.dumps(batch), encoding="utf-8"
  )

  records, sources = module.load_trial_summaries([tmp_path])
  assert len(records) == len(sources) == 3
  report = module.summarize_trials(records, sources=sources)
  group = report["groups"][0]
  assert group["identity"] == {
    key: value
    for key, value in module._identity(trial())[0].items()
  }
  assert group["overall"] == {
    "total": 2,
    "valid": 0,
    "invalid": 2,
    "missing": 0,
    "successes": 0,
    "failures": 0,
    "success_rate": None,
  }
  assert "batch_nonzero_returncode" in group["trials"][0]["reasons"]
  assert "batch_downgraded_child_validity" in group["trials"][0]["reasons"]
  assert "batch_child_validity_conflict" in group["trials"][1]["reasons"]
  assert group["trials"][1]["evidence_status"] == "invalid"
  assert "seed_0000/summary.json" in group["trials"][0]["source"]
  assert report["ungrouped"] == [{
    "source": sources[2],
    "evidence_status": "missing",
    "reasons": [
      "missing_task",
      "missing_model_family",
      "missing_deployment_manifest_sha256",
      "missing_prediction_horizon",
      "missing_action_start_index",
      "missing_execute_steps",
      "missing_tactile_sent_to_model",
      "missing_guidance",
      "missing_formal_metrics_valid",
      "missing_referenced_summary",
      "missing_seed",
      "missing_status",
    ],
  }]


def test_directory_scan_uses_parent_batch_instead_of_counting_child_directly(
  tmp_path,
):
  module = load_module()
  child = tmp_path / "seed_0000" / "summary.json"
  child.parent.mkdir()
  child.write_text(json.dumps(trial()), encoding="utf-8")
  (tmp_path / "batch_summary.json").write_text(json.dumps({
    "schema": "poker_tactile_batch_v1",
    "trials": [{
      "summary": "seed_0000/summary.json",
      "valid": False,
      "timed_out": True,
      "returncode": -15,
    }],
  }), encoding="utf-8")

  records, sources = module.load_trial_summaries([tmp_path])
  assert len(records) == len(sources) == 1
  group = module.summarize_trials(records, sources=sources)["groups"][0]
  assert group["overall"]["invalid"] == 1
  assert group["overall"]["valid"] == 0
  assert "batch_timed_out" in group["trials"][0]["reasons"]


def test_nonfinite_json_is_rejected(tmp_path):
  module = load_module()
  source = tmp_path / "summary.json"
  source.write_text('{"guidance_scale": NaN}')
  with pytest.raises(ValueError, match="non-finite JSON constant"):
    module.load_trial_summaries([source])
