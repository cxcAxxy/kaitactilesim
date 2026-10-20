"""Whiteboard reference options survive batch dispatch without changing outcomes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKCELL = ROOT / "scripts/workcell"


def _load(monkeypatch, filename):
  monkeypatch.syspath_prepend(str(WORKCELL))
  spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), WORKCELL / filename)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.mark.parametrize(
  ("filename", "model_args"),
  [
    ("run_shared_task_pi05_policy.py", ["--server", "ws://localhost:10000"]),
    ("run_shared_task_egosteer_policy.py", ["--server", "ws://localhost:10000"]),
    ("run_shared_task_egotouch_policy.py", ["--model-python", "python", "--model-project", "/missing/model"]),
  ],
)
@pytest.mark.parametrize("task", ["whiteboard-wipe", "whiteboard-wipe-horizontal"])
def test_reference_options_do_not_read_missing_dataset_at_parse(
  monkeypatch, filename, model_args, task,
):
  runner = _load(monkeypatch, filename)
  common = [
    "--deployment-manifest", "/missing/deployment.json",
    "--output-dir", "/missing/output", "--seed", "1", "--no-record",
    "--contact-reference-hdf5", "/missing/expert.h5",
    "--contact-demo-root", "/missing/demos", *model_args,
  ]

  args = runner.parse_args(["--task", task, *common])

  assert args.contact_reference_hdf5 == Path("/missing/expert.h5")
  assert args.contact_demo_root == Path("/missing/demos")
  assert args.record is False
  assert args.guard_mode == "observe"
  with pytest.raises(SystemExit):
    runner.parse_args(["--task", "bulb-screw", *common])


@pytest.mark.parametrize("task", ["whiteboard-wipe", "whiteboard-wipe-horizontal"])
def test_batch_forwards_contact_reference_and_keeps_diagnostic_errors_separate(
  monkeypatch, tmp_path, task,
):
  batch = _load(monkeypatch, "evaluate_shared_task_policy_batch.py")
  manifest = {
    "task": task, "model_family": "egosteer", "deployment_id": "fixture",
    "checkpoint_path": "/missing/checkpoint", "checkpoint_sha256": "abc",
    "prediction_horizon": 10, "action_dim": 78,
    "observation_contract": {"cameras": ["head"]},
    "action_representation": "fixture",
  }
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  output = tmp_path / "output"
  expert_path = tmp_path / "missing_expert.h5"
  demo_root = tmp_path / "missing_demos"
  contact_diagnostic = {"status": "error", "error": "reference unavailable"}
  commands = []
  monkeypatch.setattr(batch, "verify_evaluation_sources", lambda *_: {"status": "not_available"})
  monkeypatch.setattr(batch, "_enforce_source_integrity", lambda *_: False)
  monkeypatch.setattr(batch, "_environment", lambda *_: {})

  def fake_trial(command, **_kwargs):
    commands.append(command)
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    report = {
      **manifest,
      "execute_steps": 5, "controller": None, "action_interpolation": None,
      "control_hz": 30, "guard_mode": "observe", "diagnostic_only": True,
      "formal_metrics_valid": False, "diagnostic_reasons": ["guard_mode_observe"],
      "status": "success", "task_success": True, "physics_valid": True,
      "stop_reason": "success", "contact_reference_hdf5": str(expert_path),
      "contact_demo_root": str(demo_root), "contact_intensity": contact_diagnostic,
    }
    (trial / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(batch.subprocess, "run", fake_trial)
  batch.main([
    "--deployment-manifest", str(manifest_path), "--output-dir", str(output),
    "--seeds", "1", "--video-count", "0", "--server", "ws://localhost:10000",
    "--contact-reference-hdf5", str(expert_path), "--contact-demo-root", str(demo_root),
  ])

  assert "--no-record" in commands[0]
  assert commands[0][commands[0].index("--contact-reference-hdf5") + 1] == str(expert_path)
  assert commands[0][commands[0].index("--contact-demo-root") + 1] == str(demo_root)
  protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
  summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  assert protocol["contact_reference_hdf5"] == str(expert_path)
  assert summary["contact_demo_root"] == str(demo_root)
  row = summary["results"][0]
  assert row["identity_errors"] == {}
  assert row["contact_intensity"] == contact_diagnostic
  assert row["termination"] == "success"
  assert row["valid_trial"] is True
  assert row["record_error"] is None
