# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/workcell"))
import evaluate_poker_tactile_policy_batch as batch_runner


@pytest.mark.parametrize("drift", (None, "task", "formal", "diagnostic"))
def test_shared_batch_routes_native_runner_and_rejects_identity_drift(tmp_path, monkeypatch, drift):
  manifest = dict(task="bulb-screw", model_family="ftp1", prediction_horizon=32,
    action_start_index=1, execute_steps=5, deployment_id="frozen-task-identity")
  path = tmp_path / "deployment.json"
  path.write_text(json.dumps(manifest))
  monkeypatch.setattr(batch_runner, "load_manifest", lambda *args, **kwargs: (path, manifest))
  commands = []

  class Child:
    def __init__(self, command, **kwargs):
      commands.append(command)
      output = Path(command[command.index("--output-dir") + 1])
      output.mkdir()
      report = dict(manifest, valid_trial=True, trial_validity="valid", physics_valid=True,
        task_success=False, formal_metrics_valid=False, diagnostic_only=True, status="task_not_completed",
        deployment_manifest_sha256=batch_runner.sha256_file(path))
      if drift == "task":
        report["task"] = "poker-draw"
      elif drift == "formal":
        report["formal_metrics_valid"] = True
      elif drift == "diagnostic":
        report["diagnostic_only"] = False
      (output / "summary.json").write_text(json.dumps(report))

    def wait(self, **kwargs):
      return 0

  monkeypatch.setattr(batch_runner.subprocess, "Popen", Child)
  output = tmp_path / "batch"
  status = batch_runner.main(["--deployment-manifest", str(path), "--output-dir", str(output),
    "--seeds", "0", "--video-count", "0", "--max-sim-seconds", "0.2"], shared_tasks=True)
  assert status == (1 if drift else 0)
  command = commands[0]
  assert command[1].endswith("run_shared_task_tactile_policy.py")
  assert command[command.index("--task") + 1] == "bulb-screw"
  assert command[command.index("--action-interpolation") + 1] == "zoh"
  summary = json.loads((output / "batch_summary.json").read_text())
  assert summary["schema"] == "task_tactile_batch_v2"
  assert summary["valid_trials"] == (0 if drift else 1)
  assert summary["success_rate"] == (None if drift else 0)


@pytest.mark.parametrize("extra", [["--save-raw"], ["--task", "poker-draw"], ["--seed=4"]])
def test_shared_batch_rejects_unsupported_or_identity_overriding_flags(tmp_path, monkeypatch, extra):
  manifest = dict(task="bulb-screw", model_family="ftp1", execute_steps=5)
  path = tmp_path / "manifest.json"
  monkeypatch.setattr(batch_runner, "load_manifest", lambda *args, **kwargs: (path, manifest))
  output = tmp_path / "batch"
  with pytest.raises(SystemExit):
    batch_runner.main(["--deployment-manifest", str(path), "--output-dir", str(output),
      "--seeds", "0", "--video-count", "0", *extra], shared_tasks=True)
  assert not output.exists()
