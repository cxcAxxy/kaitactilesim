"""Regression checks for PickPlace EgoSteer review-optional trials."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace


def test_no_record_trial_keeps_model_outcome(tmp_path, monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/workcell"))
  module = run_path(str(
    Path(__file__).parents[1] / "scripts/workcell/run_pickplace_egosteer_policy.py"
  ))
  runner = module["main"]
  globals_ = runner.__globals__

  async def verify_server(_server, _deployment):
    return None

  async def run_model(_args):
    return globals_["base"].RolloutStats(
      requests=2, action_steps=60, stable_success=True
    )

  monkeypatch.setitem(globals_, "verify_server", verify_server)
  monkeypatch.setattr(globals_["base"], "_parse_args", lambda: SimpleNamespace())
  monkeypatch.setattr(globals_["base"], "_run", run_model)
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "schema": "pickplace_egosteer_deployment_v1",
    "deployment_id": "test",
    "checkpoint_path": "/checkpoint",
    "checkpoint_sha256": "hash",
    "prediction_horizon": 30,
    "action_dim": 48,
    "observation_contract": {"cameras": ["head"]},
    "action_representation": {},
  }))
  output = tmp_path / "trial"
  monkeypatch.setattr(sys, "argv", [
    "run_pickplace_egosteer_policy.py",
    "--deployment-manifest", str(manifest),
    "--server", "ws://example.invalid",
    "--output-dir", str(output),
    "--seed", "0",
    "--execute-steps", "30",
    "--no-record",
  ])
  runner()
  result = json.loads((output / "summary.json").read_text())
  assert result["status"] == "success"
  assert result["evaluation"] == {"success": True}
  assert result["sim_seconds"] == 2.0
  assert result["stats"]["requests"] == 2
  assert not (output / "review").exists()
