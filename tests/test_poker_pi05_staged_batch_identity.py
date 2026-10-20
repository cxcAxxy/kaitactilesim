"""Batch/rollout duration contracts without GPU allocation or model requests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
  workcell = ROOT / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  try:
    path = workcell / name
    spec = importlib.util.spec_from_file_location(f"duration_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(workcell))


@pytest.fixture
def batch_run(tmp_path, monkeypatch):
  batch = load_script("evaluate_poker_pi05_policy_batch.py")
  runner = load_script("run_poker_pi05_policy.py")
  # Legacy manifests keep this control-flow test independent of checkpoint
  # assets. Manifest byte checks and every per-seed identity check stay real.
  manifest = {
    "schema": batch.POKER_DEPLOYMENT_SCHEMA_V1,
    "task": "poker-draw",
    "model_family": "pi0.5",
    "deployment_id": "d" * 64,
    "checkpoint_path": "/checkpoint/20000",
    "checkpoint_sha256": "c" * 64,
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"policy_output_is_absolute": True},
  }
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text(json.dumps(manifest))
  output = tmp_path / "evaluation"
  commands = []
  summary_overrides = {}

  def rollout(command, **kwargs):
    commands.append(command)
    # Parse the exact real child command, including full-duration guard
    # normalization, instead of constructing a summary from batch expectations.
    args = runner.parse_args(command[2:])
    args.output_dir.mkdir()
    summary = {
      "task": "poker-draw",
      "deployment_id": manifest["deployment_id"],
      "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": manifest["model_family"],
      "prediction_horizon": manifest["prediction_horizon"],
      "model_action_dim": manifest["model_action_dim"],
      "action_dim": manifest["action_dim"],
      "execute_steps": args.execute_steps,
      "success_hold_seconds": args.success_hold_seconds,
      "full_duration_evaluation": bool(
        args.staged_evaluation_version or args.full_duration_evaluation
      ),
      "control_hz": runner.CONTROL_HZ,
      "guard_mode": args.guard_mode,
      "diagnostic_only": True,
      "action_interpolation": args.action_interpolation,
      "penetration_guard_enabled": not args.disable_penetration_guard,
      "status": "task_not_completed",
      "sim_seconds": args.max_sim_seconds,
      "stop_reason": "time_limit",
      **summary_overrides,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary))
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(batch.subprocess, "run", rollout)

  def run(extra_args=()):
    monkeypatch.setattr(sys, "argv", [
      str(Path(batch.__file__)),
      "--output-dir", str(output),
      "--deployment-manifest", str(manifest_path),
      "--seeds", "0", "7",
      "--execute-steps", "50",
      "--action-interpolation", "linear",
      "--guard-mode", "strict",
      "--video-count", "0",
      *extra_args,
    ])
    batch.main()
    return (
      json.loads((output / "protocol.json").read_text()),
      json.loads((output / "summary.json").read_text()),
    )

  return run, commands, summary_overrides, output


@pytest.mark.parametrize("staged,explicit_full", [
  (False, False),
  (False, True),
  (True, False),
  (True, True),
])
def test_batch_effective_duration_matches_every_child_identity(
  batch_run, staged, explicit_full,
):
  run, commands, _, output = batch_run
  arguments = []
  if staged:
    arguments.extend(("--staged-evaluation-version", "staged_v1"))
  if explicit_full:
    arguments.append("--full-duration-evaluation")
  effective_full = staged or explicit_full
  protocol, aggregate = run(arguments)

  assert aggregate["complete"] is True
  assert aggregate["valid_trials"] == aggregate["attempted"] == 2
  assert [row["seed"] for row in aggregate["trials"]] == [0, 7]
  assert len(commands) == 2
  for command, seed in zip(commands, [0, 7]):
    assert ("--full-duration-evaluation" in command) is effective_full
    assert ("--disable-penetration-guard" in command) is effective_full
    assert ("--staged-evaluation-version" in command) is staged
    assert command[command.index("--seed") + 1] == str(seed)
    summary = json.loads((output / f"seed_{seed:03d}" / "summary.json").read_text())
    assert summary["full_duration_evaluation"] is effective_full
    assert summary["penetration_guard_enabled"] is not effective_full

  expected_version = "staged_v1" if staged else None
  for metadata in (protocol, aggregate):
    assert metadata["full_duration_evaluation"] is effective_full
    assert metadata["staged_evaluation_version"] == expected_version
  assert protocol["penetration_guard"]["enabled"] is not effective_full
  assert protocol["penetration_guard"]["fallen_card_guard_enabled"] is not effective_full
  assert aggregate["penetration_guard_enabled"] is not effective_full


@pytest.mark.parametrize("staged", [False, True])
def test_batch_duration_identity_still_rejects_an_inconsistent_child(batch_run, staged):
  run, commands, overrides, output = batch_run
  overrides["full_duration_evaluation"] = not staged
  arguments = ["--staged-evaluation-version", "staged_v1"] if staged else []
  with pytest.raises(RuntimeError, match="trial identity mismatch.*full_duration_evaluation"):
    run(arguments)
  assert len(commands) == 1
  assert not (output / "summary.json").exists()


def test_staged_duration_fix_does_not_relax_checkpoint_identity(batch_run):
  run, commands, overrides, output = batch_run
  overrides["checkpoint_sha256"] = "wrong-checkpoint"
  with pytest.raises(RuntimeError, match="trial identity mismatch.*checkpoint_sha256"):
    run(["--staged-evaluation-version", "staged_v1"])
  assert len(commands) == 1
  assert not (output / "summary.json").exists()
