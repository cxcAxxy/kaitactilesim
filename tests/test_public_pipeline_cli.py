import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from kaihand_tactile_env.pipeline.catalog import TASKS

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COMMANDS = {
  "collect": "scripts/collect/collect.py",
  "convert-dataset": "scripts/convert/convert.py",
  "deploy-policy": "scripts/deploy/deploy.py",
  "evaluate-policy": "scripts/evaluate/evaluate.py",
}


def invoke(script, *arguments):
  environment = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
  }
  return subprocess.run(
    [sys.executable, str(ROOT / script), *arguments],
    cwd=ROOT,
    env=environment,
    text=True,
    capture_output=True,
    timeout=30,
    check=False,
  )


@pytest.mark.parametrize("name", PUBLIC_COMMANDS)
def test_public_pixi_entrypoint_help_is_executable(name):
  config = tomllib.loads((ROOT / "pixi.toml").read_text())
  task = config["tasks"][name]
  assert task["cmd"] == f"python {PUBLIC_COMMANDS[name]}"
  assert "[stable]" in task["description"]
  result = invoke(PUBLIC_COMMANDS[name], "--help")
  assert result.returncode == 0, result.stderr
  assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize(
  ("name", "rows_key"),
  [
    ("collect", "tasks"),
    ("convert-dataset", "adapters"),
    ("deploy-policy", "deployments"),
    ("evaluate-policy", "runners"),
  ],
)
def test_public_capability_query_needs_no_dataset_or_model(name, rows_key):
  result = invoke(PUBLIC_COMMANDS[name], "--list-support")
  assert result.returncode == 0, result.stderr
  payload = json.loads(result.stdout)
  assert {row["task"] for row in payload[rows_key]} == set(TASKS)


@pytest.mark.parametrize("task", ["whiteboard-wipe", "whiteboard-wipe-horizontal-line"])
def test_deployment_prepare_plan_uses_runtime_without_creating_outputs(tmp_path, task):
  runtime = tmp_path / "runtimes.toml"
  runtime.write_text(
    '[runtimes.pi05]\npython="model-env/bin/python"\nproject_root="openpi"\n'
  )
  output = tmp_path / "deployment"
  result = invoke(
    PUBLIC_COMMANDS["deploy-policy"],
    "prepare", "--task", task, "--model-family", "pi05",
    "--checkpoint", str(tmp_path / "checkpoint"), "--output-dir", str(output),
    "--runtime-config", str(runtime), "--runtime", "pi05", "--dry-run",
  )
  assert result.returncode == 0, result.stderr
  payload = json.loads(result.stdout)
  command = payload["prepare"]["command"]
  assert command[0] == str(tmp_path / "model-env/bin/python")
  assert command[command.index("--openpi-root") + 1] == str(tmp_path / "openpi")
  assert not output.exists()


def test_line_collection_fixed_stains_plan_does_not_create_dataset(tmp_path):
  output = tmp_path / "line_raw"
  result = invoke(
    PUBLIC_COMMANDS["collect"],
    "--task", "whiteboard-wipe-horizontal-line", "--fixed-stains",
    "--target-successes", "200", "--max-attempts", "1000", "--workers", "8",
    "--output-dir", str(output), "--staging-root", str(tmp_path / "staging"),
    "--render-backend", "hardware", "--dry-run",
  )
  assert result.returncode == 0, result.stderr
  payload = json.loads(result.stdout)
  assert payload["task"] == "whiteboard-wipe-horizontal-line"
  assert payload["target_successes"] == 200
  assert payload["workers"] == 8
  assert "--seed" not in payload["first_command"]
  assert not output.exists()


def test_unknown_runtime_returns_failure_before_starting_prepare(tmp_path):
  runtime = tmp_path / "runtimes.toml"
  runtime.write_text('[runtimes.pi05]\npython="model-env/bin/python"\n')
  output = tmp_path / "deployment"
  result = invoke(
    PUBLIC_COMMANDS["deploy-policy"],
    "prepare", "--task", "whiteboard-wipe", "--model-family", "pi05",
    "--checkpoint", str(tmp_path / "checkpoint"), "--output-dir", str(output),
    "--runtime-config", str(runtime), "--runtime", "typo",
  )
  assert result.returncode != 0
  assert "unknown runtime" in result.stderr
  assert not output.exists()
