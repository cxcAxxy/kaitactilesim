"""Lightweight CLI/protocol checks; no robot allocation or network access."""

from pathlib import Path
from runpy import run_path

import pytest


@pytest.fixture
def module(monkeypatch):
  directory = Path(__file__).parents[1] / "scripts" / "workcell"
  monkeypatch.syspath_prepend(str(directory))
  return run_path(str(directory / "run_poker_egosteer_policy.py"))


def test_defaults_are_poker_only_and_bounded(module):
  args = module["parse_args"]([])
  assert args.server == "ws://127.0.0.1:8766"
  assert args.control_side == "right"
  assert args.execute_steps == 5
  assert args.max_wrist_jump == 0.15
  assert args.real_time is True
  assert args.record
  assert args.disable_penetration_guard is False
  assert (args.review_width, args.review_height) == (1920, 1080)
  assert module["CONTROLLER"] == "poker-contact-feedback-v2"
  assert "face-down card" in module["INSTRUCTION"]


@pytest.mark.parametrize(
  "arguments",
  [
    ["--execute-steps", "0"],
    ["--seed", "-1"],
    ["--max-requests", "-1"],
    ["--max-sim-seconds", "nan"],
    ["--viewer-hz", "0"],
    ["--max-wrist-jump", "inf"],
    ["--hand-ik-tolerance", "0.1"],
  ],
)
def test_invalid_arguments_rejected(module, arguments):
  with pytest.raises(SystemExit):
    module["parse_args"](arguments)


def test_headless_is_not_real_time(module):
  args = module["parse_args"](["--viewer", "none"])
  assert args.real_time is False
  assert args.record is True
  args = module["parse_args"](["--viewer", "none", "--record-fps", "10"])
  assert args.record and args.record_fps == 10
  assert module["parse_args"](["--no-record"]).record is False
  assert module["parse_args"](["--disable-penetration-guard"]).disable_penetration_guard


@pytest.mark.parametrize(
  "metadata",
  [
    {},
    {"action_dim": 108, "action_horizon": 50},
    {"action_dim": 48, "action_horizon": 0},
    {"action_dim": 48, "action_horizon": 3.5},
  ],
)
def test_wrong_server_contract_rejected(module, metadata):
  with pytest.raises(RuntimeError):
    module["validate_server"](metadata)


@pytest.mark.parametrize("horizon", [5, 20, 32, 50])
def test_server_horizon_is_model_owned(module, horizon):
  assert (
    module["validate_server"](
      {"action_dim": 48, "action_horizon": horizon}, execute_steps=5
    )
    == horizon
  )


def test_execution_cannot_exceed_model_horizon(module):
  with pytest.raises(RuntimeError, match="exceeds server horizon"):
    module["validate_server"](
      {"action_dim": 48, "action_horizon": 20}, execute_steps=21
    )
