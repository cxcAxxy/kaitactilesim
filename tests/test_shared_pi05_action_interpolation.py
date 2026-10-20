from __future__ import annotations

import sys
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def runner():
  workcell = Path(__file__).parents[1] / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  try:
    yield run_path(str(workcell / "run_shared_task_pi05_policy.py"))
  finally:
    sys.path.remove(str(workcell))


@pytest.fixture(scope="module")
def batch():
  workcell = Path(__file__).parents[1] / "scripts/workcell"
  yield run_path(str(workcell / "evaluate_shared_task_policy_batch.py"))


def _runner_arguments() -> list[str]:
  return [
    "--task",
    "bulb-screw",
    "--server",
    "ws://127.0.0.1:18784",
    "--deployment-manifest",
    "/tmp/deployment.json",
    "--output-dir",
    "/tmp/evaluation",
    "--seed",
    "0",
  ]


def test_runner_defaults_to_linear_and_keeps_zoh_ablation(runner):
  parse_args = runner["parse_args"]

  defaults = parse_args(_runner_arguments())
  assert defaults.action_interpolation == "linear"
  assert defaults.guard_mode == "observe"
  assert (
    parse_args(
      [*_runner_arguments(), "--action-interpolation", "zoh"]
    ).action_interpolation
    == "zoh"
  )


def test_batch_defaults_to_linear_and_keeps_zoh_ablation(batch):
  parse_args = batch["parse_args"]
  common = [
    "--deployment-manifest",
    "/tmp/deployment.json",
    "--output-dir",
    "/tmp/evaluation",
    "--seeds",
    "0",
    "--video-count",
    "0",
  ]

  _, args, passthrough = parse_args(common)
  assert args.action_interpolation == "linear"
  assert args.guard_mode == "observe"
  assert passthrough == []
  _, args, passthrough = parse_args([*common, "--action-interpolation", "zoh"])
  assert args.action_interpolation == "zoh"
  assert passthrough == []


def test_runner_and_batch_publish_the_same_controller_version(runner, batch):
  assert runner["CONTROLLER_VERSION"] == batch["PI05_CONTROLLER_VERSION"]


def test_runner_validity_combines_provenance_interpolation_and_stage_context(runner):
  validity = runner["_evaluation_validity_metadata"]
  formal = {
    "schema": "shared_task_pi05_deployment_v2",
    "diagnostic_only": False,
    "required_action_interpolation": "linear",
  }
  observe_args = runner["parse_args"](_runner_arguments())
  assert validity(observe_args, formal) == {
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "diagnostic_context": None,
    "diagnostic_reasons": ["guard_mode_observe"],
    "guard_mode": "observe",
    "whiteboard_contact_guards_enabled": None,
  }
  linear_args = runner["parse_args"](
    [*_runner_arguments(), "--guard-mode", "strict"]
  )
  zoh_args = runner["parse_args"](
    [
      *_runner_arguments(),
      "--action-interpolation",
      "zoh",
      "--guard-mode",
      "strict",
    ]
  )

  assert validity(linear_args, formal) == {
    "diagnostic_only": False,
    "formal_metrics_valid": True,
    "diagnostic_context": None,
    "diagnostic_reasons": [],
    "guard_mode": "strict",
    "whiteboard_contact_guards_enabled": None,
  }
  zoh = validity(zoh_args, formal)
  assert zoh["diagnostic_only"] is True
  assert zoh["formal_metrics_valid"] is False
  assert zoh["diagnostic_reasons"] == [
    "action_interpolation_zoh_differs_from_required_linear"
  ]

  historical = {"schema": "shared_task_pi05_deployment_v1"}
  staged = validity(
    linear_args,
    historical,
    diagnostic_context="reference_stage_initialization",
  )
  assert staged["diagnostic_reasons"] == [
    "legacy_v1_unbound_deployment",
    "reference_stage_initialization",
  ]
  assert staged["formal_metrics_valid"] is False


def test_current_controller_target_uses_goal_stream_not_measured_qpos(runner):
  arm = np.arange(7, dtype=float) + 0.25
  right_names = [f"hand_r_test_joint_{index}" for index in range(20)]
  left_names = [f"hand_l_test_joint_{index}" for index in range(20)]
  all_names = tuple(left_names + right_names)
  all_targets = np.arange(40, dtype=float) + 10.0
  simulation = SimpleNamespace(
    arm_goal={"right": arm},
    command_state=lambda: (np.zeros(14), all_targets, all_names),
    data=SimpleNamespace(qpos=np.full(27, -123.0)),
  )
  joint_names = [f"right_arm_joint{index}" for index in range(1, 8)] + right_names

  target = runner["current_controller_target"](simulation, joint_names)

  np.testing.assert_array_equal(target[:7], arm)
  np.testing.assert_array_equal(target[7:], all_targets[20:])
  assert not np.any(target == -123.0)


def test_linear_substep_starts_at_controller_target_and_reuses_endpoint(runner):
  interpolate = runner["linear_substep_action"]
  initial = np.zeros(27)
  first_endpoint = np.ones(27)
  second_endpoint = np.full(27, 3.0)
  timestep = 0.002
  interval = 1.0 / 30.0

  first_tick = interpolate(
    initial,
    first_endpoint,
    interval_start_s=0.0,
    interval_end_s=interval,
    current_time_s=0.0,
    timestep_s=timestep,
  )
  completed_first = interpolate(
    initial,
    first_endpoint,
    interval_start_s=0.0,
    interval_end_s=interval,
    current_time_s=interval - timestep,
    timestep_s=timestep,
  )
  first_tick_next_interval = interpolate(
    completed_first,
    second_endpoint,
    interval_start_s=interval,
    interval_end_s=2.0 * interval,
    current_time_s=interval,
    timestep_s=timestep,
  )

  np.testing.assert_allclose(first_tick, 0.06)
  np.testing.assert_array_equal(completed_first, first_endpoint)
  np.testing.assert_allclose(first_tick_next_interval, 1.12)


@pytest.mark.parametrize(
  ("start", "end", "timing"),
  [
    (np.zeros(26), np.ones(26), (0.0, 0.1, 0.0, 0.002)),
    (np.zeros(27), np.ones(27), (0.0, 0.0, 0.0, 0.002)),
    (np.zeros(27), np.ones(27), (0.0, 0.1, np.nan, 0.002)),
  ],
)
def test_linear_substep_rejects_invalid_contract(runner, start, end, timing):
  with pytest.raises(ValueError, match="finite width-27 actions"):
    runner["linear_substep_action"](
      start,
      end,
      interval_start_s=timing[0],
      interval_end_s=timing[1],
      current_time_s=timing[2],
      timestep_s=timing[3],
    )
