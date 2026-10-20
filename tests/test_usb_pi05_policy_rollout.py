from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
  path = ROOT / "scripts/workcell/run_usb_pi05_policy.py"
  spec = importlib.util.spec_from_file_location("run_usb_pi05_policy", path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def deployment():
  return {
    "deployment_id": "d" * 64,
    "checkpoint_path": "/checkpoint/11000",
    "checkpoint_sha256": "c" * 64,
    "model_action_dim": 32,
    "action_dim": 27,
    "prediction_horizon": 30,
    "control_hz": 30,
    "joint_names": [f"joint_{index}" for index in range(27)],
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"policy_output_is_absolute": True},
  }


def metadata():
  item = deployment()
  return {
    "evaluation_task": "usb-insert",
    "deployment_id": item["deployment_id"],
    "model_family": "pi0.5",
    "checkpoint_path": item["checkpoint_path"],
    "checkpoint_sha256": item["checkpoint_sha256"],
    "prediction_horizon": 30,
    "action_horizon": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "joint_names": item["joint_names"],
    "observation_contract": item["observation_contract"],
    "action_representation": item["action_representation"],
  }


def test_server_contract_uses_model_horizon():
  runner = load_runner()
  assert runner.validate_server(metadata(), deployment(), 30) == 30
  with pytest.raises(RuntimeError, match=r"\[1, 30\]"):
    runner.validate_server(metadata(), deployment(), 31)


def test_defaults_use_observe_guards_and_ninety_seconds():
  runner = load_runner()
  args = runner.parse_args(
    ["--deployment-manifest", "/manifest.json", "--output-dir", "/trial"]
  )
  assert args.guard_mode == "observe"
  assert args.max_sim_seconds == 90.0


def test_task_guards_observe_by_default_but_strict_mode_stops():
  runner = load_runner()
  args = runner.parse_args(
    [
      "--deployment-manifest", "/manifest.json",
      "--output-dir", "/trial",
      "--disable-penetration-guard",
    ]
  )
  assert args.disable_penetration_guard is True
  state = SimpleNamespace(
    socket_normal_load_n=0.0,
    axial_resistance_n=0.0,
    wall_normal_load_n=0.0,
    maximum_socket_penetration_m=0.0004,
  )
  outcome = SimpleNamespace(state=state)
  simulation = SimpleNamespace(
    object_pose=lambda _name: np.array([0.0, 0.0, 0.8]),
    data=SimpleNamespace(
      time=0.0,
      qpos=np.zeros(1),
      qvel=np.zeros(1),
      qacc=np.zeros(1),
      ctrl=np.zeros(1),
      warning=(),
    ),
  )
  runner.configure_evaluation_guards(simulation, "observe")
  runner.guard_usb_state(outcome, simulation)
  guard_report = runner.evaluation_guard_report(simulation)
  assert guard_report["guard_events"][0]["kind"] == "usb_socket_penetration"
  assert guard_report["guard_events"][0]["recovery"] == "continue"

  runner.configure_evaluation_guards(simulation, "strict")
  with pytest.raises(RuntimeError, match="socket penetration"):
    runner.guard_usb_state(outcome, simulation)
  runner.guard_usb_state(outcome, simulation, penetration_guard_enabled=False)
  state.socket_normal_load_n = runner.usb_config.MAX_SOCKET_NORMAL_LOAD_N + 1.0
  with pytest.raises(RuntimeError, match="socket total normal load"):
    runner.guard_usb_state(
      outcome, simulation, penetration_guard_enabled=False
    )


def test_nonfinite_simulation_is_fatal_even_in_observe_mode():
  runner = load_runner()
  state = SimpleNamespace(
    socket_normal_load_n=0.0,
    axial_resistance_n=0.0,
    wall_normal_load_n=0.0,
    maximum_socket_penetration_m=0.0,
  )
  simulation = SimpleNamespace(
    object_pose=lambda _name: np.array([0.0, 0.0, 0.8]),
    data=SimpleNamespace(
      time=0.0,
      qpos=np.array([np.nan]),
      qvel=np.zeros(1),
      qacc=np.zeros(1),
      ctrl=np.zeros(1),
      warning=(),
    ),
  )
  runner.configure_evaluation_guards(simulation, "observe")

  with pytest.raises(RuntimeError, match="nonfinite simulation state"):
    runner.guard_usb_state(SimpleNamespace(state=state), simulation)

  report = runner.evaluation_guard_report(simulation, status="error")
  assert report["physics_valid"] is False
  assert report["stop_reason"] == "fatal_error"


def test_recording_error_invalidates_formal_metrics_without_physics_relabel():
  runner = load_runner()
  report = {
    "physics_valid": True,
    "formal_metrics_valid": True,
    "stop_reason": "success",
  }
  runner._mark_record_error(report, RuntimeError("encoder failed"))
  assert report["physics_valid"] is True
  assert report["formal_metrics_valid"] is False
  assert report["stop_reason"] == "fatal_error"
  assert report["record_error"] == "RuntimeError: encoder failed"


def test_nonfinite_policy_action_is_recoverable_only_in_observe_mode():
  runner = load_runner()
  simulation = SimpleNamespace(data=SimpleNamespace(time=2.0))
  error = RuntimeError("policy actions contain nonfinite values")
  runner.configure_evaluation_guards(simulation, "observe")
  assert runner.recover_nonfinite_action(simulation, error) is True
  assert simulation._evaluation_action_rejected is True
  event = runner.evaluation_guard_report(simulation)["guard_events"][0]
  assert event["kind"] == "nonfinite_action"
  assert event["recovery"] == "hold_last_valid_command_and_replan"

  runner.configure_evaluation_guards(simulation, "strict")
  assert runner.recover_nonfinite_action(simulation, error) is False


def test_server_contract_rejects_wrong_checkpoint():
  runner = load_runner()
  wrong = metadata()
  wrong["checkpoint_sha256"] = "wrong"
  with pytest.raises(RuntimeError, match="identity mismatch"):
    runner.validate_server(wrong, deployment(), 8)


def test_actions_require_exact_physical_shape_and_finite_values():
  runner = load_runner()
  values = np.zeros((30, 27), dtype=np.float32)
  actual = runner.validated_actions({"actions": values}, horizon=30, action_dim=27)
  assert actual.shape == (30, 27)
  with pytest.raises(RuntimeError, match="shape"):
    runner.validated_actions(
      {"actions": np.zeros((30, 32))}, horizon=30, action_dim=27
    )
  values[2, 5] = np.nan
  with pytest.raises(RuntimeError, match="nonfinite"):
    runner.validated_actions({"actions": values}, horizon=30, action_dim=27)
