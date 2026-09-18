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


def test_penetration_guard_is_optional_but_load_guard_remains():
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
    data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1)),
  )
  with pytest.raises(RuntimeError, match="socket penetration"):
    runner.guard_usb_state(outcome, simulation)
  runner.guard_usb_state(outcome, simulation, penetration_guard_enabled=False)
  state.socket_normal_load_n = runner.usb_config.MAX_SOCKET_NORMAL_LOAD_N + 1.0
  with pytest.raises(RuntimeError, match="socket total normal load"):
    runner.guard_usb_state(
      outcome, simulation, penetration_guard_enabled=False
    )


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
