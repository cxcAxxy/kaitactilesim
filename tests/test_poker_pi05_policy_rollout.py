from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
  workcell = ROOT / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  try:
    path = workcell / "run_poker_pi05_policy.py"
    spec = importlib.util.spec_from_file_location("run_poker_pi05_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(workcell))


def deployment():
  return {
    "deployment_id": "d" * 64,
    "checkpoint_path": "/checkpoint/25000",
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
    "evaluation_task": "poker-draw",
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


def test_server_contract_uses_checkpoint_horizon():
  runner = load_runner()
  assert runner.validate_server(metadata(), deployment(), 30) == 30
  with pytest.raises(RuntimeError, match=r"\[1, 30\]"):
    runner.validate_server(metadata(), deployment(), 31)


def test_server_contract_rejects_wrong_task_and_missing_wrist():
  runner = load_runner()
  wrong = metadata()
  wrong["evaluation_task"] = "usb-insert"
  with pytest.raises(RuntimeError, match="identity mismatch"):
    runner.validate_server(wrong, deployment(), 8)

  wrong = metadata()
  wrong["observation_contract"] = {
    **wrong["observation_contract"],
    "cameras": ["head"],
  }
  with pytest.raises(RuntimeError, match="identity mismatch"):
    runner.validate_server(wrong, deployment(), 8)


def test_randomization_and_review_camera_guards():
  runner = load_runner()
  with pytest.raises(SystemExit):
    runner.parse_args(
      [
        "--deployment-manifest",
        "manifest.json",
        "--output-dir",
        "trial",
        "--xy-jitter-mm",
        "5.1",
      ]
    )
  with pytest.raises(SystemExit):
    runner.parse_args(
      [
        "--deployment-manifest",
        "manifest.json",
        "--output-dir",
        "trial",
        "--review-second-camera",
        "right_wrist",
      ]
    )

  args = runner.parse_args(
    [
      "--deployment-manifest",
      "manifest.json",
      "--output-dir",
      "trial",
      "--disable-penetration-guard",
    ]
  )
  assert args.disable_penetration_guard is True


def test_penetration_guard_can_be_disabled_without_disabling_other_guards():
  runner = load_runner()

  simulation = SimpleNamespace(
    data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1)),
    object_pose=lambda _name: np.array([0.0, 0.0, 0.84, 1.0, 0.0, 0.0, 0.0]),
  )
  observer = SimpleNamespace(
    sim=simulation,
    _supported=lambda: True,
    _card_table_clearance=lambda: -0.0007,
  )

  with pytest.raises(RuntimeError, match="penetrated supported tabletop"):
    runner.guard_poker_state(observer)
  assert runner.guard_poker_state(
    observer, penetration_guard_enabled=False
  ) == pytest.approx(0.0007)

  simulation.data.qpos[0] = np.nan
  with pytest.raises(RuntimeError, match="nonfinite simulation state"):
    runner.guard_poker_state(observer, penetration_guard_enabled=False)
