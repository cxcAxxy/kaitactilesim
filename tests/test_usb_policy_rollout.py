"""USB EgoSteer rollout contract checks without network or rendering."""

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.usb_insert.review_metrics import UsbPolicyOutcome


@pytest.fixture
def module(monkeypatch):
  directory = Path(__file__).parents[1] / "scripts" / "workcell"
  monkeypatch.syspath_prepend(str(directory))
  return run_path(str(directory / "run_usb_egosteer_policy.py"))


def deployment():
  return {
    "deployment_id": "usb-test",
    "model_family": "EgoSteer",
    "checkpoint_path": "/checkpoint",
    "checkpoint_sha256": "abc",
    "prediction_horizon": 32,
    "action_dim": 48,
    "observation_contract": {
      "cameras": ["head"],
      "tactile_sent_to_model": False,
      "image_history": 6,
      "image_stride": 30,
    },
    "action_representation": {"type": "relative"},
  }


def metadata(contract):
  return {
    "task": "usb-insert",
    "deployment_id": contract["deployment_id"],
    "model_family": contract["model_family"],
    "checkpoint_path": contract["checkpoint_path"],
    "checkpoint_sha256": contract["checkpoint_sha256"],
    "action_horizon": contract["prediction_horizon"],
    "action_dim": contract["action_dim"],
    "observation_contract": contract["observation_contract"],
    "action_representation": contract["action_representation"],
  }


def test_usb_defaults_match_training_randomization(module):
  args = module["parse_args"](
    ["--deployment-manifest", "/manifest.json", "--output-dir", "/trial"]
  )
  assert args.server == "ws://127.0.0.1:18782"
  assert args.control_side == "right"
  assert args.execute_steps == 5
  assert args.xy_jitter_mm == 10.0
  assert args.yaw_jitter_deg == 5.0
  assert args.record
  assert args.disable_penetration_guard is False
  assert (args.review_width, args.review_height) == (1920, 1080)
  assert "USB plug" in module["INSTRUCTION"]


def test_penetration_guard_is_optional_but_load_guard_remains(module):
  args = module["parse_args"](
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
    module["_guard_usb_state"](outcome, simulation)
  module["_guard_usb_state"](
    outcome, simulation, penetration_guard_enabled=False
  )
  state.socket_normal_load_n = (
    module["usb_config"].MAX_SOCKET_NORMAL_LOAD_N + 1.0
  )
  with pytest.raises(RuntimeError, match="socket total normal load"):
    module["_guard_usb_state"](
      outcome, simulation, penetration_guard_enabled=False
    )


@pytest.mark.parametrize(
  "arguments",
  [
    ["--execute-steps", "0"],
    ["--seed", "-1"],
    ["--max-sim-seconds", "nan"],
    ["--xy-jitter-mm", "-1"],
    ["--yaw-jitter-deg", "inf"],
    ["--hand-ik-tolerance", "0.1"],
  ],
)
def test_invalid_arguments_rejected(module, arguments):
  base = ["--deployment-manifest", "/manifest.json", "--output-dir", "/trial"]
  with pytest.raises(SystemExit):
    module["parse_args"]([*base, *arguments])


def test_zero_randomization_is_supported(module):
  args = module["parse_args"](
    [
      "--deployment-manifest",
      "/manifest.json",
      "--output-dir",
      "/trial",
      "--xy-jitter-mm",
      "0",
      "--yaw-jitter-deg",
      "0",
    ]
  )
  assert args.xy_jitter_mm == args.yaw_jitter_deg == 0.0


def test_server_identity_and_horizon_are_enforced(module):
  contract = deployment()
  assert (
    module["validate_server"](
      metadata(contract), execute_steps=8, deployment=contract
    )
    == 32
  )
  with pytest.raises(RuntimeError, match="exceeds server horizon"):
    module["validate_server"](
      metadata(contract), execute_steps=33, deployment=contract
    )
  wrong = metadata(contract)
  wrong["checkpoint_sha256"] = "wrong"
  with pytest.raises(RuntimeError, match="identity mismatch"):
    module["validate_server"](wrong, execute_steps=8, deployment=contract)


def test_usb_server_accepts_matching_right_wrist_contract(module):
  contract = deployment()
  contract["observation_contract"] = {
    **contract["observation_contract"],
    "cameras": ["head", "right_wrist"],
  }
  assert module["validate_server"](
    metadata(contract), execute_steps=8, deployment=contract
  ) == 32


def test_usb_manifest_cannot_be_replaced_by_other_task(module, tmp_path):
  payload = {
    "task": "poker-draw",
    "deployment_id": "x",
    "model_family": "EgoSteer",
    "checkpoint_path": "/x",
    "checkpoint_sha256": "x",
    "model_config_path": "/x",
    "model_config_sha256": "x",
    "normalizer_path": "/x",
    "normalizer_sha256": "x",
    "prediction_horizon": 32,
    "action_dim": 48,
    "observation_contract": {},
    "action_representation": {},
  }
  path = tmp_path / "manifest.json"
  path.write_text(__import__("json").dumps(payload))
  with pytest.raises(RuntimeError, match="task='poker-draw'"):
    module["load_deployment_manifest"](path)


def test_unaligned_positive_axis_projection_is_not_insertion_progress():
  outcome = UsbPolicyOutcome.__new__(UsbPolicyOutcome)
  outcome.state = SimpleNamespace(
    success=False,
    seated=False,
    bottom_out_confirmed=False,
    shell_fits_aperture=False,
    insertion_depth_m=0.034,
    lateral_error_m=(0.1, 0.1),
    orientation_error_rad=2.7,
    axial_resistance_n=0.0,
    socket_normal_load_n=0.0,
    maximum_socket_penetration_m=0.0,
  )
  outcome.maximum_lift_m = 0.0
  outcome.maximum_insertion_depth_m = 0.0

  assert outcome.stage() == "pickup"
  assert outcome.snapshot(None)["insertion_depth_mm"] == 0.0
