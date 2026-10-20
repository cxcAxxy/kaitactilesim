"""USB compilation happens before the listener and leaves scored RNG unchanged."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server():
  path = ROOT / "scripts/workcell/serve_usb_pi05_policy.py"
  spec = importlib.util.spec_from_file_location("usb_warmup_under_test", path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def manifest(*, horizon=50, schema="usb_pi05_deployment_v2"):
  return {
    "schema": schema,
    "model_config_name": "pi05_kaihand_h50" if horizon == 50 else "pi05_kaihand",
    "prediction_horizon": horizon,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "suggested_execute_steps": 8,
    "joint_names": [f"joint_{i}" for i in range(27)],
    "instruction": "Insert the USB plug.",
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"policy_output_is_absolute": True},
    "required_action_interpolation": "linear",
    "checkpoint_path": "/mock/checkpoints/20000",
    "normalizer_asset_id": "normalizer_h50_usb_insert_hand_tail_v1",
    "model_family": "pi0.5",
    "deployment_id": "test-usb-deployment",
    "checkpoint_sha256": "a" * 64,
    "checkpoint_step": 20000,
  }


@pytest.mark.parametrize("horizon", [30, 50])
def test_warmup_exact_observation_and_shape_restores_rng(server, horizon):
  payload = manifest(horizon=horizon)
  original_rng = object()
  requests = []

  class Policy:
    _rng = original_rng

    def infer(self, request):
      requests.append(request)
      self._rng = object()
      return {"actions": np.zeros((horizon, 27), dtype=np.float32)}

  policy = Policy()
  times = iter([10.0, 42.0])
  result = server.warmup_policy(policy, payload, monotonic=lambda: next(times))
  assert policy._rng is original_rng
  assert len(requests) == 1
  request = requests[0]
  assert list(request) == ["head_image", "right_wrist_image", "state", "prompt"]
  for key in ("head_image", "right_wrist_image"):
    assert request[key].shape == (240, 320, 3)
    assert request[key].dtype == np.uint8
  assert request["state"].shape == (27,)
  assert request["state"].dtype == np.float32
  assert request["prompt"] == payload["instruction"]
  assert result == {
    "completed": True,
    "duration_s": 32.0,
    "request_image_shape_hwc": [240, 320, 3],
    "state_dim": 27,
    "action_shape": [horizon, 27],
    "rng_restored": True,
  }


def test_warmup_honors_manifest_camera_request_keys(server):
  payload = manifest()
  payload["observation_contract"]["request_image_keys"] = {
    "head": "custom_head",
    "right_wrist": "custom_wrist",
  }
  requests = []

  class Policy:
    _rng = object()

    def infer(self, request):
      requests.append(request)
      return {"actions": np.zeros((50, 27), dtype=np.float32)}

  server.warmup_policy(Policy(), payload)
  assert list(requests[0]) == ["custom_head", "custom_wrist", "state", "prompt"]


@pytest.mark.parametrize(
  "response",
  [
    None,
    {},
    {"actions": np.zeros((30, 27), dtype=np.float32)},
    {"actions": np.zeros((50, 32), dtype=np.float32)},
    {"actions": np.full((50, 27), np.nan)},
    {"actions": np.full((50, 27), np.inf)},
    {"actions": np.full((50, 27), "invalid")},
  ],
)
def test_warmup_invalid_response_fails_with_rng_restored(server, response):
  original_rng = object()

  class Policy:
    _rng = original_rng

    def infer(self, _request):
      self._rng = object()
      return response

  policy = Policy()
  with pytest.raises(RuntimeError, match="USB pi0.5 warmup"):
    server.warmup_policy(policy, manifest())
  assert policy._rng is original_rng


def test_warmup_inference_error_restores_rng(server):
  original_rng = object()

  class Policy:
    _rng = original_rng

    def infer(self, _request):
      self._rng = object()
      raise RuntimeError("mock compilation failed")

  policy = Policy()
  with pytest.raises(RuntimeError, match="mock compilation failed"):
    server.warmup_policy(policy, manifest())
  assert policy._rng is original_rng


def test_warmup_rejects_policy_without_restorable_rng(server):
  with pytest.raises(RuntimeError, match="expose its JAX RNG"):
    server.warmup_policy(object(), manifest())


def configure_main(server, monkeypatch, payload, *, fail_warmup=False):
  events = []
  policy_arguments = {}
  original_rng = object()
  metadata = {
    "action_dim": payload["action_dim"],
    "model_action_dim": payload["model_action_dim"],
    "action_horizon": payload["prediction_horizon"],
    "control_hz": payload["control_hz"],
    "suggested_replan_steps": payload["suggested_execute_steps"],
    "joint_names": payload["joint_names"],
    "image_keys": ["observation.images.head", "observation.images.right_wrist"],
    "action_semantics": payload["action_representation"],
  }
  config = SimpleNamespace(
    policy_metadata=metadata,
    model=SimpleNamespace(action_horizon=payload["prediction_horizon"], action_dim=32),
    data=SimpleNamespace(use_delta_arm_actions=True),
  )

  class Policy:
    _rng = original_rng

    def infer(self, _request):
      events.append("warmup")
      self._rng = object()
      if fail_warmup:
        raise RuntimeError("mock compilation failed")
      return {
        "actions": np.zeros((payload["prediction_horizon"], 27), dtype=np.float32)
      }

  policy = Policy()

  def create_trained_policy(given_config, checkpoint, **kwargs):
    assert given_config is config
    assert checkpoint == payload["checkpoint_path"]
    policy_arguments.update(kwargs)
    events.append("create_policy")
    return policy

  class WebsocketPolicyServer:
    def __init__(self, *, policy, host, port, metadata):
      events.append("server_init")
      assert policy._rng is original_rng
      assert (host, port) == ("0.0.0.0", 18804)
      if payload["schema"] == server.SCHEMA_V2:
        assert metadata["server_warmup"]["completed"] is True
        assert metadata["server_warmup"]["rng_restored"] is True
        assert metadata["server_warmup"]["action_shape"] == [
          payload["prediction_horizon"],
          27,
        ]
      else:
        assert "server_warmup" not in metadata

    def serve_forever(self):
      events.append("listen")

  modules = {
    name: ModuleType(name)
    for name in ("openpi", "openpi.policies", "openpi.serving", "openpi.training")
  }
  modules["openpi.policies"].policy_config = SimpleNamespace(
    create_trained_policy=create_trained_policy
  )
  modules["openpi.serving"].websocket_policy_server = SimpleNamespace(
    WebsocketPolicyServer=WebsocketPolicyServer
  )
  modules["openpi.training"].config = SimpleNamespace(get_config=lambda _name: config)
  for name, module in modules.items():
    monkeypatch.setitem(sys.modules, name, module)
  monkeypatch.setattr(server, "_load_manifest", lambda path: (path, payload))
  monkeypatch.setattr(
    server, "verify_pi05_deployment", lambda *_args: events.append("verify")
  )
  monkeypatch.setattr(server, "bind_checkpoint_assets", lambda given, _manifest: given)
  provenance = (
    {"record": {"normalizer": {"use_quantile_norm": True}}}
    if payload["schema"] == server.SCHEMA_V2
    else None
  )
  monkeypatch.setattr(
    server,
    "checkpoint_contract",
    lambda _path: (
      payload["model_config_name"],
      payload["normalizer_asset_id"],
      provenance,
    ),
  )
  return events, policy, original_rng, policy_arguments


@pytest.mark.parametrize(
  "horizon,schema",
  [
    (50, "usb_pi05_deployment_v2"),
    (30, "usb_pi05_deployment_v2"),
    (30, "usb_pi05_deployment_v1"),
  ],
)
def test_main_warmup_precedes_listener_and_legacy_h30_unchanged(
  server, monkeypatch, caplog, horizon, schema
):
  payload = manifest(horizon=horizon, schema=schema)
  events, policy, original_rng, kwargs = configure_main(server, monkeypatch, payload)
  with caplog.at_level(logging.INFO):
    server.main(["--deployment-manifest", "/mock/deployment.json", "--port", "18804"])
  assert policy._rng is original_rng
  if schema == server.SCHEMA_V2:
    assert events == ["verify", "create_policy", "warmup", "server_init", "listen"]
    assert "WARMUP_START task=usb-insert" in caplog.text
    assert "WARMUP_OK" in caplog.text
    assert kwargs["expected_use_quantile_norm"] is True
  else:
    assert events == ["verify", "create_policy", "server_init", "listen"]
    assert "WARMUP_START" not in caplog.text
    assert kwargs["expected_use_quantile_norm"] is None
  assert kwargs["default_prompt"] == payload["instruction"]


def test_failed_warmup_never_constructs_or_opens_listener(server, monkeypatch):
  events, policy, original_rng, _kwargs = configure_main(
    server, monkeypatch, manifest(), fail_warmup=True
  )
  with pytest.raises(RuntimeError, match="mock compilation failed"):
    server.main(["--deployment-manifest", "/mock/deployment.json", "--port", "18804"])
  assert policy._rng is original_rng
  assert events == ["verify", "create_policy", "warmup"]
