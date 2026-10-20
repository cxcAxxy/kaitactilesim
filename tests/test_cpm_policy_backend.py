from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "scripts/workcell/cpm_policy_backend.py"


def _load_backend_module():
  spec = importlib.util.spec_from_file_location("cpm_policy_backend_test", BACKEND_PATH)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _manifest(project_root: Path, *, scale=1.0, variant="pi05+touchguidance"):
  return {
    "schema": "poker_tactile_deployment_v1",
    "task": "poker-draw",
    "model_family": "pi05+cpm",
    "deployment_id": "d" * 64,
    "prediction_horizon": 30,
    "execute_steps": 5,
    "source": {
      "base_checkpoint": "/models/pi05-20000",
      "base_checkpoint_sha256": "b" * 64,
      "cpm_checkpoint": "/models/cpm/best.pt",
      "cpm_checkpoint_sha256": "c" * 64,
      "data_root": "/datasets/cpm",
      "tokenizer_model": "/models/tokenizer.model",
    },
    "guidance": {
      "num_steps": 10,
      "guidance_steps": 3,
      "scale": scale,
      "variant": variant,
      "compare_base": True,
      "return_trace": False,
    },
    "runtime": {
      "project_root": str(project_root),
      "device": "cuda:0",
      "precision": "bfloat16",
    },
  }


def _policy_metadata(execute_steps=5):
  return {
    "schema": "openpi_poker_touchguide_policy_v1",
    "task": "poker-draw",
    "prediction_horizon": 30,
    "execute_steps": execute_steps,
    "control_hz": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "image_shape_hwc": [240, 320, 3],
    "tactile_shape": [5, 3, 7, 5],
    "joint_names": [f"joint_{index}" for index in range(27)],
    "instruction": "draw the poker card",
  }


def _default_response(request):
  actions = np.arange(30 * 27, dtype=np.float32).reshape(30, 27) / 1000
  return {
    "actions": actions,
    "base_actions": actions - np.float32(0.01),
    "diagnostics": {
      "variant": request["variant"],
      "noise_seed": np.uint64(123),
      "steps": [{"guided": True, "score_before": np.float32(0.2)}],
    },
  }


def _install_fake_openpi(monkeypatch, capture, response=None):
  openpi = ModuleType("openpi")
  openpi.__path__ = []
  touchguide = ModuleType("openpi.touchguide")
  touchguide.__path__ = []
  inference = ModuleType("openpi.touchguide.inference")
  policy = ModuleType("openpi.touchguide.policy")
  sampler = ModuleType("openpi.touchguide.sampler")

  class FakeGuidanceConfig:
    def __init__(self, *, num_steps, guidance_steps, scale):
      self.num_steps = num_steps
      self.guidance_steps = guidance_steps
      self.scale = scale

  def load_touchguide(**kwargs):
    capture["load"] = kwargs
    return SimpleNamespace(metadata={"loaded": True})

  class FakePolicy:
    def __init__(self, loaded, *, tokenizer_model, execute_steps):
      capture["policy_init"] = {
        "loaded": loaded,
        "tokenizer_model": tokenizer_model,
        "execute_steps": execute_steps,
      }
      self.metadata = _policy_metadata(execute_steps)

    def infer(self, request):
      capture["request"] = request
      selected = capture.get("response", response)
      return (selected or _default_response)(request) if callable(selected or _default_response) else selected

  sampler.GuidanceConfig = FakeGuidanceConfig
  inference.load_touchguide = load_touchguide
  policy.PokerTouchGuidePolicy = FakePolicy
  openpi.touchguide = touchguide
  touchguide.inference = inference
  touchguide.policy = policy
  touchguide.sampler = sampler
  for name, module in {
    "openpi": openpi,
    "openpi.touchguide": touchguide,
    "openpi.touchguide.inference": inference,
    "openpi.touchguide.policy": policy,
    "openpi.touchguide.sampler": sampler,
  }.items():
    monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def backend_fixture(tmp_path, monkeypatch):
  # The backend intentionally mutates sys.path for the lifetime of a worker.
  # Give each test an isolated list so pytest can restore the process value.
  monkeypatch.setattr(sys, "path", list(sys.path))
  project = tmp_path / "openpi-main"
  project.mkdir()
  capture = {}
  _install_fake_openpi(monkeypatch, capture)
  module = _load_backend_module()
  backend = module.CPMPolicyBackend(_manifest(project))
  return module, backend, capture, project


def _request():
  return {
    "state": [0.0] * 27,
    "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
    "right_wrist_image": np.full((240, 320, 3), 127, dtype=np.uint8),
    "tactile_map_raw": np.zeros((5, 3, 7, 5), dtype=np.float32),
    "prompt": "draw the poker card",
    "episode_seed": 7,
    "request_index": 2,
    "execute_steps": 5,
  }


def test_loads_declared_openpi_runtime_and_exposes_bound_metadata(backend_fixture):
  _module, backend, capture, project = backend_fixture
  loaded = capture["load"]
  assert loaded["base_checkpoint"] == "/models/pi05-20000"
  assert loaded["cpm_checkpoint"] == "/models/cpm/best.pt"
  assert loaded["data_root"] == "/datasets/cpm"
  assert loaded["device"] == "cuda:0"
  assert loaded["precision"] == "bfloat16"
  assert loaded["guidance"].num_steps == 10
  assert loaded["guidance"].guidance_steps == 3
  assert loaded["guidance"].scale == 1.0
  assert capture["policy_init"]["tokenizer_model"] == "/models/tokenizer.model"
  assert str(project / "src") == sys.path[0]

  metadata = backend.metadata()
  assert metadata["schema"] == "poker_tactile_deployment_v1"
  assert metadata["policy_schema"] == "openpi_poker_touchguide_policy_v1"
  assert metadata["model_family"] == "pi05+cpm"
  assert metadata["prediction_horizon"] == 30
  assert metadata["action_start_index"] == 0
  assert metadata["tactile_sent_to_model"] is True
  assert metadata["observation_contract"]["tactile_sent_to_model"] is True
  assert metadata["observation_contract"]["tactile_shape"] == [5, 3, 7, 5]
  assert metadata["source"]["cpm_checkpoint_sha256"] == "c" * 64
  json.dumps(metadata, allow_nan=False)

  metadata["source"]["cpm_checkpoint"] = "mutated"
  assert backend.metadata()["source"]["cpm_checkpoint"] == "/models/cpm/best.pt"


def test_infer_accepts_json_lists_and_returns_json_safe_h30(backend_fixture):
  _module, backend, capture, _project = backend_fixture
  request = _request()
  request["head_image"] = request["head_image"].tolist()
  request["tactile_map_raw"] = request["tactile_map_raw"].tolist()
  result = backend.infer(request)

  sent = capture["request"]
  assert sent["state"].shape == (27,)
  assert sent["state"].dtype == np.float32
  assert sent["head_image"].shape == (240, 320, 3)
  assert sent["head_image"].dtype == np.uint8
  assert sent["right_wrist_image"].dtype == np.uint8
  assert sent["tactile_map_raw"].shape == (5, 3, 7, 5)
  assert sent["tactile_map_raw"].dtype == np.float32
  assert sent["variant"] == "pi05+touchguidance"
  assert sent["compare_base"] is True
  assert sent["return_trace"] is False

  assert np.asarray(result["actions"]).shape == (30, 27)
  assert np.asarray(result["base_actions"]).shape == (30, 27)
  assert result["diagnostics"]["variant"] == "pi05+touchguidance"
  assert result["diagnostics"]["noise_seed"] == 123
  json.dumps(result, allow_nan=False)


def test_scale_zero_selects_frozen_pi05_control(tmp_path, monkeypatch):
  monkeypatch.setattr(sys, "path", list(sys.path))
  project = tmp_path / "openpi-main"
  project.mkdir()
  capture = {}
  _install_fake_openpi(monkeypatch, capture)
  module = _load_backend_module()
  backend = module.CPMPolicyBackend(_manifest(project, scale=0.0, variant="pi05"))
  result = backend.infer(_request())
  assert capture["request"]["variant"] == "pi05"
  assert capture["load"]["guidance"].scale == 0.0
  assert result["diagnostics"]["variant"] == "pi05"


@pytest.mark.parametrize(
  ("edit", "match"),
  [
    (lambda row: row.update(schema="old_schema"), "identity mismatch"),
    (lambda row: row.update(model_family="pi0.5"), "identity mismatch"),
    (lambda row: row.update(prediction_horizon=50), "identity mismatch"),
    (lambda row: row.update(execute_steps=0), "execute_steps"),
    (lambda row: row.update(execute_steps=31), "execute_steps"),
    (lambda row: row["guidance"].update(num_steps=9), "num_steps"),
    (
      lambda row: row["guidance"].update(variant="pi05", scale=1.0),
      "requires guidance.scale=0",
    ),
    (
      lambda row: row["guidance"].update(variant="pi05+touchguidance", scale=0.0),
      "guided variant requires",
    ),
  ],
)
def test_manifest_rejects_identity_horizon_and_guidance_drift(tmp_path, edit, match):
  project = tmp_path / "openpi-main"
  project.mkdir()
  manifest = _manifest(project)
  edit(manifest)
  module = _load_backend_module()
  with pytest.raises(ValueError, match=match):
    module.CPMPolicyBackend(manifest)


@pytest.mark.parametrize(
  ("edit", "match"),
  [
    (lambda row: row.update(execute_steps=6), "manifest-bound"),
    (lambda row: row.update(episode_seed=True), "episode_seed"),
    (lambda row: row.update(request_index=-1), "request_index"),
    (lambda row: row.update(state=[0.0] * 26), "state must have shape"),
    (lambda row: row.update(state=[float("nan")] * 27), "finite"),
    (
      lambda row: row.update(head_image=np.full((240, 320, 3), 0.5)),
      "integer RGB values",
    ),
    (lambda row: row.update(tactile_map_raw=np.zeros((5, 3, 7, 4))), "tactile_map_raw"),
  ],
)
def test_request_validation_rejects_contract_drift(backend_fixture, edit, match):
  _module, backend, _capture, _project = backend_fixture
  request = _request()
  edit(request)
  with pytest.raises(ValueError, match=match):
    backend.infer(request)


@pytest.mark.parametrize(
  ("response", "match"),
  [
    (
      lambda request: {
        "actions": np.zeros((29, 27), dtype=np.float32),
        "diagnostics": {"variant": request["variant"]},
      },
      "actions must have shape",
    ),
    (
      lambda request: {
        "actions": np.full((30, 27), np.nan, dtype=np.float32),
        "diagnostics": {"variant": request["variant"]},
      },
      "finite",
    ),
    (
      lambda _request: {
        "actions": np.zeros((30, 27), dtype=np.float32),
        "diagnostics": {"variant": "pi05"},
      },
      "diagnostics variant",
    ),
  ],
)
def test_rejects_invalid_openpi_response(backend_fixture, response, match):
  _module, backend, capture, _project = backend_fixture
  capture["response"] = response
  with pytest.raises(ValueError, match=match):
    backend.infer(_request())
