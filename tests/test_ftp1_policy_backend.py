"""FTP-1 worker backend validates transport and normalizes exactly once."""

from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.ftp1_policy_contract import (
  RIGHT_HAND_FAAS_INDICES,
  RIGHT_JOINT_NAMES,
)
from kaihand_tactile_env.shared.tactile_task_contracts import (
  ftp1_training_binding,
  task_contract,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def backend_module():
  path = ROOT / "scripts/workcell/ftp1_policy_backend.py"
  spec = importlib.util.spec_from_file_location("ftp1_policy_backend_under_test", path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def manifest(tmp_path):
  checkpoint = tmp_path / "24999"
  checkpoint.mkdir()
  (checkpoint / "hpt_tokenizer").mkdir()
  (checkpoint / "normalization/KaiHandSimPokerDraw").mkdir(parents=True)
  (checkpoint / "model.safetensors").write_bytes(b"mock")
  (checkpoint / "hpt_tokenizer/KaiHandSimTaxelV1_matrix_7_5_3.safetensors").write_bytes(
    b"mock"
  )
  (checkpoint / "model_config.json").write_text(
    json.dumps(
      {
        "action_dim": 120,
        "action_horizon": 32,
        "discrete_state_input": False,
        "state_input_mode": "adarms",
        "use_tactile_input": True,
        "disable_history": True,
      }
    ),
    encoding="utf-8",
  )
  (checkpoint / "train_config.json").write_text(
    json.dumps(
      {
        "single_arm_action_rep_dim": 48,
        "reserved_action_dim": 15,
        "disable_history": True,
        "action_down_sample_steps": 1,
        "proprioception_pose_rep": "relative",
        "action_pose_rep": "relative",
        "proprioception_joint_rep": "abs",
        "action_joint_rep": "mix",
        "independent_norm_mode": "all",
        "norm_time_dim": False,
        "norm_type": "zscore",
        "norm_image_tactile_mode": "channel_wise",
      }
    ),
    encoding="utf-8",
  )
  (checkpoint / "tactile_input_config_file.json").write_text(
    json.dumps(
      {
        "KaiHandSimPokerDraw": {
          "right_tactile_fingers": {
            "shape": [1, 5, 7, 5, 3],
            "function_areas": [0, 1, 2, 3, 4],
            "sensor": "KaiHandSimTaxelV1",
            "type": "matrix",
          }
        }
      }
    ),
    encoding="utf-8",
  )
  (checkpoint / "normalization/KaiHandSimPokerDraw/independent_norm_stats_all_t0_zscore.json").write_text(
    json.dumps(
      {
        "norm_time_dim": False,
        "norm_type": "zscore",
        "params": {
          "state": {"mean": [0.0], "std": [1.0]},
          "actions": {"mean": [0.0], "std": [1.0]},
          "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": {"mean": [0.0], "std": [1.0]}},
        },
        "data_dim": {
          "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": 3},
          "actions": 120,
          "state": 120,
        },
      }
    ),
    encoding="utf-8",
  )
  project = tmp_path / "ftp1-policy"
  (project / "src").mkdir(parents=True)
  python = tmp_path / "python"
  python.write_text("mock", encoding="utf-8")
  tokenizer = tmp_path / "paligemma_tokenizer.model"
  tokenizer.write_bytes(b"frozen tokenizer")
  return {
    "schema": "poker_tactile_deployment_v1",
    "task": "poker-draw",
    "model_family": "ftp1",
    "deployment_id": "ftp1-test",
    "checkpoint_sha256": "a" * 64,
    "prediction_horizon": 32,
    "action_start_index": 1,
    "execute_steps": 8,
    "model_action_dim": 120,
    "action_dim": 27,
    "control_hz": 30,
    "joint_names": list(RIGHT_JOINT_NAMES),
    "source": {
      "checkpoint": str(checkpoint),
      "domain_name": "KaiHandSimPokerDraw",
      "tokenizer_model": str(tokenizer),
    },
    "runtime": {
      "project_root": str(project),
      "python": str(python),
      "device": "cpu",
      "num_inference_steps": 7,
    },
    "files": {str(tokenizer.resolve()): "mock-sha256"},
  }


def task_manifest(tmp_path, recipe="arm_relative_hand_absolute"):
  payload = manifest(tmp_path)
  checkpoint = Path(payload["source"]["checkpoint"])
  task = "bulb-screw"
  contract = task_contract(task)

  old_norm = (
    checkpoint
    / "normalization/KaiHandSimPokerDraw/independent_norm_stats_all_t0_zscore.json"
  )
  norm = (
    checkpoint
    / "normalization"
    / contract["domain_name"]
    / "independent_norm_stats_all_t0_zscore.json"
  )
  norm.parent.mkdir(parents=True)
  norm.write_text(old_norm.read_text(encoding="utf-8"), encoding="utf-8")
  (checkpoint / "tactile_input_config_file.json").write_text(
    json.dumps(
      {
        contract["domain_name"]: {
          "right_tactile_fingers": {
            "shape": [1, 5, 7, 5, 3],
            "function_areas": [0, 1, 2, 3, 4],
            "sensor": "KaiHandSimTaxelV1",
            "type": "matrix",
          }
        }
      }
    ),
    encoding="utf-8",
  )

  corpus = tmp_path / "bulb-screw-corpus"
  corpus.mkdir()
  conversion = corpus / "conversion_manifest.json"
  conversion.write_text(
    json.dumps(
      {
        "task": task,
        "instruction": contract["instruction"],
        "fps": 30,
        "ftp1_action_horizon": 32,
        "action_supervision": {
          "contract": "ftp1-native-future-measured-state-trajectory-v1",
          "source": (
            "measured wrist poses and hand joint positions; actions are built "
            "by the native FTP-1 loader"
          ),
          "controller_targets_used": False,
          "raw_controller_targets": (
            "may be present in Raw; not used as this export's action supervision"
          ),
          "pi05_recorded_controller_target_equivalent": False,
          "online_deployment_ready": False,
        },
      }
    ),
    encoding="utf-8",
  )
  dataset_config = tmp_path / "bulb-screw-dataset-config.json"
  dataset_config.write_text(
    json.dumps(
      {
        "datasets": [
          {
            "name": contract["domain_name"],
            "path": str(corpus),
            "enabled": True,
          }
        ]
      }
    ),
    encoding="utf-8",
  )
  train_path = checkpoint / "train_config.json"
  train = json.loads(train_path.read_text(encoding="utf-8"))
  train.update(dataset_config_path=str(dataset_config), action_joint_rep=recipe)
  train_path.write_text(json.dumps(train), encoding="utf-8")
  binding = ftp1_training_binding(checkpoint, task)

  payload.update(
    schema="tactile_deployment_v2",
    task=task,
    instruction=contract["instruction"],
    diagnostic_only=True,
    formal_metrics_valid=False,
    required_action_interpolation="zoh",
    action_representation="absolute-right-arm7-hand20-joints",
    controller_version="shared-tactile-direct-joints-v1",
    codec_version="ftp1-kaihand-task-v2",
    native_evaluation_protocol=(
      "TaskPolicyAdapter (current task completion semantics)"
    ),
    training_protocol_transfer=(
      "recorded training instruction is preserved; native simulator "
      "completion remains authoritative"
    ),
    observation_contract={
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": True,
      "tactile_shape": [5, 3, 7, 5],
      "tactile_unit": "N",
      **{
        key: contract[key]
        for key in (
          "tactile_target_body",
          "tactile_target_geometries",
          "tactile_provider",
        )
      },
    },
  )
  payload["source"].update(
    domain_name=contract["domain_name"],
    action_joint_rep=recipe,
    training_binding=binding,
  )
  payload["files"].update(
    {
      str(dataset_config.resolve()): "mock-dataset-config-sha256",
      str(conversion.resolve()): "mock-conversion-manifest-sha256",
    }
  )
  return payload


def request():
  return {
    "state": np.linspace(-0.3, 0.3, 27, dtype=np.float32),
    "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
    "right_wrist_image": np.ones((240, 320, 3), dtype=np.uint8),
    "tactile_map_raw": np.arange(5 * 3 * 7 * 5, dtype=np.float32).reshape(5, 3, 7, 5),
    "world_T_head": np.eye(4, dtype=np.float64),
    "world_T_wrist": np.eye(4, dtype=np.float64),
    "prompt": "Draw the face-down card.",
    "seed": 11,
    "index": 4,
    "execute_steps": 8,
  }


class FakeWrapper:
  instances = []

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.requests = []
    self.__class__.instances.append(self)
    self.model_config = SimpleNamespace(use_tactile_input=True)
    self.domain_norm_stats = {
      "params": {"state": {}, "actions": {}, "tactile": {}}
    }
    self.tactile_key_type_map = {"right_tactile_fingers": "matrix"}
    self.tokenizer = object()

  def get_action_horizon(self):
    return 32

  def get_action_dim(self):
    return 120

  def get_state_dim(self):
    return 120

  def infer(self, **kwargs):
    self.requests.append(kwargs)
    actions = np.zeros((32, 120), dtype=np.float32)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    actions[:, 9:16] = 0.2
    actions[:, 16 + np.asarray(RIGHT_HAND_FAAS_INDICES)] = -0.1
    actions[:, 48:] = 1234.0
    return actions


def build_backend(monkeypatch, backend_module, tmp_path, payload=None):
  FakeWrapper.instances.clear()
  monkeypatch.setattr(
    backend_module.FTP1PolicyBackend,
    "_import_wrapper_class",
    lambda _self: FakeWrapper,
  )
  monkeypatch.setattr(
    backend_module,
    "_seeded_torch_rng",
    lambda _seed, _device: contextlib.nullcontext(),
  )
  monkeypatch.setattr(
    backend_module.FTP1PolicyBackend,
    "_frozen_tokenizer_download",
    lambda _self: contextlib.nullcontext(),
  )
  monkeypatch.setattr(
    backend_module.FTP1PolicyBackend,
    "_batch_safe_image_resize",
    lambda _self: contextlib.nullcontext(),
  )
  return backend_module.FTP1PolicyBackend(payload or manifest(tmp_path))


def test_backend_loads_wrapper_with_checkpoint_normalization_once(
  monkeypatch, backend_module, tmp_path
):
  backend = build_backend(monkeypatch, backend_module, tmp_path)
  wrapper = FakeWrapper.instances[-1]
  assert wrapper.kwargs == {
    "checkpoint_dir": str((tmp_path / "24999").resolve()),
    "domain_name": "KaiHandSimPokerDraw",
    "device": "cpu",
    "num_inference_steps": 7,
    "skip_normalization": False,
  }

  metadata = backend.metadata()
  assert metadata["prediction_horizon"] == 32
  assert metadata["action_start_index"] == 1
  assert metadata["max_execute_steps"] == 31
  assert metadata["model_action_dim"] == 120
  assert metadata["action_dim"] == 27
  assert metadata["observation_contract"]["tactile_sent_to_model"] is True
  assert metadata["action_representation"]["normalization"] == "checkpoint_wrapper_once"


def test_infer_passes_raw_canonical_inputs_and_decodes_absolute_joints(
  monkeypatch, backend_module, tmp_path
):
  backend = build_backend(monkeypatch, backend_module, tmp_path)
  payload = request()
  response = backend.infer(payload)
  call = FakeWrapper.instances[-1].requests[-1]

  assert set(call["images"]) == {"camera_ego_rgb_0", "right_wrist_camera_rgb_0"}
  assert call["images"]["camera_ego_rgb_0"] is payload["head_image"]
  assert call["state"].shape == (1, 120)
  assert call["state"].dtype == np.float32
  np.testing.assert_array_equal(call["state"][0, 9:16], payload["state"][:7])
  assert call["tactiles"]["right_tactile_fingers"].shape == (1, 5, 7, 5, 3)
  np.testing.assert_array_equal(
    call["tactiles"]["right_tactile_fingers"][0, 2, :, :, 1],
    payload["tactile_map_raw"][2, 1],
  )
  assert call["tactile_function_areas"] == {
    "right_tactile_fingers": [0, 1, 2, 3, 4]
  }
  assert call["tactile_sensors"] == {
    "right_tactile_fingers": "KaiHandSimTaxelV1"
  }
  assert call["action_mask"].shape == (120,)
  assert int(call["action_mask"].sum()) == 36

  assert response["actions"].shape == (32, 27)
  assert response["actions"].dtype == np.float32
  np.testing.assert_allclose(
    response["actions"][:, :7],
    np.broadcast_to(payload["state"][:7] + 0.2, (32, 7)),
  )
  np.testing.assert_allclose(
    response["actions"][:, 7:],
    np.broadcast_to(payload["state"][7:] - 0.1, (32, 20)),
  )
  diagnostics = response["diagnostics"]
  assert diagnostics["action_start_index"] == 1
  assert diagnostics["execute_steps"] == 8
  assert diagnostics["inference_seed"] == 11 * 1_000_003 + 4
  assert diagnostics["normalization_applied_by_wrapper"] is True
  assert diagnostics["inactive_model_dimensions_executed"] is False
  assert diagnostics["model_action_shape"] == [32, 120]
  assert all(value >= 0.0 for value in diagnostics["timing_s"].values())


def test_task_scoped_backend_uses_training_bound_absolute_hand_recipe(
  monkeypatch, backend_module, tmp_path
):
  payload = task_manifest(tmp_path)
  backend = build_backend(monkeypatch, backend_module, tmp_path, payload)
  metadata = backend.metadata()

  assert metadata["schema"] == "tactile_deployment_v2"
  assert metadata["task"] == "bulb-screw"
  assert metadata["domain_name"] == "KaiHandSimBulbScrew"
  assert metadata["instruction"] == task_contract("bulb-screw")["instruction"]
  assert metadata["action_representation"]["action_joint_rep"] == (
    "arm_relative_hand_absolute"
  )
  assert metadata["protocol_transfer"]["diagnostic_only"] is True
  assert metadata["protocol_transfer"]["formal_metrics_valid"] is False
  assert metadata["observation_contract"]["tactile_target_body"] == "bulb"

  inference_request = request()
  inference_request["prompt"] = task_contract("bulb-screw")["instruction"]
  response = backend.infer(inference_request)
  np.testing.assert_allclose(
    response["actions"][:, :7],
    np.broadcast_to(inference_request["state"][:7] + 0.2, (32, 7)),
  )
  np.testing.assert_allclose(response["actions"][:, 7:], -0.1)
  diagnostics = response["diagnostics"]
  assert diagnostics["task"] == "bulb-screw"
  assert diagnostics["domain_name"] == "KaiHandSimBulbScrew"
  assert diagnostics["action_joint_rep"] == "arm_relative_hand_absolute"
  assert diagnostics["prompt_matches_recorded_instruction"] is True
  assert diagnostics["diagnostic_only"] is True
  assert diagnostics["formal_metrics_valid"] is False


@pytest.mark.parametrize(
  ("corrupt", "match"),
  [
    (
      lambda payload: payload["source"].update(domain_name="KaiHandSimPokerDraw"),
      "source.domain_name",
    ),
    (
      lambda payload: payload["source"].update(action_joint_rep="mix"),
      "differs from training_binding",
    ),
    (
      lambda payload: payload["source"]["training_binding"].update(
        dataset_root="/wrong/corpus"
      ),
      "does not match checkpoint provenance",
    ),
    (
      lambda payload: payload.update(instruction="Complete the newer protocol."),
      "recorded FTP-1 training instruction",
    ),
    (
      lambda payload: payload["files"].pop(
        payload["source"]["training_binding"]["conversion_manifest_path"]
      ),
      "frozen tokenizer/training provenance",
    ),
  ],
)
def test_task_scoped_manifest_rejects_cross_task_or_relabelled_assets(
  monkeypatch, backend_module, tmp_path, corrupt, match
):
  payload = task_manifest(tmp_path)
  corrupt(payload)
  with pytest.raises(ValueError, match=match):
    build_backend(monkeypatch, backend_module, tmp_path, payload)


def test_task_scoped_request_rejects_native_completion_prompt_relabel(
  monkeypatch, backend_module, tmp_path
):
  payload = task_manifest(tmp_path)
  backend = build_backend(monkeypatch, backend_module, tmp_path, payload)
  inference_request = request()
  inference_request["prompt"] = "Screw the bulb until the native success predicate passes."
  with pytest.raises(ValueError, match="recorded FTP-1 training instruction"):
    backend.infer(inference_request)


@pytest.mark.parametrize("corruption", ("missing", "wrong"))
def test_nonpoker_training_binding_rejects_unproven_action_supervision(
  tmp_path, corruption
):
  payload = task_manifest(tmp_path)
  conversion = Path(
    payload["source"]["training_binding"]["conversion_manifest_path"]
  )
  document = json.loads(conversion.read_text(encoding="utf-8"))
  if corruption == "missing":
    document.pop("action_supervision")
  else:
    document["action_supervision"]["controller_targets_used"] = True
  conversion.write_text(json.dumps(document), encoding="utf-8")

  with pytest.raises(ValueError, match="action_supervision"):
    ftp1_training_binding(payload["source"]["checkpoint"], payload["task"])


@pytest.mark.parametrize(
  ("field", "value", "match"),
  [
    ("prediction_horizon", 50, "prediction_horizon"),
    ("action_start_index", 0, "action_start_index"),
    ("model_action_dim", 27, "model_action_dim"),
    ("model_family", "pi0.5", "model_family"),
  ],
)
def test_manifest_contract_fails_closed(
  monkeypatch, backend_module, tmp_path, field, value, match
):
  payload = manifest(tmp_path)
  payload[field] = value
  with pytest.raises(ValueError, match=match):
    build_backend(monkeypatch, backend_module, tmp_path, payload)


def test_request_contract_rejects_off_by_one_execution_and_shape_drift(
  monkeypatch, backend_module, tmp_path
):
  backend = build_backend(monkeypatch, backend_module, tmp_path)
  payload = request()
  payload["execute_steps"] = 32
  with pytest.raises(ValueError, match=r"\[1,31\]"):
    backend.infer(payload)

  payload = request()
  payload["tactile_map_raw"] = np.zeros((5, 7, 5, 3), dtype=np.float32)
  with pytest.raises(ValueError, match="tactile_map_raw must have shape"):
    backend.infer(payload)

  payload = request()
  payload["extra"] = True
  with pytest.raises(ValueError, match=r"unknown=\['extra'\]"):
    backend.infer(payload)


def test_checkpoint_contract_rejects_same_shape_with_different_semantics(
  monkeypatch, backend_module, tmp_path
):
  payload = manifest(tmp_path)
  train_path = Path(payload["source"]["checkpoint"]) / "train_config.json"
  train = json.loads(train_path.read_text(encoding="utf-8"))
  train["action_joint_rep"] = "absolute"
  train_path.write_text(json.dumps(train), encoding="utf-8")
  with pytest.raises(ValueError, match="action_joint_rep"):
    build_backend(monkeypatch, backend_module, tmp_path, payload)


def test_frozen_tokenizer_binding_allows_only_declared_paligemma_asset(
  monkeypatch, backend_module, tmp_path
):
  tokenizer = tmp_path / "paligemma_tokenizer.model"
  tokenizer.write_bytes(b"frozen")
  original = object()
  download = SimpleNamespace(maybe_download=original)
  real_import = backend_module.importlib.import_module

  def import_module(name):
    if name == "openpi.shared.download":
      return download
    return real_import(name)

  monkeypatch.setattr(backend_module.importlib, "import_module", import_module)
  backend = backend_module.FTP1PolicyBackend.__new__(backend_module.FTP1PolicyBackend)
  backend._tokenizer_model = tokenizer

  with backend._frozen_tokenizer_download():
    assert download.maybe_download(
      "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    ) == tokenizer
    with pytest.raises(RuntimeError, match="undeclared download"):
      download.maybe_download("gs://unexpected/weights")
  assert download.maybe_download is original


def test_image_resize_workaround_preserves_and_restores_batch_axis(
  monkeypatch, backend_module
):
  class FakeTensor:
    def __init__(self, ndim):
      self.ndim = ndim

    def unsqueeze(self, axis):
      assert axis == 0
      return FakeTensor(self.ndim + 1)

  def original(_images, *_args, **_kwargs):
    return FakeTensor(3)

  image_tools = SimpleNamespace(resize_with_pad_torch=original)
  real_import = backend_module.importlib.import_module

  def import_module(name):
    if name == "openpi.shared.image_tools":
      return image_tools
    return real_import(name)

  monkeypatch.setattr(backend_module.importlib, "import_module", import_module)
  backend = backend_module.FTP1PolicyBackend.__new__(backend_module.FTP1PolicyBackend)

  with backend._batch_safe_image_resize():
    resized = image_tools.resize_with_pad_torch(FakeTensor(4), 224, 224)
    assert resized.ndim == 4
    untouched = image_tools.resize_with_pad_torch(FakeTensor(3), 224, 224)
    assert untouched.ndim == 3
  assert image_tools.resize_with_pad_torch is original


def test_infer_accepts_runner_aliases_and_redundant_images(
  monkeypatch, backend_module, tmp_path
):
  backend = build_backend(monkeypatch, backend_module, tmp_path)
  payload = request()
  payload.update(
    episode_seed=payload["seed"],
    request_index=payload["index"],
    images={
      "head": payload["head_image"].copy(),
      "right_wrist": payload["right_wrist_image"].copy(),
    },
    sim_time=0.25,
  )
  response = backend.infer(payload)
  assert response["diagnostics"]["seed"] == payload["episode_seed"]
  assert response["diagnostics"]["index"] == payload["request_index"]

  payload["request_index"] += 1
  with pytest.raises(ValueError, match="disagree"):
    backend.infer(payload)
