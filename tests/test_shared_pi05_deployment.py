from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import pytest
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
  path = ROOT / "scripts/workcell" / name
  spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


@dataclasses.dataclass(frozen=True)
class Assets:
  asset_id: str = "kaihand"


@dataclasses.dataclass(frozen=True)
class LeRobotKaiHandDataConfig:
  assets: Assets = dataclasses.field(default_factory=Assets)
  use_delta_arm_actions: bool = True


@dataclasses.dataclass(frozen=True)
class Model:
  action_dim: int = 32
  action_horizon: int = 30


@dataclasses.dataclass(frozen=True)
class Config:
  data: LeRobotKaiHandDataConfig = dataclasses.field(
    default_factory=LeRobotKaiHandDataConfig
  )
  model: Model = dataclasses.field(default_factory=Model)
  policy_metadata: dict = dataclasses.field(default_factory=dict)


def policy_metadata() -> dict:
  return {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": 30,
    "control_hz": 30,
    "suggested_replan_steps": 8,
    "joint_names": [f"joint_{index}" for index in range(27)],
    "action_semantics": {"policy_output_is_absolute": True},
    "image_keys": [
      "observation.images.head",
      "observation.images.right_wrist",
    ],
  }


def manifest(checkpoint: Path) -> dict:
  normalizer = checkpoint / "assets/normalizer/norm_stats.json"
  normalizer.parent.mkdir(parents=True)
  normalizer.write_text("{}", encoding="utf-8")
  return {
    "schema": "shared_task_pi05_deployment_v1",
    "task": "bulb-screw",
    "deployment_id": "deployment-id",
    "model_family": "pi0.5",
    "model_config_name": "pi05_kaihand",
    "normalizer_asset_id": "normalizer",
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": "checkpoint-hash",
    "checkpoint_step": 20000,
    "prediction_horizon": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "suggested_execute_steps": 8,
    "joint_names": [f"joint_{index}" for index in range(27)],
    "instruction": TASK_INSTRUCTIONS["bulb-screw"],
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"policy_output_is_absolute": True},
    "checkpoint_files": [
      {
        "path": "assets/normalizer/norm_stats.json",
        "size": normalizer.stat().st_size,
        "sha256": "not-checked-at-server-start",
      }
    ],
  }


def test_prepare_contract_matches_shared_task_runner():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  server = load_script("serve_shared_task_pi05_policy.py")
  assert prepare.CONFIG_NAME == "pi05_kaihand"
  assert prepare.TASK_INSTRUCTIONS == TASK_INSTRUCTIONS
  assert server.TASK_INSTRUCTIONS == TASK_INSTRUCTIONS
  assert (
    prepare.validate_train_config(Config(policy_metadata=policy_metadata()))
    == policy_metadata()
  )


def test_shared_batch_environment_includes_openpi_client(monkeypatch):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

  environment = batch._environment()
  pythonpath = environment["PYTHONPATH"].split(os.pathsep)

  assert pythonpath == [
    str(ROOT / "src"),
    str(ROOT.parent / "openpi/packages/openpi-client/src"),
    "/existing/pythonpath",
  ]
  assert Path(environment["__EGL_VENDOR_LIBRARY_FILENAMES"]).is_file()


def test_prepare_rejects_wrong_camera_contract():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  metadata = policy_metadata()
  metadata["image_keys"] = ["observation.images.head"]
  with pytest.raises(RuntimeError, match="config contract mismatch"):
    prepare.validate_train_config(Config(policy_metadata=metadata))


def test_server_loads_bulb_manifest_and_binds_normalizer(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  resolved, loaded = server.load_manifest(path)
  config = Config(policy_metadata=policy_metadata())
  bound = server.bind_checkpoint_assets(config, loaded)
  metadata = server.validate_runtime_config(bound, loaded)
  advertised = server.deployment_metadata(metadata, resolved, loaded)

  assert bound.data.assets.asset_id == "normalizer"
  assert config.data.assets.asset_id == "kaihand"
  assert advertised["evaluation_task"] == "bulb-screw"
  assert advertised["checkpoint_path"] == str(checkpoint)
  assert advertised["prediction_horizon"] == 30
  assert advertised["observation_contract"]["cameras"] == [
    "head",
    "right_wrist",
  ]


def test_server_rejects_usb_manifest(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  payload.update(schema="usb_pi05_deployment_v1", task="usb-insert")
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(RuntimeError, match="expected shared_task"):
    server.load_manifest(path)


def test_server_rejects_task_prompt_drift(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  payload["instruction"] = "Use a different task prompt."
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(RuntimeError, match="frozen task prompt"):
    server.load_manifest(path)


def test_server_rejects_checkpoint_path_traversal(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  payload["checkpoint_files"] = [{"path": "../outside", "size": 0, "sha256": "unused"}]
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(RuntimeError, match="unsafe checkpoint file path"):
    server.load_manifest(path)


@pytest.mark.parametrize("asset_id", [None, "", "../normalizer", "a/b", ".", ".."])
def test_server_rejects_invalid_normalizer_asset_id(tmp_path, asset_id):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  payload["normalizer_asset_id"] = asset_id
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(RuntimeError, match="one directory name"):
    server.load_manifest(path)
