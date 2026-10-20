from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  _canonical_hash,
  _normalizer_semantic_sha256,
  verify_pi05_deployment,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
  path = ROOT / "scripts/workcell" / name
  spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def write_json(path: Path, value: dict) -> bytes:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(value, sort_keys=True, indent=2).encode()
  path.write_bytes(payload)
  return payload


def identity(payload: bytes) -> dict:
  return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@dataclasses.dataclass(frozen=True)
class Assets:
  asset_id: str = "kaihand"
  assets_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class LeRobotKaiHandDataConfig:
  assets: Assets = dataclasses.field(default_factory=Assets)
  use_delta_arm_actions: bool = True
  use_quantile_norm: bool = True

  def create(self, _assets_dirs, _model):
    return types.SimpleNamespace(use_quantile_norm=self.use_quantile_norm)


@dataclasses.dataclass(frozen=True)
class Model:
  action_dim: int = 32
  action_horizon: int = 50


@dataclasses.dataclass(frozen=True)
class Config:
  name: str = "pi05_kaihand_h50"
  model: Model = dataclasses.field(default_factory=Model)
  data: LeRobotKaiHandDataConfig = dataclasses.field(
    default_factory=LeRobotKaiHandDataConfig
  )
  policy_metadata: dict = dataclasses.field(default_factory=dict)
  assets_dirs: Path = Path("fixture-assets-never-read")


def bound_checkpoint(tmp_path: Path, *, horizon=50):
  prepare = load_script("prepare_poker_pi05_deployment.py")
  checkpoint = tmp_path / "24999"
  (checkpoint / "params").mkdir(parents=True)
  (checkpoint / "params/fixture-model").write_bytes(b"mock parameter bytes")
  (checkpoint / "_CHECKPOINT_METADATA").write_bytes(b"committed")
  asset_id = "normalizer_h50" if horizon == 50 else "normalizer"
  stats_path = checkpoint / f"assets/{asset_id}/norm_stats.json"
  stats_bytes = write_json(
    stats_path,
    {
      "norm_stats": {
        key: {statistic: [0.0] * 27 for statistic in ("mean", "std", "q01", "q99")}
        for key in ("actions", "state")
      }
    },
  )
  contract = {
    "id": "legacy-measured-arm-hybrid-v1",
    "legacy": True,
    "dimensions": 27,
    "joint_order": [f"joint_{index}" for index in range(27)],
    "arm_0_6": "measured fixture arm state",
    "hand_7_26": "fixture commanded hand endpoint",
    "evaluation_execution": {
      "endpoint_rate_hz": 30,
      "physical_substep_interpolation_required": "linear",
      "zero_order_hold_equivalent": False,
    },
  }
  row_identity = {"size_bytes": 10, "sha256": "a" * 64}
  metadata_names = (
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
    "meta/info.json",
    "meta/kaihand_source_episodes.jsonl",
    "meta/tasks.jsonl",
  )
  snapshot = {
    "schema": "kaihand_fast_pi05_video_v2",
    "task": "poker-draw",
    "instruction": prepare.EXPECTED_TASK,
    "repo_id": "poker-fixture",
    "cameras": ["head", "right_wrist"],
    "action_contract": contract,
    "metadata_artifacts": {name: row_identity for name in metadata_names},
    "total_episodes": 1,
    "episodes": [
      {
        "output_episode_index": 0,
        "action_contract": contract["id"],
        "artifacts": {
          name: row_identity
          for name in (
            "data.parquet",
            "stats.json",
            "head.mp4",
            "right_wrist.mp4",
          )
        },
      }
    ],
  }
  snapshot_bytes = write_json(
    checkpoint / "assets/conversion_manifest.snapshot.json", snapshot
  )
  conversion = {
    "filename": "kaihand_fast_pi05_conversion.json",
    "schema": snapshot["schema"],
    **identity(snapshot_bytes),
  }
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    _fast_snapshot_artifact_aggregate,
  )

  dataset_root = str(tmp_path / "dataset-never-read")
  stats_record = {
    "schema": "kaihand_norm_stats_provenance_v1",
    "dataset_root": dataset_root,
    "conversion_manifest": {
      key: conversion[key] for key in ("filename", "sha256", "size_bytes")
    },
    "parameters": {"prefix_frames": 0, "action_horizon": horizon, "arm_dim": 7},
    "norm_stats": {"filename": "norm_stats.json", **identity(stats_bytes)},
  }
  sidecar_bytes = json.dumps(stats_record, sort_keys=True).encode()
  record = {
    "schema": "kaihand_training_provenance_v1",
    "binding_status": "bound",
    "task": "poker-draw",
    "config_name": "pi05_kaihand_h50" if horizon == 50 else "pi05_kaihand",
    "source_environment": None,
    "conversion_manifest": conversion,
    "action_contract": contract,
    "action_contract_sha256": _canonical_hash(contract),
    "dataset": {
      "root": dataset_root,
      "repo_id": "poker-fixture",
      "manifest_repo_id": "poker-fixture",
      "artifacts": _fast_snapshot_artifact_aggregate(snapshot),
    },
    "training_action_transform": {
      "action_dim": 27,
      "model_action_dim": 32,
      "action_horizon": horizon,
      "arm_delta_mask": [True] * 7 + [False] * 20,
      "skip_episode_prefix_frames": 0,
    },
    "normalizer": {
      "asset_id": asset_id,
      "keys": ["actions", "state"],
      "use_quantile_norm": True,
      "semantic_sha256": _normalizer_semantic_sha256(stats_path),
      "stats_provenance": {
        "filename": "norm_stats.provenance.json",
        **identity(sidecar_bytes),
        "record": stats_record,
      },
    },
  }
  write_json(checkpoint / "assets/training_provenance.json", record)
  return prepare, checkpoint, record


def mock_openpi(monkeypatch, record):
  transform = record["training_action_transform"]
  contract = record["action_contract"]
  metadata = {
    "action_dim": 27,
    "model_action_dim": 32,
    "action_horizon": transform["action_horizon"],
    "control_hz": 30,
    "suggested_replan_steps": 8,
    "image_keys": ["observation.images.head", "observation.images.right_wrist"],
    "joint_names": contract["joint_order"],
    "action_semantics": {
      "contract_id": contract["id"],
      "arm_0_6": contract["arm_0_6"],
      "hand_7_26": contract["hand_7_26"],
      "policy_output_is_absolute": True,
    },
  }
  config = Config(
    name=record["config_name"],
    model=Model(action_horizon=transform["action_horizon"]),
    policy_metadata=metadata,
  )
  names = (
    "openpi",
    "openpi.training",
    "openpi.training.config",
    "openpi.policies",
    "openpi.policies.policy_config",
    "openpi.serving",
    "openpi.serving.websocket_policy_server",
  )
  modules = {name: types.ModuleType(name) for name in names}
  for name, module in modules.items():
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
      parent, child = name.rsplit(".", 1)
      setattr(modules[parent], child, module)
  modules["openpi.training.config"].get_config = (
    lambda name: config if name == config.name else pytest.fail(name)
  )
  return config, modules


def prepared_manifest(tmp_path, monkeypatch, *, horizon=50):
  prepare, checkpoint, record = bound_checkpoint(tmp_path, horizon=horizon)
  config, modules = mock_openpi(monkeypatch, record)
  openpi_root = tmp_path / "openpi"
  for relative in (
    "src/openpi/training/config.py",
    "packages/openpi-client/src/openpi_client/__init__.py",
  ):
    path = openpi_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture source\n", encoding="utf-8")
  evaluation_root = tmp_path / "simulator"
  source = evaluation_root / "src/kaihand_tactile_env/fixture.py"
  source.parent.mkdir(parents=True)
  source.write_text("fixture evaluator\n", encoding="utf-8")
  monkeypatch.setattr(prepare, "ROOT", evaluation_root)
  monkeypatch.setattr(prepare, "evaluation_source_paths", lambda: [source])
  output = tmp_path / "deployment"
  assert (
    prepare.main(
      [
        "--checkpoint",
        str(checkpoint),
        "--openpi-root",
        str(openpi_root),
        "--output-dir",
        str(output),
      ]
    )
    == 0
  )
  path = output / "deployment_manifest.json"
  manifest = json.loads(path.read_bytes())
  return prepare, path, manifest, config, modules, evaluation_root


def test_bound_h50_uses_checkpoint_specific_config_and_normalizer(tmp_path):
  prepare, checkpoint, _ = bound_checkpoint(tmp_path)
  config_name, asset_id, binding = prepare.checkpoint_contract(checkpoint)
  assert config_name == "pi05_kaihand_h50"
  assert asset_id == "normalizer_h50"
  assert binding["record"]["training_action_transform"]["action_horizon"] == 50


@pytest.mark.parametrize(
  "mutate,match",
  [
    (lambda r: r.update(task="install-ram"), "task differs"),
    (lambda r: r.update(config_name="pi05_kaihand"), "transform mismatch"),
    (lambda r: r["normalizer"].update(semantic_sha256="b" * 64), "semantic SHA"),
    (lambda r: r["normalizer"].update(asset_id="../normalizer"), "unsafe"),
    (
      lambda r: r["normalizer"]["stats_provenance"]["record"]["parameters"].update(
        action_horizon=30
      ),
      "statistics are not bound",
    ),
  ],
)
def test_checkpoint_rejects_wrong_task_config_asset_or_stats(tmp_path, mutate, match):
  prepare, checkpoint, record = bound_checkpoint(tmp_path)
  mutate(record)
  write_json(checkpoint / "assets/training_provenance.json", record)
  with pytest.raises(RuntimeError, match=match):
    prepare.checkpoint_contract(checkpoint)


def test_h50_dry_run_is_readonly(tmp_path, monkeypatch, capsys):
  prepare, checkpoint, record = bound_checkpoint(tmp_path)
  mock_openpi(monkeypatch, record)
  output = tmp_path / "never-created"
  monkeypatch.setattr(
    prepare, "checkpoint_identity", lambda *_: pytest.fail("dry-run hashed weights")
  )
  assert (
    prepare.main(
      [
        "--checkpoint",
        str(checkpoint),
        "--openpi-root",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--dry-run",
      ]
    )
    == 0
  )
  result = json.loads(capsys.readouterr().out)
  assert result["valid"] and result["prediction_horizon"] == 50
  assert result["normalizer_asset_id"] == "normalizer_h50"
  assert not output.exists()


def test_bound_h50_manifest_authenticates_sources_and_provenance(tmp_path, monkeypatch):
  prepare, _, manifest, _, _, evaluation_root = prepared_manifest(tmp_path, monkeypatch)
  assert manifest["schema"] == prepare.SCHEMA_V2
  assert manifest["deployment_id"] == _canonical_hash(
    prepare.deployment_identity(manifest)
  )
  assert manifest["diagnostic_only"] is True
  assert manifest["formal_metrics_valid"] is False
  assert manifest["required_action_interpolation"] == "linear"
  verify_pi05_deployment(manifest, evaluation_root=evaluation_root)
  source = evaluation_root / manifest["evaluation_source_files"][0]["path"]
  source.write_text("changed evaluator\n", encoding="utf-8")
  with pytest.raises(RuntimeError, match="evaluation source content SHA"):
    verify_pi05_deployment(manifest, evaluation_root=evaluation_root)


@pytest.mark.parametrize(
  "field",
  [
    "normalizer_asset_id",
    "training_provenance",
    "instruction",
    "openpi_source_sha256",
    "evaluation_source_sha256",
    "formal_metrics_valid",
  ],
)
def test_manifest_new_identity_fields_reject_tampering(tmp_path, monkeypatch, field):
  prepare, _, manifest, _, _, _ = prepared_manifest(tmp_path, monkeypatch)
  manifest[field] = "changed"
  with pytest.raises(RuntimeError, match="authenticate"):
    prepare.validate_deployment_identity(manifest)


def test_server_warms_frozen_h50_before_listening_without_consuming_rng(
  tmp_path, monkeypatch
):
  _, path, manifest, original_config, modules, _ = prepared_manifest(
    tmp_path, monkeypatch
  )
  server = load_script("serve_poker_pi05_policy.py")
  events = []
  rng = object()

  class Policy:
    _rng = rng

    def infer(self, request):
      assert request["head_image"].shape == (240, 320, 3)
      assert request["state"].shape == (27,)
      self._rng = object()
      events.append("warmup")
      return {"actions": np.zeros((50, 27), dtype=np.float32)}

  policy = Policy()

  def create_policy(config, checkpoint, **kwargs):
    assert config.name == "pi05_kaihand_h50"
    assert config.data.assets.asset_id == "normalizer_h50"
    assert config.data.assets.assets_dir == str(Path(checkpoint) / "assets")
    assert kwargs["expected_use_quantile_norm"] is True
    events.append("created")
    return policy

  class Server:
    def __init__(self, **kwargs):
      metadata = kwargs["metadata"]
      assert metadata["model_config_name"] == "pi05_kaihand_h50"
      assert metadata["normalizer_asset_id"] == "normalizer_h50"
      assert metadata["training_provenance"] == manifest["training_provenance"]
      assert metadata["server_warmup"]["rng_restored"] is True
      assert metadata["required_action_interpolation"] == "linear"
      assert metadata["diagnostic_only"] is True
      assert metadata["formal_metrics_valid"] is False
      assert policy._rng is rng
      events.append("server")

    def serve_forever(self):
      events.append("listen")

  monkeypatch.setattr(
    server, "verify_pi05_deployment", lambda *_, **__: events.append("verified")
  )
  modules["openpi.policies.policy_config"].create_trained_policy = create_policy
  modules["openpi.serving.websocket_policy_server"].WebsocketPolicyServer = Server
  server.main(["--deployment-manifest", str(path)])
  assert events == ["verified", "created", "warmup", "server", "listen"]
  assert original_config.data.assets.asset_id == "kaihand"


def test_warmup_restores_rng_even_if_inference_fails():
  server = load_script("serve_poker_pi05_policy.py")
  rng = object()

  class Policy:
    _rng = rng

    def infer(self, _request):
      self._rng = object()
      raise RuntimeError("fixture inference failed")

  policy = Policy()
  manifest = {
    "prediction_horizon": 50,
    "action_dim": 27,
    "instruction": "fixture",
    "observation_contract": {"image_shape_hwc": [240, 320, 3]},
  }
  with pytest.raises(RuntimeError, match="inference failed"):
    server.warmup_policy(policy, manifest)
  assert policy._rng is rng


def test_historical_unbound_h30_checkpoint_stays_compatible(tmp_path, monkeypatch):
  prepare, checkpoint, record = bound_checkpoint(tmp_path, horizon=30)
  (checkpoint / "assets/training_provenance.json").unlink()
  (checkpoint / "assets/conversion_manifest.snapshot.json").unlink()
  assert prepare.checkpoint_contract(checkpoint) == ("pi05_kaihand", "normalizer", None)
  _, modules = mock_openpi(monkeypatch, record)
  openpi_root = tmp_path / "openpi"
  for relative in (
    "src/openpi/training/config.py",
    "src/openpi/policies/kaihand_policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/serving/websocket_policy_server.py",
  ):
    path = openpi_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture source\n", encoding="utf-8")
  output = tmp_path / "deployment"
  assert (
    prepare.main(
      [
        "--checkpoint",
        str(checkpoint),
        "--openpi-root",
        str(openpi_root),
        "--output-dir",
        str(output),
      ]
    )
    == 0
  )
  path = output / "deployment_manifest.json"
  manifest = json.loads(path.read_bytes())
  assert manifest["schema"] == prepare.SCHEMA_V1
  assert manifest["prediction_horizon"] == 30
  assert "training_provenance" not in manifest
  server = load_script("serve_poker_pi05_policy.py")
  assert server._load_manifest(path)[1]["model_config_name"] == "pi05_kaihand"
  assert "openpi.policies.policy_config" in modules


def test_runtime_config_rejects_horizon_joint_semantics_or_normalization_change(
  tmp_path, monkeypatch
):
  prepare, checkpoint, record = bound_checkpoint(tmp_path)
  config, _ = mock_openpi(monkeypatch, record)
  _, _, binding = prepare.checkpoint_contract(checkpoint)
  assert (
    prepare.validate_train_config(config, config.name, binding)
    == config.policy_metadata
  )
  with pytest.raises(RuntimeError, match="model horizon"):
    prepare.validate_train_config(
      dataclasses.replace(config, model=Model(action_horizon=30)), config.name, binding
    )
  metadata = dict(config.policy_metadata)
  metadata["joint_names"] = list(reversed(metadata["joint_names"]))
  with pytest.raises(RuntimeError, match="joint order"):
    prepare.validate_train_config(
      dataclasses.replace(config, policy_metadata=metadata), config.name, binding
    )
  metadata = dict(config.policy_metadata)
  metadata["action_semantics"] = {"policy_output_is_absolute": False}
  with pytest.raises(RuntimeError, match="action semantics"):
    prepare.validate_train_config(
      dataclasses.replace(config, policy_metadata=metadata), config.name, binding
    )
  with pytest.raises(RuntimeError, match="normalization mode"):
    prepare.validate_train_config(
      dataclasses.replace(
        config, data=dataclasses.replace(config.data, use_quantile_norm=False)
      ),
      config.name,
      binding,
    )
