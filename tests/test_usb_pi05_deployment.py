from __future__ import annotations

import builtins
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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
  asset_id: str


@dataclasses.dataclass(frozen=True)
class Data:
  assets: Assets


@dataclasses.dataclass(frozen=True)
class Config:
  data: Data


def test_prepare_uses_shared_kaihand_config_and_checkpoint_normalizer(tmp_path):
  prepare = load_script("prepare_usb_pi05_deployment.py")
  checkpoint = tmp_path / "25000"
  normalizer = checkpoint / "assets/normalizer/norm_stats.json"
  normalizer.parent.mkdir(parents=True)
  normalizer.write_text("{}", encoding="utf-8")

  assert prepare.CONFIG_NAME == "pi05_kaihand"
  assert prepare.normalizer_asset_id(checkpoint) == "normalizer"


def test_prepare_rejects_checkpoint_without_normalizer(tmp_path):
  prepare = load_script("prepare_usb_pi05_deployment.py")
  with pytest.raises(ValueError, match="normalization stats"):
    prepare.normalizer_asset_id(tmp_path / "25000")


def test_prepare_legacy_hash_exports_import_without_model_or_simulator(monkeypatch):
  real_import = builtins.__import__

  def import_without_project_dependencies(name, *args, **kwargs):
    if name == "openpi" or name.startswith(("openpi.", "kaihand_tactile_env")):
      pytest.fail("legacy hashing helpers must remain dependency-free")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_project_dependencies)
  prepare = load_script("prepare_usb_pi05_deployment.py")
  assert prepare.CONFIG_NAME == "pi05_kaihand"
  assert callable(prepare.checkpoint_identity)
  assert len(prepare._canonical_hash({"legacy": True})) == 64


def test_usb_pi05_batch_uses_tracked_egl_vendor_configuration():
  batch = load_script("evaluate_usb_pi05_policy_batch.py")
  assert batch.EGL_VENDOR == ROOT / "scripts/collect/nvidia_egl_vendor.json"
  assert batch.EGL_VENDOR.is_file()
  assert batch.DEFAULT_MAX_SIM_SECONDS == 90.0


@pytest.mark.parametrize(
  "script",
  ["evaluate_usb_policy_batch.py", "evaluate_usb_pi05_policy_batch.py"],
)
def test_usb_batches_reject_invalid_physics_and_aggregate_guard_events(script):
  batch = load_script(script)
  reason, valid = batch.classify(
    {
      "status": "error",
      "physics_valid": False,
      "error": "FatalSimulationError: simulation solver warning",
    }
  )
  assert (reason, valid) == ("fatal_physics", False)
  reason, valid = batch.classify(
    {
      "status": "success",
      "evaluation": {"success": True},
      "physics_valid": True,
      "record_error": "RuntimeError: encoder failed",
    }
  )
  assert (reason, valid) == ("recording_error", False)
  assert batch.classify(
    {"status": "success", "task_success": True, "physics_valid": True}
  ) == ("success", True)
  assert batch.guard_event_counts(
    [
      {
        "guard_events": [
          {"kind": "usb_socket_penetration", "count": 3},
          {"kind": "nonfinite_action", "count": 1},
        ]
      },
      {"guard_events": [{"kind": "usb_socket_penetration", "count": 2}]},
    ]
  ) == {"usb_socket_penetration": 5, "nonfinite_action": 1}


def test_server_binds_manifest_normalizer_to_runtime_config():
  server = load_script("serve_usb_pi05_policy.py")
  config = Config(data=Data(assets=Assets(asset_id="kaihand")))

  bound = server.bind_checkpoint_assets(config, {"normalizer_asset_id": "normalizer"})

  assert bound.data.assets.asset_id == "normalizer"
  assert config.data.assets.asset_id == "kaihand"


@pytest.mark.parametrize("asset_id", [None, "", "../normalizer", "a/b", ".", ".."])
def test_server_rejects_invalid_normalizer_asset_id(asset_id):
  server = load_script("serve_usb_pi05_policy.py")
  config = Config(data=Data(assets=Assets(asset_id="kaihand")))
  with pytest.raises(RuntimeError, match="one directory name"):
    server.bind_checkpoint_assets(config, {"normalizer_asset_id": asset_id})


SAFE_ASSET = "normalizer_h50_usb_insert_hand_tail_v1"


def _json_bytes(value):
  return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _file_identity(payload):
  return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _bound_checkpoint(tmp_path, monkeypatch, *, horizon=50, v2=True):
  prepare = load_script("prepare_usb_pi05_deployment.py")
  checkpoint = tmp_path / "25000"
  (checkpoint / "params").mkdir(parents=True)
  (checkpoint / "params/weights.bin").write_bytes(b"tiny fixture, not model weights")
  (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
  asset_id = SAFE_ASSET if v2 else ("normalizer_h50" if horizon == 50 else "normalizer")
  directory = checkpoint / "assets" / asset_id
  directory.mkdir(parents=True)
  stats = {
    "norm_stats": {
      key: {
        "mean": [0.0] * 27,
        "std": [1.0] * 27,
        "q01": [-1.0] * 27,
        "q99": [1.0] * 27,
      }
      for key in ("state", "actions")
    }
  }
  stats_bytes = _json_bytes(stats)
  (directory / "norm_stats.json").write_bytes(stats_bytes)
  contract = {
    "id": "legacy-measured-arm-hybrid-v1",
    "legacy": True,
    "dimensions": 27,
    "joint_order": [f"joint_{index}" for index in range(27)],
    "arm_0_6": "measured arm labels",
    "hand_7_26": "commanded hand labels",
    "evaluation_execution": {
      "endpoint_rate_hz": 30,
      "physical_substep_interpolation_required": "linear",
      "zero_order_hold_equivalent": False,
    },
  }
  # These paths deliberately do not exist: deployment must be portable and
  # validate the frozen snapshot without accessing the training/source dataset.
  dataset_root = "/detached/usb_insert/pi05/0921_200"
  source_root = "/detached/usb_insert/pi05/0920_200"
  artifact = {"size_bytes": 1, "sha256": "a" * 64}
  snapshot = {
    "schema_version": prepare.USB_MIGRATION_SCHEMA,
    "task": "usb_insert",
    "output_dir": dataset_root,
    "action_contract": contract,
    "lerobot": {"repo_id": "usb_insert_0921_200"},
    "lineage": {"source_dataset": source_root, "source_conversion_manifest": artifact},
    "metadata_artifacts": {
      name: artifact
      for name in (
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/kaihand_usb_source_snapshot.json",
        "meta/kaihand_usb_conversion_manifest.json",
        "meta/interpolation_manifest.json",
      )
    },
    "total_episodes": 1,
    "episodes": [
      {
        "output_episode_index": 0,
        "parquet_path": "data/chunk-000/episode_000000.parquet",
        "parquet_size_bytes": 1,
        "parquet_sha256": "a" * 64,
        "source_parquet": {
          "path": f"{source_root}/data/chunk-000/episode_000000.parquet",
          **artifact,
        },
      }
    ],
  }
  snapshot_bytes = _json_bytes(snapshot)
  (checkpoint / "assets" / prepare.CONVERSION_SNAPSHOT_FILENAME).write_bytes(
    snapshot_bytes
  )
  conversion = {
    "filename": prepare.USB_MIGRATION_FILENAME,
    "schema": prepare.USB_MIGRATION_SCHEMA,
    **_file_identity(snapshot_bytes),
  }
  stats_record = {
    "schema": "kaihand_norm_stats_provenance_v1",
    "dataset_root": dataset_root,
    "conversion_manifest": {
      key: conversion[key] for key in ("filename", "sha256", "size_bytes")
    },
    "parameters": {"prefix_frames": 0, "action_horizon": horizon, "arm_dim": 7},
    "norm_stats": {"filename": "norm_stats.json", **_file_identity(stats_bytes)},
  }
  report_bytes = _json_bytes({"test_report": True})
  if v2:
    stats_record.update(
      schema="kaihand_norm_stats_provenance_v2",
      normalization_policy={"method": "usb_insert_hand_tail_minmax_v1"},
      scale_report={
        "filename": "norm_stats.scale_report.json",
        **_file_identity(report_bytes),
      },
    )
    (directory / "norm_stats.scale_report.json").write_bytes(report_bytes)
  sidecar_bytes = _json_bytes(stats_record)
  (directory / "norm_stats.provenance.json").write_bytes(sidecar_bytes)
  dataset = {
    "repo_id": "usb_insert_0921_200",
    "manifest_repo_id": "usb_insert_0921_200",
    "root": dataset_root,
  }
  dataset["artifacts"] = prepare._usb_snapshot_artifacts(snapshot, dataset)
  record = {
    "schema": "kaihand_training_provenance_v1",
    "binding_status": "bound",
    "task": "usb_insert",
    "config_name": "pi05_kaihand_h50" if horizon == 50 else "pi05_kaihand",
    "dataset": dataset,
    "conversion_manifest": conversion,
    "action_contract": contract,
    "action_contract_sha256": prepare._canonical_hash(contract),
    "normalizer": {
      "asset_id": asset_id,
      "keys": ["actions", "state"],
      "semantic_sha256": prepare._normalizer_semantic_sha256(
        directory / "norm_stats.json"
      ),
      "use_quantile_norm": True,
      "stats_provenance": {
        "filename": "norm_stats.provenance.json",
        **_file_identity(sidecar_bytes),
        "record": stats_record,
      },
    },
    "training_action_transform": {
      "action_dim": 27,
      "model_action_dim": 32,
      "action_horizon": horizon,
      "arm_delta_mask": [True] * 7 + [False] * 20,
      "skip_episode_prefix_frames": 0,
    },
  }
  (checkpoint / "assets" / prepare.TRAINING_PROVENANCE_FILENAME).write_bytes(
    _json_bytes(record)
  )
  portable_calls = []

  def validate_v2(actual_record, **assets):
    assert actual_record == record
    assert assets == {
      "conversion_manifest_bytes": snapshot_bytes,
      "norm_stats_bytes": stats_bytes,
      "sidecar_bytes": sidecar_bytes,
      "scale_report_bytes": report_bytes,
    }
    portable_calls.append(assets)

  monkeypatch.setattr(prepare, "_validate_v2_normalization", validate_v2)
  return prepare, checkpoint, record, portable_calls


def _rewrite_training_record(prepare, checkpoint, record):
  path = checkpoint / "assets" / prepare.TRAINING_PROVENANCE_FILENAME
  path.write_bytes(_json_bytes(record))


def test_bound_h50_resolves_config_and_safe_asset_from_checkpoint(
  tmp_path, monkeypatch
):
  prepare, checkpoint, record, portable_calls = _bound_checkpoint(tmp_path, monkeypatch)
  config_name, asset_id, binding = prepare.checkpoint_contract(checkpoint)
  assert config_name == "pi05_kaihand_h50"
  assert asset_id == SAFE_ASSET
  assert binding["record"] == record
  assert binding["status"] == "bound"
  assert len(portable_calls) == 1


@pytest.mark.parametrize("horizon", [30, 50])
def test_bound_legacy_stats_keep_checkpoint_specific_assets(
  tmp_path, monkeypatch, horizon
):
  prepare, checkpoint, record, _ = _bound_checkpoint(
    tmp_path, monkeypatch, horizon=horizon, v2=False
  )
  config_name, asset_id, binding = prepare.checkpoint_contract(checkpoint)
  assert config_name == record["config_name"]
  assert asset_id == ("normalizer" if horizon == 30 else "normalizer_h50")
  assert binding["record"]["normalizer"]["stats_provenance"]["record"][
    "schema"
  ].endswith("v1")


@pytest.mark.parametrize(
  ("field", "value", "message"),
  [
    ("task", "usb-unplug", "task"),
    ("config_name", "pi05_other", "config"),
    ("binding_status", "unbound", "bound"),
  ],
)
def test_prepare_rejects_other_task_config_or_unbound_provenance(
  tmp_path, monkeypatch, field, value, message
):
  prepare, checkpoint, record, _ = _bound_checkpoint(tmp_path, monkeypatch)
  record[field] = value
  _rewrite_training_record(prepare, checkpoint, record)
  with pytest.raises(RuntimeError, match=message):
    prepare.checkpoint_contract(checkpoint)


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("action_horizon", 30),
    ("action_dim", 32),
    ("model_action_dim", 27),
    ("arm_delta_mask", [True] * 27),
    ("arm_delta_mask", [1] * 7 + [0] * 20),
  ],
)
def test_prepare_rejects_inconsistent_action_transform(
  tmp_path, monkeypatch, field, value
):
  prepare, checkpoint, record, _ = _bound_checkpoint(tmp_path, monkeypatch)
  record["training_action_transform"][field] = value
  _rewrite_training_record(prepare, checkpoint, record)
  with pytest.raises(RuntimeError, match="action transform"):
    prepare.checkpoint_contract(checkpoint)


@pytest.mark.parametrize("asset_id", ["normalizer", "../normalizer", "a/b", ".."])
def test_prepare_rejects_wrong_or_unsafe_checkpoint_asset(
  tmp_path, monkeypatch, asset_id
):
  prepare, checkpoint, record, _ = _bound_checkpoint(tmp_path, monkeypatch)
  record["normalizer"]["asset_id"] = asset_id
  _rewrite_training_record(prepare, checkpoint, record)
  with pytest.raises(ValueError, match="normalization stats|one directory name"):
    prepare.checkpoint_contract(checkpoint)


def test_prepare_rejects_wrong_normalizer_semantics(tmp_path, monkeypatch):
  prepare, checkpoint, record, _ = _bound_checkpoint(tmp_path, monkeypatch)
  record["normalizer"]["semantic_sha256"] = "f" * 64
  _rewrite_training_record(prepare, checkpoint, record)
  with pytest.raises(RuntimeError, match="semantic SHA"):
    prepare.checkpoint_contract(checkpoint)


def test_prepare_rejects_changed_conversion_snapshot(tmp_path, monkeypatch):
  prepare, checkpoint, _, _ = _bound_checkpoint(tmp_path, monkeypatch)
  snapshot = checkpoint / "assets" / prepare.CONVERSION_SNAPSHOT_FILENAME
  payload = json.loads(snapshot.read_bytes())
  payload["total_episodes"] = 2
  snapshot.write_bytes(_json_bytes(payload))
  with pytest.raises(RuntimeError, match="snapshot identity"):
    prepare.checkpoint_contract(checkpoint)


@pytest.mark.parametrize(
  "filename", ["norm_stats.provenance.json", "norm_stats.scale_report.json"]
)
def test_v2_requires_checkpoint_local_sidecar_and_report(
  tmp_path, monkeypatch, filename
):
  prepare, checkpoint, _, _ = _bound_checkpoint(tmp_path, monkeypatch)
  (checkpoint / "assets" / SAFE_ASSET / filename).unlink()
  with pytest.raises(FileNotFoundError):
    prepare.checkpoint_contract(checkpoint)


def test_v2_portable_validator_failure_prevents_deployment(tmp_path, monkeypatch):
  prepare, checkpoint, _, _ = _bound_checkpoint(tmp_path, monkeypatch)

  def reject(*_args, **_kwargs):
    raise RuntimeError("scale report binding mismatch")

  monkeypatch.setattr(prepare, "_validate_v2_normalization", reject)
  with pytest.raises(RuntimeError, match="scale report binding"):
    prepare.checkpoint_contract(checkpoint)


def test_server_rejects_manifest_provenance_not_matching_checkpoint(
  tmp_path, monkeypatch
):
  prepare, path, manifest, _, _, _ = _prepared_manifest(tmp_path, monkeypatch)
  server = load_script("serve_usb_pi05_policy.py")
  monkeypatch.setattr(server, "checkpoint_contract", prepare.checkpoint_contract)
  manifest["training_provenance"]["provenance_asset"]["sha256"] = "f" * 64
  manifest["deployment_id"] = prepare._canonical_hash(
    prepare.deployment_identity(manifest)
  )
  path.write_bytes(_json_bytes(manifest))
  with pytest.raises(RuntimeError, match="checkpoint assets"):
    server._load_manifest(path)


@dataclasses.dataclass(frozen=True)
class RuntimeAssets:
  asset_id: str = "kaihand"
  assets_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class LeRobotKaiHandDataConfig:
  assets: RuntimeAssets = RuntimeAssets()
  use_delta_arm_actions: bool = True


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
  name: str
  data: LeRobotKaiHandDataConfig
  model: object
  policy_metadata: dict


def _mock_openpi(monkeypatch, record):
  horizon = record["training_action_transform"]["action_horizon"]
  contract = record["action_contract"]
  config = RuntimeConfig(
    name=record["config_name"],
    data=LeRobotKaiHandDataConfig(),
    model=SimpleNamespace(action_horizon=horizon, action_dim=32),
    policy_metadata={
      "action_dim": 27,
      "model_action_dim": 32,
      "action_horizon": horizon,
      "control_hz": 30,
      "suggested_replan_steps": 8,
      "joint_names": contract["joint_order"],
      "image_keys": ["observation.images.head", "observation.images.right_wrist"],
      "action_semantics": {
        "contract_id": contract["id"],
        "arm_0_6": contract["arm_0_6"],
        "hand_7_26": contract["hand_7_26"],
        "policy_output_is_absolute": True,
      },
    },
  )
  modules = {
    name: ModuleType(name)
    for name in (
      "openpi",
      "openpi.training",
      "openpi.training.config",
      "openpi.policies",
      "openpi.policies.policy_config",
      "openpi.serving",
      "openpi.serving.websocket_policy_server",
    )
  }
  for name, module in modules.items():
    monkeypatch.setitem(sys.modules, name, module)
  modules["openpi"].training = modules["openpi.training"]
  modules["openpi.training"].config = modules["openpi.training.config"]
  modules["openpi.training.config"].get_config = lambda name: (
    config if name == config.name else pytest.fail(name)
  )
  modules["openpi.policies"].policy_config = modules["openpi.policies.policy_config"]
  modules["openpi.serving"].websocket_policy_server = modules[
    "openpi.serving.websocket_policy_server"
  ]
  return config, modules


def _prepared_manifest(tmp_path, monkeypatch, *, legacy_unbound=False):
  prepare, checkpoint, record, portable_calls = _bound_checkpoint(
    tmp_path,
    monkeypatch,
    horizon=30 if legacy_unbound else 50,
    v2=not legacy_unbound,
  )
  if legacy_unbound:
    (checkpoint / "assets" / prepare.TRAINING_PROVENANCE_FILENAME).unlink()
    (checkpoint / "assets" / prepare.CONVERSION_SNAPSHOT_FILENAME).unlink()
  config, modules = _mock_openpi(monkeypatch, record)
  openpi_root = tmp_path / "openpi"
  for relative in (
    "src/openpi/training/config.py",
    "src/openpi/policies/kaihand_policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/serving/websocket_policy_server.py",
    "src/openpi/transforms.py",
    "src/openpi/shared/normalize.py",
    "src/openpi/policies/policy.py",
    "src/openpi/training/checkpoints.py",
    "src/openpi/training/kaihand_normalization.py",
    "src/openpi/training/kaihand_normalization_provenance.py",
  ):
    path = openpi_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture source bytes, never executed\n")
  output = tmp_path / "deployment"
  prepare.main(
    [
      "--checkpoint",
      str(checkpoint),
      "--output-dir",
      str(output),
      "--openpi-root",
      str(openpi_root),
    ]
  )
  path = output / "deployment_manifest.json"
  return prepare, path, json.loads(path.read_text()), config, modules, portable_calls


def test_prepare_mock_builds_h50_v2_bound_to_safe_checkpoint_asset(
  tmp_path, monkeypatch
):
  prepare, _, manifest, _, _, portable_calls = _prepared_manifest(tmp_path, monkeypatch)
  assert manifest["schema"] == "usb_pi05_deployment_v2"
  assert manifest["model_config_name"] == "pi05_kaihand_h50"
  assert manifest["normalizer_asset_id"] == SAFE_ASSET
  assert manifest["prediction_horizon"] == 50
  assert manifest["deployment_id"] == prepare._canonical_hash(
    prepare.deployment_identity(manifest)
  )
  assert len(portable_calls) == 2
  assert any(
    row["path"] == f"assets/{SAFE_ASSET}/norm_stats.scale_report.json"
    for row in manifest["checkpoint_files"]
  )


def test_server_mock_verifies_hashes_binds_checkpoint_asset_and_normalization_mode(
  tmp_path, monkeypatch
):
  prepare, path, manifest, original_config, modules, _ = _prepared_manifest(
    tmp_path, monkeypatch
  )
  server = load_script("serve_usb_pi05_policy.py")
  monkeypatch.setattr(server, "checkpoint_contract", prepare.checkpoint_contract)
  events = []
  policy_arguments = {}
  real_verify = server.verify_pi05_deployment

  def verify(payload):
    real_verify(payload)
    events.append("verified")

  def create_policy(config, checkpoint, **kwargs):
    assert config.name == "pi05_kaihand_h50"
    assert config.data.assets.asset_id == SAFE_ASSET
    assert config.data.assets.assets_dir == str(Path(checkpoint) / "assets")
    assert checkpoint == manifest["checkpoint_path"]
    policy_arguments.update(kwargs)
    events.append("created")
    return object()

  class Server:
    def __init__(self, **kwargs):
      assert kwargs["metadata"]["normalizer_asset_id"] == SAFE_ASSET
      events.append("server")

    def serve_forever(self):
      events.append("listen")

  monkeypatch.setattr(server, "verify_pi05_deployment", verify)
  monkeypatch.setattr(
    server,
    "warmup_policy",
    lambda *_args: {
      "completed": True,
      "duration_s": 0.0,
      "action_shape": [50, 27],
      "rng_restored": True,
    },
  )
  modules["openpi.policies.policy_config"].create_trained_policy = create_policy
  modules["openpi.serving.websocket_policy_server"].WebsocketPolicyServer = Server
  server.main(["--deployment-manifest", str(path)])
  assert events == ["verified", "created", "server", "listen"]
  assert policy_arguments["expected_use_quantile_norm"] is True
  assert original_config.data.assets.asset_id == "kaihand"


def test_legacy_unbound_h30_prepare_and_server_keep_normalizer_and_unspecified_mode(
  tmp_path, monkeypatch
):
  prepare, path, manifest, _, modules, portable_calls = _prepared_manifest(
    tmp_path, monkeypatch, legacy_unbound=True
  )
  assert manifest["schema"] == "usb_pi05_deployment_v1"
  assert manifest["model_config_name"] == "pi05_kaihand"
  assert manifest["normalizer_asset_id"] == "normalizer"
  assert manifest["prediction_horizon"] == 30
  assert "training_provenance" not in manifest
  assert portable_calls == []
  server = load_script("serve_usb_pi05_policy.py")
  monkeypatch.setattr(server, "checkpoint_contract", prepare.checkpoint_contract)
  policy_arguments = {}

  def create_policy(_config, _checkpoint, **kwargs):
    policy_arguments.update(kwargs)
    return object()

  class Server:
    def __init__(self, **_kwargs):
      pass

    def serve_forever(self):
      pass

  modules["openpi.policies.policy_config"].create_trained_policy = create_policy
  modules["openpi.serving.websocket_policy_server"].WebsocketPolicyServer = Server
  server.main(["--deployment-manifest", str(path)])
  assert policy_arguments["expected_use_quantile_norm"] is None


@pytest.mark.parametrize(
  "entrypoint", ["run_usb_pi05_policy.py", "evaluate_usb_pi05_policy_batch.py"]
)
def test_usb_evaluation_loaders_accept_v2_and_reject_identity_tampering(
  tmp_path, monkeypatch, entrypoint
):
  _, path, manifest, _, _, _ = _prepared_manifest(tmp_path, monkeypatch)
  real_import = builtins.__import__

  def import_without_openpi(name, *args, **kwargs):
    if name == "openpi" or name.startswith("openpi."):
      pytest.fail("simulator-only USB loaders must not import OpenPI")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", import_without_openpi)
  module = load_script(entrypoint)
  loader = getattr(module, "load_deployment_manifest", None) or module.load_manifest
  assert loader(path)[1]["prediction_horizon"] == 50
  manifest["normalizer_asset_id"] = "normalizer_h50"
  path.write_bytes(_json_bytes(manifest))
  with pytest.raises(RuntimeError, match="authenticate"):
    loader(path)


def test_unified_deployment_loader_accepts_usb_v2_and_authenticates_identity(
  tmp_path, monkeypatch
):
  from kaihand_tactile_env.pipeline import deployment

  _, path, manifest, _, _, _ = _prepared_manifest(tmp_path, monkeypatch)
  assert (
    deployment.load_deployment_manifest(path)[1]["normalizer_asset_id"] == SAFE_ASSET
  )
  manifest["instruction"] = "different instruction"
  path.write_bytes(_json_bytes(manifest))
  with pytest.raises(deployment.DeploymentError, match="authenticate"):
    deployment.load_deployment_manifest(path)


@pytest.mark.parametrize(
  "field",
  ["normalizer_asset_id", "training_provenance", "instruction", "openpi_source_sha256"],
)
def test_v2_deployment_identity_authenticates_new_fields(tmp_path, monkeypatch, field):
  prepare, _, manifest, _, _, _ = _prepared_manifest(tmp_path, monkeypatch)
  manifest[field] = {} if field == "training_provenance" else "changed"
  with pytest.raises(RuntimeError, match="authenticate"):
    prepare.validate_deployment_identity(manifest)
