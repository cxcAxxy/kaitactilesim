from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.pipeline.deployment import adapter_for, build_evaluation_plan
from kaihand_tactile_env.shared import pi05_deployment_integrity as integrity

ROOT = Path(__file__).resolve().parents[1]


def canonical(value) -> str:
  return hashlib.sha256(
    json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
  ).hexdigest()


def load_script(name):
  workcell = ROOT / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  try:
    spec = importlib.util.spec_from_file_location(
      name.removesuffix(".py"), workcell / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(workcell))


def rows(root, names):
  result = []
  for name in names:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    result.append(
      {
        "path": name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      }
    )
  return result, canonical(result)


@pytest.fixture
def bound_manifest(tmp_path, monkeypatch):
  joints = [f"right_arm_joint{i}" for i in range(1, 8)] + [
    f"right_hand_joint{i}" for i in range(20)
  ]
  record = {
    "task": "poker-draw",
    "config_name": "pi05_kaihand_h50",
    "source_environment": None,
    "action_contract": {
      "id": integrity.LEGACY_ACTION_CONTRACT,
      "arm_0_6": "measured next endpoint",
      "hand_7_26": "absolute controller target",
      "joint_order": joints,
      "evaluation_execution": {"physical_substep_interpolation_required": "linear"},
    },
    "training_action_transform": {
      "action_horizon": 50,
      "action_dim": 27,
      "model_action_dim": 32,
      "arm_delta_mask": [True] * 7 + [False] * 20,
    },
    "normalizer": {"asset_id": "normalizer_h50"},
  }
  provenance = {"status": "bound", "record": record}
  monkeypatch.setattr(
    integrity,
    "load_checkpoint_training_provenance",
    lambda checkpoint: copy.deepcopy(provenance),
  )
  checkpoint = tmp_path / "20000"
  checkpoint_rows, checkpoint_hash = rows(
    checkpoint,
    ["_CHECKPOINT_METADATA", "assets/normalizer_h50/norm_stats.json", "params/weights"],
  )
  project = tmp_path / "openpi"
  openpi_rows, openpi_hash = rows(
    project,
    [
      "packages/openpi-client/src/openpi_client/__init__.py",
      "src/openpi/training/config.py",
    ],
  )
  evaluation = tmp_path / "evaluation"
  evaluation_rows, evaluation_hash = rows(evaluation, ["runner.py"])
  manifest = {
    "schema": integrity.POKER_DEPLOYMENT_SCHEMA_V2,
    "task": "poker-draw",
    "model_family": "pi0.5",
    "model_config_name": "pi05_kaihand_h50",
    "normalizer_asset_id": "normalizer_h50",
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_step": 20000,
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "joint_names": joints,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": integrity.action_representation_from_provenance(record),
    "training_provenance": provenance,
    "control_hz": 30,
    "suggested_execute_steps": 50,
    "instruction": integrity.POKER_INSTRUCTION,
    "model_project_path": str(project),
    "openpi_source_sha256": openpi_hash,
    "evaluation_source_sha256": evaluation_hash,
    "required_action_interpolation": "linear",
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "checkpoint_files": checkpoint_rows,
    "openpi_source_files": openpi_rows,
    "evaluation_source_files": evaluation_rows,
  }
  manifest["deployment_id"] = canonical(integrity.poker_deployment_identity(manifest))
  path = tmp_path / "deployment.json"
  path.write_text(json.dumps(manifest))
  return manifest, path, evaluation


def test_h50_manifest_and_h30_legacy_identity_are_supported(bound_manifest):
  manifest, path, _ = bound_manifest
  runner = load_script("run_poker_pi05_policy.py")
  batch = load_script("evaluate_poker_pi05_policy_batch.py")
  assert runner.load_deployment_manifest(path)[1]["prediction_horizon"] == 50
  assert batch.load_manifest(path)[1]["normalizer_asset_id"] == "normalizer_h50"
  historical = {
    **manifest,
    "schema": integrity.POKER_DEPLOYMENT_SCHEMA_V1,
    "prediction_horizon": 30,
  }
  assert (
    tuple(integrity.poker_deployment_identity(historical))
    == integrity.POKER_DEPLOYMENT_V1_ID_FIELDS
  )
  assert integrity.deployment_diagnostic_reasons(historical) == [
    "legacy_v1_unbound_deployment"
  ]


@pytest.mark.parametrize(
  "key,value",
  [
    ("prediction_horizon", 30),
    ("normalizer_asset_id", "normalizer"),
    ("required_action_interpolation", "zoh"),
    ("formal_metrics_valid", True),
    ("model_config_name", "pi05_kaihand"),
    ("control_hz", 20),
  ],
)
def test_h50_semantics_fail_closed(bound_manifest, key, value):
  manifest, _, _ = bound_manifest
  changed = {**manifest, key: value}
  with pytest.raises(RuntimeError):
    integrity.validate_poker_deployment_semantics(changed)


def test_h50_server_contract_accepts_k50_not_k51(bound_manifest):
  manifest, _, _ = bound_manifest
  runner = load_script("run_poker_pi05_policy.py")
  metadata = {**manifest, "evaluation_task": "poker-draw", "action_horizon": 50}
  assert runner.validate_server(metadata, manifest, 50) == 50
  with pytest.raises(RuntimeError, match=r"\[1, 50\]"):
    runner.validate_server(metadata, manifest, 51)
  metadata["normalizer_asset_id"] = "normalizer"
  with pytest.raises(RuntimeError, match="identity mismatch"):
    runner.validate_server(metadata, manifest, 50)


def test_h50_runtime_integrity_checks_frozen_sources_and_manifest_bytes(bound_manifest):
  manifest, path, evaluation = bound_manifest
  expected = hashlib.sha256(path.read_bytes()).hexdigest()
  report = integrity.inspect_shared_pi05_runtime_integrity(
    path, manifest, evaluation_root=evaluation, expected_manifest_sha256=expected
  )
  assert report["status"] == "verified"
  assert report["training_provenance"]["status"] == "verified"
  (Path(manifest["model_project_path"]) / "src/openpi/training/config.py").write_bytes(
    b"changed"
  )
  report = integrity.inspect_shared_pi05_runtime_integrity(
    path, manifest, evaluation_root=evaluation, expected_manifest_sha256=expected
  )
  assert report["status"] == "mismatch"
  assert report["openpi_source"]["status"] == "mismatch"


def test_h50_full_verifier_checks_scene_sources_and_checkpoint(bound_manifest):
  manifest, _, evaluation = bound_manifest
  integrity.verify_pi05_deployment(manifest, evaluation_root=evaluation)
  (evaluation / "runner.py").write_bytes(b"changed")
  with pytest.raises(RuntimeError, match="evaluation source content SHA-256 mismatch"):
    integrity.verify_pi05_deployment(manifest, evaluation_root=evaluation)


def test_h50_runtime_rejects_forged_identity(bound_manifest):
  manifest, path, evaluation = bound_manifest
  manifest["deployment_id"] = "0" * 64
  path.write_text(json.dumps(manifest))
  report = integrity.inspect_shared_pi05_runtime_integrity(
    path, manifest, evaluation_root=evaluation
  )
  assert report["status"] == "mismatch"
  assert "deployment_id does not authenticate" in " ".join(report["errors"])


def test_poker_uses_physics_substep_linear_targets():
  runner = load_script("run_poker_pi05_policy.py")
  start, endpoint = np.zeros(27), np.ones(27)
  first = runner.linear_substep_action(
    start,
    endpoint,
    interval_start_s=0.0,
    interval_end_s=1 / 30,
    current_time_s=0.0,
    timestep_s=0.001,
  )
  np.testing.assert_allclose(first, 0.03)
  last = runner.linear_substep_action(
    start,
    endpoint,
    interval_start_s=0.0,
    interval_end_s=1 / 30,
    current_time_s=0.033,
    timestep_s=0.001,
  )
  np.testing.assert_allclose(last, endpoint)


def test_unified_adapter_accepts_v2_and_propagates_k50(bound_manifest, tmp_path):
  manifest, path, _ = bound_manifest
  adapter = adapter_for("poker-draw", "pi05")
  assert integrity.POKER_DEPLOYMENT_SCHEMA_V2 in adapter.manifest_schemas
  plan = build_evaluation_plan(
    path,
    manifest,
    adapter,
    output_dir=tmp_path / "rollouts",
    service_mode="managed",
    endpoint="ws://127.0.0.1:18810",
    profile=None,
    num_trials=20,
    seed_start=0,
    video_count=20,
    execute_steps="horizon",
  )
  assert plan.command[plan.command.index("--execute-steps") + 1] == "horizon"
  assert plan.command[plan.command.index("--video-count") + 1] == "20"


def test_batch_rejects_forged_h50_manifest(bound_manifest):
  manifest, path, _ = bound_manifest
  manifest["deployment_id"] = "0" * 64
  path.write_text(json.dumps(manifest))
  batch = load_script("evaluate_poker_pi05_policy_batch.py")
  with pytest.raises(RuntimeError, match="deployment_id does not authenticate"):
    batch.load_manifest(path)
