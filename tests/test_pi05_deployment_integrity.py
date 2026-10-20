from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from kaihand_tactile_env.shared.pi05_deployment_integrity import (
  BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG,
  SHARED_DEPLOYMENT_ID_FIELDS,
  _fast_snapshot_artifact_aggregate,
  action_representation_from_provenance,
  deployment_diagnostic_reasons,
  inspect_shared_pi05_runtime_integrity,
  load_checkpoint_training_provenance,
  unbound_training_provenance,
  validate_shared_deployment_semantics,
  verify_pi05_deployment,
)
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_TASKS,
  current_variant_environment,
)


def _rows(root: Path, paths: list[Path]) -> tuple[list[dict], str]:
  rows = [
    {
      "path": path.relative_to(root).as_posix(),
      "size": path.stat().st_size,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in paths
  ]
  encoded = json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return rows, hashlib.sha256(encoded).hexdigest()


def _canonical(value) -> str:
  encoded = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
  ).encode()
  return hashlib.sha256(encoded).hexdigest()


def _manifest(tmp_path: Path) -> tuple[dict, Path, Path]:
  checkpoint = tmp_path / "checkpoint"
  (checkpoint / "params").mkdir(parents=True)
  (checkpoint / "assets").mkdir()
  committed = checkpoint / "_CHECKPOINT_METADATA"
  weights = checkpoint / "params/weights.bin"
  stats = checkpoint / "assets/stats.json"
  committed.write_bytes(b"metadata")
  weights.write_bytes(b"weight-A")
  stats.write_bytes(b"stats")
  checkpoint_rows, checkpoint_hash = _rows(
    checkpoint,
    [
      committed,
      stats,
      weights,
    ],
  )
  project = tmp_path / "openpi"
  source = project / "src/openpi/training/config.py"
  source.parent.mkdir(parents=True)
  source.write_bytes(b"config-A")
  source_rows, source_hash = _rows(project, [source])
  return (
    {
      "schema": "usb_pi05_deployment_v1",
      "checkpoint_path": str(checkpoint),
      "checkpoint_files": checkpoint_rows,
      "checkpoint_sha256": checkpoint_hash,
      "model_project_path": str(project),
      "openpi_source_files": source_rows,
      "openpi_source_sha256": source_hash,
    },
    weights,
    source,
  )


def test_pi05_deployment_rejects_same_size_checkpoint_change(tmp_path: Path) -> None:
  manifest, weights, _ = _manifest(tmp_path)
  verify_pi05_deployment(manifest)
  weights.write_bytes(b"weight-B")
  with pytest.raises(RuntimeError, match="checkpoint content SHA-256 mismatch"):
    verify_pi05_deployment(manifest)


def test_pi05_deployment_rejects_source_change_and_extra_file(tmp_path: Path) -> None:
  manifest, _, source = _manifest(tmp_path)
  source.write_bytes(b"config-B")
  with pytest.raises(RuntimeError, match="OpenPI source content SHA-256 mismatch"):
    verify_pi05_deployment(manifest)
  source.write_bytes(b"config-A")
  (Path(manifest["checkpoint_path"]) / "params/extra.bin").write_bytes(b"new")
  with pytest.raises(RuntimeError, match="checkpoint file inventory changed"):
    verify_pi05_deployment(manifest)


def _shared_runtime_manifest(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
  project = tmp_path / "openpi"
  openpi_source = project / "src/openpi/training/config.py"
  openpi_source.parent.mkdir(parents=True)
  openpi_source.write_bytes(b"config-A")
  client_source = project / "packages/openpi-client/src/openpi_client/__init__.py"
  client_source.parent.mkdir(parents=True)
  client_source.write_bytes(b"client-A")
  openpi_rows, openpi_hash = _rows(project, [client_source, openpi_source])

  evaluation = tmp_path / "evaluation"
  evaluation_source = evaluation / "runner.py"
  evaluation.mkdir()
  evaluation_source.write_bytes(b"runner-A")
  evaluation_rows, evaluation_hash = _rows(evaluation, [evaluation_source])
  checkpoint_rows = [{"path": "_CHECKPOINT_METADATA", "size": 8, "sha256": "1" * 64}]
  identity = {
    "task": "bulb-screw",
    "model_family": "pi0.5",
    "model_config_name": "pi05_kaihand",
    "normalizer_asset_id": "normalizer",
    "checkpoint_path": str(tmp_path / "checkpoint"),
    "checkpoint_sha256": _canonical(checkpoint_rows),
    "checkpoint_step": 24999,
    "prediction_horizon": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "suggested_execute_steps": 8,
    "joint_names": [f"joint_{index}" for index in range(27)],
    "instruction": "screw the bulb",
    "observation_contract": {"cameras": ["head", "right_wrist"]},
    "action_representation": {"policy_output_is_absolute": True},
    "model_project_path": str(project),
    "openpi_source_sha256": openpi_hash,
    "evaluation_source_sha256": evaluation_hash,
  }
  assert tuple(identity) == SHARED_DEPLOYMENT_ID_FIELDS
  manifest = {
    "schema": "shared_task_pi05_deployment_v1",
    "deployment_id": _canonical(identity),
    **identity,
    "checkpoint_files": checkpoint_rows,
    "openpi_source_files": openpi_rows,
    "evaluation_source_files": evaluation_rows,
  }
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return manifest, path, evaluation, openpi_source


def test_child_runtime_integrity_reports_actual_verified_digests(
  tmp_path: Path,
) -> None:
  manifest, path, evaluation, _ = _shared_runtime_manifest(tmp_path)
  expected_manifest = hashlib.sha256(path.read_bytes()).hexdigest()

  report = inspect_shared_pi05_runtime_integrity(
    path,
    manifest,
    evaluation_root=evaluation,
    expected_manifest_sha256=expected_manifest,
  )

  assert report["status"] == "verified"
  assert report["manifest"]["actual_sha256"] == expected_manifest
  assert report["deployment_id"]["recomputed"] == manifest["deployment_id"]
  assert report["openpi_source"]["actual_sha256"] == manifest["openpi_source_sha256"]
  assert (
    report["evaluation_source"]["actual_sha256"] == manifest["evaluation_source_sha256"]
  )


def test_child_runtime_integrity_rejects_forged_deployment_id(tmp_path: Path) -> None:
  manifest, path, evaluation, _ = _shared_runtime_manifest(tmp_path)
  manifest["deployment_id"] = "0" * 64
  path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

  report = inspect_shared_pi05_runtime_integrity(
    path,
    manifest,
    evaluation_root=evaluation,
    expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  )

  assert report["status"] == "mismatch"
  assert "deployment_id does not authenticate" in " ".join(report["errors"])


def test_child_runtime_integrity_rejects_openpi_source_drift(tmp_path: Path) -> None:
  manifest, path, evaluation, openpi_source = _shared_runtime_manifest(tmp_path)
  openpi_source.write_bytes(b"config-B")

  report = inspect_shared_pi05_runtime_integrity(
    path,
    manifest,
    evaluation_root=evaluation,
    expected_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  )

  assert report["status"] == "mismatch"
  assert report["openpi_source"]["status"] == "mismatch"
  assert "content SHA-256 mismatch" in report["openpi_source"]["error"]


def _bound_bulb_provenance(
  checkpoint: Path,
  *,
  source_hash_verification: str = "recomputed",
  task: str = "bulb-screw",
  config_name: str = "pi05_kaihand_bulb_controller_target_v1",
  action_horizon: int = 30,
  asset_id: str = "normalizer",
  contract_id: str = "recorded-controller-target-v1",
  instruction: str | None = None,
  source_environment: object = ...,
) -> tuple[dict, dict]:
  assets = checkpoint / "assets"
  normalizer = assets / asset_id
  normalizer.mkdir(parents=True)
  values = {
    "state": {
      name: [float(index) for index in range(27)]
      for name in ("mean", "std", "q01", "q99")
    },
    "actions": {
      name: [float(index + 1) for index in range(27)]
      for name in ("mean", "std", "q01", "q99")
    },
  }
  normalizer_path = normalizer / "norm_stats.json"
  normalizer_path.write_text(json.dumps({"norm_stats": values}), encoding="utf-8")
  normalizer_bytes = normalizer_path.read_bytes()
  semantic = {
    key: {
      statistic: {"shape": [27], "values": statistic_values}
      for statistic, statistic_values in statistics.items()
    }
    for key, statistics in values.items()
  }
  joints = [f"joint_{index}" for index in range(27)]
  contract = {
    "id": contract_id,
    "action_type": "absolute_joint_position_target_rad",
    "dimensions": 27,
    "joint_order": joints,
    "source_control_hz": 100,
    "training_row_alignment": {},
    "evaluation_execution": {
      "endpoint_rate_hz": 30,
      "physical_substep_interpolation_required": "linear",
      "zero_order_hold_equivalent": False,
    },
    "legacy": contract_id == "legacy-measured-arm-hybrid-v1",
    "arm_0_6": "recorded arm controller targets",
    "hand_7_26": "recorded hand controller targets",
    "source_sampling": "fixture",
  }
  snapshot = {
    "schema": "kaihand_fast_pi05_video_v2",
    "task": task,
    "source_hash_verification": source_hash_verification,
    "repo_id": "conversion/repo",
    "action_contract": contract,
    "source_environment": {
      "mechanics_version": "phase_aware_strict_completion_v15",
      "model_fingerprint": (
        "cac34f5fc061c4d854be34563a312e83b9352dd2629a4215908fcbf778eb1d46"
      ),
    },
    "cameras": ["head", "right_wrist"],
    "total_episodes": 1,
    "metadata_artifacts": {
      relative: {"size_bytes": index, "sha256": f"{index:x}" * 64}
      for index, relative in enumerate(
        (
          "meta/episodes.jsonl",
          "meta/episodes_stats.jsonl",
          "meta/info.json",
          "meta/kaihand_source_episodes.jsonl",
          "meta/tasks.jsonl",
        ),
        start=1,
      )
    },
    "episodes": [
      {
        "output_episode_index": 0,
        "action_contract": contract_id,
        "artifacts": {
          "data.parquet": {"size_bytes": 10, "sha256": "a" * 64},
          "stats.json": {"size_bytes": 9, "sha256": "b" * 64},
          "head.mp4": {"size_bytes": 11, "sha256": "c" * 64},
          "right_wrist.mp4": {"size_bytes": 12, "sha256": "d" * 64},
        },
      }
    ],
  }
  if task in VARIANT_TASKS:
    snapshot["instruction"] = (
      TASK_INSTRUCTIONS[task] if instruction is None else instruction
    )
    snapshot["source_environment"] = current_variant_environment(task)
  if source_environment is not ...:
    snapshot["source_environment"] = source_environment
  snapshot_bytes = (json.dumps(snapshot, indent=2) + "\n").encode()
  (assets / "conversion_manifest.snapshot.json").write_bytes(snapshot_bytes)
  stats_record = {
    "schema": "kaihand_norm_stats_provenance_v1",
    "dataset_root": "/dataset",
    "conversion_manifest": {
      "filename": "kaihand_fast_pi05_conversion.json",
      "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
      "size_bytes": len(snapshot_bytes),
    },
    "parameters": {
      "prefix_frames": 0,
      "action_horizon": action_horizon,
      "arm_dim": 7,
    },
    "norm_stats": {
      "filename": "norm_stats.json",
      "sha256": hashlib.sha256(normalizer_bytes).hexdigest(),
      "size_bytes": len(normalizer_bytes),
    },
  }
  stats_provenance_bytes = (
    json.dumps(stats_record, sort_keys=True, indent=2) + "\n"
  ).encode()
  provenance = {
    "schema": "kaihand_training_provenance_v1",
    "binding_status": "bound",
    "task": task,
    "config_name": config_name,
    "dataset": {
      # The loader-facing id is allowed to differ from the manifest repo id;
      # both still bind to the same dataset root and snapshotted authority.
      "repo_id": "local/training-alias",
      "root": "/dataset",
      "manifest_repo_id": "conversion/repo",
      "artifacts": _fast_snapshot_artifact_aggregate(snapshot),
    },
    "conversion_manifest": {
      "filename": "kaihand_fast_pi05_conversion.json",
      "schema": "kaihand_fast_pi05_video_v2",
      "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
      "size_bytes": len(snapshot_bytes),
    },
    "action_contract": contract,
    "action_contract_sha256": _canonical(contract),
    "source_environment": snapshot["source_environment"],
    "normalizer": {
      "asset_id": asset_id,
      "keys": ["actions", "state"],
      "semantic_sha256": _canonical(semantic),
      "use_quantile_norm": True,
      "stats_provenance": {
        "filename": "norm_stats.provenance.json",
        "sha256": hashlib.sha256(stats_provenance_bytes).hexdigest(),
        "size_bytes": len(stats_provenance_bytes),
        "record": stats_record,
      },
    },
    "training_action_transform": {
      "action_dim": 27,
      "model_action_dim": 32,
      "action_horizon": action_horizon,
      "arm_delta_mask": [True] * 7 + [False] * 20,
      "skip_episode_prefix_frames": 0,
    },
  }
  (assets / "training_provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
  )
  return provenance, snapshot


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_bound_variant_provenance_accepts_exact_frozen_task_prompt(
  tmp_path: Path, task: str,
) -> None:
  checkpoint = tmp_path / task / "24999"
  provenance, snapshot = _bound_bulb_provenance(
    checkpoint,
    task=task,
    config_name=(BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG
                 if task == "bulb-screw-inplace" else "pi05_kaihand_h50"),
    action_horizon=50,
    contract_id=("recorded-controller-target-v1" if task == "bulb-screw-inplace"
                 else "legacy-measured-arm-hybrid-v1"),
  )

  binding = load_checkpoint_training_provenance(checkpoint)

  assert snapshot["instruction"] == TASK_INSTRUCTIONS[task]
  assert provenance["source_environment"] == current_variant_environment(task)
  assert binding["record"]["task"] == task


@pytest.mark.parametrize("environment_change", [
  "missing_protocol", "legacy_protocol", "unknown_protocol", "null_protocol",
  "unknown_field", "no_environment",
])
def test_usb_bound_provenance_requires_exact_current_three_field_environment(
  tmp_path: Path, environment_change: str,
):
  environment = current_variant_environment("usb-unplug")
  assert set(environment) == {
    "mechanics_version", "model_fingerprint", "completion_protocol",
  }
  if environment_change == "missing_protocol":
    environment.pop("completion_protocol")
  elif environment_change == "legacy_protocol":
    environment["completion_protocol"] = "usb-unplug-legacy-return-release"
  elif environment_change == "unknown_protocol":
    environment["completion_protocol"] = "unknown-protocol"
  elif environment_change == "null_protocol":
    environment["completion_protocol"] = None
  elif environment_change == "unknown_field":
    environment["unreviewed"] = "injected"
  elif environment_change == "no_environment":
    environment = None
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint,
    task="usb-unplug",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    contract_id="legacy-measured-arm-hybrid-v1",
    source_environment=environment,
  )
  with pytest.raises(RuntimeError, match="source environment|completion protocol"):
    load_checkpoint_training_provenance(checkpoint)


def test_usb_bound_provenance_cannot_drift_from_frozen_snapshot_environment(tmp_path: Path):
  checkpoint = tmp_path / "24999"
  provenance, _ = _bound_bulb_provenance(
    checkpoint,
    task="usb-unplug",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    contract_id="legacy-measured-arm-hybrid-v1",
  )
  provenance["source_environment"]["completion_protocol"] = "usb-unplug-legacy-return-release"
  (checkpoint / "assets/training_provenance.json").write_text(
    json.dumps(provenance), encoding="utf-8",
  )
  with pytest.raises(RuntimeError, match="source environment is not exactly bound"):
    load_checkpoint_training_provenance(checkpoint)


def test_usb_bound_provenance_rejects_runtime_completion_protocol_drift(tmp_path: Path, monkeypatch):
  from kaihand_tactile_env.tasks.usb_unplug import config

  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint,
    task="usb-unplug",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    contract_id="legacy-measured-arm-hybrid-v1",
  )
  monkeypatch.setattr(config, "COMPLETION_PROTOCOL", "usb-unplug-future-protocol")
  with pytest.raises(RuntimeError, match="completion protocol differs from evaluation"):
    load_checkpoint_training_provenance(checkpoint)


def test_other_task_environment_does_not_accept_usb_completion_protocol(tmp_path: Path):
  environment = current_variant_environment("whiteboard-wipe-horizontal-line")
  assert set(environment) == {"mechanics_version", "model_fingerprint"}
  environment["completion_protocol"] = "usb-unplug-held-clear-2s-v2"
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint,
    task="whiteboard-wipe-horizontal-line",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    contract_id="legacy-measured-arm-hybrid-v1",
    source_environment=environment,
  )
  with pytest.raises(RuntimeError, match="source environment is not exactly bound"):
    load_checkpoint_training_provenance(checkpoint)


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_bound_variant_provenance_rejects_prompt_drift(
  tmp_path: Path, task: str,
) -> None:
  checkpoint = tmp_path / task / "24999"
  _bound_bulb_provenance(
    checkpoint,
    task=task,
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    contract_id="legacy-measured-arm-hybrid-v1",
    instruction=f"stale prompt for {task}",
  )

  with pytest.raises(RuntimeError, match="instruction differs from the frozen task prompt"):
    load_checkpoint_training_provenance(checkpoint)


@pytest.mark.parametrize("task", ("bulb-screw", "bulb-screw-inplace"))
def test_bound_bulb_provenance_authenticates_assets_and_formal_contract(
  tmp_path: Path, task: str,
) -> None:
  checkpoint = tmp_path / "24999"
  inplace = task == "bulb-screw-inplace"
  config_name = (BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG if inplace
                 else "pi05_kaihand_bulb_controller_target_v1")
  horizon = 50 if inplace else 30
  asset_id = "normalizer_h50" if inplace else "normalizer"
  provenance, _ = _bound_bulb_provenance(
    checkpoint, task=task, config_name=config_name,
    action_horizon=horizon, asset_id=asset_id,
  )

  binding = load_checkpoint_training_provenance(checkpoint)
  manifest = {
    "schema": "shared_task_pi05_deployment_v2",
    "task": task,
    "model_config_name": config_name,
    "checkpoint_path": str(checkpoint),
    "prediction_horizon": horizon,
    "model_action_dim": 32,
    "action_dim": 27,
    "joint_names": provenance["action_contract"]["joint_order"],
    "normalizer_asset_id": asset_id,
    "action_representation": action_representation_from_provenance(provenance),
    "training_provenance": binding,
    "required_action_interpolation": "linear",
    "diagnostic_only": False,
    "formal_metrics_valid": True,
  }

  validate_shared_deployment_semantics(manifest)
  assert deployment_diagnostic_reasons(manifest) == []
  assert deployment_diagnostic_reasons(manifest, action_interpolation="zoh") == [
    "action_interpolation_zoh_differs_from_required_linear"
  ]
  if task == "bulb-screw-inplace":
    wrong_task = {**manifest, "task": "bulb-screw"}
    with pytest.raises(RuntimeError, match="task differs from checkpoint provenance"):
      validate_shared_deployment_semantics(wrong_task)
    import copy

    legacy_config = copy.deepcopy(manifest)
    legacy_config["model_config_name"] = "pi05_kaihand"
    legacy_config["training_provenance"]["record"]["config_name"] = "pi05_kaihand"
    with pytest.raises(RuntimeError, match="dedicated controller-target config"):
      validate_shared_deployment_semantics(legacy_config, verify_checkpoint_assets=False)


def test_bound_bulb_h50_preserves_horizon_and_normalizer_provenance(
  tmp_path: Path,
) -> None:
  checkpoint = tmp_path / "24999"
  provenance, _ = _bound_bulb_provenance(
    checkpoint,
    config_name="pi05_kaihand_bulb_controller_target_h50_v1",
    action_horizon=50,
    asset_id="normalizer_h50",
  )
  binding = load_checkpoint_training_provenance(checkpoint)
  manifest = {
    "schema": "shared_task_pi05_deployment_v2",
    "task": "bulb-screw",
    "model_config_name": "pi05_kaihand_bulb_controller_target_h50_v1",
    "checkpoint_path": str(checkpoint),
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "joint_names": provenance["action_contract"]["joint_order"],
    "normalizer_asset_id": "normalizer_h50",
    "action_representation": action_representation_from_provenance(provenance),
    "training_provenance": binding,
    "required_action_interpolation": "linear",
    "diagnostic_only": False,
    "formal_metrics_valid": True,
  }

  validate_shared_deployment_semantics(manifest)
  assert deployment_diagnostic_reasons(manifest) == []


@pytest.mark.parametrize("config_name", (
  "pi05_kaihand", "pi05_kaihand_h50",
  "pi05_kaihand_bulb_controller_target_v1",
  "pi05_kaihand_bulb_controller_target_h50_v1",
))
def test_inplace_rejects_legacy_and_pickup_training_config(tmp_path, config_name):
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint, task="bulb-screw-inplace", config_name=config_name,
    action_horizon=50, asset_id="normalizer_h50",
  )
  with pytest.raises(RuntimeError, match="its own dedicated controller-target config"):
    load_checkpoint_training_provenance(checkpoint)


@pytest.mark.parametrize("task", ("bulb-screw", "install-ram", "sponge-grasp"))
def test_inplace_training_config_cannot_be_relabelled_for_another_task(tmp_path, task):
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint, task=task,
    config_name=BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG,
    action_horizon=50, asset_id="normalizer_h50",
  )
  with pytest.raises(RuntimeError, match="identities cannot be relabeled"):
    load_checkpoint_training_provenance(checkpoint)


def test_inplace_rejects_measured_arm_actions_and_old_pickup_environment(tmp_path):
  checkpoint = tmp_path / "measured" / "24999"
  _bound_bulb_provenance(
    checkpoint, task="bulb-screw-inplace",
    config_name=BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG,
    action_horizon=50, asset_id="normalizer_h50",
    contract_id="legacy-measured-arm-hybrid-v1",
  )
  with pytest.raises(RuntimeError, match="requires recorded-controller-target-v1"):
    load_checkpoint_training_provenance(checkpoint)
  checkpoint = tmp_path / "wrong-scene" / "24999"
  _bound_bulb_provenance(
    checkpoint, task="bulb-screw-inplace",
    config_name=BULB_SCREW_INPLACE_CONTROLLER_TARGET_H50_CONFIG,
    action_horizon=50, asset_id="normalizer_h50",
    source_environment={
      "mechanics_version": "phase_aware_strict_completion_v15",
      "model_fingerprint": "cac34f5fc061c4d854be34563a312e83b9352dd2629a4215908fcbf778eb1d46",
    },
  )
  with pytest.raises(RuntimeError, match="mechanics/model identity differs"):
    load_checkpoint_training_provenance(checkpoint)


def test_bound_whiteboard_h50_preserves_horizon_and_normalizer_provenance(
  tmp_path: Path,
) -> None:
  checkpoint = tmp_path / "24999"
  provenance, _ = _bound_bulb_provenance(
    checkpoint,
    task="whiteboard-wipe",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    asset_id="normalizer_h50",
    contract_id="legacy-measured-arm-hybrid-v1",
  )
  binding = load_checkpoint_training_provenance(checkpoint)
  manifest = {
    "schema": "shared_task_pi05_deployment_v2",
    "task": "whiteboard-wipe",
    "model_config_name": "pi05_kaihand_h50",
    "checkpoint_path": str(checkpoint),
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "joint_names": provenance["action_contract"]["joint_order"],
    "normalizer_asset_id": "normalizer_h50",
    "action_representation": action_representation_from_provenance(provenance),
    "training_provenance": binding,
    "required_action_interpolation": "linear",
    "diagnostic_only": False,
    "formal_metrics_valid": True,
  }

  validate_shared_deployment_semantics(manifest)
  assert deployment_diagnostic_reasons(manifest) == []
  with pytest.raises(RuntimeError, match="dimensions differ"):
    validate_shared_deployment_semantics({**manifest, "prediction_horizon": 30})
  with pytest.raises(RuntimeError, match="normalizer asset differs"):
    validate_shared_deployment_semantics(
      {**manifest, "normalizer_asset_id": "normalizer"}
    )
  manifest["task"] = binding["record"]["task"] = "bulb-screw"
  with pytest.raises(RuntimeError, match="dedicated controller-target config"):
    validate_shared_deployment_semantics(manifest, verify_checkpoint_assets=False)


@pytest.mark.parametrize(
  ("config_name", "action_horizon"),
  [
    ("pi05_kaihand", 50),
    ("pi05_kaihand_h50", 30),
    ("pi05_kaihand_bulb_controller_target_v1", 50),
    ("pi05_kaihand_bulb_controller_target_h50_v1", 30),
  ],
)
def test_provenance_rejects_horizon_inconsistent_with_named_config(
  tmp_path: Path, config_name: str, action_horizon: int
) -> None:
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint,
    config_name=config_name,
    action_horizon=action_horizon,
  )
  with pytest.raises(RuntimeError, match="training action transform mismatch"):
    load_checkpoint_training_provenance(checkpoint)


def test_h50_provenance_rejects_h30_normalization_statistics(tmp_path: Path) -> None:
  checkpoint = tmp_path / "24999"
  provenance, _ = _bound_bulb_provenance(
    checkpoint,
    task="whiteboard-wipe",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    asset_id="normalizer_h50",
    contract_id="legacy-measured-arm-hybrid-v1",
  )
  provenance["normalizer"]["stats_provenance"]["record"]["parameters"][
    "action_horizon"
  ] = 30
  (checkpoint / "assets/training_provenance.json").write_text(
    json.dumps(provenance), encoding="utf-8"
  )

  with pytest.raises(RuntimeError, match="normalization statistics are not bound"):
    load_checkpoint_training_provenance(checkpoint)


def test_bound_provenance_rejects_normalizer_tamper(tmp_path: Path) -> None:
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(checkpoint)
  normalizer = checkpoint / "assets/normalizer/norm_stats.json"
  document = json.loads(normalizer.read_text(encoding="utf-8"))
  document["norm_stats"]["actions"]["mean"][0] = 999.0
  normalizer.write_text(json.dumps(document), encoding="utf-8")

  with pytest.raises(RuntimeError, match="normalizer semantic SHA-256"):
    load_checkpoint_training_provenance(checkpoint)


def test_bound_provenance_recomputes_dataset_aggregate_from_snapshot(
  tmp_path: Path,
) -> None:
  checkpoint = tmp_path / "24999"
  provenance, _ = _bound_bulb_provenance(checkpoint)
  provenance["dataset"]["artifacts"]["file_count"] -= 1
  (checkpoint / "assets/training_provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
  )

  with pytest.raises(RuntimeError, match="differs from the conversion snapshot"):
    load_checkpoint_training_provenance(checkpoint)


def test_bound_bulb_provenance_requires_recomputed_source_hashes(
  tmp_path: Path,
) -> None:
  checkpoint = tmp_path / "24999"
  _bound_bulb_provenance(
    checkpoint,
    source_hash_verification="trusted_acquisition_sidecar",
  )

  with pytest.raises(RuntimeError, match="recomputed source HDF5 hashes"):
    load_checkpoint_training_provenance(checkpoint)


def test_unbound_v2_is_irreversibly_diagnostic() -> None:
  manifest = {
    "schema": "shared_task_pi05_deployment_v2",
    "model_config_name": "pi05_kaihand",
    "action_representation": {
      "contract_id": "legacy-measured-arm-hybrid-v1",
      "policy_output_is_absolute": True,
    },
    "training_provenance": unbound_training_provenance("legacy checkpoint"),
    "required_action_interpolation": "linear",
    "diagnostic_only": True,
    "formal_metrics_valid": False,
  }
  validate_shared_deployment_semantics(manifest)
  assert deployment_diagnostic_reasons(manifest) == ["legacy checkpoint"]

  manifest.update(diagnostic_only=False, formal_metrics_valid=True)
  with pytest.raises(RuntimeError, match="must be diagnostic-only"):
    validate_shared_deployment_semantics(manifest)
