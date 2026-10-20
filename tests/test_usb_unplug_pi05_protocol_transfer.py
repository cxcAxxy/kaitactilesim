"""Opt-in old-policy/new-completion evaluation must never rewrite training."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from kaihand_tactile_env.shared import pi05_deployment_integrity as integrity
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS
from kaihand_tactile_env.shared.task_variants import current_variant_environment

from test_pi05_deployment_integrity import _bound_bulb_provenance, _canonical
from test_shared_pi05_deployment import Config, Model, load_script, policy_metadata


def _legacy_checkpoint(tmp_path: Path, *, protocol: object = ...) -> Path:
  checkpoint = tmp_path / "10000"
  environment = current_variant_environment("usb-unplug")
  environment.pop("completion_protocol")
  if protocol is not ...:
    environment["completion_protocol"] = protocol
  _bound_bulb_provenance(
    checkpoint,
    task="usb-unplug",
    config_name="pi05_kaihand_h50",
    action_horizon=50,
    asset_id="normalizer_h50",
    contract_id=integrity.LEGACY_ACTION_CONTRACT,
    instruction=integrity.USB_UNPLUG_LEGACY_INSTRUCTION,
    source_environment=environment,
  )
  (checkpoint / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
  (checkpoint / "params").mkdir()
  return checkpoint


def _binding(checkpoint: Path) -> dict:
  return integrity.load_checkpoint_training_provenance(
    checkpoint, allow_usb_unplug_protocol_transfer_diagnostic=True,
  )


def _manifest(checkpoint: Path) -> dict:
  provenance = _binding(checkpoint)
  record = provenance["record"]
  normalizer = checkpoint / "assets/normalizer_h50/norm_stats.json"
  payload = {
    "schema": integrity.SHARED_DEPLOYMENT_SCHEMA_V2,
    "task": "usb-unplug",
    "model_family": "pi0.5",
    "model_config_name": record["config_name"],
    "normalizer_asset_id": record["normalizer"]["asset_id"],
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": "a" * 64,
    "checkpoint_step": 10000,
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "suggested_execute_steps": 8,
    "joint_names": record["action_contract"]["joint_order"],
    "instruction": integrity.USB_UNPLUG_LEGACY_INSTRUCTION,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": integrity.action_representation_from_provenance(record),
    "model_project_path": str(checkpoint.parent / "openpi"),
    "openpi_source_sha256": "b" * 64,
    "evaluation_source_sha256": "c" * 64,
    "training_provenance": provenance,
    "required_action_interpolation": "linear",
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "checkpoint_files": [{
      "path": "assets/normalizer_h50/norm_stats.json",
      "size": normalizer.stat().st_size,
      "sha256": hashlib.sha256(normalizer.read_bytes()).hexdigest(),
    }],
    "openpi_source_files": [],
    "evaluation_source_files": [],
  }
  payload["deployment_id"] = _canonical(integrity.shared_deployment_identity(payload))
  return payload


@pytest.mark.parametrize("recorded_protocol", [..., integrity.USB_UNPLUG_LEGACY_COMPLETION_PROTOCOL])
def test_explicit_transfer_preserves_original_assets_prompt_and_horizon(tmp_path, recorded_protocol):
  checkpoint = _legacy_checkpoint(tmp_path, protocol=recorded_protocol)
  before = {path: path.read_bytes() for path in (checkpoint / "assets").rglob("*") if path.is_file()}
  with pytest.raises(RuntimeError, match="instruction differs"):
    integrity.load_checkpoint_training_provenance(checkpoint)

  binding = _binding(checkpoint)

  transfer = binding["completion_protocol_transfer"]
  assert transfer["training_completion_protocol"] == integrity.USB_UNPLUG_LEGACY_COMPLETION_PROTOCOL
  assert transfer["evaluation_completion_protocol"] == "usb-unplug-held-clear-2s-v2"
  assert transfer["training_completion_protocol_inferred_from_bound_prompt"] is (recorded_protocol is ...)
  assert transfer["policy_instruction"] == integrity.USB_UNPLUG_LEGACY_INSTRUCTION
  assert transfer["evaluation_instruction"] == TASK_INSTRUCTIONS["usb-unplug"]
  assert transfer["policy_instruction_preserved"] is True
  assert transfer["diagnostic_only"] is True
  assert transfer["formal_metrics_valid"] is False
  assert transfer["training_source_environment"] == binding["record"]["source_environment"]
  assert transfer["evaluation_source_environment"] == current_variant_environment("usb-unplug")
  assert binding["record"]["training_action_transform"]["action_horizon"] == 50
  assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("protocol", [None, "unknown", "usb-unplug-held-clear-2s-v2"])
def test_transfer_rejects_unknown_null_or_current_recorded_protocol(tmp_path, protocol):
  checkpoint = _legacy_checkpoint(tmp_path, protocol=protocol)
  with pytest.raises(RuntimeError, match="only accepts the archived return/release"):
    _binding(checkpoint)


@pytest.mark.parametrize("changed_field", ["model_fingerprint", "mechanics_version"])
def test_transfer_cannot_bypass_real_environment_drift(tmp_path, monkeypatch, changed_field):
  checkpoint = _legacy_checkpoint(tmp_path)
  from kaihand_tactile_env.shared import task_variants

  changed = current_variant_environment("usb-unplug")
  changed[changed_field] = "e" * 64 if changed_field == "model_fingerprint" else "different-physics"
  monkeypatch.setattr(task_variants, "current_variant_environment", lambda _task: changed)
  with pytest.raises(RuntimeError, match="cannot bypass mechanics/model"):
    _binding(checkpoint)


def test_transfer_cannot_target_future_unreviewed_completion_protocol(tmp_path, monkeypatch):
  checkpoint = _legacy_checkpoint(tmp_path)
  from kaihand_tactile_env.tasks.usb_unplug import config

  monkeypatch.setattr(config, "COMPLETION_PROTOCOL", "future-rule")
  with pytest.raises(RuntimeError, match="only targets held-clear-2s-v2"):
    _binding(checkpoint)


def test_transfer_requires_exact_legacy_prompt_not_just_a_usb_name(tmp_path):
  checkpoint = tmp_path / "10000"
  _bound_bulb_provenance(
    checkpoint,
    task="usb-unplug", config_name="pi05_kaihand_h50", action_horizon=50,
    contract_id=integrity.LEGACY_ACTION_CONTRACT,
  )
  with pytest.raises(RuntimeError, match="instruction differs"):
    _binding(checkpoint)


def test_transfer_does_not_bypass_bound_dataset_or_normalizer_authentication(tmp_path):
  checkpoint = _legacy_checkpoint(tmp_path)
  normalizer = checkpoint / "assets/normalizer_h50/norm_stats.json"
  document = json.loads(normalizer.read_text())
  document["norm_stats"]["actions"]["mean"][0] += 1
  normalizer.write_text(json.dumps(document), encoding="utf-8")
  with pytest.raises(RuntimeError, match="normalizer semantic SHA-256"):
    _binding(checkpoint)


def test_transfer_requires_snapshot_content_hash(tmp_path):
  checkpoint = _legacy_checkpoint(tmp_path)
  snapshot = checkpoint / "assets/conversion_manifest.snapshot.json"
  document = json.loads(snapshot.read_text())
  document["instruction"] = TASK_INSTRUCTIONS["usb-unplug"]
  snapshot.write_text(json.dumps(document), encoding="utf-8")
  with pytest.raises(RuntimeError, match="snapshot size|snapshot SHA-256"):
    _binding(checkpoint)


@pytest.mark.parametrize("upgrade", ["formal", "prompt", "reason", "recorded_environment", "protocol", "none"])
def test_transfer_manifest_cannot_upgrade_or_rewrite_its_diagnostic_scope(tmp_path, upgrade):
  checkpoint = _legacy_checkpoint(tmp_path)
  payload = _manifest(checkpoint)
  integrity.validate_shared_deployment_semantics(payload)
  if upgrade == "formal":
    payload.update(diagnostic_only=False, formal_metrics_valid=True)
  elif upgrade == "prompt":
    payload["instruction"] = TASK_INSTRUCTIONS["usb-unplug"]
  elif upgrade == "reason":
    payload["training_provenance"]["completion_protocol_transfer"]["reason"] = "formal"
  elif upgrade == "recorded_environment":
    payload["training_provenance"]["completion_protocol_transfer"]["training_source_environment"]["completion_protocol"] = "usb-unplug-held-clear-2s-v2"
  elif upgrade == "protocol":
    payload["training_provenance"]["completion_protocol_transfer"]["evaluation_completion_protocol"] = "unknown"
  else:
    payload["training_provenance"]["completion_protocol_transfer"] = None
  with pytest.raises(RuntimeError, match="USB protocol transfer"):
    integrity.validate_shared_deployment_semantics(payload)


def test_transfer_identity_authenticates_nested_protocol_record(tmp_path):
  checkpoint = _legacy_checkpoint(tmp_path)
  payload = _manifest(checkpoint)
  assert payload["deployment_id"] == _canonical(integrity.shared_deployment_identity(payload))
  payload["training_provenance"]["completion_protocol_transfer"]["reason"] = "changed"
  with pytest.raises(RuntimeError, match="deployment_id does not authenticate"):
    integrity.verify_pi05_deployment(payload)


def test_transfer_cannot_be_stripped_or_replaced_by_unbound_provenance(tmp_path):
  checkpoint = _legacy_checkpoint(tmp_path)
  payload = _manifest(checkpoint)
  payload["training_provenance"].pop("completion_protocol_transfer")
  with pytest.raises(RuntimeError):
    integrity.validate_shared_deployment_semantics(payload)
  payload["training_provenance"] = integrity.unbound_training_provenance("missing")
  payload["model_config_name"] = integrity.LEGACY_KAIHAND_CONFIG
  with pytest.raises(RuntimeError, match="unbound diagnostics cannot bypass"):
    integrity.validate_shared_deployment_semantics(payload)


def test_transfer_cannot_bypass_v2_binding_by_downgrading_manifest_schema(tmp_path):
  payload = _manifest(_legacy_checkpoint(tmp_path))
  payload["schema"] = integrity.SHARED_DEPLOYMENT_SCHEMA_V1
  with pytest.raises(RuntimeError, match="requires bound v2"):
    integrity.validate_shared_deployment_semantics(payload)


def test_server_runner_and_validity_share_transfer_identity_without_changing_prompt(tmp_path, monkeypatch):
  checkpoint = _legacy_checkpoint(tmp_path)
  payload = _manifest(checkpoint)
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/workcell"))
  server = load_script("serve_shared_task_pi05_policy.py")
  runner = load_script("run_shared_task_pi05_policy.py")

  assert server.load_manifest(path)[1] == payload
  assert runner.load_deployment(path, "usb-unplug")[1] == payload
  metadata = server.deployment_metadata(
    {**policy_metadata(), "action_horizon": 50, "joint_names": payload["joint_names"]}, path, payload,
  )
  assert runner.validate_server(metadata, payload, 50) == 50
  assert metadata["diagnostic_only"] is True
  assert metadata["formal_metrics_valid"] is False
  assert metadata["diagnostic_reasons"] == [integrity.USB_UNPLUG_TRANSFER_REASON]
  args = types.SimpleNamespace(task="usb-unplug", action_interpolation="linear", guard_mode="strict", diagnostic_relax_whiteboard_contact_guards=False)
  validity = runner._evaluation_validity_metadata(args, payload)
  assert validity["diagnostic_only"] is True
  assert validity["formal_metrics_valid"] is False
  bad_metadata = copy.deepcopy(metadata)
  bad_metadata["completion_protocol_transfer"]["evaluation_completion_protocol"] = "forged"
  with pytest.raises(RuntimeError, match="server deployment identity mismatch"):
    runner.validate_server(bad_metadata, payload, 50)


def test_prepare_dry_run_is_explicit_and_preserves_old_instruction(tmp_path, monkeypatch, capsys):
  checkpoint = _legacy_checkpoint(tmp_path)
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  binding = _binding(checkpoint)
  representation = integrity.action_representation_from_provenance(binding["record"])
  config = Config(
    name="pi05_kaihand_h50", model=Model(action_horizon=50),
    policy_metadata={**policy_metadata(), "action_horizon": 50, "action_semantics": representation},
  )
  training = types.ModuleType("openpi.training")
  training.config = types.SimpleNamespace(get_config=lambda _name: config)
  monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
  monkeypatch.setitem(sys.modules, "openpi.training", training)
  args = [
    "--task", "usb-unplug", "--checkpoint", str(checkpoint),
    "--output-dir", str(tmp_path / "deployment"), "--openpi-root", str(tmp_path), "--dry-run",
  ]
  with pytest.raises(RuntimeError, match="instruction differs"):
    prepare.main(args)
  assert prepare.main([*args, "--allow-usb-unplug-protocol-transfer-diagnostic"]) == 0
  result = json.loads(capsys.readouterr().out)
  assert result["valid"] is True
  assert result["prediction_horizon"] == 50
  assert result["instruction"] == integrity.USB_UNPLUG_LEGACY_INSTRUCTION
  assert result["diagnostic_only"] is True
  assert result["formal_metrics_valid"] is False
  assert not (tmp_path / "deployment").exists()


def test_prepare_rejects_transfer_on_other_task_or_legacy_unbound_bypass(tmp_path):
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  base = ["--checkpoint", str(tmp_path), "--output-dir", str(tmp_path / "unused")]
  with pytest.raises(SystemExit):
    prepare.parse_args([*base, "--task", "bulb-unscrew", "--allow-usb-unplug-protocol-transfer-diagnostic"])
  with pytest.raises(SystemExit):
    prepare.parse_args([*base, "--task", "usb-unplug", "--allow-usb-unplug-protocol-transfer-diagnostic", "--allow-legacy-unbound-diagnostic"])


def test_prepared_transfer_passes_full_small_fixture_verifier_and_child_integrity(tmp_path, monkeypatch):
  checkpoint = _legacy_checkpoint(tmp_path)
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  binding = _binding(checkpoint)
  representation = integrity.action_representation_from_provenance(binding["record"])
  config = Config(
    name="pi05_kaihand_h50", model=Model(action_horizon=50),
    policy_metadata={**policy_metadata(), "action_horizon": 50, "action_semantics": representation},
  )
  training = types.ModuleType("openpi.training")
  training.config = types.SimpleNamespace(get_config=lambda _name: config)
  monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
  monkeypatch.setitem(sys.modules, "openpi.training", training)
  project = tmp_path / "openpi"
  for relative in ("src/openpi/training/config.py", "packages/openpi-client/src/openpi_client/base_policy.py"):
    source = project / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# isolated test source", encoding="utf-8")
  evaluation = tmp_path / "evaluation"
  evaluation.mkdir()
  local_source = evaluation / "runner.py"
  local_source.write_text("# isolated test evaluator", encoding="utf-8")
  monkeypatch.setattr(prepare, "ROOT", evaluation)
  monkeypatch.setattr(prepare, "evaluation_source_paths", lambda _task: [local_source])
  output = tmp_path / "deployment"

  assert prepare.main([
    "--task", "usb-unplug", "--checkpoint", str(checkpoint),
    "--output-dir", str(output), "--openpi-root", str(project),
    "--allow-usb-unplug-protocol-transfer-diagnostic",
  ]) == 0

  manifest_path = output / "deployment_manifest.json"
  payload = json.loads(manifest_path.read_text())
  integrity.verify_pi05_deployment(payload, evaluation_root=evaluation)
  report = integrity.inspect_shared_pi05_runtime_integrity(
    manifest_path, payload, evaluation_root=evaluation,
    expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
  )
  assert report["status"] == "verified", report
  assert report["training_provenance"]["status"] == "verified"
  assert integrity.deployment_diagnostic_reasons(payload) == [integrity.USB_UNPLUG_TRANSFER_REASON]
