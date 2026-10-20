from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.pipeline.catalog import EVALUATION_RUNNERS
from kaihand_tactile_env.pipeline.deployment import (
  CapabilityError,
  adapter_for,
  build_prepare_plan,
  load_deployment_manifest,
  run_deployment,
  verify_deployment,
)
from kaihand_tactile_env.pipeline.runtime import RuntimeProfile
from kaihand_tactile_env.shared.tactile_deployment import (
  FAMILIES,
  JOINT_NAMES,
  SCHEMA,
  identity,
  sha256_file,
  validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
WORKCELL = ROOT / "scripts/workcell"


def _load_script(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def worker_module():
  return _load_script(
    "tactile_policy_worker_test", WORKCELL / "tactile_policy_worker.py"
  )


@pytest.fixture(scope="module")
def binding_module():
  sys.path.insert(0, str(WORKCELL))
  return _load_script(
    "tactile_policy_binding_test", WORKCELL / "tactile_policy_binding.py"
  )


@pytest.fixture(scope="module")
def evaluate_module():
  return _load_script(
    "tactile_evaluate_test", ROOT / "scripts/evaluate/evaluate.py"
  )


def _manifest(tmp_path: Path, family: str, *, execute_steps: int = 5):
  marker = tmp_path / f"{family.replace('+', '_')}.artifact"
  marker.write_bytes(b"frozen artifact")
  horizon, model_dim, offset = FAMILIES[family]
  payload = {
    "schema": SCHEMA,
    "task": "poker-draw",
    "model_family": family,
    "checkpoint_path": str(marker),
    "checkpoint_sha256": sha256_file(marker),
    "runtime": {
      "project_root": str(tmp_path),
      "python": sys.executable,
      "device": "cuda",
      "precision": "bfloat16",
    },
    "prediction_horizon": horizon,
    "model_action_dim": model_dim,
    "action_start_index": offset,
    "execute_steps": execute_steps,
    "action_dim": 27,
    "control_hz": 30,
    "joint_names": JOINT_NAMES,
    "instruction": "Draw the card.",
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": True,
      "tactile_shape": [5, 3, 7, 5],
    },
    "action_representation": "absolute-right-arm7-hand20-joints",
    "required_action_interpolation": "zoh",
    "controller_version": "poker-tactile-direct-joints-v1",
    "codec_version": (
      "ftp1-kaihand-poker-v1" if family == "ftp1" else "touchguide-native-v1"
    ),
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "files": {str(marker): sha256_file(marker)},
  }
  if family == "pi05+cpm":
    payload["guidance"] = {
      "num_steps": 10,
      "guidance_steps": 3,
      "scale": 1.0,
      "variant": "pi05+touchguidance",
    }
  payload["deployment_id"] = identity(payload)
  path = tmp_path / f"{family.replace('+', '_')}.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path, payload, marker


@pytest.mark.parametrize(
  ("family", "horizon", "model_dim", "offset", "maximum_k"),
  (("pi05+cpm", 30, 32, 0, 30), ("ftp1", 32, 120, 1, 31)),
)
def test_manifest_identity_inventory_and_native_horizon(
  tmp_path, family, horizon, model_dim, offset, maximum_k
):
  _, payload, marker = _manifest(tmp_path, family, execute_steps=maximum_k)
  validated = validate_manifest(payload, verify_files=True)
  assert (
    validated["prediction_horizon"],
    validated["model_action_dim"],
    validated["action_start_index"],
  ) == (horizon, model_dim, offset)

  tampered = copy.deepcopy(payload)
  tampered["execute_steps"] = maximum_k - 1
  with pytest.raises(ValueError, match="identity mismatch"):
    validate_manifest(tampered)

  too_long = copy.deepcopy(payload)
  too_long["execute_steps"] = maximum_k + 1
  too_long["deployment_id"] = identity(too_long)
  with pytest.raises(ValueError, match="future horizon"):
    validate_manifest(too_long)

  marker.write_bytes(b"changed artifact")
  with pytest.raises(ValueError, match="artifact/source changed"):
    validate_manifest(payload, verify_files=True)


def test_catalog_registers_task_scoped_ftp_but_keeps_cpm_poker_only():
  from kaihand_tactile_env.shared.tactile_task_contracts import SHARED_TACTILE_TASKS
  tactile_pairs = {
    pair for pair in EVALUATION_RUNNERS if pair[1] in {"pi05+cpm", "ftp1"}
  }
  assert tactile_pairs == {("poker-draw", "pi05+cpm"), ("poker-draw", "ftp1"),
                           *((task, "ftp1") for task in SHARED_TACTILE_TASKS)}
  for task in SHARED_TACTILE_TASKS:
    assert adapter_for(task, "ftp1").evaluate_backend == "evaluate_task_tactile_policy_batch.py"
    with pytest.raises(CapabilityError):
      adapter_for(task, "pi05+cpm")
  for family in ("pi05+cpm", "ftp1"):
    adapter = adapter_for("poker-draw", family)
    assert adapter.runtime_mode == "local-worker"
    assert adapter.evaluate_backend == "evaluate_poker_tactile_policy_batch.py"
  with pytest.raises(CapabilityError, match="no deployment adapter"):
    adapter_for("usb-insert", "pi05+cpm")


@pytest.mark.parametrize("family", ("pi05+cpm", "ftp1"))
def test_unified_deployment_entrypoints_plan_tactile_pipeline(tmp_path, family):
  manifest_path, payload, _ = _manifest(tmp_path, family, execute_steps=5)
  profile = RuntimeProfile(
    name=f"test-{family}",
    python=Path(sys.executable),
    project_root=tmp_path,
    env={"OMP_NUM_THREADS": "2"},
    config_path=tmp_path / "runtimes.toml",
    config_sha256="b" * 64,
  )

  resolved, loaded, adapter = load_deployment_manifest(manifest_path)
  assert resolved == manifest_path.resolve()
  assert loaded == payload
  assert (adapter.task, adapter.model_family, adapter.runtime_mode) == (
    "poker-draw", family, "local-worker"
  )

  verification = verify_deployment(
    manifest_path, profile=profile, execute_external=False
  )
  assert verification["status"] == "planned"
  verify_command = verification["external_validation"]["command"]
  assert verify_command[0] == sys.executable
  assert verify_command[-1] == "--validate-only"

  prepare = build_prepare_plan(
    "poker-draw",
    family,
    checkpoint=tmp_path / "checkpoint",
    output_dir=tmp_path / "prepared",
    profile=profile,
    backend_args=("--execute-steps", "5"),
  )
  assert prepare.command[prepare.command.index("--model-family") + 1] == family
  assert prepare.command[prepare.command.index("--model-python") + 1] == sys.executable
  assert prepare.command[prepare.command.index("--model-project") + 1] == str(tmp_path)

  run = run_deployment(
    manifest_path,
    output_dir=tmp_path / "run",
    profile=profile,
    num_trials=1,
    video_count=0,
    smoke_seconds=2.0,
    dry_run=True,
  )
  assert run["status"] == "planned"
  assert run["model_family"] == family
  assert run["runtime_mode"] == "local-worker"
  assert run["max_sim_seconds"] == 2.0
  assert run["evaluation"]["command"][0] == sys.executable


@pytest.mark.parametrize(("family", "execute_steps"), (("pi05+cpm", 4), ("ftp1", 7)))
def test_evaluate_defaults_to_manifest_bound_k(
  tmp_path, capsys, evaluate_module, family, execute_steps
):
  manifest_path, _, _ = _manifest(tmp_path, family, execute_steps=execute_steps)
  assert evaluate_module.main([
    "--task", "poker-draw",
    "--model-family", family,
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(tmp_path / "output"),
    "--num-trials", "1",
    "--video-count", "0",
    "--dry-run",
  ]) == 0
  plan = json.loads(capsys.readouterr().out)
  assert plan["execute_steps"] == execute_steps
  command = plan["command"]
  assert command[command.index("--execute-steps") + 1] == str(execute_steps)


def test_ftp_horizon_means_31_future_actions(tmp_path, capsys, evaluate_module):
  manifest_path, payload, _ = _manifest(tmp_path, "ftp1", execute_steps=31)
  args = [
    "--task", "poker-draw",
    "--model-family", "ftp1",
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(tmp_path / "output"),
    "--num-trials", "1",
    "--video-count", "0",
    "--execute-steps", "horizon",
    "--dry-run",
  ]
  assert evaluate_module.main(args) == 0
  assert json.loads(capsys.readouterr().out)["execute_steps"] == 31
  assert evaluate_module._resolve_execution_modes(("horizon",), payload)[0]["value"] == 31
  with pytest.raises(ValueError, match="exceeds deployment prediction_horizon"):
    evaluate_module._resolve_execution_modes((32,), payload)


def test_evaluate_rejects_tactile_k_override(tmp_path, evaluate_module):
  manifest_path, _, _ = _manifest(tmp_path, "ftp1", execute_steps=1)
  with pytest.raises(ValueError, match="execute_steps is frozen"):
    evaluate_module.main([
      "--task", "poker-draw",
      "--model-family", "ftp1",
      "--deployment-manifest", str(manifest_path),
      "--output-dir", str(tmp_path / "output"),
      "--num-trials", "1",
      "--video-count", "0",
      "--execute-steps", "5",
      "--dry-run",
    ])


def test_worker_wire_round_trip_is_recursive_and_copies(worker_module):
  array = np.arange(12, dtype=np.float32).reshape(3, 4)
  value = {
    "array": array,
    "nested": [np.uint8(7), np.array([True, False], dtype=np.bool_)],
  }
  restored = worker_module.unpack(worker_module.pack(value))
  np.testing.assert_array_equal(restored["array"], array)
  np.testing.assert_array_equal(restored["nested"][1], [True, False])
  assert restored["nested"][0] == 7
  assert restored["array"].flags.writeable
  with pytest.raises(ValueError, match="object arrays"):
    worker_module.pack(np.array([object()], dtype=object))
  with pytest.raises(ValueError, match="unsupported wire dtype"):
    worker_module.unpack({"__ndarray__": "YQ==", "dtype": "|S1", "shape": [1]})


def test_binding_handshake_binds_worker_to_manifest(
  tmp_path, binding_module
):
  manifest_path, payload, _ = _manifest(tmp_path, "ftp1", execute_steps=5)
  binding = binding_module.PokerTactileBinding()
  binding.load_manifest(manifest_path)
  metadata = {
    key: payload[key]
    for key in (
      "model_family",
      "task",
      "prediction_horizon",
      "action_start_index",
      "action_dim",
      "control_hz",
      "deployment_id",
      "model_action_dim",
      "joint_names",
      "checkpoint_sha256",
      "execute_steps",
    )
  }
  metadata.update(
    deployment_manifest_sha256=sha256_file(manifest_path),
    tactile_sent_to_model=True,
  )
  assert binding.validate_server(metadata, 5) == 32
  for key in (
    "model_family",
    "task",
    "prediction_horizon",
    "action_start_index",
    "action_dim",
    "control_hz",
    "deployment_id",
    "model_action_dim",
    "joint_names",
    "checkpoint_sha256",
    "execute_steps",
    "deployment_manifest_sha256",
    "tactile_sent_to_model",
  ):
    changed = copy.deepcopy(metadata)
    changed[key] = False if key == "tactile_sent_to_model" else "tampered"
    with pytest.raises(RuntimeError):
      binding.validate_server(changed, 5)
  with pytest.raises(ValueError, match="execute_steps is frozen"):
    binding.validate_server(metadata, 4)


def test_binding_finalizes_native_validity_without_changing_task_result(
  binding_module,
):
  binding = binding_module.PokerTactileBinding()
  report = {
    "status": "task_not_completed",
    "physics_valid": True,
    "task_success": False,
    "stop_reason": "time_limit",
  }
  binding.finalize_report(report)
  assert report["native_success"] is False
  assert report["valid_trial"] is True
  assert report["trial_validity"] == "valid"
  assert report["diagnostic_only"] is True
  assert report["formal_metrics_valid"] is False

