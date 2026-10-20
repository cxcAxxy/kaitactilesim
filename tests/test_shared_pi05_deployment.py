from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.shared.evaluation_guards import (
  configure_evaluation_guards,
  evaluation_guard_report,
)
from kaihand_tactile_env.shared.policy_tasks import (
  TASK_INSTRUCTIONS,
  TaskPolicyAdapter,
)

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
  name: str = "pi05_kaihand"
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
  assert prepare.SCHEMA == "shared_task_pi05_deployment_v2"
  assert server.SCHEMAS == {
    "shared_task_pi05_deployment_v1",
    "shared_task_pi05_deployment_v2",
  }
  assert prepare.TASK_INSTRUCTIONS == TASK_INSTRUCTIONS
  assert server.TASK_INSTRUCTIONS == TASK_INSTRUCTIONS
  assert (
    prepare.validate_train_config(Config(policy_metadata=policy_metadata()))
    == policy_metadata()
  )


def test_prepare_h50_requires_matching_config_model_and_metadata():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  metadata = {**policy_metadata(), "action_horizon": 50}
  config = Config(
    name="pi05_kaihand_h50",
    model=Model(action_horizon=50),
    policy_metadata=metadata,
  )

  assert prepare.validate_train_config(config) == metadata
  with pytest.raises(RuntimeError, match="contract mismatch"):
    prepare.validate_train_config(dataclasses.replace(config, name="pi05_kaihand"))
  with pytest.raises(RuntimeError, match="model horizon"):
    prepare.validate_train_config(dataclasses.replace(config, model=Model()))


def test_prepare_accepts_dedicated_bulb_h50_config():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  metadata = {**policy_metadata(), "action_horizon": 50}
  config = Config(
    name="pi05_kaihand_bulb_controller_target_h50_v1",
    model=Model(action_horizon=50),
    policy_metadata=metadata,
  )

  assert prepare.validate_train_config(config) == metadata


def test_prepare_h50_dry_run_selects_provenance_normalizer(
  monkeypatch, tmp_path, capsys
):
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  checkpoint = tmp_path / "24999"
  assets = checkpoint / "assets"
  (assets / "normalizer_h50").mkdir(parents=True)
  (assets / "normalizer_h50/norm_stats.json").write_text("{}", encoding="utf-8")
  (assets / "training_provenance.json").write_text("{}", encoding="utf-8")
  (assets / "conversion_manifest.snapshot.json").write_text("{}", encoding="utf-8")
  (checkpoint / "params").mkdir()
  (checkpoint / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
  metadata = {**policy_metadata(), "action_horizon": 50}
  config = Config(
    name="pi05_kaihand_h50",
    model=Model(action_horizon=50),
    policy_metadata=metadata,
  )
  record = {
    "task": "whiteboard-wipe",
    "config_name": config.name,
    "normalizer": {"asset_id": "normalizer_h50"},
  }
  training = types.ModuleType("openpi.training")

  def get_config(name):
    assert name == config.name
    return config

  training.config = types.SimpleNamespace(get_config=get_config)
  monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
  monkeypatch.setitem(sys.modules, "openpi.training", training)
  monkeypatch.setattr(
    prepare,
    "load_checkpoint_training_provenance",
    lambda _checkpoint: {"status": "bound", "record": record},
  )
  monkeypatch.setattr(
    prepare,
    "action_representation_from_provenance",
    lambda _record: metadata["action_semantics"],
  )

  assert (
    prepare.main(
      [
        "--task",
        "whiteboard-wipe",
        "--checkpoint",
        str(checkpoint),
        "--openpi-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "deployment"),
        "--dry-run",
      ]
    )
    == 0
  )
  result = json.loads(capsys.readouterr().out)
  assert result["prediction_horizon"] == 50
  assert result["normalizer_asset_id"] == "normalizer_h50"
  assert not (tmp_path / "deployment").exists()


def test_server_accepts_bound_h50_config_without_changing_execution_hint(
  monkeypatch, tmp_path
):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "24999"
  payload = manifest(checkpoint)
  payload.update(
    schema="shared_task_pi05_deployment_v2",
    task="whiteboard-wipe",
    model_config_name="pi05_kaihand_h50",
    prediction_horizon=50,
    instruction=TASK_INSTRUCTIONS["whiteboard-wipe"],
  )
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")
  # Asset authentication and horizon provenance are covered by integrity tests.
  monkeypatch.setattr(server, "validate_shared_deployment_semantics", lambda _: None)

  _, loaded = server.load_manifest(path)
  metadata = {**policy_metadata(), "action_horizon": 50}
  config = Config(
    name="pi05_kaihand_h50",
    model=Model(action_horizon=50),
    policy_metadata=metadata,
  )
  assert server.validate_runtime_config(config, loaded) == metadata
  assert loaded["suggested_execute_steps"] == 8


def test_server_reads_bound_checkpoint_normalization_mode():
  server = load_script("serve_shared_task_pi05_policy.py")
  payload = {
    "training_provenance": {
      "status": "bound",
      "record": {"normalizer": {"use_quantile_norm": False}},
    }
  }

  assert server.checkpoint_use_quantile_norm(payload) is False
  payload["training_provenance"]["record"]["normalizer"].pop("use_quantile_norm")
  with pytest.raises(RuntimeError, match="use_quantile_norm"):
    server.checkpoint_use_quantile_norm(payload)
  assert server.checkpoint_use_quantile_norm({}) is None


@pytest.mark.parametrize(
  ("task", "package"),
  [
    ("bulb-screw", "bulb_screw"),
    ("install-ram", "install_ram"),
    ("vase-wipe", "vase_wipe"),
    ("whiteboard-wipe", "whiteboard_wipe"),
    ("whiteboard-wipe-horizontal", "whiteboard_wipe_horizontal"),
    ("whiteboard-wipe-horizontal-line", "whiteboard_wipe_horizontal_line"),
    ("sponge-grasp", "sponge_grasp"),
  ],
)
def test_prepare_freezes_task_local_evaluation_semantics(task, package):
  prepare = load_script("prepare_shared_task_pi05_deployment.py")

  paths = prepare.evaluation_source_paths(task)
  relative = [path.relative_to(ROOT).as_posix() for path in paths]

  assert len(paths) == len(set(paths))
  assert paths == prepare.evaluation_source_paths(task)
  assert f"src/kaihand_tactile_env/tasks/{package}/config.py" in relative
  assert f"src/kaihand_tactile_env/tasks/{package}/task.py" in relative
  assert f"src/kaihand_tactile_env/tasks/{package}/scene.xml" in relative


def test_prepare_freezes_whiteboard_cleaning_model_and_sponge_dependency():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")

  whiteboard = {
    path.relative_to(ROOT).as_posix()
    for path in prepare.evaluation_source_paths("whiteboard-wipe")
  }
  sponge = {
    path.relative_to(ROOT).as_posix()
    for path in prepare.evaluation_source_paths("sponge-grasp")
  }

  assert "src/kaihand_tactile_env/tasks/whiteboard_wipe/cleaning.py" in whiteboard
  assert "src/kaihand_tactile_env/tasks/vase_wipe/sponge.py" in sponge


def test_line_prepare_freezes_both_horizontal_and_shared_whiteboard_dependencies():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")
  relative = {
    path.relative_to(ROOT).as_posix()
    for path in prepare.evaluation_source_paths("whiteboard-wipe-horizontal-line")
  }
  for package in (
    "whiteboard_wipe_horizontal_line", "whiteboard_wipe_horizontal", "whiteboard_wipe",
  ):
    assert f"src/kaihand_tactile_env/tasks/{package}/config.py" in relative
    assert f"src/kaihand_tactile_env/tasks/{package}/task.py" in relative
  assert "src/kaihand_tactile_env/tasks/whiteboard_wipe/cleaning.py" in relative


def test_prepare_freezes_full_openpi_server_and_client_source_closure():
  prepare = load_script("prepare_shared_task_pi05_deployment.py")

  paths = {
    path.relative_to(ROOT.parent / "openpi").as_posix()
    for path in prepare.openpi_source_paths(ROOT.parent / "openpi")
  }

  assert {
    "src/openpi/training/config.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/policies/kaihand_policy.py",
    "src/openpi/serving/websocket_policy_server.py",
    "packages/openpi-client/src/openpi_client/base_policy.py",
    "packages/openpi-client/src/openpi_client/msgpack_numpy.py",
    "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
  } <= paths


def test_shared_batch_freezes_content_addressed_read_only_manifest(tmp_path):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  source = tmp_path / "source.json"
  payload = {"task": "bulb-screw", "deployment_id": "test"}
  source.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  output = tmp_path / "output"
  output.mkdir()

  frozen, digest, size = batch.freeze_deployment_manifest(source, payload, output)

  expected_digest = hashlib.sha256(source.read_bytes()).hexdigest()
  assert digest == expected_digest
  assert size == source.stat().st_size
  assert frozen.name == f"deployment_manifest.{expected_digest}.json"
  assert frozen.parent == output / "deployment_snapshot"
  assert frozen.read_bytes() == source.read_bytes()
  assert stat.S_IMODE(frozen.stat().st_mode) == 0o444
  assert stat.S_IMODE(frozen.parent.stat().st_mode) == 0o555
  assert batch.frozen_manifest_report(frozen, digest)["status"] == "verified"

  source.write_text('{"changed": true}', encoding="utf-8")
  assert batch.frozen_manifest_report(frozen, digest)["status"] == "verified"


def test_shared_batch_detects_frozen_manifest_mutation(tmp_path):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  source = tmp_path / "source.json"
  payload = {"task": "bulb-screw", "deployment_id": "test"}
  source.write_text(json.dumps(payload), encoding="utf-8")
  output = tmp_path / "output"
  output.mkdir()
  frozen, digest, _ = batch.freeze_deployment_manifest(source, payload, output)
  frozen.chmod(0o644)
  writable = batch.frozen_manifest_report(frozen, digest)
  assert writable["status"] == "mismatch"
  assert "writable" in writable["error"]
  frozen.write_text('{"changed": true}', encoding="utf-8")

  report = batch.frozen_manifest_report(frozen, digest)

  assert report["status"] == "mismatch"
  assert report["actual_sha256"] != digest


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


def test_shared_batch_only_requires_source_snapshot_for_pi05_or_declared_manifest():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  assert batch._enforce_source_integrity("pi05", {})
  assert not batch._enforce_source_integrity("egosteer", {})
  assert not batch._enforce_source_integrity("egotouch", {})
  assert batch._enforce_source_integrity("egosteer", {"evaluation_source_files": []})


def test_shared_batch_preserves_explicit_egl_vendor(monkeypatch, tmp_path):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  vendor = tmp_path / "nvidia.json"
  vendor.write_text("{}", encoding="utf-8")
  monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))

  environment = batch._environment()

  assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(vendor)


def test_shared_batch_counts_lowercase_collision_guard_as_valid_failure():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  reason, valid = batch.classify(
    {"status": "error", "error": "RuntimeError: hand struck the whiteboard"}
  )

  assert (reason, valid) == ("collision_guard", True)


def test_shared_batch_counts_whiteboard_evaluation_force_guard_as_valid_failure():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  reason, valid = batch.classify(
    {
      "status": "error",
      "error": (
        "RuntimeError: whiteboard normal force exceeded evaluation guard threshold"
      ),
    }
  )

  assert (reason, valid) == ("force_guard", True)


def test_shared_batch_counts_sponge_integrity_guard_as_valid_failure():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  reason, valid = batch.classify(
    {"status": "error", "error": "RuntimeError: Sponge element inverted"}
  )

  assert (reason, valid) == ("sponge_integrity_guard", True)


def test_shared_batch_counts_sponge_hand_table_penetration_as_valid_failure():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  reason, valid = batch.classify(
    {"status": "error", "error": "RuntimeError: Hand/table penetration at 2.0s"}
  )

  assert (reason, valid) == ("penetration_guard", True)


def test_shared_batch_uses_structured_guard_outcomes_before_error_text():
  batch = load_script("evaluate_shared_task_policy_batch.py")

  assert batch.classify({
    "status": "task_not_completed",
    "stop_reason": "time_limit",
    "physics_valid": True,
    "guard_events": [{"kind": "force_guard"}],
  }) == ("time_limit", True)
  assert batch.classify({
    "status": "error",
    "stop_reason": "guard",
    "physics_valid": True,
    "guard_events": [{"kind": "force_guard"}],
    "error": "unrecognized wording",
  }) == ("force_guard", True)
  assert batch.classify({
    "status": "error",
    "stop_reason": "fatal_error",
    "physics_valid": False,
    "guard_events": [{"kind": "sponge_integrity_guard"}],
  }) == ("sponge_integrity_guard", False)
  assert batch.classify({
    "status": "success",
    "task_success": True,
    "physics_valid": True,
    "record_error": "RuntimeError: encoder failed",
  }) == ("record_error", False)
  assert batch.classify({
    "status": "error",
    "stop_reason": "fatal_error",
    "physics_valid": True,
    "error": "renderer force exceeded an internal limit",
  }) == ("fatal_error", False)


def test_shared_batch_invalidates_formal_metrics_after_recording_failure():
  batch = load_script("evaluate_shared_task_policy_batch.py")
  summary = batch._batch_summary(
    {"formal_metrics_valid": True},
    1,
    [{
      "valid_trial": False,
      "success": True,
      "termination": "record_error",
      "formal_metrics_valid": False,
      "action_clipping": None,
    }],
    0.0,
    {"status": "verified"},
  )

  assert summary["formal_metrics_valid"] is False
  assert summary["formal_success_rate"] is None
  assert summary["successes"] == 0
  assert summary["task_successes"] == 1


def _whiteboard_guard_adapter(*, relaxed: bool, board_force: float, hand_force: float):
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = "whiteboard-wipe"
  adapter.diagnostic_relax_whiteboard_contact_guards = relaxed
  adapter.whiteboard_force_guard_observed = False
  adapter.whiteboard_collision_guard_observed = False
  adapter.state = types.SimpleNamespace(board_normal_force_n=board_force)
  adapter.simulation = types.SimpleNamespace(
    data=types.SimpleNamespace(
      time=0.0,
      qpos=np.zeros(1),
      qvel=np.zeros(1),
      warning=(),
    ),
    direct_hand_board_force=hand_force,
    object_pose=lambda _name: np.array([0.0, 0.0, 0.7]),
  )
  return adapter


def test_whiteboard_contact_guards_remain_enabled_by_default():
  adapter = _whiteboard_guard_adapter(
    relaxed=False,
    board_force=8.01,
    hand_force=0.0,
  )

  with pytest.raises(RuntimeError, match="normal force exceeded"):
    adapter._guard()
  assert adapter.whiteboard_force_guard_observed is True


def test_whiteboard_force_guard_is_strictly_above_threshold():
  adapter = _whiteboard_guard_adapter(
    relaxed=False,
    board_force=8.0,
    hand_force=0.0,
  )

  adapter._guard()

  assert adapter.whiteboard_force_guard_observed is False


def test_whiteboard_diagnostic_mode_observes_contact_guards_without_stopping():
  adapter = _whiteboard_guard_adapter(
    relaxed=True,
    board_force=8.01,
    hand_force=0.51,
  )

  adapter._guard()

  assert adapter.whiteboard_force_guard_observed is True
  assert adapter.whiteboard_collision_guard_observed is True
  assert adapter.metadata()["diagnostic_only"] is True
  assert adapter.metadata()["formal_metrics_valid"] is False
  assert adapter.metadata()["cleaning_model_has_upper_normal_force_gate"] is False
  assert adapter.metadata()["cleaning_can_accumulate_above_board_force_guard"] is True


def test_whiteboard_diagnostic_mode_keeps_object_drop_guard():
  adapter = _whiteboard_guard_adapter(
    relaxed=True,
    board_force=8.01,
    hand_force=0.51,
  )
  adapter.simulation.object_pose = lambda _name: np.array([0.0, 0.0, 0.49])

  with pytest.raises(RuntimeError, match="eraser fell below"):
    adapter._guard()


def test_observe_mode_records_all_same_step_whiteboard_guards():
  adapter = _whiteboard_guard_adapter(
    relaxed=False,
    board_force=8.01,
    hand_force=0.51,
  )
  adapter.simulation.object_pose = lambda _name: np.array([0.0, 0.0, 0.49])
  configure_evaluation_guards(adapter.simulation, "observe")

  adapter._guard()

  events = evaluation_guard_report(adapter.simulation)["guard_events"]
  assert {event["kind"] for event in events} == {
    "force_guard",
    "collision_guard",
    "object_dropped",
  }
  assert all(event["first_sim_time_s"] == 0.0 for event in events)


def test_shared_pi05_runner_restricts_contact_guard_relaxation_to_whiteboard(
  monkeypatch,
):
  monkeypatch.syspath_prepend(str(ROOT / "scripts/workcell"))
  runner = load_script("run_shared_task_pi05_policy.py")
  common = [
    "--server",
    "ws://127.0.0.1:18784",
    "--deployment-manifest",
    "/tmp/deployment.json",
    "--output-dir",
    "/tmp/output",
    "--seed",
    "0",
    "--diagnostic-relax-whiteboard-contact-guards",
  ]

  args = runner.parse_args(["--task", "whiteboard-wipe", *common])
  assert args.diagnostic_relax_whiteboard_contact_guards is True

  with pytest.raises(SystemExit):
    runner.parse_args(["--task", "bulb-screw", *common])


@pytest.mark.parametrize(
  ("script", "family_args"),
  [
    (
      "run_shared_task_egosteer_policy.py",
      ["--server", "ws://127.0.0.1:18784"],
    ),
    (
      "run_shared_task_egotouch_policy.py",
      ["--model-python", "/tmp/python", "--model-project", "/tmp/model"],
    ),
  ],
)
def test_shared_task_runners_default_to_observe_and_allow_strict(
  monkeypatch, script, family_args
):
  monkeypatch.syspath_prepend(str(ROOT / "scripts/workcell"))
  runner = load_script(script)
  common = [
    "--task", "vase-wipe",
    "--deployment-manifest", "/tmp/deployment.json",
    "--output-dir", "/tmp/output",
    "--seed", "0",
    *family_args,
  ]

  assert runner.parse_args(common).guard_mode == "observe"
  assert runner.parse_args([*common, "--guard-mode", "strict"]).guard_mode == "strict"


def test_shared_batch_requires_complete_requested_reference_comparison(tmp_path):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  trial = tmp_path / "seed_000"
  review = trial / "review"
  review.mkdir(parents=True)
  reference = (tmp_path / "reference").resolve()
  for name in (*batch._REVIEW_ARTIFACTS, *batch._COMPARISON_ARTIFACTS):
    (review / name).write_bytes(b"artifact")
  (review / "review.json").write_text(
    json.dumps(
      {
        "model_input_cameras_displayed": ["head", "right_wrist"],
        "reference_dataset": str(reference),
        "reference_episode_index": 0,
        "comparison_plots": {"status": "ok"},
      }
    ),
    encoding="utf-8",
  )
  deployment = {"observation_contract": {"cameras": ["head", "right_wrist"]}}

  assert batch.validate_recorded_review(trial, deployment, reference, 0) is None

  (review / "evaluation_right_wrist_state.png").unlink()
  assert batch.validate_recorded_review(trial, deployment, reference, 0) == (
    "incomplete_comparison_artifact"
  )


def test_shared_batch_rejects_unavailable_requested_reference_comparison(tmp_path):
  batch = load_script("evaluate_shared_task_policy_batch.py")
  trial = tmp_path / "seed_000"
  review = trial / "review"
  review.mkdir(parents=True)
  reference = (tmp_path / "reference").resolve()
  for name in (*batch._REVIEW_ARTIFACTS, *batch._COMPARISON_ARTIFACTS):
    (review / name).write_bytes(b"artifact")
  (review / "review.json").write_text(
    json.dumps(
      {
        "model_input_cameras_displayed": ["head", "right_wrist"],
        "reference_dataset": str(reference),
        "reference_episode_index": 0,
        "comparison_plots": {"status": "reference_unavailable"},
      }
    ),
    encoding="utf-8",
  )
  deployment = {"observation_contract": {"cameras": ["head", "right_wrist"]}}

  assert batch.validate_recorded_review(trial, deployment, reference, 0) == (
    "reference_comparison_unavailable"
  )


def test_bulb_launcher_forwards_and_verifies_reference_dataset():
  script = (ROOT / "scripts/workcell/run_bulb_pi05_0920_evaluation.sh").read_text(
    encoding="utf-8"
  )

  assert "BULB_PI05_REFERENCE_DATASET" in script
  assert '--reference-dataset "$REFERENCE_DATASET"' in script
  assert 'comparison.get("status") != "ok"' in script
  assert 'grep -q "WARMUP_OK" "$log"' in script
  for artifact in (
    "evaluation_right_wrist_state.png",
    "evaluation_right_hand_actuated_dof.png",
    "evaluation_right_fingertip_tactile.png",
  ):
    assert artifact in script


def test_sponge_launcher_freezes_0924_contract_and_steps():
  script = (ROOT / "scripts/workcell/run_sponge_pi05_0924_evaluation.sh").read_text(
    encoding="utf-8"
  )

  assert "SHARED_PI05_TASK=sponge-grasp" in script
  assert "sponge-grasp_bs128_25k" in script
  assert "SPONGE_PI05_REFERENCE_DATASET" in script
  assert "set -- 20000 24999" in script
  assert "--check --task sponge-grasp --render-backend hardware" in script


def test_whiteboard_launcher_freezes_0923_contract_and_steps():
  script = (ROOT / "scripts/workcell/run_whiteboard_pi05_0923_evaluation.sh").read_text(
    encoding="utf-8"
  )

  assert "SHARED_PI05_TASK=whiteboard-wipe" in script
  assert "whiteboard-wipe_bs128_25k" in script
  assert "WHITEBOARD_PI05_REFERENCE_DATASET" in script
  assert "set -- 20000 24999" in script
  assert "--check --task whiteboard-wipe --render-backend hardware" in script
  assert "WHITEBOARD_PI05_EXECUTE_STEPS" in script
  assert "WHITEBOARD_PI05_DIAGNOSTIC_RELAX_CONTACT_GUARDS" in script

  shared = (ROOT / "scripts/workcell/run_bulb_pi05_0920_evaluation.sh").read_text(
    encoding="utf-8"
  )
  assert '--execute-steps "$EXECUTE_STEPS"' in shared
  assert "--diagnostic-relax-whiteboard-contact-guards" in shared


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


def test_server_warmup_compiles_frozen_contract_and_restores_rng(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  payload = manifest(tmp_path / "20000")
  original_rng = object()

  class Policy:
    _rng = original_rng

    def infer(self, request):
      self.request = request
      self._rng = object()
      return {"actions": np.zeros((30, 27), dtype=np.float32)}

  policy = Policy()
  times = iter((10.0, 43.25))

  result = server.warmup_policy(policy, payload, monotonic=lambda: next(times))

  assert policy._rng is original_rng
  assert set(policy.request) == {
    "head_image",
    "right_wrist_image",
    "state",
    "prompt",
  }
  assert policy.request["head_image"].shape == (240, 320, 3)
  assert policy.request["head_image"].dtype == np.uint8
  assert policy.request["right_wrist_image"].shape == (240, 320, 3)
  assert policy.request["right_wrist_image"].dtype == np.uint8
  assert policy.request["state"].shape == (27,)
  assert policy.request["state"].dtype == np.float32
  assert policy.request["prompt"] == TASK_INSTRUCTIONS["bulb-screw"]
  assert result == {
    "completed": True,
    "duration_s": 33.25,
    "request_image_shape_hwc": [240, 320, 3],
    "state_dim": 27,
    "action_shape": [30, 27],
    "rng_restored": True,
  }


@pytest.mark.parametrize(
  ("actions", "message"),
  [
    (np.zeros((29, 27), dtype=np.float32), "must have shape"),
    (np.full((30, 27), np.nan, dtype=np.float32), "finite numeric"),
  ],
)
def test_server_warmup_rejects_invalid_actions_and_restores_rng(
  tmp_path, actions, message
):
  server = load_script("serve_shared_task_pi05_policy.py")
  payload = manifest(tmp_path / "20000")
  original_rng = object()

  class Policy:
    _rng = original_rng

    def infer(self, _request):
      self._rng = object()
      return {"actions": actions}

  policy = Policy()
  with pytest.raises(RuntimeError, match=message):
    server.warmup_policy(policy, payload)
  assert policy._rng is original_rng


def test_server_warmup_restores_rng_when_inference_fails(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  payload = manifest(tmp_path / "20000")
  original_rng = object()

  class Policy:
    _rng = original_rng

    def infer(self, _request):
      self._rng = object()
      raise RuntimeError("compile failed")

  policy = Policy()
  with pytest.raises(RuntimeError, match="compile failed"):
    server.warmup_policy(policy, payload)
  assert policy._rng is original_rng


def test_server_main_finishes_warmup_before_listening(monkeypatch, tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  payload = manifest(tmp_path / "20000")
  manifest_path = tmp_path / "deployment_manifest.json"
  events = []
  create_policy_kwargs = {}

  class Policy:
    _rng = object()

    def infer(self, _request):
      events.append("warmup")
      return {"actions": np.zeros((30, 27), dtype=np.float32)}

  class WebsocketPolicyServer:
    def __init__(self, *, policy, host, port, metadata):
      events.append("server_init")
      assert isinstance(policy, Policy)
      assert (host, port) == ("0.0.0.0", 18783)
      assert metadata["server_warmup"]["completed"] is True

    def serve_forever(self):
      events.append("listen")

  openpi = types.ModuleType("openpi")
  policies = types.ModuleType("openpi.policies")
  serving = types.ModuleType("openpi.serving")
  training = types.ModuleType("openpi.training")

  def create_trained_policy(*_args, **kwargs):
    create_policy_kwargs.update(kwargs)
    events.append("create_policy")
    return Policy()

  policies.policy_config = types.SimpleNamespace(
    create_trained_policy=create_trained_policy
  )
  serving.websocket_policy_server = types.SimpleNamespace(
    WebsocketPolicyServer=WebsocketPolicyServer
  )
  training.config = types.SimpleNamespace(get_config=lambda _name: object())
  monkeypatch.setitem(sys.modules, "openpi", openpi)
  monkeypatch.setitem(sys.modules, "openpi.policies", policies)
  monkeypatch.setitem(sys.modules, "openpi.serving", serving)
  monkeypatch.setitem(sys.modules, "openpi.training", training)
  monkeypatch.setattr(server, "load_manifest", lambda _path: (manifest_path, payload))
  monkeypatch.setattr(server, "verify_pi05_deployment", lambda *_args, **_kwargs: None)
  monkeypatch.setattr(
    server, "bind_checkpoint_assets", lambda config, _manifest: config
  )
  monkeypatch.setattr(
    server, "validate_runtime_config", lambda _config, _manifest: policy_metadata()
  )

  server.main(["--deployment-manifest", str(manifest_path)])

  assert events == ["create_policy", "warmup", "server_init", "listen"]
  assert create_policy_kwargs["expected_use_quantile_norm"] is None


def test_server_rejects_usb_manifest(tmp_path):
  server = load_script("serve_shared_task_pi05_policy.py")
  checkpoint = tmp_path / "20000"
  payload = manifest(checkpoint)
  payload.update(schema="usb_pi05_deployment_v1", task="usb-insert")
  path = tmp_path / "deployment_manifest.json"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(RuntimeError, match="expected one of"):
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
