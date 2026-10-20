import json
import os
import signal
import sys
from pathlib import Path
from runpy import run_path

import pytest
from kaihand_tactile_env.pipeline import deployment
from kaihand_tactile_env.pipeline.catalog import TASKS
from kaihand_tactile_env.pipeline.deployment import (
  CapabilityError,
  DeploymentError,
  ProcessPlan,
  adapter_catalog,
  adapter_for,
  build_evaluation_plan,
  build_prepare_plan,
  build_serve_plan,
  load_deployment_manifest,
  run_deployment,
  validate_server_metadata,
  write_pipeline_deployment,
)
from kaihand_tactile_env.pipeline.runtime import RuntimeProfile


def frozen_manifest(
  root: Path,
  *,
  task: str = "pick-place",
  family: str = "pi0.5",
  schema: str = "pickplace_pi05_deployment_v1",
) -> Path:
  path = root / "deployment_manifest.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({
    "schema": schema,
    "task": task,
    "model_family": family,
    "deployment_id": "deployment-123",
    "checkpoint_path": str(root / "checkpoint"),
    "checkpoint_sha256": "a" * 64,
    "prediction_horizon": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"type": "absolute_joint_target"},
    "model_project_path": str(root / "openpi"),
  }))
  return path


def runtime_profile(tmp_path: Path, *, project: str = "model") -> RuntimeProfile:
  return RuntimeProfile(
    name="test",
    python=tmp_path / "venv/bin/python",
    project_root=tmp_path / project,
    env={"CUDA_VISIBLE_DEVICES": "2", "PYTHONPATH": "/user/runtime/src"},
    config_path=tmp_path / "runtimes.toml",
    config_sha256="b" * 64,
  )


def test_all_eight_pi05_tasks_have_complete_deployment_stages():
  assert set(TASKS) == {
    task for task, family in deployment.ADAPTERS if family == "pi05"
  }
  for task in TASKS:
    adapter = adapter_for(task, "pi05")
    assert adapter.runtime_mode == "websocket-sidecar"
    assert adapter.capabilities == {
      "prepare": True,
      "verify": True,
      "serve": True,
      "run": True,
      "evaluate": True,
    }


def test_registry_is_honest_about_worker_embedded_and_incomplete_backends():
  egotouch = adapter_for("pick-place", "egotouch")
  assert egotouch.runtime_mode == "local-worker"
  assert egotouch.capabilities["prepare"] is True
  assert egotouch.capabilities["serve"] is False

  trex = adapter_for("poker-draw", "pi05+trex")
  assert trex.runtime_mode == "embedded"
  assert trex.capabilities["prepare"] is False
  assert trex.capabilities["verify"] is False
  assert trex.capabilities["serve"] is False
  assert trex.capabilities["run"] is True

  shared_egosteer = adapter_for("bulb-screw", "egosteer")
  assert shared_egosteer.capabilities["prepare"] is False
  assert shared_egosteer.capabilities["serve"] is False
  assert shared_egosteer.capabilities["evaluate"] is True


def test_catalog_exposes_each_stage_and_backend():
  row = next(
    item for item in adapter_catalog()
    if (item["task"], item["model_family"]) == ("whiteboard-wipe", "pi05")
  )
  assert row["prepare_backend"] == "prepare_shared_task_pi05_deployment.py"
  assert row["serve_backend"] == "serve_shared_task_pi05_policy.py"
  assert row["evaluate_backend"] == "evaluate_shared_task_policy_batch.py"
  assert row["capabilities"]["verify"] is True


def test_shared_pi05_prepare_reuses_existing_script_and_runtime(tmp_path):
  profile = runtime_profile(tmp_path, project="openpi")
  plan = build_prepare_plan(
    "whiteboard-wipe",
    "pi05",
    checkpoint=tmp_path / "10000",
    output_dir=tmp_path / "deployment",
    profile=profile,
    backend_args=("--hash-workers", "7"),
  )
  assert plan.command[0] == str(profile.python)
  assert plan.command[1].endswith("prepare_shared_task_pi05_deployment.py")
  assert plan.command[plan.command.index("--task") + 1] == "whiteboard-wipe"
  assert plan.command[plan.command.index("--openpi-root") + 1] == str(
    profile.project_root
  )
  assert plan.environment["CUDA_VISIBLE_DEVICES"] == "2"
  python_paths = plan.environment["PYTHONPATH"].split(os.pathsep)
  assert str(deployment.ROOT / "src") in python_paths
  assert str(profile.project_root / "src") in python_paths
  assert str(profile.project_root / "packages/openpi-client/src") in python_paths
  assert "/user/runtime/src" in python_paths


def test_prepare_default_preserves_current_python_symlink(tmp_path, monkeypatch):
  python = tmp_path / "model-env/bin/python"
  python.parent.mkdir(parents=True)
  python.symlink_to(sys.executable)
  monkeypatch.setattr(deployment.sys, "executable", str(python))

  plan = build_prepare_plan(
    "bulb-screw", "pi05", checkpoint=tmp_path / "checkpoint",
    output_dir=tmp_path / "deployment",
  )

  assert plan.command[0] == str(python)


def test_serve_manifest_preserves_model_python_symlink(tmp_path):
  python = tmp_path / "model-env/bin/python"
  python.parent.mkdir(parents=True)
  python.symlink_to(sys.executable)
  path = frozen_manifest(tmp_path / "deployment")
  manifest_path, manifest, adapter = load_deployment_manifest(path)
  manifest["model_python"] = str(python)

  plan = build_serve_plan(manifest_path, manifest, adapter)

  assert plan.command[0] == str(python)


def test_prepare_rejects_owned_options_and_trex_is_not_claimed(tmp_path):
  with pytest.raises(DeploymentError, match="owns these options"):
    build_prepare_plan(
      "pick-place",
      "pi05",
      checkpoint=tmp_path / "checkpoint",
      output_dir=tmp_path / "out",
      backend_args=("--output-dir", "/other"),
    )
  with pytest.raises(CapabilityError, match="prepare is not implemented"):
    build_prepare_plan(
      "poker-draw",
      "pi05+trex",
      checkpoint=tmp_path / "checkpoint",
      output_dir=tmp_path / "out",
    )


def test_manifest_selects_trex_only_with_explicit_override(tmp_path):
  path = frozen_manifest(
    tmp_path,
    task="poker-draw",
    schema="poker_pi05_deployment_v1",
  )
  _, _, base = load_deployment_manifest(path)
  assert base.model_family == "pi05"
  _, _, trex = load_deployment_manifest(path, model_family="pi05+trex")
  assert trex.model_family == "pi05+trex"


def test_serve_is_explicitly_inapplicable_to_local_and_embedded(tmp_path):
  manifest_path = frozen_manifest(tmp_path)
  _, manifest, _ = load_deployment_manifest(manifest_path)
  with pytest.raises(CapabilityError, match="serve is not applicable"):
    build_serve_plan(
      manifest_path,
      manifest,
      adapter_for("pick-place", "egotouch"),
    )
  with pytest.raises(CapabilityError, match="serve is not applicable"):
    build_serve_plan(
      manifest_path,
      manifest,
      adapter_for("poker-draw", "pi05+trex"),
    )


def test_fixed_bind_backend_rejects_an_unsupported_host(tmp_path):
  manifest_path = frozen_manifest(
    tmp_path,
    task="whiteboard-wipe",
    schema="shared_task_pi05_deployment_v2",
  )
  _, manifest, adapter = load_deployment_manifest(manifest_path)
  assert adapter.bind_host == "0.0.0.0"
  plan = build_serve_plan(manifest_path, manifest, adapter)
  assert plan.endpoint == "ws://127.0.0.1:18783"
  with pytest.raises(CapabilityError, match="fixed bind host"):
    build_serve_plan(
      manifest_path,
      manifest,
      adapter,
      host="127.0.0.1",
    )
  with pytest.raises(CapabilityError, match="local sidecar"):
    build_serve_plan(
      manifest_path,
      manifest,
      adapter,
      endpoint="ws://policy.example:18783",
    )
  with pytest.raises(CapabilityError, match="plaintext"):
    build_serve_plan(
      manifest_path,
      manifest,
      adapter,
      endpoint="wss://127.0.0.1:18783",
    )


def test_readiness_requires_matching_websocket_metadata(tmp_path):
  manifest_path = frozen_manifest(tmp_path)
  _, manifest, adapter = load_deployment_manifest(manifest_path)
  metadata = {
    "evaluation_task": "pick-place",
    "model_family": "pi0.5",
    "deployment_id": manifest["deployment_id"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "action_horizon": manifest["prediction_horizon"],
    "action_dim": manifest["action_dim"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
  }
  assert validate_server_metadata(metadata, manifest, adapter) is metadata
  with pytest.raises(DeploymentError, match="identity mismatch"):
    validate_server_metadata({**metadata, "deployment_id": "stale"}, manifest, adapter)
  with pytest.raises(DeploymentError, match="observation_contract"):
    validate_server_metadata({
      **metadata,
      "observation_contract": {
        **manifest["observation_contract"],
        "cameras": ["head"],
      },
    }, manifest, adapter)
  with pytest.raises(DeploymentError, match="action_representation"):
    validate_server_metadata({
      **metadata,
      "action_representation": {"type": "joint_delta"},
    }, manifest, adapter)


def test_evaluation_plan_preserves_model_defaults_and_separates_python(tmp_path):
  manifest_path = frozen_manifest(tmp_path)
  _, manifest, adapter = load_deployment_manifest(manifest_path)
  plan = build_evaluation_plan(
    manifest_path,
    manifest,
    adapter,
    output_dir=tmp_path / "run/evaluation",
    service_mode="external",
    endpoint="ws://127.0.0.1:18783",
    profile=runtime_profile(tmp_path, project="openpi"),
    num_trials=2,
    seed_start=4,
    video_count=1,
  )
  assert plan.command[0] == sys.executable
  assert "--execute-steps" not in plan.command
  assert plan.command[plan.command.index("--num-trials") + 1] == "2"
  assert plan.command[-2:] == ("--server", "ws://127.0.0.1:18783")
  assert plan.environment.get("CUDA_VISIBLE_DEVICES") != "2"
  assert "/user/runtime/src" not in plan.environment["PYTHONPATH"].split(os.pathsep)
  assert "127.0.0.1" in plan.environment["NO_PROXY"].split(",")
  assert "127.0.0.1" in plan.environment["no_proxy"].split(",")


def test_egotouch_evaluation_uses_worker_runtime_not_sim_python(tmp_path):
  path = frozen_manifest(
    tmp_path,
    task="pick-place",
    family="EgoTouch",
    schema="pickplace_egotouch_deployment_v1",
  )
  payload = json.loads(path.read_text())
  payload.update({
    "prediction_horizon": 50,
    "action_dim": 108,
    "model_python": str(tmp_path / "old/python"),
    "model_project_path": str(tmp_path / "old/project"),
  })
  path.write_text(json.dumps(payload))
  manifest_path, manifest, adapter = load_deployment_manifest(path)
  profile = runtime_profile(tmp_path, project="egotouch")
  plan = build_evaluation_plan(
    manifest_path,
    manifest,
    adapter,
    output_dir=tmp_path / "run/evaluation",
    service_mode="managed",
    endpoint=None,
    profile=profile,
    num_trials=1,
    seed_start=0,
    video_count=0,
  )
  assert plan.command[0] == sys.executable
  assert plan.command[plan.command.index("--model-python") + 1] == str(profile.python)
  assert plan.command[plan.command.index("--model-project") + 1] == str(
    profile.project_root
  )
  assert "--server" not in plan.command
  assert "--execute-steps" not in plan.command
  assert plan.environment["CUDA_VISIBLE_DEVICES"] == "2"


def test_managed_run_owns_only_its_server_and_writes_separate_record(
  tmp_path, monkeypatch
):
  manifest_path = frozen_manifest(tmp_path / "deployment")
  processes = []
  prior_sigint = signal.getsignal(signal.SIGINT)
  prior_sigterm = signal.getsignal(signal.SIGTERM)

  class FakeProcess:
    def __init__(self, *args, **kwargs):
      self.args = args
      self.kwargs = kwargs
      self.terminated = False
      self.pid = 4321 + len(processes)
      self.returncode = None
      self.is_server = len(processes) == 0
      processes.append(self)

    def poll(self):
      return self.returncode

    def terminate(self):
      self.terminated = True
      self.returncode = 0

    def wait(self, timeout=None):
      if not self.is_server and self.returncode is None:
        self.returncode = 0
      return self.returncode

    def kill(self):
      self.returncode = -9

  monkeypatch.setattr(deployment.subprocess, "Popen", FakeProcess)
  monkeypatch.setattr(
    deployment, "_assert_managed_endpoint_available", lambda *a, **k: None
  )
  monkeypatch.setattr(
    deployment.os,
    "killpg",
    lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
  )
  monkeypatch.setattr(
    deployment,
    "verify_deployment",
    lambda *args, **kwargs: {"status": "verified"},
  )
  monkeypatch.setattr(
    deployment,
    "wait_for_server",
    lambda endpoint, manifest, adapter, **kwargs: {
      "deployment_id": manifest["deployment_id"]
    },
  )

  output = tmp_path / "managed_run"
  result = run_deployment(
    manifest_path,
    output_dir=output,
    num_trials=1,
    video_count=0,
    smoke_seconds=2.0,
  )
  assert result["status"] == "complete"
  assert result["diagnostic_only"] is True
  assert result["formal_metrics_valid"] is False
  assert processes[0].terminated is True
  assert all(process.kwargs["start_new_session"] is True for process in processes)
  assert signal.getsignal(signal.SIGINT) is prior_sigint
  assert signal.getsignal(signal.SIGTERM) is prior_sigterm
  persisted = json.loads((output / "run_manifest.json").read_text())
  assert persisted["deployment_manifest_rewritten"] is False
  assert persisted["managed_server_stopped"] is True
  assert persisted["evaluation_dir"] == str(output / "evaluation")
  assert (output / "server.log").is_file()


def test_external_run_never_creates_or_stops_a_server(tmp_path, monkeypatch):
  manifest_path = frozen_manifest(tmp_path / "deployment")
  monkeypatch.setattr(
    deployment,
    "verify_deployment",
    lambda *args, **kwargs: {"status": "verified"},
  )
  monkeypatch.setattr(
    deployment,
    "wait_for_server",
    lambda endpoint, manifest, adapter, **kwargs: {
      "deployment_id": manifest["deployment_id"]
    },
  )

  class EvaluationProcess:
    pid = 5432
    returncode = None

    def __init__(self, command, **kwargs):
      assert command[1].endswith("scripts/evaluate/evaluate.py")
      assert kwargs["start_new_session"] is True

    def poll(self):
      return self.returncode

    def wait(self, timeout=None):
      self.returncode = 0
      return self.returncode

    def terminate(self):
      pytest.fail("completed evaluation must not need PID fallback termination")

    def kill(self):
      pytest.fail("completed evaluation must not be killed")

  monkeypatch.setattr(deployment.subprocess, "Popen", EvaluationProcess)
  monkeypatch.setattr(
    deployment.os,
    "killpg",
    lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
  )
  output = tmp_path / "external_run"
  result = run_deployment(
    manifest_path,
    output_dir=output,
    service_mode="external",
    endpoint="127.0.0.1:19000",
  )
  assert result["managed_server"] is False
  assert result["endpoint"] == "ws://127.0.0.1:19000"
  assert not (output / "server.log").exists()


@pytest.mark.parametrize("mode,seconds", [("observe", 90.0), ("strict", 120.0)])
def test_deploy_run_forwards_guard_mode_and_simulation_limit(tmp_path, mode, seconds):
  manifest_path = frozen_manifest(tmp_path / "deployment")
  result = run_deployment(
    manifest_path,
    output_dir=tmp_path / "run",
    service_mode="external",
    endpoint="ws://127.0.0.1:18780",
    max_sim_seconds=seconds,
    guard_mode=mode,
    dry_run=True,
  )
  command = result["evaluation"]["command"]
  assert command[command.index("--max-sim-seconds") + 1] == str(seconds)
  assert command[command.index("--guard-mode") + 1] == mode
  assert result["diagnostic_only"] is (mode == "observe")
  assert result["formal_metrics_requested"] is (mode == "strict")


def test_pipeline_prepare_record_never_rewrites_frozen_manifest(tmp_path):
  output = tmp_path / "prepared"
  manifest_path = frozen_manifest(output)
  before = manifest_path.read_bytes()
  plan = ProcessPlan(("python", "prepare.py"), tmp_path, {})
  target = write_pipeline_deployment(
    output,
    task="pick-place",
    model_family="pi05",
    prepare_plan=plan,
    profile=None,
    verification={"status": "verified"},
  )
  assert manifest_path.read_bytes() == before
  record = json.loads(target.read_text())
  assert record["frozen_manifest_rewritten"] is False
  assert record["deployment_id"] == "deployment-123"


def test_cli_lists_the_same_registry(capsys):
  module = run_path(
    str(Path(__file__).parents[1] / "scripts/deploy/deploy.py"),
    run_name="deploy_interface_test",
  )
  assert module["main"](["--list-support"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["deployments"] == list(adapter_catalog())
