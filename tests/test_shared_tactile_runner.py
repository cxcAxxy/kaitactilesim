from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared.tactile_deployment import JOINT_NAMES
from kaihand_tactile_env.shared.tactile_task_contracts import SHARED_TACTILE_TASKS

ROOT = Path(__file__).resolve().parents[1]
WORKCELL = ROOT / "scripts/workcell"


def _load_script(name: str, filename: str):
  sys.path.insert(0, str(WORKCELL))
  try:
    spec = importlib.util.spec_from_file_location(name, WORKCELL / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(WORKCELL))


@pytest.fixture(scope="module")
def runner():
  return _load_script(
    "shared_tactile_runner_under_test", "run_shared_task_pi05_policy.py"
  )


@pytest.fixture(scope="module")
def entrypoint():
  return _load_script(
    "shared_tactile_entrypoint_under_test", "run_shared_task_tactile_policy.py"
  )


def _argv(tmp_path: Path, *, output_name="trial"):
  return [
    "--task", "bulb-screw",
    "--server", "local-worker",
    "--deployment-manifest", str(tmp_path / "deployment.json"),
    "--output-dir", str(tmp_path / output_name),
    "--seed", "3",
    "--execute-steps", "1",
    "--max-requests", "1",
    "--max-sim-seconds", "1",
    "--action-interpolation", "zoh",
    "--guard-mode", "strict",
    "--no-record",
    "--no-save-first-request",
  ]


def _deployment():
  return {
    "task": "bulb-screw",
    "deployment_id": "deployment",
    "model_family": "ftp1",
    "checkpoint_path": "/checkpoint",
    "checkpoint_sha256": "a" * 64,
    "prediction_horizon": 4,
    "action_dim": 27,
    "joint_names": JOINT_NAMES,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [2, 2, 3],
      "tactile_sent_to_model": True,
    },
    "action_representation": "absolute-right-arm7-hand20-joints",
    "instruction": "Complete the task.",
  }


class _Simulation:
  def __init__(self):
    self.data = SimpleNamespace(time=0.0)
    self.model = SimpleNamespace()
    self.timestep = 0.01


class _Adapter:
  def __init__(self):
    self.simulation = _Simulation()
    self.success = False

  def stage(self):
    return "policy"

  def snapshot(self):
    return {"success": self.success}

  def report(self):
    return {"success": self.success, "native_metric": 1}

  def step(self):
    self.simulation.data.time += self.simulation.timestep


class _Renderer:
  backend_info = {"test": True}

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False

  def capture(self, data, camera):
    return {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}


class _Clipping:
  def __init__(self, *args):
    pass

  def record(self, action, stage):
    return {
      "applied_action": np.asarray(action).tolist(),
      "clipped_values": 0,
      "stage": stage,
    }

  def summary(self):
    return {"clipped_values": 0}


class _Worker:
  def __init__(self, binding, *, advance_during_inference=False):
    self.binding = binding
    self.advance_during_inference = advance_during_inference
    self.closed = False

  def get_server_metadata(self):
    return {"ready": True}

  def infer(self, request):
    self.binding.infer_times.append(self.binding.simulation.data.time)
    if self.advance_during_inference:
      self.binding.simulation.data.time += 0.01
    actions = np.stack(
      [np.full(27, value, dtype=np.float64) for value in (10, 20, 30, 40)]
    )
    return {"actions": actions, "server_timing": {}}

  def close(self):
    self.closed = True


class _Binding:
  action_start_index = 1

  def __init__(self, *, advance_during_inference=False):
    self.deployment = _deployment()
    self.advance_during_inference = advance_during_inference
    self.attached = False
    self.finalized = False
    self.request_times = []
    self.infer_times = []
    self.worker = None

  def load_manifest(self, path):
    return Path(path), self.deployment

  def diagnostic_reasons(self, interpolation):
    if interpolation != "zoh":
      raise ValueError("zoh required")
    return ["test_tactile_diagnostic"]

  def report_identity(self):
    return {
      "controller": "shared-tactile-direct-joints-v1",
      "action_start_index": self.action_start_index,
      "deployment_manifest_sha256": "b" * 64,
      "simulation_paused_during_inference": True,
      "expert_used": False,
    }

  def start(self, output):
    self.worker = _Worker(
      self, advance_during_inference=self.advance_during_inference
    )
    return self.worker

  def validate_server(self, metadata, execute_steps):
    assert execute_steps == 1
    return 4

  def attach(self, simulation):
    self.simulation = simulation
    self.attached = True

  def request(self, simulation, images, state, seed, index, execute_steps):
    self.request_times.append(simulation.data.time)
    return {
      "state": state,
      "sim_time": simulation.data.time,
      "tactile_map_raw": np.zeros((5, 3, 7, 5), dtype=np.float32),
    }

  def request_log(self, request, response):
    return {"observation_sim_time": request["sim_time"]}

  def request_arrays(self, request):
    return {"tactile_map_raw": request["tactile_map_raw"]}

  def finalize_report(self, report):
    self.finalized = True
    report.update(valid_trial=True, native_success=bool(report.get("task_success")))


class _HoldWorker(_Worker):
  def infer(self, request):
    self.binding.infer_times.append(self.binding.simulation.data.time)
    actions = np.broadcast_to(
      np.asarray(request["state"], dtype=np.float64), (4, 27)
    ).copy()
    return {"actions": actions, "server_timing": {}}


class _HoldBinding(_Binding):
  def start(self, output):
    self.worker = _HoldWorker(self)
    return self.worker


def _patch_rollout(monkeypatch, runner, adapter, applied):
  monkeypatch.setattr(runner, "create_task_policy_adapter", lambda *a, **k: adapter)
  monkeypatch.setattr(runner, "ensure_evaluation_physics", lambda *a, **k: None)
  monkeypatch.setattr(runner, "start_whiteboard_contact_recording", lambda adapter: None)
  monkeypatch.setattr(runner, "WorkcellRenderer", _Renderer)
  monkeypatch.setattr(runner, "ActionClipDiagnostics", _Clipping)
  monkeypatch.setattr(
    runner, "joint_limits", lambda simulation, names: (np.full(27, -100), np.full(27, 100))
  )
  monkeypatch.setattr(
    runner, "current_controller_target", lambda simulation, names: np.zeros(27)
  )
  monkeypatch.setattr(
    runner, "right_joint_state", lambda simulation, names: np.zeros(27)
  )
  monkeypatch.setattr(
    runner,
    "validated_actions",
    lambda response, **kwargs: np.asarray(response["actions"]),
  )

  def apply_action(simulation, names, action, lower, upper):
    applied.append(np.asarray(action).copy())
    return 0

  monkeypatch.setattr(runner, "apply_action", apply_action)
  monkeypatch.setattr(
    runner,
    "evaluation_guard_report",
    lambda *a, **k: {
      "guard_events": [],
      "stop_reason": None,
      "physics_valid": True,
      "formal_metrics_valid": True,
      "diagnostic_only": False,
    },
  )


def test_binding_executes_future_offset_and_finalizes(monkeypatch, runner, tmp_path):
  args = runner.parse_args(_argv(tmp_path))
  binding = _Binding()
  adapter = _Adapter()
  applied = []
  _patch_rollout(monkeypatch, runner, adapter, applied)

  report = runner.run(args, policy_binding=binding)

  assert binding.attached is True
  assert binding.request_times == binding.infer_times == [0.0]
  assert len(applied) == 1
  np.testing.assert_array_equal(applied[0], np.full(27, 20))
  assert binding.worker.closed is True
  assert binding.finalized is True
  assert report["valid_trial"] is True
  assert report["evaluation"]["native_metric"] == 1
  assert report["controller"] == "shared-tactile-direct-joints-v1"
  assert json.loads((args.output_dir / "summary.json").read_text())["valid_trial"]


def test_binding_rejects_simulation_advance_and_still_closes(
  monkeypatch, runner, tmp_path
):
  args = runner.parse_args(_argv(tmp_path, output_name="advanced"))
  binding = _Binding(advance_during_inference=True)
  adapter = _Adapter()
  _patch_rollout(monkeypatch, runner, adapter, [])

  with pytest.raises(RuntimeError, match="simulation advanced"):
    runner.run(args, policy_binding=binding)

  assert binding.worker.closed is True
  assert binding.finalized is True
  summary = json.loads((args.output_dir / "summary.json").read_text())
  assert summary["status"] == "error"


def test_original_path_still_uses_pi05_loader(monkeypatch, runner, tmp_path):
  args = runner.parse_args(_argv(tmp_path, output_name="pi05"))

  def pi05_loader(path, task):
    raise LookupError("pi05 loader selected")

  monkeypatch.setattr(runner, "load_deployment", pi05_loader)
  with pytest.raises(LookupError, match="pi05 loader selected"):
    runner.run(args)


def test_binding_rejects_task_mismatch_and_expert_preroll(runner, tmp_path):
  args = runner.parse_args(_argv(tmp_path, output_name="mismatch"))
  binding = _Binding()
  binding.deployment["task"] = "install-ram"
  with pytest.raises(RuntimeError, match="does not match --task"):
    runner.run(args, policy_binding=binding)

  args = runner.parse_args(
    [*_argv(tmp_path, output_name="preroll"), "--pre-roll-actions", "/tmp/actions.parquet"]
  )
  with pytest.raises(ValueError, match="do not permit expert pre-roll"):
    runner.run(args, policy_binding=_Binding())

  args = runner.parse_args(
    [*_argv(tmp_path, output_name="hash"), "--expected-deployment-manifest-sha256", "c" * 64]
  )
  with pytest.raises(RuntimeError, match="hash differs from parent batch"):
    runner.run(args, policy_binding=_Binding())
  assert not args.output_dir.exists()


def test_tactile_entrypoint_defaults_and_explicit_rejections(entrypoint, tmp_path):
  base = [
    "--task", "bulb-screw",
    "--deployment-manifest", str(tmp_path / "deployment.json"),
    "--output-dir", str(tmp_path / "entry"),
    "--seed", "0",
  ]
  args = entrypoint.parse_tactile_args(base)
  assert args.server == "local-worker"
  assert args.action_interpolation == "zoh"

  explicit = entrypoint.parse_tactile_args(
    [*base, "--server=worker-ignored", "--action-interpolation=linear"]
  )
  assert explicit.server == "worker-ignored"
  assert explicit.action_interpolation == "linear"

  with pytest.raises(SystemExit, match="save-raw is not supported"):
    entrypoint.parse_tactile_args([*base, "--save-raw"])
  with pytest.raises(SystemExit, match="do not permit expert pre-roll"):
    entrypoint.parse_tactile_args(
      [*base, "--pre-roll-actions", "/tmp/actions.parquet"]
    )


def test_all_native_shared_task_adapters_take_mock_policy_physics_steps(
  monkeypatch, runner, tmp_path
):
  """CPU smoke only: real task physics/guards, but no renderer or model weights."""

  monkeypatch.setattr(runner, "WorkcellRenderer", _Renderer)
  monkeypatch.setattr(runner, "start_whiteboard_contact_recording", lambda adapter: None)
  for task in SHARED_TACTILE_TASKS:
    output = tmp_path / task
    values = _argv(tmp_path, output_name=task)
    values[values.index("bulb-screw")] = task
    values.extend(("--max-sim-seconds", "0.006", "--guard-mode", "observe"))
    args = runner.parse_args(values)
    binding = _HoldBinding()
    binding.deployment["task"] = task

    report = runner.run(args, policy_binding=binding)

    assert report["task"] == task
    assert report["stats"]["requests"] == 1
    assert report["stats"]["action_steps"] == 1
    assert 0.0 < report["sim_seconds"] <= 0.006 + 1.0e-9
    assert report["evaluation"] is not None
    assert report["physics_valid"] is True
    assert binding.request_times == binding.infer_times == [0.0]
    assert binding.worker.closed is True
    assert binding.finalized is True
    assert (output / "summary.json").is_file()
