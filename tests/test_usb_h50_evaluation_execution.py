from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)] + [
  f"hand_r_test_{i}" for i in range(20)
]


def load_script(name):
  spec = importlib.util.spec_from_file_location(
    name, ROOT / "scripts/workcell" / f"{name}.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class Simulation:
  timestep = 0.001

  def __init__(self, *, initial=0.25, time=0.0):
    self.data = SimpleNamespace(time=time, qpos=np.full(27, -99.0))
    self.arm_goal = {"right": np.full(7, initial)}
    self.hand = np.full(20, initial)
    self.commands = []

  def command_state(self):
    return np.zeros(14), self.hand, JOINTS[7:]

  def set_arm_joint_goal(self, _side, value):
    self.arm_goal["right"] = np.asarray(value).copy()

  def set_hand_joint_targets(self, _names, value):
    self.hand = np.asarray(value).copy()
    return 20

  def step(self):
    self.commands.append(np.concatenate((self.arm_goal["right"], self.hand)))
    self.data.time += self.timestep


def advance(stepper, action):
  return stepper.advance(
    action, max_sim_seconds=90.0, after_step=lambda _: None, should_stop=lambda: False
  )


def test_initial_endpoint_comes_from_controller_targets_not_measured_qpos():
  runner = load_script("run_usb_pi05_policy")
  simulation = Simulation(initial=0.25)
  stepper = runner.ActionStepper(
    simulation, JOINTS, np.zeros(27), np.ones(27), "linear"
  )
  np.testing.assert_array_equal(stepper.previous_endpoint, np.full(27, 0.25))
  advance(stepper, np.ones(27))
  np.testing.assert_allclose(simulation.commands[0], 0.25 + 0.75 * 0.03)
  np.testing.assert_allclose(simulation.commands[-1], 1.0)
  assert len(simulation.commands) == 34


def test_clipping_is_once_per_endpoint_and_persists_across_chunks():
  runner = load_script("run_usb_pi05_policy")
  simulation = Simulation(initial=0.0)
  stepper = runner.ActionStepper(
    simulation, JOINTS, np.zeros(27), np.ones(27), "linear"
  )
  assert advance(stepper, np.full(27, 2.0)) == 27
  np.testing.assert_array_equal(stepper.previous_endpoint, np.ones(27))
  simulation.data.qpos.fill(-888)
  before = len(simulation.commands)
  advance(stepper, np.zeros(27))
  assert np.all(simulation.commands[before] < 1)
  assert np.all(simulation.commands[before] > 0.9)
  np.testing.assert_array_equal(stepper.previous_endpoint, np.zeros(27))


def test_h50_k50_runs_fifty_endpoints_and_keeps_thirty_hz_epoch():
  runner = load_script("run_usb_pi05_policy")
  simulation = Simulation(time=5.0)
  stepper = runner.ActionStepper(
    simulation, JOINTS, np.zeros(27), np.ones(27), "linear"
  )
  actions = runner.validated_actions(
    {"actions": np.ones((50, 27))}, horizon=50, action_dim=27
  )
  for action in actions:
    advance(stepper, action)
  assert stepper.control_tick == 50
  assert simulation.data.time == pytest.approx(5 + 50 / 30, abs=simulation.timestep)
  assert len(simulation.commands) > 1600
  with pytest.raises(RuntimeError, match="shape"):
    runner.validated_actions({"actions": np.ones((30, 27))}, horizon=50, action_dim=27)


def test_rejected_nonfinite_prediction_holds_last_valid_command():
  runner = load_script("run_usb_pi05_policy")
  simulation = Simulation(initial=0.1)
  stepper = runner.ActionStepper(
    simulation, JOINTS, np.zeros(27), np.ones(27), "linear"
  )
  advance(stepper, np.full(27, 0.6))
  before = len(simulation.commands)
  assert advance(stepper, None) == 0
  np.testing.assert_allclose(simulation.commands[before:], 0.6)
  np.testing.assert_allclose(stepper.previous_endpoint, 0.6)


def test_historical_v1_preserves_zero_order_hold_controller():
  runner = load_script("run_usb_pi05_policy")
  assert runner.action_execution({"schema": "usb_pi05_deployment_v1"}) == "zoh"
  simulation = Simulation(initial=0.25)
  stepper = runner.ActionStepper(simulation, JOINTS, np.zeros(27), np.ones(27), "zoh")
  advance(stepper, np.ones(27))
  np.testing.assert_allclose(simulation.commands, 1.0)


@pytest.mark.parametrize("guard", ["observe", "strict"])
def test_source_environment_null_is_diagnostic_even_with_strict_guards(guard):
  runner = load_script("run_usb_pi05_policy")
  validity = runner.evaluation_validity(
    {
      "schema": "usb_pi05_deployment_v2",
      "training_provenance": {"record": {"source_environment": None}},
    },
    guard,
  )
  assert validity["diagnostic_only"] is True
  assert validity["formal_metrics_valid"] is False
  assert "source_environment_unbound" in validity["diagnostic_reasons"]


def prepared(tmp_path, monkeypatch):
  spec = importlib.util.spec_from_file_location(
    "usb_fixture_execution", ROOT / "tests/test_usb_pi05_deployment.py"
  )
  fixture = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, spec.name, fixture)
  spec.loader.exec_module(fixture)
  _, path, manifest, *_ = fixture._prepared_manifest(tmp_path, monkeypatch)
  return path, manifest


def test_new_v2_binds_complete_sim_and_model_sources_and_linear_execution(
  tmp_path, monkeypatch
):
  path, manifest = prepared(tmp_path, monkeypatch)
  runner = load_script("run_usb_pi05_policy")
  assert runner.action_execution(manifest) == "linear"
  entries = {row["path"] for row in manifest["evaluation_source_files"]}
  assert "scripts/workcell/run_usb_pi05_policy.py" in entries
  assert "src/kaihand_tactile_env/tasks/usb_insert/scene.xml" in entries
  assert "src/kaihand_tactile_env/shared/simulation.py" in entries
  proof = runner.inspect_trial_integrity(path, manifest)
  assert proof["status"] == "verified", proof
  assert proof["training_provenance"]["status"] == "verified"


def test_simulator_portable_proof_never_imports_model_dependencies(
  tmp_path, monkeypatch
):
  path, manifest = prepared(tmp_path, monkeypatch)
  runner = load_script("run_usb_pi05_policy")
  original_import = builtins.__import__

  def without_model(name, *args, **kwargs):
    if name == "openpi" or name.startswith("openpi."):
      pytest.fail("simulator byte-proof verification must not import model modules")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", without_model)
  assert runner.inspect_trial_integrity(path, manifest)["status"] == "verified"


@pytest.mark.parametrize(
  "change",
  [
    "source",
    "manifest",
    "provenance",
    "normalizer",
    "sidecar",
    "scale_report",
    "conversion_snapshot",
  ],
)
def test_child_trial_rejects_source_manifest_provenance_and_normalizer_drift(
  tmp_path, monkeypatch, change
):
  path, manifest = prepared(tmp_path, monkeypatch)
  runner = load_script("run_usb_pi05_policy")
  if change == "source":
    source = (
      Path(manifest["model_project_path"]) / manifest["openpi_source_files"][0]["path"]
    )
    source.write_text("changed source")
  elif change == "manifest":
    other = dict(manifest, instruction="changed prompt")
    path.write_text(json.dumps(other))
  elif change == "provenance":
    asset = Path(manifest["checkpoint_path"]) / "assets/training_provenance.json"
    record = json.loads(asset.read_text())
    record["normalizer"]["use_quantile_norm"] = False
    asset.write_text(json.dumps(record))
  elif change == "normalizer":
    asset = (
      Path(manifest["checkpoint_path"])
      / "assets"
      / manifest["normalizer_asset_id"]
      / "norm_stats.json"
    )
    asset.write_text("{}")
  else:
    relative = {
      "sidecar": f"{manifest['normalizer_asset_id']}/norm_stats.provenance.json",
      "scale_report": f"{manifest['normalizer_asset_id']}/norm_stats.scale_report.json",
      "conversion_snapshot": "conversion_manifest.snapshot.json",
    }[change]
    (Path(manifest["checkpoint_path"]) / "assets" / relative).write_text("{}")
  assert runner.inspect_trial_integrity(path, manifest)["status"] == "mismatch"


def test_prepare_dry_run_checks_full_numeric_proof_without_hashing_or_output(
  tmp_path, monkeypatch, capsys
):
  spec = importlib.util.spec_from_file_location(
    "usb_fixture_dry_run", ROOT / "tests/test_usb_pi05_deployment.py"
  )
  fixture = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, spec.name, fixture)
  spec.loader.exec_module(fixture)
  prepare, checkpoint, record, validation_calls = fixture._bound_checkpoint(
    tmp_path, monkeypatch
  )
  fixture._mock_openpi(monkeypatch, record)
  (checkpoint / "params/_METADATA").write_text("{}")
  monkeypatch.setattr(
    prepare,
    "checkpoint_identity",
    lambda *_: pytest.fail("dry-run must not hash model weights"),
  )
  output = tmp_path / "unused_deployment"
  assert (
    prepare.main(
      ["--checkpoint", str(checkpoint), "--output-dir", str(output), "--dry-run"]
    )
    == 0
  )
  result = json.loads(capsys.readouterr().out)
  assert result["valid"] is True
  assert result["prediction_horizon"] == 50
  assert result["normalizer_asset_id"] == fixture.SAFE_ASSET
  assert len(validation_calls) == 1
  assert not output.exists()
  (checkpoint / "params/_METADATA").unlink()
  with pytest.raises(RuntimeError, match="not committed"):
    prepare.main(
      ["--checkpoint", str(checkpoint), "--output-dir", str(output), "--dry-run"]
    )


def test_cpu_rollout_uses_h50_linear_controller_and_stays_diagnostic(
  tmp_path, monkeypatch
):
  """Use the real CPU physics, replacing GPU rendering and model inference only."""
  runner = load_script("run_usb_pi05_policy")
  simulation = runner.ArmHandSimulation(scene="usb-insert")
  _, _, names = simulation.command_state()
  joint_names = JOINTS[:7] + [name for name in names if name.startswith("hand_r_")]
  contract = {
    "endpoint_rate_hz": 30,
    "physical_substep_interpolation_required": "linear",
    "zero_order_hold_equivalent": False,
  }
  manifest = {
    "schema": "usb_pi05_deployment_v2",
    "task": "usb-insert",
    "model_family": "pi0.5",
    "deployment_id": "d" * 64,
    "checkpoint_path": "/mock/20000",
    "checkpoint_step": 20000,
    "checkpoint_sha256": "c" * 64,
    "prediction_horizon": 50,
    "model_action_dim": 32,
    "action_dim": 27,
    "control_hz": 30,
    "joint_names": joint_names,
    "instruction": "Insert USB",
    "required_action_interpolation": "linear",
    "model_config_name": "pi05_kaihand_h50",
    "normalizer_asset_id": "normalizer_h50_usb_insert_hand_tail_v1",
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"policy_output_is_absolute": True},
    "training_provenance": {
      "record": {
        "source_environment": None,
        "action_contract": {"evaluation_execution": contract},
      }
    },
  }
  path = tmp_path / "manifest.json"
  path.write_text(json.dumps(manifest))
  monkeypatch.setattr(runner, "load_deployment_manifest", lambda _: (path, manifest))
  monkeypatch.setattr(
    runner, "inspect_trial_integrity", lambda *_: {"status": "verified"}
  )
  monkeypatch.setattr(runner, "ArmHandSimulation", lambda **_: simulation)

  class Renderer:
    backend_info = {"backend": "test_cpu_no_gpu"}

    def __init__(self, *_args, **_kwargs):
      pass

    def __enter__(self):
      return self

    def __exit__(self, *_args):
      pass

    def capture(self, *_args):
      return {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}

  class Client:
    _ws = SimpleNamespace(close=lambda: None)

    def __init__(self, *_args):
      pass

    def get_server_metadata(self):
      return {**manifest, "evaluation_task": "usb-insert", "action_horizon": 50}

    def infer(self, payload):
      return {"actions": np.tile(payload["state"], (50, 1))}

  monkeypatch.setattr(runner, "WorkcellRenderer", Renderer)
  monkeypatch.setitem(
    sys.modules,
    "openpi_client",
    SimpleNamespace(
      websocket_client_policy=SimpleNamespace(WebsocketClientPolicy=Client)
    ),
  )
  args = runner.parse_args(
    [
      "--deployment-manifest",
      str(path),
      "--output-dir",
      str(tmp_path / "trial"),
      "--execute-steps",
      "50",
      "--max-sim-seconds",
      "0.04",
      "--no-record",
      "--no-save-first-request",
    ]
  )
  report = runner.run(args)
  assert report["status"] == "task_not_completed"
  assert report["action_interpolation"] == "linear"
  assert report["prediction_horizon"] == 50
  assert report["stats"]["action_steps"] == 2
  assert report["stats"]["requests"] == 1
  assert report["formal_metrics_valid"] is False
  assert "source_environment_unbound" in report["diagnostic_reasons"]
