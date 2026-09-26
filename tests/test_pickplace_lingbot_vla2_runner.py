"""Contract and lightweight rollout checks for the LingBot PickPlace runner."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/workcell"


@pytest.fixture
def runner(monkeypatch):
  monkeypatch.syspath_prepend(str(SCRIPTS))
  return importlib.import_module("run_pickplace_lingbot_vla2_policy")


def deployment(runner):
  return {
    "schema": runner.SCHEMA,
    "task": "pick-place",
    "deployment_id": "test-deployment",
    "model_family": runner.MODEL_FAMILY,
    "checkpoint_path": "/checkpoint",
    "checkpoint_sha256": "f" * 64,
    "prediction_horizon": runner.PREDICTION_HORIZON,
    "model_action_dim": runner.MODEL_ACTION_DIM,
    "action_dim": runner.ACTION_DIM,
    "control_hz": runner.CONTROL_HZ,
    "joint_names": list(runner.RIGHT_JOINT_NAMES),
    "instruction": runner.INSTRUCTION,
    "reference_dataset": str(runner.DEFAULT_REFERENCE_DATASET),
    "reference_episode_index": 0,
    "observation_contract": dict(runner.OBSERVATION_CONTRACT),
    "action_representation": dict(runner.ACTION_REPRESENTATION),
  }


def server_metadata(runner, manifest):
  return {
    "evaluation_task": "pick-place",
    "deployment_id": manifest["deployment_id"],
    "model_family": runner.MODEL_FAMILY,
    "checkpoint_path": manifest["checkpoint_path"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "prediction_horizon": manifest["prediction_horizon"],
    "action_horizon": manifest["prediction_horizon"],
    "model_action_dim": manifest["model_action_dim"],
    "action_dim": manifest["action_dim"],
    "control_hz": manifest["control_hz"],
    "joint_names": manifest["joint_names"],
    "observation_contract": manifest["observation_contract"],
    "action_representation": manifest["action_representation"],
    "instruction": manifest["instruction"],
    "reference_dataset": manifest["reference_dataset"],
  }


def test_identity_requires_frozen_checkpoint_and_camera_contract(runner):
  manifest = deployment(runner)
  runner.validate_deployment(manifest)
  assert runner.validate_server(server_metadata(runner, manifest), manifest, 16) == 50

  changed = deployment(runner)
  changed["joint_names"] = changed["joint_names"][::-1]
  with pytest.raises(RuntimeError, match="joint_names"):
    runner.validate_deployment(changed)

  metadata = server_metadata(runner, manifest)
  metadata["checkpoint_sha256"] = "wrong"
  with pytest.raises(RuntimeError, match="checkpoint_sha256"):
    runner.validate_server(metadata, manifest, 16)


def test_reference_must_match_manifest_episode_zero(tmp_path, monkeypatch, runner):
  dataset = tmp_path / "training_dataset"
  dataset.mkdir()
  other = tmp_path / "other_dataset"
  other.mkdir()
  manifest = deployment(runner)
  manifest["reference_dataset"] = str(dataset)
  assert runner.bound_reference(manifest, None, 0) == dataset.resolve()
  assert runner.bound_reference(manifest, dataset, 0) == dataset.resolve()
  with pytest.raises(RuntimeError, match="differs from the frozen training"):
    runner.bound_reference(manifest, other, 0)
  with pytest.raises(RuntimeError, match="episode 00"):
    runner.bound_reference(manifest, None, 1)
  manifest["reference_episode_index"] = 1
  with pytest.raises(RuntimeError, match="reference_episode_index"):
    runner.validate_deployment(manifest)

  manifest["reference_episode_index"] = 0
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  client_package = ModuleType("openpi_client")
  client_package.websocket_client_policy = SimpleNamespace()
  monkeypatch.setitem(sys.modules, "openpi_client", client_package)
  output = tmp_path / "wrong_reference_trial"
  args = runner.parse_args([
    "--server", "localhost:8006", "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output), "--seed", "0", "--execute-steps", "1",
    "--reference-dataset", str(other), "--no-record",
  ])
  with pytest.raises(RuntimeError, match="differs from the frozen training"):
    runner.run(args)
  assert not output.exists()


def test_default_trial_time_limit_is_ninety_seconds(runner):
  args = runner.parse_args([
    "--server", "localhost:8006", "--deployment-manifest", "/unused/manifest.json",
    "--output-dir", "/unused/output", "--seed", "0", "--execute-steps", "16",
  ])
  assert args.max_sim_seconds == 90.0


def test_request_uses_flat_lerobot_keys_and_two_uint8_cameras(runner):
  images = {
    "head": np.zeros((240, 320, 3), dtype=np.uint8),
    "right_wrist": np.full((240, 320, 3), 127, dtype=np.uint8),
  }
  state = np.arange(27, dtype=np.float32)
  request = runner.make_observation(images, state, "Pick up the cylinder.")
  assert tuple(request) == (
    "observation.images.head",
    "observation.images.right_wrist",
    "observation.state.right_arm_joint_position",
    "observation.state.right_hand_joint_position",
    "task",
  )
  np.testing.assert_array_equal(request["observation.state.right_arm_joint_position"], state[:7])
  np.testing.assert_array_equal(request["observation.state.right_hand_joint_position"], state[7:])
  assert request["observation.images.right_wrist"].dtype == np.uint8


def test_action_decoder_returns_absolute_arm_and_hand_targets(runner):
  response = {
    "auxiliary.action.right_arm_joint_target": np.full((50, 7), 0.7),
    "action.right_hand_joint_position": np.full((50, 20), 0.3),
  }
  actions = runner.validated_actions(response, horizon=50)
  assert actions.shape == (50, 27)
  np.testing.assert_array_equal(actions[:, :7], response["auxiliary.action.right_arm_joint_target"])
  np.testing.assert_array_equal(actions[:, 7:], response["action.right_hand_joint_position"])
  with pytest.raises(RuntimeError, match="finite shape"):
    runner.validated_actions({**response, "action.right_hand_joint_position": np.ones((49, 20))}, horizon=50)


def test_no_record_rollout_executes_absolute_targets_and_writes_summary(tmp_path, monkeypatch, runner):
  manifest = deployment(runner)
  dataset = tmp_path / "training_dataset"
  dataset.mkdir()
  manifest["reference_dataset"] = str(dataset)
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  seen = {"requests": [], "actions": []}

  class FakeClient:
    def __init__(self, server):
      assert server == "localhost:8006"
      self._ws = SimpleNamespace(close=lambda: None)

    def get_server_metadata(self):
      return server_metadata(runner, manifest)

    def infer(self, observation):
      seen["requests"].append(observation)
      return {
        "auxiliary.action.right_arm_joint_target": np.full((50, 7), 0.7),
        "action.right_hand_joint_position": np.full((50, 20), 0.3),
      }

  client_package = ModuleType("openpi_client")
  client_package.websocket_client_policy = SimpleNamespace(WebsocketClientPolicy=FakeClient)
  monkeypatch.setitem(sys.modules, "openpi_client", client_package)

  reference_package = ModuleType("lingbot_pickplace_reference")
  reference_package.REFERENCE_TACTILE_SOURCE = "fake_solver_contact"
  reference_package.load_pickplace_reference = lambda root, output_episode_index: SimpleNamespace(dataset_root=root)
  monkeypatch.setitem(sys.modules, "lingbot_pickplace_reference", reference_package)

  from kaihand_tactile_env.shared import (
    contact_tactile,
    evaluation_comparison,
    openwam_evaluation_plots,
  )

  monkeypatch.setattr(contact_tactile, "SolverDistributedTactileProvider", lambda _model: object())

  class FakeTrace:
    def __init__(self, *, tactile_provider):
      self.times = []

    def capture(self, simulation):
      timestamp = simulation.data.time
      if not self.times or timestamp > self.times[-1]:
        self.times.append(timestamp)

    def finish(self, review_dir):
      review_dir.mkdir(parents=True, exist_ok=True)
      count = len(self.times)
      np.savez_compressed(
        review_dir / "evaluation_rollout_trace.npz",
        simulation_time_s=np.asarray(self.times),
        state_29=np.zeros((count, 29)),
        normal_taxel_force_n=np.zeros((count, 5, 7, 5)),
        tangent_taxel_force_n=np.zeros((count, 5, 7, 5, 2)),
      )
      return {"rollout": {"samples": count}, "reference": {}, "tactile": {}, "artifacts": {}}

  class FakePlots:
    def __init__(self, reference, *, artifact_prefix):
      assert artifact_prefix == "evaluation"
      self.reference = reference
      self.times = []

    def capture(self, timestamp, _state, **_forces):
      self.times.append(timestamp)

    def finish(self, review_dir):
      assert self.times == [0.0, 1 / 30, 2 / 30]
      artifacts = {}
      for name in ("right_wrist_state", "right_hand_actuated_dof", "right_fingertip_tactile"):
        path = review_dir / f"evaluation_{name}.png"
        path.write_bytes(b"fake chart")
        artifacts[name] = str(path)
      metadata = review_dir / "evaluation_evaluation_plots.json"
      artifacts["metadata"] = str(metadata)
      return {"reference": {"dataset_root": str(self.reference.dataset_root)}, "artifacts": artifacts}

  monkeypatch.setattr(evaluation_comparison, "EvaluationComparisonTrace", FakeTrace)
  monkeypatch.setattr(openwam_evaluation_plots, "OpenWAMEvaluationPlots", FakePlots)

  class FakeSimulation:
    def __init__(self, *, scene):
      assert scene == "pick-place"
      self.model = object()
      self.timestep = 1 / 30
      self.data = SimpleNamespace(time=0.0, qpos=np.ones(27), qvel=np.zeros(27))

    def reset(self, **kwargs):
      assert kwargs["randomized_objects"] == ("cylinder",)

    def object_pose(self, name):
      assert name == "cylinder"
      return np.zeros(7)

    def object_twist(self, name):
      assert name == "cylinder"
      return np.zeros(6)

    def step(self):
      self.data.time += 1 / 30

  class FakeRenderer:
    def __init__(self, _model, cameras):
      assert tuple(camera.name for camera in cameras) == runner.CAMERAS
      self.backend_info = {"backend": "fake"}

    def __enter__(self):
      return self

    def __exit__(self, *_args):
      return None

    def capture(self, _data, camera):
      return {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}

  class FakeStabilizer:
    def __init__(self, _simulation, **kwargs):
      self.active = False

    def after_command(self):
      pass

    def after_physics_step(self):
      pass

    def closure(self):
      return 0.0

    def close(self):
      pass

  class FakeFreeClose:
    def __init__(self, _simulation, **kwargs):
      pass

    def before_physics_step(self, **kwargs):
      pass

    def close(self):
      pass

  class FakeDetector:
    def __init__(self, *, timestep):
      self.succeeded = False

    def update(self, **kwargs):
      pass

  def apply_action(_simulation, names, action, _lower, _upper):
    assert names == list(runner.RIGHT_JOINT_NAMES)
    seen["actions"].append(action.copy())
    return 0

  monkeypatch.setattr(runner, "ArmHandSimulation", FakeSimulation)
  monkeypatch.setattr(runner, "WorkcellRenderer", FakeRenderer)
  monkeypatch.setattr(runner, "AutomaticGraspStabilizer", FakeStabilizer)
  monkeypatch.setattr(runner, "AdaptiveFreeCloseForce", FakeFreeClose)
  monkeypatch.setattr(runner, "StablePlacementDetector", FakeDetector)
  monkeypatch.setattr(runner, "right_joint_state", lambda *_args: np.ones(27, dtype=np.float32))
  monkeypatch.setattr(runner, "joint_limits", lambda *_args: (np.full(27, -2.0), np.full(27, 2.0)))
  monkeypatch.setattr(runner, "apply_action", apply_action)
  monkeypatch.setattr(runner, "cylinder_is_in_box", lambda _simulation: False)

  output = tmp_path / "trial"
  args = runner.parse_args([
    "--server", "localhost:8006",
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output),
    "--seed", "0",
    "--execute-steps", "2",
    "--max-sim-seconds", str(2 / 30),
    "--no-record",
  ])
  report = runner.run(args)
  assert report["status"] == "task_not_completed"
  assert report["stats"]["requests"] == 1
  assert report["stats"]["action_steps"] == 2
  assert len(seen["requests"]) == 1
  assert len(seen["actions"]) == 2
  np.testing.assert_allclose(seen["actions"][0][:7], 0.7)
  np.testing.assert_allclose(seen["actions"][0][7:], 0.3)
  assert (output / "review").is_dir()
  assert not (output / "review/review.mp4").exists()
  assert report["comparison_plots"]["status"] == "ok"
  assert report["comparison_plots"]["rollout"]["samples"] == 3
  for filename in (
    "evaluation_rollout_trace.npz",
    "evaluation_right_wrist_state.png",
    "evaluation_right_hand_actuated_dof.png",
    "evaluation_right_fingertip_tactile.png",
  ):
    assert (output / "review" / filename).is_file()
  assert (output / "first_request.npz").is_file()
  assert json.loads((output / "summary.json").read_text())["status"] == "task_not_completed"


@pytest.mark.skipif(
  os.environ.get("KAIHAND_LINGBOT_RECORD_SMOKE") != "1",
  reason="opt-in raw-HDF5 and headless-rendering integration test",
)
@pytest.mark.parametrize("record", [True, False], ids=["video", "charts_only"])
def test_one_step_produces_real_30hz_comparison(tmp_path, monkeypatch, runner, record):
  """Exercise the actual scene, tactile solver, reference loader, and artifacts."""
  dataset = runner.DEFAULT_REFERENCE_DATASET
  if not dataset.is_dir():
    pytest.skip(f"reference dataset is unavailable: {dataset}")
  manifest = deployment(runner)
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  class FakeClient:
    def __init__(self, _server):
      self._ws = SimpleNamespace(close=lambda: None)

    def get_server_metadata(self):
      return server_metadata(runner, manifest)

    def infer(self, observation):
      arm = observation["observation.state.right_arm_joint_position"]
      hand = observation["observation.state.right_hand_joint_position"]
      return {
        "auxiliary.action.right_arm_joint_target": np.repeat(arm[None, :], 50, axis=0),
        "action.right_hand_joint_position": np.repeat(hand[None, :], 50, axis=0),
      }

  client_package = ModuleType("openpi_client")
  client_package.websocket_client_policy = SimpleNamespace(WebsocketClientPolicy=FakeClient)
  monkeypatch.setitem(sys.modules, "openpi_client", client_package)

  output = tmp_path / "trial"
  args = runner.parse_args([
    "--server", "localhost:8006",
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output),
    "--seed", "0",
    "--execute-steps", "1",
    "--max-sim-seconds", str(1 / 30),
    "--object-xy-jitter", "0",
    "--object-yaw-jitter", "0",
    "--review-width", "640",
    "--review-height", "360",
    "--record" if record else "--no-record",
  ])
  report = runner.run(args)
  assert report["status"] == "task_not_completed"
  assert report["comparison_plots"]["status"] == "ok"
  assert report["comparison_plots"]["rollout"]["samples"] == 2
  if record:
    assert report["video"]["frame_count"] == 2
    assert report["video"]["time_series_displayed"] is False
    assert report["video"]["model_input_cameras_displayed"] == ["head", "right_wrist"]
    assert (output / "review/review.mp4").is_file()
  else:
    assert "video" not in report
    assert not (output / "review/review.mp4").exists()
  for file_name in (
    "evaluation_rollout_trace.npz",
    "evaluation_right_wrist_state.png",
    "evaluation_right_hand_actuated_dof.png",
    "evaluation_right_fingertip_tactile.png",
  ):
    assert (output / "review" / file_name).is_file()
