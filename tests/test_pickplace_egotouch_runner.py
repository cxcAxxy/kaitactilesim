"""Contract checks for the local PickPlace EgoTouch rollout runner."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def runner():
  root = Path(__file__).parents[1]
  import sys

  sys.path.insert(0, str(root / "src"))
  sys.path.insert(0, str(root / "scripts/workcell"))
  try:
    yield run_path(str(root / "scripts/workcell/run_pickplace_egotouch_policy.py"))
  finally:
    sys.path.remove(str(root / "scripts/workcell"))
    sys.path.remove(str(root / "src"))


def _canonical_sha256(value) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(payload).hexdigest()


def _deployment(
  tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict]:
  checkpoint = tmp_path / "epoch_0050.pt"
  checkpoint.write_bytes(b"candidate weights")
  binding = tmp_path / "egotouch_binding.json"
  binding_payload = {"stats": {"pos": {"mean": [0, 0, 0]}}, "views": ["head"]}
  binding.write_text(json.dumps(binding_payload, indent=2), encoding="utf-8")
  model_python = tmp_path / "python"
  model_python.write_bytes(b"frozen python interpreter")
  model_project = tmp_path / "egotouch"
  source = model_project / "inference/embodiment_policy.py"
  source.parent.mkdir(parents=True)
  source.write_text("# frozen model source\n", encoding="utf-8")
  source_files = [{
    "path": "inference/embodiment_policy.py",
    "bytes": source.stat().st_size,
    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
  }]
  manifest_payload = {
    "schema": "pickplace_egotouch_deployment_v1",
    "task": "pick-place",
    "deployment_id": "test-deployment",
    "model_family": "EgoTouch",
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "binding_path": str(binding),
    "binding_sha256": _canonical_sha256(binding_payload),
    "binding_file_sha256": hashlib.sha256(binding.read_bytes()).hexdigest(),
    "model_weights": "ema",
    "prediction_horizon": 50,
    "action_dim": 108,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "view_slots": ["head", "left_wrist", "right_wrist"],
      "view_mask": [True, False, True],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"type": "absolute"},
    "model_python": str(model_python),
    "model_python_sha256": hashlib.sha256(model_python.read_bytes()).hexdigest(),
    "model_project_path": str(model_project),
    "model_source_sha256": _canonical_sha256(source_files),
    "model_source_files": source_files,
  }
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
  return (
    manifest,
    checkpoint,
    binding,
    model_python,
    model_project,
    manifest_payload,
  )


def test_manifest_and_worker_identity_are_fail_closed(tmp_path, runner):
  manifest, checkpoint, binding, model_python, model_project, deployment = (
    _deployment(tmp_path)
  )
  (
    _, loaded, selected_checkpoint, selected_binding,
    selected_python, selected_project,
  ) = runner["load_deployment"](
    manifest, checkpoint, binding, model_python, model_project
  )
  assert loaded == deployment
  assert selected_checkpoint == checkpoint
  assert selected_binding == binding
  assert selected_python == model_python
  assert selected_project == model_project

  metadata = {
    "checkpoint_sha256": deployment["checkpoint_sha256"],
    "binding_sha256": deployment["binding_sha256"],
    "model_weights": "ema",
    "action_horizon": 50,
    "action_dim": 108,
    "cameras": ["head", "right_wrist"],
    "view_slots": ["head", "left_wrist", "right_wrist"],
    "view_mask": [True, False, True],
  }
  assert runner["_validate_worker_metadata"](metadata, deployment, 5) == 50
  metadata["cameras"] = ["head"]
  with pytest.raises(RuntimeError, match="deployment mismatch"):
    runner["_validate_worker_metadata"](metadata, deployment, 5)

  wrong = json.loads(manifest.read_text())
  wrong["observation_contract"]["view_mask"] = [True, True, True]
  manifest.write_text(json.dumps(wrong), encoding="utf-8")
  with pytest.raises(RuntimeError, match="view_mask"):
    runner["load_deployment"](
      manifest, checkpoint, binding, model_python, model_project
    )


def test_manifest_binds_explicit_model_runtime_paths(tmp_path, runner):
  manifest, checkpoint, binding, model_python, model_project, _ = _deployment(
    tmp_path
  )
  other_python = tmp_path / "other-python"
  other_python.write_bytes(model_python.read_bytes())
  with pytest.raises(RuntimeError, match="--model-python differs"):
    runner["load_deployment"](
      manifest, checkpoint, binding, other_python, model_project
    )

  other_project = tmp_path / "other-project"
  other_project.mkdir()
  with pytest.raises(RuntimeError, match="--model-project differs"):
    runner["load_deployment"](
      manifest, checkpoint, binding, model_python, other_project
    )


def test_manifest_binds_python_and_model_source_hashes(tmp_path, runner):
  manifest, checkpoint, binding, model_python, model_project, deployment = (
    _deployment(tmp_path)
  )
  model_python.write_bytes(b"changed interpreter")
  with pytest.raises(RuntimeError, match="model Python SHA-256 mismatch"):
    runner["load_deployment"](
      manifest, checkpoint, binding, model_python, model_project
    )

  model_python.write_bytes(b"frozen python interpreter")
  source = model_project / deployment["model_source_files"][0]["path"]
  source.write_text("# changed model source\n", encoding="utf-8")
  with pytest.raises(RuntimeError, match="source files changed"):
    runner["load_deployment"](
      manifest, checkpoint, binding, model_python, model_project
    )


def test_manifest_binds_model_source_aggregate(tmp_path, runner):
  manifest, checkpoint, binding, model_python, model_project, deployment = (
    _deployment(tmp_path)
  )
  deployment["model_source_sha256"] = "0" * 64
  manifest.write_text(json.dumps(deployment), encoding="utf-8")
  with pytest.raises(RuntimeError, match="aggregate SHA-256 mismatch"):
    runner["load_deployment"](
      manifest, checkpoint, binding, model_python, model_project
    )


def test_requested_render_environment_is_explicit(monkeypatch, runner):
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  monkeypatch.setenv("KAIHAND_RENDER_BACKEND", "software")
  monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
  requested = runner["_requested_render_environment"]()
  assert requested["MUJOCO_GL"] == "osmesa"
  assert requested["KAIHAND_RENDER_BACKEND"] == "software"
  assert requested["PYOPENGL_PLATFORM"] is None


def test_model_runtime_report_records_frozen_identity_and_device(tmp_path, runner):
  _, _, _, model_python, model_project, deployment = _deployment(tmp_path)
  runtime = runner["_model_runtime_report"](
    deployment, model_python, model_project, "cpu"
  )
  assert runtime == {
    "model_python": str(model_python),
    "model_python_sha256": deployment["model_python_sha256"],
    "model_project_path": str(model_project),
    "model_source_sha256": deployment["model_source_sha256"],
    "model_device": "cpu",
  }


def test_request_contains_synchronized_named_rgbs_and_physical_poses(
  tmp_path, monkeypatch, runner
):
  head = np.full((240, 320, 3), 7, dtype=np.uint8)
  wrist = np.full((240, 320, 3), 19, dtype=np.uint8)
  captures = []

  class Renderer:
    def calibration(self, data, camera):
      assert data == "frozen-state"
      assert camera.name == "head"
      return SimpleNamespace(world_from_camera=np.eye(4))

    def capture(self, data, camera):
      assert data == "frozen-state"
      captures.append(camera.name)
      return {"rgb": {"head": head, "right_wrist": wrist}[camera.name]}

  wrists = {
    "hand_l_base_link_site": np.eye(4),
    "hand_r_base_link_site": np.diag([1.0, 1.0, 1.0, 1.0]),
  }
  tips = {}
  for side in ("l", "r"):
    for index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
      pose = np.eye(4)
      pose[0, 3] = index + (10 if side == "r" else 0)
      link = "link6" if finger == "thumb" else "link4"
      tips[f"hand_{side}_{finger}_{link}_site"] = pose

  def named_pose(_simulation, name):
    return (wrists | tips)[name]

  monkeypatch.setitem(runner["build_request"].__globals__, "_named_site_pose", named_pose)
  simulation = SimpleNamespace(data="frozen-state")
  cameras = {
    name: SimpleNamespace(name=name) for name in ("head", "right_wrist")
  }
  save = tmp_path / "first_request.npz"
  request = runner["build_request"](
    simulation, Renderer(), cameras, noise_seed=42, save_input=save
  )
  assert captures == ["head", "right_wrist"]
  assert tuple(request["rgbs"]) == ("head", "right_wrist")
  np.testing.assert_array_equal(
    np.frombuffer(base64.b64decode(request["rgbs"]["head"]), dtype=np.uint8),
    head.ravel(),
  )
  np.testing.assert_array_equal(
    np.frombuffer(base64.b64decode(request["rgbs"]["right_wrist"]), dtype=np.uint8),
    wrist.ravel(),
  )
  assert "rgb" not in request and "tactile" not in request
  assert np.asarray(request["wrists"]).shape == (2, 4, 4)
  assert np.asarray(request["tips_relative"]).shape == (2, 5, 4, 4)
  assert request["noise_seed"] == 42
  assert request["save_input"] == str(save.resolve())


def test_cli_and_worker_command_include_explicit_files_and_device(tmp_path, runner):
  project = tmp_path / "egotouch"
  project.mkdir()
  checkpoint = tmp_path / "epoch.pt"
  binding = tmp_path / "binding.json"
  manifest = tmp_path / "deployment.json"
  args = runner["parse_args"]([
    "--model-python", "/env/bin/python",
    "--model-project", str(project),
    "--model-device", "cpu",
    "--checkpoint", str(checkpoint),
    "--binding", str(binding),
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "trial"),
    "--seed", "3",
    "--execute-steps", "5",
    "--max-requests", "1",
    "--no-record",
  ])
  model_python = Path("/resolved/env/bin/python")
  command = runner["_worker_command"](
    args, checkpoint, binding, model_python, project
  )
  assert command[0] == str(model_python)
  assert command[command.index("--project") + 1] == str(project)
  assert command[-6:] == [
    "--checkpoint", str(checkpoint), "--binding", str(binding),
    "--device", "cpu",
  ]
  assert args.max_requests == 1 and args.record is False
  assert tuple(args.hand_ik_orientation_weights) == (0.02, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
  ("initial_returncode", "wait_returncode"),
  ((17, None), (None, 23)),
)
def test_worker_shutdown_rejects_nonzero_exit_before_or_after_close(
  runner, initial_returncode, wait_returncode
):
  class Stdin:
    def __init__(self):
      self.payload = ""

    def write(self, value):
      self.payload += value

    def flush(self):
      pass

  class Process:
    def __init__(self):
      self.returncode = initial_returncode
      self.stdin = Stdin()
      self.waited = False

    def poll(self):
      return self.returncode

    def wait(self, timeout=None):
      assert timeout == 0.25
      self.waited = True
      self.returncode = wait_returncode
      return self.returncode

    def kill(self):
      pytest.fail("a completed worker must not be killed")

  process = Process()
  expected = initial_returncode if initial_returncode is not None else wait_returncode
  with pytest.raises(RuntimeError, match=f"return code {expected}"):
    runner["_shutdown_worker"](process, timeout=0.25)
  if initial_returncode is None:
    assert process.waited is True
    assert process.stdin.payload == '{"close":true}\n'
  else:
    assert process.waited is False
    assert process.stdin.payload == ""


def test_worker_shutdown_accepts_clean_close(runner):
  writes = []

  class Process:
    returncode = None
    stdin = SimpleNamespace(write=writes.append, flush=lambda: None)

    def poll(self):
      return self.returncode

    def wait(self, timeout=None):
      assert timeout == 0.25
      self.returncode = 0
      return self.returncode

  process = Process()
  runner["_shutdown_worker"](process, timeout=0.25)
  assert writes == ['{"close":true}\n']


def test_wrist_guard_raises_typed_policy_rejection_before_execution(runner):
  targets = SimpleNamespace(
    site_positions_world=np.asarray([[1.0, 0.0, 0.0]], dtype=float)
  )
  decoded = SimpleNamespace(for_side=lambda side: targets)

  class Simulation:
    @staticmethod
    def current_pose_matrix(side):
      assert side == "right"
      return np.zeros(3), np.eye(3)

    @staticmethod
    def set_pose_target(*_args, **_kwargs):
      pytest.fail("a wrist-jump rejection must occur before commanding the arm")

  args = SimpleNamespace(control_side="right", max_wrist_jump=0.15)
  stats = runner["RolloutStats"]()
  with pytest.raises(runner["PolicyActionRejected"]) as rejected:
    runner["_command_action_step"](Simulation(), decoded, 0, args, stats)
  assert isinstance(rejected.value, RuntimeError)
  assert rejected.value.guard == "wrist_target_jump"


def test_bimanual_rejection_rolls_back_entire_command_and_diagnostics(
  runner, monkeypatch
):
  sides = ("left", "right")
  targets = {
    side: SimpleNamespace(
      site_positions_world=np.asarray([[0.01, 0.0, 0.0]], dtype=float),
      site_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
      fingertips_world=np.zeros((1, 5, 3), dtype=float),
      fingertip_rotations_world=None,
    )
    for side in sides
  }
  decoded = SimpleNamespace(for_side=targets.__getitem__)

  class Simulation:
    def __init__(self):
      self._arm_goal = {
        "left": np.asarray([1.0, 2.0]),
        "right": np.asarray([3.0, 4.0]),
      }
      self._pose_target_position = {
        "left": np.asarray([1.0, 0.0, 0.0]),
        "right": np.asarray([2.0, 0.0, 0.0]),
      }
      self._pose_target_rotation = {
        "left": np.eye(3),
        "right": np.eye(3) * 2.0,
      }
      self._hand_targets = {
        "left": {"left_joint": 0.1},
        "right": {"right_joint": 0.2},
      }
      self._target_mocap = {"left": 0, "right": 1}
      self.data = SimpleNamespace(
        time=1.25,
        mocap_pos=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        mocap_quat=np.asarray(
          [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
      )
      self.commanded_side = None

    @staticmethod
    def current_pose_matrix(_side):
      return np.zeros(3), np.eye(3)

    def set_pose_target(self, side, *_args):
      self.commanded_side = side
      self._arm_goal[side][:] = 10.0 if side == "left" else 20.0
      self._pose_target_position[side][:] = 30.0
      self._pose_target_rotation[side][:] = 40.0
      mocap_id = self._target_mocap[side]
      self.data.mocap_pos[mocap_id, :] = 50.0
      self.data.mocap_quat[mocap_id, :] = 60.0
      return SimpleNamespace(
        success=True,
        position_error=0.0,
        orientation_error=0.0,
        joint_positions=np.asarray([0.3, 0.4]),
      )

    def solve_hand_ik(self, side, *_args, **_kwargs):
      return SimpleNamespace(
        success=side == "left",
        position_error=0.001 if side == "left" else 0.5,
        joint_names=(f"{side}_joint",),
        joint_positions=np.asarray([0.9]),
      )

    def set_hand_joint_targets(self, names, positions):
      self._hand_targets[self.commanded_side][names[0]] = float(positions[0])
      return len(names)

  simulation = Simulation()
  runner["configure_evaluation_guards"](simulation, "observe")
  original = {
    "arm": {side: simulation._arm_goal[side].copy() for side in sides},
    "position": {
      side: simulation._pose_target_position[side].copy() for side in sides
    },
    "rotation": {
      side: simulation._pose_target_rotation[side].copy() for side in sides
    },
    "hand": {side: dict(simulation._hand_targets[side]) for side in sides},
    "mocap_pos": simulation.data.mocap_pos.copy(),
    "mocap_quat": simulation.data.mocap_quat.copy(),
  }
  monkeypatch.setitem(
    runner["_command_action_step"].__globals__,
    "_hand_ik_diagnostic",
    lambda _simulation, **kwargs: {
      "side": kwargs["side"],
      "command_applied": kwargs["command_applied"],
    },
  )
  diagnostics = [{"side": "older", "command_applied": True}]
  args = SimpleNamespace(
    control_side="both",
    max_wrist_jump=1.0,
    max_arm_position_error=0.1,
    max_arm_orientation_error=0.1,
    strict_ik=False,
    hand_ik_tolerance=0.01,
    max_hand_error=0.1,
    hold_failed_hand_ik=False,
  )

  result = runner["_command_action_step"](
    simulation, decoded, 0, args, runner["RolloutStats"](),
    diagnostics=diagnostics,
  )

  assert result is diagnostics
  for side in sides:
    np.testing.assert_array_equal(simulation._arm_goal[side], original["arm"][side])
    np.testing.assert_array_equal(
      simulation._pose_target_position[side], original["position"][side]
    )
    np.testing.assert_array_equal(
      simulation._pose_target_rotation[side], original["rotation"][side]
    )
    assert simulation._hand_targets[side] == original["hand"][side]
  np.testing.assert_array_equal(simulation.data.mocap_pos, original["mocap_pos"])
  np.testing.assert_array_equal(
    simulation.data.mocap_quat, original["mocap_quat"]
  )
  assert diagnostics[0] == {"side": "older", "command_applied": True}
  assert [row["side"] for row in diagnostics[1:]] == ["left", "right"]
  assert all(row["command_applied"] is False for row in diagnostics[1:])
  assert all(row["command_rolled_back"] is True for row in diagnostics[1:])
  assert simulation._evaluation_action_rejected is True
  events = simulation._evaluation_guards.summary()
  assert len(events) == 1
  assert events[0]["kind"] == "hand_reachability"
  assert events[0]["message"] == "policy action rejected by hand_reachability"
  assert events[0]["measured_max"] == pytest.approx(0.5)
  assert events[0]["threshold"] == pytest.approx(0.1)
