from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.workcell.config import CameraConfig, WorkcellConfig
from kaihand_tactile_env.workcell.recording import (
  EpisodeRecorder,
  validate_episode,
  wait_until_object_stable,
)
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation
from kaihand_tactile_env.workcell.tactile import SolverContactTactileProvider


class _ScriptedSimulation:
  def __init__(self, twists: list[tuple[float, float]], timestep: float = 0.01):
    self.timestep = timestep
    self.data = SimpleNamespace(time=0.0)
    self._twists = twists
    self._index = -1

  def step(self) -> None:
    self._index += 1
    self.data.time += self.timestep

  def object_twist(self, _name: str) -> np.ndarray:
    linear, angular = self._twists[min(self._index, len(self._twists) - 1)]
    return np.array([linear, 0.0, 0.0, angular, 0.0, 0.0])


class _FakeRenderer:
  def calibration(self, _data: Any, camera: CameraConfig) -> Any:
    return SimpleNamespace(
      intrinsic=np.eye(3),
      fovy_degrees=45.0,
      world_from_camera=np.eye(4),
    )

  def capture(self, _data: Any, camera: CameraConfig) -> dict[str, np.ndarray]:
    return {
      "rgb": np.zeros((camera.height, camera.width, 3), dtype=np.uint8),
      "depth": np.ones((camera.height, camera.width), dtype=np.float32),
      "segmentation": np.zeros(
        (camera.height, camera.width, 2), dtype=np.int32
      ),
    }


def test_terminal_stability_resets_window_and_exits_on_velocity_event() -> None:
  twists = (
    [(0.03, 0.1)] * 2
    + [(0.01, 0.1)] * 5
    + [(0.01, 0.3)]
    + [(0.01, 0.1)] * 10
  )
  simulation = _ScriptedSimulation(twists)
  observed: list[tuple[float, str]] = []

  result = wait_until_object_stable(
    simulation,  # type: ignore[arg-type]
    "cylinder",
    observer=lambda sim, phase: observed.append((sim.data.time, phase)),
    stable_duration=0.1,
    maximum_duration=1.0,
  )

  assert result.steps == 18
  assert result.elapsed_seconds == pytest.approx(0.18)
  assert result.stable_seconds == pytest.approx(0.1)
  assert result.linear_speed == pytest.approx(0.01)
  assert result.angular_speed == pytest.approx(0.1)
  assert len(observed) == result.steps
  assert {phase for _time, phase in observed} == {"terminal_settle"}


def test_terminal_stability_has_a_safety_timeout() -> None:
  simulation = _ScriptedSimulation([(0.03, 0.3)] * 10)

  with pytest.raises(RuntimeError, match="did not remain below"):
    wait_until_object_stable(
      simulation,  # type: ignore[arg-type]
      "cylinder",
      stable_duration=0.02,
      maximum_duration=0.05,
    )

  assert simulation.data.time == pytest.approx(0.05)


def test_record_terminal_appends_exact_synchronized_sample_once(tmp_path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  camera = CameraConfig("head", width=4, height=3)
  config = WorkcellConfig(
    cameras=(camera,),
    tactile_provider=SolverContactTactileProvider.source,
  )
  output = tmp_path / "terminal.h5"

  with EpisodeRecorder(
    output,
    simulation,
    config,
    renderer=_FakeRenderer(),  # type: ignore[arg-type]
  ) as recorder:
    recorder.record_initial()
    simulation.step(3)
    simulation.data.qvel.fill(0.0)
    mujoco.mj_forward(simulation.model, simulation.data)
    recorder.record_terminal()
    recorder.record_terminal()
    final_pose = simulation.object_pose("cylinder")
    final_twist = simulation.object_twist("cylinder")
    recorder.set_outcome(
      {
        "success": True,
        "final_object_pose": final_pose.tolist(),
        "final_object_twist": final_twist.tolist(),
      }
    )

  report = validate_episode(output)
  assert report.valid
  assert not any("cylinder is stable" in warning for warning in report.warnings)
  assert report.state_samples == 2
  assert report.camera_samples == {"head": 2}
  with h5py.File(output, "r") as file:
    timestamps = np.asarray(file["state/timestamp"])
    camera_timestamps = np.asarray(file["cameras/head/timestamp"])
    np.testing.assert_allclose(timestamps, (0.0, 0.006))
    np.testing.assert_allclose(camera_timestamps, timestamps)
    assert np.all(np.diff(timestamps) > 0.0)
    assert np.all(np.diff(camera_timestamps) > 0.0)
    np.testing.assert_array_equal(file["cameras/head/state_index"][:], (0, 1))
    np.testing.assert_allclose(file["objects/cylinder/pose_wxyz"][-1], final_pose)
    assert file["tactile_proxy/normal_force"].shape[0] == 2
    assert file["contacts/frame_count"].shape[0] == 2
    phase = file["commands/phase"][-1]
    if isinstance(phase, bytes):
      phase = phase.decode()
    assert phase == "terminal_settle"
    outcome = json.loads(file.attrs["outcome_json"])
    np.testing.assert_allclose(outcome["final_object_pose"], final_pose)
    np.testing.assert_allclose(outcome["final_object_twist"], final_twist)


def test_terminal_retargets_same_time_camera_to_new_final_state(tmp_path) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  camera = CameraConfig("head", width=4, height=3)
  config = WorkcellConfig(
    cameras=(camera,),
    tactile_provider=SolverContactTactileProvider.source,
  )
  output = tmp_path / "camera_state_index.h5"

  with EpisodeRecorder(
    output,
    simulation,
    config,
    renderer=_FakeRenderer(),  # type: ignore[arg-type]
  ) as recorder:
    recorder.record_initial()
    for _ in range(17):
      simulation.step()
      recorder.observe(simulation, "return_home")
    simulation.data.qvel.fill(0.0)
    mujoco.mj_forward(simulation.model, simulation.data)
    recorder.record_terminal()
    final_pose = simulation.object_pose("cylinder")
    final_twist = simulation.object_twist("cylinder")
    recorder.set_outcome(
      {
        "final_object_pose": final_pose.tolist(),
        "final_object_twist": final_twist.tolist(),
      }
    )

  report = validate_episode(output)
  assert report.valid
  assert not any("cylinder is stable" in warning for warning in report.warnings)
  with h5py.File(output, "r") as file:
    state_timestamps = np.asarray(file["state/timestamp"])
    camera_timestamps = np.asarray(file["cameras/head/timestamp"])
    state_indices = np.asarray(file["cameras/head/state_index"])
    np.testing.assert_allclose(state_timestamps, (0.0, 0.01, 0.02, 0.03, 0.034))
    np.testing.assert_allclose(camera_timestamps, (0.0, 0.034))
    np.testing.assert_array_equal(state_indices, (0, 4))
    assert camera_timestamps[-1] == state_timestamps[state_indices[-1]]


def _write_strict_terminal_episode(
  output: Path,
  *,
  with_camera: bool = False,
) -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False)
  cameras = (CameraConfig("head", width=4, height=3),) if with_camera else ()
  config = WorkcellConfig(
    cameras=cameras, tactile_provider=SolverContactTactileProvider.source
  )
  renderer = _FakeRenderer() if with_camera else None
  with EpisodeRecorder(
    output,
    simulation,
    config,
    renderer=renderer,  # type: ignore[arg-type]
  ) as recorder:
    simulation.data.qvel.fill(0.0)
    mujoco.mj_forward(simulation.model, simulation.data)
    recorder.record_initial()
    for _ in range(51):
      simulation.data.time += simulation.timestep
      recorder.observe(simulation, "terminal_settle")
    recorder.record_terminal()
    pose = simulation.object_pose("cylinder")
    twist = simulation.object_twist("cylinder")
    recorder.set_outcome(
      {
        "final_object_pose": pose.tolist(),
        "final_object_twist": twist.tolist(),
        "terminal_stability": {
          "elapsed_seconds": 51 * simulation.timestep,
          "stable_seconds": 0.1,
          "linear_speed": float(np.linalg.norm(twist[:3])),
          "angular_speed": float(np.linalg.norm(twist[3:])),
          "steps": 51,
        },
      }
    )


def test_validator_rejects_unsynchronized_or_unstable_strict_terminal(
  tmp_path,
) -> None:
  output = tmp_path / "invalid_terminal.h5"
  _write_strict_terminal_episode(output)
  with h5py.File(output, "r+") as file:
    file["commands/phase"][-1] = "return_home"
    file["objects/cylinder/twist_linear_angular"][-1, 0] = 0.02
    outcome = json.loads(file.attrs["outcome_json"])
    outcome["final_object_pose"][0] += 0.001
    file.attrs["outcome_json"] = json.dumps(outcome)

  report = validate_episode(output)
  assert not report.valid
  assert any("does not end in terminal_settle" in error for error in report.errors)
  assert any("velocity exceeds" in error for error in report.errors)
  assert any("final_object_pose is not synchronized" in error for error in report.errors)
  assert any("final_object_twist is not synchronized" in error for error in report.errors)


@pytest.mark.parametrize(
  ("field", "value", "expected_error"),
  (
    (
      "elapsed_seconds",
      None,
      "elapsed_seconds is missing or non-finite",
    ),
    (
      "stable_seconds",
      0.099,
      "stable_seconds must be at least",
    ),
    (
      "elapsed_seconds",
      0.099,
      "elapsed_seconds must be at least stable_seconds",
    ),
    ("steps", 0, "steps must be a positive integer"),
    ("linear_speed", float("nan"), "linear_speed is missing or non-finite"),
    (
      "linear_speed",
      0.01,
      "linear_speed is not synchronized with the final state",
    ),
    (
      "angular_speed",
      0.2,
      "angular_speed is outside the stability threshold",
    ),
  ),
)
def test_validator_rejects_tampered_terminal_stability_metadata(
  tmp_path,
  field: str,
  value: float | None,
  expected_error: str,
) -> None:
  output = tmp_path / f"invalid_{field}_{expected_error[:5]}.h5"
  _write_strict_terminal_episode(output)
  with h5py.File(output, "r+") as file:
    outcome = json.loads(file.attrs["outcome_json"])
    if value is None:
      del outcome["terminal_stability"][field]
    else:
      outcome["terminal_stability"][field] = value
    file.attrs["outcome_json"] = json.dumps(outcome)

  report = validate_episode(output)
  assert not report.valid
  assert any(expected_error in error for error in report.errors)


def test_validator_rejects_unstable_sample_in_terminal_window(tmp_path) -> None:
  output = tmp_path / "invalid_terminal_window.h5"
  _write_strict_terminal_episode(output)
  with h5py.File(output, "r+") as file:
    file["objects/cylinder/twist_linear_angular"][-5, 0] = 0.02

  report = validate_episode(output)
  assert not report.valid
  assert any(
    "terminal stability window contains cylinder velocity above thresholds"
    in error
    for error in report.errors
  )


def test_validator_excludes_float_rounded_stability_left_boundary(
  tmp_path,
) -> None:
  output = tmp_path / "rounded_terminal_window.h5"
  _write_strict_terminal_episode(output)
  terminal_time = 3.659999999999818
  rounded_left_boundary = 3.559999999999829
  with h5py.File(output, "r+") as file:
    timestamps = np.array(
      [
        rounded_left_boundary,
        *(3.57 + np.arange(9) * 0.01),
        3.657999999999818,
        terminal_time,
      ]
    )
    assert timestamps.shape == file["state/timestamp"].shape
    assert rounded_left_boundary > terminal_time - 0.1
    file["state/timestamp"][:] = timestamps
    # This is the last sample before the 50 accepted physics steps.  It must
    # remain outside the strict window despite lying a few ulps over t - 0.1.
    file["objects/cylinder/twist_linear_angular"][0, 3] = 0.209

  assert validate_episode(output).valid

  with h5py.File(output, "r+") as file:
    # The next recorded state is unambiguously inside the terminal window.
    file["objects/cylinder/twist_linear_angular"][1, 3] = 0.2
  report = validate_episode(output)
  assert not report.valid
  assert any(
    "terminal stability window contains cylinder velocity above thresholds"
    in error
    for error in report.errors
  )


def test_validator_requires_terminal_camera_state_synchronization(tmp_path) -> None:
  output = tmp_path / "invalid_terminal_camera.h5"
  _write_strict_terminal_episode(output, with_camera=True)

  initial_report = validate_episode(output)
  assert initial_report.valid
  with h5py.File(output, "r+") as file:
    state_timestamps = np.asarray(file["state/timestamp"])
    assert state_timestamps[-1] - state_timestamps[-2] == pytest.approx(0.002)
    file["cameras/head/timestamp"][-1] -= 0.001
    file["cameras/head/state_index"][-1] -= 1

  report = validate_episode(output)
  assert not report.valid
  assert any(
    "terminal camera head timestamp is not synchronized" in error
    for error in report.errors
  )
  assert any(
    "terminal camera head state_index does not refer to the final state" in error
    for error in report.errors
  )


def test_validator_rejects_missing_configured_terminal_camera(tmp_path) -> None:
  output = tmp_path / "missing_terminal_camera.h5"
  _write_strict_terminal_episode(output, with_camera=True)
  with h5py.File(output, "r+") as file:
    del file["cameras/head"]

  report = validate_episode(output)
  assert not report.valid
  assert any(
    "terminal camera head group is missing" in error for error in report.errors
  )


def test_validator_keeps_old_camera_manifest_compatible(tmp_path) -> None:
  output = tmp_path / "old_camera_manifest.h5"
  _write_strict_terminal_episode(output, with_camera=True)
  with h5py.File(output, "r+") as file:
    del file["cameras"].attrs["configured_names_json"]

  assert validate_episode(output).valid


def test_validator_warns_for_unsettled_legacy_episode(tmp_path) -> None:
  output = tmp_path / "legacy_unsettled.h5"
  _write_strict_terminal_episode(output)
  with h5py.File(output, "r+") as file:
    file["objects/cylinder/twist_linear_angular"][-1, 0] = 0.021
    outcome = json.loads(file.attrs["outcome_json"])
    del outcome["terminal_stability"]
    file.attrs["outcome_json"] = json.dumps(outcome)

  report = validate_episode(output)
  assert report.valid
  assert any("legacy episode ends before cylinder is stable" in warning for warning in report.warnings)
