from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import model_fingerprint
from kaihand_tactile_env.shared.policy_tasks import TaskPolicyAdapter
from kaihand_tactile_env.tasks.bulb_screw import config

_MODULE = run_path(
  str(Path(__file__).parents[1] / "scripts/workcell/diagnose_bulb_replay_and_stages.py")
)
_initial_replay_state_validation = _MODULE["_initial_replay_state_validation"]
_quaternion_angle = _MODULE["_quaternion_angle"]
_recorded_controller_target_actions = _MODULE["_recorded_controller_target_actions"]
_linear_substep_action = _MODULE["_linear_substep_action"]
_stage_contract_validation = _MODULE["_stage_contract_validation"]
_validated_published_parquet = _MODULE["_validated_published_parquet"]
_replay_parser = _MODULE["_replay_parser"]
restore_reference_stage = _MODULE["restore_reference_stage"]
select_stage_index = _MODULE["select_stage_index"]
_RUNNER = run_path(
  str(Path(__file__).parents[1] / "scripts/workcell/run_shared_task_pi05_policy.py")
)
_evaluation_validity_metadata = _RUNNER["_evaluation_validity_metadata"]


def test_selects_stable_rows_for_all_three_diagnostic_stages(tmp_path: Path):
  path = tmp_path / "source.h5"
  timestamps = np.arange(12, dtype=float) * 0.1
  phases = np.array(
    ["grasp", "lift", "lift", "lift", "transfer", "transfer", "transfer"]
    + ["engage_thread"] * 5,
    dtype=object,
  )
  engaged = np.arange(12) >= 7
  with h5py.File(path, "w") as file:
    file.attrs["control_hz"] = 10
    state = file.create_group("state")
    state.create_dataset("timestamp", data=timestamps)
    commands = file.create_group("commands")
    commands.create_dataset(
      "phase", data=phases, dtype=h5py.string_dtype(encoding="utf-8")
    )
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("engaged", data=engaged)
  with h5py.File(path) as file:
    assert select_stage_index(file, "stable_grasp") == 3
    assert select_stage_index(file, "socket_hover") == 6
    assert select_stage_index(file, "threaded") == 9


def test_stage_selector_rejects_missing_phase(tmp_path: Path):
  path = tmp_path / "source.h5"
  with h5py.File(path, "w") as file:
    file.attrs["control_hz"] = 10
    state = file.create_group("state")
    state.create_dataset("timestamp", data=[0.0, 0.1])
    commands = file.create_group("commands")
    commands.create_dataset("phase", data=["grasp", "grasp"], dtype=h5py.string_dtype())
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("engaged", data=[False, False])
  with h5py.File(path) as file:
    with pytest.raises(ValueError, match="no transfer phase"):
      select_stage_index(file, "socket_hover")


def test_threaded_stage_requires_continuous_engagement_window(tmp_path: Path):
  path = tmp_path / "source.h5"
  with h5py.File(path, "w") as file:
    file.attrs["control_hz"] = 10
    state = file.create_group("state")
    state.create_dataset("timestamp", data=[0.0, 0.1, 0.2, 0.3])
    commands = file.create_group("commands")
    commands.create_dataset(
      "phase", data=["engage_thread"] * 4, dtype=h5py.string_dtype()
    )
    bulb = file.create_group("bulb_screw")
    # A single-point check at t=0.2 would incorrectly accept this trace.
    bulb.create_dataset("engaged", data=[True, False, True, True])
  with h5py.File(path) as file:
    with pytest.raises(ValueError, match="not continuous.*index 1"):
      select_stage_index(file, "threaded")


def test_selects_stable_threaded_and_loaded_pre_tightening_windows(
  tmp_path: Path,
):
  path = tmp_path / "source.h5"
  string = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as file:
    file.attrs["control_hz"] = 10
    state = file.create_group("state")
    state.create_dataset("timestamp", data=np.arange(8, dtype=float) * 0.1)
    commands = file.create_group("commands")
    commands.create_dataset("phase", data=["turn"] * 5 + ["tighten"] * 3, dtype=string)
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("engaged", data=np.ones(8, dtype=bool))
    bulb.create_dataset(
      "clockwise_turns", data=[0.21, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28, 0.28]
    )
    bulb.create_dataset(
      "fingertip_normal_force_n",
      data=np.asarray([[2.5] * 5] * 5 + [[5.0] * 5] * 3),
    )
    bulb.create_dataset("right_hand_nonpad_contact_count", data=np.zeros(8))
    bulb.create_dataset("robot_contact_count", data=np.full(8, 5))
    bulb.create_dataset("right_hand_contact_count", data=np.full(8, 5))
    bulb.create_dataset("seated", data=[False] * 5 + [True] * 3)
    bulb.create_dataset("bulb_lit", data=[False] * 7 + [True])
    bulb.create_dataset("hand_clockwise_torque_nm", data=[0.0] * 5 + [0.2] * 3)

  with h5py.File(path) as file:
    assert select_stage_index(file, "late_threaded_stable") == 2
    assert select_stage_index(file, "seated_loaded") == 6


def test_stage_contract_checks_the_post_settle_policy_state():
  state = SimpleNamespace(
    engaged=True,
    seated=False,
    bulb_lit=False,
    fingertip_normal_force_n=(2.1,) * 5,
    right_hand_nonpad_contact_count=0,
    right_hand_contact_count=5,
    robot_contact_count=5,
    hand_clockwise_torque_nm=0.0,
    clockwise_turns=0.24,
    insertion_depth_m=0.0069,
    lateral_error_m=0.0,
    tilt_rad=0.0,
    linear_speed_m_s=0.0,
    angular_speed_rad_s=0.0,
  )

  verified = _stage_contract_validation("late_threaded_stable", state)
  assert verified["status"] == "verified"
  assert verified["failed_checks"] == []

  state.fingertip_normal_force_n = (1.9, 2.1, 2.1, 2.1, 2.1)
  mismatch = _stage_contract_validation("late_threaded_stable", state)
  assert mismatch["status"] == "mismatch"
  assert mismatch["failed_checks"] == ["five_fingertips_loaded_ge_2n"]


def test_quaternion_error_ignores_equivalent_sign():
  quaternion = np.array([0.5, 0.5, 0.5, 0.5])
  assert _quaternion_angle(quaternion, -quaternion) == pytest.approx(0.0)
  with pytest.raises(ValueError, match="nonzero norm"):
    _quaternion_angle(np.zeros(4), quaternion)
  with pytest.raises(ValueError, match="finite"):
    _quaternion_angle(np.array([np.nan, 0.0, 0.0, 1.0]), quaternion)


def test_threaded_stage_rejects_sparse_samples_as_continuous(tmp_path: Path):
  path = tmp_path / "source.h5"
  with h5py.File(path, "w") as file:
    file.attrs["control_hz"] = 100
    state = file.create_group("state")
    state.create_dataset("timestamp", data=[0.0, 0.1, 0.2, 0.3])
    commands = file.create_group("commands")
    commands.create_dataset(
      "phase", data=["engage_thread"] * 4, dtype=h5py.string_dtype()
    )
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("engaged", data=[True] * 4)
  with h5py.File(path) as file:
    with pytest.raises(ValueError, match="too sparse"):
      select_stage_index(file, "threaded")


def test_replay_initial_state_validates_object_pose_and_named_robot_state(
  tmp_path: Path,
):
  adapter = TaskPolicyAdapter("bulb-screw", 3)
  simulation = adapter.simulation
  joint_names, qpos, qvel = simulation.joint_state()
  pose = simulation.object_pose("bulb")
  path = tmp_path / "source.h5"
  string = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as file:
    state = file.create_group("state")
    state.create_dataset("joint_names", data=joint_names, dtype=string)
    state.create_dataset("robot_joint_position", data=[qpos])
    state.create_dataset("robot_joint_velocity", data=[qvel])
    objects = file.create_group("objects")
    bulb = objects.create_group("bulb")
    bulb.create_dataset("pose_wxyz", data=[pose])

  with h5py.File(path) as file:
    report = _initial_replay_state_validation(simulation, file)
  assert report["status"] == "verified"
  assert report["joint_count"] == len(joint_names)
  assert set(report["errors"]) == {
    "object_position_m",
    "object_orientation_rad",
    "robot_qpos_rms_rad",
    "robot_qpos_maximum_abs_rad",
    "robot_qvel_rms_rad_s",
    "robot_qvel_maximum_abs_rad_s",
  }

  with h5py.File(path, "r+") as file:
    quaternion = file["objects/bulb/pose_wxyz"][0, 3:]
    angle = 1.0e-4
    quaternion[:] = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    file["objects/bulb/pose_wxyz"][0, 3:] = quaternion
  with h5py.File(path) as file:
    with pytest.raises(RuntimeError, match="object_orientation_rad"):
      _initial_replay_state_validation(simulation, file)


def test_replay_initial_state_rejects_robot_qpos_or_qvel_mismatch(tmp_path: Path):
  adapter = TaskPolicyAdapter("bulb-screw", 3)
  simulation = adapter.simulation
  joint_names, qpos, qvel = simulation.joint_state()
  path = tmp_path / "source.h5"
  string = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as file:
    state = file.create_group("state")
    state.create_dataset("joint_names", data=joint_names, dtype=string)
    changed_qpos = qpos.copy()
    changed_qvel = qvel.copy()
    changed_qpos[4] += 1.0e-4
    changed_qvel[7] += 2.0e-4
    state.create_dataset("robot_joint_position", data=[changed_qpos])
    state.create_dataset("robot_joint_velocity", data=[changed_qvel])
    objects = file.create_group("objects")
    bulb = objects.create_group("bulb")
    bulb.create_dataset("pose_wxyz", data=[simulation.object_pose("bulb")])
  with h5py.File(path) as file:
    with pytest.raises(
      RuntimeError,
      match="robot_qpos_maximum_abs_rad.*robot_qvel_maximum_abs_rad_s",
    ):
      _initial_replay_state_validation(simulation, file)


def test_loads_recorded_controller_targets_at_action_provenance_rows(
  tmp_path: Path,
):
  arm_names = [f"right_arm_joint{index}" for index in range(1, 8)]
  hand_names = [f"hand_r_joint{index}" for index in range(20)]
  joint_names = arm_names + hand_names
  arm_goals = np.arange(4 * 7, dtype=float).reshape(4, 7)
  hand_targets = np.arange(4 * 20, dtype=float).reshape(4, 20) + 100.0
  path = tmp_path / "source.h5"
  string = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as file:
    commands = file.create_group("commands")
    commands.create_dataset(
      "arm_joint_names",
      data=[f"left_arm_joint{index}" for index in range(1, 8)] + arm_names,
      dtype=string,
    )
    commands.create_dataset(
      "arm_joint_target",
      data=np.concatenate((np.zeros_like(arm_goals), arm_goals), axis=1),
    )
    commands.create_dataset("hand_joint_names", data=hand_names, dtype=string)
    commands.create_dataset("hand_joint_target", data=hand_targets)
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("right_arm_joint_goal", data=arm_goals)

  actions = _recorded_controller_target_actions(
    path, joint_names, np.array([1, 3], dtype=np.int64)
  )

  np.testing.assert_array_equal(actions[:, :7], arm_goals[[1, 3]])
  np.testing.assert_array_equal(actions[:, 7:], hand_targets[[1, 3]])
  assert (
    _replay_parser()
    .parse_args(
      ["--dataset-root", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )
    .action_source
    == "published"
  )
  assert (
    _replay_parser()
    .parse_args(
      ["--dataset-root", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )
    .control_interpolation
    == "zoh"
  )
  assert (
    _replay_parser()
    .parse_args(
      [
        "--dataset-root",
        str(tmp_path),
        "--control-interpolation",
        "linear",
        "--control-rate-hz",
        "100",
        "--output",
        str(tmp_path / "out.json"),
      ]
    )
    .control_interpolation
    == "linear"
  )

  with h5py.File(path, "r+") as file:
    file["bulb_screw/right_arm_joint_goal"][3, 0] = np.nan
  with pytest.raises(ValueError, match="finite"):
    _recorded_controller_target_actions(
      path, joint_names, np.array([3], dtype=np.int64)
    )


def test_linear_substep_interpolates_previous_endpoint_to_current_endpoint():
  start = np.zeros(3)
  end = np.ones(3)

  first = _linear_substep_action(
    start,
    end,
    interval_start_s=0.0,
    interval_end_s=0.01,
    current_time_s=0.0,
    timestep_s=0.002,
  )
  last = _linear_substep_action(
    start,
    end,
    interval_start_s=0.0,
    interval_end_s=0.01,
    current_time_s=0.008,
    timestep_s=0.002,
  )

  np.testing.assert_allclose(first, 0.2)
  np.testing.assert_allclose(last, 1.0)
  with pytest.raises(ValueError, match="must be positive"):
    _linear_substep_action(
      start,
      end,
      interval_start_s=0.0,
      interval_end_s=0.0,
      current_time_s=0.0,
      timestep_s=0.002,
    )


def test_validates_real_parquet_rows_against_hdf5_reconstruction(tmp_path: Path):
  pyarrow = pytest.importorskip("pyarrow")
  parquet = pytest.importorskip("pyarrow.parquet")
  actions = np.arange(54, dtype=np.float32).reshape(2, 27) / 100.0
  observed = actions - np.float32(0.5)
  indices = np.array([4, 7], dtype=np.int64)
  path = tmp_path / "episode.parquet"
  table = pyarrow.table(
    {
      "action": actions.tolist(),
      "observation.state": observed.tolist(),
      "provenance.action_source_state_index": indices,
    }
  )
  parquet.write_table(table, path)

  loaded_actions, loaded_observed, loaded_indices, report = (
    _validated_published_parquet(path, actions, observed, indices)
  )

  np.testing.assert_array_equal(loaded_actions, actions)
  np.testing.assert_array_equal(loaded_observed, observed)
  np.testing.assert_array_equal(loaded_indices, indices)
  assert report["status"] == "float32_exact"

  changed = actions.copy()
  changed[1, 3] += np.float32(0.01)
  with pytest.raises(ValueError, match="parquet actions differ"):
    _validated_published_parquet(path, changed, observed, indices)


def test_reference_stage_context_is_diagnostic_without_whiteboard_relaxation():
  args = SimpleNamespace(
    task="bulb-screw",
    diagnostic_relax_whiteboard_contact_guards=False,
    guard_mode="strict",
  )
  formal = _evaluation_validity_metadata(args)
  staged = _evaluation_validity_metadata(
    args,
    diagnostic_context="reference_stage_initialization",
  )
  observed = _evaluation_validity_metadata(
    SimpleNamespace(
      task="bulb-screw",
      diagnostic_relax_whiteboard_contact_guards=False,
      guard_mode="observe",
    )
  )

  assert formal == {
    "diagnostic_only": False,
    "formal_metrics_valid": True,
    "diagnostic_context": None,
    "diagnostic_reasons": [],
    "guard_mode": "strict",
    "whiteboard_contact_guards_enabled": None,
  }
  assert staged == {
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "diagnostic_context": "reference_stage_initialization",
    "diagnostic_reasons": ["reference_stage_initialization"],
    "guard_mode": "strict",
    "whiteboard_contact_guards_enabled": None,
  }
  assert observed == {
    "diagnostic_only": True,
    "formal_metrics_valid": False,
    "diagnostic_context": None,
    "diagnostic_reasons": ["guard_mode_observe"],
    "guard_mode": "observe",
    "whiteboard_contact_guards_enabled": None,
  }


def test_restores_reference_controller_and_equality_state(tmp_path: Path):
  source_adapter = TaskPolicyAdapter("bulb-screw", 3)
  source = source_adapter.simulation
  source.initialize_threaded()
  source.hold_current_arm_position("right")
  qpos, qvel = source.full_state()
  arm_targets, hand_targets, hand_names = source.command_state()
  path = tmp_path / "source.h5"
  metadata = {
    "scene": "bulb-screw",
    "model_layout": "task-isolated-v1",
    "mechanics_version": config.MECHANICS_VERSION,
    "initial_position_randomization": {"seed": 3},
  }
  string = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["model_fingerprint"] = model_fingerprint(source.model_path)
    file.attrs["control_hz"] = 5
    state = file.create_group("state")
    state.create_dataset("timestamp", data=[0.0, 0.2, 0.4])
    state.create_dataset("qpos", data=np.stack([qpos] * 3))
    state.create_dataset("qvel", data=np.stack([qvel] * 3))
    commands = file.create_group("commands")
    commands.create_dataset("phase", data=["engage_thread"] * 3, dtype=string)
    commands.create_dataset("hand_joint_names", data=hand_names, dtype=string)
    commands.create_dataset("hand_joint_target", data=np.stack([hand_targets] * 3))
    commands.create_dataset("arm_joint_target", data=np.stack([arm_targets] * 3))
    bulb = file.create_group("bulb_screw")
    bulb.create_dataset("engaged", data=[True] * 3)
    bulb.create_dataset("eq_data", data=np.stack([source.model.eq_data] * 3))
    bulb.create_dataset("eq_active", data=np.stack([source.data.eq_active] * 3))
    bulb.create_dataset("bulb_lit", data=[False] * 3)
    monitor = source_adapter.monitor.measure()
    for name in (
      "clockwise_turns",
      "insertion_depth_m",
      "lateral_error_m",
      "tilt_rad",
      "thread_contact_load_n",
      "seated",
      "success",
    ):
      bulb.create_dataset(name, data=[getattr(monitor, name)] * 3)
    objects = file.create_group("objects")
    object_group = objects.create_group("bulb")
    object_group.create_dataset(
      "pose_wxyz", data=np.stack([source.object_pose("bulb")] * 3)
    )

  target_adapter = TaskPolicyAdapter("bulb-screw", 3)
  report = restore_reference_stage(target_adapter, path, "threaded", settle_seconds=0.0)

  assert report["source_index"] == 1
  assert report["diagnostic_only"] is True
  assert report["formal_metrics_valid"] is False
  assert report["model_fingerprint"] == model_fingerprint(source.model_path)
  assert report["mechanics_version"] == config.MECHANICS_VERSION
  assert report["restore_position_error_m"] < 1e-12
  assert target_adapter.simulation.thread_engaged
  assert target_adapter.simulation.data.time == 0.0
  assert target_adapter.success is False
  assert target_adapter.initial_object_pose.shape == (7,)

  with pytest.raises(ValueError, match="pickup seed differs"):
    restore_reference_stage(
      TaskPolicyAdapter("bulb-screw", 8), path, "threaded", settle_seconds=0.0
    )

  with h5py.File(path, "r+") as file:
    file.attrs["model_fingerprint"] = "not-the-current-model"
  with pytest.raises(ValueError, match="model fingerprint differs"):
    restore_reference_stage(
      TaskPolicyAdapter("bulb-screw", 3), path, "threaded", settle_seconds=0.0
    )

  with h5py.File(path, "r+") as file:
    file.attrs["model_fingerprint"] = model_fingerprint(source.model_path)
    metadata["mechanics_version"] = "obsolete-mechanics"
    file.attrs["metadata_json"] = json.dumps(metadata)
  with pytest.raises(ValueError, match="mechanics version differs"):
    restore_reference_stage(
      TaskPolicyAdapter("bulb-screw", 3), path, "threaded", settle_seconds=0.0
    )
