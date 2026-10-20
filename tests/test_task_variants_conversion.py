"""Conversion contracts specific to the independently identified task variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_SUCCESS_FIELDS,
  VARIANT_TASKS,
)

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import convert_usb_to_lerobot_v3 as converter  # noqa: E402

_OBJECTS = {
  "usb-unplug": "usb_plug",
  "whiteboard-wipe-horizontal": "eraser",
  "whiteboard-wipe-horizontal-line": "eraser",
  "bulb-unscrew": "bulb",
  "bulb-screw-inplace": "bulb",
}


def _successful_outcome(task: str) -> dict[str, object]:
  outcome: dict[str, object] = {
    "success": True,
    "task": task,
    "object_name": _OBJECTS[task],
    **dict.fromkeys(VARIANT_SUCCESS_FIELDS[task], True),
  }
  if task in {"whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}:
    count = 8 if task.endswith("-line") else 25
    outcome["ink_remaining"] = [0.0] * count
  if task == "usb-unplug":
    from kaihand_tactile_env.tasks.usb_unplug import config

    state = {
      "completion_protocol": config.COMPLETION_PROTOCOL,
      "extraction_hold_now": True, "extraction_hold_verified": True,
      "extraction_hold_duration_s": 2.0, "insertion_depth_m": -.003,
      "right_hand_contact_count": 1, "table_contact_count": 0, "socket_contact_count": 0,
    }
    outcome.update({key: state[key] for key in (
      "completion_protocol", "extraction_hold_now", "extraction_hold_verified", "extraction_hold_duration_s",
    )}, final_state=state)
    outcome.update(placed_on_table=False, released=False, hand_retreated=False)
  return outcome


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_variant_task_specs_keep_new_identity_instruction_clock_and_diagnostics(task):
  spec = converter.TASK_SPECS[task]
  assert spec.name == task
  assert spec.key == task.replace("-", "_")
  assert spec.object_name == _OBJECTS[task]
  assert spec.control_hz == 100
  assert spec.instruction == TASK_INSTRUCTIONS[task]
  assert spec.diagnostic_group == task.replace("-", "_")


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_variant_success_gate_requires_every_canonical_history_flag(task):
  spec = converter.TASK_SPECS[task]
  outcome = _successful_outcome(task)
  converter._validate_success_gate(Path("episode.h5"), spec, outcome)

  fields = (("extraction_hold_now", "extraction_hold_verified", "extraction_hold_duration_s",
             "completion_protocol", "final_state") if task == "usb-unplug" else VARIANT_SUCCESS_FIELDS[task])
  for field in fields:
    invalid = dict(outcome)
    invalid.pop(field)
    with pytest.raises(ValueError, match=field):
      converter._validate_success_gate(Path("episode.h5"), spec, invalid)

  if task in {"whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}:
    invalid = dict(outcome)
    invalid.pop("ink_remaining")
    with pytest.raises(ValueError, match="ink state"):
      converter._validate_success_gate(Path("episode.h5"), spec, invalid)


@pytest.mark.parametrize(
  ("task", "legacy_outcome"),
  (
    (
      "usb-unplug",
      {
        "success": True,
        "task": "usb-unplug",
        "object_name": "usb_plug",
        "released": True,
        "grasp_verified": True,
        "active_bottom_out_confirmed": True,
        "insertion": {
          "success": True,
          "seated": True,
          "bottom_out_confirmed": True,
        },
      },
    ),
    (
      "bulb-unscrew",
      {
        "success": True,
        "task": "bulb-unscrew",
        "object_name": "bulb",
        "tightening_verified": True,
        "state": {"success": True, "seated": True},
      },
    ),
  ),
)
def test_variant_success_gate_rejects_inverse_task_outcome_relabelling(
  task, legacy_outcome
):
  with pytest.raises(ValueError, match="physical success gate failed"):
    converter._validate_success_gate(
      Path("episode.h5"), converter.TASK_SPECS[task], legacy_outcome
    )


def test_tabletop_unscrew_rejects_legacy_cradle_success_and_uses_table_instruction():
  outcome = _successful_outcome("bulb-unscrew")
  outcome.pop("placed_on_table")
  outcome["placed_in_cradle"] = True
  with pytest.raises(ValueError, match="placed_on_table"):
    converter._validate_success_gate(
      Path("old_cradle_episode.h5"), converter.TASK_SPECS["bulb-unscrew"], outcome
    )
  instruction = converter.TASK_SPECS["bulb-unscrew"].instruction
  assert "ordinary tabletop" in instruction
  assert "cradle" not in instruction


def _write_short_unplug_raw(path: Path, outcome: dict[str, object]) -> None:
  state_count = 11
  camera_times = np.array([0.0, 0.034, 0.066, 0.1])
  state_times = np.arange(state_count, dtype=np.float64) / 100.0
  camera_indices = np.searchsorted(state_times, camera_times, side="right") - 1
  string = h5py.string_dtype("utf-8")
  with h5py.File(path, "w") as file:
    file.attrs.update(
      schema_version=converter.RAW_SCHEMA,
      metadata_json=json.dumps({"scene": "usb-unplug", "active_objects": ["usb_plug"]}),
      outcome_json=json.dumps(outcome),
      camera_hz=30,
      control_hz=100,
      physics_hz=500,
    )
    state = file.create_group("state")
    state.create_dataset("timestamp", data=state_times)
    state.create_dataset("joint_names", data=converter.RIGHT_JOINT_NAMES, dtype=string)
    for name in (
      "robot_joint_position",
      "robot_joint_velocity",
      "robot_joint_effort",
    ):
      state.create_dataset(name, data=np.zeros((state_count, 27)))

    commands = file.create_group("commands")
    commands.create_dataset(
      "arm_joint_names", data=converter.RIGHT_ARM_JOINT_NAMES, dtype=string
    )
    commands.create_dataset(
      "hand_joint_names", data=converter.RIGHT_HAND_JOINT_NAMES, dtype=string
    )
    commands.create_dataset("arm_joint_target", data=np.zeros((state_count, 7)))
    commands.create_dataset("hand_joint_target", data=np.zeros((state_count, 20)))
    commands.create_dataset("phase", data=["extract"] * state_count, dtype=string)

    objects = file.create_group("objects/usb_plug")
    pose = np.zeros((state_count, 7))
    pose[:, 3] = 1.0
    objects.create_dataset("pose_wxyz", data=pose)
    objects.create_dataset("twist_linear_angular", data=np.zeros((state_count, 6)))

    wrench = file.create_group("wrist_wrench")
    wrench.create_dataset("side_names", data=("left", "right"), dtype=string)
    for name in (
      "force_local_n",
      "force_world_n",
      "torque_local_nm",
      "torque_world_nm",
    ):
      wrench.create_dataset(name, data=np.zeros((state_count, 2, 3)))
    wrench.create_dataset("origin_world_m", data=np.zeros((state_count, 2, 3)))
    wrench.create_dataset(
      "world_from_sensor_rotation",
      data=np.tile(np.eye(3), (state_count, 2, 1, 1)),
    )

    tactile = file.create_group("tactile_proxy")
    tactile.create_dataset(
      "link_names",
      data=tuple(f"hand_r_finger_{index}" for index in range(5)),
      dtype=string,
    )
    for name in ("contact", "contact_count", "normal_force"):
      tactile.create_dataset(name, data=np.zeros((state_count, 5)))
    for name in ("centroid_world", "force_local", "force_world", "torque_world"):
      tactile.create_dataset(name, data=np.zeros((state_count, 5, 3)))

    diagnostics = file.create_group("usb_unplug")
    for field in VARIANT_SUCCESS_FIELDS["usb-unplug"]:
      diagnostics.create_dataset(field, data=np.ones(state_count, dtype=bool))
    diagnostics.create_dataset(
      "withdrawal_resistance_n", data=np.linspace(0.0, 0.4, state_count)
    )

    cameras = file.create_group("cameras")
    for name in ("head", "right_wrist"):
      camera = cameras.create_group(name)
      camera.create_dataset("timestamp", data=camera_times)
      camera.create_dataset("state_index", data=camera_indices)
      camera.create_dataset("rgb", data=np.zeros((4, 2, 3, 3), dtype=np.uint8))
      camera.create_dataset("world_from_camera", data=np.tile(np.eye(4), (4, 1, 1)))


def test_short_unplug_raw_preflight_exports_new_diagnostics(tmp_path):
  outcome = _successful_outcome("usb-unplug")
  raw = tmp_path / "episode.h5"
  sidecar = tmp_path / "episode.json"
  _write_short_unplug_raw(raw, outcome)
  sidecar.write_text("{}", encoding="utf-8")
  source = converter.SourceEpisode(
    episode_index=0,
    hdf5_path=raw,
    sidecar_path=sidecar,
    relative_hdf5_path=raw.name,
    hdf5_sha256="0" * 64,
    hdf5_size_bytes=raw.stat().st_size,
    state_samples=11,
    camera_samples=4,
    metadata={},
    outcome=outcome,
    task="usb-unplug",
  )

  plan = converter._preflight_episode(source)
  assert plan.exported_frames == 3
  assert plan.phase_names == ("extract",)
  outputs = {stream.output for stream in plan.diagnostics}
  assert {
    f"auxiliary.task.usb_unplug.{field}"
    for field in VARIANT_SUCCESS_FIELDS["usb-unplug"]
  } <= outputs
  assert "auxiliary.task.usb_unplug.withdrawal_resistance_n" in outputs
