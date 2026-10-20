"""Independent single-line policy, success, Raw and force-reference contracts."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.policy_tasks import (
  HORIZONTAL_WHITEBOARD_TASKS,
  TASK_INSTRUCTIONS,
  TaskPolicyAdapter,
)
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_SUCCESS_FIELDS,
  VARIANT_TASKS,
  create_variant_expert,
  current_variant_environment,
  validate_variant_outcome,
)
from kaihand_tactile_env.shared.variant_recording import record_variant_episode
from kaihand_tactile_env.shared.whiteboard_contact_recorder import (
  start_whiteboard_contact_recording,
)
from kaihand_tactile_env.shared.whiteboard_contact_reference import (
  load_whiteboard_contact_reference,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line import config
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.execution import (
  HorizontalLineWhiteboardWipeExecutor,
)
from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.task import (
  HorizontalLineWhiteboardWipeSimulation,
)

TASK = "whiteboard-wipe-horizontal-line"
GROUP = "whiteboard_wipe_horizontal_line"


def _outcome(task=TASK, count=8):
  return {
    "task": task,
    "success": True,
    **dict.fromkeys(VARIANT_SUCCESS_FIELDS[task], True),
    "ink_remaining": [0.0] * count,
  }


def test_line_factory_and_instruction_have_independent_identity():
  assert TASK in VARIANT_TASKS
  assert TASK in HORIZONTAL_WHITEBOARD_TASKS
  assert "single top horizontal" in TASK_INSTRUCTIONS[TASK]
  assert TASK_INSTRUCTIONS[TASK] != TASK_INSTRUCTIONS["whiteboard-wipe-horizontal"]
  simulation, executor = create_variant_expert(TASK, seed=17)
  assert isinstance(simulation, HorizontalLineWhiteboardWipeSimulation)
  assert isinstance(executor, HorizontalLineWhiteboardWipeExecutor)
  assert simulation.scene == TASK
  assert simulation.task_config is config
  assert simulation.remaining.shape == (8,)
  assert simulation.ink_randomization["seed"] == 17


def test_line_outcome_requires_all_physical_history_gates():
  outcome = _outcome()
  validate_variant_outcome(TASK, outcome)
  for gate in VARIANT_SUCCESS_FIELDS[TASK]:
    invalid = copy.deepcopy(outcome)
    invalid[gate] = False
    with pytest.raises(ValueError, match=gate):
      validate_variant_outcome(TASK, invalid)
  with pytest.raises(ValueError, match="physical success gate"):
    validate_variant_outcome(TASK, {"success": True})


@pytest.mark.parametrize(
  "remaining", ([0.0] * 25, [], [False] * 8, [float("nan")] * 8, [0.001] * 8, [-0.1] * 8)
)
def test_line_outcome_rejects_wrong_count_or_unclean_ink(remaining):
  outcome = _outcome()
  outcome["ink_remaining"] = remaining
  with pytest.raises(ValueError, match="ink"):
    validate_variant_outcome(TASK, outcome)


def test_old_z_keeps_25_patch_gate_and_cannot_relabel_as_line():
  old_task = "whiteboard-wipe-horizontal"
  validate_variant_outcome(old_task, _outcome(old_task, 25))
  with pytest.raises(ValueError, match="ink state"):
    validate_variant_outcome(old_task, _outcome(old_task, 8))
  with pytest.raises(ValueError, match="mismatched task"):
    validate_variant_outcome(TASK, _outcome(old_task, 8))


def test_line_policy_idle_cannot_claim_pickup_release_or_success():
  adapter = TaskPolicyAdapter(TASK, seed=0)
  assert adapter.simulation.scene == TASK
  assert adapter.simulation.remaining.shape == (8,)
  for _ in range(70):
    adapter.step()
  assert not adapter.success
  assert not adapter.state.pickup_verified
  assert not adapter.state.released_on_table
  assert not adapter.state.home_return_verified
  assert adapter.snapshot()["stage"] == "pick_eraser"
  assert adapter.metadata()["ink_pattern"] == config.INK_PATTERN
  assert adapter.metadata()["ink_patch_count"] == 8


def test_line_policy_requires_opposed_grasp_and_lift_before_release():
  adapter = TaskPolicyAdapter.__new__(TaskPolicyAdapter)
  adapter.task = TASK
  adapter.initial_object_pose = np.zeros(7)
  adapter.horizontal_opposed_grasp_verified = False
  adapter.horizontal_pickup_verified = False
  loads = np.zeros(10)
  pose = np.zeros(7)
  sim = SimpleNamespace(
    forces=SimpleNamespace(read=lambda data: SimpleNamespace(normal_force_n=loads)),
    data=None, table_force=0.0, object_pose=lambda name: pose,
    remaining=np.zeros(8), object_twist=lambda name: np.zeros(6),
  )
  adapter.simulation = sim
  pose[2] = 0.15
  adapter._update_horizontal_pickup()
  assert not adapter.horizontal_pickup_verified
  assert not adapter._whiteboard_release_condition()
  pose[2] = 0.0
  sim.table_force = 0.3
  loads[5:7] = [0.06, 0.2]
  adapter._update_horizontal_pickup()
  pose[2] = 0.15
  sim.table_force = 0.0
  adapter._update_horizontal_pickup()
  assert adapter.horizontal_pickup_verified
  loads[:] = 0
  sim.table_force = 0.3
  assert adapter._whiteboard_release_condition()
  sim.remaining[0] = 0.1
  assert not adapter._whiteboard_release_condition()


def test_line_policy_success_includes_both_arm_and_open_hand_home():
  adapter = TaskPolicyAdapter.__new__(TaskPolicyAdapter)
  adapter.task = TASK
  adapter.maximum_lift_m = 0.15
  adapter.release_stable_s = 0.2
  adapter.horizontal_pickup_verified = True
  qpos = np.zeros(21)
  sim = SimpleNamespace(
    remaining=np.zeros(8),
    forces=SimpleNamespace(read=lambda data: SimpleNamespace(normal_force_n=np.zeros(10))),
    object_twist=lambda name: np.zeros(6), data=SimpleNamespace(qpos=qpos),
    _arm_qpos={"right": np.array([0])}, _hand_qpos={"right": np.arange(1, 21)},
    task_config=SimpleNamespace(ARM_HOME={"right": np.zeros(1)}, OPEN_HAND_MAX_ERROR_RAD=0.06),
    spread_grip=np.zeros(20), board_force=0.0, board_tangent_force=0.0, table_force=0.3,
  )
  adapter.simulation = sim
  assert adapter._whiteboard_state().success
  qpos[1] = 0.07
  assert not adapter._whiteboard_state().success
  qpos[:] = 0
  qpos[0] = 0.02
  assert not adapter._whiteboard_state().success
  qpos[:] = 0
  adapter.horizontal_pickup_verified = False
  assert not adapter._whiteboard_state().success


def test_line_raw_uses_post_physics_clock_and_eight_patch_diagnostics(tmp_path):
  sim = HorizontalLineWhiteboardWipeSimulation(add_genesis_probes=False)

  class ShortDiagnostic:
    observer = None

    def run(self):
      sim.step(200)
      return {"success": False, "reason": "short diagnostic, not a successful expert"}

  output = tmp_path / "line_diagnostic"
  result = record_variant_episode(sim, ShortDiagnostic(), output, cameras=())
  assert not result["success"]
  assert result["physics_recording_audit"]["passed"]
  assert result["physics_recording_audit"]["observed_physics_steps"] == 200
  assert sim.physics_observer is None
  assert sim.post_physics_observer is None
  with h5py.File(output / "raw/episode.h5", "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    assert metadata["scene"] == TASK
    assert metadata["initial_state"] == "tabletop-eraser-and-face-up-board"
    assert metadata["ink_pattern"] == "z_top_bar_v1"
    assert metadata["ink_patch_count"] == 8
    assert metadata["mechanics_version"] == config.MECHANICS_VERSION
    timestamps = file["state/timestamp"][:]
    np.testing.assert_allclose(timestamps, np.arange(21) / 100, atol=1e-12)
    for field in ("ink_remaining", "patch_work_j", "patch_stroke_m", "patch_loaded_time_s"):
      assert file[f"{GROUP}/{field}"].shape == (len(timestamps), 8)
    np.testing.assert_array_equal(file[f"{GROUP}/ink_remaining"][:], 1)
    assert "whiteboard_wipe_horizontal" not in file


def test_line_contact_recorder_is_read_only_and_retains_new_task_metadata():
  baseline = TaskPolicyAdapter(TASK, seed=3, guard_mode="observe")
  observed = TaskPolicyAdapter(TASK, seed=3, guard_mode="observe")
  recorder = start_whiteboard_contact_recording(observed)
  assert recorder is not None
  assert recorder.error is None
  for _ in range(10):
    baseline.step()
    observed.step()
  for field in ("qpos", "qvel", "ctrl"):
    np.testing.assert_array_equal(
      getattr(observed.simulation.data, field), getattr(baseline.simulation.data, field)
    )
  assert len(recorder._rows) == 10
  assert recorder._initial_metadata["ink_pattern"] == "z_top_bar_v1"
  recorder.detach()


def _reference(path, *, task=TASK, mechanics=None, count=8, group=GROUP):
  environment = current_variant_environment(TASK)
  metadata = {
    "scene": task, **environment,
    "ink_pattern": "z_top_bar_v1", "ink_patch_count": count,
  }
  if mechanics is not None:
    metadata["mechanics_version"] = mechanics
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["outcome_json"] = json.dumps(_outcome(TASK, count))
    file["state/timestamp"] = [0.0, 0.01]
    file["tactile_contact_force/timestamp"] = [0.0, 0.01]
    names = [f"hand_r_{finger}_link{6 if finger == 'thumb' else 4}" for finger in ("thumb", "index", "middle", "ring", "pinky")]
    file["tactile_contact_force/link_names"] = np.asarray(names, dtype=h5py.string_dtype())
    file["tactile_contact_force/normal_force_n"] = np.zeros((2, 5))
    file[f"{group}/board_normal_force_n"] = [0.0, 2.0]
    file[f"{group}/board_tangent_force_n"] = [0.0, 1.0]
    file[f"{group}/direct_hand_board_force_n"] = [0.0, 0.0]
  return path


def test_line_contact_reference_uses_own_raw_group_and_current_mechanics(tmp_path):
  path = _reference(tmp_path / "line.h5")
  trace = load_whiteboard_contact_reference(task=TASK, reference_hdf5=path)
  assert trace.metadata["task"] == TASK
  assert trace.metadata["ink_patch_count"] == 8
  assert trace.metadata["mechanics_version"] == config.MECHANICS_VERSION
  np.testing.assert_array_equal(trace.channels["board_normal_n"], [0.0, 2.0])


@pytest.mark.parametrize("change", (
  {"task": "whiteboard-wipe-horizontal"},
  {"mechanics": "horizontal_fixed_board_contact_v1"},
  {"count": 25},
  {"group": "whiteboard_wipe_horizontal"},
))
def test_line_contact_reference_rejects_z_or_wrong_mechanics(tmp_path, change):
  path = _reference(tmp_path / "wrong.h5", **change)
  with pytest.raises(ValueError):
    load_whiteboard_contact_reference(task=TASK, reference_hdf5=path)


def test_line_reference_never_falls_back_to_z_demo(tmp_path):
  old = tmp_path / "_metadata/whiteboard-wipe-horizontal"
  old.mkdir(parents=True)
  (old / "replay.json").write_text(json.dumps({
    "task": "whiteboard-wipe-horizontal", "completed": True,
    "source_hdf5": "old_z.h5",
  }))
  with pytest.raises(FileNotFoundError, match=TASK):
    load_whiteboard_contact_reference(task=TASK, demo_root=tmp_path)


def test_line_conversion_preflight_rejects_wrong_scene_even_if_flags_relabelled(tmp_path):
  workcell = Path(__file__).parents[1] / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  import convert_usb_to_lerobot_v3 as converter

  raw = tmp_path / "old_z.h5"
  outcome = _outcome()
  with h5py.File(raw, "w") as file:
    file.attrs["schema_version"] = converter.RAW_SCHEMA
    file.attrs["metadata_json"] = json.dumps({
      "scene": "whiteboard-wipe-horizontal", "active_objects": ["eraser"],
    })
    file.attrs["outcome_json"] = json.dumps(outcome)
  source = converter.SourceEpisode(
    episode_index=0, hdf5_path=raw, sidecar_path=raw.with_suffix(".json"),
    relative_hdf5_path=raw.name, hdf5_sha256="0" * 64,
    hdf5_size_bytes=raw.stat().st_size, state_samples=0, camera_samples=0,
    metadata={}, outcome=outcome, task=TASK,
  )
  with pytest.raises(ValueError, match="source metadata does not match"):
    converter._preflight_episode(source)
