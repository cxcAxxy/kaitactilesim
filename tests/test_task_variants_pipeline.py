"""Addition-only task identities, physical outcome gates, and public plans."""

import copy
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.pipeline.catalog import EVALUATION_RUNNERS, TASKS
from kaihand_tactile_env.pipeline.collection import COLLECTION_SPECS
from kaihand_tactile_env.pipeline.conversion import ADAPTERS as CONVERSION_ADAPTERS
from kaihand_tactile_env.pipeline.deployment import adapter_for as deployment_adapter
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS, TaskPolicyAdapter
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_SUCCESS_FIELDS,
  VARIANT_TASKS,
  validate_variant_outcome,
)
from kaihand_tactile_env.shared.variant_recording import record_variant_episode


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_additions_have_explicit_public_backends_and_distinct_instructions(task):
  assert task in TASKS
  command = COLLECTION_SPECS[task].build_command(Path("/repo"), "/python", Path("/out"), 42)
  assert command[1].endswith("record_task_variant.py")
  assert command[command.index("--task") + 1] == task
  assert CONVERSION_ADAPTERS[(task, "pi05")].backend == "convert_shared_to_lerobot.py"
  assert CONVERSION_ADAPTERS[(task, "lerobot-v3")].backend == "convert_kaihand_to_lerobot_v3.py"
  assert EVALUATION_RUNNERS[(task, "pi05")] == "evaluate_shared_task_policy_batch.py"
  assert all(deployment_adapter(task, "pi05").capabilities.values())
  assert TASK_INSTRUCTIONS[task]


@pytest.mark.parametrize("task", VARIANT_TASKS)
def test_new_task_success_requires_each_physical_history_gate(task):
  outcome = {"task": task, "success": True,
             **dict.fromkeys(VARIANT_SUCCESS_FIELDS[task], True)}
  if task in {"whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}:
    count = 8 if task.endswith("-line") else 25
    outcome["ink_remaining"] = [0.0] * count
  validate_variant_outcome(task, outcome)
  for name in VARIANT_SUCCESS_FIELDS[task]:
    invalid = copy.deepcopy(outcome)
    invalid[name] = False
    with pytest.raises(ValueError, match=name):
      validate_variant_outcome(task, invalid)
  with pytest.raises(ValueError):
    validate_variant_outcome(task, {"success": True})


@pytest.mark.parametrize("old,new,geom", (
  ("usb-insert", "usb-unplug", "usb_plug_handle"),
  ("bulb-screw", "bulb-unscrew", "bulb_globe"),
  ("whiteboard-wipe", "whiteboard-wipe-horizontal", "board_surface"),
  ("whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line", "board_surface"),
))
def test_variant_registration_never_disables_original_collision_or_allows_relabelling(old, new, geom):
  assert default_model_path(old) != default_model_path(new)
  assert model_fingerprint(default_model_path(old)) != model_fingerprint(default_model_path(new))
  assert TASK_INSTRUCTIONS.get(old) != TASK_INSTRUCTIONS[new]
  for active, inactive in ((old, new), (new, old)):
    sim = ArmHandSimulation(scene=active, add_genesis_probes=False)
    geom_id = sim.model.geom(geom).id
    assert sim.model.geom_contype[geom_id] or sim.model.geom_conaffinity[geom_id]
    assert sim.model.geom_rgba[geom_id, 3] > 0
    sim.reset()
    assert sim.model.geom_contype[geom_id] or sim.model.geom_conaffinity[geom_id]
    with pytest.raises(ValueError, match="independent task models"):
      sim.set_scene(inactive)
    assert sim.scene == active


def test_horizontal_model_policy_uses_same_physical_cleaning_gates_in_new_frame():
  adapter = TaskPolicyAdapter("whiteboard-wipe-horizontal", 0)
  assert adapter.simulation.scene == "whiteboard-wipe-horizontal"
  np.testing.assert_array_equal(adapter.simulation.task_config.BOARD_NORMAL, [0, 0, 1])
  assert adapter.object_name == "eraser"
  assert not adapter.success
  adapter.step()
  assert not adapter.success
  assert adapter.snapshot()["stage"] == "pick_eraser"
  for _ in range(60):
    adapter.step()
  assert not adapter.state.released_on_table
  assert not adapter.state.home_return_verified
  assert not adapter.state.pickup_verified
  assert not adapter.success
  assert adapter.stage() == "pick_eraser"


def test_horizontal_pickup_requires_opposed_grasp_before_unloaded_lift():
  from types import SimpleNamespace

  adapter = TaskPolicyAdapter("whiteboard-wipe-horizontal", 0)
  loads = np.zeros(10)
  pose = adapter.initial_object_pose.copy()
  sim = SimpleNamespace(
    forces=SimpleNamespace(read=lambda data: SimpleNamespace(normal_force_n=loads)),
    data=None, table_force=0.0, object_pose=lambda name: pose, remaining=np.ones(25),
  )
  adapter.simulation = sim
  pose[2] += 0.15
  adapter._update_horizontal_pickup()
  assert not adapter.horizontal_pickup_verified
  assert not adapter._whiteboard_release_condition()
  pose[2] = adapter.initial_object_pose[2]
  sim.table_force = 0.3
  loads[5:7] = [0.06, 0.2]
  adapter._update_horizontal_pickup()
  assert adapter.horizontal_opposed_grasp_verified
  assert not adapter.horizontal_pickup_verified
  pose[2] += 0.15
  sim.table_force = 0.0
  adapter._update_horizontal_pickup()
  assert adapter.horizontal_pickup_verified
  assert not adapter._whiteboard_release_condition()


@pytest.mark.parametrize("task", ("usb-unplug", "bulb-unscrew"))
def test_installed_policy_state_cannot_succeed_from_reset_or_light_state(task):
  adapter = TaskPolicyAdapter(task, 0)
  for _ in range(60):
    adapter.step()
  assert not adapter.success
  assert adapter.stage() == "grasp_installed_object"
  assert not adapter.state.grasp_verified
  assert not adapter.state.fully_extracted
  assert not adapter.snapshot()["success"]


def test_variant_recorder_keeps_failed_outcome_and_actual_physics_clock(tmp_path):
  import h5py
  from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
    HorizontalWhiteboardWipeSimulation,
  )

  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)

  class ShortDiagnostic:
    observer = None

    def run(self):
      for _ in range(200):
        sim.step()
      return {"success": False, "reason": "short recording diagnostic, not an expert"}

  result = record_variant_episode(sim, ShortDiagnostic(), tmp_path / "diagnostic", cameras=())
  assert result["success"] is False
  assert result["physics_recording_audit"]["observed_physics_steps"] == 200
  assert result["physics_recording_audit"]["expected_physics_steps"] == 200
  assert result["physics_recording_audit"]["terminal_physics_state_unchanged"]
  assert sim.physics_observer is None
  assert sim.post_physics_observer is None
  with h5py.File(tmp_path / "diagnostic/raw/episode.h5") as file:
    timestamps = file["state/timestamp"][:]
    np.testing.assert_allclose(timestamps, np.arange(21) / 100, atol=1e-12)
    assert len(file["whiteboard_wipe_horizontal/ink_remaining"]) == len(timestamps)
    assert file["whiteboard_wipe_horizontal/ink_remaining"][:].min() == 1


def test_horizontal_pre_and_post_hooks_keep_distinct_unmodified_physics_epochs():
  from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
    HorizontalWhiteboardWipeSimulation,
  )

  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  pre, post = [], []
  sim.physics_observer = lambda current: pre.append(float(current.data.time))
  sim.post_physics_observer = lambda current: post.append(float(current.data.time))
  sim.step(3)
  np.testing.assert_allclose(pre, np.arange(3) * sim.timestep, atol=1e-12)
  np.testing.assert_allclose(post, np.arange(1, 4) * sim.timestep, atol=1e-12)
  assert sim.data.time == post[-1]


def test_variant_recorder_refuses_existing_output(tmp_path):
  from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import (
    HorizontalWhiteboardWipeSimulation,
  )

  sim = HorizontalWhiteboardWipeSimulation(add_genesis_probes=False)
  output = tmp_path / "existing"
  output.mkdir()
  (output / "keep.txt").write_text("user data")
  with pytest.raises(FileExistsError):
    record_variant_episode(sim, object(), output, cameras=())
  assert (output / "keep.txt").read_text() == "user data"


def test_variant_recorder_rejects_indirect_thread_carriage_force(tmp_path):
  from kaihand_tactile_env.shared.task_variants import create_variant_expert

  sim, expert = create_variant_expert("bulb-unscrew", 0)
  dof = sim.model.joint("bulb_thread_depth").dofadr[0]

  def inject_carriage_force(simulation):
    simulation.data.qfrc_applied[dof] = 0.1

  sim.physics_observer = inject_carriage_force

  class ShortDiagnostic:
    observer = None
    monitor = expert.monitor

    def run(self):
      sim.step()
      return {"success": False}

  with pytest.raises(RuntimeError, match="mechanism force assistance"):
    record_variant_episode(sim, ShortDiagnostic(), tmp_path / "carriage_force", cameras=())
  assert sim.physics_observer is inject_carriage_force


def test_variant_recorder_rejects_pose_rewrite_after_final_physics_step(tmp_path):
  from kaihand_tactile_env.shared.task_variants import create_variant_expert

  sim, expert = create_variant_expert("usb-unplug", 0)

  class InvalidDiagnostic:
    observer = None
    monitor = expert.monitor

    def run(self):
      sim.step()
      sim.data.qpos[sim._object_qpos["usb_plug"]] += 0.01
      return {"success": False}

  with pytest.raises(RuntimeError, match="terminal state changed"):
    record_variant_episode(sim, InvalidDiagnostic(), tmp_path / "terminal_rewrite", cameras=())


@pytest.mark.parametrize("task", ("usb-unplug", "bulb-unscrew"))
def test_installed_raw_records_contact_forces_without_visual_target_geoms(tmp_path, task):
  import h5py
  from kaihand_tactile_env.shared.task_variants import create_variant_expert

  sim, expert = create_variant_expert(task, 0)

  class ShortInstalledDiagnostic:
    observer = None
    monitor = expert.monitor

    def run(self):
      for _ in range(100):
        sim.task_phase = "installed_diagnostic"
        sim.step()
        self.monitor.update()
      return {"success": False, "reason": "short installed diagnostic, not expert"}

  output = tmp_path / task
  result = record_variant_episode(sim, ShortInstalledDiagnostic(), output, cameras=())
  assert not result["success"]
  assert result["physics_recording_audit"]["observed_physics_steps"] == 100
  assert result["physics_recording_audit"]["passed"]
  with h5py.File(output / "raw/episode.h5") as file:
    np.testing.assert_allclose(
      file["tactile_contact_force/timestamp"][:], file["state/timestamp"][:], atol=1e-12,
    )
    assert "matching recorded state timestamp" in file["tactile_contact_force"].attrs["clock_semantics"]
    assert file["commands/phase"].asstr()[-1] in {"installed_diagnostic", "terminal_settle"}
    group = file[task.replace("-", "_")]
    assert all(len(values) == len(file["state/timestamp"]) for values in group.values())
