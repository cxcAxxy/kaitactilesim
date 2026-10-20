import json

import numpy as np
import pytest

from kaihand_tactile_env.shared.policy_tasks import create_task_policy_adapter
from kaihand_tactile_env.shared.staged_recording import StagedStateLog
from kaihand_tactile_env.shared.whiteboard_contact_recorder import start_whiteboard_contact_recording


@pytest.mark.parametrize("task", ["whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"])
def test_integrated_contact_hook_geometry_and_log(task, tmp_path):
  adapter = create_task_policy_adapter(task, 0, guard_mode="observe")
  contact = start_whiteboard_contact_recording(adapter)
  assert contact.error is None
  original_callback = adapter.simulation.evaluation_contact_observer
  adapter.enable_staged_evaluation()
  sim = adapter.simulation
  observer = adapter.staged_evaluation
  log = StagedStateLog(tmp_path, observer, identity={"seed": 0, "prediction_horizon": 50,
     "execute_steps": 50, "checkpoint_sha256": "a" * 64})
  log.record(0)
  for _ in range(5):
    adapter.step()
  staged = observer.report()
  assert staged["measurements"]["simulation_time_s"] == pytest.approx(sim.data.time)
  assert staged["measurements"]["contact_force_source"] == "integrated_solver_contacts_before_post_step_mj_forward"
  assert staged["continuous_metrics"]["elapsed_s"] == pytest.approx(sim.data.time)
  assert staged["stage_count"] == 1
  assert staged["complete_success"] is None
  assert len(contact._rows) == 5
  qpos, qvel, ctrl = sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ctrl.copy()
  log.record(1)
  payload = log.finish({"physics_valid": True, "task_success": adapter.success,
                        "diagnostic_only": True, "formal_metrics_valid": False})
  for before, after in ((qpos, sim.data.qpos), (qvel, sim.data.qvel), (ctrl, sim.data.ctrl)):
    assert np.array_equal(before, after)
  adapter.detach_staged_contact_observer()
  assert sim.evaluation_contact_observer is original_callback
  lines = [json.loads(line) for line in (tmp_path / "staged_states.jsonl").read_text().splitlines()]
  assert len(lines) == 2
  assert all(row["seed"] == 0 and row["prediction_horizon"] == row["execute_steps"] == 50 for row in lines)
  assert payload["full_duration_evaluation"] is True


@pytest.mark.parametrize("task,count", [("bulb-screw", 3), ("bulb-unscrew", 2), ("install-ram", 2), ("usb-unplug", 2)])
def test_native_shared_adapters_staged_without_mechanics_changes(task, count):
  adapter = create_task_policy_adapter(task, 0, guard_mode="observe")
  adapter.enable_staged_evaluation()
  for _ in range(5):
    adapter.step()
  summary = adapter.report()
  staged = summary["staged_evaluation"]
  assert staged["stage_count"] == count
  assert staged["highest_stage"] == 0
  assert summary["success"] is adapter.success
  assert staged["measurements"]["near_mouth_distance_m"] is not None
  assert staged["measurements"]["native_task_state"] is not None
