"""Independent staged protocol: no pickup or light-derived success."""
from copy import deepcopy

from evaluation_workbench.task_fields import task_presentation
from kaihand_tactile_env.shared.staged_evaluation import StagedEvaluationMonitor


def sample(**native):
  return {"time_s": 0.0, "seated_now": False, "thread_engaged_now": True,
          "robot_contact_count": 5, "native_task_state": native}


def observe(monitor, value, duration=0.22):
  start = 0 if monitor._last_time is None else monitor._last_time + 0.01
  for tick in range(round(duration / .01) + 1):
    monitor.observe({**value, "time_s": start + tick * .01}, timestep=.01)
  return monitor.report()


def test_initial_thread_or_visual_light_is_not_earned_progress():
  m = StagedEvaluationMonitor("bulb-screw-inplace")
  value = {**sample(initially_threaded_verified=True, bulb_lit=True), "seated_now": True}
  result = observe(m, value)
  assert result["highest_stage"] == 0
  assert result["complete_success"] is False
  assert "picked_up" not in result["named_stages"]


def test_ordered_native_physical_gates_no_pickup_required():
  m = StagedEvaluationMonitor("bulb-screw-inplace")
  value = sample(initially_threaded_verified=True, grasp_verified=True)
  assert observe(m, value)["highest_stage"] == 1
  value["native_task_state"]["effective_rotation_verified"] = True
  value["seated_now"] = True
  assert observe(m, value)["highest_stage"] == 2
  value["native_task_state"]["tightening_verified"] = True
  assert observe(m, value)["highest_stage"] == 3
  value["native_task_state"]["released_stably"] = True
  # A native release flag alone must not hide remaining robot support.
  assert observe(m, value)["highest_stage"] == 3
  value["robot_contact_count"] = 0
  result = observe(m, value)
  assert result["highest_stage"] == 4 and result["complete_success"] is True
  assert list(result["first_times"].values()) == sorted(result["first_times"].values())


def test_stale_or_missing_native_gates_cannot_finish():
  m = StagedEvaluationMonitor("bulb-screw-inplace")
  value = {**sample(), "seated_now": True, "robot_contact_count": 0}
  before = deepcopy(value)
  assert observe(m, value)["highest_stage"] == 0
  assert value == before


def test_display_separates_inplace_and_pickup_tasks():
  result = task_presentation("bulb-screw-inplace")
  assert result["title"] == "灯泡原位旋入"
  assert "原位" in result["task_note"]
  assert "rotation_from_initial_deg" in result["primary_task_keys"]
  assert task_presentation("bulb-screw")["title"] != result["title"]
