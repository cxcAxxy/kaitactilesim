import numpy as np
import pytest
from kaihand_tactile_env.shared.action_diagnostics import (
  ActionClipDiagnostics,
  merge_action_clip_summaries,
)


def _tracker():
  return ActionClipDiagnostics(
    ("arm", "index", "middle"),
    np.array([-1.0, -1.0, -1.0]),
    np.array([1.0, 1.0, 1.0]),
  )


def test_action_clipping_is_attributed_to_joint_and_task_stage():
  tracker = _tracker()

  event = tracker.record(np.array([-1.2, 0.0, 1.4]), "grasp_and_lift")
  tracker.record(np.array([0.0, 2.0, 0.0]), "screw_thread")
  summary = tracker.summary()

  assert event["clip_mask"] == [True, False, True]
  assert event["clipped_joint_names"] == ["arm", "middle"]
  assert event["applied_action"] == pytest.approx([-1.0, 0.0, 1.0])
  assert event["maximum_abs_excess_rad"] == pytest.approx(0.4)
  assert summary["action_steps"] == 2
  assert summary["total_values"] == 6
  assert summary["clipped_values"] == 3
  assert summary["by_joint"]["arm"]["lower_clips"] == 1
  assert summary["by_joint"]["index"]["upper_clips"] == 1
  assert summary["by_joint"]["middle"]["upper_clips"] == 1
  assert summary["by_stage"]["grasp_and_lift"]["clipped_values_by_joint"] == {
    "arm": 1,
    "middle": 1,
  }
  assert summary["by_stage"]["screw_thread"]["clipped_values_by_joint"] == {"index": 1}


def test_action_clipping_summaries_merge_across_trials():
  first = _tracker()
  first.record(np.array([-1.2, 0.0, 1.4]), "grasp_and_lift")
  first.record(np.array([0.0, 2.0, 0.0]), "screw_thread")
  second = _tracker()
  second.record(np.array([-2.0, 0.0, 0.0]), "screw_thread")

  merged = merge_action_clip_summaries([first.summary(), second.summary()])

  assert merged["trial_count"] == 2
  assert merged["action_steps"] == 3
  assert merged["total_values"] == 9
  assert merged["clipped_values"] == 4
  assert merged["by_joint"]["arm"]["maximum_abs_excess_rad"] == pytest.approx(1.0)
  assert merged["by_stage"]["screw_thread"]["action_steps"] == 2
  assert merged["by_stage"]["screw_thread"]["clipped_values_by_joint"] == {
    "arm": 1,
    "index": 1,
  }


def test_empty_action_clipping_merge_is_well_formed():
  assert merge_action_clip_summaries([]) == {
    "trial_count": 0,
    "action_steps": 0,
    "total_values": 0,
    "clipped_values": 0,
    "clipped_fraction": 0.0,
    "by_joint": {},
    "by_stage": {},
  }
