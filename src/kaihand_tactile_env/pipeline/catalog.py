"""Canonical task names and existing evaluation backends.

Stage capabilities stay explicit: a conversion adapter does not imply that a
model can be prepared or evaluated for the same task.
"""

from kaihand_tactile_env.shared.tactile_task_contracts import SHARED_TACTILE_TASKS

TASKS = (
  "pick-place",
  "poker-draw",
  "usb-insert",
  "bulb-screw",
  "vase-wipe",
  "install-ram",
  "whiteboard-wipe",
  "sponge-grasp",
  "usb-unplug",
  "whiteboard-wipe-horizontal",
  "whiteboard-wipe-horizontal-line",
  "bulb-unscrew",
  "bulb-screw-inplace",
)

MODEL_FAMILIES = ("egosteer", "pi05", "pi05+trex", "pi05+cpm", "ftp1", "egotouch", "lingbot-vla2")

EVALUATION_RUNNERS = {
  ("pick-place", "egosteer"): "evaluate_pickplace_policy_batch.py",
  ("pick-place", "pi05"): "evaluate_pickplace_policy_batch.py",
  ("pick-place", "egotouch"): "evaluate_pickplace_policy_batch.py",
  ("pick-place", "lingbot-vla2"): "evaluate_pickplace_lingbot_vla2_batch.py",
  ("poker-draw", "egosteer"): "evaluate_poker_policy_batch.py",
  ("poker-draw", "pi05"): "evaluate_poker_pi05_policy_batch.py",
  ("poker-draw", "pi05+trex"): "evaluate_poker_pi05+trex_policy_batch.py",
  ("poker-draw", "pi05+cpm"): "evaluate_poker_tactile_policy_batch.py",
  ("poker-draw", "ftp1"): "evaluate_poker_tactile_policy_batch.py",
  ("usb-insert", "egosteer"): "evaluate_usb_policy_batch.py",
  ("usb-insert", "pi05"): "evaluate_usb_pi05_policy_batch.py",
  ("bulb-screw", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("bulb-screw", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("bulb-screw", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("install-ram", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("vase-wipe", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "egosteer"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe", "egotouch"): "evaluate_shared_task_policy_batch.py",
  ("sponge-grasp", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("usb-unplug", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe-horizontal", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("whiteboard-wipe-horizontal-line", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("bulb-unscrew", "pi05"): "evaluate_shared_task_policy_batch.py",
  ("bulb-screw-inplace", "pi05"): "evaluate_shared_task_policy_batch.py",
}

# Interface registration is independent of task-specific trained assets.
# CPM remains poker-only: its current base/scorer implementation is task-bound.
EVALUATION_RUNNERS.update({
  (task, "ftp1"): "evaluate_task_tactile_policy_batch.py"
  for task in SHARED_TACTILE_TASKS
})
