"""Explicit contracts for additional, independently identified tasks."""

from __future__ import annotations

import math

import numpy as np

VARIANT_TASKS = (
  "usb-unplug", "whiteboard-wipe-horizontal", "bulb-unscrew",
  "whiteboard-wipe-horizontal-line",
  "bulb-screw-inplace",
)
WHITEBOARD_VARIANT_INK_COUNTS = {
  "whiteboard-wipe-horizontal": 25,
  "whiteboard-wipe-horizontal-line": 8,
}
VARIANT_SUCCESS_FIELDS = {
  # Archived USB return/release results predate completion_protocol.  The
  # current hold-only protocol is validated separately below, never relabelled
  # from these historical flags.
  "usb-unplug": (
    "initially_seated_verified", "grasp_verified", "fully_extracted",
    "placed_on_table", "released", "hand_retreated",
  ),
  "bulb-unscrew": (
    "initially_installed_verified", "grasp_verified", "thread_disengaged",
    "fully_extracted", "placed_on_table", "released", "hand_retreated",
  ),
  "bulb-screw-inplace": (
    "initially_threaded_verified", "grasp_verified", "effective_rotation_verified",
    "tightening_verified", "released_stably",
  ),
  "whiteboard-wipe-horizontal": (
    "pickup_verified", "released_on_table", "home_return_verified",
  ),
  "whiteboard-wipe-horizontal-line": (
    "pickup_verified", "released_on_table", "home_return_verified",
  ),
}


def validate_variant_outcome(task: str, outcome: dict, *, require_current_usb_protocol: bool = False) -> None:
  """Validate recorded protocol gates, rejecting inverse-task relabelling.

  Current collection/conversion callers require the explicit USB hold protocol;
  legacy gates remain readable for archived return/release results only.
  """
  if task not in VARIANT_TASKS:
    raise ValueError(f"not an additional task: {task!r}")
  if not isinstance(outcome, dict) or outcome.get("success") is not True:
    raise ValueError(f"{task}: task outcome is not successful")
  if "task" in outcome and outcome["task"] != task:
    raise ValueError(f"{task}: mismatched task outcome: {outcome['task']!r}")
  if task == "bulb-screw-inplace" and outcome.get("task") != task:
    raise ValueError(f"{task}: explicit independent task identity is required")
  if task == "usb-unplug":
    from ..tasks.usb_unplug import config as unplug

    protocol = outcome.get("completion_protocol")
    if protocol is not None and protocol != unplug.COMPLETION_PROTOCOL:
      raise ValueError(f"{task}: unsupported completion_protocol: {protocol!r}")
    if protocol == unplug.COMPLETION_PROTOCOL:
      if outcome.get("task") != task:
        raise ValueError(f"{task}: explicit independent task identity is required")
      for name in ("extraction_hold_now", "extraction_hold_verified"):
        if outcome.get(name) is not True:
          raise ValueError(f"{task}: physical success gate failed: {name}")
      duration = outcome.get("extraction_hold_duration_s")
      if (type(duration) not in (int, float) or not math.isfinite(duration)
          or duration < unplug.EXTRACTION_HOLD_DWELL_S - 1e-10):
        raise ValueError(f"{task}: physical success gate failed: extraction_hold_duration_s")
      state = outcome.get("final_state")
      if not isinstance(state, dict) or state.get("completion_protocol") != protocol:
        raise ValueError(f"{task}: physical success gate failed: final_state completion_protocol")
      depth = state.get("insertion_depth_m")
      if (state.get("extraction_hold_now") is not True
          or state.get("extraction_hold_verified") is not True
          or type(depth) not in (int, float) or not math.isfinite(depth)
          or depth > unplug.FULL_CLEAR_DEPTH_M
          or type(state.get("right_hand_contact_count")) is not int
          or state["right_hand_contact_count"] <= 0
          or type(state.get("table_contact_count")) is not int
          or state["table_contact_count"] != 0
          or type(state.get("socket_contact_count")) is not int
          or state["socket_contact_count"] != 0):
        raise ValueError(f"{task}: physical success gate failed: final_state held_clear")
      state_duration = state.get("extraction_hold_duration_s")
      if (type(state_duration) not in (int, float) or not math.isfinite(state_duration)
          or state_duration < unplug.EXTRACTION_HOLD_DWELL_S - 1e-10
          or abs(state_duration - duration) > 1e-8):
        raise ValueError(f"{task}: physical success gate failed: final_state extraction_hold_duration_s")
      return
    if require_current_usb_protocol or any(key.startswith("extraction_hold_") for key in outcome):
      raise ValueError(f"{task}: physical success gate failed: completion_protocol; "
                       "legacy return/release data cannot use the current hold-only instruction")
  for name in VARIANT_SUCCESS_FIELDS[task]:
    if outcome.get(name) is not True:
      raise ValueError(f"{task}: physical success gate failed: {name}")
  if task in WHITEBOARD_VARIANT_INK_COUNTS:
    remaining = outcome.get("ink_remaining")
    if (
      not isinstance(remaining, list)
      or len(remaining) != WHITEBOARD_VARIANT_INK_COUNTS[task]
      or any(isinstance(value, bool) for value in remaining)
    ):
      raise ValueError(f"{task}: invalid ink state")
    try:
      values = np.asarray(remaining, dtype=float)
    except (TypeError, ValueError) as error:
      raise ValueError(f"{task}: invalid ink state") from error
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1e-9):
      raise ValueError(f"{task}: ink is not completely cleaned")


def current_variant_environment(task: str) -> dict[str, str]:
  from .config import default_model_path, model_fingerprint, task_config

  if task not in VARIANT_TASKS:
    raise ValueError(f"not an additional task: {task!r}")
  config = task_config(task)
  environment = {
    "mechanics_version": getattr(config, "MECHANICS_VERSION", f"{task}-v1"),
    "model_fingerprint": model_fingerprint(default_model_path(task)),
  }
  if task == "usb-unplug":
    environment["completion_protocol"] = config.COMPLETION_PROTOCOL
  return environment


def create_variant_expert(task: str, seed: int | None = None):
  """Lazy imports keep collection plans and capability queries model-free."""
  if task == "whiteboard-wipe-horizontal-line":
    from ..tasks.whiteboard_wipe_horizontal_line.execution import (
      HorizontalLineWhiteboardWipeExecutor,
    )
    from ..tasks.whiteboard_wipe_horizontal_line.task import (
      HorizontalLineWhiteboardWipeSimulation,
    )

    sim = HorizontalLineWhiteboardWipeSimulation(ink_seed=seed)
    return sim, HorizontalLineWhiteboardWipeExecutor(sim)
  if task == "whiteboard-wipe-horizontal":
    from ..tasks.whiteboard_wipe_horizontal.execution import (
      HorizontalWhiteboardWipeExecutor,
    )
    from ..tasks.whiteboard_wipe_horizontal.task import (
      HorizontalWhiteboardWipeSimulation,
    )

    sim = HorizontalWhiteboardWipeSimulation(ink_seed=seed)
    return sim, HorizontalWhiteboardWipeExecutor(sim)
  if task == "usb-unplug":
    from ..tasks.usb_unplug.execution import UsbUnplugExecutor
    from ..tasks.usb_unplug.setup import UsbUnplugSimulation

    sim = UsbUnplugSimulation(seed=0 if seed is None else seed)
    return sim, UsbUnplugExecutor(sim)
  if task == "bulb-unscrew":
    from ..tasks.bulb_unscrew.execution import BulbUnscrewExecutor
    from ..tasks.bulb_unscrew.task import BulbUnscrewSimulation

    sim = BulbUnscrewSimulation(position_seed=seed)
    return sim, BulbUnscrewExecutor(sim)
  if task == "bulb-screw-inplace":
    from ..tasks.bulb_screw_inplace.execution import BulbScrewInplaceExecutor
    from ..tasks.bulb_screw_inplace.task import BulbScrewInplaceSimulation

    sim = BulbScrewInplaceSimulation(position_seed=seed)
    return sim, BulbScrewInplaceExecutor(sim)
  raise ValueError(f"not an additional task: {task!r}")
