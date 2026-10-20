"""Pure-Python staged-evaluation protocol, shared by recording and offline tools.

This catalog describes the *current* code.  It is not evidence that an archived
``staged_v1`` recording used the same observer or native task implementation.
Archived definitions remain authoritative; callers must verify source witnesses
before supplying their implementation identity to ``assert_rules_compatible``.
Nothing here imports physics, recomputes a recorded result, or upgrades a
diagnostic result to a formal result.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any


EVALUATION_VERSION = "staged_v1"
RULES_SCHEMA_VERSION = "kaihand-evaluation-rules-v1"
USB_UNPLUG_COMPLETION_PROTOCOL = "usb-unplug-held-clear-2s-v2"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WHITEBOARD_TASKS = (
  "whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line",
)
STAGE_NAMES = {
  "bulb-screw": ("picked_up", "held_near_socket", "threaded_and_seated"),
  "bulb-screw-inplace": ("stable_inplace_grasp", "threaded_and_seated", "tightening_verified", "released_stably"),
  "bulb-unscrew": ("fully_extracted", "released_stably_on_table"),
  "install-ram": ("picked_up_and_held_near_slot", "inserted_and_seated"),
  "usb-insert": ("picked_up", "held_near_socket", "inserted_and_seated"),
  "usb-unplug": ("fully_extracted", "extracted_and_held_2s"),
  "poker-draw": ("dragged_on_table", "stable_grasp", "terminal_hold_over_3s"),
  **{task: ("picked_up",) for task in WHITEBOARD_TASKS},
}
OBJECT_NAMES = {
  "bulb-screw": "bulb", "bulb-screw-inplace": "bulb", "bulb-unscrew": "bulb", "install-ram": "ram",
  "usb-insert": "usb_plug", "usb-unplug": "usb_plug", "poker-draw": "card",
  **{task: "eraser" for task in WHITEBOARD_TASKS},
}


@dataclass(frozen=True)
class StageThresholds:
  """Fixed SI-unit thresholds; serialize alongside every rollout's result."""

  lift_m: float = 0.02
  near_mouth_m: float = 0.05
  grasp_hold_s: float = 0.15
  finger_load_n: float = 0.025
  maximum_opposition_cosine: float = 0.50
  near_hold_s: float = 0.15
  seating_hold_s: float = 0.20
  clear_hold_s: float = 0.10
  usb_extracted_hold_s: float = 2.0
  table_placement_hold_s: float = 0.05
  release_hold_s: float = 0.20
  minimum_table_support_n: float = 0.01
  maximum_release_linear_speed_m_s: float = 0.02
  maximum_release_angular_speed_rad_s: float = 0.20
  drag_displacement_m: float = 0.02
  drag_hold_s: float = 0.05
  maximum_drag_clearance_m: float = 0.005
  minimum_transport_m: float = 0.02
  poker_terminal_hold_s: float = 3.0
  poker_terminal_position_tolerance_m: float = 0.03
  poker_terminal_face_cosine: float = 0.80
  poker_terminal_linear_speed_m_s: float = 0.02
  poker_terminal_angular_speed_rad_s: float = 0.20
  observation_gap_steps: float = 1.5

  def __post_init__(self):
    for name, value in asdict(self).items():
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if self.maximum_opposition_cosine > 1 or self.poker_terminal_face_cosine > 1:
      raise ValueError("cosine thresholds must not exceed one")


THRESHOLD_UNITS = {
  "lift_m": "m", "near_mouth_m": "m", "grasp_hold_s": "s",
  "finger_load_n": "N", "maximum_opposition_cosine": "1",
  "near_hold_s": "s", "seating_hold_s": "s", "clear_hold_s": "s",
  "usb_extracted_hold_s": "s",
  "table_placement_hold_s": "s", "release_hold_s": "s",
  "minimum_table_support_n": "N", "maximum_release_linear_speed_m_s": "m/s",
  "maximum_release_angular_speed_rad_s": "rad/s", "drag_displacement_m": "m",
  "drag_hold_s": "s", "maximum_drag_clearance_m": "m", "minimum_transport_m": "m",
  "poker_terminal_hold_s": "s", "poker_terminal_position_tolerance_m": "m",
  "poker_terminal_face_cosine": "1", "poker_terminal_linear_speed_m_s": "m/s",
  "poker_terminal_angular_speed_rad_s": "rad/s", "observation_gap_steps": "1",
}
OBSERVATION_UNITS = {
  **{key: "s" for key in (
    "time_s", "simulation_time_s", "contact_interval_start_s", "contact_interval_end_s",
  )},
  **{key: "m" for key in (
    "object_position_xyz_m", "initial_object_position_xyz_m", "work_end_xyz_m",
    "mouth_position_xyz_m", "relative_work_end_socket_xyz_m", "lift_from_initial_m",
    "planar_displacement_m", "near_mouth_distance_m", "table_clearance_m",
    "poker_terminal_position_error_m",
  )},
  "object_pose_wxyz": ["m", "m", "m", "1", "1", "1", "1"],
  "object_twist_linear_angular": ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"],
  "linear_speed_m_s": "m/s", "angular_speed_rad_s": "rad/s",
  "finger_load_n": "N", "finger_contact_normal_vectors_n": "N",
  "opposition_cosine": "1", "poker_face_to_head_cosine": "1", "poker_face_to_robot_cosine": "1",
  **{f"{role}_contact_count": "count" for role in ("hand", "robot", "table", "initial_support", "socket")},
  **{f"{role}_normal_load_n": "N" for role in ("hand", "robot", "table", "initial_support", "socket")},
}
TASK_REGISTRY = {
  task: {
    "object_name": OBJECT_NAMES[task],
    "stage_definitions": {f"stage{index}": name for index, name in enumerate(names, 1)},
    "complete_success_defined": task not in WHITEBOARD_TASKS,
    "scope": "pickup_only; cleaning_metrics_are_separate" if task in WHITEBOARD_TASKS else "ordered_stages",
  }
  for task, names in STAGE_NAMES.items()
}
TASK_REGISTRY["usb-unplug"]["completion_protocol"] = USB_UNPLUG_COMPLETION_PROTOCOL


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                    allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
  return hashlib.sha256(_canonical(value)).hexdigest()


def _version(version: str) -> None:
  if version != EVALUATION_VERSION:
    raise ValueError(f"unsupported staged evaluation version: {version!r}")


def _sha256(value: Any, label: str) -> None:
  if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
    raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _implementation(document: Mapping[str, Any]) -> dict[str, Any]:
  identity = {}
  if document.get("implementation_sha256") is not None:
    _sha256(document["implementation_sha256"], "implementation_sha256")
    identity["implementation_sha256"] = document["implementation_sha256"]
  recorded = document.get("recorded_rules_identity")
  if recorded is not None:
    if not isinstance(recorded, Mapping) or not recorded:
      raise ValueError("recorded_rules_identity must be a nonempty mapping")
    hashes = [key for key in recorded if isinstance(key, str) and (key == "sha256" or key.endswith("_sha256"))]
    if not hashes:
      raise ValueError("recorded_rules_identity requires a SHA-256 identity")
    for key in hashes:
      _sha256(recorded[key], f"recorded_rules_identity.{key}")
    identity["recorded_rules_identity"] = deepcopy(dict(recorded))
  _canonical(identity)
  return identity


def _catalog(version: str, thresholds: StageThresholds) -> dict[str, Any]:
  _version(version)
  return {
    "schema_version": RULES_SCHEMA_VERSION,
    "evaluation_version": version,
    "scope": "current_code_catalog_not_archived_episode",
    "catalog_only": True,
    "task_registry": deepcopy(TASK_REGISTRY),
    "thresholds": asdict(thresholds),
    "threshold_units": deepcopy(THRESHOLD_UNITS),
    "observation_units": deepcopy(OBSERVATION_UNITS),
    "stage_achievement_policy": "ordered_cumulative_ever_reached",
    "time_basis": "mujoco_data_time_seconds",
    "recording_schema": {
      "summary_key": "staged_evaluation", "report_file": "staged_evaluation.json",
      "state_file": "staged_states.jsonl", "stage_flags": "boolean; latched in order",
      "missing_measurements": "null; never infer success from missing evidence",
      "first_times": "seconds or null", "whiteboard_complete_success": None,
      "archive_authority": "recorded stage_definitions and thresholds; never relabel with current catalog",
    },
    "evidence_layers": {
      "native": "preserve recorded task_success and its original validity flags",
      "staged": "separate ordered observation flags; do not replace native task_success",
      "offline": "display archived evidence; no silent rejudgment or diagnostic-to-formal upgrade",
    },
    "identity_limit": "catalog hash excludes observer, native task and recording implementation; verify source witnesses separately",
  }


def export_rules(version: str = EVALUATION_VERSION, *, thresholds: StageThresholds | Mapping[str, float] | None = None,
                 implementation_sha256: str | None = None,
                 recorded_rules_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Export a deterministic, caller-owned catalog and optional supplied identity.

  A supplied digest is an assertion by the caller, not file verification by this
  pure module.  The catalog hash alone is deliberately insufficient for merging
  archived evaluations, even when both say ``staged_v1``.
  """
  if thresholds is None:
    thresholds = StageThresholds()
  elif isinstance(thresholds, Mapping):
    thresholds = StageThresholds(**thresholds)
  if not isinstance(thresholds, StageThresholds):
    raise TypeError("thresholds must be StageThresholds or a threshold mapping")
  document = _catalog(version, thresholds)
  document["catalog_sha256"] = _digest(document)
  document.update(_implementation({"implementation_sha256": implementation_sha256,
                                  "recorded_rules_identity": recorded_rules_identity}))
  document["rules_sha256"] = rules_sha256(document)
  return document


def staged_rule_contract(version: str = EVALUATION_VERSION) -> dict[str, Any]:
  """Current display/schema catalog, explicitly not an archived rule identity."""
  return export_rules(version)


def rules_sha256(document: Mapping[str, Any] | None = None) -> str:
  """Hash canonical JSON, omitting only this protocol's self-hash field."""
  if document is None:
    return export_rules()["rules_sha256"]
  if not isinstance(document, Mapping):
    raise TypeError("rules document must be a mapping")
  return _digest({key: value for key, value in document.items() if key != "rules_sha256"})


def _compatible_descriptor(document: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(document, Mapping):
    raise TypeError("rules document must be a mapping")
  version = document.get("evaluation_version", document.get("version"))
  _version(version)
  if "version" in document and document["version"] != version:
    raise ValueError("conflicting version fields")
  identity = _implementation(document)
  if not identity:
    raise ValueError("catalog/version alone cannot establish compatibility; verified implementation identity is required")
  if "rules_sha256" in document and document["rules_sha256"] != rules_sha256(document):
    raise ValueError("rules document hash mismatch")
  if "schema_version" in document:
    if document["schema_version"] != RULES_SCHEMA_VERSION:
      raise ValueError("unsupported rules schema version")
    catalog = {key: value for key, value in document.items() if key not in (
      "catalog_sha256", "rules_sha256", "implementation_sha256", "recorded_rules_identity",
    )}
    if document.get("catalog_sha256") != _digest(catalog):
      raise ValueError("catalog hash mismatch")
    return {"catalog": catalog, **identity}
  # Archived records are compared as recorded, never completed from defaults or
  # the current task registry. Result flags and episode timestamps are not rules.
  required = ("task", "thresholds", "stage_definitions", "stage_achievement_policy", "time_basis")
  if any(key not in document or document[key] is None for key in required):
    raise ValueError("archived rules descriptor lacks recorded rule fields")
  for key in ("task", "stage_achievement_policy", "time_basis"):
    if not isinstance(document[key], str) or not document[key]:
      raise ValueError(f"archived {key} must be a nonempty string")
  for key in ("thresholds", "stage_definitions"):
    if not isinstance(document[key], Mapping) or not document[key]:
      raise ValueError(f"archived {key} must be a nonempty mapping")
  descriptor = {"version": version, **{key: deepcopy(document[key]) for key in required}, **identity}
  # Optional on older archives. If recorded, a completion protocol is itself
  # rule evidence and cannot be silently discarded during compatibility checks.
  if "completion_protocol" in document:
    protocol = document["completion_protocol"]
    if not isinstance(protocol, str) or not protocol:
      raise ValueError("archived completion_protocol must be a nonempty string")
    descriptor["completion_protocol"] = protocol
  return descriptor


def assert_rules_compatible(*documents: Mapping[str, Any]) -> str:
  """Require identical recorded rule descriptors AND supplied source identity.

  Accepts exported catalogs or archived reports augmented with
  ``implementation_sha256`` and/or ``recorded_rules_identity``.  It does not
  read/verify source files, substitute today's stage labels, or compare outcomes.
  Different descriptor formats are intentionally not implicitly equated.
  Returns the common comparison digest; raises on incomplete/mixed identities.
  """
  if not documents:
    raise ValueError("at least one rules document is required")
  digests = [_digest(_compatible_descriptor(document)) for document in documents]
  if len(set(digests)) != 1:
    raise ValueError("incompatible evaluation rules or implementation identities")
  return digests[0]
