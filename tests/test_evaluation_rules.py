"""The shared protocol must not alter physics-side evaluation or archive rules."""

import ast
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from kaihand_tactile_env.shared import evaluation_rules as rules


EXPECTED_DEFAULTS = {
  "lift_m": 0.02, "near_mouth_m": 0.05, "grasp_hold_s": 0.15,
  "finger_load_n": 0.025, "maximum_opposition_cosine": 0.50,
  "near_hold_s": 0.15, "seating_hold_s": 0.20, "clear_hold_s": 0.10,
  "usb_extracted_hold_s": 2.0,
  "table_placement_hold_s": 0.05, "release_hold_s": 0.20,
  "minimum_table_support_n": 0.01, "maximum_release_linear_speed_m_s": 0.02,
  "maximum_release_angular_speed_rad_s": 0.20, "drag_displacement_m": 0.02,
  "drag_hold_s": 0.05, "maximum_drag_clearance_m": 0.005, "minimum_transport_m": 0.02,
  "poker_terminal_hold_s": 3.0, "poker_terminal_position_tolerance_m": 0.03,
  "poker_terminal_face_cosine": 0.80, "poker_terminal_linear_speed_m_s": 0.02,
  "poker_terminal_angular_speed_rad_s": 0.20, "observation_gap_steps": 1.5,
}
EXPECTED_STAGES = {
  "bulb-screw": ("picked_up", "held_near_socket", "threaded_and_seated"),
  "bulb-screw-inplace": ("stable_inplace_grasp", "threaded_and_seated", "tightening_verified", "released_stably"),
  "bulb-unscrew": ("fully_extracted", "released_stably_on_table"),
  "install-ram": ("picked_up_and_held_near_slot", "inserted_and_seated"),
  "usb-insert": ("picked_up", "held_near_socket", "inserted_and_seated"),
  "usb-unplug": ("fully_extracted", "extracted_and_held_2s"),
  "poker-draw": ("dragged_on_table", "stable_grasp", "terminal_hold_over_3s"),
  "whiteboard-wipe": ("picked_up",),
  "whiteboard-wipe-horizontal": ("picked_up",),
  "whiteboard-wipe-horizontal-line": ("picked_up",),
}


def archived_descriptor():
  return {
    "version": "staged_v1", "task": "poker-draw", "thresholds": EXPECTED_DEFAULTS.copy(),
    # Deliberately not today's label: archives, not current display names, win.
    "stage_definitions": {"stage1": "archived_drag_label", "stage2": "archived_grasp_label"},
    "stage_achievement_policy": "ordered_cumulative_ever_reached",
    "time_basis": "mujoco_data_time_seconds",
    "recorded_rules_identity": {
      "observer_sha256": "a" * 64, "recording_sha256": "b" * 64,
      "evaluation_source_sha256": "c" * 64, "definitions_thresholds_sha256": "d" * 64,
      "sha256": "e" * 64,
    },
    "complete_success": False, "time_s": 1.0,
  }


def test_exact_original_thresholds_and_units():
  thresholds = rules.StageThresholds()
  assert asdict(thresholds) == EXPECTED_DEFAULTS
  assert set(rules.THRESHOLD_UNITS) == set(EXPECTED_DEFAULTS)
  assert rules.THRESHOLD_UNITS["observation_gap_steps"] == "1"
  assert rules.THRESHOLD_UNITS["poker_terminal_hold_s"] == "s"
  with pytest.raises(FrozenInstanceError):
    thresholds.lift_m = 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_thresholds_preserve_rejection(value):
  with pytest.raises(ValueError, match="finite and positive"):
    rules.StageThresholds(lift_m=value)


@pytest.mark.parametrize("key", ["maximum_opposition_cosine", "poker_terminal_face_cosine"])
def test_cosine_threshold_bounds_preserved(key):
  with pytest.raises(ValueError, match="cosine"):
    rules.StageThresholds(**{key: 1.01})
  assert getattr(rules.StageThresholds(**{key: 1}), key) == 1


def test_exact_stage_registry_and_undefined_whiteboard_success():
  assert rules.EVALUATION_VERSION == "staged_v1"
  assert rules.STAGE_NAMES == EXPECTED_STAGES
  assert set(rules.TASK_REGISTRY) == set(rules.OBJECT_NAMES) == set(EXPECTED_STAGES)
  for task in rules.WHITEBOARD_TASKS:
    assert rules.TASK_REGISTRY[task]["complete_success_defined"] is False
    assert rules.TASK_REGISTRY[task]["stage_definitions"] == {"stage1": "picked_up"}
  assert "pick-place" not in rules.TASK_REGISTRY
  assert rules.TASK_REGISTRY["usb-unplug"]["completion_protocol"] == "usb-unplug-held-clear-2s-v2"
  assert rules.USB_UNPLUG_COMPLETION_PROTOCOL == "usb-unplug-held-clear-2s-v2"
  assert all("completion_protocol" not in entry for task, entry in rules.TASK_REGISTRY.items()
             if task != "usb-unplug")


def test_monitor_reexports_shared_definitions_and_other_task_logic_is_unchanged():
  from kaihand_tactile_env.shared import staged_evaluation as monitor

  for name in ("StageThresholds", "EVALUATION_VERSION", "STAGE_NAMES", "OBJECT_NAMES", "FINGERS", "WHITEBOARD_TASKS"):
    assert getattr(monitor, name) is getattr(rules, name)
  tree = ast.parse(Path(monitor.__file__).read_text())
  class RemoveIntentionalTaskExtensions(ast.NodeTransformer):
    """Check original logic outside inplace-screw and USB's new hold rule.

    In particular the original bulb-unscrew placement/release implementation
    remains byte-for-AST identical after it is split away from usb-unplug.
    """
    def visit_Dict(self, node):
      self.generic_visit(node)
      kept = [(k, v) for k, v in zip(node.keys, node.values, strict=True)
              if not (isinstance(k, ast.Constant) and k.value == "bulb-screw-inplace")
              and not (k is None and isinstance(v, ast.IfExp)
                       and any(isinstance(item, ast.Name) and item.id == "USB_UNPLUG_COMPLETION_PROTOCOL"
                               for item in ast.walk(v)))]
      node.keys, node.values = [k for k, _ in kept], [v for _, v in kept]
      return node

    def visit_Set(self, node):
      self.generic_visit(node)
      node.elts = [v for v in node.elts if not isinstance(v, ast.Constant)
                   or v.value != "bulb-screw-inplace"]
      return node

    def visit_If(self, node):
      self.generic_visit(node)
      usb_hold_branch = (
        isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1
        and isinstance(node.test.left, ast.Attribute) and node.test.left.attr == "task"
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "usb-unplug"
        and any(isinstance(item, ast.Constant) and item.value == "usb_extracted_and_held"
                for statement in node.body for item in ast.walk(statement))
      )
      if usb_hold_branch:
        assert len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
        return node.orelse[0]
      if (isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1
          and isinstance(node.test.left, ast.Attribute) and node.test.left.attr == "task"
          and isinstance(node.test.comparators[0], ast.Constant)
          and node.test.comparators[0].value == "bulb-unscrew"):
        node.test.ops = [ast.In()]
        node.test.comparators = [ast.Set(elts=[ast.Constant("bulb-unscrew"), ast.Constant("usb-unplug")])]
      if (isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1
          and isinstance(node.test.left, ast.Attribute)
          and node.test.left.attr == "task"
          and isinstance(node.test.comparators[0], ast.Constant)
          and node.test.comparators[0].value == "bulb-screw-inplace"):
        assert len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
        return node.orelse[0]
      return node

  tree = RemoveIntentionalTaskExtensions().visit(tree)
  logic = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
           and node.name != "StageThresholds"]
  digest = hashlib.sha256(ast.dump(ast.Module(body=logic, type_ignores=[]), include_attributes=False).encode()).hexdigest()
  # Original A snapshot's functions/classes, excluding only extracted thresholds.
  assert digest == "689da5717f6333b40f50e523028aa5f1e7388ba68bef5d2bc82201ae0d94966c"


def test_pure_import_works_without_site_numpy_or_mujoco():
  source = Path(rules.__file__).resolve().parents[2]
  code = (
    "import sys; sys.path.insert(0, " + repr(str(source)) + "); "
    "from kaihand_tactile_env.shared.evaluation_rules import staged_rule_contract; "
    "assert staged_rule_contract()['catalog_only'] is True; "
    "assert 'numpy' not in sys.modules and 'mujoco' not in sys.modules"
  )
  subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], check=True, capture_output=True, text=True)


def test_deterministic_export_hashes_are_caller_owned():
  first, second = rules.export_rules(), rules.export_rules()
  assert first == second
  assert json.loads(json.dumps(first, allow_nan=False)) == first
  assert first["rules_sha256"] == rules.rules_sha256(first) == rules.rules_sha256()
  assert first["scope"] == "current_code_catalog_not_archived_episode"
  first["task_registry"]["usb-insert"]["stage_definitions"]["stage1"] = "mutated"
  first["thresholds"]["lift_m"] = 999
  assert rules.export_rules() == second
  changed = rules.export_rules(thresholds={"lift_m": 0.03})
  assert changed["catalog_sha256"] != second["catalog_sha256"]
  assert changed["rules_sha256"] != second["rules_sha256"]


def test_catalog_identity_does_not_claim_source_implementation_equivalence():
  first = rules.export_rules(implementation_sha256="a" * 64)
  second = rules.export_rules(implementation_sha256="b" * 64)
  assert first["catalog_sha256"] == second["catalog_sha256"]
  assert first["rules_sha256"] != second["rules_sha256"]
  assert len(rules.assert_rules_compatible(first, deepcopy(first))) == 64
  with pytest.raises(ValueError, match="incompatible"):
    rules.assert_rules_compatible(first, second)
  with pytest.raises(ValueError, match="implementation identity"):
    rules.assert_rules_compatible(rules.staged_rule_contract())


def test_export_tamper_is_not_accepted():
  document = rules.export_rules(implementation_sha256="a" * 64)
  document["thresholds"]["lift_m"] = 0.04
  with pytest.raises(ValueError, match="hash mismatch"):
    rules.assert_rules_compatible(document)
  # Updating only the outer hash must not conceal the bound catalog mutation.
  document["rules_sha256"] = rules.rules_sha256(document)
  with pytest.raises(ValueError, match="catalog hash mismatch"):
    rules.assert_rules_compatible(document)


def test_archived_definition_is_authoritative_and_outcomes_are_not_rules():
  first = archived_descriptor()
  before = deepcopy(first)
  second = deepcopy(first)
  second["complete_success"], second["time_s"] = True, 77.0
  assert rules.assert_rules_compatible(first) == rules.assert_rules_compatible(first, second)
  assert first == before
  assert first["stage_definitions"]["stage1"] == "archived_drag_label"


def test_old_usb_archives_are_not_upgraded_to_the_new_held_clear_protocol():
  archived = archived_descriptor()
  archived["task"] = "usb-unplug"
  archived["thresholds"].pop("usb_extracted_hold_s")
  archived["stage_definitions"] = {
    "stage1": "fully_extracted", "stage2": "released_stably_on_table",
  }
  before = deepcopy(archived)
  rules.assert_rules_compatible(archived, deepcopy(archived))
  assert archived == before
  assert "completion_protocol" not in archived
  current = deepcopy(archived)
  current["stage_definitions"]["stage2"] = "extracted_and_held_2s"
  current["thresholds"]["usb_extracted_hold_s"] = 2.0
  current["completion_protocol"] = rules.USB_UNPLUG_COMPLETION_PROTOCOL
  with pytest.raises(ValueError, match="incompatible"):
    rules.assert_rules_compatible(archived, current)


def test_recorded_completion_protocol_is_part_of_archive_rule_compatibility():
  first = archived_descriptor()
  first["completion_protocol"] = rules.USB_UNPLUG_COMPLETION_PROTOCOL
  rules.assert_rules_compatible(first, deepcopy(first))
  for protocol in (None, "older-protocol"):
    second = deepcopy(first)
    if protocol is None:
      second.pop("completion_protocol")
    else:
      second["completion_protocol"] = protocol
    with pytest.raises(ValueError, match="incompatible"):
      rules.assert_rules_compatible(first, second)
  first["completion_protocol"] = ""
  with pytest.raises(ValueError, match="completion_protocol"):
    rules.assert_rules_compatible(first)


@pytest.mark.parametrize("field,value", [
  ("task", "usb-insert"),
  ("thresholds", {"lift_m": 0.3}),
  ("stage_definitions", {"stage1": "dragged_on_table"}),
  ("stage_achievement_policy", "current_state"),
  ("time_basis", "control_clock"),
])
def test_distinct_archived_rule_fields_are_not_mixed(field, value):
  first, second = archived_descriptor(), archived_descriptor()
  second[field] = value
  with pytest.raises(ValueError, match="incompatible"):
    rules.assert_rules_compatible(first, second)


def test_same_version_and_thresholds_but_changed_observer_are_not_mixed():
  first, second = archived_descriptor(), archived_descriptor()
  second["recorded_rules_identity"]["observer_sha256"] = "f" * 64
  with pytest.raises(ValueError, match="incompatible"):
    rules.assert_rules_compatible(first, second)


def test_missing_or_malformed_source_identity_is_rejected():
  document = archived_descriptor()
  del document["recorded_rules_identity"]
  with pytest.raises(ValueError, match="implementation identity"):
    rules.assert_rules_compatible(document)
  for identity in ({}, {"label": "same_version"}, {"observer_sha256": "not-a-hash"}):
    document["recorded_rules_identity"] = identity
    with pytest.raises(ValueError):
      rules.assert_rules_compatible(document)
  with pytest.raises(ValueError, match="SHA-256"):
    rules.export_rules(implementation_sha256="staged_v1")


def test_unknown_versions_and_incomplete_archived_descriptors_rejected():
  with pytest.raises(ValueError, match="unsupported"):
    rules.staged_rule_contract("staged_v2")
  first, second = archived_descriptor(), archived_descriptor()
  second["version"] = "staged_v2"
  with pytest.raises(ValueError, match="unsupported"):
    rules.assert_rules_compatible(first, second)
  second = archived_descriptor()
  del second["stage_definitions"]
  with pytest.raises(ValueError, match="lacks"):
    rules.assert_rules_compatible(second)
  with pytest.raises(ValueError, match="at least one"):
    rules.assert_rules_compatible()


def test_recording_version_comes_from_same_protocol():
  from kaihand_tactile_env.shared import staged_recording

  assert staged_recording.EVALUATION_VERSION is rules.EVALUATION_VERSION
  tree = ast.parse(Path(staged_recording.__file__).read_text())
  # The module's descriptive docstring may still mention its artifact version.
  assert not any(isinstance(node, ast.Constant) and node.value == "staged_v1" for node in ast.walk(tree))
