"""Keep current hold-only USB results distinct from archived return/release data."""

from copy import deepcopy

import pytest

from kaihand_tactile_env.shared.evaluation_rules import StageThresholds, USB_UNPLUG_COMPLETION_PROTOCOL
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_SUCCESS_FIELDS, current_variant_environment, validate_variant_outcome,
)
from kaihand_tactile_env.tasks.usb_unplug import config


def outcome():
  state = {
    "completion_protocol": config.COMPLETION_PROTOCOL,
    "extraction_hold_now": True, "extraction_hold_verified": True,
    "extraction_hold_duration_s": 2.0,
    "insertion_depth_m": -.002, "right_hand_contact_count": 1,
    "socket_contact_count": 0, "table_contact_count": 0,
  }
  return {
    "task": "usb-unplug", "success": True,
    **{key: state[key] for key in (
      "completion_protocol", "extraction_hold_now", "extraction_hold_verified", "extraction_hold_duration_s",
    )},
    **dict.fromkeys(VARIANT_SUCCESS_FIELDS["usb-unplug"], False),
    "final_state": state,
  }


def test_current_hold_result_needs_none_of_the_legacy_return_gates():
  validate_variant_outcome("usb-unplug", outcome(), require_current_usb_protocol=True)
  assert config.COMPLETION_PROTOCOL == USB_UNPLUG_COMPLETION_PROTOCOL
  assert config.EXTRACTION_HOLD_DWELL_S == StageThresholds().usb_extracted_hold_s == 2.0
  assert current_variant_environment("usb-unplug")["completion_protocol"] == config.COMPLETION_PROTOCOL
  assert "completion_protocol" not in current_variant_environment("bulb-unscrew")


@pytest.mark.parametrize("key,value", [
  ("extraction_hold_duration_s", 1.999), ("extraction_hold_duration_s", True),
  ("extraction_hold_duration_s", float("nan")), ("extraction_hold_duration_s", float("inf")),
  ("extraction_hold_now", False), ("extraction_hold_verified", False),
  ("completion_protocol", None), ("completion_protocol", "unknown-version"),
  ("final_state", None), ("task", "usb-insert"),
])
def test_hold_validation_rejects_missing_or_inconsistent_summary_evidence(key, value):
  result = outcome()
  result[key] = value
  with pytest.raises(ValueError):
    validate_variant_outcome("usb-unplug", result, require_current_usb_protocol=True)


@pytest.mark.parametrize("key,value", [
  ("insertion_depth_m", -.00199), ("insertion_depth_m", float("nan")),
  ("insertion_depth_m", True), ("right_hand_contact_count", 0),
  ("right_hand_contact_count", True), ("socket_contact_count", 1),
  ("socket_contact_count", False), ("table_contact_count", 1),
  ("extraction_hold_now", False), ("extraction_hold_verified", False),
  ("extraction_hold_duration_s", 1.999), ("extraction_hold_duration_s", 2.1),
  ("completion_protocol", "unknown-version"),
])
def test_hold_validation_checks_final_physical_state_and_clock(key, value):
  result = deepcopy(outcome())
  result["final_state"][key] = value
  with pytest.raises(ValueError):
    validate_variant_outcome("usb-unplug", result, require_current_usb_protocol=True)


def test_legacy_result_is_readable_but_cannot_use_current_conversion_instruction():
  result = {"task": "usb-unplug", "success": True,
            **dict.fromkeys(VARIANT_SUCCESS_FIELDS["usb-unplug"], True)}
  validate_variant_outcome("usb-unplug", result)
  with pytest.raises(ValueError, match="completion_protocol"):
    validate_variant_outcome("usb-unplug", result, require_current_usb_protocol=True)


def test_bare_success_cannot_be_upgraded_to_new_protocol():
  with pytest.raises(ValueError):
    validate_variant_outcome("usb-unplug", {"task": "usb-unplug", "success": True,
                             "completion_protocol": config.COMPLETION_PROTOCOL})


def test_real_expert_raw_recorder_accepts_current_protocol_without_legacy_return(tmp_path):
  import json

  import h5py

  from kaihand_tactile_env.shared.variant_recording import record_variant_episode
  from kaihand_tactile_env.tasks.usb_unplug.execution import UsbUnplugExecutor
  from kaihand_tactile_env.tasks.usb_unplug.setup import Simulation

  simulation = Simulation(add_genesis_probes=False)
  result = record_variant_episode(
    simulation, UsbUnplugExecutor(simulation), tmp_path / "held_clear", cameras=(),
  )
  assert result["success"], result.get("failure_reason")
  assert result["completion_protocol"] == config.COMPLETION_PROTOCOL
  assert result["extraction_hold_duration_s"] >= 2.0 - 1e-10
  assert not result["placed_on_table"] and not result["released"] and not result["hand_retreated"]
  validate_variant_outcome("usb-unplug", result, require_current_usb_protocol=True)
  with h5py.File(tmp_path / "held_clear/raw/episode.h5") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    assert metadata["completion_protocol"] == config.COMPLETION_PROTOCOL
    assert "extraction_hold_duration_s" in file["usb_unplug"]
    assert len(file["usb_unplug/extraction_hold_duration_s"]) == len(file["state/timestamp"])
    assert file["usb_unplug/extraction_hold_duration_s"][-1] >= 2.0 - simulation.timestep - 1e-10
