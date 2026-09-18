"""Reject mislabelled or incomplete USB demonstrations before format export."""

from copy import deepcopy
from pathlib import Path
from runpy import run_path

import pytest

COMMON = run_path(
  str(Path(__file__).resolve().parents[1] / "scripts/workcell/usb_delivery_common.py")
)


@pytest.fixture
def evidence():
  metadata = {
    "scene": "usb-insert",
    "recording_contract": "usb_insert_taskspace_raw_v1",
    "observation_clock": "post_step_forward_v1",
    "controller_source_sha256": {"controller": "abc"},
  }
  outcome = {
    "object_name": "usb_plug",
    "success": True,
    "released": True,
    "grasp_verified": True,
    "active_bottom_out_confirmed": True,
    "source_files_unchanged": True,
    "controller_source_sha256_at_end": {"controller": "abc"},
    "insertion": {"success": True, "seated": True},
  }
  return metadata, outcome


def test_successful_usb_evidence_is_accepted_without_mutation(evidence):
  before = deepcopy(evidence)
  COMMON["validate_usb_outcome"](*evidence)
  assert evidence == before


@pytest.mark.parametrize(
  "gate",
  [
    "success",
    "released",
    "grasp_verified",
    "active_bottom_out_confirmed",
    "source_files_unchanged",
  ],
)
def test_success_label_cannot_bypass_original_physical_gates(evidence, gate):
  metadata, outcome = evidence
  outcome[gate] = False
  with pytest.raises(ValueError, match="gate failed"):
    COMMON["validate_usb_outcome"](metadata, outcome)


@pytest.mark.parametrize(
  "field,value",
  [
    ("scene", "poker-draw"),
    ("observation_clock", "pre_step"),
    ("recording_contract", "legacy"),
  ],
)
def test_wrong_task_and_legacy_clock_are_rejected(evidence, field, value):
  metadata, outcome = evidence
  metadata[field] = value
  with pytest.raises(ValueError, match="post-step raw contract"):
    COMMON["validate_usb_outcome"](metadata, outcome)


def test_changed_controller_identity_is_rejected(evidence):
  metadata, outcome = evidence
  outcome["controller_source_sha256_at_end"] = {"controller": "changed"}
  with pytest.raises(ValueError, match="controller changed"):
    COMMON["validate_usb_outcome"](metadata, outcome)


@pytest.mark.parametrize("gate", ["success", "seated"])
def test_missing_socket_success_is_rejected(evidence, gate):
  metadata, outcome = evidence
  del outcome["insertion"][gate]
  with pytest.raises(ValueError, match="insertion gate"):
    COMMON["validate_usb_outcome"](metadata, outcome)
