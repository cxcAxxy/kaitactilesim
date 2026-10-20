"""Public collection profile and CLI contracts without a simulation rollout."""

import importlib.util
from copy import deepcopy
import json
import sys
from pathlib import Path

import h5py
from kaihand_tactile_env.pipeline.catalog import TASKS
from kaihand_tactile_env.pipeline.collection import (
  COLLECTION_SPECS,
  recorded_runtime_contracts,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect/collect.py"
spec = importlib.util.spec_from_file_location("collection_interface_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CAMERAS = ["head", "left_wrist", "right_wrist"]


def _expected_arguments(task, output):
  common = ["--camera-hz", "30"]
  if task in {
    "usb-unplug", "whiteboard-wipe-horizontal", "bulb-unscrew",
    "whiteboard-wipe-horizontal-line",
    "bulb-screw-inplace",
  }:
    return [
      "record_task_variant.py", "--task", task,
      "--output-dir", str(output), *common, "--buffer-rows", "128",
      "--cameras", *CAMERAS, "--seed", "42",
    ]
  if task == "pick-place":
    return [
      "record_dataset.py",
      "--scene", task,
      "--output-dir", str(output),
      "--episodes", "1",
      "--workers", "1",
      "--seed", "42",
      *common,
      "--cameras", *CAMERAS,
      "--rgb-only",
      "--object", "cylinder",
      "--side", "right",
    ]
  if task == "poker-draw":
    return [
      "record_dataset.py",
      "--scene", task,
      "--output-dir", str(output),
      "--episodes", "1",
      "--workers", "1",
      "--seed", "42",
      *common,
      "--cameras", *CAMERAS,
      "--rgb-only",
      "--preset", "middle-force-precontact-v1",
      "--tactile-links", "right",
      "--acceptance-policy", "task-completion-v1",
    ]
  if task == "usb-insert":
    return [
      "record_usb_dataset.py",
      "--output-dir", str(output),
      "--episodes", "1",
      "--workers", "1",
      "--seed", "42",
      "--motion-profile", "fast",
      *common,
      "--cameras", *CAMERAS,
      "--xy-jitter-mm", "10",
      "--yaw-jitter-deg", "5",
      "--precontact-noise-mm", "0.5",
    ]
  shared = {
    "bulb-screw": [
      "refresh_light_bulb_example.py",
      "--position-seed", "42",
      *common,
      "--buffer-rows", "128",
      "--cameras", *CAMERAS,
      "--raw-only",
      "--output", str(output),
    ],
    "vase-wipe": [
      "view_vase_wipe.py",
      "--headless",
      "--run-task",
      "--output-dir", str(output),
      *common,
      "--buffer-rows", "128",
      "--cameras", *CAMERAS,
      "--raw-only",
      "--stain-seed", "42",
    ],
    "install-ram": [
      "record_install_ram_example.py",
      "--position-seed", "42",
      *common,
      "--buffer-rows", "64",
      "--cameras", *CAMERAS,
      "--raw-only",
      "--output-dir", str(output),
    ],
    "whiteboard-wipe": [
      "record_whiteboard_example.py",
      "--output-dir", str(output),
      *common,
      "--buffer-rows", "128",
      "--cameras", *CAMERAS,
      "--raw-only",
      "--ink-seed", "42",
    ],
    "sponge-grasp": [
      "record_sponge_grasp.py",
      "--output-dir", str(output),
      "--seed", "42",
      *common,
      "--buffer-rows", "128",
      "--raw-only",
      "--cameras", *CAMERAS,
    ],
  }
  return shared[task]


def test_registry_uses_canonical_tasks_and_preserves_every_command_element(tmp_path):
  assert runner.TASKS == TASKS
  assert tuple(COLLECTION_SPECS) == TASKS
  for task in TASKS:
    output = tmp_path / task
    expected = _expected_arguments(task, output)
    command = runner.command(task, output, 42)
    assert command == [
      sys.executable,
      str(ROOT / "scripts/workcell" / expected[0]),
      *expected[1:],
    ]
    assert runner.command(
      task, output, 42, profile="legacy-compatible"
    ) == command


def test_cli_lists_and_describes_profiles_without_operational_arguments(capsys):
  assert runner.main(["--list-support"]) == 0
  catalog = json.loads(capsys.readouterr().out)
  assert catalog["default_profile"] == "production-v1"
  assert [row["task"] for row in catalog["tasks"]] == list(TASKS)
  assert {row["name"]: row["resolves_to"] for row in catalog["profiles"]} == {
    "production-v1": "production-v1",
    "legacy-compatible": "production-v1",
  }

  assert runner.main([
    "--describe", "vase-wipe",
    "--profile", "legacy-compatible",
    "--fixed-stains",
  ]) == 0
  profile = json.loads(capsys.readouterr().out)
  assert profile["profile_id"] == "production-v1"
  assert profile["task"] == "vase-wipe"
  assert profile["seed_mode"] == "fixed-task-layout"
  assert profile["runtime_contract"] == {
    "authority": "episode-hdf5-root-attribute",
    "attribute": "runtime_contract_json",
    "missing_value": "unknown",
  }


def test_hardware_environment_inherits_explicit_egl_vendor(monkeypatch):
  vendor = "/configured/nvidia_egl_vendor.json"
  monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", vendor)

  environment = runner.environment("hardware")

  assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == vendor
  assert environment["MUJOCO_GL"] == "egl"
  assert environment["PYOPENGL_PLATFORM"] == "egl"
  assert environment["KAIHAND_RENDER_BACKEND"] == "hardware"


def test_manifest_records_resolved_profile_command_and_strict_legacy_resume(tmp_path):
  output = (tmp_path / "output").resolve()
  staging = (tmp_path / "staging").resolve()
  args = runner.parse_args([
    "--task", "vase-wipe",
    "--output-dir", str(output),
    "--staging-root", str(staging),
    "--profile", "legacy-compatible",
  ])
  args.staging_root = staging
  contract = runner.collection_contract(
    args, ["vase-wipe"], output, source_sha256={"source.py": "same"}
  )
  concrete = contract["production_commands"]["vase-wipe"]
  assert contract["profile"] == "production-v1"
  assert contract["resolved_profiles"]["vase-wipe"]["profile_id"] == "production-v1"
  expected_data = (
    runner.staging_episode_directory(staging, output, "vase-wipe", 0)
    / "attempt_001/data"
  )
  assert concrete["command"] == runner.command(
    "vase-wipe",
    expected_data,
    concrete["seed"],
    profile="production-v1",
  )

  legacy = {
    key: value
    for key, value in contract.items()
    if key not in runner.PROFILE_CONTRACT_FIELDS
  }
  assert runner.collection_contracts_compatible(legacy, contract)
  legacy["source_sha256"] = {"source.py": "changed"}
  assert not runner.collection_contracts_compatible(legacy, contract)
  changed_new_contract = {**contract, "production_commands": {}}
  assert not runner.collection_contracts_compatible(changed_new_contract, contract)


def test_runtime_contract_is_read_from_hdf5_and_historical_missing_is_unknown(tmp_path):
  recorded = tmp_path / "raw/recorded.h5"
  recorded.parent.mkdir()
  payload = {
    "schema": "kaihand_simulation_runtime_v1",
    "physics": {"timestep_s": 0.002},
    "tactile": {"source": "depth-derived", "force_unit": "proxy"},
  }
  with h5py.File(recorded, "w") as file:
    file.attrs["runtime_contract_json"] = json.dumps(payload)
  historical = tmp_path / "raw/historical.h5"
  with h5py.File(historical, "w"):
    pass

  result = recorded_runtime_contracts(tmp_path)
  assert result["artifacts"]["raw/recorded.h5"] == {
    "status": "recorded",
    "contract": payload,
  }
  assert result["artifacts"]["raw/historical.h5"] == {"status": "unknown"}


def test_usb_collection_profile_declares_hold_protocol_and_rejects_old_resume(tmp_path, capsys):
  from kaihand_tactile_env.tasks.usb_unplug import config

  assert runner.main(["--describe", "usb-unplug"]) == 0
  profile = json.loads(capsys.readouterr().out)
  assert profile["outcome_contract"] == config.COMPLETION_PROTOCOL
  assert profile["settings"]["completion_protocol"] == config.COMPLETION_PROTOCOL
  assert profile["settings"]["hold_s"] == config.EXTRACTION_HOLD_DWELL_S == 2.0
  output, staging = (tmp_path / "out").resolve(), (tmp_path / "staging").resolve()
  args = runner.parse_args([
    "--task", "usb-unplug", "--output-dir", str(output), "--staging-root", str(staging),
  ])
  current = runner.collection_contract(
    args, ["usb-unplug"], output, source_sha256={"source.py": "same"},
  )
  assert runner.collection_contracts_compatible(deepcopy(current), current)
  old = deepcopy(current)
  old_profile = old["resolved_profiles"]["usb-unplug"]
  old_profile["outcome_contract"] = "unplug-physical-clear-place-release-v1"
  old_profile["settings"].pop("completion_protocol")
  old_profile["settings"].pop("hold_s")
  assert not runner.collection_contracts_compatible(old, current)
  assert COLLECTION_SPECS["bulb-unscrew"].outcome_contract == "unscrew-physical-disengage-table-release-v2"
  assert "completion_protocol" not in COLLECTION_SPECS["bulb-unscrew"].settings


def _write_variant_collection_evidence(output, outcome):
  raw = output / "raw"
  raw.mkdir(parents=True)
  with h5py.File(raw / "episode.h5", "w"):
    pass
  (output / "result.json").write_text(json.dumps(outcome))
  (raw / "episode.json").write_text(json.dumps({
    "outcome": outcome, "validation": {"valid": True}, "task_audit_passed": True,
  }))


def test_usb_collection_accepts_held_clear_expert_without_placement_and_rejects_legacy(tmp_path):
  from kaihand_tactile_env.shared.task_variants import validate_variant_outcome
  from kaihand_tactile_env.tasks.usb_unplug import config

  outcome = {
    "task": "usb-unplug", "success": True,
    "completion_protocol": config.COMPLETION_PROTOCOL,
    "extraction_hold_now": True, "extraction_hold_verified": True,
    "extraction_hold_duration_s": 2.0,
    "placed_on_table": False, "released": False, "hand_retreated": False,
    "physics_recording_audit": {"passed": True},
    "final_state": {
      "completion_protocol": config.COMPLETION_PROTOCOL,
      "extraction_hold_now": True, "extraction_hold_verified": True,
      "extraction_hold_duration_s": 2.0, "insertion_depth_m": -0.003,
      "right_hand_contact_count": 1, "table_contact_count": 0, "socket_contact_count": 0,
    },
  }
  current = tmp_path / "current"
  _write_variant_collection_evidence(current, outcome)
  assert runner.outcome("usb-unplug", current) is True
  old = {
    "task": "usb-unplug", "success": True, "initially_seated_verified": True,
    "grasp_verified": True, "fully_extracted": True,
    "placed_on_table": True, "released": True, "hand_retreated": True,
    "physics_recording_audit": {"passed": True},
  }
  # Legacy interpretation remains available for archived artifacts, but the
  # current production collection may not append it under hold-only metadata.
  validate_variant_outcome("usb-unplug", old)
  historical = tmp_path / "old"
  _write_variant_collection_evidence(historical, old)
  assert runner.outcome("usb-unplug", historical) is False
  bulb = {
    "task": "bulb-unscrew", "success": True, "initially_installed_verified": True,
    "grasp_verified": True, "thread_disengaged": True, "fully_extracted": True,
    "placed_on_table": True, "released": True, "hand_retreated": True,
    "physics_recording_audit": {"passed": True},
  }
  unchanged = tmp_path / "bulb"
  _write_variant_collection_evidence(unchanged, bulb)
  assert runner.outcome("bulb-unscrew", unchanged) is True
