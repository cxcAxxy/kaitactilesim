"""Audit fixtures only; no checkpoint, model, simulator or server is loaded."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/check_bulb_inplace_evaluation.py"
SPEC = importlib.util.spec_from_file_location("independent_bulb_inplace_qa", SCRIPT)
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(directory):
    group = {"checkpoint": "/checkpoint/15000", "step": 15000}
    deployment = {"checkpoint_sha256": "a" * 64, "deployment_id": "deployment-one"}
    identity = {"task": qa.TASK, "seed": 0, "prediction_horizon": 50, "execute_steps": 50,
                "checkpoint_path": group["checkpoint"], **deployment}
    flags = {key: False for key in qa.DEFINITIONS}
    stage = {"version": "staged_v1", "task": qa.TASK, "stage_definitions": qa.DEFINITIONS,
             "stages": flags, "stage_flags": flags, "thresholds": {"dwell_s": .2},
             "full_duration_evaluation": True, "complete_success": False, "native_task_success_preserved": True,
             "expert_trajectory_used": False, "simulation_state_mutated": False}
    detail = {**identity, "staged_evaluation": stage, "physics_valid": True, "sim_seconds": 90.002,
              "task_success": False, "status": "task_not_completed", "action_interpolation": "linear", "controller": qa.CONTROLLER,
              "full_duration_evaluation": True, "diagnostic_only": True, "formal_metrics_valid": False,
              "expert_used": False, "evaluation": {"state": {"clockwise_turns": .0204, "rotation_from_initial_turns": .02},
              "threaded_initialization": {"position_randomization_applied": False, "physically_validated": True,
                  "initially_threaded_verified": True, "initially_seated": False, "initially_lit": False,
                  "initial_clockwise_turns": .0004, "bulb_pose_wxyz": [0, 0, 0, 1, 0, 0, 0]}}}
    write(directory / "summary.json", detail)
    write(directory / "staged_evaluation.json", stage)
    write(directory / "review.json", {"model_input_cameras_displayed": ["head", "right_wrist"]})
    (directory / "review.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (directory / "frames.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "evaluation_rollout_trace.npz").write_bytes(b"native-trace-fixture")
    with (directory / "staged_states.jsonl").open("w", encoding="utf-8") as stream:
        for tick in range(2701):
            stream.write(json.dumps({**identity, "control_tick": tick, "time_s": tick / 30,
                "stage_definitions": qa.DEFINITIONS, "thresholds": stage["thresholds"],
                "stages": flags, "stage_flags": flags}) + "\n")
    return group, deployment, detail


def test_valid_task_failure_is_admitted_and_net_angle_is_actual_initial_relative(tmp_path):
    group, deployment, _ = fixture(tmp_path)
    result = qa.inspect_trial(tmp_path, group, deployment, 0)
    assert result["native_success"] is False
    assert result["state_samples"] == 2701
    assert result["final_net_angle_deg"] == pytest.approx(7.2)
    assert result["net_angle_reference_deg"] == pytest.approx(99.856)


@pytest.mark.parametrize("change,match", [
    (lambda value: value.update(physics_valid=False), "Infrastructure-invalid"),
    (lambda value: value.update(sim_seconds=89.94), "full 90-second"),
    (lambda value: value.update(action_interpolation="hold"), "Wrong native identity"),
    (lambda value: value["evaluation"]["state"].update(rotation_from_initial_turns=.0204), "relative to actual initial"),
    (lambda value: value["evaluation"]["threaded_initialization"].update(position_randomization_applied=True), "Fixed, verified"),
    (lambda value: value["staged_evaluation"].update(simulation_state_mutated=True), "four-stage report mismatch"),
])
def test_missing_or_wrong_native_evidence_is_never_zero_filled(tmp_path, change, match):
    group, deployment, detail = fixture(tmp_path)
    changed = copy.deepcopy(detail)
    change(changed)
    write(tmp_path / "summary.json", changed)
    with pytest.raises(ValueError, match=match):
        qa.inspect_trial(tmp_path, group, deployment, 0)
