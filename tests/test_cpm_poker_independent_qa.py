"""Pure audit fixtures: no simulator, OpenPI, checkpoint loader or model."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/check_cpm_poker_evaluation.py"
SPEC = importlib.util.spec_from_file_location("cpm_independent_qa", SCRIPT)
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def randomized(seed=0):
    generator = np.random.default_rng(seed)
    xy = generator.uniform(-.004, .004, size=2)
    yaw = generator.uniform(-math.radians(.5), math.radians(.5))
    pose = [.58 + xy[0], -.16 + xy[1], .84175, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    return {"initial_card_pose": pose, "initial_card_randomization": {
        "schema_version": "poker-initial-card-randomization-v1", "mode": "seeded_uniform",
        "distribution": "independent_uniform", "seed": seed, "xy_jitter_m": .004,
        "yaw_jitter_rad": math.radians(.5), "perturbation_scope": "reset_only_card_xy_yaw",
        "height_unchanged": True, "tilt_unchanged": True, "observation_noise": None, "action_noise": None,
        "sampled_offset_xy_m": xy.tolist(), "sampled_yaw_offset_rad": yaw,
        "nominal_pose_wxyz": [.58, -.16, .84175, 1, 0, 0, 0], "sampled_pose_wxyz": pose,
        "table_support": {"valid": True, "minimum_corner_margin_xy_m": .05, "bottom_gap_m": .0002}}}


def request(index, seed=0):
    return {"request": index + 1, "request_index": index, "episode_seed": seed, "action_start_index": 0,
        "sim_time": index / 6, "observation_sim_time": index / 6, "tactile_sent_to_model": True,
        "tactile_map_abs_max_n": 0, "tactile_map_raw": np.zeros((5, 3, 7, 5)).tolist(),
        "world_T_head": np.eye(4).tolist(), "world_T_wrist": np.eye(4).tolist(), "state": [0] * 27,
        "predicted_actions": np.zeros((30, 27)).tolist(), "diagnostics": {
            "variant": "pi05+touchguidance", "noise_seed": int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0]),
            "guidance": {"num_steps": 10, "guidance_steps": 3, "scale": 1.0}, "base_comparison": True,
            "steps": [{"step": i, "guided": i >= 7, "gradient_rms": .001,
                "velocity_correction_rms": .0001, "latent_guidance_shift_rms": .00001,
                "padded_guidance_max_abs": 0} for i in range(10)]}}


def trial(directory, monkeypatch):
    manifest = {"checkpoint_path": "/frozen/cpm/best.pt", "checkpoint_sha256": "a" * 64, "deployment_id": "deployment-cpm"}
    identity = {"task": qa.TASK, "model_family": qa.FAMILY, "seed": 0, "prediction_horizon": 30,
                "execute_steps": 5, "action_start_index": 0, **manifest}
    detail = {**identity, "deployment_manifest_sha256": "b" * 64, "valid_trial": True,
        "trial_validity": "valid", "physics_valid": True, "task_success": False, "native_success": False,
        "evaluation": {"success": False}, "status": "task_not_completed", "sim_seconds": 90.002,
        "max_sim_seconds": 90, "full_duration_evaluation": True, "stop_reason": "time_limit",
        "guard_mode": "observe", "action_interpolation": "zoh", "controller": "poker-tactile-direct-joints-v1",
        "control_hz": 30, "diagnostic_only": True, "formal_metrics_valid": False,
        "tactile_sent_to_model": True, "tactile_conditioning_enabled": True,
        "simulation_paused_during_inference": True,
        "guidance": {"enabled": True, "steps": 3, "scale": 1.0, "variant": "pi05+touchguidance"},
        "server_metadata": {**{key: value for key, value in identity.items() if key != "seed"},
            "normalization_verified": True, "training_dependencies_verified": True,
            "deployment_manifest_sha256": "b" * 64}, **randomized()}
    write(directory / "summary.json", detail)
    np.savez_compressed(directory / "first_request.npz", predicted_actions=np.zeros((30, 27)),
                        tactile_map_raw=np.zeros((5, 3, 7, 5)),
                        head_image=np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3),
                        right_wrist_image=np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3))
    (directory / "evaluation_rollout_trace.npz").write_bytes(b"trace")
    monkeypatch.setattr(qa, "inspect_stages", lambda *args: {"stages": {key: False for key in qa.DEFINITIONS}, "stage_thresholds": {}})
    monkeypatch.setattr(qa, "inspect_requests", lambda *args: {"request_count": 540})
    monkeypatch.setattr(qa, "inspect_media", lambda *args, **kwargs: {})
    monkeypatch.setattr(qa, "inspect_raw", lambda *args: {})
    return manifest, detail


def test_randomized_card_pose_is_reproducible_for_all_requested_seeds():
    poses = [qa.inspect_randomization(randomized(seed), seed) for seed in range(10)]
    assert len({tuple(pose) for pose in poses}) == 10


def test_modified_reset_cannot_claim_to_be_the_seeded_pose():
    detail = randomized()
    detail["initial_card_randomization"]["sampled_pose_wxyz"][0] += .001
    with pytest.raises(ValueError, match="Initial pose"):
        qa.inspect_randomization(detail, 0)


def test_valid_native_task_failure_is_retained(tmp_path, monkeypatch):
    manifest, _ = trial(tmp_path, monkeypatch)
    value = qa.inspect_trial(tmp_path, manifest, "b" * 64, 0)
    assert value["native_success"] is False
    assert value["stages"] == {key: False for key in qa.DEFINITIONS}


def test_full_duration_native_success_keeps_its_original_success_stop_reason(tmp_path, monkeypatch):
    manifest, detail = trial(tmp_path, monkeypatch)
    detail.update(task_success=True, native_success=True, evaluation={"success": True},
                  status="success", stop_reason="success")
    write(tmp_path / "summary.json", detail)
    assert qa.inspect_trial(tmp_path, manifest, "b" * 64, 0)["native_success"] is True


@pytest.mark.parametrize("update,match", [
    ({"physics_valid": False}, "Infrastructure-invalid"),
    ({"raw_record_error": "missing"}, "Infrastructure-invalid"),
    ({"task_success": True}, "relabelled"),
    ({"native_success": True}, "relabelled"),
    ({"deployment_manifest_sha256": "c" * 64}, "identity mismatch"),
    ({"sim_seconds": 30}, "full 90 seconds"),
    ({"action_interpolation": "linear"}, "Physical controller"),
    ({"tactile_sent_to_model": False}, "genuinely enabled"),
    ({"tactile_conditioning_enabled": False}, "genuinely enabled"),
    ({"formal_metrics_valid": True}, "diagnostic status"),
])
def test_invalid_native_evidence_is_rejected_not_zero_filled(tmp_path, monkeypatch, update, match):
    manifest, detail = trial(tmp_path, monkeypatch)
    detail.update(update)
    write(tmp_path / "summary.json", detail)
    with pytest.raises(ValueError, match=match):
        qa.inspect_trial(tmp_path, manifest, "b" * 64, 0)


def test_real_tactile_and_guidance_requests_cover_full_closed_loop(tmp_path):
    path = tmp_path / "requests.jsonl"
    with path.open("w") as stream:
        for index in range(540):
            row = request(index)
            if index == 200:
                row["tactile_map_raw"][0][0][0][0] = row["tactile_map_abs_max_n"] = .02
            stream.write(json.dumps(row) + "\n")
    result = qa.inspect_requests(path, {"stats": {"requests": 540}}, 0)
    assert result == {"request_count": 540, "tactile_nonzero_request_count": 1, "positive_guidance_gradient_steps": 1620}


@pytest.mark.parametrize("change,match", [
    (lambda row: row.update(tactile_sent_to_model=False), "disabled"),
    (lambda row: row.update(tactile_map_raw=[]), "real tactile"),
    (lambda row: row.update(tactile_map_abs_max_n=.5), "magnitude"),
    (lambda row: row.update(observation_sim_time=.01), "advanced"),
    (lambda row: row.update(predicted_actions=None), "H30 action"),
    (lambda row: row["diagnostics"].update(noise_seed=42), "genuine guided"),
    (lambda row: row["diagnostics"]["steps"][9].update(guided=False), "did not execute"),
    (lambda row: row["diagnostics"]["steps"][9].update(gradient_rms=float("nan")), "finite CPM"),
    (lambda row: row["diagnostics"]["steps"][9].update(padded_guidance_max_abs=.001), "padded"),
])
def test_fake_or_unsynchronized_tactile_requests_are_rejected(tmp_path, change, match):
    value = request(0)
    change(value)
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match=match):
        qa.inspect_requests(path, {"stats": {"requests": 540}}, 0)


def test_batch_native_and_stage_denominators_are_own_ten_seeds(tmp_path, monkeypatch):
    manifest = {"deployment_id": "new-cpm"}
    rows = [{"seed": seed, "valid": True, "returncode": 0, "timed_out": False, "identity_matches": True,
             "summary": f"seed_{seed:04d}/summary.json", "task_success": seed == 2} for seed in range(10)]
    for row in rows:
        write(tmp_path / row["summary"], {})
    batch = {"schema": "poker_tactile_batch_v1", "task": qa.TASK, "model_family": qa.FAMILY,
             "complete": True, "valid_trials": 10, "invalid_trials": 0, "deployment_id": "new-cpm",
             "deployment_manifest_sha256": "manifest", "success_rate": .1, "trials": rows}
    write(tmp_path / "batch_summary.json", batch)
    monkeypatch.setattr(qa, "inspect_trial", lambda directory, manifest, manifest_sha, seed, **kwargs: {
        "seed": seed, "native_success": seed == 2, "initial_pose": [seed], "stage_thresholds": {},
        "stages": {"stage1": seed < 5, "stage2": seed < 2, "stage3": seed == 2}})
    result = qa.inspect_evaluation(tmp_path, manifest, "manifest")
    assert result["native_valid_trials"] == result["stage_valid_trials"] == 10
    assert result["native_success_count"] == 1
    assert result["stage_counts"] == {"stage1": 5, "stage2": 2, "stage3": 1}
    changed = copy.deepcopy(batch)
    changed["trials"].pop()
    write(tmp_path / "batch_summary.json", changed)
    with pytest.raises(ValueError, match="omitted/duplicated"):
        qa.inspect_evaluation(tmp_path, manifest, "manifest")


def test_invalid_trial_cannot_be_hidden_by_success_denominator(tmp_path):
    write(tmp_path / "batch_summary.json", {"schema": "poker_tactile_batch_v1", "task": qa.TASK,
        "model_family": qa.FAMILY, "complete": True, "valid_trials": 9, "invalid_trials": 1})
    with pytest.raises(ValueError, match="incomplete or contains invalid"):
        qa.inspect_evaluation(tmp_path, {"deployment_id": "new-cpm"}, "manifest")


def raw_fixture(directory):
    import h5py
    identity = {"seed": 0, "execute_steps": 5, "checkpoint_path": "/frozen/best.pt",
                "checkpoint_sha256": "a" * 64, "deployment_id": "cpm"}
    result = {"success": False, "status": "task_not_completed"}
    detail = {"task_success": False, "status": "task_not_completed", "sim_seconds": 90.002,
              "raw": {"validated": True, "episode": "raw/episode.h5", "result": "raw/episode.result.json",
                      "right_hand_force_curves": "force.png", "state_samples": 9002}}
    write(directory / "raw/episode.result.json", result)
    (directory / "force.png").write_bytes(b"fixture-force-image")
    times = np.linspace(0, 90.002, 9002)
    with h5py.File(directory / "raw/episode.h5", "w") as file:
        file.attrs["metadata_json"] = json.dumps({**identity, "model_family": "pi0.5"})
        file.attrs["outcome_json"] = json.dumps(result)
        file.create_dataset("state/timestamp", data=times)
        for key in ("state/qpos", "state/qvel", "commands/arm_joint_target", "commands/hand_joint_target",
                    "objects/card/pose_wxyz", "tactile_contact_force/normal_force_n", "tactile_contact_force/tangent_force_n"):
            file.create_dataset(key, shape=(9002, 3), dtype=np.float64, chunks=True, compression="gzip")
        cameras = file.create_group("cameras")
        cameras.attrs["configured_names_json"] = json.dumps(["head", "right_wrist"])
        stamps = np.linspace(0, 90.002, 2702)
        for name in ("head", "right_wrist"):
            group = cameras.create_group(name)
            group.create_dataset("timestamp", data=stamps)
            group.create_dataset("state_index", data=np.searchsorted(times, stamps, side="right") - 1)
            rgb = group.create_dataset("rgb", shape=(2702, 240, 320, 3), dtype=np.uint8,
                                       chunks=(1, 240, 320, 3), compression="gzip")
            rgb[0] = rgb[-1] = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)
    return detail, identity


def test_raw_outcome_and_both_cameras_are_independently_bound_despite_legacy_label(tmp_path):
    detail, identity = raw_fixture(tmp_path)
    result = qa.inspect_raw(tmp_path, detail, identity)
    assert result["raw_state_samples"] == 9002
    assert result["raw_camera_samples"] == {"head": 2702, "right_wrist": 2702}
    assert result["raw_metadata_model_family"] == "pi0.5"


@pytest.mark.parametrize("change,match", [
    ("identity", "Raw identity"), ("outcome", "HDF5 outcome"),
    ("nonfinite", "Nonfinite raw"), ("blank_camera", "blank"),
    ("wrong_camera_clock", "Incomplete or unsynchronized"),
])
def test_raw_corruption_is_not_admitted_as_a_task_failure(tmp_path, change, match):
    import h5py
    detail, identity = raw_fixture(tmp_path)
    with h5py.File(tmp_path / "raw/episode.h5", "r+") as file:
        if change == "identity":
            file.attrs["metadata_json"] = json.dumps({**identity, "checkpoint_sha256": "b" * 64})
        elif change == "outcome":
            file.attrs["outcome_json"] = json.dumps({"success": True, "status": "success"})
        elif change == "nonfinite":
            file["state/qpos"][3, 0] = float("nan")
        elif change == "blank_camera":
            file["cameras/right_wrist/rgb"][-1] = 0
        else:
            file["cameras/head/timestamp"][1] = -1
    with pytest.raises(ValueError, match=match):
        qa.inspect_raw(tmp_path, detail, identity)


def test_decoded_video_and_frame_manifest_must_match(tmp_path, monkeypatch):
    identity = {"seed": 0, "model_family": qa.FAMILY, "checkpoint_sha256": "a" * 64}
    review = {**identity, "model_input_cameras_displayed": ["head", "right_wrist"], "fps": 10,
              "physics_modified": False, "head_camera_modified": False, "frame_count": 901, "output_size": [1920, 1080],
              "deployment_manifest_sha256": "b" * 64, "contact_model": {"physics_timestep_used_s": .002}}
    detail = {"video": copy.deepcopy(review), "deployment_manifest_sha256": "b" * 64}
    write(tmp_path / "review.json", review)
    (tmp_path / "review.mp4").write_bytes(b"video-fixture")
    with (tmp_path / "frames.jsonl").open("w") as stream:
        for index in range(901):
            stream.write(json.dumps({"frame": index, "simulation_time_s": index / 10,
                "camera_pose_time_s": index / 10, "tactile_time_s": index / 10}) + "\n")
    probe = {"width": 1920, "height": 1080, "avg_frame_rate": "10/1", "nb_read_frames": "901", "duration": "90.1"}
    monkeypatch.setattr(qa.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"streams": [probe]})))
    assert qa.inspect_media(tmp_path, detail, identity)["model_input_frames"] == 901
    probe["nb_read_frames"] = "900"
    with pytest.raises(ValueError, match="Decoded video frame count"):
        qa.inspect_media(tmp_path, detail, identity)


def cached_media_fixture(directory, monkeypatch, *, change=None):
    identity = {"task": qa.TASK, "seed": 0, "model_family": qa.FAMILY, "checkpoint_sha256": "a" * 64,
                "deployment_id": "native-guidance-deployment"}
    review = {**{key: value for key, value in identity.items() if key != "deployment_id"},
        "model_input_cameras_displayed": ["head", "right_wrist"], "fps": 10,
        "physics_modified": False, "head_camera_modified": False, "frame_count": 901, "output_size": [1920, 1080],
        "deployment_manifest_sha256": "b" * 64, "contact_model": {"physics_timestep_used_s": .002}}
    detail = {"video": copy.deepcopy(review), "deployment_manifest_sha256": "b" * 64}
    write(directory / "review.json", review)
    (directory / "review.mp4").write_bytes(b"video-fixture")
    with (directory / "frames.jsonl").open("w") as stream:
        for index in range(901):
            stamp = index / 10
            cached = stamp - .002 if index else stamp
            row = {"frame": index, "simulation_time_s": stamp, "camera_pose_time_s": cached, "tactile_time_s": cached}
            if change is not None and index == 10:
                change(row)
            stream.write(json.dumps(row) + "\n")
    probe = {"width": 1920, "height": 1080, "avg_frame_rate": "10/1", "nb_read_frames": "901", "duration": "90.1"}
    monkeypatch.setattr(qa.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout=json.dumps({"streams": [probe]})))
    return detail, identity, review


def test_native_video_missing_deployment_id_is_sha_bound_and_one_solver_step_lag_is_explicit(tmp_path, monkeypatch):
    detail, identity, _ = cached_media_fixture(tmp_path, monkeypatch)
    result = qa.inspect_media(tmp_path, detail, identity)
    assert result["video_cached_epoch"]["zero_lag_frames"] == 1
    assert result["video_cached_epoch"]["one_physics_step_lag_frames"] == 900
    assert result["video_cached_epoch"]["maximum_simulation_minus_cached_s"] == pytest.approx(.002)


@pytest.mark.parametrize("change,match", [
    (lambda row: row.update(camera_pose_time_s=.996, tactile_time_s=.996), "neither current nor exactly one"),
    (lambda row: row.update(tactile_time_s=.997), "camera and tactile epochs differ"),
    (lambda row: row.update(camera_pose_time_s=.898, tactile_time_s=.898), "Broken cached"),
    (lambda row: row.update(camera_pose_time_s=float("nan")), "Broken cached"),
])
def test_video_epoch_contract_rejects_multistep_lag_cross_epoch_and_nonfinite(tmp_path, monkeypatch, change, match):
    detail, identity, _ = cached_media_fixture(tmp_path, monkeypatch, change=change)
    with pytest.raises(ValueError, match=match):
        qa.inspect_media(tmp_path, detail, identity)


@pytest.mark.parametrize("change", ["manifest_sha", "summary_video", "present_wrong_deployment_id"])
def test_video_missing_id_never_weakens_native_metadata_binding(tmp_path, monkeypatch, change):
    detail, identity, review = cached_media_fixture(tmp_path, monkeypatch)
    if change == "manifest_sha":
        review["deployment_manifest_sha256"] = "c" * 64
    elif change == "summary_video":
        detail["video"]["seed"] = 3
    else:
        review["deployment_id"] = "another-deployment"
    write(tmp_path / "review.json", review)
    with pytest.raises(ValueError, match="Video identity mismatch|complete native summary"):
        qa.inspect_media(tmp_path, detail, identity)


def stage_fixture(directory):
    identity = {"task": qa.TASK, "seed": 0, "prediction_horizon": 30, "execute_steps": 5,
                "checkpoint_path": "/frozen/best.pt", "checkpoint_sha256": "a" * 64, "deployment_id": "cpm"}
    flags = {"stage1": True, "stage2": False, "stage3": False}
    stage = {"version": "staged_v1", "task": qa.TASK, "stage_definitions": qa.DEFINITIONS,
             "stages": flags, "stage_flags": flags, "complete_success": False, "legacy_task_success": False,
             "native_task_success_preserved": True, "expert_trajectory_used": False,
             "simulation_state_mutated": False, "full_duration_evaluation": True,
             "thresholds": {"poker_terminal_hold_s": 3.0}, "state_log_samples": 2701}
    write(directory / "staged_evaluation.json", stage)
    with (directory / "staged_states.jsonl").open("w") as stream:
        for tick in range(2701):
            stream.write(json.dumps({**identity, "time_s": tick / 30, "control_tick": tick,
                "thresholds": stage["thresholds"], "stage_definitions": qa.DEFINITIONS,
                "stages": flags, "stage_flags": flags}) + "\n")
    return {"task_success": False, "staged_evaluation": stage}, identity


def test_stage_success_is_separate_from_native_and_bound_to_full_state_log(tmp_path):
    detail, identity = stage_fixture(tmp_path)
    result = qa.inspect_stages(tmp_path, detail, identity)
    assert result["stages"] == {"stage1": True, "stage2": False, "stage3": False}
    assert result["state_samples"] == 2701
    assert detail["task_success"] is False


@pytest.mark.parametrize("change,match", [
    (lambda value: value.update(simulation_state_mutated=True), "changed the native"),
    (lambda value: value.update(legacy_task_success=True), "overwritten"),
    (lambda value: value["thresholds"].update(poker_terminal_hold_s=.1), "3-second hold"),
    (lambda value: value.update(state_log_samples=12), "full native rollout"),
])
def test_stage_evidence_cannot_mutate_native_or_relax_3second_reference(tmp_path, change, match):
    detail, identity = stage_fixture(tmp_path)
    change(detail["staged_evaluation"])
    write(tmp_path / "staged_evaluation.json", detail["staged_evaluation"])
    with pytest.raises(ValueError, match=match):
        qa.inspect_stages(tmp_path, detail, identity)


def runtime_fixture(directory):
    snapshots, inventory = {}, {}
    for label, files in {"simulator": ["src/sim.py", "scripts/evaluate/evaluate_cpm_poker_checkpoint.py",
                                         "scripts/evaluate/isolation.py"],
                         "openpi-main": ["src/openpi/policy.py", "packages/openpi-client/src/client.py"]}.items():
        source, frozen = directory / "original" / label, directory / "runtime" / label
        hashes = {}
        for relative in files:
            for root in (source, frozen):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# independently copied runtime\n")
            (frozen / relative).chmod(0o444)
            hashes[relative] = qa.sha256(frozen / relative)
            inventory[str(frozen / relative)] = hashes[relative]
        snapshots[label] = {"source": str(source), "frozen": str(frozen), "files": hashes,
                            "source_rehashed_after_copy": True, "frozen_rehashed": True}
    simulator, openpi = directory / "runtime/simulator", directory / "runtime/openpi-main"
    snapshot = {"schema": "kaihand-cpm-isolated-runtime-v1", "snapshots": snapshots,
                "controller_source": str(directory / "original/simulator/scripts/evaluate/isolation.py"),
                "controller_sha256": qa.sha256(simulator / "scripts/evaluate/isolation.py")}
    write(directory / "runtime_snapshot.json", snapshot)
    plan = {"isolated_runtime": True, "runtime_snapshot_sha256": qa.sha256(directory / "runtime_snapshot.json"),
            "simulator_root": str(simulator), "model_project_root": str(openpi), "prior_interrupted_trials_counted": 0,
            "source_script": {"path": str(simulator / "scripts/evaluate/evaluate_cpm_poker_checkpoint.py"),
                              "sha256": qa.sha256(simulator / "scripts/evaluate/evaluate_cpm_poker_checkpoint.py")}}
    manifest = {"files": inventory, "runtime": {"project_root": str(openpi)}}
    status = {"trials": {"0": {"command": ["python", "-B", str(simulator / "scripts/workcell/run_poker_tactile_policy.py")]}}}
    return plan, manifest, status


def test_runtime_is_a_readonly_independent_copy_that_survives_original_edits(tmp_path):
    plan, manifest, status = runtime_fixture(tmp_path)
    (tmp_path / "original/openpi-main/src/openpi/policy.py").write_text("# concurrent training edits\n")
    result = qa.inspect_runtime_snapshot(tmp_path, plan, manifest, status)
    assert result["isolated_runtime"] is True
    assert result["runtime_snapshots"]["simulator"]["files_hashed"] == 3
    assert result["runtime_snapshots"]["openpi-main"]["files_hashed"] == 2


@pytest.mark.parametrize("change,match", [
    ("writable", "writable, linked or modified"),
    ("modified", "writable, linked or modified"),
    ("hardlink", "writable, linked or modified"),
    ("symlink", "contains a symlink"),
    ("new_file", "membership changed"),
    ("original_root", "original mutable checkout"),
    ("original_runner", "original mutable simulator"),
    ("outside_python", "outside the isolated"),
    ("count_old_trials", "Interrupted non-isolated"),
    ("snapshot_unbound", "does not bind"),
])
def test_runtime_aliases_writes_drift_and_old_trial_reuse_are_rejected(tmp_path, change, match):
    plan, manifest, status = runtime_fixture(tmp_path)
    path = tmp_path / "runtime/openpi-main/src/openpi/policy.py"
    if change == "writable":
        path.chmod(0o644)
    elif change == "modified":
        path.chmod(0o644)
        path.write_text("# changed after freeze\n")
        path.chmod(0o444)
    elif change in {"hardlink", "symlink"}:
        path.unlink()
        original = tmp_path / "original/openpi-main/src/openpi/policy.py"
        if change == "hardlink":
            os.link(original, path)
        else:
            path.symlink_to(original)
    elif change == "new_file":
        (tmp_path / "runtime/openpi-main/extra.py").write_text("# new unbound source\n")
    elif change == "original_root":
        manifest["runtime"]["project_root"] = str(tmp_path / "original/openpi-main")
    elif change == "original_runner":
        status["trials"]["0"]["command"][2] = str(tmp_path / "original/simulator/scripts/workcell/run_poker_tactile_policy.py")
    elif change == "outside_python":
        manifest["files"][str(tmp_path / "original/openpi-main/src/openpi/policy.py")] = "a" * 64
    elif change == "count_old_trials":
        plan["prior_interrupted_trials_counted"] = 3
    else:
        plan["runtime_snapshot_sha256"] = "b" * 64
    with pytest.raises(ValueError, match=match):
        qa.inspect_runtime_snapshot(tmp_path, plan, manifest, status)
