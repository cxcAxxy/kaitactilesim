#!/usr/bin/env python3
"""Read-only, fail-closed audit of a frozen CPM poker checkpoint's ten trials.

No simulator, runner, exporter, OpenPI or checkpoint-loader code is imported.
Task failures remain in both denominators; infrastructure-invalid trajectories
cannot silently become failures. Only an explicitly new receipt is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

TASK = "poker-draw"
FAMILY = "pi05+cpm"
DEFINITIONS = {"stage1": "dragged_on_table", "stage2": "stable_grasp", "stage3": "terminal_hold_over_3s"}
BASELINE_CATALOG_SHA = "362d072f7401319bb59bb66a736ac8256b42e17d1d65da83c1c146e3b98a92af"
BASELINE_MANIFEST_SHA = "de7ea18423235b4247aa03de1ac21db0770ba3303603ae6f1dc9bcc65d899ec4"
EXPECTED_CPM_SHA = "e5f859a5755254e55e5fec660e8d11e8b5f27b7400cc4243ad48a133951186f3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def same_number(left, right, message):
    require(finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8), message)


def array(value, shape, message):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in "iuf" and np.isfinite(result).all(), message)
    return result


def inspect_manifest(path):
    manifest = load(path)
    expected = {"schema": "poker_tactile_deployment_v1", "task": TASK, "model_family": FAMILY,
                "prediction_horizon": 30, "execute_steps": 5, "model_action_dim": 32, "action_dim": 27,
                "action_start_index": 0, "control_hz": 30, "required_action_interpolation": "zoh",
                "controller_version": "poker-tactile-direct-joints-v1", "codec_version": "touchguide-native-v1",
                "diagnostic_only": True, "formal_metrics_valid": False}
    require(all(manifest.get(key) == value for key, value in expected.items()), "Wrong frozen CPM control contract")
    canonical = {key: value for key, value in manifest.items() if key != "deployment_id"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(manifest.get("deployment_id") == digest, "Deployment canonical identity changed")
    guidance = manifest.get("guidance", {})
    require(guidance == {"num_steps": 10, "guidance_steps": 3, "scale": 1.0,
                         "variant": "pi05+touchguidance"}, "CPM guidance is disabled or changed")
    observation = manifest.get("observation_contract", {})
    require(all(observation.get(key) == value for key, value in {
        "cameras": ["head", "right_wrist"], "image_shape_hwc": [240, 320, 3],
        "tactile_sent_to_model": True, "tactile_shape": [5, 3, 7, 5], "tactile_unit": "N",
        "tactile_target_geometry": "card_core_geom"}.items()), "Wrong camera/tactile observation contract")
    source, inventory = manifest["source"], manifest["files"]
    require(Path(source["cpm_checkpoint"]).name == "best.pt" and source["cpm_checkpoint"] == manifest["checkpoint_path"],
            "CPM checkpoint must be the frozen best.pt, not a relabelled base policy")
    require(inventory.get(source["cpm_checkpoint"]) == manifest["checkpoint_sha256"], "CPM checkpoint SHA is not bound")
    require(any(Path(name).is_relative_to(Path(source["base_checkpoint"])) for name in inventory),
            "No frozen base weights")
    require(any("touchguide" in name and name.endswith(".py") for name in inventory), "No frozen TouchGuide implementation")
    for name, expected_sha in inventory.items():
        require(sha256(name) == expected_sha, f"Frozen source/artifact changed: {name}")
    runtime = manifest["runtime"]
    require(sha256(Path(runtime["python"]).resolve(strict=True)) == runtime["python_binary_sha256"],
            "Model Python executable changed")
    return manifest


def inspect_job(job, manifest_path, evaluation, manifest):
    plan, status, snapshot = load(job / "plan.json"), load(job / "status.json"), load(job / "checkpoint_snapshot.json")
    require(plan.get("schema") == "kaihand-poker-tactile-experiment-plan-v1" and plan.get("planned_trials") == 10
            and plan.get("seeds") == list(range(10)) and plan.get("automatic_retries") == 0,
            "Wrong requested ten-trial plan or retry selection")
    require(plan.get("initial_state_randomization") is True
            and plan.get("initial_randomization") == {"xy_jitter_mm": 4.0, "yaw_jitter_deg": .5, "reset_only": True},
            "Plan hides the randomized reset scope")
    groups = plan.get("groups", [])
    require(len(groups) == 1 and groups[0].get("label") == "cpm_best"
            and groups[0].get("deployment_manifest") == str(manifest_path.resolve())
            and groups[0].get("evaluation_path") == str(evaluation.resolve()), "Unexpected/extra policy group or attempt")
    require(manifest.get("checkpoint_sha256") == plan.get("expected_checkpoint_sha256") == EXPECTED_CPM_SHA
            and snapshot.get("sha256") == EXPECTED_CPM_SHA
            and snapshot.get("frozen") == manifest["checkpoint_path"]
            and snapshot.get("source") == plan.get("source_checkpoint_path")
            and snapshot.get("source_rehashed_after_copy") is True and snapshot.get("frozen_rehashed") is True
            and snapshot.get("size") == Path(manifest["checkpoint_path"]).stat().st_size,
            "Current approved CPM weight was not independently snapshotted and SHA-bound")
    require(status.get("state") == "complete" and status.get("groups", {}).get("cpm_best", {}).get("accepted_attempt") == "attempt_001",
            "Supervisor did not complete exactly the original attempt")
    require(set(status.get("trials", {})) == {str(seed) for seed in range(10)}
            and all(value.get("state") == "complete" and value.get("valid") is True and value.get("returncode") == 0
                    for value in status["trials"].values()), "Supervisor hid failed or missing seeds")
    return {"checkpoint_snapshot_sha256": sha256(job / "checkpoint_snapshot.json"),
            "plan_sha256": sha256(job / "plan.json"), "automatic_retries": 0, "accepted_attempt": "attempt_001"}


def inspect_runtime_snapshot(job, plan, manifest, status=None):
    """Verify independent code bytes without requiring original checkouts to stop changing."""
    path = job / "runtime_snapshot.json"
    require(plan.get("isolated_runtime") is True and plan.get("runtime_snapshot_sha256") == sha256(path),
            "Plan does not bind the independent runtime snapshot")
    document = load(path)
    require(document.get("schema") == "kaihand-cpm-isolated-runtime-v1"
            and set(document.get("snapshots", {})) == {"simulator", "openpi-main"},
            "Runtime snapshot does not contain the simulator and OpenPI")
    roots, totals = {}, {}
    for label, record in document["snapshots"].items():
        frozen = job / "runtime" / label
        source = Path(record["source"])
        require(record.get("frozen") == str(frozen.resolve()) and frozen.absolute() == frozen.resolve()
                and source.resolve() != frozen.resolve() and not frozen.is_symlink()
                and record.get("source_rehashed_after_copy") is True and record.get("frozen_rehashed") is True,
                "Runtime snapshot is not a separately verified canonical copy")
        expected = record.get("files", {})
        require(isinstance(expected, dict) and bool(expected), "Runtime snapshot has no file inventory")
        actual, total_bytes = set(), 0
        for target in frozen.rglob("*"):
            require(not target.is_symlink(), f"Runtime snapshot contains a symlink: {target}")
            if target.is_file():
                actual.add(target.relative_to(frozen).as_posix())
        require(actual == set(expected), f"Runtime snapshot membership changed: {label}")
        for relative, expected_sha in expected.items():
            target = frozen / relative
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                    and target.resolve().is_relative_to(frozen.resolve()) and target.absolute() == target.resolve(),
                    "Runtime snapshot file escapes the frozen root")
            info = target.stat()
            require(info.st_nlink == 1 and info.st_mode & 0o222 == 0 and sha256(target) == expected_sha,
                    f"Runtime snapshot is writable, linked or modified: {target}")
            original = source / relative
            if original.is_file():
                original_info = original.stat()
                require((info.st_dev, info.st_ino) != (original_info.st_dev, original_info.st_ino),
                        "Runtime snapshot shares the original checkout's inode")
            full_name = str(target)
            if full_name in manifest["files"]:
                require(manifest["files"][full_name] == expected_sha, "Deployment/snapshot source hash differs")
            total_bytes += info.st_size
        roots[label] = frozen
        totals[label] = {"files_hashed": len(expected), "bytes_hashed": total_bytes}
    simulator, openpi = roots["simulator"], roots["openpi-main"]
    require(plan.get("simulator_root") == str(simulator) and plan.get("model_project_root") == str(openpi)
            and manifest.get("runtime", {}).get("project_root") == str(openpi),
            "Model/simulator runtime still points at the original mutable checkout")
    script = plan.get("source_script", {})
    require(script.get("path") == str(simulator / "scripts/evaluate/evaluate_cpm_poker_checkpoint.py")
            and sha256(script["path"]) == script.get("sha256"), "Supervisor does not execute its frozen copy")
    controller_name = Path(document["controller_source"]).name
    require(sha256(simulator / "scripts/evaluate" / controller_name) == document.get("controller_sha256"),
            "Isolation controller was not captured in the frozen runtime")
    for name in manifest["files"]:
        candidate = Path(name)
        if candidate.suffix == ".py":
            require(any(candidate.is_relative_to(root) for root in roots.values()),
                    f"Deployment still binds Python from outside the isolated roots: {name}")
    if status is not None:
        runner = str(simulator / "scripts/workcell/run_poker_tactile_policy.py")
        for row in status.get("trials", {}).values():
            command = row.get("command")
            require(isinstance(command, list) and len(command) > 2 and command[2] == runner,
                    "A trial used the original mutable simulator runner")
    require(plan.get("prior_interrupted_trials_counted") == 0,
            "Interrupted non-isolated trials were counted toward the new ten trials")
    return {"runtime_snapshot_sha256": sha256(path), "isolated_runtime": True,
            "runtime_snapshots": totals, "prior_interrupted_trials_counted": 0}


def inspect_randomization(detail, seed):
    meta = detail["initial_card_randomization"]
    expected = {"schema_version": "poker-initial-card-randomization-v1", "mode": "seeded_uniform",
                "distribution": "independent_uniform", "seed": seed, "xy_jitter_m": .004,
                "yaw_jitter_rad": math.radians(.5), "perturbation_scope": "reset_only_card_xy_yaw",
                "height_unchanged": True, "tilt_unchanged": True, "observation_noise": None, "action_noise": None}
    require(all(meta.get(key) == value for key, value in expected.items()), "Wrong reset-only randomized initial state")
    generator = np.random.default_rng(seed)
    xy, yaw = generator.uniform(-.004, .004, size=2), generator.uniform(-math.radians(.5), math.radians(.5))
    require(np.allclose(array(meta["sampled_offset_xy_m"], (2,), "Invalid sampled XY"), xy, atol=1e-12, rtol=0),
            "Initial XY does not reproduce from the seed")
    same_number(meta["sampled_yaw_offset_rad"], float(yaw), "Initial yaw does not reproduce from the seed")
    nominal = array(meta["nominal_pose_wxyz"], (7,), "Invalid nominal pose")
    sampled = array(meta["sampled_pose_wxyz"], (7,), "Invalid sampled pose")
    expected_pose = nominal.copy()
    expected_pose[:2] += xy
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    w, x, y, z = nominal[3:]
    expected_pose[3:] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    require(np.allclose(sampled, expected_pose, atol=1e-12, rtol=0)
            and np.allclose(sampled, detail["initial_card_pose"], atol=1e-12, rtol=0), "Initial pose differs from the recorded reset")
    support = meta["table_support"]
    require(support.get("valid") is True and support["minimum_corner_margin_xy_m"] >= .00025
            and 0 <= support["bottom_gap_m"] <= .001, "Unsupported initial card pose")
    return sampled.tolist()


def inspect_requests(path, detail, seed):
    count = nonzero = positive_gradients = 0
    previous = None
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        require(row.get("request") == count + 1 and row.get("request_index") == count
                and row.get("episode_seed") == seed and row.get("action_start_index") == 0,
                "Request identity/order mismatch")
        stamp = row.get("sim_time")
        require(finite(stamp) and stamp >= 0 and (previous is None or 0 < stamp - previous <= .18),
                "Broken synchronized request clock")
        same_number(stamp, row.get("observation_sim_time"), "Simulation advanced during inference")
        maps = array(row.get("tactile_map_raw"), (5, 3, 7, 5), "Missing finite real tactile request tensor")
        maximum = float(np.abs(maps).max())
        same_number(row.get("tactile_map_abs_max_n"), maximum, "Logged tactile magnitude differs from real request tensor")
        require(row.get("tactile_sent_to_model") is True, "CPM tactile input was disabled")
        array(row.get("world_T_head"), (4, 4), "Missing synchronized head pose")
        array(row.get("world_T_wrist"), (4, 4), "Missing synchronized wrist pose")
        array(row.get("state"), (27,), "Invalid model state")
        array(row.get("predicted_actions"), (30, 27), "Missing finite H30 action chunk in raw-enabled request")
        diagnostic = row.get("diagnostics", {})
        noise_seed = int(np.random.SeedSequence([seed, count]).generate_state(1, dtype=np.uint64)[0])
        require(diagnostic.get("variant") == "pi05+touchguidance" and diagnostic.get("noise_seed") == noise_seed
                and diagnostic.get("guidance") == {"num_steps": 10, "guidance_steps": 3, "scale": 1.0}
                and diagnostic.get("base_comparison") is True, "Missing genuine guided-model diagnostic response")
        steps = diagnostic.get("steps", [])
        require(len(steps) == 10 and all(step.get("step") == index and step.get("guided") is (index >= 7)
                                       for index, step in enumerate(steps)), "CPM late-flow guidance did not execute")
        for step in steps[7:]:
            for key in ("gradient_rms", "velocity_correction_rms", "latent_guidance_shift_rms", "padded_guidance_max_abs"):
                require(finite(step.get(key)) and step[key] >= 0, "Missing finite CPM gradient/correction evidence")
            require(step["padded_guidance_max_abs"] == 0, "CPM guidance modified padded action coordinates")
            positive_gradients += int(step["gradient_rms"] > 0)
        nonzero += int(maximum > 0)
        previous, count = stamp, count + 1
    require(count == detail["stats"]["requests"] and count >= 538 and previous is not None and previous >= 89.7,
            "Requests do not cover the full 90-second closed loop")
    require(positive_gradients > 0, "No nonzero CPM guidance gradients were observed")
    return {"request_count": count, "tactile_nonzero_request_count": nonzero,
            "positive_guidance_gradient_steps": positive_gradients}


def inspect_stages(directory, detail, identity):
    stage = load(directory / "staged_evaluation.json")
    require(detail.get("staged_evaluation") == stage and stage.get("version") == "staged_v1"
            and stage.get("task") == TASK and stage.get("stage_definitions") == DEFINITIONS,
            "Native stage report mismatch")
    flags = stage.get("stages", {})
    require(set(flags) == set(DEFINITIONS) and all(type(value) is bool for value in flags.values())
            and stage.get("stage_flags") == flags and stage.get("complete_success") is flags["stage3"]
            and stage.get("legacy_task_success") is detail["task_success"]
            and (not flags["stage2"] or flags["stage1"]) and (not flags["stage3"] or flags["stage2"]),
            "Native/staged success fields were overwritten or stage order violated")
    require(stage.get("native_task_success_preserved") is True and stage.get("expert_trajectory_used") is False
            and stage.get("simulation_state_mutated") is False and stage.get("full_duration_evaluation") is True,
            "Stage observer changed the native task")
    require(stage.get("thresholds", {}).get("poker_terminal_hold_s") == 3.0,
            "Terminal stage must require the original 3-second hold")
    count, first, previous, old_flags = 0, None, None, None
    for line in (directory / "staged_states.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        require(all(row.get(key) == value for key, value in identity.items() if key not in {"model_family", "action_start_index"}),
                "Stage state identity drift")
        stamp, current = row.get("time_s"), row.get("stages")
        require(finite(stamp) and stamp >= 0 and (previous is None or 0 < stamp - previous <= .1)
                and row.get("control_tick") == count, "Broken 30-Hz stage state clock")
        require(row.get("thresholds") == stage["thresholds"] and row.get("stage_definitions") == DEFINITIONS
                and isinstance(current, dict) and set(current) == set(DEFINITIONS)
                and all(type(value) is bool for value in current.values()) and row.get("stage_flags") == current
                and (not current["stage2"] or current["stage1"]) and (not current["stage3"] or current["stage2"])
                and (old_flags is None or all(not value or current[key] for key, value in old_flags.items())),
                "Stage flags regressed or definitions/thresholds changed")
        first = stamp if first is None else first
        previous, old_flags, count = stamp, current, count + 1
    require(count >= 2698 and stage.get("state_log_samples") == count and first is not None and first <= .1
            and previous >= 89.93 and old_flags == flags, "Stage states do not cover the full native rollout")
    return {"stages": flags, "state_samples": count, "stage_thresholds": stage["thresholds"]}


def inspect_media(directory, detail, identity, *, ffprobe="ffprobe"):
    review = load(directory / "review.json")
    require(review.get("model_input_cameras_displayed") == ["head", "right_wrist"] and review.get("fps") == 10
            and review.get("physics_modified") is False and review.get("head_camera_modified") is False,
            "Video does not preserve both model-input cameras/physics")
    require(all(review.get(key) == value for key, value in identity.items() if key != "deployment_id")
            and ("deployment_id" not in review or review["deployment_id"] == identity.get("deployment_id")),
            "Video identity mismatch")
    manifest_sha = detail.get("deployment_manifest_sha256")
    require(isinstance(manifest_sha, str) and len(manifest_sha) == 64
            and all(character in "0123456789abcdef" for character in manifest_sha)
            and review.get("deployment_manifest_sha256") == manifest_sha and detail.get("video") == review,
            "Video metadata is not bound by the complete native summary and deployment manifest SHA")
    timestep = review.get("contact_model", {}).get("physics_timestep_used_s")
    require(finite(timestep) and timestep > 0, "Missing finite recorded physics timestep for cached video epochs")
    count, first, previous, previous_cached = 0, None, None, None
    zero_lag = one_step_lag = 0
    maximum_lag = 0.0
    for line in (directory / "frames.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        stamp = row.get("simulation_time_s")
        require(row.get("frame") == count and finite(stamp) and (previous is None or 0 < stamp - previous <= .11),
                "Broken video frame clock")
        cached = row.get("camera_pose_time_s")
        require(finite(cached) and cached >= 0 and (previous_cached is None or cached > previous_cached),
                "Broken cached camera/solver epoch clock")
        same_number(row.get("tactile_time_s"), cached, "Video camera and tactile epochs differ")
        lag = stamp - cached
        is_zero = math.isclose(lag, 0.0, rel_tol=0, abs_tol=1e-8)
        is_one_step = math.isclose(lag, timestep, rel_tol=1e-8, abs_tol=1e-8)
        require(is_zero or is_one_step, "Video cache epoch is neither current nor exactly one recorded physics step old")
        zero_lag += int(is_zero)
        one_step_lag += int(not is_zero and is_one_step)
        maximum_lag = max(maximum_lag, lag)
        first = stamp if first is None else first
        previous, previous_cached, count = stamp, cached, count + 1
    require(count == review.get("frame_count") and count >= 900 and first is not None and first <= .1
            and previous >= 89.9, "Incomplete 90-second video frame manifest")
    video = directory / "review.mp4"
    command = [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", str(video)]
    probe = json.loads(subprocess.run(command, capture_output=True, text=True, check=True, timeout=120).stdout)
    streams = probe.get("streams", [])
    require(len(streams) == 1, "No readable video stream")
    stream = streams[0]
    require(int(stream.get("nb_read_frames", 0)) == count and stream.get("avg_frame_rate") == "10/1"
            and [stream.get("width"), stream.get("height")] == review.get("output_size")
            and 89.9 <= float(stream["duration"]) <= 90.3, "Decoded video frame count/timing differs from native capture")
    return {"model_input_frames": count, "video_sha256": sha256(video), "video_probe": stream,
            "video_cached_epoch": {"physics_timestep_s": timestep, "zero_lag_frames": zero_lag,
                "one_physics_step_lag_frames": one_step_lag, "maximum_simulation_minus_cached_s": maximum_lag,
                "timing_note": "Camera and tactile share the same cached FK/solver epoch; data.time may be exactly one physics timestep later"}}


def inspect_raw(directory, detail, identity):
    import h5py
    raw = detail.get("raw", {})
    require(raw.get("validated") is True and raw.get("episode") == "raw/episode.h5", "No validated synchronized Raw episode")
    episode = directory / raw["episode"]
    result = load(directory / raw["result"])
    require(result.get("success") is detail["task_success"] and result.get("status") == detail["status"],
            "Raw outcome differs from native task result")
    with h5py.File(episode, "r") as file:
        meta, outcome = json.loads(file.attrs["metadata_json"]), json.loads(file.attrs["outcome_json"])
        for key in ("seed", "execute_steps", "checkpoint_path", "checkpoint_sha256", "deployment_id"):
            require(meta.get(key) == identity[key], f"Raw identity mismatch at {key}")
        require(outcome.get("success") is detail["task_success"] and outcome.get("status") == detail["status"],
                "HDF5 outcome differs from native task result")
        times = np.asarray(file["state/timestamp"])
        require(len(times) == raw.get("state_samples") and len(times) >= 8998 and np.isfinite(times).all()
                and np.all(np.diff(times) > 0) and times[0] <= .01 and times[-1] >= 89.99, "Incomplete finite raw state clock")
        same_number(float(times[-1]), detail["sim_seconds"], "Raw terminal state differs from native terminal time")
        for key in ("state/qpos", "state/qvel", "commands/arm_joint_target", "commands/hand_joint_target",
                    "objects/card/pose_wxyz", "tactile_contact_force/normal_force_n", "tactile_contact_force/tangent_force_n"):
            data = file[key]
            require(data.shape[0] == len(times), f"Raw synchronized length mismatch: {key}")
            for start in range(0, len(times), 512):
                require(np.isfinite(data[start:start + 512]).all(), f"Nonfinite raw evidence: {key}")
        require(json.loads(file["cameras"].attrs["configured_names_json"]) == ["head", "right_wrist"],
                "Raw model cameras changed")
        cameras = {}
        for name in ("head", "right_wrist"):
            group = file[f"cameras/{name}"]
            stamps = np.asarray(group["timestamp"])
            indices = np.asarray(group["state_index"])
            rgb = group["rgb"]
            require(len(stamps) >= 2698 and rgb.shape == (len(stamps), 240, 320, 3) and rgb.dtype == np.uint8
                    and np.isfinite(stamps).all() and np.all(np.diff(stamps) > 0)
                    and indices.shape == stamps.shape and np.all((0 <= indices) & (indices < len(times)))
                    and np.all(times[indices] <= stamps + 1e-9), "Incomplete or unsynchronized Raw model camera")
            require(np.ptp(rgb[0]) > 0 and np.ptp(rgb[-1]) > 0, "Raw model camera is blank")
            same_number(float(stamps[-1]), float(times[-1]), "Raw terminal camera not synchronized")
            cameras[name] = len(stamps)
        family_label = meta.get("model_family")
    require((directory / raw["right_hand_force_curves"]).stat().st_size > 0, "Missing raw force plot")
    return {"raw_state_samples": len(times), "raw_camera_samples": cameras, "raw_sha256": sha256(episode),
            "raw_metadata_model_family": family_label,
            "raw_family_label_note": "Legacy shared raw writer labels pi0.5; frozen manifest and worker handshake identify pi05+cpm"}


def inspect_trial(directory, manifest, manifest_sha, seed, *, ffprobe="ffprobe"):
    directory = Path(directory)
    detail = load(directory / "summary.json")
    identity = {"task": TASK, "model_family": FAMILY, "seed": seed, "prediction_horizon": 30,
                "execute_steps": 5, "action_start_index": 0, "checkpoint_path": manifest["checkpoint_path"],
                "checkpoint_sha256": manifest["checkpoint_sha256"], "deployment_id": manifest["deployment_id"]}
    require(all(detail.get(key) == value for key, value in identity.items())
            and detail.get("deployment_manifest_sha256") == manifest_sha, "Native trial identity mismatch")
    require(detail.get("valid_trial") is True and detail.get("trial_validity") == "valid"
            and detail.get("physics_valid") is True and not detail.get("error")
            and not any(key.endswith("record_error") for key in detail), "Infrastructure-invalid trajectory cannot enter a task-success denominator")
    require(type(detail.get("task_success")) is bool and detail.get("native_success") is detail["task_success"]
            and detail.get("evaluation", {}).get("success") is detail["task_success"]
            and detail.get("status") == ("success" if detail["task_success"] else "task_not_completed"),
            "Native failure/success was relabelled")
    require(finite(detail.get("sim_seconds")) and 90 - 1e-8 <= detail["sim_seconds"] <= 90.01
            and detail.get("max_sim_seconds") == 90 and detail.get("full_duration_evaluation") is True
            and detail.get("stop_reason") == ("success" if detail["task_success"] else "time_limit"),
            "Trial did not execute the full 90 seconds with its native terminal reason")
    require(detail.get("guard_mode") == "observe" and detail.get("action_interpolation") == "zoh"
            and detail.get("controller") == "poker-tactile-direct-joints-v1" and detail.get("control_hz") == 30
            and detail.get("diagnostic_only") is True and detail.get("formal_metrics_valid") is False,
            "Physical controller/protocol or diagnostic status changed")
    require(detail.get("tactile_sent_to_model") is True and detail.get("tactile_conditioning_enabled") is True
            and detail.get("simulation_paused_during_inference") is True
            and detail.get("guidance") == {"enabled": True, "steps": 3, "scale": 1.0, "variant": "pi05+touchguidance"},
            "CPM is not genuinely enabled in the closed loop")
    metadata = detail.get("server_metadata", {})
    require(metadata.get("normalization_verified") is True and metadata.get("training_dependencies_verified") is True
            and all(metadata.get(key) == value for key, value in identity.items() if key not in {"seed"})
            and metadata.get("deployment_manifest_sha256") == manifest_sha, "Worker identity/normalization handshake failed")
    require(metadata.get("source") == manifest.get("source")
            and metadata.get("package_versions") == manifest.get("runtime", {}).get("package_versions")
            and metadata.get("runtime", {}).get("project_root") == manifest.get("runtime", {}).get("project_root"),
            "Worker source/runtime differs from the frozen deployment")
    pose = inspect_randomization(detail, seed)
    stages = inspect_stages(directory, detail, identity)
    requests = inspect_requests(directory / "requests.jsonl", detail, seed)
    with np.load(directory / "first_request.npz", allow_pickle=False) as first:
        array(first["predicted_actions"], (30, 27), "Missing first H30 action chunk")
        array(first["tactile_map_raw"], (5, 3, 7, 5), "Missing first tactile map")
        for name in ("head_image", "right_wrist_image"):
            image = first[name]
            require(image.shape == (240, 320, 3) and image.dtype == np.uint8 and np.ptp(image) > 0,
                    "Missing/nonusable first model-input image")
    require((directory / "evaluation_rollout_trace.npz").stat().st_size > 0, "Missing native rollout trace")
    return {"seed": seed, "source": str(directory.resolve()), "summary_sha256": sha256(directory / "summary.json"),
            "native_success": detail["task_success"], "initial_pose": pose, **stages, **requests,
            **inspect_media(directory, detail, identity, ffprobe=ffprobe), **inspect_raw(directory, detail, identity)}


def inspect_evaluation(evaluation, manifest, manifest_sha, *, ffprobe="ffprobe"):
    batch = load(evaluation / "batch_summary.json")
    require(batch.get("schema") == "poker_tactile_batch_v1" and batch.get("task") == TASK and batch.get("model_family") == FAMILY
            and batch.get("complete") is True and batch.get("valid_trials") == 10 and batch.get("invalid_trials") == 0
            and batch.get("deployment_id") == manifest["deployment_id"] and batch.get("deployment_manifest_sha256") == manifest_sha,
            "Ten-trial batch is incomplete or contains invalid trials")
    rows = batch.get("trials", [])
    require(len(rows) == 10 and {row.get("seed") for row in rows} == set(range(10)), "Ten seeds were omitted/duplicated")
    members = []
    for row in sorted(rows, key=lambda item: item["seed"]):
        require(row.get("valid") is True and row.get("returncode") == 0 and row.get("timed_out") is False
                and row.get("identity_matches") is True, "Failed subprocess was hidden or counted as a task failure")
        summary = (evaluation / row["summary"]).resolve(strict=True)
        require(summary.is_relative_to(evaluation.resolve()), "Trial summary escapes evaluation root")
        member = inspect_trial(summary.parent, manifest, manifest_sha, row["seed"], ffprobe=ffprobe)
        require(row.get("task_success") is member["native_success"], "Batch success differs from native trial")
        members.append(member)
    count = sum(member["native_success"] for member in members)
    same_number(batch.get("success_rate"), count / 10, "Wrong native success denominator")
    require(all(member["stage_thresholds"] == members[0]["stage_thresholds"] for member in members),
            "Stage thresholds changed within the ten-trial batch")
    return {"planned_trials": 10, "native_valid_trials": 10, "native_success_count": count,
            "native_success_rate": count / 10, "stage_valid_trials": 10, "stage_definitions": DEFINITIONS,
            "stage_counts": {key: sum(member["stages"][key] for member in members) for key in DEFINITIONS},
            "initial_state_scope": "seeded_reset_only_xy_plusminus_4mm_yaw_plusminus_0.5deg",
            "unique_recorded_initial_poses": len({json.dumps(member["initial_pose"]) for member in members}), "trials": members}


def inspect_baseline(baseline):
    require(sha256(baseline / "catalog.json") == BASELINE_CATALOG_SHA
            and sha256(baseline / "artifact_manifest.json") == BASELINE_MANIFEST_SHA,
            "Original 595-trial baseline catalog/manifest changed")
    return load(baseline / "catalog.json"), load(baseline / "artifact_manifest.json")


def inspect_snapshot(snapshot, baseline, raw, manifest, manifest_sha):
    before, inventory = inspect_baseline(baseline)
    after = load(snapshot / "catalog.json")
    original = {group["id"]: group for group in before["groups"]}
    updated = {group["id"]: group for group in after["groups"]}
    require(all(updated.get(key) == value for key, value in original.items()), "Original 595 groups or scores changed")
    added = [value for key, value in updated.items() if key not in original]
    require(len(added) == 1 and len(updated) == 48 and after["counts"].get("members") == 605
            and after["counts"].get("eligible") == 585 and after["counts"].get("quarantined") == 20,
            "Expected exactly ten additive CPM members and the preserved 595")
    group = added[0]
    require(group.get("task") == TASK and group.get("model") == FAMILY and group.get("prediction_horizon") == 30
            and group.get("execute_steps") == 5 and group.get("cpm_sha256") == manifest["checkpoint_sha256"]
            and group.get("deployment_manifest_sha256") == manifest_sha
            and group.get("checkpoint") == "best@" + manifest["checkpoint_sha256"][:12],
            "Current CPM best overwrote or was confused with the historical best")
    trials = group.get("trials", [])
    require(len(trials) == 10 and {item.get("seed") for item in trials} == set(range(10)), "Platform omitted/duplicated a new CPM seed")
    indexed = {item["seed"]: item for item in raw["trials"]}
    for trial in trials:
        native = indexed[trial["seed"]]
        require(trial.get("valid") is True and trial.get("publication_status") == "eligible"
                and trial.get("native_success") is native["native_success"]
                and trial.get("summary_sha256") == native["summary_sha256"]
                and {item["key"]: item["reached"] for item in trial["stages"]} == native["stages"],
                "Published CPM outcomes or stage flags differ from native records")
    old_dirs = {Path(trial["detail_url"]).parent.as_posix() for group in before["groups"] for trial in group["trials"]}
    verified = 0
    for row in inventory["files"]:
        if Path(row["path"]).parts[0] not in old_dirs:
            continue
        path = snapshot / row["path"]
        require(path.is_file() and path.stat().st_size == row["size"] and sha256(path) == row["sha256"],
                f"Original 595 payload changed: {row['path']}")
        verified += 1
    return {"members": 605, "groups": 48, "new_members": 10, "preserved_members": 595,
            "preserved_payload_files_hashed": verified, "group_id": group["id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    require(not args.receipt.exists(), "Refusing to overwrite a QA receipt")
    receipt = {"schema": "kaihand-cpm-poker-independent-qa-v1", "passed": False,
               "created_utc": datetime.now(timezone.utc).isoformat(), "job_root": str(args.job_root.resolve()),
               "evaluation_root": str(args.evaluation_root.resolve()), "deployment_manifest": str(args.deployment_manifest.resolve())}
    try:
        manifest = inspect_manifest(args.deployment_manifest)
        manifest_sha = sha256(args.deployment_manifest)
        receipt.update(inspect_job(args.job_root, args.deployment_manifest, args.evaluation_root, manifest))
        receipt.update(inspect_runtime_snapshot(args.job_root, load(args.job_root / "plan.json"), manifest,
                                               load(args.job_root / "status.json")))
        inspect_baseline(args.baseline)
        raw = inspect_evaluation(args.evaluation_root, manifest, manifest_sha, ffprobe=args.ffprobe)
        source = manifest["source"]
        receipt.update(raw, deployment_manifest_sha256=manifest_sha, checkpoint_sha256=manifest["checkpoint_sha256"],
                       cpm_checkpoint_sha256=manifest["checkpoint_sha256"], base_checkpoint_path=source["base_checkpoint"],
                       deployment_id=manifest["deployment_id"], cpm_checkpoint_path=source["cpm_checkpoint"],
                       base_checkpoint_files={name: value for name, value in manifest["files"].items()
                                              if Path(name).is_relative_to(Path(source["base_checkpoint"]))},
                       diagnostic_only=True, formal_metrics_valid=False)
        if args.snapshot:
            receipt["snapshot"] = inspect_snapshot(args.snapshot, args.baseline, raw, manifest, manifest_sha)
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(args.receipt), "passed": receipt["passed"], "error": receipt.get("error")}, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
