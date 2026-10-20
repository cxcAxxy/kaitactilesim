#!/usr/bin/env python3
"""Independent read-only QA for the exact 25-trial in-place bulb evaluation.

No runner/exporter, simulator, model or checkpoint code is imported. Only an
explicit new receipt is written; all native and published inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener

TASK = "bulb-screw-inplace"
CONTROLLER = "pi05-absolute-joint-physics-interpolation-v2"
STEPS = (15000, 18000, 21000, 24000, 24999)
DEFINITIONS = dict(zip(("stage1", "stage2", "stage3", "stage4"), (
    "stable_inplace_grasp", "threaded_and_seated", "tightening_verified", "released_stably"), strict=True))
CHECKPOINT_ROOT = Path("/nas/chenxianchi/datasets/sim/bulb-screw-inplace/pi05/1018_200/checkpoints/"
                       "pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1/bulb-screw-inplace_bs128_25k")
BASELINE_CATALOG_SHA = "362d072f7401319bb59bb66a736ac8256b42e17d1d65da83c1c146e3b98a92af"
BASELINE_MANIFEST_SHA = "de7ea18423235b4247aa03de1ac21db0770ba3303603ae6f1dc9bcc65d899ec4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal_number(left, right, label):
    require(finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9), label)


def inspect_trial(directory, group, deployment, seed):
    detail = load(directory / "summary.json")
    staged = load(directory / "staged_evaluation.json")
    expected_identity = {"task": TASK, "seed": seed, "prediction_horizon": 50, "execute_steps": 50,
                         "checkpoint_path": group["checkpoint"], "checkpoint_sha256": deployment["checkpoint_sha256"],
                         "deployment_id": deployment["deployment_id"], "action_interpolation": "linear"}
    require(all(detail.get(key) == value for key, value in expected_identity.items()), f"Wrong native identity: {directory}")
    require(detail.get("physics_valid") is True and not detail.get("error") and not detail.get("record_error"),
            f"Infrastructure-invalid trial cannot become a task failure: {directory}")
    require(type(detail.get("task_success")) is bool and detail.get("status") == (
        "success" if detail["task_success"] else "task_not_completed"), f"Native outcome/status mismatch: {directory}")
    require(finite(detail.get("sim_seconds")) and 90 - 1e-8 <= detail["sim_seconds"] <= 90.01,
            f"Not a full 90-second simulated rollout: {directory}")
    require(detail.get("full_duration_evaluation") is True and detail.get("diagnostic_only") is True
            and detail.get("formal_metrics_valid") is False and detail.get("expert_used") is False,
            f"Evaluation must remain passive full-duration diagnostic: {directory}")
    require(detail.get("controller") == CONTROLLER, f"Wrong physical action controller: {directory}")
    require(detail.get("staged_evaluation") == staged and staged.get("version") == "staged_v1"
            and staged.get("task") == TASK and staged.get("stage_definitions") == DEFINITIONS,
            f"Independent four-stage report mismatch: {directory}")
    flags = staged.get("stages")
    require(isinstance(flags, dict) and set(flags) == set(DEFINITIONS)
            and all(type(value) is bool for value in flags.values()) and staged.get("stage_flags") == flags,
            f"Native stage flags missing: {directory}")
    require(staged.get("full_duration_evaluation") is True and staged.get("complete_success") is flags["stage4"]
            and staged.get("native_task_success_preserved") is True
            and staged.get("expert_trajectory_used") is False and staged.get("simulation_state_mutated") is False,
            f"Stage observation changed the native task: {directory}")
    evaluation = detail["evaluation"]
    state, initial = evaluation["state"], evaluation["threaded_initialization"]
    require(initial.get("position_randomization_applied") is False
            and initial.get("physically_validated") is True and initial.get("initially_threaded_verified") is True
            and initial.get("initially_seated") is False and initial.get("initially_lit") is False,
            f"Fixed, verified, unseated threaded initial state not established: {directory}")
    initial_turns, final_turns, net_turns = (initial.get("initial_clockwise_turns"),
                                          state.get("clockwise_turns"), state.get("rotation_from_initial_turns"))
    require(all(finite(value) for value in (initial_turns, final_turns, net_turns)), f"Missing angle evidence: {directory}")
    equal_number(net_turns, final_turns - initial_turns, f"Net angle is not relative to actual initial state: {directory}")
    samples, first, previous, previous_flags = 0, None, None, None
    with (directory / "staged_states.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            require(all(row.get(key) == value for key, value in expected_identity.items() if key != "action_interpolation"),
                    f"State log identity drift: {directory}")
            stamp = row.get("time_s")
            require(finite(stamp) and stamp >= 0 and (previous is None or 0 < stamp - previous <= .1)
                    and row.get("control_tick") == samples, f"Broken 30-Hz state log: {directory}")
            require(row.get("stage_definitions") == DEFINITIONS and row.get("stage_flags") == row.get("stages")
                    and row.get("thresholds") == staged.get("thresholds"), f"State log stage contract drift: {directory}")
            current_flags = row["stages"]
            require(set(current_flags) == set(DEFINITIONS) and all(type(value) is bool for value in current_flags.values())
                    and (previous_flags is None or all(not value or current_flags[key] for key, value in previous_flags.items())),
                    f"Stage evidence regressed: {directory}")
            first = stamp if first is None else first
            previous, previous_flags, samples = stamp, current_flags, samples + 1
    require(samples >= 2698 and first is not None and first <= .1 and previous >= 89.93 and previous_flags == flags,
            f"State log lacks full 90-second native coverage: {directory}")
    review = directory / "review" if (directory / "review/review.json").is_file() else directory
    require(load(review / "review.json").get("model_input_cameras_displayed") == ["head", "right_wrist"],
            f"Model head/wrist video missing: {directory}")
    for name in ("review.mp4", "frames.jsonl", "evaluation_rollout_trace.npz"):
        path = review / name if name in {"review.mp4", "frames.jsonl"} else directory / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing native artifact: {path}")
    return {"step": group["step"], "seed": seed, "source": str(directory), "summary_sha256": sha256(directory / "summary.json"),
            "native_success": detail["task_success"], "stages": flags, "state_samples": samples,
            "initial_angle_deg": initial_turns * 360, "final_net_angle_deg": net_turns * 360,
            "net_angle_reference_deg": 100 - initial_turns * 360, "initial_pose": initial.get("bulb_pose_wxyz")}


def inspect_batch(batch):
    plan, status = load(batch / "batch.json"), load(batch / "status.json")
    require(plan.get("schema") == "kaihand-pi05-bulb-inplace-stage-batch-v1" and plan.get("total_groups") == 5
            and plan.get("total_trials") == 25, "Unexpected task batch contract")
    require(plan.get("initial_state_randomization") is False and plan.get("trial_scope") == "fixed_initial_state_repeats",
            "Fixed initial poses must not be described as randomized generalization")
    settings = plan["settings"]
    require(all(settings.get(key) == value for key, value in {"prediction_horizon": 50, "execute_steps": 50,
            "num_trials": 5, "video_count": 5, "seed_start": 0, "max_sim_seconds": 90,
            "guard_mode": "observe", "staged_evaluation_version": "staged_v1"}.items()), "Batch setting drift")
    require([group.get("step") for group in plan["groups"]] == list(STEPS), "Checkpoint selection/order drift")
    require(status.get("state") == "complete" and len(status["groups"]) == 5, "Batch not complete")
    trials, groups = [], []
    for group, progress in zip(plan["groups"], status["groups"], strict=True):
        require(group.get("task") == TASK and group.get("checkpoint") == str(CHECKPOINT_ROOT / str(group["step"])),
                "Checkpoint path/task must match the exact authorized training run")
        require(progress.get("state") == "complete" and type(progress.get("attempt")) is int and progress["attempt"] >= 1,
                "Group not complete")
        attempt = batch / group["group_dir"] / f"attempt_{progress['attempt']:03d}"
        require(attempt.resolve().is_relative_to(batch.resolve()), "Attempt escapes batch")
        evaluation = attempt / "run/evaluation"
        deployment, run = load(attempt / "deployment/deployment_manifest.json"), load(attempt / "run/run_manifest.json")
        require(run.get("evaluation_returncode") == 0 and run.get("managed_server_stopped") is True
                and run.get("evaluation_process_stopped") is True, "Infrastructure did not complete cleanly")
        summary, protocol = load(evaluation / "summary.json"), load(evaluation / "protocol.json")
        require(protocol.get("max_sim_seconds") == 90 and protocol.get("guard_mode") == "observe"
                and protocol.get("action_interpolation") == "linear" and protocol.get("controller") == CONTROLLER,
                "Native protocol differs from 90-second observe")
        rows = summary.get("results")
        require(summary.get("planned") == summary.get("attempted") == summary.get("valid_trials") == 5
                and isinstance(rows, list) and len(rows) == 5 and {row.get("seed") for row in rows} == set(range(5))
                and all(row.get("valid_trial") is True for row in rows), "Invalid trials must not be zero-filled or omitted")
        members = [inspect_trial(evaluation / f"seed_{seed:03d}", group, deployment, seed) for seed in range(5)]
        staged = summary["staged_evaluation"]
        require(staged.get("valid_staged_trials") == 5, "Wrong staged denominator")
        for key, definition in DEFINITIONS.items():
            count = sum(trial["stages"][key] for trial in members)
            value = staged["stages"][key]
            require(value.get("definition") == definition and value.get("count") == count, "Aggregate stage count differs")
            equal_number(value.get("rate"), count / 5, "Aggregate staged rate differs")
        native_count = sum(trial["native_success"] for trial in members)
        require(summary.get("task_successes") == native_count, "Native batch success count differs from native trial outcomes")
        equal_number(summary.get("success_rate"), native_count / 5, "Native batch rate differs from native trial denominator")
        require(staged.get("complete_success_count") == sum(trial["stages"]["stage4"] for trial in members),
                "Staged completion aggregate differs from native stage flags")
        groups.append({"checkpoint": group["step"], "valid_trials": 5,
                       "native_success_count": native_count,
                       "stage_counts": {key: sum(trial["stages"][key] for trial in members) for key in DEFINITIONS},
                       "mean_final_net_angle_deg": sum(trial["final_net_angle_deg"] for trial in members) / 5})
        trials.extend(members)
    return {"trials": trials, "groups": groups, "total_trials": len(trials),
            "fixed_initial_state_repeats": True, "unique_recorded_initial_poses": len({json.dumps(t["initial_pose"]) for t in trials})}


def inspect_snapshot(snapshot, baseline, raw, http_base=None):
    require(sha256(baseline / "catalog.json") == BASELINE_CATALOG_SHA
            and sha256(baseline / "artifact_manifest.json") == BASELINE_MANIFEST_SHA, "Published 595-trial baseline changed")
    before, after = load(baseline / "catalog.json"), load(snapshot / "catalog.json")
    old_groups, all_groups = {g["id"]: g for g in before["groups"]}, {g["id"]: g for g in after["groups"]}
    require(all(all_groups.get(key) == value for key, value in old_groups.items()), "Original groups/trial scores were edited")
    new_groups = [group for key, group in all_groups.items() if key not in old_groups]
    require(len(all_groups) == 52 and len(new_groups) == 5
            and sorted(int(group["checkpoint"]) for group in new_groups) == list(STEPS), "Expected five additive checkpoint groups")
    counts = after["counts"]
    require(counts.get("members") == 620 and counts.get("eligible") == 600 and counts.get("quarantined") == 20,
            "Published membership/admission drift")
    old_trial_dirs = {Path(trial["detail_url"]).parent.as_posix() for group in before["groups"] for trial in group["trials"]}
    preserved, retained_bytes = 0, 0
    for row in load(baseline / "artifact_manifest.json")["files"]:
        if Path(row["path"]).parts[0] not in old_trial_dirs:
            continue
        path = snapshot / row["path"]
        require(path.is_file() and path.stat().st_size == row["size"] and sha256(path) == row["sha256"],
                f"Original 595-trial payload changed: {row['path']}")
        preserved += 1
        retained_bytes += row["size"]
    raw_by_key = {(trial["step"], trial["seed"]): trial for trial in raw["trials"]}
    details = []
    for group in new_groups:
        require(group.get("task") == TASK and group.get("model") == "pi05" and group.get("prediction_horizon") == 50
                and group.get("execute_steps") == 50 and len(group["trials"]) == 5, "New platform group identity drift")
        for trial in group["trials"]:
            original = raw_by_key[(int(group["checkpoint"]), trial["seed"])]
            require(trial.get("valid") is True and trial.get("publication_status") == "eligible"
                    and trial.get("native_success") is original["native_success"]
                    and trial.get("summary_sha256") == original["summary_sha256"], "Published admission/outcome differs from raw trial")
            require({item["key"]: item["reached"] for item in trial["stages"]} == original["stages"], "Published stage flags changed")
            directory = snapshot / Path(trial["detail_url"]).parent
            config, viz = load(directory / "workbench.json"), load(directory / "visualization.json")
            cards = [card for card in config["task_summary"] if card.get("key") == "rotation_from_initial_deg"]
            require(len(cards) == 1 and cards[0].get("unit") == "deg", "Missing terminal net-degree card")
            equal_number(cards[0].get("value"), original["final_net_angle_deg"], "Terminal net-degree card differs from raw")
            goals = config["goal_references"]
            require(goals.get("status") == "verified", "New trajectory lacks bound mechanical references")
            refs = [ref for ref in goals["references"] if ref.get("key") in {"rotation_from_initial_deg", "rotation_from_initial_turns"}]
            require(len(refs) == 1, "New trajectory lacks actual-initial-relative angle reference")
            ref = refs[0]
            target_degrees = ref["target"] * (360 if ref.get("unit") == "turn" else 1)
            equal_number(target_degrees, original["net_angle_reference_deg"], "Net reference must subtract actual initial angle")
            series = [item for item in viz["task_series"] if item.get("key") in {"rotation_from_initial_deg", "rotation_from_initial_turns"}]
            require(len(series) == 1 and len(series[0].get("goal_references", [])) == 1, "Net angle reference not attached to recorded curve")
            details.append({"id": trial["id"], "detail_url": trial["detail_url"], "video_url": trial["video_url"]})
    http = inspect_http(http_base, details) if http_base else None
    return {"members": 620, "groups": 52, "preserved_members": 595, "new_members": len(details),
            "preserved_payload_files_hashed": preserved, "preserved_payload_bytes": retained_bytes, "http": http}


def inspect_http(base, trials):
    opener = build_opener(ProxyHandler({}))
    checked = []
    for trial in trials:
        detail = urljoin(base, trial["detail_url"])
        with opener.open(Request(detail, method="HEAD"), timeout=30) as response:
            require(response.status == 200, f"Detail not available: {detail}")
        video = urljoin(base, trial["video_url"])
        with opener.open(Request(video, headers={"Range": "bytes=0-63"}), timeout=30) as response:
            require(response.status == 206 and response.headers.get("Content-Range", "").startswith("bytes 0-63/"),
                    f"Video byte-range unavailable: {video}")
            require(b"ftyp" in response.read(64), f"Not MP4 bytes: {video}")
        checked.append(trial["id"])
    return {"detail_head_200": len(checked), "mp4_range_206": len(checked), "trial_ids": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--http-base")
    parser.add_argument("--output", type=Path, required=True, help="new independent audit receipt; never overwritten")
    args = parser.parse_args()
    require(not args.output.exists(), "Audit receipt already exists")
    require(not args.snapshot or args.baseline, "Snapshot checks require an explicit baseline")
    raw = inspect_batch(args.batch.absolute())
    result = {"schema": "kaihand-bulb-inplace-independent-qa-v1", "status": "passed",
              "checked_utc": datetime.now(timezone.utc).isoformat(), "batch": str(args.batch.absolute()),
              "raw": raw, "simulation_runs_started": 0, "model_weights_loaded": False, "scores_recomputed": False}
    if args.snapshot:
        result["snapshot"] = inspect_snapshot(args.snapshot.absolute(), args.baseline.absolute(), raw, args.http_base)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "raw_trials": len(raw["trials"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
