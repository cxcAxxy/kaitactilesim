"""Offline poker Cleaning 1–3, adapted from the accepted 10 Hz audit.

Supports actual camera names and physics-quantized mixed-rate clocks.
No simulation, conversion or modification of source files.
"""

import csv
import hashlib
import json

import numpy as np


def decode(value):
  return value.decode() if isinstance(value, bytes) else str(value)


def ns(values):
  return np.rint(np.asarray(values, dtype=float) * 1e9).astype(np.int64)


def digest(path):
  result = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      result.update(block)
  return result.hexdigest()


def save(path, value):
  with path.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")


def clock_check(values, period_ns, end_ns, *, terminal=True, physics_step_ns=None):
  values = np.asarray(values, dtype=float)
  if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
    return {"valid": False, "error": "nonfinite, empty or nonvector clock"}
  actual = ns(values)
  expected = np.arange(int(np.floor(end_ns / period_ns)) + 2) * period_ns
  if physics_step_ns is not None:
    expected = np.floor(expected / physics_step_ns + 0.5 + 1e-10) * physics_step_ns
  expected = np.rint(expected).astype(np.int64)
  expected = expected[expected <= end_ns]
  extra_terminal = bool(terminal and expected[-1] != end_ns)
  if extra_terminal:
    expected = np.r_[expected, end_ns]
  missing, unexpected = np.setdiff1d(expected, actual), np.setdiff1d(actual, expected)
  delta = np.diff(actual)
  return {
    "valid": bool(len(actual) == len(expected) and np.array_equal(actual, expected)),
    "sample_count": len(actual),
    "expected_sample_count": len(expected),
    "duplicate_timestamp_count": int(len(actual) - len(np.unique(actual))),
    "reversed_intervals": int(np.sum(delta < 0)),
    "missing_expected_timestamps": len(missing),
    "unexpected_timestamps": len(unexpected),
    "missing_examples_s": (missing[:10] / 1e9).tolist(),
    "unexpected_examples_s": (unexpected[:10] / 1e9).tolist(),
    "minimum_interval_s": float(delta.min() / 1e9) if len(delta) else None,
    "maximum_interval_s": float(delta.max() / 1e9) if len(delta) else None,
    "final_interval_s": float(delta[-1] / 1e9) if len(delta) else None,
    "off_grid_terminal_frame_expected": extra_terminal,
  }


def check_clocks_and_controls(file, output):
  state = ns(file["state/timestamp"][:])
  trace = file["control/precontact_noise"]
  start = ns(trace["time_s"][:])
  step = int(round(1e9 / float(file.attrs["physics_hz"])))
  force = ns(file["tactile_contact_force/timestamp"][:])
  report = {
    "state": clock_check(file["state/timestamp"][:], 1e9 / float(file.attrs["control_hz"]), state[-1], physics_step_ns=step),
    "physics_trace": clock_check(
      trace["time_s"][:], step, state[-1] - step, terminal=False
    ),
    "state_minus_force_ns_range": [
      int((state - force).min()),
      int((state - force).max()),
    ],
    "force_cache_clock_matches": bool(
      np.array_equal(force, np.maximum(state - step, 0))
    ),
    "cameras": {},
  }
  for name in file["cameras"]:
    camera = file[f"cameras/{name}"]
    capture, pose, indices = (
      ns(camera["timestamp"][:]),
      ns(camera["pose_timestamp"][:]),
      camera["state_index"][:],
    )
    result = clock_check(camera["timestamp"][:], 1e9 / float(file.attrs["camera_hz"]), state[-1], physics_step_ns=step)
    safe = bool(np.all((indices >= 0) & (indices < len(state))))
    result.update(
      state_indices_valid=safe,
      latest_nonfuture_state_index_matches=bool(
        safe and np.array_equal(indices, np.searchsorted(state, capture, side="right") - 1)
      ),
      pose_cache_clock_matches=bool(np.array_equal(pose, np.maximum(capture - step, 0))),
      indexed_force_not_future=bool(safe and np.all(force[indices] <= pose)),
      maximum_indexed_force_age_ns=int(np.max(pose - force[indices])) if safe else None,
      max_capture_minus_pose_ns=int(np.max(capture - pose)),
      future_observation_count=int(np.sum(pose > capture)),
    )
    result["valid"] = bool(
      result["valid"]
      and safe
      and result["latest_nonfuture_state_index_matches"]
      and result["pose_cache_clock_matches"]
      and result["indexed_force_not_future"]
      and not result["future_observation_count"]
    )
    report["cameras"][name] = result
  names = [decode(v) for v in file["commands/actuator_names"][:]]
  ids = [names.index(f"right_arm_joint{i}") for i in range(1, 8)]
  recorded = file["commands/actuator_control"][:, ids]
  actual = trace["actual_ctrl_rad"][:]
  first = np.searchsorted(start, state[:-1])
  end = np.searchsorted(start, state[1:])
  count = end - first
  covered = count * step == np.diff(state)
  nonempty = count > 0
  if not np.all(nonempty):
    raise ValueError("state transition has no recorded right-arm action substeps")
  endpoint_errors = np.max(np.abs(recorded[1:] - actual[end - 1]), axis=1)
  naive_errors = np.array(
    [
      np.max(np.abs(actual[a:b] - recorded[i]))
      for i, (a, b) in enumerate(zip(first, end, strict=True))
    ]
  )
  phases = [decode(v) for v in file["commands/phase"][:]]
  with (output / "state_action_intervals.csv").open("x", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      [
        "state_t_index",
        "state_next_index",
        "state_t_s",
        "state_next_s",
        "action_start_index",
        "action_end_index_exclusive",
        "action_substeps",
        "first_action_time_s",
        "last_action_time_s",
        "all_substeps_present",
        "recorded_next_ctrl_error_rad",
        "naive_same_row_hold_error_rad",
        "phase_t",
        "phase_next",
      ]
    )
    for i, (a, b) in enumerate(zip(first, end, strict=True)):
      writer.writerow(
        [
          i,
          i + 1,
          state[i] / 1e9,
          state[i + 1] / 1e9,
          a,
          b,
          count[i],
          start[a] / 1e9,
          start[b - 1] / 1e9,
          bool(covered[i]),
          endpoint_errors[i],
          naive_errors[i],
          phases[i],
          phases[i + 1],
        ]
      )
  mapping_valid = bool(
    np.all(covered)
    and np.array_equal(start[first], state[:-1])
    and np.array_equal(start[end - 1] + step, state[1:])
    and np.max(endpoint_errors) <= 1e-12
  )
  report["right_arm_transition_mapping"] = {
    "valid": mapping_valid,
    "transitions": len(count),
    "all_substeps_covered": bool(np.all(covered)),
    "substep_count_distribution": {
      str(int(k)): int(v)
      for k, v in zip(*np.unique(count, return_counts=True), strict=True)
    },
    "max_recorded_ctrl_vs_previous_substep_error_rad": float(np.max(endpoint_errors)),
    "same_row_ctrl_is_not_a_held_next_interval_action": {
      "mismatching_intervals_above_1e_8_rad": int(np.sum(naive_errors > 1e-8)),
      "maximum_difference_rad": float(naive_errors.max()),
      "interpretation": "Different 500Hz inputs inside the next state interval are expected; do not use the 100Hz same-row ctrl as a held action.",
    },
    "full_robot_dynamics_verified": False,
    "limitation": "Full500Hz state and left-arm/finger actuator sequences were not archived. Only the right-arm executed-input timing can be fully indexed; high-level goals are not the actual ctrl.",
  }
  report["valid_for_declared_clock_semantics"] = bool(
    report["state"]["valid"]
    and report["physics_trace"]["valid"]
    and report["force_cache_clock_matches"]
    and all(v["valid"] for v in report["cameras"].values())
    and mapping_valid
  )
  return report


def check_duplicate_images(file):
  results = {}
  for name in file["cameras"]:
    camera = file[f"cameras/{name}"]
    times = camera["pose_timestamp"][:]
    wrists, tips = camera["world_from_wrist"][:], camera["world_from_fingertip"][:]
    poses = np.concatenate([wrists[..., None, :, :], tips], axis=2)
    indices = camera["state_index"][:]
    card = file["objects/card/pose_wxyz"][indices]
    seen, adjacent, repeated, suspects = {}, [], [], []
    previous = None
    for i in range(len(times)):
      pixels = np.asarray(camera["rgb"][i])
      sha = hashlib.sha256(pixels.tobytes()).hexdigest()
      if sha in seen:
        repeated.append([seen[sha], i])
      else:
        seen[sha] = i
      if sha == previous:
        translation = max(
          float(
            np.max(
              np.linalg.norm(poses[i, ..., :3, 3] - poses[i - 1, ..., :3, 3], axis=-1)
            )
          ),
          float(np.linalg.norm(card[i, :3] - card[i - 1, :3])),
        )
        relative_rotation = (
          np.swapaxes(poses[i - 1, ..., :3, :3], -1, -2) @ poses[i, ..., :3, :3]
        )
        angle = float(
          np.max(
            np.arccos(
              np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
            )
          )
        )
        row = {
          "previous_index": i - 1,
          "index": i,
          "time_s": float(times[i]),
          "max_tracked_translation_m": translation,
          "max_tracked_rotation_rad": angle,
        }
        adjacent.append(row)
        if translation > 0.002 or angle > 0.02:
          suspects.append(row)
      previous = sha
    results[name] = {
      "frames": len(times),
      "unique_pixel_hashes": len(seen),
      "identical_content_repeated_frames": len(repeated),
      "identical_adjacent_frame_pairs": len(adjacent),
      "adjacent_examples": adjacent[:15],
      "repeated_examples": repeated[:15],
      "suspected_stale_rgb_motion_pairs": suspects[:30],
      "suspect_count": len(suspects),
      "heuristic_screen": "Flag identical adjacent RGB only when a tracked wrist/finger/card translates >2mm or wrist/finger rotates >0.02rad; not proof of stale imagery because motion may be occluded.",
      "interpretation": "Matching pixels at distinct timestamps are not duplicate records by themselves (static scene/subpixel motion). No frames are removed.",
    }
  return results



