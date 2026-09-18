"""Read-only sampled trajectory screening; no MuJoCo or dynamics reconstruction.

All velocity/acceleration/residual cutoffs below are review heuristics, except
explicit recorded actuator command bounds. Contact transients remain candidates,
never an automatic reason to discard a physically successful episode.
"""

import json

import numpy as np


def _text(value):
  return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _quaternion_rotation(q):
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  w, x, y, z = np.moveaxis(q, -1, 0)
  return np.stack(
    (
      1 - 2 * (y * y + z * z),
      2 * (x * y - z * w),
      2 * (x * z + y * w),
      2 * (x * y + z * w),
      1 - 2 * (x * x + z * z),
      2 * (y * z - x * w),
      2 * (x * z - y * w),
      2 * (y * z + x * w),
      1 - 2 * (x * x + y * y),
    ),
    axis=-1,
  ).reshape((*q.shape[:-1], 3, 3))


def _world_rotation_increment(q):
  """Shortest world rotation vector q[t+1] * inverse(q[t]); q and -q agree."""
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  a, b = q[1:], q[:-1]
  scalar = a[:, 0] * b[:, 0] + np.sum(a[:, 1:] * b[:, 1:], axis=1)
  vector = -a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
  vector = np.where(scalar[:, None] < 0, -vector, vector)
  length = np.linalg.norm(vector, axis=1)
  scale = np.divide(
    2 * np.arctan2(length, np.abs(scalar)),
    length,
    out=np.zeros_like(length),
    where=length > 1e-15,
  )
  return vector * scale[:, None]


def _check_trajectory(file):
  """Accept an open read-only h5py File; return only JSON-serializable values."""
  report = {
    "schema_version": "poker-sampled-trajectory-cleaning-v1",
    "valid": False,
    "errors": [],
    "warnings": [],
    "streams": {},
    "metrics": {},
    "metadata_constraints": {},
    "simulator_loaded": False,
    "dynamics_reconstructed": False,
    "inputs_modified": False,
    "heuristic_scope": "finite sampled observations and approximate kinematic consistency; real contact acceleration and phase changes are not automatically bad data",
  }
  errors, metrics = report["errors"], report["metrics"]

  def array(path, shape=None):
    if path not in file:
      errors.append(f"missing required stream: {path}")
      return None
    value = np.asarray(file[path][:], dtype=float)
    okay = np.isfinite(value).all() and (shape is None or value.shape == shape)
    report["streams"][path] = {
      "shape": list(value.shape),
      "finite": bool(np.isfinite(value).all()),
      "nonfinite_count": int((~np.isfinite(value)).sum()),
    }
    if not okay:
      errors.append(f"nonfinite values or unexpected shape: {path}")
      return None
    return value

  def measure(label, values, times, phases, threshold, names=None, previous=None):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
      errors.append(f"nonfinite derived diagnostic: {label}")
      return
    if not len(values):
      metrics[label] = {"evaluated_samples": 0, "scope": "no positive-time intervals"}
      return
    values = values.reshape((len(values), -1))
    names = names or [str(i) for i in range(values.shape[1])]
    magnitude = np.abs(values)
    flat = int(np.argmax(magnitude))
    row, col = np.unravel_index(flat, magnitude.shape)
    hits = np.argwhere(magnitude > threshold)
    top = sorted(hits.tolist(), key=lambda rc: magnitude[tuple(rc)], reverse=True)[:20]

    def event(i, j):
      item = {
        "time_s": float(times[i]),
        "phase": str(phases[i]),
        "component": names[j],
        "value": float(values[i, j]),
      }
      if previous is not None:
        item.update(
          interval_start_s=float(previous[0][i]),
          previous_phase=str(previous[1][i]),
          phase_transition=bool(previous[1][i] != phases[i]),
        )
      return item

    metrics[label] = {
      "evaluated_samples": len(values),
      "maximum_absolute_value": float(magnitude[row, col]),
      "maximum_event": event(row, col),
      "absolute_p50_p95_p99": np.percentile(magnitude, [50, 95, 99]).tolist(),
      "per_component_maximum_absolute": dict(
        zip(names, magnitude.max(axis=0).tolist(), strict=True)
      ),
      "screening_threshold": float(threshold),
      "threshold_kind": "heuristic_review_only",
      "candidate_count": len(hits),
      "candidate_interval_count": len(set(hits[:, 0].tolist())) if len(hits) else 0,
      "top_candidates": [event(i, j) for i, j in top],
      "candidates_truncated": len(hits) > 20,
    }

  def motion(label, position, rotation, times, phases, names):
    delta = np.diff(times)
    if np.any(delta < 0):
      errors.append(f"backwards pose clock: {label}")
    indices = np.flatnonzero(delta > 0)
    report["streams"][label + "/clock"] = {
      "duplicate_cached_epochs_skipped": int((delta == 0).sum()),
      "positive_intervals": len(indices),
    }
    dt = delta[indices]
    p = position.reshape((len(times), -1, 3))
    linear = np.diff(p, axis=0)[indices] / dt[:, None, None]
    context = (times[indices], phases[indices])
    measure(
      label + "/linear_speed_m_s",
      np.linalg.norm(linear, axis=-1),
      times[indices + 1],
      phases[indices + 1],
      1.0,
      names,
      context,
    )
    r = rotation.reshape((len(times), -1, 3, 3))
    relative = r[indices + 1] @ np.swapaxes(r[indices], -1, -2)
    angles = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    measure(
      label + "/angular_speed_rad_s",
      angles / dt[:, None],
      times[indices + 1],
      phases[indices + 1],
      12.0,
      names,
      context,
    )
    if len(indices) > 1:
      midtime = (times[indices] + times[indices + 1]) / 2
      acceleration = np.diff(linear, axis=0) / np.diff(midtime)[:, None, None]
      measure(
        label + "/linear_acceleration_m_s2",
        np.linalg.norm(acceleration, axis=-1),
        times[indices[1:] + 1],
        phases[indices[1:] + 1],
        100.0,
        names,
      )
    return indices, dt

  time = array("state/timestamp")
  if time is None or time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
    errors.append("state timestamps require at least two strictly increasing samples")
    return report
  n, dt = len(time), np.diff(time)
  phases = np.asarray([_text(v) for v in file["commands/phase"][:]])
  if phases.shape != (n,):
    errors.append("phase stream does not align with state")
    return report
  names = {
    key: [_text(v) for v in file[f"state/{key}"][:]]
    for key in ("joint_names", "full_qpos_names", "full_qvel_names")
  }
  for label, values in names.items():
    if len(values) != len(set(values)):
      errors.append(f"duplicate component names: {label}")
  q = array("state/qpos", (n, len(names["full_qpos_names"])))
  v = array("state/qvel", (n, len(names["full_qvel_names"])))
  joint = array("state/robot_joint_position", (n, len(names["joint_names"])))
  velocity = array("state/robot_joint_velocity", (n, len(names["joint_names"])))
  array("state/robot_joint_effort", (n, len(names["joint_names"])))
  if q is not None and v is not None:
    scalar = [
      name
      for name in names["full_qpos_names"]
      if "/" not in name and name in names["full_qvel_names"]
    ]
    iq, iv = (
      [names["full_qpos_names"].index(k) for k in scalar],
      [names["full_qvel_names"].index(k) for k in scalar],
    )
    context = (time[:-1], phases[:-1])
    measure(
      "scalar_joint/observed_speed_rad_s",
      np.diff(q[:, iq], axis=0) / dt[:, None],
      time[1:],
      phases[1:],
      15.0,
      scalar,
      context,
    )
    measure("scalar_joint/qvel_rad_s", v[:, iv], time, phases, 15.0, scalar)
    residual = np.diff(q[:, iq], axis=0) - 0.5 * (v[:-1, iv] + v[1:, iv]) * dt[:, None]
    measure(
      "scalar_joint/trapezoidal_position_residual_rad",
      residual,
      time[1:],
      phases[1:],
      0.01,
      scalar,
      context,
    )
    measure(
      "scalar_joint/observed_acceleration_rad_s2",
      np.diff(v[:, iv], axis=0) / dt[:, None],
      time[1:],
      phases[1:],
      1500.0,
      scalar,
      context,
    )
    if joint is not None and velocity is not None:
      if all(k in scalar for k in names["joint_names"]):
        jp = [names["full_qpos_names"].index(k) for k in names["joint_names"]]
        jv = [names["full_qvel_names"].index(k) for k in names["joint_names"]]
        mismatch = max(
          float(np.max(np.abs(joint - q[:, jp]))),
          float(np.max(np.abs(velocity - v[:, jv]))),
        )
        report["published_joint_vs_full_state_maximum_error"] = mismatch
        if mismatch > 1e-12:
          errors.append(
            "published robot joints disagree with same-frame full qpos/qvel"
          )
      else:
        errors.append("published robot joint names cannot be matched to scalar state")
    for name in names["full_qpos_names"]:
      if name.endswith("/qw"):
        prefix = name[:-3]
        ids = [
          names["full_qpos_names"].index(prefix + "/" + k)
          for k in ("qw", "qx", "qy", "qz")
        ]
        norm_error = float(np.max(np.abs(np.linalg.norm(q[:, ids], axis=1) - 1)))
        report["streams"][prefix + "/full_state_quaternion"] = {
          "maximum_unit_norm_error": norm_error,
          "quaternion_sign_flips_are_not_discontinuities": True,
        }
        if norm_error > 1e-6:
          errors.append(f"nonunit full-state quaternion: {prefix}")
        position_keys = [prefix + "/" + k for k in ("x", "y", "z")]
        velocity_keys = [prefix + "/" + k for k in ("vx", "vy", "vz")]
        if all(k in names["full_qpos_names"] for k in position_keys) and all(
          k in names["full_qvel_names"] for k in velocity_keys
        ):
          xyz = q[:, [names["full_qpos_names"].index(k) for k in position_keys]]
          linear_v = v[:, [names["full_qvel_names"].index(k) for k in velocity_keys]]
          measure(
            prefix + "/full_qpos_translation_speed_m_s",
            np.linalg.norm(np.diff(xyz, axis=0) / dt[:, None], axis=1),
            time[1:],
            phases[1:],
            1.0,
            [prefix],
            context,
          )
          measure(
            prefix + "/full_qvel_translation_residual_m",
            np.linalg.norm(
              np.diff(xyz, axis=0) - 0.5 * (linear_v[:-1] + linear_v[1:]) * dt[:, None],
              axis=1,
            ),
            time[1:],
            phases[1:],
            0.002,
            [prefix],
            context,
          )
          if norm_error <= 1e-6:
            measure(
              prefix + "/full_qpos_rotation_speed_rad_s",
              np.linalg.norm(_world_rotation_increment(q[:, ids]), axis=1) / dt,
              time[1:],
              phases[1:],
              12.0,
              [prefix],
              context,
            )
  object_time = array("tactile_contact_force/timestamp", (n,))
  for name in file["objects"]:
    pose = array(f"objects/{name}/pose_wxyz", (n, 7))
    twist = array(f"objects/{name}/twist_linear_angular", (n, 6))
    if pose is None or twist is None or object_time is None:
      continue
    norm_error = float(np.max(np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1)))
    if norm_error > 1e-6:
      errors.append(f"nonunit object quaternion: {name}")
      continue
    label = f"objects/{name}"
    indices, durations = motion(
      label, pose[:, :3], _quaternion_rotation(pose[:, 3:]), object_time, phases, [name]
    )
    measure(
      label + "/recorded_linear_speed_m_s",
      np.linalg.norm(twist[:, :3], axis=1),
      object_time,
      phases,
      1.0,
      [name],
    )
    measure(
      label + "/recorded_angular_speed_rad_s",
      np.linalg.norm(twist[:, 3:], axis=1),
      object_time,
      phases,
      12.0,
      [name],
    )
    residual = (
      np.diff(pose[:, :3], axis=0)[indices]
      - 0.5 * (twist[indices, :3] + twist[indices + 1, :3]) * durations[:, None]
    )
    measure(
      label + "/trapezoidal_position_residual_m",
      np.linalg.norm(residual, axis=1),
      object_time[indices + 1],
      phases[indices + 1],
      0.002,
      [name],
    )
    rotation_residual = (
      _world_rotation_increment(pose[:, 3:])[indices]
      - 0.5 * (twist[indices, 3:] + twist[indices + 1, 3:]) * durations[:, None]
    )
    measure(
      label + "/approximate_world_angular_integral_residual_rad",
      np.linalg.norm(rotation_residual, axis=1),
      object_time[indices + 1],
      phases[indices + 1],
      0.02,
      [name],
    )
  for camera, group in file["cameras"].items():
    prefix = f"cameras/{camera}"
    times = array(prefix + "/pose_timestamp")
    if times is None or times.ndim != 1:
      continue
    state_indices = np.asarray(group["state_index"][:])
    if (
      state_indices.shape != times.shape
      or state_indices.dtype.kind not in "iu"
      or np.any(state_indices < 0)
      or np.any(state_indices >= n)
    ):
      errors.append(f"invalid camera/state phase mapping: {camera}")
      continue
    for kind, shape, labels in (
      ("wrist", (2,), ["left", "right"]),
      (
        "fingertip",
        (2, 5),
        [
          f"{s}/{f}"
          for s in ("left", "right")
          for f in ("thumb", "index", "middle", "ring", "little")
        ],
      ),
    ):
      label = prefix + "/world_from_" + kind
      transform = array(label, (len(times), *shape, 4, 4))
      if transform is None:
        continue
      r = transform[..., :3, :3]
      residual = max(
        float(np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3)))),
        float(np.max(np.abs(np.linalg.det(r) - 1))),
        float(np.max(np.abs(transform[..., 3, :] - [0, 0, 0, 1]))),
      )
      report["streams"][label]["maximum_se3_constraint_error"] = residual
      if residual > 1e-6:
        errors.append(f"invalid wrist/finger SE3: {label}")
      else:
        motion(label, transform[..., :3, 3], r, times, phases[state_indices], labels)
  metadata = json.loads(_text(file.attrs["metadata_json"]))
  noise = metadata.get("precontact_noise", {})
  if noise and "commands/actuator_control" in file:
    control_names = [_text(v) for v in file["commands/actuator_names"][:]]
    command = array("commands/actuator_control", (n, len(control_names)))
    limit = np.asarray(noise.get("right_arm_ctrlrange_rad", []), dtype=float)
    arm_names = noise.get("right_arm_actuator_names", [])
    if (
      command is not None
      and limit.shape == (7, 2)
      and len(arm_names) == 7
      and all(k in control_names for k in arm_names)
    ):
      controls = command[:, [control_names.index(k) for k in arm_names]]
      excess = np.maximum(limit[:, 0] - controls, controls - limit[:, 1])
      report["metadata_constraints"]["actual_right_arm_command_range"] = {
        "status": "pass" if np.max(excess) <= 1e-12 else "warning",
        "maximum_excess_rad": float(max(0, np.max(excess))),
        "metadata_path": "precontact_noise.right_arm_ctrlrange_rad",
        "scope": "position-servo command range, not physical joint qvel",
      }
      measure(
        "right_arm/command_range_excess_rad",
        np.maximum(excess, 0),
        time,
        phases,
        1e-12,
        arm_names,
      )
      metrics["right_arm/command_range_excess_rad"]["threshold_kind"] = (
        "explicit_recorded_command_limit"
      )
      final_noise = json.loads(_text(file.attrs["outcome_json"])).get(
        "precontact_noise", {}
      )
      stop = final_noise.get("contact_detected_time_s")
      speed_limit = np.asarray(
        noise.get("right_arm_max_velocity_rad_s", []), dtype=float
      )
      if (
        isinstance(stop, (int, float))
        and speed_limit.shape == (7,)
        and np.isfinite(speed_limit).all()
        and np.all(speed_limit > 0)
      ):
        selected = np.flatnonzero((np.arange(n - 1) > 0) & (time[1:] <= stop))
        excess_speed = np.maximum(
          np.abs(np.diff(controls, axis=0)[selected] / dt[selected, None])
          - speed_limit,
          0,
        )
        measure(
          "right_arm/precontact_command_slew_excess_rad_s",
          excess_speed,
          time[selected + 1],
          phases[selected + 1],
          1e-9,
          arm_names,
          (time[selected], phases[selected]),
        )
        metrics["right_arm/precontact_command_slew_excess_rad_s"]["threshold_kind"] = (
          "explicit_recorded_precontact_command_slew_limit"
        )
        report["metadata_constraints"]["precontact_command_slew"] = {
          "status": "pass"
          if len(selected) and np.max(excess_speed) <= 1e-9
          else "warning"
          if len(selected)
          else "not_evaluated",
          "sampled_intervals": len(selected),
          "scope": "recorded command intervals wholly before first contact; reset ctrl is unexecuted and first interval omitted; no claim about unsaved physical substeps",
        }
    report["metadata_constraints"]["physical_joint_speed"] = {
      "status": "not_asserted",
      "reason": "right_arm_max_velocity_rad_s is command slew, not an actual joint velocity bound; real qvel screening remains heuristic",
    }
  report["candidate_count"] = sum(
    value.get("candidate_count", 0) for value in metrics.values()
  )
  report["review_required"] = report["candidate_count"] > 0 or bool(errors)
  if report["candidate_count"]:
    report["warnings"].append(
      "Heuristic motion/integration candidates need phase/contact-aware review; they do not establish corrupted samples or justify deletion."
    )
  report["integration_scope"] = (
    "trapezoidal integration of decimated velocity endpoints only; angular object comparison is first-order world-frame approximation; no force, contact dynamics, free-qvel frame reconstruction, or FK replay"
  )
  report["metadata_constraints"]["physical_joint_position_limits"] = {
    "status": "not_asserted",
    "reason": "no separate physical joint-limit table is archived; actuator/effective command bounds are not silently applied as hard actual-state constraints",
  }
  report["clock_scope"] = (
    "qpos/qvel: post-integration state clock; object FK/twist: solver-cache clock; wrist/finger FK: each camera pose clock; repeated initial cached epoch excluded from derivatives"
  )
  report["valid"] = not errors
  report["valid_semantics"] = (
    "structural/finite/rigid-transform and redundant-stream consistency; heuristic candidates do not invalidate data"
  )
  return report


def check_trajectory(file):
  """Return a JSON report even when a required HDF5 stream is malformed/missing."""
  try:
    result = _check_trajectory(file)
    json.dumps(result, allow_nan=False)
    return result
  except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
    return {
      "schema_version": "poker-sampled-trajectory-cleaning-v1",
      "valid": False,
      "review_required": True,
      "errors": [
        f"trajectory inspection could not complete: {type(error).__name__}: {error}"
      ],
      "simulator_loaded": False,
      "dynamics_reconstructed": False,
      "inputs_modified": False,
    }

