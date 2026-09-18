"""Independent synthetic HDF5 audit checks; never load a robot or renderer."""

from __future__ import annotations

import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_source_audit as audit


def _synthetic_source(tmp_path, *, sigma=None, initial_contact=False):
  sigma = float(np.deg2rad(0.03)) if sigma is None else sigma
  settings = {
    "schema_version": "poker-precontact-arm-noise-settings-v1",
    "std_rad": sigma,
    "correlation_time_s": 0.15,
    "maximum_offset_rad": 3 * sigma,
    "maximum_offset_rate_rad_s": float(np.deg2rad(0.5)),
    "distribution": "bounded_ornstein_uhlenbeck_gaussian",
    "tactile_source": "solver_contact_proxy_v1",
    "contact_link_names": list(audit._PRECONTACT_LINKS),
    "contact_stop_rule": "first_any_right_fingertip_contact",
    "actuator_scope": "right_arm_seven_position_servo_controls",
    "permanent_until_reset": True,
  }
  limits = np.array(
    [
      [-3.1067, 3.1067],
      [-2.09439510239, 2.09439510239],
      [-3.1067, 3.1067],
      [-2.5307, 1.0472],
      [-3.1067, 3.1067],
      [-1.04719755120, 1.04719755120],
      [-1.57079632679, 1.57079632679],
    ]
  )
  seed, n, dt, first = 23, 20, 0.002, 7
  contact = np.zeros((n, 5), bool)
  contact[first, 1] = True
  contact[12:, 2] = True
  force = contact.astype(float) * 0.02
  latch = np.zeros(n, bool)
  latch[first + 1 :] = True
  if initial_contact:
    latch[:] = True
  nominal, actual, offset = (np.zeros((n, 7)) for _ in range(3))
  rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5052434E]))
  ou, previous, previous_actual = np.zeros(7), np.zeros(7), np.zeros(7)
  draws = 0
  for i in range(n):
    if not latch[i] and sigma:
      ou = np.exp(-dt / 0.15) * ou + sigma * np.sqrt(
        -np.expm1(-2 * dt / 0.15)
      ) * rng.normal(size=7)
      draws += 1
      maximum, rate = 3 * sigma, np.deg2rad(0.5) * dt
      lower = np.maximum.reduce(
        (
          np.full(7, -maximum),
          previous - rate,
          limits[:, 0],
          previous_actual - 3.1416 * dt,
        )
      )
      upper = np.minimum.reduce(
        (
          np.full(7, maximum),
          previous + rate,
          limits[:, 1],
          previous_actual + 3.1416 * dt,
        )
      )
      offset[i] = np.clip(np.clip(ou, -maximum, maximum), lower, upper)
      previous = offset[i]
    actual[i] = nominal[i] + offset[i]
    previous_actual = actual[i]
  initial_mask = np.array([initial_contact, False, False, False, False])
  initial_force = initial_mask.astype(float) * 0.01
  common = {
    "schema_version": "poker-precontact-arm-noise-v1",
    "configured": True,
    "settings": settings,
    "seed": seed,
    "stream_tag": 0x5052434E,
    "timestep_s": dt,
    "right_arm_actuator_names": audit._PRECONTACT_ACTUATORS,
    "right_arm_ctrlrange_rad": limits.tolist(),
    "right_arm_effective_control_bounds_rad": limits.tolist(),
    "right_arm_max_velocity_rad_s": [3.1416] * 7,
    "initial_arm_command_rad": [0.0] * 7,
    "initial_contact": initial_mask.tolist(),
    "initial_normal_force_n": initial_force.tolist(),
    "postcontact_random_sample_count": 0,
    "failed": False,
    "trace_time_semantics": "time_s is control start; tactile is completed-step solver cache",
    "control_mode_names": {"0": "joint_position_servo", "1": "bounded_cartesian"},
    "noise_offset_semantics": "actual additive random input; zero after latch; nominal may retain servo recovery",
  }
  initial = {
    **common,
    "contact_latched": initial_contact,
    "contact_detected_time_s": 0.0 if initial_contact else None,
    "contact_tactile_time_s": 0.0 if initial_contact else None,
    "stop_reason": "first_right_fingertip_contact" if initial_contact else None,
    "random_sample_count": 0,
    "physics_step_count": 0,
    "postcontact_noise_disabled": initial_contact,
    "command_handoff_count": 0,
  }
  final = {
    **common,
    "contact_latched": True,
    "contact_detected_time_s": 0.0 if initial_contact else (first + 1) * dt,
    "contact_tactile_time_s": 0.0 if initial_contact else first * dt,
    "stop_reason": "first_right_fingertip_contact",
    "random_sample_count": draws,
    "physics_step_count": n,
    "postcontact_noise_disabled": True,
    "command_handoff_count": int(not initial_contact and sigma > 0),
  }
  metadata = {
    "preset": audit.PRECONTACT_PRESET,
    "seed": seed,
    "preset_settings": {"action_noise": settings},
    "precontact_noise": initial,
  }
  outcome = {"success": True, "precontact_noise": final}
  source = tmp_path / "synthetic.h5"
  with h5py.File(source, "w") as file:
    file.attrs.update(
      metadata_json=json.dumps(metadata),
      outcome_json=json.dumps(outcome),
      physics_hz=500,
      control_hz=100,
    )
    group = file.create_group("control/precontact_noise")
    group.attrs["metadata_json"] = json.dumps(final)
    for name, values in {
      "time_s": np.arange(n) * dt,
      "tactile_time_s": np.arange(n) * dt,
      "contact": contact,
      "normal_force_n": force,
      "latched_before_step": latch,
      "noise_offset_rad": offset,
      "nominal_ctrl_rad": nominal,
      "actual_ctrl_rad": actual,
      "control_mode": (np.arange(n) >= 12).astype(np.int8),
    }.items():
      group.create_dataset(name, data=values)
    selected = np.arange(4, n, 5)
    file.create_dataset("state/timestamp", data=np.r_[0, (selected + 1) * dt])
    proxy = file.create_group("tactile_proxy")
    proxy.attrs["source"] = "solver_contact_proxy_v1"
    proxy.create_dataset(
      "link_names", data=audit._PRECONTACT_LINKS, dtype=h5py.string_dtype()
    )
    proxy.create_dataset("contact", data=np.vstack((initial_mask, contact[selected])))
    proxy.create_dataset(
      "normal_force", data=np.vstack((initial_force, force[selected]))
    )
    file.create_dataset(
      "commands/actuator_names",
      data=audit._PRECONTACT_ACTUATORS,
      dtype=h5py.string_dtype(),
    )
    file.create_dataset(
      "commands/actuator_control", data=np.vstack((np.zeros(7), actual[selected]))
    )
  return source


@pytest.mark.parametrize("initial_contact", [False, True])
@pytest.mark.parametrize(
  "sigma", [0.0, float(np.deg2rad(0.03)), float(np.deg2rad(0.05))]
)
def test_valid_seeded_trace_is_accepted_without_simulation(
  tmp_path, initial_contact, sigma
):
  source = _synthetic_source(tmp_path, sigma=sigma, initial_contact=initial_contact)
  with h5py.File(source, "r") as file:
    report = audit.audit_precontact_noise(file)
  assert report["valid"], report
  assert report["physics_step_count"] == 20
  assert report["state_samples_matched"] == 4
  assert report["random_sample_count"] == (8 if sigma and not initial_contact else 0)
  assert report["seeded_noise_projection_verified"]
  assert report["source_tactile_contact_and_force_matched"]


@pytest.mark.parametrize(
  "field",
  [
    "time_s",
    "tactile_time_s",
    "contact",
    "normal_force_n",
    "latched_before_step",
    "noise_offset_rad",
    "nominal_ctrl_rad",
    "actual_ctrl_rad",
    "control_mode",
  ],
)
def test_missing_trace_column_fails_closed(tmp_path, field):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    del file[f"control/precontact_noise/{field}"]
    report = audit.audit_precontact_noise(file)
  assert not report["valid"]


@pytest.mark.parametrize(
  "field,index,value",
  [
    ("time_s", 2, 0.0041),
    ("tactile_time_s", 2, 0.002),
    ("time_s", 1, float("nan")),
    ("normal_force_n", (0, 0), 0.1),
    ("normal_force_n", (7, 1), -0.1),
    ("noise_offset_rad", (9, 0), 0.0001),
    ("noise_offset_rad", (0, 0), 0.001),
    ("actual_ctrl_rad", (0, 0), 4),
    ("nominal_ctrl_rad", (1, 0), float("nan")),
    ("latched_before_step", 7, True),
    ("latched_before_step", 9, False),
    ("control_mode", 0, 1),
    ("control_mode", 10, 2),
  ],
)
def test_tampered_trace_fails_closed(tmp_path, field, index, value):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    file[f"control/precontact_noise/{field}"][index] = value
    report = audit.audit_precontact_noise(file)
  assert not report["valid"], report


@pytest.mark.parametrize("target", ["initial", "final"])
@pytest.mark.parametrize(
  "field,value",
  [
    ("seed", 24),
    ("stream_tag", 1),
    ("timestep_s", 0.001),
    ("configured", False),
    ("contact_latched", False),
    ("physics_step_count", 3),
    ("random_sample_count", 2),
    ("postcontact_random_sample_count", 1),
    ("command_handoff_count", 2),
    ("failed", True),
    ("right_arm_max_velocity_rad_s", [4.0] * 7),
    ("right_arm_ctrlrange_rad", [[-4, 4]] * 7),
  ],
)
def test_metadata_must_agree_with_trace(tmp_path, target, field, value):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    name = "metadata_json" if target == "initial" else "outcome_json"
    payload = json.loads(file.attrs[name])
    if payload["precontact_noise"][field] == value:
      assert isinstance(value, bool)
      value = not value
    payload["precontact_noise"][field] = value
    file.attrs[name] = json.dumps(payload)
    if target == "final":
      file["control/precontact_noise"].attrs["metadata_json"] = json.dumps(
        payload["precontact_noise"]
      )
    report = audit.audit_precontact_noise(file)
  assert not report["valid"], report


@pytest.mark.parametrize(
  "dataset,index,value",
  [
    ("state/timestamp", 4, 0.042),
    ("tactile_proxy/contact", (2, 1), True),
    ("tactile_proxy/normal_force", (3, 2), 0.1),
    ("commands/actuator_control", (1, 0), 0.1),
  ],
)
def test_100hz_original_evidence_must_match_500hz_trace(
  tmp_path, dataset, index, value
):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    file[dataset][index] = value
    report = audit.audit_precontact_noise(file)
  assert not report["valid"]


def test_seed_change_with_matching_metadata_still_fails_independent_rng(tmp_path):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    outcome = json.loads(file.attrs["outcome_json"])
    metadata["seed"] = metadata["precontact_noise"]["seed"] = 24
    outcome["precontact_noise"]["seed"] = 24
    file["control/precontact_noise"].attrs["metadata_json"] = json.dumps(
      outcome["precontact_noise"]
    )
    report = audit.audit_precontact_noise(file, metadata, outcome)
  assert not report["valid"]
  assert "seeded OU" in report["errors"][0]


def test_nominal_servo_recovery_after_latch_is_not_mislabeled_as_new_noise(tmp_path):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    group = file["control/precontact_noise"]
    nominal = group["nominal_ctrl_rad"][:]
    actual = group["actual_ctrl_rad"][:]
    # The one-time command handoff can retain the last applied target while
    # the ordinary servo recovers, with no further random additive input.
    nominal[8:] = actual[7]
    actual[8:] = nominal[8:]
    group["nominal_ctrl_rad"][:] = nominal
    group["actual_ctrl_rad"][:] = actual
    file["commands/actuator_control"][1:] = actual[np.arange(4, 20, 5)]
    report = audit.audit_precontact_noise(file)
  assert report["valid"], report
  assert report["postcontact_offset_exactly_zero"]


def test_zero_force_contact_still_stops_noise_without_invented_force_threshold(
  tmp_path,
):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    file["control/precontact_noise/normal_force_n"][7, 1] = 0.0
    report = audit.audit_precontact_noise(file)
  assert report["valid"], report
  assert report["contact_detected_time_s"] == 0.016


@pytest.mark.parametrize("preset", ["middle-force-v1", audit.RANDOMIZED_PRESET])
def test_old_presets_cannot_masquerade_as_precontact_sources(tmp_path, preset):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    metadata["preset"] = preset
    report = audit.audit_precontact_noise(file, metadata)
  assert not report["valid"]


def test_precontact_source_list_adds_only_one_module_to_randomized():
  before, scene = audit._known_source_paths(audit.RANDOMIZED_PRESET)
  after, same_scene = audit._known_source_paths(audit.PRECONTACT_PRESET)
  assert set(after) - set(before) == {"poker_draw/precontact_noise.py"}
  assert len(after) == 16
  assert "poker_draw/acceptance.py" in after
  assert scene == same_scene
  assert all(after[name] == path for name, path in before.items())


@pytest.mark.parametrize(
  "key,value",
  [
    ("std_rad", -0.01),
    ("std_rad", np.deg2rad(0.051)),
    ("std_rad", float("nan")),
    ("correlation_time_s", 0.2),
    ("maximum_offset_rad", 0.1),
    ("permanent_until_reset", False),
    ("contact_stop_rule", "normal_force_threshold"),
  ],
)
def test_incompatible_settings_rejected_even_if_all_snapshots_match(
  tmp_path, key, value
):
  source = _synthetic_source(tmp_path)
  with h5py.File(source, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    outcome = json.loads(file.attrs["outcome_json"])
    settings = deepcopy(metadata["preset_settings"]["action_noise"])
    settings[key] = value
    metadata["preset_settings"]["action_noise"] = settings
    metadata["precontact_noise"]["settings"] = settings
    outcome["precontact_noise"]["settings"] = settings
    file["control/precontact_noise"].attrs["metadata_json"] = json.dumps(
      outcome["precontact_noise"]
    )
    report = audit.audit_precontact_noise(file, metadata, outcome)
  assert not report["valid"]
