"""Negative controls for the independent USB mechanics acceptance gates."""

import copy

import numpy as np
import pytest
from kaihand_tactile_env.tasks.usb_insert.mechanics_audit import (
  ContactAudit,
  balance_residual,
)


def test_newton_euler_detects_missing_lever_arm_and_contact_torque():
  # A horizontal 2 N push 10 mm above COM and a 3 mN m contact couple.
  # I_y=0.002 kg m², so alpha_y=(.02+.003)/.002=11.5 rad/s².
  force = np.array([2.0, 0.0, 9.81])
  torque = np.array([0.0, 0.023, 0.0])
  parameters = (
    1.0,
    np.array([0.0, 0.0, -9.81]),
    np.array([2.0, 0.0, 0.0]),
    np.diag([0.001, 0.002, 0.003]),
    np.zeros(3),
    np.array([0.0, 11.5, 0.0]),
  )
  rf, rt = balance_residual(force, torque, *parameters)
  np.testing.assert_allclose(rf, 0, atol=1e-14)
  np.testing.assert_allclose(rt, 0, atol=1e-14)
  assert np.linalg.norm(
    balance_residual(force, np.array([0.0, 0.003, 0.0]), *parameters)[1]
  ) == pytest.approx(0.02)
  assert np.linalg.norm(
    balance_residual(force, np.array([0.0, 0.02, 0.0]), *parameters)[1]
  ) == pytest.approx(0.003)
  assert np.linalg.norm(balance_residual(-force, -torque, *parameters)[0]) > 1


def test_rotating_body_requires_gyroscopic_torque():
  # omega=(1,2,0), I omega=(1,4,0); cross product is (0,0,2).
  args = (
    1.0,
    np.zeros(3),
    np.zeros(3),
    np.diag([1.0, 2.0, 3.0]),
    np.array([1.0, 2.0, 0.0]),
    np.zeros(3),
  )
  np.testing.assert_allclose(
    balance_residual(np.zeros(3), np.array([0.0, 0.0, 2.0]), *args)[1], 0
  )
  assert np.linalg.norm(balance_residual(np.zeros(3), np.zeros(3), *args)[1]) == 2


def clean_audit():
  audit = ContactAudit.__new__(ContactAudit)
  audit.rows = []
  for k in range(300):
    phase = ("insert", "bottom_out", "unload", "release")[min(k // 75, 3)]
    audit.rows.append(
      {
        "time": k * 0.002,
        "phase": phase,
        "force_error": 0.0,
        "torque_error": 0.0,
        "normal": np.full(2, 0.3 if k >= 150 else 2.0),
        "tangent": np.full(2, 0.05 if k >= 150 else 0.2),
        "count": np.ones(2),
        "slip": np.full(2, 0.1),
        "cone": np.full(2, 0.2),
        "torque_ratio": np.full(2, 0.1),
        "relative_position": np.zeros(3),
        "relative_rotation": np.eye(3),
      }
    )
  return audit


def test_stable_asymmetric_loads_and_declining_grip_are_allowed():
  audit = clean_audit()
  for k, row in enumerate(audit.rows[:150]):
    row["normal"] = np.array([2 - k / 300, 1.5 - k / 300])
    row["tangent"] = np.array([0.2, 0.1])
  assert audit.summarize()[0]["passed"]


@pytest.mark.parametrize(
  "key,value,gate",
  [
    ("force_error", 0.01, "force_balance"),
    ("torque_error", 0.001, "torque_balance"),
    ("count", np.array([1.0, 0.0]), "continuous_bilateral_contact"),
    ("normal", np.array([2.0, 0.01]), "adequate_grip"),
    ("slip", np.array([11.0, 0.0]), "hold_slip_peak_mm_s"),
    ("cone", np.array([1.1, 0.2]), "friction_cone_utilization"),
    ("relative_position", np.array([0.003, 0.0, 0.0]), "hold_relative_translation_mm"),
    ("tangent", np.array([0.8, 0.2]), "hold_tangent_step_n"),
  ],
)
def test_negative_controls_reject_bad_contact_data(key, value, gate):
  audit = clean_audit()
  audit.rows[100][key] = copy.deepcopy(value)
  report, _ = audit.summarize()
  assert not report["passed"] and not report["gates"][gate]


def test_sustained_slip_is_rejected_even_without_a_fast_spike():
  audit = clean_audit()
  for row in audit.rows[:150]:
    row["slip"][:] = 5
  report, _ = audit.summarize()
  assert not report["gates"]["hold_slip_path_mm"]
  assert not report["gates"]["hold_slip_100ms_mean_mm_s"]
  assert report["gates"]["hold_slip_peak_mm_s"]


def test_missing_sample_or_phase_cannot_pass():
  audit = clean_audit()
  del audit.rows[10]
  with pytest.raises(ValueError, match="consecutive"):
    audit.summarize()
  audit = clean_audit()
  audit.rows = audit.rows[:100]
  assert not audit.summarize()[0]["passed"]
