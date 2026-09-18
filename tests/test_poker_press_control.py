from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.config import (
  _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES,
  _PRESS_FORCE_MEAN_TOLERANCE_N,
  DEFAULT_PRESS_FORCE_PER_FINGER_N,
)
from kaihand_tactile_env.tasks.poker_draw.press_control import (
  FourFingerForceController,
  PressForceMonitor,
  PressForceSummary,
)
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawExecutor


def _controller() -> FourFingerForceController:
  return FourFingerForceController(
    4,
    target_force_n=0.35,
    timestep=0.002,
    update_period_s=0.010,
    filter_time_constant_s=0.020,
    integral_gain_rad_per_n_s=np.deg2rad(7.0),
    maximum_offset_rad=np.deg2rad(5.0),
    maximum_offset_rate_rad_s=np.deg2rad(4.0),
    contact_force_n=0.0875,
    contact_recovery_rate_rad_s=np.deg2rad(1.2),
    force_deadband_n=0.025,
  )


def _qualification_fixture() -> tuple[PokerDrawExecutor, PressForceSummary]:
  executor = object.__new__(PokerDrawExecutor)
  executor.press_force_per_finger_n = DEFAULT_PRESS_FORCE_PER_FINGER_N
  executor._maximum_slide_fingertip_plane_angle_degrees = (
    _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES
  )
  target = DEFAULT_PRESS_FORCE_PER_FINGER_N
  tolerance = _PRESS_FORCE_MEAN_TOLERANCE_N
  summary = PressForceSummary(
    sample_count=500,
    force_means_n=(
      np.nextafter(target - tolerance, target),
      target,
      target + 0.01,
      np.nextafter(target + tolerance, target),
    ),
    force_peaks_n=(target,) * 4,
    contact_fractions=(1.0,) * 4,
    target_band_fractions=(0.95, 0.96, 0.97, 0.98),
    maximum_contact_gaps_s=(0.0,) * 4,
    four_finger_contact_fraction=1.0,
    four_finger_target_band_fraction=0.90,
  )
  return executor, summary


def test_force_controller_builds_and_releases_independent_offsets() -> None:
  controller = _controller()
  controller.reset((0.0, 0.20, 0.35, 0.70))

  updated = False
  for _ in range(5):
    offsets, updated = controller.observe((0.0, 0.20, 0.35, 0.70))

  assert updated
  assert offsets[0] > offsets[1] > offsets[2]
  assert offsets[2] == pytest.approx(0.0)
  assert offsets[3] < 0.0

  low_force_offset = float(offsets[0])
  for _ in range(25):
    offsets, _ = controller.observe((1.0, 0.20, 0.35, 0.70))
  assert offsets[0] < low_force_offset
  assert np.all(np.abs(offsets) <= np.deg2rad(5.0))


def test_press_monitor_reports_per_finger_force_and_longest_gap() -> None:
  monitor = PressForceMonitor(
    4,
    target_force_n=0.4,
    timestep=0.01,
    contact_force_ratio=0.25,
    target_tolerance_n=0.1,
  )
  for forces in (
    (0.4, 0.4, 0.4, 0.4),
    (0.4, 0.0, 0.4, 0.4),
    (0.4, 0.0, 0.4, 0.4),
    (0.4, 0.4, 0.4, 0.4),
  ):
    monitor.observe(forces)

  summary = monitor.summary()
  assert summary.sample_count == 4
  assert summary.force_means_n == pytest.approx((0.4, 0.2, 0.4, 0.4))
  assert summary.contact_fractions == pytest.approx((1.0, 0.5, 1.0, 1.0))
  assert summary.maximum_contact_gaps_s == pytest.approx((0.0, 0.02, 0.0, 0.0))
  assert summary.four_finger_contact_fraction == pytest.approx(0.5)
  assert summary.four_finger_target_band_fraction == pytest.approx(0.5)


def test_production_press_uses_force_target_and_legacy_preload_is_explicit() -> None:
  assert DEFAULT_PRESS_FORCE_PER_FINGER_N == pytest.approx(0.35)
  constructor = inspect.signature(PokerDrawExecutor).parameters
  press = inspect.signature(PokerDrawExecutor._press_until_four_contacts).parameters

  assert constructor["press_force_per_finger_n"].default is None
  assert press["distal_offset_degrees"].default is None


def test_slide_force_quality_accepts_exact_strict_boundaries() -> None:
  executor, summary = _qualification_fixture()

  assert executor._slide_press_control_qualified(summary)


@pytest.mark.parametrize(
  "summary",
  (
    # One 2 ms low-force sample out of 500 violates both uninterrupted
    # contact and the zero-gap requirement.
    replace(
      _qualification_fixture()[1],
      contact_fractions=(0.998, 1.0, 1.0, 1.0),
      maximum_contact_gaps_s=(0.002, 0.0, 0.0, 0.0),
      four_finger_contact_fraction=0.998,
    ),
    replace(
      _qualification_fixture()[1],
      target_band_fractions=(0.949, 0.96, 0.97, 0.98),
    ),
    replace(
      _qualification_fixture()[1],
      force_means_n=(0.314, 0.35, 0.35, 0.35),
    ),
  ),
  ids=("one-physics-step-gap", "one-finger-band-shortfall", "mean-off-target"),
)
def test_slide_force_quality_rejects_pressure_failures(
  summary: PressForceSummary,
) -> None:
  executor, _ = _qualification_fixture()

  assert not executor._slide_press_control_qualified(summary)


def test_slide_force_quality_rejects_excessive_fingertip_angle() -> None:
  executor, summary = _qualification_fixture()
  executor._maximum_slide_fingertip_plane_angle_degrees = (
    _FLAT_DRAW_MAXIMUM_FINGERTIP_PLANE_ANGLE_DEGREES + 0.01
  )

  assert not executor._slide_press_control_qualified(summary)


@pytest.mark.parametrize(
  "forces",
  (
    (0.0, 0.0, 0.0),
    (0.0, -0.01, 0.0, 0.0),
    (0.0, np.nan, 0.0, 0.0),
  ),
  ids=("wrong-count", "negative", "non-finite"),
)
def test_force_controller_rejects_invalid_measurements(
  forces: tuple[float, ...],
) -> None:
  with pytest.raises(ValueError):
    _controller().observe(forces)
