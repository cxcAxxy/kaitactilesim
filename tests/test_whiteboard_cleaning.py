"""Physical erasure gates independent of the robot controller and renderer."""

import numpy as np
import pytest
from kaihand_tactile_env.tasks.whiteboard_wipe.cleaning import CleaningProgress


@pytest.mark.parametrize(
  ("normal", "tangent", "speed", "power"),
  [
    (0.0, 0.1, 0.06, 0.006),  # The tool moves above the board.
    (1.49, 0.1, 0.06, 0.006),  # A light brush does not meet the load gate.
    (8.01, 0.1, 0.06, 0.006),  # Excessive pressing is not valid wiping.
    (2.0, 0.1, 0.0, 0.006),  # Static loading does not erase ink.
    (2.0, 0.1, 0.0049, 0.006),  # Sub-threshold slip is not a wiping stroke.
    (2.0, 0.1, 0.151, 0.006),  # Fast impact motion is rejected.
    (2.0, 0.0, 0.06, 0.006),  # No real tangential contact load.
    (2.0, 0.0199, 0.06, 0.006),
    (2.0, 0.1, 0.06, 0.0),  # No dissipated friction work.
    (2.0, 0.1, 0.06, -0.006),  # A signed driving power cannot clean.
  ],
)
def test_non_wiping_contacts_leave_all_ink(normal, tangent, speed, power):
  cleaning = CleaningProgress(2)
  for _ in range(100):
    cleaning.update(0.01, normal, [tangent] * 2, [speed] * 2, [power] * 2)
  np.testing.assert_array_equal(cleaning.remaining, 1)
  np.testing.assert_array_equal(cleaning.work_j, 0)
  np.testing.assert_array_equal(cleaning.stroke_m, 0)
  np.testing.assert_array_equal(cleaning.loaded_time_s, 0)


def test_short_high_work_contact_cannot_immediately_erase_ink():
  cleaning = CleaningProgress(1)
  # Even more than the required work cannot replace a sustained stroke.
  cleaning.update(0.01, 8.0, [1.0], [0.15], [0.15])
  assert cleaning.work_j[0] > 0.001
  assert cleaning.remaining[0] == pytest.approx(0.96)


def test_all_three_requirements_limit_progress_independently():
  cleaning = CleaningProgress(3)
  # All samples represent valid loaded motion. Different physical quantities
  # limit the first two patches; only the third completes a sufficiently long
  # stroke with enough work and contact time.
  cleaning.update(
    0.5, 3.0, [0.02, 0.1, 0.1], [0.03, 0.005, 0.06], [0.0006, 0.0005, 0.006]
  )
  np.testing.assert_allclose(cleaning.progress, [0.3, 1 / 12, 1.0])
  np.testing.assert_allclose(cleaning.remaining, [0.7, 11 / 12, 0.0])


def test_sustained_wiping_gradually_erases_only_the_loaded_patch():
  cleaning = CleaningProgress(2)
  remaining = cleaning.remaining
  for _ in range(5):
    cleaning.update(0.05, 2.0, [0.1, 0.0], [0.06, 0.06], [0.006, 0.0])
  np.testing.assert_allclose(cleaning.remaining, [0.5, 1.0])
  # Pausing off the board preserves progress without accumulating extra work.
  cleaning.update(1.0, 0.0, [0.1, 0.1], [0.06, 0.06], [0.006, 0.006])
  np.testing.assert_allclose(cleaning.remaining, [0.5, 1.0])
  for _ in range(6):
    cleaning.update(0.05, 2.0, [0.1, 0.0], [0.06, 0.06], [0.006, 0.0])
  np.testing.assert_array_equal(cleaning.remaining, [0.0, 1.0])
  assert cleaning.remaining is remaining
  completed_work = cleaning.work_j.copy()
  cleaning.update(0.1, 2.0, [0.1, 0.0], [0.06, 0.06], [0.006, 0.0])
  np.testing.assert_array_equal(cleaning.work_j, completed_work)


def test_constant_contact_is_independent_of_time_step_partition():
  coarse = CleaningProgress(1)
  fine = CleaningProgress(1)
  coarse.update(0.2, 2.0, [0.1], [0.06], [0.006])
  for _ in range(100):
    fine.update(0.002, 2.0, [0.1], [0.06], [0.006])
  for field in ("work_j", "stroke_m", "loaded_time_s", "remaining"):
    np.testing.assert_allclose(getattr(coarse, field), getattr(fine, field))


def test_reset_restores_ink_and_clears_physical_history_in_place():
  cleaning = CleaningProgress(1)
  references = [
    cleaning.remaining,
    cleaning.work_j,
    cleaning.stroke_m,
    cleaning.loaded_time_s,
  ]
  cleaning.update(0.5, 2.0, [0.1], [0.06], [0.006])
  cleaning.reset()
  np.testing.assert_array_equal(references[0], 1)
  for array in references[1:]:
    np.testing.assert_array_equal(array, 0)
  assert all(
    before is after
    for before, after in zip(
      references,
      [cleaning.remaining, cleaning.work_j, cleaning.stroke_m, cleaning.loaded_time_s],
      strict=True,
    )
  )


@pytest.mark.parametrize("patch_count", [0, -1, 1.5, True, np.bool_(True)])
def test_patch_count_must_be_positive_integer(patch_count):
  with pytest.raises(ValueError, match="patch_count"):
    CleaningProgress(patch_count)


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("dt", 0),
    ("dt", -0.01),
    ("dt", np.nan),
    ("dt", np.inf),
    ("dt", [0.01]),
    ("normal_force_n", np.inf),
    ("normal_force_n", [2.0]),
    ("per_patch_tangent_load_n", [np.nan, 0.1]),
    ("per_patch_tangent_load_n", [0.1]),
    ("per_patch_speed_m_s", [0.06, np.inf]),
    ("per_patch_speed_m_s", [[0.06, 0.06]]),
    ("per_patch_power_w", [0.006, np.nan]),
    ("per_patch_power_w", 0.006),
  ],
)
def test_invalid_inputs_raise_before_any_accumulation(field, value):
  cleaning = CleaningProgress(2)
  args = {
    "dt": 0.1,
    "normal_force_n": 2.0,
    "per_patch_tangent_load_n": [0.1, 0.1],
    "per_patch_speed_m_s": [0.06, 0.06],
    "per_patch_power_w": [0.006, 0.006],
  }
  cleaning.update(**args)
  before = [
    array.copy()
    for array in (
      cleaning.remaining,
      cleaning.work_j,
      cleaning.stroke_m,
      cleaning.loaded_time_s,
    )
  ]
  args[field] = value
  with pytest.raises(ValueError, match=field):
    cleaning.update(**args)
  for old, new in zip(
    before,
    [cleaning.remaining, cleaning.work_j, cleaning.stroke_m, cleaning.loaded_time_s],
    strict=True,
  ):
    np.testing.assert_array_equal(new, old)
