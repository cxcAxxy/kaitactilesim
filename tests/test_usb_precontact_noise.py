"""Pure array checks for USB motion noise; no simulator or renderer is created."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from kaihand_tactile_env.tasks.usb_insert import precontact_noise as noise


def test_same_seed_reproduces_sequence_and_different_seed_changes_it():
  times = np.linspace(0.0, 1.8, 181)
  first = noise.PrecontactMotionNoise(std_m=0.0004, seed=19)
  repeated = noise.PrecontactMotionNoise(std_m=0.0004, seed=19)
  other = noise.PrecontactMotionNoise(std_m=0.0004, seed=20)
  first_offsets = np.array([first.sample(t) for t in times])
  repeated_offsets = np.array([repeated.sample(t) for t in times])
  other_offsets = np.array([other.sample(t) for t in times])

  np.testing.assert_array_equal(first_offsets, repeated_offsets)
  assert not np.array_equal(first_offsets, other_offsets)
  assert first.draw_count == repeated.draw_count == other.draw_count
  assert first.draw_count > 1
  assert first.report()["seed"] == repeated.report()["seed"] == 19


@pytest.mark.parametrize("std_m,bound_m", [(0.0002, 0.0006), (0.005, 0.002)])
def test_each_axis_is_bounded_and_z_is_always_zero(std_m, bound_m):
  process = noise.PrecontactMotionNoise(std_m=std_m, seed=31)
  offsets = np.array([process.sample(t) for t in np.linspace(0.0, 6.0, 601)])

  assert process.bound_m == pytest.approx(bound_m)
  assert np.max(np.abs(offsets[:, :2])) <= bound_m + 1e-15
  np.testing.assert_array_equal(offsets[:, 2], np.zeros(len(offsets)))
  assert np.any(offsets[:, 0] != 0.0) and np.any(offsets[:, 1] != 0.0)


@pytest.mark.parametrize("std_m,bound_m", [(0.0001, 0.0003), (0.001, 0.002)])
def test_outlying_gaussian_knots_are_clipped_per_axis(monkeypatch, std_m, bound_m):
  generator = Mock()
  generator.normal.return_value = np.array([1.0, -1.0])
  monkeypatch.setattr(noise.np.random, "default_rng", Mock(return_value=generator))
  process = noise.PrecontactMotionNoise(std_m=std_m, seed=6)

  np.testing.assert_array_equal(process.sample(0.0), np.zeros(3))
  at_knot = process.sample(noise.KNOT_INTERVAL_S)
  generator.normal.assert_called_with(0, std_m, 2)
  np.testing.assert_allclose(at_knot, [bound_m, -bound_m, 0.0], atol=1e-15)


def test_offsets_change_smoothly_between_knots_without_stepwise_random_jitter():
  process = noise.PrecontactMotionNoise(std_m=0.0005, seed=61)
  interval = noise.KNOT_INTERVAL_S
  times = np.linspace(0.0, 3 * interval, 451)
  offsets = np.array([process.sample(t) for t in times])
  timestep = times[1] - times[0]

  # A bounded smooth knot transition moves only a small fraction of its
  # amplitude per 2 ms command, unlike independent noise at every step.
  maximum_step = 4.0 * process.bound_m * timestep / interval
  assert np.max(np.abs(np.diff(offsets, axis=0))) <= maximum_step + 1e-15
  assert process.draw_count < 10
  for start in (0, 150, 300):
    segment = offsets[start : start + 151]
    total_variation = np.abs(np.diff(segment, axis=0)).sum(axis=0)
    np.testing.assert_allclose(
      total_variation, np.abs(segment[-1] - segment[0]), atol=1e-14
    )


def test_offsets_join_knot_endpoints_with_negligible_velocity():
  process = noise.PrecontactMotionNoise(std_m=0.0005, seed=43)
  epsilon = 1e-4
  process.sample(0.0)
  for index in (1, 2, 3):
    knot_time = index * noise.KNOT_INTERVAL_S
    before = process.sample(knot_time - epsilon)
    at_knot = process.sample(knot_time)
    after = process.sample(knot_time + epsilon)
    assert np.max(np.abs(at_knot - before)) < process.bound_m * 1e-8
    assert np.max(np.abs(after - at_knot)) < process.bound_m * 1e-8


def test_scale_changes_amplitude_without_changing_the_random_sequence():
  full = noise.PrecontactMotionNoise(std_m=0.0004, seed=7)
  reduced = noise.PrecontactMotionNoise(std_m=0.0004, seed=7)
  for time_s in np.linspace(0.0, 1.0, 51):
    np.testing.assert_allclose(
      reduced.sample(time_s, scale=0.2), 0.2 * full.sample(time_s), atol=1e-15
    )
  assert reduced.draw_count == full.draw_count


@pytest.mark.parametrize("seed", [None, 12])
def test_zero_std_never_uses_rng_or_adds_offsets(monkeypatch, seed):
  entropy = Mock(side_effect=AssertionError("disabled noise requested entropy"))
  random_generator = Mock(side_effect=AssertionError("disabled noise created RNG"))
  monkeypatch.setattr(noise.secrets, "randbits", entropy)
  monkeypatch.setattr(noise.np.random, "default_rng", random_generator)
  process = noise.PrecontactMotionNoise(std_m=0.0, seed=seed)

  for time_s in (0.0, 0.05, 0.5, 5.0):
    np.testing.assert_array_equal(process.sample(time_s), np.zeros(3))
  assert not process.latch_contact(5.0, "close", "hand_r_index_link4_tactile_pad_col", 1.0)
  np.testing.assert_array_equal(process.random_offset_m, np.zeros(3))
  np.testing.assert_array_equal(process.recovery_offset_m, np.zeros(3))
  assert process.draw_count == 0
  assert not process.report()["enabled"]
  entropy.assert_not_called()
  random_generator.assert_not_called()


def test_first_single_finger_force_latches_immediately_and_cannot_be_rearmed():
  process = noise.PrecontactMotionNoise(std_m=0.0004, seed=13)
  process.sample(0.0)
  previous_offset = process.sample(0.25)
  assert np.any(previous_offset != 0.0)
  pad = "hand_r_pinky_link4_tactile_pad_col"
  for force in (0.0, noise.CONTACT_THRESHOLD_N / 2, noise.CONTACT_THRESHOLD_N):
    assert not process.latch_contact(0.25, "approach", pad, force)
  assert process.first_contact is None

  force = 2 * noise.CONTACT_THRESHOLD_N
  assert process.latch_contact(0.25, "approach", pad, force)
  first_contact = deepcopy(process.first_contact)
  assert first_contact == {
    "time_s": 0.25,
    "phase": "approach",
    "pad_name": pad,
    "normal_force_n": force,
  }
  np.testing.assert_array_equal(process.random_offset_m, np.zeros(3))
  np.testing.assert_array_equal(process.recovery_offset_m, previous_offset)
  draws_at_contact = process.draw_count
  for time_s, measured_force in ((0.26, 0.0), (0.4, 0.0), (2.0, 5.0)):
    assert not process.latch_contact(time_s, "close", "another_pad", measured_force)
    process.sample(time_s)
    assert process.first_contact == first_contact
    assert process.draw_count == draws_at_contact
    np.testing.assert_array_equal(process.random_offset_m, np.zeros(3))


def test_contact_handoff_preserves_position_and_finishes_recovery_after_120_ms():
  process = noise.PrecontactMotionNoise(std_m=0.0004, seed=97)
  process.sample(0.0)
  contact_time = 0.25
  before = process.sample(contact_time)
  assert np.any(before != 0.0)
  assert process.latch_contact(contact_time, "close", "index_pad", 0.0001)
  np.testing.assert_array_equal(process.sample(contact_time), before)

  elapsed = np.linspace(0.0, noise.RECOVERY_DURATION_S, 61)
  recovered = np.array([process.sample(contact_time + dt) for dt in elapsed])
  assert np.all(np.diff(np.linalg.norm(recovered, axis=1)) <= 1e-15)
  assert np.max(np.abs(np.diff(recovered, axis=0))) < 0.04 * np.max(np.abs(before))
  np.testing.assert_allclose(recovered[-1], np.zeros(3), atol=1e-15)
  for time_s in (0.5, 1.0, 8.0):
    np.testing.assert_array_equal(process.sample(time_s), np.zeros(3))
    np.testing.assert_array_equal(process.recovery_offset_m, np.zeros(3))


def test_contact_stops_actual_rng_calls_including_during_recovery(monkeypatch):
  generator = np.random.default_rng(73)
  monkeypatch.setattr(noise.np.random, "default_rng", lambda _seed: generator)
  process = noise.PrecontactMotionNoise(std_m=0.0003, seed=73)
  process.sample(0.0)
  process.sample(0.25)
  process.latch_contact(0.25, "approach", "thumb_pad", 1e-4)
  state_at_contact = deepcopy(generator.bit_generator.state)
  draws_at_contact = process.draw_count

  for time_s in (0.25, 0.3, 0.37, 1.0, 100.0):
    process.sample(time_s)
    assert generator.bit_generator.state == state_at_contact
    assert process.draw_count == draws_at_contact


def test_contact_before_first_sample_prevents_all_random_draws():
  process = noise.PrecontactMotionNoise(std_m=0.0005, seed=5)
  assert process.latch_contact(0.0, "approach", "thumb_pad", 1e-3)
  for time_s in (0.0, 0.02, 0.5, 2.0):
    np.testing.assert_array_equal(process.sample(time_s), np.zeros(3))
  assert process.draw_count == 0


def test_automatic_seeds_are_recorded_independent_and_leave_global_rng_untouched(monkeypatch):
  entropy = Mock(side_effect=[101, 202])
  monkeypatch.setattr(noise.secrets, "randbits", entropy)
  global_state = np.random.get_state()
  first = noise.PrecontactMotionNoise(std_m=0.0004)
  second = noise.PrecontactMotionNoise(std_m=0.0004)
  explicit = noise.PrecontactMotionNoise(std_m=0.0004, seed=101)
  for process in (first, second, explicit):
    process.sample(0.0)
  first_offset, second_offset = first.sample(0.15), second.sample(0.15)

  assert first.report()["seed"] == 101
  assert second.report()["seed"] == 202
  assert entropy.call_count == 2
  entropy.assert_called_with(63)
  assert not np.array_equal(first_offset, second_offset)
  np.testing.assert_array_equal(first_offset, explicit.sample(0.15))
  after = np.random.get_state()
  assert global_state[0] == after[0]
  np.testing.assert_array_equal(global_state[1], after[1])
  assert global_state[2:] == after[2:]


@pytest.mark.parametrize("std_m", [-1e-4, np.nan, np.inf, -np.inf])
def test_invalid_standard_deviation_is_rejected(std_m):
  with pytest.raises(ValueError, match="finite and nonnegative"):
    noise.PrecontactMotionNoise(std_m=std_m, seed=0)


@pytest.mark.parametrize("seed", [-1, True, False, 1.5, "12", np.nan])
def test_invalid_seed_is_rejected(seed):
  with pytest.raises(ValueError, match="nonnegative integer"):
    noise.PrecontactMotionNoise(std_m=0.0004, seed=seed)


@pytest.mark.parametrize(
  "time_s,scale",
  [(np.nan, 1.0), (np.inf, 1.0), (0.0, -0.1), (0.0, 1.1), (0.0, np.nan)],
)
def test_invalid_sample_time_or_scale_is_rejected(time_s, scale):
  process = noise.PrecontactMotionNoise(std_m=0.0004, seed=8)
  with pytest.raises(ValueError, match="nondecreasing finite time"):
    process.sample(time_s, scale=scale)
  assert process.draw_count == 0


def test_sampling_time_cannot_move_backwards():
  process = noise.PrecontactMotionNoise(std_m=0.0004, seed=8)
  process.sample(1.0)
  with pytest.raises(ValueError, match="nondecreasing finite time"):
    process.sample(0.5)
