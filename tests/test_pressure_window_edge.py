"""Full table-edge goal regressions without loading a robot or rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.pressure_window import (
  ForceLimitedPokerSimulation,
  PressureWindowExecutor,
  PressureWindowSettings,
  required_edge_travel_m,
)


def test_edge_goal_requires_real_card_travel_not_legacy_short_path() -> None:
  # A 63 mm card starts 114.13 mm inside the table edge; reaching 49% overhang
  # requires its actual centre to advance 113.50 mm, not the old 30 mm command.
  card_x, edge_x, projected_length = 0.57913, 0.465, 0.063
  travel = required_edge_travel_m(card_x, edge_x, projected_length)

  assert travel == pytest.approx(0.1135)
  assert travel > 3 * PressureWindowSettings().slide_distance_m
  final_x = card_x - travel
  actual_overhang = (edge_x - (final_x - 0.5 * projected_length)) / projected_length
  assert actual_overhang == pytest.approx(0.49)


@pytest.mark.parametrize("target_fraction", [0.01, 0.10, 0.25, 0.49])
def test_edge_travel_uses_requested_fraction_and_projected_extent(
  target_fraction: float,
) -> None:
  card_x, edge_x, projected_length = 0.61, 0.465, 0.090

  distance = required_edge_travel_m(card_x, edge_x, projected_length, target_fraction)

  target_x = edge_x + (0.5 - target_fraction) * projected_length
  assert distance == pytest.approx(card_x - target_x)


@pytest.mark.parametrize("card_x", [0.46563, 0.46, 0.43])
def test_already_at_or_beyond_requested_edge_needs_no_extra_outward_travel(
  card_x: float,
) -> None:
  assert required_edge_travel_m(card_x, 0.465, 0.063) == pytest.approx(0.0)


def test_edge_travel_is_invariant_to_world_origin() -> None:
  distance = required_edge_travel_m(0.57913, 0.465, 0.063)

  assert required_edge_travel_m(-1.42087, -1.535, 0.063) == pytest.approx(distance)


@pytest.mark.parametrize("argument", [0, 1, 2, 3])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_edge_geometry_rejects_nonfinite_values(argument: int, invalid: float) -> None:
  values = [0.57913, 0.465, 0.063, 0.49]
  values[argument] = invalid

  with pytest.raises(ValueError):
    required_edge_travel_m(*values)


@pytest.mark.parametrize("extent", [0.0, -0.063])
def test_edge_geometry_requires_positive_projected_card_extent(extent: float) -> None:
  with pytest.raises(ValueError):
    required_edge_travel_m(0.57913, 0.465, extent)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.491, 0.5, 1.0])
def test_edge_geometry_rejects_unsafe_or_nonpositive_overhang_target(
  fraction: float,
) -> None:
  with pytest.raises(ValueError):
    required_edge_travel_m(0.57913, 0.465, 0.063, fraction)


def test_default_goal_preserves_short_protocol_for_existing_callers() -> None:
  settings = PressureWindowSettings()

  assert settings.goal == "short"
  assert settings.slide_distance_m == pytest.approx(0.030)
  assert settings.max_slide_time_s == pytest.approx(30.0)
  assert settings.max_slide_travel_m == pytest.approx(0.16)
  assert settings.target_overhang_fraction == pytest.approx(0.49)
  assert settings.edge_dwell_s == pytest.approx(0.10)


@pytest.mark.parametrize("goal", ["", "edge", "table_edge", "full", None, 1])
def test_goal_is_explicitly_short_or_table_edge(goal: object) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(goal=goal)


def test_table_edge_settings_are_bounded_separately_from_legacy_short_distance() -> (
  None
):
  settings = PressureWindowSettings(
    goal="table-edge", max_slide_time_s=45.0, max_slide_travel_m=0.20
  )

  assert settings.max_slide_time_s > 12.0
  assert settings.max_slide_travel_m > 0.060


@pytest.mark.parametrize(
  "field",
  [
    "max_slide_time_s",
    "max_slide_travel_m",
    "target_overhang_fraction",
    "edge_dwell_s",
  ],
)
@pytest.mark.parametrize("invalid", [0.0, -0.1, np.nan, np.inf, -np.inf, None])
def test_table_edge_bounds_require_finite_positive_values(
  field: str,
  invalid: object,
) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(goal="table-edge", **{field: invalid})


@pytest.mark.parametrize(
  "field,invalid",
  [
    ("max_slide_time_s", 45.001),
    ("max_slide_travel_m", 0.20001),
    ("target_overhang_fraction", 0.49001),
    ("edge_dwell_s", 1.001),
  ],
)
def test_table_edge_bounds_reject_excessive_runtime_travel_or_overhang(
  field: str,
  invalid: float,
) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(goal="table-edge", **{field: invalid})


@pytest.mark.parametrize(
  "parameters",
  [
    {"slide_distance_m": 0.061},
    {"slide_distance_m": 0.030, "slide_speed_m_s": 0.002},
  ],
)
def test_short_protocol_keeps_original_runtime_and_travel_limits(
  parameters: dict[str, float],
) -> None:
  with pytest.raises(ValueError):
    PressureWindowSettings(goal="short", **parameters)


@dataclass
class _FakeSummary:
  sample_count: int = 1


def _edge_harness(
  monkeypatch: pytest.MonkeyPatch,
  *,
  transmission: float = 1.0,
  hand_moves: bool = True,
  saturated: bool = False,
  quality: bool = True,
  after_step: Callable[[str, SimpleNamespace], None] | None = None,
) -> tuple[PressureWindowExecutor, SimpleNamespace, SimpleNamespace]:
  """Replace all physics with deterministic observations, never compile a model."""
  sim = object.__new__(ForceLimitedPokerSimulation)
  sim.timestep = 0.010
  sim.data = SimpleNamespace(time=0.0)
  sim._arm_goal = {"right": np.zeros(7)}
  initial_hand = np.asarray([0.65, -0.16, 0.90])
  initial_card = np.asarray([0.57913, -0.16, 0.80, 1.0, 0.0, 0.0, 0.0])
  state = SimpleNamespace(
    sim=sim,
    ee=initial_hand.copy(),
    card=initial_card.copy(),
    forces=np.full(4, 0.35),
    samples=[],
    targets=[],
    quality=quality,
    saturated=saturated,
    monitor_resets=0,
  )
  monkeypatch.setattr(
    sim, "current_pose_matrix", lambda _side: (state.ee.copy(), np.eye(3))
  )
  monkeypatch.setattr(sim, "object_pose", lambda _name: state.card.copy())
  monkeypatch.setattr(
    sim,
    "solve_ik",
    lambda *_args, **_kwargs: SimpleNamespace(success=True, joint_positions=np.ones(7)),
  )
  monkeypatch.setattr(
    sim,
    "set_arm_joint_goal",
    lambda side, goal: sim._arm_goal.__setitem__(side, goal.copy()),
  )
  monkeypatch.setattr(
    sim, "drive_state", lambda: {"drive_saturated": float(state.saturated)}
  )
  executor = object.__new__(PressureWindowExecutor)
  executor.sim = sim
  executor.press_force_per_finger_n = 0.35
  executor._maximum_overhang_fraction = 0.0
  monkeypatch.setattr(
    executor,
    "prepare",
    lambda _plan: (np.zeros(7), initial_hand.copy(), np.eye(3)),
  )
  monkeypatch.setattr(executor, "_projected_card_length_x", lambda: 0.063)
  monkeypatch.setattr(
    executor, "_current_card_finger_normal_forces", lambda: state.forces.copy()
  )
  monkeypatch.setattr(
    executor, "_slide_press_control_qualified", lambda _summary: state.quality
  )

  def reset_monitor() -> None:
    state.monitor_resets += 1

  executor._slide_force_monitor = SimpleNamespace(
    reset=reset_monitor,
    summary=lambda: _FakeSummary(len(state.samples)),
  )

  def step(phase: str, table_edge_x: float) -> None:
    assert sim.drive_limit_n is not None
    target = sim._cartesian_drive["position"].copy()
    if hand_moves:
      state.ee[:] = target
    if phase == "slide_card":
      state.card[0] = initial_card[0] + transmission * (state.ee[0] - initial_hand[0])
    sim.data.time += sim.timestep
    if after_step is not None:
      after_step(phase, state)
    executor._maximum_overhang_fraction = max(
      executor._maximum_overhang_fraction, executor._overhang_fraction(table_edge_x)
    )
    state.samples.append((phase, sim.data.time, state.ee.copy(), state.card.copy()))
    state.targets.append(target)

  monkeypatch.setattr(executor, "_step", step)
  plan = SimpleNamespace(side="right", table_edge_x=0.465)
  return executor, plan, state


def _edge_settings(**overrides: object) -> PressureWindowSettings:
  values = {
    "goal": "table-edge",
    "slide_speed_m_s": 0.010,
    "max_slide_time_s": 20.0,
    "max_slide_travel_m": 0.16,
    "hold_seconds": 0.15,
  }
  values.update(overrides)
  return PressureWindowSettings(**values)


def test_full_goal_uses_actual_card_edge_and_exceeds_short_calibration_path(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _edge_harness(monkeypatch)

  result = executor.draw(plan, _edge_settings())

  assert result is executor.edge_outcome
  assert result["required_card_travel_m"] == pytest.approx(0.11350)
  assert result["actual_card_travel_m"] >= 0.11350
  assert result["actual_card_travel_m"] < 0.11352
  assert result["commanded_travel_m"] > 0.11
  assert result["target_reached"] is True
  assert result["held_at_edge"] is True
  assert result["full_slide_qualified"] is True
  assert result["terminal_reason"] == "edge_reached"
  assert result["final_overhang_fraction"] >= 0.49
  assert state.monitor_resets == 1
  assert {sample[0] for sample in state.samples} == {"slide_card", "edge_hold"}
  assert state.sim.drive_limit_n == 4.0
  assert executor._record_slide_force_metrics is False
  assert "model" not in state.sim.__dict__


def test_edge_stop_is_checked_each_physics_step_without_finishing_ik_segment(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def sudden_edge(phase: str, state: SimpleNamespace) -> None:
    if phase == "slide_card":
      state.card[0] = 0.4655

  executor, plan, state = _edge_harness(monkeypatch, after_step=sudden_edge)

  result = executor.draw(plan, _edge_settings())

  assert result["target_reached"] is True
  assert len([sample for sample in state.samples if sample[0] == "slide_card"]) == 1
  # The remaining point of this two-step IK segment must never be requested.
  for target in state.targets[1:]:
    np.testing.assert_array_equal(target, state.targets[0])


def test_completed_hand_command_cannot_substitute_for_card_reaching_edge(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _edge_harness(monkeypatch, transmission=0.0)

  result = executor.draw(plan, _edge_settings(max_slide_travel_m=0.12))

  assert result["commanded_travel_m"] == pytest.approx(0.12)
  assert state.ee[0] == pytest.approx(0.53)
  assert result["actual_card_travel_m"] == 0.0
  assert result["target_reached"] is False
  assert result["held_at_edge"] is False
  assert result["full_slide_qualified"] is False
  assert result["terminal_reason"] == "travel_limit"
  assert "edge_hold" not in {sample[0] for sample in state.samples}
  assert state.samples[-1][0] == "final_hold"


def test_thirty_mm_travel_limit_is_a_failure_not_an_edge_success(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, _state = _edge_harness(monkeypatch)

  result = executor.draw(plan, _edge_settings(max_slide_travel_m=0.030))

  assert result["actual_card_travel_m"] == pytest.approx(0.030)
  assert result["required_card_travel_m"] > 0.11
  assert result["target_reached"] is False
  assert result["full_slide_qualified"] is False
  assert result["terminal_reason"] == "travel_limit"


def test_low_transmission_cannot_pass_full_edge_goal_with_large_hand_motion(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, _state = _edge_harness(monkeypatch, transmission=0.2)

  result = executor.draw(plan, _edge_settings(max_slide_travel_m=0.14))

  assert result["actual_card_travel_m"] == pytest.approx(0.028)
  assert result["commanded_travel_m"] == pytest.approx(0.14)
  assert result["target_reached"] is False
  assert result["terminal_reason"] == "travel_limit"


def test_time_bound_stops_without_relabeling_it_as_a_completed_goal(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _edge_harness(monkeypatch)
  settings = _edge_settings(max_slide_time_s=0.035)

  result = executor.draw(plan, settings)

  slide_samples = [sample for sample in state.samples if sample[0] == "slide_card"]
  assert settings.max_slide_time_s <= slide_samples[-1][1]
  assert slide_samples[-1][1] < settings.max_slide_time_s + state.sim.timestep
  assert result["terminal_reason"] == "time_limit"
  assert result["target_reached"] is False
  assert result["full_slide_qualified"] is False
  assert state.sim.drive_limit_n == settings.drive_limit_n


def test_heavy_stop_requires_measured_hand_and_card_stall_with_loaded_saturated_drive(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _edge_harness(monkeypatch, hand_moves=False, saturated=True)

  result = executor.draw(plan, _edge_settings())

  slide_samples = [sample for sample in state.samples if sample[0] == "slide_card"]
  assert slide_samples[-1][1] == pytest.approx(1.01)
  assert result["terminal_reason"] == "sustained_drive_limit"
  assert result["actual_card_travel_m"] == 0.0
  assert result["full_slide_qualified"] is False
  assert state.sim.drive_limit_n == 4.0


@pytest.mark.parametrize(
  "missing_evidence", ["drive_saturation", "finger_load", "hand_stall"]
)
def test_no_stall_label_when_any_required_physical_evidence_is_missing(
  monkeypatch: pytest.MonkeyPatch,
  missing_evidence: str,
) -> None:
  executor, plan, state = _edge_harness(
    monkeypatch,
    transmission=0.0,
    hand_moves=missing_evidence == "hand_stall",
    saturated=missing_evidence != "drive_saturation",
  )
  if missing_evidence == "finger_load":
    state.forces[0] = 0.0

  result = executor.draw(plan, _edge_settings(max_slide_time_s=1.2))

  assert result["terminal_reason"] == "time_limit"
  assert result["full_slide_qualified"] is False


def test_continuous_all_finger_contact_loss_stops_and_retains_the_drive_cap(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, state = _edge_harness(monkeypatch, transmission=0.0)
  state.forces[:] = 0.0

  result = executor.draw(plan, _edge_settings())

  slide_samples = [sample for sample in state.samples if sample[0] == "slide_card"]
  assert slide_samples[-1][1] == pytest.approx(0.30)
  assert result["terminal_reason"] == "contact_lost"
  assert result["target_reached"] is False
  assert result["full_slide_qualified"] is False
  assert state.sim.drive_limit_n == 4.0


def test_noncontinuous_contact_loss_does_not_accumulate_to_false_terminal_loss(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def intermittent(phase: str, state: SimpleNamespace) -> None:
    state.forces[:] = 0.0
    if phase == "slide_card" and len(state.samples) % 20 == 19:
      state.forces[0] = 0.35

  executor, plan, _state = _edge_harness(
    monkeypatch, transmission=0.0, after_step=intermittent
  )

  result = executor.draw(plan, _edge_settings(max_slide_time_s=0.8))

  assert result["terminal_reason"] == "time_limit"


@pytest.mark.parametrize("hold_failure", ["card_rebounds", "finger_unloads"])
def test_brief_edge_crossing_is_not_success_without_loaded_endpoint_dwell(
  monkeypatch: pytest.MonkeyPatch,
  hold_failure: str,
) -> None:
  def fail_hold(phase: str, state: SimpleNamespace) -> None:
    if phase != "edge_hold":
      return
    if hold_failure == "card_rebounds":
      state.card[0] += 0.005
    else:
      state.forces[0] = 0.0

  executor, plan, _state = _edge_harness(monkeypatch, after_step=fail_hold)

  result = executor.draw(plan, _edge_settings())

  assert result["target_reached"] is True
  assert result["terminal_reason"] == "edge_reached"
  assert result["held_at_edge"] is False
  assert result["edge_dwell_s"] == 0.0
  assert result["full_slide_qualified"] is False


def test_reaching_and_holding_edge_does_not_bypass_full_slide_contact_quality(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor, plan, _state = _edge_harness(monkeypatch, quality=False)

  result = executor.draw(plan, _edge_settings())

  assert result["target_reached"] is True
  assert result["held_at_edge"] is True
  assert result["full_slide_qualified"] is False


def test_control_error_still_saves_failure_and_keeps_drive_limit_latched(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def failure(_phase: str, state: SimpleNamespace) -> None:
    if len(state.samples) == 2:
      raise RuntimeError("fake numerical interruption")

  executor, plan, state = _edge_harness(monkeypatch, after_step=failure)

  with pytest.raises(RuntimeError, match="fake numerical interruption"):
    executor.draw(plan, _edge_settings())

  assert executor.edge_outcome["terminal_reason"] == "control_error"
  assert executor.edge_outcome["target_reached"] is False
  assert executor.edge_outcome["full_slide_qualified"] is False
  assert executor._record_slide_force_metrics is False
  assert state.sim.drive_limit_n == 4.0


def test_edge_prepare_error_records_failure_without_starting_or_limiting_draw(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor = object.__new__(PressureWindowExecutor)
  original_error = RuntimeError("four-finger initial pressure did not settle")

  def failed_prepare(_plan: object) -> None:
    raise original_error

  monkeypatch.setattr(executor, "prepare", failed_prepare)

  with pytest.raises(RuntimeError) as captured:
    executor.draw(SimpleNamespace(), _edge_settings())

  assert captured.value is original_error
  assert executor.edge_outcome == {
    "target_reached": False,
    "terminal_reason": "prepare_error",
    "full_slide_qualified": False,
  }
  # A preparation exception must not access a simulation or activate draw.
  assert "sim" not in executor.__dict__
  assert "_record_slide_force_metrics" not in executor.__dict__


def test_short_prepare_error_preserves_original_exception_and_clears_stale_edge_result(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  executor = object.__new__(PressureWindowExecutor)
  executor.edge_outcome = {"target_reached": True, "terminal_reason": "edge_reached"}
  original_error = RuntimeError("legacy short prepare failed")

  def failed_prepare(_plan: object) -> None:
    raise original_error

  monkeypatch.setattr(executor, "prepare", failed_prepare)

  with pytest.raises(RuntimeError) as captured:
    executor.draw(SimpleNamespace(), PressureWindowSettings(goal="short"))

  assert captured.value is original_error
  assert executor.edge_outcome is None
  assert "sim" not in executor.__dict__
  assert "_record_slide_force_metrics" not in executor.__dict__
