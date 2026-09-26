import pytest
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import PokerOutcomeMonitor


def sample(monitor, **updates):
  values = dict(
    time_s=1.0,
    dt=0.002,
    contacted=True,
    supported=True,
    flat=True,
    overhang=0.49,
    displacement=0.115,
    clearance=0.0,
    opposed=True,
    face_head=0.95,
    face_robot=0.95,
    position_error=0.01,
    linear_speed=0.0,
    angular_speed=0.0,
  )
  values.update(updates)
  monitor.observe(**values)


def test_full_progress_then_stable_hold():
  m = PokerOutcomeMonitor()
  sample(m)
  sample(m)
  sample(m, supported=False, clearance=0.03)
  assert m.lifted and not m.success
  for _ in range(50):
    sample(m, supported=False, clearance=0.03)
  assert m.success


@pytest.mark.parametrize(
  "changed",
  [
    dict(opposed=False),
    dict(face_head=0.7),
    dict(position_error=0.04),
    dict(linear_speed=0.03),
    dict(angular_speed=0.3),
  ],
)
def test_no_success_with_bad_terminal_condition(changed):
  m = PokerOutcomeMonitor()
  sample(m)
  sample(m)
  sample(m, clearance=0.03)
  for _ in range(100):
    sample(m, clearance=0.03, **changed)
  assert not m.success


def test_aerial_edge_crossing_not_a_slide():
  m = PokerOutcomeMonitor()
  for _ in range(100):
    sample(m, supported=False, clearance=0.03)
  assert not m.edge_reached and not m.success


def test_no_contact_failure():
  m = PokerOutcomeMonitor()
  for _ in range(100):
    sample(m, contacted=False, opposed=False)
  assert m.report()["failure_stage"] == "no_card_contact"


def test_hold_gap_resets_timer():
  m = PokerOutcomeMonitor()
  sample(m)
  sample(m)
  for _ in range(30):
    sample(m, clearance=0.03)
  assert m.hold_seconds > 0
  sample(m, clearance=0.03, opposed=False)
  assert m.hold_seconds == 0
  assert not m.success


def test_five_second_hold_requires_continuous_lift():
  m = PokerOutcomeMonitor(required_hold_seconds=5.0, require_clearance_during_hold=True)
  sample(m)
  sample(m)
  sample(m, supported=False, clearance=0.03)
  for _ in range(2000):
    sample(m, supported=False, clearance=0.03)
  assert not m.success
  sample(m, supported=False, clearance=0.0)
  assert m.hold_seconds == 0
  for _ in range(2499):
    sample(m, supported=False, clearance=0.03)
  assert not m.success
  sample(m, supported=False, clearance=0.03)
  assert m.success
  assert m.report()["required_hold_seconds"] == 5.0
