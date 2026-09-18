"""Feedback switching tests with fake state; never allocate a robot."""

from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.policy_control import (
  FINGERS,
  PokerPolicyController,
  make_force_controller,
)
from kaihand_tactile_env.tasks.poker_draw.pressure_window import PressureWindowExecutor
from kaihand_tactile_env.tasks.poker_draw.task import PokerDrawExecutor


class FakeSimulation:
  scene = "poker-draw"
  timestep = 0.002
  drive_limit_n = None

  def __init__(self):
    self.data = SimpleNamespace(
      time=0.0, contact=[], qpos=np.zeros(7), qvel=np.zeros(7)
    )
    self._hand_targets = {
      "right": {f"hand_r_{f}_joint{j}": 0.2 for f in FINGERS for j in (2, 3)}
    }
    self._hand_targets["right"]["hand_r_thumb_joint5"] = 0.1
    ids = {"card_core_geom": 1, "poker_table_top": 2}
    self.model = SimpleNamespace(
      geom=lambda name: SimpleNamespace(id=ids.setdefault(name, len(ids) + 1))
    )
    self.last_targets = None

  def set_hand_joint_targets(self, names, values):
    self._hand_targets["right"].update(zip(names, values, strict=True))
    return len(names)

  def begin_cartesian_drive(self, limit):
    self.drive_limit_n = limit
    self._cartesian_drive = {"position": np.zeros(3), "rotation": np.eye(3)}

  def set_cartesian_drive_position(self, value):
    self._cartesian_drive["position"] = value.copy()

  def end_cartesian_drive(self):
    self.drive_limit_n = None
    del self._cartesian_drive


def target(z=1.0):
  return SimpleNamespace(
    site_positions_world=np.array([[0.4, -0.1, z]]),
    site_rotations_world=np.eye(3)[None],
  )


def setup_feedback(monkeypatch):
  sim = FakeSimulation()
  ctrl = PokerPolicyController(sim)
  ctrl.after_command(target(), 0)
  monkeypatch.setattr(ctrl, "_supported", lambda: True)
  monkeypatch.setattr(ctrl, "_near_flat_pads", lambda: True)
  monkeypatch.setattr(ctrl, "_card_table_clearance", lambda: 0.0001)
  monkeypatch.setattr(
    ctrl,
    "_current_card_face_contact_details",
    lambda: (set(FINGERS), set(), {"thumb": 0.0, **{f: 0.5 for f in FINGERS}}),
  )

  def activate():
    ctrl._press_linkage_directions = np.ones(4)
    ctrl._press_joint2_base = ctrl._press_joint3_base = np.ones(4) * 0.2
    ctrl._press_controller.reset(np.ones(4) * 0.5)
    ctrl._press_controller_active = True

  monkeypatch.setattr(ctrl, "_activate_press_force_controller", activate)
  return sim, ctrl


def advance(sim, ctrl, count):
  for _ in range(count):
    sim.data.time += sim.timestep
    ctrl.after_step()


def test_free_space_has_no_4n_cap():
  sim = FakeSimulation()
  ctrl = PokerPolicyController(sim)
  ctrl.after_command(target(), 0)
  assert sim.drive_limit_n is None
  assert ctrl.mode == "approach"


def test_loaded_contact_enters_cartesian_and_policy_withdrawal_releases(monkeypatch):
  sim, ctrl = setup_feedback(monkeypatch)
  before = sim.data.qpos.copy()
  advance(sim, ctrl, 60)
  assert ctrl.mode == "slide"
  assert sim.drive_limit_n == 4.0
  ctrl.after_command(target(1.009), 0)
  assert ctrl.mode == "manipulate"
  assert sim.drive_limit_n is None
  np.testing.assert_array_equal(sim.data.qpos, before)
  assert not ctrl.report()["source_phases_used"]


def test_force_residual_does_not_accumulate_in_next_ik_seed(monkeypatch):
  sim, ctrl = setup_feedback(monkeypatch)
  expected = ctrl.base.copy()
  sim._hand_targets["right"]["hand_r_index_joint2"] += 0.01
  ctrl.before_command()
  assert sim._hand_targets["right"] == expected


def test_sustained_low_load_stops_not_short_gap(monkeypatch):
  sim, ctrl = setup_feedback(monkeypatch)
  advance(sim, ctrl, 60)
  monkeypatch.setattr(
    ctrl,
    "_current_card_face_contact_details",
    lambda: (set(), set(), {"thumb": 0.0, **{f: 0.0 for f in FINGERS}}),
  )
  advance(sim, ctrl, 17)
  assert ctrl.mode == "slide"
  with pytest.raises(RuntimeError, match="0.30 s"):
    advance(sim, ctrl, 133)


def test_contact_support_uses_solver_force_not_positive_margin_distance(monkeypatch):
  sim = FakeSimulation()
  ctrl = PokerPolicyController(sim)
  sim.data.contact = [SimpleNamespace(geom1=1, geom2=2, dist=0.00015)]

  def force(model, data, index, output):
    output[0] = 0.04

  monkeypatch.setattr(
    "kaihand_tactile_env.tasks.poker_draw.policy_control.mujoco.mj_contactForce", force
  )
  assert ctrl._supported()


@pytest.mark.parametrize("guard_enabled", [True, False])
def test_penetration_guard_can_be_disabled_without_losing_diagnostics(
  monkeypatch, guard_enabled
):
  sim = FakeSimulation()
  sim.data.time = 0.1
  ctrl = PokerPolicyController(sim, penetration_guard_enabled=guard_enabled)
  monkeypatch.setattr(ctrl, "_supported", lambda: True)
  monkeypatch.setattr(ctrl, "_card_table_clearance", lambda: -0.0007)
  monkeypatch.setattr(
    ctrl,
    "_current_card_face_contact_details",
    lambda: (set(), set(), {"thumb": 0.0, **{f: 0.0 for f in FINGERS}}),
  )
  if guard_enabled:
    with pytest.raises(RuntimeError, match="card penetrated supported tabletop"):
      ctrl.after_step()
  else:
    ctrl.after_step()
  assert ctrl.maximum_supported_table_penetration_m == pytest.approx(0.0007)
  assert ctrl.first_penetration_limit_exceeded_s == pytest.approx(0.1)


def test_pressure_integrator_matches_capture_controller(monkeypatch):
  def initialize(self, simulation, **kwargs):
    self.press_force_per_finger_n = 0.5

  monkeypatch.setattr(PokerDrawExecutor, "__init__", initialize)
  source = PressureWindowExecutor(SimpleNamespace(timestep=0.002))._press_controller
  deployed = make_force_controller(0.002)
  rng = np.random.default_rng(12)
  for forces in rng.uniform(0, 1, size=(200, 4)):
    expected, _ = source.observe(forces)
    actual, _ = deployed.observe(forces)
    np.testing.assert_array_equal(actual, expected)


def test_pinch_unloads_excess_force_and_preserves_baseline(monkeypatch):
  sim, ctrl = setup_feedback(monkeypatch)
  ctrl.mode = "pinch"
  baseline = ctrl.base.copy()
  monkeypatch.setattr(
    ctrl,
    "_current_card_face_contact_details",
    lambda: (set(FINGERS), {"thumb"}, {"thumb": 12.0, **{f: 3.0 for f in FINGERS}}),
  )
  advance(sim, ctrl, 10)
  np.testing.assert_allclose(ctrl.pinch_residual, -np.deg2rad(0.020))
  assert ctrl.base == baseline
  ctrl.before_command()
  assert sim._hand_targets["right"] == baseline


def test_pinch_in_band_does_not_force_exact_tracking(monkeypatch):
  sim, ctrl = setup_feedback(monkeypatch)
  ctrl.mode = "pinch"
  monkeypatch.setattr(
    ctrl,
    "_current_card_face_contact_details",
    lambda: (set(FINGERS), {"thumb"}, {"thumb": 1.2, **{f: 0.3 for f in FINGERS}}),
  )
  advance(sim, ctrl, 50)
  np.testing.assert_array_equal(ctrl.pinch_residual, np.zeros(5))
