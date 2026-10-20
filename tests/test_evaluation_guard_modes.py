import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.evaluation_guards import (
  FatalSimulationError,
  TaskGuardViolation,
  configure_evaluation_guards,
  ensure_evaluation_physics,
  evaluation_guard_report,
  recover_nonfinite_action,
  task_guard,
)


def simulation():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".002" gravity="0 0 0"/>
    <worldbody><body><joint type="slide"/><geom size=".01"/></body></worldbody>
    </mujoco>
  """)
  return SimpleNamespace(model=model, data=mujoco.MjData(model), timestep=0.002)


def test_observe_guard_records_and_does_not_stop_physics():
  sim = simulation()
  configure_evaluation_guards(sim)
  for measured in (9.0, 12.0, 10.0):
    before = float(sim.data.time)
    mujoco.mj_step(sim.model, sim.data)
    ensure_evaluation_physics(sim, before_time=before)
    assert task_guard(
      sim,
      True,
      "force_guard",
      "board normal force exceeded",
      measured=measured,
      threshold=8,
    )
  report = evaluation_guard_report(sim, status="task_not_completed")
  assert report["stop_reason"] == "time_limit"
  assert report["physics_valid"] is True
  assert report["formal_metrics_valid"] is False
  assert len(report["guard_events"]) == 1
  event = report["guard_events"][0]
  assert event["count"] == 3
  assert event["measured_max"] == 12
  assert event["measured_min"] == 9
  assert event["last_sim_time_s"] > event["first_sim_time_s"]


@pytest.mark.parametrize("mode", [None, "strict"])
def test_strict_or_unconfigured_simulation_still_raises(mode):
  sim = simulation()
  if mode is not None:
    configure_evaluation_guards(sim, mode)
  with pytest.raises(TaskGuardViolation, match="fell below"):
    task_guard(
      sim,
      True,
      "object_dropped",
      "object fell below work surface",
      measured=0.4,
      threshold=0.5,
    )
  report = evaluation_guard_report(
    sim, status="error", error="TaskGuardViolation: object fell below work surface"
  )
  assert report["guard_mode"] == "strict"
  assert report["stop_reason"] == "guard"
  assert report["guard_events"][0]["count"] == 1
  assert report["formal_metrics_valid"] is True


def test_success_stops_even_with_observed_task_violations():
  sim = simulation()
  configure_evaluation_guards(sim)
  task_guard(sim, True, "collision_guard", "hand struck board")
  assert evaluation_guard_report(sim, status="success")["stop_reason"] == "success"


def test_structured_strict_policy_rejection_has_guard_stop_reason():
  sim = simulation()
  configure_evaluation_guards(sim, "strict")
  assert (
    evaluation_guard_report(sim, status="policy_action_rejected")["stop_reason"]
    == "guard"
  )


def test_infrastructure_fault_invalidates_strict_metrics_but_not_physics():
  sim = simulation()
  configure_evaluation_guards(sim, "strict")
  report = evaluation_guard_report(
    sim, status="error", error="ConnectionError: policy server unavailable"
  )
  assert report["stop_reason"] == "fatal_error"
  assert report["physics_valid"] is True
  assert report["formal_metrics_valid"] is False


@pytest.mark.parametrize("field", ["qpos", "qvel", "qacc"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e11])
def test_bad_physics_is_fatal_in_observe_mode(field, value):
  sim = simulation()
  configure_evaluation_guards(sim)
  getattr(sim.data, field)[0] = value
  with pytest.raises(FatalSimulationError):
    ensure_evaluation_physics(sim)
  report = evaluation_guard_report(sim, status="error", error="FatalSimulationError")
  assert report["stop_reason"] == "fatal_error"
  assert report["physics_valid"] is False
  json.dumps(report, allow_nan=False)


def test_autoreset_warning_and_clock_faults_are_fatal(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  sim = simulation()
  configure_evaluation_guards(sim)
  sim.data.time = 34
  sim.data.qacc[0] = 1e12
  mujoco.mj_checkAcc(sim.model, sim.data)
  assert sim.data.time == 0
  with pytest.raises(FatalSimulationError):
    ensure_evaluation_physics(sim, before_time=34)
  assert not evaluation_guard_report(sim)["physics_valid"]


def test_clock_stall_is_fatal_even_with_finite_state():
  sim = simulation()
  configure_evaluation_guards(sim)
  with pytest.raises(FatalSimulationError, match="clock"):
    ensure_evaluation_physics(sim, before_time=0)
  assert not evaluation_guard_report(sim)["physics_valid"]


def test_only_known_bad_model_actions_are_recoverable():
  sim = simulation()
  configure_evaluation_guards(sim)
  assert recover_nonfinite_action(
    sim, RuntimeError("policy actions contain nonfinite values")
  )
  assert sim._evaluation_action_rejected
  assert not recover_nonfinite_action(sim, RuntimeError("renderer failed"))
  assert not recover_nonfinite_action(sim, RuntimeError("nonfinite simulation state"))
  configure_evaluation_guards(sim, "strict")
  assert not recover_nonfinite_action(
    sim, RuntimeError("policy actions contain nonfinite values")
  )


def test_distinct_guards_do_not_merge_and_snapshots_are_independent():
  sim = simulation()
  configure_evaluation_guards(sim)
  for message in ("withdrawal resistance exceeded", "wall force exceeded"):
    task_guard(sim, True, "force_guard", message, measured=2, threshold=1.5)
  first = evaluation_guard_report(sim)
  assert len(first["guard_events"]) == 2
  first["guard_events"][0]["count"] = 500
  assert evaluation_guard_report(sim)["guard_events"][0]["count"] == 1


def test_event_details_accept_numpy_scalar_arrays():
  sim = simulation()
  configure_evaluation_guards(sim)
  task_guard(
    sim, True, "force_guard", "limit exceeded", details={"value": np.array(np.nan)}
  )
  report = evaluation_guard_report(sim)
  assert report["guard_events"][0]["last_details"]["value"] == "nan"
  json.dumps(report, allow_nan=False)
