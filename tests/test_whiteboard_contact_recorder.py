"""Integrated-step contact loads and diagnostic-only evaluation behavior."""

from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared import whiteboard_contact_recorder as contact
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.evaluation_guards import FatalSimulationError
from kaihand_tactile_env.shared.policy_tasks import TaskPolicyAdapter


def _adapter():
  names = {
    f"{name}_tactile_pad_col": index
    for index, name in enumerate(FINGERTIP_LINK_NAMES[5:])
  }
  names.update(eraser_handle=10, eraser_pad=11)
  sim = SimpleNamespace(
    model=SimpleNamespace(geom=lambda name: SimpleNamespace(id=names[name])),
    data=SimpleNamespace(
      time=0.001,
      ncon=3,
      contact=[
        SimpleNamespace(geom1=0, geom2=10),
        SimpleNamespace(geom1=11, geom2=0),
        SimpleNamespace(geom1=0, geom2=99),
      ],
    ),
    timestep=0.001,
    evaluation_contact_observer=None,
    board_force=2.0,
    board_tangent_force=1.2,
    direct_hand_board_force=0.7,
    remaining=np.ones(2),
    cleaning=SimpleNamespace(work_j=np.zeros(2)),
  )
  return SimpleNamespace(
    task="whiteboard-wipe",
    seed=1,
    simulation=sim,
    initial_object_pose=np.zeros(7),
  )


def _wrench(_model, _data, contact_id, result):
  result[:] = 0
  result[:3] = [(2, 3, 4), (4, -3, -4), (100, 100, 100)][contact_id]


def test_contact_loads_commit_only_valid_steps_and_preserve_opposed_shear(monkeypatch):
  adapter = _adapter()
  monkeypatch.setattr(contact.mujoco, "mj_contactForce", _wrench)
  recorder = contact.start_whiteboard_contact_recording(adapter)
  recorder.stage_step(adapter.simulation, 0.0)
  assert recorder._rows == []
  adapter.simulation.remaining[:] = 0.5
  recorder.commit_step()
  assert recorder._times == [0.0]
  assert recorder._durations == [0.001]
  row = dict(zip(contact.CHANNEL_NAMES, recorder._rows[0], strict=True))
  assert row["finger_thumb_normal_n"] == 6
  assert row["finger_thumb_tangent_load_n"] == 10
  assert row["grip_normal_n"] == 6
  assert row["grip_tangent_load_n"] == 10
  assert row["board_normal_n"] == 2
  assert row["hand_board_normal_n"] == 0.7
  # Signals are staged with the force sample, not read from a later live state.
  assert recorder._signal_rows[0][0] == 0
  assert adapter.simulation.data.time == 0.001
  recorder.commit_step()
  assert len(recorder._rows) == 1


def test_invalid_force_is_not_silently_replaced_with_zero(monkeypatch):
  adapter = _adapter()
  monkeypatch.setattr(contact.mujoco, "mj_contactForce", _wrench)
  recorder = contact.start_whiteboard_contact_recording(adapter)
  adapter.simulation.board_force = np.nan
  recorder.stage_step(adapter.simulation, 0)
  recorder.commit_step()
  assert len(recorder._rows) == 0
  assert recorder.rejected_samples == 1


def test_callback_errors_and_nonwhiteboard_tasks_do_not_stop_policy(monkeypatch):
  assert (
    contact.start_whiteboard_contact_recording(SimpleNamespace(task="bulb-screw"))
    is None
  )
  adapter = _adapter()
  recorder = contact.start_whiteboard_contact_recording(adapter)

  def fail(*_args):
    raise ValueError("test diagnostic failure")

  monkeypatch.setattr(contact.mujoco, "mj_contactForce", fail)
  recorder.stage_step(adapter.simulation, 0)
  recorder.commit_step()
  assert "test diagnostic failure" in recorder.error
  assert not recorder._rows
  assert adapter.simulation.data.time == 0.001
  adapter.simulation.data.time = 0.002
  recorder.stage_step(adapter.simulation, 0.001)
  assert recorder.attempted_steps == 2
  assert recorder.rejected_samples == 2


def test_integrated_solver_forces_are_captured_before_forward(monkeypatch):
  from kaihand_tactile_env.shared.simulation import ArmHandSimulation
  from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation

  sim = WhiteboardWipeSimulation.__new__(WhiteboardWipeSimulation)
  sim.data = SimpleNamespace(time=0)
  sim.model = object()
  sim.board_force = 0
  seen = []

  def integrated_step(self):
    self.data.time += 0.001
    self.solver_force = 2.0

  def forward(_model, _data):
    sim.solver_force = 99.0

  monkeypatch.setattr(ArmHandSimulation, "step", integrated_step)
  monkeypatch.setattr(contact.mujoco, "mj_forward", forward)
  sim._update_cleaning = lambda: setattr(sim, "board_force", sim.solver_force)
  sim.evaluation_contact_observer = lambda simulation, before: seen.append(
    (before, simulation.data.time, simulation.solver_force, simulation.board_force)
  )
  sim.step(2)
  assert seen == [(0, 0.001, 2.0, 2.0), (0.001, 0.002, 2.0, 2.0)]
  assert sim.solver_force == 99


def test_new_sampling_does_not_change_whiteboard_physics_or_stop_rules():
  baseline = TaskPolicyAdapter("whiteboard-wipe", seed=1, guard_mode="observe")
  observed = TaskPolicyAdapter("whiteboard-wipe", seed=1, guard_mode="observe")
  recorder = contact.start_whiteboard_contact_recording(observed)
  assert recorder.error is None
  for _ in range(10):
    baseline.step()
    observed.step()
  assert len(recorder._rows) == 10
  assert recorder._times == pytest.approx(np.arange(10) / 1000)
  for field in ("qpos", "qvel", "qacc", "ctrl"):
    np.testing.assert_array_equal(
      getattr(observed.simulation.data, field), getattr(baseline.simulation.data, field)
    )
  assert observed.success == baseline.success


def test_physics_failure_discards_pending_step_and_exports_only_cached_samples(
  monkeypatch,
):
  adapter = TaskPolicyAdapter("whiteboard-wipe", seed=1, guard_mode="observe")
  recorder = contact.start_whiteboard_contact_recording(adapter)
  adapter.step()
  step = adapter.simulation.step

  def corrupt_after_integrating():
    step()
    adapter.simulation.data.qpos[0] = np.nan

  monkeypatch.setattr(adapter.simulation, "step", corrupt_after_integrating)
  with pytest.raises(FatalSimulationError):
    adapter.step()
  assert recorder._pending is not None
  assert len(recorder._rows) == 1
  trace = recorder.trace({"physics_valid": False, "status": "error"})
  assert len(trace.time_s) == 1
  assert trace.metadata["rejected_samples"] == 1
  assert trace.metadata["physics_valid"] is False
  assert trace.signals["cleaned_fraction"].shape == (1,)
  assert trace.metadata["cleaned_fraction_clock"] == "interval_end"
  assert (
    trace.metadata["final_cleaned_fraction"] == trace.signals["cleaned_fraction"][-1]
  )
  recorder.detach()
  assert adapter.simulation.evaluation_contact_observer is None


def test_finish_missing_reference_and_export_failure_preserve_task_result(
  tmp_path, monkeypatch
):
  from kaihand_tactile_env.shared import whiteboard_contact_intensity as intensity
  from kaihand_tactile_env.shared import whiteboard_contact_reference as reference

  adapter = _adapter()
  monkeypatch.setattr(contact.mujoco, "mj_contactForce", _wrench)
  recorder = contact.start_whiteboard_contact_recording(adapter)
  recorder.stage_step(adapter.simulation, 0)
  recorder.commit_step()
  # Finish must not sample the simulation again, even with a broken live state.
  adapter.simulation.data = None
  report = {"status": "success", "task_success": True, "physics_valid": True}

  def missing_reference(**_kwargs):
    raise FileNotFoundError("expert unavailable")

  def export(_output, rollout, reference, *, reference_error):
    assert len(rollout.time_s) == 1
    assert reference is None
    assert "expert unavailable" in reference_error
    raise OSError("disk full")

  monkeypatch.setattr(reference, "load_whiteboard_contact_reference", missing_reference)
  monkeypatch.setattr(intensity, "export_contact_intensity", export)
  result = contact.finish_whiteboard_contact_recording(recorder, tmp_path, report)
  assert result["status"] == "error"
  assert "disk full" in result["reason"]
  assert report == {"status": "success", "task_success": True, "physics_valid": True}
  assert (
    contact.finish_whiteboard_contact_recording(recorder, tmp_path, report) is result
  )
