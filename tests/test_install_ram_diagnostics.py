"""Raw-force diagnostics preserve shared force semantics and physical clocks."""

import numpy as np
import pytest
from kaihand_tactile_env.tasks.install_ram.diagnostics import (
  ForceTraceRecorder,
  summarize_force_trace,
)
from kaihand_tactile_env.tasks.install_ram.task import (
  RamInstallationMonitor,
  RamInstallSimulation,
)


def test_lightweight_trace_matches_shared_solver_aggregates_and_clock(tmp_path):
  sim = RamInstallSimulation(add_genesis_probes=False)
  pad = sim.model.geom("hand_r_index_link4_tactile_pad_col").id
  # Intentional contact fixture, not a valid task trajectory: intersect the
  # free board and the real pad to ensure the equality check is nonzero.
  sim.set_object_pose("ram", sim.data.geom_xpos[pad], (1, 0, 0, 0))
  monitor = RamInstallationMonitor(sim)
  trace = ForceTraceRecorder(sim)
  state = monitor.measure()
  before = (sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ctrl.copy())
  assert trace.record("contact_fixture", state)
  assert not trace.record("duplicate_terminal", state)
  sample = trace._provider.read(sim.data)
  assert sample.normal_force_n.max() > 0
  np.testing.assert_array_equal(trace.arrays()["fn"][0], sample.normal_force_n)
  np.testing.assert_array_equal(trace.arrays()["ft"][0], sample.tangent_force_n)
  for current, saved in zip(
    (sim.data.qpos, sim.data.qvel, sim.data.ctrl), before, strict=True
  ):
    np.testing.assert_array_equal(current, saved)
  sim.step()
  assert trace.record("contact_fixture", monitor.measure())
  summary = trace.save(tmp_path / "trace.npz")
  assert summary["sample_count"] == 2
  assert summary["sample_hz"] == pytest.approx(500)
  assert (tmp_path / "trace.json").is_file()
  with np.load(tmp_path / "trace.npz") as data:
    assert data["socket"].shape == (2, 4)
    assert "linear_speed_m_s" in data
  sim.step(2)
  with pytest.raises(ValueError, match="skipped"):
    trace.record("gap", monitor.measure())


def test_raw_frequency_and_phase_boundaries_are_not_smoothed_or_bridged():
  time = np.arange(500) * 0.002
  force = np.zeros((500, 1, 2))
  force[:, 0, 0] = np.sin(2 * np.pi * 125 * time)
  normal = np.ones((500, 1))
  report = summarize_force_trace(
    time, ["known_signal"] * 500, normal, force, finger_names=("probe",)
  )
  phase = report["phase_intervals"][0]
  assert phase["peak_non_dc_frequency_hz"][0] == pytest.approx(125)
  assert phase["spectral_power_above_100hz_fraction"][0] == pytest.approx(1)
  force[:250] = 0
  force[250:] = 10
  report = summarize_force_trace(
    time, ["before"] * 250 + ["after"] * 250, normal, force, finger_names=("probe",)
  )
  assert all(
    phase["adjacent_ft_difference_rms_n"] == [0] for phase in report["phase_intervals"]
  )
