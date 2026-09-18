"""Feed controller-generated fake-step evidence through the independent HDF5 audit.

No robot model, real physics, camera, renderer, or dataset collection is started.
"""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import tict_source_audit as audit


def _helpers(filename):
  path = Path(__file__).with_name(filename)
  spec = importlib.util.spec_from_file_location(f"contract_{path.stem}", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.mark.parametrize("sigma_deg", [0.0, 0.03, 0.05])
@pytest.mark.parametrize("initial_contact", [False, True])
def test_real_noise_code_trace_matches_independent_source_audit_with_fake_physics(
  tmp_path, monkeypatch, sigma_deg, initial_contact
):
  core = _helpers("test_precontact_noise.py")
  fixture = _helpers("test_tict_precontact_audit.py")
  sim = core._simulation(monkeypatch, signals={8: 1})
  # Use the known robot's scalar speed/ranges, without loading its MuJoCo model.
  limits, effective = audit._precontact_control_limits()
  sim.arm_speed_limit = 3.1416
  sim.model.actuator_ctrlrange[7:14] = limits
  sim.model.jnt_range[7:14] = effective
  sim.data.fingertip_contact[0] = initial_contact
  sim.configure_precontact_noise(seed=23, std_rad=float(np.deg2rad(sigma_deg)))
  initial = sim.precontact_noise_metadata()
  sim.step(20)
  final = sim.precontact_noise_metadata()
  trace = sim.precontact_noise_trace()
  path = fixture._synthetic_source(tmp_path)
  selected = np.arange(4, 20, 5)
  with h5py.File(path, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    metadata.update(precontact_noise=initial)
    metadata["preset_settings"]["action_noise"] = initial["settings"]
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["outcome_json"] = json.dumps(
      {"success": True, "precontact_noise": final}
    )
    group = file["control/precontact_noise"]
    group.attrs["metadata_json"] = json.dumps(final)
    for name, values in trace.items():
      group[name][:] = values
    file["state/timestamp"][:] = np.r_[0.0, trace["time_s"][selected] + 0.002]
    file["tactile_proxy/contact"][:] = np.vstack(
      (initial["initial_contact"], trace["contact"][selected])
    )
    file["tactile_proxy/normal_force"][:] = np.vstack(
      (initial["initial_normal_force_n"], trace["normal_force_n"][selected])
    )
    file["commands/actuator_control"][:] = np.vstack(
      (np.zeros(7), trace["actual_ctrl_rad"][selected])
    )
    report = audit.audit_precontact_noise(file)
  assert report["valid"], report
  assert report["postcontact_offset_exactly_zero"]
  assert report["source_actuator_controls_matched"]
  assert report["seeded_noise_projection_verified"]
