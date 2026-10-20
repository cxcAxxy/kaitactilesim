"""Geometry denominators, contact gating, clocks, and non-mutating observers."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from kaihand_tactile_env.shared.whiteboard_geometry_metrics import (
  WhiteboardGeometryMetrics,
  _capsule_membership,
  _convex_hull_2d,
  _inside_convex,
  _polygon_area,
)


def _simulation(*, board_half_size=(0.15, 0.15, 0.01), ink_count=2,
                bottom_z=0.01, rotation=None):
  """Tiny real compiled mesh, with a controlled read-only solver contact view."""
  half_size = " ".join(str(value) for value in board_half_size)
  inks = "".join(
    f'<geom name="ink_{i}" type="capsule" pos="{(-0.02 + i * 0.04):g} 0 .0104" '
    'size=".005 .02" quat=".707106781 0 .707106781 0" contype="0" conaffinity="0"/>'
    for i in range(ink_count)
  )
  model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><compiler angle="radian"/><asset>
      <mesh name="felt" vertex="-.05 -.025 -.003 .05 -.025 -.003 .05 .025 -.003 -.05 .025 -.003
      -.05 -.025 .003 .05 -.025 .003 .05 .025 .003 -.05 .025 .003"
      face="0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"/>
    </asset><worldbody><geom name="board_surface" type="box" size="{half_size}"/>
    {inks}<body name="eraser" pos="0 0 {bottom_z + .003:g}">
      <freejoint/><geom name="eraser_pad" type="mesh" mesh="felt"/>
    </body></worldbody></mujoco>""")
  real_data = mujoco.MjData(model)
  mujoco.mj_forward(model, real_data)
  data = SimpleNamespace(
    time=0.0, geom_xpos=real_data.geom_xpos.copy(), geom_xmat=real_data.geom_xmat.copy(),
    xpos=real_data.xpos.copy(), xmat=real_data.xmat.copy(), ncon=1,
    contact=[SimpleNamespace(geom1=model.geom("eraser_pad").id,
                             geom2=model.geom("board_surface").id)],
  )
  if rotation is not None:
    # Rotate the entire scene; measurements must not depend on world-Z.
    data.geom_xpos[:] = data.geom_xpos @ rotation.T
    data.xpos[:] = data.xpos @ rotation.T
    data.geom_xmat[:] = np.einsum("ij,njk->nik", rotation,
                                  data.geom_xmat.reshape(-1, 3, 3)).reshape(-1, 9)
    data.xmat[:] = np.einsum("ij,njk->nik", rotation,
                            data.xmat.reshape(-1, 3, 3)).reshape(-1, 9)
  return SimpleNamespace(model=model, data=data,
                         ink_ids=np.array([model.geom(f"ink_{i}").id for i in range(ink_count)]),
                         remaining=np.ones(ink_count))


@pytest.fixture
def contact_force(monkeypatch):
  force = [2.0]

  def contact(_model, _data, _index, result):
    result[:] = 0.0
    result[0] = force[0]

  monkeypatch.setattr(mujoco, "mj_contactForce", contact)
  return force


def test_convex_geometry_and_capsule_footprint():
  polygon = _convex_hull_2d(np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [0, 0]]))
  assert _polygon_area(polygon) == pytest.approx(4.0)
  assert _inside_convex(np.array([[0, 0], [1, 1], [1.1, 0]]), polygon).tolist() == [True, True, False]
  assert _capsule_membership(np.array([[0, 0], [1.05, 0], [1.2, 0], [0, .2]]),
                             np.array([-1, 0]), np.array([1, 0]), .1).tolist() == [True, True, False, False]
  with pytest.raises(ValueError, match="footprint"):
    _convex_hull_2d(np.zeros((2, 2)))


def test_compiled_mesh_bottom_area_and_loaded_full_overlap(contact_force):
  sim = _simulation()
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .01
  row = metrics.update(sim)
  assert row["full_pad_bottom_area_m2"] == pytest.approx(.1 * .05, rel=1e-6)
  assert row["effective_pad_contact_area_ratio"] == pytest.approx(1.0)
  assert row["cumulative_ink_coverage_ratio"] == pytest.approx(1.0)
  assert row["pad_board_contact_duration_s"] == pytest.approx(.01)
  assert row["cleaned_fraction"] == 0.0
  assert row["ink_union_area_weighted_cleaned_fraction"] == pytest.approx(0.0)
  assert row["is_true_deformable_contact_area"] is False
  assert row["initial_ink_union_area_m2"] < .1 * .05
  json.dumps(row, allow_nan=False)


def test_tilted_scene_uses_board_coordinates_not_world_z(contact_force):
  angle = np.pi / 4
  rotation = np.array([[np.cos(angle), 0, -np.sin(angle)],
                       [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
  sim = _simulation(rotation=rotation)
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .001
  row = metrics.update(sim)
  assert row["effective_pad_contact_area_ratio"] == pytest.approx(1.0)
  assert row["cumulative_ink_coverage_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("case", ["no_solver_force", "no_pair", "hover", "penetration", "wrong_normal"])
def test_geometric_overlap_alone_never_implies_effective_contact(case, contact_force):
  sim = _simulation()
  metrics = WhiteboardGeometryMetrics(sim)
  if case == "no_solver_force":
    contact_force[0] = 0.0
  elif case == "no_pair":
    sim.data.ncon = 0
  elif case in ("hover", "penetration"):
    sim.data.xpos[sim.model.body("eraser").id, 2] += .003 if case == "hover" else -.003
  else:
    sim.data.xmat[sim.model.body("eraser").id] = np.diag([1, -1, -1]).ravel()
  sim.data.time = .01
  row = metrics.update(sim)
  assert row["effective_pad_contact_area_ratio"] == 0.0
  assert row["cumulative_ink_coverage_ratio"] == 0.0


def test_finite_board_clips_overlap_and_keeps_full_pad_denominator(contact_force):
  sim = _simulation(board_half_size=(.025, .15, .01))
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .01
  row = metrics.update(sim)
  assert row["full_pad_bottom_area_m2"] == pytest.approx(.005, rel=1e-6)
  assert row["effective_pad_contact_area_ratio"] == pytest.approx(.5, abs=.02)


def test_repeated_sweeps_do_not_double_count_target_coverage(contact_force):
  sim = _simulation()
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .01
  first = metrics.update(sim)
  for _ in range(100):
    sim.data.time += .001
    last = metrics.update(sim)
  assert last["cumulative_covered_ink_union_area_m2"] == first["cumulative_covered_ink_union_area_m2"]
  assert last["cumulative_ink_coverage_ratio"] == pytest.approx(1.0)
  assert last["geometry_sample_count"] < last["physics_observer_update_count"] / 10


def test_contact_duration_integrates_physics_dt_without_geometry_hold_leak(contact_force):
  sim = _simulation()
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .01
  metrics.update(sim)
  contact_force[0] = 0.0
  sim.data.time = .02
  row = metrics.update(sim)
  assert row["effective_pad_contact_area_ratio"] == 0.0
  assert row["elapsed_s"] == pytest.approx(.02)
  assert row["pad_board_contact_duration_s"] == pytest.approx(.01)
  assert row["pad_board_contact_time_fraction"] == pytest.approx(.5)
  assert row["time_weighted_mean_effective_area_ratio"] == pytest.approx(.5)
  assert row["contact_time_weighted_mean_effective_area_ratio"] == pytest.approx(1.0)
  assert row["peak_effective_area_ratio"] == pytest.approx(1.0)
  # A zero-duration refresh must not double-count the preceding step.
  assert metrics.update(sim)["elapsed_s"] == row["elapsed_s"]


def test_task_cached_integrated_load_survives_refreshed_solver_contacts(monkeypatch):
  sim = _simulation()
  sim.board_force = 2.0
  # Forward-updated contact arrays may differ from the integrated-step cache.
  sim.data.ncon = 0

  def forbidden(*_args):
    raise AssertionError("must use integrated task cache, not refreshed solver force")

  monkeypatch.setattr(mujoco, "mj_contactForce", forbidden)
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .001
  row = metrics.update(sim)
  assert row["effective_pad_contact_area_ratio"] == pytest.approx(1.0)
  assert row["contact_force_source"] == "task_cached_integrated_eraser_pad_board_force"
  assert row["load_geometry_alignment"] == "integrated pad-board load cached by task plus interval-end geometry"


def test_zero_integrated_task_load_overrides_positive_refreshed_solver_force(contact_force):
  sim = _simulation()
  sim.board_force = 0.0
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .001
  row = metrics.update(sim)
  assert row["effective_pad_contact_area_ratio"] == 0.0
  assert row["pad_board_contact_duration_s"] == 0.0
  assert row["contact_force_source"] == "task_cached_integrated_eraser_pad_board_force"


@pytest.mark.parametrize("force", [np.nan, np.inf, -1.0])
def test_invalid_integrated_task_load_fails_closed(force):
  sim = _simulation()
  sim.board_force = force
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = .001
  with pytest.raises(ValueError, match="cached integrated"):
    metrics.update(sim)


def test_initial_ink_union_is_fixed_and_cleaning_kept_independent(contact_force):
  sim = _simulation()
  metrics = WhiteboardGeometryMetrics(sim)
  sim.remaining[:] = .25
  sim.data.ncon = 0
  sim.data.time = .01
  row = metrics.update(sim)
  assert row["cleaned_fraction"] == .75
  assert row["ink_union_area_weighted_cleaned_fraction"] == pytest.approx(.75)
  assert row["cumulative_ink_coverage_ratio"] == 0.0
  assert row["initial_ink_union_area_m2"] == metrics.initial_ink_union_area_m2


def test_reset_and_invalid_sampling_inputs_fail_closed(contact_force):
  sim = _simulation()
  for kwargs in ({"geometry_sample_hz": 0}, {"max_gap_m": np.nan},
                 {"max_normal_angle_deg": 90}, {"pad_grid_resolution_m": -1}):
    with pytest.raises(ValueError):
      WhiteboardGeometryMetrics(sim, **kwargs)
  metrics = WhiteboardGeometryMetrics(sim)
  sim.data.time = 1
  metrics.update(sim)
  sim.data.time = 0
  with pytest.raises(ValueError, match="reset"):
    metrics.update(sim)


@pytest.mark.parametrize("task", ["whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"])
def test_real_whiteboard_geometries_without_render_or_physics_mutation(task, monkeypatch):
  if task == "whiteboard-wipe":
    from kaihand_tactile_env.tasks.whiteboard_wipe.task import WhiteboardWipeSimulation as Simulation
  elif task == "whiteboard-wipe-horizontal":
    from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal.task import HorizontalWhiteboardWipeSimulation as Simulation
  else:
    from kaihand_tactile_env.tasks.whiteboard_wipe_horizontal_line.task import HorizontalLineWhiteboardWipeSimulation as Simulation
  sim = Simulation(ink_seed=11)
  before = {name: getattr(sim.data, name).copy() for name in ("qpos", "qvel", "ctrl", "geom_xpos", "xpos")}
  time = float(sim.data.time)
  model_positions = sim.model.geom_pos.copy()

  def forbidden(*_args, **_kwargs):
    raise AssertionError("geometry observer must not forward/step/write physics")

  monkeypatch.setattr(mujoco, "mj_forward", forbidden)
  monkeypatch.setattr(mujoco, "mj_step", forbidden)
  metrics = WhiteboardGeometryMetrics(sim)
  row = metrics.update(sim)
  assert row["full_pad_bottom_area_m2"] == pytest.approx(.005469, abs=2e-6)
  assert row["initial_ink_union_area_m2"] > 0
  assert row["effective_pad_contact_area_ratio"] == 0
  assert row["cleaned_fraction"] == 0
  assert row["contact_force_source"] == "task_cached_integrated_eraser_pad_board_force"
  assert metrics.report() == metrics.snapshot()
  assert sim.data.time == time
  np.testing.assert_array_equal(sim.model.geom_pos, model_positions)
  for name, value in before.items():
    np.testing.assert_array_equal(getattr(sim.data, name), value)
