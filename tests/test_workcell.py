from __future__ import annotations

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.workcell.config import FINGERTIP_LINK_NAMES, WorkcellConfig
from kaihand_tactile_env.workcell.planning import (
  GraspExecutor,
  KnownStateGraspPlanner,
  PickPlaceExecutor,
)
from kaihand_tactile_env.workcell.recording import EpisodeRecorder, validate_episode
from kaihand_tactile_env.workcell.simulation import ARM_HOME, ArmHandSimulation
from kaihand_tactile_env.workcell.tactile import (
  GENESIS_PROBE_GEOM_PREFIX,
  GenesisProbeTactileProvider,
  LinkTactileSample,
  SolverContactTactileProvider,
  compare_link_samples,
)


@pytest.fixture(scope="module")
def simulation() -> ArmHandSimulation:
  return ArmHandSimulation()


def test_workcell_model_contract(simulation: ArmHandSimulation) -> None:
  model = simulation.model
  assert model.opt.timestep == pytest.approx(0.002)
  assert model.ncam == 3
  for name in ("table", "cylinder", "box"):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
  for name in ("overhead", "front", "head"):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
  actuator_names = {model.actuator(index).name for index in range(model.nu)}
  for side in ("l", "r"):
    for finger in ("index", "middle", "ring", "pinky"):
      assert f"hand_{side}_{finger}_joint4" in actuator_names
    expected_force_limits = {
      "thumb": 1.5,
      "index": 0.35,
      "middle": 0.4,
      "ring": 0.35,
      "pinky": 0.4,
    }
    for finger, limit in expected_force_limits.items():
      actuator = model.actuator(f"hand_{side}_{finger}_joint1")
      assert actuator.forcelimited[0]
      np.testing.assert_allclose(actuator.forcerange, (-limit, limit))
  assert model.geom("cylinder_geom").margin[0] == pytest.approx(0.001)
  assert not any("backlash" in name for name in actuator_names)
  assert "hand_l_thumb_joint4" not in actuator_names
  assert "hand_r_thumb_joint4" not in actuator_names
  probe_geoms = [
    model.geom(index)
    for index in range(model.ngeom)
    if model.geom(index).name.startswith(GENESIS_PROBE_GEOM_PREFIX)
  ]
  assert len(probe_geoms) == 350
  assert all(geom.contype[0] == 0 and geom.conaffinity[0] == 0 for geom in probe_geoms)
  for link_name in FINGERTIP_LINK_NAMES:
    geom = model.geom(f"{link_name}_tactile_pad_col")
    assert model.body(int(geom.bodyid[0])).name == link_name


def test_reset_jitter_is_reproducible(simulation: ArmHandSimulation) -> None:
  simulation.reset(seed=7, object_xy_jitter=0.01, object_yaw_jitter=np.pi)
  first = simulation.object_pose("cylinder")
  simulation.reset(seed=7, object_xy_jitter=0.01, object_yaw_jitter=np.pi)
  second = simulation.object_pose("cylinder")
  np.testing.assert_allclose(first, second)
  simulation.reset(seed=8, object_xy_jitter=0.01)
  assert not np.allclose(first[:2], simulation.object_pose("cylinder")[:2])


def test_reset_can_randomize_only_one_object(simulation: ArmHandSimulation) -> None:
  simulation.reset(
    seed=7,
    object_xy_jitter=0.01,
    object_yaw_jitter=np.pi,
    randomized_objects=("cylinder",),
  )
  cylinder = simulation.object_pose("cylinder")
  unrandomized = simulation.object_pose("cylinder")
  simulation.reset(randomized_objects=())
  assert not np.allclose(cylinder, simulation.object_pose("cylinder"))
  assert not np.allclose(unrandomized, simulation.object_pose("cylinder"))


@pytest.mark.parametrize(("object_name", "side"), (("cylinder", "right"),))
def test_known_state_grasp_lifts_and_retains_object(
  object_name: str, side: str
) -> None:
  simulation = ArmHandSimulation()
  provider = SolverContactTactileProvider(simulation.model)
  genesis_provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  maximum_force = 0.0
  maximum_probe_depth = 0.0
  touched_right_fingers = np.zeros(5, dtype=bool)

  def observe(sim: ArmHandSimulation, phase: str) -> None:
    nonlocal maximum_force, maximum_probe_depth
    if phase in {"close", "lift"}:
      maximum_force = max(
        maximum_force, float(provider.read(sim.data).normal_force.max())
      )
      genesis_provider.read(sim.data)
      maximum_probe_depth = max(
        maximum_probe_depth, float(genesis_provider.probe_depth.max())
      )
      touched_right_fingers[:] |= genesis_provider.read(sim.data).contact[5:]

  plan = KnownStateGraspPlanner(simulation).plan(object_name, side)
  result = GraspExecutor(simulation, observer=observe).execute(plan)
  assert result.success
  assert result.retained_at_end
  assert result.maximum_height >= result.initial_height + 0.06
  assert maximum_force > 0.1
  assert maximum_probe_depth > genesis_provider.contact_threshold_m
  assert touched_right_fingers.all()


@pytest.mark.parametrize("seed", (None, 0, 1, 2, 3, 4, 11, 17, 29, 39, 43))
def test_pick_place_returns_arm_home_and_places_cylinder(seed: int | None) -> None:
  simulation = ArmHandSimulation()
  if seed is not None:
    simulation.reset(
      seed=seed,
      object_xy_jitter=0.01,
      object_yaw_jitter=0.05,
      randomized_objects=("cylinder",),
    )
  tactile = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  all_fingers_seen = {phase: False for phase in ("lift", "transfer", "place")}
  thumb_contact_continuous = {phase: True for phase in ("lift", "transfer", "place")}
  all_fingers_continuous = {"transfer": True}
  place_contact_frames = np.zeros(5, dtype=np.int32)
  place_frames = 0
  phase_started: dict[str, float] = {}
  phase_finished: dict[str, float] = {}
  last_pad_contact: dict[str, np.ndarray] = {}
  last_tactile_contact: dict[str, np.ndarray] = {}
  box_wall_ids = {
    simulation.model.geom(name).id
    for name in (
      "box_wall_x_pos",
      "box_wall_x_neg",
      "box_wall_y_pos",
      "box_wall_y_neg",
    )
  }
  cylinder_geom_id = simulation.model.geom("cylinder_geom").id
  tactile_pad_ids = {
    simulation.model.geom(f"{name}_tactile_pad_col").id: index
    for index, name in enumerate(tactile.link_names[5:])
  }
  index_actuator_ids = np.array(
    [
      actuator_id
      for name, actuator_id in simulation._hand_actuators["right"].items()
      if "_index_" in name
    ],
    dtype=np.int32,
  )
  baseline_index_force_ranges = simulation.model.actuator_forcerange[
    index_actuator_ids
  ].copy()
  first_pad_contact_time = np.full(5, np.nan)
  first_tactile_time = np.full(5, np.nan)
  box_wall_penetration = False
  maximum_hand_penetration = 0.0
  minimum_place_index_force_limit = np.inf
  release_started_at: float | None = None
  release_fall_delay: float | None = None

  def observe(sim: ArmHandSimulation, phase: str) -> None:
    nonlocal box_wall_penetration
    nonlocal maximum_hand_penetration
    nonlocal minimum_place_index_force_limit
    nonlocal place_frames
    nonlocal release_fall_delay
    nonlocal release_started_at
    phase_started.setdefault(phase, float(sim.data.time) - sim.timestep)
    phase_finished[phase] = float(sim.data.time)
    sample = tactile.read(sim.data)
    right_contact = sample.contact_count[5:] > 0
    last_tactile_contact[phase] = right_contact.copy()
    newly_active = right_contact & np.isnan(first_tactile_time)
    first_tactile_time[newly_active] = sim.data.time
    if phase in all_fingers_seen:
      all_fingers_seen[phase] |= bool(np.all(right_contact))
      thumb_contact_continuous[phase] &= bool(right_contact[0])
      if phase in all_fingers_continuous:
        all_fingers_continuous[phase] &= bool(np.all(right_contact))
    if phase == "place":
      place_frames += 1
      place_contact_frames[:] += right_contact
      minimum_place_index_force_limit = min(
        minimum_place_index_force_limit,
        float(
          np.min(
            simulation.model.actuator_forcerange[index_actuator_ids, 1]
          )
        ),
      )
    pad_contact = np.zeros(5, dtype=bool)
    for index in range(sim.data.ncon):
      contact = sim.data.contact[index]
      pair = {int(contact.geom1), int(contact.geom2)}
      # Contacts after release/retreat are free-object settling, not a place
      # trajectory collision.  The controlled descent and opening must remain
      # clear of every box wall.
      if phase in {"place", "release"}:
        box_wall_penetration |= bool(
          cylinder_geom_id in pair
          and not pair.isdisjoint(box_wall_ids)
          and contact.dist < 0.0
        )
      if cylinder_geom_id not in pair:
        continue
      other_geom = contact.geom2 if contact.geom1 == cylinder_geom_id else contact.geom1
      if other_geom in tactile_pad_ids:
        finger_index = tactile_pad_ids[other_geom]
        pad_contact[finger_index] = True
        if np.isnan(first_pad_contact_time[finger_index]):
          first_pad_contact_time[finger_index] = sim.data.time
      other_body = int(sim.model.geom_bodyid[other_geom])
      if sim.model.body(other_body).name.startswith("hand_r_"):
        maximum_hand_penetration = max(
          maximum_hand_penetration, max(0.0, -float(contact.dist))
        )
    last_pad_contact[phase] = pad_contact
    if phase == "release":
      if release_started_at is None:
        release_started_at = float(sim.data.time) - sim.timestep
      if release_fall_delay is None and sim.object_twist("cylinder")[2] < -0.02:
        release_fall_delay = float(sim.data.time) - release_started_at

  result = PickPlaceExecutor(simulation, observer=observe).execute(
    KnownStateGraspPlanner(simulation).plan_pick_and_place("right")
  )
  assert result.placed_in_box
  assert all(all_fingers_seen.values())
  assert all(thumb_contact_continuous.values())
  assert all(all_fingers_continuous.values())
  assert np.all(last_pad_contact["close"])
  assert np.all(last_tactile_contact["close"])
  assert np.all(place_contact_frames / place_frames > 0.98)
  assert minimum_place_index_force_limit >= 0.50
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[index_actuator_ids],
    baseline_index_force_ranges,
  )
  assert not box_wall_penetration
  assert maximum_hand_penetration < 0.003
  assert np.all(np.isfinite(first_pad_contact_time))
  np.testing.assert_allclose(
    first_tactile_time,
    first_pad_contact_time,
    atol=simulation.timestep,
  )
  assert release_fall_delay is not None
  assert release_fall_delay < 0.08
  assert "settle_open" not in result.phases
  assert "align_above_box" not in result.phases
  assert "release_settle" not in result.phases
  assert phase_finished["close"] - phase_started["close"] < 0.5
  assert phase_finished["place"] - phase_started["place"] < 0.5
  assert simulation.data.time < 3.6
  assert result.phases[-1] == "return_home"
  np.testing.assert_allclose(
    simulation.data.qpos[simulation._arm_qpos["right"]],
    ARM_HOME["right"],
    atol=1.0e-3,
  )


def test_bimanual_genesis_provider_is_available_without_proxy_fallback(
  simulation: ArmHandSimulation,
) -> None:
  simulation.reset()
  provider = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  sample = provider.read(simulation.data)
  assert provider.available
  assert provider.source != SolverContactTactileProvider.source
  assert provider.contact_threshold_m == pytest.approx(5.0e-5)
  assert provider.bool_count_threshold == 0
  assert provider.release_debounce_steps == 10
  assert provider.layout.count == 350
  assert sample.link_names == FINGERTIP_LINK_NAMES
  assert provider.probe_depth.shape == (350,)
  assert not provider.probe_contact.any()


def test_record_validate_and_manifest(tmp_path) -> None:
  simulation = ArmHandSimulation()
  config = WorkcellConfig(
    cameras=(), tactile_provider=SolverContactTactileProvider.source
  )
  output = tmp_path / "episode.h5"
  with EpisodeRecorder(output, simulation, config, metadata={"test": True}) as recorder:
    recorder.record_initial()
    for _ in range(30):
      simulation.step()
      recorder.observe(simulation, "smoke")
    recorder.set_outcome({"success": True})
  report = validate_episode(output)
  assert report.valid
  assert report.state_samples == 7
  assert output.with_suffix(".json").is_file()
  with h5py.File(output, "r") as file:
    assert file.attrs["tactile_source"] == "solver_contact_proxy_v1"
    assert not bool(file["tactile_proxy"].attrs["is_genesis_probe_truth"])
    assert file["state/qpos"].shape[1] == simulation.model.nq
    assert file["contacts/frame_count"].shape[0] == report.state_samples


def test_record_bimanual_genesis_probe_stream(tmp_path) -> None:
  simulation = ArmHandSimulation()
  config = WorkcellConfig(
    cameras=(), tactile_provider=GenesisProbeTactileProvider.source
  )
  output = tmp_path / "genesis_episode.h5"
  with EpisodeRecorder(output, simulation, config) as recorder:
    recorder.record_initial()
    for _ in range(10):
      simulation.step()
      recorder.observe(simulation, "smoke")
  report = validate_episode(output)
  assert report.valid
  assert not report.warnings
  with h5py.File(output, "r") as file:
    assert file.attrs["tactile_source"] == GenesisProbeTactileProvider.source
    assert bool(file["tactile_genesis"].attrs["is_genesis_probe_truth"])
    assert file["tactile_genesis/probe_depth"].shape == (
      report.state_samples,
      350,
    )


def test_link_sample_comparison_metrics() -> None:
  count = len(FINGERTIP_LINK_NAMES)

  def sample(force: float, active: bool) -> LinkTactileSample:
    return LinkTactileSample(
      timestamp=0.0,
      source="test",
      link_names=FINGERTIP_LINK_NAMES,
      contact=np.full(count, active),
      normal_force=np.full(count, force),
      contact_count=np.full(count, int(active), dtype=np.int32),
      force_world=np.zeros((count, 3)),
      torque_world=np.zeros((count, 3)),
      force_local=np.zeros((count, 3)),
      centroid_world=np.zeros((count, 3)),
    )

  metrics = compare_link_samples([sample(1.0, True)], [sample(1.5, True)])
  assert metrics["normal_force_rmse_n"] == pytest.approx(0.5)
  assert metrics["contact_f1"] == pytest.approx(1.0)
