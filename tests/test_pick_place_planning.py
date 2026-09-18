from __future__ import annotations

import pytest
from kaihand_tactile_env.workcell.planning import (
  KnownStateGraspPlanner,
  PickPlaceExecutor,
)
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation


@pytest.mark.parametrize("seed", (56, 72, 80, 144, 163, 178))
def test_preflight_accepts_cartesian_place_regression_seeds(seed: int) -> None:
  simulation = ArmHandSimulation()
  simulation.reset(
    seed=seed,
    object_xy_jitter=0.01,
    object_yaw_jitter=0.05,
    randomized_objects=("cylinder",),
  )

  plan = KnownStateGraspPlanner(simulation).plan_pick_and_place("right")

  assert tuple(waypoint.phase for waypoint in plan.waypoints) == (
    "ready",
    "pregrasp",
    "approach",
    "lift",
    "transfer",
    "place",
  )
  transfer, place = plan.waypoints[-2:]
  assert transfer.end_effector_position[2] > place.end_effector_position[2]


def test_seed_56_executes_place_without_penetrating_box_walls() -> None:
  simulation = ArmHandSimulation()
  simulation.reset(
    seed=56,
    object_xy_jitter=0.01,
    object_yaw_jitter=0.05,
    randomized_objects=("cylinder",),
  )
  cylinder_geom_id = simulation.model.geom("cylinder_geom").id
  box_wall_ids = {
    simulation.model.geom(name).id
    for name in (
      "box_wall_x_pos",
      "box_wall_x_neg",
      "box_wall_y_pos",
      "box_wall_y_neg",
    )
  }
  penetrating_wall_contacts: list[tuple[str, float]] = []

  def observe(sim: ArmHandSimulation, phase: str) -> None:
    if phase not in {"place", "release"}:
      return
    for index in range(sim.data.ncon):
      contact = sim.data.contact[index]
      pair = {int(contact.geom1), int(contact.geom2)}
      if (
        cylinder_geom_id in pair
        and not pair.isdisjoint(box_wall_ids)
        and contact.dist < 0.0
      ):
        penetrating_wall_contacts.append((phase, float(contact.dist)))

  plan = KnownStateGraspPlanner(simulation).plan_pick_and_place("right")
  result = PickPlaceExecutor(simulation, observer=observe).execute(plan)

  assert result.placed_in_box
  assert not penetrating_wall_contacts
