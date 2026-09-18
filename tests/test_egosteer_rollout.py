from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
from kaihand_tactile_env.workcell.simulation import ArmHandSimulation

_ROLLOUT = run_path(
  str(
    Path(__file__).parents[1]
    / "scripts"
    / "workcell"
    / "run_egosteer_policy.py"
  )
)
AdaptiveFreeCloseForce = _ROLLOUT["AdaptiveFreeCloseForce"]
StablePlacementDetector = _ROLLOUT["StablePlacementDetector"]
_parse_args = _ROLLOUT["_parse_args"]
_model_action_horizon = _ROLLOUT["_model_action_horizon"]
_model_camera_names = _ROLLOUT["_model_camera_names"]
_validated_prediction = _ROLLOUT["_validated_prediction"]


def _finger_force_ranges(
  simulation: ArmHandSimulation, finger: str
) -> tuple[np.ndarray, np.ndarray]:
  actuator_ids = np.array(
    [
      actuator_id
      for name, actuator_id in simulation._hand_actuators["right"].items()
      if f"_{finger}_" in name
    ],
    dtype=np.int32,
  )
  return actuator_ids, simulation.model.actuator_forcerange[actuator_ids].copy()


def test_rollout_defaults_match_training_action_horizon_and_grasp_controller() -> None:
  args = _parse_args([])

  assert args.execute_steps == 5
  assert args.adaptive_grasp_force
  assert args.free_close_force_limit == pytest.approx(0.82)
  assert args.free_close_cutoff_m == pytest.approx(0.002)
  assert args.free_close_min_closure == pytest.approx(0.35)
  assert args.hold_final_seconds == pytest.approx(3.0)
  assert args.jpeg_quality == 95
  assert args.record
  assert (args.review_width, args.review_height) == (1920, 1080)
  assert (args.review_render_width, args.review_render_height) == (640, 480)


@pytest.mark.parametrize("horizon", [5, 20, 32, 50])
def test_model_declares_prediction_horizon(horizon) -> None:
  assert _model_action_horizon({"action_horizon": horizon}) == horizon


@pytest.mark.parametrize("value", [None, 0, -1, True, 3.5, "32"])
def test_invalid_model_horizon_is_rejected(value) -> None:
  with pytest.raises(RuntimeError, match="action_horizon"):
    _model_action_horizon({"action_horizon": value})


def test_prediction_must_match_advertised_horizon() -> None:
  prediction = np.zeros((20, 48))
  assert _validated_prediction(prediction, horizon=20, action_dim=48) is prediction
  with pytest.raises(RuntimeError, match="shape"):
    _validated_prediction(prediction, horizon=32, action_dim=48)


def test_camera_contract_keeps_legacy_and_accepts_optional_wrists() -> None:
  assert _model_camera_names({}) == ("head",)
  assert _model_camera_names({"cameras": ["head"]}) == ("head",)
  assert _model_camera_names(
    {"observation_contract": {"cameras": ["head", "right_wrist"]}}
  ) == ("head", "right_wrist")
  assert _model_camera_names(
    {"observation_contract": {"cameras": ["head", "left_wrist"]}}
  ) == ("head", "left_wrist")
  assert _model_camera_names(
    {
      "observation_contract": {
        "cameras": ["head", "left_wrist", "right_wrist"]
      }
    }
  ) == ("head", "left_wrist", "right_wrist")
  with pytest.raises(RuntimeError, match="supports cameras"):
    _model_camera_names({"cameras": ["right_wrist"]})


def test_stable_placement_uses_collection_terminal_thresholds() -> None:
  detector = StablePlacementDetector(timestep=0.002)

  for _ in range(49):
    assert not detector.update(
      linear_speed=0.019,
      angular_speed=0.19,
      placed_in_box=True,
      grasp_active=False,
    )

  assert detector.update(
    linear_speed=0.019,
    angular_speed=0.19,
    placed_in_box=True,
    grasp_active=False,
  )
  assert detector.required_steps == 50


@pytest.mark.parametrize(
  ("linear_speed", "angular_speed", "placed_in_box", "grasp_active"),
  (
    (0.02, 0.0, True, False),
    (0.0, 0.2, True, False),
    (0.0, 0.0, False, False),
    (0.0, 0.0, True, True),
  ),
)
def test_stable_placement_resets_when_a_condition_is_not_met(
  linear_speed: float,
  angular_speed: float,
  placed_in_box: bool,
  grasp_active: bool,
) -> None:
  detector = StablePlacementDetector(timestep=0.002)
  for _ in range(25):
    detector.update(
      linear_speed=0.0,
      angular_speed=0.0,
      placed_in_box=True,
      grasp_active=False,
    )

  assert not detector.update(
    linear_speed=linear_speed,
    angular_speed=angular_speed,
    placed_in_box=placed_in_box,
    grasp_active=grasp_active,
  )
  assert detector.stable_steps == 0



def test_free_close_force_boosts_distant_fingers_and_restores_ranges() -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False, scene="pick-place")
  simulation.reset()
  thumb_ids, baseline_thumb = _finger_force_ranges(simulation, "thumb")
  index_ids, baseline_index = _finger_force_ranges(simulation, "index")
  controller = AdaptiveFreeCloseForce(
    simulation,
    enabled=True,
    force_limit=0.82,
    distance_cutoff=0.002,
    minimum_closure=0.35,
  )

  controller.before_physics_step(closure=1.0, grasp_active=False)

  assert controller.active
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[thumb_ids], baseline_thumb
  )
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[index_ids],
    np.broadcast_to((-0.82, 0.82), baseline_index.shape),
  )

  controller.before_physics_step(closure=0.0, grasp_active=False)

  assert not controller.active
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[index_ids], baseline_index
  )

  controller.before_physics_step(closure=1.0, grasp_active=False)
  controller.before_physics_step(closure=1.0, grasp_active=True)

  assert not controller.active
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[index_ids], baseline_index
  )


def test_disabled_free_close_force_leaves_model_unchanged() -> None:
  simulation = ArmHandSimulation(add_genesis_probes=False, scene="pick-place")
  simulation.reset()
  index_ids, baseline_index = _finger_force_ranges(simulation, "index")
  controller = AdaptiveFreeCloseForce(
    simulation,
    enabled=False,
    force_limit=0.82,
    distance_cutoff=0.002,
    minimum_closure=0.35,
  )

  controller.before_physics_step(closure=1.0, grasp_active=False)

  assert not controller.active
  np.testing.assert_allclose(
    simulation.model.actuator_forcerange[index_ids], baseline_index
  )
