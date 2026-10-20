from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.shared.tactile_deployment import (
  JOINT_NAMES,
  TASK_SCHEMA,
  identity,
  prepare_manifest,
  sha256_file,
  validate_manifest,
)
from kaihand_tactile_env.shared.tactile_task_contracts import (
  SHARED_TACTILE_TASKS,
  ftp1_training_binding,
  task_contract,
)


def training_assets(tmp_path, task="bulb-screw", recipe="arm_relative_hand_absolute"):
  contract = task_contract(task)
  corpus = tmp_path / "corpus"
  corpus.mkdir()
  conversion = corpus / "conversion_manifest.json"
  conversion.write_text(json.dumps(dict(task=task, instruction=contract["instruction"],
    fps=30, ftp1_action_horizon=32, action_supervision=dict(
      contract="ftp1-native-future-measured-state-trajectory-v1",
      controller_targets_used=False, pi05_recorded_controller_target_equivalent=False))))
  config = tmp_path / "dataset_config.json"
  config.write_text(json.dumps(dict(datasets=[dict(name=contract["domain_name"], path=str(corpus), enabled=True)])))
  checkpoint = tmp_path / "checkpoint"
  checkpoint.mkdir()
  (checkpoint / "train_config.json").write_text(json.dumps(dict(
    dataset_config_path=str(config), action_joint_rep=recipe)))
  (checkpoint / "model.safetensors").write_bytes(b"test fixture, not trained weights")
  return checkpoint, config, conversion


def task_manifest(tmp_path, task="bulb-screw"):
  checkpoint, config, conversion = training_assets(tmp_path, task)
  binding = ftp1_training_binding(checkpoint, task)
  contract = task_contract(task)
  payload = dict(schema=TASK_SCHEMA, task=task, model_family="ftp1",
    checkpoint_path=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint / "model.safetensors"),
    source=dict(checkpoint=str(checkpoint), domain_name=contract["domain_name"],
      action_joint_rep=binding["action_joint_rep"], training_binding=binding),
    runtime=dict(project_root=str(tmp_path), python=sys.executable),
    prediction_horizon=32, model_action_dim=120, action_start_index=1, execute_steps=5,
    action_dim=27, control_hz=30, joint_names=JOINT_NAMES, instruction=contract["instruction"],
    observation_contract=dict(cameras=["head", "right_wrist"], image_shape_hwc=[240,320,3],
      tactile_sent_to_model=True, tactile_shape=[5,3,7,5], tactile_unit="N",
      **{key:contract[key] for key in ("tactile_target_body", "tactile_target_geometries", "tactile_provider")}),
    required_action_interpolation="zoh", action_representation="absolute-right-arm7-hand20-joints",
    controller_version="shared-tactile-direct-joints-v1", codec_version="ftp1-kaihand-task-v2",
    native_evaluation_protocol="TaskPolicyAdapter (current task completion semantics)",
    training_protocol_transfer="diagnostic only", diagnostic_only=True, formal_metrics_valid=False,
    files={str(path):sha256_file(path) for path in (checkpoint / "model.safetensors", config, conversion)})
  payload["deployment_id"] = identity(payload)
  return payload


@pytest.mark.parametrize("task", SHARED_TACTILE_TASKS)
def test_task_manifest_contracts_are_separate_from_poker(tmp_path, task):
  payload = task_manifest(tmp_path, task)
  assert validate_manifest(payload, verify_files=True)["task"] == task
  corrupted = copy.deepcopy(payload)
  corrupted["source"]["domain_name"] = "KaiHandSimPokerDraw"
  corrupted["deployment_id"] = identity(corrupted)
  with pytest.raises(ValueError, match="binding mismatch"):
    validate_manifest(corrupted)


@pytest.mark.parametrize("field,value", [("tactile_target_body", "card"),
  ("tactile_provider", "poker"), ("tactile_target_geometries", "card_core_geom")])
def test_task_manifest_rejects_wrong_tactile_target(tmp_path, field, value):
  payload = task_manifest(tmp_path)
  payload["observation_contract"][field] = value
  payload["deployment_id"] = identity(payload)
  with pytest.raises(ValueError, match="target/provider"):
    validate_manifest(payload)


def test_training_binding_rejects_relabelled_poker_corpus(tmp_path):
  checkpoint, _, conversion = training_assets(tmp_path)
  data = json.loads(conversion.read_text())
  data["task"] = "poker-draw"
  conversion.write_text(json.dumps(data))
  with pytest.raises(ValueError, match="task/instruction differs"):
    ftp1_training_binding(checkpoint, "bulb-screw")


@pytest.mark.parametrize("supervision", [None, "next_state", {},
  dict(contract="controller-targets", controller_targets_used=True)])
def test_task_manifest_rejects_unproven_action_supervision(tmp_path, supervision):
  payload = task_manifest(tmp_path)
  payload["source"]["training_binding"]["action_supervision"] = supervision
  payload["deployment_id"] = identity(payload)
  with pytest.raises(ValueError, match="future-measured-state action supervision"):
    validate_manifest(payload)


def test_cpm_nonpoker_fails_before_touching_model_assets(tmp_path):
  with pytest.raises(ValueError, match="task-native base policy"):
    prepare_manifest(task="bulb-screw", family="pi05+cpm", checkpoint=tmp_path / "missing",
      project=tmp_path, python=sys.executable)


def test_usb_ftp1_adapter_uses_native_scene_initialization_and_outcome():
  from kaihand_tactile_env.shared.policy_tasks import create_task_policy_adapter

  adapter = create_task_policy_adapter("usb-insert", 17, guard_mode="observe")
  initialization = adapter.initial_randomization

  assert adapter.simulation.scene == "usb-insert"
  assert adapter.object_name == "usb_plug"
  assert adapter.instruction == task_contract("usb-insert")["instruction"]
  assert initialization["seed"] == 17
  assert initialization["xy_jitter_m"] == pytest.approx(0.010)
  assert initialization["yaw_jitter_rad"] == pytest.approx(np.deg2rad(5.0))
  assert np.max(np.abs(initialization["offset_xy_m"])) <= 0.010
  assert abs(initialization["yaw_offset_rad"]) <= np.deg2rad(5.0)
  assert initialization["pickup_joint_wrap"]["physical_pose_changed"] is False
  assert adapter.stage() == adapter.usb_outcome.stage() == "pickup"

  before = float(adapter.simulation.data.time)
  adapter.step()
  assert float(adapter.simulation.data.time) == pytest.approx(
    before + adapter.simulation.timestep
  )
  assert adapter.state is adapter.usb_outcome.state
  assert adapter.success is adapter.usb_outcome.success is False
  snapshot = adapter.snapshot()
  assert snapshot["insertion_depth_m"] == adapter.state.insertion_depth_m
  assert snapshot["bottom_out_confirmed"] is False
  assert snapshot["seated"] is False

  metadata = adapter.metadata()
  assert metadata["native_success_requires_release_and_withdraw"] is False
  assert metadata["training_prompt_requires_release_and_withdraw"] is True
  report = adapter.report()
  assert report["initial_randomization"] == initialization
  assert report["usb_policy_outcome"]["version"] == "usb-insertion-outcome-v1"


def test_usb_ftp1_adapter_preserves_native_policy_guard_thresholds():
  from dataclasses import replace

  from kaihand_tactile_env.shared.evaluation_guards import evaluation_guard_report
  from kaihand_tactile_env.shared.policy_tasks import (
    USB_INSERT_POLICY_PENETRATION_GUARD_M,
    create_task_policy_adapter,
  )
  from kaihand_tactile_env.tasks.usb_insert import config

  adapter = create_task_policy_adapter("usb-insert", 0, guard_mode="observe")
  adapter.state = replace(
    adapter.state,
    socket_normal_load_n=config.MAX_SOCKET_NORMAL_LOAD_N + 0.1,
    axial_resistance_n=config.MAX_SOCKET_AXIAL_FORCE_N + 0.1,
    wall_normal_load_n=config.MAX_SOCKET_WALL_LOAD_N + 0.1,
    maximum_socket_penetration_m=USB_INSERT_POLICY_PENETRATION_GUARD_M + 1e-6,
  )

  adapter._guard()

  events = {row["kind"]: row for row in evaluation_guard_report(
    adapter.simulation
  )["guard_events"]}
  assert events["usb_socket_normal_load"]["threshold"] == (
    config.MAX_SOCKET_NORMAL_LOAD_N
  )
  assert events["usb_socket_axial_resistance"]["threshold"] == (
    config.MAX_SOCKET_AXIAL_FORCE_N
  )
  assert events["usb_socket_wall_load"]["threshold"] == (
    config.MAX_SOCKET_WALL_LOAD_N
  )
  assert events["usb_socket_penetration"]["threshold"] == pytest.approx(0.0003)


def test_unified_deploy_and_evaluate_route_task_manifest(tmp_path, capsys):
  from kaihand_tactile_env.pipeline.deployment import (
    build_prepare_plan,
    load_deployment_manifest,
  )
  from kaihand_tactile_env.pipeline.runtime import RuntimeProfile
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/evaluate"))
  import evaluate

  payload = task_manifest(tmp_path)
  path = tmp_path / "deployment.json"
  path.write_text(json.dumps(payload))
  _, loaded, adapter = load_deployment_manifest(path)
  assert loaded["task"] == "bulb-screw"
  assert adapter.manifest_schemas == (TASK_SCHEMA,)
  profile = RuntimeProfile(name="test", python=Path(sys.executable), project_root=tmp_path,
    env={}, config_path=tmp_path / "runtime.toml", config_sha256="a" * 64)
  plan = build_prepare_plan("bulb-screw", "ftp1", checkpoint=tmp_path / "checkpoint",
    output_dir=tmp_path / "deployment", profile=profile)
  assert plan.command[plan.command.index("--task") + 1] == "bulb-screw"
  assert evaluate.main(["--task", "bulb-screw", "--model-family", "ftp1",
    "--deployment-manifest", str(path), "--output-dir", str(tmp_path / "evaluation"),
    "--num-trials", "1", "--video-count", "0", "--dry-run"]) == 0
  description = json.loads(capsys.readouterr().out)
  assert description["execute_steps"] == 5
  assert description["command"][1].endswith("evaluate_task_tactile_policy_batch.py")


def test_shared_tactile_runner_accepts_usb_native_evaluation_core(
  tmp_path,
):
  workcell = Path(__file__).resolve().parents[1] / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  from run_shared_task_pi05_policy import TASKS as SHARED_PI05_TASKS
  from run_shared_task_tactile_policy import parse_tactile_args

  assert "usb-insert" in SHARED_PI05_TASKS
  args = parse_tactile_args([
    "--task", "usb-insert",
    "--deployment-manifest", str(tmp_path / "deployment.json"),
    "--output-dir", str(tmp_path / "trial"),
    "--seed", "0",
  ])
  assert args.task == "usb-insert"
  assert args.server == "local-worker"
  assert args.action_interpolation == "zoh"


@pytest.mark.parametrize("task", SHARED_TACTILE_TASKS)
def test_native_task_tactile_request_is_synchronized_and_read_only(task):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/workcell"))
  from kaihand_tactile_env.shared.policy_tasks import create_task_policy_adapter
  from run_usb_pi05_policy import right_joint_state
  from tactile_policy_binding import TaskTactileBinding

  adapter = create_task_policy_adapter(task, 0, guard_mode="observe")
  simulation = adapter.simulation
  binding = TaskTactileBinding()
  binding.manifest = dict(task=task, instruction=task_contract(task)["instruction"])
  binding.attach(simulation)
  state = right_joint_state(simulation, JOINT_NAMES)
  before = (simulation.data.qpos.copy(), simulation.data.qvel.copy(), float(simulation.data.time))
  images = {key: np.zeros((240,320,3), dtype=np.uint8) for key in ("head", "right_wrist")}
  request = binding.request(simulation, images, state, 0, 1, 5)
  np.testing.assert_array_equal(simulation.data.qpos, before[0])
  np.testing.assert_array_equal(simulation.data.qvel, before[1])
  assert float(simulation.data.time) == before[2] == request["sim_time"]
  assert request["tactile_map_raw"].shape == (5,3,7,5)
  assert request["prompt"] == task_contract(task)["instruction"]
  assert np.isfinite(request["tactile_map_raw"]).all()
  if task in {"vase-wipe", "sponge-grasp"}:
    assert binding.provider is simulation.evaluation_tactile_provider
  else:
    bodies = {simulation.model.body(int(simulation.model.geom_bodyid[index])).name
              for index in binding.provider._target_geom_ids}
    assert bodies == {task_contract(task)["tactile_target_body"]}
