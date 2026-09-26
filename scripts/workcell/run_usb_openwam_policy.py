#!/usr/bin/env python3
"""Run an OpenWAM KaiHand-USB checkpoint in the USB-insert simulator.

This runner speaks the native OpenWAM WebSocket protocol (``obs``/``action``)
and adapts OpenWAM's 29-D physical action
``[wrist_xyz, wrist_rot6d, right_hand_joints]`` to the KaiHand MuJoCo scene.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.egosteer_adapter import (
  _quaternion_wxyz_from_rotation,
)
from kaihand_tactile_env.shared.evaluation_video import EvaluationVideo
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  OpenWAMEvaluationPlots,
  load_openwam_reference,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert import config as usb_config
from kaihand_tactile_env.tasks.usb_insert.review_metrics import UsbPolicyOutcome
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion
from PIL import Image

CONTROL_HZ = 30
ACTION_DIM = 29
INSTRUCTION = (
  "Grasp the USB plug with the right hand, lift and align it with the "
  "upward-facing socket, insert it until seated, then release it and "
  "withdraw the hand."
)
RIGHT_HAND_JOINT_NAMES = RIGHT_HAND_ACTUATED_JOINT_NAMES


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
  value = np.asarray(value, dtype=np.float64)
  if value.shape != (6,) or not np.isfinite(value).all():
    raise ValueError(f"rot6d must be finite with shape (6,), got {value.shape}")
  first = value[:3]
  first_norm = np.linalg.norm(first)
  if first_norm < 1.0e-10:
    raise ValueError("OpenWAM action has a degenerate rotation")
  first = first / first_norm
  second = value[3:] - np.dot(first, value[3:]) * first
  second_norm = np.linalg.norm(second)
  if second_norm < 1.0e-10:
    raise ValueError("OpenWAM action has collinear rotation columns")
  second = second / second_norm
  return np.stack((first, second, np.cross(first, second)), axis=1)


def _rot6d_from_matrix(rotation: np.ndarray) -> np.ndarray:
  rotation = np.asarray(rotation, dtype=np.float64)
  return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def _world_from_wrist(simulation: ArmHandSimulation) -> np.ndarray:
  """Return the native right-hand base-link pose used by the training data."""
  site_id = int(simulation.model.site("hand_r_base_link_site").id)
  world_from_wrist = np.eye(4, dtype=np.float64)
  world_from_wrist[:3, :3] = np.asarray(
    simulation.data.site_xmat[site_id], dtype=np.float64
  ).reshape(3, 3)
  world_from_wrist[:3, 3] = np.asarray(
    simulation.data.site_xpos[site_id], dtype=np.float64
  )
  return world_from_wrist


def _control_from_native_wrist(simulation: ArmHandSimulation) -> np.ndarray:
  """Return the fixed control-site -> native hand-base transform."""
  position, rotation = simulation.current_pose_matrix("right")
  world_from_control = np.eye(4, dtype=np.float64)
  world_from_control[:3, :3] = rotation
  world_from_control[:3, 3] = position
  return np.linalg.inv(world_from_control) @ _world_from_wrist(simulation)


def _state_29(simulation: ArmHandSimulation) -> np.ndarray:
  wrist = _world_from_wrist(simulation)
  hand = np.asarray(
    [simulation.data.qpos[simulation._qpos_address[name]] for name in RIGHT_HAND_JOINT_NAMES],
    dtype=np.float32,
  )
  state = np.concatenate((wrist[:3, 3].astype(np.float32), _rot6d_from_matrix(wrist[:3, :3]), hand))
  if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
    raise RuntimeError("simulation state is not a finite 29-D KaiHand state")
  return state


def _png_b64(rgb: np.ndarray) -> str:
  buffer = io.BytesIO()
  Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
  return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenWAMClient:
  def __init__(self, url: str, *, timeout: float):
    self.url = url
    self.timeout = timeout
    self.ws = None
    self.contract = {}

  async def __aenter__(self):
    import websockets

    self.ws = await websockets.connect(
      self.url, max_size=32 * 1024 * 1024, ping_interval=None,
      open_timeout=self.timeout,
    )
    await self.ws.send(json.dumps({"type": "ping"}))
    self.contract = json.loads(await asyncio.wait_for(self.ws.recv(), self.timeout))
    if self.contract.get("type") != "pong":
      raise RuntimeError(f"OpenWAM server did not answer ping: {self.contract}")
    return self

  async def __aexit__(self, *_):
    if self.ws is not None:
      await self.ws.close()

  async def reset(self) -> None:
    assert self.ws is not None
    await self.ws.send(json.dumps({"type": "reset"}))
    response = json.loads(await asyncio.wait_for(self.ws.recv(), self.timeout))
    if response.get("type") != "reset_ack":
      raise RuntimeError(f"OpenWAM reset failed: {response}")

  async def infer(
    self, head: np.ndarray, right_wrist: np.ndarray, state: np.ndarray,
    *, required_steps: int,
  ) -> np.ndarray:
    assert self.ws is not None
    payload = {
      "type": "obs",
      "images": {
        "head_camera": _png_b64(head),
        "left_wrist_camera": None,
        "right_wrist_camera": _png_b64(right_wrist),
      },
      "prompt": INSTRUCTION,
      "state": state.astype(float).tolist(),
    }
    await self.ws.send(json.dumps(payload))
    response = json.loads(await asyncio.wait_for(self.ws.recv(), self.timeout))
    if response.get("type") == "error":
      raise RuntimeError(f"OpenWAM inference error: {response}")
    if response.get("type") != "action":
      raise RuntimeError(f"unexpected OpenWAM response: {response}")
    actions = np.asarray(response.get("action"), dtype=np.float64)
    if actions.ndim == 1 and actions.size == ACTION_DIM:
      actions = actions[None, :]
    if (
      actions.ndim != 2
      or actions.shape[0] < required_steps
      or actions.shape[1] != ACTION_DIM
      or not np.isfinite(actions).all()
    ):
      raise RuntimeError(
        f"expected at least {required_steps} OpenWAM actions with 29 values, "
        f"got {actions.shape}"
      )
    return actions


def _apply_action(simulation: ArmHandSimulation, action: np.ndarray) -> None:
  wrist = np.eye(4, dtype=np.float64)
  wrist[:3, 3] = action[:3]
  wrist[:3, :3] = _rot6d_to_matrix(action[3:9])
  control_from_wrist = _control_from_native_wrist(simulation)
  world_from_control = wrist @ np.linalg.inv(control_from_wrist)
  result = simulation.set_pose_target(
    "right", world_from_control[:3, 3], _quaternion_wxyz_from_rotation(world_from_control[:3, :3]),
  )
  if not result.success and (
    result.position_error > 0.03 or result.orientation_error > 0.35
  ):
    raise RuntimeError(
      "right arm IK failed: "
      f"position_error={result.position_error:.4f}m, "
      f"orientation_error={result.orientation_error:.4f}rad"
    )
  accepted = simulation.set_hand_joint_targets(RIGHT_HAND_JOINT_NAMES, action[9:29])
  if accepted != len(RIGHT_HAND_JOINT_NAMES):
    raise RuntimeError(f"right hand accepted {accepted}/{len(RIGHT_HAND_JOINT_NAMES)} targets")


def _guard_usb(outcome: UsbPolicyOutcome, simulation: ArmHandSimulation, *, penetration_guard: bool) -> None:
  state = outcome.state
  if state.socket_normal_load_n > usb_config.MAX_SOCKET_NORMAL_LOAD_N:
    raise RuntimeError("USB socket total normal load exceeded task limit")
  if state.axial_resistance_n > usb_config.MAX_SOCKET_AXIAL_FORCE_N:
    raise RuntimeError("USB socket axial resistance exceeded task limit")
  if state.wall_normal_load_n > usb_config.MAX_SOCKET_WALL_LOAD_N:
    raise RuntimeError("USB socket side-wall load exceeded task limit")
  if penetration_guard and state.maximum_socket_penetration_m > 0.0003:
    raise RuntimeError("USB socket penetration exceeded 0.3 mm")
  if float(simulation.object_pose("usb_plug")[2]) < 0.5:
    raise RuntimeError("USB plug fell below work surface")
  if not np.isfinite(simulation.data.qpos).all() or not np.isfinite(simulation.data.qvel).all():
    raise RuntimeError("nonfinite simulation state")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--server", default="ws://127.0.0.1:8848")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=10.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=5.0)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--response-timeout", type=float, default=600.0)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--reference-dataset",
    type=Path,
    help=(
      "LeRobot v3 dataset root used for training; recorded reviews compare "
      "against its reference episode"
    ),
  )
  parser.add_argument(
    "--reference-episode-index",
    type=int,
    default=0,
    help="LeRobot output episode used as the solid reference curves (default: 0)",
  )
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--disable-penetration-guard", action="store_true")
  args = parser.parse_args(argv)
  if args.execute_steps <= 0:
    parser.error("execute-steps must be positive")
  if args.reference_episode_index < 0:
    parser.error("reference-episode-index must be nonnegative")
  if args.record and args.reference_dataset is None:
    parser.error("--reference-dataset is required when --record is enabled")
  return args


async def run(args) -> dict:
  reference = None
  reference_dataset = None
  if args.record:
    reference_dataset = args.reference_dataset.expanduser().resolve()
    reference = load_openwam_reference(
      reference_dataset,
      output_episode_index=args.reference_episode_index,
      joint_names=RIGHT_HAND_JOINT_NAMES,
    )

  output = args.output_dir.expanduser().resolve()
  output.mkdir(parents=True, exist_ok=False)
  simulation = ArmHandSimulation(scene="usb-insert")
  simulation.reset(seed=args.seed)
  initial = initialize_for_insertion(
    simulation, seed=args.seed, xy_jitter_m=args.xy_jitter_mm / 1000.0,
    yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
  )
  outcome = UsbPolicyOutcome(simulation)
  cameras = (
    CameraConfig("head", width=320, height=240, rgb=True, depth=False, segmentation=False),
    CameraConfig("right_wrist", width=320, height=240, rgb=True, depth=False, segmentation=False),
  )
  report = {
    "task": "usb-insert",
    "controller": "openwam-eef29-v1",
    "server": args.server,
    "seed": args.seed,
    "instruction": INSTRUCTION,
    "execute_steps": args.execute_steps,
    "control_hz": CONTROL_HZ,
    "initial_randomization": initial,
    "reference_dataset": (
      None if reference_dataset is None else str(reference_dataset)
    ),
    "reference_episode_index": (
      None if reference is None else reference.output_episode_index
    ),
    "reference_source_hdf5": (
      None if reference is None else str(reference.source_hdf5)
    ),
  }
  recorder = None
  comparison = None
  comparison_last_time = None
  requests = 0
  action_steps = 0
  started = time.monotonic()
  primary_error = None
  primary_traceback = None
  finalization_failure = None

  def capture_comparison(timestamp: float) -> None:
    nonlocal comparison_last_time
    if comparison is None:
      return
    timestamp = float(timestamp)
    if (
      comparison_last_time is not None
      and timestamp <= comparison_last_time + 1.0e-10
    ):
      return
    comparison.capture(timestamp, _state_29(simulation), simulation=simulation)
    comparison_last_time = timestamp

  try:
    with WorkcellRenderer(simulation.model, cameras, visible_geom_groups=(0, 1, 2, 3, 4), shadows=False) as renderer:
      if args.record:
        recorder = EvaluationVideo(simulation, output / "review", fps=10, width=1920, height=1080,
                                   render_width=640, render_height=480, second_camera="global",
                                   include_model_wrist=True,
                                   metrics=outcome, heading="USB INSERT OPENWAM EVALUATION",
                                   metadata=report)
        assert reference is not None
        comparison = OpenWAMEvaluationPlots(
          reference, tactile_provider=recorder.provider
        )
        capture_comparison(0.0)
        recorder.capture(0, outcome.stage())
      async with OpenWAMClient(args.server, timeout=args.response_timeout) as client:
        report["server_contract"] = client.contract
        await client.reset()
        while simulation.data.time < args.max_sim_seconds and not outcome.success:
          if args.max_requests and requests >= args.max_requests:
            break
          images = {camera.name: renderer.capture(simulation.data, camera)["rgb"] for camera in cameras}
          actions = await client.infer(
            images["head"], images["right_wrist"], _state_29(simulation),
            required_steps=args.execute_steps,
          )
          requests += 1
          for action in actions[:args.execute_steps]:
            if simulation.data.time >= args.max_sim_seconds or outcome.success:
              break
            _apply_action(simulation, action)
            target_time = (action_steps + 1) / CONTROL_HZ
            while simulation.data.time < target_time - 1e-12:
              simulation.step()
              outcome.update(simulation)
              _guard_usb(outcome, simulation, penetration_guard=not args.disable_penetration_guard)
            action_steps += 1
            capture_comparison(float(simulation.data.time))
            if recorder is not None:
              recorder.capture(action_steps, outcome.stage())
          print(f"[openwam] request={requests} sim_t={simulation.data.time:.3f}s stage={outcome.stage()}", flush=True)
        report["status"] = "success" if outcome.success else "task_not_completed"
  except BaseException as error:
    primary_error = error
    primary_traceback = error.__traceback__
    report["status"] = "error"
    report["error"] = f"{type(error).__name__}: {error}"
  finally:
    artifact_errors = []

    def record_artifact_error(artifact: str, error: BaseException) -> None:
      nonlocal finalization_failure
      if finalization_failure is None:
        finalization_failure = error
      detail = f"{type(error).__name__}: {error}"
      artifact_errors.append({"artifact": artifact, "error": detail})
      if primary_error is None and len(artifact_errors) == 1:
        report["status"] = "error"
        report["error"] = detail

    if comparison is not None:
      try:
        # Preserve a terminal mid-control-step state when a safety guard fails.
        capture_comparison(float(simulation.data.time))
      except BaseException as error:
        record_artifact_error("comparison_terminal_capture", error)
    report["requests"] = requests
    report["action_steps"] = action_steps
    report["sim_seconds"] = float(simulation.data.time)
    try:
      report["evaluation"] = outcome.report()
    except BaseException as error:
      report["evaluation"] = None
      record_artifact_error("evaluation_report", error)
    report["wall_seconds"] = time.monotonic() - started
    if comparison is not None and comparison.sample_count:
      try:
        report["comparison_plots"] = comparison.finish(output / "review")
      except BaseException as error:
        record_artifact_error("comparison_plots", error)
    if recorder is not None:
      try:
        report["video"] = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=report.get("error"),
        )
      except BaseException as error:
        record_artifact_error("evaluation_video", error)
    if artifact_errors:
      report["artifact_errors"] = artifact_errors
    try:
      (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
      )
    except BaseException as error:
      record_artifact_error("summary", error)

  if primary_error is not None:
    raise primary_error.with_traceback(primary_traceback)
  if finalization_failure is not None:
    raise RuntimeError("OpenWAM evaluation artifact finalization failed") from finalization_failure
  return report


if __name__ == "__main__":
  asyncio.run(run(parse_args()))
