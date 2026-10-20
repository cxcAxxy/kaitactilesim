"""Task-scoped subprocess bindings to native closed-loop rollout engines."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.tactile_deployment import load_manifest, sha256_file
from kaihand_tactile_env.shared.tactile_task_contracts import (
  SHARED_TACTILE_TASKS,
  task_contract,
)
from tactile_policy_worker import pack, unpack

ROOT = Path(__file__).resolve().parents[2]


class TactileWorker:
  def __init__(self, manifest_path, manifest, output, timeout=600):
    self.timeout = timeout
    self.log = (output / "model_worker.log").open("x")
    runtime = manifest["runtime"]
    environment = dict(os.environ)
    environment.update(runtime.get("env", {}))
    # Never let the simulator's or another model's openpi shadow this runtime.
    project = Path(runtime["project_root"])
    environment["PYTHONPATH"] = os.pathsep.join(map(str, (
      ROOT / "src", project / "src", project / "packages/openpi-client/src")))
    environment.setdefault("OMP_NUM_THREADS", "8")
    self.process = None
    self.closed = False
    try:
      self.process = subprocess.Popen([runtime["python"], str(ROOT / "scripts/workcell/tactile_policy_worker.py"),
        "--deployment-manifest", str(manifest_path)], cwd=project, env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
        text=True, bufsize=1, start_new_session=True)
      response = self._read()
      if "ready" not in response:
        raise RuntimeError("model worker sent no ready handshake")
      self.metadata = response["ready"]
    except BaseException:
      self.close()
      raise

  def _read(self):
    with selectors.DefaultSelector() as selector:
      selector.register(self.process.stdout, selectors.EVENT_READ)
      if not selector.select(self.timeout):
        raise TimeoutError(f"tactile model worker timed out after {self.timeout}s; see model_worker.log")
      line = self.process.stdout.readline()
    if not line:
      raise RuntimeError(f"tactile model worker exited ({self.process.poll()}); see model_worker.log")
    return unpack(json.loads(line))

  def get_server_metadata(self):
    return self.metadata

  def infer(self, request):
    self.process.stdin.write(json.dumps(pack(request), allow_nan=False) + "\n")
    self.process.stdin.flush()
    return self._read()

  def close(self):
    if self.closed:
      return
    self.closed = True
    process = self.process
    try:
      if process is not None and process.poll() is None:
        try:
          process.stdin.write('{"close":true}\n')
          process.stdin.flush()
          process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
          try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
            try:
              os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
              pass
            process.wait(timeout=5)
          except ProcessLookupError:
            process.wait(timeout=5)
    finally:
      streams = (self.log,) if process is None else (process.stdin, process.stdout, self.log)
      for stream in streams:
        if stream is not None:
          stream.close()


class PokerTactileBinding:
  def load_manifest(self, path):
    self.path, self.manifest = load_manifest(path, verify_files=True)
    if self.manifest["task"] != "poker-draw":
      raise ValueError("poker rollout requires a poker-draw deployment")
    self.action_start_index = self.manifest["action_start_index"]
    return self.path, self.manifest

  def diagnostic_reasons(self, interpolation):
    if interpolation != self.manifest["required_action_interpolation"]:
      raise ValueError("tactile joint controller v1 requires zoh interpolation")
    # Initial integrations remain explicitly diagnostic until full acceptance.
    return ["tactile_native_horizon_initial_integration"]

  def report_identity(self):
    manifest = self.manifest
    guidance = manifest.get("guidance", {})
    enabled = bool(guidance.get("variant") == "pi05+touchguidance" and guidance.get("scale", 0) > 0 and guidance.get("guidance_steps", 0) > 0)
    report_guidance = dict(enabled=enabled, steps=guidance.get("guidance_steps", 0) if enabled else 0,
                           scale=guidance.get("scale", 0) if enabled else 0,
                           variant=guidance.get("variant", "not_applicable"))
    return dict(inference_backend="isolated-local-worker", model_family=manifest["model_family"],
      prediction_horizon=manifest["prediction_horizon"], action_start_index=self.action_start_index,
      controller=manifest["controller_version"], codec_version=manifest["codec_version"],
      deployment_manifest_sha256=sha256_file(self.path), guidance=report_guidance,
      tactile_observation_supplied=True,
      tactile_sent_to_model=manifest["model_family"] == "ftp1" or enabled,
      tactile_conditioning_enabled=manifest["model_family"] == "ftp1" or enabled,
      simulation_paused_during_inference=True,
      expert_used=manifest["model_family"] == "pi05+cpm" and enabled,
      diagnostic_only=True, formal_metrics_valid=False)

  def start(self, output):
    return TactileWorker(self.path, self.manifest, output)

  def validate_server(self, metadata, execute_steps):
    for key in ("model_family", "task", "prediction_horizon", "action_start_index", "action_dim", "control_hz", "deployment_id",
                "model_action_dim", "joint_names", "checkpoint_sha256", "execute_steps"):
      if metadata.get(key) != self.manifest[key]:
        raise RuntimeError(f"tactile worker handshake mismatch at {key}: {metadata.get(key)!r}")
    if metadata.get("deployment_manifest_sha256") != sha256_file(self.path):
      raise RuntimeError("tactile worker loaded a different manifest")
    if metadata.get("tactile_sent_to_model") is not True:
      raise RuntimeError("model worker did not advertise tactile input")
    if execute_steps != self.manifest["execute_steps"]:
      raise ValueError("execute_steps is frozen: prepare a separate deployment for a different K")
    return self.manifest["prediction_horizon"]

  def attach(self, simulation):
    from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
    from kaihand_tactile_env.shared.contact_tactile import (
      SolverDistributedTactileProvider,
    )
    from kaihand_tactile_env.shared.taskspace_recording import TaskspaceCapture
    self.provider = SolverDistributedTactileProvider(simulation.model,
      target_geom_names=("card_core_geom",), link_names=tuple(FINGERTIP_LINK_NAMES[5:]))
    self.taskspace = TaskspaceCapture(simulation)

  def request(self, simulation, images, state, seed, index, execute_steps):
    import mujoco
    sample = self.provider.read(simulation.data)
    maps = np.stack((sample.normal_taxel_force_n, sample.tangent_taxel_force_n[..., 0],
                     sample.tangent_taxel_force_n[..., 1]), axis=1).astype(np.float32)
    maps = maps[getattr(self, "_sample_indices", slice(None))]
    if maps.shape != (5, 3, 7, 5) or not np.isfinite(maps).all():
      raise RuntimeError("right-hand tactile sample must be finite [5,3,7,5]")
    _, wrists, _ = self.taskspace.read()
    camera_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, "head")
    if camera_id < 0:
      raise RuntimeError("model observation requires the head camera")
    world_T_head = np.eye(4)
    world_T_head[:3, :3] = simulation.data.cam_xmat[camera_id].reshape(3, 3)
    world_T_head[:3, 3] = simulation.data.cam_xpos[camera_id]
    return dict(state=state, head_image=images["head"], right_wrist_image=images["right_wrist"],
      images=images, tactile_map_raw=maps, world_T_head=world_T_head, world_T_wrist=wrists[1],
      prompt=self.manifest["instruction"], episode_seed=seed, request_index=index,
      seed=seed, index=index, execute_steps=execute_steps, sim_time=float(simulation.data.time))

  def request_log(self, request, response):
    return dict(tactile_map_abs_max_n=float(np.max(np.abs(request["tactile_map_raw"]))),
      tactile_sent_to_model=self.report_identity()["tactile_sent_to_model"],
      diagnostics=response.get("diagnostics", {}),
      tactile_map_raw=request["tactile_map_raw"].tolist(),
      world_T_head=request["world_T_head"].tolist(), world_T_wrist=request["world_T_wrist"].tolist(),
      episode_seed=request["episode_seed"], request_index=request["request_index"],
      action_start_index=self.action_start_index, observation_sim_time=request["sim_time"])

  def request_arrays(self, request):
    return {key: request[key] for key in ("tactile_map_raw", "world_T_head", "world_T_wrist")}

  def finalize_report(self, report):
    valid = report.get("physics_valid") is True and (
      report.get("status") in {"success", "task_not_completed"} or report.get("stop_reason") == "guard")
    valid = valid and not any(key.endswith("record_error") for key in report)
    report.update(native_success=bool(report.get("task_success")),
      valid_trial=valid, trial_validity="valid" if valid else "invalid",
      diagnostic_only=True, formal_metrics_valid=False)


class TaskTactileBinding(PokerTactileBinding):
  """FTP-1 shared-task binding; never relabel a poker worker as another task."""

  def load_manifest(self, path):
    self.path, self.manifest = load_manifest(path, verify_files=True)
    if self.manifest["task"] not in SHARED_TACTILE_TASKS or self.manifest["model_family"] != "ftp1":
      raise ValueError("shared tactile rollout requires a task-specific FTP-1 deployment")
    self.action_start_index = self.manifest["action_start_index"]
    return self.path, self.manifest

  def diagnostic_reasons(self, interpolation):
    return [*super().diagnostic_reasons(interpolation),
            "task_specific_tactile_acceptance_pending", "training_protocol_transfer_requires_acceptance"]

  def attach(self, simulation):
    import mujoco
    from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
    from kaihand_tactile_env.shared.contact_tactile import (
      SolverDistributedTactileProvider,
    )
    from kaihand_tactile_env.shared.taskspace_recording import TaskspaceCapture

    contract = task_contract(self.manifest["task"])
    right_links = tuple(FINGERTIP_LINK_NAMES[5:])
    if contract["tactile_provider"] == "native_sponge_flex":
      # Rigid geom filters cannot observe MuJoCo flex contacts (geom == -1).
      # Reuse the native read-only, flex-aware sensor; do not alter mechanics.
      from kaihand_tactile_env.tasks.vase_wipe.sponge import SpongeTactileProvider
      self.provider = getattr(simulation, "evaluation_tactile_provider", None)
      if not isinstance(self.provider, SpongeTactileProvider):
        raise RuntimeError("sponge tactile rollout requires its native flex-aware provider")
    else:
      body_id = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_BODY, contract["tactile_target_body"])
      if body_id < 0:
        raise RuntimeError("task tactile target body is absent")
      targets = tuple(simulation.model.geom(index).name for index in range(simulation.model.ngeom)
        if simulation.model.geom_bodyid[index] == body_id
        and (simulation.model.geom_contype[index] or simulation.model.geom_conaffinity[index]))
      self.provider = SolverDistributedTactileProvider(simulation.model,
        target_geom_names=targets, link_names=right_links)
    self._sample_indices = [self.provider.link_names.index(name) for name in right_links]
    self.taskspace = TaskspaceCapture(simulation)

  def report_identity(self):
    return {**super().report_identity(),
            "training_binding": self.manifest["source"]["training_binding"],
            "native_evaluation_protocol": self.manifest["native_evaluation_protocol"],
            "training_protocol_transfer": self.manifest["training_protocol_transfer"]}
