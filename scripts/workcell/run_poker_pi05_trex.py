#!/usr/bin/env python3
"""Run the poker pi0.5 checkpoint with its late-flow tactile expert locally."""

# Configure the source path and EGL before importing simulator modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = (Path(os.environ["KAIHAND_SIM_ROOT"]) if "KAIHAND_SIM_ROOT" in os.environ
        else Path(__file__).resolve().parents[2])
if not (ROOT / "src/kaihand_tactile_env").is_dir():
  raise RuntimeError(f"cannot find kaitactilesim source under {ROOT}")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/workcell"))
EGL_VENDOR = ROOT / "scripts/collect/nvidia_egl_vendor.json"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KAIHAND_RENDER_BACKEND", "hardware")
if EGL_VENDOR.is_file():
  os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(EGL_VENDOR))

import numpy as np
from kaihand_tactile_env.shared.config import (
  FINGERTIP_LINK_NAMES,
  CameraConfig,
  default_model_path,
)
from kaihand_tactile_env.shared.contact_tactile import SolverDistributedTactileProvider
from kaihand_tactile_env.shared.policy_cameras import (
  image_shape_hwc,
  policy_camera_names,
)
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import PokerPolicyController
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_poker_pi05_policy import (
  guard_poker_state,
  load_deployment_manifest,
  outcome_stage,
)
from run_usb_pi05_policy import (
  apply_action,
  joint_limits,
  right_joint_state,
  validated_actions,
)

CONTROL_HZ = 30
CONTROLLER = "direct-30hz-pi05-absolute-joint-v1"
PENETRATION_GUARD_THRESHOLD_M = 0.0006
DEFAULT_OPENPI_ROOT = Path("/cpfs_infra/user/chenxianchi/yidu/openpi-main")
DEFAULT_BASE_PYTORCH = Path(
  "/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_25k"
)
DEFAULT_EXPERT_CHECKPOINT = Path(
  "/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/tactile_expert_runs/"
  "late4_right_resume_3000_to_10000_20260923_203128/best.pt"
)
DEFAULT_TACTILE_ROOT = Path(
  "/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/0920_200"
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
  parser.add_argument("--base-pytorch", type=Path, default=DEFAULT_BASE_PYTORCH)
  parser.add_argument("--expert-checkpoint", type=Path, default=DEFAULT_EXPERT_CHECKPOINT)
  parser.add_argument("--tactile-root", type=Path, default=DEFAULT_TACTILE_ROOT)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--flow-seed", type=int,
                      help="Diffusion-noise seed; defaults to the episode seed")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter-mm", type=float, default=4.0)
  parser.add_argument("--yaw-jitter-deg", type=float, default=0.5)
  parser.add_argument("--execute-steps", type=int, default=30)
  parser.add_argument(
    "--tactile-refine-every", type=int, default=5,
    help="Refresh the tactile-conditioned remainder every N executed actions; "
         "the next full pi0.5 inference begins at execute-steps",
  )
  parser.add_argument("--max-requests", type=int, default=0)
  parser.add_argument("--max-sim-seconds", type=float, default=50.0)
  parser.add_argument("--success-hold-seconds", type=float, default=0.10)
  parser.add_argument(
    "--full-duration-evaluation", action="store_true",
    help="Diagnostic rollout: continue after card/table penetration or a dropped card; only nonfinite states abort",
  )
  parser.add_argument(
    "--disable-penetration-guard",
    action="store_true",
    help=(
      "Diagnostic-only: continue after supported card/table penetration exceeds "
      "0.6 mm; nonfinite-state and fallen-card guards remain enabled"
    ),
  )
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--save-raw", action=argparse.BooleanOptionalAction, default=False,
    help="Save a synchronized Raw HDF5 episode and five-finger force plot",
  )
  parser.add_argument("--record-fps", type=int, choices=(5, 10), default=10)
  parser.add_argument("--reference-dataset", type=Path)
  parser.add_argument("--reference-episode-index", type=int, default=0)
  parser.add_argument("--review-width", type=int, default=1920)
  parser.add_argument("--review-height", type=int, default=1080)
  parser.add_argument("--review-render-width", type=int, default=640)
  parser.add_argument("--review-render-height", type=int, default=480)
  parser.add_argument(
    "--review-second-camera",
    choices=("global", "right_wrist", "overhead"),
    default="global",
  )
  parser.add_argument(
    "--show-model-wrist-in-review",
    action=argparse.BooleanOptionalAction,
    default=True,
  )
  parser.add_argument(
    "--save-first-request", action=argparse.BooleanOptionalAction, default=True
  )
  args = parser.parse_args(argv)
  if args.full_duration_evaluation:
    args.disable_penetration_guard = True
  if not np.isfinite(args.success_hold_seconds) or args.success_hold_seconds <= 0:
    parser.error("success-hold-seconds must be positive and finite")
  if args.seed < 0 or args.max_requests < 0 or args.execute_steps <= 0:
    parser.error("seed/requests must be nonnegative; execute-steps must be positive")
  if args.tactile_refine_every <= 0:
    parser.error("tactile-refine-every must be positive")
  if args.flow_seed is not None and args.flow_seed < 0:
    parser.error("flow-seed must be nonnegative")
  if args.device != "cuda":
    parser.error("this bfloat16 pi0.5 tactile runner requires --device cuda")
  for key in ("xy_jitter_mm", "yaw_jitter_deg", "max_sim_seconds"):
    value = getattr(args, key)
    if not np.isfinite(value) or value < 0 or (
      key == "max_sim_seconds" and value == 0
    ):
      parser.error(f"{key} must be finite and nonnegative")
  if args.xy_jitter_mm > 5.0 or args.yaw_jitter_deg > 1.0:
    parser.error("poker randomization is limited to 5 mm XY and 1 degree yaw")
  if args.show_model_wrist_in_review and args.review_second_camera == "right_wrist":
    parser.error("review second camera duplicates the model wrist view")
  if min(
    args.review_width,
    args.review_height,
    args.review_render_width,
    args.review_render_height,
  ) <= 0:
    parser.error("review dimensions must be positive")
  if args.review_width % 2 or args.review_height % 2:
    parser.error("review output dimensions must be even")
  return args


def validate_deployment(deployment: dict, execute_steps: int) -> int:
  """Use the original deployment only for the shared task/action contract."""
  horizon = int(deployment["prediction_horizon"])
  if horizon != 30 or deployment["model_action_dim"] != 32 or deployment["action_dim"] != 27:
    raise RuntimeError("tactile policy requires a [30,32] model and 27 physical joints")
  if not 1 <= execute_steps <= horizon or deployment["control_hz"] != CONTROL_HZ:
    raise RuntimeError("incompatible execute_steps or control rate")
  observation = deployment["observation_contract"]
  if policy_camera_names(observation) != ("head", "right_wrist"):
    raise RuntimeError("tactile training used head and right_wrist cameras")
  if image_shape_hwc(observation) != (240, 320, 3):
    raise RuntimeError("tactile training used 240x320 RGB images")
  if observation.get("tactile_sent_to_model") is not False:
    raise RuntimeError("expected the original, tactile-free pi0.5 deployment manifest")
  if len(deployment["joint_names"]) != 27:
    raise RuntimeError("expected 27 right-side joints")
  return horizon


class OnlineTactile:
  """Read the same card-pad taxel signal used to build the training sidecar."""

  def __init__(self, simulation):
    self.simulation = simulation
    self.provider = SolverDistributedTactileProvider(
      simulation.model,
      target_geom_names=("card_core_geom",),
      link_names=tuple(FINGERTIP_LINK_NAMES[5:]),
    )
    self.history: deque[np.ndarray] = deque(maxlen=16)
    self.current_maps: np.ndarray | None = None
    self.capture()  # Frame zero; early history repeats this frame.

  def capture(self) -> None:
    sample = self.provider.read(self.simulation.data)
    normal = sample.normal_taxel_force_n
    tangent = sample.tangent_taxel_force_n
    maps = np.stack((normal, tangent[..., 0], tangent[..., 1]), axis=1)
    # Match sidecar F6: sum local taxel force and r x force about the local
    # fingertip-link origin. SolverContactTactileProvider's torque is different.
    normal_axis = sample.normal_axis_local[:, None, None, :]
    tangent_basis = sample.tangent_basis_local
    force_local = (
      normal[..., None] * normal_axis
      + tangent[..., 0, None] * tangent_basis[:, 0, None, None, :]
      + tangent[..., 1, None] * tangent_basis[:, 1, None, None, :]
    )
    positions = self.provider.taxel_positions_local_m.reshape(5, 7, 5, 3)
    force = force_local.sum(axis=(1, 2))
    torque = np.cross(positions, force_local).sum(axis=(1, 2))
    f6 = np.concatenate((force, torque), axis=-1).astype(np.float32)
    maps = maps.astype(np.float32)
    if not np.isfinite(f6).all() or not np.isfinite(maps).all():
      raise RuntimeError("nonfinite online tactile observation")
    self.history.append(f6)
    self.current_maps = maps

  def observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = self.history[0]
    history = [first] * (16 - len(self.history)) + list(self.history)
    return np.stack(history), self.history[-1], self.current_maps.copy()


class LocalTactilePolicy:
  """Frozen converted pi0.5 plus the selected tactile checkpoint."""

  def __init__(self, args, deployment: dict):
    openpi_src = args.openpi_root.expanduser().resolve() / "src"
    if not (openpi_src / "openpi/tactile_expert/model.py").is_file():
      raise FileNotFoundError(f"tactile OpenPI source missing: {openpi_src}")
    sys.path.insert(0, str(openpi_src))
    import safetensors.torch
    import torch
    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.tactile_expert.model import TactilePI05, TactileVelocityExpert
    from openpi.tactile_vqvae.data.pi05_trex import Pi05TrexF6Stats
    from openpi.tactile_vqvae.data.tactile_map import TactileMapStats

    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is unavailable to the tactile PyTorch policy")
    self.torch = torch
    self.Observation = Observation
    self.device = torch.device(args.device)
    base_path = args.base_pytorch.expanduser().resolve()
    expert_path = args.expert_checkpoint.expanduser().resolve()
    tactile_root = args.tactile_root.expanduser().resolve()
    expert_blob = torch.load(expert_path, map_location="cpu", weights_only=False)
    if expert_blob.get("format") != "poker_pi05_tactile_velocity_v1":
      raise RuntimeError("unsupported tactile expert checkpoint")
    if expert_blob.get("data_normalization") != "pi05_quantile_q01_q99":
      raise RuntimeError("tactile expert uses a different action normalization")
    saved_paths = expert_blob["paths"]
    for key, selected in (("base_pytorch", base_path), ("tactile_root", tactile_root)):
      if Path(saved_paths[key]).resolve() != selected:
        raise RuntimeError(f"expert checkpoint {key} does not match {selected}")
    norm_path = (
      Path(deployment["checkpoint_path"]) / "assets"
      / deployment["normalizer_asset_id"] / "norm_stats.json"
    ).resolve()
    if norm_path != Path(saved_paths["norm_stats"]).resolve():
      raise RuntimeError("expert and base deployment use different normalizers")
    with norm_path.open() as f:
      stats = json.load(f)["norm_stats"]
    self.state_q01 = np.asarray(stats["state"]["q01"], dtype=np.float32)
    self.state_q99 = np.asarray(stats["state"]["q99"], dtype=np.float32)
    self.action_q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
    self.action_q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
    if any(array.shape != (27,) for array in (
      self.state_q01, self.state_q99, self.action_q01, self.action_q99
    )):
      raise RuntimeError("expected 27-D state/action normalizer")
    self.f6_stats = Pi05TrexF6Stats.from_dataset_root(tactile_root)
    self.map_stats = TactileMapStats.from_root(tactile_root)
    f6_blob = torch.load(saved_paths["f6_checkpoint"], map_location="cpu", weights_only=False)
    map_blob = torch.load(saved_paths["map_checkpoint"], map_location="cpu", weights_only=False)
    f6_config = f6_blob["config"]
    if f6_config["window"] != 16 or f6_config["granularity"] != "hand":
      raise RuntimeError("expert requires 16-frame, per-hand F6 encoding")
    if map_blob["config"]["token_dim"] != 256 or map_blob["config"]["hand_mode"] != "right":
      raise RuntimeError("expert requires the right-hand 256-D map encoder")
    if not np.allclose(self.f6_stats.q01, np.asarray(f6_blob["stats"]["q01"])) or not np.allclose(
      self.f6_stats.q99, np.asarray(f6_blob["stats"]["q99"])
    ):
      raise RuntimeError("online F6 statistics differ from training")
    if not np.allclose(self.map_stats.q01, np.asarray(map_blob["stats"]["q01"])) or not np.allclose(
      self.map_stats.q99, np.asarray(map_blob["stats"]["q99"])
    ):
      raise RuntimeError("online map statistics differ from training")
    del map_blob, f6_blob

    with (base_path / "config.json").open() as f:
      converted_config = json.load(f)
    if converted_config["action_dim"] != 32 or converted_config["action_horizon"] != 30:
      raise RuntimeError("converted pi0.5 must predict [30,32] actions")
    config = Pi0Config(action_dim=32, action_horizon=30, max_token_len=200,
                       pi05=True, dtype="bfloat16", pytorch_compile_mode=None)
    base = PI0Pytorch(config)
    safetensors.torch.load_model(base, base_path / "model.safetensors", strict=True)
    base = base.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tactile = TactileVelocityExpert(
      f6_config, f6_config["codebook_size"], width=1024, layers=2, action_dim=32
    ).to(self.device)
    tactile.load_state_dict(expert_blob["tactile_state"], strict=True)
    self.expert_step = int(expert_blob["step"])
    del expert_blob
    self.policy = TactilePI05(base, tactile).eval()
    self.flow_cache = None
    self.chunk_start_state = None
    self.tokenizer = PaligemmaTokenizer(200)
    self.generator = torch.Generator(device=self.device)
    self.flow_seed = args.seed if args.flow_seed is None else args.flow_seed
    self.generator.manual_seed(self.flow_seed)

  def _image(self, image: np.ndarray) -> object:
    if image.shape != (240, 320, 3) or image.dtype != np.uint8:
      raise RuntimeError(f"expected 240x320 uint8 RGB, got {image.shape}/{image.dtype}")
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return self.torch.from_numpy(chw).to(self.device).float().div_(127.5).sub_(1.0).unsqueeze(0)

  @property
  def metadata(self) -> dict:
    return {"model_family": "pi0.5+trex", "expert_step": self.expert_step,
            "flow_steps": 10, "tactile_steps": [7, 8, 9, 10],
            "flow_noise_seed": self.flow_seed}

  def infer(self, images: dict[str, np.ndarray], state_raw: np.ndarray, prompt: str,
            history_raw: np.ndarray, current_raw: np.ndarray,
            maps_raw: np.ndarray) -> dict:
    torch = self.torch
    if state_raw.shape != (27,) or not np.isfinite(state_raw).all():
      raise RuntimeError("expected finite 27-D state")
    state_norm = 2.0 * (state_raw - self.state_q01) / (self.state_q99 - self.state_q01 + 1e-6) - 1.0
    prompt_ids, prompt_mask = self.tokenizer.tokenize(prompt, state_norm)
    head, wrist = self._image(images["head"]), self._image(images["right_wrist"])
    yes = torch.ones(1, dtype=torch.bool, device=self.device)
    no = torch.zeros(1, dtype=torch.bool, device=self.device)
    observation = self.Observation(
      images={"base_0_rgb": head, "left_wrist_0_rgb": torch.zeros_like(head),
              "right_wrist_0_rgb": wrist},
      image_masks={"base_0_rgb": yes, "left_wrist_0_rgb": no,
                   "right_wrist_0_rgb": yes},
      state=torch.from_numpy(np.pad(state_norm, (0, 5)).astype(np.float32)).to(self.device).unsqueeze(0),
      tokenized_prompt=torch.from_numpy(prompt_ids.astype(np.int64)).to(self.device).unsqueeze(0),
      tokenized_prompt_mask=torch.from_numpy(prompt_mask.astype(np.bool_)).to(self.device).unsqueeze(0),
    )
    noise = torch.randn((1, 30, 32), generator=self.generator, device=self.device)
    with torch.inference_mode():
      self.flow_cache = self.policy.prepare_late_flow(observation, noise=noise)
    self.chunk_start_state = state_raw.copy()
    return self.refine(history_raw, current_raw, maps_raw)

  def refine(self, history_raw: np.ndarray, current_raw: np.ndarray,
             maps_raw: np.ndarray) -> dict:
    """Refine the current 30-frame chunk without refreshing its visual/state context."""
    if self.flow_cache is None or self.chunk_start_state is None:
      raise RuntimeError("tactile refinement requires a preceding slow inference")
    torch = self.torch
    history = torch.from_numpy(self.f6_stats.normalize_hand(history_raw, 1)).to(self.device).unsqueeze(0)
    current = torch.from_numpy(self.f6_stats.normalize_hand(current_raw, 1)).to(self.device).unsqueeze(0)
    maps = torch.from_numpy(self.map_stats.normalize(maps_raw)).to(self.device).unsqueeze(0)
    with torch.inference_mode():
      normalized = self.policy.continue_late_flow(
        self.flow_cache, history=history, current=current, maps=maps,
      )[0, :, :27].float().cpu().numpy()
    actions = (normalized + 1.0) / 2.0 * (
      self.action_q99 - self.action_q01 + 1e-6
    ) + self.action_q01
    # Delta arm actions remain anchored to the beginning of this chunk.
    actions[:, :7] += self.chunk_start_state[None, :7]
    if actions.shape != (30, 27) or not np.isfinite(actions).all():
      raise RuntimeError("tactile policy produced invalid physical actions")
    return {"actions": actions}


def run(args) -> dict:
  output = args.output_dir.expanduser().resolve()
  deployment_path, deployment = load_deployment_manifest(args.deployment_manifest)
  horizon = validate_deployment(deployment, args.execute_steps)
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "task": "poker-draw",
    "controller": CONTROLLER,
    "inference_backend": "local_pytorch",
    "seed": args.seed,
    "instruction": deployment["instruction"],
    "base_deployment_manifest": str(deployment_path),
    "base_deployment_id": deployment["deployment_id"],
    "base_checkpoint_path": deployment["checkpoint_path"],
    "base_checkpoint_sha256": deployment["checkpoint_sha256"],
    "base_checkpoint_step": deployment["checkpoint_step"],
    "base_pytorch_path": str(args.base_pytorch.expanduser().resolve()),
    "expert_checkpoint_path": str(args.expert_checkpoint.expanduser().resolve()),
    "tactile_root": str(args.tactile_root.expanduser().resolve()),
    "model_family": "pi0.5+trex",
    "model_action_dim": deployment["model_action_dim"],
    "action_dim": deployment["action_dim"],
    "prediction_horizon": horizon,
    "observation_contract": deployment["observation_contract"],
    "tactile_observation_contract": {
      "f6_history": [16, 5, 6], "f6_current": [5, 6],
      "map": [5, 3, 7, 5], "history_rate_hz": CONTROL_HZ,
      "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
      "map_channels": ["normal", "tangent0", "tangent1"],
    },
    "action_representation": deployment["action_representation"],
    "execute_steps": args.execute_steps,
    "tactile_refine_every": args.tactile_refine_every,
    "max_sim_seconds": args.max_sim_seconds,
    "success_hold_seconds": args.success_hold_seconds,
    "full_duration_evaluation": args.full_duration_evaluation,
    "control_hz": CONTROL_HZ,
    "replan_period_s": args.execute_steps / CONTROL_HZ,
    "diagnostic_only": bool(args.disable_penetration_guard),
    "formal_metrics_valid": not args.disable_penetration_guard,
    "penetration_guard_enabled": not args.disable_penetration_guard,
    "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
    "expert_used": True,
    "low_level_tactile_feedback": False,
    "tactile_sent_to_model": True,
    "render_shadows": False,
  }
  started = time.monotonic()
  simulation = recorder = observer = monitor = review_metrics = raw_capture = None
  requests = tactile_refinements = action_steps = clipped_values = control_tick = 0
  maximum_supported_table_penetration_m = 0.0
  first_penetration_limit_exceeded_s = None
  error_text = None
  try:
    policy = LocalTactilePolicy(args, deployment)
    report["policy_metadata"] = policy.metadata
    report["expert_checkpoint_step"] = policy.expert_step
    print(
      f"[policy] loaded base step={deployment['checkpoint_step']} "
      f"tactile step={policy.expert_step} H={horizon} "
      f"execute_steps={args.execute_steps} "
      f"diagnostic_only={args.disable_penetration_guard}",
      flush=True,
    )

    with middle_force_simulation(default_model_path("poker-draw")) as (
      simulation,
      contact,
    ):
      report["contact_model"] = contact
      report["initial_card_randomization"] = reset_randomized_card(
        simulation,
        seed=args.seed,
        xy_jitter_m=args.xy_jitter_mm / 1000.0,
        yaw_jitter_rad=float(np.deg2rad(args.yaw_jitter_deg)),
      )
      report["initial_card_pose"] = simulation.object_pose("card").tolist()
      tactile = OnlineTactile(simulation)
      monitor = PokerOutcomeMonitor(
        required_hold_seconds=args.success_hold_seconds,
        require_clearance_during_hold=args.success_hold_seconds > 0.10,
      )
      # This object is used only for read-only contact/geometry extraction. Its
      # contact-conditioned command hooks are intentionally never called.
      observer = PokerPolicyController(
        simulation,
        penetration_guard_enabled=not args.disable_penetration_guard,
      )
      review_metrics = PokerReviewMetrics(
        report["initial_card_pose"][0], outcome_monitor=monitor
      )
      review_metrics.update(simulation)
      joint_names = list(deployment["joint_names"])
      lower, upper = joint_limits(simulation, joint_names)
      observation = deployment["observation_contract"]
      camera_names = policy_camera_names(observation)
      image_height, image_width, _ = image_shape_hwc(observation)
      cameras = {
        name: CameraConfig(
          name,
          width=image_width,
          height=image_height,
          rgb=True,
          depth=False,
          segmentation=False,
        )
        for name in camera_names
      }
      if args.save_raw:
        from poker_pi05_rollout_artifacts import PokerPi05RawCapture

        raw_capture = PokerPi05RawCapture(
          simulation, output, cameras,
          metadata={
            "model_family": "pi0.5+trex",
            "base_deployment_id": deployment["deployment_id"],
            "base_checkpoint_path": deployment["checkpoint_path"],
            "base_checkpoint_sha256": deployment["checkpoint_sha256"],
            "expert_checkpoint_path": str(args.expert_checkpoint.expanduser().resolve()),
            "seed": args.seed,
            "execute_steps": args.execute_steps,
            "tactile_refine_every": args.tactile_refine_every,
            "policy_control_hz": CONTROL_HZ,
            "initial_card_randomization": report["initial_card_randomization"],
          },
        )
      with WorkcellRenderer(
        simulation.model, tuple(cameras.values()), shadows=False
      ) as renderer, (output / "requests.jsonl").open("x", encoding="utf-8") as log:
        report["render_backend"] = renderer.backend_info
        if args.record:
          from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo

          recorder = PokerPolicyVideo(
            simulation,
            output / "review",
            fps=args.record_fps,
            width=args.review_width,
            height=args.review_height,
            render_width=args.review_render_width,
            render_height=args.review_render_height,
            second_camera=args.review_second_camera,
            include_model_wrist=(
              args.show_model_wrist_in_review and "right_wrist" in camera_names
            ),
            include_review_wrist=(
              "right_wrist" not in camera_names
              and args.review_second_camera != "right_wrist"
            ),
            comparison=True,
            reference_dataset=args.reference_dataset,
            reference_episode_index=args.reference_episode_index,
            metrics=review_metrics,
            metadata={
              "inference_backend": "local_pytorch",
              "seed": args.seed,
              "model_family": "pi0.5+trex",
              "base_checkpoint_path": deployment["checkpoint_path"],
              "base_checkpoint_sha256": deployment["checkpoint_sha256"],
              "base_pytorch_path": str(args.base_pytorch.expanduser().resolve()),
              "expert_checkpoint_path": str(args.expert_checkpoint.expanduser().resolve()),
              "expert_checkpoint_step": policy.expert_step,
              "prediction_horizon": horizon,
              "model_action_dim": deployment["model_action_dim"],
              "action_dim": deployment["action_dim"],
              "execute_steps": args.execute_steps,
              "tactile_refine_every": args.tactile_refine_every,
              "control_hz": CONTROL_HZ,
              "replan_period_s": args.execute_steps / CONTROL_HZ,
              "max_sim_seconds": args.max_sim_seconds,
              "diagnostic_only": bool(args.disable_penetration_guard),
              "formal_metrics_valid": not args.disable_penetration_guard,
              "penetration_guard_enabled": not args.disable_penetration_guard,
              "penetration_guard_threshold_m": PENETRATION_GUARD_THRESHOLD_M,
              "observation_contract": deployment["observation_contract"],
              "action_representation": deployment["action_representation"],
              "initial_randomization": report["initial_card_randomization"],
              "contact_model": contact,
            },
          )
          recorder.capture(0, outcome_stage(monitor))

        while simulation.data.time < args.max_sim_seconds and not monitor.success:
          if args.max_requests and requests >= args.max_requests:
            break
          images = {
            name: renderer.capture(simulation.data, camera)["rgb"]
            for name, camera in cameras.items()
          }
          state = right_joint_state(simulation, joint_names)
          history, current, maps = tactile.observation()
          request_started = time.monotonic()
          response = policy.infer(
            images, state, deployment["instruction"], history, current, maps
          )
          requests += 1
          actions = validated_actions(
            response, horizon=horizon, action_dim=deployment["action_dim"]
          )
          row = {
            "request": requests,
            "sim_time": float(simulation.data.time),
            "stage": outcome_stage(monitor),
            "round_trip_s": time.monotonic() - request_started,
            "inference_backend": "local_pytorch",
            "state": state.tolist(),
            "right_f6": current.tolist(),
            "right_f6_history": history.tolist() if args.save_raw else None,
            "right_map_abs_max_n": float(np.max(np.abs(maps))),
            "predicted_action_min": np.min(actions, axis=0).tolist(),
            "predicted_action_max": np.max(actions, axis=0).tolist(),
            "predicted_actions": actions.tolist() if args.save_raw else None,
            "right_tactile_map": maps.tolist() if args.save_raw else None,
          }
          log.write(json.dumps(row, ensure_ascii=False) + "\n")
          log.flush()
          if requests == 1 and args.save_first_request:
            np.savez_compressed(
              output / "first_request.npz",
              **{f"{name}_image": image for name, image in images.items()},
              state=state,
              right_f6_history=history,
              right_f6_current=current,
              right_tactile_map=maps,
              predicted_actions=actions,
            )
          print(
            f"[policy] request={requests} sim_t={simulation.data.time:.3f}s "
            f"round_trip={row['round_trip_s']:.2f}s stage={outcome_stage(monitor)}",
            flush=True,
          )

          for index in range(args.execute_steps):
            if simulation.data.time >= args.max_sim_seconds or monitor.success:
              break
            if index and index % args.tactile_refine_every == 0:
              history, current, maps = tactile.observation()
              refine_started = time.monotonic()
              refined = policy.refine(history, current, maps)
              actions = validated_actions(
                refined, horizon=horizon, action_dim=deployment["action_dim"]
              )
              tactile_refinements += 1
              log.write(json.dumps({
                "event": "tactile_refinement", "request": requests,
                "offset": index, "sim_time": float(simulation.data.time),
                "round_trip_s": time.monotonic() - refine_started,
                "right_f6": current.tolist(),
                "right_f6_history": history.tolist() if args.save_raw else None,
                "right_map_abs_max_n": float(np.max(np.abs(maps))),
                "predicted_actions": actions.tolist() if args.save_raw else None,
                "right_tactile_map": maps.tolist() if args.save_raw else None,
              }, ensure_ascii=False) + "\n")
              log.flush()
            clipped_values += apply_action(
              simulation, joint_names, actions[index], lower, upper
            )
            control_tick += 1
            target_time = control_tick / CONTROL_HZ
            while (
              simulation.data.time < target_time - 1.0e-12
              and simulation.data.time < args.max_sim_seconds - 1.0e-12
            ):
              simulation.step()
              review_metrics.update(simulation)
              observe_simulation(
                monitor, observer, report["initial_card_pose"][0]
              )
              if raw_capture is not None:
                raw_capture.observe(simulation, outcome_stage(monitor))
              supported_penetration_m = guard_poker_state(
                observer,
                penetration_guard_enabled=not args.disable_penetration_guard,
                fallen_card_guard_enabled=not args.full_duration_evaluation,
              )
              maximum_supported_table_penetration_m = max(
                maximum_supported_table_penetration_m, supported_penetration_m
              )
              if (
                supported_penetration_m > PENETRATION_GUARD_THRESHOLD_M
                and first_penetration_limit_exceeded_s is None
              ):
                first_penetration_limit_exceeded_s = float(simulation.data.time)
              if monitor.success:
                break
            tactile.capture()  # Every 30 Hz control tick, not every replan.
            action_steps += 1
            if recorder is not None:
              recorder.capture(control_tick, outcome_stage(monitor))

      report["status"] = "success" if monitor.success else "task_not_completed"
  except Exception as error:
    error_text = f"{type(error).__name__}: {error}"
    report.update(status="error", error=error_text)
    raise
  finally:
    if simulation is not None:
      report["sim_seconds"] = float(simulation.data.time)
      report["final_card_pose"] = simulation.object_pose("card").tolist()
      report["last_drive_state"] = simulation.drive_state()
    if observer is not None:
      report["terminal_metrics"] = observer.terminal_metrics()
    if monitor is not None:
      report["evaluation"] = monitor.report()
    report["stats"] = {
      "requests": requests,
      "tactile_refinements": tactile_refinements,
      "tactile_updates": requests + tactile_refinements,
      "action_steps": action_steps,
      "clipped_action_values": clipped_values,
    }
    report["penetration_diagnostics"] = {
      "guard_enabled": not args.disable_penetration_guard,
      "threshold_m": PENETRATION_GUARD_THRESHOLD_M,
      "maximum_supported_table_penetration_m": (
        maximum_supported_table_penetration_m
      ),
      "first_limit_exceeded_sim_time_s": first_penetration_limit_exceeded_s,
    }
    report["wall_seconds"] = time.monotonic() - started
    if raw_capture is not None:
      try:
        report["raw"] = raw_capture.finish(report, outcome_stage(monitor))
      except Exception as raw_error:
        report["raw_record_error"] = f"{type(raw_error).__name__}: {raw_error}"
      finally:
        raw_capture.close_incomplete()
    if recorder is not None:
      try:
        recorder.capture(control_tick, outcome_stage(monitor), force=True)
        video = recorder.finish(
          status=report.get("status", "error"),
          evaluation=report.get("evaluation"),
          error=error_text,
        )
        video.update(
          seed=args.seed,
          success=bool(report.get("evaluation", {}).get("success", False)),
          inference_requests=requests,
          tactile_refinements=tactile_refinements,
          simulation_time_s=report.get("sim_seconds"),
          wall_time_s=report["wall_seconds"],
          model_family="pi0.5+trex",
          base_checkpoint_path=deployment["checkpoint_path"],
          base_checkpoint_sha256=deployment["checkpoint_sha256"],
          base_pytorch_path=str(args.base_pytorch.expanduser().resolve()),
          expert_checkpoint_path=str(args.expert_checkpoint.expanduser().resolve()),
          expert_checkpoint_step=report.get("expert_checkpoint_step"),
          prediction_horizon=report.get("prediction_horizon"),
          model_action_dim=deployment["model_action_dim"],
          action_dim=deployment["action_dim"],
          execute_steps=args.execute_steps,
          control_hz=CONTROL_HZ,
          replan_period_s=args.execute_steps / CONTROL_HZ,
          diagnostic_only=bool(args.disable_penetration_guard),
          formal_metrics_valid=not args.disable_penetration_guard,
          penetration_guard_enabled=not args.disable_penetration_guard,
          penetration_guard_threshold_m=PENETRATION_GUARD_THRESHOLD_M,
          maximum_supported_table_penetration_m=(
            maximum_supported_table_penetration_m
          ),
          first_penetration_limit_exceeded_sim_time_s=(
            first_penetration_limit_exceeded_s
          ),
        )
        (output / "review" / "review.json").write_text(
          json.dumps(video, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["video"] = video
      except Exception as record_error:  # noqa: BLE001 - preserve rollout report on recorder failures
        report["record_error"] = f"{type(record_error).__name__}: {record_error}"
    (output / "summary.json").write_text(
      json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
      f"[report] {output / 'summary.json'} status={report.get('status')} "
      f"requests={requests} action_steps={action_steps}",
      flush=True,
    )
  return report


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
