#!/usr/bin/env python3
"""Load the real poker CPM backend and compare one paired guided/base request."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from cpm_policy_backend import CPMPolicyBackend

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(
  "/nas/chenxianchi/datasets/touchguidence/"
  "recovery_state_cpm_ratio_1_1_b1024_g8_20261013_165142/best.pt"
)
DEFAULT_DATA_ROOT = Path(
  "/nas/chenxianchi/datasets/touchguidence/augmentation/"
  "poker_fingertip_recovery_rules_v1_random10/ratio_1_1"
)
DEFAULT_BASE = Path(
  "/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k"
)
DEFAULT_TOKENIZER = Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model")
DEFAULT_OPENPI = Path("/cpfs_infra/user/chenxianchi/yidu/openpi-main")
DEFAULT_OBSERVATION = Path(
  "/nas/chenxianchi/datasets/touchguidence/eval/"
  "oa_1to1_late_steps_20261019_112405/oa_last3/seed_000/requests/"
  "request_000044.npz"
)
DEFAULT_OUTPUT = ROOT / "reports/tactile_integration_20261019/cpm_backend_smoke"
INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the "
  "fingers and thumb, lift it, and turn its face toward the robot to look at it."
)
JOINT_NAMES = [
  *(f"right_arm_joint{index}" for index in range(1, 8)),
  "hand_r_thumb_joint1",
  "hand_r_thumb_joint2",
  "hand_r_thumb_joint3",
  "hand_r_thumb_joint5",
  *(
    f"hand_r_{finger}_joint{index}"
    for finger in ("index", "middle", "ring", "pinky")
    for index in range(1, 5)
  ),
]


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
  parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
  parser.add_argument("--tokenizer-model", type=Path, default=DEFAULT_TOKENIZER)
  parser.add_argument("--openpi-project", type=Path, default=DEFAULT_OPENPI)
  parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--device", default="cuda")
  return parser.parse_args(argv)


def manifest(args, *, scale, variant):
  return {
    "schema": "poker_tactile_deployment_v1",
    "task": "poker-draw",
    "model_family": "pi05+cpm",
    "prediction_horizon": 30,
    "model_action_dim": 32,
    "action_dim": 27,
    "action_start_index": 0,
    "execute_steps": 5,
    "control_hz": 30,
    "joint_names": JOINT_NAMES,
    "instruction": INSTRUCTION,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "image_shape_hwc": [240, 320, 3],
      "tactile_sent_to_model": True,
      "tactile_shape": [5, 3, 7, 5],
      "tactile_unit": "N",
      "tactile_target_geometry": "card_core_geom",
    },
    "action_representation": "absolute-right-arm7-hand20-joints",
    "required_action_interpolation": "zoh",
    "source": {
      "base_checkpoint": str(args.base_checkpoint.resolve(strict=True)),
      "cpm_checkpoint": str(args.checkpoint.resolve(strict=True)),
      "data_root": str(args.data_root.resolve(strict=True)),
      "tokenizer_model": str(args.tokenizer_model.resolve(strict=True)),
    },
    "guidance": {
      "num_steps": 10,
      "guidance_steps": 3,
      "scale": scale,
      "variant": variant,
      "compare_base": variant != "pi05",
      "return_trace": False,
    },
    "runtime": {
      "project_root": str(args.openpi_project.resolve(strict=True)),
      "device": args.device,
      "precision": "bfloat16",
    },
  }


def load_request(path):
  source = path.resolve(strict=True)
  with np.load(source, allow_pickle=False) as payload:
    request = {
      "state": np.array(payload["state_raw"], copy=True),
      "head_image": np.array(payload["head_image"], copy=True),
      "right_wrist_image": np.array(payload["right_wrist_image"], copy=True),
      "tactile_map_raw": np.array(payload["tactile_map_raw"], copy=True),
      "prompt": INSTRUCTION,
      "episode_seed": 0,
      "request_index": 44,
      "execute_steps": 5,
    }
    source_time = float(payload["sim_time_s"]) if "sim_time_s" in payload else None
  return source, request, source_time


def finite_tree(value):
  if isinstance(value, dict):
    return all(finite_tree(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return all(finite_tree(item) for item in value)
  if isinstance(value, (float, np.floating)):
    return math.isfinite(float(value))
  return True


def summarize_output(output):
  actions = np.asarray(output["actions"], dtype=np.float32)
  steps = output["diagnostics"].get("steps", [])
  guided = [row for row in steps if row.get("guided")]
  return {
    "actions_shape": list(actions.shape),
    "actions_finite": bool(np.isfinite(actions).all()),
    "actions_min_rad": float(actions.min()),
    "actions_max_rad": float(actions.max()),
    "diagnostics_finite": finite_tree(output["diagnostics"]),
    "noise_seed": int(output["diagnostics"]["noise_seed"]),
    "denoising_steps": len(steps),
    "guided_steps": len(guided),
    "gradient_rms": [float(row["gradient_rms"]) for row in guided],
    "score_before": [float(row["score_before"][0]) for row in guided],
    "score_after_guidance_only": [
      float(row["score_after_guidance_only"][0]) for row in guided
    ],
    "latent_guidance_shift_rms": [
      float(row["latent_guidance_shift_rms"]) for row in guided
    ],
    "padded_guidance_max_abs": [
      float(row["padded_guidance_max_abs"]) for row in guided
    ],
  }


def load_and_infer(args, request, *, scale, variant):
  started = time.monotonic()
  backend = CPMPolicyBackend(manifest(args, scale=scale, variant=variant))
  loaded = time.monotonic()
  metadata = backend.metadata()
  output = backend.infer(request)
  finished = time.monotonic()
  return backend, metadata, output, {
    "load_seconds": loaded - started,
    "infer_seconds": finished - loaded,
  }


def main(argv=None):
  args = parse_args(argv)
  output_dir = args.output.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(output_dir / "smoke.log"), logging.StreamHandler()],
  )
  result_path = output_dir / "result.json"
  try:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
      raise RuntimeError("This acceptance smoke requires an explicitly selected CUDA device")
    torch.set_num_threads(8)
    source, request, source_time = load_request(args.observation)
    tactile = request["tactile_map_raw"]
    result = {
      "schema": "cpm_backend_smoke_v1",
      "status": "running",
      "source_observation": str(source),
      "source_sim_time_s": source_time,
      "input": {
        "state_shape": list(request["state"].shape),
        "head_image_shape": list(request["head_image"].shape),
        "right_wrist_image_shape": list(request["right_wrist_image"].shape),
        "tactile_shape": list(tactile.shape),
        "tactile_abs_max_n": float(np.max(np.abs(tactile))),
        "tactile_nonzero_count": int(np.count_nonzero(tactile)),
        "episode_seed": request["episode_seed"],
        "request_index": request["request_index"],
      },
      "cuda": {
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_count": torch.cuda.device_count(),
      },
      "artifacts": {
        "base_checkpoint": str(args.base_checkpoint.resolve(strict=True)),
        "cpm_checkpoint": str(args.checkpoint.resolve(strict=True)),
        "data_root": str(args.data_root.resolve(strict=True)),
        "tokenizer_model": str(args.tokenizer_model.resolve(strict=True)),
        "openpi_project": str(args.openpi_project.resolve(strict=True)),
      },
    }
    logging.info("loading paired scale=0 backend")
    baseline, baseline_metadata, baseline_output, baseline_timing = load_and_infer(
      args, request, scale=0.0, variant="pi05"
    )
    result["baseline"] = {
      "metadata": {
        key: baseline_metadata[key]
        for key in (
          "schema",
          "model_family",
          "task",
          "prediction_horizon",
          "action_start_index",
          "action_dim",
          "control_hz",
          "tactile_sent_to_model",
          "guidance",
        )
      },
      "timing": baseline_timing,
      "output": summarize_output(baseline_output),
    }
    baseline_actions = np.asarray(baseline_output["actions"], dtype=np.float32)
    del baseline, baseline_output
    gc.collect()
    torch.cuda.empty_cache()

    logging.info("loading scale=1, last-3 guided backend")
    guided, guided_metadata, guided_output, guided_timing = load_and_infer(
      args, request, scale=1.0, variant="pi05+touchguidance"
    )
    result["guided"] = {
      "metadata": {
        key: guided_metadata[key]
        for key in (
          "schema",
          "model_family",
          "task",
          "prediction_horizon",
          "action_start_index",
          "action_dim",
          "control_hz",
          "tactile_sent_to_model",
          "guidance",
        )
      },
      "timing": guided_timing,
      "output": summarize_output(guided_output),
    }
    guided_actions = np.asarray(guided_output["actions"], dtype=np.float32)
    guided_base = np.asarray(guided_output["base_actions"], dtype=np.float32)
    delta = guided_actions - baseline_actions
    base_replay_delta = guided_base - baseline_actions
    guided_summary = result["guided"]["output"]
    result["comparison"] = {
      "same_noise_seed": (
        result["baseline"]["output"]["noise_seed"]
        == guided_summary["noise_seed"]
      ),
      "guided_vs_baseline_rms_rad": float(np.sqrt(np.mean(np.square(delta)))),
      "guided_vs_baseline_max_abs_rad": float(np.max(np.abs(delta))),
      "guided_base_replay_max_abs_rad": float(np.max(np.abs(base_replay_delta))),
      "guided_base_replay_matches_scale0_atol_1e-6": bool(
        np.allclose(guided_base, baseline_actions, rtol=0, atol=1e-6)
      ),
      "every_guidance_score_non_decreasing": all(
        after >= before - 1e-6
        for before, after in zip(
          guided_summary["score_before"],
          guided_summary["score_after_guidance_only"],
          strict=True,
        )
      ),
      "all_padded_guidance_exact_zero": all(
        value == 0.0 for value in guided_summary["padded_guidance_max_abs"]
      ),
    }
    checks = {
      "baseline_h30x27_finite": (
        result["baseline"]["output"]["actions_shape"] == [30, 27]
        and result["baseline"]["output"]["actions_finite"]
      ),
      "guided_h30x27_finite": (
        guided_summary["actions_shape"] == [30, 27]
        and guided_summary["actions_finite"]
      ),
      "guided_diagnostics_finite": guided_summary["diagnostics_finite"],
      "three_guided_steps": guided_summary["guided_steps"] == 3,
      "positive_finite_gradients": (
        len(guided_summary["gradient_rms"]) == 3
        and all(value > 0 and math.isfinite(value) for value in guided_summary["gradient_rms"])
      ),
      "guidance_changes_actions": (
        result["comparison"]["guided_vs_baseline_max_abs_rad"] > 0
      ),
      **result["comparison"],
    }
    result["checks"] = checks
    result["status"] = "passed" if all(
      value if isinstance(value, bool) else math.isfinite(value) for value in checks.values()
    ) else "failed"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    logging.info("smoke status=%s result=%s", result["status"], result_path)
    del guided, guided_output
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "passed":
      raise RuntimeError("CPM backend smoke checks failed")
    return 0
  except BaseException as error:
    if not result_path.exists():
      result_path.write_text(json.dumps({
        "schema": "cpm_backend_smoke_v1",
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
      }, indent=2, allow_nan=False) + "\n")
    logging.exception("CPM backend smoke failed")
    raise


if __name__ == "__main__":
  raise SystemExit(main())
