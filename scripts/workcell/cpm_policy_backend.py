"""Strict poker TouchGuide/CPM backend for an isolated model worker.

This module deliberately does not import PyTorch or OpenPI at module import time.
``CPMPolicyBackend`` first validates the deployment manifest, prepends the
declared OpenPI checkout to ``sys.path``, and only then imports the frozen
TouchGuide loader and policy.  A simulator-side worker can therefore import
this adapter in its lightweight environment while model code remains owned by
the explicitly selected runtime.

Canonical manifest shape::

  {
    "schema": "poker_tactile_deployment_v1",
    "task": "poker-draw",
    "model_family": "pi05+cpm",
    "prediction_horizon": 30,
    "execute_steps": 5,
    "source": {
      "base_checkpoint": "/.../checkpoint",
      "cpm_checkpoint": "/.../best.pt",
      "data_root": "/.../training_data",
      "tokenizer_model": "/.../tokenizer.model"
    },
    "guidance": {
      "num_steps": 10,
      "guidance_steps": 3,
      "scale": 1.0,
      "variant": "pi05+touchguidance",
      "compare_base": true,
      "return_trace": false
    },
    "runtime": {
      "project_root": "/.../openpi-main",
      "device": "cuda",
      "precision": "bfloat16"
    }
  }

For the paired unguided control use ``variant="pi05"`` and ``scale=0``.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "poker_tactile_deployment_v1"
TASK = "poker-draw"
MODEL_FAMILY = "pi05+cpm"
PREDICTION_HORIZON = 30
MODEL_ACTION_DIM = 32
ACTION_DIM = 27
CONTROL_HZ = 30
IMAGE_SHAPE = (240, 320, 3)
TACTILE_SHAPE = (5, 3, 7, 5)
GUIDANCE_VARIANTS = ("pi05", "pi05+touchguidance")


def _mapping(value: Any, name: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{name} must be a mapping")
  return dict(value)


def _integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(f"{name} must be an integer")
  if not minimum <= value <= maximum:
    raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
  return value


def _path_text(value: Any, name: str) -> str:
  if not isinstance(value, (str, os.PathLike)):
    raise ValueError(f"{name} must be a path")
  text = os.fspath(value)
  if not text.strip():
    raise ValueError(f"{name} must be a nonempty path")
  return text


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number")
  result = float(value)
  if not math.isfinite(result) or result < minimum:
    raise ValueError(f"{name} must be finite and >= {minimum}")
  return result


def _numeric_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
  raw = np.asarray(value)
  if raw.shape != shape:
    raise ValueError(f"{name} must have shape {list(shape)}")
  if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
    raise ValueError(f"{name} must contain numeric values")
  try:
    array = np.array(raw, dtype=np.float32, copy=True)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(f"{name} must contain finite numeric values") from error
  if not np.isfinite(array).all():
    raise ValueError(f"{name} must contain only finite values")
  return array


def _rgb_array(value: Any, name: str) -> np.ndarray:
  raw = np.asarray(value)
  if raw.shape != IMAGE_SHAPE:
    raise ValueError(f"{name} must have shape {list(IMAGE_SHAPE)}")
  if raw.dtype == np.uint8:
    return np.array(raw, copy=True)
  # JSON lists do not carry a dtype.  Accept only values that can be converted
  # to uint8 without rounding or modular wraparound.
  if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
    raise ValueError(f"{name} must be uint8 RGB")
  try:
    numeric = np.asarray(raw, dtype=np.float64)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(f"{name} must be uint8 RGB") from error
  if (
    not np.isfinite(numeric).all()
    or np.any(numeric < 0)
    or np.any(numeric > 255)
    or np.any(numeric != np.floor(numeric))
  ):
    raise ValueError(f"{name} must contain integer RGB values in [0, 255]")
  return numeric.astype(np.uint8)


def _json_value(value: Any, name: str) -> Any:
  """Convert NumPy output to JSON-safe builtins while rejecting NaN/Inf."""
  if isinstance(value, np.ndarray):
    return _json_value(value.tolist(), name)
  if isinstance(value, np.generic):
    return _json_value(value.item(), name)
  if isinstance(value, Mapping):
    return {str(key): _json_value(item, f"{name}.{key}") for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item, f"{name}[]") for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    raise ValueError(f"{name} contains a nonfinite value")
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  raise TypeError(f"{name} contains a non-JSON value of type {type(value).__name__}")


def _prepend_openpi_paths(project_root: str) -> Path:
  root = Path(project_root).expanduser().resolve(strict=True)
  if not root.is_dir():
    raise ValueError(f"runtime.project_root is not a directory: {root}")
  # The checkout uses a src layout and keeps openpi-client as a sibling
  # package.  Include the root itself as requested for checkouts with a flat
  # layout, but keep src first in import precedence.
  candidates = (root / "src", root / "packages/openpi-client/src", root)
  for candidate in reversed(candidates):
    text = str(candidate)
    if text not in sys.path:
      sys.path.insert(0, text)
  return root


class CPMPolicyBackend:
  """Manifest-bound adapter around OpenPI's frozen poker TouchGuide policy."""

  def __init__(self, manifest: dict):
    self._manifest = copy.deepcopy(_mapping(manifest, "manifest"))
    self._source, self._guidance, self._runtime = self._validate_manifest(self._manifest)
    self._project_root = _prepend_openpi_paths(self._runtime["project_root"])

    # Import only after the selected checkout has been placed on sys.path.
    inference = importlib.import_module("openpi.touchguide.inference")
    policy_module = importlib.import_module("openpi.touchguide.policy")
    sampler_module = importlib.import_module("openpi.touchguide.sampler")
    guidance = sampler_module.GuidanceConfig(
      num_steps=self._guidance["num_steps"],
      guidance_steps=self._guidance["guidance_steps"],
      scale=self._guidance["scale"],
    )
    loaded = inference.load_touchguide(
      base_checkpoint=self._source["base_checkpoint"],
      cpm_checkpoint=self._source["cpm_checkpoint"],
      data_root=self._source["data_root"],
      guidance=guidance,
      device=self._runtime["device"],
      precision=self._runtime["precision"],
    )
    self._policy = policy_module.PokerTouchGuidePolicy(
      loaded,
      tokenizer_model=self._source["tokenizer_model"],
      execute_steps=self._manifest["execute_steps"],
    )
    self._lock = threading.Lock()
    self._metadata = self._build_metadata(self._policy.metadata)

  @staticmethod
  def _validate_manifest(
    manifest: dict[str, Any],
  ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
      "schema": SCHEMA,
      "task": TASK,
      "model_family": MODEL_FAMILY,
      "prediction_horizon": PREDICTION_HORIZON,
    }
    mismatches = {
      key: {"expected": value, "actual": manifest.get(key)}
      for key, value in expected.items()
      if manifest.get(key) != value
    }
    if mismatches:
      raise ValueError(f"CPM deployment identity mismatch: {mismatches}")
    _integer(
      manifest.get("prediction_horizon"),
      "prediction_horizon",
      minimum=PREDICTION_HORIZON,
      maximum=PREDICTION_HORIZON,
    )
    manifest["execute_steps"] = _integer(
      manifest.get("execute_steps"), "execute_steps", minimum=1, maximum=PREDICTION_HORIZON
    )

    source = _mapping(manifest.get("source"), "source")
    for key in ("base_checkpoint", "cpm_checkpoint", "data_root", "tokenizer_model"):
      source[key] = _path_text(source.get(key), f"source.{key}")

    guidance = _mapping(manifest.get("guidance"), "guidance")
    guidance["num_steps"] = _integer(
      guidance.get("num_steps", 10), "guidance.num_steps", minimum=10, maximum=10
    )
    guidance["guidance_steps"] = _integer(
      guidance.get("guidance_steps", 3),
      "guidance.guidance_steps",
      minimum=0,
      maximum=guidance["num_steps"] - 1,
    )
    guidance["scale"] = _finite_number(guidance.get("scale", 1.0), "guidance.scale")
    guidance["variant"] = guidance.get(
      "variant", "pi05" if guidance["scale"] == 0 else "pi05+touchguidance"
    )
    if guidance["variant"] not in GUIDANCE_VARIANTS:
      raise ValueError(f"guidance.variant must be one of {GUIDANCE_VARIANTS}")
    if guidance["variant"] == "pi05" and guidance["scale"] != 0:
      raise ValueError("guidance.variant='pi05' requires guidance.scale=0")
    if guidance["variant"] == "pi05+touchguidance" and (
      guidance["scale"] <= 0 or guidance["guidance_steps"] <= 0
    ):
      raise ValueError("guided variant requires positive scale and guidance_steps")
    for key, default in (("compare_base", True), ("return_trace", False)):
      value = guidance.get(key, default)
      if not isinstance(value, bool):
        raise ValueError(f"guidance.{key} must be boolean")
      guidance[key] = value

    runtime = _mapping(manifest.get("runtime"), "runtime")
    runtime["project_root"] = _path_text(runtime.get("project_root"), "runtime.project_root")
    runtime["device"] = runtime.get("device", "cuda")
    if not isinstance(runtime["device"], str) or not runtime["device"].strip():
      raise ValueError("runtime.device must be a nonempty string")
    runtime["precision"] = runtime.get("precision", "bfloat16")
    if runtime["precision"] not in ("bfloat16", "float32"):
      raise ValueError("runtime.precision must be bfloat16 or float32")
    return source, guidance, runtime

  def _build_metadata(self, policy_metadata: Any) -> dict[str, Any]:
    policy = _mapping(policy_metadata, "OpenPI policy metadata")
    expected = {
      "task": TASK,
      "prediction_horizon": PREDICTION_HORIZON,
      "execute_steps": self._manifest["execute_steps"],
      "control_hz": CONTROL_HZ,
      "model_action_dim": MODEL_ACTION_DIM,
      "action_dim": ACTION_DIM,
      "image_shape_hwc": list(IMAGE_SHAPE),
      "tactile_shape": list(TACTILE_SHAPE),
    }
    mismatches = {
      key: {"expected": value, "actual": policy.get(key)}
      for key, value in expected.items()
      if policy.get(key) != value
    }
    if mismatches:
      raise ValueError(f"OpenPI TouchGuide policy contract mismatch: {mismatches}")
    policy_schema = policy.get("schema")
    metadata = {
      **copy.deepcopy(policy),
      "schema": SCHEMA,
      "policy_schema": policy_schema,
      "task": TASK,
      "evaluation_task": TASK,
      "model_family": MODEL_FAMILY,
      "prediction_horizon": PREDICTION_HORIZON,
      "action_horizon": PREDICTION_HORIZON,
      "action_start_index": 0,
      "execute_steps": self._manifest["execute_steps"],
      "control_hz": CONTROL_HZ,
      "model_action_dim": MODEL_ACTION_DIM,
      "action_dim": ACTION_DIM,
      "tactile_sent_to_model": True,
      "source": copy.deepcopy(self._source),
      "guidance": copy.deepcopy(self._guidance),
      "runtime": {
        "project_root": str(self._project_root),
        "device": self._runtime["device"],
        "precision": self._runtime["precision"],
      },
      "observation_contract": {
        "cameras": ["head", "right_wrist"],
        "image_shape_hwc": list(IMAGE_SHAPE),
        "state_dim": ACTION_DIM,
        "tactile_sent_to_model": True,
        "tactile_shape": list(TACTILE_SHAPE),
        "tactile_unit": "N",
        "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
        "tactile_channels": ["normal", "tangent0", "tangent1"],
      },
      "action_representation": {
        "policy_output_is_absolute": True,
        "action_start_index": 0,
        "description": "absolute right-arm 7 plus right-hand 20 joint targets in radians",
      },
    }
    for key in ("deployment_id", "base_checkpoint_sha256", "cpm_checkpoint_sha256"):
      if key in self._manifest:
        metadata[key] = copy.deepcopy(self._manifest[key])
      elif key in self._source:
        metadata[key] = copy.deepcopy(self._source[key])
    return _json_value(metadata, "metadata")

  def metadata(self) -> dict:
    """Return a defensive, JSON-compatible copy of the frozen contract."""
    return copy.deepcopy(self._metadata)

  def infer(self, request: dict) -> dict:
    """Validate one synchronized observation and return absolute H30 targets."""
    request = _mapping(request, "request")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
      raise ValueError("prompt must be a nonempty string")
    episode_seed = _integer(
      request.get("episode_seed"), "episode_seed", minimum=0, maximum=(1 << 63) - 1
    )
    request_index = _integer(
      request.get("request_index"), "request_index", minimum=0, maximum=(1 << 63) - 1
    )
    execute_steps = _integer(
      request.get("execute_steps"), "execute_steps", minimum=1, maximum=PREDICTION_HORIZON
    )
    if execute_steps != self._manifest["execute_steps"]:
      raise ValueError(
        "request execute_steps differs from the manifest-bound execution contract"
      )
    if "variant" in request and request["variant"] != self._guidance["variant"]:
      raise ValueError("request variant differs from the manifest-bound guidance variant")

    policy_request = {
      "state": _numeric_array(request.get("state"), "state", (ACTION_DIM,)),
      "head_image": _rgb_array(request.get("head_image"), "head_image"),
      "right_wrist_image": _rgb_array(
        request.get("right_wrist_image"), "right_wrist_image"
      ),
      "tactile_map_raw": _numeric_array(
        request.get("tactile_map_raw"), "tactile_map_raw", TACTILE_SHAPE
      ),
      "prompt": prompt,
      "episode_seed": episode_seed,
      "request_index": request_index,
      "execute_steps": execute_steps,
      "variant": self._guidance["variant"],
      "compare_base": self._guidance["compare_base"],
      "return_trace": self._guidance["return_trace"],
    }
    # Do not wrap this call in torch.inference_mode(): TouchGuide's late flow
    # steps deliberately re-enable autograd for the candidate action tensor.
    with self._lock:
      response = self._policy.infer(policy_request)
    response = _mapping(response, "OpenPI policy response")

    actions = _numeric_array(response.get("actions"), "actions", (PREDICTION_HORIZON, ACTION_DIM))
    diagnostics = _mapping(response.get("diagnostics"), "diagnostics")
    if diagnostics.get("variant") != self._guidance["variant"]:
      raise ValueError("OpenPI diagnostics variant differs from the manifest")
    result = {
      "actions": _json_value(actions, "actions"),
      "diagnostics": _json_value(diagnostics, "diagnostics"),
    }
    if "base_actions" in response:
      base_actions = _numeric_array(
        response["base_actions"], "base_actions", (PREDICTION_HORIZON, ACTION_DIM)
      )
      result["base_actions"] = _json_value(base_actions, "base_actions")
    if "audit" in response:
      if not self._guidance["return_trace"]:
        raise ValueError("OpenPI returned an audit trace when return_trace is disabled")
      result["audit"] = _json_value(response["audit"], "audit")
    return result


__all__ = ["CPMPolicyBackend"]
