"""Checkpoint-bound FTP-1 worker backend for registered tactile tasks.

The module intentionally imports the external ``openpi`` project only when a
backend instance is constructed.  A worker process can therefore validate and
test this transport/codec without importing torch or loading the 7.6 GB model.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from kaihand_tactile_env.shared.ftp1_policy_contract import (
  FTP1_ACTION_HORIZON,
  FTP1_ACTION_JOINT_REPS,
  FTP1_ACTION_START_INDEX,
  FTP1_CONTROL_HZ,
  FTP1_IMAGE_SHAPE_HWC,
  FTP1_MODEL_ACTION_DIM,
  FTP1_PHYSICAL_ACTION_DIM,
  FTP1_TACTILE_FUNCTION_AREAS,
  FTP1_TACTILE_KEY,
  FTP1_TACTILE_SENSOR,
  RIGHT_JOINT_NAMES,
  decode_ftp1_actions,
  encode_ftp1_state,
  ftp1_action_mask,
  pack_ftp1_tactile,
  validate_world_transform,
)
from kaihand_tactile_env.shared.tactile_task_contracts import (
  ftp1_training_binding,
  task_contract,
)

DEPLOYMENT_SCHEMA = "poker_tactile_deployment_v1"
TASK_DEPLOYMENT_SCHEMA = "tactile_deployment_v2"
MODEL_FAMILY = "ftp1"
TASK = "poker-draw"
DOMAIN_NAME = "KaiHandSimPokerDraw"
PALIGEMMA_TOKENIZER_URL = "gs://big_vision/paligemma_tokenizer.model"
_HPT_TOKENIZER_FILENAME = "KaiHandSimTaxelV1_matrix_7_5_3.safetensors"
_REQUIRED_REQUEST_KEYS = frozenset(
  {
    "state",
    "head_image",
    "right_wrist_image",
    "tactile_map_raw",
    "world_T_head",
    "world_T_wrist",
    "prompt",
    "execute_steps",
  }
)
_OPTIONAL_REQUEST_KEYS = frozenset(
  {"seed", "index", "episode_seed", "request_index", "images", "sim_time"}
)
_REQUEST_KEYS = _REQUIRED_REQUEST_KEYS | _OPTIONAL_REQUEST_KEYS


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
  value = parent.get(key)
  if not isinstance(value, Mapping):
    raise ValueError(f"manifest.{key} must be an object")
  return value


def _require_text(parent: Mapping[str, Any], key: str, *, prefix: str) -> str:
  value = parent.get(key)
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{prefix}.{key} must be a nonempty string")
  return value.strip()


def _require_int(
  parent: Mapping[str, Any],
  key: str,
  expected: int | None = None,
  *,
  prefix: str = "manifest",
) -> int:
  value = parent.get(key)
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise ValueError(f"{prefix}.{key} must be an integer")
  result = int(value)
  if expected is not None and result != expected:
    raise ValueError(f"{prefix}.{key} must be {expected}, got {result}")
  return result


def _resolve_directory(value: str, *, field: str) -> Path:
  path = Path(value).expanduser().resolve()
  if not path.is_dir():
    raise ValueError(f"{field} is not a directory: {path}")
  return path


def _rotation_distance_rad(left: np.ndarray, right: np.ndarray) -> float:
  relative = left.T @ right
  cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
  return float(math.acos(cosine))


@contextlib.contextmanager
def _seeded_torch_rng(seed: int, device: str):
  """Seed FTP-1 diffusion locally and restore the worker's torch RNG state."""

  torch = importlib.import_module("torch")
  cuda_devices: list[int] = []
  if str(device).startswith("cuda"):
    if ":" in str(device):
      cuda_devices = [int(str(device).split(":", 1)[1])]
    else:
      cuda_devices = [int(torch.cuda.current_device())]
  with torch.random.fork_rng(devices=cuda_devices):
    torch.manual_seed(seed)
    if cuda_devices:
      torch.cuda.manual_seed_all(seed)
    yield


class FTP1PolicyBackend:
  """Load one task-bound FTP-1 checkpoint and serve synchronized observations."""

  def __init__(self, manifest: dict):
    if not isinstance(manifest, dict):
      raise TypeError("manifest must be a dict")
    self._manifest = copy.deepcopy(manifest)
    self._validate_manifest()
    self._lock = threading.Lock()

    wrapper_class = self._import_wrapper_class()
    with self._frozen_tokenizer_download():
      self._wrapper = wrapper_class(
        checkpoint_dir=str(self._checkpoint),
        domain_name=self._domain_name,
        device=self._device,
        num_inference_steps=self._num_inference_steps,
        skip_normalization=False,
      )
    if int(self._wrapper.get_action_horizon()) != FTP1_ACTION_HORIZON:
      raise RuntimeError("loaded FTP-1 wrapper has an incompatible action horizon")
    if int(self._wrapper.get_action_dim()) != FTP1_MODEL_ACTION_DIM:
      raise RuntimeError("loaded FTP-1 wrapper has an incompatible action dimension")
    if int(self._wrapper.get_state_dim()) != FTP1_MODEL_ACTION_DIM:
      raise RuntimeError("loaded FTP-1 wrapper has an incompatible state dimension")
    self._validate_loaded_wrapper()

  def _validate_manifest(self) -> None:
    self._schema = self._manifest.get("schema")
    if self._schema not in {DEPLOYMENT_SCHEMA, TASK_DEPLOYMENT_SCHEMA}:
      raise ValueError(
        "manifest.schema must be a supported FTP-1 deployment schema"
      )
    self._task = self._manifest.get("task")
    if self._schema == DEPLOYMENT_SCHEMA and self._task != TASK:
      raise ValueError(
        f"manifest.task must be {TASK!r} for {DEPLOYMENT_SCHEMA!r}"
      )
    if not isinstance(self._task, str):
      raise ValueError("manifest.task must be a registered task name")
    self._task_contract = copy.deepcopy(task_contract(self._task))
    recorded_instruction = self._task_contract["instruction"]
    if self._schema == TASK_DEPLOYMENT_SCHEMA:
      if self._manifest.get("instruction") != recorded_instruction:
        raise ValueError(
          "manifest.instruction must match the recorded FTP-1 training instruction"
        )
      if self._manifest.get("diagnostic_only") is not True:
        raise ValueError("task-scoped FTP-1 deployments must be diagnostic_only")
      if self._manifest.get("formal_metrics_valid") is not False:
        raise ValueError(
          "task-scoped FTP-1 deployments cannot claim formal metrics"
        )
      self._native_evaluation_protocol = _require_text(
        self._manifest,
        "native_evaluation_protocol",
        prefix="manifest",
      )
      self._training_protocol_transfer = _require_text(
        self._manifest,
        "training_protocol_transfer",
        prefix="manifest",
      )
      self._instruction = recorded_instruction
    else:
      declared_instruction = self._manifest.get("instruction")
      self._instruction = (
        declared_instruction.strip()
        if isinstance(declared_instruction, str) and declared_instruction.strip()
        else recorded_instruction
      )
      self._native_evaluation_protocol = None
      self._training_protocol_transfer = None
    self._diagnostic_only = self._manifest.get("diagnostic_only", True)
    self._formal_metrics_valid = self._manifest.get("formal_metrics_valid", False)
    if self._manifest.get("model_family") != MODEL_FAMILY:
      raise ValueError(f"manifest.model_family must be {MODEL_FAMILY!r}")
    _require_int(self._manifest, "prediction_horizon", FTP1_ACTION_HORIZON)
    _require_int(self._manifest, "action_start_index", FTP1_ACTION_START_INDEX)
    _require_int(self._manifest, "model_action_dim", FTP1_MODEL_ACTION_DIM)
    _require_int(self._manifest, "action_dim", FTP1_PHYSICAL_ACTION_DIM)
    _require_int(self._manifest, "control_hz", FTP1_CONTROL_HZ)
    self._execute_steps = _require_int(self._manifest, "execute_steps")
    if not 1 <= self._execute_steps <= FTP1_ACTION_HORIZON - FTP1_ACTION_START_INDEX:
      raise ValueError("manifest.execute_steps must be in [1,31]")

    source = _require_mapping(self._manifest, "source")
    checkpoint = _require_text(source, "checkpoint", prefix="manifest.source")
    self._checkpoint = _resolve_directory(
      checkpoint, field="manifest.source.checkpoint"
    )
    self._domain_name = _require_text(
      source, "domain_name", prefix="manifest.source"
    )
    expected_domain = self._task_contract["domain_name"]
    if self._domain_name != expected_domain:
      raise ValueError(
        f"manifest.source.domain_name must be {expected_domain!r} for {self._task!r}"
      )
    if self._schema == DEPLOYMENT_SCHEMA:
      self._action_joint_rep = source.get("action_joint_rep", "mix")
      if self._action_joint_rep != "mix":
        raise ValueError("legacy poker FTP-1 deployments require action_joint_rep='mix'")
      self._training_binding = None
    else:
      self._action_joint_rep = _require_text(
        source, "action_joint_rep", prefix="manifest.source"
      )
      if self._action_joint_rep not in FTP1_ACTION_JOINT_REPS:
        raise ValueError(
          f"unsupported FTP-1 action_joint_rep: {self._action_joint_rep!r}"
        )
      declared_binding = source.get("training_binding")
      if not isinstance(declared_binding, Mapping):
        raise ValueError("manifest.source.training_binding must be an object")
      if (
        declared_binding.get("task") != self._task
        or declared_binding.get("domain_name") != self._domain_name
        or declared_binding.get("instruction") != self._instruction
        or declared_binding.get("action_joint_rep") != self._action_joint_rep
      ):
        raise ValueError(
          "manifest.source task/domain/instruction/action recipe differs from "
          "training_binding"
        )
      try:
        actual_binding = ftp1_training_binding(self._checkpoint, self._task)
      except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
          "checkpoint FTP-1 training binding cannot be verified"
        ) from error
      if dict(declared_binding) != actual_binding:
        raise ValueError(
          "manifest.source.training_binding does not match checkpoint provenance"
        )
      self._training_binding = copy.deepcopy(actual_binding)

      observation = _require_mapping(self._manifest, "observation_contract")
      target_mismatches = {
        key: {
          "expected": self._task_contract[key],
          "actual": observation.get(key),
        }
        for key in (
          "tactile_target_body",
          "tactile_target_geometries",
          "tactile_provider",
        )
        if observation.get(key) != self._task_contract[key]
      }
      if target_mismatches:
        raise ValueError(
          f"manifest task tactile target/provider mismatch: {target_mismatches}"
        )
    tokenizer_model = _require_text(
      source, "tokenizer_model", prefix="manifest.source"
    )
    self._tokenizer_model = Path(tokenizer_model).expanduser().resolve()
    if not self._tokenizer_model.is_file():
      raise ValueError(
        f"manifest.source.tokenizer_model is not a file: {self._tokenizer_model}"
      )
    inventory = self._manifest.get("files")
    required_frozen_paths = {str(self._tokenizer_model)}
    if self._training_binding is not None:
      required_frozen_paths.update(
        self._training_binding[key]
        for key in ("dataset_config_path", "conversion_manifest_path")
      )
    if not isinstance(inventory, Mapping):
      raise ValueError("manifest.files must contain the frozen runtime assets")
    missing_frozen_paths = sorted(required_frozen_paths - inventory.keys())
    if missing_frozen_paths:
      raise ValueError(
        "manifest.files lacks frozen tokenizer/training provenance: "
        f"{missing_frozen_paths}"
      )
    self._validate_checkpoint_contract()

    runtime = _require_mapping(self._manifest, "runtime")
    project_root = _require_text(
      runtime, "project_root", prefix="manifest.runtime"
    )
    self._project_root = _resolve_directory(
      project_root, field="manifest.runtime.project_root"
    )
    python = _require_text(runtime, "python", prefix="manifest.runtime")
    self._python = Path(python).expanduser().resolve()
    if not self._python.is_file():
      raise ValueError(f"manifest.runtime.python is not a file: {self._python}")
    self._device = _require_text(runtime, "device", prefix="manifest.runtime")
    if not (self._device == "cpu" or self._device == "cuda" or self._device.startswith("cuda:")):
      raise ValueError("manifest.runtime.device must be cpu, cuda, or cuda:<index>")
    steps = runtime.get("num_inference_steps", 10)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
      raise ValueError("manifest.runtime.num_inference_steps must be an integer")
    self._num_inference_steps = int(steps)
    if self._num_inference_steps <= 0:
      raise ValueError("manifest.runtime.num_inference_steps must be positive")

    declared_names = self._manifest.get("joint_names")
    if declared_names is not None and tuple(declared_names) != RIGHT_JOINT_NAMES:
      raise ValueError("manifest.joint_names does not match the FTP-1 KaiHand order")

  def _read_checkpoint_json(self, name: str) -> dict[str, Any]:
    path = self._checkpoint / name
    if not path.is_file():
      raise ValueError(f"FTP-1 checkpoint is missing {name}")
    try:
      value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
      raise ValueError(f"FTP-1 checkpoint has invalid {name}") from error
    if not isinstance(value, dict):
      raise ValueError(f"FTP-1 checkpoint {name} must contain an object")
    return value

  @staticmethod
  def _expect_fields(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
  ) -> None:
    mismatches = {
      key: {"expected": expected_value, "actual": value.get(key)}
      for key, expected_value in expected.items()
      if value.get(key) != expected_value
    }
    if mismatches:
      raise ValueError(f"{context} contract mismatch: {mismatches}")

  def _validate_checkpoint_contract(self) -> None:
    """Reject checkpoints whose H32x120 values require a different codec."""

    model = self._read_checkpoint_json("model_config.json")
    self._expect_fields(
      model,
      {
        "action_dim": FTP1_MODEL_ACTION_DIM,
        "action_horizon": FTP1_ACTION_HORIZON,
        "discrete_state_input": False,
        "state_input_mode": "adarms",
        "use_tactile_input": True,
        "disable_history": True,
      },
      context="model_config.json",
    )
    train = self._read_checkpoint_json("train_config.json")
    self._expect_fields(
      train,
      {
        "single_arm_action_rep_dim": 48,
        "reserved_action_dim": 15,
        "disable_history": True,
        "action_down_sample_steps": 1,
        "proprioception_pose_rep": "relative",
        "action_pose_rep": "relative",
        "proprioception_joint_rep": "abs",
        "action_joint_rep": self._action_joint_rep,
        "independent_norm_mode": "all",
        "norm_time_dim": False,
        "norm_type": "zscore",
        "norm_image_tactile_mode": "channel_wise",
      },
      context="train_config.json",
    )
    tactile = self._read_checkpoint_json("tactile_input_config_file.json")
    expected_tactile = {
      "shape": [1, 5, 7, 5, 3],
      "function_areas": list(FTP1_TACTILE_FUNCTION_AREAS),
      "sensor": FTP1_TACTILE_SENSOR,
      "type": "matrix",
    }
    if tactile.get(self._domain_name, {}).get(FTP1_TACTILE_KEY) != expected_tactile:
      raise ValueError("tactile_input_config_file.json contract mismatch")

    for relative in (
      "model.safetensors",
      f"hpt_tokenizer/{_HPT_TOKENIZER_FILENAME}",
      f"normalization/{self._domain_name}/independent_norm_stats_all_t0_zscore.json",
    ):
      if not (self._checkpoint / relative).is_file():
        raise ValueError(f"FTP-1 checkpoint is missing {relative}")
    norm = self._read_checkpoint_json(
      f"normalization/{self._domain_name}/independent_norm_stats_all_t0_zscore.json"
    )
    self._expect_fields(
      norm,
      {
        "norm_time_dim": False,
        "norm_type": "zscore",
        "data_dim": {
          "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": 3},
          "actions": 120,
          "state": 120,
        },
      },
      context="domain normalization",
    )
    parameters = norm.get("params")
    if not isinstance(parameters, Mapping) or not {"state", "actions", "tactile"}.issubset(
      parameters
    ):
      raise ValueError("domain normalization lacks state/actions/tactile parameters")
    tactile_parameters = parameters.get("tactile")
    if not isinstance(tactile_parameters, Mapping) or (
      "KaiHandSimTaxelV1_matrix_7_5_3" not in tactile_parameters
    ):
      raise ValueError("domain normalization lacks KaiHand matrix tactile parameters")

  def _import_wrapper_class(self):
    source_root = (self._project_root / "src").resolve()
    if not source_root.is_dir():
      raise RuntimeError(f"FTP-1 project has no src directory: {source_root}")
    # ``openpi`` imports resolve from project/src, while this FTP-1 wrapper also
    # imports ``src.openpi`` and therefore needs the project root itself.
    for import_root in (self._project_root, source_root):
      import_text = str(import_root)
      if import_text not in sys.path:
        sys.path.insert(0, import_text)
    module = importlib.import_module("openpi.policies.ftp1_inference_wrapper")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(self._project_root):
      raise RuntimeError(
        f"imported FTP-1 wrapper from undeclared project: {module_path}"
      )
    return module.FTP1InferenceWrapper

  @contextlib.contextmanager
  def _frozen_tokenizer_download(self):
    """Bind PaliGemma tokenization to the manifest asset, never the network."""

    download = importlib.import_module("openpi.shared.download")
    original = download.maybe_download

    def offline_only(url: str, *args, **kwargs) -> Path:
      del args, kwargs
      if url != PALIGEMMA_TOKENIZER_URL:
        raise RuntimeError(f"FTP-1 runtime attempted an undeclared download: {url}")
      return self._tokenizer_model

    download.maybe_download = offline_only
    try:
      yield
    finally:
      download.maybe_download = original

  @contextlib.contextmanager
  def _batch_safe_image_resize(self):
    """Preserve an existing batch axis across FTP-1's image resize helper.

    The pinned FTP-1 runtime unconditionally squeezes a size-one batch after
    resizing a channels-last tensor.  Its next preprocessing step requires a
    four-dimensional ``[B,H,W,C]`` tensor, so non-224 input images otherwise
    fail before inference.  Keep the workaround local to the serialized model
    call and restore the external module immediately afterwards.
    """

    image_tools = importlib.import_module("openpi.shared.image_tools")
    original = image_tools.resize_with_pad_torch

    def preserve_batch(images, *args, **kwargs):
      had_batch = getattr(images, "ndim", None) == 4
      resized = original(images, *args, **kwargs)
      if had_batch and getattr(resized, "ndim", None) == 3:
        resized = resized.unsqueeze(0)
      return resized

    image_tools.resize_with_pad_torch = preserve_batch
    try:
      yield
    finally:
      image_tools.resize_with_pad_torch = original

  def _validate_loaded_wrapper(self) -> None:
    model_config = getattr(self._wrapper, "model_config", None)
    if model_config is None or getattr(model_config, "use_tactile_input", None) is not True:
      raise RuntimeError("loaded FTP-1 wrapper does not require tactile input")
    if getattr(self._wrapper, "domain_norm_stats", None) is None:
      raise RuntimeError("loaded FTP-1 wrapper has no domain normalization")
    params = self._wrapper.domain_norm_stats.get("params", {})
    if not {"state", "actions", "tactile"}.issubset(params):
      raise RuntimeError("loaded FTP-1 domain normalization is incomplete")
    tactile_types = getattr(self._wrapper, "tactile_key_type_map", {})
    if tactile_types.get(FTP1_TACTILE_KEY) != "matrix":
      raise RuntimeError("loaded FTP-1 wrapper lacks the KaiHand matrix tactile stream")
    if getattr(self._wrapper, "tokenizer", None) is None:
      raise RuntimeError("loaded FTP-1 wrapper has no language tokenizer")

  def metadata(self) -> dict[str, Any]:
    """Return JSON-compatible worker metadata for the runner handshake."""

    metadata = {
      "schema": self._schema,
      "task": self._task,
      "evaluation_task": self._task,
      "model_family": MODEL_FAMILY,
      "instruction": self._instruction,
      "diagnostic_only": self._diagnostic_only,
      "formal_metrics_valid": self._formal_metrics_valid,
      "deployment_id": self._manifest.get("deployment_id"),
      "checkpoint_path": str(self._checkpoint),
      "checkpoint_sha256": self._manifest.get("checkpoint_sha256"),
      "domain_name": self._domain_name,
      "prediction_horizon": FTP1_ACTION_HORIZON,
      "action_horizon": FTP1_ACTION_HORIZON,
      "action_start_index": FTP1_ACTION_START_INDEX,
      "max_execute_steps": FTP1_ACTION_HORIZON - FTP1_ACTION_START_INDEX,
      "model_action_dim": FTP1_MODEL_ACTION_DIM,
      "action_dim": FTP1_PHYSICAL_ACTION_DIM,
      "control_hz": FTP1_CONTROL_HZ,
      "execute_steps": self._execute_steps,
      "joint_names": list(RIGHT_JOINT_NAMES),
      "tactile_sent_to_model": True,
      "source": copy.deepcopy(self._manifest["source"]),
      "observation_contract": {
        "image_keys": ["head_image", "right_wrist_image"],
        "image_shape_hwc": list(FTP1_IMAGE_SHAPE_HWC),
        "state_dim": FTP1_PHYSICAL_ACTION_DIM,
        "tactile_key": FTP1_TACTILE_KEY,
        "tactile_request_shape": [5, 3, 7, 5],
        "tactile_model_shape": [1, 5, 7, 5, 3],
        "tactile_channels": [
          "normal_n",
          "grid_col_tangent_n",
          "grid_row_tangent_n",
        ],
        "tactile_unit": "N_per_taxel",
        "tactile_sent_to_model": True,
        "synchronization": "same_sim_state_latest_nonfuture_v1",
        "tactile_target_body": self._task_contract["tactile_target_body"],
        "tactile_target_geometries": self._task_contract[
          "tactile_target_geometries"
        ],
        "tactile_provider": self._task_contract["tactile_provider"],
      },
      "action_representation": {
        "model": (
          "ftp1_relative_wrist_mix_joints_120_v1"
          if self._action_joint_rep == "mix"
          else "ftp1_relative_wrist_arm_relative_hand_absolute_120_v1"
        ),
        "action_joint_rep": self._action_joint_rep,
        "physical": "absolute_right_arm7_hand20_joints_v1",
        "first_row": "observation_time_target",
        "normalization": "checkpoint_wrapper_once",
      },
      "runtime": {
        "project_root": str(self._project_root),
        "python": str(self._python),
        "device": self._device,
        "num_inference_steps": self._num_inference_steps,
      },
    }
    if self._schema == TASK_DEPLOYMENT_SCHEMA:
      metadata.update(
        native_evaluation_protocol=self._native_evaluation_protocol,
        training_protocol_transfer=self._training_protocol_transfer,
        protocol_transfer={
          "diagnostic_only": True,
          "formal_metrics_valid": False,
          "recorded_training_instruction": self._instruction,
          "native_evaluation_protocol": self._native_evaluation_protocol,
          "note": self._training_protocol_transfer,
        },
      )
    return metadata

  @staticmethod
  def _image(name: str, value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != FTP1_IMAGE_SHAPE_HWC:
      raise ValueError(
        f"{name} must have shape {FTP1_IMAGE_SHAPE_HWC}, got {image.shape}"
      )
    if image.dtype != np.uint8:
      raise ValueError(f"{name} must have dtype uint8")
    return np.ascontiguousarray(image)

  @staticmethod
  def _nonnegative_int(request: Mapping[str, Any], key: str) -> int:
    value = request.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
      raise ValueError(f"request.{key} must be an integer")
    result = int(value)
    if result < 0:
      raise ValueError(f"request.{key} must be nonnegative")
    return result

  @classmethod
  def _aliased_nonnegative_int(
    cls,
    request: Mapping[str, Any],
    primary: str,
    alias: str,
  ) -> int:
    if primary not in request and alias not in request:
      raise ValueError(f"request requires {primary} or {alias}")
    values = [
      cls._nonnegative_int(request, key)
      for key in (primary, alias)
      if key in request
    ]
    if len(set(values)) != 1:
      raise ValueError(f"request.{primary} and request.{alias} disagree")
    return values[0]

  def infer(self, request: dict) -> dict[str, Any]:
    """Infer H32 absolute joint targets from one synchronized raw observation."""

    if not isinstance(request, dict):
      raise TypeError("request must be a dict")
    missing = sorted(_REQUIRED_REQUEST_KEYS - request.keys())
    unknown = sorted(request.keys() - _REQUEST_KEYS)
    if missing or unknown:
      raise ValueError(f"request keys mismatch: missing={missing}, unknown={unknown}")

    total_started = time.perf_counter()
    seed = self._aliased_nonnegative_int(request, "seed", "episode_seed")
    index = self._aliased_nonnegative_int(request, "index", "request_index")
    execute_steps = self._nonnegative_int(request, "execute_steps")
    if not 1 <= execute_steps <= FTP1_ACTION_HORIZON - FTP1_ACTION_START_INDEX:
      raise ValueError("request.execute_steps must be in [1,31]")
    if execute_steps != self._execute_steps:
      raise ValueError(
        f"request.execute_steps must match frozen manifest value {self._execute_steps}"
      )
    prompt = request["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
      raise ValueError("request.prompt must be a nonempty string")
    normalized_prompt = prompt.strip()
    prompt_matches_recorded_instruction = normalized_prompt == self._instruction
    if (
      self._schema == TASK_DEPLOYMENT_SCHEMA
      and not prompt_matches_recorded_instruction
    ):
      raise ValueError(
        "request.prompt must match the recorded FTP-1 training instruction"
      )

    encode_started = time.perf_counter()
    head_image = self._image("request.head_image", request["head_image"])
    wrist_image = self._image(
      "request.right_wrist_image", request["right_wrist_image"]
    )
    if "images" in request:
      images = request["images"]
      if not isinstance(images, Mapping) or set(images) != {"head", "right_wrist"}:
        raise ValueError("request.images must contain exactly head and right_wrist")
      if not np.array_equal(np.asarray(images["head"]), head_image) or not np.array_equal(
        np.asarray(images["right_wrist"]), wrist_image
      ):
        raise ValueError("request.images disagrees with the flat image fields")
    if "sim_time" in request:
      sim_time = request["sim_time"]
      if isinstance(sim_time, bool) or not isinstance(
        sim_time, (int, float, np.integer, np.floating)
      ) or not np.isfinite(sim_time) or float(sim_time) < 0.0:
        raise ValueError("request.sim_time must be finite and nonnegative")
    # Validate transforms before both state encoding and action decoding so a
    # malformed frame can never reach the model.
    world_T_head = validate_world_transform(
      "request.world_T_head", request["world_T_head"]
    )
    world_T_wrist = validate_world_transform(
      "request.world_T_wrist", request["world_T_wrist"]
    )
    encoded_state = encode_ftp1_state(
      request["state"], world_T_head, world_T_wrist
    )
    tactile = pack_ftp1_tactile(request["tactile_map_raw"])
    action_mask = ftp1_action_mask()
    encode_s = time.perf_counter() - encode_started

    # Make each rollout/request pair deterministic without leaving global torch
    # RNG state changed.  The large odd multiplier avoids repeating adjacent
    # request seeds while remaining stable across Python processes.
    inference_seed = (seed * 1_000_003 + index) % (2**63 - 1)
    inference_started = time.perf_counter()
    with (
      self._lock,
      _seeded_torch_rng(inference_seed, self._device),
      self._batch_safe_image_resize(),
    ):
      model_actions = self._wrapper.infer(
        images={
          "camera_ego_rgb_0": head_image,
          "right_wrist_camera_rgb_0": wrist_image,
        },
        state=encoded_state,
        prompt=normalized_prompt,
        tactiles={FTP1_TACTILE_KEY: tactile},
        tactile_function_areas={
          FTP1_TACTILE_KEY: list(FTP1_TACTILE_FUNCTION_AREAS)
        },
        tactile_sensors={FTP1_TACTILE_KEY: FTP1_TACTILE_SENSOR},
        action_mask=action_mask,
      )
    inference_s = time.perf_counter() - inference_started

    decode_started = time.perf_counter()
    decoded = decode_ftp1_actions(
      model_actions,
      request["state"],
      world_T_head,
      world_T_wrist,
      action_joint_rep=self._action_joint_rep,
    )
    decode_s = time.perf_counter() - decode_started

    state = np.asarray(request["state"], dtype=np.float64)
    first_wrist = decoded.wrist_world_transforms[0]
    t0_wrist_position_error_m = float(
      np.linalg.norm(first_wrist[:3, 3] - world_T_wrist[:3, 3])
    )
    t0_wrist_rotation_error_rad = _rotation_distance_rad(
      world_T_wrist[:3, :3], first_wrist[:3, :3]
    )
    t0_joint_error_max = float(
      np.max(np.abs(decoded.joint_actions[0].astype(np.float64) - state))
    )
    total_s = time.perf_counter() - total_started
    return {
      "actions": decoded.joint_actions,
      "diagnostics": {
        "seed": seed,
        "index": index,
        "inference_seed": inference_seed,
        "execute_steps": execute_steps,
        "action_start_index": FTP1_ACTION_START_INDEX,
        "task": self._task,
        "domain_name": self._domain_name,
        "action_joint_rep": self._action_joint_rep,
        "recorded_training_instruction": self._instruction,
        "prompt_matches_recorded_instruction": prompt_matches_recorded_instruction,
        "diagnostic_only": self._diagnostic_only,
        "formal_metrics_valid": self._formal_metrics_valid,
        "native_evaluation_protocol": self._native_evaluation_protocol,
        "training_protocol_transfer": self._training_protocol_transfer,
        "action_supervision": (
          self._training_binding.get("action_supervision")
          if self._training_binding is not None
          else None
        ),
        "model_action_shape": [FTP1_ACTION_HORIZON, FTP1_MODEL_ACTION_DIM],
        "physical_action_shape": [
          FTP1_ACTION_HORIZON,
          FTP1_PHYSICAL_ACTION_DIM,
        ],
        "normalization_applied_by_wrapper": True,
        "inactive_model_dimensions_executed": False,
        "t0_joint_error_max": t0_joint_error_max,
        "t0_wrist_position_error_m": t0_wrist_position_error_m,
        "t0_wrist_rotation_error_rad": t0_wrist_rotation_error_rad,
        "timing_s": {
          "encode": encode_s,
          "inference": inference_s,
          "decode": decode_s,
          "total": total_s,
        },
      },
    }


__all__ = ["DEPLOYMENT_SCHEMA", "FTP1PolicyBackend", "TASK_DEPLOYMENT_SCHEMA"]
