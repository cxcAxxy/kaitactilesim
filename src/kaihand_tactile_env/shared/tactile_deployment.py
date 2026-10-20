"""Frozen, task-scoped native-horizon tactile deployment identities."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .tactile_task_contracts import ftp1_training_binding, task_contract

SCHEMA = "poker_tactile_deployment_v1"
TASK_SCHEMA = "tactile_deployment_v2"
FAMILIES = {"pi05+cpm": (30, 32, 0), "ftp1": (32, 120, 1)}
ROOT = Path(__file__).resolve().parents[3]
INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the "
  "fingers and thumb, lift it, and turn its face toward the robot to look at it."
)
JOINT_NAMES = [*(f"right_arm_joint{i}" for i in range(1, 8)),
  "hand_r_thumb_joint1", "hand_r_thumb_joint2", "hand_r_thumb_joint3", "hand_r_thumb_joint5",
  *(f"hand_r_{finger}_joint{i}" for finger in ("index", "middle", "ring", "pinky") for i in range(1, 5))]


def sha256_file(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def identity(payload):
  return hashlib.sha256(json.dumps(
    {key: value for key, value in payload.items() if key != "deployment_id"},
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
  ).encode()).hexdigest()


def validate_manifest(payload, *, verify_files=False):
  schema, task = payload.get("schema"), payload.get("task")
  if schema not in {SCHEMA, TASK_SCHEMA} or (schema == SCHEMA and task != "poker-draw"):
    raise ValueError("expected poker tactile deployment v1 or task-scoped deployment v2")
  family = payload.get("model_family")
  if family not in FAMILIES:
    raise ValueError("unsupported tactile model family")
  if schema == TASK_SCHEMA:
    contract = task_contract(task)
    if family != "ftp1":
      raise ValueError("non-poker CPM requires a task-native base policy and CPM; not implemented")
    source = payload.get("source", {})
    binding = source.get("training_binding", {})
    if (source.get("domain_name") != contract["domain_name"]
        or source.get("action_joint_rep") not in {"mix", "arm_relative_hand_absolute"}
        or binding.get("task") != task
        or binding.get("domain_name") != contract["domain_name"]
        or binding.get("instruction") != contract["instruction"]
        or binding.get("action_joint_rep") != source.get("action_joint_rep")
        or payload.get("instruction") != contract["instruction"]):
      raise ValueError("FTP-1 deployment task/domain/training binding mismatch")
    supervision = binding.get("action_supervision")
    if (not isinstance(supervision, dict)
        or supervision.get("contract") != "ftp1-native-future-measured-state-trajectory-v1"
        or supervision.get("controller_targets_used") is not False
        or supervision.get("pi05_recorded_controller_target_equivalent") is not False):
      raise ValueError("task training binding requires native future-measured-state action supervision")
    observation = payload.get("observation_contract", {})
    if (observation.get("tactile_target_body") != contract["tactile_target_body"]
        or observation.get("tactile_provider") != contract["tactile_provider"]
        or observation.get("tactile_target_geometries") != contract["tactile_target_geometries"]):
      raise ValueError("task tactile target/provider contract mismatch")
    if payload.get("diagnostic_only") is not True or payload.get("formal_metrics_valid") is not False:
      raise ValueError("new task tactile deployments are diagnostic until acceptance")
    if (observation.get("tactile_shape") != [5, 3, 7, 5]
        or observation.get("tactile_unit") != "N"
        or payload.get("controller_version") != "shared-tactile-direct-joints-v1"
        or payload.get("codec_version") != "ftp1-kaihand-task-v2"
        or payload.get("action_representation") != "absolute-right-arm7-hand20-joints"):
      raise ValueError("invalid task tactile sensor/controller/codec contract")
    for key in ("native_evaluation_protocol", "training_protocol_transfer"):
      if not isinstance(payload.get(key), str) or not payload[key].strip():
        raise ValueError(f"task tactile deployment requires {key}")
  horizon, model_dim, offset = FAMILIES[family]
  expected = dict(prediction_horizon=horizon, model_action_dim=model_dim,
                  action_start_index=offset, action_dim=27, control_hz=30,
                  joint_names=JOINT_NAMES, required_action_interpolation="zoh")
  for key, value in expected.items():
    if payload.get(key) != value:
      raise ValueError(f"invalid tactile deployment {key}")
  steps = payload.get("execute_steps")
  if type(steps) is not int or not 1 <= steps <= horizon - offset:
    raise ValueError("execute_steps exceeds native future horizon")
  observation = payload.get("observation_contract", {})
  if observation.get("cameras") != ["head", "right_wrist"] or observation.get("image_shape_hwc") != [240, 320, 3] or observation.get("tactile_sent_to_model") is not True:
    raise ValueError("invalid synchronized image/tactile contract")
  if payload.get("deployment_id") != identity(payload):
    raise ValueError("tactile deployment identity mismatch")
  inventory = payload.get("files")
  if not isinstance(inventory, dict) or not inventory:
    raise ValueError("deployment requires an artifact/source file inventory")
  if schema == TASK_SCHEMA:
    for key in ("dataset_config_path", "conversion_manifest_path"):
      if binding.get(key) not in inventory:
        raise ValueError(f"task training binding must freeze {key}")
  if verify_files:
    for name, expected_hash in inventory.items():
      if sha256_file(name) != expected_hash:
        raise ValueError(f"frozen tactile artifact/source changed: {name}")
  return payload


def load_manifest(path, *, verify_files=False):
  resolved = Path(path).expanduser().resolve(strict=True)
  return resolved, validate_manifest(json.loads(resolved.read_text()), verify_files=verify_files)


def _files_under(path, *, suffixes=None):
  path = Path(path).expanduser().resolve(strict=True)
  if path.is_file():
    return [path]
  return sorted(p for p in path.rglob("*") if p.is_file()
                and (suffixes is None or p.suffix in suffixes)
                and p.name not in {"optimizer.pt", "metadata.pt"}
                and "__pycache__" not in p.parts)


def prepare_manifest(*, family, checkpoint, project, python, execute_steps=5,
                     base_checkpoint=None, data_root=None, tokenizer_model=None,
                     guidance_scale=1.0, guidance_steps=3, domain_name=None, task="poker-draw"):
  contract = task_contract(task)
  if task != "poker-draw" and family != "ftp1":
    raise ValueError("non-poker CPM requires a task-native base policy and CPM; not implemented")
  if domain_name is not None and domain_name != contract["domain_name"]:
    raise ValueError("domain_name must match the task's frozen training domain")
  domain_name = contract["domain_name"]
  horizon, model_dim, offset = FAMILIES[family]
  checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
  project = Path(project).expanduser().resolve(strict=True)
  source = {}
  instruction = contract["instruction"]
  files = _files_under(checkpoint)
  if family == "pi05+cpm":
    source = {"cpm_checkpoint": str(checkpoint)}
    for key, value in (("base_checkpoint", base_checkpoint), ("data_root", data_root), ("tokenizer_model", tokenizer_model)):
      if value is None:
        raise ValueError(f"CPM requires {key}")
      source[key] = str(Path(value).expanduser().resolve(strict=True))
    files += _files_under(source["base_checkpoint"])
    files += _files_under(source["tokenizer_model"])
    for subdir in ("config", "meta"):
      files += _files_under(Path(source["data_root"]) / subdir, suffixes={".json", ".jsonl"})
    task_path = Path(source["data_root"]) / "meta/task_instruction.json"
    if task_path.exists():
      instruction = json.loads(task_path.read_text())["task"]
    else:
      data_manifest = json.loads((Path(source["data_root"]) / "meta/manifest.json").read_text())
      task_path = Path(data_manifest["sources"]["pi05_root"]) / "meta/tasks.jsonl"
      tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
      if len(tasks) != 1:
        raise ValueError("CPM requires one frozen instruction")
      instruction = tasks[0]["task"]
    files.append(task_path)
    settings = json.loads((Path(source["data_root"]) / "config/cpm.json").read_text())
    files += _files_under(settings["vision"]["directory"])
  else:
    source = {"checkpoint": str(checkpoint), "domain_name": domain_name}
    tokenizer = Path(tokenizer_model or "/root/.cache/openpi/big_vision/paligemma_tokenizer.model").expanduser().resolve(strict=True)
    source["tokenizer_model"] = str(tokenizer)
    files.append(tokenizer)
    for name in ("model_config.json", "train_config.json", "tactile_input_config_file.json", "model.safetensors"):
      if not (checkpoint / name).is_file():
        raise ValueError(f"FTP-1 checkpoint missing {name}")
    for name in ("normalization", "hpt_tokenizer"):
      if not (checkpoint / name).is_dir():
        raise ValueError(f"FTP-1 checkpoint missing {name}")
    if task != "poker-draw":
      binding = ftp1_training_binding(checkpoint, task)
      source.update(training_binding=binding, action_joint_rep=binding["action_joint_rep"])
      files += [Path(binding[key]) for key in ("dataset_config_path", "conversion_manifest_path")]
      model = json.loads((checkpoint / "model_config.json").read_text())
      if model.get("action_dim") != 120 or model.get("action_horizon") != 32 or model.get("use_tactile_input") is not True:
        raise ValueError("FTP-1 task checkpoint requires native H32/D120 with tactile input")
      norm = checkpoint / "normalization" / domain_name / "independent_norm_stats_all_t0_zscore.json"
      tactile = json.loads((checkpoint / "tactile_input_config_file.json").read_text())
      if not norm.is_file() or domain_name not in tactile:
        raise ValueError("FTP-1 checkpoint has no matching task normalization/tactile domain")
  # Model code and local physical/transport code are part of deployment identity.
  files += _files_under(project / "src/openpi", suffixes={".py"})
  if (project / "packages/openpi-client/src").exists():
    files += _files_under(project / "packages/openpi-client/src", suffixes={".py"})
  files += _files_under(ROOT / "src/kaihand_tactile_env", suffixes={".py", ".xml"})
  files += _files_under(ROOT / "src/kaihand_tactile_env/assets")
  for name in ("run_poker_pi05_policy.py", "run_poker_tactile_policy.py", "tactile_policy_worker.py",
               "tactile_policy_binding.py", "cpm_policy_backend.py" if family == "pi05+cpm" else "ftp1_policy_backend.py",
               "run_usb_pi05_policy.py", "poker_pi05_rollout_artifacts.py",
               "evaluate_poker_tactile_policy_batch.py", "convert_kaihand_raw_to_ftp1_zarr.py"):
    files.append(ROOT / "scripts/workcell" / name)
  if task != "poker-draw":
    files += [ROOT / "scripts/workcell" / name for name in (
      "run_shared_task_pi05_policy.py", "run_shared_task_tactile_policy.py",
      "evaluate_task_tactile_policy_batch.py")]
  files.append(ROOT / "scripts/evaluate/evaluate.py")
  inventory = {str(p): sha256_file(p) for p in sorted(set(files))}
  checkpoint_file = checkpoint if checkpoint.is_file() else checkpoint / "model.safetensors"
  payload = dict(schema=SCHEMA if task == "poker-draw" else TASK_SCHEMA, task=task, model_family=family,
    checkpoint_path=str(checkpoint), checkpoint_sha256=inventory[str(checkpoint_file)],
    checkpoint_step=checkpoint.name if checkpoint.is_dir() else "best",
    source=source, runtime=dict(project_root=str(project), python=str(Path(python).expanduser().absolute()), device="cuda", precision="bfloat16"),
    prediction_horizon=horizon, model_action_dim=model_dim, action_dim=27,
    action_start_index=offset, execute_steps=execute_steps, control_hz=30,
    joint_names=JOINT_NAMES, instruction=instruction,
    observation_contract=dict(cameras=["head", "right_wrist"], image_shape_hwc=[240, 320, 3], tactile_sent_to_model=True,
      tactile_shape=[5, 3, 7, 5], tactile_unit="N", tactile_target_geometry="card_core_geom"),
    action_representation="absolute-right-arm7-hand20-joints",
    required_action_interpolation="zoh", controller_version="poker-tactile-direct-joints-v1",
    codec_version="ftp1-kaihand-poker-v1" if family == "ftp1" else "touchguide-native-v1",
    diagnostic_only=True, formal_metrics_valid=False, files=inventory)
  if task != "poker-draw":
    payload["observation_contract"].pop("tactile_target_geometry")
    payload["observation_contract"].update({key: contract[key] for key in (
      "tactile_target_body", "tactile_target_geometries", "tactile_provider")})
    payload.update(controller_version="shared-tactile-direct-joints-v1",
                   codec_version="ftp1-kaihand-task-v2",
                   native_evaluation_protocol="TaskPolicyAdapter (current task completion semantics)",
                   training_protocol_transfer="recorded training instruction is preserved; native simulator completion remains authoritative")
  payload["runtime"]["env"] = {key: os.environ[key] for key in (
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "HF_HOME", "XDG_CACHE_HOME", "TOKENIZERS_PARALLELISM",
  ) if key in os.environ}
  version_script = """import importlib.metadata as m, json
p = {}
for name in ["torch", "transformers", "numpy", "safetensors", "sentencepiece"]:
    try:
        p[name] = m.version(name)
    except m.PackageNotFoundError:
        p[name] = None
print(json.dumps(p))
"""
  versions = subprocess.run([payload["runtime"]["python"], "-c", version_script],
    check=True, capture_output=True, text=True, timeout=30)
  payload["runtime"]["package_versions"] = json.loads(versions.stdout)
  payload["runtime"]["python_binary_sha256"] = sha256_file(Path(python).expanduser().resolve(strict=True))
  if family == "pi05+cpm":
    payload["guidance"] = dict(num_steps=10, guidance_steps=guidance_steps, scale=guidance_scale,
      variant="pi05" if guidance_scale == 0 else "pi05+touchguidance")
  payload["deployment_id"] = identity(payload)
  return validate_manifest(payload)
