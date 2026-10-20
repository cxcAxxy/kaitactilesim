#!/usr/bin/env python3
"""Read-only CPM/FTP-1 readiness inventory for the 13 evaluation tasks.

This preflight reads small JSON files and checks file presence only. It never
imports a model runtime, loads weights, starts a simulator, or hashes a weight
file. A matching result is therefore a checkpoint-contract result, not a
closed-loop model acceptance result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))

from kaihand_tactile_env.pipeline.catalog import TASKS  # noqa: E402
from kaihand_tactile_env.shared.tactile_task_contracts import (  # noqa: E402
  FTP1_TASK_CONTRACTS,
  SHARED_TACTILE_TASKS,
  ftp1_training_binding,
)

SCHEMA = "kaihand-tactile-task-readiness-v1"
FTP1_ACTION_HORIZON = 32
FTP1_MODEL_ACTION_DIM = 120
FTP1_TACTILE_KEY = "right_tactile_fingers"
FTP1_TACTILE_SENSOR = "KaiHandSimTaxelV1"
FTP1_HPT_FILENAME = "KaiHandSimTaxelV1_matrix_7_5_3.safetensors"
FTP1_PENDING_ADAPTERS = frozenset({"pick-place"})
MAX_SCAN_DEPTH = 3
MAX_SCAN_DIRECTORIES = 512
MAX_SCAN_CANDIDATES = 64

KNOWN_POKER_FTP1_CHECKPOINT = Path(
  "/nas/chenxianchi/datasets/sim/poker-draw/ftp1_checkpoints/ftp1/"
  "poker_draw_0920_200_h32_bs128_25k_ddp8_20261012_114006/24999"
)
KNOWN_CPM_BUNDLE = {
  "checkpoint": Path(
    "/nas/chenxianchi/datasets/touchguidence/"
    "recovery_state_cpm_ratio_1_1_b1024_g8_20261013_165142/best.pt"
  ),
  "base_checkpoint": Path(
    "/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k"
  ),
  "data_root": Path(
    "/nas/chenxianchi/datasets/touchguidence/augmentation/"
    "poker_fingertip_recovery_rules_v1_random10/ratio_1_1"
  ),
}

_TASK_STORAGE_NAMES = {"usb-insert": "usb_insert"}
DEFAULT_FTP1_SCAN_ROOTS = {
  task: (
    Path("/nas/chenxianchi/datasets/sim")
    / _TASK_STORAGE_NAMES.get(task, task)
    / "ftp1_checkpoints",
  )
  for task in TASKS
}

_MODEL_EXPECTED = {
  "action_dim": FTP1_MODEL_ACTION_DIM,
  "action_horizon": FTP1_ACTION_HORIZON,
  "discrete_state_input": False,
  "state_input_mode": "adarms",
  "use_tactile_input": True,
  "disable_history": True,
}
_TRAIN_EXPECTED = {
  "single_arm_action_rep_dim": 48,
  "reserved_action_dim": 15,
  "disable_history": True,
  "action_down_sample_steps": 1,
  "proprioception_pose_rep": "relative",
  "action_pose_rep": "relative",
  "proprioception_joint_rep": "abs",
  "independent_norm_mode": "all",
  "norm_time_dim": False,
  "norm_type": "zscore",
  "norm_image_tactile_mode": "channel_wise",
}
_SUPPORTED_TRAIN_ACTION_REPRESENTATIONS = frozenset(
  {"mix", "arm_relative_hand_absolute"}
)


def _read_object(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return value


def _matches(value: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
  return all(value.get(key) == expected_value for key, expected_value in expected.items())


def _error(errors: list[str], label: str, error: BaseException) -> None:
  errors.append(f"{label}: {type(error).__name__}: {error}")


def inspect_ftp_checkpoint(checkpoint: str | Path, task: str) -> dict[str, Any]:
  """Validate a task-specific FTP-1 checkpoint without importing FTP-1 or weights."""

  path = Path(checkpoint).expanduser().resolve()
  checks = {
    "checkpoint_directory": path.is_dir(),
    "registered_task_contract": task in FTP1_TASK_CONTRACTS,
    "model_h32_d120": False,
    "training_representation": False,
    "tactile_domain": False,
    "model_weights_present": False,
    "hpt_tokenizer_present": False,
    "domain_normalization": False,
    "task_dataset_binding": False,
  }
  errors: list[str] = []
  binding = None
  contract = FTP1_TASK_CONTRACTS.get(task)
  if not checks["checkpoint_directory"]:
    errors.append(f"checkpoint_directory: not a directory: {path}")
  if contract is None:
    errors.append(f"registered_task_contract: no FTP-1 adapter contract for {task!r}")

  try:
    model = _read_object(path / "model_config.json")
    checks["model_h32_d120"] = _matches(model, _MODEL_EXPECTED)
    if not checks["model_h32_d120"]:
      errors.append("model_h32_d120: model_config.json contract mismatch")
  except (OSError, ValueError, json.JSONDecodeError) as error:
    _error(errors, "model_h32_d120", error)

  try:
    train = _read_object(path / "train_config.json")
    checks["training_representation"] = _matches(train, _TRAIN_EXPECTED) and (
      train.get("action_joint_rep") in _SUPPORTED_TRAIN_ACTION_REPRESENTATIONS
    )
    if not checks["training_representation"]:
      errors.append("training_representation: train_config.json contract mismatch")
  except (OSError, ValueError, json.JSONDecodeError) as error:
    _error(errors, "training_representation", error)

  if contract is not None:
    expected_tactile = {
      "shape": [1, 5, 7, 5, 3],
      "function_areas": [0, 1, 2, 3, 4],
      "sensor": FTP1_TACTILE_SENSOR,
      "type": "matrix",
    }
    try:
      tactile = _read_object(path / "tactile_input_config_file.json")
      checks["tactile_domain"] = (
        tactile.get(contract["domain_name"], {}).get(FTP1_TACTILE_KEY)
        == expected_tactile
      )
      if not checks["tactile_domain"]:
        errors.append("tactile_domain: task domain/tactile contract mismatch")
    except (OSError, ValueError, json.JSONDecodeError) as error:
      _error(errors, "tactile_domain", error)

    norm_path = (
      path
      / "normalization"
      / contract["domain_name"]
      / "independent_norm_stats_all_t0_zscore.json"
    )
    try:
      norm = _read_object(norm_path)
      parameters = norm.get("params")
      tactile_parameters = (
        parameters.get("tactile") if isinstance(parameters, Mapping) else None
      )
      checks["domain_normalization"] = (
        norm.get("norm_time_dim") is False
        and norm.get("norm_type") == "zscore"
        and norm.get("data_dim")
        == {
          "tactile": {FTP1_HPT_FILENAME.removesuffix(".safetensors"): 3},
          "actions": FTP1_MODEL_ACTION_DIM,
          "state": FTP1_MODEL_ACTION_DIM,
        }
        and isinstance(parameters, Mapping)
        and {"state", "actions", "tactile"}.issubset(parameters)
        and isinstance(tactile_parameters, Mapping)
        and FTP1_HPT_FILENAME.removesuffix(".safetensors") in tactile_parameters
      )
      if not checks["domain_normalization"]:
        errors.append("domain_normalization: normalization contract is incomplete")
    except (OSError, ValueError, json.JSONDecodeError) as error:
      _error(errors, "domain_normalization", error)

  weights = path / "model.safetensors"
  checks["model_weights_present"] = weights.is_file() and weights.stat().st_size > 0
  if not checks["model_weights_present"]:
    errors.append("model_weights_present: nonempty model.safetensors is required")
  hpt = path / "hpt_tokenizer" / FTP1_HPT_FILENAME
  checks["hpt_tokenizer_present"] = hpt.is_file() and hpt.stat().st_size > 0
  if not checks["hpt_tokenizer_present"]:
    errors.append(f"hpt_tokenizer_present: nonempty {hpt.relative_to(path)} is required")

  if contract is not None:
    try:
      binding = ftp1_training_binding(path, task)
      checks["task_dataset_binding"] = (
        binding.get("task") == task
        and binding.get("domain_name") == contract["domain_name"]
        and binding.get("instruction") == contract["instruction"]
      )
      if not checks["task_dataset_binding"]:
        errors.append("task_dataset_binding: returned binding does not match task contract")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
      _error(errors, "task_dataset_binding", error)

  matched = all(checks.values())
  return {
    "checkpoint": str(path),
    "task": task,
    "domain_name": contract["domain_name"] if contract else None,
    "matched": matched,
    "status": "checkpoint_contract_matched" if matched else "checkpoint_contract_rejected",
    "checks": checks,
    "training_binding": binding,
    "errors": errors,
    "weights_loaded": False,
    "weights_hashed": False,
    "normalization_is_not_training_evidence": True,
  }


def discover_ftp_candidates(
  task: str,
  *,
  roots: Mapping[str, Sequence[str | Path]] | None = None,
) -> dict[str, Any]:
  """Find checkpoint-like directories with an explicit depth and count bound."""

  selected = DEFAULT_FTP1_SCAN_ROOTS if roots is None else roots
  scan_roots = tuple(Path(path).expanduser() for path in selected.get(task, ()))
  candidates: set[Path] = set()
  errors: list[str] = []
  directories_examined = 0
  truncated = False
  markers = ("model_config.json", "train_config.json", "model.safetensors")

  for root in scan_roots:
    if not root.is_dir():
      continue
    pending = [(root.resolve(), 0)]
    while pending:
      directory, depth = pending.pop()
      directories_examined += 1
      if directories_examined > MAX_SCAN_DIRECTORIES:
        truncated = True
        break
      if any((directory / marker).is_file() for marker in markers):
        candidates.add(directory)
        if len(candidates) >= MAX_SCAN_CANDIDATES:
          truncated = True
          break
      if depth >= MAX_SCAN_DEPTH:
        continue
      try:
        children = sorted(
          (child for child in directory.iterdir() if child.is_dir()), reverse=True
        )
      except OSError as error:
        _error(errors, f"scan {directory}", error)
        continue
      pending.extend((child, depth + 1) for child in children)
    if truncated:
      break

  return {
    "task": task,
    "roots": [str(path.resolve()) for path in scan_roots],
    "max_depth": MAX_SCAN_DEPTH,
    "directories_examined": directories_examined,
    "truncated": truncated,
    "candidates": [str(path) for path in sorted(candidates)],
    "errors": errors,
  }


def inspect_cpm_bundle(bundle: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
  """Check the known poker CPM/base/data identity using metadata only."""

  selected = KNOWN_CPM_BUNDLE if bundle is None else bundle
  checkpoint = Path(selected["checkpoint"]).expanduser().resolve()
  base = Path(selected["base_checkpoint"]).expanduser().resolve()
  data = Path(selected["data_root"]).expanduser().resolve()
  checks = {
    "cpm_weights_present": checkpoint.is_file() and checkpoint.stat().st_size > 0,
    "training_run_binds_data_root": False,
    "recovery_state_cpm": False,
    "base_h30_d32_pi05": False,
    "base_weights_present": False,
    "base_source_matches_dataset": False,
    "normalizer_exact_match": False,
    "training_dependencies_present": False,
  }
  errors: list[str] = []
  try:
    run = _read_object(checkpoint.parent / "run_config.json")
    contract = run.get("contract", {})
    checks["training_run_binds_data_root"] = (
      Path(contract.get("dataset_root", "")).expanduser().resolve() == data
    )
    checks["recovery_state_cpm"] = (
      contract.get("model_variant") == "recovery-state-cpm"
      and contract.get("recovery_state_cpm", {}).get("schema")
      == "touchguide_recovery_state_cpm_v1"
    )
  except (OSError, ValueError, json.JSONDecodeError) as error:
    _error(errors, "cpm_run_config", error)

  base_config = None
  try:
    base_config = _read_object(base / "config.json")
    checks["base_h30_d32_pi05"] = _matches(
      base_config,
      {
        "action_dim": 32,
        "action_horizon": 30,
        "pi05": True,
        "max_token_len": 200,
        "discrete_state_input": True,
      },
    )
  except (OSError, ValueError, json.JSONDecodeError) as error:
    _error(errors, "base_config", error)
  base_weights = base / "model.safetensors"
  checks["base_weights_present"] = (
    base_weights.is_file() and base_weights.stat().st_size > 0
  )

  try:
    manifest = _read_object(data / "meta/manifest.json")
    expected_source = Path(manifest["sources"]["base_checkpoint"]).resolve()
    observed_source = Path(base_config["source_checkpoint"]).resolve()
    checks["base_source_matches_dataset"] = expected_source == observed_source
  except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
    _error(errors, "base_source_matches_dataset", error)

  try:
    base_norm = _read_object(base / "assets/normalizer/norm_stats.json")
    data_norm = _read_object(data / "meta/action_norm_stats.json")
    checks["normalizer_exact_match"] = base_norm.get("norm_stats") == data_norm.get(
      "norm_stats"
    )
  except (OSError, ValueError, json.JSONDecodeError) as error:
    _error(errors, "normalizer_exact_match", error)

  dependencies = (
    "config/cpm.json",
    "meta/tactile_scale.json",
    "meta/action_norm_stats.json",
    "meta/cpm_normalization_v1.json",
    "meta/negative_policy.json",
  )
  checks["training_dependencies_present"] = all(
    (data / relative).is_file() for relative in dependencies
  )
  matched = all(checks.values())
  if not matched and not errors:
    errors.append("one or more CPM metadata/file-presence checks failed")
  return {
    "task": "poker-draw",
    "checkpoint": str(checkpoint),
    "base_checkpoint": str(base),
    "data_root": str(data),
    "matched": matched,
    "status": "known_checkpoint_matched" if matched else "known_checkpoint_preflight_failed",
    "checks": checks,
    "errors": errors,
    "weights_loaded": False,
    "weights_hashed": False,
  }


def _cpm_row(task: str) -> dict[str, Any]:
  if task == "poker-draw":
    result = inspect_cpm_bundle()
    return {
      **result,
      "adapter_status": "implemented_poker_only",
      "preflight_only": True,
    }
  return {
    "task": task,
    "status": "awaiting_task_specific_cpm_and_base",
    "matched": False,
    "adapter_status": "not_registered_without_task_native_assets",
    "reason": (
      "No task-native CPM scorer, H30 pi0.5 base, matched normalizer, and training "
      "data contract have been registered and verified by this adapter; the "
      "poker-bound TouchGuide policy cannot serve this task."
    ),
    "preflight_only": True,
  }


def _ftp1_row(
  task: str,
  *,
  roots: Mapping[str, Sequence[str | Path]] | None,
) -> dict[str, Any]:
  scan = discover_ftp_candidates(task, roots=roots)
  candidates = list(scan["candidates"])
  if task == "poker-draw" and KNOWN_POKER_FTP1_CHECKPOINT.is_dir():
    known = str(KNOWN_POKER_FTP1_CHECKPOINT.resolve())
    if known not in candidates:
      candidates.append(known)
  inspections = (
    [inspect_ftp_checkpoint(path, task) for path in sorted(candidates)]
    if task not in FTP1_PENDING_ADAPTERS
    else []
  )
  matched = [row for row in inspections if row["matched"]]

  if task in FTP1_PENDING_ADAPTERS:
    status = "adapter_pending"
    adapter_status = "not_implemented"
  elif matched and task == "poker-draw":
    status = "known_checkpoint_matched"
    adapter_status = "implemented"
  elif matched:
    status = "checkpoint_contract_matched_awaiting_real_closed_loop_acceptance"
    adapter_status = "implemented_shared_diagnostic"
  else:
    status = "adapter_ready_awaiting_checkpoint"
    adapter_status = "implemented_shared_diagnostic"

  contract = FTP1_TASK_CONTRACTS.get(task)
  return {
    "task": task,
    "status": status,
    "matched": bool(matched),
    "adapter_status": adapter_status,
    "domain_name": contract["domain_name"] if contract else None,
    "instruction": contract["instruction"] if contract else None,
    "scan": scan,
    "candidate_inspections": inspections,
    "preflight_only": True,
    "real_model_loaded": False,
  }


def build_readiness(
  *,
  task: str | None = None,
  checkpoint: str | Path | None = None,
  scan_roots: Mapping[str, Sequence[str | Path]] | None = None,
) -> dict[str, Any]:
  """Build the JSON-ready matrix or inspect one explicit FTP-1 checkpoint."""

  if task is not None and task not in TASKS:
    raise ValueError(f"unknown task: {task!r}")
  if checkpoint is not None:
    if task is None:
      raise ValueError("task is required with an explicit checkpoint")
    return {
      "schema": SCHEMA,
      "mode": "explicit_ftp1_checkpoint",
      "preflight_only": True,
      "inspection": inspect_ftp_checkpoint(checkpoint, task),
      "limitations": [
        "No weight tensor was loaded or hashed.",
        "A matched contract is not a real-model or closed-loop acceptance result.",
      ],
    }

  selected_tasks = (task,) if task else TASKS
  rows = [
    {
      "task": selected,
      "cpm": _cpm_row(selected),
      "ftp1": _ftp1_row(selected, roots=scan_roots),
    }
    for selected in selected_tasks
  ]
  return {
    "schema": SCHEMA,
    "mode": "task_matrix",
    "preflight_only": True,
    "task_count": len(rows),
    "tasks": rows,
    "registered_shared_ftp1_tasks": list(SHARED_TACTILE_TASKS),
    "limitations": [
      "No checkpoint weights were loaded or hashed and no simulator was started.",
      "Normalization files alone never count as a trained checkpoint.",
      "No non-poker CPM checkpoint/base pair has been accepted.",
      "A shared FTP-1 adapter contract is diagnostic and does not imply a task corpus or checkpoint exists.",
      "Old deployment manifests must be prepared again after task-contract changes.",
    ],
  }


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=TASKS)
  parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Explicit FTP-1 checkpoint directory; requires --task and performs metadata-only validation.",
  )
  args = parser.parse_args(argv)
  if args.checkpoint is not None and args.task is None:
    parser.error("--checkpoint requires --task")
  report = build_readiness(task=args.task, checkpoint=args.checkpoint)
  print(json.dumps(report, indent=2, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
