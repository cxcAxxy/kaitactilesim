#!/usr/bin/env python3
"""Read-only CPU contract checks for all six new H50 checkpoints."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
from pathlib import Path

from run_pi05_checkpoint_worker import NEW_GROUPS


def main() -> int:
  simulation_root = Path(__file__).resolve().parents[2]
  openpi_root = simulation_root.parent / "openpi"
  if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("read-only preflight requires JAX_PLATFORMS=cpu")
  sys.path.insert(0, str(simulation_root / "scripts/workcell"))
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    load_checkpoint_training_provenance,
  )
  from openpi.training import config

  for module in (config,):
    if not Path(module.__file__).resolve().is_relative_to(openpi_root):
      raise RuntimeError("preflight imported mutable OpenPI sources")
  results = []
  for task, step, checkpoint in NEW_GROUPS:
    if task == "usb-insert":
      prepare_name = "prepare_usb_pi05_deployment"
      prepare = importlib.import_module(prepare_name)
      model_config, normalizer, binding = prepare.checkpoint_contract(checkpoint)
    elif task == "poker-draw":
      prepare_name = "prepare_poker_pi05_deployment"
      prepare = importlib.import_module(prepare_name)
      model_config, normalizer, binding = prepare.checkpoint_contract(checkpoint)
    else:
      prepare_name = "prepare_shared_task_pi05_deployment"
      prepare = importlib.import_module(prepare_name)
      binding = load_checkpoint_training_provenance(checkpoint)
      model_config = binding["record"]["config_name"]
      normalizer = binding["record"]["normalizer"]["asset_id"]
    if not Path(prepare.__file__).resolve().is_relative_to(simulation_root):
      raise RuntimeError("preflight imported mutable simulation sources")
    args = [
      "--checkpoint",
      str(checkpoint),
      "--output-dir",
      str(simulation_root.parent / "preflight_no_write" / task / str(step)),
      "--openpi-root",
      str(openpi_root),
      "--dry-run",
    ]
    if task == "install-ram":
      args += ["--task", task]
    with contextlib.redirect_stdout(io.StringIO()):
      if prepare.main(args) not in (None, 0):
        raise RuntimeError(f"preflight rejected {task} step{step}")
    record = binding["record"]
    runtime_config = config.get_config(model_config)
    if (
      record["training_action_transform"]["action_horizon"] != 50
      or runtime_config.model.action_horizon != 50
    ):
      raise RuntimeError("checkpoint and runtime must both use H50")
    execution = record["action_contract"]["evaluation_execution"]
    if execution["physical_substep_interpolation_required"] != "linear":
      raise RuntimeError("new H50 checkpoints must use linear physics execution")
    result = {
      "task": task,
      "step": step,
      "checkpoint": str(checkpoint),
      "valid": True,
      "model_config_name": model_config,
      "prediction_horizon": 50,
      "execute_steps": 50,
      "normalizer_asset_id": normalizer,
      "use_quantile_norm": record["normalizer"]["use_quantile_norm"],
      "required_action_interpolation": "linear",
      "binding_status": binding["status"],
      "source_environment": record["source_environment"],
      "formal_metrics_valid": False,
      "model_weights_loaded": False,
      "outputs_written": False,
    }
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
  print(
    json.dumps(
      {
        "all_six_valid": True,
        "count": len(results),
        "simulation_root": str(simulation_root),
        "model_project_root": str(openpi_root),
      }
    ),
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
