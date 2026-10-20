#!/usr/bin/env python3
"""Freeze CPM or FTP-1 inference artifacts and their native control contract."""
# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kaihand_tactile_env.shared.tactile_deployment import prepare_manifest
from kaihand_tactile_env.shared.tactile_task_contracts import FTP1_TASK_CONTRACTS


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model-family", choices=("pi05+cpm", "ftp1"), required=True)
  parser.add_argument("--task", choices=tuple(FTP1_TASK_CONTRACTS), default="poker-draw")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--model-project", type=Path, required=True)
  parser.add_argument("--model-python", type=Path, required=True)
  parser.add_argument("--execute-steps", type=int, default=5)
  parser.add_argument("--base-checkpoint", type=Path)
  parser.add_argument("--data-root", type=Path)
  parser.add_argument("--tokenizer-model", type=Path)
  parser.add_argument("--guidance-scale", type=float, default=1.0)
  parser.add_argument("--guidance-steps", type=int, default=3)
  args = parser.parse_args(argv)
  payload = prepare_manifest(family=args.model_family, checkpoint=args.checkpoint, task=args.task,
    project=args.model_project, python=args.model_python, execute_steps=args.execute_steps,
    base_checkpoint=args.base_checkpoint, data_root=args.data_root, tokenizer_model=args.tokenizer_model,
    guidance_scale=args.guidance_scale, guidance_steps=args.guidance_steps)
  args.output_dir.mkdir(parents=True, exist_ok=False)
  path = args.output_dir / "deployment_manifest.json"
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
  print(path)


if __name__ == "__main__":
  main()
