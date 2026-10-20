#!/usr/bin/env python3
"""Unified, manifest-bound policy deployment entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
  sys.path.insert(0, str(SOURCE))

from kaihand_tactile_env.pipeline.catalog import MODEL_FAMILIES, TASKS  # noqa: E402
from kaihand_tactile_env.pipeline.deployment import (  # noqa: E402
  CapabilityError,
  DeploymentError,
  DeploymentInterrupted,
  adapter_catalog,
  build_prepare_plan,
  build_serve_plan,
  execute_prepare,
  execute_serve,
  load_deployment_manifest,
  run_deployment,
  verify_deployment,
  write_pipeline_deployment,
)
from kaihand_tactile_env.pipeline.runtime import load_runtime_profile  # noqa: E402


def _runtime_options(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--runtime-config",
    type=Path,
    help="TOML/JSON file containing named model Python/project environments",
  )
  parser.add_argument(
    "--runtime",
    help="Runtime name under [runtimes.NAME] (or use config default_runtime)",
  )


def _backend_arguments(values: list[str]) -> tuple[str, ...]:
  if values and values[0] == "--":
    values = values[1:]
  return tuple(values)


def _execute_steps(value: str) -> int | str:
  if value.strip().lower() == "horizon":
    return "horizon"
  try:
    result = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      "execute-steps must be a positive integer or 'horizon'"
    ) from error
  if result <= 0:
    raise argparse.ArgumentTypeError("execute-steps must be positive")
  return result


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--list-support",
    action="store_true",
    help="print every task/model stage capability and exit",
  )
  commands = parser.add_subparsers(dest="command")

  prepare = commands.add_parser(
    "prepare", help="freeze a checkpoint with its existing task/model validator"
  )
  prepare.add_argument("--task", choices=TASKS, required=True)
  prepare.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
  prepare.add_argument("--checkpoint", type=Path, required=True)
  prepare.add_argument("--output-dir", type=Path, required=True)
  _runtime_options(prepare)
  prepare.add_argument("--dry-run", action="store_true")
  prepare.add_argument(
    "backend_args",
    nargs=argparse.REMAINDER,
    help="model-specific prepare flags after --",
  )

  verify = commands.add_parser(
    "verify", help="verify an existing frozen deployment without rewriting it"
  )
  verify.add_argument("--deployment-manifest", type=Path, required=True)
  verify.add_argument("--model-family", choices=MODEL_FAMILIES)
  _runtime_options(verify)
  verify.add_argument(
    "--verify-hashes",
    action="store_true",
    help="LingBot only: re-hash every checkpoint shard",
  )
  verify.add_argument("--dry-run", action="store_true")

  serve = commands.add_parser(
    "serve", help="serve a websocket-sidecar deployment in the foreground"
  )
  serve.add_argument("--deployment-manifest", type=Path, required=True)
  serve.add_argument("--model-family", choices=MODEL_FAMILIES)
  serve.add_argument(
    "--host",
    help="requested bind host (fixed-host backends reject unsupported values)",
  )
  serve.add_argument("--port", type=int)
  _runtime_options(serve)
  serve.add_argument(
    "--validate-only",
    action="store_true",
    help="run the strict available verifier without starting a server",
  )
  serve.add_argument("--verify-hashes", action="store_true")
  serve.add_argument("--dry-run", action="store_true")
  serve.add_argument(
    "backend_args",
    nargs=argparse.REMAINDER,
    help="server-specific flags after --",
  )

  run = commands.add_parser(
    "run", help="manage a model sidecar when needed and invoke evaluate-policy"
  )
  run.add_argument("--deployment-manifest", type=Path, required=True)
  run.add_argument(
    "--model-family",
    choices=MODEL_FAMILIES,
    help="normally inferred; required to select pi05+trex from a pi0.5 base manifest",
  )
  run.add_argument("--output-dir", type=Path, required=True)
  run.add_argument(
    "--service-mode",
    choices=("managed", "external"),
    default="managed",
    help="start an owned sidecar or connect to a caller-owned endpoint",
  )
  run.add_argument("--endpoint", help="WebSocket URI/host:port")
  _runtime_options(run)
  run.add_argument("--num-trials", type=int, default=1)
  run.add_argument("--seed-start", type=int, default=0)
  run.add_argument("--video-count", type=int, default=0)
  run.add_argument("--execute-steps", type=_execute_steps)
  run.add_argument("--max-sim-seconds", type=float, default=90.0)
  run.add_argument("--guard-mode", choices=("observe", "strict"), default="observe")
  run.add_argument(
    "--smoke-seconds",
    type=float,
    help=(
      "diagnostic-only short maximum simulation time; never reported as a "
      "formal task metric"
    ),
  )
  run.add_argument("--server-timeout", type=float, default=600.0)
  run.add_argument("--dry-run", action="store_true")
  run.add_argument(
    "evaluator_args",
    nargs=argparse.REMAINDER,
    help="task/model evaluator flags after --",
  )
  return parser


def _json(value: object) -> None:
  print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.list_support:
    if args.command is not None:
      parser.error("--list-support cannot be combined with a subcommand")
    _json({
      "schema": "kaihand_deployment_support_v1",
      "schema_version": 1,
      "stage": "deploy",
      "operation": "list-support",
      "tasks": list(TASKS),
      "model_families": list(MODEL_FAMILIES),
      "deployments": list(adapter_catalog()),
    })
    return 0
  if args.command is None:
    parser.error("choose prepare, verify, serve or run; or use --list-support")

  profile = load_runtime_profile(args.runtime_config, args.runtime)
  if args.command == "prepare":
    backend_args = _backend_arguments(args.backend_args)
    plan = build_prepare_plan(
      args.task,
      args.model_family,
      checkpoint=args.checkpoint,
      output_dir=args.output_dir,
      profile=profile,
      backend_args=backend_args,
    )
    if args.dry_run:
      _json({
        "schema": "kaihand_deployment_prepare_plan_v1",
        "schema_version": 1,
        "stage": "deploy",
        "operation": "prepare",
        "status": "planned",
        "task": args.task,
        "model_family": args.model_family,
        "runtime_profile": None if profile is None else profile.as_dict(),
        "prepare": plan.as_dict(),
      })
      return 0
    if profile is not None:
      profile.validate(require_python=True)
    returncode = execute_prepare(plan)
    if returncode:
      return returncode
    manifest = args.output_dir.expanduser().resolve() / "deployment_manifest.json"
    verification = verify_deployment(
      manifest,
      model_family=args.model_family,
      profile=profile,
      execute_external=True,
    )
    pipeline_record = write_pipeline_deployment(
      args.output_dir,
      task=args.task,
      model_family=args.model_family,
      prepare_plan=plan,
      profile=profile,
      verification=verification,
    )
    _json({
      "schema": "kaihand_deployment_prepare_result_v1",
      "schema_version": 1,
      "stage": "deploy",
      "operation": "prepare",
      "status": "prepared",
      "deployment_manifest": str(manifest),
      "pipeline_deployment": str(pipeline_record),
      "verification": verification,
    })
    return 0

  if args.command == "verify":
    if profile is not None and not args.dry_run:
      profile.validate()
    report = verify_deployment(
      args.deployment_manifest,
      model_family=args.model_family,
      profile=profile,
      execute_external=not args.dry_run,
      verify_hashes=args.verify_hashes,
    )
    _json(report)
    return 0

  if args.command == "serve":
    manifest_path, manifest, adapter = load_deployment_manifest(
      args.deployment_manifest,
      model_family=args.model_family,
    )
    plan = build_serve_plan(
      manifest_path,
      manifest,
      adapter,
      profile=profile,
      host=args.host,
      port=args.port,
      backend_args=_backend_arguments(args.backend_args),
    )
    if args.dry_run:
      _json({
        "schema": "kaihand_deployment_serve_plan_v1",
        "schema_version": 1,
        "stage": "deploy",
        "operation": "serve",
        "status": "planned",
        "task": adapter.task,
        "model_family": adapter.model_family,
        "runtime_mode": adapter.runtime_mode,
        "serve": plan.as_dict(),
      })
      return 0
    if profile is not None:
      profile.validate(require_python=True)
    verification = verify_deployment(
      manifest_path,
      model_family=adapter.model_family,
      profile=profile,
      execute_external=True,
      verify_hashes=args.verify_hashes,
    )
    if args.validate_only:
      _json(verification)
      return 0
    return execute_serve(plan)

  if args.command == "run":
    record = run_deployment(
      args.deployment_manifest,
      output_dir=args.output_dir,
      model_family=args.model_family,
      profile=profile,
      service_mode=args.service_mode,
      endpoint=args.endpoint,
      num_trials=args.num_trials,
      seed_start=args.seed_start,
      video_count=args.video_count,
      execute_steps=args.execute_steps,
      max_sim_seconds=args.max_sim_seconds,
      guard_mode=args.guard_mode,
      smoke_seconds=args.smoke_seconds,
      server_timeout=args.server_timeout,
      evaluator_args=_backend_arguments(args.evaluator_args),
      dry_run=args.dry_run,
    )
    _json(record)
    return int(record.get("evaluation_returncode", 0))

  raise AssertionError(args.command)


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except DeploymentInterrupted as error:
    print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(128 + error.signum) from error
  except (
    CapabilityError,
    DeploymentError,
    FileExistsError,
    OSError,
    ValueError,
  ) as error:
    print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(2) from error
