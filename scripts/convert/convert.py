#!/usr/bin/env python3
"""Unified Raw -> model data or model-neutral LeRobot v3 conversion entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaihand_tactile_env.pipeline.conversion import (
  FORMATS,
  TASKS,
  adapter_catalog,
  adapter_for,
  default_cameras_for,
  inspect_raw_dataset,
  source_contract,
)
from kaihand_tactile_env.pipeline.runtime import RuntimeProfile, load_runtime_profile
from kaihand_tactile_env.shared.cameras import TRAINING_CAMERA_NAMES

ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "kaihand_conversion_plan_v1"
RESULT_SCHEMA = "kaihand_pipeline_conversion_v1"
PIPELINE_MANIFEST = "pipeline_conversion.json"
_BACKEND_MANIFEST_PATHS = (
  "conversion_manifest.json",
  "dataset_manifest.json",
  "source_snapshot_manifest.json",
  "meta/kaihand_card_conversion_manifest.json",
  "meta/kaihand_card_source_snapshot.json",
  "meta/kaihand_conversion.json",
  "meta/kaihand_conversion_manifest.json",
  "meta/kaihand_fast_pi05_conversion.json",
  "meta/kaihand_schema.json",
  "meta/kaihand_source_snapshot.json",
  "meta/kaihand_source_episodes.jsonl",
  "meta/kaihand_usb_conversion_manifest.json",
  "meta/kaihand_usb_source_snapshot.json",
  "meta/info.json",
)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--list-support", action="store_true",
    help="print the semantic-adapter registry and exit without reading Raw data",
  )
  parser.add_argument("--input-dir", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--format", choices=FORMATS)
  parser.add_argument("--task", default="auto", choices=("auto", *TASKS))
  parser.add_argument(
    "--cameras", nargs="+",
    choices=TRAINING_CAMERA_NAMES,
    help="model input cameras; defaults to the adapter's fixed set or head",
  )
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument(
    "--expected-episodes", type=int, default=0,
    help="fail unless this many selected Raw episodes are found; 0 accepts any count",
  )
  parser.add_argument("--staging-root", type=Path)
  parser.add_argument(
    "--work-dir", type=Path,
    help="persistent checkpoint directory for LeRobot v3 conversion",
  )
  parser.add_argument("--verify-source-hash", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--instruction")
  parser.add_argument("--dataset-name")
  parser.add_argument(
    "--repo-id", help="LeRobot v3 repository identifier stored in metadata",
  )
  parser.add_argument(
    "--openpi-root", type=Path, default=ROOT.parent / "openpi",
  )
  parser.add_argument("--pi05-python", type=Path)
  parser.add_argument(
    "--lerobot-v3-python", type=Path,
    help="Python with LeRobot 0.4.x/v3 support; required if this Python lacks lerobot",
  )
  parser.add_argument(
    "--runtime-config", type=Path,
    help="TOML/JSON file containing named conversion runtimes",
  )
  parser.add_argument(
    "--runtime",
    help="runtime name in --runtime-config (or use its default_runtime)",
  )
  parser.add_argument("--dry-run", action="store_true")
  raw_argv = list(argv) if argv is not None else sys.argv[1:]
  args = parser.parse_args(raw_argv)
  args._openpi_root_explicit = any(
    value == "--openpi-root" or value.startswith("--openpi-root=")
    for value in raw_argv
  )
  if args.workers <= 0:
    parser.error("--workers must be positive")
  if args.expected_episodes < 0:
    parser.error("--expected-episodes must be nonnegative")
  if not args.list_support:
    for name in ("input_dir", "output_dir", "format"):
      if getattr(args, name) is None:
        parser.error(f"--{name.replace('_', '-')} is required")
  return args


def _same_path(first: Path, second: Path) -> bool:
  return first.expanduser().resolve() == second.expanduser().resolve()


def _selected_path(
  explicit: Path | None,
  configured: Path | None,
  *,
  explicit_option: str,
) -> Path | None:
  if (
    explicit is not None
    and configured is not None
    and not _same_path(explicit, configured)
  ):
    raise ValueError(
      f"{explicit_option} conflicts with the selected runtime profile"
    )
  return explicit if explicit is not None else configured


def _openpi_root(args, runtime_profile: RuntimeProfile | None) -> Path:
  configured = runtime_profile.project_root if runtime_profile is not None else None
  explicit = (
    args.openpi_root
    if getattr(args, "_openpi_root_explicit", runtime_profile is None)
    else None
  )
  selected = _selected_path(
    explicit,
    configured,
    explicit_option="--openpi-root",
  )
  return (selected or args.openpi_root).expanduser().resolve()


def _python_for(
  explicit: Path | None,
  runtime_profile: RuntimeProfile | None,
  *,
  explicit_option: str,
  fallback: Path | str,
) -> Path | str:
  configured = runtime_profile.python if runtime_profile is not None else None
  return _selected_path(
    explicit,
    configured,
    explicit_option=explicit_option,
  ) or fallback


def _backend_path(contract, openpi_root: Path | None = None) -> Path:
  if contract.backend_location == "workcell":
    return ROOT / "scripts/workcell" / contract.backend
  if contract.backend_location == "runtime-project" and openpi_root is not None:
    return openpi_root / "examples/kaihand" / contract.backend
  raise ValueError(
    f"adapter {contract.backend!r} has unsupported backend location "
    f"{contract.backend_location!r}"
  )


def _command(args, dataset, contract, runtime_profile=None):
  if contract.output_format == "lerobot-v3":
    command = [
      str(_python_for(
        args.lerobot_v3_python,
        runtime_profile,
        explicit_option="--lerobot-v3-python",
        fallback=sys.executable,
      )),
      str(_backend_path(contract)),
      "--input-dir", str(dataset.root),
      "--output-dir", str(args.output_dir),
      "--task", dataset.task,
      "--expected-episodes", str(len(dataset.episodes)),
      "--workers", str(args.workers),
    ]
    if args.repo_id:
      command += ["--repo-id", args.repo_id]
    if args.work_dir:
      command += ["--work-dir", str(args.work_dir)]
    if args.staging_root:
      command += ["--staging-root", str(args.staging_root)]
    if args.verify_source_hash:
      command.append("--verify-source-hash")
    if args.resume:
      command.append("--resume")
    return command
  if contract.output_format == "egosteer":
    cmd = [
      str(_python_for(
        None,
        runtime_profile,
        explicit_option="runtime python",
        fallback=sys.executable,
      )),
      str(_backend_path(contract)),
      "--input-dir", str(dataset.root),
      "--output-dir", str(args.output_dir),
    ]
    if contract.worker_option == "--workers":
      cmd += ["--workers", str(args.workers)]
    if contract.supports("expected_episodes"):
      cmd += ["--expected-episodes", str(len(dataset.episodes))]
    if contract.supports("cameras"):
      cmd += ["--cameras", *dataset.cameras]
    if args.staging_root and contract.supports("staging_root"):
      cmd += ["--staging-root", str(args.staging_root)]
    if args.verify_source_hash and contract.supports("verify_source_hash"):
      cmd.append("--verify-source-hash")
    if contract.supports("task"):
      cmd += ["--task", dataset.task]
    if args.instruction and contract.supports("instruction"):
      cmd += ["--instruction", args.instruction]
    if args.dataset_name and contract.supports("dataset_name"):
      cmd += ["--dataset-name", args.dataset_name]
    return cmd
  openpi_root = _openpi_root(args, runtime_profile)
  python = _python_for(
    args.pi05_python,
    runtime_profile,
    explicit_option="--pi05-python",
    fallback=openpi_root / ".venv-pi05/bin/python",
  )
  backend_path = _backend_path(contract, openpi_root)
  cmd = [
    str(python),
    str(backend_path),
    "--input-dir", str(dataset.root),
    "--output-dir", str(args.output_dir),
    "--expected-episodes", str(len(dataset.episodes)),
    "--fingerprint-workers", str(args.workers),
  ]
  if contract.supports("image_writer_threads"):
    cmd += [
      "--image-writer-processes", "0",
      "--image-writer-threads", str(args.workers),
    ]
  if contract.supports("openpi_root"):
    cmd += ["--openpi-root", str(openpi_root)]
  if args.staging_root and contract.supports("staging_root"):
    cmd += ["--staging-root", str(args.staging_root)]
  if args.verify_source_hash and contract.supports("verify_source_hash"):
    cmd.append("--verify-source-hash")
  if contract.supports("task"):
    cmd += ["--task", dataset.task]
  if contract.supports("cameras"):
    cmd += ["--cameras", *dataset.cameras]
  if contract.supports("instruction") or contract.supports("dataset_name"):
    if args.instruction and contract.supports("instruction"):
      cmd += ["--instruction", args.instruction]
    if args.dataset_name and contract.supports("dataset_name"):
      cmd += ["--dataset-name", args.dataset_name]
  return cmd


def _session_id(root, source):
  relative = str(source.relative_to(root))
  digest = hashlib.sha256(relative.encode()).hexdigest()[:10]
  safe = "-".join(part for part in source.stem.replace("_", "-").split("-") if part)
  return f"{safe[:80]}-{digest}"


def _export_tict_job(job):
  from kaihand_tactile_env.shared.tict_export import export_tict_episode

  (
    source,
    destination,
    session_id,
    camera,
    source_sha256,
    source_sha256_verification,
    source_size_bytes,
    source_mtime_ns,
  ) = job
  return export_tict_episode(
    source,
    destination,
    session_id=session_id,
    camera=camera,
    source_sha256=source_sha256,
    source_sha256_verification=source_sha256_verification,
    source_size_bytes=source_size_bytes,
    source_mtime_ns=source_mtime_ns,
  )


def _write_json_atomic(path, payload):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  os.replace(temporary, path)


def _capture_sidecar_sha256(source: Path) -> str | None:
  """Read the acquisition digest without re-reading the large HDF5 payload."""

  sidecar = source.with_suffix(".json")
  if not sidecar.is_file():
    return None
  try:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    return None
  if not isinstance(payload, dict):
    return None
  if payload.get("episode") != source.name:
    return None
  value = payload.get("sha256", payload.get("hdf5_sha256"))
  if (
    not isinstance(value, str)
    or len(value) != 64
    or any(character not in "0123456789abcdef" for character in value.lower())
  ):
    return None
  return value.lower()


def _source_snapshot(dataset) -> tuple[dict[str, Any], dict[str, Any]]:
  """Capture file identity and literal Raw contracts around one metadata read."""

  before = {path: path.stat() for path in dataset.episodes}
  contract = source_contract(dataset)
  rows = []
  for path in dataset.episodes:
    initial = before[path]
    final = path.stat()
    initial_identity = (
      initial.st_dev,
      initial.st_ino,
      initial.st_size,
      initial.st_mtime_ns,
    )
    final_identity = (
      final.st_dev,
      final.st_ino,
      final.st_size,
      final.st_mtime_ns,
    )
    if initial_identity != final_identity:
      raise RuntimeError(f"source changed while reading its contract: {path}")
    capture_sha256 = _capture_sidecar_sha256(path)
    rows.append({
      "path": path.relative_to(dataset.root).as_posix(),
      "bytes": final.st_size,
      "mtime_ns": final.st_mtime_ns,
      "device": final.st_dev,
      "inode": final.st_ino,
      "capture_sha256": capture_sha256,
      "capture_sha256_status": (
        "declared_by_capture_sidecar_not_recomputed"
        if capture_sha256 is not None
        else "not_declared"
      ),
    })
  return ({"root": str(dataset.root), "episodes": rows}, contract)


def _assert_source_unchanged(dataset, identity: dict[str, Any]) -> None:
  expected = {
    row["path"]: (
      row["device"], row["inode"], row["bytes"], row["mtime_ns"]
    )
    for row in identity["episodes"]
  }
  for source in dataset.episodes:
    relative = source.relative_to(dataset.root).as_posix()
    stat = source.stat()
    actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if actual != expected[relative]:
      raise RuntimeError(f"source changed during conversion: {source}")


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _declared_manifest_contracts(payload: dict[str, Any]) -> dict[str, Any]:
  """Copy only fields that a backend itself labels as camera/action contracts."""

  fields = {}
  selected_names = {
    "action_alignment",
    "action_contract",
    "action_representation",
    "action_semantics",
    "camera_contract",
    "camera_views",
    "cameras",
    "observation_contract",
  }

  def visit(value: dict[str, Any], prefix: str = "") -> None:
    for key, item in value.items():
      path = f"{prefix}.{key}" if prefix else key
      if key.lower() in selected_names:
        fields[path] = item
      elif isinstance(item, dict):
        visit(item, path)

  visit(payload)
  return fields


def _backend_manifest_references(output: Path) -> list[dict[str, Any]]:
  references = []
  source_identity_paths = {
    "source_snapshot_manifest.json",
    "meta/kaihand_card_source_snapshot.json",
    "meta/kaihand_source_snapshot.json",
    "meta/kaihand_source_episodes.jsonl",
    "meta/kaihand_usb_source_snapshot.json",
  }
  for relative in _BACKEND_MANIFEST_PATHS:
    path = output / relative
    if not path.is_file() or path.is_symlink():
      continue
    reference: dict[str, Any] = {
      "path": relative,
      "sha256": _sha256_file(path),
      "role": (
        "source_identity" if relative in source_identity_paths else "manifest"
      ),
    }
    if path.suffix == ".json":
      try:
        payload = json.loads(path.read_text(encoding="utf-8"))
      except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"backend manifest is unreadable: {path}: {error}") from error
      if not isinstance(payload, dict):
        raise ValueError(f"backend manifest must be an object: {path}")
      if "schema" in payload:
        reference["schema"] = payload["schema"]
      if "schema_version" in payload:
        reference["schema_version"] = payload["schema_version"]
      declared = _declared_manifest_contracts(payload)
      if declared:
        reference["declared_contracts"] = declared
      source_references = {
        key: payload[key]
        for key in (
          "input_dir",
          "source_collection",
          "source_manifest",
          "source_root",
          "source_episodes",
        )
        if key in payload
      }
      if source_references:
        reference["declared_source_identity"] = source_references
    references.append(reference)
  return references


def _pythonpath_prepend(
  args, contract, runtime_profile: RuntimeProfile | None
) -> tuple[str, ...]:
  if contract.runtime == "in-process":
    return ()
  paths = [ROOT / "src"]
  if contract.runtime == "pi05-python":
    openpi_root = _openpi_root(args, runtime_profile)
    paths.extend((
      openpi_root / "src",
      openpi_root / "packages/openpi-client/src",
    ))
  return tuple(str(path) for path in paths)


def _uses_external_environment(
  command: list[str], runtime_profile: RuntimeProfile | None
) -> bool:
  if runtime_profile is not None:
    return True
  try:
    return not _same_path(Path(command[0]), Path(sys.executable))
  except (OSError, RuntimeError):
    return True


def _subprocess_environment(
  args,
  contract,
  runtime_profile: RuntimeProfile | None,
) -> dict[str, str]:
  environment = (
    runtime_profile.environment()
    if runtime_profile is not None
    else dict(os.environ)
  )
  existing = environment.get("PYTHONPATH", "")
  entries = [*_pythonpath_prepend(args, contract, runtime_profile)]
  entries.extend(value for value in existing.split(os.pathsep) if value)
  environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
  return environment


def _runtime_record(
  args,
  contract,
  runtime_profile: RuntimeProfile | None,
  command: list[str] | None,
) -> dict[str, Any]:
  project_root = None
  if contract.supports("openpi_root"):
    project_root = str(_openpi_root(args, runtime_profile))
  prepend = (
    _pythonpath_prepend(args, contract, runtime_profile)
    if command is not None and _uses_external_environment(command, runtime_profile)
    else ()
  )
  return {
    "kind": contract.runtime,
    "executable": command[0] if command is not None else sys.executable,
    "project_root": project_root,
    "pythonpath_prepend": list(prepend),
    "profile": runtime_profile.as_dict() if runtime_profile is not None else None,
  }


def _conversion_plan(
  args,
  dataset,
  contract,
  runtime_profile: RuntimeProfile | None,
  command: list[str] | None,
  source_identity: dict[str, Any],
  raw_contract: dict[str, Any],
) -> dict[str, Any]:
  return {
    "schema": PLAN_SCHEMA,
    "schema_version": 1,
    "stage": "convert",
    "task": dataset.task,
    "format": args.format,
    "adapter": contract.backend,
    "episodes": len(dataset.episodes),
    "cameras": dataset.cameras,
    "command": command,
    "resume": args.resume,
    "workers": {
      "requested": args.workers,
      "backend_option": contract.worker_option,
      "semantics": contract.worker_semantics,
    },
    "runtime": _runtime_record(args, contract, runtime_profile, command),
    "source_identity": source_identity,
    "source_contract": raw_contract,
  }


def _publish_pipeline_manifest(
  args,
  dataset,
  plan: dict[str, Any],
  source_identity: dict[str, Any],
) -> Path:
  output = args.output_dir.expanduser().resolve()
  if not output.is_dir() or output.is_symlink():
    raise RuntimeError(f"successful backend did not publish an output directory: {output}")
  _assert_source_unchanged(dataset, source_identity)
  manifest_path = output / PIPELINE_MANIFEST
  if manifest_path.is_symlink():
    raise FileExistsError(
      f"refusing to replace symlinked pipeline manifest: {manifest_path}"
    )
  if manifest_path.exists():
    if not args.resume:
      raise FileExistsError(
        f"refusing to overwrite backend-owned {manifest_path}"
      )
    try:
      previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
      raise FileExistsError(
        f"refusing to overwrite backend-owned {manifest_path}: {error}"
      ) from error
    if not isinstance(previous, dict) or previous.get("schema") != RESULT_SCHEMA:
      raise FileExistsError(
        f"refusing to overwrite backend-owned {manifest_path}"
      )
  backend_manifests = _backend_manifest_references(output)
  result = {
    **plan,
    "schema": RESULT_SCHEMA,
    "schema_version": 1,
    "status": "success",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "backend_returncode": 0,
    "backend_manifests": backend_manifests,
    "backend_source_identity": [
      {"path": row["path"], "sha256": row["sha256"]}
      for row in backend_manifests
      if row["role"] == "source_identity"
    ],
  }
  _write_json_atomic(manifest_path, result)
  return manifest_path


def _validate_runtime_profile(
  contract, runtime_profile: RuntimeProfile | None, *, execute: bool
) -> None:
  if runtime_profile is None:
    return
  declared = set()
  if runtime_profile.python is not None:
    declared.add("python")
  if runtime_profile.project_root is not None:
    declared.add("project_root")
  if runtime_profile.env:
    declared.add("env")
  unsupported = declared - set(contract.runtime_profile_fields)
  if unsupported:
    raise ValueError(
      f"{contract.output_format} adapter does not consume runtime profile "
      f"fields: {', '.join(sorted(unsupported))}"
    )
  if execute:
    runtime_profile.validate()


@contextmanager
def _in_process_environment(runtime_profile: RuntimeProfile | None):
  overrides = runtime_profile.env if runtime_profile is not None else {}
  previous = {name: os.environ.get(name) for name in overrides}
  try:
    os.environ.update(overrides)
    yield
  finally:
    for name, value in previous.items():
      if value is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = value


def _existing_tict_record(
  source: Path,
  destination: Path,
  session_id: str,
  camera: str,
  *,
  expected_source_sha256: str,
) -> dict[str, Any]:
  """Validate a previously atomically-published job before resume skips it."""

  audit_path = destination / "dataset_audit.json"
  selector_path = destination / "selector_manifest.json"
  sidecar_path = (
    destination / "tict_sidecars" / session_id / "fingertip_tactile_v1.npz"
  )
  required = (audit_path, selector_path, sidecar_path)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise ValueError(
      f"cannot resume incomplete EgoTouch output {destination}; missing {missing}"
    )
  try:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(
      f"cannot resume unreadable EgoTouch output {destination}: {error}"
    ) from error
  if not isinstance(audit, dict):
    raise ValueError(f"cannot resume non-object EgoTouch audit: {audit_path}")
  sessions = audit.get("sessions", [])
  source_record = audit.get("source", {})
  if (
    audit.get("contract_version") != "kaihand-tict-release-v1"
    or audit.get("passed") is not True
    or not isinstance(sessions, list)
    or len(sessions) != 1
    or not isinstance(sessions[0], dict)
    or not isinstance(source_record, dict)
    or sessions[0].get("session_id") != session_id
    or Path(str(source_record.get("path", ""))).expanduser().resolve() != source
    or source_record.get("bytes") != source.stat().st_size
  ):
    raise ValueError(
      f"cannot resume EgoTouch output with mismatched audit: {destination}"
    )
  if source_record.get("sha256") != expected_source_sha256:
    raise ValueError(
      f"cannot resume EgoTouch output whose source digest changed: {destination}"
    )
  records = selector.get("records") if isinstance(selector, dict) else None
  if not isinstance(records, list) or not records:
    raise ValueError(f"cannot resume empty EgoTouch selector: {destination}")
  try:
    import numpy as np

    with np.load(sidecar_path, allow_pickle=False) as archive:
      metadata = json.loads(str(archive["source_metadata_json"].item()))
  except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
    raise ValueError(
      f"cannot resume invalid EgoTouch tactile sidecar {sidecar_path}: {error}"
    ) from error
  if metadata.get("camera") != camera:
    raise ValueError(
      f"cannot resume EgoTouch camera mismatch at {destination}: "
      f"expected {camera}, found {metadata.get('camera')}"
    )
  return {
    "source": str(source),
    "session_id": session_id,
    "camera": camera,
    "output": str(destination),
    "status": "verified-existing",
    "frames": len(records),
    "source_sha256": source_record.get("sha256"),
  }


def _validate_resume_manifest(output: Path, dataset) -> None:
  path = output / "conversion_manifest.json"
  if not path.is_file():
    raise ValueError(f"cannot resume EgoTouch output without manifest: {path}")
  try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f"resume manifest is unreadable: {path}: {error}") from error
  if (
    not isinstance(manifest, dict)
    or manifest.get("schema") not in {
      "kaihand_unified_egotouch_batch_v1",
      "kaihand_unified_egotouch_batch_v2",
    }
    or manifest.get("task") != dataset.task
    or tuple(manifest.get("cameras", ())) != dataset.cameras
    or manifest.get("source_root") != str(dataset.root)
    or manifest.get("source_episodes") != [
      source.relative_to(dataset.root).as_posix() for source in dataset.episodes
    ]
  ):
    raise ValueError(
      "--resume output was created for a different task/camera contract: "
      f"{path}"
    )
  expected_sessions = {
    _session_id(dataset.root, source) for source in dataset.episodes
  }
  episodes_dir = output / "episodes"
  if episodes_dir.is_dir():
    extra = {entry.name for entry in episodes_dir.iterdir()} - expected_sessions
    if extra:
      raise ValueError(f"cannot resume EgoTouch output with stale sessions: {sorted(extra)}")


def _source_identity(
  source: Path, *, verify_source_hash: bool
) -> tuple[str, str, int, int]:
  """Resolve one source digest once, then reuse it for every selected camera."""

  initial = source.stat()
  if not verify_source_hash:
    recorded = _capture_sidecar_sha256(source)
    if recorded is not None:
      final = source.stat()
      if (initial.st_size, initial.st_mtime_ns) != (
        final.st_size, final.st_mtime_ns
      ):
        raise RuntimeError(f"source changed while reading its identity: {source}")
      return recorded, "trusted_capture_sidecar", final.st_size, final.st_mtime_ns
  from kaihand_tactile_env.shared.tict_export import sha256_file

  digest = sha256_file(source)
  final = source.stat()
  if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
    raise RuntimeError(f"source changed while hashing: {source}")
  return (
    digest,
    "recomputed_by_batch_runner",
    final.st_size,
    final.st_mtime_ns,
  )


def _convert_egotouch(args, dataset):
  output = args.output_dir.expanduser().resolve()
  if args.resume and not output.is_dir():
    raise ValueError(f"cannot resume missing EgoTouch output: {output}")
  if output.exists() and not args.resume:
    raise FileExistsError(f"output exists; use --resume: {output}")
  if not args.dry_run:
    output.mkdir(parents=True, exist_ok=True)
  if args.resume and output.exists():
    _validate_resume_manifest(output, dataset)
  jobs = []
  records = []
  for source in dataset.episodes:
    session = _session_id(dataset.root, source)
    has_existing = args.resume and any(
      (output / "episodes" / session / camera).is_dir()
      for camera in dataset.cameras
    )
    if args.dry_run and not has_existing:
      source_sha256 = None
      source_sha256_verification = None
      source_size_bytes = None
      source_mtime_ns = None
    else:
      (
        source_sha256,
        source_sha256_verification,
        source_size_bytes,
        source_mtime_ns,
      ) = _source_identity(
        source, verify_source_hash=args.verify_source_hash or has_existing
      )
    for camera in dataset.cameras:
      destination = output / "episodes" / session / camera
      record = {
        "source": str(source), "session_id": session, "camera": camera,
        "output": str(destination),
      }
      if destination.is_dir() and args.resume:
        assert source_sha256 is not None
        records.append(
          _existing_tict_record(
            source,
            destination,
            session,
            camera,
            expected_source_sha256=source_sha256,
          )
        )
      else:
        jobs.append(
          (
            source,
            destination,
            session,
            camera,
            source_sha256,
            source_sha256_verification,
            source_size_bytes,
            source_mtime_ns,
            record,
          )
        )
  if args.dry_run:
    return {
      "mode": "dry-run", "jobs": [job[-1] for job in jobs],
      "existing": records,
    }
  pending = {tuple(job[:8]): job[8] for job in jobs}
  with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs) or 1)) as pool:
    futures = {
      pool.submit(_export_tict_job, key): pending[key]
      for key in pending
    }
    for future in as_completed(futures):
      record = futures[future]
      record["result"] = future.result()
      record["status"] = "converted"
      records.append(record)
      _write_json_atomic(
        output / "conversion_manifest.json",
        {
          "schema": "kaihand_unified_egotouch_batch_v2",
          "task": dataset.task,
          "cameras": dataset.cameras,
          "source_root": str(dataset.root),
          "source_episodes": [
            source.relative_to(dataset.root).as_posix()
            for source in dataset.episodes
          ],
          "complete": False,
          "records": records,
        },
      )
  manifest = {
    "schema": "kaihand_unified_egotouch_batch_v2",
    "task": dataset.task,
    "cameras": dataset.cameras,
    "source_root": str(dataset.root),
    "source_episodes": [
      source.relative_to(dataset.root).as_posix() for source in dataset.episodes
    ],
    "complete": len(records) == len(dataset.episodes) * len(dataset.cameras),
    "records": sorted(records, key=lambda row: (row["session_id"], row["camera"])),
  }
  _write_json_atomic(output / "conversion_manifest.json", manifest)
  return manifest


def main(argv=None):
  args = parse_args(argv)
  if args.list_support:
    print(json.dumps({"formats": FORMATS, "adapters": adapter_catalog()}, indent=2))
    return 0
  runtime_profile = load_runtime_profile(args.runtime_config, args.runtime)
  probe = inspect_raw_dataset(args.input_dir, task=args.task, cameras=("head",))
  cameras = args.cameras or default_cameras_for(probe.task, args.format)
  dataset = (
    probe if tuple(cameras) == ("head",) else inspect_raw_dataset(
      args.input_dir, task=probe.task, cameras=cameras,
    )
  )
  if args.expected_episodes and len(dataset.episodes) != args.expected_episodes:
    raise ValueError(
      f"expected {args.expected_episodes} selected Raw episodes, "
      f"found {len(dataset.episodes)}"
    )
  contract = adapter_for(dataset, args.format)
  _validate_runtime_profile(
    contract, runtime_profile, execute=not args.dry_run
  )
  if args.format == "lerobot-v3":
    configured_python = (
      runtime_profile.python if runtime_profile is not None else None
    )
    _selected_path(
      args.lerobot_v3_python,
      configured_python,
      explicit_option="--lerobot-v3-python",
    )
    if (
      args.lerobot_v3_python is None
      and configured_python is None
      and importlib.util.find_spec("lerobot") is None
    ):
      raise ValueError(
        "this Python lacks LeRobot; pass --lerobot-v3-python PATH to the "
        "LeRobot v3 environment, or select a named runtime"
      )
    if args.lerobot_v3_python is not None and not args.lerobot_v3_python.is_file():
      raise FileNotFoundError(f"LeRobot v3 Python not found: {args.lerobot_v3_python}")
  elif args.lerobot_v3_python is not None:
    raise ValueError("--lerobot-v3-python is only supported for lerobot-v3")
  if args.format != "lerobot-v3" and (args.work_dir or args.repo_id):
    raise ValueError("--work-dir and --repo-id are only supported for lerobot-v3")
  if args.format == "lerobot-v3" and (args.instruction or args.dataset_name):
    raise ValueError("lerobot-v3 uses fixed task instructions; --instruction and --dataset-name are unsupported")
  if args.resume and not contract.resumable:
    raise ValueError(
      f"{args.format} adapter publishes atomically but does not yet support resume"
    )
  source_identity, raw_contract = _source_snapshot(dataset)
  if args.format == "egotouch":
    plan = _conversion_plan(
      args,
      dataset,
      contract,
      runtime_profile,
      None,
      source_identity,
      raw_contract,
    )
    with _in_process_environment(runtime_profile):
      result = _convert_egotouch(args, dataset)
    if args.dry_run:
      result = {**plan, **result}
    else:
      if result.get("complete") is not True:
        raise RuntimeError("EgoTouch backend did not publish a complete manifest")
      _publish_pipeline_manifest(args, dataset, plan, source_identity)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
  command = _command(args, dataset, contract, runtime_profile)
  plan = _conversion_plan(
    args,
    dataset,
    contract,
    runtime_profile,
    command,
    source_identity,
    raw_contract,
  )
  print(json.dumps(plan, indent=2, ensure_ascii=False))
  if args.dry_run:
    return 0
  if args.output_dir.exists() and not (args.resume and args.format == "lerobot-v3"):
    raise FileExistsError(f"output must not already exist: {args.output_dir}")
  run_options: dict[str, Any] = {"check": False}
  if _uses_external_environment(command, runtime_profile):
    run_options["env"] = _subprocess_environment(
      args, contract, runtime_profile
    )
  completed = subprocess.run(command, **run_options)
  if completed.returncode != 0:
    return completed.returncode
  _publish_pipeline_manifest(args, dataset, plan, source_identity)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
