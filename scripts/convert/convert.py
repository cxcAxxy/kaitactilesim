#!/usr/bin/env python3
"""Unified Raw -> EgoSteer / pi0.5 / EgoTouch conversion entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kaihand_tactile_env.pipeline.conversion import (
  FORMATS,
  TASKS,
  adapter_catalog,
  adapter_for,
  inspect_raw_dataset,
)
from kaihand_tactile_env.shared.cameras import TRAINING_CAMERA_NAMES

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PI05_BACKENDS = {
  "convert_pickplace_unified_to_lerobot.py",
  "convert_poker_unified_to_lerobot.py",
  "convert_shared_to_lerobot.py",
}


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
    "--cameras", nargs="+", default=("head",),
    choices=TRAINING_CAMERA_NAMES,
  )
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument(
    "--expected-episodes", type=int, default=0,
    help="fail unless this many selected Raw episodes are found; 0 accepts any count",
  )
  parser.add_argument("--staging-root", type=Path)
  parser.add_argument("--verify-source-hash", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--instruction")
  parser.add_argument("--dataset-name")
  parser.add_argument(
    "--openpi-root", type=Path, default=ROOT.parent / "openpi",
  )
  parser.add_argument("--pi05-python", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args(argv)
  if args.workers <= 0:
    parser.error("--workers must be positive")
  if args.expected_episodes < 0:
    parser.error("--expected-episodes must be nonnegative")
  if not args.list_support:
    for name in ("input_dir", "output_dir", "format"):
      if getattr(args, name) is None:
        parser.error(f"--{name.replace('_', '-')} is required")
  return args


def _command(args, dataset, contract):
  if contract.output_format == "egosteer":
    cmd = [
      sys.executable,
      str(ROOT / "scripts/workcell" / contract.backend),
      "--input-dir", str(dataset.root),
      "--output-dir", str(args.output_dir),
    ]
    if dataset.task in (
      "poker-draw", "usb-insert", "bulb-screw", "install-ram", "whiteboard-wipe"
    ):
      cmd += [
        "--workers", str(args.workers),
        "--expected-episodes", str(len(dataset.episodes)),
        "--cameras", *dataset.cameras,
      ]
      if args.staging_root:
        cmd += ["--staging-root", str(args.staging_root)]
      if args.verify_source_hash:
        cmd.append("--verify-source-hash")
    if contract.backend == "convert_card_to_egosteer.py":
      cmd += ["--task", dataset.task]
    if args.instruction:
      cmd += ["--instruction", args.instruction]
    if args.dataset_name:
      cmd += ["--dataset-name", args.dataset_name]
    return cmd
  openpi_root = args.openpi_root.expanduser().resolve()
  python = args.pi05_python or openpi_root / ".venv-pi05/bin/python"
  backend_path = (
    ROOT / "scripts/workcell" / contract.backend
    if contract.backend in LOCAL_PI05_BACKENDS
    else openpi_root / "examples/kaihand" / contract.backend
  )
  cmd = [
    str(python),
    str(backend_path),
    "--input-dir", str(dataset.root),
    "--output-dir", str(args.output_dir),
    "--expected-episodes", str(len(dataset.episodes)),
    "--fingerprint-workers", str(args.workers),
  ]
  if contract.backend not in LOCAL_PI05_BACKENDS:
    cmd += [
      "--image-writer-processes", "0",
      "--image-writer-threads", str(args.workers),
    ]
  if contract.backend in LOCAL_PI05_BACKENDS:
    cmd += ["--openpi-root", str(openpi_root)]
  if args.staging_root:
    cmd += ["--staging-root", str(args.staging_root)]
  if args.verify_source_hash:
    cmd.append("--verify-source-hash")
  if contract.backend in (
    "convert_card_to_lerobot.py",
    "convert_poker_unified_to_lerobot.py",
    "convert_shared_to_lerobot.py",
  ):
    cmd += ["--task", dataset.task, "--cameras", *dataset.cameras]
    if args.instruction:
      cmd += ["--instruction", args.instruction]
    if args.dataset_name:
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
    return
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
  ):
    raise ValueError(
      "--resume output was created for a different task/camera contract: "
      f"{path}"
    )


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
      ) = _source_identity(source, verify_source_hash=args.verify_source_hash)
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
          "complete": False,
          "records": records,
        },
      )
  manifest = {
    "schema": "kaihand_unified_egotouch_batch_v2",
    "task": dataset.task,
    "cameras": dataset.cameras,
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
  dataset = inspect_raw_dataset(
    args.input_dir, task=args.task, cameras=args.cameras
  )
  if args.expected_episodes and len(dataset.episodes) != args.expected_episodes:
    raise ValueError(
      f"expected {args.expected_episodes} selected Raw episodes, "
      f"found {len(dataset.episodes)}"
    )
  contract = adapter_for(dataset, args.format)
  if args.resume and not contract.resumable:
    raise ValueError(
      f"{args.format} adapter publishes atomically but does not yet support resume"
    )
  if args.format == "egotouch":
    result = _convert_egotouch(args, dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
  command = _command(args, dataset, contract)
  plan = {
    "task": dataset.task,
    "format": args.format,
    "adapter": contract.backend,
    "episodes": len(dataset.episodes),
    "cameras": dataset.cameras,
    "command": command,
  }
  print(json.dumps(plan, indent=2, ensure_ascii=False))
  if args.dry_run:
    return 0
  if args.output_dir.exists():
    raise FileExistsError(f"output must not already exist: {args.output_dir}")
  return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
  raise SystemExit(main())
