#!/usr/bin/env python3
"""Isolated JSONL CPM/FTP-1 model process. Stdout is protocol-only."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def pack(value):
  if isinstance(value, np.ndarray):
    if value.dtype.hasobject:
      raise ValueError("object arrays are forbidden")
    if value.dtype.kind not in "biuf":
      raise ValueError("unsupported wire dtype")
    return {"__ndarray__": base64.b64encode(value.tobytes()).decode(),
            "dtype": value.dtype.str, "shape": list(value.shape)}
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {key: pack(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [pack(item) for item in value]
  return value


def unpack(value):
  if isinstance(value, dict) and "__ndarray__" in value:
    dtype = np.dtype(value["dtype"])
    if dtype.hasobject or dtype.kind not in "biuf":
      raise ValueError("unsupported wire dtype")
    data = base64.b64decode(value["__ndarray__"], validate=True)
    return np.frombuffer(data, dtype=dtype).reshape(value["shape"]).copy()
  if isinstance(value, dict):
    return {key: unpack(item) for key, item in value.items()}
  if isinstance(value, list):
    return [unpack(item) for item in value]
  return value


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--deployment-manifest", type=Path, required=True)
  parser.add_argument("--validate-only", action="store_true")
  args = parser.parse_args(argv)
  protocol = sys.stdout
  sys.stdout = sys.stderr
  from kaihand_tactile_env.shared.tactile_deployment import load_manifest, sha256_file
  path, manifest = load_manifest(args.deployment_manifest, verify_files=True)
  if manifest["model_family"] == "pi05+cpm":
    from cpm_policy_backend import CPMPolicyBackend as Backend
  else:
    from ftp1_policy_backend import FTP1PolicyBackend as Backend
  backend = Backend(manifest)
  metadata = backend.metadata()
  for key in ("execute_steps", "checkpoint_path", "checkpoint_sha256"):
    metadata.setdefault(key, manifest[key])
  import importlib.metadata
  metadata["package_versions"] = {}
  for package in ("torch", "transformers", "numpy", "safetensors", "sentencepiece"):
    try:
      metadata["package_versions"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      metadata["package_versions"][package] = None
  expected_versions = manifest["runtime"].get("package_versions")
  if expected_versions is not None and metadata["package_versions"] != expected_versions:
    raise RuntimeError("model runtime package versions changed; prepare a new deployment")
  expected_python = manifest["runtime"].get("python_binary_sha256")
  if expected_python is not None and sha256_file(Path(sys.executable).resolve()) != expected_python:
    raise RuntimeError("model Python executable changed")
  # Preserve the model's advertised horizon/offset for strict parent validation.
  metadata.update(deployment_id=manifest["deployment_id"],
                  deployment_manifest_sha256=sha256_file(path))
  print(json.dumps({"ready": pack(metadata)}, allow_nan=False), file=protocol, flush=True)
  if args.validate_only:
    return
  for line in sys.stdin:
    request = unpack(json.loads(line))
    if request.get("close"):
      return
    response = backend.infer(request)
    print(json.dumps(pack(response), allow_nan=False), file=protocol, flush=True)


if __name__ == "__main__":
  main()
