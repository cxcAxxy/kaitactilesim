from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from kaihand_tactile_env.shared.evaluation_source_integrity import (
  canonical_hash,
  verify_evaluation_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def _deployment(root: Path, relative_paths: tuple[str, ...]) -> dict:
  rows = []
  for relative in relative_paths:
    path = root / relative
    rows.append(
      {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      }
    )
  return {
    "evaluation_source_files": rows,
    "evaluation_source_sha256": canonical_hash(rows),
  }


def _load_script(name: str):
  path = ROOT / "scripts/workcell" / name
  spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_evaluation_source_integrity_detects_same_size_content_drift(tmp_path):
  source = tmp_path / "source.py"
  source.write_text("one", encoding="utf-8")
  deployment = _deployment(tmp_path, ("source.py",))

  verified = verify_evaluation_sources(deployment, tmp_path)
  assert verified["status"] == "verified"
  assert verified["actual_sha256"] == deployment["evaluation_source_sha256"]

  source.write_text("two", encoding="utf-8")
  drifted = verify_evaluation_sources(deployment, tmp_path)
  assert drifted["status"] == "mismatch"
  assert drifted["changed_files"][0]["path"] == "source.py"
  assert drifted["changed_files"][0]["issues"] == ["sha256"]


def test_evaluation_source_integrity_rejects_unverifiable_legacy_manifest(tmp_path):
  report = verify_evaluation_sources({}, tmp_path)

  assert report["status"] == "unverifiable"
  assert "lacks evaluation_source_files" in report["errors"][0]


def test_evaluation_source_integrity_authenticates_rows_and_safe_paths(tmp_path):
  source = tmp_path / "source.py"
  source.write_text("payload", encoding="utf-8")
  deployment = _deployment(tmp_path, ("source.py",))
  deployment["evaluation_source_files"][0]["size"] += 1

  report = verify_evaluation_sources(deployment, tmp_path)
  assert report["status"] == "mismatch"
  assert "does not authenticate" in report["errors"][0]

  unsafe_rows = [{"path": "../outside.py", "size": 0, "sha256": "0" * 64}]
  unsafe = verify_evaluation_sources(
    {
      "evaluation_source_files": unsafe_rows,
      "evaluation_source_sha256": canonical_hash(unsafe_rows),
    },
    tmp_path,
  )
  assert unsafe["status"] == "mismatch"
  assert "unsafe noncanonical path" in unsafe["errors"][0]


def test_prepare_freezes_new_evaluation_runtime_dependencies():
  prepare = _load_script("prepare_shared_task_pi05_deployment.py")
  paths = {
    path.relative_to(ROOT).as_posix()
    for path in prepare.evaluation_source_paths("bulb-screw")
  }

  assert {
    "src/kaihand_tactile_env/shared/action_diagnostics.py",
    "src/kaihand_tactile_env/shared/config.py",
    "src/kaihand_tactile_env/shared/evaluation_video.py",
    "src/kaihand_tactile_env/shared/evaluation_source_integrity.py",
    "src/kaihand_tactile_env/shared/pickup_randomization.py",
    "src/kaihand_tactile_env/shared/policy_cameras.py",
    "src/kaihand_tactile_env/shared/rendering.py",
    "src/kaihand_tactile_env/shared/simulation.py",
    "src/kaihand_tactile_env/shared/mjcf/robot.xml",
    "src/kaihand_tactile_env/assets/workcell/mjcf/hand_asset.xml",
    "src/kaihand_tactile_env/assets/workcell/textures/dark_grey_marble_v1.png",
    "src/kaihand_tactile_env/tactile/layout.py",
    "scripts/workcell/evaluate_shared_task_policy_batch.py",
    "scripts/workcell/prepare_shared_task_pi05_deployment.py",
    "scripts/workcell/run_shared_task_pi05_policy.py",
    "scripts/workcell/run_usb_pi05_policy.py",
    "scripts/workcell/serve_shared_task_pi05_policy.py",
  } <= paths


def test_batch_summary_marks_blocked_seed_unattempted(tmp_path):
  batch = _load_script("evaluate_shared_task_policy_batch.py")
  integrity = {"status": "verified"}
  drift = {"status": "mismatch"}
  results = [
    {
      "trial_started": False,
      "valid_trial": False,
      "success": False,
      "termination": "evaluation_source_drift_before_trial",
      "action_clipping": None,
      "evaluation_source_integrity_before": drift,
      "evaluation_source_integrity_after": None,
    }
  ]

  summary = batch._batch_summary({}, 20, results, 0.0, integrity)

  assert summary["attempted"] == 0
  assert summary["valid_trials"] == 0
  assert summary["evaluation_source_integrity"]["final"] == drift
