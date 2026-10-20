"""Batch validation accepts flat outputs without migrating historical reviews."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKCELL = ROOT / "scripts/workcell"


@pytest.fixture
def batch_modules(monkeypatch):
  monkeypatch.syspath_prepend(str(ROOT / "src"))
  monkeypatch.syspath_prepend(str(WORKCELL))
  return {
    "shared": run_path(str(WORKCELL / "evaluate_shared_task_policy_batch.py")),
    "pickplace": run_path(str(WORKCELL / "evaluate_pickplace_policy_batch.py")),
  }


def _write_artifacts(trial, layout, names, metadata):
  trial.mkdir()
  (trial / "summary.json").write_text("original summary", encoding="utf-8")
  artifacts = trial if layout == "flat" else trial / "review"
  artifacts.mkdir(exist_ok=True)
  for name in names:
    (artifacts / name).write_bytes(b"artifact")
  (artifacts / "review.json").write_text(
    json.dumps(metadata), encoding="utf-8"
  )
  return artifacts


def _snapshot(trial):
  return {
    str(path.relative_to(trial)): path.read_bytes()
    for path in trial.rglob("*")
    if path.is_file()
  }


@pytest.mark.parametrize("layout", ("flat", "legacy"))
@pytest.mark.parametrize("model_cameras", (["head"], ["head", "right_wrist"]))
@pytest.mark.parametrize("diagnostic_only", (False, True))
@pytest.mark.parametrize("with_reference", (False, True))
def test_shared_batch_validates_both_layouts_without_mutation(
  tmp_path, batch_modules, layout, model_cameras, diagnostic_only, with_reference
):
  module = batch_modules["shared"]
  trial = tmp_path / "seed_001"
  reference = (tmp_path / "reference").resolve() if with_reference else None
  metadata = {
    "model_input_cameras_displayed": model_cameras,
    "diagnostic_only": diagnostic_only,
    "formal_metrics_valid": not diagnostic_only,
    "reference_dataset": str(reference),
    "reference_episode_index": 2,
    "comparison_plots": {"status": "ok"},
  }
  names = module["_REVIEW_ARTIFACTS"]
  if with_reference:
    names += module["_COMPARISON_ARTIFACTS"]
  artifacts = _write_artifacts(trial, layout, names, metadata)
  before = _snapshot(trial)

  assert module["validate_recorded_review"](
    trial,
    {"observation_contract": {"cameras": model_cameras}},
    reference,
    2,
    diagnostic_only=diagnostic_only,
  ) is None

  assert _snapshot(trial) == before
  assert (trial / "review").is_dir() is (layout == "legacy")
  assert (artifacts / "review.json").is_file()


@pytest.mark.parametrize("layout", ("flat", "legacy"))
@pytest.mark.parametrize("model_cameras", (["head"], ["head", "right_wrist"]))
@pytest.mark.parametrize("require_comparison", (False, True))
def test_pickplace_batch_validates_both_layouts_without_mutation(
  tmp_path, batch_modules, layout, model_cameras, require_comparison
):
  module = batch_modules["pickplace"]
  trial = tmp_path / "seed_001"
  metadata = {
    "completed": True,
    "second_camera": "global",
    "model_input_cameras_displayed": model_cameras,
    "comparison_plots": {"status": "ok"},
  }
  names = module["REVIEW_FILES"]
  if require_comparison:
    names += module["COMPARISON_FILES"]
  artifacts = _write_artifacts(trial, layout, names, metadata)
  if require_comparison:
    (artifacts / "evaluation_comparison.json").write_text(
      json.dumps({"status": "ok"}), encoding="utf-8"
    )
  before = _snapshot(trial)

  assert module["_validate_review_artifacts"](
    trial,
    1,
    {"observation_contract": {"cameras": model_cameras}},
    require_comparison=require_comparison,
  ) == metadata

  assert _snapshot(trial) == before
  assert (trial / "review").is_dir() is (layout == "legacy")


@pytest.mark.parametrize("layout", ("flat", "legacy"))
def test_missing_video_is_rejected_in_both_layouts(tmp_path, batch_modules, layout):
  trial = tmp_path / "seed_001"
  _write_artifacts(trial, layout, ("frames.jsonl",), {})
  before = _snapshot(trial)
  manifest = {"observation_contract": {"cameras": ["head"]}}

  assert batch_modules["shared"]["validate_recorded_review"](
    trial, manifest, None, 0
  ) == "incomplete_review_artifact"
  with pytest.raises(RuntimeError, match="review missing or empty"):
    batch_modules["pickplace"]["_validate_review_artifacts"](
      trial, 1, manifest, require_comparison=False
    )
  assert _snapshot(trial) == before


def test_partial_flat_output_does_not_fall_back_to_valid_legacy(
  tmp_path, batch_modules
):
  trial = tmp_path / "seed_001"
  manifest = {"observation_contract": {"cameras": ["head"]}}
  _write_artifacts(
    trial,
    "legacy",
    batch_modules["pickplace"]["REVIEW_FILES"],
    {
      "completed": True,
      "second_camera": "global",
      "model_input_cameras_displayed": ["head"],
      "diagnostic_only": False,
      "formal_metrics_valid": True,
    },
  )
  assert batch_modules["shared"]["validate_recorded_review"](
    trial, manifest, None, 0
  ) is None
  assert batch_modules["pickplace"]["_validate_review_artifacts"](
    trial, 1, manifest, require_comparison=False
  )["completed"] is True

  (trial / "frames.jsonl").write_text("partial new run", encoding="utf-8")
  before = _snapshot(trial)
  assert batch_modules["shared"]["validate_recorded_review"](
    trial, manifest, None, 0
  ) == "incomplete_review_artifact"
  with pytest.raises(RuntimeError, match="review missing or empty"):
    batch_modules["pickplace"]["_validate_review_artifacts"](
      trial, 1, manifest, require_comparison=False
    )
  assert _snapshot(trial) == before


def test_evaluation_runners_record_into_trial_root():
  """Check routing without importing optional model clients or running physics."""
  checked = []
  for path in sorted(WORKCELL.glob("run*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
      if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"EvaluationVideo", "PokerPolicyVideo"}
      ):
        assert len(node.args) >= 2, str(path)
        output = node.args[1]
        assert isinstance(output, ast.Name) and output.id == "output", str(path)
        checked.append(path.name)
  assert checked, "no evaluation recorder constructors were checked"
