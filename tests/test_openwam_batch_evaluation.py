from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def batch():
  path = ROOT / "scripts/workcell/evaluate_usb_openwam_batch.py"
  spec = importlib.util.spec_from_file_location(
    "evaluate_usb_openwam_batch_test", path
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _make_reference_dataset(root: Path) -> Path:
  metadata = root / "meta"
  metadata.mkdir(parents=True)
  (metadata / "info.json").write_text("{}\n", encoding="utf-8")
  (metadata / "kaihand_source_episodes.jsonl").write_text(
    '{"episode_index": 0}\n', encoding="utf-8"
  )
  return root


def _trial_args(tmp_path: Path, reference_dataset: Path | None = None):
  return SimpleNamespace(
    kaitactilesim_python=Path("/opt/kaitactilesim/bin/python"),
    port=8848,
    max_sim_seconds=50.0,
    response_timeout=600.0,
    trial_wall_limit=321.0,
    reference_dataset=reference_dataset,
    reference_episode_index=0,
  )


@pytest.mark.parametrize(
  "missing_name",
  ("info.json", "kaihand_source_episodes.jsonl"),
)
def test_explicit_reference_dataset_requires_all_metadata(
  batch, tmp_path: Path, missing_name: str
) -> None:
  dataset = _make_reference_dataset(tmp_path / "dataset")

  assert batch._validate_reference_dataset(dataset) == dataset.resolve()

  (dataset / "meta" / missing_name).unlink()
  with pytest.raises(ValueError, match="missing required metadata") as error:
    batch._validate_reference_dataset(dataset)
  assert missing_name in str(error.value)


def test_checkpoint_reference_dataset_is_validated(batch, tmp_path: Path) -> None:
  dataset = _make_reference_dataset(tmp_path / "training data")
  checkpoint = tmp_path / "checkpoint"
  checkpoint.mkdir()
  (checkpoint / "config.yaml").write_text(
    "model: openwam\n"
    "dataloader:\n"
    f'  dataset_dir: "{dataset}" # frozen training data\n'
    "  batch_size: 32\n",
    encoding="utf-8",
  )

  assert batch._checkpoint_reference_dataset(checkpoint) == dataset.resolve()

  (dataset / "meta/kaihand_source_episodes.jsonl").unlink()
  with pytest.raises(ValueError, match="kaihand_source_episodes.jsonl"):
    batch._checkpoint_reference_dataset(checkpoint)


def test_only_recorded_trials_receive_reference_arguments(
  batch, monkeypatch, tmp_path: Path
) -> None:
  dataset = _make_reference_dataset(tmp_path / "dataset")
  args = _trial_args(tmp_path, dataset)
  calls: list[list[str]] = []

  def fake_run(command, **kwargs):
    calls.append(command)
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == args.trial_wall_limit
    return SimpleNamespace(returncode=0, stdout="", stderr="")

  monkeypatch.setattr(batch.subprocess, "run", fake_run)

  recorded = batch._run_trial(args, tmp_path / "recorded", 3, 16, True)
  unrecorded = batch._run_trial(args, tmp_path / "unrecorded", 4, 32, False)

  assert recorded["status"] == "missing_summary"
  assert unrecorded["status"] == "missing_summary"
  recorded_command, unrecorded_command = calls
  assert recorded_command[
    recorded_command.index("--reference-dataset") + 1
  ] == str(dataset)
  assert recorded_command[
    recorded_command.index("--reference-episode-index") + 1
  ] == "0"
  assert "--no-record" not in recorded_command
  assert "--reference-dataset" not in unrecorded_command
  assert "--reference-episode-index" not in unrecorded_command
  assert "--no-record" in unrecorded_command


def test_nonzero_runner_exit_preserves_existing_summary(
  batch, monkeypatch, tmp_path: Path
) -> None:
  args = _trial_args(tmp_path)
  group_dir = tmp_path / "group"
  trial_dir = group_dir / "seed_007"
  trial_dir.mkdir(parents=True)
  comparison_plots = {
    "wrist_state": "review/right_wrist_state.png",
    "hand_joints": "review/right_hand_joints.png",
    "fingertip_tactile": "review/right_fingertip_tactile.png",
  }
  payload = {
    "status": "artifact_error",
    "evaluation": {"success": False, "failure_stage": "insert"},
    "error": "RuntimeError: failed after writing summary",
    "artifact_errors": ["plot export failed"],
    "comparison_plots": comparison_plots,
    "video": {"path": "review/evaluation.mp4"},
  }
  (trial_dir / "summary.json").write_text(
    json.dumps(payload), encoding="utf-8"
  )

  monkeypatch.setattr(
    batch.subprocess,
    "run",
    lambda *args, **kwargs: SimpleNamespace(
      returncode=17,
      stdout="runner stdout",
      stderr="runner stderr",
    ),
  )

  trial = batch._run_trial(args, group_dir, 7, 16, False)

  assert trial["status"] == payload["status"]
  assert trial["error"] == payload["error"]
  assert trial["comparison_plots"] == comparison_plots
  assert trial["artifact_errors"] == payload["artifact_errors"]
  assert trial["returncode"] == 17
  assert trial["stdout_tail"] == "runner stdout"
  assert trial["stderr_tail"] == "runner stderr"


def test_nonzero_exit_cannot_count_as_success(batch, monkeypatch, tmp_path: Path) -> None:
  args = _trial_args(tmp_path)
  group_dir = tmp_path / "group"
  trial_dir = group_dir / "seed_007"
  trial_dir.mkdir(parents=True)
  (trial_dir / "summary.json").write_text(json.dumps({
    "status": "success", "evaluation": {"success": True},
  }))
  monkeypatch.setattr(
    batch.subprocess, "run",
    lambda *args, **kwargs: SimpleNamespace(returncode=17, stdout="", stderr=""),
  )
  trial = batch._run_trial(args, group_dir, 7, 16, False)
  assert trial["success"] is False
  assert batch._summarize_trials([trial])["valid_trials"] == 0


def test_trial_timeout_is_reported_without_reading_summary(
  batch, monkeypatch, tmp_path: Path
) -> None:
  args = _trial_args(tmp_path)

  def timeout(command, **kwargs):
    raise subprocess.TimeoutExpired(
      command,
      kwargs["timeout"],
      output="partial stdout",
      stderr="partial stderr",
    )

  monkeypatch.setattr(batch.subprocess, "run", timeout)

  trial = batch._run_trial(args, tmp_path / "group", 9, 16, False)

  assert trial == {
    "seed": 9,
    "status": "trial_timeout",
    "timeout_seconds": 321.0,
    "stdout_tail": "partial stdout",
    "stderr_tail": "partial stderr",
  }
