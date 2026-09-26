"""The LingBot PickPlace batch must not change existing model evaluation paths."""

from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/workcell/evaluate_pickplace_lingbot_vla2_batch.py"


def _module(monkeypatch):
  monkeypatch.syspath_prepend(str(ROOT / "src"))
  monkeypatch.syspath_prepend(str(ROOT / "scripts/workcell"))
  return run_path(str(SCRIPT))


def _manifest(tmp_path, monkeypatch, **overrides):
  _module(monkeypatch)
  from lingbot_pickplace_contract import (
    ACTION_DIM,
    ACTION_REPRESENTATION,
    CONTROL_HZ,
    DEPLOYMENT_SCHEMA,
    HORIZON,
    INSTRUCTION,
    MODEL_ACTION_DIM,
    MODEL_FAMILY,
    OBSERVATION_CONTRACT,
    RIGHT_JOINT_NAMES,
  )

  payload = {
    "schema": DEPLOYMENT_SCHEMA,
    "task": "pick-place",
    "model_family": MODEL_FAMILY,
    "deployment_id": "lingbot-test",
    "checkpoint_path": "/nas/benchmark/LingBot/checkpoints/global_step_3905/hf_ckpt",
    "checkpoint_sha256": "a" * 64,
    "prediction_horizon": HORIZON,
    "model_action_dim": MODEL_ACTION_DIM,
    "action_dim": ACTION_DIM,
    "control_hz": CONTROL_HZ,
    "joint_names": list(RIGHT_JOINT_NAMES),
    "observation_contract": OBSERVATION_CONTRACT,
    "action_representation": ACTION_REPRESENTATION,
    "reference_dataset": "../training_reference",
    "reference_episode_index": 0,
    "instruction": INSTRUCTION,
  }
  payload.update(overrides)
  path = tmp_path / "deployment" / "deployment_manifest.json"
  path.parent.mkdir()
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path, payload


def _write_comparison(review: Path, reference: Path) -> None:
  review.mkdir()
  np.savez_compressed(
    review / "evaluation_rollout_trace.npz",
    simulation_time_s=np.arange(3, dtype=np.float64) / 30.0,
    state_29=np.zeros((3, 29)),
    fingertip_normal_force_n=np.zeros((3, 5)),
    fingertip_tangent_force_n=np.zeros((3, 5)),
  )
  (review / "evaluation_comparison.json").write_text(json.dumps({
    "status": "ok",
    "reference": {
      "dataset_root": str(reference), "output_episode_index": 0,
    },
  }), encoding="utf-8")
  for name in (
    "evaluation_right_wrist_state.png",
    "evaluation_right_hand_actuated_dof.png",
    "evaluation_right_fingertip_tactile.png",
  ):
    (review / name).write_bytes(b"plot")


def test_lingbot_batch_dry_run_uses_horizon_reference_and_model_views(
  tmp_path, monkeypatch, capsys
):
  path, _ = _manifest(tmp_path, monkeypatch)
  module = _module(monkeypatch)
  output = tmp_path / "evaluations"
  assert module["main"]([
    "--server", "127.0.0.1:18783",
    "--deployment-manifest", str(path),
    "--output-dir", str(output),
    "--seeds", "0", "1",
    "--video-count", "1",
    "--dry-run",
  ]) == 0
  assert not output.exists()
  result = json.loads(capsys.readouterr().out)
  protocol = result["protocol"]
  assert protocol["model_family"] == "LingBot-VLA-2.0"
  assert protocol["prediction_horizon"] == 50
  assert protocol["model_action_dim"] == 55
  assert protocol["action_dim"] == 27
  assert protocol["execute_steps"] == 50
  assert protocol["control_hz"] == 30
  assert protocol["max_sim_seconds"] == 90.0
  assert protocol["reference_dataset"] == str((tmp_path / "training_reference").resolve())
  assert protocol["review"]["model_input_cameras"] == ["head", "right_wrist"]
  assert protocol["review"]["time_series_displayed"] is False
  assert result["commands"][0][-5:] == [
    "--record", "--reference-dataset", protocol["reference_dataset"],
    "--reference-episode-index", "0",
  ]
  assert "--no-record" in result["commands"][1]


def test_lingbot_batch_cannot_override_frozen_training_reference(
  tmp_path, monkeypatch
):
  path, _ = _manifest(tmp_path, monkeypatch)
  module = _module(monkeypatch)
  with pytest.raises(ValueError, match="must match the frozen training dataset"):
    module["main"]([
      "--server", "127.0.0.1:18783",
      "--deployment-manifest", str(path),
      "--output-dir", str(tmp_path / "evaluations"),
      "--seeds", "0",
      "--reference-dataset", str(tmp_path / "other_dataset"),
      "--dry-run",
    ])


def test_lingbot_batch_environment_respects_osmesa_without_egl_vendor(
  monkeypatch
):
  module = _module(monkeypatch)
  environment = module["_environment"]
  monkeypatch.setenv("MUJOCO_GL", "osmesa")
  monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/not/an/egl/vendor.json")
  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "7")
  software = environment()
  assert software["MUJOCO_GL"] == "osmesa"
  assert software["PYOPENGL_PLATFORM"] == "osmesa"
  assert software["KAIHAND_RENDER_BACKEND"] == "software"
  assert software["LIBGL_ALWAYS_SOFTWARE"] == "1"
  assert software["GALLIUM_DRIVER"] == "llvmpipe"
  assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in software
  assert "MUJOCO_EGL_DEVICE_ID" not in software
  monkeypatch.setenv("MUJOCO_GL", "egl")
  hardware = environment()
  assert hardware["MUJOCO_GL"] == "egl"
  assert hardware["PYOPENGL_PLATFORM"] == "egl"
  assert hardware["KAIHAND_RENDER_BACKEND"] == "hardware"
  assert hardware["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(module["EGL_VENDOR"])
  assert "LIBGL_ALWAYS_SOFTWARE" not in hardware


def test_lingbot_batch_fingerprints_runner_reference_and_server(monkeypatch):
  module = _module(monkeypatch)
  hashes = module["fingerprints"]()
  assert {
    "scripts/workcell/run_pickplace_lingbot_vla2_policy.py",
    "scripts/workcell/serve_pickplace_lingbot_vla2_policy.py",
    "scripts/workcell/prepare_pickplace_lingbot_vla2_deployment.py",
    "scripts/workcell/lingbot_pickplace_contract.py",
    "scripts/workcell/lingbot_pickplace_reference.py",
    "src/kaihand_tactile_env/shared/contact_tactile.py",
    "scene",
  } <= hashes.keys()
  assert all(len(digest) == 64 for digest in hashes.values())


@pytest.mark.parametrize(
  ("override", "expected"),
  [
    ({"model_family": "pi0.5"}, "model_family"),
    ({"model_action_dim": 27}, "model_action_dim"),
    ({"action_dim": 55}, "action_dim"),
    ({"reference_dataset": ""}, "reference_dataset"),
    ({"joint_names": ["bad"]}, "joint_names"),
    ({"observation_contract": {"cameras": ["head"]}}, "observation_contract"),
  ],
)
def test_lingbot_batch_rejects_nonmatching_deployment(
  tmp_path, monkeypatch, override, expected
):
  path, _ = _manifest(tmp_path, monkeypatch, **override)
  module = _module(monkeypatch)
  with pytest.raises(ValueError, match=expected):
    module["load_manifest"](path)


def test_lingbot_batch_aggregates_trials_without_recording(
  tmp_path, monkeypatch
):
  path, payload = _manifest(tmp_path, monkeypatch)
  module = _module(monkeypatch)
  runner = module["main"]
  namespace = runner.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda: {"source": "frozen"})
  monkeypatch.setitem(namespace, "_environment", lambda: {"PYTHONPATH": "test"})
  commands = []

  def fake_run(command, **_kwargs):
    commands.append(command)
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    seed = int(command[command.index("--seed") + 1])
    success = seed == 0
    (trial / "summary.json").write_text(json.dumps({
      "task": "pick-place",
      "checkpoint_path": payload["checkpoint_path"],
      "checkpoint_sha256": payload["checkpoint_sha256"],
      "deployment_id": payload["deployment_id"],
      "model_family": "LingBot-VLA-2.0",
      "prediction_horizon": 50,
      "model_action_dim": 55,
      "action_dim": 27,
      "execute_steps": 50,
      "control_hz": 30,
      "seed": seed,
      "observation_contract": payload["observation_contract"],
      "action_representation": payload["action_representation"],
      "reference_dataset": str((tmp_path / "training_reference").resolve()),
      "reference_episode_index": 0,
      "diagnostic_only": False,
      "formal_metrics_valid": True,
      "status": "success" if success else "task_not_completed",
      "evaluation": {"success": success},
      "stats": {"requests": 2, "action_steps": 50},
    }), encoding="utf-8")
    _write_comparison(trial / "review", (tmp_path / "training_reference").resolve())
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  output = tmp_path / "evaluations"
  assert runner([
    "--server", "127.0.0.1:18783",
    "--deployment-manifest", str(path),
    "--output-dir", str(output),
    "--seeds", "0", "1",
    "--video-count", "0",
  ]) == 0
  assert len(commands) == 2
  assert all("--no-record" in command for command in commands)
  assert all(not (output / f"seed_{seed:03d}" / "review/review.mp4").exists() for seed in (0, 1))
  assert all((output / f"seed_{seed:03d}" / "review/evaluation_rollout_trace.npz").is_file() for seed in (0, 1))
  protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
  summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  assert protocol["source_hashes"] == {"source": "frozen"}
  assert protocol["reference_dataset"] == str((tmp_path / "training_reference").resolve())
  assert summary["planned"] == 2
  assert summary["valid_trials"] == 2
  assert summary["successes"] == 1
  assert summary["success_rate"] == 0.5
  assert summary["complete"] is True


def test_lingbot_review_requires_three_plots_when_reference_loaded(
  tmp_path, monkeypatch
):
  module = _module(monkeypatch)
  review = tmp_path / "review"
  reference = tmp_path / "training_reference"
  _write_comparison(review, reference)
  (review / "evaluation_right_fingertip_tactile.png").unlink()
  with pytest.raises(RuntimeError, match="comparison files missing"):
    module["_validate_comparison"](review, reference)
  (review / "evaluation_right_fingertip_tactile.png").write_bytes(b"plot")
  module["_validate_comparison"](review, reference)
  np.savez_compressed(
    review / "evaluation_rollout_trace.npz",
    simulation_time_s=np.arange(3, dtype=np.float64) / 10.0,
    state_29=np.zeros((3, 29)),
    fingertip_normal_force_n=np.zeros((3, 5)),
    fingertip_tangent_force_n=np.zeros((3, 5)),
  )
  with pytest.raises(RuntimeError, match="control-step cadence"):
    module["_validate_comparison"](review, reference)
  np.savez_compressed(
    review / "evaluation_rollout_trace.npz",
    simulation_time_s=np.arange(3, dtype=np.float64) / 30.0,
    state_29=np.zeros((3, 29)),
    fingertip_normal_force_n=np.zeros((3, 5)),
    fingertip_tangent_force_n=np.zeros((3, 5)),
  )
  with pytest.raises(RuntimeError, match="video files missing"):
    module["_validate_video"](review)
  for name in module["VIDEO_FILES"]:
    (review / name).write_bytes(b"video")
  (review / "review.json").write_text(json.dumps({
    "completed": True,
    "second_camera": "global",
    "model_input_cameras_displayed": ["head", "right_wrist"],
    "time_series_displayed": False,
    "comparison_requested": True,
    "comparison_plots": {"status": "ok"},
  }), encoding="utf-8")
  module["_validate_video"](review)
  (review / "evaluation_comparison.json").write_text(
    json.dumps({"status": "reference_unavailable", "reason": "missing raw episode"}),
    encoding="utf-8",
  )
  with pytest.raises(RuntimeError, match="comparison against its training episode 00"):
    module["_validate_comparison"](review, reference)


def test_batch_surfaces_server_failure_before_checking_trace(tmp_path, monkeypatch):
  path, _ = _manifest(tmp_path, monkeypatch)
  module = _module(monkeypatch)
  runner = module["main"]
  namespace = runner.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda: {"source": "frozen"})
  monkeypatch.setitem(namespace, "_environment", lambda: {"PYTHONPATH": "test"})

  def fake_run(command, **_kwargs):
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    (trial / "summary.json").write_text(json.dumps({
      "status": "error", "error": "first inference: chat template is missing",
    }), encoding="utf-8")
    return SimpleNamespace(returncode=1)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  with pytest.raises(RuntimeError, match="first inference: chat template is missing"):
    runner([
      "--server", "127.0.0.1:18783",
      "--deployment-manifest", str(path),
      "--output-dir", str(tmp_path / "evaluation"),
      "--seeds", "0",
    ])
