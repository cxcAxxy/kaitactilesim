"""PickPlace EgoTouch deployment/batch integration without importing torch."""

from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts/workcell/prepare_pickplace_egotouch_deployment.py"
BATCH = ROOT / "scripts/workcell/evaluate_pickplace_policy_batch.py"


def _module(path: Path, monkeypatch):
  monkeypatch.syspath_prepend(str(ROOT / "src"))
  monkeypatch.syspath_prepend(str(ROOT / "scripts/workcell"))
  return run_path(str(path))


def _binding(module) -> dict:
  slots = module["VIEW_SLOTS"]
  mask = module["VIEW_MASK"]
  records = {}
  for index in range(50):
    records[f"{index:05d}"] = {
      "images": {
        name: {"valid": valid}
        for name, valid in zip(slots, mask, strict=True)
      }
    }
  return {
    "schema_version": "egotouch-usb-tict-binding-v1",
    "pred_horizon": 50,
    "img_name": "rgb.png",
    "view_names": slots,
    "view_image_names": {
      "head": "rgb.png",
      "left_wrist": "rgb_left_wrist.png",
      "right_wrist": "rgb_right_wrist.png",
    },
    "stats": {
      name: {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
      for name in ("pos", "finger_pos", "tactile")
    },
    "selector_records": {"session": records},
  }


def _checkpoint_metadata(module, binding: dict) -> dict:
  return {
    "schema_version": module["CHECKPOINT_SCHEMA"],
    "epoch": 50,
    "global_step": 15800,
    "variant": "no_tactile",
    "binding_sha256": module["canonical_hash"](binding),
    "model_weights": "ema",
    "model_tensor_count": 50,
    "action_proj_shape": [384, 108],
    "head_v_shape": [108, 384],
    "cfg": {
      "pred_horizon": 50,
      "use_tactile_input": False,
      "action_mode": "absolute",
      "frame_mode": "camera_frame",
      "centric_mode": "ego_centric",
      "single_hand": False,
      "image_size": [240, 320],
      "num_inference_steps": 10,
    },
  }


def _prepare(tmp_path: Path, monkeypatch, epoch: int = 50) -> tuple[Path, dict]:
  module = _module(PREPARER, monkeypatch)
  binding = _binding(module)
  checkpoint = tmp_path / f"epoch_{epoch:04d}.pt"
  checkpoint.write_bytes(b"checkpoint")
  binding_path = tmp_path / "egotouch_binding.json"
  binding_path.write_text(json.dumps(binding), encoding="utf-8")
  project = tmp_path / "egotouch"
  for relative in module["MODEL_SOURCE_PATHS"]:
    source = project / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"# {relative}\n", encoding="utf-8")
  model_python = tmp_path / "python"
  model_python.write_bytes(b"python-binary")
  metadata = _checkpoint_metadata(module, binding)
  metadata["epoch"] = epoch
  monkeypatch.setitem(
    module["main"].__globals__, "inspect_checkpoint",
    lambda *_args: metadata,
  )
  output = tmp_path / "deployment"
  assert module["main"]([
    "--checkpoint", str(checkpoint),
    "--binding", str(binding_path),
    "--model-project", str(project),
    "--model-python", str(model_python),
    "--output-dir", str(output),
  ]) == 0
  manifest_path = output / "deployment_manifest.json"
  return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("epoch", [25, 50, 75])
def test_preparer_freezes_binding_views_epoch_and_all_runtime_hashes(
  tmp_path, monkeypatch, epoch
):
  manifest_path, manifest = _prepare(tmp_path, monkeypatch, epoch)
  assert manifest_path.is_file()
  assert manifest["schema"] == "pickplace_egotouch_deployment_v1"
  assert manifest["model_family"] == "EgoTouch"
  assert manifest["checkpoint_epoch"] == epoch
  assert manifest["checkpoint_global_step"] == 15800
  assert manifest["prediction_horizon"] == 50
  assert manifest["action_dim"] == 108
  assert manifest["model_weights"] == "ema"
  assert manifest["observation_contract"] == {
    "cameras": ["head", "right_wrist"],
    "image_shape_hwc": [240, 320, 3],
    "view_slots": ["head", "left_wrist", "right_wrist"],
    "view_mask": [True, False, True],
    "tactile_sent_to_model": False,
  }
  for key in (
    "deployment_id", "checkpoint_sha256", "binding_sha256",
    "binding_file_sha256", "model_source_sha256", "model_python_sha256",
  ):
    assert len(manifest[key]) == 64


def test_preparer_fails_closed_on_binding_mask_or_digest(tmp_path, monkeypatch):
  module = _module(PREPARER, monkeypatch)
  binding = _binding(module)
  metadata = _checkpoint_metadata(module, binding)
  binding["selector_records"]["session"]["00000"]["images"]["left_wrist"]["valid"] = True
  metadata["binding_sha256"] = module["canonical_hash"](binding)
  with pytest.raises(ValueError, match="visual mask"):
    module["validate_binding"](binding, metadata)
  binding = _binding(module)
  metadata["binding_sha256"] = "0" * 64
  with pytest.raises(ValueError, match="checkpoint/binding SHA-256 mismatch"):
    module["validate_binding"](binding, metadata)


def test_batch_fingerprints_all_comparison_tactile_and_decode_dependencies(
  monkeypatch,
):
  module = _module(BATCH, monkeypatch)
  hashes = module["fingerprints"](module["RUNNERS"]["EgoTouch"])
  assert {
    "src/kaihand_tactile_env/shared/evaluation_comparison.py",
    "src/kaihand_tactile_env/shared/openwam_evaluation_plots.py",
    "src/kaihand_tactile_env/shared/contact_tactile.py",
    "scripts/workcell/run_shared_task_egotouch_policy.py",
  } <= hashes.keys()


@pytest.mark.parametrize("epoch", [25, 50, 75])
def test_batch_invokes_local_egotouch_runner_without_server(
  tmp_path, monkeypatch, epoch
):
  manifest_path, manifest = _prepare(tmp_path, monkeypatch, epoch)
  module = _module(BATCH, monkeypatch)
  main = module["main"]
  namespace = main.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda _runner: {"source": "frozen"})
  commands = []
  environments = []

  def fake_run(command, **kwargs):
    commands.append(command)
    environments.append(kwargs["env"])
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    seed = int(command[command.index("--seed") + 1])
    (trial / "summary.json").write_text(json.dumps({
      "task": "pick-place",
      "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": "EgoTouch",
      "prediction_horizon": 50,
      "action_dim": 108,
      "execute_steps": 5,
      "control_hz": 30,
      "seed": seed,
      "guard_mode": "observe",
      "guard_events": [],
      "stop_reason": "time_limit",
      "physics_valid": True,
      "diagnostic_only": True,
      "formal_metrics_valid": False,
      "task_success": False,
      "hand_ik_orientation_weights_m_per_rad": [0.02, 0.0, 0.0, 0.0, 0.0],
      "hold_failed_hand_ik": False,
      "status": "task_not_completed",
      "evaluation": {"success": False},
      "stats": {"requests": 1, "action_steps": 5},
    }), encoding="utf-8")
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  output = tmp_path / "evaluation"
  assert main([
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output),
    "--seeds", "0",
    "--execute-steps", "5",
    "--video-count", "0",
    "--model-device", "cpu",
    "--render-backend", "software-osmesa",
  ]) == 0
  assert len(commands) == 1
  command = commands[0]
  assert "--server" not in command
  for option, value in (
    ("--checkpoint", manifest["checkpoint_path"]),
    ("--binding", manifest["binding_path"]),
    ("--model-python", manifest["model_python"]),
    ("--model-project", manifest["model_project_path"]),
    ("--model-device", "cpu"),
  ):
    assert command[command.index(option) + 1] == value
  weight_index = command.index("--hand-ik-orientation-weights")
  assert command[weight_index + 1:weight_index + 6] == [
    "0.02", "0.0", "0.0", "0.0", "0.0"
  ]
  assert "--no-hold-failed-hand-ik" in command
  protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
  assert protocol["model_runtime"]["model_device"] == "cpu"
  assert protocol["observation_contract"]["view_mask"] == [True, False, True]
  assert protocol["render_backend"] == "software-osmesa"
  assert protocol["kinematic_reachability_guards_enabled"] is True
  assert protocol["hand_ik_orientation_weights_m_per_rad"] == [
    0.02, 0.0, 0.0, 0.0, 0.0
  ]
  assert protocol["hold_failed_hand_ik"] is False
  environment = environments[0]
  assert environment["MUJOCO_GL"] == "osmesa"
  assert environment["PYOPENGL_PLATFORM"] == "osmesa"
  assert environment["KAIHAND_RENDER_BACKEND"] == "software"
  assert environment["LIBGL_ALWAYS_SOFTWARE"] == "1"
  assert environment["GALLIUM_DRIVER"] == "llvmpipe"
  assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in environment
  assert "MUJOCO_EGL_DEVICE_ID" not in environment


def test_batch_counts_guard_rejection_and_continues_all_seeds(
  tmp_path, monkeypatch
):
  manifest_path, manifest = _prepare(tmp_path, monkeypatch)
  module = _module(BATCH, monkeypatch)
  main = module["main"]
  namespace = main.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda _runner: {"source": "frozen"})
  invoked_seeds = []

  def fake_run(command, **_kwargs):
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    seed = int(command[command.index("--seed") + 1])
    invoked_seeds.append(seed)
    summary = {
      "task": "pick-place",
      "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": "EgoTouch",
      "prediction_horizon": 50,
      "action_dim": 108,
      "execute_steps": 5,
      "control_hz": 30,
      "seed": seed,
      "guard_mode": "strict",
      "guard_events": [],
      "physics_valid": True,
      "diagnostic_only": False,
      "formal_metrics_valid": True,
      "task_success": seed == 1,
      "hand_ik_orientation_weights_m_per_rad": [0.02, 0.0, 0.0, 0.0, 0.0],
      "hold_failed_hand_ik": False,
      "evaluation": {"success": seed == 1},
      "stats": {"requests": 1, "action_steps": 0 if seed == 0 else 5},
    }
    if seed == 0:
      summary.update({
        "status": "policy_action_rejected",
        "policy_failure": {
          "kind": "kinematic_reachability_guard",
          "guard": "wrist_target_jump",
          "message": "right wrist target jump 0.500m exceeds guard",
          "request": 1,
          "action_index": 0,
          "sim_time": 0.0,
          "action_executed": False,
        },
      })
      review = trial / "review"
      review.mkdir()
      (review / "review.json").write_text(json.dumps({
        "completed": True,
        "second_camera": "global",
        "model_input_cameras_displayed": ["head", "right_wrist"],
      }), encoding="utf-8")
      for name in ("review.mp4", "frames.jsonl", "first_frame.png", "last_frame.png"):
        (review / name).write_bytes(b"artifact")
    else:
      summary["status"] = "success"
    (trial / "summary.json").write_text(
      json.dumps(summary), encoding="utf-8"
    )
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  output = tmp_path / "guarded-evaluation"
  assert main([
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output),
    "--seeds", "0", "1",
    "--execute-steps", "5",
    "--video-count", "1",
    "--guard-mode", "strict",
    "--model-device", "cpu",
    "--render-backend", "software-osmesa",
  ]) == 0
  assert invoked_seeds == [0, 1]
  aggregate = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  assert aggregate["attempted"] == aggregate["valid_trials"] == 2
  assert aggregate["successes"] == 1
  assert aggregate["task_successes"] == 1
  assert aggregate["success_rate"] == 0.5
  assert aggregate["complete"] is True
  assert aggregate["status_counts"] == {
    "policy_action_rejected": 1,
    "success": 1,
  }
  rejected = aggregate["trials"][0]
  assert rejected["valid_trial"] is True
  assert rejected["success"] is False
  assert rejected["task_success"] is False
  assert rejected["returncode"] == 0
  assert rejected["policy_failure"]["action_executed"] is False


def test_batch_still_aborts_on_egotouch_infrastructure_error(
  tmp_path, monkeypatch
):
  manifest_path, manifest = _prepare(tmp_path, monkeypatch)
  module = _module(BATCH, monkeypatch)
  main = module["main"]
  namespace = main.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda _runner: {"source": "frozen"})
  invoked_seeds = []

  def fake_run(command, **_kwargs):
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    seed = int(command[command.index("--seed") + 1])
    invoked_seeds.append(seed)
    (trial / "summary.json").write_text(json.dumps({
      "task": "pick-place",
      "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": "EgoTouch",
      "prediction_horizon": 50,
      "action_dim": 108,
      "execute_steps": 5,
      "control_hz": 30,
      "seed": seed,
      "guard_mode": "observe",
      "guard_events": [],
      "stop_reason": "fatal_error",
      "physics_valid": True,
      "diagnostic_only": True,
      "formal_metrics_valid": False,
      "task_success": False,
      "hand_ik_orientation_weights_m_per_rad": [0.02, 0.0, 0.0, 0.0, 0.0],
      "hold_failed_hand_ik": False,
      "status": "error",
      "error": "RuntimeError: worker contract mismatch",
      "evaluation": {"success": False},
      "stats": {"requests": 0, "action_steps": 0},
    }), encoding="utf-8")
    return SimpleNamespace(returncode=1)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  output = tmp_path / "broken-evaluation"
  with pytest.raises(RuntimeError, match="batch paused"):
    main([
      "--deployment-manifest", str(manifest_path),
      "--output-dir", str(output),
      "--seeds", "0", "1",
      "--execute-steps", "5",
      "--video-count", "0",
      "--model-device", "cpu",
      "--render-backend", "software-osmesa",
    ])
  assert invoked_seeds == [0]
  aggregate = json.loads((output / "summary.json").read_text(encoding="utf-8"))
  assert aggregate["valid_trials"] == 0
  assert aggregate["complete"] is False


def test_resume_accepts_completed_egotouch_guard_rejection(monkeypatch):
  module = _module(BATCH, monkeypatch)
  rejection = {
    "kind": "kinematic_reachability_guard",
    "guard": "hand_reachability",
    "message": "right hand target is too far",
    "request": 3,
    "action_index": 4,
    "action_executed": False,
  }
  module["_require_valid_resume_rows"]([{
    "status": "policy_action_rejected",
    "success": False,
    "valid_trial": True,
    "returncode": 0,
    "policy_failure": rejection,
  }], "EgoTouch")
  with pytest.raises(RuntimeError, match="invalid or incomplete"):
    module["_require_valid_resume_rows"]([{
      "status": "policy_action_rejected",
      "success": False,
      "valid_trial": True,
      "returncode": 0,
      "policy_failure": {**rejection, "action_executed": True},
    }], "EgoTouch")


def test_batch_aborts_when_rejected_trial_review_artifacts_are_missing(
  tmp_path, monkeypatch
):
  manifest_path, manifest = _prepare(tmp_path, monkeypatch)
  module = _module(BATCH, monkeypatch)
  main = module["main"]
  namespace = main.__globals__
  monkeypatch.setitem(namespace, "fingerprints", lambda _runner: {"source": "frozen"})
  invoked_seeds = []

  def fake_run(command, **_kwargs):
    trial = Path(command[command.index("--output-dir") + 1])
    trial.mkdir()
    seed = int(command[command.index("--seed") + 1])
    invoked_seeds.append(seed)
    (trial / "summary.json").write_text(json.dumps({
      "task": "pick-place",
      "checkpoint_path": manifest["checkpoint_path"],
      "checkpoint_sha256": manifest["checkpoint_sha256"],
      "model_family": "EgoTouch",
      "prediction_horizon": 50,
      "action_dim": 108,
      "execute_steps": 5,
      "control_hz": 30,
      "seed": seed,
      "guard_mode": "strict",
      "guard_events": [],
      "stop_reason": "guard",
      "physics_valid": True,
      "diagnostic_only": False,
      "formal_metrics_valid": True,
      "task_success": False,
      "hand_ik_orientation_weights_m_per_rad": [0.02, 0.0, 0.0, 0.0, 0.0],
      "hold_failed_hand_ik": False,
      "status": "policy_action_rejected",
      "policy_failure": {
        "kind": "kinematic_reachability_guard",
        "guard": "arm_ik",
        "message": "right arm IK failed",
        "request": 1,
        "action_index": 0,
        "action_executed": False,
      },
      "evaluation": {"success": False},
      "stats": {"requests": 1, "action_steps": 0},
    }), encoding="utf-8")
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
  with pytest.raises(RuntimeError, match="review missing or empty"):
    main([
      "--deployment-manifest", str(manifest_path),
      "--output-dir", str(tmp_path / "missing-review"),
      "--seeds", "0", "1",
      "--execute-steps", "5",
      "--video-count", "1",
      "--guard-mode", "strict",
      "--model-device", "cpu",
      "--render-backend", "software-osmesa",
    ])
  assert invoked_seeds == [0]


def test_explicit_reference_requires_ok_trace_metadata_and_three_plots(
  tmp_path, monkeypatch
):
  module = _module(BATCH, monkeypatch)
  validate = module["_validate_review_artifacts"]
  trial = tmp_path / "seed_000"
  review = trial / "review"
  review.mkdir(parents=True)
  for name in module["REVIEW_FILES"]:
    if name != "review.json":
      (review / name).write_bytes(b"artifact")
  manifest = {
    "observation_contract": {"cameras": ["head", "right_wrist"]}
  }
  review_meta = {
    "completed": True,
    "second_camera": "global",
    "model_input_cameras_displayed": ["head", "right_wrist"],
    "comparison_plots": {"status": "reference_unavailable"},
  }
  (review / "review.json").write_text(
    json.dumps(review_meta), encoding="utf-8"
  )

  with pytest.raises(RuntimeError, match="comparison missing or empty"):
    validate(trial, 0, manifest, require_comparison=True)

  for name in module["COMPARISON_FILES"]:
    path = review / name
    if name == "evaluation_comparison.json":
      path.write_text(
        json.dumps({"status": "reference_unavailable"}), encoding="utf-8"
      )
    else:
      path.write_bytes(b"artifact")
  with pytest.raises(RuntimeError, match="comparison did not complete"):
    validate(trial, 0, manifest, require_comparison=True)

  review_meta["comparison_plots"]["status"] = "ok"
  (review / "review.json").write_text(
    json.dumps(review_meta), encoding="utf-8"
  )
  (review / "evaluation_comparison.json").write_text(
    json.dumps({"status": "ok"}), encoding="utf-8"
  )
  assert validate(trial, 0, manifest, require_comparison=True) == review_meta


def test_batch_render_environment_defaults_to_nvidia_egl(tmp_path, monkeypatch):
  module = _module(BATCH, monkeypatch)
  namespace = module["main"].__globals__
  vendor = tmp_path / "nvidia-egl.json"
  vendor.write_text("{}", encoding="utf-8")
  monkeypatch.setitem(namespace, "EGL_VENDOR", vendor)
  monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
  monkeypatch.setenv("GALLIUM_DRIVER", "llvmpipe")

  environment = namespace["_render_environment"]("hardware-egl")

  assert environment["MUJOCO_GL"] == "egl"
  assert environment["PYOPENGL_PLATFORM"] == "egl"
  assert environment["KAIHAND_RENDER_BACKEND"] == "hardware"
  assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(vendor)
  assert "LIBGL_ALWAYS_SOFTWARE" not in environment
  assert "GALLIUM_DRIVER" not in environment


def test_batch_software_render_does_not_require_nvidia_egl(tmp_path, monkeypatch):
  module = _module(BATCH, monkeypatch)
  namespace = module["main"].__globals__
  monkeypatch.setitem(namespace, "EGL_VENDOR", tmp_path / "missing.json")
  monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/stale/nvidia.json")
  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "7")

  environment = namespace["_render_environment"]("software-osmesa")

  assert environment["MUJOCO_GL"] == "osmesa"
  assert environment["PYOPENGL_PLATFORM"] == "osmesa"
  assert "__EGL_VENDOR_LIBRARY_FILENAMES" not in environment
  assert "MUJOCO_EGL_DEVICE_ID" not in environment


def test_dispatcher_allows_only_pickplace_egotouch_right_wrist(
  tmp_path, monkeypatch, capsys
):
  manifest_path, _ = _prepare(tmp_path, monkeypatch)
  capsys.readouterr()
  module = _module(ROOT / "scripts/evaluate/evaluate.py", monkeypatch)
  output = tmp_path / "dispatcher-evaluation"
  assert module["main"]([
    "--task", "pick-place",
    "--model-family", "egotouch",
    "--deployment-manifest", str(manifest_path),
    "--output-dir", str(output),
    "--num-trials", "1",
    "--dry-run",
    "--", "--model-device", "cpu",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["model_family"] == "egotouch"
  assert payload["cameras"] == ["head", "right_wrist"]
  assert payload["execute_steps"] == 5
  assert payload["command"][1].endswith("evaluate_pickplace_policy_batch.py")
  assert payload["command"][-2:] == ["--model-device", "cpu"]
