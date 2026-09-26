import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.pipeline.conversion import (
  adapter_catalog,
  adapter_for,
  inspect_raw_dataset,
)
from kaihand_tactile_env.shared.policy_cameras import (
  pi05_image_payload,
  policy_camera_names,
)


def raw_episode(path, scene, cameras=("head", "left_wrist", "right_wrist")):
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps({"scene": scene})
    group = file.create_group("cameras")
    for camera in cameras:
      group.create_group(camera)


def test_raw_discovery_is_recursive_and_task_auto_is_metadata_driven(tmp_path):
  first = tmp_path / "attempt_1/raw/episode.h5"
  second = tmp_path / "attempt_2/raw/episode.h5"
  raw_episode(first, "poker-draw")
  raw_episode(second, "poker-draw")

  dataset = inspect_raw_dataset(tmp_path, task="auto", cameras=("head", "right_wrist"))
  assert dataset.task == "poker-draw"
  assert dataset.episodes == (first, second)
  assert dataset.cameras == ("head", "right_wrist")
  assert adapter_for(dataset, "egosteer").backend == "convert_card_to_egosteer.py"


def test_explicit_task_filters_a_mixed_raw_root(tmp_path):
  card = tmp_path / "collection/poker-draw/000000/raw/episode.h5"
  usb = tmp_path / "collection/usb-insert/000000/raw/episode.h5"
  raw_episode(card, "poker-draw")
  raw_episode(usb, "usb-insert")

  with pytest.raises(ValueError, match="exactly one recorded scene"):
    inspect_raw_dataset(tmp_path, task="auto", cameras=("head",))
  dataset = inspect_raw_dataset(tmp_path, task="usb-insert", cameras=("head",))
  assert dataset.task == "usb-insert"
  assert dataset.episodes == (usb,)


def test_collection_discovery_selects_only_published_successes(tmp_path):
  root = tmp_path / "collection"
  success = root / "usb-insert/000001/attempt_001/data/raw/usb_000000.h5"
  failed = root / "usb-insert/000002/attempt_001/data/raw/usb_000000.h5"
  raw_episode(success, "usb-insert")
  raw_episode(failed, "usb-insert")
  (root / "collection.json").write_text(json.dumps({
    "schema": "task_collection_v2", "tasks": ["usb-insert"],
    "collection_mode": "attempts",
  }))
  (root / "summary.json").write_text(json.dumps({
    "episodes": [
      {"task": "usb-insert", "status": "success", "episode_index": 1,
       "hdf5": [success.relative_to(root).as_posix()]},
      {"task": "usb-insert", "status": "failed", "episode_index": 2,
       "hdf5": [failed.relative_to(root).as_posix()]},
    ],
    "by_task": {"usb-insert": {"success": 1}},
  }))
  dataset = inspect_raw_dataset(root, task="usb-insert", cameras=("head",))
  assert dataset.episodes == (success,)


def test_conversion_contract_rejects_missing_camera_and_accepts_shared_actions(tmp_path):
  raw_episode(tmp_path / "raw/episode.h5", "bulb-screw", cameras=("head",))
  with pytest.raises(ValueError, match="lacks requested cameras"):
    inspect_raw_dataset(tmp_path, task="auto")
  dataset = inspect_raw_dataset(tmp_path, task="auto", cameras=("head",))
  contract = adapter_for(dataset, "pi05")
  assert contract.backend == "convert_shared_to_lerobot.py"
  assert contract.resumable is True


@pytest.mark.parametrize(
  ("task", "cameras", "backend"),
  [
    ("pick-place", ("head",), "convert_pickplace_unified_to_lerobot.py"),
    ("poker-draw", ("head", "right_wrist"), "convert_poker_unified_to_lerobot.py"),
    ("bulb-screw", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
    ("install-ram", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
    ("vase-wipe", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
  ],
)
def test_remaining_pi05_adapters_are_registered(tmp_path, task, cameras, backend):
  raw_episode(tmp_path / task / "000001/attempt_001/data/raw/episode.h5", task)
  dataset = inspect_raw_dataset(tmp_path, task=task, cameras=cameras)
  contract = adapter_for(dataset, "pi05")
  assert contract.backend == backend
  assert contract.resumable is True


def test_pi05_camera_contract_is_explicit(tmp_path):
  raw_episode(tmp_path / "raw/episode.h5", "usb-insert")
  three = inspect_raw_dataset(tmp_path, cameras=("head", "left_wrist", "right_wrist"))
  with pytest.raises(ValueError, match="currently supports camera sets"):
    adapter_for(three, "pi05")
  legacy = inspect_raw_dataset(tmp_path, cameras=("head", "right_wrist"))
  assert adapter_for(legacy, "pi05").backend == "convert_usb_unified_to_lerobot.py"


def test_conversion_cli_builds_path_independent_dry_run(tmp_path, capsys):
  source = tmp_path / "anywhere/raw"
  raw_episode(source / "episode_000001_card_right.h5", "poker-draw")
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  result = module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "anywhere/output"),
    "--format", "egosteer",
    "--task", "auto",
    "--cameras", "head", "left_wrist", "right_wrist",
    "--dry-run",
  ])
  assert result == 0
  output = capsys.readouterr().out
  assert str(source.resolve()) in output
  assert str((tmp_path / "anywhere/output").resolve()) in output
  assert "convert_card_to_egosteer.py" in output


def test_conversion_cli_lists_registry_without_dataset_paths(capsys):
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"](["--list-support"]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["formats"] == ["egosteer", "pi05", "egotouch", "lerobot-v3"]
  assert payload["adapters"] == list(adapter_catalog())
  assert {
    (row["task"], row["format"])
    for row in payload["adapters"]
  } >= {
    ("pick-place", "egosteer"),
    ("poker-draw", "pi05"),
    ("usb-insert", "egotouch"),
    ("whiteboard-wipe", "egosteer"),
    ("whiteboard-wipe", "pi05"),
    ("whiteboard-wipe", "egotouch"),
    ("usb-insert", "lerobot-v3"),
  }


@pytest.mark.parametrize("task", (
  "pick-place", "poker-draw", "usb-insert", "bulb-screw", "install-ram",
  "vase-wipe", "whiteboard-wipe", "sponge-grasp",
))
def test_lerobot_v3_adapter_is_registered_for_supported_tasks(tmp_path, task):
  raw_episode(tmp_path / "episode.h5", task)
  dataset = inspect_raw_dataset(
    tmp_path, task=task, cameras=("head", "right_wrist"),
  )
  contract = adapter_for(dataset, "lerobot-v3")
  assert contract.backend == "convert_kaihand_to_lerobot_v3.py"
  assert contract.resumable is True


@pytest.mark.parametrize("task", ("poker-draw", "usb-insert", "sponge-grasp"))
def test_pi05_defaults_to_task_camera_contract(tmp_path, capsys, task):
  source = tmp_path / "raw"
  raw_episode(source / "episode.h5", task)
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source), "--output-dir", str(tmp_path / "out"),
    "--task", task, "--format", "pi05", "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["cameras"] == ["head", "right_wrist"]


def test_lerobot_v3_uses_one_unified_command_and_fixed_cameras(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(source / "episode.h5", "usb-insert")
  work_dir = tmp_path / "work"
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source), "--output-dir", str(tmp_path / "out"),
    "--task", "usb-insert", "--format", "lerobot-v3",
    "--workers", "2", "--work-dir", str(work_dir),
    "--repo-id", "kaihand/usb_test",
    "--lerobot-v3-python", sys.executable, "--resume", "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  command = payload["command"]
  assert payload["cameras"] == ["head", "right_wrist"]
  assert command[1].endswith("convert_kaihand_to_lerobot_v3.py")
  assert command[command.index("--task") + 1] == "usb-insert"
  assert command[command.index("--repo-id") + 1] == "kaihand/usb_test"
  assert command[command.index("--work-dir") + 1] == str(work_dir)
  assert "--resume" in command
  assert "--openpi-root" not in command

  with pytest.raises(ValueError, match="currently supports camera sets"):
    module["main"]([
      "--input-dir", str(source), "--output-dir", str(tmp_path / "out"),
      "--format", "lerobot-v3", "--cameras", "head",
      "--lerobot-v3-python", sys.executable, "--dry-run",
    ])


def test_lerobot_v3_defaults_to_sponge_camera_contract(tmp_path, capsys):
  raw_episode(tmp_path / "episode.h5", "sponge-grasp")
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
    "--format", "lerobot-v3", "--lerobot-v3-python", sys.executable,
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["task"] == "sponge-grasp"
  assert payload["cameras"] == ["head", "right_wrist"]


def test_lerobot_v3_resume_dispatches_existing_output(tmp_path, monkeypatch):
  source = tmp_path / "raw"
  raw_episode(source / "episode.h5", "usb-insert")
  output = tmp_path / "published"
  output.mkdir()
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  seen = []

  def fake_run(command, *, check):
    seen.append(command)
    assert check is False
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(module["subprocess"], "run", fake_run)
  assert module["main"]([
    "--input-dir", str(source), "--output-dir", str(output),
    "--format", "lerobot-v3", "--task", "usb-insert",
    "--lerobot-v3-python", sys.executable, "--resume",
  ]) == 0
  assert len(seen) == 1 and "--resume" in seen[0]


@pytest.mark.parametrize(
  ("output_format", "backend"),
  [
    ("egosteer", "convert_card_to_egosteer.py"),
    ("pi05", "convert_shared_to_lerobot.py"),
    ("egotouch", "shared.tict_export"),
  ],
)
def test_whiteboard_conversion_adapters_accept_camera_subsets(
  tmp_path, output_format, backend
):
  raw_episode(tmp_path / "000001/attempt_001/data/raw/episode.h5", "whiteboard-wipe")
  dataset = inspect_raw_dataset(
    tmp_path, task="whiteboard-wipe", cameras=("head", "left_wrist")
  )
  assert adapter_for(dataset, output_format).backend == backend


def test_whiteboard_pi05_dry_run_uses_local_openpi_wrapper(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(source / "000001/attempt_001/data/raw/episode.h5", "whiteboard-wipe")
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "pi05",
    "--task", "whiteboard-wipe",
    "--cameras", "head", "left_wrist", "right_wrist",
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["adapter"] == "convert_shared_to_lerobot.py"
  assert "convert_shared_to_lerobot.py" in payload["command"][1]
  assert "--openpi-root" in payload["command"]


def test_sponge_pi05_dry_run_uses_right_side_shared_contract(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(
    source / "sponge-grasp/000001/attempt_001/data/raw/sponge_grasp_000000.h5",
    "sponge-grasp",
  )
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "pi05",
    "--task", "sponge-grasp",
    "--cameras", "head", "right_wrist",
    "--workers", "4",
    "--dataset-name", "sponge_grasp_0924_200",
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  command = payload["command"]
  assert payload["adapter"] == "convert_shared_to_lerobot.py"
  assert "convert_shared_to_lerobot.py" in command[1]
  assert command[command.index("--task") + 1] == "sponge-grasp"
  assert command[command.index("--cameras") + 1 :] == [
    "head", "right_wrist", "--dataset-name", "sponge_grasp_0924_200",
  ]
  assert command[command.index("--fingerprint-workers") + 1] == "4"
  assert "--image-writer-threads" not in command
  assert adapter_for(
    inspect_raw_dataset(source, task="sponge-grasp", cameras=("head", "right_wrist")),
    "pi05",
  ).resumable is False


def test_usb_pi05_dry_run_uses_unified_collection_wrapper(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(
    source / "usb-insert/000001/attempt_001/data/raw/usb_000000.h5",
    "usb-insert",
  )
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "pi05",
    "--task", "usb-insert",
    "--cameras", "head", "right_wrist",
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["adapter"] == "convert_usb_unified_to_lerobot.py"
  assert "convert_usb_unified_to_lerobot.py" in payload["command"][1]
  assert "--openpi-root" in payload["command"]
  assert "--image-writer-threads" in payload["command"]


@pytest.mark.parametrize(
  ("task", "cameras", "backend"),
  [
    ("pick-place", ("head",), "convert_pickplace_unified_to_lerobot.py"),
    ("poker-draw", ("head", "right_wrist"), "convert_poker_unified_to_lerobot.py"),
    ("bulb-screw", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
    ("install-ram", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
    ("vase-wipe", ("head", "right_wrist"), "convert_shared_to_lerobot.py"),
  ],
)
def test_remaining_pi05_dry_runs_use_local_wrappers(
  tmp_path, capsys, task, cameras, backend
):
  source = tmp_path / "raw"
  raw_episode(source / task / "000001/attempt_001/data/raw/episode.h5", task)
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "pi05",
    "--task", task,
    "--cameras", *cameras,
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["adapter"] == backend
  assert backend in payload["command"][1]
  assert "--openpi-root" in payload["command"]
  assert "--image-writer-threads" not in payload["command"]


def test_conversion_cli_defaults_to_head_and_checks_episode_count(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(source / "one/episode_000001_card_right.h5", "poker-draw")
  raw_episode(source / "two/episode_000002_card_right.h5", "poker-draw")
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "egosteer",
    "--dry-run",
    "--expected-episodes", "2",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["cameras"] == ["head"]
  with pytest.raises(ValueError, match="expected 3 selected Raw episodes"):
    module["main"]([
      "--input-dir", str(source),
      "--output-dir", str(tmp_path / "other"),
      "--format", "egosteer",
      "--dry-run",
      "--expected-episodes", "3",
    ])


def test_egotouch_resume_validates_published_job(tmp_path):
  source = (tmp_path / "episode.h5").resolve()
  source.write_bytes(b"source")
  digest = "a" * 64
  source.with_suffix(".json").write_text(json.dumps({
    "episode": source.name,
    "sha256": digest,
  }))
  destination = tmp_path / "published"
  session = "episode-123"
  sidecar = destination / "tict_sidecars" / session
  sidecar.mkdir(parents=True)
  (destination / "dataset_audit.json").write_text(json.dumps({
    "contract_version": "kaihand-tict-release-v1",
    "passed": True,
    "sessions": [{"session_id": session}],
    "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": digest},
  }))
  (destination / "selector_manifest.json").write_text(json.dumps({
    "records": [{"frame_name": "00000"}],
  }))
  np.savez_compressed(
    sidecar / "fingertip_tactile_v1.npz",
    source_metadata_json=np.array(json.dumps({"camera": "head"})),
  )
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  record = module["_existing_tict_record"](
    source, destination, session, "head", expected_source_sha256=digest
  )
  assert record["status"] == "verified-existing"
  with pytest.raises(ValueError, match="camera mismatch"):
    module["_existing_tict_record"](
      source,
      destination,
      session,
      "right_wrist",
      expected_source_sha256=digest,
    )


def test_egotouch_resume_requires_matching_source_set(tmp_path):
  root = tmp_path / "raw"
  source = root / "poker-draw/000001/attempt_001/data/raw/episode.h5"
  raw_episode(source, "poker-draw")
  dataset = inspect_raw_dataset(root, task="poker-draw", cameras=("head",))
  output = tmp_path / "egotouch"
  output.mkdir()
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  validate = module["_validate_resume_manifest"]
  with pytest.raises(ValueError, match="without manifest"):
    validate(output, dataset)
  (output / "conversion_manifest.json").write_text(json.dumps({
    "schema": "kaihand_unified_egotouch_batch_v2",
    "task": "poker-draw", "cameras": ["head"],
    "source_root": str(root), "source_episodes": ["different.h5"],
  }))
  with pytest.raises(ValueError, match="different task/camera contract"):
    validate(output, dataset)
  manifest = json.loads((output / "conversion_manifest.json").read_text())
  manifest["source_episodes"] = [source.relative_to(root).as_posix()]
  (output / "conversion_manifest.json").write_text(json.dumps(manifest))
  (output / "episodes/stale-session").mkdir(parents=True)
  with pytest.raises(ValueError, match="stale sessions"):
    validate(output, dataset)


def test_egotouch_dry_run_does_not_create_output(tmp_path, capsys):
  source = tmp_path / "raw"
  raw_episode(source / "episode_000001_card_right.h5", "poker-draw")
  output = tmp_path / "planned"
  module = run_path(str(Path(__file__).parents[1] / "scripts/convert/convert.py"))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(output),
    "--format", "egotouch",
    "--cameras", "head",
    "--dry-run",
  ]) == 0
  assert not output.exists()
  payload = json.loads(capsys.readouterr().out)
  assert payload["mode"] == "dry-run"


def test_evaluation_dispatch_dry_run(tmp_path, capsys):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "usb-insert",
    "model_family": "pi0.5",
    "deployment_id": "usb-test",
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  result = module["main"]([
    "--task", "usb-insert",
    "--model-family", "pi05",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--cameras", "head", "right_wrist",
    "--num-trials", "4",
    "--seed-start", "10",
    "--video-count", "2",
    "--execute-steps", "6",
    "--max-sim-seconds", "45",
    "--record-fps", "5",
    "--dry-run",
    "--", "--server", "ws://127.0.0.1:18783",
  ])
  assert result == 0
  output = capsys.readouterr().out
  assert "evaluate_usb_pi05_policy_batch.py" in output
  assert "ws://127.0.0.1:18783" in output
  payload = json.loads(output)
  assert payload["num_trials"] == 4
  assert payload["seeds"] == [10, 11, 12, 13]
  assert payload["video_count"] == 2
  assert payload["execute_steps"] == 6
  assert payload["max_sim_seconds"] == 45.0
  assert payload["record_fps"] == 5
  assert payload["command"][-2:] == ["--server", "ws://127.0.0.1:18783"]
  assert payload["command"][payload["command"].index("--execute-steps") + 1] == "6"


def test_evaluation_cli_lists_task_model_support_without_manifest(capsys):
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert module["main"](["--list-support"]) == 0
  payload = json.loads(capsys.readouterr().out)
  rows = {(row["task"], row["model_family"]): row["backend"]
          for row in payload["runners"]}
  assert rows[("usb-insert", "pi05")] == "evaluate_usb_pi05_policy_batch.py"
  assert rows[("pick-place", "lingbot-vla2")] == "evaluate_pickplace_lingbot_vla2_batch.py"
  assert rows[("poker-draw", "pi05+trex")] == "evaluate_poker_pi05+trex_policy_batch.py"
  assert ("sponge-grasp", "pi05") not in rows


def test_lingbot_evaluation_dispatch_preserves_frozen_model_identity(tmp_path, capsys):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "pick-place", "model_family": "LingBot-VLA-2.0",
    "prediction_horizon": 50,
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert module["main"]([
    "--task", "pick-place", "--model-family", "lingbot-vla2",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--num-trials", "1", "--dry-run", "--", "--server", "127.0.0.1:8006",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["execute_steps"] == 50
  assert payload["command"][1].endswith("evaluate_pickplace_lingbot_vla2_batch.py")
  assert payload["command"][-2:] == ["--server", "127.0.0.1:8006"]


def test_evaluation_defaults_to_twenty_recorded_trials_at_horizon(
  tmp_path, capsys
):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "poker-draw",
    "model_family": "egosteer",
    "prediction_horizon": 32,
    "observation_contract": {"cameras": ["head"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert module["main"]([
    "--task", "poker-draw",
    "--model-family", "egosteer",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["num_trials"] == 20
  assert payload["seeds"] == list(range(20))
  assert payload["video_count"] == 20
  assert payload["execute_steps"] == 32
  assert len(payload["evaluations"]) == 1
  assert payload["evaluations"][0]["request"] == "horizon"
  assert payload["command"][payload["command"].index("--execute-steps") + 1] == "32"


def test_evaluation_uses_matching_checkpoint_dataset_for_reference(
  tmp_path, capsys
):
  dataset = tmp_path / "datasets/sim/install-ram/pi05/0920_200"
  legacy = tmp_path / "datasets/sim/install-ram/lerobot_v3/0920_200"
  (legacy / "meta").mkdir(parents=True)
  (legacy / "meta/info.json").write_text("{}", encoding="utf-8")
  checkpoint = dataset / "checkpoints/model/20000"
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "install-ram",
    "model_family": "pi05",
    "prediction_horizon": 30,
    "checkpoint_path": str(checkpoint),
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert module["main"]([
    "--task", "install-ram",
    "--model-family", "pi05",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  expected = str(legacy)
  assert payload["reference_dataset"] == expected
  assert payload["command"][payload["command"].index("--reference-dataset") + 1] == expected


def test_evaluation_preserves_legacy_reference_when_both_layouts_exist(tmp_path):
  dataset = tmp_path / "datasets/sim/install-ram/pi05/0920_200"
  legacy = tmp_path / "datasets/sim/install-ram/lerobot_v3/0920_200"
  for candidate in (dataset, legacy):
    (candidate / "meta").mkdir(parents=True)
    (candidate / "meta/info.json").write_text("{}", encoding="utf-8")
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))

  inferred = module["_reference_dataset_from_checkpoint"]({
    "checkpoint_path": str(dataset / "checkpoints/model/20000")
  }, "install-ram")

  assert inferred == legacy


def test_evaluation_prefers_colocated_fast_training_dataset_for_reference(
  tmp_path, capsys
):
  dataset = tmp_path / "datasets/sim/bulb-screw/pi05/0923_200"
  (dataset / "meta").mkdir(parents=True)
  (dataset / "meta/info.json").write_text("{}", encoding="utf-8")
  checkpoint = dataset / "checkpoints/pi05_kaihand/bulb-screw_bs128_25k/10000"
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "bulb-screw",
    "model_family": "pi05",
    "prediction_horizon": 30,
    "checkpoint_path": str(checkpoint),
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))

  assert module["main"]([
    "--task", "bulb-screw",
    "--model-family", "pi05",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--dry-run",
  ]) == 0

  payload = json.loads(capsys.readouterr().out)
  expected = str(dataset)
  assert payload["reference_dataset"] == expected
  assert payload["command"][payload["command"].index("--reference-dataset") + 1] == expected


def test_evaluation_explicit_reference_overrides_checkpoint_inference(
  tmp_path, capsys, monkeypatch
):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "install-ram",
    "model_family": "pi05",
    "prediction_horizon": 30,
    "checkpoint_path": (
      "/nas/chenxianchi/datasets/sim/install-ram/pi05/0920_200/"
      "checkpoints/model/20000"
    ),
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  explicit = tmp_path / "custom_reference"
  monkeypatch.chdir(tmp_path)
  assert module["main"]([
    "--task", "install-ram", "--model-family", "pi05",
    "--deployment-manifest", str(manifest),
    "--output-dir", str(tmp_path / "evaluation"),
    "--reference-dataset", explicit.name,
    "--reference-episode-index", "2", "--dry-run",
  ]) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["reference_dataset"] == str(explicit)
  assert payload["reference_episode_index"] == 2
  assert payload["command"][payload["command"].index("--reference-dataset") + 1] == str(explicit)


def test_all_registered_model_task_pairs_default_to_their_horizon(
  tmp_path, capsys
):
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert len(module["RUNNERS"]) == 20
  for task, family in module["RUNNERS"]:
    manifest = tmp_path / f"{task}_{family}.json"
    manifest.write_text(json.dumps({
      "task": task,
      "model_family": family,
      "prediction_horizon": 30,
      "observation_contract": {"cameras": ["head"]},
    }))
    assert module["main"]([
      "--task", task, "--model-family", family,
      "--deployment-manifest", str(manifest),
      "--output-dir", str(tmp_path / f"{task}_{family}_evaluation"),
      "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execute_steps"] == 30
    assert payload["command"][payload["command"].index("--execute-steps") + 1] == "30"


@pytest.mark.parametrize("missing", ["task", "model_family", "observation_contract"])
def test_evaluation_rejects_incomplete_deployment_manifest(tmp_path, missing):
  payload = {
    "task": "usb-insert", "model_family": "pi0.5",
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }
  del payload[missing]
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps(payload))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  with pytest.raises(ValueError, match="must declare"):
    module["_validate_manifest"](manifest, "usb-insert", "pi05")


def test_evaluation_rejects_child_override_of_validated_manifest(tmp_path):
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  with pytest.raises(SystemExit):
    module["parse_args"]([
      "--task", "usb-insert", "--model-family", "pi05",
      "--deployment-manifest", str(tmp_path / "approved.json"),
      "--output-dir", str(tmp_path / "evaluation"),
      "--", "--deployment-manifest", str(tmp_path / "other.json"),
    ])


def test_evaluation_rejects_more_videos_than_trials(tmp_path):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "usb-insert",
    "model_family": "egosteer",
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  with pytest.raises(SystemExit):
    module["main"]([
      "--task", "usb-insert",
      "--model-family", "egosteer",
      "--deployment-manifest", str(manifest),
      "--output-dir", str(tmp_path / "evaluation"),
      "--num-trials", "2",
      "--video-count", "3",
      "--dry-run",
    ])


def test_evaluation_rejects_execution_chunk_larger_than_manifest_horizon(tmp_path):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(json.dumps({
    "task": "usb-insert",
    "model_family": "pi0.5",
    "prediction_horizon": 4,
    "observation_contract": {"cameras": ["head", "right_wrist"]},
  }))
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  with pytest.raises(ValueError, match="exceeds deployment prediction_horizon"):
    module["main"]([
      "--task", "usb-insert",
      "--model-family", "pi05",
      "--deployment-manifest", str(manifest),
      "--output-dir", str(tmp_path / "evaluation"),
      "--execute-steps", "5",
      "--dry-run",
    ])


@pytest.mark.parametrize(
  "cameras",
  [
    ("head",),
    ("head", "left_wrist"),
    ("head", "right_wrist"),
    ("head", "left_wrist", "right_wrist"),
  ],
)
def test_policy_camera_contract_supports_optional_wrists(cameras):
  contract = {"cameras": list(cameras)}
  assert policy_camera_names(contract) == cameras
  images = {name: object() for name in cameras}
  payload = pi05_image_payload(images, contract)
  expected_keys = (
    {"image"}
    if cameras == ("head",)
    else {f"{name}_image" for name in cameras}
  )
  assert set(payload) == expected_keys


@pytest.mark.parametrize(
  "cameras",
  [
    ("left_wrist",),
    ("head", "head"),
    ("head", "right_wrist", "left_wrist"),
    ("head", "overhead"),
  ],
)
def test_policy_camera_contract_rejects_ambiguous_sets(cameras):
  with pytest.raises(RuntimeError):
    policy_camera_names({"cameras": list(cameras)})


@pytest.mark.parametrize(
  ("task", "family"),
  [
    ("bulb-screw", "egosteer"),
    ("install-ram", "pi05"),
    ("vase-wipe", "egotouch"),
    ("whiteboard-wipe", "egosteer"),
    ("whiteboard-wipe", "pi05"),
    ("whiteboard-wipe", "egotouch"),
  ],
)
def test_new_tasks_dispatch_to_shared_closed_loop_batch(
  tmp_path, capsys, task, family
):
  manifest = tmp_path / f"{task}_{family}.json"
  manifest.write_text(
    json.dumps(
      {
        "task": task,
        "model_family": family,
        "deployment_id": "test-deployment",
        "prediction_horizon": 32,
        "observation_contract": {"cameras": ["head"]},
      }
    )
  )
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  assert module["main"](
    [
      "--task",
      task,
      "--model-family",
      family,
      "--deployment-manifest",
      str(manifest),
      "--output-dir",
      str(tmp_path / "evaluation"),
      "--dry-run",
    ]
  ) == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["evaluations"][0]["command"][1].endswith(
    "evaluate_shared_task_policy_batch.py"
  )


def test_egotouch_dispatch_rejects_wrist_camera_contract(tmp_path):
  manifest = tmp_path / "deployment.json"
  manifest.write_text(
    json.dumps(
      {
        "task": "vase-wipe",
        "model_family": "egotouch",
        "observation_contract": {"cameras": ["head", "right_wrist"]},
      }
    )
  )
  module = run_path(str(Path(__file__).parents[1] / "scripts/evaluate/evaluate.py"))
  with pytest.raises(ValueError, match="single-RGB"):
    module["main"](
      [
        "--task",
        "vase-wipe",
        "--model-family",
        "egotouch",
        "--deployment-manifest",
        str(manifest),
        "--output-dir",
        str(tmp_path / "evaluation"),
        "--dry-run",
      ]
    )
