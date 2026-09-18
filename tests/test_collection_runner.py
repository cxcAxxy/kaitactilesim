"""Batch isolation, failure classification and resume without a simulation rollout."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect/collect.py"
spec = importlib.util.spec_from_file_location("collection_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _write_vase_fixture(cmd, content, *, success):
  data = Path(cmd[cmd.index("--output-dir") + 1])
  raw = data / "raw"
  raw.mkdir(parents=True)
  episode = raw / "episode.h5"
  episode.write_bytes(content)
  (raw / "episode.json").write_text(
    json.dumps(
      {
        "episode": episode.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "validation": {"valid": True},
      }
    )
  )
  (data / "result.json").write_text(json.dumps({"success": success}))


def test_commands_and_seed_isolation(tmp_path):
  seeds = {runner.episode_seed(7, task, i) for task in runner.TASKS for i in range(10)}
  assert len(seeds) == 10 * len(runner.TASKS)
  assert runner.episode_seed(7, "vase-wipe", 0) == runner.episode_seed(
    7, "vase-wipe", 0
  )
  for task in runner.TASKS:
    cmd = runner.command(task, tmp_path / task, 42)
    assert Path(cmd[1]).is_file()
    assert str(tmp_path / task) in cmd
    assert "--replace-existing" not in cmd
  for task in ("bulb-screw", "install-ram"):
    cmd = runner.command(task, tmp_path, 42)
    assert cmd[cmd.index("--position-seed") + 1] == "42"
  assert "--stain-seed" in runner.command("vase-wipe", tmp_path, 42)
  assert "--stain-seed" not in runner.command("vase-wipe", tmp_path, 42, True)
  assert "--ink-seed" in runner.command("whiteboard-wipe", tmp_path, 42)
  assert "--ink-seed" not in runner.command("whiteboard-wipe", tmp_path, 42, True)
  assert runner.environment("hardware")["LIBGL_ALWAYS_SOFTWARE"] == "0"
  software = runner.environment("software")
  assert software["KAIHAND_RENDER_BACKEND"] == "software"
  assert software["MUJOCO_GL"] == "osmesa"
  assert software["PYOPENGL_PLATFORM"] == "osmesa"


def test_existing_tasks_use_unified_three_camera_collection_contract(tmp_path):
  pick_place = runner.command("pick-place", tmp_path / "pick-place", 42)
  poker = runner.command("poker-draw", tmp_path / "poker-draw", 42)
  usb = runner.command("usb-insert", tmp_path / "usb-insert", 42)

  pick_cameras = pick_place.index("--cameras")
  poker_cameras = poker.index("--cameras")
  usb_cameras = usb.index("--cameras")
  expected = ["head", "left_wrist", "right_wrist"]
  assert pick_place[pick_cameras + 1 : pick_place.index("--rgb-only")] == expected
  assert poker[poker_cameras + 1 : poker.index("--rgb-only")] == expected
  assert usb[usb_cameras + 1 : usb.index("--xy-jitter-mm")] == expected
  for command in (pick_place, poker, usb):
    assert command[command.index("--camera-hz") + 1] == "30"
  assert pick_place[pick_place.index("--object") + 1] == "cylinder"
  assert pick_place[pick_place.index("--side") + 1] == "right"
  assert poker[poker.index("--acceptance-policy") + 1] == "task-completion-v1"
  assert usb[usb.index("--motion-profile") + 1] == "fast"


def test_new_tasks_share_the_same_robot_camera_contract(tmp_path):
  for task in ("bulb-screw", "vase-wipe", "install-ram", "whiteboard-wipe"):
    command = runner.command(task, tmp_path / task, 42)
    start = command.index("--cameras") + 1
    assert command[start : start + 3] == ["head", "left_wrist", "right_wrist"]
    assert command[command.index("--camera-hz") + 1] == "30"
    assert "--raw-only" in command
  assert runner.command("bulb-screw", tmp_path, 42)[
    runner.command("bulb-screw", tmp_path, 42).index("--buffer-rows") + 1
  ] == "128"
  assert runner.command("install-ram", tmp_path, 42)[
    runner.command("install-ram", tmp_path, 42).index("--buffer-rows") + 1
  ] == "64"
  assert runner.command("vase-wipe", tmp_path, 42)[
    runner.command("vase-wipe", tmp_path, 42).index("--buffer-rows") + 1
  ] == "128"
  assert runner.command("whiteboard-wipe", tmp_path, 42)[
    runner.command("whiteboard-wipe", tmp_path, 42).index("--buffer-rows") + 1
  ] == "128"


def test_staged_publication_verifies_bytes_and_renames_atomically(tmp_path):
  staged = tmp_path / "cpfs/episode"
  target = tmp_path / "nas/attempt/data"
  target.parent.mkdir(parents=True)
  raw = staged / "raw"
  raw.mkdir(parents=True)
  content = b"raw episode bytes"
  episode = raw / "episode.h5"
  episode.write_bytes(content)
  (raw / "episode.json").write_text(
    json.dumps(
      {
        "episode": episode.name,
        "sha256": hashlib.sha256(content).hexdigest(),
      }
    )
  )
  hashes, sizes = runner.publish_staged_data(staged, target)
  assert target.is_dir()
  assert runner.actual_artifact_hashes(target) == hashes
  assert runner.artifact_sizes(target) == sizes
  assert not list(target.parent.glob(".data.publishing-*"))


def test_nonzero_and_timeout_are_not_completed(tmp_path):
  failure = tmp_path / "failure"
  failure.mkdir()
  result = runner.run_episode(
    [sys.executable, "-c", "raise SystemExit(2)"], failure, os.environ.copy(), 3
  )
  assert result["status"] == "failed" and result["returncode"] == 2
  timeout = tmp_path / "timeout"
  timeout.mkdir()
  result = runner.run_episode(
    [sys.executable, "-c", "import time; time.sleep(20)"],
    timeout,
    os.environ.copy(),
    0.2,
  )
  assert result["status"] == "timeout"
  pid = runner.read(timeout / "process.json")["pid"]
  with pytest.raises(ProcessLookupError):
    os.kill(pid, 0)


def test_interrupt_stops_child_and_retains_log(tmp_path):
  child = "import os,signal,time; time.sleep(.1); os.kill(os.getppid(),signal.SIGINT); time.sleep(20)"
  result = runner.run_episode(
    [sys.executable, "-c", child], tmp_path, os.environ.copy(), 5
  )
  assert result["status"] == "interrupted"
  assert (tmp_path / "collector.log").exists()
  with pytest.raises(ProcessLookupError):
    os.kill(runner.read(tmp_path / "process.json")["pid"], 0)


def test_resume_preserves_failed_attempt_and_retries_interruption(
  monkeypatch, tmp_path
):
  monkeypatch.setattr(runner, "source_hashes", lambda: {"test": "123"})
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *a, **kw: SimpleNamespace(returncode=0, stdout="OK", stderr=""),
  )
  statuses = iter(["interrupted", "completed", "failed"])
  calls = []

  def capture(cmd, directory, env, timeout, stop_event=None):
    del env, timeout, stop_event
    calls.append(directory)
    status = next(statuses)
    _write_vase_fixture(cmd, b"fixture raw content", success=True)
    return {"status": status, "returncode": 0 if status == "completed" else 1}

  monkeypatch.setattr(runner, "run_episode", capture)
  args = [
    str(SCRIPT),
    "--task",
    "vase-wipe",
    "--episodes",
    "2",
    "--output-dir",
    str(tmp_path),
    "--min-free-gb",
    "0",
    "--workers",
    "1",
    "--staging-root",
    str(tmp_path.with_name(tmp_path.name + "-staging")),
  ]
  monkeypatch.setattr(sys, "argv", args)
  assert runner.main() == 130
  original = tmp_path / "vase-wipe/000000/attempt_001/status.json"
  assert runner.read(original)["status"] == "interrupted"
  monkeypatch.setattr(sys, "argv", args + ["--resume"])
  assert runner.main() == 1
  assert len(calls) == 3
  assert runner.read(original)["status"] == "interrupted"
  assert runner.main() == 1
  assert len(calls) == 3  # Both success and failed terminal attempts are preserved.
  (tmp_path / "vase-wipe/000000/attempt_002/data/raw/episode.h5").write_bytes(
    b"corrupt"
  )
  with pytest.raises(RuntimeError, match="missing/corrupt"):
    runner.main()


def test_target_successes_collects_past_failures_and_resumes_with_larger_limit(
  monkeypatch, tmp_path
):
  monkeypatch.setattr(runner, "source_hashes", lambda: {"test": "target-v1"})
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *a, **kw: SimpleNamespace(returncode=0, stdout="OK", stderr=""),
  )
  outcomes = iter((False, True, False, True))
  calls = []

  def capture(cmd, directory, env, timeout, stop_event=None):
    del env, timeout, stop_event
    calls.append(directory)
    success = next(outcomes)
    _write_vase_fixture(
      cmd, f"episode-{len(calls)}".encode(), success=success
    )
    return {"status": "completed", "returncode": 0}

  monkeypatch.setattr(runner, "run_episode", capture)
  args = [
    str(SCRIPT),
    "--task",
    "vase-wipe",
    "--target-successes",
    "2",
    "--max-attempts",
    "2",
    "--output-dir",
    str(tmp_path),
    "--min-free-gb",
    "0",
    "--workers",
    "1",
    "--staging-root",
    str(tmp_path.with_name(tmp_path.name + "-staging")),
  ]
  monkeypatch.setattr(sys, "argv", args)
  assert runner.main() == 1
  first = runner.read(tmp_path / "summary.json")
  assert first["by_task"]["vase-wipe"]["success"] == 1
  assert first["target_met"] == {"vase-wipe": False}

  resumed = args.copy()
  resumed[resumed.index("2", resumed.index("--max-attempts"))] = "5"
  resumed.append("--resume")
  monkeypatch.setattr(sys, "argv", resumed)
  assert runner.main() == 0
  assert len(calls) == 4
  final = runner.read(tmp_path / "summary.json")
  assert final["by_task"]["vase-wipe"] == {
    "attempted": 4,
    "success": 2,
    "failed": 2,
    "timeout": 0,
  }
  assert final["target_met"] == {"vase-wipe": True}


def test_parallel_target_never_launches_more_than_remaining_successes(
  monkeypatch, tmp_path
):
  monkeypatch.setattr(runner, "source_hashes", lambda: {"test": "parallel-v1"})
  monkeypatch.setattr(
    subprocess,
    "run",
    lambda *a, **kw: SimpleNamespace(returncode=0, stdout="OK", stderr=""),
  )
  calls = []

  def capture(cmd, directory, env, timeout, stop_event=None):
    del env, timeout, stop_event
    calls.append(directory)
    _write_vase_fixture(cmd, b"fixture", success=True)
    return {"status": "completed", "returncode": 0}

  monkeypatch.setattr(runner, "run_episode", capture)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      str(SCRIPT),
      "--task",
      "vase-wipe",
      "--target-successes",
      "3",
      "--max-attempts",
      "20",
      "--workers",
      "8",
      "--output-dir",
      str(tmp_path),
      "--min-free-gb",
      "0",
      "--staging-root",
      str(tmp_path.with_name(tmp_path.name + "-staging")),
    ],
  )
  assert runner.main() == 0
  assert len(calls) == 3
  summary = runner.read(tmp_path / "summary.json")
  assert summary["by_task"]["vase-wipe"]["success"] == 3


def test_whiteboard_outcome_uses_actual_result_and_raw_file(tmp_path):
  (tmp_path / "result.json").write_text(json.dumps({"success": True}))
  assert not runner.outcome("whiteboard-wipe", tmp_path)
  raw = tmp_path / "raw"
  raw.mkdir()
  (raw / "episode.h5").write_bytes(b"fixture")
  (raw / "episode.json").write_text(json.dumps({"validation": {"valid": True}}))
  assert runner.outcome("whiteboard-wipe", tmp_path)
  (raw / "episode.json").write_text(json.dumps({"validation": {"valid": False}}))
  assert not runner.outcome("whiteboard-wipe", tmp_path)
  (tmp_path / "result.json").write_text(json.dumps({"success": False}))
  assert not runner.outcome("whiteboard-wipe", tmp_path)
