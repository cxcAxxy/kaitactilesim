from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import h5py
import pytest
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.recording import (
  TerminalStability,
  ValidationReport,
)
from kaihand_tactile_env.tasks.pick_place.task import PickPlaceResult

_SCRIPT_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/workcell/record_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("record_dataset_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
record_dataset = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = record_dataset
_SPEC.loader.exec_module(record_dataset)


def _jobs(tmp_path: Path, *, episodes: int = 3, workers: int = 2):
  args = record_dataset._parse_args(
    [
      "--output-dir",
      str(tmp_path),
      "--episodes",
      str(episodes),
      "--workers",
      str(workers),
      "--start-index",
      "5",
      "--seed",
      "40",
      "--object-xy-jitter",
      "0.02",
      "--no-cameras",
    ]
  )
  return args, record_dataset._build_jobs(args)


def _summary(job: record_dataset.RecordJob) -> record_dataset.EpisodeSummary:
  return record_dataset.EpisodeSummary(
    episode_index=job.episode_index,
    output=job.output,
    success=True,
    placed_in_box=True,
    state_samples=10,
    camera_samples={},
  )


def _overwrite_job(tmp_path: Path) -> record_dataset.RecordJob:
  args = record_dataset._parse_args(
    [
      "--output-dir",
      str(tmp_path),
      "--episodes",
      "1",
      "--no-cameras",
      "--overwrite",
    ]
  )
  return record_dataset._build_jobs(args)[0]


def test_parallel_jobs_have_stable_unique_indices_seeds_and_paths(tmp_path) -> None:
  _args, jobs = _jobs(tmp_path, episodes=4, workers=8)

  assert [job.episode_index for job in jobs] == [5, 6, 7, 8]
  assert [job.episode_seed for job in jobs] == [45, 46, 47, 48]
  assert [job.output.name for job in jobs] == [
    "episode_000005_cylinder_right.h5",
    "episode_000006_cylinder_right.h5",
    "episode_000007_cylinder_right.h5",
    "episode_000008_cylinder_right.h5",
  ]
  assert len({job.output for job in jobs}) == len(jobs)
  assert all(job.object_xy_jitter == pytest.approx(0.02) for job in jobs)
  assert all(not job.config.cameras for job in jobs)
  assert all(job.config.model_path == default_model_path("pick-place") for job in jobs)
  assert pickle.loads(pickle.dumps(jobs)) == jobs


def test_task_specific_default_output_directories_are_separate() -> None:
  pick_args = record_dataset._parse_args(["--scene", "pick-place"])
  poker_args = record_dataset._parse_args(["--scene", "poker-draw"])

  assert pick_args.output_dir == Path("datasets/pick_place")
  assert poker_args.output_dir == Path("datasets/poker_draw")
  assert poker_args.object_xy_jitter == poker_args.object_yaw_jitter == 0.0


def test_poker_jobs_record_the_force_setpoint(tmp_path) -> None:
  args = record_dataset._parse_args(
    [
      "--scene",
      "poker-draw",
      "--press-force",
      "0.25",
      "--no-cameras",
      "--output-dir",
      str(tmp_path),
    ]
  )
  job = record_dataset._build_jobs(args)[0]
  assert job.press_force_per_finger_n == 0.25
  assert record_dataset._press_force_for_job(job) == 0.25
  assert "per_finger_press_shear" in record_dataset.POKER_RECORDING_CONTRACT_VERSION


@pytest.mark.parametrize("force", ("nan", "inf", "0", "-1"))
def test_record_rejects_invalid_press_force(force) -> None:
  with pytest.raises(SystemExit):
    record_dataset._parse_args(["--scene", "poker-draw", "--press-force", force])


def test_record_rejects_press_force_for_pick_place() -> None:
  with pytest.raises(SystemExit):
    record_dataset._parse_args(["--scene", "pick-place", "--press-force", "0.25"])


def test_resume_and_overwrite_are_mutually_exclusive(tmp_path) -> None:
  with pytest.raises(SystemExit):
    record_dataset._parse_args(
      [
        "--output-dir",
        str(tmp_path),
        "--resume",
        "--overwrite",
      ]
    )


def test_resume_keeps_verified_complete_and_retries_other_indices(
  tmp_path, monkeypatch: Any
) -> None:
  complete = tmp_path / "episode_000005_cylinder_right.h5"
  complete.touch()
  complete.with_suffix(".json").write_text("{}", encoding="utf-8")
  failed = tmp_path / "episode_000006_cylinder_right.failure.json"
  failed.write_text("{}", encoding="utf-8")
  monkeypatch.setattr(
    record_dataset,
    "_completed_episode_is_valid",
    lambda output, _manifest, _job: output == complete,
  )
  args = record_dataset._parse_args(
    [
      "--output-dir",
      str(tmp_path),
      "--episodes",
      "3",
      "--start-index",
      "5",
      "--resume",
      "--no-cameras",
    ]
  )

  jobs = record_dataset._build_jobs(args)

  assert [job.episode_index for job in jobs] == [6, 7]
  assert all(job.overwrite for job in jobs)


def test_resume_never_overwrites_valid_but_incompatible_episode(
  tmp_path, monkeypatch: Any
) -> None:
  output = tmp_path / "episode_000005_cylinder_right.h5"
  output.write_bytes(b"preserve valid old episode")
  manifest = output.with_suffix(".json")
  manifest.write_text("{}", encoding="utf-8")
  checks: list[object] = []

  def compatible(_output: Path, _manifest: Path, job=None) -> bool:
    checks.append(job)
    return job is None

  monkeypatch.setattr(record_dataset, "_completed_episode_is_valid", compatible)
  args = record_dataset._parse_args(
    [
      "--output-dir",
      str(tmp_path),
      "--episodes",
      "1",
      "--start-index",
      "5",
      "--resume",
      "--no-cameras",
    ]
  )

  with pytest.raises(FileExistsError, match="Use a new --output-dir"):
    record_dataset._build_jobs(args)

  assert checks[0] is not None
  assert checks[1] is None
  assert output.read_bytes() == b"preserve valid old episode"
  assert manifest.exists()


def test_resume_verifies_manifest_digest(tmp_path, monkeypatch: Any) -> None:
  output = tmp_path / "episode.h5"
  output.write_bytes(b"episode payload")
  manifest = output.with_suffix(".json")
  manifest.write_text(
    json.dumps(
      {
        "episode": output.name,
        "sha256": record_dataset._sha256_file(output),
      }
    ),
    encoding="utf-8",
  )
  monkeypatch.setattr(
    record_dataset,
    "validate_episode",
    lambda _path: ValidationReport(True, (), (), 1, {}, 0.0),
  )

  assert record_dataset._completed_episode_is_valid(output, manifest)
  output.write_bytes(b"tampered payload")
  assert not record_dataset._completed_episode_is_valid(output, manifest)


def test_resume_rejects_changed_recursive_model_fingerprint(tmp_path) -> None:
  _args, jobs = _jobs(tmp_path, episodes=1, workers=1)
  job = jobs[0]
  metadata = {
    "recording_contract": record_dataset.RECORDING_CONTRACT_VERSION,
    "episode_index": job.episode_index,
    "seed": job.episode_seed,
    "scene": job.scene,
    "object": job.object_name,
    "side": job.side,
    "tactile_links": job.tactile_links,
    "object_xy_jitter": job.object_xy_jitter,
    "object_yaw_jitter": job.object_yaw_jitter,
    "overhead_pos": job.overhead_pos,
    "overhead_lookat": list(job.overhead_lookat),
    "overhead_fovy": job.overhead_fovy,
    "front_pos": job.front_pos,
    "front_lookat": list(job.front_lookat),
    "front_fovy": job.front_fovy,
    "model_layout": "task-isolated-v1",
    "active_objects": [job.object_name],
  }
  with h5py.File(job.output, "w") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["physics_hz"] = job.config.physics_hz
    file.attrs["control_hz"] = job.config.control_hz
    file.attrs["camera_hz"] = job.config.camera_hz
    file.attrs["tactile_source"] = job.config.tactile_provider
    file.attrs["model_sha256"] = record_dataset._sha256_file(job.config.model_path)
    file.attrs["model_fingerprint"] = model_fingerprint(job.config.model_path)

  assert record_dataset._episode_matches_job(job.output, job)
  with h5py.File(job.output, "r+") as file:
    file.attrs["model_fingerprint"] = "changed-shared-include"
  assert not record_dataset._episode_matches_job(job.output, job)


def test_poker_resume_requires_same_force_controller_and_new_stream(tmp_path) -> None:
  args = record_dataset._parse_args(
    [
      "--scene",
      "poker-draw",
      "--press-force",
      "0.25",
      "--no-cameras",
      "--output-dir",
      str(tmp_path),
    ]
  )
  job = record_dataset._build_jobs(args)[0]
  metadata = {
    "recording_contract": record_dataset.POKER_RECORDING_CONTRACT_VERSION,
    "episode_index": job.episode_index,
    "seed": job.episode_seed,
    "scene": job.scene,
    "object": job.object_name,
    "side": job.side,
    "object_xy_jitter": job.object_xy_jitter,
    "object_yaw_jitter": job.object_yaw_jitter,
    "model_layout": "task-isolated-v1",
    "active_objects": [job.object_name],
    "press_force_per_finger_n": 0.25,
    "press_control": record_dataset._press_control_for_job(job),
  }
  with h5py.File(job.output, "w") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
    for name in ("physics_hz", "control_hz", "camera_hz"):
      file.attrs[name] = getattr(job.config, name)
    file.attrs["tactile_source"] = job.config.tactile_provider
    file.attrs["model_sha256"] = record_dataset._sha256_file(job.config.model_path)
    file.attrs["model_fingerprint"] = model_fingerprint(job.config.model_path)
  assert not record_dataset._episode_matches_job(job.output, job)
  with h5py.File(job.output, "r+") as file:
    file.create_group("tactile_contact_force")
  assert record_dataset._episode_matches_job(job.output, job)
  metadata["press_force_per_finger_n"] = 0.35
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
  assert not record_dataset._episode_matches_job(job.output, job)
  metadata["press_force_per_finger_n"] = 0.25
  metadata["press_control"]["controller_source_sha256"]["task.py"] = (
    "changed-controller"
  )
  with h5py.File(job.output, "r+") as file:
    file.attrs["metadata_json"] = json.dumps(metadata)
  assert not record_dataset._episode_matches_job(job.output, job)


def test_existing_episode_requires_explicit_overwrite(tmp_path) -> None:
  args, jobs = _jobs(tmp_path, episodes=1)
  jobs[0].output.touch()

  with pytest.raises(FileExistsError, match="--overwrite"):
    record_dataset._build_jobs(args)

  overwrite_args = record_dataset._parse_args(
    [
      "--output-dir",
      str(tmp_path),
      "--episodes",
      "1",
      "--start-index",
      "5",
      "--no-cameras",
      "--overwrite",
    ]
  )
  assert record_dataset._build_jobs(overwrite_args)[0].overwrite


def test_episode_worker_uses_an_exclusive_output_lock(
  tmp_path,
  monkeypatch: Any,
) -> None:
  _args, jobs = _jobs(tmp_path, episodes=1)
  job = jobs[0]
  lock_path = job.output.with_suffix(job.output.suffix + ".lock")
  monkeypatch.setattr(
    record_dataset,
    "_record_claimed_episode",
    _summary,
  )

  assert record_dataset._record_episode(job) == _summary(job)
  assert not lock_path.exists()

  lock_path.touch()
  with pytest.raises(RuntimeError, match="already claimed"):
    record_dataset._record_episode(job)


def test_single_worker_stays_in_process(tmp_path, monkeypatch: Any) -> None:
  _args, jobs = _jobs(tmp_path, episodes=2, workers=1)
  calls = []

  def record(job: record_dataset.RecordJob) -> record_dataset.EpisodeSummary:
    calls.append(job.episode_index)
    return _summary(job)

  class BombExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
      raise AssertionError("single-worker recording must not spawn a process")

  monkeypatch.setattr(record_dataset, "_record_episode", record)
  monkeypatch.setattr(record_dataset, "ProcessPoolExecutor", BombExecutor)
  results = list(record_dataset._run_jobs(jobs, workers=1))

  assert calls == [5, 6]
  assert [result.episode_index for result in results] == [5, 6]


def test_single_worker_records_failure_and_continues_batch(
  tmp_path, monkeypatch: Any
) -> None:
  _args, jobs = _jobs(tmp_path, episodes=3, workers=1)

  def record(job: record_dataset.RecordJob):
    if job.episode_index == 6:
      raise RuntimeError("planned path to place is in collision")
    return _summary(job)

  monkeypatch.setattr(record_dataset, "_record_episode", record)
  results = list(record_dataset._run_jobs(jobs, workers=1))

  assert [result.episode_index for result in results] == [5, 6, 7]
  assert isinstance(results[1], record_dataset.EpisodeFailure)
  failure_path = jobs[1].output.with_suffix(".failure.json")
  payload = json.loads(failure_path.read_text(encoding="utf-8"))
  assert payload["completed"] is False
  assert payload["episode_seed"] == 46
  assert payload["error_type"] == "RuntimeError"
  assert "place" in payload["error_message"]


def test_fail_fast_preserves_original_exception(tmp_path, monkeypatch: Any) -> None:
  _args, jobs = _jobs(tmp_path, episodes=2, workers=1)

  def fail(_job: record_dataset.RecordJob) -> record_dataset.EpisodeSummary:
    raise RuntimeError("expected planning failure")

  monkeypatch.setattr(record_dataset, "_record_episode", fail)
  with pytest.raises(RuntimeError, match="expected planning failure"):
    list(record_dataset._run_jobs(jobs, workers=1, continue_on_error=False))
  assert not jobs[0].output.with_suffix(".failure.json").exists()


def test_overwrite_clears_stale_success_before_new_planning(
  tmp_path, monkeypatch: Any
) -> None:
  job = _overwrite_job(tmp_path)
  job.output.write_bytes(b"old h5")
  job.output.with_suffix(".json").write_text("{}", encoding="utf-8")

  class PlanningFailureSimulation:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
      assert not job.output.exists()
      assert not job.output.with_suffix(".json").exists()
      raise RuntimeError("new planning failed")

  monkeypatch.setattr(record_dataset, "ArmHandSimulation", PlanningFailureSimulation)
  with pytest.raises(RuntimeError, match="new planning failed"):
    record_dataset._record_claimed_episode(job)

  assert not job.output.exists()
  assert not job.output.with_suffix(".json").exists()


def test_unsuccessful_motion_remains_partial_and_is_not_published(
  tmp_path, monkeypatch: Any
) -> None:
  job = _overwrite_job(tmp_path)

  def failed_execute(executor: Any, plan: Any) -> PickPlaceResult:
    simulation = executor.sim
    return PickPlaceResult(
      success=False,
      object_name="cylinder",
      side=plan.side,
      final_object_pose=simulation.object_pose("cylinder"),
      box_center=simulation.model.body("box").pos.copy(),
      placed_in_box=False,
      phases=("return_home",),
    )

  monkeypatch.setattr(record_dataset.PickPlaceExecutor, "execute", failed_execute)
  monkeypatch.setattr(
    record_dataset,
    "wait_until_object_stable",
    lambda *_args, **_kwargs: TerminalStability(0.1, 0.1, 0.0, 0.0, 50),
  )
  monkeypatch.setattr(record_dataset, "cylinder_is_in_box", lambda *_args: False)

  with pytest.raises(RuntimeError, match="was not placed"):
    record_dataset._record_claimed_episode(job)

  assert job.output.with_suffix(job.output.suffix + ".partial").exists()
  assert not job.output.exists()
  assert not job.output.with_suffix(".json").exists()


def test_invalid_published_episode_is_demoted_back_to_partial(
  tmp_path, monkeypatch: Any
) -> None:
  job = _overwrite_job(tmp_path)

  def successful_execute(executor: Any, plan: Any) -> PickPlaceResult:
    simulation = executor.sim
    return PickPlaceResult(
      success=True,
      object_name="cylinder",
      side=plan.side,
      final_object_pose=simulation.object_pose("cylinder"),
      box_center=simulation.model.body("box").pos.copy(),
      placed_in_box=True,
      phases=("return_home",),
    )

  monkeypatch.setattr(record_dataset.PickPlaceExecutor, "execute", successful_execute)
  monkeypatch.setattr(
    record_dataset,
    "wait_until_object_stable",
    lambda *_args, **_kwargs: TerminalStability(0.1, 0.1, 0.0, 0.0, 50),
  )
  monkeypatch.setattr(record_dataset, "cylinder_is_in_box", lambda *_args: True)
  monkeypatch.setattr(
    record_dataset,
    "validate_episode",
    lambda _path: ValidationReport(
      False, ("synthetic validation failure",), (), 1, {}, 0.0
    ),
  )

  with pytest.raises(RuntimeError, match="failed validation"):
    record_dataset._record_claimed_episode(job)

  assert job.output.with_suffix(job.output.suffix + ".partial").exists()
  assert not job.output.exists()
  assert not job.output.with_suffix(".json").exists()


def test_parallel_scheduler_caps_workers_and_yields_stable_order(
  tmp_path,
  monkeypatch: Any,
) -> None:
  _args, jobs = _jobs(tmp_path, episodes=3, workers=8)

  class FakeFuture:
    def __init__(self, job: record_dataset.RecordJob) -> None:
      self.job = job
      self.cancelled = False

    def result(self) -> record_dataset.EpisodeSummary:
      return _summary(self.job)

    def cancel(self) -> None:
      self.cancelled = True

  class FakeExecutor:
    instance: FakeExecutor | None = None

    def __init__(self, *, max_workers: int, mp_context: Any) -> None:
      self.max_workers = max_workers
      self.start_method = mp_context.get_start_method()
      self.futures: list[FakeFuture] = []
      type(self).instance = self

    def __enter__(self) -> FakeExecutor:
      return self

    def __exit__(self, *_args: Any) -> None:
      pass

    def submit(self, function: Any, job: record_dataset.RecordJob) -> FakeFuture:
      assert function is record_dataset._record_episode
      future = FakeFuture(job)
      self.futures.append(future)
      return future

  monkeypatch.setattr(record_dataset, "ProcessPoolExecutor", FakeExecutor)
  monkeypatch.setattr(
    record_dataset,
    "as_completed",
    lambda futures: reversed(tuple(futures)),
  )

  results = list(record_dataset._run_jobs(jobs, workers=8))
  executor = FakeExecutor.instance
  assert executor is not None
  assert executor.max_workers == 3
  assert executor.start_method == "spawn"
  assert [future.job.episode_index for future in executor.futures] == [5, 6, 7]
  assert [result.episode_index for result in results] == [5, 6, 7]


def test_parallel_failure_does_not_block_later_ordered_results(
  tmp_path, monkeypatch: Any
) -> None:
  _args, jobs = _jobs(tmp_path, episodes=3, workers=2)

  class FakeFuture:
    def __init__(self, job: record_dataset.RecordJob) -> None:
      self.job = job

    def result(self) -> record_dataset.EpisodeSummary:
      if self.job.episode_index == 6:
        raise RuntimeError("synthetic worker failure")
      return _summary(self.job)

    def cancel(self) -> None:
      pass

  class FakeExecutor:
    def __init__(self, **_kwargs: Any) -> None:
      self.futures: list[FakeFuture] = []

    def __enter__(self) -> FakeExecutor:
      return self

    def __exit__(self, *_args: Any) -> None:
      pass

    def submit(self, _function: Any, job: record_dataset.RecordJob) -> FakeFuture:
      future = FakeFuture(job)
      self.futures.append(future)
      return future

  monkeypatch.setattr(record_dataset, "ProcessPoolExecutor", FakeExecutor)
  monkeypatch.setattr(
    record_dataset,
    "as_completed",
    lambda futures: reversed(tuple(futures)),
  )

  results = list(record_dataset._run_jobs(jobs, workers=2))

  assert [result.episode_index for result in results] == [5, 6, 7]
  assert isinstance(results[1], record_dataset.EpisodeFailure)
  assert jobs[1].output.with_suffix(".failure.json").exists()


@pytest.mark.parametrize("workers", (0, -1))
def test_run_jobs_rejects_non_positive_workers(workers: int) -> None:
  with pytest.raises(ValueError, match="workers must be positive"):
    list(record_dataset._run_jobs((), workers))
