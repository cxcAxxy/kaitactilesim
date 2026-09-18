"""USB queue limits, cancellation and seed ownership without simulation."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
from kaihand_tactile_env.tasks.usb_insert import batch, recording


@pytest.fixture
def jobs(tmp_path):
  return [
    recording.UsbRecordJob(tmp_path / f"usb_{index:06d}.h5", index, 42, worker_count=2)
    for index in range(5)
  ]


@pytest.fixture
def pool(monkeypatch):
  state = SimpleNamespace(submitted=[], shutdown=None, stop=None, fail=False)

  class Pool:
    def __init__(self, *, max_workers, mp_context, initializer, initargs):
      assert max_workers == 2
      assert mp_context.get_start_method() == "spawn"
      assert initializer is batch._initialize_worker
      state.stop = initargs[0]

    def submit(self, function, job):
      assert function is batch._record_job
      state.submitted.append(job.episode_index)
      future = Future()
      if state.fail:
        future.set_exception(RuntimeError("worker failed"))
      else:
        future.set_result({"status": "success", "episode_index": job.episode_index})
      return future

    def shutdown(self, **kwargs):
      state.shutdown = kwargs

  monkeypatch.setattr(batch, "ProcessPoolExecutor", Pool)
  return state


def test_two_workers_submit_only_two_until_results_are_consumed(jobs, pool):
  reports = batch.run_jobs(jobs, workers=2)
  first = next(reports)
  assert pool.submitted == [0, 1]
  results = [first, *reports]
  assert sorted(row["episode_index"] for row in results) == list(range(5))
  assert pool.submitted == list(range(5))
  assert pool.shutdown == {"wait": True, "cancel_futures": True}


def test_cancel_drains_active_reports_without_starting_next_job(jobs, pool):
  stop = [False]
  reports = batch.run_jobs(jobs, 2, lambda: stop[0])
  first = next(reports)
  stop[0] = True
  results = [first, *reports]
  assert pool.submitted == [0, 1]
  assert sorted(row["episode_index"] for row in results) == [0, 1]
  assert pool.stop.is_set()
  assert pool.shutdown["wait"]


def test_close_generator_signals_workers_and_waits_for_cleanup(jobs, pool):
  reports = batch.run_jobs(jobs, 2)
  next(reports)
  reports.close()
  assert pool.stop.is_set()
  assert pool.submitted == [0, 1]
  assert pool.shutdown["wait"]


def test_worker_crash_retains_failure_report_and_stops_submission(jobs, pool):
  pool.fail = True
  reports = list(batch.run_jobs(jobs, 2))
  assert pool.submitted == [0, 1]
  assert len(reports) == 2
  for report in reports:
    assert report["status"] == "exception" and not report["success"]
    assert "worker failed" in report["failure_reason"]
    job = jobs[report["episode_index"]]
    assert job.output.with_suffix(".worker_failure.json").is_file()
    assert not job.output.exists()


def test_serial_preserves_episode_identity_and_stop_callback(jobs, monkeypatch):
  called = []

  def record(job, should_stop):
    assert not should_stop()
    called.append(job)
    return {"episode_index": job.episode_index}

  monkeypatch.setattr(recording, "record_episode", record)
  assert list(batch.run_jobs(jobs, 1, lambda: len(called) >= 2)) == [
    {"episode_index": 0},
    {"episode_index": 1},
  ]
  assert called == jobs[:2]
  for job in called:
    serial = replace(job, worker_count=1)
    assert recording.episode_seeds(job.root_seed, job.episode_index) == (
      recording.episode_seeds(serial.root_seed, serial.episode_index)
    )


@pytest.mark.parametrize("workers", [0, 3, -1, 2.5, True])
def test_invalid_worker_count_is_rejected(jobs, workers):
  with pytest.raises(ValueError):
    list(batch.run_jobs(jobs, workers))


def test_repeated_output_is_rejected_before_launch(jobs):
  with pytest.raises(ValueError, match="distinct"):
    list(batch.run_jobs([jobs[0], jobs[0]], 2))


def test_cancelled_before_start_launches_nothing(jobs, pool):
  assert list(batch.run_jobs(jobs, 2, lambda: True)) == []
  assert pool.submitted == []


def test_environment_bounds_worker_threads(monkeypatch):
  for name in batch.THREAD_VARIABLES:
    monkeypatch.setenv(name, "16")
  batch.configure_environment()
  import os

  assert all(os.environ[name] == "1" for name in batch.THREAD_VARIABLES)
