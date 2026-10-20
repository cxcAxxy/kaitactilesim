from __future__ import annotations

import gc
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKCELL = ROOT / "scripts/workcell"
BINDING_PATH = WORKCELL / "tactile_policy_binding.py"

MOCK_WORKER = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import sys
import time

mode = os.environ["TACTILE_MOCK_MODE"]
marker = Path(os.environ["TACTILE_MOCK_MARKER"])

def terminated(_signum, _frame):
    marker.write_text("SIGTERM\n", encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, terminated)
if mode == "eof":
    raise SystemExit(0)
if mode == "hang_no_ready":
    while True:
        time.sleep(1)

print(json.dumps({"ready": {"backend": "mock", "mode": mode}}), flush=True)
if mode == "ready_eof":
    raise SystemExit(0)
if mode == "hang_after_ready":
    while True:
        time.sleep(1)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("close"):
        marker.write_text("CLOSE\n", encoding="utf-8")
        raise SystemExit(0)
    print(json.dumps({"echo": request, "finite": True}), flush=True)
'''


def _load_binding():
  sys.path.insert(0, str(ROOT / "src"))
  sys.path.insert(0, str(WORKCELL))
  try:
    spec = importlib.util.spec_from_file_location(
      "tactile_policy_binding_lifecycle_test", BINDING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(WORKCELL))
    sys.path.remove(str(ROOT / "src"))


class _FastWaitProcess:
  """Delegate to a real Popen while shortening the worker's five-second grace."""

  def __init__(self, process):
    self._process = process

  def __getattr__(self, name):
    return getattr(self._process, name)

  def wait(self, timeout=None):
    if timeout is not None:
      timeout = min(timeout, 0.5)
    return self._process.wait(timeout=timeout)


def _install_real_mock_process(monkeypatch, binding, script, mode, marker):
  real_popen = subprocess.Popen
  processes = []

  def launch(_model_command, **kwargs):
    environment = dict(kwargs["env"])
    environment.update(
      TACTILE_MOCK_MODE=mode,
      TACTILE_MOCK_MARKER=str(marker),
      PYTHONDONTWRITEBYTECODE="1",
    )
    kwargs["env"] = environment
    process = real_popen([sys.executable, "-B", str(script)], **kwargs)
    processes.append(process)
    return _FastWaitProcess(process)

  monkeypatch.setattr(binding.subprocess, "Popen", launch)
  return processes


def _worker_inputs(tmp_path):
  project = tmp_path / "model-project"
  project.mkdir()
  manifest_path = tmp_path / "deployment.json"
  manifest_path.write_text("{}\n", encoding="utf-8")
  output = tmp_path / "trial"
  output.mkdir()
  manifest = {
    "runtime": {
      "project_root": str(project),
      "python": sys.executable,
    },
  }
  return manifest_path, manifest, output


def _write_mock(tmp_path):
  script = tmp_path / "mock_tactile_worker.py"
  script.write_text(MOCK_WORKER, encoding="utf-8")
  return script


def _assert_reaped(process):
  assert process.poll() is not None
  with pytest.raises(ProcessLookupError):
    os.kill(process.pid, 0)


def _open_fd_targets():
  targets = {}
  for path in Path("/proc/self/fd").iterdir():
    if not path.name.isdigit():
      continue
    try:
      targets[int(path.name)] = os.readlink(path)
    except FileNotFoundError:
      # The directory iterator's own descriptor can disappear before readlink.
      pass
  return targets


def test_real_subprocess_ready_infer_close_and_idempotent_close(tmp_path, monkeypatch):
  binding = _load_binding()
  script = _write_mock(tmp_path)
  marker = tmp_path / "closed.txt"
  processes = _install_real_mock_process(
    monkeypatch, binding, script, "normal", marker
  )
  manifest_path, manifest, output = _worker_inputs(tmp_path)

  worker = binding.TactileWorker(manifest_path, manifest, output, timeout=2)
  assert worker.get_server_metadata() == {"backend": "mock", "mode": "normal"}
  request = {"request_index": 3, "state": [0.0, 1.0]}
  assert worker.infer(request) == {"echo": request, "finite": True}
  worker.close()
  worker.close()

  assert marker.read_text() == "CLOSE\n"
  assert worker.process.stdin.closed
  assert worker.process.stdout.closed
  assert worker.log.closed
  assert len(processes) == 1
  _assert_reaped(processes[0])


def test_startup_eof_is_reported_and_process_is_reaped(tmp_path, monkeypatch):
  binding = _load_binding()
  script = _write_mock(tmp_path)
  marker = tmp_path / "unused.txt"
  processes = _install_real_mock_process(monkeypatch, binding, script, "eof", marker)
  manifest_path, manifest, output = _worker_inputs(tmp_path)

  with pytest.raises(RuntimeError, match="model worker exited"):
    binding.TactileWorker(manifest_path, manifest, output, timeout=2)

  assert len(processes) == 1
  _assert_reaped(processes[0])


def test_infer_eof_is_reported_and_close_remains_safe(tmp_path, monkeypatch):
  binding = _load_binding()
  script = _write_mock(tmp_path)
  marker = tmp_path / "unused.txt"
  processes = _install_real_mock_process(
    monkeypatch, binding, script, "ready_eof", marker
  )
  manifest_path, manifest, output = _worker_inputs(tmp_path)
  worker = binding.TactileWorker(manifest_path, manifest, output, timeout=2)

  with pytest.raises((BrokenPipeError, RuntimeError)):
    worker.infer({"request_index": 0})
  worker.close()
  worker.close()

  assert len(processes) == 1
  _assert_reaped(processes[0])


def test_startup_timeout_terminates_real_process_group(tmp_path, monkeypatch):
  binding = _load_binding()
  script = _write_mock(tmp_path)
  marker = tmp_path / "terminated.txt"
  processes = _install_real_mock_process(
    monkeypatch, binding, script, "hang_no_ready", marker
  )
  manifest_path, manifest, output = _worker_inputs(tmp_path)

  with pytest.raises(TimeoutError, match="model worker timed out"):
    binding.TactileWorker(manifest_path, manifest, output, timeout=0.05)

  assert marker.read_text() == "SIGTERM\n"
  assert len(processes) == 1
  _assert_reaped(processes[0])


def test_close_escalates_unresponsive_worker_to_sigterm_and_is_idempotent(
  tmp_path, monkeypatch
):
  binding = _load_binding()
  script = _write_mock(tmp_path)
  marker = tmp_path / "terminated.txt"
  processes = _install_real_mock_process(
    monkeypatch, binding, script, "hang_after_ready", marker
  )
  manifest_path, manifest, output = _worker_inputs(tmp_path)
  worker = binding.TactileWorker(manifest_path, manifest, output, timeout=2)

  worker.close()
  worker.close()

  assert marker.read_text() == "SIGTERM\n"
  assert len(processes) == 1
  _assert_reaped(processes[0])


def test_nonexistent_runtime_python_fails_without_starting_worker(tmp_path):
  binding = _load_binding()
  manifest_path, manifest, output = _worker_inputs(tmp_path)
  missing = tmp_path / "missing-python"
  manifest["runtime"]["python"] = str(missing)
  gc.collect()
  before = _open_fd_targets()

  with pytest.raises(FileNotFoundError):
    binding.TactileWorker(manifest_path, manifest, output, timeout=0.05)
  gc.collect()

  after = _open_fd_targets()
  # A failed exec has no child to reap and must not retain the newly opened log.
  assert set(after) <= set(before)
  log_path = str((output / "model_worker.log").resolve())
  assert all(not target.startswith(log_path) for target in after.values())
  assert Path(log_path).is_file()


def test_mock_processes_are_not_left_running_after_suite_guard():
  """Keep a cheap PID-namespace sanity check explicit in the test inventory."""
  started = time.monotonic()
  process = subprocess.Popen(
    [sys.executable, "-B", "-c", "pass"],
    start_new_session=True,
  )
  assert process.wait(timeout=2) == 0
  _assert_reaped(process)
  assert time.monotonic() - started < 2
