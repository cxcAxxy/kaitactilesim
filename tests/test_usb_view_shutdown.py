from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/workcell/view_usb_insert.py"

# Run the real USB model, monitor and script in a fresh Python process. Only
# the native viewer is replaced, so these lifecycle checks need no display.
CHILD = r"""
import signal
import sys
import threading
import time
from runpy import run_path
from types import SimpleNamespace

import mujoco.viewer
import numpy as np
from kaihand_tactile_env.tasks.usb_insert.task import UsbInsertionMonitor

script_path, mode = sys.argv[1:]
main_thread = threading.get_ident()
measuring = False
measure_count = 0
viewer_closed = False
original_measure = UsbInsertionMonitor.measure

def prior_handler(signum, frame):
    print("PRIOR_HANDLER_CALLED", flush=True)

signal.signal(signal.SIGINT, prior_handler)

def await_sigint(marker):
    active_handler = signal.getsignal(signal.SIGINT)
    assert active_handler is not prior_handler, "handler restored before cleanup"
    received = False

    def observe(signum, frame):
        nonlocal received
        received = True
        active_handler(signum, frame)

    signal.signal(signal.SIGINT, observe)
    try:
        print(marker, flush=True)
        # Unlike pause(), this cannot miss a signal delivered immediately
        # after the marker is printed and before the child starts waiting.
        while not received:
            time.sleep(0.005)
    finally:
        signal.signal(signal.SIGINT, active_handler)

def measure(self):
    global measuring, measure_count
    measure_count += 1
    measuring = True
    if (mode == "sigint" and measure_count == 2) or (
        mode == "headless" and measure_count == 1
    ):
        await_sigint("MEASUREMENT_WAITING_FOR_SIGINT")
    result = original_measure(self)
    measuring = False
    if mode in ("sigint", "headless"):
        print("MEASUREMENT_COMPLETED", flush=True)
    return result

UsbInsertionMonitor.measure = measure

class Viewer:
    def __init__(self, callback):
        self.callback = callback
        self.opt = mujoco.MjvOption()
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.running = True
        self.key_sent = False

    def __enter__(self):
        print("VIEWER_ENTERED", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global viewer_closed
        assert threading.get_ident() == main_thread
        assert not measuring, "viewer closed before the measurement completed"
        assert exc_type is None, f"viewer exited through an exception: {exc_type}"
        if mode == "sigint":
            # A second real Ctrl+C during cleanup must remain a stop request.
            await_sigint("VIEWER_CLOSING")
        else:
            print("VIEWER_CLOSING", flush=True)
        self.running = False
        viewer_closed = True
        print("VIEWER_CLOSED", flush=True)

    def is_running(self):
        return self.running

    def sync(self):
        if mode in ("q", "escape") and not self.key_sent:
            self.key_sent = True
            key = ord("Q") if mode == "q" else 256
            thread = threading.Thread(target=self.callback, args=(key,))
            thread.start()
            thread.join()
            assert not viewer_closed, "key callback closed the viewer thread"
            print("KEY_CALLBACK_COMPLETED", flush=True)
        elif mode == "window_close":
            self.running = False

def launch(model, data, *, key_callback):
    assert mode != "headless", "headless mode attempted to open a window"
    if mode == "launch_failure":
        raise RuntimeError("synthetic window initialization failure")
    assert model.body("usb_plug").id >= 0
    assert len(data.qpos) == model.nq
    return Viewer(key_callback)

mujoco.viewer.launch_passive = launch
script = run_path(script_path)
sys.argv = [script_path]
if mode == "headless":
    sys.argv.extend(["--headless", "--duration", "100000"])
try:
    script["main"]()
except RuntimeError as error:
    assert mode == "launch_failure"
    assert str(error) == "synthetic window initialization failure"
    print("LAUNCH_ERROR_PRESERVED", flush=True)
else:
    assert mode != "launch_failure"
assert signal.getsignal(signal.SIGINT) is prior_handler
assert viewer_closed == (mode not in ("headless", "launch_failure"))
print("HANDLER_RESTORED", flush=True)
signal.raise_signal(signal.SIGINT)
print("PROCESS_FINISHED", flush=True)
"""


def _wait_for_output(
  process: subprocess.Popen, output: Path, marker: str, *, timeout: float = 10.0
) -> None:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    text = output.read_text()
    if marker in text:
      return
    if process.poll() is not None:
      _, stderr = process.communicate()
      pytest.fail(f"child exited before {marker}:\n{text}\n{stderr}")
    time.sleep(0.01)
  pytest.fail(f"child did not reach {marker}:\n{output.read_text()}")


def _run_child(tmp_path: Path, mode: str) -> str:
  output = tmp_path / f"{mode}.stdout"
  with output.open("w") as stdout:
    process = subprocess.Popen(
      [sys.executable, "-u", "-c", CHILD, str(SCRIPT), mode],
      stdout=stdout,
      stderr=subprocess.PIPE,
      text=True,
    )
    try:
      if mode in ("sigint", "headless"):
        _wait_for_output(process, output, "MEASUREMENT_WAITING_FOR_SIGINT")
        process.send_signal(signal.SIGINT)
      if mode == "sigint":
        _wait_for_output(process, output, "VIEWER_CLOSING")
        process.send_signal(signal.SIGINT)
      _, stderr = process.communicate(timeout=15)
      assert process.returncode == 0, stderr
    finally:
      if process.poll() is None:
        process.kill()
        process.communicate(timeout=5)
  text = output.read_text()
  assert "Traceback" not in text + stderr
  assert "KeyboardInterrupt" not in text + stderr
  assert "HANDLER_RESTORED" in text
  assert "PRIOR_HANDLER_CALLED" in text
  assert "PROCESS_FINISHED" in text
  return text


def test_sigint_finishes_measurement_and_closes_viewer_before_restoring_handler(
  tmp_path,
):
  text = _run_child(tmp_path, "sigint")
  measurement = text.index("MEASUREMENT_COMPLETED", text.index("VIEWER_ENTERED"))
  assert measurement < text.index("VIEWER_CLOSING")
  assert text.index("VIEWER_CLOSING") < text.index("VIEWER_CLOSED")
  assert text.index("VIEWER_CLOSED") < text.index("HANDLER_RESTORED")
  assert "USB scene stopped." in text


@pytest.mark.parametrize("key", ("q", "escape"))
def test_viewer_exit_keys_request_main_thread_cleanup(tmp_path, key):
  text = _run_child(tmp_path, key)
  assert text.index("KEY_CALLBACK_COMPLETED") < text.index("VIEWER_CLOSING")
  assert "USB scene stopped." in text


def test_headless_sigint_exits_without_opening_viewer(tmp_path):
  text = _run_child(tmp_path, "headless")
  assert "MEASUREMENT_COMPLETED" in text
  assert "VIEWER_ENTERED" not in text
  assert "USB scene stopped." in text


def test_normal_window_close_preserves_report_and_restores_handler(tmp_path):
  text = _run_child(tmp_path, "window_close")
  assert "VIEWER_CLOSED" in text
  assert '"finite_state": true' in text


def test_window_creation_failure_restores_handler_and_preserves_error(tmp_path):
  text = _run_child(tmp_path, "launch_failure")
  assert "LAUNCH_ERROR_PRESERVED" in text
  assert "USB scene stopped." not in text
