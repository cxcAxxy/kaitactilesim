import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from kaihand_tactile_env.pipeline.deployment import _stop_owned_process

ROOT = Path(__file__).resolve().parents[1]


def wait_for_file(path, timeout=10):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if path.is_file():
      return
    time.sleep(0.02)
  raise AssertionError(f"child did not publish readiness: {path}")


def is_running(pid):
  # Orphaned, already-killed descendants can remain zombies until init reaps
  # them. They no longer run or own model resources.
  try:
    state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
  except FileNotFoundError:
    return False
  return state != "Z"


def wait_until_stopped(pid, timeout=3):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline and is_running(pid):
    time.sleep(0.02)
  assert not is_running(pid), f"owned model process remains running: {pid}"


def test_owned_group_cleanup_kills_stubborn_descendant_after_leader_exit(tmp_path):
  pid_file = tmp_path / "descendant.pid"
  grandchild = tmp_path / "grandchild.py"
  grandchild.write_text(
    "import os, signal, sys, time\nfrom pathlib import Path\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(60)\n"
  )
  leader = tmp_path / "leader.py"
  leader.write_text(
    "import subprocess, sys, time\nfrom pathlib import Path\n"
    "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
    "while not Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
  )
  process = subprocess.Popen(
    [sys.executable, str(leader), str(grandchild), str(pid_file)],
    start_new_session=True,
  )
  try:
    assert process.wait(timeout=10) == 0
    wait_for_file(pid_file)
    descendant_pid = int(pid_file.read_text())
    assert is_running(descendant_pid)
    _stop_owned_process(process, timeout=0.1)
    wait_until_stopped(descendant_pid)
  finally:
    # This process group was created above by this test, never discovered by
    # a broad process search or selected from an external service endpoint.
    try:
      os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
    if process.poll() is None:
      process.wait(timeout=5)


def test_foreground_serve_sigterm_reaps_owned_server(tmp_path):
  pid_file = tmp_path / "server.pid"
  server = tmp_path / "server.py"
  server.write_text(
    "import os, sys, time\nfrom pathlib import Path\n"
    "Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(60)\n"
  )
  supervisor = tmp_path / "supervisor.py"
  supervisor.write_text(
    "import os, sys\nfrom pathlib import Path\n"
    "from kaihand_tactile_env.pipeline.deployment import "
    "DeploymentInterrupted, ProcessPlan, execute_serve\n"
    "plan = ProcessPlan((sys.executable, sys.argv[1], sys.argv[2]), "
    "Path.cwd(), dict(os.environ))\n"
    "try:\n  sys.exit(execute_serve(plan))\n"
    "except DeploymentInterrupted as error:\n  sys.exit(128 + error.signum)\n"
  )
  environment = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), os.environ.get("PYTHONPATH", ""))),
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
  }
  process = subprocess.Popen(
    [sys.executable, str(supervisor), str(server), str(pid_file)],
    cwd=tmp_path,
    env=environment,
    start_new_session=True,
  )
  server_pid = None
  try:
    wait_for_file(pid_file)
    server_pid = int(pid_file.read_text())
    assert is_running(server_pid)
    process.terminate()
    assert process.wait(timeout=10) == 128 + signal.SIGTERM
    wait_until_stopped(server_pid)
  finally:
    if process.poll() is None:
      os.killpg(process.pid, signal.SIGKILL)
      process.wait(timeout=5)
    if server_pid is not None and is_running(server_pid):
      os.kill(server_pid, signal.SIGKILL)
