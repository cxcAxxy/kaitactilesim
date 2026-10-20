"""Check conservative old-job reservations and four/six-worker scheduling."""

from __future__ import annotations

import errno
import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/deploy"


@pytest.fixture
def module(monkeypatch):
  monkeypatch.syspath_prepend(str(SCRIPTS))
  spec = importlib.util.spec_from_file_location(
    "new_tasks_queue_under_test", SCRIPTS / "run_pi05_new_tasks_queue.py"
  )
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def dump(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value))


@pytest.mark.parametrize("occupied", [False, True])
def test_port_probe_sets_reuseaddr_before_bind_and_fails_closed(
  module, monkeypatch, occupied
):
  events = []

  class ProbeSocket:
    def __enter__(self):
      return self

    def __exit__(self, *_args):
      events.append("closed")

    def setsockopt(self, level, option, value):
      events.append(("reuse", level, option, value))

    def bind(self, address):
      events.append(("bind", address))
      if occupied:
        raise OSError(errno.EADDRINUSE, "real listener occupies port")

  monkeypatch.setattr(
    module,
    "socket",
    SimpleNamespace(
      socket=ProbeSocket,
      SOL_SOCKET=socket.SOL_SOCKET,
      SO_REUSEADDR=socket.SO_REUSEADDR,
    ),
  )
  assert module.port_free(18800) is not occupied
  assert events == [
    ("reuse", socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
    ("bind", ("127.0.0.1", 18800)),
    "closed",
  ]


@pytest.fixture
def local_socket():
  try:
    server = socket.socket()
  except PermissionError:
    pytest.skip("real local socket regression requires unsandboxed Linux execution")
  try:
    yield server
  finally:
    server.close()


@pytest.mark.parametrize("reuse", [False, True])
def test_real_port_probe_rejects_live_listener_even_with_reuseaddr(
  module, local_socket, reuse
):
  local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse))
  local_socket.bind(("127.0.0.1", 0))
  local_socket.listen(1)
  assert module.port_free(local_socket.getsockname()[1]) is False


def test_real_port_probe_rejects_exclusively_bound_port(module, local_socket):
  local_socket.bind(("127.0.0.1", 0))
  assert module.port_free(local_socket.getsockname()[1]) is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux TCP TIME_WAIT regression")
def test_real_port_probe_accepts_closed_server_in_time_wait(module, local_socket):
  local_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  local_socket.bind(("127.0.0.1", 0))
  port = local_socket.getsockname()[1]
  local_socket.listen(1)
  with socket.socket() as client:
    client.settimeout(2)
    client.connect(("127.0.0.1", port))
    connection, _ = local_socket.accept()
    # Make the server the active closer so its listening port enters TIME_WAIT.
    connection.close()
    assert client.recv(1) == b""
  local_socket.close()
  rows = [
    line.split()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]
  ]
  matching = [row for row in rows if int(row[1].split(":")[1], 16) == port]
  assert matching and all(row[3] == "06" for row in matching)
  with socket.socket() as legacy_probe:
    with pytest.raises(OSError) as error:
      legacy_probe.bind(("127.0.0.1", port))
    assert error.value.errno == errno.EADDRINUSE
  assert module.port_free(port) is True


@pytest.fixture
def old_job(module, tmp_path, monkeypatch):
  root = tmp_path / "old"
  groups = [{}, {}]
  for index, (task, step) in enumerate(
    (task, step) for task in ("usb-unplug", "bulb-unscrew") for step in (20000, 24999)
  ):
    output = root / task / f"step{step}_N20_video20"
    group = {
      "task": task,
      "step": step,
      "worker_pid": 8000 + index,
      "status": "complete",
      "output_dir": str(output),
    }
    groups.append(group)
    detail = {
      "pid": 8000 + index,
      "status": "complete",
      "evaluation_returncode": 0,
      "managed_server_stopped": True,
      "evaluation_process_stopped": True,
    }
    dump(output / "group_status.json", detail)
    dump(output / "run/run_manifest.json", detail)
  old = {
    "status": "running",
    "stage": "parallel_evaluations",
    "groups": groups,
    "handoff": {
      "managed_server_stopped": True,
      "evaluation_process_stopped": True,
      "no_remaining_group_started_by_serial": True,
    },
  }
  dump(root / "status.json", old)
  monkeypatch.setattr(module, "process_info", lambda pid: None)
  monkeypatch.setattr(module, "port_free", lambda port: True)
  return SimpleNamespace(root=root, status=old, groups=groups[2:])


@pytest.mark.parametrize(
  "stage",
  [
    "waiting_for_whiteboard",
    "cleaning_up_serial_handoff",
    "handoff_or_parallel_error",
    None,
  ],
)
def test_unconfirmed_handoff_reserves_total_capacity(module, old_job, stage):
  old_job.status["stage"] = stage
  dump(old_job.root / "status.json", old_job.status)
  assert module.old_reservations(old_job.root)[0] == module.CAPACITY


def test_complete_old_groups_release_capacity_only_with_cleanup(module, old_job):
  reserved, proof = module.old_reservations(old_job.root)
  assert reserved == 0
  assert all(group["released"] for group in proof["groups"])
  for index, group in enumerate(old_job.groups):
    group["status"] = "running"
    dump(old_job.root / "status.json", old_job.status)
    assert module.old_reservations(old_job.root)[0] == index + 1


@pytest.mark.parametrize(
  "missing",
  [
    "managed_server_stopped",
    "evaluation_process_stopped",
    "no_remaining_group_started_by_serial",
  ],
)
def test_missing_handoff_proof_reserves_all(module, old_job, missing):
  del old_job.status["handoff"][missing]
  dump(old_job.root / "status.json", old_job.status)
  assert module.old_reservations(old_job.root)[0] == module.CAPACITY


def test_live_worker_or_reused_pid_does_not_release_slot(module, old_job, monkeypatch):
  monkeypatch.setattr(
    module,
    "process_info",
    lambda pid: (
      {"state": "S", "starttime": 99, "ppid": 1, "exit_status": 0}
      if pid == 8000
      else None
    ),
  )
  assert module.old_reservations(old_job.root)[0] == 1


def test_zombie_success_is_already_exited_but_failure_is_not_release_proof(
  module, old_job, monkeypatch
):
  monkeypatch.setattr(
    module,
    "process_info",
    lambda pid: {"state": "Z", "starttime": 99, "ppid": 1, "exit_status": 0},
  )
  assert module.old_reservations(old_job.root)[0] == 0
  monkeypatch.setattr(
    module,
    "process_info",
    lambda pid: {"state": "Z", "starttime": 99, "ppid": 1, "exit_status": 256},
  )
  assert module.old_reservations(old_job.root)[0] == 4


def test_live_listener_keeps_slot_reserved(module, old_job, monkeypatch):
  monkeypatch.setattr(module, "port_free", lambda port: port != 18785)
  assert module.old_reservations(old_job.root)[0] == 1


@pytest.mark.parametrize(
  "kind", ["worker_cleanup", "run_cleanup", "wrong_worker_identity"]
)
def test_status_complete_alone_is_not_cleanup_proof(module, old_job, kind):
  output = Path(old_job.groups[0]["output_dir"])
  path = output / (
    "run/run_manifest.json" if kind == "run_cleanup" else "group_status.json"
  )
  detail = json.loads(path.read_text())
  detail["pid" if kind == "wrong_worker_identity" else "managed_server_stopped"] = (
    999 if kind == "wrong_worker_identity" else False
  )
  dump(path, detail)
  assert module.old_reservations(old_job.root)[0] == 1


def test_failed_unknown_and_missing_old_jobs_never_look_idle(module, old_job, tmp_path):
  old_job.status["status"] = "failed"
  dump(old_job.root / "status.json", old_job.status)
  assert module.old_reservations(old_job.root)[0] == module.CAPACITY
  assert module.old_reservations(tmp_path / "missing")[0] == module.CAPACITY


def test_confirmed_four_running_groups_reserve_four_not_total_six(module, old_job):
  assert module.CAPACITY == 6
  assert module.EXISTING_GROUP_COUNT == 4
  for group in old_job.groups:
    group["status"] = "running"
  dump(old_job.root / "status.json", old_job.status)
  reserved, evidence = module.old_reservations(old_job.root)
  assert reserved == 4
  assert module.CAPACITY - reserved == 2
  assert evidence["reserved_slots"] == 4


def test_exact_six_checkpoints_and_usb_repaired_run(module):
  groups = module.worker.NEW_GROUPS
  assert [(task, step) for task, step, _ in groups] == [
    (task, step)
    for task in ("install-ram", "poker-draw", "usb-insert")
    for step in (20000, 24999)
  ]
  assert all(
    "hand_tail_v1" in str(checkpoint)
    for task, _, checkpoint in groups
    if task == "usb-insert"
  )


@pytest.mark.parametrize("rows_key", ["results", "trials"])
def test_progress_accepts_shared_and_native_batch_rows(module, tmp_path, rows_key):
  output = tmp_path / "group"
  dump(
    output / "run/evaluation/summary.json",
    {rows_key: [{"seed": 0}, {"seed": 1}], "valid_trials": 2, "successes": 1},
  )
  status = {"groups": [{"output_dir": str(output), "status": "running"}]}
  module.refresh(status, tmp_path)
  assert status["completed_trials"] == 2
  assert status["valid_trials"] == 2
  assert status["groups"][0]["diagnostic_successes"] == 1


@pytest.mark.parametrize("capacity", [4, 6])
def test_queue_automatically_fills_released_capacity_without_touching_old_job(
  module, tmp_path, monkeypatch, capacity
):
  root = tmp_path / "new"
  root.mkdir()
  simulation = tmp_path / "snapshot/kaitactilesim"
  openpi = tmp_path / "snapshot/openpi"
  state = SimpleNamespace(tick=0, launches=[], live={}, old_reserved=4)
  monkeypatch.setattr(module, "CAPACITY", capacity)
  monkeypatch.setattr(module.worker, "EVALUATION_PARENT", tmp_path)
  monkeypatch.setattr(module, "validate_configuration", lambda *args: None)
  monkeypatch.setattr(module.signal, "signal", lambda *args: None)

  def reservation(_root):
    state.old_reserved = max(0, 4 - state.tick)
    return state.old_reserved, {"reserved_slots": state.old_reserved}

  monkeypatch.setattr(module, "old_reservations", reservation)

  class FakeProcess:
    def __init__(self, index):
      self.pid = index + 9000
      self.index = index
      self.launched = state.tick

    def poll(self):
      return 0 if state.tick >= self.launched + 3 else None

  def launch(group, *_args):
    process = FakeProcess(group["group_index"])
    state.live[process.index] = process
    group.update(status="running", worker_pid=process.pid)
    state.launches.append((state.tick, process.index))
    assert (
      sum(process.poll() is None for process in state.live.values())
      + state.old_reserved
      <= capacity
    )
    return process

  monkeypatch.setattr(module, "launch_group", launch)

  def finalize(group, code):
    assert code == 0
    group.update(status="complete", worker_returncode=0, owned_cleanup_confirmed=True)
    del state.live[group["group_index"]]
    return True

  monkeypatch.setattr(module, "finalize_worker", finalize)

  def sleep(seconds):
    assert seconds == 15
    state.tick += 1
    assert state.tick < 20

  monkeypatch.setattr(module.time, "sleep", sleep)
  assert module.run(root, simulation, openpi) == 0
  assert state.launches[0] == (1 if capacity == 4 else 0, 0)
  if capacity == 6:
    assert state.launches[:2] == [(0, 0), (0, 1)]
    assert all(tick > 0 for tick, _ in state.launches[2:])
  assert len(state.launches) == 6
  assert {index for _, index in state.launches} == set(range(6))
  status = json.loads((root / "status.json").read_text())
  assert status["status"] == "complete"
  assert status["completed_checkpoint_count"] == 6
  assert status["settings"]["formal_metrics_valid"] is False
  assert status["settings"]["execute_steps"] == "horizon"
  assert status["settings"]["guard_mode"] == "observe"
  assert status["settings"]["total_checkpoint_worker_cap"] == capacity


@pytest.mark.parametrize("stage", ["preflight", "prepare", "verify", "evaluate"])
def test_failed_worker_releases_only_proven_clean_capacity(
  module, tmp_path, monkeypatch, stage
):
  output = tmp_path / "group"
  dump(output / "group_status.json", {"stage": stage, "status": "failed"})
  monkeypatch.setattr(module, "port_free", lambda port: True)
  group = {"output_dir": str(output), "port": 18800}
  assert module.finalize_worker(group, 1) is (stage != "evaluate")
  assert group["status"] == "failed"


def test_environment_uses_frozen_roots_and_gpu_egl(module, tmp_path):
  sim = tmp_path / "frozen/kaitactilesim"
  op = tmp_path / "frozen/openpi"
  env = module.environment(tmp_path / "output", sim, op)
  assert env["PYTHONPATH"].split(":") == [
    str(sim / "src"),
    str(op / "src"),
    str(op / "packages/openpi-client/src"),
  ]
  assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == "0"
  assert env["MUJOCO_GL"] == "egl"
  assert env["LIBGL_ALWAYS_SOFTWARE"] == "0"
  assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
  assert "JAX_PLATFORMS" not in env
