"""Tiny HDF5-only equivalence checks; no robot, physics or rendering."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import contact_tactile
from kaihand_tactile_env.shared.buffered_h5 import BufferedH5File
from kaihand_tactile_env.shared.config import WorkcellConfig
from kaihand_tactile_env.shared.recording import (
  EpisodeRecorder,
  _append,
  _append_many,
  _image_stream,
  _stream,
)
from kaihand_tactile_env.tasks.usb_insert.buffered_h5 import (
  BufferedH5File as UsbBufferedH5File,
)


def layout(file):
  state = file.create_group("state")
  _stream(state, "timestamp", (), np.float64)
  _stream(state, "qpos", (3,), np.float64)
  commands = file.create_group("commands")
  _stream(commands, "phase", (), h5py.string_dtype("utf-8"))
  contacts = file.create_group("contacts")
  _stream(contacts, "frame_start", (), np.int64)
  _stream(contacts, "frame_count", (), np.int32)
  events = contacts.create_group("events")
  _stream(events, "wrench", (6,), np.float64)
  force = file.create_group("tactile_contact_force")
  _stream(force, "timestamp", (), np.float64)
  _stream(force, "normal_force_n", (5,), np.float64)
  _stream(force, "tangent_force_n", (5, 2), np.float64)
  camera = file.create_group("cameras/head")
  _stream(camera, "state_index", (), np.int64)
  _stream(camera, "pose_timestamp", (), np.float64)
  _image_stream(camera, "rgb", (4, 5, 3), np.uint8)
  file.attrs["pose_clock"] = "cached pre-integration, 2 ms before state"


@pytest.mark.parametrize("rows", [1, 63, 64, 65, 127, 128, 129, 263])
def test_matches_raw_and_usb_adapter_with_mutable_arrays_ragged_and_terminal(
  tmp_path, rows
):
  paths = [tmp_path / f"{name}.h5" for name in ("raw", "shared", "usb")]
  files = [h5py.File(path, "w") for path in paths]
  for file in files:
    layout(file)
  files[1] = BufferedH5File(files[1], rows=64)
  files[2] = UsbBufferedH5File(files[2], rows=64)
  vector = np.zeros(3)
  force = np.zeros(5)
  tangent = np.zeros((5, 2))
  for index in range(rows):
    vector[:] = [index, -index, np.nan]
    force[:] = index * 0.01
    tangent[:] = [index * 0.03, -index * 0.02]
    count = index % 7 if index != 64 else 130
    wrench = np.full((count, 6), index, dtype=float)
    for file in files:
      _append(file["state/timestamp"], index * 0.01)
      _append(file["state/qpos"], vector)
      _append(file["commands/phase"], "摸牌" if index % 2 else "slide_card")
      _append(file["contacts/frame_start"], file["contacts/events/wrench"].shape[0])
      _append(file["contacts/frame_count"], count)
      _append_many(file["contacts/events/wrench"], wrench)
      _append(file["tactile_contact_force/timestamp"], max(0, index * 0.01 - 0.002))
      _append(file["tactile_contact_force/normal_force_n"], force)
      _append(file["tactile_contact_force/tangent_force_n"], tangent)
      if index % 3 == 0:
        _append(file["cameras/head/state_index"], index)
        _append(file["cameras/head/pose_timestamp"], max(0, index * 0.01 - 0.002))
        _append(file["cameras/head/rgb"], np.full((4, 5, 3), index % 256, np.uint8))
    vector[:] = force[:] = tangent[:] = 9999
    wrench[:] = 9999
  for file in files:
    file["commands/phase"][-1] = "terminal_settle"
    file["cameras/head/state_index"][-1] = rows - 1
    np.testing.assert_array_equal(file["state/timestamp"][:], np.arange(rows) * 0.01)
    assert file["contacts/events/wrench"].shape[0] == sum(
      index % 7 if index != 64 else 130 for index in range(rows)
    )
    file.close()
  assert files[1].statistics()["pending_rows"] == 0
  with h5py.File(paths[0]) as expected:
    names = []
    expected.visititems(
      lambda n, d: names.append(n) if isinstance(d, h5py.Dataset) else None
    )
    for path in paths[1:]:
      with h5py.File(path) as actual:
        assert dict(actual.attrs) == dict(expected.attrs)
        for name in names:
          assert actual[name].dtype == expected[name].dtype
          assert actual[name].chunks == expected[name].chunks
          assert actual[name].compression == expected[name].compression
          np.testing.assert_array_equal(actual[name][:], expected[name][:])


def test_flush_and_close_drain_64_row_batches_and_small_tail(tmp_path):
  raw = h5py.File(tmp_path / "cancelled.h5.partial", "w")
  layout(raw)
  file = BufferedH5File(raw)
  assert file.statistics()["buffer_rows"] == 64
  assert file["state/qpos"] is file["state"]["qpos"]
  for index in range(131):
    _append(file["state/qpos"], [index] * 3)
    _append(file["state/timestamp"], index * 0.01)
    assert file["state/qpos"]._count < 64
  assert raw["state/qpos"].shape == (128, 3)
  file.flush()
  assert raw["state/qpos"].shape == (131, 3)
  assert file.statistics()["pending_rows"] == 0
  assert file.statistics()["batch_writes"] == 6
  _append(file["state/qpos"], [131] * 3)
  _append(file["state/timestamp"], 1.31)
  file.close()
  with h5py.File(tmp_path / "cancelled.h5.partial") as saved:
    assert saved["state/qpos"].shape == (132, 3)
    assert saved["state/qpos"][-1].tolist() == [131] * 3


def test_camera_streams_and_late_precontact_archive_remain_immediate(tmp_path):
  raw = h5py.File(tmp_path / "head.h5", "w")
  layout(raw)
  file = BufferedH5File(raw)
  pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
  _append(file["cameras/head/rgb"], pixels)
  _append(file["cameras/head/pose_timestamp"], 0.098)
  np.testing.assert_array_equal(raw["cameras/head/rgb"][0], pixels)
  assert raw["cameras/head/pose_timestamp"][0] == 0.098
  # Match EpisodeRecorder.write_precontact_noise_trace's actual require_group API.
  group = file.require_group("control").create_group("precontact_noise")
  group.attrs["metadata_json"] = json.dumps({"noise_off_after_contact": True})
  values = np.zeros((100, 7))
  group.create_dataset(
    "noise_offset_rad", data=values, compression="gzip", shuffle=True
  )
  values[:] = 123
  assert np.max(abs(raw["control/precontact_noise/noise_offset_rad"][:])) == 0
  file.close()


def test_incomplete_reservation_fails_closed_not_zero_filled(tmp_path):
  raw = h5py.File(tmp_path / "interrupted.h5.partial", "w")
  layout(raw)
  file = BufferedH5File(raw)
  stream = file["state/qpos"]
  stream.resize(2, axis=0)
  with pytest.raises(ValueError, match="exactly"):
    stream[0] = [0, 0, 0]
  with pytest.raises(RuntimeError, match="incomplete"):
    file.flush()
  with pytest.raises(RuntimeError, match="incomplete"):
    file.close()
  assert not raw.id.valid


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_buffer_size_rejected(tmp_path, rows):
  with h5py.File(tmp_path / "invalid.h5", "w") as file:
    with pytest.raises(ValueError, match="positive integer"):
      BufferedH5File(file, rows)


@pytest.mark.parametrize("interrupted", [False, True])
def test_real_episode_recorder_poker_close_paths_flush_pending_tail(
  tmp_path, monkeypatch, interrupted
):
  # Reuse the existing tiny model and fake sensors; no robot assets or renderer.
  # Only initial FK is evaluated by the fixture: this test never calls mj_step.
  from test_taskspace_recording import _Contact, _Proxy, _tiny_sim

  sim = _tiny_sim(tmp_path)
  sim.scene = "poker-draw"
  sim.genesis_probe_layout = None
  sim.object_names = sim.model_object_names = ("card",)
  sim.joint_names = ("joint_l", "joint_r")
  sim._qvel_address = {"joint_l": 0, "joint_r": 1}
  sim.command_state = lambda: (np.ones(14), np.ones(2), sim.joint_names)
  sim.joint_state = lambda: (sim.joint_names, sim.data.qpos[:2], sim.data.qvel[:2])
  sim.object_pose = lambda _: np.array([0, 0, 0.2, 1, 0, 0, 0])
  sim.object_twist = lambda _: np.zeros(6)
  sim.drive_limit_n = None
  sim.drive_state = lambda: {"drive_requested_fx_n": 0.0, "drive_actual_fx_n": 0.0}
  sim.timestep = 0.002
  monkeypatch.setattr(contact_tactile, "SolverDistributedTactileProvider", _Contact)
  monkeypatch.setenv("KAIHAND_POKER_HDF5_BUFFER_ROWS", "64")
  config = WorkcellConfig(
    model_path=sim.model_path, cameras=(), tactile_provider=_Proxy.source
  )
  output = tmp_path / "episode.h5"
  holder = SimpleNamespace(recorder=None)

  def write_rows():
    with EpisodeRecorder(
      output, sim, config, tactile_provider=_Proxy(), capture_taskspace=True
    ) as recorder:
      holder.recorder = recorder
      assert isinstance(recorder._file, BufferedH5File)
      recorder.record_initial()
      for index in range(1, 5):
        sim.data.time = index * 0.01
        sim.observation_time = index * 0.01 - 0.002
        sim.data.qpos[0] = index * 0.1
        recorder._record_state("slide_card")
      assert recorder._file.statistics()["pending_rows"] > 0
      # Mutating live arrays after their copy must not alter a pending tail.
      sim.data.qpos[0] = 1234
      recorder.set_outcome({"success": not interrupted})
      if interrupted:
        raise KeyboardInterrupt("synthetic stop after a completed frame")
      recorder.record_terminal()

  if interrupted:
    with pytest.raises(KeyboardInterrupt, match="synthetic stop"):
      write_rows()
    path = output.with_suffix(".h5.partial")
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
  else:
    write_rows()
    path = output
    assert output.with_suffix(".json").exists()
    assert not output.with_suffix(".h5.partial").exists()
  assert holder.recorder._file.statistics()["pending_rows"] == 0
  with h5py.File(path) as saved:
    assert saved.attrs["hdf5_buffer_rows"] == 64
    stats = json.loads(saved.attrs["hdf5_buffer_statistics_json"])
    assert stats["pending_rows"] == 0
    assert stats["buffer_rows"] == 64
    np.testing.assert_array_equal(saved["state/timestamp"][:], np.arange(5) * 0.01)
    np.testing.assert_allclose(saved["state/qpos"][:, 0], np.arange(5) * 0.1)
    np.testing.assert_array_equal(
      saved["tactile_contact_force/timestamp"][:],
      np.r_[0.0, np.arange(1, 5) * 0.01 - 0.002],
    )
    assert saved["commands/phase"].asstr()[-1] == (
      "slide_card" if interrupted else "terminal_settle"
    )
    assert json.loads(saved.attrs["outcome_json"])["success"] is not interrupted
