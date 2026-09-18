"""Lossless USB write batching, including mutable MuJoCo-like arrays and tails."""

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.recording import (
  _append,
  _append_many,
  _image_stream,
  _stream,
)
from kaihand_tactile_env.tasks.usb_insert.buffered_h5 import BufferedH5File


def layout(file):
  state = file.create_group("state")
  _stream(state, "timestamp", (), np.float64)
  _stream(state, "qpos", (3,), np.float64)
  _stream(state, "phase", (), h5py.string_dtype("utf-8"))
  _stream(state, "event_start", (), np.int64)
  _stream(state, "event_count", (), np.int32)
  _stream(state, "contact", (3,), np.bool_)
  _stream(file.create_group("events"), "wrench", (6,), np.float64)
  camera = file.create_group("cameras/head")
  _image_stream(camera, "rgb", (4, 5, 3), np.uint8)
  file.attrs["clock"] = "post_step_forward_v1"


def test_independent_source_audit_covers_the_new_storage_implementation():
  from kaihand_tactile_env.shared.tict_source_audit import _known_usb_source_paths
  from kaihand_tactile_env.tasks.usb_insert.recording import SOURCE_PATHS

  known, _ = _known_usb_source_paths()
  assert set(known) == set(SOURCE_PATHS)
  assert "src/kaihand_tactile_env/tasks/usb_insert/buffered_h5.py" in known


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 263])
def test_exact_samples_match_unbuffered_with_aliases_ragged_events_and_terminal(
  tmp_path, rows
):
  paths = [tmp_path / "reference.h5", tmp_path / "buffered.h5"]
  files = [h5py.File(path, "w") for path in paths]
  for file in files:
    layout(file)
  files[1] = BufferedH5File(files[1])
  state_vector = np.zeros(3)
  for index in range(rows):
    state_vector[:] = [index, -index, np.nan]
    count = index % 7
    wrench = np.full((count, 6), index, dtype=float)
    for file in files:
      state = file["state"]
      _append(state["timestamp"], index * 0.002)
      _append(state["qpos"], state_vector)
      _append(state["phase"], "抓取" if index % 2 else "approach")
      _append(state["contact"], [True, False, bool(index % 2)])
      # Direct path and group path must share one logical buffered length.
      _append(state["event_start"], file["events/wrench"].shape[0])
      _append(state["event_count"], count)
      _append_many(file["events"]["wrench"], wrench)
      if index % 50 == 0:
        _append(file["cameras/head/rgb"], np.full((4, 5, 3), index % 256, np.uint8))
    # Mutating producer-owned arrays after append must not alter saved samples.
    state_vector[:] = 9999
    wrench[:] = 9999
  assert files[1]["state/qpos"] is files[1]["state"]["qpos"]
  for file in files:
    file["state/phase"][-1] = "terminal_settle"
    np.testing.assert_array_equal(file["state/timestamp"][:], np.arange(rows) * 0.002)
    file.close()  # also the graceful-cancellation path; no explicit flush
  assert files[1].statistics()["pending_rows"] == 0
  with h5py.File(paths[0]) as reference, h5py.File(paths[1]) as actual:
    names = []
    reference.visititems(
      lambda name, value: (
        names.append(name) if isinstance(value, h5py.Dataset) else None
      )
    )
    assert actual.attrs["clock"] == reference.attrs["clock"]
    for name in names:
      assert actual[name].dtype == reference[name].dtype
      assert actual[name].chunks == reference[name].chunks
      np.testing.assert_array_equal(actual[name][:], reference[name][:])


def test_periodic_flush_commits_all_streams_and_small_tail(tmp_path):
  raw = h5py.File(tmp_path / "cancelled.h5.partial", "w")
  layout(raw)
  file = BufferedH5File(raw, rows=128)
  for index in range(1001):
    _append(file["state/qpos"], [index] * 3)
    _append(file["state/timestamp"], index * 0.002)
    assert file["state/qpos"]._count < 128
  assert raw["state/qpos"].shape == (896, 3)
  file.flush()
  assert raw["state/qpos"].shape == (1001, 3)
  assert raw["state/timestamp"].shape == (1001,)
  assert file.statistics()["batch_writes"] == 16  # 8 writes per stream, not 1001
  assert file.statistics()["pending_rows"] == 0
  _append(file["state/qpos"], [1001] * 3)
  _append(file["state/timestamp"], 2.002)
  file.close()
  with h5py.File(tmp_path / "cancelled.h5.partial") as saved:
    assert saved["state/qpos"].shape == (1002, 3)
    assert saved["state/qpos"][-1].tolist() == [1001] * 3


def test_camera_pixels_are_written_immediately_and_new_static_groups_work(tmp_path):
  raw = h5py.File(tmp_path / "camera.h5", "w")
  layout(raw)
  file = BufferedH5File(raw)
  pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
  _append(file["cameras/head/rgb"], pixels)
  np.testing.assert_array_equal(raw["cameras/head/rgb"][0], pixels)
  trace = file.create_group("precontact_noise")
  trace.create_dataset("offset", data=np.zeros((2, 3)))
  assert "offset" in trace
  assert file["precontact_noise/offset"].shape == (2, 3)
  file.close()


def test_partial_reserved_write_is_rejected_instead_of_saving_uninitialized_rows(
  tmp_path,
):
  raw = h5py.File(tmp_path / "broken.h5.partial", "w")
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
