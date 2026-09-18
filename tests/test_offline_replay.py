from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.offline_replay import (
  compose_multimodal_replay_frame,
  read_tactile_sample,
  resolve_replay_cameras,
  resolve_tactile_source,
  select_replay_frames,
)


def _strings(group: h5py.Group, name: str, values: tuple[str, ...]) -> None:
  group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))


def _base_episode(path: Path) -> None:
  with h5py.File(path, "w") as file:
    file.attrs["metadata_json"] = json.dumps({"scene": "pick-place"})
    file.attrs["camera_hz"] = 30
    state = file.create_group("state")
    state.create_dataset("timestamp", data=(0.0, 0.01, 0.02, 0.03))
    commands = file.create_group("commands")
    _strings(commands, "phase", ("a", "b", "c", "d"))
    cameras = file.create_group("cameras")
    for camera_index, name in enumerate(("right_wrist", "head", "left_wrist")):
      camera = cameras.create_group(name)
      camera.create_dataset("timestamp", data=(0.0, 0.01, 0.02, 0.03))
      camera.create_dataset("state_index", data=(0, 1, 2, 3))
      camera.create_dataset(
        "rgb",
        data=np.full((4, 12, 16, 3), camera_index * 20, dtype=np.uint8),
      )


def test_camera_resolution_and_frame_selection(tmp_path: Path) -> None:
  episode = tmp_path / "episode.h5"
  _base_episode(episode)
  with h5py.File(episode, "r") as file:
    assert resolve_replay_cameras(file, None) == (
      "head",
      "left_wrist",
      "right_wrist",
    )
    assert resolve_replay_cameras(file, ("right_wrist", "head")) == (
      "right_wrist",
      "head",
    )
    with pytest.raises(ValueError, match="no saved RGB"):
      resolve_replay_cameras(file, ("overhead",))
  np.testing.assert_array_equal(
    select_replay_frames(np.array((0.0, 0.01, 0.02, 0.03)), 10), (0, 3)
  )


def test_spatial_force_fills_missing_hand_with_zero(tmp_path: Path) -> None:
  episode = tmp_path / "spatial.h5"
  _base_episode(episode)
  right_names = tuple(FINGERTIP_LINK_NAMES[5:])
  with h5py.File(episode, "a") as file:
    group = file.create_group("tactile_contact_force")
    _strings(group, "link_names", right_names)
    group.create_dataset("timestamp", data=(0.0, 0.01, 0.02, 0.03))
    normal = np.ones((4, 5, 7, 5), dtype=np.float64)
    tangent = np.zeros((4, 5, 7, 5, 2), dtype=np.float64)
    tangent[..., 0] = 2.0
    tangent[..., 1] = -3.0
    group.create_dataset("normal_taxel_force_n", data=normal)
    group.create_dataset("tangent_taxel_force_n", data=tangent)
  with h5py.File(episode, "r") as file:
    source = resolve_tactile_source(file)
    first, second, curves = read_tactile_sample(file, source, 2)
  assert source.kind == "spatial_force"
  np.testing.assert_array_equal(first[:5], 0.0)
  np.testing.assert_array_equal(second[:5], 0.0)
  np.testing.assert_array_equal(curves[:5], 0.0)
  np.testing.assert_allclose(first[5:], 1.0)
  np.testing.assert_allclose(second[5:], np.hypot(2.0, 3.0))
  np.testing.assert_allclose(curves[5:, 0], 70.0)
  np.testing.assert_allclose(curves[5:, 1], -105.0)
  np.testing.assert_allclose(curves[5:, 2], 35.0)


def test_genesis_probe_depth_and_proxy_are_not_labeled_newtons(tmp_path: Path) -> None:
  episode = tmp_path / "genesis.h5"
  _base_episode(episode)
  with h5py.File(episode, "a") as file:
    group = file.create_group("tactile_genesis")
    _strings(group, "link_names", tuple(FINGERTIP_LINK_NAMES))
    probe_names = tuple(
      name for name in FINGERTIP_LINK_NAMES for _ in range(35)
    )
    _strings(group, "probe_link_names", probe_names)
    depth = np.full((4, 350), 0.0015, dtype=np.float64)
    contact = np.ones((4, 350), dtype=np.bool_)
    proxy = np.arange(30, dtype=np.float64).reshape(1, 10, 3).repeat(4, axis=0)
    group.create_dataset("probe_depth", data=depth)
    group.create_dataset("probe_contact_instantaneous", data=contact)
    group.create_dataset("force_local", data=proxy)
  with h5py.File(episode, "r") as file:
    source = resolve_tactile_source(file)
    first, second, curves = read_tactile_sample(file, source, 1)
  assert source.kind == "genesis_probe"
  assert source.first_unit == "mm"
  assert "not N" in source.curve_unit
  np.testing.assert_allclose(first, 1.5)
  np.testing.assert_array_equal(second, 1.0)
  np.testing.assert_array_equal(curves, proxy[1])


def test_multimodal_composer_accepts_three_cameras_and_ten_fingers() -> None:
  cameras = tuple(
    (name, np.full((120, 160, 3), value, dtype=np.uint8))
    for name, value in (
      ("head", 20),
      ("left_wrist", 60),
      ("right_wrist", 100),
    )
  )
  image = compose_multimodal_replay_frame(
    cameras,
    np.zeros((10, 7, 5)),
    np.zeros((10, 7, 5)),
    np.zeros((2, 10, 3)),
    width=960,
    height=540,
    timestamp_s=0.1,
    phase="test",
    heading="TEST",
    first_label="Fn",
    second_label="|Ft|",
    first_unit="N/taxel",
    second_unit="N/taxel",
    curve_unit="N per fingertip",
    curve_channels=("Fx", "Fy", "Fz"),
    first_maximum=0.1,
    second_maximum=0.02,
    curve_maximum=1.0,
  )
  assert image.mode == "RGB"
  assert image.size == (960, 540)


def test_complete_multimodal_export_writes_video_and_audit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  episode = tmp_path / "source.h5"
  _base_episode(episode)
  with h5py.File(episode, "a") as file:
    group = file.create_group("tactile_contact_force")
    _strings(group, "link_names", tuple(FINGERTIP_LINK_NAMES))
    group.create_dataset("timestamp", data=(0.0, 0.01, 0.02, 0.03))
    group.create_dataset(
      "normal_taxel_force_n", data=np.zeros((4, 10, 7, 5), dtype=np.float64)
    )
    group.create_dataset(
      "tangent_taxel_force_n", data=np.zeros((4, 10, 7, 5, 2), dtype=np.float64)
    )

  try:
    __import__("mujoco")
  except ImportError:
    # The offline exporter itself does not use MuJoCo.  task_video owns the
    # shared ffmpeg pipe and imports MuJoCo for its separate live renderer.
    monkeypatch.setitem(sys.modules, "mujoco", ModuleType("mujoco"))
  from kaihand_tactile_env.shared.offline_replay import export_multimodal_replay

  output = tmp_path / "review"
  report = export_multimodal_replay(
    episode,
    output,
    fps=10,
    width=960,
    height=540,
  )
  assert report["task"] == "pick-place"
  assert report["camera_names"] == ["head", "left_wrist", "right_wrist"]
  assert report["output_frame_count"] == 2
  assert (output / "review.mp4").stat().st_size > 0
  assert (output / "first_frame.png").is_file()
  assert (output / "last_frame.png").is_file()
  assert len((output / "frames.jsonl").read_text().splitlines()) == 2
  assert json.loads((output / "replay.json").read_text())["completed"] is True
