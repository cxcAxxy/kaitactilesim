from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_browser():
  path = ROOT / "scripts/workcell/episode_browser.py"
  spec = importlib.util.spec_from_file_location("episode_browser_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


browser = _load_browser()
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(request):
  return LOCAL_OPENER.open(request, timeout=5)


def _episode(root: Path, identifier: int, *, late_right: bool = False) -> Path:
  timestamps = np.asarray(
    [1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000, 1_400_000_000],
    dtype=np.int64,
  )
  path = root / f"episode_{identifier}.hdf5"
  with h5py.File(path, "w") as file:
    file.attrs["fps"] = 10
    file.attrs["task"] = "pick up the cup"
    file.attrs["collect_time_utc"] = "2026-09-24T10:10:02+00:00"
    file.attrs["vector"] = np.asarray([1, 2], dtype=np.int16)
    file.create_dataset("timestamp_ros_nsec", data=timestamps)
    observations = file.create_group("observations")
    observations.create_dataset("qpos", data=np.zeros((5, 26), dtype=np.float32))
    tactile = file.create_group("raw/revo2_touch")
    for side, offset in (("left", 0.0), ("right", 1000.0)):
      group = tactile.create_group(side)
      stamps = np.asarray(
        [
          1_050_000_000 if late_right and side == "right" else 950_000_000,
          1_150_000_000,
          1_350_000_000,
        ],
        dtype=np.int64,
      )
      data = np.zeros((3, 5, 5), dtype=np.float32)
      for sample in range(3):
        for finger in range(5):
          data[sample, finger] = [
            0.0,
            offset + sample * 100 + finger * 10 + 1,
            offset + sample * 100 + finger * 10 + 2,
            65535.0,
            offset + sample * 100 + finger * 10 + 4,
          ]
      group.create_dataset("stamp_nsec", data=stamps)
      group.create_dataset("data", data=data)
  for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
    directory = root / "video" / camera
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"episode_{identifier}.mp4").write_bytes(b"0123456789abcdef")
  return path


def _json_request(url: str, *, method: str = "GET", payload=None):
  data = None if payload is None else json.dumps(payload).encode("utf-8")
  request = urllib.request.Request(
    url,
    data=data,
    method=method,
    headers={"Content-Type": "application/json"} if data is not None else {},
  )
  with _urlopen(request) as response:
    return response.status, response.headers, json.loads(response.read())


@contextmanager
def _running_server(root: Path, static: Path):
  server = browser.create_server(root, port=0, static_root=static, cache_root=root / "playback-cache")
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  host, port = server.server_address
  try:
    yield f"http://{host}:{port}"
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_dataset_scan_is_strict_numeric_and_has_playback_summary(tmp_path: Path) -> None:
  _episode(tmp_path, 10)
  _episode(tmp_path, 2)
  (tmp_path / "episode_1.h5").write_bytes(b"ignored")
  (tmp_path / "episode_3.hdf5.bak").write_bytes(b"ignored")
  (tmp_path / "prefix_episode_4.hdf5").write_bytes(b"ignored")
  (tmp_path / "episode_bad.hdf5").write_bytes(b"ignored")

  result = browser.EpisodeStore(tmp_path).dataset()

  assert result["count"] == 2
  assert result["is_expected_count"] is False
  assert [episode["id"] for episode in result["episodes"]] == [2, 10]
  assert result["cameras"] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
  episode = result["episodes"][0]
  assert episode["frame_count"] == 5
  assert episode["fps"] == 10
  assert episode["duration_s"] == pytest.approx(0.5)
  assert episode["task"] == "pick up the cup"
  assert episode["collect_time_utc"].startswith("2026-09-24")
  assert episode["size_bytes"] == episode["total_size_bytes"]
  assert episode["size_bytes"] > episode["hdf5_size_bytes"]
  assert set(episode["videos"]) == set(result["cameras"])


def test_h264_cache_invalidates_when_source_changes_and_denies_deleted_episode(tmp_path: Path, monkeypatch) -> None:
  _episode(tmp_path, 0)
  store = browser.EpisodeStore(tmp_path)
  cache = browser.PlaybackCache(store, tmp_path / "cache")
  calls = []

  def convert(source, target):
    calls.append(source)
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"cached-video")

  monkeypatch.setattr(cache, "_convert", convert)
  try:
    cache.prepare("0")
    cache.executor.shutdown(wait=True)
    result = cache.prepare("0")
    assert result["state"] == "ready"
    assert len(calls) == 3
    assert all(url.endswith("/h264") for url in result["videos"].values())
    stream, length = cache.open_media("0", "cam_high")
    with stream:
      assert stream.read() == b"cached-video"
    assert length == 12
    source = tmp_path / "video/cam_high/episode_0.mp4"
    source.write_bytes(b"replacement source is different")
    with pytest.raises(browser.ApiError) as error:
      cache.open_media("0", "cam_high")
    assert error.value.code == "playback_not_ready"
    store.delete("0", "episode_0")
    with pytest.raises(browser.ApiError) as error:
      cache.open_media("0", "cam_left_wrist")
    assert error.value.code == "episode_not_found"
  finally:
    cache.close()


def test_real_h264_conversion_preserves_video_and_range(tmp_path: Path) -> None:
  if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("ffmpeg and ffprobe required for actual media conversion")
  _episode(tmp_path, 4)
  source = tmp_path / "video/cam_high/episode_4.mp4"
  subprocess.run([
    "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
    "-frames:v", "5", "-c:v", "mpeg4", str(source),
  ], check=True, capture_output=True)
  for camera in ("cam_left_wrist", "cam_right_wrist"):
    shutil.copyfile(source, tmp_path / "video" / camera / source.name)
  original = source.read_bytes()
  with _running_server(tmp_path, ROOT / "scripts/workcell/episode_browser_web") as base:
    for _ in range(100):
      _, _, result = _json_request(f"{base}/api/episodes/4/playback")
      if result["state"] == "ready":
        break
      time.sleep(0.1)
    assert result["state"] == "ready"
    url = base + result["videos"]["cam_high"]
    with _urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-63"})) as response:
      assert response.status == 206
      assert len(response.read()) == 64
    with _urlopen(url) as response:
      downloaded = tmp_path / "downloaded.mp4"
      downloaded.write_bytes(response.read())
    probe = subprocess.run([
      "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
      "stream=codec_name,pix_fmt,nb_frames,avg_frame_rate", "-of", "json", str(downloaded),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {"codec_name": "h264", "pix_fmt": "yuv420p", "avg_frame_rate": "10/1", "nb_frames": "5"}
  assert source.read_bytes() == original


def test_detail_converts_attributes_and_lists_schema(tmp_path: Path) -> None:
  _episode(tmp_path, 7)

  detail = browser.EpisodeStore(tmp_path).detail("7")

  assert detail["attrs"]["vector"] == [1, 2]
  schema = {entry["path"]: entry for entry in detail["schema"]}
  assert schema["raw/revo2_touch"]["kind"] == "group"
  assert schema["raw/revo2_touch/left/data"] == {
    "path": "raw/revo2_touch/left/data",
    "kind": "dataset",
    "attrs": {},
    "shape": [3, 5, 5],
    "dtype": "float32",
  }


def test_telemetry_uses_video_clock_and_latest_nonfuture_samples(tmp_path: Path) -> None:
  _episode(tmp_path, 2, late_right=True)

  telemetry = browser.EpisodeStore(tmp_path).telemetry("2", 5)

  assert [frame["video_frame_index"] for frame in telemetry["frames"]] == [0, 2, 4]
  assert telemetry["times"] == pytest.approx([0.0, 0.2, 0.4])
  assert telemetry["timestamp_times_s"] == pytest.approx([0.0, 0.2, 0.4])
  assert telemetry["frames"][1]["left"]["sample_index"] == 1
  assert telemetry["frames"][1]["left"]["stamp_nsec"] == 1_150_000_000
  assert telemetry["frames"][1]["left"]["normal"] == [101, 111, 121, 131, 141]
  assert telemetry["frames"][0]["right"]["sample_index"] is None
  assert telemetry["right"][0] == [[None, None, None]] * 5
  assert telemetry["left"][1][2] == [121, 122, 124]
  assert telemetry["channels"] == ["normal", "tangential", "proximity"]
  assert all(scale >= 1 for scale in telemetry["scales"].values())


@pytest.mark.parametrize("fps", [0, 4, 6, 30])
def test_telemetry_rejects_unsupported_fps(tmp_path: Path, fps: int) -> None:
  _episode(tmp_path, 0)

  with pytest.raises(browser.ApiError) as caught:
    browser.EpisodeStore(tmp_path).telemetry("0", fps)

  assert caught.value.status == 400
  assert caught.value.code == "invalid_fps"


def test_http_serves_static_json_and_single_byte_ranges(tmp_path: Path) -> None:
  _episode(tmp_path, 3)
  static = tmp_path / "web"
  static.mkdir()
  (static / "index.html").write_text("<h1>browser</h1>", encoding="utf-8")
  (static / "app.js").write_text("", encoding="utf-8")
  (static / "styles.css").write_text("", encoding="utf-8")

  with _running_server(tmp_path, static) as base:
    with _urlopen(f"{base}/") as response:
      assert response.read() == b"<h1>browser</h1>"
    status, _, dataset = _json_request(f"{base}/api/dataset")
    assert status == 200
    assert dataset["episodes"][0]["id"] == 3

    request = urllib.request.Request(
      f"{base}/media/3/cam_high", headers={"Range": "bytes=2-5"}
    )
    with _urlopen(request) as response:
      assert response.status == 206
      assert response.headers["Content-Range"] == "bytes 2-5/16"
      assert response.headers["Accept-Ranges"] == "bytes"
      assert response.read() == b"2345"

    request = urllib.request.Request(
      f"{base}/media/3/cam_high",
      method="HEAD",
      headers={"Range": "bytes=-4"},
    )
    with _urlopen(request) as response:
      assert response.status == 206
      assert response.headers["Content-Length"] == "4"
      assert response.headers["Content-Range"] == "bytes 12-15/16"
      assert response.read() == b""

    request = urllib.request.Request(
      f"{base}/media/3/cam_high", headers={"Range": "bytes=0-1,4-5"}
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
      _urlopen(request)
    assert caught.value.code == 416
    assert caught.value.headers["Content-Range"] == "bytes */16"
    assert json.loads(caught.value.read())["error"]["code"] == "invalid_range"


def test_delete_restore_and_conflict_are_all_or_nothing(tmp_path: Path) -> None:
  _episode(tmp_path, 8)
  static = tmp_path / "web"
  static.mkdir()
  for filename in ("index.html", "app.js", "styles.css"):
    (static / filename).write_text("", encoding="utf-8")
  paths = [
    tmp_path / "episode_8.hdf5",
    *(tmp_path / "video" / camera / "episode_8.mp4" for camera in (
      "cam_high",
      "cam_left_wrist",
      "cam_right_wrist",
    )),
  ]

  with _running_server(tmp_path, static) as base:
    with pytest.raises(urllib.error.HTTPError) as caught:
      _json_request(
        f"{base}/api/episodes/8",
        method="DELETE",
        payload={"confirm": "episode_7"},
      )
    assert caught.value.code == 400
    assert all(path.exists() for path in paths)

    _, _, deleted = _json_request(
      f"{base}/api/episodes/8",
      method="DELETE",
      payload={"confirm": "episode_8"},
    )
    assert deleted["deleted"] is True
    assert len(deleted["files"]) == 4
    assert not any(path.exists() for path in paths)

    paths[1].write_bytes(b"conflict")
    with pytest.raises(urllib.error.HTTPError) as caught:
      _json_request(
        f"{base}/api/trash/{deleted['token']}/restore",
        method="POST",
      )
    assert caught.value.code == 409
    assert json.loads(caught.value.read())["error"]["code"] == "restore_conflict"
    assert not paths[0].exists()

    paths[1].unlink()
    _, _, restored = _json_request(
      f"{base}/api/trash/{deleted['token']}/restore",
      method="POST",
    )
    assert restored["restored"] is True
    assert all(path.exists() for path in paths)


def test_delete_rolls_back_if_one_of_four_moves_fails(tmp_path: Path, monkeypatch) -> None:
  _episode(tmp_path, 9)
  paths = [
    tmp_path / "episode_9.hdf5",
    *(tmp_path / "video" / camera / "episode_9.mp4" for camera in (
      "cam_high",
      "cam_left_wrist",
      "cam_right_wrist",
    )),
  ]
  real_replace = browser.os.replace
  failed = False

  def replace(source, destination):
    nonlocal failed
    source_path = Path(source)
    if not failed and source_path == paths[2]:
      failed = True
      raise OSError("injected move failure")
    return real_replace(source, destination)

  monkeypatch.setattr(browser.os, "replace", replace)

  with pytest.raises(browser.ApiError) as caught:
    browser.EpisodeStore(tmp_path).delete("9", "episode_9")

  assert caught.value.code == "delete_failed"
  assert all(path.exists() for path in paths)
  trash = tmp_path / browser.TRASH_DIRECTORY
  assert not trash.exists() or not tuple(trash.iterdir())


def test_restore_recovers_a_delete_interrupted_while_moving(tmp_path: Path) -> None:
  _episode(tmp_path, 12)
  store = browser.EpisodeStore(tmp_path)
  trash_root = store._ensure_trash_root()
  token, entry = store._new_trash_entry(trash_root, "12")
  relative_paths = [
    "episode_12.hdf5",
    "video/cam_high/episode_12.mp4",
    "video/cam_left_wrist/episode_12.mp4",
    "video/cam_right_wrist/episode_12.mp4",
  ]
  manifest = {
    "version": 1,
    "token": token,
    "episode": "episode_12",
    "identifier": "12",
    "state": "moving",
    "files": relative_paths,
  }
  store._atomic_json(entry / "manifest.json", manifest)
  for relative in relative_paths[:2]:
    source = tmp_path / relative
    destination = entry / "files" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    browser.os.replace(source, destination)

  result = store.restore(token)

  assert result["restored"] is True
  assert result["trash_cleanup_pending"] is False
  assert all((tmp_path / relative).is_file() for relative in relative_paths)
  assert not entry.exists()


def test_restore_reports_only_trash_cleanup_as_pending(tmp_path: Path, monkeypatch) -> None:
  _episode(tmp_path, 13)
  store = browser.EpisodeStore(tmp_path)
  deleted = store.delete("13", "episode_13")

  def fail_cleanup(_entry):
    raise OSError("injected cleanup failure")

  monkeypatch.setattr(store, "_cleanup_entry", fail_cleanup)
  result = store.restore(deleted["token"])

  assert result["restored"] is True
  assert result["trash_cleanup_pending"] is True
  assert (tmp_path / "episode_13.hdf5").is_file()


def test_path_traversal_routes_and_manifest_paths_are_rejected(tmp_path: Path) -> None:
  _episode(tmp_path, 1)
  static = tmp_path / "web"
  static.mkdir()
  for filename in ("index.html", "app.js", "styles.css"):
    (static / filename).write_text("", encoding="utf-8")

  with _running_server(tmp_path, static) as base:
    with pytest.raises(urllib.error.HTTPError) as caught:
      _urlopen(f"{base}/media/1/%2e%2e")
    assert caught.value.code in {400, 404}

  manifest = {
    "files": ["episode_1.hdf5", "../outside/episode_1.mp4"],
  }
  with pytest.raises(browser.ApiError) as caught:
    browser.EpisodeStore._validated_manifest_paths(manifest, "1")
  assert caught.value.code == "invalid_manifest"
