"""Offline review synchronization and artifacts with tiny HDF5 and fake encoding."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared import poker_review as review
from kaihand_tactile_env.shared.recording import ValidationReport
from kaihand_tactile_env.shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from PIL import Image


def _source(tmp_path, second_camera="overhead"):
  source = tmp_path / "episode.h5"
  camera_times = np.array([0, 0.100, 0.200, 0.250])
  pose_times = np.array([0, 0.098, 0.198, 0.248])
  tactile_times = np.r_[0.0, np.arange(0.008, 0.249, 0.01)]
  links = ["left_unused", *reversed(RIGHT_FINGERTIP_LINK_NAMES)]
  count = len(tactile_times)
  with h5py.File(source, "w") as file:
    file.attrs["metadata_json"] = json.dumps(
      {"scene": "poker-draw", "episode_index": 1, "seed": 7}
    )
    file.attrs["outcome_json"] = json.dumps({"success": True})
    file.attrs["camera_hz"] = 10
    file.create_dataset(
      "state/timestamp", data=tactile_times + np.r_[0, np.full(count - 1, 0.002)]
    )
    file.create_dataset("commands/phase", data=[b"reach"] * count)
    for index, name in enumerate(("head", second_camera)):
      rgb = np.zeros((4, 12, 16, 3), dtype=np.uint8)
      for row in range(4):
        rgb[row, :, :, index] = 20 + row * 20
      group = file.create_group(f"cameras/{name}")
      group.create_dataset("timestamp", data=camera_times)
      group.create_dataset("pose_timestamp", data=pose_times)
      group.create_dataset("rgb", data=rgb)
    force = file.create_group("tactile_contact_force")
    force.create_dataset("timestamp", data=tactile_times)
    force.create_dataset(
      "link_names", data=np.asarray(links, dtype=h5py.string_dtype())
    )
    normal = np.zeros((count, 6, 7, 5))
    tangent = np.zeros((count, 6, 7, 5, 2))
    for finger in range(6):
      normal[:, finger] = finger * 0.001
      tangent[:, finger, :, :, 0] = -finger * 0.0002
      tangent[:, finger, :, :, 1] = finger * 0.0001
    force.create_dataset("normal_taxel_force_n", data=normal)
    force.create_dataset("tangent_taxel_force_n", data=tangent)
  return source


def _fake_pipeline(monkeypatch, *, probe_changes=None, fail_write=False):
  events = []
  validation = ValidationReport(True, (), (), 26, {"head": 4, "overhead": 4}, 0.250)

  def validate(path):
    events.append(("validate", Path(path)))
    return validation

  class Writer:
    def __init__(self, path, *, fps, width, height, executable=None):
      assert executable == "/fake/ffmpeg"
      self.path, self.fps, self.width, self.height = path, fps, width, height
      self.count = 0
      events.append(("encoder", self))

    def write(self, frame):
      assert frame.shape == (self.height, self.width, 3)
      assert frame.dtype == np.uint8
      if fail_write:
        raise RuntimeError("fake encoder failure")
      self.count += 1

    def finish(self):
      # This marker is not an encoded video; ffprobe is also replaced below.
      self.path.write_bytes(b"fake encoded content")
      events.append(("finish", self.count))

    def abort(self):
      events.append(("abort", self.count))

  def probe(path, *, backend=None):
    writer = next(value for kind, value in events if kind == "encoder")
    assert path == writer.path
    events.append(("probe", path))
    return {
      "backend": "ffprobe",
      "frame_count": writer.count,
      "duration_s": writer.count / writer.fps,
      "fps": writer.fps,
      "width": writer.width,
      "height": writer.height,
      **(probe_changes or {}),
    }

  monkeypatch.setattr(review, "validate_episode", validate)
  monkeypatch.setattr(review, "_find_ffmpeg_executable", lambda: "/fake/ffmpeg")
  monkeypatch.setattr(review, "_FfmpegPipeWriter", Writer)
  monkeypatch.setattr(review, "_probe_video", probe)
  return events


def test_frame_selection_never_repeats_rgb_and_always_keeps_terminal():
  times = np.array([0, 0.1, 0.2, 0.3, 0.35])
  np.testing.assert_array_equal(review.select_camera_frames(times, 5), [0, 2, 4])
  np.testing.assert_array_equal(review.select_camera_frames(times, 10), [0, 1, 2, 3, 4])


def test_matches_pose_clock_independently_and_uses_latest_nonfuture_tactile(tmp_path):
  source = _source(tmp_path)
  with h5py.File(source, "r+") as file:
    force_times = file["tactile_contact_force/timestamp"]
    force_times[10] = (
      0.099  # Later than image pose .098, though before camera time .100.
    )
    frames = review.plan_review_frames(file)
    assert frames[1].camera_timestamp_s == 0.100
    assert frames[1].camera_pose_timestamp_s == 0.098
    assert frames[1].tactile_index == 9
    assert frames[1].tactile_timestamp_s == pytest.approx(0.088)
    assert frames[1].tactile_age_s == pytest.approx(0.010)


@pytest.mark.parametrize(
  "mutation",
  [
    "future",
    "stale",
    "different_pose",
    "different_capture",
    "missing_pose",
    "missing_rgb",
    "frame_count",
    "unordered",
  ],
)
def test_rejects_missing_or_unaligned_evidence(tmp_path, mutation):
  source = _source(tmp_path)
  with h5py.File(source, "r+") as file:
    if mutation == "future":
      file["tactile_contact_force/timestamp"][0] = 0.001
    elif mutation == "stale":
      values = file["tactile_contact_force/timestamp"][:]
      values[values > 0.218] += 0.050
      file["tactile_contact_force/timestamp"][:] = values
    elif mutation == "different_pose":
      file["cameras/overhead/pose_timestamp"][1] -= 0.001
    elif mutation == "different_capture":
      file["cameras/overhead/timestamp"][1] += 0.001
    elif mutation == "missing_pose":
      del file["cameras/head/pose_timestamp"]
    elif mutation == "missing_rgb":
      del file["cameras/overhead/rgb"]
    elif mutation == "frame_count":
      del file["cameras/overhead/timestamp"]
      file.create_dataset("cameras/overhead/timestamp", data=[0, 0.1])
    elif mutation == "unordered":
      file["cameras/head/pose_timestamp"][2] = 0.01
    with pytest.raises((ValueError, KeyError)):
      review.plan_review_frames(file)


@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("second_camera", ["overhead", "right_wrist"])
def test_streamed_export_preserves_raw_rgb_signed_tangent_and_timing(
  tmp_path, monkeypatch, fps, second_camera
):
  source = _source(tmp_path, second_camera)
  original_digest = review._sha256(source)
  events = _fake_pipeline(monkeypatch)
  output = tmp_path / "review"
  result = review.export_poker_review(source, output, fps=fps, width=960, height=540)
  assert result["camera_names"] == ["head", second_camera]
  assert f"head above {second_camera}" in result["layout"]
  assert output.is_dir() and not output.with_name("review.partial").exists()
  assert review._sha256(source) == original_digest
  assert events[0] == ("validate", source)
  assert [kind for kind, _ in events] == ["validate", "encoder", "finish", "probe"]
  rows = list(csv.DictReader((output / "frames.csv").open()))
  assert len(rows) == result["output_frame_count"]
  assert int(rows[-1]["camera_index"]) == 3
  assert result["last_camera_pose_timestamp_s"] == 0.248
  assert result["last_camera_timestamp_s"] == 0.250
  assert result["constant_fps_duration_s"] == len(rows) / fps
  assert result["last_frame_playback_time_error_s"] == pytest.approx(
    (len(rows) - 1) / fps - 0.248
  )
  assert result["normal_color_scale_n_per_taxel"] == [0, 0.1]
  assert result["tangent_color_scale_n_per_taxel"] == [0, 0.02]
  with h5py.File(source, "r") as file:
    right = review._force_layout(file)
    for row in rows:
      index, source_index = int(row["output_index"]), int(row["camera_index"])
      force_index = int(row["tactile_index"])
      for name in ("head", second_camera):
        image = np.asarray(Image.open(output / "frames" / name / f"{index:06d}.png"))
        np.testing.assert_array_equal(image, file[f"cameras/{name}/rgb"][source_index])
      with np.load(output / "raw" / f"{index:06d}.npz", allow_pickle=False) as raw:
        expected = file["tactile_contact_force/tangent_taxel_force_n"][force_index][
          right
        ]
        np.testing.assert_array_equal(raw["tangent_taxel_force_n"], expected)
        assert np.any(raw["tangent_taxel_force_n"] < 0)
        np.testing.assert_array_equal(
          raw["tangent_magnitude_n"], np.linalg.norm(expected, axis=-1)
        )
      for name in ("normal", "tangent", "composite"):
        assert (output / "frames" / name / f"{index:06d}.png").is_file()
  assert json.loads((output / "review.json").read_text())["completed"] is True


def test_wrist_review_never_substitutes_overhead_or_ignores_its_clock(tmp_path):
  source = _source(tmp_path, "right_wrist")
  with h5py.File(source, "r+") as file:
    assert review.review_cameras(file) == ("head", "right_wrist")
    with pytest.raises(ValueError, match="missing requested camera"):
      review.plan_review_frames(file, second_camera="overhead")
    file["cameras/right_wrist/pose_timestamp"][1] -= 0.001
    with pytest.raises(ValueError, match="identical capture and pose times"):
      review.plan_review_frames(file)


def test_requesting_wrist_for_legacy_source_fails_before_creating_output(
  tmp_path, monkeypatch
):
  source = _source(tmp_path)
  _fake_pipeline(monkeypatch)
  output = tmp_path / "review"
  with pytest.raises(ValueError, match="missing requested camera 'right_wrist'"):
    review.export_poker_review(source, output, second_camera="right_wrist")
  assert not output.exists()
  assert not output.with_name("review.partial").exists()


def test_export_reads_rgb_and_force_grids_one_frame_at_a_time(tmp_path, monkeypatch):
  source = _source(tmp_path)
  _fake_pipeline(monkeypatch)
  original_getitem = h5py.Dataset.__getitem__
  calls = []

  def guarded(dataset, key, *args, **kwargs):
    if dataset.name.endswith(
      ("/rgb", "/normal_taxel_force_n", "/tangent_taxel_force_n")
    ):
      assert isinstance(key, (int, np.integer)), (
        f"bulk array read: {dataset.name} {key}"
      )
      calls.append((dataset.name, int(key)))
    return original_getitem(dataset, key, *args, **kwargs)

  monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
  result = review.export_poker_review(
    source, tmp_path / "review", width=960, height=540
  )
  assert len(calls) == result["output_frame_count"] * 4


@pytest.mark.parametrize("artifact", ["review", "review.partial"])
def test_existing_success_or_partial_directory_is_never_overwritten(
  tmp_path, monkeypatch, artifact
):
  source = _source(tmp_path)
  existing = tmp_path / artifact
  existing.mkdir()
  marker = existing / "keep.txt"
  marker.write_text("existing user artifact")
  events = _fake_pipeline(monkeypatch)
  with pytest.raises(FileExistsError):
    review.export_poker_review(source, tmp_path / "review")
  assert marker.read_text() == "existing user artifact"
  assert events == []


@pytest.mark.parametrize(
  "mutation", ["outcome", "scene", "validation", "partial", "lock", "failure"]
)
def test_requires_successful_completed_source_before_encoder(
  tmp_path, monkeypatch, mutation
):
  source = _source(tmp_path)
  events = _fake_pipeline(monkeypatch)
  if mutation in ("outcome", "scene"):
    with h5py.File(source, "r+") as file:
      file.attrs["outcome_json" if mutation == "outcome" else "metadata_json"] = (
        json.dumps(
          {"success": False} if mutation == "outcome" else {"scene": "pick-place"}
        )
      )
  elif mutation == "validation":
    monkeypatch.setattr(
      review,
      "validate_episode",
      lambda _: ValidationReport(False, ("invalid",), (), 0, {}, 0),
    )
  else:
    suffix = {"partial": ".h5.partial", "lock": ".h5.lock", "failure": ".failure.json"}[
      mutation
    ]
    source.with_suffix(suffix).touch()
  with pytest.raises(ValueError):
    review.export_poker_review(source, tmp_path / "review")
  assert not any(kind == "encoder" for kind, _ in events)
  assert not (tmp_path / "review.partial").exists()


@pytest.mark.parametrize(
  "changes", [{"frame_count": 999}, {"duration_s": 99}, {"fps": 99}, {"width": 111}]
)
def test_ffprobe_mismatch_keeps_partial_and_never_claims_completion(
  tmp_path, monkeypatch, changes
):
  source = _source(tmp_path)
  _fake_pipeline(monkeypatch, probe_changes=changes)
  output = tmp_path / "review"
  with pytest.raises(ValueError, match="ffprobe"):
    review.export_poker_review(source, output, width=960, height=540)
  assert not output.exists()
  partial = tmp_path / "review.partial"
  assert json.loads((partial / "failure.json").read_text())["completed"] is False


def test_encoder_failure_preserves_partial_and_original_error(tmp_path, monkeypatch):
  source = _source(tmp_path)
  events = _fake_pipeline(monkeypatch, fail_write=True)
  with pytest.raises(RuntimeError, match="fake encoder failure"):
    review.export_poker_review(source, tmp_path / "review", width=960, height=540)
  assert not (tmp_path / "review").exists()
  assert (tmp_path / "review.partial/frames/head/000000.png").exists()
  assert events[-1][0] == "abort"


def test_tangent_panel_displays_magnitude_without_mutating_signed_data():
  normal = np.ones((5, 7, 5)) * 0.02
  tangent = np.zeros((5, 7, 5, 2))
  tangent[:, :, :, 0] = -0.01
  original = tangent.copy()
  negative = review.force_panel(
    normal, tangent, width=600, height=260, quantity="tangent"
  )
  positive = review.force_panel(
    normal, -tangent, width=600, height=260, quantity="tangent"
  )
  np.testing.assert_array_equal(np.asarray(negative), np.asarray(positive))
  np.testing.assert_array_equal(tangent, original)


@pytest.mark.parametrize("stream_duration", [True, False])
def test_ffprobe_parses_stream_or_container_duration_without_encoding(
  tmp_path, monkeypatch, stream_duration
):
  stream = {
    "nb_read_frames": "3",
    "avg_frame_rate": "10/1",
    "width": 960,
    "height": 540,
  }
  payload = {"streams": [stream]}
  if stream_duration:
    stream["duration"] = "0.300000"
  else:
    payload["format"] = {"duration": "0.300000"}
  commands = []

  def run(command, **kwargs):
    commands.append(command)
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 60
    return SimpleNamespace(stdout=json.dumps(payload))

  monkeypatch.setattr(review.shutil, "which", lambda _: "/fake/ffprobe")
  monkeypatch.setattr(review.subprocess, "run", run)
  probe = review._probe_video(tmp_path / "review.mp4")
  assert probe == {
    "backend": "ffprobe",
    "executable": "/fake/ffprobe",
    "frame_count": 3,
    "duration_s": 0.3,
    "fps": 10.0,
    "width": 960,
    "height": 540,
  }
  assert commands[0][commands[0].index("-threads") + 1] == "1"


def _frame_checksums(
  *, count=3, ticks=1024, time_base="1/10240", width=960, height=540
):
  header = f"#format: frame checksums\n#version: 2\n#tb 0: {time_base}\n#dimensions 0: {width}x{height}\n"
  return (
    header
    + "".join(
      f"0, {i * ticks}, {i * ticks}, {ticks}, {width * height * 3}, {'0' * 32}\n"
      for i in range(count)
    )
  ).encode()


def test_missing_ffprobe_uses_bundled_ffmpeg_single_thread_full_decode(
  tmp_path, monkeypatch
):
  calls = []
  monkeypatch.setattr(review.shutil, "which", lambda _: None)
  monkeypatch.setattr(review, "_find_ffmpeg_executable", lambda: "/bundled/ffmpeg")

  def run(command, **kwargs):
    calls.append(command)
    assert kwargs["check"] is True and kwargs["timeout"] == 60
    assert "capture_output" not in kwargs
    # Only tiny checksums go to disk; decoded RGB stays inside the one ffmpeg process.
    kwargs["stdout"].write(_frame_checksums())
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(review.subprocess, "run", run)
  probe = review._probe_video(tmp_path / "review.mp4")
  assert probe["backend"] == "ffmpeg_framemd5_full_decode"
  assert probe["frame_count"] == 3
  assert probe["fps"] == 10 and probe["duration_s"] == 0.3
  assert (probe["width"], probe["height"]) == (960, 540)
  assert probe["full_decode"] is True and probe["decoder_threads"] == 1
  assert probe["decoded_time_base"] == "1/10240"
  command = calls[0]
  assert command[0] == "/bundled/ffmpeg"
  assert command[-3:] == ["-f", "framemd5", "pipe:1"]
  assert command.count("-threads") == 2
  assert all(
    command[index + 1] == "1" for index, arg in enumerate(command) if arg == "-threads"
  )
  assert command.index("-threads") < command.index("-i")
  assert "-xerror" in command and "explode" in command
  assert command[command.index("-vsync") + 1] == "0"
  review._validate_probe(probe, count=3, fps=10, width=960, height=540)


@pytest.mark.parametrize("failure", ["decode", "timeout"])
def test_complete_decoding_failure_rejects_even_a_plausible_partial_count(
  tmp_path, monkeypatch, failure
):
  def run(command, **kwargs):
    kwargs["stdout"].write(_frame_checksums())
    kwargs["stderr"].write(b"corrupt decoded frame")
    if failure == "timeout":
      raise subprocess.TimeoutExpired(command, 60)
    raise subprocess.CalledProcessError(1, command)

  monkeypatch.setattr(review.subprocess, "run", run)
  with pytest.raises(RuntimeError, match="full video decode failed.*corrupt"):
    review._probe_video(
      tmp_path / "review.mp4", backend=("ffmpeg_framemd5_full_decode", "/fake/ffmpeg")
    )


@pytest.mark.parametrize(
  "mutation",
  [
    "empty",
    "missing_dimensions",
    "missing_time_base",
    "truncated_line",
    "size",
    "pts_gap",
    "varying_duration",
  ],
)
def test_full_decode_parser_rejects_unproven_or_nonconstant_frames(mutation):
  payload = _frame_checksums()
  if mutation == "empty":
    payload = b""
  elif mutation == "missing_dimensions":
    payload = payload.replace(b"#dimensions", b"#unknown")
  elif mutation == "missing_time_base":
    payload = payload.replace(b"#tb", b"#unknown")
  elif mutation == "truncated_line":
    payload = payload[:-20]
  elif mutation == "size":
    payload = payload.replace(b"1555200", b"100")
  elif mutation == "pts_gap":
    payload = payload.replace(b"0, 1024, 1024,", b"0, 1024, 1025,")
  elif mutation == "varying_duration":
    payload = payload.replace(b"0, 1024, 1024, 1024,", b"0, 1024, 1024, 1025,")
  with pytest.raises(ValueError):
    review._parse_decoded_frames(payload.splitlines(), executable="/fake/ffmpeg")


def test_decoder_timestamp_duration_uses_rational_precision():
  probe = review._parse_decoded_frames(
    _frame_checksums(count=442).splitlines(), executable="/fake/ffmpeg"
  )
  assert probe["frame_count"] == 442
  assert probe["duration_s"] == 44.2
  review._validate_probe(probe, count=442, fps=10, width=960, height=540)


def test_missing_encoding_or_validation_dependency_fails_before_artifacts(
  tmp_path, monkeypatch
):
  source = _source(tmp_path)
  events = _fake_pipeline(monkeypatch)

  def unavailable():
    raise RuntimeError("ffmpeg unavailable")

  monkeypatch.setattr(review, "_find_ffmpeg_executable", unavailable)
  with pytest.raises(RuntimeError, match="unavailable"):
    review.export_poker_review(source, tmp_path / "review")
  assert events == []
  assert not (tmp_path / "review.partial").exists()


def test_fallback_report_never_claims_ffprobe_was_used(tmp_path, monkeypatch):
  source = _source(tmp_path)
  _fake_pipeline(
    monkeypatch,
    probe_changes={"backend": "ffmpeg_framemd5_full_decode", "full_decode": True},
  )
  result = review.export_poker_review(
    source, tmp_path / "review", width=960, height=540
  )
  assert result["ffprobe"] is None
  assert result["video_validation"]["backend"] == "ffmpeg_framemd5_full_decode"
  assert result["video_validation"]["full_decode"] is True
