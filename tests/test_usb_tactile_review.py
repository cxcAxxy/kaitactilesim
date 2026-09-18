"""Synthetic offline visual evidence checks; no MuJoCo or rendering."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.usb_tactile_review import (
  active_intervals,
  event_bracket,
  review_usb_tactile,
)
from PIL import Image


@pytest.fixture
def tactile_source(tmp_path):
  source = tmp_path / "usb.h5"
  times = np.arange(21, dtype=float) * 0.002
  solver = np.maximum(0, times - 0.002)
  frames = np.array([0, 4, 8, 12, 16, 20])
  names = [
    f"hand_{side}_{finger}_link{6 if finger == 'thumb' else 4}"
    for side in ("l", "r")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
  ]
  with h5py.File(source, "w") as file:
    file.attrs.update(
      physics_hz=500,
      camera_hz=125,
      contact_force_source="solver_contact_distributed_taxel_v1",
    )
    file.create_dataset("state/timestamp", data=times)
    force = file.create_group("tactile_contact_force")
    force.attrs.update(
      force_unit="N",
      timestamp_reference="/tactile_contact_force/timestamp",
      metadata_json=json.dumps({"target_geom_names": ["usb_plug_collision"]}),
    )
    force.create_dataset("timestamp", data=solver)
    force.create_dataset("link_names", data=names, dtype=h5py.string_dtype())
    normal = np.zeros((21, 10))
    normal[4, 6] = 1e-5
    normal[5:15, 5:7] = 0.1
    normal[9, 5:7] = 0.2
    force.create_dataset("normal_force_n", data=normal)
    tangent = np.zeros((21, 10, 2))
    tangent[5:15, 6] = [-0.01, 0.02]
    force.create_dataset("tangent_force_n", data=tangent)
    force.create_dataset("tangent_load_n", data=np.linalg.norm(tangent, axis=-1))
    camera = file.create_group("cameras/head")
    camera.create_dataset("timestamp", data=times[frames])
    camera.create_dataset("pose_timestamp", data=solver[frames])
    camera.create_dataset("state_index", data=frames)
    pixels = np.broadcast_to(
      np.arange(6, dtype=np.uint8)[:, None, None, None], (6, 24, 32, 3)
    )
    camera.create_dataset("rgb", data=pixels)
  return source


def test_brackets_are_causal_and_preserve_episode_boundaries():
  times = np.array([0.0, 0.098, 0.198, 0.214])
  assert event_bracket(times, 0.1) == {"at_or_before": 1, "at_or_after": 2}
  assert event_bracket(times, 0.198) == {"at_or_before": 2, "at_or_after": 2}
  assert event_bracket(times, -0.002) == {"at_or_before": None, "at_or_after": 0}
  assert event_bracket(times, 0.216) == {"at_or_before": 3, "at_or_after": None}


def test_intervals_preserve_gaps_single_sample_and_final_active():
  assert active_intervals(np.array([0, 1, 0, 1, 1])) == [(1, 1), (3, 4)]
  assert active_intervals(np.zeros(0, dtype=bool)) == []
  assert active_intervals(np.zeros(5, dtype=bool)) == []
  assert active_intervals(np.ones(1, dtype=bool)) == [(0, 0)]


def test_full_evidence_uses_solver_clock_signed_forces_and_original_rgb(
  tactile_source, tmp_path
):
  output = tmp_path / "review"
  report = review_usb_tactile(tactile_source, output)
  physical, practical = report["thresholds"]
  assert physical["global"]["first_active_index"] == 4
  assert practical["global"]["first_active_index"] == 5
  assert physical["global"]["last_active_index"] == 14
  assert physical["global"]["first_inactive_after_last_index"] == 15
  assert len(physical["per_link"]) == 10
  assert not physical["per_link"]["hand_l_thumb_link6"]["has_signal"]
  first = practical["events"][0]
  assert first["state_timestamp_s"] == pytest.approx(0.010)
  assert first["solver_timestamp_s"] == pytest.approx(0.008)
  assert first["tangent_force_n"][6] == [-0.01, 0.02]
  before, after = first["images"]["at_or_before"], first["images"]["at_or_after"]
  assert before["camera_frame_index"] == 1
  assert before["event_delta_ms"] == pytest.approx(-2)
  assert after["camera_frame_index"] == 2
  assert after["event_delta_ms"] == pytest.approx(6)
  assert before["physical_force_at_image_pose"]["raw_state_index"] == 4
  assert after["physical_force_at_image_pose"]["active_mask"][6]
  with h5py.File(tactile_source, "r") as file:
    for group in report["thresholds"]:
      for event in group["events"]:
        for frame in event["images"].values():
          if frame is not None:
            with Image.open(output / frame["rgb_path"]) as image:
              assert np.array_equal(
                np.asarray(image), file["cameras/head/rgb"][frame["camera_frame_index"]]
              )
  assert (output / "index.html").is_file()
  assert (output / physical["contact_sheet"]).is_file()
  saved = json.loads((output / "review.json").read_text())
  assert saved["review_status"] == "requires_human_review"
  assert saved["rgb_tactile_solver_clock_max_error_s"] == 0
  assert len(saved["overview_images"]) == 3
  assert saved["warnings"] == []
  with pytest.raises(FileExistsError):
    review_usb_tactile(tactile_source, output)


def test_rejects_camera_tactile_epoch_mismatch(tactile_source, tmp_path):
  with h5py.File(tactile_source, "r+") as file:
    file["cameras/head/pose_timestamp"][2] += 0.001
  with pytest.raises(ValueError, match="RGB pose clock"):
    review_usb_tactile(tactile_source, tmp_path / "review")
  assert not (tmp_path / "review").exists()


def test_no_signal_keeps_overview_and_does_not_invent_events(tactile_source, tmp_path):
  with h5py.File(tactile_source, "r+") as file:
    for name in ("normal_force_n", "tangent_force_n", "tangent_load_n"):
      file[f"tactile_contact_force/{name}"][:] = 0
  report = review_usb_tactile(tactile_source, tmp_path / "review")
  assert not report["thresholds"][0]["global"]["has_signal"]
  assert report["thresholds"][0]["events"] == []
  assert report["selected_rgb_frames"] == 3
  assert len(report["warnings"]) == 2


def test_censored_contact_has_no_invented_release(tactile_source, tmp_path):
  with h5py.File(tactile_source, "r+") as file:
    file["tactile_contact_force/normal_force_n"][:, 5] = 0.1
  report = review_usb_tactile(tactile_source, tmp_path / "review")
  summary = report["thresholds"][0]["global"]
  assert summary["onset_censored_by_episode_start"]
  assert summary["offset_censored_by_episode_end"]
  assert summary["first_inactive_after_last_index"] is None
  assert not any(
    event["kind"] == "first_inactive_after_last"
    for event in report["thresholds"][0]["events"]
  )


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf")])
def test_bad_threshold_rejected(tactile_source, tmp_path, threshold):
  with pytest.raises(ValueError, match="thresholds"):
    review_usb_tactile(
      tactile_source, tmp_path / "review", signal_threshold_n=threshold
    )


def test_synthetic_probe_is_not_accepted_as_physical_force(tactile_source, tmp_path):
  with h5py.File(tactile_source, "r+") as file:
    file.attrs["contact_force_source"] = "genesis_probe_bimanual_clean_v1"
  with pytest.raises(ValueError, match="physical solver"):
    review_usb_tactile(tactile_source, tmp_path / "review")
