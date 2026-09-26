"""Small contract tests for legacy Poker review reformatting."""

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.workcell import reformat_legacy_poker_evaluation as legacy


def _review() -> dict:
  return {
    "schema": legacy.SOURCE_SCHEMA,
    "task": "poker-draw",
    "fps": 10,
    "output_size": [1920, 1080],
    "review_render_size": [640, 480],
    "second_camera": "global",
    "model_input_cameras_displayed": ["head", "right_wrist"],
    "layout": (
      "head/right-wrist/global-or-named RGB; bilateral Fn/|Ft| maps; "
      "bilateral 3-axis mean curves"
    ),
    "frame_count": 2,
    "fingertip_order": [f"finger_{index}" for index in range(10)],
  }


def _frame(index: int, names: list[str]) -> dict:
  return {
    "frame": index,
    "control_tick": 3 * index,
    "simulation_time_s": index / 10,
    "camera_pose_time_s": max(0.0, index / 10 - 0.002),
    "tactile_time_s": max(0.0, index / 10 - 0.002),
    "fingertip_order": names,
    "phase": "test",
    "task_metrics": {},
    "normal_taxel_force_n": np.zeros((10, 7, 5)).tolist(),
    "tangent_taxel_force_n": np.zeros((10, 7, 5, 2)).tolist(),
  }


def test_known_layout_and_crops_exclude_legacy_panel_labels(tmp_path: Path):
  metadata = tmp_path / "review.json"
  metadata.write_text(json.dumps(_review()), encoding="utf-8")
  assert legacy._load_source_review(metadata)["frame_count"] == 2

  rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
  for value, box in enumerate(legacy.CAMERA_CROPS_XYXY.values(), start=1):
    left, top, right, bottom = box
    rgb[top:bottom, left:right] = value
  crops = legacy._crop_legacy_cameras(rgb)
  assert crops["head"].shape == (297, 393, 3)
  assert crops["right_wrist"].shape == (297, 393, 3)
  assert crops["global"].shape == (299, 393, 3)
  assert [int(crops[name][0, 0, 0]) for name in crops] == [1, 2, 3]
  rgb[105, 150] = 99
  assert crops["head"][0, 0, 0] == 1  # A detached crop, not the decode buffer.


def test_frame_log_validates_10hz_alignment_and_full_tactile_grids(tmp_path: Path):
  review = _review()
  rows = [_frame(index, review["fingertip_order"]) for index in range(2)]
  log = tmp_path / "frames.jsonl"
  log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
  frames = legacy._load_frame_log(log, review)
  assert len(frames) == 2
  assert frames[1]["normal_taxel_force_n"].shape == (10, 7, 5)
  assert frames[1]["tangent_taxel_force_n"].shape == (10, 7, 5, 2)

  rows[1]["control_tick"] = 2
  log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
  with pytest.raises(ValueError, match="control_tick mismatch"):
    legacy._load_frame_log(log, review)

  rows[1]["control_tick"] = 3
  rows[1]["tangent_taxel_force_n"] = np.zeros((10, 7, 5)).tolist()
  log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
  with pytest.raises(ValueError, match="tactile grid shape"):
    legacy._load_frame_log(log, review)


def test_preview_compositor_declares_legacy_source(monkeypatch):
  calls = []

  def capture(*args, **kwargs):
    calls.append((args, kwargs))
    return "composited"

  monkeypatch.setattr(legacy, "compose_evaluation_frame", capture)
  row = _frame(0, _review()["fingertip_order"])
  row["normal_taxel_force_n"] = np.zeros((10, 7, 5))
  row["tangent_taxel_force_n"] = np.zeros((10, 7, 5, 2))
  assert legacy._compose_legacy_frame(
    np.zeros((1080, 1920, 3), dtype=np.uint8), row
  ) == "composited"
  args, kwargs = calls[0]
  assert args[0].shape == (297, 393, 3)
  assert kwargs["model_wrist_rgb"].shape == (297, 393, 3)
  assert kwargs["second_camera_label"] == "GLOBAL / REVIEW ONLY"
  assert kwargs["heading"] == "POKER LEGACY REFORMAT / 10 FPS"


def test_existing_output_is_never_overwritten(tmp_path: Path):
  source = tmp_path / "seed_000"
  review = source / "review"
  review.mkdir(parents=True)
  for name in ("review.mp4", "frames.jsonl", "review.json"):
    (review / name).touch()
  (source / "summary.json").touch()
  (source / "legacy_reformatted").mkdir()
  with pytest.raises(FileExistsError, match="refusing to overwrite"):
    legacy.reformat(source)
