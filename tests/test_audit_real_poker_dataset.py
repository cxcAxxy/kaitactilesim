from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
  "audit_real_poker", Path(__file__).resolve().parents[1] / "scripts/workcell/audit_real_poker_dataset.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_clock_duplicate_reverse_gap():
  result = audit.clock_check(np.array([0, 10, 10, 5, 35], dtype=np.int64), 10)
  assert result["duplicate_transitions"] == [2]
  assert result["reversed_transitions"] == [3]
  assert result["gap_transitions"] == [4]
  assert result["estimated_missing_samples"] == 2
  assert not audit.clock_check([], 10)["valid"]


def test_duplicate_runs_including_terminal():
  assert audit.duplicate_runs([]) == []
  assert audit.duplicate_runs(["a", "a", "b", "c", "c", "c"]) == [
    {"start": 0, "end": 1, "length": 2}, {"start": 3, "end": 5, "length": 3}
  ]


def test_trajectory_distinguishes_arm_hand_and_nan():
  values = np.zeros((20, 26))
  values[5, 0] = 0.18
  values[5, 7] = 0.18  # Below hand threshold, above arm threshold.
  values[8, 20] = 1.2
  values[10, 2] = np.nan
  stamps = np.arange(20, dtype=np.int64) * 33_333_333
  result = audit.trajectory_check(values, stamps, 0.15, 0.2)
  assert result["nonfinite_cells"] == [[10, 2]]
  assert result["hand_out_of_0_1_cells"] == [[8, 20]]
  events = result["jump_events"]
  assert {(e["frame"], e["dimension"]) for e in events} == {(5, 0), (6, 0), (8, 20), (9, 20)}
  assert events[0]["one_frame_return_spike"]
  assert not audit.trajectory_check(values[:, :7], stamps, 0.15, 0.2)["valid"]


def test_h5_semantics_and_independent_tactile_clock(tmp_path):
  path = tmp_path / "episode_0.hdf5"
  values = np.arange(20 * 26, dtype=np.float32).reshape(20, 26) / 10000
  with h5py.File(path, "w") as f:
    f.attrs.update(qpos_layout="interleaved_LhRh", arm_dof=14, hand_dof_per_side=6,
                   action_source="armcmd_handcmd", task="pick up the cup", fps=30)
    f["timestamp_ros_nsec"] = np.arange(20, dtype=np.int64) * 33_333_333
    f["observations/qpos"] = values
    f["slave_state"] = values
    f["action"] = values + 0.01
    f["master_action"] = values + 0.01
    f["action_actual_shift"] = np.concatenate([values[1:], values[-1:]])
    f["raw/revo2_touch/left/stamp_nsec"] = np.arange(34, dtype=np.int64) * 20_000_000
    f["raw/revo2_touch/left/data"] = np.zeros((34, 5, 5))
  result = audit.audit_h5(path, SimpleNamespace(arm_step=0.15, hand_step=0.2))
  assert not result["aligned_length_mismatches"]
  assert result["relations"]["action_equals_master_action"]["equal"]
  assert not result["relations"]["action_equals_qpos"]["equal"]
  assert result["relations"]["actual_shift_equals_next_qpos_except_last"]["equal"]
  assert result["actual_shift_last_equals_last_qpos"]
  assert result["tactile"]["left"]["sample_count_matches"]
  assert not result["text_fields"]


def test_missing_video_is_reported(tmp_path):
  result = audit.audit_video(tmp_path / "missing.mp4", 10, 30)
  assert "error" in result


def test_semantics_reports_command_matching_gaps(tmp_path):
  episode = {"episode": "episode_15", "frames": 1123,
             "attributes": {"action_source": "armcmd_handcmd", "stream_stats_json": '{"arm_cmd_matched_frames": 1121}'}}
  audit.write_semantics(tmp_path, [episode])
  result = json.loads((tmp_path / "semantics.json").read_text())
  assert result["command_matching_coverage_warnings"] == [
    {"episode": "episode_15", "frames": 1123, "arm_cmd_matched_frames": 1121, "unmatched_frames": 2}]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_full_video_decode_duplicates_and_frame_mismatch(tmp_path):
  path = tmp_path / "constant.mkv"
  subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=32x32:r=10",
                  "-frames:v", "5", "-c:v", "ffv1", "-threads", "1", str(path)], check=True)
  result = audit.audit_video(path, 6, 10)
  assert "error" not in result
  assert result["decoded_frames"] == 5
  assert result["adjacent_duplicate_frames"] == 4
  assert result["duplicate_runs"] == [{"start": 0, "end": 4, "length": 5}]
  assert result["frame_count_matches"] is False
  assert result["presentation_clock_seconds"]["estimated_missing_samples"] == 0
