"""Only recorded media clocks permit synchronization; paths remain trial-local."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_v1.adapters import InputChangedError, TrialData
from evaluation_v1.media import inspect_media


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "trial"
        self.root.mkdir()
        self.trial = TrialData(self.root, {}, "whiteboard-wipe",
                               time_s=np.array([5., 5.1, 5.15]),
                               task_tactile_clock_aligned=True)

    def video(self, path="review/review.mp4"):
        target = self.root / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2")
        return target

    def bundle(self, **changes):
        self.trial.provenance.clear()  # This helper starts a new fixture snapshot.
        self.video()
        record = {"schema": "kaihand-policy-evaluation-review-v3", "completed": True,
                  "fps": 10, "frame_count": 3, "video_clock": "simulation time; final off-grid frame allowed",
                  "first_pose_time_s": 5., "last_pose_time_s": 5.15}
        record.update(changes)
        (self.root / "review/review.json").write_text(json.dumps(record))
        rows = [{"frame": index, "simulation_time_s": time, "camera_pose_time_s": time}
                for index, time in enumerate((5., 5.1, 5.15))]
        self.frames(rows)
        return rows, record

    def frames(self, rows):
        (self.root / "review/frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    def test_known_bundle_uses_indices_and_preserves_off_grid_end(self):
        self.bundle()
        media = inspect_media(self.trial)
        self.assertEqual(media["status"], "available")
        self.assertEqual(media["sync_status"], "mapped")
        np.testing.assert_allclose(media["video_time_s"], [0., .1, .2])
        np.testing.assert_allclose(media["time_s"], [0., .1, .15])
        self.assertEqual(media["export_name"], "rollout.mp4")
        self.assertEqual(media["trial_time_origin_s"], 5.)
        self.assertEqual(len(media["provenance"]), 2)
        self.assertTrue(all(row["sha256"] for row in media["provenance"]))

    def test_local_rollout_without_binding_plays_independently(self):
        self.video("rollout.mp4")
        media = inspect_media(self.trial)
        self.assertEqual(media["status"], "available")
        self.assertEqual(media["sync_status"], "unverified")
        self.assertEqual(media["video_time_s"], [])

    def test_external_path_url_and_symlink_are_not_exported(self):
        outside = Path(self.tmp.name) / "outside.mp4"
        outside.write_bytes(b"\x00\x00\x00\x18ftypisom12345678")
        for candidate in (str(outside), "https://example.invalid/movie.mp4", "../outside.mp4"):
            self.trial.summary = {"video_path": candidate}
            media = inspect_media(self.trial)
            self.assertEqual(media["status"], "missing")
            self.assertIsNone(media["source_path"])
        (self.root / "rollout.mp4").symlink_to(outside)
        self.trial.summary = {}
        self.assertEqual(inspect_media(self.trial)["status"], "missing")

    def test_header_and_extension_must_describe_mp4(self):
        (self.root / "rollout.mp4").write_text("not a video")
        self.assertEqual(inspect_media(self.trial)["status"], "missing")

    def test_total_duration_does_not_replace_missing_fps(self):
        self.bundle(fps=None, playback_duration_s=.3)
        media = inspect_media(self.trial)
        self.assertEqual(media["sync_status"], "unverified")
        self.assertEqual(media["time_s"], [])

    def test_explicit_video_timestamps_do_not_require_fps(self):
        rows, _ = self.bundle(fps=None)
        for row, time in zip(rows, (0., .13, .28)):
            row["video_time_s"] = time
        self.frames(rows)
        media = inspect_media(self.trial)
        self.assertEqual(media["sync_status"], "mapped")
        self.assertEqual(media["video_time_s"], [0., .13, .28])

    def test_summary_can_supply_known_recorder_manifest(self):
        _, record = self.bundle()
        (self.root / "review/review.json").unlink()
        self.trial.summary = {"video": record}
        self.assertEqual(inspect_media(self.trial)["sync_status"], "mapped")

    def test_conflicting_summary_fps_blocks_sync(self):
        _, record = self.bundle()
        self.trial.summary = {"video": dict(record, fps=30)}
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")

    def test_wrong_task_or_trial_identity_blocks_sync(self):
        self.bundle(task="sponge-grasp")
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")
        self.bundle(seed=2)
        self.trial.summary = {"seed": 3}
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")

    def test_incomplete_recording_and_unverified_trial_clock_block_sync(self):
        self.bundle(completed=False)
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")
        self.bundle()
        self.trial.task_tactile_clock_aligned = False
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")

    def test_noncontiguous_indices_and_nonmonotonic_times_block_sync(self):
        rows, _ = self.bundle()
        rows[1]["frame"] = 4
        self.frames(rows)
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")
        rows, _ = self.bundle()
        rows[1]["simulation_time_s"] = 5.
        rows[1]["camera_pose_time_s"] = 5.
        self.frames(rows)
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")

    def test_camera_clock_mismatch_and_partial_explicit_pts_block_sync(self):
        rows, _ = self.bundle()
        rows[1]["camera_pose_time_s"] = 5.2
        self.frames(rows)
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")
        rows, _ = self.bundle()
        rows[0]["video_time_s"] = 0.
        self.frames(rows)
        self.assertEqual(inspect_media(self.trial)["sync_status"], "unverified")

    def test_repeat_inspection_deduplicates_provenance_and_detects_changes(self):
        self.bundle()
        first = inspect_media(self.trial)
        self.assertEqual(len(first["provenance"]), 2)
        second = inspect_media(self.trial)
        self.assertEqual(second["provenance"], [])
        self.assertEqual(len(self.trial.provenance), 2)
        (self.root / "review/frames.jsonl").write_text("{}")
        with self.assertRaises(InputChangedError):
            inspect_media(self.trial)


if __name__ == "__main__":
    unittest.main()
