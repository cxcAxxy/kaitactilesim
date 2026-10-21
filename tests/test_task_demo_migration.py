"""Migration copies exact evidence bytes and never edits the archived source."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from migrate_task_demo import copy_bundle, verify_bundle


class TaskDemoMigrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "archive"
        self.source.mkdir()
        self.output = self.root / "project/task_demo"
        self.raw = self.root / "outside/raw/episode.h5"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"immutable Raw fixture")
        self.sha = hashlib.sha256(self.raw.read_bytes()).hexdigest()
        self.replay = self.source / "_metadata/task/review/replay.json"
        self.replay.parent.mkdir(parents=True)
        self.metadata = {"schema_version": "kaihand-offline-multimodal-replay-v1",
                         "completed": True, "task": "task", "source_hdf5": str(self.raw),
                         "source_sha256": self.sha}
        self.replay.write_text(json.dumps(self.metadata))
        self.video = self.source / "task.mp4"
        self.video.write_bytes(b"video bytes")
        (self.source / "alias.mp4").hardlink_to(self.video)

    def test_complete_copy_preserves_bytes_hardlinks_and_reads_without_sources(self):
        before = self.replay.read_bytes()
        result = copy_bundle(self.source, self.output)
        self.assertEqual(result["external_raw_count"], 1)
        self.assertEqual(result["source_writes"], 0)
        self.assertEqual(self.replay.read_bytes(), before)
        self.assertEqual((self.output / "task.mp4").stat().st_ino,
                         (self.output / "alias.mp4").stat().st_ino)
        self.assertEqual((self.output / self.replay.relative_to(self.source)).read_bytes(), before)
        self.source.rename(self.root / "unavailable_archive")
        self.raw.rename(self.raw.with_suffix(".unavailable"))
        verified = verify_bundle(self.output)
        self.assertEqual(verified["verified_files"], 4)
        self.assertEqual(verified["verified_replays"], 1)
        self.assertFalse(verified["source_access_required"])

    def test_overwrite_is_refused(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "existing.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            copy_bundle(self.source, self.output)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_declared_raw_checksum_mismatch_cannot_publish(self):
        self.metadata["source_sha256"] = "0" * 64
        self.replay.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "differs from replay SHA"):
            copy_bundle(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.raw.read_bytes(), b"immutable Raw fixture")

    def test_source_symlink_cannot_publish(self):
        (self.source / "linked.h5").symlink_to(self.raw)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            copy_bundle(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_copy_audit_detects_tampered_local_raw(self):
        copy_bundle(self.source, self.output)
        raw = next((self.output / "_raw_sources").rglob("*.h5"))
        raw.write_bytes(b"x" * len(b"immutable Raw fixture"))
        with self.assertRaisesRegex(ValueError, "SHA|hash|checksum"):
            verify_bundle(self.output)

    def test_verify_requires_manifest(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            verify_bundle(self.output)

    def test_verify_rejects_extra_payload(self):
        copy_bundle(self.source, self.output)
        (self.output / "unaudited.h5").write_bytes(b"not in copy manifest")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_bundle(self.output)

    def test_verify_cannot_fall_back_to_nas_when_raw_mapping_is_removed(self):
        copy_bundle(self.source, self.output)
        manifest = self.output / "migration_manifest.json"
        document = json.loads(manifest.read_text())
        row = document["raw_sources"].pop()
        (self.output / row["path"]).unlink()
        document["replays"][0]["source_local_path"] = str(self.raw)
        manifest.write_text(json.dumps(document))
        self.assertTrue(self.raw.is_file())
        with self.assertRaisesRegex(ValueError, "Raw is not bound"):
            verify_bundle(self.output)

    def test_verify_rejects_absolute_unlisted_replay_audit_path(self):
        copy_bundle(self.source, self.output)
        manifest = self.output / "migration_manifest.json"
        document = json.loads(manifest.read_text())
        document["replays"][0]["path"] = str(self.replay)
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "each copied replay"):
            verify_bundle(self.output)

    def test_verify_rejects_omitted_replay_audit(self):
        copy_bundle(self.source, self.output)
        manifest = self.output / "migration_manifest.json"
        document = json.loads(manifest.read_text())
        document["replays"] = []
        manifest.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "omitted"):
            verify_bundle(self.output)

    def test_output_parent_symlink_cannot_write_a_stage_into_the_source(self):
        alias = self.root / "source-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        before = sorted(self.source.rglob("*"))
        with self.assertRaisesRegex(ValueError, "Symlink"):
            copy_bundle(self.source, alias / "nested-output")
        self.assertEqual(sorted(self.source.rglob("*")), before)

    def test_published_root_inherits_source_directory_permissions(self):
        self.source.chmod(0o751)
        copy_bundle(self.source, self.output)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o751)

    def test_historical_internal_raw_is_counted_without_becoming_external(self):
        historical = self.source / "_metadata/archive_pre_update"
        historical.mkdir()
        internal_raw = historical / "episode.h5"
        internal_raw.write_bytes(b"historical internal Raw")
        historical_replay = dict(self.metadata, source_hdf5=str(internal_raw),
                                 source_sha256=hashlib.sha256(internal_raw.read_bytes()).hexdigest())
        (historical / "replay.json").write_text(json.dumps(historical_replay))
        result = copy_bundle(self.source, self.output)
        self.assertEqual(result["raw_count"], 2)
        self.assertEqual(result["external_raw_count"], 1)
        self.assertEqual(result["replay_count"], 2)
        self.source.rename(self.root / "unavailable_archive")
        self.raw.rename(self.raw.with_suffix(".unavailable"))
        verified = verify_bundle(self.output)
        self.assertEqual(verified["verified_replays"], 2)


if __name__ == "__main__":
    unittest.main()
