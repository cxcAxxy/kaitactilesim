"""Offline migration preserves declared identities and never reads old copies."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_progress_experiment.adapters import load_reference_episode
from evaluation_v2.expert import export_reference, load_expert_report
from evaluation_unified.export import export_staged
from evaluation_workbench.export import _verify
from kaihand_tactile_env.shared.task_demo import DEFAULT_DEMO_ROOT
import evaluate_unified
import export_evaluation_workbench_suite as suite_cli


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def row(path, original, root):
    return {"path": str(path.relative_to(root)), "original_path": str(original),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


class EvaluationMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old_root = self.root / "old_nas"
        self.demo = self.root / "task_demo"
        self.source = self.old_root / "recording/episode.h5"
        self.source.parent.mkdir(parents=True)
        names = [f"hand_r_{finger}_link{link}" for finger, link in
                 (("thumb", 6), ("index", 4), ("middle", 4), ("ring", 4), ("pinky", 4))]
        with h5py.File(self.source, "w") as file:
            file.attrs.update(schema_version="kaihand_tactile_episode_v1", control_hz=100,
                metadata_json=json.dumps({"scene": "whiteboard-wipe", "episode_index": 0,
                                          "recording_contract": "whiteboard_wipe_shared_raw_v1"}),
                outcome_json=json.dumps({"success": True}))
            file.create_dataset("state/timestamp", data=np.arange(4) / 100)
            file.create_dataset("whiteboard_wipe/ink_remaining", data=[[1, 1], [1, .5], [.5, .5], [0, 0]])
            file.create_dataset("whiteboard_wipe/board_force_n", data=[0, 1, 2, 0])
            force = file.create_group("tactile_contact_force")
            force.attrs.update(force_unit="N", clock_semantics="native fixture seconds")
            force.create_dataset("timestamp", data=np.arange(4) / 100)
            force.create_dataset("normal_force_n", data=np.ones((4, 5)))
            force.create_dataset("link_names", data=np.array(names, dtype=h5py.string_dtype()))
        self.replay = self.demo / "_metadata/whiteboard-wipe/replay.json"
        self.old_replay = self.old_root / self.replay.relative_to(self.demo)
        write_json(self.replay, {"schema_version": "kaihand-offline-multimodal-replay-v1", "completed": True,
            "task": "whiteboard-wipe", "source_hdf5": str(self.source), "source_episode_index": 0,
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest()})
        self.old_replay.parent.mkdir(parents=True)
        shutil.copyfile(self.replay, self.old_replay)
        self.local_source = self.demo / "_raw/whiteboard-wipe/episode.h5"
        self.local_source.parent.mkdir(parents=True)
        shutil.copyfile(self.source, self.local_source)
        self.manifest = {"schema": "kaihand-task-demo-migration-v1", "original_root": str(self.old_root),
                         "files": [row(self.replay, self.old_replay, self.demo)],
                         "raw_sources": [row(self.local_source, self.source, self.demo)]}
        self.manifest_path = self.demo / "migration_manifest.json"
        write_json(self.manifest_path, self.manifest)

    def export(self, output=None):
        return export_reference([self.replay], output or self.root / "reference", reference_id="migration-fixture",
            declared_by="test curator", selection_rule="All explicit fixture attempts", selection_evidence="synthetic test",
            declare_expert_reference=True)

    def no_old_reads(self):
        """Guard Python, bounded os.open readers, and the native HDF reader."""
        stack = ExitStack()
        old_path_open, old_os_open, old_hdf = Path.open, os.open, h5py.File
        def check(path):
            if isinstance(path, (str, bytes, os.PathLike)):
                candidate = Path(os.fsdecode(path)).absolute()
                if candidate.is_relative_to(self.old_root):
                    raise AssertionError(f"Old NAS bytes were opened: {path}")
        def path_open(path, *args, **kwargs):
            check(path)
            return old_path_open(path, *args, **kwargs)
        def os_open(path, *args, **kwargs):
            check(path)
            return old_os_open(path, *args, **kwargs)
        def hdf_open(path, *args, **kwargs):
            check(path)
            return old_hdf(path, *args, **kwargs)
        stack.enter_context(patch.object(Path, "open", path_open))
        stack.enter_context(patch.object(os, "open", os_open))
        stack.enter_context(patch.object(h5py, "File", hdf_open))
        return stack

    def test_import_preserves_original_identity_and_pins_local_bytes(self):
        with self.no_old_reads():
            report = load_expert_report(self.replay)
        self.assertEqual(report["trial"]["source"], str(self.source))
        self.assertEqual(report["trial"]["resolved_source"], str(self.local_source))
        hdf = next(item for item in report["provenance"] if item["role"] == "expert_hdf5")
        self.assertEqual(hdf["path"], str(self.source))
        self.assertEqual(hdf["local_path"], str(self.local_source))
        self.assertNotIn("resolved_path", hdf)  # Not a filesystem/symlink alias.
        self.assertTrue(report["reference_import"]["source_sha256_verified"])
        self.assertEqual(len(report["provenance"]), 3)

    def test_export_then_progress_consumer_uses_only_local_bytes(self):
        with self.no_old_reads():
            result = self.export()
            path = self.root / "reference" / result["trials"][0]["report_path"]
            episode = load_reference_episode(path)
            _verify(episode["provenance"])  # Migration is not a generic alias exemption.
        self.assertEqual(episode["progress"], [0, .25, .5, 1])
        self.assertEqual(episode["metadata"]["source_hdf5"], str(self.source))
        self.assertEqual(episode["metadata"]["local_source_hdf5"], str(self.local_source))
        migrated = [item for item in episode["provenance"] if item.get("original_path") == str(self.source)]
        self.assertEqual(len(migrated), 1)
        self.assertEqual(migrated[0]["path"], str(self.local_source))
        membership = json.loads((path.parent.parent / "reference_manifest.json").read_text())
        self.assertEqual(membership["members"][0]["source"], str(self.source))

    def test_full_local_hdf_hash_not_size_or_old_copy_is_verified(self):
        with self.local_source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([byte[0] ^ 1]))
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_expert_report(self.replay)

    def test_missing_raw_mapping_refuses_existing_old_copy(self):
        self.manifest["raw_sources"] = []
        write_json(self.manifest_path, self.manifest)
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "refusing a NAS fallback"):
            load_expert_report(self.replay)

    def test_manifest_sha_mismatch_refuses_existing_old_copy(self):
        self.manifest["raw_sources"][0]["sha256"] = "f" * 64
        write_json(self.manifest_path, self.manifest)
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "SHA-256 disagrees"):
            load_expert_report(self.replay)

    def test_project_default_root_missing_manifest_never_becomes_legacy_nas(self):
        self.manifest_path.unlink()
        with patch("evaluation_v2.expert.DEFAULT_DEMO_ROOT", self.demo), self.no_old_reads():
            with self.assertRaisesRegex(ValueError, "requires migration_manifest.json"):
                load_expert_report(self.replay)

    def test_replay_bytes_are_bound_to_manifest_sha(self):
        self.replay.write_bytes(self.replay.read_bytes().replace(b'"whiteboard-wipe"', b'"whiteboard-fake"'))
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "SHA-256 disagrees"):
            load_expert_report(self.replay)

    def test_downstream_rejects_forged_local_path(self):
        result = self.export()
        path = self.root / "reference" / result["trials"][0]["report_path"]
        report = json.loads(path.read_text())
        next(item for item in report["provenance"] if item["role"] == "expert_hdf5")["local_path"] = str(self.source)
        write_json(path, report)
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "local source differs"):
            load_reference_episode(path)

    def test_downstream_rejects_changed_migration_manifest(self):
        result = self.export()
        path = self.root / "reference" / result["trials"][0]["report_path"]
        self.manifest["audit_note"] = "changed since reference export"
        write_json(self.manifest_path, self.manifest)
        with self.no_old_reads(), self.assertRaisesRegex(ValueError, "SHA256 or size mismatch"):
            load_reference_episode(path)

    def test_export_cannot_write_inside_local_raw_directory(self):
        with self.assertRaisesRegex(ValueError, "outside original"):
            self.export(self.local_source.parent / "reference")

    def test_default_root_is_project_local_in_python_and_both_clis(self):
        self.assertEqual(DEFAULT_DEMO_ROOT, Path(__file__).resolve().parents[1] / "task_demo")
        self.assertEqual(inspect.signature(export_staged).parameters["task_demo_root"].default, DEFAULT_DEMO_ROOT)
        batch = {"trial_count": 0, "task_count": 0, "coverage": {}, "mode_counts": {}, "evaluation_profile": "fixture"}
        with redirect_stdout(io.StringIO()), patch("evaluation_unified.export.export_staged", return_value=batch) as unified:
            evaluate_unified.main(["export", "--source-root", str(self.root), "--output", str(self.root / "out")])
        self.assertEqual(unified.call_args.kwargs["task_demo_root"], DEFAULT_DEMO_ROOT)
        with redirect_stdout(io.StringIO()), patch.object(suite_cli, "export_suite", return_value=batch) as suite:
            suite_cli.main(["--trial-dir", str(self.root), "--output", str(self.root / "out"),
                            "--previous-workbench", str(self.root / "previous")])
        self.assertEqual(suite.call_args.kwargs["task_demo_root"], DEFAULT_DEMO_ROOT)
