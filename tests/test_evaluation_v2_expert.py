"""Explicit HDF5 expert import; synthetic files verify no guesswork or writes."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.tactile import analyze_tactile
from evaluation_v2.expert import load_expert_report, export_reference
from evaluation_v2.baselines import compare_to_reference
import export_expert_reference as CLI


def fixture(root, *, name="one", success=True, tactile=True):
    root = root / name
    root.mkdir()
    source = root / "episode.h5"
    time = np.arange(31) / 100
    normal = np.column_stack([np.ones(31), np.arange(31) / 10])
    grid = np.broadcast_to(normal[:, :, None, None] / 4, (31, 2, 2, 2)).copy()
    positions = np.array([[[0, 0, 0], [.001, 0, 0], [0, .001, 0], [.001, .001, 0]]] * 2)
    with h5py.File(source, "w") as file:
        file.attrs.update(schema_version="kaihand_tactile_episode_v1", control_hz=100,
                          metadata_json=json.dumps({"scene": "synthetic-task", "episode_index": 0, "recording_id": name}),
                          outcome_json=json.dumps({"success": success}) if success is not None else "{}")
        file.create_dataset("state/timestamp", data=time)
        file.create_dataset("commands/phase", data=np.array([b"held"] * len(time)))
        if tactile:
            group = file.create_group("tactile_contact_force")
            group.attrs.update(force_unit="N", taxel_unit="N_per_taxel",
                               clock_semantics="synthetic fixture native seconds",
                               metadata_json=json.dumps({"force_unit": "N", "grid_shape": [2, 2],
                                   "tangent_basis": ["grid_col_positive", "grid_row_positive"]}))
            for key, values in {"timestamp": time, "normal_force_n": normal, "normal_taxel_force_n": grid,
                                "tangent_taxel_force_n": np.zeros((*grid.shape, 2)),
                                "taxel_positions_local_m": positions, "link_names": np.array([b"pad-A", b"pad-B"])}.items():
                group.create_dataset(key, data=values)
    replay = root / "replay.json"
    replay.write_text(json.dumps({"schema_version": "kaihand-offline-multimodal-replay-v1", "completed": True,
        "task": "synthetic-task", "source_hdf5": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_episode_index": 0}))
    return replay, source, time, normal, grid, positions


def refresh(replay):
    data = json.loads(replay.read_text())
    data["source_sha256"] = hashlib.sha256(Path(data["source_hdf5"]).read_bytes()).hexdigest()
    replay.write_text(json.dumps(data))


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class ExpertImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.replay, self.source, self.time, self.normal, self.grid, self.positions = fixture(self.root)

    def export(self, replay_paths=None, destination=None, **overrides):
        kwargs = dict(reference_id="explicit-test-set", declared_by="synthetic-test-curator",
                      selection_rule="All explicit attempts; no success filter", selection_evidence="Synthetic controlled fixture",
                      declare_expert_reference=True)
        kwargs.update(overrides)
        return export_reference(replay_paths or [self.replay], destination or self.root / "export", **kwargs)

    def test_real_native_values_match_v1_and_originals_remain_unchanged(self):
        before = snapshot(self.root)
        report = load_expert_report(self.replay)
        expected = analyze_tactile(self.time, self.normal, channel_names=["pad-A", "pad-B"],
            normal_taxel_force_n=self.grid, tangent_taxel_force_n=np.zeros((*self.grid.shape, 2)),
            taxel_positions_local_m=self.positions)
        self.assertEqual(report["tactile"], expected)
        self.assertEqual(snapshot(self.root), before)
        self.assertTrue(report["reference_import"]["source_sha256_verified"])
        self.assertAlmostEqual(report["reference_import"]["native_tactile_cadence_hz"], 100)
        self.assertEqual(report["reference_import"]["native_tactile_sample_count"], 31)
        self.assertEqual(report["outcome"]["status"], "pass")
        self.assertEqual(report["compatibility_hints"]["status"], "unverified")
        self.assertNotIn("reference_compatibility", report)

    def test_command_phases_are_not_independent_scopes(self):
        report = load_expert_report(self.replay)
        self.assertEqual(set(report["tactile"]["scopes"]), {"overall"})
        self.assertEqual(report["scope_intervals"], [])
        self.assertEqual(report["compatibility_hints"]["phase_conditions"], {})

    def test_missing_tactile_does_not_erase_outcome_or_invent_proxy_force(self):
        other = fixture(self.root, name="no_tactile", success=False, tactile=False)[0]
        report = load_expert_report(other)
        self.assertEqual(report["outcome"]["status"], "fail")
        self.assertEqual(report["tactile"]["status"], "missing")
        self.assertEqual(report["tactile"]["scopes"], {})
        self.assertIsNone(report["reference_import"]["native_tactile_cadence_hz"])

    def test_missing_or_conflicting_boolean_outcome_is_unknown(self):
        for outcome in ({}, {"success": True, "task_success": False}, {"success": "true"}):
            with self.subTest(outcome=outcome):
                with h5py.File(self.source, "r+") as file: file.attrs["outcome_json"] = json.dumps(outcome)
                refresh(self.replay)
                self.assertEqual(load_expert_report(self.replay)["outcome"]["status"], "unknown")

    def test_no_unit_guess_and_no_normal_reconstruction_from_taxels(self):
        with h5py.File(self.source, "r+") as file: file["tactile_contact_force"].attrs["force_unit"] = "counts"
        refresh(self.replay)
        self.assertEqual(load_expert_report(self.replay)["tactile"]["status"], "missing")
        with h5py.File(self.source, "r+") as file:
            file["tactile_contact_force"].attrs["force_unit"] = "N"
            del file["tactile_contact_force/normal_force_n"]
        refresh(self.replay)
        self.assertEqual(load_expert_report(self.replay)["tactile"]["status"], "missing")

    def test_unknown_geometry_does_not_erase_normal_force(self):
        with h5py.File(self.source, "r+") as file: del file["tactile_contact_force/taxel_positions_local_m"]
        refresh(self.replay)
        report = load_expert_report(self.replay)
        self.assertEqual(report["tactile"]["scopes"]["overall"]["channels"]["pad-A"]["intensity"]["all_observed"]["mean_n"], 1)
        self.assertEqual(report["evidence_quality"]["taxel_geometry_status"], "missing")
        self.assertEqual(report["reference_import"]["geometry_sha256_by_channel"], {})

    def test_undeclared_tangent_basis_stays_unknown_not_relabelled_xy(self):
        with h5py.File(self.source, "r+") as file:
            file["tactile_contact_force"].attrs["metadata_json"] = json.dumps({"force_unit": "N"})
        refresh(self.replay)
        report = load_expert_report(self.replay)
        self.assertEqual(report["tactile"]["scopes"]["overall"]["channels"]["pad-A"]["tangent"]["status"], "missing")

    def test_invalid_native_clock_is_not_replaced_by_state_clock(self):
        with h5py.File(self.source, "r+") as file: file["tactile_contact_force/timestamp"][1] = 0
        refresh(self.replay)
        report = load_expert_report(self.replay)
        self.assertEqual(report["tactile"]["status"], "missing")
        self.assertAlmostEqual(report["efficiency"]["simulation_duration_s"], .3)

    def test_missing_force_samples_remain_unknown_not_zero(self):
        with h5py.File(self.source, "r+") as file: file["tactile_contact_force/normal_force_n"][10, 0] = float("nan")
        refresh(self.replay)
        report = load_expert_report(self.replay)
        channel = report["tactile"]["scopes"]["overall"]["channels"]["pad-A"]
        self.assertEqual(channel["status"], "partial")
        self.assertEqual(channel["intensity"]["all_observed"]["mean_n"], 1)
        json.dumps(report, allow_nan=False)

    def test_replay_schema_completed_sha_and_task_binding_required(self):
        original = json.loads(self.replay.read_text())
        mutations = [{"schema_version": "unknown"}, {"completed": False},
                     {"source_sha256": "0" * 64}, {"task": "other-task"},
                     {"source_hdf5": "relative.h5"}, {"source_episode_index": 4}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.replay.write_text(json.dumps({**original, **mutation}))
                with self.assertRaises(ValueError): load_expert_report(self.replay)

    def test_unsupported_hdf_schema_is_not_guessed(self):
        with h5py.File(self.source, "r+") as file: file.attrs["schema_version"] = "different-recording-format"
        refresh(self.replay)
        with self.assertRaisesRegex(ValueError, "Unsupported HDF5 schema"):
            load_expert_report(self.replay)

    def test_external_and_soft_links_are_not_unhashed_evidence(self):
        external = self.root / "external.h5"
        with h5py.File(external, "w") as file: file.create_dataset("force", data=self.normal)
        with h5py.File(self.source, "r+") as file:
            group = file["tactile_contact_force"]
            del group["normal_force_n"]
            group["normal_force_n"] = h5py.ExternalLink(str(external), "force")
        refresh(self.replay)
        with self.assertRaisesRegex(ValueError, "Linked HDF5 evidence"):
            load_expert_report(self.replay)

    def test_same_file_cannot_be_replayed_as_three_independent_references(self):
        copy = self.root / "second_replay.json"
        copy.write_bytes(self.replay.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate source recording"):
            self.export([self.replay, copy])
        self.assertFalse((self.root / "export").exists())

    def test_copied_identical_hdf_hash_is_also_duplicate(self):
        other = self.root / "copy"
        other.mkdir()
        copied = other / "episode.h5"
        shutil.copyfile(self.source, copied)
        data = json.loads(self.replay.read_text())
        data["source_hdf5"] = str(copied)
        replay = other / "replay.json"
        replay.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Duplicate source recording"):
            self.export([self.replay, replay])

    def test_reference_declaration_is_not_automatically_enabled_by_success(self):
        with self.assertRaisesRegex(ValueError, "declare-expert-reference"):
            self.export(declare_expert_reference=False)
        self.assertFalse((self.root / "export").exists())

    def test_export_preserves_fail_unknown_and_writes_provenance_bound_manifest(self):
        fail = fixture(self.root, name="failure", success=False)[0]
        unknown = fixture(self.root, name="unknown", success=None)[0]
        before = {str(path): path.read_bytes() for path in [self.replay, self.source, fail, unknown]}
        result = self.export([self.replay, fail, unknown])
        self.assertEqual(result["independent_recording_count"], 3)
        self.assertEqual(result["outcome_counts"], {"pass": 1, "fail": 1, "unknown": 1})
        manifest = json.loads((self.root / "export/reference_manifest.json").read_text())
        self.assertEqual(len(manifest["members"]), 3)
        references = [json.loads((self.root / "export" / row["report_path"]).read_text()) for row in result["trials"]]
        self.assertEqual(len({row["trial"]["trial_id"] for row in references}), 3)
        for path, raw in before.items(): self.assertEqual(Path(path).read_bytes(), raw)
        # An import does not magically certify phase, calibration or task matching.
        candidate = deepcopy(references[0])
        candidate["trial"].update(trial_id="different", source="/other/source")
        candidate["provenance"][-1].update(path="/other/source.h5", sha256="a" * 64)
        comparison = compare_to_reference([candidate], references, reference_manifest=manifest)
        self.assertEqual(comparison["status"], "unavailable")
        self.assertIn("compatibility_unverified", comparison["trials"][0]["reasons"])

        # A future audited sidecar can add compatibility without rewriting the
        # hashed source report. Unverified import hints are not declarations.
        contexts = {}
        for row in references + [candidate]:
            context = deepcopy(row["compatibility_hints"])
            context.update(status="verified", task_definition_id="synthetic-task-v1", sensor_semantics_id="synthetic-sensor-v1")
            for geometry in context["channels"].values(): geometry["status"] = "verified"
            context["phase_conditions"] = {"overall": {"status": "verified", "condition_id": "whole-recording",
                "definition_sha256": "b" * 64, "independent_of_tactile": True,
                "evidence": ["Synthetic fixture bounds independently declared by the test"]}}
            contexts[row["trial"]["trial_id"]] = context
        audited = compare_to_reference([candidate], references, reference_manifest=manifest, compatibility=contexts)
        self.assertEqual(audited["status"], "available")
        self.assertNotIn("reference_compatibility", candidate)

    def test_export_refuses_overwrite_and_writes_inside_original_directories(self):
        self.export()
        before = snapshot(self.root / "export")
        with self.assertRaises(FileExistsError): self.export()
        self.assertEqual(snapshot(self.root / "export"), before)
        with self.assertRaisesRegex(ValueError, "outside original"):
            self.export(destination=self.source.parent / "new-reference")
        self.assertFalse((self.source.parent / "new-reference").exists())

    def test_cli_requires_membership_flag_and_exports_without_source_changes(self):
        arguments = ["--replay-json", str(self.replay), "--output-dir", str(self.root / "cli"),
                     "--reference-id", "test", "--declared-by", "test", "--selection-rule", "all fixtures",
                     "--selection-evidence", "synthetic fixtures"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            CLI.main(arguments)
        self.assertEqual(error.exception.code, 2)
        out = io.StringIO()
        before = snapshot(self.source.parent)
        with redirect_stdout(out): self.assertEqual(CLI.main(arguments + ["--declare-expert-reference"]), 0)
        self.assertEqual(json.loads(out.getvalue())["independent_recording_count"], 1)
        self.assertEqual(snapshot(self.source.parent), before)


if __name__ == "__main__":
    unittest.main()
