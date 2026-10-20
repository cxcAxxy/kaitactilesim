"""Synthetic population, compatibility and leakage checks; never replay a trial."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/evaluation_v2/baselines.py"
SPEC = importlib.util.spec_from_file_location("v2_baselines_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compare = MODULE.compare_to_reference


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def report(name, value, *, outcome="pass", scope_status="available", channel="finger"):
    protocol = {"schema_version": "test-protocol", "analysis_bin_s": .1}
    return {
        "schema_version": "task-tactile-report-v1",
        "trial": {"trial_id": name, "task": "task-a", "source": f"/archive/{name}",
                  "cohort": {"tactile_sample_interval_s": .01}},
        "protocol": protocol, "protocol_sha256": MODULE._hash(protocol),
        "provenance": [{"path": f"/archive/{name}/tactile.npz", "sha256": digest(name),
                        "role": "native_control_cadence_tactile"}],
        "outcome": {"status": outcome, "formal_metrics_valid": True, "diagnostic_only": False},
        "reference_compatibility": {
            "status": "verified", "task_definition_id": "task-a-contract-1",
            "sensor_cadence_hz": 100, "force_unit": "N", "sensor_semantics_id": "declared-physics-1",
            "channels": {channel: {"status": "verified", "geometry_sha256": digest("same geometry")}},
            "phase_conditions": {"held": {"status": "verified", "condition_id": "object-held",
                "definition_sha256": digest("held-condition-1"), "independent_of_tactile": True,
                "evidence": ["independent object pose/support label with verified common clock"]}},
        },
        "tactile": {"schema_version": "tactile-metrics-v1", "scopes": {
            "held": {"status": scope_status, "duration_s": 5, "sample_count": 501,
                "channels": {channel: {"status": "available", "intensity": {
                    "all_observed": {"mean_n": value, "p95_n": value},
                    "sampled_peak_n": value, "contact": {"mean_n": value}},
                    "change": {"status": "available", "rms_change_n": value},
                    "continuity": {"status": "available", "observed_contact_duration_s": value}}}}}},
    }


def manifest(references):
    return {"schema_version": "conditional-reference-manifest-v1", "reference_id": "set-a",
            "role": "expert_reference", "declared_by": "test-curator", "evidence": "explicit synthetic manifest",
            "selection_rule": "all declared attempts, including failures and unknown outcomes",
            "members": [{"trial_id": row["trial"]["trial_id"], "source": row["trial"]["source"],
                         "observation_sha256": row["provenance"][0]["sha256"]} for row in references]}


def run(candidate=None, references=None, **options):
    references = references if references is not None else [report("r1", 1), report("r2", 2), report("r3", 9)]
    candidates = candidate if isinstance(candidate, list) else [candidate if candidate is not None else report("candidate", 2)]
    return compare(candidates, references, reference_manifest=options.pop("reference_manifest", manifest(references)), **options)


class ConditionalBaselinesTest(unittest.TestCase):
    def test_episode_first_quantiles_and_tied_empirical_interval(self):
        result = run()
        self.assertEqual(result["status"], "available")
        item = result["trials"][0]["metrics"][0]
        self.assertEqual(item["percentile_rank_interval"], [1 / 3, 2 / 3])
        group = result["groups"][item["reference_group_id"]]
        self.assertEqual(group["reference_episode_count"], 3)
        self.assertEqual(group["statistics"]["median"], 2)
        self.assertEqual(group["statistics"]["mad"], 1)
        self.assertEqual(group["statistics"]["p25"], 1.5)
        self.assertAlmostEqual(group["statistics"]["p95"], 8.3)
        self.assertNotIn("score", result)
        self.assertEqual(result["time_alignment"], "none")

    def test_long_episode_does_not_get_more_reference_weight(self):
        references = [report("r1", 0), report("r2", 1), report("r3", 10)]
        references[2]["tactile"]["scopes"]["held"].update(sample_count=1_000_000, duration_s=10000)
        result = run(references=references)
        self.assertEqual(next(iter(result["groups"].values()))["statistics"]["median"], 1)

    def test_fail_unknown_reference_and_candidate_rows_are_retained(self):
        refs = [report("r1", 1, outcome="pass"), report("r2", 2, outcome="fail"), report("r3", 3, outcome="unknown")]
        candidates = [report("c1", 1, outcome="fail"), report("c2", 2, outcome="unknown")]
        result = run(candidates, refs)
        self.assertEqual(result["reference_population"]["accepted_outcome_counts"], {"pass": 1, "fail": 1, "unknown": 1})
        self.assertEqual(result["candidate_population"]["outcome_counts"], {"pass": 0, "fail": 1, "unknown": 1})
        self.assertEqual([r["outcome"]["status"] for r in result["trials"]], ["fail", "unknown"])
        self.assertEqual(result["status"], "available")

    def test_unreached_reference_phase_stays_in_compatible_denominator(self):
        refs = [report("r1", 1), report("r2", 2), report("r3", 3), report("r4", None, outcome="fail", scope_status="not_reached")]
        refs[-1]["tactile"]["scopes"]["held"]["channels"] = None
        result = run(references=refs)
        group = next(iter(result["groups"].values()))
        self.assertEqual(group["compatible_episode_count"], 4)
        self.assertEqual(group["reference_episode_count"], 3)
        self.assertEqual(group["missing_reasons"], {"scope_not_reached": 1})
        self.assertEqual(group["outcome_counts"]["fail"], 1)

    def test_unreached_candidate_remains_unavailable_not_silently_dropped(self):
        candidate = report("c", None, outcome="fail", scope_status="not_reached")
        result = run(candidate)
        self.assertEqual(len(result["trials"]), 1)
        self.assertEqual(result["trials"][0]["status"], "unavailable")
        self.assertIn("scope_not_reached", result["trials"][0]["reasons"])

    def test_no_manifest_never_infers_reference_from_success(self):
        result = run(reference_manifest=None)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reference_population"]["accepted_report_count"], 0)
        self.assertIn("reference_manifest_missing", result["trials"][0]["reasons"])
        self.assertEqual(result["groups"], {})

    def test_noncanonical_manifest_and_null_provenance_refuse_comparison(self):
        refs = [report("r1", 1), report("r2", 2), report("r3", 3)]
        declared = manifest(refs)
        declared["untrusted_metadata"] = float("nan")
        result = run(references=refs, reference_manifest=declared)
        self.assertIn("reference_manifest_not_canonical_json", result["trials"][0]["reasons"])
        candidate = report("c", 2)
        candidate["provenance"] = None
        self.assertIn("observation_identity_unverified", run(candidate)["trials"][0]["reasons"])

    def test_explicit_membership_is_hash_and_path_bound(self):
        refs = [report("r1", 1), report("r2", 2), report("r3", 3)]
        declared = manifest(refs)
        declared["members"][0]["observation_sha256"] = digest("wrong")
        declared["members"][1]["source"] = "/different/location"
        result = run(references=refs, reference_manifest=declared)
        self.assertEqual(result["reference_population"]["accepted_report_count"], 1)
        for row in result["reference_trials"][:2]:
            self.assertIn("reference_membership_provenance_mismatch", row["reasons"])

    def test_missing_declared_member_and_extra_report_visible(self):
        refs = [report("r1", 1), report("r2", 2), report("extra", 3)]
        declared = manifest([refs[0], refs[1], report("absent", 4)])
        result = run(references=refs, reference_manifest=declared)
        self.assertEqual(result["reference_population"]["missing_member_ids"], ["absent"])
        self.assertIn("not_a_declared_reference_member", result["reference_trials"][2]["reasons"])

    def test_same_hash_under_different_path_is_leakage(self):
        candidate = report("copy", 2)
        candidate["provenance"][0]["sha256"] = digest("r1")
        self.assertIn("candidate_reference_source_overlap", run(candidate)["trials"][0]["reasons"])

    def test_same_normalized_source_or_observation_path_is_leakage(self):
        for change in ("source", "observation"):
            with self.subTest(change=change):
                candidate = report("copy", 2)
                if change == "source": candidate["trial"]["source"] = "/archive/sub/../r1"
                else: candidate["provenance"][0]["path"] = "/archive/r1/tactile.npz"
                self.assertIn("candidate_reference_source_overlap", run(candidate)["trials"][0]["reasons"])

    def test_duplicate_references_do_not_inflate_episode_count(self):
        refs = [report("r1", 1), report("copy", 1), report("r3", 3)]
        refs[1]["provenance"][0]["sha256"] = refs[0]["provenance"][0]["sha256"]
        result = run(references=refs)
        self.assertEqual(result["reference_population"]["accepted_report_count"], 1)
        self.assertIn("duplicate_reference_observation", result["reference_trials"][0]["reasons"])
        self.assertEqual(result["trials"][0]["status"], "unavailable")

    def test_duplicate_candidates_remain_visible_but_unavailable(self):
        candidates = [report("c1", 1, outcome="fail"), report("c2", 1, outcome="unknown")]
        candidates[1]["provenance"][0]["sha256"] = candidates[0]["provenance"][0]["sha256"]
        result = run(candidates)
        self.assertEqual(len(result["trials"]), 2)
        for row in result["trials"]:
            self.assertIn("duplicate_candidate_observation", row["reasons"])

    def test_shared_non_observation_config_hash_is_not_source_overlap(self):
        refs = [report("r1", 1), report("r2", 2), report("r3", 3)]
        candidate = report("c", 2)
        for row in refs + [candidate]:
            row["provenance"].append({"path": "/shared/config.json", "sha256": digest("config"), "role": "config"})
        self.assertEqual(run(candidate, refs)["status"], "available")

    def test_uncertified_legacy_report_refuses_comparison(self):
        candidate = report("c", 2)
        del candidate["reference_compatibility"]
        self.assertIn("compatibility_unverified", run(candidate)["trials"][0]["reasons"])

    def test_sidecar_supports_v1_without_rewriting_it(self):
        candidate = report("c", 2)
        context = candidate.pop("reference_compatibility")
        original = deepcopy(candidate)
        result = run(candidate, compatibility={"c": context})
        self.assertEqual(result["status"], "available")
        self.assertEqual(candidate, original)

    def test_embedded_and_sidecar_disagreement_is_not_overridden(self):
        candidate = report("c", 2)
        context = deepcopy(candidate["reference_compatibility"])
        context["sensor_semantics_id"] = "different"
        self.assertIn("conflicting_compatibility_declarations", run(candidate, compatibility={"c": context})["trials"][0]["reasons"])

    def test_geometry_task_sensor_phase_and_cadence_split_groups(self):
        changes = {
            "geometry": lambda c: c["channels"]["finger"].update(geometry_sha256=digest("other")),
            "task": lambda c: c.update(task_definition_id="different-task-definition"),
            "sensor": lambda c: c.update(sensor_semantics_id="different-bandwidth"),
            "phase": lambda c: c["phase_conditions"]["held"].update(definition_sha256=digest("different-phase")),
            "cadence": lambda c: c.update(sensor_cadence_hz=30),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                candidate = report("c", 2)
                change(candidate["reference_compatibility"])
                if name == "cadence": candidate["trial"]["cohort"]["tactile_sample_interval_s"] = 1 / 30
                row = run(candidate)["trials"][0]
                self.assertEqual(row["status"], "unavailable")
                self.assertIn("no_compatible_reference_group", row["reasons"])

    def test_observed_contact_is_not_independent_phase_truth(self):
        candidate = report("c", 2)
        candidate["reference_compatibility"]["phase_conditions"]["held"]["independent_of_tactile"] = False
        self.assertIn("independent_phase_condition_unverified", run(candidate)["trials"][0]["reasons"])

    def test_incomplete_channel_is_not_treated_as_a_complete_conditional_metric(self):
        candidate = report("c", 2)
        candidate["tactile"]["scopes"]["held"]["channels"]["finger"]["status"] = "partial"
        self.assertIn("channel_partial", run(candidate)["trials"][0]["reasons"])

    def test_diagnostic_and_formal_contexts_are_never_mixed(self):
        candidate = report("c", 2)
        candidate["outcome"].update(diagnostic_only=True, formal_metrics_valid=False)
        self.assertIn("no_compatible_reference_group", run(candidate)["trials"][0]["reasons"])

    def test_nonfinite_bool_null_are_missing_but_zero_is_observed(self):
        for value in (None, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                result = run(report("c", value))
                self.assertEqual(result["trials"][0]["status"], "unavailable")
                json.dumps(result, allow_nan=False)
        result = run(report("c", 0))
        self.assertEqual(result["trials"][0]["metrics"][0]["value"], 0)
        self.assertEqual(result["trials"][0]["metrics"][0]["percentile_rank_interval"], [0, 0])

    def test_insufficient_reference_is_not_a_population_band(self):
        result = run(references=[report("r1", 1), report("r2", 2)])
        group = next(iter(result["groups"].values()))
        self.assertEqual(group["status"], "insufficient_reference_episodes")
        self.assertIsNone(group["statistics"])
        self.assertIsNone(result["trials"][0]["metrics"][0]["percentile_rank_interval"])

    def test_protocol_hash_and_recorded_cadence_conflicts_rejected(self):
        candidate = report("c", 2)
        candidate["protocol"]["analysis_bin_s"] = 1
        self.assertIn("metric_protocol_hash_mismatch", run(candidate)["trials"][0]["reasons"])
        candidate = report("c", 2)
        candidate["reference_compatibility"]["sensor_cadence_hz"] = 30
        self.assertIn("sensor_cadence_conflicts_with_report", run(candidate)["trials"][0]["reasons"])

    def test_arbitrary_channels_and_additive_v2_report_supported(self):
        refs = [report("r1", 1, channel="not-a-finger"), report("r2", 2, channel="not-a-finger"), report("r3", 3, channel="not-a-finger")]
        candidate = report("c", 2, channel="not-a-finger")
        candidate["schema_version"] = "task-tactile-report-v2"
        candidate["expected_contact"] = {"status": "missing"}
        result = run(candidate, refs)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["trials"][0]["metrics"][0]["channel"], "not-a-finger")

    def test_input_immutable_and_optional_custom_dictionary_path(self):
        candidate = report("c", 2)
        original = deepcopy(candidate)
        result = run(candidate, metric_specs=[{"name": "custom", "path": ["intensity", "all_observed", "mean_n"], "unit": "N"}])
        self.assertEqual(len(result["trials"][0]["metrics"]), 1)
        self.assertEqual(candidate, original)

    def test_invalid_api_arguments_rejected(self):
        for value in (0, 1, True, 2.5):
            with self.assertRaises(ValueError): run(minimum_reference_episodes=value)
        with self.assertRaises(ValueError): run(metric_specs=[])
        with self.assertRaises(ValueError): run(metric_specs=[{"name": "x", "path": "foo.bar", "unit": "N"}])
        with self.assertRaises(ValueError): compare([None], [])


if __name__ == "__main__":
    unittest.main()
