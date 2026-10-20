"""A new presentation snapshot never rewrites its preserved H50 parent."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
import evaluation_goal_reference_snapshot as snapshot
from evaluation_workbench.export import _digest, _document, _embedded, _files
from evaluation_workbench.suite import _page


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parent_site(tmp_path, task="bulb-screw", horizon=50):
    source = tmp_path / "preserved"
    episode = source / "000_task_seed_000"
    episode.mkdir(parents=True)
    visual = {"clock_aligned": True, "time_origin_s": 0., "task_series": [
        {"key": "clockwise_turns", "unit": "turn", "time_s": [0, .1, .2],
         "values": [0, None, .02], "gap_after": [False, True, False]},
        {"key": "hand_clockwise_torque_nm", "unit": "N·m", "time_s": [0], "values": [None]},
    ]}
    report = {"trial": {"task": task, "trial_id": "one", "prediction_horizon": horizon},
              "outcome": {"status": "fail"}, "visualization": {"sha256": "archived"},
              "recorded_staged_evaluation": {"stages": [{"reached": False}]}}
    config = {"task": task, "seed": 0, "task_summary": []}
    loads = {"normal_total_n": [0, None, 1]}
    for filename, value in (("report.json", report), ("visualization.json", visual),
                            ("workbench.json", config), ("tactile_loads.json", loads)):
        json_file(episode / filename, value)
    (episode / "report.html").write_text(_page(report, visual, loads, config), encoding="utf-8")
    (episode / "media").mkdir()
    (episode / "media/model.mp4").write_bytes(b"independent-media-test")
    (source / "assets").mkdir()
    (source / "assets/dashboard.js").write_text("legacy-js")
    (source / "assets/dashboard.css").write_text("legacy-css")
    json_file(source / "verification.json", {"historical_source_verification": True})
    json_file(source / "batch.json", {"evaluation_profile": "recorded-staged", "prediction_horizon": horizon,
        "trials": [{"task": task, "report_path": "000_task_seed_000/report.html", "prediction_horizon": horizon}],
        "trial_count": 1, "previous_url": "../old/index.html"})
    json_file(source / "artifact_manifest.json", {"files": [_digest(p) for p in _files(source)]})
    return source, episode, report, visual


def goals(task, report):
    return {"schema_version": "evaluation-goal-references-v1", "task": task, "status": "verified",
            "references": [{"key": "clockwise_turns", "unit": "turn", "target": 100 / 360,
                            "lower": 100 / 360 - .08, "upper": 100 / 360 + .08,
                            "relation": "target", "label": "100°", "note": "Not task success"}],
            "warnings": [], "provenance": []}


@pytest.mark.parametrize("turn_key,degree_key", (
    ("clockwise_turns", "clockwise_angle_deg"),
    ("turns_removed", "angle_removed_deg"),
    ("rotation_from_initial_turns", "rotation_from_initial_deg"),
))
def test_degree_references_attach_to_legacy_series_without_mutating_inputs(turn_key, degree_key):
    reference = {"key": degree_key, "unit": "deg", "target": 100.,
                 "lower": 71.2, "upper": 128.8, "relation": "target"}
    series = {"key": turn_key, "unit": "turn", "values": [0., None, .25]}
    before = deepcopy((reference, series))
    attached, = snapshot._references_for_series([reference], series)
    assert attached["key"] == turn_key and attached["unit"] == "turn"
    for key in ("target", "lower", "upper"):
        assert attached[key] == pytest.approx(reference[key] / 360)
    converted, = snapshot._references_for_series([attached], {"key": degree_key, "unit": "deg"})
    assert converted == reference
    assert (reference, series) == before
    assert snapshot._references_for_series([reference], {"key": degree_key, "unit": "rad"}) == []


def test_degree_goal_snapshot_keeps_archived_turn_samples(tmp_path, monkeypatch):
    source, episode, _, visual = parent_site(tmp_path)
    def degree_goals(task, report):
        result = goals(task, report)
        result["references"] = [{"key": "clockwise_angle_deg", "unit": "deg", "target": 100.,
                                  "lower": 71.2, "upper": 128.8, "relation": "target", "label": "100°"}]
        return result
    monkeypatch.setattr(snapshot, "build_goal_references", degree_goals)
    output = tmp_path / "degrees"
    snapshot.export_snapshot(source, output)
    new_visual = _document(output / episode.name / "visualization.json")
    assert snapshot._strip_references(new_visual) == visual
    reference, = new_visual["task_series"][0]["goal_references"]
    assert reference["key"] == "clockwise_turns" and reference["unit"] == "turn"
    assert reference["target"] == pytest.approx(100 / 360)
    assert _document(output / episode.name / "goal_references.json")["references"][0]["unit"] == "deg"
    assert snapshot.verify_snapshot(output)["parent_unchanged"] is True


def test_new_snapshot_preserves_native_evidence_and_parent_and_independently_copies_media(tmp_path, monkeypatch):
    source, episode, report, visual = parent_site(tmp_path)
    before = {p: p.read_bytes() for p in _files(source)}
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    output = tmp_path / "annotated"
    result = snapshot.export_snapshot(source, output)
    assert result["status_counts"]["verified"] == 1
    assert {p: p.read_bytes() for p in _files(source)} == before
    target = output / episode.name
    new_report, new_visual = _document(target / "report.json"), _document(target / "visualization.json")
    snapshot._same_native_evidence(report, new_report)
    assert snapshot._strip_references(new_visual) == visual
    assert new_visual["task_series"][0]["values"] == [0, None, .02]
    assert len(new_visual["task_series"][0]["goal_references"]) == 1
    assert "goal_references" not in new_visual["task_series"][1]
    _, embedded = _embedded((target / "report.html").read_text(), "kai-evaluation-data")
    assert embedded == {"report": new_report, "visualization": new_visual}
    assert (target / "media/model.mp4").read_bytes() == b"independent-media-test"
    assert (target / "media/model.mp4").stat().st_ino != (episode / "media/model.mp4").stat().st_ino
    assert not any(p.is_symlink() for p in output.rglob("*"))
    assert _document(output / "verification.json")["source_projects_reverified"] is False
    assert snapshot.verify_snapshot(output)["parent_unchanged"] is True


def test_missing_target_sources_retains_unknown_without_fabricating_lines(tmp_path, monkeypatch):
    source, episode, _, visual = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", lambda task, report: {
        **goals(task, report), "status": "unavailable", "references": [], "warnings": ["missing binding"]})
    output = tmp_path / "unknown"
    result = snapshot.export_snapshot(source, output)
    assert result["status_counts"]["unavailable"] == 1
    assert _document(output / episode.name / "visualization.json") == visual


def test_other_tasks_are_copied_without_inventing_goal_references(tmp_path, monkeypatch):
    source, episode, _, _ = parent_site(tmp_path, task="vase-wipe")
    monkeypatch.setattr(snapshot, "build_goal_references", lambda *_: pytest.fail("No inferred goals for other tasks"))
    result = snapshot.export_snapshot(source, tmp_path / "copy")
    assert result["status_counts"]["not_applicable"] == 1
    assert not (tmp_path / "copy" / episode.name / "goal_references.json").exists()
    assert (tmp_path / "copy" / episode.name / "report.html").read_bytes() == (episode / "report.html").read_bytes()


def test_existing_output_is_never_overwritten_even_if_empty(tmp_path):
    source, *_ = parent_site(tmp_path)
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        snapshot.export_snapshot(source, output)
    assert not list(output.iterdir())


@pytest.mark.parametrize("horizon", (30, None))
def test_only_explicit_h50_snapshots_are_admitted(tmp_path, horizon):
    source, *_ = parent_site(tmp_path, horizon=horizon)
    with pytest.raises(ValueError, match="H50"):
        snapshot.export_snapshot(source, tmp_path / "bad")


def test_modified_parent_artifact_stops_before_publication(tmp_path):
    source, episode, *_ = parent_site(tmp_path)
    (episode / "report.json").write_text("{}")
    with pytest.raises(ValueError):
        snapshot.export_snapshot(source, tmp_path / "not_published")
    assert not (tmp_path / "not_published").exists()


def test_inconsistent_embedded_evidence_is_not_silently_replaced(tmp_path, monkeypatch):
    source, episode, report, visual = parent_site(tmp_path)
    other = deepcopy(report)
    other["outcome"]["status"] = "pass"
    (episode / "report.html").write_text(_page(other, visual, {}, {}))
    json_file(source / "artifact_manifest.json", {"files": [
        _digest(p) for p in _files(source) if p.name != "artifact_manifest.json"]})
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    with pytest.raises(ValueError, match="embedded"):
        snapshot.export_snapshot(source, tmp_path / "inconsistent")
    assert not (tmp_path / "inconsistent").exists()


def test_verification_detects_modified_new_artifact(tmp_path, monkeypatch):
    source, episode, *_ = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    output = tmp_path / "result"
    snapshot.export_snapshot(source, output)
    (output / episode.name / "visualization.json").write_text("{}")
    with pytest.raises(ValueError):
        snapshot.verify_snapshot(output)


def test_verification_detects_unlisted_output_and_preserved_parent_change(tmp_path, monkeypatch):
    source, episode, *_ = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    output = tmp_path / "result"
    snapshot.export_snapshot(source, output)
    (source / episode.name / "media/model.mp4").write_bytes(b"externally changed")
    with pytest.raises(ValueError):
        snapshot.verify_snapshot(output)


def test_trial_path_cannot_escape_parent(tmp_path):
    source, *_ = parent_site(tmp_path)
    with pytest.raises(ValueError, match="path"):
        snapshot._trial_directory(source, {"report_path": "../elsewhere/report.html"})


def test_new_parent_file_cannot_be_claimed_unchanged(tmp_path, monkeypatch):
    source, *_ = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    output = tmp_path / "result"
    snapshot.export_snapshot(source, output)
    (source / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        snapshot.verify_snapshot(output)


def test_parent_export_does_not_accept_unlisted_evidence(tmp_path):
    source, *_ = parent_site(tmp_path)
    (source / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        snapshot.export_snapshot(source, tmp_path / "result")


def test_native_outcome_changes_cannot_hide_behind_visualization_metadata():
    before = {"outcome": {"status": "fail"}, "visualization": {"sha256": "a"}}
    after = deepcopy(before)
    after["outcome"]["status"] = "pass"
    with pytest.raises(ValueError, match="native"):
        snapshot._same_native_evidence(before, after)


@pytest.mark.parametrize("task,keys,unit,targets,relations", (
    ("install-ram", ("insertion_depth_m", "orientation_error_rad"), ("m", "rad"),
     (.006, .017453292519943295), ("target", "at_most")),
    ("poker-draw", ("card_displacement_now_mm", "card_displacement_peak_mm"), ("mm", "mm"),
     (50., 50.), ("at_least", "at_least")),
    ("usb-insert", ("insertion_depth_mm", "orientation_error_deg"), ("mm", "deg"),
     (12., 3.), ("target", "at_most")),
    ("usb-unplug", ("insertion_depth_m",), ("m",), (-.002,), ("at_most",)),
))
def test_ram_and_poker_annotate_only_the_matching_series(tmp_path, monkeypatch, task, keys, unit, targets, relations):
    source, episode, report, visual = parent_site(tmp_path, task=task)
    visual["task_series"] = [
        {"key": key, "unit": dimension, "time_s": [0, .1, .2],
         "values": [-.01, None, .001], "gap_after": [False, True, False]}
        for key, dimension in zip(keys, unit)
    ] + [{"key": "axial_resistance_n", "unit": "N", "time_s": [0], "values": [None]}]
    json_file(episode / "visualization.json", visual)
    config = _document(episode / "workbench.json")
    loads = _document(episode / "tactile_loads.json")
    (episode / "report.html").write_text(_page(report, visual, loads, config), encoding="utf-8")
    json_file(source / "artifact_manifest.json", {"files": [
        _digest(p) for p in _files(source) if p.name != "artifact_manifest.json"]})

    def task_goals(task, report):
        return {**goals(task, report), "references": [
            {"key": key, "unit": dimension, "target": value, "relation": relation,
             "label": "verified source reference", "note": "Not complete task success"}
            for key, dimension, value, relation in zip(keys, unit, targets, relations)]}

    monkeypatch.setattr(snapshot, "build_goal_references", task_goals)
    output = tmp_path / "extended"
    result = snapshot.export_snapshot(source, output)
    assert result["status_counts"]["verified"] == 1
    annotated = _document(output / episode.name / "visualization.json")
    assert snapshot._strip_references(annotated) == visual
    assert [row["goal_references"][0]["target"] for row in annotated["task_series"][:len(keys)]] == list(targets)
    assert "goal_references" not in annotated["task_series"][len(keys)]
    snapshot._same_native_evidence(report, _document(output / episode.name / "report.json"))
    assert snapshot.verify_snapshot(output)["parent_unchanged"]


def test_chained_snapshot_rebinds_goals_without_touching_either_parent(tmp_path, monkeypatch):
    source, episode, _, visual = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    first = tmp_path / "first_reference_snapshot"
    snapshot.export_snapshot(source, first)
    source_before = {p: p.read_bytes() for p in _files(source)}
    first_before = {p: p.read_bytes() for p in _files(first)}
    second = tmp_path / "second_reference_snapshot"
    snapshot.export_snapshot(first, second)
    assert {p: p.read_bytes() for p in _files(source)} == source_before
    assert {p: p.read_bytes() for p in _files(first)} == first_before
    new_visual = _document(second / episode.name / "visualization.json")
    assert snapshot._strip_references(new_visual) == visual
    assert len(new_visual["task_series"][0]["goal_references"]) == 1
    assert snapshot.verify_snapshot(second)["parent_unchanged"]


def test_chained_snapshot_drops_stale_lines_if_source_is_now_unavailable(tmp_path, monkeypatch):
    source, episode, _, visual = parent_site(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", goals)
    first = tmp_path / "first_reference_snapshot"
    snapshot.export_snapshot(source, first)
    monkeypatch.setattr(snapshot, "build_goal_references", lambda task, report: {
        **goals(task, report), "status": "unavailable", "references": [], "warnings": ["missing source"]})
    second = tmp_path / "source_unavailable_snapshot"
    result = snapshot.export_snapshot(first, second)
    assert result["status_counts"]["unavailable"] == 1
    assert _document(second / episode.name / "visualization.json") == visual
    assert _document(second / episode.name / "workbench.json")["goal_references"]["references"] == []
    assert _document(first / episode.name / "visualization.json")["task_series"][0]["goal_references"]


def fake_area(task, report, *, time_origin_s):
    return {"schema_version": snapshot.AREA_SCHEMA, "status": "verified", "task": task,
            "series": [{"key": key, "unit": "fraction", "label": key,
                        "time_s": [0., .1, .2], "values": values, "gap_after": [False, True, False],
                        "presentation_extension": snapshot.AREA_SCHEMA}
                       for key, values in (("effective_pad_contact_area_ratio", [None, 0., .5]),
                                           ("contact_time_weighted_mean_effective_area_ratio", [None, 0., .4]),
                                           ("maximum_ink_remaining", [1., .9, .8]))],
            "summary": [{"key": "contact_time_weighted_mean_effective_area_ratio", "unit": "fraction",
                         "label": "接触期间均值", "value": .4, "status": "recorded",
                         "presentation_extension": snapshot.AREA_SCHEMA}],
            "provenance": [], "warnings": [], "binding": {"origin_simulation_time_s": time_origin_s}}


def board_goals(task, report):
    return {**goals(task, report), "references": [
        {"key": "cleaned_fraction", "unit": "fraction", "target": 1.,
         "relation": "target", "reference_kind": "ideal"},
        {"key": "maximum_ink_remaining", "unit": "fraction", "target": 1e-9,
         "relation": "at_most"},
    ]}


def board_parent(tmp_path):
    source, episode, report, visual = parent_site(tmp_path, task="whiteboard-wipe")
    visual["task_series"][0].update(key="cleaned_fraction", unit="fraction")
    json_file(episode / "visualization.json", visual)
    (episode / "report.html").write_text(_page(report, visual, _document(episode / "tactile_loads.json"),
                                               _document(episode / "workbench.json")), encoding="utf-8")
    json_file(source / "artifact_manifest.json", {"files": [
        _digest(p) for p in _files(source) if p.name != "artifact_manifest.json"]})
    return source, episode, report, visual


def test_whiteboard_only_exposes_bound_series_and_preserves_native_arrays_and_labels(tmp_path, monkeypatch):
    source, episode, report, visual = board_parent(tmp_path)
    before = {p: p.read_bytes() for p in _files(source)}
    monkeypatch.setattr(snapshot, "build_goal_references", board_goals)
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", fake_area)
    output = tmp_path / "whiteboard"
    result = snapshot.export_snapshot(source, output)
    assert result["whiteboard_contact_area_status_counts"]["verified"] == 1
    target = output / episode.name
    new_visual = _document(target / "visualization.json")
    assert snapshot._strip_display_extensions(new_visual) == visual
    assert len(new_visual["task_series"]) == len(visual["task_series"]) + 3
    assert new_visual["task_series"][0]["goal_references"][0]["reference_kind"] == "ideal"
    assert new_visual["task_series"][-1]["goal_references"][0]["target"] == 1e-9
    assert all("goal_references" not in row for row in new_visual["task_series"][2:4])
    config = _document(target / "workbench.json")
    assert config["task_summary"][0]["value"] == .4
    assert snapshot.AREA_KEYS <= set(config["task_presentation"]["primary_task_keys"])
    snapshot._same_native_evidence(report, _document(target / "report.json"))
    _, embedded = _embedded((target / "report.html").read_text(), "kai-evaluation-data")
    assert embedded["visualization"] == new_visual
    assert {p: p.read_bytes() for p in _files(source)} == before
    assert snapshot.verify_snapshot(output)["parent_unchanged"]


def test_whiteboard_chained_snapshots_rebind_without_duplicate_curves_or_summary(tmp_path, monkeypatch):
    source, episode, _, visual = board_parent(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", board_goals)
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", fake_area)
    first, second = tmp_path / "first", tmp_path / "second"
    snapshot.export_snapshot(source, first)
    snapshot.export_snapshot(first, second)
    new_visual = _document(second / episode.name / "visualization.json")
    assert snapshot._strip_display_extensions(new_visual) == visual
    assert len(new_visual["task_series"]) == len(visual["task_series"]) + 3
    assert len(_document(second / episode.name / "workbench.json")["task_summary"]) == 1
    assert snapshot.verify_snapshot(second)["parent_unchanged"]


def test_whiteboard_missing_source_or_unaligned_clock_never_reuses_old_curves(tmp_path, monkeypatch):
    source, episode, _, visual = board_parent(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", board_goals)
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", fake_area)
    first = tmp_path / "first"
    snapshot.export_snapshot(source, first)

    def unavailable(task, report, *, time_origin_s):
        result = fake_area(task, report, time_origin_s=time_origin_s)
        result.update(status="unavailable", series=[], summary=[{
            "key": "contact_time_weighted_mean_effective_area_ratio", "value": None,
            "status": "unavailable", "presentation_extension": snapshot.AREA_SCHEMA}])
        return result

    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", unavailable)
    second = tmp_path / "second"
    result = snapshot.export_snapshot(first, second)
    new_visual = _document(second / episode.name / "visualization.json")
    assert snapshot._strip_references(new_visual) == visual
    assert result["whiteboard_contact_area_status_counts"]["unavailable"] == 1
    assert _document(second / episode.name / "workbench.json")["task_summary"][0]["value"] is None


def test_whiteboard_extension_cannot_overwrite_a_native_curve():
    area = fake_area("whiteboard-wipe", {}, time_origin_s=0.)
    visual = {"task_series": [{"key": "effective_pad_contact_area_ratio", "values": [.2]}]}
    with pytest.raises(ValueError, match="colliding"):
        snapshot._attach_area(visual, {"task_summary": []}, area)


def test_usb_source_copy_is_only_used_when_explicitly_supplied(tmp_path, monkeypatch):
    source, *_ = parent_site(tmp_path, task="usb-insert")
    calls = []

    def capture(task, report, **kwargs):
        calls.append(kwargs)
        return goals(task, report)

    monkeypatch.setattr(snapshot, "build_goal_references", capture)
    snapshot.export_snapshot(source, tmp_path / "without_copy")
    snapshot.export_snapshot(source, tmp_path / "with_copy", verified_usb_source_copy=tmp_path)
    assert calls == [{}, {"source_root": tmp_path}]


def test_portal_only_snapshot_preserves_all_episode_bytes_and_internal_batch_evidence(tmp_path, monkeypatch):
    source, episode, *_ = parent_site(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in _files(source)}
    original_batch = _document(source / "batch.json")
    monkeypatch.setattr(snapshot, "build_goal_references", lambda *_args, **_kwargs: pytest.fail("No rule reread for a portal update"))
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", lambda *_args, **_kwargs: pytest.fail("No geometry import for a portal update"))
    output = tmp_path / "presentation"
    result = snapshot.export_snapshot(source, output, presentation_only=True)
    assert result["presentation_only"] is True
    assert {p.relative_to(source): p.read_bytes() for p in _files(source)} == before
    for relative, raw in before.items():
        if str(relative) not in snapshot.PRESENTATION_FILES:
            assert (output / relative).read_bytes() == raw
    assert _document(output / episode.name / "report.json") == _document(episode / "report.json")
    new_batch = _document(output / "batch.json")
    snapshot._same_archived_batch(original_batch, new_batch)
    assert new_batch["presentation_extension"]["internal_audit_metadata_preserved"]
    assert new_batch["presentation_extension"]["show_run_mode"] is False
    assert new_batch["presentation_extension"]["show_coverage_and_sources"] is False
    html = (output / "index.html").read_text()
    for removed in ('id="ks-mode"', 'id="ks-coverage"', 'class="ks-audit"', '数据覆盖与来源', '正式标志'):
        assert removed not in html
    assert _document(output / "verification.json")["trial_files_unchanged"]
    assert snapshot.verify_snapshot(output)["parent_unchanged"]


def test_portal_only_retains_existing_goal_and_area_annotations_without_rebinding(tmp_path, monkeypatch):
    source, episode, *_ = board_parent(tmp_path)
    monkeypatch.setattr(snapshot, "build_goal_references", board_goals)
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", fake_area)
    reference = tmp_path / "reference"
    snapshot.export_snapshot(source, reference)
    reference_bytes = {p.relative_to(reference): p.read_bytes() for p in _files(reference)}
    monkeypatch.setattr(snapshot, "build_goal_references", lambda *_a, **_k: pytest.fail("Do not rebind archived goals"))
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", lambda *_a, **_k: pytest.fail("Do not reload archived area"))
    output = tmp_path / "clean_presentation"
    result = snapshot.export_snapshot(reference, output, presentation_only=True)
    assert result["status_counts"]["verified"] == 1
    assert result["whiteboard_contact_area_status_counts"]["verified"] == 1
    for relative, raw in reference_bytes.items():
        if str(relative) not in snapshot.PRESENTATION_FILES:
            assert (output / relative).read_bytes() == raw
    assert (output / episode.name / "whiteboard_contact_area.json").exists()


def test_portal_only_whitelist_prevents_unintended_scientific_asset_mutation(tmp_path, monkeypatch):
    source, *_ = parent_site(tmp_path)
    original_copy = snapshot._copy

    def alter_copy(job, target, artifacts):
        original_copy(job, target, artifacts)
        if target.name == "dashboard.js":
            target.write_bytes(b"unexpected-replay-change")

    monkeypatch.setattr(snapshot, "_copy", alter_copy)
    output = tmp_path / "invalid_portal"
    with pytest.raises(ValueError, match="protected archived artifact"):
        snapshot.export_snapshot(source, output, presentation_only=True)
    assert not output.exists()


def test_portal_metadata_cannot_change_recorded_mode_or_stage_evidence():
    before = {"trials": [{"mode": "diagnostic"}], "mode_counts": {"diagnostic": 1}, "stage_groups": [{"count": 1}]}
    after = deepcopy(before)
    after["presentation_extension"] = {"show_run_mode": False}
    snapshot._same_archived_batch(before, after)
    after["trials"][0]["mode"] = "formal"
    with pytest.raises(ValueError, match="archived batch evidence"):
        snapshot._same_archived_batch(before, after)


def navigation_trial(**overrides):
    trial = {"task": "install-ram", "trial_id": "ram:checkpoint:seed_000", "model": "pi0.5",
             "checkpoint": "20000", "seed": 0, "prediction_horizon": 50,
             "report_path": "000_ram_seed_000/report.html", "outcome": "pass",
             "source": "/private/internal/source", "provenance": {"secret": "/private/manifest"},
             "staged": {"native_result": {"task_success": False}, "complete_success": True,
                        "stages": [{"label": "到达", "reached": True}]}}
    trial.update(overrides)
    return trial


def navigation_parent(tmp_path):
    source, episode, *_ = parent_site(tmp_path, task="install-ram")
    batch = _document(source / "batch.json")
    batch["trials"] = [navigation_trial(report_path=episode.name + "/report.html")]
    batch["goal_reference_extension"] = {
        "status_counts": {"verified": 1, "unavailable": 0, "not_applicable": 0},
        "whiteboard_contact_area_status_counts": {"verified": 0, "unavailable": 0, "not_applicable": 1},
        "source_report": "/preserved/reference/source", "references": {"goal": 6}}
    batch["presentation_extension"] = {"show_run_mode": False, "show_coverage_and_sources": False}
    json_file(source / "batch.json", batch)
    for name in ("workbench.js", "workbench.css"):
        (source / "assets" / name).write_text("old-workbench-ui")
    json_file(source / "artifact_manifest.json", {"files": [
        _digest(p) for p in _files(source) if p.name != "artifact_manifest.json"]})
    return source, episode


def test_navigation_index_is_minimal_deterministic_and_keeps_native_unknown_separate():
    ram_late = navigation_trial(trial_id="ram:late", seed=8, checkpoint="20")
    ram_early = navigation_trial(trial_id="ram:early", seed=3, checkpoint="9",
                                report_path="early_ram/report.html",
                                staged={"native_result": {"task_success": True}, "complete_success": False})
    board = navigation_trial(task="whiteboard-wipe", trial_id="whiteboard:pickup",
                             report_path="board_pickup/report.html", checkpoint="20", seed=None,
                             staged={"native_result": {"task_success": False}, "complete_success": None})
    unknown = navigation_trial(task="usb-insert", trial_id="usb:unknown",
                               report_path="usb_unknown/report.html", staged={"complete_success": True})
    inputs = [board, ram_late, unknown, ram_early]
    before = deepcopy(inputs)
    result = snapshot._task_navigation_index(inputs)
    assert inputs == before
    assert set(result) == {"schema_version", "prediction_horizon", "trials"}
    assert result["schema_version"] == "evaluation-task-navigation-v1"
    assert result["prediction_horizon"] == 50
    rows = result["trials"]
    assert [row["trial_id"] for row in rows] == ["ram:early", "ram:late", "usb:unknown", "whiteboard:pickup"]
    assert [row["native_success"] for row in rows] == [True, False, None, False]
    assert [row["seed"] for row in rows] == [3, 8, 0, None]
    assert all(set(row) == {"task", "trial_id", "model", "checkpoint", "seed", "prediction_horizon",
                            "report_path", "native_success"} for row in rows)
    assert [row["trial_id"] for row in rows if row["task"] == "install-ram"] == ["ram:early", "ram:late"]
    assert "/private/" not in json.dumps(result)


@pytest.mark.parametrize("value", (30, None, "50", 50.0))
def test_navigation_index_rejects_nonexplicit_h50(value):
    with pytest.raises(ValueError, match="H50"):
        snapshot._task_navigation_index([navigation_trial(prediction_horizon=value)])


@pytest.mark.parametrize("path", ("/absolute/report.html", "../escape/report.html", "a/../report.html",
                                  "a\\report.html", "a/report.html?x=1", "a%2fb/report.html",
                                  "a.b/report.html", "白板/report.html"))
def test_navigation_index_rejects_unsafe_report_paths(path):
    with pytest.raises(ValueError, match="report path"):
        snapshot._task_navigation_index([navigation_trial(report_path=path)])


@pytest.mark.parametrize("change", ({"report_path": "different/report.html"}, {"trial_id": "different"}))
def test_navigation_index_rejects_duplicate_ids_or_paths(change):
    with pytest.raises(ValueError, match="Duplicate"):
        snapshot._task_navigation_index([navigation_trial(), navigation_trial(**change)])


@pytest.mark.parametrize("field,value", (("task", ""), ("trial_id", None), ("checkpoint", 20000),
                                        ("model", []), ("seed", True), ("seed", -1), ("seed", "1")))
def test_navigation_index_requires_explicit_safe_identity_types(field, value):
    with pytest.raises(ValueError, match="Navigation"):
        snapshot._task_navigation_index([navigation_trial(**{field: value})])


@pytest.mark.parametrize("success", (0, 1, "false", "true"))
def test_navigation_index_never_coerces_native_task_success(success):
    with pytest.raises(ValueError, match="boolean or null"):
        snapshot._task_navigation_index([navigation_trial(staged={"native_result": {"task_success": success}})])


def test_detail_navigation_only_replaces_explicit_ui_assets_and_keeps_every_episode_byte(tmp_path, monkeypatch):
    source, episode = navigation_parent(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in _files(source)}
    original_batch = _document(source / "batch.json")
    monkeypatch.setattr(snapshot, "build_goal_references", lambda *_a, **_k: pytest.fail("No goal importer for detail UI"))
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", lambda *_a, **_k: pytest.fail("No area importer for detail UI"))
    output = tmp_path / "detail_navigation"
    result = snapshot.export_snapshot(source, output, presentation_only=True, detail_navigation=True)
    assert result["presentation_only"] and result["detail_navigation"] and result["parent_unchanged"]
    assert result["status_counts"]["verified"] == 1
    assert {p.relative_to(source): p.read_bytes() for p in _files(source)} == before
    allowed = snapshot.PRESENTATION_FILES | snapshot.DETAIL_NAVIGATION_FILES
    for relative, raw in before.items():
        if str(relative) not in allowed:
            assert (output / relative).read_bytes() == raw
    assert (output / episode.name / "report.html").read_bytes() == (episode / "report.html").read_bytes()
    for name in ("workbench.js", "workbench.css"):
        assert (output / "assets" / name).read_bytes() == (snapshot.ANALYSIS / "evaluation_workbench/web" / name).read_bytes()
    new_batch = _document(output / "batch.json")
    snapshot._same_archived_batch(original_batch, new_batch)
    assert new_batch["goal_reference_extension"] == original_batch["goal_reference_extension"]
    extension = new_batch["presentation_extension"]
    assert extension["detail_navigation"] and extension["show_task_stages"]
    assert extension["trial_files_unchanged"] and extension["native_outcomes_rescored"] is False
    assert extension["navigation_index"] == "trajectory_index.json"
    index = _document(output / "trajectory_index.json")
    assert index == snapshot._task_navigation_index(original_batch["trials"])
    proof = _document(output / "verification.json")
    assert proof["detail_navigation"] and proof["trial_files_unchanged"]
    assert snapshot.verify_snapshot(output)["parent_unchanged"]


@pytest.mark.parametrize("asset", ("assets/workbench.js", "assets/workbench.css", "trajectory_index.json"))
def test_portal_only_without_detail_flag_does_not_broaden_asset_whitelist(tmp_path, monkeypatch, asset):
    source, _ = navigation_parent(tmp_path)
    if asset == "trajectory_index.json":
        json_file(source / asset, {"previous": "preserve this navigation index"})
        json_file(source / "artifact_manifest.json", {"files": [
            _digest(p) for p in _files(source) if p.name != "artifact_manifest.json"]})
    original_copy = snapshot._copy

    def alter_copy(job, target, artifacts):
        original_copy(job, target, artifacts)
        if str(target).endswith("/" + asset):
            target.write_bytes(b"unexpected-ui-change")

    monkeypatch.setattr(snapshot, "_copy", alter_copy)
    output = tmp_path / "invalid_portal"
    with pytest.raises(ValueError, match="protected archived artifact"):
        snapshot.export_snapshot(source, output, presentation_only=True)
    assert not output.exists()


def test_detail_navigation_still_protects_dashboard_and_all_episode_files(tmp_path, monkeypatch):
    source, _ = navigation_parent(tmp_path)
    original_copy = snapshot._copy

    def alter_copy(job, target, artifacts):
        original_copy(job, target, artifacts)
        if target.name == "dashboard.js":
            target.write_bytes(b"unexpected-replay-change")

    monkeypatch.setattr(snapshot, "_copy", alter_copy)
    output = tmp_path / "invalid_detail"
    with pytest.raises(ValueError, match="protected archived artifact"):
        snapshot.export_snapshot(source, output, presentation_only=True, detail_navigation=True)
    assert not output.exists()


def test_detail_navigation_api_and_cli_require_presentation_only(tmp_path):
    with pytest.raises(ValueError, match="requires presentation-only"):
        snapshot.export_snapshot(tmp_path / "absent", tmp_path / "new", detail_navigation=True)
    with pytest.raises(SystemExit) as failure:
        snapshot.main(["--detail-navigation"])
    assert failure.value.code == 2


def test_visual_refresh_is_an_independent_ui_only_snapshot_without_rescoring(tmp_path, monkeypatch):
    source, episode = navigation_parent(tmp_path)
    before = {p.relative_to(source): p.read_bytes() for p in _files(source)}
    old_batch = _document(source / "batch.json")
    monkeypatch.setattr(snapshot, "build_goal_references", lambda *_a, **_k: pytest.fail("No new reference rules"))
    monkeypatch.setattr(snapshot, "build_whiteboard_contact_area", lambda *_a, **_k: pytest.fail("No geometry reread"))
    output = tmp_path / "studio"
    result = snapshot.export_snapshot(source, output, presentation_only=True,
                                      detail_navigation=True, visual_refresh=True)
    assert result["visual_refresh"] and result["parent_unchanged"]
    assert result["native_outcomes_rescored"] is False
    assert {p.relative_to(source): p.read_bytes() for p in _files(source)} == before
    allowed = snapshot.PRESENTATION_FILES | snapshot.DETAIL_NAVIGATION_FILES
    for relative, raw in before.items():
        if str(relative) not in allowed:
            assert (output / relative).read_bytes() == raw
    assert (output / episode.name / "report.html").read_bytes() == (episode / "report.html").read_bytes()
    batch = _document(output / "batch.json")
    snapshot._same_archived_batch(old_batch, batch)
    extension = batch["presentation_extension"]
    assert extension["visual_refresh"] and extension["layout_version"] == "evaluation-studio-v1"
    assert extension["model_first"] and extension["expert_clock_independent"]
    assert extension["overview_views"] == ["cards", "list"]
    assert extension["trial_files_unchanged"]
    html = (output / "index.html").read_text()
    assert "<title>KaiTactileSim 官方评估平台 | H50</title>" in html
    assert "<h1>KaiTactileSim 官方评估平台</h1>" in html
    assert 'class="ks-eyebrow">KaiTactileSim Evaluation Platform' in html
    assert "机器人操作任务评估 · 动作回放 · 触觉分析" in html
    assert 'class="ks-brand-footer"' in html
    assert extension["branding"]["scope"] == "project_owned_interface_identity_only"
    assert extension["branding"]["evaluation_status_changed"] is False
    assert _document(output / "verification.json")["visual_refresh"]
    assert snapshot.verify_snapshot(output)["parent_unchanged"]


@pytest.mark.parametrize("presentation,detail", [(False, False), (True, False), (False, True)])
def test_visual_refresh_needs_explicit_presentation_and_detail_modes(tmp_path, presentation, detail):
    with pytest.raises(ValueError, match="Visual refresh requires"):
        snapshot.export_snapshot(tmp_path / "absent", tmp_path / "new", visual_refresh=True,
                                 presentation_only=presentation, detail_navigation=detail)


def test_visual_refresh_cli_does_not_bypass_presentation_guard():
    for flags in (["--visual-refresh"], ["--visual-refresh", "--presentation-only"],
                  ["--visual-refresh", "--detail-navigation"]):
        with pytest.raises(SystemExit) as failure:
            snapshot.main(flags)
        assert failure.value.code == 2


@pytest.mark.parametrize("asset", ["assets/dashboard.js", "000_task_seed_000/report.html",
                                  "000_task_seed_000/workbench.json", "000_task_seed_000/visualization.json"])
def test_visual_refresh_cannot_mutate_native_data_or_replay_assets(tmp_path, monkeypatch, asset):
    source, _ = navigation_parent(tmp_path)
    original_copy = snapshot._copy

    def alter_copy(job, target, artifacts):
        original_copy(job, target, artifacts)
        if str(target).endswith("/" + asset):
            target.write_bytes(b"forbidden-scientific-change")

    monkeypatch.setattr(snapshot, "_copy", alter_copy)
    output = tmp_path / "invalid_studio"
    with pytest.raises(ValueError, match="protected archived artifact"):
        snapshot.export_snapshot(source, output, presentation_only=True,
                                 detail_navigation=True, visual_refresh=True)
    assert not output.exists()
