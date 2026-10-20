import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from evaluation_status.rich_detail_data import (  # noqa: E402
    build_rich_case,
    resolve_rich_source,
)
from evaluation_tactile_load_extension.data import _LABELS  # noqa: E402
from evaluation_v1.dashboard import render_dashboard  # noqa: E402


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def _fixture(tmp_path, *, model="pi05", task="bulb-screw", review_layout=False, stage_status="valid"):
    source = tmp_path / "evaluations" / f"{model}-source" / "seed_000"
    source.mkdir(parents=True)
    media = source / "review" if review_layout else source
    media.mkdir(exist_ok=True)
    horizon, execute = (30, 5) if model == "pi05+cpm" else (32, 5) if model == "ftp1" else (50, 50)
    identity = {"task": task, "seed": 0, "model_family": model, "prediction_horizon": horizon,
                "execute_steps": execute, "checkpoint_sha256": "a" * 64, "deployment_id": "b" * 64}
    group = {"id": "group", "task": task, "title": task, "model": model, "checkpoint": "best" if model == "pi05+cpm" else "10000",
             "checkpoint_sha256": "a" * 64, "prediction_horizon": horizon, "execute_steps": execute,
             "mode": "diagnostic", "protocol": {"guard_mode": "observe"}, "rule_identity": {"sha256": "c" * 64}}
    stages = [{"key": "stage1", "name": "picked_up", "reached": True}]
    member = {"id": "published-trial", "seed": 0, "valid": True, "native_success": False,
              "status": "task_not_completed", "publication_status": "eligible", "staged_evidence_status": stage_status,
              "staged_evidence_basis": "recorded_archive" if stage_status == "valid" else None,
              "stages": stages if stage_status == "valid" else [], "complete_stage_success": False if stage_status == "valid" else None,
              "notes": []}
    time = np.arange(5, dtype=float) / 30
    taxels = np.full((5, 5, 7, 5), 0.01)
    tangent = np.full((5, 5, 7, 5, 2), 0.001)
    normal = taxels.sum(axis=(2, 3))
    ft = np.linalg.norm(tangent.sum(axis=(2, 3)), axis=-1)
    np.savez_compressed(media / "evaluation_rollout_trace.npz", time_s=time, simulation_time_s=time,
                        normal_taxel_force_n=taxels, tangent_taxel_force_n=tangent,
                        fingertip_normal_force_n=normal, fingertip_tangent_force_n=ft,
                        fingertip_link_names=np.array(list(_LABELS)), state_29=np.zeros((5, 29)),
                        wrist_state=np.zeros((5, 9)), hand_joint_position=np.zeros((5, 20)))
    comparison = {"schema": "kaihand-evaluation-comparison-v1",
                  "artifacts": {"rollout_trace": str(media / "evaluation_rollout_trace.npz")},
                  "tactile": {"unit": "N", "taxel_grid": [7, 5], "link_names": list(_LABELS),
                              "Fn": "sum of all 7x5 normal taxel forces",
                              "Ft": "sqrt(sum(Ft_x over 7x5)^2 + sum(Ft_y over 7x5)^2)"}}
    video = {**identity, "schema": "kaihand-policy-evaluation-review-v3", "completed": True,
             "fps": 30, "frame_count": 5, "video_clock": "simulation time; explicit recorded frames",
             "comparison_plots": comparison}
    summary = {**identity, "status": "task_not_completed", "task_success": False, "physics_valid": True,
               "sim_seconds": float(time[-1]), "max_sim_seconds": 90, "diagnostic_only": True,
               "formal_metrics_valid": False, "video": video,
               "evaluation": {"task": task, "success": False, "state": {"clockwise_turns": 0.25}}}
    _json(source / "summary.json", summary)
    _json(media / "review.json", video)
    (media / "review.mp4").write_bytes(b"\0\0\0\x18ftypisom00000000")
    frames = [{"frame": i, "simulation_time_s": float(t), "camera_pose_time_s": float(t), "tactile_time_s": float(t),
               "phase": "recorded_phase", "task_metrics": {"clockwise_turns": float(i) / 20}}
              for i, t in enumerate(time)]
    _jsonl(media / "frames.jsonl", frames)
    if stage_status != "unknown":
        payload = {"version": "staged_v1", "stage_definitions": {"stage1": "picked_up"}, "stage_flags": {"stage1": True},
                   "first_times": {"stage1": 0.05}, "complete_success": False, "thresholds": {"arbitrary_recorded": 123},
                   "stage_achievement_policy": "ordered_cumulative_ever_reached", "time_basis": "mujoco_data_time_seconds"}
        _json(source / "staged_evaluation.json", payload)
        _jsonl(source / "staged_states.jsonl", [{"time_s": float(t), "simulation_time_s": float(t),
                                                "measurements": {"hand_normal_load_n": 0.5}} for t in time])
    authority = tmp_path / "reports" / "published" / "summary.json"
    _json(authority, summary)
    member["summary_sha256"] = _sha(authority)
    return source, authority, group, member


def _build(fixture):
    source, authority, group, member = fixture
    return build_rich_case(source, group, member, published_summary_path=authority,
                           expected_summary_sha256=member["summary_sha256"])


@pytest.mark.parametrize("model,horizon", [("pi05", 50), ("ftp1", 32), ("pi05+cpm", 30)])
def test_rich_preserves_real_horizon_and_published_native_result(tmp_path, model, horizon):
    f = _fixture(tmp_path, model=model)
    source_before = {path: path.read_bytes() for path in f[0].rglob("*") if path.is_file()}
    case = _build(f)
    assert case["report"]["trial"]["prediction_horizon"] == horizon
    assert case["config"]["prediction_horizon"] == horizon
    assert case["tactile_loads"]["prediction_horizon"] == horizon
    assert case["report"]["outcome"]["status"] == "fail"
    assert case["source_binding"]["binding_basis"] == "exact_bytes"
    assert case["availability"]["native_samples"] == 5
    assert len(case["visualization"]["tactile_series"]["time_s"]) == 5
    assert case["state_trace"]["status"] == "available"
    assert case["config"]["expert_video"]["status"] == "missing"
    assert case["config"]["diagnostic_plots"]["status"] == "missing"
    assert case["visualization"]["task_series"][0]["key"] == "clockwise_angle_deg"
    assert case["visualization"]["task_series"][0]["values"][-1] == 72.0
    assert "kai-evaluation-data" in render_dashboard(case["report"], case["visualization"])
    assert all(path.read_bytes() == raw for path, raw in source_before.items())


def test_rich_only_allows_serialization_difference_in_summary_binding(tmp_path):
    f = _fixture(tmp_path)
    source, authority, _, _ = f
    payload = json.loads((source / "summary.json").read_text())
    (source / "summary.json").write_text(json.dumps(payload, separators=(",", ":")))
    case = _build(f)
    assert case["source_binding"]["binding_basis"] == "json_object_equal"
    assert case["source_binding"]["producer_summary"]["sha256"] != case["source_binding"]["published_summary"]["sha256"]
    assert case["source_binding"]["published_summary"]["sha256"] == _sha(authority)
    payload["physics_valid"] = 1
    _json(source / "summary.json", payload)
    with pytest.raises(ValueError, match="JSON objects differ"):
        _build(f)


@pytest.mark.parametrize("edit,match", [
    (lambda group, member: group.update(prediction_horizon=30), "prediction_horizon"),
    (lambda group, member: group.update(execute_steps=5), "execute_steps"),
    (lambda group, member: group.update(checkpoint_sha256="f" * 64), "checkpoint_sha256"),
    (lambda group, member: group.update(model="ftp1"), "model family"),
    (lambda group, member: member.update(seed=1), "seed"),
    (lambda group, member: member.update(native_success=True), "native outcome"),
    (lambda group, member: member.update(publication_status="quarantined_replay_only"), "Quarantined"),
    (lambda group, member: member.update(summary_sha256="f" * 64), "Published summary SHA"),
])
def test_rich_rejects_identity_outcome_and_authority_mismatch(tmp_path, edit, match):
    f = _fixture(tmp_path)
    edit(f[2], f[3])
    with pytest.raises(ValueError, match=match):
        _build(f)


def test_rich_masks_cpm_raw_stage_flags_without_discarding_record(tmp_path):
    case = _build(_fixture(tmp_path, model="pi05+cpm", task="poker-draw", stage_status="invalid"))
    stage = case["config"]["staged_evaluation"]
    assert stage["recorded_raw_stage_flags"] == {"stage1": True}
    assert stage["staged_result"]["stage_flags"] == {"stage1": None}
    assert stage["staged_result"]["complete_success"] is None
    assert stage["stages"][0]["reached"] is None
    assert stage["stages"][0]["first_time_s"] is None
    assert stage["evidence_status"] == "invalid"
    assert any("已记录" in warning for warning in stage["warnings"])
    assert case["report"]["outcome"]["recorded_success"] is False


def test_rich_ego_review_layout_does_not_invent_task_progress_or_actions(tmp_path):
    f = _fixture(tmp_path, model="egotouch", task="pick-place", review_layout=True, stage_status="unknown")
    frames = f[0] / "review/frames.jsonl"
    rows = [json.loads(line) for line in frames.read_text().splitlines()]
    for row in rows:
        row["task_metrics"] = {}
    _jsonl(frames, rows)
    case = _build(f)
    assert case["visualization"]["task_series"] == []
    assert "staged_evaluation" not in case["config"]
    assert case["availability"]["applied_actions"] == "not_recorded"
    assert case["availability"]["task_series"] == 0
    assert case["visualization"]["media"]["sync_status"] == "mapped"


def test_rich_rejects_stage_flag_drift_instead_of_rejudging(tmp_path):
    f = _fixture(tmp_path)
    p = f[0] / "staged_evaluation.json"
    data = json.loads(p.read_text())
    data["stage_flags"]["stage1"] = False
    _json(p, data)
    with pytest.raises(ValueError, match="stage flags/definitions"):
        _build(f)


@pytest.mark.parametrize("field,value", [("first_times", {"stage1": 0.1}), ("thresholds", {"changed": 456})])
def test_rich_stage_first_times_and_thresholds_are_bound_to_published_summary(tmp_path, field, value):
    f = _fixture(tmp_path)
    source, authority, _, member = f
    sidecar = json.loads((source / "staged_evaluation.json").read_text())
    summary = json.loads(authority.read_text())
    summary["staged_evaluation"] = copy.deepcopy(sidecar)
    _json(authority, summary)
    _json(source / "summary.json", summary)
    member["summary_sha256"] = _sha(authority)
    case = _build(f)
    assert case["config"]["staged_evaluation"]["data_evidence"]["stage_payload_binding"] == "published_summary_embedded_payload"
    sidecar[field] = value
    _json(source / "staged_evaluation.json", sidecar)
    with pytest.raises(ValueError, match="stage payload differs"):
        _build(f)


def test_rich_without_archived_stage_payload_does_not_certify_live_first_times(tmp_path):
    case = _build(_fixture(tmp_path))
    stage = case["config"]["staged_evaluation"]
    assert stage["stages"][0]["reached"] is True
    assert stage["stages"][0]["first_time_s"] is None
    assert stage["staged_result"]["thresholds"] == {}
    assert stage["data_evidence"]["stage_payload_binding"] == "recorded_observations_only"
    assert case["config"]["media_views"]["status"] != "verified"


def test_rich_native_arrays_preserve_unknown_state_as_null(tmp_path):
    f = _fixture(tmp_path)
    trace = f[0] / "evaluation_rollout_trace.npz"
    with np.load(trace, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["wrist_state"][1, 0] = np.nan
    np.savez_compressed(trace, **data)
    assert _build(f)["state_trace"]["wrist_state"][1][0] is None


def _inventory(root):
    files = [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p), "size": p.stat().st_size}
             for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"]
    _json(root / "artifact_manifest.json", {"schema": "published-fixture", "files": files})


def test_resolver_selects_only_bound_published_source_and_rejects_escape(tmp_path):
    source, authority, group, member = _fixture(tmp_path)
    reports = tmp_path / "reports"
    catalog_root = reports / "r3"
    catalog_root.mkdir()
    detail = authority.parent / "index.html"
    detail.write_text("published detail")
    native = {"id": member["id"], "seed": 0, "valid": True, "native_success": False,
              "source": str(source), "summary_path": str(source / "summary.json"), "summary_sha256": _sha(authority)}
    _json(authority.parent / "trial.json", native)
    _inventory(authority.parent)
    group.update(source_snapshot="pi05_h50")
    member.update(detail_url="../published/index.html")
    binding = resolve_rich_source(catalog_root, group, member, evaluation_root=tmp_path / "evaluations")
    assert binding["kind"] == "build" and Path(binding["source_dir"]) == source
    assert binding["expected_summary_sha256"] == member["summary_sha256"]
    native["source"] = str(tmp_path / "outside")
    native["summary_path"] = str(tmp_path / "outside/summary.json")
    _json(authority.parent / "trial.json", native)
    _inventory(authority.parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_rich_source(catalog_root, group, member, evaluation_root=tmp_path / "evaluations")


def test_rich_whiteboard_copies_recorded_integral_not_display_sample_mean(tmp_path):
    f = _fixture(tmp_path, task="whiteboard-wipe")
    payload_path = f[0] / "staged_evaluation.json"
    payload = json.loads(payload_path.read_text())
    geometry = {"schema_version": "kaihand-whiteboard-geometric-area-v1",
                "measurement_type": "loaded_rigid_mesh_geometric_area_estimate", "is_true_deformable_contact_area": False,
                "area_denominator": "compiled_felt_mesh_flat_bottom_polygon_area",
                "area_integration": "physics_dt_weighted_zero_order_hold_of_sampled_geometry_with_physics_rate_loaded_contact_gate",
                "full_pad_bottom_area_m2": 0.01, "effective_pad_contact_area_m2": 0.002,
                "effective_pad_contact_area_ratio": 0.2, "contact_time_weighted_mean_effective_area_ratio": 0.8,
                "pad_board_contact_duration_s": 2.0, "pad_board_loaded_contact": True}
    payload["continuous_metrics"] = geometry
    _json(payload_path, payload)
    rows = [json.loads(line) for line in (f[0] / "staged_states.jsonl").read_text().splitlines()]
    for index, row in enumerate(rows):
        row["continuous_metrics"] = copy.deepcopy(geometry)
        if index == 0:
            row["continuous_metrics"].update(pad_board_loaded_contact=False, pad_board_contact_duration_s=0.0)
    _jsonl(f[0] / "staged_states.jsonl", rows)
    case = _build(f)
    area = case["config"]["whiteboard_contact_area"]
    assert area["status"] == "recorded"
    assert area["summary"][0]["value"] == 0.8
    assert area["series"][0]["values"][0] is None
    assert area["series"][0]["values"][1] == 0.2
    assert area["binding"]["native_values_recomputed"] is False
