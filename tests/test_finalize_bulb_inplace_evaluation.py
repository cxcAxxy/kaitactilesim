"""Publication-controller tests with tiny files and mocked child processes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bulb_publication_controller", ROOT / "scripts/analysis/finalize_bulb_inplace_evaluation.py")
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def armed(tmp_path):
    control = tmp_path / "evaluations/_jobs/publication"
    control.mkdir(parents=True)
    source = tmp_path / "implementation.py"
    source.write_text("reviewed bytes")
    batch = tmp_path / "batch.json"
    _json(batch, {"frozen": True})
    portal = tmp_path / "reports/index.html"
    portal.parent.mkdir()
    portal.write_text("previous portal")
    plan = {"schema": controller.SCHEMA, "control_dir": str(control), "source_files": [controller.digest(source)],
            "fixed_inputs": [controller.digest(batch)], "previous_root_index": controller.digest(portal),
            "snapshot": str(portal.parent / "new-snapshot"), "environment": {"PYTHONPATH": "reviewed/path"}}
    return control, plan


def test_dependency_guards_allow_provenance_roles_but_reject_changed_bytes(armed):
    _, plan = armed
    with_role = {**plan["source_files"][0], "role": "strict_source", "canonical_path": "unused"}
    controller.check_rows([with_role])
    Path(with_role["path"]).write_text("changed bytes")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        controller.guards(plan)


def test_changed_portal_blocks_work_without_overwriting_another_publication(armed):
    _, plan = armed
    portal = Path(plan["previous_root_index"]["path"])
    portal.write_text("another publisher")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        controller.guards(plan)
    assert portal.read_text() == "another publisher"


def test_producer_identity_is_pid_reuse_safe_and_never_sends_signals(monkeypatch):
    plan = {"producer_identity": {"pid": 123, "start_ticks": 42}, "wait_deadline_unix": 1000}
    monkeypatch.setattr(controller, "process_identity", lambda pid: {"pid": 123, "start_ticks": 42})
    controller.producer_guard(plan, {"controller_pid": 123}, now=100)
    monkeypatch.setattr(controller, "process_identity", lambda pid: {"pid": 123, "start_ticks": 99})
    with pytest.raises(ValueError, match="changed identity"):
        controller.producer_guard(plan, {"controller_pid": 123}, now=100)
    monkeypatch.setattr(controller, "process_identity", lambda pid: None)
    with pytest.raises(ValueError, match="disappeared"):
        controller.producer_guard(plan, {"controller_pid": 123}, now=100)


def test_publication_wait_is_bounded_to_24_hours(monkeypatch):
    plan = {"producer_identity": {"pid": 123, "start_ticks": 42}, "wait_deadline_unix": 1000}
    with pytest.raises(ValueError, match="24-hour"):
        controller.producer_guard(plan, {"controller_pid": 123}, now=1000)


def test_new_implementation_file_blocks_armed_watcher(armed, monkeypatch):
    _, plan = armed
    plan["code_inventory"] = ["original.py"]
    monkeypatch.setattr(controller, "code_inventory", lambda: ["original.py", "new.py"])
    with pytest.raises(ValueError, match="membership changed"):
        controller.guards(plan)


def test_child_process_failure_is_terminal_and_records_exit_status(armed, monkeypatch):
    control, plan = armed
    state = {"children": []}
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda: 7))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        controller.run_child(plan, {"stage": "export", "argv": ["reviewed", "command"]}, control, state, {})
    assert state["children"][0]["returncode"] == 7
    assert Path(plan["previous_root_index"]["path"]).read_text() == "previous portal"
    assert (control / "export.log").is_file()


def test_child_success_without_independent_passed_receipt_is_rejected(armed, monkeypatch):
    control, plan = armed
    state = {"children": []}
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda: 0))
    _json(control / "independent_qa.json", {"status": "failed"})
    with pytest.raises(ValueError, match="passed independent receipt"):
        controller.run_child(plan, {"stage": "independent_qa", "argv": ["reviewed", "command"]}, control, state, {})


def test_browser_environment_only_changes_child_proxy_and_tmp_settings(armed, monkeypatch):
    _, plan = armed
    monkeypatch.setenv("HTTP_PROXY", "http://not-used-by-local-qa")
    environment = controller._child_environment(plan, "/dev/shm/unique-browser-temp")
    assert "HTTP_PROXY" not in environment
    assert environment["TMPDIR"] == "/dev/shm/unique-browser-temp"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert controller.os.environ["HTTP_PROXY"] == "http://not-used-by-local-qa"


def _publication(armed, *, changed=False):
    control, plan = armed
    snapshot = Path(plan["snapshot"])
    snapshot.mkdir()
    previous = snapshot / "previous_root_index.html"
    previous.write_text("previous portal")
    portal = Path(plan["previous_root_index"]["path"])
    portal.write_text("our new portal")
    _json(snapshot / "portal_publication.json", {"new_sha256": controller.digest(portal)["sha256"]})
    if changed:
        portal.write_text("a later publication")
    return control, plan, portal


def test_failed_post_publish_http_check_can_restore_only_our_own_root_bytes(armed):
    control, plan, portal = _publication(armed)
    receipt = controller._rollback_owned_publication(plan, control)
    assert receipt["status"] == "restored_previous_portal"
    assert portal.read_text() == "previous portal"
    assert (Path(plan["snapshot"]) / "portal_publication.json").exists()
    assert (control / "publication_rollback.json").exists()


def test_rollback_never_clobbers_a_later_external_publication(armed):
    control, plan, portal = _publication(armed, changed=True)
    receipt = controller._rollback_owned_publication(plan, control)
    assert receipt["status"] == "not_replaced"
    assert portal.read_text() == "a later publication"


def test_failed_watcher_is_not_automatically_restarted(armed, monkeypatch):
    control, plan = armed
    monkeypatch.setattr(controller, "EVALUATIONS", control.parent.parent)
    _json(control / "plan.json", plan)
    _json(control / "plan_binding.json", controller.digest(control / "plan.json"))
    _json(control / "status.json", {"state": "failed"})
    with pytest.raises(ValueError, match="not resumed or retried"):
        controller.watch(control)


def test_mutated_plan_is_rejected_before_any_child_or_publish(armed, monkeypatch):
    control, plan = armed
    monkeypatch.setattr(controller, "EVALUATIONS", control.parent.parent)
    _json(control / "plan.json", plan)
    _json(control / "plan_binding.json", controller.digest(control / "plan.json"))
    plan["commands"] = [{"stage": "unreviewed", "argv": ["false"]}]
    _json(control / "plan.json", plan)
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        controller.watch(control)


def test_start_detaches_only_watcher_not_models_or_simulations(armed, monkeypatch):
    old_control, plan = armed
    control = old_control.parent / "new-controller"
    plan["control_dir"] = str(control)
    monkeypatch.setattr(controller, "make_plan", lambda requested: plan)
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=456)

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    result = controller.start(control)
    assert result["pid"] == 456 and result["simulation_runs_started"] == 0
    assert len(calls) == 1 and "--watch" in calls[0][0]
    assert calls[0][1]["start_new_session"] is True
    assert (control / "plan_binding.json").exists()
    assert (control / "watcher.log").exists()
