#!/usr/bin/env python3
"""Wait for the exact 25-trial job, audit an additive site, then publish it.

No model/evaluation retry or simulator is started here. --start only launches
this publication watcher; a failed check is terminal and keeps the prior site.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS = Path("/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations")
BATCH = EVALUATIONS / "_jobs/pi05_bulb_screw_inplace_h50_k50_n5_20261020"
CONTROL = EVALUATIONS / "_jobs/pi05_bulb_inplace_publication_20261020"
BASELINE = ROOT / "reports/kaitactilesim_unified_workbench_20261020_r5"
SNAPSHOT = ROOT / "reports/kaitactilesim_unified_workbench_20261020_r6"
PYTHON = ROOT / ".pixi/envs/default/bin/python"
SCHEMA = "kaihand-bulb-inplace-publication-controller-v1"
JOB_SCHEMA = "kaihand-pi05-bulb-inplace-stage-batch-v1"
BASELINE_CATALOG_SHA256 = "362d072f7401319bb59bb66a736ac8256b42e17d1d65da83c1c146e3b98a92af"
BASELINE_MANIFEST_SHA256 = "de7ea18423235b4247aa03de1ac21db0770ba3303603ae6f1dc9bcc65d899ec4"
ROOT_INDEX_SHA256 = "d600837f760eaf4888eb2de08e044c37f9311b238b216663490ce5273b2d49fd"
STEPS = (15000, 18000, 21000, 24000, 24999)
SCRIPTS = {
    "exporter": ROOT / "scripts/analysis/export_additive_evaluation_workbench.py",
    "independent_qa": ROOT / "scripts/analysis/check_bulb_inplace_evaluation.py",
    "browser_qa": ROOT / "scripts/analysis/check_additive_evaluation_workbench_browser.cjs",
}
IMPLEMENTATIONS = (
    "scripts/analysis/evaluation_status/additive_workbench_export.py",
    "scripts/analysis/evaluation_status/data.py", "scripts/analysis/evaluation_status/export.py",
    "scripts/analysis/evaluation_status/rich_detail_data.py", "scripts/analysis/evaluation_status/rich_reference_data.py",
    "scripts/analysis/evaluation_status/rich_workbench_export.py", "scripts/analysis/evaluation_status/reference_workbench_export.py",
    "scripts/analysis/evaluation_status/unified_workbench_export.py", "scripts/analysis/evaluation_status/unified_workbench_page.py",
    "scripts/analysis/evaluation_unified/staged.py", "scripts/analysis/evaluation_workbench/goal_references.py",
    "scripts/analysis/evaluation_workbench/task_fields.py", "scripts/analysis/evaluation_workbench/data.py",
    "scripts/analysis/evaluation_workbench/experts.py", "scripts/analysis/evaluation_workbench/model_plots.py",
    "scripts/analysis/evaluation_workbench/suite.py", "scripts/analysis/evaluation_workbench/diagnostic_plots.py",
)
BROWSER_ENV = {
    "NODE_PATH": "/tmp/contact_web_qa.CJzQS7/node_modules",
    "LD_LIBRARY_PATH": "/tmp/contact_web_qa.CJzQS7/runtime-libs/usr/lib/x86_64-linux-gnu",
    "KAI_EVAL_CHROME": "/tmp/contact_web_qa.CJzQS7/browsers/chromium-1248/chrome-linux64/chrome",
}


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("Expected bounded regular JSON: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular non-symlink dependency: " + str(path))
    before = path.stat()
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Dependency changed while hashing: " + str(path))
    return {"path": str(path.absolute()), "size": after.st_size, "sha256": value}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def update_status(control, state):
    state["updated_utc"] = stamp()
    descriptor, name = tempfile.mkstemp(prefix=".publication-status-", dir=control)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    Path(name).replace(control / "status.json")


def check_rows(rows):
    for row in rows:
        actual = digest(row["path"])
        if (actual["sha256"], actual["size"]) != (row["sha256"], row["size"]):
            raise ValueError("Frozen dependency changed: " + row["path"])


def process_identity(pid):
    """Read PID/start tick only; never signal or adopt a replacement process."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] in {"Z", "X", "x"}:
            return None
        return {"pid": pid, "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def producer_guard(plan, status, *, now=None):
    if (time.time() if now is None else now) >= plan["wait_deadline_unix"]:
        raise ValueError("24-hour publication wait deadline reached; no automatic evaluation or publication retry")
    expected = plan["producer_identity"]
    if status.get("controller_pid") != expected["pid"] or process_identity(expected["pid"]) != expected:
        raise ValueError("Evaluation producer disappeared, became a zombie, or changed identity; no retry or signal")


def code_inventory():
    return sorted(str(path.absolute()) for path in (ROOT / "scripts/analysis").rglob("*")
                  if path.is_file() and path.suffix in {".py", ".js", ".cjs", ".css"})


def guards(plan, *, portal=True):
    check_rows(plan["source_files"])
    if "code_inventory" in plan and code_inventory() != plan["code_inventory"]:
        raise ValueError("Offline analysis implementation file membership changed while armed")
    check_rows(plan["fixed_inputs"])
    if portal:
        check_rows([plan["previous_root_index"]])


def _child_environment(plan, temporary):
    environment = dict(os.environ)
    environment.update(plan["environment"])
    environment["TMPDIR"] = str(temporary)
    # Only QA/publication child processes are changed; no global proxy settings.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = environment["NO_PROXY"]
    return environment


def make_plan(control, *, batch=BATCH, baseline=BASELINE, snapshot=SNAPSHOT):
    control, batch, baseline, snapshot = map(lambda path: Path(path).absolute(), (control, batch, baseline, snapshot))
    if control.parent != EVALUATIONS / "_jobs" or control.exists() or control.is_symlink():
        raise ValueError("Choose a fresh control directory directly inside evaluation-root/_jobs")
    if batch != BATCH or baseline != BASELINE or snapshot != SNAPSHOT:
        raise ValueError("This controller only publishes the explicitly reviewed batch and r5-to-r6 extension")
    if snapshot.exists() or snapshot.is_symlink() or snapshot.parent.resolve(strict=True) != baseline.parent.resolve(strict=True):
        raise ValueError("r6 must be a fresh sibling output; no overwrite or automatic retry")
    batch_plan = read_json(batch / "batch.json")
    batch_status = read_json(batch / "status.json")
    if batch_plan.get("schema") != JOB_SCHEMA or batch_status.get("schema") != JOB_SCHEMA:
        raise ValueError("Requested job schema differs")
    if batch_status.get("state", batch_status.get("status")) not in {"running", "complete"}:
        raise ValueError("Requested job is not running or complete")
    if (batch_plan.get("initial_state_randomization") is not False
            or batch_plan.get("trial_scope") != "fixed_initial_state_repeats"
            or [group.get("step") for group in batch_plan.get("groups", [])] != list(STEPS)
            or any(group.get("task") != "bulb-screw-inplace" for group in batch_plan.get("groups", []))):
        raise ValueError("Requested task/checkpoints/fixed-initial-state scope changed")
    expected = ((baseline / "catalog.json", BASELINE_CATALOG_SHA256),
                (baseline / "artifact_manifest.json", BASELINE_MANIFEST_SHA256),
                (baseline.parent / "index.html", ROOT_INDEX_SHA256))
    for path, sha in expected:
        if digest(path)["sha256"] != sha:
            raise ValueError("Accepted baseline or shared portal changed before arming: " + str(path))
    node = shutil.which("node")
    if not node or not PYTHON.is_file():
        raise ValueError("Reviewed Python/Node executable is missing")
    if not Path(BROWSER_ENV["KAI_EVAL_CHROME"]).is_file() or not Path(BROWSER_ENV["NODE_PATH"]).is_dir():
        raise ValueError("Reviewed browser executable/dependencies are missing")
    inventory = code_inventory()
    files = sorted({Path(__file__).absolute(), *SCRIPTS.values(), *(ROOT / relative for relative in IMPLEMENTATIONS),
                    *(Path(path) for path in inventory)})
    producer = process_identity(batch_status.get("controller_pid"))
    if batch_status.get("state", batch_status.get("status")) != "complete" and producer is None:
        raise ValueError("Running batch has no live, non-zombie producer identity")
    index_url = "http://127.0.0.1:8877/" + urllib.parse.quote(snapshot.name) + "/index.html"
    baseline_url = "http://127.0.0.1:8877/" + urllib.parse.quote(baseline.name) + "/index.html"
    common = [str(PYTHON), "-B", str(SCRIPTS["exporter"]), "--output", str(snapshot)]
    return {
        "schema": SCHEMA, "created_utc": stamp(), "control_dir": str(control), "batch_root": str(batch),
        "baseline": str(baseline), "snapshot": str(snapshot), "evaluation_root": str(EVALUATIONS),
        "poll_seconds": 30, "simulation_runs_started": 0, "automatic_retries": 0,
        "producer_identity": producer, "wait_deadline_unix": time.time() + 24 * 60 * 60,
        "max_wait_seconds": 24 * 60 * 60, "code_inventory": inventory,
        "previous_root_index": digest(baseline.parent / "index.html"),
        "fixed_inputs": [digest(batch / "batch.json"), digest(baseline / "catalog.json"), digest(baseline / "artifact_manifest.json")],
        "source_files": [digest(path) for path in files],
        "environment": {"PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts/analysis"), **BROWSER_ENV},
        "root_url": "http://127.0.0.1:8877/", "snapshot_url": index_url,
        "commands": [
            {"stage": "export", "argv": [*common, "--baseline", str(baseline), "--job-root", str(batch),
                                           "--evaluation-root", str(EVALUATIONS), "--workers", "2"]},
            {"stage": "independent_qa", "argv": [str(PYTHON), "-B", str(SCRIPTS["independent_qa"]),
                "--batch", str(batch), "--snapshot", str(snapshot), "--baseline", str(baseline),
                "--http-base", index_url, "--output", str(control / "independent_qa.json")]},
            {"stage": "browser_qa", "argv": [node, str(SCRIPTS["browser_qa"]), index_url,
                str(control / "browser"), "--baseline=" + baseline_url]},
            {"stage": "verify", "argv": [*common, "--verify-only"]},
            {"stage": "publish", "argv": [*common, "--publish-only"]},
        ],
    }


def start(control):
    control = Path(control).absolute()
    plan = make_plan(control)
    control.mkdir()
    write_new(control / "plan.json", plan)
    plan_row = digest(control / "plan.json")
    write_new(control / "plan_binding.json", plan_row)
    update_status(control, {"schema": SCHEMA, "state": "starting", "stage": "detaching_watcher", "children": []})
    with (control / "watcher.log").open("xb") as log:
        child = subprocess.Popen([str(PYTHON), "-B", str(Path(__file__).resolve()), "--watch", "--control-dir", str(control)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True, start_new_session=True, env=_child_environment(plan, "/dev/shm"))
    receipt = {"schema": SCHEMA, "state": "started", "pid": child.pid, "plan": plan_row,
               "control_dir": str(control), "log": str(control / "watcher.log"), "simulation_runs_started": 0}
    write_new(control / "launcher.json", receipt)
    return receipt


def run_child(plan, command, control, state, environment):
    guards(plan)
    stage = command["stage"]
    state.update(state="running", stage=stage)
    entry = {"stage": stage, "argv": command["argv"], "started_utc": stamp(), "returncode": None}
    state["children"].append(entry)
    update_status(control, state)
    with (control / (stage + ".log")).open("xb") as log:
        child = subprocess.Popen(command["argv"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, env=environment, close_fds=True)
        entry["pid"] = child.pid
        update_status(control, state)
        entry["returncode"] = child.wait()
    entry["finished_utc"] = stamp()
    update_status(control, state)
    if entry["returncode"] != 0:
        raise RuntimeError(f"{stage} failed with exit {entry['returncode']}; see {stage}.log; no automatic retry")
    if stage in {"independent_qa", "browser_qa"}:
        result_path = control / ("independent_qa.json" if stage == "independent_qa" else "browser/additive_browser_verification.json")
        result = read_json(result_path)
        if result.get("status") != "passed":
            raise ValueError(stage + " did not produce a passed independent receipt")
        entry["receipt"] = digest(result_path)
        update_status(control, state)


def _http_final(plan):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch(url):
        with opener.open(url, timeout=30) as response:
            if response.status != 200:
                raise ValueError("Platform HTTP request did not return 200")
            return response.read(64 * 1024 * 1024)

    snapshot = Path(plan["snapshot"])
    root_bytes = fetch(plan["root_url"])
    if root_bytes != (snapshot.parent / "index.html").read_bytes() or (snapshot.name + "/index.html").encode() not in root_bytes:
        raise ValueError("Shared root does not serve the new snapshot landing")
    index_bytes = fetch(plan["snapshot_url"])
    if index_bytes != (snapshot / "index.html").read_bytes():
        raise ValueError("Served new workbench differs from the verified snapshot")
    url = urllib.parse.urljoin(plan["snapshot_url"], "catalog.json")
    catalog_bytes = fetch(url)
    if catalog_bytes != (snapshot / "catalog.json").read_bytes():
        raise ValueError("Served new catalog differs from verified bytes")
    counts = json.loads(catalog_bytes)["counts"]
    required = {"members": 620, "eligible": 600, "quarantined": 20, "groups": 52,
                "tasks": 11, "models": 4, "staged_verified": 475, "staged_unknown": 145}
    if any(counts.get(key) != value for key, value in required.items()):
        raise ValueError("Served platform counts differ from the accepted extension")
    return {"status": "passed", "root_url": plan["root_url"], "snapshot_url": plan["snapshot_url"], "counts": counts,
            "root_sha256": hashlib.sha256(root_bytes).hexdigest(), "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest()}


def _rollback_owned_publication(plan, control):
    """Restore the prior portal only if this watcher still owns the new bytes."""
    publication_path = Path(plan["snapshot"]) / "portal_publication.json"
    if not publication_path.is_file():
        return None
    publication = read_json(publication_path)
    root_index = Path(plan["previous_root_index"]["path"])
    if digest(root_index)["sha256"] != publication.get("new_sha256"):
        result = {"status": "not_replaced", "reason": "Shared portal no longer contains this watcher's publication"}
    else:
        previous = Path(plan["snapshot"]) / "previous_root_index.html"
        if digest(previous)["sha256"] != plan["previous_root_index"]["sha256"]:
            raise ValueError("Cannot roll back: saved previous portal differs from the armed bytes")
        lock_path = root_index.parent / ".evaluation-status-publish.lock"
        with lock_path.open("x") as lock:
            lock.write(json.dumps({"pid": os.getpid(), "rollback_of": plan["snapshot"]}))
        try:
            if digest(root_index)["sha256"] != publication.get("new_sha256"):
                result = {"status": "not_replaced", "reason": "Shared portal changed while acquiring publication lock"}
            else:
                descriptor, temporary = tempfile.mkstemp(prefix=".bulb-publication-rollback-", dir=root_index.parent)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(previous.read_bytes())
                Path(temporary).replace(root_index)
                result = {"status": "restored_previous_portal", "root_index": digest(root_index),
                          "reason": "Post-publication acceptance failed; only this watcher's own portal was replaced"}
        finally:
            lock_path.unlink()
    write_new(control / "publication_rollback.json", result)
    return result


def watch(control):
    control = Path(control).absolute()
    if control.parent != EVALUATIONS / "_jobs" or control.is_symlink():
        raise ValueError("Watcher control directory must be a canonical evaluation-root/_jobs child")
    binding = read_json(control / "plan_binding.json")
    check_rows([binding])
    plan = read_json(control / "plan.json")
    if plan.get("schema") != SCHEMA or plan.get("control_dir") != str(control):
        raise ValueError("Invalid watcher plan identity")
    state = read_json(control / "status.json")
    if state.get("state") in {"complete", "failed"}:
        raise ValueError("A terminal watcher is not resumed or retried automatically")
    with (control / ".watch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.update(watcher_pid=os.getpid(), state="waiting", stage="waiting_for_exact_batch", children=[])
        try:
            while True:
                guards(plan)
                status = read_json(Path(plan["batch_root"]) / "status.json")
                if status.get("schema") != JOB_SCHEMA:
                    raise ValueError("Evaluation status schema changed")
                batch_state = status.get("state", status.get("status"))
                state["batch_state"] = batch_state
                state["batch_groups"] = [{"state": row.get("state"), "attempt": row.get("attempt")}
                                         for row in status.get("groups", [])]
                update_status(control, state)
                if batch_state == "complete":
                    break
                if batch_state != "running":
                    raise ValueError("Evaluation ended without a complete accepted batch: " + str(batch_state))
                if any(row.get("state") in {"failed", "error", "cancelled"} for row in status.get("groups", [])):
                    raise ValueError("An evaluation group failed; no publication or evaluation retry")
                producer_guard(plan, status)
                time.sleep(plan["poll_seconds"])
            # Strict exact-job native/staged admissions, without executing rules.
            sys.path.insert(0, str(ROOT / "scripts/analysis"))
            sys.path.insert(0, str(ROOT / "src"))
            from evaluation_status.additive_workbench_export import _selected_job

            groups, inputs, batch = _selected_job(plan["batch_root"], plan["evaluation_root"])
            acceptance = {"schema": SCHEMA, "status": "passed", "accepted_utc": stamp(),
                "groups": len(groups), "trials": sum(len(group["trials"]) for group in groups),
                "steps": list(STEPS), "trial_scope": batch["trial_scope"], "source_files": inputs,
                "batch_status": digest(Path(plan["batch_root"]) / "status.json"), "simulation_runs_started": 0}
            write_new(control / "batch_acceptance.json", acceptance)
            state["batch_acceptance"] = digest(control / "batch_acceptance.json")
            if Path(plan["snapshot"]).exists():
                raise ValueError("New snapshot target appeared; no overwrite or retry")
            temporary = Path(tempfile.mkdtemp(prefix="bulb-inplace-publication-", dir="/dev/shm"))
            state["temporary_directory"] = str(temporary)
            environment = _child_environment(plan, temporary)
            for command in plan["commands"]:
                # Producer metadata is immutable after accepted completion.
                check_rows([acceptance["batch_status"], *inputs])
                run_child(plan, command, control, state, environment)
            guards(plan, portal=False)
            http = _http_final(plan)
            write_new(control / "http_verification.json", http)
            final = {"schema": SCHEMA, "status": "passed", "completed_utc": stamp(),
                     "counts": http["counts"], "snapshot": plan["snapshot"], "root_url": plan["root_url"],
                     "checks": {"strict_job": state["batch_acceptance"],
                                "independent_qa": digest(control / "independent_qa.json"),
                                "browser_qa": digest(control / "browser/additive_browser_verification.json"),
                                "http": digest(control / "http_verification.json")},
                     "existing_595_scores_and_episode_bytes_preserved": True,
                     "simulation_runs_started": 0, "automatic_retries": 0, "children": state["children"]}
            write_new(control / "acceptance.json", final)
            state.update(state="complete", stage="published_and_http_verified", acceptance=digest(control / "acceptance.json"))
            update_status(control, state)
            return final
        except Exception as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", failed_utc=stamp())
            try:
                published_here = any(entry.get("stage") == "publish" and entry.get("returncode") == 0
                                     for entry in state.get("children", []))
                rollback = _rollback_owned_publication(plan, control) if published_here else None
                if rollback is not None:
                    state["publication_rollback"] = rollback
            except Exception as rollback_error:
                state["publication_rollback_error"] = f"{type(rollback_error).__name__}: {rollback_error}"
            update_status(control, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--start", action="store_true")
    modes.add_argument("--watch", action="store_true")
    parser.add_argument("--control-dir", type=Path, default=CONTROL)
    args = parser.parse_args()
    result = start(args.control_dir) if args.start else watch(args.control_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
