"""Read completed archived staged rollouts without importing a simulator.

Recorded labels, cumulative stage achievements and final measurements remain
separate.  No current task rule is executed, and no video/tactile clock is
silently aligned to the staged observer's clock.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import statistics
from pathlib import Path, PurePosixPath

MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_STATE_BYTES = 128 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
MAX_STATE_ROWS = 250_000
MAX_CANDIDATES = 20_000
MAX_INVENTORY_FILE_BYTES = 256 * 1024 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RULE_FILES = {
    "observer_sha256": "src/kaihand_tactile_env/shared/staged_evaluation.py",
    "recording_sha256": "src/kaihand_tactile_env/shared/staged_recording.py",
}
_IDENTITY_KEYS = ("task", "seed", "prediction_horizon", "checkpoint_sha256", "deployment_id")
_TACTILE_INTEGRITY_PROFILE = "tactile_manifest_inventory_v1"
_TACTILE_RUNTIME_FILES = {
    "observer_sha256": "src/kaihand_tactile_env/shared/staged_evaluation.py",
    "recording_sha256": "src/kaihand_tactile_env/shared/staged_recording.py",
    "evaluation_rules_sha256": "src/kaihand_tactile_env/shared/evaluation_rules.py",
    "deployment_loader_sha256": "src/kaihand_tactile_env/shared/tactile_deployment.py",
    "runner_sha256": "scripts/workcell/run_poker_pi05_policy.py",
    "tactile_runner_sha256": "scripts/workcell/run_poker_tactile_policy.py",
    "binding_sha256": "scripts/workcell/tactile_policy_binding.py",
    "worker_sha256": "scripts/workcell/tactile_policy_worker.py",
    "batch_runner_sha256": "scripts/workcell/evaluate_poker_tactile_policy_batch.py",
}
_INVENTORY_FILE_CACHE = {}


class StagedImportError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _fail(message, status="invalid"):
    raise StagedImportError(status, message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    _fail(f"Nonfinite JSON number: {value}")


def _float(value):
    value = float(value)
    if not math.isfinite(value):
        _fail("Nonfinite JSON number")
    return value


def _parse(raw):
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant,
                       parse_float=_float)
    if not isinstance(value, dict):
        _fail("Expected a JSON object")
    return value


def _file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _open(path, limit, *, allow_empty=False):
    # Parent aliases can resolve to canonical storage; the actual artifact must
    # be a regular non-symlink leaf and is checked again after reading.
    path = Path(path).absolute()
    if path.is_symlink():
        _fail(f"Artifact leaf must not be a symlink: {path}")
    handle = os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                              | getattr(os, "O_NONBLOCK", 0)), "rb")
    before = os.fstat(handle.fileno())
    minimum = 0 if allow_empty else 1
    if (not stat.S_ISREG(before.st_mode)
            or before.st_size < minimum or before.st_size > limit):
        handle.close()
        _fail(f"Artifact is not a bounded regular file: {path}")
    return path, handle, before


def _finish(path, handle, before, digest, role):
    after = os.fstat(handle.fileno())
    if (_file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(path.stat())):
        _fail(f"Source changed while reading: {path}")
    result = {"path": str(path), "size": after.st_size, "sha256": digest, "role": role}
    if path.resolve() != path:
        result["canonical_path"] = str(path.resolve())
    return result


def _read(path, role, provenance):
    path, handle, before = _open(path, MAX_METADATA_BYTES)
    with handle:
        raw = handle.read(MAX_METADATA_BYTES + 1)
        if len(raw) != before.st_size or len(raw) > MAX_METADATA_BYTES:
            _fail(f"Source changed or exceeded size bound: {path}")
        row = _finish(path, handle, before, hashlib.sha256(raw).hexdigest(), role)
    value = _parse(raw)
    provenance.append(row)
    return value


def _inventory_file(path, expected_sha256):
    path = Path(path).absolute()
    if path.is_symlink() or not _sha(expected_sha256):
        _fail(f"Invalid tactile inventory file declaration: {path}")
    try:
        current = _file_identity(path.stat())
    except OSError as error:
        _fail(f"Missing tactile inventory file: {path}: {error}")
    cached = _INVENTORY_FILE_CACHE.get(str(path))
    if cached is not None:
        if cached["expected_sha256"] != expected_sha256 or cached["identity"] != current:
            _fail(f"Tactile inventory file changed during import: {path}")
        return dict(cached["row"])
    # Frozen source inventories legitimately include empty ``__init__.py``
    # files.  Metadata and state artifacts remain non-empty by default.
    path, handle, before = _open(path, MAX_INVENTORY_FILE_BYTES, allow_empty=True)
    checksum, size = hashlib.sha256(), 0
    with handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            checksum.update(block)
            size += len(block)
        row = _finish(path, handle, before, checksum.hexdigest(), "tactile_inventory_file")
    if size != before.st_size or row["sha256"] != expected_sha256:
        _fail(f"Tactile inventory file SHA mismatch: {path}")
    _INVENTORY_FILE_CACHE[str(path)] = {
        "expected_sha256": expected_sha256,
        "identity": _file_identity(path.stat()),
        "row": row,
    }
    return dict(row)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _identity(summary, required_prediction_horizon=50):
    if type(required_prediction_horizon) is not int or required_prediction_horizon <= 0:
        raise ValueError("required_prediction_horizon must be a positive integer")
    identity = {key: summary.get(key) for key in _IDENTITY_KEYS}
    if not isinstance(identity["task"], str) or not identity["task"]:
        _fail("Missing explicit task identity")
    if type(identity["seed"]) is not int or identity["seed"] < 0:
        _fail("Invalid seed identity")
    if type(identity["prediction_horizon"]) is not int:
        _fail("Missing or invalid explicit prediction_horizon", "unknown_horizon")
    if identity["prediction_horizon"] != required_prediction_horizon:
        _fail(
            "Only explicitly declared "
            f"prediction_horizon={required_prediction_horizon} is included",
            "wrong_horizon",
        )
    if not _sha(identity["checkpoint_sha256"]) or not _sha(identity["deployment_id"]):
        _fail("Missing checkpoint/deployment content identity")
    identity["checkpoint_path"] = summary.get("checkpoint_path")
    identity["model_family"] = summary.get("model_family")
    return identity


def _stage_contract(staged):
    if staged.get("version") != "staged_v1":
        _fail("Missing/unsupported explicit staged evaluation version", "not_staged")
    if "completion_protocol" in staged:
        protocol = staged["completion_protocol"]
        if protocol is not None and (not isinstance(protocol, str) or not protocol):
            _fail("Recorded completion_protocol must be a nonempty string or null")
    definitions, flags = staged.get("stage_definitions"), staged.get("stages")
    if (not isinstance(definitions, dict) or not definitions or not isinstance(flags, dict)
            or set(definitions) != set(flags)
            or any(not isinstance(v, str) or not v for v in definitions.values())
            or len(set(definitions.values())) != len(definitions)
            or any(type(v) is not bool for v in flags.values())):
        _fail("Invalid recorded stage definitions/boolean flags")
    if _canonical(staged.get("stage_flags")) != _canonical(flags):
        _fail("Recorded stage_flags contradict stages")
    if _canonical(staged.get("named_stages")) != _canonical({definitions[k]: v for k, v in flags.items()}):
        _fail("Recorded named_stages contradict stage definitions")
    if type(staged.get("stage_count")) is not int or staged["stage_count"] != len(flags):
        _fail("Recorded stage_count contradicts flags")
    first = staged.get("first_times")
    if not isinstance(first, dict) or set(first) != set(flags):
        _fail("Invalid recorded first_times")
    for key, value in first.items():
        if value is not None and (not _number(value) or value < 0):
            _fail("Invalid recorded first achievement time")
        if flags[key] is False and value is not None:
            _fail("Unachieved stage has a recorded achievement time")
    if staged.get("complete_success") is not None and type(staged["complete_success"]) is not bool:
        _fail("complete_success must be boolean or null")
    if staged.get("success") != staged.get("complete_success"):
        _fail("Recorded staged success fields disagree")
    thresholds = staged.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds or any(not _number(v) for v in thresholds.values()):
        _fail("Missing/invalid recorded thresholds")
    if not isinstance(staged.get("measurements"), dict):
        _fail("Missing recorded final measurements")
    if staged.get("time_basis") != "mujoco_data_time_seconds":
        _fail("Unknown staged observation clock")


def _completion(summary, staged):
    if summary.get("status") not in ("success", "task_not_completed"):
        _fail("Run has no normal terminal summary status", "not_completed")
    if summary.get("error") or summary.get("record_error"):
        _fail("Run records an execution/recording error", "not_completed")
    if summary.get("full_duration_evaluation") is not True or staged.get("full_duration_evaluation") is not True:
        _fail("Full-duration staged run is not explicitly recorded", "not_completed")
    end, limit = summary.get("sim_seconds"), summary.get("max_sim_seconds")
    if not _number(end) or not _number(limit) or limit <= 0 or end + 1e-8 < limit:
        _fail("Recorded run has not reached its declared duration", "not_completed")
    if staged.get("state_log") != "staged_states.jsonl":
        _fail("Unknown or unsafe staged state-log filename")
    if type(staged.get("state_log_samples")) is not int or staged["state_log_samples"] < 1:
        _fail("No completed staged state-log samples", "not_completed")
    video = summary.get("video")
    if isinstance(video, dict) and video.get("completed") is not True:
        _fail("Recorded video is not finalized", "not_completed")
    if type(summary.get("task_success")) is not bool:
        _fail("Native task_success must be an explicit boolean")
    if type(staged.get("legacy_task_success")) is not bool or staged["legacy_task_success"] != summary["task_success"]:
        _fail("Staged legacy_task_success contradicts the native label")
    return {"completed": True, "basis": "terminal_summary_and_full_recorded_duration",
            "sim_seconds": end, "required_duration_s": limit,
            "native_task_success_is_not_completion_filter": True}


def _manifest_path(summary, trial):
    path = summary.get("deployment_manifest")
    if not isinstance(path, str) or not path:
        _fail("Missing bound deployment manifest path")
    path = Path(path)
    if not path.is_absolute():
        path = trial / path
    if not re.fullmatch(r"deployment_manifest(?:\.[0-9a-f]{64})?\.json", path.name):
        _fail("Unsupported deployment manifest filename")
    return path


def _recorded_contract_identity(staged, identity, binding):
    contract = {key: staged.get(key) for key in (
        "version", "task", "stage_definitions", "thresholds", "stage_achievement_policy", "time_basis")}
    if "completion_protocol" in staged:
        # Preserve only archived evidence, including an explicitly unknown
        # null. Never fill an older archive from today's task catalog.
        contract["completion_protocol"] = staged["completion_protocol"]
        identity["completion_protocol"] = staged["completion_protocol"]
    identity["definitions_thresholds_sha256"] = _digest(contract)
    identity["binding"] = binding
    identity["sha256"] = _digest(identity)
    identity["current_rules_reexecuted"] = False
    return identity


def _rules(summary, staged, trial, provenance):
    integrity = summary.get("deployment_integrity")
    if not isinstance(integrity, dict) or integrity.get("status") != "verified" or integrity.get("errors"):
        _fail("Deployment integrity is not recorded as verified")
    source = integrity.get("evaluation_source", {})
    if (source.get("status") != "verified" or not _sha(source.get("actual_sha256"))
            or source.get("expected_sha256") != source.get("actual_sha256")
            or source.get("errors") or source.get("changed_files")):
        _fail("Recorded evaluation-source integrity is invalid")
    manifest_declaration = integrity.get("manifest", {})
    if not isinstance(manifest_declaration, dict):
        _fail("Invalid deployment manifest integrity declaration")
    if (manifest_declaration.get("status") not in (None, "verified")
            or manifest_declaration.get("errors") or manifest_declaration.get("error")):
        _fail("Manifest integrity contains a recorded error")
    hashes = [value for value in [manifest_declaration.get("expected_sha256"),
              manifest_declaration.get("actual_sha256"), integrity.get("manifest_sha256")]
              if value is not None]
    if not hashes or not all(_sha(v) for v in hashes) or len(set(hashes)) != 1:
        _fail("Missing/contradictory manifest SHA binding")
    path = _manifest_path(summary, trial)
    manifest = _read(path, "staged_deployment_manifest", provenance)
    if provenance[-1]["sha256"] != hashes[0]:
        _fail("Deployment manifest SHA mismatch")
    for key in ("task", "checkpoint_sha256", "deployment_id", "prediction_horizon"):
        if manifest.get(key) != summary.get(key) or type(manifest.get(key)) is not type(summary.get(key)):
            _fail(f"Manifest identity mismatch: {key}")
    rows = manifest.get("evaluation_source_files")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        _fail("Missing frozen evaluation-source inventory")
    inventory_sha = _digest(rows)
    if inventory_sha != manifest.get("evaluation_source_sha256") or inventory_sha != source["actual_sha256"]:
        _fail("Evaluation-source inventory SHA mismatch")
    paths = [row.get("path") for row in rows]
    if any(not isinstance(p, str) or PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts for p in paths):
        _fail("Unsafe source inventory path")
    if len(set(paths)) != len(paths):
        _fail("Duplicate source inventory paths")
    identity = {"evaluation_source_sha256": inventory_sha}
    for name, relative in _RULE_FILES.items():
        match = [row for row in rows if row["path"] == relative]
        if len(match) != 1 or not _sha(match[0].get("sha256")):
            _fail(f"Missing archived staged rule identity: {relative}")
        identity[name] = match[0]["sha256"]
    return _recorded_contract_identity(
        staged, identity, "sha_verified_manifest_inventory_and_recorded_contract"
    )


def _tactile_rules(summary, staged, trial, provenance):
    path = _manifest_path(summary, trial)
    manifest = _read(path, "tactile_staged_deployment_manifest", provenance)
    manifest_sha = provenance[-1]["sha256"]
    summary_sha = summary.get("deployment_manifest_sha256")
    server = summary.get("server_metadata")
    if not _sha(summary_sha) or not isinstance(server, dict):
        _fail("Missing tactile manifest SHA or worker ready handshake")
    if manifest_sha != summary_sha or server.get("deployment_manifest_sha256") != manifest_sha:
        _fail("Tactile manifest SHA/worker handshake binding mismatch")
    if manifest.get("schema") != "poker_tactile_deployment_v1":
        _fail("Tactile staged profile requires poker_tactile_deployment_v1")
    family = manifest.get("model_family")
    if family not in {"pi05+cpm", "ftp1"}:
        _fail("Unsupported tactile model family")
    declared_id = manifest.get("deployment_id")
    if not _sha(declared_id) or declared_id != _digest(
            {key: value for key, value in manifest.items() if key != "deployment_id"}):
        _fail("Tactile manifest deployment_id does not authenticate its content")
    expected_horizon = 30 if family == "pi05+cpm" else 32
    expected_offset = 0 if family == "pi05+cpm" else 1
    expected_model_dim = 32 if family == "pi05+cpm" else 120
    fixed = {
        "task": "poker-draw",
        "model_family": family,
        "deployment_id": declared_id,
        "checkpoint_path": manifest.get("checkpoint_path"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "prediction_horizon": expected_horizon,
        "action_start_index": expected_offset,
        "execute_steps": manifest.get("execute_steps"),
    }
    if (manifest.get("prediction_horizon") != expected_horizon
            or manifest.get("action_start_index") != expected_offset
            or type(manifest.get("execute_steps")) is not int
            or not 1 <= manifest["execute_steps"] <= expected_horizon - expected_offset):
        _fail("Invalid tactile manifest horizon/action window")
    for key, value in fixed.items():
        if (manifest.get(key) != value or type(manifest.get(key)) is not type(value)
                or summary.get(key) != value or type(summary.get(key)) is not type(value)
                or server.get(key) != value or type(server.get(key)) is not type(value)):
            _fail(f"Tactile summary/worker/manifest identity mismatch: {key}")
    if (not isinstance(manifest.get("checkpoint_path"), str)
            or not Path(manifest["checkpoint_path"]).is_absolute()
            or not _sha(manifest.get("checkpoint_sha256"))
            or manifest.get("model_action_dim") != expected_model_dim
            or type(manifest.get("model_action_dim")) is not int
            or manifest.get("action_dim") != 27
            or type(manifest.get("action_dim")) is not int
            or manifest.get("control_hz") != 30
            or type(manifest.get("control_hz")) is not int
            or manifest.get("controller_version") != "poker-tactile-direct-joints-v1"
            or manifest.get("required_action_interpolation") != "zoh"):
        _fail("Invalid tactile manifest model/control contract")
    for key in ("checkpoint_step", "model_action_dim", "action_dim", "control_hz",
                "controller_version", "required_action_interpolation"):
        summary_key = "controller" if key == "controller_version" else key
        if (summary.get(summary_key) != manifest.get(key)
                or type(summary.get(summary_key)) is not type(manifest.get(key))):
            _fail(f"Tactile summary/manifest identity mismatch: {key}")
    for key in ("model_action_dim", "action_dim", "control_hz", "joint_names"):
        if (server.get(key) != manifest.get(key)
                or type(server.get(key)) is not type(manifest.get(key))):
            _fail(f"Tactile worker/manifest identity mismatch: {key}")
    if (summary.get("tactile_sent_to_model") is not True
            or server.get("tactile_sent_to_model") is not True
            or summary.get("inference_backend") != "isolated-local-worker"
            or summary.get("simulation_paused_during_inference") is not True):
        _fail("Tactile worker execution contract is not recorded")
    if (staged.get("execute_steps") != manifest["execute_steps"]
            or type(staged.get("execute_steps")) is not int):
        _fail("Staged result does not bind tactile execute_steps")
    inventory = manifest.get("files")
    if (not isinstance(inventory, dict) or not inventory
            or any(not isinstance(name, str) or not Path(name).is_absolute()
                   or not _sha(value) for name, value in inventory.items())):
        _fail("Invalid tactile manifest file inventory")
    checkpoint = Path(manifest["checkpoint_path"])
    checkpoint_file = checkpoint if family == "pi05+cpm" else checkpoint / "model.safetensors"
    if inventory.get(str(checkpoint_file)) != manifest["checkpoint_sha256"]:
        _fail("Tactile checkpoint SHA is not bound in the manifest inventory")
    verified_rows = []
    for name, expected_sha in sorted(inventory.items()):
        row = _inventory_file(name, expected_sha)
        provenance.append(row)
        verified_rows.append({key: row[key] for key in ("path", "size", "sha256")})
    runtime = dict(_TACTILE_RUNTIME_FILES)
    runtime["backend_sha256"] = (
        "scripts/workcell/cpm_policy_backend.py" if family == "pi05+cpm"
        else "scripts/workcell/ftp1_policy_backend.py"
    )
    identity = {
        "evaluation_source_sha256": _digest(inventory),
        "tactile_manifest_sha256": manifest_sha,
        "tactile_manifest_files_sha256": _digest(inventory),
        "tactile_verified_inventory_sha256": _digest(verified_rows),
        "worker_handshake_sha256": _digest(server),
    }
    for key, suffix in runtime.items():
        matches = [(name, value) for name, value in inventory.items() if name.endswith(suffix)]
        if len(matches) != 1:
            _fail(f"Missing/ambiguous tactile runtime identity: {suffix}")
        identity[key] = matches[0][1]
    identity["runtime_chain_sha256"] = _digest({key: identity[key] for key in runtime})
    identity["binding_basis"] = _TACTILE_INTEGRITY_PROFILE
    return _recorded_contract_identity(staged, identity, _TACTILE_INTEGRITY_PROFILE)


def _inspect(
    trial_dir,
    required_prediction_horizon=50,
    integrity_profile="h50_deployment_integrity",
):
    trial = Path(trial_dir).absolute()
    provenance = []
    summary = _read(trial / "summary.json", "staged_summary", provenance)
    identity = _identity(summary, required_prediction_horizon)
    embedded = summary.get("staged_evaluation")
    if not isinstance(embedded, dict):
        _fail("No explicit staged evaluation payload", "not_staged")
    staged = _read(trial / "staged_evaluation.json", "staged_result", provenance)
    if _canonical(staged) != _canonical(embedded):
        _fail("Staged sidecar differs from its summary payload")
    for key in _IDENTITY_KEYS:
        if staged.get(key) != summary.get(key) or type(staged.get(key)) is not type(summary.get(key)):
            _fail(f"Staged result identity mismatch: {key}")
    _stage_contract(staged)
    completion = _completion(summary, staged)
    if integrity_profile == "h50_deployment_integrity":
        rule_identity = _rules(summary, staged, trial, provenance)
    elif integrity_profile == _TACTILE_INTEGRITY_PROFILE:
        rule_identity = _tactile_rules(summary, staged, trial, provenance)
    else:
        raise ValueError(f"Unsupported staged integrity profile: {integrity_profile}")
    return trial, summary, staged, identity, completion, rule_identity, provenance


def discover_staged_trials(root: Path, *, include_jobs: bool = False, extra_roots=()) -> dict:
    """Discover bounded canonical cohorts; folder names never establish H50.

    Job directories are opt-in. Byte-identical aliases are deduplicated with
    canonical paths preferred; conflicting copies of the same recorded run
    identity are all excluded, never selected by success or highest stage.
    """
    root = Path(root).absolute()
    if not root.is_dir():
        raise ValueError(f"Discovery root does not exist: {root}")
    candidates, selections = [], {}
    patterns = ["*/*/*staged*/seed_*/summary.json"]
    if include_jobs:
        patterns += ["_jobs/*/groups/*/*/run/evaluation/seed_*/summary.json",
                     "_jobs/*/groups/*/*/evaluation/seed_*/summary.json"]
    for pattern in patterns:
        for path in root.glob(pattern):
            if len(candidates) >= MAX_CANDIDATES:
                raise ValueError("Staged discovery candidate bound exceeded")
            candidates.append(path.parent)
            selections[path.parent] = {"kind": "jobs_scan" if pattern.startswith("_jobs/") else "canonical_cohort"}
    extra_roots = [Path(path).absolute() for path in extra_roots]
    for extra in extra_roots:
        if not extra.is_dir():
            raise ValueError(f"Explicit extra root does not exist: {extra}")
        info = {"kind": "explicit_extra_root", "root": str(extra), "warnings": [], "provenance": []}
        # This one known layout is metadata evidence only, never a selector.
        if extra.name == "evaluation" and extra.parent.name == "run":
            group_status = extra.parent.parent / "group_status.json"
            if group_status.is_file():
                info["group_status"] = _read(group_status, "staged_group_status", info["provenance"])
                info["warnings"].append("Explicit extra cohort: group status is retained separately from individual run completion.")
        for path in extra.glob("seed_*/summary.json"):
            if len(candidates) >= MAX_CANDIDATES:
                raise ValueError("Staged discovery candidate bound exceeded")
            candidates.append(path.parent)
            selections.setdefault(path.parent, info)
    candidates = sorted(set(candidates), key=lambda p: (
        selections[p]["kind"] != "canonical_cohort", str(p)))
    result = {"schema": "evaluation-unified-staged-discovery-v1", "root": str(root),
              "required_prediction_horizon": 50, "include_jobs": include_jobs,
              "extra_roots": [str(path) for path in extra_roots],
              "included": [], "excluded": [], "duplicates": []}
    runs, conflicts = {}, set()
    for path in candidates:
        try:
            _, _, _, identity, completion, rules, provenance = _inspect(path)
            entry = {"source": str(path), "identity": identity, "aliases": [],
                     "completion": completion, "rule_identity": rules,
                     "provenance": provenance + selections[path].get("provenance", []),
                     "selection": selections[path], "warnings": selections[path].get("warnings", [])}
            key = tuple(identity.get(k) for k in _IDENTITY_KEYS) + (rules["sha256"],)
            fingerprint = tuple(row["sha256"] for row in provenance[:2])
            if key in conflicts:
                _fail("Conflicting copies of the same run identity", "conflict")
            if key in runs:
                old, old_fingerprint = runs[key]
                if fingerprint != old_fingerprint:
                    result["included"].remove(old)
                    for source in [old["source"], *old["aliases"]]:
                        result["excluded"].append({"source": source, "status": "conflict",
                                                   "reason": "Conflicting copies of the same run identity"})
                    conflicts.add(key)
                    _fail("Conflicting copies of the same run identity", "conflict")
                old["aliases"].append(str(path))
                result["duplicates"].append({"source": str(path), "canonical_source": old["source"],
                                              "reason": "same_identity_and_summary_sidecar_sha256"})
            else:
                runs[key] = entry, fingerprint
                result["included"].append(entry)
        except (OSError, ValueError, TypeError, KeyError) as error:
            result["excluded"].append({"source": str(path), "status": getattr(error, "status", "invalid"),
                                       "reason": str(error)})
    result["counts"] = {"candidates": len(candidates), "included": len(result["included"]),
                        "excluded": len(result["excluded"]), "duplicates": len(result["duplicates"])}
    return result


def _clock(row):
    timestamp = row.get("simulation_time_s")
    measurements = row.get("measurements")
    if not isinstance(measurements, dict) or not _number(timestamp) or timestamp < 0:
        _fail("State log has an invalid simulation timestamp/measurements")
    for value in (row.get("time_s"), measurements.get("time_s"), measurements.get("simulation_time_s")):
        if not _number(value) or abs(value - timestamp) > 1e-9:
            _fail("State log observation clocks disagree")
    return timestamp


def _state_log(trial, summary, staged, provenance, retain, extra_invariant_keys=()):
    path, handle, before = _open(trial / "staged_states.jsonl", MAX_STATE_BYTES)
    digest, rows, times, last, count, total = hashlib.sha256(), [], [], None, 0, 0
    contact_sources, missing_fields = set(), set()
    invariant_keys = (*_IDENTITY_KEYS, *extra_invariant_keys,
                      "version", "stage_definitions", "thresholds",
                      "time_basis", "stage_achievement_policy", "completion_protocol")
    with handle:
        while True:
            raw = handle.readline(MAX_ROW_BYTES + 1)
            if not raw:
                break
            total += len(raw)
            if len(raw) > MAX_ROW_BYTES or total > MAX_STATE_BYTES or count >= MAX_STATE_ROWS:
                _fail("State log exceeds bounded size/row limits")
            if not raw.endswith(b"\n"):
                _fail("State log has an incomplete final record")
            digest.update(raw)
            row = _parse(raw)
            for key in invariant_keys:
                if (key == "completion_protocol" and (key in row) != (key in staged)
                        or row.get(key) != staged.get(key) or type(row.get(key)) is not type(staged.get(key))):
                    _fail(f"State log identity/recorded contract changed: {key}")
            _stage_contract(row)
            timestamp = _clock(row)
            if times and timestamp <= times[-1]:
                _fail("State log timestamps must be strictly increasing")
            tick = row.get("control_tick")
            if type(tick) is not int or tick < 0 or (last is not None and tick <= last["control_tick"]):
                _fail("State log control ticks must be strictly increasing")
            if last is not None and any(last["stages"][k] and not row["stages"][k] for k in last["stages"]):
                _fail("Recorded cumulative stage flag regressed")
            source = row["measurements"].get("contact_force_source")
            if isinstance(source, str):
                contact_sources.add(source)
            missing = row["measurements"].get("missing_measurements", [])
            if not isinstance(missing, list) or any(not isinstance(v, str) for v in missing):
                _fail("Invalid missing-measurement declaration")
            missing_fields.update(missing)
            times.append(timestamp)
            if retain:
                rows.append(row)
            count += 1
            last = row
        witness = _finish(path, handle, before, digest.hexdigest(), "staged_state_log")
    if total != before.st_size or count != staged["state_log_samples"] or last is None:
        _fail("State log sample count/size contradicts completed report")
    for key, value in last.items():
        if key != "control_tick" and (key not in staged or staged[key] != value):
            _fail(f"State log final record contradicts staged result: {key}")
    if abs(times[-1] - summary["sim_seconds"]) > 1e-8:
        _fail("State log final clock contradicts summary duration")
    provenance.append(witness)
    intervals = [b - a for a, b in zip(times, times[1:], strict=False)]
    nominal = statistics.median(intervals) if intervals else None
    evidence = {"status": "verified", "path": str(path), "sample_count": count,
                "time_start_s": times[0], "time_end_s": times[-1],
                "clock": staged["time_basis"], "sampling_declaration": staged.get("state_log_sampling"),
                "nominal_interval_s": nominal,
                "interval_gap_count": sum(dt > nominal * 1.5 for dt in intervals) if nominal else 0,
                "observer_gap_count": staged.get("observation_gap_count"),
                "contact_force_sources": sorted(contact_sources),
                "missing_measurements": sorted(missing_fields),
                "video_tactile_alignment": "not_performed",
                "physical_phase_alignment": "recorded_association_not_independently_certified"}
    return evidence, rows


def load_staged_result(
    trial_dir: Path,
    *,
    include_state_log: bool = False,
    required_prediction_horizon: int = 50,
    integrity_profile: str = "h50_deployment_integrity",
) -> dict:
    """Project three recorded evidence layers; always stream-verify the log.

    ``include_state_log`` controls retaining rows in the returned object, not
    validation.  A bad hash, identity, clock or truncated log fails closed.
    """
    trial, summary, staged, identity, completion, rules, provenance = _inspect(
        trial_dir, required_prediction_horizon, integrity_profile
    )
    extras = ("execute_steps",) if integrity_profile == _TACTILE_INTEGRITY_PROFILE else ()
    evidence, rows = _state_log(
        trial, summary, staged, provenance, include_state_log, extras
    )
    # Metadata may have been replaced during the longer streaming read.
    for witness in provenance[:-1]:
        if witness["role"] == "tactile_inventory_file":
            actual = _inventory_file(witness["path"], witness["sha256"])
        else:
            check = []
            _read(Path(witness["path"]), witness["role"], check)
            actual = check[0]
        if actual["sha256"] != witness["sha256"] or actual["size"] != witness["size"]:
            _fail(f"Source changed during import: {witness['path']}")
    fields = ("version", "stage_definitions", "stage_flags", "named_stages", "first_times",
              "named_first_times", "highest_stage", "complete_success", "continuous_metrics",
              "thresholds", "stage_achievement_policy", "time_basis", "failure_stage",
              "instantaneous_conditions", "continuous_durations_s", "transport_verified",
              "controlled_table_placement_verified", "whiteboard_complete_success_defined")
    projection = {key: staged.get(key) for key in fields}
    if "completion_protocol" in staged:
        projection["completion_protocol"] = staged["completion_protocol"]
    projection["final_facts"] = staged["measurements"]
    projection["final_time_s"] = staged.get("simulation_time_s")
    result = {"schema": "evaluation-unified-staged-result-v1", "source": str(trial),
              "identity": identity, "completion": completion,
              "native_result": {key: summary.get(key) for key in (
                  "task_success", "status", "stop_reason", "physics_valid", "diagnostic_only", "formal_metrics_valid")},
              "staged_result": projection, "data_evidence": evidence, "rule_identity": rules,
              "provenance": provenance,
              "warnings": ["Recorded native success, cumulative stages and final facts are distinct layers.",
                           "Archived definitions and source identities are preserved; current rules are not rerun.",
                           "State, contact, tactile and video clocks are not resampled or implicitly aligned."]}
    if include_state_log:
        result["state_rows"] = rows
    return result
