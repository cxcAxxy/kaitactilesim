"""Transactional finalization of metadata for a newly exported staged suite."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from evaluation_workbench.export import _digest, _document, _verify, verify_rollback
from evaluation_workbench.suite import SCHEMA, _portal, implementation


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def publish_unified_metadata(output, groups, original_guard):
    output = Path(output).resolve(strict=True)
    batch = _document(output / "batch.json")
    if batch.get("schema_version") != SCHEMA or batch.get("evaluation_profile") != "recorded-staged":
        raise ValueError("Only a freshly exported staged suite may be finalized")
    batch["stage_groups"] = groups
    batch["source_preservation"] = original_guard
    verification = _document(output / "verification.json")
    verification["unified_import"] = {"profile": "recorded-staged", "trial_count": batch["trial_count"],
        "task_count": batch["task_count"], "rule_group_count": len(groups), "current_rules_rescored": False,
        "native_outcome_preserved": True, "original_projects": original_guard,
        "all_records_diagnostic": all(t["mode"] == "diagnostic" for t in batch["trials"])}
    verification["implementation"] = implementation()
    manifest = _document(output / "artifact_manifest.json")
    rows = {row["path"]: row for row in manifest["files"]}
    replacements = {output / "batch.json": _encoded(batch), output / "verification.json": _encoded(verification),
                    output / "index.html": _portal(batch).encode()}
    before = {}
    for path, raw in replacements.items():
        _verify([rows[str(path)]])
        before[path] = path.read_bytes()
        import hashlib
        rows[str(path)] = {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest_path = output / "artifact_manifest.json"
    before[manifest_path] = manifest_path.read_bytes()
    original_manifest_digest = _digest(manifest_path)
    replacements[manifest_path] = _encoded({**manifest, "files": list(rows.values())})
    lock = output / ".complete-model-plots.lock"
    with lock.open("xb"):
        pass
    committed = []
    try:
        with tempfile.TemporaryDirectory(prefix=".unified-finalize-", dir=output) as directory:
            stage = Path(directory)
            for i, (path, raw) in enumerate(replacements.items()):
                (stage / f"next-{i}").write_bytes(raw)
                (stage / f"before-{i}").write_bytes(before[path])
            _verify([original_manifest_digest])
            verify_rollback(output / "rollback.json")
            _verify(verification["implementation"])
            try:
                for i, path in enumerate(replacements):
                    os.replace(stage / f"next-{i}", path)
                    committed.append((i, path))
                for path in replacements:
                    if path != manifest_path:
                        _verify([rows[str(path)]])
                verify_rollback(output / "rollback.json")
            except BaseException:
                for i, path in reversed(committed):
                    os.replace(stage / f"before-{i}", path)
                raise
    finally:
        lock.unlink()
