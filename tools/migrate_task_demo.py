"""Copy and checksum an archived task-demo bundle without changing its sources.

Original replay/publication bytes remain untouched. A separate relocation
manifest binds original absolute identities to local files, including every
Raw referenced by current AND historical replay metadata. This is not a
training-dataset migration and never deletes the NAS archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/nas/chenxianchi/datasets/sim/task_demo")
MANIFEST_NAME = "migration_manifest.json"
SCHEMA = "kaihand-task-demo-migration-v1"
SHA = re.compile(r"[0-9a-f]{64}")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024


def signature(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def digest(path):
    before = signature(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    if before != signature(path):
        raise ValueError(f"Source changed while hashing: {path}")
    return result.hexdigest()


def regular_file(path):
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected an explicit absolute path: {path}")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink in archived evidence path: {parent}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Not a regular evidence file: {path}")


def safe_directory_path(path):
    path = Path(path).absolute()
    if path == Path(path.anchor) or ".." in path.parts:
        raise ValueError(f"Expected a non-root, normalized directory path: {path}")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink in bundle directory path: {parent}")
    return path


def copy_bundle(source, output):
    source, output = safe_directory_path(source), safe_directory_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite a bundle: {output}")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and destination must be disjoint")
    entries = sorted(source.rglob("*"))
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"Expected a real archive directory: {source}")
    if any(path.is_symlink() for path in entries):
        raise ValueError("Symlinks in the source bundle are unsupported")
    originals = [path for path in entries if path.is_file()]
    if not originals or any(not stat.S_ISREG(path.stat().st_mode)
                            for path in entries if not path.is_dir()):
        raise ValueError("Expected a nonempty bundle of regular files")
    if (source / MANIFEST_NAME).exists():
        raise ValueError("Source already contains a migration manifest")

    # Discover all declared Raw identities BEFORE copying, including archives.
    replays, declared = [], {}
    for path in originals:
        if path.name != "replay.json":
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        if (replay.get("schema_version") != "kaihand-offline-multimodal-replay-v1"
                or replay.get("completed") is not True):
            raise ValueError(f"Unsupported or unfinished replay: {path}")
        name, sha = replay.get("source_hdf5"), replay.get("source_sha256")
        if not isinstance(name, str) or not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError(f"Replay lacks an explicit Raw/SHA binding: {path}")
        raw = Path(name)
        regular_file(raw)
        if raw in declared and declared[raw] != sha:
            raise ValueError(f"Conflicting replay declarations for {raw}")
        declared[raw] = sha
        replays.append({"path": str(path.relative_to(source)), "task": replay["task"],
                        "source_original_path": str(raw), "source_sha256": sha})
    if not replays:
        raise ValueError("No completed task-demo replay declarations found")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".task_demo-migrating-", dir=output.parent))
    inode_copies = {}
    copied, raws = [], []

    def copy_one(original, relative):
        regular_file(original)
        before = signature(original)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        inode = before[:2]
        if inode in inode_copies:
            first, sha, previous = inode_copies[inode]
            if previous != before:
                raise ValueError(f"Hard-linked source changed during migration: {original}")
            os.link(first, destination)
        else:
            result = hashlib.sha256()
            with original.open("rb") as reader, destination.open("xb") as writer:
                for chunk in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                    result.update(chunk)
                    writer.write(chunk)
            sha = result.hexdigest()
            shutil.copystat(original, destination)
            if digest(destination) != sha:
                raise ValueError(f"Destination checksum mismatch: {destination}")
            inode_copies[inode] = (destination, sha, before)
        if before != signature(original):
            raise ValueError(f"Source changed during migration: {original}")
        expected = declared.get(original)
        if expected and sha != expected:
            raise ValueError(f"Original Raw differs from replay SHA: {original}")
        row = {"path": str(relative), "original_path": str(original),
               "sha256": sha, "size": before[2]}
        print(f"copied {relative} ({before[2]} bytes)", flush=True)
        return row

    try:
        for original in originals:
            copied.append(copy_one(original, original.relative_to(source)))
        for raw, sha in sorted(declared.items()):
            if raw.is_relative_to(source):
                continue
            relative = Path("_raw_sources") / sha / raw.name
            raws.append(copy_one(raw, relative))
        by_original = {row["original_path"]: row for row in copied + raws}
        for replay in replays:
            row = by_original[replay["source_original_path"]]
            if row["sha256"] != replay["source_sha256"]:
                raise ValueError(f"Raw checksum mismatch for {replay['path']}")
            replay["source_local_path"] = row["path"]
        # Check that the archived inventory did not change during the copy.
        if [p for p in sorted(source.rglob("*")) if p.is_file()] != originals:
            raise ValueError("Archive inventory changed while migrating")
        for original in originals + list(declared):
            saved = inode_copies[signature(original)[:2]][2]
            if signature(original) != saved:
                raise ValueError(f"Archive changed before publication: {original}")
        document = {
            "schema": SCHEMA, "original_root": str(source),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_kind": "recorded_scripted_demonstrations_not_human_teleoperation",
            "source_modified": False, "physics_rerun": False,
            "files": copied, "raw_sources": raws, "replays": replays,
            "missing_current_tasks": ["bulb-screw-inplace", "vase-wipe"],
            "scope": "Original media/metadata/archive plus every replay-bound Raw; full training/conversion datasets are not copied.",
            "summary": {"original_file_count": len(copied), "external_raw_count": len(raws),
                        "replay_count": len(replays), "copied_file_count": len(copied) + len(raws),
                        "raw_count": len(declared),
                        "logical_bytes": sum(row["size"] for row in copied + raws)},
        }
        with (stage / MANIFEST_NAME).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        # mkdtemp starts at 0700; the published archive must remain as readable
        # as its source, including when migration is run by a different owner.
        stage.chmod(stat.S_IMODE(source.stat().st_mode))
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Destination appeared during copy: {output}")
        stage.rename(output)
    except BaseException:
        print(f"Migration not published; partial copy retained for inspection: {stage}", flush=True)
        raise
    return {"output": str(output), **document["summary"], "source_writes": 0}


def verify_bundle(root):
    # Import only stdlib helper code, never initialize physics or rendering.
    import sys
    sys.path.insert(0, str(PROJECT / "src"))
    from kaihand_tactile_env.shared.task_demo import resolve_demo_path

    root = safe_directory_path(root)
    manifest = root / MANIFEST_NAME
    regular_file(manifest)
    manifest_before = signature(manifest)
    if manifest_before[2] > MAX_MANIFEST_BYTES:
        raise ValueError("Migration manifest exceeds size limit")
    with manifest.open("rb") as stream:
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("Migration manifest exceeds size limit")
    # Validate every path, identity, duplicate key and size through the same
    # bounded parser used by readers, including manifests with empty tables.
    resolve_demo_path(manifest, demo_root=root)
    if signature(manifest) != manifest_before:
        raise ValueError("Migration manifest changed during verification")
    document = json.loads(payload)
    if document.get("schema") != SCHEMA:
        raise ValueError("Unsupported migration manifest schema")
    rows = document["files"] + document["raw_sources"]
    if not rows:
        raise ValueError("Migration manifest contains no copied files")
    by_original = {row["original_path"]: row for row in rows}
    expected_paths = {row["path"] for row in rows}

    def local_inventory():
        paths = list(root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError("Symlink in migrated bundle inventory")
        found = set()
        for path in paths:
            if path.is_dir() or path == manifest:
                continue
            regular_file(path)
            found.add(str(path.relative_to(root)))
        if found != expected_paths:
            raise ValueError("Bundle file inventory differs from migration manifest")

    local_inventory()
    verified_signatures = {}
    for row in rows:
        destination = root / row["path"]
        before = signature(destination)
        actual = resolve_demo_path(row["original_path"], demo_root=root,
                                   expected_sha256=row["sha256"], verify_hash=True)
        if actual != destination:
            raise ValueError(f"Relocation did not resolve to this bundle: {row['original_path']}")
        if signature(actual) != before:
            raise ValueError(f"Copied file changed during verification: {actual}")
        verified_signatures[row["path"]] = before
    expected_replays = {row["path"] for row in document["files"]
                        if Path(row["path"]).name == "replay.json"}
    replay_rows = document.get("replays")
    if not isinstance(replay_rows, list) or not expected_replays:
        raise ValueError("Migration manifest lacks its replay audit")
    audited_replays = set()
    for row in replay_rows:
        name = row.get("path") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in expected_replays or name in audited_replays:
            raise ValueError("Replay audit must name each copied replay exactly once")
        audited_replays.add(name)
        path = root / name
        replay = json.loads(path.read_text(encoding="utf-8"))
        if signature(path) != verified_signatures[name]:
            raise ValueError(f"Verified replay changed during audit: {path}")
        if (not isinstance(replay, dict)
                or replay.get("schema_version") != "kaihand-offline-multimodal-replay-v1"
                or replay.get("completed") is not True):
            raise ValueError(f"Unsupported or unfinished copied replay: {path}")
        original = replay.get("source_hdf5")
        raw_row = by_original.get(original) if isinstance(original, str) else None
        if (raw_row is None or row.get("source_original_path") != original
                or row.get("source_local_path") != raw_row["path"]
                or row.get("source_sha256") != raw_row["sha256"]
                or replay.get("source_sha256") != raw_row["sha256"]
                or row.get("task") != replay.get("task")):
            raise ValueError(f"Replay Raw is not bound to a copied manifest file: {name}")
        raw = resolve_demo_path(original, demo_root=root,
                                expected_sha256=replay["source_sha256"])
        if raw != root / raw_row["path"]:
            raise ValueError(f"Replay relocation differs from copy audit: {name}")
    if audited_replays != expected_replays:
        raise ValueError("Replay audit omitted a copied replay")
    if signature(manifest) != manifest_before:
        raise ValueError("Migration manifest changed during verification")
    local_inventory()
    for relative, before in verified_signatures.items():
        if signature(root / relative) != before:
            raise ValueError(f"Copied file changed before verification finished: {relative}")
    return {"output": str(root), "verified_files": len(rows),
            "verified_replays": len(document["replays"]), "source_access_required": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    copy = commands.add_parser("copy", help="Copy to a fresh directory; never delete the source")
    copy.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    copy.add_argument("--output", type=Path, default=PROJECT / "task_demo")
    verify = commands.add_parser("verify", help="Fully SHA-check local payloads without reading NAS")
    verify.add_argument("root", type=Path, nargs="?", default=PROJECT / "task_demo")
    args = parser.parse_args(argv)
    result = copy_bundle(args.source, args.output) if args.command == "copy" else verify_bundle(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
