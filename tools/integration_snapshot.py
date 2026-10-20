"""Copy two read-only working trees into this independent integration tree.

Dependencies, Git metadata, runs and generated reports are deliberately not
copied. The manifest records actual working-tree bytes, not just Git HEAD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT.parent / "kaitactilesim"
ANALYSIS = ROOT.parent / "kaitactilesim_laest"
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".pixi"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(source, relative):
    path = source / relative
    if path.is_symlink() or path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        part = child.relative_to(path)
        if any(name in SKIP for name in part.parts) or child.suffix in {".pyc", ".pyo"}:
            continue
        if child.is_file() or child.is_symlink():
            yield child


def snapshot():
    if ROOT in (SIMULATION, ANALYSIS) or (ROOT / "integration_manifest.json").exists():
        raise ValueError("Use a fresh independent integration tree")
    groups = [(SIMULATION, name) for name in (
        "src", "scripts", "tests", "docs", "configs", ".gitignore", "README.md",
        "LICENSE", "pyproject.toml", "pixi.toml", "pixi.lock",
    )]
    groups += [(ANALYSIS, "scripts/analysis"), (ANALYSIS, "tests")]
    rows = []
    for source, relative in groups:
        for original in candidates(source, relative):
            destination = ROOT / original.relative_to(source)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Integration collision: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            row = {"source": str(original), "destination": str(destination.relative_to(ROOT))}
            if original.is_symlink():
                target = str(original.readlink())
                destination.symlink_to(target)
                row.update(kind="symlink", target=target)
            else:
                before = digest(original)
                shutil.copy2(original, destination)
                if digest(destination) != before or digest(original) != before:
                    raise ValueError(f"Source changed while copying: {original}")
                row.update(kind="file", sha256=before, size=original.stat().st_size)
            rows.append(row)
    for row in rows:
        original = Path(row["source"])
        if row["kind"] == "file" and digest(original) != row["sha256"]:
            raise ValueError(f"Source changed during integration snapshot: {original}")
        if row["kind"] == "symlink" and str(original.readlink()) != row["target"]:
            raise ValueError("Source link changed during integration snapshot")
    heads = {str(source): subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(source), "rev-parse", "HEAD"], text=True,
    ).strip() for source in (SIMULATION, ANALYSIS)}
    document = {"schema_version": "unified-working-tree-snapshot-v1", "source_heads": heads,
                "source_writes": 0, "copied_file_count": len(rows), "files": rows,
                "excluded": ["Git metadata", "environments", "caches", "runs", "reports", "external datasets"]}
    (ROOT / "integration_manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"copied_file_count": len(rows), "output": str(ROOT), "source_writes": 0}))


if __name__ == "__main__":
    snapshot()
