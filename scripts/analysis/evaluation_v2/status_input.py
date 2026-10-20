"""Read finished groups without changing a running simulator or its status file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def completed_status_trials(path: Path) -> tuple[list[Path], dict]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    status = json.loads(raw)
    if not isinstance(status, dict) or not isinstance(status.get("groups"), list):
        raise ValueError("Status must contain a groups list")
    trials, included, skipped = [], [], []
    for group in status["groups"]:
        if not isinstance(group, dict):
            raise ValueError("Malformed status group")
        if group.get("status") != "complete" or type(group.get("evaluation_returncode")) is not int or group["evaluation_returncode"] != 0:
            skipped.append({"task": group.get("task"), "step": group.get("step"), "status": group.get("status")})
            continue
        if not isinstance(group.get("output_dir"), str):
            raise ValueError("Completed status group lacks output_dir")
        output = Path(group["output_dir"])
        output = (path.parent / output).resolve(strict=True) if not output.is_absolute() else output.resolve(strict=True)
        if not output.is_relative_to(path.parent):
            raise ValueError("Status group output must stay inside the status directory")
        # Explicit bounded layouts used by both legacy and managed runners.
        # Never recursively discover arbitrary stale/partial nested runs.
        layouts = []
        for candidate in (output, output / "evaluation", output / "run/evaluation"):
            summaries = sorted(candidate.glob("seed_*/summary.json"))
            if summaries:
                if not candidate.resolve(strict=True).is_relative_to(output):
                    raise ValueError("Evaluation directory escapes its completed group")
                layouts.append((candidate, summaries))
        if not layouts:
            raise ValueError(f"Completed group has no seed summaries: {output}")
        if len(layouts) != 1:
            raise ValueError("Completed group contains ambiguous evaluation layouts")
        evaluation_dir, summaries = layouts[0]
        for summary in summaries:
            resolved = summary.resolve(strict=True)
            if not resolved.is_relative_to(output):
                raise ValueError("Seed summary escapes its completed group")
            trials.append(resolved.parent)
        included.append({"task": group.get("task"), "step": group.get("step"),
                         "output_dir": str(output), "evaluation_dir": str(evaluation_dir),
                         "trial_count": len(summaries)})
    if len(trials) != len(set(trials)):
        raise ValueError("Status contains duplicate trial sources")
    return trials, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                    "included_groups": included, "skipped_groups": skipped,
                    "semantics": "A read-only snapshot; only groups explicitly complete with return code 0 are selected. In-progress groups are not inspected or changed."}
