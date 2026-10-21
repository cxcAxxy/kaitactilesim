#!/usr/bin/env python3
"""Run the ten-trial CPM evaluation from independent, read-only code copies.

An earlier job was interrupted because another editor changed its source tree.
Nothing from that job is counted or overwritten; only infrastructure is fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPENPI = Path('/cpfs_infra/user/chenxianchi/yidu/openpi-main')
SCOPES = {
    'simulator': (ROOT, ('src', 'scripts/workcell', 'scripts/evaluate',
                       'scripts/collect/nvidia_egl_vendor.json', 'configs')),
    'openpi-main': (OPENPI, ('src', 'packages/openpi-client/src', 'pyproject.toml')),
}


def sha256(path):
    before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
        raise ValueError(f'Source changed during hashing: {path}')
    return value


def source_inventory(root, scopes):
    result = {}
    for relative in scopes:
        entry = root / relative
        paths = sorted(entry.rglob('*')) if entry.is_dir() else [entry]
        for path in paths:
            if '__pycache__' in path.parts or path.suffix in {'.pyc', '.pyo'}:
                continue
            if path.is_symlink():
                raise ValueError(f'Code snapshots cannot follow symlinks: {path}')
            if path.is_file():
                result[str(path.relative_to(root))] = sha256(path)
    return result


def freeze_tree(source, target, scopes):
    """Copy bytes, not links, and reject any source membership/content drift."""
    if target.exists():
        raise ValueError('Refusing to replace an existing runtime snapshot')
    before = source_inventory(source, scopes)
    if not before:
        raise ValueError('Empty runtime snapshot')
    target.mkdir(parents=True)
    for relative in before:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    frozen = source_inventory(target, scopes)
    after = source_inventory(source, scopes)
    if before != frozen or before != after:
        raise ValueError('Runtime source changed during snapshot creation; no inference is allowed')
    for relative in frozen:
        (target / relative).chmod(0o444)
    return dict(source=str(source), frozen=str(target), files=frozen,
                source_rehashed_after_copy=True, frozen_rehashed=True)


def save(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    temporary.replace(path)


def run_job(job):
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    save(job / 'status.json', dict(schema='kaihand-poker-tactile-experiment-status-v1',
        state='freezing_sources', controller_pid=os.getpid(), updated_utc=stamp(),
        groups={}, trials={}, automatic_retries=0))
    try:
        snapshots = {label: freeze_tree(source, job / 'runtime' / label, scopes)
                     for label, (source, scopes) in SCOPES.items()}
        save(job / 'runtime_snapshot.json', dict(schema='kaihand-cpm-isolated-runtime-v1',
            created_utc=stamp(), controller_source=str(Path(__file__).resolve()),
            controller_sha256=sha256(Path(__file__)), snapshots=snapshots))
        simulator = job / 'runtime/simulator'
        module_path = simulator / 'scripts/evaluate/evaluate_cpm_poker_checkpoint.py'
        spec = importlib.util.spec_from_file_location('frozen_cpm_supervisor', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = simulator
        module.MODEL_PROJECT = job / 'runtime/openpi-main'
        plan = module.make_plan(job, 3)
        plan.update(isolated_runtime=True, simulator_root=str(simulator),
            model_project_root=str(module.MODEL_PROJECT),
            runtime_snapshot_sha256=sha256(job / 'runtime_snapshot.json'),
            prior_interrupted_job=str(job.with_name('cpm_poker_best_n10_20261021')),
            prior_interruption_reason='External training-source edits broke the prior manifest inventory; no prior trials count toward this ten-trial batch.',
            prior_interrupted_trials_counted=0)
        save(job / 'plan.json', plan)
        return module.run_job(job)
    except BaseException as error:
        status_path = job / 'status.json'
        status = json.loads(status_path.read_text())
        if status.get('state') == 'freezing_sources':
            status.update(state='failed', error=str(error), updated_utc=stamp())
            save(status_path, status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    job = args.job_root.absolute()
    if args.run:
        return run_job(job)
    if job.exists() or job.resolve() != job:
        parser.error('A fresh canonical job-root is required; no overwrite/resume')
    job.mkdir(parents=True)
    for name in ('logs', 'tmp', 'cache/mpl', 'assets/cpm'):
        (job / name).mkdir(parents=True)
    if not args.start:
        return run_job(job)
    with (job / 'logs/controller.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            '--job-root', str(job), '--run'], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(state='started', controller_pid=process.pid, job_root=str(job))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
