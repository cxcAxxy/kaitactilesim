#!/usr/bin/env python3
"""Freeze the requested CPM checkpoint, then run exactly ten independent trials.

Only native runners determine success. No retry/score selection is performed.
--start detaches this bounded supervisor; plan/status/logs remain in job-root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/nas/chenxianchi/datasets/touchguidence/recovery_state_cpm_ratio_1_1_b1024_g8_20261013_165142/best.pt')
EXPECTED_SHA = 'e5f859a5755254e55e5fec660e8d11e8b5f27b7400cc4243ad48a133951186f3'
BASE = Path('/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k')
DATA = Path('/nas/chenxianchi/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10/ratio_1_1')
MODEL_PROJECT = Path('/cpfs_infra/user/chenxianchi/yidu/openpi-main')
MODEL_PYTHON = Path('/cpfs_infra/user/chenxianchi/miniconda3/envs/touchguidance/bin/python')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular non-symlink file: {path}')
    before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'File changed during hashing: {path}')
    return value


def save(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    pending.replace(path)


def make_plan(job, concurrency=3):
    checkpoint = job / 'assets/cpm/best.pt'
    manifest = job / 'deployments/cpm_best/deployment_manifest.json'
    evaluation = job / 'groups/cpm_best/attempt_001/evaluation'
    return dict(schema='kaihand-poker-tactile-experiment-plan-v1', created_utc=stamp(),
        job_root=str(job), task='poker-draw', seeds=list(range(10)), planned_trials=10,
        initial_state_randomization=True, trial_scope='randomized_initial_state_trials',
        initial_randomization=dict(xy_jitter_mm=4.0, yaw_jitter_deg=0.5, reset_only=True),
        concurrency=concurrency, automatic_retries=0, expected_checkpoint_sha256=EXPECTED_SHA,
        source_checkpoint_path=str(SOURCE), frozen_checkpoint_path=str(checkpoint),
        groups=[dict(label='cpm_best', task='poker-draw', model_family='pi05+cpm',
            checkpoint=str(checkpoint), source_checkpoint_path=str(SOURCE),
            checkpoint_step='best', checkpoint_sha256=EXPECTED_SHA,
            prediction_horizon=30, action_start_index=0, execute_steps=5,
            deployment_manifest=str(manifest), evaluation_path=str(evaluation))],
        protocol=dict(max_sim_seconds=90.0, guard_mode='observe', action_interpolation='zoh',
            execute_steps=5, control_hz=30, record_fps=10, save_raw=True,
            staged_evaluation_version='staged_v1', diagnostic_only=True, formal_metrics_valid=False),
        attempt_selection='Exactly attempt_001; every requested seed is retained, including failures.',
        raw_metadata_note='Legacy shared Raw model_family label may read pi0.5; deployment SHA and model worker identify pi05+cpm.',
        source_script=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))))


def prepare_command(plan):
    group = plan['groups'][0]
    return [sys.executable, '-B', str(ROOT / 'scripts/workcell/prepare_poker_tactile_deployment.py'),
        '--task', 'poker-draw', '--model-family', 'pi05+cpm', '--checkpoint', group['checkpoint'],
        '--output-dir', str(Path(group['deployment_manifest']).parent),
        '--model-project', str(MODEL_PROJECT), '--model-python', str(MODEL_PYTHON),
        '--base-checkpoint', str(BASE), '--data-root', str(DATA),
        '--tokenizer-model', '/root/.cache/openpi/big_vision/paligemma_tokenizer.model',
        '--execute-steps', '5', '--guidance-scale', '1', '--guidance-steps', '3']


def trial_command(plan, seed, manifest_sha):
    group = plan['groups'][0]
    if seed not in plan['seeds']:
        raise ValueError('Unrequested seed')
    return [sys.executable, '-B', str(ROOT / 'scripts/workcell/run_poker_tactile_policy.py'),
        '--deployment-manifest', group['deployment_manifest'],
        '--expected-deployment-manifest-sha256', manifest_sha,
        '--output-dir', str(Path(group['evaluation_path']) / f'seed_{seed:04d}'),
        '--seed', str(seed), '--execute-steps', '5', '--max-sim-seconds', '90',
        '--guard-mode', 'observe', '--record', '--record-fps', '10', '--save-raw',
        '--xy-jitter-mm', '4', '--yaw-jitter-deg', '0.5',
        '--staged-evaluation-version', 'staged_v1', '--action-interpolation', 'zoh']


def trial_row(seed, output, manifest, manifest_sha, returncode, *, timed_out=False):
    path = output / f'seed_{seed:04d}/summary.json'
    try:
        trial = json.loads(path.read_text())
    except (OSError, ValueError):
        trial = {}
    identity = all(trial.get(key) == manifest[key] for key in (
        'task', 'model_family', 'prediction_horizon', 'action_start_index', 'execute_steps', 'deployment_id'))
    identity = identity and trial.get('deployment_manifest_sha256') == manifest_sha and trial.get('seed') == seed
    valid = (returncode == 0 and not timed_out and identity and trial.get('valid_trial') is True
        and trial.get('trial_validity') == 'valid' and trial.get('physics_valid') is True
        and type(trial.get('task_success')) is bool and trial.get('formal_metrics_valid') is False
        and trial.get('diagnostic_only') is True and not any(key.endswith('record_error') for key in trial))
    return dict(seed=seed, summary=str(path.relative_to(output)), returncode=returncode,
        timed_out=timed_out, valid=valid, task_success=trial.get('task_success') if valid else None,
        status=trial.get('status', 'missing_summary'), formal_metrics_valid=trial.get('formal_metrics_valid'),
        identity_matches=identity, invalid_reason=None if valid else 'supervised_trial_failed_validation')


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_job(job):
    plan = json.loads((job / 'plan.json').read_text())
    status = dict(schema='kaihand-poker-tactile-experiment-status-v1', state='freezing',
        controller_pid=os.getpid(), updated_utc=stamp(), automatic_retries=0,
        groups={'cpm_best': dict(state='pending', accepted_attempt=None,
            evaluation_path=plan['groups'][0]['evaluation_path'])}, trials={})
    running = {}
    environment = dict(os.environ, TMPDIR=str(job / 'tmp'), MPLCONFIGDIR=str(job / 'cache/mpl'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUJOCO_GL='egl',
        KAIHAND_RENDER_BACKEND='hardware')

    def update():
        status['updated_utc'] = stamp()
        save(job / 'status.json', status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        if digest(SOURCE) != plan['expected_checkpoint_sha256']:
            raise ValueError('Source checkpoint differs from the explicitly reviewed current version')
        target = Path(plan['frozen_checkpoint_path'])
        shutil.copyfile(SOURCE, target)
        if digest(target) != EXPECTED_SHA or digest(SOURCE) != EXPECTED_SHA:
            raise ValueError('Checkpoint changed during snapshot creation')
        target.chmod(0o444)
        shutil.copyfile(SOURCE.parent / 'run_config.json', target.parent / 'run_config.json')
        save(job / 'checkpoint_snapshot.json', dict(created_utc=stamp(), source=str(SOURCE),
            frozen=str(target), sha256=EXPECTED_SHA, size=target.stat().st_size,
            source_rehashed_after_copy=True, frozen_rehashed=True))
        status['state'] = status['groups']['cpm_best']['state'] = 'preparing'
        status['prepare_command'] = prepare_command(plan)
        update()
        with (job / 'logs/prepare.log').open('x') as log:
            subprocess.run(status['prepare_command'], cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
        manifest_path = Path(plan['groups'][0]['deployment_manifest'])
        manifest = json.loads(manifest_path.read_text())
        manifest_sha = digest(manifest_path)
        if manifest['checkpoint_sha256'] != EXPECTED_SHA:
            raise ValueError('Prepared deployment did not bind the frozen checkpoint')
        output = Path(plan['groups'][0]['evaluation_path'])
        output.mkdir(parents=True, exist_ok=False)
        batch = dict(schema='poker_tactile_batch_v1', task='poker-draw', model_family='pi05+cpm',
            prediction_horizon=30, action_start_index=0, execute_steps=5,
            deployment_id=manifest['deployment_id'], deployment_manifest_sha256=manifest_sha,
            complete=False, trials=[], diagnostic_only=True, formal_metrics_valid=False)
        status.update(state='running', deployment_id=manifest['deployment_id'],
            deployment_manifest_sha256=manifest_sha)
        status['groups']['cpm_best'].update(state='running', deployment_manifest=str(manifest_path))
        pending = list(plan['seeds'])
        started = time.monotonic()
        while pending or running:
            while pending and len(running) < plan['concurrency']:
                seed = pending.pop(0)
                command = trial_command(plan, seed, manifest_sha)
                log = (output / f'seed_{seed:04d}.log').open('x')
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                running[seed] = (process, log, time.monotonic())
                status['trials'][str(seed)] = dict(state='running', pid=process.pid, started_utc=stamp(), command=command)
            for seed, (process, log, since) in list(running.items()):
                timed_out = process.poll() is None and time.monotonic() - since > 3600
                if timed_out:
                    stop(process)
                code = process.poll()
                if code is None:
                    continue
                log.close()
                row = trial_row(seed, output, manifest, manifest_sha, code, timed_out=timed_out)
                batch['trials'].append(row)
                batch['trials'].sort(key=lambda item: item['seed'])
                status['trials'][str(seed)].update(state='complete' if row['valid'] else 'invalid',
                    valid=row['valid'], task_success=row['task_success'], ended_utc=stamp(), returncode=code)
                del running[seed]
                print(json.dumps(dict(event='trial_finished', **row)), flush=True)
            valid = [row for row in batch['trials'] if row['valid']]
            batch.update(wall_seconds=time.monotonic() - started, valid_trials=len(valid),
                invalid_trials=len(batch['trials']) - len(valid),
                success_rate=sum(row['task_success'] for row in valid) / len(valid) if valid else None,
                complete=len(batch['trials']) == len(plan['seeds']))
            save(output / 'batch_summary.json', batch)
            status['groups']['cpm_best'].update(observed_trials=len(batch['trials']), valid_trials=len(valid),
                invalid_trials=batch['invalid_trials'], batch_complete=batch['complete'],
                batch_summary=str(output / 'batch_summary.json'))
            update()
            if pending or running:
                time.sleep(10)
        if digest(target) != EXPECTED_SHA or digest(manifest_path) != manifest_sha:
            raise ValueError('Frozen checkpoint/deployment changed during evaluation')
        complete = batch['complete'] and batch['invalid_trials'] == 0
        status['state'] = 'complete' if complete else 'incomplete'
        status['groups']['cpm_best'].update(state=status['state'], ended_utc=stamp(),
            accepted_attempt='attempt_001' if complete else None, returncode=0 if complete else 1)
        update()
        return 0 if complete else 1
    except BaseException as error:
        for process, log, _ in running.values():
            stop(process)
            log.close()
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        update()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    job = args.job_root.absolute()
    if args.run:
        return run_job(job)
    if job.exists() or job.resolve() != job:
        parser.error('A fresh canonical job-root is required; no resume/overwrite')
    job.mkdir(parents=True)
    for name in ('logs', 'tmp', 'cache/mpl', 'assets/cpm'):
        (job / name).mkdir(parents=True)
    save(job / 'plan.json', make_plan(job, args.concurrency))
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
