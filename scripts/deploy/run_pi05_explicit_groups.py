#!/usr/bin/env python3
"""Run four explicit frozen H50/K50 groups, then publish canonical archives.

--start creates a fresh CPFS job and detaches its controller. Each checkpoint
owns one model service and runs seeds 0..19 serially. Only processes created by
this job can be stopped. Global archive catalogs and earlier jobs are untouched.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ARCHIVE_ROOT = Path('/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations')
SNAPSHOT_ROOT = Path('/cpfs_infra/user/chenxianchi/runtime_snapshots')
JOBS_ROOT = ARCHIVE_ROOT / '_jobs'
MODEL_PYTHON = Path('/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python')
CONTROLLER_PYTHON = Path('/usr/local/bin/python3.11')
TOKENIZER_SOURCE = Path('/nas/chenxianchi/datasets/sim/_evaluations/pi05_h50_k50_serial_20261013_111821_frozen_r2/.cache/openpi/big_vision/paligemma_tokenizer.model')
TASKS = ('whiteboard-wipe-horizontal-line', 'bulb-screw')
SETTINGS = dict(prediction_horizon=50, execute_steps=50, num_trials=20,
                video_count=20, seed_start=0, max_sim_seconds=90,
                guard_mode='observe', server_timeout=900, gpu_device=0,
                parallel_groups=4)


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(path):
    path = Path(path).absolute()
    require('..' not in path.parts, 'Parent traversal is forbidden')
    for item in (path, *path.parents):
        require(not item.is_symlink(), f'Unexpected path symlink: {item}')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f'Not a regular archive file: {path}')
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f'Archive file changed during hash: {path}')
    return digest.hexdigest()


def process_info(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rindex(')') + 2:].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), starttime=int(fields[19]))


def port_free(port):
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False


def validate_plan(plan, *, existing_job=False, allow_published=False):
    require(isinstance(plan, dict), 'Plan must be a JSON object')
    job = safe(plan['job_root'])
    require(job.parent == JOBS_ROOT and job.name.startswith('pi05_line_bulb_'),
            'Job must be a new explicitly scoped CPFS _jobs/pi05_line_bulb_* directory')
    require(job.is_dir() if existing_job else not job.exists(), 'Job root existence/collision mismatch')
    simulation, openpi = map(safe, (plan['simulation_root'], plan['openpi_root']))
    require(simulation.name == 'kaitactilesim' and openpi.name == 'openpi'
            and simulation.parent == openpi.parent and simulation.parent.parent == SNAPSHOT_ROOT,
            'Simulation/OpenPI must be sibling runtime_snapshot copies')
    require(plan.get('settings') == SETTINGS, 'Evaluation settings must match authorized H50/K50/20/90/observe plan')
    groups = plan.get('groups', [])
    require(len(groups) == 4 and {(g['task'], g['step']) for g in groups}
            == {(task, step) for task in TASKS for step in (20000, 24999)},
            'Plan must contain exactly the two selected tasks at both checkpoints')
    require([g['port'] for g in groups] == list(range(18810, 18814)), 'Group ports must be 18810..18813')
    for group in groups:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', group['dataset']) is not None,
                'Unsafe dataset label')
        checkpoint = safe(group['checkpoint'])
        parts = checkpoint.parts
        require('pi05' in parts, 'Checkpoint is not a pi05 dataset checkpoint')
        offset = parts.index('pi05')
        require(parts[offset - 1] == group['task'] and parts[offset + 1] == group['dataset']
                and parts[offset + 2] == 'checkpoints' and checkpoint.name == str(group['step']),
                'Checkpoint task/dataset/step binding mismatch')
        archive = safe(group['archive_dir'])
        require(archive == ARCHIVE_ROOT / group['task'] / 'pi05'
                / f"step{group['step']}_H50K50_{group['dataset']}", 'Wrong canonical archive path')
        if not allow_published:
            require(not archive.exists(), f'Archive group already exists: {archive}')
    return plan


def verify_frozen(plan):
    simulation, openpi = Path(plan['simulation_root']), Path(plan['openpi_root'])
    for source in (simulation / 'src', simulation / 'scripts', simulation / 'configs',
                   openpi / 'src', openpi / 'scripts', openpi / 'packages'):
        require(source.is_dir() and not source.is_symlink(), f'Missing frozen sources: {source}')
        for current, dirs, names in os.walk(source, followlinks=False):
            for item in [Path(current), *(Path(current) / name for name in dirs + names)]:
                require(not item.is_symlink() and not item.stat().st_mode & 0o222,
                        f'Sources are not immutable physical copies: {item}')
    for interpreter in (MODEL_PYTHON, simulation / '.pixi/envs/default/bin/python', CONTROLLER_PYTHON):
        require(interpreter.is_file() and os.access(interpreter, os.X_OK), f'Missing interpreter: {interpreter}')
    for group in plan['groups']:
        for name in ('_CHECKPOINT_METADATA', 'params/_METADATA'):
            require((Path(group['checkpoint']) / name).is_file(), 'Uncommitted checkpoint metadata')
    report = read(simulation.parent / 'preflight_report.json')
    rows = report.get('checkpoints', [])
    require((report.get('all_four_valid') is True or report.get('all_valid') is True)
            and report.get('count', len(rows)) == 4
            and report.get('simulation_root') == str(simulation)
            and report.get('model_project_root') == str(openpi)
            and [(r.get('task'), r.get('step'), r.get('checkpoint')) for r in rows]
            == [(g['task'], g['step'], g['checkpoint']) for g in plan['groups']]
            and all(r.get('valid') is True and r.get('prediction_horizon') == r.get('execute_steps') == 50
                    and r.get('binding_status') == 'bound' and r.get('required_action_interpolation') == 'linear'
                    for r in rows), 'All four exact frozen checkpoints must pass CPU provenance preflight')


def environment(plan):
    job, simulation, openpi = map(Path, (plan['job_root'], plan['simulation_root'], plan['openpi_root']))
    result = dict(os.environ)
    result.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join((str(simulation / 'src'), str(openpi / 'src'),
                                   str(openpi / 'packages/openpi-client/src'))),
        CUDA_VISIBLE_DEVICES='0', MUJOCO_EGL_DEVICE_ID='0', MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl', KAIHAND_RENDER_BACKEND='hardware', LIBGL_ALWAYS_SOFTWARE='0',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LP_NUM_THREADS='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENPI_DATA_HOME=str(job / '.cache/openpi'), HF_HOME=str(job / '.cache/huggingface'),
        MPLCONFIGDIR=str(job / '.cache/matplotlib'),
        __EGL_VENDOR_LIBRARY_FILENAMES='/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json')
    result.pop('JAX_PLATFORMS', None)
    result.pop('JAX_PLATFORM_NAME', None)
    return result


def output_dir(plan, index):
    group = plan['groups'][index]
    return Path(plan['job_root']) / 'groups' / group['task'] / f"step{group['step']}_N20_video20"


def start(plan):
    validate_plan(plan)
    verify_frozen(plan)
    require(TOKENIZER_SOURCE.is_file(), 'Cached PaliGemma tokenizer is missing')
    require(all(port_free(g['port']) for g in plan['groups']), 'A selected local service port is occupied')
    job = Path(plan['job_root'])
    job.mkdir(parents=True, exist_ok=False)
    cache = job / '.cache/openpi/big_vision'
    cache.mkdir(parents=True)
    shutil.copy2(TOKENIZER_SOURCE, cache / TOKENIZER_SOURCE.name)
    write(job / 'plan.json', plan)
    write(job / 'status.json', dict(status='starting', started_utc=now(), total_trials=80))
    for name in ('preflight_report.json', 'snapshot_record.json'):
        source = Path(plan['simulation_root']).parent / name
        if source.is_file():
            shutil.copy2(source, job / name)
    script = Path(plan['simulation_root']) / 'scripts/deploy' / Path(__file__).name
    with (job / 'controller.log').open('xb', buffering=0) as log:
        process = subprocess.Popen([str(CONTROLLER_PYTHON), str(script), '--plan', str(job / 'plan.json')],
            cwd=plan['simulation_root'], env=environment(plan), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(job / 'controller_identity.json', process_info(process.pid) or {'pid': process.pid})
    print(json.dumps(dict(pid=process.pid, job_root=str(job), status=str(job / 'status.json'),
                          log=str(job / 'controller.log')), indent=2), flush=True)
    return process.pid


def load_api(plan):
    simulation, openpi = Path(plan['simulation_root']), Path(plan['openpi_root'])
    sys.path[:0] = [str(simulation / 'src'), str(openpi / 'src'), str(openpi / 'packages/openpi-client/src')]
    os.environ.update(environment(plan))
    from kaihand_tactile_env.pipeline import deployment, runtime
    require(Path(deployment.__file__).resolve().is_relative_to(simulation)
            and Path(runtime.__file__).resolve().is_relative_to(simulation), 'Imported mutable simulation API')
    profile = runtime.load_runtime_profile(simulation / 'configs/runtimes.example.toml', 'pi05')
    profile = dataclasses.replace(profile, python=MODEL_PYTHON, project_root=openpi,
                                  env={**profile.env, **{key: environment(plan)[key] for key in
                                       ('CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}})
    profile.validate(require_python=True, require_project=True)
    return SimpleNamespace(deployment=deployment, profile=profile)


def validate_native(evaluation, group, deployment):
    summary = read(evaluation / 'summary.json')
    rows = summary.get('results', summary.get('trials', []))
    require(summary.get('task') == group['task'] and summary.get('prediction_horizon') == summary.get('execute_steps') == 50
            and summary.get('planned') == summary.get('attempted') == summary.get('valid_trials') == 20
            and summary.get('deployment_id') == deployment['deployment_id']
            and summary.get('checkpoint_sha256') == deployment['checkpoint_sha256']
            and summary.get('diagnostic_only') is True and summary.get('formal_metrics_valid') is False
            and len(rows) == 20 and {r['seed'] for r in rows} == set(range(20))
            and all(r.get('valid_trial') is True for r in rows), 'Evaluation did not finish exactly 20 valid H50/K50 trials')
    for row in rows:
        seed = row['seed']
        require(row.get('summary') == f'seed_{seed:03d}/summary.json', 'Unsafe native seed summary path')
        detail = read(evaluation / row['summary'])
        require(detail.get('task') == group['task'] and detail.get('seed') == seed
                and detail.get('checkpoint_path') == group['checkpoint']
                and detail.get('checkpoint_sha256') == deployment['checkpoint_sha256']
                and detail.get('deployment_id') == deployment['deployment_id']
                and detail.get('prediction_horizon') == detail.get('execute_steps') == 50
                and detail.get('deployment_integrity', {}).get('status') == 'verified'
                and not detail.get('deployment_integrity', {}).get('errors'), 'Seed deployment/identity verification failed')
        require((evaluation / f'seed_{seed:03d}/review.mp4').is_file()
                or (evaluation / f'seed_{seed:03d}/review/review.mp4').is_file(), 'Missing trial video')
    return summary


def rename_noreplace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, 'renameat2', None)
    error = errno.ENOSYS
    if operation is not None:
        operation.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        operation.restype = ctypes.c_int
        if operation(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
            return
        error = ctypes.get_errno()
    if error not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
        raise OSError(error, os.strerror(error), str(destination))

    # BMCPFS supports ordinary directory rename, but not RENAME_NOREPLACE.
    # Claim the exact unused name atomically; an existing directory (including
    # an empty one), file or symlink is never eligible for this fallback.
    source, destination = safe(source), Path(destination).absolute()
    parent = safe(destination.parent)
    source_info = source.lstat()
    require(stat.S_ISDIR(source_info.st_mode) and stat.S_ISDIR(parent.lstat().st_mode),
            'Fallback publication requires ordinary source/parent directories')
    require(source_info.st_dev == parent.stat().st_dev, 'Fallback publication must be same-filesystem')
    os.mkdir(destination, 0o700)
    claim = destination.lstat()
    claim_identity = (claim.st_dev, claim.st_ino)
    claim_fd = None
    try:
        claim_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(claim_fd)
        current = destination.lstat()
        require(stat.S_ISDIR(current.st_mode)
                and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino) == claim_identity,
                'Fallback destination no longer matches this publication claim')
        require(not any(destination.iterdir()), 'Fallback destination claim acquired external data')
        # POSIX rename cannot replace a nonempty destination directory. If
        # another writer adds a file after the emptiness check, rename fails
        # without removing either that file or the source evaluation.
        os.rename(source, destination)
    except BaseException:
        try:
            current = destination.lstat()
            if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == claim_identity:
                # rmdir is deliberately nonrecursive and refuses nonempty
                # claims. Never remove a replacement inode or another file.
                os.rmdir(destination)
        except (FileNotFoundError, OSError):
            pass
        raise
    finally:
        if claim_fd is not None:
            os.close(claim_fd)


def publish_archive(plan, index, deployment):
    group = plan['groups'][index]
    source = safe(output_dir(plan, index) / 'run/evaluation')
    destination = safe(group['archive_dir'])
    require(not destination.exists(), 'Canonical archive already exists; refusing overwrite')
    summary = validate_native(source, group, deployment)
    files = []
    for current, dirs, names in os.walk(source, followlinks=False):
        for name in dirs:
            require(not (Path(current) / name).is_symlink(), 'Symlink directory in evaluation')
        for name in names:
            path = Path(current) / name
            files.append(dict(relative_path=str(path.relative_to(source)), size=path.stat().st_size, sha256=sha(path)))
    manifest = dict(schema='kaihand-explicit-evaluation-archive-v1', task=group['task'], model='pi05',
        step=group['step'], dataset=group['dataset'], checkpoint=group['checkpoint'],
        checkpoint_sha256=deployment['checkpoint_sha256'], deployment_id=deployment['deployment_id'],
        model_config_name=deployment.get('model_config_name'), training_provenance=deployment.get('training_provenance'),
        prediction_horizon=50, execute_steps=50, valid_trials=20, diagnostic_only=True, formal_metrics_valid=False,
        job_root=plan['job_root'], original_evaluation_dir=str(source), archive_dir=str(destination),
        source_simulation_root=plan['simulation_root'], source_openpi_root=plan['openpi_root'],
        original_metadata_unchanged=True, summary_sha256=sha(source / 'summary.json'), files=files,
        created_utc=now(), publication='atomic_no_overwrite_same_filesystem_rename_with_compatibility_link')
    require(not (source / 'archive_manifest.json').exists(), 'Evaluation already contains an archive manifest')
    write(source / 'archive_manifest.json', manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(source.stat().st_dev == destination.parent.stat().st_dev, 'Archive publication must be same-filesystem')
    rename_noreplace(source, destination)
    try:
        os.symlink(str(destination), source, target_is_directory=True)
        require(source.resolve(strict=True) == destination, 'Evaluation compatibility symlink failed')
    except BaseException:
        if source.is_symlink() and os.readlink(source) == str(destination):
            source.unlink()
        if not source.exists() and not source.is_symlink():
            rename_noreplace(destination, source)
        raise
    return dict(archive_dir=str(destination), archive_manifest=str(destination / 'archive_manifest.json'),
                completed_trials=20, valid_trials=20, diagnostic_successes=summary.get('successes', 0))


def run_worker(plan, index):
    validate_plan(plan, existing_job=True)
    verify_frozen(plan)
    group = plan['groups'][index]
    output = output_dir(plan, index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    status = dict(status='running', stage='preflight', pid=os.getpid(), identity=process_info(os.getpid()),
        task=group['task'], checkpoint_step=group['step'], checkpoint=group['checkpoint'],
        output_dir=str(output), started_utc=now(), errors=[])
    def update(**values):
        status.update(values, updated_utc=now())
        write(output / 'group_status.json', status)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    record = None
    update()
    try:
        api = load_api(plan)
        module, profile = api.deployment, api.profile
        deployment_dir = output / 'deployment'
        manifest_path = deployment_dir / 'deployment_manifest.json'
        update(stage='prepare')
        prepare = module.build_prepare_plan(group['task'], 'pi05', checkpoint=Path(group['checkpoint']),
                                            output_dir=deployment_dir, profile=profile)
        require(module.execute_prepare(prepare) == 0, 'Deployment prepare failed')
        update(stage='verify')
        verification = module.verify_deployment(manifest_path, profile=profile)
        module.write_pipeline_deployment(deployment_dir, task=group['task'], model_family='pi05',
            prepare_plan=prepare, profile=profile, verification=verification)
        deployment = read(manifest_path)
        require(deployment.get('task') == group['task'] and deployment.get('checkpoint_path') == group['checkpoint']
                and deployment.get('prediction_horizon') == 50, 'Prepared checkpoint/task/H50 identity mismatch')
        update(stage='evaluate', deployment_id=deployment['deployment_id'],
               run_manifest=str(output / 'run/run_manifest.json'))
        record = module.run_deployment(manifest_path, output_dir=output / 'run', profile=profile,
            service_mode='managed', endpoint=f"ws://127.0.0.1:{group['port']}", num_trials=20,
            video_count=20, seed_start=0, execute_steps=50, max_sim_seconds=90, guard_mode='observe', server_timeout=900)
        update(evaluation_returncode=record.get('evaluation_returncode'),
               managed_server_stopped=record.get('managed_server_stopped'),
               evaluation_process_stopped=record.get('evaluation_process_stopped'))
        require(record.get('evaluation_returncode') == 0
                and record.get('managed_server_stopped') is True and record.get('evaluation_process_stopped') is True
                and port_free(group['port']), 'Owned evaluator/service cleanup failed')
        update(stage='publishing')
        published = publish_archive(plan, index, deployment)
        update(status='complete', stage='complete', finished_utc=now(), owned_cleanup_confirmed=True, **published)
        return 0
    except BaseException as error:
        manifest = output / 'run/run_manifest.json'
        if manifest.is_file():
            record = read(manifest)
        clean = (record is None and status['stage'] in ('preflight', 'prepare', 'verify')) or (
            record is not None and record.get('managed_server_stopped') is True
            and record.get('evaluation_process_stopped') is True)
        status['errors'].append(f'{type(error).__name__}: {error}')
        update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               owned_cleanup_confirmed=bool(clean and port_free(group['port'])), finished_utc=now())
        return 1
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def launch_worker(plan, index):
    job, simulation = Path(plan['job_root']), Path(plan['simulation_root'])
    with (job / f'group_{index}.log').open('xb', buffering=0) as log:
        return subprocess.Popen([str(simulation / '.pixi/envs/default/bin/python'),
            str(simulation / 'scripts/deploy' / Path(__file__).name), '--plan', str(job / 'plan.json'),
            '--worker-index', str(index)], cwd=simulation, env=environment(plan),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def refresh(plan, status):
    for index, group in enumerate(status['groups']):
        output = output_dir(plan, index)
        worker_status = output / 'group_status.json'
        if worker_status.is_file():
            detail = read(worker_status)
            group.update(worker_status=detail, stage=detail.get('stage'))
            if group.get('status') not in ('complete', 'failed'):
                group['status'] = detail.get('status', group['status'])
        run_manifest = output / 'run/run_manifest.json'
        if run_manifest.is_file():
            record = read(run_manifest)
            group['owned_run_pids'] = {key: process_info(record[key]) for key in ('server_pid', 'evaluation_pid')
                                      if isinstance(record.get(key), int)}
        summary = output / 'run/evaluation/summary.json'
        if summary.is_file():
            native = read(summary)
            rows = native.get('results', native.get('trials', []))
            group.update(completed_trials=len(rows), valid_trials=native.get('valid_trials', 0),
                         completed_seeds=[r['seed'] for r in rows if r.get('valid_trial') is True],
                         diagnostic_successes=native.get('successes', 0))
    status.update(updated_utc=now(), completed_trials=sum(g.get('completed_trials', 0) for g in status['groups']),
                  valid_trials=sum(g.get('valid_trials', 0) for g in status['groups']),
                  completed_checkpoint_count=sum(g.get('status') == 'complete' for g in status['groups']))
    write(Path(plan['job_root']) / 'status.json', status)


def run_controller(plan):
    validate_plan(plan, existing_job=True)
    verify_frozen(plan)
    job = Path(plan['job_root'])
    status = dict(status='running', stage='parallel_evaluations', pid=os.getpid(), identity=process_info(os.getpid()),
        started_utc=now(), total_trials=80, total_checkpoint_count=4, settings=plan['settings'], errors=[],
        groups=[{**group, 'group_index': index, 'status': 'queued', 'output_dir': str(output_dir(plan, index))}
                for index, group in enumerate(plan['groups'])])
    workers = {}
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (job / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for index, group in enumerate(status['groups']):
                try:
                    require(port_free(group['port']), 'Selected service port became occupied')
                    process = launch_worker(plan, index)
                    workers[index] = process
                    group.update(status='running', worker_pid=process.pid, worker_identity=process_info(process.pid))
                except Exception as error:
                    group.update(status='failed', stage='launch_failed', error=str(error))
                    status['errors'].append(f"{group['task']} step{group['step']}: launch failed: {error}")
            while workers:
                refresh(plan, status)
                for index, process in list(workers.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    group = status['groups'][index]
                    # The worker can publish its final status between the last
                    # refresh and poll(). Re-read after observing process exit.
                    status_path = output_dir(plan, index) / 'group_status.json'
                    detail = read(status_path) if status_path.is_file() else {}
                    group.update(worker_status=detail, stage=detail.get('stage'))
                    success = code == 0 and detail.get('status') == 'complete' and detail.get('owned_cleanup_confirmed') is True
                    group.update(status='complete' if success else 'failed', worker_returncode=code, finished_utc=now())
                    if not success:
                        status['errors'].append(f"{group['task']} step{group['step']}: worker exit {code}")
                    del workers[index]
                if workers:
                    time.sleep(10)
            status.update(status='failed' if status['errors'] else 'complete', stage='complete', finished_utc=now())
            refresh(plan, status)
            return int(bool(status['errors']))
    except BaseException as error:
        # Only direct Popen objects created by this controller are in this set.
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        for process in workers.values():
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                status['errors'].append(f'Owned worker {process.pid} did not finish cleanup')
        status['errors'].append(f'{type(error).__name__}: {error}')
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      stage='controller_error', finished_utc=now())
        refresh(plan, status)
        return 1
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--worker-index', type=int, choices=range(4))
    args = parser.parse_args()
    require(not (args.start and args.worker_index is not None), '--start and --worker-index are exclusive')
    plan = read(args.plan)
    if args.start:
        start(plan)
        return 0
    return run_worker(plan, args.worker_index) if args.worker_index is not None else run_controller(plan)


if __name__ == '__main__':
    raise SystemExit(main())
