#!/usr/bin/env python3
"""Opt-in, mutable six-slot scheduler for an already running frozen staged queue.

The original plan, frozen runtime, protocols and active workers are not changed.
An armed successor is ready before SIGSTOP targets only the identified original
controller. A two-phase activation prevents it from launching before handoff is
committed. No automatic resume is permitted after the launch barrier opens.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
CONTROLLER_PYTHON = Path('/usr/local/bin/python3.11')
CAP = 6
EXTRA_TASK = 'usb-insert'


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def ready_read(path):
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f'Expected physical file: {path}')
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed while hashing')
    return digest


def proc(pid):
    require(type(pid) is int and pid > 1, 'Invalid process PID')
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode() for part in (root / 'cmdline').read_bytes().split(b'\0') if part]
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), starttime=int(fields[19]), argv=argv)
    except FileNotFoundError:
        return None


def identity_matches(expected, current):
    return isinstance(expected, dict) and current is not None and type(expected.get('pid')) is int \
        and type(expected.get('starttime')) is int and expected['pid'] == current['pid'] \
        and expected['starttime'] == current['starttime']


def exact_process(expected, *, states=None):
    require(isinstance(expected, dict), 'Process identity must be a JSON object')
    current = proc(expected.get('pid'))
    require(identity_matches(expected, current), 'Process identity missing or PID reused')
    if 'argv' in expected and current['state'] != 'Z':
        require(current['argv'] == expected['argv'], 'Process command line changed')
    require(states is None or current['state'] in states, 'Unexpected process state')
    return current


def signal_process(expected, signum, *, states=None):
    """Pin the actual process, never a later occupant of the same PID."""
    exact_process(expected, states=states)
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        descriptor = os.pidfd_open(expected['pid'], 0)
        try:
            exact_process(expected, states=states)
            signal.pidfd_send_signal(descriptor, signum, None, 0)
        finally:
            os.close(descriptor)
    else:
        exact_process(expected, states=states)
        os.kill(expected['pid'], signum)


def load_context(plan_path):
    plan_path = Path(plan_path).absolute()
    require(not plan_path.is_symlink(), 'Original plan must not be linked')
    plan = read(plan_path)
    require(plan_path == Path(plan['job_root']) / 'plan.json', 'Use the original saved job plan')
    frozen = Path(plan['simulation_root']) / 'scripts/deploy'
    sys.path.insert(0, str(frozen))
    specification = importlib.util.spec_from_file_location('handoff_frozen_staged', frozen / 'run_pi05_staged_groups.py')
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    require(Path(driver.base.__file__).resolve().parent == frozen, 'Imported non-frozen helper')
    driver.validate_plan(plan, existing_job=True, allow_published=True)
    driver.verify_frozen(plan)
    require(plan['settings']['parallel_groups'] == 4, 'This explicit handoff is only four to six')
    status = read(Path(plan['job_root']) / 'status.json')
    require(status.get('status') == 'running' and status.get('run_id') == plan['run_id']
            and status.get('staged_evaluation_version') == 'staged_v1', 'Original staged queue is not running')
    owner = exact_process(status.get('identity', {}))
    require(owner['state'] not in ('T', 't', 'Z', 'X'), 'Original controller is already stopped or exited')
    expected_argv = [str(CONTROLLER_PYTHON), str(frozen / 'run_pi05_staged_groups.py'), '--plan', str(plan_path)]
    require(owner['pid'] == status.get('pid') and owner['argv'] == expected_argv, 'Original controller command identity mismatch')
    with (Path(plan['job_root']) / 'controller.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise ValueError('Original controller does not hold its ownership lock')
    indices = [i for i, g in enumerate(plan['groups']) if g['task'] == EXTRA_TASK]
    require(len(indices) == 2 and {plan['groups'][i]['step'] for i in indices} == {20000, 24999}, 'Authorized USB checkpoint pair missing')
    for index in indices:
        require(status['groups'][index]['status'] in ('queued', 'running', 'complete'), 'Authorized USB group failed before handoff')
        if status['groups'][index]['status'] == 'queued':
            guard_fresh_group(plan, driver, index)
    proof = dict(plan_sha256=sha(plan_path), frozen_driver_sha256=sha(frozen / 'run_pi05_staged_groups.py'),
                 frozen_helper_sha256=sha(frozen / 'run_pi05_explicit_groups.py'),
                 preflight_sha256=sha(Path(plan['simulation_root']).parent / 'preflight_report.json'))
    return plan, driver, owner, status, proof


def guard_fresh_group(plan, driver, index):
    job = Path(plan['job_root'])
    group = plan['groups'][index]
    for path in (driver.output_dir(plan, index), job / f'group_{index}.log', Path(group['archive_dir'])):
        require(not path.exists() and not path.is_symlink(), f'Group already claimed/started/published: {path}')
    require(driver.port_free(group['port']), 'Selected queued group port is occupied')


def worker_index(plan, info):
    prefix = [str(Path(plan['simulation_root']) / '.pixi/envs/default/bin/python'),
              str(Path(plan['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
              '--plan', str(Path(plan['job_root']) / 'plan.json'), '--worker-index']
    require(info['argv'][:5] == prefix and len(info['argv']) == 6, 'Unknown child process of original controller')
    index = int(info['argv'][5])
    require(0 <= index < len(plan['groups']), 'Original child worker index invalid')
    return index


def paused_snapshot(plan, driver, owner):
    exact_process(owner, states={'T', 't'})
    status = read(Path(plan['job_root']) / 'status.json')
    require(identity_matches(owner, status.get('identity')), 'Paused controller status identity mismatch')
    require([(g['task'], g['step']) for g in status['groups']]
            == [(g['task'], g['step']) for g in plan['groups']], 'Paused queue identities differ from original plan')
    children = (Path('/proc') / str(owner['pid']) / 'task' / str(owner['pid']) / 'children').read_text().split()
    adopted = {}
    for child in children:
        info = proc(int(child))
        require(info is not None and info['ppid'] == owner['pid'], 'Original child ownership changed during snapshot')
        if info['state'] == 'Z':
            matches = [i for i, row in enumerate(status['groups']) if row.get('worker_pid') == info['pid']
                       and identity_matches(row.get('worker_identity'), info)]
            if not matches:
                for i, group in enumerate(plan['groups']):
                    detail = ready_read(driver.output_dir(plan, i) / 'group_status.json')
                    if detail and detail.get('task') == group['task'] and detail.get('checkpoint_step') == group['step'] \
                            and identity_matches(detail.get('identity'), info):
                        matches.append(i)
            require(len(matches) == 1, 'Unidentified exited original child')
            index = matches[0]
        else:
            index = worker_index(plan, info)
        require(index not in adopted, 'Duplicate original worker ownership')
        adopted[index] = info
    for index, row in enumerate(status['groups']):
        if row.get('status') == 'running' and index not in adopted:
            identity = row.get('worker_identity')
            require(isinstance(identity, dict), 'Running worker lacks starttime identity')
            current = proc(identity['pid'])
            require(current is None or identity_matches(identity, current), 'Worker PID was reused before adoption')
            require(current is None or current['state'] == 'Z', 'Unaccounted live original worker')
            adopted[index] = {**identity, 'state': 'gone' if current is None else 'Z'}
    # A status write can lag a just-reaped worker and its replacement. Keep
    # the exited identity for terminal reconciliation, but only actual child
    # processes consume the four original slots.
    require(1 <= len(adopted) <= len(plan['groups']) and len(children) <= 4,
            'Expected at most four actual original owned workers for adoption')
    extras = [i for i, g in enumerate(plan['groups']) if g['task'] == EXTRA_TASK]
    for index in extras:
        if index not in adopted:
            require(status['groups'][index]['status'] in ('queued', 'complete'), 'USB worker not accounted for during pause')
            if status['groups'][index]['status'] == 'queued':
                guard_fresh_group(plan, driver, index)
    descendants = {}
    for index in adopted:
        run = ready_read(driver.output_dir(plan, index) / 'run/run_manifest.json')
        if run:
            rows = {}
            for key in ('server_pid', 'evaluation_pid'):
                pid = run.get(key)
                if type(pid) is int:
                    current = proc(pid)
                    if current is not None:
                        rows[key] = current
            descendants[str(index)] = rows
    return dict(status=status, adopted={str(index): info for index, info in adopted.items()},
                owned_run_processes=descendants, extra_indices=extras)


def wait_file(path, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, 'Successor scheduler exited before activation')
        payload = ready_read(path)
        if payload is not None:
            return payload
        time.sleep(0.05)
    raise TimeoutError(f'Successor scheduler readiness timeout: {path}')


def handoff(plan_path, parallel_groups=CAP):
    require(parallel_groups == CAP, 'Only explicitly authorized four to six handoff is supported')
    plan, driver, owner, _status, proof = load_context(plan_path)
    job = Path(plan['job_root'])
    lock = (job / '_scheduler_handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = job / ('_scheduler_handoff_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    directory.mkdir()
    (directory / 'claims').mkdir()
    record = dict(schema='kaihand-staged-queue-handoff-v1', status='arming', parallel_groups=CAP,
                  original_parallel_groups=4, original_controller=owner, plan=str(Path(plan_path).absolute()),
                  source_proof=proof, scheduler_sha256=sha(__file__), created_utc=now(), new_workers_started=0)
    write(directory / 'record.json', record)
    successor, stopped = None, False
    try:
        with (directory / 'supervisor.log').open('xb', buffering=0) as stream:
            successor = subprocess.Popen([str(CONTROLLER_PYTHON), str(Path(__file__).absolute()),
                '--plan', record['plan'], '--supervise', str(directory), '--lock-fd', str(lock.fileno())],
                cwd=job, env=driver.base.environment(plan), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        ready = wait_file(directory / 'ready.json', successor)
        require(identity_matches(ready.get('identity'), proc(successor.pid)), 'Successor identity/readiness mismatch')
        exact_process(owner)
        signal_process(owner, signal.SIGSTOP)
        stopped = True
        deadline = time.monotonic() + 5
        while exact_process(owner)['state'] not in ('T', 't'):
            require(time.monotonic() < deadline, 'Original controller did not pause')
            time.sleep(0.02)
        snapshot = paused_snapshot(plan, driver, owner)
        write(directory / 'original_status.json', snapshot['status'])
        write(directory / 'activation.json', {**snapshot, 'original_controller': owner, 'source_proof': proof})
        wait_file(directory / 'adopted_ready.json', successor)
        # This durable barrier commits ownership. Never resume the old queue
        # after it exists: new workers may have been spawned already.
        write(directory / 'go.json', dict(committed_utc=now(), parallel_groups=CAP))
        print(json.dumps(dict(supervisor_pid=successor.pid, handoff_dir=str(directory),
                              original_controller_paused=owner['pid'], parallel_groups=CAP), indent=2), flush=True)
        return directory
    except BaseException as error:
        if not (directory / 'go.json').exists():
            if successor is not None and successor.poll() is None:
                successor.terminate()
                successor.wait(timeout=15)
            if stopped:
                # Also cancels a still-pending STOP if the process was briefly
                # uninterruptible; identity remains pinned by pidfd.
                signal_process(owner, signal.SIGCONT)
            record.update(status='rolled_back_before_launch', error=f'{type(error).__name__}: {error}', updated_utc=now())
            write(directory / 'record.json', record)
        raise
    finally:
        lock.close()


def claim(directory, index, payload):
    path = directory / 'claims' / f'group_{index:03d}.json'
    with path.open('x') as stream:
        json.dump(payload, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def exited(identity):
    current = proc(identity['pid'])
    if current is None:
        return True
    require(identity_matches(identity, current), 'Adopted worker PID reused; do not signal original controller')
    return current['state'] in ('Z', 'X')


def cleanup_confirmed(plan, driver, index, detail, expected=None):
    if detail.get('owned_cleanup_confirmed') is not True or not driver.port_free(plan['groups'][index]['port']):
        return False
    run = ready_read(driver.output_dir(plan, index) / 'run/run_manifest.json')
    if run is None:
        return detail.get('stage') in ('preflight', 'prepare', 'verify')
    if run.get('managed_server_stopped') is not True or run.get('evaluation_process_stopped') is not True:
        return False
    for key in ('server_pid', 'evaluation_pid'):
        pid = run.get(key)
        if type(pid) is not int:
            return False
        current = proc(pid)
        if current is not None:
            if expected and key in expected and not identity_matches(expected[key], current):
                return False
            if current['state'] not in ('Z', 'X'):
                return False
    return True


def finish_original(owner, adopted, *, lock_path=None, timeout=30):
    require(all(exited(info) for info in adopted.values()), 'Original workers still alive; cannot finish controller')
    exact_process(owner, states={'T', 't'})
    # TERM is pending while stopped. CONT then dispatches the handler before
    # normal queue execution can claim another group. Original Popen workers
    # are all gone/zombie, so its cleanup does not signal any live worker.
    signal_process(owner, signal.SIGTERM, states={'T', 't'})
    signal_process(owner, signal.SIGCONT, states={'T', 't'})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = proc(owner['pid'])
        if current is None or current['state'] in ('Z', 'X'):
            if lock_path is not None:
                with Path(lock_path).open('r') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(0.05)
                        continue
                    fcntl.flock(lock, fcntl.LOCK_UN)
            return
        require(identity_matches(owner, current), 'Original controller PID reused during finish')
        time.sleep(0.05)
    raise TimeoutError('Original controller cleanup did not finish')


def supervise(plan_path, directory, lock_fd):
    directory = Path(directory)
    # The inherited flock keeps the global handoff claim alive even after the
    # launcher exits. Do not close/reopen its shared open-file description.
    require(lock_fd is not None and os.fstat(lock_fd).st_ino
            == (directory.parent / '_scheduler_handoff.lock').stat().st_ino, 'Successor ownership lock missing')
    record = read(directory / 'record.json')
    require(record.get('scheduler_sha256') == sha(__file__), 'Successor scheduler code differs from armed proof')
    plan, driver, owner, _status, proof = load_context(plan_path)
    require(proof == record['source_proof'] and identity_matches(record['original_controller'], owner), 'Handoff source/owner proof changed')
    write(directory / 'ready.json', dict(identity=proc(os.getpid()), ready_utc=now()))
    while not (directory / 'activation.json').exists():
        time.sleep(0.05)
    activation = read(directory / 'activation.json')
    require(activation['source_proof'] == proof and identity_matches(owner, activation['original_controller']), 'Activation proof mismatch')
    exact_process(owner, states={'T', 't'})
    adopted = {int(index): info for index, info in activation['adopted'].items()}
    status = activation['status']
    require(status['run_id'] == plan['run_id'], 'Activation queue run mismatch')
    for index, info in adopted.items():
        require(info.get('ppid') == owner['pid'] and (exited(info) or worker_index(plan, exact_process(info)) == index),
                'Adopted worker ownership/identity mismatch')
        claim(directory, index, dict(origin='adopted', group_index=index, identity=info, claimed_utc=now()))
        status['groups'][index].update(status='running', worker_pid=info['pid'], worker_identity=info, scheduler_origin='adopted')
    write(directory / 'adopted_ready.json', dict(adopted_count=len(adopted), ready_utc=now()))
    while not (directory / 'go.json').exists():
        time.sleep(0.05)
    require(read(directory / 'go.json').get('parallel_groups') == CAP, 'Committed cap mismatch')
    status.update(pid=os.getpid(), identity=proc(os.getpid()), status='running', stage='parallel_staged_handoff',
        effective_parallel_groups=CAP, original_parallel_groups=4,
        scheduler_handoff=dict(directory=str(directory), parallel_groups=CAP, original_parallel_groups=4,
                               original_controller=owner, original_controller_state='paused', adopted_count=len(adopted)))
    record.update(status='active', supervisor=proc(os.getpid()), activated_utc=now())
    write(directory / 'record.json', record)
    workers, pending_adopted, quarantined = {}, dict(adopted), set()
    old_finished = False
    try:
        while workers or pending_adopted or any(row['status'] == 'queued' for row in status['groups']):
            if not old_finished:
                exact_process(owner, states={'T', 't'})
            done = []
            for index, info in pending_adopted.items():
                if exited(info):
                    done.append((index, None))
            for index, process in workers.items():
                code = process.poll()
                if code is not None:
                    done.append((index, code))
            for index, code in done:
                row = status['groups'][index]
                detail = ready_read(driver.output_dir(plan, index) / 'group_status.json') or {}
                clean = cleanup_confirmed(plan, driver, index, detail,
                                          activation.get('owned_run_processes', {}).get(str(index)))
                complete = (code is None or code == 0) and detail.get('status') == 'complete' and clean and detail.get('valid_trials') == 20
                row.update(status='complete' if complete else 'failed', worker_status=detail,
                           worker_returncode=code, owned_cleanup_confirmed=clean, finished_utc=now())
                if not complete:
                    status['errors'].append(f"{row['task']} step{row['step']}: handoff worker incomplete")
                if not clean:
                    quarantined.add(index)
                pending_adopted.pop(index, None)
                workers.pop(index, None)
            if not pending_adopted and not old_finished and all(status['groups'][index].get('owned_cleanup_confirmed') is True
                                                             for index in adopted):
                finish_original(owner, adopted, lock_path=Path(plan['job_root']) / 'controller.lock')
                old_finished = True
                status['scheduler_handoff']['original_controller_state'] = 'finished'
                record.update(original_controller_finished_utc=now())
                write(directory / 'record.json', record)
                # Old owner may have written interrupted status during cleanup;
                # only now restore authoritative successor status.
                driver.refresh(plan, status)
            capacity = max(0, CAP - len(workers) - len(pending_adopted) - len(quarantined))
            queued = [row for row in status['groups'] if row['status'] == 'queued']
            for row in queued[:capacity]:
                index = row['group_index']
                try:
                    guard_fresh_group(plan, driver, index)
                    claim_path = claim(directory, index, dict(origin='successor', group_index=index, claimed_utc=now()))
                    process = driver.launch_worker(plan, index)
                    workers[index] = process
                    row.update(status='running', worker_pid=process.pid, worker_identity=proc(process.pid), scheduler_origin='successor')
                    record['new_workers_started'] += 1
                    write(claim_path, dict(origin='successor', group_index=index, identity=row['worker_identity'], launched_utc=now()))
                    write(directory / 'record.json', record)
                except Exception as error:
                    row.update(status='failed', stage='handoff_launch_failed', error=str(error))
                    status['errors'].append(f"{row['task']} step{row['step']}: handoff launch failed: {error}")
            driver.refresh(plan, status)
            write(directory / 'status.json', status)
            if workers or pending_adopted:
                time.sleep(10)
            elif capacity == 0 and queued:
                for row in queued:
                    row.update(status='failed', stage='not_launched_owned_cleanup_unconfirmed')
                status['errors'].append('Remaining queue blocked by unconfirmed worker cleanup')
        status.update(status='failed' if status['errors'] else 'complete', stage='complete', finished_utc=now())
        record.update(status=status['status'], finished_utc=now(),
                      original_controller_state='finished' if old_finished else 'paused_requires_recovery')
        write(directory / 'record.json', record)
        driver.refresh(plan, status)
        write(directory / 'status.json', status)
        return int(bool(status['errors']))
    except BaseException as error:
        # Never resume the original queue after the launch barrier, and never
        # terminate healthy evaluation workers merely because scheduling failed.
        record.update(status='supervisor_failed_requires_recovery', error=f'{type(error).__name__}: {error}',
                      original_controller_state='finished' if old_finished else 'paused', updated_utc=now())
        write(directory / 'record.json', record)
        status.update(status='failed', stage='scheduler_recovery_required', scheduler_error=record['error'])
        driver.refresh(plan, status)
        write(directory / 'status.json', status)
        return 1
    finally:
        os.close(lock_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--parallel-groups', type=int, default=CAP)
    parser.add_argument('--handoff', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--supervise', type=Path)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    require(sum((args.handoff, args.preflight, args.supervise is not None)) <= 1, 'Select one handoff mode')
    require(args.parallel_groups == CAP, 'Only explicitly authorized cap six is supported')
    if args.supervise:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'received scheduler signal {signum}')
        prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            return supervise(args.plan, args.supervise, args.lock_fd)
        except BaseException as error:
            if (args.supervise / 'go.json').exists():
                record = read(args.supervise / 'record.json')
                record.update(status='supervisor_failed_requires_recovery', error=f'{type(error).__name__}: {error}', updated_utc=now())
                write(args.supervise / 'record.json', record)
            raise
        finally:
            for sig, handler in prior.items():
                signal.signal(sig, handler)
    if args.handoff:
        handoff(args.plan, args.parallel_groups)
    else:
        plan, _driver, owner, status, proof = load_context(args.plan)
        print(json.dumps(dict(valid=True, mode='read_only_preflight', original_controller=owner,
            active_groups=sum(row['status'] == 'running' for row in status['groups']), parallel_groups=CAP,
            original_parallel_groups=4, authorized_extra_task=EXTRA_TASK, source_proof=proof), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
