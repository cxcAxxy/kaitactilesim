#!/usr/bin/env python3
"""Independent, resource-gated retry of the two failed staged Poker groups.

Never signals existing controllers or workers, and never edits the original job.
The exact frozen staged worker owns preparation, rollout, cleanup and publication.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
GIB = 1024 ** 3
CAP = 2
GLOBAL_CAP = 8
OLD_CAP = 6
FRESH_SECONDS = 60
SAMPLE_SECONDS = 20
FIX_SOURCES = {
    'staged_observer_sha256': 'src/kaihand_tactile_env/shared/staged_evaluation.py',
    'poker_batch_sha256': 'scripts/workcell/evaluate_poker_pi05_policy_batch.py',
}
CPU_CGROUP = Path('/sys/fs/cgroup/cpu')
MEMORY_CGROUP = Path('/sys/fs/cgroup/memory')
V2_CGROUP = Path('/sys/fs/cgroup')
OWNED_SERVICE_IDENTITIES = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def ready_read(path):
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def proc(pid):
    require(type(pid) is int and pid > 1, 'Invalid process PID')
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode() for part in (root / 'cmdline').read_bytes().split(b'\0') if part]
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), starttime=int(fields[19]), argv=argv)
    except FileNotFoundError:
        return None


def matches(expected, current):
    return (isinstance(expected, dict) and current is not None
            and type(expected.get('pid')) is int and type(expected.get('starttime')) is int
            and expected['pid'] == current['pid'] and expected['starttime'] == current['starttime']
            and ('argv' not in expected or expected['argv'] == current['argv']))


def load_driver(plan):
    """Load the exact frozen helper even if a caller already imported a mutable one."""
    directory = Path(plan['simulation_root']) / 'scripts/deploy'
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    helper = load('poker_retry_frozen_helper', directory / 'run_pi05_explicit_groups.py')
    previous = sys.modules.get('run_pi05_explicit_groups')
    sys.modules['run_pi05_explicit_groups'] = helper
    try:
        driver = load('poker_retry_frozen_staged', directory / 'run_pi05_staged_groups.py')
    finally:
        if previous is None:
            sys.modules.pop('run_pi05_explicit_groups', None)
        else:
            sys.modules['run_pi05_explicit_groups'] = previous
    require(Path(driver.__file__).resolve().parent == directory
            and Path(driver.base.__file__).resolve().parent == directory, 'Imported non-frozen worker/helper')
    return driver


def old_plan(plan, driver):
    root = driver.safe(plan['retry_of'])
    require(root.parent == driver.JOBS_ROOT and root != Path(plan['job_root']), 'Invalid original job scope')
    original = read(root / 'plan.json')
    require(original.get('job_root') == str(root), 'Original plan job mismatch')
    indices = plan.get('replaces_group_indices')
    require(indices == [10, 11], 'Retry replaces only original Poker groups 10/11')
    require(plan.get('authorized_tasks') == ['poker-draw'] and len(plan['groups']) == 2
            and [g['step'] for g in plan['groups']] == [20000, 24999]
            and plan['settings']['parallel_groups'] == CAP and plan.get('port_start') == 18830,
            'Retry must contain exactly the authorized Poker pair with cap two')
    require({key: value for key, value in plan['settings'].items() if key != 'parallel_groups'}
            == {key: value for key, value in original['settings'].items() if key != 'parallel_groups'},
            'Retry may not change evaluation settings')
    for group, index in zip(plan['groups'], indices):
        prior = original['groups'][index]
        require(all(group[key] == prior[key] for key in ('task', 'step', 'dataset', 'checkpoint', 'port')),
                'Retry checkpoint/task/dataset/port differs from failed original group')
    return original


def cleaned(driver, output, group, detail, expected_pids=None):
    if detail.get('owned_cleanup_confirmed') is not True or not driver.port_free(group['port']):
        return False
    run = ready_read(output / 'run/run_manifest.json')
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
            expected = (expected_pids or {}).get(key)
            if expected is not None and not matches(expected, current):
                return False
            if current['state'] not in ('Z', 'X'):
                return False
    return True


def listener_inodes(port):
    """Every local LISTEN socket at the reused port must belong to our server."""
    result = set()
    for path in (Path('/proc/net/tcp'), Path('/proc/net/tcp6')):
        try:
            lines = path.read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        for line in lines:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                result.add(fields[9])
    return result


def socket_inodes(pid):
    result = set()
    for descriptor in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if target.startswith('socket:[') and target.endswith(']'):
            result.add(target[8:-1])
    return result


def direct_children(pid):
    try:
        return [int(value) for value in (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()]
    except FileNotFoundError:
        return []


def owned_retry_listener(plan, driver, index):
    """Authenticate actual socket ownership, even before server_pid publication."""
    job = Path(plan['job_root'])
    claim = ready_read(job / 'claims' / f'group_{index:03d}.json') or {}
    worker = proc(claim.get('identity', {}).get('pid')) if claim.get('identity') else None
    expected_argv = [str(Path(plan['simulation_root']) / '.pixi/envs/default/bin/python'),
        str(Path(plan['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
        '--plan', str(job / 'plan.json'), '--worker-index', str(index)]
    controller = ready_read(job / 'controller_identity.json') or {}
    if (claim.get('group_index') != index or not matches(claim.get('identity'), worker)
            or worker['state'] in ('T', 't', 'Z', 'X') or worker['argv'] != expected_argv
            or worker['ppid'] != controller.get('pid') or not matches(controller, proc(controller.get('pid')))):
        return False
    output = driver.output_dir(plan, index)
    run = ready_read(output / 'run/run_manifest.json') or {}
    deployment = ready_read(output / 'deployment/deployment_manifest.json') or {}
    command = run.get('server', {}).get('command')
    if (not isinstance(command, list) or run.get('endpoint') != f"ws://127.0.0.1:{plan['groups'][index]['port']}"
            or not deployment.get('deployment_id') or run.get('deployment_id') != deployment['deployment_id']
            or deployment.get('task') != 'poker-draw'
            or deployment.get('checkpoint_path') != plan['groups'][index]['checkpoint']):
        return False
    candidates = direct_children(worker['pid'])
    if type(run.get('server_pid')) is int:
        candidates.insert(0, run['server_pid'])
    sockets = listener_inodes(plan['groups'][index]['port'])
    if not sockets:
        return False
    cache_key = (str(job), index)
    for pid in dict.fromkeys(candidates):
        server = proc(pid)
        if (server is None or server['ppid'] != worker['pid'] or server['argv'] != command
                or server['state'] in ('T', 't', 'Z', 'X') or server['starttime'] < worker['starttime']):
            continue
        pinned = OWNED_SERVICE_IDENTITIES.get(cache_key)
        if pinned is not None and not matches(pinned, server):
            continue
        if sockets <= socket_inodes(pid) and matches(server, proc(pid)) and matches(worker, proc(worker['pid'])):
            OWNED_SERVICE_IDENTITIES.setdefault(cache_key, server.copy())
            return True
    return False


def pin_original_cleanup(plan, driver):
    """Called only before launch, while both original/retry ports are free."""
    original = old_plan(plan, driver)
    status = read(Path(plan['retry_of']) / 'status.json')
    proofs = []
    for index in plan['replaces_group_indices']:
        output = driver.output_dir(original, index)
        detail = read(output / 'group_status.json')
        row = status['groups'][index]
        require(row['status'] == 'failed' and detail.get('status') == 'failed'
                and cleaned(driver, output, original['groups'][index], detail, row.get('owned_run_pids')),
                'Cannot pin original Poker cleanup before retry')
        run = read(output / 'run/run_manifest.json')
        pids = [dict(kind='worker', pid=detail['pid'], identity=detail.get('identity'))]
        pids.extend(dict(kind=key, pid=run[key], identity=row.get('owned_run_pids', {}).get(key))
                    for key in ('server_pid', 'evaluation_pid'))
        require(all(proc(item['pid']) is None or proc(item['pid'])['state'] in ('Z', 'X') for item in pids),
                'Original Poker process still alive when pinning cleanup')
        proofs.append(dict(group_index=index, port=original['groups'][index]['port'], output_dir=str(output),
            group_status_sha256=driver.sha(output / 'group_status.json'),
            run_manifest_sha256=driver.sha(output / 'run/run_manifest.json'), exited_processes=pids))
    return proofs


def verify_pinned_cleanup(plan, driver, original, row, proof, retry_index):
    index = plan['replaces_group_indices'][retry_index]
    output = driver.output_dir(original, index)
    require(proof.get('group_index') == index and proof.get('port') == original['groups'][index]['port']
            and proof.get('output_dir') == str(output), 'Pinned original Poker cleanup scope mismatch')
    require(driver.sha(output / 'group_status.json') == proof['group_status_sha256']
            and driver.sha(output / 'run/run_manifest.json') == proof['run_manifest_sha256'],
            'Original failed Poker cleanup records changed after authentication')
    for item in proof['exited_processes']:
        current = proc(item['pid'])
        require(current is None or (current['state'] in ('Z', 'X')
                and (item.get('identity') is None or matches(item['identity'], current))),
                'Original Poker PID appeared live or was reused after cleanup')
    if not driver.port_free(original['groups'][index]['port']):
        require(owned_retry_listener(plan, driver, retry_index),
                'Reused Poker port is occupied by an unverified or external listener')
    return True


def original_snapshot(plan, driver, *, wall_time=None):
    original = old_plan(plan, driver)
    status = read(Path(plan['retry_of']) / 'status.json')
    require(status.get('run_id') == original['run_id']
            and status.get('staged_evaluation_version') == 'staged_v1'
            and len(status.get('groups', [])) == len(original['groups']), 'Original status identity mismatch')
    groups = status['groups']
    for index, row in enumerate(groups):
        require(row.get('group_index') == index and all(row.get(key) == original['groups'][index][key]
                for key in ('task', 'step', 'dataset', 'checkpoint', 'port')), 'Original status group identity changed')
    running = [row for row in groups if row['status'] == 'running']
    queued = sum(row['status'] == 'queued' for row in groups)
    quarantined = sum(row['status'] == 'failed' and row.get('owned_cleanup_confirmed') is not True for row in groups)
    require(len(running) + quarantined <= OLD_CAP, 'Original job exceeds its six leased slots')
    if status.get('status') == 'running':
        stamp = datetime.fromisoformat(status['updated_utc']).timestamp()
        age = (time.time() if wall_time is None else wall_time) - stamp
        require(-5 <= age <= FRESH_SECONDS, 'Original running status is stale')
        identity = status.get('identity')
        owner = proc(status.get('pid'))
        require(matches(identity, owner) and owner['state'] not in ('T', 't', 'Z', 'X'),
                'Original scheduler PID/starttime/command is not live')
        require(str(Path(plan['retry_of']) / 'plan.json') in owner['argv'], 'Original scheduler owns another plan')
    else:
        require(status.get('status') in ('complete', 'failed') and status.get('stage') == 'complete'
                and not running and not queued and not quarantined, 'Original scheduler requires recovery')
        # Completed terminal statuses need not remain fresh forever.
        require(all(row.get('owned_cleanup_confirmed') is True for row in groups),
                'Original terminal job has unconfirmed cleanup')
    for row in running:
        identity = row.get('worker_identity')
        current = proc(row.get('worker_pid'))
        expected = [str(Path(original['simulation_root']) / '.pixi/envs/default/bin/python'),
            str(Path(original['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
            '--plan', str(Path(plan['retry_of']) / 'plan.json'), '--worker-index', str(row['group_index'])]
        require(matches(identity, current) and current['state'] not in ('T', 't', 'Z', 'X')
                and current['argv'] == expected, 'Original worker PID/starttime/command mismatch')
    record = ready_read(Path(plan['job_root']) / 'retry_record.json') or {}
    pinned = record.get('original_cleanup_proof')
    require(pinned is None or isinstance(pinned, list) and len(pinned) == 2,
            'Malformed pinned original Poker cleanup proof')
    for retry_index, index in enumerate(plan['replaces_group_indices']):
        row = groups[index]
        confirmed = verify_pinned_cleanup(plan, driver, original, row, pinned[retry_index], retry_index) if pinned else (
            cleaned(driver, driver.output_dir(original, index), original['groups'][index],
                    row.get('worker_status', {}), row.get('owned_run_pids')))
        require(row['status'] == 'failed' and row.get('owned_cleanup_confirmed') is True and confirmed,
                'Original Poker group is not failed with confirmed owned cleanup')
    # A still-queued old job can refill to six at any time. Budget its full cap,
    # not merely a transient count observed between two launches.
    return dict(live_groups=len(running), queued_groups=queued, quarantined_groups=quarantined,
                reserved_groups=OLD_CAP if queued else len(running) + quarantined,
                terminal=status.get('status') != 'running', updated_utc=status.get('updated_utc'))


def source_proof(plan, driver):
    driver.verify_frozen(plan)
    trees = ('src', 'scripts', 'configs')
    sources = []
    for root, names in ((Path(plan['simulation_root']), trees),
                        (Path(plan['openpi_root']), ('src', 'scripts', 'packages'))):
        for tree in names:
            for path in sorted((root / tree).rglob('*')):
                if path.is_file():
                    sources.append((str(path), driver.sha(path)))
    directory = Path(plan['simulation_root']) / 'scripts/deploy'
    return dict(frozen_driver_sha256=driver.sha(directory / 'run_pi05_staged_groups.py'),
        frozen_helper_sha256=driver.sha(directory / 'run_pi05_explicit_groups.py'),
        preflight_sha256=driver.sha(Path(plan['simulation_root']).parent / 'preflight_report.json'),
        original_plan_sha256=driver.sha(Path(plan['retry_of']) / 'plan.json'),
        readonly_source_files=len(sources),
        readonly_source_digest=hashlib.sha256(json.dumps(sources, separators=(',', ':')).encode()).hexdigest(),
        **{key: driver.sha(Path(plan['simulation_root']) / relative) for key, relative in FIX_SOURCES.items()})


def validate_retry(plan, driver, *, existing_job=False):
    driver.validate_plan(plan, existing_job=existing_job, allow_published=existing_job)
    old_plan(plan, driver)
    driver.verify_frozen(plan)
    original_snapshot(plan, driver)


def start(plan):
    driver = load_driver(plan)
    validate_retry(plan, driver)
    proof = source_proof(plan, driver)
    require(all(driver.port_free(g['port']) for g in plan['groups']), 'Poker retry service port is occupied')
    original_cleanup_proof = pin_original_cleanup(plan, driver)
    tokenizer = Path(plan['retry_of']) / '.cache/openpi/big_vision' / driver.base.TOKENIZER_SOURCE.name
    require(tokenizer.is_file() and not tokenizer.is_symlink(), 'Original cached tokenizer missing')
    job = driver.safe(plan['job_root'])
    job.mkdir(parents=True, exist_ok=False)
    tools = job / '_scheduler_tools'
    tools.mkdir()
    controller = tools / Path(__file__).name
    shutil.copyfile(Path(__file__), controller)
    controller.chmod(0o444)
    cache = job / '.cache/openpi/big_vision'
    cache.mkdir(parents=True)
    shutil.copy2(tokenizer, cache / tokenizer.name)
    driver.write(job / 'plan.json', plan)
    for name in ('preflight_report.json', 'snapshot_record.json'):
        path = Path(plan['simulation_root']).parent / name
        require(path.is_file() and not path.is_symlink(), f'Frozen provenance missing: {name}')
        shutil.copy2(path, job / name)
    standard = Path(plan['retry_of']) / 'stage_standard.json'
    if standard.is_file():
        shutil.copy2(standard, job / standard.name)
    proof.update(controller_sha256=driver.sha(controller), plan_sha256=driver.sha(job / 'plan.json'))
    driver.write(job / 'retry_record.json', dict(schema='pi05-poker-staged-retry-v1', retry_of=plan['retry_of'],
        replaces_group_indices=plan['replaces_group_indices'], source_proof=proof,
        original_cleanup_proof=original_cleanup_proof, created_utc=driver.now()))
    driver.write(job / 'status.json', dict(status='starting', stage='resource_gated_poker_retry',
        total_trials=40, retry_of=plan['retry_of'], replaces_group_indices=plan['replaces_group_indices']))
    with (job / 'controller.log').open('xb', buffering=0) as stream:
        process = subprocess.Popen([str(driver.base.CONTROLLER_PYTHON), str(controller),
            '--plan', str(job / 'plan.json')], cwd=plan['simulation_root'], env=driver.base.environment(plan),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    identity = proc(process.pid)
    require(identity is not None, 'Detached retry controller identity unavailable')
    driver.write(job / 'controller_identity.json', identity)
    print(json.dumps(dict(pid=process.pid, identity=identity, job_root=str(job),
        status=str(job / 'status.json'), total_trials=40), indent=2), flush=True)
    return process.pid


def key_values(path):
    return {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}


def resource_snapshot():
    if (V2_CGROUP / 'cpu.max').is_file():
        quota, period = (V2_CGROUP / 'cpu.max').read_text().split()
        require(quota != 'max', 'CPU quota must be bounded')
        cpu = key_values(V2_CGROUP / 'cpu.stat')
        usage_ns, cores = cpu['usage_usec'] * 1000, int(quota) / int(period)
        throttle_ns = cpu.get('throttled_usec', 0) * 1000
        maximum = (V2_CGROUP / 'memory.max').read_text().strip()
        require(maximum != 'max', 'Memory quota must be bounded')
        limit, used = int(maximum), int((V2_CGROUP / 'memory.current').read_text())
        memory = key_values(V2_CGROUP / 'memory.stat')
    else:
        cpu = key_values(CPU_CGROUP / 'cpu.stat')
        usage_ns = int((CPU_CGROUP / 'cpuacct.usage').read_text())
        cores = int((CPU_CGROUP / 'cpu.cfs_quota_us').read_text()) / int((CPU_CGROUP / 'cpu.cfs_period_us').read_text())
        throttle_ns = cpu['throttled_time']
        limit = int((MEMORY_CGROUP / 'memory.limit_in_bytes').read_text())
        used = int((MEMORY_CGROUP / 'memory.usage_in_bytes').read_text())
        memory = key_values(MEMORY_CGROUP / 'memory.stat')
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    total, allocated = [int(value.strip()) * 1024 ** 2 for value in result.stdout.strip().split(',')]
    return dict(monotonic=time.monotonic(), cpu_usage_ns=usage_ns, cpu_quota_cores=cores,
        nr_throttled=cpu['nr_throttled'], throttled_ns=throttle_ns,
        gpu_free_bytes=total - allocated,
        memory_headroom_bytes=limit - used + memory.get('total_inactive_file', memory.get('inactive_file', 0)))


def evaluate_samples(samples):
    require(len(samples) >= 2, 'Resource gate requires actual interval samples')
    elapsed = samples[-1]['monotonic'] - samples[0]['monotonic']
    require(elapsed >= SAMPLE_SECONDS - 0.1, 'Resource gate must observe at least twenty seconds')
    cpu = (samples[-1]['cpu_usage_ns'] - samples[0]['cpu_usage_ns']) / (elapsed * 1e9)
    throttle = samples[-1]['nr_throttled'] - samples[0]['nr_throttled']
    throttle_ns = samples[-1]['throttled_ns'] - samples[0]['throttled_ns']
    gpu = min(row['gpu_free_bytes'] for row in samples)
    memory = min(row['memory_headroom_bytes'] for row in samples)
    passed = (all(row['cpu_quota_cores'] == 12 for row in samples) and 0 <= cpu < 10.5
              and throttle == throttle_ns == 0 and gpu >= 12 * GIB and memory >= 20 * GIB)
    return dict(passed=passed, elapsed_s=elapsed, cpu_average_cores=cpu,
        cpu_limit_cores=10.5, nr_throttled_delta=throttle, throttled_ns_delta=throttle_ns,
        minimum_gpu_free_gib=gpu / GIB, minimum_memory_headroom_gib=memory / GIB,
        samples=len(samples))


def capacity_allows(original, workers, quarantined):
    occupied = len(workers) + len(quarantined)
    return occupied < CAP and original['reserved_groups'] + occupied < GLOBAL_CAP


def warmup_ready(plan, driver, row):
    if row.get('warmup_verified') is True:
        return True
    if row['status'] == 'failed' and row.get('owned_cleanup_confirmed') is True:
        # A failed cold start has no service left to warm; retry the healthy peer
        # under the same cold-start resource gate instead of cancelling it.
        row['warmup_bypass_reason'] = 'failed_peer_cleaned_requires_new_cold_start_gate'
        return True
    output = driver.output_dir(plan, 0)
    if row['status'] == 'complete' and row.get('owned_cleanup_confirmed') is True:
        detail = ready_read(output / 'group_status.json') or {}
        deployment = ready_read(output / 'deployment/deployment_manifest.json')
        if (detail.get('status') == 'complete' and detail.get('valid_trials') == 20
                and deployment is not None and cleaned(driver, output, plan['groups'][0], detail,
                                                       row.get('owned_run_pids'))):
            # A full validated 20-seed group may finish between polls, after its
            # server has exited. Authenticate full traces/contracts rather than
            # trust a status string or require an already-retired server PID.
            driver.validate_native(output / 'run/evaluation', plan['groups'][0], deployment)
            row.update(warmup_verified=True, warmup_sim_seconds=90,
                warmup_evidence='validated_completed_twenty_seed_group', warmup_verified_utc=driver.now())
            return True
    path = output / 'run/evaluation/seed_000/staged_states.jsonl'
    run = ready_read(output / 'run/run_manifest.json')
    if not path.is_file() or path.is_symlink() or run is None or type(run.get('server_pid')) is not int:
        return False
    current = proc(run['server_pid'])
    owned = row.get('owned_run_pids', {}).get('server_pid')
    if not matches(owned, current) or current['state'] in ('T', 't', 'Z', 'X'):
        return False
    if current['ppid'] != row.get('worker_pid') or current['argv'] != run.get('server', {}).get('command'):
        return False
    pinned = row.setdefault('warmup_server_identity', current)
    require(matches(pinned, current), 'Poker warmup server PID was reused or command changed')
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        lines = stream.read().splitlines()
    observation = None
    for line in reversed(lines):
        try:
            observation = json.loads(line)
            break
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    deployment = ready_read(output / 'deployment/deployment_manifest.json')
    if not observation or deployment is None:
        return False
    stamp = observation.get('time_s')
    if (type(stamp) not in (int, float) or not 3.4 < stamp <= 90.1
            or observation.get('version') != 'staged_v1' or observation.get('task') != 'poker-draw'
            or observation.get('seed') != 0 or observation.get('prediction_horizon') != 50
            or observation.get('execute_steps') != 50
            or observation.get('checkpoint_sha256') != deployment.get('checkpoint_sha256')
            or observation.get('deployment_id') != deployment.get('deployment_id')):
        return False
    row.update(warmup_verified=True, warmup_sim_seconds=stamp, warmup_verified_utc=driver.now())
    return True


def sample_gate(plan, driver, status):
    samples = [resource_snapshot()]
    while samples[-1]['monotonic'] - samples[0]['monotonic'] < SAMPLE_SECONDS:
        time.sleep(1)
        original = original_snapshot(plan, driver)
        status['original_resource_scope'] = original
        status['stage'] = 'sampling_resources_before_next_poker_launch'
        driver.refresh(plan, status)
        samples.append(resource_snapshot())
    return evaluate_samples(samples)


def guard_fresh(plan, driver, index):
    job = Path(plan['job_root'])
    for path in (driver.output_dir(plan, index), job / f'group_{index}.log', Path(plan['groups'][index]['archive_dir'])):
        require(not path.exists() and not path.is_symlink(), f'Poker retry group already exists: {path}')
    require(driver.port_free(plan['groups'][index]['port']), 'Poker retry selected port became occupied')


def verify_launch_proof(plan, driver, record):
    proof = record['source_proof']
    directory = Path(plan['simulation_root']) / 'scripts/deploy'
    for key, path in (('controller_sha256', Path(__file__)),
                      ('plan_sha256', Path(plan['job_root']) / 'plan.json'),
                      ('frozen_driver_sha256', directory / 'run_pi05_staged_groups.py'),
                      ('frozen_helper_sha256', directory / 'run_pi05_explicit_groups.py'),
                      ('preflight_sha256', Path(plan['simulation_root']).parent / 'preflight_report.json'),
                      ('original_plan_sha256', Path(plan['retry_of']) / 'plan.json'),
                      *((key, Path(plan['simulation_root']) / relative) for key, relative in FIX_SOURCES.items())):
        require(driver.sha(path) == proof[key], f'Poker retry launch source identity changed: {key}')


def run_controller(plan):
    driver = load_driver(plan)
    validate_retry(plan, driver, existing_job=True)
    job = Path(plan['job_root'])
    require(not Path(__file__).stat().st_mode & 0o222, 'Actual retry controller must be a readonly copy')
    record = read(job / 'retry_record.json')
    verify_launch_proof(plan, driver, record)
    proof = source_proof(plan, driver)
    require(all(record['source_proof'][key] == value for key, value in proof.items()), 'Frozen retry source proof changed')
    identity = proc(os.getpid())
    require(identity is not None, 'Retry controller identity unavailable')
    status = dict(status='running', stage='resource_gated_poker_retry', run_id=plan['run_id'],
        staged_evaluation_version='staged_v1', retry_of=plan['retry_of'],
        replaces_group_indices=plan['replaces_group_indices'], pid=os.getpid(), identity=identity,
        started_utc=driver.now(), total_trials=40, total_checkpoint_count=2, effective_parallel_groups=2,
        global_parallel_cap=8, settings=plan['settings'], authorized_tasks=['poker-draw'], errors=[],
        source_proof=record['source_proof'],
        groups=[{**group, 'group_index': index, 'status': 'queued',
                 'output_dir': str(driver.output_dir(plan, index)), 'replaces_group_index': 10 + index}
                for index, group in enumerate(plan['groups'])])
    workers, quarantined = {}, set()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}; workers are intentionally left running')
    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (job / 'controller.lock').open('x') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (job / 'claims').mkdir(exist_ok=False)
            driver.write(job / 'controller_identity.json', identity)
            while workers or any(row['status'] == 'queued' for row in status['groups']):
                for index, process in list(workers.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    row = status['groups'][index]
                    detail = ready_read(driver.output_dir(plan, index) / 'group_status.json') or {}
                    clean = cleaned(driver, driver.output_dir(plan, index), plan['groups'][index],
                                    detail, row.get('owned_run_pids'))
                    complete = code == 0 and clean and detail.get('status') == 'complete' and detail.get('valid_trials') == 20
                    row.update(status='complete' if complete else 'failed', worker_returncode=code,
                               worker_status=detail, owned_cleanup_confirmed=clean, finished_utc=driver.now())
                    del workers[index]
                    if not clean:
                        quarantined.add(index)
                        row['stage'] = 'owned_cleanup_unconfirmed'
                    if not complete:
                        status['errors'].append(f"Poker step{row['step']}: worker incomplete (exit {code})")
                driver.refresh(plan, status)
                queued = [row for row in status['groups'] if row['status'] == 'queued']
                if queued:
                    index = queued[0]['group_index']
                    try:
                        original = original_snapshot(plan, driver)
                        status['original_resource_scope'] = original
                        ready = index == 0 or warmup_ready(plan, driver, status['groups'][0])
                        if ready and capacity_allows(original, workers, quarantined):
                            gate = sample_gate(plan, driver, status)
                            status['last_resource_gate'] = gate
                            if gate['passed']:
                                original = original_snapshot(plan, driver)
                                require(capacity_allows(original, workers, quarantined), 'Global resource slots changed during sampling')
                                verify_launch_proof(plan, driver, record)
                                guard_fresh(plan, driver, index)
                                claim = job / 'claims' / f'group_{index:03d}.json'
                                with claim.open('x') as stream:
                                    json.dump(dict(group_index=index, claimed_utc=driver.now()), stream)
                                    stream.flush()
                                    os.fsync(stream.fileno())
                                process = driver.launch_worker(plan, index)
                                workers[index] = process
                                worker_identity = proc(process.pid)
                                require(worker_identity is not None, 'Launched Poker worker identity unavailable')
                                expected_argv = [str(Path(plan['simulation_root']) / '.pixi/envs/default/bin/python'),
                                    str(Path(plan['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
                                    '--plan', str(job / 'plan.json'), '--worker-index', str(index)]
                                require(worker_identity['pid'] == process.pid
                                        and worker_identity['ppid'] == os.getpid()
                                        and worker_identity['argv'] == expected_argv,
                                        'Launched Poker worker PID/parent/command does not match its exclusive claim')
                                status['groups'][index].update(status='running', worker_pid=process.pid,
                                    worker_identity=worker_identity, launched_utc=driver.now())
                                driver.write(claim, dict(group_index=index, identity=worker_identity, launched_utc=driver.now()))
                                status.pop('launch_gate_error', None)
                                status['stage'] = 'waiting_first_poker_real_rollout_warmup' if index == 0 else 'two_poker_groups_running'
                            else:
                                status['stage'] = 'waiting_for_resource_headroom'
                                status['resource_retry_when_old_drained'] = original['queued_groups'] == 0 and original['live_groups'] <= 5
                        else:
                            status['stage'] = 'waiting_first_poker_real_rollout_warmup' if not ready else 'waiting_for_global_capacity'
                    except (ValueError, FileNotFoundError, json.JSONDecodeError, subprocess.SubprocessError) as error:
                        # A stale old status, PID transition or tight resources
                        # delays launches. Existing rollouts are never touched.
                        status.update(stage='launch_gate_waiting', launch_gate_error=f'{type(error).__name__}: {error}')
                        if index in workers:
                            raise
                        if (job / 'claims' / f'group_{index:03d}.json').exists():
                            status['groups'][index].update(status='failed', stage='claimed_launch_requires_recovery',
                                                          error=str(error), owned_cleanup_confirmed=False)
                            quarantined.add(index)
                            status['errors'].append(f'Poker group {index}: claimed launch failed: {error}')
                driver.refresh(plan, status)
                if workers or any(row['status'] == 'queued' for row in status['groups']):
                    time.sleep(10)
            status.update(status='failed' if status['errors'] else 'complete', stage='complete', finished_utc=driver.now())
            driver.refresh(plan, status)
            return int(bool(status['errors']))
    except BaseException as error:
        # No terminate/kill/SIGCONT/SIGSTOP: all real workers retain their jobs.
        status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      stage='controller_recovery_required', recovery_required=True,
                      controller_error=f'{type(error).__name__}: {error}')
        driver.write(job / 'recovery_required.json', dict(identity=identity, error=status['controller_error'],
            active_workers={str(index): status['groups'][index].get('worker_identity') for index in workers},
            quarantined_group_indices=sorted(quarantined), created_utc=driver.now()))
        driver.refresh(plan, status)
        return 1
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    if args.start:
        start(plan)
        return 0
    require(args.plan.absolute() == Path(plan['job_root']) / 'plan.json', 'Use the saved retry job plan')
    return run_controller(plan)


if __name__ == '__main__':
    raise SystemExit(main())
