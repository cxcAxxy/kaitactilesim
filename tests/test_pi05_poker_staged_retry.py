"""Temporary fixtures/mocks only: no GPU jobs, signals, or real archive writes."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/deploy/retry_pi05_poker_staged_groups.py'


def load(path):
    spec = importlib.util.spec_from_file_location('poker_retry_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Load a readonly controller copy exactly as the real detached process does.
    controller = tmp_path / 'controller.py'
    controller.write_bytes(SCRIPT.read_bytes())
    controller.chmod(0o444)
    retry = load(controller)
    archive = tmp_path / 'archive'
    jobs = archive / '_jobs'
    old = jobs / 'pi05_staged_v1_original'
    job = jobs / 'pi05_staged_v1_poker_fix'
    sim = tmp_path / 'snapshot/kaitactilesim'
    openpi = tmp_path / 'snapshot/openpi'
    groups = [dict(task='whiteboard-wipe', step=20000 + index, dataset='0920_200',
        checkpoint=f'/checkpoints/{index}', port=18820 + index,
        archive_dir=str(archive / f'old_group_{index}')) for index in range(18)]
    for index, step in zip((10, 11), (20000, 24999)):
        groups[index].update(task='poker-draw', step=step, checkpoint=f'/poker/{step}')
    settings = dict(parallel_groups=4, num_trials=20, execute_steps=50,
                    prediction_horizon=50, max_sim_seconds=90, staged_evaluation_version='staged_v1')
    original = dict(run_id='staged_v1_original', job_root=str(old), simulation_root='/original/kaitactilesim',
                    openpi_root='/original/openpi', settings=settings, groups=groups)
    plan = dict(run_id='staged_v1_poker_fix', job_root=str(job), simulation_root=str(sim),
        openpi_root=str(openpi), settings={**settings, 'parallel_groups': 2}, retry_of=str(old),
        replaces_group_indices=[10, 11], authorized_tasks=['poker-draw'], port_start=18830,
        groups=[{**groups[index], 'archive_dir': str(archive / f'retry_{index}')} for index in (10, 11)])
    def output(config, index):
        group = config['groups'][index]
        return Path(config['job_root']) / 'groups' / group['task'] / f"step{group['step']}_N20_video20"
    def refresh(config, status):
        for index, row in enumerate(status['groups']):
            native = retry.ready_read(output(config, index) / 'run/evaluation/summary.json')
            if native:
                row.update(valid_trials=native['valid_trials'], completed_trials=native['attempted'])
        status.update(completed_trials=sum(row.get('completed_trials', 0) for row in status['groups']),
            valid_trials=sum(row.get('valid_trials', 0) for row in status['groups']),
            active_groups=sum(row['status'] == 'running' for row in status['groups']))
        write(Path(config['job_root']) / 'status.json', status)
    driver = SimpleNamespace(JOBS_ROOT=jobs, safe=lambda path: Path(path), read=retry.read,
        write=write, sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        now=lambda: datetime.now(timezone.utc).isoformat(), output_dir=output,
        validate_plan=MagicMock(), verify_frozen=MagicMock(), port_free=MagicMock(return_value=True),
        refresh=refresh, base=SimpleNamespace(TOKENIZER_SOURCE=Path('/cached/tokenizer.model'),
            CONTROLLER_PYTHON=Path('/usr/local/bin/python3.11'), environment=lambda config: {}))
    status = dict(status='failed', stage='complete', run_id=original['run_id'],
        staged_evaluation_version='staged_v1', updated_utc='2020-01-01T00:00:00+00:00',
        groups=[{**group, 'group_index': index, 'status': 'failed' if index in (10, 11) else 'complete',
                 'owned_cleanup_confirmed': True, 'worker_status': dict(stage='evaluate', owned_cleanup_confirmed=True)}
                for index, group in enumerate(groups)])
    write(old / 'plan.json', original)
    write(old / 'status.json', status)
    for index in (10, 11):
        write(output(original, index) / 'group_status.json', dict(status='failed', stage='evaluate',
            pid=8000 + index, identity=dict(pid=8000 + index, starttime=10), owned_cleanup_confirmed=True))
        write(output(original, index) / 'run/run_manifest.json', dict(server_pid=9000 + index,
            evaluation_pid=9100 + index, managed_server_stopped=True, evaluation_process_stopped=True))
    (old / '.cache/openpi/big_vision').mkdir(parents=True)
    (old / '.cache/openpi/big_vision/tokenizer.model').write_bytes(b'cached tokenizer')
    write(sim.parent / 'preflight_report.json', {'passed': True})
    write(sim.parent / 'snapshot_record.json', {'frozen': True})
    write(old / 'stage_standard.json', {'unchanged_threshold': 3})
    monkeypatch.setattr(retry, 'load_driver', lambda _: driver)
    monkeypatch.setattr(retry, 'proc', lambda pid: None)
    return SimpleNamespace(retry=retry, driver=driver, plan=plan, original=original,
        old=old, job=job, output=output, status=status, monkeypatch=monkeypatch)


def samples(**overrides):
    baseline = dict(monotonic=0, cpu_usage_ns=0, cpu_quota_cores=12, nr_throttled=100,
        throttled_ns=1000, gpu_free_bytes=15 * 1024 ** 3, memory_headroom_bytes=30 * 1024 ** 3)
    end = {**baseline, 'monotonic': 20, 'cpu_usage_ns': 180 * 10 ** 9, **overrides}
    return [baseline, end]


def test_resource_gate_passes_twenty_second_headroom(fixture):
    report = fixture.retry.evaluate_samples(samples())
    assert report['passed'] is True
    assert report['cpu_average_cores'] == 9
    assert report['nr_throttled_delta'] == 0


@pytest.mark.parametrize('overrides', [
    {'cpu_usage_ns': 210 * 10 ** 9},
    {'nr_throttled': 101},
    {'throttled_ns': 1001},
    {'gpu_free_bytes': 12 * 1024 ** 3 - 1},
    {'memory_headroom_bytes': 20 * 1024 ** 3 - 1},
    {'cpu_quota_cores': 8},
])
def test_resource_gate_delays_any_unsafe_launch(fixture, overrides):
    assert fixture.retry.evaluate_samples(samples(**overrides))['passed'] is False


def test_gate_cannot_skip_the_twenty_second_sampling_interval(fixture):
    with pytest.raises(ValueError, match='twenty seconds'):
        fixture.retry.evaluate_samples(samples(monotonic=19))


def test_original_terminal_status_may_be_old_but_requires_cleanup(fixture):
    scope = fixture.retry.original_snapshot(fixture.plan, fixture.driver)
    assert scope['terminal'] is True and scope['reserved_groups'] == 0
    fixture.status['groups'][0]['owned_cleanup_confirmed'] = False
    write(fixture.old / 'status.json', fixture.status)
    with pytest.raises(ValueError, match='unconfirmed cleanup'):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)


def test_retry_never_accepts_another_checkpoint_or_nonfailed_original(fixture):
    fixture.plan['groups'][1]['checkpoint'] = '/another/checkpoint'
    with pytest.raises(ValueError, match='differs'):
        fixture.retry.old_plan(fixture.plan, fixture.driver)
    fixture.plan['groups'][1]['checkpoint'] = fixture.original['groups'][11]['checkpoint']
    fixture.status['groups'][10]['status'] = 'complete'
    write(fixture.old / 'status.json', fixture.status)
    with pytest.raises(ValueError, match='not failed'):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)


def test_old_running_job_requires_fresh_scheduler_and_worker_identities(fixture):
    fixture.status.update(status='running', stage='parallel_staged_handoff', pid=500,
        identity=dict(pid=500, starttime=20), updated_utc=datetime.now(timezone.utc).isoformat())
    fixture.status['groups'][0].update(status='running', worker_pid=600,
        worker_identity=dict(pid=600, starttime=30))
    owner = dict(pid=500, starttime=20, state='S', argv=['controller', str(fixture.old / 'plan.json')])
    worker = dict(pid=600, starttime=30, state='S', argv=[
        '/original/kaitactilesim/.pixi/envs/default/bin/python',
        '/original/kaitactilesim/scripts/deploy/run_pi05_staged_groups.py',
        '--plan', str(fixture.old / 'plan.json'), '--worker-index', '0'])
    fixture.monkeypatch.setattr(fixture.retry, 'proc', lambda pid: {500: owner, 600: worker}.get(pid))
    write(fixture.old / 'status.json', fixture.status)
    assert fixture.retry.original_snapshot(fixture.plan, fixture.driver)['live_groups'] == 1
    worker['starttime'] = 31
    with pytest.raises(ValueError, match='worker PID'):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)
    worker['starttime'] = 30
    fixture.status['updated_utc'] = '2020-01-01T00:00:00+00:00'
    write(fixture.old / 'status.json', fixture.status)
    with pytest.raises(ValueError, match='stale'):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)


def test_global_cap_accounts_for_old_queued_refill_and_quarantine(fixture):
    retry = fixture.retry
    assert retry.capacity_allows({'reserved_groups': 6}, {}, set())
    assert retry.capacity_allows({'reserved_groups': 6}, {0: object()}, set())
    assert not retry.capacity_allows({'reserved_groups': 7}, {0: object()}, set())
    assert not retry.capacity_allows({'reserved_groups': 6}, {0: object()}, {1})


def test_actual_sampling_loop_observes_twenty_seconds_and_keeps_old_job_readonly(fixture):
    elapsed = [0]
    before = (fixture.old / 'status.json').read_bytes()
    def snapshot():
        sample = samples()[0]
        sample.update(monotonic=elapsed[0], cpu_usage_ns=elapsed[0] * 9 * 10 ** 9)
        return sample
    fixture.monkeypatch.setattr(fixture.retry, 'resource_snapshot', snapshot)
    fixture.monkeypatch.setattr(fixture.retry.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    status = {'groups': []}
    report = fixture.retry.sample_gate(fixture.plan, fixture.driver, status)
    assert report['passed'] is True and report['samples'] == 21 and elapsed[0] == 20
    assert (fixture.old / 'status.json').read_bytes() == before


def test_per_launch_proof_checks_both_changed_evaluator_sources(fixture):
    retry, driver, plan = fixture.retry, fixture.driver, fixture.plan
    directory = Path(plan['simulation_root']) / 'scripts/deploy'
    fixture.job.mkdir(parents=True)
    write(fixture.job / 'plan.json', plan)
    paths = {
        'controller_sha256': Path(retry.__file__),
        'plan_sha256': fixture.job / 'plan.json',
        'frozen_driver_sha256': directory / 'run_pi05_staged_groups.py',
        'frozen_helper_sha256': directory / 'run_pi05_explicit_groups.py',
        'preflight_sha256': Path(plan['simulation_root']).parent / 'preflight_report.json',
        'original_plan_sha256': fixture.old / 'plan.json',
        **{key: Path(plan['simulation_root']) / relative for key, relative in retry.FIX_SOURCES.items()},
    }
    for path in paths.values():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'frozen source fixture')
    record = {'source_proof': {key: driver.sha(path) for key, path in paths.items()}}
    retry.verify_launch_proof(plan, driver, record)
    paths['staged_observer_sha256'].write_bytes(b'changed observer')
    with pytest.raises(ValueError, match='staged_observer_sha256'):
        retry.verify_launch_proof(plan, driver, record)


def warmup_files(fixture, *, stamp=3.401):
    output = fixture.output(fixture.plan, 0)
    server = dict(pid=700, starttime=40, state='S', ppid=600, argv=['python', 'frozen_server', '--port', '18830'])
    row = dict(status='running', worker_pid=600, owned_run_pids={'server_pid': copy.deepcopy(server)})
    write(output / 'run/run_manifest.json', dict(server_pid=700, server={'command': server['argv']}))
    write(output / 'deployment/deployment_manifest.json', dict(checkpoint_sha256='sha', deployment_id='deployment'))
    trace = output / 'run/evaluation/seed_000/staged_states.jsonl'
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps(dict(time_s=stamp, task='poker-draw', version='staged_v1', seed=0,
        prediction_horizon=50, execute_steps=50, checkpoint_sha256='sha', deployment_id='deployment')) + '\n')
    fixture.monkeypatch.setattr(fixture.retry, 'proc', lambda pid: server if pid == 700 else None)
    return row, server, trace


def test_second_launch_needs_real_seed_zero_warmup_not_a_service_port(fixture):
    row, _, trace = warmup_files(fixture, stamp=3.4)
    assert fixture.retry.warmup_ready(fixture.plan, fixture.driver, row) is False
    data = json.loads(trace.read_text())
    data['time_s'] = 3.401
    trace.write_text(json.dumps(data) + '\n{"partial":')
    assert fixture.retry.warmup_ready(fixture.plan, fixture.driver, row) is True
    assert row['warmup_verified'] is True
    assert row['warmup_server_identity']['starttime'] == 40


@pytest.mark.parametrize('bad', ['pid_reuse', 'wrong_parent', 'wrong_command', 'wrong_seed', 'wrong_checkpoint'])
def test_warmup_rejects_stale_or_foreign_processes_and_observations(fixture, bad):
    row, server, trace = warmup_files(fixture)
    if bad == 'pid_reuse':
        server['starttime'] = 41
    elif bad == 'wrong_parent':
        server['ppid'] = 999
    elif bad == 'wrong_command':
        server['argv'] = ['foreign-model-server']
    else:
        data = json.loads(trace.read_text())
        data['seed' if bad == 'wrong_seed' else 'checkpoint_sha256'] = 19 if bad == 'wrong_seed' else 'foreign'
        trace.write_text(json.dumps(data) + '\n')
    assert fixture.retry.warmup_ready(fixture.plan, fixture.driver, row) is False


def test_completed_first_group_authenticates_full_results_before_bypassing_dead_server(fixture):
    row, server, _ = warmup_files(fixture)
    row.update(status='complete', owned_cleanup_confirmed=True, valid_trials=20)
    output = fixture.output(fixture.plan, 0)
    write(output / 'group_status.json', dict(status='complete', valid_trials=20,
                                            owned_cleanup_confirmed=True))
    write(output / 'run/run_manifest.json', dict(server_pid=700, evaluation_pid=701,
        managed_server_stopped=True, evaluation_process_stopped=True))
    fixture.monkeypatch.setattr(fixture.retry, 'proc', lambda pid: None)
    fixture.driver.validate_native = MagicMock(return_value=({}, {}, []))
    assert fixture.retry.warmup_ready(fixture.plan, fixture.driver, row) is True
    fixture.driver.validate_native.assert_called_once_with(output / 'run/evaluation', fixture.plan['groups'][0],
                                                           {'checkpoint_sha256': 'sha', 'deployment_id': 'deployment'})
    assert row['warmup_evidence'] == 'validated_completed_twenty_seed_group'


def test_completed_status_string_without_validated_results_is_not_a_warmup_proof(fixture):
    row, _, _ = warmup_files(fixture)
    row.update(status='complete', owned_cleanup_confirmed=True, valid_trials=20)
    fixture.monkeypatch.setattr(fixture.retry, 'proc', lambda pid: None)
    assert fixture.retry.warmup_ready(fixture.plan, fixture.driver, row) is False


def test_failed_cold_start_with_unknown_evaluator_cleanup_reserves_capacity(fixture):
    output = fixture.output(fixture.plan, 0)
    write(output / 'run/run_manifest.json', dict(server_pid=700, managed_server_stopped=True))
    detail = dict(status='failed', stage='evaluate', owned_cleanup_confirmed=True)
    assert fixture.retry.cleaned(fixture.driver, output, fixture.plan['groups'][0], detail) is False


def test_output_logs_archives_and_claims_are_never_overwritten(fixture):
    retry, driver, plan = fixture.retry, fixture.driver, fixture.plan
    fixture.job.mkdir(parents=True)
    log = fixture.job / 'group_0.log'
    log.write_text('original bytes')
    with pytest.raises(ValueError, match='already exists'):
        retry.guard_fresh(plan, driver, 0)
    assert log.read_text() == 'original bytes'


def controller_fixture(fixture):
    fixture.job.mkdir(parents=True)
    write(fixture.job / 'plan.json', fixture.plan)
    write(fixture.job / 'retry_record.json', {'source_proof': {'proof': 'same'}})
    fixture.monkeypatch.setattr(fixture.retry, 'source_proof', lambda *args: {'proof': 'same'})
    fixture.monkeypatch.setattr(fixture.retry, 'verify_launch_proof', lambda *args: None)
    current = dict(pid=999, starttime=50, state='R', argv=['controller'])
    def process_info(pid):
        if pid in (600, 601):
            return dict(pid=pid, starttime=60, ppid=os.getpid(), state='R', argv=[
                str(Path(fixture.plan['simulation_root']) / '.pixi/envs/default/bin/python'),
                str(Path(fixture.plan['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
                '--plan', str(fixture.job / 'plan.json'), '--worker-index', str(pid - 600)])
        return current
    fixture.monkeypatch.setattr(fixture.retry, 'proc', process_info)
    fixture.monkeypatch.setattr(fixture.retry, 'original_snapshot', lambda *args: dict(
        live_groups=6, queued_groups=2, reserved_groups=6, quarantined_groups=0))
    fixture.monkeypatch.setattr(fixture.retry, 'cleaned', lambda *args: True)
    return current


def test_controller_sequential_warmup_then_two_groups_and_exactly_forty_trials(fixture):
    controller_fixture(fixture)
    tick = [0]
    events = []
    processes = []
    old_bytes = (fixture.old / 'status.json').read_bytes()
    old_plan_bytes = (fixture.old / 'plan.json').read_bytes()
    def sleep(seconds):
        tick[0] += 1
    def gate(*args):
        events.append(('gate', tick[0]))
        return {'passed': True}
    def warmed(plan, driver, row):
        events.append(('warmup', tick[0]))
        return tick[0] >= 2
    def launch(plan, index):
        events.append(('launch', index, tick[0]))
        began = tick[0]
        process = MagicMock(pid=600 + index)
        def poll():
            if tick[0] - began < 4:
                return None
            write(fixture.output(plan, index) / 'group_status.json', dict(
                status='complete', valid_trials=20, owned_cleanup_confirmed=True))
            write(fixture.output(plan, index) / 'run/evaluation/summary.json', dict(
                valid_trials=20, attempted=20, trials=[{'seed': seed} for seed in range(20)]))
            return 0
        process.poll.side_effect = poll
        processes.append(process)
        return process
    fixture.driver.launch_worker = launch
    fixture.monkeypatch.setattr(fixture.retry.time, 'sleep', sleep)
    fixture.monkeypatch.setattr(fixture.retry, 'sample_gate', gate)
    fixture.monkeypatch.setattr(fixture.retry, 'warmup_ready', warmed)
    assert fixture.retry.run_controller(fixture.plan) == 0
    status = json.loads((fixture.job / 'status.json').read_text())
    assert status['status'] == 'complete'
    assert status['valid_trials'] == status['completed_trials'] == status['total_trials'] == 40
    launches = [event for event in events if event[0] == 'launch']
    assert launches == [('launch', 0, 0), ('launch', 1, 2)]
    assert events.index(('warmup', 2)) < events.index(('gate', 2)) < events.index(('launch', 1, 2))
    assert len(list((fixture.job / 'claims').glob('*.json'))) == 2
    for process in processes:
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
    assert (fixture.old / 'status.json').read_bytes() == old_bytes
    assert (fixture.old / 'plan.json').read_bytes() == old_plan_bytes


def test_controller_exception_does_not_kill_a_healthy_worker(fixture):
    controller_fixture(fixture)
    process = MagicMock(pid=600)
    process.poll.return_value = None
    fixture.driver.launch_worker = MagicMock(return_value=process)
    fixture.monkeypatch.setattr(fixture.retry, 'sample_gate', lambda *args: {'passed': True})
    def interrupted(_seconds):
        raise KeyboardInterrupt('test interruption')
    fixture.monkeypatch.setattr(fixture.retry.time, 'sleep', interrupted)
    assert fixture.retry.run_controller(fixture.plan) == 1
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    status = json.loads((fixture.job / 'status.json').read_text())
    assert status['stage'] == 'controller_recovery_required'
    assert (fixture.job / 'recovery_required.json').exists()


def test_failed_clean_peer_does_not_cancel_the_second_checkpoint(fixture):
    controller_fixture(fixture)
    tick = [0]
    launched = []
    processes = []
    def launch(plan, index):
        launched.append(index)
        process = MagicMock(pid=600 + index)
        def poll():
            complete = index == 1
            write(fixture.output(plan, index) / 'group_status.json', dict(
                status='complete' if complete else 'failed', valid_trials=20 if complete else 0,
                owned_cleanup_confirmed=True))
            if complete:
                write(fixture.output(plan, index) / 'run/evaluation/summary.json', dict(valid_trials=20, attempted=20))
            return 0 if complete else 1
        process.poll.side_effect = poll
        processes.append(process)
        return process
    fixture.driver.launch_worker = launch
    fixture.monkeypatch.setattr(fixture.retry, 'sample_gate', lambda *args: {'passed': True})
    fixture.monkeypatch.setattr(fixture.retry.time, 'sleep', lambda _: tick.__setitem__(0, tick[0] + 1))
    assert fixture.retry.run_controller(fixture.plan) == 1
    assert launched == [0, 1]
    final = json.loads((fixture.job / 'status.json').read_text())
    assert [row['status'] for row in final['groups']] == ['failed', 'complete']
    assert final['valid_trials'] == 20
    for process in processes:
        process.terminate.assert_not_called()
        process.kill.assert_not_called()


def test_insufficient_resources_wait_then_retry_without_starting_two_at_once(fixture):
    controller_fixture(fixture)
    ticks = [0]
    gates = []
    process = MagicMock(pid=600)
    process.poll.return_value = None
    fixture.driver.launch_worker = MagicMock(return_value=process)
    def gate(*args):
        gates.append(ticks[0])
        return {'passed': ticks[0] > 0}
    def sleep(_seconds):
        ticks[0] += 1
        if ticks[0] >= 2:
            raise KeyboardInterrupt('finish waiting test')
    fixture.monkeypatch.setattr(fixture.retry.time, 'sleep', sleep)
    fixture.monkeypatch.setattr(fixture.retry, 'sample_gate', gate)
    assert fixture.retry.run_controller(fixture.plan) == 1
    assert gates == [0, 1]
    fixture.driver.launch_worker.assert_called_once_with(fixture.plan, 0)


def test_start_copies_readonly_controller_cache_and_provenance_detached(fixture):
    fixture.monkeypatch.setattr(fixture.retry, 'validate_retry', lambda *args, **kwargs: None)
    fixture.monkeypatch.setattr(fixture.retry, 'source_proof', lambda *args: {'proof': 'same'})
    fixture.monkeypatch.setattr(fixture.retry, 'proc', lambda pid: dict(pid=pid, starttime=50) if pid == 1234 else None)
    process = MagicMock(pid=1234)
    popen = MagicMock(return_value=process)
    fixture.monkeypatch.setattr(fixture.retry.subprocess, 'Popen', popen)
    assert fixture.retry.start(fixture.plan) == 1234
    command = popen.call_args.args[0]
    assert command[0] == '/usr/local/bin/python3.11'
    copied = Path(command[1])
    assert copied.parent == fixture.job / '_scheduler_tools'
    assert not copied.stat().st_mode & 0o222
    assert popen.call_args.kwargs['start_new_session'] is True
    assert (fixture.job / '.cache/openpi/big_vision/tokenizer.model').read_bytes() == b'cached tokenizer'
    assert json.loads((fixture.job / 'stage_standard.json').read_text()) == {'unchanged_threshold': 3}
    before = (fixture.job / 'plan.json').read_bytes()
    with pytest.raises(FileExistsError):
        fixture.retry.start(fixture.plan)
    assert (fixture.job / 'plan.json').read_bytes() == before


def owned_listener_files(fixture, *, published_server_pid=True):
    retry, driver, plan = fixture.retry, fixture.driver, fixture.plan
    proofs = retry.pin_original_cleanup(plan, driver)
    write(fixture.job / 'retry_record.json', {'original_cleanup_proof': proofs})
    controller = dict(pid=500, starttime=20, state='S', ppid=1, argv=['readonly-retry-controller'])
    worker = dict(pid=600, starttime=30, state='S', ppid=500, argv=[
        str(Path(plan['simulation_root']) / '.pixi/envs/default/bin/python'),
        str(Path(plan['simulation_root']) / 'scripts/deploy/run_pi05_staged_groups.py'),
        '--plan', str(fixture.job / 'plan.json'), '--worker-index', '0'])
    server = dict(pid=700, starttime=40, state='S', ppid=600,
                  argv=['model-python', 'frozen-server', '--port', '18830'])
    write(fixture.job / 'controller_identity.json', controller)
    write(fixture.job / 'claims/group_000.json', {'group_index': 0, 'identity': worker})
    output = fixture.output(plan, 0)
    run = dict(status='starting', endpoint='ws://127.0.0.1:18830', deployment_id='deployment',
               server={'command': server['argv']})
    if published_server_pid:
        run['server_pid'] = 700
    write(output / 'run/run_manifest.json', run)
    write(output / 'deployment/deployment_manifest.json', dict(task='poker-draw',
        checkpoint_path=plan['groups'][0]['checkpoint'], deployment_id='deployment'))
    table = {500: controller, 600: worker, 700: server}
    fixture.monkeypatch.setattr(retry, 'proc', lambda pid: table.get(pid))
    fixture.monkeypatch.setattr(retry, 'direct_children', lambda pid: [700] if pid == 600 else [])
    fixture.monkeypatch.setattr(retry, 'listener_inodes', lambda port: {'owned-inode'} if port == 18830 else set())
    fixture.monkeypatch.setattr(retry, 'socket_inodes', lambda pid: {'owned-inode'} if pid == 700 else set())
    driver.port_free.side_effect = lambda port: port != 18830
    return proofs, table, run


@pytest.mark.parametrize('published_server_pid', [True, False])
def test_pinned_original_cleanup_accepts_only_our_reused_listener_even_before_pid_publication(
    fixture, published_server_pid,
):
    owned_listener_files(fixture, published_server_pid=published_server_pid)
    before = (fixture.old / 'status.json').read_bytes()
    scope = fixture.retry.original_snapshot(fixture.plan, fixture.driver)
    assert scope['terminal'] is True
    assert fixture.retry.OWNED_SERVICE_IDENTITIES[(str(fixture.job), 0)]['starttime'] == 40
    assert (fixture.old / 'status.json').read_bytes() == before


@pytest.mark.parametrize('bad', ['external_socket', 'mixed_sockets', 'foreign_child', 'pid_reuse', 'missing_claim', 'missing_run'])
def test_pinned_cleanup_never_treats_an_external_listener_as_retry_owned(fixture, bad):
    _, table, _ = owned_listener_files(fixture, published_server_pid=False)
    retry = fixture.retry
    if bad == 'external_socket':
        fixture.monkeypatch.setattr(retry, 'listener_inodes', lambda port: {'external-inode'})
    elif bad == 'mixed_sockets':
        fixture.monkeypatch.setattr(retry, 'listener_inodes', lambda port: {'owned-inode', 'external-inode'})
    elif bad == 'foreign_child':
        table[700]['ppid'] = 999
    elif bad == 'pid_reuse':
        assert retry.original_snapshot(fixture.plan, fixture.driver)
        table[700]['starttime'] = 41
    elif bad == 'missing_claim':
        (fixture.job / 'claims/group_000.json').unlink()
    else:
        (fixture.output(fixture.plan, 0) / 'run/run_manifest.json').unlink()
    with pytest.raises(ValueError, match='unverified or external listener'):
        retry.original_snapshot(fixture.plan, fixture.driver)


@pytest.mark.parametrize('changed', ['group_status', 'run_manifest', 'old_pid'])
def test_pinned_original_cleanup_requires_immutable_records_and_old_pids_to_stay_exited(fixture, changed):
    _, table, _ = owned_listener_files(fixture)
    output = fixture.output(fixture.original, 10)
    if changed == 'old_pid':
        table[9010] = dict(pid=9010, starttime=100, state='S', argv=['resurrected-old-server'])
        message = 'appeared live'
    else:
        path = output / ('run/run_manifest.json' if changed == 'run_manifest' else 'group_status.json')
        payload = json.loads(path.read_text())
        payload['changed_after_authentication'] = True
        write(path, payload)
        message = 'records changed'
    with pytest.raises(ValueError, match=message):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)


def test_starting_metadata_without_our_actual_socket_is_not_a_port_ownership_proof(fixture):
    owned_listener_files(fixture, published_server_pid=False)
    fixture.monkeypatch.setattr(fixture.retry, 'socket_inodes', lambda pid: set())
    with pytest.raises(ValueError, match='unverified or external listener'):
        fixture.retry.original_snapshot(fixture.plan, fixture.driver)
