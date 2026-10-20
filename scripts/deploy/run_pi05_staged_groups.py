#!/usr/bin/env python3
"""Detached, explicitly authorized staged_v1 H50/K50 evaluation queue.

Existing canonical groups, legacy summaries, frozen bundles and global catalogs
are never overwritten. Each publication is a new sibling run-version directory.
Only this controller's direct Popen workers are eligible for shutdown.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import run_pi05_explicit_groups as base

sys.dont_write_bytecode = True
VERSION = 'staged_v1'
ARCHIVE_ROOT = base.ARCHIVE_ROOT
JOBS_ROOT = ARCHIVE_ROOT / '_jobs'
SNAPSHOT_ROOT = base.SNAPSHOT_ROOT
OBSERVER_SOURCE = 'src/kaihand_tactile_env/shared/staged_evaluation.py'
TASKS = ('bulb-screw', 'bulb-unscrew', 'install-ram', 'poker-draw', 'usb-insert',
         'usb-unplug', 'whiteboard-wipe', 'whiteboard-wipe-horizontal',
         'whiteboard-wipe-horizontal-line')
BOARDS = frozenset(task for task in TASKS if task.startswith('whiteboard-'))
STAGE_NAMES = {
    'bulb-screw': ('picked_up', 'held_near_socket', 'threaded_and_seated'),
    'bulb-unscrew': ('fully_extracted', 'released_stably_on_table'),
    'install-ram': ('picked_up_and_held_near_slot', 'inserted_and_seated'),
    'usb-insert': ('picked_up', 'held_near_socket', 'inserted_and_seated'),
    'usb-unplug': ('fully_extracted', 'released_stably_on_table'),
    'poker-draw': ('dragged_on_table', 'stable_grasp', 'terminal_hold_over_3s'),
    **{task: ('picked_up',) for task in BOARDS},
}
BOARD_NUMERIC_METRICS = ('cleaned_fraction', 'ink_union_area_weighted_cleaned_fraction',
    'effective_pad_contact_area_ratio', 'cumulative_ink_coverage_ratio',
    'time_weighted_mean_effective_area_ratio', 'full_pad_bottom_area_m2', 'initial_ink_union_area_m2')
SETTINGS = {**base.SETTINGS, 'staged_evaluation_version': VERSION}
require, safe, read, write, sha = base.require, base.safe, base.read, base.write, base.sha
now, process_info, port_free = base.now, base.process_info, base.port_free


def validate_plan(plan, *, existing_job=False, allow_published=False):
    require(isinstance(plan, dict), 'Staged plan must be a JSON object')
    run_id = plan.get('run_id')
    require(isinstance(run_id, str) and re.fullmatch(r'staged_v1_[A-Za-z0-9_-]+', run_id), 'Invalid staged_v1 run_id')
    job = safe(plan['job_root'])
    require(job.parent == JOBS_ROOT and job.name == 'pi05_' + run_id, 'Wrong staged CPFS job scope')
    require(job.is_dir() if existing_job else not job.exists(), 'Staged job existence/collision mismatch')
    simulation, openpi = map(safe, (plan['simulation_root'], plan['openpi_root']))
    require(simulation.name == 'kaitactilesim' and openpi.name == 'openpi'
            and simulation.parent == openpi.parent and simulation.parent.parent == SNAPSHOT_ROOT,
            'Staged runtime roots must be sibling frozen copies')
    settings = plan.get('settings')
    require(isinstance(settings, dict) and settings.keys() == SETTINGS.keys(), 'Staged settings schema mismatch')
    require(type(settings['parallel_groups']) is int and 1 <= settings['parallel_groups'] <= 4,
            'Staged parallel checkpoint cap must be 1..4')
    require(all(settings[key] == value for key, value in SETTINGS.items() if key != 'parallel_groups'),
            'Staged evaluation requires H50/K50/20/90/observe/version settings')
    authorized = plan.get('authorized_tasks')
    require(isinstance(authorized, list) and authorized and len(set(authorized)) == len(authorized)
            and set(authorized) <= set(TASKS), 'Explicit authorized_tasks must be a unique supported task subset')
    groups = plan.get('groups')
    require(isinstance(groups, list) and len(groups) == 2 * len(authorized)
            and {(g['task'], g['step']) for g in groups}
            == {(task, step) for task in authorized for step in (20000, 24999)},
            'Each explicitly authorized task must have exactly both selected checkpoints')
    port_start = plan.get('port_start', 18820)
    require(type(port_start) is int and (port_start == 18820 or (
            port_start == 18830 and authorized == ['poker-draw'])),
            'Only default staged ports or the explicit Poker retry pair are permitted')
    require([g['port'] for g in groups] == list(range(port_start, port_start + len(groups))),
            'Staged groups require unique consecutive ports in plan order')
    for group in groups:
        require(isinstance(group['dataset'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', group['dataset']),
                'Unsafe training dataset identifier')
        checkpoint = safe(group['checkpoint'])
        parts = checkpoint.parts
        require('pi05' in parts, 'Checkpoint must use pi05 dataset tree')
        position = parts.index('pi05')
        source_task = 'usb_insert' if group['task'] == 'usb-insert' else group['task']
        require(parts[position - 1] == source_task and parts[position + 1] == group['dataset']
                and parts[position + 2] == 'checkpoints' and checkpoint.name == str(group['step']),
                'Staged checkpoint task/dataset/step mismatch')
        archive = safe(group['archive_dir'])
        require(archive == ARCHIVE_ROOT / group['task'] / 'pi05'
                / f"step{group['step']}_H50K50_{group['dataset']}_{run_id}",
                'Staged archive must be a new run-version sibling, not an old canonical directory')
        if not allow_published:
            require(not archive.exists(), f'Staged archive already exists: {archive}')
    return plan


def verify_frozen(plan):
    simulation, openpi = Path(plan['simulation_root']), Path(plan['openpi_root'])
    for root in (simulation / 'src', simulation / 'scripts', simulation / 'configs',
                 openpi / 'src', openpi / 'scripts', openpi / 'packages'):
        require(root.is_dir() and not root.is_symlink(), 'Missing physical frozen sources')
        for current, directories, names in os.walk(root, followlinks=False):
            for path in [Path(current), *(Path(current) / name for name in directories + names)]:
                require(not path.is_symlink() and not path.stat().st_mode & 0o222,
                        f'Staged evaluation cannot import writable/linked source: {path}')
    require((simulation / OBSERVER_SOURCE).is_file(), 'Frozen staged observer module missing')
    for interpreter in (base.MODEL_PYTHON, base.CONTROLLER_PYTHON,
                        simulation / '.pixi/envs/default/bin/python'):
        require(interpreter.is_file() and os.access(interpreter, os.X_OK), 'Staged runtime Python missing')
    for group in plan['groups']:
        for name in ('_CHECKPOINT_METADATA', 'params/_METADATA'):
            require((Path(group['checkpoint']) / name).is_file(), 'Staged checkpoint is not committed')
    report = read(simulation.parent / 'preflight_report.json')
    rows = report.get('checkpoints', [])
    require(report.get('all_selected_valid') is True and report.get('count') == len(plan['groups'])
            and report.get('staged_evaluation_version') == VERSION
            and report.get('simulation_root') == str(simulation) and report.get('model_project_root') == str(openpi)
            and [(r.get('task'), r.get('step'), r.get('checkpoint')) for r in rows]
            == [(g['task'], g['step'], g['checkpoint']) for g in plan['groups']]
            and all(r.get('valid') is True and r.get('prediction_horizon') == r.get('execute_steps') == 50
                    and r.get('binding_status') == 'bound' and r.get('required_action_interpolation') == 'linear'
                    for r in rows), 'Exact staged frozen-source CPU preflight is required; legacy reports are not accepted')


def output_dir(plan, index):
    return base.output_dir(plan, index)


def start(plan):
    validate_plan(plan)
    verify_frozen(plan)
    require(base.TOKENIZER_SOURCE.is_file(), 'Cached tokenizer missing')
    require(all(port_free(g['port']) for g in plan['groups']), 'A staged local service port is occupied')
    job = Path(plan['job_root'])
    job.mkdir(parents=True, exist_ok=False)
    cache = job / '.cache/openpi/big_vision'
    cache.mkdir(parents=True)
    shutil.copy2(base.TOKENIZER_SOURCE, cache / base.TOKENIZER_SOURCE.name)
    write(job / 'plan.json', plan)
    write(job / 'status.json', dict(status='starting', stage='launching', staged_evaluation_version=VERSION,
                                  started_utc=now(), total_trials=20 * len(plan['groups'])))
    for name in ('preflight_report.json', 'snapshot_record.json'):
        path = Path(plan['simulation_root']).parent / name
        if path.is_file():
            shutil.copy2(path, job / name)
    with (job / 'controller.log').open('xb', buffering=0) as stream:
        process = subprocess.Popen([str(base.CONTROLLER_PYTHON),
            str(Path(plan['simulation_root']) / 'scripts/deploy' / Path(__file__).name),
            '--plan', str(job / 'plan.json')], cwd=plan['simulation_root'], env=base.environment(plan),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(job / 'controller_identity.json', process_info(process.pid) or {'pid': process.pid})
    print(json.dumps(dict(pid=process.pid, job_root=str(job), status=str(job / 'status.json'),
                          staged_evaluation_version=VERSION), indent=2), flush=True)
    return process.pid


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def finite_json(value):
    """Allow geometry metadata/unknown values, but no non-finite measurements."""
    if value is None or type(value) in (str, bool):
        return True
    if type(value) in (float, int):
        return finite_number(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_json(item) for key, item in value.items())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return False


def validate_report(report, task):
    require(isinstance(report, dict) and report.get('version') == VERSION and report.get('task') == task,
            'Staged observer report version/task missing or mismatched')
    count = report.get('stage_count')
    require(type(count) is int and count == len(STAGE_NAMES[task]), 'Staged observer stage_count mismatch')
    keys = {f'stage{number}' for number in range(1, count + 1)}
    stages = report.get('stages')
    definitions = report.get('stage_definitions')
    require(isinstance(stages, dict) and set(stages) == keys and all(type(v) is bool for v in stages.values())
            and report.get('stage_flags') == stages and isinstance(definitions, dict)
            and definitions == {f'stage{i + 1}': name for i, name in enumerate(STAGE_NAMES[task])},
            'Staged observer flags/definitions are malformed')
    require(report.get('full_duration_evaluation') is True
            and report.get('native_task_success_preserved') is True
            and report.get('expert_trajectory_used') is False
            and report.get('simulation_state_mutated') is False,
            'Staged observer must be full-duration, passive, and preserve legacy task outcomes')
    complete = report.get('complete_success')
    require(complete is None if task in BOARDS else type(complete) is bool,
            'Whiteboard has continuous outcome, object complete_success must be boolean')
    require(task in BOARDS or complete == stages[f'stage{count}'],
            'Object complete_success disagrees with the observed final stage')
    require(finite_number(report.get('time_s')) and report['time_s'] >= 0,
            'Staged observer simulation time is not finite')
    require(type(report.get('observations')) is int and report['observations'] > 0,
            'Staged observer has no real observations')
    continuous = report.get('continuous_metrics')
    require(isinstance(continuous, dict) and finite_json(continuous),
            'Continuous staged geometry metadata contains a non-finite measurement')
    if task in BOARDS:
        require(all(finite_number(continuous.get(key)) and continuous[key] >= 0
                    for key in BOARD_NUMERIC_METRICS)
                and continuous['full_pad_bottom_area_m2'] > 0 and continuous['initial_ink_union_area_m2'] > 0,
                'Whiteboard continuous outcome measurements/positive area denominators missing')
    return report


def aggregate_reports(reports, task):
    require(reports, 'Cannot aggregate absent staged observations')
    for report in reports:
        validate_report(report, task)
    definitions = reports[0]['stage_definitions']
    require(all(report['stage_definitions'] == definitions for report in reports),
            'Stage definitions changed across seeds')
    count = len(reports)
    stages = {name: dict(definition=definitions[name], count=sum(r['stages'][name] for r in reports),
                         rate=sum(r['stages'][name] for r in reports) / count)
              for name in definitions}
    metric_keys = set().union(*(r['continuous_metrics'] for r in reports))
    continuous = {}
    for key in sorted(metric_keys):
        values = [r['continuous_metrics'].get(key) for r in reports]
        numbers = [value for value in values if finite_number(value)]
        if numbers or any(value is None for value in values):
            continuous[key] = dict(mean=sum(numbers) / len(numbers) if numbers else None, count=len(numbers))
    complete_count = None if task in BOARDS else sum(r['complete_success'] for r in reports)
    return dict(version=VERSION, task=task, trials=count, stages=stages,
                complete_success_count=complete_count,
                complete_success_rate=None if complete_count is None else complete_count / count,
                continuous_metrics=continuous, diagnostic_only=True, formal_metrics_valid=False,
                metric_scope='staged diagnostic measurements; not a formal task-success rate')


def validate_states(path, group, seed, deployment):
    require(path.is_file() and not path.is_symlink(), 'Full staged_states.jsonl observation record missing')
    count, first_time, last_time = 0, None, None
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank staged observation record')
            payload = json.loads(line)
            require(payload.get('version') == VERSION and payload.get('task') == group['task']
                    and payload.get('seed') == seed and payload.get('prediction_horizon') == 50
                    and payload.get('execute_steps') == 50
                    and payload.get('checkpoint_sha256') == deployment['checkpoint_sha256'],
                    'Staged observation identity/version mismatch')
            stamp = payload.get('time_s')
            require(finite_number(stamp) and stamp >= 0
                    and (last_time is None or last_time <= stamp <= last_time + 0.1),
                    'Staged observation time invalid or 30Hz coverage has a gap')
            first_time = stamp if first_time is None else first_time
            last_time, count = stamp, count + 1
    require(count >= 2698 and first_time is not None and first_time <= 0.1 and last_time >= 89.93,
            'Staged observation record does not cover the full 90 simulated seconds at 30Hz')
    return dict(rows=count, first_time_s=first_time, last_time_s=last_time)


def validate_native(evaluation, group, deployment):
    summary = base.validate_native(evaluation, group, deployment)
    require(isinstance(summary.get('staged_evaluation'), dict)
            and summary['staged_evaluation'].get('version') == VERSION,
            'Legacy-only batch summary cannot be staged_v1')
    reports, observations = [], []
    for row in summary.get('results', summary.get('trials')):
        seed = row['seed']
        directory = safe(evaluation / f'seed_{seed:03d}')
        detail = read(directory / 'summary.json')
        report = validate_report(detail.get('staged_evaluation'), group['task'])
        require(finite_number(detail.get('sim_seconds')) and detail['sim_seconds'] >= 89.93,
                'Staged rollout stopped early instead of preserving full-duration evaluation')
        require(read(directory / 'staged_evaluation.json') == report,
                'Staged report sidecar and original seed summary disagree')
        observations.append(validate_states(directory / 'staged_states.jsonl', group, seed, deployment))
        reports.append(report)
    aggregate = aggregate_reports(reports, group['task'])
    return summary, aggregate, observations


def publish_archive(plan, index, deployment):
    group = plan['groups'][index]
    source = safe(output_dir(plan, index) / 'run/evaluation')
    destination = safe(group['archive_dir'])
    require(not destination.exists(), 'Staged archive collision; original canonical data is never replaced')
    summary, aggregate, observations = validate_native(source, group, deployment)
    files = []
    for current, directories, names in os.walk(source, followlinks=False):
        for name in directories:
            require(not (Path(current) / name).is_symlink(), 'Symlink in staged evaluation directory')
        for name in names:
            path = Path(current) / name
            files.append(dict(relative_path=str(path.relative_to(source)), size=path.stat().st_size, sha256=sha(path)))
    manifest = dict(schema='kaihand-staged-evaluation-archive-v1', staged_evaluation_version=VERSION,
        run_id=plan['run_id'], task=group['task'], model='pi05', step=group['step'], dataset=group['dataset'],
        checkpoint=group['checkpoint'], checkpoint_sha256=deployment['checkpoint_sha256'],
        deployment_id=deployment['deployment_id'], training_provenance=deployment.get('training_provenance'),
        model_config_name=deployment.get('model_config_name'), prediction_horizon=50, execute_steps=50,
        valid_trials=20, diagnostic_only=True, formal_metrics_valid=False, full_duration_evaluation=True,
        staged_aggregate=aggregate, observation_coverage=observations, files=files,
        summary_sha256=sha(source / 'summary.json'), original_metadata_unchanged=True,
        job_root=plan['job_root'], original_evaluation_dir=str(source), archive_dir=str(destination),
        frozen_simulation_root=plan['simulation_root'], frozen_openpi_root=plan['openpi_root'], created_utc=now())
    require(not (source / 'archive_manifest.json').exists(), 'Staged archive manifest already exists')
    write(source / 'archive_manifest.json', manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(source.stat().st_dev == destination.parent.stat().st_dev, 'Staged archive must stay on same filesystem')
    base.rename_noreplace(source, destination)
    try:
        os.symlink(str(destination), source, target_is_directory=True)
        require(source.resolve(strict=True) == destination, 'Staged evaluation compatibility link failed')
    except BaseException:
        if source.is_symlink() and os.readlink(source) == str(destination):
            source.unlink()
        if not source.exists() and not source.is_symlink():
            base.rename_noreplace(destination, source)
        raise
    return dict(archive_dir=str(destination), completed_trials=20, valid_trials=20,
                staged_aggregate=aggregate, diagnostic_successes=summary.get('successes', 0))


def run_worker(plan, index):
    validate_plan(plan, existing_job=True, allow_published=True)
    verify_frozen(plan)
    require(0 <= index < len(plan['groups']), 'Staged worker index outside explicit plan')
    group = plan['groups'][index]
    require(not safe(group['archive_dir']).exists(), 'This staged group was already published')
    output = output_dir(plan, index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    status = dict(status='running', stage='preflight', staged_evaluation_version=VERSION,
        pid=os.getpid(), identity=process_info(os.getpid()), task=group['task'], checkpoint_step=group['step'],
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
        api = base.load_api(plan)
        module, profile = api.deployment, api.profile
        directory = output / 'deployment'
        manifest_path = directory / 'deployment_manifest.json'
        update(stage='prepare')
        prepare = module.build_prepare_plan(group['task'], 'pi05', checkpoint=Path(group['checkpoint']),
                                            output_dir=directory, profile=profile)
        require(module.execute_prepare(prepare) == 0, 'Staged deployment prepare failed')
        update(stage='verify')
        verification = module.verify_deployment(manifest_path, profile=profile)
        module.write_pipeline_deployment(directory, task=group['task'], model_family='pi05',
            prepare_plan=prepare, profile=profile, verification=verification)
        deployment = read(manifest_path)
        require(deployment.get('task') == group['task'] and deployment.get('checkpoint_path') == group['checkpoint']
                and deployment.get('prediction_horizon') == 50, 'Staged prepared checkpoint/task/H50 mismatch')
        observer_rows = [row for row in deployment.get('evaluation_source_files', []) if row.get('path') == OBSERVER_SOURCE]
        require(len(observer_rows) == 1 and observer_rows[0].get('sha256')
                == sha(Path(plan['simulation_root']) / OBSERVER_SOURCE),
                'Staged observer is not authenticated by the deployment runtime source closure')
        update(stage='evaluate', deployment_id=deployment['deployment_id'], run_manifest=str(output / 'run/run_manifest.json'))
        record = module.run_deployment(manifest_path, output_dir=output / 'run', profile=profile,
            service_mode='managed', endpoint=f"ws://127.0.0.1:{group['port']}", num_trials=20, video_count=20,
            seed_start=0, execute_steps=50, max_sim_seconds=90, guard_mode='observe', server_timeout=900,
            evaluator_args=('--staged-evaluation-version', VERSION))
        require(record.get('evaluation_returncode') == 0 and record.get('managed_server_stopped') is True
                and record.get('evaluation_process_stopped') is True and port_free(group['port']),
                'Staged owned evaluator/service cleanup failed')
        update(stage='publishing', managed_server_stopped=True, evaluation_process_stopped=True)
        published = publish_archive(plan, index, deployment)
        update(status='complete', stage='complete', owned_cleanup_confirmed=True, finished_utc=now(), **published)
        return 0
    except BaseException as error:
        path = output / 'run/run_manifest.json'
        if path.is_file():
            record = read(path)
        clean = (record is None and status['stage'] in ('preflight', 'prepare', 'verify')) or (
            record is not None and record.get('managed_server_stopped') is True and record.get('evaluation_process_stopped') is True)
        status['errors'].append(f'{type(error).__name__}: {error}')
        update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               owned_cleanup_confirmed=bool(clean and port_free(group['port'])), finished_utc=now())
        return 1
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


def launch_worker(plan, index):
    job, simulation = Path(plan['job_root']), Path(plan['simulation_root'])
    with (job / f'group_{index}.log').open('xb', buffering=0) as stream:
        return subprocess.Popen([str(simulation / '.pixi/envs/default/bin/python'),
            str(simulation / 'scripts/deploy' / Path(__file__).name), '--plan', str(job / 'plan.json'),
            '--worker-index', str(index)], cwd=simulation, env=base.environment(plan), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def read_ready(path):
    """Native progress writers may not yet have atomically published a JSON."""
    try:
        return read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def refresh(plan, status):
    for index, group in enumerate(status['groups']):
        output = output_dir(plan, index)
        path = output / 'group_status.json'
        if path.is_file():
            detail = read_ready(path)
            if detail is not None:
                group.update(worker_status=detail, stage=detail.get('stage'))
                if 'staged_aggregate' in detail:
                    group['staged_aggregate'] = detail['staged_aggregate']
        summary_path = output / 'run/evaluation/summary.json'
        if summary_path.is_file():
            summary = read_ready(summary_path)
            if summary is not None:
                rows = summary.get('results', summary.get('trials', []))
                group.update(completed_trials=len(rows), valid_trials=summary.get('valid_trials', 0),
                             completed_seeds=[row['seed'] for row in rows if row.get('valid_trial') is True],
                             staged_evaluation=summary.get('staged_evaluation'))
        run_path = output / 'run/run_manifest.json'
        if run_path.is_file():
            run = read_ready(run_path)
            if run is not None:
                group['owned_run_pids'] = {key: process_info(run[key]) for key in ('server_pid', 'evaluation_pid')
                                          if isinstance(run.get(key), int)}
    status.update(updated_utc=now(), completed_trials=sum(g.get('completed_trials', 0) for g in status['groups']),
                  valid_trials=sum(g.get('valid_trials', 0) for g in status['groups']),
                  active_groups=sum(g['status'] == 'running' for g in status['groups']),
                  completed_checkpoint_count=sum(g['status'] == 'complete' for g in status['groups']))
    write(Path(plan['job_root']) / 'status.json', status)


def run_controller(plan):
    validate_plan(plan, existing_job=True)
    verify_frozen(plan)
    job = Path(plan['job_root'])
    status = dict(status='running', stage='parallel_staged_evaluations', staged_evaluation_version=VERSION,
        run_id=plan['run_id'], pid=os.getpid(), identity=process_info(os.getpid()), started_utc=now(),
        total_trials=20 * len(plan['groups']), total_checkpoint_count=len(plan['groups']),
        authorized_tasks=plan['authorized_tasks'], settings=plan['settings'], errors=[],
        groups=[{**group, 'group_index': index, 'status': 'queued', 'output_dir': str(output_dir(plan, index))}
                for index, group in enumerate(plan['groups'])])
    workers, quarantined_slots = {}, set()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    prior = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (job / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while workers or any(g['status'] == 'queued' for g in status['groups']):
                for index, process in list(workers.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    group = status['groups'][index]
                    path = output_dir(plan, index) / 'group_status.json'
                    detail = (read_ready(path) or {}) if path.is_file() else {}
                    clean = detail.get('owned_cleanup_confirmed') is True and port_free(group['port'])
                    complete = code == 0 and detail.get('status') == 'complete' and clean and detail.get('valid_trials') == 20
                    group.update(status='complete' if complete else 'failed', worker_status=detail,
                                 worker_returncode=code, owned_cleanup_confirmed=clean, finished_utc=now())
                    del workers[index]
                    if not complete:
                        status['errors'].append(f"{group['task']} step{group['step']}: worker exit {code}")
                    # A failed group cannot cancel healthy peers. An unconfirmed
                    # cleanup reserves its capacity rather than reusing it.
                    if not clean:
                        quarantined_slots.add(index)
                        group['stage'] = 'owned_cleanup_unconfirmed'
                free = max(0, plan['settings']['parallel_groups'] - len(workers) - len(quarantined_slots))
                queued = [g for g in status['groups'] if g['status'] == 'queued']
                for group in queued[:free]:
                    index = group['group_index']
                    try:
                        require(port_free(group['port']), 'Staged selected port is occupied')
                        process = launch_worker(plan, index)
                        workers[index] = process
                        group.update(status='running', worker_pid=process.pid, worker_identity=process_info(process.pid))
                    except Exception as error:
                        group.update(status='failed', stage='launch_failed', error=str(error))
                        status['errors'].append(f"{group['task']} step{group['step']}: launch failed: {error}")
                refresh(plan, status)
                if workers:
                    time.sleep(10)
                elif free == 0 and queued:
                    for group in queued:
                        group.update(status='failed', stage='not_launched_owned_cleanup_unconfirmed')
                    status['errors'].append('Remaining groups not launched: all service capacity is quarantined')
            status.update(status='failed' if status['errors'] else 'complete', stage='complete', finished_utc=now())
            refresh(plan, status)
            return int(bool(status['errors']))
    except BaseException as error:
        for process in workers.values():
            if process.poll() is None:
                process.terminate()
        for process in workers.values():
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                status['errors'].append(f'Owned staged worker {process.pid} cleanup timed out')
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
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    require(not (args.start and args.worker_index is not None), '--start cannot select a worker')
    plan = read(args.plan)
    if args.start:
        start(plan)
        return 0
    return run_worker(plan, args.worker_index) if args.worker_index is not None else run_controller(plan)


if __name__ == '__main__':
    raise SystemExit(main())
