#!/usr/bin/env python3
"""Wait for the isolated CPM n10 job, audit a fresh snapshot, then publish it.

This controller never starts/retries a simulator or model. Failed gates are
terminal, preserve all receipts, and leave the previous shared portal intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS = Path('/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations')
JOB = EVALUATIONS / '_jobs/cpm_poker_best_n10_20261021_r2'
CONTROL = EVALUATIONS / '_jobs/cpm_poker_best_n10_publication_20261021_r3'
BASELINE = ROOT / 'reports/kaitactilesim_unified_workbench_20261020_r5'
SNAPSHOT = ROOT / 'reports/kaitactilesim_unified_workbench_20261021_cpm_n10_r2'
PYTHON = ROOT / '.pixi/envs/default/bin/python'
BASELINE_CATALOG_SHA = '362d072f7401319bb59bb66a736ac8256b42e17d1d65da83c1c146e3b98a92af'
BASELINE_MANIFEST_SHA = 'de7ea18423235b4247aa03de1ac21db0770ba3303603ae6f1dc9bcc65d899ec4'
OLD_ROOT_SHA = 'd600837f760eaf4888eb2de08e044c37f9311b238b216663490ce5273b2d49fd'
URL = 'http://127.0.0.1:8877/' + SNAPSHOT.name + '/index.html'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected regular non-symlink input: {path}')
    before = path.stat()
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'Input changed while hashing: {path}')
    return value


def save(path, value):
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    pending.replace(path)


def source_files():
    return sorted(str(path) for directory in (ROOT / 'scripts/analysis', ROOT / 'src')
        for path in directory.rglob('*') if path.is_file()
        and '__pycache__' not in path.parts and path.suffix in {'.py', '.js', '.cjs', '.css'})


def process_identity(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in {'Z', 'X', 'x'}:
            return None
        return dict(pid=pid, start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def make_plan(control):
    if control != CONTROL or control.exists() or control.is_symlink():
        raise ValueError('Use the exact fresh CPM publication controller directory')
    if SNAPSHOT.exists() or SNAPSHOT.is_symlink():
        raise ValueError('CPM platform output must be fresh; no overwrite or automatic retry')
    checks = {BASELINE / 'catalog.json': BASELINE_CATALOG_SHA,
              BASELINE / 'artifact_manifest.json': BASELINE_MANIFEST_SHA,
              ROOT / 'reports/index.html': OLD_ROOT_SHA}
    if any(digest(path) != expected for path, expected in checks.items()):
        raise ValueError('Accepted baseline or current shared portal changed')
    producer = load(JOB / 'status.json')
    if producer.get('state') not in {'running', 'complete'}:
        raise ValueError('The isolated CPM producer is neither running nor complete')
    identity = process_identity(producer.get('controller_pid'))
    if producer['state'] != 'complete' and identity is None:
        raise ValueError('No live producer identity; no adoption/retry')
    analysis = ROOT / 'scripts/analysis'
    raw = [str(PYTHON), '-B', str(analysis / 'check_cpm_poker_evaluation.py'),
        '--job-root', str(JOB), '--deployment-manifest', str(JOB / 'deployments/cpm_best/deployment_manifest.json'),
        '--evaluation-root', str(JOB / 'groups/cpm_best/attempt_001/evaluation'), '--baseline', str(BASELINE)]
    export = [str(PYTHON), '-B', str(analysis / 'export_cpm_additive_evaluation_workbench.py'), '--output', str(SNAPSHOT)]
    files = source_files()
    plan_files = [JOB / 'plan.json', JOB / 'runtime_snapshot.json', JOB / 'checkpoint_snapshot.json',
                  JOB / 'deployments/cpm_best/deployment_manifest.json']
    plan = dict(schema='kaihand-cpm-poker-publication-controller-v1', created_utc=stamp(),
        job_root=str(JOB), control_dir=str(control), baseline=str(BASELINE), snapshot=str(SNAPSHOT),
        producer_identity=identity, wait_deadline_unix=time.time() + 5 * 3600,
        simulation_runs_started=0, automatic_retries=0, source_inventory=files,
        source_sha256={path: digest(path) for path in files},
        input_sha256={str(path): digest(path) for path in [*checks, *plan_files]},
        environment=dict(TMPDIR=str(control / 'tmp'), MPLCONFIGDIR=str(control / 'cache/mpl'),
            NODE_PATH='/tmp/contact_web_qa.CJzQS7/node_modules',
            LD_LIBRARY_PATH='/tmp/contact_web_qa.CJzQS7/runtime-libs/usr/lib/x86_64-linux-gnu',
            KAI_EVAL_CHROME='/tmp/contact_web_qa.CJzQS7/browsers/chromium-1248/chrome-linux64/chrome',
            PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2'),
        commands=[
            dict(stage='raw_qa', argv=[*raw, '--receipt', str(JOB / 'qa/raw_evidence_r2.json')]),
            dict(stage='export', argv=[*export, '--baseline', str(BASELINE), '--job-root', str(JOB),
                '--evaluation-root', str(EVALUATIONS), '--qa-receipt', str(JOB / 'qa/raw_evidence_r2.json'), '--workers', '2']),
            dict(stage='snapshot_qa', argv=[*raw, '--snapshot', str(SNAPSHOT),
                '--receipt', str(JOB / 'qa/platform_evidence_r2.json')]),
            dict(stage='browser_qa', argv=['node', str(analysis / 'check_cpm_additive_evaluation_workbench_browser.cjs'),
                URL, str(SNAPSHOT / 'browser_qa'), '--baseline=http://127.0.0.1:8877/' + BASELINE.name + '/index.html']),
            dict(stage='verify', argv=[*export, '--verify-only']),
            dict(stage='publish', argv=[*export, '--publish-only']),
        ])
    return plan


def guards(plan, *, portal=True):
    if source_files() != plan['source_inventory']:
        raise ValueError('Publication implementation file membership changed')
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen publication implementation changed: ' + path)
    for path, expected in plan['input_sha256'].items():
        if not portal and path == str(ROOT / 'reports/index.html'):
            continue
        if digest(path) != expected:
            raise ValueError('Frozen publication input changed: ' + path)


def run(control):
    plan = load(control / 'plan.json')
    state = dict(schema=plan['schema'], state='waiting', stage=None, started_utc=stamp(),
                 controller_pid=os.getpid(), simulation_runs_started=0, automatic_retries=0)
    def update():
        state['updated_utc'] = stamp()
        save(control / 'status.json', state)
    update()
    try:
        while True:
            guards(plan)
            status = load(JOB / 'status.json')
            state['producer_state'] = status.get('state')
            state['producer_groups'] = status.get('groups')
            update()
            if status.get('state') == 'complete':
                break
            if status.get('state') != 'running':
                raise ValueError('CPM producer did not complete: ' + str(status.get('state')))
            if time.time() >= plan['wait_deadline_unix']:
                raise ValueError('Five-hour publication wait deadline reached; no automatic retry')
            if process_identity(status.get('controller_pid')) != plan['producer_identity']:
                raise ValueError('CPM producer disappeared or changed process identity')
            time.sleep(20)
        env = dict(os.environ, **plan['environment'])
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key, None)
        env.update(NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        for command in plan['commands']:
            guards(plan)
            state.update(state='running', stage=command['stage'])
            update()
            with (control / 'logs' / (command['stage'] + '.log')).open('x') as log:
                result = subprocess.run(command['argv'], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                raise ValueError(f"Publication gate failed: {command['stage']} (exit {result.returncode})")
            guards(plan, portal=command['stage'] != 'publish')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:8877/', timeout=15) as response:
            landing = response.read().decode()
        if SNAPSHOT.name not in landing:
            raise ValueError('Published shared HTTP entry does not point to CPM snapshot')
        with opener.open(URL.replace('index.html', 'catalog.json'), timeout=15) as response:
            catalog = json.load(response)
        if catalog['counts']['members'] != 605 or len(catalog['groups']) != 48:
            raise ValueError('Published HTTP catalog is not the expected 605/48 snapshot')
        state.update(state='complete', stage='published', snapshot_url=URL,
            members=605, groups=48, ended_utc=stamp())
        update()
        return 0
    except BaseException as error:
        state.update(state='failed', error=str(error), ended_utc=stamp())
        update()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    if args.status:
        print(json.dumps(load(CONTROL / 'status.json'), ensure_ascii=False, indent=2))
        return 0
    if args.run:
        return run(CONTROL)
    if not args.start:
        parser.error('Choose --start, --run, or --status')
    plan = make_plan(CONTROL)
    CONTROL.mkdir()
    for name in ('tmp', 'cache/mpl', 'logs'):
        (CONTROL / name).mkdir(parents=True)
    save(CONTROL / 'plan.json', plan)
    with (CONTROL / 'logs/controller.log').open('x') as log:
        process = subprocess.Popen([str(PYTHON), '-B', str(Path(__file__).resolve()), '--run'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(state='started', controller_pid=process.pid, control_dir=str(CONTROL))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
