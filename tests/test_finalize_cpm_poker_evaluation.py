import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analysis/finalize_cpm_poker_evaluation.py'
spec = importlib.util.spec_from_file_location('cpm_publication', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_drift_blocks_publication(tmp_path, monkeypatch):
    source = tmp_path / 'implementation.py'
    source.write_text('one')
    monkeypatch.setattr(module, 'source_files', lambda: [str(source)])
    plan = dict(source_inventory=[str(source)], source_sha256={str(source): module.digest(source)}, input_sha256={})
    module.guards(plan)
    source.write_text('two')
    with pytest.raises(ValueError, match='implementation changed'):
        module.guards(plan)


def test_source_membership_drift_is_not_ignored(monkeypatch):
    monkeypatch.setattr(module, 'source_files', lambda: ['new.py'])
    with pytest.raises(ValueError, match='membership changed'):
        module.guards(dict(source_inventory=[], source_sha256={}, input_sha256={}))


def test_bounded_commands_cannot_start_models(tmp_path, monkeypatch):
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'status.json').write_text(json.dumps(dict(state='complete', controller_pid=None)))
    control = tmp_path / 'control'
    for key, value in dict(ROOT=tmp_path, JOB=job, CONTROL=control,
                           BASELINE=tmp_path / 'reports/baseline', SNAPSHOT=tmp_path / 'reports/fresh',
                           BASELINE_CATALOG_SHA='hash', BASELINE_MANIFEST_SHA='hash', OLD_ROOT_SHA='hash').items():
        monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(module, 'digest', lambda path: 'hash')
    monkeypatch.setattr(module, 'source_files', lambda: [])
    plan = module.make_plan(control)
    assert plan['simulation_runs_started'] == plan['automatic_retries'] == 0
    assert [row['stage'] for row in plan['commands']] == ['raw_qa', 'export', 'snapshot_qa', 'browser_qa', 'verify', 'publish']
    assert plan['commands'][-1]['argv'][-1] == '--publish-only'
    assert all(not any('run_poker_' in value or 'evaluate_cpm_' in value for value in row['argv'])
               for row in plan['commands'])
    assert 4.9 * 3600 <= plan['wait_deadline_unix'] - module.time.time() <= 5 * 3600


def test_failed_producer_keeps_previous_portal(tmp_path, monkeypatch):
    control, job = tmp_path / 'control', tmp_path / 'job'
    control.mkdir()
    job.mkdir()
    portal = tmp_path / 'index.html'
    portal.write_text('old portal')
    plan = dict(schema='test', source_inventory=[], source_sha256={}, input_sha256={})
    (control / 'plan.json').write_text(json.dumps(plan))
    (job / 'status.json').write_text(json.dumps(dict(state='interrupted')))
    monkeypatch.setattr(module, 'JOB', job)
    monkeypatch.setattr(module, 'source_files', lambda: [])
    with pytest.raises(ValueError, match='did not complete'):
        module.run(control)
    status = json.loads((control / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert status['automatic_retries'] == 0
    assert portal.read_text() == 'old portal'


def test_pid_guard_does_not_adopt_invalid_pid():
    assert module.process_identity(-1) is None
    assert module.process_identity(None) is None
