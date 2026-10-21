import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluate/evaluate_cpm_poker_checkpoint.py'
spec = importlib.util.spec_from_file_location('cpm_checkpoint_batch', SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_only_requested_checkpoint_ten_seeds(tmp_path):
    plan = batch.make_plan(tmp_path)
    assert plan['seeds'] == list(range(10))
    assert plan['planned_trials'] == 10 and len(plan['groups']) == 1
    assert plan['groups'][0]['model_family'] == 'pi05+cpm'
    assert plan['initial_state_randomization'] is True
    assert plan['automatic_retries'] == 0
    assert plan['frozen_checkpoint_path'] != plan['source_checkpoint_path']


def test_command_has_native_diagnostic_protocol(tmp_path):
    plan = batch.make_plan(tmp_path)
    command = batch.trial_command(plan, 9, 'a' * 64)
    for option in ('--save-raw', '--record', '--expected-deployment-manifest-sha256'):
        assert option in command
    assert command[command.index('--seed') + 1] == '9'
    assert command[command.index('--action-interpolation') + 1] == 'zoh'
    assert command[command.index('--staged-evaluation-version') + 1] == 'staged_v1'
    with pytest.raises(ValueError, match='Unrequested'):
        batch.trial_command(plan, 10, 'a' * 64)


@pytest.mark.parametrize('edit,code,timeout,expected', [
    ({}, 0, False, True), ({'seed': 2}, 0, False, False),
    ({'deployment_id': 'other'}, 0, False, False),
    ({'formal_metrics_valid': True}, 0, False, False),
    ({'physics_valid': False}, 0, False, False),
    ({'video_record_error': 'bad'}, 0, False, False),
    ({}, 1, False, False), ({}, 0, True, False),
])
def test_native_validity_preserved(tmp_path, edit, code, timeout, expected):
    manifest = dict(task='poker-draw', model_family='pi05+cpm', prediction_horizon=30,
        action_start_index=0, execute_steps=5, deployment_id='frozen')
    summary = dict(manifest, seed=0, deployment_manifest_sha256='hash', valid_trial=True,
        trial_validity='valid', physics_valid=True, task_success=False,
        diagnostic_only=True, formal_metrics_valid=False, status='task_not_completed')
    summary.update(edit)
    folder = tmp_path / 'seed_0000'
    folder.mkdir()
    (folder / 'summary.json').write_text(json.dumps(summary))
    row = batch.trial_row(0, tmp_path, manifest, 'hash', code, timed_out=timeout)
    assert row['valid'] is expected
    assert row['task_success'] is (False if expected else None)


def test_missing_is_not_zero_percent_failure(tmp_path):
    manifest = dict(task='poker-draw', model_family='pi05+cpm', prediction_horizon=30,
        action_start_index=0, execute_steps=5, deployment_id='frozen')
    row = batch.trial_row(0, tmp_path, manifest, 'hash', 1)
    assert row['valid'] is False and row['task_success'] is None
