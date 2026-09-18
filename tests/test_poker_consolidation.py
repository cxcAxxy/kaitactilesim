"""Tiny synthetic HDF5 tests; no physics, rendering or training data mutation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


@pytest.fixture
def merger(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/workcell'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('poker_consolidation_test_module', scripts / 'consolidate_poker_successes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(tmp_path, merger):
    source = tmp_path / 'source/raw/episode_000008_card_right.h5'
    source.parent.mkdir(parents=True)
    with h5py.File(source, 'w') as f:
        f.attrs['metadata_json'] = json.dumps({'episode_index': 8, 'seed': 1008, 'acceptance_policy': 'strict-force-v1'})
        f.attrs['outcome_json'] = '{"success": true}'
        f.create_dataset('cameras/head/rgb', data=np.arange(300, dtype='uint8').reshape(10, 10, 3), compression='gzip', chunks=(1, 10, 3))
        f.create_dataset('state/qpos', data=np.arange(20).reshape(10, 2))
        f.create_dataset('phase', data=['a', 'b'], dtype=h5py.string_dtype())
        f['state'].attrs['names'] = ['x', 'y']
    sha = merger.digest(source)
    manifest = {'sha256': sha, 'episode': source.name, 'outcome': {'success': True}}
    merger.save(source.with_suffix('.json'), manifest)
    execution = tmp_path / 'source/logs/episode_000008_card_right/execution.json'
    merger.save(execution, {'episode_index': 8, 'episode_seed': 1008, 'completed': True,
                            'argv': ['capture', '--start-index', '8'], 'validation': {'sha256': sha}})
    audit = tmp_path / 'source/cleaning/first4_v2/old/cleaning.json'
    merger.save(audit, {'source_sha256': sha, 'episode_index': 8, 'status': 'review_required', 'errors': [], 'criteria': {'4_state_discontinuities': 'review_required'}})
    row = dict(source=str(source), source_root=str(tmp_path / 'source'), source_batch='source',
               source_episode_index=8, source_local_index=0, episode_index=0, seed=1008,
               source_sha256=sha, cleaning_report=str(audit), execution_report=str(execution), previous_split='val')
    row.update(source_size_bytes=source.stat().st_size, cleaning_status='review_required',
               acceptance_policy='strict-force-v1')
    output = tmp_path / 'merged'
    (output / 'raw').mkdir(parents=True)
    return source, row, output


def test_copy_reindexes_only_root_metadata_and_preserves_seed_and_original(merger, tmp_path, monkeypatch):
    source, row, output = source_fixture(tmp_path, merger)
    monkeypatch.setattr(merger, 'validate_episode', lambda p: SimpleNamespace(valid=True))
    result = merger.prepare_copy(row, output)
    assert merger.digest(source) == row['source_sha256']
    assert result['sha256'] != row['source_sha256']
    target = Path(result['path'])
    with h5py.File(source) as old, h5py.File(target) as new:
        assert merger.stored_stream_fingerprint(old) == merger.stored_stream_fingerprint(new)
        metadata = json.loads(new.attrs['metadata_json'])
        assert metadata['episode_index'] == 0
        assert metadata['seed'] == 1008
        assert metadata['acceptance_policy'] == 'strict-force-v1'
        assert metadata['dataset_provenance']['source_episode_index'] == 8
        assert metadata['dataset_provenance']['previous_split'] == 'val'
    assert json.loads(target.with_suffix('.json').read_text())['sha256'] == merger.digest(target)
    report = json.loads((output / 'logs/episode_000000_card_right/execution.json').read_text())
    assert report['episode_index'] == 0
    assert report['original_episode_index'] == 8
    assert 'argv' not in report
    audit = json.loads((output / 'cleaning/first4_v2/episode_000000_card_right/cleaning.json').read_text())
    assert audit['status'] == 'review_required'


def test_copy_rejects_changed_source_without_removing_it(merger, tmp_path):
    source, row, output = source_fixture(tmp_path, merger)
    with h5py.File(source, 'r+') as f:
        f['state/qpos'][0, 0] = 999
    with pytest.raises(ValueError, match='SHA changed'):
        merger.prepare_copy(row, output)
    assert source.exists()
    assert not list((output / 'raw').iterdir())


def test_fingerprint_detects_vlen_and_group_attribute_changes(merger, tmp_path):
    source, _, _ = source_fixture(tmp_path, merger)
    with h5py.File(source, 'r+') as f:
        first = merger.stored_stream_fingerprint(f)
        f.attrs['metadata_json'] = '{}'
        assert merger.stored_stream_fingerprint(f) == first
        f['phase'][0] = 'different'
        second = merger.stored_stream_fingerprint(f)
        assert second != first
        f['state'].attrs['names'] = ['y', 'x']
        assert merger.stored_stream_fingerprint(f) != second


def test_safe_regular_rejects_links(merger, tmp_path):
    source = tmp_path / 'source'
    source.write_text('keep')
    link = tmp_path / 'link'
    link.symlink_to(source)
    with pytest.raises(ValueError):
        merger.safe_regular(link)
    assert source.read_text() == 'keep'


def test_full_consolidation_keeps_canonical_data_and_relative_source_views(merger, tmp_path, monkeypatch):
    source, row, output = source_fixture(tmp_path, merger)
    (output / 'raw').rmdir()
    output.rmdir()
    root = Path(row['source_root'])
    failed = root / 'raw/episode_000009_card_right.h5.partial'
    failed.write_bytes(b'failed')
    plan = {'sources': [str(root)], 'episodes': [row],
            'failed_raw': [{'path': str(failed), 'bytes': 6}], 'failed_logs': []}
    monkeypatch.setattr(merger, 'validate_episode', lambda p: SimpleNamespace(valid=True))
    merger.apply(plan, output)
    target = output / 'raw/episode_000000_card_right.h5'
    assert target.is_file() and not target.is_symlink()
    assert not source.exists() and not failed.exists()
    view = root / 'raw/episode_000000_card_right.h5'
    assert view.is_symlink() and view.resolve() == target
    assert (root / 'provenance/pre_reindex/raw_manifests/episode_000008_card_right.json').is_file()
    assert (root / 'REORGANIZED.json').exists()
    assert json.loads((output / 'manifest.json').read_text())['episode_count'] == 1
    assert json.loads((output / 'cleaning/first4_v2/summary.json').read_text())['counts']['review_required'] == 1
