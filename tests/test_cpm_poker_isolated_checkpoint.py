import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluate/evaluate_cpm_poker_isolated_checkpoint.py'
spec = importlib.util.spec_from_file_location('cpm_isolated', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_copies_independent_bytes_not_links(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'frozen'
    (source / 'src').mkdir(parents=True)
    path = source / 'src/model.py'
    path.write_text('old')
    result = module.freeze_tree(source, target, ['src'])
    path.write_text('new')
    assert (target / 'src/model.py').read_text() == 'old'
    assert result['source_rehashed_after_copy'] is True
    assert result['frozen_rehashed'] is True
    assert path.stat().st_ino != (target / 'src/model.py').stat().st_ino
    assert (target / 'src/model.py').stat().st_mode & 0o222 == 0


def test_symlinks_cannot_reintroduce_source_drift(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'source.py').write_text('source')
    (source / 'linked.py').symlink_to(source / 'source.py')
    with pytest.raises(ValueError, match='symlinks'):
        module.freeze_tree(source, tmp_path / 'frozen', ['linked.py'])


def test_source_drift_rejected_before_inference(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    path = source / 'model.py'
    path.write_text('one')
    original = module.shutil.copyfile
    def changed(src, dst):
        result = original(src, dst)
        path.write_text('two')
        return result
    monkeypatch.setattr(module.shutil, 'copyfile', changed)
    with pytest.raises(ValueError, match='changed during snapshot'):
        module.freeze_tree(source, tmp_path / 'frozen', ['model.py'])


def test_existing_snapshot_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match='existing'):
        module.freeze_tree(tmp_path, tmp_path, ['model.py'])
