#!/usr/bin/env python3
"""Consolidate audited poker successes with explicit provenance and bounded I/O.

Plan only by default. --apply creates a standalone canonical dataset, verifies
every HDF5 stream before retiring its source, then replaces source raw/log views
with relative links. Source acquisition logs and Cleaning evidence are archived,
not rewritten as if collection used the new indices. Failed raw data is removed
only after every successful episode has been transferred and verified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import h5py
import numpy as np

from collect_poker_batch import wrapper_lock
from kaihand_tactile_env.shared.recording import validate_episode
from kaihand_tactile_env.shared.egosteer_archive import validate_poker_outcome
from poker_cleaning_checks import digest
from check_poker_cleaning import summarize

PATTERN = re.compile(r'episode_(\d+)_card_right(?:\..+)?$')


def name(index):
    return f'episode_{index:06d}_card_right'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def safe_regular(path):
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(f'Expected regular non-symlink file: {path}')


def stored_stream_fingerprint(file):
    """Hash all datasets plus group/dataset attrs, excluding ONLY root attrs.

    Hash stored chunks for fixed-size datasets (including compressed RGB);
    hash decoded values for variable-length strings so heap pointers are not
    mistaken for their contents. Copying HDF5 preserves the chunk layout.
    """
    result = hashlib.sha256()

    def update(value):
        a = np.asarray(value)
        result.update(str(a.shape).encode())
        result.update(str(a.dtype).encode())
        if a.dtype.hasobject or a.dtype.kind == 'U':
            for item in a.reshape(-1):
                if isinstance(item, np.ndarray):
                    update(item)
                else:
                    data = item if isinstance(item, bytes) else str(item).encode()
                    result.update(len(data).to_bytes(8, 'little'))
                    result.update(data)
        else:
            result.update(a.tobytes())

    def visit(key, obj):
        result.update(key.encode())
        result.update(type(obj).__name__.encode())
        for attr in sorted(obj.attrs):
            result.update(attr.encode())
            update(obj.attrs[attr])
        if not isinstance(obj, h5py.Dataset):
            return
        result.update(str((obj.shape, obj.dtype, obj.chunks, obj.compression,
                           obj.compression_opts, obj.shuffle, obj.fletcher32)).encode())
        if obj.chunks and not obj.dtype.hasobject:
            for i in range(obj.id.get_num_chunks()):
                info = obj.id.get_chunk_info(i)
                mask, raw = obj.id.read_direct_chunk(info.chunk_offset)
                result.update(str((info.chunk_offset, mask)).encode())
                result.update(raw)
        elif obj.ndim and obj.shape[0]:
            for start in range(0, obj.shape[0], 64):
                update(obj[start:start + 64])
        else:
            update(obj[()])

    file.visititems(visit)
    return result.hexdigest()


def inspect_inventory(root, expected, first_index):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Invalid source root: {root}')
    if (root / 'REORGANIZED.json').exists():
        raise ValueError(f'Source already reorganized: {root}')
    reports = {}
    successes = []
    for p in sorted((root / 'logs').glob('episode_*/execution.json')):
        safe_regular(p)
        r = json.loads(p.read_text())
        index = int(r['episode_index'])
        if p.parent.name != name(index) or index in reports:
            raise ValueError(f'Inconsistent acquisition identity: {p}')
        reports[index] = r
        if r.get('completed') is True:
            successes.append((index, p, r))
    if len(successes) != expected:
        raise ValueError(f'{root}: expected {expected} successes, found {len(successes)}')
    cleaning_root = root / 'cleaning/first4_merge_20260911'
    if not cleaning_root.exists():
        cleaning_root = root / 'cleaning/first4_v2'
    summary_path = cleaning_root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['inspected_episodes'] != expected or summary['counts']['fail'] or summary['counts']['error']:
        raise ValueError(f'Incomplete or hard-failing Cleaning: {summary_path}')
    audits = {}
    for p in cleaning_root.glob('episode_*/cleaning.json'):
        r = json.loads(p.read_text())
        audits[(r['episode_index'], r.get('source_sha256'))] = (p, r)
    old_splits = {}
    old_selection = root / 'processing/dual_format_20260909/source_selection.json'
    if old_selection.exists():
        old_splits = {r['episode_index']: r['split'] for r in json.loads(old_selection.read_text())['episodes']}
    rows = []
    for offset, (index, execution_path, execution) in enumerate(successes):
        path = root / 'raw' / (name(index) + '.h5')
        sidecar = path.with_suffix('.json')
        safe_regular(path)
        safe_regular(sidecar)
        manifest = json.loads(sidecar.read_text())
        sha = manifest['sha256']
        if (manifest.get('episode') != path.name or manifest.get('outcome', {}).get('success') is not True
                or sha != execution.get('validation', {}).get('sha256')):
            raise ValueError(f'Invalid success manifest: {path}')
        audit_path, audit = audits[(index, sha)]
        if (audit.get('errors') or not audit.get('audit_complete') or not audit.get('identity_valid')
                or audit.get('status') not in ('review_required', 'pass_with_scope_limit')):
            raise ValueError(f'Invalid Cleaning report: {audit_path}')
        with h5py.File(path, 'r') as file:
            metadata = json.loads(file.attrs['metadata_json'])
            outcome = json.loads(file.attrs['outcome_json'])
            validate_poker_outcome(metadata, outcome)
            if metadata['episode_index'] != index or metadata['seed'] != execution['episode_seed']:
                raise ValueError(f'HDF5 index/seed mismatch: {path}')
        rows.append(dict(source=str(path), source_root=str(root), source_batch=root.name,
                         source_episode_index=index, episode_index=first_index + offset,
                         source_local_index=offset, seed=execution['episode_seed'], source_sha256=sha,
                         source_size_bytes=path.stat().st_size, cleaning_report=str(audit_path),
                         cleaning_status=audit['status'], execution_report=str(execution_path),
                         acceptance_policy=metadata.get('acceptance_policy', 'strict-force-v1'),
                         previous_split=old_splits.get(index)))
    success_indices = {r['source_episode_index'] for r in rows}
    failed_raw = []
    for p in sorted((root / 'raw').iterdir()):
        safe_regular(p)
        match = PATTERN.fullmatch(p.name)
        if match is None:
            raise ValueError(f'Unknown raw entry: {p}')
        index = int(match.group(1))
        if index in success_indices:
            if p.name not in (name(index) + '.h5', name(index) + '.json'):
                raise ValueError(f'Unexpected extra file for success: {p}')
        else:
            if index not in reports or reports[index].get('completed') is not False:
                raise ValueError(f'Unclassified raw file, refusing delete: {p}')
            if p.suffix == '.h5':
                raise ValueError(f'Closed but unaccepted HDF5 needs separate review: {p}')
            failed_raw.append(dict(path=str(p), bytes=p.stat().st_size, reason='recorded_failed_attempt'))
    # Previously identified interrupted episode: inspect only named episode files.
    trash = root / '.trash'
    if trash.exists():
        for p in sorted(trash.rglob('*')):
            if p.is_dir() and not p.is_symlink():
                continue
            safe_regular(p)
            match = PATTERN.fullmatch(p.name)
            if (not match or int(match.group(1)) in success_indices
                    or not p.name.endswith(('.h5.partial', '.h5.lock'))):
                raise ValueError(f'Unknown trash entry, refusing delete: {p}')
            failed_raw.append(dict(path=str(p), bytes=p.stat().st_size, reason='previously_quarantined_interruption'))
    failed_logs = []
    for p in sorted((root / 'logs').glob('episode_*')):
        match = PATTERN.fullmatch(p.name)
        if not match or p.is_symlink() or not p.is_dir():
            raise ValueError(f'Unsafe acquisition log: {p}')
        index = int(match.group(1))
        if index not in success_indices:
            files = []
            for child in sorted(p.rglob('*')):
                if child.is_dir() and not child.is_symlink():
                    continue
                safe_regular(child)
                files.append(dict(path=str(child), bytes=child.stat().st_size))
            failed_logs.append(dict(path=str(p), episode_index=index, files=files,
                                    execution=reports.get(index), reason='failed_or_interrupted_acquisition'))
    return rows, failed_raw, failed_logs


def prepare_copy(row, destination):
    """Keep original until the copied data and all metadata are verified."""
    source = Path(row['source'])
    if digest(source) != row['source_sha256']:
        raise ValueError(f'Source SHA changed: {source}')
    target = destination / 'raw' / (name(row['episode_index']) + '.h5')
    partial = target.with_suffix('.h5.partial')
    if target.exists() or partial.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, partial)
    if digest(partial) != row['source_sha256']:
        raise ValueError(f'Copy SHA mismatch: {partial}')
    with h5py.File(source, 'r') as file:
        stream_sha = stored_stream_fingerprint(file)
        original_attrs = {k: file.attrs[k] for k in file.attrs}
    metadata = json.loads(original_attrs['metadata_json'])
    metadata['episode_index'] = row['episode_index']
    metadata['dataset_provenance'] = {
        k: row[k] for k in ('source_batch', 'source_episode_index', 'source_sha256',
                            'source_local_index', 'seed', 'previous_split')}
    metadata['dataset_provenance']['operation'] = 'success_selection_and_contiguous_reindex_v1'
    with h5py.File(partial, 'r+') as file:
        file.attrs['metadata_json'] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        file.flush()
    with h5py.File(partial, 'r') as file:
        if stored_stream_fingerprint(file) != stream_sha:
            raise ValueError(f'Dataset payload changed: {partial}')
        if set(file.attrs) != set(original_attrs):
            raise ValueError('Root attribute set changed unexpectedly')
        for key in original_attrs:
            if key != 'metadata_json' and not np.array_equal(file.attrs[key], original_attrs[key]):
                raise ValueError(f'Unexpected root attribute change: {key}')
        if json.loads(file.attrs['metadata_json']) != metadata:
            raise ValueError('Reindex metadata mismatch')
    valid = validate_episode(partial)
    if not valid.valid:
        raise ValueError(f'Reindexed schema invalid: {valid.errors}')
    sha = digest(partial)
    os.replace(partial, target)
    manifest = json.loads(source.with_suffix('.json').read_text())
    manifest.update(episode=target.name, sha256=sha,
                    reindex_provenance=metadata['dataset_provenance'])
    save(target.with_suffix('.json'), manifest)
    report = copy.deepcopy(json.loads(Path(row['execution_report']).read_text()))
    report.update(episode_index=row['episode_index'], record_type='derived_success_reindex_not_new_execution',
                  original_episode_index=row['source_episode_index'], original_source_sha256=row['source_sha256'])
    # Historical argv stays in provenance; do not present it as this dataset's command.
    report['original_argv'] = report.pop('argv', [])
    report['validation']['sha256'] = sha
    report['validation']['all_dataset_payloads_and_nonroot_attributes_unchanged'] = True
    save(destination / 'logs' / target.stem / 'execution.json', report)
    audit = copy.deepcopy(json.loads(Path(row['cleaning_report']).read_text()))
    audit.update(source=str(target), episode_index=row['episode_index'], source_sha256=sha,
                 source_modified_by_audit=False, reindex_verification={
                     'original_source_sha256': row['source_sha256'],
                     'original_episode_index': row['source_episode_index'],
                     'all_dataset_streams_and_attributes_unchanged': True,
                     'stream_fingerprint': stream_sha,
                     'only_root_metadata_episode_index_and_provenance_changed': True,
                     'scope': 'Original first4 result transferred after exact stream verification; not a new simulation or removal of review warnings.'})
    audit_dir = destination / 'cleaning/first4_v2' / target.stem
    save(audit_dir / 'cleaning.json', audit)
    intervals = Path(row['cleaning_report']).parent / 'state_action_intervals.csv'
    if intervals.exists():
        shutil.copyfile(intervals, audit_dir / intervals.name)
    return {**row, 'path': str(target), 'sha256': sha,
            'stream_fingerprint': stream_sha, 'all_streams_unchanged': True}


def archive_source_views(root, records, destination):
    archive = root / 'provenance/pre_reindex'
    archive.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'cleaning'):
        if (root / folder).exists():
            (root / folder).rename(archive / folder)
    # Move small source sidecars and surviving source logs, not duplicate HDF5.
    raw = root / 'raw'
    for row in records:
        original_sidecar = Path(row['source']).with_suffix('.json')
        if original_sidecar.exists():
            saved = archive / 'raw_manifests' / original_sidecar.name
            saved.parent.mkdir(exist_ok=True)
            original_sidecar.rename(saved)
    if list(raw.iterdir()):
        raise RuntimeError(f'Raw contains unclassified files: {raw}')
    for row in records:
        stem = name(row['episode_index'])
        for suffix in ('.h5', '.json'):
            p = raw / (stem + suffix)
            p.symlink_to(os.path.relpath(destination / 'raw' / p.name, raw))
        log = root / 'logs' / stem
        log.parent.mkdir(exist_ok=True)
        log.symlink_to(os.path.relpath(destination / 'logs' / stem, log.parent), target_is_directory=True)
        audit = root / 'cleaning/first4_v2' / stem
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.symlink_to(os.path.relpath(destination / 'cleaning/first4_v2' / stem, audit.parent), target_is_directory=True)
    save(root / 'REORGANIZED.json', {
        'read_only_source_view': True, 'canonical_dataset': str(destination),
        'episode_count': len(records), 'indices': [r['episode_index'] for r in records],
        'source_to_new_mapping': records,
        'historical_logs_and_cleaning': str(archive),
        'do_not_resume_capture_here': True,
        'historical_processing_files_keep_original_indices_and_hashes': True})


def apply(plan, destination):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'raw').mkdir()
    save(destination / 'merge_plan.json', plan)
    records = []
    started = time.monotonic()
    with (destination / 'operations.jsonl').open('x') as journal:
        for row in plan['episodes']:
            if shutil.disk_usage(destination).free < max(2 * 1024**3, row['source_size_bytes'] * 3):
                raise RuntimeError('Insufficient temporary disk space; originals retained for unprocessed episodes')
            record = prepare_copy(row, destination)
            records.append(record)
            journal.write(json.dumps({'verified_copy': record}) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            # Retire only this exact, SHA-verified success after durable copy publication.
            with open(record['path'], 'rb') as stream:
                os.fsync(stream.fileno())
            for durable_path in (destination / 'raw', destination / 'logs' / name(row['episode_index']),
                                 destination / 'cleaning/first4_v2' / name(row['episode_index'])):
                fd = os.open(durable_path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            if digest(Path(row['source'])) != row['source_sha256']:
                raise RuntimeError('Original changed before retirement')
            Path(row['source']).unlink()
            journal.write(json.dumps({'retired_original': row['source']}) + '\n')
            journal.flush()
            print(f"verified {len(records)}/{len(plan['episodes'])}: {row['source_batch']} "
                  f"{row['source_episode_index']} -> {row['episode_index']} ({time.monotonic()-started:.1f}s)", flush=True)
        # Failure deletion is delayed until every successful payload is safe.
        deletions = list(plan['failed_raw'])
        for folder in plan['failed_logs']:
            deletions.extend(folder['files'])
        for item in deletions:
            path = Path(item['path'])
            safe_regular(path)
            if path.stat().st_size != item['bytes']:
                raise RuntimeError(f'Failure file changed: {path}')
        save(destination / 'deleted_failures.json', {'permanent': True, 'files': deletions,
             'bytes': sum(x['bytes'] for x in deletions), 'attempts': plan['failed_logs']})
        for item in deletions:
            Path(item['path']).unlink()
        for folder in plan['failed_logs']:
            path = Path(folder['path'])
            for child in sorted(path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if child.is_dir():
                    child.rmdir()
            path.rmdir()
        for root in map(Path, plan['sources']):
            trash = root / '.trash'
            if trash.exists():
                for folder in sorted(trash.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                    if folder.is_dir():
                        folder.rmdir()
                trash.rmdir()
            archive_source_views(root, [r for r in records if r['source_root'] == str(root)], destination)
    all_audits = [json.loads((destination / 'cleaning/first4_v2' / name(r['episode_index']) / 'cleaning.json').read_text()) for r in records]
    summarize(destination, destination / 'cleaning/first4_v2', all_audits, False, True)
    for root in map(Path, plan['sources']):
        subset = [a for r, a in zip(records, all_audits, strict=True) if r['source_root'] == str(root)]
        summarize(root, root / 'cleaning/first4_v2', subset, False, True)
    save(destination / 'manifest.json', {'schema_version': 'poker-merged-successes-v1',
         'completed': True, 'episodes': records, 'episode_count': len(records),
         'all_payloads_verified_unchanged': True, 'failed_episodes_included': False,
         'source_layout': 'canonical physical data here; source batches contain relative links',
         'acceptance_policy_counts': dict(Counter(r['acceptance_policy'] for r in records)),
         'cleaning_status_counts': dict(Counter(r['cleaning_status'] for r in records)),
         'new_simulations': 0, 'wall_seconds': time.monotonic() - started})
    save(destination / 'REORGANIZED.json', {'canonical_success_dataset': True,
         'episode_count': len(records), 'do_not_resume_capture_here': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path)
    parser.add_argument('--expected-counts', nargs='+', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        if args.apply:
            parser.error('verify-only cannot be combined with apply')
        verify_consolidated(args.output_dir.resolve())
        return
    if not args.sources or not args.expected_counts:
        parser.error('sources and expected-counts are required for consolidation')
    if len(args.sources) != len(args.expected_counts) or any(n <= 0 for n in args.expected_counts):
        parser.error('one positive expected count is required for each source')
    roots = [p.resolve() for p in args.sources]
    destination = args.output_dir.resolve()
    if len(set(roots)) != len(roots) or destination.exists() or any(
            destination.is_relative_to(r) or r.is_relative_to(destination) for r in roots):
        parser.error('sources must be distinct; output must be new and outside source roots')
    with wrapper_lock():
        rows, failed_raw, failed_logs = [], [], []
        for root, expected in zip(roots, args.expected_counts, strict=True):
            r, f, logs = inspect_inventory(root, expected, len(rows))
            rows.extend(r)
            failed_raw.extend(f)
            failed_logs.extend(logs)
        plan = {'sources': list(map(str, roots)), 'destination': str(destination),
                'episodes': rows, 'failed_raw': failed_raw, 'failed_logs': failed_logs,
                'storage_mode': 'move_verified_successes_then_relative_source_views'}
        print(json.dumps({'successes': len(rows), 'source_counts': args.expected_counts,
              'failed_raw_files': len(failed_raw), 'failed_log_directories': len(failed_logs),
              'failed_raw_bytes': sum(f['bytes'] for f in failed_raw),
              'output': str(destination), 'apply': args.apply}), flush=True)
        if args.apply:
            apply(plan, destination)


def verify_consolidated(destination):
    """Independent post-operation audit, one closed HDF5 file at a time."""
    manifest = json.loads((destination / 'manifest.json').read_text())
    rows = manifest['episodes']
    if not manifest.get('completed') or [r['episode_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Incomplete/noncontiguous manifest')
    if len({(r['source_batch'], r['source_episode_index']) for r in rows}) != len(rows):
        raise ValueError('A source episode was included more than once')
    if len({r['stream_fingerprint'] for r in rows}) != len(rows):
        raise ValueError('Duplicate complete episode payloads need review')
    expected_files = {name(r['episode_index']) + suffix for r in rows for suffix in ('.h5', '.json')}
    if {p.name for p in (destination / 'raw').iterdir()} != expected_files:
        raise ValueError('Canonical raw inventory mismatch')
    sources = Counter()
    frames = states = 0
    for i, row in enumerate(rows):
        p = destination / 'raw' / (name(i) + '.h5')
        safe_regular(p)
        sidecar = json.loads(p.with_suffix('.json').read_text())
        if digest(p) != row['sha256'] or sidecar['sha256'] != row['sha256'] or sidecar['episode'] != p.name:
            raise ValueError(f'Final SHA/name mismatch: {p}')
        with h5py.File(p, 'r') as file:
            m = json.loads(file.attrs['metadata_json'])
            outcome = json.loads(file.attrs['outcome_json'])
            if m['episode_index'] != i or m['seed'] != row['seed'] or outcome != sidecar['outcome'] or outcome.get('success') is not True:
                raise ValueError(f'Final metadata mismatch: {p}')
            if stored_stream_fingerprint(file) != row['stream_fingerprint']:
                raise ValueError(f'Final stream fingerprint mismatch: {p}')
            frames += len(file['cameras/head/rgb'])
            states += len(file['state/qpos'])
        execution = json.loads((destination / 'logs' / p.stem / 'execution.json').read_text())
        audit = json.loads((destination / 'cleaning/first4_v2' / p.stem / 'cleaning.json').read_text())
        if (not execution['completed'] or execution['episode_index'] != i
                or execution['validation']['sha256'] != row['sha256'] or audit['source_sha256'] != row['sha256']
                or audit.get('errors') or audit['status'] != row['cleaning_status']):
            raise ValueError(f'Final logs/Cleaning identity mismatch: {p}')
        root = Path(row['source_root'])
        if (root / 'raw' / p.name).resolve() != p:
            raise ValueError(f'Source view link mismatch: {root}')
        sources[str(root)] += 1
        if (i + 1) % 25 == 0:
            print(f'final validation {i+1}/{len(rows)}', flush=True)
    for source_root, count in sources.items():
        raw = Path(source_root) / 'raw'
        if len(list(raw.glob('*.h5'))) != count or len(list(raw.iterdir())) != count * 2:
            raise ValueError(f'Source view count mismatch: {raw}')
    result = {'valid': True, 'episode_count': len(rows), 'contiguous_indices': [0, len(rows)-1],
              'source_counts': dict(sources), 'all_files_sha256_verified': True,
              'all_stream_fingerprints_match_pre_reindex': True,
              'source_view_links_verified': True, 'all_task_outcomes_success': True,
              'duplicate_source_episodes': 0, 'duplicate_complete_streams': 0,
              'head_rgb_frames': frames, 'state_samples': states,
              'cleaning_status_counts': dict(Counter(r['cleaning_status'] for r in rows)),
              'manual_motion_review_cleared': False, 'new_simulations': 0}
    save(destination / 'final_verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
