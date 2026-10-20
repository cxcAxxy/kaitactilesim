"""New-directory-only export; stable code/reports are inputs, never edit targets."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from evaluation_v2.suite import implementation_manifest as stable_implementation

from . import SCHEMA_VERSION
from .adapters import load_policy_episode, load_reference_episode
from .metrics import analyze_episode, summarize_population


def digest(path):
    path = Path(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'Input changed while hashing: {path}')
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'size': before.st_size}


def read_json(path):
    path = Path(path)
    before = digest(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    if digest(path) != before:
        raise ValueError(f'Input changed during read: {path}')
    return value


def verify_rows(rows):
    for row in rows:
        actual = digest(Path(row['path']))
        if actual['sha256'] != row['sha256']:
            raise ValueError(f'Protected source changed: {row["path"]}')


def protected_snapshot(site):
    analysis_root = Path(__file__).resolve().parents[1]
    files = set()
    for directory in (analysis_root / 'evaluation_v1', analysis_root / 'evaluation_v2', site):
        for path in directory.rglob('*'):
            if '__pycache__' not in path.parts and path.is_file():
                if path.is_symlink():
                    raise ValueError(f'Stable snapshot contains a symlink: {path}')
                files.add(path)
    files.update(analysis_root / name for name in (
        'evaluate_trials.py', 'evaluate_suite.py', 'export_expert_reference.py', 'serve_evaluation.py'))
    return [digest(path) for path in sorted(files)]


def implementation():
    package = Path(__file__).parent
    paths = sorted(package.glob('*.py')) + sorted((package / 'web').glob('*'))
    paths += [package.parent / 'compare_task_progress.py']
    return [digest(path) for path in paths if path.is_file()]


def _json_write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def _copy_media(episode, report_path, index):
    """Policy media only; never guess an expert video-to-recording mapping."""
    media = deepcopy(episode.get('media') or {})
    if episode['role'] != 'policy' or media.get('status') != 'available':
        return {'status': 'missing', 'reason': 'No verified local video mapping supplied'}, None
    raw = media.get('url')
    if not isinstance(raw, str) or ':' in raw or '\\' in raw:
        return {'status': 'missing', 'reason': 'No safe local media URL'}, None
    relative = Path(raw)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe source media path')
    source = (report_path.parent / relative).resolve(strict=True)
    if not source.is_relative_to(report_path.parent.resolve()):
        raise ValueError('Media must remain inside the immutable source report directory')
    row = digest(source)
    if media.get('sha256') != row['sha256']:
        raise ValueError('Media SHA does not match its archived descriptor')
    media['url'] = f'media/{index:03d}.mp4'
    return media, (source, media['url'], row)


def export_experiment(stable_site, output, reference_reports=()):
    site = Path(stable_site).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if output.parent != site.parent:
        raise ValueError('Experimental output must be a new sibling of the stable site, preserving its rollback link')
    protected = protected_snapshot(site)
    source_implementation = stable_implementation()
    current_implementation = implementation()
    batch = read_json(site / 'batch.json')
    if batch.get('implementation') != source_implementation:
        raise ValueError('Stable export source fingerprint differs from current V1/V2; do not silently replace the rollback baseline')
    if batch.get('horizon_selection', {}).get('required_prediction_horizon') != 50:
        raise ValueError('This experiment requires the explicit H50-only stable cohort')
    selected, skipped = [], []
    for entry in batch['trials']:
        if entry['task'] != 'whiteboard-wipe':
            skipped.append({'trial_id': entry['trial_id'], 'task': entry['task'], 'reason': 'outside_explicit_whiteboard_pilot'})
            continue
        if entry.get('prediction_horizon') != 50:
            raise ValueError('Non-H50 entry in H50 stable cohort')
        relative = Path(entry['report_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe archived report path')
        report_path = (site / relative).with_suffix('.json').resolve(strict=True)
        if not report_path.is_relative_to(site):
            raise ValueError('Source report escaped stable site')
        selected.append((load_policy_episode(report_path), report_path))
    if not selected:
        raise ValueError('No whiteboard-wipe policy trials in the supplied stable site')
    for reference in reference_reports:
        path = Path(reference).resolve(strict=True)
        selected.append((load_reference_episode(path), path))
    ids = [episode['id'] for episode, _ in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate episodes must not inflate the population')
    episodes, videos, provenance = [], [], []
    for index, (episode, report_path) in enumerate(selected):
        analyzed = analyze_episode(episode)
        media, copy = _copy_media(episode, report_path, index)
        if copy:
            videos.append(copy)
        analyzed.update(metadata=episode.get('metadata', {}), warnings=episode.get('warnings', []),
                        media=media, source_report=str(report_path),
                        signal_descriptors={key: {k: v for k, v in descriptor.items() if k != 'values'}
                                            for key, descriptor in episode['signals'].items()})
        episodes.append(analyzed)
        provenance.extend(episode.get('provenance', []))
    baseline_url = f'../{quote(site.name, safe="")}/index.html'
    rollback = {
        'schema_version': 'evaluation-experiment-rollback-v1', 'stable_site': str(site),
        'stable_url': baseline_url, 'protected_files': protected,
        'strategy': 'Additive opt-in experiment. No old source, report, default entry, simulator, or running job is changed.',
        'return_instructions': 'Open stable_url / stable_site/index.html. Existing evaluate_suite.py remains the stable exporter. No file restoration, git reset, or deletion is necessary.',
        'verification_command': f'.pixi/envs/default/bin/python -B scripts/analysis/compare_task_progress.py --verify-rollback {output / "rollback.json"}',
    }
    data = {
        'schema_version': SCHEMA_VERSION, 'experimental': True, 'baseline_url': baseline_url,
        'selection': {'input_policy_trials': batch['trial_count'], 'selected_policy_trials': len(episodes) - len(reference_reports),
                      'scope': 'whiteboard-wipe only; every selected success/failure/unknown is retained', 'skipped_tasks': skipped},
        'episodes': episodes, 'population': summarize_population(episodes),
        'comparison': {'status': 'case_reference_only', 'quantitative_comparison': 'unverified',
                       'reasons': ['Expert and policy sensor cadence and physical sampling phase are not equivalent or independently calibrated.',
                                   'Initial ink distribution, geometry, contact posture and dynamics are not certified as matched.',
                                   'One selected scripted reference is not an expert population or an unbiased expert success-rate estimate.'],
                       'no_score': True, 'no_time_warp': True, 'no_causal_claim': True},
        'provenance': provenance, 'implementation': current_implementation,
    }
    # Compute first, then publish a separate immutable artifact directory.
    verify_rows(protected)
    verify_rows(provenance)
    verify_rows(current_implementation)
    json.dumps(data, allow_nan=False)
    output.mkdir(exist_ok=False)
    (output / 'assets').mkdir()
    for path in sorted((Path(__file__).parent / 'web').glob('*')):
        if path.is_file():
            with (output / 'assets' / path.name).open('xb') as stream:
                stream.write(path.read_bytes())
    if videos:
        (output / 'media').mkdir()
    for source, target, row in videos:
        h = hashlib.sha256()
        with source.open('rb') as original, (output / target).open('xb') as copied:
            for chunk in iter(lambda: original.read(4 * 1024 * 1024), b''):
                copied.write(chunk)
                h.update(chunk)
        if h.hexdigest() != row['sha256']:
            raise ValueError('Video changed while copying; output is incomplete, use a new directory')
    _json_write(output / 'comparison.json', data)
    _json_write(output / 'rollback.json', rollback)
    from .page import render
    page = render(data)
    verify_rows(protected)
    verify_rows(provenance)
    verify_rows(current_implementation)
    _json_write(output / 'verification.json', {'status': 'passed', 'protected_file_count': len(protected),
        'stable_source_and_report_unchanged': True, 'source_provenance_rechecked': True,
        'no_running_batch_ingested': True, 'new_simulation_trials': 0})
    # The navigable entry is the final write; a failed recheck never publishes it.
    with (output / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(page)
    return data
