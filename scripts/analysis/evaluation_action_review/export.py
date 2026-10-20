"""Additive, source-bound pilot export. Never writes into a source site."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from evaluation_v2.suite import implementation_manifest
from evaluation_progress_experiment.adapters import load_policy_episode
from evaluation_progress_experiment.export import (
    _copy_media, _json_write, digest, protected_snapshot, read_json, verify_rows,
)

from . import SCHEMA_VERSION
from .events import build_review
from .motion import load_wrist_motion


def implementation():
    package = Path(__file__).parent
    paths = sorted(package.glob('*.py')) + sorted((package / 'web').glob('*'))
    paths += [package.parent / 'review_action_evidence.py']
    # Imported adapter/export helpers are also part of the actual implementation.
    paths += sorted((package.parent / 'evaluation_progress_experiment').glob('*.py'))
    return [digest(p) for p in paths if p.is_file()]


def export_review(stable_site, output):
    site = Path(stable_site).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if output.parent != site.parent:
        raise ValueError('Review output must be a new sibling of the stable site')
    protected = protected_snapshot(site)
    batch = read_json(site / 'batch.json')
    if batch.get('implementation') != implementation_manifest():
        raise ValueError('Stable implementation changed; refusing to replace the baseline')
    if batch.get('horizon_selection', {}).get('required_prediction_horizon') != 50:
        raise ValueError('An explicit H50-only stable cohort is required')
    source_code = implementation()
    episodes, videos, provenance, skipped = [], [], [], []
    for entry in batch['trials']:
        if entry.get('prediction_horizon') != 50:
            raise ValueError('Non-H50 trial in stable cohort')
        if entry['task'] != 'whiteboard-wipe':
            skipped.append({'trial_id': entry['trial_id'], 'task': entry['task'],
                            'reason': 'outside_explicit_whiteboard_pilot'})
            continue
        relative = Path(entry['report_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source report path')
        path = (site / relative).with_suffix('.json').resolve(strict=True)
        if not path.is_relative_to(site):
            raise ValueError('Report escaped stable site')
        episode = load_policy_episode(path)
        motion = load_wrist_motion(episode)
        review = build_review(episode, motion=motion)
        media, copy = _copy_media(episode, path, len(episodes))
        if copy:
            videos.append(copy)
        # These two archived videos were frame-inspected, not inferred from a task name.
        if media.get('sha256') in {
            '8febcefd9890250ec3be7923cc50d6fb6a7dfebb6a06c4f0b9a033ecc8737afa',
            'e28542122e5e14da975ecac87a0638bbbcd2598ff17843ce625208717d92f784',
        }:
            media['views'] = {'source_width': 1920, 'source_height': 1080,
                              'verification': 'Frame-inspected archived videos; CSS display crop only',
                              'crops': [
                                  {'id': 'head', 'label': '主视角', 'x': 214, 'y': 78, 'width': 434, 'height': 324},
                                  {'id': 'wrist', 'label': '手腕近景', 'x': 214, 'y': 440, 'width': 434, 'height': 294},
                                  {'id': 'global', 'label': '全局', 'x': 214, 'y': 746, 'width': 434, 'height': 324},
                              ]}
        review.update(media=media, source_report=str(path),
                      display_name=f"{path.parent.name} · {episode['id'].split(':')[1]}",
                      source_report_url=f'../{quote(site.name)}/{quote(path.parent.name)}/report.html',
                      metadata=episode.get('metadata', {}),
                      warnings=episode.get('warnings', []),
                      motion_status=motion.get('status'),
                      motion_warnings=motion.get('warnings', []),
                      motion_limitations=motion.get('limitations', []))
        episodes.append(review)
        provenance.extend(episode.get('provenance', []))
        provenance.extend(motion.get('provenance', []))
    if not episodes:
        raise ValueError('No whiteboard-wipe episodes in the H50 stable cohort')
    if len({e['id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episodes')
    provenance = list({row['path']: row for row in provenance}.values())
    baseline = f'../{quote(site.name)}/index.html'
    data = {
        'schema_version': SCHEMA_VERSION, 'experimental': True, 'baseline_url': baseline,
        'selection': {'input_policy_trials': batch['trial_count'],
                      'selected_policy_trials': len(episodes), 'required_prediction_horizon': 50,
                      'scope': 'whiteboard-wipe pilot; all eligible outcomes retained',
                      'skipped_tasks': skipped},
        'episodes': episodes, 'provenance': provenance, 'implementation': source_code,
        'semantics': {
            'purpose': 'Within-recording video and task-response inspection, not expert regularity scoring',
            'candidate_rules': 'Fixed exploratory observation rules; not validated failure detectors',
            'comparison': 'Manual same-trajectory contrast; conditions need not match',
            'causality': 'No causal attribution or new performance score',
            'missing': 'Missing records stay missing; unmarked intervals are not certified normal',
        },
    }
    rollback = {'schema_version': 'evaluation-experiment-rollback-v1',
                'stable_site': str(site), 'stable_url': baseline,
                'protected_files': protected,
                'strategy': 'Additive new directory only; open stable_url to return. No reset, restore or deletion.',
                'verification_command': '.pixi/envs/default/bin/python -B scripts/analysis/review_action_evidence.py '
                                        f'--verify-rollback {output / "rollback.json"}'}
    verify_rows(protected)
    verify_rows(provenance)
    verify_rows(source_code)
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
            raise ValueError('Video changed while copying; incomplete output has no index')
    _json_write(output / 'review.json', data)
    _json_write(output / 'rollback.json', rollback)
    from .page import render
    page = render(data)
    verify_rows(protected)
    verify_rows(provenance)
    verify_rows(source_code)
    _json_write(output / 'verification.json', {
        'status': 'passed', 'protected_file_count': len(protected),
        'stable_source_and_report_unchanged': True, 'source_provenance_rechecked': True,
        'new_simulation_trials': 0, 'no_running_batch_ingested': True,
    })
    # A navigable artifact is published only after the final recheck.
    with (output / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(page)
    return data
