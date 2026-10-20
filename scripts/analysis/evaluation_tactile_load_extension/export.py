"""Copy a selected archived report into a new sibling, then add display-only charts."""
from __future__ import annotations

from html import escape
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

from evaluation_progress_experiment.export import digest, protected_snapshot, read_json, verify_rows, _json_write
from evaluation_v2.suite import implementation_manifest
from .data import load_extension


def implementation():
    package = Path(__file__).parent
    files = sorted(package.glob('*.py')) + sorted((package / 'web').glob('*'))
    files += [package.parent / 'extend_tactile_loads.py']
    files += sorted((package.parent / 'evaluation_progress_experiment').glob('*.py'))
    return [digest(path) for path in files if path.is_file()]


def _safe_files(root):
    paths = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in archived report: {path}')
        if path.is_file() and '__pycache__' not in path.parts:
            paths.append(path)
    return paths


def _copy_checked(source, destination, row):
    destination.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with source.open('rb') as original, destination.open('xb') as copied:
        for chunk in iter(lambda: original.read(4 * 1024 * 1024), b''):
            copied.write(chunk)
            h.update(chunk)
    if h.hexdigest() != row['sha256']:
        raise ValueError('Source changed during copy; incomplete output is not published')


def export_extension(source_report, h50_report, output):
    source = Path(source_report).resolve(strict=True)
    h50 = Path(h50_report).resolve(strict=True)
    if source.name != 'report.json' or h50.name != 'report.json':
        raise ValueError('Both inputs must be archived report.json files')
    site, stable = source.parent.parent, h50.parent.parent
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if output.parent != site.parent or output.parent != stable.parent:
        raise ValueError('Output must be a new sibling of the archived and H50 sites')
    protected = protected_snapshot(stable)
    batch = read_json(stable / 'batch.json')
    if batch.get('implementation') != implementation_manifest():
        raise ValueError('Stable implementation fingerprint changed')
    if batch.get('horizon_selection', {}).get('required_prediction_horizon') != 50:
        raise ValueError('Explicit H50-only baseline required')
    if not any(entry.get('prediction_horizon') == 50 and
               (stable / entry.get('report_path', '')).with_suffix('.json').resolve() == h50
               for entry in batch.get('trials', [])):
        raise ValueError('Selected H50 report is not a member of the stable cohort')
    originals = _safe_files(source.parent) + _safe_files(site / 'assets')
    if not (source.parent / 'report.html').is_file():
        raise ValueError('Source report page missing')
    if not all((site / 'assets' / name).is_file() for name in ('dashboard.js', 'dashboard.css', 'bootstrap.js')):
        raise ValueError('Source dashboard assets missing')
    protected = list({row['path']: row for row in protected + [digest(p) for p in originals]}.values())
    snapshots = {row['path']: row for row in protected}
    source_code = implementation()
    extension = load_extension(source, h50)
    provenance = extension.get('provenance', [])
    extension.update(source_page=f'../../{quote(site.name)}/{quote(source.parent.name)}/report.html',
                     stable_page=f'../../{quote(stable.name)}/{quote(h50.parent.name)}/report.html',
                     implementation=source_code)
    payload = json.dumps(extension, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = (source.parent / 'report.html').read_text(encoding='utf-8')
    if html.count('</head>') != 1 or html.count('</body>') != 1 or 'kai-evaluation-data' not in html:
        raise ValueError('Unsupported archived dashboard page structure')
    embedded = re.findall(r'<script\b[^>]*\bid="kai-evaluation-data"[^>]*>(.*?)</script>', html, flags=re.S)
    if len(embedded) != 1:
        raise ValueError('Archived HTML must contain exactly one data payload')
    payload_data = json.loads(embedded[0])
    if (payload_data.get('report') != read_json(source) or
            payload_data.get('visualization') != read_json(source.parent / 'visualization.json')):
        raise ValueError('Archived HTML payload differs from the verified report/visualization JSON')
    html = html.replace('</head>', '<link rel="stylesheet" href="../assets/tactile_loads.css">\n</head>')
    html = html.replace('</body>', f'<script type="application/json" id="kai-tactile-load-data">{payload}</script>\n<script src="../assets/tactile_loads.js" defer></script>\n</body>')
    baseline_url = f'../{quote(stable.name)}/index.html'
    rollback = {'schema_version': 'evaluation-experiment-rollback-v1', 'stable_site': str(stable),
                'stable_url': baseline_url, 'protected_files': protected,
                'source_page': str(source.parent / 'report.html'),
                'strategy': 'Additive display-only copy. Original page, V1/V2 code and H50 site remain unchanged.',
                'verification_command': '.pixi/envs/default/bin/python -B scripts/analysis/extend_tactile_loads.py '
                                        f'--verify-rollback {output / "rollback.json"}'}
    for rows in (protected, provenance, source_code):
        verify_rows(rows)
    output.mkdir(exist_ok=False)
    for path in originals:
        # Keep the original dashboard code, data, audit and video byte-identical.
        # Publish the augmented navigable report only after final verification.
        if path == source.parent / 'report.html':
            continue
        target = output / path.relative_to(site)
        _copy_checked(path, target, snapshots[str(path.resolve())])
    for path in sorted((Path(__file__).parent / 'web').glob('*')):
        if path.is_file():
            _copy_checked(path, output / 'assets' / path.name, digest(path))
    destination = output / source.parent.name
    _json_write(destination / 'tactile_loads.json', extension)
    _json_write(output / 'rollback.json', rollback)
    for rows in (protected, provenance, source_code):
        verify_rows(rows)
    _json_write(output / 'verification.json', {
        'status': 'passed', 'protected_file_count': len(protected),
        'original_report_and_stable_h50_unchanged': True,
        'only_selected_h50_trial': True, 'new_simulation_trials': 0,
        'source_provenance_rechecked': True, 'evaluation_metrics_recomputed': False,
    })
    with (destination / 'report.html').open('x', encoding='utf-8') as stream:
        stream.write(html)
    with (output / 'index.html').open('x', encoding='utf-8') as stream:
        link = f'{quote(source.parent.name)}/report.html'
        stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                     '<title>H50 · 五指法向与切向载荷</title><main style="max-width:800px;margin:4rem auto;padding:1rem;font:16px/1.8 system-ui">'
                     '<h1>五指法向与切向载荷 · 独立扩展版</h1><p>原报告版式保留，仅增加触觉载荷时间线。只包含已验证的这条 H50 轨迹，不纳入 H30。</p>'
                     f'<p><a href="{escape(link, quote=True)}">打开扩展报告 →</a></p>'
                     f'<p><a href="{escape(baseline_url, quote=True)}">回到原 H50 稳定版</a></p></main></html>')
    return extension
