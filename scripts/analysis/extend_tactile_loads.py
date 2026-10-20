#!/usr/bin/env python3
"""Add five-finger Fn/Ft timelines in a new report copy, preserving old H50 artifacts."""
import argparse
import json
from pathlib import Path

from evaluation_tactile_load_extension.export import export_extension
from evaluation_progress_experiment.export import read_json, verify_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path)
    parser.add_argument('--h50-report', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-rollback', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_rollback:
            data = read_json(args.verify_rollback)
            if data.get('schema_version') != 'evaluation-experiment-rollback-v1':
                raise ValueError('Not a rollback manifest')
            verify_rows(data['protected_files'])
            print(json.dumps({'verified': True, 'protected_file_count': len(data['protected_files'])}))
            return 0
        if not all((args.source_report, args.h50_report, args.output_dir)):
            parser.error('--source-report, --h50-report and --output-dir are required')
        data = export_extension(args.source_report, args.h50_report, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Tactile extension stopped: {error}\n')
    print(json.dumps({'output': str(args.output_dir.resolve()), 'samples': len(data['time_s']),
                      'original_and_h50_unchanged': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
