#!/usr/bin/env python3
"""Export a separate H50 action-evidence review; never overwrite the stable site."""
import argparse
import json
from pathlib import Path

from evaluation_action_review.export import export_review
from evaluation_progress_experiment.export import read_json, verify_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable-site', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-rollback', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_rollback:
            value = read_json(args.verify_rollback)
            if value.get('schema_version') != 'evaluation-experiment-rollback-v1':
                raise ValueError('Not a rollback manifest')
            verify_rows(value['protected_files'])
            print(json.dumps({'verified': True, 'protected_file_count': len(value['protected_files'])}))
            return 0
        if not args.stable_site or not args.output_dir:
            parser.error('--stable-site and --output-dir are required')
        result = export_review(args.stable_site, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'Review stopped: {exc}\n')
    print(json.dumps({'output': str(args.output_dir.resolve()), 'episodes': len(result['episodes']),
                      'stable_unchanged': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
