"""The action review only publishes a verified new sibling artifact."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from evaluation_action_review.export import export_review
from evaluation_action_review.page import render
from evaluation_progress_experiment.export import read_json, verify_rows
from evaluation_v2.suite import implementation_manifest


class ActionReviewExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / 'stable'
        self.site.mkdir()
        report = self.site / '000_whiteboard_seed_000'
        report.mkdir()
        (report / 'report.json').write_text('{}')
        (self.site / 'index.html').write_text('immutable stable entry')
        self.batch = {'implementation': implementation_manifest(), 'trial_count': 2,
            'horizon_selection': {'required_prediction_horizon': 50}, 'trials': [
                {'task': 'whiteboard-wipe', 'trial_id': 'policy', 'prediction_horizon': 50,
                 'report_path': '000_whiteboard_seed_000/report.html'},
                {'task': 'pick-place', 'trial_id': 'other', 'prediction_horizon': 50,
                 'report_path': 'not_loaded/report.html'}]}
        self.write_batch()
        self.episode = {'id': 'whiteboard-wipe:step24999:policy', 'role': 'policy',
            'task': 'whiteboard-wipe', 'outcome': 'fail',
            'time_s': [0, 1, 2, 3], 'progress': [0, 0, 0, .4], 'gap_after': [False]*4,
            'signals': {'board_normal': {'label': 'Board force', 'unit': 'N',
                         'contact_role': 'eraser_board', 'values': [0, 1, 2, 3]}},
            'metadata': {}, 'media': {'status': 'missing'}, 'provenance': []}

    def write_batch(self):
        (self.site / 'batch.json').write_text(json.dumps(self.batch))

    def export(self, path=None):
        motion = {'status': 'missing', 'time_s': self.episode['time_s'], 'position_m': [None]*4}
        with patch('evaluation_action_review.export.load_policy_episode', return_value=self.episode), \
             patch('evaluation_action_review.export.load_wrist_motion', return_value=motion):
            return export_review(self.site, path or self.root / 'review')

    def test_export_retains_failure_all_candidates_and_stable_bytes(self):
        before = {str(p): p.read_bytes() for p in self.site.rglob('*') if p.is_file()}
        result = self.export()
        self.assertEqual(result['episodes'][0]['outcome'], 'fail')
        self.assertEqual(len(result['episodes'][0]['events']), 2)
        self.assertEqual(result['selection']['selected_policy_trials'], 1)
        self.assertEqual(len(result['selection']['skipped_tasks']), 1)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.site.rglob('*') if p.is_file()})
        verify_rows(read_json(self.root / 'review/rollback.json')['protected_files'])
        self.assertTrue((self.root / 'review/index.html').exists())

    def test_existing_or_nested_output_rejected(self):
        with self.assertRaises(FileExistsError): self.export(self.site)
        with self.assertRaises(ValueError): self.export(self.site / 'review')

    def test_h30_anywhere_rejected_even_outside_pilot(self):
        self.batch['trials'][1]['prediction_horizon'] = 30
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'H50'): self.export()
        self.assertFalse((self.root / 'review').exists())

    def test_h50_selection_required(self):
        self.batch['horizon_selection']['required_prediction_horizon'] = 30
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'H50'): self.export()

    def test_changed_implementation_rejected(self):
        self.batch['implementation']['sha256'] = '0'*64
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'implementation changed'): self.export()

    def test_unsafe_report_path_rejected(self):
        self.batch['trials'][0]['report_path'] = '../elsewhere/report.html'
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'Unsafe'): self.export()

    def test_failed_final_verification_never_publishes_index(self):
        with patch('evaluation_action_review.export.verify_rows',
                   side_effect=[None, None, None, ValueError('changed during output')]):
            with self.assertRaisesRegex(ValueError, 'changed during output'): self.export()
        self.assertFalse((self.root / 'review/index.html').exists())

    def test_html_data_cannot_escape_script(self):
        html = render({'text': '</script><script>alert(1)</script>'})
        self.assertNotIn('</script><script>alert', html)
        self.assertIn('\\u003c/script', html)

    def test_rollback_checks_but_never_restores_changed_files(self):
        self.export()
        cli = Path(__file__).resolve().parents[1] / 'scripts/analysis/review_action_evidence.py'
        command = [sys.executable, '-B', str(cli), '--verify-rollback', str(self.root/'review/rollback.json')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.site/'index.html').write_text('test-only modification')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual((self.site/'index.html').read_text(), 'test-only modification')


if __name__ == '__main__':
    unittest.main()
