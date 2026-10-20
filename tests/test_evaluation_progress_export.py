"""The experiment must remain optional and leave stable source/artifacts intact."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from evaluation_progress_experiment.export import export_experiment, read_json, verify_rows
from evaluation_progress_experiment.page import render
from evaluation_v2.suite import implementation_manifest


class ProgressExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / 'stable'
        self.site.mkdir()
        self.report = self.site / '000_whiteboard_seed_000'
        self.report.mkdir()
        (self.report / 'report.json').write_text('{}')
        (self.site / 'index.html').write_text('unchanged stable page')
        self.batch = {'implementation': implementation_manifest(), 'trial_count': 2,
            'horizon_selection': {'required_prediction_horizon': 50}, 'trials': [
                {'task':'whiteboard-wipe','trial_id':'policy','prediction_horizon':50,
                 'report_path':'000_whiteboard_seed_000/report.html'},
                {'task':'pick-place','trial_id':'other','prediction_horizon':50,'report_path':'not_loaded/report.html'}]}
        self.write_batch()
        self.episode = {'id':'policy','role':'policy','task':'whiteboard-wipe','outcome':'fail',
            'time_s':[0,1,2], 'progress':[0,.1,.4], 'gap_after':[False,False,False],
            'signals':{'board_normal':{'label':'Board force','unit':'N','contact_role':'eraser_board','values':[0,1,2]}},
            'metadata':{'compatibility':'unverified'},'media':{'status':'missing'},'provenance':[]}

    def write_batch(self):
        (self.site / 'batch.json').write_text(json.dumps(self.batch))

    def export(self, path=None, **kwargs):
        with patch('evaluation_progress_experiment.export.load_policy_episode', return_value=self.episode):
            return export_experiment(self.site,path or self.root/'experiment',**kwargs)

    def test_separate_export_preserves_stable_files_and_retains_failure(self):
        before={str(p):p.read_bytes() for p in self.site.rglob('*') if p.is_file()}
        result=self.export()
        self.assertEqual(result['episodes'][0]['outcome'],'fail')
        self.assertEqual(result['episodes'][0]['final_progress'],.4)
        self.assertEqual(result['selection']['selected_policy_trials'],1)
        self.assertEqual(len(result['selection']['skipped_tasks']),1)
        self.assertEqual(result['baseline_url'],'../stable/index.html')
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.site.rglob('*') if p.is_file()})
        rollback=read_json(self.root/'experiment/rollback.json')
        verify_rows(rollback['protected_files'])
        self.assertTrue(read_json(self.root/'experiment/verification.json')['stable_source_and_report_unchanged'])

    def test_existing_output_and_nonsibling_output_rejected(self):
        with self.assertRaises(FileExistsError):self.export(self.site)
        with self.assertRaises(ValueError):self.export(self.site/'experiment')

    def test_non_h50_baseline_rejected(self):
        self.batch['horizon_selection']['required_prediction_horizon']=30
        self.write_batch()
        with self.assertRaisesRegex(ValueError,'H50'):self.export()
        self.assertFalse((self.root/'experiment').exists())

    def test_stale_source_fingerprint_rejected(self):
        self.batch['implementation']['sha256']='0'*64
        self.write_batch()
        with self.assertRaisesRegex(ValueError,'fingerprint'):self.export()

    def test_unsafe_report_path_rejected(self):
        self.batch['trials'][0]['report_path']='../elsewhere/report.html'
        self.write_batch()
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.export()

    def test_rollback_cli_detects_changes_without_restoring_or_deleting(self):
        self.export()
        cli=Path(__file__).resolve().parents[1]/'scripts/analysis/compare_task_progress.py'
        command=[sys.executable,'-B',str(cli),'--verify-rollback',str(self.root/'experiment/rollback.json')]
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        (self.site/'index.html').write_text('intentional test-only change')
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('Protected source changed',result.stderr)
        self.assertEqual((self.site/'index.html').read_text(),'intentional test-only change')

    def test_render_escapes_embedded_script_markup(self):
        html=render({'text':'</script><script>alert(1)</script>'})
        self.assertNotIn('</script><script>alert',html)
        self.assertIn('\\u003c/script',html)

    def test_post_export_verification_failure_does_not_publish_entry(self):
        with patch('evaluation_progress_experiment.export.verify_rows',
                   side_effect=[None,None,None,ValueError('changed during output')]):
            with self.assertRaisesRegex(ValueError,'changed during output'):
                self.export()
        self.assertFalse((self.root/'experiment/index.html').exists())


if __name__=='__main__':
    unittest.main()
