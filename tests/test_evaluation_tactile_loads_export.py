"""The requested legacy layout is extended without editing any original bytes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
from evaluation_tactile_load_extension.export import export_extension
from evaluation_progress_experiment.export import read_json, verify_rows
from evaluation_v2.suite import implementation_manifest


class TactileLoadExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old = self.root / 'old'
        self.stable = self.root / 'stable'
        self.report_dir = self.old / '009_case'
        self.report_dir.mkdir(parents=True)
        self.stable_report = self.stable / '004_case/report.json'
        self.stable_report.parent.mkdir(parents=True)
        self.stable_report.write_text('{}')
        (self.stable / 'index.html').write_text('stable entry')
        self.batch = {'implementation': implementation_manifest(),
                      'horizon_selection': {'required_prediction_horizon': 50},
                      'trials': [{'prediction_horizon': 50, 'report_path': '004_case/report.html'}]}
        self.write_batch()
        (self.old / 'assets').mkdir()
        for name in ('dashboard.js', 'dashboard.css', 'bootstrap.js'):
            (self.old / 'assets' / name).write_text(f'original {name}')
        (self.report_dir / 'report.json').write_text('{}')
        (self.report_dir / 'visualization.json').write_text('{}')
        (self.report_dir / 'audit.html').write_text('original audit')
        (self.report_dir / 'report.html').write_text('<html><head></head><body><div id="kai-evaluation"></div><script id="kai-evaluation-data" type="application/json">{"report":{},"visualization":{}}</script></body></html>')
        (self.report_dir / 'media').mkdir()
        (self.report_dir / 'media/rollout.mp4').write_bytes(b'original video')
        self.data = {'time_s': [0, 1], 'provenance': [], 'semantics': {'note': '</script><script>bad()</script>'}}

    def write_batch(self):
        (self.stable / 'batch.json').write_text(json.dumps(self.batch))

    def export(self, output=None):
        with patch('evaluation_tactile_load_extension.export.load_extension', return_value=self.data.copy()):
            return export_extension(self.report_dir / 'report.json', self.stable_report, output or self.root / 'new')

    def test_new_copy_keeps_old_page_and_old_assets_unchanged(self):
        before = {str(p): p.read_bytes() for root in (self.old, self.stable) for p in root.rglob('*') if p.is_file()}
        data = self.export()
        after = {str(p): p.read_bytes() for root in (self.old, self.stable) for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        verify_rows(read_json(self.root / 'new/rollback.json')['protected_files'])
        self.assertEqual((self.root / 'new/assets/dashboard.js').read_bytes(), (self.old / 'assets/dashboard.js').read_bytes())
        for name in ('report.json', 'visualization.json', 'audit.html', 'media/rollout.mp4'):
            self.assertEqual((self.root / 'new/009_case' / name).read_bytes(), (self.report_dir / name).read_bytes())
        html = (self.root / 'new/009_case/report.html').read_text()
        self.assertIn('tactile_loads.js', html)
        self.assertIn('kai-tactile-load-data', html)
        self.assertNotIn('</script><script>bad()', html)
        self.assertEqual(data['source_page'], '../../old/009_case/report.html')

    def test_existing_and_nested_destinations_are_rejected(self):
        with self.assertRaises(FileExistsError): self.export(self.old)
        with self.assertRaises(ValueError): self.export(self.old / 'nested')

    def test_h30_cohort_or_nonmember_rejected(self):
        self.batch['trials'][0]['prediction_horizon'] = 30
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'not a member'): self.export()
        self.batch['horizon_selection']['required_prediction_horizon'] = 30
        self.write_batch()
        with self.assertRaisesRegex(ValueError, 'H50'): self.export()

    def test_source_symlink_rejected(self):
        (self.report_dir / 'linked.json').symlink_to(self.report_dir / 'report.json')
        with self.assertRaisesRegex(ValueError, 'Symlink'): self.export()

    def test_divergent_embedded_payload_rejected(self):
        page = self.report_dir / 'report.html'
        page.write_text(page.read_text().replace('"report":{}', '"report":{"different":true}'))
        with self.assertRaisesRegex(ValueError, 'payload differs'): self.export()
        self.assertFalse((self.root / 'new').exists())

    def test_final_verification_failure_publishes_no_page(self):
        with patch('evaluation_tactile_load_extension.export.verify_rows', side_effect=[None, None, None, ValueError('changed')]):
            with self.assertRaisesRegex(ValueError, 'changed'): self.export()
        self.assertFalse((self.root / 'new/009_case/report.html').exists())
        self.assertFalse((self.root / 'new/index.html').exists())


if __name__ == '__main__':
    unittest.main()
