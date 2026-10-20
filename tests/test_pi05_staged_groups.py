"""Temporary-fixture tests: never launch services or touch evaluation archives."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

DEPLOY = Path(__file__).resolve().parents[1] / 'scripts/deploy'
sys.path.insert(0, str(DEPLOY))
SPEC = importlib.util.spec_from_file_location('staged_groups_under_test', DEPLOY / 'run_pi05_staged_groups.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class StagedDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive'
        self.snapshots = self.root / 'runtime_snapshots'
        self.run_id = 'staged_v1_test'
        self.job = self.archive / '_jobs' / ('pi05_' + self.run_id)
        self.sim = self.snapshots / 'frozen/kaitactilesim'
        self.openpi = self.snapshots / 'frozen/openpi'
        self.tokenizer = self.root / 'paligemma_tokenizer.model'
        self.tokenizer.write_bytes(b'cached tokenizer')
        self.model_python = self.root / 'model-python'
        self.model_python.write_text('#!/bin/sh\nexit 0\n')
        self.model_python.chmod(0o755)
        interpreter = self.sim / '.pixi/envs/default/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(self.model_python.read_bytes())
        interpreter.chmod(0o755)
        for tree in (self.sim / 'src', self.sim / 'scripts', self.sim / 'configs',
                     self.openpi / 'src', self.openpi / 'scripts', self.openpi / 'packages'):
            tree.mkdir(parents=True)
            (tree / 'fixture.py').write_text('# immutable fixture\n')
        observer = self.sim / driver.OBSERVER_SOURCE
        observer.parent.mkdir(parents=True)
        observer.write_text('# authenticated staged observer fixture\n')
        for tree in (self.sim / 'src', self.sim / 'scripts', self.sim / 'configs',
                     self.openpi / 'src', self.openpi / 'scripts', self.openpi / 'packages'):
            for current, directories, names in os.walk(tree):
                for name in names:
                    (Path(current) / name).chmod(0o444)
                for name in directories:
                    (Path(current) / name).chmod(0o555)
                Path(current).chmod(0o555)
        self.patches = [patch.object(driver, 'ARCHIVE_ROOT', self.archive),
            patch.object(driver, 'JOBS_ROOT', self.archive / '_jobs'),
            patch.object(driver, 'SNAPSHOT_ROOT', self.snapshots),
            patch.object(driver.base, 'MODEL_PYTHON', self.model_python),
            patch.object(driver.base, 'TOKENIZER_SOURCE', self.tokenizer)]
        for item in self.patches:
            item.start()
        self.make_plan(('whiteboard-wipe-horizontal-line', 'bulb-screw'))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        for current, directories, _names in os.walk(self.root):
            Path(current).chmod(0o755)
            for name in directories:
                path = Path(current) / name
                if not path.is_symlink():
                    path.chmod(0o755)
        self.temp.cleanup()

    def make_plan(self, tasks):
        self.plan = dict(run_id=self.run_id, job_root=str(self.job), simulation_root=str(self.sim),
            openpi_root=str(self.openpi), authorized_tasks=list(tasks), settings=dict(driver.SETTINGS), groups=[])
        for task in tasks:
            for step in (20000, 24999):
                data_task = 'usb_insert' if task == 'usb-insert' else task
                checkpoint = self.root / 'sim' / data_task / 'pi05/1012_200/checkpoints/config/run' / str(step)
                (checkpoint / 'params').mkdir(parents=True, exist_ok=True)
                (checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
                (checkpoint / 'params/_METADATA').write_text('{}')
                self.plan['groups'].append(dict(task=task, step=step, checkpoint=str(checkpoint),
                    dataset='1012_200', port=18820 + len(self.plan['groups']),
                    archive_dir=str(self.archive / task / 'pi05' / f'step{step}_H50K50_1012_200_{self.run_id}')))
        self.report = dict(all_selected_valid=True, count=len(self.plan['groups']),
            staged_evaluation_version='staged_v1', simulation_root=str(self.sim), model_project_root=str(self.openpi),
            checkpoints=[dict(task=g['task'], step=g['step'], checkpoint=g['checkpoint'], valid=True,
                prediction_horizon=50, execute_steps=50, binding_status='bound', required_action_interpolation='linear')
                for g in self.plan['groups']])
        driver.write(self.sim.parent / 'preflight_report.json', self.report)

    def job_ready(self):
        self.job.mkdir(parents=True)
        driver.write(self.job / 'plan.json', self.plan)

    def observer_report(self, task, complete=True):
        names = driver.STAGE_NAMES[task]
        flags = {f'stage{i + 1}': bool(complete) for i in range(len(names))}
        continuous = {}
        if task in driver.BOARDS:
            continuous = {key: 0.25 for key in driver.BOARD_NUMERIC_METRICS}
            continuous.update(schema_version='geometry_v1', measurement_type='rigid estimate',
                is_true_deformable_contact_area=False, thresholds={'load_n': 0.1},
                pad_grid_spacing_m=[0.001, 0.001], geometry_sample_age_s=None,
                contact_time_weighted_mean_effective_area_ratio=None)
        return dict(version='staged_v1', task=task, stage_count=len(names), stages=flags, stage_flags=flags.copy(),
            stage_definitions={f'stage{i + 1}': name for i, name in enumerate(names)},
            complete_success=None if task in driver.BOARDS else bool(complete), time_s=90.0, observations=9001,
            full_duration_evaluation=True, native_task_success_preserved=True, expert_trajectory_used=False,
            simulation_state_mutated=False, continuous_metrics=continuous)

    def write_trace(self, path, group, seed, *, count=2701, gap=False):
        with path.open('w') as stream:
            for index in range(count):
                payload = dict(version='staged_v1', task=group['task'], seed=seed, prediction_horizon=50,
                    execute_steps=50, checkpoint_sha256='a' * 64, time_s=index / 30 + (1 if gap and index else 0))
                stream.write(json.dumps(payload) + '\n')

    def make_evaluation(self, index=0):
        group = self.plan['groups'][index]
        evaluation = driver.output_dir(self.plan, index) / 'run/evaluation'
        evaluation.mkdir(parents=True)
        deployment = dict(task=group['task'], checkpoint_path=group['checkpoint'], prediction_horizon=50,
            checkpoint_sha256='a' * 64, deployment_id='deployment', model_config_name='from_training_provenance')
        rows = []
        reports = []
        for seed in range(20):
            directory = evaluation / f'seed_{seed:03d}'
            directory.mkdir()
            report = self.observer_report(group['task'])
            reports.append(report)
            driver.write(directory / 'summary.json', dict(task=group['task'], seed=seed,
                checkpoint_path=group['checkpoint'], checkpoint_sha256='a' * 64, deployment_id='deployment',
                prediction_horizon=50, execute_steps=50, sim_seconds=90.0,
                deployment_integrity={'status': 'verified', 'errors': []}, staged_evaluation=report))
            driver.write(directory / 'staged_evaluation.json', report)
            self.write_trace(directory / 'staged_states.jsonl', group, seed)
            (directory / 'review.mp4').write_bytes(b'valid video fixture')
            rows.append(dict(seed=seed, valid_trial=True, summary=f'seed_{seed:03d}/summary.json'))
        driver.write(evaluation / 'summary.json', dict(task=group['task'], prediction_horizon=50, execute_steps=50,
            planned=20, attempted=20, valid_trials=20, checkpoint_sha256='a' * 64, deployment_id='deployment',
            diagnostic_only=True, formal_metrics_valid=False, results=rows, successes=0,
            staged_evaluation=driver.aggregate_reports(reports, group['task'])))
        return evaluation, deployment

    def test_authorized_subset_and_all_nine_pairs(self):
        driver.validate_plan(self.plan)
        driver.verify_frozen(self.plan)
        self.make_plan(driver.TASKS)
        driver.validate_plan(self.plan)
        driver.verify_frozen(self.plan)
        self.assertEqual(len(self.plan['groups']), 18)
        self.assertEqual(self.plan['groups'][-1]['port'], 18837)
        self.assertFalse(self.job.exists())

    def test_missing_pair_or_unauthorized_task_is_rejected(self):
        self.plan['authorized_tasks'].append('usb-insert')
        with self.assertRaisesRegex(ValueError, 'exactly both'):
            driver.validate_plan(self.plan)

    def test_explicit_poker_retry_reuses_only_its_failed_service_ports(self):
        self.make_plan(('poker-draw',))
        self.plan['port_start'] = 18830
        for index, group in enumerate(self.plan['groups']):
            group['port'] = 18830 + index
        driver.validate_plan(self.plan)
        driver.verify_frozen(self.plan)
        self.plan['groups'][1]['port'] = 18830
        with self.assertRaisesRegex(ValueError, 'unique consecutive ports'):
            driver.validate_plan(self.plan)

    def test_poker_retry_port_override_cannot_select_other_ports_or_tasks(self):
        self.make_plan(('poker-draw',))
        for invalid in (18831, 18920, True, '18830'):
            self.plan['port_start'] = invalid
            with self.subTest(port_start=invalid), self.assertRaisesRegex(ValueError, 'explicit Poker retry'):
                driver.validate_plan(self.plan)
        self.make_plan(('usb-insert',))
        self.plan['port_start'] = 18830
        self.plan['groups'][0]['port'] = 18830
        self.plan['groups'][1]['port'] = 18831
        with self.assertRaisesRegex(ValueError, 'explicit Poker retry'):
            driver.validate_plan(self.plan)

    def test_horizon_and_parallel_cap_are_not_bypassed(self):
        self.plan['settings']['parallel_groups'] = 5
        with self.assertRaisesRegex(ValueError, 'cap'):
            driver.validate_plan(self.plan)
        self.plan['settings']['parallel_groups'] = 4
        self.plan['settings']['execute_steps'] = 8
        with self.assertRaisesRegex(ValueError, 'H50/K50'):
            driver.validate_plan(self.plan)

    def test_usb_only_explicit_dataset_alias(self):
        self.make_plan(('usb-insert',))
        driver.validate_plan(self.plan)
        self.plan['groups'][0]['checkpoint'] = self.plan['groups'][0]['checkpoint'].replace('/usb_insert/', '/usb-insert/')
        with self.assertRaisesRegex(ValueError, 'task/dataset/step'):
            driver.validate_plan(self.plan)

    def test_new_version_sibling_preserves_existing_canonical(self):
        group = self.plan['groups'][0]
        legacy = self.archive / group['task'] / 'pi05' / 'step20000_H50K50_1012_200'
        legacy.mkdir(parents=True)
        (legacy / 'original.json').write_text('legacy unchanged')
        driver.validate_plan(self.plan)
        self.plan['groups'][0]['archive_dir'] = str(legacy)
        with self.assertRaisesRegex(ValueError, 'new run-version sibling'):
            driver.validate_plan(self.plan)
        self.assertEqual((legacy / 'original.json').read_text(), 'legacy unchanged')

    def test_existing_staged_archive_or_job_is_never_overwritten(self):
        Path(self.plan['groups'][0]['archive_dir']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            driver.validate_plan(self.plan)
        self.job_ready()
        with self.assertRaisesRegex(ValueError, 'existence/collision'):
            driver.validate_plan(self.plan)

    def test_real_preflight_gate_rejects_legacy_and_wrong_source(self):
        self.report['staged_evaluation_version'] = None
        driver.write(self.sim.parent / 'preflight_report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'legacy reports'):
            driver.verify_frozen(self.plan)
        self.report['staged_evaluation_version'] = 'staged_v1'
        self.report['checkpoints'][0]['checkpoint'] = '/not/the/checkpoint'
        driver.write(self.sim.parent / 'preflight_report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'preflight'):
            driver.verify_frozen(self.plan)

    def test_frozen_gate_rejects_writable_observer(self):
        (self.sim / driver.OBSERVER_SOURCE).chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'writable/linked'):
            driver.verify_frozen(self.plan)

    def test_start_detaches_with_real_preflight_and_frozen_path(self):
        process = MagicMock(pid=123456789)
        with patch.object(driver, 'port_free', return_value=True), \
                patch.object(driver.subprocess, 'Popen', return_value=process) as popen:
            self.assertEqual(driver.start(self.plan), process.pid)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(popen.call_args.args[0][0], str(driver.base.CONTROLLER_PYTHON))
        self.assertEqual(Path(popen.call_args.args[0][1]).name, 'run_pi05_staged_groups.py')
        self.assertEqual((self.job / '.cache/openpi/big_vision/paligemma_tokenizer.model').read_bytes(), b'cached tokenizer')

    def test_actual_per_task_stage_counts_and_definitions(self):
        for task in driver.TASKS:
            driver.validate_report(self.observer_report(task), task)
        report = self.observer_report('install-ram')
        report['stage_count'] = 3
        with self.assertRaisesRegex(ValueError, 'stage_count'):
            driver.validate_report(report, 'install-ram')

    def test_geometry_metadata_nulls_and_numeric_only_aggregation(self):
        task = 'whiteboard-wipe'
        reports = [self.observer_report(task), self.observer_report(task, False)]
        reports[1]['continuous_metrics']['cleaned_fraction'] = 0.75
        result = driver.aggregate_reports(reports, task)
        self.assertEqual(result['stages']['stage1']['count'], 1)
        self.assertEqual(result['continuous_metrics']['cleaned_fraction'], {'mean': 0.5, 'count': 2})
        self.assertEqual(result['continuous_metrics']['contact_time_weighted_mean_effective_area_ratio'], {'mean': None, 'count': 0})
        self.assertNotIn('is_true_deformable_contact_area', result['continuous_metrics'])
        self.assertNotIn('thresholds', result['continuous_metrics'])
        self.assertIsNone(result['complete_success_rate'])
        self.assertFalse(result['formal_metrics_valid'])

    def test_nonfinite_required_measurement_or_false_passive_proof_fails(self):
        report = self.observer_report('whiteboard-wipe')
        report['continuous_metrics']['cleaned_fraction'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            driver.validate_report(report, 'whiteboard-wipe')
        report = self.observer_report('bulb-screw')
        report['simulation_state_mutated'] = True
        with self.assertRaisesRegex(ValueError, 'passive'):
            driver.validate_report(report, 'bulb-screw')

    def test_complete_success_rate_is_final_stage_not_legacy_outcome(self):
        reports = [self.observer_report('bulb-screw'), self.observer_report('bulb-screw', False)]
        result = driver.aggregate_reports(reports, 'bulb-screw')
        self.assertEqual(result['complete_success_rate'], 0.5)
        self.assertTrue(result['diagnostic_only'])
        reports[1]['complete_success'] = True
        with self.assertRaisesRegex(ValueError, 'final stage'):
            driver.aggregate_reports(reports, 'bulb-screw')

    def test_trace_requires_full_duration_and_identity(self):
        path = self.root / 'trace.jsonl'
        group = self.plan['groups'][0]
        deployment = {'checkpoint_sha256': 'a' * 64}
        self.write_trace(path, group, 0)
        self.assertEqual(driver.validate_states(path, group, 0, deployment)['rows'], 2701)
        with self.assertRaisesRegex(ValueError, 'identity'):
            driver.validate_states(path, group, 1, deployment)
        self.write_trace(path, group, 0, count=10)
        with self.assertRaisesRegex(ValueError, 'full 90'):
            driver.validate_states(path, group, 0, deployment)
        self.write_trace(path, group, 0, count=10, gap=True)
        with self.assertRaisesRegex(ValueError, 'gap'):
            driver.validate_states(path, group, 0, deployment)

    def test_publication_full_twenty_preserves_original_metadata(self):
        self.job_ready()
        source, deployment = self.make_evaluation()
        original = (source / 'summary.json').read_bytes()
        result = driver.publish_archive(self.plan, 0, deployment)
        destination = Path(result['archive_dir'])
        self.assertTrue(source.is_symlink())
        self.assertEqual(os.readlink(source), str(destination))
        self.assertEqual((destination / 'summary.json').read_bytes(), original)
        manifest = driver.read(destination / 'archive_manifest.json')
        self.assertEqual(manifest['staged_aggregate']['trials'], 20)
        self.assertEqual(len(manifest['observation_coverage']), 20)
        self.assertFalse(manifest['formal_metrics_valid'])
        with self.assertRaisesRegex(ValueError, 'Unexpected path symlink'):
            driver.publish_archive(self.plan, 0, deployment)

    def test_partial_native_progress_write_does_not_abort_controller(self):
        self.job_ready()
        output = driver.output_dir(self.plan, 0)
        (output / 'run/evaluation').mkdir(parents=True)
        (output / 'run/evaluation/summary.json').write_text('{"partial":')
        status = dict(groups=[dict(g, status='queued') for g in self.plan['groups']])
        driver.refresh(self.plan, status)
        self.assertEqual(status['valid_trials'], 0)

    def test_legacy_only_batch_is_not_reclassified_as_staged(self):
        self.job_ready()
        evaluation, deployment = self.make_evaluation()
        summary = driver.read(evaluation / 'summary.json')
        summary.pop('staged_evaluation')
        driver.write(evaluation / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'Legacy-only'):
            driver.validate_native(evaluation, self.plan['groups'][0], deployment)

    def test_publication_link_failure_rolls_back_only_owned_directory(self):
        self.job_ready()
        source, deployment = self.make_evaluation()
        original = (source / 'summary.json').read_bytes()
        with patch.object(driver.os, 'symlink', side_effect=PermissionError('fixture denied link')):
            with self.assertRaises(PermissionError):
                driver.publish_archive(self.plan, 0, deployment)
        self.assertTrue(source.is_dir())
        self.assertFalse(source.is_symlink())
        self.assertFalse(Path(self.plan['groups'][0]['archive_dir']).exists())
        self.assertEqual((source / 'summary.json').read_bytes(), original)

    def fake_api(self, include_observer=True):
        group = self.plan['groups'][0]
        manifest = dict(task=group['task'], checkpoint_path=group['checkpoint'], prediction_horizon=50,
            deployment_id='deployment', checkpoint_sha256='a' * 64,
            evaluation_source_files=[{'path': driver.OBSERVER_SOURCE, 'sha256': driver.sha(self.sim / driver.OBSERVER_SOURCE)}]
                if include_observer else [])
        def prepare(_plan):
            directory = driver.output_dir(self.plan, 0) / 'deployment'
            directory.mkdir()
            driver.write(directory / 'deployment_manifest.json', manifest)
            return 0
        module = SimpleNamespace(build_prepare_plan=MagicMock(return_value=SimpleNamespace()),
            execute_prepare=MagicMock(side_effect=prepare), verify_deployment=MagicMock(return_value={'valid': True}),
            write_pipeline_deployment=MagicMock(),
            run_deployment=MagicMock(return_value={'evaluation_returncode': 0,
                'managed_server_stopped': True, 'evaluation_process_stopped': True}))
        return SimpleNamespace(deployment=module, profile=SimpleNamespace())

    def test_worker_authenticates_observer_and_passes_real_runtime_flag(self):
        self.job_ready()
        api = self.fake_api()
        with patch.object(driver.base, 'load_api', return_value=api), \
                patch.object(driver, 'port_free', return_value=True), \
                patch.object(driver, 'publish_archive', return_value={'valid_trials': 20, 'completed_trials': 20}):
            self.assertEqual(driver.run_worker(self.plan, 0), 0)
        kwargs = api.deployment.run_deployment.call_args.kwargs
        self.assertEqual(kwargs['evaluator_args'], ('--staged-evaluation-version', 'staged_v1'))
        self.assertEqual(kwargs['execute_steps'], 50)
        self.assertEqual(kwargs['max_sim_seconds'], 90)
        self.assertNotIn('model_config_name', api.deployment.build_prepare_plan.call_args.kwargs)

    def test_worker_refuses_unbound_observer_before_any_evaluation(self):
        self.job_ready()
        api = self.fake_api(include_observer=False)
        with patch.object(driver.base, 'load_api', return_value=api), patch.object(driver, 'port_free', return_value=True):
            self.assertEqual(driver.run_worker(self.plan, 0), 1)
        api.deployment.run_deployment.assert_not_called()
        status = driver.read(driver.output_dir(self.plan, 0) / 'group_status.json')
        self.assertEqual(status['status'], 'failed')
        self.assertIn('source closure', status['errors'][0])

    def test_unconfirmed_cleanup_quarantines_capacity_without_stopping_peers(self):
        self.make_plan(('bulb-screw', 'install-ram', 'usb-insert'))
        self.job_ready()
        launched, terminated = [], []
        test = self
        class FakeProcess:
            def __init__(self, index):
                self.index, self.pid = index, 10000000 + index
            def poll(self):
                clean = self.index >= 4
                driver.write(driver.output_dir(test.plan, self.index) / 'group_status.json',
                    dict(status='complete', stage='complete', owned_cleanup_confirmed=clean, valid_trials=20))
                return 0
            def terminate(self):
                terminated.append(self.index)
            def wait(self, timeout):
                return 0
        def launch(_plan, index):
            driver.output_dir(self.plan, index).mkdir(parents=True)
            launched.append(index)
            return FakeProcess(index)
        with patch.object(driver, 'port_free', return_value=True), patch.object(driver, 'launch_worker', side_effect=launch), \
                patch.object(driver.time, 'sleep'):
            self.assertEqual(driver.run_controller(self.plan), 1)
        result = driver.read(self.job / 'status.json')
        self.assertEqual(launched, [0, 1, 2, 3])
        self.assertEqual(terminated, [])
        self.assertTrue(all(g['stage'] == 'not_launched_owned_cleanup_unconfirmed' for g in result['groups'][4:]))

    def test_queue_caps_four_and_continues_failed_group_with_final_status_reread(self):
        self.make_plan(('bulb-screw', 'install-ram', 'usb-insert'))
        self.job_ready()
        live, peaks, launched, terminated = set(), [], [], []
        test = self
        class FakeProcess:
            def __init__(self, index):
                self.index, self.pid = index, 10000000 + index
            def poll(self):
                if self.index not in live:
                    return 1 if self.index == 0 else 0
                live.remove(self.index)
                output = driver.output_dir(test.plan, self.index)
                driver.write(output / 'group_status.json', dict(status='failed' if self.index == 0 else 'complete',
                    stage='complete', owned_cleanup_confirmed=True, valid_trials=0 if self.index == 0 else 20))
                return 1 if self.index == 0 else 0
            def terminate(self):
                terminated.append(self.index)
            def wait(self, timeout):
                return self.poll()
        def launch(_plan, index):
            output = driver.output_dir(self.plan, index)
            output.mkdir(parents=True)
            driver.write(output / 'group_status.json', dict(status='running', stage='evaluate'))
            live.add(index)
            launched.append(index)
            peaks.append(len(live))
            return FakeProcess(index)
        with patch.object(driver, 'port_free', return_value=True), patch.object(driver, 'launch_worker', side_effect=launch), \
                patch.object(driver.time, 'sleep'):
            self.assertEqual(driver.run_controller(self.plan), 1)
        result = driver.read(self.job / 'status.json')
        self.assertEqual(launched, list(range(6)))
        self.assertEqual(max(peaks), 4)
        self.assertEqual(result['groups'][0]['status'], 'failed')
        self.assertTrue(all(g['status'] == 'complete' for g in result['groups'][1:]))
        self.assertEqual(terminated, [])


if __name__ == '__main__':
    unittest.main()
