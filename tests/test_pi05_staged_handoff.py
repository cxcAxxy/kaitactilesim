"""Handoff safety tests; only temporary fixtures and owned probe processes."""
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_pi05_staged_groups as staged_fixtures

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/deploy/handoff_pi05_staged_queue.py'
SPEC = importlib.util.spec_from_file_location('staged_handoff_under_test', SCRIPT)
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = staged_fixtures.StagedDriverTests('test_authorized_subset_and_all_nine_pairs')
        self.fixture.setUp()
        self.fixture.make_plan(('whiteboard-wipe-horizontal-line', 'bulb-screw', 'usb-insert', 'usb-unplug'))
        self.fixture.job_ready()
        self.plan, self.driver = self.fixture.plan, sys.modules['staged_groups_under_test'] if 'staged_groups_under_test' in sys.modules else None
        # The fixture module keeps its fully implemented production driver.
        import test_pi05_staged_groups
        self.driver = test_pi05_staged_groups.driver
        self.job = self.fixture.job
        self.owner = dict(pid=50000000, ppid=1, starttime=1234, state='S', argv=[str(handoff.CONTROLLER_PYTHON),
            str(self.fixture.sim / 'scripts/deploy/run_pi05_staged_groups.py'), '--plan', str(self.job / 'plan.json')])
        self.status = dict(status='running', run_id=self.plan['run_id'], staged_evaluation_version='staged_v1',
            pid=self.owner['pid'], identity=self.owner.copy(), errors=[], settings=self.plan['settings'],
            groups=[dict(g, group_index=i, status='running' if i < 4 else 'queued') for i, g in enumerate(self.plan['groups'])])
        self.processes = {self.owner['pid']: self.owner}
        for index in range(4):
            info = self.worker(index)
            self.processes[info['pid']] = info
            self.status['groups'][index].update(worker_pid=info['pid'], worker_identity=info.copy())
        handoff.write(self.job / 'status.json', self.status)
        self.proof = {'plan_sha256': handoff.sha(self.job / 'plan.json'), 'fixture': 'unchanged'}
        self.oldlock = (self.job / 'controller.lock').open('w')
        fcntl.flock(self.oldlock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def tearDown(self):
        if not self.oldlock.closed:
            self.oldlock.close()
        self.fixture.tearDown()

    def worker(self, index, state='S'):
        return dict(pid=50000100 + index, ppid=self.owner['pid'], starttime=2000 + index, state=state,
            argv=[str(self.fixture.sim / '.pixi/envs/default/bin/python'),
                str(self.fixture.sim / 'scripts/deploy/run_pi05_staged_groups.py'), '--plan',
                str(self.job / 'plan.json'), '--worker-index', str(index)])

    def context(self, *_args):
        return self.plan, self.driver, self.owner.copy(), self.status.copy(), self.proof

    def snapshot(self):
        return dict(status=json.loads(json.dumps(self.status)),
            adopted={str(i): self.worker(i) for i in range(4)}, extra_indices=[4, 5], owned_run_processes={})

    def synthetic_signal(self, expected, signum, **_kwargs):
        self.assertEqual(expected['pid'], self.owner['pid'])
        if signum == signal.SIGSTOP:
            self.owner['state'] = 'T'
        elif signum == signal.SIGCONT:
            self.owner['state'] = 'S'

    def test_identity_missing_starttime_or_reused_pid_is_rejected(self):
        with patch.object(handoff, 'proc', return_value=self.owner):
            with self.assertRaisesRegex(ValueError, 'identity'):
                handoff.exact_process({'pid': self.owner['pid']})
            with self.assertRaisesRegex(ValueError, 'reused'):
                handoff.exact_process({**self.owner, 'starttime': 999})

    def test_fresh_group_refuses_existing_log_output_or_archive(self):
        for target in (self.job / 'group_4.log', self.driver.output_dir(self.plan, 4),
                       Path(self.plan['groups'][4]['archive_dir'])):
            with self.subTest(target=str(target)):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('existing attempt')
                with self.assertRaisesRegex(ValueError, 'claimed/started/published'):
                    handoff.guard_fresh_group(self.plan, self.driver, 4)
                target.unlink()

    def test_claim_is_exclusive_and_preserves_original_claim(self):
        directory = self.job / 'test-handoff'
        (directory / 'claims').mkdir(parents=True)
        path = handoff.claim(directory, 4, {'owner': 'first'})
        with self.assertRaises(FileExistsError):
            handoff.claim(directory, 4, {'owner': 'second'})
        self.assertEqual(handoff.read(path), {'owner': 'first'})

    def test_paused_snapshot_recovers_unreported_replacement_and_exited_tail(self):
        self.owner['state'] = 'T'
        self.processes[self.worker(0)['pid']] = None
        self.processes[self.worker(1)['pid']]['state'] = 'Z'
        self.processes[self.worker(1)['pid']]['argv'] = []
        self.processes[self.worker(4)['pid']] = self.worker(4)
        children_path = f"/proc/{self.owner['pid']}/task/{self.owner['pid']}/children"
        original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == children_path:
                return ' '.join(str(self.worker(i)['pid']) for i in (1, 2, 3, 4))
            return original_read(path, *args, **kwargs)
        with patch.object(handoff, 'proc', side_effect=lambda pid: self.processes.get(pid)), \
                patch.object(Path, 'read_text', read_text), patch.object(self.driver, 'port_free', return_value=True):
            snapshot = handoff.paused_snapshot(self.plan, self.driver, self.owner)
        self.assertEqual(set(snapshot['adopted']), {'0', '1', '2', '3', '4'})
        self.assertEqual(snapshot['adopted']['0']['state'], 'gone')
        self.assertEqual(snapshot['adopted']['1']['state'], 'Z')
        self.assertEqual(snapshot['adopted']['4']['starttime'], 2004)

    def test_handoff_stops_only_owner_and_does_not_change_plan_or_frozen_sources(self):
        successor = MagicMock(pid=50000999)
        successor.poll.return_value = None
        child = dict(pid=successor.pid, starttime=3000, state='S', ppid=1, argv=['probe'])
        self.processes[child['pid']] = child
        before = (self.job / 'plan.json').read_bytes()
        with patch.object(handoff, 'load_context', side_effect=self.context), \
                patch.object(handoff, 'proc', side_effect=lambda pid: self.processes.get(pid)), \
                patch.object(handoff.subprocess, 'Popen', return_value=successor), \
                patch.object(handoff, 'wait_file', return_value={'identity': child}), \
                patch.object(handoff, 'paused_snapshot', side_effect=lambda *_args: self.snapshot()), \
                patch.object(handoff, 'signal_process', side_effect=self.synthetic_signal) as signals:
            directory = handoff.handoff(self.job / 'plan.json')
        self.assertEqual(signals.call_args_list[0].args[1], signal.SIGSTOP)
        self.assertEqual(len(signals.call_args_list), 1)
        self.assertTrue((directory / 'go.json').is_file())
        self.assertEqual((self.job / 'plan.json').read_bytes(), before)
        self.assertFalse(any(self.driver.output_dir(self.plan, i).exists() for i in range(8)))
        successor.terminate.assert_not_called()

    def test_prelaunch_readiness_failure_resumes_owner_without_any_worker(self):
        successor = MagicMock(pid=50000999)
        successor.poll.return_value = None
        child = dict(pid=successor.pid, starttime=3000, state='S', ppid=1, argv=['probe'])
        self.processes[child['pid']] = child
        with patch.object(handoff, 'load_context', side_effect=self.context), \
                patch.object(handoff, 'proc', side_effect=lambda pid: self.processes.get(pid)), \
                patch.object(handoff.subprocess, 'Popen', return_value=successor), \
                patch.object(handoff, 'wait_file', side_effect=[{'identity': child}, TimeoutError('adoption failed')]), \
                patch.object(handoff, 'paused_snapshot', side_effect=lambda *_args: self.snapshot()), \
                patch.object(handoff, 'signal_process', side_effect=self.synthetic_signal) as signals:
            with self.assertRaises(TimeoutError):
                handoff.handoff(self.job / 'plan.json')
        self.assertEqual([call.args[1] for call in signals.call_args_list], [signal.SIGSTOP, signal.SIGCONT])
        successor.terminate.assert_called_once()
        records = list(self.job.glob('_scheduler_handoff_*/record.json'))
        self.assertEqual(handoff.read(records[0])['status'], 'rolled_back_before_launch')
        self.assertFalse((records[0].parent / 'go.json').exists())

    def test_partial_status_is_not_accepted_as_a_terminal_worker(self):
        path = self.job / 'partial.json'
        path.write_text('{"status":')
        self.assertIsNone(handoff.ready_read(path))

    def test_cleanup_checks_real_server_and_evaluator_exit_not_just_flags(self):
        output = self.driver.output_dir(self.plan, 0)
        (output / 'run').mkdir(parents=True)
        handoff.write(output / 'run/run_manifest.json', dict(managed_server_stopped=True,
            evaluation_process_stopped=True, server_pid=50001001, evaluation_pid=50001002))
        detail = {'owned_cleanup_confirmed': True, 'stage': 'complete'}
        with patch.object(self.driver, 'port_free', return_value=True), patch.object(handoff, 'proc', return_value=None):
            self.assertTrue(handoff.cleanup_confirmed(self.plan, self.driver, 0, detail))
        with patch.object(self.driver, 'port_free', return_value=True), \
                patch.object(handoff, 'proc', return_value=dict(pid=50001001, starttime=5, state='S')):
            self.assertFalse(handoff.cleanup_confirmed(self.plan, self.driver, 0, detail))

    def test_original_finish_term_before_cont_only_after_old_workers_exit(self):
        self.owner['state'] = 'T'
        workers = {i: self.worker(i, 'Z') for i in range(4)}
        for row in workers.values():
            self.processes[row['pid']] = row
        def finish_signal(expected, signum, **_kwargs):
            self.assertEqual(expected['pid'], self.owner['pid'])
            if signum == signal.SIGCONT:
                self.owner['state'] = 'Z'
                self.oldlock.close()
        with patch.object(handoff, 'proc', side_effect=lambda pid: self.processes.get(pid)), \
                patch.object(handoff, 'signal_process', side_effect=finish_signal) as signals:
            handoff.finish_original(self.owner, workers, lock_path=self.job / 'controller.lock')
        self.assertEqual([call.args[1] for call in signals.call_args_list], [signal.SIGTERM, signal.SIGCONT])

    def test_original_finish_rejects_live_worker_or_pid_reuse_without_signaling(self):
        identity = self.worker(0)
        for current in (identity, {**identity, 'starttime': 999, 'state': 'Z'}):
            with self.subTest(current=current), patch.object(handoff, 'proc', return_value=current), \
                    patch.object(handoff, 'signal_process') as signals:
                with self.assertRaises(ValueError):
                    handoff.finish_original(self.owner, {0: identity})
                signals.assert_not_called()

    def test_supervisor_adopts_four_adds_two_then_finishes_unique_existing_queue(self):
        directory = self.job / '_scheduler_handoff_test'
        (directory / 'claims').mkdir(parents=True)
        self.owner['state'] = 'T'
        source_before = (self.job / 'plan.json').read_bytes()
        observer_before = handoff.sha(self.fixture.sim / self.driver.OBSERVER_SOURCE)
        handoff.write(directory / 'record.json', dict(source_proof=self.proof, original_controller=self.owner,
            scheduler_sha256=handoff.sha(SCRIPT), new_workers_started=0))
        handoff.write(directory / 'activation.json', {**self.snapshot(), 'original_controller': self.owner, 'source_proof': self.proof})
        handoff.write(directory / 'go.json', {'parallel_groups': 6})
        lock = (self.job / '_scheduler_handoff.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX)
        owned_fd = os.dup(lock.fileno())
        launched, peak, new_live = [], [], set()
        test = self
        def completed(index):
            output = self.driver.output_dir(self.plan, index)
            (output / 'run/evaluation').mkdir(parents=True, exist_ok=True)
            handoff.write(output / 'group_status.json', dict(status='complete', stage='complete',
                owned_cleanup_confirmed=True, valid_trials=20, completed_trials=20))
            handoff.write(output / 'run/run_manifest.json', dict(managed_server_stopped=True,
                evaluation_process_stopped=True, server_pid=60001000 + index, evaluation_pid=60002000 + index))
            handoff.write(output / 'run/evaluation/summary.json', dict(valid_trials=20,
                results=[dict(seed=seed, valid_trial=True) for seed in range(20)]))
        class FakeProcess:
            def __init__(self, index):
                self.index, self.pid, self.polls = index, 50000100 + index, 0
            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                completed(self.index)
                new_live.discard(self.index)
                test.processes[self.pid]['state'] = 'Z'
                return 0
        def launch(_plan, index):
            self.assertNotIn(index, launched)
            self.driver.output_dir(self.plan, index).mkdir(parents=True)
            launched.append(index)
            new_live.add(index)
            info = self.worker(index)
            info['ppid'] = os.getpid()
            self.processes[info['pid']] = info
            peak.append(len(new_live) + sum(self.processes[self.worker(i)['pid']]['state'] != 'Z' for i in range(4)))
            return FakeProcess(index)
        def sleep(_seconds):
            for index in range(4):
                self.processes[self.worker(index)['pid']]['state'] = 'Z'
                completed(index)
        def finish(_owner, _adopted, **_kwargs):
            self.assertTrue(all(handoff.exited(info) for info in _adopted.values()))
            self.owner['state'] = 'Z'
            self.oldlock.close()
        try:
            with patch.object(handoff, 'load_context', side_effect=self.context), \
                    patch.object(handoff, 'proc', side_effect=lambda pid: self.processes.get(pid)), \
                    patch.object(self.driver, 'port_free', return_value=True), \
                    patch.object(self.driver, 'launch_worker', side_effect=launch), \
                    patch.object(handoff, 'finish_original', side_effect=finish), \
                    patch.object(handoff.time, 'sleep', side_effect=sleep):
                self.assertEqual(handoff.supervise(self.job / 'plan.json', directory, owned_fd), 0)
        finally:
            lock.close()
        result = handoff.read(self.job / 'status.json')
        self.assertEqual(max(peak), 6)
        self.assertEqual(launched, [4, 5, 6, 7])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['valid_trials'], 160)
        self.assertEqual(result['effective_parallel_groups'], 6)
        self.assertEqual(result['settings']['parallel_groups'], 4)
        self.assertEqual(len(list((directory / 'claims').glob('group_*.json'))), 8)
        self.assertEqual((self.job / 'plan.json').read_bytes(), source_before)
        self.assertEqual(handoff.sha(self.fixture.sim / self.driver.OBSERVER_SOURCE), observer_before)

    @unittest.skipUnless(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'pidfd unavailable')
    def test_pidfd_stop_continue_only_an_owned_short_probe(self):
        process = subprocess.Popen([str(handoff.CONTROLLER_PYTHON), '-c', 'import time; time.sleep(20)'])
        try:
            identity = handoff.proc(process.pid)
            deadline = time.monotonic() + 0.5
            while identity is None and time.monotonic() < deadline:
                time.sleep(0.01)
                identity = handoff.proc(process.pid)
            if identity is None:
                self.skipTest('Owned subprocess is not visible in this sandbox /proc view')
            handoff.signal_process(identity, signal.SIGSTOP)
            deadline = time.monotonic() + 2
            while handoff.proc(process.pid)['state'] != 'T':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            handoff.signal_process(identity, signal.SIGCONT, states={'T'})
            self.assertIsNone(process.poll())
        finally:
            # This is solely the process created in this test, not an evaluation.
            process.terminate()
            process.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
