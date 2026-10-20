'use strict';

const assert = require('node:assert/strict');
const ui = require('../scripts/analysis/evaluation_status/web/tactile_experiment.js');

const trials = [
  {seed: 0, observed: true, evidence_status: 'valid', native_success: true},
  {seed: 1, observed: true, evidence_status: 'invalid', native_success: null},
  {seed: 2, observed: false, evidence_status: 'missing', native_success: null},
  {seed: 3, observed: true, evidence_status: 'valid', native_success: false},
  {seed: 4, observed: true, evidence_status: 'missing', native_success: null},
];

assert.deepEqual(ui.countsFromTrials(trials), {
  planned: 5, observed: 4, valid: 2, invalid: 1, missing: 2,
});
assert.deepEqual(ui.aggregate([{trials}, {trials: trials.slice(0, 2)}]), {
  planned: 7, observed: 6, valid: 3, invalid: 2, missing: 2,
});

assert.equal(ui.ratioLabel({known: 0, successes: 0}), '未定义');
assert.equal(ui.ratioLabel({known: 2, successes: 1}), '1 / 2 (50.0%)');
assert.equal(ui.ratioLabel({known: 5, successes: 0}), '0 / 5 (0.0%)');
assert.equal(ui.statusLabel('running'), '运行中');
assert.equal(ui.statusLabel('partial'), '部分完成');
assert.equal(ui.stageLabel({name: 'dragged_on_table', key: 'stage1'}), '拖牌');
assert.equal(ui.stageLabel({name: 'stable_grasp', key: 'stage2'}), '稳定夹持');
assert.equal(ui.stageLabel({name: 'terminal_hold_over_3s'}), '展示姿态持牌超过 3 秒');
assert.equal(ui.stageLabel({name: 'unknown_recorded_stage'}), 'unknown_recorded_stage');
assert.equal(ui.evidenceLabel({evidence_status: 'invalid'}), '无效（不计分）');
assert.equal(ui.evidenceLabel({evidence_status: 'valid', native_success: false}), '原生未完成');

assert.match(ui.protocolLabel({max_sim_seconds: 90, guard_mode: 'observe',
  controller: 'poker-tactile-direct-joints-v1', action_interpolation: 'zoh',
  success_hold_seconds: 0.1, staged_evaluation_version: 'staged_v1'}),
/90s.*guard=observe.*zoh.*hold=0.1s.*staged_v1/);

assert.equal(ui.safeUrl('trials/cpm_best/seed_0000/index.html',
  'http://127.0.0.1:8877/poker/index.html'),
'http://127.0.0.1:8877/poker/trials/cpm_best/seed_0000/index.html');
assert.equal(ui.safeUrl('../h50/index.html',
  'http://127.0.0.1:8877/poker/index.html'),
'http://127.0.0.1:8877/h50/index.html');
assert.equal(ui.safeUrl('../../escape.html', 'http://127.0.0.1:8877/poker/index.html'), null);
assert.equal(ui.safeUrl('https://example.com/', 'http://127.0.0.1:8877/poker/index.html'), null);
assert.equal(ui.safeUrl('%2e%2e/escape.html', 'http://127.0.0.1:8877/poker/index.html'), null);

console.log('tactile experiment UI tests passed');
