'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const sourcePath = path.join(__dirname, '../scripts/analysis/evaluation_status/web/experiment.js');
const cssPath = path.join(__dirname, '../scripts/analysis/evaluation_status/web/experiment.css');
const ui = require(sourcePath);

function trials(count, overrides = {}) {
  return Array.from({length: count}, (_, seed) => ({seed, valid: true,
    native_success: false, status: 'task_not_completed', ...overrides}));
}
function group(overrides = {}) {
  return {id: 'group', task: 'usb-insert', title: 'USB 插入', model: 'pi05', checkpoint: '10000',
    prediction_horizon: 50, execute_steps: 50, source_kind: 'job', mode: 'diagnostic',
    publication_status: 'eligible', valid_trials: 5, invalid_trials: 0,
    native_success_count: 0, native_known_count: 5,
    complete_stage_success_count: 0, complete_stage_known_count: 5,
    protocol: {rule_sha256: 'a'.repeat(64)}, trials: trials(5), ...overrides};
}

test('450-member experiment keeps 430 scoring records and 20 quarantined replays distinct', () => {
  const groups = [
    group({id: '10k', checkpoint: '10000', trials: trials(45), valid_trials: 45}),
    group({id: '15k', checkpoint: '15000', trials: trials(45), valid_trials: 45}),
    group({id: '20k', checkpoint: '20000', trials: trials(160), valid_trials: 160}),
    group({id: '20k-q', checkpoint: '20000', mode: 'quarantined',
      publication_status: 'quarantined_replay_only', valid_trials: 0, invalid_trials: 20,
      native_success_count: 0, native_known_count: 0,
      complete_stage_success_count: 0, complete_stage_known_count: 0,
      trials: trials(20, {valid: false, native_success: null, recorded_native_success: false, status: 'quarantined'})}),
    group({id: '24999', checkpoint: '24999', trials: trials(180), valid_trials: 180}),
  ];
  assert.deepEqual(ui.derivedCounts(groups), {member: 450, eligible: 430, quarantined: 20});
  assert.deepEqual(ui.checkpointSummary({checkpoint_labels: ['10000', '15000', '20000', '24999']}, groups), [
    {checkpoint: '10000', member: 45, eligible: 45, quarantined: 0},
    {checkpoint: '15000', member: 45, eligible: 45, quarantined: 0},
    {checkpoint: '20000', member: 180, eligible: 160, quarantined: 20},
    {checkpoint: '24999', member: 180, eligible: 180, quarantined: 0},
  ]);
  assert.deepEqual(ui.experimentCounts({experiment: {
    member_trials: 450, eligible_trials: 430, quarantined_trials: 20}}),
  {member: 450, eligible: 430, quarantined: 20});
});

test('legacy count aliases remain readable without changing the new contract', () => {
  assert.deepEqual(ui.experimentCounts({experiment: {
    total_records: 450, eligible_records: 430, quarantined_records: 20}}),
  {member: 450, eligible: 430, quarantined: 20});
});

test('task and checkpoint filters are conjunctive and ordering is task then checkpoint', () => {
  const rows = [
    group({id: 'usb-24999', task: 'usb-unplug', checkpoint: '24999'}),
    group({id: 'bulb-15000', task: 'bulb-screw', checkpoint: '15000'}),
    group({id: 'usb-10000', task: 'usb-unplug', checkpoint: '10000'}),
    group({id: 'bulb-10000', task: 'bulb-screw', checkpoint: '10000'}),
  ];
  const taskOrder = ['bulb-screw', 'usb-unplug'];
  const checkpoints = ['10000', '15000', '20000', '24999'];
  assert.deepEqual(ui.sortGroups(rows, taskOrder, checkpoints).map(row => row.id),
    ['bulb-10000', 'bulb-15000', 'usb-10000', 'usb-24999']);
  assert.deepEqual(ui.filterGroups(rows, {task: 'usb-unplug', checkpoint: '10000'}, taskOrder, checkpoints)
    .map(row => row.id), ['usb-10000']);
});

test('zero denominators and quarantined native outcomes never appear as zero percent', () => {
  const quarantined = group({mode: 'quarantined', publication_status: 'quarantined_replay_only'});
  const trial = {valid: false, native_success: null, recorded_native_success: true, status: 'quarantined'};
  assert.equal(ui.denominatorLabel(0, 0), '未定义');
  assert.equal(ui.denominatorLabel(0, null), '未定义');
  assert.equal(ui.denominatorLabel(2, 5), '2 / 5');
  assert.equal(ui.trialLabel(quarantined, trial), '隔离回看 · 不计成绩');
  assert.equal(ui.isEligible(quarantined, trial), false);
});

test('USB completion-rule generations remain visibly distinct', () => {
  const historical = group({task: 'usb-unplug', stages: [
    {name: 'fully_extracted'}, {name: 'released_stably_on_table'}]});
  const current = group({task: 'usb-unplug', stages: [
    {name: 'fully_extracted'}, {name: 'extracted_and_held_2s'}]});
  assert.match(ui.ruleLabel(historical), /历史规则.*放回桌面/);
  assert.match(ui.ruleLabel(current), /新版规则.*手持.*2 秒/);
  assert.notEqual(ui.ruleLabel(historical), ui.ruleLabel(current));
});

test('task coverage excludes unrelated empty tasks and counts quarantine separately', () => {
  const normal = group({task: 'whiteboard-wipe-horizontal-line', title: '水平白板线迹擦拭', trials: trials(30)});
  const quarantined = group({task: 'whiteboard-wipe-horizontal-line', title: '水平白板线迹擦拭',
    publication_status: 'quarantined_replay_only', mode: 'quarantined',
    trials: trials(20, {valid: false, status: 'quarantined'})});
  const coverage = ui.taskCoverage([
    {task: 'whiteboard-wipe-horizontal-line', title: '水平白板线迹擦拭'},
    {task: 'vase-wipe', title: '花瓶擦拭'},
  ], [normal, quarantined]);
  assert.equal(coverage.length, 1);
  assert.deepEqual(coverage[0], {task: 'whiteboard-wipe-horizontal-line', title: '水平白板线迹擦拭',
    group_count: 2, member_trials: 50, eligible_trials: 30, quarantined_trials: 20});
});

test('links are constrained to safe report and sibling URLs', () => {
  assert.equal(ui.safeUrl('trials/a/index.html', 'http://localhost/status/'),
    'http://localhost/status/trials/a/index.html');
  assert.equal(ui.safeUrl('../egotouch.html', 'http://localhost/status/'),
    'http://localhost/egotouch.html');
  for (const value of ['javascript:alert(1)', 'data:text/html,x', '../../secret',
    'x/../secret', '%2e%2e/secret', 'x\\report.html']) {
    assert.equal(ui.safeUrl(value, 'http://localhost/status/'), null, value);
  }
});

test('new assets expose stable QA selectors and never inject arbitrary HTML', () => {
  const source = fs.readFileSync(sourcePath, 'utf8');
  const css = fs.readFileSync(cssPath, 'utf8');
  assert.doesNotMatch(source, /\.innerHTML\s*=|insertAdjacentHTML|\beval\s*\(|new Function\s*\(/);
  for (const selector of ['kex-member-count', 'kex-eligible-count', 'kex-quarantine-count',
    'kex-checkpoints', 'kex-task-grid', 'kex-visible-count', 'kex-clear-filters', 'kex-groups']) {
    assert.match(source, new RegExp(selector));
  }
  assert.match(source, /白板“拾取并稳定持有”不是擦净或完整任务成功/);
  assert.match(source, /不恢复有效性、原生成功或阶段分母/);
  assert.match(source, /不跨协议汇总成功率/);
  assert.match(css, /@media \(max-width: 760px\)/);
  assert.match(css, /\.kex-group-quarantined/);
});
