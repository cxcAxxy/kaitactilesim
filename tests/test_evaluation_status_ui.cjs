'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const sourcePath = path.join(__dirname, '../scripts/analysis/evaluation_status/web/status.js');
const ui = require(sourcePath);

function group(overrides = {}) {
  return {
    id: 'group', task: 'usb-insert', model: 'pi05', checkpoint: '20000',
    prediction_horizon: 50, execute_steps: 5, source_kind: 'archive', mode: 'diagnostic',
    protocol: {max_sim_seconds: 90, guard_mode: 'observe', controller: 'native', action_interpolation: 'linear'},
    trials: [], ...overrides,
  };
}

test('latest job groups sort before archives without ranking across protocols', () => {
  const rows = [
    group({id: 'archive', source_kind: 'archive'}),
    group({id: 'job-12s', source_kind: 'job', protocol: {max_sim_seconds: 12}}),
    group({id: 'job-2s', source_kind: 'job', protocol: {max_sim_seconds: 2}}),
  ];
  assert.deepEqual(ui.filterGroups(rows, {task: '', model: '', checkpoint: '', source_kind: '', mode: ''}).map(row => row.id),
    ['job-2s', 'job-12s', 'archive']);
  assert.equal(rows.length, 3, 'protocol-distinct groups must remain separate');
});

test('task, model, checkpoint, source and mode filters are conjunctive', () => {
  const rows = [
    group({id: 'wanted', task: 'poker-draw', model: 'ftp1', checkpoint: 'best', source_kind: 'job'}),
    group({id: 'wrong-mode', task: 'poker-draw', model: 'ftp1', checkpoint: 'best', source_kind: 'job', mode: 'formal'}),
    group({id: 'wrong-model', task: 'poker-draw', model: 'pi05+cpm', checkpoint: 'best', source_kind: 'job'}),
  ];
  const state = {task: 'poker-draw', model: 'ftp1', checkpoint: 'best', source_kind: 'job', mode: 'diagnostic'};
  assert.deepEqual(ui.filterGroups(rows, state).map(row => row.id), ['wanted']);
});

test('coverage retains all declared tasks including zero-record tasks', () => {
  const tasks = Array.from({length: 13}, (_, index) => ({task: `task-${index}`, title: `任务 ${index}`}));
  const coverage = ui.taskCoverage(tasks, [group({task: 'task-0', trials: [{}, {}]}), group({task: 'task-0', trials: [{}]})]);
  assert.equal(coverage.length, 13);
  assert.deepEqual(coverage[0], {...tasks[0], group_count: 2, trial_count: 3});
  assert.deepEqual(coverage[12], {...tasks[12], group_count: 0, trial_count: 0});
});

test('validity and native outcomes remain distinct', () => {
  assert.equal(ui.nativeLabel({valid: true, native_success: true}), '原生成功');
  assert.equal(ui.nativeLabel({valid: true, native_success: false}), '原生未完成');
  assert.equal(ui.nativeLabel({valid: true, native_success: null}), '原生结果未知');
  assert.equal(ui.nativeLabel({valid: false, native_success: true}), '无效记录 · 原生结果不计');
  assert.equal(ui.nativeLabel({valid: null, native_success: false}), '有效性未知 · 原生结果未知');
});

test('protocol presentation keeps duration and controller distinctions visible', () => {
  const two = ui.protocolLabel({max_sim_seconds: 2, guard_mode: 'observe', controller: 'a'});
  const twelve = ui.protocolLabel({max_sim_seconds: 12, guard_mode: 'observe', controller: 'a'});
  const strict = ui.protocolLabel({max_sim_seconds: 2, guard_mode: 'strict', controller: 'a'});
  assert.match(two, /时长: 2s/);
  assert.match(twelve, /时长: 12s/);
  assert.match(strict, /守卫: strict/);
  assert.notEqual(two, twelve);
  assert.notEqual(two, strict);
});

test('links reject active content, escapes and non-http protocols', () => {
  assert.equal(ui.safeUrl('details/trial/report.html', 'http://localhost/status/'),
    'http://localhost/status/details/trial/report.html');
  assert.equal(ui.safeUrl('../kaitactilesim_official/case/report.html', 'http://localhost/status/'),
    'http://localhost/kaitactilesim_official/case/report.html');
  assert.equal(ui.safeUrl('https://127.0.0.1:8876/history/', 'http://localhost/'),
    'https://127.0.0.1:8876/history/');
  for (const value of ['javascript:alert(1)', 'data:text/html,x', '../../secret',
    'x/../secret', '%2e%2e/secret', 'x\\report.html', 'x\nreport.html']) {
    assert.equal(ui.safeUrl(value, 'http://localhost/status/'), null, value);
  }
});

test('page source uses textContent and no arbitrary HTML injection', () => {
  const source = fs.readFileSync(sourcePath, 'utf8');
  assert.doesNotMatch(source, /\.innerHTML\s*=|insertAdjacentHTML|\beval\s*\(|new Function\s*\(/);
  assert.match(source, /textContent/);
  assert.match(source, /不跨组汇总成功率/);
  assert.match(source, /无效记录不进入成功分母/);
  assert.match(source, /待训练、归一化与运行任务/);
  assert.match(source, /来源标记 formal/);
  assert.match(source, /id = 'es-visible-count'/);
  assert.match(source, /id = 'es-groups'/);
  assert.match(source, /`es-\$\{key\.replace/);
  assert.match(source, /相关评估资料/);
});
