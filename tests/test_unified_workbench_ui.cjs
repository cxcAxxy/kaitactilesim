'use strict';
const assert = require('node:assert/strict');
const ui = require('../scripts/analysis/evaluation_status/web/unified_workbench.js');

function trial(seed, native, stages, extra = {}) {
  return {id: 'seed_' + seed, seed, valid: true, native_success: native,
    publication_status: 'eligible', staged_evidence_status: 'valid', staged_evidence_basis: 'recorded_archive',
    complete_stage_success: stages.every(Boolean), stages: stages.map((reached, i) => ({key: 's' + i, name: '阶段' + i, reached})), ...extra};
}
const first = {id: 'pi05_20k', task: 'card', model: 'pi05', checkpoint: '20000',
  prediction_horizon: 50, execute_steps: 5, cohort_id: 'h50', protocol: {guard_mode: 'stop'},
  rule_identity: 'h50-rule', trials: [trial(0, true, [true, true]), trial(1, false, [true, false]),
    trial(2, null, [false, false]), trial(3, true, [true, true], {valid: false, publication_status: 'quarantined'})]};
const second = {id: 'ftp_20k', task: 'card', model: 'ftp1', checkpoint: '20000',
  prediction_horizon: 32, execute_steps: 5, cohort_id: 'tactile', protocol: {guard_mode: 'observe'},
  rule_identity: 'tactile-rule', trials: [trial(0, false, [true, false], {staged_evidence_basis: 'tactile_manifest_inventory_v1'})]};
const cpm = {id: 'cpm', task: 'card', model: 'pi05+cpm', checkpoint: 'best',
  prediction_horizon: 30, execute_steps: 5, cohort_id: 'tactile', protocol: {guard_mode: 'observe'},
  trials: [trial(0, false, [true, true], {staged_evidence_status: 'invalid', complete_stage_success: true})]};
const white = {id: 'white', task: 'white-easy', model: 'pi05', checkpoint: '24999',
  trials: [trial(0, false, [true], {complete_stage_success: null})]};
const groups = [first, second, cpm, white];

assert.deepEqual(ui.summarizeGroup(first), {
  members: 4, eligible: 3, excluded: 1, native: {known: 2, successes: 1},
  complete: {known: 3, successes: 1}, stageEvidence: 3,
  stages: [{key: 's0', name: '阶段0', known: 3, successes: 2}, {key: 's1', name: '阶段1', known: 3, successes: 1}],
});
assert.deepEqual(ui.summarizeGroup(cpm).native, {known: 1, successes: 0});
assert.deepEqual(ui.summarizeGroup(cpm).complete, {known: 0, successes: 0});
assert.equal(ui.summarizeGroup(cpm).stages[0].known, 0);
assert.equal(ui.summarizeGroup(cpm).stages[0].successes, 0);
assert.equal(ui.stageCategory(cpm.trials[0]), 'unknown');
assert.equal(ui.stageCategory(first.trials[3]), 'unknown');
assert.equal(ui.nativeOutcome(first.trials[3]), 'invalid');
assert.equal(ui.eligible(trial(9, true, [true], {publication_status: 'quarantined_replay_only'})), false);
assert.equal(ui.nativeOutcome(trial(9, true, [true], {publication_status: 'quarantined_replay_only'})), 'invalid');
assert.equal(ui.nativeOutcome(first.trials[2]), 'unknown');
assert.deepEqual(ui.summarizeGroup(white).native, {known: 1, successes: 0});
assert.deepEqual(ui.summarizeGroup(white).complete, {known: 0, successes: 0});
assert.equal(ui.summarizeGroup(white).stages[0].successes, 1, 'whiteboard pickup is not native success');
assert.equal(ui.summarizeGroup({trials: []}).native.known, 0);
assert.equal(ui.ratioLabel({known: 0, successes: 0}), '未定义');
assert.equal(ui.ratioLabel({known: 5, successes: 0}), '0 / 5 (0.0%)');
assert.equal(ui.ratioLabel({known: 3, successes: 1}), '1 / 3 (33.3%)');
assert.equal(ui.ratioLabel({known: 575, successes: 16}), '16 / 575 (2.8%)');
assert.equal(ui.ratioLabel({known: 320, successes: 3}), '3 / 320 (0.9%)');

assert.equal(ui.filterGroups(groups, {}).length, 4);
assert.equal(ui.filterGroups(groups, {task: 'card'}).length, 3);
assert.deepEqual(ui.filterGroups(groups, {model: 'ftp1'}), [second]);
assert.deepEqual(ui.filterGroups(groups, {checkpoint: '20000'}), [first, second]);
assert.deepEqual(ui.filterGroups(groups, {cohort: 'h50'}), [first]);
assert.deepEqual(ui.filterGroups(groups, {protocol: ui.protocolKey(second)}), [second]);
assert.notEqual(ui.protocolKey(first), ui.protocolKey(second));
assert.equal(ui.protocolKey({protocol: {a: 1, b: 2}}), ui.protocolKey({protocol: {b: 2, a: 1}}));
assert.deepEqual(ui.filterGroups(groups, {outcome: 'pass', search: 'seed_0', stage: 'complete'}), groups,
  'outcome/stage/search must never choose the success-rate denominator');
assert.equal(ui.selectedTrials(groups, {model: 'pi05', outcome: 'pass'}).length, 1);
assert.equal(ui.selectedTrials(groups, {model: 'pi05', outcome: 'invalid'}).length, 1);
assert.equal(ui.selectedTrials(groups, {task: 'card', stage: 'unknown'}).length, 2);
assert.equal(ui.selectedTrials(groups, {search: 'seed_0001'}).length, 1);
assert.equal(ui.selectedTrials(groups, {search: 'ftp1'}).length, 1);
assert.equal(ui.stageCategory(first.trials[0]), 'complete');
assert.equal(ui.stageCategory(first.trials[1]), 'partial');
assert.equal(ui.stageCategory(first.trials[2]), 'none');
assert.equal(ui.stageCategory(trial(1, false, [null], {complete_stage_success: null})), 'unknown');
assert.equal(ui.stageEligible(trial(1, false, [true], {staged_evidence_status: undefined})), false);
assert.equal(ui.evidenceBasisLabel('recorded_archive'), '归档阶段记录');
assert.doesNotMatch(ui.evidenceBasisLabel('recorded_archive'), /严格|校验/);
assert.match(ui.protocolLabel(second), /H32.*K5.*guard=observe/);
assert.equal(ui.modelLabel('pi05+cpm'), 'CPM（openpi-main + π0.5）');
assert.equal(ui.stageLabel({name: 'dragged_on_table'}), '拖牌');
assert.equal(ui.stageLabel({name: '拿起板擦'}), '拿起板擦');
assert.equal(ui.stageLabel({name: 'picked_up'}, 'whiteboard-wipe'), '抓起板擦');
assert.equal(ui.stageLabel({name: 'fully_extracted'}, 'bulb-unscrew'), '完全拧出');
assert.equal(ui.stageLabel({name: 'picked_up', label: '原归档阶段名'}), '原归档阶段名');
assert.equal(ui.stageLabel({name: 'new_unrecognized_stage'}), 'new_unrecognized_stage');
assert.equal(ui.checkpointLabel('24999'), 'step 24999');
assert.equal(ui.checkpointLabel('best'), 'best');

const base = 'http://127.0.0.1:8877/unified/index.html';
assert.equal(ui.safeUrl('../h50/card/report.html', base), 'http://127.0.0.1:8877/h50/card/report.html');
assert.equal(ui.safeUrl('trials/seed_0/index.html', base), 'http://127.0.0.1:8877/unified/trials/seed_0/index.html');
assert.equal(ui.safeUrl('javascript:alert(1)', base), null);
assert.equal(ui.safeUrl('https://example.com/', base), null);
assert.equal(ui.safeUrl('//example.com/', base), null);
assert.equal(ui.safeUrl('%2e%2e/escape.html', base), null);
assert.equal(ui.safeUrl('..%2fescape.html', base), null);
assert.equal(ui.safeUrl('..\\escape.html', base), null);
assert.equal(ui.safeUrl('../../escape.html', 'http://127.0.0.1:8877/reports/unified/index.html'), null);
assert.equal(ui.schema, 'kaihand-unified-evaluation-workbench-v1');

// The production-sized shape: nine H50 tasks, six EgoTouch groups and five tactile groups.
const fixture = [];
const fixtureTasks = ['card', 'ram', 'usb-in', 'usb-out', 'bulb-in', 'bulb-out', 'white-easy', 'white-mid', 'white-hard'];
fixtureTasks.forEach(task => ['10000', '15000', '20000', '24999'].forEach(checkpoint => {
  const count = ['10000', '15000'].includes(checkpoint) ? 5 : 20;
  const quarantined = task === 'white-easy' && checkpoint === '20000';
  fixture.push({id: task + checkpoint, task, model: 'pi05', checkpoint,
    trials: Array.from({length: count}, (_, seed) => trial(seed, false, [false], quarantined ?
      {valid: false, publication_status: 'quarantined_replay_only'} : {}))});
}));
for (let i = 0; i < 6; i += 1) fixture.push({id: 'ego' + i, task: 'pick-place', model: 'egotouch', checkpoint: String(i),
  trials: Array.from({length: 20}, (_, seed) => trial(seed, false, [], {staged_evidence_status: 'unknown', complete_stage_success: null}))});
fixture.push({id: 'cpm', task: 'card', model: 'pi05+cpm', checkpoint: 'best',
  trials: Array.from({length: 5}, (_, seed) => trial(seed, false, [true], {staged_evidence_status: 'invalid'}))});
['10000', '15000', '20000', '24999'].forEach(checkpoint => fixture.push({id: 'ftp' + checkpoint,
  task: 'card', model: 'ftp1', checkpoint, trials: Array.from({length: 5}, (_, seed) => trial(seed, false, [false]))}));
assert.equal(fixture.length, 47);
assert.equal(ui.defaultSummaryMode, 'policy');
assert.equal(ui.summaryRows(fixture).length, 47, 'the default must preserve policy/checkpoint rows, not task-mixed rows');
assert.equal(ui.summaryRows(fixture, 'unrecognized').length, 47);
assert.equal(ui.summaryRows(fixture, 'task').length, 10);
assert.equal(ui.summaryRows(fixture, 'policy').length, 47);
const sortedPolicyRows = ui.summaryRows(fixture);
assert.equal(sortedPolicyRows.slice(0, 9).every(group => group.task === 'card'), true,
  'all nine poker policy/checkpoint groups must be adjacent');
assert.equal(sortedPolicyRows.slice(9).some(group => group.task === 'card'), false);
assert.deepEqual(sortedPolicyRows.slice(0, 9).map(group => [group.model, group.checkpoint]),
  [['pi05', '10000'], ['pi05', '15000'], ['pi05', '20000'], ['pi05', '24999'], ['pi05+cpm', 'best'],
    ['ftp1', '10000'], ['ftp1', '15000'], ['ftp1', '20000'], ['ftp1', '24999']]);
assert.equal(fixture[9].task, 'usb-in', 'display sorting must not mutate the source catalog order');
assert.equal(sortedPolicyRows.every(group => fixture.includes(group)), true, 'display sorting must preserve original groups');
assert.match(ui.coverageLabel(fixture), /595.*575.*20.*47/s);
assert.doesNotMatch(ui.coverageLabel(fixture), /\d\s*\/\s*\d|\d(?:\.\d+)?%/,
  'the overview must not present any mixed success rate');
assert.match(ui.coverageLabel([]), /0 条归档/);
assert.deepEqual(ui.selectionSummary(fixture), {members: 595, eligible: 575, excluded: 20,
  native: {known: 575, successes: 0}, complete: {known: 450, successes: 0}});
assert.equal(ui.summarizeGroup(ui.taskSummaryRows(fixture).find(group => group.task === 'card')).stages[0].known, 70,
  'the five CPM invalid stages must not enter the task-stage denominator');
const ramModelOptions = ui.modelOptions(fixture, 'ram');
assert.equal(ramModelOptions.length, 4, 'every task must retain all global model choices');
assert.equal(ramModelOptions.find(option => option.value === 'pi05+cpm').current, 0);
assert.equal(ramModelOptions.find(option => option.value === 'ftp1').current, 0);
assert.deepEqual(ui.filterGroups(fixture, {task: 'ram', model: 'pi05+cpm'}), [],
  'a RAM + CPM selection is unavailable, not an instruction to fall back to all models');
assert.equal(ui.taskSummaryRows(fixture).find(group => group.task === 'ram').trials.length, 50,
  'stage keys shared across tasks must not merge task rows');

const held = {id: 'held', task: 'usb-out', trials: Array.from({length: 10}, (_, seed) => trial(seed, false, [], {
  complete_stage_success: seed < 3,
  stages: [{key: 'stage1', name: 'fully_extracted', reached: true},
    {key: 'stage2', name: 'extracted_and_held_2s', reached: seed < 3}]}))};
const released = {id: 'released', task: 'usb-out', trials: Array.from({length: 40}, (_, seed) => trial(seed, false, [], {
  complete_stage_success: false,
  stages: [{key: 'stage1', name: 'fully_extracted', reached: seed < 14},
    {key: 'stage2', name: 'released_stably_on_table', reached: false}]}))};
const usbRow = ui.taskSummaryRows([held, released])[0];
assert.deepEqual(usbRow.complete_breakdowns.map(value => [value.terminal_stage.name, value.successes, value.known]),
  [['extracted_and_held_2s', 3, 10], ['released_stably_on_table', 0, 40]]);
assert.deepEqual(ui.summarizeGroup(usbRow).stages.map(value => [value.name, value.successes, value.known]),
  [['fully_extracted', 24, 50], ['extracted_and_held_2s', 3, 10], ['released_stably_on_table', 0, 40]],
  'the same stage2 key must not merge two different meanings');
assert.notEqual(ui.stageIdentity({key: 'stage2', name: 'extracted_and_held_2s'}, 'usb-out'),
  ui.stageIdentity({key: 'stage2', name: 'released_stably_on_table'}, 'usb-out'));

const metricGroups = [
  {task: 'white', trials: [trial(0, false, [true]), trial(1, false, [true])],
    metrics: [{key: 'cleaned_fraction', value: 10, known: 2, unit: '%'}]},
  {task: 'white', trials: [trial(0, false, [true])],
    metrics: [{key: 'cleaned_fraction', value: 40, known: 1, unit: '%'}]},
  {task: 'white', trials: [trial(0, false, [true], {valid: false, publication_status: 'quarantined_replay_only'})],
    metrics: [{key: 'cleaned_fraction', value: 1000, known: 1, unit: '%'}]},
];
assert.equal(ui.aggregateMetrics(metricGroups)[0].value, 20, 'group means must be episode-count weighted');
assert.equal(ui.aggregateMetrics(metricGroups)[0].known, 3, 'quarantined metric values cannot enter the mean');

// Minimal DOM exercise: verify the actual renderer, not only the statistics helpers.
class TestElement {
  constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; this.listeners = {}; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children.forEach(child => { child.parentNode = null; }); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  appendChild(child) {
    if (child.parentNode) child.parentNode.children = child.parentNode.children.filter(item => item !== child);
    child.parentNode = this; this.children.push(child); return child;
  }
  insertBefore(child, before) {
    const index = this.children.indexOf(before);
    child.parentNode = this; this.children.splice(index, 0, child); return child;
  }
  replaceChildren(...children) { this.textContent = ''; children.forEach(child => this.appendChild(child)); }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  emit(event) { this.listeners[event]({target: this}); }
}
const testElements = new Map();
const testRoot = new TestElement('main');
['ks-trials', 'ks-visible', 'ks-task-tabs', 'ks-comparison', 'ks-counts', 'ks-model', 'ks-checkpoint',
  'ks-protocol', 'ks-cohort', 'ks-outcome', 'ks-stage', 'ks-search', 'ks-summary-mode', 'ks-summary-note',
  'ks-summary-total', 'ks-task-note'].forEach(id => {
  const node = new TestElement('div'); node.id = id; testElements.set(id, node); testRoot.appendChild(node);
});
const doc = {baseURI: 'http://127.0.0.1:8877/new/index.html',
  getElementById: id => testElements.get(id) || null, createElement: tag => new TestElement(tag)};
const controller = ui.mount({schema: ui.schema, groups: fixture,
  tasks: [...new Set(fixture.map(group => group.task))].map(task => ({task, title: task}))}, doc);
const element = id => testElements.get(id);
assert.equal(element('ks-summary-mode').value, 'policy');
assert.equal(element('ks-comparison').children.length, 47);
assert.equal(element('ks-trials').children.length, 595);
assert.equal(element('ks-comparison').children.every(row => row.children.length === 8), true);
assert.equal(element('ks-comparison').children[0].children[1].children[0].textContent, 'π0.5');
assert.equal(element('ks-comparison').children[0].children[2].children[0].textContent, 'step 10000');
assert.match(element('ks-comparison').children[0].children[2].className, /ks-summary-checkpoint/);
assert.doesNotMatch(element('ks-summary-total').textContent, /\d\s*\/\s*\d|\d(?:\.\d+)?%/);
element('ks-summary-mode').value = 'task'; element('ks-summary-mode').emit('change');
assert.equal(element('ks-comparison').children.length, 10);
assert.equal(element('ks-comparison').children.every(row => row.children.length === 8), true);
assert.equal(element('ks-trials').children.length, 595, 'summary mode must not change the underlying trial list');
element('ks-summary-mode').value = 'policy'; element('ks-summary-mode').emit('change');
element('ks-task-tabs').children.find(button => button.dataset.task === 'ram').emit('click');
assert.equal(element('ks-model').children.filter(option => option.value).length, 4);
element('ks-model').value = 'pi05+cpm'; element('ks-model').emit('change');
assert.equal(controller.state.model, 'pi05+cpm');
assert.equal(element('ks-model').value, 'pi05+cpm');
assert.match(element('ks-comparison').textContent, /该任务暂无该模型评估记录/);
assert.equal(element('ks-comparison').children[0].children[0].colSpan, 8);
console.log('unified workbench UI tests passed');
