'use strict';
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {localUrl, compareCatalogs, terminalAndReference, isDefaultFavicon404, isCancelledVideoRange} =
  require('../scripts/analysis/check_additive_evaluation_workbench_browser.cjs');

function catalogs() {
  const groups = Array.from({length: 47}, (_, index) => ({id: `old-${index}`, task: 'old-task', model: ['pi05', 'ftp1', 'pi05+cpm', 'egotouch'][index % 4],
    checkpoint: '20000', trials: Array.from({length: index < 31 ? 13 : 12}, (_, seed) => ({id: `old-${index}-${seed}`, seed, native_success: false}))}));
  const baseline = {counts: {members: 595}, groups};
  const added = ['15000', '18000', '21000', '24000', '24999'].map(step => ({id: `new-${step}`, task: 'bulb-screw-inplace', model: 'pi05',
    checkpoint: step, prediction_horizon: 50, execute_steps: 50, initial_state_randomization: false, trial_scope: 'fixed_initial_state_repeats',
    trials: Array.from({length: 5}, (_, seed) => ({id: `new-${step}-${seed}`, seed, valid: true,
      publication_status: 'eligible', staged_evidence_status: 'valid', stages: [{}, {}, {}, {}]}))}));
  const catalog = {schema: 'kaihand-unified-evaluation-workbench-v1', counts: {members: 620, eligible: 600, quarantined: 20},
    tasks: Array.from({length: 11}, (_, index) => ({task: `task-${index}`})), groups: [...structuredClone(groups), ...added]};
  return {catalog, baseline};
}

test('URLs only permit literal localhost without credentials or non-HTTP transports', () => {
  for (const host of ['127.0.0.1', 'localhost', '[::1]']) assert.equal(localUrl(`http://${host}:8877/report/index.html`).port, '8877');
  for (const url of ['http://192.168.0.54:8877/', 'http://example.com/', 'file:///tmp/a', 'http://user:password@localhost/'])
    assert.throws(() => localUrl(url));
  assert.equal(localUrl('../r5/index.html', 'http://localhost/r6/index.html').href, 'http://localhost/r5/index.html');
});

test('only the same-origin browser-default favicon 404 is separately recorded, never app failures', () => {
  const message = 'Failed to load resource: the server responded with a status of 404 (Report file not found)';
  const origin = 'http://127.0.0.1:8877';
  assert.equal(isDefaultFavicon404(message, {url: origin + '/favicon.ico'}, origin), true);
  for (const relative of ['/assets/workbench.js', '/media/review.mp4', '/plots/diagnostic.png', '/nested/favicon.ico', '/favicon.ico?cache=1'])
    assert.equal(isDefaultFavicon404(message, {url: origin + relative}, origin), false);
  assert.equal(isDefaultFavicon404(message, {url: 'http://localhost:8877/favicon.ico'}, origin), false);
  assert.equal(isDefaultFavicon404('JavaScript TypeError', {url: origin + '/favicon.ico'}, origin), false);
  assert.equal(isDefaultFavicon404(message.replace('404', '500'), {url: origin + '/favicon.ico'}, origin), false);
});

test('intentional cancellation requires an exact known trial movie URL, media type, and ERR_ABORTED', () => {
  const url = 'http://127.0.0.1:8877/r6/trial-a/media/rollout.mp4';
  const known = new Set([url]);
  const request = {url, resource_type: 'media', error_text: 'net::ERR_ABORTED'};
  assert.equal(isCancelledVideoRange(request, known), true);
  for (const change of [{url: url + '?changed=1'}, {url: 'http://example.com/movie.mp4'},
    {url: 'http://127.0.0.1:8877/r6/expert.mp4'}, {resource_type: 'script'},
    {error_text: 'net::ERR_CONNECTION_RESET'}, {error_text: 'net::ERR_FAILED'}])
    assert.equal(isCancelledVideoRange({...request, ...change}, known), false);
});

test('additive catalog preserves exact 47 dictionaries and exactly adds five distinct groups', () => {
  const {catalog, baseline} = catalogs();
  const result = compareCatalogs(catalog, baseline);
  assert.equal(result.retained.length, 47); assert.equal(result.added.length, 5);
});

test('non-score changes to an old group are also forbidden', () => {
  const {catalog, baseline} = catalogs();
  catalog.groups[0].warnings = ['changed'];
  assert.throws(() => compareCatalogs(catalog, baseline), /Every field/);
});

test('different checkpoints, missing members, duplicate seeds and promoted old score are rejected', () => {
  const changes = [
    catalog => { catalog.groups[47].checkpoint = '12000'; },
    catalog => { catalog.counts.members = 619; },
    catalog => { catalog.groups[47].trials[0].seed = 1; },
    catalog => { catalog.groups[0].trials[0].native_success = true; },
    catalog => { catalog.groups[47].trial_scope = 'randomized'; },
  ];
  for (const change of changes) {
    const {catalog, baseline} = catalogs(); change(catalog);
    assert.throws(() => compareCatalogs(catalog, baseline));
  }
});

function terminalFixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'additive-reference-test-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const summary = {task: 'bulb-screw-inplace', seed: 0, evaluation: {state: {clockwise_turns: .1004, rotation_from_initial_turns: .1},
    threaded_initialization: {initial_clockwise_turns: .0004, position_randomization_applied: false}}};
  const raw = JSON.stringify(summary);
  fs.writeFileSync(path.join(directory, 'summary.json'), raw);
  const trial = {seed: 0, summary_sha256: crypto.createHash('sha256').update(raw).digest('hex')};
  const reference = {key: 'rotation_from_initial_deg', unit: 'deg', target: 99.856, relation: 'target', binding: {
    absolute_target_deg: 100, initial_angle_turns: .0004, initial_angle_source: 'evaluation.threaded_initialization.initial_clockwise_turns',
    fixed_shared_relative_target: false, summary: {sha256: trial.summary_sha256}}};
  const config = {task_summary: [{key: 'rotation_from_initial_deg', unit: 'deg', value: 36, source: 'evaluation.state.rotation_from_initial_turns'}],
    goal_references: {task: 'bulb-screw-inplace', status: 'verified', references: [reference],
      binding: {source_code_executed: false, rules_reexecuted: false, evaluation_source_sha256: 'a'.repeat(64)},
      provenance: [{role: 'goal_reference_reviewed_source'}]}};
  const viz = {task_series: [{key: reference.key, unit: 'deg', values: [0, 36], time_s: [0, 90], goal_references: [structuredClone(reference)]}]};
  return {directory, trial, config, viz};
}

test('terminal net degrees and target are independently bound to the actual per-trial reset angle', t => {
  const {directory, trial, config, viz} = terminalFixture(t);
  const result = terminalAndReference(directory, trial, config, viz);
  assert.equal(result.value, 36); assert.equal(result.reference.target, 99.856);
});

test('a fixed 100-degree relative target or unmatched curve reference is rejected', t => {
  const {directory, trial, config, viz} = terminalFixture(t);
  config.goal_references.references[0].target = 100;
  viz.task_series[0].goal_references[0].target = 100;
  assert.throws(() => terminalAndReference(directory, trial, config, viz));
});

test('tampered copied raw summary cannot supply terminal or target facts', t => {
  const {directory, trial, config, viz} = terminalFixture(t);
  fs.appendFileSync(path.join(directory, 'summary.json'), '\n');
  assert.throws(() => terminalAndReference(directory, trial, config, viz));
});
