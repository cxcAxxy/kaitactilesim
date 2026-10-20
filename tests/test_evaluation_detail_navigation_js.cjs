'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {nativeOutcomeStatus, stageBadgeDefinitions, stageEvidenceMessage, validateWorkbenchIdentity,
  mountFallbackAudit, whiteboardAreaMessage, taskNavigationRows, mountTrajectoryNavigation,
  mountDetailHeader, mountStagedPanel, mountGoalReferenceStatus, mountBulbAngleSummary} = require('../scripts/analysis/evaluation_workbench/web/workbench.js');

// Small DOM sufficient for the additive header, with real listener teardown.
// Browser integration separately exercises the complete playback dashboard.
class Element {
  constructor(tag, doc) { this.tagName = tag; this.ownerDocument = doc; this.children = []; this.attributes = {}; this.dataset = {}; this.events = new Map(); this._text = ''; }
  appendChild(node) { node.remove(); this.children.push(node); node.parentNode = this; return node; }
  insertBefore(node, before) { node.remove(); const at = this.children.indexOf(before); this.children.splice(at < 0 ? this.children.length : at, 0, node); node.parentNode = this; return node; }
  get nextSibling() { return this.parentNode?.children[this.parentNode.children.indexOf(this) + 1] || null; }
  prepend(node) { node.remove(); this.children.unshift(node); node.parentNode = this; }
  remove() { if (this.parentNode) { const at = this.parentNode.children.indexOf(this); this.parentNode.children.splice(at, 1); this.parentNode = null; } }
  replaceChildren(...nodes) { [...this.children].forEach(node => node.remove()); this._text = ''; nodes.forEach(node => this.appendChild(node)); }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  set href(value) { this.setAttribute('href', value); }
  get href() { return this.attributes.href; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(type, callback) { if (!this.events.has(type)) this.events.set(type, new Set()); this.events.get(type).add(callback); }
  removeEventListener(type, callback) { this.events.get(type)?.delete(callback); }
  emit(type) { this.events.get(type)?.forEach(callback => callback({target: this})); }
}
function parent() { const doc = {createElement: tag => new Element(tag, doc)}; return new Element('div', doc); }
function all(node, className) { return [node, ...node.children.flatMap(child => all(child, className))].filter(item => (item.className || '').split(' ').includes(className)); }
function trial(id, seed, checkpoint = '20000', task = 'usb-insert') { return {task, trial_id: id, model: 'pi0.5', checkpoint, seed, prediction_horizon: 50, report_path: id + '/report.html', native_success: false}; }
function index(trials) { return {schema_version: 'evaluation-task-navigation-v1', prediction_horizon: 50, trials}; }
const rows = [trial('c', 0, '24999'), trial('b', 1), trial('a', 0), trial('other', 0, '20000', 'install-ram')];
function environment(data, current = 'b') {
  const requests = [], navigations = [];
  return {requests, navigations, AbortController, location: {protocol: 'http:', pathname: '/snapshot/' + current + '/report.html', assign: href => navigations.push(href)}, fetch: async (url, options) => { requests.push({url, options}); return {ok: true, json: async () => data}; }};
}

test('visible bulb angle strip separates live, true terminal and verified target readings', () => {
  const root = parent();
  const visualization = {clock_aligned: true, task_series: [{key: 'angle_removed_deg', label: '灯泡净旋出角度（逆时针）', unit: 'deg',
    time_s: [0, 1, 2], values: [0, -20, 80], gap_after: [false, true, false], goal_references: [{key: 'angle_removed_deg', unit: 'deg', relation: 'target', target: 131.5}]}]};
  const config = {task: 'bulb-unscrew', goal_references: {status: 'verified'}, task_summary: [{key: 'angle_removed_deg', unit: 'deg', value: 75, status: 'recorded'}]};
  const mounted = mountBulbAngleSummary(root, visualization, config), terminal = all(root, 'wb-bulb-angle-terminal')[0], target = all(root, 'wb-bulb-angle-target')[0];
  assert.equal(mounted.node.dataset.metricKey, 'angle_removed_deg');
  assert.match(all(root, 'wb-bulb-angle-current')[0].textContent, /0\.000°/);
  assert.match(terminal.textContent, /75\.000°/); assert.match(target.textContent, /131\.5°/);
  const terminalBefore = terminal.textContent, targetBefore = target.textContent;
  mounted.update(1);
  assert.match(all(root, 'wb-bulb-angle-current')[0].textContent, /-20\.000°/);
  assert.equal(terminal.textContent, terminalBefore); assert.equal(target.textContent, targetBefore);
  mounted.update(1.5); assert.match(all(root, 'wb-bulb-angle-current')[0].textContent, /该时刻未记录/);
  assert.equal(terminal.textContent, terminalBefore);
  assert.match(root.textContent, /反向回拧会抵消.*达到参考角仍须/);
  mounted.destroy(); assert.equal(root.children.length, 0);
});

test('missing final and relative-angle targets remain explicitly unavailable', () => {
  const root = parent(), mounted = mountBulbAngleSummary(root, {task_series: []}, {task: 'bulb-screw-inplace', goal_references: {status: 'unavailable'}});
  assert.match(all(root, 'wb-bulb-angle-terminal')[0].textContent, /未记录.*不使用最后一个有效采样替代/);
  assert.match(all(root, 'wb-bulb-angle-target')[0].textContent, /未设可核验目标.*实际初态/);
  mounted.destroy(); assert.equal(mountBulbAngleSummary(root, {}, {task: 'usb-insert'}), null);
});

test('reference availability is visible with reasons and does not invent metrics', () => {
  const root = parent(), mounted = mountGoalReferenceStatus(root, {}, {goal_references: {status: 'not_applicable', reason: '本条未记录任务状态，不套用其他任务参考。'}});
  assert.equal(mounted.node.dataset.status, 'not_applicable'); assert.equal(mounted.node.dataset.referenceCount, '0');
  assert.match(root.textContent, /未记录任务状态/); assert.doesNotMatch(root.textContent, /100°|12 mm/);
  mounted.destroy(); assert.equal(root.children.length, 0);
});

test('native task success remains independent of audit and staged completion', () => {
  const auditPass = {outcome: {status: 'pass'}};
  assert.equal(nativeOutcomeStatus(auditPass, {native_result: {task_success: false}, staged_result: {complete_success: true}}), 'fail');
  assert.equal(nativeOutcomeStatus({outcome: {status: 'fail'}}, {native_result: {task_success: true}, staged_result: {complete_success: false}}), 'pass');
  for (const value of [undefined, null, 0, 1, 'true', 'false']) assert.equal(nativeOutcomeStatus(auditPass, {native_result: {task_success: value}}), 'unknown');
  assert.equal(nativeOutcomeStatus(auditPass), 'pass');
  assert.equal(nativeOutcomeStatus({outcome: {status: 'unexpected'}}), 'unknown');
});

test('stage badges strictly copy cumulative flags and preserve unknown/missing data', () => {
  const staged = {stages: [{key: 'one', label: '拾取并稳定持有', reached: true, first_time_s: 1}, {key: 'two', label: '持有并接近插口', reached: false}, {key: 'three', label: '插入并就位', reached: null}, {key: 'four', name: '标签回退', reached: 'true', first_time_s: -1}]};
  const before = JSON.stringify(staged), badges = stageBadgeDefinitions(staged);
  assert.deepEqual(badges.map(item => item.reached), [true, false, null, null]);
  assert.deepEqual(badges.map(item => item.key), ['one', 'two', 'three', 'four']);
  assert.equal(badges[0].first_time_s, 1); assert.equal(badges[3].first_time_s, null);
  assert.equal(badges[3].label, '标签回退'); assert.equal(JSON.stringify(staged), before);
  assert.deepEqual(stageBadgeDefinitions(null), []); assert.deepEqual(stageBadgeDefinitions({stages: null}), []);
});

test('same-task rows sort by checkpoint/seed, bind current identity, and do not mutate index', () => {
  const data = index(rows), before = JSON.stringify(data);
  const selected = taskNavigationRows(data, {task: 'usb-insert', trial_id: 'b', report_path: 'b/report.html'});
  assert.deepEqual(selected.trials.map(item => item.trial_id), ['a', 'b', 'c']); assert.equal(selected.currentIndex, 1);
  assert.equal(JSON.stringify(data), before);
  assert.equal(taskNavigationRows(data, {task: 'install-ram', trial_id: 'b'}), null);
  assert.equal(taskNavigationRows(data, {task: 'usb-insert', trial_id: 'b', report_path: 'a/report.html'}), null);
  assert.equal(taskNavigationRows(data, {task: 'usb-insert', trial_id: 'missing'}), null);
});

test('unsafe URLs, duplicate identities, non-H50 rows, and bad schema disable navigation', () => {
  const identity = {task: 'usb-insert', trial_id: 'b'};
  for (const path of ['../b/report.html', '/b/report.html', '//evil/report.html', 'https://evil/b/report.html', 'b/report.html?time=1', 'b/report.html#time=2', 'b/../a/report.html', 'b%2fa/report.html', 'b\\report.html', 'javascript:alert(1)']) {
    assert.equal(taskNavigationRows(index([{...trial('b', 0), report_path: path}]), identity), null, path);
  }
  assert.equal(taskNavigationRows({...index(rows), prediction_horizon: 30}, identity), null);
  assert.equal(taskNavigationRows({...index(rows), schema_version: 'unknown'}, identity), null);
  assert.equal(taskNavigationRows(index([trial('b', 0), {...trial('a', 1), prediction_horizon: 30}]), identity), null);
  assert.equal(taskNavigationRows(index([trial('b', 0), trial('b', 1)]), identity), null);
  assert.equal(taskNavigationRows(index([trial('b', 0), {...trial('a', 1), report_path: 'b/report.html'}]), identity), null);
  assert.equal(taskNavigationRows(index([{...trial('b', 0), seed: -1}]), identity), null);
});

test('unknown seed is accepted by the index contract, sorts last within a checkpoint, and remains unknown', async () => {
  const unknown = {...trial('unknown', 0), seed: null};
  const unknown2 = {...trial('unknown2', 0), seed: null};
  const data = index([unknown2, trial('b', 1), unknown, trial('a', 0)]);
  const selected = taskNavigationRows(data, {task: 'usb-insert', trial_id: 'b'});
  assert.deepEqual(selected.trials.map(item => item.trial_id), ['a', 'b', 'unknown', 'unknown2']);
  const root = parent(), mounted = mountTrajectoryNavigation(root, {task: 'usb-insert', trial_id: 'b'}, environment(data));
  await mounted.ready;
  const options = all(root, 'wb-trajectory-select')[0].children.flatMap(group => group.children);
  assert.match(options[2].textContent, /^seed 未知 ·/); assert.match(options[3].textContent, /^seed 未知 ·/);
  assert.match(all(root, 'wb-trajectory-next')[0].title, /seed 未知$/); mounted.destroy();
});

test('navigation mounts grouped options, relative links, and full-page same-task switching', async () => {
  const root = parent(), env = environment(index(rows)), mounted = mountTrajectoryNavigation(root, {task: 'usb-insert', trial_id: 'b'}, env);
  assert.equal(all(root, 'wb-trajectory-select')[0].disabled, true);
  await mounted.ready;
  assert.equal(mounted.node.dataset.status, 'ready');
  assert.equal(all(root, 'wb-trajectory-count')[0].textContent, '当前 2 / 3 条');
  assert.equal(all(root, 'wb-trajectory-prev')[0].href, '../a/report.html');
  assert.equal(all(root, 'wb-trajectory-next')[0].href, '../c/report.html');
  const select = all(root, 'wb-trajectory-select')[0];
  assert.equal(select.value, 'b/report.html'); assert.equal(select.children.length, 2);
  assert.equal(select.children[0].label, 'pi0.5 · step 20000'); assert.equal(select.children[0].children.length, 2);
  select.emit('change'); assert.deepEqual(env.navigations, []);
  select.value = 'other/report.html'; select.emit('change'); assert.deepEqual(env.navigations, []);
  select.value = 'javascript:alert(1)'; select.emit('change'); assert.deepEqual(env.navigations, []);
  select.value = 'c/report.html'; select.emit('change'); assert.deepEqual(env.navigations, ['../c/report.html']);
  assert.equal(env.requests[0].url, '../trajectory_index.json');
  assert.equal(env.requests[0].options.mode, 'same-origin'); assert.equal(env.requests[0].options.credentials, 'same-origin');
  mounted.destroy(); mounted.destroy();
  assert.equal(root.children.length, 0); assert.equal(env.requests[0].options.signal.aborted, true);
  select.value = 'a/report.html'; select.emit('change'); assert.equal(env.navigations.length, 1, 'listener removed on destroy');
});

test('single task entry disables both boundaries and retains current selection', async () => {
  const root = parent(), mounted = mountTrajectoryNavigation(root, {task: 'usb-insert', trial_id: 'b'}, environment(index([trial('b', 0)])));
  await mounted.ready;
  assert.equal(all(root, 'wb-trajectory-select')[0].disabled, true);
  for (const cls of ['wb-trajectory-prev', 'wb-trajectory-next']) {
    const anchor = all(root, cls)[0]; assert.equal(anchor.getAttribute('href'), null); assert.equal(anchor.getAttribute('aria-disabled'), 'true');
  }
  assert.equal(all(root, 'wb-trajectory-count')[0].textContent, '当前 1 / 1 条'); mounted.destroy();
});

test('file mode, missing index, parse error, invalid identity degrade without throwing', async () => {
  const environments = [
    {...environment(index(rows)), location: {protocol: 'file:'}},
    {...environment(index(rows)), fetch: async () => ({ok: false})},
    {...environment(index(rows)), fetch: async () => { throw new Error('offline'); }},
    {...environment(index(rows)), fetch: async () => ({ok: true, json: async () => { throw new Error('bad JSON'); }})},
    environment(index(rows), 'wrong'), environment(index([trial('other', 0, '20000', 'install-ram')])),
  ];
  for (const env of environments) {
    const root = parent(), mounted = mountTrajectoryNavigation(root, {task: 'usb-insert', trial_id: 'b'}, env);
    await mounted.ready; assert.equal(mounted.node.dataset.status, 'unavailable');
    assert.equal(all(root, 'wb-trajectory-select')[0].disabled, true);
    assert.match(all(root, 'wb-trajectory-status')[0].textContent, /当前回放和指标不受影响/);
    mounted.destroy();
  }
});

test('destroy while fetch is pending aborts and prevents detached-node updates', async () => {
  let finish, signal;
  const env = {...environment(index(rows)), fetch: (_, options) => { signal = options.signal; return new Promise(resolve => { finish = resolve; }); }};
  const root = parent(), mounted = mountTrajectoryNavigation(root, {task: 'usb-insert', trial_id: 'b'}, env);
  await Promise.resolve(); mounted.destroy(); assert.equal(signal.aborted, true);
  finish({ok: true, json: async () => index(rows)}); await mounted.ready;
  assert.equal(root.children.length, 0); assert.equal(mounted.node.dataset.status, 'loading');
});

test('header badges remain whole-trajectory evidence and native success does not imply staged completion', async () => {
  const root = parent(), staged = {native_result: {task_success: true}, stages: [{key: 'stage1', label: '拾取并稳定持有', reached: true}, {key: 'stage2', label: '持有并接近插口', reached: false}, {key: 'stage3', label: '插入并就位', reached: null}], staged_result: {complete_success: false}};
  const payload = {report: {trial: {seed: 1, cohort: {model_family: 'pi0.5'}}, outcome: {status: 'fail', diagnostic_only: true, formal_metrics_valid: false}}};
  const data = {task: 'usb-insert', trial_id: 'b'}, config = {generic: true, checkpoint: '20000', suite_url: '../index.html', staged_evaluation: staged};
  const before = JSON.stringify({payload, data, config}), mounted = mountDetailHeader(root, payload, data, config, environment(index(rows)));
  await mounted.navigation.ready;
  const badges = all(root, 'wb-stage-badge');
  assert.deepEqual(badges.map(badge => badge.dataset.reached), ['true', 'false', 'unknown']);
  assert.deepEqual(badges.map(badge => badge.dataset.stage), ['stage1', 'stage2', 'stage3']);
  assert.match(badges[0].textContent, /^✓ 拾取/); assert.match(badges[1].textContent, /^— 持有/); assert.match(badges[2].textContent, /^\? 插入/);
  assert.equal(all(root, 'wb-verdict')[0].textContent, '原任务结果 · 成功');
  assert.equal(all(root, 'wb-header-stage-count')[0].textContent, '1 / 3 曾达到');
  assert.match(all(root, 'wb-header-stage-note')[0].textContent, /不随游标变化/);
  assert.equal(JSON.stringify({payload, data, config}), before, 'presentation must not alter archived inputs');
  mounted.destroy(); assert.equal(root.children.length, 0);
});

test('whiteboard pickup-only header explicitly does not claim wiping completion', async () => {
  const root = parent(), payload = {report: {trial: {seed: 0}, outcome: {status: 'fail'}}};
  const data = {task: 'whiteboard-wipe', trial_id: 'w'};
  const config = {staged_evaluation: {native_result: {task_success: false}, staged_result: {complete_success: null}, stages: [{key: 'stage1', label: '拾取并稳定持有', reached: true}]}};
  const mounted = mountDetailHeader(root, payload, data, config, {location: {protocol: 'file:'}});
  await mounted.navigation.ready;
  assert.match(all(root, 'wb-header-stage-note')[0].textContent, /只记录拿起，不能视为擦拭完成/);
  assert.equal(all(root, 'wb-verdict')[0].textContent, '原任务结果 · 失败'); mounted.destroy();
});

function unifiedFixture() {
  function group(id, model, checkpoint, horizon, task = 'poker-draw') {
    return {id, task, model, checkpoint, prediction_horizon: horizon, execute_steps: model === 'pi05' ? 50 : 5,
      protocol: {guard_mode: 'observe'}, trials: []};
  }
  function row(id, seed, extra = {}) {
    return {id, seed, valid: true, native_success: false, publication_status: 'eligible',
      staged_evidence_status: 'valid', detail_url: id + '/report.html', ...extra};
  }
  const pi05 = group('g-pi05', 'pi05', '24999', 50);
  pi05.trials = [row('pi0', 0), row('q0', 1, {valid: false, native_success: null, publication_status: 'quarantined_replay_only'})];
  const cpm = group('g-cpm', 'pi05+cpm', 'best', 30);
  cpm.trials = [row('cpm0', 0, {staged_evidence_status: 'invalid'})];
  const ftp = group('g-ftp', 'ftp1', '10000', 32);
  ftp.trials = [row('ftp0', 0), row('ftp1', 1)];
  const ram = group('g-ram', 'pi05', '24999', 50, 'install-ram'); ram.trials = [row('ram0', 0)];
  return {schema: 'kaihand-unified-evaluation-workbench-v1', groups: [pi05, cpm, ftp, ram]};
}
function unifiedIdentity(extra = {}) {
  return {task: 'poker-draw', group_id: 'g-cpm', trial_id: 'cpm0', model: 'pi05+cpm', checkpoint: 'best',
    seed: 0, prediction_horizon: 30, execute_steps: 5, valid: true, native_success: false,
    publication_status: 'eligible', staged_evidence_status: 'invalid', stage_recorded: true, ...extra};
}
function unifiedEnvironment(data = unifiedFixture()) {
  const env = environment(data);
  env.location = {protocol: 'http:', pathname: '/rich/cpm0/report.html', href: 'http://localhost/rich/cpm0/report.html',
    assign: href => env.navigations.push(href)};
  return env;
}
test('unified same-task navigation spans native H50, CPM H30 and FTP H32 without borrowing identity', () => {
  const catalog = unifiedFixture(), before = JSON.stringify(catalog);
  const selected = taskNavigationRows(catalog, {...unifiedIdentity(), report_path: 'cpm0/report.html'});
  assert.equal(selected.trials.length, 5);
  assert.deepEqual(new Set(selected.trials.map(item => item.prediction_horizon)), new Set([50, 30, 32]));
  assert.deepEqual(new Set(selected.trials.map(item => item.model)), new Set(['pi05', 'pi05+cpm', 'ftp1']));
  assert.equal(selected.trials.find(item => item.trial_id === 'q0').native_success, null);
  for (const change of [{task: 'install-ram'}, {model: 'ftp1'}, {group_id: 'g-ftp'}, {checkpoint: '24999'},
    {seed: 1}, {prediction_horizon: 50}, {execute_steps: 50}, {report_path: 'ftp0/report.html'}]) {
    assert.equal(taskNavigationRows(catalog, {...unifiedIdentity(), ...change}), null, JSON.stringify(change));
  }
  assert.equal(JSON.stringify(catalog), before);
});
test('unified navigation rejects unsafe paths, duplicate IDs and invalid typed horizons', () => {
  for (const path of ['../cpm0/report.html', 'https://evil/cpm0/report.html', '/cpm0/report.html',
    'cpm0/report.html#time=1', 'cpm0/report.html?time=1', 'cpm0%2f../report.html', 'cpm0\\report.html']) {
    const catalog = unifiedFixture(); catalog.groups[1].trials[0].detail_url = path;
    assert.equal(taskNavigationRows(catalog, unifiedIdentity()), null, path);
  }
  for (const edit of [catalog => { catalog.groups[1].prediction_horizon = true; },
    catalog => { catalog.groups[1].execute_steps = 31; },
    catalog => { catalog.groups[2].trials[0].id = 'cpm0'; }]) {
    const catalog = unifiedFixture(); edit(catalog);
    assert.equal(taskNavigationRows(catalog, unifiedIdentity()), null);
  }
});
test('unified navigation fetches the complete catalog and uses bound same-snapshot destinations', async () => {
  const root = parent(), env = unifiedEnvironment();
  const mounted = mountTrajectoryNavigation(root, unifiedIdentity(), env, {catalog_url: '../catalog.json'});
  await mounted.ready;
  assert.equal(mounted.node.dataset.status, 'ready');
  assert.match(all(root, 'wb-trajectory-count')[0].textContent, /\/ 5 条$/);
  assert.equal(env.requests[0].url, '../catalog.json');
  const select = all(root, 'wb-trajectory-select')[0];
  assert.equal(select.children.length, 3);
  assert.ok(select.children.some(group => /CPM.*best.*H30\/K5/.test(group.label)));
  assert.ok(select.children.some(group => /FTP-1.*10000.*H32\/K5/.test(group.label)));
  assert.match(select.textContent, /隔离历史 · 不计分/);
  select.value = 'ftp0/report.html'; select.emit('change');
  assert.deepEqual(env.navigations, ['http://localhost/rich/ftp0/report.html']);
  mounted.destroy();
  const unsafe = mountTrajectoryNavigation(parent(), unifiedIdentity(), unifiedEnvironment(), {catalog_url: 'https://evil/catalog.json'});
  await unsafe.ready; assert.equal(unsafe.node.dataset.status, 'unavailable'); unsafe.destroy();
});
test('rich workbench accepts only matching positive reported, load and config horizons', () => {
  for (const horizon of [30, 32, 50]) {
    const payload = {report: {trial: {trial_id: 'internal', task: 'poker-draw', seed: 0, prediction_horizon: horizon}}};
    const data = {trial_id: 'internal', task: 'poker-draw', prediction_horizon: horizon};
    const config = {trial_id: 'internal', checkpoint: 'best', prediction_horizon: horizon,
      unified_identity: unifiedIdentity({prediction_horizon: horizon})};
    assert.equal(validateWorkbenchIdentity(payload, data, config), true);
    assert.throws(() => validateWorkbenchIdentity(payload, {...data, prediction_horizon: 50 + horizon}, config), /identity mismatch/);
    assert.throws(() => validateWorkbenchIdentity(payload, data, {...config, unified_identity: {...config.unified_identity, seed: 1}}), /identity mismatch/);
    assert.throws(() => validateWorkbenchIdentity(payload, data, {...config, checkpoint: '24999'}), /identity mismatch/);
  }
});
test('CPM rich header displays true H/K and unverified recorded stages, not missing or reached stages', async () => {
  const root = parent(), published = unifiedIdentity();
  const payload = {report: {trial: {trial_id: 'internal', task: 'poker-draw', seed: 0, prediction_horizon: 30,
    cohort: {model_family: 'pi05+cpm'}}, outcome: {status: 'pass'}}};
  const data = {task: 'poker-draw', trial_id: 'internal', prediction_horizon: 30};
  const config = {checkpoint: 'best', unified_identity: published,
    unified_navigation: {catalog_url: '../catalog.json', overview_url: '../index.html'},
    audit_links: [{label: '原始记录', url: '../../original/cpm/report.html'}],
    staged_evaluation: {stages: [{key: 'stage1', label: '拖牌', reached: true, first_time_s: 2}],
      staged_result: {complete_success: true}, warnings: ['checkpoint SHA mismatch']}};
  const mounted = mountDetailHeader(root, payload, data, config, unifiedEnvironment()); await mounted.navigation.ready;
  assert.equal(all(root, 'wb-horizon')[0].textContent, 'H30 / K5');
  assert.match(all(root, 'wb-trial')[0].textContent, /CPM.*best.*seed 000.*H30 \/ K5/);
  assert.equal(all(root, 'wb-overview-link')[0].href, '../index.html');
  assert.equal(all(root, 'wb-header-stage-count')[0].textContent, '已记录阶段，但来源校验未通过');
  assert.equal(all(root, 'wb-stage-badge')[0].dataset.reached, 'unknown');
  assert.doesNotMatch(all(root, 'wb-stage-badge')[0].title, /首次/);
  assert.equal(all(root, 'wb-verdict')[0].textContent, '原任务结果 · 失败');
  assert.match(all(root, 'wb-evidence-audit')[0].textContent, /checkpoint SHA mismatch/);
  assert.equal(all(root, 'wb-original-record-link')[0].href, '../../original/cpm/report.html');
  mounted.destroy();
});
test('quarantine masks header, stage panel and first times but preserves an auditable original record', async () => {
  const published = unifiedIdentity({valid: false, publication_status: 'quarantined_replay_only', native_success: true});
  const staged = {stages: [{key: 'stage1', label: '拿起', reached: true, first_time_s: 1}],
    staged_result: {complete_success: true}, native_result: {task_success: true}};
  assert.equal(nativeOutcomeStatus({outcome: {status: 'pass'}}, staged, published), 'unknown');
  assert.deepEqual(stageBadgeDefinitions(staged, published).map(stage => [stage.reached, stage.first_time_s]), [[null, null]]);
  assert.match(stageEvidenceMessage(published).note, /不进入当前成功率或阶段分母/);
  const root = parent(), panel = mountStagedPanel(root, staged, () => { throw new Error('Unverified stage must not seek'); }, 90, published);
  assert.equal(panel.node, undefined);
  assert.equal(all(root, 'wb-stage-card')[0].dataset.reached, 'unknown');
  assert.equal(all(root, 'wb-stage-first-time').length, 0);
  assert.match(all(root, 'wb-staged-scope')[0].textContent, /隔离轨迹 · 阶段不计分/);
  panel.update(1); panel.destroy(); assert.equal(root.children.length, 0);
});
test('generic fallback collapses raw green audit verdicts for unified pages and restores legacy DOM on teardown', () => {
  const root = parent(); root.querySelector = selector => all(root, selector.slice(1))[0] || null;
  const header = new Element('header', root.ownerDocument); header.className = 'ke-header';
  const summary = new Element('div', root.ownerDocument); summary.className = 'ke-summary';
  const pill = new Element('span', root.ownerDocument); pill.className = 'ke-pill ke-pass'; pill.textContent = '通过'; summary.appendChild(pill);
  const media = new Element('div', root.ownerDocument); media.className = 'ke-layout';
  root.appendChild(header); root.appendChild(summary); root.appendChild(media);
  const before = [...root.children];
  assert.equal(mountFallbackAudit(root, null), null, 'legacy pages without unified publication identity stay untouched');
  assert.deepEqual(root.children, before);
  const audit = mountFallbackAudit(root, unifiedIdentity({valid: false, publication_status: 'quarantined_replay_only'}));
  assert.equal(audit.node.open, false);
  assert.equal(summary.parentNode.parentNode, audit.node);
  assert.equal(header.parentNode.parentNode, audit.node);
  assert.equal(media.parentNode, root, 'playback controls and layout remain available');
  assert.match(audit.node.textContent, /已隔离或无效，不计分/);
  assert.equal(pill.textContent, '通过', 'raw archive labels remain unchanged inside collapsed audit only');
  audit.destroy(); assert.deepEqual(root.children, before);
});
test('recorded whiteboard geometry is displayed without claiming renewed verification', () => {
  assert.match(whiteboardAreaMessage({status: 'recorded'}), /保留归档曲线.*不重新积分/);
  assert.match(whiteboardAreaMessage({status: 'recorded'}), /几何估计（未重新认证）/);
  assert.doesNotMatch(whiteboardAreaMessage({status: 'recorded'}), /来源未能核实/);
  assert.match(whiteboardAreaMessage({status: 'unavailable'}), /不将缺测写成 0/);
});
test('unified stage downloads use the provided page configuration and legacy downloads retain their original sidecar', () => {
  const staged = {stages: [], staged_result: {complete_success: false}};
  for (const published of [undefined, unifiedIdentity({staged_evidence_status: 'valid'}), unifiedIdentity()]) {
    const root = parent(), panel = mountStagedPanel(root, staged, null, 90, published);
    const download = all(root, 'wb-link')[0];
    assert.equal(download.href, published ? 'workbench.json' : 'staged_evaluation.json');
    assert.equal(download.textContent, published ? '下载本页配置（含阶段记录与来源）' : '下载阶段记录与来源');
    panel.destroy();
  }
});
