'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../scripts/analysis/evaluation_workbench/web/suite.js'), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.className = ''; this.children = []; this.parentNode = null;
    this.attributes = {}; this.dataset = {}; this.listeners = {}; this._text = ''; this.value = '';
  }
  get nextSibling() { if (!this.parentNode) return null; return this.parentNode.children[this.parentNode.children.indexOf(this) + 1] || null; }
  detach(child) { if (child.parentNode) { const siblings = child.parentNode.children; siblings.splice(siblings.indexOf(child), 1); } }
  appendChild(child) { this.detach(child); this.children.push(child); child.parentNode = this; return child; }
  insertBefore(child, before) { if (!before) return this.appendChild(child); this.detach(child); this.children.splice(this.children.indexOf(before), 0, child); child.parentNode = this; return child; }
  replaceChildren(...children) { this.children.forEach(child => {child.parentNode = null;}); this.children = []; this._text = ''; children.forEach(child => this.appendChild(child)); }
  set textContent(value) { this.replaceChildren(); this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  querySelector(selector) {
    return descendants(this).find(child => selector.startsWith('.') ? child.className.split(' ').includes(selector.slice(1))
      : selector === 'p:not(.ks-eyebrow)' ? child.tagName === 'p' && child.className !== 'ks-eyebrow'
      : child.tagName === selector) || null;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  fire(type, value = this.value) { this.value = value; this.listeners[type]({target: this}); }
}
function descendants(node) { return node.children.flatMap(child => [child, ...descendants(child)]); }
function byClass(root, name) { return descendants(root).filter(child => child.className.split(' ').includes(name)); }

function fixture() {
  const trials = [
    ['usb-unplug', '20000', 0, false, false, [true, false, null]],
    ['usb-unplug', '24999', 1, true, false, [true, true, false]],
    ['whiteboard-wipe', '20000', null, null, null, [true]],
  ].map(([task, checkpoint, seed, native, complete, reached], index) => ({
    task, checkpoint, seed, model: 'pi0.5', title: task, trial_id: 'id_' + index,
    outcome: native === true ? 'pass' : native === false ? 'fail' : 'unknown',
    report_path: '00' + index + '_case/report.html', prediction_horizon: 50,
    tactile: 'available', tangent: 'available', expert: 'available', plots: 'available',
    staged: {complete_success: complete, native_result: {task_success: native},
      stages: reached.map((flag, i) => ({label: '阶段 ' + i, reached: flag})),
      source_selection: {group_status: 'completed'}},
  }));
  return {evaluation_profile: 'recorded-staged', task_count: 2, trial_count: 3, trials,
    tasks: {'usb-unplug': {title: 'USB 拔出与放回', trial_count: 2}, 'whiteboard-wipe': {title: '白板擦拭', trial_count: 1}},
    stage_groups: []};
}
function mount({storage = null, blockedStorage = false} = {}) {
  const nodes = {}, main = new Element('main'), header = main.appendChild(new Element('header'));
  const title = header.appendChild(new Element('h1')); title.textContent = '旧标题';
  const eyebrow = header.appendChild(new Element('p')); eyebrow.className = 'ks-eyebrow'; eyebrow.textContent = '旧英文标题';
  const subtitle = header.appendChild(new Element('p')); subtitle.textContent = '旧副标题';
  const previous = header.appendChild(new Element('a')); previous.className = 'ks-old'; previous.textContent = '旧入口'; previous.href = '../old/index.html';
  const ids = ['kai-suite-data', 'ks-counts', 'ks-task-tabs', 'ks-model', 'ks-checkpoint', 'ks-outcome',
    'ks-stage', 'ks-stage-groups', 'ks-search', 'ks-task-note', 'ks-visible', 'ks-trials'];
  ids.forEach(id => {nodes[id] = main.appendChild(new Element()); nodes[id].id = id;});
  nodes['kai-suite'] = main;
  const group = new Element('details'); group.className = 'ks-stage-groups'; main.insertBefore(group, nodes['ks-trials']);
  const batch = fixture(), serialized = JSON.stringify(batch); nodes['kai-suite-data'].textContent = serialized;
  const document = {createElement: tag => new Element(tag), getElementById: id => nodes[id] || null,
    querySelector: selector => selector === '#kai-suite > header' ? header : selector === '.ks-stage-groups' ? group : null};
  const sandbox = {document};
  if (blockedStorage) Object.defineProperty(sandbox, 'localStorage', {get() {throw Error('Storage denied');}});
  else if (storage) sandbox.localStorage = storage;
  vm.runInNewContext(source, sandbox);
  const viewButtons = () => byClass(main, 'ks-view-button');
  return {nodes, main, header, title, eyebrow, subtitle, previous, document, group, batch, serialized, viewButtons, grid: nodes['ks-trials']};
}

test('studio shell puts grouped statistics immediately below the count and above trajectories', () => {
  const app = mount();
  assert.equal(app.title.textContent, 'KaiTactileSim 官方评估平台');
  assert.equal(app.document.title, 'KaiTactileSim 官方评估平台 | H50');
  assert.equal(app.eyebrow.textContent, 'KaiTactileSim Evaluation Platform');
  assert.equal(app.subtitle.textContent, '机器人操作任务评估 · 动作回放 · 触觉分析');
  assert.equal(app.previous.textContent, '历史 H50 版本');
  assert.equal(app.previous.href, '../old/index.html');
  assert.match(byClass(app.main, 'ks-brand-footer')[0].textContent, /KaiTactileSim 官方评估平台/);
  assert.doesNotMatch(app.main.textContent, /已认证|官方排名/);
  assert.equal(app.nodes['ks-counts'].parentNode, app.header);
  assert.match(app.nodes['ks-counts'].textContent, /2类任务3条 H50 轨迹/);
  assert.equal(app.nodes['ks-visible'].parentNode.className, 'ks-results-bar');
  assert.equal(app.nodes['ks-visible'].parentNode.nextSibling, app.group);
  assert.equal(app.group.nextSibling, app.grid);
  assert.equal(app.nodes['kai-suite-data'].textContent, app.serialized, 'No archival evaluation data is rewritten');
});

test('card/list toggle changes presentation only, stays selected through filtering, and persists safely', () => {
  const values = new Map(), storage = {getItem: key => values.get(key), setItem: (key, value) => values.set(key, value)};
  const app = mount({storage});
  const [cards, list] = app.viewButtons();
  assert.equal(cards.id, 'ks-view-cards'); assert.equal(list.id, 'ks-view-list');
  assert.equal(cards.attributes['aria-pressed'], 'true');
  assert.equal(list.attributes['aria-pressed'], 'false');
  const hrefs = byClass(app.grid, 'ks-case').map(card => card.href);
  list.fire('click');
  assert.equal(app.grid.dataset.view, 'list');
  assert.equal(list.attributes['aria-pressed'], 'true'); assert.equal(cards.attributes['aria-pressed'], 'false');
  assert.deepEqual(byClass(app.grid, 'ks-case').map(card => card.href), hrefs);
  assert.equal(values.get('kai-evaluation-suite-view-v1'), 'list');
  app.nodes['ks-search'].fire('input', 'seed_001');
  assert.equal(byClass(app.grid, 'ks-case').length, 1); assert.equal(app.grid.dataset.view, 'list');
  assert.equal(app.nodes['ks-visible'].parentNode.nextSibling, app.group);
  assert.equal(app.group.nextSibling, app.grid, 'Filtering and view switching keep statistics above trajectories');
  assert.equal(mount({storage}).grid.dataset.view, 'list', 'Reload retains preference');
  cards.fire('click'); assert.equal(app.grid.dataset.view, 'cards');
});

test('blocked or unavailable storage falls back without preventing display or switching', () => {
  for (const options of [{}, {blockedStorage: true}, {storage: {getItem: () => 'unexpected', setItem: () => {throw Error('Write denied');}}}]) {
    const app = mount(options);
    assert.equal(app.grid.dataset.view, 'cards'); assert.equal(byClass(app.grid, 'ks-case').length, 3);
    app.viewButtons()[1].fire('click'); assert.equal(app.grid.dataset.view, 'list');
  }
});

test('native success, audit outcome, cumulative stages and whiteboard limitations remain separate', () => {
  const app = mount(), cards = byClass(app.grid, 'ks-case');
  const success = cards.find(card => card.href === '001_case/report.html'), unknown = cards.find(card => card.href === '002_case/report.html');
  assert.match(success.textContent, /原任务 · 成功/);
  assert.match(success.textContent, /完整阶段：未通过 · 原任务：成功/);
  assert.match(cards[0].textContent, /\? 阶段 2/);
  assert.match(unknown.textContent, /seed 未知/);
  assert.match(unknown.textContent, /原任务 · 未知/);
  assert.match(unknown.textContent, /未定义（仅拿起）/);
  assert.match(unknown.textContent, /不代表擦拭完成/);
  assert.doesNotMatch(app.main.textContent, /运行模式|正式标志|数据覆盖与来源/);
});
