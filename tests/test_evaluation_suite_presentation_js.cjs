'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'scripts/analysis/evaluation_workbench/web/suite.js'), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.className = '';
    this.dataset = {};
    this.listeners = {};
    this.attributes = {};
    this.value = '';
    this._text = '';
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  fire(type, value = this.value) {
    this.value = value;
    assert.equal(typeof this.listeners[type], 'function', `missing ${type} listener`);
    this.listeners[type]({target: this});
  }
}

function descendants(node, className) {
  return node.children.flatMap(child => [
    ...(child.className.split(' ').includes(className) ? [child] : []),
    ...descendants(child, className),
  ]);
}

function fixture() {
  const specs = [
    ['install-ram', 'pi0_5', '20000', 0, 'formal', 'pass', true, 'available', 'reference'],
    ['install-ram', 'pi0_5', '24999', 5, 'diagnostic', 'fail', false, 'available', 'reference'],
    ['usb-insert', 'pi0', '20000', 5, 'unverified', 'unknown', null, 'available', 'model_only'],
    ['usb-insert', 'pi0', '24999', 6, 'formal', 'pass', true, 'missing', null],
  ];
  const trials = specs.map(([task, model, checkpoint, seed, mode, outcome, complete, plots, plotKind], i) => ({
    trial_id: `trial_${i}`, report_path: `00${i}_case/report.html`, task, model, checkpoint, seed, mode, outcome,
    title: task === 'install-ram' ? '内存条插入' : 'USB 插入',
    tactile: i === 3 ? 'missing' : 'available', tangent: 'available',
    expert: i === 3 ? 'missing' : 'available', plots, plot_kind: plotKind,
    staged: {
      complete_success: complete,
      stages: [{label: '接近', reached: complete === null ? false : true}, {label: '就位', reached: complete}],
      native_result: {task_success: complete}, source_selection: {group_status: 'completed'},
    },
  }));
  return {
    evaluation_profile: 'recorded-staged', task_count: 2, trial_count: trials.length,
    mode_counts: {formal: 2, diagnostic: 1, unverified: 1},
    generated_utc: '2026-10-15T02:21:24.145599+00:00',
    coverage: {normal: 3, tangent: 4, expert: 3, plots: 3},
    provenance: {source_manifest: 'internal-only-source.json'},
    tasks: {
      'install-ram': {title: '内存条插入', trial_count: 2, task_note: '保留内存条任务说明'},
      'usb-insert': {title: 'USB 插入', trial_count: 2},
    },
    stage_groups: trials.map(t => ({
      task: t.task, model: t.model, checkpoint: t.checkpoint, mode: t.mode,
      title: t.title, count: 1, stages: [{label: '接近', reached: 1, unknown: 0}],
      complete_success_defined: true, complete_success_count: Number(t.staged.complete_success === true),
      native_success_count: Number(t.staged.native_result.task_success === true),
      rule_sha256: 'a'.repeat(64), cohort: 'internal-cohort', checkpoint_identity: 'weights-sha',
    })),
    trials,
  };
}

function mount(batch = fixture()) {
  const ids = ['kai-suite-data', 'ks-counts', 'ks-task-tabs', 'ks-model', 'ks-checkpoint',
    'ks-outcome', 'ks-stage', 'ks-stage-groups', 'ks-search', 'ks-visible', 'ks-task-note', 'ks-trials'];
  const nodes = Object.fromEntries(ids.map(id => [id, new Element()]));
  nodes['kai-suite-data'].textContent = JSON.stringify(batch);
  const document = {
    createElement: tag => new Element(tag),
    getElementById: id => {
      assert(!['ks-mode', 'ks-coverage'].includes(id), `removed presentation node must not be consulted: ${id}`);
      return nodes[id] || null;
    },
  };
  vm.runInNewContext(source, {document}, {filename: 'suite.js'});
  return {nodes, batch, grid: nodes['ks-trials'], cards: () => descendants(nodes['ks-trials'], 'ks-case')};
}

test('clean portal mounts without mode or coverage nodes and leaves internal evidence unchanged', () => {
  const batch = fixture(), before = JSON.stringify(batch);
  const app = mount(batch);
  assert.equal(app.cards().length, 4);
  assert.deepEqual(app.nodes['ks-counts'].children.map(node => node.textContent), ['2类任务', '4条 H50 轨迹']);
  assert.equal(descendants(app.grid, 'ks-mode').length, 0);
  assert.doesNotMatch(app.grid.textContent + app.nodes['ks-counts'].textContent, /正式标志|诊断轨迹|快照时间|模型法向载荷/);
  assert.equal(JSON.stringify(batch), before);
  assert.equal(app.nodes['kai-suite-data'].textContent, before);
});

test('task, model, checkpoint, recorded outcome, stage, and seed filters still work', () => {
  const app = mount(), nodes = app.nodes;
  nodes['ks-model'].fire('change', 'pi0_5');
  assert.equal(app.cards().length, 2);
  nodes['ks-model'].fire('change', '');
  nodes['ks-checkpoint'].fire('change', '24999');
  assert.equal(app.cards().length, 2);
  nodes['ks-checkpoint'].fire('change', '');
  nodes['ks-outcome'].fire('change', 'unknown');
  assert.equal(app.cards().length, 1);
  nodes['ks-outcome'].fire('change', '');
  nodes['ks-stage'].fire('change', 'complete');
  assert.equal(app.cards().length, 2);
  nodes['ks-stage'].fire('change', 'partial');
  assert.equal(app.cards().length, 1);
  nodes['ks-stage'].fire('change', 'none');
  assert.equal(app.cards().length, 1);
  nodes['ks-stage'].fire('change', '');
  nodes['ks-search'].fire('input', 'seed_005');
  assert.equal(app.cards().length, 2);
  nodes['ks-search'].fire('input', '');
  nodes['ks-task-tabs'].children.find(button => button.dataset.task === 'install-ram').fire('click');
  assert.equal(app.cards().length, 2);
  assert.equal(nodes['ks-stage-groups'].children.length, 2);
  assert.match(nodes['ks-task-note'].textContent, /保留内存条任务说明/);
  assert.equal(nodes['ks-model'].children.length, 2, 'task selection narrows model options');
  nodes['ks-search'].fire('input', 'does-not-exist');
  assert.equal(app.cards().length, 0);
  assert.match(app.grid.textContent, /没有符合当前筛选条件/);
});

test('per-case availability and native/stage outcomes remain visible without mode badges', () => {
  const app = mount(), cards = app.cards();
  assert.match(app.grid.textContent, /✓ Fn \/ Ft/);
  assert.match(app.grid.textContent, /✓ 专家示范/);
  assert.match(app.grid.textContent, /✓ 模型全程图/);
  assert.match(app.grid.textContent, /✓ 参考对照图/);
  assert.match(app.grid.textContent, /— Fn \/ Ft/);
  assert.match(app.grid.textContent, /— 专家示范/);
  assert.match(app.grid.textContent, /数据审核口径 · 记录为通过/);
  assert.match(app.grid.textContent, /完整阶段：通过 · 原任务：成功/);
  assert(cards.every(card => /^[A-Za-z0-9_-]+\/report\.html$/.test(card.href)));
});

test('two summary statistics share two columns on desktop and mobile', () => {
  const css = fs.readFileSync(path.join(root, 'scripts/analysis/evaluation_workbench/web/suite.css'), 'utf8');
  const rules = [...css.matchAll(/\.ks-counts\{([^}]+)\}/g)].map(match => match[1]);
  assert(rules.length >= 2);
  assert(rules.every(rule => /grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/.test(rule)));
  assert.doesNotMatch(css, /\.ks-mode/);
});
