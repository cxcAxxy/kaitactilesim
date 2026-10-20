'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(require.resolve('../scripts/analysis/evaluation_workbench/web/workbench.js'), 'utf8');
const {taskDisplayTitle, mountDetailHeader} = require('../scripts/analysis/evaluation_workbench/web/workbench.js');

class Element {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase(); this.ownerDocument = doc; this.children = []; this.attributes = {}; this.dataset = {}; this._text = ''; this.events = new Map();
    const classes = new Set();
    this.classList = {add: (...values) => values.forEach(value => classes.add(value)), remove: (...values) => values.forEach(value => classes.delete(value)), contains: value => classes.has(value), toggle: (value, force) => { const next = force ?? !classes.has(value); if (next) classes.add(value); else classes.delete(value); return next; }};
  }
  appendChild(node) { node.remove(); this.children.push(node); node.parentNode = this; return node; }
  prepend(node) { node.remove(); this.children.unshift(node); node.parentNode = this; }
  remove() { if (this.parentNode) { this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; } }
  set textContent(value) { this.children.forEach(child => { child.parentNode = null; }); this.children = []; this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(type, fn) { this.events.set(type, fn); }
  removeEventListener(type, fn) { if (this.events.get(type) === fn) this.events.delete(type); }
}
function fixture() {
  const doc = {title: 'Archived page', createElement: tag => new Element(tag, doc)};
  return {doc, root: new Element('main', doc)};
}
function all(node, className) { return [node, ...node.children.flatMap(child => all(child, className))].filter(item => (item.className || '').split(' ').includes(className)); }

test('explicit task names cover all supported tasks and prefer archived presentation', () => {
  const expected = {'bulb-screw': '灯泡旋入与释放', 'bulb-screw-inplace': '灯泡原位旋入', 'bulb-unscrew': '灯泡旋出与放回', 'install-ram': '内存条安装', 'poker-draw': '卡片抽取与举起', 'usb-insert': 'USB 插入', 'usb-unplug': 'USB 拔出', 'whiteboard-wipe': '倾斜白板擦拭', 'whiteboard-wipe-horizontal': '水平白板擦拭', 'whiteboard-wipe-horizontal-line': '水平白板线迹擦拭'};
  for (const [task, title] of Object.entries(expected)) assert.equal(taskDisplayTitle({}, task), title);
  assert.equal(taskDisplayTitle({task_presentation: {title: '归档任务名称'}}, 'usb-insert'), '归档任务名称');
  assert.equal(taskDisplayTitle({}, 'custom-task'), 'custom-task');
  assert.equal(taskDisplayTitle({}, 'constructor'), 'constructor');
  assert.equal(taskDisplayTitle({}, '/path/usb-insert.py'), '/path/usb-insert.py', 'a filename is not interpreted as a task');
  assert.equal(taskDisplayTitle({}, null), '轨迹分析');
});

test('detail branding updates only presentation, retains archive links, and restores browser title', async () => {
  const {doc, root} = fixture();
  const payload = {report: {trial: {seed: 7}, outcome: {status: 'pass', diagnostic_only: true, formal_metrics_valid: false}}};
  const data = {task: 'usb-insert', trial_id: 'usb007'};
  const config = {suite_url: '../index.html', previous_workbench_url: '../../old/usb/report.html', stable_url: '../../stable/usb/report.html', staged_evaluation: {native_result: {task_success: false}, staged_result: {complete_success: true}, stages: []}};
  const before = JSON.stringify({payload, data, config});
  const mounted = mountDetailHeader(root, payload, data, config, {location: {protocol: 'file:'}});
  await mounted.navigation.ready;
  assert.equal(doc.title, 'USB 插入 | KaiTactileSim');
  assert.equal(all(root, 'wb-identity')[0].children.find(node => node.tagName === 'H1').textContent, 'USB 插入 · 轨迹分析');
  assert.equal(all(root, 'wb-platform-name')[0].textContent, 'KaiTactileSim Evaluation Platform');
  assert.equal(all(root, 'wb-horizon')[0].textContent, 'H50');
  assert.deepEqual(all(root, 'wb-links')[0].children.map(node => [node.textContent, node.href]), [['H50 稳定版', '../../stable/usb/report.html'], ['评估总览', '../index.html'], ['历史版本', '../../old/usb/report.html']]);
  assert.equal(all(root, 'wb-verdict')[0].textContent, '原任务结果 · 失败');
  assert.equal(JSON.stringify({payload, data, config}), before, 'brand ownership must not imply formal eligibility or rescore native/staged results');
  mounted.destroy(); mounted.destroy(); assert.equal(doc.title, 'Archived page'); assert.equal(root.children.length, 0);
});

test('fallback dashboard receives the same brand, footer and reversible browser title', () => {
  const {doc, root} = fixture(), host = {ownerDocument: doc, querySelector: selector => selector === '.kai-eval' ? root : null};
  const window = {}, context = {window, console};
  vm.runInNewContext(source, context);
  const payload = {report: {trial: {trial_id: 'fallback', seed: 0}, outcome: {status: 'fail'}}};
  const data = {task: 'install-ram', trial_id: 'fallback', prediction_horizon: 50, channels: [], time_s: []};
  const config = {generic: true, trial_id: 'fallback', prediction_horizon: 50};
  const before = JSON.stringify({payload, data, config});
  const mounted = window.KaiEvaluationWorkbench.mount(host, payload, data, config);
  assert.equal(doc.title, '内存条安装 | KaiTactileSim');
  assert.equal(all(root, 'wb-platform-name')[0].textContent, 'KaiTactileSim Evaluation Platform');
  assert.match(all(root, 'wb-footer')[0].textContent, /^KaiTactileSim 官方评估平台/);
  assert.match(all(root, 'wb-footer')[0].textContent, /归档规则.*不代表评估认证/);
  assert.equal(JSON.stringify({payload, data, config}), before);
  mounted.destroy(); assert.equal(doc.title, 'Archived page'); assert.equal(root.children.length, 0);
});

test('independent diagnostic page uses task names, formal labels and reversible title', () => {
  const {doc, root} = fixture();
  const config = {task: 'bulb-unscrew', status: 'unavailable', source_label: 'archived-source', workbench_url: '../bulb-unscrew/report.html', plots: []};
  doc.getElementById = id => id === 'kai-diagnostic-plots' ? root : id === 'kai-diagnostic-config' ? {textContent: JSON.stringify(config)} : null;
  const window = {};
  vm.runInNewContext(source, {window, document: doc, console});
  assert.equal(doc.title, '灯泡旋出与放回 · 全程诊断图 | KaiTactileSim');
  assert.equal(root.children.find(node => node.tagName === 'H1').textContent, '灯泡旋出与放回 · 全程诊断图');
  const back = all(root, 'wb-link')[0]; assert.equal(back.textContent, '返回轨迹分析'); assert.equal(back.href, config.workbench_url);
  assert.match(all(root, 'wb-footer')[0].textContent, /KaiTactileSim 官方评估平台/);
  assert.equal(all(root, 'wb-plots-source')[0].textContent, 'archived-source');
  window.kaiDiagnosticPlots.destroy(); window.kaiDiagnosticPlots.destroy();
  assert.equal(doc.title, 'Archived page'); assert.equal(root.children.length, 0);
});

test('expert and evaluation-rule labels preserve independent clocks and source evidence', () => {
  assert.match(source, /el\('strong', '', heading, '专家示范'\)/);
  assert.match(source, /expertTitle\.title = config\.title/);
  assert.match(source, /recorded_title: config\.title/);
  assert.match(source, /config\.source_label \|\| '已声明的专家示范视频'/);
  assert.match(source, /config\.comparison_note \|\| '未建立与模型轨迹的时间或任务阶段对应关系/);
  assert.match(source, /专家播放与模型播放互不联动/);
  assert.match(source, /阶段记录与评估规则/);
  assert.match(source, /这是诊断记录，不覆盖页首的原任务判定/);
  assert.match(source, /当前整合工程的规则不重算旧结果/);
});
