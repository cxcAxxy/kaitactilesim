'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {test} = require('node:test');
const {mountSectionNavigation} = require('../scripts/analysis/evaluation_workbench/web/workbench.js');

class Element {
  constructor(tag, doc) { this.tagName = tag.toUpperCase(); this.ownerDocument = doc; this.children = []; this.attributes = {}; this.events = new Map(); this.scrolls = []; }
  appendChild(node) { node.remove(); this.children.push(node); node.parentNode = this; return node; }
  remove() { if (this.parentNode) { this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; } }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(type, callback) { this.events.set(type, callback); }
  removeEventListener(type, callback) { if (this.events.get(type) === callback) this.events.delete(type); }
  click() { const event = {prevented: false, preventDefault() { this.prevented = true; }}; this.events.get('click')?.(event); return event; }
  scrollIntoView(options) { this.scrolls.push(options); }
}
function fixture() { const doc = {createElement: tag => new Element(tag, doc)}; return new Element('div', doc); }

test('section links scroll within this page and expert entry unfolds its independent reference', () => {
  const host = fixture(), model = host.ownerDocument.createElement('main'), expert = host.ownerDocument.createElement('details');
  expert.open = false;
  const modelClock = {value: 7}, expertClock = {value: 3};
  const mounted = mountSectionNavigation(host, [
    {id: 'wb-task-progress', label: '任务进度', node: model},
    {id: 'wb-expert-reference', label: '专家参考 ↗', node: expert},
  ]);
  assert.equal(mounted.node.attributes['aria-label'], '轨迹分析分区');
  const [taskLink, expertLink] = mounted.node.children;
  assert.equal(taskLink.href, '#wb-task-progress'); assert.equal(expertLink.href, '#wb-expert-reference');
  assert.match(expertLink.className, /wb-reference-link/);
  assert.equal(expertLink.click().prevented, true); assert.equal(expert.open, true);
  assert.deepEqual(expert.scrolls, [{behavior: 'auto', block: 'start'}]);
  assert.equal(taskLink.click().prevented, true); assert.equal(model.scrolls.length, 1);
  assert.deepEqual(modelClock, {value: 7}); assert.deepEqual(expertClock, {value: 3});
  mounted.destroy(); assert.equal(host.children.length, 0);
  expert.open = false; expertLink.click(); assert.equal(expert.open, false, 'detached navigation does not reopen media');
});

test('only mounted local sections with safe fragment identifiers create navigation entries', () => {
  const host = fixture(), target = host.ownerDocument.createElement('section');
  const mounted = mountSectionNavigation(host, [
    {id: 'wb-tactile-analysis', label: '触觉分析', node: target},
    {id: 'wb-unavailable', label: '缺测区域', node: null},
    {id: '../elsewhere', label: 'invalid', node: target},
    {id: 'https://example.com', label: 'invalid', node: target},
  ]);
  assert.equal(mounted.node.children.length, 1);
  assert.equal(target.id, 'wb-tactile-analysis');
  mounted.destroy(); mounted.destroy();
});

test('studio keeps model-first layout, all native plots, cleanup and independent expert clock', () => {
  const source = fs.readFileSync(require.resolve('../scripts/analysis/evaluation_workbench/web/workbench.js'), 'utf8');
  const css = fs.readFileSync(require.resolve('../scripts/analysis/evaluation_workbench/web/workbench.css'), 'utf8');
  assert.match(source, /panel\.open = false/);
  assert.match(source, /move\(root\.querySelector\('\.ke-spatial-panel'\), tactile\)/);
  assert.match(source, /section\(right, '01', '任务关键指标'/);
  assert.match(source, /mountExpertPlayer\(root, expertDisplayConfig/);
  assert.match(source, /if \(expertPlayer\) root\.insertBefore\(expertPlayer\.node, bottom\)/);
  assert.match(source, /mountSectionNavigation\(transport/);
  assert.match(source, /sectionNavigation\.destroy\(\)/);
  assert.match(source, /'normal-total'[\s\S]*'tangent-total'[\s\S]*'normal-fingers'[\s\S]*'tangent-fingers'/);
  assert.match(source, /video\.pause\(\); video\.currentTime = clamp\(value, 0, total\)/);
  assert.match(css, /\.wb-root\.wb-studio \.wb-workspace\{grid-template-columns:/);
  assert.match(css, /\.wb-root\.wb-studio \.wb-tactile-workspace\{display:grid/);
  assert.match(css, /scroll-margin-top:calc\(var\(--wb-sticky-top\)/);
  assert.match(css, /prefers-reduced-motion:reduce/);
});
