'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const component = require('../scripts/analysis/evaluation_v1/web/dashboard.js');

test('left selection never uses a future frame and preserves the final off-grid frame', () => {
  const time = [0, .1, .2, .237];
  assert.equal(component.leftIndex(time, -.1), -1);
  assert.equal(component.leftIndex(time, .199), 1);
  assert.equal(component.leftIndex(time, .237), 3);
  assert.equal(component.leftIndex(time, 9), 3);
  assert.equal(component.observedIndex(time, 9, .1, []), -1);
});

test('readouts do not hold across gaps or beyond observation end', () => {
  const time = [0, .1, 1, 1.1];
  assert.equal(component.observedIndex(time, .1, .1, [false, true]), 1);
  assert.equal(component.observedIndex(time, .15, .1, [false, true]), -1);
  assert.equal(component.observedIndex(time, .4, .1, []), -1);
  assert.equal(component.observedIndex(time, 1.05, .1, []), 2);
  assert.equal(component.observedIndex(time, 1.2, .1, []), -1);
});

test('curves keep zero, isolate peaks, break at null and declared gaps', () => {
  const runs = component.knownSegments([0, .1, .2, .3, .4, .5], [0, 100, null, 0, 3, 0], [false, false, false, true, false, false], .1);
  assert.deepEqual(runs, [[[0, 0], [.1, 100]], [[.3, 0]], [[.4, 3], [.5, 0]]]);
});

test('curves also break for inferred long gaps', () => {
  assert.deepEqual(component.knownSegments([0, .1, 1], [0, 1, 9], [], .1), [[[0, 0], [.1, 1]], [[1, 9]]]);
});

test('declared gap flags take precedence over fallback cadence thresholds', () => {
  const times = [0, .1, .3], flags = [false, false, false];
  assert.equal(component.observedIndex(times, .28, .1, flags), 1);
  assert.deepEqual(component.knownSegments(times, [1, 2, 3], flags, .1), [[[0, 1], [.1, 2], [.3, 3]]]);
});

test('media only uses safe local relative URLs', () => {
  assert.equal(component.relativeURL('media/rollout.mp4'), 'media/rollout.mp4');
  assert.equal(component.relativeURL('/evaluation/run/media/rollout.mp4'), '/evaluation/run/media/rollout.mp4');
  for (const value of ['https://example.com/a.mp4', '//example.com/a', '../a.mp4', '%2e%2e/a.mp4', 'javascript:alert(1)', 'media\\a.mp4', 'media/%252e%252e/a.mp4', '\na.mp4', '/%2fexample.com/a', '/%0aevil.mp4']) assert.equal(component.relativeURL(value), null, value);
});

test('no unsafe HTML insertion or shared DOM identifiers are used', () => {
  const source = fs.readFileSync(path.join(__dirname, '../scripts/analysis/evaluation_v1/web/dashboard.js'), 'utf8');
  assert.doesNotMatch(source, /\.innerHTML\s*=|insertAdjacentHTML|\beval\s*\(|new Function\s*\(/);
  assert.doesNotMatch(source, /document\.getElementById|document\.querySelector|\.id\s*=/);
  assert.equal(typeof component.mount, 'function');
});

test('CSS rules are component scoped', () => {
  const css = fs.readFileSync(path.join(__dirname, '../scripts/analysis/evaluation_v1/web/dashboard.css'), 'utf8');
  for (const match of css.matchAll(/(?:^|\})\s*([^{}]+)\{/g)) {
    const selector = match[1].trim();
    if (selector.startsWith('@')) continue;
    for (const item of selector.split(',')) assert.ok(item.trim().startsWith('.kai-eval'), item);
  }
});
