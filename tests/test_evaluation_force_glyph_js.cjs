'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const component = require('../scripts/analysis/evaluation_v1/web/dashboard.js');

function close(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)),
    `${message || 'value'}: expected ${expected}, received ${actual}`);
}

function source() {
  return {
    time_s: [0, .033, .067, .1],
    nominal_interval_s: 1 / 30,
    gap_after: [false, false, false, false],
    normal_force_n: [[0], [1], [8], [2]],
    center_xy: [[null], [[1, 2]], [[2, 3]], [[3, 4]]],
    normal_change_n: [[null], [null], [null], [2]],
    tangent_change_n: [[null], [null], [null], [[-3, 4]]],
  };
}

test('glyph helpers are independently reusable without a browser DOM', () => {
  for (const name of ['glyphSample', 'glyphRadius', 'glyphBounds', 'glyphPoint', 'glyphVector']) {
    assert.equal(typeof component[name], 'function', name);
  }
});

test('glyph samples retain the native force peak and report a past-only display hold', () => {
  const glyph = source();
  const exact = component.glyphSample(glyph, .067, 0);
  assert.equal(exact.index, 2);
  assert.equal(exact.sampleTime, .067);
  assert.equal(exact.normal, 8);
  assert.equal(exact.lag, 0);
  const held = component.glyphSample(glyph, .08, 0);
  assert.equal(held.index, 2);
  assert.equal(held.normal, 8);
  assert.deepEqual(held.center, [2, 3]);
  close(held.lag, .013);
  // A displayed hold must not create a new t-0.1 endpoint or recompute deltas.
  assert.equal(held.normalChange, null);
  assert.equal(held.tangentChange, null);
  assert.deepEqual(component.glyphSample(glyph, .1, 0).tangentChange, [-3, 4]);
});

test('glyph sampling never fills outside the observed clock or across a declared gap', () => {
  const glyph = source();
  glyph.gap_after[1] = true;
  assert.equal(component.glyphSample(glyph, .033, 0).normal, 1);
  for (const time of [-.01, .05, .11, NaN]) {
    const value = component.glyphSample(glyph, time, 0);
    assert.deepEqual(value, {index: -1, sampleTime: null, lag: null,
      normal: null, center: null, normalChange: null, tangentChange: null});
  }
  assert.equal(component.glyphSample(glyph, .067, 0).normal, 8);
});

test('glyph sampling distinguishes zero force and zero change from missing measurements', () => {
  const glyph = source();
  glyph.normal_change_n[0] = [0];
  glyph.tangent_change_n[0] = [[0, 0]];
  const zero = component.glyphSample(glyph, 0, 0);
  assert.equal(zero.normal, 0);
  assert.equal(zero.center, null);
  assert.equal(zero.normalChange, 0);
  assert.deepEqual(zero.tangentChange, [0, 0]);
  glyph.normal_force_n[1] = [null];
  glyph.center_xy[1] = [null];
  const missing = component.glyphSample(glyph, .033, 0);
  assert.equal(missing.index, 1);
  assert.equal(missing.normal, null);
  assert.equal(missing.center, null);
  assert.equal(missing.normalChange, null);
  assert.equal(missing.tangentChange, null);
  assert.equal(component.glyphSample(glyph, .033, 9).normal, null);
  assert.equal(component.glyphSample(null, 0, 0).index, -1);
});

test('glyph readouts reject invalid forces and incomplete two-component vectors', () => {
  for (const invalid of [null, [1], [1, 2, 3], [1, NaN], [Infinity, 0], ['1', 2]]) {
    const glyph = source();
    glyph.center_xy[3] = [invalid];
    glyph.tangent_change_n[3] = [invalid];
    const value = component.glyphSample(glyph, .1, 0);
    assert.equal(value.center, null);
    assert.equal(value.tangentChange, null);
    assert.equal(value.normal, 2);
  }
  for (const invalid of [null, -1, NaN, Infinity, '2', true]) {
    const glyph = source();
    glyph.normal_force_n[3] = [invalid];
    assert.equal(component.glyphSample(glyph, .1, 0).normal, null);
  }
});

test('normal-force circle area is proportional to force under one fixed scale', () => {
  const maximum = 16, maximumRadius = 20;
  for (const force of [0, .25, 1, 4, 16, 64]) {
    const radius = component.glyphRadius(force, maximum, maximumRadius);
    close(radius * radius, maximumRadius * maximumRadius * force / maximum,
      `area proportionality at ${force} N`);
  }
  close(component.glyphRadius(4, maximum, maximumRadius) /
    component.glyphRadius(1, maximum, maximumRadius), 2);
  assert.ok(component.glyphRadius(64, maximum, maximumRadius) > maximumRadius,
    'an above-scale value must not silently saturate at the global peak radius');
});

test('normal-force scaling preserves zero but never turns unknown scale into zero force', () => {
  assert.equal(component.glyphRadius(0, 0, 20), 0);
  assert.equal(component.glyphRadius(1, 0, 20), null);
  for (const invalid of [null, undefined, NaN, Infinity, -1, '1', true]) {
    assert.equal(component.glyphRadius(invalid, 4, 20), null);
    assert.equal(component.glyphRadius(1, invalid, 20), null);
    assert.equal(component.glyphRadius(1, 4, invalid), null);
  }
  assert.equal(component.glyphRadius(1, 4, 0), null);
});

test('layout bounds are common across channels of the same unit without mixing mm and indices', () => {
  const bounds = component.glyphBounds([
    {coordinate_status: 'index', unit: 'index', taxel_xy: [[0, 0], [4, 6]]},
    {coordinate_status: 'index', unit: 'index', taxel_xy: [[-2, 1], [6, 5]]},
    {coordinate_status: 'physical', unit: 'mm', taxel_xy: [[-20, -10], [30, 10]]},
    {coordinate_status: 'physical', unit: 'mm', taxel_xy: [[40, 15], [NaN, 1], [1]]},
    {coordinate_status: 'index', unit: 'index', taxel_xy: []},
  ]);
  assert.deepEqual(bounds.index, {minX: -2, maxX: 6, minY: 0, maxY: 6,
    centerX: 2, centerY: 3, range: 8});
  assert.deepEqual(bounds.mm, {minX: -20, maxX: 40, minY: -10, maxY: 15,
    centerX: 10, centerY: 2.5, range: 60});
  assert.deepEqual(component.glyphBounds([]), {});
  assert.deepEqual(component.glyphBounds([{taxel_xy: [[2, 3]]}]).index,
    {minX: 2, maxX: 2, minY: 3, maxY: 3, centerX: 2, centerY: 3, range: 1});
});

test('glyph projection uses equal axis scale and a shared origin across channels', () => {
  const bounds = component.glyphBounds([
    {coordinate_status: 'index', unit: 'index', taxel_xy: [[0, 0], [4, 6]]},
  ]).index;
  const center = component.glyphPoint([2, 3], bounds, 120, 150, 100, true);
  const right = component.glyphPoint([3, 3], bounds, 120, 150, 100, true);
  const down = component.glyphPoint([2, 4], bounds, 120, 150, 100, true);
  assert.deepEqual(center, [150, 100]);
  close(right[0] - center[0], 20);
  close(down[1] - center[1], 20);
  close(Math.hypot(right[0] - center[0], right[1] - center[1]),
    Math.hypot(down[0] - center[0], down[1] - center[1]));
  assert.equal(component.glyphPoint(null, bounds, 120, 150, 100, true), null);
  assert.equal(component.glyphPoint([0, 0], null, 120, 150, 100, true), null);
});

test('index row zero stays at the top while physical local y points upward', () => {
  const bounds = {centerX: 2, centerY: 3, range: 6};
  const indexTop = component.glyphPoint([2, 0], bounds, 120, 100, 100, true);
  const indexBottom = component.glyphPoint([2, 6], bounds, 120, 100, 100, true);
  assert.deepEqual(indexTop, [100, 40]);
  assert.deepEqual(indexBottom, [100, 160]);
  const physicalLow = component.glyphPoint([2, 0], bounds, 120, 100, 100, false);
  const physicalHigh = component.glyphPoint([2, 6], bounds, 120, 100, 100, false);
  assert.deepEqual(physicalLow, [100, 160]);
  assert.deepEqual(physicalHigh, [100, 40]);
});

test('tangent compass preserves signed components and uses fixed linear force-vector scale', () => {
  const value = component.glyphVector([3, 4], 10, 20);
  assert.deepEqual(value, {magnitude: 5, dx: 6, dy: -8});
  close(Math.hypot(value.dx, value.dy), 20 * value.magnitude / 10);
  assert.deepEqual(component.glyphVector([-3, -4], 10, 20),
    {magnitude: 5, dx: -6, dy: 8});
  assert.deepEqual(component.glyphVector([12, 16], 10, 20),
    {magnitude: 20, dx: 24, dy: -32});
  assert.deepEqual(component.glyphVector([0, 0], 0, 20),
    {magnitude: 0, dx: 0, dy: 0});
});

test('tangent compass never substitutes a direction for missing vectors or unknown scale', () => {
  for (const vector of [null, [1], [1, 2, 3], [null, 0], [0, NaN], [Infinity, 1]]) {
    assert.equal(component.glyphVector(vector, 10, 20), null);
  }
  assert.equal(component.glyphVector([1, 0], 0, 20), null);
  for (const invalid of [null, NaN, Infinity, -1]) {
    assert.equal(component.glyphVector([1, 0], invalid, 20), null);
    assert.equal(component.glyphVector([1, 0], 10, invalid), null);
  }
});
