'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {goalReferences, chartRange, referenceNumber, referenceText, knownSegments, observedIndex} = require('../scripts/analysis/evaluation_v1/web/dashboard.js');

const turnReference = () => ({key: 'clockwise_turns', unit: 'turn', target: 100 / 360, relation: 'target',
  label: '机械到座目标 100°', note: '仍需啮合、对准和机械接触。', lower: 100 / 360 - .08, upper: 100 / 360 + .08});
const turnSeries = (...references) => ({key: 'clockwise_turns', unit: 'turn', values: [0, .02, .04], goal_references: references});

test('legacy or other-task series without explicit references never infer a target', () => {
  assert.deepEqual(goalReferences({key: 'clockwise_turns', unit: 'turn'}), []);
  assert.deepEqual(goalReferences({key: 'turns_removed', unit: 'turn'}), []);
  assert.deepEqual(goalReferences({key: 'cleaned_fraction', unit: 'fraction'}), []);
  assert.deepEqual(goalReferences(null), []);
  assert.deepEqual(chartRange([{values: [0, 2]}]), {low: 0, high: 2.16});
  assert.deepEqual(chartRange([{values: [0, 2]}], []), {low: 0, high: 2.16});
  assert.deepEqual(chartRange([{values: [null, 0]}]), {low: 0, high: 1});
  assert.deepEqual(chartRange([{values: [-2, 1]}]), {low: -2, high: 1.24});
});

test('source-verified mechanical target and full tolerance are included even for tiny or zero motion', () => {
  const refs = goalReferences(turnSeries(turnReference()));
  assert.equal(refs.length, 1);
  for (const values of [[0, 0], [0, 48.04 / 360], [null, null]]) {
    const range = chartRange([{values}], refs);
    assert(range.high > refs[0].upper);
    assert(range.low < refs[0].lower);
    for (const value of values.filter(value => typeof value === 'number')) assert(value >= range.low && value <= range.high);
  }
});

test('signed depth clearance uses an at-most threshold, never manufactures a symmetric band', () => {
  const input = {key: 'insertion_depth_m', unit: 'm', target: -.003, relation: 'at_most', label: '完全离座阈值'};
  const refs = goalReferences({key: input.key, unit: input.unit, goal_references: [input]});
  assert.equal(refs[0].lower, undefined);
  assert.equal(refs[0].upper, undefined);
  assert.match(referenceText(refs[0]), /参考阈值 ≤ -0\.003 m/);
  const range = chartRange([{values: [.006, .007]}], refs);
  assert(range.low < -.003);
  assert(range.high > .007);
});

test('mismatched, malformed, nonfinite and coercible inputs are rejected rather than silently plotted', () => {
  const good = turnReference();
  for (const ref of [null, [], 'bad', {...good, key: 'turns_removed'}, {...good, unit: 'deg'},
    {...good, target: '0.2778'}, {...good, target: NaN}, {...good, target: Infinity}, {...good, target: -Infinity},
    {...good, target: Number.MAX_VALUE}, {...good, relation: 'approximately'}, {...good, relation: null}]) {
    assert.deepEqual(goalReferences(turnSeries(ref)), [], JSON.stringify(ref));
  }
  assert.deepEqual(goalReferences({key: good.key, goal_references: [good]}), []);
  assert.deepEqual(goalReferences({unit: good.unit, goal_references: [good]}), []);
});

test('malformed tolerance is omitted without losing an otherwise valid target', () => {
  const good = turnReference();
  for (const bounds of [{lower: NaN}, {upper: Infinity}, {lower: '.19'}, {lower: .4}, {upper: .1}, {lower: .2, upper: .2}, {lower: undefined, upper: undefined}]) {
    const refs = goalReferences(turnSeries({...good, ...bounds}));
    assert.equal(refs.length, 1);
    assert.equal(refs[0].lower, undefined);
    assert.equal(refs[0].upper, undefined);
  }
});

test('reference text keeps physical values and units readable, not rounded to 0.01 m', () => {
  assert.equal(referenceNumber(.007111111111), '0.007111');
  assert.equal(referenceNumber(.277777777777), '0.277778');
  assert.equal(referenceNumber(0), '0');
  assert.notEqual(referenceNumber(1e-10), '0');
  assert.equal(referenceNumber(null), '—');
  assert.equal(referenceText({target: .007111111111, unit: 'm', relation: 'target'}), '机械目标 0.007111 m');
});

test('whiteboard average cleanliness is an ideal reference, never a mechanical success gate', () => {
  const ref = {key: 'cleaned_fraction', unit: 'fraction', target: 1, relation: 'target', reference_kind: 'ideal'};
  const parsed = goalReferences({key: ref.key, unit: ref.unit, goal_references: [ref]});
  assert.equal(parsed[0].reference_kind, 'ideal');
  assert.equal(referenceText(parsed[0]), '理想参考 1 fraction');
  assert.match(referenceText({target: 1e-9, unit: 'fraction', relation: 'at_most'}), /≤ 1\.0000e-9 fraction/);
  assert.notEqual(referenceNumber(1e-9), '0');
});

test('RAM orientation uses an at-most upper limit with an explicit zero-to-limit range', () => {
  const target = Math.PI / 180;
  const input = {key: 'orientation_error_rad', unit: 'rad', target, relation: 'at_most', lower: 0, upper: target,
    label: '姿态误差上限 1°', note: '理想为0°；仅该项允许范围，不代表完整成功。'};
  const refs = goalReferences({key: input.key, unit: input.unit, goal_references: [input]});
  assert.deepEqual(refs[0], input);
  assert.equal(referenceText(refs[0]), '参考阈值 ≤ 0.017453 rad');
  for (const values of [[0, 0], [null, null], [.4, .3]]) {
    const range = chartRange([{values}], refs);
    assert(range.low < 0);
    assert(range.high > target);
  }
});

test('RAM insertion includes the 6 mm target and 5.8–6.2 mm tolerance in range', () => {
  const input = {key: 'insertion_depth_m', unit: 'm', target: .006, lower: .0058, upper: .0062, relation: 'target'};
  const refs = goalReferences({key: input.key, unit: input.unit, goal_references: [input]});
  const range = chartRange([{values: [0, .001, null]}], refs);
  assert(range.high > .0062);
  assert.equal(refs[0].lower, .0058);
  assert.equal(refs[0].upper, .0062);
  assert.equal(referenceText(refs[0]), '机械目标 0.006 m');
});

test('Poker current and peak displacement share a lower-bound line without inventing a success band', () => {
  for (const key of ['card_displacement_now_mm', 'card_displacement_peak_mm']) {
    const ref = {key, unit: 'mm', target: 50, relation: 'at_least'};
    const series = {key, unit: 'mm', values: [0, 10, -5], goal_references: [ref]};
    const before = JSON.stringify(series), refs = goalReferences(series);
    assert.equal(refs.length, 1);
    assert.equal(refs[0].lower, undefined);
    assert.equal(refs[0].upper, undefined);
    assert.equal(referenceText(refs[0]), '参考阈值 ≥ 50 mm');
    assert(chartRange([{values: series.values}], refs).high > 50);
    assert(chartRange([{values: [null, null]}], refs).high > 50);
    assert.equal(JSON.stringify(series), before, 'No conversion to progress or mutation of raw displacement');
  }
});

test('finite single-sided ranges cannot cross to the forbidden side of their threshold', () => {
  const values = [
    {relation: 'at_most', target: 1, lower: 0, upper: 2},
    {relation: 'at_most', target: 1, lower: .5, upper: .9},
    {relation: 'at_least', target: 50, lower: 40, upper: 70},
    {relation: 'at_least', target: 50, lower: 50, upper: Infinity},
  ];
  for (const value of values) {
    const refs = goalReferences(turnSeries({...value, key: 'clockwise_turns', unit: 'turn'}));
    assert.equal(refs.length, 1);
    assert.equal(refs[0].lower, undefined);
    assert.equal(refs[0].upper, undefined);
  }
  const refs = goalReferences(turnSeries({key: 'clockwise_turns', unit: 'turn', relation: 'at_least', target: 50, lower: 50, upper: 70}));
  assert.equal(refs[0].lower, 50);
  assert.equal(refs[0].upper, 70);
});

test('rendering metadata is copied without mutating original values, gaps or source references', () => {
  const raw = turnReference(), series = turnSeries(raw);
  series.time_s = [0, .1, 1, 1.1]; series.values = [0, null, .04, .05]; series.gap_after = [false, true, false, false];
  const before = JSON.stringify(series), refs = goalReferences(series);
  chartRange([{values: series.values}], refs);
  refs[0].target = .5;
  assert.equal(JSON.stringify(series), before);
  assert.deepEqual(knownSegments(series.time_s, series.values, series.gap_after, .1), [[[0, 0]], [[1, .04], [1.1, .05]]]);
  assert.equal(observedIndex(series.time_s, .5, .1, series.gap_after), -1);
});
