'use strict';
const assert = require('node:assert/strict');
const {leftIndex, observedIndex, stepTime, knownRuns, yMaximum, cropStyle, modelCropConfig, expertVideoURL, diagnosticPlotURL, expertDisplayConfig, normalizeBulbAngleDisplay, degreeSymbolDashboardPayload, remountLegacyAngleDashboard, displayUnit, displayedGoalReferences, goalReferenceStatus, bulbAngleState} = require('../scripts/analysis/evaluation_workbench/web/workbench.js');
const {goalReferences, normalizeBulbAngleSeries, referenceText} = require('../scripts/analysis/evaluation_v1/web/dashboard.js');
assert.equal(leftIndex([0, 1, 2], 1.5), 1);
assert.equal(leftIndex([0, 1, 2], -.5), -1);
const data = {time_s: [0, 1, 2, 4], nominal_interval_s: 1, gap_after: [false, false, true, false]};
assert.equal(observedIndex(data, 2), 2);
assert.equal(observedIndex(data, 2.1), -1);
assert.equal(observedIndex(data, 4), 3);
assert.equal(observedIndex(data, 4.1), -1);
assert.equal(observedIndex(data, NaN), -1);
assert.equal(observedIndex({...data, gap_after: [false, false, false, false]}, 3.9), 2, 'declared continuity takes precedence over nominal cadence');
assert.equal(observedIndex({...data, gap_after: undefined}, 3.6), -1);
assert.equal(observedIndex({...data, gap_after: undefined}, 3), 2, 'fallback matches accepted r2 age limit');
assert.equal(observedIndex({...data, time_s: []}, 0), -1);
assert.equal(stepTime([0, 1, 2], 1.4, -1), 1);
assert.equal(stepTime([0, 1, 2], 1, -1), 0);
assert.equal(stepTime([0, 1, 2], 1, 1), 2);
assert.equal(stepTime([0, 1, 2], 2, 1), 2);
assert.equal(yMaximum([[null, 0], [1, 5, null]]), 5.4);
assert.equal(yMaximum([[null, null]]), 1);
assert.deepEqual(knownRuns(data, [null, 5, null, 9]), [[[1, 5]], [[4, 9]]]);
assert.deepEqual(knownRuns(data, [1, 2, 3, 9]), [[[0, 1], [1, 2], [2, 3]], [[4, 9]]]);
const cfg = {status: 'verified', source_width: 1920, source_height: 1080, views: [{id: 'head', x: 214, y: 78, width: 434, height: 324}]};
const crop = cropStyle(cfg, 'head');
assert.equal(crop.aspect, '434 / 324');
assert(Math.abs(parseFloat(crop.width) * 434 / 100 - 1920) < 1e-8);
assert(Math.abs(parseFloat(crop.height) * 324 / 100 - 1080) < 1e-8);
assert.equal(cropStyle({...cfg, status: 'unknown'}, 'head'), null);
assert.equal(cropStyle(cfg, 'mosaic'), null);
assert.equal(cropStyle({...cfg, views: [{id: 'head', x: 1800, y: 0, width: 434, height: 324}]}, 'head'), null);
const mediaViews = {...cfg, sha256: 'a'.repeat(64)};
assert.equal(modelCropConfig(mediaViews, {sha256: 'a'.repeat(64), sync_status: 'mapped'}), mediaViews);
assert.equal(modelCropConfig(mediaViews, {sha256: 'a'.repeat(64), sync_status: 'unmapped'}), mediaViews, 'camera layout is independent of clock mapping');
assert.equal(modelCropConfig(mediaViews, {sha256: 'b'.repeat(64)}), null, 'crop cannot be borrowed from another video');
assert.equal(modelCropConfig({...mediaViews, status: 'full_frame_only'}, {sha256: 'a'.repeat(64)}), null);
assert.equal(modelCropConfig({...mediaViews, sha256: 'unbound'}, {sha256: 'unbound'}), null);
assert.equal(modelCropConfig(null, null), null);
const expert = {status: 'available', clock: 'independent_video_seconds', url: 'media/expert.mp4'};
assert.equal(expertVideoURL(expert), 'media/expert.mp4');
for (const url of ['../expert.mp4', 'https://example.com/expert.mp4', '//example.com/expert.mp4', 'media/../../expert.mp4', 'javascript:alert(1)', 'media/expert.mp4?x=1']) assert.equal(expertVideoURL({...expert, url}), null);
assert.equal(expertVideoURL({...expert, clock: 'model_simulation_seconds'}), null);
assert.equal(expertVideoURL(null), null);
assert.equal(diagnosticPlotURL({url: 'plots/evaluation_right_wrist_state.png'}), 'plots/evaluation_right_wrist_state.png');
assert.equal(diagnosticPlotURL({url: 'plots/bulb/evaluation_right_wrist_state.png'}), 'plots/bulb/evaluation_right_wrist_state.png');
for (const url of ['../wrist.png', 'plots/../wrist.png', 'https://example.com/wrist.png', 'plots/wrist.svg']) assert.equal(diagnosticPlotURL({url}), null);
const expertReference = {...expert, reference_relation: 'same_task_other_recording', comparison_note: '同任务的另一条录制，不是 PNG 蓝线对应 episode；初始条件可能不同。独立播放，不进行阶段对齐，不参与模型评分。色标单位必须分别核对。'};
const modelOnly = {status: 'available', comparison_kind: 'model_only'};
const displayed = expertDisplayConfig(expertReference, modelOnly);
assert.match(displayed.comparison_note, /参考源不同.*未绘制专家参考线/);
assert.doesNotMatch(displayed.comparison_note, /PNG 蓝线/);
assert.match(displayed.comparison_note, /独立播放.*不进行阶段对齐.*色标单位/);
assert.equal(expertReference.comparison_note.includes('PNG 蓝线'), true, 'display copy must not mutate archived configuration');
assert.equal(displayed.reference_relation, expertReference.reference_relation);
assert.equal(displayed.url, expertReference.url);
assert.equal(expertDisplayConfig(expertReference, {status: 'available', comparison_kind: 'reference_overlay'}), expertReference);
assert.equal(expertDisplayConfig(expertReference, {...modelOnly, status: 'missing'}), expertReference);
assert.match(expertDisplayConfig({...expertReference, comparison_note: '与 PNG 蓝线对应同一源录制；仍不代表与模型初始条件相同。'}, modelOnly).comparison_note, /属于同一录制.*未绘制专家参考线/);
assert.equal(expertDisplayConfig(null, modelOnly), null);
assert.equal(displayUnit('deg'), '°');
assert.equal(displayUnit('m'), 'm');

const legacySeries = [
  {key: 'clockwise_turns', unit: 'turn', label: '灯泡顺时针圈数', values: [0, 100 / 360, null], target: 100 / 360,
    lower: 100 / 360 - .08, upper: 100 / 360 + .08, goal_references: [{key: 'clockwise_turns', unit: 'turn', relation: 'target', target: 100 / 360, lower: 100 / 360 - .08, upper: 100 / 360 + .08,
      label: '名义机械到座角度：100° / 0.2778 圈', note: '螺纹接合后的净顺时针旋入圈数；原机械到座圈数条件换算为 ±0.08 圈。容差带只是多项到座条件之一；还需对准、实际承力、拧紧确认和松手稳定，入带不等于完整成功。'}]},
  {key: 'turns_removed', unit: 'turn', label: '移除圈数', values: [0, .25, -.5], goal_references: [
    {key: 'turns_removed', unit: 'turn', relation: 'target', target: 131.5 / 360,
      label: '名义螺纹脱扣角度：131.5° / 0.3653 圈', note: '从安装位计算的名义净旋出圈数，不是拧动次数或新的物理判据。必须另查实际脱扣、完全取出、放回及释放。到线不等于完整成功。'}]},
  {key: 'rotation_from_initial_turns', unit: 'turn', label: '相对初始圈数', values: [0, .5, 1]},
];
const legacyPayload = {visualization: {task_series: legacySeries}, report: {trial: {trial_id: 'legacy'}}};
const legacyConfig = {task_presentation: {primary_task_keys: ['clockwise_turns', 'insertion_depth_m', 'rotation_from_initial_turns'], secondary_task_keys: ['turns_removed', 'thread_error_m']}};
const archivedBytes = JSON.stringify({legacyPayload, legacyConfig});
const normalizedAngles = normalizeBulbAngleDisplay(legacyPayload, legacyConfig);
assert.equal(normalizedAngles.converted, true);
assert.deepEqual(normalizedAngles.payload.visualization.task_series.map(series => [series.key, series.unit, series.label]), [
  ['clockwise_angle_deg', 'deg', '灯泡净旋入角度（顺时针）'],
  ['angle_removed_deg', 'deg', '灯泡净旋出角度（逆时针）'],
  ['rotation_from_initial_deg', 'deg', '相对实际初态的净旋入角度'],
]);
assert.deepEqual(normalizedAngles.payload.visualization.task_series.map(series => series.values), [[0, 100, null], [0, 90, -180], [0, 180, 360]]);
const convertedTurn = normalizedAngles.payload.visualization.task_series[0];
assert.equal(convertedTurn.target, 100);
assert.equal(convertedTurn.lower, 100 - 28.8);
assert.equal(convertedTurn.upper, 100 + 28.8);
assert.equal(convertedTurn.goal_references[0].key, 'clockwise_angle_deg');
assert.equal(convertedTurn.goal_references[0].unit, 'deg');
assert.equal(convertedTurn.goal_references[0].target, 100);
assert.equal(convertedTurn.goal_references[0].lower, 100 - 28.8);
assert.equal(convertedTurn.goal_references[0].upper, 100 + 28.8);
assert.equal(convertedTurn.goal_references[0].label, '名义机械到座旋入角度：100°');
assert.match(convertedTurn.goal_references[0].note, /±28\.8°/);
assert.doesNotMatch(convertedTurn.goal_references[0].note, /0\.2778|0\.08|圈/);
assert.match(convertedTurn.goal_references[0].note, /还需对准.*入带不等于完整成功/);
const convertedRemovalReference = normalizedAngles.payload.visualization.task_series[1].goal_references[0];
assert.equal(convertedRemovalReference.label, '名义螺纹脱扣角度：131.5°');
assert.doesNotMatch(convertedRemovalReference.label + convertedRemovalReference.note, /0\.3653|圈/);
assert.match(convertedRemovalReference.note, /必须另查.*到线不等于完整成功/);
assert.deepEqual(normalizedAngles.config.task_presentation.primary_task_keys, ['clockwise_angle_deg', 'insertion_depth_m', 'rotation_from_initial_deg']);
assert.deepEqual(normalizedAngles.config.task_presentation.secondary_task_keys, ['angle_removed_deg', 'thread_error_m']);
assert.equal(JSON.stringify({legacyPayload, legacyConfig}), archivedBytes, 'legacy compatibility must not mutate embedded archive data');

const normalizedAgain = normalizeBulbAngleDisplay(normalizedAngles.payload, normalizedAngles.config);
assert.equal(normalizedAgain.converted, false, 'canonical degree series must not be converted twice');
assert.deepEqual(normalizedAgain.payload.visualization.task_series.map(series => series.values), [[0, 100, null], [0, 90, -180], [0, 180, 360]]);
const dashboardAngles = normalizeBulbAngleSeries(legacySeries[0]);
assert.equal(dashboardAngles.unit, 'deg');
assert.equal(dashboardAngles.goal_references[0].unit, 'deg');
assert.equal(dashboardAngles.values[1], 100, 'dashboard legacy adaptation must convert values exactly once');
assert.equal(goalReferences(dashboardAngles)[0].target, 100, 'converted goal reference remains bound by canonical key and unit');
assert.equal(referenceText(goalReferences(dashboardAngles)[0]), '机械目标 100 °');
assert.equal(dashboardAngles.goal_references[0].label, convertedTurn.goal_references[0].label);
assert.equal(dashboardAngles.goal_references[0].note, convertedTurn.goal_references[0].note);
assert.equal(normalizeBulbAngleSeries(dashboardAngles), dashboardAngles, 'canonical dashboard series must not be converted twice');
assert.equal(normalizedAgain.payload.visualization.task_series[0].unit, 'deg', 'canonical exported unit remains deg');

const legacyDashboardPayload = degreeSymbolDashboardPayload(normalizedAngles.payload);
assert.notEqual(legacyDashboardPayload, normalizedAngles.payload);
assert.equal(legacyDashboardPayload.visualization.task_series[0].unit, '°');
assert.equal(legacyDashboardPayload.visualization.task_series[0].goal_references[0].unit, '°');
assert.equal(normalizedAngles.payload.visualization.task_series[0].unit, 'deg', 'legacy dashboard display clone must not mutate canonical payload');
assert.equal(degreeSymbolDashboardPayload(legacyPayload), legacyPayload, 'turn payload is normalized before legacy-dashboard display adaptation');

const legacyMetadata = {...legacyConfig, goal_references: {status: 'verified', references: legacySeries[0].goal_references},
  task_summary: [{key: 'clockwise_turns', unit: 'turn', value: -.5, status: 'recorded'}]};
const metadataBefore = JSON.stringify(legacyMetadata), normalizedMetadata = normalizeBulbAngleDisplay(legacyPayload, legacyMetadata);
assert.equal(normalizedMetadata.config.goal_references.references[0].unit, 'deg');
assert.equal(normalizedMetadata.config.goal_references.references[0].target, 100);
assert.equal(normalizedMetadata.config.task_summary[0].key, 'clockwise_angle_deg');
assert.equal(normalizedMetadata.config.task_summary[0].value, -180);
assert.equal(JSON.stringify(legacyMetadata), metadataBefore);
assert.equal(normalizeBulbAngleDisplay(normalizedMetadata.payload, normalizedMetadata.config).converted, false);

const angleSeries = {...convertedTurn, time_s: [0, 1, 2], values: [0, 35, 70], gap_after: [false, true, false]};
const angleViz = {clock_aligned: true, task_series: [angleSeries]};
const angleConfig = {task: 'bulb-screw', goal_references: {status: 'verified'}, task_summary: [{key: 'clockwise_angle_deg', unit: 'deg', value: 64.5, status: 'recorded'}]};
assert.equal(bulbAngleState(angleViz, angleConfig, 1).current, 35);
assert.equal(bulbAngleState(angleViz, angleConfig, 1).terminal, 64.5);
assert.equal(bulbAngleState(angleViz, angleConfig, 1).target, 100);
assert.equal(bulbAngleState(angleViz, angleConfig, 1).lower, 71.2);
assert.equal(bulbAngleState(angleViz, angleConfig, 1.5).current, null, 'known gap cannot be filled by holding previous angle');
assert.equal(bulbAngleState(angleViz, angleConfig, 3).current, null, 'no end extrapolation');
assert.equal(bulbAngleState({...angleViz, clock_aligned: false}, angleConfig, 1).current, null);
assert.equal(bulbAngleState(angleViz, {...angleConfig, task_summary: []}, 2).terminal, null, 'last curve sample is not terminal evidence');
assert.equal(bulbAngleState(angleViz, {...angleConfig, task_summary: [{...angleConfig.task_summary[0], status: 'unknown'}]}, 1).terminal, null);
assert.equal(bulbAngleState(angleViz, {...angleConfig, task_summary: [{...angleConfig.task_summary[0], unit: 'turn'}]}, 1).terminal, null);
assert.equal(bulbAngleState(angleViz, {...angleConfig, goal_references: {status: 'unavailable'}}, 1).target, null);
assert.equal(bulbAngleState(angleViz, {...angleConfig, task: 'poker-draw'}, 1), null);
assert.equal(bulbAngleState(angleViz, {...angleConfig, task: 'bulb-screw-inplace'}, 1).target, null, 'absolute goal cannot be reused for initial-relative angle');
assert.equal(bulbAngleState({task_series: []}, angleConfig, 0).current, null);
assert.equal(goalReferenceStatus(angleViz, angleConfig).count, 1);
assert.equal(goalReferenceStatus(angleViz, angleConfig).bands, 1);
assert.match(goalReferenceStatus(angleViz, angleConfig).note, /不等于完整任务成功/);
assert.equal(goalReferenceStatus({task_series: []}, angleConfig).status, 'unavailable', 'verified source does not imply a visible targetline');
assert.equal(goalReferenceStatus(angleViz, {goal_references: {status: 'unavailable'}}).count, 0);
assert.match(goalReferenceStatus({}, {goal_references: {status: 'not_applicable', reason: '归档没有任务状态记录。'}}).note, /没有任务状态记录/);
for (const input of [angleSeries, {...angleSeries, goal_references: [{...angleSeries.goal_references[0], unit: 'turn'}]},
  {...angleSeries, goal_references: [{...angleSeries.goal_references[0], target: NaN}]},
  {...angleSeries, goal_references: [{...angleSeries.goal_references[0], lower: 120, upper: 160}]}]) {
  const accepted = displayedGoalReferences(input), dashboardAccepted = goalReferences(input);
  assert.deepEqual(accepted.map(({key, unit, target, lower, upper, relation}) => ({key, unit, target, lower, upper, relation})),
    dashboardAccepted.map(({key, unit, target, lower, upper, relation}) => ({key, unit, target, lower, upper, relation})), 'summary acceptance agrees with chart references');
}

const savedDashboard = globalThis.KaiEvaluationDashboard;
const savedView = globalThis.kaiEvaluationView;
try {
  const calls = [];
  globalThis.kaiEvaluationView = {destroy() { calls.push('destroy'); }};
  globalThis.KaiEvaluationDashboard = {mount(host, payload) {
    calls.push(['mount', host, payload]); return {kind: 'replacement'};
  }};
  const legacyHost = {kind: 'host'};
  assert.equal(remountLegacyAngleDashboard(legacyHost, normalizedAngles.payload), true);
  assert.equal(calls[0], 'destroy');
  assert.equal(calls[1][0], 'mount');
  assert.equal(calls[1][1], legacyHost);
  assert.equal(calls[1][2].visualization.task_series[0].unit, '°');
  assert.equal(globalThis.kaiEvaluationView.kind, 'replacement');

  let newDashboardMounted = false;
  globalThis.kaiEvaluationView = {destroy() { throw new Error('new dashboard must not be remounted'); }};
  globalThis.KaiEvaluationDashboard = {mount() { newDashboardMounted = true; }, normalizeBulbAngleSeries() {}};
  assert.equal(remountLegacyAngleDashboard(legacyHost, normalizedAngles.payload), false);
  assert.equal(newDashboardMounted, false);
} finally {
  globalThis.KaiEvaluationDashboard = savedDashboard;
  if (savedView === undefined) delete globalThis.kaiEvaluationView;
  else globalThis.kaiEvaluationView = savedView;
}
console.log('Workbench helpers: native clocks, gaps, keyboard, fixed scales, verified crops and isolated expert URL passed.');
