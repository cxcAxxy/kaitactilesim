'use strict';
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {trialFolder,sharedScales} = require('../scripts/analysis/evaluation_v2/web/hub.js');
test('only generated same-root report paths are accepted', () => {
  assert.equal(trialFolder('003_whiteboard-wipe_seed_000/report.html'),'003_whiteboard-wipe_seed_000/');
  ['../report.html','https://other/report.html','//host/report.html','/root/report.html','x/%2e%2e/report.html','x\\report.html','x/report.html?q=1','x/y/report.html','<script>/report.html','x/report.html\n'].forEach(path => assert.equal(trialFolder(path),null));
});
test('shared presentation ranges use maxima without mutating data', () => {
  const first = {contact_glyph:{force_scale_max_n:3,tangent_change_scale_max_n:.8},spatial_frames:{scale_max_n:2}};
  const second = {contact_glyph:{force_scale_max_n:8,tangent_change_scale_max_n:.4},spatial_frames:{scale_max_n:1}};
  const before = JSON.stringify([first,second]);
  assert.deepEqual(sharedScales([first,second]),{normal_force_max_n:8,taxel_force_max_n:2,tangent_change_max_n:.8});
  assert.equal(JSON.stringify([first,second]),before);
  assert.deepEqual(sharedScales([{}]),{normal_force_max_n:null,taxel_force_max_n:null,tangent_change_max_n:null});
  assert.equal(sharedScales([{contact_glyph:{force_scale_max_n:0}}]).normal_force_max_n,0);
});
