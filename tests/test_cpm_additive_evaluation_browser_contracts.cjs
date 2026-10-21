'use strict';
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {compareCatalogs, checkRetainedBytes, CPM_SHA, LABEL} =
  require('../scripts/analysis/check_cpm_additive_evaluation_workbench_browser.cjs');

function fixture() {
  const baseline = {schema: 'kaihand-unified-evaluation-workbench-v1', counts: {members: 595},
    tasks: [{task: 'poker-draw'}], historical_summary: {not_counted_again: true}, links: [], duplicates: [],
    groups: Array.from({length: 47}, (_, ordinal) => ({id: 'old-group-' + ordinal,
      model: ordinal ? 'pi05' : 'pi05+cpm', task: 'poker-draw', checkpoint: ordinal ? String(ordinal) : 'best',
      trials: Array.from({length: ordinal < 31 ? 13 : 12}, (_, seed) => ({id: `old-${ordinal}-${seed}`, seed,
        valid: true, staged_evidence_status: ordinal ? 'valid' : 'invalid'}))}))};
  const group = {id: 'current-cpm-identity', task: 'poker-draw', model: 'pi05+cpm', checkpoint: LABEL,
    checkpoint_sha256: CPM_SHA, prediction_horizon: 30, execute_steps: 5, initial_state_randomization: true,
    deployment_manifest_sha256: 'e'.repeat(64), trials: Array.from({length: 10}, (_, seed) => ({
      id: 'new-cpm-' + seed, seed, valid: true, publication_status: 'eligible', staged_evidence_status: 'valid',
      stages: [1, 2, 3].map(index => ({key: 'stage' + index, name: 'stage' + index, reached: index === 1}))}))};
  const catalog = structuredClone(baseline);
  catalog.groups.push(group);
  catalog.counts = {members: 605, eligible: 585, quarantined: 20};
  return {baseline, catalog};
}
test('605-member browser contract retains old CPM invalid stages and separates the current SHA checkpoint', () => {
  const {catalog, baseline} = fixture();
  const group = compareCatalogs(catalog, baseline);
  assert.equal(group.checkpoint, 'best@e5f859a57552');
  assert.equal(catalog.groups[0].checkpoint, 'best');
  assert.equal(catalog.groups[0].trials[0].staged_evidence_status, 'invalid');
});
for (const [name, mutate] of [
  ['old-stage promotion', catalog => { catalog.groups[0].trials[0].staged_evidence_status = 'valid'; }],
  ['old details mutation', catalog => { catalog.groups[0].trials[0].detail_url = 'other/report.html'; }],
  ['pooling new with old best', catalog => { catalog.groups.at(-1).checkpoint = 'best'; }],
  ['another best weight', catalog => { catalog.groups.at(-1).checkpoint_sha256 = '0'.repeat(64); }],
  ['missing requested seed', catalog => { catalog.groups.at(-1).trials.pop(); }],
  ['duplicate requested seed', catalog => { catalog.groups.at(-1).trials[9].seed = 0; }],
  ['identity reuse', catalog => { catalog.groups.at(-1).trials[0].id = catalog.groups[0].trials[0].id; }],
  ['different tactile horizon', catalog => { catalog.groups.at(-1).prediction_horizon = 50; }],
]) test('browser contract rejects ' + name, () => {
  const {catalog, baseline} = fixture(); mutate(catalog);
  assert.throws(() => compareCatalogs(catalog, baseline));
});
test('retained artifact inspection hashes old detail content, not just copied manifest rows', () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'cpm-browser-contract-'));
  const baseline = path.join(temporary, 'baseline'), snapshot = path.join(temporary, 'snapshot');
  try {
    for (const directory of [baseline, snapshot]) fs.mkdirSync(path.join(directory, 'old-trial'), {recursive: true});
    const bytes = Buffer.from('old cpm detail with invalid stages');
    const row = {path: 'old-trial/report.html', size: bytes.length,
      sha256: crypto.createHash('sha256').update(bytes).digest('hex')};
    for (const directory of [baseline, snapshot]) {
      fs.writeFileSync(path.join(directory, row.path), bytes);
      fs.writeFileSync(path.join(directory, 'artifact_manifest.json'), JSON.stringify({files: [row]}));
    }
    assert.equal(checkRetainedBytes(snapshot, baseline), 1);
    fs.writeFileSync(path.join(snapshot, row.path), 'forged historical cpm detail');
    assert.throws(() => checkRetainedBytes(snapshot, baseline));
  } finally { fs.rmSync(temporary, {recursive: true, force: true}); }
});
