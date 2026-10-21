'use strict';

// Localhost-only, artifact-only QA of 595 retained + 10 new CPM members.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {identityFields, scriptsFromHtml, loadContract, safeLocalUrl} =
  require('./check_rich_evaluation_workbench_browser.cjs');
const {checkReferenceContract} = require('./check_reference_evaluation_workbench_browser.cjs');
const {localUrl, isDefaultFavicon404, isCancelledVideoRange} =
  require('./check_additive_evaluation_workbench_browser.cjs');

const CPM_SHA = 'e5f859a5755254e55e5fec660e8d11e8b5f27b7400cc4243ad48a133951186f3';
const LABEL = 'best@' + CPM_SHA.slice(0, 12);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function compareCatalogs(catalog, baseline) {
  assert.equal(catalog.schema, 'kaihand-unified-evaluation-workbench-v1');
  assert.deepEqual([catalog.counts.members, catalog.counts.eligible, catalog.counts.quarantined, catalog.groups.length],
    [605, 585, 20, 48]);
  assert.deepEqual([baseline.counts.members, baseline.groups.length], [595, 47]);
  const oldIds = new Set(baseline.groups.map(group => group.id));
  const byId = new Map(catalog.groups.map(group => [group.id, group]));
  assert.equal(byId.size, 48);
  baseline.groups.forEach(group => assert.deepEqual(byId.get(group.id), group, 'Retained group changed: ' + group.id));
  ['tasks', 'historical_summary', 'links', 'duplicates'].forEach(key => assert.deepEqual(catalog[key], baseline[key]));
  const additions = catalog.groups.filter(group => !oldIds.has(group.id));
  assert.equal(additions.length, 1);
  const group = additions[0];
  assert.deepEqual([group.task, group.model, group.checkpoint, group.checkpoint_sha256,
    group.prediction_horizon, group.execute_steps, group.initial_state_randomization],
  ['poker-draw', 'pi05+cpm', LABEL, CPM_SHA, 30, 5, true]);
  assert.match(group.deployment_manifest_sha256, /^[0-9a-f]{64}$/);
  assert.equal(group.trials.length, 10);
  assert.deepEqual(group.trials.map(trial => trial.seed).sort((a, b) => a - b), Array.from({length: 10}, (_, i) => i));
  const oldTrials = new Set(baseline.groups.flatMap(row => row.trials.map(trial => trial.id)));
  const all = catalog.groups.flatMap(row => row.trials.map(trial => trial.id));
  assert.equal(new Set(all).size, 605);
  group.trials.forEach(trial => {
    assert(!oldTrials.has(trial.id)); assert.equal(trial.valid, true);
    assert.equal(trial.publication_status, 'eligible'); assert.equal(trial.staged_evidence_status, 'valid');
    assert.equal(trial.stages.length, 3);
  });
  return group;
}
function snapshotDirectory(value) {
  const url = localUrl(value), reports = path.resolve(__dirname, '../../reports');
  const relative = decodeURIComponent(new URL('./', url).pathname).replace(/^\/+/, '');
  assert(relative && !relative.split('/').includes('..'));
  const directory = path.resolve(reports, relative);
  assert(directory.startsWith(reports + path.sep)); assert(fs.statSync(directory).isDirectory());
  return directory;
}
async function mapLimit(rows, limit, callback) {
  let next = 0;
  await Promise.all(Array.from({length: Math.min(rows.length, limit)}, async () => {
    while (next < rows.length) await callback(rows[next++]);
  }));
}
async function reveal(locator) {
  await locator.evaluate(node => {
    for (let parent = node.parentElement; parent; parent = parent.parentElement)
      if (parent.tagName === 'DETAILS') parent.open = true;
  });
  await locator.scrollIntoViewIfNeeded();
}
async function referenceStroke(card) {
  await reveal(card);
  await card.page().evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const pixels = await card.locator('canvas').evaluate(canvas => {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let purple = 0;
    for (let i = 0; i < data.length; i += 4)
      if (Math.abs(data[i] - 128) < 9 && Math.abs(data[i + 1] - 96) < 9 && Math.abs(data[i + 2] - 159) < 9) purple++;
    return purple;
  });
  assert(pixels > 20, 'Task reference must be painted, not only present in JSON');
}
function checkRetainedBytes(snapshot, baseline) {
  const replace = new Set(['catalog.json', 'index.html', 'verification.json', 'artifact_manifest.json',
    'previous_root_index.html', 'reference_coverage.json']);
  const current = new Map(json(path.join(snapshot, 'artifact_manifest.json')).files.map(row => [row.path, row]));
  let checked = 0;
  for (const row of json(path.join(baseline, 'artifact_manifest.json')).files) {
    if (replace.has(row.path)) continue;
    assert.deepEqual(current.get(row.path), row, 'Old inventoried bytes changed: ' + row.path);
    const file = path.resolve(snapshot, row.path);
    assert(file.startsWith(snapshot + path.sep));
    assert.equal(fs.statSync(file).size, row.size); assert.equal(hash(file), row.sha256);
    checked++;
  }
  return checked;
}
async function main() {
  const [target, output, baselineFlag, ...extra] = process.argv.slice(2);
  assert(target && output && baselineFlag?.startsWith('--baseline=') && !extra.length,
    'usage: node check_cpm_additive_evaluation_workbench_browser.cjs URL SNAPSHOT/browser_qa --baseline=R5_URL');
  const requested = localUrl(target), baselineUrl = localUrl(baselineFlag.slice(11), target);
  assert.equal(requested.origin, baselineUrl.origin);
  const snapshot = snapshotDirectory(target), baseline = snapshotDirectory(baselineUrl.href);
  assert.equal(path.resolve(output), path.join(snapshot, 'browser_qa'), 'Receipt must be inside this snapshot/browser_qa');
  assert.notEqual(snapshot, baseline);
  const catalog = json(path.join(snapshot, 'catalog.json')), original = json(path.join(baseline, 'catalog.json'));
  const group = compareCatalogs(catalog, original), preserved = checkRetainedBytes(snapshot, baseline);
  const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
  const browser = await chromium.launch({headless: true,
    ...((process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE) ?
      {executablePath: process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server']});
  fs.mkdirSync(output, {recursive: true});
  const errors = [], consoleErrors = [], ignoredDefaultFaviconErrors = [], failedRequests = [], cancelled = [], checks = [];
  const knownVideoUrls = new Set(group.trials.map(trial => safeLocalUrl(trial.video_url, target, requested.origin)));
  let retainedHeads = 0, ranges = 0, referenceStrokes = 0;
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
    page.setDefaultTimeout(30000);
    page.on('pageerror', error => errors.push({url: page.url(), error: String(error)}));
    page.on('console', message => {
      if (message.type() !== 'error') return;
      const detail = {url: page.url(), message: message.text(), location: message.location()};
      (isDefaultFavicon404(detail.message, detail.location, requested.origin) ? ignoredDefaultFaviconErrors : consoleErrors).push(detail);
    });
    page.on('requestfailed', request => {
      const detail = {url: request.url(), resource_type: request.resourceType(), error_text: request.failure()?.errorText};
      (isCancelledVideoRange(detail, knownVideoUrls) ? cancelled : failedRequests).push(detail);
    });
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.ks-case');
    assert.deepEqual(await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent)), catalog);
    assert.equal(await page.locator('.ks-case').count(), 605);
    assert.equal(await page.locator('#ks-summary-mode').inputValue(), 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 48);
    await page.selectOption('#ks-model', 'pi05+cpm');
    assert.equal(await page.locator('#ks-comparison tr').count(), 2, 'Old and current CPM must remain separate rows');
    assert.equal(await page.locator('.ks-case:visible').count(), 15);
    await page.selectOption('#ks-checkpoint', LABEL);
    assert.equal(await page.locator('.ks-case:visible').count(), 10);
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    assert.equal(await page.locator('#ks-comparison tr').getAttribute('data-checkpoint'), LABEL);
    assert.equal((await page.locator('.ks-summary-n').innerText()).trim(), '10 / 10');
    await page.selectOption('#ks-checkpoint', 'best');
    assert.equal(await page.locator('.ks-case:visible').count(), 5);
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    await page.selectOption('#ks-checkpoint', ''); await page.selectOption('#ks-model', '');
    await page.screenshot({path: path.join(output, 'overview-605.png')});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.screenshot({path: path.join(output, 'overview-605-mobile.png')});
    await page.setViewportSize({width: 1480, height: 1000});
    await mapLimit(original.groups.flatMap(row => row.trials), 6, async trial => {
      const response = await page.request.head(safeLocalUrl(trial.detail_url, target, requested.origin));
      try { assert.equal(response.status(), 200, 'Retained detail missing: ' + trial.id); retainedHeads++; }
      finally { await response.dispose(); }
    });
    for (const trial of group.trials) {
      const detailUrl = safeLocalUrl(trial.detail_url, target, requested.origin);
      assert.equal((await page.goto(detailUrl, {waitUntil: 'domcontentloaded'})).status(), 200);
      await page.waitForSelector('.wb-root'); await page.waitForSelector('.wb-trajectory-nav[data-status="ready"]');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      const {payload, loads, config} = scriptsFromHtml(await page.content()), visual = payload.visualization;
      Object.entries(identityFields(group, trial)).forEach(([key, value]) => assert.deepEqual(config.unified_identity[key], value));
      assert.equal(config.trial_scope.initial_state_randomization, true);
      loadContract(loads, trial.id);
      assert.equal(config.diagnostic_plots.status, 'available');
      assert.equal(config.diagnostic_plots.comparison_kind, 'model_only');
      assert(visual.spatial_frames && visual.spatial_frames.status !== 'missing');
      const references = checkReferenceContract(config.goal_references, visual, 'poker-draw');
      assert.equal(config.goal_references.status, 'verified');
      assert.equal(await page.locator('.wb-goal-status[data-status="verified"]').count(), 1);
      for (const reference of references) {
        const card = page.locator(`[data-task-key="${reference.key}"]`);
        assert.equal(await card.count(), 1); await referenceStroke(card); referenceStrokes++;
      }
      assert.equal(await page.locator('.wb-stage-card').count(), 3);
      for (const stage of trial.stages)
        assert.equal(await page.locator(`.wb-stage-card[data-stage="${stage.key}"]`).getAttribute('data-reached'), String(stage.reached));
      assert.equal(await page.locator('.wb-chart').count(), 4);
      const video = page.locator('.ke-media-panel .ke-video');
      await page.waitForFunction(() => {
        const video = document.querySelector('.ke-media-panel .ke-video');
        return !!video && !video.error && video.readyState >= 2 && video.videoWidth === 1920 && video.videoHeight === 1080 && video.duration > 90;
      });
      assert.equal(config.media_views.status, 'verified'); assert.equal(config.media_views.sha256, visual.media.sha256);
      // Use the exact authenticated view IDs. H30 receives the same reviewed
      // spatial renderer contract, never a guessed crop or a H50 identity.
      for (const camera of ['head', 'wrist', 'global']) {
        assert(config.media_views.views.some(view => view.id === camera));
        const button = page.locator(`.wb-camera[data-camera="${camera}"]`);
        assert.equal(await button.count(), 1, 'Authenticated camera must be usable: ' + camera);
        await button.click(); assert.equal(await video.getAttribute('data-camera'), camera);
        assert.equal(await button.getAttribute('aria-pressed'), 'true');
        assert(await video.evaluate(node => node.style.width.endsWith('%') && node.style.height.endsWith('%')));
      }
      await page.locator('.wb-camera[data-camera="head"]').click();
      const start = await video.evaluate(node => node.currentTime);
      await video.evaluate(async node => { node.muted = true; await node.play(); });
      await page.waitForFunction(time => document.querySelector('.ke-media-panel .ke-video').currentTime > time + .1, start);
      await video.evaluate(node => node.pause()); assert.equal(await video.evaluate(node => node.error), null);
      const mediaUrl = safeLocalUrl(visual.media.url, detailUrl, requested.origin);
      const response = await page.request.get(mediaUrl, {headers: {Range: 'bytes=0-1023'}});
      try { assert.equal(response.status(), 206); assert.match(response.headers()['content-range'], /^bytes 0-1023\/\d+$/);
        const bytes = await response.body(); assert.equal(bytes.length, 1024); assert.equal(bytes.toString('ascii', 4, 8), 'ftyp'); ranges++; }
      finally { await response.dispose(); }
      const back = await page.locator('.wb-overview-link').getAttribute('href');
      assert.equal(safeLocalUrl(back, detailUrl, requested.origin), new URL(target).href);
      await page.setViewportSize({width: 390, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), trial.id + ' mobile overflow');
      if (trial.seed === 0) { await page.evaluate(() => scrollTo(0, 0)); await page.screenshot({path: path.join(output, 'cpm-seed0-mobile.png')}); }
      await page.setViewportSize({width: 1480, height: 1000});
      checks.push({trial_id: trial.id, seed: trial.seed, native_success: trial.native_success,
        stage_flags: Object.fromEntries(trial.stages.map(stage => [stage.key, stage.reached])), camera_views: ['head', 'wrist', 'global'],
        video_decoded_and_played: true, reference_count: references.length});
      process.stdout.write(JSON.stringify({stage: 'new_cpm_dom', checked: checks.length, total: 10, trial_id: trial.id}) + '\n');
    }
    await Promise.all([page.waitForURL(target), page.locator('.wb-overview-link').click()]);
    await page.waitForSelector('.ks-case'); assert.equal(await page.locator('.ks-case').count(), 605);
    assert.deepEqual(errors, []); assert.deepEqual(consoleErrors, []); assert.deepEqual(failedRequests, []);
    const result = {status: 'passed', members: 605, eligible: 585, quarantined: 20, groups: 48,
      retained_group_dicts_unchanged: 47, retained_members_unchanged: 595, retained_payload_files_hashed: preserved,
      retained_detail_heads: retainedHeads, new_dom_trials: checks.length, new_video_decode_and_play_checks: checks.length,
      video_range_checks: ranges, reference_canvas_strokes: referenceStrokes, old_and_new_best_separate: true,
      snapshot_directory: snapshot, catalog_sha256: hash(path.join(snapshot, 'catalog.json')), snapshot_url: target,
      baseline_url: baselineUrl.href, model_weights_read: 0, simulation_runs_started: 0,
      browser_errors: errors, console_errors: consoleErrors, ignored_browser_default_favicon_errors: ignoredDefaultFaviconErrors,
      failed_requests: failedRequests, cancelled_video_range_requests: cancelled, checked_trials: checks};
    fs.writeFileSync(path.join(output, 'cpm_additive_browser_verification.json'), JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
    process.stdout.write(JSON.stringify({...result, checked_trials: undefined}) + '\n');
  } finally { await browser.close(); }
}

module.exports = {compareCatalogs, checkRetainedBytes, CPM_SHA, LABEL};
if (require.main === module) main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
