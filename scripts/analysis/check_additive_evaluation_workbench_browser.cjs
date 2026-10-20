'use strict';

// Localhost-only, read-only checks of an already exported 620-member snapshot.
// Reuses the retained rich/reference contracts without changing older QA.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {identityFields, scoreProjection, scriptsFromHtml, loadContract, safeLocalUrl} =
  require('./check_rich_evaluation_workbench_browser.cjs');
const {checkReferenceContract, referenceNumbers} = require('./check_reference_evaluation_workbench_browser.cjs');

const TASK = 'bulb-screw-inplace';
const ANGLE = 'rotation_from_initial_deg';
const STEPS = ['15000', '18000', '21000', '24000', '24999'];
const STAGES = ['stable_inplace_grasp', 'threaded_and_seated', 'tightening_verified', 'released_stably'];
const finite = value => typeof value === 'number' && Number.isFinite(value);
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));

function localUrl(value, base) {
  const url = new URL(value, base);
  assert(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password);
  assert(['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname), 'Browser QA is restricted to localhost');
  return url;
}
function isDefaultFavicon404(message, location, origin) {
  try {
    const url = new URL(location?.url);
    return url.origin === origin && url.pathname === '/favicon.ico' && !url.search && !url.hash &&
      /^Failed to load resource: the server responded with a status of 404\b/.test(message);
  } catch { return false; }
}
function isCancelledVideoRange(request, knownVideoUrls) {
  return request.resource_type === 'media' && request.error_text === 'net::ERR_ABORTED' &&
    knownVideoUrls.has(request.url);
}
function snapshotDirectory(value) {
  const url = localUrl(value);
  const reports = path.resolve(__dirname, '../../reports');
  const relative = decodeURIComponent(new URL('./', url).pathname).replace(/^\/+/, '');
  assert(relative && !relative.split('/').includes('..'));
  const directory = path.resolve(reports, relative);
  assert(directory.startsWith(reports + path.sep), 'Snapshot must be inside local reports');
  assert(fs.statSync(directory).isDirectory());
  return directory;
}
function compareCatalogs(catalog, baseline) {
  assert.equal(catalog.schema, 'kaihand-unified-evaluation-workbench-v1');
  assert.deepEqual([catalog.counts.members, catalog.counts.eligible, catalog.counts.quarantined,
    catalog.groups.length, catalog.tasks.length, new Set(catalog.groups.map(group => group.model)).size],
  [620, 600, 20, 52, 11, 4]);
  assert.deepEqual([baseline.counts.members, baseline.groups.length], [595, 47]);
  const byId = new Map(catalog.groups.map(group => [group.id, group]));
  assert.equal(byId.size, 52, 'No duplicate groups');
  for (const group of baseline.groups) assert.deepEqual(byId.get(group.id), group,
    'Every field of the original 47 groups must remain unchanged: ' + group.id);
  const retained = baseline.groups.map(group => byId.get(group.id));
  assert.deepEqual(scoreProjection({groups: retained}), scoreProjection(baseline));
  const ids = new Set(baseline.groups.map(group => group.id));
  const added = catalog.groups.filter(group => !ids.has(group.id));
  assert.equal(added.length, 5);
  assert.deepEqual(added.map(group => String(group.checkpoint)).sort(), [...STEPS].sort());
  const previousTrials = new Set(baseline.groups.flatMap(group => group.trials.map(trial => trial.id)));
  const allIds = catalog.groups.flatMap(group => group.trials.map(trial => trial.id));
  assert.equal(allIds.length, 620); assert.equal(new Set(allIds).size, 620);
  for (const group of added) {
    assert.equal(group.task, TASK); assert.equal(group.model, 'pi05');
    assert.equal(group.prediction_horizon, 50); assert.equal(group.execute_steps, 50);
    assert.equal(group.initial_state_randomization, false); assert.equal(group.trial_scope, 'fixed_initial_state_repeats');
    assert.equal(group.trials.length, 5);
    assert.deepEqual(group.trials.map(trial => trial.seed).sort(), [0, 1, 2, 3, 4]);
    for (const trial of group.trials) {
      assert(!previousTrials.has(trial.id)); assert.equal(trial.valid, true);
      assert.equal(trial.publication_status, 'eligible'); assert.equal(trial.staged_evidence_status, 'valid');
      assert.equal(trial.stages.length, 4);
    }
  }
  return {retained, added};
}
function terminalAndReference(directory, trial, config, viz) {
  const raw = fs.readFileSync(path.join(directory, 'summary.json'));
  assert.equal(crypto.createHash('sha256').update(raw).digest('hex'), trial.summary_sha256);
  const summary = JSON.parse(raw);
  assert.equal(summary.task, TASK); assert.equal(summary.seed, trial.seed);
  const state = summary.evaluation?.state || {}, initial = summary.evaluation?.threaded_initialization || {};
  assert(finite(state.rotation_from_initial_turns) && finite(state.clockwise_turns) && finite(initial.initial_clockwise_turns));
  assert.equal(initial.position_randomization_applied, false);
  assert(Math.abs(state.clockwise_turns - state.rotation_from_initial_turns - initial.initial_clockwise_turns) < 1e-10);
  const value = state.rotation_from_initial_turns * 360;
  const cards = config.task_summary.filter(card => card.key === ANGLE);
  assert.equal(cards.length, 1); assert.equal(cards[0].unit, 'deg'); assert.equal(cards[0].value, value);
  assert.match(cards[0].source, /rotation_from_initial_turns$/);
  const refs = checkReferenceContract(config.goal_references, viz, TASK);
  const reference = refs.find(item => item.key === ANGLE);
  assert(reference, 'Every relative-angle curve must have its own authenticated goal');
  assert.equal(refs.filter(item => item.key === ANGLE).length, 1);
  assert.equal(reference.binding.fixed_shared_relative_target, false);
  assert.equal(reference.binding.initial_angle_source, 'evaluation.threaded_initialization.initial_clockwise_turns');
  assert.equal(reference.binding.initial_angle_turns, initial.initial_clockwise_turns);
  assert.equal(reference.binding.summary.sha256, trial.summary_sha256);
  assert(Math.abs(reference.target - (reference.binding.absolute_target_deg - initial.initial_clockwise_turns * 360)) < 1e-9);
  return {value, reference, refs};
}
async function mapLimit(rows, limit, callback) {
  let next = 0;
  await Promise.all(Array.from({length: Math.min(rows.length, limit)}, async () => {
    while (next < rows.length) { const index = next++; await callback(rows[index], index); }
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
    const bytes = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let purple = 0;
    for (let i = 0; i < bytes.length; i += 4)
      if (Math.abs(bytes[i] - 128) < 9 && Math.abs(bytes[i + 1] - 96) < 9 && Math.abs(bytes[i + 2] - 159) < 9) purple++;
    return purple;
  });
  assert(pixels > 20, 'The source-bound target must really be painted on the canvas');
}
async function overviewChecks(page, target, catalog) {
  assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
  await page.waitForSelector('.ks-case');
  const embedded = await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent));
  assert.deepEqual(embedded, catalog);
  assert.equal(await page.locator('.ks-case').count(), 620);
  assert.equal(await page.locator('#ks-summary-mode').inputValue(), 'policy');
  assert.equal(await page.locator('#ks-comparison tr').count(), 52);
  await page.locator(`#ks-task-tabs [data-task="${TASK}"]`).click();
  assert.equal(await page.locator('.ks-case:visible').count(), 25);
  const rows = await page.locator('#ks-comparison tr').evaluateAll(nodes => nodes.map(node => ({
    task: node.dataset.task, model: node.dataset.model, checkpoint: node.dataset.checkpoint,
    n: node.querySelector('.ks-summary-n')?.textContent.trim(), stages: node.querySelector('.ks-summary-stages')?.textContent,
  })));
  assert.equal(rows.length, 5);
  assert.deepEqual(rows.map(row => row.checkpoint).sort(), [...STEPS].sort());
  rows.forEach(row => { assert.equal(row.task, TASK); assert.equal(row.model, 'pi05'); assert.equal(row.n, '5 / 5'); });
  for (const step of STEPS) {
    await page.selectOption('#ks-checkpoint', step);
    assert.equal(await page.locator('.ks-case:visible').count(), 5);
    assert.equal(await page.locator('#ks-comparison tr').count(), 1);
    assert.equal(await page.locator('#ks-comparison tr').getAttribute('data-checkpoint'), step);
  }
  await page.selectOption('#ks-checkpoint', '');
  await page.locator('#ks-task-tabs [data-task=""]').click();
  assert.equal(await page.locator('#ks-comparison tr').count(), 52);
  assert.equal(await page.locator('.ks-case:visible').count(), 620);
}

async function main() {
  const [target, output, ...flags] = process.argv.slice(2);
  assert(target && output && flags.length === 1 && flags[0].startsWith('--baseline='),
    'usage: node check_additive_evaluation_workbench_browser.cjs URL QA_DIRECTORY --baseline=OLD_R5_URL');
  const requested = localUrl(target), baselineUrl = localUrl(flags[0].slice(11), target);
  assert.equal(requested.origin, baselineUrl.origin, 'Baseline must use the same localhost platform');
  const snapshot = snapshotDirectory(requested.href), baseline = snapshotDirectory(baselineUrl.href);
  assert.notEqual(snapshot, baseline);
  const catalog = json(path.join(snapshot, 'catalog.json')), original = json(path.join(baseline, 'catalog.json'));
  const {retained, added} = compareCatalogs(catalog, original);
  const allOld = retained.flatMap(group => group.trials.map(trial => ({group, trial})));
  const allNew = added.flatMap(group => group.trials.map(trial => ({group, trial})));
  const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
  const browser = await chromium.launch({headless: true,
    ...((process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE) ?
      {executablePath: process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server']});
  const errors = [], consoleErrors = [], ignoredDefaultFaviconErrors = [], failedRequests = [],
    cancelledVideoRangeRequests = [], checks = [];
  const knownVideoUrls = new Set(allNew.map(({trial}) => safeLocalUrl(trial.video_url, target, requested.origin)));
  let legacyHeads = 0, ranges = 0, referenceStrokes = 0;
  fs.mkdirSync(output, {recursive: true});
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
    page.setDefaultTimeout(30000);
    page.on('pageerror', error => errors.push({url: page.url(), message: String(error)}));
    page.on('console', message => {
      if (message.type() !== 'error') return;
      const detail = {url: page.url(), message: message.text(), location: message.location()};
      // Chromium asks for /favicon.ico even when the app declares no icon.
      // Record this existing root-level cosmetic 404, but never suppress a
      // script, stylesheet, plot, media, or other request's error.
      (isDefaultFavicon404(detail.message, detail.location, requested.origin) ?
        ignoredDefaultFaviconErrors : consoleErrors).push(detail);
    });
    page.on('requestfailed', request => {
      const detail = {url: request.url(), resource_type: request.resourceType(),
        error_text: request.failure()?.errorText, range: request.headers().range || null};
      // A seek/navigation can cancel the previous byte-range read of a known
      // trial movie. That exact cancellation is not a failed media decode;
      // every movie still must independently decode, play, and return 206.
      (isCancelledVideoRange(detail, knownVideoUrls) ? cancelledVideoRangeRequests : failedRequests).push(detail);
    });
    const origin = requested.origin;
    await overviewChecks(page, target, catalog);
    await page.screenshot({path: path.join(output, 'overview-620.png')});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Overview mobile overflow');
    await page.screenshot({path: path.join(output, 'overview-620-mobile.png')});
    await page.setViewportSize({width: 1480, height: 1000});
    await mapLimit(allOld, 6, async ({trial}) => {
      const response = await page.request.head(safeLocalUrl(trial.detail_url, target, origin));
      try { assert.equal(response.status(), 200, 'Retained detail missing: ' + trial.id); legacyHeads++; }
      finally { await response.dispose(); }
    });
    for (const {group, trial} of allNew) {
      const url = safeLocalUrl(trial.detail_url, target, origin);
      assert.equal((await page.goto(url, {waitUntil: 'domcontentloaded'})).status(), 200);
      await page.waitForSelector('.wb-root');
      await page.waitForSelector('.wb-trajectory-nav[data-status="ready"]');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      const {payload, loads, config} = scriptsFromHtml(await page.content()), viz = payload.visualization;
      for (const [key, expected] of Object.entries(identityFields(group, trial)))
        assert.deepEqual(config.unified_identity[key], expected, trial.id + ' identity.' + key);
      assert.equal(config.trial_scope.initial_state_randomization, false);
      assert.equal(config.trial_scope.kind, 'fixed_initial_state_repeats');
      loadContract(loads, trial.id);
      const spatial = viz.spatial_frames;
      assert(spatial && spatial.status !== 'missing');
      assert.equal(spatial.normal_taxel_force_n.length, spatial.time_s.length);
      for (const frame of [spatial.normal_taxel_force_n[0], spatial.normal_taxel_force_n.at(-1)]) {
        assert.equal(frame.length, 5);
        frame.forEach(finger => { assert.equal(finger.length, 7); finger.forEach(row => assert.equal(row.length, 5)); });
      }
      const directory = path.resolve(snapshot, path.dirname(trial.detail_url));
      assert(directory.startsWith(snapshot + path.sep));
      const terminal = terminalAndReference(directory, trial, config, viz);
      assert.deepEqual(Object.values(config.staged_evaluation.staged_result.stage_definitions), STAGES);
      assert.equal(await page.locator('.wb-stage-card').count(), 4);
      assert.equal(await page.locator('.wb-stage-badge').count(), 4);
      for (const stage of trial.stages)
        assert.equal(await page.locator(`.wb-stage-card[data-stage="${stage.key}"]`).getAttribute('data-reached'), String(stage.reached));
      assert.equal(await page.locator('.wb-chart').count(), 4);
      assert.equal(await page.locator('.wb-finger').count(), 5);
      await page.locator('.wb-finger').nth(2).click();
      await page.locator('.ke-mode-button').first().click();
      assert.equal(await page.locator('.ke-heat-card[aria-hidden="false"]').count(), 1);
      assert.equal(await page.locator('.ke-heatmap').count(), 5);
      assert.equal(await page.locator('.ke-task-chart-card').count(), viz.task_series.length);
      assert.equal(await page.locator('.wb-trajectory-select option').count(), 25);
      assert.match(await page.locator('.wb-horizon').innerText(), /H50.*K50/);
      assert.equal(await page.locator('.wb-goal-status').getAttribute('data-status'), 'verified');
      for (const reference of terminal.refs) {
        const card = page.locator(`[data-task-key="${reference.key}"]`);
        assert.equal(await card.count(), 1);
        await referenceStroke(card); referenceStrokes++;
      }
      const prominent = page.locator(`.wb-bulb-angle-summary[data-metric-key="${ANGLE}"]`);
      const final = prominent.locator('.wb-bulb-angle-terminal');
      assert.equal(await final.count(), 1); assert(await final.isVisible());
      const finalText = await final.innerText();
      assert(finalText.includes(terminal.value.toFixed(3))); assert.match(finalText, /终态|最终/);
      assert.equal(await page.locator(`.wb-task-final[data-metric-key="${ANGLE}"]`).count(), 1);
      const targetText = await prominent.locator('.wb-bulb-angle-target strong').innerText();
      assert(Math.abs(parseFloat(targetText) - terminal.reference.target) < .0006);
      const video = page.locator('.ke-media-panel .ke-video');
      await page.waitForFunction(() => {
        const video = document.querySelector('.ke-media-panel .ke-video');
        return !!video && !video.error && video.readyState >= 2 && video.videoWidth > 0 && video.duration > 0;
      });
      for (const camera of ['head', 'wrist']) {
        const button = page.locator(`.wb-camera[data-camera="${camera}"]`);
        assert.equal(await button.count(), 1, 'Both model input views must be usable');
        await button.click();
        assert.equal(await video.getAttribute('data-camera'), camera);
      }
      const before = await video.evaluate(element => element.currentTime);
      await video.evaluate(async element => { element.muted = true; await element.play(); });
      await page.waitForFunction(start => document.querySelector('.ke-media-panel .ke-video').currentTime > start + .1, before);
      await video.evaluate(element => element.pause());
      assert.equal(await video.evaluate(element => element.error), null, 'Recorded model video must decode and play');
      const mediaUrl = safeLocalUrl(viz.media.url, url, origin);
      const range = await page.request.get(mediaUrl, {headers: {Range: 'bytes=0-1023'}});
      try {
        assert.equal(range.status(), 206); assert.match(range.headers()['content-range'], /^bytes 0-1023\/\d+$/);
        const bytes = await range.body(); assert.equal(bytes.length, 1024); assert.equal(bytes.toString('ascii', 4, 8), 'ftyp');
        ranges++;
      } finally { await range.dispose(); }
      const series = viz.task_series.find(item => item.key === ANGLE);
      const observations = series.values.map((value, index) => ({value, index})).filter(item => finite(item.value));
      const samples = [observations.reduce((a, b) => a.value <= b.value ? a : b),
        observations.reduce((a, b) => a.value >= b.value ? a : b)];
      for (const sample of samples) {
        await page.locator('.ke-slider').evaluate((slider, time) => {
          slider.value = String(time); slider.dispatchEvent(new Event('input', {bubbles: true}));
        }, series.time_s[sample.index]);
        await page.waitForFunction(time => Math.abs(Number(document.querySelector('.ke-slider').value) - time) < 1e-7,
          series.time_s[sample.index]);
        const reading = await prominent.locator('.wb-bulb-angle-current strong').innerText();
        assert(Math.abs(parseFloat(reading.replace(/−/g, '-')) - sample.value) < .0006);
        assert.equal(await final.innerText(), finalText, 'Cursor may not overwrite raw terminal net angle');
      }
      await page.setViewportSize({width: 390, height: 844});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), trial.id + ' mobile overflow');
      if (trial.seed === 0) {
        await page.evaluate(() => scrollTo(0, 0));
        await page.screenshot({path: path.join(output, `inplace-${group.checkpoint}-mobile.png`)});
      }
      await page.setViewportSize({width: 1480, height: 1000});
      const back = await page.locator('.wb-overview-link').getAttribute('href');
      assert.equal(safeLocalUrl(back, url, origin), new URL(target).href);
      checks.push({trial_id: trial.id, group_id: group.id, checkpoint: group.checkpoint, seed: trial.seed,
        final_net_angle_deg: terminal.value, relative_target_deg: terminal.reference.target,
        relative_reference: referenceNumbers(terminal.reference), video_decoded_and_played: true});
      process.stdout.write(JSON.stringify({stage: 'new_dom', checked: checks.length, total: 25, trial_id: trial.id}) + '\n');
    }
    await Promise.all([page.waitForURL(target), page.locator('.wb-overview-link').click()]);
    await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('.ks-case').count(), 620);
    assert.deepEqual(errors, [], 'All pages must remain free of JavaScript errors');
    assert.deepEqual(consoleErrors, [], 'All pages must remain free of console/resource errors');
    assert.deepEqual(failedRequests, [], 'Only intentional ERR_ABORTED reads of known trial videos may be cancelled');
    const result = {status: 'passed', members: 620, eligible: 600, quarantined: 20, groups: 52, tasks: 11, models: 4,
      retained_group_dicts_unchanged: 47, retained_trial_scores_unchanged: 595, retained_detail_heads: legacyHeads,
      new_dom_trials: checks.length, new_video_decode_and_play_checks: checks.length, video_range_checks: ranges,
      reference_canvas_strokes: referenceStrokes, fixed_initial_state_scope_visible: true,
      independent_checkpoint_rows: 5, trials_per_checkpoint: 5, overview_return: true,
      mobile_overflow: false, model_weights_read: 0, simulation_runs_started: 0,
      snapshot_url: target, baseline_url: baselineUrl.href, browser_errors: errors,
      console_errors: consoleErrors, ignored_browser_default_favicon_errors: ignoredDefaultFaviconErrors,
      failed_requests: failedRequests, cancelled_video_range_requests: cancelledVideoRangeRequests,
      checked_trials: checks};
    fs.writeFileSync(path.join(output, 'additive_browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify({...result, checked_trials: undefined}) + '\n');
  } finally { await browser.close(); }
}

module.exports = {localUrl, compareCatalogs, terminalAndReference, isDefaultFavicon404, isCancelledVideoRange};
if (require.main === module) main().catch(error => {
  process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1;
});
