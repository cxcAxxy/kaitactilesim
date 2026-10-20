'use strict';

// Offline artifact comparisons plus local-browser regression checks. No model,
// simulator, checkpoint, or evaluator is imported or started by this script.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {scoreProjection, scriptsFromHtml} = require('./check_rich_evaluation_workbench_browser.cjs');
const finite = value => typeof value === 'number' && Number.isFinite(value);
const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const angleKeys = {clockwise_turns: 'clockwise_angle_deg', turns_removed: 'angle_removed_deg',
  rotation_from_initial_turns: 'rotation_from_initial_deg'};

function normalizedReference(reference) {
  const result = {...reference};
  if (angleKeys[result.key] && result.unit === 'turn') {
    result.key = angleKeys[result.key]; result.unit = 'deg';
    for (const key of ['target', 'lower', 'upper']) if (finite(result[key])) result[key] *= 360;
  }
  return result;
}
function referenceNumbers(reference) {
  const normalized = normalizedReference(reference);
  return Object.fromEntries(['key', 'unit', 'target', 'relation', 'lower', 'upper', 'reference_kind']
    .filter(key => Object.hasOwn(normalized, key)).map(key => [key, normalized[key]]));
}
function referencesBySeries(visualization) {
  return visualization.task_series.flatMap(series => (series.goal_references || []).map(reference => {
    const normalized = normalizedReference(reference);
    assert.equal(normalized.key, angleKeys[series.key] || series.key, 'Reference must bind to the plotted quantity');
    assert.equal(normalized.unit, angleKeys[series.key] && series.unit === 'turn' ? 'deg' : series.unit);
    return normalized;
  }));
}
function checkReferenceContract(goals, visualization, task) {
  assert(goals && typeof goals === 'object', 'Every trial needs an explicit reference availability state');
  assert.equal(goals.task, task);
  assert(['verified', 'unavailable', 'not_applicable'].includes(goals.status), 'Reference availability must be explicit');
  assert(Array.isArray(goals.references));
  const plotted = referencesBySeries(visualization);
  if (goals.status !== 'verified') {
    assert.equal(goals.references.length, 0, 'Unverified references cannot expose numerical targets');
    assert.equal(plotted.length, 0, 'Unverified references cannot remain on a canvas');
    if (goals.status === 'unavailable') assert(goals.warnings?.length, 'Unavailable references need a reason');
    return [];
  }
  assert(goals.references.length > 0);
  assert.equal(goals.binding.source_code_executed, false);
  assert.equal(goals.binding.rules_reexecuted, false);
  assert.match(goals.binding.evaluation_source_sha256 || goals.binding.deployment_manifest_sha256, /^[0-9a-f]{64}$/);
  assert(goals.provenance.some(row => row.role === 'goal_reference_reviewed_source'));
  for (const reference of goals.references) {
    const normalized = normalizedReference(reference);
    assert(finite(normalized.target));
    assert(['target', 'at_least', 'at_most'].includes(normalized.relation));
    if (Object.hasOwn(normalized, 'lower') || Object.hasOwn(normalized, 'upper')) {
      assert(finite(normalized.lower) && finite(normalized.upper));
      assert(normalized.lower <= normalized.target && normalized.target <= normalized.upper);
    }
    const attached = plotted.filter(item => item.key === normalized.key);
    assert.equal(attached.length, 1, 'A verified reference must be attached to exactly one matching recorded curve');
    assert.deepEqual(referenceNumbers(attached[0]), referenceNumbers(reference));
    if (task === 'poker-draw' || normalized.key === 'angle_removed_deg') {
      assert(!Object.hasOwn(normalized, 'lower') && !Object.hasOwn(normalized, 'upper'),
        'Never fabricate a finite success band for an unbounded/native nominal reference');
    }
  }
  assert.equal(plotted.length, goals.references.length, 'No orphan or fabricated curve references');
  for (const series of visualization.task_series) if (/contact.*area|torque/.test(series.key))
    assert.equal(series.goal_references?.length || 0, 0, 'No invented contact-area or torque success target');
  return plotted;
}
function rawTerminalDegrees(report, trial, task) {
  const summaryFile = path.join(report.trial.source, 'summary.json');
  const bytes = fs.readFileSync(summaryFile);
  const digest = crypto.createHash('sha256').update(bytes).digest('hex');
  assert.equal(digest, trial.summary_sha256, 'Terminal display source must be the original published trial summary');
  const summary = JSON.parse(bytes.toString('utf8'));
  assert.equal(summary.task, task); assert.equal(summary.seed, trial.seed);
  const state = summary.evaluation?.state || {};
  const rawKey = task === 'bulb-unscrew' ? 'turns_removed' : 'clockwise_turns';
  assert(finite(state[rawKey]), 'An absent raw terminal angle must not be fabricated from the plotted endpoint');
  return {key: angleKeys[rawKey], value: state[rawKey] * 360, source: summaryFile,
    source_key: rawKey, sha256: digest};
}
function snapshotDirectory(url) {
  const reports = path.resolve(__dirname, '../../reports');
  const relative = decodeURIComponent(new URL('./', url).pathname).replace(/^\/+/, '');
  assert(relative && !relative.split('/').includes('..'));
  const directory = path.resolve(reports, relative);
  assert(directory.startsWith(reports + path.sep));
  assert(fs.statSync(directory).isDirectory());
  return directory;
}
function detailDirectory(snapshot, trial) {
  const directory = path.resolve(snapshot, path.dirname(trial.detail_url));
  assert(directory.startsWith(snapshot + path.sep));
  return directory;
}
function normalizedSeries(series) {
  if (!angleKeys[series.key] || series.unit !== 'turn') return series;
  return {...series, key: angleKeys[series.key], unit: 'deg', values: series.values.map(value => finite(value) ? value * 360 : value)};
}

function checkRetainedArtifactIdentities(snapshot, baseline) {
  const original = readJson(path.join(baseline, 'artifact_manifest.json'));
  const current = new Map(readJson(path.join(snapshot, 'artifact_manifest.json')).files.map(row => [row.path, row]));
  const changedRoot = new Set(['catalog.json', 'index.html', 'previous_root_index.html', 'verification.json',
    'artifact_manifest.json', 'assets/workbench.js', 'assets/workbench.css']);
  const changedDetail = new Set(['report.json', 'visualization.json', 'workbench.json', 'report.html', 'audit.html', 'goal_references.json']);
  let retained = 0;
  for (const row of original.files) {
    if (changedRoot.has(row.path) || changedDetail.has(path.basename(row.path))) continue;
    const after = current.get(row.path);
    assert(after, 'Reference snapshot must retain existing video, plots, tactile samples and sidecars: ' + row.path);
    assert.deepEqual([after.sha256, after.size], [row.sha256, row.size], 'Non-reference artifact bytes must remain identical: ' + row.path);
    retained += 1;
  }
  return retained;
}

async function main() {
  const [target, output, ...flags] = process.argv.slice(2);
  assert(target && output, 'usage: node check_reference_evaluation_workbench_browser.cjs URL QA_DIRECTORY [--baseline=URL]');
  assert(flags.every(flag => flag.startsWith('--baseline=')));
  const requested = new URL(target);
  assert(['127.0.0.1', 'localhost', '[::1]'].includes(requested.hostname), 'Browser QA is restricted to localhost');
  const baselineUrl = new URL(flags.find(flag => flag.startsWith('--baseline='))?.slice(11) ||
    '../kaitactilesim_unified_workbench_20261020_r4/index.html', target).href;
  const snapshot = snapshotDirectory(target), baseline = snapshotDirectory(baselineUrl);
  const retainedArtifacts = checkRetainedArtifactIdentities(snapshot, baseline);
  const catalog = readJson(path.join(snapshot, 'catalog.json'));
  const original = readJson(path.join(baseline, 'catalog.json'));
  assert.deepEqual(scoreProjection(catalog), scoreProjection(original), 'All 595 native/staged admissions must be preserved');
  assert.deepEqual([catalog.counts.members, catalog.counts.eligible, catalog.counts.quarantined, catalog.groups.length], [595, 575, 20, 47]);
  const previous = new Map(original.groups.flatMap(group => group.trials.map(trial => [trial.id, {group, trial}])));
  const contracts = [], statuses = {}, hashes = new Map();
  let oldReferencesPreserved = 0, newReferenceTrials = 0, bulbTerminalTrials = 0;
  for (const group of catalog.groups) for (const trial of group.trials) {
    const old = previous.get(trial.id), directory = detailDirectory(snapshot, trial);
    const config = readJson(path.join(directory, 'workbench.json'));
    const viz = readJson(path.join(directory, 'visualization.json'));
    const report = readJson(path.join(directory, 'report.json'));
    const oldDirectory = detailDirectory(baseline, old.trial);
    const oldConfig = readJson(path.join(oldDirectory, 'workbench.json'));
    const oldViz = readJson(path.join(oldDirectory, 'visualization.json'));
    assert.deepEqual(config.unified_identity, oldConfig.unified_identity, 'Reference annotations cannot change admission or H/K identity');
    assert.deepEqual(report.outcome, readJson(path.join(oldDirectory, 'report.json')).outcome);
    const refs = checkReferenceContract(config.goal_references, viz, group.task);
    statuses[config.goal_references.status] = (statuses[config.goal_references.status] || 0) + 1;
    if (oldConfig.goal_references?.status === 'verified') {
      assert.equal(config.goal_references.status, 'verified');
      assert.deepEqual(refs.map(referenceNumbers), referencesBySeries(oldViz).map(referenceNumbers),
        trial.id + ' cannot lose/change a previously verified reference');
      oldReferencesPreserved += 1;
    } else if (group.task !== 'pick-place') newReferenceTrials += 1;
    if (group.task === 'pick-place') assert.equal(config.goal_references.status, 'not_applicable', 'No object reference is inferred for EgoTouch');
    const originalSeries = oldViz.task_series.map(normalizedSeries);
    assert.equal(viz.task_series.length, originalSeries.length);
    viz.task_series.map(normalizedSeries).forEach((series, index) => {
      const prior = originalSeries[index];
      assert.equal(series.key, prior.key); assert.equal(series.unit, prior.unit);
      assert.deepEqual(series.time_s, prior.time_s, 'Recorded task clocks must not change');
      assert.deepEqual(series.values, prior.values, 'Recorded task values must not be rescored');
    });
    let terminal = null;
    if (['bulb-screw', 'bulb-unscrew'].includes(group.task)) {
      terminal = rawTerminalDegrees(report, trial, group.task);
      const cards = config.task_summary.filter(row => row.key === terminal.key);
      assert.equal(cards.length, 1, 'Exactly one final net-angle card is required for each bulb trial');
      assert.equal(cards[0].unit, 'deg'); assert.equal(cards[0].value, terminal.value);
      assert.match(cards[0].label, /终态|最终/); assert.match(cards[0].source, new RegExp(terminal.source_key + '$'));
      bulbTerminalTrials += 1;
    }
    const sourcePaths = config.goal_references.status === 'verified' && !oldConfig.goal_references ? config.goal_references.provenance : [];
    for (const row of sourcePaths) {
      const key = `${row.path}:${row.sha256}`;
      if (!hashes.has(key)) {
        const bytes = fs.readFileSync(row.path);
        assert.equal(bytes.length, row.size); assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), row.sha256);
        hashes.set(key, true);
      }
    }
    contracts.push({group, trial, config, refs, terminal, series: viz.task_series.map(normalizedSeries), directory});
  }
  assert.equal(oldReferencesPreserved, 360); assert.equal(newReferenceTrials, 115); assert.equal(bulbTerminalTrials, 100);
  process.stdout.write(JSON.stringify({stage: 'reference_payloads', checked: contracts.length, statuses, oldReferencesPreserved,
    newReferenceTrials, bulbTerminalTrials, new_source_files_hashed: hashes.size}) + '\n');

  const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [], rendered = [], screenshots = new Set();
  let renderedReferences = 0, referenceStrokes = 0, terminalCursorChecks = 0;
  fs.mkdirSync(output, {recursive: true});
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
    page.setDefaultTimeout(30000); page.on('pageerror', error => errors.push({url: page.url(), message: String(error)}));
    const selected = contracts.filter(item => item.trial.id === item.group.trials[0].id);
    // Ensure at least one nontrivial current/terminal distinction is exercised,
    // even if the first seed of every bulb group happens to remain at zero.
    for (const task of ['bulb-screw', 'bulb-unscrew']) {
      const varied = contracts.filter(item => item.group.task === task).find(item => {
        const values = item.series.find(series => series.key === item.terminal.key).values.filter(finite);
        return Math.max(...values) - Math.min(...values) > .001;
      });
      if (varied && !selected.includes(varied)) selected.push(varied);
    }
    for (const item of selected) {
      const {group, trial, config, refs, terminal, series} = item;
      const url = new URL(trial.detail_url, target).href;
      assert.equal((await page.goto(url, {waitUntil: 'domcontentloaded'})).status(), 200);
      await page.waitForSelector('.wb-root'); assert.equal(await page.locator('.wb-start-error').count(), 0);
      const fromHtml = scriptsFromHtml(await page.content());
      assert.deepEqual(fromHtml.config.goal_references, config.goal_references, 'HTML must embed the exact verified reference artifact');
      assert.equal(await page.locator('.ke-goal-reference').count(), refs.length);
      assert.equal(await page.locator('.wb-chart').count(), 4);
      assert.equal(await page.locator('.wb-finger').count(), 5);
      const status = page.locator('.wb-goal-status');
      assert.equal(await status.count(), 1, 'Visible reference availability must be present for every trajectory');
      assert.equal(await status.getAttribute('data-status'), config.goal_references.status);
      assert.equal(Number(await status.getAttribute('data-reference-count')), refs.length);
      for (const reference of refs) {
        const card = page.locator(`[data-task-key="${reference.key}"]`);
        assert.equal(await card.count(), 1);
        await card.evaluate(node => { for (let parent = node.parentElement; parent; parent = parent.parentElement) if (parent.tagName === 'DETAILS') parent.open = true; });
        assert(await card.isVisible(), 'Reference-bearing task card must remain discoverable after opening its supplementary fold');
        await card.scrollIntoViewIfNeeded();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.match(await card.innerText(), /不代表完整任务成功/);
        assert.equal(await card.locator('.ke-goal-reference-band').count(), finite(reference.lower) && finite(reference.upper) ? 1 : 0);
        const purple = await card.locator('canvas').evaluate(canvas => {
          const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
          let count = 0;
          for (let i = 0; i < pixels.length; i += 4) if (Math.abs(pixels[i] - 128) < 9 && Math.abs(pixels[i + 1] - 96) < 9 && Math.abs(pixels[i + 2] - 159) < 9) count++;
          return count;
        });
        assert(purple > 20, trial.id + '/' + reference.key + ': reference stroke must actually be drawn');
        renderedReferences += 1; referenceStrokes += 1;
      }
      if (terminal) {
        const angle = series.find(row => row.key === terminal.key);
        const card = page.locator(`[data-task-key="${terminal.key}"]`);
        const final = page.locator(`.wb-bulb-angle-summary[data-metric-key="${terminal.key}"] .wb-bulb-angle-terminal`);
        assert.equal(await page.locator(`.wb-task-final[data-metric-key="${terminal.key}"]`).count(), 1);
        assert.equal(await final.count(), 1); assert(await final.isVisible(), 'Terminal angle must be directly discoverable');
        const finalText = await final.innerText();
        assert(finalText.includes(terminal.value.toFixed(3))); assert.match(finalText, /终态|最终/); assert.match(finalText, /°/);
        const targetText = await page.locator(`.wb-bulb-angle-summary[data-metric-key="${terminal.key}"] .wb-bulb-angle-target strong`).innerText();
        const reference = refs.find(row => row.key === terminal.key);
        if (reference) assert(Math.abs(parseFloat(targetText) - reference.target) < 1e-8, 'Prominent target must match the per-trial reference, not the terminal angle');
        else assert.match(targetText, /未设|未核验|不可用/);
        const observed = angle.values.map((value, index) => ({value, index})).filter(row => finite(row.value));
        const minimum = observed.reduce((a, b) => a.value <= b.value ? a : b);
        const maximum = observed.reduce((a, b) => a.value >= b.value ? a : b);
        const readings = [];
        for (const sample of [minimum, maximum]) {
          await page.locator('.ke-slider').evaluate((slider, time) => {
            slider.value = String(time); slider.dispatchEvent(new Event('input', {bubbles: true}));
          }, angle.time_s[sample.index]);
          await page.waitForFunction(time => Math.abs(Number(document.querySelector('.ke-slider').value) - time) < 1e-7, angle.time_s[sample.index]);
          const text = await card.locator('.ke-task-chart-heading > strong').innerText();
          assert.match(text, /°/);
          assert(Math.abs(parseFloat(text.replace(/−/g, '-')) - sample.value) < .0006, 'Current angle reading must follow the native recorded cursor');
          const prominent = await page.locator(`.wb-bulb-angle-summary[data-metric-key="${terminal.key}"] .wb-bulb-angle-current strong`).innerText();
          assert(Math.abs(parseFloat(prominent.replace(/−/g, '-')) - sample.value) < .0006, 'Prominent current-angle card must share the recorded cursor');
          assert.equal(await final.innerText(), finalText, 'Terminal angle is a fixed raw-summary fact, not the moving cursor');
          readings.push(text);
        }
        if (maximum.value - minimum.value > .001) assert.notEqual(readings[0], readings[1]);
        terminalCursorChecks += 1;
      }
      if (trial.staged_evidence_status === 'invalid' || !trial.valid) {
        assert.equal(await page.locator('.wb-stage-badge[data-reached="true"], .wb-stage-card[data-reached="true"]').count(), 0);
        assert.equal(await page.locator('.wb-stage-first-time:not(:disabled)').count(), 0);
      }
      const kind = group.model === 'pi05' ? group.task : group.model;
      if (!screenshots.has(kind)) {
        screenshots.add(kind); await page.evaluate(() => scrollTo(0, 0));
        await page.screenshot({path: path.join(output, kind.replace(/[^a-z0-9_-]/gi, '_') + '-references.png')});
        await page.setViewportSize({width: 390, height: 844});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No mobile overflow from goal labels or terminal cards');
        await page.screenshot({path: path.join(output, kind.replace(/[^a-z0-9_-]/gi, '_') + '-references-mobile.png')});
        await page.setViewportSize({width: 1480, height: 1000});
      }
      rendered.push({trial_id: trial.id, group_id: group.id, task: group.task, status: config.goal_references.status, references: refs.length});
      if (rendered.length % 10 === 0) process.stdout.write(JSON.stringify({stage: 'reference_dom', checked: rendered.length, total: selected.length}) + '\n');
    }
    assert.equal(new Set(rendered.map(row => row.group_id)).size, 47);
    assert.deepEqual(errors, []);
    const result = {status: 'passed', members: 595, eligible: 575, quarantined: 20, groups: 47,
      scores_and_admissions_unchanged: true, recorded_task_samples_unchanged: true,
      retained_non_reference_artifact_sha_identities: retainedArtifacts,
      inherited_verified_reference_trials_preserved: oldReferencesPreserved,
      newly_bound_or_explicit_unavailable_trials: newReferenceTrials, reference_status_counts: statuses,
      new_source_files_hashed: hashes.size, terminal_angles_compared_with_original_raw_summary: bulbTerminalTrials,
      rendered_trials: rendered.length, rendered_groups: 47, rendered_references: renderedReferences,
      reference_canvas_strokes_checked: referenceStrokes, terminal_current_cursor_checks: terminalCursorChecks,
      no_fake_egotouch_reference: true, no_fabricated_torque_or_contact_area_targets: true,
      invalid_stage_and_quarantine_not_promoted: true, no_mobile_overflow: true,
      model_weights_read: 0, simulation_runs_started: 0, baseline_url: baselineUrl, snapshot_url: target,
      browser_errors: errors, dom_trials: rendered};
    fs.writeFileSync(path.join(output, 'reference_browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify({...result, dom_trials: undefined}) + '\n');
  } finally { await browser.close(); }
}

module.exports = {normalizedReference, referenceNumbers, referencesBySeries, checkReferenceContract, rawTerminalDegrees};
if (require.main === module) main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
