'use strict';

// Read-only, localhost-only QA of published artifacts; never starts a model.
// By default all 595 detail payloads are checked and every policy group is
// rendered. --all-dom additionally renders every retained trajectory.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const expectedSchema = 'kaihand-unified-evaluation-workbench-v1';
const scriptIds = ['kai-evaluation-data', 'kai-tactile-load-data', 'kai-workbench-config'];
const finite = value => typeof value === 'number' && Number.isFinite(value);
function sameJson(actual, expected, label) { assert.deepEqual(actual, expected, label); }
function identityFields(group, trial) {
  return {group_id: group.id, trial_id: trial.id, task: group.task, model: group.model,
    checkpoint: String(group.checkpoint), seed: trial.seed,
    prediction_horizon: group.prediction_horizon, execute_steps: group.execute_steps,
    valid: trial.valid, native_success: trial.native_success,
    publication_status: trial.publication_status};
}
function scoreProjection(catalog) {
  return catalog.groups.map(group => ({id: group.id, task: group.task, model: group.model,
    checkpoint: group.checkpoint, checkpoint_sha256: group.checkpoint_sha256,
    prediction_horizon: group.prediction_horizon, execute_steps: group.execute_steps,
    mode: group.mode, protocol: group.protocol, rule_identity: group.rule_identity,
    trials: group.trials.map(trial => ({id: trial.id, seed: trial.seed, valid: trial.valid,
      native_success: trial.native_success, publication_status: trial.publication_status,
      staged_evidence_status: trial.staged_evidence_status, stages: trial.stages,
      complete_stage_success: trial.complete_stage_success,
      continuous_metrics: trial.continuous_metrics, summary_sha256: trial.summary_sha256}))}));
}
function scriptsFromHtml(html) {
  const result = new Map();
  for (const match of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)) {
    const id = match[1].match(/\bid\s*=\s*["']([^"']+)["']/i)?.[1];
    if (!scriptIds.includes(id)) continue;
    assert(!result.has(id), 'Duplicate rich detail payload: ' + id);
    result.set(id, JSON.parse(match[2]));
  }
  scriptIds.forEach(id => assert(result.has(id), 'Missing rich detail payload: ' + id));
  return {payload: result.get(scriptIds[0]), loads: result.get(scriptIds[1]), config: result.get(scriptIds[2])};
}
function clock(times, label) {
  assert(Array.isArray(times) && times.length, label + ' must have recorded samples');
  times.forEach((time, index) => {
    assert(finite(time), label + ' contains a nonfinite clock');
    if (index) assert(time > times[index - 1], label + ' must preserve increasing native timestamps');
  });
}
function loadContract(loads, label) {
  clock(loads.time_s, label + ' load clock');
  assert.equal(loads.channels.length, 5, label + ' must identify five fingers');
  assert.equal(new Set(loads.channels.map(channel => channel.name)).size, 5);
  assert.equal(loads.unit, 'N');
  for (const [rowsKey, totalKey] of [['normal_force_n', 'normal_total_n'], ['tangent_force_n', 'tangent_total_n']]) {
    const rows = loads[rowsKey], totals = loads[totalKey];
    assert.equal(rows.length, loads.time_s.length, label + ' ' + rowsKey + ' clock length');
    assert.equal(totals.length, rows.length, label + ' ' + totalKey + ' clock length');
    rows.forEach((row, index) => {
      assert(Array.isArray(row) && row.length === 5, label + ' requires five scalar loads per sample');
      row.forEach(value => assert(value === null || finite(value) && value >= 0, label + ' invalid load value'));
      if (row.every(finite)) {
        const expected = row.reduce((sum, value) => sum + value, 0);
        assert(finite(totals[index]) && Math.abs(totals[index] - expected) <= 1e-9 * Math.max(1, expected),
          label + ' total must be the five scalar magnitudes, not a different force quantity');
      } else assert.equal(totals[index], null, label + ' missing finger must not become a partial/zero hand total');
    });
  }
}
function safeLocalUrl(value, base, origin) {
  assert(typeof value === 'string' && value, 'Missing published URL');
  const url = new URL(value, base);
  assert.equal(url.origin, origin, 'Published artifact must stay on the local platform');
  assert(['http:', 'https:'].includes(url.protocol));
  assert(!url.username && !url.password);
  return url.href;
}
async function mapLimit(rows, limit, callback) {
  let next = 0;
  await Promise.all(Array.from({length: Math.min(limit, rows.length)}, async () => {
    while (next < rows.length) { const index = next++; await callback(rows[index], index); }
  }));
}
function taskCurveKey(task) {
  return {'bulb-screw': ['clockwise_turns', 'clockwise_angle_deg'],
    'bulb-unscrew': ['turns_removed', 'angle_removed_deg'],
    'install-ram': ['insertion_depth_m'], 'usb-insert': ['insertion_depth_mm', 'insertion_depth_m'],
    'usb-unplug': ['insertion_depth_m'], 'poker-draw': ['card_displacement_now_mm'],
    'whiteboard-wipe': ['cleaned_fraction'], 'whiteboard-wipe-horizontal': ['cleaned_fraction'],
    'whiteboard-wipe-horizontal-line': ['cleaned_fraction']}[task] || [];
}

async function main() {
  const [target, output, ...flags] = process.argv.slice(2);
  assert(target && output, 'usage: node check_rich_evaluation_workbench_browser.cjs URL QA_DIRECTORY [--all-dom] [--baseline=URL]');
  assert(flags.every(flag => flag === '--all-dom' || flag.startsWith('--baseline=')));
  const requested = new URL(target);
  assert(['127.0.0.1', 'localhost', '[::1]'].includes(requested.hostname), 'QA is restricted to the local platform');
  const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');
  const browser = await chromium.launch({headless: true,
    ...((process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE) ?
      {executablePath: process.env.KAI_EVAL_CHROME || process.env.KAI_CHROMIUM_EXECUTABLE} : {}),
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [], failures = [], contracts = new Map(), checkedDom = [], downloadUrls = new Set();
  let auditContextsChecked = 0;
  try {
    const page = await browser.newPage({viewport: {width: 1480, height: 1000}});
    page.setDefaultTimeout(30000);
    page.on('pageerror', error => errors.push({url: page.url(), message: String(error)}));
    assert.equal((await page.goto(target, {waitUntil: 'domcontentloaded'})).status(), 200);
    await page.waitForSelector('.ks-case');
    const overviewUrl = page.url(), origin = new URL(overviewUrl).origin;
    const snapshotPrefix = new URL('./', overviewUrl).pathname;
    const catalogUrl = new URL('catalog.json', overviewUrl).href;
    const catalog = await page.evaluate(() => JSON.parse(document.getElementById('kai-suite-data').textContent));
    assert.equal(catalog.schema, expectedSchema);
    sameJson({members: catalog.counts.members, eligible: catalog.counts.eligible, quarantined: catalog.counts.quarantined},
      {members: 595, eligible: 575, quarantined: 20}, 'Published membership/admissions');
    assert.equal(catalog.groups.length, 47);
    assert.equal(catalog.tasks.length, 10);
    assert.equal(await page.locator('.ks-case').count(), 595);
    assert.equal(await page.locator('#ks-summary-mode').inputValue(), 'policy');
    assert.equal(await page.locator('#ks-comparison tr').count(), 47);
    assert.doesNotMatch(await page.locator('#ks-summary-total').innerText(), /%|16\s*\/\s*575|3\s*\/\s*320/);
    const baselineArg = flags.find(flag => flag.startsWith('--baseline='));
    const baselineUrl = safeLocalUrl(baselineArg ? baselineArg.slice('--baseline='.length) :
      '../kaitactilesim_unified_workbench_20261020_r3/catalog.json', overviewUrl, origin);
    const baselineResponse = await page.request.get(baselineUrl);
    assert.equal(baselineResponse.status(), 200);
    const baseline = await baselineResponse.json(); await baselineResponse.dispose();
    sameJson(scoreProjection(catalog), scoreProjection(baseline), 'Rich presentation must preserve every original result and admission');
    const all = catalog.groups.flatMap(group => group.trials.map(trial => ({group, trial})));
    assert.equal(all.length, 595);
    const taskCounts = new Map(catalog.tasks.map(task => [task.task, all.filter(item => item.group.task === task.task).length]));
    const artifactUrls = new Set();
    const recordUrl = value => artifactUrls.add(value);
    fs.mkdirSync(output, {recursive: true});
    await page.screenshot({path: path.join(output, 'overview.png'), fullPage: false});

    await mapLimit(all, 3, async ({group, trial}, index) => {
      const detailUrl = safeLocalUrl(trial.detail_url, overviewUrl, origin);
      assert(new URL(detailUrl).pathname.startsWith(snapshotPrefix), 'Every retained trial must use the new rich snapshot');
      const response = await page.request.get(detailUrl);
      try {
        assert.equal(response.status(), 200, trial.id + ' detail');
        const {payload, loads, config} = scriptsFromHtml(await response.text());
        const identity = config.unified_identity;
        assert(identity && typeof identity === 'object', trial.id + ' needs a bound unified identity');
        for (const [key, expected] of Object.entries(identityFields(group, trial)))
          sameJson(identity[key], expected, trial.id + ' identity.' + key);
        const stagedStatus = identity.staged_evidence_status ?? identity.staged_status;
        assert.equal(stagedStatus, trial.staged_evidence_status, trial.id + ' must not promote invalid stages');
        assert.equal(payload.report.trial.task, group.task);
        assert.equal(payload.report.trial.seed, trial.seed);
        assert.equal(payload.report.trial.prediction_horizon, group.prediction_horizon);
        assert.equal(loads.task, group.task);
        assert.equal(loads.prediction_horizon, group.prediction_horizon);
        assert.equal(loads.trial_id, payload.report.trial.trial_id);
        assert.equal(config.trial_id, loads.trial_id, trial.id + ' internal legacy/report identity');
        assert.equal(config.prediction_horizon, group.prediction_horizon);
        loadContract(loads, trial.id);
        const viz = payload.visualization;
        assert(viz && Array.isArray(viz.task_series));
        if (group.task === 'pick-place') assert.equal(viz.task_series.length, 0, 'EgoTouch has no recorded continuous object progress');
        else assert(viz.task_series.some(series => taskCurveKey(group.task).includes(series.key)),
          trial.id + ' lost its recorded task-specific time series');
        viz.task_series.forEach(series => {
          clock(series.time_s, trial.id + ' task curve ' + series.key);
          assert.equal(series.values.length, series.time_s.length);
          series.values.forEach(value => assert(value === null || finite(value)));
        });
        const spatial = viz.spatial_frames;
        assert(spatial && spatial.status !== 'missing', trial.id + ' recorded spatial Fn must be available');
        clock(spatial.time_s, trial.id + ' spatial clock');
        assert.equal(spatial.normal_taxel_force_n.length, spatial.time_s.length);
        [spatial.normal_taxel_force_n[0], spatial.normal_taxel_force_n.at(-1)].forEach(frame => {
          assert.equal(frame.length, 5); frame.forEach(finger => {
            assert.equal(finger.length, 7); finger.forEach(row => assert.equal(row.length, 5));
          });
        });
        const plots = config.diagnostic_plots;
        assert.equal(plots?.status, 'available', trial.id + ' native diagnostic plots');
        sameJson(plots.plots.map(plot => plot.id).sort(), ['hand-dof', 'tactile', 'wrist']);
        plots.plots.forEach(plot => recordUrl(safeLocalUrl(plot.url, detailUrl, origin)));
        if (group.model === 'pi05+cpm' || group.model === 'ftp1' ||
            group.task === 'usb-insert' && ['10000', '15000'].includes(String(group.checkpoint))) {
          assert.equal(plots.comparison_kind, 'model_only', trial.id + ' missing reference cannot become an expert comparison');
        }
        const navigation = config.unified_navigation;
        assert(navigation && typeof navigation === 'object');
        assert.equal(safeLocalUrl(navigation.overview_url, detailUrl, origin), overviewUrl);
        assert.equal(safeLocalUrl(navigation.catalog_url, detailUrl, origin), catalogUrl);
        assert(viz.media?.url, trial.id + ' needs the recorded model video');
        recordUrl(safeLocalUrl(viz.media.url, detailUrl, origin));
        if (config.expert_video?.status === 'available') {
          assert.equal(config.expert_video.clock, 'independent_video_seconds');
          assert(/^[0-9a-f]{64}$/.test(config.expert_video.sha256));
          recordUrl(safeLocalUrl(config.expert_video.url, detailUrl, origin));
        }
        if (trial.publication_status === 'quarantined_replay_only' || trial.staged_evidence_status === 'invalid') {
          const auditResponse = await page.request.get(new URL('audit.html', detailUrl).href);
          try {
            assert.equal(auditResponse.status(), 200);
            const audit = await auditResponse.text();
            const context = audit.match(/<([a-z][\w:-]*)\b[^>]*\bclass=["'][^"']*\brich-audit-context\b[^"']*["'][^>]*>([\s\S]*?)<\/\1\s*>/i);
            assert(context, trial.id + ' original audit requires a visible current-admission banner');
            const text = context[2].replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
            if (trial.publication_status === 'quarantined_replay_only')
              assert.match(text, /隔离.*(?:不计入|不计分)/, 'Raw historical audit must not imply renewed validity');
            else assert.match(text, /阶段.*(?:失效|未通过|不计)/, 'Raw CPM stage records remain invalid in audit context');
            auditContextsChecked += 1;
          } finally { await auditResponse.dispose(); }
        }
        contracts.set(trial.id, {detailUrl, taskCurves: viz.task_series.length,
          plotKind: plots.comparison_kind || 'recorded_comparison', expert: config.expert_video?.status === 'available',
          horizon: group.prediction_horizon, nativeSamples: loads.time_s.length});
      } catch (error) { failures.push({trial_id: trial.id, url: detailUrl, error: String(error.stack || error)}); }
      finally { await response.dispose(); }
      if ((index + 1) % 50 === 0) process.stdout.write(JSON.stringify({stage: 'payloads', checked: index + 1, total: all.length, failures: failures.length}) + '\n');
    });
    assert.deepEqual(failures, [], 'Every retained trajectory must satisfy the rich artifact contract');
    await mapLimit([...artifactUrls], 8, async url => {
      const response = await page.request.head(url);
      try { assert.equal(response.status(), 200, 'Missing detail media/plot: ' + url); }
      finally { await response.dispose(); }
    });

    const domSamples = flags.includes('--all-dom') ? all : all.filter(({group, trial}, index) =>
      group.trials[0].id === trial.id || group.model === 'pi05+cpm' || trial.publication_status === 'quarantined_replay_only');
    const screenshotKinds = new Set();
    for (let index = 0; index < domSamples.length; index += 1) {
      const {group, trial} = domSamples[index], detail = contracts.get(trial.id);
      assert.equal((await page.goto(detail.detailUrl, {waitUntil: 'domcontentloaded'})).status(), 200);
      await page.waitForSelector('.wb-root');
      assert.equal(await page.locator('.wb-start-error').count(), 0, trial.id + ' renderer error');
      await page.waitForSelector('.wb-trajectory-nav[data-status="ready"]');
      assert.equal(await page.locator('.ke-transport').count(), 1, 'Exactly one model time cursor');
      assert.equal(await page.locator('.wb-chart').count(), 4, 'Four scalar load panels');
      assert.equal(await page.locator('.wb-finger').count(), 5);
      assert.equal(await page.locator('.ke-task-chart-card').count(), detail.taskCurves);
      assert.equal(await page.locator('.wb-trajectory-select option').count(), taskCounts.get(group.task));
      const back = await page.locator('.wb-overview-link').getAttribute('href');
      assert.equal(safeLocalUrl(back, detail.detailUrl, origin), overviewUrl);
      const horizonLabel = await page.locator('.wb-horizon').innerText();
      assert.match(horizonLabel, new RegExp('^H' + group.prediction_horizon + '(?:\\b|\\s|/)'),
        'H30/H32 must not be relabelled H50');
      assert(horizonLabel.includes('K' + group.execute_steps), 'The real execution length must remain visible');
      const quarantined = trial.publication_status === 'quarantined_replay_only';
      if (quarantined || trial.staged_evidence_status === 'invalid') {
        assert.equal(await page.locator('.wb-stage-badge[data-reached="true"], .wb-stage-card[data-reached="true"]').count(), 0,
          'Invalid or quarantined stage records must not be displayed as achievements');
        assert.equal(await page.locator('.wb-stage-first-time:not(:disabled)').count(), 0);
        assert.match(await page.locator('.wb-root').innerText(), /未知|不计分|隔离/);
      }
      if (quarantined) {
        assert.equal(await page.locator('.wb-verdict.wb-pass').count(), 0);
        assert.match(await page.locator('.wb-root').innerText(), /隔离/);
      } else {
        assert.equal(await page.locator('.wb-verdict.wb-' + (trial.native_success === true ? 'pass' : trial.native_success === false ? 'fail' : 'unknown')).count(), 1);
      }
      if (group.task === 'pick-place') assert.equal(await page.locator('.wb-stage-badge, .wb-stage-card').count(), 0);
      const readout = await page.evaluate(() => {
        const loads = JSON.parse(document.getElementById('kai-tactile-load-data').textContent);
        let index = 0;
        loads.normal_total_n.forEach((value, i) => { if (typeof value === 'number' && value > (loads.normal_total_n[index] || 0)) index = i; });
        if (!index) index = Math.floor(loads.time_s.length / 3);
        return {time: loads.time_s[index], normal: loads.normal_total_n[index], tangent: loads.tangent_total_n[index]};
      });
      const stageSummary = await page.locator('.wb-header-stage-count').allTextContents();
      await page.locator('.ke-slider').evaluate((slider, time) => {
        slider.value = String(time); slider.dispatchEvent(new Event('input', {bubbles: true}));
      }, readout.time);
      await page.waitForFunction(time => Math.abs(Number(document.querySelector('[data-load-chart="normal-total"]').getAttribute('aria-valuenow')) - time) < 1e-7, readout.time);
      for (const [kind, expected] of [['normal-total', readout.normal], ['tangent-total', readout.tangent]]) {
        const reading = await page.locator('[data-load-panel="' + kind + '"] .wb-reading').innerText();
        if (finite(expected)) assert(reading.includes(expected.toFixed(3)), trial.id + ' native ' + kind + ' readout');
        else assert.match(reading, /未知|未记录|不可用/);
      }
      sameJson(await page.locator('.wb-header-stage-count').allTextContents(), stageSummary, 'Cumulative stages must not change with the cursor');
      const totalBefore = await page.locator('[data-load-panel="normal-total"] .wb-reading').innerText();
      await page.locator('.wb-finger').nth(2).click();
      await page.locator('.ke-mode-button').first().click();
      assert.equal(await page.locator('.ke-heat-card[aria-hidden="false"]').count(), 1);
      assert.equal(await page.locator('[data-load-panel="normal-total"] .wb-reading').innerText(), totalBefore);
      assert.equal(await page.locator('.wb-plots-tabs [role="tab"]').count(), 3);
      await page.locator('.wb-plots-tabs [role="tab"]').nth(1).click();
      await page.waitForFunction(() => { const image = document.querySelector('.wb-plots-image'); return image?.complete && image.naturalWidth > 0; });
      if (detail.plotKind === 'model_only') assert.equal(await page.locator('.wb-plots-reference').count(), 0);
      if (detail.expert) {
        assert.equal(await page.locator('.wb-expert-video').count(), 1);
        assert.equal(await page.locator('.wb-expert-video').evaluate(video => video.paused), true);
        const modelTime = await page.locator('.ke-slider').inputValue();
        await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
        await page.locator('.wb-expert-slider').evaluate(slider => {
          slider.value = String(Number(slider.max) / 3); slider.dispatchEvent(new Event('input', {bubbles: true}));
        });
        assert.equal(await page.locator('.ke-slider').inputValue(), modelTime, 'Expert playback clock must remain independent');
      }
      if (['bulb-screw', 'bulb-unscrew'].includes(group.task)) {
        const angle = group.task === 'bulb-screw' ? 'clockwise_angle_deg' : 'angle_removed_deg';
        assert.equal(await page.locator('[data-task-key="' + angle + '"]').count(), 1, 'Bulb time curves must display degrees');
        assert.doesNotMatch(await page.locator('[data-task-key="' + angle + '"] h3').innerText(), /圈数/);
      }
      const anchors = await page.locator('.wb-root a[href]').evaluateAll(nodes => nodes.map(node => ({
        raw: node.getAttribute('href'), absolute: node.href,
      })));
      anchors.filter(link => link.raw && !link.raw.startsWith('#') && !/^(?:blob:|data:)/i.test(link.raw))
        .forEach(link => downloadUrls.add(safeLocalUrl(link.absolute, detail.detailUrl, origin)));
      const stageDownloads = await page.locator('.wb-staged-panel a[href]').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      stageDownloads.forEach(href => assert.equal(new URL(href, detail.detailUrl).pathname,
        new URL('workbench.json', detail.detailUrl).pathname,
        'Unified stage downloads must use the existing configuration, not a missing sidecar'));
      const kind = quarantined ? 'quarantine' : group.model === 'pi05+cpm' ? 'cpm_invalid' : group.model === 'ftp1' ? 'ftp1' :
        group.model === 'egotouch' ? 'egotouch' : group.task;
      if (!screenshotKinds.has(kind)) {
        screenshotKinds.add(kind);
        await page.evaluate(() => scrollTo(0, 0));
        await page.screenshot({path: path.join(output, kind + '.png'), fullPage: false});
        await page.setViewportSize({width: 390, height: 844});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), kind + ' mobile overflow');
        await page.screenshot({path: path.join(output, kind + '_mobile.png'), fullPage: false});
        await page.setViewportSize({width: 1480, height: 1000});
      }
      checkedDom.push({trial_id: trial.id, group_id: group.id, task: group.task, model: group.model,
        checkpoint: group.checkpoint, horizon: group.prediction_horizon, seed: trial.seed});
      if ((index + 1) % 10 === 0) process.stdout.write(JSON.stringify({stage: 'dom', checked: index + 1, total: domSamples.length}) + '\n');
    }
    assert.equal(new Set(checkedDom.map(row => row.group_id)).size, 47);
    await mapLimit([...downloadUrls], 8, async url => {
      const response = await page.request.head(url);
      try { assert.equal(response.status(), 200, 'Broken detail download/navigation link: ' + url); }
      finally { await response.dispose(); }
    });
    const poker = all.find(({group}) => group.task === 'poker-draw' && group.model === 'pi05');
    await page.goto(contracts.get(poker.trial.id).detailUrl, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('.wb-trajectory-nav[data-status="ready"]');
    const pokerOptions = await page.locator('.wb-trajectory-select optgroup').evaluateAll(nodes => nodes.map(node => node.label));
    assert(pokerOptions.some(label => /ftp/i.test(label)) && pokerOptions.some(label => /cpm/i.test(label)),
      'Same-task navigation must include all three Poker model families');
    for (const model of ['ftp1', 'pi05+cpm']) {
      const destination = all.find(({group}) => group.task === 'poker-draw' && group.model === model);
      const expectedUrl = contracts.get(destination.trial.id).detailUrl;
      const option = await page.locator('.wb-trajectory-select option').evaluateAll((nodes, expected) => {
        const config = JSON.parse(document.getElementById('kai-workbench-config').textContent);
        const catalog = new URL(config.unified_navigation.catalog_url, document.baseURI);
        return nodes.find(node => new URL(node.value, catalog).href === expected)?.value;
      }, expectedUrl);
      assert(option, 'Cross-model navigation needs a direct option for ' + model);
      await Promise.all([page.waitForURL(expectedUrl), page.selectOption('.wb-trajectory-select', option)]);
      await page.waitForSelector('.wb-trajectory-nav[data-status="ready"]');
      const selected = await page.evaluate(() => JSON.parse(document.getElementById('kai-workbench-config').textContent).unified_identity);
      assert.equal(selected.model, model);
      assert.equal(await page.locator('.ke-slider').inputValue(), '0', 'Another trajectory must start on its own time axis');
    }
    await Promise.all([page.waitForURL(overviewUrl), page.locator('.wb-overview-link').click()]);
    await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('.ks-case').count(), 595);
    assert.deepEqual(errors, [], 'Rich detail pages must not produce browser errors');
    const result = {status: 'passed', members: 595, eligible: 575, quarantined: 20,
      groups: 47, tasks: 10, original_scores_and_admissions_preserved: true,
      rich_payload_contracts_checked: contracts.size, linked_media_and_plots_checked: artifactUrls.size,
      detail_download_and_navigation_urls_checked: downloadUrls.size,
      dom_trials_checked: checkedDom.length, dom_groups_checked: 47,
      all_dom: flags.includes('--all-dom'), default_policy_rows: 47,
      h30_h32_h50_preserved: true, fn_ft_scalar_totals_and_native_clock: 'passed',
      spatial_taxel_layout: 'five fingers × 7 × 5', native_readouts_and_shared_cursor: 'passed',
      task_time_series_and_bulb_degree_display: 'passed', native_three_plot_gallery: 'passed',
      no_fabricated_expert_comparison: true, cpm_invalid_and_quarantine_not_promoted: true,
      invalid_stage_and_quarantine_audit_contexts_checked: auditContextsChecked,
      egotouch_no_fabricated_object_progress_or_stages: true,
      same_task_cross_model_navigation: 'passed', overview_return: 'passed',
      mobile_overflow: false, simulation_runs_started: 0, model_weights_read: 0,
      baseline_url: baselineUrl, overview_url: overviewUrl, browser_errors: errors, checked_dom: checkedDom};
    fs.writeFileSync(path.join(output, 'browser_verification.json'), JSON.stringify(result, null, 2) + '\n');
    process.stdout.write(JSON.stringify({...result, checked_dom: undefined}) + '\n');
  } finally { await browser.close(); }
}

module.exports = {identityFields, scoreProjection, scriptsFromHtml, clock, loadContract, safeLocalUrl};
if (require.main === module)
  main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
