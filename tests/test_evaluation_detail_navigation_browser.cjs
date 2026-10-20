'use strict';
// Optional, local-only browser QA using the already installed Playwright/browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

async function embedded(page, id) {
  return JSON.parse(await page.locator('#' + id).textContent());
}

async function responseText(page, url) {
  const response = await page.request.get(url);
  assert(response.ok(), `Expected accessible local snapshot: ${url}`);
  return response.text();
}

async function noOverflow(page, message) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), message);
}

async function seekModel(page, value) {
  await page.locator('.ke-slider').evaluate((slider, time) => {
    slider.value = String(time); slider.dispatchEvent(new Event('input', {bubbles: true}));
  }, value);
  await page.waitForFunction(time => Math.abs(Number(document.querySelector('.ke-slider').value) - time) < .001, value);
}

async function assertDetail(page, expected, index, portal) {
  await page.waitForSelector('.wb-header-stages .wb-stage-badge');
  await page.waitForFunction(() => {
    const select = document.querySelector('.wb-trajectory-select');
    return select && !select.disabled && select.options.length > 1;
  });
  assert.equal(await page.locator('.wb-start-error').count(), 0);
  const config = await embedded(page, 'kai-workbench-config');
  const payload = await embedded(page, 'kai-evaluation-data');
  assert.equal(config.task, expected.task);
  assert.equal(config.trial_id, expected.trial_id);
  assert.equal(config.seed, expected.seed);
  assert.equal(String(config.checkpoint), String(expected.checkpoint));
  assert.equal(payload.report.trial.trial_id, expected.trial_id);
  const sourceStages = config.staged_evaluation.stages;
  assert.deepEqual(sourceStages, expected.staged.stages,
    `${expected.task}: stage names, flags and first times come from archived data`);
  const badges = await page.locator('.wb-header-stages .wb-stage-badge').evaluateAll(nodes => nodes.map(node => ({
    key: node.dataset.stage, reached: node.dataset.reached, text: node.textContent,
  })));
  assert.equal(badges.length, sourceStages.length);
  badges.forEach((badge, i) => {
    assert.equal(badge.key, sourceStages[i].key);
    assert.equal(badge.reached, sourceStages[i].reached === true ? 'true' : sourceStages[i].reached === false ? 'false' : 'unknown');
    assert(badge.text.includes(sourceStages[i].label), 'Header reuses the task-specific archived stage label');
  });
  const native = config.staged_evaluation.native_result.task_success;
  assert.equal(native, expected.staged.native_result.task_success);
  assert.match(await page.locator('.wb-verdict').textContent(),
    native === true ? /原任务结果.*成功/ : native === false ? /原任务结果.*失败/ : /原任务结果.*未知/);
  const group = index.trials.filter(trial => trial.task === expected.task);
  const position = group.findIndex(trial => trial.trial_id === expected.trial_id);
  assert(position >= 0);
  const options = await page.locator('.wb-trajectory-select option').evaluateAll(nodes =>
    nodes.map(node => ({value: node.value, text: node.textContent, selected: node.selected})));
  assert.equal(options.length, group.length, 'The selector includes only the same task, across checkpoints');
  assert.deepEqual(options.map(option => option.value), group.map(trial => trial.report_path));
  assert.equal(options.filter(option => option.selected).length, 1);
  assert.equal(await page.locator('.wb-trajectory-select').inputValue(), expected.report_path);
  assert.equal(await page.locator('.wb-trajectory-prev').getAttribute('aria-disabled'), String(position === 0));
  assert.equal(await page.locator('.wb-trajectory-next').getAttribute('aria-disabled'), String(position === group.length - 1));
  if (position === 0) assert.equal(await page.locator('.wb-trajectory-prev').getAttribute('href'), null);
  if (position === group.length - 1) assert.equal(await page.locator('.wb-trajectory-next').getAttribute('href'), null);
  assert.match(await page.locator('.wb-trajectory-count').textContent(), new RegExp(`${position + 1}\\s*/\\s*${group.length}`));
  assert.equal(new URL(await page.getByRole('link', {name: '返回任务列表', exact: true}).getAttribute('href'), page.url()).href,
    new URL(portal).href);
  return {config, payload, badges, position, group};
}

(async () => {
  const [portal, previous, output] = process.argv.slice(2);
  if (!portal || !previous || !output) throw new Error('Expected new portal URL, preserved parent portal URL, and QA output directory');
  assert.notEqual(new URL(portal).href, new URL(previous).href);
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({executablePath: process.env.KAI_CHROMIUM_EXECUTABLE,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const errors = [];
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(portal);
    await page.waitForSelector('.ks-case');
    const batch = await embedded(page, 'kai-suite-data');
    assert.equal(batch.trial_count, 360);
    assert.equal(batch.task_count, 9);
    assert.equal(await page.locator('.ks-case').count(), 360);
    assert.equal(await page.locator('#ks-mode, .ks-mode, .ks-audit, #ks-coverage').count(), 0,
      'The previously cleaned overview stays cleaned');
    const index = JSON.parse(await responseText(page, new URL('trajectory_index.json', portal).href));
    assert.equal(index.schema_version, 'evaluation-task-navigation-v1');
    assert.equal(index.prediction_horizon, 50);
    assert.equal(index.trials.length, 360);
    assert(index.trials.every(row => row.prediction_horizon === 50
      && /^[A-Za-z0-9_-]+\/report\.html$/.test(row.report_path)));
    assert.equal(new Set(index.trials.map(row => row.report_path)).size, 360);
    assert.equal(new Set(index.trials.map(row => row.trial_id)).size, 360);
    const parentBatch = JSON.parse(await responseText(page, new URL('batch.json', previous).href));
    assert.deepEqual(batch.trials, parentBatch.trials, 'Presentation changes never rescore source trials');
    assert.deepEqual(batch.stage_groups, parentBatch.stage_groups);
    await page.screenshot({path: path.join(output, 'portal-desktop.png')});

    const tasks = [...new Set(batch.trials.map(trial => trial.task))];
    for (const task of tasks) {
      const candidates = batch.trials.filter(trial => trial.task === task);
      const trial = candidates.find(row => row.staged.stages.some(stage => stage.reached === true)) || candidates[0];
      const trialURL = new URL(trial.report_path, portal).href;
      assert.equal(await responseText(page, trialURL), await responseText(page, new URL(trial.report_path, previous).href),
        `${task}: report HTML, embedded native data and media references are byte-identical to the parent`);
      await page.goto(trialURL);
      const detail = await assertDetail(page, trial, index, portal);
      assert.match(await page.locator('.wb-header-stages').textContent(), /累计|曾经|全程/,
        'A header summary must not imply the displayed badges are current-cursor conditions');
      if (task.startsWith('whiteboard-wipe')) {
        assert.equal(detail.config.staged_evaluation.staged_result.complete_success, null);
        assert.equal(detail.badges.length, 1, 'The archived whiteboard protocol defines pickup only');
        assert.match(await page.locator('.wb-header-stages').textContent(), /拾取.*(?:不|未).*擦拭|完整.*(?:未定义|不代表)|(?:不|未).*完整/,
          'Reaching the only pickup stage must not be presented as complete wiping success');
        for (const key of ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio']) {
          assert.equal(await page.locator(`[data-task-key="${key}"]`).count(), 1, 'Existing contact-area charts remain');
        }
      }
      assert.equal(await page.locator('.ke-goal-reference').count(),
        detail.payload.visualization.task_series.reduce((sum, item) => sum + (item.goal_references?.length || 0), 0));
      assert.equal(await page.locator('.wb-chart').count(), 4, 'Normal/tangent total and five-finger charts remain');
      assert.equal(await page.locator('.wb-finger').count(), 5);
      const sourcePayload = await page.locator('#kai-evaluation-data').textContent();
      const sourceConfig = await page.locator('#kai-workbench-config').textContent();
      const beforeStages = await page.locator('.wb-header-stages').innerHTML();
      const maximum = Number(await page.locator('.ke-slider').getAttribute('max'));
      await seekModel(page, 0);
      await seekModel(page, maximum * .65);
      assert.equal(await page.locator('.wb-header-stages').innerHTML(), beforeStages,
        'Cumulative header stages do not turn off or advance when the model cursor moves');
      assert.equal(await page.locator('#kai-evaluation-data').textContent(), sourcePayload);
      assert.equal(await page.locator('#kai-workbench-config').textContent(), sourceConfig);
      await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
      const modelTime = await page.locator('.ke-slider').inputValue();
      await page.locator('.wb-expert-slider').evaluate(slider => {
        slider.value = String(Number(slider.max) / 3); slider.dispatchEvent(new Event('input', {bubbles: true}));
      });
      assert.equal(await page.locator('.ke-slider').inputValue(), modelTime, 'Expert timeline remains independent');
      await noOverflow(page, `${task}: no desktop horizontal overflow`);
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({path: path.join(output, `${task}-desktop.png`)});
      await page.setViewportSize({width: 390, height: 844});
      await page.evaluate(() => scrollTo(0, 0));
      await noOverflow(page, `${task}: no mobile horizontal overflow`);
      await page.screenshot({path: path.join(output, `${task}-mobile.png`)});
      await page.setViewportSize({width: 1440, height: 1000});
    }

    const nativeSuccesses = batch.trials.filter(trial => trial.staged.native_result.task_success === true);
    assert(nativeSuccesses.length > 0, 'Include genuine archived native-success cases');
    for (const trial of nativeSuccesses) {
      await page.goto(new URL(trial.report_path, portal).href);
      await assertDetail(page, trial, index, portal);
      assert.equal(trial.staged.complete_success, false, 'This batch has native success without staged terminal success');
      assert(await page.locator('.wb-header-stages .wb-stage-badge[data-reached="false"]').count() > 0);
      assert.match(await page.locator('.wb-verdict').textContent(), /成功/);
      await page.screenshot({path: path.join(output, `${trial.task}-native-success-partial-stage.png`)});
    }

    const task = 'usb-insert';
    const group = index.trials.filter(trial => trial.task === task);
    const trialFor = row => batch.trials.find(trial => trial.trial_id === row.trial_id);
    await page.goto(new URL(group[0].report_path, portal).href);
    await assertDetail(page, trialFor(group[0]), index, portal);
    for (const target of [group[1], group[2]]) {
      await Promise.all([
        page.waitForURL(new URL(target.report_path, portal).href),
        page.locator('.wb-trajectory-next').click(),
      ]);
      await assertDetail(page, trialFor(target), index, portal);
    }
    await Promise.all([
      page.waitForURL(new URL(group[1].report_path, portal).href),
      page.locator('.wb-trajectory-prev').click(),
    ]);
    await assertDetail(page, trialFor(group[1]), index, portal);
    const last = group[group.length - 1];
    assert.notEqual(last.checkpoint, group[0].checkpoint, 'The selector spans the task checkpoints');
    await Promise.all([
      page.waitForURL(new URL(last.report_path, portal).href),
      page.locator('.wb-trajectory-select').selectOption(last.report_path),
    ]);
    await assertDetail(page, trialFor(last), index, portal);
    assert.equal(await page.locator('.wb-trajectory-next').getAttribute('aria-disabled'), 'true');
    assert.equal(await page.locator('.wb-trajectory-next').getAttribute('href'), null);
    await page.setViewportSize({width: 390, height: 844});
    await noOverflow(page, 'Mobile selector remains within viewport at last trajectory');
    await page.screenshot({path: path.join(output, 'usb-navigation-last-mobile.png')});

    // Failed navigation metadata is isolated from all underlying evaluation UI.
    await page.route('**/trajectory_index.json', route => route.fulfill({status: 404,
      contentType: 'application/json', body: '{"error":"test-only unavailable navigation index"}'}));
    await page.goto(new URL(group[0].report_path, portal).href);
    await page.waitForSelector('.wb-header-stages .wb-stage-badge');
    await page.waitForFunction(() => /不可用|无法|未能|失败/.test(document.querySelector('.wb-trajectory-status')?.textContent || ''));
    assert.equal(await page.locator('.wb-start-error').count(), 0);
    assert.equal(await page.locator('.wb-trajectory-select').isDisabled(), true);
    assert.equal(await page.locator('.wb-trajectory-prev').getAttribute('aria-disabled'), 'true');
    assert.equal(await page.locator('.wb-trajectory-next').getAttribute('aria-disabled'), 'true');
    assert.equal(await page.locator('.wb-trajectory-prev').getAttribute('href'), null);
    assert.equal(await page.locator('.wb-trajectory-next').getAttribute('href'), null);
    assert.equal(await page.locator('.wb-chart').count(), 4);
    await seekModel(page, 1);
    assert.equal(await page.locator('.ke-slider').inputValue(), '1', 'Replay still responds without the navigation index');
    await page.waitForFunction(() =>
      Number(document.querySelector('[data-load-chart="normal-total"]').getAttribute('aria-valuenow')) === 1);
    await noOverflow(page, 'Missing index fallback has no mobile overflow');
    const back = page.getByRole('link', {name: '返回任务列表', exact: true});
    await back.click();
    await page.waitForSelector('.ks-case');
    assert.equal(page.url(), new URL(portal).href, 'The fallback retains a working return-to-list link');
    await page.unroute('**/trajectory_index.json');

    await page.goto(new URL(group[0].report_path, previous).href);
    await page.waitForSelector('.wb-header');
    assert.equal(await page.locator('.wb-header-stages, .wb-trajectory-nav').count(), 0,
      'The preserved parent did not acquire the new shared assets');
    assert.equal(await page.locator('.wb-chart').count(), 4);
    const parentIndex = await page.request.get(new URL('trajectory_index.json', previous).href);
    assert.equal(parentIndex.status(), 404, 'No navigation metadata was written into the preserved parent');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: 360, task_count: tasks.length,
      native_success_cases: nativeSuccesses.length,
      checks: ['all task header badges match archived cumulative stages', 'native results remain independent',
        'whiteboard pickup is not wiping completion', 'scrubbing cannot alter header stage results',
        'same-task previous/next/dropdown and checkpoint changes', 'selected position and disabled boundaries',
        'desktop/mobile', 'target lines and contact-area charts preserved', 'Fn/Ft charts and independent expert clock',
        'missing index safe fallback', 'parent HTML and trial evidence unchanged']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
