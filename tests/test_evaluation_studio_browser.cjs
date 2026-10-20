'use strict';
// Optional local-only visual regression QA. Reuses cached browser and existing HTTP server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

async function embedded(page, id) { return JSON.parse(await page.locator('#' + id).textContent()); }
async function textResponse(page, url) {
  const response = await page.request.get(url);
  assert(response.ok(), `Expected accessible local artifact: ${url}`);
  return response.text();
}
async function noOverflow(page, label) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${label}: no document horizontal overflow`);
}
async function seek(page, selector, time) {
  await page.locator(selector).evaluate((input, value) => {
    input.value = String(value); input.dispatchEvent(new Event('input', {bubbles: true}));
  }, time);
}
async function detailReady(page) {
  await page.waitForSelector('.wb-header-stages .wb-stage-badge');
  await page.waitForFunction(() => document.querySelector('.wb-trajectory-select')?.options.length > 1);
  assert.equal(await page.locator('.wb-start-error').count(), 0);
}
async function assertStages(page, trial) {
  const config = await embedded(page, 'kai-workbench-config');
  assert.equal(config.trial_id, trial.trial_id);
  assert.equal(config.task, trial.task);
  const original = trial.staged.stages;
  assert.deepEqual(config.staged_evaluation.stages, original);
  const badges = await page.locator('.wb-header-stages .wb-stage-badge').evaluateAll(nodes => nodes.map(n => ({
    key: n.dataset.stage, reached: n.dataset.reached, text: n.textContent,
  })));
  assert.equal(badges.length, original.length);
  original.forEach((stage, i) => {
    assert.equal(badges[i].key, stage.key);
    assert.equal(badges[i].reached, stage.reached === true ? 'true' : stage.reached === false ? 'false' : 'unknown');
    assert(badges[i].text.includes(stage.label));
  });
  const native = trial.staged.native_result.task_success;
  assert.equal(config.staged_evaluation.native_result.task_success, native);
  assert.match(await page.locator('.wb-verdict').textContent(), native === true ? /原任务结果.*成功/ : native === false ? /原任务结果.*失败/ : /原任务结果.*未知/);
  const before = await page.locator('.wb-header-stages').innerHTML();
  await seek(page, '.ke-slider', Number(await page.locator('.ke-slider').getAttribute('max')) * .55);
  assert.equal(await page.locator('.wb-header-stages').innerHTML(), before, 'Cumulative stages do not change with model clock');
  await seek(page, '.ke-slider', 0);
  return config;
}

(async () => {
  const [portal, parent, output] = process.argv.slice(2);
  if (!portal || !parent || !output) throw new Error('Expected new studio URL, parent URL, and screenshot output directory');
  assert.notEqual(new URL(portal).href, new URL(parent).href);
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
    assert.equal(await page.locator('#ks-mode, .ks-mode, .ks-audit, #ks-coverage').count(), 0);
    const priorBatch = JSON.parse(await textResponse(page, new URL('batch.json', parent).href));
    assert.deepEqual(batch.trials, priorBatch.trials, 'No rescore or source data changes');
    assert.deepEqual(batch.stage_groups, priorBatch.stage_groups);
    const index = JSON.parse(await textResponse(page, new URL('trajectory_index.json', portal).href));
    assert.equal(index.trials.length, 360);
    assert(index.trials.every(row => row.prediction_horizon === 50));
    await noOverflow(page, 'Overview cards desktop');
    await page.screenshot({path: path.join(output, 'overview-cards-desktop.png')});
    await page.locator('#ks-view-list').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'list');
    assert.equal(await page.locator('.ks-case').count(), 360);
    assert.equal(await page.locator('#ks-view-list').getAttribute('aria-pressed'), 'true');
    await noOverflow(page, 'Overview list desktop');
    await page.screenshot({path: path.join(output, 'overview-list-desktop.png')});
    await page.reload(); await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'list', 'View preference survives reload');
    await page.locator('#ks-task-tabs button[data-task="usb-insert"]').click();
    assert.equal(await page.locator('.ks-case').count(), 40);
    assert.match(await page.locator('#ks-visible').textContent(), /40\s*\/\s*360/);
    await page.locator('#ks-checkpoint').selectOption('24999');
    assert.equal(await page.locator('.ks-case').count(), 20);
    await page.locator('#ks-search').fill('seed_019');
    assert.equal(await page.locator('.ks-case').count(), 1);
    assert.match(await page.locator('.ks-identity').textContent(), /24999.*019/);
    await page.setViewportSize({width: 390, height: 844});
    await noOverflow(page, 'Overview filtered list mobile');
    await page.screenshot({path: path.join(output, 'overview-filtered-list-mobile.png')});
    await page.locator('#ks-search').fill('');
    await page.locator('#ks-checkpoint').selectOption('');
    await page.locator('#ks-task-tabs button[data-task=""]').click();
    await page.locator('#ks-view-cards').click();
    assert.equal(await page.locator('#ks-trials').getAttribute('data-view'), 'cards');
    await noOverflow(page, 'Overview cards mobile');
    await page.screenshot({path: path.join(output, 'overview-cards-mobile.png')});
    await page.setViewportSize({width: 1440, height: 1000});

    const tasks = [...new Set(batch.trials.map(trial => trial.task))];
    for (const task of tasks) {
      const trial = batch.trials.find(row => row.task === task && row.staged.stages.some(stage => stage.reached === true)) || batch.trials.find(row => row.task === task);
      await page.goto(new URL(trial.report_path, portal).href);
      await detailReady(page);
      const config = await assertStages(page, trial);
      const payload = await embedded(page, 'kai-evaluation-data');
      assert.equal(await textResponse(page, new URL(trial.report_path, portal).href), await textResponse(page, new URL(trial.report_path, parent).href), `${task}: archived HTML unchanged`);
      assert.equal(await page.locator('.wb-expert-panel').evaluate(node => node.open), false, 'Expert replay starts collapsed');
      assert.equal(await page.locator('.wb-chart').count(), 4);
      assert.equal(await page.locator('.wb-finger').count(), 5);
      assert.equal(await page.locator('.ke-goal-reference').count(), payload.visualization.task_series.reduce((sum, item) => sum + (item.goal_references?.length || 0), 0));
      if (task.startsWith('whiteboard-wipe')) {
        for (const key of ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio']) assert.equal(await page.locator(`[data-task-key="${key}"]`).count(), 1);
        assert.match(await page.locator('.wb-header-stages').textContent(), /拾取.*(?:不|未).*擦拭|完整.*(?:未定义|不代表)|(?:不|未).*完整/);
      }
      const wrist = page.locator('.wb-cameras .wb-camera[data-camera="wrist"]');
      assert.equal(await wrist.count(), 1, `${task}: wrist camera retained`);
      const time = await page.locator('.ke-slider').inputValue();
      await wrist.click();
      assert.equal(await page.locator('.ke-video').getAttribute('data-camera'), 'wrist');
      assert.equal(await page.locator('.ke-slider').inputValue(), time);
      await page.locator('.wb-cameras .wb-camera[data-camera="mosaic"]').click();
      assert.equal(await page.locator('.ke-video').getAttribute('data-camera'), 'mosaic');
      await page.locator('.wb-camera').first().click();
      await page.evaluate(() => scrollTo(0, 0));
      const model = await page.locator('.ke-media-panel').boundingBox();
      const primary = await page.locator('.wb-task-grid').first().boundingBox();
      assert(model.y < 1000 && primary.y < 1000, `${task}: model replay and task charts both begin in desktop first screen`);
      assert(model.x + model.width <= primary.x + 2, `${task}: model replay is beside task metrics on desktop`);
      assert(await page.locator('.wb-header h1').evaluate(node => parseFloat(getComputedStyle(node).fontSize) >= 20));
      await noOverflow(page, `${task} desktop`);
      await page.screenshot({path: path.join(output, `${task}-desktop.png`)});
      await page.setViewportSize({width: 390, height: 844});
      await page.evaluate(() => scrollTo(0, 0));
      await noOverflow(page, `${task} mobile`);
      await page.screenshot({path: path.join(output, `${task}-mobile.png`)});
      await page.setViewportSize({width: 1440, height: 1000});

      if (task === 'usb-insert') await page.locator('.wb-section-nav a[href="#wb-expert-reference"]').click();
      else await page.locator('.wb-expert-panel > summary').click();
      assert.equal(await page.locator('.wb-expert-panel').evaluate(node => node.open), true);
      await page.waitForFunction(() => !document.querySelector('.wb-expert-slider').disabled);
      const modelClock = await page.locator('.ke-slider').inputValue();
      const expertTime = Number(await page.locator('.wb-expert-slider').getAttribute('max')) / 3;
      await seek(page, '.wb-expert-slider', expertTime);
      await page.waitForFunction(time => {
        const video = document.querySelector('.wb-expert-video');
        return video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime - time) < .05;
      }, expertTime);
      assert.equal(await page.locator('.ke-slider').inputValue(), modelClock, 'Expert clock remains independent');
      if (task === 'usb-insert') {
        await page.locator('.wb-expert-play').click();
        await page.waitForFunction(time => {
          const video = document.querySelector('.wb-expert-video');
          return !video.paused && video.readyState >= 2 && video.currentTime > time + .25;
        }, expertTime);
        await page.locator('.wb-expert-play').click();
        assert.equal(await page.locator('.wb-expert-video').evaluate(video => video.paused), true);
        assert.equal(await page.locator('.ke-slider').inputValue(), modelClock, 'Playing expert video does not move model clock');
        await page.locator('.wb-expert-panel').screenshot({path: path.join(output, 'expert-expanded-independent.png')});
      }
      await page.locator('.wb-expert-panel > summary').click();
      await page.locator('.wb-finger[data-finger="3"]').click();
      assert.equal(await page.locator('.wb-root').getAttribute('data-selected-finger'), '3');
      await page.getByRole('button', {name: '接触力圆图', exact: true}).click();
      assert.equal(await page.locator('.ke-glyph-grid').isVisible(), true);
      assert.equal(await page.locator('.ke-glyph-card:not(.wb-inactive)').count(), 1);
      if (task === 'usb-insert') await page.locator('.ke-spatial-panel').screenshot({path: path.join(output, 'tactile-glyph.png')});
      await page.getByRole('button', {name: '阵列热图', exact: true}).click();
      assert.equal(await page.locator('.ke-heat-card:not(.wb-inactive)').count(), 1);
      assert.equal((config.diagnostic_plots?.plots || []).length, 3);
      assert.equal(await page.locator('.wb-plots-tabs [role="tab"]').count(), 3);
      // Some presentation versions put original whole-run plots in an optional disclosure.
      await page.locator('.wb-plots-panel').evaluate(node => {
        for (let ancestor = node.parentElement; ancestor; ancestor = ancestor.parentElement) if (ancestor.tagName === 'DETAILS') ancestor.open = true;
      });
      const tabs = page.locator('.wb-plots-tabs [role="tab"]');
      for (let n = 0; n < 3; n++) {
        await tabs.nth(n).click();
        assert.equal(await tabs.nth(n).getAttribute('aria-selected'), 'true');
        await page.waitForFunction(() => { const img = document.querySelector('.wb-plots-image'); return img.complete && img.naturalWidth > 0; });
      }
    }
    for (const trial of batch.trials.filter(row => row.staged.native_result.task_success === true)) {
      await page.goto(new URL(trial.report_path, portal).href);
      await detailReady(page); await assertStages(page, trial);
      assert.match(await page.locator('.wb-verdict').textContent(), /成功/);
      assert(await page.locator('.wb-header-stages .wb-stage-badge[data-reached="false"]').count() > 0);
    }
    const group = index.trials.filter(row => row.task === 'usb-insert');
    const first = group[0], next = group[1], last = group[group.length - 1];
    await page.goto(new URL(first.report_path, portal).href); await detailReady(page);
    assert.equal(await page.locator('.wb-trajectory-prev').getAttribute('aria-disabled'), 'true');
    await Promise.all([page.waitForURL(new URL(next.report_path, portal).href), page.locator('.wb-trajectory-next').click()]);
    await detailReady(page);
    assert.equal((await embedded(page, 'kai-workbench-config')).trial_id, next.trial_id);
    await Promise.all([page.waitForURL(new URL(first.report_path, portal).href), page.locator('.wb-trajectory-prev').click()]);
    await detailReady(page);
    await Promise.all([page.waitForURL(new URL(last.report_path, portal).href), page.locator('.wb-trajectory-select').selectOption(last.report_path)]);
    await detailReady(page);
    assert.equal((await embedded(page, 'kai-workbench-config')).trial_id, last.trial_id);
    assert.equal(await page.locator('.wb-trajectory-next').getAttribute('aria-disabled'), 'true');
    assert.equal(await page.locator('.ke-slider').inputValue(), '0');
    const storagePage = await browser.newPage({viewport: {width: 390, height: 844}});
    await storagePage.addInitScript(() => {
      Object.defineProperty(window, 'localStorage', {configurable: true,
        get() { throw new DOMException('Storage disabled for local-only QA', 'SecurityError'); }});
    });
    storagePage.on('pageerror', error => errors.push(error.message));
    await storagePage.goto(portal); await storagePage.waitForSelector('.ks-case');
    assert.equal(await storagePage.locator('.ks-case').count(), 360);
    await storagePage.locator('#ks-view-list').click();
    assert.equal(await storagePage.locator('#ks-trials').getAttribute('data-view'), 'list');
    await noOverflow(storagePage, 'Storage-unavailable mobile fallback');
    await storagePage.close();
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, tasks: 9, trials: 360,
      checks: ['cards/list/filter/search', 'task-specific cumulative stage and native independence', 'model/metrics desktop first screen',
        'collapsed independent expert replay', 'model wrist/mosaic cameras', 'Fn/Ft and array/glyph finger selection',
        'all goal references and whiteboard area', 'three original diagnostic images', 'same-task navigation', 'desktop/mobile no overflow', 'source trials and detail HTML unchanged']}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
