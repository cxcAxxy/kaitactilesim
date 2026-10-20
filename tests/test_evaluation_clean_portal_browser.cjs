'use strict';
// Optional local-only QA. Reuses an already installed Playwright and browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.KAI_PLAYWRIGHT_MODULE || 'playwright');

function stageCategory(trial) {
  if (trial.staged.complete_success === true) return 'complete';
  return trial.staged.stages.some(stage => stage.reached === true) ? 'partial' : 'none';
}

async function embedded(page, id) {
  return JSON.parse(await page.locator('#' + id).textContent());
}

(async () => {
  const [portal, previous, output] = process.argv.slice(2);
  if (!portal || !previous || !output) {
    throw new Error('Expected new portal URL, preserved parent portal URL, and QA output directory');
  }
  assert.notEqual(new URL(portal).href, new URL(previous).href, 'New UI must not replace the parent portal');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({
    executablePath: process.env.KAI_CHROMIUM_EXECUTABLE,
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
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
    assert.equal(await page.locator('.ks-stat').count(), 2, 'Only task and H50 trajectory totals remain');
    assert.equal(await page.locator('#ks-task-tabs button').count(), 10);
    assert.match(await page.locator('#ks-counts').textContent(), /9.*类任务.*360.*H50/s);
    assert.equal(await page.locator('#ks-mode').count(), 0);
    assert.equal(await page.locator('.ks-mode').count(), 0, 'No per-card run-mode badges');
    assert.equal(await page.locator('.ks-audit, #ks-coverage').count(), 0);
    const visible = await page.locator('body').innerText();
    assert.doesNotMatch(visible, /运行模式|正式标志|数据覆盖与来源|快照时间：|下载轨迹清单|查看 H50 筛选记录|查看导出验证/);
    assert.equal(await page.locator('a[href="batch.json"], a[href="selection.json"], a[href="verification.json"]').count(), 0);

    // Hide internal controls, never remove or reinterpret their underlying evidence.
    const batchResponse = await page.request.get(new URL('batch.json', portal).href);
    assert(batchResponse.ok());
    const storedBatch = await batchResponse.json();
    assert.deepEqual(storedBatch.mode_counts, batch.mode_counts);
    assert.deepEqual(storedBatch.trials, batch.trials);
    assert.equal(storedBatch.mode_counts.diagnostic, 360);
    assert(batch.trials.every(trial => trial.mode === 'diagnostic'));
    assert(batch.trials.every(trial => trial.staged?.source_cohort && trial.staged?.rule_identity
      && trial.staged?.data_evidence && trial.staged?.source_selection));
    assert(batch.source_preservation, 'Source preservation/provenance remains in the machine-readable batch');
    const preservedBatchResponse = await page.request.get(new URL('batch.json', previous).href);
    assert(preservedBatchResponse.ok());
    const preservedBatch = await preservedBatchResponse.json();
    assert.deepEqual(batch.trials, preservedBatch.trials, 'All trial identity, mode, outcome, and provenance remain unchanged');
    assert.deepEqual(batch.stage_groups, preservedBatch.stage_groups);
    assert.deepEqual(batch.goal_reference_extension, preservedBatch.goal_reference_extension);

    await page.screenshot({path: path.join(output, 'portal-desktop.png')});
    await page.locator('#ks-task-tabs button[data-task="usb-insert"]').click();
    assert.equal(await page.locator('.ks-case').count(), 40);
    const checkpoint = String(batch.trials.find(trial => trial.task === 'usb-insert').checkpoint);
    await page.locator('#ks-checkpoint').selectOption(checkpoint);
    assert.equal(await page.locator('.ks-case').count(), 20);
    await page.locator('#ks-search').fill('seed_005');
    assert.equal(await page.locator('.ks-case').count(), 1);
    assert.match(await page.locator('.ks-case .ks-identity').textContent(), /seed 005/);
    await page.locator('#ks-search').fill('');
    await page.locator('#ks-checkpoint').selectOption('');
    await page.locator('#ks-task-tabs button[data-task=""]').click();
    for (const stage of ['complete', 'partial', 'none']) {
      await page.locator('#ks-stage').selectOption(stage);
      assert.equal(await page.locator('.ks-case').count(), batch.trials.filter(trial => stageCategory(trial) === stage).length,
        `${stage}: stage filtering remains functional`);
    }
    await page.locator('#ks-stage').selectOption('');
    await page.locator('#ks-search').fill('seed_005');
    assert.equal(await page.locator('.ks-case').count(), 18);
    await page.locator('#ks-search').fill('');
    assert.equal(await page.locator('.ks-case').count(), 360);

    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => window.scrollTo(0, 0));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No mobile horizontal overflow');
    await page.screenshot({path: path.join(output, 'portal-mobile.png')});
    await page.setViewportSize({width: 1440, height: 1000});

    for (const task of ['usb-insert', 'usb-unplug', 'whiteboard-wipe']) {
      const trial = batch.trials.find(row => row.task === task && row.seed === 7) || batch.trials.find(row => row.task === task);
      const trialURL = new URL(trial.report_path, portal).href;
      // The presentation cleanup must not rewrite even the underlying report HTML.
      const currentResponse = await page.request.get(trialURL);
      const priorResponse = await page.request.get(new URL(trial.report_path, previous).href);
      assert(currentResponse.ok() && priorResponse.ok());
      assert.equal(await currentResponse.text(), await priorResponse.text(), `${task}: report HTML remains byte-identical`);
      await page.goto(trialURL);
      await page.waitForSelector('.wb-header');
      assert.equal(await page.locator('.wb-start-error').count(), 0);
      const payload = await embedded(page, 'kai-evaluation-data');
      const series = payload.visualization.task_series;
      const references = series.filter(item => item.goal_references?.length);
      assert.equal(references.length, task === 'usb-unplug' ? 1 : 2);
      assert.equal(await page.locator('.ke-goal-reference').count(), task === 'usb-unplug' ? 1 : 2);
      if (task === 'usb-insert') {
        assert.equal(references.find(item => item.key === 'insertion_depth_mm').goal_references[0].target, 12);
        assert.equal(references.find(item => item.key === 'orientation_error_deg').goal_references[0].target, 3);
      } else if (task === 'usb-unplug') {
        assert.equal(references[0].goal_references[0].target, -.002);
      } else {
        assert.equal(references.find(item => item.key === 'maximum_ink_remaining').goal_references[0].target, 1e-9);
        for (const key of ['effective_pad_contact_area_ratio', 'contact_time_weighted_mean_effective_area_ratio']) {
          const area = series.find(item => item.key === key);
          assert(area, 'Previously added whiteboard contact-area series remains');
          assert.equal(area.presentation_extension, 'whiteboard-contact-area-v1');
          assert.equal(await page.locator(`[data-task-key="${key}"]`).count(), 1);
        }
      }
      const back = page.getByRole('link', {name: '返回任务列表', exact: true});
      assert.equal(new URL(await back.getAttribute('href'), trialURL).href, new URL(portal).href);
      await back.click();
      await page.waitForSelector('.ks-case');
      assert.equal(page.url(), new URL(portal).href, 'Returning from a copied report stays inside the new UI snapshot');
      assert.equal(await page.locator('#ks-mode, .ks-audit').count(), 0);
    }

    await page.goto(previous);
    await page.waitForSelector('.ks-case');
    assert.equal(await page.locator('#ks-mode').count(), 1, 'Preserved parent still has its original run-mode filter');
    assert.equal(await page.locator('.ks-mode').count(), 360);
    assert.equal(await page.locator('.ks-stat').count(), 4);
    assert.equal(await page.locator('.ks-audit').count(), 1);
    await page.locator('.ks-audit summary').click();
    assert.match(await page.locator('.ks-audit').innerText(), /数据覆盖与来源.*快照时间：/s);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, trial_count: 360, task_count: 9,
      checks: ['no internal mode controls/badges', 'no public coverage/source download section',
        'two summary stats', 'task/checkpoint/seed/stage filters', 'desktop/mobile',
        'machine-readable mode and provenance retained', 'USB/whiteboard report HTML untouched',
        'reference lines and whiteboard area retained', 'return navigation remains inside new snapshot', 'preserved parent unchanged']}));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
